# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""High-level AI functions that wrap descriptors into map_batches calls.

These functions create stateful wrapper classes that:
1. Accept a Descriptor (serializable, lightweight)
2. Lazily call ``instantiate()`` on the worker to load the model once
3. Process each batch through the loaded model

Usage::

    import vane
    from vane.ai.functions import embed_text, classify_text

    conn = vane.connect()
    rel = conn.sql("SELECT text FROM documents")

    # Text embedding — returns relation with 'embedding' column
    embedded = embed_text(
        rel,
        "text",
        provider="transformers",
        model="sentence-transformers/all-MiniLM-L6-v2",
    )

    # Text classification — returns relation with 'label' column
    classified = classify_text(
        rel,
        "text",
        labels=["positive", "negative"],
        provider="transformers",
    )
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import pyarrow as pa

import duckdb
from vane._expression_udf import _build_actor_map_batches_expression
from vane._expressions import as_expression, is_expression
from vane.ai.options import (
    AnthropicPromptOptions,
    AnthropicProviderOptions,
    GoogleEmbeddingOptions,
    GooglePromptOptions,
    GoogleProviderOptions,
    OpenAIEmbeddingOptions,
    OpenAIPromptOptions,
    OpenAIProviderOptions,
    VLLMPromptOptions,
    VLLMProviderOptions,
)
from vane.ai.provider import load_provider
from vane.ai.providers.vllm import VLLMPrompterDescriptor
from vane.ai.typing import UDFOptions

if TYPE_CHECKING:
    from vane import Expression, Relation
    from vane.ai.provider import Provider


def _resolve_provider(provider: str | Provider | None, default: str = "transformers") -> Provider:
    """Resolve a provider argument to a Provider instance."""
    if provider is None:
        return load_provider(default)
    if isinstance(provider, str):
        return load_provider(provider)
    return provider


def _run_async(coro: Any) -> Any:
    """Run an awaitable, handling the case where a loop is already running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Retry / on_error helpers
# ---------------------------------------------------------------------------

_OnError = Literal["raise", "log", "ignore"]


class RetryAfterError(Exception):
    """Retryable error carrying the requested wait time (in seconds).

    Providers raise this when they receive a rate-limit (429) or
    service-unavailable (503) response with a ``Retry-After`` header.
    The retry helpers honour :attr:`retry_after` for the sleep duration.
    """

    def __init__(self, retry_after: float, original: Exception | None = None) -> None:
        super().__init__(str(original) if original else "RetryAfterError")
        self.retry_after = retry_after
        self.__cause__ = original


def _retry_call(
    fn: Any,
    *args: Any,
    max_retries: int = 3,
    on_error: _OnError = "raise",
    default: Any = None,
    **kwargs: Any,
) -> Any:
    """Call *fn* with exponential-backoff retry and on_error handling.

    Args:
        fn: Callable (sync or async) to invoke.
        max_retries: Number of retry attempts after the first failure (0 = no retries).
        on_error: ``"raise"`` re-raises on final failure; ``"log"`` and
            ``"ignore"`` return *default*.
        default: Value to return when on_error is not ``"raise"``.
    """
    last_exc: Exception | None = None
    for attempt in range(1 + max(0, max_retries)):
        try:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                result = _run_async(result)
            return result
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                if isinstance(exc, RetryAfterError):
                    wait = min(exc.retry_after, 120)
                else:
                    wait = min(2**attempt, 30)  # 1, 2, 4, 8, ... capped at 30s
                time.sleep(wait)

    # All retries exhausted
    assert last_exc is not None
    if on_error == "raise":
        # Unwrap RetryAfterError to expose the original exception
        if isinstance(last_exc, RetryAfterError) and last_exc.__cause__:
            raise last_exc.__cause__
        raise last_exc
    return default


async def _retry_call_async(
    fn: Any,
    *args: Any,
    max_retries: int = 3,
    on_error: _OnError = "raise",
    default: Any = None,
    **kwargs: Any,
) -> Any:
    """Async variant of :func:`_retry_call`."""
    last_exc: Exception | None = None
    for attempt in range(1 + max(0, max_retries)):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                if isinstance(exc, RetryAfterError):
                    wait = min(exc.retry_after, 120)
                else:
                    wait = min(2**attempt, 30)
                await asyncio.sleep(wait)

    assert last_exc is not None
    if on_error == "raise":
        if isinstance(last_exc, RetryAfterError) and last_exc.__cause__:
            raise last_exc.__cause__
        raise last_exc
    return default


def _map_batches_kwargs(
    udf_opts: UDFOptions,
    execution_backend: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build keyword arguments for ``rel.map_batches()``."""
    num_gpus = udf_opts.num_gpus
    if udf_opts.actor_number is not None and num_gpus is None:
        raise ValueError("UDFOptions.num_gpus is required when actor_number is set")

    kwargs: dict[str, Any] = {
        "batch_size": udf_opts.batch_size,
        "gpus": num_gpus,
    }
    if execution_backend is not None:
        backend = str(execution_backend).strip().lower()
        if backend not in ("subprocess_task", "subprocess_actor", "ray_task", "ray_actor"):
            raise ValueError("execution_backend must be one of: subprocess_task, subprocess_actor, ray_task, ray_actor")
        kwargs["execution_backend"] = backend
        if udf_opts.actor_number is not None:
            if backend not in ("subprocess_actor", "ray_actor"):
                raise ValueError(
                    "UDFOptions.actor_number is only supported for execution_backend='subprocess_actor' or 'ray_actor'"
                )
            kwargs["actor_number"] = udf_opts.actor_number
    elif udf_opts.actor_number is not None:
        kwargs["actor_number"] = udf_opts.actor_number
    if extra:
        kwargs.update(extra)
    return kwargs


def _merge_options(*objects: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for obj in objects:
        if obj is None:
            continue
        if hasattr(obj, "to_descriptor_options"):
            merged.update(obj.to_descriptor_options())
        elif isinstance(obj, dict):
            merged.update(obj)
        else:
            raise TypeError(f"Unsupported AI options object: {type(obj).__name__}")
    return merged


def _adapt_batch_wrapper_for_backend(wrapper: Any, execution_backend: str | None, *, force_actor: bool = False) -> Any:
    backend = str(execution_backend or "").strip().lower()
    if backend in ("subprocess_actor", "ray_actor") or (backend == "" and force_actor):

        class _ConfiguredAIBatchActor:
            def __init__(self) -> None:
                self._wrapper = wrapper

            def __call__(self, table: pa.Table) -> pa.Table:
                return self._wrapper(table)

        return _ConfiguredAIBatchActor

    if backend in ("", "subprocess_task", "ray_task"):

        def _run_ai_batch(table: pa.Table) -> pa.Table:
            return wrapper(table)

        return _run_ai_batch

    return wrapper


# ---------------------------------------------------------------------------
# Text chunking utilities
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    max_chars: int = 2000,
    overlap_chars: int = 200,
) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: The input text to chunk.
        max_chars: Maximum characters per chunk.
        overlap_chars: Number of overlapping characters between chunks.

    Returns:
        List of text chunks. Returns ``[text]`` if text fits in one chunk.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    step = max(1, max_chars - overlap_chars)
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def _weighted_average_embeddings(
    embeddings: list[Any],
    weights: list[float],
) -> Any:
    """Compute length-weighted average of embeddings."""
    arr = np.array(embeddings, dtype=np.float64)
    w = np.array(weights, dtype=np.float64)
    w /= w.sum()
    averaged = (arr * w[:, np.newaxis]).sum(axis=0)
    norm = np.linalg.norm(averaged)
    if norm > 0:
        averaged /= norm
    return averaged.astype(np.float32)


def _schema_for_embedding(dimensions: int | None) -> dict[str, str]:
    if dimensions is None:
        return {"embedding": "FLOAT[]"}
    return {"embedding": f"FLOAT[{dimensions}]"}


def _actor_number_or_one(udf_opts: UDFOptions) -> int:
    return udf_opts.actor_number or 1


def _gpus_or_zero(udf_opts: UDFOptions) -> float:
    if udf_opts.num_gpus is None:
        return 0
    return float(udf_opts.num_gpus)


def _resolve_ai_batch_size(udf_opts: UDFOptions, default: int = 32) -> int:
    if udf_opts.batch_size and udf_opts.batch_size > 0:
        return udf_opts.batch_size
    return default


def _normalize_embeddings(values: list[Any]) -> list[Any]:
    normalized: list[Any] = []
    for value in values:
        if value is None:
            normalized.append(None)
            continue
        arr = np.asarray(value, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        normalized.append(arr)
    return normalized


def _as_positive_int(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)) and value > 0:
        return int(value)
    return None


def _embedding_zero_size(descriptor: Any, arrow_type: Any | None) -> int:
    if arrow_type is not None and pa.types.is_fixed_size_list(arrow_type):
        return arrow_type.list_size

    dimensions = descriptor.get_dimensions()
    for value in (getattr(dimensions, "size", None), getattr(dimensions, "list_size", None)):
        size = _as_positive_int(value)
        if size is not None:
            return size

    as_arrow_type = getattr(dimensions, "as_arrow_type", None)
    if callable(as_arrow_type):
        dimension_type = as_arrow_type()
        if pa.types.is_fixed_size_list(dimension_type):
            return dimension_type.list_size
        size = _as_positive_int(getattr(dimension_type, "list_size", None))
        if size is not None:
            return size

    raise ValueError("Could not determine embedding dimension for zero fallback")


# ---------------------------------------------------------------------------
# Module-level wrapper classes (must be at module level for pickle)
# ---------------------------------------------------------------------------


class _EmbedTextBatch:
    """Stateful wrapper — model loaded once per actor via instantiate()."""

    def __init__(
        self,
        descriptor: Any,
        column: str,
        output_column: str,
        max_chunk_chars: int | None = None,
        chunk_overlap_chars: int = 200,
        max_retries: int = 3,
        on_error: _OnError = "raise",
        normalize: bool = False,
        arrow_type: Any | None = None,
    ) -> None:
        self._descriptor = descriptor
        self._column = column
        self._output_column = output_column
        self._max_chunk_chars = max_chunk_chars
        self._chunk_overlap_chars = chunk_overlap_chars
        self._max_retries = max_retries
        self._on_error: _OnError = on_error
        self._normalize = normalize
        # Keep expression construction lazy: unknown OpenAI-compatible models can
        # probe dimensions over the network, so callers pass a schema-aligned
        # Arrow type when they already know the dimensions.
        self._arrow_type = arrow_type
        self._embedder = None  # lazy: instantiate on first __call__

    def _ensure_embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = self._descriptor.instantiate()
        return self._embedder

    def _zero_fill(self, count: int) -> list[Any]:
        """Zero embeddings when possible; nulls when the dimension is unknowable."""
        try:
            zero_size = _embedding_zero_size(self._descriptor, self._arrow_type)
        except Exception:
            return [None] * count
        return [np.zeros(zero_size, dtype=np.float32) for _ in range(count)]

    def __call__(self, table: pa.Table) -> pa.Table:
        texts = table.column(self._column).to_pylist()
        texts = [t if t is not None else "" for t in texts]

        if self._max_chunk_chars is not None:
            result = self._embed_with_chunking(texts)
        else:
            result = _retry_call(
                self._ensure_embedder().embed_text,
                texts,
                max_retries=self._max_retries,
                on_error=self._on_error,
            )
            if result is None:
                result = self._zero_fill(len(texts))

        if self._normalize:
            result = _normalize_embeddings(result)

        arrow_type = self._arrow_type or pa.list_(pa.float32())
        embeddings = pa.array(
            [None if r is None else (r.tolist() if hasattr(r, "tolist") else list(r)) for r in result],
            type=arrow_type,
        )
        return pa.table({self._output_column: embeddings})

    def _embed_with_chunking(self, texts: list[str]) -> list[Any]:
        """Embed texts with automatic chunking for long inputs."""
        # Build chunk plan: (original_idx, chunk_text, chunk_weight)
        all_chunks: list[str] = []
        chunk_map: list[list[tuple[int, float]]] = []  # per-original-text

        for text in texts:
            chunks = chunk_text(
                text,
                max_chars=self._max_chunk_chars,  # type: ignore[arg-type]
                overlap_chars=self._chunk_overlap_chars,
            )
            entry: list[tuple[int, float]] = []
            for c in chunks:
                entry.append((len(all_chunks), float(len(c))))
                all_chunks.append(c)
            chunk_map.append(entry)

        # Embed all chunks in one batch
        chunk_embeddings = self._ensure_embedder().embed_text(all_chunks)
        if inspect.isawaitable(chunk_embeddings):
            chunk_embeddings = _run_async(chunk_embeddings)

        # Reassemble: weighted average for multi-chunk texts
        results: list[Any] = []
        for entry in chunk_map:
            if len(entry) == 1:
                results.append(chunk_embeddings[entry[0][0]])
            else:
                embs = [chunk_embeddings[idx] for idx, _ in entry]
                weights = [w for _, w in entry]
                results.append(_weighted_average_embeddings(embs, weights))
        return results


class _ClassifyTextBatch:
    """Stateful wrapper for text classification."""

    def __init__(
        self,
        descriptor: Any,
        column: str,
        output_column: str,
        labels: list[str],
        max_retries: int = 3,
        on_error: _OnError = "raise",
    ) -> None:
        self._descriptor = descriptor
        self._column = column
        self._output_column = output_column
        self._labels = labels
        self._max_retries = max_retries
        self._on_error: _OnError = on_error
        self._classifier = None  # lazy: instantiate on first __call__

    def __call__(self, table: pa.Table) -> pa.Table:
        if self._classifier is None:
            self._classifier = self._descriptor.instantiate()
        texts = table.column(self._column).to_pylist()
        texts = [t if t is not None else "" for t in texts]
        results = _retry_call(
            self._classifier.classify_text,
            texts,
            self._labels,
            max_retries=self._max_retries,
            on_error=self._on_error,
        )
        if results is None:
            results = [None] * len(texts)

        return pa.table({self._output_column: results})


class _PromptBatch:
    """Stateful wrapper for LLM prompting.

    Supports both plain text and structured output (Pydantic models).
    When ``return_format`` is set, responses are serialized to JSON strings.
    When ``image_columns`` is set, image data from those columns is packed
    alongside text into multimodal message tuples.
    """

    def __init__(
        self,
        descriptor: Any,
        column: str,
        output_column: str,
        max_api_concurrency: int | None = None,
        return_format: Any | None = None,
        image_columns: list[str] | None = None,
        max_retries: int = 3,
        on_error: _OnError = "raise",
    ) -> None:
        self._descriptor = descriptor
        self._column = column
        self._output_column = output_column
        self._max_api_concurrency = max_api_concurrency
        self._return_format = return_format
        self._image_columns = image_columns or []
        self._max_retries = max_retries
        self._on_error: _OnError = on_error
        self._prompter = None  # lazy: instantiate on first __call__

    def _serialize_result(self, result: Any) -> str | None:
        """Convert a prompt result to a string for the output column."""
        if result is None:
            return None
        if isinstance(result, str):
            return result
        # Structured output — Pydantic model or dict
        if hasattr(result, "model_dump_json"):
            return result.model_dump_json()
        if hasattr(result, "json"):
            return result.json()
        import json

        return json.dumps(result, default=str)

    def __call__(self, table: pa.Table) -> pa.Table:
        if self._prompter is None:
            self._prompter = self._descriptor.instantiate()
        texts = table.column(self._column).to_pylist()
        texts = [t if t is not None else "" for t in texts]

        # Build per-row message tuples (text + optional image columns)
        image_lists: list[list[Any]] = [table.column(col_name).to_pylist() for col_name in self._image_columns]

        def build_messages(idx: int) -> tuple[Any, ...]:
            parts: list[Any] = [texts[idx]]
            for img_col in image_lists:
                val = img_col[idx]
                if val is not None:
                    parts.append(val)
            return tuple(parts)

        # Use batch API if available (e.g. vLLM's continuous batching)
        if hasattr(self._prompter, "prompt_batch"):
            results = _retry_call(
                self._prompter.prompt_batch,
                texts,
                max_retries=self._max_retries,
                on_error=self._on_error,
            )
            if results is None:
                results = [None] * len(texts)
            if self._return_format is not None:
                results = [self._serialize_result(r) for r in results]
            return pa.table({self._output_column: results})

        has_images = bool(self._image_columns)
        max_retries = self._max_retries
        on_error = self._on_error

        async def run_all() -> list[str | None]:
            if self._max_api_concurrency is not None and self._max_api_concurrency > 0:
                sem = asyncio.Semaphore(self._max_api_concurrency)

                async def limited(idx: int) -> str | None:
                    async with sem:
                        msgs = build_messages(idx) if has_images else (texts[idx],)
                        result = await _retry_call_async(
                            self._prompter.prompt,
                            msgs,
                            max_retries=max_retries,
                            on_error=on_error,
                        )
                        return self._serialize_result(result) if self._return_format else result

                return await asyncio.gather(*(limited(i) for i in range(len(texts))))

            async def single(idx: int) -> str | None:
                msgs = build_messages(idx) if has_images else (texts[idx],)
                result = await _retry_call_async(
                    self._prompter.prompt,
                    msgs,
                    max_retries=max_retries,
                    on_error=on_error,
                )
                return self._serialize_result(result) if self._return_format else result

            return await asyncio.gather(*(single(i) for i in range(len(texts))))

        results = _run_async(run_all())
        return pa.table({self._output_column: results})


def _build_ai_batch_expression(
    wrapper: Any,
    *,
    input_name: str,
    input_expr: Any,
    output_column: str,
    output_type: str,
    udf_opts: UDFOptions,
    name: str,
) -> Any:
    actor_callable = _adapt_batch_wrapper_for_backend(wrapper, "subprocess_actor", force_actor=True)
    return _build_actor_map_batches_expression(
        actor_callable,
        name=name,
        inputs={input_name: as_expression(input_expr)},
        schema={output_column: output_type},
        batch_size=_resolve_ai_batch_size(udf_opts),
        row_preserving=True,
        actor_number=_actor_number_or_one(udf_opts),
        gpus=_gpus_or_zero(udf_opts),
    )


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------


def embed_text(
    rel: Any,
    column: str,
    *,
    provider: str | Provider | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    output_column: str = "embedding",
    max_chunk_chars: int | None = None,
    chunk_overlap_chars: int = 200,
    execution_backend: str | None = None,
    **options: Any,
) -> Any:
    """Embed a text column using the specified provider.

    Args:
        rel: A DuckDB relation containing the source data.
        column: Name of the text column to embed.
        provider: Provider name or instance (default: ``"transformers"``).
        model: Model identifier (provider-specific default if ``None``).
        dimensions: Output embedding dimensions (model default if ``None``).
        output_column: Name of the output column (default: ``"embedding"``).
        max_chunk_chars: If set, texts longer than this are split into
            overlapping chunks, embedded separately, and combined via
            length-weighted average. ``None`` disables chunking.
        chunk_overlap_chars: Characters of overlap between adjacent chunks
            (default: 200). Only used when ``max_chunk_chars`` is set.
        execution_backend: Optional UDF backend. If omitted, the relation API infers task backend
            from the active runner.
        **options: Forwarded to the provider's ``get_text_embedder``.

    Returns:
        A new relation with the ``output_column`` appended.
    """
    prov = _resolve_provider(provider, "transformers")
    descriptor = prov.get_text_embedder(model=model, dimensions=dimensions, **options)
    udf_opts = descriptor.get_udf_options()

    wrapper = _EmbedTextBatch(
        descriptor,
        column,
        output_column,
        max_chunk_chars=max_chunk_chars,
        chunk_overlap_chars=chunk_overlap_chars,
        max_retries=udf_opts.max_retries,
        on_error=udf_opts.on_error,
    )
    kwargs = _map_batches_kwargs(udf_opts, execution_backend)
    kwargs["schema"] = {output_column: "FLOAT[]"}
    udf = _adapt_batch_wrapper_for_backend(
        wrapper,
        kwargs.get("execution_backend"),
        force_actor="actor_number" in kwargs,
    )
    return rel.map_batches(udf, **kwargs)


def embed(
    text: Any,
    *,
    provider: str | Provider = "openai",
    model: str | None = None,
    provider_options: OpenAIProviderOptions | GoogleProviderOptions | dict[str, Any] | None = None,
    embedding_options: OpenAIEmbeddingOptions | GoogleEmbeddingOptions | dict[str, Any] | None = None,
    dimensions: int | None = None,
    normalize: bool | None = None,
) -> Any:
    """Build an expression that embeds a text expression through an AI provider.

    If provider options do not set ``concurrency``, the expression uses one
    actor. Provider ``concurrency`` maps internally to UDF ``actor_number``.
    Prefer provider environment variables such as ``OPENAI_API_KEY`` or
    ``GOOGLE_API_KEY`` over passing API keys in code or SQL text.
    """
    prov = _resolve_provider(provider, "openai")
    descriptor_options = _merge_options(provider_options, embedding_options)
    try:
        descriptor = prov.get_text_embedder(model=model, dimensions=dimensions, **descriptor_options)
    except NotImplementedError as exc:
        raise ValueError(f"Provider {provider!r} is not an embedding provider") from exc
    udf_opts = descriptor.get_udf_options()

    output_column = "embedding"
    output_type = _schema_for_embedding(dimensions)[output_column]
    wrapper = _EmbedTextBatch(
        descriptor,
        "text",
        output_column,
        max_retries=udf_opts.max_retries,
        on_error=udf_opts.on_error,
        normalize=bool(normalize),
        arrow_type=pa.list_(pa.float32(), dimensions) if dimensions is not None else pa.list_(pa.float32()),
    )
    return _build_ai_batch_expression(
        wrapper,
        input_name="text",
        input_expr=text,
        output_column=output_column,
        output_type=output_type,
        udf_opts=udf_opts,
        name="ai_embed",
    )


# ---------------------------------------------------------------------------
# classify_text
# ---------------------------------------------------------------------------


def classify_text(
    rel: Any,
    column: str,
    *,
    labels: list[str],
    provider: str | Provider | None = None,
    model: str | None = None,
    output_column: str = "label",
    execution_backend: str | None = None,
    **options: Any,
) -> Any:
    """Classify a text column using zero-shot classification.

    Args:
        rel: A DuckDB relation containing the source data.
        column: Name of the text column to classify.
        labels: List of candidate labels.
        provider: Provider name or instance (default: ``"transformers"``).
        model: Model identifier (provider-specific default if ``None``).
        output_column: Name of the output column (default: ``"label"``).
        execution_backend: Optional UDF backend. If omitted, the relation API infers task backend
            from the active runner.
        **options: Forwarded to the provider's ``get_text_classifier``.

    Returns:
        A new relation with the ``output_column`` appended.
    """
    prov = _resolve_provider(provider, "transformers")
    descriptor = prov.get_text_classifier(model=model, **options)
    udf_opts = descriptor.get_udf_options()

    wrapper = _ClassifyTextBatch(
        descriptor,
        column,
        output_column,
        labels,
        max_retries=udf_opts.max_retries,
        on_error=udf_opts.on_error,
    )
    kwargs = _map_batches_kwargs(udf_opts, execution_backend)
    kwargs["schema"] = {output_column: "VARCHAR"}
    udf = _adapt_batch_wrapper_for_backend(
        wrapper,
        kwargs.get("execution_backend"),
        force_actor="actor_number" in kwargs,
    )
    return rel.map_batches(udf, **kwargs)


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


def _build_native_vllm_expression(messages: Any, descriptor: VLLMPrompterDescriptor) -> duckdb.Expression:
    """Build the native row-preserving ``vllm()`` expression."""
    messages_expr = as_expression(messages)
    # DuckDB's concat() ignores NULL arguments: concat(NULL, '') returns '',
    # and concat('system\n\n', NULL, '') returns 'system\n\n'.
    prompt_arguments = [messages_expr, duckdb.ConstantExpression("")]
    if descriptor.system_message:
        prompt_arguments.insert(
            0,
            duckdb.ConstantExpression(f"{descriptor.system_message}\n\n"),
        )
    prompt_expr = duckdb.FunctionExpression("concat", *prompt_arguments)

    native_options = descriptor.build_physical_vllm_options()
    try:
        options_json = json.dumps(
            native_options,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("vLLM native options must be JSON-serializable") from exc

    return duckdb.FunctionExpression(
        "vllm",
        prompt_expr,
        duckdb.ConstantExpression(descriptor.model_name),
        duckdb.ConstantExpression(options_json),
    )


def _prompt_relation(
    rel: Any,
    column: str,
    *,
    image_columns: list[str] | None = None,
    provider: str | Provider | None = "openai",
    model: str | None = None,
    provider_options: (
        OpenAIProviderOptions
        | VLLMProviderOptions
        | AnthropicProviderOptions
        | GoogleProviderOptions
        | dict[str, Any]
        | None
    ) = None,
    prompt_options: (
        OpenAIPromptOptions | VLLMPromptOptions | AnthropicPromptOptions | GooglePromptOptions | dict[str, Any] | None
    ) = None,
    system_message: str | None = None,
    return_format: Any | None = None,
    use_chat_completions: bool = True,
    output_column: str = "response",
    execution_backend: str | None = None,
    **options: Any,
) -> Any:
    """Generate responses for a relation column via ``rel.map_batches()``.

    ``execution_backend`` is optional; when omitted, the relation API infers
    the task backend from the active runner. Prefer provider environment
    variables such as ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, or
    ``GOOGLE_API_KEY`` over passing API keys in call options.
    """
    prov = _resolve_provider(provider, "openai")
    prompter_kwargs: dict[str, Any] = {
        "model": model,
        "system_message": system_message,
    }
    if return_format is not None:
        prompter_kwargs["return_format"] = return_format
    if not use_chat_completions:
        prompter_kwargs["use_chat_completions"] = False
    prompter_kwargs.update(_merge_options(provider_options, prompt_options, options))

    try:
        descriptor = prov.get_prompter(**prompter_kwargs)
    except NotImplementedError as exc:
        raise ValueError(f"Provider {provider!r} is not a prompt provider") from exc

    if isinstance(descriptor, VLLMPrompterDescriptor):
        if image_columns:
            raise ValueError("native vLLM prompting does not support image_columns")
        relation_alias = getattr(rel, "alias", None)
        column_expr = (
            duckdb.ColumnExpression(relation_alias, column)
            if isinstance(relation_alias, str) and relation_alias
            else duckdb.ColumnExpression(column)
        )
        expression = _build_native_vllm_expression(column_expr, descriptor).alias(output_column)
        input_columns = [
            (
                duckdb.ColumnExpression(relation_alias, name)
                if isinstance(relation_alias, str) and relation_alias
                else duckdb.ColumnExpression(name)
            )
            for name in rel.columns
        ]
        return rel.select(*input_columns, expression)

    udf_opts = descriptor.get_udf_options()

    wrapper = _PromptBatch(
        descriptor,
        column,
        output_column,
        udf_opts.max_api_concurrency,
        return_format=return_format,
        image_columns=image_columns,
        max_retries=udf_opts.max_retries,
        on_error=udf_opts.on_error,
    )
    udf_opts_copy = UDFOptions(
        actor_number=udf_opts.actor_number,
        num_gpus=udf_opts.num_gpus,
        max_retries=udf_opts.max_retries,
        on_error=udf_opts.on_error,
        batch_size=udf_opts.batch_size or 1,
        max_api_concurrency=udf_opts.max_api_concurrency,
    )
    kwargs = _map_batches_kwargs(udf_opts_copy, execution_backend)
    kwargs["schema"] = {output_column: "VARCHAR"}
    udf = _adapt_batch_wrapper_for_backend(
        wrapper,
        kwargs.get("execution_backend"),
        force_actor="actor_number" in kwargs,
    )
    return rel.map_batches(udf, **kwargs)


def _prompt_expression(
    messages: Any,
    *,
    provider: str | Provider = "openai",
    model: str | None = None,
    provider_options: (
        OpenAIProviderOptions
        | VLLMProviderOptions
        | AnthropicProviderOptions
        | GoogleProviderOptions
        | dict[str, Any]
        | None
    ) = None,
    prompt_options: (
        OpenAIPromptOptions | VLLMPromptOptions | AnthropicPromptOptions | GooglePromptOptions | dict[str, Any] | None
    ) = None,
    system_message: str | None = None,
) -> Any:
    """Build a row-preserving expression prompt.

    Supported expression kwargs are ``provider``, ``model``,
    ``provider_options``, ``prompt_options``, and ``system_message``. Prefer
    provider environment variables such as ``OPENAI_API_KEY``,
    ``ANTHROPIC_API_KEY``, or ``GOOGLE_API_KEY`` over passing API keys in
    prompt options.
    """
    prov = _resolve_provider(provider, "openai")
    descriptor_options = _merge_options(provider_options, prompt_options)
    try:
        descriptor = prov.get_prompter(model=model, system_message=system_message, **descriptor_options)
    except NotImplementedError as exc:
        raise ValueError(f"Provider {provider!r} is not a prompt provider") from exc

    if isinstance(descriptor, VLLMPrompterDescriptor):
        return _build_native_vllm_expression(messages, descriptor)

    udf_opts = descriptor.get_udf_options()

    wrapper = _PromptBatch(
        descriptor,
        "messages",
        "response",
        udf_opts.max_api_concurrency,
        max_retries=udf_opts.max_retries,
        on_error=udf_opts.on_error,
    )
    return _build_ai_batch_expression(
        wrapper,
        input_name="messages",
        input_expr=messages,
        output_column="response",
        output_type="VARCHAR",
        udf_opts=udf_opts,
        name="ai_prompt",
    )


def _is_relation_like(value: Any) -> bool:
    return hasattr(value, "map_batches") and hasattr(value, "select")


_PROMPT_RELATION_ONLY_KWARGS = (
    "output_column",
    "return_format",
    "image_columns",
    "use_chat_completions",
    "execution_backend",
)

_PROMPT_ARGUMENT_UNSET = object()


def _reject_relation_only_prompt_kwargs(kwargs: dict[str, Any]) -> None:
    unsupported = [name for name in _PROMPT_RELATION_ONLY_KWARGS if name in kwargs]
    if unsupported:
        raise TypeError(
            "vane.ai.prompt expression API does not support: "
            + ", ".join(unsupported)
            + ". Rename the output with .alias(...); use the relation API "
            "prompt(rel, column, ...) for return_format/image_columns/execution_backend."
        )


@overload
def prompt(first: Relation, column: str, **kwargs: Any) -> Relation: ...


@overload
def prompt(first: Expression, column: None = None, **kwargs: Any) -> Expression: ...


@overload
def prompt(*, rel: Relation, column: str, **kwargs: Any) -> Relation: ...


def prompt(
    first: Any = _PROMPT_ARGUMENT_UNSET,
    column: str | None = None,
    *,
    rel: Any = _PROMPT_ARGUMENT_UNSET,
    **kwargs: Any,
) -> Any:
    """Generate LLM responses from either a relation column or expression.

    ``prompt(rel, "column", ...)`` and ``prompt(rel=rel, column="column",
    ...)`` preserve the relation API.
    The relation API accepts ``execution_backend``; when omitted, it infers the
    task backend from the active runner.
    ``prompt(vane.col("column"), ...)`` returns a row-preserving expression.
    The expression API supports ``provider``, ``model``,
    ``provider_options``, ``prompt_options``, and ``system_message``. When
    provider options do not set ``concurrency``, the expression API uses one
    actor. Prefer provider environment variables such as ``OPENAI_API_KEY``,
    ``ANTHROPIC_API_KEY``, or ``GOOGLE_API_KEY`` over passing API keys in call
    options or SQL text.
    """
    if first is not _PROMPT_ARGUMENT_UNSET and rel is not _PROMPT_ARGUMENT_UNSET:
        raise TypeError("vane.ai.prompt received both first and rel; pass only one relation argument")
    if first is _PROMPT_ARGUMENT_UNSET and rel is _PROMPT_ARGUMENT_UNSET:
        raise TypeError("vane.ai.prompt requires a messages expression or a relation via rel=")

    target = rel if rel is not _PROMPT_ARGUMENT_UNSET else first
    if is_expression(target) or (column is None and not _is_relation_like(target)):
        if column is not None:
            raise TypeError("vane.ai.prompt expression API accepts a single messages expression")
        _reject_relation_only_prompt_kwargs(kwargs)
        return _prompt_expression(target, **kwargs)
    if column is None:
        raise TypeError("vane.ai.prompt relation API requires a column name")
    return _prompt_relation(target, column, **kwargs)
