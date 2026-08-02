# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""High-level AI functions that wrap descriptors into map_batches calls.

These functions create stateful wrapper classes that:
1. Accept a Descriptor (serializable, lightweight)
2. Lazily call ``instantiate()`` on the worker to load the model once
3. Process each batch through the loaded model

Usage::

    import vane
    from vane.ai.functions import classify_text, embed

    conn = vane.connect()
    rel = conn.sql("SELECT text FROM documents")

    # Text embedding — returns relation with 'embedding' column
    embedded = embed(
        rel,
        vane.col("text"),
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
from typing import Any, Literal, overload

import numpy as np
import pyarrow as pa
from typing_extensions import Unpack

import duckdb
from vane._expression_udf import _build_actor_map_batches_expression, _build_map_batches_expression
from vane._expressions import as_expression, is_expression
from vane._typing import Expression, Relation
from vane.ai.options import (
    AnthropicPromptOptions,
    AnthropicProviderOptions,
    EmbedOptions,
    GooglePromptOptions,
    GoogleProviderOptions,
    OpenAIPromptOptions,
    OpenAIProviderOptions,
    VLLMPromptOptions,
    VLLMProviderOptions,
    validate_embed_options,
)
from vane.ai.protocols import NativePrompterPlan
from vane.ai.provider import Provider, ProviderCapabilityError, _ProviderResultError, load_provider
from vane.ai.providers.vllm import NativeVLLMPromptPlan, _build_native_vllm_options_argument
from vane.ai.typing import UDFOptions


def _resolve_provider(provider: str | Provider | None, default: str = "transformers") -> Provider:
    """Resolve a provider argument to a Provider instance."""
    if provider is None:
        return load_provider(default)
    if isinstance(provider, str):
        return load_provider(provider)
    return provider


class _MissingAsyncRuntimeError(RuntimeError):
    """A wrapper needed its executor-bound async runtime but none was bound."""


def _missing_async_runtime() -> _MissingAsyncRuntimeError:
    return _MissingAsyncRuntimeError(
        "AI batch wrappers must be driven by a UDF executor that binds an "
        "async runtime via bind_async_runtime(); they do not own event loops"
    )


def _provider_is_loop_bound(provider: Any, method_name: str) -> bool:
    """Whether a cached provider statically shows event-loop-bound state that
    must be released before its loop is torn down.

    Positive signals:

    * the provider exposes ``aclose`` (a client that must be awaited shut) —
      a *sufficient* signal, never a necessary one;
    * the driven method (``embed_text`` / ``prompt``) is a coroutine function
      — the protocol-sanctioned way to own a loop-bound async client
      (``AsyncOpenAI`` / ``AsyncAnthropic`` / ``genai.Client``).

    The strongest signal — an awaitable actually observed from the driven
    method at runtime — is delivered separately: the wrappers pass their
    ``_mark_loop_bound`` callback as ``on_awaitable`` to the retry helpers, so
    even a plain ``def`` that returns an awaitable (which static inspection
    cannot classify) upgrades to loop-bound the moment it runs.

    A missing ``aclose()`` is never taken as proof of synchronicity: the public
    ``TextEmbedder`` / ``Prompter`` protocols permit an async provider that does
    not expose it, and treating such a provider as synchronous strands its
    loop-bound client across event loops (issue #139). Genuinely synchronous
    providers (e.g. the Transformers embedder) match no signal and stay cached
    across per-task executors so their model is not reloaded.
    """
    if getattr(provider, "aclose", None) is not None:
        return True
    method = getattr(provider, method_name, None)
    return method is not None and inspect.iscoroutinefunction(inspect.unwrap(method))


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


_TRANSIENT_EMBED_HTTP_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _is_transient_embed_error(exc: Exception) -> bool:
    """Whether an idempotent Embed request failed for a transient reason."""

    if isinstance(exc, RetryAfterError):
        return True

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True

        for status in (
            getattr(current, "status_code", None),
            getattr(current, "code", None),
            getattr(getattr(current, "response", None), "status_code", None),
        ):
            if isinstance(status, int) and not isinstance(status, bool):
                if status in _TRANSIENT_EMBED_HTTP_STATUS_CODES:
                    return True

        # OpenAI connection errors retain their underlying httpx transport
        # error as the cause.  Avoid importing an optional SDK just to classify
        # it; httpx itself is likewise optional for non-network providers.
        try:
            import httpx
        except ImportError:
            pass
        else:
            if isinstance(current, httpx.TransportError):
                return True

        current = current.__cause__

    return False


def _retry_call(
    fn: Any,
    *args: Any,
    max_retries: int = 3,
    on_error: _OnError = "raise",
    default: Any = None,
    run_async: Any = None,
    on_awaitable: Any = None,
    retry_if: Any = None,
    **kwargs: Any,
) -> Any:
    """Call *fn* with exponential-backoff retry and on_error handling.

    Args:
        fn: Callable (sync or async) to invoke.
        max_retries: Number of retry attempts after the first failure (0 = no retries).
        on_error: ``"raise"`` re-raises on final failure; ``"log"`` and
            ``"ignore"`` return *default*.
        default: Value to return when on_error is not ``"raise"``.
        run_async: Callable used to drive awaitable results. Batch wrappers
            pass their executor-bound runtime so cached SDK clients see one
            loop across batches; required when *fn* returns an awaitable.
        on_awaitable: Optional zero-arg callback invoked the first time *fn*
            returns an awaitable — lets a wrapper learn its provider is
            loop-bound even when static inspection cannot tell (a plain
            ``def`` that returns an awaitable), so ``close()`` releases it.
        retry_if: Optional predicate that must accept an exception and return
            true for retryable failures. By default, ordinary exceptions keep
            the historical retry behavior.
    """
    last_exc: Exception | None = None
    for attempt in range(1 + max(0, max_retries)):
        try:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                if on_awaitable is not None:
                    on_awaitable()
                if run_async is None:
                    result.close()
                    raise _missing_async_runtime()
                result = run_async(result)
            return result
        except _MissingAsyncRuntimeError:
            # Programming error, not a transient API failure: never retried,
            # never converted to a default by on_error.
            raise
        except Exception as exc:
            last_exc = exc
            if isinstance(exc, (ProviderCapabilityError, _ProviderResultError)):
                break
            if retry_if is not None and not retry_if(exc):
                break
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
    on_awaitable: Any = None,
    **kwargs: Any,
) -> Any:
    """Async variant of :func:`_retry_call`.

    ``on_awaitable`` mirrors :func:`_retry_call`: invoked when *fn* returns an
    awaitable, so a wrapper learns its provider is loop-bound even when the
    method is a plain ``def`` returning an awaitable (which
    ``iscoroutinefunction`` cannot classify).
    """
    last_exc: Exception | None = None
    for attempt in range(1 + max(0, max_retries)):
        try:
            result = fn(*args, **kwargs)
            if on_awaitable is not None and inspect.isawaitable(result):
                on_awaitable()
            return await result
        except Exception as exc:
            last_exc = exc
            if isinstance(exc, (ProviderCapabilityError, _ProviderResultError)):
                break
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


def _embedding_provider_family(provider: Any) -> str | None:
    """Return the closed options family implemented by a built-in provider."""

    module = type(provider).__module__
    if module == "vane.ai.providers.openai":
        return "openai"
    if module == "vane.ai.providers.google":
        return "google"
    if module == "vane.ai.providers.transformers":
        return "transformers"
    name = getattr(provider, "name", None)
    if isinstance(name, str) and name.casefold() in {"openai", "google", "transformers"}:
        return name.casefold()
    return None


_BUILTIN_EMBED_PROVIDER_SENSITIVE_FIELDS = {
    "vane.ai.providers.openai": frozenset({"api_key", "organization"}),
    "vane.ai.providers.google": frozenset({"api_key"}),
}


def _reject_builtin_embed_provider_credentials(provider: Provider) -> None:
    """Keep built-in Provider presets containing secrets out of Embed plans."""

    sensitive_fields = _BUILTIN_EMBED_PROVIDER_SENSITIVE_FIELDS.get(type(provider).__module__)
    configured = getattr(provider, "_options", None)
    if sensitive_fields is None or not isinstance(configured, dict):
        return
    offending = sorted(field for field in sensitive_fields if configured.get(field) is not None)
    if not offending:
        return
    label = "field" if len(offending) == 1 else "fields"
    raise ValueError(
        f"Embed provider configuration cannot include inline credential or sensitive {label}: "
        f"{', '.join(offending)}; configure credentials through the environment or runtime secret management"
    )


def _validate_embedding_dimension(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("dimensions must be a positive integer or None")
    return int(value)


def _resolve_embedding_dimension(descriptor: Any, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    try:
        metadata = descriptor.get_dimensions()
        resolved = getattr(metadata, "size", metadata)
    except Exception as exc:
        model = getattr(descriptor, "get_model", lambda: "unknown")()
        raise ValueError(
            f"Cannot determine embedding dimensions for model {model!r} without network or model loading; "
            "pass dimensions=... explicitly"
        ) from exc
    if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved <= 0:
        raise ValueError("Provider embedding dimension metadata must be a positive integer")
    return int(resolved)


def _prepare_embed_call(
    provider: Any,
    model: str | None,
    dimensions: int | None,
    on_error: str,
    options: dict[str, Any],
    *,
    relation: bool,
) -> tuple[Any, int, UDFOptions, bool, int | None, int, str | None, bool]:
    """Resolve one Embed call without performing network or model I/O."""

    if provider is None:
        raise TypeError("provider must be a provider name or Provider object, not None")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("model must be a non-empty string or None")
    explicit_dimensions = _validate_embedding_dimension(dimensions)
    if on_error not in {"raise", "ignore"}:
        raise ValueError("on_error must be 'raise' or 'ignore'")

    resolved_provider = _resolve_provider(provider, "openai")
    if not isinstance(resolved_provider, Provider):
        raise TypeError("provider must be a provider name or Provider object")
    _reject_builtin_embed_provider_credentials(resolved_provider)
    prepared = validate_embed_options(
        _embedding_provider_family(resolved_provider),
        options,
        relation=relation,
    )
    normalize = prepared.pop("normalize", False)
    execution_backend = prepared.pop("execution_backend", None)
    max_chunk_chars = prepared.pop("max_chunk_chars", None)
    chunk_overlap_chars = prepared.pop("chunk_overlap_chars", 200)
    actor_number_explicit = "actor_number" in prepared
    batch_size = prepared.pop("batch_size", None)
    max_retries = prepared.pop("max_retries", None)
    actor_number = prepared.pop("actor_number", None)

    try:
        descriptor = resolved_provider.get_text_embedder(
            model=model,
            dimensions=explicit_dimensions,
            **prepared,
        )
    except NotImplementedError as exc:
        provider_name = getattr(resolved_provider, "name", type(resolved_provider).__name__)
        raise ValueError(f"Provider {provider_name!r} is not an embedding provider") from exc

    resolved_dimensions = _resolve_embedding_dimension(descriptor, explicit_dimensions)
    udf_options = descriptor.get_udf_options()
    udf_options.on_error = on_error
    if batch_size is not None:
        udf_options.batch_size = batch_size
    if max_retries is not None:
        udf_options.max_retries = max_retries
    if actor_number is not None:
        udf_options.actor_number = actor_number

    return (
        descriptor,
        resolved_dimensions,
        udf_options,
        normalize,
        max_chunk_chars,
        chunk_overlap_chars,
        execution_backend,
        actor_number_explicit,
    )


def _adapt_batch_wrapper_for_backend(wrapper: Any, execution_backend: str | None, *, force_actor: bool = False) -> Any:
    backend = str(execution_backend or "").strip().lower()
    if backend in ("subprocess_actor", "ray_actor") or (backend == "" and force_actor):

        class _ConfiguredAIBatchActor:
            def __init__(self) -> None:
                self._wrapper = wrapper

            def bind_async_runtime(self, run_async: Any) -> None:
                bind = getattr(self._wrapper, "bind_async_runtime", None)
                if bind is not None:
                    bind(run_async)

            def close(self) -> None:
                close = getattr(self._wrapper, "close", None)
                if close is not None:
                    close()

            def __call__(self, table: pa.Table) -> pa.Table:
                return self._wrapper(table)

        return _ConfiguredAIBatchActor

    if backend in ("", "subprocess_task", "ray_task"):

        def _run_ai_batch(table: pa.Table) -> pa.Table:
            return wrapper(table)

        # Task executors are per-invocation: the executor binds a runtime,
        # runs the batch, and closes it — client lifecycle matches the task.
        bind = getattr(wrapper, "bind_async_runtime", None)
        if bind is not None:
            _run_ai_batch.bind_async_runtime = bind  # type: ignore[attr-defined]
            _run_ai_batch.close = wrapper.close  # type: ignore[attr-defined]

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


# ---------------------------------------------------------------------------
# Module-level wrapper classes (must be at module level for pickle)
# ---------------------------------------------------------------------------


class _EmbedTextBatch:
    """Stateful wrapper — model loaded once per actor via instantiate().

    Async execution is driven exclusively through an executor-bound
    ``run_async`` capability (see ``UDFExecutor._bind_async_runtime``); the
    wrapper never creates event loops or threads. The provider runtime is
    instantiated inside the bound loop so its async SDK client binds to the
    loop that serves every batch, and ``close()`` releases the client on
    that same loop at actor shutdown.
    """

    def __init__(
        self,
        descriptor: Any,
        column: str,
        output_column: str,
        dimensions: int,
        max_chunk_chars: int | None = None,
        chunk_overlap_chars: int = 200,
        max_retries: int = 3,
        on_error: _OnError = "raise",
        normalize: bool = False,
    ) -> None:
        self._descriptor = descriptor
        self._column = column
        self._output_column = output_column
        self._dimensions = dimensions
        self._max_chunk_chars = max_chunk_chars
        self._chunk_overlap_chars = chunk_overlap_chars
        self._max_retries = max_retries
        self._on_error: _OnError = on_error
        self._normalize = normalize
        self._arrow_type = pa.list_(pa.float32(), dimensions)
        self._embedder = None  # lazy: instantiate on first __call__
        self._run_async: Any = None  # executor-bound capability
        self._embedder_loop_bound = False  # set once the embedder is known loop-bound

    def bind_async_runtime(self, run_async: Any) -> None:
        """Receive the executor-owned async driver (see UDFExecutor)."""
        self._run_async = run_async

    def _require_run_async(self) -> Any:
        if self._run_async is None:
            raise _missing_async_runtime()
        return self._run_async

    def _mark_loop_bound(self) -> None:
        self._embedder_loop_bound = True

    def _ensure_embedder(self) -> Any:
        if self._embedder is None:
            run_async = self._require_run_async()

            async def _instantiate() -> Any:
                # Construct inside the bound loop so the async SDK client's
                # connection pool binds to the loop serving every batch.
                return self._descriptor.instantiate()

            self._embedder = run_async(_instantiate())
            if _provider_is_loop_bound(self._embedder, "embed_text"):
                self._embedder_loop_bound = True
        return self._embedder

    def close(self) -> None:
        """Release a loop-bound embedder on the bound loop. Idempotent.

        ``TextEmbedder.embed_text`` may be sync or async. A loop-bound embedder
        — a coroutine ``embed_text``, an observed awaitable result, or an
        ``aclose()`` hook — is dropped so the next batch re-instantiates on the
        fresh loop (issue #139); a missing ``aclose()`` is never treated as
        proof of synchronicity. A proven-synchronous embedder (e.g. the
        Transformers embedder) holds no loop-bound state and stays cached:
        task backends close the executor after every invocation while the
        process-local callable cache keeps the wrapper, so dropping it would
        reload the model on each task.
        """
        embedder = self._embedder
        if embedder is None or self._run_async is None:
            return
        if not self._embedder_loop_bound:
            return
        self._embedder = None
        aclose = getattr(embedder, "aclose", None)
        if aclose is not None:
            self._run_async(aclose())

    def __getstate__(self) -> dict[str, Any]:
        # The cached client and the bound runtime capability are process-local.
        state = self.__dict__.copy()
        state["_embedder"] = None
        state["_run_async"] = None
        state["_embedder_loop_bound"] = False  # recomputed on next _ensure_embedder()
        return state

    def _coerce_embedding(self, value: Any) -> np.ndarray:
        try:
            embedding = np.asarray(value, dtype=np.float32)
        except Exception as exc:
            raise TypeError("Provider returned a non-numeric embedding") from exc
        if embedding.ndim != 1 or embedding.size != self._dimensions:
            raise TypeError(
                f"Provider {self._descriptor.get_provider()!r} model {self._descriptor.get_model()!r} "
                f"returned an embedding with length {embedding.size}; expected {self._dimensions}"
            )
        return embedding

    def _invoke_embedder(self, texts: list[str]) -> list[np.ndarray]:
        raw = _retry_call(
            self._ensure_embedder().embed_text,
            texts,
            max_retries=self._max_retries,
            on_error="raise",
            run_async=self._require_run_async(),
            on_awaitable=self._mark_loop_bound,
            retry_if=_is_transient_embed_error,
        )
        try:
            values = list(raw)
        except TypeError as exc:
            raise TypeError("Provider embedding result must be a sequence") from exc
        if len(values) != len(texts):
            raise ValueError(
                f"Provider returned {len(values)} embeddings for {len(texts)} inputs; "
                "embedding calls must preserve row count and order"
            )
        return [self._coerce_embedding(value) for value in values]

    def _embed_texts(self, texts: list[str]) -> list[np.ndarray | None]:
        if not texts:
            return []
        try:
            return self._invoke_embedder(texts)
        except _MissingAsyncRuntimeError:
            raise
        except ProviderCapabilityError:
            if self._on_error == "raise":
                raise
            return [None] * len(texts)
        except Exception:
            if self._on_error == "raise":
                raise

        # A batch-level failure does not identify the failing row. Isolate the
        # inputs so on_error="ignore" nulls only rows that fail independently.
        isolated: list[np.ndarray | None] = []
        for text in texts:
            try:
                isolated.append(self._invoke_embedder([text])[0])
            except _MissingAsyncRuntimeError:
                raise
            except Exception:
                isolated.append(None)
        return isolated

    def __call__(self, table: pa.Table) -> pa.Table:
        texts = table.column(self._column).to_pylist()
        active_indices = [index for index, text in enumerate(texts) if text is not None]
        results: list[Any] = [None] * len(texts)

        if active_indices:
            active_texts = [texts[index] for index in active_indices]
            if self._max_chunk_chars is not None:
                active_results = self._embed_with_chunking(active_texts)
            else:
                active_results = self._embed_texts(active_texts)

            if self._normalize:
                active_results = _normalize_embeddings(active_results)
            for index, value in zip(active_indices, active_results, strict=True):
                results[index] = value

        embeddings = pa.array(
            [None if value is None else value.tolist() for value in results],
            type=self._arrow_type,
        )
        return pa.table({self._output_column: embeddings})

    def _embed_with_chunking(self, texts: list[str]) -> list[np.ndarray | None]:
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

        chunk_embeddings = self._embed_texts(all_chunks)

        # Reassemble: weighted average for multi-chunk texts
        results: list[np.ndarray | None] = []
        for entry in chunk_map:
            embeddings = [chunk_embeddings[index] for index, _ in entry]
            if any(embedding is None for embedding in embeddings):
                results.append(None)
                continue
            if len(entry) == 1:
                results.append(embeddings[0])
            else:
                weights = [w for _, w in entry]
                results.append(_weighted_average_embeddings(embeddings, weights))
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
    alongside text into multimodal message tuples. List-valued image cells are
    expanded in order and NULL or zero-length image values are skipped.

    Async execution is driven exclusively through an executor-bound
    ``run_async`` capability (see ``UDFExecutor._bind_async_runtime``); the
    wrapper never creates event loops or threads. The provider runtime is
    instantiated inside the bound loop so its async SDK client binds to the
    loop that serves every batch, and ``close()`` releases the client on
    that same loop at actor shutdown.
    """

    def __init__(
        self,
        descriptor: Any,
        column: str,
        output_column: str,
        max_api_concurrency: int | None = None,
        return_format: Any | None = None,
        image_columns: list[str] | None = None,
        propagate_null_prompts: bool = False,
        max_retries: int = 3,
        on_error: _OnError = "raise",
    ) -> None:
        self._descriptor = descriptor
        self._column = column
        self._output_column = output_column
        self._max_api_concurrency = max_api_concurrency
        self._return_format = return_format
        self._image_columns = image_columns or []
        self._propagate_null_prompts = propagate_null_prompts
        self._max_retries = max_retries
        self._on_error: _OnError = on_error
        self._prompter = None  # lazy: instantiate on first __call__
        self._run_async: Any = None  # executor-bound capability
        self._prompter_loop_bound = False  # set once the prompter is known loop-bound

    def bind_async_runtime(self, run_async: Any) -> None:
        """Receive the executor-owned async driver (see UDFExecutor)."""
        self._run_async = run_async

    def _require_run_async(self) -> Any:
        if self._run_async is None:
            raise _missing_async_runtime()
        return self._run_async

    def _mark_loop_bound(self) -> None:
        self._prompter_loop_bound = True

    def _ensure_prompter(self) -> Any:
        if self._prompter is None:
            run_async = self._require_run_async()

            async def _instantiate() -> Any:
                # Construct inside the bound loop so the async SDK client's
                # connection pool binds to the loop serving every batch.
                return self._descriptor.instantiate()

            self._prompter = run_async(_instantiate())
            if _provider_is_loop_bound(self._prompter, "prompt"):
                self._prompter_loop_bound = True
        return self._prompter

    def close(self) -> None:
        """Release a loop-bound prompter on the bound loop. Idempotent.

        ``Prompter.prompt`` is ``async`` by protocol, so a conforming prompter
        owns loop-bound state (its async SDK client) and is dropped on close so
        the next batch re-instantiates on the fresh loop (issue #139),
        ``aclose()`` being awaited when the client exposes it; a missing
        ``aclose()`` is never treated as proof of synchronicity. A provider
        that proves synchronous (no coroutine ``prompt``, no observed
        awaitable, no ``aclose``) stays cached so per-task executors do not
        reload it. (vLLM prompting is planner-only and is never cached here.)
        """
        prompter = self._prompter
        if prompter is None or self._run_async is None:
            return
        if not self._prompter_loop_bound:
            return
        self._prompter = None
        aclose = getattr(prompter, "aclose", None)
        if aclose is not None:
            self._run_async(aclose())

    def __getstate__(self) -> dict[str, Any]:
        # The cached client and the bound runtime capability are process-local.
        state = self.__dict__.copy()
        state["_prompter"] = None
        state["_run_async"] = None
        state["_prompter_loop_bound"] = False  # recomputed on next _ensure_prompter()
        return state

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

        return json.dumps(result, default=str)

    def __call__(self, table: pa.Table) -> pa.Table:
        texts = table.column(self._column).to_pylist()
        active_indices = [idx for idx, text in enumerate(texts) if not (self._propagate_null_prompts and text is None)]
        results: list[Any] = [None] * len(texts)
        if self._propagate_null_prompts and not active_indices:
            return pa.table({self._output_column: pa.array(results, type=pa.string())})

        self._ensure_prompter()
        texts = [t if t is not None else "" for t in texts]

        # Build per-row message tuples (text + optional image columns)
        image_lists: list[list[Any]] = [table.column(col_name).to_pylist() for col_name in self._image_columns]

        def has_image_data(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, (bytes, bytearray, memoryview)):
                return len(value) > 0
            return True

        def build_messages(idx: int) -> tuple[Any, ...]:
            parts: list[Any] = [texts[idx]]
            for img_col in image_lists:
                val = img_col[idx]
                if isinstance(val, list):
                    parts.extend(item for item in val if has_image_data(item))
                elif has_image_data(val):
                    parts.append(val)
            return tuple(parts)

        row_messages = [build_messages(idx) for idx in range(len(texts))]

        # Keep text-only rows on the batch API (e.g. vLLM's continuous
        # batching), even when other rows in the Arrow batch contain images.
        prompt_batch = getattr(self._prompter, "prompt_batch", None)
        if callable(prompt_batch):
            text_indices = [idx for idx in active_indices if len(row_messages[idx]) == 1]
            prompt_indices = [idx for idx in active_indices if len(row_messages[idx]) > 1]
        else:
            text_indices = []
            prompt_indices = active_indices

        if text_indices:
            text_results = _retry_call(
                prompt_batch,
                [texts[idx] for idx in text_indices],
                max_retries=self._max_retries,
                on_error=self._on_error,
                run_async=self._require_run_async(),
                on_awaitable=self._mark_loop_bound,
            )
            if text_results is None:
                text_results = [None] * len(text_indices)
            if self._return_format is not None:
                text_results = [self._serialize_result(result) for result in text_results]
            for idx, result in zip(text_indices, text_results, strict=True):
                results[idx] = result

        max_retries = self._max_retries
        on_error = self._on_error

        async def run_all() -> list[str | None]:
            if self._max_api_concurrency is not None and self._max_api_concurrency > 0:
                sem = asyncio.Semaphore(self._max_api_concurrency)

                async def limited(idx: int) -> str | None:
                    async with sem:
                        result = await _retry_call_async(
                            self._prompter.prompt,
                            row_messages[idx],
                            max_retries=max_retries,
                            on_error=on_error,
                            on_awaitable=self._mark_loop_bound,
                        )
                        return self._serialize_result(result) if self._return_format else result

                return await asyncio.gather(*(limited(idx) for idx in prompt_indices))

            async def single(idx: int) -> str | None:
                result = await _retry_call_async(
                    self._prompter.prompt,
                    row_messages[idx],
                    max_retries=max_retries,
                    on_error=on_error,
                    on_awaitable=self._mark_loop_bound,
                )
                return self._serialize_result(result) if self._return_format else result

            return await asyncio.gather(*(single(idx) for idx in prompt_indices))

        if prompt_indices:
            prompt_results = self._require_run_async()(run_all())
            for idx, result in zip(prompt_indices, prompt_results, strict=True):
                results[idx] = result
        return pa.table({self._output_column: results})


class _ValidateVLLMStructuredOutputBatch:
    """Apply the Pydantic runtime contract to native vLLM output."""

    def __init__(
        self,
        return_format: Any,
        input_column: str,
        output_column: str,
        on_error: _OnError,
    ) -> None:
        self._return_format = return_format
        self._input_column = input_column
        self._output_column = output_column
        self._on_error = on_error

    def __call__(self, table: pa.Table) -> pa.Table:
        results: list[str | None] = []
        for raw_text in table.column(self._input_column).to_pylist():
            if raw_text is None:
                results.append(None)
                continue
            try:
                validated = self._return_format.model_validate_json(raw_text)
                results.append(validated.model_dump_json())
            except Exception:
                if self._on_error == "raise":
                    raise
                results.append(None)
        return pa.table({self._output_column: pa.array(results, type=pa.string())})


def _build_ai_batch_expression(
    wrapper: Any,
    *,
    input_name: str,
    input_expr: Any,
    output_column: str,
    output_type: str,
    udf_opts: UDFOptions,
    name: str,
    execution_backend: str | None = None,
    force_actor: bool = True,
) -> Any:
    actor_backend = execution_backend in {"subprocess_actor", "ray_actor"}
    if execution_backend is None and force_actor:
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

    configured = _adapt_batch_wrapper_for_backend(
        wrapper,
        execution_backend,
        force_actor=actor_backend,
    )
    return _build_map_batches_expression(
        configured,
        name=name,
        inputs={input_name: as_expression(input_expr)},
        schema={output_column: output_type},
        batch_size=_resolve_ai_batch_size(udf_opts),
        row_preserving=True,
        gpus=_gpus_or_zero(udf_opts),
        execution_backend=execution_backend,
        actor_number=_actor_number_or_one(udf_opts) if actor_backend else None,
    )


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------


def _validated_embed_text(text: Any) -> Any:
    """Add a bind-only VARCHAR guard that is removed during planning."""
    return duckdb.FunctionExpression("__vane_ai_embed", text)


def _embed_expression(
    text: Any,
    *,
    provider: Any,
    model: str | None,
    dimensions: int | None,
    on_error: str,
    options: dict[str, Any],
) -> Any:
    if not is_expression(text):
        raise TypeError("vane.ai.embed expression API requires a text Expression")
    descriptor, resolved_dimensions, udf_opts, normalize, _, _, _, _ = _prepare_embed_call(
        provider,
        model,
        dimensions,
        on_error,
        options,
        relation=False,
    )
    wrapper = _EmbedTextBatch(
        descriptor,
        "text",
        "embedding",
        resolved_dimensions,
        max_retries=udf_opts.max_retries,
        on_error=udf_opts.on_error,
        normalize=normalize,
    )
    output_type = f"FLOAT[{resolved_dimensions}]"
    return _build_ai_batch_expression(
        wrapper,
        input_name="text",
        input_expr=_validated_embed_text(text),
        output_column="embedding",
        output_type=output_type,
        udf_opts=udf_opts,
        name="ai_embed",
    ).cast(output_type)


def _embed_relation(
    rel: Any,
    text: Any,
    *,
    provider: Any,
    model: str | None,
    dimensions: int | None,
    on_error: str,
    output_column: str,
    options: dict[str, Any],
) -> Any:
    if not _is_relation_like(rel):
        raise TypeError("vane.ai.embed relation API requires a Relation")
    if not is_expression(text):
        raise TypeError("vane.ai.embed relation API requires a text Expression")
    if not isinstance(output_column, str) or not output_column.strip():
        raise ValueError("output_column must be a non-empty string")

    (
        descriptor,
        resolved_dimensions,
        udf_opts,
        normalize,
        max_chunk_chars,
        chunk_overlap_chars,
        execution_backend,
        actor_number_explicit,
    ) = _prepare_embed_call(
        provider,
        model,
        dimensions,
        on_error,
        options,
        relation=True,
    )
    wrapper = _EmbedTextBatch(
        descriptor,
        "text",
        output_column,
        resolved_dimensions,
        max_chunk_chars=max_chunk_chars,
        chunk_overlap_chars=chunk_overlap_chars,
        max_retries=udf_opts.max_retries,
        on_error=udf_opts.on_error,
        normalize=normalize,
    )
    output_type = f"FLOAT[{resolved_dimensions}]"
    expression = _build_ai_batch_expression(
        wrapper,
        input_name="text",
        input_expr=_validated_embed_text(text),
        output_column=output_column,
        output_type=output_type,
        udf_opts=udf_opts,
        name="ai_embed",
        execution_backend=execution_backend,
        force_actor=actor_number_explicit,
    ).cast(output_type)
    return rel.select(duckdb.StarExpression(), expression.alias(output_column))


_EMBED_ARGUMENT_UNSET: Any = object()


class _DefaultEmbedOutputColumn(str):
    """Distinguish an omitted Relation-only keyword from an explicit value."""


_EMBED_OUTPUT_COLUMN_DEFAULT = _DefaultEmbedOutputColumn("embedding")


@overload
def embed(
    text: Expression,
    *,
    provider: str | Provider = "openai",
    model: str | None = None,
    dimensions: int | None = None,
    on_error: Literal["raise", "ignore"] = "raise",
    **options: Unpack[EmbedOptions],
) -> Expression: ...


@overload
def embed(
    *,
    text: Expression,
    provider: str | Provider = "openai",
    model: str | None = None,
    dimensions: int | None = None,
    on_error: Literal["raise", "ignore"] = "raise",
    **options: Unpack[EmbedOptions],
) -> Expression: ...


@overload
def embed(
    rel: Relation,
    text: Expression,
    *,
    provider: str | Provider = "openai",
    model: str | None = None,
    dimensions: int | None = None,
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: str = "embedding",
    **options: Unpack[EmbedOptions],
) -> Relation: ...


@overload
def embed(
    *,
    rel: Relation,
    text: Expression,
    provider: str | Provider = "openai",
    model: str | None = None,
    dimensions: int | None = None,
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: str = "embedding",
    **options: Unpack[EmbedOptions],
) -> Relation: ...


def embed(
    first: Expression | Relation = _EMBED_ARGUMENT_UNSET,
    /,
    text: Expression = _EMBED_ARGUMENT_UNSET,
    *,
    rel: Relation = _EMBED_ARGUMENT_UNSET,
    provider: str | Provider = "openai",
    model: str | None = None,
    dimensions: int | None = None,
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: str = _EMBED_OUTPUT_COLUMN_DEFAULT,
    **options: Unpack[EmbedOptions],
) -> Expression | Relation:
    """Embed an Expression or append an embedding column to a Relation."""

    if first is not _EMBED_ARGUMENT_UNSET and rel is not _EMBED_ARGUMENT_UNSET:
        raise TypeError("vane.ai.embed received both first and rel; pass only one relation argument")

    relation = rel if rel is not _EMBED_ARGUMENT_UNSET else first
    if relation is not _EMBED_ARGUMENT_UNSET and _is_relation_like(relation):
        if text is _EMBED_ARGUMENT_UNSET:
            raise TypeError("vane.ai.embed relation API requires a text Expression")
        resolved_output_column = "embedding" if output_column is _EMBED_OUTPUT_COLUMN_DEFAULT else output_column
        return _embed_relation(
            relation,
            text,
            provider=provider,
            model=model,
            dimensions=dimensions,
            on_error=on_error,
            output_column=resolved_output_column,
            options=options,
        )

    if rel is not _EMBED_ARGUMENT_UNSET:
        raise TypeError("vane.ai.embed rel= must be a Relation")
    if first is not _EMBED_ARGUMENT_UNSET and text is not _EMBED_ARGUMENT_UNSET:
        raise TypeError("vane.ai.embed expression API accepts a single text Expression")
    expression = text if first is _EMBED_ARGUMENT_UNSET else first
    if expression is _EMBED_ARGUMENT_UNSET:
        raise TypeError("vane.ai.embed requires a text Expression or a Relation plus text Expression")
    if output_column is not _EMBED_OUTPUT_COLUMN_DEFAULT:
        raise TypeError("vane.ai.embed expression API does not accept output_column; use .alias(...)")
    return _embed_expression(
        expression,
        provider=provider,
        model=model,
        dimensions=dimensions,
        on_error=on_error,
        options=options,
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


def _build_native_vllm_expression(messages: Any, descriptor: NativeVLLMPromptPlan) -> duckdb.Expression:
    """Build the native row-preserving ``vllm()`` expression."""
    messages_expr = as_expression(messages)
    if descriptor.system_message:
        # Unlike concat(), the || operator propagates NULL inputs.
        messages_expr = duckdb.FunctionExpression(
            "||",
            duckdb.ConstantExpression(f"{descriptor.system_message}\n\n"),
            messages_expr,
        )

    options_argument = _build_native_vllm_options_argument(descriptor.build_physical_vllm_options())

    return duckdb.FunctionExpression(
        "vllm",
        messages_expr,
        duckdb.ConstantExpression(descriptor.model_name),
        duckdb.ConstantExpression(options_argument),
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

    ``execution_backend`` is optional for Python UDF providers; native vLLM
    prompting rejects it because execution is owned by the query runner.
    Prefer provider environment variables such as ``OPENAI_API_KEY``,
    ``ANTHROPIC_API_KEY``, or ``GOOGLE_API_KEY`` over passing API keys in call
    options.
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

    if isinstance(descriptor, NativeVLLMPromptPlan):
        if image_columns:
            raise ValueError("native vLLM prompting does not support image_columns")
        if execution_backend is not None:
            raise ValueError(
                "execution_backend applies only to Python UDF providers; "
                "native vLLM routing is derived from the query runner"
            )
        relation_alias = getattr(rel, "alias", None)
        column_expr = (
            duckdb.ColumnExpression(relation_alias, column)
            if isinstance(relation_alias, str) and relation_alias
            else duckdb.ColumnExpression(column)
        )
        expression = _build_native_vllm_expression(column_expr, descriptor)
        native_return_format = descriptor.return_format
        if hasattr(native_return_format, "model_validate"):
            input_column = "__vane_vllm_response"
            raw_on_error = descriptor.vllm_options.get("on_error", "raise")
            validation_on_error: _OnError = (
                "ignore" if str(raw_on_error).strip().lower() in {"ignore", "log", "null"} else "raise"
            )
            validator = _ValidateVLLMStructuredOutputBatch(
                native_return_format,
                input_column,
                output_column,
                validation_on_error,
            )
            expression = _build_map_batches_expression(
                validator.__call__,
                name="validate_vllm_structured_output",
                inputs={input_column: expression},
                schema={output_column: "VARCHAR"},
                batch_size=None,
                row_preserving=True,
                gpus=0,
            )
        expression = expression.alias(output_column)
        return rel.select(expression)
    if isinstance(descriptor, NativePrompterPlan):
        raise ValueError(f"Unsupported native prompt plan {type(descriptor).__name__}")

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

    if isinstance(descriptor, NativeVLLMPromptPlan):
        return _build_native_vllm_expression(messages, descriptor)
    if isinstance(descriptor, NativePrompterPlan):
        raise ValueError(f"Unsupported native prompt plan {type(descriptor).__name__}")

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
    The relation API accepts ``execution_backend`` for Python UDF providers;
    native vLLM prompting rejects explicit values because the query runner owns
    its execution mode.
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
