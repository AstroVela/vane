# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""High-level AI functions that wrap descriptors into map_batches calls.

These functions create Actor wrapper classes with reconstructible local state that:
1. Accept a Descriptor (serializable, lightweight)
2. Lazily call ``instantiate()`` on the worker to load the model once
3. Process each batch through the loaded model

Usage::

    import vane
    from vane.ai.functions import embed

    conn = vane.connect()
    rel = conn.sql("SELECT text FROM documents")

    # Text embedding — returns relation with 'embedding' column
    embedded = embed(
        rel,
        vane.col("text"),
        provider="transformers",
        model="sentence-transformers/all-MiniLM-L6-v2",
    )

"""

from __future__ import annotations

import asyncio
import datetime
import email.utils
import inspect
import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeGuard, cast, overload

import numpy as np
import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]
from typing_extensions import Unpack

import vane
from vane._expression_udf import _build_actor_map_batches_expression, _build_map_batches_expression
from vane._expressions import as_expression, is_expression
from vane._typing import Expression, Relation
from vane.ai._media import PromptMedia
from vane.ai._schema import (
    OutputValidationError,
    RawResponseSerializationError,
    StructuredOutputSpec,
    compile_return_format,
    validate_raw_response_json,
)
from vane.ai.options import (
    EmbedOptions,
    PromptOptions,
    normalize_prompt_options,
    validate_embed_options,
)
from vane.ai.protocols import NativeInferencePlan, NativePrompterPlan
from vane.ai.provider import (
    Provider,
    ProviderCapabilityError,
    _ProviderResultError,
    _safe_original_error_summary,
    _safe_provider_capability_error,
    _safe_provider_execution_error,
    load_provider,
)
from vane.ai.providers.vllm import _build_native_vllm_options_argument
from vane.ai.typing import JSONSchema, UDFOptions

if TYPE_CHECKING:
    from pydantic import BaseModel  # type: ignore[import-not-found, import-untyped, unused-ignore]

    from vane.ai.protocols import Prompter, TextEmbedder
else:
    BaseModel = Any


def _resolve_provider(provider: str | Provider) -> Provider:
    """Resolve a provider argument to a Provider instance."""
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

_OnError = Literal["raise", "ignore"]

logger = logging.getLogger(__name__)


def _validate_on_error(on_error: object) -> _OnError:
    if not isinstance(on_error, str) or on_error not in ("raise", "ignore"):
        raise ValueError("on_error must be 'raise' or 'ignore'")
    return cast(_OnError, on_error)


def _log_substituted_failure(exc: Exception, *, on_error: _OnError, attempts: int | None = None) -> None:
    """Record one bounded WARNING when a provider failure is replaced with NULL.

    Called at each site that swallows a provider error and yields NULL under
    ``on_error='ignore'`` — including the native vLLM executor, whose ``"null"``
    policy is the lowered form of ``'ignore'`` (see the mapping in
    ``vane/ai/providers/vllm.py``); a no-op under ``'raise'`` so callers cannot
    log a failure they are about to re-raise. ``attempts`` is the number of tries made
    when the caller is a retry loop (omitted where no attempt count is
    meaningful). Only the exception class, its sanitized numeric-status summary,
    and that count are emitted — never prompt text, row payloads, option
    mappings, credentials, or a traceback (vane#105) — so a silent data
    degradation becomes observable without leaking sensitive input.
    """
    if on_error == "raise":
        return
    summary = _safe_original_error_summary(exc)
    if attempts is not None:
        logger.warning(
            "vane.ai substituted NULL after %d attempt(s) for a failed provider call: %s",
            attempts,
            summary,
        )
    else:
        logger.warning("vane.ai substituted NULL for a failed provider call: %s", summary)


# Sentinel a retry-helper caller can pass as ``default`` to tell an
# already-logged substituted failure apart from a genuine provider NULL —
# the two must not be conflated, or a NULL that fails downstream result
# validation would be swallowed without its own warning.
_SUBSTITUTED_FAILURE = object()


def _rebuild_retry_after_error(args: tuple[Any, ...], state: dict[str, Any]) -> RetryAfterError:
    """Reconstruct a pickled :class:`RetryAfterError` from its sanitized state.

    The constructor derives the message and status attributes from an
    ``original`` exception, so feeding a pickled message back through it would
    misassign the message string to ``retry_after`` and require re-transporting
    the original provider error — defeating the sanitization. Rebuild the
    instance directly from its already-sanitized state instead.
    """
    exc = RetryAfterError.__new__(RetryAfterError)
    exc.args = args
    exc.__dict__.update(state)
    return exc


class RetryAfterError(Exception):
    """Retryable error carrying the requested wait time (in seconds).

    Providers raise this when they receive a rate-limit (429) or
    service-unavailable (503) response with a ``Retry-After`` header.
    The retry helpers honour :attr:`retry_after` for the sleep duration when
    it is a finite, non-negative number (capped at 120s); any other value
    falls back to exponential backoff (see :func:`_retry_wait_seconds`).
    """

    status_code: int  # normalized HTTP status; absent when never discovered

    def __init__(self, retry_after: float, original: Exception | None = None, status: int | None = None) -> None:
        self.retry_after = retry_after
        if original is None:
            super().__init__("RetryAfterError")
        else:
            super().__init__(_safe_original_error_summary(original))
            for name in ("status_code", "status", "code"):
                try:
                    value = getattr(original, name, None)
                except Exception:
                    continue
                if type(value) is int and -999_999 <= value <= 999_999:
                    setattr(self, name, value)
        if status is not None:
            # A status discovered only on the original's attached response is
            # not a direct attribute, so the whitelist above cannot see it;
            # the caller's normalized discovery wins over any direct copy.
            self.status_code = status

    def __reduce__(self) -> tuple[Any, ...]:
        # The default BaseException reduction rebuilds via ``__init__(*self.args)``,
        # which feeds the sanitized message in as ``retry_after`` and regenerates
        # a generic message — losing the summary across a pickle boundary (Ray
        # worker -> driver). Rebuild from the already-sanitized instance state
        # (numeric ``retry_after`` plus whitelisted status attributes) without
        # re-running constructor sanitization or resurrecting the original error.
        return (_rebuild_retry_after_error, (self.args, dict(self.__dict__)))


def _retry_wait_seconds(exc: Exception, attempt: int) -> float:
    """Seconds to sleep before the next retry attempt.

    A :class:`RetryAfterError` whose ``retry_after`` is a finite, non-negative
    number is honoured, capped at 120s. Any other value falls back to
    exponential backoff: a negative, NaN, or infinite delay — reachable from a
    malformed but parseable provider ``Retry-After`` header (Google parses the
    header with ``float()``) or a misbehaving custom provider — must never reach
    ``time.sleep``/``asyncio.sleep``, where a negative or NaN value raises
    ``ValueError`` (sync) or can hang the retry forever (async NaN), replacing
    the provider's retry/``on_error`` behaviour with a spurious sleep failure
    (issue #469).
    """
    if isinstance(exc, RetryAfterError):
        retry_after = exc.retry_after
        if isinstance(retry_after, (int, float)) and not isinstance(retry_after, bool):
            try:
                wait = float(retry_after)
            except OverflowError:  # int beyond float range — not a usable delay
                wait = math.inf
            if math.isfinite(wait) and wait >= 0:
                return min(wait, 120)
    return float(min(2**attempt, 30))  # 1, 2, 4, 8, ... capped at 30s


_TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 409, 425, 429})


def _is_transient_provider_error(exc: Exception) -> bool:
    """Whether a provider request failed for a transient reason."""

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
            getattr(getattr(current, "response", None), "status", None),
        ):
            if isinstance(status, int) and not isinstance(status, bool):
                if status in _TRANSIENT_HTTP_STATUS_CODES or 500 <= status <= 599:
                    return True

        # OpenAI connection errors retain their underlying httpx transport
        # error as the cause.  Avoid importing an optional SDK just to classify
        # it; httpx itself is likewise optional for non-network providers.
        try:
            import httpx  # type: ignore[import-not-found, import-untyped, unused-ignore]
        except ImportError:
            pass
        else:
            if isinstance(current, httpx.TransportError):
                return True

        try:
            import aiohttp  # type: ignore[import-not-found, import-untyped, unused-ignore]
        except ImportError:
            pass
        else:
            if isinstance(current, aiohttp.ClientConnectionError):
                return True

        current = current.__cause__

    return False


_RATE_LIMIT_STATUS_CODES = frozenset({429, 503})
_DEFAULT_RETRY_AFTER_SECONDS = 5.0


def _provider_status_code(exc: Exception) -> int | None:
    """Discover an HTTP status from a provider SDK error.

    Providers surface the status differently — OpenAI/Anthropic expose
    ``status_code``, Google exposes ``code``, and some attach it to the
    response — so probe the common locations in a stable order.
    """
    for status in (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
        getattr(getattr(exc, "response", None), "status", None),
    ):
        if isinstance(status, int) and not isinstance(status, bool):
            return status
    return None


def _utcnow() -> datetime.datetime:
    """Current UTC time; a seam so fixed-clock tests can pin HTTP-date parsing."""
    return datetime.datetime.now(datetime.timezone.utc)


def _retry_after_http_date_delay(raw: str) -> float | None:
    """Seconds until an HTTP-date ``Retry-After`` value, or ``None`` when the
    value is not a parseable date.

    A date already in the past means "retry immediately", so it clamps to zero
    rather than being rejected as malformed.
    """
    try:
        when = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        # RFC 5322 parses a "-0000" offset to a naive datetime; HTTP-dates are
        # always GMT, so interpret it as UTC.
        when = when.replace(tzinfo=datetime.timezone.utc)
    return max(0.0, (when - _utcnow()).total_seconds())


def _parse_retry_after_header(exc: Exception) -> float | None:
    """Return a usable ``Retry-After`` delay from a provider error's response.

    RFC 9110 permits either delay-seconds or an HTTP-date; both are accepted,
    a date being converted to the remaining delay. Returns ``None`` when no
    response, no header, or a malformed value is present. A negative, NaN, or
    infinite delay-seconds header (all parseable by ``float()``, e.g. ``"-1"``,
    ``"nan"``, ``"1e999"``) is treated as malformed and rejected so it never
    reaches a retry sleep that would raise or hang (issue #469).
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None) or {}
    try:
        raw = headers.get("Retry-After") or headers.get("retry-after")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        date_delay = _retry_after_http_date_delay(raw) if isinstance(raw, str) else None
        if date_delay is None:
            return None
        parsed = date_delay
    if math.isfinite(parsed) and parsed >= 0:
        return parsed
    return None


def _retry_after_error(exc: Exception) -> RetryAfterError | None:
    """Build a :class:`RetryAfterError` for a rate-limited/unavailable provider
    error, or ``None`` when the error does not qualify.

    Shared by every HTTP provider (issue #148) so 429/503 retry timing is
    uniform: the server-requested ``Retry-After`` is honoured when present and
    usable, otherwise a default wait applies. The returned error carries the
    original for its sanitized summary and status attributes only.
    """
    status = _provider_status_code(exc)
    if status not in _RATE_LIMIT_STATUS_CODES:
        return None
    retry_after = _parse_retry_after_header(exc)
    if retry_after is None:
        retry_after = _DEFAULT_RETRY_AFTER_SECONDS
    return RetryAfterError(retry_after=retry_after, original=exc, status=status)


def _retry_call(
    fn: Any,
    *args: Any,
    max_retries: int = 3,
    on_error: _OnError = "raise",
    default: Any = None,
    run_async: Any = None,
    on_awaitable: Any = None,
    **kwargs: Any,
) -> Any:
    """Call *fn* with exponential-backoff retry and on_error handling.

    Args:
        fn: Callable (sync or async) to invoke.
        max_retries: Number of retry attempts after the first failure (0 = no retries).
        on_error: ``"raise"`` re-raises on final failure; ``"ignore"``
            returns *default*.
        default: Value to return when on_error is not ``"raise"``.
        run_async: Callable used to drive awaitable results. Batch wrappers
            pass their executor-bound runtime so cached SDK clients see one
            loop across batches; required when *fn* returns an awaitable.
        on_awaitable: Optional zero-arg callback invoked the first time *fn*
            returns an awaitable — lets a wrapper learn its provider is
            loop-bound even when static inspection cannot tell (a plain
            ``def`` that returns an awaitable), so ``close()`` releases it.
    """
    _validate_on_error(on_error)
    last_exc: Exception | None = None
    attempts = 0
    for attempt in range(1 + max(0, max_retries)):
        attempts = attempt + 1
        wait: float | None = None
        try:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                if on_awaitable is not None:
                    on_awaitable()
                if run_async is None:
                    if inspect.iscoroutine(result):
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
            if isinstance(
                exc,
                (ProviderCapabilityError, _ProviderResultError, OutputValidationError, RawResponseSerializationError),
            ):
                break
            if not _is_transient_provider_error(exc):
                break
            if attempt < max_retries:
                wait = _retry_wait_seconds(exc, attempt)
        # Do not sleep while the provider exception is the active exception.
        # Otherwise an interrupt can retain the raw SDK error as its context.
        if wait is not None:
            time.sleep(wait)

    # All retries exhausted
    assert last_exc is not None
    if on_error == "raise":
        raise last_exc
    _log_substituted_failure(last_exc, on_error=on_error, attempts=attempts)
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
    _validate_on_error(on_error)
    last_exc: Exception | None = None
    attempts = 0
    for attempt in range(1 + max(0, max_retries)):
        attempts = attempt + 1
        wait: float | None = None
        try:
            result = fn(*args, **kwargs)
            if on_awaitable is not None and inspect.isawaitable(result):
                on_awaitable()
            return await result
        except Exception as exc:
            last_exc = exc
            if isinstance(
                exc,
                (ProviderCapabilityError, _ProviderResultError, OutputValidationError, RawResponseSerializationError),
            ):
                break
            if not _is_transient_provider_error(exc):
                break
            if attempt < max_retries:
                wait = _retry_wait_seconds(exc, attempt)
        # Keep cancellation outside the raw provider exception handler so the
        # provider error cannot become CancelledError.__context__.
        if wait is not None:
            await asyncio.sleep(wait)

    assert last_exc is not None
    if on_error == "raise":
        raise last_exc
    _log_substituted_failure(last_exc, on_error=on_error, attempts=attempts)
    return default


def _map_batches_kwargs(
    udf_opts: UDFOptions,
    execution_backend: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build keyword arguments for ``rel.map_batches()``."""
    num_gpus = udf_opts.num_gpus
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
        resolved = descriptor.get_dimensions()
    except Exception as exc:
        model = getattr(descriptor, "get_model", lambda: "unknown")()
        raise ValueError(
            f"Cannot determine embedding dimensions for model {model!r} without network or model loading; "
            "pass dimensions=... explicitly"
        ) from exc
    if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved <= 0:
        raise ValueError("Provider get_dimensions() must return a positive integer")
    return int(resolved)


def _prepare_embed_call(
    provider: Any,
    model: str | None,
    dimensions: int | None,
    on_error: str,
    options: Mapping[str, Any],
    *,
    relation: bool,
) -> tuple[Any, int, UDFOptions, bool, int | None, int, str | None]:
    """Resolve one Embed call without performing network or model I/O."""

    if provider is None:
        raise TypeError("provider must be a provider name or Provider object, not None")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("model must be a non-empty string or None")
    explicit_dimensions = _validate_embedding_dimension(dimensions)
    on_error = _validate_on_error(on_error)

    resolved_provider = _resolve_provider(provider)
    if not isinstance(resolved_provider, Provider):
        raise TypeError("provider must be a provider name or Provider object")
    family = _embedding_provider_family(resolved_provider)
    prepared = validate_embed_options(family, options, relation=relation)
    normalize = prepared.pop("normalize", False)
    execution_backend = prepared.pop("execution_backend", None)
    max_chunk_chars = prepared.pop("max_chunk_chars", None)
    chunk_overlap_chars = prepared.pop("chunk_overlap_chars", 200)
    batch_size = prepared.pop("batch_size", None)
    max_retries = prepared.pop("max_retries", None)
    actor_number = prepared.pop("actor_number", None)

    try:
        descriptor = resolved_provider.get_text_embedder(
            model=model,
            dimensions=explicit_dimensions,
            options=prepared,
        )
    except NotImplementedError as exc:
        provider_name = getattr(resolved_provider, "name", type(resolved_provider).__name__)
        raise ValueError(f"Provider {provider_name!r} is not an embedding provider") from exc

    resolved_dimensions = _resolve_embedding_dimension(descriptor, explicit_dimensions)
    descriptor_resources = descriptor.get_udf_options()
    udf_options = UDFOptions(
        actor_number=actor_number if actor_number is not None else 1,
        num_gpus=descriptor_resources.num_gpus,
        max_retries=max_retries if max_retries is not None else 3,
        on_error=on_error,
        batch_size=batch_size if batch_size is not None else 64,
    )

    return (
        descriptor,
        resolved_dimensions,
        udf_options,
        normalize,
        max_chunk_chars,
        chunk_overlap_chars,
        execution_backend,
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


def _descriptor_identity(descriptor: Any) -> tuple[str, str]:
    """Read optional diagnostic identity without trusting descriptor errors."""
    values: list[str] = []
    for method_name, fallback in (("get_provider", "unknown"), ("get_model", "unknown")):
        try:
            method = getattr(descriptor, method_name, None)
            value = method() if callable(method) else None
        except Exception:
            value = None
        candidate = value[:256].strip() if isinstance(value, str) else ""
        values.append(candidate or fallback)
    return values[0], values[1]


def _normalize_embedding(value: np.ndarray) -> np.ndarray:
    """L2-normalize a finite float32 embedding without float32 overflow."""
    working = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(working))
    if not np.isfinite(norm):
        raise _ProviderResultError("Embedding normalization produced a non-finite norm")
    if norm > 0:
        working /= norm
    return working.astype(np.float32)


# ---------------------------------------------------------------------------
# Module-level wrapper classes (must be at module level for pickle)
# ---------------------------------------------------------------------------


class _EmbedTextBatch:
    """Actor-local wrapper — model loaded once per instance via instantiate().

    Async execution is driven exclusively through an executor-bound
    ``run_async`` capability (see ``UDFExecutor._bind_async_runtime``); the
    wrapper never creates event loops or threads. The provider runtime is
    instantiated inside the bound loop so its async SDK client binds to the
    loop that serves every batch, and ``close()`` releases the client on
    that same loop at actor shutdown. Actor reconstruction creates a fresh
    wrapper and reloads the model.
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
        _validate_on_error(on_error)
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
        self._embedder: TextEmbedder | None = None  # lazy: instantiate on first __call__
        self._run_async: Callable[[Awaitable[Any]], Any] | None = None  # executor-bound capability
        self._embedder_loop_bound = False  # set once the embedder is known loop-bound

    def bind_async_runtime(self, run_async: Callable[[Awaitable[Any]], Any]) -> None:
        """Receive the executor-owned async driver (see UDFExecutor)."""
        self._run_async = run_async

    def _require_run_async(self) -> Callable[[Awaitable[Any]], Any]:
        if self._run_async is None:
            raise _missing_async_runtime()
        return self._run_async

    def _mark_loop_bound(self) -> None:
        self._embedder_loop_bound = True

    def _ensure_embedder(self) -> TextEmbedder:
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
        close_error: Exception | None = None
        try:
            aclose = getattr(embedder, "aclose", None)
            if aclose is not None:
                self._run_async(aclose())
        except Exception as exc:
            close_error = exc
        if close_error is not None:
            provider, model = _descriptor_identity(self._descriptor)
            raise _safe_provider_execution_error(
                provider,
                model,
                "Embed cleanup",
                close_error,
            ) from None

    def __getstate__(self) -> dict[str, Any]:
        # The cached client and the bound runtime capability are process-local.
        state = self.__dict__.copy()
        state["_embedder"] = None
        state["_run_async"] = None
        state["_embedder_loop_bound"] = False  # recomputed on next _ensure_embedder()
        return state

    def _coerce_embedding(self, value: Any) -> np.ndarray:
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                embedding = np.asarray(value, dtype=np.float32)
        except Exception:
            raise _ProviderResultError("Provider returned a non-numeric embedding") from None
        if embedding.ndim != 1:
            provider, model = _descriptor_identity(self._descriptor)
            raise _ProviderResultError(
                f"Provider {provider!r} model {model!r} "
                f"returned an embedding with shape {embedding.shape}; expected a one-dimensional "
                f"embedding with length {self._dimensions}"
            )
        if embedding.size != self._dimensions:
            provider, model = _descriptor_identity(self._descriptor)
            raise _ProviderResultError(
                f"Provider {provider!r} model {model!r} "
                f"returned an embedding with length {embedding.size}; expected {self._dimensions}"
            )
        if not bool(np.isfinite(embedding).all()):
            provider, model = _descriptor_identity(self._descriptor)
            raise _ProviderResultError(
                f"Provider {provider!r} model {model!r} returned an embedding containing non-finite components"
            )
        return embedding

    def _coerce_embedding_rows(
        self,
        values: list[Any],
        *,
        allow_nulls: bool,
        normalize: bool = False,
    ) -> list[np.ndarray | None]:
        """Enforce the typed embedding contract for known result rows.

        A row-aligned result violation is already attributable to one input,
        so ``on_error="ignore"`` can null that row without calling the provider
        again. ``allow_nulls`` is reserved for rows already nulled by that
        policy or by a failed chunk.
        """
        embeddings: list[np.ndarray | None] = []
        for value in values:
            if value is None and allow_nulls:
                embeddings.append(None)
                continue
            try:
                embedding = self._coerce_embedding(value)
                if normalize:
                    embedding = self._coerce_embedding(_normalize_embedding(embedding))
            except _ProviderResultError as exc:
                if self._on_error == "raise":
                    raise
                _log_substituted_failure(exc, on_error=self._on_error)
                embeddings.append(None)
                continue
            embeddings.append(embedding)
        return embeddings

    def _invoke_embedder(self, texts: list[str]) -> list[np.ndarray | None]:
        capability_error: ProviderCapabilityError | None = None
        provider_error: Exception | None = None
        try:
            raw = _retry_call(
                self._ensure_embedder().embed_text,
                texts,
                max_retries=self._max_retries,
                on_error="raise",
                run_async=self._require_run_async(),
                on_awaitable=self._mark_loop_bound,
            )
        except _MissingAsyncRuntimeError:
            raise
        except ProviderCapabilityError as exc:
            capability_error = exc
        except (_ProviderResultError, OutputValidationError, RawResponseSerializationError):
            raise
        except Exception as exc:
            provider_error = exc
        if capability_error is not None:
            raise _safe_provider_capability_error(capability_error) from None
        if provider_error is not None:
            provider, model = _descriptor_identity(self._descriptor)
            raise _safe_provider_execution_error(provider, model, "Embed execution", provider_error) from None
        try:
            values = list(raw)
        except TypeError:
            raise _ProviderResultError("Provider embedding result must be a sequence") from None
        if len(values) != len(texts):
            raise _ProviderResultError(
                f"Provider returned {len(values)} embeddings for {len(texts)} inputs; "
                "embedding calls must preserve row count and order"
            )
        return self._coerce_embedding_rows(values, allow_nulls=False)

    def _embed_texts(self, texts: list[str]) -> list[np.ndarray | None]:
        if not texts:
            return []
        try:
            return self._invoke_embedder(texts)
        except _MissingAsyncRuntimeError:
            raise
        except ProviderCapabilityError as exc:
            if self._on_error == "raise":
                raise
            _log_substituted_failure(exc, on_error=self._on_error)
            return [None] * len(texts)
        except Exception:
            if self._on_error == "raise":
                raise

        # A batch-level failure does not identify the failing row. Isolate the
        # inputs so on_error="ignore" nulls only rows that fail independently.
        # The batch error itself is not logged here: each isolated row that
        # actually fails logs its own substitution below (a row that recovers
        # on isolation is not a substitution and must stay silent).
        isolated: list[np.ndarray | None] = []
        for text in texts:
            try:
                isolated.append(self._invoke_embedder([text])[0])
            except _MissingAsyncRuntimeError:
                raise
            except Exception as exc:
                _log_substituted_failure(exc, on_error=self._on_error)
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

            active_results = self._coerce_embedding_rows(
                active_results,
                allow_nulls=True,
                normalize=self._normalize,
            )
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


class _PromptBatch:
    """Actor-local row-preserving wrapper for ordered text/media Prompt parts."""

    def __init__(
        self,
        descriptor: Any,
        message_columns: list[str],
        output_column: str,
        max_concurrency_per_actor: int | None = None,
        single_message: bool = False,
        return_format: StructuredOutputSpec | None = None,
        return_raw_response: bool = False,
        max_retries: int = 3,
        on_error: _OnError = "raise",
        supports_media_inputs: bool | None = None,
    ) -> None:
        if not message_columns:
            raise ValueError("Prompt message_columns cannot be empty")
        _validate_on_error(on_error)
        self._descriptor = descriptor
        self._message_columns = list(message_columns)
        self._output_column = output_column
        self._max_concurrency_per_actor = max_concurrency_per_actor
        self._single_message = single_message
        self._return_format = return_format
        self._return_raw_response = return_raw_response
        self._max_retries = max_retries
        self._on_error: _OnError = on_error
        if supports_media_inputs is None:
            supports_media = getattr(descriptor, "supports_image_inputs", None)
            supports_media_inputs = True if not callable(supports_media) else bool(supports_media())
        self._supports_media_inputs = supports_media_inputs
        self._prompter: Prompter | None = None  # lazy: instantiate on first __call__
        self._run_async: Callable[[Awaitable[Any]], Any] | None = None  # executor-bound capability
        self._prompter_loop_bound = False  # set once the prompter is known loop-bound

    def bind_async_runtime(self, run_async: Callable[[Awaitable[Any]], Any]) -> None:
        """Receive the executor-owned async driver (see UDFExecutor)."""
        self._run_async = run_async

    def _require_run_async(self) -> Callable[[Awaitable[Any]], Any]:
        if self._run_async is None:
            raise _missing_async_runtime()
        return self._run_async

    def _mark_loop_bound(self) -> None:
        self._prompter_loop_bound = True

    def _ensure_prompter(self) -> Prompter:
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
        close_error: Exception | None = None
        try:
            aclose = getattr(prompter, "aclose", None)
            if aclose is not None:
                self._run_async(aclose())
        except Exception as exc:
            close_error = exc
        if close_error is not None:
            provider, model = _descriptor_identity(self._descriptor)
            raise _safe_provider_execution_error(
                provider,
                model,
                "Prompt cleanup",
                close_error,
            ) from None

    def __getstate__(self) -> dict[str, Any]:
        # The cached client and the bound runtime capability are process-local.
        state = self.__dict__.copy()
        state["_prompter"] = None
        state["_run_async"] = None
        state["_prompter_loop_bound"] = False  # recomputed on next _ensure_prompter()
        return state

    def _require_media_support(self) -> None:
        if self._supports_media_inputs:
            return
        provider, model = _descriptor_identity(self._descriptor)
        raise ValueError(f"Provider {provider!r} model {model!r} does not support Prompt media inputs")

    def _materialized_prompt_media(self, value: Any) -> PromptMedia:
        """Decode the locator-free value produced by the native FILE boundary."""
        if not isinstance(value, Mapping) or set(value) != {"content_type", "data", "error"}:
            raise TypeError("Prompt FILE input did not cross the native media boundary")
        error = value["error"]
        if error is not None:
            if error == "unsupported":
                self._require_media_support()
                raise RuntimeError("Prompt FILE capability validation produced an invalid result")
            if error == "read":
                provider, model = _descriptor_identity(self._descriptor)
                safe_error = OSError("FILE access failed")
                raise _safe_provider_execution_error(provider, model, "Prompt FILE read", safe_error) from None
            if error == "mime":
                raise ValueError("Prompt FILE MIME type is missing and content detection was inconclusive")
            if error == "invalid_mime":
                raise ValueError("FILE content_type must be a valid MIME type")
            if error == "empty":
                raise ValueError("Prompt FILE content cannot be zero length")
            if error == "too_large":
                raise ValueError("Prompt FILE content exceeds the 20 MiB materialization limit")
            if error == "vector_too_large":
                raise ValueError("Prompt FILE inputs exceed the 128 MiB materialization-vector limit")
            raise RuntimeError("Prompt FILE materialization produced an unknown error")

        data = value["data"]
        content_type = value["content_type"]
        if not isinstance(data, bytes):
            raise TypeError("Prompt FILE materialization did not produce bytes")
        return PromptMedia(data, content_type)

    def _append_message_value(self, parts: list[Any], value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            parts.append(value)
            return
        if isinstance(value, (bytes, bytearray, memoryview)):
            self._require_media_support()
            image = bytes(value)
            if not image:
                raise ValueError("Prompt image BLOB cannot be zero length")
            parts.append(image)
            return
        if isinstance(value, Mapping):
            parts.append(self._materialized_prompt_media(value))
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                if item is None:
                    continue
                if isinstance(item, (bytes, bytearray, memoryview)):
                    self._require_media_support()
                    image = bytes(item)
                    if not image:
                        raise ValueError("Prompt image BLOB cannot be zero length")
                    parts.append(image)
                elif isinstance(item, Mapping):
                    parts.append(self._materialized_prompt_media(item))
                else:
                    raise TypeError("Prompt media lists must contain only BLOB, FILE, or NULL values")
            return
        raise TypeError(
            "Prompt messages expressions must produce VARCHAR, BLOB, BLOB[], FILE, or FILE[] values; "
            f"got {type(value).__name__}"
        )

    def _build_row_messages(self, columns: list[list[Any]], index: int) -> tuple[Any, ...] | None:
        if self._single_message and columns[0][index] is None:
            return None
        parts: list[Any] = []
        for column in columns:
            self._append_message_value(parts, column[index])
        return tuple(parts) if parts else None

    def _validate_result(self, result: Any) -> str | None:
        if self._return_raw_response:
            return validate_raw_response_json(result)
        if self._return_format is not None:
            if result is None:
                raise OutputValidationError("Prompt provider returned NULL for a structured output row")
            return self._return_format.validate_json(result)
        if result is None or isinstance(result, str):
            return result
        raise _ProviderResultError(
            f"Prompt provider returned {type(result).__name__}; basic Prompt results must be text or NULL"
        )

    def __call__(self, table: pa.Table) -> pa.Table:
        columns = [table.column(name).to_pylist() for name in self._message_columns]
        row_count = table.num_rows
        results: list[str | None] = [None] * row_count
        row_messages: dict[int, tuple[Any, ...]] = {}
        for index in range(row_count):
            try:
                messages = self._build_row_messages(columns, index)
            except Exception as exc:
                if self._on_error == "raise":
                    raise
                _log_substituted_failure(exc, on_error=self._on_error)
                continue
            if messages is not None:
                row_messages[index] = messages

        if not row_messages:
            return pa.table({self._output_column: pa.array(results, type=pa.string())})

        capability_error: ProviderCapabilityError | None = None
        provider_error: Exception | None = None
        try:
            prompter = self._ensure_prompter()
        except _MissingAsyncRuntimeError:
            raise
        except ProviderCapabilityError as exc:
            capability_error = exc
        except Exception as exc:
            provider_error = exc
        if capability_error is not None:
            raise _safe_provider_capability_error(capability_error) from None
        if provider_error is not None:
            provider, model = _descriptor_identity(self._descriptor)
            raise _safe_provider_execution_error(provider, model, "Prompt initialization", provider_error) from None

        max_retries = self._max_retries
        on_error = self._on_error

        async def invoke(index: int) -> str | None:
            capability_error: ProviderCapabilityError | None = None
            provider_error: Exception | None = None
            try:
                result = await _retry_call_async(
                    prompter.prompt,
                    row_messages[index],
                    max_retries=max_retries,
                    on_error=on_error,
                    default=_SUBSTITUTED_FAILURE,
                    on_awaitable=self._mark_loop_bound,
                )
            except ProviderCapabilityError as exc:
                capability_error = exc
            except (_ProviderResultError, OutputValidationError, RawResponseSerializationError):
                raise
            except Exception as exc:
                provider_error = exc
            if capability_error is not None:
                raise _safe_provider_capability_error(capability_error) from None
            if provider_error is not None:
                provider, model = _descriptor_identity(self._descriptor)
                raise _safe_provider_execution_error(provider, model, "Prompt execution", provider_error) from None
            if result is _SUBSTITUTED_FAILURE:
                # The retry helper already logged this substitution before
                # returning the sentinel; surface it as a NULL row.
                return None
            try:
                return self._validate_result(result)
            except (_ProviderResultError, OutputValidationError, RawResponseSerializationError) as exc:
                if on_error == "raise":
                    raise
                _log_substituted_failure(exc, on_error=on_error)
                return None

        async def run_all(indices: list[int]) -> list[str | None]:
            if self._max_concurrency_per_actor is None:
                calls = [invoke(index) for index in indices]
            else:
                semaphore = asyncio.Semaphore(self._max_concurrency_per_actor)

                async def limited(index: int) -> str | None:
                    async with semaphore:
                        return await invoke(index)

                calls = [limited(index) for index in indices]

            tasks = [asyncio.create_task(call) for call in calls]
            try:
                return await asyncio.gather(*tasks)
            except BaseException:
                # ``gather`` propagates the first failure but leaves sibling
                # requests running.  Drain their cancellation before this
                # batch returns so a failed row cannot leak work into the
                # next batch or outlive the executor-owned event loop.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        indices = list(row_messages)
        prompt_results = self._require_run_async()(run_all(indices))
        for index, result in zip(indices, prompt_results, strict=True):
            results[index] = result
        return pa.table({self._output_column: pa.array(results, type=pa.string())})


class _ValidateStructuredOutputBatch:
    """Validate native structured decoding before exposing a typed value."""

    def __init__(
        self,
        return_format: StructuredOutputSpec,
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
                results.append(self._return_format.validate_json(raw_text))
            except OutputValidationError as exc:
                if self._on_error == "raise":
                    raise
                _log_substituted_failure(exc, on_error=self._on_error)
                results.append(None)
        return pa.table({self._output_column: pa.array(results, type=pa.string())})


def _build_ai_batch_expression(
    wrapper: Any,
    *,
    inputs: Mapping[str, Any],
    output_column: str,
    output_type: str,
    udf_opts: UDFOptions,
    name: str,
    execution_backend: str | None = None,
    force_actor: bool = True,
) -> Expression:
    actor_backend = execution_backend in {"subprocess_actor", "ray_actor"}
    if execution_backend is None and force_actor:
        actor_callable = _adapt_batch_wrapper_for_backend(wrapper, "subprocess_actor", force_actor=True)
        return _build_actor_map_batches_expression(
            actor_callable,
            name=name,
            inputs={name: as_expression(expression) for name, expression in inputs.items()},
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
        inputs={name: as_expression(expression) for name, expression in inputs.items()},
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


def _validated_embed_text(text: Any) -> Expression:
    """Add a bind-only VARCHAR guard that is removed during planning."""
    return vane.FunctionExpression("__vane_ai_embed", text)


def _star_excluding_existing_output_column(rel: Relation, output_column: str) -> Expression:
    existing_output_column = next(
        (column for column in rel.columns if column.casefold() == output_column.casefold()),
        None,
    )
    if existing_output_column is None:
        return vane.StarExpression()
    return vane.StarExpression(exclude=[existing_output_column])


def _embed_expression(
    text: Any,
    *,
    provider: Any,
    model: str | None,
    dimensions: int | None,
    on_error: _OnError,
    options: Mapping[str, Any],
) -> Expression:
    if not is_expression(text):
        raise TypeError("vane.ai.embed expression API requires a text Expression")
    descriptor, resolved_dimensions, udf_opts, normalize, _, _, _ = _prepare_embed_call(
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
        inputs={"text": _validated_embed_text(text)},
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
    on_error: _OnError,
    output_column: str,
    options: Mapping[str, Any],
) -> Relation:
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
        inputs={"text": _validated_embed_text(text)},
        output_column=output_column,
        output_type=output_type,
        udf_opts=udf_opts,
        name="ai_embed",
        execution_backend=execution_backend,
    ).cast(output_type)
    star = _star_excluding_existing_output_column(rel, output_column)
    return rel.select(star, expression.alias(output_column))


_EMBED_ARGUMENT_UNSET: Any = object()


class _DefaultEmbedOutputColumn(str):
    """Distinguish an omitted Relation-only keyword from an explicit value."""


_EMBED_OUTPUT_COLUMN_DEFAULT = _DefaultEmbedOutputColumn("embedding")


@overload
def embed(
    text: Expression,
    /,
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
    /,
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
# prompt
# ---------------------------------------------------------------------------


def _prompt_provider_family(provider: Any) -> str | None:
    module = type(provider).__module__
    families = {
        "vane.ai.providers.openai": "openai",
        "vane.ai.providers.anthropic": "anthropic",
        "vane.ai.providers.google": "google",
        "vane.ai.providers.vllm": "vllm",
        "vane.ai.providers.sglang": "sglang",
    }
    if module in families:
        return families[module]
    name = getattr(provider, "name", None)
    if isinstance(name, str) and name.casefold() in {"openai", "anthropic", "google", "vllm", "sglang"}:
        return name.casefold()
    return None


def _prepare_prompt_call(
    provider: Any,
    model: str | None,
    return_format: Any,
    system_message: str | None,
    return_raw_response: bool,
    on_error: str,
    options: Mapping[str, Any],
    *,
    relation: bool,
) -> tuple[Any, UDFOptions, str | None, StructuredOutputSpec | None]:
    """Resolve one Prompt call without endpoint or model I/O."""

    if provider is None:
        raise TypeError("provider must be a provider name or Provider object, not None")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("model must be a non-empty string or None")
    if system_message is not None and not isinstance(system_message, str):
        raise TypeError("system_message must be a string or None")
    if not isinstance(return_raw_response, bool):
        raise TypeError("return_raw_response must be a bool")
    on_error = _validate_on_error(on_error)
    structured_output = compile_return_format(return_format)

    resolved_provider = _resolve_provider(provider)
    if not isinstance(resolved_provider, Provider):
        raise TypeError("provider must be a provider name or Provider object")
    family = _prompt_provider_family(resolved_provider)
    prepared = normalize_prompt_options(family, options, relation=relation)

    execution_backend = prepared.pop("execution_backend", None)
    batch_size = prepared.pop("batch_size", None)
    actor_number = prepared.pop("actor_number", None)
    max_concurrency = prepared.pop("max_concurrency_per_actor", None)
    max_retries = prepared.pop("max_retries", None)

    if family in {"vllm", "sglang"}:
        if return_raw_response:
            raise ValueError(f"Provider {family!r} does not support return_raw_response")
        prepared["actor_number"] = actor_number if actor_number is not None else 1
        prepared["batch_size"] = batch_size if batch_size is not None else 128
        prepared["max_retries"] = max_retries if max_retries is not None else 0

    try:
        descriptor = resolved_provider.get_prompter(
            model=model,
            system_message=system_message,
            return_format=structured_output.schema if structured_output is not None else None,
            return_raw_response=return_raw_response,
            options=prepared,
        )
    except NotImplementedError as exc:
        provider_name = getattr(resolved_provider, "name", type(resolved_provider).__name__)
        raise ValueError(f"Provider {provider_name!r} is not a prompt provider") from exc

    if isinstance(descriptor, NativeInferencePlan):
        descriptor.on_error = on_error
        return descriptor, UDFOptions(on_error=on_error), execution_backend, structured_output
    if isinstance(descriptor, NativePrompterPlan):
        return descriptor, UDFOptions(on_error=on_error), execution_backend, structured_output

    if (
        family == "openai"
        and structured_output is not None
        and type(descriptor).__module__ == "vane.ai.providers.openai"
        and getattr(descriptor, "supports_strict_structured_outputs", lambda: False)()
    ):
        structured_output.validate_openai_gpt_contract(descriptor.get_model())

    descriptor_resources = descriptor.get_udf_options()
    udf_options = UDFOptions(
        actor_number=actor_number if actor_number is not None else 1,
        num_gpus=descriptor_resources.num_gpus,
        max_retries=max_retries if max_retries is not None else 3,
        on_error=on_error,
        batch_size=batch_size if batch_size is not None else 32,
        max_concurrency_per_actor=(
            max_concurrency
            if max_concurrency is not None
            else 32
            if family == "openai"
            else 16
            if family in {"anthropic", "google"}
            else None
        ),
    )
    return descriptor, udf_options, execution_backend, structured_output


def _normalize_prompt_messages(messages: Any) -> tuple[list[Expression], bool]:
    if is_expression(messages):
        return [messages], True
    if not isinstance(messages, list):
        raise TypeError("Prompt messages must be an Expression or a non-empty list of Expressions")
    if not messages:
        raise ValueError("Prompt messages list cannot be empty")
    for index, expression in enumerate(messages):
        if not is_expression(expression):
            raise TypeError(f"Prompt messages[{index}] must be an Expression")
    return list(messages), False


def _prompt_relation_types(rel: Relation, messages: list[Expression]) -> list[str]:
    types = [str(value).upper() for value in rel.select(*messages).types]
    allowed = {"VARCHAR", "BLOB", "BLOB[]", "FILE", "FILE[]"}
    for index, value in enumerate(types):
        if value not in allowed:
            raise TypeError(
                f"Prompt messages[{index}] must have type VARCHAR, BLOB, BLOB[], FILE, or FILE[]; got {value}"
            )
    return types


def _validated_prompt_message(message: Any, *, media_policy: Literal["all", "file", "text"] = "all") -> Expression:
    """Add a bind-only Prompt guard; FILE inputs lower to native materialization."""
    if media_policy == "text":
        # The hidden BOOLEAN overload uses TRUE for strict text-only native plans.
        return vane.FunctionExpression("__vane_ai_prompt", message, vane.ConstantExpression(True))
    if media_policy == "file":
        # FALSE permits text plus FILE while continuing to reject eager BLOB input.
        return vane.FunctionExpression("__vane_ai_prompt", message, vane.ConstantExpression(False))
    return vane.FunctionExpression("__vane_ai_prompt", message)


def _build_native_vllm_expression(messages: list[Any], descriptor: NativeInferencePlan) -> vane.Expression:
    """Build the native row-preserving ``vllm()`` expression."""
    message_expressions = [as_expression(message) for message in messages]
    if len(message_expressions) == 1:
        messages_expr = message_expressions[0]
    else:
        any_present = message_expressions[0].isnotnull()
        for expression in message_expressions[1:]:
            any_present = any_present | expression.isnotnull()
        combined = vane.FunctionExpression("concat", *message_expressions)
        messages_expr = vane.CaseExpression(any_present, combined)
    if descriptor.system_message:
        # Unlike concat(), the || operator propagates NULL inputs.
        messages_expr = vane.FunctionExpression(
            "||",
            vane.ConstantExpression(f"{descriptor.system_message}\n\n"),
            messages_expr,
        )

    options_argument = _build_native_vllm_options_argument(
        descriptor.build_physical_vllm_options(), engine=descriptor.get_engine()
    )

    return vane.FunctionExpression(
        "vllm",
        messages_expr,
        vane.ConstantExpression(descriptor.model_name),
        vane.ConstantExpression(options_argument),
    )


def _cast_structured_prompt_output(
    expression: Expression,
    structured_output: StructuredOutputSpec,
) -> Expression:
    """Decode validated JSON text into the public structured type."""
    return expression.cast("JSON").cast(structured_output.duckdb_type)


def _prompt_relation(
    rel: Any,
    messages: Any,
    *,
    provider: Any,
    model: str | None = None,
    return_format: Any = None,
    system_message: str | None = None,
    return_raw_response: bool = False,
    on_error: _OnError = "raise",
    output_column: str = "response",
    options: Mapping[str, Any],
) -> Relation:
    if not _is_relation_like(rel):
        raise TypeError("vane.ai.prompt relation API requires a Relation")
    message_expressions, single_message = _normalize_prompt_messages(messages)
    message_types = _prompt_relation_types(rel, message_expressions)
    if not isinstance(output_column, str) or not output_column.strip():
        raise ValueError("output_column must be a non-empty string")

    descriptor, udf_options, execution_backend, structured_output = _prepare_prompt_call(
        provider,
        model,
        return_format,
        system_message,
        return_raw_response,
        on_error,
        options,
        relation=True,
    )

    if isinstance(descriptor, NativeInferencePlan):
        if any(value != "VARCHAR" for value in message_types):
            raise ValueError(f"Provider {descriptor.get_provider()!r} does not support Prompt media inputs")
        expression = _build_native_vllm_expression(message_expressions, descriptor)
        if structured_output is not None:
            input_column = "__vane_vllm_response"
            validator = _ValidateStructuredOutputBatch(
                structured_output,
                input_column,
                output_column,
                udf_options.on_error,
            )
            expression = _cast_structured_prompt_output(
                _build_map_batches_expression(
                    validator.__call__,
                    name="validate_vllm_structured_output",
                    inputs={input_column: expression},
                    schema={output_column: "VARCHAR"},
                    batch_size=None,
                    row_preserving=True,
                    gpus=0,
                ),
                structured_output,
            )
        expression = expression.alias(output_column)
        star = _star_excluding_existing_output_column(rel, output_column)
        return rel.select(star, expression)
    if isinstance(descriptor, NativePrompterPlan):
        raise ValueError(f"Unsupported native prompt plan {type(descriptor).__name__}")
    supports_media_inputs = bool(descriptor.supports_image_inputs())
    if not supports_media_inputs and any(value in {"BLOB", "BLOB[]"} for value in message_types):
        raise ValueError(
            f"Provider {descriptor.get_provider()!r} model {descriptor.get_model()!r} "
            "does not support Prompt image inputs"
        )

    input_names = [f"message_{index}" for index in range(len(message_expressions))]
    media_policy: Literal["all", "file", "text"] = "all" if supports_media_inputs else "file"
    validated_messages = [
        _validated_prompt_message(message, media_policy=media_policy) for message in message_expressions
    ]
    wrapper = _PromptBatch(
        descriptor,
        input_names,
        output_column,
        udf_options.max_concurrency_per_actor,
        single_message=single_message,
        return_format=None if return_raw_response else structured_output,
        return_raw_response=return_raw_response,
        max_retries=udf_options.max_retries,
        on_error=udf_options.on_error,
        supports_media_inputs=supports_media_inputs,
    )
    expression = _build_ai_batch_expression(
        wrapper,
        inputs=dict(zip(input_names, validated_messages, strict=True)),
        output_column=output_column,
        output_type="VARCHAR",
        udf_opts=udf_options,
        name="ai_prompt",
        execution_backend=execution_backend,
    )
    if structured_output is not None and not return_raw_response:
        expression = _cast_structured_prompt_output(expression, structured_output)
    else:
        expression = expression.cast("VARCHAR")
    star = _star_excluding_existing_output_column(rel, output_column)
    return rel.select(star, expression.alias(output_column))


def _prompt_expression(
    messages: Any,
    *,
    provider: Any,
    model: str | None = None,
    return_format: Any = None,
    system_message: str | None = None,
    return_raw_response: bool = False,
    on_error: _OnError = "raise",
    options: Mapping[str, Any],
) -> Expression:
    message_expressions, single_message = _normalize_prompt_messages(messages)
    descriptor, udf_options, _, structured_output = _prepare_prompt_call(
        provider,
        model,
        return_format,
        system_message,
        return_raw_response,
        on_error,
        options,
        relation=False,
    )

    if isinstance(descriptor, NativeInferencePlan):
        validated_messages = [
            _validated_prompt_message(message, media_policy="text") for message in message_expressions
        ]
        expression = _build_native_vllm_expression(validated_messages, descriptor)
        if structured_output is None:
            return expression
        input_column = "__vane_vllm_response"
        validator = _ValidateStructuredOutputBatch(
            structured_output,
            input_column,
            "response",
            udf_options.on_error,
        )
        return _cast_structured_prompt_output(
            _build_map_batches_expression(
                validator.__call__,
                name="validate_vllm_structured_output",
                inputs={input_column: expression},
                schema={"response": "VARCHAR"},
                batch_size=None,
                row_preserving=True,
                gpus=0,
            ),
            structured_output,
        )
    if isinstance(descriptor, NativePrompterPlan):
        raise ValueError(f"Unsupported native prompt plan {type(descriptor).__name__}")

    input_names = [f"message_{index}" for index in range(len(message_expressions))]
    supports_media_inputs = bool(descriptor.supports_image_inputs())
    media_policy: Literal["all", "file", "text"] = "all" if supports_media_inputs else "file"
    validated_messages = [
        _validated_prompt_message(message, media_policy=media_policy) for message in message_expressions
    ]
    wrapper = _PromptBatch(
        descriptor,
        input_names,
        "response",
        udf_options.max_concurrency_per_actor,
        single_message=single_message,
        return_format=None if return_raw_response else structured_output,
        return_raw_response=return_raw_response,
        max_retries=udf_options.max_retries,
        on_error=udf_options.on_error,
        supports_media_inputs=supports_media_inputs,
    )
    expression = _build_ai_batch_expression(
        wrapper,
        inputs=dict(zip(input_names, validated_messages, strict=True)),
        output_column="response",
        output_type="VARCHAR",
        udf_opts=udf_options,
        name="ai_prompt",
    )
    if structured_output is not None and not return_raw_response:
        return _cast_structured_prompt_output(expression, structured_output)
    return expression.cast("VARCHAR")


def _is_relation_like(value: Any) -> TypeGuard[Relation]:
    return hasattr(value, "map_batches") and hasattr(value, "select")


_PROMPT_ARGUMENT_UNSET: Any = object()


class _DefaultPromptOutputColumn(str):
    """Distinguish an omitted Relation-only keyword from an explicit value."""


_PROMPT_OUTPUT_COLUMN_DEFAULT = _DefaultPromptOutputColumn("response")


@overload
def prompt(
    messages: Expression | list[Expression],
    /,
    *,
    return_format: type[BaseModel] | JSONSchema | None = None,
    system_message: str | None = None,
    provider: str | Provider = "openai",
    model: str | None = None,
    return_raw_response: bool = False,
    on_error: Literal["raise", "ignore"] = "raise",
    **options: Unpack[PromptOptions],
) -> Expression: ...


@overload
def prompt(
    *,
    messages: Expression | list[Expression],
    return_format: type[BaseModel] | JSONSchema | None = None,
    system_message: str | None = None,
    provider: str | Provider = "openai",
    model: str | None = None,
    return_raw_response: bool = False,
    on_error: Literal["raise", "ignore"] = "raise",
    **options: Unpack[PromptOptions],
) -> Expression: ...


@overload
def prompt(
    rel: Relation,
    /,
    messages: Expression | list[Expression],
    *,
    return_format: type[BaseModel] | JSONSchema | None = None,
    system_message: str | None = None,
    provider: str | Provider = "openai",
    model: str | None = None,
    return_raw_response: bool = False,
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: str = "response",
    **options: Unpack[PromptOptions],
) -> Relation: ...


@overload
def prompt(
    *,
    rel: Relation,
    messages: Expression | list[Expression],
    return_format: type[BaseModel] | JSONSchema | None = None,
    system_message: str | None = None,
    provider: str | Provider = "openai",
    model: str | None = None,
    return_raw_response: bool = False,
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: str = "response",
    **options: Unpack[PromptOptions],
) -> Relation: ...


def prompt(
    first: Expression | list[Expression] | Relation = _PROMPT_ARGUMENT_UNSET,
    /,
    messages: Expression | list[Expression] = _PROMPT_ARGUMENT_UNSET,
    *,
    rel: Relation = _PROMPT_ARGUMENT_UNSET,
    return_format: type[BaseModel] | JSONSchema | None = None,
    system_message: str | None = None,
    provider: str | Provider = "openai",
    model: str | None = None,
    return_raw_response: bool = False,
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: str = _PROMPT_OUTPUT_COLUMN_DEFAULT,
    **options: Unpack[PromptOptions],
) -> Expression | Relation:
    """Prompt over ordered VARCHAR, BLOB, BLOB[], FILE, or FILE[] Expressions.

    ``return_format`` accepts a Pydantic model class or the portable JSON
    Schema subset and exposes a native STRUCT. ``return_raw_response=True``
    instead exposes the provider SDK response body as JSON VARCHAR while
    still using any supplied schema to constrain generation.
    """
    if first is not _PROMPT_ARGUMENT_UNSET and rel is not _PROMPT_ARGUMENT_UNSET:
        raise TypeError("vane.ai.prompt received both first and rel; pass only one relation argument")

    relation = rel if rel is not _PROMPT_ARGUMENT_UNSET else first
    if relation is not _PROMPT_ARGUMENT_UNSET and _is_relation_like(relation):
        if messages is _PROMPT_ARGUMENT_UNSET:
            raise TypeError("vane.ai.prompt relation API requires messages")
        resolved_output = "response" if output_column is _PROMPT_OUTPUT_COLUMN_DEFAULT else output_column
        return _prompt_relation(
            relation,
            messages,
            provider=provider,
            model=model,
            return_format=return_format,
            system_message=system_message,
            return_raw_response=return_raw_response,
            on_error=on_error,
            output_column=resolved_output,
            options=options,
        )

    if rel is not _PROMPT_ARGUMENT_UNSET:
        raise TypeError("vane.ai.prompt rel= must be a Relation")
    if first is not _PROMPT_ARGUMENT_UNSET and messages is not _PROMPT_ARGUMENT_UNSET:
        raise TypeError("vane.ai.prompt expression API accepts one messages argument")
    expression_messages = messages if first is _PROMPT_ARGUMENT_UNSET else first
    if expression_messages is _PROMPT_ARGUMENT_UNSET:
        raise TypeError("vane.ai.prompt requires messages or a Relation plus messages")
    if output_column is not _PROMPT_OUTPUT_COLUMN_DEFAULT:
        raise TypeError("vane.ai.prompt expression API does not accept output_column; use .alias(...)")
    return _prompt_expression(
        expression_messages,
        provider=provider,
        model=model,
        return_format=return_format,
        system_message=system_message,
        return_raw_response=return_raw_response,
        on_error=on_error,
        options=options,
    )
