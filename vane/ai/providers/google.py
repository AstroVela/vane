# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Google Generative AI (Gemini) provider for Vane AI.

Supports text embedding via ``embed_content`` and basic text/image Prompt
calls via ``generate_content``.

Prompt calls must name a model, either per call (``model=...``) or through
``GoogleProvider(prompt_model=...)``. Embed calls use the provider's pinned
default unless overridden per call or through
``GoogleProvider(embedding_model=...)``.

Requires::

    pip install 'vane-ai[google]'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from vane.ai._redaction import unwrap_sensitive_options, wrap_sensitive_options
from vane.ai._schema import serialize_raw_response
from vane.ai.options import validate_prompt_options
from vane.ai.protocols import PrompterDescriptor, TextEmbedderDescriptor
from vane.ai.provider import Provider, ProviderCapabilityError, _ProviderResultError
from vane.ai.typing import EmbeddingDimensions, UDFOptions

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vane.ai.protocols import Prompter, TextEmbedder
    from vane.ai.typing import Embedding, Options


def _guess_media_type(data: bytes) -> str | None:
    """Guess image MIME type from magic bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:4] == b"GIF8":
        return "image/gif"
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _raise_retry_after_on_google_error(exc: Exception) -> None:
    """Re-raise *exc* as a :class:`RetryAfterError` when the Google API
    returns 429 (rate-limited) or 503 (service unavailable).

    Parses the ``Retry-After`` header if present; otherwise falls back to
    a 5-second default wait.
    """
    from vane.ai.functions import RetryAfterError

    code = getattr(exc, "code", None)
    if code not in (429, 503):
        return  # not retryable

    # Try to extract Retry-After from the response headers
    response = getattr(exc, "response", None)
    retry_after: float | None = None
    if response is not None:
        headers = getattr(response, "headers", None) or {}
        raw = headers.get("Retry-After") or headers.get("retry-after")
        if raw is not None:
            try:
                retry_after = float(raw)
            except (TypeError, ValueError):
                pass
    if retry_after is None:
        retry_after = 5.0  # default wait for 429/503

    raise RetryAfterError(retry_after=retry_after, original=exc) from exc


_EMBED_CAPABILITY_FIELD_SUFFIXES = (
    "model",
    "dimensions",
    "outputdimensionality",
    "tasktype",
)


def _google_error_fields(value: Any) -> list[str]:
    fields: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in {"field", "param"} and isinstance(item, str):
                fields.append(item)
            fields.extend(_google_error_fields(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            fields.extend(_google_error_fields(item))
    return fields


def _is_embedding_capability_error(exc: Exception) -> bool:
    """Classify only structured endpoint/model embedding failures."""
    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "").strip().upper()
    if code in {404, 405, 501} or status in {"NOT_FOUND", "UNIMPLEMENTED"}:
        return True
    if code not in {400, 422}:
        return False

    fields = _google_error_fields(getattr(exc, "details", None))
    direct_field = getattr(exc, "field", None) or getattr(exc, "param", None)
    if isinstance(direct_field, str):
        fields.append(direct_field)
    for error_field in fields:
        normalized = "".join(character for character in error_field.casefold() if character.isalnum())
        if normalized.endswith(_EMBED_CAPABILITY_FIELD_SUFFIXES):
            return True
    return False


def _is_prompt_capability_error(exc: Exception) -> bool:
    """Classify endpoint/model failures that are only knowable at runtime."""
    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "").strip().upper()
    if code in {404, 405, 501} or status in {"NOT_FOUND", "UNIMPLEMENTED"}:
        return True
    if code not in {400, 422}:
        return False
    fields = _google_error_fields(getattr(exc, "details", None))
    direct_field = getattr(exc, "field", None) or getattr(exc, "param", None)
    if isinstance(direct_field, str):
        fields.append(direct_field)
    return any("model" in field.casefold() or "endpoint" in field.casefold() for field in fields)


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

# Default output dimensionality per known embedding model, from the Gemini
# embeddings guide (https://ai.google.dev/gemini-api/docs/embeddings). Only
# models with trusted metadata belong here; any other model requires the
# caller to supply ``dimensions`` explicitly.
_EMBEDDING_DIMS: dict[str, int] = {
    "gemini-embedding-001": 3072,
    "gemini-embedding-2": 3072,
}

# Documented ``output_dimensionality`` bounds per known embedding model.
_EMBEDDING_DIM_RANGE: dict[str, tuple[int, int]] = {
    "gemini-embedding-001": (128, 3072),
    "gemini-embedding-2": (128, 3072),
}

# Per-request input cap for Gemini embedding requests. The embeddings guide
# does not publish a batch-size number, but the ``batchEmbedContents``
# endpoint (which multi-input ``embed_content`` calls use) rejects larger
# batches with "BatchEmbedContentsRequest.requests: at most 100 requests can
# be in one batch", so 100 is the server-enforced limit.
_EMBED_BATCH_LIMIT = 100
_EMBED_REQUEST_OPTIONS = frozenset({"task_type", "title"})

# Request options rejected per model before dispatch. Gemini 3.6 Flash and
# 3.5 Flash-Lite deprecate the classic sampling parameters: the API ignores
# them today and returns HTTP 400 in future model generations
# (https://ai.google.dev/gemini-api/docs/latest-model).
_MODEL_UNSUPPORTED_OPTIONS: dict[str, frozenset[str]] = {
    "gemini-3.6-flash": frozenset({"temperature", "top_p", "top_k"}),
    "gemini-3.5-flash-lite": frozenset({"temperature", "top_p", "top_k"}),
}


def _canonical_model_id(model_name: str) -> str:
    """Strip the Gemini API ``models/`` resource prefix for local lookups.

    The Google Gen AI SDK accepts both ``gemini-3.6-flash`` and
    ``models/gemini-3.6-flash``; local metadata and capability tables key on
    the bare ID, while the caller-provided value is sent to the SDK verbatim.
    """
    return model_name.removeprefix("models/")


def _validate_prompt_options(model_name: str, options: Mapping[str, Any]) -> None:
    """Reject request options the selected model is documented not to support.

    Raises:
        ValueError: If ``options`` contains keys listed for ``model_name``
            in :data:`_MODEL_UNSUPPORTED_OPTIONS`.
    """
    unsupported = _MODEL_UNSUPPORTED_OPTIONS.get(_canonical_model_id(model_name), frozenset())
    offending = sorted(unsupported.intersection(options))
    if offending:
        raise ValueError(
            f"Google model {model_name!r} does not support options {offending}: "
            "the Gemini API ignores these sampling parameters and rejects them "
            "in future model generations. Remove them from the request."
        )


def _validate_embedding_dimensions(model_name: str, dimensions: int | None) -> None:
    """Validate the embedding-dimensions configuration for ``model_name``.

    Raises:
        ValueError: If ``dimensions`` is omitted for a model without trusted
            metadata in :data:`_EMBEDDING_DIMS`, is not a positive integer,
            or falls outside the documented range for a known model.
    """
    canonical = _canonical_model_id(model_name)
    if dimensions is None:
        if canonical not in _EMBEDDING_DIMS:
            raise ValueError(
                f"Cannot derive embedding dimensions for Google model {model_name!r}. "
                "Pass dimensions=... or configure GoogleProvider(embedding_dimensions=...)."
            )
        return
    if isinstance(dimensions, bool) or not isinstance(dimensions, int):
        raise ValueError(f"Embedding dimensions must be a positive integer, got {dimensions!r}.")
    if dimensions < 1:
        raise ValueError(f"Embedding dimensions must be a positive integer, got {dimensions!r}.")
    bounds = _EMBEDDING_DIM_RANGE.get(canonical)
    if bounds is not None and not bounds[0] <= dimensions <= bounds[1]:
        raise ValueError(
            f"Google model {model_name!r} supports output dimensionality "
            f"between {bounds[0]} and {bounds[1]}, got {dimensions}."
        )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class GoogleProvider(Provider):
    """Provider backed by Google Generative AI (Gemini).

    Embed calls default to :attr:`DEFAULT_TEXT_EMBEDDER`. Prompt calls still
    require a model per call or through ``prompt_model``. Call-site arguments
    always win over constructor configuration.

    Args:
        name: Optional display-name override (default ``"google"``).
        api_key: Google API key. Prefer the ``GOOGLE_API_KEY`` environment
            variable over passing keys in code.
        prompt_model: Model used by ``get_prompter`` when the call does not
            pass ``model=...``.
        embedding_model: Model used by ``get_text_embedder`` when the call
            does not pass ``model=...``.
        embedding_dimensions: Embedding output dimensionality used when the
            call does not pass ``dimensions=...``. Required (here or per
            call) for embedding models without trusted dimension metadata.

    All parameters are named; a mistyped keyword raises :class:`TypeError`
    instead of silently leaking into API request options.
    """

    DEFAULT_TEXT_EMBEDDER: ClassVar[str] = "gemini-embedding-2"
    _CLIENT_KEYS: ClassVar[frozenset[str]] = frozenset({"api_key"})

    def __init__(
        self,
        name: str | None = None,
        *,
        api_key: str | None = None,
        prompt_model: str | None = None,
        embedding_model: str | None = None,
        embedding_dimensions: int | None = None,
    ):
        self._name = name or "google"
        self._prompt_model = prompt_model
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._options: dict[str, Any] = {}
        if api_key is not None:
            self._options["api_key"] = api_key

    @property
    def name(self) -> str:
        return self._name

    def _split_options(self, options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        merged = {**self._options, **options}
        provider_options = {k: v for k, v in merged.items() if k in self._CLIENT_KEYS}
        request_options = {k: v for k, v in merged.items() if k not in self._CLIENT_KEYS}
        return provider_options, request_options

    def get_text_embedder(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        **options: Any,
    ) -> TextEmbedderDescriptor:
        """Build an embedder descriptor for the selected or default model.

        Raises:
            ValueError: If dimensions cannot be resolved for the selected
                model or the model/option combination is invalid.
        """
        provider_options, embed_options = self._split_options(options)
        unknown = sorted(set(embed_options) - _EMBED_REQUEST_OPTIONS)
        if unknown:
            raise TypeError(f"Unsupported Google Embed option(s): {', '.join(unknown)}")
        model_name = model if model is not None else self._embedding_model
        if model_name is None:
            model_name = self.DEFAULT_TEXT_EMBEDDER
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                f"Google embedding model must be a non-empty string, got {model_name!r}. "
                "Pass model=... or configure GoogleProvider(embedding_model=...)."
            )
        if dimensions is None:
            dimensions = self._embedding_dimensions
        return GoogleTextEmbedderDescriptor(
            model_name=model_name,
            provider_name=self._name,
            provider_options=provider_options,
            dimensions=dimensions,
            embed_options=embed_options,
        )

    def get_prompter(
        self,
        model: str | None = None,
        system_message: str | None = None,
        return_format: dict[str, Any] | None = None,
        return_raw_response: bool = False,
        **options: Any,
    ) -> PrompterDescriptor:
        """Build a prompter descriptor for an explicitly selected model.

        Raises:
            ValueError: If neither ``model=...`` nor the provider's
                ``prompt_model`` is configured, or if the selected model
                does not support one of the requested options.
        """
        provider_options, prompt_options = self._split_options(options)
        validate_prompt_options("google", prompt_options, relation=False)
        model_name = model if model is not None else self._prompt_model
        if model_name is None:
            raise ValueError(
                "No prompt model configured for the Google provider. "
                "Pass model=... or configure GoogleProvider(prompt_model=...)."
            )
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                f"Google prompt model must be a non-empty string, got {model_name!r}. "
                "Pass model=... or configure GoogleProvider(prompt_model=...)."
            )
        return GooglePrompterDescriptor(
            model_name=model_name,
            provider_name=self._name,
            provider_options=provider_options,
            system_message=system_message,
            return_format=return_format,
            return_raw_response=return_raw_response,
            prompt_options=prompt_options,
        )


# ---------------------------------------------------------------------------
# Text Embedding
# ---------------------------------------------------------------------------


@dataclass
class GoogleTextEmbedderDescriptor(TextEmbedderDescriptor):
    """Serializable factory for a Google Generative AI text embedder.

    ``model_name`` is required. ``dimensions`` is required unless the model
    has trusted metadata in :data:`_EMBEDDING_DIMS`; both are validated at
    construction time, before anything ships to workers.

    The default UDF ``batch_size`` matches the per-request input cap
    (:data:`_EMBED_BATCH_LIMIT`); the embedder additionally chunks
    oversized batches as defense in depth.
    """

    model_name: str
    provider_name: str = "google"
    provider_options: dict[str, Any] = field(default_factory=dict)
    dimensions: int | None = None
    embed_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.embed_options) - _EMBED_REQUEST_OPTIONS)
        if unknown:
            raise TypeError(f"Unsupported Google Embed option(s): {', '.join(unknown)}")
        _validate_embedding_dimensions(self.model_name, self.dimensions)
        if _canonical_model_id(self.model_name) == "gemini-embedding-2":
            unsupported = sorted(
                option for option in ("task_type", "title") if self.embed_options.get(option) is not None
            )
            if unsupported:
                raise ValueError(
                    f"Google model {self.model_name!r} does not support embedding option(s): {', '.join(unsupported)}"
                )
        task_type = self.embed_options.get("task_type")
        title = self.embed_options.get("title")
        if title is not None and task_type != "RETRIEVAL_DOCUMENT":
            raise ValueError("Google embedding title is only valid with task_type='RETRIEVAL_DOCUMENT'")
        self.provider_options = wrap_sensitive_options(self.provider_options)
        self.embed_options = wrap_sensitive_options(self.embed_options)

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> Options:
        return dict(self.embed_options)

    def get_dimensions(self) -> EmbeddingDimensions:
        if self.dimensions is not None:
            return EmbeddingDimensions(size=self.dimensions)
        return EmbeddingDimensions(size=_EMBEDDING_DIMS[_canonical_model_id(self.model_name)])

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(
            batch_size=_EMBED_BATCH_LIMIT,
            max_retries=3,
            on_error="raise",
            actor_number=None,
            num_gpus=0,
        )

    def is_async(self) -> bool:
        return True

    def instantiate(self) -> TextEmbedder:
        return GoogleTextEmbedder(
            provider_options=self.provider_options,
            model=self.model_name,
            dimensions=self.dimensions,
            provider_name=self.provider_name,
            **self.embed_options,
        )


class GoogleTextEmbedder:
    """Text embedder using Google Generative AI ``embed_content``."""

    def __init__(
        self,
        provider_options: dict[str, Any],
        model: str,
        dimensions: int | None = None,
        provider_name: str = "google",
        **options: Any,
    ):
        from google import genai

        # Restore plaintext credentials sealed by the descriptor; plain dicts
        # from direct callers pass through unchanged.
        provider_options = unwrap_sensitive_options(provider_options)
        options = unwrap_sensitive_options(options)
        api_key = provider_options.get("api_key")
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._provider_name = provider_name
        self._model = model
        self._dimensions = dimensions
        self._options = dict(options)

    async def aclose(self) -> None:
        """Release the SDK client's async connection pool on the owning loop."""
        await self._client.aio.aclose()

    async def embed_text(self, text: list[str]) -> list[Embedding]:
        """Embed *text*, chunking into per-request batches under the API cap.

        Each input is sent as its own ``types.Content``, so aggregating
        models such as ``gemini-embedding-2`` still return one embedding
        per input. Requests are capped at :data:`_EMBED_BATCH_LIMIT` inputs
        and results are concatenated in input order, so an oversized arrow
        batch can never produce a single oversized API call. The result
        always contains exactly one embedding per input.
        """
        from google.genai import types

        config = dict(self._options)
        if self._dimensions is not None:
            config["output_dimensionality"] = self._dimensions

        embeddings: list[Embedding] = []
        for start in range(0, len(text), _EMBED_BATCH_LIMIT):
            chunk = text[start : start + _EMBED_BATCH_LIMIT]
            kwargs: dict[str, Any] = {
                "model": self._model,
                "contents": [types.Content(parts=[types.Part.from_text(text=t)]) for t in chunk],
            }
            if config:
                kwargs["config"] = config
            try:
                result = await self._client.aio.models.embed_content(**kwargs)
            except Exception as exc:
                _raise_retry_after_on_google_error(exc)
                if _is_embedding_capability_error(exc):
                    raise ProviderCapabilityError(
                        getattr(self, "_provider_name", "google"),
                        self._model,
                        "embedding endpoint/model",
                        original_error=exc,
                    ) from exc
                raise
            chunk_embeddings = result.embeddings or []
            if len(chunk_embeddings) != len(chunk):
                raise _ProviderResultError(
                    f"Google embed_content returned {len(chunk_embeddings)} embeddings for {len(chunk)} inputs; "
                    "embedding calls must preserve row count and order"
                )
            embeddings.extend(np.array(e.values, dtype=np.float32) for e in chunk_embeddings)
        return embeddings


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


@dataclass
class GooglePrompterDescriptor(PrompterDescriptor):
    """Serializable factory for a basic Gemini text/image prompter."""

    model_name: str
    provider_name: str = "google"
    provider_options: dict[str, Any] = field(default_factory=dict)
    system_message: str | None = None
    return_format: dict[str, Any] | None = None
    return_raw_response: bool = False
    prompt_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("Google prompt model must be a non-empty string")
        validate_prompt_options("google", self.prompt_options, relation=False)
        _validate_prompt_options(self.model_name, self.prompt_options)
        self.provider_options = wrap_sensitive_options(self.provider_options)
        self.prompt_options = wrap_sensitive_options(self.prompt_options)

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> Options:
        return dict(self.prompt_options)

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(
            max_retries=self.prompt_options.get("max_retries", 3),
            actor_number=self.prompt_options.get("actor_number", 1),
            num_gpus=0,
            batch_size=self.prompt_options.get("batch_size", 32),
            max_concurrency_per_actor=self.prompt_options.get("max_concurrency_per_actor", 16),
        )

    def instantiate(self) -> Prompter:
        return GooglePrompter(
            provider_options=self.provider_options,
            provider_name=self.provider_name,
            model=self.model_name,
            system_message=self.system_message,
            return_format=self.return_format,
            return_raw_response=self.return_raw_response,
            **self.prompt_options,
        )


class GooglePrompter:
    """Async basic text/image prompter using Gemini ``generate_content``."""

    def __init__(
        self,
        provider_options: dict[str, Any],
        model: str,
        system_message: str | None = None,
        return_format: dict[str, Any] | None = None,
        return_raw_response: bool = False,
        provider_name: str = "google",
        **options: Any,
    ) -> None:
        from google import genai
        from google.genai import types

        # Restore plaintext credentials sealed by the descriptor; plain dicts
        # from direct callers pass through unchanged.
        provider_options = unwrap_sensitive_options(provider_options)
        options = unwrap_sensitive_options(options)
        api_key = provider_options.get("api_key")
        http_options = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))
        self._client = (
            genai.Client(api_key=api_key, http_options=http_options)
            if api_key
            else genai.Client(http_options=http_options)
        )
        self._provider_name = provider_name
        self._model = model
        self._system_message = system_message
        self._return_format = return_format
        self._return_raw_response = return_raw_response
        self._options = {
            k: v
            for k, v in options.items()
            if k in {"temperature", "top_p", "top_k", "max_output_tokens", "stop_sequences"} and v is not None
        }

    async def aclose(self) -> None:
        """Release the SDK client's async connection pool on the owning loop."""
        await self._client.aio.aclose()

    def _requested_capability(self) -> str:
        structured = getattr(self, "_return_format", None) is not None
        raw = getattr(self, "_return_raw_response", False)
        if structured and raw:
            return "structured Prompt generation with raw response body"
        if structured:
            return "structured Prompt generation"
        if raw:
            return "Prompt raw response body"
        return "basic Prompt text/image generation"

    # --- Multimodal message processing -----------------------------------

    def _process_message(self, msg: Any) -> Any:
        from google.genai import types

        if isinstance(msg, str):
            return types.Part.from_text(text=msg)
        if isinstance(msg, bytes):
            media_type = _guess_media_type(msg)
            if media_type is None:
                raise ValueError("Prompt image BLOB has an unsupported or unrecognized image format")
            return types.Part.from_bytes(data=msg, mime_type=media_type)
        raise TypeError(f"Unsupported Prompt content type: {type(msg).__name__}")

    # --- API call --------------------------------------------------------

    async def prompt(self, messages: tuple[Any, ...]) -> str | None:
        from google.genai import types

        contents = [types.Content(role="user", parts=[self._process_message(message) for message in messages])]
        config_kwargs: dict[str, Any] = {}
        if self._system_message is not None:
            config_kwargs["system_instruction"] = self._system_message
        for k in ("temperature", "top_p", "top_k", "max_output_tokens", "stop_sequences"):
            if k in self._options:
                config_kwargs[k] = self._options[k]
        return_format = getattr(self, "_return_format", None)
        if return_format is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = return_format

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            _raise_retry_after_on_google_error(exc)
            if _is_prompt_capability_error(exc):
                raise ProviderCapabilityError(
                    self._provider_name,
                    self._model,
                    self._requested_capability(),
                    original_error=exc,
                ) from exc
            raise

        # Record token usage metrics
        um = getattr(response, "usage_metadata", None)
        if um is not None:
            from vane.ai.metrics import record_token_metrics

            record_token_metrics(
                protocol="prompt",
                model=self._model,
                provider="google",
                input_tokens=getattr(um, "prompt_token_count", None),
                output_tokens=getattr(um, "candidates_token_count", None),
                total_tokens=getattr(um, "total_token_count", None),
            )

        if getattr(self, "_return_raw_response", False):
            return serialize_raw_response(
                response,
                exclude={"automatic_function_calling_history", "parsed", "sdk_http_response"},
            )

        text = response.text
        return text if text else None
