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
from vane.ai.options import (
    _require_prompt_int,
    _require_prompt_number,
    _validate_prompt_stop_sequences,
)
from vane.ai.options import (
    validate_embed_options as validate_closed_embed_options,
)
from vane.ai.protocols import PrompterDescriptor, TextEmbedderDescriptor
from vane.ai.provider import Provider, ProviderCapabilityError, _ProviderResultError
from vane.ai.providers._mime import ImageMimePolicy
from vane.ai.typing import UDFOptions

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vane.ai.protocols import Prompter, TextEmbedder
    from vane.ai.typing import Embedding, Options


# https://ai.google.dev/gemini-api/docs/image-understanding#supported-image-formats
_IMAGE_MIME_POLICY = ImageMimePolicy(
    provider_name="Google",
    supported_mime_types=frozenset(
        {
            "image/heic",
            "image/heif",
            "image/jpeg",
            "image/png",
            "image/webp",
        }
    ),
)


def _retry_after_error_from_google_error(exc: Exception) -> Exception | None:
    """Build a safe retry signal for Google 429/503 responses, if applicable.

    Parses the ``Retry-After`` header if present; otherwise falls back to
    a 5-second default wait.
    """
    from vane.ai.functions import RetryAfterError

    code = getattr(exc, "code", None)
    if code not in (429, 503):
        return None

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

    return RetryAfterError(retry_after=retry_after, original=exc)


def _raise_retry_after_on_google_error(exc: Exception) -> None:
    """Compatibility helper used by focused provider retry tests."""
    retry_error = _retry_after_error_from_google_error(exc)
    if retry_error is not None:
        raise retry_error from None


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
_PROMPT_REQUEST_OPTIONS = frozenset({"temperature", "top_p", "top_k", "max_output_tokens", "stop_sequences"})

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


def _validate_google_prompt_model_options(model_name: str, options: Mapping[str, Any]) -> None:
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
        prompt_model: Model used by ``get_prompter`` when the call does not
            pass ``model=...``.
        embedding_model: Model used by ``get_text_embedder`` when the call
            does not pass ``model=...``.
        embedding_dimensions: Trusted fixed output dimensionality used to
            type results for models without built-in metadata. Unlike an
            explicit call-level ``dimensions=...``, it is not sent as a
            server-side dimensionality override.

    All parameters are named; a mistyped keyword raises :class:`TypeError`
    instead of silently leaking into API request options.
    """

    DEFAULT_TEXT_EMBEDDER: ClassVar[str] = "gemini-embedding-2"

    def __init__(
        self,
        name: str | None = None,
        *,
        prompt_model: str | None = None,
        embedding_model: str | None = None,
        embedding_dimensions: int | None = None,
    ):
        self._name = name or "google"
        self._prompt_model = prompt_model
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions

    @property
    def name(self) -> str:
        return self._name

    def get_text_embedder(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> TextEmbedderDescriptor:
        """Build an embedder descriptor for the selected or default model.

        Raises:
            ValueError: If dimensions cannot be resolved for the selected
                model or the model/option combination is invalid.
        """
        resolved_options = dict(options or {})
        model_name = model if model is not None else self._embedding_model
        if model_name is None:
            model_name = self.DEFAULT_TEXT_EMBEDDER
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                f"Google embedding model must be a non-empty string, got {model_name!r}. "
                "Pass model=... or configure GoogleProvider(embedding_model=...)."
            )
        request_dimensions = dimensions
        if dimensions is None and _canonical_model_id(model_name) not in _EMBEDDING_DIMS:
            dimensions = self._embedding_dimensions
        return GoogleTextEmbedderDescriptor(
            model_name=model_name,
            provider_name=self._name,
            dimensions=dimensions,
            request_dimensions=request_dimensions,
            options=resolved_options,
        )

    def get_prompter(
        self,
        model: str | None = None,
        system_message: str | None = None,
        return_format: dict[str, Any] | None = None,
        return_raw_response: bool = False,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> PrompterDescriptor:
        """Build a prompter descriptor for an explicitly selected model.

        Raises:
            ValueError: If neither ``model=...`` nor the provider's
                ``prompt_model`` is configured, or if the selected model
                does not support one of the requested options.
        """
        resolved_options = dict(options or {})
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
            system_message=system_message,
            return_format=return_format,
            return_raw_response=return_raw_response,
            options=resolved_options,
        )


# ---------------------------------------------------------------------------
# Text Embedding
# ---------------------------------------------------------------------------


@dataclass
class GoogleTextEmbedderDescriptor(TextEmbedderDescriptor):
    """Serializable factory for a Google Generative AI text embedder.

    ``model_name`` is required. ``dimensions`` fixes the output type when the
    provider supplies metadata for an otherwise unknown model, while
    ``request_dimensions`` records only an explicit public call override.

    The default UDF ``batch_size`` matches the per-request input cap
    (:data:`_EMBED_BATCH_LIMIT`); the embedder additionally chunks
    oversized batches as defense in depth.
    """

    model_name: str
    provider_name: str = "google"
    dimensions: int | None = None
    request_dimensions: int | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("Google embedding model must be a non-empty string")
        unknown = sorted(set(self.options) - _EMBED_REQUEST_OPTIONS)
        if unknown:
            raise TypeError(f"Unsupported Google Embed option(s): {', '.join(unknown)}")
        validated_options = validate_closed_embed_options("google", self.options, relation=False)
        if _canonical_model_id(self.model_name) in _MODEL_UNSUPPORTED_OPTIONS:
            raise ValueError(f"Google model {self.model_name!r} supports Prompt, not Embed")
        _validate_embedding_dimensions(self.model_name, self.dimensions)
        if _canonical_model_id(self.model_name) == "gemini-embedding-2":
            unsupported = sorted(
                option for option in ("task_type", "title") if validated_options.get(option) is not None
            )
            if unsupported:
                raise ValueError(
                    f"Google model {self.model_name!r} does not support embedding option(s): {', '.join(unsupported)}"
                )
        task_type = validated_options.get("task_type")
        title = validated_options.get("title")
        if title is not None and task_type != "RETRIEVAL_DOCUMENT":
            raise ValueError("Google embedding title is only valid with task_type='RETRIEVAL_DOCUMENT'")
        self.options = wrap_sensitive_options(validated_options)

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> Options:
        return dict(self.options)

    def get_dimensions(self) -> int:
        if self.dimensions is not None:
            return self.dimensions
        return _EMBEDDING_DIMS[_canonical_model_id(self.model_name)]

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(num_gpus=0)

    def is_async(self) -> bool:
        return True

    def instantiate(self) -> TextEmbedder:
        return GoogleTextEmbedder(
            options=self.options,
            model=self.model_name,
            dimensions=self.request_dimensions,
            provider_name=self.provider_name,
        )


class GoogleTextEmbedder:
    """Text embedder using Google Generative AI ``embed_content``."""

    def __init__(
        self,
        options: dict[str, Any],
        model: str,
        dimensions: int | None = None,
        provider_name: str = "google",
    ):
        from google import genai
        from google.genai import types

        options = unwrap_sensitive_options(options)
        http_options = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))
        self._client = genai.Client(http_options=http_options)
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
            retry_error = None
            capability_error: ProviderCapabilityError | None = None
            try:
                result = await self._client.aio.models.embed_content(**kwargs)
            except Exception as exc:
                retry_error = _retry_after_error_from_google_error(exc)
                if retry_error is None and _is_embedding_capability_error(exc):
                    capability_error = ProviderCapabilityError(
                        getattr(self, "_provider_name", "google"),
                        self._model,
                        "embedding endpoint/model",
                        original_error=exc,
                    )
                elif retry_error is None:
                    raise
            if retry_error is not None:
                raise retry_error from None
            if capability_error is not None:
                raise capability_error from None
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


def _validate_google_prompt_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Validate options owned by the Google Prompt adapter."""

    copied = dict(options)
    unknown = sorted(set(copied) - _PROMPT_REQUEST_OPTIONS)
    if unknown:
        raise TypeError(f"Unsupported Google Prompt option(s): {', '.join(unknown)}")
    _require_prompt_number(copied, "temperature", minimum=0, nullable=True)
    _require_prompt_int(copied, "max_output_tokens", minimum=1, nullable=True)
    _require_prompt_int(copied, "top_k", minimum=0, nullable=True)
    _require_prompt_number(copied, "top_p", minimum=0, maximum=1, nullable=True)
    _validate_prompt_stop_sequences(copied)
    return copied


@dataclass
class GooglePrompterDescriptor(PrompterDescriptor):
    """Serializable factory for a basic Gemini text/image prompter."""

    model_name: str
    provider_name: str = "google"
    system_message: str | None = None
    return_format: dict[str, Any] | None = None
    return_raw_response: bool = False
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("Google prompt model must be a non-empty string")
        validated_options = _validate_google_prompt_options(self.options)
        if _canonical_model_id(self.model_name) in _EMBEDDING_DIMS:
            raise ValueError(f"Google model {self.model_name!r} supports Embed, not Prompt")
        _validate_google_prompt_model_options(self.model_name, validated_options)
        self.options = wrap_sensitive_options(validated_options)

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> Options:
        return dict(self.options)

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(num_gpus=0)

    def instantiate(self) -> Prompter:
        return GooglePrompter(
            options=self.options,
            provider_name=self.provider_name,
            model=self.model_name,
            system_message=self.system_message,
            return_format=self.return_format,
            return_raw_response=self.return_raw_response,
        )


class GooglePrompter:
    """Async basic text/image prompter using Gemini ``generate_content``."""

    def __init__(
        self,
        options: dict[str, Any],
        model: str,
        system_message: str | None = None,
        return_format: dict[str, Any] | None = None,
        return_raw_response: bool = False,
        provider_name: str = "google",
    ) -> None:
        from google import genai
        from google.genai import types

        options = unwrap_sensitive_options(options)
        http_options = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))
        self._client = genai.Client(http_options=http_options)
        self._provider_name = provider_name
        self._model = model
        self._system_message = system_message
        self._return_format = return_format
        self._return_raw_response = return_raw_response
        self._options = {k: v for k, v in options.items() if k in _PROMPT_REQUEST_OPTIONS and v is not None}

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
            media_type = _IMAGE_MIME_POLICY.require_supported(msg)
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

        retry_error = None
        capability_error: ProviderCapabilityError | None = None
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            retry_error = _retry_after_error_from_google_error(exc)
            if retry_error is None and _is_prompt_capability_error(exc):
                capability_error = ProviderCapabilityError(
                    self._provider_name,
                    self._model,
                    self._requested_capability(),
                    original_error=exc,
                )
            elif retry_error is None:
                raise
        if retry_error is not None:
            raise retry_error from None
        if capability_error is not None:
            raise capability_error from None

        if getattr(self, "_return_raw_response", False):
            return serialize_raw_response(
                response,
                exclude={"automatic_function_calling_history", "parsed", "sdk_http_response"},
            )

        text = response.text
        return text if text else None
