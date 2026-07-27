# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Google Generative AI (Gemini) provider for Vane AI.

Supports text embedding via ``embed_content`` and prompting via
``generate_content`` with multimodal input (text + images) and
structured output via ``response_schema``.

Vane does not choose Google models: every prompt or embedding call must
name a model, either per call (``model=...``) or through explicit
provider configuration (``GoogleProvider(prompt_model=...,
embedding_model=...)``). Missing configuration raises :class:`ValueError`
at expression-build time.

Requires::

    pip install 'vane-ai[google]'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from vane.ai._redaction import unwrap_sensitive_options, wrap_sensitive_options
from vane.ai.protocols import PrompterDescriptor, TextEmbedderDescriptor
from vane.ai.provider import Provider, ProviderImportError
from vane.ai.typing import EmbeddingDimensions, UDFOptions

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vane.ai.protocols import Prompter, TextEmbedder
    from vane.ai.typing import Embedding, Options


def _guess_media_type(data: bytes) -> str:
    """Guess image MIME type from magic bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:4] == b"GIF8":
        return "image/gif"
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


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

# Conversation roles supported by the Gemini API, mapped from the
# OpenAI/Anthropic-style role names used in Vane message dicts to the wire
# role names Gemini expects. ``system`` is handled separately via
# ``GenerateContentConfig.system_instruction``; anything else is rejected.
_CONVERSATION_ROLES: dict[str, str] = {
    "user": "user",
    "assistant": "model",
}


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

    The provider ships no default model IDs: a model must be supplied per
    call (``model=...``) or through the named constructor parameters below.
    Call-site arguments always win over constructor configuration.

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
        """Build an embedder descriptor for an explicitly selected model.

        Raises:
            ValueError: If neither ``model=...`` nor the provider's
                ``embedding_model`` is configured, or if dimensions cannot
                be resolved for the selected model.
        """
        provider_options, embed_options = self._split_options(options)
        model_name = model if model is not None else self._embedding_model
        if model_name is None:
            raise ValueError(
                "No embedding model configured for the Google provider. "
                "Pass model=... or configure GoogleProvider(embedding_model=...)."
            )
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
        return_format: Any | None = None,
        **options: Any,
    ) -> PrompterDescriptor:
        """Build a prompter descriptor for an explicitly selected model.

        Raises:
            ValueError: If neither ``model=...`` nor the provider's
                ``prompt_model`` is configured, or if the selected model
                does not support one of the requested options.
        """
        provider_options, prompt_options = self._split_options(options)
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
        _validate_embedding_dimensions(self.model_name, self.dimensions)
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
            batch_size=self.embed_options.get("batch_size", _EMBED_BATCH_LIMIT),
            max_retries=self.embed_options.get("max_retries", 3),
            on_error=self.embed_options.get("on_error", "raise"),
            actor_number=self.embed_options.get("actor_number"),
            num_gpus=self.embed_options.get("num_gpus"),
        )

    def is_async(self) -> bool:
        return True

    def instantiate(self) -> TextEmbedder:
        return GoogleTextEmbedder(
            provider_options=self.provider_options,
            model=self.model_name,
            dimensions=self.dimensions,
            **self.embed_options,
        )


class GoogleTextEmbedder:
    """Text embedder using Google Generative AI ``embed_content``."""

    def __init__(
        self,
        provider_options: dict[str, Any],
        model: str,
        dimensions: int | None = None,
        **options: Any,
    ):
        from google import genai

        # Restore plaintext credentials sealed by the descriptor; plain dicts
        # from direct callers pass through unchanged.
        provider_options = unwrap_sensitive_options(provider_options)
        options = unwrap_sensitive_options(options)
        api_key = provider_options.get("api_key")
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._model = model
        self._dimensions = dimensions
        self._options = {
            k: v
            for k, v in options.items()
            if k
            not in {
                "api_key",
                "on_error",
                "actor_number",
                "num_gpus",
                "concurrency",
                "max_api_concurrency",
                "max_retries",
                "model",
                "batch_size",
            }
        }

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
                raise
            chunk_embeddings = result.embeddings or []
            if len(chunk_embeddings) != len(chunk):
                raise ValueError(
                    f"Google embed_content returned {len(chunk_embeddings)} embeddings for "
                    f"{len(chunk)} inputs with model {self._model!r}; embedding calls "
                    "must return exactly one embedding per input row."
                )
            embeddings.extend(np.array(e.values, dtype=np.float32) for e in chunk_embeddings)
        return embeddings


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


@dataclass
class GooglePrompterDescriptor(PrompterDescriptor):
    """Serializable factory for a Google Generative AI (Gemini) prompter.

    Supports structured output via ``response_schema`` and multimodal
    input (text + images via ``Part.from_bytes``).

    ``model_name`` is required, and the model/option combination is
    validated at construction time, before anything ships to workers.
    """

    model_name: str
    provider_name: str = "google"
    provider_options: dict[str, Any] = field(default_factory=dict)
    system_message: str | None = None
    return_format: Any | None = None
    prompt_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            on_error=self.prompt_options.get("on_error", "raise"),
            actor_number=self.prompt_options.get("actor_number"),
            num_gpus=self.prompt_options.get("num_gpus"),
            max_api_concurrency=self.prompt_options.get("max_api_concurrency", 16),
        )

    def instantiate(self) -> Prompter:
        return GooglePrompter(
            provider_options=self.provider_options,
            model=self.model_name,
            system_message=self.system_message,
            return_format=self.return_format,
            **self.prompt_options,
        )


class GooglePrompter:
    """Async prompter using Google Generative AI ``generate_content``.

    Features:
    - Multimodal: str, bytes (images), numpy arrays → Gemini content parts
    - Conversations: role-tagged dicts become ordered ``Content`` turns
      (``user`` → ``user``, ``assistant`` → ``model``); ``system`` messages
      route through ``system_instruction``; other roles are rejected.
    - Structured Output: ``return_format`` (Pydantic BaseModel) uses
      ``response_mime_type="application/json"`` + ``response_schema``.
    """

    def __init__(
        self,
        provider_options: dict[str, Any],
        model: str,
        system_message: str | None = None,
        return_format: Any | None = None,
        **options: Any,
    ):
        from google import genai

        # Restore plaintext credentials sealed by the descriptor; plain dicts
        # from direct callers pass through unchanged.
        provider_options = unwrap_sensitive_options(provider_options)
        options = unwrap_sensitive_options(options)
        api_key = provider_options.get("api_key")
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._model = model
        self._system_message = system_message
        self._return_format = return_format
        self._options = {
            k: v
            for k, v in options.items()
            if k
            not in {
                "api_key",
                "on_error",
                "actor_number",
                "num_gpus",
                "concurrency",
                "max_api_concurrency",
                "model",
                "batch_size",
                "max_retries",
            }
        }

    # --- Multimodal message processing -----------------------------------

    def _process_message(self, msg: Any) -> Any:
        """Convert a message part into a Gemini content part."""
        from google.genai import types

        if isinstance(msg, str):
            return types.Part.from_text(text=msg)
        if isinstance(msg, bytes):
            media_type = _guess_media_type(msg)
            return types.Part.from_bytes(data=msg, mime_type=media_type)
        if isinstance(msg, dict):
            # Structured content part. Only explicit text parts convert;
            # anything else raises instead of degrading into a ``str()`` repr.
            if isinstance(msg.get("text"), str):
                return types.Part.from_text(text=msg["text"])
            if isinstance(msg.get("content"), str):
                return types.Part.from_text(text=msg["content"])
            raise ValueError(f"Unsupported dict content part for the Google provider: {msg!r}")
        # numpy array
        type_name = type(msg).__name__
        mod = getattr(type(msg), "__module__", "")
        if type_name == "ndarray" and "numpy" in mod:
            return self._process_ndarray(msg)
        raise ValueError(f"Unsupported multimodal content type: {type(msg)}")

    def _process_ndarray(self, arr: Any) -> Any:
        import io

        from google.genai import types

        try:
            from PIL import Image
        except ImportError as exc:
            raise ProviderImportError("image", function="ndarray image input") from exc

        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")

    def _content_parts(self, content: Any) -> list[Any]:
        """Convert a role-tagged message's ``content`` into ordered Gemini parts."""
        if isinstance(content, (list, tuple)):
            return [self._process_message(part) for part in content]
        return [self._process_message(content)]

    # --- API call --------------------------------------------------------

    async def prompt(self, messages: tuple[Any, ...]) -> Any:
        """Generate a response for the given message(s).

        Supports multimodal content (str, bytes, numpy arrays) and
        structured output via ``response_schema``.

        Role-tagged dicts (``{"role": ..., "content": ...}``) become genuine
        ordered conversation turns: ``user`` maps to a ``user`` turn,
        ``assistant`` to a ``model`` turn, and ``system`` messages are routed
        through ``GenerateContentConfig.system_instruction`` (combined with
        the descriptor's ``system_message`` in declaration order). Any other
        role raises :class:`ValueError`. Untagged parts between role-tagged
        messages are grouped into user turns, preserving order.
        """
        from google.genai import types

        contents: list[Any] = []
        system_texts: list[str] = []
        if self._system_message:
            system_texts.append(self._system_message)

        # Untagged parts accumulate into a single user turn until a
        # role-tagged message closes it.
        pending_parts: list[Any] = []

        def _flush_pending() -> None:
            if pending_parts:
                contents.append(types.Content(role="user", parts=list(pending_parts)))
                pending_parts.clear()

        for msg in messages:
            if isinstance(msg, dict) and "role" in msg:
                role = msg["role"]
                content = msg.get("content", "")
                if role == "system":
                    if not isinstance(content, str):
                        raise ValueError(
                            f"Google system messages must be plain text, got content of type {type(content).__name__}."
                        )
                    system_texts.append(content)
                    continue
                mapped_role = _CONVERSATION_ROLES.get(role)
                if mapped_role is None:
                    supported = sorted((*_CONVERSATION_ROLES, "system"))
                    raise ValueError(
                        f"Unsupported message role {role!r} for the Google provider. Supported roles: {supported}."
                    )
                _flush_pending()
                contents.append(types.Content(role=mapped_role, parts=self._content_parts(content)))
            else:
                pending_parts.append(self._process_message(msg))

        _flush_pending()

        # Build config
        config_kwargs: dict[str, Any] = {}
        system_instruction = "\n\n".join(text for text in system_texts if text)
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        for k in ("temperature", "top_p", "top_k", "max_output_tokens"):
            if k in self._options:
                config_kwargs[k] = self._options[k]

        # Structured output: JSON mode with schema
        if self._return_format is not None:
            config_kwargs["response_mime_type"] = "application/json"
            if hasattr(self._return_format, "model_json_schema"):
                config_kwargs["response_schema"] = self._return_format.model_json_schema()
            elif isinstance(self._return_format, dict):
                config_kwargs["response_schema"] = self._return_format

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            _raise_retry_after_on_google_error(exc)
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

        if self._return_format is not None:
            # Parse JSON response into Pydantic model if applicable
            import json

            text = response.text
            if text:
                data = json.loads(text)
                if hasattr(self._return_format, "model_validate"):
                    return self._return_format.model_validate(data)
                return data
            return None

        if response.text:
            return response.text
        return None
