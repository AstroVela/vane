# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Google Generative AI (Gemini) provider for Vane AI.

Supports text embedding via ``embed_content`` and prompting via
``generate_content`` with multimodal input (text + images) and
structured output via ``response_schema``.

Requires::

    pip install 'vane-ai[google]'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from vane.ai._redaction import unwrap_sensitive_options, wrap_sensitive_options
from vane.ai.protocols import PrompterDescriptor, TextEmbedderDescriptor
from vane.ai.provider import Provider
from vane.ai.typing import EmbeddingDimensions, UDFOptions, api_worker_options

if TYPE_CHECKING:
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

_EMBEDDING_DIMS: dict[str, int] = {
    "text-embedding-004": 768,
    "embedding-001": 768,
}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class GoogleProvider(Provider):
    """Provider backed by Google Generative AI (Gemini)."""

    DEFAULT_EMBED_MODEL = "text-embedding-004"
    DEFAULT_PROMPT_MODEL = "gemini-2.0-flash"
    _CLIENT_KEYS: ClassVar[frozenset[str]] = frozenset({"api_key"})

    def __init__(self, name: str | None = None, **options: Any):
        self._name = name or "google"
        self._options: dict[str, Any] = options

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
        provider_options, embed_options = self._split_options(options)
        model_name = model or self.DEFAULT_EMBED_MODEL
        return GoogleTextEmbedderDescriptor(
            provider_name=self._name,
            provider_options=provider_options,
            model_name=model_name,
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
        provider_options, prompt_options = self._split_options(options)
        return GooglePrompterDescriptor(
            provider_name=self._name,
            provider_options=provider_options,
            model_name=model or prompt_options.pop("model", self.DEFAULT_PROMPT_MODEL),
            system_message=system_message,
            return_format=return_format,
            prompt_options=prompt_options,
        )


# ---------------------------------------------------------------------------
# Text Embedding
# ---------------------------------------------------------------------------


@dataclass
class GoogleTextEmbedderDescriptor(TextEmbedderDescriptor):
    """Serializable factory for a Google Generative AI text embedder."""

    provider_name: str = "google"
    provider_options: dict[str, Any] = field(default_factory=dict)
    model_name: str = "text-embedding-004"
    dimensions: int | None = None
    embed_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.provider_options = wrap_sensitive_options(self.provider_options)
        self.embed_options = wrap_sensitive_options(self.embed_options)

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> Options:
        return dict(self.embed_options)

    def get_dimensions(self) -> EmbeddingDimensions:
        if self.dimensions:
            return EmbeddingDimensions(size=self.dimensions)
        size = _EMBEDDING_DIMS.get(self.model_name, 768)
        return EmbeddingDimensions(size=size)

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(
            max_retries=self.embed_options.get("max_retries", 3),
            on_error=self.embed_options.get("on_error", "raise"),
            **api_worker_options(self.embed_options),
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
        kwargs: dict[str, Any] = {
            "model": self._model,
            "contents": text,
        }
        config = dict(self._options)
        if self._dimensions is not None:
            config["output_dimensionality"] = self._dimensions
        if config:
            kwargs["config"] = config

        try:
            result = await self._client.aio.models.embed_content(**kwargs)
        except Exception as exc:
            _raise_retry_after_on_google_error(exc)
            raise
        return [np.array(e.values, dtype=np.float32) for e in result.embeddings]


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


@dataclass
class GooglePrompterDescriptor(PrompterDescriptor):
    """Serializable factory for a Google Generative AI (Gemini) prompter.

    Supports structured output via ``response_schema`` and multimodal
    input (text + images via ``Part.from_bytes``).
    """

    provider_name: str = "google"
    provider_options: dict[str, Any] = field(default_factory=dict)
    model_name: str = "gemini-2.0-flash"
    system_message: str | None = None
    return_format: Any | None = None
    prompt_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            max_api_concurrency=self.prompt_options.get("max_api_concurrency", 16),
            **api_worker_options(self.prompt_options),
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
            # Dict content part — convert to text representation
            return types.Part.from_text(text=str(msg.get("content", msg)))
        # numpy array
        type_name = type(msg).__name__
        mod = getattr(type(msg), "__module__", "")
        if type_name == "ndarray" and "numpy" in mod:
            return self._process_ndarray(msg)
        raise ValueError(f"Unsupported multimodal content type: {type(msg)}")

    def _process_ndarray(self, arr: Any) -> Any:
        import io

        from google.genai import types
        from PIL import Image

        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")

    # --- API call --------------------------------------------------------

    async def prompt(self, messages: tuple[Any, ...]) -> Any:
        """Generate a response for the given message(s).

        Supports multimodal content (str, bytes, numpy arrays) and
        structured output via ``response_schema``.
        """
        from google.genai import types

        # Build content parts
        parts: list[Any] = []
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg:
                # Complete message — extract text content
                parts.append(types.Part.from_text(text=str(msg.get("content", ""))))
            else:
                parts.append(self._process_message(msg))

        contents = [types.Content(role="user", parts=parts)]

        # Build config
        config_kwargs: dict[str, Any] = {}
        if self._system_message:
            config_kwargs["system_instruction"] = self._system_message
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
