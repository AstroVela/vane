# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI provider for Vane AI.

Supports text embedding via the OpenAI Embeddings API and basic text/image
Prompt calls via the Responses API or Chat Completions API.

Requires::

    pip install 'vane-ai[openai]'
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from vane.ai._redaction import unwrap_sensitive_options, wrap_sensitive_options
from vane.ai.options import validate_embed_options, validate_prompt_options
from vane.ai.protocols import PrompterDescriptor, TextEmbedderDescriptor
from vane.ai.provider import Provider, ProviderCapabilityError, _ProviderResultError
from vane.ai.typing import EmbeddingDimensions, UDFOptions

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vane.ai.protocols import Prompter, TextEmbedder
    from vane.ai.typing import Embedding, Options


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

_MODEL_DIMS: dict[str, int] = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

_DIMENSION_OVERRIDABLE = {"text-embedding-3-small", "text-embedding-3-large"}

# Per-model max input token limit (single text).
# Texts exceeding this are chunked and their embeddings weight-averaged.
_MODEL_INPUT_TOKEN_LIMITS: dict[str, int] = {
    "text-embedding-ada-002": 8191,
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
}
_DEFAULT_INPUT_TOKEN_LIMIT = 8192

_EMBED_CAPABILITY_ERROR_PARAMS = frozenset({"model", "dimensions", "encoding_format"})
_EMBED_CAPABILITY_ERROR_CODES = frozenset(
    {
        "invalid_dimensions",
        "invalid_model",
        "model_not_found",
        "model_not_supported",
        "unsupported_dimensions",
        "unsupported_encoding_format",
        "unsupported_model",
    }
)


def _decode_openai_embedding_base64(value: str) -> np.ndarray:
    raw = base64.b64decode(value)
    return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)


def _get_input_token_limit(model: str) -> int:
    """Return the per-input token limit for *model*, defaulting to 8192."""
    return _MODEL_INPUT_TOKEN_LIMITS.get(model, _DEFAULT_INPUT_TOKEN_LIMIT)


def _chunk_text(text: str, char_size: int) -> list[str]:
    """Split *text* into character-level chunks of at most *char_size*."""
    return [text[i : i + char_size] for i in range(0, len(text), char_size)]


def _is_embedding_capability_error(exc: Exception) -> bool:
    """Classify only structured endpoint/model embedding failures."""
    status_code = getattr(exc, "status_code", None)
    if status_code in {404, 405, 501}:
        return True
    if status_code not in {400, 422}:
        return False

    param = getattr(exc, "param", None)
    code = getattr(exc, "code", None)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        details = body.get("error", body)
        if isinstance(details, dict):
            param = param or details.get("param")
            code = code or details.get("code")

    normalized_param = str(param).strip().casefold() if param is not None else ""
    normalized_code = str(code).strip().casefold() if code is not None else ""
    return normalized_param in _EMBED_CAPABILITY_ERROR_PARAMS or normalized_code in _EMBED_CAPABILITY_ERROR_CODES


def _is_prompt_capability_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {404, 405, 501}:
        return True
    if status_code not in {400, 422}:
        return False
    param = getattr(exc, "param", None)
    code = getattr(exc, "code", None)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        details = body.get("error", body)
        if isinstance(details, dict):
            param = param or details.get("param")
            code = code or details.get("code")
    normalized_param = str(param or "").strip().casefold()
    normalized_code = str(code or "").strip().casefold()
    capability_params = {"model", "image", "input_image", "input"}
    return normalized_param in capability_params or any(
        marker in normalized_code for marker in ("model_not_found", "unsupported_model", "unsupported_image")
    )


# OpenAI-specific keys sealed in addition to the shared sensitive-key table.
# ``organization`` identifies the paying account and must not leak via repr,
# but it is not a generic credential, so it stays out of the shared table
# (which also drives SQL inline-credential rejection). Suffix matching covers
# nested forms such as an ``OpenAI-Organization`` request header.
_EXTRA_SENSITIVE_KEYS = frozenset({"organization"})


def _wrap_openai_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Seal shared sensitive keys plus OpenAI-specific ones (``organization``) at any depth."""
    return wrap_sensitive_options(options, extra_keys=_EXTRA_SENSITIVE_KEYS)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAIProvider(Provider):
    """Provider backed by the OpenAI API (or any compatible endpoint)."""

    DEFAULT_TEXT_EMBEDDER = "text-embedding-3-small"
    DEFAULT_PROMPTER_MODEL = "gpt-4o-mini"

    def __init__(self, name: str | None = None, **options: Any):
        self._name = name or "openai"
        self._options: dict[str, Any] = options

    @property
    def name(self) -> str:
        return self._name

    _CLIENT_KEYS = {"api_key", "base_url", "organization", "timeout", "max_retries"}
    _PROMPT_CLIENT_KEYS = {"api_key", "base_url", "organization", "timeout"}
    _EMBED_CLIENT_KEYS = _CLIENT_KEYS - {"max_retries"}
    _EMBED_REQUEST_KEYS = {"encoding_format", "batch_token_limit", "input_text_token_limit"}

    def get_text_embedder(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        **options: Any,
    ) -> TextEmbedderDescriptor:
        merged = {**self._options, **options}
        unknown = sorted(set(merged) - self._EMBED_CLIENT_KEYS - self._EMBED_REQUEST_KEYS)
        if unknown:
            raise TypeError(f"Unsupported OpenAI Embed option(s): {', '.join(unknown)}")
        validate_embed_options(
            "openai",
            {
                key: value
                for key, value in merged.items()
                if key in self._EMBED_REQUEST_KEYS or key in {"base_url", "timeout"}
            },
            relation=False,
        )
        provider_opts = {key: value for key, value in merged.items() if key in self._EMBED_CLIENT_KEYS}
        embed_opts = {key: value for key, value in merged.items() if key in self._EMBED_REQUEST_KEYS}
        return OpenAITextEmbedderDescriptor(
            provider_name=self._name,
            provider_options=provider_opts,
            model_name=model or self.DEFAULT_TEXT_EMBEDDER,
            dimensions=dimensions,
            embed_options=embed_opts,
        )

    def get_prompter(
        self,
        model: str | None = None,
        system_message: str | None = None,
        **options: Any,
    ) -> PrompterDescriptor:
        provider_opts = {key: value for key, value in self._options.items() if key in self._PROMPT_CLIENT_KEYS}
        prompt_options = {key: value for key, value in self._options.items() if key not in self._PROMPT_CLIENT_KEYS}
        prompt_options.update(options)
        validation_options = {key: value for key, value in provider_opts.items() if key in {"base_url", "timeout"}}
        validation_options.update(prompt_options)
        validate_prompt_options("openai", validation_options, relation=False)
        for key in ("base_url", "timeout"):
            if key in prompt_options:
                value = prompt_options.pop(key)
                if value is None:
                    provider_opts.pop(key, None)
                else:
                    provider_opts[key] = value
        use_chat_completions = prompt_options.pop("use_chat_completions", False)
        return OpenAIPrompterDescriptor(
            provider_name=self._name,
            provider_options=provider_opts,
            model_name=model or self.DEFAULT_PROMPTER_MODEL,
            system_message=system_message,
            use_chat_completions=use_chat_completions,
            prompt_options=prompt_options,
        )


# ---------------------------------------------------------------------------
# Text Embedding
# ---------------------------------------------------------------------------


@dataclass
class OpenAITextEmbedderDescriptor(TextEmbedderDescriptor):
    """Serializable factory for an OpenAI text embedder."""

    provider_name: str = "openai"
    provider_options: dict[str, Any] = field(default_factory=dict)
    model_name: str = "text-embedding-3-small"
    dimensions: int | None = None
    embed_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.embed_options) - OpenAIProvider._EMBED_REQUEST_KEYS)
        if unknown:
            raise TypeError(f"Unsupported OpenAI Embed option(s): {', '.join(unknown)}")
        if self.dimensions is not None and (
            isinstance(self.dimensions, bool) or not isinstance(self.dimensions, int) or self.dimensions <= 0
        ):
            raise ValueError("Embedding dimensions must be a positive integer")
        if (
            self.dimensions is not None
            and self.model_name in _MODEL_DIMS
            and self.model_name not in _DIMENSION_OVERRIDABLE
        ):
            raise ValueError(f"Model {self.model_name!r} does not support custom dimensions")
        if (
            self.dimensions is not None
            and self.model_name in _DIMENSION_OVERRIDABLE
            and self.dimensions > _MODEL_DIMS[self.model_name]
        ):
            raise ValueError(
                f"Model {self.model_name!r} supports at most {_MODEL_DIMS[self.model_name]} dimensions, "
                f"got {self.dimensions}"
            )
        self.provider_options = _wrap_openai_options(self.provider_options)
        self.embed_options = _wrap_openai_options(self.embed_options)

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> Options:
        return dict(self.embed_options)

    def get_dimensions(self) -> EmbeddingDimensions:
        if self.dimensions is not None:
            return EmbeddingDimensions(size=self.dimensions, dtype=pa.float32())
        if self.model_name in _MODEL_DIMS:
            return EmbeddingDimensions(size=_MODEL_DIMS[self.model_name], dtype=pa.float32())
        raise ValueError(
            f"Cannot determine embedding dimensions for OpenAI-compatible model {self.model_name!r} "
            "from trusted local metadata; pass dimensions=... explicitly"
        )

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(
            batch_size=64,
            max_retries=3,
            on_error="raise",
            actor_number=None,
            num_gpus=0,
        )

    def is_async(self) -> bool:
        return True

    def instantiate(self) -> TextEmbedder:
        provider_options = dict(self.provider_options)
        # Embed retries belong to Vane's row-aware wrapper. Disable SDK retry
        # stacking so max_retries has one deterministic meaning.
        provider_options["max_retries"] = 0
        return OpenAITextEmbedder(
            provider_options=provider_options,
            provider_name=self.provider_name,
            model=self.model_name,
            dimensions=self.dimensions,
            encoding_format=self.embed_options.get("encoding_format", "float"),
            batch_token_limit=self.embed_options.get("batch_token_limit", 300_000),
            input_text_token_limit=self.embed_options.get("input_text_token_limit", None),
        )


class OpenAITextEmbedder:
    """Async text embedder using the OpenAI Embeddings API.

    Two-level token limiting:

    * **batch_token_limit** — max estimated tokens per API request (default 300k).
    * **input_text_token_limit** — max tokens for a single input text.
      Texts exceeding this are split into character chunks, embedded
      separately, and recombined via length-weighted averaging + L2
      normalisation.  The estimation is conservative: ``ceil(len(text) / 3)``
      (≈ 1 token per 3 chars), which is O(1) and avoids a tiktoken
      dependency.
    """

    def __init__(
        self,
        provider_options: dict[str, Any],
        model: str,
        dimensions: int | None = None,
        encoding_format: str = "float",
        batch_token_limit: int = 300_000,
        input_text_token_limit: int | None = None,
        provider_name: str = "openai",
    ):
        from openai import AsyncOpenAI

        if encoding_format not in {"float", "base64"}:
            raise ValueError("encoding_format must be 'float' or 'base64'")
        # Restore plaintext credentials sealed by the descriptor; plain dicts
        # from direct callers pass through unchanged.
        provider_options = unwrap_sensitive_options(provider_options)
        self._provider_name = provider_name
        self._model = model
        self._dimensions = dimensions
        self._encoding_format = encoding_format
        self._batch_token_limit = batch_token_limit
        self._input_text_token_limit = (
            input_text_token_limit if input_text_token_limit is not None else _get_input_token_limit(model)
        )
        # Filter out non-OpenAI keys before passing to client
        client_opts = {
            k: v
            for k, v in provider_options.items()
            if k in {"api_key", "base_url", "organization", "timeout", "max_retries"}
        }
        self._client = AsyncOpenAI(**client_opts)

    async def aclose(self) -> None:
        """Release the SDK client's connection pool on the owning loop."""
        await self._client.close()

    async def embed_text(self, text: list[str]) -> list[Embedding]:
        embeddings: list[Embedding] = []
        batch: list[str] = []
        batch_tokens = 0
        approx_chars_per_token = 3

        def estimate_tokens(value: str) -> int:
            return (len(value) + approx_chars_per_token - 1) // approx_chars_per_token

        async def flush() -> None:
            nonlocal batch, batch_tokens
            if not batch:
                return
            result = await self._embed_batch(batch)
            embeddings.extend(result)
            batch = []
            batch_tokens = 0

        for item in text:
            if item is None:
                item = ""
            est_tokens = estimate_tokens(item)
            single_input_limit = min(self._input_text_token_limit, self._batch_token_limit)

            if est_tokens > single_input_limit:
                # Oversized single input — flush pending batch, chunk, embed,
                # then recombine via weighted average + L2 normalisation.
                await flush()
                chunk_char_size = single_input_limit * approx_chars_per_token
                chunks = _chunk_text(item, chunk_char_size)
                chunk_embeddings: list[Embedding] = []
                chunk_batch: list[str] = []
                chunk_batch_tokens = 0
                for chunk in chunks:
                    chunk_tokens = estimate_tokens(chunk)
                    if chunk_batch and chunk_batch_tokens + chunk_tokens > self._batch_token_limit:
                        chunk_embeddings.extend(await self._embed_batch(chunk_batch))
                        chunk_batch = []
                        chunk_batch_tokens = 0
                    chunk_batch.append(chunk)
                    chunk_batch_tokens += chunk_tokens
                if chunk_batch:
                    chunk_embeddings.extend(await self._embed_batch(chunk_batch))
                chunk_lens = np.array(
                    [len(c) for c in chunks],
                    dtype=np.float64,
                )
                avg = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
                norm = np.linalg.norm(avg)
                if norm > 0:
                    avg = avg / norm
                embeddings.append(avg)
                continue

            if batch and est_tokens + batch_tokens > self._batch_token_limit:
                await flush()
            batch.append(item)
            batch_tokens += est_tokens

        await flush()
        return embeddings

    async def _embed_batch(self, texts: list[str]) -> list[Embedding]:
        from openai import OpenAIError

        try:
            encoding_format = getattr(self, "_encoding_format", "float")
            kwargs: dict[str, Any] = {
                "input": texts,
                "model": self._model,
                "encoding_format": encoding_format,
            }
            if self._dimensions is not None:
                kwargs["dimensions"] = self._dimensions
            response = await self._client.embeddings.create(**kwargs)
            response_data = list(response.data)
            if len(response_data) != len(texts):
                raise _ProviderResultError(
                    f"OpenAI Embeddings API returned {len(response_data)} embeddings for {len(texts)} inputs; "
                    "embedding calls must preserve row count and order"
                )
            indices = [getattr(item, "index", None) for item in response_data]
            if any(index is not None for index in indices):
                if any(type(index) is not int for index in indices) or sorted(indices) != list(range(len(texts))):
                    raise _ProviderResultError(
                        "OpenAI Embeddings API returned invalid embedding indices; "
                        "embedding calls must preserve row count and order"
                    )
                response_data = [
                    item for _, item in sorted(zip(indices, response_data, strict=True), key=lambda pair: pair[0])
                ]
            if hasattr(response, "usage") and response.usage is not None:
                from vane.ai.metrics import record_token_metrics

                record_token_metrics(
                    protocol="embed",
                    model=self._model,
                    provider="openai",
                    input_tokens=getattr(response.usage, "prompt_tokens", None),
                    total_tokens=getattr(response.usage, "total_tokens", None),
                )
            if encoding_format == "base64":
                return [_decode_openai_embedding_base64(e.embedding) for e in response_data]
            return [np.array(e.embedding, dtype=np.float32) for e in response_data]
        except OpenAIError as ex:
            if _is_embedding_capability_error(ex):
                raise ProviderCapabilityError(
                    getattr(self, "_provider_name", "openai"),
                    self._model,
                    "embedding endpoint/model",
                    original_error=ex,
                ) from ex
            raise


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


@dataclass
class OpenAIPrompterDescriptor(PrompterDescriptor):
    """Serializable factory for a basic text/image OpenAI prompter."""

    provider_name: str = "openai"
    provider_options: dict[str, Any] = field(default_factory=dict)
    model_name: str = "gpt-4o-mini"
    system_message: str | None = None
    use_chat_completions: bool = False
    prompt_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("OpenAI prompt model must be a non-empty string")
        validation_options = {
            key: value for key, value in self.provider_options.items() if key in {"base_url", "timeout"}
        }
        validation_options.update(self.prompt_options)
        validate_prompt_options("openai", validation_options, relation=False)
        if self.prompt_options.get("stop_sequences") is not None and not self.use_chat_completions:
            raise ValueError("OpenAI stop_sequences requires use_chat_completions=True")
        self.provider_options = _wrap_openai_options(self.provider_options)
        self.prompt_options = _wrap_openai_options(self.prompt_options)

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
            max_concurrency_per_actor=self.prompt_options.get("max_concurrency_per_actor", 32),
        )

    def instantiate(self) -> Prompter:
        return OpenAIPrompter(
            provider_options=self.provider_options,
            provider_name=self.provider_name,
            model=self.model_name,
            system_message=self.system_message,
            use_chat_completions=self.use_chat_completions,
            **self.prompt_options,
        )


class OpenAIPrompter:
    """Async basic text/image prompter for Responses or Chat Completions."""

    def __init__(
        self,
        provider_options: dict[str, Any],
        model: str,
        system_message: str | None = None,
        use_chat_completions: bool = False,
        provider_name: str = "openai",
        **options: Any,
    ) -> None:
        from openai import AsyncOpenAI

        # Restore plaintext credentials sealed by the descriptor; plain dicts
        # from direct callers pass through unchanged.
        provider_options = unwrap_sensitive_options(provider_options)
        options = unwrap_sensitive_options(options)
        self._provider_name = provider_name
        self._model = model
        self._system_message = system_message
        self._use_chat_completions = use_chat_completions
        self._options = {
            key: value
            for key, value in options.items()
            if key in {"temperature", "max_output_tokens", "top_p", "stop_sequences"} and value is not None
        }
        client_opts = {
            k: v for k, v in provider_options.items() if k in {"api_key", "base_url", "organization", "timeout"}
        }
        client_opts["max_retries"] = 0
        self._client = AsyncOpenAI(**client_opts)

    async def aclose(self) -> None:
        """Release the SDK client's connection pool on the owning loop."""
        await self._client.close()

    # --- Multimodal message processing -----------------------------------

    def _process_message(self, msg: Any) -> dict[str, Any]:
        """Convert one validated Vane text/image part to the OpenAI shape."""
        if isinstance(msg, str):
            return self._process_str(msg)
        if isinstance(msg, bytes):
            return self._process_bytes(msg)
        raise TypeError(f"Unsupported Prompt content type: {type(msg).__name__}")

    def _process_str(self, msg: str) -> dict[str, Any]:
        if self._use_chat_completions:
            return {"type": "text", "text": msg}
        return {"type": "input_text", "text": msg}

    def _process_bytes(self, msg: bytes) -> dict[str, Any]:
        import base64

        mime_type = _guess_mime_type(msg)
        if mime_type is None:
            raise ValueError("Prompt image BLOB has an unsupported or unrecognized image format")
        b64 = base64.b64encode(msg).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"
        return self._build_image_part(data_url)

    def _build_image_part(self, data_url: str) -> dict[str, Any]:
        if self._use_chat_completions:
            return {"type": "image_url", "image_url": {"url": data_url}}
        return {"type": "input_image", "image_url": data_url}

    # --- API dispatch -----------------------------------------------------

    def _record_usage(self, response: Any) -> None:
        """Extract token usage from an API response and record metrics."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        from vane.ai.metrics import record_token_metrics

        # Chat Completions API: prompt_tokens / completion_tokens / total_tokens
        # Responses API: input_tokens / output_tokens / total_tokens
        record_token_metrics(
            protocol="prompt",
            model=self._model,
            provider="openai",
            input_tokens=(getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)),
            output_tokens=(getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)),
            total_tokens=getattr(usage, "total_tokens", None),
        )

    def _chat_completions_options(self) -> dict[str, Any]:
        options = dict(self._options)
        if "max_output_tokens" in options:
            options["max_tokens"] = options["max_output_tokens"]
        options.pop("max_output_tokens", None)
        if "stop_sequences" in options:
            options["stop"] = options.pop("stop_sequences")
        return options

    def _responses_options(self) -> dict[str, Any]:
        return dict(self._options)

    async def _prompt_chat_completions(self, messages: list[dict[str, Any]]) -> str | None:
        """Prompt using the Chat Completions API."""
        options = self._chat_completions_options()
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                **options,
            )
        except Exception as exc:
            if _is_prompt_capability_error(exc):
                raise ProviderCapabilityError(
                    self._provider_name,
                    self._model,
                    "basic Prompt text/image generation",
                    original_error=exc,
                ) from exc
            raise
        self._record_usage(response)
        return response.choices[0].message.content

    async def _prompt_responses(self, messages: list[dict[str, Any]]) -> str | None:
        """Prompt using the Responses API."""
        options = self._responses_options()
        try:
            response = await self._client.responses.create(
                model=self._model,
                input=messages,
                **options,
            )
        except Exception as exc:
            if _is_prompt_capability_error(exc):
                raise ProviderCapabilityError(
                    self._provider_name,
                    self._model,
                    "basic Prompt text/image generation",
                    original_error=exc,
                ) from exc
            raise
        self._record_usage(response)
        return response.output_text

    async def prompt(self, messages: tuple[Any, ...]) -> str | None:
        chat_messages: list[dict[str, Any]] = []
        if self._system_message is not None:
            chat_messages.append({"role": "system", "content": self._system_message})
        chat_messages.append({"role": "user", "content": [self._process_message(msg) for msg in messages]})

        if self._use_chat_completions:
            return await self._prompt_chat_completions(chat_messages)
        return await self._prompt_responses(chat_messages)


def _guess_mime_type(data: bytes) -> str | None:
    """Guess a supported image MIME type from magic bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:4] == b"GIF8":
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None
