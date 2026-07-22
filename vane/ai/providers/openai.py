# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI provider for Vane AI.

Supports text embedding via the OpenAI Embeddings API and prompting via
the Chat Completions API or the newer Responses API (with Structured Output).

Requires::

    pip install 'vane-ai[openai]'
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pyarrow as pa

from vane.ai._redaction import unwrap_sensitive_options, wrap_sensitive_options
from vane.ai.protocols import PrompterDescriptor, TextEmbedderDescriptor
from vane.ai.provider import Provider
from vane.ai.typing import EmbeddingDimensions, UDFOptions, api_worker_options

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vane.ai.protocols import Prompter, TextEmbedder
    from vane.ai.typing import Embedding, Options

# Keys that configure the OpenAI SDK client itself (as opposed to embed/prompt
# request options). ``max_retries`` here is the SDK client's internal retry
# count, not the vane UDF retry option.
_OPENAI_CLIENT_KEYS = frozenset({"api_key", "base_url", "organization", "timeout", "max_retries"})


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


def _decode_openai_embedding_base64(value: str) -> np.ndarray:
    raw = base64.b64decode(value)
    return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)


def _get_input_token_limit(model: str) -> int:
    """Return the per-input token limit for *model*, defaulting to 8192."""
    return _MODEL_INPUT_TOKEN_LIMITS.get(model, _DEFAULT_INPUT_TOKEN_LIMIT)


def _chunk_text(text: str, char_size: int) -> list[str]:
    """Split *text* into character-level chunks of at most *char_size*."""
    return [text[i : i + char_size] for i in range(0, len(text), char_size)]


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

    _CLIENT_KEYS: ClassVar[frozenset[str]] = _OPENAI_CLIENT_KEYS

    def _split_options(self, options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Merge provider-level defaults with call options and split them.

        Call-level values override provider-constructor defaults. Client
        configuration (``_CLIENT_KEYS``) goes to ``provider_options``;
        everything else flows to the embed/prompt request options, so
        provider-level defaults such as ``batch_size=``,
        ``max_api_concurrency=``, or ``on_error=`` take effect instead of
        being silently dropped.
        """
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
        return OpenAITextEmbedderDescriptor(
            provider_name=self._name,
            provider_options=provider_options,
            model_name=model or embed_options.pop("model", self.DEFAULT_TEXT_EMBEDDER),
            dimensions=dimensions,
            embed_options=embed_options,
        )

    def get_prompter(
        self,
        model: str | None = None,
        system_message: str | None = None,
        return_format: Any | None = None,
        use_chat_completions: bool = True,
        **options: Any,
    ) -> PrompterDescriptor:
        provider_options, prompt_options = self._split_options(options)
        return OpenAIPrompterDescriptor(
            provider_name=self._name,
            provider_options=provider_options,
            model_name=model or prompt_options.pop("model", self.DEFAULT_PROMPTER_MODEL),
            system_message=system_message,
            return_format=return_format,
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
    embed_options: dict[str, Any] = field(default_factory=lambda: {"batch_size": 64, "max_retries": 3})

    def __post_init__(self) -> None:
        if (
            self.dimensions is not None
            and self.model_name in _MODEL_DIMS
            and self.model_name not in _DIMENSION_OVERRIDABLE
        ):
            raise ValueError(f"Model {self.model_name!r} does not support custom dimensions")
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
        # Unknown model with custom base_url — probe the server. Only client
        # configuration reaches the probe client; descriptors built by older
        # code (or direct consumers) may carry extra keys in provider_options.
        from openai import OpenAI as OpenAIClient

        client_opts = {
            k: v for k, v in unwrap_sensitive_options(self.provider_options).items() if k in _OPENAI_CLIENT_KEYS
        }
        client = OpenAIClient(**client_opts)
        response = client.embeddings.create(
            input="dimension probe",
            model=self.model_name,
            encoding_format="float",
        )
        size = len(response.data[0].embedding)
        return EmbeddingDimensions(size=size, dtype=pa.float32())

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(
            max_retries=0,  # OpenAI client handles retries internally
            on_error=self.embed_options.get("on_error", "raise"),
            **api_worker_options(self.embed_options, default_batch_size=64),
        )

    def is_async(self) -> bool:
        return True

    def instantiate(self) -> TextEmbedder:
        return OpenAITextEmbedder(
            provider_options=self.provider_options,
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
      normalisation.  The estimation is conservative: ``len(text) // 3``
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
    ):
        from openai import AsyncOpenAI

        if encoding_format not in {"float", "base64"}:
            raise ValueError("encoding_format must be 'float' or 'base64'")
        # Restore plaintext credentials sealed by the descriptor; plain dicts
        # from direct callers pass through unchanged.
        provider_options = unwrap_sensitive_options(provider_options)
        self._model = model
        self._dimensions = dimensions
        self._encoding_format = encoding_format
        self._batch_token_limit = batch_token_limit
        self._input_text_token_limit = (
            input_text_token_limit if input_text_token_limit is not None else _get_input_token_limit(model)
        )
        # Filter out non-OpenAI keys before passing to client
        client_opts = {k: v for k, v in provider_options.items() if k in _OPENAI_CLIENT_KEYS}
        self._client = AsyncOpenAI(**client_opts)

    async def embed_text(self, text: list[str]) -> list[Embedding]:
        embeddings: list[Embedding] = []
        batch: list[str] = []
        batch_tokens = 0
        approx_chars_per_token = 3

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
            est_tokens = len(item) // approx_chars_per_token

            if est_tokens > self._input_text_token_limit:
                # Oversized single input — flush pending batch, chunk, embed,
                # then recombine via weighted average + L2 normalisation.
                await flush()
                chunk_char_size = self._input_text_token_limit * approx_chars_per_token
                chunks = _chunk_text(item, chunk_char_size)
                chunk_embeddings = await self._embed_batch(chunks)
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

            if est_tokens + batch_tokens >= self._batch_token_limit:
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
                return [_decode_openai_embedding_base64(e.embedding) for e in response.data]
            return [np.array(e.embedding, dtype=np.float32) for e in response.data]
        except OpenAIError as ex:
            raise ValueError("OpenAI embed_text error") from ex


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


@dataclass
class OpenAIPrompterDescriptor(PrompterDescriptor):
    """Serializable factory for an OpenAI prompter.

    Supports both Chat Completions API and the newer Responses API,
    with optional Structured Output via ``return_format``.
    """

    provider_name: str = "openai"
    provider_options: dict[str, Any] = field(default_factory=dict)
    model_name: str = "gpt-4o-mini"
    system_message: str | None = None
    return_format: Any | None = None
    use_chat_completions: bool = True
    prompt_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            max_retries=0,  # OpenAI client handles retries internally
            on_error=self.prompt_options.get("on_error", "raise"),
            max_api_concurrency=self.prompt_options.get("max_api_concurrency", 32),
            **api_worker_options(self.prompt_options),
        )

    def instantiate(self) -> Prompter:
        return OpenAIPrompter(
            provider_options=self.provider_options,
            model=self.model_name,
            system_message=self.system_message,
            return_format=self.return_format,
            use_chat_completions=self.use_chat_completions,
            **self.prompt_options,
        )


class OpenAIPrompter:
    """Async prompter supporting Chat Completions and Responses APIs.

    Features:
    - Chat Completions API: ``completions.create()`` / ``completions.parse()``
    - Responses API: ``responses.create()`` / ``responses.parse()``
    - Structured Output: pass ``return_format`` (Pydantic BaseModel) to get
      typed responses parsed by the API.
    - Multimodal input: str, bytes (images), numpy arrays (images), and
      pre-built dicts are all supported as message parts.
    """

    def __init__(
        self,
        provider_options: dict[str, Any],
        model: str,
        system_message: str | None = None,
        return_format: Any | None = None,
        use_chat_completions: bool = True,
        **options: Any,
    ):
        from openai import AsyncOpenAI

        # Restore plaintext credentials sealed by the descriptor; plain dicts
        # from direct callers pass through unchanged.
        provider_options = unwrap_sensitive_options(provider_options)
        options = unwrap_sensitive_options(options)
        self._model = model
        self._system_message = system_message
        self._return_format = return_format
        self._use_chat_completions = use_chat_completions
        self._options = {
            k: v
            for k, v in options.items()
            if k
            not in {
                "on_error",
                "actor_number",
                "num_gpus",
                "concurrency",
                "max_api_concurrency",
                "model",
                "batch_size",
            }
        }
        client_opts = {k: v for k, v in provider_options.items() if k in _OPENAI_CLIENT_KEYS}
        self._client = AsyncOpenAI(**client_opts)

    # --- Multimodal message processing -----------------------------------

    def _process_message(self, msg: Any) -> dict[str, Any]:
        """Convert a message part into an OpenAI content-part dict.

        Dispatches based on type:
        - str → text part
        - bytes → base64-encoded image/file part (MIME auto-detected)
        - numpy ndarray → PNG-encoded image part
        - dict → passed through as-is (already a content part)
        """
        if isinstance(msg, str):
            return self._process_str(msg)
        if isinstance(msg, bytes):
            return self._process_bytes(msg)
        if isinstance(msg, dict):
            return msg
        # numpy array — check without hard import
        type_name = type(msg).__name__
        mod = getattr(type(msg), "__module__", "")
        if type_name == "ndarray" and "numpy" in mod:
            return self._process_ndarray(msg)
        raise ValueError(f"Unsupported multimodal content type: {type(msg)}")

    def _process_str(self, msg: str) -> dict[str, Any]:
        if self._use_chat_completions:
            return {"type": "text", "text": msg}
        return {"type": "input_text", "text": msg}

    def _process_bytes(self, msg: bytes) -> dict[str, Any]:
        import base64

        mime_type = _guess_mime_type(msg)
        b64 = base64.b64encode(msg).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"

        if mime_type.startswith("image/"):
            return self._build_image_part(data_url)
        return self._build_file_part(data_url)

    def _process_ndarray(self, arr: Any) -> dict[str, Any]:
        import base64
        import io

        from PIL import Image

        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"
        return self._build_image_part(data_url)

    def _build_image_part(self, data_url: str) -> dict[str, Any]:
        if self._use_chat_completions:
            return {"type": "image_url", "image_url": {"url": data_url}}
        return {"type": "input_image", "image_url": data_url}

    def _build_file_part(self, data_url: str, filename: str = "file") -> dict[str, Any]:
        if self._use_chat_completions:
            return {
                "type": "file",
                "file": {"filename": filename, "file_data": data_url},
            }
        return {
            "type": "input_file",
            "filename": filename,
            "file_data": data_url,
        }

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
        if "max_tokens" not in options and "max_output_tokens" in options:
            options["max_tokens"] = options["max_output_tokens"]
        options.pop("max_output_tokens", None)
        return options

    def _responses_options(self) -> dict[str, Any]:
        options = dict(self._options)
        if "max_output_tokens" not in options and "max_tokens" in options:
            options["max_output_tokens"] = options["max_tokens"]
        options.pop("max_tokens", None)
        return options

    async def _prompt_chat_completions(self, messages: list[dict[str, Any]]) -> Any:
        """Prompt using the Chat Completions API."""
        options = self._chat_completions_options()
        if self._return_format is not None:
            response = await self._client.chat.completions.parse(
                model=self._model,
                messages=messages,
                response_format=self._return_format,
                **options,
            )
            result = response.choices[0].message.parsed
        else:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                **options,
            )
            result = response.choices[0].message.content
        self._record_usage(response)
        return result

    async def _prompt_responses(self, messages: list[dict[str, Any]]) -> Any:
        """Prompt using the Responses API."""
        options = self._responses_options()
        if self._return_format is not None:
            response = await self._client.responses.parse(
                model=self._model,
                input=messages,
                text_format=self._return_format,
                **options,
            )
            result = response.output_parsed
        else:
            response = await self._client.responses.create(
                model=self._model,
                input=messages,
                **options,
            )
            result = response.output_text
        self._record_usage(response)
        return result

    async def prompt(self, messages: tuple[Any, ...]) -> Any:
        """Generate a response for the given message(s).

        Each element of *messages* may be a ``str``, ``bytes`` (image),
        ``numpy.ndarray`` (image), or a pre-built ``dict``.

        - A ``dict`` with a ``"role"`` key is treated as a **complete message**
          and appended directly to the message list (e.g. assistant turns).
        - All other elements are assembled into a single user message with a
          content array (multimodal).

        Dispatches to Chat Completions or Responses API based on
        ``use_chat_completions``.  When ``return_format`` is set,
        uses ``parse()`` for structured output.
        """
        chat_messages: list[dict[str, Any]] = []
        if self._system_message:
            chat_messages.append({"role": "system", "content": self._system_message})

        # Separate complete messages (dicts with "role") from content parts
        content_parts: list[dict[str, Any]] = []

        def _flush_content() -> None:
            if not content_parts:
                return
            if len(content_parts) == 1 and content_parts[0].get("type") in (
                "text",
                "input_text",
            ):
                chat_messages.append({"role": "user", "content": content_parts[0]["text"]})
            else:
                chat_messages.append({"role": "user", "content": list(content_parts)})
            content_parts.clear()

        for msg in messages:
            if isinstance(msg, dict) and "role" in msg:
                _flush_content()
                chat_messages.append(msg)
            else:
                content_parts.append(self._process_message(msg))

        _flush_content()

        if self._use_chat_completions:
            return await self._prompt_chat_completions(chat_messages)
        return await self._prompt_responses(chat_messages)


def _guess_mime_type(data: bytes) -> str:
    """Guess MIME type from magic bytes. Returns image type or octet-stream."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:4] == b"GIF8":
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"%PDF":
        return "application/pdf"
    return "application/octet-stream"
