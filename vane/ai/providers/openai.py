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
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa

from vane.ai._redaction import unwrap_sensitive_options, wrap_sensitive_options
from vane.ai._schema import (
    _is_known_openai_prompt_model,
    _openai_structured_outputs_capability,
    _uses_openai_strict_structured_outputs,
    serialize_raw_response,
)
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
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_OPENAI_OFFICIAL_API_HOSTS = frozenset(
    {
        "ae.api.openai.com",
        "api.openai.com",
        "au.api.openai.com",
        "ca.api.openai.com",
        "eu.api.openai.com",
        "gb.api.openai.com",
        "in.api.openai.com",
        "jp.api.openai.com",
        "kr.api.openai.com",
        "sg.api.openai.com",
        "us.api.openai.com",
    }
)

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
_OPENAI_RESPONSES_ONLY_MODELS = frozenset(
    {
        "o1-pro",
        "o1-pro-2025-03-19",
        "o3-pro",
        "o3-pro-2025-06-10",
        "gpt-5-pro",
        "gpt-5-pro-2025-10-06",
    }
)
_OPENAI_CHAT_COMPLETIONS_ONLY_MODELS = frozenset(
    {
        "gpt-4o-search-preview",
        "gpt-4o-search-preview-2025-03-11",
        "gpt-4o-mini-search-preview",
        "gpt-4o-mini-search-preview-2025-03-11",
    }
)
_OPENAI_TEXT_ONLY_PROMPT_MODELS = frozenset(
    {
        "o1-mini",
        "o1-mini-2024-09-12",
        "o1-preview",
        "o1-preview-2024-09-12",
        "o3-mini",
        "o3-mini-2025-01-31",
        "gpt-4o-search-preview",
        "gpt-4o-search-preview-2025-03-11",
        "gpt-4o-mini-search-preview",
        "gpt-4o-mini-search-preview-2025-03-11",
    }
)


def _uses_official_openai_endpoint(base_url: Any) -> bool:
    if base_url is None:
        return True
    if not isinstance(base_url, str):
        return False
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.casefold() == "https"
        and parsed.hostname is not None
        and parsed.hostname.casefold() in _OPENAI_OFFICIAL_API_HOSTS
        and port in {None, 443}
        and parsed.path.rstrip("/") == "/v1"
        and not parsed.query
        and not parsed.fragment
        and parsed.username is None
        and parsed.password is None
    )


def _uses_openai_max_completion_tokens(model: str) -> bool:
    normalized = model.strip().casefold()
    return (
        normalized == "gpt-5"
        or normalized.startswith(("gpt-5-", "gpt-5."))
        or (
            len(normalized) >= 2
            and normalized[0] == "o"
            and normalized[1].isdigit()
            and (len(normalized) == 2 or normalized[2] in {"-", "."})
        )
    )


def _validate_openai_prompt_capabilities(
    model: str,
    options: Mapping[str, Any],
    *,
    return_format: dict[str, Any] | None,
) -> None:
    """Reject only documented conflicts for known models on the official API."""
    if not _uses_official_openai_endpoint(options.get("base_url")):
        return
    normalized = model.strip().casefold()
    if normalized in _MODEL_DIMS:
        raise ValueError(f"OpenAI model {model!r} supports Embed, not Prompt")
    use_chat_completions = bool(options.get("use_chat_completions", False))
    if use_chat_completions and normalized in _OPENAI_RESPONSES_ONLY_MODELS:
        raise ValueError(f"OpenAI model {model!r} is available through the Responses API only")
    if not use_chat_completions and normalized in _OPENAI_CHAT_COMPLETIONS_ONLY_MODELS:
        raise ValueError(f"OpenAI model {model!r} is available through Chat Completions only")
    if return_format is not None and _openai_structured_outputs_capability(model) == "unsupported":
        raise ValueError(f"OpenAI model {model!r} does not support structured Prompt output")


def _decode_openai_embedding_base64(value: str) -> np.ndarray:
    raw = base64.b64decode(value)
    return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)


def _get_input_token_limit(model: str) -> int:
    """Return the per-input token limit for *model*, defaulting to 8192."""
    return _MODEL_INPUT_TOKEN_LIMITS.get(model, _DEFAULT_INPUT_TOKEN_LIMIT)


class _TokenEstimator:
    def __init__(self, encoding: Any | None = None) -> None:
        self.encoding = encoding
        self._max_token_bytes: int | None = None

    def __call__(self, value: str) -> int:
        if self.encoding is None:
            return len(value.encode("utf-8"))
        return len(self.encoding.encode_ordinary(value))

    def max_token_bytes(self) -> int | None:
        if self.encoding is None or not isinstance(getattr(self.encoding, "n_vocab", None), int):
            return None
        if self._max_token_bytes is None:
            lengths: list[int] = []
            for token in range(self.encoding.n_vocab):
                try:
                    lengths.append(len(self.encoding.decode_single_token_bytes(token)))
                except KeyError:
                    continue
            self._max_token_bytes = max(lengths)
        return self._max_token_bytes


def _build_token_estimator(model: str, *, use_openai_tokenizer: bool) -> _TokenEstimator:
    """Return one estimator shared by request batching and input chunking."""
    if not use_openai_tokenizer:
        return _TokenEstimator()

    import tiktoken

    return _TokenEstimator(tiktoken.encoding_for_model(model))


def _chunk_tokenized_text(text: str, limit: int, estimate_tokens: _TokenEstimator) -> list[str]:
    """Split with tokenizer guidance while preserving Unicode boundaries."""
    encoding = estimate_tokens.encoding
    assert encoding is not None
    chunks: list[str] = []
    remaining = text
    while remaining:
        tokens = encoding.encode_ordinary(remaining)
        if len(tokens) <= limit:
            chunks.append(remaining)
            break

        guided_bytes = b"".join(encoding.decode_single_token_bytes(token) for token in tokens[:limit])
        candidate = guided_bytes.decode("utf-8", errors="ignore")
        while candidate and estimate_tokens(candidate) > limit:
            candidate = candidate[:-1]

        if not candidate:
            byte_budget = estimate_tokens.max_token_bytes()
            max_bytes = byte_budget * limit if byte_budget is not None else None
            prefix_bytes = 0
            for end, character in enumerate(remaining, start=1):
                prefix_bytes += len(character.encode("utf-8"))
                if max_bytes is not None and prefix_bytes > max_bytes:
                    break
                prefix = remaining[:end]
                if estimate_tokens(prefix) <= limit:
                    candidate = prefix
                    break

        if not candidate:
            raise ValueError(
                "OpenAI embedding token limit is too small for one input character; increase the configured limit"
            )
        chunks.append(candidate)
        remaining = remaining[len(candidate) :]
    return chunks


def _chunk_text_by_token_limit(text: str, limit: int, estimate_tokens: Any) -> list[str]:
    """Split text on character boundaries without exceeding a token estimate."""
    if isinstance(estimate_tokens, _TokenEstimator) and estimate_tokens.encoding is not None:
        return _chunk_tokenized_text(text, limit, estimate_tokens)

    chunks: list[str] = []
    start = 0
    while start < len(text):
        if estimate_tokens(text[start : start + 1]) > limit:
            raise ValueError(
                "OpenAI embedding token limit is too small for one input character; increase the configured limit"
            )
        low = start + 1
        high = len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if estimate_tokens(text[start:middle]) <= limit:
                low = middle
            else:
                high = middle - 1
        chunks.append(text[start:low])
        start = low
    return chunks


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


def _structured_output_name(schema: dict[str, Any]) -> str:
    title = schema.get("title")
    if (
        isinstance(title, str)
        and 1 <= len(title) <= 64
        and all(character.isascii() and (character.isalnum() or character in "_-") for character in title)
    ):
        return title
    return "vane_response"


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

    def __init__(self, name: str | None = None):
        self._name = name or "openai"

    @property
    def name(self) -> str:
        return self._name

    _EMBED_OPTIONS = {"base_url", "timeout", "encoding_format", "batch_token_limit", "input_text_token_limit"}
    _PROMPT_OPTIONS = {
        "base_url",
        "timeout",
        "use_chat_completions",
        "temperature",
        "max_output_tokens",
        "top_p",
        "stop_sequences",
    }

    def get_text_embedder(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> TextEmbedderDescriptor:
        resolved_options = dict(options or {})
        return OpenAITextEmbedderDescriptor(
            provider_name=self._name,
            model_name=model or self.DEFAULT_TEXT_EMBEDDER,
            dimensions=dimensions,
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
        resolved_options = dict(options or {})
        return OpenAIPrompterDescriptor(
            provider_name=self._name,
            model_name=model or self.DEFAULT_PROMPTER_MODEL,
            system_message=system_message,
            return_format=return_format,
            return_raw_response=return_raw_response,
            options=resolved_options,
        )


# ---------------------------------------------------------------------------
# Text Embedding
# ---------------------------------------------------------------------------


@dataclass
class OpenAITextEmbedderDescriptor(TextEmbedderDescriptor):
    """Serializable factory for an OpenAI text embedder."""

    provider_name: str = "openai"
    model_name: str = "text-embedding-3-small"
    dimensions: int | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("OpenAI embedding model must be a non-empty string")
        unknown = sorted(set(self.options) - OpenAIProvider._EMBED_OPTIONS)
        if unknown:
            raise TypeError(f"Unsupported OpenAI Embed option(s): {', '.join(unknown)}")
        validated_options = validate_embed_options("openai", self.options, relation=False)
        normalized_model = self.model_name.strip().casefold()
        official_endpoint = _uses_official_openai_endpoint(validated_options.get("base_url"))
        if official_endpoint and _is_known_openai_prompt_model(normalized_model):
            raise ValueError(f"OpenAI model {self.model_name!r} supports Prompt, not Embed")
        if self.dimensions is not None and (
            isinstance(self.dimensions, bool) or not isinstance(self.dimensions, int) or self.dimensions <= 0
        ):
            raise ValueError("Embedding dimensions must be a positive integer")
        if (
            official_endpoint
            and self.dimensions is not None
            and normalized_model in _MODEL_DIMS
            and normalized_model not in _DIMENSION_OVERRIDABLE
        ):
            raise ValueError(f"Model {self.model_name!r} does not support custom dimensions")
        if (
            official_endpoint
            and self.dimensions is not None
            and normalized_model in _DIMENSION_OVERRIDABLE
            and self.dimensions > _MODEL_DIMS[normalized_model]
        ):
            raise ValueError(
                f"Model {self.model_name!r} supports at most {_MODEL_DIMS[normalized_model]} dimensions, "
                f"got {self.dimensions}"
            )
        if not official_endpoint and self.dimensions is None:
            raise ValueError(
                f"Cannot determine embedding dimensions for OpenAI-compatible model {self.model_name!r} "
                "from trusted local metadata; pass dimensions=... explicitly"
            )
        self.options = _wrap_openai_options(validated_options)

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> Options:
        return dict(self.options)

    def get_dimensions(self) -> EmbeddingDimensions:
        if self.dimensions is not None:
            return EmbeddingDimensions(size=self.dimensions, dtype=pa.float32())
        normalized_model = self.model_name.strip().casefold()
        if _uses_official_openai_endpoint(self.options.get("base_url")) and normalized_model in _MODEL_DIMS:
            return EmbeddingDimensions(size=_MODEL_DIMS[normalized_model], dtype=pa.float32())
        raise ValueError(
            f"Cannot determine embedding dimensions for OpenAI-compatible model {self.model_name!r} "
            "from trusted local metadata; pass dimensions=... explicitly"
        )

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(num_gpus=0)

    def is_async(self) -> bool:
        return True

    def instantiate(self) -> TextEmbedder:
        return OpenAITextEmbedder(
            options=self.options,
            provider_name=self.provider_name,
            model=self.model_name,
            dimensions=self.dimensions,
        )


class OpenAITextEmbedder:
    """Async text embedder using the OpenAI Embeddings API.

    Two-level token limiting:

    * **batch_token_limit** — max estimated tokens per API request (default 300k).
    * **input_text_token_limit** — max tokens for a single input text.
      Texts exceeding this are split into character chunks, embedded
      separately, and recombined via token-weighted averaging + L2
      normalisation. Known OpenAI embedding models use their model-specific
      tokenizer; unknown compatible models use UTF-8 byte length as a safe
      upper bound.
    """

    def __init__(
        self,
        options: dict[str, Any],
        model: str,
        dimensions: int | None = None,
        provider_name: str = "openai",
    ):
        from openai import AsyncOpenAI

        options = unwrap_sensitive_options(options)
        encoding_format = options.get("encoding_format", "float")
        if encoding_format not in {"float", "base64"}:
            raise ValueError("encoding_format must be 'float' or 'base64'")
        self._provider_name = provider_name
        self._model = model
        self._dimensions = dimensions
        self._encoding_format = encoding_format
        self._batch_token_limit = options.get("batch_token_limit", 300_000)
        input_text_token_limit = options.get("input_text_token_limit")
        self._input_text_token_limit = (
            input_text_token_limit if input_text_token_limit is not None else _get_input_token_limit(model)
        )
        self._estimate_tokens = _build_token_estimator(
            model,
            use_openai_tokenizer=(
                model in _MODEL_INPUT_TOKEN_LIMITS and _uses_official_openai_endpoint(options.get("base_url"))
            ),
        )
        client_opts = {
            "base_url": options.get("base_url") or _OPENAI_DEFAULT_BASE_URL,
            **({"timeout": options["timeout"]} if options.get("timeout") is not None else {}),
        }
        # Retries belong to Vane's row-aware wrapper, so the SDK must not
        # stack its own retries underneath the public max_retries contract.
        client_opts["max_retries"] = 0
        self._client = AsyncOpenAI(**client_opts)

    async def aclose(self) -> None:
        """Release the SDK client's connection pool on the owning loop."""
        await self._client.close()

    async def embed_text(self, text: list[str]) -> list[Embedding]:
        embeddings: list[Embedding] = []
        batch: list[str] = []
        batch_tokens = 0
        estimate_tokens = self._estimate_tokens

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
                chunks = _chunk_text_by_token_limit(item, single_input_limit, estimate_tokens)
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
                    [estimate_tokens(c) for c in chunks],
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

        capability_error: ProviderCapabilityError | None = None
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
                capability_error = ProviderCapabilityError(
                    getattr(self, "_provider_name", "openai"),
                    self._model,
                    "embedding endpoint/model",
                    original_error=ex,
                )
            else:
                raise
        if capability_error is not None:
            raise capability_error from None
        raise AssertionError("OpenAI embedding request completed without a result")


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


@dataclass
class OpenAIPrompterDescriptor(PrompterDescriptor):
    """Serializable factory for a basic text/image OpenAI prompter."""

    provider_name: str = "openai"
    model_name: str = "gpt-4o-mini"
    system_message: str | None = None
    return_format: dict[str, Any] | None = None
    return_raw_response: bool = False
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("OpenAI prompt model must be a non-empty string")
        unknown = sorted(set(self.options) - OpenAIProvider._PROMPT_OPTIONS)
        if unknown:
            raise TypeError(f"Unsupported OpenAI Prompt option(s): {', '.join(unknown)}")
        validated_options = validate_prompt_options("openai", self.options, relation=False)
        if validated_options.get("stop_sequences") is not None and not validated_options.get(
            "use_chat_completions", False
        ):
            raise ValueError("OpenAI stop_sequences requires use_chat_completions=True")
        _validate_openai_prompt_capabilities(
            self.model_name,
            validated_options,
            return_format=self.return_format,
        )
        self.options = _wrap_openai_options(validated_options)

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> Options:
        return dict(self.options)

    def supports_strict_structured_outputs(self) -> bool:
        return _uses_official_openai_endpoint(self.options.get("base_url")) and _uses_openai_strict_structured_outputs(
            self.model_name
        )

    def supports_image_inputs(self) -> bool:
        return not (
            _uses_official_openai_endpoint(self.options.get("base_url"))
            and self.model_name.strip().casefold() in _OPENAI_TEXT_ONLY_PROMPT_MODELS
        )

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(num_gpus=0)

    def instantiate(self) -> Prompter:
        return OpenAIPrompter(
            options=self.options,
            provider_name=self.provider_name,
            model=self.model_name,
            system_message=self.system_message,
            return_format=self.return_format,
            return_raw_response=self.return_raw_response,
            strict_structured_outputs=self.supports_strict_structured_outputs(),
        )


class OpenAIPrompter:
    """Async basic text/image prompter for Responses or Chat Completions."""

    def __init__(
        self,
        options: dict[str, Any],
        model: str,
        system_message: str | None = None,
        return_format: dict[str, Any] | None = None,
        return_raw_response: bool = False,
        provider_name: str = "openai",
        strict_structured_outputs: bool | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        options = unwrap_sensitive_options(options)
        self._provider_name = provider_name
        self._model = model
        self._system_message = system_message
        self._use_chat_completions = bool(options.get("use_chat_completions", False))
        self._return_format = return_format
        self._return_raw_response = return_raw_response
        self._official_openai_endpoint = _uses_official_openai_endpoint(options.get("base_url"))
        self._strict_structured_outputs = (
            self._official_openai_endpoint and _uses_openai_strict_structured_outputs(model)
            if strict_structured_outputs is None
            else strict_structured_outputs
        )
        self._options = {
            key: value
            for key, value in options.items()
            if key in {"temperature", "max_output_tokens", "top_p", "stop_sequences"} and value is not None
        }
        client_opts = {
            "base_url": options.get("base_url") or _OPENAI_DEFAULT_BASE_URL,
            **({"timeout": options["timeout"]} if options.get("timeout") is not None else {}),
        }
        client_opts["max_retries"] = 0
        self._client = AsyncOpenAI(**client_opts)

    async def aclose(self) -> None:
        """Release the SDK client's connection pool on the owning loop."""
        await self._client.close()

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
            token_limit_name = (
                "max_completion_tokens"
                if getattr(self, "_official_openai_endpoint", False) and _uses_openai_max_completion_tokens(self._model)
                else "max_tokens"
            )
            options[token_limit_name] = options["max_output_tokens"]
        options.pop("max_output_tokens", None)
        if "stop_sequences" in options:
            options["stop"] = options.pop("stop_sequences")
        return_format = getattr(self, "_return_format", None)
        if return_format is not None:
            json_schema = {
                "name": _structured_output_name(return_format),
                "schema": return_format,
            }
            if getattr(
                self,
                "_strict_structured_outputs",
                _uses_openai_strict_structured_outputs(self._model),
            ):
                json_schema["strict"] = True
            options["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
        return options

    def _responses_options(self) -> dict[str, Any]:
        options = dict(self._options)
        return_format = getattr(self, "_return_format", None)
        if return_format is not None:
            response_format = {
                "type": "json_schema",
                "name": _structured_output_name(return_format),
                "schema": return_format,
            }
            if getattr(
                self,
                "_strict_structured_outputs",
                _uses_openai_strict_structured_outputs(self._model),
            ):
                response_format["strict"] = True
            options["text"] = {"format": response_format}
        return options

    async def _prompt_chat_completions(self, messages: list[dict[str, Any]]) -> str | None:
        """Prompt using the Chat Completions API."""
        options = self._chat_completions_options()
        capability_error: ProviderCapabilityError | None = None
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                **options,
            )
        except Exception as exc:
            if _is_prompt_capability_error(exc):
                capability_error = ProviderCapabilityError(
                    self._provider_name,
                    self._model,
                    self._requested_capability(),
                    original_error=exc,
                )
            else:
                raise
        if capability_error is not None:
            raise capability_error from None
        self._record_usage(response)
        if getattr(self, "_return_raw_response", False):
            return serialize_raw_response(response)
        return response.choices[0].message.content

    async def _prompt_responses(self, messages: list[dict[str, Any]]) -> str | None:
        """Prompt using the Responses API."""
        options = self._responses_options()
        capability_error: ProviderCapabilityError | None = None
        try:
            response = await self._client.responses.create(
                model=self._model,
                input=messages,
                **options,
            )
        except Exception as exc:
            if _is_prompt_capability_error(exc):
                capability_error = ProviderCapabilityError(
                    self._provider_name,
                    self._model,
                    self._requested_capability(),
                    original_error=exc,
                )
            else:
                raise
        if capability_error is not None:
            raise capability_error from None
        self._record_usage(response)
        if getattr(self, "_return_raw_response", False):
            return serialize_raw_response(response)
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
