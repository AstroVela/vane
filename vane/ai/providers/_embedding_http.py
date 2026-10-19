# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared, worker-owned HTTP embedding lifecycle and bounded request planning."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from vane.ai._client_config import _text, copy_client_options
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec, MultimodalInputSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._embedding_requests import ManagedTextEmbedder, _EmbeddingBatchError, _is_request_wide_error
from vane.ai._redaction import unwrap_sensitive_options
from vane.ai.options import _HTTP_EMBED_OPTIONS, validate_embed_options
from vane.ai.protocols import ImageEmbedderDescriptor, MultimodalEmbedderDescriptor, TextEmbedderDescriptor
from vane.ai.provider import _SafeProviderError, _translate_missing_provider_dependency
from vane.ai.providers._embedding_http_codecs import EmbeddingHTTPCodec
from vane.ai.typing import UDFOptions


@dataclass
class _HTTPEmbeddingDescriptor:
    model: str
    dimensions: int | None
    options: dict[str, Any]
    provider_name: str
    family: str
    client_options: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise EmbeddingConfigurationError("HTTP embedding requires an explicit model name")
        if type(self.dimensions) is not int or self.dimensions <= 0:
            raise EmbeddingConfigurationError("HTTP embedding requires explicit positive dimensions")
        self.options = copy.deepcopy(validate_embed_options(self.family, self.options, relation=False))
        if self.options.get("runtime") != "http" or self.options.keys() - _HTTP_EMBED_OPTIONS:
            raise EmbeddingConfigurationError("HTTP embedding descriptor requires HTTP request options")
        self.client_options = copy_client_options(self.client_options)
        _text(unwrap_sensitive_options(self.client_options).get("api_key"), f"{self.family.upper()}_API_KEY")
        self._codec()  # All protocol/model metadata is checked without SDK imports or I/O.

    def _codec(self) -> EmbeddingHTTPCodec:
        return EmbeddingHTTPCodec(self.family, self.model, self.get_dimensions(), self.options)

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model

    def get_options(self) -> dict[str, Any]:
        return copy.deepcopy(self.options)

    def get_dimensions(self) -> int:
        assert self.dimensions is not None
        return self.dimensions

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(num_gpus=0)

    def is_async(self) -> bool:
        return True

    def instantiate(self) -> HTTPEmbedder:
        return HTTPEmbedder(self)


class HTTPTextEmbedderDescriptor(_HTTPEmbeddingDescriptor, TextEmbedderDescriptor):
    def supports_chunking(self) -> bool:
        return False


class HTTPImageEmbedderDescriptor(_HTTPEmbeddingDescriptor, ImageEmbedderDescriptor):
    def __post_init__(self) -> None:
        super().__post_init__()
        self._codec().spec.validate_layout((EmbeddingPartSpec("image"),))


class HTTPMultimodalEmbedderDescriptor(_HTTPEmbeddingDescriptor, MultimodalEmbedderDescriptor):
    def get_input_spec(self) -> MultimodalInputSpec:
        return self._codec().spec


class _HTTPStatusError(_SafeProviderError):
    def __init__(self, status: int, headers: Any, *, request_wide: bool):
        super().__init__(f"Embedding HTTP request failed with status {status}")
        self.status_code = status
        self.code = "INVALID_AUTHENTICATION" if request_wide else None
        self.response = SimpleNamespace(status_code=status, headers={"Retry-After": headers.get("Retry-After")})


class HTTPEmbedder(ManagedTextEmbedder):
    def __init__(self, descriptor: _HTTPEmbeddingDescriptor):
        self._codec = descriptor._codec()
        self._dimensions = descriptor.get_dimensions()
        options = descriptor.get_options()
        self._url = options["base_url"].rstrip("/") + (
            "/encode" if self._codec.input_format == "encode" else "/embeddings"
        )
        self._timeout = options.get("timeout", 60.0)
        self._request_concurrency = options.get("max_concurrency_per_actor", 4)
        # SGLang 0.5.17 can lose completed request state while preparing later
        # media rows in one request. Concurrent single-row requests still let
        # the server batch inference; qualified deployments can opt into more.
        default_batch_size = (
            1 if descriptor.family == "sglang" and self._codec.input_format in {"multimodal", "encode"} else 8
        )
        self._request_batch_size = options.get("request_batch_size", default_batch_size)
        self._max_request_bytes = options.get("max_request_bytes", 32 * 1024**2)
        self._client_options = copy_client_options(descriptor.client_options)
        self._client: Any = None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            with _translate_missing_provider_dependency("embedding_http", "httpx"):
                import httpx

            key = unwrap_sensitive_options(self._client_options).get("api_key")
            self._client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {key}"} if key else {},
                timeout=self._timeout,
                limits=httpx.Limits(
                    max_connections=self._request_concurrency, max_keepalive_connections=self._request_concurrency
                ),
                follow_redirects=False,
                trust_env=False,
            )
        return self._client

    async def embed_text(self, text: list[str]) -> list[Any]:
        return await self.embed_multimodal(
            [EmbeddingContent((EmbeddingPart(EmbeddingPartSpec("text"), value),)) for value in text]
        )

    async def embed_image(self, images: list[Any]) -> list[Any]:
        return await self.embed_multimodal(
            [EmbeddingContent((EmbeddingPart(EmbeddingPartSpec("image"), value),)) for value in images]
        )

    def _batches(self, rows: list[bytes]) -> Iterator[tuple[list[int], list[bytes]]]:
        overhead = len(self._codec.prefix) + len(self._codec.suffix)
        start, size = 0, overhead
        for end, row in enumerate(rows):
            extra = len(row) + int(end > start)
            if end > start and (end - start == self._request_batch_size or size + extra > self._max_request_bytes):
                yield list(range(start, end)), rows[start:end]
                start, size, extra = end, overhead, len(row)
            size += extra
        if start < len(rows):
            yield list(range(start, len(rows))), rows[start:]

    async def embed_multimodal(self, contents: list[EmbeddingContent]) -> list[Any]:
        # Validate EVERY row before client creation or dispatch. Local errors are
        # outside the provider-error handler, including Base64/JSON expansion.
        invalid = False
        try:
            rows = [self._codec.encode(content) for content in contents]
        except EmbeddingConfigurationError:
            raise
        except (TypeError, ValueError):
            invalid = True
        if invalid:
            raise EmbeddingConfigurationError("HTTP embedding input cannot be encoded")
        overhead = len(self._codec.prefix) + len(self._codec.suffix)
        if any(overhead + len(row) > self._max_request_bytes for row in rows):
            raise EmbeddingConfigurationError("HTTP embedding row exceeds the serialized request byte budget")
        return await self._run_requests(self._batches(rows), self._request, [None] * len(rows))

    def _vector(self, item: dict[str, Any]) -> Any:
        values = item.get("embedding")
        if (
            not isinstance(values, list)
            or len(values) != self._dimensions
            or any(type(value) not in {int, float} for value in values)
        ):
            raise ValueError("Invalid HTTP embedding vector")
        with np.errstate(over="ignore", invalid="ignore"):
            vector = np.asarray(values, dtype=np.float32)
        if not np.isfinite(vector).all():
            raise ValueError("Nonfinite HTTP embedding vector")
        return vector

    async def _request(self, rows: list[bytes]) -> list[Any]:
        from vane.ai.functions import _retry_after_error

        body = self._codec.body(rows)
        self.metrics.request_bytes += len(body)
        response = await self._get_client().post(self._url, content=body, headers={"Content-Type": "application/json"})
        # Never retain arbitrary response bodies in status errors or diagnostics.
        payload = None
        try:
            payload = response.json()
        except (ValueError, UnicodeError):
            pass
        if response.status_code != 200:
            error = _HTTPStatusError(
                response.status_code,
                response.headers,
                request_wide=_is_request_wide_error(SimpleNamespace(body=payload)),
            )
            retry = None if _is_request_wide_error(error) else _retry_after_error(error)
            raise retry or error
        if self._codec.input_format == "encode":
            data = [payload] if len(rows) == 1 and isinstance(payload, dict) else payload
            if not isinstance(data, list) or len(data) != len(rows) or any(not isinstance(item, dict) for item in data):
                raise _EmbeddingBatchError("SGLang encode response must preserve input row count and order")
            # Native /encode returns input-ordered results rather than indices.
            for item in data:
                meta = item.get("meta_info")
                tokens = meta.get("prompt_tokens") if isinstance(meta, dict) else None
                if type(tokens) is int and tokens >= 0:
                    self.metrics.input_tokens += tokens
            return self._decode_response_vectors(data, self._vector)
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != len(rows):
            raise _EmbeddingBatchError("HTTP embedding response must preserve input row count")
        assert isinstance(payload, dict)
        raw_indices = [item.get("index") if isinstance(item, dict) else None for item in data]
        if any(type(index) is not int for index in raw_indices):
            raise _EmbeddingBatchError("HTTP embedding response requires integer input indices")
        indices = cast(list[int], raw_indices)
        if sorted(indices) != list(range(len(rows))):
            raise _EmbeddingBatchError("HTTP embedding response requires a permutation of input indices")
        data = [item for _, item in sorted(zip(indices, data, strict=True), key=lambda pair: pair[0])]
        usage = payload.get("usage", {})
        tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if type(tokens) is int and tokens >= 0:
            self.metrics.input_tokens += tokens
        return self._decode_response_vectors(data, self._vector)
