# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Serializable descriptors and CPU-only clients for a local native engine."""

from __future__ import annotations

import asyncio
import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vane.ai import _embedding_ipc as ipc
from vane.ai._audio_embedding import AudioClip, AudioInputSpec
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec, MultimodalInputSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._embedding_requests import ManagedEmbedder
from vane.ai._errors import _raise_if_resource_error
from vane.ai._video_embedding import VideoClip, VideoInputSpec
from vane.ai.embedding import Modality
from vane.ai.options import _SERVICE_EMBED_OPTIONS
from vane.ai.protocols import (
    AudioEmbedderDescriptor,
    ImageEmbedderDescriptor,
    MultimodalEmbedderDescriptor,
    TextEmbedderDescriptor,
    VideoEmbedderDescriptor,
)
from vane.ai.provider import ProviderImportError, _ProviderResultError
from vane.ai.typing import UDFOptions
from vane.execution._async_dispatch import map_async_ordered


@dataclass
class _ServiceDescriptor:
    model: str
    dimensions: int | None
    options: dict[str, Any]
    provider_name: str
    family: str

    def __post_init__(self) -> None:
        self.options = copy.deepcopy(self.options)
        if self.options.keys() - _SERVICE_EMBED_OPTIONS or self.options.get("runtime") != "service":
            raise EmbeddingConfigurationError("Embedding service requires explicit service options")
        if (
            not isinstance(self.model, str)
            or not self.model.strip()
            or len(self.model) > 4096
            or type(self.dimensions) is not int
            or not 1 <= self.dimensions <= 16384
        ):
            raise EmbeddingConfigurationError("Embedding service requires model and bounded positive dimensions")
        path, identity = self.options.get("service_path"), self.options.get("service_id")
        if not isinstance(path, str) or not Path(path).is_absolute() or "\x00" in path:
            raise EmbeddingConfigurationError("service_path must be an absolute local path")
        if (
            not isinstance(identity, str)
            or not 1 <= len(identity) <= 128
            or not identity.isascii()
            or not all(ch.isalnum() or ch in "-_." for ch in identity)
        ):
            raise EmbeddingConfigurationError("service_id requires 1 to 128 ASCII letters, digits, '-', '_' or '.'")
        timeout = self.options.setdefault("timeout", 60.0)
        concurrency = self.options.setdefault("max_concurrency_per_actor", 8)
        if type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
            raise EmbeddingConfigurationError("Embedding service timeout must be finite and positive")
        if type(concurrency) is not int or not 1 <= concurrency <= 1024:
            raise EmbeddingConfigurationError("Embedding service concurrency must be between 1 and 1024")
        ipc.input_spec(self.options)

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model

    def get_dimensions(self) -> int:
        assert self.dimensions is not None
        return self.dimensions

    def get_options(self) -> dict[str, Any]:
        return copy.deepcopy(self.options)

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(num_gpus=0)

    def is_async(self) -> bool:
        return True

    def get_service_spec(self) -> MultimodalInputSpec:
        return ipc.input_spec(self.options)

    def get_request_options(self) -> dict[str, Any] | None:
        return None

    def instantiate(self) -> ServiceEmbedder:
        return ServiceEmbedder(self)


class ServiceTextDescriptor(_ServiceDescriptor, TextEmbedderDescriptor):
    def __post_init__(self) -> None:
        super().__post_init__()
        ipc.input_spec(self.options).validate_layout((EmbeddingPartSpec("text"),))

    def supports_chunking(self) -> bool:
        return False


class ServiceImageDescriptor(_ServiceDescriptor, ImageEmbedderDescriptor):
    def __post_init__(self) -> None:
        super().__post_init__()
        ipc.input_spec(self.options).validate_layout((EmbeddingPartSpec("image"),))


class ServiceVideoDescriptor(_ServiceDescriptor, VideoEmbedderDescriptor):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.get_input_spec()

    def get_input_spec(self) -> VideoInputSpec:
        spec = ipc.input_spec(self.options).video
        if spec is None:
            raise EmbeddingConfigurationError("Video requires video in input_modalities")
        return spec


class ServiceAudioDescriptor(_ServiceDescriptor, AudioEmbedderDescriptor):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.get_input_spec()

    def get_input_spec(self) -> AudioInputSpec:
        spec = ipc.input_spec(self.options).audio
        if spec is None:
            raise EmbeddingConfigurationError("Audio requires audio in input_modalities")
        return spec


class ServiceMultimodalDescriptor(_ServiceDescriptor, MultimodalEmbedderDescriptor):
    def get_input_spec(self) -> MultimodalInputSpec:
        return ipc.input_spec(self.options)


class ServiceEmbedder(ManagedEmbedder):
    def __init__(self, descriptor: _ServiceDescriptor):
        self.descriptor = descriptor
        self.spec = descriptor.get_service_spec()
        self.options = descriptor.get_options()
        self.closed = False
        self.metrics = {"queue_seconds": 0.0, "copy_seconds": 0.0, "request_seconds": 0.0, "input_bytes": 0}
        self._zmq_client: Any = None

    async def aclose(self) -> None:
        self.closed = True
        if self._zmq_client is not None:
            self._zmq_client.close()
            self._zmq_client = None

    async def embed_text(self, values: list[str]) -> list[Any]:
        return await self._single("text", values)

    async def embed_image(self, values: list[Any]) -> list[Any]:
        return await self._single("image", values)

    async def embed_video(self, values: list[VideoClip]) -> list[Any]:
        return await self._single("video", values)

    async def embed_audio(self, values: list[AudioClip]) -> list[Any]:
        return await self._single("audio", values)

    async def _single(self, kind: Modality, values: list[Any]) -> list[Any]:
        return await self.embed_multimodal(
            [EmbeddingContent((EmbeddingPart(EmbeddingPartSpec(kind), value),)) for value in values]
        )

    async def embed_multimodal(self, contents: list[EmbeddingContent]) -> list[Any]:
        from vane.ai.functions import _log_substituted_failure

        # Entire batch validation precedes any connection/model work, even when
        # on_error='ignore'. No replay/retry after uncertain service completion.
        plans = [ipc.describe(content, self.spec) for content in contents]
        for parts, _, size in plans:
            ipc.encode(self._reservation(parts, size, "0" * 32))

        async def invoke(plan: Any, check_stopped: Callable[[], None]) -> Any:
            check_stopped()
            try:
                result = await asyncio.wait_for(self._request(plan), self.options["timeout"])
                return getattr(self, "_validate_vector", lambda value: value)(result)
            except (EmbeddingConfigurationError, MemoryError, ProviderImportError, _ProviderResultError):
                raise
            except Exception as error:
                _raise_if_resource_error(error)
                if getattr(self, "_request_on_error", "raise") == "raise":
                    raise
                _log_substituted_failure(error, on_error="ignore")
                return None

        return await map_async_ordered(plans, invoke, max_concurrency=self.options["max_concurrency_per_actor"])

    async def _request(self, plan: Any) -> Any:
        if self.closed:
            raise RuntimeError("Embedding service client is closed")
        from vane.ai._embedding_zmq import Client

        if self._zmq_client is None:
            self._zmq_client = Client(self)
        return await self._zmq_client.request(plan)

    @staticmethod
    def _check_response(response: Any, request_id: str) -> None:
        if not isinstance(response, dict) or response.get("request_id") != request_id:
            raise EmbeddingConfigurationError("Embedding service response identity mismatch")
        if response.get("error") == "configuration":
            raise EmbeddingConfigurationError("Embedding service rejected input or model/service configuration")
        if response.get("error") == "resource":
            raise MemoryError("Embedding service exhausted model or host resources")
        if response.get("error") == "result":
            raise _ProviderResultError("Embedding service returned an invalid model result")
        if "error" in response:
            raise RuntimeError("Embedding service request failed")

    def _reservation(self, parts: Any, size: int, request_id: str) -> dict[str, Any]:
        reservation = {
            "version": ipc.VERSION,
            "service_id": self.options["service_id"],
            "family": self.descriptor.family,
            "model": self.descriptor.model,
            "dimensions": self.descriptor.dimensions,
            "request_id": request_id,
            "parts": parts,
            "nbytes": size,
        }
        encoding = self.descriptor.get_request_options()
        if encoding is not None:
            reservation["encoding"] = encoding
        return reservation
