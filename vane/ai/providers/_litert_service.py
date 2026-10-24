# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""LiteRT service clients and the service's single SDK-owning thread."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Literal

import numpy as np

from vane.ai._audio_embedding import AudioInputSpec
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPartSpec, MultimodalInputSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._embedding_outputs import EmbeddingOutput, EmbeddingOutputSpec
from vane.ai._litert_options import LITERT_MODELS, validate_litert_model, validate_litert_options
from vane.ai.options import _SERVICE_EMBED_OPTIONS
from vane.ai.protocols import (
    AudioEmbedderDescriptor,
    ImageEmbedderDescriptor,
    MultimodalEmbedderDescriptor,
    TextEmbedderDescriptor,
)
from vane.ai.provider import _ProviderResultError
from vane.ai.providers._embedding_service import ServiceEmbedder, _ServiceDescriptor


def encoding_identity(options: Mapping[str, Any]) -> dict[str, Any]:
    return {name: options[name] for name in ("model_sha256", "max_length", "vision_tokens_per_image")}


class _LiteRTServiceDescriptor(_ServiceDescriptor):
    kind: Literal["text", "image", "audio", "multimodal"] = "text"

    def __init__(self, model: str | None, dimensions: int | None, options: Mapping[str, Any]) -> None:
        model, dimensions = validate_litert_model(model, dimensions, self.kind)
        prepared = validate_litert_options(options, text=self.kind == "text", service=True)
        endpoint = {key: value for key, value in prepared.items() if key in _SERVICE_EMBED_OPTIONS}
        super().__init__(model, dimensions, endpoint, "litert", "litert")
        self.options = prepared | self.options

    def instantiate(self) -> LiteRTServiceEmbedder:
        return LiteRTServiceEmbedder(self)

    def get_request_options(self) -> dict[str, Any]:
        return encoding_identity(self.options) | {"input_type": self.options.get("input_type", "document")}

    def get_transport_input_spec(self) -> MultimodalInputSpec:
        kinds = LITERT_MODELS[self.model] if self.kind == "multimodal" else (self.kind,)
        limit = self.options["max_input_bytes"]
        return MultimodalInputSpec(
            frozenset(EmbeddingPartSpec(kind) for kind in kinds),
            audio=AudioInputSpec(16000, 16000 * 180, max_channels=1, max_input_bytes=limit)
            if "audio" in kinds
            else None,
            max_input_bytes=limit,
        )

    def get_output_spec(self, *, dimensions: int | None = None) -> EmbeddingOutputSpec:
        if dimensions is not None and (type(dimensions) is not int or dimensions != self.dimensions):
            raise EmbeddingConfigurationError("LiteRT output dimensions must match the descriptor")
        kinds = LITERT_MODELS[self.model] if self.kind == "multimodal" else (self.kind,)
        return EmbeddingOutputSpec(
            (
                EmbeddingOutput(
                    "embedding", self.get_dimensions(), self.model + ":" + self.options["model_sha256"], kinds
                ),
            )
        )


class LiteRTServiceTextDescriptor(_LiteRTServiceDescriptor, TextEmbedderDescriptor):
    def supports_chunking(self) -> bool:
        return False


class LiteRTServiceImageDescriptor(_LiteRTServiceDescriptor, ImageEmbedderDescriptor):
    kind: Literal["text", "image", "audio", "multimodal"] = "image"


class LiteRTServiceAudioDescriptor(_LiteRTServiceDescriptor, AudioEmbedderDescriptor):
    kind: Literal["text", "image", "audio", "multimodal"] = "audio"

    def get_input_spec(self) -> AudioInputSpec:
        spec = self.get_transport_input_spec().audio
        assert spec is not None
        return spec


class LiteRTServiceMultimodalDescriptor(_LiteRTServiceDescriptor, MultimodalEmbedderDescriptor):
    kind: Literal["text", "image", "audio", "multimodal"] = "multimodal"

    def get_input_spec(self) -> MultimodalInputSpec:
        return self.get_transport_input_spec()


class LiteRTServiceEmbedder(ServiceEmbedder):
    async def _request(self, plan: Any) -> Any:
        try:
            values = await super()._request(plan)
        except (EmbeddingConfigurationError, _ProviderResultError):
            raise
        except (TypeError, ValueError, OverflowError):
            raise _ProviderResultError("LiteRT service returned an invalid embedding") from None
        norm = np.linalg.norm(values.astype(np.float64))
        if not np.isfinite(norm) or norm <= 0 or abs(norm - 1) > 1e-3:
            raise _ProviderResultError("LiteRT service embeddings must have a nonzero unit norm within 1e-3")
        return (values / norm).astype(np.float32)


async def _drain(future: asyncio.Future[Any]) -> Any:
    """An SDK call owns its input until the sole SDK thread finishes."""
    while True:
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            if future.done():
                return future.result()


class LiteRTServiceRuntime:
    def __init__(self, descriptor: Any) -> None:
        self.descriptor = descriptor
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vane-litert-service")
        self.initialization: asyncio.Future[Any] | None = None
        self.closing: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self.initialization is not None or self.closing is not None:
            raise RuntimeError("LiteRT service runtime is already started or closed")
        self.initialization = asyncio.get_running_loop().run_in_executor(self.executor, self.descriptor.instantiate)
        await asyncio.shield(self.initialization)

    async def _embed_content(self, content: EmbeddingContent, *, input_type: str) -> np.ndarray:
        if self.closing is not None or self.initialization is None or not self.initialization.done():
            raise RuntimeError("LiteRT service runtime is not running")
        runtime = self.initialization.result()
        rows = [[(part.spec.kind, part.value) for part in content.parts]]
        future = asyncio.get_running_loop().run_in_executor(self.executor, partial(runtime._run, rows, role=input_type))
        return (await _drain(future))[0]

    async def _close(self) -> None:
        if self.initialization is not None:
            try:
                runtime = await _drain(self.initialization)
            except BaseException:
                pass  # Startup owns the original initialization error.
            else:
                await _drain(asyncio.get_running_loop().run_in_executor(self.executor, runtime.close))
        self.executor.shutdown(wait=True)

    async def aclose(self) -> None:
        closing = self.closing
        if closing is None or closing.done() and (closing.cancelled() or closing.exception() is not None):
            closing = self.closing = asyncio.create_task(self._close())
        await _drain(closing)
