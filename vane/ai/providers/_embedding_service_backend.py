# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Model ownership behind the shared embedding service lifecycle.

Construction validates metadata only. SDKs and model resources belong to start(),
after the transport has exclusively reserved its rendezvous path.
"""

from __future__ import annotations

from typing import Any, Protocol

from vane.ai._embedding_content import EmbeddingContent, MultimodalInputSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError


class ServiceBackend(Protocol):
    descriptor: Any

    def get_input_spec(self) -> MultimodalInputSpec: ...

    async def start(self) -> None: ...

    async def embed(self, content: EmbeddingContent) -> Any: ...

    async def aclose(self) -> None: ...


class NativeServiceBackend:
    """Delegate SGLang/vLLM encoding and teardown to their existing runtime."""

    def __init__(self, family: str, model: str, dimensions: int, options: dict[str, Any]):
        from vane.ai.providers._native_inference import NativeMultimodalEmbedderDescriptor

        self.descriptor = NativeMultimodalEmbedderDescriptor(model, dimensions, options, family, family)
        self.runtime: Any = None

    def get_input_spec(self) -> MultimodalInputSpec:
        return self.descriptor.get_input_spec()

    async def start(self) -> None:
        if self.runtime is not None:
            raise RuntimeError("Embedding backend is already started")
        self.runtime = self.descriptor.instantiate()
        self.runtime._ensure_engine()

    async def embed(self, content: EmbeddingContent) -> Any:
        return await self.runtime._embed_content(content)

    async def aclose(self) -> None:
        if self.runtime is not None:
            await self.runtime.aclose()
            self.runtime = None


def create_service_backend(
    family: str, model: str, dimensions: int, options: dict[str, Any]
) -> ServiceBackend:
    if family in {"sglang", "vllm"}:
        return NativeServiceBackend(family, model, dimensions, options)
    raise EmbeddingConfigurationError("Embedding service family must be sglang or vllm")
