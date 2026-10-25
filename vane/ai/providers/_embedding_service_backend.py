# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Model ownership behind the shared embedding service lifecycle.

Construction validates metadata only. SDKs and model resources belong to start(),
after the transport has exclusively reserved its rendezvous path.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, TypeVar

from vane.ai import _embedding_ipc as ipc
from vane.ai._embedding_content import EmbeddingContent, MultimodalInputSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError

_Result = TypeVar("_Result")


async def owned_result(future: asyncio.Future[_Result]) -> _Result:
    """Join actual owned work before propagating even repeated cancellation."""
    cancelled = False
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    result = future.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


class ServiceBackend(Protocol):
    descriptor: Any
    restart_required: bool

    def get_input_spec(self) -> MultimodalInputSpec: ...

    def validate_request(self, context: ipc.RequestContext, parts: list[dict[str, Any]]) -> None: ...

    async def start(self) -> None: ...

    async def embed(self, content: EmbeddingContent, context: ipc.RequestContext) -> Any: ...

    async def aclose(self) -> None: ...


class NativeServiceBackend:
    """Delegate SGLang/vLLM encoding and teardown to their existing runtime."""

    restart_required = False

    def __init__(self, family: str, model: str, dimensions: int, options: dict[str, Any]):
        from vane.ai.providers._native_inference import NativeMultimodalEmbedderDescriptor

        self.descriptor = NativeMultimodalEmbedderDescriptor(model, dimensions, options, family, family)
        self.runtime: Any = None

    def get_input_spec(self) -> MultimodalInputSpec:
        return self.descriptor.get_input_spec()

    def validate_request(self, context: ipc.RequestContext, parts: list[dict[str, Any]]) -> None:
        if context.encoding != "default" or context.model_contract is not None or context.litert is not None:
            raise EmbeddingConfigurationError(
                "Native embedding service requires default encoding and no model contract"
            )

    async def start(self) -> None:
        if self.runtime is not None:
            raise RuntimeError("Embedding backend is already started")
        self.runtime = self.descriptor.instantiate()
        self.runtime._ensure_engine()

    async def embed(self, content: EmbeddingContent, context: ipc.RequestContext) -> Any:
        self.validate_request(context, [])
        return await self.runtime._embed_content(content)

    async def aclose(self) -> None:
        if self.runtime is not None:
            await self.runtime.aclose()
            self.runtime = None


class LiteRTServiceBackend:
    """Preserve the existing LiteRT descriptor, SDK thread and request encoder."""

    restart_required = False

    def __init__(self, model: str, dimensions: int, options: dict[str, Any]):
        from vane.ai.providers.litert import LiteRTMultimodalEmbedderDescriptor

        self.descriptor = LiteRTMultimodalEmbedderDescriptor(model, dimensions, options=options)
        self.runtime: Any = None

    def get_input_spec(self) -> MultimodalInputSpec:
        return self.descriptor.get_input_spec()

    def validate_request(self, context: ipc.RequestContext, parts: list[dict[str, Any]]) -> None:
        from vane.ai.providers._litert_service import encoding_identity

        expected = encoding_identity(self.descriptor.get_options())
        encoding = context.litert
        if (
            encoding is None
            or context.encoding != "default"
            or context.model_contract is not None
            or any(encoding.to_wire()[key] != value for key, value in expected.items())
            or encoding.input_type == "query"
            and (len(parts) != 1 or parts[0]["kind"] != "text")
        ):
            raise EmbeddingConfigurationError("LiteRT service encoding or retrieval-role mismatch")

    async def start(self) -> None:
        from vane.ai.providers._litert_service import LiteRTServiceRuntime

        if self.runtime is not None:
            raise RuntimeError("Embedding backend is already started")
        self.runtime = LiteRTServiceRuntime(self.descriptor)
        await self.runtime.start()

    async def embed(self, content: EmbeddingContent, context: ipc.RequestContext) -> Any:
        self.validate_request(context, [{"kind": part.spec.kind} for part in content.parts])
        assert context.litert is not None and self.runtime is not None
        return await self.runtime._embed_content(content, input_type=context.litert.input_type)

    async def aclose(self) -> None:
        if self.runtime is not None:
            await self.runtime.aclose()
            self.runtime = None


def create_service_backend(family: str, model: str, dimensions: int, options: dict[str, Any]) -> ServiceBackend:
    if family in {"sglang", "vllm"}:
        return NativeServiceBackend(family, model, dimensions, options)
    if family == "mlx":
        from vane.ai.providers._mlx_service import MLXServiceBackend

        return MLXServiceBackend(model, dimensions, options)
    if family == "litert":
        return LiteRTServiceBackend(model, dimensions, options)
    raise EmbeddingConfigurationError("Embedding service family must be sglang, vllm, mlx or litert")
