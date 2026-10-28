# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only ONNX service descriptors and one service-owned model thread."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

import numpy as np

from vane.ai import _embedding_ipc as ipc
from vane.ai._embedding_content import EmbeddingContent
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._onnx_options import validate_onnx_model, validate_onnx_options
from vane.ai.options import _SERVICE_EMBED_OPTIONS
from vane.ai.provider import _ProviderResultError
from vane.ai.providers._embedding_service import (
    ServiceAudioDescriptor,
    ServiceEmbedder,
    ServiceImageDescriptor,
    ServiceMultimodalDescriptor,
    ServiceTextDescriptor,
    ServiceVideoDescriptor,
    _ServiceDescriptor,
)
from vane.ai.providers._embedding_service_backend import owned_result
from vane.ai.providers._onnx_model import validate_parts
from vane.ai.providers.onnx import ONNXMultimodalDescriptor, input_spec, output_spec


class _ONNXServiceDescriptor(_ServiceDescriptor):
    def __init__(self, model: str | None, dimensions: int | None, options: Mapping[str, Any]):
        model, dimensions = validate_onnx_model(model, dimensions)
        prepared = validate_onnx_options(options, text=self.input_kind == "text", service=True)
        endpoint = {key: value for key, value in prepared.items() if key in _SERVICE_EMBED_OPTIONS}
        super().__init__(model, dimensions, endpoint, "onnx", "onnx")
        self.options = prepared | self.options

    def get_transport_input_spec(self) -> Any:
        return input_spec(self.options["max_input_bytes"])

    def get_request_options(self) -> dict[str, Any]:
        return ipc.ONNXEncoding(
            self.options["model_sha256"], self.options["max_length"], self.options.get("input_type", "document")
        ).to_wire()

    def get_output_spec(self, *, dimensions: int | None = None) -> Any:
        if dimensions is not None and (type(dimensions) is not int or dimensions != self.dimensions):
            raise EmbeddingConfigurationError("ONNX output dimensions must match the descriptor")
        assert self.dimensions is not None
        return output_spec(self.model, self.dimensions, self.options["model_sha256"], self.input_kind)

    def supports_image_queries(self) -> bool:
        return self.input_kind == "video"

    def instantiate(self) -> ONNXServiceEmbedder:
        return ONNXServiceEmbedder(self)


class ONNXServiceTextDescriptor(_ONNXServiceDescriptor, ServiceTextDescriptor):
    pass


class ONNXServiceImageDescriptor(_ONNXServiceDescriptor, ServiceImageDescriptor):
    pass


class ONNXServiceAudioDescriptor(_ONNXServiceDescriptor, ServiceAudioDescriptor):
    pass


class ONNXServiceVideoDescriptor(_ONNXServiceDescriptor, ServiceVideoDescriptor):
    pass


class ONNXServiceMultimodalDescriptor(_ONNXServiceDescriptor, ServiceMultimodalDescriptor):
    pass


def service_descriptor(kind: str, model: str | None, dimensions: int | None, options: Mapping[str, Any]) -> Any:
    return {
        "text": ONNXServiceTextDescriptor,
        "image": ONNXServiceImageDescriptor,
        "audio": ONNXServiceAudioDescriptor,
        "video": ONNXServiceVideoDescriptor,
        "multimodal": ONNXServiceMultimodalDescriptor,
    }[kind](model, dimensions, options)


class ONNXServiceEmbedder(ServiceEmbedder):
    async def _embed_contents(self, contents: list[EmbeddingContent], encoding: str) -> list[Any]:
        if not isinstance(contents, list) or any(not isinstance(content, EmbeddingContent) for content in contents):
            raise EmbeddingConfigurationError("ONNX service requires a list of decoded typed inputs")
        validate_parts(
            [[(part.spec.kind, part.value) for part in content.parts] for content in contents],
            self.spec.max_input_bytes,
        )
        return await super()._embed_contents(contents, encoding)

    async def _request(self, plan: Any) -> Any:
        try:
            values = await super()._request(plan)
        except (EmbeddingConfigurationError, _ProviderResultError):
            raise
        except (TypeError, ValueError, OverflowError):
            raise _ProviderResultError("ONNX service returned an invalid embedding") from None
        norm = np.linalg.norm(values.astype(np.float64))
        if not np.isfinite(norm) or norm <= 0 or abs(norm - 1) > 1e-3:
            raise _ProviderResultError("ONNX service embeddings must have nonzero unit norm within 1e-3")
        return (values / norm).astype(np.float32)


async def _join(future: asyncio.Future[Any]) -> Any:
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            pass
        except Exception:
            break
    return future.result()


class ONNXServiceBackend:
    restart_required = False

    def __init__(self, model: str, dimensions: int, options: dict[str, Any]):
        self.descriptor = ONNXMultimodalDescriptor(model, dimensions, options=options)
        self.executor: ThreadPoolExecutor | None = None
        self.initialization: asyncio.Future[Any] | None = None
        self.closing: asyncio.Task[None] | None = None

    def get_input_spec(self) -> Any:
        return self.descriptor.get_input_spec()

    def validate_request(self, context: ipc.RequestContext, parts: list[dict[str, Any]]) -> None:
        contract = context.onnx
        options = self.descriptor.get_options()
        if (
            contract is None
            or context.encoding != "default"
            or contract.model_sha256 != options["model_sha256"]
            or contract.max_length != options["max_length"]
            or contract.input_type == "query"
            and [part["kind"] for part in parts] != ["text"]
        ):
            raise EmbeddingConfigurationError("ONNX service bundle/encoding or retrieval-role mismatch")

    async def start(self) -> None:
        if self.executor is not None:
            raise RuntimeError("ONNX service backend is already started or closed")
        if self.closing is not None:
            if not self.closing.done():
                raise RuntimeError("ONNX service backend is closing")
            self.closing.result()
            self.closing = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vane-onnx-service")
        self.initialization = asyncio.get_running_loop().run_in_executor(self.executor, self.descriptor.instantiate)
        await asyncio.shield(self.initialization)

    async def embed(self, content: EmbeddingContent, context: ipc.RequestContext) -> Any:
        self.validate_request(context, [{"kind": part.spec.kind} for part in content.parts])
        if self.closing is not None or self.initialization is None or self.executor is None:
            raise RuntimeError("ONNX service backend is not running")
        assert context.onnx is not None
        runtime = self.initialization.result()
        rows = [[(part.spec.kind, part.value) for part in content.parts]]
        future = asyncio.get_running_loop().run_in_executor(
            self.executor, partial(runtime._run, rows, role=context.onnx.input_type)
        )
        return (await owned_result(future))[0]

    async def _close(self) -> None:
        if self.initialization is not None:
            try:
                runtime = await _join(self.initialization)
            except BaseException:
                pass
            else:
                assert self.executor is not None
                await _join(asyncio.get_running_loop().run_in_executor(self.executor, runtime.close))
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    async def aclose(self) -> None:
        closing = self.closing
        if closing is None or closing.done() and (closing.cancelled() or closing.exception() is not None):
            closing = self.closing = asyncio.create_task(self._close())
        while True:
            try:
                await asyncio.shield(closing)
                return
            except asyncio.CancelledError:
                if closing.cancelled():
                    raise
