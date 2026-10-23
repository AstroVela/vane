# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""MLX metadata, validation and model ownership for the common service.

No transport or server lives here. Metadata and raw-input checks require only
the base installation; model SDK imports belong exclusively to backend start.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from vane.ai import _embedding_ipc as ipc
from vane.ai._audio_embedding import AudioClip
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec, MultimodalInputSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._errors import _is_resource_error
from vane.ai._video_embedding import VideoClip
from vane.ai.options import _validate_mlx_service_options
from vane.ai.providers._embedding_service_backend import owned_result
from vane.ai.providers._mlx_media import (
    MAX_IMAGE_PIXELS,
    MIN_AUDIO_SAMPLES,
    validate_audio,
    validate_image,
    validate_video,
)
from vane.ai.providers._mlx_multimodal import joint_plan

if TYPE_CHECKING:
    from vane.ai.providers._embedding_service import _ServiceMetadata


def model_contract(options: dict[str, Any]) -> ipc.ModelContract:
    return ipc.ModelContract(1, options["revision"], options["dtype"], options["max_length"], options["overlength"])


def describe_service(
    kind: str, model: str | None, dimensions: int | None, options: dict[str, Any]
) -> tuple[str, int, dict[str, Any], _ServiceMetadata]:
    from vane.ai.providers import mlx
    from vane.ai.providers._embedding_service import _ServiceMetadata

    prepared = _validate_mlx_service_options(options, text=kind == "text")
    contract = model_contract(prepared)
    native_options = contract.to_wire()
    native_options.pop("encoding_version")
    if kind == "text":
        native_options["input_type"] = prepared["input_type"]
    factory = {
        "text": mlx.MLXTextEmbedderDescriptor,
        "image": mlx.MLXImageEmbedderDescriptor,
        "video": mlx.MLXVideoEmbedderDescriptor,
        "audio": mlx.MLXAudioEmbedderDescriptor,
        "multimodal": mlx.MLXMultimodalEmbedderDescriptor,
    }[kind]
    native = factory(model, dimensions, options=native_options)
    encoding = (
        "text_query"
        if kind == "text" and prepared["input_type"] == "query"
        else "text_document"
        if kind == "text"
        else kind
    )
    metadata = _ServiceMetadata(
        mlx.MLXMultimodalEmbedderDescriptor().get_input_spec(),
        native.get_output_spec(),
        ipc.RequestContext(encoding, contract),
        kind == "video" and native.supports_image_queries(),
    )
    return native.get_model(), native.get_dimensions(), prepared, metadata


def validate_request(context: ipc.RequestContext, expected: ipc.ModelContract, parts: list[dict[str, Any]]) -> None:
    if context.model_contract != expected or context.encoding == "default":
        raise EmbeddingConfigurationError("MLX service encoding/model contract mismatch")
    kind = "text" if context.encoding in {"text_document", "text_query"} else context.encoding
    if kind != "multimodal" and [part["kind"] for part in parts] != [kind]:
        raise EmbeddingConfigurationError("MLX unary encoding requires exactly one matching input part")
    for part in parts:
        if part["kind"] in {"image", "video"}:
            buffers = part["buffers"] if part["kind"] == "video" else [part["buffer"]]
            if any(item["shape"][0] * item["shape"][1] > MAX_IMAGE_PIXELS for item in buffers):
                raise EmbeddingConfigurationError("MLX image exceeds the pixel budget")
            if part["kind"] == "video" and any(item["shape"] != buffers[0]["shape"] for item in buffers):
                raise EmbeddingConfigurationError("MLX video frames must have equal H/W within a clip")
        elif part["kind"] == "audio" and part["buffer"]["shape"][0] < MIN_AUDIO_SAMPLES:
            raise EmbeddingConfigurationError("MLX audio requires at least 161 samples")


def _normalized_part(part: EmbeddingPart) -> EmbeddingPart:
    value = part.value
    if part.spec.kind == "video":
        value = VideoClip(
            tuple(value.frames), tuple(float(t) for t in value.frame_times), tuple(int(i) for i in value.frame_indices)
        )
    elif part.spec.kind == "audio":
        value = AudioClip(value.samples, int(value.sample_rate))
    return EmbeddingPart(part.spec, value)


def preflight(contents: list[EmbeddingContent], spec: MultimodalInputSpec, encoding: str) -> list[EmbeddingContent]:
    if encoding == "multimodal":
        plan = joint_plan(contents, spec)
        media = {"image": plan.images, "video": plan.videos, "audio": plan.audios}
        return [
            EmbeddingContent(
                tuple(
                    _normalized_part(
                        EmbeddingPart(
                            EmbeddingPartSpec(part.kind),
                            part.text if part.kind == "text" else media[part.kind][part.index],
                        )
                    )
                    for part in row
                )
            )
            for row in plan.rows
        ]
    if not isinstance(contents, list):
        raise EmbeddingConfigurationError("MLX embedding methods require a list of inputs")
    for row in contents:
        if len(row.parts) != 1:
            raise EmbeddingConfigurationError("MLX unary encoding requires one input")
        part = row.parts[0]
        value = part.value
        if part.spec.kind == "text":
            if not isinstance(value, str):
                raise EmbeddingConfigurationError("MLX embed_text requires strings")
            try:
                value.encode("utf-8")
            except UnicodeError:
                raise EmbeddingConfigurationError("MLX text requires valid UTF-8") from None
        else:
            {"image": validate_image, "audio": validate_audio, "video": validate_video}[part.spec.kind](value)
    return [EmbeddingContent(tuple(_normalized_part(part) for part in row.parts)) for row in contents]


def _create_core(options: dict[str, Any]) -> Any:
    # Called only on the backend owner thread, after endpoint reservation.
    from vane.ai.providers.mlx import _MLXModelCore

    return _MLXModelCore(options)


class MLXServiceBackend:
    """One model owner and one submission at a time; transport owns admission."""

    def __init__(self, model: str, dimensions: int, options: dict[str, Any]):
        from vane.ai.providers.mlx import MLXMultimodalEmbedderDescriptor

        self.descriptor = MLXMultimodalEmbedderDescriptor(model, dimensions, options=options)
        self._options = self.descriptor.get_options()
        self.contract = model_contract(self._options)
        self.state = "CREATED"
        self.restart_required = False
        self._core: Any = None
        self._executor: ThreadPoolExecutor | None = None
        self._lock = asyncio.Lock()
        self._closing: asyncio.Task[None] | None = None

    def get_input_spec(self) -> MultimodalInputSpec:
        return self.descriptor.get_input_spec()

    def _check_resource_state(self) -> None:
        if self.restart_required:
            raise MemoryError("MLX model resources are unavailable; explicitly restart the embedding service")

    def validate_request(self, context: ipc.RequestContext, parts: list[dict[str, Any]]) -> None:
        validate_request(context, self.contract, parts)
        self._check_resource_state()

    async def _submit(self, operation: Callable[[], Any]) -> Any:
        assert self._executor is not None
        owner = asyncio.get_running_loop().run_in_executor(self._executor, operation)
        return await owned_result(owner)

    def _load(self) -> None:
        # Assign inside actual work: cancellation of the waiter must never
        # orphan a successfully constructed model before its owner can close it.
        self._core = _create_core(self._options)

    async def start(self) -> None:
        if self.state != "CREATED":
            raise RuntimeError("MLX service backend can start only once")
        self.state = "STARTING"
        try:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vane-mlx-service")
            async with self._lock:
                await self._submit(self._load)
            if self._closing is not None:
                raise RuntimeError("MLX embedding backend closed during startup")
            self.state = "READY"
        except BaseException as error:
            self.state = "FAILED"
            self.restart_required = _is_resource_error(error)
            raise

    def _encode(self, content: EmbeddingContent, context: ipc.RequestContext) -> Any:
        assert self._core is not None
        return self._core.encode(content, context.encoding)

    async def embed(self, content: EmbeddingContent, context: ipc.RequestContext) -> Any:
        if context.model_contract != self.contract or context.encoding == "default":
            raise EmbeddingConfigurationError("MLX service encoding/model contract mismatch")
        self._check_resource_state()
        if self.state != "READY":
            raise RuntimeError("MLX embedding backend is not accepting work")
        async with self._lock:
            self._check_resource_state()
            if self.state not in {"READY", "DRAINING"}:
                raise RuntimeError("MLX embedding backend is not ready")
            try:
                # The lock covers submission AND actual completion. Neither
                # cancelled waiters nor queued requests create executor queues.
                return await self._submit(lambda: self._encode(content, context))
            except BaseException as error:
                if _is_resource_error(error):
                    self.restart_required = True
                    self.state = "FAILED"
                raise

    def _close_core(self) -> None:
        if self._core is not None:
            try:
                self._core.close()
            finally:
                self._core = None

    async def _close(self) -> None:
        self.state = "DRAINING"
        try:
            async with self._lock:
                if self._executor is not None:
                    await self._submit(self._close_core)
            self.state = "CLOSED"
        except BaseException as error:
            self.state = "FAILED"
            self.restart_required |= _is_resource_error(error)
            raise
        finally:
            # Owned work has joined, including a failed close. No queued model
            # work exists here; join the idle owner thread before dropping it.
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None

    async def aclose(self) -> None:
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        await owned_result(self._closing)
