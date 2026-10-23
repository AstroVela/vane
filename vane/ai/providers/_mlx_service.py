# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""MLX metadata, validation and model ownership for the common service.

No transport or server lives here. Metadata and raw-input checks require only
the base installation; model SDK imports belong exclusively to backend start.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vane.ai import _embedding_ipc as ipc
from vane.ai._audio_embedding import AudioClip
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec, MultimodalInputSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._video_embedding import VideoClip
from vane.ai.options import _validate_mlx_service_options
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
