# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""EmbeddingGemma 2 through explicit native or shared ONNX Runtime sessions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

from vane.ai._audio_embedding import AudioInputSpec
from vane.ai._embedding_content import EmbeddingPartSpec, MultimodalInputSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._embedding_outputs import EmbeddingOutput, EmbeddingOutputSpec
from vane.ai._onnx_options import ONNX_ENCODING_VERSION, validate_onnx_model, validate_onnx_options
from vane.ai._video_embedding import VideoInputSpec
from vane.ai.embedding import Modality
from vane.ai.protocols import (
    AudioEmbedderDescriptor,
    ImageEmbedderDescriptor,
    MultimodalEmbedderDescriptor,
    TextEmbedderDescriptor,
    VideoEmbedderDescriptor,
)
from vane.ai.provider import Provider
from vane.ai.typing import UDFOptions

Kind = Literal["text", "image", "audio", "video", "multimodal"]


def input_spec(limit: int) -> MultimodalInputSpec:
    return MultimodalInputSpec(
        frozenset(EmbeddingPartSpec(kind) for kind in ("text", "image", "video", "audio")),
        video=VideoInputSpec(max_frames=32, max_input_bytes=limit),
        audio=AudioInputSpec(16000, 16000 * 180, max_channels=1, max_input_bytes=limit),
        max_input_bytes=limit,
    )


def output_spec(model: str, dimensions: int, digest: str, kind: str) -> EmbeddingOutputSpec:
    kinds: tuple[Modality, ...] = (
        ("text", "image", "video", "audio") if kind == "multimodal" else (cast(Modality, kind),)
    )
    return EmbeddingOutputSpec(
        (EmbeddingOutput("embedding", dimensions, f"{model}:onnx-py{ONNX_ENCODING_VERSION}:{digest}", kinds),)
    )


class _ONNXDescriptor:
    kind: Kind = "text"

    def __init__(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ):
        self.model, self.dimensions = validate_onnx_model(model, dimensions)
        self.options = validate_onnx_options(options or {}, text=self.kind == "text")

    def get_provider(self) -> str:
        return "onnx"

    def get_model(self) -> str:
        return self.model

    def get_dimensions(self) -> int:
        return self.dimensions

    def get_options(self) -> dict[str, Any]:
        return dict(self.options)

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(num_gpus=1 if self.options["backend"] == "cuda" else 0)

    def get_output_spec(self, *, dimensions: int | None = None) -> EmbeddingOutputSpec:
        if dimensions is not None and (type(dimensions) is not int or dimensions != self.dimensions):
            raise EmbeddingConfigurationError("ONNX output dimensions must match the descriptor")
        return output_spec(self.model, self.dimensions, self.options["model_sha256"], self.kind)

    def instantiate(self) -> Any:
        from vane.ai.providers._onnx_model import ONNXEmbedder

        return ONNXEmbedder(self.dimensions, self.options)


class ONNXTextDescriptor(_ONNXDescriptor, TextEmbedderDescriptor):
    def supports_chunking(self) -> bool:
        return False


class ONNXImageDescriptor(_ONNXDescriptor, ImageEmbedderDescriptor):
    kind: Kind = "image"


class ONNXAudioDescriptor(_ONNXDescriptor, AudioEmbedderDescriptor):
    kind: Kind = "audio"

    def get_input_spec(self) -> AudioInputSpec:
        value = input_spec(self.options["max_input_bytes"]).audio
        assert value is not None
        return value


class ONNXVideoDescriptor(_ONNXDescriptor, VideoEmbedderDescriptor):
    kind: Kind = "video"

    def get_input_spec(self) -> VideoInputSpec:
        value = input_spec(self.options["max_input_bytes"]).video
        assert value is not None
        return value

    def supports_image_queries(self) -> bool:
        return True


class ONNXMultimodalDescriptor(_ONNXDescriptor, MultimodalEmbedderDescriptor):
    kind: Kind = "multimodal"

    def get_input_spec(self) -> MultimodalInputSpec:
        return input_spec(self.options["max_input_bytes"])


class ONNXProvider(Provider):
    def __init__(self, name: str | None = None):
        if name not in (None, "onnx"):
            raise EmbeddingConfigurationError("The ONNX provider cannot be renamed")

    @property
    def name(self) -> str:
        return "onnx"

    def _descriptor(
        self, kind: Kind, model: str | None, dimensions: int | None, options: Mapping[str, Any] | None
    ) -> Any:
        if options is not None and options.get("runtime") == "service":
            from vane.ai.providers._onnx_service import service_descriptor

            return service_descriptor(kind, model, dimensions, options)
        return {
            "text": ONNXTextDescriptor,
            "image": ONNXImageDescriptor,
            "audio": ONNXAudioDescriptor,
            "video": ONNXVideoDescriptor,
            "multimodal": ONNXMultimodalDescriptor,
        }[kind](model, dimensions, options=options)

    def get_text_embedder(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> TextEmbedderDescriptor:
        return self._descriptor("text", model, dimensions, options)

    def get_image_embedder(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> ImageEmbedderDescriptor:
        return self._descriptor("image", model, dimensions, options)

    def get_audio_embedder(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> AudioEmbedderDescriptor:
        return self._descriptor("audio", model, dimensions, options)

    def get_video_embedder(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> VideoEmbedderDescriptor:
        return self._descriptor("video", model, dimensions, options)

    def get_multimodal_embedder(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> MultimodalEmbedderDescriptor:
        return self._descriptor("multimodal", model, dimensions, options)
