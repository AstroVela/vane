# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""LiteRT-LM 0.18 HTTP: ordered image parts and PCM16 WAV, one vector per row."""

from __future__ import annotations

import io
import math
import wave
from collections.abc import Mapping
from typing import Any

import numpy as np

from vane.ai._audio_embedding import AudioClip, AudioInputSpec
from vane.ai._client_config import _text, copy_client_options
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec, MultimodalInputSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._embedding_outputs import EmbeddingOutput, EmbeddingOutputSpec
from vane.ai._litert_options import LITERT_MODELS, validate_litert_model, validate_litert_options
from vane.ai._redaction import unwrap_sensitive_options
from vane.ai._video_embedding import VideoClip, VideoInputSpec
from vane.ai.protocols import (
    AudioEmbedderDescriptor,
    ImageEmbedderDescriptor,
    MultimodalEmbedderDescriptor,
    TextEmbedderDescriptor,
    VideoEmbedderDescriptor,
)
from vane.ai.provider import _ProviderResultError, _translate_missing_provider_dependency
from vane.ai.providers._embedding_http import HTTPEmbedder, _HTTPEmbeddingDescriptor
from vane.ai.providers._embedding_http_codecs import EmbeddingHTTPCodec, json_bytes

DOCUMENT_PREFIX = "title: none | text: "
QUERY_PREFIX = "task: search result | query: "


class LiteRTHTTPCodec(EmbeddingHTTPCodec):
    def __init__(self, model: str, dimensions: int, options: dict[str, Any], kind: str):
        self.input_format = "multimodal"
        self.kind = kind
        self.role = options.get("input_type", "document")
        modalities = list(LITERT_MODELS[model])
        if "image" in modalities:
            modalities.append("video")
        parts = modalities if kind == "multimodal" else [kind]
        limit = options["max_input_bytes"]
        self.spec = MultimodalInputSpec(
            frozenset(EmbeddingPartSpec(part) for part in parts),
            video=VideoInputSpec(max_frames=64, max_input_bytes=limit) if "video" in parts else None,
            audio=AudioInputSpec(16000, 16000 * 180, max_channels=1, max_input_bytes=limit)
            if "audio" in parts
            else None,
            max_input_bytes=limit,
        )
        metadata = {
            "model": model,
            "dimensions": dimensions,
            "encoding_format": "float",
            "normalize": True,
            "insert_special_tokens": True,
        }
        self.prefix = json_bytes(metadata)[:-1] + b',"input":['
        self.suffix = b"]}"

    @staticmethod
    def _image_size(value: Any) -> int:
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.uint8
            or value.ndim != 3
            or value.shape[2] != 3
            or min(value.shape[:2]) < 1
            or value.shape[0] * value.shape[1] > 16 * 1024**2
        ):
            raise EmbeddingConfigurationError("LiteRT HTTP images require RGB uint8 HWC arrays within 16 Mi pixels")
        return value.nbytes

    def _video(self, value: Any) -> tuple[Any, ...]:
        if not isinstance(value, VideoClip):
            raise EmbeddingConfigurationError("LiteRT HTTP video requires a decoded VideoClip")
        assert self.spec.video is not None
        self.spec.video.validate_count(len(value.frames))
        if (
            len(value.frames) != len(value.frame_times)
            or len(value.frames) != len(value.frame_indices)
            or any(type(index) is not int or index < 0 for index in value.frame_indices)
            or any(a >= b for a, b in zip(value.frame_indices, value.frame_indices[1:]))
            or any(
                isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time) or time < 0
                for time in value.frame_times
            )
            or any(a > b for a, b in zip(value.frame_times, value.frame_times[1:]))
        ):
            raise EmbeddingConfigurationError("LiteRT HTTP video requires ordered frame indices and timestamps")
        return value.frames

    @staticmethod
    def _audio_size(value: Any) -> int:
        if (
            not isinstance(value, AudioClip)
            or type(value.sample_rate) is not int
            or value.sample_rate != 16000
            or not isinstance(value.samples, np.ndarray)
            or value.samples.dtype not in (np.float32, np.float64)
            or value.samples.ndim != 2
            or value.samples.shape[1] != 1
            or not 1 <= value.samples.shape[0] <= 16000 * 180
            or not np.isfinite(value.samples).all()
            or np.any(np.abs(value.samples) > 1)
        ):
            raise EmbeddingConfigurationError("LiteRT HTTP audio requires finite mono 16 kHz PCM in [-1, 1]")
        return value.samples.nbytes

    @staticmethod
    def _media(kind: str, value: Any) -> bytes:
        with io.BytesIO() as buffer:
            if kind == "image":
                with _translate_missing_provider_dependency("embedding_http", "PIL"):
                    from PIL import Image

                with Image.fromarray(np.ascontiguousarray(value)) as image:
                    image.save(buffer, format="PNG")
            else:
                samples = np.rint(np.clip(value.samples[:, 0].astype(np.float64) * 32768, -32768, 32767))
                with wave.open(buffer, "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(16000)
                    writer.writeframes(samples.astype("<i2").tobytes())
            return buffer.getvalue()

    def encode(self, content: EmbeddingContent) -> bytes:
        import base64

        if not isinstance(content, EmbeddingContent):
            raise EmbeddingConfigurationError("LiteRT HTTP requires typed EmbeddingContent rows")
        self.spec.validate_layout(tuple(part.spec for part in content.parts))
        active = []
        raw_size = 0
        for part in content.parts:
            kind, value = part.spec.kind, part.value
            if value is None:
                continue
            if kind == "text":
                if not isinstance(value, str):
                    raise EmbeddingConfigurationError("LiteRT HTTP text requires strings")
                if self.kind != "text" and not value.strip():
                    continue
                raw_size += len(value.encode("utf-8"))
                active.append((kind, value))
            elif kind == "video":
                for frame in self._video(value):
                    raw_size += self._image_size(frame)
                    active.append(("image", frame))
            else:
                raw_size += self._image_size(value) if kind == "image" else self._audio_size(value)
                active.append((kind, value))
            if raw_size > self.spec.max_input_bytes:
                raise EmbeddingConfigurationError("LiteRT HTTP row exceeds the combined decoded input byte budget")
        if not active:
            raise EmbeddingConfigurationError("LiteRT HTTP row has no nonempty parts")
        encoded_size, first_text = 0, True
        messages = []
        for kind, value in active:
            if kind == "text":
                if first_text:
                    value = (QUERY_PREFIX if self.role == "query" else DOCUMENT_PREFIX) + value
                    first_text = False
                encoded_size += len(value.encode("utf-8"))
                messages.append({"type": "text", "text": value})
            else:
                data = self._media(kind, value)
                encoded_size += len(data)
                encoded = base64.b64encode(data).decode("ascii")
                messages.append(
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}
                    if kind == "image"
                    else {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}
                )
            if encoded_size > self.spec.max_input_bytes:
                raise EmbeddingConfigurationError("LiteRT HTTP row exceeds the encoded media and prefix byte budget")
        return json_bytes(messages)


class _LiteRTHTTPDescriptor(_HTTPEmbeddingDescriptor):
    kind = "text"

    def __init__(
        self, model: str | None, dimensions: int | None, options: Mapping[str, Any], client_options: dict[str, Any]
    ) -> None:
        selected, width = validate_litert_model(model, dimensions, "image" if self.kind == "video" else self.kind)
        super().__init__(selected, width, dict(options), "litert", "litert", client_options)

    def __post_init__(self) -> None:
        self.options = validate_litert_options(self.options, text=self.kind == "text", http=True)
        self.client_options = copy_client_options(self.client_options)
        _text(unwrap_sensitive_options(self.client_options).get("api_key"), "LITERT_API_KEY")
        self._codec()

    def _codec(self) -> LiteRTHTTPCodec:
        return LiteRTHTTPCodec(self.model, self.get_dimensions(), self.options, self.kind)

    def get_output_spec(self, *, dimensions: int | None = None) -> EmbeddingOutputSpec:
        if dimensions is not None and (type(dimensions) is not int or dimensions != self.dimensions):
            raise EmbeddingConfigurationError("LiteRT output dimensions must match the descriptor")
        kinds = list(LITERT_MODELS[self.model]) if self.kind == "multimodal" else [self.kind]
        if self.kind == "multimodal" and "image" in kinds:
            kinds.append("video")
        return EmbeddingOutputSpec(
            (
                EmbeddingOutput(
                    "embedding", self.get_dimensions(), self.model + ":" + self.options["model_sha256"], tuple(kinds)
                ),
            )
        )

    def instantiate(self) -> LiteRTHTTPEmbedder:
        return LiteRTHTTPEmbedder(self)


class LiteRTHTTPTextDescriptor(_LiteRTHTTPDescriptor, TextEmbedderDescriptor):
    def supports_chunking(self) -> bool:
        return False


class LiteRTHTTPImageDescriptor(_LiteRTHTTPDescriptor, ImageEmbedderDescriptor):
    kind = "image"


class LiteRTHTTPVideoDescriptor(_LiteRTHTTPDescriptor, VideoEmbedderDescriptor):
    kind = "video"

    def get_input_spec(self) -> VideoInputSpec:
        spec = self._codec().spec.video
        assert spec is not None
        return spec

    def supports_image_queries(self) -> bool:
        return True


class LiteRTHTTPAudioDescriptor(_LiteRTHTTPDescriptor, AudioEmbedderDescriptor):
    kind = "audio"

    def get_input_spec(self) -> AudioInputSpec:
        spec = self._codec().spec.audio
        assert spec is not None
        return spec


class LiteRTHTTPMultimodalDescriptor(_LiteRTHTTPDescriptor, MultimodalEmbedderDescriptor):
    kind = "multimodal"

    def get_input_spec(self) -> MultimodalInputSpec:
        return self._codec().spec


class LiteRTHTTPEmbedder(HTTPEmbedder):
    async def embed_video(self, clips: list[VideoClip]) -> list[Any]:
        return await self.embed_multimodal(
            [EmbeddingContent((EmbeddingPart(EmbeddingPartSpec("video"), value),)) for value in clips]
        )

    async def embed_audio(self, clips: list[AudioClip]) -> list[Any]:
        return await self.embed_multimodal(
            [EmbeddingContent((EmbeddingPart(EmbeddingPartSpec("audio"), value),)) for value in clips]
        )

    def _vector(self, item: dict[str, Any]) -> Any:
        vector = super()._vector(item)
        norm = np.linalg.norm(vector.astype(np.float64))
        if not np.isfinite(norm) or norm <= 0 or abs(norm - 1) > 1e-3:
            raise _ProviderResultError("LiteRT HTTP embeddings require unit norm within 1e-3")
        return (vector / norm).astype(np.float32)
