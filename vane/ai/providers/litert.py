# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit local EmbeddingGemma 2 execution through LiteRT-LM 0.18."""

from __future__ import annotations

import hashlib
import io
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np

from vane.ai._audio_embedding import AudioClip, AudioInputSpec
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPartSpec, MultimodalInputSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._embedding_outputs import EmbeddingOutput, EmbeddingOutputSpec
from vane.ai._litert_options import LITERT_MODELS, validate_litert_model, validate_litert_options
from vane.ai.protocols import (
    AudioEmbedderDescriptor,
    ImageEmbedderDescriptor,
    MultimodalEmbedderDescriptor,
    TextEmbedderDescriptor,
)
from vane.ai.provider import Provider, _ProviderResultError, _translate_missing_provider_dependency
from vane.ai.typing import UDFOptions

MAX_BATCH_BYTES = 64 * 1024**2
MAX_BATCH_ROWS = 256
MAX_IMAGE_PIXELS = 16 * 1024**2
DOCUMENT_PREFIX = "title: none | text: "
QUERY_PREFIX = "task: search result | query: "


class _LiteRTDescriptor:
    kind: Literal["text", "image", "audio", "multimodal"] = "text"

    def __init__(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> None:
        self._model, self._dimensions = validate_litert_model(model, dimensions, self.kind)
        self._options = validate_litert_options(options or {}, text=self.kind == "text")

    def get_provider(self) -> str:
        return "litert"

    def get_model(self) -> str:
        return self._model

    def get_dimensions(self) -> int:
        return self._dimensions

    def get_options(self) -> dict[str, Any]:
        return dict(self._options)

    def get_udf_options(self) -> UDFOptions:
        # The explicit subprocess owns its SDK backend. Vane GPU reservations
        # identify CUDA devices and cannot identify a Metal/OpenCL device.
        return UDFOptions(num_gpus=0)

    def get_output_spec(self, *, dimensions: int | None = None) -> EmbeddingOutputSpec:
        if dimensions is not None and (type(dimensions) is not int or dimensions != self._dimensions):
            raise EmbeddingConfigurationError("LiteRT output dimensions must match the descriptor")
        modalities = LITERT_MODELS[self._model] if self.kind == "multimodal" else (self.kind,)
        return EmbeddingOutputSpec(
            (
                EmbeddingOutput(
                    "embedding", self._dimensions, self._model + ":" + self._options["model_sha256"], modalities
                ),
            )
        )

    def instantiate(self) -> LiteRTEmbedder:
        return LiteRTEmbedder(self._model, self._dimensions, self._options, kind=self.kind)


class LiteRTTextEmbedderDescriptor(_LiteRTDescriptor, TextEmbedderDescriptor):
    def supports_chunking(self) -> bool:
        return False


class LiteRTImageEmbedderDescriptor(_LiteRTDescriptor, ImageEmbedderDescriptor):
    kind: Literal["text", "image", "audio", "multimodal"] = "image"


class LiteRTAudioEmbedderDescriptor(_LiteRTDescriptor, AudioEmbedderDescriptor):
    kind: Literal["text", "image", "audio", "multimodal"] = "audio"

    def get_input_spec(self) -> AudioInputSpec:
        return AudioInputSpec(
            sample_rate=16000, max_samples=16000 * 180, max_channels=1, max_input_bytes=self._options["max_input_bytes"]
        )


class LiteRTMultimodalEmbedderDescriptor(_LiteRTDescriptor, MultimodalEmbedderDescriptor):
    kind: Literal["text", "image", "audio", "multimodal"] = "multimodal"

    def get_input_spec(self) -> MultimodalInputSpec:
        modalities = LITERT_MODELS[self._model]
        return MultimodalInputSpec(
            frozenset(EmbeddingPartSpec(kind) for kind in modalities),
            audio=AudioInputSpec(
                sample_rate=16000,
                max_samples=16000 * 180,
                max_channels=1,
                max_input_bytes=self._options["max_input_bytes"],
            )
            if "audio" in modalities
            else None,
            max_input_bytes=self._options["max_input_bytes"],
        )


class LiteRTProvider(Provider):
    def __init__(self, name: str | None = None) -> None:
        if name not in (None, "litert"):
            raise EmbeddingConfigurationError("The LiteRT provider cannot be renamed")

    @property
    def name(self) -> str:
        return "litert"

    def get_text_embedder(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> TextEmbedderDescriptor:
        if options is not None and options.get("runtime") == "service":
            from vane.ai.providers._litert_service import LiteRTServiceTextDescriptor

            return LiteRTServiceTextDescriptor(model, dimensions, options)
        return LiteRTTextEmbedderDescriptor(model, dimensions, options=options)

    def get_image_embedder(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> ImageEmbedderDescriptor:
        if options is not None and options.get("runtime") == "service":
            from vane.ai.providers._litert_service import LiteRTServiceImageDescriptor

            return LiteRTServiceImageDescriptor(model, dimensions, options)
        return LiteRTImageEmbedderDescriptor(model, dimensions, options=options)

    def get_audio_embedder(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> AudioEmbedderDescriptor:
        if options is not None and options.get("runtime") == "service":
            from vane.ai.providers._litert_service import LiteRTServiceAudioDescriptor

            return LiteRTServiceAudioDescriptor(model, dimensions, options)
        return LiteRTAudioEmbedderDescriptor(model, dimensions, options=options)

    def get_multimodal_embedder(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> MultimodalEmbedderDescriptor:
        if options is not None and options.get("runtime") == "service":
            from vane.ai.providers._litert_service import LiteRTServiceMultimodalDescriptor

            return LiteRTServiceMultimodalDescriptor(model, dimensions, options)
        return LiteRTMultimodalEmbedderDescriptor(model, dimensions, options=options)


class LiteRTEmbedder:
    def __init__(self, model: str, dimensions: int, options: Mapping[str, Any], *, kind: str) -> None:
        descriptor = {
            "text": LiteRTTextEmbedderDescriptor,
            "image": LiteRTImageEmbedderDescriptor,
            "audio": LiteRTAudioEmbedderDescriptor,
            "multimodal": LiteRTMultimodalEmbedderDescriptor,
        }[kind](model, dimensions, options=options)
        self._options = descriptor.get_options()
        self._dimensions = descriptor.get_dimensions()
        self._modalities = LITERT_MODELS[model]
        self._engine: Any = None
        self._closed = False
        path = Path(self._options["model_path"])
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024**2), b""):
                digest.update(chunk)
        if digest.hexdigest() != self._options["model_sha256"]:
            raise EmbeddingConfigurationError("LiteRT model file does not match model_sha256")
        with _translate_missing_provider_dependency("litert", "litert_lm"):
            import litert_lm  # type: ignore[import-not-found, import-untyped, unused-ignore]

        self._sdk = litert_lm
        with litert_lm.ModelInfo(str(path)) as info:
            if not info.is_embedding_model or info.embedding is None or info.embedding.dimension != 768:
                raise EmbeddingConfigurationError("LiteRT model must declare a native 768-dimensional embedding output")
            actual = info.input_modalities
            if (actual.text, actual.vision, actual.audio) != tuple(
                item in self._modalities for item in ("text", "image", "audio")
            ):
                raise EmbeddingConfigurationError("LiteRT model modalities do not match the selected model declaration")
            signatures = info.embedding.signature_selection
            if signatures is None or self._options["max_length"] not in signatures:
                raise EmbeddingConfigurationError("LiteRT model does not contain the requested input signature")
            if "image" in self._modalities and self._options["vision_tokens_per_image"] > info.max_vision_token_budget:
                raise EmbeddingConfigurationError("LiteRT model does not contain the requested vision token budget")
        backend = (
            litert_lm.Backend.CPU(thread_count=self._options["num_threads"])
            if self._options["backend"] == "cpu"
            else litert_lm.Backend.GPU()
        )
        self._embedding_options = litert_lm.EmbeddingOptions(
            normalize=True,
            insert_special_tokens=True,
            input_overflow_strategy=litert_lm.InputOverflowStrategy.ERROR,
            output_size=dimensions,
            vision_tokens_per_image=self._options["vision_tokens_per_image"] if "image" in self._modalities else None,
        )
        self._engine = litert_lm.EmbeddingEngine(
            str(path),
            backend=backend,
            vision_backend=backend if "image" in self._modalities else None,
            audio_backend=backend if "audio" in self._modalities else None,
            cache_dir=self._options["cache_dir"],
            max_input_length=self._options["max_length"],
            vision_tokens_per_image=self._options["vision_tokens_per_image"] if "image" in self._modalities else None,
        )

    def _check_alive(self) -> None:
        if self._closed:
            raise RuntimeError("LiteRT embedder is closed")

    def _validate_part(self, kind: str, value: Any) -> int:
        if kind not in self._modalities:
            raise EmbeddingConfigurationError("Selected LiteRT model does not support this input modality")
        if kind == "text":
            if not isinstance(value, str):
                raise EmbeddingConfigurationError("LiteRT text inputs must be strings")
            return len(value.encode("utf-8"))
        if kind == "image":
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.uint8
                or value.ndim != 3
                or value.shape[2] != 3
                or min(value.shape[:2]) <= 0
                or value.shape[0] * value.shape[1] > MAX_IMAGE_PIXELS
            ):
                raise EmbeddingConfigurationError(
                    "LiteRT images require nonempty RGB uint8 HWC arrays within 16 Mi pixels"
                )
            return value.nbytes
        if (
            not isinstance(value, AudioClip)
            or type(value.sample_rate) is not int
            or value.sample_rate != 16000
            or not isinstance(value.samples, np.ndarray)
            or value.samples.ndim != 2
            or value.samples.shape[1] != 1
            or not 1 <= value.samples.shape[0] <= 16000 * 180
            or value.samples.dtype not in (np.float32, np.float64)
            or not np.isfinite(value.samples).all()
            or np.any(np.abs(value.samples) > 1)
        ):
            raise EmbeddingConfigurationError(
                "LiteRT audio requires finite mono 16 kHz floating PCM in [-1, 1], up to 180 seconds"
            )
        return value.samples.nbytes

    def _encode_part(self, kind: str, value: Any) -> Any:
        if kind == "text":
            return value
        if kind == "image":
            with _translate_missing_provider_dependency("litert", "PIL"):
                from PIL import Image  # type: ignore[import-not-found, import-untyped, unused-ignore]

            with io.BytesIO() as buffer, Image.fromarray(np.ascontiguousarray(value)) as image:
                image.save(buffer, format="PNG")
                return self._sdk.Content.ImageBytes(buffer.getvalue())
        samples = np.rint(np.clip(value.samples[:, 0].astype(np.float64) * 32768, -32768, 32767)).astype("<i2")
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(samples.tobytes())
            return self._sdk.Content.AudioBytes(buffer.getvalue())

    def _run(self, rows: list[list[tuple[str, Any]]], *, role: str = "document") -> list[np.ndarray]:
        self._check_alive()
        if len(rows) > MAX_BATCH_ROWS:
            raise EmbeddingConfigurationError("LiteRT batches support at most 256 rows")
        total = 0
        for row in rows:
            if not 1 <= len(row) <= 16:
                raise EmbeddingConfigurationError("LiteRT inputs require 1 to 16 parts per row")
            size = sum(self._validate_part(kind, value) for kind, value in row)
            if size > self._options["max_input_bytes"]:
                raise EmbeddingConfigurationError("LiteRT input row exceeds max_input_bytes")
            total += size
            if total > MAX_BATCH_BYTES:
                raise EmbeddingConfigurationError("LiteRT batch exceeds the 64 MiB input budget")
        if not rows:
            return []
        encoded = []
        encoded_total = 0
        for row in rows:
            parts = []
            first_text = True
            size = 0
            for kind, value in row:
                if kind == "text" and first_text:
                    value = (QUERY_PREFIX if role == "query" else DOCUMENT_PREFIX) + value
                    first_text = False
                part = self._encode_part(kind, value)
                size += len(part.encode("utf-8")) if isinstance(part, str) else len(part.bytes)
                parts.append(part)
            if size > self._options["max_input_bytes"]:
                raise EmbeddingConfigurationError("LiteRT encoded input exceeds max_input_bytes including prefixes")
            encoded_total += size
            if encoded_total > MAX_BATCH_BYTES:
                raise EmbeddingConfigurationError("LiteRT batch exceeds the 64 MiB encoded input budget")
            encoded.append(parts)
        responses = self._engine.compute_embedding_batch(encoded, options=self._embedding_options)
        if len(responses) != len(rows):
            raise _ProviderResultError("LiteRT must return one embedding per input row")
        result = []
        for response in responses:
            try:
                values = np.asarray(response.embedding, dtype=np.float32)
            except (TypeError, ValueError, OverflowError):
                raise _ProviderResultError("LiteRT returned a nonnumeric embedding") from None
            if values.shape != (self._dimensions,) or not np.isfinite(values).all():
                raise _ProviderResultError("LiteRT returned an invalid embedding shape or nonfinite values")
            norm = np.linalg.norm(values.astype(np.float64))
            if not np.isfinite(norm) or norm <= 0 or abs(norm - 1) > 1e-3:
                raise _ProviderResultError("LiteRT embeddings must have a nonzero unit norm within 1e-3")
            result.append((values / norm).astype(np.float32))
        return result

    def embed_text(self, texts: list[str]) -> list[np.ndarray]:
        if not isinstance(texts, list):
            raise EmbeddingConfigurationError("LiteRT embed_text requires a list of strings")
        return self._run([[("text", text)] for text in texts], role=self._options.get("input_type", "document"))

    def embed_image(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if not isinstance(images, list):
            raise EmbeddingConfigurationError("LiteRT embed_image requires a list of RGB arrays")
        return self._run([[("image", image)] for image in images])

    def embed_audio(self, clips: list[AudioClip]) -> list[np.ndarray]:
        if not isinstance(clips, list):
            raise EmbeddingConfigurationError("LiteRT embed_audio requires a list of AudioClip values")
        return self._run([[("audio", clip)] for clip in clips])

    def embed_multimodal(self, inputs: list[EmbeddingContent]) -> list[np.ndarray]:
        if not isinstance(inputs, list):
            raise EmbeddingConfigurationError("LiteRT embed_multimodal requires a list of EmbeddingContent values")
        rows: list[list[tuple[str, Any]]] = []
        for item in inputs:
            if not isinstance(item, EmbeddingContent) or any(part.spec.mime_type is not None for part in item.parts):
                raise EmbeddingConfigurationError("LiteRT multimodal inputs require decoded typed parts")
            rows.append([(part.spec.kind, part.value) for part in item.parts])
        return self._run(rows)

    def close(self) -> None:
        if self._closed:
            return
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        self._closed = True

    async def aclose(self) -> None:
        self.close()
