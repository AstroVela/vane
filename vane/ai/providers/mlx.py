# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""EmbeddingGemma 2 text, image, audio and video encoding on worker-local Metal."""

from __future__ import annotations

import gc
import json
import math
import platform
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from vane.ai._audio_embedding import AudioClip, AudioInputSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._embedding_outputs import EmbeddingOutput, EmbeddingOutputSpec
from vane.ai._video_embedding import VideoClip, VideoInputSpec
from vane.ai.options import (
    _validate_mlx_audio_options,
    _validate_mlx_image_options,
    _validate_mlx_text_options,
    _validate_mlx_video_options,
)
from vane.ai.protocols import (
    AudioEmbedderDescriptor,
    ImageEmbedderDescriptor,
    TextEmbedderDescriptor,
    VideoEmbedderDescriptor,
)
from vane.ai.provider import Provider, _translate_missing_provider_dependency
from vane.ai.typing import UDFOptions

MODEL_ID = "mlx-community/embeddinggemma-2-bf16"
DIMENSIONS = 768
MAX_IMAGE_PIXELS = 16 * 1024 * 1024
AUDIO_SAMPLE_RATE = 16000
MIN_AUDIO_SAMPLES = 161
MAX_AUDIO_SAMPLES = 480000
MAX_AUDIO_INPUT_BYTES = 4 * 1024 * 1024
MAX_VIDEO_FRAMES = 12
MAX_VIDEO_INPUT_BYTES = 64 * 1024 * 1024
_IMAGE_INPUT_KEYS = frozenset({"input_ids", "attention_mask", "pixel_values", "image_position_ids"})
_AUDIO_INPUT_KEYS = frozenset({"input_ids", "attention_mask", "input_features", "input_features_mask"})
_VIDEO_INPUT_KEYS = frozenset(
    {"input_ids", "attention_mask", "pixel_values_videos", "video_position_ids", "num_frames_per_video"}
)


class _MLXDescriptor:
    _validate_options = staticmethod(_validate_mlx_text_options)

    def __init__(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> None:
        if model is not None and model != MODEL_ID:
            raise EmbeddingConfigurationError("MLX supports only mlx-community/embeddinggemma-2-bf16")
        if dimensions is not None and (type(dimensions) is not int or dimensions != DIMENSIONS):
            raise EmbeddingConfigurationError("MLX EmbeddingGemma 2 requires dimensions=768")
        self._options = self._validate_options(options if options is not None else {})

    def get_provider(self) -> str:
        return "mlx"

    def get_model(self) -> str:
        return MODEL_ID

    def get_dimensions(self) -> int:
        return DIMENSIONS

    def get_options(self) -> dict[str, Any]:
        return dict(self._options)

    def get_udf_options(self) -> UDFOptions:
        # Vane's GPU resource identities are CUDA UUIDs; Metal is worker-local.
        return UDFOptions(num_gpus=0)


class MLXTextEmbedderDescriptor(_MLXDescriptor, TextEmbedderDescriptor):
    def supports_chunking(self) -> bool:
        return False

    def instantiate(self) -> MLXTextEmbedder:
        return MLXTextEmbedder(self.get_options())


class MLXImageEmbedderDescriptor(_MLXDescriptor, ImageEmbedderDescriptor):
    _validate_options = staticmethod(_validate_mlx_image_options)

    def get_output_spec(self, *, dimensions: int | None = None) -> EmbeddingOutputSpec:
        if dimensions is not None and (type(dimensions) is not int or dimensions != DIMENSIONS):
            raise EmbeddingConfigurationError("MLX EmbeddingGemma 2 requires dimensions=768")
        return EmbeddingOutputSpec((EmbeddingOutput("embedding", DIMENSIONS, MODEL_ID, ("image",)),))

    def instantiate(self) -> MLXImageEmbedder:
        return MLXImageEmbedder(self.get_options())


class MLXAudioEmbedderDescriptor(_MLXDescriptor, AudioEmbedderDescriptor):
    _validate_options = staticmethod(_validate_mlx_audio_options)

    def get_input_spec(self) -> AudioInputSpec:
        return AudioInputSpec(
            sample_rate=AUDIO_SAMPLE_RATE,
            max_samples=MAX_AUDIO_SAMPLES,
            max_channels=1,
            max_input_bytes=MAX_AUDIO_INPUT_BYTES,
        )

    def get_output_spec(self, *, dimensions: int | None = None) -> EmbeddingOutputSpec:
        if dimensions is not None and (type(dimensions) is not int or dimensions != DIMENSIONS):
            raise EmbeddingConfigurationError("MLX EmbeddingGemma 2 requires dimensions=768")
        return EmbeddingOutputSpec((EmbeddingOutput("embedding", DIMENSIONS, MODEL_ID, ("audio",)),))

    def instantiate(self) -> MLXAudioEmbedder:
        return MLXAudioEmbedder(self.get_options())


class MLXVideoEmbedderDescriptor(_MLXDescriptor, VideoEmbedderDescriptor):
    _validate_options = staticmethod(_validate_mlx_video_options)

    def get_input_spec(self) -> VideoInputSpec:
        return VideoInputSpec(frame_count=None, max_frames=MAX_VIDEO_FRAMES, max_input_bytes=MAX_VIDEO_INPUT_BYTES)

    def get_output_spec(self, *, dimensions: int | None = None) -> EmbeddingOutputSpec:
        if dimensions is not None and (type(dimensions) is not int or dimensions != DIMENSIONS):
            raise EmbeddingConfigurationError("MLX EmbeddingGemma 2 requires dimensions=768")
        return EmbeddingOutputSpec((EmbeddingOutput("embedding", DIMENSIONS, MODEL_ID, ("video",)),))

    def supports_image_queries(self) -> bool:
        return True

    def instantiate(self) -> MLXVideoEmbedder:
        return MLXVideoEmbedder(self.get_options())


class MLXProvider(Provider):
    def __init__(self, name: str | None = None) -> None:
        if name not in (None, "mlx"):
            raise EmbeddingConfigurationError("The MLX provider cannot be renamed")

    @property
    def name(self) -> str:
        return "mlx"

    def get_text_embedder(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> MLXTextEmbedderDescriptor:
        return MLXTextEmbedderDescriptor(model, dimensions, options=options)

    def get_image_embedder(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> MLXImageEmbedderDescriptor:
        return MLXImageEmbedderDescriptor(model, dimensions, options=options)

    def get_audio_embedder(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> MLXAudioEmbedderDescriptor:
        return MLXAudioEmbedderDescriptor(model, dimensions, options=options)

    def get_video_embedder(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> MLXVideoEmbedderDescriptor:
        return MLXVideoEmbedderDescriptor(model, dimensions, options=options)


class _MLXEmbedder:
    _validate_options = staticmethod(_validate_mlx_text_options)

    def __init__(self, options: Mapping[str, Any]) -> None:
        self._options = self._validate_options(options)
        self._model: Any = None
        self._tokenizer: Any = None
        self._processor: Any = None
        self._mx: Any = None
        self._closed = False
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise EmbeddingConfigurationError("MLX inference requires Apple Silicon macOS arm64 and Metal")
        with (
            _translate_missing_provider_dependency("mlx", "mlx"),
            _translate_missing_provider_dependency("mlx", "mlx_vlm"),
            _translate_missing_provider_dependency("mlx", "transformers"),
            _translate_missing_provider_dependency("mlx", "huggingface_hub"),
        ):
            import mlx.core as mx  # type: ignore[import-not-found, import-untyped, unused-ignore]
            from huggingface_hub import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
                snapshot_download,
            )
            from mlx.utils import tree_flatten  # type: ignore[import-not-found, import-untyped, unused-ignore]
            from mlx_vlm.embedding_loader import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
                load_embedding_model,
            )

        if not mx.metal.is_available():
            raise EmbeddingConfigurationError("MLX requires an available Apple Metal GPU")
        self._mx = mx
        try:
            snapshot = Path(
                snapshot_download(
                    repo_id=MODEL_ID,
                    revision=self._options["revision"],
                    local_files_only=self._options["local_files_only"],
                )
            )
            config = json.loads((snapshot / "config.json").read_text())
            if (
                config.get("model_type") != "embedding_gemma2"
                or config.get("text_config", {}).get("embedding_dim") != DIMENSIONS
                or config.get("dtype") != "bfloat16"
                or config.get("quantization") is not None
                or config.get("quantization_config") is not None
            ):
                raise EmbeddingConfigurationError(
                    "MLX requires an unquantized BF16 EmbeddingGemma 2 checkpoint with 768 dimensions"
                )
            with mx.stream(mx.gpu):
                self._model = load_embedding_model(snapshot, lazy=False, strict=True)
                weights = dict(tree_flatten(self._model.parameters()))
                for key in ("language_model.embed_tokens.weight", "language_model.embedding_projection.weight"):
                    if key not in weights or weights[key].dtype != mx.bfloat16:
                        raise EmbeddingConfigurationError("MLX requires BF16 text embedding and projection weights")
                if any(
                    mx.issubdtype(weight.dtype, mx.floating) and weight.dtype not in (mx.bfloat16, mx.float32)
                    for weight in weights.values()
                ):
                    raise EmbeddingConfigurationError("MLX checkpoint floating weights must be BF16 or FP32")
                self._model.eval()
                self._load_processor(snapshot, config, weights)
            mx.synchronize(mx.gpu)
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                setattr(error, "cleanup_errors", (cleanup_error,))
            raise

    def _load_processor(self, snapshot: Path, config: dict[str, Any], weights: dict[str, Any]) -> None:
        with _translate_missing_provider_dependency("mlx", "transformers"):
            from transformers import AutoTokenizer  # type: ignore[import-not-found, import-untyped, unused-ignore]

        self._tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)

    @staticmethod
    def _validate_output(values: np.ndarray, count: int) -> list[np.ndarray]:
        if values.shape != (count, DIMENSIONS):
            raise ValueError("MLX text_embeds must have shape (input rows, 768)")
        if not np.isfinite(values).all():
            raise ValueError("MLX text_embeds must be finite")
        norms = np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= 0) or np.any(np.abs(norms - 1) > 1e-2):
            raise ValueError("MLX text_embeds must have nonzero unit norms within 1e-2")
        normalized = (values / norms).astype(np.float32)
        if np.any(np.abs(np.linalg.norm(normalized.astype(np.float64), axis=1) - 1) > 1e-4):
            raise ValueError("MLX float32 output norms must be within 1e-4 of one")
        return list(normalized)

    def _infer(self, encoded: Mapping[str, np.ndarray], count: int) -> list[np.ndarray]:
        mx = self._mx
        with mx.stream(mx.gpu):
            inputs = {key: mx.array(value) for key, value in encoded.items()}
            values = self._model(**inputs).text_embeds.astype(mx.float32)
            mx.eval(values)
            vectors = np.array(values, dtype=np.float32, copy=True)
        return self._validate_output(vectors, count)

    def _validate_media_tokens(
        self, encoded: Mapping[str, Any], keys: frozenset[str], count: int, kind: str
    ) -> dict[str, np.ndarray]:
        if not isinstance(encoded, Mapping) or set(encoded) != keys:
            raise EmbeddingConfigurationError(f"MLX {kind} processor returned unexpected model inputs")
        inputs = {key: np.asarray(value) for key, value in encoded.items()}
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        if (
            ids.ndim != 2
            or ids.shape != mask.shape
            or ids.shape[0] != count
            or ids.shape[1] == 0
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0)
            or not (np.issubdtype(mask.dtype, np.integer) or mask.dtype == np.bool_)
            or not np.isin(mask, (0, 1)).all()
            or not (mask.sum(axis=1) > 0).all()
        ):
            raise EmbeddingConfigurationError(f"MLX {kind} processor requires aligned integer IDs and binary masks")
        if ids.shape[1] > self._options["max_length"]:
            raise EmbeddingConfigurationError(f"MLX {kind} exceeds max_length after media token expansion")
        return inputs

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        error: BaseException | None = None
        try:
            if self._mx is not None:
                self._mx.synchronize(self._mx.gpu)
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._model = self._tokenizer = self._processor = None
            gc.collect()
            if self._mx is not None:
                try:
                    self._mx.clear_cache()
                except BaseException as cleanup_error:
                    if error is None:
                        raise
                    setattr(error, "cleanup_errors", (*getattr(error, "cleanup_errors", ()), cleanup_error))

    async def aclose(self) -> None:
        self.close()


class MLXTextEmbedder(_MLXEmbedder):
    def _prepare_text_inputs(self, texts: list[str]) -> dict[str, Any]:
        prefix = "task: search result | query: " if self._options["input_type"] == "query" else "title: none | text: "
        rows = []
        for text in texts:
            row = self._tokenizer(prefix + text, truncation=False, padding=False)
            length = len(row["input_ids"])
            if not 0 < length <= self._options["max_length"]:
                raise EmbeddingConfigurationError(
                    "MLX text exceeds max_length including its retrieval prefix and special tokens"
                )
            rows.append(row)
        encoded = self._tokenizer.pad(rows, padding=True, return_tensors="np")
        ids, mask = np.asarray(encoded["input_ids"]), np.asarray(encoded["attention_mask"])
        if (
            ids.ndim != 2
            or ids.shape != mask.shape
            or ids.shape[0] != len(texts)
            or ids.shape[1] > self._options["max_length"]
            or not np.issubdtype(ids.dtype, np.integer)
            or not np.isin(mask, (0, 1)).all()
            or not (mask.sum(axis=1) > 0).all()
        ):
            raise EmbeddingConfigurationError("MLX tokenizer must return aligned integer input IDs and attention masks")
        return {"input_ids": ids, "attention_mask": mask}

    def embed_text(self, texts: list[str]) -> list[np.ndarray]:
        if self._closed:
            raise RuntimeError("MLX embedder is closed")
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise EmbeddingConfigurationError("MLX embed_text requires a list of strings")
        if not texts:
            return []
        return self._infer(self._prepare_text_inputs(texts), len(texts))


class MLXImageEmbedder(_MLXEmbedder):
    _validate_options = staticmethod(_validate_mlx_image_options)

    def _load_processor(self, snapshot: Path, config: dict[str, Any], weights: dict[str, Any]) -> None:
        vision = config.get("vision_config") or {}
        if any(
            vision.get(key) != value
            for key, value in {
                "model_type": "gemma4_vision",
                "dtype": "bfloat16",
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "default_output_length": 280,
            }.items()
        ):
            raise EmbeddingConfigurationError("MLX image encoding requires the fixed BF16 Gemma 4 vision configuration")
        if self._model.vision_tower is None or self._model.embed_vision is None:
            raise EmbeddingConfigurationError("MLX image encoding requires the vision tower and projection")
        if not any(key.startswith("vision_tower.") for key in weights) or (
            "embed_vision.embedding_projection.weight" not in weights
            or weights["embed_vision.embedding_projection.weight"].dtype != self._mx.bfloat16
        ):
            raise EmbeddingConfigurationError("MLX image encoding requires trained BF16 vision and projection weights")
        processor_config = json.loads((snapshot / "processor_config.json").read_text()).get("image_processor", {})
        expected = {
            "max_soft_tokens": 280,
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "do_convert_rgb": True,
            "do_resize": True,
            "do_rescale": True,
            "do_normalize": False,
            "rescale_factor": 1 / 255,
        }
        if any(processor_config.get(key) != value for key, value in expected.items()):
            raise EmbeddingConfigurationError("MLX image encoding requires the fixed checkpoint image processor")
        with (
            _translate_missing_provider_dependency("mlx", "mlx_vlm"),
            _translate_missing_provider_dependency("mlx", "PIL"),
        ):
            from mlx_vlm.models.embedding_gemma2 import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
                EmbeddingGemma2Processor,
            )

            self._processor = EmbeddingGemma2Processor.from_pretrained(
                str(snapshot), local_files_only=True, trust_remote_code=False
            )

    def _validate_image_inputs(self, encoded: Mapping[str, Any], count: int) -> dict[str, np.ndarray]:
        if set(encoded) != _IMAGE_INPUT_KEYS:
            raise EmbeddingConfigurationError("MLX image processor must return exactly the four declared model inputs")
        inputs = {key: np.asarray(value) for key, value in encoded.items()}
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        if (
            ids.ndim != 2
            or ids.shape != mask.shape
            or ids.shape[0] != count
            or ids.shape[1] == 0
            or not np.issubdtype(ids.dtype, np.integer)
            or not (np.issubdtype(mask.dtype, np.integer) or mask.dtype == np.bool_)
            or np.any(ids < 0)
            or not np.isin(mask, (0, 1)).all()
            or not (mask.sum(axis=1) > 0).all()
        ):
            raise EmbeddingConfigurationError("MLX image processor must return aligned integer IDs and binary masks")
        if ids.shape[1] > self._options["max_length"]:
            raise EmbeddingConfigurationError("MLX image exceeds max_length after visual token expansion")
        pixels, positions = inputs["pixel_values"], inputs["image_position_ids"]
        if (
            pixels.ndim != 3
            or pixels.shape[0] != count
            or not 0 < pixels.shape[1] <= 2520
            or pixels.shape[2] != 16 * 16 * 3
            or not np.issubdtype(pixels.dtype, np.floating)
            or pixels.dtype == np.float16
            or not np.isfinite(pixels).all()
            or positions.shape != (*pixels.shape[:2], 2)
            or not np.issubdtype(positions.dtype, np.integer)
        ):
            raise EmbeddingConfigurationError(
                "MLX image processor must return finite floating patches and integer positions"
            )
        valid = (positions >= 0).all(axis=-1)
        padding = (positions == -1).all(axis=-1)
        if not (valid | padding).all() or not valid.any(axis=1).all() or np.any(positions[valid] >= 10240):
            raise EmbeddingConfigurationError(
                "MLX image patch positions must be valid coordinates or paired -1 padding"
            )
        return inputs

    def embed_image(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if self._closed:
            raise RuntimeError("MLX embedder is closed")
        if not isinstance(images, list):
            raise EmbeddingConfigurationError("MLX embed_image requires a list of RGB uint8 HWC arrays")
        for value in images:
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.uint8
                or value.ndim != 3
                or value.shape[2] != 3
                or value.shape[0] <= 0
                or value.shape[1] <= 0
                or value.shape[0] * value.shape[1] > MAX_IMAGE_PIXELS
            ):
                raise EmbeddingConfigurationError(
                    "MLX images require RGB uint8 HWC arrays within the 16 Mi pixel budget"
                )
        if not images:
            return []
        with _translate_missing_provider_dependency("mlx", "PIL"):
            from PIL import Image  # type: ignore[import-not-found, import-untyped, unused-ignore]

        with ExitStack() as stack:
            pil_images = []
            for value in images:
                image = Image.fromarray(np.ascontiguousarray(value))
                stack.callback(image.close)
                pil_images.append([image])
            encoded = self._processor(
                images=pil_images,
                # PIL is converted back to HWC arrays by the SDK; heights 1/3
                # are ambiguous to its channel-axis inference.
                images_kwargs={"input_data_format": "channels_last"},
                return_tensors="np",
                padding=True,
                truncation=False,
            )
            return self._infer(self._validate_image_inputs(encoded, len(images)), len(images))


class MLXAudioEmbedder(_MLXEmbedder):
    _validate_options = staticmethod(_validate_mlx_audio_options)

    def _load_processor(self, snapshot: Path, config: dict[str, Any], weights: dict[str, Any]) -> None:
        expected_audio = {
            "model_type": "gemma4_audio",
            "dtype": "bfloat16",
            "hidden_size": 1024,
            "num_hidden_layers": 12,
            "output_proj_dims": 1536,
            "subsampling_conv_channels": [128, 32],
        }
        audio = config.get("audio_config") or {}
        if any(audio.get(key) != value for key, value in expected_audio.items()):
            raise EmbeddingConfigurationError("MLX audio requires the fixed BF16 Gemma 4 audio configuration")
        if getattr(self._model, "audio_tower", None) is None or getattr(self._model, "embed_audio", None) is None:
            raise EmbeddingConfigurationError("MLX audio requires the audio tower and projection")
        if not any(key.startswith("audio_tower.") for key in weights) or (
            "embed_audio.embedding_projection.weight" not in weights
            or weights["embed_audio.embedding_projection.weight"].dtype != self._mx.bfloat16
        ):
            raise EmbeddingConfigurationError("MLX audio requires trained BF16 audio and projection weights")
        expected = {
            "sampling_rate": AUDIO_SAMPLE_RATE,
            "feature_size": 128,
            "frame_length": 320,
            "hop_length": 160,
            "fft_length": 512,
            "dither": 0,
            "padding_side": "right",
            "preemphasis": 0,
            "input_scale_factor": 1,
        }
        extractor_config = json.loads((snapshot / "processor_config.json").read_text()).get("feature_extractor", {})
        if any(extractor_config.get(key) != value for key, value in expected.items()):
            raise EmbeddingConfigurationError("MLX audio requires the fixed checkpoint feature extractor")
        with (
            _translate_missing_provider_dependency("mlx", "mlx_vlm"),
            _translate_missing_provider_dependency("mlx", "PIL"),
        ):
            from mlx_vlm.models.embedding_gemma2 import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
                EmbeddingGemma2Processor,
            )

            self._processor = EmbeddingGemma2Processor.from_pretrained(
                str(snapshot), local_files_only=True, trust_remote_code=False
            )
        if any(getattr(self._processor.feature_extractor, key, None) != value for key, value in expected.items()):
            raise EmbeddingConfigurationError("MLX audio feature extractor differs from the fixed checkpoint")
        if self._processor.audio_token_id != 258881 or config.get("audio_token_id") != 258881:
            raise EmbeddingConfigurationError("MLX audio requires the fixed checkpoint audio token")

    def _validate_audio_inputs(self, encoded: Mapping[str, Any], lengths: list[int]) -> dict[str, np.ndarray]:
        inputs = self._validate_media_tokens(encoded, _AUDIO_INPUT_KEYS, len(lengths), "audio")
        features, mask = inputs["input_features"], inputs["input_features_mask"]
        # The fixed extractor pads the waveform to 128 samples, then unfolds
        # 321-sample windows after 160 samples of left padding. Check T too:
        # checking only the returned mask's prefix would miss a truncated tail.
        padded_samples = ((max(lengths) + 127) // 128) * 128
        feature_length = (padded_samples - 161) // 160 + 1
        if (
            features.shape != (len(lengths), feature_length, 128)
            or features.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not np.isfinite(features).all()
            or mask.shape != features.shape[:2]
            or not (np.issubdtype(mask.dtype, np.integer) or mask.dtype == np.bool_)
            or not np.isin(mask, (0, 1)).all()
        ):
            raise EmbeddingConfigurationError("MLX audio requires complete finite mel features and binary masks")
        expected = np.arange(feature_length)[None, :] * 160 + 160 < np.asarray(lengths)[:, None]
        if not expected.any(axis=1).all() or not np.array_equal(mask, expected):
            raise EmbeddingConfigurationError("MLX audio mask must preserve every original waveform's valid prefix")
        token_positions = inputs["input_ids"] == self._processor.audio_token_id
        if (
            not np.array_equal(token_positions.sum(axis=1), expected[:, ::4].sum(axis=1))
            or not (inputs["attention_mask"][token_positions] == 1).all()
        ):
            raise EmbeddingConfigurationError("MLX audio tokens must match the twice-subsampled valid mel mask")
        return inputs

    def embed_audio(self, clips: list[AudioClip]) -> list[np.ndarray]:
        if self._closed:
            raise RuntimeError("MLX embedder is closed")
        if not isinstance(clips, list):
            raise EmbeddingConfigurationError("MLX embed_audio requires a list of AudioClip values")
        for clip in clips:
            if not isinstance(clip, AudioClip):
                raise EmbeddingConfigurationError("MLX embed_audio requires AudioClip values")
            if (
                isinstance(clip.sample_rate, (bool, np.bool_))
                or not isinstance(clip.sample_rate, (int, np.integer))
                or clip.sample_rate != AUDIO_SAMPLE_RATE
            ):
                raise EmbeddingConfigurationError("MLX audio requires sample_rate=16000; resample first")
            samples = clip.samples
            if (
                not isinstance(samples, np.ndarray)
                or samples.ndim != 2
                or samples.shape[1] != 1
                or samples.dtype not in (np.dtype("float32"), np.dtype("float64"))
            ):
                raise EmbeddingConfigurationError("MLX audio requires mono float32/float64 PCM shaped (samples, 1)")
            if not MIN_AUDIO_SAMPLES <= len(samples) <= MAX_AUDIO_SAMPLES:
                raise EmbeddingConfigurationError(
                    "MLX audio requires 161 to 480000 samples; short tails and truncation are unsupported"
                )
            if samples.nbytes > MAX_AUDIO_INPUT_BYTES:
                raise EmbeddingConfigurationError("MLX audio exceeds the 4 MiB decoded input budget")
            if not np.isfinite(samples).all() or np.any(samples < -1) or np.any(samples > 1):
                raise EmbeddingConfigurationError("MLX audio requires finite floating PCM in [-1, 1]")
        if not clips:
            return []
        # Validate the entire batch before allocating independent SDK waveforms.
        waveforms = [np.array(clip.samples[:, 0], dtype=np.float32, order="C", copy=True) for clip in clips]
        encoded = self._processor(
            audio=[[waveform] for waveform in waveforms],
            text_kwargs={"padding": True, "truncation": False},
            audio_kwargs={
                "sampling_rate": AUDIO_SAMPLE_RATE,
                "padding": "longest",
                "truncation": False,
                "pad_to_multiple_of": 128,
                "return_attention_mask": True,
            },
            return_tensors="np",
        )
        return self._infer(self._validate_audio_inputs(encoded, [len(clip.samples) for clip in clips]), len(clips))


class MLXVideoEmbedder(MLXImageEmbedder):
    """Share fixed vision loading; encode clips through the SDK video processor."""

    _validate_options = staticmethod(_validate_mlx_video_options)

    def _load_processor(self, snapshot: Path, config: dict[str, Any], weights: dict[str, Any]) -> None:
        super()._load_processor(snapshot, config, weights)
        expected = {
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "max_soft_tokens": 140,
            "do_resize": True,
            "do_rescale": True,
            "do_convert_rgb": True,
            "do_normalize": False,
            "rescale_factor": 1 / 255,
            "add_timestamps": False,
        }
        video_config = json.loads((snapshot / "processor_config.json").read_text()).get("video_processor", {})
        if any(video_config.get(key) != value for key, value in expected.items()):
            raise EmbeddingConfigurationError("MLX video requires the fixed checkpoint video processor")
        if any(getattr(self._processor.video_processor, key, None) != value for key, value in expected.items()):
            raise EmbeddingConfigurationError("MLX video processor differs from the fixed checkpoint")
        if self._processor.video_token_id != 258884 or config.get("video_token_id") != 258884:
            raise EmbeddingConfigurationError("MLX video requires the fixed checkpoint video token")

    def _validate_video_inputs(self, encoded: Mapping[str, Any], counts: list[int]) -> dict[str, np.ndarray]:
        inputs = self._validate_media_tokens(encoded, _VIDEO_INPUT_KEYS, len(counts), "video")
        frame_counts = inputs.pop("num_frames_per_video")
        if (
            frame_counts.shape != (len(counts),)
            or not np.issubdtype(frame_counts.dtype, np.integer)
            or not np.array_equal(frame_counts, counts)
        ):
            raise EmbeddingConfigurationError("MLX video processor must preserve each clip's frame count")
        pixels, positions = inputs["pixel_values_videos"], inputs["video_position_ids"]
        if (
            pixels.ndim != 3
            or pixels.shape[0] != sum(counts)
            or not 0 < pixels.shape[1] <= 1260
            or pixels.shape[2] != 768
            or pixels.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not np.isfinite(pixels).all()
            or positions.shape != (*pixels.shape[:2], 2)
            or not np.issubdtype(positions.dtype, np.integer)
        ):
            raise EmbeddingConfigurationError("MLX video requires finite patches and positions for every input frame")
        valid, padding = (positions >= 0).all(axis=-1), (positions == -1).all(axis=-1)
        if not (valid | padding).all() or not valid.any(axis=1).all() or np.any(positions[valid] >= 10240):
            raise EmbeddingConfigurationError(
                "MLX video patch positions must be valid coordinates or paired -1 padding"
            )
        per_frame_tokens = []
        for frame_positions, frame_valid in zip(positions, valid):
            coords = frame_positions[frame_valid]
            if not np.array_equal(frame_valid, np.arange(len(frame_valid)) < len(coords)):
                raise EmbeddingConfigurationError("MLX video patch padding must follow the complete frame grid")
            width, height = (int(value) + 1 for value in coords.max(axis=0))
            # Complete row-major 3x3 pooling grids prevent silently omitted or
            # repeated patches from satisfying only a total-token check.
            if height % 3 or width % 3 or len(coords) != height * width:
                raise EmbeddingConfigurationError("MLX video patches must form complete 3x3 pooling grids")
            # The SDK stores (x, y), with x advancing first in each row.
            expected = np.column_stack((np.tile(np.arange(width), height), np.repeat(np.arange(height), width)))
            if not np.array_equal(coords, expected):
                raise EmbeddingConfigurationError("MLX video patch positions must preserve the ordered frame grid")
            per_frame_tokens.append(height * width // 9)
        offset = 0
        for row, count in enumerate(counts):
            tokens = inputs["input_ids"][row] == self._processor.video_token_id
            if (
                int(tokens.sum()) != sum(per_frame_tokens[offset : offset + count])
                or not (inputs["attention_mask"][row][tokens] == 1).all()
            ):
                raise EmbeddingConfigurationError("MLX video tokens must match every frame's pooled patch grid")
            offset += count
        return inputs

    def embed_video(self, clips: list[VideoClip]) -> list[np.ndarray]:
        if self._closed:
            raise RuntimeError("MLX embedder is closed")
        if not isinstance(clips, list):
            raise EmbeddingConfigurationError("MLX embed_video requires a list of VideoClip values")
        for clip in clips:
            if not isinstance(clip, VideoClip):
                raise EmbeddingConfigurationError("MLX embed_video requires VideoClip values")
            if not all(
                isinstance(value, (tuple, list)) for value in (clip.frames, clip.frame_times, clip.frame_indices)
            ):
                raise EmbeddingConfigurationError("MLX video requires frame, time and index sequences")
            count = len(clip.frames)
            if not 1 <= count <= MAX_VIDEO_FRAMES or len(clip.frame_times) != count or len(clip.frame_indices) != count:
                raise EmbeddingConfigurationError("MLX video requires 1 to 12 frames with aligned times and indices")
            if any(
                isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or index < 0
                for index in clip.frame_indices
            ) or any(a >= b for a, b in zip(clip.frame_indices, clip.frame_indices[1:])):
                raise EmbeddingConfigurationError(
                    "MLX video frame indices must be nonnegative strictly increasing integers"
                )
            if any(
                isinstance(t, (bool, np.bool_))
                or not isinstance(t, (int, float, np.integer, np.floating))
                or not math.isfinite(t)
                or t < 0
                for t in clip.frame_times
            ) or any(a > b for a, b in zip(clip.frame_times, clip.frame_times[1:])):
                raise EmbeddingConfigurationError("MLX video frame times must be finite nonnegative and nondecreasing")
            total_bytes = 0
            shape = None
            for frame in clip.frames:
                if (
                    not isinstance(frame, np.ndarray)
                    or frame.dtype != np.uint8
                    or frame.ndim != 3
                    or frame.shape[2] != 3
                    or frame.shape[0] <= 0
                    or frame.shape[1] <= 0
                    or frame.shape[0] * frame.shape[1] > MAX_IMAGE_PIXELS
                ):
                    raise EmbeddingConfigurationError(
                        "MLX video frames require RGB uint8 HWC within the 16 Mi pixel budget"
                    )
                if shape is not None and frame.shape != shape:
                    raise EmbeddingConfigurationError("MLX video frames must have the same H/W within each clip")
                shape = frame.shape
                total_bytes += frame.nbytes
            if total_bytes > MAX_VIDEO_INPUT_BYTES:
                raise EmbeddingConfigurationError("MLX video exceeds the 64 MiB decoded clip budget")
        if not clips:
            return []
        videos = [np.stack(clip.frames) for clip in clips]
        encoded = self._processor(
            videos=videos,
            text_kwargs={"padding": True, "truncation": False},
            videos_kwargs={
                "input_data_format": "channels_last",
                "do_sample_frames": False,
                "add_timestamps": False,
                "return_metadata": False,
            },
            return_tensors="np",
        )
        return self._infer(self._validate_video_inputs(encoded, [len(clip.frames) for clip in clips]), len(clips))
