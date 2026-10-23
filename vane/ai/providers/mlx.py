# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""EmbeddingGemma 2 unary and ordered joint encoding on worker-local Metal."""

from __future__ import annotations

import gc
import json
import platform
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from vane.ai._audio_embedding import AudioClip, AudioInputSpec
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPartSpec, MultimodalInputSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._embedding_outputs import EmbeddingOutput, EmbeddingOutputSpec
from vane.ai._video_embedding import VideoClip, VideoInputSpec
from vane.ai.options import (
    _validate_mlx_audio_options,
    _validate_mlx_image_options,
    _validate_mlx_multimodal_options,
    _validate_mlx_text_options,
    _validate_mlx_video_options,
)
from vane.ai.protocols import (
    AudioEmbedderDescriptor,
    ImageEmbedderDescriptor,
    MultimodalEmbedderDescriptor,
    TextEmbedderDescriptor,
    VideoEmbedderDescriptor,
)
from vane.ai.provider import Provider, _translate_missing_provider_dependency
from vane.ai.providers._mlx_media import (
    AUDIO_SAMPLE_RATE,
    MAX_AUDIO_INPUT_BYTES,
    MAX_AUDIO_SAMPLES,
    MAX_IMAGE_PIXELS,
    MAX_VIDEO_FRAMES,
    MAX_VIDEO_INPUT_BYTES,
    audio_tokens,
    patch_tokens,
    validate_audio,
    validate_frame_counts,
    validate_image,
    validate_video,
)
from vane.ai.providers._mlx_multimodal import MEDIA_TOKENS, JointPlan, joint_plan, validate_structure
from vane.ai.typing import UDFOptions

MODEL_ID = "mlx-community/embeddinggemma-2-bf16"
DIMENSIONS = 768
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


class MLXMultimodalEmbedderDescriptor(_MLXDescriptor, MultimodalEmbedderDescriptor):
    _validate_options = staticmethod(_validate_mlx_multimodal_options)

    def get_input_spec(self) -> MultimodalInputSpec:
        return MultimodalInputSpec(
            supported_parts=frozenset(
                {
                    EmbeddingPartSpec("text"),
                    EmbeddingPartSpec("image"),
                    EmbeddingPartSpec("video"),
                    EmbeddingPartSpec("audio"),
                }
            ),
            video=VideoInputSpec(max_frames=MAX_VIDEO_FRAMES, max_input_bytes=MAX_VIDEO_INPUT_BYTES),
            audio=AudioInputSpec(
                sample_rate=AUDIO_SAMPLE_RATE,
                max_samples=MAX_AUDIO_SAMPLES,
                max_channels=1,
                max_input_bytes=MAX_AUDIO_INPUT_BYTES,
            ),
            max_parts=16,
            max_input_bytes=64 * 1024**2,
        )

    def get_output_spec(self, *, dimensions: int | None = None) -> EmbeddingOutputSpec:
        if dimensions is not None and (type(dimensions) is not int or dimensions != DIMENSIONS):
            raise EmbeddingConfigurationError("MLX EmbeddingGemma 2 requires dimensions=768")
        return EmbeddingOutputSpec(
            (EmbeddingOutput("embedding", DIMENSIONS, MODEL_ID, ("text", "image", "video", "audio")),)
        )

    def instantiate(self) -> MLXMultimodalEmbedder:
        return MLXMultimodalEmbedder(self.get_options())


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

    def get_multimodal_embedder(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> MLXMultimodalEmbedderDescriptor:
        return MLXMultimodalEmbedderDescriptor(model, dimensions, options=options)


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
    def _prepare_text_inputs(self, texts: list[str], *, input_type: str | None = None) -> dict[str, Any]:
        role = self._options.get("input_type", "document") if input_type is None else input_type
        if not isinstance(role, str) or role not in {"query", "document"}:
            raise EmbeddingConfigurationError("MLX input_type must be one of: document, query")
        prefix = "task: search result | query: " if role == "query" else "title: none | text: "
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

    def embed_text(self, texts: list[str], *, input_type: str | None = None) -> list[np.ndarray]:
        if self._closed:
            raise RuntimeError("MLX embedder is closed")
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise EmbeddingConfigurationError("MLX embed_text requires a list of strings")
        if not texts:
            return []
        return self._infer(self._prepare_text_inputs(texts, input_type=input_type), len(texts))


class _MLXMediaEmbedder(_MLXEmbedder):
    """One processor factory and selected tower checks for every media encoder."""

    _media_kinds: frozenset[str] = frozenset()

    def _load_processor(self, snapshot: Path, config: dict[str, Any], weights: dict[str, Any]) -> None:
        media_config = json.loads((snapshot / "processor_config.json").read_text())
        if "image" in self._media_kinds:
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
                raise EmbeddingConfigurationError(
                    "MLX image encoding requires the fixed BF16 Gemma 4 vision configuration"
                )
            if self._model.vision_tower is None or self._model.embed_vision is None:
                raise EmbeddingConfigurationError("MLX image encoding requires the vision tower and projection")
            if not any(key.startswith("vision_tower.") for key in weights) or (
                "embed_vision.embedding_projection.weight" not in weights
                or weights["embed_vision.embedding_projection.weight"].dtype != self._mx.bfloat16
            ):
                raise EmbeddingConfigurationError(
                    "MLX image encoding requires trained BF16 vision and projection weights"
                )
            processor_config = media_config.get("image_processor", {})
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
        if "audio" in self._media_kinds:
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
            expected_extractor = {
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
            extractor_config = media_config.get("feature_extractor", {})
            if any(extractor_config.get(key) != value for key, value in expected_extractor.items()):
                raise EmbeddingConfigurationError("MLX audio requires the fixed checkpoint feature extractor")
        if "video" in self._media_kinds:
            expected_video = {
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
            video_config = media_config.get("video_processor", {})
            if any(video_config.get(key) != value for key, value in expected_video.items()):
                raise EmbeddingConfigurationError("MLX video requires the fixed checkpoint video processor")
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
        if "audio" in self._media_kinds:
            if any(
                getattr(self._processor.feature_extractor, key, None) != value
                for key, value in expected_extractor.items()
            ):
                raise EmbeddingConfigurationError("MLX audio feature extractor differs from the fixed checkpoint")
            if self._processor.audio_token_id != 258881 or config.get("audio_token_id") != 258881:
                raise EmbeddingConfigurationError("MLX audio requires the fixed checkpoint audio token")
        if "video" in self._media_kinds:
            if any(
                getattr(self._processor.video_processor, key, None) != value for key, value in expected_video.items()
            ):
                raise EmbeddingConfigurationError("MLX video processor differs from the fixed checkpoint")
            if self._processor.video_token_id != 258884 or config.get("video_token_id") != 258884:
                raise EmbeddingConfigurationError("MLX video requires the fixed checkpoint video token")


class MLXImageEmbedder(_MLXMediaEmbedder):
    _media_kinds = frozenset({"image"})

    _validate_options = staticmethod(_validate_mlx_image_options)

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
        patch_tokens(
            inputs["pixel_values"],
            inputs["image_position_ids"],
            count,
            max_patches=2520,
            kind="image",
            strict_grid=False,
        )
        return inputs

    def embed_image(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if self._closed:
            raise RuntimeError("MLX embedder is closed")
        if not isinstance(images, list):
            raise EmbeddingConfigurationError("MLX embed_image requires a list of RGB uint8 HWC arrays")
        for value in images:
            validate_image(value, max_pixels=MAX_IMAGE_PIXELS)
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


class MLXAudioEmbedder(_MLXMediaEmbedder):
    _media_kinds = frozenset({"audio"})

    _validate_options = staticmethod(_validate_mlx_audio_options)

    def _validate_audio_inputs(self, encoded: Mapping[str, Any], lengths: list[int]) -> dict[str, np.ndarray]:
        inputs = self._validate_media_tokens(encoded, _AUDIO_INPUT_KEYS, len(lengths), "audio")
        expected_tokens = audio_tokens(inputs, lengths)
        token_positions = inputs["input_ids"] == self._processor.audio_token_id
        if (
            not np.array_equal(token_positions.sum(axis=1), expected_tokens)
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
            validate_audio(clip, max_input_bytes=MAX_AUDIO_INPUT_BYTES)
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

    _media_kinds = frozenset({"image", "video"})

    _validate_options = staticmethod(_validate_mlx_video_options)

    def _validate_video_inputs(self, encoded: Mapping[str, Any], counts: list[int]) -> dict[str, np.ndarray]:
        inputs = self._validate_media_tokens(encoded, _VIDEO_INPUT_KEYS, len(counts), "video")
        frame_counts = inputs.pop("num_frames_per_video")
        validate_frame_counts(frame_counts, counts)
        per_frame_tokens = patch_tokens(
            inputs["pixel_values_videos"], inputs["video_position_ids"], sum(counts), max_patches=1260, kind="video"
        )
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
            validate_video(clip, max_pixels=MAX_IMAGE_PIXELS, max_input_bytes=MAX_VIDEO_INPUT_BYTES)
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


class MLXMultimodalEmbedder(_MLXMediaEmbedder):
    """One ordered document input per row, one shared model forward per batch."""

    _validate_options = staticmethod(_validate_mlx_multimodal_options)
    _media_kinds = frozenset({"image", "video", "audio"})

    @staticmethod
    def _validate_output(values: np.ndarray, count: int) -> list[np.ndarray]:
        try:
            return _MLXEmbedder._validate_output(values, count)
        except ValueError:
            # A broken joint output contract must not become nullable rows or
            # recursive smaller forwards under the public ignore policy.
            raise EmbeddingConfigurationError(
                "MLX joint model outputs violate the finite float32[768] unit-vector contract"
            ) from None

    def _load_processor(self, snapshot: Path, config: dict[str, Any], weights: dict[str, Any]) -> None:
        super()._load_processor(snapshot, config, weights)
        self._tokenizer = self._processor.tokenizer
        for name, (token, token_id) in MEDIA_TOKENS.items():
            if (
                getattr(self._processor, name, None) != token
                or self._tokenizer.convert_tokens_to_ids(token) != token_id
            ):
                raise EmbeddingConfigurationError("MLX joint encoding requires the fixed processor media vocabulary")
        if (
            self._processor.image_token_id != 258880
            or config.get("image_token_id") != 258880
            or self._tokenizer.pad_token_id != 0
            or self._tokenizer.bos_token_id != 2
            or self._tokenizer.eos_token_id != 1
        ):
            raise EmbeddingConfigurationError("MLX joint encoding requires the fixed checkpoint token identities")

    def _prepare_multimodal_inputs(self, plan: JointPlan) -> dict[str, np.ndarray]:
        prompts = []
        for row_index, row in enumerate(plan.rows):
            has_text = any(part.kind == "text" for part in row)
            content = [dict(type=part.kind, **({"text": part.text} if part.kind == "text" else {})) for part in row]
            conversation: list[dict[str, object]] = []
            if has_text:
                conversation.append({"role": "system", "content": "title: none | text: "})
            conversation.append({"role": "user", "content": content})
            prompt = self._processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            expected = ("title: none | text: " if has_text else "") + "".join(
                part.text if part.kind == "text" else MEDIA_TOKENS[part.kind + "_token"][0] for part in row
            )
            if not isinstance(prompt, str) or prompt != expected:
                raise EmbeddingConfigurationError(f"MLX joint row {row_index} template changed the ordered parts")
            prompts.append(prompt)
        kwargs: dict[str, object] = dict(
            text=prompts, return_tensors="np", text_kwargs={"padding": True, "truncation": False}
        )
        keys = {"input_ids", "attention_mask"}
        if plan.images:
            kwargs.update(
                images=[list(row) for row in plan.images_by_row], images_kwargs={"input_data_format": "channels_last"}
            )
            keys.update(("pixel_values", "image_position_ids"))
        if plan.videos:
            kwargs.update(
                videos=[np.stack(clip.frames) for clip in plan.videos],
                videos_kwargs={
                    "input_data_format": "channels_last",
                    "do_sample_frames": False,
                    "add_timestamps": False,
                    "return_metadata": False,
                },
            )
            keys.update(("pixel_values_videos", "video_position_ids", "num_frames_per_video"))
        if plan.audios:
            kwargs.update(
                audio=[np.array(clip.samples[:, 0], dtype=np.float32, order="C", copy=True) for clip in plan.audios],
                audio_kwargs={
                    "sampling_rate": AUDIO_SAMPLE_RATE,
                    "padding": "longest",
                    "truncation": False,
                    "pad_to_multiple_of": 128,
                    "return_attention_mask": True,
                },
            )
            keys.update(("input_features", "input_features_mask"))
        encoded = self._processor(**kwargs)
        try:
            inputs = self._validate_media_tokens(encoded, frozenset(keys), len(plan.rows), "joint")
        except EmbeddingConfigurationError:
            raise
        except (TypeError, ValueError):
            # NumPy conversion can fail before shape/dtype validation, e.g.
            # for ragged arrays. Preserve the same fatal contract policy.
            raise EmbeddingConfigurationError("MLX joint processor returned invalid array data") from None
        image_tokens = (
            patch_tokens(
                inputs["pixel_values"],
                inputs["image_position_ids"],
                len(plan.images),
                max_patches=2520,
                kind="joint image",
            )
            if plan.images
            else []
        )
        video_tokens = []
        if plan.videos:
            counts = [len(clip.frames) for clip in plan.videos]
            validate_frame_counts(inputs.pop("num_frames_per_video"), counts)
            per_frame = patch_tokens(
                inputs["pixel_values_videos"],
                inputs["video_position_ids"],
                sum(counts),
                max_patches=1260,
                kind="joint video",
            )
            offset = 0
            for count in counts:
                video_tokens.append(per_frame[offset : offset + count])
                offset += count
        audio_counts = audio_tokens(inputs, [len(clip.samples) for clip in plan.audios]) if plan.audios else []
        validate_structure(inputs, plan, image_tokens, video_tokens, audio_counts)
        return inputs

    def embed_multimodal(self, inputs: list[EmbeddingContent]) -> list[np.ndarray]:
        if self._closed:
            raise RuntimeError("MLX embedder is closed")
        plan = joint_plan(inputs, MLXMultimodalEmbedderDescriptor().get_input_spec())
        if not plan.rows:
            return []
        return self._infer(self._prepare_multimodal_inputs(plan), len(plan.rows))


class _MLXModelCore(MLXMultimodalEmbedder, MLXVideoEmbedder, MLXAudioEmbedder, MLXTextEmbedder):
    """One explicit full-media owner with the existing unary and joint operations.

    All operations share the inherited model, processor, forward and close path.
    Native descriptors retain their narrower loading and dependency boundaries.
    The service exclusively owns this core; operations neither construct nor
    close another embedder. All calls must run on its owning execution thread.
    """

    def _load_processor(self, snapshot: Path, config: dict[str, Any], weights: dict[str, Any]) -> None:
        MLXMultimodalEmbedder._load_processor(self, snapshot, config, weights)
        # Keep the original unary AutoTokenizer. Joint rendering continues to
        # use processor.tokenizer; sharing weights need not change text tokens.
        _MLXEmbedder._load_processor(self, snapshot, config, weights)

    def encode(self, content: EmbeddingContent, encoding: str) -> np.ndarray:
        if encoding == "multimodal":
            return self.embed_multimodal([content])[0]
        kind = "text" if encoding in {"text_document", "text_query"} else encoding
        if kind not in {"text", "image", "video", "audio"} or len(content.parts) != 1:
            raise EmbeddingConfigurationError("MLX unary encoding requires exactly one matching part")
        part = content.parts[0]
        if part.spec != EmbeddingPartSpec(kind):
            raise EmbeddingConfigurationError("MLX request encoding does not match its input part")
        if kind == "text":
            return self.embed_text([part.value], input_type="query" if encoding == "text_query" else "document")[0]
        return getattr(self, "embed_" + kind)([part.value])[0]
