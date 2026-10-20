# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""EmbeddingGemma 2 text and image encoding on Metal, loaded only by workers."""

from __future__ import annotations

import gc
import json
import platform
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._embedding_outputs import EmbeddingOutput, EmbeddingOutputSpec
from vane.ai.options import _validate_mlx_image_options, _validate_mlx_text_options
from vane.ai.protocols import ImageEmbedderDescriptor, TextEmbedderDescriptor
from vane.ai.provider import Provider, _translate_missing_provider_dependency
from vane.ai.typing import UDFOptions

MODEL_ID = "mlx-community/embeddinggemma-2-bf16"
DIMENSIONS = 768
MAX_IMAGE_PIXELS = 16 * 1024 * 1024
_IMAGE_INPUT_KEYS = frozenset({"input_ids", "attention_mask", "pixel_values", "image_position_ids"})


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
