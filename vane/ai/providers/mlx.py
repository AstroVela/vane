# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""EmbeddingGemma 2 text encoding on Apple Metal, loaded only by the worker."""

from __future__ import annotations

import gc
import json
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai.options import _validate_mlx_text_options
from vane.ai.protocols import TextEmbedderDescriptor
from vane.ai.provider import Provider, _translate_missing_provider_dependency
from vane.ai.typing import UDFOptions

MODEL_ID = "mlx-community/embeddinggemma-2-bf16"
DIMENSIONS = 768


class MLXTextEmbedderDescriptor(TextEmbedderDescriptor):
    def __init__(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> None:
        if model is not None and model != MODEL_ID:
            raise EmbeddingConfigurationError("MLX PR1 supports only mlx-community/embeddinggemma-2-bf16")
        if dimensions is not None and (type(dimensions) is not int or dimensions != DIMENSIONS):
            raise EmbeddingConfigurationError("MLX EmbeddingGemma 2 requires dimensions=768")
        self._options = _validate_mlx_text_options(options if options is not None else {})

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

    def supports_chunking(self) -> bool:
        return False

    def instantiate(self) -> MLXTextEmbedder:
        return MLXTextEmbedder(self.get_options())


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


class MLXTextEmbedder:
    def __init__(self, options: Mapping[str, Any]) -> None:
        self._options = _validate_mlx_text_options(options)
        self._model: Any = None
        self._tokenizer: Any = None
        self._mx: Any = None
        self._closed = False
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise EmbeddingConfigurationError("MLX PR1 inference requires Apple Silicon macOS arm64 and Metal")
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
            from transformers import AutoTokenizer  # type: ignore[import-not-found, import-untyped, unused-ignore]

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
                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(snapshot), local_files_only=True, trust_remote_code=False
                )
            mx.synchronize(mx.gpu)
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                setattr(error, "cleanup_errors", (cleanup_error,))
            raise

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

    def embed_text(self, texts: list[str]) -> list[np.ndarray]:
        if self._closed:
            raise RuntimeError("MLX embedder is closed")
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise EmbeddingConfigurationError("MLX embed_text requires a list of strings")
        if not texts:
            return []
        encoded = self._prepare_text_inputs(texts)
        mx = self._mx
        with mx.stream(mx.gpu):
            inputs = {key: mx.array(value) for key, value in encoded.items()}
            values = self._model(**inputs).text_embeds.astype(mx.float32)
            mx.eval(values)
            vectors = np.array(values, dtype=np.float32, copy=True)
        return self._validate_output(vectors, len(texts))

    def close(self) -> None:
        if self._closed:
            return
        if self._mx is not None:
            self._mx.synchronize(self._mx.gpu)
        self._model = self._tokenizer = None
        gc.collect()
        if self._mx is not None:
            self._mx.clear_cache()
        self._closed = True

    async def aclose(self) -> None:
        self.close()
