# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Cosmos text/video encoders behind the Transformers provider.

Only descriptors are constructed while planning. Model code and tensors stay
inside the executing worker, using the GPU allocated by the UDF runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._video_embedding import VideoClip, VideoInputSpec
from vane.ai.options import validate_embed_options
from vane.ai.protocols import TextEmbedderDescriptor, VideoEmbedderDescriptor
from vane.ai.provider import _translate_missing_provider_dependency
from vane.ai.typing import UDFOptions

COSMOS_MODEL = "nvidia/Cosmos-Embed1-224p"
_MODEL_OPTIONS = frozenset({"cache_folder", "device", "local_files_only", "revision", "trust_remote_code", "dtype"})


@dataclass
class _CosmosDescriptor:
    model: str
    dimensions: int | None = None
    options: dict[str, Any] = field(default_factory=dict)
    provider_name: str = "transformers"

    def __post_init__(self) -> None:
        if self.model != COSMOS_MODEL:
            raise EmbeddingConfigurationError(f"Cosmos embedding supports only {COSMOS_MODEL}")
        if self.dimensions is not None and (type(self.dimensions) is not int or self.dimensions != 256):
            raise EmbeddingConfigurationError("Cosmos-Embed1-224p produces exactly 256 dimensions")
        unknown = sorted(set(self.options) - _MODEL_OPTIONS)
        if unknown:
            raise TypeError(f"Unsupported Cosmos embedding option(s): {', '.join(unknown)}")
        self.options = validate_embed_options("transformers", self.options, relation=False)
        if self.options.get("trust_remote_code") is not True:
            raise EmbeddingConfigurationError("Cosmos requires trust_remote_code=True and a pinned revision")
        if self.options.get("device") not in {"cuda", "cuda:0"}:
            raise EmbeddingConfigurationError("Cosmos requires device='cuda' (the worker's allocated GPU)")
        if self.options.get("dtype") not in {"float32", "float16"}:
            raise EmbeddingConfigurationError("Cosmos requires an explicit dtype='float32' or 'float16'")

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model

    def get_options(self) -> dict[str, Any]:
        return dict(self.options)

    def get_dimensions(self) -> int:
        return 256

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(num_gpus=1)

    def instantiate(self) -> CosmosEmbedder:
        return CosmosEmbedder(self.model, self.options)


class CosmosTextEmbedderDescriptor(_CosmosDescriptor, TextEmbedderDescriptor):
    def supports_chunking(self) -> bool:
        return False


class CosmosVideoEmbedderDescriptor(_CosmosDescriptor, VideoEmbedderDescriptor):
    def get_input_spec(self) -> VideoInputSpec:
        return VideoInputSpec(frame_count=8, max_frames=8)


class CosmosEmbedder:
    def __init__(self, model: str, options: dict[str, Any]) -> None:
        with (
            _translate_missing_provider_dependency("cosmos", "torch"),
            _translate_missing_provider_dependency("cosmos", "transformers"),
            _translate_missing_provider_dependency("cosmos", "torchvision"),
            _translate_missing_provider_dependency("cosmos", "einops"),
        ):
            import torch
            from transformers import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
                AutoModel,
                AutoProcessor,
            )
            from transformers.utils import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
                is_offline_mode,
            )

            if not torch.cuda.is_available():
                raise EmbeddingConfigurationError("Cosmos requires CUDA in the executing worker")
            # The upstream QFormer constructor loads a BERT config separately
            # without forwarding local_files_only. Require process-wide offline
            # mode for that option instead of allowing an unexpected download.
            if options.get("local_files_only") and not is_offline_mode():
                raise EmbeddingConfigurationError(
                    "Cosmos local_files_only requires HF_HUB_OFFLINE=1 before worker startup; "
                    "cache bert-base-uncased/config.json in each worker's default Hugging Face cache as well"
                )
            self._torch = torch
            self._device = options["device"]
            self._dtype = {"float32": torch.float32, "float16": torch.float16}[options["dtype"]]
            loading: dict[str, Any] = {
                key: options[key] for key in ("revision", "trust_remote_code", "local_files_only") if key in options
            }
            if options.get("cache_folder") is not None:
                loading["cache_dir"] = options["cache_folder"]
            self._processor = AutoProcessor.from_pretrained(model, **loading)
            if (
                self._processor.num_video_frames != 8
                or self._processor.resolution != 224
                or self._processor.max_txt_len != 128
            ):
                raise EmbeddingConfigurationError("Cosmos processor does not match the supported 224p input contract")
            self._model = AutoModel.from_pretrained(model, torch_dtype=self._dtype, **loading)
            if (
                self._model.config.embed_dim != 256
                or self._model.config.num_video_frames != 8
                or self._model.config.resolution != 224
                or self._model.config.transformer_engine
                or self._model.config.use_fp8
            ):
                raise EmbeddingConfigurationError("Cosmos model does not match the supported 224p FP32/FP16 contract")
            self._model.to(self._device, dtype=self._dtype).eval()

    def embed_text(self, texts: list[str]) -> list[np.ndarray]:
        # The official processor truncates implicitly; validate the exact token
        # budget first so a query never silently changes its meaning.
        for text in texts:
            tokens = self._processor.tokenizer.encode(text, add_special_tokens=True, truncation=False)
            if len(tokens) > self._processor.max_txt_len:
                raise EmbeddingConfigurationError("Cosmos text exceeds 128 tokens including special tokens")
        with self._torch.inference_mode():
            inputs = self._processor(text=texts).to(self._device, dtype=self._dtype)
            vectors = self._model.get_text_embeddings(**inputs).text_proj
            return list(vectors.float().cpu().numpy())

    def embed_video(self, clips: list[VideoClip]) -> list[np.ndarray]:
        # Preprocess clips separately so different videos can have different
        # resolutions in one batch, then batch only the fixed-size model input.
        videos = []
        for clip in clips:
            if len(clip.frames) != 8 or any(frame.shape != clip.frames[0].shape for frame in clip.frames):
                raise EmbeddingConfigurationError("Cosmos requires eight frames of the same shape within each clip")
            frames = np.stack(clip.frames).transpose(0, 3, 1, 2)[None].copy()
            videos.append(self._processor(videos=frames)["videos"])
        with self._torch.inference_mode():
            batch = self._torch.cat(videos, dim=0).to(self._device, dtype=self._dtype)
            vectors = self._model.get_video_embeddings(videos=batch).visual_proj
            return list(vectors.float().cpu().numpy())
