# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL shared text/image/video embeddings using the built-in HF model.

No repository model code is executed. Sampling belongs to the caller; token
overflow is an error instead of silent truncation of text or visual evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._video_embedding import VideoClip, VideoInputSpec
from vane.ai.options import validate_embed_options
from vane.ai.protocols import ImageEmbedderDescriptor, TextEmbedderDescriptor, VideoEmbedderDescriptor
from vane.ai.provider import _translate_missing_provider_dependency
from vane.ai.typing import UDFOptions

QWEN_DIMENSIONS = {"Qwen/Qwen3-VL-Embedding-2B": 2048, "Qwen/Qwen3-VL-Embedding-8B": 4096}
_OPTIONS = frozenset(
    {
        "cache_folder",
        "device",
        "local_files_only",
        "revision",
        "dtype",
        "instruction",
        "max_frames",
        "max_input_bytes",
        "max_length",
        "max_pixels",
    }
)


@dataclass
class _QwenDescriptor:
    model: str
    dimensions: int | None = None
    options: dict[str, Any] = field(default_factory=dict)
    provider_name: str = "transformers"

    def __post_init__(self) -> None:
        if self.model not in QWEN_DIMENSIONS:
            raise EmbeddingConfigurationError("Select a Qwen3-VL-Embedding model, not an Instruct model")
        native = QWEN_DIMENSIONS[self.model]
        if self.dimensions is None:
            self.dimensions = native
        if type(self.dimensions) is not int or not 32 <= self.dimensions <= native:
            raise EmbeddingConfigurationError(f"Qwen embedding dimensions must be between 32 and {native}")
        if self.options.keys() - _OPTIONS:
            raise TypeError("Unsupported Qwen embedding options: " + ", ".join(sorted(self.options.keys() - _OPTIONS)))
        self.options = validate_embed_options("transformers", self.options, relation=False)
        if self.options.get("device") not in {"cpu", "cuda", "cuda:0"}:
            raise EmbeddingConfigurationError("Qwen requires explicit device='cpu' or 'cuda'")
        if self.options.get("dtype") not in {"float32", "float16"}:
            raise EmbeddingConfigurationError("Qwen requires an explicit float32 or float16 dtype")
        for key, default, low, high in (
            ("max_frames", 64, 1, 256),
            ("max_input_bytes", 64 * 1024**2, 1, 64 * 1024**2),
            ("max_length", 8192, 1, 32768),
            ("max_pixels", 512 * 512, 4096, 1024 * 1024),
        ):
            value = self.options.setdefault(key, default)
            if type(value) is not int or not low <= value <= high:
                raise EmbeddingConfigurationError(f"Qwen {key} must be between {low} and {high}")
        instruction = self.options.setdefault("instruction", "Represent the user's input.")
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 4000:
            raise EmbeddingConfigurationError("Qwen instruction must contain 1 to 4000 characters")

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model

    def get_options(self) -> dict[str, Any]:
        return dict(self.options)

    def get_dimensions(self) -> int:
        assert self.dimensions is not None
        return self.dimensions

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(num_gpus=int(self.options["device"].startswith("cuda")))

    def instantiate(self) -> QwenEmbedder:
        return QwenEmbedder(self)


class QwenTextEmbedderDescriptor(_QwenDescriptor, TextEmbedderDescriptor):
    def supports_chunking(self) -> bool:
        return False


class QwenImageEmbedderDescriptor(_QwenDescriptor, ImageEmbedderDescriptor):
    pass


class QwenVideoEmbedderDescriptor(_QwenDescriptor, VideoEmbedderDescriptor):
    def get_input_spec(self) -> VideoInputSpec:
        return VideoInputSpec(max_frames=self.options["max_frames"], max_input_bytes=self.options["max_input_bytes"])

    def supports_image_queries(self) -> bool:
        return True


class QwenEmbedder:
    def __init__(self, descriptor: _QwenDescriptor) -> None:
        with (
            _translate_missing_provider_dependency("qwen", "torch"),
            _translate_missing_provider_dependency("qwen", "transformers"),
        ):
            import torch
            from transformers import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
                AutoModel,
                AutoProcessor,
            )

        self.descriptor = descriptor
        self.options = descriptor.options
        self.torch = torch
        self.device = self.options["device"]
        loading: dict[str, Any] = {
            key: self.options[key] for key in ("revision", "local_files_only") if key in self.options
        }
        if "cache_folder" in self.options:
            loading["cache_dir"] = self.options["cache_folder"]
        loading["trust_remote_code"] = False
        self.processor = AutoProcessor.from_pretrained(descriptor.model, padding_side="right", **loading)
        self.model, info = AutoModel.from_pretrained(
            descriptor.model, output_loading_info=True, torch_dtype=getattr(torch, self.options["dtype"]), **loading
        )
        if info["missing_keys"] or info["mismatched_keys"] or info.get("error_msgs"):
            raise EmbeddingConfigurationError("Qwen embedding checkpoint does not match the encoder")
        if self.model.config.text_config.hidden_size != QWEN_DIMENSIONS[descriptor.model]:
            raise EmbeddingConfigurationError("Qwen model has an unexpected embedding dimension")
        self.model.to(self.device).eval()

    def _encode(self, kind: str, value: Any) -> Any:
        content = {"type": kind, kind: value} if kind != "video" else {"type": "video"}
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.options["instruction"]}]},
            {"role": "user", "content": [content]},
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        kwargs: dict[str, Any] = {}
        if kind == "image":
            kwargs = {
                "images": [value],
                "images_kwargs": {"min_pixels": 4096, "max_pixels": self.options["max_pixels"]},
            }
        elif kind == "video":
            from transformers.video_utils import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
                VideoMetadata,
            )

            if any(frame.shape != value.frames[0].shape for frame in value.frames):
                raise EmbeddingConfigurationError("Qwen video frames within a clip must have the same shape")
            if any(time > (2**53 - 1) / 1_000_000 for time in value.frame_times):
                raise EmbeddingConfigurationError("Qwen video timestamp exceeds the microsecond clock")
            ticks = [round(time * 1_000_000) for time in value.frame_times]
            metadata = VideoMetadata(
                total_num_frames=max(len(ticks), ticks[-1] + 1), fps=1_000_000, frames_indices=ticks
            )
            kwargs = {
                "videos": [np.stack(value.frames)],
                "videos_kwargs": {
                    "video_metadata": [metadata],
                    "do_sample_frames": False,
                    "size": {"shortest_edge": 4096, "longest_edge": self.options["max_pixels"]},
                },
            }
        inputs = self.processor(text=[text], padding=True, truncation=False, return_tensors="pt", **kwargs)
        if inputs["input_ids"].shape[1] > self.options["max_length"]:
            raise EmbeddingConfigurationError("Qwen input exceeds max_length; reduce upstream sampling or text")
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            hidden = self.model(**inputs, use_cache=False).last_hidden_state
            mask = inputs["attention_mask"]
            index = mask.shape[1] - 1 - mask.flip(dims=[1]).argmax(dim=1)
            vector = hidden[self.torch.arange(hidden.shape[0], device=hidden.device), index]
            return vector[0, : self.descriptor.dimensions].float().cpu().numpy()

    def embed_text(self, texts: list[str]) -> list[Any]:
        return [self._encode("text", text) for text in texts]

    def embed_image(self, images: list[Any]) -> list[Any]:
        return [self._encode("image", image) for image in images]

    def embed_video(self, clips: list[VideoClip]) -> list[Any]:
        return [self._encode("video", clip) for clip in clips]
