# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Local Qwen3-VL text/image/video encoding paired with trained SQR checkpoints.

Descriptors are lightweight. Only the executing GPU worker loads model code
and weights. This provider has one explicit BF16 encoding contract.
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._video_embedding import VideoInputSpec
from vane.ai.protocols import (
    ImageEmbedderDescriptor,
    TextEmbedderDescriptor,
    VideoEmbedderDescriptor,
)
from vane.ai.provider import Provider, _translate_missing_provider_dependency
from vane.ai.typing import UDFOptions

PROVIDER = "qwen3_vl"
INSTRUCTION = "Represent the user's input."
CONTRACT = {
    "device": "cuda",
    "dtype": "bfloat16",
    "attention": "sdpa",
    "instruction": INSTRUCTION,
    "add_special_tokens": True,
    "pooling": "last_nonpadding",
    "dimensions": 2048,
    "max_frames": 64,
    "max_input_bytes": 8 * 1024**2,
    "max_frame_pixels": 65536,
    "max_input_tokens": 8192,
    "max_query_tokens": 128,
    "local_files_only": True,
    "trust_remote_code": False,
}


@dataclass
class _Descriptor:
    model: str
    dimensions: int = 2048

    def __post_init__(self):
        if type(self.dimensions) is not int or self.dimensions != CONTRACT["dimensions"]:
            raise EmbeddingConfigurationError("The Qwen3-VL SQR encoder requires dimensions=2048")
        if (
            not isinstance(self.model, str)
            or not self.model
            or "://" in self.model
            or not Path(self.model).is_absolute()
        ):
            raise EmbeddingConfigurationError("The Qwen3-VL SQR encoder requires an absolute local checkpoint path")

    def get_provider(self):
        return PROVIDER

    def get_model(self):
        return self.model

    def get_options(self):
        return dict(CONTRACT)

    def get_dimensions(self):
        return self.dimensions

    def get_udf_options(self):
        return UDFOptions(num_gpus=1)

    def instantiate(self):
        return Qwen3VLEmbedder(self.model)


class Qwen3VLTextDescriptor(_Descriptor, TextEmbedderDescriptor):
    def supports_chunking(self):
        return False


class Qwen3VLImageDescriptor(_Descriptor, ImageEmbedderDescriptor):
    pass


class Qwen3VLVideoDescriptor(_Descriptor, VideoEmbedderDescriptor):
    def get_input_spec(self):
        return VideoInputSpec(max_frames=CONTRACT["max_frames"], max_input_bytes=CONTRACT["max_input_bytes"])

    def supports_image_queries(self):
        return True


class Qwen3VLProvider(Provider):
    def __init__(self, name=None):
        if name not in (None, PROVIDER):
            raise EmbeddingConfigurationError("The Qwen3-VL encoding identity cannot be renamed")

    @property
    def name(self):
        return PROVIDER

    @staticmethod
    def _descriptor(cls, model, dimensions, options):
        # Vane consumes normalization/execution options before this factory.
        # This provider has one trained encoding contract, with no role overrides.
        if options:
            raise EmbeddingConfigurationError("The Qwen3-VL SQR encoder accepts no encoding overrides")
        return cls(model, dimensions if dimensions is not None else CONTRACT["dimensions"])

    def get_text_embedder(self, model=None, dimensions=None, *, options=None):
        return self._descriptor(Qwen3VLTextDescriptor, model, dimensions, options)

    def get_image_embedder(self, model=None, dimensions=None, *, options=None):
        return self._descriptor(Qwen3VLImageDescriptor, model, dimensions, options)

    def get_video_embedder(self, model=None, dimensions=None, *, options=None):
        return self._descriptor(Qwen3VLVideoDescriptor, model, dimensions, options)


def pool_last(hidden, mask):
    """Keep the last real token, including an appended EOS, in padded batches."""
    import torch

    if not mask.any(dim=1).all().item():
        raise EmbeddingConfigurationError("Qwen embedding requires nonempty token sequences")
    positions = mask.shape[1] - 1 - mask.flip(1).argmax(1)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), positions].float()


def video_payload(clip):
    """Preserve the caller's exact frames and absolute microsecond timestamps."""
    count = len(clip.frames)
    if not 2 <= count <= CONTRACT["max_frames"] or count % 2:
        raise EmbeddingConfigurationError("Qwen3-VL requires an even number of frames from 2 to 64")
    if len(clip.frame_times) != count or len(clip.frame_indices) != count:
        raise EmbeddingConfigurationError("Qwen frame metadata must align with its decoded frames")
    if any(not math.isfinite(t) or not 0 <= t < 2**53 / 1_000_000 for t in clip.frame_times):
        raise EmbeddingConfigurationError("Qwen frame timestamps must fit nonnegative exact microseconds")
    ticks = [round(t * 1_000_000) for t in clip.frame_times]
    if any(a >= b for a, b in zip(ticks, ticks[1:])):
        raise EmbeddingConfigurationError("Qwen video requires distinct increasing microsecond timestamps")
    if any(type(i) is not int or i < 0 for i in clip.frame_indices) or any(
        a >= b for a, b in zip(clip.frame_indices, clip.frame_indices[1:])
    ):
        raise EmbeddingConfigurationError("Qwen video requires increasing frame indices")
    shape = clip.frames[0].shape
    for frame in clip.frames:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or frame.shape != shape:
            raise EmbeddingConfigurationError("Qwen frames require same-shaped UInt8 RGB pixels")
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0 or h % 32 or w % 32 or h * w > CONTRACT["max_frame_pixels"]:
            raise EmbeddingConfigurationError("Qwen frame dimensions must align to 32 pixels within the pixel budget")
    if sum(frame.nbytes for frame in clip.frames) > CONTRACT["max_input_bytes"]:
        raise EmbeddingConfigurationError("Qwen decoded frames exceed the configured byte budget")
    return np.stack(clip.frames), {"fps": 1_000_000, "frames_indices": ticks, "total_num_frames": ticks[-1] + 1}


class Qwen3VLEmbedder:
    def __init__(self, path):
        with (
            _translate_missing_provider_dependency(PROVIDER, "torch"),
            _translate_missing_provider_dependency(PROVIDER, "safetensors"),
            _translate_missing_provider_dependency(PROVIDER, "transformers"),
            _translate_missing_provider_dependency(PROVIDER, "torchvision"),
            _translate_missing_provider_dependency(PROVIDER, "PIL"),
        ):
            import torch
            from safetensors import safe_open
            from transformers import AutoConfig, AutoProcessor, Qwen3VLModel

            self.model = None
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise EmbeddingConfigurationError("The selected Qwen BF16 provider requires a CUDA GPU supporting BF16")
            config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            if config.model_type != "qwen3_vl" or config.text_config.hidden_size != CONTRACT["dimensions"]:
                raise EmbeddingConfigurationError(
                    "The selected checkpoint must be Qwen3-VL with 2048 hidden dimensions"
                )
            rows = []
            for shard in Path(path).glob("*.safetensors"):
                with safe_open(shard, framework="pt") as weights:
                    for key in weights.keys():
                        if key.endswith("language_model.embed_tokens.weight"):
                            shape = weights.get_slice(key).get_shape()
                            if len(shape) != 2 or shape[0] <= 0 or shape[1] != CONTRACT["dimensions"]:
                                raise EmbeddingConfigurationError(
                                    "The trained token embeddings have incompatible dimensions"
                                )
                            rows.append(shape[0])
            if len(rows) != 1:
                raise EmbeddingConfigurationError(
                    "The local checkpoint must contain exactly one trained token embedding matrix"
                )
            config.text_config.vocab_size = rows[0]
            self.processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            if self.processor.__class__.__name__ != "Qwen3VLProcessor":
                raise EmbeddingConfigurationError("The selected checkpoint must provide its Qwen3VLProcessor")
            vocabulary = self.processor.tokenizer.get_vocab()
            if not vocabulary or min(vocabulary.values()) < 0 or max(vocabulary.values()) >= rows[0]:
                raise EmbeddingConfigurationError("The checkpoint tokenizer references untrained token rows")
            processor_vision = self.processor.video_processor
            if (processor_vision.patch_size, processor_vision.merge_size, processor_vision.temporal_patch_size) != (
                16,
                2,
                2,
            ):
                raise EmbeddingConfigurationError(
                    "Qwen3-VL requires the checkpoint's 16x2 spatial and 2-frame temporal patches"
                )
            if (
                config.vision_config.patch_size,
                config.vision_config.spatial_merge_size,
                config.vision_config.temporal_patch_size,
            ) != (16, 2, 2):
                raise EmbeddingConfigurationError("Checkpoint and processor vision patches must match")
            model, loading = Qwen3VLModel.from_pretrained(
                path,
                config=config,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                output_loading_info=True,
            )
            if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")) or any(
                not key.startswith("lm_head.") for key in loading.get("unexpected_keys", [])
            ):
                raise EmbeddingConfigurationError("The local checkpoint lacks the exact trained Qwen model weights")
            self.model = model.to("cuda").eval().requires_grad_(False)

    def _text_inputs(self, texts):
        conversations = [
            [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": text},
            ]
            for text in texts
        ]
        prompts = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        inputs = self.processor.tokenizer(
            prompts,
            add_special_tokens=True,
            truncation=False,
            padding=True,
            return_tensors="pt",
        )
        if inputs["attention_mask"].sum(1).max().item() > CONTRACT["max_query_tokens"]:
            raise EmbeddingConfigurationError("Query exceeds the trained SQR token budget; truncation is forbidden")
        return inputs

    def _infer(self, inputs):
        import torch

        if inputs["attention_mask"].sum(1).max().item() > CONTRACT["max_input_tokens"]:
            raise EmbeddingConfigurationError("Qwen input exceeds the token budget; truncation is forbidden")
        moved = {key: value.to("cuda") for key, value in inputs.items()}
        for key in ("pixel_values", "pixel_values_videos"):
            if key in moved:
                moved[key] = moved[key].to(torch.bfloat16)
        with torch.inference_mode():
            hidden = self.model(**moved, use_cache=False, return_dict=True).last_hidden_state
            if hidden.ndim != 3 or hidden.shape[:2] != moved["attention_mask"].shape or hidden.shape[2] != 2048:
                raise EmbeddingConfigurationError("Qwen returned an incompatible hidden-state shape")
            pooled = pool_last(hidden, moved["attention_mask"])
            if pooled.shape[1] != CONTRACT["dimensions"] or not torch.isfinite(pooled).all().item():
                raise EmbeddingConfigurationError("Qwen returned invalid trained embeddings")
            if not (pooled.norm(dim=-1) > 1e-12).all().item():
                raise EmbeddingConfigurationError("Qwen returned a zero embedding")
            return list(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())

    def embed_text(self, texts):
        return self._infer(self._text_inputs(texts)) if texts else []

    def embed_video(self, clips):
        if not clips:
            return []
        payloads = [video_payload(clip) for clip in clips]
        conversations = [
            [
                {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
                {"role": "user", "content": [{"type": "video"}]},
            ]
            for _ in clips
        ]
        prompts = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=prompts,
            videos=[p[0] for p in payloads],
            video_metadata=[p[1] for p in payloads],
            do_resize=False,
            do_sample_frames=False,
            add_special_tokens=True,
            truncation=False,
            padding=True,
            return_tensors="pt",
        )
        return self._infer(inputs)

    def embed_image(self, images):
        if not images:
            return []
        for frame in images:
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise EmbeddingConfigurationError("Qwen image inputs require UInt8 RGB pixels")
            h, w = frame.shape[:2]
            if h <= 0 or w <= 0 or h % 32 or w % 32 or h * w > CONTRACT["max_frame_pixels"]:
                raise EmbeddingConfigurationError("Qwen image dimensions must align to its 32-pixel patch budget")
        conversations = [
            [
                {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
                {"role": "user", "content": [{"type": "image"}]},
            ]
            for _ in images
        ]
        prompts = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=prompts,
            images=images,
            do_resize=False,
            add_special_tokens=True,
            truncation=False,
            padding=True,
            return_tensors="pt",
        )
        return self._infer(inputs)

    def close(self):
        import torch

        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    async def aclose(self):
        self.close()
