# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Worker-owned Python inference engines. No serving process or HTTP client.

The regular AI actor owns the engine and its asynchronous runtime. Descriptors
carry configuration only; importing this module does not import model SDKs.
Text-only native Prompt plans continue to use the physical inference operator.
"""

from __future__ import annotations

import copy
import inspect
import io
import json
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._media import PromptMedia, normalize_media_content_type
from vane.ai._video_embedding import VideoClip, VideoInputSpec
from vane.ai.options import (
    _NATIVE_EMBED_OPTIONS,
    _NATIVE_MEDIA_PROMPT_OPTIONS,
    _reject_sensitive_embed_options,
    normalize_prompt_options,
)
from vane.ai.protocols import (
    ImageEmbedderDescriptor,
    PrompterDescriptor,
    TextEmbedderDescriptor,
    VideoEmbedderDescriptor,
)
from vane.ai.provider import Provider, _translate_missing_provider_dependency
from vane.ai.typing import UDFOptions

IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp", "image/gif"})
VIDEO_TYPES = frozenset({"video/mp4", "video/webm", "video/quicktime"})


def _positive(options: dict[str, Any], name: str, default: int, maximum: int) -> int:
    value = options.setdefault(name, default)
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"Native inference {name} must be between 1 and {maximum}")
    return value


def _engine_options(family: str, options: dict[str, Any], *, embedding: bool) -> None:
    from vane.ai.providers.vllm import _validate_vllm_json

    for name in ("processor_kwargs", "pooling_args", "generate_args", "chat_template_kwargs"):
        if name in options:
            if not isinstance(options[name], Mapping):
                raise TypeError(f"{name} must be a mapping")
            _validate_vllm_json(options[name], name)
            options[name] = copy.deepcopy(dict(options[name]))
    template = options.get("chat_template")
    if template is not None and (not isinstance(template, str) or not template.strip()):
        raise ValueError("chat_template must be a nonempty string")
    if {"tokenize", "add_generation_prompt", "chat_template", "conversation"} & options.get(
        "chat_template_kwargs", {}
    ).keys():
        raise ValueError("chat_template_kwargs cannot override template rendering arguments")
    args = options.setdefault("engine_args", {})
    if not isinstance(args, Mapping):
        raise TypeError("engine_args must be a mapping")
    _validate_vllm_json(args, "engine_args")
    args = options["engine_args"] = copy.deepcopy(dict(args))
    # Model/task ownership cannot be changed inside the low-level options.
    reserved = {"model", "model_path", "runner", "is_embedding", "skip_tokenizer_init"}
    if reserved & args.keys():
        raise ValueError("Native inference owns engine_args: " + ", ".join(sorted(reserved & args.keys())))
    if args.get("trust_remote_code", False) is not False:
        raise ValueError("Native multimodal inference requires trust_remote_code=False")
    args["trust_remote_code"] = False
    # An engine must stay inside its Vane worker's resource reservation.
    if family == "vllm":
        parallel = ("tensor_parallel_size", "pipeline_parallel_size")
        if args.get("distributed_executor_backend", "mp") != "mp" or args.get("data_parallel_size", 1) != 1:
            raise ValueError("Native multimodal vLLM requires worker-local mp execution and data_parallel_size=1")
        args["distributed_executor_backend"] = "mp"
        args["runner"] = "pooling" if embedding else "generate"
    else:
        parallel = ("tp_size", "pp_size")
        if args.get("nnodes", 1) != 1 or args.get("dp_size", 1) != 1:
            raise ValueError("Native multimodal SGLang requires nnodes=1 and dp_size=1")
        args["is_embedding"] = embedding
    required = 1
    for name in parallel:
        value = args.get(name, 1)
        if type(value) is not int or value < 1:
            raise ValueError(f"engine_args.{name} must be a positive integer")
        required *= value
    gpus = options.setdefault("gpus_per_actor", required)
    if type(gpus) is not int or gpus < 0 or (gpus and gpus < required):
        raise ValueError("gpus_per_actor must be a nonnegative integer covering engine parallelism")
    _positive(options, "max_input_bytes", 64 * 1024**2, 64 * 1024**2)
    if embedding or family == "vllm":
        _positive(options, "max_frames", 64, 256)
    elif "max_frames" in options:
        raise ValueError("Configure native SGLang video sampling through engine_args.mm_process_config")


@dataclass
class _NativeEmbeddingDescriptor:
    model: str
    dimensions: int | None
    options: dict[str, Any]
    provider_name: str
    family: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise EmbeddingConfigurationError("Native embedding requires an explicit model name")
        _reject_sensitive_embed_options(self.options)
        self.options = copy.deepcopy(self.options)
        if self.options.keys() - _NATIVE_EMBED_OPTIONS:
            raise TypeError(
                "Unsupported native embedding options: "
                + ", ".join(sorted(self.options.keys() - _NATIVE_EMBED_OPTIONS))
            )
        if type(self.dimensions) is not int or self.dimensions <= 0:
            raise EmbeddingConfigurationError("Native embedding requires explicit positive dimensions")
        override = self.options.setdefault("supports_overriding_dimensions", False)
        if type(override) is not bool:
            raise ValueError("supports_overriding_dimensions must be a bool")
        paired = self.options.setdefault("paired_image_queries", False)
        if type(paired) is not bool:
            raise ValueError("paired_image_queries must be a bool")
        instruction = self.options.setdefault("instruction", "Represent the user's input.")
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 4000:
            raise ValueError("embedding instruction must contain 1 to 4000 characters")
        _engine_options(self.family, self.options, embedding=True)
        pooling = self.options.get("pooling_args", {})
        if {"dimensions", "task"} & pooling.keys():
            raise ValueError("pooling_args cannot override dimensions or task")
        if self.family == "sglang" and pooling:
            raise ValueError("SGLang exposes no pooling_args; configure the embedding model through engine_args")
        processor = self.options.get("processor_kwargs", {})
        if {"text", "images", "videos", "padding", "truncation", "return_tensors"} & processor.keys():
            raise ValueError("processor_kwargs cannot override inputs, padding or truncation")
        text_options = processor.get("text_kwargs", {})
        if not isinstance(text_options, Mapping) or text_options.get("truncation", False) is not False:
            raise ValueError("Embedding processor cannot truncate inputs")
        video = processor.get("videos_kwargs", {})
        if not isinstance(video, Mapping):
            raise TypeError("processor_kwargs.videos_kwargs must be a mapping")
        for values in (processor, video):
            if {"video_metadata", "num_frames", "fps"} & values.keys() or values.get(
                "do_sample_frames", False
            ) is not False:
                raise ValueError("Embedding frame selection and timestamps belong to the input clip")

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model

    def get_options(self) -> dict[str, Any]:
        return copy.deepcopy(self.options)

    def get_dimensions(self) -> int:
        assert self.dimensions is not None
        return self.dimensions

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(num_gpus=self.options["gpus_per_actor"])

    def is_async(self) -> bool:
        return True

    def instantiate(self) -> NativeEmbedder:
        return NativeEmbedder(self)


class NativeTextEmbedderDescriptor(_NativeEmbeddingDescriptor, TextEmbedderDescriptor):
    def supports_chunking(self) -> bool:
        return False


class NativeImageEmbedderDescriptor(_NativeEmbeddingDescriptor, ImageEmbedderDescriptor):
    pass


class NativeVideoEmbedderDescriptor(_NativeEmbeddingDescriptor, VideoEmbedderDescriptor):
    def get_input_spec(self) -> VideoInputSpec:
        return VideoInputSpec(max_frames=self.options["max_frames"], max_input_bytes=self.options["max_input_bytes"])

    def supports_image_queries(self) -> bool:
        return bool(self.options["paired_image_queries"])


@dataclass
class NativeMediaPrompterDescriptor(PrompterDescriptor):
    provider_name: str
    family: str
    model: str
    system_message: str | None = None
    return_format: dict[str, Any] | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Native multimodal prompt requires an explicit model name")
        self.options = copy.deepcopy(self.options)
        normalize_prompt_options(self.family, self.options, relation=False)
        if self.options.keys() - _NATIVE_MEDIA_PROMPT_OPTIONS:
            raise TypeError("Unsupported native multimodal Prompt options")
        media = self.options.get("media_mime_types")
        if not isinstance(media, list) or not media or any(not isinstance(value, str) for value in media):
            raise ValueError("media_mime_types must be a nonempty MIME type list")
        normalized = [normalize_media_content_type(value) for value in media]
        if len(set(normalized)) != len(normalized) or set(normalized) - IMAGE_TYPES - VIDEO_TYPES:
            raise ValueError("Native prompt supports distinct declared image and video MIME types")
        self.options["media_mime_types"] = normalized
        _engine_options(self.family, self.options, embedding=False)
        generate = self.options.get("generate_args", {})
        if {
            "prompt",
            "input_ids",
            "image_data",
            "video_data",
            "audio_data",
            "request_id",
            "rid",
            "stream",
        } & generate.keys():
            raise ValueError("generate_args cannot override inputs, request identity or streaming")
        sampling = generate.get("sampling_params", {})
        if not isinstance(sampling, Mapping):
            raise TypeError("generate_args.sampling_params must be a mapping")
        if {"json_schema", "structured_outputs", "guided_decoding"} & sampling.keys():
            raise ValueError("Use return_format for structured output")
        if sampling.get("n", 1) != 1 or sampling.get("best_of", 1) != 1:
            raise ValueError("Native Prompt requires one output per input")
        top = {
            "temperature": "temperature",
            "top_p": "top_p",
            "stop_sequences": "stop",
            "max_tokens": "max_tokens" if self.family == "vllm" else "max_new_tokens",
        }
        for key, native_key in top.items():
            if key in self.options and native_key in sampling:
                raise ValueError(f"Configure {native_key} once, at the top level or in sampling_params")
        token_key = "max_tokens" if self.family == "vllm" else "max_new_tokens"
        if token_key in sampling:
            _positive(dict(sampling), token_key, 1024, 131072)
        else:
            _positive(self.options, "max_tokens", 1024, 131072)
        for name, maximum in (("temperature", None), ("top_p", 1)):
            value = self.options.get(name)
            if value is not None and (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or (maximum and value > maximum)
            ):
                raise ValueError(f"Invalid native inference {name}")
        stop = self.options.get("stop_sequences")
        if stop is not None and (
            not isinstance(stop, list) or any(not isinstance(value, str) or not value for value in stop)
        ):
            raise ValueError("stop_sequences must contain nonempty strings")
        self.return_format = copy.deepcopy(self.return_format)

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model

    def get_options(self) -> dict[str, Any]:
        return copy.deepcopy(self.options)

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(num_gpus=self.options["gpus_per_actor"])

    def supports_image_inputs(self) -> bool:
        # The existing Prompt boundary uses this flag for FILE media as well;
        # the closed MIME allowlist below validates the selected modalities.
        return True

    def supported_media_mime_types(self) -> frozenset[str]:
        return frozenset(self.options["media_mime_types"])

    def instantiate(self) -> NativePrompter:
        return NativePrompter(self)


def _clip_data(clip: VideoClip) -> tuple[np.ndarray, dict[str, Any]]:
    if not clip.frames or len(clip.frames) != len(clip.frame_times):
        raise EmbeddingConfigurationError("Video frames and timestamps must be nonempty and aligned")
    if any(frame.shape != clip.frames[0].shape for frame in clip.frames):
        raise EmbeddingConfigurationError("Video frames within a clip must have the same shape")
    if any(not math.isfinite(value) or not 0 <= value <= (2**53 - 1) / 1_000_000 for value in clip.frame_times):
        raise EmbeddingConfigurationError("Video timestamp exceeds the microsecond clock")
    ticks = [round(time * 1_000_000) for time in clip.frame_times]
    return np.stack(clip.frames), {
        "fps": 1_000_000,
        "frames_indices": ticks,
        "total_num_frames": max(len(ticks), ticks[-1] + 1),
        "do_sample_frames": False,
    }


class _NativeRuntime:
    def __init__(self, descriptor: Any):
        self.descriptor = descriptor
        self.options = descriptor.options
        self.engine: Any = None
        self.closed = False
        with _translate_missing_provider_dependency(descriptor.family, descriptor.family):
            from transformers import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
                AutoProcessor,
                PreTrainedTokenizerBase,
            )

        args = self.options["engine_args"]
        loading = {key: args[key] for key in ("revision",) if key in args}
        self.processor = AutoProcessor.from_pretrained(descriptor.model, trust_remote_code=False, **loading)
        tokenizer_model = args.get("tokenizer" if descriptor.family == "vllm" else "tokenizer_path")
        if tokenizer_model or args.get("tokenizer_revision"):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_model or descriptor.model,
                revision=args.get("tokenizer_revision", args.get("revision")),
                trust_remote_code=False,
            )
            if hasattr(self.processor, "tokenizer"):
                self.processor.tokenizer = tokenizer
            else:
                self.processor = tokenizer
        self._is_tokenizer = isinstance(self.processor, PreTrainedTokenizerBase)

    def _ensure_engine(self) -> Any:
        if self.closed:
            raise RuntimeError("Native inference engine is closed")
        if self.engine is None:
            family = self.descriptor.family
            with _translate_missing_provider_dependency(family, family):
                if family == "vllm":
                    from vllm import AsyncEngineArgs, AsyncLLMEngine

                    self.engine = AsyncLLMEngine.from_engine_args(
                        AsyncEngineArgs(model=self.descriptor.model, **self.options["engine_args"])
                    )
                else:
                    from sglang import Engine  # type: ignore[import-not-found, import-untyped, unused-ignore]

                    self.engine = Engine(model_path=self.descriptor.model, **self.options["engine_args"])
        return self.engine

    def _render(self, content: list[dict[str, Any]], system: str | None) -> str:
        user_content: Any = content
        system_content: Any = [{"type": "text", "text": system}]
        if self._is_tokenizer:
            if any(part["type"] != "text" for part in content):
                raise EmbeddingConfigurationError("The selected tokenizer cannot format image or video inputs")
            user_content = "\n".join(part["text"] for part in content)
            system_content = system
        messages = []
        if system:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})
        kwargs = dict(self.options.get("chat_template_kwargs", {}))
        if "chat_template" in self.options:
            kwargs["chat_template"] = self.options["chat_template"]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        engine, self.engine = self.engine, None
        try:
            if engine is not None:
                result = engine.shutdown()
                if inspect.isawaitable(result):
                    await result
        finally:
            self.processor = None

    async def _vllm_result(self, stream: Any) -> Any:
        final = None
        try:
            async for output in stream:
                final = output
        finally:
            await stream.aclose()
        if final is None or not final.finished:
            raise ValueError("Native vLLM returned no completed result")
        return final


class NativeEmbedder(_NativeRuntime):
    async def _embed(self, kind: str, value: Any) -> Any:
        if kind == "video":
            self.descriptor.get_input_spec().validate_count(len(value.frames))
            size = sum(frame.nbytes for frame in value.frames)
        else:
            size = len(value.encode("utf-8")) if kind == "text" else value.nbytes
        if size > self.options["max_input_bytes"]:
            raise EmbeddingConfigurationError("Embedding input exceeds max_input_bytes")
        content = {"type": kind, kind: value} if kind != "video" else {"type": "video"}
        text = self._render([content], self.options["instruction"])
        media: dict[str, Any] = {}
        if kind == "image":
            media["image"] = [value]
        elif kind == "video":
            media["video"] = [_clip_data(value)]
        dimensions = self.descriptor.dimensions if self.options["supports_overriding_dimensions"] else None
        if self.descriptor.family == "vllm":
            from vllm import PoolingParams

            params = PoolingParams(task="embed", dimensions=dimensions, **self.options.get("pooling_args", {}))
            request = {
                "prompt": text,
                "multi_modal_data": media,
                "mm_processor_kwargs": copy.deepcopy(self.options.get("processor_kwargs", {})),
            }
            result = await self._vllm_result(self._ensure_engine().encode(request, params, uuid.uuid4().hex))
            return result.outputs.data.tolist()
        # SGLang's raw decoded-video input does not carry timestamps. Its
        # native processor_output input does: precompute tokens and pixels
        # together, retaining the caller's frame grid without resampling.
        kwargs: dict[str, Any] = {}
        processing = copy.deepcopy(self.options.get("processor_kwargs", {}))
        payload: dict[str, Any] = {}
        if kind == "video":
            from transformers.video_utils import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
                VideoMetadata,
            )

            frames, metadata = media["video"][0]
            metadata = {key: item for key, item in metadata.items() if key != "do_sample_frames"}
            processing.setdefault("videos_kwargs", {}).update(
                video_metadata=[VideoMetadata(**metadata)], do_sample_frames=False
            )
            payload["videos"] = [frames]
        elif kind == "image":
            payload["images"] = media["image"]
        if payload or processing:
            inputs = self.processor(
                text=[text],
                padding=False,
                truncation=False,
                return_tensors="pt",
                **payload,
                **processing,
            )
            ids = inputs["input_ids"][0].tolist()
            limit = self.options["engine_args"].get("context_length")
            if limit is not None and len(ids) > limit:
                raise EmbeddingConfigurationError("Embedding input exceeds context_length")
            tokenizer = self.processor if self._is_tokenizer else self.processor.tokenizer
            text = tokenizer.decode(ids, skip_special_tokens=False)
            if tokenizer.encode(text, add_special_tokens=False) != ids:
                raise EmbeddingConfigurationError("SGLang processor tokens do not round-trip")
            if kind != "text":
                kwargs[kind + "_data"] = [{"format": "processor_output", **dict(inputs)}]
        result = await self._ensure_engine().async_encode(prompt=text, dimensions=dimensions, **kwargs)
        if not isinstance(result, dict) or "embedding" not in result:
            raise ValueError("Native SGLang returned no embedding")
        return result["embedding"]

    async def embed_text(self, values: list[str]) -> list[Any]:
        return [await self._embed("text", value) for value in values]

    async def embed_image(self, values: list[Any]) -> list[Any]:
        return [await self._embed("image", value) for value in values]

    async def embed_video(self, values: list[VideoClip]) -> list[Any]:
        return [await self._embed("video", value) for value in values]


class NativePrompter(_NativeRuntime):
    async def prompt(self, messages: tuple[Any, ...]) -> str:
        from PIL import Image  # type: ignore[import-not-found, import-untyped, unused-ignore]

        from vane.ai.providers.openai import _IMAGE_MIME_POLICY

        content, images, videos = [], [], []
        size = 0
        for value in messages:
            if isinstance(value, str):
                size += len(value.encode("utf-8"))
                content.append({"type": "text", "text": value})
            else:
                mime = (
                    value.content_type
                    if isinstance(value, PromptMedia)
                    else _IMAGE_MIME_POLICY.require_supported(value)
                )
                data = bytes(value)
                size += len(data)
                if mime not in self.descriptor.supported_media_mime_types():
                    raise ValueError("Media MIME type was not declared by the native model")
                if size > self.options["max_input_bytes"]:
                    raise ValueError("Native prompt exceeds max_input_bytes")
                if mime.startswith("image/"):
                    with Image.open(io.BytesIO(data)) as image:
                        images.append(image.convert("RGB"))
                    content.append({"type": "image"})
                else:
                    if self.descriptor.family == "vllm":
                        from vllm.multimodal.media import ImageMediaIO, VideoMediaIO

                        videos.append(
                            VideoMediaIO(ImageMediaIO(), num_frames=self.options["max_frames"]).load_bytes(data).media
                        )
                    else:
                        videos.append(data)
                    content.append({"type": "video"})
            if size > self.options["max_input_bytes"]:
                raise ValueError("Native prompt exceeds max_input_bytes")
        text = self._render(content, self.descriptor.system_message)
        generate = copy.deepcopy(self.options.get("generate_args", {}))
        params = dict(generate.pop("sampling_params", {}))
        params.update({key: self.options[key] for key in ("temperature", "top_p") if self.options.get(key) is not None})
        if self.options.get("stop_sequences") is not None:
            params["stop"] = self.options["stop_sequences"]
        if self.descriptor.family == "vllm":
            from vllm import SamplingParams

            params.setdefault("max_tokens", self.options.get("max_tokens", 1024))
            if self.descriptor.return_format is not None:
                from vllm.sampling_params import StructuredOutputsParams

                params["structured_outputs"] = StructuredOutputsParams(json=self.descriptor.return_format)
            sampling = SamplingParams(**params)
            media = {}
            if images:
                media["image"] = images
            if videos:
                media["video"] = videos
            result = await self._vllm_result(
                self._ensure_engine().generate(
                    {"prompt": text, "multi_modal_data": media}, sampling, uuid.uuid4().hex, **generate
                )
            )
            if len(result.outputs) != 1 or result.outputs[0].finish_reason != "stop":
                raise ValueError("Native vLLM generation did not stop normally")
            return result.outputs[0].text
        params.setdefault("max_new_tokens", self.options.get("max_tokens", 1024))
        if self.descriptor.return_format is not None:
            params["json_schema"] = json.dumps(self.descriptor.return_format)
        result = await self._ensure_engine().async_generate(
            prompt=text,
            image_data=images or None,
            video_data=videos or None,
            sampling_params=params,
            **generate,
        )
        finish = result.get("meta_info", {}).get("finish_reason", {}) if isinstance(result, dict) else {}
        if finish.get("type") != "stop" or not isinstance(result.get("text"), str):
            raise ValueError("Native SGLang generation did not stop normally")
        return result["text"]


class NativeEmbeddingProviderMixin(Provider):
    _native_family: str

    def get_text_embedder(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> NativeTextEmbedderDescriptor:
        return NativeTextEmbedderDescriptor(
            model or "", dimensions, dict(options or {}), self.name, self._native_family
        )

    def get_image_embedder(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> NativeImageEmbedderDescriptor:
        return NativeImageEmbedderDescriptor(
            model or "", dimensions, dict(options or {}), self.name, self._native_family
        )

    def get_video_embedder(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> NativeVideoEmbedderDescriptor:
        return NativeVideoEmbedderDescriptor(
            model or "", dimensions, dict(options or {}), self.name, self._native_family
        )

    def _media_prompter(
        self,
        model: str | None,
        system_message: str | None,
        return_format: dict[str, Any] | None,
        return_raw_response: bool,
        options: Mapping[str, Any] | None,
    ) -> NativeMediaPrompterDescriptor:
        if return_raw_response:
            raise ValueError("Native multimodal inference does not support return_raw_response")
        return NativeMediaPrompterDescriptor(
            self.name, self._native_family, model or "", system_message, return_format, dict(options or {})
        )
