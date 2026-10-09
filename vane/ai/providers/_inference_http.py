# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit HTTP inference for vLLM and SGLang, using their declared wire formats."""

from __future__ import annotations

import base64
import io
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from vane.ai._client_config import copy_client_options
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._embedding_requests import _EmbeddingBatchError, _is_request_wide_error
from vane.ai._media import PromptMedia, normalize_media_content_type
from vane.ai._video_embedding import VideoClip, VideoInputSpec
from vane.ai.options import (
    _HTTP_EMBED_OPTIONS,
    _HTTP_PROMPT_OPTIONS,
    _validate_base_url_option,
    normalize_prompt_options,
    validate_embed_options,
)
from vane.ai.protocols import (
    ImageEmbedderDescriptor,
    PrompterDescriptor,
    TextEmbedderDescriptor,
    VideoEmbedderDescriptor,
)
from vane.ai.provider import Provider, ProviderCapabilityError, _translate_missing_provider_dependency
from vane.ai.providers._openai_client_config import create_openai_client
from vane.ai.providers.openai import (
    _IMAGE_MIME_POLICY,
    OpenAIPrompter,
    OpenAIPrompterDescriptor,
    _is_embedding_capability_error,
)
from vane.ai.typing import UDFOptions

IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp", "image/gif"})
VIDEO_TYPES = frozenset({"video/mp4", "video/webm", "video/quicktime"})
QWEN_DIMENSIONS = {"Qwen/Qwen3-VL-Embedding-2B": 2048, "Qwen/Qwen3-VL-Embedding-8B": 4096}


def client_snapshot(provider: str, options: dict[str, Any]) -> dict[str, Any]:
    # Never import OPENAI_* settings into another serving deployment. Snapshot
    # the intended endpoint and credential before serializing a worker plan.
    return copy_client_options(
        {
            "api_key": os.getenv(provider.upper() + "_API_KEY") or "EMPTY",
            "base_url": options["base_url"],
            "organization": None,
            "project": None,
        }
    )


def validate_http(options: dict[str, Any], allowed: frozenset[str]) -> dict[str, Any]:
    result = dict(options)
    if result.get("transport") != "http":
        raise ValueError("remote inference requires transport='http'")
    if result.keys() - allowed:
        raise TypeError("Unsupported HTTP inference options: " + ", ".join(sorted(result.keys() - allowed)))
    if not result.get("base_url"):
        raise ValueError("HTTP inference requires an explicit base_url")
    _validate_base_url_option(result, api="Prompt")
    timeout = result.setdefault("timeout", 120.0)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("HTTP inference timeout must be finite and positive")
    return result


def image_url(image: Any) -> str:
    from PIL import Image  # type: ignore[import-not-found, import-untyped, unused-ignore]

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def video_payload(clip: VideoClip) -> tuple[str, dict[str, Any]]:
    from PIL import Image  # type: ignore[import-not-found, import-untyped, unused-ignore]

    frames = []
    for frame in clip.frames:
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="JPEG", quality=95, subsampling=0)
        frames.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    # vLLM's JPEG sequence accepts a clock and selected indices. Express the
    # original presentation timestamps on a microsecond grid, including VFR.
    if not clip.frames or len(clip.frames) != len(clip.frame_times):
        raise EmbeddingConfigurationError("video frames and timestamps must be nonempty and aligned")
    if any(not math.isfinite(value) or not 0 <= value <= (2**53 - 1) / 1_000_000 for value in clip.frame_times):
        raise EmbeddingConfigurationError("video timestamp exceeds the HTTP microsecond clock")
    ticks = [round(value * 1_000_000) for value in clip.frame_times]
    metadata = {
        "fps": 1_000_000,
        "frames_indices": ticks,
        "total_num_frames": max(len(ticks), ticks[-1] + 1),
        "num_frames": -1,
        "do_sample_frames": False,
    }
    return "data:video/jpeg;base64," + ",".join(frames), metadata


@dataclass
class _HTTPEmbeddingDescriptor:
    model: str
    dimensions: int | None
    options: dict[str, Any]
    provider_name: str
    api_family: str
    client_options: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise EmbeddingConfigurationError("HTTP embedding requires a model name")
        self.options = validate_http(self.options, _HTTP_EMBED_OPTIONS)
        validate_embed_options(self.api_family, self.options, relation=False)
        known = QWEN_DIMENSIONS.get(self.model)
        if self.dimensions is None:
            self.dimensions = known
        if type(self.dimensions) is not int or self.dimensions <= 0 or (known and self.dimensions > known):
            raise EmbeddingConfigurationError("HTTP embedding requires valid explicit dimensions or a known model")
        for name, default, maximum in (("max_frames", 64, 256), ("max_input_bytes", 64 * 1024**2, 64 * 1024**2)):
            value = self.options.setdefault(name, default)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"HTTP embedding {name} must be between 1 and {maximum}")
        paired = self.options.setdefault("paired_image_queries", known is not None)
        if type(paired) is not bool:
            raise ValueError("paired_image_queries must be a bool")
        instruction = self.options.setdefault("instruction", "Represent the user's input.")
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 4000:
            raise ValueError("embedding instruction must contain 1 to 4000 characters")
        if self.api_family == "sglang" and instruction != "Represent the user's input.":
            raise ValueError("SGLang embedding instruction is configured by the serving chat template")
        self.client_options = copy_client_options(self.client_options or client_snapshot(self.api_family, self.options))

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
        return UDFOptions(num_gpus=0)

    def is_async(self) -> bool:
        return True

    def instantiate(self) -> HTTPEmbedder:
        return HTTPEmbedder(self)


class HTTPTextEmbedderDescriptor(_HTTPEmbeddingDescriptor, TextEmbedderDescriptor):
    def supports_chunking(self) -> bool:
        return False


class HTTPImageEmbedderDescriptor(_HTTPEmbeddingDescriptor, ImageEmbedderDescriptor):
    pass


class HTTPVideoEmbedderDescriptor(_HTTPEmbeddingDescriptor, VideoEmbedderDescriptor):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.api_family != "vllm":
            raise EmbeddingConfigurationError("SGLang's embedding API cannot preserve decoded video timestamps")

    def get_input_spec(self) -> VideoInputSpec:
        return VideoInputSpec(max_frames=self.options["max_frames"], max_input_bytes=self.options["max_input_bytes"])

    def supports_image_queries(self) -> bool:
        return cast(bool, self.options["paired_image_queries"])


class HTTPEmbedder:
    def __init__(self, descriptor: _HTTPEmbeddingDescriptor) -> None:
        with _translate_missing_provider_dependency("openai", "openai"):
            from openai import AsyncOpenAI

        self.descriptor = descriptor
        self.client = create_openai_client(AsyncOpenAI, descriptor.client_options, descriptor.options)

    async def aclose(self) -> None:
        await self.client.close()

    async def _embed(self, kind: str, values: list[Any]) -> list[Any]:

        results = []
        for value in values:
            item = value if kind == "text" else image_url(value) if kind == "image" else video_payload(value)
            body: dict[str, Any] = {
                "model": self.descriptor.model,
                "encoding_format": "float",
                "dimensions": self.descriptor.dimensions,
            }
            if self.descriptor.api_family == "sglang":
                body["input"] = [{kind: item}]
            else:
                if kind == "text":
                    part = {"type": "text", "text": item}
                elif kind == "image":
                    part = {"type": "image_url", "image_url": {"url": item}}
                else:
                    url, metadata = cast(tuple[str, dict[str, Any]], item)
                    part = {"type": "video_url", "video_url": {"url": url}}
                    body["media_io_kwargs"] = {"video": metadata}
                body["messages"] = [
                    {"role": "system", "content": self.descriptor.options["instruction"]},
                    {"role": "user", "content": [part]},
                ]
            results.append(await self._request(body))
        return results

    async def _request(self, body: dict[str, Any]) -> Any:
        from json import JSONDecodeError

        from openai import OpenAIError

        error: Exception | None = None
        try:
            response = await self.client.post("/embeddings", cast_to=dict[str, Any], body=body)
            data = response.get("data") if isinstance(response, dict) else None
            if (
                not isinstance(data, list)
                or len(data) != 1
                or not isinstance(data[0], dict)
                or type(data[0].get("index")) is not int
                or data[0]["index"] != 0
            ):
                raise _EmbeddingBatchError("embedding response must contain exactly one vector at index zero")
            if "embedding" not in data[0]:
                raise _EmbeddingBatchError("embedding response is missing its vector")
            return data[0]["embedding"]
        except (JSONDecodeError, UnicodeDecodeError):
            error = _EmbeddingBatchError("HTTP embedding response could not be decoded")
        except OpenAIError as exc:
            if _is_request_wide_error(exc):
                raise
            if _is_embedding_capability_error(exc):
                error = ProviderCapabilityError(
                    self.descriptor.provider_name, self.descriptor.model, "multimodal embedding", original_error=exc
                )
            else:
                from vane.ai.functions import _retry_after_error

                error = _retry_after_error(exc)
                if error is None:
                    raise
        assert error is not None
        raise error from None

    async def embed_text(self, values: list[Any]) -> list[Any]:
        return await self._embed("text", values)

    async def embed_image(self, values: list[Any]) -> list[Any]:
        return await self._embed("image", values)

    async def embed_video(self, values: list[Any]) -> list[Any]:
        return await self._embed("video", values)


class HTTPPrompterDescriptor(PrompterDescriptor):
    def __init__(
        self,
        provider: str,
        family: str,
        model: str | None,
        system_message: str | None,
        return_format: dict[str, Any] | None,
        return_raw_response: bool,
        options: Mapping[str, Any],
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("HTTP prompt requires an explicit model name")
        prepared = validate_http(dict(options), _HTTP_PROMPT_OPTIONS)
        normalize_prompt_options(family, prepared, relation=False)
        media = prepared.get("media_mime_types", [])
        if not isinstance(media, list) or any(not isinstance(value, str) for value in media):
            raise ValueError("media_mime_types must be a MIME type list")
        self.http_options = dict(prepared)
        self.media_types = frozenset(normalize_media_content_type(value) for value in media)
        if self.media_types - IMAGE_TYPES - VIDEO_TYPES:
            raise ValueError("HTTP prompt currently supports declared image and video MIME types")
        if len(media) != len(self.media_types):
            raise ValueError("media_mime_types must be distinct")
        mapped = {
            k: v for k, v in prepared.items() if k in {"base_url", "timeout", "temperature", "top_p", "stop_sequences"}
        }
        if "max_tokens" in prepared:
            mapped["max_output_tokens"] = prepared["max_tokens"]
        mapped["use_chat_completions"] = True
        self.delegate = OpenAIPrompterDescriptor(
            provider_name=provider,
            model_name=model,
            system_message=system_message,
            return_format=return_format,
            return_raw_response=return_raw_response,
            options=mapped,
            client_options=client_snapshot(family, prepared),
        )

    def get_provider(self) -> str:
        return self.delegate.get_provider()

    def get_model(self) -> str:
        return self.delegate.get_model()

    def get_udf_options(self) -> UDFOptions:
        return self.delegate.get_udf_options()

    def get_options(self) -> dict[str, Any]:
        return dict(self.http_options)

    def supports_image_inputs(self) -> bool:
        return bool(self.media_types)

    def supported_media_mime_types(self) -> frozenset[str] | None:
        return self.media_types or None

    def instantiate(self) -> HTTPPrompter:
        result = HTTPPrompter(
            options=self.delegate.options,
            model=self.delegate.model_name,
            system_message=self.delegate.system_message,
            return_format=self.delegate.return_format,
            return_raw_response=self.delegate.return_raw_response,
            provider_name=self.delegate.provider_name,
            client_options=self.delegate.client_options,
            strict_structured_outputs=False,
        )
        result.media_types = self.media_types
        return result


class HTTPPrompter(OpenAIPrompter):
    media_types: frozenset[str]

    def _process_bytes(self, msg: bytes | PromptMedia) -> dict[str, Any]:
        mime = msg.content_type if isinstance(msg, PromptMedia) else _IMAGE_MIME_POLICY.require_supported(msg)
        if mime not in self.media_types:
            raise ValueError("media MIME type was not declared by the HTTP model")
        if mime.startswith("video/"):
            return {
                "type": "video_url",
                "video_url": {"url": "data:" + mime + ";base64," + base64.b64encode(bytes(msg)).decode("ascii")},
            }
        return super()._process_bytes(msg)


class HTTPProviderMixin(Provider):
    """HTTP factories shared by the two native inference provider facades."""

    _http_family: str

    def _http_prompter(
        self,
        model: str | None,
        system_message: str | None,
        return_format: dict[str, Any] | None,
        return_raw_response: bool,
        options: Mapping[str, Any] | None,
    ) -> HTTPPrompterDescriptor:
        return HTTPPrompterDescriptor(
            self.name, self._http_family, model, system_message, return_format, return_raw_response, options or {}
        )

    def get_text_embedder(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> HTTPTextEmbedderDescriptor:
        return HTTPTextEmbedderDescriptor(model or "", dimensions, dict(options or {}), self.name, self._http_family)

    def get_image_embedder(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> HTTPImageEmbedderDescriptor:
        return HTTPImageEmbedderDescriptor(model or "", dimensions, dict(options or {}), self.name, self._http_family)

    def get_video_embedder(
        self, model: str | None = None, dimensions: int | None = None, *, options: Mapping[str, Any] | None = None
    ) -> HTTPVideoEmbedderDescriptor:
        return HTTPVideoEmbedderDescriptor(model or "", dimensions, dict(options or {}), self.name, self._http_family)
