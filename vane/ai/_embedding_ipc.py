# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded JSON metadata and typed CPU buffer layouts for embedding IPC."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from vane.ai._audio_embedding import AudioClip, AudioInputSpec
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec, MultimodalInputSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._video_embedding import VideoClip, VideoInputSpec

VERSION = 2
MAX_CONTROL_BYTES = 1024**2


@dataclass(frozen=True)
class ModelContract:
    encoding_version: int
    revision: str
    dtype: str
    max_length: int
    overlength: str

    def __post_init__(self) -> None:
        if (
            type(self.encoding_version) is not int
            or self.encoding_version != 1
            or not isinstance(self.revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", self.revision) is None
            or self.dtype != "bfloat16"
            or type(self.max_length) is not int
            or not 1 <= self.max_length <= 2048
            or self.overlength != "error"
        ):
            raise EmbeddingConfigurationError("Invalid embedding model contract")

    def to_wire(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiteRTEncoding:
    model_sha256: str
    max_length: int
    vision_tokens_per_image: int
    input_type: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.model_sha256) is None
            or type(self.max_length) is not int
            or self.max_length not in (128, 256, 512, 1024, 2048, 8192)
            or type(self.vision_tokens_per_image) is not int
            or self.vision_tokens_per_image not in (70, 140)
            or not isinstance(self.input_type, str)
            or self.input_type not in {"query", "document"}
        ):
            raise EmbeddingConfigurationError("Invalid LiteRT service encoding")

    def to_wire(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RequestContext:
    encoding: str = "default"
    model_contract: ModelContract | None = None
    request_id: str | None = None
    litert: LiteRTEncoding | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.encoding, str) or self.encoding not in {
            "default",
            "text_document",
            "text_query",
            "image",
            "video",
            "audio",
            "multimodal",
        }:
            raise EmbeddingConfigurationError("Invalid embedding request encoding")
        if self.request_id is not None and (
            not isinstance(self.request_id, str) or re.fullmatch(r"[0-9a-f]{32}", self.request_id) is None
        ):
            raise EmbeddingConfigurationError("Invalid embedding request identity")

    @classmethod
    def from_wire(cls, message: dict[str, Any]) -> RequestContext:
        value = message["model_contract"]
        if value is not None:
            fields = {"encoding_version", "revision", "dtype", "max_length", "overlength"}
            if not isinstance(value, dict) or set(value) != fields:
                raise EmbeddingConfigurationError("Invalid embedding model contract fields")
            value = ModelContract(**value)
        encoding = message["encoding"]
        if isinstance(encoding, dict):
            fields = {"model_sha256", "max_length", "vision_tokens_per_image", "input_type"}
            if set(encoding) != fields or value is not None:
                raise EmbeddingConfigurationError("Invalid LiteRT service encoding fields")
            return cls("default", None, message.get("request_id"), LiteRTEncoding(**encoding))
        return cls(encoding, value, message.get("request_id"))


def input_spec(options: dict[str, Any]) -> MultimodalInputSpec:
    modalities = options.get("input_modalities", ["text", "image", "video"])
    if (
        not isinstance(modalities, list)
        or not modalities
        or any(not isinstance(item, str) or item not in {"text", "image", "video", "audio"} for item in modalities)
        or len(set(modalities)) != len(modalities)
    ):
        raise EmbeddingConfigurationError("input_modalities must declare distinct decoded modalities")
    limit = options.get("max_input_bytes", 64 * 1024**2)
    frames = options.get("max_frames", 64)
    if type(limit) is not int or not 1 <= limit <= 64 * 1024**2:
        raise EmbeddingConfigurationError("max_input_bytes must be between 1 and 64 MiB")
    if type(frames) is not int or not 1 <= frames <= 256:
        raise EmbeddingConfigurationError("max_frames must be between 1 and 256")
    audio = None
    if "audio" in modalities:
        rate, samples = options.get("sample_rate"), options.get("max_samples")
        if type(rate) is not int or not 1 <= rate <= 384000 or type(samples) is not int or not 1 <= samples <= 2**24:
            raise EmbeddingConfigurationError("Audio requires explicit bounded sample_rate and max_samples")
        audio = AudioInputSpec(rate, samples, max_channels=1, max_input_bytes=limit)
    elif "sample_rate" in options or "max_samples" in options:
        raise EmbeddingConfigurationError("Audio limits require audio in input_modalities")
    return MultimodalInputSpec(
        frozenset(EmbeddingPartSpec(kind) for kind in modalities),
        video=VideoInputSpec(max_frames=frames, max_input_bytes=limit) if "video" in modalities else None,
        audio=audio,
        max_input_bytes=limit,
    )


def describe(
    content: EmbeddingContent, spec: MultimodalInputSpec
) -> tuple[list[dict[str, Any]], list[np.ndarray[Any, Any]], int]:
    """Validate transport types/budgets before admission; describe array copies.

    Standalone TextInput preserves blank strings. MultimodalInput's conversion
    layer owns omission of blank parts and rejection of all-empty rows.
    """
    spec.validate_layout(tuple(part.spec for part in content.parts))
    parts: list[dict[str, Any]] = []
    arrays: list[np.ndarray[Any, Any]] = []
    offset = 0

    def buffer(value: Any, kind: str) -> dict[str, Any]:
        nonlocal offset
        if not isinstance(value, np.ndarray) or not value.size:
            raise EmbeddingConfigurationError("Decoded media requires nonempty CPU arrays")
        if kind in {"image", "video"}:
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                raise EmbeddingConfigurationError("Decoded images and frames require uint8 HWC RGB")
        elif value.dtype not in (np.dtype("float32"), np.dtype("float64")) or value.ndim != 2 or value.shape[1] != 1:
            raise EmbeddingConfigurationError("Native audio requires mono floating PCM shaped (samples, 1)")
        if offset + value.nbytes > spec.max_input_bytes:
            raise EmbeddingConfigurationError("Embedding media exceeds input byte budget")
        result = {"offset": offset, "shape": list(value.shape), "dtype": value.dtype.str, "nbytes": value.nbytes}
        arrays.append(value)
        offset += value.nbytes
        return result

    for part in content.parts:
        kind, value = part.spec.kind, part.value
        entry: dict[str, Any] = {"kind": kind}
        if kind == "text":
            if not isinstance(value, str):
                raise EmbeddingConfigurationError("Embedding text must be a string")
            encoded = value.encode("utf-8")
            if offset + len(encoded) > spec.max_input_bytes:
                raise EmbeddingConfigurationError("Embedding row exceeds combined input byte budget")
            entry.update(offset=offset, nbytes=len(encoded))
            arrays.append(np.frombuffer(encoded, dtype=np.uint8))
            offset += len(encoded)
        elif kind == "image":
            entry["buffer"] = buffer(value, kind)
        elif kind == "video":
            if not isinstance(value, VideoClip) or spec.video is None:
                raise EmbeddingConfigurationError("Expected a decoded VideoClip")
            spec.video.validate_count(len(value.frames))
            times, indices = value.frame_times, value.frame_indices
            if (
                len(times) != len(value.frames)
                or len(indices) != len(times)
                or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times)
                or any(a > b for a, b in zip(times, times[1:]))
                or any(type(i) is not int or i < 0 for i in indices)
                or any(a >= b for a, b in zip(indices, indices[1:]))
            ):
                raise EmbeddingConfigurationError("Video timestamps and indices must be ordered and aligned")
            entry.update(
                buffers=[buffer(frame, kind) for frame in value.frames], times=list(times), indices=list(indices)
            )
        else:
            if not isinstance(value, AudioClip) or spec.audio is None or value.sample_rate != spec.audio.sample_rate:
                raise EmbeddingConfigurationError("Audio must match the declared sample_rate; resample first")
            entry.update(buffer=buffer(value.samples, kind), sample_rate=value.sample_rate)
            if (
                len(value.samples) > spec.audio.max_samples
                or not np.isfinite(value.samples).all()
                or np.any(np.abs(value.samples) > 1)
            ):
                raise EmbeddingConfigurationError("Audio exceeds sample budget or contains invalid PCM")
        parts.append(entry)
    return parts, arrays, offset


def validate_layout(parts: Any, size: Any, spec: MultimodalInputSpec) -> int:
    """Validate all offsets/shapes/dtypes and metadata before allocating a buffer."""
    if type(size) is not int or not 0 <= size <= spec.max_input_bytes or not isinstance(parts, list):
        raise EmbeddingConfigurationError("Invalid embedding buffer size or parts")
    if not 1 <= len(parts) <= spec.max_parts:
        raise EmbeddingConfigurationError("Invalid embedding part count")
    offset = 0
    for part in parts:
        if not isinstance(part, dict) or part.get("kind") not in {p.kind for p in spec.supported_parts}:
            raise EmbeddingConfigurationError("Unsupported embedding part")
        kind = part["kind"]
        keys = {
            "text": {"kind", "offset", "nbytes"},
            "image": {"kind", "buffer"},
            "video": {"kind", "buffers", "times", "indices"},
            "audio": {"kind", "buffer", "sample_rate"},
        }[kind]
        if set(part) != keys:
            raise EmbeddingConfigurationError("Invalid embedding part fields")
        if kind == "text":
            if (
                type(part["offset"]) is not int
                or part["offset"] != offset
                or type(part["nbytes"]) is not int
                or part["nbytes"] < 0
                or offset + part["nbytes"] > size
            ):
                raise EmbeddingConfigurationError("Text exceeds or overlaps the reserved buffer")
            offset += part["nbytes"]
            continue
        buffers = part["buffers"] if kind == "video" else [part["buffer"]]
        if not isinstance(buffers, list) or not buffers or len(buffers) > 256:
            raise EmbeddingConfigurationError("Invalid media buffer count")
        for item in buffers:
            if not isinstance(item, dict) or set(item) != {"offset", "shape", "dtype", "nbytes"}:
                raise EmbeddingConfigurationError("Invalid array descriptor")
            shape = item["shape"]
            if (
                not isinstance(shape, list)
                or not 1 <= len(shape) <= 3
                or any(type(n) is not int or n <= 0 or n > spec.max_input_bytes for n in shape)
            ):
                raise EmbeddingConfigurationError("Invalid array shape")
            dtype = item["dtype"]
            allowed = (np.dtype("float32").str, np.dtype("float64").str) if kind == "audio" else ("|u1",)
            if not isinstance(dtype, str) or dtype not in allowed:
                raise EmbeddingConfigurationError("Invalid array dtype")
            nbytes = math.prod(shape) * np.dtype(dtype).itemsize
            if (
                type(item["offset"]) is not int
                or item["offset"] != offset
                or type(item["nbytes"]) is not int
                or item["nbytes"] != nbytes
                or offset + nbytes > size
            ):
                raise EmbeddingConfigurationError("Array exceeds or overlaps the reserved buffer")
            if (kind == "audio" and (len(shape) != 2 or shape[1] != 1)) or (
                kind != "audio" and (len(shape) != 3 or shape[2] != 3)
            ):
                raise EmbeddingConfigurationError("Invalid decoded media layout")
            offset += nbytes
        if kind == "video":
            assert spec.video is not None
            spec.video.validate_count(len(buffers))
            times, indices = part["times"], part["indices"]
            if (
                not isinstance(times, list)
                or not isinstance(indices, list)
                or len(times) != len(buffers)
                or len(indices) != len(buffers)
                or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times)
                or any(type(i) is not int or i < 0 for i in indices)
                or any(a > b for a, b in zip(times, times[1:]))
                or any(a >= b for a, b in zip(indices, indices[1:]))
            ):
                raise EmbeddingConfigurationError("Invalid frame timeline")
        if kind == "audio":
            assert spec.audio is not None
            if (
                type(part["sample_rate"]) is not int
                or part["sample_rate"] != spec.audio.sample_rate
                or buffers[0]["shape"][0] > spec.audio.max_samples
                or buffers[0]["nbytes"] > spec.audio.max_input_bytes
            ):
                raise EmbeddingConfigurationError("Invalid audio sample rate or length")
    if offset != size:
        raise EmbeddingConfigurationError("Embedding exceeds or mismatches its byte budget")
    return max(1, size)


def restore(parts: list[dict[str, Any]], memory: Any) -> EmbeddingContent:
    def array_value(item: dict[str, Any]) -> np.ndarray[Any, Any]:
        return np.ndarray(item["shape"], dtype=item["dtype"], buffer=memory, offset=item["offset"])

    values = []
    value: Any
    for part in parts:
        kind = part["kind"]
        if kind == "text":
            start, length = part["offset"], part["nbytes"]
            value = bytes(memory[start : start + length]).decode("utf-8") if length else ""
        elif kind == "video":
            value = VideoClip(
                tuple(array_value(item) for item in part["buffers"]), tuple(part["times"]), tuple(part["indices"])
            )
        elif kind == "audio":
            value = AudioClip(array_value(part["buffer"]), part["sample_rate"])
        else:
            value = array_value(part["buffer"])
        values.append(EmbeddingPart(EmbeddingPartSpec(kind), value))
    return EmbeddingContent(tuple(values))


def encode(value: Any) -> bytes:
    data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_CONTROL_BYTES:
        raise EmbeddingConfigurationError("Embedding control message exceeds 1 MiB")
    return data
