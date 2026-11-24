# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Serializable multimodal contracts and bounded Arrow input conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from vane.ai._audio_embedding import AudioInputSpec, audio_clips_from_arrow
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._video_embedding import VideoInputSpec, video_clips_from_arrow
from vane.ai.embedding import Modality


@dataclass(frozen=True)
class EmbeddingPartSpec:
    kind: Modality
    mime_type: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"text", "image", "video", "audio"}:
            raise ValueError("EmbeddingPartSpec requires a supported modality")
        if self.mime_type is not None and (
            self.kind == "text" or not isinstance(self.mime_type, str) or not self.mime_type.startswith(f"{self.kind}/")
        ):
            raise ValueError("EmbeddingPartSpec MIME type must match its media modality")


@dataclass(frozen=True)
class EmbeddingPart:
    spec: EmbeddingPartSpec
    value: Any


@dataclass(frozen=True)
class EmbeddingContent:
    parts: tuple[EmbeddingPart, ...]


@dataclass(frozen=True)
class MultimodalInputSpec:
    """Closed representations supported by one model/protocol combination."""

    supported_parts: frozenset[EmbeddingPartSpec]
    video: VideoInputSpec | None = None
    audio: AudioInputSpec | None = None
    max_parts: int = 16
    max_input_bytes: int = 16 * 1024**2

    def __post_init__(self) -> None:
        if not self.supported_parts or any(not isinstance(part, EmbeddingPartSpec) for part in self.supported_parts):
            raise ValueError("MultimodalInputSpec requires supported part specifications")
        object.__setattr__(self, "supported_parts", frozenset(self.supported_parts))
        for name in ("max_parts", "max_input_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"MultimodalInputSpec {name} must be a positive integer")
        if EmbeddingPartSpec("video") in self.supported_parts and not isinstance(self.video, VideoInputSpec):
            raise ValueError("Decoded video requires a VideoInputSpec")
        if EmbeddingPartSpec("audio") in self.supported_parts and not isinstance(self.audio, AudioInputSpec):
            raise ValueError("Decoded audio requires an AudioInputSpec")

    def validate_layout(self, parts: tuple[EmbeddingPartSpec, ...]) -> None:
        if not 1 <= len(parts) <= self.max_parts:
            raise EmbeddingConfigurationError(f"Multimodal embedding requires 1 to {self.max_parts} parts")
        for part in parts:
            if part not in self.supported_parts:
                representation = part.mime_type or "decoded"
                raise EmbeddingConfigurationError(
                    f"Selected model/protocol does not support {part.kind} input ({representation})"
                )


def image_values(column: pa.ChunkedArray) -> list[Any]:
    import vane
    from vane._image import _image_arrow_scalar_to_numpy, _ImageArrowType

    arrow_type = column.type
    if not isinstance(arrow_type, _ImageArrowType):
        raise TypeError("Embedding requires a decoded IMAGE column")
    dtype = vane.image_type(arrow_type.mode, arrow_type.height, arrow_type.width)
    return [None if not value.is_valid else _image_arrow_scalar_to_numpy(value, dtype) for value in column]


def part_values(column: pa.ChunkedArray, part: EmbeddingPartSpec, spec: MultimodalInputSpec) -> list[Any]:
    if part.mime_type is not None:
        if not (
            pa.types.is_binary(column.type) or pa.types.is_large_binary(column.type) or pa.types.is_null(column.type)
        ):
            raise TypeError("Encoded embedding inputs require a BLOB column")
        # Check sizes before materializing Python bytes, including each row.
        for scalar in column:
            if scalar.is_valid and not 0 < scalar.as_buffer().size <= spec.max_input_bytes:
                raise EmbeddingConfigurationError("Encoded embedding part is empty or exceeds the byte budget")
        return column.to_pylist()
    if part.kind == "text":
        if not (
            pa.types.is_string(column.type) or pa.types.is_large_string(column.type) or pa.types.is_null(column.type)
        ):
            raise TypeError("TextInput requires a VARCHAR column")
        return column.to_pylist()
    if part.kind == "image":
        return image_values(column)
    if part.kind == "video":
        assert spec.video is not None
        return video_clips_from_arrow(column, spec.video)
    assert spec.audio is not None
    return audio_clips_from_arrow(column, spec.audio)


def content_from_columns(
    table: pa.Table, parts: tuple[EmbeddingPartSpec, ...], spec: MultimodalInputSpec
) -> list[EmbeddingContent]:
    spec.validate_layout(parts)
    columns = [part_values(table.column(f"part_{index}"), part, spec) for index, part in enumerate(parts)]
    results = []
    for values in zip(*columns, strict=True):
        active = tuple(
            EmbeddingPart(part, value)
            for part, value in zip(parts, values, strict=True)
            if value is not None and not (part.kind == "text" and not value.strip())
        )
        if not active:
            raise EmbeddingConfigurationError("Multimodal embedding row has no nonempty parts")
        total_bytes = 0
        for part in active:
            if part.spec.mime_type is not None:
                total_bytes += len(part.value)
            elif part.spec.kind == "text":
                total_bytes += len(part.value.encode("utf-8"))
            elif part.spec.kind == "image":
                total_bytes += part.value.nbytes
            elif part.spec.kind == "video":
                total_bytes += sum(frame.nbytes for frame in part.value.frames)
            else:
                total_bytes += part.value.samples.nbytes
        if total_bytes > spec.max_input_bytes:
            raise EmbeddingConfigurationError("Multimodal embedding row exceeds the combined input byte budget")
        results.append(EmbeddingContent(active))
    return results
