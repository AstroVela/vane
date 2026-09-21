# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Ordered, bounded decoded clips shared by video embedding providers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from vane.ai._embedding_inputs import EmbeddingConfigurationError

if TYPE_CHECKING:
    from vane._image import Image


@dataclass(frozen=True)
class VideoClip:
    """One clip, with frames in presentation order and original timestamps."""

    frames: tuple[Image, ...]
    frame_times: tuple[float, ...]
    frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class VideoInputSpec:
    """Static provider input limits, available without loading a model.

    Frames are decoded UInt8 RGB images. Model resizing and normalization
    belong to the provider; temporal selection belongs to the caller.
    """

    frame_count: int | None = None
    max_frames: int = 256
    max_input_bytes: int = 64 * 1024**2

    def __post_init__(self) -> None:
        for name in ("max_frames", "max_input_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"VideoInputSpec {name} must be a positive integer")
        if self.frame_count is not None and (
            type(self.frame_count) is not int or not 1 <= self.frame_count <= self.max_frames
        ):
            raise ValueError("VideoInputSpec frame_count must be positive and at most max_frames")

    def validate_count(self, count: int) -> None:
        if count == 0 or count > self.max_frames:
            raise EmbeddingConfigurationError(f"Video clip requires 1 to {self.max_frames} frames")
        if self.frame_count is not None and count != self.frame_count:
            raise EmbeddingConfigurationError(f"Selected video model requires exactly {self.frame_count} frames")


def video_clips_from_arrow(column: pa.ChunkedArray, spec: VideoInputSpec) -> list[VideoClip | None]:
    """Read typed pixel buffers, keeping list offsets and NULL clips intact."""
    import vane
    from vane._image import _image_arrow_scalar_to_numpy, _ImageArrowType
    from vane.execution.udf_file_contract import _struct_field_index

    dtype = column.type
    if not (pa.types.is_list(dtype) or pa.types.is_large_list(dtype)) or not pa.types.is_struct(dtype.value_type):
        raise TypeError("EmbedVideo requires a LIST of video frame records")
    fields = dtype.value_type
    index_field, time_field, data_field = (
        _struct_field_index(fields, name, boundary="EmbedVideo", path="frames")
        for name in ("frame_index", "frame_time", "data")
    )
    # Governed UDF transport infers ordinary leaves. Empty/all-NULL leaves can
    # therefore be Arrow null even when the native binder checked their types.
    # A null leaf contains no non-NULL value to coerce; actual frames still go
    # through the required-index/timestamp checks below.
    if (
        fields.field(index_field).type not in (pa.int64(), pa.null())
        or fields.field(time_field).type not in (pa.float64(), pa.null())
        or not isinstance(fields.field(data_field).type, _ImageArrowType)
    ):
        raise TypeError("EmbedVideo frame records require frame_index BIGINT, frame_time DOUBLE, and data IMAGE")
    image_type = fields.field(data_field).type
    image_dtype = vane.image_type(image_type.mode, image_type.height, image_type.width)
    result: list[VideoClip | None] = []
    for value in column:
        if not value.is_valid:
            result.append(None)
            continue
        records = value.values
        spec.validate_count(len(records))
        if records.null_count:
            raise EmbeddingConfigurationError("Video clips cannot contain NULL frames")
        indices = records.field(index_field).to_pylist()
        times = records.field(time_field).to_pylist()
        if any(index is None or index < 0 for index in indices):
            raise EmbeddingConfigurationError("Video frame indices must be nonnegative")
        if any(t is None or not math.isfinite(t) or t < 0 for t in times):
            raise EmbeddingConfigurationError("Video frame timestamps must be finite and nonnegative")
        if any(a >= b for a, b in zip(indices, indices[1:])) or any(a > b for a, b in zip(times, times[1:])):
            raise EmbeddingConfigurationError("Video frames must have increasing indices and nondecreasing timestamps")
        frames = []
        total_bytes = 0
        for image in records.field(data_field):
            if not image.is_valid:
                raise EmbeddingConfigurationError("Video clips cannot contain NULL images")
            storage = image.value
            pixels = storage.values if image_dtype.is_fixed_shape_image() else storage["data"].values
            total_bytes += pixels.nbytes
            if total_bytes > spec.max_input_bytes:
                raise EmbeddingConfigurationError("Video clip exceeds the selected model's decoded byte limit")
            frame = _image_arrow_scalar_to_numpy(image, image_dtype)
            if frame.dtype.name != "uint8" or frame.ndim != 3 or frame.shape[2] != 3:
                raise EmbeddingConfigurationError("Video embedding requires decoded UInt8 RGB images")
            frames.append(frame)
        result.append(VideoClip(tuple(frames), tuple(times), tuple(indices)))
    return result
