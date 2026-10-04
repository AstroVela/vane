# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded decoded audio clips shared by embedding providers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from vane.ai._embedding_inputs import EmbeddingConfigurationError


@dataclass(frozen=True)
class AudioClip:
    """Frame-major floating PCM samples, with their actual sample rate."""

    samples: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class AudioInputSpec:
    """Static model limits; decoding, resampling and windowing precede embedding."""

    sample_rate: int
    max_samples: int
    max_channels: int = 2
    max_input_bytes: int = 16 * 1024**2

    def __post_init__(self) -> None:
        for name in ("sample_rate", "max_samples", "max_channels", "max_input_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"AudioInputSpec {name} must be a positive integer")


def audio_clips_from_arrow(column: pa.ChunkedArray, spec: AudioInputSpec) -> list[AudioClip | None]:
    """Read tensor buffers without expanding every sample into a Python object."""
    from vane._tensor import _validate_arrow_tensor_type, _validate_shape
    from vane.execution.udf_file_contract import _struct_field_index

    dtype = column.type
    if not pa.types.is_struct(dtype):
        raise TypeError("EmbedAudio requires STRUCT(sample_rate INTEGER, data TENSOR)")
    rate_field, data_field = (
        _struct_field_index(dtype, name, boundary="EmbedAudio", path="audio") for name in ("sample_rate", "data")
    )
    tensor_type = dtype.field(data_field).type
    if dtype.field(rate_field).type not in (pa.int32(), pa.int64(), pa.null()):
        raise TypeError("EmbedAudio sample_rate must be INTEGER or BIGINT")
    if isinstance(tensor_type, pa.FixedShapeTensorType):
        shape = tensor_type.shape
        if tensor_type.permutation not in (None, [], list(range(len(shape)))):
            raise TypeError("EmbedAudio requires frame-major tensors")
        value_type = tensor_type.value_type
        variable = False
    else:
        tensor_type = _validate_arrow_tensor_type(tensor_type)
        shape, value_type, variable = tensor_type.uniform_shape, tensor_type.value_type, True
    if len(shape) != 2 or value_type not in (pa.float32(), pa.float64()):
        raise TypeError("EmbedAudio data must be a rank-two FLOAT or DOUBLE tensor (samples, channels)")
    result: list[AudioClip | None] = []
    for row in column:
        if not row.is_valid:
            result.append(None)
            continue
        rate, value = row[rate_field].as_py(), row[data_field]
        if rate != spec.sample_rate:
            raise EmbeddingConfigurationError(
                f"Audio embedding requires sample_rate={spec.sample_rate}; resample first"
            )
        if not value.is_valid:
            raise EmbeddingConfigurationError("Audio clips cannot contain NULL samples")
        storage = value.value
        pixels = storage["data"].values if variable else storage.values
        if pixels.null_count or pixels.nbytes > spec.max_input_bytes:
            raise EmbeddingConfigurationError("Audio clip contains NULL samples or exceeds the decoded byte limit")
        actual = _validate_shape(storage["shape"].as_py() if variable else shape, shape, len(pixels))
        if not 1 <= actual[0] <= spec.max_samples or not 1 <= actual[1] <= spec.max_channels:
            raise EmbeddingConfigurationError(
                f"Audio clip requires 1 to {spec.max_samples} samples and 1 to {spec.max_channels} channels"
            )
        samples = pixels.to_numpy(zero_copy_only=True).reshape(actual)
        if not np.isfinite(samples).all() or np.any(np.abs(samples) > 1):
            raise EmbeddingConfigurationError("Audio samples must be finite floating PCM in [-1, 1]")
        result.append(AudioClip(samples, rate))
    return result
