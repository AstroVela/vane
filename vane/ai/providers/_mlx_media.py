# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Pure fixed-checkpoint media validation, shared by unary and joint MLX."""

from __future__ import annotations

import math

import numpy as np

from vane.ai._audio_embedding import AudioClip
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._video_embedding import VideoClip

MAX_IMAGE_PIXELS = 16 * 1024 * 1024
AUDIO_SAMPLE_RATE = 16000
MIN_AUDIO_SAMPLES = 161
MAX_AUDIO_SAMPLES = 480000
MAX_AUDIO_INPUT_BYTES = 4 * 1024 * 1024
MAX_VIDEO_FRAMES = 12
MAX_VIDEO_INPUT_BYTES = 64 * 1024 * 1024


def validate_image(value: object, *, max_pixels: int = MAX_IMAGE_PIXELS) -> None:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint8
        or value.ndim != 3
        or value.shape[2] != 3
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or value.shape[0] * value.shape[1] > max_pixels
    ):
        raise EmbeddingConfigurationError("MLX images require RGB uint8 HWC arrays within the 16 Mi pixel budget")


def validate_audio(clip: AudioClip, *, max_input_bytes: int = MAX_AUDIO_INPUT_BYTES) -> None:
    if not isinstance(clip, AudioClip):
        raise EmbeddingConfigurationError("MLX embed_audio requires AudioClip values")
    if (
        isinstance(clip.sample_rate, (bool, np.bool_))
        or not isinstance(clip.sample_rate, (int, np.integer))
        or clip.sample_rate != AUDIO_SAMPLE_RATE
    ):
        raise EmbeddingConfigurationError("MLX audio requires sample_rate=16000; resample first")
    samples = clip.samples
    if (
        not isinstance(samples, np.ndarray)
        or samples.ndim != 2
        or samples.shape[1] != 1
        or samples.dtype not in (np.dtype("float32"), np.dtype("float64"))
    ):
        raise EmbeddingConfigurationError("MLX audio requires mono float32/float64 PCM shaped (samples, 1)")
    if not MIN_AUDIO_SAMPLES <= len(samples) <= MAX_AUDIO_SAMPLES:
        raise EmbeddingConfigurationError(
            "MLX audio requires 161 to 480000 samples; short tails and truncation are unsupported"
        )
    if samples.nbytes > max_input_bytes:
        raise EmbeddingConfigurationError("MLX audio exceeds the 4 MiB decoded input budget")
    if not np.isfinite(samples).all() or np.any(samples < -1) or np.any(samples > 1):
        raise EmbeddingConfigurationError("MLX audio requires finite floating PCM in [-1, 1]")


def validate_video(
    clip: VideoClip, *, max_pixels: int = MAX_IMAGE_PIXELS, max_input_bytes: int = MAX_VIDEO_INPUT_BYTES
) -> None:
    if not isinstance(clip, VideoClip):
        raise EmbeddingConfigurationError("MLX embed_video requires VideoClip values")
    if not all(isinstance(value, (tuple, list)) for value in (clip.frames, clip.frame_times, clip.frame_indices)):
        raise EmbeddingConfigurationError("MLX video requires frame, time and index sequences")
    count = len(clip.frames)
    if not 1 <= count <= MAX_VIDEO_FRAMES or len(clip.frame_times) != count or len(clip.frame_indices) != count:
        raise EmbeddingConfigurationError("MLX video requires 1 to 12 frames with aligned times and indices")
    if any(
        isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or index < 0
        for index in clip.frame_indices
    ) or any(a >= b for a, b in zip(clip.frame_indices, clip.frame_indices[1:])):
        raise EmbeddingConfigurationError("MLX video frame indices must be nonnegative strictly increasing integers")
    if any(
        isinstance(t, (bool, np.bool_))
        or not isinstance(t, (int, float, np.integer, np.floating))
        or not math.isfinite(t)
        or t < 0
        for t in clip.frame_times
    ) or any(a > b for a, b in zip(clip.frame_times, clip.frame_times[1:])):
        raise EmbeddingConfigurationError("MLX video frame times must be finite nonnegative and nondecreasing")
    total_bytes = 0
    shape = None
    for frame in clip.frames:
        validate_image(frame, max_pixels=max_pixels)
        if shape is not None and frame.shape != shape:
            raise EmbeddingConfigurationError("MLX video frames must have the same H/W within each clip")
        shape = frame.shape
        total_bytes += frame.nbytes
    if total_bytes > max_input_bytes:
        raise EmbeddingConfigurationError("MLX video exceeds the 64 MiB decoded clip budget")


def patch_tokens(
    pixels: np.ndarray,
    positions: np.ndarray,
    media_items: int,
    *,
    max_patches: int,
    kind: str,
    strict_grid: bool = True,
) -> list[int]:
    if (
        pixels.ndim != 3
        or pixels.shape[0] != media_items
        or not 0 < pixels.shape[1] <= max_patches
        or pixels.shape[2] != 768
        or (
            pixels.dtype not in (np.dtype("float32"), np.dtype("float64"))
            if strict_grid
            else not np.issubdtype(pixels.dtype, np.floating) or pixels.dtype == np.float16
        )
        or not np.isfinite(pixels).all()
        or positions.shape != (*pixels.shape[:2], 2)
        or not np.issubdtype(positions.dtype, np.integer)
    ):
        raise EmbeddingConfigurationError(f"MLX {kind} requires finite patches and positions for every input frame")
    valid, padding = (positions >= 0).all(axis=-1), (positions == -1).all(axis=-1)
    if not (valid | padding).all() or not valid.any(axis=1).all() or np.any(positions[valid] >= 10240):
        raise EmbeddingConfigurationError(f"MLX {kind} patch positions must be valid coordinates or paired -1 padding")
    tokens = []
    if not strict_grid:
        return []
    for frame_positions, frame_valid in zip(positions, valid):
        coords = frame_positions[frame_valid]
        if not np.array_equal(frame_valid, np.arange(len(frame_valid)) < len(coords)):
            raise EmbeddingConfigurationError(f"MLX {kind} patch padding must follow the complete frame grid")
        width, height = (int(value) + 1 for value in coords.max(axis=0))
        # Complete row-major 3x3 pooling grids prevent silently omitted or
        # repeated patches from satisfying only a total-token check.
        if height % 3 or width % 3 or len(coords) != height * width:
            raise EmbeddingConfigurationError(f"MLX {kind} patches must form complete 3x3 pooling grids")
        # The SDK stores (x, y), with x advancing first in each row.
        expected = np.column_stack((np.tile(np.arange(width), height), np.repeat(np.arange(height), width)))
        if not np.array_equal(coords, expected):
            raise EmbeddingConfigurationError(f"MLX {kind} patch positions must preserve the ordered frame grid")
        tokens.append(height * width // 9)
    return tokens


def audio_tokens(inputs: dict[str, np.ndarray], lengths: list[int]) -> list[int]:
    features, mask = inputs["input_features"], inputs["input_features_mask"]
    # The fixed extractor pads the waveform to 128 samples, then unfolds
    # 321-sample windows after 160 samples of left padding. Check T too:
    # checking only the returned mask's prefix would miss a truncated tail.
    padded_samples = ((max(lengths) + 127) // 128) * 128
    feature_length = (padded_samples - 161) // 160 + 1
    if (
        features.shape != (len(lengths), feature_length, 128)
        or features.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or not np.isfinite(features).all()
        or mask.shape != features.shape[:2]
        or not (np.issubdtype(mask.dtype, np.integer) or mask.dtype == np.bool_)
        or not np.isin(mask, (0, 1)).all()
    ):
        raise EmbeddingConfigurationError("MLX audio requires complete finite mel features and binary masks")
    expected = np.arange(feature_length)[None, :] * 160 + 160 < np.asarray(lengths)[:, None]
    if not expected.any(axis=1).all() or not np.array_equal(mask, expected):
        raise EmbeddingConfigurationError("MLX audio mask must preserve every original waveform's valid prefix")
    return [int(value) for value in expected[:, ::4].sum(axis=1)]


def validate_frame_counts(frame_counts: np.ndarray, counts: list[int]) -> None:
    if (
        frame_counts.shape != (len(counts),)
        or not np.issubdtype(frame_counts.dtype, np.integer)
        or not np.array_equal(frame_counts, counts)
    ):
        raise EmbeddingConfigurationError("MLX video processor must preserve each clip's frame count")
