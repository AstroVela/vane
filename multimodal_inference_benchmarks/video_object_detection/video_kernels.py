"""Shared numerical kernels for the Vane and Ray Data video benchmarks.

Keep framework-specific Arrow/Ray block handling out of this module.  Both
benchmarks call these functions so execution-engine comparisons cannot drift
because one side silently changes image preprocessing or YOLO result handling.
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torchvision
from PIL import Image

RAY_DATA_TARGET_MAX_BLOCK_SIZE_BYTES = 128 * 1024 * 1024
RAY_DATA_MAX_SAFE_BLOCK_SIZE_NUMERATOR = 3
RAY_DATA_MAX_SAFE_BLOCK_SIZE_DENOMINATOR = 2


@dataclass(frozen=True)
class VideoGpuTransportConfig:
    """Byte limits aligned with Ray Data's block transport."""

    target_max_block_bytes: int
    input_hard_max_bytes: int
    output_hard_max_bytes: int


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, "").strip()
    value = int(raw_value) if raw_value else int(default)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def video_gpu_transport_config_from_env() -> VideoGpuTransportConfig:
    """Resolve the common Ray Data/Vane GPU block transport policy."""
    target_max_block_bytes = _positive_int_env(
        "VIDEO_TARGET_MAX_BLOCK_BYTES",
        RAY_DATA_TARGET_MAX_BLOCK_SIZE_BYTES,
    )
    default_safe_block_bytes = (
        target_max_block_bytes
        * RAY_DATA_MAX_SAFE_BLOCK_SIZE_NUMERATOR
        // RAY_DATA_MAX_SAFE_BLOCK_SIZE_DENOMINATOR
    )
    # Ray Data's map bundler preserves a complete upstream block and may add a
    # short block until it reaches one compute batch.  A target-plus-one-row
    # cap incorrectly slices that soft bundle (for example, 5 + 110 rows).
    # Its block shaper uses the same 1.5x safe threshold, which is the bounded
    # admission envelope Vane needs for this transport contract.
    input_hard_max_bytes = _positive_int_env(
        "VIDEO_GPU_INPUT_HARD_MAX_BYTES",
        default_safe_block_bytes,
    )
    if input_hard_max_bytes < target_max_block_bytes:
        raise ValueError(
            "VIDEO_GPU_INPUT_HARD_MAX_BYTES must be at least VIDEO_TARGET_MAX_BLOCK_BYTES"
        )
    default_output_hard_max_bytes = default_safe_block_bytes
    output_hard_max_bytes = _positive_int_env(
        "VIDEO_GPU_OUTPUT_HARD_MAX_BYTES",
        default_output_hard_max_bytes,
    )
    if output_hard_max_bytes < target_max_block_bytes:
        raise ValueError(
            "VIDEO_GPU_OUTPUT_HARD_MAX_BYTES must be at least VIDEO_TARGET_MAX_BLOCK_BYTES"
        )

    return VideoGpuTransportConfig(
        target_max_block_bytes=target_max_block_bytes,
        input_hard_max_bytes=input_hard_max_bytes,
        output_hard_max_bytes=output_hard_max_bytes,
    )


def frames_to_torch_tensor(
    frames: Iterable[np.ndarray],
    device: torch.device | None,
) -> torch.Tensor:
    """Convert uint8 RGB frames with Ray Data's reference TorchVision kernel.

    ``device=None`` preserves the original Ray Data benchmark boundary: PIL
    conversion and ``torch.stack`` produce a CPU tensor, and Ultralytics moves
    that tensor to the model device inside ``BasePredictor.preprocess``.
    """
    stack = torch.stack(
        [torchvision.transforms.functional.to_tensor(Image.fromarray(frame)) for frame in frames],
        dim=0,
    )
    return stack if device is None else stack.to(device=device)


def resize_rgb_frame(frame: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Resize one RGB frame with the benchmark's reference PIL operation."""
    return np.array(Image.fromarray(frame).resize((width, height)))


def crop_bbox_to_png(
    frame: np.ndarray,
    bbox: Iterable[float],
    *,
    pil_image: Image.Image | None = None,
    png_buffer: io.BytesIO | None = None,
) -> bytes:
    """Crop and PNG-encode one detection using the shared reference settings."""
    x1, y1, x2, y2 = map(int, bbox)
    source_image = pil_image if pil_image is not None else Image.fromarray(frame)
    cropped_image = source_image.crop((x1, y1, x2, y2))
    output = png_buffer if png_buffer is not None else io.BytesIO()
    output.seek(0)
    output.truncate(0)
    cropped_image.save(output, format="PNG", compress_level=2)
    return output.getvalue()


def yolo_result_to_features(result: Any) -> list[dict[str, Any]]:
    """Convert one Ultralytics result with the benchmark's reference algorithm."""
    return [
        {
            "label": label,
            "confidence": confidence.item(),
            "bbox": bbox.tolist(),
        }
        for label, confidence, bbox in zip(
            result.names,
            result.boxes.conf,
            result.boxes.xyxy,
            strict=False,
        )
    ]


__all__ = [
    "RAY_DATA_TARGET_MAX_BLOCK_SIZE_BYTES",
    "VideoGpuTransportConfig",
    "crop_bbox_to_png",
    "frames_to_torch_tensor",
    "resize_rgb_frame",
    "video_gpu_transport_config_from_env",
    "yolo_result_to_features",
]
