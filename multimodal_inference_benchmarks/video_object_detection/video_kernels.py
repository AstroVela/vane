# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Numerical kernels shared by the Vane and Ray Data video benchmarks.

Keep framework-specific Arrow/Ray block handling out of this module. Both
benchmarks use the tensor and YOLO conversion helpers. Ray Data also uses the
Pillow crop/PNG helper as its reference implementation, while Vane deliberately
composes its native image crop and encode expressions.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torchvision
from PIL import Image


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
    "crop_bbox_to_png",
    "frames_to_torch_tensor",
    "resize_rgb_frame",
    "yolo_result_to_features",
]
