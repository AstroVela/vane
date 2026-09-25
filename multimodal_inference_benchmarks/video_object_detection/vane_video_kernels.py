# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Vane-only video fast path; the Ray Data reference kernels stay unchanged."""

from functools import partial

import numpy as np
import torch
from ultralytics.models.yolo.detect.predict import DetectionPredictor


def batch_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """Preserve the reference RGB/FP32 normalization with one batched conversion."""
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected NHWC uint8 RGB frames")
    # Arrow may expose read-only NumPy storage. from_numpy only reads it here:
    # contiguous() and float() allocate storage before the in-place division.
    return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float().div_(255)


class OriginalFramePredictor(DetectionPredictor):
    """Keep YOLO's NMS/result construction while reusing this call's RGB frames."""

    def __init__(self, *args, original_frames=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_frames = original_frames

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        frames = self.original_frames
        if frames is None or len(frames) != len(img):
            raise ValueError("Original frames must match the current inference batch")
        if tuple(frames.shape[1:3]) != tuple(img.shape[2:]):
            raise ValueError("This fast path requires already-resized tensor inputs")
        # A list tells the existing postprocessor the original images are ready.
        # No conversion hook or other process-global library state is replaced.
        return super().postprocess(preds, img, list(frames), **kwargs)


def predict_with_original_frames(model, tensor, frames):
    """Bind originals for one synchronous call, releasing them even on failure."""
    if model.predictor is not None:
        if not isinstance(model.predictor, OriginalFramePredictor):
            raise TypeError("The Vane fast path owns its YOLO predictor")
        model.predictor.original_frames = frames
    factory = partial(OriginalFramePredictor, original_frames=frames)
    try:
        return model(tensor, verbose=False, predictor=factory)
    finally:
        if isinstance(model.predictor, OriginalFramePredictor):
            model.predictor.original_frames = None
