from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def _as_uint8_rgb_batch(frames: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    if isinstance(frames, np.ndarray) and frames.ndim == 4:
        batch = frames
    else:
        batch = np.stack(list(frames), axis=0)

    if batch.ndim != 4 or batch.dtype != np.uint8 or batch.shape[-1] != 3:
        raise ValueError("frames must be a uint8 RGB batch with shape (N, H, W, 3)")
    if not batch.flags.c_contiguous or not batch.flags.writeable:
        batch = np.array(batch, copy=True, order="C")
    return batch


def frames_to_tensor_batch(
    frames: np.ndarray | Sequence[np.ndarray],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    batch = _as_uint8_rgb_batch(frames)
    if device is None:
        return torch.from_numpy(batch).permute(0, 3, 1, 2).to(dtype=torch.float32).div_(255.0)
    return torch.from_numpy(batch).to(device=device).permute(0, 3, 1, 2).to(dtype=torch.float32).div_(255.0)
