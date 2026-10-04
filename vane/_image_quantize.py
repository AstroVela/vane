# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, bounded GIF median-cut palette shared with the native contract."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def _quantize_gif(source: np.ndarray, check: Callable[[], None]) -> tuple[np.ndarray, np.ndarray]:
    pixels = source.reshape(-1, source.shape[2])
    palette: npt.NDArray[np.uint8] = np.zeros((256, 3), dtype=np.uint8)
    if pixels.shape[1] == 1:
        palette[:] = np.arange(256, dtype=np.uint8)[:, None]
        return source[:, :, 0], palette
    counts: npt.NDArray[np.int64] = np.zeros(32768, dtype=np.int64)
    sums: npt.NDArray[np.int64] = np.zeros((32768, 3), dtype=np.int64)
    unique: set[int] = set()
    exact = True
    for begin in range(0, len(pixels), 16384):
        check()
        block: npt.NDArray[np.int64] = pixels[begin : begin + 16384].astype(np.int64)
        codes = (block[:, 0] << 16) | (block[:, 1] << 8) | block[:, 2]
        if exact:
            unique.update(np.unique(codes).tolist())
            if len(unique) > 256:
                exact = False
                unique.clear()
        bins = ((block[:, 0] >> 3) << 10) | ((block[:, 1] >> 3) << 5) | (block[:, 2] >> 3)
        counts += np.bincount(bins, minlength=32768)
        for channel in range(3):
            sums[:, channel] += np.bincount(bins, weights=block[:, channel], minlength=32768).astype(np.int64)
    lookup: npt.NDArray[np.uint8] = np.zeros(32768, dtype=np.uint8)
    colors = np.array(sorted(unique), dtype=np.int64)
    if exact:
        palette[: len(colors)] = (colors[:, None] >> np.array([16, 8, 0])) & 255
    else:
        coordinates = (np.arange(32768)[:, None] >> np.array([10, 5, 0])) & 31

        def box(bins: np.ndarray) -> tuple[np.ndarray, int, int, int]:
            extent = np.ptp(coordinates[bins], axis=0)
            axis = int(np.argmax(extent))
            population = int(counts[bins].sum())
            return bins, population, axis, population * int(extent[axis]) if len(bins) > 1 else -1

        boxes = [box(np.flatnonzero(counts))]
        while len(boxes) < 256:
            check()
            selected = max(range(len(boxes)), key=lambda i: boxes[i][3])
            bins, population, axis, score = boxes[selected]
            if score < 0:
                break
            bins = bins[np.lexsort((bins, coordinates[bins, axis]))]
            split = int(np.searchsorted(np.cumsum(counts[bins]), (population + 1) // 2)) + 1
            split = min(max(split, 1), len(bins) - 1)
            boxes[selected] = box(bins[:split])
            boxes.append(box(bins[split:]))
        for index, (bins, population, _, _) in enumerate(boxes):
            palette[index] = (sums[bins].sum(axis=0) + population // 2) // population
            lookup[bins] = index
    indices: npt.NDArray[np.uint8] = np.empty(len(pixels), dtype=np.uint8)
    for begin in range(0, len(pixels), 16384):
        check()
        block = pixels[begin : begin + 16384].astype(np.int64)
        if exact:
            codes = (block[:, 0] << 16) | (block[:, 1] << 8) | block[:, 2]
            indices[begin : begin + len(block)] = np.searchsorted(colors, codes)
        else:
            bins = ((block[:, 0] >> 3) << 10) | ((block[:, 1] >> 3) << 5) | (block[:, 2] >> 3)
            indices[begin : begin + len(block)] = lookup[bins]
    return indices.reshape(source.shape[:2]), palette
