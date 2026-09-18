# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit embedding input policies with model-specific token counting."""

from __future__ import annotations

from collections.abc import Callable


class EmbeddingConfigurationError(ValueError):
    """A worker-discovered configuration error, never a nullable row error."""


def split_text(text: str, limit: int, count: Callable[[str], int], *, first_only: bool = False) -> list[str]:
    """Split on Unicode boundaries, checking every piece against the tokenizer.

    ``count`` includes any prompt and special tokens used by the model. Token
    counts need not be monotonic: binary search is only a packing heuristic;
    every accepted candidate is measured and need not be the longest prefix.
    """
    pieces = []
    remaining = text
    while remaining:
        if count(remaining) <= limit:
            pieces.append(remaining)
            break
        low, high = 1, len(remaining)
        accepted = 0
        while low <= high:
            middle = (low + high) // 2
            if count(remaining[:middle]) <= limit:
                accepted = middle
                low = middle + 1
            else:
                high = middle - 1
        if not accepted:
            # BPE merges can make several characters shorter than one. Do
            # not claim a single-character failure proves no prefix fits.
            accepted = next((i for i in range(1, len(remaining) + 1) if count(remaining[:i]) <= limit), 0)
        if not accepted:
            raise ValueError("Embedding token budget cannot fit an input chunk with its prompt")
        pieces.append(remaining[:accepted])
        if first_only:
            break
        remaining = remaining[accepted:]
    return pieces
