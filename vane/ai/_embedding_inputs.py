# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit embedding input policies with model-specific token counting."""

from __future__ import annotations

from collections.abc import Callable, Iterator


class EmbeddingConfigurationError(ValueError):
    """A worker-discovered configuration error, never a nullable row error."""


def split_text(text: str, limit: int, count: Callable[[str], int], *, first_only: bool = False) -> list[str]:
    """Split on Unicode boundaries, checking every piece against the tokenizer.

    ``count`` includes any prompt and special tokens used by the model. Token
    counts need not be monotonic: binary search is only a packing heuristic.
    Full chunking backtracks when a chosen boundary strands the suffix;
    truncation only requires a fitting prefix, not a partition of the tail.
    """
    if not text:
        return []
    size = len(text)

    def candidates(start: int) -> Iterator[int]:
        measured: dict[int, bool] = {}

        def fits(end: int) -> bool:
            if end not in measured:
                measured[end] = count(text[start:end]) <= limit
            return measured[end]

        if fits(size):
            yield size
            return
        low, high = start + 1, size - 1
        accepted = 0
        while low <= high:
            middle = (low + high) // 2
            if fits(middle):
                accepted = middle
                low = middle + 1
            else:
                high = middle - 1
        if accepted:
            yield accepted
        # If the heuristic strands the tail, consider every other boundary,
        # including longer prefixes that BPE merges may make fit again.
        for end in range(size - 1, start, -1):
            if end != accepted and fits(end):
                yield end

    # Iterative search avoids the recursion limit for many small chunks. Each
    # unpartitionable suffix is explored once, even if several prefixes fit.
    boundaries = [0]
    pending = [candidates(0)]
    dead: set[int] = set()
    while pending:
        end = next(pending[-1], None)
        if end is None:
            dead.add(boundaries.pop())
            pending.pop()
        elif end not in dead:
            boundaries.append(end)
            if first_only or end == size:
                return [text[start:stop] for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)]
            pending.append(candidates(end))
    raise ValueError("Embedding token budget cannot fit an input chunk with its prompt")
