"""Env-gated row-count scaling for the benchmark suite.

Callgrind is 20-50x, so the O(rows) benches at full N make the CI sweep too slow. `scaled(n)` shrinks row counts
ONLY when `BENCH_SCALE=<divisor>` is set (which the CI sweep sets); unset -> full N, so local walltime A/B is
unchanged. A gate bench and the engine floor it is compared against share a base N, so routing BOTH through
`scaled()` keeps them at an identical scaled N and the binding fraction stays valid. Scaling reduces row counts
only; it must never change the data patterns the benches depend on (real nulls, mixed ASCII, LIMIT-no-ORDER-BY).
A floor keeps a scaled bench row-dominated so per-element work still dominates; the small-N `*_gate` probes are
already fast and are NOT scaled.
"""

from __future__ import annotations

import os

FLOOR = 20_000  # a scaled bench never drops below this (stays row-dominated, ~10x the range(2048) probes)


def bench_scale() -> int:
    """Return the divisor from `BENCH_SCALE` (>=1); 1 (no scaling) if unset/invalid."""
    v = os.environ.get("BENCH_SCALE")
    if not v:
        return 1
    try:
        return max(int(v), 1)
    except ValueError:
        return 1


def scaled(n: int) -> int:
    """Return `n` at full scale, or `max(n // BENCH_SCALE, min(n, FLOOR))` when scaling is enabled."""
    d = bench_scale()
    if d <= 1:
        return n
    return max(n // d, min(n, FLOOR))
