"""Shared fixtures + marker registration for the benchmark suite. See benchmarks/README.md.

Markers are registered here (not via pyproject `markers=`) because pyproject sets `filterwarnings = ["error"]`,
so an unregistered mark would raise as a collection error. Every benchmark must carry EXACTLY ONE of `gate` /
`informational` so the two CI steps (`-m gate`, `-m informational`) cover the suite with no overlap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _scale import bench_scale, scaled  # noqa: F401  (re-exported as the shared home; used by the modules)

import duckdb

if TYPE_CHECKING:
    from collections.abc import Iterator


def pytest_configure(config: pytest.Config) -> None:
    """Register the gate/informational markers (required under filterwarnings=error)."""
    config.addinivalue_line(
        "markers",
        "gate: binding-dominated, instruction-count gate-able under Callgrind (deterministic).",
    )
    config.addinivalue_line(
        "markers",
        "informational: engine/library-diluted or streaming; reported, never gated.",
    )


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a fresh single-threaded connection, closed on teardown.

    `threads=1` pins engine parallelism so counts/walltime don't shift with the runner core count. The
    concurrency module overrides this deliberately.
    """
    c = duckdb.connect(config={"threads": 1})
    yield c
    c.close()
