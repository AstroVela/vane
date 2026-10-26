"""Result-cardinality (rows-to-Python) sweep via LIMIT n, no ORDER BY. See benchmarks/README.md.

`SELECT * FROM src LIMIT n` early-stops the scan, so per-row conversion dominates and the slope is monotone in n.
A steeper slope on one build is a per-row conversion regression. n=100 is overhead, n=100_000 is throughput.
(An ORDER BY version was dropped: the top-N sort swamped the signal.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _scale import scaled

import duckdb

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_codspeed import BenchmarkFixture

# scale the source rows AND the top-N by the same factor, keeping small-N points fixed and SRC_ROWS >= max(LIMITS).
SRC_ROWS = scaled(200_000)
LIMITS = [100, 1_000, 10_000, scaled(100_000)]


@pytest.fixture(scope="module")
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    # source materialized ONCE (module-scoped) and identical across the n sweep; per-test build would add noise
    c = duckdb.connect(config={"threads": 1})
    c.execute(
        "CREATE TABLE src AS "
        f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b, ('s_' || i) AS s FROM range({SRC_ROWS}) t(i)"
    )
    yield c
    c.close()


def _query(n: int) -> str:
    return f"SELECT a, b, s FROM src LIMIT {n}"


@pytest.mark.gate  # fetchall materializes n rows -> binding-dominated; small-n end is the noise-free gate
@pytest.mark.parametrize("n", LIMITS)
def test_limit_fetchall(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, n: int) -> None:
    q = _query(n)
    con.execute(q).fetchall()  # warm
    benchmark(lambda: con.execute(q).fetchall())


@pytest.mark.gate  # df() materializes n rows to numpy columns -> binding-dominated
@pytest.mark.parametrize("n", LIMITS)
def test_limit_df(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, n: int) -> None:
    q = _query(n)
    con.sql(q).df()  # warm
    benchmark(lambda: con.sql(q).df())


@pytest.mark.informational  # to_arrow_table re-runs the query GIL-released (engine-parallel) -> not gated
@pytest.mark.parametrize("n", LIMITS)
def test_limit_to_arrow(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, n: int) -> None:
    q = _query(n)
    con.sql(q).to_arrow_table()  # warm
    benchmark(lambda: con.sql(q).to_arrow_table())
