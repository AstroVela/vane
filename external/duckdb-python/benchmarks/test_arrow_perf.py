"""Arrow read/write: Table + RecordBatchReader + dictionary sweep. See benchmarks/README.md.

READ aggregates over real columns (arrow answers count(*) from metadata); WRITE drains the lazy reader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest
from _scale import scaled

import numpy as np

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

    import duckdb

N = scaled(500_000)
DICT_UNIQUE = [2, 1_000, 50_000]  # UNIQUE-value counts (cardinality sweep), not row counts -> NOT scaled
WRITE_Q_NUM = f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({N}) t(i)"
WRITE_Q_STR = f"SELECT ('str_value_' || i) AS s FROM range({N}) t(i)"

# informational: every bench here is engine-parallel or library/streaming dominated. READ = engine aggregate
# dominates; WRITE (to_arrow/pl) re-runs the query GIL-released. Would trip on engine/submodule bumps, not binding.
pytestmark = pytest.mark.informational


@pytest.fixture(scope="module")
def arrow_numeric() -> pa.Table:
    return pa.table(
        {
            "a": pa.array(range(N), type=pa.int64()),
            "b": pa.array([i * 1.5 for i in range(N)], type=pa.float64()),
        }
    )


@pytest.fixture(scope="module")
def arrow_string() -> pa.Table:
    return pa.table({"s": pa.array([f"str_value_{i}" for i in range(N)], type=pa.string())})


@pytest.fixture(scope="module")
def arrow_numeric_batches(arrow_numeric: pa.Table) -> tuple[pa.Schema, list[pa.RecordBatch]]:
    # RecordBatches are immutable/re-readable, so a fresh reader can be built from them every round
    return arrow_numeric.schema, arrow_numeric.to_batches(max_chunksize=50_000)


@pytest.fixture(scope="module")
def arrow_dict_tables() -> dict[int, pa.Table]:
    # deterministic indices (i % U) so the instruction count is reproducible (no PRNG)
    tables = {}
    for u in DICT_UNIQUE:
        uniques = pa.array([f"category_value_{i}" for i in range(u)], type=pa.string())
        idx = pa.array(np.arange(N, dtype="int32") % u, type=pa.int32())
        tables[u] = pa.table({"c": pa.DictionaryArray.from_arrays(idx, uniques)})
    return tables


# READ: arrow -> duckdb. sum/length force a full scan.


def test_read_arrow_numeric(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, arrow_numeric: pa.Table
) -> None:
    con.register("t_num", arrow_numeric)
    con.execute("SELECT sum(a), sum(b) FROM t_num").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT sum(a), sum(b) FROM t_num").fetchall())


def test_read_arrow_string(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, arrow_string: pa.Table) -> None:
    con.register("t_str", arrow_string)
    con.execute("SELECT count(s), sum(length(s)) FROM t_str").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT count(s), sum(length(s)) FROM t_str").fetchall())


def test_read_arrow_reader_numeric(
    benchmark: BenchmarkFixture,
    con: duckdb.DuckDBPyConnection,
    arrow_numeric_batches: tuple[pa.Schema, list[pa.RecordBatch]],
) -> None:
    # same factory as the Table read, but STREAMING: a fresh reader per round, drained by the engine
    schema, batches = arrow_numeric_batches

    def run() -> list:
        reader = pa.RecordBatchReader.from_batches(schema, iter(batches))
        con.register("t_rdr", reader)
        return con.execute("SELECT sum(a), sum(b) FROM t_rdr").fetchall()

    run()  # warm
    benchmark(run)


@pytest.mark.parametrize("unique", DICT_UNIQUE)
def test_read_arrow_dictionary(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, arrow_dict_tables: dict[int, pa.Table], unique: int
) -> None:
    # per-value dictionary DECODE cost slopes with the unique count (mirrors core test_arrow_dictionaries_scan)
    con.register("t_dict", arrow_dict_tables[unique])
    con.execute("SELECT count(c), sum(length(c)) FROM t_dict").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT count(c), sum(length(c)) FROM t_dict").fetchall())


# WRITE: duckdb -> arrow, consumer fully materializes / drains the stream.


def test_write_arrow_table_numeric(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    benchmark(lambda: con.sql(WRITE_Q_NUM).to_arrow_table())


def test_write_arrow_table_string(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    benchmark(lambda: con.sql(WRITE_Q_STR).to_arrow_table())


def test_write_arrow_reader_consumed(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    def run() -> int:
        reader = con.sql(WRITE_Q_NUM).to_arrow_reader(100_000)
        rows = 0
        for batch in reader:  # drain the lazy stream so duckdb produces every batch
            rows += batch.num_rows
        return rows

    benchmark(run)


def test_write_polars_numeric(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    benchmark(lambda: con.sql(WRITE_Q_NUM).pl())


def test_write_polars_string(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    benchmark(lambda: con.sql(WRITE_Q_STR).pl())
