"""Python UDFs: native scalar (one call per row) and vectorized arrow (one call per chunk). See benchmarks/README.md.

Each UDF is wrapped in a sum()/length() aggregate so the engine runs it on every row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _scale import scaled

from duckdb.sqltypes import BIGINT, DOUBLE, VARCHAR

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

    import duckdb

pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")

NATIVE_N = scaled(200_000)  # native = one Python call per row, keep moderate
ARROW_N = scaled(1_000_000)  # arrow = one Python call per chunk (vectorized), can be large


def _bench(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, query: str) -> None:
    con.execute(query).fetchall()  # warm the engine + import caches
    benchmark(lambda: con.execute(query).fetchall())


# NATIVE scalar UDF: per-row TupleBuilder(args) + PyObject_CallObject + TransformPythonObject(result). The Python
# call dominates; the sum() consume is negligible -> gate.


@pytest.mark.gate
def test_udf_native_int_1arg(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    con.create_function("add_one", lambda x: x + 1, [BIGINT], BIGINT)
    _bench(benchmark, con, f"SELECT sum(add_one(i::BIGINT)) FROM range({NATIVE_N}) t(i)")


@pytest.mark.gate
def test_udf_native_int_2arg(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    con.create_function("add2", lambda a, b: a + b, [BIGINT, BIGINT], BIGINT)
    _bench(benchmark, con, f"SELECT sum(add2(i::BIGINT, (i + 1)::BIGINT)) FROM range({NATIVE_N}) t(i)")


@pytest.mark.gate
def test_udf_native_double_1arg(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    con.create_function("scale", lambda x: x * 1.5, [DOUBLE], DOUBLE)
    _bench(benchmark, con, f"SELECT sum(scale((i * 1.0)::DOUBLE)) FROM range({NATIVE_N}) t(i)")


@pytest.mark.gate
def test_udf_native_string(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    con.create_function("up", lambda s: s.upper(), [VARCHAR], VARCHAR)
    _bench(
        benchmark,
        con,
        f"SELECT sum(length(up(s))) FROM (SELECT ('str_value_' || i) AS s FROM range({NATIVE_N}) t(i))",
    )


@pytest.mark.gate
def test_udf_native_null_inputs(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    # DEFAULT null handling short-circuits NULL inputs (SetNull) WITHOUT calling the UDF: measures the validity
    # short-circuit, so the UDF only ever sees non-NULL rows.
    con.create_function("add_one", lambda x: x + 1, [BIGINT], BIGINT)
    _bench(
        benchmark,
        con,
        "SELECT sum(add_one(v)) FROM "
        f"(SELECT CASE WHEN i % 2 = 0 THEN NULL ELSE i::BIGINT END AS v FROM range({NATIVE_N}) t(i))",
    )


# ARROW (vectorized) UDF: ConvertDataChunkToPyArrowTable -> pc op -> ConvertArrowTableToVector cast. pyarrow lib
# work + per-chunk conversion + 1M engine -> informational.


@pytest.mark.informational
def test_udf_arrow_int(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    con.create_function("arrow_add_one", lambda x: pc.add(x, 1), [BIGINT], BIGINT, type="arrow")
    _bench(benchmark, con, f"SELECT sum(arrow_add_one(i::BIGINT)) FROM range({ARROW_N}) t(i)")


@pytest.mark.informational
def test_udf_arrow_double(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    con.create_function("arrow_scale", lambda x: pc.multiply(x, 1.5), [DOUBLE], DOUBLE, type="arrow")
    _bench(benchmark, con, f"SELECT sum(arrow_scale((i * 1.0)::DOUBLE)) FROM range({ARROW_N}) t(i)")


@pytest.mark.informational
def test_udf_arrow_null_inputs(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    # DEFAULT null handling on the vectorized path compacts the validity (selvec) before the call and reconstructs
    # the result vector after: this measures the selvec compaction/reconstruction cost.
    con.create_function("arrow_add_one", lambda x: pc.add(x, 1), [BIGINT], BIGINT, type="arrow")
    _bench(
        benchmark,
        con,
        "SELECT sum(arrow_add_one(v)) FROM "
        f"(SELECT CASE WHEN i % 2 = 0 THEN NULL ELSE i::BIGINT END AS v FROM range({ARROW_N}) t(i))",
    )
