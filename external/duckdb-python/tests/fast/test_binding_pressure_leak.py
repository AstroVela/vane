"""Sustained-iteration leak guards for the binding object-pinning paths.

CodSpeed measures per-call cost and can't see a refcount imbalance in the object-pinning graph until it OOMs, so
this plain assertion test runs each pinning path N times and asserts RSS and object growth stay flat. Covers what
test_relation_dependency_leak.py does not: register/unregister, native + arrow UDF create/run/remove, executemany.
"""

import gc
import os

import pytest

import numpy as np
import pandas as pd

try:
    import pyarrow as pa

    can_arrow = True
except ImportError:
    can_arrow = False

from duckdb.sqltypes import BIGINT

psutil = pytest.importorskip("psutil")

ITERS = 100
ROWS = 100_000
_EM_ROWS = [(i, i * 1.5, f"s{i}") for i in range(5_000)]


def _rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (10**9)


def check_flat(fn, cursor, iters=ITERS, obj_slack=20_000):
    """Assert RSS and tracked-object count stay flat across `iters` calls of `fn`."""
    fn(cursor)  # warm one-time caches so they are not counted as growth
    gc.collect()
    start_rss = _rss_gb()
    start_obj = len(gc.get_objects())
    for _ in range(iters):
        fn(cursor)
    gc.collect()
    end_rss = _rss_gb()
    end_obj = len(gc.get_objects())
    # RSS ratio bound mirrors test_relation_dependency_leak.py (growth must stay well under 3x)...
    assert end_rss / 3 < start_rss, f"RSS grew {start_rss:.3f} -> {end_rss:.3f} GB over {iters} iters"
    # ...plus an object-count bound, which catches a Python-object pin that is too small to move RSS.
    assert end_obj - start_obj < obj_slack, f"tracked objects grew by {end_obj - start_obj} over {iters} iters"


# --------------------------------------------------------------------------- #
# Pinning paths (one full pin/unpin cycle per call).
# --------------------------------------------------------------------------- #


def register_unregister_arrow(cursor):
    tbl = pa.table({"a": pa.array(np.arange(ROWS), type=pa.int64())})
    cursor.register("t_reg", tbl)
    cursor.execute("SELECT sum(a) FROM t_reg").fetchall()
    cursor.unregister("t_reg")


def register_unregister_pandas(cursor):
    df = pd.DataFrame({"a": np.arange(ROWS)})
    cursor.register("t_reg", df)
    cursor.execute("SELECT sum(a) FROM t_reg").fetchall()
    cursor.unregister("t_reg")


def native_udf_cycle(cursor):
    cursor.create_function("f_leak", lambda x: x + 1, [BIGINT], BIGINT)
    cursor.execute("SELECT sum(f_leak(i::BIGINT)) FROM range(10000) t(i)").fetchall()
    cursor.remove_function("f_leak")


def arrow_udf_cycle(cursor):
    import pyarrow.compute as pc

    cursor.create_function("af_leak", lambda x: pc.add(x, 1), [BIGINT], BIGINT, type="arrow")
    cursor.execute("SELECT sum(af_leak(i::BIGINT)) FROM range(50000) t(i)").fetchall()
    cursor.remove_function("af_leak")


def executemany_cycle(cursor):
    cursor.execute("CREATE OR REPLACE TABLE t_em (a BIGINT, b DOUBLE, c VARCHAR)")
    cursor.executemany("INSERT INTO t_em VALUES (?, ?, ?)", _EM_ROWS)


class TestBindingPressureLeak:
    def test_register_unregister_arrow_leak(self, duckdb_cursor):
        if not can_arrow:
            pytest.skip("pyarrow not installed")
        check_flat(register_unregister_arrow, duckdb_cursor)

    def test_register_unregister_pandas_leak(self, duckdb_cursor):
        check_flat(register_unregister_pandas, duckdb_cursor)

    def test_native_udf_cycle_leak(self, duckdb_cursor):
        check_flat(native_udf_cycle, duckdb_cursor)

    def test_arrow_udf_cycle_leak(self, duckdb_cursor):
        if not can_arrow:
            pytest.skip("pyarrow not installed")
        check_flat(arrow_udf_cycle, duckdb_cursor)

    def test_executemany_leak(self, duckdb_cursor):
        check_flat(executemany_cycle, duckdb_cursor)
