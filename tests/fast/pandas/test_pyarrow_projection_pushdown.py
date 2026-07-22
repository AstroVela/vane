# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pytest

import vane

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("pyarrow.dataset")
_ = pytest.importorskip("pandas", "2.0.0")


class TestArrowDFProjectionPushdown:
    def test_projection_pushdown_no_filter(self, duckdb_cursor):
        duckdb_conn = vane.connect()
        duckdb_conn.execute("CREATE TABLE test (a  INTEGER, b INTEGER, c INTEGER)")
        duckdb_conn.execute("INSERT INTO  test VALUES (1,1,1),(10,10,10),(100,10,100),(NULL,NULL,NULL)")
        duck_tbl = duckdb_conn.table("test")
        arrow_table = duck_tbl.df().convert_dtypes(dtype_backend="pyarrow")
        duckdb_conn.register("testarrowtable", arrow_table)
        assert duckdb_conn.execute("SELECT sum(a) FROM  testarrowtable").fetchall() == [(111,)]
