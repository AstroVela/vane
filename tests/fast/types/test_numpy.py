# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import datetime

import numpy as np

import vane


class TestNumpyDatetime64:
    def test_numpy_datetime64(self, duckdb_cursor):
        duckdb_con = vane.connect()

        duckdb_con.execute("create table tbl(col TIMESTAMP)")
        duckdb_con.execute(
            "insert into tbl VALUES (CAST(? AS TIMESTAMP WITHOUT TIME ZONE))",
            parameters=[np.datetime64("2022-02-08T06:01:38.761310")],
        )
        assert [(datetime.datetime(2022, 2, 8, 6, 1, 38, 761310),)] == duckdb_con.execute(
            "select * from tbl"
        ).fetchall()

    def test_numpy_datetime_big(self):
        duckdb_con = vane.connect()

        duckdb_con.execute("create table test (date DATE)")
        duckdb_con.execute("INSERT INTO TEST VALUES ('2263-02-28')")

        res1 = duckdb_con.execute("select * from test").fetchnumpy()
        date_value = {"date": np.array(["2263-02-28"], dtype="datetime64[us]")}
        assert res1 == date_value

    def test_numpy_enum_conversion(self, duckdb_cursor):
        arr = np.array(["a", "b", "c"])
        rel = duckdb_cursor.sql("select * from arr")
        res = rel.fetchnumpy()["column0"]
        np.testing.assert_equal(res, arr)
