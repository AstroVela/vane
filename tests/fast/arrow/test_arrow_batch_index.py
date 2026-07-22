# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pytest

import vane

pa = pytest.importorskip("pyarrow")


class TestArrowBatchIndex:
    def test_arrow_batch_index(self, duckdb_cursor):
        con = vane.connect()
        df = con.execute("SELECT * FROM range(10000000) t(i)").df()
        arrow_tbl = pa.Table.from_pandas(df)  # noqa: F841

        con.execute("CREATE TABLE tbl AS SELECT * FROM arrow_tbl")

        result = con.execute("SELECT * FROM tbl LIMIT 5").fetchall()
        assert [x[0] for x in result] == [0, 1, 2, 3, 4]

        result = con.execute("SELECT * FROM tbl LIMIT 5 OFFSET 777778").fetchall()
        assert [x[0] for x in result] == [777778, 777779, 777780, 777781, 777782]
