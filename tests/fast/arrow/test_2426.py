# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pytest

import vane

pytest.importorskip("pyarrow")

try:
    can_run = True
except Exception:
    can_run = False


class Test2426:
    def test_2426(self, duckdb_cursor):
        if not can_run:
            return

        con = vane.connect()
        con.execute("Create Table test (a integer)")

        for i in range(1024):
            for _j in range(2):
                con.execute("Insert Into test values ('" + str(i) + "')")
        con.execute("Insert Into test values ('5000')")
        con.execute("Insert Into test values ('6000')")
        sql = """
        SELECT  a, COUNT(*) AS repetitions
        FROM    test
        GROUP BY a
        """

        result_df = con.execute(sql).df()

        arrow_table = con.execute(sql).to_arrow_table()

        arrow_df = arrow_table.to_pandas()
        assert result_df["repetitions"].sum() == arrow_df["repetitions"].sum()
