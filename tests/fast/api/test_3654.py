# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pandas as pd

import vane

try:
    import pyarrow as pa

    can_run = True
except Exception:
    can_run = False


class Test3654:
    def test_3654_pandas(self, duckdb_cursor):
        df1 = pd.DataFrame(
            {
                "id": [1, 1, 2],
            }
        )
        con = vane.connect()
        con.register("df1", df1)
        rel = con.view("df1")
        print(rel.execute().fetchall())
        assert rel.execute().fetchall() == [(1,), (1,), (2,)]

    def test_3654_arrow(self, duckdb_cursor):
        if not can_run:
            return

        df1 = pd.DataFrame(
            {
                "id": [1, 1, 2],
            }
        )
        table = pa.Table.from_pandas(df1)
        con = vane.connect()
        con.register("df1", table)
        rel = con.view("df1")
        print(rel.execute().fetchall())
        assert rel.execute().fetchall() == [(1,), (1,), (2,)]
