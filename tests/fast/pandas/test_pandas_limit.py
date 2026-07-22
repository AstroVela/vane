# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import vane


class TestPandasLimit:
    def test_pandas_limit(self, duckdb_cursor):
        con = vane.connect()
        df = con.execute("select * from range(10000000) tbl(i)").df()  # noqa: F841

        con.execute("SET threads=8")

        limit_df = con.execute("SELECT * FROM df WHERE i=334 OR i>9967864 LIMIT 5").df()
        assert list(limit_df["i"]) == [334, 9967865, 9967866, 9967867, 9967868]
