# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pandas as pd

import vane


class TestPandasUpdateList:
    def test_pandas_update_list(self, duckdb_cursor):
        duckdb_cursor = vane.connect(":memory:")
        duckdb_cursor.execute("create table t (l int[])")
        duckdb_cursor.execute("insert into t values ([1, 2]), ([3,4])")
        duckdb_cursor.execute("update t set l = [5, 6]")
        expected = pd.DataFrame({"l": [[5, 6], [5, 6]]})
        res = duckdb_cursor.execute("select * from t").fetchdf()
        pd.testing.assert_frame_equal(expected, res)
