#!/usr/bin/env python
# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.


import pandas as pd

import vane


class TestUnicode:
    def test_unicode_pandas_scan(self, duckdb_cursor):
        con = vane.connect(database=":memory:", read_only=False)
        test_df = pd.DataFrame.from_dict({"i": [1, 2, 3], "j": ["a", "c", "ë"]})
        con.register("test_df_view", test_df)
        con.execute("SELECT i, j, LENGTH(j) FROM test_df_view").fetchall()
