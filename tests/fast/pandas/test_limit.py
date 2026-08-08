# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pandas as pd

import vane


class TestLimitPandas:
    def test_limit_df(self, duckdb_cursor):
        df_in = pd.DataFrame(
            {
                "numbers": [1, 2, 3, 4, 5],
            }
        )
        limit_df = vane.limit(df_in, 2)
        assert len(limit_df.execute().fetchall()) == 2

    def test_aggregate_df(self, duckdb_cursor):
        df_in = pd.DataFrame(
            {
                "numbers": [1, 2, 2, 2],
            }
        )
        aggregate_df = vane.aggregate(df_in, "count(numbers)", "numbers").order("all")
        assert aggregate_df.execute().fetchall() == [(1,), (3,)]
