# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import vane


class TestColumnOrder:
    def test_column_order(self, duckdb_cursor):
        to_execute = """
		CREATE OR REPLACE TABLE t1 AS (
			SELECT NULL AS col1,
			NULL::TIMESTAMPTZ AS timepoint,
			NULL::DATE AS date,
		);
		SELECT timepoint, date, col1 FROM t1;
		"""
        df = vane.execute(to_execute).fetchdf()
        cols = list(df.columns)
        assert cols == ["timepoint", "date", "col1"]
