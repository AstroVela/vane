# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

from pathlib import Path

import vane

try:
    import pyarrow
    import pyarrow.parquet

    can_run = True
except Exception:
    can_run = False


class TestArrowReads:
    def test_multiple_queries_same_relation(self, duckdb_cursor):
        if not can_run:
            return
        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")
        userdata_parquet_table = pyarrow.parquet.read_table(parquet_filename)
        userdata_parquet_table.validate(full=True)
        rel = vane.from_arrow(userdata_parquet_table)
        assert rel.aggregate("(avg(salary))::INT").execute().fetchone()[0] == 149005
        assert rel.aggregate("(avg(salary))::INT").execute().fetchone()[0] == 149005
