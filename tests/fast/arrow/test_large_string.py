# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import vane

try:
    import pyarrow as pa

    can_run = True
except Exception:
    can_run = False


class TestArrowLargeString:
    def test_large_string_type(self, duckdb_cursor):
        if not can_run:
            return

        schema = pa.schema([("data", pa.large_string())])
        inputs = [pa.array(["foo", "baaaar", "b"], type=pa.large_string())]
        arrow_table = pa.Table.from_arrays(inputs, schema=schema)

        rel = vane.from_arrow(arrow_table)
        res = rel.execute().fetchall()
        assert res == [("foo",), ("baaaar",), ("b",)]
