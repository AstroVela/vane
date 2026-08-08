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


def create_binary_table(type):
    schema = pa.schema([("data", type)])
    inputs = [pa.array([b"foo", b"bar", b"baz"], type=type)]
    return pa.Table.from_arrays(inputs, schema=schema)


class TestArrowBinary:
    def test_binary_types(self, duckdb_cursor):
        if not can_run:
            return

        # Fixed Size Binary
        arrow_table = create_binary_table(pa.binary(3))
        rel = vane.from_arrow(arrow_table)
        res = rel.execute().fetchall()
        assert res == [(b"foo",), (b"bar",), (b"baz",)]

        # Normal Binary
        arrow_table = create_binary_table(pa.binary())
        rel = vane.from_arrow(arrow_table)
        res = rel.execute().fetchall()
        assert res == [(b"foo",), (b"bar",), (b"baz",)]

        # Large Binary
        arrow_table = create_binary_table(pa.large_binary())
        rel = vane.from_arrow(arrow_table)
        res = rel.execute().fetchall()
        assert res == [(b"foo",), (b"bar",), (b"baz",)]
