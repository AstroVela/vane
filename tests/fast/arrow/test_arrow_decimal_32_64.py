# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

from decimal import Decimal

import pytest

import vane

pa = pytest.importorskip("pyarrow")


class TestArrowDecimalTypes:
    def test_decimal_32(self, duckdb_cursor):
        duckdb_cursor = vane.connect()
        duckdb_cursor.execute("SET arrow_output_version = 1.5")
        decimal_32 = pa.Table.from_pylist(
            [
                {"data": Decimal("100.20")},
                {"data": Decimal("110.21")},
                {"data": Decimal("31.20")},
                {"data": Decimal("500.20")},
            ],
            pa.schema([("data", pa.decimal32(5, 2))]),
        )
        # Test scan
        assert duckdb_cursor.execute("FROM decimal_32").fetchall() == [
            (Decimal("100.20"),),
            (Decimal("110.21"),),
            (Decimal("31.20"),),
            (Decimal("500.20"),),
        ]
        # Test filter pushdown
        assert duckdb_cursor.execute("SELECT COUNT(*) FROM decimal_32 where data > 100 and data < 200 ").fetchall() == [
            (2,)
        ]

        # Test write
        arrow_table = duckdb_cursor.execute("FROM decimal_32").to_arrow_table()

        assert arrow_table.equals(decimal_32)

    def test_decimal_64(self, duckdb_cursor):
        duckdb_cursor = vane.connect()
        duckdb_cursor.execute("SET arrow_output_version = 1.5")
        decimal_64 = pa.Table.from_pylist(
            [
                {"data": Decimal("1000.231")},
                {"data": Decimal("1100.231")},
                {"data": Decimal("999999999999.231")},
                {"data": Decimal("500.20")},
            ],
            pa.schema([("data", pa.decimal64(16, 3))]),
        )

        # Test scan
        assert duckdb_cursor.execute("FROM decimal_64").fetchall() == [
            (Decimal("1000.231"),),
            (Decimal("1100.231"),),
            (Decimal("999999999999.231"),),
            (Decimal("500.200"),),
        ]

        # Test Filter pushdown
        assert duckdb_cursor.execute(
            "SELECT COUNT(*) FROM decimal_64 WHERE data > 1000 and data < 1200"
        ).fetchall() == [(2,)]

        # Test write
        arrow_table = duckdb_cursor.execute("FROM decimal_64").to_arrow_table()
        assert arrow_table.equals(decimal_64)
