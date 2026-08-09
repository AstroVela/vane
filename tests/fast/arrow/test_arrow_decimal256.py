# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

from decimal import Decimal

import pytest

import vane

pa = pytest.importorskip("pyarrow")


class TestArrowDecimal256:
    def test_decimal_256_throws(self, duckdb_cursor):
        with vane.connect() as conn:
            pa_decimal256 = pa.Table.from_pylist(  # noqa: F841
                [{"data": Decimal("100.00")} for _ in range(4)],
                pa.schema([("data", pa.decimal256(12, 4))]),
            )
            with pytest.raises(
                vane.NotImplementedException, match="Unsupported Internal Arrow Type for Decimal d:12,4,256"
            ):
                conn.execute("select * from pa_decimal256;").to_arrow_table().to_pylist()
