# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pytest

import vane

pa = pytest.importorskip("pyarrow")


class TestArrowBinaryView:
    def test_arrow_binary_view(self, duckdb_cursor):
        con = vane.connect()
        tab = pa.table({"x": pa.array([b"abc", b"thisisaverybigbinaryyaymorethanfifteen", None], pa.binary_view())})
        assert con.execute("FROM tab").fetchall() == [(b"abc",), (b"thisisaverybigbinaryyaymorethanfifteen",), (None,)]
        # By default we won't export a view
        assert not con.execute("FROM tab").to_arrow_table().equals(tab)
        # We do the binary view from 1.4 onwards
        con.execute("SET arrow_output_version = 1.4")
        assert con.execute("FROM tab").to_arrow_table().equals(tab)

        assert con.execute("FROM tab where x = 'thisisaverybigbinaryyaymorethanfifteen'").fetchall() == [
            (b"thisisaverybigbinaryyaymorethanfifteen",)
        ]
