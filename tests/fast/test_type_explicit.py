# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import vane
import vane.sqltypes as duckdb_types


class TestMap:
    def test_array_list_tuple_ambiguity(self):
        con = vane.connect()
        res = con.sql("SELECT $arg", params={"arg": (1, 2)}).fetchall()[0][0]
        assert res == [1, 2]

        # By using an explicit vane.Value with an array type, we should convert the input as an array
        # and get an array (tuple) back
        typ = vane.array_type(duckdb_types.BIGINT, 2)
        val = vane.Value((1, 2), typ)
        res = con.sql("SELECT $arg", params={"arg": val}).fetchall()[0][0]
        assert res == (1, 2)

        val = vane.Value([3, 4], typ)
        res = con.sql("SELECT $arg", params={"arg": val}).fetchall()[0][0]
        assert res == (3, 4)
