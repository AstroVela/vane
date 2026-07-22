# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import vane


class Test6315:
    def test_6315(self, duckdb_cursor):
        # segfault when accessing description after fetching rows
        c = vane.connect(":memory:")
        rv = c.execute("select * from sqlite_master where type = 'table'")
        rv.fetchall()
        desc = rv.description
        names = [x[0] for x in desc]
        assert names == ["type", "name", "tbl_name", "rootpage", "sql"]

        # description of relation
        rel = c.sql("select * from sqlite_master where type = 'table'")
        desc = rel.description
        names = [x[0] for x in desc]
        assert names == ["type", "name", "tbl_name", "rootpage", "sql"]

        rel.fetchall()
        desc = rel.description
        names = [x[0] for x in desc]
        assert names == ["type", "name", "tbl_name", "rootpage", "sql"]
