# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import vane


class TestRAPIFunctions:
    def test_rapi_str_print(self, duckdb_cursor):
        res = duckdb_cursor.query("select 42::INT AS a, 84::BIGINT AS b")
        assert str(res) is not None
        res.show()

    def test_rapi_relation_sql_query(self):
        res = vane.table_function("range", [10])
        assert res.sql_query() == 'SELECT * FROM "range"(10)'
