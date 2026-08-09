# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pytest

import vane


class TestRAPIFunctions:
    def test_rapi_str_print(self, duckdb_cursor):
        res = duckdb_cursor.query("select 42::INT AS a, 84::BIGINT AS b")
        assert str(res) is not None
        res.show()

    def test_rapi_relation_sql_query(self):
        res = vane.table_function("range", [10])
        assert res.sql_query() == 'SELECT * FROM "range"(10)'

    def test_rapi_relation_sql_query_after_catalog_change(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE sql_query_catalog_change(x INTEGER)")
        relation = duckdb_cursor.table("sql_query_catalog_change").limit(1).project("x")
        duckdb_cursor.execute("DROP TABLE sql_query_catalog_change")

        assert relation.sql_query() == ""
        with pytest.raises(vane.CatalogException):
            relation.fetchall()
