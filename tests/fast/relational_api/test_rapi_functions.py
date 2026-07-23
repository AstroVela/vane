import pytest

import duckdb


class TestRAPIFunctions:
    def test_rapi_str_print(self, duckdb_cursor):
        res = duckdb_cursor.query("select 42::INT AS a, 84::BIGINT AS b")
        assert str(res) is not None
        res.show()

    def test_rapi_relation_sql_query(self):
        res = duckdb.table_function("range", [10])
        assert res.sql_query() == 'SELECT * FROM "range"(10)'

    def test_rapi_relation_sql_query_after_catalog_change(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE sql_query_catalog_change(x INTEGER)")
        relation = duckdb_cursor.table("sql_query_catalog_change").limit(1).project("x")
        duckdb_cursor.execute("DROP TABLE sql_query_catalog_change")

        assert relation.sql_query() == ""
        with pytest.raises(duckdb.CatalogException):
            relation.fetchall()
