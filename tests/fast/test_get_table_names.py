# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pytest

import vane


class TestGetTableNames:
    def test_table_success(self, duckdb_cursor):
        conn = vane.connect()
        table_names = conn.get_table_names("SELECT * FROM my_table1, my_table2, my_table3")
        assert table_names == {"my_table2", "my_table3", "my_table1"}

    def test_table_fail(self, duckdb_cursor):
        conn = vane.connect()
        conn.close()
        with pytest.raises(vane.ConnectionException, match="Connection already closed"):
            conn.get_table_names("SELECT * FROM my_table1, my_table2, my_table3")

    def test_qualified_parameter_basic(self):
        conn = vane.connect()

        # Default (qualified=False)
        table_names = conn.get_table_names("SELECT * FROM test_table")
        assert table_names == {"test_table"}

        # Explicit qualified=False
        table_names = conn.get_table_names("SELECT * FROM test_table", qualified=False)
        assert table_names == {"test_table"}

    def test_qualified_parameter_schemas(self):
        conn = vane.connect()

        # Default (qualified=False)
        query = "SELECT * FROM test_schema.schema_table, main_table"
        table_names = conn.get_table_names(query)
        assert table_names == {"schema_table", "main_table"}

        # Test with qualified names
        table_names = conn.get_table_names(query, qualified=True)
        assert table_names == {"test_schema.schema_table", "main_table"}

    def test_qualified_parameter_catalogs(self):
        conn = vane.connect()

        # Test with qualified names including catalogs
        query = "SELECT * FROM catalog1.test_schema.catalog_table, regular_table"

        # Default (qualified=False)
        table_names = conn.get_table_names(query)
        assert table_names == {"catalog_table", "regular_table"}

        # With qualified=True
        table_names = conn.get_table_names(query, qualified=True)
        assert table_names == {"catalog1.test_schema.catalog_table", "regular_table"}

    def test_qualified_parameter_quoted_identifiers(self):
        conn = vane.connect()

        # Test with quoted identifiers
        query = 'SELECT * FROM "Schema.With.Dots"."Table.With.Dots", "Table With Spaces"'

        # Default (qualified=False)
        table_names = conn.get_table_names(query)
        assert table_names == {"Table.With.Dots", "Table With Spaces"}

        # With qualified=True
        table_names = conn.get_table_names(query, qualified=True)
        assert table_names == {'"Schema.With.Dots"."Table.With.Dots"', '"Table With Spaces"'}

    def test_expanded_views(self):
        conn = vane.connect()
        conn.execute("CREATE TABLE my_table(i INT)")
        conn.execute("CREATE VIEW v1 AS SELECT * FROM my_table")

        # Test that v1 expands to my_table
        query = "SELECT col_a FROM v1"

        # Default (qualified=False)
        table_names = conn.get_table_names(query)
        assert table_names == {"my_table"}

        # With qualified=True
        table_names = conn.get_table_names(query, qualified=True)
        assert table_names == {"my_table"}

    def test_expanded_views_with_schema(self):
        conn = vane.connect()
        conn.execute("CREATE SCHEMA my_schema")
        conn.execute("CREATE TABLE my_schema.my_table(i INT)")
        conn.execute("CREATE VIEW v1 AS SELECT * FROM my_schema.my_table")

        # Test that v1 expands to my_table
        query = "SELECT col_a FROM v1"

        # Default (qualified=False)
        table_names = conn.get_table_names(query)
        assert table_names == {"my_table"}

        # With qualified=True
        table_names = conn.get_table_names(query, qualified=True)
        assert table_names == {"my_schema.my_table"}

    def test_select_function(self):
        conn = vane.connect()
        query = "SELECT EXTRACT(second FROM i) FROM timestamps;"

        # Default (qualified=False)
        table_names = conn.get_table_names(query)
        assert table_names == {"timestamps"}

        # With qualified=True
        table_names = conn.get_table_names(query, qualified=True)
        assert table_names == {"timestamps"}
