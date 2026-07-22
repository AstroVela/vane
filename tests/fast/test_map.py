# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pytest

import vane


class TestMap:
    def test_scalar_map_appends_typed_value_column(self, duckdb_cursor):
        def add_one(value):
            return value + 1

        relation = duckdb_cursor.sql("select i::INTEGER as x from range(3) t(i)")

        result = relation.map(
            add_one,
            return_type=vane.sqltypes.INTEGER,
            execution_backend="subprocess_task",
        )

        assert result.columns == ["x", "value"]
        assert result.types == [vane.sqltypes.INTEGER, vane.sqltypes.INTEGER]
        assert result.fetchall() == [(0, 1), (1, 2), (2, 3)]

    def test_scalar_map_passes_each_input_column(self, duckdb_cursor):
        def add_columns(left, right):
            return left + right

        relation = duckdb_cursor.sql("select i::INTEGER as left, (i * 10)::INTEGER as right from range(3) t(i)")

        result = relation.map(
            add_columns,
            return_type=vane.sqltypes.INTEGER,
            execution_backend="subprocess_task",
        )

        assert result.fetchall() == [(0, 0, 0), (1, 10, 11), (2, 20, 22)]

    def test_scalar_map_requires_return_type(self, duckdb_cursor):
        def add_one(value):
            return value + 1

        relation = duckdb_cursor.sql("select 1::INTEGER as x")

        with pytest.raises(TypeError):
            relation.map(add_one)

        with pytest.raises(TypeError):
            relation.map(add_one, return_type=None)

    def test_scalar_map_rejects_removed_dataframe_schema(self, duckdb_cursor):
        def add_one(value):
            return value + 1

        relation = duckdb_cursor.sql("select 1::INTEGER as x")

        with pytest.raises(TypeError):
            relation.map(add_one, schema={"x": vane.sqltypes.INTEGER})

    def test_scalar_map_requires_callable(self, duckdb_cursor):
        relation = duckdb_cursor.sql("select 1::INTEGER as x")

        with pytest.raises(TypeError):
            relation.map(42, return_type=vane.sqltypes.INTEGER)

    def test_map_batches_basic(self, duckdb_cursor):
        relation = duckdb_cursor.sql("select i from range(5) tbl(i)")

        def double_values(table):
            import pyarrow as pa

            values = table.column(0).to_pylist()
            return pa.table({"x": [value * 2 for value in values]})

        result = relation.map_batches(
            double_values,
            schema={"x": vane.sqltypes.INTEGER},
        )

        assert result.fetchall() == [(0,), (2,), (4,), (6,), (8,)]

    def test_create_table_function_map_batches(self, duckdb_cursor):
        def triple_values(table):
            import pyarrow as pa

            values = table.column(0).to_pylist()
            return pa.table({"x": [value * 3 for value in values]})

        duckdb_cursor.create_table_function(
            "map_batches_test",
            triple_values,
            schema={"x": vane.sqltypes.INTEGER},
        )
        result = duckdb_cursor.sql("select * from map_batches_test((select i from range(3) tbl(i)))")

        assert result.fetchall() == [(0,), (3,), (6,)]
