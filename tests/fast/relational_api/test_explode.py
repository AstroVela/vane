# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

import duckdb

EXPECTED_COLUMNS = ["x", "a", "y"]
EXPECTED_ROWS = [(10, 20, 30), (10, 21, 30)]


def middle_list_relation(connection):
    return connection.sql("SELECT 10::INTEGER AS x, [20, 21]::INTEGER[] AS a, 30::INTEGER AS y")


def test_explode_serialized_query_matches_direct_binding(duckdb_cursor):
    exploded = middle_list_relation(duckdb_cursor).explode("a")
    serialized = duckdb_cursor.sql(exploded.sql_query())

    assert exploded.columns == EXPECTED_COLUMNS
    assert exploded.fetchall() == EXPECTED_ROWS
    assert serialized.columns == EXPECTED_COLUMNS
    assert serialized.fetchall() == EXPECTED_ROWS


def test_explode_preserves_column_order_through_limit(duckdb_cursor):
    limited = middle_list_relation(duckdb_cursor).explode("a").limit(10)

    assert limited.columns == EXPECTED_COLUMNS
    assert limited.fetchall() == EXPECTED_ROWS


def test_explode_preserves_positional_insert_values(duckdb_cursor):
    duckdb_cursor.execute("CREATE TABLE sink(x INTEGER, a INTEGER, y INTEGER)")

    middle_list_relation(duckdb_cursor).explode("a").insert_into("sink")

    assert duckdb_cursor.sql("SELECT * FROM sink ORDER BY x, a, y").fetchall() == EXPECTED_ROWS


@pytest.mark.parametrize("duplicate_name", ["a", "A"])
def test_explode_rejects_ambiguous_column_names(duckdb_cursor, duplicate_name):
    relation = duckdb_cursor.sql(f'SELECT [20, 21]::INTEGER[] AS a, [30, 31]::INTEGER[] AS "{duplicate_name}"')

    with pytest.raises(duckdb.BinderException, match='Ambiguous reference to column name "a"'):
        relation.explode("a")
