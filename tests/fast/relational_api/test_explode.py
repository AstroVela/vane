# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

import vane

EXPECTED_COLUMNS = ["x", "a", "y"]
EXPECTED_ROWS = [(10, 20, 30), (10, 21, 30)]


def middle_collection_relation(connection, collection_type="INTEGER[]"):
    return connection.sql(f"SELECT 10::INTEGER AS x, [20, 21]::{collection_type} AS a, 30::INTEGER AS y")


def duplicate_non_target_relation(connection, duplicate_name="x"):
    return connection.sql(
        f'SELECT [20, 21]::INTEGER[] AS a, 10::INTEGER AS x, 30::INTEGER AS "{duplicate_name}"'
    ).explode("a")


@pytest.mark.parametrize("collection_type", ["INTEGER[]", "INTEGER[2]"])
def test_explode_serialized_query_matches_direct_binding(duckdb_cursor, collection_type):
    exploded = middle_collection_relation(duckdb_cursor, collection_type).explode("a")
    serialized = duckdb_cursor.sql(exploded.sql_query())

    assert exploded.columns == EXPECTED_COLUMNS
    assert exploded.fetchall() == EXPECTED_ROWS
    assert serialized.columns == EXPECTED_COLUMNS
    assert serialized.types == exploded.types
    assert serialized.fetchall() == EXPECTED_ROWS


@pytest.mark.parametrize("collection_type", ["INTEGER[]", "INTEGER[2]"])
def test_explode_unique_target_name_is_case_insensitive(duckdb_cursor, collection_type):
    exploded = middle_collection_relation(duckdb_cursor, collection_type).explode("A")
    serialized = duckdb_cursor.sql(exploded.sql_query())

    assert exploded.columns == EXPECTED_COLUMNS
    assert exploded.fetchall() == EXPECTED_ROWS
    assert serialized.columns == EXPECTED_COLUMNS
    assert serialized.types == exploded.types
    assert serialized.fetchall() == EXPECTED_ROWS


def test_explode_case_insensitive_target_requires_unique_match(duckdb_cursor):
    relation = duckdb_cursor.sql('SELECT [20, 21]::INTEGER[] AS target, [30, 31]::INTEGER[] AS "TARGET"')

    with pytest.raises(vane.BinderException, match="Ambiguous reference to column"):
        relation.explode("Target")


@pytest.mark.parametrize("duplicate_name", ["x", "X"])
def test_explode_serialized_query_preserves_duplicate_non_target_names(duckdb_cursor, duplicate_name):
    exploded = duplicate_non_target_relation(duckdb_cursor, duplicate_name)
    serialized = duckdb_cursor.sql(exploded.sql_query())
    expected_rows = [(20, 10, 30), (21, 10, 30)]

    assert exploded.columns == ["a", "x", duplicate_name]
    assert serialized.types == exploded.types
    assert exploded.fetchall() == expected_rows
    assert serialized.columns == ["a", "x", duplicate_name]
    assert serialized.fetchall() == expected_rows


def test_explode_duplicate_non_target_names_survive_filter_serialization(duckdb_cursor):
    filtered = duplicate_non_target_relation(duckdb_cursor).filter("a > 20")
    serialized = duckdb_cursor.sql(filtered.sql_query())

    assert filtered.columns == ["a", "x", "x"]
    assert serialized.columns == filtered.columns
    assert serialized.types == filtered.types
    assert serialized.fetchall() == filtered.fetchall() == [(21, 10, 30)]


def test_explode_duplicate_non_target_names_survive_union_serialization(duckdb_cursor):
    exploded = duplicate_non_target_relation(duckdb_cursor)
    other = duckdb_cursor.sql("SELECT 22::INTEGER AS a, 11::INTEGER AS x, 31::INTEGER AS x")
    unioned = exploded.union(other)
    serialized = duckdb_cursor.sql(unioned.sql_query())

    assert unioned.columns == ["a", "x", "x"]
    assert serialized.columns == unioned.columns
    assert serialized.types == unioned.types
    assert serialized.fetchall() == unioned.fetchall() == [(20, 10, 30), (21, 10, 30), (22, 11, 31)]


@pytest.mark.parametrize("duplicate_name", ["x", "X"])
def test_explode_serialization_uses_deduplicated_target_binding(duckdb_cursor, duplicate_name):
    exploded = duckdb_cursor.sql(
        f'SELECT 10::INTEGER AS x, 20::INTEGER AS "{duplicate_name}", [30, 31]::INTEGER[] AS x_1'
    ).explode("x_1")
    serialized = duckdb_cursor.sql(exploded.sql_query())

    assert exploded.columns == ["x", duplicate_name, "x_1"]
    assert serialized.columns == exploded.columns
    assert serialized.types == exploded.types
    assert serialized.fetchall() == exploded.fetchall() == [(10, 20, 30), (10, 20, 31)]


def test_explode_preserves_column_order_through_limit(duckdb_cursor):
    limited = middle_collection_relation(duckdb_cursor).explode("a").limit(10)

    assert limited.columns == EXPECTED_COLUMNS
    assert limited.fetchall() == EXPECTED_ROWS


def test_explode_preserves_positional_insert_values(duckdb_cursor):
    duckdb_cursor.execute("CREATE TABLE sink(x INTEGER, a INTEGER, y INTEGER)")

    middle_collection_relation(duckdb_cursor).explode("a").insert_into("sink")

    assert duckdb_cursor.sql("SELECT * FROM sink ORDER BY x, a, y").fetchall() == EXPECTED_ROWS


@pytest.mark.parametrize(
    ("duplicate_name", "target", "expected_rows"),
    [
        ("a", "a", [(20, [30, 31]), (21, [30, 31])]),
        ("A", "a", [(20, [30, 31]), (21, [30, 31])]),
        ("A", "A", [([20, 21], 30), ([20, 21], 31)]),
    ],
)
def test_explode_serialization_preserves_duplicate_target_resolution(
    duckdb_cursor, duplicate_name, target, expected_rows
):
    relation = duckdb_cursor.sql(f'SELECT [20, 21]::INTEGER[] AS a, [30, 31]::INTEGER[] AS "{duplicate_name}"')
    exploded = relation.explode(target)
    serialized = duckdb_cursor.sql(exploded.sql_query())

    assert serialized.columns == exploded.columns == ["a", duplicate_name]
    assert serialized.types == exploded.types
    assert serialized.fetchall() == exploded.fetchall() == expected_rows
