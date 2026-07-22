# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

import duckdb


def test_repartition_random(duckdb_cursor):
    rel = duckdb_cursor.query("select i from range(10) t(i)")
    result = rel.repartition().fetchall()
    assert sorted(result) == [(i,) for i in range(10)]


def test_repartition_hash(duckdb_cursor):
    rel = duckdb_cursor.query("select i, i % 2 as k from range(10) t(i)")
    result = rel.repartition(4, "k").fetchall()
    assert sorted(result) == [(i, i % 2) for i in range(10)]


def test_repartition_kwargs(duckdb_cursor):
    rel = duckdb_cursor.query("select i from range(5) t(i)")
    result = rel.repartition("i", num_partitions=2).fetchall()
    assert sorted(result) == [(i,) for i in range(5)]


def test_repartition_invalid_partitions(duckdb_cursor):
    rel = duckdb_cursor.query("select i from range(5) t(i)")
    with pytest.raises(duckdb.InvalidInputException):
        rel.repartition(0)


def test_repartition_deduplicates_binding_names(duckdb_cursor):
    relation = duckdb_cursor.sql("SELECT 1 AS x, 2 AS x")

    assert relation.repartition(2).fetchall() == [(1, 2)]


def test_repartition_binds_qualified_partition_expression(duckdb_cursor):
    left = duckdb_cursor.sql("SELECT * FROM (VALUES (1), (2)) data(left_value)").set_alias("left_data")
    right = duckdb_cursor.sql("SELECT * FROM (VALUES (1, 10), (2, 20)) data(right_key, right_value)").set_alias(
        "right_data"
    )
    joined = left.join(right, "left_data.left_value = right_data.right_key")

    result = joined.repartition(2, "left_data.left_value").project("left_data.left_value, right_data.right_value")

    assert sorted(result.fetchall()) == [(1, 10), (2, 20)]


@pytest.mark.parametrize("exchange_method", ["repartition", "local_exchange"])
def test_exchange_join_direct_result_preserves_column_order(duckdb_cursor, exchange_method):
    left = duckdb_cursor.sql("SELECT * FROM (VALUES (1), (2)) data(left_value)").set_alias("left_data")
    right = duckdb_cursor.sql("SELECT * FROM (VALUES (1, 10), (2, 20)) data(right_key, right_value)").set_alias(
        "right_data"
    )
    joined = left.join(right, "left_data.left_value = right_data.right_key")

    result = getattr(joined, exchange_method)(2)

    assert result.columns == ["left_value", "right_key", "right_value"]
    assert sorted(result.fetchall()) == [(1, 1, 10), (2, 2, 20)]


@pytest.mark.parametrize("exchange_method", ["repartition", "local_exchange"])
def test_exchange_chain_executes_with_query_verification(duckdb_cursor, exchange_method):
    duckdb_cursor.execute("PRAGMA enable_verification")
    relation = getattr(duckdb_cursor.sql("SELECT 1 AS i"), exchange_method)(2).project("i + 1 AS value")

    assert relation.fetchall() == [(2,)]
    assert "PROJECTION" in relation.explain()


@pytest.mark.parametrize(
    ("exchange_method", "plan_node"),
    [("repartition", "REPARTITION"), ("local_exchange", "LOCAL_EXCHANGE")],
)
@pytest.mark.parametrize("operation", ["filter", "order", "distinct"])
def test_relational_operations_preserve_exchange(duckdb_cursor, exchange_method, plan_node, operation):
    relation = duckdb_cursor.sql("SELECT * FROM (VALUES (2), (1), (1)) data(i)")
    exchanged = getattr(relation, exchange_method)(2)

    if operation == "filter":
        result = exchanged.filter("i > 1")
        expected = [(2,)]
    elif operation == "order":
        result = exchanged.order("i")
        expected = [(1,), (1,), (2,)]
    else:
        result = exchanged.distinct()
        expected = [(1,), (2,)]

    assert plan_node in result.explain()
    rows = result.fetchall()
    if operation == "order":
        assert rows == expected
    else:
        assert sorted(rows) == expected


@pytest.mark.parametrize("exchange_method", ["repartition", "local_exchange"])
@pytest.mark.parametrize("outer_operation", ["project", "filter"])
def test_distinct_preserves_collation_through_exchange(duckdb_cursor, exchange_method, outer_operation):
    duckdb_cursor.execute("CREATE TABLE collated_values(value VARCHAR COLLATE nocase)")
    duckdb_cursor.execute("INSERT INTO collated_values VALUES ('A'), ('a')")
    relation = getattr(duckdb_cursor.table("collated_values"), exchange_method)(2).distinct()

    if outer_operation == "project":
        relation = relation.project("value")
    else:
        relation = relation.filter("value = 'a'")

    rows = relation.fetchall()
    assert len(rows) == 1
    assert rows[0][0].lower() == "a"


@pytest.mark.parametrize(
    ("exchange_method", "plan_node"),
    [("repartition", "REPARTITION"), ("local_exchange", "LOCAL_EXCHANGE")],
)
@pytest.mark.parametrize(
    ("aggregate", "groups", "expected"),
    [
        ("sum(i) AS total", None, [(4,)]),
        ("k, sum(i) AS total", "k", [(0, 2), (1, 2)]),
    ],
)
def test_aggregate_preserves_exchange(duckdb_cursor, exchange_method, plan_node, aggregate, groups, expected):
    relation = duckdb_cursor.sql("SELECT i, i % 2 AS k FROM (VALUES (1), (1), (2)) data(i)")
    exchanged = getattr(relation, exchange_method)(2)

    result = exchanged.aggregate(aggregate) if groups is None else exchanged.aggregate(aggregate, groups)

    assert plan_node in result.explain()
    assert sorted(result.fetchall()) == expected


@pytest.mark.parametrize(
    ("exchange_method", "plan_node"),
    [("repartition", "REPARTITION"), ("local_exchange", "LOCAL_EXCHANGE")],
)
@pytest.mark.parametrize("operation", ["project", "filter", "order"])
def test_exchange_over_join_survives_uncorrelated_subquery(duckdb_cursor, exchange_method, plan_node, operation):
    left = duckdb_cursor.sql("SELECT * FROM (VALUES (1), (2)) data(id)").set_alias("left_data")
    right = duckdb_cursor.sql("SELECT * FROM (VALUES (1, 10), (2, 20)) data(id, value)").set_alias("right_data")
    relation = getattr(left.join(right, "left_data.id = right_data.id"), exchange_method)(2)

    if operation == "project":
        result = relation.project("right_data.value + (SELECT 1) AS value")
        expected = [(11,), (21,)]
    elif operation == "filter":
        result = relation.filter("(SELECT true)")
        expected = [(1, 1, 10), (2, 2, 20)]
    else:
        result = relation.order("(SELECT 1)")
        expected = [(1, 1, 10), (2, 2, 20)]

    assert plan_node in result.explain()
    assert sorted(result.fetchall()) == expected


@pytest.mark.parametrize("exchange_method", ["repartition", "local_exchange"])
@pytest.mark.parametrize("operation", ["join", "cross", "union", "except", "intersect"])
def test_binary_relational_operations_fail_before_discarding_exchange(duckdb_cursor, exchange_method, operation):
    left = duckdb_cursor.sql("SELECT 1 AS i").set_alias("left_data")
    right = duckdb_cursor.sql("SELECT 1 AS i").set_alias("right_data")
    exchanged = getattr(left, exchange_method)(2)

    with pytest.raises(duckdb.NotImplementedException, match="would discard"):
        if operation == "join":
            exchanged.join(right, "left_data.i = right_data.i")
        elif operation == "cross":
            exchanged.cross(right)
        elif operation == "union":
            exchanged.union(right)
        elif operation == "except":
            exchanged.except_(right)
        else:
            exchanged.intersect(right)


@pytest.mark.parametrize("exchange_method", ["repartition", "local_exchange"])
def test_python_replacement_scan_fails_before_discarding_exchange(duckdb_cursor, exchange_method):
    relation_with_exchange = getattr(duckdb_cursor.sql("SELECT 1 AS i"), exchange_method)(2)  # noqa: F841

    with pytest.raises(duckdb.NotImplementedException, match="would discard"):
        duckdb_cursor.sql("SELECT * FROM relation_with_exchange")


@pytest.mark.parametrize("exchange_method", ["repartition", "local_exchange"])
def test_non_sql_exchange_has_no_sql_string(duckdb_cursor, exchange_method):
    relation = getattr(duckdb_cursor.sql("SELECT 1 AS i"), exchange_method)(2).project("i + 1 AS value")

    assert relation.sql_query() == ""


@pytest.mark.parametrize("exchange_method", ["repartition", "local_exchange"])
@pytest.mark.parametrize("operation", ["create", "create_view", "insert_into"])
def test_sql_terminal_operations_fail_before_discarding_exchange(duckdb_cursor, exchange_method, operation):
    relation = getattr(duckdb_cursor.sql("SELECT 1 AS i"), exchange_method)(2)
    target = f"exchange_{exchange_method}_{operation}"
    if operation == "insert_into":
        duckdb_cursor.execute(f"CREATE TABLE {target} (i INTEGER)")

    with pytest.raises(duckdb.NotImplementedException, match="discard the exchange"):
        getattr(relation, operation)(target)
