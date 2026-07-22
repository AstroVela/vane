from collections import Counter
from itertools import product

# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.
import pytest

import duckdb
from duckdb import ColumnExpression

STANDARD_VECTOR_SIZE = 2048
CROSS_JOIN_BOUNDARY_CASES = (
    (1, 1),
    (2, 1),
    (1, 2),
    (3, 5),
    (10, 10),
    (STANDARD_VECTOR_SIZE, 2),
    (2, STANDARD_VECTOR_SIZE),
    (STANDARD_VECTOR_SIZE + 1, 2),
    (2, STANDARD_VECTOR_SIZE + 1),
    (2 * STANDARD_VECTOR_SIZE + 17, 3),
    (3, 2 * STANDARD_VECTOR_SIZE + 17),
)


@pytest.fixture
def con():
    conn = duckdb.connect()
    # Main relation
    conn.execute(
        """
        create table tbl_a as (SELECT * FROM (VALUES
            (1, 1),
            (2, 1),
            (3, 2)
        ) AS t(a, b))
    """
    )

    # Other relation
    conn.execute(
        """
        create table tbl_b as (SELECT * FROM (VALUES
            (1, 4),
            (3, 5),
        ) AS t(a, b))
    """
    )
    return conn


class TestRAPIJoins:
    def test_outer_join(self, con):
        a = con.table("tbl_a")
        b = con.table("tbl_b")
        expr = ColumnExpression("tbl_a.b") == ColumnExpression("tbl_b.a")
        rel = a.join(b, expr, "outer")
        res = rel.fetchall()
        assert res == [(1, 1, 1, 4), (2, 1, 1, 4), (3, 2, None, None), (None, None, 3, 5)]

    def test_inner_join(self, con):
        a = con.table("tbl_a")
        b = con.table("tbl_b")
        expr = ColumnExpression("tbl_a.b") == ColumnExpression("tbl_b.a")
        rel = a.join(b, expr, "inner")
        res = rel.fetchall()
        assert res == [(1, 1, 1, 4), (2, 1, 1, 4)]

    def test_anti_join(self, con):
        a = con.table("tbl_a")
        b = con.table("tbl_b")
        expr = ColumnExpression("tbl_a.b") == ColumnExpression("tbl_b.a")
        rel = a.join(b, expr, "anti")
        res = rel.fetchall()
        # Only output the row(s) from A where the condition is false
        assert res == [(3, 2)]

    def test_left_join(self, con):
        a = con.table("tbl_a")
        b = con.table("tbl_b")
        expr = ColumnExpression("tbl_a.b") == ColumnExpression("tbl_b.a")
        rel = a.join(b, expr, "left")
        res = rel.fetchall()
        assert res == [(1, 1, 1, 4), (2, 1, 1, 4), (3, 2, None, None)]

    def test_right_join(self, con):
        a = con.table("tbl_a")
        b = con.table("tbl_b")
        expr = ColumnExpression("tbl_a.b") == ColumnExpression("tbl_b.a")
        rel = a.join(b, expr, "right")
        res = rel.fetchall()
        assert res == [(1, 1, 1, 4), (2, 1, 1, 4), (None, None, 3, 5)]

    def test_semi_join(self, con):
        a = con.table("tbl_a")
        b = con.table("tbl_b")
        expr = ColumnExpression("tbl_a.b") == ColumnExpression("tbl_b.a")
        rel = a.join(b, expr, "semi")
        res = rel.fetchall()
        assert res == [(1, 1), (2, 1)]

    def test_cross_join(self, con):
        a = con.table("tbl_a")
        b = con.table("tbl_b")
        rel = a.cross(b)
        res = rel.fetchall()
        assert res == [(1, 1, 1, 4), (2, 1, 1, 4), (3, 2, 1, 4), (1, 1, 3, 5), (2, 1, 3, 5), (3, 2, 3, 5)]

    def test_cross_join_qualified_non_key_projection(self, con):
        result = con.table("tbl_a").cross(con.table("tbl_b")).project("tbl_a.b, tbl_b.b")

        assert result.order("1, 2").limit(1).fetchall() == [(1, 4)]

    def test_real_table_join_qualified_non_key_projection(self, con):
        relation = con.table("tbl_a").join(con.table("tbl_b"), "tbl_a.b = tbl_b.a")

        assert relation.project("tbl_a.a, tbl_b.b").order("1").fetchall() == [(1, 4), (2, 4)]

    def test_order_by_ordinal_preserves_relation_ordering(self, con):
        relation = con.sql("SELECT * FROM (VALUES (2, 'b'), (1, 'a')) data(id, label)")

        assert relation.order("1").limit(1).fetchone() == (1, "a")

    def test_order_by_ordinal_uses_visible_columns_after_using_join(self, con):
        left = con.sql("SELECT * FROM (VALUES (1, 'L1'), (2, 'L2')) data(id, value)").set_alias("l")
        right = con.sql("SELECT * FROM (VALUES (1, 'z'), (2, 'a')) data(id, sort_key)").set_alias("r")
        relation = left.join(right, "id")

        assert relation.order("3").limit(1).project("*").fetchall() == [(2, "L2", "a")]

    def test_filter_columns_expression(self, con):
        relation = con.sql("SELECT * FROM (VALUES (1, 2), (1, -1)) data(a, b)")

        assert relation.filter("COLUMNS(*) > 0").fetchall() == [(1, 2)]

    def test_order_by_subquery_uses_sql_binding(self, con):
        relation = con.sql("SELECT * FROM (VALUES (2), (1)) data(id)")

        assert relation.order("(SELECT -id)").fetchall() == [(2,), (1,)]

    @pytest.mark.parametrize(
        ("operation", "expected"),
        [
            ("direct", [(2, 1), (2, 1), (3, 1)]),
            ("filter", [(2, 1)]),
            ("ordered_limit", [(3, 1)]),
            ("distinct", [(2, 1), (2, 1), (3, 1)]),
        ],
    )
    def test_qualified_join_bindings_survive_unary_relations(self, con, operation, expected):
        con.execute("PRAGMA enable_verification")
        con.execute(
            "CREATE TEMP TABLE employees AS SELECT * FROM "
            "(VALUES (1, -1, 'manager'), (2, 1, 'alpha'), (2, 1, 'beta'), (3, 1, 'gamma')) "
            "data(emp_id, superior_emp_id, variant)"
        )
        employees = con.table("employees")
        left = employees.set_alias("emp1")
        right = employees.set_alias("emp2")
        relation = left.join(right, "emp1.superior_emp_id = emp2.emp_id")
        if operation == "filter":
            relation = relation.filter("emp1.variant = 'beta'")
        elif operation == "ordered_limit":
            relation = relation.order("emp1.emp_id DESC").limit(1)
        elif operation == "distinct":
            relation = relation.distinct()

        result = relation.project("emp1.emp_id, emp2.emp_id AS manager_id")

        assert sorted(result.fetchall()) == sorted(expected)

    @pytest.mark.parametrize(
        ("operation", "expected"),
        [
            ("project", [(1, 10), (1, 20), (2, 10), (2, 20)]),
            ("filter", [(2, 10), (2, 20)]),
            ("order", [(2, 20), (2, 10), (1, 20), (1, 10)]),
            ("aggregate", [(1, 30), (2, 30)]),
        ],
    )
    def test_cross_join_qualified_bindings_survive_unary_operations(self, con, operation, expected):
        left = con.sql("SELECT * FROM (VALUES (1), (2)) AS data(i)").set_alias("l")
        right = con.sql("SELECT * FROM (VALUES (10), (20)) AS data(i)").set_alias("r")
        relation = left.cross(right)

        if operation == "project":
            result = relation.project("l.i AS left_i, r.i AS right_i")
        elif operation == "filter":
            result = relation.filter("l.i = 2")
        elif operation == "order":
            result = relation.order("l.i DESC, r.i DESC")
        else:
            result = relation.aggregate("l.i AS left_i, sum(r.i) AS total", "l.i")

        rows = result.fetchall()
        if operation == "order":
            assert rows == expected
        else:
            assert sorted(rows) == expected

    @pytest.mark.parametrize(
        ("operation", "expected"),
        [
            ("project", [(2, 10), (2, 20)]),
            ("filter", [(2, 20)]),
            ("order", [(2, 20), (2, 10)]),
            ("aggregate", [(2, 30)]),
        ],
    )
    def test_cross_join_qualified_bindings_survive_inheriting_filter(self, con, operation, expected):
        left = con.sql("SELECT * FROM (VALUES (1), (2)) AS data(i)").set_alias("l")
        right = con.sql("SELECT * FROM (VALUES (10), (20)) AS data(i)").set_alias("r")
        filtered = left.cross(right).filter("l.i = 2")

        if operation == "project":
            result = filtered.project("l.i AS left_i, r.i AS right_i")
        elif operation == "filter":
            result = filtered.filter("r.i = 20")
        elif operation == "order":
            result = filtered.order("r.i DESC")
        else:
            result = filtered.aggregate("l.i AS left_i, sum(r.i) AS total", "l.i")

        rows = result.fetchall()
        if operation == "order":
            assert rows == expected
        else:
            assert sorted(rows) == expected

    def test_cross_join_limit_remains_before_aggregate(self, con):
        left = con.sql("SELECT 1 AS a FROM range(3)").set_alias("l")
        right = con.sql("SELECT 10 AS b").set_alias("r")

        result = left.cross(right).limit(1).aggregate("sum(a)")

        assert result.fetchall() == [(1,)]

    @pytest.mark.parametrize("boundary", ["distinct", "limit", "order"])
    def test_direct_bound_relation_does_not_emit_unbindable_sql(self, con, boundary):
        left = con.sql("SELECT * FROM (VALUES (1, 10)) data(id, left_value)").set_alias("l")
        right = con.sql("SELECT * FROM (VALUES (1, 20)) data(id, right_value)").set_alias("r")
        relation = left.join(right, "l.id = r.id")
        if boundary == "distinct":
            relation = relation.distinct()
        elif boundary == "limit":
            relation = relation.limit(1)
        else:
            relation = relation.order("r.right_value")
        relation = relation.project("r.right_value")

        assert relation.fetchall() == [(20,)]
        assert relation.sql_query() == ""

        with pytest.raises(duckdb.NotImplementedException, match="faithfully represented"):
            con.sql("SELECT * FROM relation")

    @pytest.mark.parametrize("operation", ["filter", "order", "project", "aggregate", "distinct", "limit", "alias"])
    def test_direct_bound_projection_remains_chainable(self, con, operation):
        left = con.sql("SELECT * FROM (VALUES (1, 10), (2, 20)) data(id, left_value)").set_alias("l")
        right = con.sql("SELECT * FROM (VALUES (1, 100), (2, 200)) data(id, right_value)").set_alias("r")
        relation = left.join(right, "l.id = r.id").distinct().project("r.right_value AS x")

        if operation == "filter":
            result = relation.filter("x > 100")
        elif operation == "order":
            result = relation.order("x DESC")
        elif operation == "project":
            result = relation.project("x + 1 AS y")
        elif operation == "aggregate":
            result = relation.aggregate("sum(x)")
        elif operation == "distinct":
            result = relation.distinct()
        elif operation == "limit":
            result = relation.limit(1)
        else:
            result = relation.set_alias("projected").filter("projected.x >= 100")

        rows = result.fetchall()
        assert result.sql_query() == ""
        if operation == "filter":
            assert rows == [(200,)]
        elif operation == "order":
            assert rows == [(200,), (100,)]
        elif operation == "project":
            assert sorted(rows) == [(101,), (201,)]
        elif operation == "aggregate":
            assert rows == [(300,)]
        elif operation == "limit":
            assert len(rows) == 1
            assert rows[0] in {(100,), (200,)}
        else:
            assert sorted(rows) == [(100,), (200,)]

    @pytest.mark.parametrize("boundary", ["distinct", "limit", "order"])
    def test_correlated_subquery_after_join_boundary_is_not_serialized(self, con, boundary):
        left = con.sql("SELECT * FROM (VALUES (1, 10)) data(id, left_value)").set_alias("l")
        right = con.sql("SELECT * FROM (VALUES (1, 20)) data(id, right_value)").set_alias("r")
        relation = left.join(right, "l.id = r.id")
        if boundary == "distinct":
            relation = relation.distinct()
        elif boundary == "limit":
            relation = relation.limit(1)
        else:
            relation = relation.order("r.right_value")
        relation = relation.project("(SELECT r.right_value) AS value")

        assert relation.fetchall() == [(20,)]
        assert relation.sql_query() == ""

        with pytest.raises(duckdb.NotImplementedException, match="faithfully represented"):
            con.sql("SELECT * FROM relation")

    def test_explicit_alias_restores_serializable_scope_for_correlated_subquery(self, con):
        left = con.sql("SELECT * FROM (VALUES (1, 10)) data(id, left_value)").set_alias("l")
        right = con.sql("SELECT * FROM (VALUES (1, 20)) data(id, right_value)").set_alias("r")
        relation = (
            left.join(right, "l.id = r.id")
            .distinct()
            .set_alias("joined")
            .project("(SELECT joined.right_value) AS value")
        )

        sql = relation.sql_query()
        assert sql
        assert con.sql(sql).fetchall() == relation.fetchall() == [(20,)]

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("(SELECT 42) AS value", [(42,), (42,)]),
            ("(SELECT r.value FROM (VALUES (42)) r(value)) AS value", [(42,), (42,)]),
            ("(SELECT (SELECT r.value) FROM (VALUES (42)) r(value)) AS value", [(42,), (42,)]),
        ],
    )
    def test_uncorrelated_subquery_after_join_boundary_remains_serializable(self, con, expression, expected):
        left = con.sql("SELECT * FROM (VALUES (1), (2)) data(id)").set_alias("l")
        right = con.sql("SELECT * FROM (VALUES (1), (2)) data(id)").set_alias("r")
        relation = left.join(right, "l.id = r.id").distinct().project(expression)

        sql = relation.sql_query()
        assert sql
        assert relation.fetchall() == expected
        assert con.sql(sql).fetchall() == expected

    @pytest.mark.parametrize("boundary", ["distinct", "limit", "order"])
    def test_struct_field_after_join_boundary_remains_serializable(self, con, boundary):
        left = con.sql("SELECT {'a': 7} AS payload, 1 AS id").set_alias("l")
        right = con.sql("SELECT 1 AS id").set_alias("r")
        relation = left.join(right, "l.id = r.id")
        if boundary == "distinct":
            relation = relation.distinct()
        elif boundary == "limit":
            relation = relation.limit(1)
        else:
            relation = relation.order("l.id")
        result = relation.project("payload.a AS value")

        sql = result.sql_query()
        assert sql
        assert result.fetchall() == [(7,)]
        assert con.sql(sql).fetchall() == [(7,)]
        assert con.sql("SELECT * FROM result").fetchall() == [(7,)]

    def test_lambda_field_after_join_boundary_remains_serializable(self, con):
        left = con.sql("SELECT [{'a': 7}] AS payloads, 1 AS id").set_alias("l")
        right = con.sql("SELECT 1 AS id").set_alias("r")
        result = (
            left.join(right, "l.id = r.id").distinct().project("list_transform(payloads, item -> item.a) AS values")
        )

        sql = result.sql_query()
        assert sql
        assert result.fetchall() == [([7],)]
        assert con.sql(sql).fetchall() == [([7],)]

    def test_deep_struct_field_after_join_boundary_remains_serializable(self, con):
        left = con.sql("SELECT {'a': {'b': {'c': {'d': 7}}}} AS payload, 1 AS id").set_alias("l")
        right = con.sql("SELECT 1 AS id").set_alias("r")
        result = left.join(right, "l.id = r.id").distinct().project("payload.a.b.c.d AS value")

        sql = result.sql_query()
        assert sql
        assert result.fetchall() == [(7,)]
        assert con.sql(sql).fetchall() == [(7,)]

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("(SELECT l.id FROM (VALUES (99)) l(other)) AS value", [(7,)]),
            ("(SELECT l.payload.a FROM (VALUES (99)) payload(a)) AS value", [(8,)]),
        ],
    )
    def test_nested_alias_without_matching_column_does_not_hide_join_binding(self, con, expression, expected):
        left = con.sql("SELECT 7 AS id, {'a': 8} AS payload, 1 AS join_key").set_alias("l")
        right = con.sql("SELECT 1 AS join_key").set_alias("r")
        result = left.join(right, "l.join_key = r.join_key").distinct().project(expression)

        assert result.fetchall() == expected
        assert result.sql_query() == ""
        with pytest.raises(duckdb.NotImplementedException, match="faithfully represented"):
            con.sql("SELECT * FROM result")

    def test_nested_alias_with_matching_column_hides_join_binding(self, con):
        left = con.sql("SELECT 1 AS id").set_alias("l")
        right = con.sql("SELECT 1 AS id, 100 AS value").set_alias("r")
        result = (
            left.join(right, "l.id = r.id").distinct().project("(SELECT r.value FROM (VALUES (42)) r(value)) AS value")
        )

        sql = result.sql_query()
        assert sql
        assert result.fetchall() == [(42,)]
        assert con.sql(sql).fetchall() == [(42,)]

    def test_nested_subquery_output_column_hides_join_binding(self, con):
        left = con.sql("SELECT 1 AS id").set_alias("l")
        right = con.sql("SELECT 1 AS id, 100 AS right_value").set_alias("r")
        result = (
            left.join(right, "l.id = r.id")
            .distinct()
            .project("(SELECT r.right_value FROM (SELECT 42 AS right_value) r) AS value")
        )

        sql = result.sql_query()
        assert sql
        assert result.fetchall() == [(42,)]
        assert con.sql(sql).fetchall() == [(42,)]

    def test_from_subquery_alias_is_not_visible_inside_its_own_query(self, con):
        left = con.sql("SELECT 7 AS id, 1 AS join_key").set_alias("l")
        right = con.sql("SELECT 1 AS join_key").set_alias("r")
        result = (
            left.join(right, "l.join_key = r.join_key")
            .distinct()
            .project("(SELECT id FROM (SELECT l.id) l(id)) AS value")
        )

        assert result.fetchall() == [(7,)]
        assert result.sql_query() == ""
        with pytest.raises(duckdb.NotImplementedException, match="faithfully represented"):
            con.sql("SELECT * FROM result")

    def test_table_function_alias_is_not_visible_inside_its_arguments(self, con):
        left = con.sql("SELECT 2 AS id, 1 AS join_key").set_alias("l")
        right = con.sql("SELECT 1 AS join_key").set_alias("r")
        result = (
            left.join(right, "l.join_key = r.join_key")
            .distinct()
            .project("(SELECT count(*) FROM range(l.id) l(id)) AS value")
        )

        assert result.fetchall() == [(2,)]
        assert result.sql_query() == ""
        with pytest.raises(duckdb.NotImplementedException, match="faithfully represented"):
            con.sql("SELECT * FROM result")

    def test_pivot_source_alias_is_not_visible_after_pivot(self, con):
        left = con.sql("SELECT 7 AS id, 1 AS join_key").set_alias("l")
        right = con.sql("SELECT 1 AS join_key").set_alias("r")
        result = (
            left.join(right, "l.join_key = r.join_key")
            .distinct()
            .project(
                "(SELECT l.id FROM (VALUES (1, 'a')) l(id, pivot_key) PIVOT (count(*) FOR pivot_key IN ('a'))) AS value"
            )
        )

        assert result.fetchall() == [(7,)]
        assert result.sql_query() == ""
        with pytest.raises(duckdb.NotImplementedException, match="faithfully represented"):
            con.sql("SELECT * FROM result")

    @pytest.mark.parametrize("expression", ["rowid_source.rowid AS row_id", "rowid AS row_id"])
    @pytest.mark.parametrize("boundary", ["distinct", "limit", "order"])
    def test_virtual_rowid_after_join_boundary_is_not_serialized(self, con, boundary, expression):
        con.execute("CREATE TABLE rowid_source(value INTEGER)")
        con.execute("INSERT INTO rowid_source VALUES (10), (20)")
        left = con.table("rowid_source")
        right = con.sql("SELECT 1 AS join_key").set_alias("r")
        relation = left.cross(right)
        if boundary == "distinct":
            relation = relation.distinct()
        elif boundary == "limit":
            relation = relation.limit(2)
        else:
            relation = relation.order("rowid_source.value")
        result = relation.project(expression)

        assert sorted(result.fetchall()) == [(0,), (1,)]
        assert result.sql_query() == ""
        with pytest.raises(duckdb.NotImplementedException, match="faithfully represented"):
            con.sql("SELECT * FROM result")

    @pytest.mark.parametrize("expression", ["rowid_source.rowid AS row_id", "rowid AS row_id"])
    @pytest.mark.parametrize("boundary", ["filter", "distinct", "limit", "order"])
    def test_virtual_rowid_after_single_source_boundary_is_not_serialized(self, con, boundary, expression):
        con.execute("CREATE TABLE rowid_source(value INTEGER)")
        con.execute("INSERT INTO rowid_source VALUES (10), (20)")
        relation = con.table("rowid_source")
        if boundary == "filter":
            relation = relation.filter("value > 0")
        elif boundary == "distinct":
            relation = relation.distinct()
        elif boundary == "limit":
            relation = relation.limit(2)
        else:
            relation = relation.order("rowid_source.value")
        result = relation.project(expression)

        assert sorted(result.fetchall()) == [(0,), (1,)]
        assert result.sql_query() == ""
        with pytest.raises(duckdb.NotImplementedException, match="faithfully represented"):
            con.sql("SELECT * FROM result")

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("read_parquet.filename AS value", None),
            ("filename AS value", None),
            ("read_parquet.file_index AS value", [(0,)]),
            ("file_index AS value", [(0,)]),
            ("read_parquet.file_row_number AS value", [(0,)]),
            ("file_row_number AS value", [(0,)]),
        ],
    )
    def test_table_function_virtual_column_after_join_boundary_is_not_serialized(
        self, con, tmp_path, expression, expected
    ):
        parquet_path = tmp_path / "virtual_columns.parquet"
        con.execute(f"COPY (SELECT 1 AS id) TO '{parquet_path}' (FORMAT PARQUET)")
        left = con.table_function("read_parquet", [str(parquet_path)])
        right = con.sql("SELECT 1 AS join_key").set_alias("r")
        result = left.cross(right).distinct().project(expression)
        if expected is None:
            expected = [(str(parquet_path),)]

        assert result.fetchall() == expected
        assert result.sql_query() == ""
        with pytest.raises(duckdb.NotImplementedException, match="faithfully represented"):
            con.sql("SELECT * FROM result")

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("read_parquet.filename AS value", None),
            ("filename AS value", None),
            ("read_parquet.file_index AS value", [(0,)]),
            ("file_index AS value", [(0,)]),
            ("read_parquet.file_row_number AS value", [(0,)]),
            ("file_row_number AS value", [(0,)]),
        ],
    )
    @pytest.mark.parametrize("boundary", ["filter", "distinct", "limit", "order"])
    def test_table_function_virtual_column_after_single_source_boundary_is_not_serialized(
        self, con, tmp_path, expression, expected, boundary
    ):
        parquet_path = tmp_path / "single_source_virtual_columns.parquet"
        con.execute(f"COPY (SELECT 1 AS id) TO '{parquet_path}' (FORMAT PARQUET)")
        relation = con.table_function("read_parquet", [str(parquet_path)])
        if boundary == "filter":
            relation = relation.filter("id > 0")
        elif boundary == "distinct":
            relation = relation.distinct()
        elif boundary == "limit":
            relation = relation.limit(1)
        else:
            relation = relation.order("id")
        result = relation.project(expression)
        if expected is None:
            expected = [(str(parquet_path),)]

        assert result.fetchall() == expected
        assert result.sql_query() == ""
        with pytest.raises(duckdb.NotImplementedException, match="faithfully represented"):
            con.sql("SELECT * FROM result")

    @pytest.mark.parametrize("boundary", ["distinct", "limit", "order"])
    def test_join_boundary_survives_uncorrelated_subquery(self, con, boundary):
        left = con.sql("SELECT * FROM (VALUES (1), (1), (2)) data(id)").set_alias("l")
        right = con.sql("SELECT * FROM (VALUES (1, 10), (2, 20)) data(id, value)").set_alias("r")
        relation = left.join(right, "l.id = r.id")
        if boundary == "distinct":
            relation = relation.distinct()
            expected = [(10,), (20,)]
        elif boundary == "limit":
            relation = relation.limit(1)
            expected = [(10,)]
        else:
            relation = relation.order("r.value DESC")
            expected = [(20,), (10,), (10,)]

        result = relation.project("r.value + (SELECT 0) AS value")

        rows = result.fetchall()
        if boundary == "distinct":
            assert sorted(rows) == expected
        else:
            assert rows == expected

    def test_distinct_join_can_feed_aggregate(self, con):
        left = con.sql("SELECT * FROM (VALUES (1), (1), (2)) data(id)").set_alias("l")
        right = con.sql("SELECT * FROM (VALUES (1, 10), (2, 20)) data(id, value)").set_alias("r")

        result = left.join(right, "l.id = r.id").distinct().aggregate("count(*)")

        assert result.fetchall() == [(2,)]

    def test_distinct_using_join_uses_visible_output_columns(self, con):
        left = con.sql("SELECT 1::INTEGER AS id, 'same' AS left_value").set_alias("l")
        right = con.sql("SELECT * FROM (VALUES ('1', 'same'), ('01', 'same')) data(id, right_value)").set_alias("r")
        relation = left.join(right, "id").distinct()

        expected = [(1, "same", "same")]
        assert relation.fetchall() == expected
        assert relation.limit(10).fetchall() == expected
        assert relation.aggregate("count(*)").fetchall() == [(1,)]

    def test_distinct_full_outer_using_join_uses_coalesced_key(self, con):
        left = con.sql("SELECT * FROM (VALUES (1, 'left'), (1, 'left')) data(id, left_value)").set_alias("l")
        right = con.sql("SELECT * FROM (VALUES (2, 'right'), (2, 'right')) data(id, right_value)").set_alias("r")
        relation = left.join(right, "id", "outer").distinct()

        expected = [(1, "left", None), (2, None, "right")]
        assert relation.order("id").limit(10).fetchall() == expected
        assert relation.aggregate("count(*)").fetchall() == [(2,)]

    @pytest.mark.parametrize("join_type", ["semi", "anti"])
    def test_distinct_semi_anti_join_uses_surviving_output(self, con, join_type):
        left = con.sql("SELECT * FROM (VALUES (1, 'one'), (2, 'two'), (2, 'two')) data(id, value)").set_alias("l")
        right = con.sql("SELECT * FROM (VALUES (1), (3)) data(id)").set_alias("r")

        result = left.join(right, "l.id = r.id", join_type).distinct().project("l.id, l.value")

        expected = [(1, "one")] if join_type == "semi" else [(2, "two")]
        assert result.fetchall() == expected

    @pytest.mark.parametrize("consumer", ["limit", "distinct"])
    def test_window_order_over_join_remains_chainable(self, con, consumer):
        left = con.sql("SELECT * FROM (VALUES (1), (1), (2)) data(id)").set_alias("l")
        right = con.sql("SELECT * FROM (VALUES (1, 10), (2, 20)) data(id, value)").set_alias("r")
        ordered = left.join(right, "l.id = r.id").order("row_number() OVER (ORDER BY l.id DESC)")

        if consumer == "limit":
            result = ordered.limit(1).project("l.id, r.value")
            assert result.fetchall() == [(2, 20)]
        else:
            result = ordered.distinct().project("l.id, r.value").order("1")
            assert result.fetchall() == [(1, 10), (2, 20)]

    def test_chained_order_reuses_duplicate_volatile_projection(self, con):
        left = con.sql("SELECT * FROM range(4) data(id)").set_alias("l")
        right = con.sql("SELECT 1 AS value").set_alias("r")
        relation = left.cross(right)

        con.execute("CREATE SEQUENCE direct_order_sequence START 1")
        relation.order("nextval('direct_order_sequence'), nextval('direct_order_sequence')").fetchall()

        con.execute("CREATE SEQUENCE chained_order_sequence START 1")
        relation.order("nextval('chained_order_sequence'), nextval('chained_order_sequence')").limit(4).fetchall()

        con.execute("CREATE SEQUENCE distinct_order_sequence START 1")
        relation.order("nextval('distinct_order_sequence'), nextval('distinct_order_sequence') + 0").limit(4).fetchall()

        assert con.sql("SELECT currval('direct_order_sequence')").fetchone() == (4,)
        assert con.sql("SELECT currval('chained_order_sequence')").fetchone() == (4,)
        assert con.sql("SELECT currval('distinct_order_sequence')").fetchone() == (8,)

    def test_explicit_alias_restores_serializable_scope_after_join_boundary(self, con):
        left = con.sql("SELECT * FROM (VALUES (1, 10)) data(id, left_value)").set_alias("l")
        right = con.sql("SELECT * FROM (VALUES (1, 20)) data(id, right_value)").set_alias("r")
        relation = left.join(right, "l.id = r.id").distinct().set_alias("joined").project("joined.right_value")

        sql = relation.sql_query()
        assert sql
        assert con.sql(sql).fetchall() == relation.fetchall() == [(20,)]

    def test_unqualified_projection_after_join_boundary_remains_serializable(self, con):
        left = con.sql("SELECT * FROM (VALUES (1, 10)) data(id, left_value)").set_alias("l")
        right = con.sql("SELECT * FROM (VALUES (1, 20)) data(id, right_value)").set_alias("r")
        relation = left.join(right, "l.id = r.id").distinct().project("right_value")

        sql = relation.sql_query()
        assert sql
        assert con.sql(sql).fetchall() == relation.fetchall() == [(20,)]

    @pytest.mark.parametrize("operation", ["create", "create_view", "insert_into"])
    def test_direct_bound_relation_rejects_sql_terminal_operations(self, con, operation):
        left = con.sql("SELECT * FROM (VALUES (1, 10)) data(id, left_value)").set_alias("l")
        right = con.sql("SELECT * FROM (VALUES (1, 20)) data(id, right_value)").set_alias("r")
        relation = left.join(right, "l.id = r.id").distinct().project("r.right_value")
        target = f"direct_bound_{operation}"
        if operation == "insert_into":
            con.execute(f"CREATE TABLE {target} (right_value INTEGER)")

        with pytest.raises(duckdb.NotImplementedException, match="faithfully represented"):
            getattr(relation, operation)(target)

    @pytest.mark.parametrize("threads", [1, 4])
    @pytest.mark.parametrize("left_count,right_count", CROSS_JOIN_BOUNDARY_CASES)
    def test_cross_join_vector_boundaries(self, con, threads, left_count, right_count):
        con.execute(f"SET threads={threads}")
        left = con.sql(f"SELECT range AS left_value FROM range({left_count})")
        right = con.sql(f"SELECT range AS right_value FROM range({right_count})")

        rows = left.cross(right).fetchall()
        expected_count = left_count * right_count
        expected_rows = Counter(product(range(left_count), range(right_count)))

        assert len(rows) == expected_count
        assert Counter(rows) == expected_rows
        assert left.cross(right).aggregate("count(*)").fetchone() == (expected_count,)
        assert con.execute(
            f"SELECT count(*) FROM range({left_count}) l CROSS JOIN range({right_count}) r"
        ).fetchone() == (expected_count,)

    @pytest.mark.parametrize("threads", [1, 4])
    def test_cross_join_preserves_null_multiset(self, con, threads):
        con.execute(f"SET threads={threads}")
        left = con.sql("SELECT * FROM (VALUES (1), (NULL), (1)) AS left_values(value)")
        right = con.sql("SELECT * FROM (VALUES (10), (NULL)) AS right_values(value)")

        rows = left.cross(right).fetchall()
        expected_rows = Counter(product((1, None, 1), (10, None)))

        assert len(rows) == 6
        assert Counter(rows) == expected_rows
        assert left.cross(right).aggregate("count(*)").fetchone() == (6,)
