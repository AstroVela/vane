# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

from collections import Counter
from itertools import product

import pytest

import vane
from vane import ColumnExpression

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
    conn = vane.connect()
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
