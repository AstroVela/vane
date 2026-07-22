# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

import vane


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
    with pytest.raises(vane.InvalidInputException):
        rel.repartition(0)
