# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for flat_map UDF (one-to-many row expansion via INOUT_FUNCTION)."""

import vane


def test_flat_map_basic():
    """Basic flat_map: one row -> multiple rows."""
    con = vane.connect()

    def expand(row):
        val = row["x"]
        for i in range(val):
            yield {"x": val, "i": i}

    rel = con.sql("SELECT 3 AS x")
    result = rel.flat_map(
        expand,
        schema={"x": con.type("INTEGER"), "i": con.type("INTEGER")},
        execution_backend="subprocess_task",
    ).fetchall()

    assert len(result) == 3
    assert result == [(3, 0), (3, 1), (3, 2)]


def test_flat_map_multiple_rows():
    """flat_map over multiple input rows."""
    con = vane.connect()

    def split_words(row):
        text = row["text"]
        for word in text.split():
            yield {"word": word}

    rel = con.sql("SELECT * FROM (VALUES ('hello world'), ('foo bar baz')) AS t(text)")
    result = rel.flat_map(
        split_words,
        schema={"word": con.type("VARCHAR")},
        execution_backend="subprocess_task",
    ).fetchall()

    assert len(result) == 5
    words = [r[0] for r in result]
    assert words == ["hello", "world", "foo", "bar", "baz"]


def test_flat_map_generator():
    """flat_map with a generator function."""
    con = vane.connect()

    def repeat(row):
        n = row["n"]
        s = row["s"]
        for _ in range(n):
            yield {"s": s}

    rel = con.sql("SELECT * FROM (VALUES (2, 'a'), (3, 'b')) AS t(n, s)")
    result = rel.flat_map(
        repeat,
        schema={"s": con.type("VARCHAR")},
        execution_backend="subprocess_task",
    ).fetchall()

    assert len(result) == 5
    assert result == [("a",), ("a",), ("b",), ("b",), ("b",)]


def test_flat_map_single_dict():
    """flat_map returning a single dict (not a generator)."""
    con = vane.connect()

    def double_value(row):
        return {"x": row["x"] * 2}

    rel = con.sql("SELECT 5 AS x")
    result = rel.flat_map(
        double_value,
        schema={"x": con.type("INTEGER")},
        execution_backend="subprocess_task",
    ).fetchall()

    assert len(result) == 1
    assert result == [(10,)]


def test_flat_map_none_skip():
    """flat_map returning None should produce zero rows for that input."""
    con = vane.connect()

    def maybe_expand(row):
        if row["x"] > 2:
            yield {"x": row["x"]}
        # Returning None implicitly for x <= 2

    rel = con.sql("SELECT * FROM (VALUES (1), (2), (3), (4)) AS t(x)")
    result = rel.flat_map(
        maybe_expand,
        schema={"x": con.type("INTEGER")},
        execution_backend="subprocess_task",
    ).fetchall()

    assert len(result) == 2
    assert result == [(3,), (4,)]


def test_flat_map_empty_input():
    """flat_map with empty input should produce empty output."""
    con = vane.connect()

    def expand(row):
        yield {"x": row["x"]}

    rel = con.sql("SELECT 1 AS x WHERE false")
    result = rel.flat_map(
        expand,
        schema={"x": con.type("INTEGER")},
        execution_backend="subprocess_task",
    ).fetchall()

    assert len(result) == 0


def test_flat_map_passthrough_columns():
    """flat_map should support returning columns that weren't in the input."""
    con = vane.connect()

    def tag_and_split(row):
        path = row["path"]
        for i, part in enumerate(path.split("/")):
            if part:
                yield {"path": path, "part": part, "depth": i}

    rel = con.sql("SELECT '/a/b/c' AS path")
    result = rel.flat_map(
        tag_and_split,
        schema={
            "path": con.type("VARCHAR"),
            "part": con.type("VARCHAR"),
            "depth": con.type("INTEGER"),
        },
        execution_backend="subprocess_task",
    ).fetchall()

    assert len(result) == 3  # 'a', 'b', 'c' (empty string from leading / is filtered)
    parts = [r[1] for r in result]
    assert parts == ["a", "b", "c"]


def test_flat_map_streaming_breaker_subprocess_task():
    """Streaming flat_map should support one-to-many table output."""
    con = vane.connect()

    def expand(row):
        yield {"y": row["x"]}
        yield {"y": row["x"] + 10}

    result = (
        con.sql("SELECT 1::INTEGER AS x")
        .flat_map(
            expand,
            schema={"y": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
            batch_size=1,
            output_batch_size=1,
            streaming_breaker=True,
        )
        .fetchall()
    )

    assert result == [(1,), (11,)]
