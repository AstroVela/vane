# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Expression-level scalar UDF tests for ``vane.func``."""

from __future__ import annotations

import gc
import os

import pytest


def test_expression_helpers_and_vane_func_are_public():
    import vane

    assert "func" in vane.__all__
    assert "col" in vane.__all__
    assert "lit" in vane.__all__
    assert "sql_expr" in vane.__all__

    assert callable(vane.func)
    assert callable(vane.func.batch)

    assert isinstance(vane.col("x"), vane.Expression)
    assert isinstance(vane.lit(1), vane.Expression)
    assert isinstance(vane.sql_expr("x + 1"), vane.Expression)

    con = vane.connect()
    out = con.sql("select 2::INTEGER as x").select((vane.col("x") + vane.lit(3)).alias("y"))

    assert out.fetchall() == [(5,)]


def test_vane_function_scalar_map_expression_local():
    import vane

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(3) t(i)")

    out = rel.select(vane.col("x"), add_one(vane.col("x")).alias("y"))

    assert out.fetchall() == [(0, 1), (1, 2), (2, 3)]


def test_vane_function_expression_pickles_duckdb_type_annotations():
    import vane

    def add_one(value):
        return value + 1

    add_one.__annotations__ = {
        "value": vane.sqltypes.INTEGER,
        "return": vane.sqltypes.INTEGER,
    }
    decorated = vane.func(return_dtype="INTEGER")(add_one)

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(3) t(i)")

    assert rel.select(decorated(vane.col("x")).alias("y")).fetchall() == [(1,), (2,), (3,)]


def test_vane_function_immediate_call_without_expression():
    import vane

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    assert add_one(41) == 42


def test_vane_function_rejects_missing_qualified_name_without_fallback():
    import vane

    class CallableObject:
        def __call__(self, value):
            return value

    with pytest.raises(vane.InvalidInputException, match="must expose a non-empty __qualname__"):
        vane.func(return_dtype="INTEGER")(CallableObject())


def test_vane_function_rejects_empty_explicit_name_without_defaulting():
    import vane

    with pytest.raises(vane.InvalidInputException, match="name must be a non-empty string"):

        @vane.func(return_dtype="INTEGER", name="")
        def identity(value):
            return value


def test_vane_function_binds_instance_method_for_eager_call():
    import vane
    from vane._expression_udf import VaneFunction

    class Scaler:
        factor = 2

        @vane.func(return_dtype="INTEGER")
        def scale(self, value):
            return self.factor * value

    scaler = Scaler()

    assert scaler.scale(3) == 6
    assert isinstance(Scaler.__dict__["scale"], VaneFunction)


def test_vane_function_bound_instance_method_preserves_wrapper_metadata():
    import vane

    class Scaler:
        @vane.func(return_dtype="INTEGER", name="scale_sql")
        def scale(self, value):
            """Scale one value."""
            return value * 2

    bound = Scaler().scale

    assert bound.return_dtype == "INTEGER"
    assert bound.sql_name == "scale_sql"
    assert bound.__name__ == "scale"
    assert bound.__qualname__.endswith("Scaler.scale")
    assert bound.__doc__ == "Scale one value."


def test_vane_function_binds_instance_method_for_expression_call():
    import vane

    class Scaler:
        def __init__(self, factor: int) -> None:
            self.factor = factor

        @vane.func(return_dtype="INTEGER")
        def scale(self, value):
            return self.factor * value

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(3) t(i)")

    out = rel.select(Scaler(2).scale(vane.col("x")).alias("y"))

    assert out.fetchall() == [(0,), (2,), (4,)]


def test_vane_function_bound_method_excludes_self_from_expression_arguments(monkeypatch):
    import vane
    import vane._expression_udf as expression_udf

    captured = {}
    sentinel = object()

    def capture_expression(fn, name, return_dtype, args, kwargs):
        captured.update(fn=fn, name=name, return_dtype=return_dtype, args=args, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr(expression_udf, "_build_map_expression", capture_expression)

    class Scaler:
        def __init__(self, factor: int) -> None:
            self.factor = factor

        @vane.func(return_dtype="INTEGER", name="scale_sql")
        def scale(self, value):
            return self.factor * value

    scaler = Scaler(2)
    value = vane.col("x")

    result = scaler.scale(value)

    assert result is sentinel
    assert captured["args"] == (value,)
    assert captured["kwargs"] == {}
    assert captured["fn"].__self__ is scaler


def test_vane_function_bound_instance_method_allows_literal_keyword_arguments():
    import vane

    class Scaler:
        def __init__(self, factor: int) -> None:
            self.factor = factor

        @vane.func(return_dtype="INTEGER")
        def scale(self, value, *, offset):
            return self.factor * value + offset

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(3) t(i)")

    out = rel.select(Scaler(2).scale(vane.col("x"), offset=1).alias("y"))

    assert out.fetchall() == [(1,), (3,), (5,)]


def test_vane_function_bound_instance_method_pickle_round_trip():
    import vane
    from vane.pickle import dumps, loads

    class Scaler:
        def __init__(self, factor: int) -> None:
            self.factor = factor

        @vane.func(return_dtype="INTEGER")
        def scale(self, value):
            return self.factor * value

    restored = loads(dumps(Scaler(2).scale))

    assert restored(3) == 6


def test_vane_function_bound_instance_method_expression_survives_wrapper_gc():
    import vane

    class Scaler:
        def __init__(self, factor: int) -> None:
            self.factor = factor

        @vane.func(return_dtype="INTEGER")
        def scale(self, value):
            return self.factor * value

    def make_expression():
        bound_wrapper = Scaler(2).scale
        return bound_wrapper(vane.col("x")).alias("y")

    expr = make_expression()
    gc.collect()

    con = vane.connect()
    out = con.sql("select i::INTEGER as x from range(3) t(i)").select(expr)

    assert out.fetchall() == [(0,), (2,), (4,)]


def test_vane_function_bound_instance_method_reports_pickle_failure():
    import vane

    pickle_error = "scaler instance cannot be pickled"

    class UnpickleableScaler:
        def __getstate__(self) -> None:
            raise TypeError(pickle_error)

        @vane.func(return_dtype="INTEGER")
        def scale(self, value):
            return value

    with pytest.raises(
        (TypeError, vane.InvalidInputException),
        match=r"scaler instance cannot be pickled|serializ|pickle",
    ):
        UnpickleableScaler().scale(vane.col("x"))


def test_vane_function_requires_return_dtype_for_expression():
    import vane

    @vane.func
    def add_one(value):
        return value + 1

    with pytest.raises(vane.InvalidInputException, match="return_dtype is required"):
        add_one(vane.col("x"))


def test_vane_function_rejects_expression_keyword_arguments():
    import vane

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    with pytest.raises(vane.InvalidInputException, match="keyword argument"):
        add_one(value=vane.col("x"))


def test_vane_function_allows_literal_keyword_arguments():
    import vane

    @vane.func(return_dtype="INTEGER")
    def add_n(value, *, n):
        return value + n

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(3) t(i)")

    out = rel.select(add_n(vane.col("x"), n=10).alias("y"))

    assert sorted(out.fetchall()) == [(10,), (11,), (12,)]


def test_vane_function_supports_multiple_and_nested_scalar_udfs():
    import vane

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    @vane.func(return_dtype="INTEGER")
    def times_two(value):
        return value * 2

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(3) t(i)")

    out = rel.select(
        add_one(vane.col("x")).alias("a"),
        times_two(vane.col("x")).alias("b"),
        times_two(add_one(vane.col("x"))).alias("nested"),
    )

    assert out.fetchall() == [(1, 0, 2), (2, 2, 4), (3, 4, 6)]


def test_expression_udf_callable_pickle_survives_live_function_gc():
    import vane

    def make_expression():
        @vane.func(return_dtype="INTEGER")
        def add_one(value):
            return value + 1

        return add_one(vane.col("x")).alias("y")

    expr = make_expression()
    gc.collect()

    con = vane.connect()
    out = con.sql("select i::INTEGER as x from range(3) t(i)").select(expr)

    assert out.fetchall() == [(1,), (2,), (3,)]


def test_vane_function_scalar_map_expression_ray_backend_explain():
    import vane

    old_runner = os.environ.get("VANE_RUNNER")
    try:
        vane.configure(runner="ray")

        @vane.func(return_dtype="INTEGER")
        def add_one(value):
            return value + 1

        con = vane.connect()
        rel = con.sql("select i::INTEGER as x from range(3) t(i)")
        plan = rel.select(add_one(vane.col("x")).alias("y")).explain()

        assert "execution_backend:" in plan
        assert "ray_task" in plan
    finally:
        if old_runner is None:
            os.environ.pop("VANE_RUNNER", None)
        else:
            os.environ["VANE_RUNNER"] = old_runner


def test_vane_function_scalar_map_uses_ray_runner_from_env(monkeypatch):
    import vane

    monkeypatch.setenv("VANE_RUNNER", "ray")

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    expr = add_one(vane.col("x")).alias("y")
    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(3) t(i)")
    plan = rel.select(expr).explain()

    assert "execution_backend:" in plan
    assert "ray_task" in plan
    assert "subprocess_task" not in plan
