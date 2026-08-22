# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for relation write runner selection and lifecycle."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

_RELATION_MUTATIONS = [
    ("insert_into", "INSERT INTO"),
    ("insert_values", "INSERT"),
    ("update", "UPDATE"),
    ("delete", "DELETE"),
    ("ctas", "CTAS"),
]
_RELATION_MUTATION_IDS = ["insert-into", "insert-values", "update", "delete", "ctas"]


def _execute_relation_mutation(vane, connection, operation):
    if operation == "insert_into":
        connection.sql("SELECT 3 AS value").insert_into("target")
    elif operation == "insert_values":
        connection.table("target").insert([4])
    elif operation == "update":
        connection.table("target").update(
            {"value": vane.ConstantExpression(42)},
            condition=vane.ColumnExpression("value") == 1,
        )
    elif operation == "delete":
        connection.table("target").delete(condition=vane.ColumnExpression("value") == 2)
    elif operation == "ctas":
        connection.sql("SELECT 7 AS value").create("created_target")
    else:
        raise AssertionError(f"unknown relation mutation: {operation}")


def _new_mutation_connection(vane):
    connection = vane.connect()
    connection.execute("CREATE TABLE target (value INTEGER)")
    connection.execute("INSERT INTO target VALUES (1), (2)")
    return connection


@pytest.mark.parametrize(
    ("method_name", "extension"),
    [("write_csv", "csv"), ("write_parquet", "parquet")],
)
def test_format_convenience_write_with_unset_runner_uses_generic_copy_relation(
    tmp_path, monkeypatch, method_name, extension
):
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    import vane

    calls = []

    class FakeRayRunner:
        def run_write(self, relation):
            calls.append(relation)
            return {"ok": True}

    runners = types.ModuleType("vane.runners")
    runners.set_runner_ray = lambda *_args, **_kwargs: FakeRayRunner()
    monkeypatch.setitem(sys.modules, "vane.runners", runners)

    target = tmp_path / f"distributed.{extension}"
    relation = vane.connect().sql("select 1 as x")
    getattr(relation, method_name)(str(target))

    assert len(calls) == 1
    assert calls[0].type == "WRITE_FILE_RELATION"
    logical_plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_write_relation(
        calls[0],
        f"generic-copy-{extension}-convenience",
    )
    assert logical_plan is not None
    assert not target.exists()


def test_write_file_with_unset_runner_dispatches_generic_copy_relation(tmp_path, monkeypatch):
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    import vane

    captured = []

    class FakeRayRunner:
        def run_write(self, relation):
            captured.append(relation)
            return {"ok": True}

    runners = types.ModuleType("vane.runners")
    runners.set_runner_ray = lambda *_args, **_kwargs: FakeRayRunner()
    monkeypatch.setitem(sys.modules, "vane.runners", runners)

    target = tmp_path / "distributed.csv"
    connection = vane.connect()
    connection.sql("select 1 as x").write_file(str(target), format="csv")

    assert len(captured) == 1
    assert captured[0].type == "WRITE_FILE_RELATION"
    logical_plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_write_relation(
        captured[0],
        "generic-copy-relation",
    )
    assert logical_plan is not None
    assert not target.exists()


@pytest.mark.parametrize("runner_value", [None, "", "ray"])
def test_relation_mutations_dispatch_ray_without_local_execution(monkeypatch, runner_value):
    if runner_value is None:
        monkeypatch.delenv("VANE_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VANE_RUNNER", runner_value)
    import vane

    relation_types = []
    logical_plans = []

    class FakeRayRunner:
        def run_write(self, relation, **_kwargs):
            relation_types.append(relation.type)
            logical_plans.append(
                vane.ray_cxx.PyLogicalPlan.from_duckdb_write_relation(
                    relation,
                    f"relation-mutation-{len(logical_plans)}",
                )
            )
            return {"ok": True}

    runners = types.ModuleType("vane.runners")
    runners.set_runner_ray = lambda *_args, **_kwargs: FakeRayRunner()
    monkeypatch.setitem(sys.modules, "vane.runners", runners)

    connection = _new_mutation_connection(vane)
    for operation, _expected_name in _RELATION_MUTATIONS:
        _execute_relation_mutation(vane, connection, operation)

    assert relation_types == [
        "INSERT_RELATION",
        "INSERT_RELATION",
        "UPDATE_RELATION",
        "DELETE_RELATION",
        "CREATE_TABLE_RELATION",
    ]
    assert len(logical_plans) == len(_RELATION_MUTATIONS)
    assert connection.execute("SELECT * FROM target ORDER BY value").fetchall() == [(1,), (2,)]
    assert connection.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'created_target'"
    ).fetchone() == (0,)


def test_relation_mutations_run_with_explicit_local_fast(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    import vane

    connection = _new_mutation_connection(vane)
    for operation, _expected_name in _RELATION_MUTATIONS:
        _execute_relation_mutation(vane, connection, operation)

    assert connection.execute("SELECT * FROM target ORDER BY value").fetchall() == [(3,), (4,), (42,)]
    assert connection.execute("SELECT * FROM created_target").fetchall() == [(7,)]


def test_nested_ray_mutation_does_not_reuse_cached_local_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local")
    import vane

    local_calls = []
    ray_calls = []

    class FakeLocalRunner:
        def run_write(self, relation):
            local_calls.append(relation.type)
            monkeypatch.setenv("VANE_RUNNER", "ray")
            connection.sql("SELECT 42 AS value").insert_into("target")
            return {"ok": True}

    class FakeRayRunner:
        def run_write(self, relation, **_kwargs):
            ray_calls.append(relation.type)
            return {"ok": True}

    runners = types.ModuleType("vane.runners")
    runners.set_runner_local = lambda *_args, **_kwargs: FakeLocalRunner()
    runners.set_runner_ray = lambda *_args, **_kwargs: FakeRayRunner()
    monkeypatch.setitem(sys.modules, "vane.runners", runners)

    connection = vane.connect()
    connection.execute("CREATE TABLE target (value INTEGER)")
    connection.sql("SELECT 1 AS value").write_parquet(str(tmp_path / "nested.parquet"))

    assert local_calls == ["WRITE_FILE_RELATION"]
    assert ray_calls == ["INSERT_RELATION"]
    assert connection.execute("SELECT count(*) FROM target").fetchone() == (0,)


@pytest.mark.parametrize(("operation", "expected_name"), _RELATION_MUTATIONS, ids=_RELATION_MUTATION_IDS)
def test_ray_relation_mutations_reject_explicit_transactions(monkeypatch, operation, expected_name):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    import vane

    calls = []

    class FakeRayRunner:
        def run_write(self, relation, **_kwargs):
            calls.append(relation)
            return {"ok": True}

    runners = types.ModuleType("vane.runners")
    runners.set_runner_ray = lambda *_args, **_kwargs: FakeRayRunner()
    monkeypatch.setitem(sys.modules, "vane.runners", runners)

    connection = _new_mutation_connection(vane)
    connection.execute("BEGIN")
    try:
        with pytest.raises(
            vane.InvalidInputException,
            match=rf"Ray {expected_name} requires DuckDB auto-commit mode",
        ):
            _execute_relation_mutation(vane, connection, operation)

        assert calls == []
        assert connection.execute("SELECT * FROM target ORDER BY value").fetchall() == [(1,), (2,)]
        assert connection.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'created_target'"
        ).fetchone() == (0,)
    finally:
        connection.execute("ROLLBACK")


@pytest.mark.parametrize(("operation", "expected_name"), _RELATION_MUTATIONS, ids=_RELATION_MUTATION_IDS)
def test_relation_mutations_reject_local_fte_runner(monkeypatch, operation, expected_name):
    monkeypatch.setenv("VANE_RUNNER", "local")
    import vane

    connection = _new_mutation_connection(vane)
    with pytest.raises(
        vane.InvalidInputException,
        match=rf"{expected_name} requires VANE_RUNNER=ray or VANE_RUNNER=local-fast",
    ):
        _execute_relation_mutation(vane, connection, operation)

    assert connection.execute("SELECT * FROM target ORDER BY value").fetchall() == [(1,), (2,)]
    assert connection.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'created_target'"
    ).fetchone() == (0,)


@pytest.mark.parametrize(("operation", "_expected_name"), _RELATION_MUTATIONS, ids=_RELATION_MUTATION_IDS)
def test_ray_relation_mutation_failures_never_execute_locally(monkeypatch, operation, _expected_name):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    import vane

    class FailingRayRunner:
        def run_write(self, relation, **_kwargs):
            raise RuntimeError(f"injected distributed {operation} failure")

    runners = types.ModuleType("vane.runners")
    runners.set_runner_ray = lambda *_args, **_kwargs: FailingRayRunner()
    monkeypatch.setitem(sys.modules, "vane.runners", runners)

    connection = _new_mutation_connection(vane)
    with pytest.raises(RuntimeError, match=rf"injected distributed {operation} failure"):
        _execute_relation_mutation(vane, connection, operation)

    assert connection.execute("SELECT * FROM target ORDER BY value").fetchall() == [(1,), (2,)]
    assert connection.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'created_target'"
    ).fetchone() == (0,)


def test_write_failure_releases_cache_and_preserves_configured_native_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    import vane
    import vane.runners as runners_module
    import vane.runners.ray.runner as ray_runner_module

    created_runners = []
    set_runner_calls = 0

    class FailingRayRunner:
        def __init__(self, *args):
            self.runner_number = len(created_runners) + 1
            self.calls = 0
            self.close_calls = 0
            self.init_args = args
            created_runners.append(self)

        def run_write(self, relation):
            self.calls += 1
            raise RuntimeError(f"injected write failure from runner {self.runner_number}")

        def close(self):
            self.close_calls += 1

    vane_runners = vane
    vane_runners.teardown_runner()
    monkeypatch.setattr(ray_runner_module, "RayRunner", FailingRayRunner)
    configured_runner = runners_module.set_runner_ray(
        "ray://configured",
        max_task_backlog=17,
    )
    real_set_runner_ray = runners_module.set_runner_ray

    def tracking_set_runner_ray(*args, **kwargs):
        nonlocal set_runner_calls
        set_runner_calls += 1
        return real_set_runner_ray(*args, **kwargs)

    monkeypatch.setattr(runners_module, "set_runner_ray", tracking_set_runner_ray)

    connection = vane.connect()
    try:
        for attempt in (1, 2):
            target = tmp_path / f"failed-{attempt}.parquet"
            with pytest.raises(RuntimeError, match="injected write failure from runner 1"):
                connection.sql(f"select {attempt} as x").write_parquet(str(target))

        assert set_runner_calls == 2
        assert len(created_runners) == 1
        assert created_runners[0].calls == 2
        assert created_runners[0].close_calls == 0
        assert created_runners[0].init_args == ("ray://configured", 17)
        assert vane_runners.get_runner() is configured_runner
    finally:
        vane_runners.teardown_runner()


def test_write_failure_cleanup_survives_closed_connection(tmp_path):
    target = tmp_path / "closed-connection.parquet"
    script = """
import os
import sys

import vane
import vane.runners.ray.runner as ray_runner_module

os.environ["VANE_RUNNER"] = "ray"
connection = None


class ClosingFailingRayRunner:
    def __init__(self, *_args):
        pass

    def run_write(self, relation):
        connection.close()
        raise RuntimeError("original write failure")

    def close(self):
        pass


vane.teardown_runner()
ray_runner_module.RayRunner = ClosingFailingRayRunner
connection = vane.connect()

try:
    connection.sql("select 1 as x").write_parquet(sys.argv[1])
except RuntimeError as exc:
    assert str(exc) == "original write failure"
else:
    raise AssertionError("expected the injected write failure")

assert isinstance(vane.get_runner(), ClosingFailingRayRunner)
"""
    subprocess.run([sys.executable, "-c", script, str(target)], check=True, timeout=20)


def test_write_parquet_with_local_fast_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    import vane

    conn = vane.connect()
    target = tmp_path / "out.parquet"
    conn.sql("select 1 as x").write_parquet(str(target))

    assert conn.sql(f"select * from read_parquet('{target}')").fetchall() == [(1,)]


def test_write_csv_with_local_fast_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    import vane

    conn = vane.connect()
    target = tmp_path / "out.csv"
    conn.sql("select 1 as x").write_csv(str(target))

    assert conn.sql(f"select * from read_csv('{target}')").fetchall() == [(1,)]


def test_write_file_with_local_fast_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    import vane

    conn = vane.connect()
    target = tmp_path / "out.csv"
    conn.sql("select 1 as x union all select 2 as x").write_file(str(target), format="csv")

    assert conn.sql(f"select * from read_csv('{target}') order by 1").fetchall() == [(1,), (2,)]


def test_invalid_runner_env_raises_clear_error(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "rya")
    import vane

    conn = vane.connect()

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    rel = conn.sql("select 1::INTEGER as x")
    with pytest.raises(Exception, match="[Ii]nvalid runner"):
        rel.select(add_one(vane.col("x")).alias("y")).fetchall()
