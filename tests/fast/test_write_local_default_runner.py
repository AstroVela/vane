# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for relation write runner selection and lifecycle."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest


def test_write_parquet_with_unset_runner_dispatches_ray(tmp_path, monkeypatch):
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

    target = tmp_path / "distributed.parquet"
    vane.connect().sql("select 1 as x").write_parquet(str(target))

    assert len(calls) == 1
    assert not target.exists()


def test_insert_into_with_ray_runner_dispatches_write_without_local_execution(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    import vane

    calls = []

    class FakeRayRunner:
        def run_write(self, relation):
            calls.append(relation)
            return {"ok": True}

    runners = types.ModuleType("vane.runners")
    runners.set_runner_ray = lambda *_args, **_kwargs: FakeRayRunner()
    monkeypatch.setitem(sys.modules, "vane.runners", runners)

    connection = vane.connect()
    connection.execute("CREATE TABLE target (value INTEGER)")
    connection.sql("SELECT 42 AS value").insert_into("target")

    assert len(calls) == 1
    assert connection.execute("SELECT count(*) FROM target").fetchone() == (0,)


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
import vane

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
