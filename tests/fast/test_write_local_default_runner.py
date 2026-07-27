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

    runners = types.ModuleType("duckdb.runners")
    runners.set_runner_ray = lambda *_args, **_kwargs: FakeRayRunner()
    monkeypatch.setitem(sys.modules, "duckdb.runners", runners)

    target = tmp_path / "distributed.parquet"
    vane.connect().sql("select 1 as x").write_parquet(str(target))

    assert len(calls) == 1
    assert not target.exists()


def test_write_failure_invalidates_cached_and_native_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    import duckdb
    import duckdb.runners.ray.runner as ray_runner_module
    import vane

    created_runners = []

    class FailingRayRunner:
        def __init__(self, *_args):
            self.runner_number = len(created_runners) + 1
            self.calls = 0
            created_runners.append(self)

        def run_write(self, relation):
            self.calls += 1
            raise RuntimeError(f"injected write failure from runner {self.runner_number}")

        def close(self):
            pass

    vane_runners = duckdb.vane_runners_cpp
    vane_runners.teardown_runner()
    monkeypatch.setattr(ray_runner_module, "RayRunner", FailingRayRunner)

    connection = vane.connect()
    try:
        for runner_number in (1, 2):
            target = tmp_path / f"failed-{runner_number}.parquet"
            with pytest.raises(RuntimeError, match=f"injected write failure from runner {runner_number}"):
                connection.sql(f"select {runner_number} as x").write_parquet(str(target))

        assert [runner.calls for runner in created_runners] == [1, 1]
        assert vane_runners.get_runner() is None
    finally:
        vane_runners.teardown_runner()


def test_write_failure_cleanup_survives_closed_connection(tmp_path):
    target = tmp_path / "closed-connection.parquet"
    script = """
import os
import sys

import duckdb
import duckdb.runners.ray.runner as ray_runner_module
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


duckdb.vane_runners_cpp.teardown_runner()
ray_runner_module.RayRunner = ClosingFailingRayRunner
connection = vane.connect()

try:
    connection.sql("select 1 as x").write_parquet(sys.argv[1])
except RuntimeError as exc:
    assert str(exc) == "original write failure"
else:
    raise AssertionError("expected the injected write failure")

assert duckdb.vane_runners_cpp.get_runner() is None
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
