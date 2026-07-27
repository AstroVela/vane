# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for relation write runner selection and lifecycle."""

from __future__ import annotations

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


def test_write_failure_forgets_per_database_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    import vane

    created_runners = []

    class FailingRayRunner:
        def __init__(self, runner_number):
            self.runner_number = runner_number
            self.calls = 0

        def run_write(self, relation):
            self.calls += 1
            raise RuntimeError(f"injected write failure from runner {self.runner_number}")

    def create_runner(*_args, **_kwargs):
        runner = FailingRayRunner(len(created_runners) + 1)
        created_runners.append(runner)
        return runner

    runners = types.ModuleType("duckdb.runners")
    runners.set_runner_ray = create_runner
    monkeypatch.setitem(sys.modules, "duckdb.runners", runners)

    connection = vane.connect()
    for runner_number in (1, 2):
        target = tmp_path / f"failed-{runner_number}.parquet"
        with pytest.raises(RuntimeError, match=f"injected write failure from runner {runner_number}"):
            connection.sql(f"select {runner_number} as x").write_parquet(str(target))

    assert [runner.calls for runner in created_runners] == [1, 1]


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
