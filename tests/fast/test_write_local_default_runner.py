# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests: explicit local-fast writes must not require a runner."""

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

    runners = types.ModuleType("vane.runners")
    runners.set_runner_ray = lambda *_args, **_kwargs: FakeRayRunner()
    monkeypatch.setitem(sys.modules, "vane.runners", runners)

    target = tmp_path / "distributed.parquet"
    vane.connect().sql("select 1 as x").write_parquet(str(target))

    assert len(calls) == 1
    assert not target.exists()


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
