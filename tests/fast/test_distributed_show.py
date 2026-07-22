# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types

import pyarrow as pa

import vane


class _FakeRayRunner:
    def __init__(self, tables: list[pa.Table]) -> None:
        self.tables = tables
        self.calls: list[tuple[vane.DuckDBPyRelation, int | None]] = []

    def run_iter_tables(self, relation, results_buffer_size=None):
        self.calls.append((relation, results_buffer_size))
        yield from self.tables


def _install_fake_ray_runner(monkeypatch, runner: _FakeRayRunner) -> None:
    runners = types.ModuleType("vane.runners")
    runners.set_runner_ray = lambda *_args, **_kwargs: runner
    monkeypatch.setitem(sys.modules, "vane.runners", runners)


def test_relation_show_materializes_through_ray(monkeypatch, capsys):
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    runner = _FakeRayRunner(
        [
            pa.table({"value": [41]}),
            pa.table({"value": [42]}),
        ]
    )
    _install_fake_ray_runner(monkeypatch, runner)

    connection = vane.connect()
    relation = connection.sql("SELECT 999 AS value")

    relation.show()

    output = capsys.readouterr().out
    assert "41" in output
    assert "42" in output
    assert "999" not in output
    assert len(runner.calls) == 1
    limited_relation, results_buffer_size = runner.calls[0]
    assert "LIMIT 10000" in limited_relation.sql_query().upper()
    assert results_buffer_size == 1


def test_relation_show_uses_local_execution(monkeypatch, capsys):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    relation = vane.sql("SELECT 7 AS value")

    relation.show()

    output = capsys.readouterr().out
    assert "7" in output


def test_relation_show_preserves_duplicate_column_names(monkeypatch, capsys):
    monkeypatch.setenv("VANE_RUNNER", "")
    table = pa.Table.from_arrays([pa.array([10]), pa.array([20])], names=["a", "a"])
    runner = _FakeRayRunner([table])
    _install_fake_ray_runner(monkeypatch, runner)

    connection = vane.connect()
    connection.sql("SELECT 1 AS a, 2 AS a").show()

    output = capsys.readouterr().out
    assert output.count(" a ") == 2
    assert "10" in output
    assert "20" in output


def test_relation_show_handles_empty_distributed_result(monkeypatch, capsys):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    runner = _FakeRayRunner([])
    _install_fake_ray_runner(monkeypatch, runner)

    connection = vane.connect()
    connection.sql("SELECT NULL::VARCHAR AS name WHERE FALSE").show()

    output = capsys.readouterr().out
    assert "name" in output
    assert "0 rows" in output
