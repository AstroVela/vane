# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import vane
from tests.fast.test_distributed_result_consumers import _install_fake_ray_runner, _TransportedPlanRunner
from vane.runners.copy_outcome import CopyResultUnavailableError


class _SQLRunner(_TransportedPlanRunner):
    def __init__(self):
        super().__init__()
        self.writes = []
        self.outcome = {
            "copy_operation_id": "sql-copy-operation",
            "rows_copied": 13,
            "copy_cleanup_warnings": ["cleanup pending"],
        }

    def run_write(self, relation):
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_write_relation(relation, f"copy-{len(self.writes)}")
        self.writes.append(pickle.loads(pickle.dumps(plan)))
        return self.outcome


@pytest.mark.parametrize("initial", ["local-fast", "ray"])
@pytest.mark.parametrize("later", ["local-fast", "ray", "invalid"])
def test_connection_and_derived_relations_keep_runner_policy(monkeypatch, initial, later):
    runner = _SQLRunner()
    factory_calls = _install_fake_ray_runner(monkeypatch, runner)
    monkeypatch.setenv("VANE_RUNNER", initial)
    with vane.connect() as connection:
        relation = connection.sql("SELECT ?::BIGINT AS value", params=[7])
        assert factory_calls == []
        monkeypatch.setenv("VANE_RUNNER", later)
        with connection.cursor() as cursor:
            assert cursor.sql("SELECT 1")._get_runner_type() == initial
            assert cursor.execute("SELECT ?::BIGINT AS value", [8]).fetchall() == [(8,)]
        assert relation.filter("value > 0").project("value + 1 AS value").fetchall() == [(8,)]
        assert connection.execute("SELECT 9::BIGINT AS value").fetchall() == [(9,)]
        assert connection.sql("SELECT 1")._get_runner_type() == initial
        assert len(runner.plans) == (3 if initial == "ray" else 0)
        assert os.environ["VANE_RUNNER"] == later
        if later == "invalid":
            with pytest.raises(vane.InvalidInputException, match="VANE_RUNNER"):
                vane.connect()
        else:
            with vane.connect() as fresh:
                assert fresh.sql("SELECT 1")._get_runner_type() == later


@pytest.mark.parametrize("initial", ["local-fast", "ray"])
@pytest.mark.parametrize("transport", [False, True])
def test_udf_physical_planning_keeps_source_policy(monkeypatch, initial, transport):
    monkeypatch.setenv("VANE_RUNNER", initial)
    with vane.connect() as connection:
        monkeypatch.setenv("VANE_RUNNER", "ray" if initial == "local-fast" else "local-fast")

        @vane.func(return_dtype="BIGINT")
        def identity(value):
            return value

        relation = connection.sql("SELECT 1::BIGINT AS value").select(identity(vane.col("value")))
        logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, f"policy-{initial}-{transport}")
        assert logical.session_config()["VANE_RUNNER"] == initial
        if transport:
            logical = pickle.loads(pickle.dumps(logical))
        with vane.connect() as planning_connection:
            physical = logical.to_physical_plan(planning_connection)
            payload = physical.collect_udf_nodes(conn=planning_connection)[0]["payload"]
            assert payload["execution_backend"] == ("ray_task" if initial == "ray" else "subprocess_task")


@pytest.mark.parametrize("method", ["execute", "sql"])
@pytest.mark.parametrize("parameters", ["none", "positional", "named", "statement"])
def test_sql_copy_to_reuses_relation_write_runner(monkeypatch, tmp_path, method, parameters):
    runner = _SQLRunner()
    factory_calls = _install_fake_ray_runner(monkeypatch, runner)
    target = tmp_path / "copy.parquet"
    with vane.connect() as connection:
        if parameters == "none":
            query, values = f"COPY (SELECT 7::BIGINT AS value) TO '{target}' (FORMAT PARQUET)", None
        elif parameters == "positional":
            query, values = "COPY (SELECT ?::BIGINT AS value) TO ? (FORMAT PARQUET)", [7, str(target)]
        else:
            query = "COPY (SELECT $value::BIGINT AS value) TO $path (FORMAT PARQUET)"
            values = {"value": 7, "path": str(target)}
            if parameters == "statement":
                query = connection.extract_statements(query)[0]
        monkeypatch.setenv("VANE_RUNNER", "local-fast")
        result = connection.execute(query, values) if method == "execute" else connection.sql(query, params=values)
        if method == "execute":
            assert result is connection
            assert [column[0] for column in connection.description] == ["Count"]
            assert connection.fetchall() == [(13,)]
            assert connection.fetchall() == []
        else:
            assert result is None
        assert len(factory_calls) == len(runner.writes) == 1
        assert runner.writes[0].session_config()["VANE_RUNNER"] == "ray"
        assert not target.exists()


@pytest.mark.parametrize("method", ["execute", "sql"])
def test_sql_copy_local_fast_stays_native(monkeypatch, tmp_path, method):
    runner = _SQLRunner()
    factory_calls = _install_fake_ray_runner(monkeypatch, runner)
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as connection:
        monkeypatch.setenv("VANE_RUNNER", "ray")
        target = tmp_path / "native.parquet"
        query = "COPY (SELECT i FROM range(?) t(i)) TO ? (FORMAT PARQUET)"
        params = [3, str(target)]
        if method == "execute":
            assert connection.execute(query, params).fetchall() == [(3,)]
        else:
            assert connection.sql(query, params=params) is None
        assert pq.read_table(target).column("i").to_pylist() == [0, 1, 2]
        assert factory_calls == []


@pytest.mark.parametrize("method", ["execute", "sql"])
@pytest.mark.parametrize("form", ["from", "return_files", "return_stats", "transaction"])
def test_unsupported_ray_copy_fails_before_dispatch(monkeypatch, tmp_path, method, form):
    runner = _SQLRunner()
    factory_calls = _install_fake_ray_runner(monkeypatch, runner)
    with vane.connect() as connection:
        target = tmp_path / "unsupported.parquet"
        query = f"COPY (SELECT 1 AS value) TO '{target}' (FORMAT PARQUET"
        if form == "from":
            connection.execute("CREATE TABLE items(value BIGINT)")
            query = f"COPY items FROM '{target}' (FORMAT PARQUET)"
        else:
            query += (f", {form}" if form.startswith("return_") else "") + ")"
        if form == "transaction":
            connection.begin()
        try:
            with pytest.raises((vane.NotImplementedException, vane.InvalidInputException), match="COPY|RETURN_"):
                getattr(connection, method)(query)
            assert factory_calls == []
            assert not target.exists()
        finally:
            if form == "transaction":
                connection.rollback()


@pytest.mark.parametrize("hook", ["begin", "close", "interrupt"])
def test_sql_copy_revalidates_after_parameter_conversion(monkeypatch, tmp_path, hook):
    runner = _SQLRunner()
    factory_calls = _install_fake_ray_runner(monkeypatch, runner)
    connection = vane.connect()

    class Parameters(list):
        triggered = False

        def __len__(self):
            if not self.triggered:
                self.triggered = True
                getattr(connection, hook)()
            return super().__len__()

    try:
        with pytest.raises((vane.InvalidInputException, vane.ConnectionException, vane.InterruptException)):
            connection.execute(
                "COPY (SELECT ? AS value) TO ? (FORMAT PARQUET)",
                Parameters([7, str(tmp_path / "reentrant.parquet")]),
            )
        assert factory_calls == []
    finally:
        connection.close()


def test_copy_keeps_replacement_scan_during_runner_binding(monkeypatch, tmp_path):
    runner = _SQLRunner()
    _install_fake_ray_runner(monkeypatch, runner)
    with vane.connect() as connection:
        source_table = pa.table({"value": [1, 2, 3]})
        connection.execute(
            "COPY (SELECT value + $offset AS value FROM source_table) TO $path (FORMAT PARQUET)",
            {"offset": 7, "path": str(tmp_path / "arrow.parquet")},
        )
        assert len(runner.writes) == 1
        assert source_table.num_rows == 3


@pytest.mark.parametrize("method", ["execute", "sql"])
def test_shared_sql_entry_drains_preceding_copy_and_queries(monkeypatch, tmp_path, method):
    runner = _SQLRunner()
    _install_fake_ray_runner(monkeypatch, runner)
    with vane.connect() as connection:
        query = f"SELECT 1::BIGINT; COPY (SELECT 2 AS value) TO '{tmp_path / 'out.parquet'}' (FORMAT PARQUET); SELECT ?::BIGINT"
        result = connection.execute(query, [3]) if method == "execute" else connection.sql(query, params=[3])
        assert len(runner.writes) == 1
        assert result.fetchall() == [(3,)]
        assert len(runner.plans) == 2
        assert runner.closed_iterators == 2


def test_executemany_uses_shared_runner_entry(monkeypatch, tmp_path):
    runner = _SQLRunner()
    _install_fake_ray_runner(monkeypatch, runner)
    with vane.connect() as connection:
        connection.executemany("SELECT ?::BIGINT", [[1], [2]])
        assert connection.fetchall() == [(2,)]
        assert len(runner.plans) == runner.closed_iterators == 2
        connection.executemany(
            "COPY (SELECT ? AS value) TO ? (FORMAT PARQUET)",
            [[1, str(tmp_path / "first.parquet")], [2, str(tmp_path / "second.parquet")]],
        )
        assert connection.fetchall() == [(13,)]
        assert len(runner.writes) == 2


def test_copy_failure_never_runs_locally(monkeypatch, tmp_path):
    class FailingRunner:
        def run_write(self, relation):
            raise RuntimeError("injected COPY failure")

    calls = _install_fake_ray_runner(monkeypatch, FailingRunner())
    target = tmp_path / "failure.parquet"
    with vane.connect() as connection:
        with pytest.raises(RuntimeError, match="injected COPY failure"):
            connection.execute("COPY (SELECT 1) TO ? (FORMAT PARQUET)", [str(target)])
        assert len(calls) == 1
        assert not target.exists()
        assert connection.description is None


def test_copy_result_error_preserves_committed_outcome(monkeypatch, tmp_path):
    runner = _SQLRunner()
    runner.outcome["rows_copied"] = "malformed count"
    _install_fake_ray_runner(monkeypatch, runner)
    with vane.connect() as connection:
        with pytest.raises(CopyResultUnavailableError) as raised:
            connection.execute("COPY (SELECT 1) TO ? (FORMAT PARQUET)", [str(tmp_path / "committed.parquet")])
        assert raised.value.safe_to_retry is False
        assert raised.value.operation_id == "sql-copy-operation"
        assert raised.value.cleanup_warnings == ("cleanup pending",)
        assert len(runner.writes) == 1


@pytest.mark.parametrize("method", ["execute", "sql"])
def test_sql_copy_to_runs_on_real_ray(ray_local, monkeypatch, tmp_path, method):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"value": list(range(12))}), source)
    target = tmp_path / "output.parquet"
    with vane.connect() as connection:
        vane.attach_function(
            lambda value: os.getpid(),
            connection=connection,
            alias="worker_pid",
            parameters=["BIGINT"],
            return_dtype="BIGINT",
        )
        monkeypatch.setenv("VANE_RUNNER", "local-fast")
        query = "COPY (SELECT value, worker_pid(value) AS pid FROM read_parquet($source) WHERE value >= $min) TO $target (FORMAT PARQUET)"
        params = {"source": str(source), "min": 7, "target": str(target)}
        try:
            if method == "execute":
                assert connection.execute(query, params).fetchall() == [(5,)]
            else:
                assert connection.sql(query, params=params) is None
            assert os.environ["VANE_RUNNER"] == "local-fast"
            files = [str(path) for path in target.glob("*.parquet")]
            assert files
            with vane.connect() as inspector:
                rows = inspector.execute("SELECT * FROM read_parquet(?) ORDER BY value", [files]).fetchall()
            assert [row[0] for row in rows] == list(range(7, 12))
            assert all(row[1] != os.getpid() for row in rows)
        finally:
            vane.teardown_runner()
