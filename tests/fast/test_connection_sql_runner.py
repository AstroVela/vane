# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pickle
import threading

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


def _install_sql_runner(monkeypatch, runner, runner_type):
    calls = _install_fake_ray_runner(monkeypatch, runner)
    if runner_type == "local":
        monkeypatch.setenv("VANE_RUNNER", "local")

        def set_runner_local():
            calls.append((None, False))
            return runner

        monkeypatch.setattr(vane._native, "set_runner_local", set_runner_local)
    return calls


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
            with pytest.raises(vane.InvalidInputException, match="Invalid runner"):
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
@pytest.mark.parametrize("runner_type", ["local", "ray"])
def test_sql_copy_to_reuses_relation_write_runner(monkeypatch, tmp_path, method, parameters, runner_type):
    runner = _SQLRunner()
    factory_calls = _install_sql_runner(monkeypatch, runner, runner_type)
    target = tmp_path / "copy.parquet"
    with vane.connect() as connection:
        if parameters == "none":
            query, values = f"COPY (SELECT 7::BIGINT AS value) TO '{target}' (FORMAT PARQUET)", None
        elif parameters == "positional":
            # DuckDB numbers COPY's filename placeholder before its query placeholders.
            query, values = "COPY (SELECT ?::BIGINT AS value) TO ? (FORMAT PARQUET)", [str(target), 7]
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
        assert runner.writes[0].session_config()["VANE_RUNNER"] == runner_type
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
        params = [str(target), 3]
        if method == "execute":
            assert connection.execute(query, params).fetchall() == [(3,)]
        else:
            assert connection.sql(query, params=params) is None
        assert pq.read_table(target).column("i").to_pylist() == [0, 1, 2]
        assert factory_calls == []


@pytest.mark.parametrize("method", ["execute", "sql"])
@pytest.mark.parametrize("runner_type", ["local", "ray"])
@pytest.mark.parametrize("destination", ["stdout", "parameter", "expression", "descriptor", "device", "fifo"])
def test_runner_copy_rejects_non_file_destinations(monkeypatch, tmp_path, method, runner_type, destination):
    runner = _SQLRunner()
    calls = _install_sql_runner(monkeypatch, runner, runner_type)
    params = None
    if destination == "stdout":
        target = "STDOUT"
    elif destination == "parameter":
        target, params = "$path", {"path": "/dev/stdout"}
    elif destination == "expression":
        target = "('/dev/' || 'stdout')"
    elif destination == "descriptor":
        target = "'/dev/fd/1'"
    elif destination == "device":
        target = repr(os.devnull)
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("named pipes require mkfifo")
        fifo = tmp_path / "output.fifo"
        os.mkfifo(fifo)
        target = f"'{fifo}'"
    with vane.connect() as connection:
        query = f"COPY (SELECT 7 AS value) TO {target} (FORMAT CSV)"
        with pytest.raises(vane.NotImplementedException, match="file dataset destination"):
            if method == "execute":
                connection.execute(query, params)
            else:
                connection.sql(query, params=params)
    assert calls == runner.writes == []
    assert sorted(path.name for path in tmp_path.iterdir()) == (["output.fifo"] if destination == "fifo" else [])


@pytest.mark.parametrize("method", ["execute", "sql"])
def test_local_fast_copy_stdout_remains_native(monkeypatch, capfd, method):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as connection:
        result = getattr(connection, method)("COPY (SELECT 7 AS value) TO STDOUT (FORMAT CSV, HEADER true)")
        if method == "execute":
            assert result.fetchall() == [(1,)]
        else:
            assert result is None
    assert "value\n7\n" in capfd.readouterr().out


@pytest.mark.parametrize("runner_type", ["local", "ray"])
def test_sql_copy_keeps_commit_when_interrupt_races_with_result(monkeypatch, tmp_path, runner_type):
    class CommittedRunner(_SQLRunner):
        def run_write(self, relation):
            result = super().run_write(relation)
            connection.interrupt()
            return result

    runner = CommittedRunner()
    _install_sql_runner(monkeypatch, runner, runner_type)
    with vane.connect() as connection:
        assert connection.execute(
            "COPY (SELECT 1) TO ? (FORMAT PARQUET)", [str(tmp_path / "committed.parquet")]
        ).fetchall() == [(13,)]
    from vane._query_interrupt import has_query_interrupt_check

    assert not has_query_interrupt_check()


@pytest.mark.parametrize("method", ["execute", "sql"])
@pytest.mark.parametrize("runner_type", ["local", pytest.param("ray", marks=pytest.mark.real_ray)])
def test_connection_interrupt_stops_runner_copy(monkeypatch, tmp_path, request, method, runner_type):
    waiting = threading.Event()
    finished = threading.Event()
    errors = []
    if runner_type == "ray":
        request.getfixturevalue("ray_local")
        from vane.runners.ray import driver

        resolve = driver._RayProgressSession.resolve

        def observe_wait(self, ref):
            waiting.set()
            return resolve(self, ref)

        monkeypatch.setattr(driver._RayProgressSession, "resolve", observe_wait)
    else:
        from vane.runners.local import runner as local_module

        execute_fragment = local_module._InProcessFragmentExecutor.__call__

        def observe_fragment(self, *args, **kwargs):
            waiting.set()
            return execute_fragment(self, *args, **kwargs)

        monkeypatch.setattr(local_module._InProcessFragmentExecutor, "__call__", observe_fragment)
    vane.teardown_runner()
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    connection = vane.connect()
    target = tmp_path / "interrupted.parquet"

    def write():
        try:
            query = f"COPY (SELECT sum(i) FROM range(100000000000000) t(i)) TO '{target}' (FORMAT PARQUET)"
            getattr(connection, method)(query)
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    worker = threading.Thread(target=write)
    worker.start()
    try:
        assert waiting.wait(30), errors
        connection.interrupt()
        assert finished.wait(30), "COPY did not stop after connection.interrupt()"
        assert len(errors) == 1 and isinstance(errors[0], vane.InterruptException), errors
        assert not any(path.is_file() for path in tmp_path.rglob("*.parquet"))
        assert not list(tmp_path.rglob("committed"))
        assert connection.execute("SELECT 77::BIGINT").fetchall() == [(77,)]
    finally:
        if worker.is_alive():
            connection.interrupt()
        vane.teardown_runner()
        worker.join(30)
        assert not worker.is_alive()
        connection.close()


@pytest.mark.parametrize("method", ["execute", "sql"])
@pytest.mark.parametrize("form", ["from", "return_files", "return_stats", "transaction"])
@pytest.mark.parametrize("runner_type", ["local", "ray"])
def test_unsupported_runner_copy_fails_before_dispatch(monkeypatch, tmp_path, method, form, runner_type):
    runner = _SQLRunner()
    factory_calls = _install_sql_runner(monkeypatch, runner, runner_type)
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
                Parameters([str(tmp_path / "reentrant.parquet"), 7]),
            )
        assert factory_calls == []
    finally:
        connection.close()


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
            [[str(tmp_path / "first.parquet"), 1], [str(tmp_path / "second.parquet"), 2]],
        )
        assert connection.fetchall() == [(13,)]
        assert len(runner.writes) == 2


@pytest.mark.parametrize("runner_type", ["local", "ray"])
def test_copy_failure_never_runs_locally(monkeypatch, tmp_path, runner_type):
    class FailingRunner:
        def run_write(self, relation):
            raise RuntimeError("injected COPY failure")

    calls = _install_sql_runner(monkeypatch, FailingRunner(), runner_type)
    target = tmp_path / "failure.parquet"
    with vane.connect() as connection:
        with pytest.raises(RuntimeError, match="injected COPY failure"):
            connection.execute("COPY (SELECT 1) TO ? (FORMAT PARQUET)", [str(target)])
        assert len(calls) == 1
        assert not target.exists()
        assert connection.description is None


@pytest.mark.parametrize("runner_type", ["local", "ray"])
def test_copy_result_error_preserves_committed_outcome(monkeypatch, tmp_path, runner_type):
    runner = _SQLRunner()
    runner.outcome["rows_copied"] = "malformed count"
    _install_sql_runner(monkeypatch, runner, runner_type)
    with vane.connect() as connection:
        with pytest.raises(CopyResultUnavailableError) as raised:
            connection.execute("COPY (SELECT 1) TO ? (FORMAT PARQUET)", [str(tmp_path / "committed.parquet")])
        assert raised.value.safe_to_retry is False
        assert raised.value.operation_id == "sql-copy-operation"
        assert raised.value.cleanup_warnings == ("cleanup pending",)
        assert len(runner.writes) == 1


@pytest.mark.parametrize("method", ["execute", "sql", "executemany"])
@pytest.mark.parametrize("cleanup_error", [False, True])
def test_sql_copy_to_runs_on_local_fte(monkeypatch, tmp_path, method, cleanup_error):
    from vane.runners.local import runner as local_module

    source = tmp_path / "source.parquet"
    target = tmp_path / "output.parquet"
    pq.write_table(pa.table({"value": list(range(12))}), source)
    if cleanup_error:

        class CleanupError(RuntimeError):
            def __str__(self):
                raise AssertionError("cleanup diagnostics must not call exception formatting hooks")

        shutdown = local_module._shutdown_local_write_resources

        def fail_after_shutdown(*args, **kwargs):
            return [*shutdown(*args, **kwargs), CleanupError("planned cleanup failure after commit")]

        monkeypatch.setattr(local_module, "_shutdown_local_write_resources", fail_after_shutdown)
    vane.teardown_runner()
    monkeypatch.setenv("VANE_RUNNER", "local")
    try:
        with vane.connect() as connection:
            monkeypatch.setenv("VANE_RUNNER", "invalid")
            query = "COPY (SELECT * FROM read_parquet($source) WHERE value >= $min) TO $target (FORMAT PARQUET)"
            params = {"source": str(source), "min": 7, "target": str(target)}

            def write():
                if method == "sql":
                    assert connection.sql(query, params=params) is None
                elif method == "executemany":
                    assert connection.executemany(query, [params]).fetchall() == [(5,)]
                else:
                    assert connection.execute(query, params).fetchall() == [(5,)]

            if cleanup_error:
                with pytest.raises(CopyResultUnavailableError, match="planned cleanup failure after commit") as error:
                    write()
                assert error.value.operation_id
                assert error.value.safe_to_retry is False
                assert error.value.write_state == "committed"
            else:
                write()
            assert os.environ["VANE_RUNNER"] == "invalid"
            files = list(target.glob("*.parquet"))
            assert files
            assert sorted(pq.read_table([str(path) for path in files]).column("value").to_pylist()) == list(
                range(7, 12)
            )
    finally:
        vane.teardown_runner()


@pytest.mark.parametrize("method", ["execute", "sql"])
@pytest.mark.parametrize("source_kind", ["parquet", "arrow"])
def test_sql_copy_to_runs_on_real_ray(ray_local, monkeypatch, tmp_path, method, source_kind):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    source = tmp_path / "source.parquet"
    source_table = pa.table({"value": list(range(12))})
    pq.write_table(source_table, source)
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
        source_query = "read_parquet($source)" if source_kind == "parquet" else "source_table"
        query = f"COPY (SELECT value, worker_pid(value) AS pid FROM {source_query} WHERE value >= $min) TO $target (FORMAT PARQUET)"
        params = {"min": 7, "target": str(target)}
        if source_kind == "parquet":
            params["source"] = str(source)
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
