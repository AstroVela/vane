# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import vane
from vane.execution import ref_bundle, udf_subprocess
from vane.execution.request_admission import (
    RequestAdmissionLimits,
    RequestCancelled,
    RequestExecutionTimeout,
    RequestQueueFull,
    RequestQueueTimeout,
)
from vane.execution.udf_data_admission import DataAdmissionLimits, DataAdmissionWaitLimits
from vane.execution.udf_runtime_admission import TaskAdmissionLimits


@pytest.fixture
def native_environment(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    gc.collect()
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 420_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    with monkeypatch.context() as cpu_count:
        cpu_count.setattr(udf_subprocess.os, "cpu_count", lambda: 1)
        tasks = udf_subprocess._GlobalSubprocessTaskRuntime()
    monkeypatch.setattr(udf_subprocess, "_GLOBAL_TASK_RUNTIME", tasks)
    yield manager
    tasks.close(kill=True)
    gc.collect()
    assert manager.snapshot()["usage_bytes"] == 0


def _wait(predicate, future=None):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if future is not None and future.done():
            future.result()
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError("local query did not reach the expected state")


def _gated(cursor, tmp_path, value, *, actor=False):
    directory = str(tmp_path)

    def process(table):
        from pathlib import Path

        value = table.column(0)[0].as_py()
        Path(directory, f"entered-{value}").touch()
        if value == 1:
            deadline = time.monotonic() + 25
            while not Path(directory, "release").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("fixture worker was not released")
                time.sleep(0.01)
        return table

    class Model:
        def __call__(self, table):
            return process(table)

    return cursor.sql(f"SELECT {int(value)}::BIGINT AS x").map_batches(
        Model if actor else process,
        schema={"x": vane.sqltypes.BIGINT},
        execution_backend="subprocess_actor" if actor else "subprocess_task",
        actor_number=1 if actor else None,
    )


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
@pytest.mark.parametrize("tracked", [False, True])
def test_existing_query_entry_points_use_the_session_runtime(native_environment, entry, tracked):
    with vane.connect() as connection:
        runtime = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(1, 2), track_data=tracked, track_graph=tracked
        )
        with connection.cursor() as cursor:
            for value in (7, 8):
                if entry == "execute":
                    result = cursor.execute("SELECT ?::BIGINT AS x", [value])
                elif entry == "sql":
                    result = cursor.sql("SELECT ?::BIGINT AS x", params=[value])
                else:
                    result = cursor.sql("SELECT ?::BIGINT AS x", params=[value]).project("x + 1 AS x")
                assert result.fetchall() == [(value + (entry == "relation"),)]
        state = runtime.resource_snapshot()["request_admission"]
        assert state["executed_requests"] == state["completed_requests"] == 2
        assert state["active_requests"] == state["queued_requests"] == 0
        assert connection.sql("SELECT 42").fetchall() == [(42,)]
    assert runtime.resource_snapshot()["closed"]


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
def test_subprocess_udfs_use_captured_session_configuration(native_environment, monkeypatch, entry):
    monkeypatch.setenv("AWS_VANE_LOCAL_QUERY_TEST", "captured")
    with vane.connect() as connection:

        @vane.func(return_dtype="VARCHAR")
        def session_value(value):
            return os.environ["AWS_VANE_LOCAL_QUERY_TEST"]

        vane.attach_function(session_value, alias="session_value", parameters=["BIGINT"], connection=connection)
        runtime = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(1, 2), task_limit=TaskAdmissionLimits(1, 4), track_data=True
        )
        monkeypatch.setenv("AWS_VANE_LOCAL_QUERY_TEST", "later")
        with connection.cursor() as cursor:
            if entry == "execute":
                result = cursor.execute("SELECT session_value(7::BIGINT)")
            elif entry == "sql":
                result = cursor.sql("SELECT session_value(7::BIGINT)")
            else:
                result = cursor.sql("SELECT 7::BIGINT AS x").project("session_value(x)")
            assert result.fetchall() == [("captured",)]
        state = runtime.resource_snapshot()
        assert state["request_admission"]["active_requests"] == 0
        assert state["task_admission"]["running_tasks"] == 0
        assert state["data"]["retained_bytes"] == 0


@pytest.mark.parametrize("mode", ["graph", "byte_wait"])
@pytest.mark.parametrize("with_udf", [False, True])
def test_native_metadata_ignores_later_distributed_settings(native_environment, monkeypatch, mode, with_udf):
    monkeypatch.setenv("VANE_DISTRIBUTED_JOIN_STRATEGY", "hash")
    monkeypatch.setenv("AWS_VANE_LOCAL_QUERY_TEST", "captured")
    with vane.connect() as connection:

        @vane.func(return_dtype="VARCHAR")
        def session_value(value):
            return f"{os.environ['AWS_VANE_LOCAL_QUERY_TEST']}:{value}"

        if with_udf:
            vane.attach_function(session_value, alias="session_value", parameters=["BIGINT"], connection=connection)
        runtime = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(1, 1),
            track_graph=mode == "graph",
            data_limit=(
                DataAdmissionLimits(16_384, 2_048, 2_048, wait=DataAdmissionWaitLimits(4, 5))
                if mode == "byte_wait"
                else None
            ),
        )
        projection = "session_value(COALESCE(i, j))" if with_udf else "i, j"
        sql = f"SELECT {projection} FROM range(2) a(i) FULL JOIN range(3) b(j) ON i = j ORDER BY j"
        expected = [(f"captured:{value}",) for value in range(3)] if with_udf else [(0, 0), (1, 1), (None, 2)]
        assert connection.execute(sql).fetchall() == expected

        # Distributed broadcast restrictions must not invalidate an unchanged
        # native plan; UDF workers still receive the captured session settings.
        monkeypatch.setenv("VANE_DISTRIBUTED_JOIN_STRATEGY", "broadcast_right")
        monkeypatch.setenv("AWS_VANE_LOCAL_QUERY_TEST", "later")
        assert connection.execute(sql).fetchall() == expected
        assert connection.sql(sql).fetchall() == expected
        assert os.environ["VANE_DISTRIBUTED_JOIN_STRATEGY"] == "broadcast_right"
        state = runtime.resource_snapshot()["request_admission"]
        assert state["completed_requests"] == 3
        assert state["active_requests"] == state["queued_requests"] == 0


def _attach_correlated_udf(connection, function, *, actor):
    if actor:

        @vane.cls(return_dtype="VARCHAR", actor_number=1)
        class Model:
            def __call__(self, value):
                return function(value)

        udf = Model()
    else:
        udf = vane.func(return_dtype="VARCHAR")(function)
    vane.attach_function(udf, alias="correlated_udf", parameters=["BIGINT"], connection=connection)


@pytest.mark.parametrize("graph_mode", ["off", "on", "byte_wait"])
@pytest.mark.parametrize("actor", [False, True])
def test_correlated_subquery_udf_obeys_output_limit(native_environment, graph_mode, actor):
    with vane.connect() as connection:

        def oversized(value):
            return "x" * 10_000

        _attach_correlated_udf(connection, oversized, actor=actor)
        runtime = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(1, 1),
            task_limit=TaskAdmissionLimits(1, 4),
            data_limit=DataAdmissionLimits(
                16_384, 2_048, 2_048, wait=DataAdmissionWaitLimits(4, 5) if graph_mode == "byte_wait" else None
            ),
            track_graph=graph_mode == "on",
        )
        for sql in (
            "SELECT correlated_udf(0::BIGINT)",
            "SELECT i, (SELECT correlated_udf(j) FROM range(2) t(j) WHERE j < r.i LIMIT 1) FROM range(2) r(i)",
            "SELECT i, v FROM range(2) r(i), "
            "LATERAL (SELECT correlated_udf(j) AS v FROM range(100) t(j) WHERE j < r.i) t",
        ):
            with pytest.raises(Exception, match="output batch exceeds data limit"):
                connection.execute(sql).fetchall()
        state = runtime.resource_snapshot()
        assert state["request_admission"]["active_requests"] == 0
        assert state["task_admission"]["running_tasks"] == 0
        assert state["data"]["retained_bytes"] == 0


@pytest.mark.parametrize("track_graph", [False, True])
@pytest.mark.parametrize("actor", [False, True])
def test_correlated_subquery_uses_session_and_task_admission(
    native_environment, monkeypatch, tmp_path, track_graph, actor
):
    monkeypatch.setenv("AWS_VANE_LOCAL_QUERY_TEST", "captured")
    directory = str(tmp_path)

    def captured(value):
        from pathlib import Path

        Path(directory, "entered").touch()
        deadline = time.monotonic() + 25
        while not Path(directory, "release").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("correlated worker was not released")
            time.sleep(0.01)
        return os.environ["AWS_VANE_LOCAL_QUERY_TEST"]

    with vane.connect() as connection:
        _attach_correlated_udf(connection, captured, actor=actor)
        runtime = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(1, 1),
            task_limit=TaskAdmissionLimits(1, 4),
            data_limit=DataAdmissionLimits(16_384, 2_048, 2_048),
            track_graph=track_graph,
        )
        monkeypatch.setenv("AWS_VANE_LOCAL_QUERY_TEST", "later")
        relation = connection.sql(
            "SELECT i, (SELECT correlated_udf(j) FROM range(2) t(j) WHERE j < r.i LIMIT 1) "
            "FROM range(2) r(i) ORDER BY i"
        )
        with ThreadPoolExecutor(max_workers=1) as threads:
            future = threads.submit(relation.fetchall)
            try:
                _wait(lambda: (tmp_path / "entered").exists(), future)
                state = runtime.resource_snapshot()
                assert state["task_admission"]["running_tasks"] == 1
                assert state["data"]["reservations"] == 1
                if track_graph:
                    assert len(state["prepared_query_graphs"]) == 1
                    assert len(state["prepared_query_graphs"][0]["udf_node_ids"]) == 1
            finally:
                (tmp_path / "release").touch()
            assert future.result(timeout=10) == [(0, None), (1, "captured")]
        state = runtime.resource_snapshot()
        assert state["task_admission"]["running_tasks"] == 0
        assert state["data"]["usage_bytes"] == 0


@pytest.mark.parametrize("scan", ["range", "parquet"])
@pytest.mark.parametrize("with_udf", [False, True])
def test_correlated_metadata_preserves_native_scans(native_environment, tmp_path, scan, with_udf):
    with vane.connect() as connection:
        if scan == "parquet":
            path = tmp_path / "correlated.parquet"
            pq.write_table(pa.table({"i": [0, 1]}), path)
            connection.read_parquet(str(path)).create_view("correlated_rows")
            source = "correlated_rows"
        else:
            source = "range(2)"
        if with_udf:
            _attach_correlated_udf(connection, lambda value: str(value), actor=False)
        runtime = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(1, 1),
            data_limit=DataAdmissionLimits(16_384, 2_048, 2_048, wait=DataAdmissionWaitLimits(4, 5)),
        )
        value = "correlated_udf(j)" if with_udf else "j::VARCHAR"
        sql = f"SELECT i, (SELECT {value} FROM {source} t(j) WHERE j < r.i LIMIT 1) FROM {source} r(i) ORDER BY i"
        for _ in range(2):
            assert connection.execute(sql).fetchall() == [(0, None), (1, "0")]
        assert runtime.resource_snapshot()["data"]["usage_bytes"] == 0


@pytest.mark.parametrize("action", ["interrupt", "close", "expire"])
def test_queued_query_is_bounded_and_cancellable_before_native_start(native_environment, tmp_path, action):
    with vane.connect() as connection:
        runtime = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(1, 1, queue_timeout=0.4 if action == "expire" else 15)
        )
        with connection.cursor() as first, connection.cursor() as second, connection.cursor() as third:
            running = _gated(first, tmp_path, 1)
            queued = _gated(second, tmp_path, 2)
            with ThreadPoolExecutor(max_workers=2) as threads:
                first_future = threads.submit(running.fetchall)
                try:
                    _wait(lambda: (tmp_path / "entered-1").exists(), first_future)
                    second_future = threads.submit(queued.fetchall)
                    _wait(lambda: runtime.resource_snapshot()["request_admission"]["queued_requests"] == 1)
                    assert not (tmp_path / "entered-2").exists()
                    with pytest.raises(RequestQueueFull):
                        third.execute("SELECT 3")
                    if action == "interrupt":
                        second.interrupt()
                    elif action == "close":
                        second.close()
                    with pytest.raises(RequestQueueTimeout if action == "expire" else RequestCancelled):
                        second_future.result(timeout=10)
                    assert not (tmp_path / "entered-2").exists()
                    assert runtime.resource_snapshot()["request_admission"]["running_requests"] == 1
                finally:
                    (tmp_path / "release").touch()
                assert first_future.result(timeout=15) == [(1,)]
            assert third.execute("SELECT 3").fetchall() == [(3,)]


@pytest.mark.parametrize("actor", [False, True])
@pytest.mark.parametrize("action", ["interrupt", "close", "drain"])
def test_running_query_cancellation_and_drain_preserve_other_cursors(native_environment, tmp_path, actor, action):
    with vane.connect() as connection:
        runtime = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(2, 2), task_limit=TaskAdmissionLimits(1, 4), track_data=True
        )
        with connection.cursor() as first, connection.cursor() as second:
            running = _gated(first, tmp_path, 1, actor=actor)
            with ThreadPoolExecutor(max_workers=1) as threads:
                future = threads.submit(running.fetchall)
                try:
                    _wait(lambda: (tmp_path / "entered-1").exists(), future)
                    if action == "interrupt":
                        first.interrupt()
                    elif action == "close":
                        first.close()
                    else:
                        runtime.drain()
                        with pytest.raises(RuntimeError, match="draining"):
                            second.execute("SELECT 9")
                        (tmp_path / "release").touch()
                    if action == "drain":
                        assert future.result(timeout=15) == [(1,)]
                    else:
                        with pytest.raises(RequestCancelled):
                            future.result(timeout=15)
                        assert _gated(second, tmp_path, 2, actor=actor).fetchall() == [(2,)]
                finally:
                    (tmp_path / "release").touch()
            state = runtime.resource_snapshot()
            assert state["request_admission"]["active_requests"] == 0
            assert state["task_admission"]["running_tasks"] == 0
            assert state["data"]["retained_bytes"] == 0


@pytest.mark.parametrize("task_limited", [False, True])
def test_small_byte_budget_progresses_through_native_projection_and_partial_batches(native_environment, task_limited):
    def expand(table):
        return pa.table({"blob": [b"x" * 65_536 for _ in range(len(table))]})

    with vane.connect() as connection:
        runtime = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(2, 2),
            task_limit=TaskAdmissionLimits(1, 8) if task_limited else None,
            data_limit=DataAdmissionLimits(420_000, 140_000, 70_000, wait=DataAdmissionWaitLimits(8, 10)),
        )
        with connection.cursor() as cursor:
            for _ in range(2):
                relation = cursor.sql("SELECT i::BIGINT AS x FROM range(3) t(i)").map_batches(
                    expand,
                    schema={"blob": vane.sqltypes.BLOB},
                    execution_backend="subprocess_task",
                    batch_size=1,
                    min_task_batch_size=1,
                    task_input_max_bytes=8,
                )
                result = relation.project("octet_length(blob)::BIGINT AS size").map_batches(
                    lambda table: table,
                    schema={"size": vane.sqltypes.BIGINT},
                    execution_backend="subprocess_task",
                    batch_size=2,
                    min_task_batch_size=2,
                    task_input_max_bytes=70_000,
                )
                assert result.fetchall() == [(65_536,)] * 3
                assert runtime.resource_snapshot()["data"]["usage_bytes"] == 0


@pytest.mark.parametrize("with_udf", [False, True])
def test_byte_wait_preserves_native_parquet_scans(native_environment, tmp_path, with_udf):
    path = tmp_path / "input.parquet"
    pq.write_table(pa.table({"x": [7, 8, 9]}), path)
    with vane.connect() as connection:
        runtime = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(1, 1),
            data_limit=DataAdmissionLimits(420_000, 140_000, 70_000, wait=DataAdmissionWaitLimits(4, 5)),
        )
        relation = connection.read_parquet(str(path))
        if with_udf:
            relation = relation.map_batches(
                lambda table: table, schema={"x": vane.sqltypes.BIGINT}, execution_backend="subprocess_task"
            )
        assert relation.fetchall() == [(7,), (8,), (9,)]
        assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
        assert runtime.resource_snapshot()["data"]["usage_bytes"] == 0


def test_execution_deadline_prevents_udf_start(native_environment, tmp_path):
    with vane.connect() as connection:
        runtime = connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1), execution_timeout=0)
        relation = _gated(connection, tmp_path, 2)
        with pytest.raises(RequestExecutionTimeout):
            relation.fetchall()
        assert not (tmp_path / "entered-2").exists()
        state = runtime.resource_snapshot()["request_admission"]
        assert state["active_requests"] == 0
        assert state["execution_timed_out_requests"] == 1


@pytest.mark.parametrize("track_data", [False, True])
def test_connection_close_retries_failed_input_cleanup(native_environment, monkeypatch, tmp_path, track_data):
    manager = native_environment
    connection = vane.connect()
    runtime = connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1), track_data=track_data)
    try:
        with monkeypatch.context() as fault:

            def fail_cleanup(*args, **kwargs):
                raise RuntimeError("injected native-query input cleanup failure")

            fault.setattr(manager, "_release_input_ack_ref", fail_cleanup)
            with pytest.raises(Exception, match="input cleanup failure"):
                _gated(connection, tmp_path, 2).fetchall()
            assert manager.snapshot()["active_input_leases"] == 1
            state = runtime.resource_snapshot()["request_admission"]
            assert state["running_requests"] == state["cleanup_pending_requests"] == 1
            with pytest.raises(RuntimeError, match="cleanup failed"):
                connection.close()
            assert manager.snapshot()["active_input_leases"] == 1
        connection.close()
        assert manager.snapshot()["usage_bytes"] == 0
        assert runtime.resource_snapshot()["closed"]
    finally:
        connection.close()
        runtime.close(kill=True)


def test_configuration_is_explicit_session_owned_and_immutable(native_environment):
    with vane.connect() as first, vane.connect() as second:
        runtime = first.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
        with pytest.raises(vane.InvalidInputException, match="once"):
            first.configure_local_runtime(request_limit=RequestAdmissionLimits(2, 2))
        with first.cursor() as cursor:
            with pytest.raises(vane.InvalidInputException, match="session owner"):
                cursor.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
        runtime.drain()
        with pytest.raises(RuntimeError, match="draining"):
            first.execute("SELECT 1")
        assert second.execute("SELECT 2").fetchall() == [(2,)]
        second.execute("CREATE TABLE unaffected(x INTEGER)")
        assert second.sql("SELECT * FROM unaffected").fetchall() == []


@pytest.mark.parametrize("sql", ["CREATE TABLE rejected(x INTEGER)", "BEGIN", "PREPARE q AS SELECT 1"])
def test_configured_runtime_rejects_unsupported_statements_before_execution(native_environment, sql):
    with vane.connect() as connection:
        runtime = connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
        with pytest.raises(vane.InvalidInputException, match="read-only"):
            connection.execute(sql)
        assert connection.execute("SELECT 7").fetchall() == [(7,)]
        assert runtime.resource_snapshot()["request_admission"]["executed_requests"] == 1


@pytest.mark.parametrize("mode", ["off", "on", "byte_wait"])
@pytest.mark.parametrize("entry", ["execute", "sql", "table_function", "executemany"])
def test_runtime_rejects_json_execution_before_nested_udf(native_environment, tmp_path, mode, entry):
    marker = str(tmp_path / "nested-udf-started")
    with vane.connect() as connection:

        @vane.func(return_dtype="VARCHAR")
        def oversized(value):
            from pathlib import Path

            Path(marker).touch()
            return "x" * 10_000

        vane.attach_function(oversized, alias="nested_udf", parameters=["BIGINT"], connection=connection)
        serialized = connection.execute(
            "SELECT json_serialize_sql('SELECT nested_udf(7::BIGINT) AS value')"
        ).fetchall()[0][0]
        runtime = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(1, 1),
            task_limit=TaskAdmissionLimits(1, 1),
            data_limit=DataAdmissionLimits(
                16_384, 2_048, 2_048, wait=DataAdmissionWaitLimits(4, 5) if mode == "byte_wait" else None
            ),
            track_graph=mode == "on",
        )
        sql = "SELECT * FROM json_execute_serialized_sql(?)"
        with pytest.raises(vane.InvalidInputException, match="local runtime.*json_execute_serialized_sql"):
            if entry == "execute":
                result = connection.execute(sql, [serialized])
            elif entry == "sql":
                result = connection.sql(sql, params=[serialized])
            elif entry == "table_function":
                result = connection.table_function("json_execute_serialized_sql", [serialized])
            else:
                result = connection.executemany(sql, [[serialized], [serialized]])
            result.fetchall()
        assert not (tmp_path / "nested-udf-started").exists()
        state = runtime.resource_snapshot()
        assert state["request_admission"]["active_requests"] == 0
        assert state["task_admission"]["running_tasks"] == 0
        assert state["data"]["usage_bytes"] == 0
        # Rejection must release admission and preserve native JSON helpers.
        assert connection.execute("SELECT json_deserialize_sql(json_serialize_sql('SELECT 42'))").fetchall()[0][0]
        assert connection.execute("SELECT 7").fetchall() == [(7,)]
        with pytest.raises(vane.InvalidInputException, match="output batch exceeds data limit"):
            connection.execute("SELECT nested_udf(7::BIGINT)").fetchall()
        assert (tmp_path / "nested-udf-started").exists()
        assert runtime.resource_snapshot()["data"]["usage_bytes"] == 0


@pytest.mark.parametrize("track_graph", [False, True])
@pytest.mark.parametrize("shape", ["macro", "correlated"])
def test_runtime_rejects_json_execution_inside_owned_plans(native_environment, tmp_path, shape, track_graph):
    marker = str(tmp_path / "nested-udf-started")
    with vane.connect() as connection:

        @vane.func(return_dtype="BIGINT")
        def nested_udf(value):
            from pathlib import Path

            Path(marker).touch()
            return value

        vane.attach_function(nested_udf, alias="nested_udf", parameters=["BIGINT"], connection=connection)
        serialized = connection.execute(
            "SELECT json_serialize_sql('SELECT nested_udf(7::BIGINT) AS value')"
        ).fetchall()[0][0]
        argument = serialized.replace("'", "''")
        connection.execute(
            f"CREATE MACRO nested_json() AS TABLE SELECT * FROM json_execute_serialized_sql('{argument}')"
        )
        sql = (
            "SELECT * FROM nested_json()"
            if shape == "macro"
            else "SELECT i, (SELECT value FROM nested_json() WHERE value > i LIMIT 1) FROM range(2) r(i)"
        )
        # Also reject a relation whose nested connection was bound before the
        # runtime was configured. The executable plan is the admission boundary.
        relation = connection.sql(sql)
        runtime = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(1, 1), track_graph=track_graph
        )
        with pytest.raises(vane.InvalidInputException, match="local runtime.*json_execute_serialized_sql"):
            relation.fetchall()
        assert not (tmp_path / "nested-udf-started").exists()
        assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
        assert connection.execute("SELECT 7").fetchall() == [(7,)]


@pytest.mark.parametrize("track_graph", [False, True])
def test_deadline_does_not_wait_for_nested_json_execution(native_environment, tmp_path, track_graph):
    marker = str(tmp_path / "nested-udf-started")
    with vane.connect() as connection:

        @vane.func(return_dtype="BIGINT")
        def slow(value):
            from pathlib import Path

            Path(marker).touch()
            time.sleep(2)
            return value

        vane.attach_function(slow, alias="nested_slow", parameters=["BIGINT"], connection=connection)
        serialized = connection.execute("SELECT json_serialize_sql('SELECT nested_slow(7::BIGINT)')").fetchall()[0][0]
        runtime = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(1, 1), execution_timeout=0.05, track_graph=track_graph
        )
        with pytest.raises((vane.InvalidInputException, RequestExecutionTimeout)) as raised:
            connection.execute("SELECT * FROM json_execute_serialized_sql(?)", [serialized]).fetchall()
        if isinstance(raised.value, vane.InvalidInputException):
            assert "local runtime" in str(raised.value) and "json_execute_serialized_sql" in str(raised.value)
        # Under load the deadline may win the rejection race. In either case
        # the inner query must never start; this avoids a wall-clock assertion.
        assert not (tmp_path / "nested-udf-started").exists()
        assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0


def test_executemany_rebinds_each_native_request(native_environment):
    with vane.connect() as connection:
        runtime = connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
        assert connection.executemany("SELECT ?::BIGINT", [[1], [2], [3]]).fetchall() == [(3,)]
        assert runtime.resource_snapshot()["request_admission"]["completed_requests"] == 3


def test_native_preparation_error_retains_owners_across_exception_translation(native_environment, monkeypatch):
    from vane.execution.udf_actor_pool_lifecycle import OwnedActorPoolsError

    class Owner:
        failed = True
        pending = True

        def shutdown(self, *, kill=False):
            if self.failed:
                raise RuntimeError("injected retained preparation cleanup")
            self.pending = False

        def cleanup_pending(self):
            return self.pending

    owner = Owner()
    connection = vane.connect()
    runtime = connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))

    def fail_preparation(*args, **kwargs):
        raise OwnedActorPoolsError(
            "original preparation failure", owned_actor_pools=[owner], creation_error=ValueError("initialization")
        )

    try:
        with monkeypatch.context() as fault:
            fault.setattr(runtime._runtime, "_prepare", fail_preparation)
            with pytest.raises(Exception, match="original preparation failure"):
                connection.execute("SELECT 7")
        state = runtime.resource_snapshot()["request_admission"]
        assert state["running_requests"] == state["cleanup_pending_requests"] == 1
        with pytest.raises(RuntimeError, match="cleanup failed"):
            connection.close()
        assert owner.pending
        owner.failed = False
        connection.close()
        assert not owner.pending
        assert runtime.resource_snapshot()["closed"]
    finally:
        owner.failed = False
        connection.close()


def test_owner_close_cancels_running_and_queued_children(native_environment, tmp_path):
    connection = vane.connect()
    runtime = connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
    first, second = connection.cursor(), connection.cursor()
    with ThreadPoolExecutor(max_workers=3) as threads:
        first_future = threads.submit(_gated(first, tmp_path, 1).fetchall)
        try:
            _wait(lambda: (tmp_path / "entered-1").exists(), first_future)
            second_future = threads.submit(_gated(second, tmp_path, 2).fetchall)
            _wait(lambda: runtime.resource_snapshot()["request_admission"]["queued_requests"] == 1)
            threads.submit(connection.close).result(timeout=15)
            with pytest.raises(RequestCancelled):
                first_future.result(timeout=5)
            with pytest.raises(RuntimeError, match="drain|cancel|closed"):
                second_future.result(timeout=5)
            assert not (tmp_path / "entered-2").exists()
            assert runtime.resource_snapshot()["closed"]
        finally:
            (tmp_path / "release").touch()
            connection.close()


@pytest.mark.parametrize("runner", ["ray", "local"])
def test_configuration_rejects_other_runners(monkeypatch, runner):
    monkeypatch.setenv("VANE_RUNNER", runner)
    with vane.connect() as connection:
        with pytest.raises(vane.InvalidInputException, match="local-fast"):
            connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))


def test_two_active_queries_share_one_task_allowance(native_environment, tmp_path):
    with vane.connect() as connection:
        runtime = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(2, 2), task_limit=TaskAdmissionLimits(1, 4)
        )
        with connection.cursor() as first, connection.cursor() as second, ThreadPoolExecutor(max_workers=2) as threads:
            active = threads.submit(_gated(first, tmp_path, 1).fetchall)
            try:
                _wait(lambda: (tmp_path / "entered-1").exists(), active)
                pending = threads.submit(_gated(second, tmp_path, 2).fetchall)
                _wait(lambda: runtime.resource_snapshot()["task_admission"]["queued_tasks"] == 1, pending)
                snapshot = runtime.resource_snapshot()
                assert snapshot["request_admission"]["running_requests"] == 2
                assert snapshot["task_admission"]["running_tasks"] == 1
                assert not (tmp_path / "entered-2").exists()
            finally:
                (tmp_path / "release").touch()
            assert active.result(timeout=15) == [(1,)]
            assert pending.result(timeout=15) == [(2,)]
            assert runtime.resource_snapshot()["task_admission"]["running_tasks"] == 0


def test_interrupt_native_sql_and_reuse_its_cursor(native_environment, monkeypatch):
    from vane.execution.local_query import _NativeQuery

    started = threading.Event()
    original = _NativeQuery.started

    def record(self, interrupt):
        original(self, interrupt)
        started.set()

    monkeypatch.setattr(_NativeQuery, "started", record)
    with vane.connect() as connection:
        runtime = connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
        with ThreadPoolExecutor(max_workers=1) as threads:
            future = threads.submit(connection.execute, "SELECT sum(i) FROM range(1000000000000) t(i)")
            try:
                assert started.wait(10)
            finally:
                connection.interrupt()
            with pytest.raises(RequestCancelled):
                future.result(timeout=10)
        assert connection.execute("SELECT 42").fetchall() == [(42,)]
        assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0


@pytest.mark.parametrize(
    "entry",
    [
        "execute",
        "sql",
        "executemany",
        "relation",
        "table_function",
        "len",
        "project",
        "filter",
        "order",
        "aggregate",
        "str",
        "relation_query",
        "explain",
        "table",
        "view",
        "values",
        "from_arrow",
        "from_parquet",
        "read_csv",
        "read_json",
        "sqltype",
        "extract_statements",
        "register",
        "cursor",
        "configure",
    ],
)
@pytest.mark.parametrize("timed", [False, True])
def test_arrow_input_reentry_is_rejected_before_connection_locks(native_environment, entry, timed):
    # Arrow can invoke its Python iterator on a native worker while the caller
    # owns the connection locks. Isolate a regression so it cannot hang pytest.
    script = textwrap.dedent(
        """
        import faulthandler
        import sys
        import pyarrow as pa
        import vane
        from vane.execution.request_admission import RequestAdmissionLimits, RequestExecutionTimeout

        faulthandler.dump_traceback_later(8, exit=True)
        entry, timed = sys.argv[1], sys.argv[2] == "True"
        with vane.connect(config={"threads": 2}) as connection:
            connection.execute("CREATE TABLE t AS SELECT 42 AS x")
            connection.execute("CREATE VIEW v AS SELECT * FROM t")
            nested = connection.sql("SELECT 42 AS x")
            rejected = []

            def batches():
                try:
                    if entry == "execute":
                        connection.execute("SELECT 42")
                    elif entry == "sql":
                        connection.sql("SELECT 42").fetchall()
                    elif entry == "executemany":
                        connection.executemany("SELECT ?", [[42]])
                    elif entry == "relation":
                        nested.fetchall()
                    elif entry == "table_function":
                        connection.table_function("range", [1]).fetchall()
                    elif entry == "len":
                        len(nested)
                    elif entry == "project":
                        nested.project("x + 1")
                    elif entry == "filter":
                        nested.filter("x > 0")
                    elif entry == "order":
                        nested.order("x")
                    elif entry == "aggregate":
                        nested.aggregate("sum(x)")
                    elif entry == "str":
                        str(nested)
                    elif entry == "relation_query":
                        nested.query("n", "SELECT * FROM n")
                    elif entry == "explain":
                        nested.explain()
                    elif entry == "table":
                        connection.table("t")
                    elif entry == "view":
                        connection.view("v")
                    elif entry == "values":
                        connection.values([42])
                    elif entry == "from_arrow":
                        connection.from_arrow(pa.table({"x": [42]}))
                    elif entry == "from_parquet":
                        connection.from_parquet("unused.parquet")
                    elif entry == "read_csv":
                        connection.read_csv("unused.csv")
                    elif entry == "read_json":
                        connection.read_json("unused.json")
                    elif entry == "sqltype":
                        connection.sqltype("INTEGER")
                    elif entry == "extract_statements":
                        connection.extract_statements("SELECT 42")
                    elif entry == "register":
                        connection.register("another_input", pa.table({"x": [42]}))
                    elif entry == "cursor":
                        connection.cursor()
                    elif entry == "configure":
                        connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
                    else:
                        raise AssertionError(entry)
                except vane.InvalidInputException as error:
                    assert "reentrant queries" in str(error), str(error)
                    rejected.append(entry)
                else:
                    raise AssertionError("reentrant query was accepted")
                yield pa.record_batch({"x": [1]})

            reader = pa.RecordBatchReader.from_batches(pa.schema([("x", pa.int64())]), batches())
            relation = connection.from_arrow(reader)
            runtime = connection.configure_local_runtime(
                request_limit=RequestAdmissionLimits(1, 1), execution_timeout=0.1 if timed else None
            )
            try:
                assert relation.fetchall() == [(1,)]
            except RequestExecutionTimeout:
                assert timed
            assert rejected == [entry], rejected
            state = runtime.resource_snapshot()["request_admission"]
            assert state["active_requests"] == 0, state
            assert state["executed_requests"] == 1, state
            assert connection.execute("SELECT 7").fetchall() == [(7,)]
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, entry, str(timed)], capture_output=True, text=True, timeout=15
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("target", ["cursor", "parent", "sibling"])
@pytest.mark.parametrize("source", ["reader", "capsule"])
@pytest.mark.parametrize("threads", [1, 2])
def test_arrow_callback_close_checks_ownership_before_waiting(native_environment, target, source, threads):
    script = textwrap.dedent(
        """
        import faulthandler
        import sys
        import threading
        import pyarrow as pa
        import pyarrow.dataset
        import vane
        from vane.execution.request_admission import RequestAdmissionLimits

        faulthandler.dump_traceback_later(8, exit=True)
        target, source, threads = sys.argv[1], sys.argv[2], int(sys.argv[3])
        with vane.connect(config={"threads": threads}) as parent:
            runtime = parent.configure_local_runtime(
                request_limit=RequestAdmissionLimits(1, 1), execution_timeout=0.5
            )
            with parent.cursor() as cursor, parent.cursor() as sibling:
                callback_threads = []
                owner_thread = threading.get_ident()

                def batches():
                    callback_threads.append(threading.get_ident())
                    closing = {"cursor": cursor, "parent": parent, "sibling": sibling}[target]
                    if target == "sibling":
                        closing.close()
                    else:
                        try:
                            closing.close()
                        except vane.InvalidInputException as error:
                            assert "close a cursor reentrantly" in str(error), str(error)
                        else:
                            raise AssertionError("callback closed its active query")
                    yield pa.record_batch({"x": [1]})

                reader = pa.RecordBatchReader.from_batches(pa.schema([("x", pa.int64())]), batches())
                if source == "capsule":
                    reader = reader.__arrow_c_stream__()
                assert cursor.from_arrow(reader).fetchall() == [(1,)]
                assert callback_threads
                if source == "reader":
                    # Scanner.from_batches invokes the original reader on an Arrow worker.
                    assert callback_threads[0] != owner_thread
                assert cursor.execute("SELECT 7").fetchall() == [(7,)]
                assert parent.execute("SELECT 8").fetchall() == [(8,)]
                assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
        assert runtime.resource_snapshot()["closed"]
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, target, source, str(threads)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("target", ["cursor", "parent"])
def test_control_thread_can_close_during_arrow_input_callback(native_environment, target):
    script = textwrap.dedent(
        """
        import faulthandler
        import sys
        import threading
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        import pyarrow as pa
        import vane
        from vane.execution.request_admission import RequestAdmissionLimits, RequestCancelled

        faulthandler.dump_traceback_later(8, exit=True)
        entered, release = threading.Event(), threading.Event()
        with vane.connect(config={"threads": 2}) as parent:
            runtime = parent.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
            with parent.cursor() as cursor, ThreadPoolExecutor(max_workers=2) as workers:
                def batches():
                    entered.set()
                    assert release.wait(5)
                    yield pa.record_batch({"x": [1]})

                reader = pa.RecordBatchReader.from_batches(pa.schema([("x", pa.int64())]), batches())
                relation = cursor.from_arrow(reader)
                query = workers.submit(relation.fetchall)
                closing = None
                try:
                    assert entered.wait(5)
                    closing = workers.submit((cursor if sys.argv[1] == "cursor" else parent).close)
                    try:
                        closing.result(timeout=0.1)
                    except FutureTimeoutError:
                        pass
                    else:
                        raise AssertionError("close did not wait for the input callback")
                finally:
                    release.set()
                try:
                    query.result(timeout=5)
                except RequestCancelled:
                    pass
                else:
                    raise AssertionError("close did not cancel the active request")
                closing.result(timeout=5)
                assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
        assert runtime.resource_snapshot()["closed"]
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run([sys.executable, "-I", "-c", script, target], capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("entry", ["relation", "sql"])
def test_prebuilt_arrow_scanner_is_rejected_before_hidden_callbacks(native_environment, configured, entry):
    script = textwrap.dedent(
        """
        import faulthandler
        import sys
        import pyarrow as pa
        import pyarrow.dataset as ds
        import vane
        from vane.execution.request_admission import RequestAdmissionLimits

        faulthandler.dump_traceback_later(8, exit=True)
        configured, entry = sys.argv[1] == "True", sys.argv[2]
        called = []
        with vane.connect(config={"threads": 2}) as connection:
            def batches():
                called.append(True)
                if configured:
                    connection.close()
                yield pa.record_batch({"x": [1]})

            reader = pa.RecordBatchReader.from_batches(pa.schema([("x", pa.int64())]), batches())
            scanner = ds.Scanner.from_batches(reader)
            relation = connection.from_arrow(scanner)
            connection.register("source", scanner)
            if configured:
                runtime = connection.configure_local_runtime(
                    request_limit=RequestAdmissionLimits(1, 1), execution_timeout=0.5
                )
            def execute():
                if entry == "relation":
                    return relation.fetchall()
                return connection.execute("SELECT * FROM source").fetchall()
            if configured:
                try:
                    execute()
                except vane.InvalidInputException as error:
                    assert "prebuilt Arrow Scanners" in str(error), str(error)
                else:
                    raise AssertionError("opaque Scanner was accepted")
                assert not called
                assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
            else:
                assert execute() == [(1,)]
                assert called == [True]
            assert connection.execute("SELECT 7").fetchall() == [(7,)]
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(configured), entry], capture_output=True, text=True, timeout=15
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("phase", ["unpickle", "setup", "next"])
@pytest.mark.parametrize(
    "target,propagate", [("cursor", False), ("parent", False), ("sibling", False), ("cursor", True), ("parent", True)]
)
def test_datasource_callback_close_checks_ownership(native_environment, phase, target, propagate):
    script = textwrap.dedent(
        """
        import builtins
        import faulthandler
        import sys
        import threading
        import pyarrow as pa
        import vane
        from vane.datasource import DataSource, DataSourceTask
        from vane.execution.request_admission import RequestAdmissionLimits

        faulthandler.dump_traceback_later(8, exit=True)
        phase, target, propagate = sys.argv[1], sys.argv[2], sys.argv[3] == "True"
        owner_thread = threading.get_ident()
        seen, attempted = set(), []
        rendezvous = threading.Barrier(2)

        def callback():
            ident = threading.get_ident()
            if ident not in seen:
                seen.add(ident)
                rendezvous.wait(timeout=4)
            if ident == owner_thread or attempted:
                return
            attempted.append(ident)
            if target == "sibling":
                closing.close()
                return
            try:
                closing.close()
            except vane.InvalidInputException as error:
                assert "close a cursor reentrantly" in str(error), str(error)
                if propagate:
                    raise
            else:
                raise AssertionError("DataSource callback closed its active query")

        # Tasks are unpickled on native threads. Resolve the live connections
        # through a process-local hook, without trying to pickle a connection.
        builtins._vane_datasource_reentry = callback

        class Task(DataSourceTask):
            def __init__(self, phase):
                self.phase = phase
            def __setstate__(self, state):
                import builtins
                self.__dict__.update(state)
                if self.phase == "unpickle":
                    builtins._vane_datasource_reentry()
            def execute(self):
                import builtins
                import pyarrow as pa
                if self.phase == "setup":
                    builtins._vane_datasource_reentry()
                def batches():
                    if self.phase == "next":
                        builtins._vane_datasource_reentry()
                    yield pa.record_batch({"x": [1]})
                return batches()

        class Source(DataSource):
            def __init__(self, phase):
                self.phase = phase
            @property
            def schema(self):
                return {"x": "BIGINT"}
            def get_tasks(self):
                for _ in range(100):
                    yield Task(self.phase)

        with vane.connect(config={"threads": 2}) as parent:
            runtime = parent.configure_local_runtime(
                request_limit=RequestAdmissionLimits(1, 1), execution_timeout=1
            )
            with parent.cursor() as cursor, parent.cursor() as sibling:
                closing = {"cursor": cursor, "parent": parent, "sibling": sibling}[target]
                relation = cursor.from_datasource(Source(phase)).aggregate("sum(x)")
                try:
                    rows = relation.fetchall()
                except Exception as error:
                    assert propagate, str(error)
                    assert "close a cursor reentrantly" in str(error), str(error)
                else:
                    assert not propagate
                    assert rows == [(100,)], rows
                assert len(attempted) == 1 and attempted[0] != owner_thread, attempted
                state = runtime.resource_snapshot()["request_admission"]
                assert state["active_requests"] == 0, state
                assert state["executed_requests"] == 1, state
                assert cursor.execute("SELECT 7").fetchall() == [(7,)]
                assert parent.execute("SELECT 8").fetchall() == [(8,)]
        assert runtime.resource_snapshot()["closed"]
        del builtins._vane_datasource_reentry
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, phase, target, str(propagate)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("phase", ["setup", "next"])
@pytest.mark.parametrize("target", ["cursor", "parent"])
def test_control_thread_can_close_during_datasource_callback(native_environment, phase, target):
    script = textwrap.dedent(
        """
        import builtins
        import faulthandler
        import sys
        import threading
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        import pyarrow as pa
        import vane
        from vane.datasource import DataSource, DataSourceTask
        from vane.execution.request_admission import RequestAdmissionLimits, RequestCancelled

        faulthandler.dump_traceback_later(8, exit=True)
        phase, target = sys.argv[1:]
        entered, release = threading.Event(), threading.Event()
        rendezvous, seen = threading.Barrier(2), set()
        query_threads = []
        def callback():
            ident = threading.get_ident()
            if ident not in seen:
                seen.add(ident)
                rendezvous.wait(timeout=4)
            if ident != query_threads[0]:
                entered.set()
                assert release.wait(5)
        builtins._vane_datasource_reentry = callback

        class Task(DataSourceTask):
            def __init__(self, phase):
                self.phase = phase
            def execute(self):
                import builtins
                import pyarrow as pa
                if self.phase == "setup":
                    builtins._vane_datasource_reentry()
                def batches():
                    if self.phase == "next":
                        builtins._vane_datasource_reentry()
                    yield pa.record_batch({"x": [1]})
                return batches()
        class Source(DataSource):
            @property
            def schema(self):
                return {"x": "BIGINT"}
            def get_tasks(self):
                for _ in range(100):
                    yield Task(phase)

        with vane.connect(config={"threads": 2}) as parent:
            runtime = parent.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
            with parent.cursor() as cursor, ThreadPoolExecutor(max_workers=2) as workers:
                relation = cursor.from_datasource(Source()).aggregate("sum(x)")
                def execute():
                    query_threads.append(threading.get_ident())
                    return relation.fetchall()
                query = workers.submit(execute)
                closing = None
                try:
                    assert entered.wait(5)
                    closing = workers.submit((cursor if target == "cursor" else parent).close)
                    try:
                        closing.result(timeout=0.1)
                    except FutureTimeoutError:
                        pass
                    else:
                        raise AssertionError("close did not wait for the input callback")
                finally:
                    release.set()
                try:
                    query.result(timeout=5)
                except RequestCancelled:
                    pass
                else:
                    raise AssertionError("close did not cancel the active request")
                closing.result(timeout=5)
                assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
        assert runtime.resource_snapshot()["closed"]
        del builtins._vane_datasource_reentry
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, phase, target], capture_output=True, text=True, timeout=15
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("entry", ["execute", "relation", "executemany"])
@pytest.mark.parametrize("cancel_outcome", ["return", "raise", "close"])
def test_interrupt_fences_python_cancellation_until_it_returns(native_environment, monkeypatch, entry, cancel_outcome):
    from vane.execution.local_query import _NativeQuery
    from vane.execution.udf_local_request import LocalModelRequest

    started, cancelled, resume = threading.Event(), threading.Event(), threading.Event()
    requests = []
    original_started = _NativeQuery.started
    original_cancel = LocalModelRequest.cancel

    def record_started(self, interrupt):
        original_started(self, interrupt)
        requests.append(self.request)
        started.set()

    def pause_after_cancel(self):
        result = original_cancel(self)
        if requests and self is requests[0]:
            cancelled.set()
            assert resume.wait(10), "cancellation was not resumed"
            if cancel_outcome == "raise":
                raise RuntimeError("injected cancellation callback failure")
        return result

    monkeypatch.setattr(_NativeQuery, "started", record_started)
    monkeypatch.setattr(LocalModelRequest, "cancel", pause_after_cancel)
    with vane.connect() as connection:
        following = connection.sql("SELECT 42")
        runtime = connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
        with ThreadPoolExecutor(max_workers=2) as threads:
            first = threads.submit(connection.execute, "SELECT sum(i) FROM range(1000000000000) t(i)")
            interrupt = None
            try:
                assert started.wait(10)
                interrupt = threads.submit(connection.interrupt)
                assert cancelled.wait(10)
                with pytest.raises(RequestCancelled):
                    first.result(timeout=5)
                assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
                # The original request has retired, but its public interrupt
                # still owns the cursor fence. No subsequent native query starts.
                with pytest.raises(vane.InterruptException):
                    if entry == "execute":
                        connection.execute("SELECT 42")
                    elif entry == "relation":
                        following.fetchall()
                    else:
                        connection.executemany("SELECT ?", [[42]])
                assert len(requests) == 1
                if cancel_outcome == "close":
                    connection.close()
            finally:
                resume.set()
                if interrupt is not None:
                    if cancel_outcome == "raise":
                        with pytest.raises(RuntimeError, match="injected cancellation callback failure"):
                            interrupt.result(timeout=10)
                    else:
                        interrupt.result(timeout=10)
                if not first.done():
                    connection.interrupt()
            if cancel_outcome != "close":
                assert connection.execute("SELECT 7").fetchall() == [(7,)]
            assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0


@pytest.mark.parametrize("delivery", ["reader", "table"])
def test_materialized_arrow_result_releases_execution_before_consumption(native_environment, delivery):
    with vane.connect() as connection:
        runtime = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(1, 0), data_limit=DataAdmissionLimits(420_000, 140_000, 70_000)
        )
        with connection.cursor() as first, connection.cursor() as second:
            relation = first.sql("SELECT i::BIGINT AS x FROM range(4) t(i)").map_batches(
                lambda table: pa.table({"value": [str(x.as_py()) for x in table.column(0)]}),
                schema={"value": vane.sqltypes.VARCHAR},
                execution_backend="subprocess_task",
            )
            result = (
                relation.fetch_record_batch(rows_per_batch=2) if delivery == "reader" else relation.to_arrow_table()
            )
            assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
            assert runtime.resource_snapshot()["data"]["usage_bytes"] == 0
            assert second.execute("SELECT 9").fetchall() == [(9,)]
            table = result.read_all() if delivery == "reader" else result
            assert table.column(0).to_pylist() == ["0", "1", "2", "3"]


def test_lazy_relation_cannot_bypass_closed_session(native_environment):
    connection = vane.connect()
    runtime = connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
    relation = connection.sql("SELECT 7")
    connection.close()
    with pytest.raises(vane.ConnectionException, match="closed"):
        relation.fetchall()
    assert runtime.resource_snapshot()["closed"]


def test_explain_analyze_cannot_execute_outside_runtime_admission(native_environment, tmp_path):
    with vane.connect() as connection:
        runtime = connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 0))
        with pytest.raises(vane.InvalidInputException, match="EXPLAIN ANALYZE"):
            _gated(connection, tmp_path, 2).explain("analyze")
        assert not (tmp_path / "entered-2").exists()
        assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0


@pytest.mark.parametrize("action", ["interrupt", "close"])
def test_cancellation_before_request_publication_cancels_the_ticket(native_environment, monkeypatch, action):
    with vane.connect() as connection:
        runtime = connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1, queue_timeout=15))
        occupied = runtime._runtime.request()
        created, resume = threading.Event(), threading.Event()
        original = runtime._runtime.request

        def pause():
            request = original()
            created.set()
            assert resume.wait(10)
            return request

        monkeypatch.setattr(runtime._runtime, "request", pause)
        with connection.cursor() as cursor, ThreadPoolExecutor(max_workers=2) as threads:
            future = threads.submit(cursor.execute, "SELECT 7")
            closing = None
            try:
                assert created.wait(10)
                if action == "interrupt":
                    cursor.interrupt()
                else:
                    closing = threads.submit(cursor.close)
                    with pytest.raises(FutureTimeoutError):
                        closing.result(timeout=0.1)
                resume.set()
                with pytest.raises(RequestCancelled):
                    future.result(timeout=2)
                if closing is not None:
                    closing.result(timeout=2)
            finally:
                resume.set()
                occupied.shutdown()
                if action == "interrupt":
                    cursor.interrupt()
        assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0


def test_reentrant_close_during_preparation_preserves_the_cursor(native_environment, monkeypatch):
    from vane.execution.local_query import _NativeQuery

    with vane.connect() as connection:
        runtime = connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))

        def close_during_prepare(*args):
            connection.close()

        with monkeypatch.context() as fault:
            fault.setattr(_NativeQuery, "prepare", close_during_prepare)
            with pytest.raises(vane.Error, match="reentrantly"):
                connection.execute("SELECT 7")
        assert connection.execute("SELECT 42").fetchall() == [(42,)]
        assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0


@pytest.mark.parametrize("action", ["close", "interrupt"])
def test_cancel_query_owned_actor_during_initialization(native_environment, tmp_path, action):
    directory = str(tmp_path)

    class Model:
        def __init__(self):
            from pathlib import Path

            Path(directory, "initializing").touch()
            deadline = time.monotonic() + 25
            while not Path(directory, "release").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("fixture initializer was not released")
                time.sleep(0.01)

        def __call__(self, table):
            return table

    with vane.connect() as connection:
        runtime = connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
        with connection.cursor() as cursor, ThreadPoolExecutor(max_workers=2) as threads:
            relation = cursor.sql("SELECT 7::BIGINT AS x").map_batches(
                Model, schema={"x": vane.sqltypes.BIGINT}, execution_backend="subprocess_actor", actor_number=1
            )
            future = threads.submit(relation.fetchall)
            try:
                _wait(lambda: (tmp_path / "initializing").exists(), future)
                threads.submit(getattr(cursor, action)).result(timeout=10)
                with pytest.raises(RequestCancelled):
                    future.result(timeout=5)
                if action == "interrupt":
                    assert cursor.execute("SELECT 42").fetchall() == [(42,)]
            finally:
                (tmp_path / "release").touch()
        assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
        assert connection.execute("SELECT 42").fetchall() == [(42,)]


@pytest.mark.parametrize("track_data", [False, True])
def test_failed_output_grant_cleanup_retains_native_request(native_environment, monkeypatch, tmp_path, track_data):
    manager = native_environment
    send = udf_subprocess._send_message

    def fail_delivery(sock, kind, payload=b""):
        if kind == udf_subprocess._MSG_OUTPUT_GRANT_GRANTED:
            raise OSError("injected native-query grant delivery failure")
        return send(sock, kind, payload)

    def fail_cleanup(*args, **kwargs):
        raise OSError("injected native-query grant cleanup failure")

    connection = vane.connect()
    runtime = connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1), track_data=track_data)
    try:
        with monkeypatch.context() as fault:
            fault.setattr(udf_subprocess, "_send_message", fail_delivery)
            fault.setattr(manager, "release_output_grant", fail_cleanup)
            with pytest.raises(Exception, match="grant (delivery|cleanup) failure"):
                _gated(connection, tmp_path, 2).fetchall()
            assert manager.snapshot()["output_grant_bytes"] > 0
            state = runtime.resource_snapshot()["request_admission"]
            assert state["running_requests"] == state["cleanup_pending_requests"] == 1
            with pytest.raises(RuntimeError, match="cleanup failed"):
                connection.close()
        connection.close()
        assert manager.snapshot()["usage_bytes"] == 0
        assert runtime.resource_snapshot()["closed"]
    finally:
        connection.close()
        runtime.close(kill=True)
