# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

import vane
from vane.execution.request_admission import (
    RequestAdmissionLimits,
    RequestCancelled,
    RequestQueueFull,
    RequestQueueTimeout,
)
from vane.execution.udf_data_admission import DataAdmissionLimits
from vane.execution.udf_local_model import LocalModelRuntime
from vane.execution.udf_runtime_admission import TaskAdmissionLimits


def _plan(cursor, function, value, *, actor=True):
    relation = cursor.sql(f"SELECT {int(value)}::BIGINT AS x").map_batches(
        function,
        schema={"x": vane.sqltypes.BIGINT},
        execution_backend="subprocess_actor" if actor else "subprocess_task",
        actor_number=1 if actor else None,
    )
    return vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(cursor)


def _bindings(plan, cursor):
    return {str(node["node_id"]): "model" for node in plan.collect_udf_nodes(conn=cursor)}


def _values(result):
    return [value for table in result.partition_payloads for value in table.column(0).to_pylist()]


def _wait_file(path):
    deadline = time.monotonic() + 15
    while not path.exists():
        assert time.monotonic() < deadline, f"worker never entered: {path.name}"
        time.sleep(0.01)


@pytest.mark.parametrize("operation", ["cancel", "timeout", "drain"])
def test_request_queue_waits_without_model_borrows_or_task_bytes(monkeypatch, tmp_path, operation):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    directory = str(tmp_path)

    class Model:
        def __init__(self):
            from pathlib import Path

            with (Path(directory) / "constructors").open("a") as output:
                output.write(f"{os.getpid()}\n")

        def __call__(self, table):
            from pathlib import Path

            if table.column(0)[0].as_py() == 1:
                (Path(directory) / "entered").touch()
                deadline = time.monotonic() + 30
                while not (Path(directory) / "release").exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("test did not release model")
                    time.sleep(0.01)
            return table

    with vane.connect() as connection:
        cursors = [connection.cursor() for _ in range(3)]
        plans = [_plan(cursor, Model, i + 1) for i, cursor in enumerate(cursors)]
        runtime = LocalModelRuntime(
            session_id=plans[0].session_id(),
            session_config=plans[0].session_config(),
            request_limit=RequestAdmissionLimits(1, 1),
            task_limit=TaskAdmissionLimits(1, 2),
            data_limit=DataAdmissionLimits(8192, 2048, 2048),
        )
        model = runtime.register(
            "model", version="v1", payload=plans[0].collect_udf_nodes(conn=cursors[0])[0]["payload"]
        )
        model.prewarm()
        with model.acquire() as borrow:
            pids = borrow.pool.worker_pids()
        first = runtime.request()
        with ThreadPoolExecutor(max_workers=2) as threads:
            future = threads.submit(first.execute, plans[0], _bindings(plans[0], cursors[0]), conn=cursors[0])
            try:
                _wait_file(tmp_path / "entered")
                before = runtime.resource_snapshot()
                second = runtime.request(queue_timeout=0.1 if operation == "timeout" else 10)
                with pytest.raises(RequestQueueFull):
                    runtime.request()
                waiting = threads.submit(second.execute, plans[1], _bindings(plans[1], cursors[1]), conn=cursors[1])
                during = runtime.resource_snapshot()
                assert before["active_borrows"] == during["active_borrows"] == 1
                assert before["data"]["reservations"] == during["data"]["reservations"] == 1
                assert before["data"]["usage_bytes"] == during["data"]["usage_bytes"]
                assert during["task_admission"]["queries"] == 1
                if operation == "cancel":
                    assert second.cancel()
                elif operation == "drain":
                    with pytest.raises(TimeoutError, match="active execution or cleanup"):
                        runtime.close()
                with pytest.raises(RequestQueueTimeout if operation == "timeout" else RequestCancelled):
                    waiting.result(timeout=5)
            finally:
                (tmp_path / "release").touch()
            result = future.result(timeout=10)
            assert _values(result) == [1]
        if operation != "drain":
            later = runtime.request().execute(plans[2], _bindings(plans[2], cursors[2]), conn=cursors[2])
            assert _values(later) == [3]
            with model.acquire() as borrow:
                assert borrow.pool.worker_pids() == pids
        assert len((tmp_path / "constructors").read_text().splitlines()) == 1
        runtime.close(timeout=5)
        assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
        for cursor in cursors:
            cursor.close()


@pytest.mark.parametrize("backend", ["subprocess_actor", "subprocess_task"])
def test_slow_result_consumers_keep_byte_ownership_after_request_finishes(monkeypatch, backend):
    from vane.execution import udf_subprocess as local

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    views = []
    captured = False
    track_output = local.track_local_shm_output

    def retain_output_view(task, result):
        nonlocal captured
        track_output(task, result)
        if not captured:
            captured = True
            views.append(result[1][0].to_table())

    monkeypatch.setattr(local, "track_local_shm_output", retain_output_view)

    class Model:
        def __call__(self, table):
            return table

    with vane.connect() as connection:
        first_cursor, second_cursor = connection.cursor(), connection.cursor()
        function = Model if backend == "subprocess_actor" else lambda table: table
        first_plan = _plan(first_cursor, function, 7, actor=backend == "subprocess_actor")
        second_plan = _plan(second_cursor, function, 8, actor=backend == "subprocess_actor")
        runtime = LocalModelRuntime(
            session_id=first_plan.session_id(),
            session_config=first_plan.session_config(),
            request_limit=RequestAdmissionLimits(1, 1),
            data_limit=DataAdmissionLimits(4096, 2048, 2048),
        )
        try:
            # Query-owned actors and task-only plans share the same request gate.
            result = runtime.request().execute(first_plan, {}, conn=first_cursor)
            assert _values(result) == [7]
            assert views[0].column(0).to_pylist() == [7]
            assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
            assert runtime.resource_snapshot()["data"]["retained_bytes"] > 0
            with pytest.raises(Exception, match="data admission capacity exceeded"):
                runtime.request().execute(second_plan, {}, conn=second_cursor)
            assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
            del result
            views.clear()
            gc.collect()
            assert runtime.resource_snapshot()["data"]["retained_bytes"] == 0
            # A failed execution is not replayed; the caller builds a fresh plan.
            plan = _plan(second_cursor, function, 9, actor=backend == "subprocess_actor")
            result = runtime.request().execute(plan, {}, conn=second_cursor)
            assert _values(result) == [9]
            runtime.close(timeout=5)
            assert _values(result) == [9]
        finally:
            views.clear()
            gc.collect()
            runtime.close(timeout=5, kill=True)
            first_cursor.close()
            second_cursor.close()


@pytest.mark.parametrize("failure", ["udf_error", "worker_exit"])
def test_failed_native_request_returns_capacity_and_shared_model_can_recover(monkeypatch, failure):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    class Model:
        def __call__(self, table):
            if table.column(0)[0].as_py() < 0:
                if failure == "worker_exit":
                    os._exit(2)
                raise ValueError("planned native request failure")
            return table

    with vane.connect() as connection:
        first_cursor, second_cursor = connection.cursor(), connection.cursor()
        first_plan, second_plan = _plan(first_cursor, Model, -1), _plan(second_cursor, Model, 2)
        runtime = LocalModelRuntime(
            session_id=first_plan.session_id(),
            session_config=first_plan.session_config(),
            request_limit=RequestAdmissionLimits(1, 1),
            task_limit=TaskAdmissionLimits(1, 2),
            track_data=True,
        )
        runtime.register("model", version="v1", payload=first_plan.collect_udf_nodes(conn=first_cursor)[0]["payload"])
        try:
            with pytest.raises(Exception):
                runtime.request().execute(first_plan, _bindings(first_plan, first_cursor), conn=first_cursor)
            assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
            assert runtime.resource_snapshot()["active_borrows"] == 0
            result = runtime.request().execute(second_plan, _bindings(second_plan, second_cursor), conn=second_cursor)
            assert _values(result) == [2]
        finally:
            runtime.close(timeout=5, kill=True)
            first_cursor.close()
            second_cursor.close()


def test_request_without_udfs_uses_native_execution(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as connection:
        relation = connection.sql("SELECT 42 AS x")
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(connection)
        with LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            request_limit=RequestAdmissionLimits(1, 1),
        ) as runtime:
            assert _values(runtime.request().execute(plan, {}, conn=connection)) == [42]
