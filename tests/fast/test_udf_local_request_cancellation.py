# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest

import vane
from vane.execution import ref_bundle, udf_local_request
from vane.execution.request_admission import RequestAdmissionLimits, RequestCancelled
from vane.execution.udf_data_admission import DataAdmissionLimits
from vane.execution.udf_local_model import LocalModelRuntime
from vane.execution.udf_runtime_admission import TaskAdmissionLimits


def _plan(cursor, function, value, *, actor):
    relation = cursor.sql(f"SELECT {value}::BIGINT AS x").map_batches(
        function,
        schema={"x": vane.sqltypes.BIGINT},
        execution_backend="subprocess_actor" if actor else "subprocess_task",
        actor_number=2 if actor else None,
    )
    return vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(cursor)


def _values(result):
    return [value for table in result.partition_payloads for value in table.column(0).to_pylist()]


def _wait(predicate, message):
    deadline = time.monotonic() + 15
    while not predicate():
        assert time.monotonic() < deadline, message
        time.sleep(0.01)


@pytest.mark.parametrize("actor", [False, True])
@pytest.mark.parametrize("accounting", ["default", "tracked", "limited"])
@pytest.mark.parametrize("task_limited", [False, True])
def test_cancel_running_native_udf_preserves_other_query_and_pool(
    monkeypatch, tmp_path, actor, accounting, task_limited
):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    marker = str(tmp_path)

    def process(table):
        from pathlib import Path

        value = table.column(0)[0].as_py()
        Path(marker, f"entered-{value}").write_text(str(os.getpid()))
        if value in (1, 2):
            deadline = time.monotonic() + 25
            while not Path(marker, f"release-{value}").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("test worker was not released")
                time.sleep(0.01)
        return table

    class Model:
        def __call__(self, table):
            return process(table)

    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    with vane.connect() as connection, connection.cursor() as other, connection.cursor() as later:
        cursors = [connection, other, later]
        plans = [
            _plan(cursor, Model if actor else process, index + 1, actor=actor) for index, cursor in enumerate(cursors)
        ]
        runtime = LocalModelRuntime(
            session_id=plans[0].session_id(),
            session_config=plans[0].session_config(),
            request_limit=RequestAdmissionLimits(2, 1),
            task_limit=TaskAdmissionLimits(2, 2) if task_limited else None,
            track_data=accounting == "tracked",
            data_limit=DataAdmissionLimits(8192, 1024, 1024) if accounting == "limited" else None,
        )
        bindings = [{} for _ in plans]
        model = None
        if actor:
            model = runtime.register(
                "model", version="v1", payload=plans[0].collect_udf_nodes(conn=connection)[0]["payload"]
            )
            model.prewarm()
            bindings = [
                {str(node["node_id"]): "model" for node in plan.collect_udf_nodes(conn=cursor)}
                for plan, cursor in zip(plans, cursors)
            ]
        first, second, queued = runtime.request(), runtime.request(), runtime.request()
        try:
            with ThreadPoolExecutor(max_workers=2) as threads:
                futures = [threads.submit(first.execute, plans[0], bindings[0], conn=connection)]
                try:
                    _wait(lambda: (tmp_path / "entered-1").exists(), "first worker did not start")
                    futures.append(threads.submit(second.execute, plans[1], bindings[1], conn=other))
                    # Constant task plans share one physical worker slot. The
                    # second query must survive cancellation while queued;
                    # two-actor models exercise a concurrently running peer.
                    if actor:
                        _wait(lambda: (tmp_path / "entered-2").exists(), "second actor did not start")
                    else:
                        from vane.execution.udf_subprocess import _global_task_runtime

                        task_runtime = _global_task_runtime()

                        def shared_pool_attached():
                            with task_runtime.cond:
                                return any(pool.ref_count == 2 for pool in task_runtime.pools.values())

                        _wait(shared_pool_attached, "second query did not attach to the shared task pool")
                    assert first.cancel()
                    with pytest.raises(RequestCancelled):
                        futures[0].result(timeout=15)

                    def other_entered():
                        if futures[1].done():
                            futures[1].result()
                        return (tmp_path / "entered-2").exists()

                    _wait(other_entered, "second query did not survive cancellation")
                    surviving_pid = int((tmp_path / "entered-2").read_text())
                    assert not first.cancel()
                    assert not futures[1].done()
                    first.shutdown()
                    assert first.state == "cancelled"
                    assert queued.state == "ready"
                    if not actor:
                        (tmp_path / "release-2").touch()
                        assert _values(futures[1].result(timeout=15)) == [2]
                    assert _values(queued.execute(plans[2], bindings[2], conn=later)) == [3]
                    if model is not None:
                        with model.acquire() as borrow:
                            assert surviving_pid in borrow.pool.worker_pids()
                    (tmp_path / "release-2").touch()
                    assert _values(futures[1].result(timeout=15)) == [2]
                finally:
                    (tmp_path / "release-1").touch()
                    (tmp_path / "release-2").touch()
                    first.cancel()
                    second.cancel()
            assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
            assert manager.snapshot()["usage_bytes"] == 0
        finally:
            runtime.close(timeout=15, kill=True)


@pytest.mark.parametrize("before_start", [False, True])
def test_cancel_native_sql_at_start_and_reuse_cursor(monkeypatch, before_start):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    entered, proceed = threading.Event(), threading.Event()
    original = udf_local_request._NativeRequestCancellation.started

    def started(self, conn):
        if not before_start:
            original(self, conn)
        entered.set()
        assert proceed.wait(5)
        if before_start:
            original(self, conn)

    monkeypatch.setattr(udf_local_request._NativeRequestCancellation, "started", started)
    with vane.connect() as connection:
        relation = connection.sql("SELECT sum(i) AS x FROM range(1000000000000) t(i)")
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(connection)
        with LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            request_limit=RequestAdmissionLimits(1, 1),
        ) as runtime:
            request = runtime.request()
            with ThreadPoolExecutor(max_workers=1) as threads:
                future = threads.submit(request.execute, plan, {}, conn=connection)
                try:
                    assert entered.wait(5)
                    assert request.cancel()
                finally:
                    proceed.set()
                with pytest.raises(RequestCancelled):
                    future.result(timeout=5)
            assert not request.cancel()
            assert connection.sql("SELECT 42").fetchall() == [(42,)]
            assert request.state == "cancelled"


def test_native_start_callback_validation_and_failure_preserve_connection(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as connection:

        def plan():
            relation = connection.sql("SELECT 7")
            return vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(
                connection
            )

        runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
        with pytest.raises(TypeError, match="must be callable"):
            runner.execute_native(connection, plan(), native_execution_started=42)

        def failed(conn):
            assert conn is connection
            raise ValueError("start callback failed")

        with pytest.raises(ValueError, match="start callback failed"):
            runner.execute_native(connection, plan(), native_execution_started=failed)
        calls = []
        assert _values(runner.execute_native(connection, plan(), native_execution_started=calls.append)) == [7]
        assert calls == [connection]


@pytest.mark.parametrize("actor", [False, True])
def test_cancel_native_task_admission_wait_does_not_interrupt_active_request(monkeypatch, tmp_path, actor):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    marker = str(tmp_path)

    def process(table):
        from pathlib import Path

        value = table.column(0)[0].as_py()
        Path(marker, f"entered-{value}").touch()
        if value == 1:
            deadline = time.monotonic() + 25
            while not Path(marker, "release").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("test worker was not released")
                time.sleep(0.01)
        return table

    class Model:
        def __call__(self, table):
            return process(table)

    with vane.connect() as first_conn, first_conn.cursor() as second_conn:
        plans = [
            _plan(cursor, Model if actor else process, index, actor=actor)
            for index, cursor in enumerate((first_conn, second_conn), 1)
        ]
        runtime = LocalModelRuntime(
            session_id=plans[0].session_id(),
            session_config=plans[0].session_config(),
            request_limit=RequestAdmissionLimits(2, 1),
            task_limit=TaskAdmissionLimits(1, 2),
        )
        first, second = runtime.request(), runtime.request()
        try:
            with ThreadPoolExecutor(max_workers=2) as threads:
                active = threads.submit(first.execute, plans[0], {}, conn=first_conn)
                waiting = None
                try:
                    _wait(lambda: (tmp_path / "entered-1").exists(), "first request did not enter")
                    waiting = threads.submit(second.execute, plans[1], {}, conn=second_conn)
                    _wait(
                        lambda: runtime.resource_snapshot()["task_admission"]["queued_tasks"] == 1,
                        "second request did not queue",
                    )
                    assert second.cancel()
                    with pytest.raises(RequestCancelled):
                        waiting.result(timeout=10)
                    assert not (tmp_path / "entered-2").exists()
                    assert not active.done()
                    assert runtime.resource_snapshot()["task_admission"]["queued_tasks"] == 0
                finally:
                    (tmp_path / "release").touch()
                assert _values(active.result(timeout=15)) == [1]
        finally:
            runtime.close(timeout=15, kill=True)


@pytest.mark.parametrize("actor", [False, True])
@pytest.mark.parametrize("track_data", [False, True])
def test_cancel_native_output_wait_keeps_other_consumers_bytes(monkeypatch, actor, track_data):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    held = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": list(range(8192))}))
    held_bytes = manager.snapshot()["usage_bytes"]

    def process(table):
        return pa.table({"x": [7] * 8192})

    class Model:
        def __call__(self, table):
            return process(table)

    try:
        with vane.connect() as connection:
            plan = _plan(connection, Model if actor else process, 1, actor=actor)
            with LocalModelRuntime(
                session_id=plan.session_id(),
                session_config=plan.session_config(),
                request_limit=RequestAdmissionLimits(1, 1),
                task_limit=TaskAdmissionLimits(1, 1),
                track_data=track_data,
            ) as runtime:
                request = runtime.request()
                with ThreadPoolExecutor(max_workers=1) as threads:
                    future = threads.submit(request.execute, plan, {}, conn=connection)
                    try:
                        _wait(lambda: manager.snapshot()["waiting_output_grants"] > 0, "output grant did not wait")
                        assert request.cancel()
                        with pytest.raises(RequestCancelled):
                            future.result(timeout=15)
                    finally:
                        request.cancel()
                request.shutdown()
                assert request.state == "cancelled"
                assert manager.snapshot()["usage_bytes"] == held_bytes
                assert manager.snapshot()["waiting_output_grants"] == 0
    finally:
        for owner in held[1]:
            owner.release()
    assert manager.snapshot()["usage_bytes"] == 0


@pytest.mark.parametrize("accounting", ["default", "tracked", "limited"])
@pytest.mark.parametrize("actor", [False, True])
def test_cancel_retains_failed_output_grant_cleanup(monkeypatch, accounting, actor):
    from vane.execution import udf_subprocess

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    entered, proceed = threading.Event(), threading.Event()
    send = udf_subprocess._send_message

    def failed_delivery(sock, kind, payload=b""):
        if kind == udf_subprocess._MSG_OUTPUT_GRANT_GRANTED:
            entered.set()
            assert proceed.wait(10)
            raise OSError("cancelled grant delivery")
        return send(sock, kind, payload)

    def failed_cleanup(*args, **kwargs):
        raise OSError("cancelled grant cleanup")

    class Model:
        def __call__(self, table):
            return table

    with vane.connect() as connection:
        plan = _plan(connection, Model if actor else lambda table: table, 1, actor=actor)
        runtime = LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            request_limit=RequestAdmissionLimits(1, 1),
            track_data=accounting == "tracked",
            data_limit=DataAdmissionLimits(4096, 1024, 1024) if accounting == "limited" else None,
        )
        request, queued = runtime.request(), runtime.request()
        try:
            with monkeypatch.context() as fault:
                fault.setattr(udf_subprocess, "_send_message", failed_delivery)
                fault.setattr(manager, "release_output_grant", failed_cleanup)
                with ThreadPoolExecutor(max_workers=2) as threads:
                    future = threads.submit(request.execute, plan, {}, conn=connection)
                    cancellation = None
                    try:
                        assert entered.wait(10)
                        cancellation = threads.submit(request.cancel)
                        _wait(lambda: request.state == "cancelling", "cancellation not accepted")
                    finally:
                        proceed.set()
                    assert cancellation.result(timeout=10)
                    with pytest.raises(RequestCancelled):
                        future.result(timeout=15)
                assert manager.snapshot()["output_grant_bytes"] > 0
                assert request.state == "cancelling" and queued.state == "queued"
                with pytest.raises(RuntimeError, match="request cleanup failed"):
                    request.shutdown()
                if accounting == "limited":
                    assert runtime.resource_snapshot()["data"]["usage_bytes"] > 0
            request.shutdown(kill=True)
            assert request.state == "cancelled" and queued.state == "ready"
            assert manager.snapshot()["usage_bytes"] == 0
        finally:
            proceed.set()
            runtime.close(timeout=15, kill=True)


def test_cancel_mixed_native_pipeline_and_reuse_registered_model(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    marker = str(tmp_path / "entered")
    release = str(tmp_path / "release")

    def produce(table):
        return table

    class Model:
        def __call__(self, table):
            from pathlib import Path

            if table.column(0)[0].as_py() == 1:
                Path(marker).touch()
                deadline = time.monotonic() + 20
                while not Path(release).exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("test model was not released")
                    time.sleep(0.01)
            return table

    with vane.connect() as connection:

        def plan(value):
            relation = (
                connection.sql(f"SELECT {value}::BIGINT AS x")
                .map_batches(produce, schema={"x": vane.sqltypes.BIGINT}, execution_backend="subprocess_task")
                .map_batches(
                    Model, schema={"x": vane.sqltypes.BIGINT}, execution_backend="subprocess_actor", actor_number=1
                )
            )
            physical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(
                connection
            )
            model_node = next(
                node
                for node in physical.collect_udf_nodes(conn=connection)
                if node["payload"]["execution_backend"] == "subprocess_actor"
            )
            return physical, model_node

        first_plan, node = plan(1)
        with LocalModelRuntime(
            session_id=first_plan.session_id(),
            session_config=first_plan.session_config(),
            request_limit=RequestAdmissionLimits(1, 1),
            task_limit=TaskAdmissionLimits(1, 2),
            data_limit=DataAdmissionLimits(8192, 1024, 1024),
        ) as runtime:
            runtime.register("model", version="v1", payload=node["payload"])
            request = runtime.request()
            with ThreadPoolExecutor(max_workers=1) as threads:
                future = threads.submit(request.execute, first_plan, {str(node["node_id"]): "model"}, conn=connection)
                try:
                    _wait(lambda: (tmp_path / "entered").exists(), "mixed pipeline did not reach the model")
                    assert request.cancel()
                    with pytest.raises(RequestCancelled):
                        future.result(timeout=15)
                finally:
                    (tmp_path / "release").touch()
                    request.cancel()
            next_plan, node = plan(2)
            assert _values(runtime.request().execute(next_plan, {str(node["node_id"]): "model"}, conn=connection)) == [
                2
            ]
            state = runtime.resource_snapshot()
            assert state["active_borrows"] == state["request_admission"]["active_requests"] == 0
            assert state["task_admission"]["running_tasks"] == state["data"]["usage_bytes"] == 0
