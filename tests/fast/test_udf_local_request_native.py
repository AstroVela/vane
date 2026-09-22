# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

import vane
from vane.execution import ref_bundle
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


@pytest.mark.parametrize("mode", ["default", "tracked", "byte_limited", "unit_limited"])
@pytest.mark.parametrize("task_limited", [False, True])
@pytest.mark.parametrize("cleanup", ["request", "runtime"])
@pytest.mark.parametrize("actor", [False, True])
def test_failed_output_grant_cleanup_retains_native_request(monkeypatch, mode, task_limited, cleanup, actor):
    from vane.execution import udf_subprocess as local

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)

    class Identity:
        def __call__(self, table):
            return table

    function = Identity if actor else lambda table: table
    send = local._send_message
    failures = set()

    def fail_delivery(sock, msg_type, payload=b""):
        if msg_type == local._MSG_OUTPUT_GRANT_GRANTED:
            failures.add("delivery")
            raise OSError("injected output grant delivery failure")
        return send(sock, msg_type, payload)

    def fail_cleanup(*args, **kwargs):
        failures.add("cleanup")
        raise OSError("injected output grant cleanup failure")

    with vane.connect() as connection:
        plan = _plan(connection, function, 7, actor=actor)
        runtime = LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            request_limit=RequestAdmissionLimits(1, 1),
            track_data=mode == "tracked",
            data_limit=DataAdmissionLimits(
                100_000, 1024, 70_000, unit_reservation_ratio=0.5 if mode == "unit_limited" else None
            )
            if mode in {"byte_limited", "unit_limited"}
            else None,
            task_limit=TaskAdmissionLimits(1, 1) if task_limited else None,
        )
        request, queued = runtime.request(), runtime.request()
        try:
            with monkeypatch.context() as fault:
                fault.setattr(local, "_send_message", fail_delivery)
                fault.setattr(manager, "release_output_grant", fail_cleanup)
                with pytest.raises(Exception, match="output grant (delivery|cleanup) failure"):
                    request.execute(plan, {}, conn=connection)
                gc.collect()
                assert failures == {"delivery", "cleanup"}
                assert manager.snapshot()["active_input_leases"] == 0
                assert manager.snapshot()["output_grant_bytes"] > 0
                assert request.state == "running"
                assert queued.state == "queued"
                assert runtime.resource_snapshot()["request_admission"]["cleanup_pending_requests"] == 1
                if mode == "unit_limited":
                    data = runtime.resource_snapshot()["data"]
                    assert data["unit_budget"]["usage_bytes"] == data["usage_bytes"] > 0
                    assert data["unit_budget"]["inactive_usage_bytes"] == data["usage_bytes"]
                    assert request.resource_graph_snapshot() is not None
                if cleanup == "request":
                    with pytest.raises(RuntimeError, match="request cleanup failed"):
                        request.shutdown()
                    assert queued.state == "queued"
                else:
                    with pytest.raises(RuntimeError, match="cleanup failed during runtime close"):
                        runtime.close()
                assert request.state == "running"
                assert manager.snapshot()["output_grant_bytes"] > 0
            if cleanup == "request":
                request.shutdown()
                assert queued.state == "ready"
                fresh = _plan(connection, function, 8, actor=actor)
                assert _values(queued.execute(fresh, {}, conn=connection)) == [8]
            else:
                runtime.close(timeout=5)
                assert queued.state == "drained"
            gc.collect()
            assert request.state == "finished"
            assert manager.snapshot()["usage_bytes"] == 0
            assert runtime.resource_snapshot()["request_admission"]["cleanup_pending_requests"] == 0
        finally:
            request.shutdown(kill=True)
            queued.shutdown()
            runtime.close(timeout=5, kill=True)
            for grant_id in list(manager._output_grants):
                manager.release_output_grant(grant_id)


@pytest.mark.parametrize("track_data", [False, True])
@pytest.mark.parametrize("task_limited", [False, True])
@pytest.mark.parametrize("cleanup", ["request", "runtime"])
@pytest.mark.parametrize("actor", [False, True])
def test_failed_input_cleanup_retains_native_request(monkeypatch, track_data, task_limited, cleanup, actor):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)

    class Identity:
        def __call__(self, table):
            return table

    function = Identity if actor else lambda table: table
    with vane.connect() as connection:
        plan = _plan(connection, function, 7, actor=actor)
        runtime = LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            request_limit=RequestAdmissionLimits(1, 1),
            track_data=track_data,
            task_limit=TaskAdmissionLimits(1, 1) if task_limited else None,
        )
        request, queued = runtime.request(), runtime.request()
        try:
            with monkeypatch.context() as fault:

                def fail_cleanup(*args, **kwargs):
                    raise RuntimeError("injected request input cleanup failure")

                fault.setattr(manager, "_release_input_ack_ref", fail_cleanup)
                with pytest.raises(Exception, match="input cleanup failure"):
                    request.execute(plan, {}, conn=connection)
                gc.collect()
                assert manager.snapshot()["active_input_leases"] == 1
                assert request.state == "running"
                assert queued.state == "queued"
                state = runtime.resource_snapshot()
                assert ("data" in state) == track_data
                assert state["request_admission"]["cleanup_pending_requests"] == 1
                if cleanup == "request":
                    with pytest.raises(RuntimeError, match="request cleanup failed"):
                        request.shutdown()
                    assert queued.state == "queued"
                else:
                    with pytest.raises(RuntimeError, match="cleanup failed during runtime close"):
                        runtime.close()
                assert request.state == "running"
                assert manager.snapshot()["active_input_leases"] == 1
            if cleanup == "request":
                request.shutdown()
                assert queued.state == "ready"
                fresh = _plan(connection, function, 8, actor=actor)
                assert _values(queued.execute(fresh, {}, conn=connection)) == [8]
            else:
                runtime.close(timeout=5)
                assert queued.state == "drained"
            assert request.state == "finished"
            assert manager.snapshot()["active_input_leases"] == 0
            assert runtime.resource_snapshot()["request_admission"]["cleanup_pending_requests"] == 0
        finally:
            request.shutdown(kill=True)
            queued.shutdown()
            runtime.close(timeout=5, kill=True)


def test_untracked_request_owns_input_cleanup_during_payload_setup(monkeypatch):
    from vane.execution import udf_subprocess as local

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    make_payload = local.make_local_ref_bundle_worker_payload

    def fail(*args, **kwargs):
        raise RuntimeError("injected request setup or cleanup failure")

    def fail_descriptor(*args, **kwargs):
        if kwargs.get("input_lease_id") is not None:
            fail()
        return make_payload(*args, **kwargs)

    with vane.connect() as connection:
        plan = _plan(connection, lambda table: table, 7, actor=False)
        runtime = LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            request_limit=RequestAdmissionLimits(1, 1),
            task_limit=TaskAdmissionLimits(1, 1),
        )
        request, queued = runtime.request(), runtime.request()
        try:
            with monkeypatch.context() as fault:
                fault.setattr(manager, "_release_input_ack_ref", fail)
                fault.setattr(local, "make_local_ref_bundle_worker_payload", fail_descriptor)
                with pytest.raises(Exception, match="setup or cleanup failure"):
                    request.execute(plan, {}, conn=connection)
                gc.collect()
                assert manager.snapshot()["active_input_leases"] == 1
                assert request.state == "running"
                assert queued.state == "queued"
            request.shutdown()
            assert request.state == "finished"
            assert queued.state == "ready"
            assert manager.snapshot()["active_input_leases"] == 0
        finally:
            request.shutdown(kill=True)
            queued.shutdown()
            runtime.close(timeout=5, kill=True)


@pytest.mark.parametrize("track_data", [False, True])
@pytest.mark.parametrize("task_limited", [False, True])
@pytest.mark.parametrize("actor", [False, True])
def test_native_preparation_error_survives_input_cleanup(monkeypatch, track_data, task_limited, actor):
    from vane.execution import udf_subprocess as local

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    make_payload, release_input = local.make_local_ref_bundle_worker_payload, manager._release_input_ack_ref
    armed = threading.Event()

    class Identity:
        def __call__(self, table):
            return table

    def fail_descriptor(*args, **kwargs):
        # The downstream UDF prepares its ref bundle on the native dispatcher;
        # the upstream worker's materialized-input conversion must succeed.
        if kwargs.get("input_lease_id") is not None and not threading.current_thread().name.startswith(
            "vane-udf-subprocess"
        ):
            armed.set()
            raise ValueError("primary downstream input preparation error")
        return make_payload(*args, **kwargs)

    def fail_cleanup(ref):
        if armed.is_set():
            raise OSError("secondary downstream input cleanup error")
        return release_input(ref)

    with vane.connect() as connection:
        relation = connection.sql("SELECT 7::BIGINT AS x")
        for _ in range(2):
            relation = relation.map_batches(
                Identity if actor else lambda table: table,
                schema={"x": vane.sqltypes.BIGINT},
                execution_backend="subprocess_actor" if actor else "subprocess_task",
                actor_number=1 if actor else None,
            )
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(connection)
        runtime = LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            request_limit=RequestAdmissionLimits(1, 1),
            track_data=track_data,
            task_limit=TaskAdmissionLimits(1, 1) if task_limited else None,
        )
        request, queued = runtime.request(), runtime.request()
        try:
            with monkeypatch.context() as fault:
                fault.setattr(local, "make_local_ref_bundle_worker_payload", fail_descriptor)
                fault.setattr(manager, "_release_input_ack_ref", fail_cleanup)
                with pytest.raises(Exception, match="primary downstream input preparation error") as info:
                    request.execute(plan, {}, conn=connection)
                assert armed.is_set()
                assert "request cleanup failed" in str(info.value.__cause__)
                assert request.state == "running" and queued.state == "queued"
                assert manager.snapshot()["active_input_leases"] == 1
            request.shutdown()
            assert request.state == "finished" and queued.state == "ready"
            assert manager.snapshot()["active_input_leases"] == 0
        finally:
            request.shutdown(kill=True)
            queued.shutdown()
            runtime.close(timeout=5, kill=True)


@pytest.mark.parametrize("operation", ["acquire", "prewarm"])
@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("request_limited", [False, True])
def test_returned_model_handle_obeys_drain(monkeypatch, tmp_path, operation, resident, request_limited):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    marker = str(tmp_path / "constructed")

    class Model:
        def __init__(self):
            from pathlib import Path

            Path(marker).touch()

        def __call__(self, table):
            return table

    with vane.connect() as connection:
        plan = _plan(connection, Model, 1)
        runtime = LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            request_limit=RequestAdmissionLimits(1, 1) if request_limited else None,
        )
        model = runtime.register("model", version="v1", payload=plan.collect_udf_nodes(conn=connection)[0]["payload"])
        borrow = None
        try:
            if resident:
                model.prewarm()
            runtime.drain()
            with pytest.raises(RuntimeError, match="draining"):
                borrow = getattr(model, operation)()
            assert (tmp_path / "constructed").exists() == resident
            assert runtime.resource_snapshot()["active_borrows"] == 0
        finally:
            if borrow is not None:
                borrow.release()
            runtime.close(timeout=5, kill=True)


@pytest.mark.parametrize("fence", ["drain", "close"])
@pytest.mark.parametrize("resident", [False, True])
def test_claimed_request_prepares_after_drain_without_authorizing_public_handles(
    monkeypatch, tmp_path, fence, resident
):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    marker = str(tmp_path / "constructors")

    class Model:
        def __init__(self):
            from pathlib import Path

            with Path(marker).open("a") as output:
                output.write(f"{os.getpid()}\n")

        def __call__(self, table):
            return table

    with vane.connect() as connection:
        plan = _plan(connection, Model, 7)
        runtime = LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            request_limit=RequestAdmissionLimits(1, 1),
            task_limit=TaskAdmissionLimits(1, 1),
            track_data=True,
        )
        model = runtime.register("model", version="v1", payload=plan.collect_udf_nodes(conn=connection)[0]["payload"])
        entered, proceed = threading.Event(), threading.Event()
        prepare = runtime._prepare

        def blocked_prepare(*args, **kwargs):
            entered.set()
            assert proceed.wait(10)
            return prepare(*args, **kwargs)

        monkeypatch.setattr(runtime, "_prepare", blocked_prepare)
        try:
            if resident:
                model.prewarm()
            request, queued = runtime.request(), runtime.request()
            with ThreadPoolExecutor(max_workers=1) as threads:
                future = threads.submit(request.execute, plan, _bindings(plan, connection), conn=connection)
                try:
                    assert entered.wait(5)
                    assert request.state == "running"
                    if fence == "close":
                        with pytest.raises(TimeoutError, match="active execution"):
                            runtime.close()
                    else:
                        runtime.drain()
                    assert queued.state == "drained"
                    for operation in (model.acquire, model.prewarm):
                        borrow = None
                        try:
                            with pytest.raises(RuntimeError, match="draining"):
                                borrow = operation()
                        finally:
                            if borrow is not None:
                                borrow.release()
                    assert (tmp_path / "constructors").exists() == resident
                finally:
                    proceed.set()
                assert _values(future.result(timeout=15)) == [7]
            assert len((tmp_path / "constructors").read_text().splitlines()) == 1
            assert runtime.resource_snapshot()["active_borrows"] == 0
            assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
            runtime.close(timeout=5)
        finally:
            proceed.set()
            runtime.close(timeout=5, kill=True)


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
