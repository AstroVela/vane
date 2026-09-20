# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import time
import uuid

import pyarrow as pa
import pytest

import vane
from vane import pickle as vane_pickle
from vane.execution import ref_bundle
from vane.execution.udf import build_executor
from vane.execution.udf_data_admission import DataAdmissionCapacityError, DataAdmissionLimits
from vane.execution.udf_local_model import LocalModelRuntime
from vane.execution.udf_runtime_admission import TaskAdmissionLimits


def _wait_result(executor):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = executor.take_ready_result()
        if result is not None:
            return result
        time.sleep(0.01)
    raise TimeoutError("byte-admitted subprocess did not finish")


class _Plan:
    def __init__(self, payload):
        self.nodes = [{"node_id": "one", "payload": payload}]

    def session_id(self):
        return "test"

    def session_config(self):
        return {}

    def collect_udf_nodes(self, conn=None):
        return self.nodes

    def set_udf_actor_handles(self, options, conn=None):
        self.options = options["one"]


@pytest.fixture
def strict_transport(monkeypatch):
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    yield manager
    assert manager.snapshot()["task_reserved_bytes"] == 0
    assert manager.snapshot()["usage_bytes"] == 0


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
@pytest.mark.parametrize("limited", [False, True])
def test_retained_view_refuses_new_work_then_retries_with_shared_model_or_pool(strict_transport, backend, limited):
    def expand(table):
        return pa.table({"x": [b"x" * 65536]})

    class Expand:
        def __call__(self, table):
            return expand(table)

    payload = dict(
        function_pickle=vane_pickle.dumps(Expand if backend == "subprocess_actor" else expand),
        call_mode="map_batches",
        execution_backend=backend,
        actor_number=1,
        udf_worker_slots=1,
        produce_ref_bundle_output=True,
        streaming_output_mode="local_shm_ref_bundle",
    )
    runtime = LocalModelRuntime(
        session_id="test",
        session_config={},
        data_limit=DataAdmissionLimits(100_000, 1024, 70_000),
        task_limit=TaskAdmissionLimits(1, 4) if limited else None,
    )
    if backend == "subprocess_actor":
        runtime.register("expand", version="1", payload=payload)
    executors, resources, refs = [], [], []
    try:
        for _ in range(2):
            plan = _Plan(payload)
            resources.extend(runtime.prepare(plan, {"one": "expand"} if backend == "subprocess_actor" else {}))
            executors.append(build_executor(payload, plan.options))
        first, second = executors
        if backend == "subprocess_task":
            assert first._task_pool is second._task_pool
        assert first.request_task_admission(8)
        assert runtime.resource_snapshot()["data"]["reserved_bytes"] == 71_024
        first.submit(pa.table({"x": [1]}))
        result = _wait_result(first)
        assert not isinstance(result, BaseException), str(result)
        refs.extend(result[1])
        view = result[1][0].to_table().column(0)
        for ref in refs:
            ref.release()
        refs.clear()
        data = runtime.resource_snapshot()["data"]
        assert data["reserved_bytes"] == 0
        assert 65_536 < data["retained_bytes"] < 70_000
        # The transport ACK can return its credit, but the consumer view keeps
        # the runtime bytes. No worker may wait while holding a second grant.
        with pytest.raises(DataAdmissionCapacityError, match="runtime"):
            second.request_task_admission(8)
        assert second.task_admission_state()["state"] == "idle"
        assert view[0].as_py() == b"x" * 65536
        del view
        gc.collect()
        assert runtime.resource_snapshot()["data"]["usage_bytes"] == 0
        second.request_task_admission(8)
        second.submit(pa.table({"x": [2]}))
        result = _wait_result(second)
        assert not isinstance(result, BaseException), str(result)
        refs.extend(result[1])
    finally:
        for executor in executors:
            executor.close(kill=True)
        for resource in resources:
            resource.shutdown(kill=True)
        for ref in refs:
            ref.release()
        runtime.close(timeout=5, kill=True)
    assert runtime.resource_snapshot()["data"]["usage_bytes"] == 0
    assert strict_transport.snapshot()["waiting_output_grants"] == 0


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_failed_grant_delivery_and_cleanup_keep_runtime_owned(strict_transport, monkeypatch, backend):
    from vane.execution import udf_subprocess as local

    def identity(table):
        return table

    class Identity:
        def __call__(self, table):
            return table

    payload = dict(
        function_pickle=vane_pickle.dumps(Identity if backend == "subprocess_actor" else identity),
        call_mode="map_batches",
        execution_backend=backend,
        actor_number=1,
        udf_worker_slots=1,
        produce_ref_bundle_output=True,
        streaming_output_mode="local_shm_ref_bundle",
    )
    runtime = LocalModelRuntime(
        session_id="test",
        session_config={},
        data_limit=DataAdmissionLimits(100_000, 1024, 70_000),
        task_limit=TaskAdmissionLimits(1, 4),
    )
    plan = _Plan(payload)
    resources = runtime.prepare(plan, {})
    executor = build_executor(payload, plan.options)
    send = local._send_message

    def fail_delivery(sock, msg_type, payload=b""):
        if msg_type == local._MSG_OUTPUT_GRANT_GRANTED:
            raise RuntimeError("planned grant delivery failure")
        return send(sock, msg_type, payload)

    def fail_cleanup(*args, **kwargs):
        raise RuntimeError("planned grant cleanup failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(local, "_send_message", fail_delivery)
            patch.setattr(strict_transport, "release_output_grant", fail_cleanup)
            executor.request_task_admission(8)
            executor.submit(pa.table({"x": [1]}))
            assert executor._wait_for_pending_futures(15)
            grant_bytes = strict_transport.snapshot()["output_grant_bytes"]
            assert grant_bytes > 0
            data = runtime.resource_snapshot()["data"]
            assert data["tasks"] == 0
            assert data["reservations"] == 1
            assert data["usage_bytes"] >= grant_bytes
            with pytest.raises(RuntimeError, match="planned grant cleanup failure"):
                resources[-1].shutdown()
            with pytest.raises(TimeoutError, match="active queries or tasks"):
                runtime.close()
    finally:
        executor.close(kill=True)
        for resource in resources:
            try:
                resource.shutdown(kill=True)
            except RuntimeError as exc:
                # An actor pool reports the earlier worker cleanup failure on
                # its first shutdown, even when the explicit retry can finish.
                assert backend == "subprocess_actor" and "planned grant cleanup failure" in str(exc)
                resource.shutdown(kill=True)
        runtime.close(timeout=5, kill=True)
    assert runtime.resource_snapshot()["data"]["usage_bytes"] == 0
    assert strict_transport.snapshot()["output_grant_bytes"] == 0


@pytest.mark.parametrize("limited", [False, True])
def test_budget_wakeup_preserves_retryable_refusal(strict_transport, limited):
    payload = dict(
        function_pickle=vane_pickle.dumps(lambda table: table),
        call_mode="map_batches",
        execution_backend="subprocess_task",
        udf_worker_slots=1,
        produce_ref_bundle_output=True,
        streaming_output_mode="local_shm_ref_bundle",
    )
    runtime = LocalModelRuntime(
        session_id="test",
        session_config={},
        data_limit=DataAdmissionLimits(71_024, 1024, 70_000),
        task_limit=TaskAdmissionLimits(1, 4) if limited else None,
    )
    executors, resources, refs = [], [], []
    try:
        for _ in range(2):
            plan = _Plan(payload)
            resources.extend(runtime.prepare(plan, {}))
            executors.append(build_executor(payload, plan.options))
        first, second = executors
        second.register_wakeup(second.task_admission_state)
        first.request_task_admission(8)
        first.submit(pa.table({"x": [1]}))
        second.request_task_admission(8)
        assert second.task_admission_state()["state"] == "requested"
        result = _wait_result(first)
        assert not isinstance(result, BaseException)
        refs.extend(result[1])
        # A transport notification can arrive after the pool callback has
        # recorded a byte refusal but before the dispatcher reads it.
        strict_transport.wake_waiters()
        with pytest.raises(DataAdmissionCapacityError):
            second.task_admission_state()
        second.stats()  # The executor must not cache it as a wakeup failure.
        for ref in refs:
            ref.release()
        refs.clear()
        second.request_task_admission(8)
        second.submit(pa.table({"x": [2]}))
        result = _wait_result(second)
        assert not isinstance(result, BaseException)
        refs.extend(result[1])
    finally:
        for executor in executors:
            executor.close(kill=True)
        for resource in resources:
            resource.shutdown(kill=True)
        for ref in refs:
            ref.release()
        runtime.close(timeout=5, kill=True)
    assert strict_transport.snapshot()["waiting_output_grants"] == 0


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
@pytest.mark.parametrize("limited", [False, True])
def test_native_multistage_plan_completes_with_output_reservations(monkeypatch, backend, limited):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 300_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)

    def expand(table):
        return pa.table({"blob": [b"x" * 65536]})

    class Expand:
        def __call__(self, table):
            return expand(table)

    def consume(table):
        return pa.table({"size": [len(table.column(0)[0].as_py())]})

    with vane.connect() as connection:
        relation = (
            connection.sql("SELECT 1::INTEGER AS x")
            .map_batches(
                Expand if backend == "subprocess_actor" else expand,
                schema={"blob": vane.sqltypes.BLOB},
                execution_backend=backend,
                **({"actor_number": 1} if backend == "subprocess_actor" else {}),
            )
            .map_batches(consume, schema={"size": vane.sqltypes.BIGINT}, execution_backend="subprocess_task")
        )
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(connection)
        runtime = LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            data_limit=DataAdmissionLimits(300_000, 70_000, 70_000),
            task_limit=TaskAdmissionLimits(1, 8) if limited else None,
        )
        resources = runtime.prepare(plan, {}, conn=connection)
        try:
            result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(connection, plan)
            assert [row for table in result.partition_payloads for row in table.column(0).to_pylist()] == [65_536]
        finally:
            for resource in resources:
                resource.shutdown(kill=True)
            runtime.close(timeout=5, kill=True)
        del result
        gc.collect()
        assert runtime.resource_snapshot()["data"]["usage_bytes"] == 0
        assert manager.snapshot()["waiting_output_grants"] == manager.snapshot()["usage_bytes"] == 0


@pytest.mark.parametrize("oversized", ["input", "output"])
def test_native_oversized_batch_fails_promptly_and_returns_every_reservation(monkeypatch, tmp_path, oversized):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    marker = str(tmp_path / "called")

    def execute(table):
        from pathlib import Path

        Path(marker).touch()
        return pa.table({"blob": [b"x" * 65536]})

    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 300_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    with vane.connect() as connection:
        source = connection.sql("SELECT repeat('a', 65536) AS x" if oversized == "input" else "SELECT 1 AS x")
        relation = source.map_batches(execute, schema={"blob": vane.sqltypes.BLOB}, execution_backend="subprocess_task")
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(connection)
        runtime = LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            data_limit=DataAdmissionLimits(100_000, 1024, 1024),
            task_limit=TaskAdmissionLimits(1, 4),
        )
        resources = runtime.prepare(plan, {}, conn=connection)
        try:
            with pytest.raises(Exception, match=f"{oversized} batch exceeds data limit"):
                vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(connection, plan)
        finally:
            for resource in resources:
                resource.shutdown(kill=True)
            runtime.close(timeout=5, kill=True)
    assert (tmp_path / "called").exists() == (oversized == "output")
    assert runtime.resource_snapshot()["data"]["usage_bytes"] == 0
    assert runtime.resource_snapshot()["task_admission"]["running_tasks"] == 0
    assert manager.snapshot()["usage_bytes"] == 0


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
@pytest.mark.parametrize("failure", ["udf", "worker_exit", "cancel", "unused_ready"])
def test_subprocess_failure_returns_input_output_and_execution_reservations(
    strict_transport, tmp_path, backend, failure
):
    entered = str(tmp_path / "entered")

    def fail(table):
        import os
        from pathlib import Path

        Path(entered).touch()
        if failure == "worker_exit":
            os._exit(19)
        if failure == "udf":
            raise ValueError("planned byte-admission failure")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            time.sleep(0.01)
        raise TimeoutError("cancellation did not stop worker")

    class Fail:
        def __call__(self, table):
            return fail(table)

    payload = dict(
        function_pickle=vane_pickle.dumps(Fail if backend == "subprocess_actor" else fail),
        call_mode="map_batches",
        execution_backend=backend,
        actor_number=1,
        udf_worker_slots=1,
        produce_ref_bundle_output=True,
        streaming_output_mode="local_shm_ref_bundle",
    )
    runtime = LocalModelRuntime(
        session_id="test",
        session_config={},
        data_limit=DataAdmissionLimits(100_000, 1024, 70_000),
        task_limit=TaskAdmissionLimits(1, 4),
    )
    plan = _Plan(payload)
    resources = runtime.prepare(plan, {})
    executor = build_executor(payload, plan.options)
    try:
        executor.request_task_admission(8)
        assert runtime.resource_snapshot()["data"]["reserved_bytes"] == 71_024
        if failure != "unused_ready":
            executor.submit(pa.table({"x": [1]}))
            if failure == "cancel":
                deadline = time.monotonic() + 10
                while not (tmp_path / "entered").exists():
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                assert runtime.resource_snapshot()["data"]["input_bytes"] > 0
            else:
                assert isinstance(_wait_result(executor), BaseException)
    finally:
        executor.close(kill=True)
        for resource in resources:
            resource.shutdown(kill=True)
        runtime.close(timeout=5, kill=True)
    data = runtime.resource_snapshot()["data"]
    assert data["tasks"] == data["queries"] == data["reservations"] == data["usage_bytes"] == 0
    admission = runtime.resource_snapshot()["task_admission"]
    assert admission["running_tasks"] == admission["ready_tasks"] == admission["queries"] == 0


@pytest.mark.parametrize("failure", ["schedule", "spawn", "input_allocate", "output_wrap", "grant_response"])
def test_failed_task_submission_returns_all_byte_reservations(strict_transport, monkeypatch, failure):
    from vane.execution import udf_subprocess as local

    def identity(table):
        return table

    def fail(*args, **kwargs):
        raise RuntimeError("planned scheduler failure")

    payload = dict(
        function_pickle=vane_pickle.dumps(identity),
        call_mode="map_batches",
        execution_backend="subprocess_task",
        udf_worker_slots=1,
        produce_ref_bundle_output=True,
        streaming_output_mode="local_shm_ref_bundle",
    )
    runtime = LocalModelRuntime(
        session_id="test",
        session_config={},
        data_limit=DataAdmissionLimits(100_000, 1024, 70_000),
        task_limit=TaskAdmissionLimits(1, 4),
    )
    plan = _Plan(payload)
    resources = runtime.prepare(plan, {})
    executor = build_executor(payload, plan.options)
    try:
        with monkeypatch.context() as patch:
            executor.request_task_admission(8)
            if failure == "schedule":
                patch.setattr(local._global_task_runtime().executor, "submit", fail)
                with pytest.raises(RuntimeError, match="planned scheduler failure"):
                    executor.submit(pa.table({"x": [1]}))
            else:
                if failure == "spawn":
                    patch.setattr(executor._task_pool, "_spawn_worker", fail)
                elif failure == "input_allocate":
                    patch.setattr(ref_bundle, "_create_shm", fail)
                elif failure == "output_wrap":
                    patch.setattr(local, "make_local_shm_ref_bundle_result_from_descriptor", fail)
                else:
                    send = local._send_message

                    def fail_output_grant(sock, msg_type, payload=b""):
                        if msg_type == local._MSG_OUTPUT_GRANT_GRANTED:
                            fail()
                        return send(sock, msg_type, payload)

                    patch.setattr(local, "_send_message", fail_output_grant)
                executor.submit(pa.table({"x": [1]}))
                assert isinstance(_wait_result(executor), BaseException)
    finally:
        executor.close(kill=True)
        for resource in resources:
            resource.shutdown(kill=True)
        runtime.close(timeout=5, kill=True)
    assert runtime.resource_snapshot()["data"]["usage_bytes"] == 0
