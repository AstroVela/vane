# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import ExitStack
from types import SimpleNamespace

import pytest
from local_runtime_helpers import local_runtime_diagnostics

import vane
from vane import pickle as vane_pickle
from vane.execution import ref_bundle, udf_subprocess
from vane.execution.request_admission import RequestAdmissionLimits
from vane.execution.udf_data_admission import DataAdmissionLimits, DataAdmissionWaitLimits
from vane.execution.udf_local_model import LocalModelRuntime
from vane.execution.udf_runtime_admission import TaskAdmissionLimits
from vane.execution.udf_subprocess import ensure_local_subprocess_actor_pools_for_plan


class _Identity:
    def __call__(self, table):
        return table


def _payload():
    return {
        "function_pickle": vane_pickle.dumps(_Identity),
        "execution_backend": "subprocess_actor",
        "call_mode": "map_batches",
        "actor_number": 1,
    }


@pytest.mark.parametrize("session", [None, "other"])
def test_direct_model_binding_requires_the_owning_session(monkeypatch, session):
    from vane.execution import udf_subprocess

    payload = _payload()
    constructed = []

    def unexpected_worker(*args, **kwargs):
        constructed.append(True)
        raise AssertionError("cross-session binding started a worker")

    monkeypatch.setattr(udf_subprocess, "LocalSubprocessActorPool", unexpected_worker)
    with LocalModelRuntime(session_id="owner", session_config={}) as runtime:
        model = runtime.register("model", version="v1", payload=payload)

        class Plan:
            def session_id(self):
                return session

            def collect_udf_nodes(self, conn=None):
                return [
                    {
                        "node_id": "1",
                        "payload": payload,
                        "executor_options": {"local_model_pool": model, "session_config": {}},
                    }
                ]

            def set_udf_actor_handles(self, options, conn=None):
                raise AssertionError("cross-session handles were published")

        with pytest.raises(ValueError, match="session"):
            ensure_local_subprocess_actor_pools_for_plan(Plan())
        assert not constructed
        assert runtime.resource_snapshot()["active_borrows"] == 0


@pytest.mark.parametrize("close", [False, True])
@pytest.mark.parametrize("request_limited", [False, True])
def test_registration_rechecks_drain_after_payload_serialization(monkeypatch, close, request_limited):
    from vane.execution import udf_local_model

    runtime = LocalModelRuntime(
        session_id="session",
        session_config={},
        request_limit=RequestAdmissionLimits(1, 1) if request_limited else None,
    )
    entered, resume = threading.Event(), threading.Event()
    serialize = udf_local_model._payload_bytes

    def pause(payload):
        entered.set()
        assert resume.wait(5)
        return serialize(payload)

    monkeypatch.setattr(udf_local_model, "_payload_bytes", pause)
    try:
        with ThreadPoolExecutor(max_workers=1) as threads:
            future = threads.submit(runtime.register, "late", version="v1", payload=_payload())
            try:
                assert entered.wait(5)
                if close:
                    runtime.close()
                else:
                    runtime.drain()
            finally:
                resume.set()
            with pytest.raises(RuntimeError, match="draining|closed"):
                future.result(timeout=5)
        assert not runtime._models
        assert runtime.resource_snapshot()["registered_models"] == 0
    finally:
        resume.set()
        runtime.close()


@pytest.mark.parametrize("request_limited", [False, True])
def test_close_waits_for_complete_model_publication(monkeypatch, request_limited):
    runtime = LocalModelRuntime(
        session_id="session",
        session_config={},
        request_limit=RequestAdmissionLimits(1, 1) if request_limited else None,
    )
    registered, resume, closing = threading.Event(), threading.Event(), threading.Event()
    register = runtime._registry.register

    def pause(*args, **kwargs):
        register(*args, **kwargs)
        registered.set()
        assert resume.wait(5)

    def close():
        closing.set()
        runtime.close()

    monkeypatch.setattr(runtime._registry, "register", pause)
    try:
        with ThreadPoolExecutor(max_workers=2) as threads:
            publishing = threads.submit(runtime.register, "model", version="v1", payload=_payload())
            try:
                assert registered.wait(5)
                shutdown = threads.submit(close)
                assert closing.wait(5)
                with pytest.raises(FutureTimeoutError):
                    shutdown.result(timeout=0.1)
            finally:
                resume.set()
            model = publishing.result(timeout=5)
            shutdown.result(timeout=5)
        assert runtime._models["model"] is model
        assert runtime.resource_snapshot()["closed"]
    finally:
        resume.set()
        runtime.close()


@pytest.mark.parametrize("tracked", [False, True])
def test_native_plan_can_be_prepared_for_sequential_executions(monkeypatch, tmp_path, tracked):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    marker = str(tmp_path / "initializations")

    class Model:
        def __init__(self):
            with open(marker, "a") as output:
                output.write("initialized\n")

        def __call__(self, table):
            return table

    with vane.connect() as connection:
        relation = connection.sql("SELECT 7::INTEGER AS x").map_batches(
            Model,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="subprocess_actor",
            actor_number=1,
        )
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(connection)
        node = plan.collect_udf_nodes(conn=connection)[0]
        with LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            track_data=tracked,
            task_limit=TaskAdmissionLimits(1, 2) if tracked else None,
        ) as runtime:
            runtime.register("model", version="v1", payload=node["payload"])
            for _ in range(2):
                resources = runtime.prepare(plan, {str(node["node_id"]): "model"}, conn=connection)
                assert runtime.resource_snapshot()["active_borrows"] == 1
                try:
                    result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(connection, plan)
                    assert [value for table in result.partition_payloads for value in table.column(0).to_pylist()] == [
                        7
                    ]
                finally:
                    for resource in resources:
                        resource.shutdown()
                assert runtime.resource_snapshot()["active_borrows"] == 0
            assert (tmp_path / "initializations").read_text().splitlines() == ["initialized"]


def test_preparation_validation_does_not_block_reentrant_drain(monkeypatch):
    from vane.execution import udf_local_model

    payload = _payload()
    with LocalModelRuntime(session_id="session", session_config={}) as runtime:
        runtime.register("model", version="v1", payload=payload)
        plan = SimpleNamespace(
            session_id=lambda: "session",
            session_config=lambda: {},
            collect_udf_nodes=lambda **kwargs: [{"node_id": "1", "payload": payload}],
        )
        fingerprint = udf_local_model._model_fingerprint
        with ThreadPoolExecutor(max_workers=1) as threads:

            def validate(payload):
                threads.submit(runtime.drain).result(timeout=3)
                return fingerprint(payload)

            monkeypatch.setattr(udf_local_model, "_model_fingerprint", validate)
            with pytest.raises(RuntimeError, match="draining"):
                runtime.prepare(plan, {"1": "model"})
        assert runtime.resource_snapshot()["active_borrows"] == 0


@pytest.mark.parametrize("task_limited", [False, True])
def test_concurrent_small_budget_requests_reuse_a_model_and_drain_native_views(monkeypatch, tmp_path, task_limited):
    import pyarrow as pa

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    transport = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 420_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", transport)
    with monkeypatch.context() as cpu:
        cpu.setattr(udf_subprocess.os, "cpu_count", lambda: 1)
        workers = udf_subprocess._GlobalSubprocessTaskRuntime()
    monkeypatch.setattr(udf_subprocess, "_GLOBAL_TASK_RUNTIME", workers)
    marker = str(tmp_path / "initializations")

    def produce(table):
        return pa.table({"blob": [b"x" * 65_536 for _ in range(len(table))]})

    class Model:
        def __init__(self):
            with open(marker, "a") as output:
                output.write("initialized\n")

        def __call__(self, table):
            return table

    try:
        with vane.connect() as connection, ExitStack() as stack:
            cursors = [connection.cursor(), connection.cursor()]
            for cursor in cursors:
                stack.callback(cursor.close)
                cursor.execute("SET threads=2")
            plans, nodes = [], []
            for cursor in cursors:
                relation = (
                    cursor.sql("SELECT i FROM range(3) t(i)")
                    .map_batches(
                        produce,
                        schema={"blob": vane.sqltypes.BLOB},
                        execution_backend="subprocess_task",
                        batch_size=1,
                        min_task_batch_size=1,
                        task_input_max_bytes=8,
                    )
                    .project("octet_length(blob)::BIGINT AS size")
                    .map_batches(
                        Model,
                        schema={"size": vane.sqltypes.BIGINT},
                        execution_backend="subprocess_actor",
                        actor_number=1,
                        batch_size=2,
                        min_task_batch_size=2,
                    )
                )
                plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(
                    cursor
                )
                plans.append(plan)
                nodes.append(
                    next(
                        node
                        for node in plan.collect_udf_nodes(conn=cursor)
                        if node["payload"]["execution_backend"] == "subprocess_actor"
                    )
                )
            with LocalModelRuntime(
                session_id=plans[0].session_id(),
                session_config=plans[0].session_config(),
                request_limit=RequestAdmissionLimits(2, 2),
                task_limit=TaskAdmissionLimits(1, 8) if task_limited else None,
                data_limit=DataAdmissionLimits(420_000, 140_000, 70_000, wait=DataAdmissionWaitLimits(8, 15)),
            ) as runtime:
                model = runtime.register("model", version="v1", payload=nodes[0]["payload"])

                def execute(request, index, barrier):
                    barrier.wait(timeout=5)
                    result = request.execute(plans[index], {str(nodes[index]["node_id"]): "model"}, conn=cursors[index])
                    assert [value for table in result.partition_payloads for value in table.column(0).to_pylist()] == [
                        65_536
                    ] * 3
                    assert result.task_stats["udf_completed_rows"] > 0
                    assert result.task_stats["udf_emitted_bytes"] > 0

                for _ in range(3):
                    requests = [runtime.request(), runtime.request()]
                    with ThreadPoolExecutor(max_workers=2) as threads:
                        barrier = threading.Barrier(2)
                        futures = [
                            threads.submit(execute, request, index, barrier) for index, request in enumerate(requests)
                        ]
                        try:
                            with local_runtime_diagnostics(runtime, tmp_path / "concurrent-small-budget"):
                                for future in futures:
                                    future.result(timeout=30)
                        finally:
                            for request in requests:
                                request.cancel()
                    gc.collect()
                    snapshot = runtime.resource_snapshot()
                    assert snapshot["active_borrows"] == snapshot["request_admission"]["active_requests"] == 0
                    assert snapshot["data"]["usage_bytes"] == snapshot["data"]["queued_byte_admissions"] == 0
                    assert transport.snapshot()["usage_bytes"] == workers.execution_capacity.reserved_slots == 0
                    if task_limited:
                        assert snapshot["task_admission"]["running_tasks"] == 0
                with model.acquire() as borrow:
                    assert len(borrow.pool.worker_pids()) == 1
                assert (tmp_path / "initializations").read_text().splitlines() == ["initialized"]
    finally:
        workers.close(kill=True)


@pytest.mark.parametrize("snapshot_fails", [False, True])
def test_failure_diagnostics_preserve_admission_and_the_original_error(monkeypatch, tmp_path, snapshot_fails):
    monkeypatch.delenv("VANE_TEST_DIAGNOSTICS_DIR", raising=False)
    with LocalModelRuntime(
        session_id="session", session_config={}, request_limit=RequestAdmissionLimits(1, 1)
    ) as runtime:
        ready, queued = runtime.request(), runtime.request()
        directory = tmp_path / "failure"
        original = AssertionError("original timeout")
        try:
            with monkeypatch.context() as fault:
                if snapshot_fails:

                    def fail():
                        raise RuntimeError("diagnostic failure")

                    fault.setattr(runtime, "resource_snapshot", fail)
                with pytest.raises(AssertionError) as error:
                    with local_runtime_diagnostics(runtime, directory):
                        raise original
            assert error.value is original
            assert ready.state == "ready" and queued.state == "queued"
            assert (directory / "threads.txt").stat().st_size > 0
            if not snapshot_fails:
                snapshot = json.loads((directory / "resources.json").read_text())
                assert snapshot["runtime"]["request_admission"]["active_requests"] == 1
                assert "transport" in snapshot
        finally:
            ready.cancel()
            queued.cancel()
