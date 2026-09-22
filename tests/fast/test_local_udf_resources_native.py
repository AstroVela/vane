# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import threading
import time
import uuid

import pyarrow as pa
import pytest

import vane
from vane import pickle as vane_pickle
from vane.execution import ref_bundle, udf_subprocess
from vane.execution.local_resource_graph import LocalResourceUnitContext
from vane.execution.udf import build_executor
from vane.execution.udf_data_admission import DataAdmissionLimits
from vane.execution.udf_data_lease import RuntimeDataLedger
from vane.execution.udf_executor_cleanup import QueryExecutorCleanup
from vane.execution.udf_local_model import LocalModelRuntime
from vane.execution.udf_resource_usage import UnitResourceActivity
from vane.execution.udf_runtime_admission import RuntimeTaskAdmission, TaskAdmissionLimits


def _wait(predicate):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise TimeoutError("UDF diagnostic condition did not become ready")


@pytest.fixture
def transport(monkeypatch):
    gc.collect()
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    yield manager
    gc.collect()
    assert manager.snapshot()["usage_bytes"] == 0


@pytest.mark.parametrize("tracking", ["graph_only", "data", "bytes"])
@pytest.mark.parametrize("limited_tasks", [False, True])
def test_native_mixed_plan_attributes_each_invocation_while_reusing_its_model(monkeypatch, tracking, limited_tasks):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    observed = []
    complete = udf_subprocess.UDFExecutor._complete_task_submit

    def observe(executor, *args, **kwargs):
        observed.append((executor.resource_identity(), runtime.resource_snapshot()["udf_units"]))
        return complete(executor, *args, **kwargs)

    monkeypatch.setattr(udf_subprocess.UDFExecutor, "_complete_task_submit", observe)

    class Identity:
        def __call__(self, table):
            return table

    with vane.connect() as conn:
        relation = (
            conn.sql("SELECT 7::INTEGER AS x")
            .map_batches(
                Identity, schema={"x": vane.sqltypes.INTEGER}, execution_backend="subprocess_actor", actor_number=1
            )
            .map_batches(lambda table: table, schema={"x": vane.sqltypes.INTEGER}, execution_backend="subprocess_task")
        )
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(conn)
        nodes = plan.collect_udf_nodes(conn=conn)
        model_node = next(node for node in nodes if node["payload"]["execution_backend"] == "subprocess_actor")
        with LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            track_graph=True,
            track_data=tracking == "data",
            data_limit=DataAdmissionLimits(16_384, 2048, 2048) if tracking == "bytes" else None,
            task_limit=TaskAdmissionLimits(2, 4) if limited_tasks else None,
        ) as runtime:
            model = runtime.register("identity", version="1", payload=model_node["payload"])
            model.prewarm()
            query_ids = set()
            worker_pids = []
            for _ in range(2):
                with model.acquire() as pool:
                    worker_pids.append(pool.pool.worker_pids())
                resources = runtime.prepare(plan, {str(model_node["node_id"]): "identity"}, conn=conn)
                try:
                    before = runtime.resource_snapshot()
                    assert len(before["udf_units"]) == 2
                    query_ids.add(before["udf_units"][0]["query_id"])
                    result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(conn, plan)
                    assert [row for table in result.partition_payloads for row in table.column(0).to_pylist()] == [7]
                    del result
                finally:
                    for resource in resources:
                        resource.shutdown()
                gc.collect()
                _wait(lambda: not runtime.resource_snapshot()["udf_units"])
            assert len(query_ids) == 2 and worker_pids[0] == worker_pids[1]
        assert {identity["query_id"] for identity, _ in observed} == query_ids
        assert {identity["backend"] for identity, _ in observed} == {"subprocess_task", "subprocess_actor"}
        for identity, units in observed:
            own = next(unit for unit in units if unit["resource_unit_id"] == identity["resource_unit_id"])
            assert own["completing_tasks"] >= 1
            if tracking == "graph_only":
                assert own["data"] is None
            else:
                assert own["data"]["output_bytes"] > 0


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("kind", ["input", "output"])
def test_real_transport_wait_is_attributed_and_cancellation_or_release_retires_it(
    transport, backend, limited, cancel, kind
):
    def expand(table):
        import pyarrow as pa

        return pa.table({"x": ["x" * (65_536 if kind == "output" else 1)]})

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
    identity = LocalResourceUnitContext("query", "unit", "node", backend)
    activity = UnitResourceActivity(identity.to_dict())
    ledger = RuntimeDataLedger()
    query = ledger.open_query()
    task_runtime = RuntimeTaskAdmission(TaskAdmissionLimits(1, 2)) if limited else None
    task_query = task_runtime.open_query() if task_runtime else None
    actor_pool = udf_subprocess.LocalSubprocessActorPool(payload, 1) if backend == "subprocess_actor" else None
    executor = build_executor(
        payload,
        {
            "local_resource_unit": identity,
            "local_resource_activity": activity,
            "local_data_scope": query,
            **({"local_task_admission": task_query} if task_query else {}),
            **({"local_actor_pool": actor_pool} if actor_pool else {}),
        },
    )
    occupied = transport.acquire_allocation(70_000)
    refs = []
    try:
        assert executor.request_task_admission(0)
        executor.submit_with_id(1, pa.table({"x": ["x" * (65_536 if kind == "input" else 1)]}))
        _wait(lambda: activity.snapshot()["waiting_by_reason"][f"shared_memory_{kind}"] == 1)
        assert activity.snapshot()["running_tasks"] == 0
        _wait(lambda: transport.snapshot()["waiting_output_grants"] == (1 if kind == "output" else 0))
        if task_runtime:
            _wait(lambda: task_runtime.snapshot()["running_tasks"] == 0)
        if cancel:
            executor.close(kill=True)
        else:
            transport.release_allocation(occupied)
            occupied = 0
            result = _wait(executor.take_ready_result)
            assert not isinstance(result[2], BaseException), result[2]
            refs.extend(result[2][1])
            query.shutdown()
            ledger.close()
            usage = ledger.unit_snapshots()["unit"]["usage"]
            assert usage["output_bytes"] == usage["retained_bytes"] > (65_536 if kind == "output" else 0)
        _wait(lambda: not any(value for key, value in activity.snapshot().items() if key.endswith("_tasks")))
    finally:
        transport.release_allocation(occupied)
        executor.close(kill=True)
        if actor_pool:
            actor_pool.shutdown(kill=True)
        query.shutdown()
        if task_query:
            task_query.shutdown()
            task_runtime.close()
        for ref in refs:
            ref.release()
        ledger.close()
    assert not ledger.unit_snapshots()


def test_completion_callback_remains_visible_after_executor_shutdown_timeout(monkeypatch):
    identity = LocalResourceUnitContext("query", "unit", "node", "subprocess_task")
    activity = UnitResourceActivity(identity.to_dict())
    cleanup = QueryExecutorCleanup()
    executor = build_executor(
        {
            "function_pickle": vane_pickle.dumps(lambda table: table),
            "call_mode": "map_batches",
            "execution_backend": "subprocess_task",
            "udf_worker_slots": 1,
        },
        {"local_resource_unit": identity, "local_resource_activity": activity, "local_executor_cleanup": cleanup},
    )
    entered, proceed = threading.Event(), threading.Event()
    complete = executor._complete_task_submit

    def blocked(*args, **kwargs):
        entered.set()
        assert proceed.wait(timeout=10)
        return complete(*args, **kwargs)

    monkeypatch.setattr(executor, "_complete_task_submit", blocked)
    monkeypatch.setattr(udf_subprocess, "_subprocess_shutdown_grace_s", lambda: 0.05)
    try:
        executor.request_task_admission(0)
        executor.submit_with_id(1, pa.table({"x": [1]}))
        assert entered.wait(timeout=10)
        with pytest.raises(RuntimeError, match="pending"):
            executor.close(kill=True)
        assert activity.snapshot()["completing_tasks"] == 1
    finally:
        proceed.set()
        _wait(lambda: activity.snapshot()["completing_tasks"] == 0)
        executor.close(kill=True)
        cleanup.shutdown(kill=True)


@pytest.mark.parametrize("backend", ["ray_task", "ray_actor"])
def test_local_activity_cannot_be_routed_to_ray(backend):
    with pytest.raises(ValueError, match="local subprocess"):
        build_executor(
            {"execution_backend": backend, "call_mode": "map_batches"},
            {"local_resource_activity": UnitResourceActivity({})},
        )


@pytest.mark.parametrize("context", [None, LocalResourceUnitContext("q", "different", "n", "subprocess_task")])
def test_activity_requires_matching_invocation_identity(context):
    with pytest.raises(ValueError, match="matching resource unit"):
        build_executor(
            {"execution_backend": "subprocess_task", "call_mode": "map_batches"},
            {
                "local_resource_unit": context,
                "local_resource_activity": UnitResourceActivity({"query_id": "q", "resource_unit_id": "unit"}),
            },
        )
