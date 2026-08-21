# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path

import pytest
from ray_test_profile import ray_test_object_store_options

import vane

ray = pytest.importorskip("ray")
Cluster = pytest.importorskip("ray.cluster_utils").Cluster
pytestmark = [
    pytest.mark.real_ray,
    pytest.mark.ray_cluster_owner,
    pytest.mark.ray_fault,
]

_FAULT_RAY_CLUSTER = None

import vane.runners.ray.worker_handle as worker_handle_mod
from vane.runners.ray import driver as ray_driver
from vane.runners.ray import worker as worker_mod
from vane.runners.ray.fte_fragment_scheduler import (
    _stop_fte_status_watchers,
    ensure_fte_fragment_progress_topology,
)
from vane.runners.ray.query_resource_graph import (
    QueryAllocation,
    QueryResourceGraph,
    ResourceUnitSpec,
    ResourceVector,
)
from vane.runners.ray.query_resource_graph_builder import native_fragment_unit_id_for_fragment
from vane.runners.ray.query_resource_runtime import (
    clear_query_resource_managers,
    register_query_resource_graph,
)
from vane.runners.ray.worker_handle import RayWorkerActorHandle as _ProductionRayWorkerActorHandle


class RayWorkerActorHandle(_ProductionRayWorkerActorHandle):
    def __init__(self, actor_handle, *, memory_capacity_bytes, worker_id, node_id=None):
        super().__init__(
            actor_handle,
            memory_capacity_bytes=memory_capacity_bytes,
            worker_id=worker_id,
            node_id=str(node_id or ray.get_runtime_context().get_node_id()),
        )


class _ControlFaultRayWorkerActorHandle(RayWorkerActorHandle):
    def _ensure_fragment_progress_topology(self, query_id, fragment_id, fragment_plan):
        topology = {
            "schema": "pipeline_topology",
            "pipelines": [
                {
                    "pipeline_id": 1,
                    "operators": ["TABLE_SCAN"],
                    "operator_details": [{}],
                }
            ],
        }
        return ensure_fte_fragment_progress_topology(
            query_id,
            fragment_id,
            lambda: topology,
        )


@ray.remote(max_concurrency=8)
class _FteControlFaultActor:
    def __init__(self, *, finish_attempts: bool = False) -> None:
        self.finish_attempts = finish_attempts
        self.requests = []
        self.statuses = {}
        self.wait_calls = 0

    def register_fragments(self, fragments):
        return {"registered": len(fragments), "existing": 0, "total": len(fragments)}

    def fte_create_task(self, request):
        self.requests.append(request)
        task_id = dict(request["task_id"])
        status = {
            "state": "FINISHED" if self.finish_attempts else "RUNNING",
            "task_id": task_id,
            "version": 1,
            "stats": [task_id["attempt_id"]],
        }
        self.statuses[self._key(task_id)] = status
        return self._control_status("fte_create_task", status)

    def fte_add_splits(
        self,
        task_id,
        _source_node_id,
        _splits,
        _fte_control_dependency=None,
    ):
        status = self.statuses[self._key(task_id)]
        status["version"] = int(status.get("version", 0)) + 1
        return self._control_status("fte_add_splits", status)

    def fte_no_more_splits(
        self,
        task_id,
        _source_node_id,
        _fte_control_dependency=None,
    ):
        status = self.statuses[self._key(task_id)]
        status["version"] = int(status.get("version", 0)) + 1
        return self._control_status("fte_no_more_splits", status)

    def fte_get_task_status(self, task_id):
        return dict(self.statuses[self._key(task_id)])

    async def fte_wait_task_status(self, task_id, _min_version=None, timeout_s=None):
        self.wait_calls += 1
        if self.finish_attempts:
            return dict(self.statuses[self._key(task_id)])
        await asyncio.sleep(float(timeout_s or 1.0))
        return dict(self.statuses[self._key(task_id)])

    def fte_get_task_info(self, task_id):
        return {"status": dict(self.statuses[self._key(task_id)]), "task_id": task_id}

    def fte_ack_task_result(self, task_id, _fte_control_dependency=None):
        return self._control_status(
            "fte_ack_task_result",
            self.statuses[self._key(task_id)],
        )

    def fte_release_task_result(self, task_id, _fte_control_dependency=None):
        return self._control_status(
            "fte_release_task_result",
            self.statuses[self._key(task_id)],
        )

    def fte_cancel_task(self, task_id, _fte_control_dependency=None):
        status = self.statuses.get(self._key(task_id), {"task_id": dict(task_id), "version": 0})
        status["state"] = "CANCELED"
        status["version"] = int(status.get("version", 0)) + 1
        self.statuses[self._key(task_id)] = status
        return self._control_status("fte_cancel_task", status)

    def fte_drop_query(self, _query_id):
        self.statuses.clear()
        return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

    def prepare_shutdown(self):
        """Acknowledge the production worker quiescence handshake."""

    def created_requests(self):
        return [
            {
                "task_id": dict(request["task_id"]),
                "initial_splits": request.get("initial_splits") or {},
                "no_more_splits": list(request.get("no_more_splits") or []),
            }
            for request in self.requests
        ]

    def wait_call_count(self):
        return self.wait_calls

    @staticmethod
    def _control_status(operation, status):
        result = dict(status)
        result["_fte_control_operation"] = operation
        result["_fte_control_applied"] = True
        return result

    @staticmethod
    def _key(task_id):
        return (
            f"{task_id['query_id']}.{task_id['fragment_execution_id']}."
            f"{task_id['partition_id']}.{task_id['attempt_id']}"
        )


class _RayFteTask:
    def __init__(self) -> None:
        self.plan_calls = 0

    def name(self):
        return "scan-task"

    def context(self):
        query_id = "query-real-kill"
        node_id = "7"
        fragment_id = f"{query_id}:node:{node_id}"
        return {
            "query_id": query_id,
            "node_id": node_id,
            "resource_query_id": query_id,
            "resource_unit_id": native_fragment_unit_id_for_fragment(query_id, fragment_id),
        }

    def task_context(self):
        return {"query_idx": 0, "last_node_id": 7, "task_id": 0, "node_ids": [7]}

    def Inputs(self):
        return {
            "3": {
                "kind": "scan_split_batch",
                "data": {
                    "splits": [
                        {
                            "split_id": "fault-before-kill",
                            "estimated_bytes": len(b"payload-before-kill"),
                            "data": b"payload-before-kill",
                        }
                    ]
                },
            }
        }

    def exchange_sink_instance(self):
        return None

    def plan(self):
        self.plan_calls += 1
        return {"plan": "unused-by-control-fault-actor"}


class _NativeDynamicScanWorkerTask:
    def __init__(
        self,
        *,
        query_id: str,
        node_id: str,
        split_batch: bytes,
        plan,
        fragment_node_id: str | None = None,
        name: str = "native-dynamic-scan-worker-task",
    ) -> None:
        self.query_id = query_id
        self.node_id = str(node_id)
        self.fragment_node_id = str(fragment_node_id if fragment_node_id is not None else node_id)
        self.split_batch = split_batch
        self._plan = plan
        self._name = str(name)

    def name(self):
        return self._name

    def context(self):
        fragment_id = f"{self.query_id}:node:{self.fragment_node_id}"
        return {
            "query_id": self.query_id,
            "node_id": self.fragment_node_id,
            "resource_query_id": self.query_id,
            "resource_unit_id": native_fragment_unit_id_for_fragment(self.query_id, fragment_id),
        }

    def task_context(self):
        try:
            last_node_id = int(self.node_id)
        except ValueError:
            last_node_id = 0
        return {"query_idx": 0, "last_node_id": last_node_id, "task_id": 0, "node_ids": [last_node_id]}

    def Inputs(self):
        return {self.node_id: {"kind": "scan_split_batch", "data": self.split_batch}}

    def exchange_sink_instance(self):
        return None

    def plan(self):
        return self._plan


def _clear_fte_state() -> None:
    _stop_fte_status_watchers()
    clear_query_resource_managers()
    worker_handle_mod._FTE_FRAGMENT_EXECUTION_IDS.clear()
    worker_handle_mod._FTE_QUERY_NEXT_FRAGMENT_EXECUTION_ID.clear()
    worker_handle_mod._FTE_STABLE_TASK_IDENTITY_KEYS_BY_RESOURCE_QUERY.clear()
    worker_handle_mod._FTE_FRAGMENT_EXECUTIONS.clear()
    worker_handle_mod._FTE_PARTITION_OWNERS.clear()
    worker_handle_mod._FTE_SEQUENCES.clear()
    worker_handle_mod._FTE_FRAGMENT_STATES.clear()
    worker_handle_mod._FTE_WORKER_HANDLES.clear()
    worker_handle_mod._FTE_RETRY_DELAYS.clear()
    worker_handle_mod._FTE_SCHEDULERS.clear()
    worker_handle_mod._FTE_CLOSING_QUERIES.clear()
    worker_handle_mod._FTE_ACTIVE_OPERATIONS_BY_QUERY.clear()
    worker_handle_mod._FTE_ACTIVE_TEARDOWN_OPERATIONS_BY_QUERY.clear()


def _register_fault_query(tasks) -> None:
    tasks = list(tasks)
    if not tasks:
        raise ValueError("fault query requires at least one task")
    query_ids = {str(task.context()["query_id"]) for task in tasks}
    if len(query_ids) != 1:
        raise ValueError("fault query tasks must share one query_id")
    query_id = query_ids.pop()
    fragment_ids = sorted({f"{query_id}:node:{task.context()['node_id']}" for task in tasks})
    target_output_block_bytes = 1024 * 1024
    units = tuple(
        ResourceUnitSpec(
            query_id=query_id,
            resource_unit_id=native_fragment_unit_id_for_fragment(query_id, fragment_id),
            physical_node_id=f"node:{fragment_id.rsplit(':node:', 1)[1]}:native-fragment",
            unit_kind="native_fragment",
            backend="ray_worker",
            input_unit_ids=(),
            per_task=ResourceVector(),
            target_output_block_bytes=target_output_block_bytes,
            generator_buffer_blocks=1,
            max_concurrency=max(1, len(tasks)),
        )
        for fragment_id in fragment_ids
    )
    allocation_resources = ResourceVector(
        cpu=max(1, len(tasks)),
        heap_bytes=max(1, len(tasks)) * 64 * 1024 * 1024,
        # These tests exercise FTE recovery rather than object-store
        # backpressure.  A streaming unit keeps 25% of the default allocation
        # protected for output handoff, so size the fixture such that the
        # remaining 75% admits every native generator window concurrently.
        object_store_bytes=(max(1, len(tasks)) * target_output_block_bytes * 4 + 2) // 3,
    )
    manager = register_query_resource_graph(
        QueryResourceGraph(
            query_id=query_id,
            plan_digest=f"sha256:fault:{query_id}",
            units=units,
            terminal_unit_ids=tuple(unit.resource_unit_id for unit in units),
        ),
        QueryAllocation(
            resources=allocation_resources,
            generation=1,
        ),
    )
    for unit in units:
        manager.update_unit_state(unit.resource_unit_id, runnable=True)


def _init_ray_for_fault_test(monkeypatch) -> None:
    global _FAULT_RAY_CLUSTER

    test_file = Path(__file__).resolve()
    pythonpath_entries = [str(test_file.parent), str(test_file.parents[1])]
    vane_package_parent = str(Path(vane.__file__).resolve().parent.parent)
    pythonpath_entries.append(vane_package_parent)
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    pythonpath = os.pathsep.join(dict.fromkeys(pythonpath_entries))
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    monkeypatch.setenv("VANE_FTE_RETRY_INITIAL_DELAY_S", "0")
    if _FAULT_RAY_CLUSTER is not None:
        if ray.is_initialized():
            return
        _shutdown_ray_for_fault_test()
    elif ray.is_initialized():
        _shutdown_ray_for_fault_test()

    runtime_env_vars = {
        "PYTHONPATH": pythonpath,
        "PYTHONWARNINGS": os.environ.get("PYTHONWARNINGS", ""),
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
        "VANE_FTE_RETRY_INITIAL_DELAY_S": os.environ["VANE_FTE_RETRY_INITIAL_DELAY_S"],
        "VANE_FTE_STATUS_WAIT_TIMEOUT_S": os.environ.get("VANE_FTE_STATUS_WAIT_TIMEOUT_S", "5"),
        "VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S": os.environ.get("VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S", "0"),
        "VANE_FTE_SPLIT_QUEUE_SPACE_WAIT_TIMEOUT_S": os.environ.get("VANE_FTE_SPLIT_QUEUE_SPACE_WAIT_TIMEOUT_S", "0.1"),
    }
    # The default Cluster lifecycle starts Ray's fate-sharing reaper and
    # registers a process-exit hook.  These tests own cleanup explicitly; a
    # reaper that outlives a faulted actor can otherwise terminate pytest
    # during session finalization, before its JUnit report is written.
    cluster = Cluster(shutdown_at_exit=False)
    try:
        from ray._private.resource_and_label_spec import ResourceAndLabelSpec

        resources = ResourceAndLabelSpec().resolve(is_head=True)
        object_store_options = ray_test_object_store_options()
        object_store_options.setdefault(
            "object_store_memory",
            resources.object_store_memory,
        )
        cluster.add_node(
            include_dashboard=False,
            num_cpus=int(os.environ.get("VANE_TEST_RAY_NUM_CPUS", "4")),
            num_gpus=0,
            **object_store_options,
        )
        ray.init(
            address=cluster.address,
            ignore_reinit_error=True,
            log_to_driver=True,
            runtime_env={"env_vars": runtime_env_vars},
        )
    except BaseException:
        cluster.shutdown()
        raise
    _FAULT_RAY_CLUSTER = cluster


def _shutdown_ray_for_fault_test() -> None:
    global _FAULT_RAY_CLUSTER

    cluster = _FAULT_RAY_CLUSTER
    try:
        try:
            # Result-handle watchers await Ray ObjectRefs on this loop. Stop
            # them before ray.shutdown() destroys the driver's core worker.
            ray_driver.shutdown_background_event_loop()
        finally:
            ray.shutdown()
    finally:
        _FAULT_RAY_CLUSTER = None
        if cluster is not None and os.environ.get("VANE_TEST_EXTERNAL_RAY_CLUSTER_CLEANUP") != "1":
            cluster.shutdown()


@pytest.fixture(scope="module", autouse=True)
def _fault_ray_runtime():
    yield
    _clear_fte_state()
    if ray.is_initialized() or _FAULT_RAY_CLUSTER is not None:
        _shutdown_ray_for_fault_test()


def _build_native_scan_worker_task(
    con,
    tmp_path,
    *,
    query_id: str,
    fragment_node_id: str,
    file_name: str,
    start: int,
    stop: int,
) -> tuple[_NativeDynamicScanWorkerTask, str]:
    src = tmp_path / file_name
    con.execute(
        f"""
        COPY (
            SELECT i::BIGINT AS i
            FROM range({int(start)}, {int(stop)}) tbl(i)
        ) TO '{src}' (FORMAT PARQUET)
        """
    )
    relation = con.sql(f"SELECT i FROM read_parquet('{src}')")
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)
    scan_split_batches = dict(plan.scan_split_batch_map())
    assert len(scan_split_batches) == 1
    source_node_id, split_batches = next(iter(scan_split_batches.items()))
    assert len(split_batches) == 1
    split_batch = bytes(split_batches[0])
    return (
        _NativeDynamicScanWorkerTask(
            query_id=query_id,
            node_id=str(source_node_id),
            fragment_node_id=fragment_node_id,
            split_batch=split_batch,
            plan=plan,
            name=f"native-dynamic-scan-{fragment_node_id}",
        ),
        str(source_node_id),
    )


def test_real_ray_actor_kill_replays_fte_task_on_replacement(monkeypatch):
    monkeypatch.setenv("VANE_FTE_STATUS_WAIT_TIMEOUT_S", "5")
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S", "0")
    _init_ray_for_fault_test(monkeypatch)
    _clear_fte_state()

    task = _RayFteTask()
    _register_fault_query([task])
    actor0 = _FteControlFaultActor.remote(finish_attempts=False)
    actor1 = _FteControlFaultActor.remote(finish_attempts=True)
    handle0 = _ControlFaultRayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-a")
    _ControlFaultRayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-b")

    try:
        task_handle = handle0.submit_tasks([task])[0]
        assert str(task_handle.task_id) == "query-real-kill.0.0.0"
        assert [str(handle.task_id) for handle in handle0.pop_fte_result_handles("query-real-kill")] == [
            "query-real-kill.0.0.0"
        ]
        assert ray.get(actor0.created_requests.remote())[0]["task_id"]["attempt_id"] == 0

        task_handle.done()
        deadline = time.monotonic() + 5.0
        while ray.get(actor0.wait_call_count.remote()) == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ray.get(actor0.wait_call_count.remote()) > 0

        ray.kill(actor0, no_restart=True)
        with pytest.raises(Exception):
            asyncio.run(asyncio.wait_for(task_handle.get_result(), timeout=10.0))
        handle0.wait_fte_worker_failure_reconciliation(timeout_s=10.0)
        retry_handles = handle0.pop_fte_result_handles("query-real-kill")
        assert len(retry_handles) == 1
        retry_handle = retry_handles[0]
        result = asyncio.run(asyncio.wait_for(retry_handle.get_result(), timeout=10.0))
        retry_requests = ray.get(actor1.created_requests.remote())

        assert result.ok
        assert str(task_handle.task_id) == "query-real-kill.0.0.0"
        assert str(retry_handle.task_id) == "query-real-kill.0.0.1"
        assert retry_requests[0]["task_id"]["attempt_id"] == 1
        retried_split = retry_requests[0]["initial_splits"]["3"][0]
        assert retried_split["split_id"] == "fault-before-kill"
        assert retried_split["data"] == task.Inputs()["3"]["data"]
        assert "worker-a" not in worker_handle_mod._FTE_WORKER_HANDLES
        assert (
            "query-real-kill",
            "query-real-kill:node:7",
            0,
        ) not in worker_handle_mod._FTE_PARTITION_OWNERS
    finally:
        try:
            ray.kill(actor1, no_restart=True)
        except Exception:
            pass
        _clear_fte_state()


def test_real_ray_actor_kill_without_replacement_fails_fte_fragment_execution(monkeypatch):
    monkeypatch.setenv("VANE_FTE_STATUS_WAIT_TIMEOUT_S", "5")
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S", "0")
    _init_ray_for_fault_test(monkeypatch)
    _clear_fte_state()

    task = _RayFteTask()
    _register_fault_query([task])
    actor0 = _FteControlFaultActor.remote(finish_attempts=False)
    handle0 = _ControlFaultRayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-solo")

    try:
        task_handle = handle0.submit_tasks([task])[0]
        assert str(task_handle.task_id) == "query-real-kill.0.0.0"
        task_handle.done()

        deadline = time.monotonic() + 5.0
        while ray.get(actor0.wait_call_count.remote()) == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ray.get(actor0.wait_call_count.remote()) > 0

        ray.kill(actor0, no_restart=True)
        with pytest.raises(Exception):
            asyncio.run(asyncio.wait_for(task_handle.get_result(), timeout=10.0))
        handle0.wait_fte_worker_failure_reconciliation(timeout_s=10.0)

        assert "worker-solo" not in worker_handle_mod._FTE_WORKER_HANDLES
        assert (
            "query-real-kill",
            "query-real-kill:node:7",
            0,
        ) not in worker_handle_mod._FTE_PARTITION_OWNERS
        stage = worker_handle_mod._FTE_FRAGMENT_EXECUTIONS[("query-real-kill", "query-real-kill:node:7")]
        partition = stage.partitions[0]
        assert stage.failed is True
        assert partition.failed is True
        assert partition.running_attempts == {}
        assert handle0.fte_pressure_stats()["running_attempt_count"] == 0
    finally:
        _clear_fte_state()


def test_real_ray_actor_kill_replays_native_dynamic_scan_on_replacement(monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("VANE_FTE_STATUS_WAIT_TIMEOUT_S", "5")
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S", "0")
    monkeypatch.setenv("VANE_FTE_SPLIT_QUEUE_SPACE_WAIT_TIMEOUT_S", "0.1")
    _init_ray_for_fault_test(monkeypatch)
    _clear_fte_state()

    con = vane.connect()
    src = tmp_path / "native_dynamic_scan_retry.parquet"
    con.execute(
        f"""
        COPY (
            SELECT i::BIGINT AS i
            FROM range(6) tbl(i)
        ) TO '{src}' (FORMAT PARQUET)
        """
    )
    relation = con.sql(f"SELECT sum(i) AS total FROM read_parquet('{src}')")
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)
    scan_split_batches = dict(plan.scan_split_batch_map())
    assert len(scan_split_batches) == 1
    node_id, split_batches = next(iter(scan_split_batches.items()))
    assert len(split_batches) == 1
    split_batch = bytes(split_batches[0])

    actor0 = worker_mod.RayWorkerActor.options(num_cpus=0).remote(1, 0, 1 << 30, 1 << 60)
    actor1 = worker_mod.RayWorkerActor.options(num_cpus=0).remote(1, 0, 1 << 30, 1 << 60)
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-native-a")
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-native-b")

    try:
        task = _NativeDynamicScanWorkerTask(
            query_id="query-native-kill",
            node_id=str(node_id),
            split_batch=split_batch,
            plan=plan,
        )
        _register_fault_query([task])
        task_handle = handle0.submit_tasks([task])[0]
        assert str(task_handle.task_id) == "query-native-kill.0.0.0"
        assert [str(handle.task_id) for handle in handle0.pop_fte_result_handles("query-native-kill")] == [
            "query-native-kill.0.0.0"
        ]
        task_handle.done()

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            info = ray.get(actor0.fte_get_task_info.remote(task_handle.task_id.to_dict()))
            status = info["status"]
            if status.get("state") == "RUNNING" and int(status.get("queued_split_count", 0)) == 0:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("native dynamic scan did not enter blocked RUNNING state")

        ray.kill(actor0, no_restart=True)

        with pytest.raises(Exception):
            asyncio.run(asyncio.wait_for(task_handle.get_result(), timeout=10.0))
        handle0.wait_fte_worker_failure_reconciliation(timeout_s=10.0)
        retry_handles = handle1.pop_fte_result_handles("query-native-kill")
        assert len(retry_handles) == 1
        retry_handle = retry_handles[0]

        retry_info = ray.get(actor1.fte_get_task_info.remote(retry_handle.task_id.to_dict()))
        if retry_info["status"].get("state") == "RUNNING":
            handle1.task_input_stream_exhausted([str(node_id)])
        result = asyncio.run(asyncio.wait_for(retry_handle.get_result(), timeout=20.0))

        assert result.ok
        assert result.has_output
        assert str(task_handle.task_id) == "query-native-kill.0.0.0"
        assert str(retry_handle.task_id) == "query-native-kill.0.0.1"
        final_info = ray.get(actor1.fte_get_task_info.remote(retry_handle.task_id.to_dict()))
        raw_result = final_info["result"]
        if isinstance(raw_result, dict):
            raw_result = raw_result["result"]
        output_refs, metadata, *_ = raw_result
        assert metadata[0][0] == 1
        output = ray.get(output_refs[0])
        assert output.column(0).to_pylist() == [15]
        assert "worker-native-a" not in worker_handle_mod._FTE_WORKER_HANDLES
        assert (
            "query-native-kill",
            f"query-native-kill:node:{node_id}",
            0,
        ) not in worker_handle_mod._FTE_PARTITION_OWNERS
    finally:
        for actor in (actor0, actor1):
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        con.close()
        _clear_fte_state()


def test_real_ray_full_query_worker_loss_uses_retry_output(monkeypatch, tmp_path):
    pa = pytest.importorskip("pyarrow")
    monkeypatch.setenv("VANE_FTE_STATUS_WAIT_TIMEOUT_S", "5")
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S", "0")
    monkeypatch.setenv("VANE_FTE_SPLIT_QUEUE_SPACE_WAIT_TIMEOUT_S", "0.1")
    _init_ray_for_fault_test(monkeypatch)
    _clear_fte_state()

    con = vane.connect()
    src = tmp_path / "full_query_retry_input.parquet"
    con.execute(
        f"""
        COPY (
            SELECT i::BIGINT AS i
            FROM range(6) tbl(i)
        ) TO '{src}' (FORMAT PARQUET)
        """
    )
    relation = con.sql(f"SELECT i FROM read_parquet('{src}')")
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)
    scan_split_batches = dict(plan.scan_split_batch_map())
    assert len(scan_split_batches) == 1
    node_id, split_batches = next(iter(scan_split_batches.items()))
    assert len(split_batches) == 1
    split_batch = bytes(split_batches[0])

    actor0 = worker_mod.RayWorkerActor.options(num_cpus=0).remote(1, 0, 1 << 30, 1 << 60)
    actor1 = worker_mod.RayWorkerActor.options(num_cpus=0).remote(1, 0, 1 << 30, 1 << 60)
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-full-a")
    handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-full-b")

    try:
        task = _NativeDynamicScanWorkerTask(
            query_id="query-full-kill",
            node_id=str(node_id),
            split_batch=split_batch,
            plan=plan,
        )
        _register_fault_query([task])
        task_handle = handle0.submit_tasks([task])[0]
        assert str(task_handle.task_id) == "query-full-kill.0.0.0"
        assert [str(handle.task_id) for handle in handle0.pop_fte_result_handles("query-full-kill")] == [
            "query-full-kill.0.0.0"
        ]
        task_handle.done()

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            info = ray.get(actor0.fte_get_task_info.remote(task_handle.task_id.to_dict()))
            status = info["status"]
            if status.get("state") == "RUNNING" and int(status.get("queued_split_count", 0)) == 0:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("native full-query scan did not enter blocked RUNNING state")

        ray.kill(actor0, no_restart=True)

        with pytest.raises(Exception):
            asyncio.run(asyncio.wait_for(task_handle.get_result(), timeout=10.0))
        handle0.wait_fte_worker_failure_reconciliation(timeout_s=10.0)
        retry_handles = handle1.pop_fte_result_handles("query-full-kill")
        assert len(retry_handles) == 1
        retry_handle = retry_handles[0]

        retry_info = ray.get(actor1.fte_get_task_info.remote(retry_handle.task_id.to_dict()))
        if retry_info["status"].get("state") == "RUNNING":
            handle1.task_input_stream_exhausted([str(node_id)])
        result = asyncio.run(asyncio.wait_for(retry_handle.get_result(), timeout=20.0))

        assert result.ok
        assert result.has_output
        assert str(task_handle.task_id) == "query-full-kill.0.0.0"
        assert str(retry_handle.task_id) == "query-full-kill.0.0.1"
        final_info = ray.get(actor1.fte_get_task_info.remote(retry_handle.task_id.to_dict()))
        raw_result = final_info["result"]
        if isinstance(raw_result, dict):
            raw_result = raw_result["result"]
        output_refs, metadata, *_ = raw_result
        assert sum(meta[0] for meta in metadata) == 6

        retry_tables = [ray.get(ref) for ref in output_refs]
        retry_table = pa.concat_tables(retry_tables)
        downstream = vane.connect()
        try:
            downstream.register("retry_output", retry_table)
            count, total = downstream.execute("SELECT count(*)::BIGINT, sum(c0)::BIGINT FROM retry_output").fetchone()
        finally:
            downstream.close()

        assert count == 6
        assert total == 15
        assert retry_table.column(0).to_pylist() == [0, 1, 2, 3, 4, 5]
        assert "worker-full-a" not in worker_handle_mod._FTE_WORKER_HANDLES
        assert (
            "query-full-kill",
            f"query-full-kill:node:{node_id}",
            0,
        ) not in worker_handle_mod._FTE_PARTITION_OWNERS
    finally:
        for actor in (actor0, actor1):
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        con.close()
        _clear_fte_state()


def test_real_ray_host_loss_replays_all_owned_full_query_outputs(monkeypatch, tmp_path):
    pa = pytest.importorskip("pyarrow")
    monkeypatch.setenv("VANE_FTE_STATUS_WAIT_TIMEOUT_S", "5")
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S", "0")
    monkeypatch.setenv("VANE_FTE_SPLIT_QUEUE_SPACE_WAIT_TIMEOUT_S", "0.1")
    _init_ray_for_fault_test(monkeypatch)
    _clear_fte_state()

    con = vane.connect()
    query_id = "query-host-full-kill"
    task_a, source_a = _build_native_scan_worker_task(
        con,
        tmp_path,
        query_id=query_id,
        fragment_node_id="scan-a",
        file_name="host_loss_input_a.parquet",
        start=0,
        stop=4,
    )
    task_b, source_b = _build_native_scan_worker_task(
        con,
        tmp_path,
        query_id=query_id,
        fragment_node_id="scan-b",
        file_name="host_loss_input_b.parquet",
        start=4,
        stop=8,
    )
    _register_fault_query([task_a, task_b])

    actor0 = worker_mod.RayWorkerActor.options(num_cpus=0).remote(2, 0, 1 << 30, 1 << 60)
    handle0 = RayWorkerActorHandle(actor0, memory_capacity_bytes=1 << 60, worker_id="worker-host-a")

    try:
        task_handles = handle0.submit_tasks([task_a, task_b])
        assert {str(handle.task_id) for handle in task_handles} == {
            f"{query_id}.0.0.0",
            f"{query_id}.1.0.0",
        }
        assert {str(handle.task_id) for handle in handle0.pop_fte_result_handles(query_id)} == {
            f"{query_id}.0.0.0",
            f"{query_id}.1.0.0",
        }
        for task_handle in task_handles:
            task_handle.done()

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            infos = [
                ray.get(actor0.fte_get_task_info.remote(task_handle.task_id.to_dict())) for task_handle in task_handles
            ]
            if all(
                info["status"].get("state") == "RUNNING" and int(info["status"].get("queued_split_count", 0)) == 0
                for info in infos
            ):
                break
            time.sleep(0.05)
        else:
            raise AssertionError(f"host-loss native scans did not enter blocked RUNNING state: {infos!r}")

        actor1 = worker_mod.RayWorkerActor.options(num_cpus=0).remote(2, 0, 1 << 30, 1 << 60)
        handle1 = RayWorkerActorHandle(actor1, memory_capacity_bytes=1 << 60, worker_id="worker-host-b")
        ray.kill(actor0, no_restart=True)

        for task_handle in task_handles:
            with pytest.raises(Exception):
                asyncio.run(asyncio.wait_for(task_handle.get_result(), timeout=10.0))
        handle0.wait_fte_worker_failure_reconciliation(timeout_s=10.0)
        retry_handles = handle1.pop_fte_result_handles(query_id)
        assert len(retry_handles) == 2

        handle1.task_input_stream_exhausted([source_a, source_b])
        results = [
            asyncio.run(asyncio.wait_for(retry_handle.get_result(), timeout=20.0)) for retry_handle in retry_handles
        ]
        assert all(result.ok and result.has_output for result in results)
        assert {str(handle.task_id) for handle in task_handles} == {
            f"{query_id}.0.0.0",
            f"{query_id}.1.0.0",
        }
        assert {str(handle.task_id) for handle in retry_handles} == {
            f"{query_id}.0.0.1",
            f"{query_id}.1.0.1",
        }

        retry_tables = []
        total_rows_from_metadata = 0
        for retry_handle in retry_handles:
            final_info = ray.get(actor1.fte_get_task_info.remote(retry_handle.task_id.to_dict()))
            raw_result = final_info["result"]
            if isinstance(raw_result, dict):
                raw_result = raw_result["result"]
            output_refs, metadata, *_ = raw_result
            total_rows_from_metadata += sum(meta[0] for meta in metadata)
            retry_tables.extend(ray.get(ref) for ref in output_refs)
        assert total_rows_from_metadata == 8

        retry_table = pa.concat_tables(retry_tables)
        downstream = vane.connect()
        try:
            downstream.register("retry_output", retry_table)
            count, total, min_value, max_value = downstream.execute(
                """
                SELECT
                    count(*)::BIGINT,
                    sum(c0)::BIGINT,
                    min(c0)::BIGINT,
                    max(c0)::BIGINT
                FROM retry_output
                """
            ).fetchone()
        finally:
            downstream.close()

        assert count == 8
        assert total == 28
        assert min_value == 0
        assert max_value == 7
        assert sorted(retry_table.column(0).to_pylist()) == list(range(8))
        assert "worker-host-a" not in worker_handle_mod._FTE_WORKER_HANDLES
        assert (query_id, f"{query_id}:node:scan-a", 0) not in worker_handle_mod._FTE_PARTITION_OWNERS
        assert (query_id, f"{query_id}:node:scan-b", 0) not in worker_handle_mod._FTE_PARTITION_OWNERS
        assert handle0.fte_pressure_stats()["running_attempt_count"] == 0
        assert handle1.fte_pressure_stats()["running_attempt_count"] == 0
    finally:
        for actor in (locals().get("actor0"), locals().get("actor1")):
            if actor is None:
                continue
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        con.close()
        _clear_fte_state()
