# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

import duckdb
from duckdb._ray_cxx import validate_plan_serialization_for_submission
from duckdb._ray_errors import RemoteRayException
from duckdb.runners.ray.cluster_resource_coordinator import NodeCapacity
from duckdb.runners.ray.query_execution_graph import (
    ActorPlacement,
    NodeResourceAllocation,
    QueryAllocation,
    QueryExecutionGraph,
    ResourceVector,
    StageResourceSpec,
)
from duckdb.runners.ray.query_resource_runtime import (
    clear_query_resource_managers,
    get_query_resource_manager,
)

_GIB = 1024**3


class _FakeLogicalPlan:
    def __init__(self, physical_plan, events):
        self._physical_plan = physical_plan
        self._events = events

    def to_physical_plan(self, conn):
        assert conn is not None
        self._events.append("physical_plan")
        return self._physical_plan


class _ValidatingLogicalPlan(_FakeLogicalPlan):
    def to_physical_plan(self, conn):
        physical_plan = super().to_physical_plan(conn)
        validate_plan_serialization_for_submission(physical_plan)
        return physical_plan


class _FakePhysicalPlan:
    def __init__(self, query_id, metadata, events):
        self._query_id = query_id
        self._metadata = metadata
        self._events = events

    def idx(self):
        return self._query_id

    def collect_execution_stages(self, conn=None):
        assert conn is not None
        self._events.append("collect_graph")
        return self._metadata


class _RegistrationOnlyIdxPhysicalPlan(_FakePhysicalPlan):
    def __init__(self, query_id, metadata, events):
        super().__init__(query_id, metadata, events)
        self.idx_calls = 0

    def idx(self):
        self.idx_calls += 1
        if self.idx_calls > 2:
            raise RuntimeError("physical plan idx was re-entered after registration")
        return super().idx()


class _FakeCoordinator:
    def __init__(self, events):
        self._events = events
        self.released = []
        self.allocations = {}
        self.capacity_updates = []

    def register_query(self, demand):
        self._events.append("coordinator_register")
        resources = ResourceVector(
            cpu=8,
            gpu=1,
            heap_bytes=16 * _GIB,
            object_store_bytes=4 * _GIB,
        )
        allocation = QueryAllocation(
            resources=resources,
            node_allocations=(NodeResourceAllocation(node_id="node-a", resources=resources),),
            actor_placements=tuple(
                ActorPlacement(
                    stage_id=bundle.stage_id,
                    actor_index=bundle.actor_index,
                    node_id="node-a",
                )
                for bundle in demand.actor_bundles
            ),
            generation=7,
        )
        self.allocations[demand.query_id] = allocation
        return allocation

    def update_node_capacities(self, capacities):
        self.capacity_updates.append(tuple(capacities))
        return None

    def release_query(self, query_id, generation):
        self.released.append((query_id, generation))
        self._events.append("coordinator_release")
        self.allocations.pop(query_id, None)
        return True

    def snapshot(self):
        return {
            "queries": {
                query_id: {"allocation": allocation.to_dict()} for query_id, allocation in self.allocations.items()
            }
        }


def _metadata(query_id: str) -> dict:
    return {
        "query_id": query_id,
        "nodes": [
            {
                "node_id": "0",
                "node_name": "ScanSource",
                "input_node_ids": [],
                "is_sink": False,
                "num_partitions": 4,
                "udf_payload": None,
            },
            {
                "node_id": "1",
                "node_name": "StreamingUDF",
                "input_node_ids": ["0"],
                "is_sink": False,
                "num_partitions": 4,
                "udf_payload": {
                    "query_id": query_id,
                    "stage_id": f"stage:{query_id}:node:1:udf",
                    "execution_backend": "ray_actor",
                    "actor_pool_size": 1,
                    "cpus": 1.0,
                    "gpus": 1.0,
                    "memory_bytes": 4 * _GIB,
                    "udf_output_target_max_bytes": 128 * 1024**2,
                    "udf_task_input_max_bytes": 128 * 1024**2,
                },
            },
        ],
        "terminal_node_ids": ["1"],
    }


@pytest.fixture(autouse=True)
def _clean_query_runtime():
    clear_query_resource_managers()
    yield
    clear_query_resource_managers()


def _runner(events, coordinator):
    from duckdb.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    runner._duckdb_conn = object()
    runner._env_overrides = {}
    runner._query_resource_coordinator = coordinator
    runner._query_resource_lock = threading.RLock()
    runner._query_allocations = {}
    runner._query_graphs = {}
    runner._active_udf_actors = []
    runner._active_udf_actors_by_plan = {}
    runner._active_vllm_actors = []
    runner.curr_plans = {}
    runner.curr_streams = {}
    runner._plan_query_ids = {}
    runner._leased_result_partition_refs = {}
    runner._result_partition_ref_counters = {}
    node_resources = ResourceVector(
        cpu=8,
        gpu=1,
        heap_bytes=16 * _GIB,
        object_store_bytes=4 * _GIB,
    )

    def _read_node_capacities():
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        events.append("capacity")
        return (NodeCapacity("node-a", node_resources),)

    runner._read_query_node_capacities = _read_node_capacities
    runner._ensure_duckdb_conn = lambda: runner._duckdb_conn
    runner._precreate_vllm_actors = lambda plan: events.append("vllm_ready") or []
    runner._get_plan_runner = lambda: SimpleNamespace(run_plan=lambda plan, conn: "stream")
    return runner_cls, runner


def test_driver_starts_plan_runner_before_opening_actor_readiness_gate():
    events = []
    coordinator = _FakeCoordinator(events)
    runner_cls, runner = _runner(events, coordinator)
    query_id = "query-driver-order"
    physical_plan = _FakePhysicalPlan(query_id, _metadata(query_id), events)

    def _precreate(plan, graph, allocation):
        assert graph.query_id == query_id
        assert allocation.actor_node_ids_for_stage(f"stage:{query_id}:node:1:udf") == ("node-a",)
        manager = get_query_resource_manager(query_id)
        actor_stage = manager.snapshot()["stages"][f"stage:{query_id}:node:1:udf"]
        assert actor_stage["actor_ready"] is False
        events.append("actors_created")
        return [SimpleNamespace(shutdown=lambda: None)]

    runner._precreate_udf_actors = _precreate

    def _wait_for_ready(_actor_pools):
        manager = get_query_resource_manager(query_id)
        actor_stage = manager.snapshot()["stages"][f"stage:{query_id}:node:1:udf"]
        assert actor_stage["actor_ready"] is False
        events.append("actors_ready")

    runner._wait_for_udf_actors_ready = _wait_for_ready

    def _run_plan(plan, conn):
        manager = get_query_resource_manager(query_id)
        actor_stage = manager.snapshot()["stages"][f"stage:{query_id}:node:1:udf"]
        assert actor_stage["actor_ready"] is False
        events.append("plan_runner")
        return "stream"

    runner._get_plan_runner = lambda: SimpleNamespace(run_plan=_run_plan)

    asyncio.run(runner_cls.run_plan(runner, _FakeLogicalPlan(physical_plan, events)))

    assert events == [
        "physical_plan",
        "collect_graph",
        "capacity",
        "coordinator_register",
        "actors_created",
        "vllm_ready",
        "plan_runner",
        "actors_ready",
    ]
    manager = get_query_resource_manager(query_id)
    assert manager.snapshot()["stages"][f"stage:{query_id}:node:1:udf"]["actor_ready"] is True
    assert runner.curr_streams[query_id] == "stream"


def test_run_plan_does_not_read_physical_plan_id_after_registration():
    events: list[str] = []
    coordinator = _FakeCoordinator(events)
    runner_cls, runner = _runner(events, coordinator)
    runner._precreate_udf_actors = lambda *_args: []
    runner._mark_query_actor_stages_ready = lambda _graph: None
    query_id = "query-single-use-plan-id"
    physical_plan = _RegistrationOnlyIdxPhysicalPlan(
        query_id,
        _metadata(query_id),
        events,
    )

    asyncio.run(
        runner_cls.run_plan(
            runner,
            _FakeLogicalPlan(physical_plan, events),
        )
    )

    assert physical_plan.idx_calls == 2
    assert runner._plan_query_ids[query_id] == query_id


def test_driver_rolls_back_graph_and_cluster_allocation_when_actor_initialization_fails():
    events = []
    coordinator = _FakeCoordinator(events)
    runner_cls, runner = _runner(events, coordinator)
    query_id = "query-driver-rollback"
    physical_plan = _FakePhysicalPlan(query_id, _metadata(query_id), events)

    def _fail_precreate(plan, graph, allocation):
        events.append("actors_initializing")
        raise RuntimeError("model initialization failed")

    runner._precreate_udf_actors = _fail_precreate

    with pytest.raises(RuntimeError, match="model initialization failed"):
        asyncio.run(runner_cls.run_plan(runner, _FakeLogicalPlan(physical_plan, events)))

    with pytest.raises(KeyError, match="query graph is not registered"):
        get_query_resource_manager(query_id)
    assert coordinator.released == [(query_id, 7)]
    assert query_id not in runner.curr_plans
    assert "plan_runner" not in events


def test_copy_registration_keeps_streaming_udf_admission_bounded_when_ray_nodes_is_delayed(
    monkeypatch,
):
    from duckdb.runners.ray import driver as driver_module
    from duckdb.runners.ray.query_resource_runtime import register_query_graph

    events: list[str] = []
    coordinator = _FakeCoordinator(events)
    runner_cls, runner = _runner(events, coordinator)
    runner._query_resource_lock = threading.Lock()
    runner._read_query_node_capacities = lambda: runner_cls._read_query_node_capacities()
    runner._precreate_udf_actors = lambda *_args: []
    runner._precreate_vllm_actors = lambda *_args: []
    runner._get_plan_runner = lambda: SimpleNamespace(
        run_copy_plan=lambda _plan, _conn: {"rows_copied": 1},
    )
    runner._mark_query_actor_stages_ready = lambda _graph: None
    runner._build_local_progress_snapshot = lambda query_id, _started_at: {
        "query_id": query_id,
        "state": "FINISHED",
    }
    runner._teardown_plan_resources = lambda *_args, **_kwargs: None
    runner._open_query_resource_admission = lambda _query_id: None

    streaming_query_id = "query-streaming-admission"
    streaming_stage = StageResourceSpec(
        query_id=streaming_query_id,
        stage_id=f"stage:{streaming_query_id}:udf",
        physical_node_id="node:streaming:udf",
        stage_kind="udf",
        backend="ray_task",
        input_stage_ids=(),
        per_task=ResourceVector(cpu=1, heap_bytes=128),
        target_output_block_bytes=64,
        generator_buffer_blocks=1,
        max_concurrency=None,
    )
    streaming_graph = QueryExecutionGraph(
        query_id=streaming_query_id,
        plan_digest="sha256:streaming-admission",
        stages=(streaming_stage,),
        terminal_stage_ids=(streaming_stage.stage_id,),
    )
    streaming_resources = ResourceVector(
        cpu=2,
        heap_bytes=4096,
        object_store_bytes=4096,
    )
    streaming_allocation = QueryAllocation(
        resources=streaming_resources,
        node_allocations=(
            NodeResourceAllocation(
                node_id="node-a",
                resources=streaming_resources,
            ),
        ),
        actor_placements=(),
        generation=1,
    )
    streaming_manager = register_query_graph(
        streaming_graph,
        streaming_allocation,
    )
    streaming_manager.update_stage_state(
        streaming_stage.stage_id,
        runnable=True,
        actor_ready=True,
    )
    coordinator.allocations[streaming_query_id] = streaming_allocation
    runner._query_graphs[streaming_query_id] = streaming_graph
    runner._query_allocations[streaming_query_id] = streaming_allocation

    nodes_started = threading.Event()
    nodes_release = threading.Event()

    def _delayed_nodes():
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        assert runner._query_resource_lock.acquire(blocking=False)
        runner._query_resource_lock.release()
        nodes_started.set()
        assert nodes_release.wait(timeout=2.0)
        return [
            {
                "Alive": True,
                "NodeID": "node-a",
                "Resources": {
                    "CPU": 32.0,
                    "GPU": 4.0,
                    "memory": 64 * _GIB,
                    "object_store_memory": 32 * _GIB,
                },
            }
        ]

    monkeypatch.setattr(driver_module.ray, "nodes", _delayed_nodes)
    copy_query_id = "query-copy-delayed-capacity"
    copy_plan = _FakePhysicalPlan(
        copy_query_id,
        _metadata(copy_query_id),
        events,
    )
    lease_request = {
        "request_id": "request:streaming-during-copy",
        "query_id": streaming_query_id,
        "stage_id": streaming_stage.stage_id,
        "task_id": "task:streaming-during-copy",
        "attempt_id": "attempt:streaming-during-copy",
        "node_id": None,
        "retained_input_bytes": 0,
        "resources": {
            "cpu": 1.0,
            "gpu": 0.0,
            "heap_bytes": 128,
            "object_store_bytes": 0,
        },
    }

    async def _run_concurrently():
        copy_task = asyncio.create_task(
            runner_cls.run_copy_plan(
                runner,
                _FakeLogicalPlan(copy_plan, events),
            )
        )
        try:
            assert await asyncio.to_thread(nodes_started.wait, 1.0)
            started_at = time.monotonic()
            lease = await asyncio.wait_for(
                runner_cls.acquire_query_task_lease(runner, lease_request),
                timeout=0.25,
            )
            assert time.monotonic() - started_at < 0.25
            assert lease["granted"] is True
            assert copy_task.done() is False
            released = await asyncio.wait_for(
                runner_cls.release_query_task_lease(
                    runner,
                    lease_request["request_id"],
                    lease["lease"]["lease_id"],
                    lease["lease"]["attempt_id"],
                ),
                timeout=0.25,
            )
            assert released == {"released": True}
            nodes_release.set()
            return await asyncio.wait_for(copy_task, timeout=1.0)
        finally:
            nodes_release.set()

    outcome = asyncio.run(_run_concurrently())

    assert outcome.result == {"rows_copied": 1}
    assert outcome.final_progress_snapshot == {
        "query_id": copy_query_id,
        "state": "FINISHED",
    }


def test_registration_does_not_overwrite_a_newer_capacity_snapshot():
    events: list[str] = []
    coordinator = _FakeCoordinator(events)
    runner_cls, runner = _runner(events, coordinator)
    older = NodeCapacity(
        "node-old",
        ResourceVector(
            cpu=2,
            heap_bytes=2 * _GIB,
            object_store_bytes=1 * _GIB,
        ),
    )
    newer = NodeCapacity(
        "node-new",
        ResourceVector(
            cpu=8,
            heap_bytes=8 * _GIB,
            object_store_bytes=4 * _GIB,
        ),
    )
    runner._query_node_capacities = (newer,)
    runner._query_resource_last_capacity_refresh_at = 20.0

    capacity = runner_cls._apply_query_capacity_snapshot(
        runner,
        (older,),
        snapshot_started_at=10.0,
    )

    assert capacity == newer.resources
    assert runner._query_node_capacities == (newer,)
    assert runner._query_resource_last_capacity_refresh_at == 20.0
    assert coordinator.capacity_updates == []


def test_pending_allocation_teardown_fences_query_id_reuse_until_remote_drop_finishes():
    from duckdb.runners.ray.driver import (
        _DeferredQueryAllocationTeardown,
        _PreparedQueryResourceRegistration,
    )
    from duckdb.runners.ray.query_graph_builder import build_query_execution_graph

    events: list[str] = []
    coordinator = _FakeCoordinator(events)
    runner_cls, runner = _runner(events, coordinator)
    runner._query_resource_lock = threading.Lock()
    runner._open_query_resource_admission = lambda _query_id: None
    query_id = "query-generation-fence"
    graph = build_query_execution_graph(_metadata(query_id))
    node = NodeCapacity(
        "node-a",
        ResourceVector(
            cpu=8,
            gpu=1,
            heap_bytes=16 * _GIB,
            object_store_bytes=4 * _GIB,
        ),
    )
    teardown = _DeferredQueryAllocationTeardown(
        query_id=query_id,
        generation=3,
        reason="old generation lost actor placement",
    )
    runner._query_allocation_teardowns_pending = {query_id: teardown}
    runner._query_allocation_teardowns_claimed = set()
    runner._query_allocation_teardown_futures = set()
    runner._query_terminal_errors = {}
    drop_started = threading.Event()
    allow_drop = threading.Event()

    def _drop_query_fragments(actual_query_id, *, release_resources):
        assert actual_query_id == query_id
        assert release_resources is False
        drop_started.set()
        assert allow_drop.wait(timeout=2.0)

    runner._drop_query_fragments_after_admission_fence_sync = _drop_query_fragments
    worker = threading.Thread(
        target=runner_cls._run_query_allocation_teardowns,
        args=(runner, (teardown,)),
    )
    worker.start()
    assert drop_started.wait(timeout=1.0)

    prepared = _PreparedQueryResourceRegistration(
        graph=graph,
        node_capacities=(node,),
        capacity_snapshot_started_at=time.monotonic(),
    )
    try:
        with pytest.raises(RuntimeError, match="pending allocation-loss teardown"):
            runner_cls._commit_query_resource_registration(
                runner,
                prepared,
                deferred_teardowns=[],
            )
    finally:
        allow_drop.set()
        worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert query_id not in runner._query_allocation_teardowns_pending

    registered_graph, allocation = runner_cls._commit_query_resource_registration(
        runner,
        prepared,
        deferred_teardowns=[],
    )
    assert registered_graph is graph
    assert allocation.generation == 7


def test_failed_allocation_teardown_remains_retryable():
    from duckdb.runners.ray.driver import _DeferredQueryAllocationTeardown

    events: list[str] = []
    runner_cls, runner = _runner(events, _FakeCoordinator(events))
    query_id = "query-teardown-retry"
    teardown = _DeferredQueryAllocationTeardown(
        query_id=query_id,
        generation=4,
        reason="old generation lost actor placement",
    )
    runner._query_allocation_teardowns_pending = {query_id: teardown}
    runner._query_allocation_teardowns_claimed = set()
    runner._query_allocation_teardown_futures = set()
    runner._query_graphs = {query_id: object()}
    runner._query_allocations = {
        query_id: SimpleNamespace(generation=teardown.generation + 1),
    }
    runner._query_terminal_errors = {query_id: teardown.reason}
    attempts = 0

    def _drop_query_fragments(_query_id, *, release_resources):
        nonlocal attempts
        assert release_resources is False
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient remote teardown failure")

    runner._drop_query_fragments_after_admission_fence_sync = _drop_query_fragments

    runner_cls._run_query_allocation_teardowns(runner, (teardown,))
    assert runner._query_allocation_teardowns_pending == {query_id: teardown}
    assert "transient remote teardown failure" in runner._query_terminal_errors[query_id]

    runner._query_graphs = {}
    runner._query_allocations = {}
    retry = runner_cls._synchronize_query_allocations(runner)
    assert retry == (teardown,)
    runner_cls._run_query_allocation_teardowns(runner, retry)

    assert attempts == 2
    assert runner._query_allocation_teardowns_pending == {}


@pytest.mark.parametrize("entrypoint", ["run_plan", "run_copy_plan"])
def test_driver_rejects_non_serializable_plan_before_query_registration(entrypoint):
    events = []
    coordinator = _FakeCoordinator(events)
    runner_cls, runner = _runner(events, coordinator)
    query_id = f"query-plan-serialization-failure-{entrypoint}"
    physical_plan = duckdb.ray_cxx._make_non_serializable_physical_plan_for_test(query_id)

    with pytest.raises(
        RuntimeError,
        match=f"distributed physical plan serialization preflight failed for query_id={query_id}",
    ) as exc_info:
        coroutine = getattr(runner_cls, entrypoint)(runner, _ValidatingLogicalPlan(physical_plan, events))
        asyncio.run(coroutine)

    assert isinstance(exc_info.value, RemoteRayException)
    assert isinstance(exc_info.value.__cause__, duckdb.NotImplementedException)
    assert "INTENTIONALLY_NON_SERIALIZABLE operator cannot be serialized" in str(exc_info.value.__cause__)
    with pytest.raises(KeyError, match="query graph is not registered"):
        get_query_resource_manager(query_id)
    assert coordinator.released == []
    assert coordinator.allocations == {}
    assert runner._query_graphs == {}
    assert runner._query_allocations == {}
    assert query_id not in runner.curr_plans
    assert query_id not in runner.curr_streams
    assert query_id not in runner._plan_query_ids
    assert events == ["physical_plan"]


def test_driver_exposes_query_task_and_output_lease_api():
    from duckdb.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    required = {
        "acquire_query_task_lease",
        "mark_query_task_lease_submitted",
        "release_query_task_lease",
        "acquire_query_output_block_lease",
        "handoff_query_output_block_lease",
        "release_query_output_block_lease",
    }
    assert required.issubset(dir(runner_cls))


def test_driver_maintenance_refreshes_ray_capacity_usage_and_heartbeat_atomically():
    from duckdb.runners.ray.cluster_resource_coordinator import (
        ClusterQueryResourceCoordinator,
        NodeCapacity,
    )
    from duckdb.runners.ray.driver import RayQueryDriverActor
    from duckdb.runners.ray.query_graph_builder import (
        build_query_demand,
        build_query_execution_graph,
    )
    from duckdb.runners.ray.query_resource_manager import TaskRequest
    from duckdb.runners.ray.query_resource_runtime import register_query_graph

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    query_id = "query-driver-maintenance"
    graph = build_query_execution_graph(_metadata(query_id))
    initial_node = NodeCapacity(
        "node-a",
        ResourceVector(
            cpu=8,
            gpu=1,
            heap_bytes=16 * _GIB,
            object_store_bytes=4 * _GIB,
        ),
    )
    coordinator = ClusterQueryResourceCoordinator((initial_node,), heartbeat_timeout_s=30)
    demand = build_query_demand(
        graph,
        initial_node.resources,
    )
    allocation = coordinator.register_query(demand, now=0)
    manager = register_query_graph(graph, allocation)
    for stage in graph.stages:
        manager.update_stage_state(
            stage.stage_id,
            runnable=True,
            actor_ready=stage.backend != "ray_actor",
        )
    fte_stage = next(stage for stage in graph.stages if stage.backend == "ray_worker")
    task_grant = manager.try_acquire_task(
        TaskRequest(
            query_id=query_id,
            stage_id=fte_stage.stage_id,
            task_id="fte-task-1",
            attempt_id="fte-attempt-1",
            node_id="node-a",
        )
    )
    assert task_grant.granted

    runner._query_resource_lock = threading.RLock()
    runner._query_resource_coordinator = coordinator
    runner._query_graphs = {query_id: graph}
    runner._query_allocations = {query_id: allocation}
    runner._query_node_capacities = (initial_node,)
    runner._query_resource_admission_loop = None
    shrunk_node = NodeCapacity(
        "node-a",
        ResourceVector(
            cpu=3,
            gpu=1,
            heap_bytes=8 * _GIB,
            object_store_bytes=768 * 1024**2,
        ),
    )

    runner_cls._maintain_query_resources_once(
        runner,
        capacities=(shrunk_node,),
        now=5,
    )

    query_snapshot = coordinator.snapshot()["queries"][query_id]
    manager_snapshot = manager.snapshot()
    assert query_snapshot["observed_usage"] == manager_snapshot["usage"]
    assert query_snapshot["expires_at"] == 35
    assert manager_snapshot["allocation"] == query_snapshot["allocation"]
    assert runner._query_node_capacities == (shrunk_node,)

    def _capacity_unavailable():
        raise RuntimeError("GCS temporarily unavailable")

    runner._read_query_node_capacities = _capacity_unavailable
    cached = runner_cls._maintain_query_resources_once(runner, now=10)

    assert cached == {
        "query_count": 1,
        "capacity_cached": True,
        "capacity_error": "GCS temporarily unavailable",
    }
    assert coordinator.snapshot()["queries"][query_id]["expires_at"] == 40
    assert runner._query_resource_last_capacity_refresh_at == 5


def test_driver_cancels_query_when_fixed_actor_placement_node_is_lost():
    from duckdb.runners.ray.cluster_resource_coordinator import (
        ClusterQueryResourceCoordinator,
        NodeCapacity,
    )
    from duckdb.runners.ray.driver import RayQueryDriverActor
    from duckdb.runners.ray.query_graph_builder import (
        build_query_demand,
        build_query_execution_graph,
    )
    from duckdb.runners.ray.query_resource_runtime import register_query_graph

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    query_id = "query-actor-node-loss"
    graph = build_query_execution_graph(_metadata(query_id))
    node_resources = ResourceVector(
        cpu=8,
        gpu=1,
        heap_bytes=16 * _GIB,
        object_store_bytes=4 * _GIB,
    )
    coordinator = ClusterQueryResourceCoordinator(
        (NodeCapacity("node-a", node_resources),),
        heartbeat_timeout_s=30,
    )
    allocation = coordinator.register_query(
        build_query_demand(graph, node_resources),
        now=0,
    )
    manager = register_query_graph(graph, allocation)
    dropped = []

    runner._query_resource_coordinator = coordinator
    runner._query_graphs = {query_id: graph}
    runner._query_allocations = {query_id: allocation}
    runner._query_terminal_errors = {}
    runner._query_resource_lock = threading.Lock()

    def _drop_query_fragments(actual_query_id):
        assert runner._query_resource_lock.acquire(blocking=False)
        runner._query_resource_lock.release()
        dropped.append(actual_query_id)

    runner._get_plan_runner = lambda: SimpleNamespace(drop_query_fragments=_drop_query_fragments)

    coordinator.update_node_capacities((NodeCapacity("node-b", node_resources),))
    teardowns = runner_cls._synchronize_query_allocations(runner)
    runner_cls._run_query_allocation_teardowns(runner, teardowns)

    snapshot = manager.snapshot()
    assert snapshot["cancelled"] is True
    assert snapshot["cancel_reason"] == "ray_actor_placement_lost"
    assert snapshot["allocation_admission_open"] is False
    assert coordinator.snapshot()["queries"][query_id]["state"] == "ACTOR_PLACEMENT_LOST"
    assert "cannot migrate in place" in runner._query_terminal_errors[query_id]
    assert dropped == [query_id]
