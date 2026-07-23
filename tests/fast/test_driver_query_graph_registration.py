# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

import duckdb
from duckdb._ray_cxx import validate_plan_serialization_for_submission
from duckdb.runners.ray.query_execution_graph import (
    ActorPlacement,
    NodeResourceAllocation,
    QueryAllocation,
    ResourceVector,
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


class _FakeCoordinator:
    def __init__(self, events):
        self._events = events
        self.released = []
        self.allocations = {}

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
    runner._refresh_query_capacity = lambda: ResourceVector(
        cpu=8,
        gpu=1,
        heap_bytes=16 * _GIB,
        object_store_bytes=4 * _GIB,
    )
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
        "coordinator_register",
        "actors_created",
        "vllm_ready",
        "plan_runner",
        "actors_ready",
    ]
    manager = get_query_resource_manager(query_id)
    assert manager.snapshot()["stages"][f"stage:{query_id}:node:1:udf"]["actor_ready"] is True
    assert runner.curr_streams[query_id] == "stream"


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
    runner._get_plan_runner = lambda: SimpleNamespace(
        drop_query_fragments=lambda actual_query_id: dropped.append(actual_query_id)
    )

    coordinator.update_node_capacities((NodeCapacity("node-b", node_resources),))
    runner_cls._synchronize_query_allocations(runner)

    snapshot = manager.snapshot()
    assert snapshot["cancelled"] is True
    assert snapshot["cancel_reason"] == "ray_actor_placement_lost"
    assert snapshot["allocation_admission_open"] is False
    assert coordinator.snapshot()["queries"][query_id]["state"] == "ACTOR_PLACEMENT_LOST"
    assert "cannot migrate in place" in runner._query_terminal_errors[query_id]
    assert dropped == [query_id]
