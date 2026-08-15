# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import vane
from vane._ray_cxx import validate_plan_serialization_for_submission
from vane._ray_errors import RemoteRayException
from vane.runners.ray.cluster_resource_coordinator import NodeCapacity
from vane.runners.ray.query_resource_graph import (
    QueryAllocation,
    QueryResourceGraph,
    ResourceUnitSpec,
    ResourceVector,
)
from vane.runners.ray.query_resource_runtime import (
    clear_query_resource_managers,
    get_query_resource_manager,
)

_GIB = 1024**3
_OWNER_ID = "query-graph-test-owner"
_SESSION_ID = "query-graph-test-session"


class _FakeConnection:
    def __init__(self):
        self.closed = False

    def cursor(self):
        return _FakeConnection()

    def close(self):
        self.closed = True


class _FakeLogicalPlan:
    def __init__(self, physical_plan, events):
        self._physical_plan = physical_plan
        self._events = events

    def to_physical_plan(self, conn, effective_session_config):
        assert conn is not None
        assert effective_session_config == {}
        self._events.append("physical_plan")
        return self._physical_plan

    def idx(self):
        return self._physical_plan.idx()

    def operation_fingerprint(self):
        return f"test-plan:{self._physical_plan.idx()}"

    def session_id(self):
        return _SESSION_ID

    def session_config(self):
        return {}

    def has_explicit_s3_credentials(self):
        return False


class _ValidatingLogicalPlan(_FakeLogicalPlan):
    def to_physical_plan(self, conn, effective_session_config):
        physical_plan = super().to_physical_plan(conn, effective_session_config)
        validate_plan_serialization_for_submission(physical_plan)
        return physical_plan


class _FakePhysicalPlan:
    def __init__(self, query_id, metadata, events):
        self._query_id = query_id
        self._metadata = metadata
        self._events = events

    def idx(self):
        return self._query_id

    def session_id(self):
        return _SESSION_ID

    def session_config(self):
        return {}

    def collect_query_resource_graph_metadata(self, conn=None):
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
        self.states = {}
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
            generation=7,
        )
        self.allocations[demand.query_id] = allocation
        self.states[demand.query_id] = "RUNNING"
        return allocation

    def update_node_capacities(self, capacities):
        self.capacity_updates.append(tuple(capacities))
        return None

    def query_state(self, query_id, generation):
        allocation = self.allocations[query_id]
        if int(generation) != allocation.generation:
            raise ValueError("stale allocation generation")
        return self.states[query_id]

    def release_query(self, query_id, generation):
        self.released.append((query_id, generation))
        self._events.append("coordinator_release")
        self.allocations.pop(query_id, None)
        self.states.pop(query_id, None)
        return True

    def snapshot(self):
        return {
            "queries": {
                query_id: {
                    "allocation": allocation.to_dict(),
                    "state": self.states[query_id],
                }
                for query_id, allocation in self.allocations.items()
            }
        }


class _ZeroBudgetCoordinator(_FakeCoordinator):
    def register_query(self, demand):
        self._events.append("coordinator_register")
        allocation = QueryAllocation(
            resources=ResourceVector(),
            generation=7,
        )
        self.allocations[demand.query_id] = allocation
        self.states[demand.query_id] = "RUNNING"
        return allocation


class _QueryStateMustNotBeReadCoordinator(_FakeCoordinator):
    def query_state(self, query_id, generation):
        del query_id, generation
        raise AssertionError("aggregate soft-budget registration must not query a legacy admission state")


def _metadata(query_id: str) -> dict:
    return {
        "query_id": query_id,
        "nodes": [
            {
                "node_id": "0",
                "node_name": "ScanSource",
                "input_node_ids": [],
                "is_sink": False,
                "is_materialization_barrier": False,
                "materialized_input_node_ids": [],
                "num_partitions": 4,
                "udf_payload": None,
            },
            {
                "node_id": "1",
                "node_name": "StreamingUDF",
                "input_node_ids": ["0"],
                "is_sink": False,
                "is_materialization_barrier": False,
                "materialized_input_node_ids": [],
                "num_partitions": 4,
                "udf_payload": {
                    "query_id": query_id,
                    "resource_unit_id": f"resource:{query_id}:udf:node:1",
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
    from vane.runners.ray.driver import BoundedReplayMap, RayQueryDriverActor, _DriverSession

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    runner._duckdb_conn = _FakeConnection()
    runner._client_ids = {_OWNER_ID}
    runner._detaching_client_ids = set()
    runner._detached_client_results = BoundedReplayMap(capacity=65_536)
    runner._session_lock = threading.RLock()
    runner._closed_session_owners = BoundedReplayMap(capacity=65_536)
    runner._plan_lifecycles = {}
    runner._plan_session_ids = {}
    runner._plan_connections = {}
    runner._plan_teardown_condition = threading.Condition(runner._session_lock)
    runner._plan_teardowns_in_progress = set()
    runner._sessions = {
        _SESSION_ID: _DriverSession(
            owner_id=_OWNER_ID,
            config={},
            connection=runner._duckdb_conn.cursor(),
            s3_config={},
        )
    }
    runner._query_resource_coordinator = coordinator
    runner._query_resource_lock = threading.RLock()
    runner._query_allocations = {}
    runner._query_resource_graphs = {}
    runner._active_udf_actors = []
    runner._active_udf_actors_by_plan = {}
    runner._active_udf_actor_by_unit = {}
    runner._query_udf_actor_nodes = {}
    runner._query_udf_session_configs = {}
    runner._query_udf_actor_activation_tasks = {}
    runner._active_vllm_actors = []
    runner._active_vllm_actors_by_plan = {}
    runner.curr_plans = {}
    runner.curr_streams = {}
    runner._async_result_streams = {}
    runner._plan_query_ids = {}
    runner._query_terminal_errors = {}
    runner._query_resource_admission_loop = None
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
    runner._precreate_vllm_actors = lambda plan, *, query_connection, session_config: events.append("vllm_ready") or []
    runner._get_plan_runner = lambda: SimpleNamespace(run_plan=lambda plan, conn: "stream")
    return runner_cls, runner


def test_driver_starts_plan_runner_without_waiting_for_lazy_actor_pool():
    events = []
    coordinator = _FakeCoordinator(events)
    runner_cls, runner = _runner(events, coordinator)
    query_id = "query-driver-order"
    physical_plan = _FakePhysicalPlan(query_id, _metadata(query_id), events)

    def _precreate(plan, graph, *, query_connection, session_config):
        assert query_connection is not None
        assert session_config == {}
        assert graph.query_id == query_id
        manager = get_query_resource_manager(query_id)
        actor_unit = manager.snapshot()["units"][f"resource:{query_id}:udf:node:1"]
        assert actor_unit["actor_ready"] is False
        events.append("actor_locators")
        return []

    runner._precreate_udf_actors = _precreate
    vllm_pool = SimpleNamespace(shutdown=lambda: None)

    def _precreate_vllm(plan, *, query_connection, session_config):
        assert plan is physical_plan
        assert query_connection is not None
        assert session_config == {}
        events.append("vllm_ready")
        runner._active_vllm_actors.append(vllm_pool)
        runner._active_vllm_actors_by_plan[query_id] = [vllm_pool]
        return [vllm_pool]

    runner._precreate_vllm_actors = _precreate_vllm

    def _run_plan(plan, conn):
        manager = get_query_resource_manager(query_id)
        actor_unit = manager.snapshot()["units"][f"resource:{query_id}:udf:node:1"]
        assert actor_unit["actor_ready"] is False
        events.append("plan_runner")
        return "stream"

    runner._get_plan_runner = lambda: SimpleNamespace(run_plan=_run_plan)

    asyncio.run(
        runner_cls.run_plan(
            runner,
            _OWNER_ID,
            _SESSION_ID,
            _FakeLogicalPlan(physical_plan, events),
        )
    )

    assert events == [
        "physical_plan",
        "collect_graph",
        "capacity",
        "coordinator_register",
        "actor_locators",
        "vllm_ready",
        "plan_runner",
    ]
    manager = get_query_resource_manager(query_id)
    snapshot = manager.snapshot()
    assert snapshot["units"][f"resource:{query_id}:udf:node:1"]["actor_ready"] is False
    assert snapshot["submitted_actor_slots"] == []
    assert runner.curr_streams[query_id] == "stream"
    assert runner._active_vllm_actors_by_plan[query_id] == [vllm_pool]


def test_driver_opens_admission_with_a_zero_soft_budget_for_ray_core_liveness():
    events = []
    coordinator = _ZeroBudgetCoordinator(events)
    runner_cls, runner = _runner(events, coordinator)
    runner._precreate_udf_actors = lambda *_args, **_kwargs: []
    query_id = "query-pending-minimum"
    metadata = _metadata(query_id)
    metadata["nodes"][1]["udf_payload"]["execution_backend"] = "ray_task"
    physical_plan = _FakePhysicalPlan(query_id, metadata, events)

    asyncio.run(
        runner_cls.run_plan(
            runner,
            _OWNER_ID,
            _SESSION_ID,
            _FakeLogicalPlan(physical_plan, events),
        )
    )

    snapshot = get_query_resource_manager(query_id).snapshot()
    assert snapshot["allocation"]["resources"] == ResourceVector().to_dict()
    assert snapshot["allocation_admission_open"] is True
    assert snapshot["ray_core_owns_placement"] is True
    assert runner.curr_streams[query_id] == "stream"


def test_driver_does_not_read_a_legacy_coordinator_admission_state():
    events = []
    coordinator = _QueryStateMustNotBeReadCoordinator(events)
    runner_cls, runner = _runner(events, coordinator)
    runner._precreate_udf_actors = lambda *_args, **_kwargs: []
    query_id = "query-invalid-coordinator-state"
    physical_plan = _FakePhysicalPlan(query_id, _metadata(query_id), events)

    asyncio.run(
        runner_cls.run_plan(
            runner,
            _OWNER_ID,
            _SESSION_ID,
            _FakeLogicalPlan(physical_plan, events),
        )
    )

    assert get_query_resource_manager(query_id).allocation.generation == 7
    assert coordinator.released == []
    assert runner._query_resource_graphs[query_id].query_id == query_id
    assert runner._query_allocations[query_id].generation == 7


def test_run_plan_does_not_read_physical_plan_id_after_registration():
    events: list[str] = []
    coordinator = _FakeCoordinator(events)
    runner_cls, runner = _runner(events, coordinator)
    runner._precreate_udf_actors = lambda *_args, **_kwargs: []
    query_id = "query-single-use-plan-id"
    physical_plan = _RegistrationOnlyIdxPhysicalPlan(
        query_id,
        _metadata(query_id),
        events,
    )

    asyncio.run(
        runner_cls.run_plan(
            runner,
            _OWNER_ID,
            _SESSION_ID,
            _FakeLogicalPlan(physical_plan, events),
        )
    )

    assert physical_plan.idx_calls == 2
    assert runner._plan_query_ids[query_id] == query_id


def test_run_plan_cancellation_releases_registration_before_startup_worker_claim():
    events: list[str] = []
    coordinator = _FakeCoordinator(events)
    runner_cls, runner = _runner(events, coordinator)
    query_id = "query-cancelled-before-startup"
    physical_plan = _FakePhysicalPlan(query_id, _metadata(query_id), events)
    blocker_started = threading.Event()
    blocker_release = threading.Event()
    startup_entered = threading.Event()
    native_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="vane-test-native-saturated",
    )
    runner._driver_native_executor = native_executor

    runner._precreate_udf_actors = lambda *_args, **_kwargs: startup_entered.set() or []

    async def _exercise() -> None:
        loop = asyncio.get_running_loop()
        registration_complete = asyncio.Event()
        blocker_future = None
        register_query_resources = runner_cls._register_query_resources

        def _occupy_native_executor() -> None:
            blocker_started.set()
            assert blocker_release.wait(timeout=2.0)

        async def _register_then_saturate(
            plan,
            *,
            query_connection,
            expected_plan_id=None,
        ):
            nonlocal blocker_future
            registered = await register_query_resources(
                runner,
                plan,
                query_connection=query_connection,
                expected_plan_id=expected_plan_id,
            )
            blocker_future = loop.run_in_executor(
                native_executor,
                _occupy_native_executor,
            )
            while not blocker_started.is_set():
                await asyncio.sleep(0)
            registration_complete.set()
            return registered

        runner._register_query_resources = _register_then_saturate
        run_plan = asyncio.create_task(
            runner_cls.run_plan(
                runner,
                _OWNER_ID,
                _SESSION_ID,
                _FakeLogicalPlan(physical_plan, events),
            )
        )
        try:
            await asyncio.wait_for(
                registration_complete.wait(),
                timeout=1.0,
            )
            await asyncio.sleep(0)
            assert startup_entered.is_set() is False
            run_plan.cancel()
            await asyncio.sleep(0)
            blocker_release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(run_plan, timeout=1.0)
            assert blocker_future is not None
            await blocker_future
        finally:
            blocker_release.set()

    try:
        asyncio.run(_exercise())
    finally:
        blocker_release.set()
        runner_cls._shutdown_driver_executors(runner)

    with pytest.raises(KeyError, match="query resource graph is not registered"):
        get_query_resource_manager(query_id)
    assert startup_entered.is_set() is False
    assert coordinator.released == [(query_id, 7)]
    assert query_id not in runner._query_resource_graphs
    assert query_id not in runner._query_allocations


def test_run_plan_cancellation_after_startup_claim_tears_down_once():
    events: list[str] = []
    coordinator = _FakeCoordinator(events)
    runner_cls, runner = _runner(events, coordinator)
    query_id = "query-cancelled-after-startup-claim"
    physical_plan = _FakePhysicalPlan(query_id, _metadata(query_id), events)
    startup_claimed = threading.Event()
    startup_release = threading.Event()
    fragment_drops: list[str] = []

    def _precreate_udf_actors(*_args, **_kwargs):
        startup_claimed.set()
        assert startup_release.wait(timeout=2.0)
        return []

    def _drop_query_fragments(actual_query_id: str) -> None:
        fragment_drops.append(actual_query_id)
        runner._release_query_resources(
            actual_query_id,
            reason="test_cancelled_during_startup",
        )

    runner._precreate_udf_actors = _precreate_udf_actors
    runner._drop_query_fragments_sync = _drop_query_fragments

    async def _exercise() -> None:
        run_plan = asyncio.create_task(
            runner_cls.run_plan(
                runner,
                _OWNER_ID,
                _SESSION_ID,
                _FakeLogicalPlan(physical_plan, events),
            )
        )
        try:
            while not startup_claimed.is_set():
                await asyncio.sleep(0)
            run_plan.cancel()
            startup_release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(run_plan, timeout=1.0)
        finally:
            startup_release.set()

    asyncio.run(_exercise())

    with pytest.raises(KeyError, match="query resource graph is not registered"):
        get_query_resource_manager(query_id)
    assert fragment_drops == []
    assert coordinator.released == [(query_id, 7)]
    assert query_id not in runner.curr_plans
    assert query_id not in runner.curr_streams
    assert query_id not in runner._plan_query_ids
    assert query_id not in runner._query_resource_graphs
    assert query_id not in runner._query_allocations


def test_driver_rolls_back_graph_and_cluster_allocation_when_actor_initialization_fails():
    events = []
    coordinator = _FakeCoordinator(events)
    runner_cls, runner = _runner(events, coordinator)
    query_id = "query-driver-rollback"
    physical_plan = _FakePhysicalPlan(query_id, _metadata(query_id), events)

    def _fail_precreate(plan, graph, *, query_connection, session_config):
        assert query_connection is not None
        assert session_config == {}
        events.append("actors_initializing")
        raise RuntimeError("model initialization failed")

    runner._precreate_udf_actors = _fail_precreate

    with pytest.raises(RuntimeError, match="model initialization failed"):
        asyncio.run(
            runner_cls.run_plan(
                runner,
                _OWNER_ID,
                _SESSION_ID,
                _FakeLogicalPlan(physical_plan, events),
            )
        )

    with pytest.raises(KeyError, match="query resource graph is not registered"):
        get_query_resource_manager(query_id)
    assert coordinator.released == [(query_id, 7)]
    assert query_id not in runner.curr_plans
    assert "plan_runner" not in events


def test_copy_registration_keeps_streaming_udf_admission_bounded_when_ray_nodes_is_delayed(
    monkeypatch,
):
    from vane.runners.ray import driver as driver_module
    from vane.runners.ray.query_resource_runtime import register_query_resource_graph

    events: list[str] = []
    coordinator = _FakeCoordinator(events)
    runner_cls, runner = _runner(events, coordinator)
    runner._query_resource_lock = threading.Lock()
    runner._read_query_node_capacities = lambda: runner_cls._read_query_node_capacities()
    runner._precreate_udf_actors = lambda *_args, **_kwargs: []
    runner._precreate_vllm_actors = lambda *_args, **_kwargs: []
    runner._get_plan_runner = lambda: SimpleNamespace(
        run_copy_plan=lambda _plan, _conn, on_execution_started: (
            on_execution_started()
            or {
                "rows_copied": 1,
                "copy_output_committed": True,
            }
        ),
    )
    runner._build_local_progress_snapshot = lambda query_id, _started_at: {
        "query_id": query_id,
        "state": "FINISHED",
    }
    runner._teardown_plan_resources = lambda *_args, **_kwargs: None
    runner._open_query_resource_admission = lambda _query_id: None

    streaming_query_id = "query-streaming-admission"
    streaming_unit = ResourceUnitSpec(
        query_id=streaming_query_id,
        resource_unit_id=f"resource:{streaming_query_id}:udf",
        physical_node_id="node:streaming:udf",
        unit_kind="ray_task_udf",
        backend="ray_task",
        input_unit_ids=(),
        per_task=ResourceVector(cpu=1, heap_bytes=128),
        target_output_block_bytes=64,
        generator_buffer_blocks=1,
        max_concurrency=None,
    )
    streaming_graph = QueryResourceGraph(
        query_id=streaming_query_id,
        plan_digest="sha256:streaming-admission",
        units=(streaming_unit,),
        terminal_unit_ids=(streaming_unit.resource_unit_id,),
    )
    streaming_resources = ResourceVector(
        cpu=2,
        heap_bytes=4096,
        object_store_bytes=4096,
    )
    streaming_allocation = QueryAllocation(
        resources=streaming_resources,
        generation=1,
    )
    streaming_manager = register_query_resource_graph(
        streaming_graph,
        streaming_allocation,
    )
    streaming_manager.update_unit_state(
        streaming_unit.resource_unit_id,
        runnable=True,
    )
    coordinator.allocations[streaming_query_id] = streaming_allocation
    coordinator.states[streaming_query_id] = "RUNNING"
    runner._query_resource_graphs[streaming_query_id] = streaming_graph
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
        "resource_unit_id": streaming_unit.resource_unit_id,
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
        "query_generation_capability": runner_cls._issue_query_task_admission_capability(
            runner,
            streaming_query_id,
        ),
    }

    async def _run_concurrently():
        copy_task = asyncio.create_task(
            runner_cls.run_copy_plan(
                runner,
                _OWNER_ID,
                _SESSION_ID,
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

    assert outcome.result == {
        "rows_copied": 1,
        "copy_output_committed": True,
        "copy_operation_id": copy_query_id,
        "copy_write_state": "committed",
        "copy_cleanup_state": "complete",
        "copy_cleanup_warnings": [],
    }
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


def test_registration_does_not_overwrite_a_newer_empty_capacity_snapshot():
    events: list[str] = []
    coordinator = _FakeCoordinator(events)
    runner_cls, runner = _runner(events, coordinator)
    older = NodeCapacity(
        "node-old",
        ResourceVector(cpu=2, object_store_bytes=1 * _GIB),
    )
    runner._query_node_capacities = ()
    runner._query_resource_last_capacity_refresh_at = 20.0

    capacity = runner_cls._apply_query_capacity_snapshot(
        runner,
        (older,),
        snapshot_started_at=10.0,
    )

    assert capacity == ResourceVector()
    assert runner._query_node_capacities == ()
    assert runner._query_resource_last_capacity_refresh_at == 20.0
    assert coordinator.capacity_updates == []


@pytest.mark.parametrize("entrypoint", ["run_plan", "run_copy_plan"])
def test_driver_rejects_non_serializable_plan_before_query_registration(entrypoint):
    events = []
    coordinator = _FakeCoordinator(events)
    runner_cls, runner = _runner(events, coordinator)
    query_id = f"query-plan-serialization-failure-{entrypoint}"
    physical_plan = vane.ray_cxx._make_non_serializable_physical_plan_for_test(query_id)

    with pytest.raises(
        RuntimeError,
        match=f"distributed physical plan serialization preflight failed for query_id={query_id}",
    ) as exc_info:
        coroutine = getattr(runner_cls, entrypoint)(
            runner,
            _OWNER_ID,
            _SESSION_ID,
            _ValidatingLogicalPlan(physical_plan, events),
        )
        asyncio.run(coroutine)

    assert isinstance(exc_info.value, RemoteRayException)
    assert isinstance(exc_info.value.__cause__, vane.NotImplementedException)
    assert "INTENTIONALLY_NON_SERIALIZABLE operator cannot be serialized" in str(exc_info.value.__cause__)
    with pytest.raises(KeyError, match="query resource graph is not registered"):
        get_query_resource_manager(query_id)
    assert coordinator.released == []
    assert coordinator.allocations == {}
    assert runner._query_resource_graphs == {}
    assert runner._query_allocations == {}
    assert query_id not in runner.curr_plans
    assert query_id not in runner.curr_streams
    assert query_id not in runner._plan_query_ids
    assert events == ["physical_plan"]


def test_driver_exposes_query_task_and_output_lease_api():
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    required = {
        "acquire_query_task_lease",
        "release_query_task_lease",
        "handoff_query_task_lease_to_teardown",
        "acquire_query_output_block_lease",
        "handoff_query_output_block_lease",
        "release_query_output_block_lease",
    }
    assert required.issubset(dir(runner_cls))


def test_driver_maintenance_refreshes_ray_capacity_usage_and_heartbeat_atomically():
    from vane.runners.ray.cluster_resource_coordinator import (
        ClusterQueryResourceCoordinator,
        NodeCapacity,
    )
    from vane.runners.ray.driver import RayQueryDriverActor
    from vane.runners.ray.query_resource_graph_builder import (
        build_query_demand,
        build_query_resource_graph,
    )
    from vane.runners.ray.query_resource_manager import TaskRequest
    from vane.runners.ray.query_resource_runtime import register_query_resource_graph

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    query_id = "query-driver-maintenance"
    graph = build_query_resource_graph(_metadata(query_id))
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
        (initial_node,),
    )
    allocation = coordinator.register_query(demand, now=0)
    manager = register_query_resource_graph(graph, allocation)
    for unit in graph.units:
        manager.update_unit_state(
            unit.resource_unit_id,
            runnable=True,
        )
    native_fragment_unit = next(unit for unit in graph.units if unit.backend == "ray_worker")
    task_grant = manager.try_acquire_task(
        TaskRequest(
            query_id=query_id,
            resource_unit_id=native_fragment_unit.resource_unit_id,
            task_id="fte-task-1",
            attempt_id="fte-attempt-1",
            node_id="node-a",
        )
    )
    assert task_grant.granted
    actor_unit = next(unit for unit in graph.units if unit.backend == "ray_actor")
    manager.set_submitted_actor_slots(actor_unit.resource_unit_id, {0})
    manager.set_ready_actor_slots(actor_unit.resource_unit_id, {0: "node-a"})

    runner._query_resource_lock = threading.RLock()
    runner._query_resource_coordinator = coordinator
    runner._query_resource_graphs = {query_id: graph}
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
    assert manager_snapshot["soft_allocation_usage"] != manager_snapshot["usage"]
    assert query_snapshot["observed_usage"] == manager_snapshot["soft_allocation_usage"]
    assert query_snapshot["observed_usage"]["cpu"] > manager_snapshot["usage"]["cpu"]
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

    grown_node = NodeCapacity(
        "node-a",
        ResourceVector(
            cpu=16,
            gpu=2,
            heap_bytes=32 * _GIB,
            object_store_bytes=8 * _GIB,
        ),
    )
    runner_cls._maintain_query_resources_once(
        runner,
        capacities=(grown_node,),
        now=15,
    )

    grown_allocation = coordinator.snapshot()["queries"][query_id]["allocation"]["resources"]
    assert (
        grown_allocation
        == ResourceVector(
            cpu=1,
            gpu=1,
            heap_bytes=4 * _GIB,
            object_store_bytes=8 * _GIB,
        ).to_dict()
    )


def test_driver_maintenance_reuses_a_valid_empty_capacity_snapshot_after_gcs_failure():
    from vane.runners.ray.cluster_resource_coordinator import ClusterQueryResourceCoordinator
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    runner._query_resource_lock = threading.RLock()
    runner._query_resource_coordinator = ClusterQueryResourceCoordinator(())
    runner._query_resource_graphs = {}
    runner._query_allocations = {}
    runner._query_node_capacities = ()
    runner._query_resource_last_capacity_refresh_at = 5.0

    def _capacity_unavailable():
        raise RuntimeError("GCS temporarily unavailable")

    runner._read_query_node_capacities = _capacity_unavailable

    result = runner_cls._maintain_query_resources_once(runner, now=10)

    assert result == {
        "query_count": 0,
        "capacity_cached": True,
        "capacity_error": "GCS temporarily unavailable",
    }
    assert runner._query_node_capacities == ()
    assert runner._query_resource_last_capacity_refresh_at == 5.0


def test_driver_keeps_aggregate_soft_reservation_when_capacity_moves_nodes():
    from vane.runners.ray.cluster_resource_coordinator import (
        ClusterQueryResourceCoordinator,
        NodeCapacity,
    )
    from vane.runners.ray.driver import RayQueryDriverActor
    from vane.runners.ray.query_resource_graph_builder import (
        build_query_demand,
        build_query_resource_graph,
    )
    from vane.runners.ray.query_resource_runtime import register_query_resource_graph

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    query_id = "query-actor-node-loss"
    graph = build_query_resource_graph(_metadata(query_id))
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
        build_query_demand(graph, (NodeCapacity("node-a", node_resources),)),
        now=0,
    )
    manager = register_query_resource_graph(graph, allocation)
    dropped = []

    runner._query_resource_coordinator = coordinator
    runner._query_resource_graphs = {query_id: graph}
    runner._query_allocations = {query_id: allocation}
    runner._query_terminal_errors = {}
    runner._query_resource_lock = threading.Lock()

    def _drop_query_fragments(actual_query_id):
        assert runner._query_resource_lock.acquire(blocking=False)
        runner._query_resource_lock.release()
        dropped.append(actual_query_id)

    runner._get_plan_runner = lambda: SimpleNamespace(drop_query_fragments=_drop_query_fragments)

    coordinator.update_node_capacities((NodeCapacity("node-b", node_resources),))
    runner_cls._synchronize_query_allocations(runner)

    snapshot = manager.snapshot()
    assert snapshot["cancelled"] is False
    assert snapshot["allocation_admission_open"] is True
    assert snapshot["allocation"]["resources"] == allocation.resources.to_dict()
    assert "node_allocations" not in snapshot["allocation"]
    assert snapshot["ray_core_owns_placement"] is True
    assert coordinator.snapshot()["queries"][query_id]["state"] == "RUNNING"
    assert runner._query_terminal_errors == {}
    assert dropped == []


def test_unrelated_rebalance_cannot_reopen_a_pending_phase_frontier():
    from vane.runners.ray.cluster_resource_coordinator import (
        ClusterQueryResourceCoordinator,
    )
    from vane.runners.ray.driver import RayQueryDriverActor
    from vane.runners.ray.query_resource_graph import MaterializationBarrierSpec
    from vane.runners.ray.query_resource_graph_builder import build_query_demand
    from vane.runners.ray.query_resource_manager import TaskRequest
    from vane.runners.ray.query_resource_runtime import register_query_resource_graph

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    query_id = "query-phase-fence-rebalance"
    upstream = ResourceUnitSpec(
        query_id=query_id,
        resource_unit_id=f"resource:{query_id}:upstream",
        physical_node_id="node:upstream:udf",
        unit_kind="ray_task_udf",
        backend="ray_task",
        input_unit_ids=(),
        per_task=ResourceVector(cpu=1, heap_bytes=10),
        target_output_block_bytes=0,
        generator_buffer_blocks=0,
        max_concurrency=None,
    )
    materializer = ResourceUnitSpec(
        query_id=query_id,
        resource_unit_id=f"resource:{query_id}:materializer",
        physical_node_id="node:materializer:native-fragment",
        unit_kind="native_fragment",
        backend="ray_worker",
        input_unit_ids=(upstream.resource_unit_id,),
        per_task=ResourceVector(),
        target_output_block_bytes=0,
        generator_buffer_blocks=0,
        max_concurrency=4,
    )
    downstream = ResourceUnitSpec(
        query_id=query_id,
        resource_unit_id=f"resource:{query_id}:downstream",
        physical_node_id="node:downstream:udf",
        unit_kind="ray_task_udf",
        backend="ray_task",
        input_unit_ids=(materializer.resource_unit_id,),
        per_task=ResourceVector(cpu=1, heap_bytes=20),
        target_output_block_bytes=0,
        generator_buffer_blocks=0,
        max_concurrency=None,
    )
    graph = QueryResourceGraph(
        query_id=query_id,
        plan_digest="sha256:phase-fence-rebalance",
        units=(upstream, materializer, downstream),
        terminal_unit_ids=(downstream.resource_unit_id,),
        materialization_barriers=(
            MaterializationBarrierSpec(
                query_id=query_id,
                barrier_id=f"barrier:{query_id}:node:materializer",
                physical_node_id="materializer",
                materializer_unit_id=materializer.resource_unit_id,
                materialized_input_unit_ids=(upstream.resource_unit_id,),
            ),
        ),
    )
    node = NodeCapacity(
        "node-a",
        ResourceVector(cpu=2, heap_bytes=30, object_store_bytes=100),
    )
    coordinator = ClusterQueryResourceCoordinator((node,))
    allocation = coordinator.register_query(build_query_demand(graph, (node,)), now=0)
    transitions = []
    manager = register_query_resource_graph(
        graph,
        allocation,
        on_eligible_units_change=lambda eligible, fence_epoch: transitions.append((eligible, fence_epoch)),
    )
    for unit in graph.units:
        manager.update_unit_state(unit.resource_unit_id, runnable=True)

    runner._query_resource_coordinator = coordinator
    runner._query_resource_graphs = {query_id: graph}
    runner._query_allocations = {query_id: allocation}
    runner._query_node_capacities = (node,)
    runner._query_resource_last_capacity_refresh_at = 0.0
    runner._query_resource_lock = threading.RLock()
    runner._session_lock = threading.RLock()
    runner._plan_teardown_condition = threading.Condition(runner._session_lock)
    runner._plan_teardowns_in_progress = set()
    runner._plan_session_ids = {query_id: _SESSION_ID}
    runner._active_udf_actor_by_unit = {}
    runner._active_udf_actors = []
    runner._active_udf_actors_by_plan = {}
    runner._signal_query_resource_change = lambda _query_id: None

    assert manager.mark_materialization_barrier_completed_for_node("materializer")
    assert len(transitions) == 1
    eligible, fence_epoch = transitions[0]
    assert manager.snapshot()["allocation_admission_open"] is False

    runner_cls._maintain_query_resources_once(
        runner,
        capacities=(node,),
        now=5,
    )
    unrelated_generation = coordinator.snapshot()["queries"][query_id]["allocation"]["generation"]
    assert unrelated_generation > allocation.generation

    blocked = manager.try_acquire_task(
        TaskRequest(query_id, downstream.resource_unit_id, "before-phase-refresh", "0", None)
    )
    pending_snapshot = manager.snapshot()
    assert pending_snapshot["allocation"]["generation"] == unrelated_generation
    assert pending_snapshot["allocation_admission_open"] is False
    assert not blocked.granted and blocked.blocked_reason == "allocation_pending"

    runner_cls._transition_query_execution_phase(
        runner,
        query_id,
        eligible,
        fence_epoch,
    )

    opened = manager.try_acquire_task(
        TaskRequest(query_id, downstream.resource_unit_id, "after-phase-refresh", "0", None)
    )
    opened_snapshot = manager.snapshot()
    assert opened_snapshot["allocation"]["generation"] > unrelated_generation
    assert opened_snapshot["allocation_admission_open"] is True
    assert opened.granted


def test_driver_keeps_drain_admission_open_after_soft_budget_shrink():
    from vane.runners.ray.driver import RayQueryDriverActor
    from vane.runners.ray.query_resource_manager import TaskRequest
    from vane.runners.ray.query_resource_runtime import register_query_resource_graph

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    query_id = "query-allocation-debt-drain"
    ray_task = ResourceUnitSpec(
        query_id=query_id,
        resource_unit_id=f"resource:{query_id}:udf:node:1",
        physical_node_id="node:1:udf",
        unit_kind="ray_task_udf",
        backend="ray_task",
        input_unit_ids=(),
        per_task=ResourceVector(cpu=1, heap_bytes=100),
        target_output_block_bytes=0,
        generator_buffer_blocks=0,
        max_concurrency=None,
    )
    native = ResourceUnitSpec(
        query_id=query_id,
        resource_unit_id=f"resource:{query_id}:fragment:node:2",
        physical_node_id="node:2:native-fragment",
        unit_kind="native_fragment",
        backend="ray_worker",
        input_unit_ids=(),
        per_task=ResourceVector(),
        target_output_block_bytes=0,
        generator_buffer_blocks=0,
        max_concurrency=4,
    )
    graph = QueryResourceGraph(
        query_id=query_id,
        plan_digest="sha256:allocation-debt-drain",
        units=(ray_task, native),
        terminal_unit_ids=(ray_task.resource_unit_id, native.resource_unit_id),
    )
    initial_resources = ResourceVector(cpu=2, heap_bytes=200)
    initial_allocation = QueryAllocation(
        resources=initial_resources,
        generation=1,
    )
    manager = register_query_resource_graph(graph, initial_allocation)
    manager.update_unit_state(ray_task.resource_unit_id, runnable=True)
    manager.update_unit_state(native.resource_unit_id, runnable=True)
    for index in range(2):
        grant = manager.try_acquire_task(TaskRequest(query_id, ray_task.resource_unit_id, f"ray-{index}", "0", None))
        assert grant.granted

    debt_allocation = QueryAllocation(resources=ResourceVector(), generation=2)
    runner._query_resource_coordinator = SimpleNamespace(
        snapshot=lambda: {
            "queries": {
                query_id: {
                    "allocation": debt_allocation.to_dict(),
                    "state": "RUNNING",
                }
            }
        }
    )
    runner._query_resource_graphs = {query_id: graph}
    runner._query_allocations = {query_id: initial_allocation}

    runner_cls._synchronize_query_allocations(runner)

    snapshot = manager.snapshot()
    assert snapshot["allocation_admission_open"] is True
    assert snapshot["soft_allocation_debt"] == ResourceVector(cpu=2, heap_bytes=200).to_dict()
    blocked = manager.try_acquire_task(TaskRequest(query_id, ray_task.resource_unit_id, "ray-new", "0", None))
    assert not blocked.granted and blocked.blocked_reason == "liveness_task_active"
    drain = manager.try_acquire_task(TaskRequest(query_id, native.resource_unit_id, "native-drain", "0", "node-a"))
    assert drain.granted and not drain.liveness
