# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import pickle
import threading
import time
import uuid
import weakref
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import vane
from tests.result_stream_helpers import collect_result_stream
from vane.runners.common import PartitionMetadata
from vane.runners.fte.fte_exchange import ExchangeSinkHandle, ExchangeSinkInstanceHandle
from vane.runners.ray import driver, partition_metadata
from vane.runners.ray.partition_metadata import PartitionMetadataAccessor
from vane.runners.ray.safe_get import QueryDeadlineExceeded
from vane.runners.ray.worker import (
    _normalize_native_task_result,
    _validate_fte_output_publication,
)


@contextmanager
def _registered_low_level_plan(
    plan,
    con,
    *,
    node_id=None,
    refresh_phase_allocation=False,
):
    """Exercise the internal C++ runner under the mandatory graph contract."""
    from vane.runners.ray.query_resource_graph import (
        QueryAllocation,
        ResourceVector,
    )
    from vane.runners.ray.query_resource_graph_builder import build_query_resource_graph
    from vane.runners.ray.query_resource_runtime import (
        register_query_resource_graph,
        release_query_resource_manager,
    )

    if node_id is None:
        import ray

        if not ray.is_initialized():
            raise RuntimeError("low-level distributed plan registration requires an initialized Ray runtime")
        node_id = str(ray.get_runtime_context().get_node_id())
    node_id = str(node_id).strip()
    if not node_id:
        raise ValueError("low-level distributed plan registration requires a non-empty node_id")

    graph = build_query_resource_graph(
        plan.collect_query_resource_graph_metadata(conn=con),
        env={
            "VANE_TARGET_OUTPUT_BLOCK_BYTES": str(1024**2),
        },
    )
    allocation_resources = ResourceVector(
        cpu=128,
        gpu=8,
        heap_bytes=1 << 50,
        object_store_bytes=1 << 50,
    )
    generation = 1
    manager_holder = {}

    def _refresh_phase_allocation(_eligible_unit_ids, allocation_fence_epoch):
        nonlocal generation
        if not refresh_phase_allocation:
            return
        generation += 1
        manager_holder["manager"].update_allocation(
            QueryAllocation(
                resources=allocation_resources,
                generation=generation,
            ),
            reopen_fence_epoch=allocation_fence_epoch,
        )

    manager = register_query_resource_graph(
        graph,
        QueryAllocation(
            resources=allocation_resources,
            generation=1,
        ),
        on_eligible_units_change=_refresh_phase_allocation,
    )
    manager_holder["manager"] = manager
    for unit in graph.units:
        manager.update_unit_state(
            unit.resource_unit_id,
            runnable=True,
        )
    try:
        yield graph
    finally:
        release_query_resource_manager(graph.query_id, reason="test_complete")


def _make_test_physical_plan(con=None):
    con = vane.connect() if con is None else con
    relation = con.sql("SELECT 1 AS i")
    return vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)


class _DummyStream:
    def __init__(self, items):
        self.items = list(items)
        self.loop = None
        self.ready_callback = None

    def next_nowait(self):
        if not self.items:
            raise StopIteration
        return self.items.pop(0)

    def set_ready_callback(self, loop, callback):
        self.loop = loop
        self.ready_callback = callback

    def arm_ready_notification(self):
        raise AssertionError("an exhausted or readable dummy stream must not arm a notification")

    def clear_ready_callback(self):
        self.loop = None
        self.ready_callback = None


class _NotifyingStream:
    def __init__(self):
        self.items = []
        self.closed = False
        self.loop = None
        self.ready_callback = None
        self.armed = asyncio.Event()

    def next_nowait(self):
        if self.items:
            return self.items.pop(0)
        if self.closed:
            raise StopIteration
        return None

    def set_ready_callback(self, loop, callback):
        self.loop = loop
        self.ready_callback = callback

    def arm_ready_notification(self):
        self.armed.set()
        if self.items or self.closed:
            self.loop.call_soon(self.ready_callback)

    def clear_ready_callback(self):
        self.loop = None
        self.ready_callback = None

    def publish(self, item):
        self.items.append(item)
        if self.loop is not None and self.ready_callback is not None:
            self.loop.call_soon_threadsafe(self.ready_callback)


class _FakeOutputLeaseOwner:
    def __init__(self) -> None:
        self.states = ["unit_queue"]
        self.released = False

    def transition_to(self, state):
        self.states.append(str(state))
        return True

    def release(self):
        if self.released:
            return False
        self.released = True
        return True


_TEST_RUNTIME_OWNER_ID = "test-runtime-owner"
_TEST_SESSION_ID = "test-vane-session"
_TEST_SESSION_CONFIG: dict[str, str] = {}


class _FakePhysicalPlanWithoutPlanAttr:
    def __init__(
        self,
        plan_id: str = "fake-plan",
        *,
        session_id: str = _TEST_SESSION_ID,
        session_config: dict[str, str] | None = None,
    ) -> None:
        self._plan_id = plan_id
        self._session_id = session_id
        self._session_config = dict(_TEST_SESSION_CONFIG if session_config is None else session_config)

    def idx(self) -> str:
        return self._plan_id

    def session_id(self) -> str:
        return self._session_id

    def session_config(self) -> dict[str, str]:
        return dict(self._session_config)

    def has_explicit_s3_credentials(self) -> bool:
        return False


class _FakeLogicalPlan:
    def __init__(self, physical_plan: _FakePhysicalPlanWithoutPlanAttr) -> None:
        self.physical_plan = physical_plan

    def idx(self) -> str:
        return self.physical_plan.idx()

    def to_physical_plan(self, _conn, _effective_session_config):
        return self.physical_plan

    def session_id(self) -> str:
        return self.physical_plan.session_id()

    def session_config(self) -> dict[str, str]:
        return self.physical_plan.session_config()

    def has_explicit_s3_credentials(self) -> bool:
        return self.physical_plan.has_explicit_s3_credentials()


class _FakeConnection:
    def __init__(self) -> None:
        self.cursors: list[_FakeConnection] = []
        self.statements: list[str] = []
        self.closed = False

    def cursor(self) -> _FakeConnection:
        cursor = _FakeConnection()
        self.cursors.append(cursor)
        return cursor

    def execute(self, statement: str) -> _FakeConnection:
        self.statements.append(statement)
        return self

    def close(self) -> None:
        self.closed = True


def _initialize_test_query_driver_client(client, opened_sessions=None):
    client._lease_token = "test-client-lease-token"
    client._client_lease_timeout_s = 60.0
    client._client_heartbeat_interval_s = 10.0
    client._client_heartbeat_rpc_timeout_s = 10.0
    client._client_heartbeat_stop = threading.Event()
    client._client_heartbeat_thread = None
    client._client_heartbeat_error = ""
    client._opened_sessions = dict(opened_sessions or {})
    client._uncertain_sessions = {}
    client._opening_session_ids = set()
    client._closing_session_ids = set()
    client._closed_session_ids = driver.BoundedReplayMap(capacity=65_536)
    client._session_closes_in_progress = set()
    client._session_condition = threading.Condition()
    client._client_closing = False
    client._client_close_in_progress = False


def _make_local_query_driver_actor():
    cls = driver.RayQueryDriverActor.__ray_metadata__.modified_class
    runner = cls.__new__(cls)
    runner.curr_streams = {}
    runner.curr_plans = {}
    runner._async_result_streams = {}
    runner._plan_query_ids = {}
    runner._query_terminal_errors = {}
    runner._leased_result_partition_refs = {}
    runner._result_partition_ref_counters = {}
    runner._query_resource_graphs = {}
    runner._query_allocations = {}
    runner._duckdb_conn = _FakeConnection()
    runner.plan_runner = None
    runner._active_udf_actors = []
    runner._active_udf_actors_by_plan = {}
    runner._active_udf_actor_by_unit = {}
    runner._query_udf_actor_lifecycle_locks = {}
    runner._query_udf_actor_nodes = {}
    runner._query_udf_session_configs = {}
    runner._query_udf_actor_activation_tasks = {}
    runner._query_resource_admission_loop = None
    runner._active_vllm_actors = []
    runner._active_vllm_actors_by_plan = {}
    runner._plan_runner_lifecycle_lock = threading.RLock()
    runner._driver_shutdown_started = False
    runner._driver_shutdown_complete = False
    runner._client_ids = {_TEST_RUNTIME_OWNER_ID}
    runner._client_lease_timeout_s = 60.0
    runner._client_heartbeat_interval_s = 10.0
    runner._client_leases = {
        _TEST_RUNTIME_OWNER_ID: driver._ClientOwnerLease(
            lease_token="test-owner-lease-token",
            expires_at=float("inf"),
        )
    }
    runner._detaching_client_ids = set()
    runner._detached_client_results = driver.BoundedReplayMap(capacity=65_536)
    runner._client_detach_locks = {}
    runner._expired_client_cleanup_tasks = {}
    runner._client_lease_maintenance_task = None
    runner._client_lease_maintenance_stop = None
    runner._client_lease_maintenance_error = ""
    runner._client_lease_maintenance_failures = 0
    runner._session_lock = threading.RLock()
    runner._closed_session_owners = driver.BoundedReplayMap(capacity=65_536)
    runner._plan_lifecycles = {}
    runner._plan_session_ids = {}
    runner._plan_connections = {}
    runner._plan_teardown_condition = threading.Condition(runner._session_lock)
    runner._plan_teardowns_in_progress = set()
    session_connection = runner._duckdb_conn.cursor()
    runner._sessions = {
        _TEST_SESSION_ID: driver._DriverSession(
            owner_id=_TEST_RUNTIME_OWNER_ID,
            config=dict(_TEST_SESSION_CONFIG),
            connection=session_connection,
            s3_config={},
        )
    }
    runner._test_session_connection = session_connection
    return cls, runner


def _query_registration_stub(query_id: str):
    async def _register(
        _self,
        _plan,
        *,
        query_connection,
        expected_plan_id=None,
    ):
        del query_connection, expected_plan_id
        return SimpleNamespace(query_id=query_id, units=()), object()

    return _register


def _run_actor_copy_plan(runner, plan):
    return runner.run_copy_plan(
        _TEST_RUNTIME_OWNER_ID,
        _TEST_SESSION_ID,
        plan,
    )


def _run_actor_stream_plan(runner, plan):
    return runner.run_plan(
        _TEST_RUNTIME_OWNER_ID,
        _TEST_SESSION_ID,
        plan,
    )


def _committed_copy_result(**values):
    result = {
        "rows_copied": 1,
        "copy_output_committed": True,
    }
    result.update(values)
    return result


def test_driver_connection_applies_duckdb_execution_width(monkeypatch):
    statements = []

    class _Connection:
        def execute(self, statement):
            statements.append(statement)

    monkeypatch.setenv("VANE_DUCKDB_THREADS", "7")

    driver._apply_duckdb_thread_setting(_Connection())

    assert statements == ["SET threads=7"]


def test_driver_reconciles_reconstructed_actor_location_and_public_leases():
    from vane.runners.ray.query_resource_graph import (
        QueryAllocation,
        QueryResourceGraph,
        ResourceUnitSpec,
        ResourceVector,
    )
    from vane.runners.ray.query_resource_manager import TaskRequest
    from vane.runners.ray.query_resource_runtime import (
        register_query_resource_graph,
        release_query_resource_manager,
    )

    cls, runner = _make_local_query_driver_actor()
    runner._ensure_query_resource_admission_state()
    query_id = "query-reconstructed-actor-location"
    resource_unit_id = "resource:query-reconstructed-actor-location:actor"
    resources = ResourceVector(cpu=2, heap_bytes=8, object_store_bytes=20)
    unit = ResourceUnitSpec(
        query_id=query_id,
        resource_unit_id=resource_unit_id,
        physical_node_id="7",
        unit_kind="ray_actor_pool",
        backend="ray_actor",
        input_unit_ids=(),
        per_task=ResourceVector(object_store_bytes=1),
        target_output_block_bytes=1,
        generator_buffer_blocks=2,
        max_concurrency=None,
        resident_per_actor=ResourceVector(cpu=1, heap_bytes=4),
        actor_pool_size=1,
        actor_prefetch_depth=1,
    )
    graph = QueryResourceGraph(
        query_id=query_id,
        plan_digest="sha256:reconstructed-actor-location",
        units=(unit,),
        terminal_unit_ids=(resource_unit_id,),
    )
    allocation = QueryAllocation(
        resources=resources,
        generation=1,
    )
    manager = register_query_resource_graph(graph, allocation)
    try:
        manager.set_submitted_actor_slots(resource_unit_id, {0})
        manager.set_ready_actor_slots(resource_unit_id, {0: "node-a"})
        manager.update_unit_state(resource_unit_id, runnable=True)
        grant = manager.try_acquire_task(
            TaskRequest(
                query_id=query_id,
                resource_unit_id=resource_unit_id,
                task_id="task:actor:0",
                attempt_id="attempt:0",
                node_id=None,
                retained_input_bytes=1,
            )
        )
        assert grant.granted
        pool = SimpleNamespace(
            _vane_retired=False,
            _vane_location_nonce="pool-nonce",
            actors=[object()],
            actor_node_ids=["node-a"],
            _confirmed_ready={0},
        )
        runner._active_udf_actor_by_unit = {query_id: {resource_unit_id: pool}}
        generation_capability = runner._issue_query_task_admission_capability(query_id)
        runner._query_task_lease_requests["public-request"] = {
            "manager_lease_id": grant.lease.lease_id,
            "lease": {"node_id": "node-a"},
        }

        result = asyncio.run(
            cls.report_query_udf_actor_location(
                runner,
                {
                    "query_id": query_id,
                    "resource_unit_id": resource_unit_id,
                    "actor_index": 0,
                    "pool_nonce": "pool-nonce",
                    "node_id": "node-b",
                },
                generation_capability,
            )
        )

        assert result == {
            "accepted": True,
            "node_id": "node-b",
            "moved_task_lease_count": 1,
        }
        assert pool.actor_node_ids == ["node-b"]
        assert manager.snapshot()["task_leases"][grant.lease.lease_id]["node_id"] == "node-b"
        assert runner._query_task_lease_requests["public-request"]["lease"]["node_id"] == "node-b"
    finally:
        release_query_resource_manager(query_id, reason="test complete")


@pytest.mark.parametrize(
    ("lease_timeout", "heartbeat_interval", "message"),
    [
        ("0", "0.1", "LEASE_TIMEOUT"),
        ("nan", "0.1", "LEASE_TIMEOUT"),
        ("3", "0", "HEARTBEAT_INTERVAL"),
        ("3", "1.1", "one third"),
    ],
)
def test_client_owner_lease_settings_reject_unsafe_values(
    monkeypatch,
    lease_timeout,
    heartbeat_interval,
    message,
):
    monkeypatch.setenv("VANE_RAY_CLIENT_LEASE_TIMEOUT_S", lease_timeout)
    monkeypatch.setenv(
        "VANE_RAY_CLIENT_HEARTBEAT_INTERVAL_S",
        heartbeat_interval,
    )

    with pytest.raises(ValueError, match=message):
        driver._client_owner_lease_settings()


def test_client_owner_lease_settings_accept_one_third_heartbeat_interval(
    monkeypatch,
):
    monkeypatch.setenv("VANE_RAY_CLIENT_LEASE_TIMEOUT_S", "30")
    monkeypatch.setenv("VANE_RAY_CLIENT_HEARTBEAT_INTERVAL_S", "10")

    assert driver._client_owner_lease_settings() == (30.0, 10.0)


@pytest.mark.parametrize("maintenance_kind", ["client_lease", "query_resource"])
def test_maintenance_loops_continue_after_pre311_asyncio_timeout(
    monkeypatch,
    maintenance_kind,
):
    class _Pre311AsyncioTimeout(Exception):
        pass

    runner_cls = driver.RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    maintenance_calls = []

    async def scenario():
        stop = asyncio.Event()
        if maintenance_kind == "client_lease":
            runner._client_lease_maintenance_stop = stop
            runner._client_heartbeat_interval_s = 0.25
            runner._client_lease_maintenance_error = ""
            runner._client_lease_maintenance_failures = 0
            runner._schedule_expired_client_reclamations = lambda: maintenance_calls.append(None)
            maintenance_loop = runner_cls._client_lease_maintenance_loop
        else:
            runner._query_resource_maintenance_stop = stop
            runner._query_resource_maintenance_interval_s = 0.25
            runner._query_resource_maintenance_error = ""
            runner._query_resource_maintenance_failures = 0
            runner._maintain_query_resources_once = lambda: maintenance_calls.append(None)
            maintenance_loop = runner_cls._query_resource_maintenance_loop

        wait_calls = 0

        async def wait_for(awaitable, *, timeout):
            nonlocal wait_calls
            assert timeout == 0.25
            wait_calls += 1
            if wait_calls == 1:
                awaitable.close()
                raise _Pre311AsyncioTimeout
            stop.set()
            return await awaitable

        monkeypatch.setattr(driver.asyncio, "TimeoutError", _Pre311AsyncioTimeout)
        monkeypatch.setattr(driver.asyncio, "wait_for", wait_for)

        await maintenance_loop(runner)
        assert wait_calls == 2

    asyncio.run(scenario())

    assert maintenance_calls == [None, None]


@pytest.mark.parametrize("raw", ["0", "-1", "nan", "inf", "invalid"])
def test_copy_reconciliation_timeout_rejects_unsafe_values(monkeypatch, raw):
    monkeypatch.setenv("VANE_RAY_COPY_RECONCILIATION_TIMEOUT_S", raw)

    with pytest.raises(ValueError):
        driver._copy_reconciliation_timeout_s()


def test_copy_reconciliation_timeout_reads_environment(monkeypatch):
    monkeypatch.setenv("VANE_RAY_COPY_RECONCILIATION_TIMEOUT_S", "0.25")

    assert driver._copy_reconciliation_timeout_s() == 0.25


def test_invalid_copy_reconciliation_timeout_prevents_write_submission(monkeypatch):
    calls = []

    class _RemoteMethod:
        @staticmethod
        def remote(*args):
            calls.append(args)
            return object()

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = _TEST_RUNTIME_OWNER_ID
    _initialize_test_query_driver_client(client, {_TEST_SESSION_ID: {}})
    client.runner = SimpleNamespace(run_copy_plan=_RemoteMethod())
    monkeypatch.setenv("VANE_RAY_COPY_RECONCILIATION_TIMEOUT_S", "0")

    with pytest.raises(ValueError, match="VANE_RAY_COPY_RECONCILIATION_TIMEOUT_S"):
        client.run_copy_plan(_FakePhysicalPlanWithoutPlanAttr("invalid-reconciliation-timeout"))

    assert calls == []


def test_driver_constructor_scrubs_inherited_session_environment(monkeypatch):
    cls = driver.RayQueryDriverActor.__ray_metadata__.modified_class
    monkeypatch.setenv("AWS_ISSUE75_INHERITED_SECRET", "inherited-aws")
    monkeypatch.setenv("DUCKDB_ISSUE75_INHERITED_SECRET", "inherited-duckdb")
    monkeypatch.setenv("VANE_ISSUE75_INHERITED_SECRET", "inherited-vane")
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(current_actor=object()),
    )
    monkeypatch.setattr(driver.asyncio, "get_running_loop", object)
    monkeypatch.setattr(driver, "set_event_loop", lambda _loop: None)
    monkeypatch.setattr(driver, "_set_global_event_loop", lambda _loop: None)
    monkeypatch.setattr(cls, "_create_query_resource_coordinator", lambda _self: object())
    monkeypatch.setattr(cls, "_ensure_duckdb_conn", lambda _self: object())
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: object())
    monkeypatch.setattr(cls, "_start_client_lease_maintenance", lambda _self: None)
    monkeypatch.setattr(cls, "_start_query_resource_maintenance", lambda _self: None)

    cls({}, 1)

    assert "AWS_ISSUE75_INHERITED_SECRET" not in os.environ
    assert "DUCKDB_ISSUE75_INHERITED_SECRET" not in os.environ
    assert "VANE_ISSUE75_INHERITED_SECRET" not in os.environ
    assert os.environ["VANE_RUNNER"] == "ray"


def _bind_test_query_resource_owner(
    runner,
    plan_id: str,
    *,
    query_id: str | None = None,
):
    from vane.runners.ray.query_resource_graph import (
        QueryAllocation,
        QueryResourceGraph,
        ResourceUnitSpec,
        ResourceVector,
    )
    from vane.runners.ray.query_resource_runtime import register_query_resource_graph

    query_id = str(plan_id if query_id is None else query_id)
    unit = ResourceUnitSpec(
        query_id=query_id,
        resource_unit_id=f"resource:{query_id}:result",
        physical_node_id="result",
        unit_kind="native_fragment",
        backend="ray_worker",
        input_unit_ids=(),
        per_task=ResourceVector(),
        target_output_block_bytes=1,
        generator_buffer_blocks=1,
        max_concurrency=1,
    )
    graph = QueryResourceGraph(
        query_id=query_id,
        plan_digest=f"sha256:{query_id}",
        units=(unit,),
        terminal_unit_ids=(unit.resource_unit_id,),
    )
    resources = ResourceVector(cpu=1, heap_bytes=1, object_store_bytes=1)
    manager = register_query_resource_graph(
        graph,
        QueryAllocation(
            resources=resources,
            generation=1,
        ),
    )
    manager.update_unit_state(unit.resource_unit_id, runnable=True)
    _bind_test_plan_session(runner, plan_id, query_id=query_id)
    return manager


def _bind_test_plan_session(
    runner,
    plan_id: str,
    *,
    query_id: str | None = None,
    query_connection=None,
) -> driver._PlanLifecycle:
    plan_key = str(plan_id)
    query_key = str(plan_key if query_id is None else query_id)
    lifecycle = driver._PlanLifecycle(
        plan_id=plan_key,
        session_id=_TEST_SESSION_ID,
    )
    if query_key:
        lifecycle.set_query_id(query_key)
    if query_connection is not None:
        lifecycle.set_query_connection(query_connection)
    assert lifecycle.begin_startup()
    if query_key:
        assert lifecycle.begin_fragment_execution()
        assert lifecycle.finish_setup(succeeded=True) is False
    else:
        assert lifecycle.finish_setup(succeeded=False) is True
    runner._plan_lifecycles[plan_key] = lifecycle
    runner._plan_session_ids[plan_key] = _TEST_SESSION_ID
    runner._plan_query_ids[plan_key] = query_key
    if query_connection is not None:
        runner._plan_connections[plan_key] = query_connection
    runner._sessions[_TEST_SESSION_ID].plan_ids.add(plan_key)
    return lifecycle


def _test_plan_teardown_state(runner, plan_id: str):
    lifecycle = runner._plan_lifecycles[str(plan_id)]
    query_id, query_connection, drop_fragments = lifecycle.wait_for_setup()
    return lifecycle, query_id, query_connection, drop_fragments


def _release_test_plan_session_state(cls, runner, plan_id: str) -> None:
    lifecycle, _query_id, query_connection, _drop_fragments = _test_plan_teardown_state(runner, plan_id)
    lifecycle.request_close()
    cls._release_plan_session_state(runner, lifecycle, _query_id, query_connection)


def _run_test_plan_preparation(
    cls,
    runner,
    session,
    logical_plan,
    plan_id: str,
    *,
    copy_plan: bool,
):
    lifecycle = cls._register_plan_lifecycle(runner, _TEST_SESSION_ID, session, plan_id)
    try:
        if copy_plan:
            return cls._prepare_copy_plan_sync(
                runner,
                _TEST_SESSION_ID,
                session,
                logical_plan,
                plan_id,
                lifecycle,
            )
        return cls._prepare_plan_sync(
            runner,
            _TEST_SESSION_ID,
            session,
            logical_plan,
            plan_id,
            lifecycle,
        )
    except BaseException:
        lifecycle.request_close()
        lifecycle.finish_setup(succeeded=False)
        cls._teardown_plan_resources(runner, plan_id)
        raise


def _fake_task_context_info(task_id):
    task_id = driver.FteTaskAttemptId.coerce(task_id)
    fragment_execution_id = int(task_id.fragment_execution_id)
    partition_id = int(task_id.partition_id)
    return {
        "query_idx": fragment_execution_id,
        "last_node_id": fragment_execution_id,
        "task_id": partition_id,
        "node_ids": [fragment_execution_id],
    }


def _fake_task_attempt_id(task_id):
    return driver.FteTaskAttemptId.coerce(task_id)


def test_ray_progress_snapshot_timeout_returns_none(monkeypatch):
    class _FakeRemoteMethod:
        def remote(self, *_args):
            return "snapshot-ref"

    class _FakeRunner:
        progress_snapshot = _FakeRemoteMethod()

    timeouts = []

    def _fake_get(_ref, *, timeout=None):
        timeouts.append(timeout)
        raise driver.ray.exceptions.GetTimeoutError("snapshot timed out")

    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _fake_get)

    assert (
        driver._ray_progress_snapshot_or_none(
            _FakeRunner(),
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            "plan-id",
            123.0,
        )
        is None
    )
    assert timeouts == [0.1]


def test_owned_thread_side_effect_finishes_before_cancellation_is_exposed():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    executor = driver.ThreadPoolExecutor(max_workers=1)

    def _mutate():
        started.set()
        assert release.wait(timeout=1.0)
        finished.set()

    async def _cancel():
        task = asyncio.create_task(driver._run_in_executor_with_owned_side_effects(executor, _mutate))
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.001)
        assert started.is_set()

        assert task.cancel() is True
        await asyncio.sleep(0)
        assert task.done() is False
        assert task.cancel() is True
        await asyncio.sleep(0)
        assert task.done() is False

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(_cancel())
    finally:
        executor.shutdown(wait=True)
    assert finished.is_set()


def test_query_driver_run_plan_cancellation_waits_for_startup_and_tears_down(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr("cancelled-plan"))
    started = threading.Event()
    release = threading.Event()
    events = []

    def _start(
        _self,
        plan,
        plan_id,
        _graph,
        _query_connection,
        _session_config,
        _lifecycle,
    ):
        assert plan is logical_plan.physical_plan
        assert plan_id == "cancelled-plan"
        assert _lifecycle.begin_fragment_execution()
        started.set()
        assert release.wait(timeout=1.0)
        events.append("started")
        return True

    def _cleanup(_self, plan_id):
        _, query_id, _query_connection, drop_fragments = _test_plan_teardown_state(runner, plan_id)
        assert query_id == "cancelled-plan"
        assert drop_fragments is True
        events.append(("cleanup", plan_id))
        _release_test_plan_session_state(cls, runner, plan_id)

    monkeypatch.setattr(cls, "_register_query_resources", _query_registration_stub("cancelled-plan"))
    monkeypatch.setattr(cls, "_run_plan_sync", _start)
    monkeypatch.setattr(cls, "_teardown_plan_resources", _cleanup)

    async def _cancel():
        task = asyncio.create_task(_run_actor_stream_plan(runner, logical_plan))
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.001)
        assert started.is_set()

        assert task.cancel() is True
        await asyncio.sleep(0)
        assert task.done() is False
        assert task.cancel() is True
        await asyncio.sleep(0)
        assert task.done() is False

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_cancel())

    assert events == ["started", ("cleanup", "cancelled-plan")]
    assert "cancelled-plan" not in runner._plan_session_ids
    assert "cancelled-plan" not in runner._sessions[_TEST_SESSION_ID].plan_ids


def test_close_plan_waits_for_startup_before_closing_owned_resources(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "close-during-stream-startup"
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr(plan_id))
    startup_started = threading.Event()
    startup_release = threading.Event()

    class _Pool:
        def __init__(self):
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    pool = _Pool()

    def _precreate_vllm_actors(
        _self,
        _plan,
        *,
        query_connection,
        session_config,
    ):
        assert session_config == _TEST_SESSION_CONFIG
        startup_started.set()
        assert startup_release.wait(timeout=2.0)
        assert query_connection.closed is False
        runner._active_vllm_actors.append(pool)
        runner._active_vllm_actors_by_plan[plan_id] = [pool]
        return [pool]

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", _precreate_vllm_actors)
    monkeypatch.setattr(cls, "_register_query_resources", _query_registration_stub(plan_id))
    monkeypatch.setattr(cls, "_release_query_resources", lambda *_args, **_kwargs: None)

    async def _close_during_startup():
        run_task = asyncio.create_task(_run_actor_stream_plan(runner, logical_plan))
        assert await asyncio.to_thread(startup_started.wait, 1.0)
        query_connection = runner._test_session_connection.cursors[-1]
        close_task = asyncio.create_task(
            cls.close_plan(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
                plan_id,
            )
        )
        for _ in range(100):
            lifecycle = runner._plan_lifecycles[plan_id]
            if lifecycle.close_requested():
                break
            await asyncio.sleep(0.001)
        assert lifecycle.close_requested() is True
        assert close_task.done() is False
        assert query_connection.closed is False

        startup_release.set()
        with pytest.raises(RuntimeError, match="closed before fragment startup"):
            await run_task
        await asyncio.wait_for(close_task, timeout=1.0)
        return query_connection

    query_connection = asyncio.run(_close_during_startup())

    assert query_connection.closed is True
    assert pool.shutdown_calls == 1
    assert runner._active_vllm_actors == []
    assert runner._active_vllm_actors_by_plan == {}
    assert plan_id not in runner._plan_lifecycles
    assert plan_id not in runner._plan_session_ids
    assert plan_id not in runner._plan_connections
    assert plan_id not in runner._plan_query_ids


def test_session_close_does_not_block_actor_loop_during_stream_startup(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    physical_plan = _FakePhysicalPlanWithoutPlanAttr("slow-stream-startup")
    startup_started = threading.Event()
    startup_release = threading.Event()

    class _BlockingLogicalPlan(_FakeLogicalPlan):
        def to_physical_plan(self, _conn, _effective_session_config):
            startup_started.set()
            assert startup_release.wait(timeout=1.0)
            return physical_plan

    class _PlanRunner:
        @staticmethod
        def run_plan(_plan, _conn):
            return _DummyStream([])

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(
        cls,
        "_register_query_resources",
        _query_registration_stub("slow-stream-startup"),
    )

    def _cleanup(_self, plan_id):
        runner.curr_plans.pop(plan_id, None)
        runner.curr_streams.pop(plan_id, None)
        _release_test_plan_session_state(cls, runner, plan_id)

    monkeypatch.setattr(cls, "_cleanup_finished_plan", _cleanup)

    async def _close_during_startup():
        run_task = asyncio.create_task(_run_actor_stream_plan(runner, _BlockingLogicalPlan(physical_plan)))
        assert await asyncio.to_thread(startup_started.wait, 1.0)
        close_task = asyncio.create_task(
            cls.close_session(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
            )
        )
        for _ in range(100):
            if runner._sessions[_TEST_SESSION_ID].closing:
                break
            await asyncio.sleep(0)
        assert runner._sessions[_TEST_SESSION_ID].closing is True
        await asyncio.wait_for(asyncio.sleep(0), timeout=0.1)
        assert close_task.done() is False
        startup_release.set()
        await asyncio.wait_for(run_task, timeout=1.0)
        await asyncio.wait_for(close_task, timeout=1.0)

    asyncio.run(_close_during_startup())

    assert _TEST_SESSION_ID not in runner._sessions


def test_query_driver_run_copy_plan_passes_distributed_physical_plan_wrapper(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    physical_plan = _FakePhysicalPlanWithoutPlanAttr("copy-plan")
    logical_plan = _FakeLogicalPlan(physical_plan)
    captured = {"lifecycle": []}

    class _PlanRunner:
        def run_copy_plan(self, plan, conn, on_execution_started):
            captured["plan"] = plan
            captured["conn"] = conn
            captured["copy_thread"] = threading.current_thread().name
            on_execution_started()
            return _committed_copy_result(ok=True)

    def _precreate_udf_actors(
        _self,
        _plan,
        _graph,
        *,
        query_connection,
        session_config,
    ):
        assert query_connection is runner._test_session_connection.cursors[-1]
        assert session_config == _TEST_SESSION_CONFIG
        assert runner._plan_query_ids["copy-plan"] == "copy-plan"
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        captured["actor_init_thread"] = threading.current_thread().name
        return []

    monkeypatch.setattr(cls, "_precreate_udf_actors", _precreate_udf_actors)
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(
        cls,
        "_register_query_resources",
        _query_registration_stub("copy-plan"),
    )
    monkeypatch.setattr(
        cls,
        "_drop_query_fragments_sync",
        lambda _self, _query_id: captured["lifecycle"].append("teardown"),
    )

    def _final_progress_snapshot(_self, query_id, _started_at):
        captured["lifecycle"].append("snapshot")
        return {"query_id": query_id, "state": "FINISHED"}

    monkeypatch.setattr(cls, "_build_local_progress_snapshot", _final_progress_snapshot)

    outcome = asyncio.run(_run_actor_copy_plan(runner, logical_plan))

    assert isinstance(outcome, driver.CopyPlanOutcome)
    assert outcome.result["ok"] is True
    assert outcome.result["copy_operation_id"] == "copy-plan"
    assert outcome.result["copy_write_state"] == "committed"
    assert outcome.result["copy_cleanup_state"] == "complete"
    assert outcome.final_progress_snapshot == {"query_id": "copy-plan", "state": "FINISHED"}
    assert captured["plan"] is physical_plan
    assert captured["conn"] is runner._test_session_connection.cursors[-1]
    assert captured["actor_init_thread"].startswith("vane-driver-native")
    assert captured["copy_thread"].startswith("vane-driver-copy")
    assert captured["lifecycle"] == ["snapshot", "teardown"]


def test_query_driver_copy_progresses_while_default_executor_is_saturated(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    runner._driver_executors_shutdown = False
    runner._driver_copy_executor = driver.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="test-driver-copy",
    )
    runner._driver_native_executor = driver.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="test-driver-native",
    )
    plan_id = "copy-default-executor-saturated"
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr(plan_id))
    default_started = threading.Event()
    default_release = threading.Event()
    copy_started = threading.Event()
    native_producer_started = threading.Event()
    actor_loop = None

    class _PlanRunner:
        @staticmethod
        def run_copy_plan(_plan, _conn, on_execution_started):
            on_execution_started()
            copy_started.set()
            assert actor_loop is not None
            producer = asyncio.run_coroutine_threadsafe(
                driver._run_in_executor(
                    runner._get_driver_native_executor(),
                    native_producer_started.set,
                ),
                actor_loop,
            )
            producer.result(timeout=1.0)
            return _committed_copy_result()

    def _teardown(_self, actual_plan_id):
        assert actual_plan_id == plan_id
        _, _query_id, _query_connection, drop_fragments = _test_plan_teardown_state(runner, actual_plan_id)
        assert drop_fragments is True
        _release_test_plan_session_state(cls, runner, actual_plan_id)

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(cls, "_register_query_resources", _query_registration_stub(plan_id))
    monkeypatch.setattr(cls, "_build_local_progress_snapshot", lambda *_args: {"state": "FINISHED"})
    monkeypatch.setattr(cls, "_teardown_plan_resources", _teardown)

    def _occupy_default_executor():
        default_started.set()
        assert default_release.wait(timeout=2.0)

    async def _run_with_saturated_default_executor():
        nonlocal actor_loop
        actor_loop = asyncio.get_running_loop()
        default_executor = driver.ThreadPoolExecutor(max_workers=1, thread_name_prefix="saturated-default")
        asyncio.get_running_loop().set_default_executor(default_executor)
        blocker = asyncio.create_task(asyncio.to_thread(_occupy_default_executor))
        try:
            while not default_started.is_set():
                await asyncio.sleep(0)
            copy = asyncio.create_task(_run_actor_copy_plan(runner, logical_plan))
            for _ in range(1000):
                if copy_started.is_set() and native_producer_started.is_set():
                    break
                await asyncio.sleep(0.001)
            assert copy_started.is_set()
            assert native_producer_started.is_set()
            return await asyncio.wait_for(copy, timeout=1.0)
        finally:
            default_release.set()
            await blocker
            default_executor.shutdown(wait=True)

    try:
        outcome = asyncio.run(_run_with_saturated_default_executor())
    finally:
        cls._shutdown_driver_executors(runner)

    assert outcome.write_state == "committed"
    assert outcome.cleanup_state == "complete"


def test_close_plan_cancels_queued_copy_before_connection_teardown(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    runner._driver_executors_shutdown = False
    runner._driver_copy_executor = driver.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="test-queued-copy",
    )
    plan_id = "copy-queued-before-native-start"
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr(plan_id))
    blocker_started = threading.Event()
    blocker_release = threading.Event()
    native_calls = 0

    def _occupy_copy_executor():
        blocker_started.set()
        assert blocker_release.wait(timeout=10.0)

    blocker = runner._driver_copy_executor.submit(_occupy_copy_executor)
    assert blocker_started.wait(timeout=1.0)

    class _PlanRunner:
        @staticmethod
        def run_copy_plan(_plan, _conn, on_execution_started):
            nonlocal native_calls
            native_calls += 1
            on_execution_started()
            return _committed_copy_result()

    def _teardown_once(
        _self,
        lifecycle,
        query_id,
        query_connection,
        *,
        drop_fragments,
    ):
        assert query_id == plan_id
        assert drop_fragments is False
        cls._release_plan_session_state(runner, lifecycle, query_id, query_connection)

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(cls, "_register_query_resources", _query_registration_stub(plan_id))
    monkeypatch.setattr(cls, "_teardown_plan_resources_once", _teardown_once)

    async def _close_while_copy_is_queued():
        copy_task = asyncio.create_task(_run_actor_copy_plan(runner, logical_plan))
        for _ in range(1000):
            lifecycle = runner._plan_lifecycles.get(plan_id)
            if lifecycle is not None and lifecycle._startup_future is not None:
                break
            await asyncio.sleep(0.001)
        else:
            raise AssertionError("COPY startup was not queued")
        query_connection = runner._plan_connections[plan_id]
        await asyncio.wait_for(
            cls.close_plan(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
                plan_id,
            ),
            timeout=1.0,
        )
        with pytest.raises(RuntimeError, match="closed before fragment startup"):
            await asyncio.wait_for(copy_task, timeout=1.0)
        return query_connection

    try:
        query_connection = asyncio.run(_close_while_copy_is_queued())
        assert blocker_release.is_set() is False
        assert native_calls == 0
        assert query_connection.closed is True
        assert plan_id not in runner._plan_lifecycles
        assert plan_id not in runner._plan_session_ids
        assert plan_id not in runner._sessions[_TEST_SESSION_ID].plan_ids
    finally:
        blocker_release.set()
        blocker.result(timeout=1.0)
        cls._shutdown_driver_executors(runner)


def test_query_driver_committed_copy_preserves_late_actor_initialization_warning(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    physical_plan = _FakePhysicalPlanWithoutPlanAttr("copy-plan-terminal")
    logical_plan = _FakeLogicalPlan(physical_plan)
    teardown_calls = []

    class _PlanRunner:
        def run_copy_plan(self, _plan, _conn, on_execution_started):
            on_execution_started()
            runner._query_terminal_errors["copy-query-terminal"] = "Ray actor UDF pool initialization failed"
            return _committed_copy_result(ok=True)

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(
        cls,
        "_register_query_resources",
        _query_registration_stub("copy-query-terminal"),
    )

    def _teardown(_self, plan_id):
        _, query_id, _query_connection, drop_fragments = _test_plan_teardown_state(runner, plan_id)
        teardown_calls.append((plan_id, query_id, drop_fragments))
        runner._query_terminal_errors.pop(query_id, None)
        _release_test_plan_session_state(cls, runner, plan_id)

    monkeypatch.setattr(cls, "_teardown_plan_resources", _teardown)

    outcome = asyncio.run(_run_actor_copy_plan(runner, logical_plan))

    assert teardown_calls == [
        ("copy-plan-terminal", "copy-query-terminal", True),
    ]
    assert outcome.write_state == "committed"
    assert outcome.cleanup_state == "complete"
    assert any("Ray actor UDF pool initialization failed" in warning for warning in outcome.cleanup_warnings)


def test_query_driver_copy_progress_failure_returns_committed_warning(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr("copy-progress-contract-failure"))
    teardown_calls = []

    class _PlanRunner:
        def run_copy_plan(self, _plan, _conn, on_execution_started):
            on_execution_started()
            return _committed_copy_result()

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(
        cls,
        "_register_query_resources",
        _query_registration_stub("copy-progress-contract-failure"),
    )
    monkeypatch.setattr(
        cls,
        "_build_local_progress_snapshot",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("invalid progress topology")),
    )

    def _teardown(_self, plan_id):
        _, query_id, _query_connection, drop_fragments = _test_plan_teardown_state(runner, plan_id)
        teardown_calls.append((plan_id, query_id, drop_fragments))
        _release_test_plan_session_state(cls, runner, plan_id)

    monkeypatch.setattr(
        cls,
        "_teardown_plan_resources",
        _teardown,
    )

    outcome = asyncio.run(_run_actor_copy_plan(runner, logical_plan))

    assert teardown_calls == [
        (
            "copy-progress-contract-failure",
            "copy-progress-contract-failure",
            True,
        )
    ]
    assert outcome.operation_id == "copy-progress-contract-failure"
    assert outcome.write_state == "committed"
    assert outcome.cleanup_state == "complete"
    assert outcome.final_progress_snapshot is None
    assert len(outcome.cleanup_warnings) == 1
    assert "progress finalization failed: RuntimeError: invalid progress topology" in outcome.cleanup_warnings[0]


def test_query_driver_concurrent_copy_retries_share_one_write(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "copy-concurrent-singleflight"
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr(plan_id))
    plan_started = threading.Event()
    plan_release = threading.Event()
    plan_calls = 0
    teardown_calls = 0

    class _PlanRunner:
        @staticmethod
        def run_copy_plan(_plan, _conn, on_execution_started):
            nonlocal plan_calls
            plan_calls += 1
            on_execution_started()
            plan_started.set()
            assert plan_release.wait(timeout=2.0)
            return _committed_copy_result(rows_copied=17)

    def _teardown(_self, actual_plan_id):
        nonlocal teardown_calls
        teardown_calls += 1
        assert actual_plan_id == plan_id
        _, _query_id, _query_connection, drop_fragments = _test_plan_teardown_state(runner, actual_plan_id)
        assert drop_fragments is True
        _release_test_plan_session_state(cls, runner, actual_plan_id)

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(cls, "_register_query_resources", _query_registration_stub(plan_id))
    monkeypatch.setattr(cls, "_build_local_progress_snapshot", lambda *_args: {"state": "FINISHED"})
    monkeypatch.setattr(cls, "_teardown_plan_resources", _teardown)

    async def _run_concurrent_retries():
        first_task = asyncio.create_task(_run_actor_copy_plan(runner, logical_plan))
        assert await asyncio.to_thread(plan_started.wait, 1.0)
        second_task = asyncio.create_task(_run_actor_copy_plan(runner, logical_plan))
        await asyncio.sleep(0.01)
        assert plan_calls == 1
        plan_release.set()
        return await asyncio.gather(first_task, second_task)

    first, second = asyncio.run(_run_concurrent_retries())

    assert first == second
    assert first.result["rows_copied"] == 17
    assert first.write_state == "committed"
    assert plan_calls == 1
    assert teardown_calls == 1


def test_query_driver_copy_result_logging_failure_replays_committed_success(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "copy-committed-result-logging-failure"
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr(plan_id))
    plan_calls = 0
    teardown_calls = []

    class _PlanRunner:
        def run_copy_plan(self, _plan, _conn, on_execution_started):
            nonlocal plan_calls
            plan_calls += 1
            on_execution_started()
            return _committed_copy_result(rows_copied=5)

    def _teardown(_self, actual_plan_id):
        _, query_id, _query_connection, drop_fragments = _test_plan_teardown_state(runner, actual_plan_id)
        teardown_calls.append((actual_plan_id, query_id, drop_fragments))
        _release_test_plan_session_state(cls, runner, actual_plan_id)

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(cls, "_register_query_resources", _query_registration_stub(plan_id))
    monkeypatch.setattr(cls, "_build_local_progress_snapshot", lambda *_args: {"state": "FINISHED"})
    monkeypatch.setattr(cls, "_teardown_plan_resources", _teardown)
    monkeypatch.setattr(
        driver,
        "_log_copy_result_debug",
        lambda *_args: (_ for _ in ()).throw(OSError("stderr unavailable")),
    )

    first = asyncio.run(_run_actor_copy_plan(runner, logical_plan))
    replayed = asyncio.run(_run_actor_copy_plan(runner, logical_plan))

    assert first == replayed
    assert first.write_state == "committed"
    assert first.cleanup_state == "complete"
    assert any("stderr unavailable" in warning for warning in first.cleanup_warnings)
    assert plan_calls == 1
    assert teardown_calls == [(plan_id, plan_id, True)]


def test_query_driver_committed_copy_teardown_is_retryable_without_reexecution(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "copy-committed-cleanup-retry"
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr(plan_id))
    plan_calls = 0
    teardown_calls = 0

    class _PlanRunner:
        def run_copy_plan(self, _plan, _conn, on_execution_started):
            nonlocal plan_calls
            plan_calls += 1
            on_execution_started()
            return _committed_copy_result(rows_copied=7)

    def _teardown(_self, actual_plan_id):
        nonlocal teardown_calls
        teardown_calls += 1
        _, query_id, _query_connection, drop_fragments = _test_plan_teardown_state(runner, actual_plan_id)
        assert (actual_plan_id, query_id, drop_fragments) == (plan_id, plan_id, True)
        if teardown_calls == 1:
            raise RuntimeError("planned post-commit teardown failure")
        _release_test_plan_session_state(cls, runner, actual_plan_id)

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(cls, "_register_query_resources", _query_registration_stub(plan_id))
    monkeypatch.setattr(cls, "_build_local_progress_snapshot", lambda *_args: {"state": "FINISHED"})
    monkeypatch.setattr(cls, "_teardown_plan_resources", _teardown)

    first = asyncio.run(_run_actor_copy_plan(runner, logical_plan))
    recovered = asyncio.run(
        cls.recover_copy_plan(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            plan_id,
        )
    )
    cleaned = asyncio.run(
        cls.retry_copy_cleanup(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            plan_id,
        )
    )
    replayed = asyncio.run(_run_actor_copy_plan(runner, logical_plan))

    assert first.cleanup_state == "pending"
    assert "planned post-commit teardown failure" in first.cleanup_warnings[0]
    assert recovered.outcome == first
    assert recovered.error is None
    assert cleaned.cleanup_state == "complete"
    assert replayed == cleaned
    assert plan_calls == 1
    assert teardown_calls == 2


def test_query_driver_copy_failure_is_replayed_without_reexecution(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "copy-failed-before-commit"
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr(plan_id))
    plan_calls = 0

    class _PlanRunner:
        def run_copy_plan(self, _plan, _conn, on_execution_started):
            nonlocal plan_calls
            plan_calls += 1
            on_execution_started()
            raise ValueError("planned failure before commit")

    def _teardown(_self, actual_plan_id):
        assert actual_plan_id == plan_id
        _, _query_id, _query_connection, drop_fragments = _test_plan_teardown_state(runner, actual_plan_id)
        assert drop_fragments is True
        _release_test_plan_session_state(cls, runner, actual_plan_id)

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(cls, "_register_query_resources", _query_registration_stub(plan_id))
    monkeypatch.setattr(cls, "_teardown_plan_resources", _teardown)

    for operation in (
        lambda: _run_actor_copy_plan(runner, logical_plan),
        lambda: _run_actor_copy_plan(runner, logical_plan),
    ):
        with pytest.raises(ValueError, match="planned failure before commit"):
            asyncio.run(operation())

    recovered = asyncio.run(
        cls.recover_copy_plan(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            plan_id,
        )
    )

    assert recovered.outcome is None
    assert isinstance(recovered.error, ValueError)
    assert str(recovered.error) == "planned failure before commit"
    assert plan_calls == 1


def test_query_driver_unknown_native_copy_outcome_is_structured_and_never_reexecuted(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "copy-native-outcome-unknown"
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr(plan_id))
    plan_calls = 0
    unknown_result = {
        "rows_copied": 3,
        "copy_output_committed": False,
        "copy_output_outcome_unknown": True,
        "copy_output_outcome_error": "marker PUT failed and readback was inconclusive",
        "copy_output_base_path": "s3://bucket/out",
        "copy_output_run_id": "run-unknown",
        "copy_output_manifest_path": "s3://bucket/out.duckdb_commit/run-unknown/manifest.txt",
        "copy_output_committed_marker_path": "s3://bucket/out.duckdb_commit/run-unknown/committed",
    }

    class _PlanRunner:
        def run_copy_plan(self, _plan, _conn, on_execution_started):
            nonlocal plan_calls
            plan_calls += 1
            on_execution_started()
            return unknown_result

    def _teardown(_self, actual_plan_id):
        assert actual_plan_id == plan_id
        _, _query_id, _query_connection, drop_fragments = _test_plan_teardown_state(runner, actual_plan_id)
        assert drop_fragments is True
        _release_test_plan_session_state(cls, runner, actual_plan_id)

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(cls, "_register_query_resources", _query_registration_stub(plan_id))
    monkeypatch.setattr(cls, "_teardown_plan_resources", _teardown)

    for operation in (
        lambda: _run_actor_copy_plan(runner, logical_plan),
        lambda: _run_actor_copy_plan(runner, logical_plan),
    ):
        with pytest.raises(driver.CopyOutcomeUnknownError) as error:
            asyncio.run(operation())
        assert error.value.operation_id == plan_id
        assert error.value.base_path == "s3://bucket/out"
        assert error.value.run_id == "run-unknown"
        assert error.value.manifest_path.endswith("/manifest.txt")
        assert error.value.committed_marker_path.endswith("/committed")
        assert error.value.safe_to_retry is False
        assert error.value.cleanup_warnings == ()
        assert "readback was inconclusive" in str(error.value)

    recovered = asyncio.run(
        cls.recover_copy_plan(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            plan_id,
        )
    )

    assert recovered.outcome is None
    assert isinstance(recovered.error, driver.CopyOutcomeUnknownError)
    assert recovered.error.run_id == "run-unknown"
    assert plan_calls == 1


def test_query_driver_evicted_copy_outcome_refuses_reexecution(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    first_plan_id = "copy-evicted-terminal-first"
    second_plan_id = "copy-evicted-terminal-second"
    logical_plans = {
        plan_id: _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr(plan_id))
        for plan_id in (first_plan_id, second_plan_id)
    }
    plan_calls = {first_plan_id: 0, second_plan_id: 0}

    class _PlanRunner:
        @staticmethod
        def run_copy_plan(plan, _conn, on_execution_started):
            plan_id = str(plan.idx())
            plan_calls[plan_id] += 1
            on_execution_started()
            return _committed_copy_result(rows_copied=plan_calls[plan_id])

    async def _register(
        _self,
        plan,
        *,
        query_connection,
        expected_plan_id=None,
    ):
        del query_connection
        plan_id = str(plan.idx())
        assert expected_plan_id == plan_id
        return SimpleNamespace(query_id=plan_id, units=()), object()

    def _teardown(_self, plan_id):
        _, query_id, _query_connection, drop_fragments = _test_plan_teardown_state(runner, plan_id)
        assert query_id == plan_id
        assert drop_fragments is True
        _release_test_plan_session_state(cls, runner, plan_id)

    runner._copy_operation_terminal = driver.BoundedReplayMap(capacity=1)
    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(cls, "_register_query_resources", _register)
    monkeypatch.setattr(cls, "_build_local_progress_snapshot", lambda *_args: {"state": "FINISHED"})
    monkeypatch.setattr(cls, "_teardown_plan_resources", _teardown)

    first = asyncio.run(_run_actor_copy_plan(runner, logical_plans[first_plan_id]))
    second = asyncio.run(_run_actor_copy_plan(runner, logical_plans[second_plan_id]))

    assert first.operation_id == first_plan_id
    assert second.operation_id == second_plan_id
    assert runner._copy_operation_terminal.get(first_plan_id) is None
    with pytest.raises(driver.CopyOutcomeUnknownError, match=first_plan_id):
        asyncio.run(_run_actor_copy_plan(runner, logical_plans[first_plan_id]))
    with pytest.raises(driver.CopyOutcomeUnknownError, match=first_plan_id):
        asyncio.run(
            cls.recover_copy_plan(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
                first_plan_id,
            )
        )
    assert asyncio.run(_run_actor_copy_plan(runner, logical_plans[second_plan_id])) == second
    assert plan_calls == {
        first_plan_id: 1,
        second_plan_id: 1,
    }


def test_query_driver_unknown_copy_recovery_never_submits_a_write():
    cls, runner = _make_local_query_driver_actor()

    with pytest.raises(driver.CopyOutcomeUnknownError, match="unknown-copy-operation") as error:
        asyncio.run(
            cls.recover_copy_plan(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
                "unknown-copy-operation",
            )
        )

    assert error.value.operation_id == "unknown-copy-operation"


def test_query_driver_copy_recovery_timeout_preserves_inflight_write(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    operation_id = "copy-recovery-timeout"
    outcome = driver.CopyPlanOutcome(
        operation_id=operation_id,
        result=_committed_copy_result(rows_copied=17),
        final_progress_snapshot={"state": "FINISHED"},
    )
    monkeypatch.setenv("VANE_RAY_COPY_RECONCILIATION_TIMEOUT_S", "0.01")

    async def _exercise():
        cls._ensure_copy_operation_state(runner)
        release = asyncio.Event()

        async def _finish_copy():
            await release.wait()
            return outcome

        operation_task = asyncio.create_task(_finish_copy())
        runner._copy_operation_identities[operation_id] = (
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
        )
        runner._copy_operations_inflight[operation_id] = driver._CopyOperationInFlight(
            owner_id=_TEST_RUNTIME_OWNER_ID,
            session_id=_TEST_SESSION_ID,
            task=operation_task,
        )
        try:
            with pytest.raises(driver.CopyOutcomeUnknownError, match=operation_id) as error:
                await cls.recover_copy_plan(
                    runner,
                    _TEST_RUNTIME_OWNER_ID,
                    _TEST_SESSION_ID,
                    operation_id,
                )

            assert error.value.operation_id == operation_id
            assert operation_task.done() is False
            assert operation_task.cancelled() is False

            release.set()
            await operation_task
            recovered = await cls.recover_copy_plan(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
                operation_id,
            )
            assert recovered.outcome == outcome
            assert recovered.error is None
        finally:
            release.set()
            if not operation_task.done():
                await operation_task

    asyncio.run(_exercise())


def test_copy_outcome_unknown_error_preserves_operation_id_across_serialization():
    error = driver.CopyOutcomeUnknownError(
        "unknown-copy-operation",
        "s3://bucket/out",
        "run-unknown",
        "s3://bucket/out.duckdb_commit/run-unknown/manifest.txt",
        "s3://bucket/out.duckdb_commit/run-unknown/committed",
        "marker readback failed",
        ("teardown warning",),
    )

    restored = pickle.loads(pickle.dumps(error))

    assert type(restored) is driver.CopyOutcomeUnknownError
    assert restored.operation_id == error.operation_id
    assert restored.base_path == error.base_path
    assert restored.run_id == error.run_id
    assert restored.manifest_path == error.manifest_path
    assert restored.committed_marker_path == error.committed_marker_path
    assert restored.detail == error.detail
    assert restored.cleanup_warnings == error.cleanup_warnings
    assert restored.safe_to_retry is False
    assert str(restored) == str(error)


def test_copy_outcome_unknown_error_is_exported_from_ray_runner_package():
    from vane.runners.ray import CopyOutcomeUnknownError

    assert CopyOutcomeUnknownError is driver.CopyOutcomeUnknownError


def test_query_driver_copy_progress_cancellation_waits_before_teardown(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "copy-progress-cancellation"
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr(plan_id))
    snapshot_started = threading.Event()
    snapshot_release = threading.Event()
    snapshot_finished = threading.Event()
    teardown_calls = []

    class _PlanRunner:
        @staticmethod
        def run_copy_plan(_plan, _conn, on_execution_started):
            on_execution_started()
            return _committed_copy_result(ok=True)

    def _build_snapshot(_self, query_id, _started_at):
        assert query_id == plan_id
        snapshot_started.set()
        assert snapshot_release.wait(timeout=1.0)
        snapshot_finished.set()
        return {"query_id": query_id}

    def _teardown(_self, actual_plan_id):
        assert snapshot_finished.is_set()
        _, query_id, _query_connection, drop_fragments = _test_plan_teardown_state(runner, actual_plan_id)
        teardown_calls.append((actual_plan_id, query_id, drop_fragments))
        _release_test_plan_session_state(cls, runner, actual_plan_id)

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(
        cls,
        "_register_query_resources",
        _query_registration_stub(plan_id),
    )
    monkeypatch.setattr(cls, "_build_local_progress_snapshot", _build_snapshot)
    monkeypatch.setattr(cls, "_teardown_plan_resources", _teardown)

    async def _cancel_during_snapshot():
        copy_task = asyncio.create_task(_run_actor_copy_plan(runner, logical_plan))
        assert await asyncio.to_thread(snapshot_started.wait, 1.0)
        copy_task.cancel()
        await asyncio.sleep(0.01)
        assert copy_task.done() is False
        assert teardown_calls == []
        snapshot_release.set()
        with pytest.raises(asyncio.CancelledError):
            await copy_task

    asyncio.run(_cancel_during_snapshot())

    assert teardown_calls == [(plan_id, plan_id, True)]
    assert runner._sessions[_TEST_SESSION_ID].active_operations == 0
    assert plan_id not in runner._plan_session_ids


def test_query_driver_copy_operation_cancellation_waits_for_native_commit(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "copy-native-commit-during-cancellation"
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr(plan_id))
    native_started = threading.Event()
    native_release = threading.Event()
    native_finished = threading.Event()
    teardown_started = threading.Event()
    plan_calls = 0

    class _PlanRunner:
        @staticmethod
        def run_copy_plan(_plan, _conn, on_execution_started):
            nonlocal plan_calls
            plan_calls += 1
            on_execution_started()
            native_started.set()
            assert native_release.wait(timeout=2.0)
            native_finished.set()
            return _committed_copy_result(rows_copied=11)

    def _teardown(_self, actual_plan_id):
        _, query_id, _query_connection, drop_fragments = _test_plan_teardown_state(runner, actual_plan_id)
        assert (actual_plan_id, query_id, drop_fragments) == (plan_id, plan_id, True)
        teardown_started.set()
        _release_test_plan_session_state(cls, runner, actual_plan_id)

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(cls, "_register_query_resources", _query_registration_stub(plan_id))
    monkeypatch.setattr(cls, "_build_local_progress_snapshot", lambda *_args: {"state": "FINISHED"})
    monkeypatch.setattr(cls, "_teardown_plan_resources", _teardown)

    async def _cancel_while_native_write_is_running():
        copy_task = asyncio.create_task(_run_actor_copy_plan(runner, logical_plan))
        assert await asyncio.to_thread(native_started.wait, 1.0)
        operation_task = runner._copy_operations_inflight[plan_id].task
        operation_task.cancel()
        assert await asyncio.to_thread(teardown_started.wait, 1.0)
        assert native_finished.is_set() is False
        native_release.set()
        outcome = await asyncio.wait_for(copy_task, timeout=1.0)
        recovered = await cls.recover_copy_plan(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            plan_id,
        )
        return outcome, recovered

    outcome, recovered = asyncio.run(_cancel_while_native_write_is_running())

    assert outcome.write_state == "committed"
    assert outcome.result["rows_copied"] == 11
    assert recovered.outcome == outcome
    assert plan_calls == 1
    assert teardown_started.is_set()


def test_query_driver_copy_teardown_cancellation_reports_cleanup_complete(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "copy-teardown-cancellation"
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr(plan_id))
    teardown_started = threading.Event()
    teardown_release = threading.Event()
    teardown_finished = threading.Event()

    class _PlanRunner:
        @staticmethod
        def run_copy_plan(_plan, _conn, on_execution_started):
            on_execution_started()
            return _committed_copy_result(rows_copied=13)

    def _teardown(_self, actual_plan_id):
        _, query_id, _query_connection, drop_fragments = _test_plan_teardown_state(runner, actual_plan_id)
        assert (actual_plan_id, query_id, drop_fragments) == (plan_id, plan_id, True)
        teardown_started.set()
        assert teardown_release.wait(timeout=2.0)
        _release_test_plan_session_state(cls, runner, actual_plan_id)
        teardown_finished.set()

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(cls, "_register_query_resources", _query_registration_stub(plan_id))
    monkeypatch.setattr(cls, "_build_local_progress_snapshot", lambda *_args: {"state": "FINISHED"})
    monkeypatch.setattr(cls, "_teardown_plan_resources", _teardown)

    async def _cancel_during_teardown():
        copy_task = asyncio.create_task(_run_actor_copy_plan(runner, logical_plan))
        assert await asyncio.to_thread(teardown_started.wait, 1.0)
        operation_task = runner._copy_operations_inflight[plan_id].task
        operation_task.cancel()
        await asyncio.sleep(0.01)
        assert copy_task.done() is False
        teardown_release.set()
        outcome = await asyncio.wait_for(copy_task, timeout=1.0)
        replayed = await _run_actor_copy_plan(runner, logical_plan)
        return outcome, replayed

    outcome, replayed = asyncio.run(_cancel_during_teardown())

    assert teardown_finished.is_set()
    assert outcome == replayed
    assert outcome.write_state == "committed"
    assert outcome.cleanup_state == "complete"
    assert not any("CancelledError" in warning for warning in outcome.cleanup_warnings)


def test_query_driver_copy_starts_without_eager_actor_or_topology_barriers(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr("copy-lazy-actor-startup"))
    events: list[str] = []

    class _PlanRunner:
        @staticmethod
        def run_copy_plan(_plan, _conn, on_execution_started):
            on_execution_started()
            events.append("plan")
            return _committed_copy_result(rows_copied=1)

    monkeypatch.setattr(
        cls,
        "_precreate_udf_actors",
        lambda *_args, **_kwargs: events.append("actor-locators") or [],
    )
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(
        cls,
        "_register_query_resources",
        _query_registration_stub("copy-lazy-actor-startup"),
    )
    monkeypatch.setattr(cls, "_build_local_progress_snapshot", lambda *_args: {"state": "FINISHED"})
    monkeypatch.setattr(cls, "_teardown_plan_resources", lambda *_args, **_kwargs: None)

    outcome = asyncio.run(_run_actor_copy_plan(runner, logical_plan))

    assert outcome.result["rows_copied"] == 1
    assert outcome.write_state == "committed"
    assert events == ["actor-locators", "plan"]
    assert not hasattr(cls, "_wait_for_udf_actors_ready")
    assert not hasattr(cls, "_mark_query_actor_units_ready")


def test_ray_query_driver_client_copy_refreshes_progress_and_uses_final_snapshot(monkeypatch):
    class _Future:
        def __init__(self, value, *, timeouts=0):
            self.value = value
            self.timeouts = timeouts

        def result(self, timeout=None):
            if self.timeouts:
                self.timeouts -= 1
                raise FutureTimeoutError
            return self.value

        def done(self):
            return False

    class _Ref:
        def __init__(self, value, *, timeouts=0):
            self._future = _Future(value, timeouts=timeouts)

        def future(self):
            return self._future

    class _RemoteMethod:
        def __init__(self, factory):
            self.factory = factory
            self.calls = []

        def remote(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self.factory()

    running_snapshot = {
        "query_id": "copy-plan",
        "state": "RUNNING",
        "fragments": [{"id": "fragment-1"}],
    }
    final_snapshot = {
        "query_id": "copy-plan",
        "state": "FINISHED",
        "fragments": [{"id": "fragment-1"}],
    }
    outcome = driver.CopyPlanOutcome(
        result={"rows_copied": 7},
        final_progress_snapshot=final_snapshot,
        operation_id="copy-plan",
    )

    class _Runner:
        open_session = _RemoteMethod(lambda: _Ref(True))
        run_copy_plan = _RemoteMethod(lambda: _Ref(outcome, timeouts=2))
        progress_snapshot = _RemoteMethod(lambda: _Ref(running_snapshot))

    class _Renderer:
        instances = []

        def __init__(self, snapshot_getter):
            self.snapshot_getter = snapshot_getter
            self.interval_s = 0.01
            self.snapshots = []
            self.finish_calls = []
            self.__class__.instances.append(self)

        def update(self):
            self.snapshots.append(self.snapshot_getter())

        def finish(self, **kwargs):
            self.finish_calls.append(kwargs)

    monkeypatch.setattr(driver, "ProgressRenderer", _Renderer)
    monkeypatch.setattr(driver, "progress_enabled", lambda: True)
    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = _TEST_RUNTIME_OWNER_ID
    _initialize_test_query_driver_client(client)
    client.runner = _Runner()

    result = client.run_copy_plan(_FakePhysicalPlanWithoutPlanAttr("copy-plan"))

    renderer = _Renderer.instances[0]
    assert result == {"rows_copied": 7}
    assert renderer.snapshots == [running_snapshot, running_snapshot]
    assert len(client.runner.progress_snapshot.calls) == 2
    assert renderer.finish_calls == [
        {
            "final_state": "FINISHED",
            "final_snapshot": final_snapshot,
        }
    ]


def test_ray_query_driver_client_recovers_ambiguous_copy_without_resubmission(monkeypatch):
    submit_ref = object()
    recovery_ref = object()

    class _RemoteMethod:
        def __init__(self, result):
            self.result = result
            self.calls = []

        def remote(self, *args):
            self.calls.append(args)
            return self.result

    outcome = driver.CopyPlanOutcome(
        operation_id="copy-ambiguous-response",
        write_state="committed",
        cleanup_state="complete",
        result=_committed_copy_result(rows_copied=11),
        final_progress_snapshot={"state": "FINISHED"},
    )
    runner = SimpleNamespace(
        run_copy_plan=_RemoteMethod(submit_ref),
        recover_copy_plan=_RemoteMethod(recovery_ref),
    )
    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = _TEST_RUNTIME_OWNER_ID
    _initialize_test_query_driver_client(client, {_TEST_SESSION_ID: {}})
    client.runner = runner
    resolutions = []

    def _resolve(ref, **kwargs):
        resolutions.append((ref, kwargs))
        if ref is submit_ref:
            raise FutureTimeoutError("COPY response wait timed out")
        assert ref is recovery_ref
        return driver.CopyPlanRecovery(
            operation_id=outcome.operation_id,
            outcome=outcome,
        )

    monkeypatch.setenv("VANE_RAY_COPY_RECONCILIATION_TIMEOUT_S", "30")
    monkeypatch.setattr(driver, "progress_enabled", lambda: False)
    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)
    monkeypatch.setattr(
        driver.ray,
        "cancel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ambiguous COPY recovery must not cancel the committed operation")
        ),
    )

    plan = _FakePhysicalPlanWithoutPlanAttr("copy-ambiguous-response")
    result = client.run_copy_plan(plan)

    assert result["rows_copied"] == 11
    assert runner.run_copy_plan.calls == [
        (
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            plan,
        )
    ]
    assert runner.recover_copy_plan.calls == [
        (
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            outcome.operation_id,
        )
    ]
    assert resolutions == [
        (submit_ref, {}),
        (
            recovery_ref,
            {
                "timeout": 30.0,
                "honor_query_deadline": False,
                "honor_object_get_timeout": False,
            },
        ),
    ]


def test_ray_query_driver_client_bounds_pending_copy_recovery_without_resubmission(monkeypatch):
    copy_future = object()
    recovery_future = object()
    plan = _FakePhysicalPlanWithoutPlanAttr("copy-pending-recovery")
    resolutions = []
    recovery_waits = 0

    class _RemoteMethod:
        def __init__(self, result):
            self.result = result
            self.calls = []

        def remote(self, *args):
            self.calls.append(args)
            return self.result

    run_copy_plan = _RemoteMethod(copy_future)
    recover_copy_plan = _RemoteMethod(recovery_future)
    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = _TEST_RUNTIME_OWNER_ID
    _initialize_test_query_driver_client(client, {_TEST_SESSION_ID: {}})
    client.runner = SimpleNamespace(
        run_copy_plan=run_copy_plan,
        recover_copy_plan=recover_copy_plan,
    )

    def _resolve(ref, **kwargs):
        nonlocal recovery_waits
        resolutions.append((ref, kwargs))
        if ref is copy_future:
            raise FutureTimeoutError("COPY response wait timed out")
        assert ref is recovery_future
        recovery_waits += 1
        if recovery_waits == 1:
            raise FutureTimeoutError("COPY recovery wait timed out")
        raise PermissionError("recovery ObjectRef was waited more than once")

    monkeypatch.setenv("VANE_RAY_COPY_RECONCILIATION_TIMEOUT_S", "0.25")
    monkeypatch.setenv("VANE_RAY_OBJECT_GET_TIMEOUT_S", "0")
    monkeypatch.setattr(driver, "progress_enabled", lambda: False)
    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)
    monkeypatch.setattr(
        driver.ray,
        "cancel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ambiguous COPY recovery must not cancel the write")
        ),
    )

    with pytest.raises(driver.CopyOutcomeUnknownError) as error:
        client.run_copy_plan(plan)

    assert error.value.operation_id == "copy-pending-recovery"
    assert recovery_waits == 1
    assert run_copy_plan.calls == [
        (
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            plan,
        )
    ]
    assert recover_copy_plan.calls == [
        (
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            "copy-pending-recovery",
        )
    ]
    assert resolutions == [
        (copy_future, {}),
        (
            recovery_future,
            {
                "timeout": 0.25,
                "honor_query_deadline": False,
                "honor_object_get_timeout": False,
            },
        ),
    ]


def test_ray_query_driver_client_recovers_invalid_copy_response_without_resubmission(monkeypatch):
    submit_ref = object()
    recovery_ref = object()

    class _RemoteMethod:
        def __init__(self, result):
            self.result = result
            self.calls = []

        def remote(self, *args):
            self.calls.append(args)
            return self.result

    invalid_outcome = driver.CopyPlanOutcome(
        operation_id="wrong-operation",
        result=_committed_copy_result(rows_copied=11),
        final_progress_snapshot={"state": "FINISHED"},
    )
    recovered_outcome = driver.CopyPlanOutcome(
        operation_id="copy-invalid-response",
        result=_committed_copy_result(rows_copied=11),
        final_progress_snapshot={"state": "FINISHED"},
    )
    runner = SimpleNamespace(
        run_copy_plan=_RemoteMethod(submit_ref),
        recover_copy_plan=_RemoteMethod(recovery_ref),
    )
    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = _TEST_RUNTIME_OWNER_ID
    _initialize_test_query_driver_client(client, {_TEST_SESSION_ID: {}})
    client.runner = runner

    def _resolve(ref, **_kwargs):
        if ref is submit_ref:
            return invalid_outcome
        assert ref is recovery_ref
        return driver.CopyPlanRecovery(
            operation_id=recovered_outcome.operation_id,
            outcome=recovered_outcome,
        )

    monkeypatch.setattr(driver, "progress_enabled", lambda: False)
    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)

    plan = _FakePhysicalPlanWithoutPlanAttr("copy-invalid-response")
    result = client.run_copy_plan(plan)

    assert result["rows_copied"] == 11
    assert runner.run_copy_plan.calls == [
        (
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            plan,
        )
    ]
    assert runner.recover_copy_plan.calls == [
        (
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            "copy-invalid-response",
        )
    ]


def test_ray_query_driver_client_maps_invalid_recovery_outcome_to_unknown(monkeypatch):
    submit_ref = object()
    recovery_ref = object()

    class _RemoteMethod:
        def __init__(self, result):
            self.result = result

        def remote(self, *_args):
            return self.result

    mismatched_outcome = driver.CopyPlanOutcome(
        operation_id="wrong-operation",
        result=_committed_copy_result(),
        final_progress_snapshot=None,
    )
    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = _TEST_RUNTIME_OWNER_ID
    _initialize_test_query_driver_client(client, {_TEST_SESSION_ID: {}})
    client.runner = SimpleNamespace(
        run_copy_plan=_RemoteMethod(submit_ref),
        recover_copy_plan=_RemoteMethod(recovery_ref),
    )

    def _resolve(ref, **_kwargs):
        if ref is submit_ref:
            raise FutureTimeoutError("COPY response wait timed out")
        assert ref is recovery_ref
        return driver.CopyPlanRecovery(
            operation_id="copy-invalid-recovery",
            outcome=mismatched_outcome,
        )

    monkeypatch.setattr(driver, "progress_enabled", lambda: False)
    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)

    with pytest.raises(driver.CopyOutcomeUnknownError) as error:
        client.run_copy_plan(_FakePhysicalPlanWithoutPlanAttr("copy-invalid-recovery"))

    assert error.value.operation_id == "copy-invalid-recovery"


def test_ray_query_driver_client_stream_waits_through_progress_session(monkeypatch):
    class _Future:
        def __init__(self, value):
            self.value = value

        def result(self, timeout=None):
            return self.value

    class _Ref:
        def __init__(self, value):
            self._future = _Future(value)

        def future(self):
            return self._future

    class _RemoteMethod:
        def __init__(self, value):
            self.value = value

        def remote(self, *_args, **_kwargs):
            return _Ref(self.value)

    partition_ref = object()

    class _Runner:
        open_session = _RemoteMethod(True)
        run_plan = _RemoteMethod(None)
        get_next_partition = _RemoteMethod(partition_ref)

    class _ProgressSession:
        instances = []

        def __init__(self, runner, owner_id, session_id, plan_id, started_at):
            self.resolved = []
            self.finish_calls = []
            self.__class__.instances.append(self)

        def resolve(self, ref):
            self.resolved.append(ref)
            return None

        def finish(self, **kwargs):
            self.finish_calls.append(kwargs)

    monkeypatch.setattr(driver, "_RayProgressSession", _ProgressSession)
    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = _TEST_RUNTIME_OWNER_ID
    _initialize_test_query_driver_client(client)
    client.runner = _Runner()

    assert list(client.stream_plan(_FakePhysicalPlanWithoutPlanAttr("stream-plan"))) == []

    progress = _ProgressSession.instances[0]
    assert len(progress.resolved) == 1
    assert progress.resolved[0].future().result() is partition_ref
    assert progress.finish_calls == [{"final_state": "FINISHED"}]


def test_ray_query_driver_client_stream_keeps_captured_runner_during_concurrent_close(monkeypatch):
    run_ref = object()
    partition_ref = object()
    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = _TEST_RUNTIME_OWNER_ID
    _initialize_test_query_driver_client(client, {_TEST_SESSION_ID: {}})

    class _RunMethod:
        @staticmethod
        def remote(*_args):
            client.runner = None
            return run_ref

    class _NextMethod:
        calls = []

        @classmethod
        def remote(cls, *args):
            cls.calls.append(args)
            return partition_ref

    runner = SimpleNamespace(
        run_plan=_RunMethod(),
        get_next_partition=_NextMethod(),
    )
    client.runner = runner
    monkeypatch.setattr(driver, "resolve_object_refs_blocking", lambda ref, **_kwargs: None)
    monkeypatch.setattr(driver, "progress_enabled", lambda: False)

    assert list(client.stream_plan(_FakePhysicalPlanWithoutPlanAttr("stream-plan"))) == []
    assert _NextMethod.calls == [
        (
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            "stream-plan",
            runner,
        )
    ]


def test_ray_query_driver_client_stream_start_failure_cancels_and_retries_close(monkeypatch):
    run_future = object()
    close_future = object()
    resolved = []
    cancelled = []

    class _RemoteMethod:
        def __init__(self, result):
            self.result = result
            self.calls = []

        def remote(self, *args):
            self.calls.append(args)
            return self.result

    runner = SimpleNamespace(
        run_plan=_RemoteMethod(run_future),
        close_plan=_RemoteMethod(close_future),
    )
    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = _TEST_RUNTIME_OWNER_ID
    _initialize_test_query_driver_client(client, {_TEST_SESSION_ID: {}})
    client.runner = runner

    def _resolve(ref, **kwargs):
        resolved.append((ref, kwargs))
        if ref is run_future:
            raise RuntimeError("planned stream startup failure")
        assert ref is close_future
        return None

    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)
    monkeypatch.setattr(driver.ray, "cancel", lambda ref, *, force: cancelled.append((ref, force)))

    plan = _FakePhysicalPlanWithoutPlanAttr("failed-stream-start")
    with pytest.raises(RuntimeError, match="planned stream startup failure"):
        list(client.stream_plan(plan))

    assert cancelled == [(run_future, False)]
    assert runner.close_plan.calls == [
        (
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            "failed-stream-start",
        )
    ]
    assert resolved == [
        (run_future, {}),
        (
            run_future,
            {
                "timeout": 300,
                "honor_query_deadline": False,
            },
        ),
        (
            close_future,
            {
                "timeout": 300,
                "honor_query_deadline": False,
            },
        ),
    ]


def test_ray_query_driver_client_stream_start_reports_teardown_failure(monkeypatch):
    run_future = object()
    close_future = object()

    class _RemoteMethod:
        def __init__(self, result):
            self.result = result

        def remote(self, *_args):
            return self.result

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = _TEST_RUNTIME_OWNER_ID
    _initialize_test_query_driver_client(client, {_TEST_SESSION_ID: {}})
    client.runner = SimpleNamespace(
        run_plan=_RemoteMethod(run_future),
        close_plan=_RemoteMethod(close_future),
    )
    client._runtime_is_unavailable_or_replaced = lambda: False

    def _resolve(ref, **_kwargs):
        if ref is run_future:
            raise RuntimeError("planned stream startup failure")
        assert ref is close_future
        raise RuntimeError("planned stream teardown failure")

    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)
    monkeypatch.setattr(driver.ray, "cancel", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="failed and teardown also failed") as error:
        list(client.stream_plan(_FakePhysicalPlanWithoutPlanAttr("failed-stream-cleanup")))

    assert "planned stream startup failure" in str(error.value)
    assert "planned stream teardown failure" in str(error.value)


def test_ray_query_driver_client_stream_preserves_primary_and_cleanup_failures(monkeypatch):
    run_future = object()
    partition_future = object()
    close_future = object()
    primary_error = ValueError("planned mid-stream query failure")
    cleanup_error = RuntimeError("planned mid-stream cleanup failure")

    class _RemoteMethod:
        def __init__(self, result):
            self.result = result

        def remote(self, *_args):
            return self.result

    class _ProgressSession:
        def __init__(self, *_args):
            self.finished = False

        def resolve(self, ref):
            assert ref is partition_future
            raise primary_error

        def finish(self, **_kwargs):
            self.finished = True

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = _TEST_RUNTIME_OWNER_ID
    _initialize_test_query_driver_client(client, {_TEST_SESSION_ID: {}})
    client.runner = SimpleNamespace(
        run_plan=_RemoteMethod(run_future),
        get_next_partition=_RemoteMethod(partition_future),
        close_plan=_RemoteMethod(close_future),
    )
    client._runtime_is_unavailable_or_replaced = lambda: False

    def _resolve(ref, **_kwargs):
        if ref is run_future:
            return None
        assert ref is close_future
        raise cleanup_error

    monkeypatch.setattr(driver, "_RayProgressSession", _ProgressSession)
    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)

    with pytest.raises(driver.QueryExecutionCleanupError) as error:
        list(client.stream_plan(_FakePhysicalPlanWithoutPlanAttr("mid-stream-cleanup")))

    assert error.value.primary_error is primary_error
    assert error.value.cleanup_errors == (cleanup_error,)
    assert error.value.__cause__ is primary_error
    assert "planned mid-stream query failure" in str(error.value)
    assert "planned mid-stream cleanup failure" in str(error.value)


def test_ray_query_driver_client_stream_failure_accepts_concurrent_detach_cleanup(monkeypatch):
    run_future = object()
    close_future = object()

    class _RunMethod:
        @staticmethod
        def remote(*_args):
            client.runner = None
            return run_future

    class _CloseMethod:
        @staticmethod
        def remote(*_args):
            return close_future

    runner = SimpleNamespace(
        run_plan=_RunMethod(),
        close_plan=_CloseMethod(),
    )
    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = _TEST_RUNTIME_OWNER_ID
    _initialize_test_query_driver_client(client, {_TEST_SESSION_ID: {}})
    client.runner = runner

    def _resolve(ref, **_kwargs):
        if ref is run_future:
            raise RuntimeError("planned stream startup failure")
        assert ref is close_future
        raise PermissionError("runtime owner was detached")

    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)
    monkeypatch.setattr(driver.ray, "cancel", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="planned stream startup failure"):
        list(client.stream_plan(_FakePhysicalPlanWithoutPlanAttr("detached-stream-cleanup")))


def test_ray_query_driver_client_copy_failure_recovers_without_cancel_or_close(monkeypatch):
    copy_future = object()
    recovery_future = object()
    resolved = []
    cancelled = []

    class _RemoteMethod:
        def __init__(self, result):
            self.result = result
            self.calls = []

        def remote(self, *args):
            self.calls.append(args)
            return self.result

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = _TEST_RUNTIME_OWNER_ID
    _initialize_test_query_driver_client(client, {_TEST_SESSION_ID: {}})
    client.runner = SimpleNamespace(
        run_copy_plan=_RemoteMethod(copy_future),
        recover_copy_plan=_RemoteMethod(recovery_future),
    )

    def _resolve(ref, **kwargs):
        resolved.append((ref, kwargs))
        if ref is copy_future:
            raise RuntimeError("planned COPY failure")
        assert ref is recovery_future
        return driver.CopyPlanRecovery(
            operation_id="failed-copy",
            error=RuntimeError("planned COPY failure"),
        )

    monkeypatch.setenv("VANE_RAY_COPY_RECONCILIATION_TIMEOUT_S", "30")
    monkeypatch.setattr(driver, "progress_enabled", lambda: False)
    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)
    monkeypatch.setattr(driver.ray, "cancel", lambda ref, *, force: cancelled.append((ref, force)))
    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)

    with pytest.raises(RuntimeError, match="planned COPY failure"):
        client.run_copy_plan(_FakePhysicalPlanWithoutPlanAttr("failed-copy"))

    assert cancelled == []
    assert resolved == [
        (copy_future, {}),
        (
            recovery_future,
            {
                "timeout": 30.0,
                "honor_query_deadline": False,
                "honor_object_get_timeout": False,
            },
        ),
    ]
    assert client.runner.recover_copy_plan.calls == [
        (
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            "failed-copy",
        )
    ]


@pytest.mark.parametrize(
    "structured_operation_error",
    (False, True),
    ids=("plain-timeout", "structured-unknown"),
)
def test_ray_query_driver_client_maps_recovery_rpc_creation_failure_to_unknown(
    monkeypatch,
    structured_operation_error,
):
    copy_future = object()
    operation_id = "copy-recovery-creation-failed"
    operation_error = (
        driver.CopyOutcomeUnknownError(
            operation_id,
            base_path="s3://bucket/out",
            run_id="run-recovery-creation-failed",
            manifest_path="s3://bucket/out.duckdb_commit/run-recovery-creation-failed/manifest.txt",
            committed_marker_path="s3://bucket/out.duckdb_commit/run-recovery-creation-failed/committed",
            detail="native marker readback failed",
            cleanup_warnings=("native cleanup failed",),
        )
        if structured_operation_error
        else FutureTimeoutError("COPY response wait timed out")
    )

    class _RemoteMethod:
        def __init__(self, result=None, *, error=None):
            self.result = result
            self.error = error
            self.calls = []

        def remote(self, *args):
            self.calls.append(args)
            if self.error is not None:
                raise self.error
            return self.result

    run_copy_plan = _RemoteMethod(copy_future)
    recover_copy_plan = _RemoteMethod(error=RuntimeError("recovery RPC could not be created"))
    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = _TEST_RUNTIME_OWNER_ID
    _initialize_test_query_driver_client(client, {_TEST_SESSION_ID: {}})
    client.runner = SimpleNamespace(
        run_copy_plan=run_copy_plan,
        recover_copy_plan=recover_copy_plan,
    )
    monkeypatch.setattr(client, "_runtime_is_unavailable_or_replaced", lambda: False)

    monkeypatch.setattr(driver, "progress_enabled", lambda: False)

    def _resolve(ref, **_kwargs):
        assert ref is copy_future
        raise operation_error

    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)

    with pytest.raises(driver.CopyOutcomeUnknownError) as error:
        client.run_copy_plan(_FakePhysicalPlanWithoutPlanAttr(operation_id))

    assert error.value.operation_id == operation_id
    if structured_operation_error:
        assert error.value is operation_error
        assert error.value.base_path == "s3://bucket/out"
        assert error.value.run_id == "run-recovery-creation-failed"
        assert error.value.manifest_path.endswith("/manifest.txt")
        assert error.value.committed_marker_path.endswith("/committed")
        assert error.value.detail == "native marker readback failed"
        assert error.value.cleanup_warnings == ("native cleanup failed",)
    assert len(run_copy_plan.calls) == 1
    assert recover_copy_plan.calls == [
        (
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            operation_id,
        )
    ]


@pytest.mark.parametrize(
    "structured_operation_error",
    (False, True),
    ids=("plain-timeout", "structured-unknown"),
)
def test_ray_query_driver_client_maps_recovery_resolution_failure_to_unknown(
    monkeypatch,
    structured_operation_error,
):
    copy_future = object()
    recovery_future = object()
    operation_id = "copy-recovery-resolution-failed"
    operation_error = (
        driver.CopyOutcomeUnknownError(
            operation_id,
            base_path="s3://bucket/out",
            run_id="run-recovery-resolution-failed",
            manifest_path="s3://bucket/out.duckdb_commit/run-recovery-resolution-failed/manifest.txt",
            committed_marker_path="s3://bucket/out.duckdb_commit/run-recovery-resolution-failed/committed",
            detail="native marker readback failed",
            cleanup_warnings=("native cleanup failed",),
        )
        if structured_operation_error
        else FutureTimeoutError("COPY response wait timed out")
    )

    class _RemoteMethod:
        def __init__(self, result):
            self.result = result
            self.calls = []

        def remote(self, *args):
            self.calls.append(args)
            return self.result

    run_copy_plan = _RemoteMethod(copy_future)
    recover_copy_plan = _RemoteMethod(recovery_future)
    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = _TEST_RUNTIME_OWNER_ID
    _initialize_test_query_driver_client(client, {_TEST_SESSION_ID: {}})
    client.runner = SimpleNamespace(
        run_copy_plan=run_copy_plan,
        recover_copy_plan=recover_copy_plan,
    )
    monkeypatch.setattr(client, "_runtime_is_unavailable_or_replaced", lambda: False)

    def _resolve(ref, **_kwargs):
        if ref is copy_future:
            raise operation_error
        assert ref is recovery_future
        raise PermissionError("runtime owner expired during COPY recovery")

    monkeypatch.setattr(driver, "progress_enabled", lambda: False)
    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)

    with pytest.raises(driver.CopyOutcomeUnknownError) as error:
        client.run_copy_plan(_FakePhysicalPlanWithoutPlanAttr(operation_id))

    assert error.value.operation_id == operation_id
    if structured_operation_error:
        assert error.value is operation_error
        assert error.value.base_path == "s3://bucket/out"
        assert error.value.run_id == "run-recovery-resolution-failed"
        assert error.value.manifest_path.endswith("/manifest.txt")
        assert error.value.committed_marker_path.endswith("/committed")
        assert error.value.detail == "native marker readback failed"
        assert error.value.cleanup_warnings == ("native cleanup failed",)
    assert len(run_copy_plan.calls) == 1
    assert recover_copy_plan.calls == [
        (
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            operation_id,
        )
    ]


def test_ray_query_driver_client_retries_pending_copy_cleanup_by_operation_id(monkeypatch):
    cleanup_ref = object()
    operation_id = "copy-cleanup-client-retry"

    class _RemoteMethod:
        def __init__(self, result):
            self.result = result
            self.calls = []

        def remote(self, *args):
            self.calls.append(args)
            return self.result

    retry_copy_cleanup = _RemoteMethod(cleanup_ref)
    outcome = driver.CopyPlanOutcome(
        operation_id=operation_id,
        result=_committed_copy_result(
            copy_operation_id=operation_id,
            copy_write_state="committed",
            copy_cleanup_state="complete",
            copy_cleanup_warnings=[],
        ),
        final_progress_snapshot={"state": "FINISHED"},
    )
    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = _TEST_RUNTIME_OWNER_ID
    _initialize_test_query_driver_client(client, {_TEST_SESSION_ID: {}})
    client.runner = SimpleNamespace(retry_copy_cleanup=retry_copy_cleanup)

    monkeypatch.setattr(
        driver,
        "resolve_object_refs_blocking",
        lambda ref, **kwargs: (
            outcome
            if ref is cleanup_ref and kwargs == {"honor_query_deadline": False}
            else (_ for _ in ()).throw(AssertionError("unexpected cleanup resolution"))
        ),
    )

    result = client.retry_copy_cleanup(operation_id)

    assert result["copy_cleanup_state"] == "complete"
    assert retry_copy_cleanup.calls == [
        (
            _TEST_RUNTIME_OWNER_ID,
            operation_id,
        )
    ]


def test_query_driver_run_plan_passes_distributed_physical_plan_wrapper(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    physical_plan = _FakePhysicalPlanWithoutPlanAttr("stream-plan")
    logical_plan = _FakeLogicalPlan(physical_plan)
    stream = _DummyStream([])
    captured = {}

    class _PlanRunner:
        def run_plan(self, plan, conn):
            captured["plan"] = plan
            captured["conn"] = conn
            return stream

    def _precreate_udf_actors(
        _self,
        _plan,
        _graph,
        *,
        query_connection,
        session_config,
    ):
        assert query_connection is runner._test_session_connection.cursors[-1]
        assert session_config == _TEST_SESSION_CONFIG
        assert runner._plan_query_ids["stream-plan"] == "stream-plan"
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        captured["startup_thread"] = threading.current_thread().name
        return []

    monkeypatch.setattr(cls, "_precreate_udf_actors", _precreate_udf_actors)
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(
        cls,
        "_register_query_resources",
        _query_registration_stub("stream-plan"),
    )
    monkeypatch.setattr(
        cls,
        "_release_query_resources",
        lambda _self, _query_id, reason, **_kwargs: None,
    )

    asyncio.run(_run_actor_stream_plan(runner, logical_plan))

    assert captured["plan"] is physical_plan
    assert captured["conn"] is runner._test_session_connection.cursors[-1]
    assert captured["startup_thread"].startswith("vane-driver-native")
    assert runner.curr_plans["stream-plan"] is physical_plan
    assert runner.curr_streams["stream-plan"] is stream


def test_query_driver_run_plan_start_failure_runs_complete_teardown(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    physical_plan = _FakePhysicalPlanWithoutPlanAttr("failed-plan")
    logical_plan = _FakeLogicalPlan(physical_plan)
    calls = []

    class _PlanRunner:
        def run_plan(self, _plan, _conn):
            raise ValueError("submission failed")

    def _cleanup_udf_actors(_self, plan_id):
        calls.append(("actors", plan_id))
        raise RuntimeError("actor cleanup failed")

    def _drop_fragments(_self, query_id, *, release_resources):
        calls.append(("fragments", query_id, release_resources))
        raise RuntimeError("fragment cleanup failed")

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(
        cls,
        "_register_query_resources",
        _query_registration_stub("failed-query"),
    )
    monkeypatch.setattr(cls, "_cleanup_udf_actor_pools", _cleanup_udf_actors)
    monkeypatch.setattr(
        cls,
        "_drop_query_fragments_after_admission_fence_sync",
        _drop_fragments,
    )

    with pytest.raises(RuntimeError, match="failed to start and teardown also failed") as exc_info:
        asyncio.run(_run_actor_stream_plan(runner, logical_plan))

    message = str(exc_info.value)
    assert "submission failed" in message
    assert "actor cleanup failed" in message
    assert "fragment cleanup failed" in message
    assert calls == [
        ("actors", "failed-plan"),
        ("fragments", "failed-query", False),
    ]
    assert "failed-plan" not in runner.curr_plans
    assert "failed-plan" not in runner.curr_streams
    assert runner._plan_query_ids["failed-plan"] == "failed-query"
    assert runner._plan_session_ids["failed-plan"] == _TEST_SESSION_ID

    monkeypatch.setattr(cls, "_cleanup_udf_actor_pools", lambda *_args: None)
    monkeypatch.setattr(cls, "_drop_query_fragments_sync", lambda *_args: None)
    cls._cleanup_finished_plan(runner, "failed-plan")

    assert "failed-plan" not in runner._plan_query_ids
    assert "failed-plan" not in runner._plan_session_ids


def test_teardown_plan_resources_attempts_every_owned_release(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "teardown-plan"
    calls = []
    attempts = {
        "output": 0,
        "actors": 0,
        "vllm": 0,
        "fragments": 0,
    }

    class _OutputOwner:
        def release(self):
            calls.append("output")
            attempts["output"] += 1
            if attempts["output"] == 1:
                raise RuntimeError("output release failed")

    runner.curr_plans[plan_id] = object()
    runner.curr_streams[plan_id] = _DummyStream([])
    _bind_test_plan_session(runner, plan_id, query_id="teardown-query")
    runner._leased_result_partition_refs = {plan_id: {"0": (object(), _OutputOwner())}}
    runner._result_partition_ref_counters = {plan_id: 1}

    def _cleanup_actors(_self, actual_plan_id):
        calls.append(f"actors:{actual_plan_id}")
        attempts["actors"] += 1
        if attempts["actors"] == 1:
            raise RuntimeError("actor release failed")

    def _cleanup_vllm_actors(_self, actual_plan_id):
        calls.append(f"vllm:{actual_plan_id}")
        attempts["vllm"] += 1
        if attempts["vllm"] == 1:
            raise RuntimeError("vllm actor release failed")

    def _drop_fragments(_self, query_id, *, release_resources=True):
        calls.append(f"fragments:{query_id}:release={release_resources}")
        attempts["fragments"] += 1
        if attempts["fragments"] == 1:
            raise RuntimeError("fragment release failed")

    monkeypatch.setattr(cls, "_cleanup_udf_actor_pools", _cleanup_actors)
    monkeypatch.setattr(cls, "_cleanup_vllm_actor_pools", _cleanup_vllm_actors)
    monkeypatch.setattr(cls, "_drop_query_fragments_sync", _drop_fragments)
    monkeypatch.setattr(
        cls,
        "_drop_query_fragments_after_admission_fence_sync",
        _drop_fragments,
    )

    with pytest.raises(RuntimeError, match="teardown failed") as exc_info:
        cls._teardown_plan_resources(
            runner,
            plan_id,
        )

    assert "output release failed" in str(exc_info.value)
    assert "actor release failed" in str(exc_info.value)
    assert "vllm actor release failed" in str(exc_info.value)
    assert "fragment release failed" in str(exc_info.value)
    assert calls == [
        "output",
        "actors:teardown-plan",
        "vllm:teardown-plan",
        "fragments:teardown-query:release=False",
    ]
    assert plan_id not in runner.curr_plans
    assert plan_id not in runner.curr_streams
    assert runner._plan_query_ids[plan_id] == "teardown-query"
    assert plan_id in runner._plan_session_ids
    assert list(runner._leased_result_partition_refs[plan_id]) == ["0"]
    assert runner._result_partition_ref_counters == {plan_id: 1}

    cls._teardown_plan_resources(
        runner,
        plan_id,
    )

    assert calls == [
        "output",
        "actors:teardown-plan",
        "vllm:teardown-plan",
        "fragments:teardown-query:release=False",
        "output",
        "actors:teardown-plan",
        "vllm:teardown-plan",
        "fragments:teardown-query:release=True",
    ]
    assert plan_id not in runner._plan_query_ids
    assert plan_id not in runner._plan_session_ids
    assert runner._leased_result_partition_refs == {}
    assert runner._result_partition_ref_counters == {}


def test_query_connection_close_failure_retains_plan_ownership_for_teardown_retry(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "query-connection-close-retry"
    query_id = "query-connection-close-retry-query"

    class _RetryingQueryConnection:
        def __init__(self):
            self.close_calls = 0
            self.closed = False

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("planned query connection close failure")
            self.closed = True

    query_connection = _RetryingQueryConnection()
    lifecycle = _bind_test_plan_session(
        runner,
        plan_id,
        query_id=query_id,
        query_connection=query_connection,
    )
    runner._query_terminal_errors[query_id] = "planned terminal marker"
    fragment_drops = []
    monkeypatch.setattr(
        cls,
        "_drop_query_fragments_sync",
        lambda _self, actual_query_id: fragment_drops.append(actual_query_id),
    )

    with pytest.raises(RuntimeError, match="planned query connection close failure"):
        cls._teardown_plan_resources(runner, plan_id)

    assert query_connection.close_calls == 1
    assert runner._plan_lifecycles[plan_id] is lifecycle
    assert runner._plan_session_ids[plan_id] == _TEST_SESSION_ID
    assert runner._plan_query_ids[plan_id] == query_id
    assert runner._plan_connections[plan_id] is query_connection
    assert plan_id in runner._sessions[_TEST_SESSION_ID].plan_ids
    assert runner._query_terminal_errors[query_id] == "planned terminal marker"

    cls._teardown_plan_resources(runner, plan_id)

    assert query_connection.close_calls == 2
    assert query_connection.closed is True
    assert fragment_drops == [query_id, query_id]
    assert plan_id not in runner._plan_lifecycles
    assert plan_id not in runner._plan_session_ids
    assert plan_id not in runner._plan_query_ids
    assert plan_id not in runner._plan_connections
    assert plan_id not in runner._sessions[_TEST_SESSION_ID].plan_ids
    assert query_id not in runner._query_terminal_errors


@pytest.mark.parametrize(
    ("cleanup_method", "active_attr", "by_plan_attr"),
    [
        ("_cleanup_udf_actor_pools", "_active_udf_actors", "_active_udf_actors_by_plan"),
        ("_cleanup_vllm_actor_pools", "_active_vllm_actors", "_active_vllm_actors_by_plan"),
    ],
)
def test_query_actor_pool_cleanup_retains_failed_pool_for_retry(cleanup_method, active_attr, by_plan_attr):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "actor-cleanup-retry"

    class _Pool:
        def __init__(self):
            self.attempts = 0

        def shutdown(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("transient actor cleanup failure")

    pool = _Pool()
    setattr(runner, active_attr, [pool])
    setattr(runner, by_plan_attr, {plan_id: [pool]})

    with pytest.raises(RuntimeError, match="transient actor cleanup failure"):
        getattr(cls, cleanup_method)(runner, plan_id)

    assert getattr(runner, active_attr) == [pool]
    assert getattr(runner, by_plan_attr) == {plan_id: [pool]}

    getattr(cls, cleanup_method)(runner, plan_id)

    assert getattr(runner, active_attr) == []
    assert getattr(runner, by_plan_attr) == {}


def test_failed_execution_owner_cleanup_blocks_query_resource_release(monkeypatch):
    from vane.runners.ray.query_resource_runtime import (
        get_query_resource_manager,
        release_query_resource_manager,
    )

    cls, runner = _make_local_query_driver_actor()
    plan_id = "execution-owner-release-gate"
    query_id = "execution-owner-release-gate-query"
    manager = _bind_test_query_resource_owner(runner, plan_id, query_id=query_id)
    runner.curr_plans[plan_id] = object()
    runner.curr_streams[plan_id] = _DummyStream([])
    fragment_calls = []

    def fail_actor_cleanup(_self, actual_plan_id):
        assert actual_plan_id == plan_id
        raise RuntimeError("planned live actor owner")

    def quiesce_without_release(_self, actual_query_id, *, release_resources):
        fragment_calls.append((actual_query_id, release_resources))

    monkeypatch.setattr(cls, "_cleanup_udf_actor_pools", fail_actor_cleanup)
    monkeypatch.setattr(
        cls,
        "_drop_query_fragments_after_admission_fence_sync",
        quiesce_without_release,
    )

    try:
        with pytest.raises(RuntimeError, match="planned live actor owner"):
            cls._teardown_plan_resources(
                runner,
                plan_id,
            )

        assert fragment_calls == [(query_id, False)]
        assert get_query_resource_manager(query_id) is manager
        assert runner._plan_query_ids[plan_id] == query_id
    finally:
        release_query_resource_manager(query_id, reason="test_complete")


def test_teardown_fence_failure_retains_retryable_query_ownership(monkeypatch):
    from vane.runners.ray.query_resource_runtime import (
        get_query_resource_manager,
        release_query_resource_manager,
    )

    cls, runner = _make_local_query_driver_actor()
    plan_id = "teardown-fence-owner"
    query_id = "teardown-fence-query"
    manager = _bind_test_query_resource_owner(runner, plan_id, query_id=query_id)
    runner.curr_plans[plan_id] = object()
    runner.curr_streams[plan_id] = _DummyStream([])

    def fail_before_owner_release(_self, actual_query_id):
        raise driver.QueryTeardownOwnershipError(f"planned admission fence failure for {actual_query_id}")

    monkeypatch.setattr(cls, "_drop_query_fragments_sync", fail_before_owner_release)

    try:
        with pytest.raises(RuntimeError, match="planned admission fence failure"):
            cls._teardown_plan_resources(
                runner,
                plan_id,
            )

        assert runner._plan_query_ids[plan_id] == query_id
        assert get_query_resource_manager(query_id) is manager
    finally:
        release_query_resource_manager(query_id, reason="test_complete")


def test_concurrent_plan_teardown_runs_owned_cleanup_once(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "concurrent-teardown"
    query_id = "concurrent-teardown-query"
    _bind_test_plan_session(runner, plan_id, query_id=query_id)
    runner.curr_plans[plan_id] = object()
    runner.curr_streams[plan_id] = _DummyStream([])
    cleanup_started = threading.Event()
    cleanup_release = threading.Event()
    cleanup_calls = []
    errors = []

    def _drop_fragments(_self, actual_query_id):
        cleanup_calls.append(actual_query_id)
        cleanup_started.set()
        cleanup_release.wait(timeout=1.0)

    monkeypatch.setattr(cls, "_drop_query_fragments_sync", _drop_fragments)

    def _cleanup():
        try:
            cls._cleanup_finished_plan(runner, plan_id)
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=_cleanup)
    second = threading.Thread(target=_cleanup)
    first.start()
    assert cleanup_started.wait(timeout=1.0)
    second.start()
    cleanup_release.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert cleanup_calls == [query_id]
    assert plan_id not in runner._plan_session_ids


def test_close_session_does_not_deadlock_with_plan_teardown(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "close-during-teardown"
    query_id = "close-during-teardown-query"
    _bind_test_plan_session(runner, plan_id, query_id=query_id)
    runner.curr_plans[plan_id] = object()
    runner.curr_streams[plan_id] = _DummyStream([])
    cleanup_started = threading.Event()
    cleanup_release = threading.Event()
    close_cleanup_started = threading.Event()
    cleanup_entries = 0
    cleanup_entries_lock = threading.Lock()
    cleanup_calls = []
    errors = []

    def _drop_fragments(_self, actual_query_id):
        cleanup_calls.append(actual_query_id)
        cleanup_started.set()
        cleanup_release.wait(timeout=1.0)

    original_cleanup_finished_plan = cls._cleanup_finished_plan

    def _tracked_cleanup_finished_plan(self, actual_plan_id):
        nonlocal cleanup_entries
        with cleanup_entries_lock:
            cleanup_entries += 1
            if cleanup_entries == 2:
                close_cleanup_started.set()
        return original_cleanup_finished_plan(self, actual_plan_id)

    async def _run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(cls, "_drop_query_fragments_sync", _drop_fragments)
    monkeypatch.setattr(cls, "_cleanup_finished_plan", _tracked_cleanup_finished_plan)
    monkeypatch.setattr(asyncio, "to_thread", _run_inline)

    def _cleanup():
        try:
            cls._cleanup_finished_plan(runner, plan_id)
        except BaseException as exc:
            errors.append(exc)

    def _close():
        try:
            asyncio.run(
                cls.close_session(
                    runner,
                    _TEST_RUNTIME_OWNER_ID,
                    _TEST_SESSION_ID,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    cleanup_thread = threading.Thread(target=_cleanup, daemon=True)
    close_thread = threading.Thread(target=_close, daemon=True)
    cleanup_thread.start()
    assert cleanup_started.wait(timeout=1.0)
    close_thread.start()
    assert close_cleanup_started.wait(timeout=1.0)
    cleanup_release.set()
    cleanup_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert not cleanup_thread.is_alive()
    assert not close_thread.is_alive()
    assert errors == []
    assert cleanup_calls == [query_id]
    assert runner._closed_session_owners.get(_TEST_SESSION_ID) == _TEST_RUNTIME_OWNER_ID
    assert _TEST_SESSION_ID not in runner._sessions


def test_session_close_waits_until_query_connection_is_closed():
    cls, runner = _make_local_query_driver_actor()
    plan_id = "close-during-query-connection-release"
    query_close_started = threading.Event()
    query_close_release = threading.Event()
    errors = []

    class _BlockingQueryConnection:
        def close(self):
            query_close_started.set()
            assert query_close_release.wait(timeout=1.0)

    query_connection = _BlockingQueryConnection()
    _bind_test_plan_session(
        runner,
        plan_id,
        query_id="",
        query_connection=query_connection,
    )

    def _teardown():
        try:
            cls._cleanup_finished_plan(runner, plan_id)
        except BaseException as exc:
            errors.append(exc)

    def _close_session():
        try:
            asyncio.run(
                cls.close_session(
                    runner,
                    _TEST_RUNTIME_OWNER_ID,
                    _TEST_SESSION_ID,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    teardown_thread = threading.Thread(target=_teardown, daemon=True)
    close_thread = threading.Thread(target=_close_session, daemon=True)
    teardown_thread.start()
    assert query_close_started.wait(timeout=1.0)
    close_thread.start()
    for _ in range(100):
        with runner._sessions[_TEST_SESSION_ID].condition:
            if runner._sessions[_TEST_SESSION_ID].closing:
                break
        time.sleep(0.001)

    assert close_thread.is_alive()
    assert runner._test_session_connection.closed is False

    query_close_release.set()
    teardown_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert not teardown_thread.is_alive()
    assert not close_thread.is_alive()
    assert errors == []
    assert runner._test_session_connection.closed is True


def test_close_session_waits_for_active_copy_operation():
    cls, runner = _make_local_query_driver_actor()
    session = runner._sessions[_TEST_SESSION_ID]
    cls._begin_session_operation(session, _TEST_SESSION_ID)

    async def _close_after_copy_finishes():
        close_task = asyncio.create_task(
            cls.close_session(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
            )
        )
        for _ in range(100):
            if session.closing:
                break
            await asyncio.sleep(0)
        assert session.closing is True
        assert close_task.done() is False
        with pytest.raises(RuntimeError, match="Vane session is closing"):
            cls._begin_session_operation(session, _TEST_SESSION_ID)
        cls._end_session_operation(session)
        await asyncio.wait_for(close_task, timeout=1.0)

    asyncio.run(_close_after_copy_finishes())

    assert session.closed is True
    assert runner._test_session_connection.closed is True
    assert _TEST_SESSION_ID not in runner._sessions


def test_detach_fences_new_session_work_and_replays_result():
    cls, runner = _make_local_query_driver_actor()
    runner._client_ids.add("other-owner")
    session = runner._sessions[_TEST_SESSION_ID]
    cls._begin_session_operation(session, _TEST_SESSION_ID)

    async def _detach_after_copy_finishes():
        detach_task = asyncio.create_task(
            cls.detach_client(
                runner,
                _TEST_RUNTIME_OWNER_ID,
            )
        )
        for _ in range(100):
            if _TEST_RUNTIME_OWNER_ID in runner._detaching_client_ids:
                break
            await asyncio.sleep(0)
        assert _TEST_RUNTIME_OWNER_ID in runner._detaching_client_ids
        with pytest.raises(PermissionError, match="attached client owner"):
            await cls.open_session(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                "late-session",
                {},
            )
        cls._end_session_operation(session)
        assert await asyncio.wait_for(detach_task, timeout=1.0) is False
        assert await cls.detach_client(runner, _TEST_RUNTIME_OWNER_ID) is False

    asyncio.run(_detach_after_copy_finishes())

    assert _TEST_RUNTIME_OWNER_ID not in runner._client_ids
    assert _TEST_SESSION_ID not in runner._sessions
    assert runner._detached_client_results[_TEST_RUNTIME_OWNER_ID] is False


def test_client_owner_lease_heartbeat_renews_and_cannot_resurrect_expired_generation():
    cls, runner = _make_local_query_driver_actor()
    lease = runner._client_leases[_TEST_RUNTIME_OWNER_ID]
    lease.lease_token = "owner-generation-a"
    lease.expires_at = time.monotonic() + 1.0
    original_expiry = lease.expires_at

    assert cls.heartbeat_client(
        runner,
        _TEST_RUNTIME_OWNER_ID,
        "owner-generation-a",
    )
    assert lease.expires_at > original_expiry

    lease.expires_at = 10.0
    assert cls._mark_expired_client_leases(runner, now=9.0) == ()
    assert cls._mark_expired_client_leases(runner, now=10.0) == (_TEST_RUNTIME_OWNER_ID,)
    assert lease.state == "expired"

    with pytest.raises(PermissionError, match="lease is no longer active"):
        cls.heartbeat_client(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            "owner-generation-a",
        )
    with pytest.raises(PermissionError, match="lease is no longer active"):
        cls.heartbeat_client(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            "owner-generation-b",
        )
    assert lease.state == "expired"


def test_expired_client_reclamation_retains_ownership_until_retry_succeeds():
    cls, runner = _make_local_query_driver_actor()
    other_owner = "surviving-owner"
    runner._client_ids.add(other_owner)
    cls._ensure_client_lease_state(runner)
    plan_id = "expired-owner-plan"
    _bind_test_plan_session(runner, plan_id, query_id="expired-owner-query")
    runner.curr_plans[plan_id] = object()
    runner.curr_streams[plan_id] = _DummyStream([])
    lease = runner._client_leases[_TEST_RUNTIME_OWNER_ID]
    lease.lease_token = "expired-owner-generation"
    lease.expires_at = 5.0
    cleanup_attempts = []

    def _cleanup(actual_plan_id):
        cleanup_attempts.append(actual_plan_id)
        if len(cleanup_attempts) == 1:
            raise RuntimeError("planned expired-owner teardown failure")
        runner.curr_plans.pop(actual_plan_id, None)
        runner.curr_streams.pop(actual_plan_id, None)
        runner._plan_query_ids.pop(actual_plan_id, None)
        runner._plan_session_ids.pop(actual_plan_id, None)
        runner._sessions[_TEST_SESSION_ID].plan_ids.discard(actual_plan_id)

    runner._cleanup_finished_plan = _cleanup

    async def _reclaim_with_retry():
        assert cls._mark_expired_client_leases(runner, now=5.0) == (_TEST_RUNTIME_OWNER_ID,)
        with pytest.raises(
            RuntimeError,
            match="planned expired-owner teardown failure",
        ):
            await cls._detach_client_owner(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                expired=True,
            )

        assert _TEST_RUNTIME_OWNER_ID in runner._client_ids
        assert _TEST_RUNTIME_OWNER_ID in runner._detaching_client_ids
        assert runner._plan_session_ids[plan_id] == _TEST_SESSION_ID
        assert runner._client_leases[_TEST_RUNTIME_OWNER_ID].state == "expired"
        assert "planned expired-owner teardown failure" in (
            runner._client_leases[_TEST_RUNTIME_OWNER_ID].last_cleanup_error
        )

        assert await cls.open_session(
            runner,
            other_owner,
            "survivor-during-expired-cleanup-retry",
            {},
        )
        await cls.close_session(
            runner,
            other_owner,
            "survivor-during-expired-cleanup-retry",
        )

        assert (
            await cls._detach_client_owner(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                expired=True,
            )
            is False
        )

    asyncio.run(_reclaim_with_retry())

    assert cleanup_attempts == [plan_id, plan_id]
    assert _TEST_RUNTIME_OWNER_ID not in runner._client_ids
    assert other_owner in runner._client_ids
    assert _TEST_SESSION_ID not in runner._sessions
    assert plan_id not in runner._plan_session_ids
    assert runner._detached_client_results[_TEST_RUNTIME_OWNER_ID] is False


def test_expired_client_reclamation_cancels_owned_active_operation():
    cls, runner = _make_local_query_driver_actor()
    runner._sessions[_TEST_SESSION_ID].plan_ids.clear()
    runner._client_ids.add("surviving-owner")
    cls._ensure_client_lease_state(runner)
    lease = runner._client_leases[_TEST_RUNTIME_OWNER_ID]
    lease.expires_at = 1.0

    async def _reclaim_active_operation():
        operation_started = asyncio.Event()

        async def _operation():
            session = runner._sessions[_TEST_SESSION_ID]
            cls._begin_session_operation(session, _TEST_SESSION_ID)
            operation_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cls._end_session_operation(session)

        operation = asyncio.create_task(_operation())
        await operation_started.wait()
        assert cls._mark_expired_client_leases(runner, now=1.0) == (_TEST_RUNTIME_OWNER_ID,)
        assert (
            await cls._detach_client_owner(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                expired=True,
            )
            is False
        )
        with pytest.raises(asyncio.CancelledError):
            await operation

    asyncio.run(_reclaim_active_operation())

    assert _TEST_RUNTIME_OWNER_ID not in runner._client_ids
    assert "surviving-owner" in runner._client_ids
    assert _TEST_SESSION_ID not in runner._sessions


def test_expired_client_reclamation_joins_plan_startup_before_session_close(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    runner._client_ids.add("surviving-owner")
    cls._ensure_client_lease_state(runner)
    plan_id = "expired-owner-blocked-startup"
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr(plan_id))
    startup_started = threading.Event()
    startup_release = threading.Event()

    class _Pool:
        def __init__(self):
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    pool = _Pool()

    def _precreate_vllm_actors(
        _self,
        _plan,
        *,
        query_connection,
        session_config,
    ):
        assert session_config == _TEST_SESSION_CONFIG
        startup_started.set()
        assert startup_release.wait(timeout=2.0)
        assert query_connection.closed is False
        runner._active_vllm_actors.append(pool)
        runner._active_vllm_actors_by_plan[plan_id] = [pool]
        return [pool]

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", _precreate_vllm_actors)
    monkeypatch.setattr(cls, "_register_query_resources", _query_registration_stub(plan_id))
    monkeypatch.setattr(cls, "_release_query_resources", lambda *_args, **_kwargs: None)

    class _CancelCountingTask(asyncio.Task):
        def __init__(self, coroutine):
            super().__init__(coroutine)
            self.cancel_calls = 0

        def cancel(self, *args, **kwargs):
            self.cancel_calls += 1
            return super().cancel(*args, **kwargs)

    async def _expire_during_startup():
        run_task = _CancelCountingTask(_run_actor_stream_plan(runner, logical_plan))
        assert await asyncio.to_thread(startup_started.wait, 1.0)
        query_connection = runner._test_session_connection.cursors[-1]
        runner._client_leases[_TEST_RUNTIME_OWNER_ID].expires_at = 1.0
        cls._schedule_expired_client_reclamations(runner)
        cleanup_task = runner._expired_client_cleanup_tasks[_TEST_RUNTIME_OWNER_ID]
        for _ in range(100):
            if run_task.cancel_calls:
                break
            await asyncio.sleep(0.001)
        assert run_task.cancel_calls == 1
        assert run_task.done() is False
        assert cleanup_task.done() is False
        assert query_connection.closed is False

        startup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        assert await asyncio.wait_for(cleanup_task, timeout=1.0) is False
        return query_connection, run_task

    query_connection, run_task = asyncio.run(_expire_during_startup())

    assert run_task.cancel_calls == 1
    assert query_connection.closed is True
    assert pool.shutdown_calls == 1
    assert runner._active_vllm_actors == []
    assert runner._active_vllm_actors_by_plan == {}
    assert plan_id not in runner._plan_lifecycles
    assert plan_id not in runner._plan_session_ids
    assert plan_id not in runner._plan_connections
    assert plan_id not in runner._plan_query_ids
    assert _TEST_SESSION_ID not in runner._sessions


def test_expired_client_reclamation_cancels_blocked_partition_read():
    from vane.runners.ray.query_resource_runtime import (
        release_query_resource_manager,
    )

    cls, runner = _make_local_query_driver_actor()
    runner._client_ids.add("surviving-owner")
    cls._ensure_client_lease_state(runner)
    plan_id = "expired-owner-blocked-read"
    query_id = "expired-owner-blocked-query"
    manager = _bind_test_query_resource_owner(
        runner,
        plan_id,
        query_id=query_id,
    )
    read_started = threading.Event()
    release_read = threading.Event()

    class _BlockingStream:
        def __init__(self):
            self.loop = None
            self.ready_callback = None

        def next_nowait(self):
            read_started.set()
            if release_read.is_set():
                raise StopIteration
            return None

        def set_ready_callback(self, loop, callback):
            self.loop = loop
            self.ready_callback = callback

        def arm_ready_notification(self):
            if release_read.is_set():
                self.loop.call_soon(self.ready_callback)

        def clear_ready_callback(self):
            self.loop = None
            self.ready_callback = None

        def release(self):
            release_read.set()
            if self.loop is not None and self.ready_callback is not None:
                self.loop.call_soon_threadsafe(self.ready_callback)

    runner.curr_plans[plan_id] = object()
    stream = _BlockingStream()
    runner.curr_streams[plan_id] = stream
    runner._drop_query_fragments_sync = lambda _query_id: stream.release()
    lease = runner._client_leases[_TEST_RUNTIME_OWNER_ID]
    lease.expires_at = time.monotonic() + 60.0

    async def _reclaim_blocked_read():
        partition_read = asyncio.create_task(
            cls.get_next_partition(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
                plan_id,
            )
        )
        assert await asyncio.to_thread(read_started.wait, 1.0)
        lease.expires_at = 1.0
        cls._mark_expired_client_leases(runner, now=1.0)
        cls._schedule_expired_client_reclamations(runner)
        cleanup_task = runner._expired_client_cleanup_tasks[_TEST_RUNTIME_OWNER_ID]
        with pytest.raises(asyncio.CancelledError):
            await partition_read
        assert await asyncio.wait_for(cleanup_task, timeout=1.0) is False

    try:
        asyncio.run(_reclaim_blocked_read())
        assert manager.snapshot()["external_consumer_waiting"] is False
        assert _TEST_RUNTIME_OWNER_ID not in runner._client_ids
        assert _TEST_SESSION_ID not in runner._sessions
        assert plan_id not in runner._plan_session_ids
    finally:
        release_read.set()
        release_query_resource_manager(query_id, reason="test_complete")


def test_open_session_revalidates_owner_inside_registry_lock(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    runner._sessions = {}
    runner._client_ids.add("other-owner")
    normalize_started = threading.Event()
    normalize_release = threading.Event()
    open_errors = []

    def _delayed_normalize(config):
        normalize_started.set()
        assert normalize_release.wait(timeout=1.0)
        return dict(config)

    monkeypatch.setattr(driver, "_normalize_session_config", _delayed_normalize)

    def _open():
        try:
            asyncio.run(
                cls.open_session(
                    runner,
                    _TEST_RUNTIME_OWNER_ID,
                    "late-session",
                    {},
                )
            )
        except BaseException as exc:
            open_errors.append(exc)

    open_thread = threading.Thread(target=_open)
    open_thread.start()
    assert normalize_started.wait(timeout=1.0)

    assert asyncio.run(cls.detach_client(runner, _TEST_RUNTIME_OWNER_ID)) is False
    normalize_release.set()
    open_thread.join(timeout=1.0)

    assert not open_thread.is_alive()
    assert len(open_errors) == 1
    assert isinstance(open_errors[0], PermissionError)
    assert "late-session" not in runner._sessions


def test_open_session_defers_credential_resolution_until_plan_preparation(monkeypatch):
    from vane.runners.ray import worker as worker_module

    cls, runner = _make_local_query_driver_actor()
    runner._sessions = {}
    monkeypatch.setattr(
        worker_module,
        "_effective_duckdb_s3_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("session open must not resolve credentials")),
    )

    assert (
        asyncio.run(
            cls.open_session(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                "deferred-credential-session",
                {"AWS_PROFILE": "analytics"},
            )
        )
        is True
    )
    assert runner._sessions["deferred-credential-session"].s3_config == {}


def test_last_owner_detach_retries_runtime_shutdown_failure():
    cls, runner = _make_local_query_driver_actor()
    runner._sessions = {}
    shutdown_calls = []
    maintenance_restarts = []
    runner._start_client_lease_maintenance = lambda: maintenance_restarts.append("restart")

    async def _shutdown_runtime():
        shutdown_calls.append("shutdown")
        if len(shutdown_calls) == 1:
            raise RuntimeError("planned shutdown failure")

    runner._shutdown_runtime = _shutdown_runtime

    async def _detach_with_retry():
        with pytest.raises(RuntimeError, match="planned shutdown failure"):
            await cls.detach_client(runner, _TEST_RUNTIME_OWNER_ID)
        assert _TEST_RUNTIME_OWNER_ID in runner._client_ids
        assert _TEST_RUNTIME_OWNER_ID not in runner._detaching_client_ids
        assert await cls.detach_client(runner, _TEST_RUNTIME_OWNER_ID) is True
        assert await cls.detach_client(runner, _TEST_RUNTIME_OWNER_ID) is True

    asyncio.run(_detach_with_retry())

    assert shutdown_calls == ["shutdown", "shutdown"]
    assert maintenance_restarts == ["restart"]
    assert _TEST_RUNTIME_OWNER_ID not in runner._client_ids
    assert runner._detached_client_results[_TEST_RUNTIME_OWNER_ID] is True


def test_driver_sessions_keep_config_and_close_lifecycle_independent():
    cls, runner = _make_local_query_driver_actor()
    session_b_config = {
        "AWS_ACCESS_KEY_ID": "session-b-key",
        "AWS_SECRET_ACCESS_KEY": "session-b-secret",
    }

    assert asyncio.run(
        cls.open_session(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            "session-b",
            session_b_config,
        )
    )
    session_b_config["AWS_ACCESS_KEY_ID"] = "mutated-after-open"

    assert runner._sessions[_TEST_SESSION_ID].config == {}
    assert runner._sessions["session-b"].config == {
        "AWS_ACCESS_KEY_ID": "session-b-key",
        "AWS_SECRET_ACCESS_KEY": "session-b-secret",
    }

    asyncio.run(
        cls.close_session(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
        )
    )
    asyncio.run(
        cls.close_session(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
        )
    )

    assert _TEST_SESSION_ID not in runner._sessions
    assert cls._require_session(runner, _TEST_RUNTIME_OWNER_ID, "session-b").config == {
        "AWS_ACCESS_KEY_ID": "session-b-key",
        "AWS_SECRET_ACCESS_KEY": "session-b-secret",
    }

    runner._client_ids.add("other-runtime-owner")
    with pytest.raises(PermissionError, match="owning runtime client"):
        cls._require_session(runner, "other-runtime-owner", "session-b")


def test_close_unknown_session_tombstones_ambiguous_open():
    cls, runner = _make_local_query_driver_actor()
    session_id = "ambiguous-open-session"

    asyncio.run(
        cls.close_session(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            session_id,
        )
    )

    assert runner._closed_session_owners[session_id] == _TEST_RUNTIME_OWNER_ID
    with pytest.raises(RuntimeError, match="identity was already closed"):
        asyncio.run(
            cls.open_session(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                session_id,
                {},
            )
        )

    runner._client_ids.add("other-runtime-owner")
    with pytest.raises(PermissionError, match="owning runtime client"):
        asyncio.run(
            cls.close_session(
                runner,
                "other-runtime-owner",
                session_id,
            )
        )


def test_close_plan_cancellation_waits_for_owned_teardown():
    cls, runner = _make_local_query_driver_actor()
    plan_id = "cancelled-close-plan"
    _bind_test_plan_session(runner, plan_id, query_id="")
    cleanup_started = threading.Event()
    cleanup_release = threading.Event()
    cleanup_finished = threading.Event()

    def _cleanup(actual_plan_id):
        assert actual_plan_id == plan_id
        cleanup_started.set()
        assert cleanup_release.wait(timeout=1.0)
        cleanup_finished.set()

    runner._cleanup_finished_plan = _cleanup

    async def _cancel_close():
        close_task = asyncio.create_task(
            cls.close_plan(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
                plan_id,
            )
        )
        assert await asyncio.to_thread(cleanup_started.wait, 1.0)
        close_task.cancel()
        await asyncio.sleep(0.01)
        assert close_task.done() is False
        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await close_task

    asyncio.run(_cancel_close())

    assert cleanup_finished.is_set()


def test_close_plan_is_idempotent_after_plan_teardown():
    cls, runner = _make_local_query_driver_actor()
    cleanup_calls = []
    runner._cleanup_finished_plan = lambda plan_id: cleanup_calls.append(plan_id)

    asyncio.run(
        cls.close_plan(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            "already-closed-plan",
        )
    )

    assert cleanup_calls == []


def test_close_plan_rejects_plan_from_another_session():
    cls, runner = _make_local_query_driver_actor()
    runner._plan_session_ids["other-session-plan"] = "other-session"
    cleanup_calls = []
    runner._cleanup_finished_plan = lambda plan_id: cleanup_calls.append(plan_id)

    with pytest.raises(PermissionError, match="does not belong"):
        asyncio.run(
            cls.close_plan(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
                "other-session-plan",
            )
        )

    assert cleanup_calls == []


@pytest.mark.parametrize(
    "first_owner",
    [_TEST_RUNTIME_OWNER_ID, "other-runtime-owner"],
)
def test_two_runtime_clients_keep_session_config_and_close_order_independent(first_owner):
    cls, runner = _make_local_query_driver_actor()
    other_owner = "other-runtime-owner"
    runner._sessions = {}
    runner._client_ids.add(other_owner)
    sessions = {
        _TEST_RUNTIME_OWNER_ID: (
            "session-a",
            {
                "AWS_ACCESS_KEY_ID": "session-a-key",
                "AWS_SECRET_ACCESS_KEY": "session-a-secret",
            },
        ),
        other_owner: (
            "session-b",
            {
                "AWS_ACCESS_KEY_ID": "session-b-key",
                "AWS_SECRET_ACCESS_KEY": "session-b-secret",
            },
        ),
    }

    for owner_id, (session_id, config) in sessions.items():
        assert asyncio.run(cls.open_session(runner, owner_id, session_id, config)) is True

    first_session_id, _ = sessions[first_owner]
    surviving_owner = other_owner if first_owner == _TEST_RUNTIME_OWNER_ID else _TEST_RUNTIME_OWNER_ID
    surviving_session_id, surviving_config = sessions[surviving_owner]

    assert asyncio.run(cls.detach_client(runner, first_owner)) is False

    assert first_session_id not in runner._sessions
    surviving_session = cls._require_session(runner, surviving_owner, surviving_session_id)
    assert surviving_session.config == surviving_config
    with pytest.raises(PermissionError, match="attached client owner"):
        cls._require_session(runner, first_owner, surviving_session_id)

    asyncio.run(cls.close_session(runner, surviving_owner, surviving_session_id))


@pytest.mark.parametrize("copy_plan", [False, True])
def test_driver_rejects_active_plan_identity_collision(copy_plan):
    cls, runner = _make_local_query_driver_actor()
    session = runner._sessions[_TEST_SESSION_ID]
    plan_id = "duplicate-plan"
    runner._plan_session_ids[plan_id] = "other-session"

    class _LogicalPlan:
        @staticmethod
        def idx():
            return plan_id

    existing_cursor_count = len(session.connection.cursors)
    with pytest.raises(RuntimeError, match="query plan identity is already active"):
        _run_test_plan_preparation(
            cls,
            runner,
            session,
            _LogicalPlan(),
            plan_id,
            copy_plan=copy_plan,
        )

    assert len(session.connection.cursors) == existing_cursor_count
    assert plan_id not in session.plan_ids


@pytest.mark.parametrize("copy_plan", [False, True])
@pytest.mark.parametrize("orphaned_owner", ["teardown", "vllm_pool", "resource_graph"])
def test_driver_rejects_plan_identity_with_orphaned_resource_owner(copy_plan, orphaned_owner):
    cls, runner = _make_local_query_driver_actor()
    session = runner._sessions[_TEST_SESSION_ID]
    plan_id = "orphaned-plan-owner"
    if orphaned_owner == "teardown":
        runner._plan_teardowns_in_progress.add(plan_id)
    elif orphaned_owner == "vllm_pool":
        runner._active_vllm_actors_by_plan[plan_id] = [object()]
    else:
        runner._query_resource_graphs[plan_id] = object()

    class _LogicalPlan:
        @staticmethod
        def idx():
            return plan_id

    with pytest.raises(RuntimeError, match="query plan identity is already active"):
        _run_test_plan_preparation(
            cls,
            runner,
            session,
            _LogicalPlan(),
            plan_id,
            copy_plan=copy_plan,
        )

    assert plan_id not in runner._plan_lifecycles
    assert plan_id not in runner._plan_session_ids
    assert plan_id not in session.plan_ids


def test_teardown_fails_closed_for_orphaned_actor_pool():
    cls, runner = _make_local_query_driver_actor()
    plan_id = "orphaned-actor-pool"
    runner._active_vllm_actors_by_plan[plan_id] = [object()]

    with pytest.raises(RuntimeError, match="active query plan is missing its lifecycle"):
        cls._teardown_plan_resources(runner, plan_id)


@pytest.mark.parametrize("copy_plan", [False, True])
def test_driver_applies_session_s3_config_to_query_cursor(copy_plan):
    cls, runner = _make_local_query_driver_actor()
    session = runner._sessions[_TEST_SESSION_ID]
    session.config = {
        "AWS_ACCESS_KEY_ID": "session-key",
        "AWS_SECRET_ACCESS_KEY": "session-secret",
        "AWS_REGION": "us-east-2",
    }
    session.s3_config = dict(session.config)

    class _LogicalPlan:
        @staticmethod
        def idx():
            return "s3-config-plan"

        @staticmethod
        def has_explicit_s3_credentials():
            return False

        @staticmethod
        def to_physical_plan(_connection, effective_session_config):
            assert effective_session_config == session.s3_config
            raise RuntimeError("planned stop after query cursor configuration")

    with pytest.raises(RuntimeError, match="planned stop after query cursor configuration"):
        _run_test_plan_preparation(
            cls,
            runner,
            session,
            _LogicalPlan(),
            "s3-config-plan",
            copy_plan=copy_plan,
        )

    query_connection = session.connection.cursors[-1]
    assert "SET s3_access_key_id='session-key'" in query_connection.statements
    assert "SET s3_secret_access_key='session-secret'" in query_connection.statements
    assert "SET s3_region='us-east-2'" in query_connection.statements
    assert query_connection.closed is True


@pytest.mark.parametrize("copy_plan", [False, True])
def test_driver_explicit_s3_settings_bypass_and_clear_session_credentials(monkeypatch, copy_plan):
    from vane.runners.ray import worker as worker_module

    cls, runner = _make_local_query_driver_actor()
    session = runner._sessions[_TEST_SESSION_ID]
    session.config = {
        "AWS_ACCESS_KEY_ID": "incomplete-environment-key",
        "AWS_PROFILE": "unavailable-profile",
    }
    session.s3_config = {
        "AWS_ACCESS_KEY_ID": "stale-profile-key",
        "AWS_SECRET_ACCESS_KEY": "stale-profile-secret",
        "AWS_SESSION_TOKEN": "stale-profile-token",
    }
    monkeypatch.setattr(
        worker_module,
        "_resolve_session_aws_credentials",
        lambda _config: (_ for _ in ()).throw(AssertionError("explicit DuckDB credentials must skip resolution")),
    )

    class _LogicalPlan:
        @staticmethod
        def idx():
            return "explicit-s3-config-plan"

        @staticmethod
        def has_explicit_s3_credentials():
            return True

        @staticmethod
        def to_physical_plan(_connection, effective_session_config):
            assert effective_session_config == {}
            raise RuntimeError("planned stop after explicit S3 baseline reset")

    with pytest.raises(RuntimeError, match="planned stop after explicit S3 baseline reset"):
        _run_test_plan_preparation(
            cls,
            runner,
            session,
            _LogicalPlan(),
            "explicit-s3-config-plan",
            copy_plan=copy_plan,
        )

    query_connection = session.connection.cursors[-1]
    assert "SET s3_access_key_id=''" in query_connection.statements
    assert "SET s3_secret_access_key=''" in query_connection.statements
    assert "SET s3_session_token=''" in query_connection.statements
    assert query_connection.closed is True


def test_copy_session_refresh_does_not_block_actor_loop(monkeypatch):
    from vane.runners.ray import worker as worker_module

    cls, runner = _make_local_query_driver_actor()
    session = runner._sessions[_TEST_SESSION_ID]
    refresh_started = threading.Event()
    refresh_release = threading.Event()

    def _delayed_refresh(config, effective_config, *, use_session_credentials):
        assert use_session_credentials is True
        refresh_started.set()
        assert refresh_release.wait(timeout=1.0)
        return dict(effective_config)

    class _LogicalPlan:
        @staticmethod
        def idx():
            return "slow-copy-session-refresh"

        @staticmethod
        def has_explicit_s3_credentials():
            return False

        @staticmethod
        def to_physical_plan(_connection, _effective_session_config):
            raise RuntimeError("planned stop after COPY session refresh")

    monkeypatch.setattr(worker_module, "_refresh_effective_duckdb_s3_config", _delayed_refresh)

    async def _refresh_while_serving_control_requests():
        copy_task = asyncio.create_task(
            cls._run_copy_plan_for_session(
                runner,
                _TEST_SESSION_ID,
                session,
                _LogicalPlan(),
            )
        )
        try:
            assert await asyncio.to_thread(refresh_started.wait, 1.0)
            assert cls.ping(runner, _TEST_RUNTIME_OWNER_ID) is True
        finally:
            refresh_release.set()
        with pytest.raises(RuntimeError, match="planned stop after COPY session refresh"):
            await asyncio.wait_for(copy_task, timeout=1.0)

    asyncio.run(_refresh_while_serving_control_requests())


def test_copy_resource_registration_does_not_block_actor_loop(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    session = runner._sessions[_TEST_SESSION_ID]
    registration_started = threading.Event()
    registration_release = threading.Event()
    plan_id = "slow-copy-resource-registration"

    def _delayed_registration(
        _self,
        _plan,
        query_connection,
        expected_plan_id=None,
    ):
        assert query_connection is runner._test_session_connection.cursors[-1]
        assert expected_plan_id == plan_id
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        registration_started.set()
        assert registration_release.wait(timeout=1.0)
        raise RuntimeError("planned stop after COPY resource registration")

    def _teardown(_self, actual_plan_id):
        assert actual_plan_id == plan_id
        _, _query_id, _query_connection, drop_fragments = _test_plan_teardown_state(runner, actual_plan_id)
        assert drop_fragments is False
        _release_test_plan_session_state(cls, runner, actual_plan_id)

    monkeypatch.setattr(cls, "_prepare_query_resource_registration", _delayed_registration)
    monkeypatch.setattr(cls, "_teardown_plan_resources", _teardown)

    async def _register_while_serving_control_requests():
        copy_task = asyncio.create_task(
            cls._run_copy_plan_for_session(
                runner,
                _TEST_SESSION_ID,
                session,
                _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr(plan_id)),
            )
        )
        try:
            assert await asyncio.to_thread(registration_started.wait, 1.0)
            assert cls.ping(runner, _TEST_RUNTIME_OWNER_ID) is True
        finally:
            registration_release.set()
        with pytest.raises(RuntimeError, match="planned stop after COPY resource registration"):
            await asyncio.wait_for(copy_task, timeout=1.0)

    asyncio.run(_register_while_serving_control_requests())

    assert plan_id not in session.plan_ids
    assert plan_id not in runner._plan_session_ids
    assert runner._test_session_connection.cursors[-1].closed is True


@pytest.mark.parametrize("copy_plan", [False, True])
def test_driver_rejects_logical_to_physical_plan_identity_change(copy_plan):
    cls, runner = _make_local_query_driver_actor()
    session = runner._sessions[_TEST_SESSION_ID]
    physical_plan = _FakePhysicalPlanWithoutPlanAttr("physical-plan")

    class _ChangedIdentityLogicalPlan:
        @staticmethod
        def idx():
            return "logical-plan"

        @staticmethod
        def has_explicit_s3_credentials():
            return False

        @staticmethod
        def to_physical_plan(_connection, _effective_session_config):
            return physical_plan

    existing_cursor_count = len(session.connection.cursors)
    with pytest.raises(RuntimeError, match="logical/physical query plan identity changed"):
        _run_test_plan_preparation(
            cls,
            runner,
            session,
            _ChangedIdentityLogicalPlan(),
            "logical-plan",
            copy_plan=copy_plan,
        )

    assert len(session.connection.cursors) == existing_cursor_count + 1
    assert session.connection.cursors[-1].closed is True
    assert session.plan_ids == set()
    assert runner._plan_session_ids == {}
    assert runner._plan_connections == {}


@pytest.mark.parametrize("copy_plan", [False, True])
@pytest.mark.parametrize(
    ("physical_session_id", "physical_session_config"),
    [
        ("other-session", _TEST_SESSION_CONFIG),
        (_TEST_SESSION_ID, {"AWS_ACCESS_KEY_ID": "other-key"}),
    ],
)
def test_driver_revalidates_physical_plan_session(
    copy_plan,
    physical_session_id,
    physical_session_config,
):
    cls, runner = _make_local_query_driver_actor()
    session = runner._sessions[_TEST_SESSION_ID]
    physical_plan = _FakePhysicalPlanWithoutPlanAttr(
        "physical-session-plan",
        session_id=physical_session_id,
        session_config=physical_session_config,
    )

    class _LogicalPlan:
        @staticmethod
        def idx():
            return "physical-session-plan"

        @staticmethod
        def session_id():
            return _TEST_SESSION_ID

        @staticmethod
        def session_config():
            return dict(_TEST_SESSION_CONFIG)

        @staticmethod
        def has_explicit_s3_credentials():
            return False

        @staticmethod
        def to_physical_plan(_connection, _effective_session_config):
            return physical_plan

    existing_cursor_count = len(session.connection.cursors)
    with pytest.raises(ValueError, match="session mismatch|session config changed"):
        if copy_plan:
            asyncio.run(
                cls.run_copy_plan(
                    runner,
                    _TEST_RUNTIME_OWNER_ID,
                    _TEST_SESSION_ID,
                    _LogicalPlan(),
                )
            )
        else:
            asyncio.run(
                cls.run_plan(
                    runner,
                    _TEST_RUNTIME_OWNER_ID,
                    _TEST_SESSION_ID,
                    _LogicalPlan(),
                )
            )

    assert len(session.connection.cursors) == existing_cursor_count + 1
    assert session.connection.cursors[-1].closed is True
    assert session.plan_ids == set()
    assert session.active_operations == 0
    assert runner._plan_session_ids == {}
    assert runner._plan_connections == {}


def test_normalize_native_task_result_preserves_schema_and_stats():
    m = vane.ray_cxx
    result = m.NativeDistributedTaskResult(
        ["payload"],
        [m.NativePartitionMetadata(3, 42)],
        {"names": ["x"], "types": ["INTEGER"]},
        [1, 2, 3],
        "ok",
        24601,
        {"attempt_id": 2},
        {"processed_input_rows": 3, "processed_input_bytes": 42},
    )

    (
        payloads,
        partition_metadatas,
        result_schema,
        stats,
        completion_status,
        flight_port,
        exchange_sink_instance,
        task_stats,
    ) = _normalize_native_task_result(result)

    assert payloads == ["payload"]
    assert partition_metadatas == [PartitionMetadata(3, 42)]
    assert result_schema == {"names": ["x"], "types": ["INTEGER"]}
    assert stats == [1, 2, 3]
    assert completion_status == "ok"
    assert flight_port == 24601
    assert exchange_sink_instance == {"attempt_id": 2}
    assert task_stats == {"processed_input_rows": 3, "processed_input_bytes": 42}


def test_run_plan_return_uses_native_completed_sink_descriptor(monkeypatch):
    from vane.runners.ray import worker as worker_module

    events: list[tuple[str, str]] = []
    original_require = worker_module.require_ray_cxx_attr

    def fake_require(name, hint=None):
        assert hint
        if name == "begin_flight_shuffle_query_execution":
            return lambda query_id: events.append(("begin", query_id))
        if name == "end_flight_shuffle_query_execution":
            return lambda query_id: events.append(("end", query_id))
        return original_require(name, hint=hint)

    completed_descriptor = {
        "query_id": "query-native-descriptor",
        "attempt_id": 2,
        "flight_server_epoch": "worker-epoch",
    }
    native_result = vane.ray_cxx.NativeDistributedTaskResult(
        [],
        [],
        None,
        [],
        "ok",
        31337,
        completed_descriptor,
        {},
    )

    class DummyWorker:
        _env_overrides: dict[str, str] = {}

        @staticmethod
        def _begin_worker_native_execution(query_id, task_id=""):
            assert task_id == ""
            events.append(("worker_begin", query_id))

        @staticmethod
        def _end_worker_native_execution(query_id, task_id=""):
            assert task_id == ""
            events.append(("worker_end", query_id))

        @staticmethod
        def _worker_native_query_is_closing(_query_id):
            return False

        @staticmethod
        def _worker_native_task_is_closing(task_id):
            assert task_id == ""
            return False

        @staticmethod
        def _execute_native_task(*args, **kwargs):
            events.append(("execute", "query-native-descriptor"))
            return native_result

    monkeypatch.setattr(worker_module, "require_ray_cxx_attr", fake_require)
    actor_class = worker_module.RayWorkerActor.__ray_metadata__.modified_class
    query_lease = {
        "lease_id": "lease-native-descriptor",
        "query_id": "resource-query-native-descriptor",
        "execution_query_id": "query-native-descriptor",
        "resource_unit_id": "resource:test:fragment:native-descriptor",
        "attempt_id": "query-native-descriptor.0.0.0",
        "target_output_block_bytes": 1,
        "output_window_bytes": 1,
    }

    result = asyncio.run(
        actor_class.run_plan_return(
            DummyWorker(),
            object(),
            None,
            query_lease,
            exchange_sink_instance={
                "query_id": "query-native-descriptor",
                "attempt_id": 2,
            },
        )
    )

    assert result[4] == 31337
    assert result[5] == completed_descriptor
    assert events == [
        ("worker_begin", "query-native-descriptor"),
        ("begin", "query-native-descriptor"),
        ("execute", "query-native-descriptor"),
        ("end", "query-native-descriptor"),
        ("worker_end", "query-native-descriptor"),
    ]


def test_run_plan_return_cancellation_waits_for_native_execution(monkeypatch):
    from vane.runners.ray import worker as worker_module

    events: list[tuple[str, str]] = []
    native_started = threading.Event()
    release_native = threading.Event()

    def fake_require(name, hint=None):
        assert hint
        if name == "begin_flight_shuffle_query_execution":
            return lambda query_id: events.append(("begin", query_id))
        if name == "end_flight_shuffle_query_execution":
            return lambda query_id: events.append(("end", query_id))
        raise AssertionError(f"unexpected C++ binding lookup: {name}")

    native_result = vane.ray_cxx.NativeDistributedTaskResult(
        [],
        [],
        None,
        [],
        "ok",
        31338,
        None,
        {},
    )

    class DummyWorker:
        _env_overrides: dict[str, str] = {}

        @staticmethod
        def _begin_worker_native_execution(query_id, task_id=""):
            assert task_id == ""
            events.append(("worker_begin", query_id))

        @staticmethod
        def _end_worker_native_execution(query_id, task_id=""):
            assert task_id == ""
            events.append(("worker_end", query_id))

        @staticmethod
        def _worker_native_query_is_closing(_query_id):
            return False

        @staticmethod
        def _worker_native_task_is_closing(task_id):
            assert task_id == ""
            return False

        @staticmethod
        def _execute_native_task(*args, **kwargs):
            events.append(("execute", "query-native-cancel"))
            native_started.set()
            assert release_native.wait(timeout=2.0)
            events.append(("write_complete", "query-native-cancel"))
            return native_result

    monkeypatch.setattr(worker_module, "require_ray_cxx_attr", fake_require)
    actor_class = worker_module.RayWorkerActor.__ray_metadata__.modified_class
    query_lease = {
        "lease_id": "lease-native-cancel",
        "query_id": "resource-query-native-cancel",
        "execution_query_id": "query-native-cancel",
        "resource_unit_id": "resource:test:fragment:native-cancel",
        "attempt_id": "query-native-cancel.0.0.1",
        "target_output_block_bytes": 1,
        "output_window_bytes": 1,
    }

    async def run():
        task = asyncio.create_task(
            actor_class.run_plan_return(
                DummyWorker(),
                object(),
                None,
                query_lease,
            )
        )
        assert await asyncio.to_thread(native_started.wait, 1.0)

        task.cancel()
        await asyncio.sleep(0.05)

        assert task.done() is False
        assert ("end", "query-native-cancel") not in events
        assert ("worker_end", "query-native-cancel") not in events

        release_native.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(run())

    assert events == [
        ("worker_begin", "query-native-cancel"),
        ("begin", "query-native-cancel"),
        ("execute", "query-native-cancel"),
        ("write_complete", "query-native-cancel"),
        ("end", "query-native-cancel"),
        ("worker_end", "query-native-cancel"),
    ]


def test_normalize_native_task_result_rejects_legacy_shapes():
    with pytest.raises(TypeError, match="execute_native must return NativeDistributedTaskResult"):
        _normalize_native_task_result(([], [], None))


def test_fte_output_publication_validates_metadata_without_hard_estimate_caps():
    lease = {
        "lease_id": "lease-1",
        "query_id": "q",
        "resource_unit_id": "resource:q:fragment:node:1",
        "attempt_id": "q.0.0.0",
        "target_output_block_bytes": 10,
        "output_window_bytes": 20,
    }

    assert _validate_fte_output_publication(
        [PartitionMetadata(1, 10), PartitionMetadata(0, 0)],
        lease,
    ) == (10, 1)

    assert _validate_fte_output_publication([PartitionMetadata(1, 11)], lease) == (11,)
    assert _validate_fte_output_publication(
        [PartitionMetadata(1, 10), PartitionMetadata(1, 10), PartitionMetadata(0, 0)],
        lease,
    ) == (10, 10, 1)
    assert (
        _validate_fte_output_publication(
            [],
            {**lease, "target_output_block_bytes": 0, "output_window_bytes": 0},
        )
        == ()
    )
    with pytest.raises(RuntimeError, match="missing positive size_bytes"):
        _validate_fte_output_publication([PartitionMetadata(1, 0)], lease)


class _RequiredFteWorkerCallbacks:
    @property
    def worker_incarnation_id(self):
        return f"incarnation-{self.worker_id}"

    def fte_cancel_task(self, _task_id):
        return {
            "state": "CANCELED",
            "failure": {
                "error_code": "TASK_CANCELED",
                "message": "task canceled",
            },
        }

    async def fte_wait_task_status_async(self, task_id, min_version, timeout_s):
        return self.fte_wait_task_status(task_id, min_version, timeout_s)

    def mark_fte_worker_failed(self, _worker_id, _error, *, worker_incarnation_id):
        return []

    def handle_fte_task_status(self, _status):
        return []

    def fte_attempt_is_selected(self, _task_id):
        return True

    def record_fte_task_terminal(self, _task_id):
        return None

    def finish_fte_task_with_outputs(self, _task_id, _query_task_lease, outputs):
        return [_FakeOutputLeaseOwner() for _ in outputs]

    def fte_ack_task_result(self, _task_id):
        return {"state": "FINISHED"}

    def fte_release_task_result(self, _task_id):
        return {"state": "FINISHED"}

    def enqueue_fte_ack_task_result(self, task_id):
        return self.fte_ack_task_result(task_id)

    def enqueue_fte_release_task_result(self, task_id):
        return self.fte_release_task_result(task_id)


class _FakeFteStatusWorker(_RequiredFteWorkerCallbacks):
    worker_id = "worker-a"

    def __init__(self):
        self.status = {"state": "RUNNING"}
        self.calls = []
        self.terminal_attempts = []
        self.ack_calls = []
        self.release_calls = []
        self.output_transfers = []

    def fte_get_task_status(self, _task_id):
        raise AssertionError("FTE task handles must use status wait, not status polling")

    def fte_wait_task_status(self, task_id, min_version, timeout_s):
        self.calls.append(("wait", task_id, min_version, timeout_s))
        status = dict(self.status)
        status.setdefault("task_id", task_id)
        return status

    def fte_cancel_task(self, task_id):
        self.calls.append(("cancel", task_id))
        self.status = {
            "state": "CANCELED",
            "task_id": task_id,
            "failure": {
                "error_code": "TASK_CANCELED",
                "message": "task canceled",
            },
        }
        return dict(self.status)

    def fte_ack_task_result(self, task_id):
        self.ack_calls.append(task_id)
        return {"state": "FINISHED", "task_id": task_id}

    def fte_release_task_result(self, task_id):
        self.release_calls.append(task_id)
        return {"state": "FINISHED", "task_id": task_id}

    def record_fte_task_terminal(self, task_id):
        self.terminal_attempts.append(str(task_id))

    def finish_fte_task_with_outputs(self, task_id, query_task_lease, outputs):
        self.output_transfers.append((str(driver.FteTaskAttemptId.coerce(task_id)), dict(query_task_lease), outputs))
        return [_FakeOutputLeaseOwner() for _ in outputs]


def _wait_batch_ready(handle, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ready = driver.batch_wait_ready([handle])
        if ready:
            return ready
        time.sleep(0.01)
    return driver.batch_wait_ready([handle])


def test_fte_worker_task_handle_finishes_via_status_wait():
    worker = _FakeFteStatusWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    handle = driver.FteWorkerTaskHandle(task_id, worker)

    assert handle.done() is False
    worker.status = {"state": "FINISHED", "stats": [1, 2, 3]}
    assert _wait_batch_ready(handle) == [0]

    result = handle.get_result_sync()
    assert result.ok
    assert result.result_schema is None
    assert worker.calls[0] == ("wait", task_id, -1, handle.status_wait_timeout_s)
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_status_transition_uses_dedicated_executor(monkeypatch):
    worker = _FakeFteStatusWorker()
    worker.status = {"state": "FINISHED", "stats": [1]}
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    handle = driver.FteWorkerTaskHandle(task_id, worker)
    original_apply_status = handle._apply_status
    transition_threads = []

    def _apply_status(status):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        transition_threads.append(threading.current_thread().name)
        original_apply_status(status)

    handle._apply_status = _apply_status

    async def _unexpected_to_thread(*_args, **_kwargs):
        raise AssertionError("successful FTE status adoption must not use the default executor")

    monkeypatch.setattr(driver.asyncio, "to_thread", _unexpected_to_thread)

    assert _wait_batch_ready(handle) == [0]
    assert handle.get_result_sync().ok
    assert transition_threads
    assert transition_threads[0].startswith("vane-fte-status")


def test_fte_worker_task_handle_starts_one_watcher_under_concurrent_polling(
    monkeypatch,
):
    class _SingleTransferWorker(_FakeFteStatusWorker):
        def __init__(self):
            super().__init__()
            self._transfer_lock = threading.Lock()
            self._transferred = False

        def finish_fte_task_with_outputs(
            self,
            task_id,
            query_task_lease,
            outputs,
        ):
            with self._transfer_lock:
                if self._transferred:
                    raise RuntimeError("FTE task lease is not active")
                self._transferred = True
            return super().finish_fte_task_with_outputs(
                task_id,
                query_task_lease,
                outputs,
            )

    worker = _SingleTransferWorker()
    task_id = {
        "query_id": "q",
        "fragment_execution_id": 1,
        "partition_id": 2,
        "attempt_id": 0,
    }
    worker.status = {
        "state": "FINISHED",
        "result": (["payload"], [{"num_rows": 5, "size_bytes": 64}], None, []),
    }
    handle = driver.FteWorkerTaskHandle(
        task_id,
        worker,
        query_task_lease={"lease_id": "lease-result"},
    )
    background_loop = driver._ensure_background_event_loop()
    first_lookup_entered = threading.Event()
    release_first_lookup = threading.Event()
    second_lookup_entered = threading.Event()
    lookup_lock = threading.Lock()
    lookup_count = 0

    def racing_get_event_loop():
        nonlocal lookup_count
        with lookup_lock:
            lookup_count += 1
            current = lookup_count
        if current == 1:
            first_lookup_entered.set()
            assert release_first_lookup.wait(timeout=2)
        else:
            second_lookup_entered.set()
        return background_loop

    monkeypatch.setattr(driver, "_get_global_event_loop", racing_get_event_loop)
    polls = [threading.Thread(target=handle.done) for _ in range(2)]
    polls[0].start()
    assert first_lookup_entered.wait(timeout=2)
    polls[1].start()
    second_lookup_entered.wait(timeout=0.2)
    release_first_lookup.set()
    for poll in polls:
        poll.join(timeout=2)
        assert not poll.is_alive()

    assert lookup_count == 1
    assert _wait_batch_ready(handle) == [0]
    assert handle.get_result_sync().ok
    assert len(worker.output_transfers) == 1
    assert len(worker.calls) == 1


def test_fte_finish_wins_atomically_over_concurrent_cancel():
    ack_entered = threading.Event()
    release_ack = threading.Event()

    class _BlockingAckWorker(_FakeFteStatusWorker):
        def enqueue_fte_ack_task_result(self, task_id):
            ack_entered.set()
            assert release_ack.wait(timeout=2)
            return super().enqueue_fte_ack_task_result(task_id)

    worker = _BlockingAckWorker()
    task_id = {
        "query_id": "q",
        "fragment_execution_id": 1,
        "partition_id": 2,
        "attempt_id": 0,
    }
    handle = driver.FteWorkerTaskHandle(
        task_id,
        worker,
        query_task_lease={"lease_id": "lease-finish-cancel"},
    )
    status = {
        "state": "FINISHED",
        "task_id": task_id,
        "result": (["payload"], [{"num_rows": 5, "size_bytes": 64}], None, []),
    }
    finishing = threading.Thread(target=handle._apply_status, args=(status,))
    finishing.start()
    assert ack_entered.wait(timeout=2)

    cancelling = threading.Thread(target=handle.cancel)
    cancelling.start()
    release_ack.set()
    finishing.join(timeout=2)
    cancelling.join(timeout=2)

    assert not finishing.is_alive()
    assert not cancelling.is_alive()
    result = handle.get_result_sync()
    assert result.has_output is True
    assert [call[0] for call in worker.calls if call[0] == "cancel"] == []
    assert worker.release_calls == []
    assert len(worker.output_transfers) == 1


def test_fte_terminal_record_failure_is_not_masked_by_adopted_result():
    class _TerminalRecordFailWorker(_FakeFteStatusWorker):
        def __init__(self):
            super().__init__()
            self.new_owners = []

        def finish_fte_task_with_outputs(self, task_id, query_task_lease, outputs):
            self.output_transfers.append(
                (
                    str(driver.FteTaskAttemptId.coerce(task_id)),
                    dict(query_task_lease),
                    outputs,
                )
            )
            self.new_owners = [_FakeOutputLeaseOwner() for _ in outputs]
            return list(self.new_owners)

        def record_fte_task_terminal(self, _task_id):
            raise RuntimeError("planned terminal record failure")

    worker = _TerminalRecordFailWorker()
    worker.status = {
        "state": "FINISHED",
        "result": (["payload"], [{"num_rows": 5, "size_bytes": 64}], None, []),
    }
    handle = driver.FteWorkerTaskHandle(
        {
            "query_id": "q",
            "fragment_execution_id": 1,
            "partition_id": 2,
            "attempt_id": 0,
        },
        worker,
        query_task_lease={"lease_id": "lease-terminal-record"},
    )

    assert _wait_batch_ready(handle) == [0]
    with pytest.raises(RuntimeError, match="planned terminal record failure"):
        handle.get_result_sync()
    assert len(worker.new_owners) == 1
    assert worker.new_owners[0].released is True
    assert worker.release_calls == [handle.task_id.to_dict()]


def test_fte_worker_task_handle_requires_status_wait_protocol():
    class _StatusOnlyWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-without-status-wait"
        fte_wait_task_status_async = None

        def fte_get_task_status(self, task_id):
            return {"state": "FINISHED", "task_id": task_id, "stats": [1]}

    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    with pytest.raises(RuntimeError, match="must provide fte_wait_task_status_async"):
        driver.FteWorkerTaskHandle(task_id, _StatusOnlyWorker())


def test_fte_worker_task_handle_requires_worker_id():
    class _NoWorkerIdWorker(_RequiredFteWorkerCallbacks):
        def fte_wait_task_status(self, task_id, _min_version, _timeout_s):
            return {"state": "FINISHED", "task_id": task_id, "stats": [1]}

    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    with pytest.raises(AttributeError, match="worker_id"):
        driver.FteWorkerTaskHandle(task_id, _NoWorkerIdWorker())


def test_fte_worker_task_handle_requires_worker_incarnation_id():
    class _NoWorkerIncarnationIdWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-without-incarnation"

        @property
        def worker_incarnation_id(self):
            raise AttributeError("worker_incarnation_id")

        def fte_wait_task_status(self, task_id, _min_version, _timeout_s):
            return {"state": "FINISHED", "task_id": task_id, "stats": [1]}

    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    with pytest.raises(AttributeError, match="worker_incarnation_id"):
        driver.FteWorkerTaskHandle(task_id, _NoWorkerIncarnationIdWorker())


def test_fte_worker_task_handle_finishes_with_result_payload():
    worker = _FakeFteStatusWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    schema = {"names": ["x"], "types": ["INTEGER"]}
    worker.status = {
        "state": "FINISHED",
        "result": (["payload"], [{"num_rows": 5, "size_bytes": 64}], schema, [9]),
    }
    handle = driver.FteWorkerTaskHandle(
        task_id,
        worker,
        query_task_lease={"lease_id": "lease-result"},
    )

    assert _wait_batch_ready(handle) == [0]

    result = handle.get_result_sync()
    assert result.ok
    assert result.result_schema == schema
    assert worker.output_transfers == [
        (
            "q.1.2.0",
            {"lease_id": "lease-result"},
            [{"block_id": "fte-block:lease-result:0", "size_bytes": 64}],
        )
    ]
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_handle_rolls_back_new_output_owners_when_replacing_owner_fails():
    class _RaisingPreviousOwner:
        def transition_to(self, _state):
            return True

        def release(self):
            raise RuntimeError("previous owner release failed")

    class _TrackingWorker(_FakeFteStatusWorker):
        def __init__(self):
            super().__init__()
            self.new_owners = []

        def finish_fte_task_with_outputs(self, task_id, query_task_lease, outputs):
            self.output_transfers.append(
                (str(driver.FteTaskAttemptId.coerce(task_id)), dict(query_task_lease), outputs)
            )
            self.new_owners = [_FakeOutputLeaseOwner() for _ in outputs]
            return list(self.new_owners)

    worker = _TrackingWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    previous_ref = vane.ray_cxx.RayResultPartitionRef(
        "payload",
        5,
        64,
        _RaisingPreviousOwner(),
    )
    handle = driver.FteWorkerTaskHandle(
        task_id,
        worker,
        query_task_lease={"lease_id": "lease-result"},
    )

    with pytest.raises(RuntimeError, match="previous owner release failed"):
        handle._normalize_raw_result(([previous_ref], [{"num_rows": 5, "size_bytes": 64}], None, []))

    assert len(worker.new_owners) == 1
    assert worker.new_owners[0].released is True


def test_fte_worker_task_handle_releases_partial_invalid_ownership_transfer():
    class _InvalidTransferWorker(_FakeFteStatusWorker):
        def __init__(self):
            super().__init__()
            self.owner = _FakeOutputLeaseOwner()

        def finish_fte_task_with_outputs(self, _task_id, _query_task_lease, _outputs):
            return [self.owner]

    worker = _InvalidTransferWorker()
    handle = driver.FteWorkerTaskHandle(
        {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0},
        worker,
    )

    with pytest.raises(RuntimeError, match="one owner per output"):
        handle._finish_task_output_ownership(
            [
                {"block_id": "block-0", "size_bytes": 1},
                {"block_id": "block-1", "size_bytes": 1},
            ]
        )

    assert worker.owner.released is True


def test_fte_worker_task_handle_releases_adopted_and_remote_results_when_ack_fails():
    class _AckFailWorker(_FakeFteStatusWorker):
        def __init__(self):
            super().__init__()
            self.new_owners = []

        def finish_fte_task_with_outputs(self, task_id, query_task_lease, outputs):
            self.output_transfers.append(
                (str(driver.FteTaskAttemptId.coerce(task_id)), dict(query_task_lease), outputs)
            )
            self.new_owners = [_FakeOutputLeaseOwner() for _ in outputs]
            return list(self.new_owners)

        def fte_ack_task_result(self, task_id):
            self.ack_calls.append(task_id)
            raise RuntimeError("planned ack failure")

    worker = _AckFailWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    worker.status = {
        "state": "FINISHED",
        "result": (["payload"], [{"num_rows": 5, "size_bytes": 64}], None, []),
    }
    handle = driver.FteWorkerTaskHandle(
        task_id,
        worker,
        query_task_lease={"lease_id": "lease-result"},
    )

    assert _wait_batch_ready(handle) == [0]
    with pytest.raises(RuntimeError, match="planned ack failure"):
        handle.get_result_sync()

    assert worker.ack_calls == [task_id]
    assert worker.release_calls == [task_id]
    assert len(worker.new_owners) == 1
    assert worker.new_owners[0].released is True


def test_fte_worker_task_handle_defers_attempt_selection_to_query_commit():
    class _SelectionRacingWorker(_FakeFteStatusWorker):
        def fte_attempt_is_selected(self, _task_id):
            raise AssertionError("result adoption must not race the query-level selected-attempt decision")

    worker = _SelectionRacingWorker()
    task_id = {
        "query_id": "q",
        "fragment_execution_id": 1,
        "partition_id": 2,
        "attempt_id": 1,
    }
    worker.status = {
        "state": "FINISHED",
        "result": (["loser-ref"], [{"num_rows": 5, "size_bytes": 64}], None, []),
    }
    handle = driver.FteWorkerTaskHandle(
        task_id,
        worker,
        query_task_lease={"lease_id": "loser-lease"},
    )

    assert _wait_batch_ready(handle) == [0]
    result = handle.get_result_sync()

    assert result.ok
    assert worker.ack_calls == [task_id]
    assert worker.release_calls == []
    assert worker.output_transfers == [
        (
            "q.1.2.1",
            {"lease_id": "loser-lease"},
            [{"block_id": "fte-block:loser-lease:0", "size_bytes": 64}],
        )
    ]
    assert worker.terminal_attempts == ["q.1.2.1"]


def test_fte_worker_task_handle_acks_remote_result_once():
    worker = _FakeFteStatusWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    worker.status = {
        "state": "FINISHED",
        "result": (["payload"], [{"num_rows": 5, "size_bytes": 64}], None, []),
    }
    handle = driver.FteWorkerTaskHandle(task_id, worker)

    assert _wait_batch_ready(handle) == [0]
    assert handle.get_result_sync().ok
    assert handle.get_result_sync().ok

    assert worker.ack_calls == [task_id]


def test_fte_worker_task_handle_ack_does_not_release_remote_result():
    worker = _FakeFteStatusWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    worker.status = {
        "state": "FINISHED",
        "result": (["payload"], [{"num_rows": 5, "size_bytes": 64}], None, []),
    }
    handle = driver.FteWorkerTaskHandle(task_id, worker)

    assert _wait_batch_ready(handle) == [0]
    assert handle.get_result_sync().ok

    assert worker.ack_calls == [task_id]
    assert worker.release_calls == []


def test_fte_worker_task_handle_release_result_payload_calls_worker_once():
    worker = _FakeFteStatusWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    handle = driver.FteWorkerTaskHandle(task_id, worker)

    handle.release_result_payload()
    handle.release_result_payload()

    assert worker.release_calls == [task_id]


def test_fte_worker_task_handle_enqueues_result_controls_without_sync_rpc():
    class _QueuedControlWorker(_FakeFteStatusWorker):
        def __init__(self):
            super().__init__()
            self.queued_controls = []

        def fte_ack_task_result(self, _task_id):
            raise AssertionError("result ACK must not synchronously resolve a Ray control RPC")

        def fte_release_task_result(self, _task_id):
            raise AssertionError("result release must not synchronously resolve a Ray control RPC")

        def enqueue_fte_ack_task_result(self, task_id):
            self.queued_controls.append(("ack", task_id))

        def enqueue_fte_release_task_result(self, task_id):
            self.queued_controls.append(("release", task_id))

    worker = _QueuedControlWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    handle = driver.FteWorkerTaskHandle(task_id, worker)
    handle._result = driver.RayTaskResult.success([], [1], None)

    assert handle.get_result_sync().ok
    handle.release_result_payload()

    assert worker.queued_controls == [("ack", task_id), ("release", task_id)]


def test_fte_worker_task_handle_does_not_publish_finished_status_event():
    class _EventWorker(_FakeFteStatusWorker):
        def __init__(self):
            super().__init__()
            self.status_events = []

        def handle_fte_task_status(self, status):
            self.status_events.append(dict(status))
            return []

    worker = _EventWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    worker.status = {"state": "FINISHED", "stats": [1]}
    handle = driver.FteWorkerTaskHandle(task_id, worker)

    assert _wait_batch_ready(handle) == [0]

    assert worker.status_events == []


def test_fte_worker_task_handle_does_not_publish_failed_status_event():
    class _FailingEventWorker(_FakeFteStatusWorker):
        def handle_fte_task_status(self, _status):
            raise RuntimeError("publish exploded")

    worker = _FailingEventWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    worker.status = {
        "state": "FAILED",
        "failure": {
            "error_code": "GENERIC_INTERNAL_ERROR",
            "message": "remote failed",
        },
    }
    handle = driver.FteWorkerTaskHandle(task_id, worker)

    assert _wait_batch_ready(handle) == [0]

    with pytest.raises(RuntimeError, match="remote failed"):
        handle.get_result_sync()
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_handle_does_not_adopt_retry_from_data_plane():
    class _EventWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-a"

        def fte_wait_task_status(self, task_id, _min_version, _timeout_s):
            return {
                "state": "FAILED",
                "task_id": task_id,
                "failure": {
                    "error_code": "GENERIC_INTERNAL_ERROR",
                    "message": "retryable",
                },
                "version": 1,
            }

        def handle_fte_task_status(self, _status):
            raise AssertionError("the authoritative status watcher owns retry scheduling")

    handle = driver.FteWorkerTaskHandle(
        {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0},
        _EventWorker(),
    )

    with pytest.raises(RuntimeError, match="retryable"):
        asyncio.run(handle.get_result())
    assert handle.task_id.attempt_id == 0


def test_fte_worker_task_handle_malformed_status_fails_worker_and_records_terminal_once():
    class _MalformedStatusWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-malformed-status"

        def __init__(self):
            self.worker_failures = []
            self.terminal_attempts = []

        def fte_wait_task_status(self, task_id, _min_version, _timeout_s):
            return {
                "state": "MALFORMED",
                "task_id": task_id,
                "version": 1,
            }

        def mark_fte_worker_failed(self, worker_id, error, *, worker_incarnation_id):
            self.worker_failures.append((worker_id, error))
            return []

        def record_fte_task_terminal(self, task_id):
            self.terminal_attempts.append(str(driver.FteTaskAttemptId.coerce(task_id)))

    worker = _MalformedStatusWorker()
    handle = driver.FteWorkerTaskHandle(
        {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0},
        worker,
    )

    with pytest.raises(RuntimeError, match="failed to apply FTE task status.*MALFORMED"):
        asyncio.run(handle.get_result())

    assert handle.done() is True
    assert len(worker.worker_failures) == 1
    assert worker.worker_failures[0][0] == "worker-malformed-status"
    assert "status protocol failed" in str(worker.worker_failures[0][1])
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_handle_preserves_ray_oom_type_when_publishing_worker_failure():
    class _OomStatusWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-oom-status"

        def __init__(self, error):
            self.error = error
            self.worker_failures = []
            self.terminal_attempts = []

        def fte_wait_task_status(self, _task_id, _min_version, _timeout_s):
            raise self.error

        def mark_fte_worker_failed(self, worker_id, error, *, worker_incarnation_id):
            self.worker_failures.append((worker_id, error))
            return []

        def record_fte_task_terminal(self, task_id):
            self.terminal_attempts.append(str(driver.FteTaskAttemptId.coerce(task_id)))

    oom_error = driver.ray.exceptions.OutOfMemoryError("Ray killed the worker for memory pressure")
    worker = _OomStatusWorker(oom_error)
    handle = driver.FteWorkerTaskHandle(
        {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0},
        worker,
    )

    with pytest.raises(driver.ray.exceptions.OutOfMemoryError, match="memory pressure"):
        asyncio.run(handle.get_result())

    assert len(worker.worker_failures) == 1
    worker_id, published_error = worker.worker_failures[0]
    assert worker_id == "worker-oom-status"
    assert "status wait failed" in str(published_error)
    assert published_error.__cause__ is oom_error
    from vane.runners.ray.fte_fragment_scheduler import _worker_failure_payload

    assert (
        _worker_failure_payload(
            worker_id,
            published_error,
            worker_incarnation_id=worker.worker_incarnation_id,
        )["error_code"]
        == "OUT_OF_MEMORY"
    )
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_handle_publishes_failure_with_unprintable_message():
    class _UnprintableError(RuntimeError):
        def __str__(self):
            raise RuntimeError("exception message is unreadable")

    class _StatusWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-unprintable-status"

        def __init__(self, error):
            self.error = error
            self.worker_failures = []
            self.terminal_attempts = []

        def fte_wait_task_status(self, _task_id, _min_version, _timeout_s):
            raise self.error

        def mark_fte_worker_failed(self, worker_id, error, *, worker_incarnation_id):
            self.worker_failures.append((worker_id, error))
            return []

        def record_fte_task_terminal(self, task_id):
            self.terminal_attempts.append(str(driver.FteTaskAttemptId.coerce(task_id)))

    failure = _UnprintableError()
    worker = _StatusWorker(failure)
    handle = driver.FteWorkerTaskHandle(
        {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0},
        worker,
    )

    with pytest.raises(_UnprintableError):
        asyncio.run(handle.get_result())

    assert len(worker.worker_failures) == 1
    worker_id, published_error = worker.worker_failures[0]
    assert worker_id == "worker-unprintable-status"
    assert str(published_error).endswith("<unprintable _UnprintableError>")
    assert published_error.__cause__ is failure
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_handle_rejects_mismatched_status_identity():
    class _MismatchedStatusWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-mismatched-status"

        def __init__(self):
            self.worker_failures = []
            self.terminal_attempts = []

        def fte_wait_task_status(self, _task_id, _min_version, _timeout_s):
            return {
                "state": "FINISHED",
                "task_id": {
                    "query_id": "q",
                    "fragment_execution_id": 1,
                    "partition_id": 99,
                    "attempt_id": 0,
                },
                "version": 1,
            }

        def mark_fte_worker_failed(self, worker_id, error, *, worker_incarnation_id):
            self.worker_failures.append((worker_id, error))
            return []

        def record_fte_task_terminal(self, task_id):
            self.terminal_attempts.append(str(driver.FteTaskAttemptId.coerce(task_id)))

    worker = _MismatchedStatusWorker()
    handle = driver.FteWorkerTaskHandle(
        {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0},
        worker,
    )

    with pytest.raises(RuntimeError, match="status identity mismatch"):
        asyncio.run(handle.get_result())

    assert len(worker.worker_failures) == 1
    assert worker.worker_failures[0][0] == "worker-mismatched-status"
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_handle_treats_query_deadline_as_hard_failure():
    class _DeadlineWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-query-deadline"

        def __init__(self):
            self.worker_failures = []
            self.terminal_attempts = []

        def fte_wait_task_status(self, _task_id, _min_version, _timeout_s):
            raise QueryDeadlineExceeded("query deadline expired before Ray ObjectRef get")

        def mark_fte_worker_failed(self, worker_id, error, *, worker_incarnation_id):
            self.worker_failures.append((worker_id, error))
            return []

        def record_fte_task_terminal(self, task_id):
            self.terminal_attempts.append(str(driver.FteTaskAttemptId.coerce(task_id)))

    worker = _DeadlineWorker()
    handle = driver.FteWorkerTaskHandle(
        {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0},
        worker,
    )

    with pytest.raises(QueryDeadlineExceeded, match="query deadline expired"):
        asyncio.run(handle.get_result())

    assert len(worker.worker_failures) == 1
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_handle_rejects_exchange_finish_without_final_info():
    worker = _FakeFteStatusWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    worker.status = {"state": "FINISHED"}
    handle = driver.FteWorkerTaskHandle(
        task_id,
        worker,
        task_context_info={"exchange_sink_instance": {"attempt_id": 0}},
    )

    assert _wait_batch_ready(handle) == [0]
    with pytest.raises(RuntimeError, match="FINISHED without final task info"):
        handle.get_result_sync()
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_handle_accepts_exchange_finish_with_spooling_stats():
    worker = _FakeFteStatusWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    worker.status = {
        "state": "FINISHED",
        "spooling_output_stats": {"rows": 3},
    }
    handle = driver.FteWorkerTaskHandle(
        task_id,
        worker,
        task_context_info={"exchange_sink_instance": {"attempt_id": 0}},
    )

    assert _wait_batch_ready(handle) == [0]
    result = handle.get_result_sync()

    assert result.ok
    assert result.exchange_sink_instance == {"attempt_id": 0}
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_handle_uses_status_long_poll_when_available():
    class _LongPollWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-a"

        def __init__(self):
            self.calls = []

        def fte_wait_task_status(self, task_id, min_version, timeout_s):
            self.calls.append(("wait", task_id, min_version, timeout_s))
            return {
                "state": "FINISHED",
                "task_id": task_id,
                "version": 7,
                "stats": [4, 5, 6],
            }

        def fte_get_task_status(self, _task_id):
            raise AssertionError("long-poll path should not use synchronous status polling")

    worker = _LongPollWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    handle = driver.FteWorkerTaskHandle(task_id, worker)

    result = asyncio.run(handle.get_result())

    assert result.ok
    assert worker.calls == [("wait", task_id, -1, handle.status_wait_timeout_s)]


def test_fte_worker_task_handle_get_result_sync_accepts_prepopulated_result():
    class _SlowLongPollWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-a"

        def fte_wait_task_status(self, task_id, _min_version, _timeout_s):
            time.sleep(60)
            return {"state": "RUNNING", "task_id": task_id, "version": 1}

    handle = driver.FteWorkerTaskHandle(
        {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0},
        _SlowLongPollWorker(),
    )
    handle._result = driver.RayTaskResult.success([], [1, 2, 3], None)

    assert _wait_batch_ready(handle) == [0]
    result = handle.get_result_sync()

    assert result.ok
    assert handle.done() is True


def test_fte_worker_task_handle_publishes_worker_loss_without_adopting_retry():
    class _RetryWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-b"

        def fte_wait_task_status(self, task_id, _min_version, _timeout_s):
            return {
                "state": "FINISHED",
                "task_id": task_id,
                "version": 1,
                "stats": [8],
            }

    class _LostWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-a"

        def __init__(self):
            self.calls = []

        def fte_wait_task_status(self, task_id, min_version, timeout_s):
            self.calls.append(("wait", task_id, min_version, timeout_s))
            raise RuntimeError("actor lost")

        def mark_fte_worker_failed(self, worker_id, error, *, worker_incarnation_id):
            self.calls.append(("mark_failed", worker_id, error))
            return [
                driver.FteWorkerTaskHandle(
                    {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 1},
                    _RetryWorker(),
                )
            ]

    worker = _LostWorker()
    handle = driver.FteWorkerTaskHandle(
        {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0},
        worker,
    )

    with pytest.raises(RuntimeError, match="actor lost"):
        asyncio.run(handle.get_result())
    assert handle.task_id.attempt_id == 0
    assert worker.calls[0][0] == "wait"
    assert worker.calls[1][0] == "mark_failed"


def test_fte_worker_task_handle_does_not_pop_scheduler_result_registry_after_worker_lost():
    class _RetryWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-b"

        def fte_wait_task_status(self, task_id, _min_version, _timeout_s):
            return {
                "state": "FINISHED",
                "task_id": task_id,
                "version": 1,
                "stats": [13],
            }

        def record_fte_task_terminal(self, _task_id):
            return None

    class _Coordinator:
        worker_id = "coordinator"

        def __init__(self):
            self.retry = driver.FteWorkerTaskHandle(
                {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 1},
                _RetryWorker(),
            )
            self.pop_calls = 0

        def pop_fte_result_handle_for_task(self, _task_id):
            self.pop_calls += 1
            retry = self.retry
            self.retry = None
            return retry

    class _LostWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-a"

        def fte_wait_task_status(self, _task_id, _min_version, _timeout_s):
            raise RuntimeError("actor lost")

        def mark_fte_worker_failed(self, _worker_id, _error, *, worker_incarnation_id):
            return []

        def pop_fte_result_handle_for_task(self, task_id):
            return coordinator.pop_fte_result_handle_for_task(task_id)

    coordinator = _Coordinator()
    handle = driver.FteWorkerTaskHandle(
        {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0},
        _LostWorker(),
    )

    with pytest.raises(RuntimeError, match="actor lost"):
        asyncio.run(handle.get_result())

    assert handle.task_id.attempt_id == 0
    assert handle.worker_id == "worker-a"
    assert coordinator.pop_calls == 0


def test_fte_worker_task_handle_failed_status_raises_result_error():
    worker = _FakeFteStatusWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    handle = driver.FteWorkerTaskHandle(task_id, worker)
    worker.status = {
        "state": "FAILED",
        "failure": {
            "error_code": "GENERIC_INTERNAL_ERROR",
            "message": "boom",
        },
    }

    assert _wait_batch_ready(handle) == [0]
    with pytest.raises(RuntimeError, match="boom"):
        handle.get_result_sync()
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_handle_cancel_calls_worker():
    worker = _FakeFteStatusWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    handle = driver.FteWorkerTaskHandle(task_id, worker)

    handle.cancel()

    assert handle.done() is True
    assert worker.calls == [("cancel", task_id)]
    assert worker.release_calls == [task_id]


def test_get_next_partition_wraps_metadata_aware_fragment(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "plan-ok"
    manager = _bind_test_query_resource_owner(runner, plan_id)
    payload = object()
    fragment = vane.ray_cxx.RayResultPartitionRef(payload, 7, 99, _FakeOutputLeaseOwner())
    runner.curr_streams[plan_id] = _DummyStream([fragment])
    runner.curr_plans[plan_id] = object()

    class _LocalMetadataAccessor:
        def __init__(self, metadatas):
            self._metadatas = list(metadatas)

        def get_index(self, key: int):
            return self._metadatas[key]

    monkeypatch.setattr(
        PartitionMetadataAccessor,
        "from_metadata_list",
        classmethod(lambda _cls, meta: _LocalMetadataAccessor(meta)),
    )
    monkeypatch.setattr(
        partition_metadata,
        "resolve_object_refs_blocking",
        lambda value, **_kwargs: value,
    )

    result = asyncio.run(
        cls.get_next_partition(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            plan_id,
        )
    )

    assert result is not None
    assert result.partition_ref() is payload
    assert result.partition() is payload
    assert result.metadata() == PartitionMetadata(7, 99)
    assert manager.snapshot()["external_consumer_waiting"] is False


def test_get_next_partition_awaits_native_readiness_without_default_executor(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "plan-async-readiness"
    manager = _bind_test_query_resource_owner(runner, plan_id)
    payload = object()
    fragment = vane.ray_cxx.RayResultPartitionRef(payload, 7, 99, _FakeOutputLeaseOwner())
    runner.curr_plans[plan_id] = object()

    class _LocalMetadataAccessor:
        def __init__(self, metadatas):
            self._metadatas = list(metadatas)

        def get_index(self, key: int):
            return self._metadatas[key]

    monkeypatch.setattr(
        PartitionMetadataAccessor,
        "from_metadata_list",
        classmethod(lambda _cls, meta: _LocalMetadataAccessor(meta)),
    )
    monkeypatch.setattr(
        partition_metadata,
        "resolve_object_refs_blocking",
        lambda value, **_kwargs: value,
    )

    async def _unexpected_to_thread(*_args, **_kwargs):
        raise AssertionError("stream readiness must not use the default executor")

    monkeypatch.setattr(driver.asyncio, "to_thread", _unexpected_to_thread)

    async def _run():
        stream = _NotifyingStream()
        runner.curr_streams[plan_id] = stream
        partition_read = asyncio.create_task(
            cls.get_next_partition(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
                plan_id,
            )
        )
        await stream.armed.wait()
        stream.publish(fragment)
        return await partition_read

    result = asyncio.run(_run())

    assert result is not None
    assert result.partition() is payload
    assert manager.snapshot()["external_consumer_waiting"] is False


def test_get_next_partition_leases_and_releases_metadata_aware_fragment(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "plan-lease"
    _bind_test_query_resource_owner(runner, plan_id)
    payload = object()
    output_owner = _FakeOutputLeaseOwner()
    fragment = vane.ray_cxx.RayResultPartitionRef(payload, 7, 99, output_owner)
    runner.curr_streams[plan_id] = _DummyStream([fragment])
    runner.curr_plans[plan_id] = object()

    released = []

    class _FakeRemoteMethod:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args):
            return self._fn(*args)

    class _FakeReleaseOwner:
        def __init__(self):
            self.release_result_partition_ref = _FakeRemoteMethod(self._release)

        def _release(self, owner_id, session_id, owner_plan_id, release_token):
            released.append((owner_id, session_id, owner_plan_id, release_token))
            return cls.release_result_partition_ref(
                runner,
                owner_id,
                session_id,
                owner_plan_id,
                release_token,
            )

    class _LocalMetadataAccessor:
        def __init__(self, metadatas):
            self._metadatas = list(metadatas)

        def get_index(self, key: int):
            return self._metadatas[key]

    monkeypatch.setattr(
        PartitionMetadataAccessor,
        "from_metadata_list",
        classmethod(lambda _cls, meta: _LocalMetadataAccessor(meta)),
    )
    monkeypatch.setattr(
        partition_metadata,
        "resolve_object_refs_blocking",
        lambda value, **_kwargs: value,
    )

    result = asyncio.run(
        cls.get_next_partition(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            plan_id,
            release_owner=_FakeReleaseOwner(),
        )
    )

    assert result is not None
    assert result.partition() is payload
    assert released == [(_TEST_RUNTIME_OWNER_ID, _TEST_SESSION_ID, plan_id, "0")]
    assert output_owner.released is True
    assert runner._leased_result_partition_refs == {}


def test_get_next_partition_rejects_result_with_released_output_lease():
    cls, runner = _make_local_query_driver_actor()
    plan_id = "plan-released-lease"
    _bind_test_query_resource_owner(runner, plan_id)
    payload = object()
    transition_calls = []
    release_calls = []

    class _ReleasedOutputLeaseOwner:
        def transition_to(self, state):
            transition_calls.append(state)
            return False

        def release(self):
            release_calls.append("release")
            return False

    fragment = vane.ray_cxx.RayResultPartitionRef(payload, 7, 99, _ReleasedOutputLeaseOwner())
    runner.curr_streams[plan_id] = _DummyStream([fragment])
    runner.curr_plans[plan_id] = object()

    with pytest.raises(RuntimeError, match="released before external publication"):
        asyncio.run(
            cls.get_next_partition(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
                plan_id,
            )
        )

    assert transition_calls == ["external_consumer"]
    assert release_calls == ["release"]
    assert runner._leased_result_partition_refs == {}


def test_get_next_partition_releases_by_lease_id_not_object_ref(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "plan-lease-id"
    _bind_test_query_resource_owner(runner, plan_id)
    payload = object()
    output_owner = _FakeOutputLeaseOwner()
    fragment = vane.ray_cxx.RayResultPartitionRef(payload, 7, 99, output_owner)
    runner.curr_streams[plan_id] = _DummyStream([fragment])
    runner.curr_plans[plan_id] = object()

    released = []

    class _FakeRemoteMethod:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args):
            return self._fn(*args)

    class _FakeReleaseOwner:
        def __init__(self):
            self.release_result_partition_ref = _FakeRemoteMethod(self._release)

        def _release(self, owner_id, session_id, owner_plan_id, release_token):
            assert owner_id == _TEST_RUNTIME_OWNER_ID
            assert session_id == _TEST_SESSION_ID
            assert owner_plan_id == plan_id
            assert release_token is not payload
            assert isinstance(release_token, str)
            released.append(release_token)
            return cls.release_result_partition_ref(
                runner,
                owner_id,
                session_id,
                owner_plan_id,
                release_token,
            )

    class _LocalMetadataAccessor:
        def __init__(self, metadatas):
            self._metadatas = list(metadatas)

        def get_index(self, key: int):
            return self._metadatas[key]

    monkeypatch.setattr(
        PartitionMetadataAccessor,
        "from_metadata_list",
        classmethod(lambda _cls, meta: _LocalMetadataAccessor(meta)),
    )
    monkeypatch.setattr(
        partition_metadata,
        "resolve_object_refs_blocking",
        lambda value, **_kwargs: value,
    )

    result = asyncio.run(
        cls.get_next_partition(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            plan_id,
            release_owner=_FakeReleaseOwner(),
        )
    )

    assert result is not None
    assert result.partition() is payload
    assert released == ["0"]
    assert output_owner.released is True
    assert runner._leased_result_partition_refs == {}


def test_late_result_release_is_idempotent_after_plan_and_session_close():
    cls, runner = _make_local_query_driver_actor()
    plan_id = "late-result-release"
    _bind_test_plan_session(runner, plan_id, query_id="")
    runner.curr_plans[plan_id] = object()
    runner.curr_streams[plan_id] = _DummyStream([])
    release_calls = []

    class _OutputOwner:
        def release(self):
            release_calls.append("release")

    runner._leased_result_partition_refs = {
        plan_id: {
            "0": (object(), _OutputOwner()),
        }
    }
    runner._result_partition_ref_counters = {plan_id: 1}

    cls._cleanup_finished_plan(runner, plan_id)
    cls.release_result_partition_ref(
        runner,
        _TEST_RUNTIME_OWNER_ID,
        _TEST_SESSION_ID,
        plan_id,
        "0",
    )
    asyncio.run(
        cls.close_session(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
        )
    )
    cls.release_result_partition_ref(
        runner,
        _TEST_RUNTIME_OWNER_ID,
        _TEST_SESSION_ID,
        plan_id,
        "0",
    )

    assert release_calls == ["release"]
    with pytest.raises(PermissionError, match="owning runtime client"):
        cls.release_result_partition_ref(
            runner,
            "other-runtime-owner",
            _TEST_SESSION_ID,
            plan_id,
            "0",
        )


def test_result_release_failure_retains_lease_for_retry():
    cls, runner = _make_local_query_driver_actor()
    plan_id = "retry-result-release"
    _bind_test_plan_session(runner, plan_id, query_id="")
    attempts = []

    class _OutputOwner:
        def release(self):
            attempts.append("release")
            if len(attempts) == 1:
                raise RuntimeError("transient release failure")

    record = (object(), _OutputOwner())
    runner._leased_result_partition_refs = {plan_id: {"0": record}}

    with pytest.raises(RuntimeError, match="transient release failure"):
        cls.release_result_partition_ref(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            plan_id,
            "0",
        )

    assert runner._leased_result_partition_refs[plan_id]["0"] is record

    cls.release_result_partition_ref(
        runner,
        _TEST_RUNTIME_OWNER_ID,
        _TEST_SESSION_ID,
        plan_id,
        "0",
    )

    assert attempts == ["release", "release"]
    assert runner._leased_result_partition_refs == {}


def test_get_next_partition_rejects_unleased_arrow_payload():
    pa = pytest.importorskip("pyarrow")
    cls, runner = _make_local_query_driver_actor()
    plan_id = "plan-arrow"
    _bind_test_query_resource_owner(runner, plan_id)
    table = pa.table({"x": [1, 2, 3]})
    runner.curr_streams[plan_id] = _DummyStream([table])
    runner.curr_plans[plan_id] = object()

    with pytest.raises(TypeError, match="expected metadata-aware fragment"):
        asyncio.run(
            cls.get_next_partition(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
                plan_id,
            )
        )


def test_get_next_partition_rejects_non_metadata_aware_fragment():
    cls, runner = _make_local_query_driver_actor()
    plan_id = "plan-bad"
    _bind_test_query_resource_owner(runner, plan_id)
    runner.curr_streams[plan_id] = _DummyStream([{"rows": 1}])
    runner.curr_plans[plan_id] = object()

    with pytest.raises(TypeError, match="expected metadata-aware fragment"):
        asyncio.run(
            cls.get_next_partition(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
                plan_id,
            )
        )


def test_get_next_partition_surfaces_late_actor_initialization_failure_before_delivery():
    cls, runner = _make_local_query_driver_actor()
    plan_id = "plan-placement-lost"
    query_id = "query-placement-lost"
    _bind_test_query_resource_owner(runner, plan_id, query_id=query_id)
    undelivered = object()
    runner.curr_streams[plan_id] = _DummyStream([undelivered])
    runner.curr_plans[plan_id] = object()
    runner._query_terminal_errors[query_id] = "Ray actor UDF pool initialization failed"
    teardown_calls = []

    def _teardown(actual_plan_id):
        _, actual_query_id, _query_connection, drop_fragments = _test_plan_teardown_state(runner, actual_plan_id)
        teardown_calls.append((actual_plan_id, actual_query_id, drop_fragments))
        runner._query_terminal_errors.pop(actual_query_id, None)

    runner._teardown_plan_resources = _teardown

    with pytest.raises(RuntimeError, match="Ray actor UDF pool initialization failed"):
        asyncio.run(
            cls.get_next_partition(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
                plan_id,
            )
        )

    assert teardown_calls == [(plan_id, query_id, True)]
    assert runner.curr_streams[plan_id].items == [undelivered]
    from vane.runners.ray.query_resource_runtime import release_query_resource_manager

    release_query_resource_manager(query_id, reason="test_complete")


def test_terminal_query_failure_preserves_primary_when_teardown_also_fails():
    cls, runner = _make_local_query_driver_actor()
    plan_id = "terminal-cleanup-plan"
    query_id = "terminal-cleanup-query"
    runner._query_terminal_errors[query_id] = "fixed actor placement was lost"
    cleanup_error = RuntimeError("planned terminal teardown failure")
    runner._teardown_plan_resources = lambda *_args, **_kwargs: (_ for _ in ()).throw(cleanup_error)

    with pytest.raises(driver.QueryExecutionCleanupError) as error:
        cls._finish_terminal_query(runner, plan_id, query_id)

    assert isinstance(error.value.primary_error, RuntimeError)
    assert str(error.value.primary_error) == "fixed actor placement was lost"
    assert error.value.cleanup_errors == (cleanup_error,)
    assert error.value.__cause__ is error.value.primary_error


def test_get_next_partition_waits_for_teardown_without_blocking_event_loop():
    cls, runner = _make_local_query_driver_actor()
    plan_id = "plan-end"
    _bind_test_query_resource_owner(runner, plan_id, query_id="query-end")
    runner.curr_streams[plan_id] = _DummyStream([])
    runner.curr_plans[plan_id] = object()

    cleanup_started = threading.Event()
    cleanup_release = threading.Event()
    cleanup_calls = []

    def _slow_drop_query_fragments(query_id: str) -> None:
        cleanup_calls.append(query_id)
        cleanup_started.set()
        cleanup_release.wait(timeout=1.0)

    runner._drop_query_fragments_sync = _slow_drop_query_fragments

    async def _consume_to_completion():
        consume_task = asyncio.create_task(
            cls.get_next_partition(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
                plan_id,
            )
        )
        assert await asyncio.to_thread(cleanup_started.wait, 1.0)
        assert consume_task.done() is False

        # Cleanup runs on a worker thread: the driver event loop stays responsive,
        # while query completion remains fenced on deterministic teardown.
        await asyncio.sleep(0)
        assert consume_task.done() is False
        cleanup_release.set()
        return await asyncio.wait_for(consume_task, timeout=1.0)

    try:
        result = asyncio.run(_consume_to_completion())

        assert result is None
        assert cleanup_calls == ["query-end"]
        assert plan_id not in runner.curr_streams
        assert plan_id not in runner.curr_plans
        assert plan_id not in runner._plan_query_ids
    finally:
        cleanup_release.set()


def test_close_plan_runs_blocking_teardown_off_actor_event_loop():
    cls, runner = _make_local_query_driver_actor()
    cleanup_threads = []
    runner._sessions[_TEST_SESSION_ID].plan_ids.add("plan-close")
    runner._plan_session_ids["plan-close"] = _TEST_SESSION_ID

    def _cleanup(plan_id):
        assert plan_id == "plan-close"
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        cleanup_threads.append(threading.current_thread().name)

    runner._cleanup_finished_plan = _cleanup

    async def _close():
        await cls.close_plan(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            "plan-close",
        )

    asyncio.run(_close())

    assert len(cleanup_threads) == 1
    assert cleanup_threads[0].startswith("vane-driver-lifecycle")


def test_fragment_stats_runs_worker_observation_off_actor_event_loop():
    cls, runner = _make_local_query_driver_actor()
    stats_started = threading.Event()
    stats_release = threading.Event()

    class _PlanRunner:
        def fragment_stats(self):
            with pytest.raises(RuntimeError, match="no running event loop"):
                asyncio.get_running_loop()
            stats_started.set()
            stats_release.wait(timeout=1.0)
            return {"fragment_count": 3}

    runner._get_plan_runner = lambda: _PlanRunner()

    async def _observe_without_blocking():
        heartbeat = asyncio.Event()
        asyncio.get_running_loop().call_soon(heartbeat.set)
        task = asyncio.create_task(cls.fragment_stats(runner))
        assert await asyncio.to_thread(stats_started.wait, 1.0)
        await asyncio.wait_for(heartbeat.wait(), timeout=0.1)
        assert task.done() is False
        stats_release.set()
        return await asyncio.wait_for(task, timeout=1.0)

    try:
        stats = asyncio.run(_observe_without_blocking())
    finally:
        stats_release.set()

    assert stats == {"fragment_count": 3}


def test_progress_snapshot_build_runs_off_actor_event_loop(monkeypatch):
    from vane.runners.ray import fte_fragment_scheduler

    cls, runner = _make_local_query_driver_actor()
    _bind_test_plan_session(runner, "query-progress")
    build_started = threading.Event()
    build_release = threading.Event()

    def _slow_registry_snapshot(query_id):
        assert query_id == "query-progress"
        build_started.set()
        build_release.wait(timeout=1.0)
        return {
            "queries": {
                query_id: {
                    "query_id": query_id,
                    "fragment_executions": {},
                }
            }
        }

    monkeypatch.setattr(
        fte_fragment_scheduler,
        "fte_progress_registry_snapshot",
        _slow_registry_snapshot,
    )
    monkeypatch.setattr(
        fte_fragment_scheduler,
        "fte_registry_stats",
        lambda: (_ for _ in ()).throw(AssertionError("hot progress path must not use the diagnostic registry dump")),
    )

    async def _snapshot_without_blocking():
        heartbeat = asyncio.Event()
        asyncio.get_running_loop().call_soon(heartbeat.set)
        watchdog = threading.Timer(0.5, build_release.set)
        watchdog.start()
        try:
            started_at = time.monotonic()
            progress_call = cls.progress_snapshot(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
                "query-progress",
                0.0,
            )
            call_elapsed = time.monotonic() - started_at

            assert call_elapsed < 0.05
            assert hasattr(progress_call, "__await__")
            task = asyncio.create_task(progress_call)
            assert await asyncio.to_thread(build_started.wait, 1.0)
            await asyncio.wait_for(heartbeat.wait(), timeout=0.1)
            assert task.done() is False
            build_release.set()
            return await asyncio.wait_for(task, timeout=1.0)
        finally:
            build_release.set()
            watchdog.cancel()

    snapshot = asyncio.run(_snapshot_without_blocking())

    assert snapshot["query_id"] == "query-progress"


def test_progress_snapshot_returns_cached_value_while_refresh_runs():
    cls, runner = _make_local_query_driver_actor()
    _bind_test_plan_session(runner, "query-cache-progress")
    refresh_started = threading.Event()
    refresh_release = threading.Event()
    build_count = 0

    def _snapshot(query_id, _started_at):
        nonlocal build_count
        build_count += 1
        if build_count == 1:
            return {"query_id": query_id, "version": 1}
        refresh_started.set()
        refresh_release.wait(timeout=1.0)
        return {"query_id": query_id, "version": 2}

    runner._build_local_progress_snapshot = _snapshot

    async def _read_cached_during_refresh():
        first = await cls.progress_snapshot(
            runner,
            _TEST_RUNTIME_OWNER_ID,
            _TEST_SESSION_ID,
            "query-cache-progress",
            0.0,
        )
        await asyncio.sleep(0)
        second_call = asyncio.create_task(
            cls.progress_snapshot(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
                "query-cache-progress",
                0.0,
            )
        )
        assert await asyncio.to_thread(refresh_started.wait, 1.0)
        second = await asyncio.wait_for(second_call, timeout=0.1)
        assert runner._progress_snapshot_builds
        refresh_release.set()
        await asyncio.sleep(0)
        return first, second

    try:
        first, second = asyncio.run(_read_cached_during_refresh())
    finally:
        refresh_release.set()

    assert first == {"query_id": "query-cache-progress", "version": 1}
    assert second == first


def test_progress_snapshot_state_is_cancelled_and_dropped_with_query():
    cls, runner = _make_local_query_driver_actor()
    _bind_test_plan_session(runner, "query-drop-progress")
    build_started = threading.Event()
    build_release = threading.Event()

    def _slow_snapshot(query_id, _started_at):
        build_started.set()
        build_release.wait(timeout=1.0)
        return {"query_id": query_id}

    runner._build_local_progress_snapshot = _slow_snapshot

    async def _drop_active_snapshot():
        progress = asyncio.create_task(
            cls.progress_snapshot(
                runner,
                _TEST_RUNTIME_OWNER_ID,
                _TEST_SESSION_ID,
                "query-drop-progress",
                0.0,
            )
        )
        assert await asyncio.to_thread(build_started.wait, 1.0)
        cls._drop_progress_snapshot_state(runner, "query-drop-progress")
        with pytest.raises(asyncio.CancelledError):
            await progress
        build_release.set()
        await asyncio.sleep(0)

    try:
        asyncio.run(_drop_active_snapshot())
    finally:
        build_release.set()

    assert runner._progress_snapshot_builds == {}
    assert runner._progress_snapshot_cache == {}


def test_execute_native_empty_result_returns_typed_contract():
    con = vane.connect()
    con.execute("CREATE TABLE a AS SELECT i FROM range(10) tbl(i)")
    relation = con.sql("SELECT * FROM a WHERE 1=0")
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    cursor = con.cursor()
    result = runner.execute_native(cursor, plan, None, None)

    assert isinstance(result, vane.ray_cxx.NativeDistributedTaskResult)
    assert result.completion_status == "empty"
    assert list(result.partition_payloads) == []
    assert list(result.partition_metadatas) == []
    assert result.result_schema["types"] == ["BIGINT"]


def test_describe_native_progress_materializes_deferred_clone_without_execution(tmp_path):
    import ray

    con = vane.connect()
    src = tmp_path / "progress_topology_input.parquet"
    con.execute(f"COPY (SELECT i::INTEGER AS i FROM range(10) tbl(i)) TO '{src}' (FORMAT PARQUET)")
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql(f"SELECT i * 2 AS value FROM read_parquet('{src}')"),
        str(uuid.uuid4()),
    ).to_physical_plan(con)
    deferred = ray.cloudpickle.loads(ray.cloudpickle.dumps(plan))
    assert deferred.has_root() is False

    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    topology = vane.ray_cxx.describe_native_progress(con.cursor(), deferred)

    assert deferred.has_root() is False
    assert set(topology) == {"schema", "pipelines"}
    assert topology["schema"] == "pipeline_topology"
    assert topology["pipelines"]
    assert any("TABLE_SCAN" in pipeline["operators"] for pipeline in topology["pipelines"])
    assert all(set(pipeline) == {"pipeline_id", "operators", "operator_details"} for pipeline in topology["pipelines"])
    result_collector_roles = {
        pipeline["operator_details"][index].get("pipeline_role")
        for pipeline in topology["pipelines"]
        for index, operator in enumerate(pipeline["operators"])
        if operator == "RESULT_COLLECTOR"
    }
    assert result_collector_roles == {"source", "sink"}

    result = runner.execute_native(con.cursor(), deferred, None, None)
    assert result.completion_status == "ok"
    assert sum(metadata.num_rows for metadata in result.partition_metadatas) == 10
    final_pipelines = result.task_stats["pipelines"]
    assert [(pipeline["pipeline_id"], pipeline["operators"]) for pipeline in final_pipelines] == [
        (pipeline["pipeline_id"], pipeline["operators"]) for pipeline in topology["pipelines"]
    ]
    assert all(
        pipeline["total_pipeline_tasks"] > 0
        and pipeline["completed_pipeline_tasks"] == pipeline["total_pipeline_tasks"]
        and pipeline["queued_pipeline_tasks"] == 0
        and pipeline["running_pipeline_tasks"] == 0
        for pipeline in final_pipelines
    )
    scan_pipeline = next(pipeline for pipeline in final_pipelines if "TABLE_SCAN" in pipeline["operators"])
    assert scan_pipeline["input_rows"] == 10


def test_remote_exchange_sink_accepts_nested_query_id_without_exposing_result_collector(
    tmp_path,
    monkeypatch,
    request,
):
    import vane.runners.ray.worker_handle as ray_worker_handle

    vane.ray_cxx.shutdown_local_flight_service()
    request.addfinalizer(vane.ray_cxx.shutdown_local_flight_service)
    monkeypatch.setenv("VANE_FLIGHT_BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("VANE_FLIGHT_ADVERTISE_HOST", "127.0.0.1")

    class _CapturingWorker:
        def __init__(self):
            self.tasks = []

        def submit_tasks(self, tasks):
            self.tasks.extend(tasks)
            return []

        def fte_query_status(self, _query_id, _task_contexts=None):
            return {
                "failed": False,
                "finished": True,
                "matched": True,
                "selected_attempt_task_ids": [],
            }

        def pop_fte_result_handles(self, _query_id):
            return []

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_prepare_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def fte_cleanup_query(self, _query_id):
            return {}

        def task_input_stream_exhausted_for_query(self, _query_id, _source_node_ids):
            return []

        def prepare_shutdown(self):
            return None

        def finish_shutdown(self):
            return None

        def abort_shutdown(self):
            return None

    worker = _CapturingWorker()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids, _manager_instance_id: [
            vane.ray_cxx.RayWorkerRuntime("node-a", worker, 4.0, 0.0, 8 << 30)
        ],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    con = vane.connect()
    src = tmp_path / "remote_exchange_progress.parquet"
    con.sql("SELECT i::INTEGER AS i FROM range(32) tbl(i)").write_parquet(str(src))
    relation = con.read_parquet(str(src)).repartition(2)
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()

    sink_topologies = []
    sink_results = []
    with _registered_low_level_plan(plan, con, node_id="node-a"):
        stream = runner.run_plan(plan, con)
        assert collect_result_stream(stream) == []

        for task in worker.tasks:
            task_plan = task.plan()
            topology = vane.ray_cxx.describe_native_progress(con.cursor(), task_plan)
            operators = [operator for pipeline in topology["pipelines"] for operator in pipeline["operators"]]
            if "EXCHANGE_SINK" in operators:
                native_sink_instance = task.exchange_sink_instance()
                sink_instance = ExchangeSinkInstanceHandle(
                    ExchangeSinkHandle(
                        native_sink_instance["query_id"],
                        "nested-query-id-regression",
                        native_sink_instance["partition_id"],
                    ),
                    native_sink_instance["attempt_id"],
                    native_sink_instance.get("output_location"),
                ).to_dict()
                sink_instance["output_partition_count"] = native_sink_instance["output_partition_count"]
                assert "query_id" not in sink_instance
                assert sink_instance["sink_handle"]["query_id"] == native_sink_instance["query_id"]
                sink_topologies.append(topology)
                sink_results.append(
                    runner.execute_native(
                        con.cursor(),
                        task_plan,
                        exchange_sink_instance=sink_instance,
                    )
                )

    assert sink_topologies
    assert len(sink_results) == len(sink_topologies)
    assert all(result.flight_port > 0 for result in sink_results)
    assert all(result.completion_status == "executed" for result in sink_results)
    ordinary_result = runner.execute_native(con.cursor(), _make_test_physical_plan(con), None, None)
    assert ordinary_result.flight_port == 0
    assert ordinary_result.exchange_sink_instance is None
    assert all(
        "RESULT_COLLECTOR" not in pipeline["operators"]
        for topology in sink_topologies
        for pipeline in topology["pipelines"]
    )
    for topology, result in zip(sink_topologies, sink_results, strict=True):
        final_pipelines = result.task_stats["pipelines"]
        assert [(pipeline["pipeline_id"], pipeline["operators"]) for pipeline in final_pipelines] == [
            (pipeline["pipeline_id"], pipeline["operators"]) for pipeline in topology["pipelines"]
        ]
        assert sum(pipeline["input_rows"] for pipeline in final_pipelines) == 32, final_pipelines
        assert all(
            pipeline["total_pipeline_tasks"] > 0
            and pipeline["completed_pipeline_tasks"] == pipeline["total_pipeline_tasks"]
            and pipeline["queued_pipeline_tasks"] == 0
            and pipeline["running_pipeline_tasks"] == 0
            for pipeline in final_pipelines
        )
    assert all(
        pipeline["operators"] != ["EXCHANGE_SINK"] for topology in sink_topologies for pipeline in topology["pipelines"]
    )


def test_distributed_physical_plan_clone_executes_on_worker_connection():
    driver_con = vane.connect()
    worker_con = vane.connect()
    relation = driver_con.sql("SELECT 42::INTEGER AS i")
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(driver_con)

    clone = plan.clone(worker_con)

    assert clone is not plan
    assert clone.idx() == plan.idx()
    assert clone.has_root() is True
    assert plan.has_root() is True

    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    clone_result = runner.execute_native(worker_con.cursor(), clone, None, None)
    original_result = runner.execute_native(driver_con.cursor(), plan, None, None)

    assert clone_result.completion_status == "ok"
    assert original_result.completion_status == "ok"
    assert list(clone_result.partition_payloads)[0].column(0).to_pylist() == [42]
    assert list(original_result.partition_payloads)[0].column(0).to_pylist() == [42]


def test_execute_native_repartition_uses_local_exchange_not_passthrough():
    con = vane.connect()
    con.execute("SET threads=4")
    relation = con.sql("SELECT i::INTEGER AS i FROM range(32) tbl(i)").repartition(4)
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    result = runner.execute_native(con.cursor(), plan, None, None)

    assert result.completion_status == "ok"
    payloads = list(result.partition_payloads)
    assert len(payloads) == 1
    values = payloads[0].column(0).to_pylist()
    assert sorted(values) == list(range(32))
    assert values != list(range(32))
    pipelines = list(result.task_stats.get("pipelines") or [])
    assert any("REPARTITION" in list(pipeline.get("operators") or []) for pipeline in pipelines)


def test_execute_native_hash_repartition_uses_resolved_partition_expressions():
    con = vane.connect()
    con.execute("SET threads=4")
    relation = con.sql("SELECT i::INTEGER AS i, (i % 2)::INTEGER AS k FROM range(1000) tbl(i)").repartition(4, "k")
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    result = runner.execute_native(con.cursor(), plan, None, None)

    assert result.completion_status == "ok"
    metadatas = list(result.partition_metadatas)
    assert sum(metadata.num_rows for metadata in metadatas) == 1000
    pipelines = list(result.task_stats.get("pipelines") or [])
    assert any("REPARTITION" in list(pipeline.get("operators") or []) for pipeline in pipelines)


def test_execute_native_applies_dynamic_filter_domains_to_table_scan(tmp_path):
    pa = pytest.importorskip("pyarrow")

    con = vane.connect()
    src = tmp_path / "dynamic_filter_input.parquet"
    con.sql(
        """
        select i::integer as id, (i * 10)::integer as value
        from range(0, 8) tbl(i)
        """
    ).write_parquet(str(src))
    relation = con.sql(f"select id, value from read_parquet('{src}') order by id")
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    result = runner.execute_native(
        con.cursor(),
        plan,
        dynamic_filter_domains={"df0": {"column": "id", "range": [2, 4]}},
    )

    assert isinstance(result, vane.ray_cxx.NativeDistributedTaskResult)
    payloads = list(result.partition_payloads)
    assert len(payloads) == 1
    table = payloads[0]
    assert isinstance(table, pa.Table)
    assert table.column(0).to_pylist() == [2, 3, 4]
    assert table.column(1).to_pylist() == [20, 30, 40]


def test_execute_native_rejects_invalid_positional_exchange_sink_instance():
    con = vane.connect()
    cursor = con.cursor()
    plan = _make_test_physical_plan(con)
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()

    with pytest.raises(ValueError, match="exchange_sink_instance must be bytes or dict"):
        runner.execute_native(cursor, plan, None, None, None, [object()])


def test_execute_native_rejects_legacy_copy_output_string():
    con = vane.connect()
    cursor = con.cursor()
    plan = _make_test_physical_plan(con)
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()

    with pytest.raises(ValueError, match="copy_output_info must be a dict"):
        runner.execute_native(cursor, plan, None, None, "/tmp/out")


@pytest.mark.usefixtures("ray_local")
def test_run_plan_uses_distributed_worker_path(tmp_path):
    pa = pytest.importorskip("pyarrow")
    ray = pytest.importorskip("ray")

    con = vane.connect()
    src = tmp_path / "scan_typed_input.parquet"
    con.sql("select 1 as x union all select 2 as x union all select 3 as x").write_parquet(str(src))

    relation = con.sql(f"select * from read_parquet('{src}')")
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    scan_task_descriptors = dict(plan.scan_task_descriptor_map())
    assert scan_task_descriptors

    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    with _registered_low_level_plan(plan, con):
        parts = collect_result_stream(runner.run_plan(plan, con))

        assert len(parts) == 1
        assert isinstance(parts[0], vane.ray_cxx.RayResultPartitionRef)
        payload = ray.get(parts[0].object_ref)
    assert isinstance(payload, pa.Table)
    assert payload.to_pylist() == [{"c0": 1}, {"c0": 2}, {"c0": 3}]
    con.close()


@pytest.mark.timeout(60)
@pytest.mark.usefixtures("ray_local")
def test_run_plan_continues_with_final_tasks_after_order_by_barrier(tmp_path):
    pa = pytest.importorskip("pyarrow")
    ray = pytest.importorskip("ray")

    con = vane.connect()
    src = tmp_path / "barrier_order_input.parquet"
    con.sql("SELECT i::BIGINT AS x FROM range(32) tbl(i)").write_parquet(str(src))
    relation = con.read_parquet(str(src)).repartition(4).order("x DESC")
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()

    with _registered_low_level_plan(
        plan,
        con,
        refresh_phase_allocation=True,
    ) as graph:
        assert len(graph.materialization_barriers) == 1
        parts = collect_result_stream(runner.run_plan(plan, con))
        tables = ray.get([part.object_ref for part in parts])

    assert all(isinstance(table, pa.Table) for table in tables)
    assert [value for table in tables for value in table.column(0).to_pylist()] == list(reversed(range(32)))
    con.close()


@pytest.mark.usefixtures("ray_local")
def test_run_copy_plan_uses_distributed_worker_path(tmp_path, monkeypatch):
    captured = []
    monkeypatch.delenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", raising=False)

    class _DummyRunner:
        def run_write(self, relation):
            captured.append(relation)
            return {"ok": True}

    import vane.runners as runners_mod

    monkeypatch.setattr(runners_mod, "set_runner_ray", lambda *_args, **_kwargs: _DummyRunner())

    con = vane.connect()
    src = tmp_path / "copy_scan_typed_input.parquet"
    dst = tmp_path / "copy_scan_typed_output.parquet"
    con.sql("select 1 as x union all select 2 as x union all select 3 as x").write_parquet(str(src))

    monkeypatch.setenv("VANE_RUNNER", "ray")
    con.sql(f"select * from read_parquet('{src}')").write_parquet(str(dst))

    assert captured, "expected write relation to be captured"

    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        captured[0],
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    scan_task_descriptors = dict(plan.scan_task_descriptor_map())
    assert scan_task_descriptors

    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    execution_started = threading.Event()
    with _registered_low_level_plan(plan, con):
        result = runner.run_copy_plan(plan, con, execution_started.set)

    assert execution_started.is_set()
    assert result["rows_copied"] == 3
    assert result["copy_output_base_path"] == str(dst)
    assert result["copy_output_run_id"]
    assert result["copy_output_direct_write"] is True
    assert result["copy_output_committed"] is True
    assert result["copy_runner_cleanup_pending"] is False
    assert result["copy_runner_cleanup_warnings"] == []
    assert Path(result["copy_output_commit_dir"]).is_dir()
    assert Path(result["copy_output_lifecycle_path"]).is_file()
    assert Path(result["copy_output_manifest_path"]).is_file()
    assert Path(result["copy_output_committed_marker_path"]).is_file()
    assert result["copy_output_manifest_path"].endswith(
        f"{dst.name}.duckdb_commit/{result['copy_output_run_id']}/manifest.txt"
    )
    committed = vane.ray_cxx.read_committed_copy_direct_write_result(
        result["copy_output_base_path"],
        result["copy_output_run_id"],
    )
    committed_paths = [entry["final_path"] for entry in committed["files"]]
    assert committed["rows_copied"] == 3
    assert committed_paths
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    assert sorted(row[0] for row in con.read_parquet(committed_paths).fetchall()) == [1, 2, 3]
    assert not Path(str(dst) + ".duckdb_staging").exists()
    con.close()


@pytest.mark.usefixtures("ray_local")
def test_run_copy_plan_trailing_separator_uses_one_lifecycle_namespace(tmp_path, monkeypatch):
    captured = []
    monkeypatch.delenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", raising=False)

    class _DummyRunner:
        def run_write(self, relation):
            captured.append(relation)
            return {"ok": True}

    import vane.runners as runners_mod

    monkeypatch.setattr(runners_mod, "set_runner_ray", lambda *_args, **_kwargs: _DummyRunner())

    con = vane.connect()
    src = tmp_path / "copy_trailing_separator_input.parquet"
    dst = tmp_path / "copy_trailing_separator_output"
    dst.mkdir()
    raw_dst = str(dst) + os.sep
    con.sql("select 1 as x union all select 2 as x").write_parquet(str(src))

    monkeypatch.setenv("VANE_RUNNER", "ray")
    con.sql(f"select * from read_parquet('{src}')").write_parquet(raw_dst)
    assert captured, "expected write relation to be captured"

    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        captured[0],
        str(uuid.uuid4()),
    ).to_physical_plan(con)
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    with _registered_low_level_plan(plan, con):
        result = runner.run_copy_plan(plan, con)

    assert result["copy_output_base_path"] == str(dst)
    assert result["copy_output_committed"] is True
    assert Path(result["copy_output_lifecycle_path"]).parent == Path(result["copy_output_commit_dir"])
    assert Path(result["copy_output_committed_marker_path"]).parent == Path(result["copy_output_commit_dir"])
    committed_paths = [Path(entry["final_path"]) for entry in result["files"]]
    assert committed_paths
    assert all(path.is_file() for path in committed_paths)

    cleanup = vane.ray_cxx.cleanup_expired_copy_direct_write_runs(
        raw_dst,
        min_age_ms=0,
    )

    assert cleanup["scanned_runs"] == 1
    assert cleanup["cleaned_runs"] == 0
    assert cleanup["committed_runs"] == 1
    assert all(path.is_file() for path in committed_paths)
    con.close()


@pytest.mark.usefixtures("ray_local")
def test_run_copy_plan_existing_file_uses_final_lifecycle_namespace(tmp_path, monkeypatch):
    captured = []
    monkeypatch.delenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", raising=False)

    class _DummyRunner:
        def run_write(self, relation):
            captured.append(relation)
            return {"ok": True}

    import vane.runners as runners_mod

    monkeypatch.setattr(runners_mod, "set_runner_ray", lambda *_args, **_kwargs: _DummyRunner())

    con = vane.connect()
    src = tmp_path / "copy_tmp_file_input.parquet"
    dst = tmp_path / "copy_tmp_file_output.parquet"
    con.sql("select 1 as x union all select 2 as x").write_parquet(str(src))
    con.sql("select 0 as x").write_parquet(str(dst))

    monkeypatch.setenv("VANE_RUNNER", "ray")
    con.sql(f"select * from read_parquet('{src}')").write_parquet(str(dst))
    assert captured, "expected write relation to be captured"

    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        captured[0],
        str(uuid.uuid4()),
    ).to_physical_plan(con)
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    with _registered_low_level_plan(plan, con):
        result = runner.run_copy_plan(plan, con)

    assert result["copy_output_base_path"] == str(dst)
    assert result["copy_output_direct_write"] is True
    assert result["copy_output_committed"] is True
    assert Path(result["copy_output_lifecycle_path"]).parent == Path(result["copy_output_commit_dir"])
    assert Path(result["copy_output_committed_marker_path"]).parent == Path(result["copy_output_commit_dir"])
    assert not (tmp_path / f"tmp_{dst.name}.duckdb_commit").exists()
    committed_paths = [Path(entry["final_path"]) for entry in result["files"]]
    assert committed_paths
    assert all(path.is_file() for path in committed_paths)
    temporary_base = tmp_path / f"tmp_{dst.name}"
    assert all(path.parent == temporary_base for path in committed_paths)
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    assert sorted(row[0] for row in con.read_parquet([str(path) for path in committed_paths]).fetchall()) == [1, 2]
    assert con.read_parquet(str(dst)).fetchall() == [(0,)]

    cleanup = vane.ray_cxx.cleanup_expired_copy_direct_write_runs(
        str(dst),
        min_age_ms=0,
    )

    assert cleanup["scanned_runs"] == 1
    assert cleanup["cleaned_runs"] == 0
    assert cleanup["committed_runs"] == 1
    assert all(path.is_file() for path in committed_paths)
    assert not Path(str(temporary_base) + ".duckdb_staging").exists()
    con.close()


@pytest.mark.usefixtures("ray_local")
def test_run_copy_plan_leaves_stale_direct_write_cleanup_to_explicit_api(tmp_path, monkeypatch):
    captured = []
    monkeypatch.delenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", raising=False)

    class _DummyRunner:
        def run_write(self, relation):
            captured.append(relation)
            return {"ok": True}

    import vane.runners as runners_mod
    from vane.runners.ray import cleanup_copy_direct_write_lifecycle_once

    monkeypatch.setattr(runners_mod, "set_runner_ray", lambda *_args, **_kwargs: _DummyRunner())

    con = vane.connect()
    src = tmp_path / "copy_explicit_cleanup_input.parquet"
    dst = tmp_path / "copy_explicit_cleanup_output"
    dst.mkdir()
    con.sql("select 1 as x union all select 2 as x").write_parquet(str(src))

    monkeypatch.setenv("VANE_RUNNER", "ray")
    con.sql(f"select * from read_parquet('{src}')").write_parquet(str(dst))
    assert captured, "expected write relation to be captured"

    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        captured[0],
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    stale_run_id = "run-explicit-cleanup"
    stale_lifecycle = vane.ray_cxx.register_copy_direct_write_run_lifecycle(
        str(dst),
        stale_run_id,
        created_epoch_ms=1,
    )
    stale_file = dst / f"_vane_direct_write_{stale_run_id}" / "w_failed" / "part.parquet"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_bytes(b"stale")

    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    with _registered_low_level_plan(plan, con):
        result = runner.run_copy_plan(plan, con)

    assert result["copy_output_committed"] is True
    assert stale_file.is_file()
    assert Path(stale_lifecycle["copy_output_lifecycle_path"]).is_file()

    cleanup = cleanup_copy_direct_write_lifecycle_once(
        str(dst),
        min_age_ms=1,
    )

    assert cleanup["scanned_runs"] == 2
    assert cleanup["cleaned_runs"] == 1
    assert cleanup["committed_runs"] == 1
    assert cleanup["cleaned_run_ids"] == [{"base_path": str(dst), "run_id": stale_run_id}]
    assert not stale_file.exists()
    assert not Path(stale_lifecycle["copy_output_commit_dir"]).exists()
    assert Path(result["copy_output_committed_marker_path"]).is_file()
    con.close()


@pytest.mark.usefixtures("ray_local")
def test_run_copy_plan_local_staging_env_preserves_rename_path(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", "1")

    class _DummyRunner:
        def run_write(self, relation):
            captured.append(relation)
            return {"ok": True}

    import vane.runners as runners_mod

    monkeypatch.setattr(runners_mod, "set_runner_ray", lambda *_args, **_kwargs: _DummyRunner())

    con = vane.connect()
    src = tmp_path / "copy_staging_input.parquet"
    dst = tmp_path / "copy_staging_output.parquet"
    con.sql("select 10 as x union all select 20 as x").write_parquet(str(src))

    monkeypatch.setenv("VANE_RUNNER", "ray")
    con.sql(f"select * from read_parquet('{src}')").write_parquet(str(dst))

    assert captured, "expected write relation to be captured"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        captured[0],
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    with _registered_low_level_plan(plan, con):
        result = runner.run_copy_plan(plan, con)

    assert result["rows_copied"] == 2
    assert result["copy_output_base_path"] == str(dst)
    assert result["copy_output_direct_write"] is False
    assert result["copy_output_committed"] is True
    assert dst.exists()
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    assert sorted(row[0] for row in con.sql(f"select * from read_parquet('{dst}')").fetchall()) == [10, 20]
    assert not Path(str(dst) + ".duckdb_staging").exists()
    con.close()


@pytest.mark.usefixtures("ray_local")
def test_run_copy_plan_with_fte_preserves_copy_sink_output_for_existing_dir(tmp_path, monkeypatch):
    captured = []
    monkeypatch.delenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", raising=False)

    class _DummyRunner:
        def run_write(self, relation):
            captured.append(relation)
            return {"ok": True}

    import vane.runners as runners_mod

    monkeypatch.setattr(runners_mod, "set_runner_ray", lambda *_args, **_kwargs: _DummyRunner())

    con = vane.connect()
    src = tmp_path / "copy_fte_input.parquet"
    dst = tmp_path / "copy_fte_output"
    dst.mkdir()
    con.sql("select 1 as x union all select 2 as x union all select 3 as x").write_parquet(str(src))

    monkeypatch.setenv("VANE_RUNNER", "ray")
    con.sql(f"select * from read_parquet('{src}')").write_parquet(str(dst))

    assert captured, "expected write relation to be captured"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        captured[0],
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    scan_task_descriptors = dict(plan.scan_task_descriptor_map())
    assert scan_task_descriptors

    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    with _registered_low_level_plan(plan, con):
        result = runner.run_copy_plan(plan, con)

    files = sorted(path for path in dst.rglob("*") if path.is_file())
    assert result["rows_copied"] == 3
    assert files
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    assert sum(con.sql(f"select count(*) from read_parquet('{path}')").fetchone()[0] for path in files) == 3
    assert not Path(str(dst) + ".duckdb_staging").exists()
    con.close()


@pytest.mark.usefixtures("ray_local")
def test_run_copy_plan_local_direct_write_committed_reader(tmp_path, monkeypatch):
    captured = []
    monkeypatch.delenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", raising=False)

    class _DummyRunner:
        def run_write(self, relation):
            captured.append(relation)
            return {"ok": True}

    import vane.runners as runners_mod

    monkeypatch.setattr(runners_mod, "set_runner_ray", lambda *_args, **_kwargs: _DummyRunner())

    con = vane.connect()
    src = tmp_path / "copy_direct_success_input.parquet"
    dst = tmp_path / "copy_direct_success_output"
    con.sql("select 1 as x union all select 2 as x union all select 3 as x").write_parquet(str(src))

    monkeypatch.setenv("VANE_RUNNER", "ray")
    con.sql(f"select * from read_parquet('{src}')").write_parquet(str(dst))

    assert captured, "expected write relation to be captured"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        captured[0],
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    scan_task_descriptors = dict(plan.scan_task_descriptor_map())
    assert scan_task_descriptors

    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    with _registered_low_level_plan(plan, con):
        result = runner.run_copy_plan(plan, con)

    assert result["rows_copied"] == 3
    assert result["copy_output_base_path"] == str(dst)
    assert result["copy_output_run_id"]
    assert result["copy_output_direct_write"] is True
    assert result["copy_output_committed"] is True
    assert Path(result["copy_output_lifecycle_path"]).is_file()
    assert Path(result["copy_output_manifest_path"]).is_file()
    assert Path(result["copy_output_committed_marker_path"]).is_file()
    assert not Path(str(dst) + ".duckdb_staging").exists()

    committed = vane.ray_cxx.read_committed_copy_direct_write_result(
        result["copy_output_base_path"],
        result["copy_output_run_id"],
    )
    committed_paths = [entry["final_path"] for entry in committed["files"]]
    assert committed["rows_copied"] == 3
    assert committed["copy_output_direct_write"] is True
    assert committed_paths
    assert all("_vane_direct_write_" not in path for path in committed_paths)
    assert all(Path(path).parent == dst for path in committed_paths)
    assert all(Path(path).name.startswith(f"{result['copy_output_run_id']}_") for path in committed_paths)
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    assert sum(con.sql(f"select count(*) from read_parquet('{path}')").fetchone()[0] for path in committed_paths) == 3

    loser_file = dst / f"{result['copy_output_run_id']}_w_loser_part.parquet"
    loser_file.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT 999 AS x) TO '{loser_file}' (FORMAT PARQUET)")
    all_run_files = [*committed_paths, str(loser_file)]
    assert con.read_parquet(all_run_files).aggregate("count(*)").fetchone()[0] == 4

    from vane.runners.ray import read_committed_copy_direct_write_parquet

    committed_rel = read_committed_copy_direct_write_parquet(
        result["copy_output_base_path"],
        result["copy_output_run_id"],
        conn=con,
    )
    assert sorted(row[0] for row in committed_rel.fetchall()) == [1, 2, 3]
    con.close()


def test_run_copy_plan_propagates_worker_task_failure_before_finalize(tmp_path, monkeypatch):
    class _FailingTaskHandle:
        _is_done = True
        _result = None
        _future = None
        task = None
        worker_id = "worker-fail"

        def __init__(self, message, task_id):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self._error = RuntimeError(message)

        def _ensure_started(self):
            return None

        def done(self):
            return True

        def get_result_sync(self):
            raise self._error

        def cancel(self):
            self._is_done = True

        def release_result_payload(self):
            return None

    class _FailingWorkerHandle:
        def __init__(self):
            self.submit_count = 0
            self.staging_roots = []
            self.handles_by_query = {}

        def submit_tasks(self, tasks):
            handles = []
            for task in tasks:
                self.submit_count += 1
                context = task.context()
                query_id = context["query_id"]
                staging_base = context["copy_output_base"]
                run_id = context["copy_output_run_id"]
                assert staging_base
                staging_root = Path(staging_base) / run_id
                output_file = staging_root / "w_fake" / "part.parquet"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_bytes(b"partial-copy-output")
                self.staging_roots.append(staging_root)
                task_id = {
                    "query_id": query_id,
                    "fragment_execution_id": 0,
                    "partition_id": self.submit_count - 1,
                    "attempt_id": 0,
                }
                handle = _FailingTaskHandle("planned worker failure", task_id)
                handles.append(handle)
                self.handles_by_query.setdefault(query_id, []).append(handle)
            return handles

        def fte_query_status(self, query_id):
            return {
                "failed": False,
                "finished": bool(self.handles_by_query.get(query_id)),
                "selected_attempt_task_ids": [],
            }

        def pop_fte_result_handles(self, query_id):
            return list(self.handles_by_query.pop(query_id, []))

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_prepare_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def fte_cleanup_query(self, _query_id):
            return {}

        def task_input_stream_exhausted_for_query(self, _query_id, _source_node_ids):
            return []

        def prepare_shutdown(self):
            return None

        def finish_shutdown(self):
            return None

        def abort_shutdown(self):
            return None

    import vane.runners as runners_mod
    import vane.runners.ray.worker_handle as ray_worker_handle

    captured = []

    class _CapturingRunner:
        def run_write(self, relation):
            captured.append(relation)
            return {"ok": True}

    monkeypatch.setenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", "1")
    failing_worker = _FailingWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids, _manager_instance_id: [
            vane.ray_cxx.RayWorkerRuntime("worker-fail", failing_worker, 1.0, 0.0, 1024)
        ],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    con = vane.connect()
    src = tmp_path / "copy_failure_input.parquet"
    dst = tmp_path / "copy_failure_output.parquet"
    con.sql("select 1 as x union all select 2 as x").write_parquet(str(src))

    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setattr(runners_mod, "set_runner_ray", lambda *_args, **_kwargs: _CapturingRunner())
    con.sql(f"select * from read_parquet('{src}')").write_parquet(str(dst))
    assert captured, "expected write relation to be captured"

    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        captured[0],
        str(uuid.uuid4()),
    ).to_physical_plan(con)
    scan_task_descriptors = dict(plan.scan_task_descriptor_map())
    assert scan_task_descriptors

    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    with _registered_low_level_plan(plan, con, node_id="node-a"):
        with pytest.raises(ValueError, match="planned worker failure"):
            runner.run_copy_plan(plan, con)
    assert failing_worker.submit_count >= 1
    assert not dst.exists()
    assert failing_worker.staging_roots
    for staging_root in failing_worker.staging_roots:
        assert not staging_root.exists()
    assert not Path(str(dst) + ".duckdb_staging").exists()


def test_run_copy_plan_direct_write_failure_cleans_uncommitted_run(tmp_path, monkeypatch):
    class _FailingTaskHandle:
        _is_done = True
        _result = None
        _future = None
        task = None
        worker_id = "worker-direct-fail"

        def __init__(self, message, task_id):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self._error = RuntimeError(message)

        def _ensure_started(self):
            return None

        def done(self):
            return True

        def get_result_sync(self):
            raise self._error

        def cancel(self):
            self._is_done = True

        def release_result_payload(self):
            return None

    class _FailingDirectWriteWorkerHandle:
        def __init__(self):
            self.submit_count = 0
            self.output_files = []
            self.handles_by_query = {}

        def submit_tasks(self, tasks):
            handles = []
            for task in tasks:
                self.submit_count += 1
                context = task.context()
                query_id = context["query_id"]
                assert context["copy_output_base"] == ""
                run_id = context["copy_output_run_id"]
                remote_base = context["copy_output_remote_base"]
                output_file = Path(remote_base) / f"{run_id}_w_fake_part.parquet"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_bytes(b"partial-direct-copy-output")
                self.output_files.append(output_file)
                task_id = {
                    "query_id": query_id,
                    "fragment_execution_id": 0,
                    "partition_id": self.submit_count - 1,
                    "attempt_id": 0,
                }
                handle = _FailingTaskHandle("planned direct worker failure", task_id)
                handles.append(handle)
                self.handles_by_query.setdefault(query_id, []).append(handle)
            return handles

        def fte_query_status(self, query_id):
            return {
                "failed": False,
                "finished": bool(self.handles_by_query.get(query_id)),
                "selected_attempt_task_ids": [],
            }

        def pop_fte_result_handles(self, query_id):
            return list(self.handles_by_query.pop(query_id, []))

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_prepare_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def fte_cleanup_query(self, _query_id):
            return {}

        def task_input_stream_exhausted_for_query(self, _query_id, _source_node_ids):
            return []

        def prepare_shutdown(self):
            return None

        def finish_shutdown(self):
            return None

        def abort_shutdown(self):
            return None

    import vane.runners as runners_mod
    import vane.runners.ray.worker_handle as ray_worker_handle

    captured = []

    class _CapturingRunner:
        def run_write(self, relation):
            captured.append(relation)
            return {"ok": True}

    monkeypatch.delenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", raising=False)
    failing_worker = _FailingDirectWriteWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids, _manager_instance_id: [
            vane.ray_cxx.RayWorkerRuntime("worker-direct-fail", failing_worker, 1.0, 0.0, 1024)
        ],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    con = vane.connect()
    src = tmp_path / "copy_direct_failure_input.parquet"
    dst = tmp_path / "copy_direct_failure_output.parquet"
    con.sql("select 1 as x union all select 2 as x").write_parquet(str(src))

    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setattr(runners_mod, "set_runner_ray", lambda *_args, **_kwargs: _CapturingRunner())
    con.sql(f"select * from read_parquet('{src}')").write_parquet(str(dst))
    assert captured, "expected write relation to be captured"

    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        captured[0],
        str(uuid.uuid4()),
    ).to_physical_plan(con)
    scan_task_descriptors = dict(plan.scan_task_descriptor_map())
    assert scan_task_descriptors

    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    with _registered_low_level_plan(plan, con, node_id="node-a"):
        with pytest.raises(ValueError, match="planned direct worker failure"):
            runner.run_copy_plan(plan, con)
    assert failing_worker.submit_count >= 1
    assert failing_worker.output_files
    for output_file in failing_worker.output_files:
        assert not output_file.exists()
    assert not Path(str(dst) + ".duckdb_commit").exists()


def test_wait_fte_query_propagates_status_errors(monkeypatch):
    class _StatusFailingWorkerHandle:
        def __init__(self):
            self.status_calls = 0

        def fte_query_status(self, _query_id):
            self.status_calls += 1
            raise RuntimeError("status exploded")

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_prepare_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def fte_cleanup_query(self, _query_id):
            return {}

        def prepare_shutdown(self):
            return None

        def finish_shutdown(self):
            return None

        def abort_shutdown(self):
            return None

    import vane.runners.ray.worker_handle as ray_worker_handle

    failing_worker = _StatusFailingWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids, _manager_instance_id: [
            vane.ray_cxx.RayWorkerRuntime("worker-status-fail", failing_worker, 1.0, 0.0, 1024)
        ],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = vane.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        manager.register_query_owner("query-status-error", "query-status-error")
        with pytest.raises(Exception, match="status exploded"):
            manager.wait_fte_query("query-status-error", 0.01)
        assert failing_worker.status_calls == 1
    finally:
        manager.shutdown()


def test_wait_fte_query_releases_gil_while_waiting(monkeypatch):
    class _ThreadProgressWorkerHandle:
        def __init__(self):
            self.finished = False
            self.status_calls = 0
            self.status_polled = threading.Event()
            self.finished_event = threading.Event()

        def fte_query_status(self, _query_id):
            self.status_calls += 1
            self.status_polled.set()
            return {
                "failed": False,
                "finished": self.finished,
                "selected_attempt_task_ids": [],
            }

        def pop_fte_result_handles(self, _query_id):
            return []

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_prepare_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def fte_cleanup_query(self, _query_id):
            return {}

        def prepare_shutdown(self):
            return None

        def finish_shutdown(self):
            return None

        def abort_shutdown(self):
            return None

    import vane.runners.ray.worker_handle as ray_worker_handle

    worker = _ThreadProgressWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids, _manager_instance_id: [
            vane.ray_cxx.RayWorkerRuntime("worker-gil-wait", worker, 1.0, 0.0, 1024)
        ],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    def finish_after_first_status_poll():
        assert worker.status_polled.wait(1.0)
        time.sleep(0.02)
        worker.finished = True
        worker.finished_event.set()

    thread = threading.Thread(target=finish_after_first_status_poll, daemon=True)
    thread.start()

    manager = vane.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        manager.register_query_owner("query-gil-wait", "query-gil-wait")
        manager.wait_fte_query("query-gil-wait", 1.0)
        assert worker.finished_event.is_set()
        assert worker.status_calls >= 2
    finally:
        manager.shutdown()
        thread.join(1.0)


def test_wait_fte_query_rejects_malformed_query_status(monkeypatch):
    class _MalformedStatusWorkerHandle:
        def __init__(self):
            self.status_calls = 0

        def fte_query_status(self, _query_id):
            self.status_calls += 1
            return {"finished": True}

        def pop_fte_result_handles(self, _query_id):
            return []

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def prepare_shutdown(self):
            return None

        def finish_shutdown(self):
            return None

        def abort_shutdown(self):
            return None

    import vane.runners.ray.worker_handle as ray_worker_handle

    worker = _MalformedStatusWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids, _manager_instance_id: [
            vane.ray_cxx.RayWorkerRuntime("worker-status-malformed", worker, 1.0, 0.0, 1024)
        ],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = vane.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        manager.register_query_owner("query-status-malformed", "query-status-malformed")
        with pytest.raises(Exception, match="FTE query status must include boolean 'failed'"):
            manager.wait_fte_query("query-status-malformed", 1.0)
        assert worker.status_calls == 1
    finally:
        manager.shutdown()


def test_wait_fte_query_rejects_result_handles_without_task_id(monkeypatch):
    class _MalformedHandle:
        worker_id = "worker-handle-malformed"
        task_context_info = _fake_task_context_info(
            {
                "query_id": "query-handle-malformed",
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 0,
            }
        )

        def done(self):
            return True

        def get_result_sync(self):
            return vane.ray_cxx.RayTaskResult.no_output()

    class _MalformedHandleWorker:
        def __init__(self):
            self.pop_calls = 0

        def fte_query_status(self, _query_id):
            return {
                "failed": False,
                "finished": True,
                "selected_attempt_task_ids": [],
            }

        def pop_fte_result_handles(self, _query_id):
            self.pop_calls += 1
            return [_MalformedHandle()]

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def prepare_shutdown(self):
            return None

        def finish_shutdown(self):
            return None

        def abort_shutdown(self):
            return None

    import vane.runners.ray.worker_handle as ray_worker_handle

    worker = _MalformedHandleWorker()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids, _manager_instance_id: [
            vane.ray_cxx.RayWorkerRuntime("worker-handle-malformed", worker, 1.0, 0.0, 1024)
        ],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = vane.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        manager.register_query_owner("query-handle-malformed", "query-handle-malformed")
        with pytest.raises(Exception, match="FTE result handle must provide task_id"):
            manager.wait_fte_query("query-handle-malformed", 1.0)
        assert worker.pop_calls == 1
    finally:
        manager.shutdown()


def test_wait_fte_query_rejects_result_handles_without_worker_id(monkeypatch):
    class _MalformedHandle:
        def __init__(self):
            self.task_id = _fake_task_attempt_id(
                {
                    "query_id": "query-handle-missing-worker",
                    "fragment_execution_id": 0,
                    "partition_id": 0,
                    "attempt_id": 0,
                }
            )
            self.task_context_info = _fake_task_context_info(self.task_id)

        def done(self):
            return True

        def get_result_sync(self):
            return vane.ray_cxx.RayTaskResult.no_output()

    class _MalformedHandleWorker:
        def __init__(self):
            self.pop_calls = 0

        def fte_query_status(self, query_id):
            return {
                "failed": False,
                "finished": True,
                "selected_attempt_task_ids": [f"{query_id}.0.0.0"],
            }

        def pop_fte_result_handles(self, _query_id):
            self.pop_calls += 1
            return [_MalformedHandle()]

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def prepare_shutdown(self):
            return None

        def finish_shutdown(self):
            return None

        def abort_shutdown(self):
            return None

    import vane.runners.ray.worker_handle as ray_worker_handle

    worker = _MalformedHandleWorker()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids, _manager_instance_id: [
            vane.ray_cxx.RayWorkerRuntime("worker-coordinator", worker, 1.0, 0.0, 1024)
        ],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = vane.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        manager.register_query_owner("query-handle-missing-worker", "query-handle-missing-worker")
        with pytest.raises(Exception, match="worker_id"):
            manager.wait_fte_query("query-handle-missing-worker", 1.0)
        assert worker.pop_calls == 1
    finally:
        manager.shutdown()


def test_wait_fte_query_propagates_selected_attempt_handle_errors(monkeypatch):
    class _FailedSelectedAttemptHandle:
        worker_id = "worker-selected"

        def __init__(self, task_id):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self._is_done = True
            self._result = None
            self._error = RuntimeError("selected attempt failed")
            self._future = None
            self.task = None
            self.release_calls = 0

        def _ensure_started(self):
            return None

        def done(self):
            return True

        def get_result_sync(self):
            raise RuntimeError("selected attempt failed")

        def release_result_payload(self):
            self.release_calls += 1

    class _StatusSupportedWorkerHandle:
        def __init__(self):
            self.status_calls = 0
            self.pop_calls = 0
            self.handle = None

        def fte_query_status(self, query_id):
            self.status_calls += 1
            return {
                "failed": False,
                "finished": self.status_calls >= 2,
                "selected_attempt_task_ids": [f"{query_id}.0.0.0"],
            }

        def pop_fte_result_handles(self, query_id):
            self.pop_calls += 1
            if self.pop_calls != 1:
                return []
            task_id = {
                "query_id": query_id,
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 0,
            }
            self.handle = _FailedSelectedAttemptHandle(task_id)
            return [self.handle]

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_prepare_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def fte_cleanup_query(self, _query_id):
            return {}

        def prepare_shutdown(self):
            return None

        def finish_shutdown(self):
            return None

        def abort_shutdown(self):
            return None

    import vane.runners.ray.worker_handle as ray_worker_handle

    worker = _StatusSupportedWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids, _manager_instance_id: [
            vane.ray_cxx.RayWorkerRuntime("worker-selected", worker, 1.0, 0.0, 1024)
        ],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = vane.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        manager.register_query_owner("query-selected-error", "query-selected-error")
        with pytest.raises(Exception, match="selected attempt failed"):
            manager.wait_fte_query("query-selected-error", 1.0)
        assert worker.pop_calls >= 1
        assert worker.status_calls >= 2
        assert worker.handle is not None
        assert worker.handle.release_calls == 0
        manager.drop_query_fragments("query-selected-error")
        assert worker.handle.release_calls == 1
    finally:
        manager.shutdown()


def test_wait_fte_query_ignores_retry_loser_attempt_errors(monkeypatch):
    class _FailedAttemptHandle:
        worker_id = "worker-retry"

        def __init__(self, task_id):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self._is_done = True
            self._result = None
            self._error = RuntimeError("loser attempt failed")
            self._future = None
            self.task = None
            self.release_calls = 0

        def _ensure_started(self):
            return None

        def done(self):
            return True

        def get_result_sync(self):
            raise RuntimeError("loser attempt failed")

        def release_result_payload(self):
            self.release_calls += 1

    class _NoOutputAttemptHandle:
        worker_id = "worker-retry"

        def __init__(self, task_id):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self._is_done = True
            self._result = vane.ray_cxx.RayTaskResult.no_output()
            self._error = None
            self._future = None
            self.task = None

        def _ensure_started(self):
            return None

        def done(self):
            return True

        def get_result_sync(self):
            return vane.ray_cxx.RayTaskResult.no_output()

        def release_result_payload(self):
            return None

    class _StatusSupportedWorkerHandle:
        def __init__(self):
            self.status_calls = 0
            self.pop_calls = 0
            self.loser_handle = None

        def fte_query_status(self, query_id):
            self.status_calls += 1
            return {
                "failed": False,
                "finished": self.status_calls >= 2,
                "selected_attempt_task_ids": [f"{query_id}.0.0.1"],
            }

        def pop_fte_result_handles(self, query_id):
            self.pop_calls += 1
            if self.pop_calls != 1:
                return []
            failed_task_id = {
                "query_id": query_id,
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 0,
            }
            retry_task_id = {
                "query_id": query_id,
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 1,
            }
            self.loser_handle = _FailedAttemptHandle(failed_task_id)
            return [self.loser_handle, _NoOutputAttemptHandle(retry_task_id)]

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def prepare_shutdown(self):
            return None

        def finish_shutdown(self):
            return None

        def abort_shutdown(self):
            return None

    import vane.runners.ray.worker_handle as ray_worker_handle

    worker = _StatusSupportedWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids, _manager_instance_id: [
            vane.ray_cxx.RayWorkerRuntime("worker-retry", worker, 1.0, 0.0, 1024)
        ],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = vane.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        manager.register_query_owner("query-retry-loser", "query-retry-loser")
        manager.wait_fte_query("query-retry-loser", 1.0)
        assert worker.pop_calls >= 1
        assert worker.status_calls >= 2
        assert worker.loser_handle is not None
        assert worker.loser_handle.release_calls == 1
    finally:
        manager.shutdown()


def test_wait_fte_query_release_failure_preserves_failed_handle_and_releases_rest(
    monkeypatch,
):
    class _NoOutputHandle:
        worker_id = "worker-release-failure"

        def __init__(self, task_id, *, fail_release_once=False):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self._result = vane.ray_cxx.RayTaskResult.no_output()
            self.fail_release_once = fail_release_once
            self.release_calls = 0

        def done(self):
            return True

        def get_result_sync(self):
            return self._result

        def release_result_payload(self):
            self.release_calls += 1
            if self.fail_release_once:
                self.fail_release_once = False
                raise RuntimeError("planned result payload release failure")

    class _Worker:
        def __init__(self):
            self.pop_calls = 0
            self.handles = []

        def fte_query_status(self, query_id):
            return {
                "failed": False,
                "finished": True,
                "selected_attempt_task_ids": [
                    f"{query_id}.0.0.0",
                    f"{query_id}.0.1.0",
                ],
            }

        def pop_fte_result_handles(self, query_id):
            self.pop_calls += 1
            if self.pop_calls != 1:
                return []
            self.handles = [
                _NoOutputHandle(
                    {
                        "query_id": query_id,
                        "fragment_execution_id": 0,
                        "partition_id": 0,
                        "attempt_id": 0,
                    },
                    fail_release_once=True,
                ),
                _NoOutputHandle(
                    {
                        "query_id": query_id,
                        "fragment_execution_id": 0,
                        "partition_id": 1,
                        "attempt_id": 0,
                    }
                ),
            ]
            return list(self.handles)

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_prepare_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def fte_cleanup_query(self, _query_id):
            return {}

        def prepare_shutdown(self):
            return None

        def finish_shutdown(self):
            return None

        def abort_shutdown(self):
            return None

    import vane.runners.ray.worker_handle as ray_worker_handle

    worker = _Worker()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids, _manager_instance_id: [
            vane.ray_cxx.RayWorkerRuntime(
                "worker-release-failure",
                worker,
                1.0,
                0.0,
                1024,
            )
        ],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = vane.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        manager.register_query_owner("query-release-failure", "query-release-failure")
        with pytest.raises(Exception, match="planned result payload release failure"):
            manager.wait_fte_query("query-release-failure", 1.0)

        assert worker.handles[0].release_calls == 1
        assert worker.handles[1].release_calls == 1
        manager.drop_query_fragments("query-release-failure")
        assert worker.handles[0].release_calls == 2
        assert worker.handles[1].release_calls == 1
    finally:
        manager.shutdown()


def test_wait_fte_query_does_not_drain_pending_retry_loser_attempt(monkeypatch):
    class _PendingAttemptHandle:
        worker_id = "worker-retry-pending"

        def __init__(self, task_id):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self._is_done = False
            self._result = None
            self._error = None
            self._future = None
            self.task = None
            self.done_calls = 0
            self.get_result_sync_calls = 0

        def _ensure_started(self):
            return None

        def done(self):
            self.done_calls += 1
            return False

        def get_result_sync(self):
            self.get_result_sync_calls += 1
            raise AssertionError("pending loser attempt should not be drained")

        def release_result_payload(self):
            return None

    class _NoOutputAttemptHandle:
        worker_id = "worker-retry-pending"

        def __init__(self, task_id):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self._is_done = True
            self._result = vane.ray_cxx.RayTaskResult.no_output()
            self._error = None
            self._future = None
            self.task = None

        def _ensure_started(self):
            return None

        def done(self):
            return True

        def get_result_sync(self):
            return vane.ray_cxx.RayTaskResult.no_output()

        def release_result_payload(self):
            return None

    class _StatusSupportedWorkerHandle:
        def __init__(self):
            self.status_calls = 0
            self.pop_calls = 0
            self.pending_handle = None

        def fte_query_status(self, query_id):
            self.status_calls += 1
            return {
                "failed": False,
                "finished": self.status_calls >= 2,
                "selected_attempt_task_ids": [f"{query_id}.0.0.1"],
            }

        def pop_fte_result_handles(self, query_id):
            self.pop_calls += 1
            if self.pop_calls != 1:
                return []
            loser_task_id = {
                "query_id": query_id,
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 0,
            }
            selected_task_id = {
                "query_id": query_id,
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 1,
            }
            self.pending_handle = _PendingAttemptHandle(loser_task_id)
            return [self.pending_handle, _NoOutputAttemptHandle(selected_task_id)]

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def prepare_shutdown(self):
            return None

        def finish_shutdown(self):
            return None

        def abort_shutdown(self):
            return None

    import vane.runners.ray.worker_handle as ray_worker_handle

    worker = _StatusSupportedWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids, _manager_instance_id: [
            vane.ray_cxx.RayWorkerRuntime("worker-retry-pending", worker, 1.0, 0.0, 1024)
        ],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = vane.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        manager.register_query_owner("query-retry-pending", "query-retry-pending")
        manager.wait_fte_query("query-retry-pending", 0.1)
        assert worker.pending_handle is not None
        assert worker.pending_handle.get_result_sync_calls == 0
    finally:
        manager.shutdown()


def test_wait_fte_query_clears_cached_handles_after_failed_status(monkeypatch):
    class _PendingAttemptHandle:
        worker_id = "worker-stale-failed"

        def __init__(self, task_id):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self._is_done = False
            self._result = None
            self._error = None
            self._future = None
            self.task = None
            self.done_calls = 0

        def _ensure_started(self):
            return None

        def done(self):
            self.done_calls += 1
            return False

        def get_result_sync(self):
            raise AssertionError("stale cached handle should have been cleared")

        def release_result_payload(self):
            return None

    class _StatusFailsAfterCollectWorkerHandle:
        def __init__(self):
            self.status_calls = 0
            self.pop_calls = 0
            self.pending_handle = None

        def fte_query_status(self, query_id):
            self.status_calls += 1
            if self.status_calls == 1:
                return {
                    "failed": False,
                    "finished": False,
                    "selected_attempt_task_ids": [],
                }
            if self.status_calls == 2:
                return {
                    "failed": True,
                    "finished": False,
                    "selected_attempt_task_ids": [],
                }
            return {
                "failed": False,
                "finished": True,
                "selected_attempt_task_ids": [f"{query_id}.0.0.0"],
            }

        def pop_fte_result_handles(self, query_id):
            self.pop_calls += 1
            if self.pop_calls != 1:
                return []
            task_id = {
                "query_id": query_id,
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 0,
            }
            self.pending_handle = _PendingAttemptHandle(task_id)
            return [self.pending_handle]

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def prepare_shutdown(self):
            return None

        def finish_shutdown(self):
            return None

        def abort_shutdown(self):
            return None

    import vane.runners.ray.worker_handle as ray_worker_handle

    worker = _StatusFailsAfterCollectWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids, _manager_instance_id: [
            vane.ray_cxx.RayWorkerRuntime("worker-stale-failed", worker, 1.0, 0.0, 1024)
        ],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = vane.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        manager.register_query_owner("query-stale-failed", "query-stale-failed")
        with pytest.raises(Exception, match="FTE query failed"):
            manager.wait_fte_query("query-stale-failed", 0.1)
        assert worker.pending_handle is not None
        done_calls_after_failure = worker.pending_handle.done_calls

        manager.wait_fte_query("query-stale-failed", 0.001)

        assert worker.pending_handle.done_calls == done_calls_after_failure
    finally:
        manager.shutdown()


def test_wait_fte_query_timeout_preserves_collected_handles(monkeypatch):
    class _ReadyAfterQueryTimeoutHandle:
        worker_id = "worker-timeout-preserve"
        _result = None
        _error = None
        _future = None
        _is_done = False
        task = None

        def __init__(self, task_id, worker):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self.worker = worker
            self.get_result_sync_calls = 0

        def _ensure_started(self):
            return None

        def done(self):
            return self.worker.finish_query

        def get_result_sync(self):
            self.get_result_sync_calls += 1
            return vane.ray_cxx.RayTaskResult.no_output()

        def release_result_payload(self):
            return None

    class _TimeoutThenFinishedWorkerHandle:
        def __init__(self):
            self.finish_query = False
            self.pop_calls = 0
            self.handle = None

        def fte_query_status(self, query_id):
            return {
                "failed": False,
                "finished": self.finish_query,
                "selected_attempt_task_ids": [f"{query_id}.0.0.0"] if self.finish_query else [],
            }

        def pop_fte_result_handles(self, query_id):
            self.pop_calls += 1
            if self.pop_calls != 1:
                return []
            task_id = {
                "query_id": query_id,
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 0,
            }
            self.handle = _ReadyAfterQueryTimeoutHandle(task_id, self)
            return [self.handle]

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def prepare_shutdown(self):
            return None

        def finish_shutdown(self):
            return None

        def abort_shutdown(self):
            return None

    import vane.runners.ray.worker_handle as ray_worker_handle

    worker = _TimeoutThenFinishedWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids, _manager_instance_id: [
            vane.ray_cxx.RayWorkerRuntime("worker-timeout-preserve", worker, 1.0, 0.0, 1024)
        ],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = vane.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        manager.register_query_owner("query-timeout-preserve", "query-timeout-preserve")
        with pytest.raises(Exception, match="timed out waiting for FTE query"):
            manager.wait_fte_query("query-timeout-preserve", 0.001)
        assert worker.handle is not None
        assert worker.handle.get_result_sync_calls == 0

        worker.finish_query = True
        manager.wait_fte_query("query-timeout-preserve", 1.0)

        assert worker.handle.get_result_sync_calls == 1
        assert worker.pop_calls >= 2
    finally:
        manager.shutdown()


def test_wait_fte_query_respects_timeout_after_finished_status_during_drain(monkeypatch):
    class _EventuallyReadyAttemptHandle:
        worker_id = "worker-drain-timeout"
        _result = None
        _error = None
        _future = None
        _is_done = False
        task = None

        def __init__(self, task_id):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self.ready_at = time.monotonic() + 0.2
            self.get_result_sync_calls = 0

        def _ensure_started(self):
            return None

        def done(self):
            return time.monotonic() >= self.ready_at

        def get_result_sync(self):
            self.get_result_sync_calls += 1
            return vane.ray_cxx.RayTaskResult.no_output()

        def release_result_payload(self):
            return None

    class _SlowResultWorkerHandle:
        def __init__(self):
            self.pop_calls = 0
            self.handle = None

        def fte_query_status(self, query_id):
            return {
                "failed": False,
                "finished": True,
                "selected_attempt_task_ids": [f"{query_id}.0.0.0"],
            }

        def pop_fte_result_handles(self, query_id):
            self.pop_calls += 1
            if self.pop_calls != 1:
                return []
            time.sleep(0.02)
            task_id = {
                "query_id": query_id,
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 0,
            }
            self.handle = _EventuallyReadyAttemptHandle(task_id)
            return [self.handle]

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def prepare_shutdown(self):
            return None

        def finish_shutdown(self):
            return None

        def abort_shutdown(self):
            return None

    import vane.runners.ray.worker_handle as ray_worker_handle

    worker = _SlowResultWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids, _manager_instance_id: [
            vane.ray_cxx.RayWorkerRuntime("worker-drain-timeout", worker, 1.0, 0.0, 1024)
        ],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = vane.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        manager.register_query_owner("query-drain-timeout", "query-drain-timeout")
        with pytest.raises(Exception, match="timed out draining FTE result handles"):
            manager.wait_fte_query("query-drain-timeout", 0.001)
        assert worker.handle is not None
        assert worker.handle.get_result_sync_calls == 0

        time.sleep(0.25)
        manager.wait_fte_query("query-drain-timeout", 1.0)

        assert worker.handle.get_result_sync_calls == 1
        assert worker.pop_calls >= 1
    finally:
        manager.shutdown()


def test_worker_manager_close_session_attempts_every_worker_before_retry(monkeypatch):
    import vane.runners.ray.worker_handle as ray_worker_handle

    calls = []

    class _SessionWorker:
        def __init__(self, worker_id, *, fail):
            self.worker_id = worker_id
            self.fail = fail

        def close_session(self, session_id):
            calls.append((self.worker_id, session_id))
            if self.fail:
                raise RuntimeError(f"planned close failure on {self.worker_id}")

        def prepare_shutdown(self):
            return None

        def finish_shutdown(self):
            return None

        def abort_shutdown(self):
            return None

    first = _SessionWorker("worker-a", fail=True)
    second = _SessionWorker("worker-b", fail=False)
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids, _manager_instance_id: [
            vane.ray_cxx.RayWorkerRuntime("worker-a", first, 1.0, 0.0, 1024),
            vane.ray_cxx.RayWorkerRuntime("worker-b", second, 1.0, 0.0, 1024),
        ],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    try:
        runner.warm_up()
        with pytest.raises(Exception, match="planned close failure on worker-a"):
            runner.close_session("session-a")

        assert sorted(calls) == [
            ("worker-a", "session-a"),
            ("worker-b", "session-a"),
        ]

        first.fail = False
        runner.close_session("session-a")

        assert sorted(calls) == [
            ("worker-a", "session-a"),
            ("worker-a", "session-a"),
            ("worker-b", "session-a"),
            ("worker-b", "session-a"),
        ]
    finally:
        runner.shutdown()


def test_ray_query_driver_clients_attach_to_job_runtime(monkeypatch):
    class _FakeMethod:
        def __init__(self, label):
            self.label = label
            self.calls = []

        def remote(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return (self.label, args, kwargs)

    class _FakeHandle:
        def __init__(self, name):
            self.name = name
            self.attach_client = _FakeMethod(("attach", name))
            self.ping = _FakeMethod(("ping", name))

    handle = _FakeHandle("job-runtime")
    option_calls = []
    remote_calls = []

    def _fake_options(**kwargs):
        option_calls.append(kwargs)

        class _Factory:
            def remote(self, runtime_config, duckdb_memory_bytes):
                assert runtime_config == {"PYTHONPATH": "/vane"}
                assert duckdb_memory_bytes == 50
                remote_calls.append((runtime_config, duckdb_memory_bytes, handle))
                return handle

        return _Factory()

    monkeypatch.setattr(driver, "_maybe_set_distributed_cluster_capacity", lambda: None)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(
            gcs_address="gcs-a",
            get_job_id=lambda: "job-a",
        ),
    )
    monkeypatch.setattr(
        driver,
        "get_head_node",
        lambda: {"NodeID": "a" * 56, "Resources": {"memory": 1_000}},
    )
    monkeypatch.setattr(driver, "_collect_vane_env_overrides", lambda: {"PYTHONPATH": "/vane"})
    monkeypatch.setattr(driver.RayQueryDriverActor, "options", _fake_options)
    monkeypatch.setattr(driver, "resolve_object_refs_blocking", lambda *_args, **_kwargs: True)

    client_a = driver.RayQueryDriverClient()
    client_b = driver.RayQueryDriverClient()

    assert client_a.runner is handle
    assert client_b.runner is handle
    assert client_a._owner_id != client_b._owner_id
    assert len(option_calls) == 2
    assert option_calls[0]["name"] == option_calls[1]["name"]
    assert option_calls[0]["name"].startswith("vane-query-runtime-")
    assert all(options["namespace"] == "vane" for options in option_calls)
    assert all(options["get_if_exists"] is True for options in option_calls)
    assert option_calls[0]["memory"] == 50
    assert option_calls[0]["runtime_env"]["env_vars"] == {"PYTHONPATH": "/vane"}
    assert option_calls[1]["runtime_env"]["env_vars"] == {"PYTHONPATH": "/vane"}
    assert remote_calls[0][:2] == ({"PYTHONPATH": "/vane"}, 50)
    assert remote_calls[1][:2] == ({"PYTHONPATH": "/vane"}, 50)
    assert handle.attach_client.calls == [
        (
            (
                client_a._owner_id,
                {"PYTHONPATH": "/vane"},
                client_a._lease_token,
            ),
            {},
        ),
        (
            (
                client_b._owner_id,
                {"PYTHONPATH": "/vane"},
                client_b._lease_token,
            ),
            {},
        ),
    ]
    assert handle.ping.calls == []


def test_ray_query_driver_client_heartbeat_retries_transient_failure_early(
    monkeypatch,
):
    heartbeat_calls = []

    class _HeartbeatMethod:
        @staticmethod
        def remote(owner_id, lease_token):
            heartbeat_calls.append((owner_id, lease_token))
            return f"heartbeat-{len(heartbeat_calls)}"

    class _Stop:
        def __init__(self):
            self.waits = []

        def wait(self, timeout):
            self.waits.append(timeout)
            return len(self.waits) == 3

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    _initialize_test_query_driver_client(client)
    client.runner = SimpleNamespace(heartbeat_client=_HeartbeatMethod())
    client._runtime_is_unavailable_or_replaced = lambda: False
    stop = _Stop()
    resolve_calls = []

    def _resolve(ref, **kwargs):
        resolve_calls.append((ref, kwargs))
        if len(resolve_calls) == 1:
            raise TimeoutError("transient heartbeat wait timeout")
        return True

    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)

    driver.RayQueryDriverClient._client_heartbeat_worker(
        weakref.ref(client),
        stop,
        4.0,
        4.0,
    )

    assert stop.waits == [4.0, 1.0, 4.0]
    assert heartbeat_calls == [
        ("owner-a", "test-client-lease-token"),
        ("owner-a", "test-client-lease-token"),
    ]
    assert client._client_heartbeat_error == ""


def test_ray_query_driver_client_retries_named_runtime_that_is_shutting_down(monkeypatch):
    class _FakeMethod:
        def __init__(self, label):
            self.label = label
            self.calls = []

        def remote(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self.label

    class _FakeHandle:
        def __init__(self, name):
            self.name = name
            self.attach_client = _FakeMethod(("attach", name))
            self.ping = _FakeMethod(("ping", name))
            self.detach_client = _FakeMethod(("detach", name))

    closing_handle = _FakeHandle("closing-runtime")
    replacement_handle = _FakeHandle("replacement-runtime")
    handles = iter((closing_handle, replacement_handle))
    created = []

    class _Factory:
        @staticmethod
        def remote(_runtime_config, _duckdb_memory_bytes):
            handle = next(handles)
            created.append(handle)
            return handle

    monkeypatch.setattr(driver, "_maybe_set_distributed_cluster_capacity", lambda: None)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(
            gcs_address="gcs-a",
            get_job_id=lambda: "job-a",
        ),
    )
    monkeypatch.setattr(
        driver,
        "get_head_node",
        lambda: {"NodeID": "a" * 56, "Resources": {"memory": 1_000}},
    )
    monkeypatch.setattr(driver, "_collect_vane_env_overrides", lambda: {})
    monkeypatch.setattr(driver.RayQueryDriverActor, "options", lambda **_kwargs: _Factory())
    monkeypatch.setattr(driver.time, "sleep", lambda _seconds: None)

    def _resolve(ref, **_kwargs):
        if ref == ("attach", "closing-runtime"):
            raise driver.RayRuntimeShuttingDownError("Ray query runtime is shutting down")
        return True

    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)

    client = driver.RayQueryDriverClient()

    assert created == [closing_handle, replacement_handle]
    assert client.runner is replacement_handle
    assert len(closing_handle.attach_client.calls) == 1
    assert len(replacement_handle.attach_client.calls) == 1
    assert len(replacement_handle.ping.calls) == 0


def test_ray_query_driver_client_retires_drained_named_runtime_before_retry(monkeypatch):
    class _FakeMethod:
        def __init__(self, label):
            self.label = label
            self.calls = []

        def remote(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self.label

    class _FakeHandle:
        def __init__(self, name):
            self.name = name
            self.attach_client = _FakeMethod(("attach", name))
            self.runtime_replacement_ready = _FakeMethod(("replacement-ready", name))

    closing_handle = _FakeHandle("closing-runtime")
    replacement_handle = _FakeHandle("replacement-runtime")
    active_handle = closing_handle
    created = []
    killed = []

    class _Factory:
        @staticmethod
        def remote(_runtime_config, _duckdb_memory_bytes):
            created.append(active_handle)
            return active_handle

    monkeypatch.setattr(driver, "_maybe_set_distributed_cluster_capacity", lambda: None)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(
            gcs_address="gcs-a",
            get_job_id=lambda: "job-a",
        ),
    )
    monkeypatch.setattr(
        driver,
        "get_head_node",
        lambda: {"NodeID": "a" * 56, "Resources": {"memory": 1_000}},
    )
    monkeypatch.setattr(driver, "_collect_vane_env_overrides", lambda: {})
    monkeypatch.setattr(driver.RayQueryDriverActor, "options", lambda **_kwargs: _Factory())
    monkeypatch.setattr(driver.time, "sleep", lambda _seconds: None)

    def _resolve(ref, **_kwargs):
        if ref == ("attach", "closing-runtime"):
            raise driver.RayRuntimeShuttingDownError("Ray query runtime is shutting down")
        if ref == ("replacement-ready", "closing-runtime"):
            return True
        return True

    def _kill(actor, *, no_restart):
        nonlocal active_handle
        assert actor is closing_handle
        assert no_restart is True
        killed.append(actor)
        active_handle = replacement_handle

    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)
    monkeypatch.setattr(driver.ray, "kill", _kill)

    client = driver.RayQueryDriverClient()

    assert created == [closing_handle, replacement_handle]
    assert killed == [closing_handle]
    assert client.runner is replacement_handle
    assert len(closing_handle.runtime_replacement_ready.calls) == 1


def test_runtime_replacement_classifier_rejects_permanent_actor_init_failure():
    actor_loss = driver.ray.exceptions.RayActorError(
        error_msg="actor disappeared before attach",
    )
    actor_init_failure = driver.ray.exceptions.RayActorError(
        error_msg="actor constructor failed",
        actor_init_failed=True,
    )
    assert driver._runtime_actor_is_unavailable(actor_loss)
    assert driver._runtime_actor_is_unavailable(actor_init_failure)
    assert driver._runtime_actor_is_being_replaced(actor_loss)
    assert not driver._runtime_actor_is_being_replaced(actor_init_failure)

    class _WrappedShutdown(RuntimeError):
        @staticmethod
        def as_instanceof_cause():
            return driver.RayRuntimeShuttingDownError("Ray query runtime is shutting down")

    wrapped_shutdown = _WrappedShutdown()
    assert driver._runtime_actor_is_being_replaced(wrapped_shutdown)


def test_runtime_wait_timeout_classifier_rejects_completed_remote_timeout():
    local_wait_timeout = FutureTimeoutError("ObjectRef is still pending")
    remote_method_timeout = driver.ray.exceptions.RayTaskError(
        "detach_client",
        "remote detach timeout",
        TimeoutError("remote detach timeout"),
    )
    restored_remote_timeout = TimeoutError("transported remote detach timeout")
    restored_remote_timeout.remote_exception_type = "builtins.TimeoutError"

    assert driver._runtime_error_is_wait_timeout(local_wait_timeout)
    assert not driver._runtime_error_is_wait_timeout(remote_method_timeout)
    assert not driver._runtime_error_is_wait_timeout(restored_remote_timeout)


def test_runtime_wait_timeout_classifier_accepts_pre311_future_timeout(monkeypatch):
    class _Pre311FutureTimeoutError(Exception):
        pass

    monkeypatch.setattr(driver, "FutureTimeoutError", _Pre311FutureTimeoutError)

    assert driver._runtime_error_is_wait_timeout(_Pre311FutureTimeoutError("ObjectRef is still pending"))


def test_failed_attach_cleanup_skips_detach_for_actor_init_failure():
    class _DetachMethod:
        @staticmethod
        def remote(_owner_id):
            raise AssertionError("a failed actor process cannot retain an attached owner")

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    runner = SimpleNamespace(detach_client=_DetachMethod())
    attach_error = driver.ray.exceptions.RayActorError(
        error_msg="actor constructor failed",
        actor_init_failed=True,
    )

    client._detach_after_failed_attach(runner, attach_error)


def test_ray_runner_creates_one_driver_client_for_concurrent_sessions(monkeypatch):
    from vane.runners.ray import runner as runner_module

    created = []

    class _FakeClient:
        def __init__(self):
            created.append(self)

    ray_runner = object.__new__(runner_module.RayRunner)
    ray_runner.query_driver_client = None
    ray_runner._session_ids = set()
    ray_runner._closed_session_ids = runner_module.BoundedReplayMap(capacity=65_536)
    ray_runner._session_lock = threading.RLock()
    ray_runner._closed = False
    monkeypatch.setattr(runner_module, "RayQueryDriverClient", _FakeClient)

    clients = []

    def _get_client(session_id):
        clients.append(ray_runner._client_for_session(session_id))

    first = threading.Thread(target=_get_client, args=("session-a",))
    second = threading.Thread(target=_get_client, args=("session-b",))
    first.start()
    second.start()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert len(created) == 1
    assert clients == [created[0], created[0]]
    assert ray_runner._session_ids == {"session-a", "session-b"}


def test_ray_runner_close_before_session_registration_is_terminal(monkeypatch):
    from vane.runners.ray import runner as runner_module

    ray_runner = object.__new__(runner_module.RayRunner)
    ray_runner.query_driver_client = None
    ray_runner._session_ids = set()
    ray_runner._closed_session_ids = runner_module.BoundedReplayMap(capacity=65_536)
    ray_runner._session_lock = threading.RLock()
    ray_runner._closed = False
    monkeypatch.setattr(
        runner_module,
        "RayQueryDriverClient",
        lambda: (_ for _ in ()).throw(AssertionError("closed session must not create a client")),
    )

    ray_runner.close_session("session-a")

    with pytest.raises(RuntimeError, match="Vane session is closed"):
        ray_runner._client_for_session("session-a")
    assert set(ray_runner._closed_session_ids) == {"session-a"}


def test_ray_runner_close_is_terminal_and_idempotent(monkeypatch):
    from vane.runners.ray import runner as runner_module

    closed_clients = []

    class _FakeClient:
        def close(self):
            closed_clients.append(self)

    ray_runner = object.__new__(runner_module.RayRunner)
    original_client = _FakeClient()
    ray_runner.query_driver_client = original_client
    ray_runner._session_ids = {"session-a"}
    ray_runner._closed_session_ids = runner_module.BoundedReplayMap(capacity=65_536)
    ray_runner._session_lock = threading.RLock()
    ray_runner._closed = False
    with runner_module._RAY_RUNNERS_LOCK:
        runner_module._RAY_RUNNERS.add(ray_runner)
    monkeypatch.setattr(runner_module, "RayQueryDriverClient", _FakeClient)

    ray_runner.close()
    ray_runner.close()

    with runner_module._RAY_RUNNERS_LOCK:
        assert ray_runner not in runner_module._RAY_RUNNERS

    with pytest.raises(RuntimeError, match="RayRunner is closed"):
        ray_runner._client_for_session("session-b")
    assert closed_clients == [original_client]
    assert ray_runner._session_ids == set()
    assert ray_runner._closed is True


def test_ray_runner_retries_pending_copy_cleanup_by_operation_id():
    from vane.runners.ray import runner as runner_module

    calls = []
    expected = {
        "copy_operation_id": "copy-cleanup-runner-retry",
        "copy_cleanup_state": "complete",
    }

    class _FakeClient:
        def retry_copy_cleanup(self, operation_id):
            calls.append(operation_id)
            return expected

    ray_runner = object.__new__(runner_module.RayRunner)
    ray_runner.query_driver_client = _FakeClient()
    ray_runner._session_ids = set()
    ray_runner._closed_session_ids = runner_module.BoundedReplayMap(capacity=65_536)
    ray_runner._session_lock = threading.RLock()
    ray_runner._closed = False

    result = ray_runner.retry_copy_cleanup("copy-cleanup-runner-retry")

    assert result is expected
    assert calls == ["copy-cleanup-runner-retry"]


def test_connection_close_notification_reenters_runner_registry_lock(monkeypatch):
    from vane.runners.ray import runner as runner_module

    target_session_id = "reentrant-session"
    calls = []

    class _LiveRunner:
        def close_session(self, session_id):
            calls.append(session_id)

    monkeypatch.setattr(runner_module, "_RAY_RUNNERS", {_LiveRunner()})
    registry_lock = runner_module._RAY_RUNNERS_LOCK

    registry_lock.acquire()
    try:
        reacquired = registry_lock.acquire(blocking=False)
        assert reacquired, "connection finalizers must be able to re-enter the runner registry lock"
        if reacquired:
            registry_lock.release()
    finally:
        registry_lock.release()

    with registry_lock:
        runner_module.notify_connection_closed(target_session_id)

    assert calls.count(target_session_id) == 1


def test_connection_close_notification_attempts_every_live_runner(monkeypatch):
    from vane.runners.ray import runner as runner_module

    target_session_id = "session-a"
    calls = []

    def runners_called_for(session_id):
        return sorted(name for name, called_session_id in calls if called_session_id == session_id)

    class _LiveRunner:
        def __init__(self, name, *, fail):
            self.name = name
            self.fail = fail

        def close_session(self, session_id):
            calls.append((self.name, session_id))
            if self.fail and session_id == target_session_id:
                raise RuntimeError(f"planned close failure on {self.name}")

    failing = _LiveRunner("runner-a", fail=True)
    succeeding = _LiveRunner("runner-b", fail=False)
    monkeypatch.setattr(runner_module, "_RAY_RUNNERS", {failing, succeeding})

    # A connection finalized by another test may notify the process-global
    # runner registry while this monkeypatch is active.
    runner_module.notify_connection_closed("unrelated-session")
    assert runners_called_for("unrelated-session") == ["runner-a", "runner-b"]

    with pytest.raises(RuntimeError, match="planned close failure on runner-a"):
        runner_module.notify_connection_closed(target_session_id)

    assert runners_called_for(target_session_id) == ["runner-a", "runner-b"]

    failing.fail = False
    runner_module.notify_connection_closed(target_session_id)

    assert runners_called_for(target_session_id) == ["runner-a", "runner-a", "runner-b", "runner-b"]


def test_ray_runner_session_start_and_close_are_serialized(monkeypatch):
    from vane.runners.ray import runner as runner_module

    client_init_started = threading.Event()
    client_init_release = threading.Event()
    close_started = threading.Event()

    class _FakeClient:
        def __init__(self):
            client_init_started.set()
            assert client_init_release.wait(timeout=1.0)

        def close(self):
            close_started.set()

    ray_runner = object.__new__(runner_module.RayRunner)
    ray_runner.query_driver_client = None
    ray_runner._session_ids = set()
    ray_runner._closed_session_ids = runner_module.BoundedReplayMap(capacity=65_536)
    ray_runner._session_lock = threading.RLock()
    ray_runner._closed = False
    monkeypatch.setattr(runner_module, "RayQueryDriverClient", _FakeClient)

    session_thread = threading.Thread(
        target=ray_runner._client_for_session,
        args=("session-a",),
        daemon=True,
    )
    close_thread = threading.Thread(target=ray_runner.close, daemon=True)
    session_thread.start()
    assert client_init_started.wait(timeout=1.0)
    close_thread.start()
    try:
        assert close_started.wait(timeout=0.05) is False
    finally:
        client_init_release.set()
    session_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert not session_thread.is_alive()
    assert not close_thread.is_alive()
    assert close_started.is_set()
    assert ray_runner._closed is True


def test_ray_query_driver_client_serializes_first_session_open_with_close(monkeypatch):
    open_started = threading.Event()
    open_release = threading.Event()
    events = []

    class _OpenMethod:
        @staticmethod
        def remote(owner_id, session_id, config):
            events.append(("open", owner_id, session_id, config))
            open_started.set()
            assert open_release.wait(timeout=1.0)
            return "open-ref"

    class _CloseMethod:
        @staticmethod
        def remote(owner_id, session_id):
            events.append(("close", owner_id, session_id))
            return "close-ref"

    class _Plan:
        @staticmethod
        def session_id():
            return "session-a"

        @staticmethod
        def session_config():
            return {"AWS_ACCESS_KEY_ID": "key-a"}

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    client._ray_gcs_address = "gcs-a"
    _initialize_test_query_driver_client(client)
    client.runner = SimpleNamespace(
        open_session=_OpenMethod(),
        close_session=_CloseMethod(),
    )
    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="gcs-a"),
    )
    monkeypatch.setattr(driver, "resolve_object_refs_blocking", lambda ref, **_kwargs: ref)
    open_errors = []
    close_errors = []

    def _open():
        try:
            client._ensure_session(_Plan())
        except BaseException as exc:
            open_errors.append(exc)

    def _close():
        try:
            client.close_session("session-a")
        except BaseException as exc:
            close_errors.append(exc)

    open_thread = threading.Thread(target=_open)
    close_thread = threading.Thread(target=_close)
    open_thread.start()
    assert open_started.wait(timeout=1.0)
    close_thread.start()
    for _ in range(100):
        with client._session_condition:
            if "session-a" in client._closing_session_ids:
                break
        time.sleep(0.001)
    open_release.set()
    open_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert not open_thread.is_alive()
    assert not close_thread.is_alive()
    assert len(open_errors) == 1
    assert "closed while it was opening" in str(open_errors[0])
    assert close_errors == []
    assert events == [
        ("open", "owner-a", "session-a", {"AWS_ACCESS_KEY_ID": "key-a"}),
        ("close", "owner-a", "session-a"),
    ]
    assert client._opened_sessions == {}
    assert client._opening_session_ids == set()
    assert client._closing_session_ids == set()


def test_ray_query_driver_client_closes_session_after_ambiguous_open_failure(monkeypatch):
    open_ref = object()
    close_ref = object()
    close_calls = []

    class _OpenMethod:
        @staticmethod
        def remote(*_args):
            return open_ref

    class _CloseMethod:
        @staticmethod
        def remote(owner_id, session_id):
            close_calls.append((owner_id, session_id))
            return close_ref

    class _Plan:
        @staticmethod
        def session_id():
            return "session-a"

        @staticmethod
        def session_config():
            return {"AWS_ACCESS_KEY_ID": "key-a"}

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    client._ray_gcs_address = "gcs-a"
    _initialize_test_query_driver_client(client)
    client.runner = SimpleNamespace(
        open_session=_OpenMethod(),
        close_session=_CloseMethod(),
    )
    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="gcs-a"),
    )

    def _resolve(ref, **_kwargs):
        if ref is open_ref:
            raise FutureTimeoutError("ambiguous open timeout")
        assert ref is close_ref
        return None

    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)

    with pytest.raises(FutureTimeoutError, match="ambiguous open timeout"):
        client._ensure_session(_Plan())

    assert client._opened_sessions == {}
    assert client._uncertain_sessions == {
        "session-a": {
            "AWS_ACCESS_KEY_ID": "key-a",
        }
    }

    client.close_session("session-a")

    assert close_calls == [("owner-a", "session-a")]
    assert client._uncertain_sessions == {}
    assert client._closing_session_ids == set()


def test_ray_query_driver_client_does_not_close_session_after_rejected_open(monkeypatch):
    open_ref = object()

    class _OpenMethod:
        @staticmethod
        def remote(*_args):
            return open_ref

    class _CloseMethod:
        @staticmethod
        def remote(*_args):
            raise AssertionError("an unrecorded rejected session must not receive a close RPC")

    class _Plan:
        @staticmethod
        def session_id():
            return "session-a"

        @staticmethod
        def session_config():
            return {}

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    _initialize_test_query_driver_client(client)
    client.runner = SimpleNamespace(
        open_session=_OpenMethod(),
        close_session=_CloseMethod(),
    )
    monkeypatch.setattr(
        driver,
        "resolve_object_refs_blocking",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            driver.VaneSessionOpenRejectedError("Vane session identity collision")
        ),
    )

    with pytest.raises(driver.VaneSessionOpenRejectedError, match="identity collision"):
        client._ensure_session(_Plan())

    assert client._opened_sessions == {}
    assert client._uncertain_sessions == {}

    client.close_session("session-a")

    assert set(client._closed_session_ids) == {"session-a"}


def test_ray_query_driver_client_detach_supersedes_session_close_waiting_on_open(monkeypatch):
    open_started = threading.Event()
    open_release = threading.Event()
    detach_started = threading.Event()
    detach_release = threading.Event()
    events = []

    class _OpenMethod:
        @staticmethod
        def remote(owner_id, session_id, config):
            events.append(("open", owner_id, session_id, config))
            open_started.set()
            assert open_release.wait(timeout=1.0)
            return "open-ref"

    class _CloseMethod:
        @staticmethod
        def remote(owner_id, session_id):
            events.append(("close", owner_id, session_id))
            return "close-ref"

    class _DetachMethod:
        @staticmethod
        def remote(owner_id):
            events.append(("detach", owner_id))
            detach_started.set()
            assert detach_release.wait(timeout=1.0)
            return "detach-ref"

    class _Plan:
        @staticmethod
        def session_id():
            return "session-a"

        @staticmethod
        def session_config():
            return {"AWS_ACCESS_KEY_ID": "key-a"}

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    client._ray_gcs_address = "gcs-a"
    _initialize_test_query_driver_client(client)
    client.runner = SimpleNamespace(
        open_session=_OpenMethod(),
        close_session=_CloseMethod(),
        detach_client=_DetachMethod(),
    )

    def _resolve(ref, **_kwargs):
        return False if ref == "detach-ref" else ref

    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)
    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="gcs-a"),
    )
    errors = []

    def _run(call):
        try:
            call()
        except BaseException as exc:
            errors.append(exc)

    open_thread = threading.Thread(target=_run, args=(lambda: client._ensure_session(_Plan()),))
    session_close_thread = threading.Thread(target=_run, args=(lambda: client.close_session("session-a"),))
    client_close_thread = threading.Thread(target=_run, args=(client.close,))
    open_thread.start()
    assert open_started.wait(timeout=1.0)
    session_close_thread.start()
    for _ in range(100):
        with client._session_condition:
            if "session-a" in client._closing_session_ids:
                break
        time.sleep(0.001)
    client_close_thread.start()
    open_release.set()
    assert detach_started.wait(timeout=1.0)

    session_close_thread.join(timeout=1.0)
    assert not session_close_thread.is_alive()
    assert not any(event[0] == "close" for event in events)

    detach_release.set()
    open_thread.join(timeout=1.0)
    client_close_thread.join(timeout=1.0)

    assert not open_thread.is_alive()
    assert not client_close_thread.is_alive()
    assert len(errors) == 1
    assert "closed while it was opening" in str(errors[0])
    assert events == [
        ("open", "owner-a", "session-a", {"AWS_ACCESS_KEY_ID": "key-a"}),
        ("detach", "owner-a"),
    ]
    assert client.runner is None
    assert client._opened_sessions == {}
    assert client._closing_session_ids == set()


def test_ray_query_driver_client_session_close_failure_stays_fenced_and_retryable(monkeypatch):
    close_calls = []

    class _CloseMethod:
        @staticmethod
        def remote(owner_id, session_id):
            close_calls.append((owner_id, session_id))
            return len(close_calls)

    class _Plan:
        @staticmethod
        def session_id():
            return "session-a"

        @staticmethod
        def session_config():
            return {}

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    client._ray_gcs_address = "gcs-a"
    _initialize_test_query_driver_client(client, {"session-a": {}})
    client.runner = SimpleNamespace(close_session=_CloseMethod())
    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="gcs-a"),
    )

    def _resolve(ref, **_kwargs):
        if ref == 1:
            raise RuntimeError("planned session close failure")
        return None

    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)

    with pytest.raises(RuntimeError, match="planned session close failure"):
        client.close_session("session-a")

    assert client._opened_sessions == {"session-a": {}}
    assert client._closing_session_ids == {"session-a"}
    with pytest.raises(RuntimeError, match="Vane session is closing"):
        client._ensure_session(_Plan())

    client.close_session("session-a")

    assert close_calls == [("owner-a", "session-a"), ("owner-a", "session-a")]
    assert client._opened_sessions == {}
    assert client._closing_session_ids == set()


@pytest.mark.parametrize(
    ("ray_initialized", "current_gcs_address", "current_job_id"),
    [
        (False, "gcs-a", "job-a"),
        (True, "gcs-b", "job-a"),
        (True, "gcs-a", "job-b"),
    ],
)
def test_ray_query_driver_client_session_close_is_terminal_after_runtime_changes(
    monkeypatch,
    ray_initialized,
    current_gcs_address,
    current_job_id,
):
    class _CloseMethod:
        @staticmethod
        def remote(_owner_id, _session_id):
            raise AssertionError("stale Ray runtime must not receive a session-close RPC")

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    client._ray_gcs_address = "gcs-a"
    client._ray_job_id = "job-a"
    _initialize_test_query_driver_client(client, {"session-a": {}})
    client.runner = SimpleNamespace(close_session=_CloseMethod())

    monkeypatch.setattr(driver.ray, "is_initialized", lambda: ray_initialized)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(
            gcs_address=current_gcs_address,
            get_job_id=lambda: current_job_id,
        ),
    )

    client.close_session("session-a")

    assert client._opened_sessions == {}
    assert client._uncertain_sessions == {}
    assert client._closing_session_ids == set()
    assert client._session_closes_in_progress == set()
    assert set(client._closed_session_ids) == {"session-a"}


def test_ray_query_driver_client_session_close_is_terminal_after_runtime_actor_loss(monkeypatch):
    class _CloseMethod:
        @staticmethod
        def remote(_owner_id, _session_id):
            return "close-ref"

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    client._ray_gcs_address = "gcs-a"
    _initialize_test_query_driver_client(client, {"session-a": {}})
    client.runner = SimpleNamespace(close_session=_CloseMethod())

    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="gcs-a"),
    )
    monkeypatch.setattr(
        driver,
        "resolve_object_refs_blocking",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            driver.ray.exceptions.RayActorError(error_msg="runtime actor exited")
        ),
    )

    client.close_session("session-a")

    assert client._opened_sessions == {}
    assert client._uncertain_sessions == {}
    assert client._closing_session_ids == set()
    assert client._session_closes_in_progress == set()
    assert set(client._closed_session_ids) == {"session-a"}


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_ray_query_driver_client_detaches_when_attach_is_ambiguous(monkeypatch, failure_type):
    events = []

    class _FakeMethod:
        def __init__(self, operation):
            self.operation = operation

        def remote(self, *args):
            events.append((self.operation, *args))
            return self.operation

    class _FakeHandle:
        attach_client = _FakeMethod("attach")
        ping = _FakeMethod("ping")
        detach_client = _FakeMethod("detach")

    handle = _FakeHandle()

    class _Factory:
        @staticmethod
        def remote(_runtime_config, _duckdb_memory_bytes):
            return handle

    monkeypatch.setattr(driver, "_maybe_set_distributed_cluster_capacity", lambda: None)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(
            gcs_address="gcs-a",
            get_job_id=lambda: "job-a",
        ),
    )
    monkeypatch.setattr(
        driver,
        "get_head_node",
        lambda: {"NodeID": "a" * 56, "Resources": {"memory": 1_000}},
    )
    monkeypatch.setattr(driver, "_collect_vane_env_overrides", lambda: {})
    monkeypatch.setattr(driver.RayQueryDriverActor, "options", lambda **_kwargs: _Factory())

    def resolve(ref, **kwargs):
        events.append(("resolve", ref, kwargs))
        if ref == "attach":
            raise failure_type("planned attach failure")
        return True

    monkeypatch.setattr(driver, "resolve_object_refs_blocking", resolve)
    monkeypatch.setattr(
        driver.ray,
        "kill",
        lambda actor, *, no_restart: events.append(("kill", actor, no_restart)),
    )

    with pytest.raises(failure_type, match="planned attach failure"):
        driver.RayQueryDriverClient()

    attach_event = next(event for event in events if event[0] == "attach")
    detach_event = next(event for event in events if event[0] == "detach")
    assert attach_event[1] == detach_event[1]
    assert all(event[0] != "ping" for event in events)
    assert ("kill", handle, True) in events


def test_ray_query_driver_client_retries_detach_after_ambiguous_attach(monkeypatch):
    detach_calls = []

    class _DetachMethod:
        @staticmethod
        def remote(owner_id):
            detach_calls.append(owner_id)
            return "detach-ref"

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    client._ray_gcs_address = "gcs-a"
    runner = SimpleNamespace(detach_client=_DetachMethod())
    resolve_attempts = 0

    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="gcs-a"),
    )
    monkeypatch.setattr(driver.time, "sleep", lambda _seconds: None)

    def _resolve(ref, **kwargs):
        nonlocal resolve_attempts
        assert ref == "detach-ref"
        assert kwargs["honor_query_deadline"] is False
        resolve_attempts += 1
        if resolve_attempts == 1:
            raise RuntimeError("transient detach failure")
        return False

    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)

    client._detach_after_failed_attach(runner, RuntimeError("ambiguous attach"))

    assert detach_calls == ["owner-a", "owner-a"]
    assert resolve_attempts == 2


def test_ray_query_driver_client_reuses_ambiguous_detach_ref_after_timeout(monkeypatch):
    detach_calls = []
    resolve_attempts = 0

    class _DetachMethod:
        @staticmethod
        def remote(owner_id):
            detach_calls.append(owner_id)
            return "detach-ref"

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    client._ray_gcs_address = "gcs-a"
    runner = SimpleNamespace(detach_client=_DetachMethod())

    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="gcs-a"),
    )
    monkeypatch.setattr(driver.time, "sleep", lambda _seconds: None)

    def _resolve(ref, **kwargs):
        nonlocal resolve_attempts
        assert ref == "detach-ref"
        assert kwargs["honor_query_deadline"] is False
        resolve_attempts += 1
        if resolve_attempts == 1:
            raise FutureTimeoutError("ambiguous detach timeout")
        return False

    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)

    client._detach_after_failed_attach(runner, RuntimeError("ambiguous attach"))

    assert detach_calls == ["owner-a"]
    assert resolve_attempts == 2


def test_ray_query_driver_client_preserves_attach_error_when_detach_was_not_needed(monkeypatch):
    class _WrappedPermissionError(RuntimeError):
        @staticmethod
        def as_instanceof_cause():
            return PermissionError("owner was never attached")

    class _DetachMethod:
        @staticmethod
        def remote(owner_id):
            assert owner_id == "owner-a"
            return "detach-ref"

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    client._ray_gcs_address = "gcs-a"
    runner = SimpleNamespace(detach_client=_DetachMethod())

    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="gcs-a"),
    )
    monkeypatch.setattr(
        driver,
        "resolve_object_refs_blocking",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_WrappedPermissionError()),
    )

    client._detach_after_failed_attach(runner, RuntimeError("ambiguous attach"))


def test_ray_query_driver_client_reports_unconfirmed_detach_after_ambiguous_attach(monkeypatch):
    detach_calls = []

    class _DetachMethod:
        @staticmethod
        def remote(owner_id):
            detach_calls.append(owner_id)
            return "detach-ref"

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    client._ray_gcs_address = "gcs-a"
    runner = SimpleNamespace(detach_client=_DetachMethod())
    attach_error = RuntimeError("ambiguous attach")

    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="gcs-a"),
    )
    monkeypatch.setattr(driver.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        driver,
        "resolve_object_refs_blocking",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("persistent detach failure")),
    )

    with pytest.raises(RuntimeError, match="owner detach could not be confirmed") as error:
        client._detach_after_failed_attach(runner, attach_error)

    assert error.value.__cause__ is attach_error
    assert detach_calls == ["owner-a"] * driver._RUNTIME_ATTACH_CLEANUP_ATTEMPTS


@pytest.mark.parametrize("close_order", [(0, 1), (1, 0)])
def test_ray_query_driver_client_close_keeps_shared_runtime_until_last_owner(monkeypatch, close_order):
    events = []

    class _FakeMethod:
        def __init__(self, operation, actor_name):
            self.operation = operation
            self.actor_name = actor_name

        def remote(self, owner_id):
            events.append((self.operation, self.actor_name, owner_id))
            return (self.operation, self.actor_name)

    class _FakeHandle:
        def __init__(self, name):
            self.name = name
            self.ping = _FakeMethod("ping", name)
            self.detach_client = _FakeMethod("detach", name)

    shared_handle = _FakeHandle("job-runtime")
    clients = []
    for index in range(2):
        client = object.__new__(driver.RayQueryDriverClient)
        client._owner_id = f"owner-{index}"
        client._ray_gcs_address = "gcs-a"
        _initialize_test_query_driver_client(client)
        client.runner = shared_handle
        clients.append(client)

    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="gcs-a"),
    )

    def resolve(ref, **_kwargs):
        events.append(("resolve", *ref))
        if ref[0] == "detach":
            detach_count = sum(event[0] == "detach" for event in events)
            return detach_count == 2
        return True

    monkeypatch.setattr(driver, "resolve_object_refs_blocking", resolve)
    monkeypatch.setattr(
        driver.ray,
        "kill",
        lambda actor, *, no_restart: events.append(("kill", actor.name, no_restart)),
    )

    first, second = close_order
    clients[first].close()

    assert clients[first].runner is None
    assert clients[second].runner is not None
    ping_ref = clients[second].runner.ping.remote(clients[second]._owner_id)
    assert resolve(ping_ref) is True

    clients[second].close()

    assert events == [
        ("detach", "job-runtime", f"owner-{first}"),
        ("resolve", "detach", "job-runtime"),
        ("ping", "job-runtime", f"owner-{second}"),
        ("resolve", "ping", "job-runtime"),
        ("detach", "job-runtime", f"owner-{second}"),
        ("resolve", "detach", "job-runtime"),
        ("kill", "job-runtime", True),
    ]


def test_ray_query_driver_client_close_remains_retryable_after_detach_failure(monkeypatch):
    class _DetachMethod:
        @staticmethod
        def remote(owner_id):
            assert owner_id == "owner-a"
            return "detach-ref"

    runner = SimpleNamespace(detach_client=_DetachMethod())
    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    client._ray_gcs_address = "gcs-a"
    _initialize_test_query_driver_client(client, {"session-a": {}})
    client.runner = runner

    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="gcs-a"),
    )
    monkeypatch.setattr(
        driver,
        "resolve_object_refs_blocking",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("planned detach failure")),
    )

    with pytest.raises(RuntimeError, match="planned detach failure"):
        client.close()

    assert client.runner is runner
    assert client._opened_sessions == {"session-a": {}}
    assert client._client_closing is True
    assert client._client_heartbeat_stop.is_set()


def test_ray_query_driver_client_close_accepts_expiry_reaper_takeover(
    monkeypatch,
):
    class _DetachMethod:
        @staticmethod
        def remote(owner_id):
            assert owner_id == "owner-a"
            return "detach-ref"

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    client._ray_gcs_address = "gcs-a"
    _initialize_test_query_driver_client(client, {"session-a": {}})
    client.runner = SimpleNamespace(detach_client=_DetachMethod())

    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(
            gcs_address="gcs-a",
            get_job_id=lambda: "job-a",
        ),
    )
    monkeypatch.setattr(
        driver,
        "resolve_object_refs_blocking",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PermissionError("expired owner cleanup is already in progress")
        ),
    )

    client.close()

    assert client.runner is None
    assert client._opened_sessions == {}
    assert client._client_closing is False
    assert client._client_heartbeat_stop.is_set()


def test_ray_query_driver_client_close_is_terminal_when_ray_stops_during_detach(monkeypatch):
    class _DetachMethod:
        @staticmethod
        def remote(owner_id):
            assert owner_id == "owner-a"
            return "detach-ref"

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    client._ray_gcs_address = "gcs-a"
    _initialize_test_query_driver_client(client, {"session-a": {}})
    client.runner = SimpleNamespace(detach_client=_DetachMethod())
    initialized = iter((True, False))

    monkeypatch.setattr(driver.ray, "is_initialized", lambda: next(initialized))
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="gcs-a"),
    )
    monkeypatch.setattr(
        driver,
        "resolve_object_refs_blocking",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Ray is not initialized")),
    )

    client.close()

    assert client.runner is None
    assert client._opened_sessions == {}
    assert client._client_closing is False
    assert client._client_close_in_progress is False


def test_ray_query_driver_client_close_remains_retryable_after_kill_failure(monkeypatch):
    class _DetachMethod:
        @staticmethod
        def remote(owner_id):
            assert owner_id == "owner-a"
            return "detach-ref"

    runner = SimpleNamespace(detach_client=_DetachMethod())
    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    client._ray_gcs_address = "gcs-a"
    _initialize_test_query_driver_client(client, {"session-a": {}})
    client.runner = runner
    kill_calls = []

    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="gcs-a"),
    )
    monkeypatch.setattr(driver, "resolve_object_refs_blocking", lambda *_args, **_kwargs: True)

    def _kill(actor, *, no_restart):
        assert actor is runner
        assert no_restart is True
        kill_calls.append(actor)
        if len(kill_calls) == 1:
            raise RuntimeError("planned kill failure")

    monkeypatch.setattr(driver.ray, "kill", _kill)

    with pytest.raises(RuntimeError, match="planned kill failure"):
        client.close()

    assert client.runner is runner
    assert client._opened_sessions == {"session-a": {}}
    assert client._client_closing is True

    client.close()

    assert client.runner is None
    assert client._opened_sessions == {}
    assert client._client_closing is False
    assert kill_calls == [runner, runner]


def test_ray_query_driver_client_close_is_terminal_after_runtime_actor_loss(monkeypatch):
    class _DetachMethod:
        @staticmethod
        def remote(owner_id):
            assert owner_id == "owner-a"
            return "detach-ref"

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    client._ray_gcs_address = "gcs-a"
    _initialize_test_query_driver_client(client, {"session-a": {}})
    client.runner = SimpleNamespace(detach_client=_DetachMethod())

    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="gcs-a"),
    )
    monkeypatch.setattr(
        driver,
        "resolve_object_refs_blocking",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            driver.ray.exceptions.RayActorError(error_msg="runtime actor exited")
        ),
    )

    client.close()

    assert client.runner is None
    assert client._opened_sessions == {}
    assert client._client_closing is False
    assert client._client_close_in_progress is False


def test_ray_query_driver_client_concurrent_close_detaches_once(monkeypatch):
    detach_calls: list[str] = []
    kill_calls = []
    detach_started = threading.Event()
    allow_detach = threading.Event()

    class _DetachMethod:
        @staticmethod
        def remote(owner_id):
            detach_calls.append(owner_id)
            return "detach-ref"

    runner = SimpleNamespace(detach_client=_DetachMethod())
    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    client._ray_gcs_address = "gcs-a"
    _initialize_test_query_driver_client(client, {"session-a": {}})
    client.runner = runner

    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        driver.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="gcs-a"),
    )

    def _resolve(*_args, **_kwargs):
        detach_started.set()
        assert allow_detach.wait(timeout=5)
        return True

    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _resolve)
    monkeypatch.setattr(
        driver.ray,
        "kill",
        lambda actor, *, no_restart: kill_calls.append((actor, no_restart)),
    )

    errors: list[BaseException] = []

    def _close() -> None:
        try:
            client.close()
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=_close)
    second = threading.Thread(target=_close)
    first.start()
    assert detach_started.wait(timeout=5)
    second.start()
    allow_detach.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert detach_calls == ["owner-a"]
    assert kill_calls == [(runner, True)]
    assert client.runner is None


def test_ray_query_driver_client_close_before_open_is_terminal():
    class _Plan:
        @staticmethod
        def session_id():
            return "session-a"

        @staticmethod
        def session_config():
            return {}

    client = object.__new__(driver.RayQueryDriverClient)
    client._owner_id = "owner-a"
    _initialize_test_query_driver_client(client)
    client.runner = SimpleNamespace()

    client.close_session("session-a")

    assert set(client._closed_session_ids) == {"session-a"}
    with pytest.raises(RuntimeError, match="Vane session is closed"):
        client._ensure_session(_Plan())
