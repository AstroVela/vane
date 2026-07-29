# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from duckdb.runners.ray.fragment_worker_lifecycle import FteWorkerLifecycleMixin


class _RemoteMethod:
    def __init__(self, events: list[str], result: object, event: str = "shutdown-rpc") -> None:
        self._events = events
        self._result = result
        self._event = event

    def remote(self) -> object:
        self._events.append(self._event)
        return self._result


class _Actor:
    def __init__(self, events: list[str], result: object) -> None:
        self.finish_shutdown = _RemoteMethod(events, result, "finish-rpc")


def _lifecycle(actor: object) -> FteWorkerLifecycleMixin:
    lifecycle = FteWorkerLifecycleMixin()
    lifecycle.worker_id = ""
    lifecycle.actor_handle = actor
    return lifecycle


def test_blocking_materialization_completion_routes_native_node_to_query_stage(monkeypatch):
    from duckdb.runners.ray import query_resource_runtime

    calls: list[str] = []

    class Manager:
        def mark_materializing_stage_completed(self, stage_id: str) -> bool:
            calls.append(stage_id)
            return True

    def get_manager(query_id: str) -> Manager:
        assert query_id == "query-a"
        return Manager()

    monkeypatch.setattr(query_resource_runtime, "get_query_resource_manager", get_manager)
    lifecycle = _lifecycle(object())

    assert lifecycle.blocking_materialization_completed("query-a", "42") is True
    assert calls == ["stage:query-a:node:42:fte"]

    with pytest.raises(ValueError, match="requires query_id and node_id"):
        lifecycle.blocking_materialization_completed("", "42")


def test_worker_finish_shutdown_waits_for_graceful_rpc_before_kill(monkeypatch):
    from duckdb.runners.ray import fragment_worker_lifecycle as lifecycle_module

    events: list[str] = []
    result = object()
    lifecycle = _lifecycle(_Actor(events, result))

    def resolve(ref, **kwargs):
        assert ref is result
        assert kwargs == {"timeout": 30, "honor_query_deadline": False}
        events.append("shutdown-resolved")

    monkeypatch.setattr(lifecycle_module, "resolve_object_refs_blocking", resolve)
    monkeypatch.setattr(lifecycle_module.ray, "kill", lambda actor: events.append("kill"))

    lifecycle.finish_shutdown()

    assert events == ["finish-rpc", "shutdown-resolved", "kill"]


def test_worker_finish_shutdown_kills_actor_when_graceful_rpc_fails(monkeypatch):
    from duckdb.runners.ray import fragment_worker_lifecycle as lifecycle_module

    events: list[str] = []
    result = object()
    lifecycle = _lifecycle(_Actor(events, result))

    def fail_resolve(ref, **kwargs):
        assert ref is result
        events.append("shutdown-failed")
        raise RuntimeError("stop failed")

    monkeypatch.setattr(lifecycle_module, "resolve_object_refs_blocking", fail_resolve)
    monkeypatch.setattr(lifecycle_module.ray, "kill", lambda actor: events.append("kill"))

    with pytest.raises(RuntimeError, match="graceful shutdown failed"):
        lifecycle.finish_shutdown()

    assert events == ["finish-rpc", "shutdown-failed", "kill"]


def test_worker_two_phase_shutdown_keeps_actor_alive_until_finish(monkeypatch):
    from duckdb.runners.ray import fragment_worker_lifecycle as lifecycle_module

    events: list[str] = []
    prepare_result = object()
    finish_result = object()

    class Actor:
        prepare_shutdown = _RemoteMethod(events, prepare_result, "prepare-rpc")
        finish_shutdown = _RemoteMethod(events, finish_result, "finish-rpc")

    lifecycle = _lifecycle(Actor())

    def resolve(ref, **kwargs):
        assert kwargs == {"timeout": 30, "honor_query_deadline": False}
        if ref is prepare_result:
            events.append("prepare-resolved")
        elif ref is finish_result:
            events.append("finish-resolved")
        else:
            raise AssertionError(ref)

    monkeypatch.setattr(lifecycle_module, "resolve_object_refs_blocking", resolve)
    monkeypatch.setattr(lifecycle_module.ray, "kill", lambda actor: events.append("kill"))

    lifecycle.prepare_shutdown()
    assert events == ["prepare-rpc", "prepare-resolved"]

    lifecycle.finish_shutdown()
    assert events == ["prepare-rpc", "prepare-resolved", "finish-rpc", "finish-resolved", "kill"]


def test_failed_worker_prepare_does_not_publish_failure_recursively(monkeypatch):
    from duckdb.runners.ray import fragment_worker_lifecycle as lifecycle_module

    events: list[str] = []
    prepare_result = object()

    class Actor:
        prepare_shutdown = _RemoteMethod(events, prepare_result, "prepare-rpc")

    lifecycle = _lifecycle(Actor())
    lifecycle._begin_worker_shutdown = lambda: events.append("failure-published")

    def resolve(ref, **kwargs):
        assert ref is prepare_result
        assert kwargs == {"timeout": 30, "honor_query_deadline": False}
        events.append("prepare-resolved")

    monkeypatch.setattr(lifecycle_module, "resolve_object_refs_blocking", resolve)

    lifecycle.prepare_failed_worker_shutdown()

    assert events == ["prepare-rpc", "prepare-resolved"]


def test_query_close_interrupts_only_owned_native_cursors():
    from duckdb.runners.ray import worker as worker_module

    class Cursor:
        def __init__(self):
            self.interrupts = 0

        def interrupt(self):
            self.interrupts += 1

    class DummyActor:
        _native_execution_condition = threading.Condition()
        _native_execution_counts_by_query: dict[str, int] = {}
        _active_native_cursors: set[Cursor] = set()
        _native_cursor_query_ids: dict[Cursor, str] = {}
        _closing_native_queries: set[str] = set()

    actor = DummyActor()
    actor_class = worker_module.RayWorkerActor.__ray_metadata__.modified_class
    query_cursor = Cursor()
    other_cursor = Cursor()

    assert actor_class._register_native_cursor(actor, query_cursor, "query-a") is True
    assert actor_class._register_native_cursor(actor, other_cursor, "query-b") is True
    assert actor_class._close_worker_native_query(actor, "query-a") == []
    assert query_cursor.interrupts == 1
    assert other_cursor.interrupts == 0
    assert actor_class._worker_native_query_is_closing(actor, "query-a") is True

    late_cursor = Cursor()
    assert actor_class._register_native_cursor(actor, late_cursor, "query-a") is False
    actor_class._unregister_native_cursor(actor, query_cursor)
    actor_class._unregister_native_cursor(actor, other_cursor)
    actor_class._unregister_native_cursor(actor, late_cursor)
    actor_class._retire_worker_native_query(actor, "query-a")
    assert actor_class._worker_native_query_is_closing(actor, "query-a") is False


def test_actor_shutdown_joins_native_threads_before_closing_runtime(monkeypatch):
    from duckdb.runners.ray import worker as worker_module

    events: list[str] = []
    interrupted = threading.Event()
    native_finished = threading.Event()

    class Cursor:
        def interrupt(self):
            events.append("cursor-interrupt")
            interrupted.set()

    class Connection:
        def interrupt(self):
            events.append("connection-interrupt")

        def close(self):
            assert native_finished.is_set()
            events.append("connection-close")

    class SessionConnection:
        def close(self):
            assert native_finished.is_set()
            events.append("session-connection-close")

    class TaskManager:
        def shutdown(self):
            events.append("tasks-canceled")

    class DummyActor:
        _shutdown_lock = threading.Lock()
        _shared_conn_lock = threading.Lock()
        _session_connections_lock = threading.RLock()
        _native_execution_condition = threading.Condition()
        _native_execution_count = 1
        _native_execution_counts_by_query = {"query-a": 1}
        _active_native_cursors = {Cursor()}
        _shutdown_started = False
        _shutdown_prepared = False
        _shutdown_complete = False
        _shared_conn = Connection()
        _session_connections = {"session-a": ({}, SessionConnection())}
        _fte_task_manager = TaskManager()

    actor = DummyActor()
    actor_class = worker_module.RayWorkerActor.__ray_metadata__.modified_class

    def finish_native():
        assert interrupted.wait(timeout=5)
        events.append("native-finished")
        native_finished.set()
        with actor._native_execution_condition:
            actor._active_native_cursors.clear()
            actor._native_execution_count -= 1
            actor._native_execution_counts_by_query.clear()
            actor._native_execution_condition.notify_all()

    native_thread = threading.Thread(target=finish_native)
    native_thread.start()
    monkeypatch.setattr(
        worker_module,
        "require_ray_cxx_attr",
        lambda name, hint=None: (
            lambda: (
                events.append("flight-stopped")
                if name == "shutdown_local_flight_service"
                else (_ for _ in ()).throw(AssertionError(name))
            )
        ),
    )
    monkeypatch.setattr(
        worker_module,
        "_clear_datasource_factory_registry",
        lambda: events.append("datasource-registry-cleared"),
    )

    actor_class._prepare_worker_runtime_shutdown(actor)
    native_thread.join(timeout=5)

    assert native_thread.is_alive() is False
    assert actor._shutdown_started is True
    assert actor._shutdown_prepared is True
    assert actor._shutdown_complete is False
    assert actor._shared_conn is None
    assert events == [
        "tasks-canceled",
        "connection-interrupt",
        "cursor-interrupt",
        "native-finished",
        "session-connection-close",
        "connection-close",
    ]

    actor_class._finish_worker_runtime_shutdown(actor)
    assert actor._shutdown_complete is True
    assert events[-2:] == ["flight-stopped", "datasource-registry-cleared"]
    with pytest.raises(RuntimeError, match="shutting down"):
        actor_class._begin_worker_native_execution(actor, "query-a")
    with pytest.raises(RuntimeError, match="shutting down"):
        actor_class._ensure_worker_runtime_running(actor)


def test_query_teardown_waits_for_pre_registration_native_admission():
    from duckdb.runners.ray import worker as worker_module

    class DummyActor:
        _shutdown_started = False
        _native_execution_condition = threading.Condition()
        _native_execution_count = 0
        _native_execution_counts_by_query: dict[str, int] = {}
        _active_native_cursors: set[object] = set()
        _native_cursor_query_ids: dict[object, str] = {}
        _closing_native_queries: set[str] = set()

    actor = DummyActor()
    actor_class = worker_module.RayWorkerActor.__ray_metadata__.modified_class
    query_id = "query-admitted-before-native-registration"

    # Model the exact gap after Python admission but before the C++ registry's
    # begin_query_execution call.
    actor_class._begin_worker_native_execution(actor, query_id)
    assert actor_class._close_worker_native_query(actor, query_id) == []

    async def exercise_barrier():
        barrier = asyncio.create_task(
            actor_class._wait_worker_native_executions_for_query(actor, query_id, timeout_s=1.0)
        )
        await asyncio.sleep(0)
        assert barrier.done() is False
        with pytest.raises(RuntimeError, match="active executions"):
            actor_class._retire_worker_native_query(actor, query_id)

        actor_class._end_worker_native_execution(actor, query_id)
        await barrier
        actor_class._retire_worker_native_query(actor, query_id)

    asyncio.run(exercise_barrier())
    assert actor._native_execution_count == 0
    assert actor._native_execution_counts_by_query == {}
    assert actor_class._worker_native_query_is_closing(actor, query_id) is False


def test_fte_task_manager_rejects_new_tasks_after_shutdown():
    from duckdb.runners.fte.fte_config import FteWorkerAdmissionConfig
    from duckdb.runners.fte.fte_worker_runtime import FteWorkerTaskManager

    async def execute(_request):
        raise AssertionError("shut-down manager must not execute a task")

    manager = FteWorkerTaskManager(
        execute,
        admission_config=FteWorkerAdmissionConfig(
            max_running_tasks=1,
            mode="test",
            memory_budget_bytes=1,
        ),
    )
    assert manager.shutdown() == {"removed": 0, "canceled": 0}
    assert manager.shutdown() == {"removed": 0, "canceled": 0}
    with pytest.raises(RuntimeError, match="shut down"):
        asyncio.run(manager.create_task({}))


def test_fte_task_manager_shutdown_retries_retained_task_cleanup():
    from duckdb.runners.fte import FteTaskState
    from duckdb.runners.fte.fte_config import FteWorkerAdmissionConfig
    from duckdb.runners.fte.fte_worker_runtime import FteWorkerTaskManager

    async def execute(_request):
        raise AssertionError("injected task must not execute")

    class Execution:
        def __init__(self):
            self.status = SimpleNamespace(state=FteTaskState.RUNNING)
            self.cancel_calls = 0
            self.release_calls = 0

        def cancel(self):
            self.cancel_calls += 1
            self.status.state = FteTaskState.CANCELED

        def release_result(self, *, reason):
            assert reason == "worker_shutdown"
            self.release_calls += 1
            if self.release_calls == 1:
                raise RuntimeError("planned release failure")

    manager = FteWorkerTaskManager(
        execute,
        admission_config=FteWorkerAdmissionConfig(
            max_running_tasks=1,
            mode="test",
            memory_budget_bytes=1,
        ),
    )
    execution = Execution()
    manager.tasks["task"] = execution
    manager.running_tasks.add("task")

    with pytest.raises(RuntimeError, match="planned release failure"):
        manager.shutdown()

    assert manager.tasks == {"task": execution}
    assert execution.cancel_calls == 1
    assert manager.shutdown() == {"removed": 1, "canceled": 0}
    assert manager.tasks == {}
    assert execution.release_calls == 2


def test_fte_task_manager_shutdown_does_not_admit_queued_task_from_completion_callback():
    from duckdb.runners.fte import FteTaskState
    from duckdb.runners.fte.fte_config import FteWorkerAdmissionConfig
    from duckdb.runners.fte.fte_worker_runtime import FteWorkerTaskManager

    async def execute(_request):
        raise AssertionError("queued task must not execute during shutdown")

    manager = FteWorkerTaskManager(
        execute,
        admission_config=FteWorkerAdmissionConfig(
            max_running_tasks=1,
            mode="test",
            memory_budget_bytes=1,
        ),
    )
    started = []

    class Execution:
        def __init__(self, state, *, on_release=None):
            self.status = SimpleNamespace(state=state)
            self.memory_requirement_bytes = 1
            self.on_release = on_release

        def cancel(self):
            self.status.state = FteTaskState.CANCELED

        def release_result(self, *, reason):
            assert reason == "worker_shutdown"
            if self.on_release is not None:
                self.on_release()

    def finish_running_task_during_shutdown():
        manager.running_tasks.discard("running")
        manager._drain_queue()

    running = Execution(FteTaskState.RUNNING, on_release=finish_running_task_during_shutdown)
    queued = Execution(FteTaskState.QUEUED)
    manager.tasks["running"] = running
    manager.tasks["queued"] = queued
    manager.running_tasks.add("running")
    manager.queued_tasks.append("queued")
    manager._start_execution = lambda execution, *, reason: started.append((execution, reason))

    assert manager.shutdown() == {"removed": 2, "canceled": 2}
    assert started == []
    assert list(manager.queued_tasks) == []


def test_actor_task_manager_creation_is_rejected_after_shutdown_starts():
    from duckdb.runners.ray import worker as worker_module

    class DummyActor:
        _shutdown_lock = threading.RLock()
        _shutdown_started = True
        _fte_task_manager = None

    actor_class = worker_module.RayWorkerActor.__ray_metadata__.modified_class

    with pytest.raises(RuntimeError, match="shutting down"):
        actor_class._get_fte_task_manager(DummyActor())


def test_actor_session_connection_creation_is_rejected_after_shutdown_starts():
    from duckdb.runners.ray import worker as worker_module

    actor_class = worker_module.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    actor._shutdown_started = True
    actor._session_connections_lock = threading.RLock()
    actor._session_connections = {}
    actor._closed_session_ids = worker_module.BoundedReplayMap(capacity=16)

    with pytest.raises(RuntimeError, match="shutting down"):
        actor_class._get_session_conn(actor, "session-a", {})


def test_actor_shutdown_retries_failed_session_connection_close():
    from duckdb.runners.ray import worker as worker_module

    class SessionConnection:
        close_calls = 0

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("planned session connection close failure")

    session_connection = SessionConnection()

    class DummyActor:
        _shutdown_lock = threading.RLock()
        _shared_conn_lock = threading.Lock()
        _session_connections_lock = threading.RLock()
        _native_execution_condition = threading.Condition()
        _native_execution_count = 0
        _native_execution_counts_by_query: dict[str, int] = {}
        _active_native_cursors: set[object] = set()
        _native_cursor_query_ids: dict[object, str] = {}
        _closing_native_queries: set[str] = set()
        _shutdown_started = False
        _shutdown_prepared = False
        _shutdown_complete = False
        _shared_conn = None
        _session_connections = {"session-a": ({}, session_connection)}
        _fte_task_manager = None

    actor = DummyActor()
    actor_class = worker_module.RayWorkerActor.__ray_metadata__.modified_class

    with pytest.raises(RuntimeError, match="planned session connection close failure"):
        actor_class._prepare_worker_runtime_shutdown(actor)

    assert actor._session_connections == {"session-a": ({}, session_connection)}
    actor_class._prepare_worker_runtime_shutdown(actor)

    assert actor._session_connections == {}
    assert actor._shutdown_prepared is True
    assert session_connection.close_calls == 2
