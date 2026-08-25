# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import pickle
import threading
import time
import uuid
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import vane
import vane.runners.fte.backends.native.backend as native_backend_mod
from tests.result_stream_helpers import collect_result_stream
from vane.runners.fte import FteTaskAttemptId, FteTaskId, FteTaskState, TaskResultState
from vane.runners.fte.backends.native import (
    NativeFteWorkerManagerBackend,
    NativeTaskResultHandle,
    NativeWorkerHandle,
)
from vane.runners.fte.backends.native.backend import (
    _BackgroundEventLoop,
    _flight_exchange_node_id_from_env,
    _NativeFteProgressRegistry,
    _NativeFteRegisteredFragment,
)
from vane.runners.fte.fte_config import FTE_WORKER_RUNTIME
from vane.runners.progress import build_progress_snapshot


def _task_id(partition_id: int, *, query_id: str = "q") -> dict[str, int | str]:
    return {
        "query_id": query_id,
        "fragment_execution_id": 0,
        "partition_id": partition_id,
        "attempt_id": 0,
    }


def _scan_split_batch(split_id: str, data: Any) -> dict[str, Any]:
    return {
        "splits": [
            {
                "split_id": split_id,
                "estimated_bytes": len(data) if isinstance(data, bytes) else None,
                "data": data,
            }
        ]
    }


class _FakeNativeWorkerTask:
    def __init__(
        self,
        *,
        name: str = "native-task",
        context: dict[str, Any] | None = None,
        task_context: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        plan: Any = None,
        exchange_sink_config: Any = None,
    ) -> None:
        self._name = name
        self._context = dict(context or {})
        self._task_context = dict(task_context or {})
        self._inputs = dict(inputs or {})
        self._plan = {"plan": "native"} if plan is None else plan
        self._exchange_sink_config = exchange_sink_config

    def name(self):
        return self._name

    def context(self):
        return dict(self._context)

    def task_context(self):
        return dict(self._task_context)

    def Inputs(self):
        return dict(self._inputs)

    def plan(self):
        return self._plan

    def exchange_sink_config(self):
        return self._exchange_sink_config


class _QueryLifecycleBackend:
    def register_query_owner(self, query_id, owner_query_id):
        assert str(query_id)
        assert str(owner_query_id)


def test_native_background_event_loop_concurrent_first_submit_starts_once(monkeypatch):
    background = _BackgroundEventLoop("native-fte-concurrent-start")
    original_thread_main = background._thread_main
    thread_main_started = threading.Event()
    release_thread_main = threading.Event()
    call_count = 0
    call_count_lock = threading.Lock()

    def gated_thread_main():
        nonlocal call_count
        with call_count_lock:
            call_count += 1
        thread_main_started.set()
        assert release_thread_main.wait(timeout=1.0)
        original_thread_main()

    monkeypatch.setattr(background, "_thread_main", gated_thread_main)
    submit_barrier = threading.Barrier(8)

    def submit(index):
        submit_barrier.wait(timeout=1.0)
        future = background.submit(asyncio.sleep(0, result=index))
        return future.result(timeout=1.0)

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            submitted = [executor.submit(submit, index) for index in range(8)]
            assert thread_main_started.wait(timeout=1.0)
            release_thread_main.set()
            assert sorted(future.result(timeout=2.0) for future in submitted) == list(range(8))
    finally:
        release_thread_main.set()
        background.shutdown(timeout_s=1.0)

    assert call_count == 1


def test_native_background_event_loop_submit_during_shutdown_is_rejected(monkeypatch):
    background = _BackgroundEventLoop("native-fte-submit-during-shutdown")
    background.start()
    event_loop = background._loop
    assert event_loop is not None
    original_call_soon_threadsafe = event_loop.call_soon_threadsafe
    stop_scheduling_started = threading.Event()
    allow_stop_scheduling = threading.Event()

    def gated_call_soon_threadsafe(callback, *args, **kwargs):
        if callback == event_loop.stop:
            stop_scheduling_started.set()
            assert allow_stop_scheduling.wait(timeout=1.0)
        return original_call_soon_threadsafe(callback, *args, **kwargs)

    monkeypatch.setattr(event_loop, "call_soon_threadsafe", gated_call_soon_threadsafe)

    with ThreadPoolExecutor(max_workers=2) as executor:
        shutdown = executor.submit(background.shutdown, 1.0)
        assert stop_scheduling_started.wait(timeout=1.0)
        submitted = executor.submit(background.submit, asyncio.sleep(0, result="late"))
        allow_stop_scheduling.set()
        shutdown.result(timeout=2.0)
        with pytest.raises(RuntimeError, match="stopping|closed"):
            submitted.result(timeout=2.0)


def test_native_background_event_loop_shutdown_stops_published_loop_before_run(monkeypatch):
    original_new_event_loop = asyncio.new_event_loop
    run_forever_entered = threading.Event()
    allow_run_forever = threading.Event()
    stop_scheduled = threading.Event()

    def gated_new_event_loop():
        event_loop = original_new_event_loop()
        original_run_forever = event_loop.run_forever
        original_call_soon_threadsafe = event_loop.call_soon_threadsafe

        def gated_run_forever():
            run_forever_entered.set()
            assert allow_run_forever.wait(timeout=1.0)
            original_run_forever()

        def tracked_call_soon_threadsafe(callback, *args, **kwargs):
            if callback == event_loop.stop:
                stop_scheduled.set()
            return original_call_soon_threadsafe(callback, *args, **kwargs)

        monkeypatch.setattr(event_loop, "run_forever", gated_run_forever)
        monkeypatch.setattr(event_loop, "call_soon_threadsafe", tracked_call_soon_threadsafe)
        return event_loop

    monkeypatch.setattr(asyncio, "new_event_loop", gated_new_event_loop)
    background = _BackgroundEventLoop("native-fte-stop-before-run")
    try:
        background.start()
        assert run_forever_entered.wait(timeout=1.0)
        with ThreadPoolExecutor(max_workers=1) as executor:
            shutdown = executor.submit(background.shutdown, 1.0)
            assert stop_scheduled.wait(timeout=1.0)
            allow_run_forever.set()
            shutdown.result(timeout=2.0)
    finally:
        allow_run_forever.set()
        background.shutdown(timeout_s=1.0)


def test_native_background_event_loop_shutdown_completes_pending_future():
    background = _BackgroundEventLoop("native-fte-cancel-pending")

    async def wait_forever():
        await asyncio.Event().wait()

    future = background.submit(wait_forever())
    background.shutdown(timeout_s=1.0)

    with pytest.raises(CancelledError):
        future.result(timeout=1.0)


def test_native_background_event_loop_request_shutdown_fences_and_shutdown_joins_sync_work():
    background = _BackgroundEventLoop("native-fte-join-sync-work")
    work_started = threading.Event()
    release_work = threading.Event()
    work_finished = threading.Event()
    shutdown_finished = threading.Event()
    shutdown_errors: list[BaseException] = []

    def blocking_work():
        work_started.set()
        try:
            assert release_work.wait(timeout=5.0)
        finally:
            work_finished.set()

    async def run_blocking_work():
        await asyncio.to_thread(blocking_work)

    future = background.submit(run_blocking_work())
    assert work_started.wait(timeout=1.0)
    background.request_shutdown()
    with pytest.raises(RuntimeError, match="stopping|closed"):
        background.submit(asyncio.sleep(0))

    def shutdown():
        try:
            background.shutdown(timeout_s=3.0)
        except BaseException as exc:
            shutdown_errors.append(exc)
        finally:
            shutdown_finished.set()

    shutdown_thread = threading.Thread(target=shutdown)
    shutdown_thread.start()
    try:
        deadline = time.monotonic() + 1.0
        while not future.cancelled() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert future.cancelled()
        assert not shutdown_finished.wait(timeout=0.05)
        release_work.set()
        shutdown_thread.join(timeout=5.0)
    finally:
        release_work.set()
        shutdown_thread.join(timeout=5.0)

    assert not shutdown_thread.is_alive()
    assert shutdown_errors == []
    assert work_finished.is_set()
    with pytest.raises(CancelledError):
        future.result(timeout=1.0)


def test_native_background_event_loop_operation_timeout_cancels_coroutine():
    background = _BackgroundEventLoop(
        "native-fte-operation-timeout",
        operation_timeout_s=0.05,
    )
    coroutine_finished = threading.Event()

    async def wait_forever():
        try:
            await asyncio.Event().wait()
        finally:
            coroutine_finished.set()

    try:
        with pytest.raises(TimeoutError, match="operation timed out"):
            background.run(wait_forever())
        assert coroutine_finished.wait(timeout=1.0)
    finally:
        background.shutdown(timeout_s=1.0)


def test_native_background_event_loop_reloads_result_completed_at_timeout_boundary(monkeypatch):
    class _RacingFuture:
        def __init__(self):
            self.result_calls = []

        def result(self, timeout=None):
            self.result_calls.append(timeout)
            if len(self.result_calls) == 1:
                raise native_backend_mod.FutureTimeoutError
            return "completed"

        def done(self):
            return True

    async def unused():
        return None

    future = _RacingFuture()
    background = _BackgroundEventLoop("native-fte-timeout-boundary")

    def submit(coroutine):
        coroutine.close()
        return future

    monkeypatch.setattr(background, "submit", submit)

    assert background.run(unused(), timeout_s=0.25) == "completed"
    assert future.result_calls == [0.25, None]


def test_native_background_event_loop_catches_pre311_future_timeout(monkeypatch):
    class _Pre311FutureTimeout(Exception):
        pass

    class _PendingFuture:
        def __init__(self):
            self.cancelled = False

        def result(self, timeout=None):
            assert timeout == 0.25
            raise _Pre311FutureTimeout

        def done(self):
            return False

        def cancel(self):
            self.cancelled = True

    async def unused():
        return None

    pending = _PendingFuture()
    background = _BackgroundEventLoop("native-fte-pre311-future-timeout")

    def submit(coroutine):
        coroutine.close()
        return pending

    monkeypatch.setattr(native_backend_mod, "FutureTimeoutError", _Pre311FutureTimeout)
    monkeypatch.setattr(background, "submit", submit)

    with pytest.raises(TimeoutError, match="operation timed out"):
        background.run(unused(), timeout_s=0.25)
    assert pending.cancelled is True


def test_native_worker_handle_shutdown_forwards_timeout(monkeypatch):
    worker = NativeWorkerHandle("worker-shutdown-timeout", lambda _request: None)
    shutdown_timeouts = []
    monkeypatch.setattr(worker._loop, "shutdown", lambda *, timeout_s: shutdown_timeouts.append(timeout_s))

    worker.shutdown(timeout_s=12.5)

    assert shutdown_timeouts == [12.5]


def _captured_native_copy_plan(tmp_path, monkeypatch, *, local_staging: bool):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    if local_staging:
        monkeypatch.setenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", "1")
    else:
        monkeypatch.delenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", raising=False)

    setup_conn = vane.connect()
    src = tmp_path / "native_copy_failure_input.parquet"
    setup_conn.sql("select 1 as x union all select 2 as x").write_parquet(str(src))
    setup_conn.close()

    import vane.runners as runners_mod

    captured = []

    class _CapturingRunner:
        def run_write(self, relation):
            captured.append(relation)
            return {"ok": True}

    monkeypatch.setenv("VANE_RUNNER", "local")
    monkeypatch.setattr(runners_mod, "set_runner_local", lambda *_args, **_kwargs: _CapturingRunner())

    con = vane.connect()
    dst = tmp_path / "native_copy_failure_output.parquet"
    con.sql(f"select * from read_parquet('{src}')").write_parquet(str(dst))
    assert captured, "expected local write relation to be captured"

    query_id = str(uuid.uuid4())
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_write_relation(
        captured[0],
        query_id,
    ).to_physical_plan(con)
    assert plan.scan_split_batch_map()
    return con, dst, query_id, plan


def _capture_native_copy_relation(tmp_path, monkeypatch, *, local_staging: bool):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    if local_staging:
        monkeypatch.setenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", "1")
    else:
        monkeypatch.delenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", raising=False)

    setup_conn = vane.connect()
    src = tmp_path / "native_copy_isolation_input.parquet"
    setup_conn.sql("select 1 as x union all select 2 as x").write_parquet(str(src))
    setup_conn.close()

    import vane.runners as runners_mod

    captured = []

    class _CapturingRunner:
        def run_write(self, relation):
            captured.append(relation)
            return {"ok": True}

    monkeypatch.setenv("VANE_RUNNER", "local")
    monkeypatch.setattr(runners_mod, "set_runner_local", lambda *_args, **_kwargs: _CapturingRunner())

    con = vane.connect()
    dst = tmp_path / "native_copy_isolation_output.parquet"
    con.sql(f"select * from read_parquet('{src}')").write_parquet(str(dst))
    assert captured, "expected local write relation to be captured"
    return con, dst, captured[0]


def _sql_string_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def test_native_worker_handle_reuses_fte_worker_admission_cap():
    started: list[int] = []
    release_first = threading.Event()

    def execute_fn(request):
        partition_id = int(request["task_id"]["partition_id"])
        started.append(partition_id)
        if partition_id == 0:
            release_first.wait(timeout=5.0)
        return {"partition": partition_id}

    worker = NativeWorkerHandle("worker-1", execute_fn, max_running_tasks=1)
    try:
        task0 = _task_id(0)
        task1 = _task_id(1)

        status0 = worker.fte_create_task({"task_id": task0, "fragment_id": "q:scan"})
        status1 = worker.fte_create_task({"task_id": task1, "fragment_id": "q:scan"})

        assert status0["state"] == FteTaskState.RUNNING.value
        assert status1["state"] == FteTaskState.QUEUED.value
        assert status1["executor_running_task_count"] == 1
        assert status1["executor_queued_task_count"] == 1
        assert status1["executor_queue_position"] == 0

        for _ in range(50):
            if started == [0]:
                break
            time.sleep(0.01)
        assert started == [0]

        release_first.set()
        for _ in range(100):
            status1 = worker.fte_get_task_status(task1)
            if started == [0, 1] and status1["state"] == FteTaskState.FINISHED.value:
                break
            time.sleep(0.01)

        assert started == [0, 1]
        assert status1["state"] == FteTaskState.FINISHED.value
        assert status1["executor_running_task_count"] == 0
        assert status1["executor_queued_task_count"] == 0
    finally:
        worker.fte_drop_query("q")
        worker.shutdown()


def test_native_worker_handle_logs_fte_admission_when_enabled(monkeypatch, capsys):
    monkeypatch.setenv("VANE_FTE_ADMISSION_DEBUG", "1")
    started: list[int] = []
    release_first = threading.Event()

    def execute_fn(request):
        partition_id = int(request["task_id"]["partition_id"])
        started.append(partition_id)
        if partition_id == 0:
            release_first.wait(timeout=5.0)
        return {"partition": partition_id}

    worker = NativeWorkerHandle("worker-log", execute_fn, max_running_tasks=1)
    try:
        task0 = _task_id(0, query_id="query-log")
        task1 = _task_id(1, query_id="query-log")

        worker.fte_create_task({"task_id": task0, "fragment_id": "query-log:scan"})
        worker.fte_create_task({"task_id": task1, "fragment_id": "query-log:scan"})

        for _ in range(50):
            if started == [0]:
                break
            time.sleep(0.01)
        assert started == [0]

        release_first.set()
        for _ in range(100):
            status1 = worker.fte_get_task_status(task1)
            snapshot = worker.snapshot()
            if (
                started == [0, 1]
                and status1["state"] == FteTaskState.FINISHED.value
                and not snapshot["executor_running_task_count"]
            ):
                break
            time.sleep(0.01)

        assert started == [0, 1]
    finally:
        worker.fte_drop_query("query-log")
        worker.shutdown()

    captured = capsys.readouterr().err
    assert "[vane-fte-admission" in captured
    assert "worker_id=worker-log" in captured
    assert "event=manager_init" in captured
    assert "event=start_task" in captured
    assert "event=queue_task" in captured
    assert "reason=max_running_tasks" in captured
    assert "reason=drain" in captured
    assert "event=task_done" in captured


def test_native_worker_manager_logs_submit_tasks_when_enabled(monkeypatch, capsys):
    monkeypatch.setenv("VANE_FTE_ADMISSION_DEBUG", "1")

    def execute_fn(request):
        return {"task": request["task_id"]}

    backend = NativeFteWorkerManagerBackend(execute_fn=execute_fn, max_running_tasks=2)
    try:
        handles = backend.submit_tasks(
            [
                {
                    "task_id": _task_id(0, query_id="query-submit-log"),
                    "fragment_id": "query-submit-log:scan",
                    "task_context": {"query_id": "query-submit-log", "task_id": 0},
                },
                {
                    "task_id": _task_id(1, query_id="query-submit-log"),
                    "fragment_id": "query-submit-log:scan",
                    "task_context": {"query_id": "query-submit-log", "task_id": 1},
                },
            ]
        )
        assert len(handles) == 2
        backend.wait_query("query-submit-log", 2.0)
    finally:
        backend.shutdown()

    captured = capsys.readouterr().err
    assert "[vane-fte-native-submit" in captured
    assert "event=submit_tasks_enter" in captured
    assert "event=submit_task_before" in captured
    assert "event=submit_task_after" in captured
    assert "event=submit_tasks_exit" in captured
    assert "batch_size=2" in captured
    assert "submitted_count=2" in captured
    assert "task_id=query-submit-log.0.0.0" in captured
    assert "task_id=query-submit-log.0.1.0" in captured
    assert "worker_id=native-worker-0" in captured
    assert "worker_max_running=2" in captured


def test_native_worker_snapshots_report_resource_capacity():
    def execute_fn(request):
        return {"task": request["task_id"]}

    backend = NativeFteWorkerManagerBackend(
        execute_fn=execute_fn,
        num_workers=2,
        num_cpus=12,
        total_memory_bytes=1200,
    )
    try:
        snapshots = list(backend.worker_snapshots())
    finally:
        backend.shutdown()

    assert [snapshot["worker_id"] for snapshot in snapshots] == ["native-worker-0", "native-worker-1"]
    assert [snapshot["num_cpus"] for snapshot in snapshots] == [6.0, 6.0]
    assert [snapshot["CPU"] for snapshot in snapshots] == [6.0, 6.0]
    assert [snapshot["total_memory_bytes"] for snapshot in snapshots] == [600, 600]
    assert [snapshot["memory"] for snapshot in snapshots] == [600, 600]


def test_native_worker_manager_drop_query_cancels_running_and_queued_tasks():
    started = threading.Event()
    release = threading.Event()
    drop_errors: list[BaseException] = []

    def execute_fn(_request):
        started.set()
        release.wait(timeout=5.0)
        return {"ok": True}

    task0 = _task_id(0, query_id="query-drop")
    task1 = _task_id(1, query_id="query-drop")
    worker = NativeWorkerHandle("worker-1", execute_fn, max_running_tasks=1)
    backend = NativeFteWorkerManagerBackend(workers=[worker])
    try:
        handles = backend.submit_tasks(
            [
                {
                    "task_id": task0,
                    "fragment_id": "query-drop:scan",
                },
                {
                    "task_id": task1,
                    "fragment_id": "query-drop:scan",
                },
            ]
        )
        assert len(handles) == 2
        assert started.wait(timeout=2.0)
        snapshot = backend.worker_snapshots()[0]
        assert snapshot["executor_running_task_count"] == 1
        assert snapshot["executor_queued_task_count"] == 1

        def drop_query():
            try:
                backend.drop_query("query-drop")
            except BaseException as exc:  # pragma: no cover - asserted below
                drop_errors.append(exc)

        drop_thread = threading.Thread(target=drop_query)
        drop_thread.start()
        time.sleep(0.05)

        assert drop_thread.is_alive()
        assert worker.fte_get_task_status_cached(task0)["state"] == FteTaskState.RUNNING.value

        release.set()
        drop_thread.join(timeout=2.0)
        assert not drop_thread.is_alive()
        assert drop_errors == []

        assert backend.pop_fte_result_handles("query-drop") == []
        query_status = backend.fte_query_status("query-drop")
        assert query_status["canceled"] is True
        assert query_status["scheduler_state"] == "CANCELED"
        assert worker.fte_get_task_status(task0)["state"] == FteTaskState.CANCELED.value
        assert worker.fte_get_task_status(task1)["state"] == FteTaskState.CANCELED.value
        snapshot = backend.worker_snapshots()[0]
        assert snapshot["executor_running_task_count"] == 0
        assert snapshot["executor_queued_task_count"] == 0
    finally:
        release.set()
        backend.shutdown()


def test_native_worker_cancel_barrier_outlives_operation_timeout():
    started = threading.Event()
    release = threading.Event()
    background = _BackgroundEventLoop(
        "native-fte-cancel-barrier-timeout",
        operation_timeout_s=0.05,
    )

    def execute_fn(_request):
        started.set()
        assert release.wait(timeout=2.0)
        return {"ok": True}

    worker = NativeWorkerHandle(
        "worker-cancel-barrier-timeout",
        execute_fn,
        loop=background,
    )
    task_id = _task_id(0, query_id="query-cancel-barrier-timeout")
    cancel_results: list[dict[str, Any]] = []
    cancel_errors: list[BaseException] = []
    cancel_thread: threading.Thread | None = None
    cancel_started = threading.Event()

    try:
        worker.fte_create_task(
            {
                "task_id": task_id,
                "fragment_id": "query-cancel-barrier-timeout:copy",
            }
        )
        assert started.wait(timeout=1.0)

        def cancel_task():
            cancel_started.set()
            try:
                cancel_results.append(worker.fte_cancel_task(task_id))
            except BaseException as exc:  # pragma: no cover - asserted below
                cancel_errors.append(exc)

        cancel_thread = threading.Thread(target=cancel_task)
        cancel_thread.start()
        assert cancel_started.wait(timeout=1.0)
        time.sleep(0.15)
        outlived_operation_timeout = cancel_thread.is_alive()
    finally:
        release.set()
        if cancel_thread is not None:
            cancel_thread.join(timeout=2.0)
        try:
            worker.fte_drop_query("query-cancel-barrier-timeout")
        finally:
            background.shutdown()

    assert outlived_operation_timeout
    assert cancel_thread is not None
    assert not cancel_thread.is_alive()
    assert cancel_errors == []
    assert cancel_results[0]["state"] == FteTaskState.CANCELED.value


def test_native_worker_drop_query_barrier_outlives_operation_timeout():
    started = threading.Event()
    release = threading.Event()
    background = _BackgroundEventLoop(
        "native-fte-drop-barrier-timeout",
        operation_timeout_s=0.05,
    )

    def execute_fn(_request):
        started.set()
        assert release.wait(timeout=2.0)
        return {"ok": True}

    worker = NativeWorkerHandle(
        "worker-drop-barrier-timeout",
        execute_fn,
        loop=background,
    )
    query_id = "query-drop-barrier-timeout"
    task_id = _task_id(0, query_id=query_id)
    drop_results: list[dict[str, int]] = []
    drop_errors: list[BaseException] = []
    drop_thread: threading.Thread | None = None
    drop_started = threading.Event()

    try:
        worker.fte_create_task(
            {
                "task_id": task_id,
                "fragment_id": f"{query_id}:copy",
            }
        )
        assert started.wait(timeout=1.0)

        def drop_query():
            drop_started.set()
            try:
                drop_results.append(worker.fte_drop_query(query_id))
            except BaseException as exc:  # pragma: no cover - asserted below
                drop_errors.append(exc)

        drop_thread = threading.Thread(target=drop_query)
        drop_thread.start()
        assert drop_started.wait(timeout=1.0)
        time.sleep(0.15)
        outlived_operation_timeout = drop_thread.is_alive()
    finally:
        release.set()
        if drop_thread is not None:
            drop_thread.join(timeout=2.0)
        try:
            background.run_owned_side_effects(worker._manager.drop_query(query_id))
        finally:
            background.shutdown()

    assert outlived_operation_timeout
    assert drop_thread is not None
    assert not drop_thread.is_alive()
    assert drop_errors == []
    assert drop_results == [{"removed": 1, "canceled": 1}]


def test_native_worker_manager_drop_query_fans_out_after_worker_failure():
    calls = []

    class _ProgressRegistry:
        def drop_query(self, query_id):
            calls.append(("progress", query_id))

    class _Worker:
        def __init__(self, worker_id, *, fail):
            self.worker_id = worker_id
            self.fail = fail

        def fte_drop_query(self, query_id):
            calls.append((self.worker_id, query_id))
            if self.fail:
                raise RuntimeError(f"{self.worker_id} drop failed")
            return {"tasks_removed": 1, "tasks_canceled": 1}

    backend = object.__new__(NativeFteWorkerManagerBackend)
    backend._handles_lock = threading.Lock()
    backend._handles_by_query = {"query-best-effort": [object()]}
    backend._stable_task_identity_lock = threading.Lock()
    backend._stable_task_identity_keys_by_query = {"query-best-effort": {7: "logical-task"}}
    backend._progress_registry = _ProgressRegistry()
    backend._workers = [
        _Worker("worker-dead", fail=True),
        _Worker("worker-live", fail=False),
    ]
    backend._dropped_queries = {}

    with pytest.raises(RuntimeError, match="worker-dead drop failed"):
        backend.drop_query("query-best-effort")

    assert calls == [
        ("progress", "query-best-effort"),
        ("worker-dead", "query-best-effort"),
        ("worker-live", "query-best-effort"),
    ]
    assert "query-best-effort" not in backend._handles_by_query
    assert "query-best-effort" not in backend._stable_task_identity_keys_by_query
    assert backend._dropped_queries["query-best-effort"] == {
        "removed": 1,
        "canceled": 1,
        "worker_errors": ["worker-dead: RuntimeError: worker-dead drop failed"],
    }


def test_native_worker_terminal_task_stats_merge_completed_split_queue_stats():
    def execute_fn(request):
        queue = request["fte_scan_source_queues"]["scan"]
        split = queue.wait_for_next()
        assert split["state"] == "SPLIT"
        return {"task_stats": {"processed_input_rows": 1, "processed_input_bytes": 2}}

    worker = NativeWorkerHandle("worker-1", execute_fn, max_running_tasks=1)
    try:
        task = _task_id(0, query_id="query-split-stats")
        worker.fte_create_task(
            {
                "task_id": task,
                "fragment_id": "query-split-stats:scan",
                "worker_runtime": FTE_WORKER_RUNTIME,
                "dynamic_scan_source_node_ids": ["scan"],
                "initial_splits": {
                    "scan": [
                        {
                            "sequence_id": 1,
                            "kind": "scan_split",
                            "split_id": "scan-1",
                            "data": b"not-a-real-scan-split-batch",
                        }
                    ]
                },
                "no_more_splits": ["scan"],
            }
        )

        for _ in range(100):
            status = worker.fte_get_task_status(task)
            if status["state"] == FteTaskState.FINISHED.value:
                break
            time.sleep(0.01)

        assert status["state"] == FteTaskState.FINISHED.value
        assert status["task_stats"]["processed_input_rows"] == 1
        assert status["task_stats"]["submitted_split_count"] == 1
        assert status["task_stats"]["consumed_split_count"] == 1
        assert status["task_stats"]["completed_split_count"] == 1
        assert status["task_stats"]["queue_wait_ms"] >= 0
    finally:
        worker.fte_drop_query("query-split-stats")
        worker.shutdown()


def test_native_task_result_handle_polls_status_result_and_ack():
    def execute_fn(request):
        return {"ok": request["task_id"]["partition_id"]}

    worker = NativeWorkerHandle("worker-1", execute_fn)
    try:
        task = _task_id(3)
        worker.fte_create_task({"task_id": task, "fragment_id": "q:scan"})
        handle = NativeTaskResultHandle(worker, task, task_context={"query_id": "q", "task_id": 3})

        for _ in range(100):
            poll = handle.poll()
            if poll.state is not TaskResultState.NOT_READY:
                break
            time.sleep(0.01)

        assert handle.task_context() == {"query_id": "q", "task_id": 3}
        assert handle.fte_task_id() == "q.0.3.0"
        assert handle.worker_id == "worker-1"
        assert handle.worker_id() == "worker-1"
        assert handle.exchange_node_id == _flight_exchange_node_id_from_env()
        assert handle.exchange_node_id() == _flight_exchange_node_id_from_env()
        assert handle.task_id.query_id == "q"
        assert handle.task_id.fragment_execution_id == 0
        assert handle.task_id.partition_id == 3
        assert handle.task_id.attempt_id == 0
        assert handle.task_context_info == {
            "query_idx": 0,
            "last_node_id": 3,
            "task_id": 3,
            "node_ids": [3],
        }
        assert poll.state is TaskResultState.MATERIALIZED_OUTPUT
        assert poll.output == {"ok": 3}
        assert handle.done() is True
        cxx_result = handle.get_result_sync()
        assert cxx_result.ok is True
        assert cxx_result.has_output is True
        assert handle.acked is False
        handle.ack()
        assert handle.acked is True
    finally:
        worker.fte_drop_query("q")
        worker.shutdown()


def test_native_task_result_handle_rejects_mismatched_status_identity():
    expected_task = _task_id(0, query_id="query-native-status-identity")
    mismatched_task = _task_id(1, query_id="query-native-status-identity")
    callback_events = []

    class _MismatchedWorker:
        worker_id = "native-worker-mismatched-status"

        def fte_get_task_status_cached(self, _task_id):
            return {
                "state": FteTaskState.FINISHED.value,
                "task_id": expected_task,
                "task_id_string": FteTaskAttemptId.coerce(mismatched_task).__str__(),
                "result": "wrong-output",
            }

        def fte_get_task_info(self, _task_id):
            return {
                "status": {
                    "state": FteTaskState.FINISHED.value,
                    "task_id": expected_task,
                    "task_id_string": str(FteTaskAttemptId.coerce(mismatched_task)),
                    "result": "wrong-output",
                }
            }

    handle = NativeTaskResultHandle(
        _MismatchedWorker(),
        expected_task,
        status_callback=lambda _handle, status, error: callback_events.append((dict(status), error)),
    )

    poll = handle.poll()
    assert poll.state is TaskResultState.ERROR
    assert poll.output is None
    assert poll.error is not None
    assert "status identity mismatch" in str(poll.error)
    assert callback_events[-1][0]["state"] == FteTaskState.FAILED.value
    assert callback_events[-1][0]["failure"]["error_code"] == "NATIVE_BACKEND_ERROR"
    assert callback_events[-1][1] is poll.error

    with pytest.raises(RuntimeError, match="status identity mismatch"):
        handle.status_snapshot()
    assert callback_events[-1][1] is not None

    with pytest.raises(RuntimeError, match="status identity mismatch"):
        handle.info_snapshot()
    assert callback_events[-1][1] is not None


def test_native_task_result_handle_validates_enum_terminal_failure_payload():
    task = _task_id(0, query_id="query-native-enum-terminal")

    class _EnumTerminalWorker:
        worker_id = "native-worker-enum-terminal"

        def fte_get_task_status_cached(self, _task_id):
            return {
                "state": FteTaskState.FAILED,
                "task_id": task,
                "failure": {"message": "out of memory"},
            }

    handle = NativeTaskResultHandle(_EnumTerminalWorker(), task)

    with pytest.raises(ValueError, match="requires error_code"):
        handle.status_snapshot()


def test_native_backend_rejects_mismatched_create_task_identity():
    expected_task = _task_id(0, query_id="query-native-create-identity")
    mismatched_task = _task_id(1, query_id="query-native-create-identity")

    class _MismatchedCreateWorker:
        worker_id = "native-worker-mismatched-create"

        def fte_create_task(self, _request):
            return {
                "state": FteTaskState.RUNNING.value,
                "task_id": mismatched_task,
            }

        def snapshot(self):
            return {
                "worker_id": self.worker_id,
                "executor_running_task_count": 0,
                "executor_queued_task_count": 0,
                "executor_max_running_tasks": 1,
            }

        def shutdown(self):
            pass

    backend = NativeFteWorkerManagerBackend(workers=[_MismatchedCreateWorker()])
    try:
        with pytest.raises(RuntimeError, match="status identity mismatch"):
            backend.submit_tasks(
                [
                    {
                        "task_id": expected_task,
                        "fragment_id": "query-native-create-identity:scan",
                    }
                ]
            )
    finally:
        backend.shutdown()


def test_native_task_result_handle_uses_cached_status_and_release_clears_result(monkeypatch):
    def execute_fn(_request):
        return {"payload": "large-result"}

    worker = NativeWorkerHandle("worker-cache", execute_fn)
    try:
        task = _task_id(31, query_id="query-cache")
        worker.fte_create_task({"task_id": task, "fragment_id": "query-cache:scan"})
        handle = NativeTaskResultHandle(worker, task)

        for _ in range(100):
            poll = handle.poll()
            if poll.state is not TaskResultState.NOT_READY:
                break
            time.sleep(0.01)

        def fail_blocking_status(_task_id):
            raise AssertionError("blocking status API should not be used by native result handles")

        monkeypatch.setattr(worker, "fte_get_task_status", fail_blocking_status)

        status = handle.status_snapshot()
        assert status["state"] == FteTaskState.FINISHED.value
        assert status["result"] == {"payload": "large-result"}
        poll = handle.poll()
        assert poll.state is TaskResultState.MATERIALIZED_OUTPUT
        assert poll.output == {"payload": "large-result"}

        handle.ack()
        assert handle.acked is True
        assert handle.status_snapshot()["result"] == {"payload": "large-result"}
        handle.release_result_payload()
        assert handle.status_snapshot().get("result") is None
    finally:
        worker.fte_drop_query("query-cache")
        worker.shutdown()


def test_native_worker_manager_backend_submit_wait_drop_and_shutdown():
    def execute_fn(request):
        return {"task": request["task_id"], "context": request.get("task_context")}

    backend = NativeFteWorkerManagerBackend(execute_fn=execute_fn, max_running_tasks=2)
    try:
        handles = backend.submit_tasks(
            [
                {
                    "task_id": _task_id(0, query_id="query-a"),
                    "fragment_id": "query-a:scan",
                    "task_context": {"query_id": "query-a", "task_id": 0},
                },
                {
                    "task_id": _task_id(1, query_id="query-a"),
                    "fragment_id": "query-a:scan",
                    "task_context": {"query_id": "query-a", "task_id": 1},
                },
            ]
        )

        assert len(handles) == 2
        assert backend.worker_snapshots()[0]["worker_id"] == "native-worker-0"

        outputs = backend.wait_query("query-a", 2.0)
        assert [output["task"]["partition_id"] for output in outputs] == [0, 1]
        assert all(handle.acked for handle in handles)
        assert all(handle.status_snapshot().get("result") is None for handle in handles)

        backend.drop_query("query-a")
        assert backend.wait_query("query-a", 0.0) == []
    finally:
        backend.shutdown()

    with pytest.raises(RuntimeError, match="shut down"):
        backend.submit_tasks([])


def test_native_worker_manager_shutdown_attempts_all_workers_and_is_retryable():
    shutdown_calls = []

    class _Worker:
        def __init__(self, worker_id, *, fail_once=False):
            self.worker_id = worker_id
            self.fail_once = fail_once

        def shutdown(self):
            shutdown_calls.append(self.worker_id)
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError(f"{self.worker_id} loop did not stop")

    backend = NativeFteWorkerManagerBackend(
        workers=[
            _Worker("first", fail_once=True),
            _Worker("second"),
        ]
    )

    with pytest.raises(RuntimeError, match="first.*loop did not stop"):
        backend.shutdown()

    assert shutdown_calls == ["first", "second"]
    assert backend._closed is False
    with pytest.raises(RuntimeError, match="shutting down"):
        backend.submit_tasks([])

    backend.shutdown()

    assert shutdown_calls == ["first", "second", "first", "second"]
    assert backend._closed is True


def test_native_worker_manager_shutdown_shares_timeout_across_workers(monkeypatch):
    shutdown_timeouts = []

    class _Worker:
        def __init__(self, worker_id):
            self.worker_id = worker_id

        def shutdown(self, *, timeout_s):
            shutdown_timeouts.append((self.worker_id, timeout_s))

    monotonic_values = iter([10.0, 10.25, 10.75])
    monkeypatch.setattr(native_backend_mod.time, "monotonic", lambda: next(monotonic_values))
    backend = NativeFteWorkerManagerBackend(workers=[_Worker("first"), _Worker("second")])

    backend.shutdown(timeout_s=1.0)

    assert shutdown_timeouts == [
        ("first", pytest.approx(0.75)),
        ("second", pytest.approx(0.25)),
    ]
    assert backend._closed is True


def test_native_worker_manager_exposes_ray_compatible_query_status_and_handles():
    release = threading.Event()

    def execute_fn(request):
        release.wait(timeout=5.0)
        return {"partition": request["task_id"]["partition_id"]}

    backend = NativeFteWorkerManagerBackend(execute_fn=execute_fn)
    try:
        backend.submit_tasks(
            [
                {
                    "task_id": _task_id(0, query_id="query-a"),
                    "fragment_id": "query-a:scan",
                    "task_context": {
                        "query_idx": 1,
                        "last_node_id": 7,
                        "task_id": 9,
                        "node_ids": [7],
                    },
                }
            ]
        )

        status = backend.fte_query_status("query-a")
        assert status["failed"] is False
        assert status["finished"] is False
        assert status["running_count"] == 1

        release.set()
        status = backend.wait_fte_query("query-a", 2.0)
        assert status["finished"] is True
        assert status["selected_attempt_task_ids"] == ["query-a.0.0.0"]

        handles = backend.pop_fte_result_handles("query-a")
        assert len(handles) == 1
        assert handles[0].task_context_info == {
            "query_idx": 1,
            "last_node_id": 7,
            "task_id": 9,
            "node_ids": [7],
        }
        assert handles[0].get_result_sync().ok is True
        assert backend.pop_fte_result_handles("query-a") == []
    finally:
        backend.drop_query("query-a")
        backend.shutdown()


def test_native_worker_manager_scopes_query_status_by_task_context():
    release_second = threading.Event()
    first_context = {
        "query_idx": 1,
        "last_node_id": 7,
        "task_id": 9,
        "node_ids": [7],
    }
    second_context = {
        "query_idx": 1,
        "last_node_id": 7,
        "task_id": 10,
        "node_ids": [7],
    }

    def execute_fn(request):
        if request["task_id"]["partition_id"] == 1:
            release_second.wait(timeout=5.0)
        return {"partition": request["task_id"]["partition_id"]}

    backend = NativeFteWorkerManagerBackend(execute_fn=execute_fn, max_running_tasks=2)
    try:
        backend.submit_tasks(
            [
                {
                    "task_id": _task_id(0, query_id="query-scoped"),
                    "fragment_id": "query-scoped:scan",
                    "task_context": first_context,
                },
                {
                    "task_id": _task_id(1, query_id="query-scoped"),
                    "fragment_id": "query-scoped:scan",
                    "task_context": second_context,
                },
            ]
        )

        deadline = time.monotonic() + 2.0
        while True:
            first_status = backend.fte_query_status("query-scoped", [first_context])
            if first_status["finished"]:
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)

        assert first_status["matched"] is True
        assert first_status["partition_count"] == 1
        assert first_status["selected_attempt_task_ids"] == ["query-scoped.0.0.0"]
        assert len(backend.wait_query("query-scoped", 1.0, [first_context])) == 1

        global_status = backend.fte_query_status("query-scoped")
        assert global_status["finished"] is False
        assert global_status["partition_count"] == 2

        unmatched_status = backend.fte_query_status(
            "query-scoped",
            [
                {
                    "query_idx": 1,
                    "last_node_id": 7,
                    "task_id": 99,
                    "node_ids": [7],
                }
            ],
        )
        assert unmatched_status["matched"] is False
        assert unmatched_status["finished"] is False
        assert unmatched_status["partition_count"] == 0
    finally:
        release_second.set()
        backend.drop_query("query-scoped")
        backend.shutdown()


def test_native_worker_manager_scoped_query_status_keeps_failures_global():
    release_first = threading.Event()
    first_context = {
        "query_idx": 2,
        "last_node_id": 8,
        "task_id": 11,
        "node_ids": [8],
    }
    second_context = {
        "query_idx": 2,
        "last_node_id": 8,
        "task_id": 12,
        "node_ids": [8],
    }

    def execute_fn(request):
        if request["task_id"]["partition_id"] == 0:
            release_first.wait(timeout=5.0)
            return {"partition": 0}
        raise RuntimeError("other scoped fragment failed")

    backend = NativeFteWorkerManagerBackend(execute_fn=execute_fn, max_running_tasks=2)
    try:
        backend.submit_tasks(
            [
                {
                    "task_id": _task_id(0, query_id="query-scoped-failure"),
                    "fragment_id": "query-scoped-failure:scan",
                    "task_context": first_context,
                },
                {
                    "task_id": _task_id(1, query_id="query-scoped-failure"),
                    "fragment_id": "query-scoped-failure:scan",
                    "task_context": second_context,
                },
            ]
        )

        deadline = time.monotonic() + 2.0
        while True:
            scoped_status = backend.fte_query_status("query-scoped-failure", [first_context])
            if scoped_status["failed"]:
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)

        assert scoped_status["matched"] is True
        assert scoped_status["finished"] is False
        assert scoped_status["failed_count"] == 0
        assert scoped_status["failed_partitions"] == []
    finally:
        release_first.set()
        backend.drop_query("query-scoped-failure")
        backend.shutdown()


def test_native_worker_manager_task_input_stream_exhausted_seals_fte_runtime_sources():
    executed = threading.Event()

    def execute_fn(request):
        executed.set()
        return {"splits": request["initial_splits"]}

    backend = NativeFteWorkerManagerBackend(execute_fn=execute_fn)
    try:
        backend.submit_tasks(
            [
                {
                    "task_id": _task_id(0, query_id="query-a"),
                    "fragment_id": "query-a:scan",
                    "worker_runtime": "fte",
                    "source_node_ids": ["7"],
                    "initial_splits": {
                        "7": [{"sequence_id": 0, "kind": "scan_split", "split_id": "scan-0", "data": b"a"}],
                    },
                }
            ]
        )

        time.sleep(0.02)
        assert executed.is_set() is False

        backend.task_input_stream_exhausted("query-a", ["7"])
        outputs = backend.wait_query("query-a", 2.0)

        assert executed.is_set() is True
        assert outputs[0]["splits"]["7"][0]["data"] == b"a"
    finally:
        backend.drop_query("query-a")
        backend.shutdown()


def test_native_worker_task_request_converts_inputs_to_dynamic_splits():
    task = _FakeNativeWorkerTask(
        name="Repartition",
        context={"query_id": "query-dynamic", "node_id": "3"},
        task_context={"task_id": 9, "last_node_id": "3"},
        inputs={
            "1": {
                "kind": "scan_split_batch",
                "data": _scan_split_batch("scan-0", b"scan-split-payload"),
            },
            "3": {
                "kind": "exchange_source_task",
                "data": {
                    "partition_indices": [0, 1],
                    "source_partition_count": 2,
                    "source_task_count": 2,
                },
            },
        },
        plan={"plan": "template"},
    )

    request = NativeFteWorkerManagerBackend._request_from_task(task)

    assert request["worker_runtime"] == FTE_WORKER_RUNTIME
    assert request["fragment_id"] == "query-dynamic:Repartition:3"
    assert request["source_node_ids"] == ["1", "3"]
    assert request["dynamic_scan_source_node_ids"] == ["1"]
    assert request["dynamic_exchange_source_node_ids"] == ["3"]
    assert "scan_split_batch:1" not in request["context"]
    assert "exchange_source_task:3" not in request["context"]
    assert "scan_split_batch_nodes" not in request["context"]
    assert "exchange_source_task_nodes" not in request["context"]

    scan_splits = request["initial_splits"]["1"]
    assert len(scan_splits) == 1
    assert scan_splits[0]["kind"] == "scan_split"
    assert scan_splits[0]["sequence_id"] == 0
    assert scan_splits[0]["split_id"] == "scan-0"
    assert scan_splits[0]["data"] == _scan_split_batch("scan-0", b"scan-split-payload")

    exchange_splits = request["initial_splits"]["3"]
    assert [split["sequence_id"] for split in exchange_splits] == [0, 1]
    assert [split["source_partition_id"] for split in exchange_splits] == [0, 1]
    assert [split["data"]["partition_indices"] for split in exchange_splits] == [[0], [1]]


def test_native_worker_task_request_derives_fte_exchange_sink_identity():
    def make_request(
        *,
        source_task_partition_id: int,
        event_task_id: int,
        attempt_id: int,
        physical_suffix: str,
    ) -> dict[str, Any]:
        exchange_source = vane.ray_cxx.make_exchange_source_task_descriptor_for_test(
            [
                {
                    "partition_id": 0,
                    "source_task_partition_id": source_task_partition_id,
                    "attempt_id": attempt_id,
                    "node_id": f"node-{physical_suffix}",
                    "flight_host": f"host-{physical_suffix}",
                    "flight_port": 5000 + attempt_id,
                    "flight_server_epoch": f"epoch-{physical_suffix}",
                    "files": [
                        {
                            "path": f"random-{physical_suffix}__sink_{source_task_partition_id}__attempt_{attempt_id}",
                            "file_size": 11,
                        }
                    ],
                }
            ],
            [0],
            1,
            1,
        )
        task = _FakeNativeWorkerTask(
            context={
                "query_id": "query-sink-identity",
                "node_id": "3",
                "fragment_execution_id": 4,
                "attempt_id": attempt_id,
            },
            task_context={
                "query_idx": 0,
                "last_node_id": 3,
                "task_id": event_task_id,
                "node_ids": [3],
            },
            inputs={
                "3": {
                    "kind": "exchange_source_task",
                    "data": exchange_source,
                }
            },
            exchange_sink_config={
                "query_id": "query-sink-identity",
                "output_partition_count": 1,
                "output_location_prefix": "materialized",
            },
        )
        return NativeFteWorkerManagerBackend._request_from_task(task)

    first_order = make_request(
        source_task_partition_id=41,
        event_task_id=7,
        attempt_id=0,
        physical_suffix="first",
    )
    reversed_order = make_request(
        source_task_partition_id=41,
        event_task_id=99,
        attempt_id=2,
        physical_suffix="retry",
    )
    different_source = make_request(
        source_task_partition_id=42,
        event_task_id=7,
        attempt_id=0,
        physical_suffix="second",
    )

    stable_identity = first_order["exchange_sink_instance"]["sink_handle"]["task_partition_id"]
    assert reversed_order["exchange_sink_instance"]["sink_handle"]["task_partition_id"] == stable_identity
    assert different_source["exchange_sink_instance"]["sink_handle"]["task_partition_id"] != stable_identity
    assert stable_identity != (4 << 32) | 7
    assert reversed_order["exchange_sink_instance"]["attempt_id"] == 2
    assert reversed_order["exchange_sink_instance"]["output_location"] == (
        f"materialized__sink_{stable_identity}__attempt_2"
    )
    assert first_order["exchange_sink_instance"]["sink_handle"]["task_partition_id"] == stable_identity


def test_native_worker_task_request_derives_stable_task_sink_identity_from_inputs():
    def make_request(event_task_id: int, source_task_partition_id: int) -> dict[str, Any]:
        task = _FakeNativeWorkerTask(
            context={"query_id": "query-plan-derived", "node_id": "4"},
            task_context={
                "query_idx": 0,
                "last_node_id": 4,
                "task_id": event_task_id,
                "node_ids": [3, 4],
            },
            inputs={
                "3": {
                    "kind": "exchange_source_task",
                    "data": {
                        "partition_indices": [0],
                        "source_task_partition_ids": [source_task_partition_id],
                        "source_partition_count": 1,
                        "source_task_count": 1,
                    },
                }
            },
            exchange_sink_config={
                "query_id": "query-plan-derived",
                "output_partition_count": 1,
                "output_location_prefix": "shuffle",
            },
        )
        return NativeFteWorkerManagerBackend._request_from_task(task)

    first_order = make_request(7, 41)
    reversed_order = make_request(99, 41)
    different_source = make_request(7, 42)

    first_identity = first_order["exchange_sink_instance"]["sink_handle"]["task_partition_id"]
    assert reversed_order["exchange_sink_instance"]["sink_handle"]["task_partition_id"] == first_identity
    assert different_source["exchange_sink_instance"]["sink_handle"]["task_partition_id"] != first_identity
    assert first_order["exchange_sink_instance"]["output_location"] == (f"shuffle__sink_{first_identity}__attempt_0")


def test_native_worker_task_request_distinguishes_scan_splits_by_stable_id():
    def make_request(split_id: str, event_task_id: int) -> dict[str, Any]:
        task = _FakeNativeWorkerTask(
            context={"query_id": "query-duplicate-scan", "node_id": "4"},
            task_context={
                "query_idx": 0,
                "last_node_id": 4,
                "task_id": event_task_id,
                "node_ids": [3, 4],
            },
            inputs={
                "3": {
                    "kind": "scan_split_batch",
                    "data": _scan_split_batch(split_id, b"same-file-payload"),
                }
            },
            exchange_sink_config={
                "query_id": "query-duplicate-scan",
                "output_partition_count": 1,
                "output_location_prefix": "shuffle",
            },
        )
        return NativeFteWorkerManagerBackend._request_from_task(task)

    first = make_request("file-0", 99)
    same_logical_split = make_request("file-0", 12)
    repeated_occurrence = make_request("file-1", 98)

    assert (
        first["exchange_sink_instance"]["sink_handle"]["task_partition_id"]
        == same_logical_split["exchange_sink_instance"]["sink_handle"]["task_partition_id"]
    )
    assert (
        first["exchange_sink_instance"]["sink_handle"]["task_partition_id"]
        != repeated_occurrence["exchange_sink_instance"]["sink_handle"]["task_partition_id"]
    )


def test_native_worker_task_request_uses_explicit_static_source_partition_identity():
    def make_request(event_task_id: int) -> dict[str, Any]:
        task = _FakeNativeWorkerTask(
            context={
                "query_id": "query-static-source",
                "node_id": "4",
                "stable_task_partition_id": "3",
            },
            task_context={
                "query_idx": 0,
                "last_node_id": 4,
                "task_id": event_task_id,
                "node_ids": [3, 4],
            },
            exchange_sink_config={
                "query_id": "query-static-source",
                "output_partition_count": 1,
                "output_location_prefix": "shuffle",
            },
        )
        return NativeFteWorkerManagerBackend._request_from_task(task)

    first_order = make_request(7)
    reversed_order = make_request(99)

    assert (
        reversed_order["exchange_sink_instance"]["sink_handle"]["task_partition_id"]
        == (first_order["exchange_sink_instance"]["sink_handle"]["task_partition_id"])
    )


def test_native_worker_task_request_uses_scheduler_identity_for_source_free_sink():
    def make_request(task_id: int, attempt_id: int) -> dict[str, Any]:
        task = _FakeNativeWorkerTask(
            context={
                "query_id": "query-source-free-sink",
                "node_id": "3",
                "fragment_execution_id": 4,
                "attempt_id": attempt_id,
            },
            task_context={
                "query_idx": 0,
                "last_node_id": 3,
                "task_id": task_id,
                "node_ids": [3],
            },
            exchange_sink_config={
                "query_id": "query-source-free-sink",
                "output_partition_count": 1,
                "output_location_prefix": "range",
            },
        )
        return NativeFteWorkerManagerBackend._request_from_task(task)

    first = make_request(7, 0)
    retry = make_request(7, 2)
    other_task = make_request(8, 0)

    stable_identity = first["exchange_sink_instance"]["sink_handle"]["task_partition_id"]
    assert retry["exchange_sink_instance"]["sink_handle"]["task_partition_id"] == stable_identity
    assert other_task["exchange_sink_instance"]["sink_handle"]["task_partition_id"] != stable_identity
    assert retry["exchange_sink_instance"]["attempt_id"] == 2
    assert retry["exchange_sink_instance"]["output_location"] == f"range__sink_{stable_identity}__attempt_2"


def test_native_worker_manager_rejects_stable_task_identity_collisions():
    backend = NativeFteWorkerManagerBackend(execute_fn=lambda request: request)
    identity_key_field = native_backend_mod._NATIVE_STABLE_TASK_IDENTITY_KEY
    requests = [
        {
            "task_id": _task_id(1, query_id="query-collision"),
            "exchange_sink_instance": {"sink_handle": {"task_partition_id": 123}},
            identity_key_field: "logical-task-a",
        },
        {
            "task_id": _task_id(2, query_id="query-collision"),
            "exchange_sink_instance": {"sink_handle": {"task_partition_id": 123}},
            identity_key_field: "logical-task-b",
        },
    ]
    try:
        with pytest.raises(ValueError, match="stable native FTE task identity collision"):
            backend._register_stable_task_identities(requests)
    finally:
        backend.shutdown()


def test_native_fte_runtime_starts_dynamic_source_and_removes_initial_splits_before_execute():
    executed = threading.Event()
    captured: list[dict[str, Any]] = []

    def execute_fn(request):
        captured.append(dict(request))
        executed.set()
        return {"ok": True}

    backend = NativeFteWorkerManagerBackend(execute_fn=execute_fn)
    try:
        backend.submit_tasks(
            [
                {
                    "task_id": _task_id(0, query_id="query-dynamic-scan"),
                    "fragment_id": "query-dynamic-scan:scan",
                    "worker_runtime": FTE_WORKER_RUNTIME,
                    "source_node_ids": ["7"],
                    "dynamic_scan_source_node_ids": ["7"],
                    "initial_splits": {
                        "7": [{"sequence_id": 0, "kind": "scan_split", "split_id": "scan-0", "data": b"scan"}],
                    },
                }
            ]
        )

        assert executed.wait(timeout=1.0)
        outputs = backend.wait_query("query-dynamic-scan", 2.0)

        assert outputs == [{"ok": True}]
        assert captured[0]["initial_splits"] == {}
        assert captured[0]["dynamic_scan_source_node_ids"] == ["7"]
        assert "fte_scan_source_queues" in captured[0]
        assert "7" in captured[0]["fte_scan_source_queues"]
    finally:
        backend.drop_query("query-dynamic-scan")
        backend.shutdown()


def test_native_task_result_handle_normalizes_native_distributed_result_for_cxx():
    ray_cxx = vane.ray_cxx

    def execute_fn(_request):
        return ray_cxx.NativeDistributedTaskResult(
            ["payload"],
            [ray_cxx.NativePartitionMetadata(2, 16)],
            {"names": ["value"], "types": ["INTEGER"]},
            [1, 2, 3],
            "FINISHED",
            123,
            {"sink": "attempt-0"},
        )

    worker = NativeWorkerHandle("worker-1", execute_fn)
    try:
        task = _task_id(4)
        worker.fte_create_task({"task_id": task, "fragment_id": "q:scan"})
        handle = NativeTaskResultHandle(worker, task)

        for _ in range(100):
            if handle.done():
                break
            time.sleep(0.01)

        result = handle.get_result_sync()
        assert result.ok is True
        assert result.has_output is True
        assert result.flight_port == 123
        assert result.result_schema == {"names": ["value"], "types": ["INTEGER"]}
        assert result.exchange_sink_instance == {"sink": "attempt-0"}
    finally:
        worker.fte_drop_query("q")
        worker.shutdown()


def test_native_task_result_handle_normalizes_tuple_with_completion_status_for_cxx():
    def execute_fn(_request):
        return (
            ["payload"],
            [(5, 64)],
            {"names": ["value"], "types": ["VARCHAR"]},
            [9],
            "FINISHED",
            77,
            {"sink": "attempt-1"},
            {"rows": 5},
        )

    worker = NativeWorkerHandle("worker-1", execute_fn)
    try:
        task = _task_id(5)
        worker.fte_create_task({"task_id": task, "fragment_id": "q:scan"})
        handle = NativeTaskResultHandle(worker, task)

        for _ in range(100):
            if handle.done():
                break
            time.sleep(0.01)

        result = handle.get_result_sync()
        assert result.ok is True
        assert result.has_output is True
        assert result.flight_port == 77
        assert result.result_schema == {"names": ["value"], "types": ["VARCHAR"]}
        assert result.exchange_sink_instance == {"sink": "attempt-1"}
    finally:
        worker.fte_drop_query("q")
        worker.shutdown()


def test_cxx_python_task_result_handle_polls_native_handle_without_ray_driver():
    def execute_fn(_request):
        return (
            [],
            [],
            None,
            [],
            "FINISHED",
            55,
            None,
            {},
        )

    worker = NativeWorkerHandle("worker-1", execute_fn)
    try:
        task = _task_id(8)
        worker.fte_create_task(
            {
                "task_id": task,
                "fragment_id": "q:scan",
            }
        )
        handle = NativeTaskResultHandle(
            worker,
            task,
            task_context={
                "query_idx": 2,
                "last_node_id": 4,
                "task_id": 6,
                "node_ids": [4],
            },
        )

        result = vane.ray_cxx.python_task_result_handle_for_test(handle)

        assert result == {
            "worker_id": _flight_exchange_node_id_from_env(),
            "has_output": True,
            "flight_port": 55,
        }
        assert handle.acked is True
        assert handle.status_snapshot().get("result") is None
    finally:
        worker.fte_drop_query("q")
        worker.shutdown()


def test_cxx_python_task_result_handle_requires_ack_contract():
    class MissingAckHandle:
        worker_id = "worker-missing-ack"
        task_id = FteTaskAttemptId.coerce(_task_id(9))
        task_context_info = {
            "query_idx": 2,
            "last_node_id": 4,
            "task_id": 9,
            "node_ids": [4],
        }

        def done(self):
            return True

        def get_result_sync(self):
            return vane.ray_cxx.RayTaskResult.success([], [], None, 55)

        def release_result_payload(self):
            return None

    with pytest.raises(Exception, match="ack"):
        vane.ray_cxx.python_task_result_handle_for_test(MissingAckHandle())


def test_cxx_distributed_runner_accepts_python_backend_without_ray_worker_startup():
    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.worker_snapshots_calls = 0
            self.shutdown_calls = 0

        def worker_snapshots(self):
            self.worker_snapshots_calls += 1
            return [
                {
                    "worker_id": "native-worker-0",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024,
                }
            ]

        def shutdown(self):
            self.shutdown_calls += 1

    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)

    runner.warm_up()
    stats = runner.fragment_stats()
    runner.shutdown()

    assert backend.worker_snapshots_calls == 1
    assert backend.shutdown_calls == 1
    assert stats["workers"] == {}
    assert stats["totals"] == {}


def test_cxx_distributed_runner_reads_native_backend_fragment_stats():
    started = threading.Event()
    release = threading.Event()

    def execute_fn(_request):
        started.set()
        release.wait(timeout=5.0)
        return None

    backend = NativeFteWorkerManagerBackend(execute_fn=execute_fn, max_running_tasks=1)
    try:
        backend.submit_tasks(
            [
                {
                    "task_id": _task_id(0, query_id="query-stats"),
                    "fragment_id": "query-stats:scan",
                },
                {
                    "task_id": _task_id(1, query_id="query-stats"),
                    "fragment_id": "query-stats:scan",
                },
            ]
        )
        assert started.wait(timeout=2.0)

        runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
        stats = runner.fragment_stats()

        assert stats["workers"]["native-worker-0"]["executor_running_task_count"] == 1
        assert stats["workers"]["native-worker-0"]["executor_queued_task_count"] == 1
        assert stats["workers"]["native-worker-0"]["executor_max_running_tasks"] == 1
        assert stats["workers"]["native-worker-0"]["executor_admission_limited"] == 1
        assert stats["totals"]["executor_running_task_count"] == 1
        assert stats["totals"]["executor_queued_task_count"] == 1
    finally:
        release.set()
        backend.drop_query("query-stats")
        backend.shutdown()


def test_native_backend_query_status_builds_local_progress_snapshot():
    started = threading.Event()
    release = threading.Event()

    def execute_fn(request):
        callback = request.get("native_progress_callback")
        if callback is not None:
            callback(
                {
                    "processed_input_rows": 5,
                    "physical_input_bytes": 128,
                    "total_pipeline_tasks": 2,
                    "queued_pipeline_tasks": 0,
                    "running_pipeline_tasks": 1,
                    "completed_pipeline_tasks": 1,
                    "pipelines": [
                        {
                            "pipeline_id": 1,
                            "operators": ["TABLE_SCAN"],
                            "operator_details": [{}],
                            "input_rows": 5,
                            "input_bytes": 128,
                            "output_rows": 5,
                            "output_bytes": 128,
                            "total_pipeline_tasks": 2,
                            "queued_pipeline_tasks": 0,
                            "running_pipeline_tasks": 1,
                            "completed_pipeline_tasks": 1,
                        }
                    ],
                }
            )
        started.set()
        release.wait(timeout=5.0)
        return {"ok": True}

    backend = NativeFteWorkerManagerBackend(execute_fn=execute_fn, max_running_tasks=1)
    try:
        backend.submit_tasks(
            [
                {
                    "task_id": _task_id(0, query_id="query-progress"),
                    "fragment_id": "query-progress:scan",
                    "dynamic_scan_source_node_ids": ["1"],
                }
            ]
        )
        assert started.wait(timeout=2.0)

        query_status = backend.fte_query_status("query-progress")
        fragment = query_status["fragment_executions"]["query-progress:scan"]
        partition = fragment["partitions"]["0"]
        assert fragment["running_count"] == 1
        assert fragment["pending_submission_count"] == 0
        assert fragment["progress_topology"] == {
            "schema": "pipeline_topology",
            "pipelines": [
                {
                    "pipeline_id": 1,
                    "operators": ["TABLE_SCAN"],
                    "operator_details": [{}],
                }
            ],
        }
        assert partition["running_attempts"]
        assert partition["running_attempts"][0]["task_stats"]["processed_input_rows"] == 5

        snapshot = build_progress_snapshot({"queries": {"query-progress": query_status}}, "query-progress")
        assert snapshot["running_pipeline_tasks"] == 1
        assert snapshot["total_pipeline_tasks"] == 2
        assert snapshot["processed_rows"] == 5
        assert snapshot["fragments"][0]["pipelines"][0]["processed_rows"] == 5
    finally:
        release.set()
        backend.drop_query("query-progress")
        backend.shutdown()


def test_native_progress_topology_ignores_live_counters_and_merges_stable_identity():
    fragment = _NativeFteRegisteredFragment(
        query_id="query-progress-topology",
        fragment_id="query-progress-topology:scan",
        fragment_execution_id=0,
    )

    def metrics(counter: str, *, include_udf_name: bool, include_copy_pipeline: bool):
        udf_details = {
            "pipeline_role": "source",
            "udf_completed_input_rows": counter,
        }
        if include_udf_name:
            udf_details["udf_name"] = "ai_prompt"
        pipelines = [
            {
                "pipeline_id": 1,
                "operators": ["STREAMING_UDF"],
                "operator_details": [udf_details],
            }
        ]
        if include_copy_pipeline:
            pipelines.append(
                {
                    "pipeline_id": 2,
                    "operators": ["COPY_TO_FILE"],
                    "operator_details": [{}],
                }
            )
        return {"running_attempts": [{"task_stats": {"pipelines": pipelines}}]}

    initial_metrics = metrics("0", include_udf_name=False, include_copy_pipeline=False)
    updated_metrics = metrics("1", include_udf_name=True, include_copy_pipeline=True)
    _NativeFteProgressRegistry._merge_fragment_progress_topology_locked(fragment, initial_metrics)
    _NativeFteProgressRegistry._merge_fragment_progress_topology_locked(fragment, updated_metrics)

    assert fragment.progress_topology_unavailable is False
    assert (
        updated_metrics["running_attempts"][0]["task_stats"]["pipelines"][0]["operator_details"][0][
            "udf_completed_input_rows"
        ]
        == "1"
    )
    assert fragment.progress_topology == {
        "schema": "pipeline_topology",
        "pipelines": [
            {
                "pipeline_id": 1,
                "operators": ["STREAMING_UDF"],
                "operator_details": [{"pipeline_role": "source", "udf_name": "ai_prompt"}],
            },
            {
                "pipeline_id": 2,
                "operators": ["COPY_TO_FILE"],
                "operator_details": [{}],
            },
        ],
    }


@pytest.mark.parametrize(
    "bad_metrics",
    [
        pytest.param(
            {
                "running_attempts": [
                    {
                        "task_stats": {
                            "pipelines": [
                                {
                                    "pipeline_id": 1,
                                    "operators": ["STREAMING_UDF"],
                                    "operator_details": [{}],
                                }
                            ]
                        }
                    }
                ]
            },
            id="conflicting-operators",
        ),
        pytest.param({"running_attempts": "not-a-sequence"}, id="malformed-attempts"),
    ],
)
def test_native_progress_topology_unavailable_does_not_break_query_status(bad_metrics):
    query_id = "query-progress-unavailable"
    fragment_id = f"{query_id}:scan"
    registry = _NativeFteProgressRegistry()
    registry.register_requests(
        [
            {
                "task_id": _task_id(0, query_id=query_id),
                "fragment_id": fragment_id,
            }
        ]
    )
    initial_metrics = {
        "running_attempts": [
            {
                "task_stats": {
                    "pipelines": [
                        {
                            "pipeline_id": 1,
                            "operators": ["TABLE_SCAN"],
                            "operator_details": [{}],
                        }
                    ]
                }
            }
        ]
    }

    registry.record_partition_metrics(query_id, fragment_id, "0", initial_metrics)
    registry.record_partition_metrics(query_id, fragment_id, "0", bad_metrics)
    registry.record_partition_metrics(query_id, fragment_id, "0", initial_metrics)

    status = registry.query_status(query_id)
    assert status["fragment_executions"][fragment_id]["progress_topology"] == {
        "schema": "pipeline_topology",
        "pipelines": [],
    }
    snapshot = build_progress_snapshot({"queries": {query_id: status}}, query_id)
    assert snapshot["fragments"][0]["pipelines"] == []


def test_native_progress_snapshot_normalizes_stable_identity_across_attempts():
    query_id = "query-progress-stable-identity"
    fragment_id = f"{query_id}:scan"
    registry = _NativeFteProgressRegistry()
    registry.register_requests(
        [
            {
                "task_id": _task_id(0, query_id=query_id),
                "fragment_id": fragment_id,
            }
        ]
    )

    def task_stats(operator_details, input_rows):
        return {
            "pipelines": [
                {
                    "pipeline_id": 1,
                    "operators": ["STREAMING_UDF"],
                    "operator_details": [operator_details],
                    "input_rows": input_rows,
                    "total_pipeline_tasks": 1,
                    "running_pipeline_tasks": 1,
                }
            ]
        }

    registry.record_partition_metrics(
        query_id,
        fragment_id,
        "0",
        {
            "state": "RUNNING",
            "running_attempts": [
                {
                    "task_stats": task_stats(
                        {
                            "pipeline_role": "source",
                            "udf_completed_input_rows": "1",
                        },
                        1,
                    )
                },
                {
                    "task_stats": task_stats(
                        {
                            "pipeline_role": "source",
                            "udf_name": "ai_prompt",
                            "udf_completed_input_rows": "2",
                        },
                        2,
                    )
                },
            ],
        },
    )

    status = registry.query_status(query_id)
    snapshot = build_progress_snapshot({"queries": {query_id: status}}, query_id)
    first_attempt_details = status["fragment_executions"][fragment_id]["partitions"]["0"]["running_attempts"][0][
        "task_stats"
    ]["pipelines"][0]["operator_details"][0]
    assert first_attempt_details == {
        "pipeline_role": "source",
        "udf_completed_input_rows": "1",
    }
    assert snapshot["fragments"][0]["pipelines"] == [
        {
            "id": "1.1",
            "display_id": "1.1",
            "name": "ai_prompt(source)",
            "state": "R",
            "processed_rows": 3,
            "processed_bytes": 0,
            "output_rows": 0,
            "output_bytes": 0,
            "queued_pipeline_tasks": 0,
            "running_pipeline_tasks": 2,
            "completed_pipeline_tasks": 0,
            "total_pipeline_tasks": 2,
        }
    ]


def test_native_backend_progress_query_status_uses_cached_task_stats(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def execute_fn(request):
        callback = request.get("native_progress_callback")
        if callback is not None:
            callback({"processed_input_rows": 17})
        started.set()
        release.wait(timeout=5.0)
        return {"ok": True}

    backend = NativeFteWorkerManagerBackend(execute_fn=execute_fn, max_running_tasks=1)
    try:
        handles = backend.submit_tasks(
            [
                {
                    "task_id": _task_id(0, query_id="query-progress-cache"),
                    "fragment_id": "query-progress-cache:scan",
                }
            ]
        )
        assert started.wait(timeout=2.0)

        def fail_blocking_status(_task_id):
            raise AssertionError("progress status should use the native status cache")

        monkeypatch.setattr(handles[0]._worker, "fte_get_task_status", fail_blocking_status)

        query_status = backend.fte_query_status("query-progress-cache")
        partition = query_status["fragment_executions"]["query-progress-cache:scan"]["partitions"]["0"]
        assert partition["running_attempts"][0]["task_stats"]["processed_input_rows"] == 17
    finally:
        release.set()
        backend.drop_query("query-progress-cache")
        backend.shutdown()


def test_native_backend_progress_registry_uses_registration_order_for_fragment_display():
    started = threading.Event()
    release = threading.Event()

    def execute_fn(_request):
        started.set()
        release.wait(timeout=5.0)
        return {"ok": True}

    backend = NativeFteWorkerManagerBackend(execute_fn=execute_fn, max_running_tasks=1)
    try:
        backend.submit_tasks(
            [
                {
                    "task_id": _task_id(0, query_id="query-order"),
                    "fragment_id": "query-order:ScanSource:1",
                },
                {
                    "task_id": _task_id(1, query_id="query-order"),
                    "fragment_id": "query-order:Repartition:3",
                },
                {
                    "task_id": _task_id(2, query_id="query-order"),
                    "fragment_id": "query-order:Repartition:3",
                },
            ]
        )
        assert started.wait(timeout=2.0)

        query_status = backend.fte_query_status("query-order")
        assert query_status["partition_count"] == 3
        assert query_status["fragment_executions"]["query-order:ScanSource:1"]["fragment_execution_id"] == 0
        assert query_status["fragment_executions"]["query-order:Repartition:3"]["fragment_execution_id"] == 1

        snapshot = build_progress_snapshot({"queries": {"query-order": query_status}}, "query-order")
        assert snapshot["total_pipeline_tasks"] == 0
        assert snapshot["total_partitions"] == 3
        assert [fragment["id"] for fragment in snapshot["fragments"]] == [
            "query-order:ScanSource:1",
            "query-order:Repartition:3",
        ]
    finally:
        release.set()
        backend.drop_query("query-order")
        backend.shutdown()


def test_native_backend_pop_refreshes_final_progress_registry_snapshot():
    started = threading.Event()
    release = threading.Event()

    def execute_fn(_request):
        started.set()
        release.wait(timeout=5.0)
        return {"ok": True}

    backend = NativeFteWorkerManagerBackend(execute_fn=execute_fn, max_running_tasks=1)
    try:
        handles = backend.submit_tasks(
            [
                {
                    "task_id": _task_id(0, query_id="query-pop-progress"),
                    "fragment_id": "query-pop-progress:scan",
                }
            ]
        )
        assert started.wait(timeout=2.0)
        running_status = backend.fte_query_status("query-pop-progress")
        assert running_status["finished"] is False

        release.set()
        deadline = time.monotonic() + 2.0
        while not handles[0].done() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert handles[0].done()

        popped = backend.pop_fte_result_handles("query-pop-progress")
        assert popped == handles
        final_status = backend.fte_query_status("query-pop-progress")
        snapshot = build_progress_snapshot(
            {"queries": {"query-pop-progress": final_status}},
            "query-pop-progress",
        )

        assert final_status["finished"] is True
        assert snapshot["completed_pipeline_tasks"] == snapshot["total_pipeline_tasks"] == 0
        assert snapshot["completed_partitions"] == snapshot["total_partitions"] == 1
    finally:
        release.set()
        backend.drop_query("query-pop-progress")
        backend.shutdown()


def test_native_popped_handle_completion_updates_progress_registry_snapshot():
    started = threading.Event()
    release = threading.Event()

    def execute_fn(_request):
        started.set()
        release.wait(timeout=5.0)
        return {"ok": True}

    backend = NativeFteWorkerManagerBackend(execute_fn=execute_fn, max_running_tasks=1)
    try:
        handles = backend.submit_tasks(
            [
                {
                    "task_id": _task_id(0, query_id="query-popped-progress"),
                    "fragment_id": "query-popped-progress:scan",
                }
            ]
        )
        assert started.wait(timeout=2.0)

        popped = backend.pop_fte_result_handles("query-popped-progress")
        assert popped == handles
        running_status = backend.fte_query_status("query-popped-progress")
        assert running_status["finished"] is False

        release.set()
        deadline = time.monotonic() + 2.0
        while not popped[0].done() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert popped[0].done()

        final_status = backend.fte_query_status("query-popped-progress")
        snapshot = build_progress_snapshot(
            {"queries": {"query-popped-progress": final_status}},
            "query-popped-progress",
        )

        assert final_status["finished"] is True
        assert snapshot["completed_pipeline_tasks"] == snapshot["total_pipeline_tasks"] == 0
        assert snapshot["completed_partitions"] == snapshot["total_partitions"] == 1
    finally:
        release.set()
        backend.drop_query("query-popped-progress")
        backend.shutdown()


def test_cxx_distributed_runner_sends_planrunner_tasks_to_python_backend():
    class NoOutputHandle:
        def __init__(self, task, partition_id: int):
            context = task.context()
            query_id = context["query_id"]
            self.task_context_info = task.task_context()
            self.task_id = FteTaskAttemptId(FteTaskId(query_id, 0, partition_id), 0)
            self.worker_id = "native-worker-0"
            self.acked = False
            self.released = False

        def done(self):
            return True

        def get_result_sync(self):
            return vane.ray_cxx.RayTaskResult.no_output()

        def ack(self):
            self.acked = True

        def release_result_payload(self):
            self.released = True

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.submitted_task_names = []
            self.exhausted_calls = []
            self.status_calls = []
            self.handles = []

        def worker_snapshots(self):
            return [
                {
                    "worker_id": "native-worker-0",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, tasks):
            handles = []
            for task in tasks:
                self.submitted_task_names.append(task.name())
                handle = NoOutputHandle(task, len(self.handles) + len(handles))
                handles.append(handle)
            self.handles.extend(handles)
            return handles

        def task_input_stream_exhausted(self, query_id, source_node_ids):
            self.exhausted_calls.append((query_id, list(source_node_ids)))
            return []

        def fte_query_status(self, query_id):
            self.status_calls.append(query_id)
            return {
                "finished": True,
                "failed": False,
                "selected_attempt_task_ids": [str(handle.task_id) for handle in self.handles],
                "message": "finished",
            }

        def drop_query(self, _query_id):
            pass

        def shutdown(self):
            pass

    con = vane.connect()
    relation = con.sql("SELECT 1 AS i")
    query_id = f"native-backend-bridge-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        query_id,
    ).to_physical_plan(con)
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)

    parts = collect_result_stream(runner.run_plan(plan, con))

    assert parts == []
    assert backend.submitted_task_names
    assert backend.status_calls
    assert all(call[0] == query_id for call in backend.exhausted_calls)
    assert backend.status_calls[-1] == query_id
    assert all(handle.acked for handle in backend.handles)
    assert all(handle.released for handle in backend.handles)


def test_cxx_python_backend_uses_later_nonempty_failed_partition_detail():
    class Backend(_QueryLifecycleBackend):
        def worker_snapshots(self):
            return [
                {
                    "worker_id": "native-worker-failed-detail",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, _tasks):
            return []

        def task_input_stream_exhausted(self, _query_id, _source_node_ids):
            return []

        def fte_query_status(self, _query_id):
            return {
                "failed": True,
                "finished": False,
                "message": "  ",
                "scheduler_failure": "\t",
                "selected_attempt_task_ids": [],
                "failed_partitions": [
                    {"latest_failure": None},
                    {
                        "latest_failure": {
                            "message": "\n",
                            "failure_reason": "provider TimeoutError: request timed out",
                        }
                    },
                ],
            }

        def drop_query(self, _query_id):
            pass

        def shutdown(self):
            pass

    con = vane.connect()
    query_id = f"python-backend-failed-detail-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(Backend())
    try:
        with pytest.raises(RuntimeError, match="provider TimeoutError: request timed out"):
            collect_result_stream(runner.run_plan(plan, con))
    finally:
        runner.drop_query_fragments(query_id)
        runner.shutdown()
        con.close()


@pytest.mark.parametrize(
    "middle",
    ["x" * 16_384, "界" * 2_048],
    ids=["character-limit", "utf8-byte-limit"],
)
def test_cxx_python_backend_bounds_failed_partition_detail_without_losing_edges(middle):
    prefix = "provider TimeoutError: request timed out"
    suffix = "terminal provider cause"

    class Backend(_QueryLifecycleBackend):
        def worker_snapshots(self):
            return [
                {
                    "worker_id": "native-worker-bounded-failed-detail",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, _tasks):
            return []

        def task_input_stream_exhausted(self, _query_id, _source_node_ids):
            return []

        def fte_query_status(self, _query_id):
            return {
                "failed": True,
                "finished": False,
                "selected_attempt_task_ids": [],
                "failed_partitions": [
                    {"latest_failure": f"{prefix}:{middle}:{suffix}"},
                ],
            }

        def drop_query(self, _query_id):
            pass

        def shutdown(self):
            pass

    con = vane.connect()
    query_id = f"python-backend-bounded-failed-detail-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(Backend())
    try:
        with pytest.raises(RuntimeError) as exc_info:
            collect_result_stream(runner.run_plan(plan, con))
        detail = str(exc_info.value)
        assert prefix in detail
        assert suffix in detail
        assert len(detail.encode("utf-8")) < 8 * 1024
    finally:
        runner.drop_query_fragments(query_id)
        runner.shutdown()
        con.close()


def test_cxx_python_backend_rejects_empty_selected_attempt_task_id():
    class Backend(_QueryLifecycleBackend):
        def worker_snapshots(self):
            return [
                {
                    "worker_id": "native-worker-empty-selected-id",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, _tasks):
            return []

        def task_input_stream_exhausted(self, _query_id, _source_node_ids):
            return []

        def fte_query_status(self, _query_id):
            return {
                "finished": True,
                "failed": False,
                "selected_attempt_task_ids": [""],
            }

        def drop_query(self, _query_id):
            pass

        def shutdown(self):
            pass

    con = vane.connect()
    query_id = f"python-backend-empty-selected-id-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(Backend())
    try:
        with pytest.raises(RuntimeError, match="entries must be non-empty"):
            collect_result_stream(runner.run_plan(plan, con))
    finally:
        runner.drop_query_fragments(query_id)
        runner.shutdown()
        con.close()


@pytest.mark.parametrize("selected_attempt_task_ids", [["selected.0", "selected.0"], ["selected.0"]])
def test_cxx_python_backend_rejects_invalid_selected_attempt_handle_coverage(selected_attempt_task_ids):
    class Backend(_QueryLifecycleBackend):
        def worker_snapshots(self):
            return [
                {
                    "worker_id": "native-worker-invalid-selected-coverage",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, _tasks):
            return []

        def task_input_stream_exhausted(self, _query_id, _source_node_ids):
            return []

        def fte_query_status(self, _query_id):
            return {
                "finished": True,
                "failed": False,
                "selected_attempt_task_ids": selected_attempt_task_ids,
            }

        def drop_query(self, _query_id):
            pass

        def shutdown(self):
            pass

    con = vane.connect()
    query_id = f"python-backend-invalid-selected-coverage-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(Backend())
    expected = "entries must be unique" if len(selected_attempt_task_ids) == 2 else "result-handle validation"
    try:
        with pytest.raises(RuntimeError, match=expected):
            collect_result_stream(runner.run_plan(plan, con))
    finally:
        runner.drop_query_fragments(query_id)
        runner.shutdown()
        con.close()


def test_cxx_python_backend_rejects_duplicate_handles_for_selected_attempt():
    class DuplicateHandle:
        def __init__(self, task):
            context = task.context()
            self.task_context_info = task.task_context()
            self.task_id = FteTaskAttemptId(FteTaskId(context["query_id"], 0, 0), 0)
            self.worker_id = "native-worker-duplicate-selected-handle"

        def done(self):
            raise AssertionError("duplicate selected handles must be rejected before polling")

        def get_result_sync(self):
            raise AssertionError("duplicate selected handles must not be materialized")

        def ack(self):
            pass

        def release_result_payload(self):
            pass

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.handles = []

        def worker_snapshots(self):
            return [
                {
                    "worker_id": "native-worker-duplicate-selected-handle",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, tasks):
            handles = []
            for task in tasks:
                handle = DuplicateHandle(task)
                handles.extend((handle, DuplicateHandle(task)))
            self.handles.extend(handles)
            return handles

        def task_input_stream_exhausted(self, _query_id, _source_node_ids):
            return []

        def fte_query_status(self, _query_id):
            return {
                "finished": True,
                "failed": False,
                "selected_attempt_task_ids": [str(self.handles[0].task_id)],
            }

        def drop_query(self, _query_id):
            pass

        def shutdown(self):
            pass

    con = vane.connect()
    query_id = f"python-backend-duplicate-selected-handle-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(Backend())
    try:
        with pytest.raises(RuntimeError, match="multiple result handles for one selected attempt"):
            collect_result_stream(runner.run_plan(plan, con))
    finally:
        runner.drop_query_fragments(query_id)
        runner.shutdown()
        con.close()


def test_cxx_python_backend_releases_batch_when_later_result_handle_is_malformed():
    class Handle:
        def __init__(self, task, *, malformed=False):
            context = task.context()
            self.task_context_info = task.task_context()
            if not malformed:
                self.task_id = FteTaskAttemptId(FteTaskId(context["query_id"], 0, 0), 0)
            self.worker_id = "native-worker-partial-handle-batch"
            self.release_calls = 0

        def release_result_payload(self):
            self.release_calls += 1

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.handles = []

        def worker_snapshots(self):
            return [
                {
                    "worker_id": "native-worker-partial-handle-batch",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, tasks):
            if self.handles:
                return []
            self.handles = [Handle(tasks[0]), Handle(tasks[0], malformed=True)]
            return self.handles

        def task_input_stream_exhausted(self, _query_id, _source_node_ids):
            return []

        def fte_query_status(self, _query_id):
            return {
                "finished": True,
                "failed": False,
                "selected_attempt_task_ids": [],
            }

        def drop_query(self, _query_id):
            pass

        def shutdown(self):
            pass

    con = vane.connect()
    query_id = f"python-backend-partial-handle-batch-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    try:
        with pytest.raises(RuntimeError, match="FTE result handle must provide task_id"):
            collect_result_stream(runner.run_plan(plan, con))
        assert [handle.release_calls for handle in backend.handles] == [1, 1]
    finally:
        runner.drop_query_fragments(query_id)
        runner.shutdown()
        con.close()


def test_cxx_python_backend_drains_but_does_not_publish_handles_when_selection_is_empty():
    class UnselectedHandle:
        def __init__(self, task, partition_id):
            context = task.context()
            self.task_context_info = task.task_context()
            self.task_id = FteTaskAttemptId(FteTaskId(context["query_id"], 0, partition_id), 0)
            self.worker_id = "native-worker-unselected-empty"
            self.get_result_calls = 0
            self.ack_calls = 0
            self.release_calls = 0

        def done(self):
            return True

        def get_result_sync(self):
            self.get_result_calls += 1
            return vane.ray_cxx.RayTaskResult.no_output()

        def ack(self):
            self.ack_calls += 1

        def release_result_payload(self):
            self.release_calls += 1

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.handles = []

        def worker_snapshots(self):
            return [
                {
                    "worker_id": "native-worker-unselected-empty",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, tasks):
            new_handles = [UnselectedHandle(task, len(self.handles) + index) for index, task in enumerate(tasks)]
            self.handles.extend(new_handles)
            return new_handles

        def task_input_stream_exhausted(self, _query_id, _source_node_ids):
            return []

        def fte_query_status(self, _query_id):
            return {
                "finished": True,
                "failed": False,
                "selected_attempt_task_ids": [],
            }

        def drop_query(self, _query_id):
            pass

        def shutdown(self):
            pass

    con = vane.connect()
    query_id = f"python-backend-unselected-empty-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    try:
        assert collect_result_stream(runner.run_plan(plan, con)) == []
        assert backend.handles
        assert all(handle.get_result_calls == 1 for handle in backend.handles)
        assert all(handle.ack_calls == 0 for handle in backend.handles)
        assert all(handle.release_calls == 1 for handle in backend.handles)
    finally:
        runner.drop_query_fragments(query_id)
        runner.shutdown()
        con.close()


def test_cxx_python_backend_releases_submit_handle_returned_after_query_drop():
    submit_started = threading.Event()
    allow_submit_return = threading.Event()

    class LateHandle:
        def __init__(self, task):
            context = task.context()
            query_id = context["query_id"]
            self.task_context_info = task.task_context()
            self.task_id = FteTaskAttemptId(FteTaskId(query_id, 0, 0), 0)
            self.worker_id = "late-worker"
            self.release_calls = 0

        def done(self):
            return True

        def get_result_sync(self):
            return vane.ray_cxx.RayTaskResult.no_output()

        def ack(self):
            pass

        def release_result_payload(self):
            self.release_calls += 1

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.handle = None
            self.drop_calls = []

        def worker_snapshots(self):
            return [
                {
                    "worker_id": "late-worker",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, tasks):
            self.handle = LateHandle(tasks[0])
            submit_started.set()
            assert allow_submit_return.wait(timeout=5.0)
            return [self.handle]

        def task_input_stream_exhausted(self, _query_id, _source_node_ids):
            return []

        def fte_query_status(self, _query_id):
            return {
                "finished": True,
                "failed": False,
                "selected_attempt_task_ids": [],
            }

        def drop_query(self, query_id):
            self.drop_calls.append(str(query_id))
            allow_submit_return.set()

        def shutdown(self):
            pass

    con = vane.connect()
    query_id = f"python-backend-late-submit-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    drop_errors = []

    def drop_while_submit_is_blocked():
        try:
            assert submit_started.wait(timeout=5.0)
            runner.drop_query_fragments(query_id)
        except BaseException as exc:
            drop_errors.append(exc)
        finally:
            allow_submit_return.set()

    drop_thread = threading.Thread(target=drop_while_submit_is_blocked)
    drop_thread.start()
    try:
        with pytest.raises(RuntimeError, match="query is closing"):
            collect_result_stream(runner.run_plan(plan, con))
    finally:
        allow_submit_return.set()
        drop_thread.join(timeout=5.0)

    assert not drop_thread.is_alive()
    assert drop_errors == []
    # The first sweep unblocks the in-flight submit; the second catches work
    # that crossed the fence before that submit returned.
    assert backend.drop_calls == [query_id, query_id]
    assert backend.handle is not None
    assert backend.handle.release_calls == 1
    con.close()


@pytest.mark.parametrize(
    ("exchange_sink_instance", "expect_released_after_run", "cleanup_mode", "release_failures"),
    [
        (None, False, "drop", 0),
        (None, False, "drop", 1),
        (None, False, "shutdown", 0),
        (
            {
                "task_partition_id": 0,
                "attempt_id": 0,
                "output_partition_count": 1,
                "output_location": "/tmp/fake-exchange-output",
            },
            True,
            "drop",
            0,
        ),
    ],
    ids=[
        "final-output-drop",
        "final-output-drop-release-retry",
        "final-output-shutdown",
        "exchange-output-drop",
    ],
)
def test_cxx_streaming_runner_output_handle_release_lifecycle(
    exchange_sink_instance,
    expect_released_after_run: bool,
    cleanup_mode: str,
    release_failures: int,
):
    pa = pytest.importorskip("pyarrow")

    class OutputHandle:
        def __init__(self, task, partition_id: int):
            context = task.context()
            query_id = context["query_id"]
            self.task_context_info = task.task_context()
            self.task_id = FteTaskAttemptId(FteTaskId(query_id, 0, partition_id), 0)
            self.worker_id = "native-worker-0"
            self.acked = False
            self.released = False
            self.release_calls = 0
            self.exchange_sink_instance = exchange_sink_instance

        def done(self):
            return True

        def get_result_sync(self):
            table = pa.table({"i": [1]})
            return vane.ray_cxx.RayTaskResult.success(
                [table],
                [],
                None,
                0,
                self.exchange_sink_instance,
            )

        def ack(self):
            self.acked = True

        def release_result_payload(self):
            self.release_calls += 1
            if self.release_calls <= release_failures:
                raise RuntimeError("planned transient result release failure")
            self.released = True

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.handles = []
            self.drop_calls = []
            self.shutdown_calls = 0

        def worker_snapshots(self):
            return [
                {
                    "worker_id": "native-worker-0",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, tasks):
            handles = []
            for task in tasks:
                handle = OutputHandle(task, len(self.handles) + len(handles))
                handles.append(handle)
            self.handles.extend(handles)
            return handles

        def task_input_stream_exhausted(self, _query_id, _source_node_ids):
            return []

        def fte_query_status(self, query_id):
            return {
                "finished": True,
                "failed": False,
                "selected_attempt_task_ids": [str(handle.task_id) for handle in self.handles],
                "message": "finished",
            }

        def drop_query(self, query_id):
            self.drop_calls.append(str(query_id))

        def shutdown(self):
            self.shutdown_calls += 1

    con = vane.connect()
    con.execute("SET threads=3")
    relation = con.sql("SELECT 1 AS i")
    query_id = f"streaming-output-release-lifecycle-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        query_id,
    ).to_physical_plan(con)
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)

    parts = collect_result_stream(runner.run_plan(plan, con))

    assert parts
    assert all(handle.acked for handle in backend.handles)
    assert all(handle.released is expect_released_after_run for handle in backend.handles)
    assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is not None

    if cleanup_mode == "shutdown":
        runner.shutdown()
        assert backend.shutdown_calls == 1
        assert all(handle.released for handle in backend.handles)
        runner.drop_query_fragments(query_id)
    else:
        runner.drop_query_fragments(query_id)

    expected_drop_calls = [] if cleanup_mode == "shutdown" else [query_id]
    assert backend.drop_calls == expected_drop_calls
    assert all(handle.released for handle in backend.handles)
    expected_release_calls = 2 if release_failures else 1
    assert all(handle.release_calls == expected_release_calls for handle in backend.handles)
    assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is None


@pytest.mark.parametrize("cleanup_mode", ["drop", "shutdown"])
def test_cxx_backend_cleanup_waits_for_active_output_delivery(cleanup_mode):
    pa = pytest.importorskip("pyarrow")
    status_started = threading.Event()
    allow_status = threading.Event()

    class OutputHandle:
        def __init__(self, task, partition_id: int):
            context = task.context()
            query_id = context["query_id"]
            self.task_context_info = task.task_context()
            self.task_id = FteTaskAttemptId(FteTaskId(query_id, 0, partition_id), 0)
            self.worker_id = "native-worker-0"
            self.acked = False
            self.release_calls = 0

        def done(self):
            return True

        def get_result_sync(self):
            return vane.ray_cxx.RayTaskResult.success(
                [pa.table({"i": [1]})],
                [],
                None,
                0,
                None,
            )

        def ack(self):
            self.acked = True

        def release_result_payload(self):
            self.release_calls += 1

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.handles = []
            self.cleanup_release_observations = []

        def worker_snapshots(self):
            return [
                {
                    "worker_id": "native-worker-0",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, tasks):
            handles = [OutputHandle(task, index) for index, task in enumerate(tasks)]
            self.handles.extend(handles)
            return handles

        def task_input_stream_exhausted(self, _query_id, _source_node_ids):
            return []

        def fte_query_status(self, _query_id):
            status_started.set()
            assert allow_status.wait(timeout=5.0)
            return {
                "finished": True,
                "failed": False,
                "selected_attempt_task_ids": [str(handle.task_id) for handle in self.handles],
                "message": "finished",
            }

        def _allow_cleanup(self):
            self.cleanup_release_observations.append([handle.release_calls for handle in self.handles])
            allow_status.set()

        def drop_query(self, _query_id):
            self._allow_cleanup()

        def shutdown(self):
            self._allow_cleanup()

    con = vane.connect()
    con.execute("SET threads=1")
    query_id = f"active-output-delivery-{cleanup_mode}-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    stream = runner.run_plan(plan, con)
    consumed = []
    consume_errors: list[BaseException] = []

    def consume() -> None:
        try:
            consumed.extend(collect_result_stream(stream))
        except BaseException as exc:
            consume_errors.append(exc)

    consumer = threading.Thread(target=consume)
    consumer.start()
    assert status_started.wait(timeout=5.0)

    if cleanup_mode == "drop":
        runner.drop_query_fragments(query_id)
    else:
        runner.shutdown()

    consumer.join(timeout=5.0)
    assert not consumer.is_alive()
    assert consume_errors == []
    assert consumed
    assert backend.cleanup_release_observations[0]
    assert all(release_calls == 0 for release_calls in backend.cleanup_release_observations[0])
    assert all(handle.acked for handle in backend.handles)
    assert all(handle.release_calls == 1 for handle in backend.handles)

    if cleanup_mode == "drop":
        runner.shutdown()
    else:
        runner.drop_query_fragments(query_id)


def test_cxx_backend_drop_query_failure_is_not_silently_accepted():
    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.drop_calls = []

        def drop_query(self, query_id):
            self.drop_calls.append(str(query_id))
            raise RuntimeError("planned backend drop failure")

    query_id = f"backend-drop-failure-{uuid.uuid4()}"
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    runner._register_query_owner_for_test(query_id, query_id)

    with pytest.raises(RuntimeError, match="planned backend drop failure"):
        runner.drop_query_fragments(query_id)

    assert backend.drop_calls == [query_id]


def test_cxx_backend_requires_drop_query_contract():
    class Backend(_QueryLifecycleBackend):
        def shutdown(self):
            pass

    query_id = f"backend-missing-drop-{uuid.uuid4()}"
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(Backend())
    runner._register_query_owner_for_test(query_id, query_id)

    with pytest.raises(RuntimeError, match="drop_query"):
        runner.drop_query_fragments(query_id)

    runner.shutdown()


def test_cxx_backend_registration_failure_rolls_back_new_replay_state():
    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.drop_calls = []
            self.registration_side_effect = False

        def register_query_owner(self, _query_id, _owner_query_id):
            self.registration_side_effect = True
            raise RuntimeError("planned lifecycle registration failure")

        def drop_query(self, query_id):
            assert self.registration_side_effect
            self.registration_side_effect = False
            self.drop_calls.append(str(query_id))

    con = vane.connect()
    query_id = f"registration-replay-rollback-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)

    with pytest.raises(RuntimeError, match="planned lifecycle registration failure"):
        runner.run_plan(plan, con)

    assert backend.drop_calls == [query_id]
    assert backend.registration_side_effect is False
    assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is None


def test_cxx_backend_registration_failure_fences_owner_until_drop():
    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.registration_calls = []
            self.drop_calls = []

        def register_query_owner(self, query_id, owner_query_id):
            self.registration_calls.append((str(query_id), str(owner_query_id)))
            if len(self.registration_calls) == 1:
                raise RuntimeError("planned lifecycle registration failure")

        def drop_query(self, query_id):
            self.drop_calls.append(str(query_id))

    query_id = f"registration-fence-{uuid.uuid4()}"
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)

    with pytest.raises(RuntimeError, match="planned lifecycle registration failure"):
        runner._register_query_owner_for_test(query_id, query_id)

    with pytest.raises(RuntimeError, match="cannot register closing FTE query lifecycle"):
        runner._register_query_owner_for_test(query_id, query_id)
    assert backend.registration_calls == [(query_id, query_id)]

    runner.drop_query_fragments(query_id)
    assert backend.drop_calls == [query_id]

    runner._register_query_owner_for_test(query_id, query_id)
    assert backend.registration_calls == [(query_id, query_id), (query_id, query_id)]
    runner.drop_query_fragments(query_id)
    assert backend.drop_calls == [query_id, query_id]


def test_cxx_backend_serializes_overlapping_drop_and_reuses_query_id():
    drop_started = threading.Event()
    allow_first_drop = threading.Event()

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.drop_calls = []

        def drop_query(self, query_id):
            self.drop_calls.append(str(query_id))
            if len(self.drop_calls) == 1:
                drop_started.set()
                assert allow_first_drop.wait(timeout=5.0)

    query_id = f"backend-overlapping-drop-{uuid.uuid4()}"
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    runner._register_query_owner_for_test(query_id, query_id)
    errors: list[BaseException] = []

    def drop() -> None:
        try:
            runner.drop_query_fragments(query_id)
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=drop)
    second = threading.Thread(target=drop)
    first.start()
    assert drop_started.wait(timeout=5.0)
    second.start()
    assert second.is_alive()

    allow_first_drop.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert backend.drop_calls == [query_id]

    runner._register_query_owner_for_test(query_id, query_id)
    runner.drop_query_fragments(query_id)
    assert backend.drop_calls == [query_id, query_id]


def test_cxx_backend_overlapping_drop_joins_failure_and_retry_generation():
    thread_count = 6
    caller_barrier = threading.Barrier(thread_count)
    drop_condition = threading.Condition()

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.drop_calls = []

        def drop_query(self, query_id):
            with drop_condition:
                self.drop_calls.append(str(query_id))
                call_number = len(self.drop_calls)
                drop_condition.notify_all()
                if call_number == 1:
                    # A correct single-flight implementation admits only this
                    # leader. Keep it open long enough for the other callers to
                    # join; the predicate only completes early for the broken
                    # implementation that calls every backend drop.
                    drop_condition.wait_for(lambda: len(self.drop_calls) == thread_count, timeout=0.5)
            if call_number == 1:
                raise RuntimeError("planned shared teardown failure")

    query_id = f"backend-overlapping-drop-failure-{uuid.uuid4()}"
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    runner._register_query_owner_for_test(query_id, query_id)
    errors: list[BaseException] = []

    def drop() -> None:
        try:
            caller_barrier.wait(timeout=5.0)
            runner.drop_query_fragments(query_id)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=drop) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert [thread for thread in threads if thread.is_alive()] == []
    assert len(errors) == thread_count
    assert len({str(error) for error in errors}) == 1
    assert "planned shared teardown failure" in str(errors[0])
    assert backend.drop_calls == [query_id]

    runner.drop_query_fragments(query_id)
    runner._register_query_owner_for_test(query_id, query_id)
    runner.drop_query_fragments(query_id)
    assert backend.drop_calls == [query_id, query_id, query_id]


def test_cxx_backend_routes_nested_execution_drop_to_registered_resource_owner():
    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.registrations = []
            self.drop_calls = []

        def register_query_owner(self, query_id, owner_query_id):
            self.registrations.append((str(query_id), str(owner_query_id)))

        def drop_query(self, query_id):
            self.drop_calls.append(str(query_id))

    execution_query_id = f"nested-execution-{uuid.uuid4()}"
    resource_query_id = f"resource-owner-{uuid.uuid4()}"
    con = vane.connect()
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        f"source-plan-{uuid.uuid4()}",
    ).to_physical_plan(con)
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    try:
        assert vane.ray_cxx._register_query_python_replay_state(resource_query_id, plan) is True
        assert vane.ray_cxx._lookup_query_connection_snapshot(resource_query_id) is not None

        runner._register_query_owner_for_test(execution_query_id, resource_query_id)
        runner.drop_query_fragments(execution_query_id)

        assert backend.registrations == [(execution_query_id, resource_query_id)]
        assert backend.drop_calls == [execution_query_id, resource_query_id]
        assert vane.ray_cxx._lookup_query_connection_snapshot(resource_query_id) is None
    finally:
        vane.ray_cxx._cleanup_query_python_replay_state(resource_query_id)
        con.close()


@pytest.mark.parametrize("cleanup_mode", ["drop", "shutdown"])
def test_cxx_ray_manager_cleans_nested_execution_resource_owner_state(cleanup_mode):
    execution_query_id = f"ray-nested-execution-{uuid.uuid4()}"
    resource_query_id = f"ray-resource-owner-{uuid.uuid4()}"
    con = vane.connect()
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        f"source-plan-{uuid.uuid4()}",
    ).to_physical_plan(con)
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    try:
        assert vane.ray_cxx._register_query_python_replay_state(resource_query_id, plan) is True
        assert vane.ray_cxx._lookup_query_connection_snapshot(resource_query_id) is not None

        runner._register_query_owner_for_test(execution_query_id, resource_query_id)
        if cleanup_mode == "drop":
            runner.drop_query_fragments(execution_query_id)
        else:
            runner.shutdown()

        assert vane.ray_cxx._lookup_query_connection_snapshot(resource_query_id) is None
    finally:
        vane.ray_cxx._cleanup_query_python_replay_state(resource_query_id)
        con.close()


def test_cxx_backend_registers_order_by_internal_queries_under_resource_owner(tmp_path, monkeypatch):
    from vane.runners.local.runner import _InProcessFragmentExecutor

    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    con = vane.connect()
    for partition_id in range(4):
        source = tmp_path / f"source-{partition_id}.parquet"
        con.execute(
            f"COPY (SELECT i + {partition_id * 1000} AS i FROM range(1000) t(i)) TO '{source}' (FORMAT PARQUET)"
        )

    resource_query_id = f"orderby-resource-owner-{uuid.uuid4()}"
    relation = con.sql(f"SELECT i FROM read_parquet('{tmp_path}/source-*.parquet') ORDER BY i DESC")
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        resource_query_id,
    ).to_physical_plan(con)
    backend = NativeFteWorkerManagerBackend(
        execute_fn=_InProcessFragmentExecutor(),
        max_running_tasks=4,
        num_workers=2,
        num_cpus=4,
    )
    registrations: list[tuple[str, str]] = []
    drop_calls: list[str] = []
    original_register_query_owner = backend.register_query_owner
    original_drop_query = backend.drop_query

    def register_query_owner(query_id, owner_query_id):
        registrations.append((str(query_id), str(owner_query_id)))
        return original_register_query_owner(query_id, owner_query_id)

    def drop_query(query_id):
        drop_calls.append(str(query_id))
        return original_drop_query(query_id)

    backend.register_query_owner = register_query_owner
    backend.drop_query = drop_query
    try:
        runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
        partitions = collect_result_stream(runner.run_plan(plan, con))
        runner.drop_query_fragments(resource_query_id)

        assert partitions
        assert registrations[0] == (resource_query_id, resource_query_id)
        internal_query_ids = {query_id for query_id, _owner_query_id in registrations[1:]}
        assert len(internal_query_ids) == 3
        assert all(owner_query_id == resource_query_id for _query_id, owner_query_id in registrations)
        assert {query_id.rsplit("_", 2)[-2] for query_id in internal_query_ids} == {
            "range",
            "sample",
            "stage",
        }
        assert drop_calls[-1] == resource_query_id
        assert set(drop_calls[:-1]) == internal_query_ids
    finally:
        backend.shutdown()
        con.close()


def test_cxx_backend_drop_waits_for_owner_registration_publication():
    registration_started = threading.Event()
    allow_registration = threading.Event()

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.drop_calls = []

        def register_query_owner(self, _query_id, _owner_query_id):
            registration_started.set()
            assert allow_registration.wait(timeout=5.0)

        def drop_query(self, query_id):
            self.drop_calls.append(str(query_id))

    query_id = f"register-drop-race-{uuid.uuid4()}"
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    errors: list[BaseException] = []

    def register() -> None:
        try:
            runner._register_query_owner_for_test(query_id, query_id)
        except BaseException as exc:
            errors.append(exc)

    def drop() -> None:
        try:
            runner.drop_query_fragments(query_id)
        except BaseException as exc:
            errors.append(exc)

    registration_thread = threading.Thread(target=register)
    drop_thread = threading.Thread(target=drop)
    registration_thread.start()
    assert registration_started.wait(timeout=5.0)
    drop_thread.start()
    assert drop_thread.is_alive()
    assert backend.drop_calls == []

    allow_registration.set()
    registration_thread.join(timeout=5.0)
    drop_thread.join(timeout=5.0)

    assert not registration_thread.is_alive()
    assert not drop_thread.is_alive()
    assert errors == []
    assert backend.drop_calls == [query_id]


def test_cxx_backend_drop_waits_for_all_owner_registrations():
    second_registration_started = threading.Event()
    allow_second_registration = threading.Event()

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.drop_calls = []

        def register_query_owner(self, query_id, _owner_query_id):
            if str(query_id) == second_execution_query_id:
                second_registration_started.set()
                assert allow_second_registration.wait(timeout=5.0)

        def drop_query(self, query_id):
            self.drop_calls.append(str(query_id))

    resource_query_id = f"owner-registration-race-{uuid.uuid4()}"
    first_execution_query_id = f"{resource_query_id}-execution-a"
    second_execution_query_id = f"{resource_query_id}-execution-b"
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    runner._register_query_owner_for_test(first_execution_query_id, resource_query_id)
    errors: list[BaseException] = []

    def register_second() -> None:
        try:
            runner._register_query_owner_for_test(second_execution_query_id, resource_query_id)
        except BaseException as exc:
            errors.append(exc)

    def drop_first() -> None:
        try:
            runner.drop_query_fragments(first_execution_query_id)
        except BaseException as exc:
            errors.append(exc)

    registration_thread = threading.Thread(target=register_second)
    drop_thread = threading.Thread(target=drop_first)
    registration_thread.start()
    assert second_registration_started.wait(timeout=5.0)
    drop_thread.start()
    assert drop_thread.is_alive()
    assert backend.drop_calls == []

    allow_second_registration.set()
    registration_thread.join(timeout=5.0)
    drop_thread.join(timeout=5.0)

    assert not registration_thread.is_alive()
    assert not drop_thread.is_alive()
    assert errors == []
    assert backend.drop_calls == [
        first_execution_query_id,
        second_execution_query_id,
        resource_query_id,
    ]


def test_cxx_backend_re_registering_dropping_owner_is_not_treated_as_idempotent_success():
    """Drop must reject concurrent idempotent re-registration until it clears the mapping.

    Previously the C++ register_query_owner() honored the existing same-owner mapping
    before checking closed/dropping fences, so a re-registration racing with an
    in-progress drop returned success without forwarding the new generation to the
    Python backend. The drop then destroyed the stale mapping, leaving the new query
    without a usable owner. The registration must fail while the owner is dropping and
    succeed again once the drop clears the stale mapping.
    """
    drop_started = threading.Event()
    allow_first_drop = threading.Event()

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.registration_calls = []
            self.drop_calls = []

        def register_query_owner(self, query_id, owner_query_id):
            self.registration_calls.append((str(query_id), str(owner_query_id)))

        def drop_query(self, query_id):
            self.drop_calls.append(str(query_id))
            if len(self.drop_calls) == 1:
                drop_started.set()
                assert allow_first_drop.wait(timeout=5.0)

    query_id = f"backend-dropping-reuse-{uuid.uuid4()}"
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    runner._register_query_owner_for_test(query_id, query_id)
    assert backend.registration_calls == [(query_id, query_id)]

    drop_errors: list[BaseException] = []
    registration_errors: list[BaseException] = []

    def drop() -> None:
        try:
            runner.drop_query_fragments(query_id)
        except BaseException as exc:
            drop_errors.append(exc)

    def re_register() -> None:
        try:
            runner._register_query_owner_for_test(query_id, query_id)
        except BaseException as exc:
            registration_errors.append(exc)

    drop_thread = threading.Thread(target=drop)
    drop_thread.start()
    assert drop_started.wait(timeout=5.0)
    assert backend.drop_calls == [query_id]

    re_register_thread = threading.Thread(target=re_register)
    re_register_thread.start()
    re_register_thread.join(timeout=5.0)
    assert not re_register_thread.is_alive()

    assert len(registration_errors) == 1
    assert isinstance(registration_errors[0], RuntimeError)
    assert "cannot register closing FTE query lifecycle" in str(registration_errors[0])
    # The Python backend never observed the racing re-registration.
    assert backend.registration_calls == [(query_id, query_id)]

    allow_first_drop.set()
    drop_thread.join(timeout=5.0)
    assert not drop_thread.is_alive()
    assert drop_errors == []
    assert backend.drop_calls == [query_id]

    # Now that the drop completed and cleared the stale mapping, the re-registration
    # must succeed and forward the new generation to the Python backend.
    runner._register_query_owner_for_test(query_id, query_id)
    assert backend.registration_calls == [(query_id, query_id), (query_id, query_id)]

    runner.drop_query_fragments(query_id)
    assert backend.drop_calls == [query_id, query_id]


def test_cxx_backend_drop_failure_preserves_replay_state_for_active_submit_until_retry():
    """Backend drop failure must not destroy outer replay/datasource state in flight.

    The internal drop only drains active result-handle operations when the backend
    teardown succeeds. The outer wrapper must therefore keep the cached Python
    replay state (and datasource factories) for the query until a retry or shutdown
    quiesces in-flight work; releasing them eagerly while a submit is still running
    would leave the in-flight query without a usable owner.
    """
    submit_started = threading.Event()
    allow_submit_return = threading.Event()
    retry_drop_started = threading.Event()

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.drop_calls = 0
            self.replay_visible_during_submit = []

        def worker_snapshots(self):
            return [
                {
                    "worker_id": "blocked-submit-worker",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, tasks):
            submit_started.set()
            assert allow_submit_return.wait(timeout=5.0)
            self.replay_visible_during_submit.append(
                vane.ray_cxx._lookup_query_connection_snapshot(query_id) is not None
            )
            return []

        def task_input_stream_exhausted(self, _query_id, _source_node_ids):
            return []

        def fte_query_status(self, _query_id):
            return {
                "finished": True,
                "failed": False,
                "selected_attempt_task_ids": [],
            }

        def drop_query(self, query_id):
            self.drop_calls += 1
            if self.drop_calls == 1:
                raise RuntimeError("planned backend teardown failure")
            retry_drop_started.set()

        def shutdown(self):
            pass

    query_id = f"backend-dropping-state-{uuid.uuid4()}"
    con = vane.connect()
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    run_errors: list[BaseException] = []
    retry_errors: list[BaseException] = []

    def run() -> None:
        try:
            collect_result_stream(runner.run_plan(plan, con))
        except BaseException as exc:
            run_errors.append(exc)

    def retry_drop() -> None:
        try:
            runner.drop_query_fragments(query_id)
        except BaseException as exc:
            retry_errors.append(exc)

    run_thread = threading.Thread(target=run)
    retry_thread = threading.Thread(target=retry_drop)
    try:
        run_thread.start()
        assert submit_started.wait(timeout=5.0)
        assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is not None

        with pytest.raises(RuntimeError, match="planned backend teardown failure"):
            runner.drop_query_fragments(query_id)

        assert run_thread.is_alive()
        assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is not None
        assert backend.drop_calls == 1

        retry_thread.start()
        assert retry_drop_started.wait(timeout=5.0)
        assert retry_thread.is_alive()
        assert backend.drop_calls == 2

        # The successful retry must wait for the already-admitted submit before it
        # releases the replay state.
        assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is not None
        allow_submit_return.set()
        run_thread.join(timeout=5.0)
        retry_thread.join(timeout=5.0)

        assert not run_thread.is_alive()
        assert not retry_thread.is_alive()
        assert retry_errors == []
        assert len(run_errors) == 1
        assert "query is closing" in str(run_errors[0])
        assert backend.replay_visible_during_submit == [True]
        assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is None
    finally:
        allow_submit_return.set()
        run_thread.join(timeout=5.0)
        if retry_thread.ident is not None:
            retry_thread.join(timeout=5.0)
        vane.ray_cxx._cleanup_query_python_replay_state(query_id)
        try:
            runner.shutdown()
        except BaseException:
            pass
        con.close()


def test_cxx_backend_drop_rejects_running_shutdown_without_waiting():
    """A teardown must not wait on shutdown while its caller can block shutdown."""
    submit_started = threading.Event()
    allow_submit_return = threading.Event()
    shutdown_started = threading.Event()
    drop_invoked = threading.Event()

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.drop_calls = []
            self.replay_visible_during_submit = []

        def worker_snapshots(self):
            return [
                {
                    "worker_id": "shutdown-race-worker",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, _tasks):
            submit_started.set()
            assert allow_submit_return.wait(timeout=5.0)
            self.replay_visible_during_submit.append(
                vane.ray_cxx._lookup_query_connection_snapshot(query_id) is not None
            )
            return []

        def task_input_stream_exhausted(self, _query_id, _source_node_ids):
            return []

        def fte_query_status(self, _query_id):
            return {
                "finished": True,
                "failed": False,
                "selected_attempt_task_ids": [],
            }

        def drop_query(self, query_id):
            self.drop_calls.append(str(query_id))

        def shutdown(self):
            shutdown_started.set()

    query_id = f"backend-shutdown-drop-race-{uuid.uuid4()}"
    con = vane.connect()
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    run_errors: list[BaseException] = []
    shutdown_errors: list[BaseException] = []
    drop_errors: list[BaseException] = []

    def run() -> None:
        try:
            collect_result_stream(runner.run_plan(plan, con))
        except BaseException as exc:
            run_errors.append(exc)

    def shutdown() -> None:
        try:
            runner.shutdown()
        except BaseException as exc:
            shutdown_errors.append(exc)

    def drop() -> None:
        drop_invoked.set()
        try:
            runner.drop_query_fragments(query_id)
        except BaseException as exc:
            drop_errors.append(exc)

    run_thread = threading.Thread(target=run)
    shutdown_thread = threading.Thread(target=shutdown)
    drop_thread = threading.Thread(target=drop)
    try:
        run_thread.start()
        assert submit_started.wait(timeout=5.0)
        assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is not None

        shutdown_thread.start()
        assert shutdown_started.wait(timeout=5.0)

        drop_thread.start()
        assert drop_invoked.wait(timeout=5.0)
        drop_thread.join(timeout=5.0)
        assert not drop_thread.is_alive()
        assert shutdown_thread.is_alive()
        assert len(drop_errors) == 1
        assert "cannot tear down FTE query while Python backend is shutting down" in str(drop_errors[0])
        assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is not None

        allow_submit_return.set()
        run_thread.join(timeout=5.0)
        shutdown_thread.join(timeout=5.0)
        drop_thread.join(timeout=5.0)

        assert not run_thread.is_alive()
        assert not shutdown_thread.is_alive()
        assert not drop_thread.is_alive()
        assert shutdown_errors == []
        assert len(drop_errors) == 1
        assert len(run_errors) == 1
        assert "query is closing" in str(run_errors[0])
        assert backend.drop_calls == []
        assert backend.replay_visible_during_submit == [True]
        assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is None
    finally:
        allow_submit_return.set()
        for thread in (run_thread, shutdown_thread, drop_thread):
            if thread.ident is not None:
                thread.join(timeout=5.0)
        vane.ray_cxx._cleanup_query_python_replay_state(query_id)
        con.close()


def test_cxx_backend_preserves_owner_state_until_shutdown_retry():
    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.shutdown_calls = 0
            self.drop_calls = []

        def drop_query(self, query_id):
            self.drop_calls.append(str(query_id))

        def shutdown(self):
            self.shutdown_calls += 1
            if self.shutdown_calls == 1:
                raise RuntimeError("planned Python backend shutdown failure")

    query_id = f"backend-shutdown-retry-{uuid.uuid4()}"
    con = vane.connect()
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    try:
        assert vane.ray_cxx._register_query_python_replay_state(query_id, plan) is True
        runner._register_query_owner_for_test(query_id, query_id)

        with pytest.raises(RuntimeError, match="planned Python backend shutdown failure"):
            runner.shutdown()

        assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is not None
        with pytest.raises(RuntimeError, match="after Python backend shutdown failed"):
            runner.drop_query_fragments(query_id)
        assert backend.drop_calls == []

        runner.shutdown()

        assert backend.shutdown_calls == 2
        assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is None
    finally:
        vane.ray_cxx._cleanup_query_python_replay_state(query_id)
        con.close()


def test_cxx_backend_successful_shutdown_releases_unobserved_submission_exception():
    import gc
    import weakref

    submission_started = threading.Event()
    release_submission = threading.Event()
    exception_refs: list[weakref.ReferenceType[RuntimeError]] = []

    class SubmissionSentinelError(RuntimeError):
        pass

    class Backend(_QueryLifecycleBackend):
        def worker_snapshots(self):
            return [
                {
                    "worker_id": "shutdown-submission-error-worker",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, _tasks):
            submission_started.set()
            assert release_submission.wait(timeout=5)
            error = SubmissionSentinelError("planned unobserved submission failure")
            exception_refs.append(weakref.ref(error))
            raise error

        def shutdown(self):
            release_submission.set()

    con = vane.connect()
    query_id = f"shutdown-submission-error-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    stream = runner.run_plan(plan, con)
    try:
        assert submission_started.wait(timeout=5)
        runner.shutdown()
        assert exception_refs
        deadline = time.monotonic() + 5
        while exception_refs[0]() is not None and time.monotonic() < deadline:
            gc.collect()
            time.sleep(0.01)
        assert exception_refs[0]() is None
    finally:
        release_submission.set()
        del stream
        con.close()


def test_cxx_backend_serializes_concurrent_shutdown_calls():
    shutdown_started = threading.Event()
    allow_shutdown = threading.Event()

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1
            shutdown_started.set()
            assert allow_shutdown.wait(timeout=5.0)

    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    errors: list[BaseException] = []

    def shutdown() -> None:
        try:
            runner.shutdown()
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=shutdown)
    second = threading.Thread(target=shutdown)
    first.start()
    assert shutdown_started.wait(timeout=5.0)
    second.start()
    assert second.is_alive()

    allow_shutdown.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert backend.shutdown_calls == 1


def test_cxx_backend_requires_shutdown_contract():
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(_QueryLifecycleBackend())

    with pytest.raises(RuntimeError, match="shutdown"):
        runner.shutdown()


def test_cxx_python_backend_poll_error_retains_result_handle_until_drop():
    class ErrorHandle:
        def __init__(self, request):
            self.task_id = FteTaskAttemptId.coerce(request["task_id"])
            self.task_context_info = dict(request["task_context_info"])
            self.worker_id = "native-worker-error"
            self.exchange_node_id = _flight_exchange_node_id_from_env()
            self.release_calls = 0

        def done(self):
            return True

        def get_result_sync(self):
            raise RuntimeError("planned Python backend poll failure")

        def release_result_payload(self):
            self.release_calls += 1

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.handle = None

        def worker_snapshots(self):
            return [
                {
                    "worker_id": "native-worker-error",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, tasks):
            if self.handle is not None:
                return []
            request = NativeFteWorkerManagerBackend._request_from_task(tasks[0])
            self.handle = ErrorHandle(request)
            return [self.handle]

        def task_input_stream_exhausted(self, _query_id, _source_node_ids):
            return []

        def fte_query_status(self, _query_id):
            assert self.handle is not None
            return {
                "failed": False,
                "finished": True,
                "selected_attempt_task_ids": [str(self.handle.task_id)],
            }

        def drop_query(self, _query_id):
            pass

        def shutdown(self):
            pass

    con = vane.connect()
    query_id = f"python-backend-poll-error-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)

    with pytest.raises(Exception, match="planned Python backend poll failure"):
        collect_result_stream(runner.run_plan(plan, con))

    assert backend.handle is not None
    assert backend.handle.release_calls == 0
    runner.drop_query_fragments(query_id)
    assert backend.handle.release_calls == 1
    con.close()


def test_cxx_python_backend_ack_error_retains_result_handle_until_drop():
    class ErrorHandle:
        def __init__(self, request):
            self.task_id = FteTaskAttemptId.coerce(request["task_id"])
            self.task_context_info = dict(request["task_context_info"])
            self.worker_id = "native-worker-ack-error"
            self.exchange_node_id = _flight_exchange_node_id_from_env()
            self.ack_calls = 0
            self.release_calls = 0

        def done(self):
            return True

        def get_result_sync(self):
            return vane.ray_cxx.RayTaskResult.no_output()

        def ack(self):
            self.ack_calls += 1
            raise RuntimeError("planned Python backend ack failure")

        def release_result_payload(self):
            self.release_calls += 1

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.handle = None

        def worker_snapshots(self):
            return [
                {
                    "worker_id": "native-worker-ack-error",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, tasks):
            if self.handle is not None:
                return []
            request = NativeFteWorkerManagerBackend._request_from_task(tasks[0])
            self.handle = ErrorHandle(request)
            return [self.handle]

        def task_input_stream_exhausted(self, _query_id, _source_node_ids):
            return []

        def fte_query_status(self, _query_id):
            assert self.handle is not None
            return {
                "failed": False,
                "finished": True,
                "selected_attempt_task_ids": [str(self.handle.task_id)],
            }

        def drop_query(self, _query_id):
            pass

        def shutdown(self):
            pass

    con = vane.connect()
    query_id = f"python-backend-ack-error-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)

    with pytest.raises(Exception, match="planned Python backend ack failure"):
        collect_result_stream(runner.run_plan(plan, con))

    assert backend.handle is not None
    assert backend.handle.ack_calls == 1
    assert backend.handle.release_calls == 0
    runner.drop_query_fragments(query_id)
    assert backend.handle.release_calls == 1
    con.close()


def test_cxx_python_backend_cleanup_preserves_utf8_when_bounding_query_id():
    class ErrorHandle:
        def __init__(self, request):
            self.task_id = FteTaskAttemptId.coerce(request["task_id"])
            self.task_context_info = dict(request["task_context_info"])
            self.worker_id = "native-worker-unicode-cleanup"
            self.exchange_node_id = _flight_exchange_node_id_from_env()
            self.release_enabled = False

        def done(self):
            return True

        def get_result_sync(self):
            return vane.ray_cxx.RayTaskResult.no_output()

        def ack(self):
            raise RuntimeError("planned Unicode cleanup ack failure")

        def release_result_payload(self):
            if not self.release_enabled:
                raise RuntimeError("planned Unicode cleanup release failure")

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.handle = None

        def worker_snapshots(self):
            return [
                {
                    "worker_id": "native-worker-unicode-cleanup",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, tasks):
            if self.handle is not None:
                return []
            request = NativeFteWorkerManagerBackend._request_from_task(tasks[0])
            self.handle = ErrorHandle(request)
            return [self.handle]

        def task_input_stream_exhausted(self, _query_id, _source_node_ids):
            return []

        def fte_query_status(self, _query_id):
            assert self.handle is not None
            return {
                "failed": False,
                "finished": True,
                "selected_attempt_task_ids": [str(self.handle.task_id)],
            }

        def drop_query(self, _query_id):
            pass

        def shutdown(self):
            pass

    con = vane.connect()
    query_id = "界" * 100
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)

    with pytest.raises(Exception, match="planned Unicode cleanup ack failure"):
        collect_result_stream(runner.run_plan(plan, con))

    with pytest.raises(Exception) as cleanup_error:
        runner.drop_query_fragments(query_id)
    assert "planned Unicode cleanup release failure" in str(cleanup_error.value)
    assert "..." in str(cleanup_error.value)

    assert backend.handle is not None
    backend.handle.release_enabled = True
    runner.drop_query_fragments(query_id)
    con.close()


def test_cxx_python_backend_finished_query_respects_exhausted_drain_timeout(monkeypatch):
    class PendingHandle:
        def __init__(self, request):
            self.task_id = FteTaskAttemptId.coerce(request["task_id"])
            self.task_context_info = dict(request["task_context_info"])
            self.worker_id = "native-worker-drain-timeout"
            self.exchange_node_id = _flight_exchange_node_id_from_env()
            self.ready = threading.Event()
            self.release_calls = 0

        def done(self):
            return self.ready.is_set()

        def get_result_sync(self):
            return vane.ray_cxx.RayTaskResult.no_output()

        def release_result_payload(self):
            self.release_calls += 1

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.handle = None

        def worker_snapshots(self):
            return [
                {
                    "worker_id": "native-worker-drain-timeout",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, tasks):
            if self.handle is not None:
                return []
            request = NativeFteWorkerManagerBackend._request_from_task(tasks[0])
            self.handle = PendingHandle(request)
            return [self.handle]

        def task_input_stream_exhausted(self, _query_id, _source_node_ids):
            return []

        def fte_query_status(self, _query_id):
            assert self.handle is not None
            return {
                "failed": False,
                "finished": True,
                "selected_attempt_task_ids": [str(self.handle.task_id)],
            }

        def drop_query(self, _query_id):
            pass

        def shutdown(self):
            pass

    monkeypatch.setenv("VANE_FTE_QUERY_WAIT_TIMEOUT_S", "0.000001")
    con = vane.connect()
    query_id = f"python-backend-drain-timeout-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    outcome = []

    def collect():
        try:
            collect_result_stream(runner.run_plan(plan, con))
        except BaseException as error:  # pragma: no cover - asserted below
            outcome.append(error)

    thread = threading.Thread(target=collect)
    thread.start()
    try:
        thread.join(timeout=1.0)
        if thread.is_alive():
            assert backend.handle is not None
            backend.handle.ready.set()
            thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert len(outcome) == 1
        assert "timed out draining Python backend FTE result handles" in str(outcome[0])
        assert backend.handle is not None
        assert backend.handle.release_calls == 0
        runner.drop_query_fragments(query_id)
        assert backend.handle.release_calls == 1
    finally:
        if backend.handle is not None:
            backend.handle.ready.set()
        if thread.is_alive():
            thread.join(timeout=2.0)
        con.close()


def test_cxx_run_plan_startup_failure_cleans_query_replay_snapshot():
    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.drop_calls = []

        def drop_query(self, query_id):
            self.drop_calls.append(str(query_id))

    con = vane.connect()
    con.execute("SET threads=3")
    query_id = f"stream-startup-cleanup-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    deferred_plan = pickle.loads(pickle.dumps(plan))
    assert deferred_plan.has_root() is False
    assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is None

    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    with pytest.raises(ValueError, match="has no root"):
        runner.run_plan(deferred_plan, con)

    assert backend.drop_calls == [query_id]
    assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is None
    con.close()


def test_cxx_run_plan_startup_and_cleanup_failures_are_aggregated():
    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.drop_calls = 0

        def drop_query(self, _query_id):
            self.drop_calls += 1
            if self.drop_calls == 1:
                raise RuntimeError("planned stream startup cleanup failure")

    con = vane.connect()
    query_id = f"stream-startup-cleanup-error-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    deferred_plan = pickle.loads(pickle.dumps(plan))

    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    try:
        with pytest.raises(RuntimeError) as error:
            runner.run_plan(deferred_plan, con)

        assert "has no root" in str(error.value)
        assert "planned stream startup cleanup failure" in str(error.value)
        assert backend.drop_calls == 1
        assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is not None

        runner.drop_query_fragments(query_id)
        assert backend.drop_calls == 2
        assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is None
    finally:
        vane.ray_cxx._cleanup_query_python_replay_state(query_id)
        con.close()


def test_native_cxx_run_copy_plan_failure_cleans_local_staging(tmp_path, monkeypatch):
    con, dst, query_id, plan = _captured_native_copy_plan(tmp_path, monkeypatch, local_staging=True)
    staging_roots: list[Path] = []

    def execute_fn(request):
        context = request["context"]
        staging_base = context["copy_output_base"]
        run_id = context["copy_output_run_id"]
        assert staging_base
        staging_root = Path(staging_base) / run_id
        output_file = staging_root / "native_worker_fail" / "part.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(b"partial-native-copy-output")
        staging_roots.append(staging_root)
        raise RuntimeError("planned native copy failure")

    backend = NativeFteWorkerManagerBackend(execute_fn=execute_fn)
    try:
        runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
        with pytest.raises(ValueError, match="planned native copy failure"):
            runner.run_copy_plan(plan, con)

        assert staging_roots
        for staging_root in staging_roots:
            assert not staging_root.exists()
        assert not dst.exists()
        assert not Path(str(dst) + ".duckdb_staging").exists()
        assert backend.pop_fte_result_handles(query_id) == []
    finally:
        backend.shutdown()
        con.close()


def test_native_cxx_run_copy_plan_preserves_worker_plan_exception_cause(tmp_path, monkeypatch):
    from vane import _native
    from vane._ray_errors import RemoteRayException

    con, dst, query_id, plan = _captured_native_copy_plan(tmp_path, monkeypatch, local_staging=True)
    submission_calls = []
    dropped_queries = []
    partial_staging_roots: list[Path] = []
    drop_observed_partial_output: list[bool] = []

    class Backend(_QueryLifecycleBackend):
        def worker_snapshots(self):
            return [
                {
                    "worker_id": "native-worker-0",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024,
                }
            ]

        def submit_tasks(self, tasks):
            submission_calls.append(len(tasks))
            context = tasks[0].context()
            staging_root = Path(context["copy_output_base"]) / context["copy_output_run_id"]
            partial_output = staging_root / "partial-submit-worker" / "part.parquet"
            partial_output.parent.mkdir(parents=True, exist_ok=True)
            partial_output.write_bytes(b"partial-submit-output")
            partial_staging_roots.append(staging_root)
            tasks[0].plan()
            return []

        def drop_query(self, actual_query_id):
            dropped_queries.append(actual_query_id)
            drop_observed_partial_output.append(any(root.exists() for root in partial_staging_roots))

    def fail_lookup(actual_query_id):
        raise vane.NotImplementedException(f"copy plan lookup sentinel for {actual_query_id}")

    monkeypatch.setattr(_native.ray_cxx, "_lookup_query_udf_registrations", fail_lookup)
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(Backend())
    try:
        with pytest.raises(
            RuntimeError,
            match=f"distributed worker task submission failed for query_id={query_id}",
        ) as exc_info:
            runner.run_copy_plan(plan, con)
    finally:
        con.close()

    assert isinstance(exc_info.value, RemoteRayException)
    assert isinstance(exc_info.value.__cause__, vane.NotImplementedException)
    assert exc_info.value.__cause__.__traceback__ is not None
    assert f"copy plan lookup sentinel for {query_id}" in str(exc_info.value.__cause__)
    assert submission_calls == [1]
    assert dropped_queries == [query_id]
    assert drop_observed_partial_output == [True]
    assert partial_staging_roots
    assert all(not root.exists() for root in partial_staging_roots)
    assert not dst.exists()


@pytest.mark.parametrize(
    ("fail_finalize", "expected_error_name", "expected_write_state"),
    (
        (False, "CopyResultUnavailableError", "committed"),
        (True, "CopyOutcomeUnknownError", None),
    ),
    ids=("committed", "outcome-unknown"),
)
def test_native_cxx_extension_write_preserves_terminal_state_when_result_marshalling_fails(
    fail_finalize,
    expected_error_name,
    expected_write_state,
):
    from vane.runners import copy_outcome

    class NoOutputHandle:
        def __init__(self, task, partition_id):
            query_id = task.context()["query_id"]
            self.task_context_info = task.task_context()
            self.task_id = FteTaskAttemptId(FteTaskId(query_id, 0, partition_id), 0)
            self.worker_id = "extension-write-worker"

        def done(self):
            return True

        def get_result_sync(self):
            return vane.ray_cxx.RayTaskResult.no_output()

        def ack(self):
            return None

        def release_result_payload(self):
            return None

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.handles = []
            self.drop_calls = []

        def worker_snapshots(self):
            return [
                {
                    "worker_id": "extension-write-worker",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024 * 1024 * 1024,
                }
            ]

        def submit_tasks(self, tasks):
            handles = [NoOutputHandle(task, len(self.handles) + index) for index, task in enumerate(tasks)]
            self.handles.extend(handles)
            return handles

        def task_input_stream_exhausted(self, _query_id, _source_node_ids):
            return []

        def fte_query_status(self, _query_id):
            return {
                "finished": True,
                "failed": False,
                "selected_attempt_task_ids": [str(handle.task_id) for handle in self.handles],
            }

        def drop_query(self, query_id):
            self.drop_calls.append(query_id)

        def shutdown(self):
            return None

    con = vane.connect()
    query_id = f"extension-result-marshalling-{uuid.uuid4()}"
    vane.ray_cxx._register_coordinator_only_extension_write_for_test(con)
    plan = vane.ray_cxx._make_coordinator_only_extension_write_plan_for_test(
        query_id,
        fail_finalize=fail_finalize,
        conn=con,
    )
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
    runner._fail_next_extension_result_marshalling_for_test()
    expected_error = getattr(copy_outcome, expected_error_name)

    try:
        with pytest.raises(expected_error) as error:
            runner.run_copy_plan(plan, con)

        assert error.value.operation_id == query_id
        assert error.value.safe_to_retry is False
        assert "planned extension result marshalling failure" in str(error.value)
        if expected_write_state is not None:
            assert error.value.write_state == expected_write_state
        else:
            assert "planned coordinator-only extension finalization failure" in str(error.value)
        assert backend.drop_calls == [query_id]
    finally:
        con.close()


def test_native_cxx_committed_copy_returns_backend_cleanup_warning(tmp_path, monkeypatch):
    con, _dst, relation = _capture_native_copy_relation(tmp_path, monkeypatch, local_staging=True)

    from vane.runners.local.runner import _InProcessFragmentExecutor

    backend = NativeFteWorkerManagerBackend(
        execute_fn=_InProcessFragmentExecutor(),
        max_running_tasks=2,
    )
    query_id = f"copy-cleanup-failure-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_write_relation(
        relation,
        query_id,
    ).to_physical_plan(con)
    original_drop_query = backend.drop_query
    drop_calls = []

    def failing_drop_query(actual_query_id):
        drop_calls.append(str(actual_query_id))
        original_drop_query(actual_query_id)
        raise RuntimeError("planned copy backend cleanup failure")

    backend.drop_query = failing_drop_query
    try:
        runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
        result = runner.run_copy_plan(plan, con)

        assert drop_calls == [query_id]
        assert result["copy_output_committed"] is True
        assert result["copy_runner_cleanup_pending"] is True
        assert any(
            "planned copy backend cleanup failure" in warning for warning in result["copy_runner_cleanup_warnings"]
        )
    finally:
        backend.shutdown()
        con.close()


def test_native_cxx_run_copy_plan_successive_local_staging_runs_use_distinct_paths(tmp_path, monkeypatch):
    con, dst, relation = _capture_native_copy_relation(tmp_path, monkeypatch, local_staging=True)

    from vane.runners.local.runner import _InProcessFragmentExecutor

    executor = _InProcessFragmentExecutor()
    backend = NativeFteWorkerManagerBackend(execute_fn=executor, max_running_tasks=2)
    results = []
    try:
        runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
        for _ in range(2):
            plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_write_relation(
                relation,
                str(uuid.uuid4()),
            ).to_physical_plan(con)
            result = runner.run_copy_plan(plan, con)
            results.append(result)

        first, second = results
        first_path = first["files"][0]["staging_path"]
        second_path = second["files"][0]["staging_path"]

        assert first["copy_output_run_id"] != second["copy_output_run_id"]
        for result in results:
            assert result["copy_total_ms"] >= 0
            assert result["copy_run_plan_ms"] >= 0
            assert result["copy_staging_write_ms"] >= 0
            assert result["copy_finalize_ms"] >= 0
            assert result["copy_cleanup_ms"] >= 0
            assert result["copy_runner_cleanup_ms"] >= 0
            assert result["copy_runner_cleanup_pending"] is False
            assert result["copy_runner_cleanup_warnings"] == []
            assert result["copy_selected_file_count"] == len(result["files"])
            assert result["copy_duplicate_file_count"] == 0
        assert first_path != second_path
        assert ".duckdb_staging" in first_path
        assert ".duckdb_staging" in second_path
        assert first["copy_output_run_id"] in first_path
        assert second["copy_output_run_id"] in second_path
        assert first["copy_output_run_id"] not in second_path
        assert second["copy_output_run_id"] not in first_path
        assert Path(first_path).parent.name.startswith("w_")
        assert Path(second_path).parent.name.startswith("w_")
        assert not Path(str(dst) + ".duckdb_staging").exists()
    finally:
        backend.shutdown()
        executor.close()
        con.close()


def test_in_process_fragment_executor_uses_thread_local_duckdb_resources(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from vane.runners.local import runner as local_runner

    class FakeCursor:
        def __init__(self, conn_id: int) -> None:
            self.conn_id = conn_id
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeConn:
        def __init__(self, conn_id: int) -> None:
            self.conn_id = conn_id
            self.closed = False
            self.executed: list[str] = []

        def execute(self, sql: str) -> None:
            self.executed.append(sql)

        def cursor(self) -> FakeCursor:
            return FakeCursor(self.conn_id)

        def close(self) -> None:
            self.closed = True

    conn_lock = threading.Lock()
    connections: list[FakeConn] = []

    def fake_connect() -> FakeConn:
        with conn_lock:
            conn = FakeConn(len(connections))
            connections.append(conn)
            return conn

    clone_lock = threading.Lock()
    active_clones = 0
    max_active_clones = 0

    class FakePlan:
        def clone(self, conn: FakeConn) -> tuple[str, int]:
            nonlocal active_clones, max_active_clones
            with clone_lock:
                active_clones += 1
                max_active_clones = max(max_active_clones, active_clones)
            try:
                time.sleep(0.05)
                return ("cloned", conn.conn_id)
            finally:
                with clone_lock:
                    active_clones -= 1

    runner_lock = threading.Lock()
    runner_ids: list[int] = []
    execute_barrier = threading.Barrier(2)

    class FakePlanRunner:
        def __init__(self) -> None:
            with runner_lock:
                self.runner_id = len(runner_ids)
                runner_ids.append(self.runner_id)

        def execute_native(
            self,
            cursor: FakeCursor,
            plan: tuple[str, int],
            *_args: Any,
        ) -> dict[str, int]:
            execute_barrier.wait(timeout=2.0)
            return {
                "conn_id": cursor.conn_id,
                "plan_conn_id": int(plan[1]),
                "runner_id": self.runner_id,
            }

    def fake_require(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "DistributedPhysicalPlanRunner":
            return FakePlanRunner
        if name == "merge_scan_split_batches":
            return lambda values: values
        raise AssertionError(f"unexpected ray_cxx attr: {name}")

    monkeypatch.setattr(vane, "connect", fake_connect)
    monkeypatch.setattr(local_runner, "require_ray_cxx_attr", fake_require)

    executor = local_runner._InProcessFragmentExecutor()
    requests = [
        {"fragment_plan": FakePlan(), "context": {}, "task_id": _task_id(partition_id)} for partition_id in range(2)
    ]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(executor, request) for request in requests]
            results = [future.result(timeout=5.0) for future in futures]

        assert {result["conn_id"] for result in results} == {0, 1}
        assert {result["plan_conn_id"] for result in results} == {0, 1}
        assert {result["runner_id"] for result in results} == {0, 1}
        assert max_active_clones == 1
    finally:
        executor.close()

    assert len(connections) == 2
    assert all(conn.closed for conn in connections)
    for conn in connections:
        assert "SET local_exchange_streaming=true" in conn.executed
        assert "SET local_exchange_buffer_bytes = '32MB'" in conn.executed
        assert "SET arrow_large_buffer_size=true" in conn.executed


def test_in_process_fragment_executor_close_does_not_release_live_resources(monkeypatch):
    from vane.runners.local import runner as local_runner

    execute_started = threading.Event()
    release_execute = threading.Event()
    cursor_interrupted = threading.Event()

    class FakeCursor:
        def __init__(self) -> None:
            self.closed = False

        def interrupt(self) -> None:
            cursor_interrupted.set()

        def close(self) -> None:
            self.closed = True

    class FakeConn:
        def __init__(self) -> None:
            self.cursor_instance = FakeCursor()
            self.closed = False

        def cursor(self) -> FakeCursor:
            return self.cursor_instance

        def close(self) -> None:
            self.closed = True

    class FakePlanRunner:
        def execute_native(self, *_args: Any) -> dict[str, bool]:
            execute_started.set()
            assert release_execute.wait(timeout=2.0)
            return {"ok": True}

    monkeypatch.setattr(
        NativeFteWorkerManagerBackend,
        "materialize_task_context",
        staticmethod(lambda *_args, **_kwargs: {}),
    )
    monkeypatch.setattr(local_runner, "require_ray_cxx_attr", lambda *_args, **_kwargs: lambda values: values)

    executor = local_runner._InProcessFragmentExecutor(close_timeout_s=1.0)
    conn = FakeConn()
    executor._connections.append(conn)
    monkeypatch.setattr(executor, "_get_conn", lambda: conn)
    monkeypatch.setattr(executor, "_get_plan_runner", lambda: FakePlanRunner())

    with ThreadPoolExecutor(max_workers=1) as pool:
        execution = pool.submit(
            executor,
            {"fragment_plan": object(), "context": {}, "task_id": _task_id(0)},
        )
        assert execute_started.wait(timeout=1.0)
        try:
            with pytest.raises(RuntimeError, match="did not drain.*active_executions=1"):
                executor.close(timeout_s=0.05)
            assert cursor_interrupted.is_set()
            assert conn.closed is False
        finally:
            release_execute.set()
        assert execution.result(timeout=2.0) == {"ok": True}

    executor.close(timeout_s=1.0)
    assert conn.closed is True
    assert conn.cursor_instance.closed is True


def test_in_process_fragment_executor_unregisters_cursor_before_close(monkeypatch):
    from vane.runners.local import runner as local_runner

    close_started = threading.Event()
    release_close = threading.Event()
    interrupt_calls = []

    class FakeCursor:
        def __init__(self) -> None:
            self.closed = False

        def interrupt(self) -> None:
            interrupt_calls.append("interrupt")
            if self.closed:
                raise RuntimeError("cursor already closed")

        def close(self) -> None:
            self.closed = True
            close_started.set()
            assert release_close.wait(timeout=2.0)

    class FakeConn:
        def __init__(self) -> None:
            self.cursor_instance = FakeCursor()
            self.closed = False

        def cursor(self) -> FakeCursor:
            return self.cursor_instance

        def close(self) -> None:
            self.closed = True

    class FakePlanRunner:
        def execute_native(self, *_args: Any) -> dict[str, bool]:
            return {"ok": True}

    monkeypatch.setattr(
        NativeFteWorkerManagerBackend,
        "materialize_task_context",
        staticmethod(lambda *_args, **_kwargs: {}),
    )
    monkeypatch.setattr(local_runner, "require_ray_cxx_attr", lambda *_args, **_kwargs: lambda values: values)

    executor = local_runner._InProcessFragmentExecutor(close_timeout_s=1.0)
    conn = FakeConn()
    executor._connections.append(conn)
    monkeypatch.setattr(executor, "_get_conn", lambda: conn)
    monkeypatch.setattr(executor, "_get_plan_runner", lambda: FakePlanRunner())

    with ThreadPoolExecutor(max_workers=1) as pool:
        execution = pool.submit(
            executor,
            {"fragment_plan": object(), "context": {}, "task_id": _task_id(0)},
        )
        assert close_started.wait(timeout=1.0)
        try:
            executor.request_shutdown()
            assert interrupt_calls == []
        finally:
            release_close.set()
        assert execution.result(timeout=2.0) == {"ok": True}

    executor.close(timeout_s=1.0)
    assert conn.closed is True


def test_in_process_fragment_executor_interrupt_ownership_blocks_cursor_close():
    from vane.runners.local import runner as local_runner

    first_interrupt_started = threading.Event()
    release_first_interrupt = threading.Event()
    close_progressed = threading.Event()
    close_blocked_on_lifecycle = threading.Event()
    close_started = threading.Event()
    close_finished = threading.Event()
    release_close = threading.Event()
    interrupt_during_close = threading.Event()
    close_thread_id = []
    shutdown_errors = []

    class OrderedCursorRegistry(set[Any]):
        def __init__(self, *values):
            super().__init__(values)
            self._values = list(values)

        def __iter__(self):
            return iter(list(self._values))

        def discard(self, value):
            super().discard(value)
            if value in self._values:
                self._values.remove(value)

        def clear(self):
            super().clear()
            self._values.clear()

    class ObservedCondition(threading.Condition):
        def __enter__(self):
            if close_thread_id and threading.get_ident() == close_thread_id[0]:
                if self._lock.acquire(blocking=False):
                    return self
                close_blocked_on_lifecycle.set()
                close_progressed.set()
                self._lock.acquire()
                return self
            return super().__enter__()

    class BlockingCursor:
        def interrupt(self) -> None:
            first_interrupt_started.set()
            assert release_first_interrupt.wait(timeout=2.0)

    class ClosingCursor:
        def __init__(self) -> None:
            self.closed = False
            self.interrupt_calls = 0

        def interrupt(self) -> None:
            self.interrupt_calls += 1
            if close_started.is_set() and not close_finished.is_set():
                interrupt_during_close.set()
            if self.closed:
                raise RuntimeError("cursor already closed")

        def close(self) -> None:
            self.closed = True
            close_started.set()
            close_progressed.set()
            try:
                assert release_close.wait(timeout=2.0)
            finally:
                close_finished.set()

    blocking_cursor = BlockingCursor()
    closing_cursor = ClosingCursor()
    executor = local_runner._InProcessFragmentExecutor(close_timeout_s=1.0)
    lifecycle_lock = threading.RLock()
    executor._resources_lock = lifecycle_lock
    executor._resources_condition = ObservedCondition(lifecycle_lock)
    executor._active_cursors = OrderedCursorRegistry(blocking_cursor, closing_cursor)

    def request_shutdown():
        try:
            executor.request_shutdown()
        except BaseException as exc:
            shutdown_errors.append(exc)

    def unregister_and_close():
        close_thread_id.append(threading.get_ident())
        executor._unregister_cursor(closing_cursor)
        closing_cursor.close()

    shutdown_thread = threading.Thread(target=request_shutdown)
    shutdown_thread.start()
    close_thread = threading.Thread(target=unregister_and_close)
    close_thread_started = False
    try:
        assert first_interrupt_started.wait(timeout=1.0)
        close_thread.start()
        close_thread_started = True
        assert close_progressed.wait(timeout=1.0)
        close_started_before_interrupts_finished = close_started.is_set()
        close_blocked_before_interrupts_finished = close_blocked_on_lifecycle.is_set()
        release_first_interrupt.set()
        shutdown_thread.join(timeout=2.0)
    finally:
        release_first_interrupt.set()
        release_close.set()
        shutdown_thread.join(timeout=2.0)
        if close_thread_started:
            close_thread.join(timeout=2.0)

    assert close_started_before_interrupts_finished is False
    assert close_blocked_before_interrupts_finished is True
    assert shutdown_thread.is_alive() is False
    assert close_thread.is_alive() is False
    assert shutdown_errors == []
    assert closing_cursor.interrupt_calls == 1
    assert interrupt_during_close.is_set() is False
    assert close_started.is_set()
    assert close_finished.is_set()
    executor._unregister_cursor(blocking_cursor)
    executor.close(timeout_s=1.0)


def test_in_process_fragment_executor_reports_active_cursor_interrupt_failure():
    from vane.runners.local import runner as local_runner

    class ActiveCursor:
        def interrupt(self) -> None:
            raise RuntimeError("active cursor interrupt failed")

    cursor = ActiveCursor()
    executor = local_runner._InProcessFragmentExecutor(close_timeout_s=1.0)
    executor._active_cursors.add(cursor)

    with pytest.raises(RuntimeError, match="active cursor interrupt failed"):
        executor.request_shutdown()

    executor._unregister_cursor(cursor)
    executor.close(timeout_s=1.0)


def test_in_process_fragment_executor_registration_failure_releases_execution_ownership(monkeypatch):
    from vane.runners.local import runner as local_runner

    class FakeCursor:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeConn:
        def __init__(self) -> None:
            self.cursor_instance = FakeCursor()

        def cursor(self) -> FakeCursor:
            return self.cursor_instance

    monkeypatch.setattr(
        NativeFteWorkerManagerBackend,
        "materialize_task_context",
        staticmethod(lambda *_args, **_kwargs: {}),
    )
    monkeypatch.setattr(local_runner, "require_ray_cxx_attr", lambda *_args, **_kwargs: lambda values: values)

    executor = local_runner._InProcessFragmentExecutor()
    conn = FakeConn()
    monkeypatch.setattr(executor, "_get_conn", lambda: conn)

    def fail_register(_cursor):
        raise RuntimeError("cursor registration failed")

    monkeypatch.setattr(executor, "_register_cursor", fail_register)

    with pytest.raises(RuntimeError, match="cursor registration failed"):
        executor({"fragment_plan": object(), "context": {}, "task_id": _task_id(0)})

    assert executor._in_flight == 0
    assert conn.cursor_instance.closed is True
    executor.close(timeout_s=0)


def test_native_cxx_run_copy_plan_selected_attempt_ignores_duplicate_copy_output(tmp_path, monkeypatch):
    con, dst, query_id, plan = _captured_native_copy_plan(tmp_path, monkeypatch, local_staging=True)

    import pyarrow as pa

    class CopyOutputHandle:
        def __init__(self, task_id, task_context_info, file_path: Path, rows: int):
            self.task_id = task_id
            self.task_context_info = dict(task_context_info)
            self.worker_id = "native-worker-0"
            self.exchange_node_id = _flight_exchange_node_id_from_env()
            self.file_path = file_path
            self.rows = rows
            self.file_size = file_path.stat().st_size
            self.get_result_calls = 0
            self.acked = False
            self.released = False

        def done(self):
            return True

        def get_result_sync(self):
            self.get_result_calls += 1
            table = pa.table(
                {
                    "file_path": [str(self.file_path)],
                    "rows": [self.rows],
                    "file_size_bytes": [self.file_size],
                    "footer_size_bytes": [None],
                    "column_statistics": [None],
                    "partition_keys": [None],
                }
            )
            return vane.ray_cxx.RayTaskResult.success([table], [], None)

        def ack(self):
            self.acked = True

        def release_result_payload(self):
            self.released = True

    class Backend(_QueryLifecycleBackend):
        def __init__(self):
            self.handles: list[CopyOutputHandle] = []
            self.selected_task_id: str | None = None
            self.duplicate_file: Path | None = None
            self.staging_root: Path | None = None
            self.drop_calls: list[str] = []

        def worker_snapshots(self):
            return [
                {
                    "worker_id": "native-worker-0",
                    "num_cpus": 1.0,
                    "num_gpus": 0.0,
                    "total_memory_bytes": 1024,
                }
            ]

        def submit_tasks(self, tasks):
            if self.handles:
                return []
            if not tasks:
                return []
            request = NativeFteWorkerManagerBackend._request_from_task(tasks[0])
            context = request["context"]
            staging_base = Path(context["copy_output_base"])
            run_id = str(context["copy_output_run_id"])
            self.staging_root = staging_base / run_id
            selected_file = self.staging_root / "selected" / "part.parquet"
            duplicate_file = self.staging_root / "duplicate" / "part.parquet"
            selected_file.parent.mkdir(parents=True, exist_ok=True)
            duplicate_file.parent.mkdir(parents=True, exist_ok=True)

            selected_conn = vane.connect()
            selected_conn.execute(
                f"COPY (select 101::integer as x) TO {_sql_string_literal(str(selected_file))} (FORMAT PARQUET)"
            )
            selected_conn.close()
            duplicate_conn = vane.connect()
            duplicate_conn.execute(
                f"COPY (select 999::integer as x) TO {_sql_string_literal(str(duplicate_file))} (FORMAT PARQUET)"
            )
            duplicate_conn.close()

            selected_task_id = FteTaskAttemptId.coerce(request["task_id"])
            duplicate_task_id = FteTaskAttemptId(
                FteTaskId(
                    selected_task_id.query_id,
                    selected_task_id.fragment_execution_id,
                    selected_task_id.partition_id,
                ),
                selected_task_id.attempt_id + 1,
            )
            selected = CopyOutputHandle(
                selected_task_id,
                request["task_context_info"],
                selected_file,
                rows=1,
            )
            duplicate = CopyOutputHandle(
                duplicate_task_id,
                request["task_context_info"],
                duplicate_file,
                rows=1,
            )
            self.handles = [selected, duplicate]
            self.selected_task_id = str(selected_task_id)
            self.duplicate_file = duplicate_file
            return list(self.handles)

        def task_input_stream_exhausted(self, _query_id, _source_node_ids):
            return []

        def fte_query_status(self, _query_id):
            assert self.selected_task_id is not None
            return {
                "finished": True,
                "failed": False,
                "selected_attempt_task_ids": [self.selected_task_id],
            }

        def pop_fte_result_handles(self, _query_id):
            return []

        def drop_query(self, query_id):
            self.drop_calls.append(str(query_id))

        def shutdown(self):
            pass

    backend = Backend()
    try:
        runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
        result = runner.run_copy_plan(plan, con)

        assert result["copy_selected_file_count"] == 1
        assert result["copy_duplicate_file_count"] == 0
        assert result["rows_copied"] == 1
        assert con.sql(f"select list(x order by x) from read_parquet('{dst}')").fetchone()[0] == [101]
        assert backend.handles[0].get_result_calls == 1
        assert backend.handles[1].get_result_calls == 0
        assert backend.handles[0].acked is True
        assert backend.handles[1].acked is False
        assert all(handle.released for handle in backend.handles)
        assert backend.duplicate_file is not None
        assert not backend.duplicate_file.exists()
        assert backend.staging_root is not None
        assert not backend.staging_root.exists()
        assert backend.drop_calls == [query_id]
    finally:
        con.close()


def test_native_cxx_run_copy_plan_failure_cleans_direct_write_run(tmp_path, monkeypatch):
    con, dst, query_id, plan = _captured_native_copy_plan(tmp_path, monkeypatch, local_staging=False)
    run_dirs: list[Path] = []

    def execute_fn(request):
        context = request["context"]
        assert context["copy_output_base"] == ""
        run_id = context["copy_output_run_id"]
        remote_base = context["copy_output_remote_base"]
        run_dir = Path(remote_base) / f"_vane_direct_write_{run_id}"
        output_file = run_dir / "native_worker_fail" / "part.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(b"partial-native-direct-copy-output")
        run_dirs.append(run_dir)
        raise RuntimeError("planned native direct copy failure")

    backend = NativeFteWorkerManagerBackend(execute_fn=execute_fn)
    try:
        runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
        with pytest.raises(ValueError, match="planned native direct copy failure"):
            runner.run_copy_plan(plan, con)

        assert run_dirs
        for run_dir in run_dirs:
            assert not run_dir.exists()
        assert not Path(str(dst) + ".duckdb_commit").exists()
        assert backend.pop_fte_result_handles(query_id) == []
    finally:
        backend.shutdown()
        con.close()


def test_native_cxx_run_copy_plan_cancellation_cleans_local_staging(tmp_path, monkeypatch):
    con, dst, query_id, plan = _captured_native_copy_plan(tmp_path, monkeypatch, local_staging=True)
    partial_written = threading.Event()
    release_worker = threading.Event()
    staging_roots: list[Path] = []

    def execute_fn(request):
        context = request["context"]
        staging_base = context["copy_output_base"]
        run_id = context["copy_output_run_id"]
        assert staging_base
        staging_root = Path(staging_base) / run_id
        output_file = staging_root / "native_worker_cancel" / "part.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(b"partial-native-copy-output-before-cancel")
        staging_roots.append(staging_root)
        partial_written.set()
        release_worker.wait(timeout=5.0)
        return {"unexpected": True}

    backend = NativeFteWorkerManagerBackend(execute_fn=execute_fn)
    outcomes: list[Any] = []

    def run_copy_plan():
        runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
        try:
            outcomes.append(runner.run_copy_plan(plan, con))
        except BaseException as exc:  # pragma: no cover - asserted below
            outcomes.append(exc)

    thread = threading.Thread(target=run_copy_plan)
    thread.start()
    try:
        assert partial_written.wait(timeout=2.0)
        assert staging_roots
        assert any(staging_root.exists() for staging_root in staging_roots)

        backend.drop_query(query_id)

        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert outcomes
        assert isinstance(outcomes[0], ValueError)
        assert "canceled" in str(outcomes[0])
        for staging_root in staging_roots:
            assert not staging_root.exists()
        assert not dst.exists()
        assert not Path(str(dst) + ".duckdb_staging").exists()
        assert backend.pop_fte_result_handles(query_id) == []
    finally:
        release_worker.set()
        if thread.is_alive():
            backend.drop_query(query_id)
            thread.join(timeout=5.0)
        backend.shutdown()
        con.close()


def test_native_cxx_run_copy_plan_cancellation_cleans_direct_write_run(tmp_path, monkeypatch):
    con, dst, query_id, plan = _captured_native_copy_plan(tmp_path, monkeypatch, local_staging=False)
    partial_written = threading.Event()
    release_worker = threading.Event()
    run_dirs: list[Path] = []

    def execute_fn(request):
        context = request["context"]
        assert context["copy_output_base"] == ""
        run_id = context["copy_output_run_id"]
        remote_base = context["copy_output_remote_base"]
        run_dir = Path(remote_base) / f"_vane_direct_write_{run_id}"
        output_file = run_dir / "native_worker_cancel" / "part.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(b"partial-native-direct-copy-output-before-cancel")
        run_dirs.append(run_dir)
        partial_written.set()
        release_worker.wait(timeout=5.0)
        return {"unexpected": True}

    backend = NativeFteWorkerManagerBackend(execute_fn=execute_fn)
    outcomes: list[Any] = []

    def run_copy_plan():
        runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
        try:
            outcomes.append(runner.run_copy_plan(plan, con))
        except BaseException as exc:  # pragma: no cover - asserted below
            outcomes.append(exc)

    thread = threading.Thread(target=run_copy_plan)
    thread.start()
    try:
        assert partial_written.wait(timeout=2.0)
        assert run_dirs
        assert any(run_dir.exists() for run_dir in run_dirs)

        backend.drop_query(query_id)

        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert outcomes
        assert isinstance(outcomes[0], ValueError)
        assert "canceled" in str(outcomes[0])
        for run_dir in run_dirs:
            assert not run_dir.exists()
        assert not Path(str(dst) + ".duckdb_commit").exists()
        assert backend.pop_fte_result_handles(query_id) == []
    finally:
        release_worker.set()
        if thread.is_alive():
            backend.drop_query(query_id)
            thread.join(timeout=5.0)
        backend.shutdown()
        con.close()


def test_native_task_result_handle_normalizes_none_as_no_output_for_cxx():
    def execute_fn(_request):
        return None

    worker = NativeWorkerHandle("worker-1", execute_fn)
    try:
        task = _task_id(6)
        worker.fte_create_task({"task_id": task, "fragment_id": "q:scan"})
        handle = NativeTaskResultHandle(worker, task)

        for _ in range(100):
            if handle.done():
                break
            time.sleep(0.01)

        result = handle.get_result_sync()
        assert result.ok is True
        assert result.has_output is False
    finally:
        worker.fte_drop_query("q")
        worker.shutdown()
