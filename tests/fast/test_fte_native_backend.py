# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

import vane
from vane.runners.fte import FteTaskAttemptId, FteTaskId, FteTaskState, TaskResultState
from vane.runners.fte.backends.native import (
    NativeFteWorkerManagerBackend,
    NativeTaskResultHandle,
    NativeWorkerHandle,
)
from vane.runners.fte.backends.native.backend import _flight_exchange_node_id_from_env
from vane.runners.fte.fte_config import FTE_WORKER_RUNTIME
from vane.runners.progress import build_progress_snapshot


def _task_id(partition_id: int, *, query_id: str = "q") -> dict[str, int | str]:
    return {
        "query_id": query_id,
        "fragment_execution_id": 0,
        "partition_id": partition_id,
        "attempt_id": 0,
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
        exchange_sink_instance: Any = None,
    ) -> None:
        self._name = name
        self._context = dict(context or {})
        self._task_context = dict(task_context or {})
        self._inputs = dict(inputs or {})
        self._plan = {"plan": "native"} if plan is None else plan
        self._exchange_sink_instance = exchange_sink_instance

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

    def exchange_sink_instance(self):
        return self._exchange_sink_instance


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
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        captured[0],
        query_id,
    ).to_physical_plan(con)
    assert plan.scan_task_descriptor_map()
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

        backend.drop_query("query-drop")

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
                            "kind": "scan_task",
                            "data": b"not-a-real-scan-descriptor",
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
    assert callback_events[-1][1] is poll.error

    with pytest.raises(RuntimeError, match="status identity mismatch"):
        handle.status_snapshot()
    assert callback_events[-1][1] is not None

    with pytest.raises(RuntimeError, match="status identity mismatch"):
        handle.info_snapshot()
    assert callback_events[-1][1] is not None


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
                        "7": [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
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
            "1": {"kind": "scan_task", "data": b"scan-descriptor"},
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
    assert "scan_task:1" not in request["context"]
    assert "exchange_source_task:3" not in request["context"]
    assert "scan_task_nodes" not in request["context"]
    assert "exchange_source_task_nodes" not in request["context"]

    scan_splits = request["initial_splits"]["1"]
    assert len(scan_splits) == 1
    assert scan_splits[0]["kind"] == "scan_task"
    assert scan_splits[0]["sequence_id"] == 0
    assert scan_splits[0]["data"] == b"scan-descriptor"

    exchange_splits = request["initial_splits"]["3"]
    assert [split["sequence_id"] for split in exchange_splits] == [0, 1]
    assert [split["source_partition_id"] for split in exchange_splits] == [0, 1]
    assert [split["data"]["partition_indices"] for split in exchange_splits] == [[0], [1]]


def test_native_fte_runtime_removes_dynamic_initial_splits_before_execute():
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
                        "7": [{"sequence_id": 0, "kind": "scan_task", "data": b"scan"}],
                    },
                }
            ]
        )

        time.sleep(0.02)
        assert executed.is_set() is False

        backend.task_input_stream_exhausted("query-dynamic-scan", ["7"])
        outputs = backend.wait_query("query-dynamic-scan", 2.0)

        assert outputs == [{"ok": True}]
        assert executed.is_set() is True
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


def test_cxx_distributed_runner_accepts_python_backend_without_ray_worker_startup():
    class Backend:
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

    assert backend.worker_snapshots_calls == 1
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
                            "stage_ids": [],
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
                    "stage_ids": [],
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

    class Backend:
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

    parts = list(iter(runner.run_plan(plan, con)))

    assert parts == []
    assert backend.submitted_task_names
    assert backend.status_calls
    assert all(call[0] == query_id for call in backend.exhausted_calls)
    assert backend.status_calls[-1] == query_id
    assert all(handle.acked for handle in backend.handles)
    assert all(handle.released for handle in backend.handles)


@pytest.mark.parametrize(
    ("exchange_sink_instance", "expect_released_after_run"),
    [
        (None, False),
        (
            {
                "task_partition_id": 0,
                "attempt_id": 0,
                "output_partition_count": 1,
                "output_location": "/tmp/fake-exchange-output",
            },
            True,
        ),
    ],
    ids=["final-output", "exchange-output"],
)
def test_cxx_streaming_runner_output_handle_release_lifecycle(
    exchange_sink_instance,
    expect_released_after_run: bool,
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
            self.released = True

    class Backend:
        def __init__(self):
            self.handles = []
            self.drop_calls = []

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
            pass

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

    parts = list(iter(runner.run_plan(plan, con)))

    assert parts
    assert all(handle.acked for handle in backend.handles)
    assert all(handle.released is expect_released_after_run for handle in backend.handles)
    assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is not None

    runner.drop_query_fragments(query_id)

    assert backend.drop_calls == [query_id]
    assert all(handle.released for handle in backend.handles)
    assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is None


def test_cxx_backend_drop_query_failure_is_not_silently_accepted():
    class Backend:
        def __init__(self):
            self.drop_calls = []

        def drop_query(self, query_id):
            self.drop_calls.append(str(query_id))
            raise RuntimeError("planned backend drop failure")

    query_id = f"backend-drop-failure-{uuid.uuid4()}"
    backend = Backend()
    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)

    with pytest.raises(RuntimeError, match="planned backend drop failure"):
        runner.drop_query_fragments(query_id)

    assert backend.drop_calls == [query_id]


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

    class Backend:
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
        list(runner.run_plan(plan, con))

    assert backend.handle is not None
    assert backend.handle.release_calls == 0
    runner.drop_query_fragments(query_id)
    assert backend.handle.release_calls == 1
    con.close()


def test_cxx_run_plan_startup_failure_cleans_query_replay_snapshot():
    class Backend:
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
    class Backend:
        def drop_query(self, _query_id):
            raise RuntimeError("planned stream startup cleanup failure")

    con = vane.connect()
    query_id = f"stream-startup-cleanup-error-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql("SELECT 1 AS i"),
        query_id,
    ).to_physical_plan(con)
    deferred_plan = pickle.loads(pickle.dumps(plan))

    runner = vane.ray_cxx.DistributedPhysicalPlanRunner(Backend())
    with pytest.raises(RuntimeError) as error:
        runner.run_plan(deferred_plan, con)

    assert "has no root" in str(error.value)
    assert "planned stream startup cleanup failure" in str(error.value)
    assert vane.ray_cxx._lookup_query_connection_snapshot(query_id) is None
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


def test_native_cxx_run_copy_plan_surfaces_backend_cleanup_failure(tmp_path, monkeypatch):
    con, _dst, relation = _capture_native_copy_relation(tmp_path, monkeypatch, local_staging=True)

    from vane.runners.local.runner import _InProcessFragmentExecutor

    backend = NativeFteWorkerManagerBackend(
        execute_fn=_InProcessFragmentExecutor(),
        max_running_tasks=2,
    )
    query_id = f"copy-cleanup-failure-{uuid.uuid4()}"
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
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
        with pytest.raises(RuntimeError, match="planned copy backend cleanup failure"):
            runner.run_copy_plan(plan, con)

        assert drop_calls == [query_id]
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
            plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
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
        if name == "merge_scan_task_descriptors":
            return lambda values: values
        raise AssertionError(f"unexpected ray_cxx attr: {name}")

    monkeypatch.setattr(vane, "connect", fake_connect)
    monkeypatch.setattr(local_runner, "require_ray_cxx_attr", fake_require)

    executor = local_runner._InProcessFragmentExecutor()
    request = {"fragment_plan": FakePlan(), "context": {}}
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(executor, request) for _ in range(2)]
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

    class Backend:
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
        assert all(handle.acked for handle in backend.handles)
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
