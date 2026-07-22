# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from vane.runners.fte import TaskResultState
from vane.runners.fte.backends.ray import (
    RayTaskResultHandleAdapter,
    RayWorkerHandleAdapter,
    RayWorkerManagerBackend,
)


class _FakeWorkerHandle:
    worker_id = "worker-1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def fte_create_task(self, request):
        self.calls.append(("fte_create_task", (request,)))
        return {"state": "CREATED", "request": request}

    def fte_add_splits(self, task_id, source_node_id, splits):
        self.calls.append(("fte_add_splits", (task_id, source_node_id, splits)))
        return {"state": "UPDATED", "split_count": len(splits)}

    def fte_no_more_splits(self, task_id, source_node_id):
        self.calls.append(("fte_no_more_splits", (task_id, source_node_id)))
        return {"state": "UPDATED"}

    def fte_update_task(self, task_id, update):
        self.calls.append(("fte_update_task", (task_id, update)))
        return {"state": "UPDATED", "update": update}

    def fte_wait_task_status(self, task_id, min_version=None, timeout_s=None):
        self.calls.append(("fte_wait_task_status", (task_id, min_version, timeout_s)))
        return {"state": "RUNNING", "version": min_version}

    def fte_cancel_task(self, task_id):
        self.calls.append(("fte_cancel_task", (task_id,)))
        return {"state": "CANCELED"}

    def optional_method(self):
        return "delegated"


def test_ray_worker_handle_adapter_delegates_worker_protocol_methods():
    fake = _FakeWorkerHandle()
    adapter = RayWorkerHandleAdapter(fake)

    assert adapter.worker_id == "worker-1"
    assert adapter.fte_create_task({"task": 1})["state"] == "CREATED"
    assert adapter.fte_add_splits("task.0", "source-a", [{"sequence_id": 1}]) == {
        "state": "UPDATED",
        "split_count": 1,
    }
    assert adapter.fte_no_more_splits("task.0", "source-a") == {"state": "UPDATED"}
    assert adapter.fte_update_task("task.0", {"x": 1})["update"] == {"x": 1}
    assert adapter.fte_wait_task_status("task.0", 3, 0.5) == {"state": "RUNNING", "version": 3}
    assert adapter.fte_cancel_task("task.0") == {"state": "CANCELED"}
    assert adapter.optional_method() == "delegated"

    assert fake.calls == [
        ("fte_create_task", ({"task": 1},)),
        ("fte_add_splits", ("task.0", "source-a", [{"sequence_id": 1}])),
        ("fte_no_more_splits", ("task.0", "source-a")),
        ("fte_update_task", ("task.0", {"x": 1})),
        ("fte_wait_task_status", ("task.0", 3, 0.5)),
        ("fte_cancel_task", ("task.0",)),
    ]


class _FakeDoneHandle:
    def __init__(self, *, done: bool, result=None, error: BaseException | None = None) -> None:
        self.task_id = "query-a.1.2.3"
        self.worker_id = "worker-1"
        self.task_context_info = {"query_id": "query-a", "task_id": 2}
        self._done = done
        self._result = result
        self._error = error
        self.acked = False

    def done(self):
        return self._done

    def get_result_sync(self):
        if self._error is not None:
            raise self._error
        return self._result

    def ack(self):
        self.acked = True


def test_ray_task_result_handle_adapter_normalizes_done_handle_states():
    not_ready = RayTaskResultHandleAdapter(_FakeDoneHandle(done=False, result="ignored"))
    assert not_ready.task_context() == {"query_id": "query-a", "task_id": 2}
    assert not_ready.fte_task_id() == "query-a.1.2.3"
    assert not_ready.worker_id() == "worker-1"
    assert not_ready.poll().state is TaskResultState.NOT_READY

    no_output = RayTaskResultHandleAdapter(_FakeDoneHandle(done=True, result=None))
    assert no_output.poll().state is TaskResultState.NO_OUTPUT

    output = RayTaskResultHandleAdapter(_FakeDoneHandle(done=True, result="payload"))
    poll = output.poll()
    assert poll.state is TaskResultState.MATERIALIZED_OUTPUT
    assert poll.output == "payload"

    error = RayTaskResultHandleAdapter(_FakeDoneHandle(done=True, error=RuntimeError("boom")))
    poll = error.poll()
    assert poll.state is TaskResultState.ERROR
    assert isinstance(poll.error, RuntimeError)

    raw = _FakeDoneHandle(done=True, result="payload")
    RayTaskResultHandleAdapter(raw).ack()
    assert raw.acked is True


class _FakePollHandle:
    worker_id = "worker-2"
    task_context_info = {"query_id": "query-b"}
    task_id = "query-b.1.0.0"

    def __init__(self, value):
        self.value = value
        self.acked = False

    def poll(self):
        return self.value

    def AckPollResult(self):
        self.acked = True


@pytest.mark.parametrize(
    ("raw_poll", "expected_state", "expected_output"),
    [
        ((False, None), TaskResultState.NOT_READY, None),
        ((True, None), TaskResultState.NO_OUTPUT, None),
        ((True, (False, "ignored")), TaskResultState.NO_OUTPUT, None),
        ((True, (True, "payload")), TaskResultState.MATERIALIZED_OUTPUT, "payload"),
        ({"state": "MATERIALIZED_OUTPUT", "output": "payload"}, TaskResultState.MATERIALIZED_OUTPUT, "payload"),
    ],
)
def test_ray_task_result_handle_adapter_normalizes_poll_method(raw_poll, expected_state, expected_output):
    raw = _FakePollHandle(raw_poll)
    adapter = RayTaskResultHandleAdapter(raw)

    poll = adapter.poll()

    assert poll.state is expected_state
    assert poll.output == expected_output
    adapter.ack()
    assert raw.acked is True


class _FakeCoordinator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.submitted_handles = [_FakeDoneHandle(done=True, result="submitted")]
        self.exhaustion_handles = [_FakeDoneHandle(done=True, result="exhausted")]
        self.popped_handles = [_FakeDoneHandle(done=True, result="popped")]

    def worker_snapshots(self):
        self.calls.append(("worker_snapshots", ()))
        return [{"worker_id": "worker-1"}]

    def submit_tasks(self, tasks):
        self.calls.append(("submit_tasks", (tasks,)))
        return self.submitted_handles

    def task_input_stream_exhausted_for_query(self, query_id, source_node_ids):
        self.calls.append(("task_input_stream_exhausted_for_query", (query_id, source_node_ids)))
        return self.exhaustion_handles

    def wait_fte_query(self, query_id, timeout_s):
        self.calls.append(("wait_fte_query", (query_id, timeout_s)))
        return {"query_id": query_id, "finished": True, "failed": False}

    def fte_query_status(self, query_id):
        self.calls.append(("fte_query_status", (query_id,)))
        return {
            "query_id": query_id,
            "finished": True,
            "failed": False,
            "selected_attempt_task_ids": ["query-a.0.0.0"],
        }

    def pop_fte_result_handles(self, query_id):
        self.calls.append(("pop_fte_result_handles", (query_id,)))
        return self.popped_handles

    def fte_drop_query(self, query_id):
        self.calls.append(("fte_drop_query", (query_id,)))

    def shutdown(self):
        self.calls.append(("shutdown", ()))


def test_ray_worker_manager_backend_delegates_and_collects_result_handles():
    coordinator = _FakeCoordinator()
    backend = RayWorkerManagerBackend(coordinator)

    assert backend.worker_snapshots() == [{"worker_id": "worker-1"}]
    submitted = backend.submit_tasks(("task-a",))
    assert len(submitted) == 1
    assert submitted[0].poll().output == "submitted"

    backend.task_input_stream_exhausted("query-a", ("source-a",))
    handles = backend.wait_query("query-a", 2.0)

    assert [handle.poll().output for handle in handles] == ["submitted", "exhausted", "popped"]

    backend.drop_query("query-a")
    backend.shutdown()

    assert coordinator.calls == [
        ("worker_snapshots", ()),
        ("submit_tasks", (["task-a"],)),
        ("task_input_stream_exhausted_for_query", ("query-a", ["source-a"])),
        ("wait_fte_query", ("query-a", 2.0)),
        ("pop_fte_result_handles", ("query-a",)),
        ("fte_drop_query", ("query-a",)),
        ("shutdown", ()),
    ]


def test_ray_worker_manager_backend_exposes_cxx_query_status_contract():
    coordinator = _FakeCoordinator()
    backend = RayWorkerManagerBackend(coordinator)

    assert backend.fte_query_status("query-a") == {
        "query_id": "query-a",
        "finished": True,
        "failed": False,
        "selected_attempt_task_ids": ["query-a.0.0.0"],
    }
    assert coordinator.calls == [("fte_query_status", ("query-a",))]
