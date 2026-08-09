# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

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

    def enqueue_fte_cancel_task(self, task_id):
        self.calls.append(("enqueue_fte_cancel_task", (task_id,)))
        return ("cancel", task_id)

    def resolve_fte_cancel_task(self, cancellation):
        self.calls.append(("resolve_fte_cancel_task", (cancellation,)))
        return {"state": "CANCELED", "cancellation": cancellation}

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
    assert adapter.fte_cancel_task("task.0") == {
        "state": "CANCELED",
        "cancellation": ("cancel", "task.0"),
    }
    cancellation = adapter.enqueue_fte_cancel_task("task.1")
    assert cancellation == ("cancel", "task.1")
    assert adapter.resolve_fte_cancel_task(cancellation) == {
        "state": "CANCELED",
        "cancellation": ("cancel", "task.1"),
    }
    assert adapter.optional_method() == "delegated"

    assert fake.calls == [
        ("fte_create_task", ({"task": 1},)),
        ("fte_add_splits", ("task.0", "source-a", [{"sequence_id": 1}])),
        ("fte_no_more_splits", ("task.0", "source-a")),
        ("fte_update_task", ("task.0", {"x": 1})),
        ("fte_wait_task_status", ("task.0", 3, 0.5)),
        ("enqueue_fte_cancel_task", ("task.0",)),
        ("resolve_fte_cancel_task", (("cancel", "task.0"),)),
        ("enqueue_fte_cancel_task", ("task.1",)),
        ("resolve_fte_cancel_task", (("cancel", "task.1"),)),
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
        self.release_count = 0

    def done(self):
        return self._done

    def get_result_sync(self):
        if self._error is not None:
            raise self._error
        return self._result

    def ack(self):
        self.acked = True

    def release_result_payload(self):
        self.release_count += 1


class _FailTwiceDoneHandle(_FakeDoneHandle):
    def release_result_payload(self):
        self.release_count += 1
        if self.release_count <= 2:
            raise RuntimeError("planned release failure")


class _FailOnceDoneHandle(_FakeDoneHandle):
    def release_result_payload(self):
        self.release_count += 1
        if self.release_count == 1:
            raise RuntimeError("planned release failure")


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

    def materialization_barrier_completed(self, query_id, node_id):
        self.calls.append(("materialization_barrier_completed", (query_id, node_id)))

    def wait_fte_query(self, query_id, timeout_s):
        self.calls.append(("wait_fte_query", (query_id, timeout_s)))
        return {"query_id": query_id, "finished": True, "failed": False}

    def fte_query_status(self, query_id, task_context_filter):
        self.calls.append(("fte_query_status", (query_id, task_context_filter)))
        result = {
            "query_id": query_id,
            "finished": True,
            "failed": False,
            "selected_attempt_task_ids": ["query-a.0.0.0"],
        }
        if task_context_filter is not None:
            result["matched"] = bool(task_context_filter)
        return result

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
    backend.register_query_owner("query-a", "query-a")

    assert backend.worker_snapshots() == [{"worker_id": "worker-1"}]
    submitted = backend.submit_tasks(({"query_id": "query-a", "resource_query_id": "query-a"},))
    assert len(submitted) == 1
    assert submitted[0].poll().output == "submitted"

    exhausted = backend.task_input_stream_exhausted("query-a", ("source-a",))
    assert [handle.poll().output for handle in exhausted] == ["exhausted"]
    backend.materialization_barrier_completed("query-a", "7")
    handles = backend.wait_query("query-a", 2.0)

    assert [handle.poll().output for handle in handles] == ["popped"]

    backend.drop_query("query-a")
    backend.shutdown()

    assert coordinator.calls == [
        ("worker_snapshots", ()),
        ("submit_tasks", ([{"query_id": "query-a", "resource_query_id": "query-a"}],)),
        ("task_input_stream_exhausted_for_query", ("query-a", ["source-a"])),
        ("materialization_barrier_completed", ("query-a", "7")),
        ("wait_fte_query", ("query-a", 2.0)),
        ("pop_fte_result_handles", ("query-a",)),
        ("fte_drop_query", ("query-a",)),
        ("shutdown", ()),
    ]


def test_ray_worker_manager_backend_requires_fte_drop_query_contract():
    coordinator = _FakeCoordinator()
    setattr(coordinator, "fte_drop_query", None)
    backend = RayWorkerManagerBackend(coordinator)
    backend.register_query_owner("query-a", "query-a")

    with pytest.raises(TypeError, match="callable fte_drop_query"):
        backend.drop_query("query-a")

    setattr(
        coordinator,
        "fte_drop_query",
        lambda query_id: coordinator.calls.append(("fte_drop_query", (query_id,))),
    )
    backend.drop_query("query-a")
    assert coordinator.calls == [("fte_drop_query", ("query-a",))]


def test_ray_worker_manager_backend_requires_shutdown_contract():
    coordinator = _FakeCoordinator()
    setattr(coordinator, "shutdown", None)
    backend = RayWorkerManagerBackend(coordinator)

    with pytest.raises(TypeError, match="callable shutdown"):
        backend.shutdown()

    setattr(coordinator, "shutdown", lambda: coordinator.calls.append(("shutdown", ())))
    backend.shutdown()
    assert coordinator.calls == [("shutdown", ())]


def test_ray_worker_manager_backend_exposes_cxx_query_status_contract():
    coordinator = _FakeCoordinator()
    backend = RayWorkerManagerBackend(coordinator)
    backend.register_query_owner("query-a", "query-a")

    assert backend.fte_query_status("query-a") == {
        "query_id": "query-a",
        "finished": True,
        "failed": False,
        "selected_attempt_task_ids": ["query-a.0.0.0"],
    }
    task_context = {
        "query_idx": 1,
        "last_node_id": 2,
        "task_id": 3,
        "node_ids": [2],
    }
    assert backend.fte_query_status("query-a", (task_context,)) == {
        "query_id": "query-a",
        "finished": True,
        "failed": False,
        "matched": True,
        "selected_attempt_task_ids": ["query-a.0.0.0"],
    }
    assert coordinator.calls == [
        ("fte_query_status", ("query-a", None)),
        ("fte_query_status", ("query-a", [task_context])),
    ]


def test_ray_worker_manager_backend_releases_popped_handles_excluded_by_filter():
    coordinator = _FakeCoordinator()
    selected = _FakeDoneHandle(done=True, result="selected")
    discarded = _FakeDoneHandle(done=True, result="discarded")
    discarded.task_context_info = {"query_id": "query-a", "task_id": 3}
    coordinator.popped_handles = [selected, discarded]
    backend = RayWorkerManagerBackend(coordinator)
    backend.register_query_owner("query-a", "query-a")

    handles = backend.wait_query(
        "query-a",
        1.0,
        ({"query_id": "query-a", "task_id": 2},),
    )

    assert [handle.poll().output for handle in handles] == ["selected"]
    assert selected.release_count == 0
    assert discarded.release_count == 1


def test_ray_worker_manager_backend_filters_full_task_context_values():
    coordinator = _FakeCoordinator()
    selected = _FakeDoneHandle(done=True, result="selected")
    selected.task_context_info = {
        "query_idx": 1,
        "last_node_id": 2,
        "task_id": 3,
        "node_ids": [2, 4],
    }
    coordinator.popped_handles = [selected]
    backend = RayWorkerManagerBackend(coordinator)
    backend.register_query_owner("query-a", "query-a")

    handles = backend.wait_query(
        "query-a",
        1.0,
        (
            {
                "query_idx": 1,
                "last_node_id": 2,
                "task_id": 3,
                "node_ids": [2, 4],
            },
        ),
    )

    assert [handle.poll().output for handle in handles] == ["selected"]
    assert selected.release_count == 0


def test_ray_worker_manager_backend_releases_batch_when_handle_filtering_fails():
    coordinator = _FakeCoordinator()
    first = _FakeDoneHandle(done=True, result="first")
    second = _FakeDoneHandle(done=True, result="second")

    def fail_task_context():
        raise RuntimeError("planned task context failure")

    second.task_context = fail_task_context
    coordinator.popped_handles = [first, second]
    backend = RayWorkerManagerBackend(coordinator)
    backend.register_query_owner("query-a", "query-a")

    with pytest.raises(RuntimeError, match="planned task context failure"):
        backend.wait_query(
            "query-a",
            1.0,
            ({"query_id": "query-a", "task_id": 2},),
        )

    assert first.release_count == 1
    assert second.release_count == 1


def test_ray_worker_manager_backend_releases_selected_handles_when_filtered_cleanup_fails():
    coordinator = _FakeCoordinator()
    selected = _FakeDoneHandle(done=True, result="selected")
    discarded = _FailOnceDoneHandle(done=True, result="discarded")
    discarded.task_context_info = {"query_id": "query-a", "task_id": 3}
    coordinator.popped_handles = [selected, discarded]
    backend = RayWorkerManagerBackend(coordinator)
    backend.register_query_owner("query-a", "query-a")

    with pytest.raises(RuntimeError, match="failed to release 1 filtered result handle"):
        backend.wait_query(
            "query-a",
            1.0,
            ({"query_id": "query-a", "task_id": 2},),
        )

    assert selected.release_count == 1
    assert discarded.release_count == 1
    backend.drop_query("query-a")
    assert discarded.release_count == 2


class _BlockedCoordinator(_FakeCoordinator):
    def __init__(self) -> None:
        super().__init__()
        self.submit_started = threading.Event()
        self.exhaustion_started = threading.Event()
        self.pop_started = threading.Event()
        self.allow_submit = threading.Event()
        self.allow_exhaustion = threading.Event()
        self.allow_pop = threading.Event()

    def submit_tasks(self, tasks):
        self.calls.append(("submit_tasks", (tasks,)))
        self.submit_started.set()
        assert self.allow_submit.wait(timeout=5.0)
        return self.submitted_handles

    def task_input_stream_exhausted_for_query(self, query_id, source_node_ids):
        self.calls.append(("task_input_stream_exhausted_for_query", (query_id, source_node_ids)))
        self.exhaustion_started.set()
        assert self.allow_exhaustion.wait(timeout=5.0)
        return self.exhaustion_handles

    def pop_fte_result_handles(self, query_id):
        self.calls.append(("pop_fte_result_handles", (query_id,)))
        self.pop_started.set()
        assert self.allow_pop.wait(timeout=5.0)
        return self.popped_handles

    def fte_drop_query(self, query_id):
        super().fte_drop_query(query_id)
        self.allow_submit.set()
        self.allow_exhaustion.set()
        self.allow_pop.set()


@pytest.mark.parametrize("producer", ["submit", "exhaustion", "wait"])
def test_ray_worker_manager_backend_releases_handles_returned_after_drop(producer):
    coordinator = _BlockedCoordinator()
    backend = RayWorkerManagerBackend(coordinator)
    backend.register_query_owner("query-a", "query-a")
    result: list[RayTaskResultHandleAdapter] = []
    error: list[BaseException] = []

    def produce_handles() -> None:
        try:
            if producer == "submit":
                result.extend(backend.submit_tasks(({"query_id": "query-a", "resource_query_id": "query-a"},)))
            elif producer == "exhaustion":
                result.extend(backend.task_input_stream_exhausted("query-a", ("source-a",)))
            else:
                result.extend(backend.wait_query("query-a", 1.0))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=produce_handles)
    thread.start()
    started = {
        "submit": coordinator.submit_started,
        "exhaustion": coordinator.exhaustion_started,
        "wait": coordinator.pop_started,
    }[producer]
    assert started.wait(timeout=5.0)

    backend.drop_query("query-a")
    if producer == "submit":
        raw_handle = coordinator.submitted_handles[0]
    elif producer == "exhaustion":
        raw_handle = coordinator.exhaustion_handles[0]
    else:
        raw_handle = coordinator.popped_handles[0]
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert error == []
    assert result == []
    assert raw_handle.release_count == 1
    assert not hasattr(backend, "_handles_by_query")
    assert backend._active_operations_by_owner == {}
    assert backend._query_owner_by_query == {}
    assert backend._closed_queries == set()
    assert backend._closed_query_owners == set()


def test_ray_worker_manager_backend_reuses_query_id_after_quiescent_drop():
    coordinator = _FakeCoordinator()
    backend = RayWorkerManagerBackend(coordinator)
    backend.register_query_owner("query-a", "query-a")
    backend.drop_query("query-a")
    backend.register_query_owner("query-a", "query-a")

    handles = backend.submit_tasks(({"query_id": "query-a", "resource_query_id": "query-a"},))

    assert len(handles) == 1
    assert coordinator.calls == [
        ("fte_drop_query", ("query-a",)),
        ("submit_tasks", ([{"query_id": "query-a", "resource_query_id": "query-a"}],)),
    ]
    assert backend._closed_queries == set()
    assert backend._closed_query_owners == set()


def test_ray_worker_manager_backend_completed_drop_cannot_unfence_reused_generation():
    second_drop_started = threading.Event()
    allow_second_drop = threading.Event()

    class PausingSecondDropCoordinator(_FakeCoordinator):
        def __init__(self):
            super().__init__()
            self.drop_calls = 0

        def fte_drop_query(self, query_id):
            super().fte_drop_query(query_id)
            self.drop_calls += 1
            if self.drop_calls == 2:
                second_drop_started.set()
                assert allow_second_drop.wait(timeout=5.0)

    coordinator = PausingSecondDropCoordinator()
    backend = RayWorkerManagerBackend(coordinator)
    backend.register_query_owner("query-a", "query-a")
    first_finish_completed = threading.Event()
    allow_first_finish_to_return = threading.Event()
    original_finish_query_drop = backend._finish_query_drop

    def pause_after_first_finish(owner_query_id: str, drop_token: object) -> None:
        original_finish_query_drop(owner_query_id, drop_token)
        first_finish_completed.set()
        assert allow_first_finish_to_return.wait(timeout=5.0)

    backend._finish_query_drop = pause_after_first_finish  # type: ignore[method-assign]
    errors: list[BaseException] = []

    def drop() -> None:
        try:
            backend.drop_query("query-a")
        except BaseException as exc:
            errors.append(exc)

    first_drop = threading.Thread(target=drop)
    first_drop.start()
    assert first_finish_completed.wait(timeout=5.0)

    backend.register_query_owner("query-a", "query-a")
    backend._finish_query_drop = original_finish_query_drop  # type: ignore[method-assign]
    second_drop = threading.Thread(target=drop)
    second_drop.start()
    assert second_drop_started.wait(timeout=5.0)

    allow_first_finish_to_return.set()
    first_drop.join(timeout=5.0)
    assert not first_drop.is_alive()
    assert set(backend._dropping_query_owners) == {"query-a"}

    allow_second_drop.set()
    second_drop.join(timeout=5.0)
    assert not second_drop.is_alive()
    assert errors == []
    assert backend._dropping_query_owners == {}


def test_ray_worker_manager_backend_rejects_submission_while_drop_is_in_progress():
    class PausingDropCoordinator(_FakeCoordinator):
        def __init__(self):
            super().__init__()
            self.drop_started = threading.Event()
            self.allow_drop = threading.Event()

        def fte_drop_query(self, query_id):
            super().fte_drop_query(query_id)
            self.drop_started.set()
            assert self.allow_drop.wait(timeout=5.0)

    coordinator = PausingDropCoordinator()
    backend = RayWorkerManagerBackend(coordinator)
    backend.register_query_owner("query-a", "query-a")
    drop_errors: list[BaseException] = []

    def drop() -> None:
        try:
            backend.drop_query("query-a")
        except BaseException as exc:
            drop_errors.append(exc)

    first_drop = threading.Thread(target=drop)
    second_drop = threading.Thread(target=drop)
    first_drop.start()
    assert coordinator.drop_started.wait(timeout=5.0)
    second_drop.start()

    assert backend.submit_tasks(({"query_id": "query-a", "resource_query_id": "query-a"},)) == []
    assert first_drop.is_alive()
    assert second_drop.is_alive()

    coordinator.allow_drop.set()
    first_drop.join(timeout=5.0)
    second_drop.join(timeout=5.0)
    assert not first_drop.is_alive()
    assert not second_drop.is_alive()
    assert drop_errors == []
    assert coordinator.calls == [("fte_drop_query", ("query-a",))]


def test_ray_worker_manager_backend_drops_nested_execution_with_resource_owner():
    coordinator = _BlockedCoordinator()
    backend = RayWorkerManagerBackend(coordinator)
    backend.register_query_owner("query-child", "query-root")
    returned: list[RayTaskResultHandleAdapter] = []

    thread = threading.Thread(
        target=lambda: returned.extend(
            backend.submit_tasks(
                (
                    {
                        "query_id": "query-child",
                        "resource_query_id": "query-root",
                    },
                )
            )
        )
    )
    thread.start()
    assert coordinator.submit_started.wait(timeout=5.0)

    backend.drop_query("query-root")
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert returned == []
    assert coordinator.submitted_handles[0].release_count == 1
    drop_calls = [call for call in coordinator.calls if call[0] == "fte_drop_query"]
    assert drop_calls == [
        ("fte_drop_query", ("query-child",)),
        ("fte_drop_query", ("query-root",)),
    ]
    assert backend._query_owner_by_query == {}
    assert backend._closed_queries == set()
    assert backend._closed_query_owners == set()


def test_ray_worker_manager_backend_requires_explicit_generation_registration():
    coordinator = _FakeCoordinator()
    backend = RayWorkerManagerBackend(coordinator)

    assert backend.fte_query_status("query-child")["canceled"] is True
    assert coordinator.calls == []
    backend.register_query_owner("query-child", "query-root")
    handles = backend.submit_tasks(
        (
            {
                "query_id": "query-child",
                "resource_query_id": "query-root",
            },
        )
    )

    assert len(handles) == 1
    assert backend._query_owner_by_query == {"query-child": "query-root"}
    backend.drop_query("query-root")
    assert backend._query_owner_by_query == {}


def test_ray_worker_manager_backend_drop_fans_out_and_retries_owner_group():
    class FailChildOnceCoordinator(_FakeCoordinator):
        def __init__(self):
            super().__init__()
            self.failed = False

        def fte_drop_query(self, query_id):
            super().fte_drop_query(query_id)
            if query_id == "query-child" and not self.failed:
                self.failed = True
                raise RuntimeError("planned child drop failure")

    coordinator = FailChildOnceCoordinator()
    backend = RayWorkerManagerBackend(coordinator)
    backend.register_query_owner("query-child", "query-root")

    with pytest.raises(RuntimeError, match="failed to drop 1 execution query lifecycle"):
        backend.drop_query("query-root")

    backend.drop_query("query-root")

    drop_calls = [call for call in coordinator.calls if call[0] == "fte_drop_query"]
    assert drop_calls == [
        ("fte_drop_query", ("query-child",)),
        ("fte_drop_query", ("query-root",)),
        ("fte_drop_query", ("query-child",)),
        ("fte_drop_query", ("query-root",)),
    ]
    assert backend._query_owner_by_query == {}


def test_ray_worker_manager_backend_retries_failed_late_release_during_shutdown():
    coordinator = _BlockedCoordinator()
    raw_handle = _FailTwiceDoneHandle(done=True, result="submitted")
    coordinator.submitted_handles = [raw_handle]
    backend = RayWorkerManagerBackend(coordinator)
    backend.register_query_owner("query-a", "query-a")
    error = []

    def submit() -> None:
        try:
            backend.submit_tasks(({"query_id": "query-a", "resource_query_id": "query-a"},))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=submit)
    thread.start()
    assert coordinator.submit_started.wait(timeout=5.0)
    with pytest.raises(RuntimeError, match="failed to retry 1 result handle release"):
        backend.drop_query("query-a")
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert len(error) == 1
    assert "failed to release 1 late result handle" in str(error[0])
    assert raw_handle.release_count == 2

    backend.shutdown()

    assert raw_handle.release_count == 3
    assert backend._cleanup_handles_by_owner == {}


def test_ray_worker_manager_backend_shutdown_is_serialized_and_idempotent():
    class PausingShutdownCoordinator(_FakeCoordinator):
        def __init__(self):
            super().__init__()
            self.shutdown_started = threading.Event()
            self.allow_shutdown = threading.Event()

        def shutdown(self):
            super().shutdown()
            self.shutdown_started.set()
            assert self.allow_shutdown.wait(timeout=5.0)

    coordinator = PausingShutdownCoordinator()
    backend = RayWorkerManagerBackend(coordinator)
    errors: list[BaseException] = []

    def shutdown() -> None:
        try:
            backend.shutdown()
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=shutdown)
    second = threading.Thread(target=shutdown)
    first.start()
    assert coordinator.shutdown_started.wait(timeout=5.0)
    second.start()
    assert second.is_alive()

    coordinator.allow_shutdown.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert coordinator.calls == [("shutdown", ())]


def test_ray_worker_manager_backend_preserves_owner_state_until_shutdown_retry():
    class FailShutdownOnceCoordinator(_FakeCoordinator):
        def __init__(self):
            super().__init__()
            self.shutdown_calls = 0

        def shutdown(self):
            super().shutdown()
            self.shutdown_calls += 1
            if self.shutdown_calls == 1:
                raise RuntimeError("planned coordinator shutdown failure")

    coordinator = FailShutdownOnceCoordinator()
    backend = RayWorkerManagerBackend(coordinator)
    backend.register_query_owner("query-a", "query-a")

    with pytest.raises(RuntimeError, match="planned coordinator shutdown failure"):
        backend.shutdown()

    assert backend._query_owner_by_query == {"query-a": "query-a"}
    with pytest.raises(RuntimeError, match="after Ray backend shutdown failed"):
        backend.drop_query("query-a")
    assert not any(call[0] == "fte_drop_query" for call in coordinator.calls)

    backend.shutdown()

    assert coordinator.shutdown_calls == 2
    assert backend._query_owner_by_query == {}


def test_ray_worker_manager_backend_rejects_mixed_query_submit_batch():
    coordinator = _FakeCoordinator()
    backend = RayWorkerManagerBackend(coordinator)

    with pytest.raises(ValueError, match="multiple query_id"):
        backend.submit_tasks(
            (
                {"query_id": "query-a", "resource_query_id": "query-a"},
                {"query_id": "query-b", "resource_query_id": "query-b"},
            )
        )

    assert coordinator.calls == []


@pytest.mark.parametrize("resource_query_id", [None, "", "   "])
def test_ray_worker_manager_backend_requires_explicit_resource_query_owner(resource_query_id):
    coordinator = _FakeCoordinator()
    backend = RayWorkerManagerBackend(coordinator)
    backend.register_query_owner("query-a", "query-a")

    with pytest.raises(ValueError, match="non-empty resource_query_id"):
        backend.submit_tasks(({"query_id": "query-a", "resource_query_id": resource_query_id},))

    assert coordinator.calls == []
