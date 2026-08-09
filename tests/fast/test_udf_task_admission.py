# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import threading
import time
import weakref
from concurrent.futures import Future

import pytest

import vane.execution.ray_control_deadline as ray_control_deadline
import vane.execution.ray_control_submission as ray_control_submission
import vane.execution.udf_task_admission as task_admission
from vane.execution.udf_admission import (
    LocalExecutionSlotPool,
    LocalSlotAdmissionAuthority,
)
from vane.execution.udf_task_admission import TaskAdmissionController

_QUERY_GENERATION_CAPABILITY = "test-query-generation-capability"


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._fn(*args, **kwargs)


class _ObjectRef:
    def __init__(self):
        self._future = Future()

    def future(self):
        return self._future

    def resolve(self, value):
        self._future.set_result(value)


def _resolved_ref(value):
    ref = _ObjectRef()
    ref.resolve(value)
    return ref


def _wait_for_remote_calls(method, count, *, timeout=1.0):
    deadline = time.monotonic() + timeout
    while len(method.calls) < count and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(method.calls) == count


def _wait_for_state(controller, expected, *, timeout=1.0):
    deadline = time.monotonic() + timeout
    state = controller.state()
    while state["state"] != expected and time.monotonic() < deadline:
        time.sleep(0.005)
        state = controller.state()
    assert state["state"] == expected
    return state


def _wait_for_request_refs(driver, count, *, timeout=1.0):
    deadline = time.monotonic() + timeout
    while len(driver.requests) < count and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(driver.requests) == count


class _FailingObjectRef:
    def __init__(self, *, callback=False):
        self._callback = callback

    def future(self):
        if not self._callback:
            raise RuntimeError("planned admission Future factory failure")

        class _FailingCallbackFuture(Future):
            def add_done_callback(self, fn, *, context=None):
                del fn, context
                raise RuntimeError("planned admission callback registration failure")

        return _FailingCallbackFuture()


class _Driver:
    def __init__(self):
        self.requests: list[_ObjectRef] = []
        self.acquire_query_task_lease = _RemoteMethod(self._acquire)
        self.cancel_query_task_lease_request = _RemoteMethod(
            lambda *_args, **_kwargs: _resolved_ref({"cancelled": True})
        )

    def _acquire(self, _request):
        ref = _ObjectRef()
        self.requests.append(ref)
        return ref


def _payload():
    return {
        "execution_backend": "ray_task",
        "query_id": "q",
        "resource_unit_id": "resource:q:udf:node:3",
        "cpus": 1.0,
        "gpus": 0.0,
        "memory_bytes": 256,
        "udf_task_input_max_bytes": 128,
    }


def test_task_admission_requires_explicit_query_driver_handle():
    with pytest.raises(ValueError, match="query driver handle"):
        TaskAdmissionController(
            _payload(),
            driver=None,
            query_generation_capability=_QUERY_GENERATION_CAPABILITY,
        )


def test_task_admission_submission_rejection_does_not_schedule_cancellation(
    monkeypatch,
):
    driver = _Driver()
    submissions = []

    def reject_submission(owner_scope, callback):
        submissions.append((owner_scope, callback))
        raise RuntimeError("control submission capacity exhausted")

    monkeypatch.setattr(task_admission, "submit_ray_control", reject_submission)
    controller = TaskAdmissionController(
        _payload(),
        driver=driver,
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
    )

    with pytest.raises(RuntimeError, match="capacity exhausted"):
        controller.request(64)

    assert len(submissions) == 1
    assert driver.acquire_query_task_lease.calls == []
    assert driver.cancel_query_task_lease_request.calls == []
    assert controller._cancellation is None
    controller.close()
    assert len(submissions) == 1


@pytest.mark.parametrize("callback", [False, True])
def test_task_admission_setup_failure_cancels_the_remote_request(callback):
    driver = _Driver()
    driver.acquire_query_task_lease = _RemoteMethod(
        lambda _request: _FailingObjectRef(callback=callback),
    )
    controller = TaskAdmissionController(
        _payload(),
        driver=driver,
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
    )
    failed = threading.Event()
    controller.register_wakeup(failed.set)

    expected = "callback registration" if callback else "Future factory"
    assert controller.request(64)

    assert failed.wait(timeout=1.0)
    _wait_for_remote_calls(driver.acquire_query_task_lease, 1)
    request = driver.acquire_query_task_lease.calls[0][0][0]
    assert controller.state()["state"] == "failed"
    with pytest.raises(RuntimeError, match=expected):
        controller.request(64)
    _wait_for_remote_calls(driver.cancel_query_task_lease_request, 1)
    assert driver.cancel_query_task_lease_request.calls == [
        ((request,), {}),
    ]


def _grant(request):
    return {
        "granted": True,
        "lease": {
            "lease_id": "lease-1",
            "query_id": request["query_id"],
            "resource_unit_id": request["resource_unit_id"],
            "task_id": request["task_id"],
            "attempt_id": request["attempt_id"],
            "node_id": "node-a",
            "resources": {
                "cpu": 1.0,
                "gpu": 0.0,
                "heap_bytes": 256,
                "object_store_bytes": request["retained_input_bytes"],
            },
            "output_window_bytes": 128,
            "liveness": False,
            "allocation_generation": 1,
        },
        "blocked_reason": "",
        "fatal": False,
        "liveness": False,
    }


def test_task_admission_has_one_unresolved_request_and_publishes_ready_lease():
    driver = _Driver()
    wakeups = []
    controller = TaskAdmissionController(
        _payload(),
        driver=driver,
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
    )
    controller.register_wakeup(lambda: wakeups.append("ready"))

    assert controller.request(64)
    assert not controller.request(64)
    _wait_for_remote_calls(driver.acquire_query_task_lease, 1)
    _wait_for_request_refs(driver, 1)
    assert len(driver.acquire_query_task_lease.calls) == 1
    request = driver.acquire_query_task_lease.calls[0][0][0]
    assert request["query_generation_capability"] == _QUERY_GENERATION_CAPABILITY
    assert request["retained_input_bytes"] == 64
    assert request["resources"]["object_store_bytes"] == 128
    assert controller.state() == {
        "state": "requested",
        "available": False,
        "retained_input_bytes": 64,
    }

    driver.requests[0].resolve(_grant(request))

    _wait_for_state(controller, "ready")
    assert wakeups == ["ready"]
    assert controller.state() == {
        "state": "ready",
        "available": True,
        "retained_input_bytes": 64,
    }
    admission = controller.take(64)
    assert admission.request_id == request["request_id"]
    assert admission.lease["lease_id"] == "lease-1"
    admission.handoff()
    assert controller.state() == {
        "state": "idle",
        "available": False,
        "retained_input_bytes": 0,
    }


def test_task_admission_does_not_consume_a_lease_for_different_input_bytes():
    driver = _Driver()
    controller = TaskAdmissionController(
        _payload(),
        driver=driver,
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
    )
    assert controller.request(64)
    _wait_for_remote_calls(driver.acquire_query_task_lease, 1)
    _wait_for_request_refs(driver, 1)
    request = driver.acquire_query_task_lease.calls[0][0][0]
    driver.requests[0].resolve(_grant(request))
    _wait_for_state(controller, "ready")

    with pytest.raises(RuntimeError, match="retained input bytes"):
        controller.take(32)

    assert controller.state()["state"] == "ready"
    admission = controller.take(64)
    assert admission.lease["lease_id"] == "lease-1"
    admission.handoff()


def test_task_admission_preserves_async_denial_reason():
    driver = _Driver()
    controller = TaskAdmissionController(
        _payload(),
        driver=driver,
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
    )
    assert controller.request(16)

    _wait_for_remote_calls(driver.acquire_query_task_lease, 1)
    _wait_for_request_refs(driver, 1)
    driver.requests[0].resolve(
        {
            "granted": False,
            "blocked_reason": "query_not_registered",
            "fatal": True,
        }
    )

    state = _wait_for_state(controller, "failed")
    assert state["state"] == "failed"
    assert "query_not_registered" in state["error"]
    with pytest.raises(RuntimeError, match="query_not_registered"):
        controller.request(16)


def test_task_admission_close_cancels_pending_and_ready_leases():
    pending_driver = _Driver()
    pending = TaskAdmissionController(
        _payload(),
        driver=pending_driver,
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
    )
    assert pending.request(32)
    _wait_for_remote_calls(pending_driver.acquire_query_task_lease, 1)
    _wait_for_request_refs(pending_driver, 1)
    pending_request = pending_driver.acquire_query_task_lease.calls[0][0][0]

    pending.close()

    assert pending.state()["state"] == "closed"
    _wait_for_remote_calls(pending_driver.cancel_query_task_lease_request, 1)
    assert pending_driver.cancel_query_task_lease_request.calls == [((pending_request,), {})]

    pending_driver.requests[0].resolve(_grant(pending_request))
    assert pending_driver.cancel_query_task_lease_request.calls == [
        ((pending_request,), {}),
    ]

    ready_driver = _Driver()
    ready = TaskAdmissionController(
        _payload(),
        driver=ready_driver,
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
    )
    assert ready.request(48)
    _wait_for_remote_calls(ready_driver.acquire_query_task_lease, 1)
    _wait_for_request_refs(ready_driver, 1)
    ready_request = ready_driver.acquire_query_task_lease.calls[0][0][0]
    ready_driver.requests[0].resolve(_grant(ready_request))
    _wait_for_state(ready, "ready")

    ready.close()

    assert ready.state()["state"] == "closed"
    _wait_for_remote_calls(ready_driver.cancel_query_task_lease_request, 1)
    assert ready_driver.cancel_query_task_lease_request.calls == [((ready_request,), {})]


def test_pending_admission_future_does_not_retain_closed_controller():
    driver = _Driver()
    controller = TaskAdmissionController(
        _payload(),
        driver=driver,
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
    )
    assert controller.request(32)
    _wait_for_remote_calls(driver.acquire_query_task_lease, 1)
    _wait_for_request_refs(driver, 1)
    request = driver.acquire_query_task_lease.calls[0][0][0]
    controller_ref = weakref.ref(controller)

    controller.close()
    del controller
    gc.collect()

    assert controller_ref() is None
    driver.requests[0].resolve(_grant(request))
    _wait_for_remote_calls(driver.cancel_query_task_lease_request, 1)
    assert driver.cancel_query_task_lease_request.calls == [((request,), {})]


def test_taken_task_admission_abandons_if_submission_never_takes_ownership():
    driver = _Driver()
    controller = TaskAdmissionController(
        _payload(),
        driver=driver,
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
    )
    assert controller.request(24)
    _wait_for_remote_calls(driver.acquire_query_task_lease, 1)
    _wait_for_request_refs(driver, 1)
    request = driver.acquire_query_task_lease.calls[0][0][0]
    driver.requests[0].resolve(_grant(request))
    _wait_for_state(controller, "ready")

    admission = controller.take(24)
    admission.release()
    admission.release()

    _wait_for_remote_calls(driver.cancel_query_task_lease_request, 1)
    assert driver.cancel_query_task_lease_request.calls == [((request,), {})]


def test_blocked_task_admission_submission_does_not_stall_other_controller():
    driver = _Driver()
    blocked_submission_entered = threading.Event()
    release_blocked_submission = threading.Event()
    original_acquire = driver._acquire

    def selectively_blocking_acquire(request):
        if request["retained_input_bytes"] == 31:
            blocked_submission_entered.set()
            if not release_blocked_submission.wait(timeout=5.0):
                raise RuntimeError("timed out waiting to release blocked task admission")
            return _resolved_ref(_grant(request))
        return original_acquire(request)

    driver.acquire_query_task_lease = _RemoteMethod(selectively_blocking_acquire)
    blocked = TaskAdmissionController(
        _payload(),
        driver=driver,
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
    )
    healthy = TaskAdmissionController(
        _payload(),
        driver=driver,
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
    )
    healthy_ready = threading.Event()
    healthy.register_wakeup(healthy_ready.set)

    try:
        started_at = time.monotonic()
        assert blocked.request(31)
        request_elapsed = time.monotonic() - started_at

        assert request_elapsed < 1.0
        assert blocked_submission_entered.wait(timeout=1.0)

        assert healthy.request(47)
        _wait_for_remote_calls(driver.acquire_query_task_lease, 2)
        _wait_for_request_refs(driver, 1)
        healthy_request = next(
            call[0][0] for call in driver.acquire_query_task_lease.calls if call[0][0]["retained_input_bytes"] == 47
        )
        driver.requests[0].resolve(_grant(healthy_request))

        assert healthy_ready.wait(timeout=1.0)
        assert healthy.state()["state"] == "ready"
    finally:
        release_blocked_submission.set()
        blocked.close()
        healthy.close()

    _wait_for_remote_calls(driver.cancel_query_task_lease_request, 2)


def test_blocked_stream_control_does_not_stall_next_admission_from_same_controller():
    driver = _Driver()
    blocked_cancel_entered = threading.Event()
    release_blocked_cancel = threading.Event()
    first_request_id = ""

    def selectively_blocking_cancel(request):
        if request["request_id"] == first_request_id:
            blocked_cancel_entered.set()
            if not release_blocked_cancel.wait(timeout=5.0):
                raise RuntimeError("timed out waiting to release blocked stream control")
        return _resolved_ref({"cancelled": True})

    driver.cancel_query_task_lease_request = _RemoteMethod(
        selectively_blocking_cancel,
    )
    controller = TaskAdmissionController(
        _payload(),
        driver=driver,
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
    )

    try:
        assert controller.request(31)
        _wait_for_remote_calls(driver.acquire_query_task_lease, 1)
        _wait_for_request_refs(driver, 1)
        first_request = driver.acquire_query_task_lease.calls[0][0][0]
        first_request_id = first_request["request_id"]
        driver.requests[0].resolve(_grant(first_request))
        _wait_for_state(controller, "ready")

        first_admission = controller.take(31)
        assert first_admission.submission_scope == f"udf-stream:{first_request_id}"
        first_admission.release()
        assert blocked_cancel_entered.wait(timeout=1.0)

        assert controller.request(47)
        _wait_for_remote_calls(driver.acquire_query_task_lease, 2)
        _wait_for_request_refs(driver, 2)
        second_request = driver.acquire_query_task_lease.calls[1][0][0]
        driver.requests[1].resolve(_grant(second_request))
        _wait_for_state(controller, "ready")

        second_admission = controller.take(47)
        assert second_admission.submission_scope == f"udf-stream:{second_request['request_id']}"
        assert second_admission.submission_scope != first_admission.submission_scope
        second_admission.release()
        _wait_for_remote_calls(driver.cancel_query_task_lease_request, 2)
        assert release_blocked_cancel.is_set() is False
    finally:
        release_blocked_cancel.set()
        controller.close()


def test_blocked_task_admission_submission_does_not_retain_closed_controller():
    driver = _Driver()
    submission_entered = threading.Event()
    release_submission = threading.Event()

    def blocked_acquire(request):
        submission_entered.set()
        if not release_submission.wait(timeout=5.0):
            raise RuntimeError("timed out waiting to release blocked task admission")
        return _resolved_ref(_grant(request))

    driver.acquire_query_task_lease = _RemoteMethod(blocked_acquire)
    controller = TaskAdmissionController(
        _payload(),
        driver=driver,
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
    )
    assert controller.request(29)
    assert submission_entered.wait(timeout=1.0)
    controller_ref = weakref.ref(controller)

    controller.close()
    del controller
    gc.collect()

    try:
        assert controller_ref() is None
    finally:
        release_submission.set()
    _wait_for_remote_calls(driver.cancel_query_task_lease_request, 1)


def test_task_admission_cancellation_retries_until_driver_acknowledges():
    driver = _Driver()
    acknowledged = threading.Event()
    attempts = 0

    def cancel(_request_id):
        nonlocal attempts
        attempts += 1
        response = _ObjectRef()
        if attempts == 1:
            response._future.set_exception(RuntimeError("planned transient cancellation failure"))
        else:
            response.resolve({"cancelled": True})
            acknowledged.set()
        return response

    driver.cancel_query_task_lease_request = _RemoteMethod(cancel)
    controller = TaskAdmissionController(
        _payload(),
        driver=driver,
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
    )
    assert controller.request(24)

    controller.close()

    assert acknowledged.wait(timeout=1.0)
    assert len(driver.cancel_query_task_lease_request.calls) == 2


def test_task_admission_cancellation_retries_async_submission_rejection(
    monkeypatch,
):
    driver = _Driver()
    submissions = []

    def submit_control(_owner_scope, callback):
        future = Future()
        submissions.append((future, callback))
        if len(submissions) > 1:
            callback()
            future.set_result(None)
        return future

    monkeypatch.setattr(task_admission, "submit_ray_control", submit_control)
    cancellation = task_admission._TaskAdmissionCancellation(
        driver=driver,
        request={
            "request_id": "async-submission-rejection",
            "query_id": "query:async-submission-rejection",
            "resources": {},
        },
        submission_scope="test-admission:async-submission-rejection",
    )

    cancellation.start()
    assert cancellation._submitting
    submissions[0][0].set_exception(RuntimeError("planned async queue rejection"))

    deadline = time.monotonic() + 1.0
    while not cancellation._done and time.monotonic() < deadline:
        time.sleep(0.005)
    assert cancellation._done
    assert len(submissions) == 2
    assert len(driver.cancel_query_task_lease_request.calls) == 1


def test_task_admission_cancellation_retries_a_hung_response(monkeypatch):
    monkeypatch.setattr(
        task_admission,
        "_TASK_ADMISSION_CLEANUP_RESPONSE_TIMEOUT_S",
        0.01,
    )
    driver = _Driver()
    acknowledged = threading.Event()
    first_response = _ObjectRef()

    def cancel(_request):
        if not driver.cancel_query_task_lease_request.calls:
            raise AssertionError("remote call must be recorded before dispatch")
        if len(driver.cancel_query_task_lease_request.calls) == 1:
            return first_response
        acknowledged.set()
        return _resolved_ref({"cancelled": True})

    driver.cancel_query_task_lease_request = _RemoteMethod(cancel)
    controller = TaskAdmissionController(
        _payload(),
        driver=driver,
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
    )
    assert controller.request(24)

    controller.close()

    assert acknowledged.wait(timeout=1.0)
    assert len(driver.cancel_query_task_lease_request.calls) == 2
    # The idempotent retry supersedes the observer without invoking arbitrary
    # callbacks through Future.cancel().
    assert not first_response._future.cancelled()


def test_task_admission_cancellation_expands_slow_response_deadline(monkeypatch):
    monkeypatch.setattr(
        task_admission,
        "_TASK_ADMISSION_CLEANUP_RESPONSE_TIMEOUT_S",
        0.01,
    )
    monkeypatch.setattr(
        task_admission,
        "_TASK_ADMISSION_CLEANUP_RESPONSE_TIMEOUT_MAX_S",
        0.04,
    )
    driver = _Driver()
    response_timers = []
    first_response = None

    def cancel(_request):
        nonlocal first_response
        response = _ObjectRef()
        if len(driver.cancel_query_task_lease_request.calls) == 1:
            first_response = response
            return response

        def acknowledge():
            response.resolve({"cancelled": True})

        timer = threading.Timer(0.015, acknowledge)
        timer.daemon = True
        timer.start()
        response_timers.append(timer)
        return response

    driver.cancel_query_task_lease_request = _RemoteMethod(cancel)
    cancellation = task_admission._TaskAdmissionCancellation(
        driver=driver,
        request={
            "request_id": "slow-cancel",
            "query_id": "query:slow-cancel",
            "resources": {},
        },
        submission_scope="test-admission:slow-cancel",
    )

    cancellation.start()

    deadline = time.monotonic() + 1.0
    while not cancellation._done and time.monotonic() < deadline:
        time.sleep(0.005)
    assert cancellation._done
    assert len(driver.cancel_query_task_lease_request.calls) == 2
    assert first_response is not None
    assert not first_response._future.cancelled()
    for timer in response_timers:
        timer.join(timeout=1.0)
        assert timer.is_alive() is False


def test_task_admission_cancellations_share_one_deadline_thread(monkeypatch):
    monkeypatch.setattr(
        task_admission,
        "_TASK_ADMISSION_CLEANUP_RESPONSE_TIMEOUT_S",
        30.0,
    )
    driver = _Driver()
    responses = []

    def cancel(_request):
        response = _ObjectRef()
        responses.append(response)
        return response

    driver.cancel_query_task_lease_request = _RemoteMethod(cancel)
    cancellations = [
        task_admission._TaskAdmissionCancellation(
            driver=driver,
            request={
                "request_id": f"shared-cancel:{index}",
                "query_id": "query:shared-cancel",
                "resources": {},
            },
            submission_scope="test-admission:shared-cancel",
        )
        for index in range(256)
    ]

    for cancellation in cancellations:
        cancellation.start()

    _wait_for_remote_calls(
        driver.cancel_query_task_lease_request,
        len(cancellations),
    )
    response_deadline = time.monotonic() + 1.0
    while len(responses) < len(cancellations) and time.monotonic() < response_deadline:
        time.sleep(0.005)
    assert len(responses) == len(cancellations)
    scheduler_threads = [thread for thread in threading.enumerate() if thread.name == "vane-ray-control-deadlines"]
    assert len(scheduler_threads) == 1

    for response in responses:
        response.resolve({"cancelled": True})
    deadline = time.monotonic() + 1.0
    while not all(cancellation._done for cancellation in cancellations) and time.monotonic() < deadline:
        time.sleep(0.005)
    assert all(cancellation._done for cancellation in cancellations)


def test_task_admission_deadline_scheduler_start_failure_is_retryable(
    monkeypatch,
):
    scheduler = ray_control_deadline.RayControlDeadlineScheduler()
    original_start = threading.Thread.start
    starts = 0

    def fail_first_start(thread):
        nonlocal starts
        if thread.name == "vane-ray-control-deadlines":
            starts += 1
            if starts == 1:
                raise RuntimeError("deadline thread unavailable")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_first_start)
    with pytest.raises(RuntimeError, match="deadline thread unavailable"):
        scheduler.ensure_started()

    assert scheduler._thread is None

    scheduler.ensure_started()
    assert scheduler._thread is not None
    assert scheduler._thread.is_alive()


def test_ray_control_submissions_isolate_blocked_owner_and_retire_idle_workers():
    executor = ray_control_submission._RayControlSubmissionExecutor(
        idle_timeout_s=0.01,
    )
    blocked_owner = "owner:blocked"
    healthy_owner = "owner:healthy"
    release_submissions = threading.Event()
    blocked_submission_entered = threading.Event()
    entered = 0

    def blocked_submission():
        nonlocal entered
        entered += 1
        blocked_submission_entered.set()
        if not release_submissions.wait(timeout=5.0):
            raise RuntimeError("timed out waiting to release blocked driver")
        return "released"

    blocked_futures = [
        executor.submit(
            blocked_owner,
            blocked_submission,
        )
        for _ in range(65)
    ]

    try:
        assert blocked_submission_entered.wait(timeout=1.0)
        assert entered == 1
        healthy = executor.submit(healthy_owner, lambda: "healthy")
        assert healthy.result(timeout=1.0) == "healthy"
        assert entered == 1
    finally:
        release_submissions.set()

    assert [future.result(timeout=1.0) for future in blocked_futures] == ["released"] * len(blocked_futures)
    completion_deadline = time.monotonic() + 1.0
    while executor._workers and time.monotonic() < completion_deadline:
        time.sleep(0.005)
    assert executor._workers == {}


def test_ray_control_submissions_require_one_recovery_capacity_slot():
    with pytest.raises(
        ValueError,
        match="pending capacity must exceed max_stalled_workers",
    ):
        ray_control_submission._RayControlSubmissionExecutor(
            max_workers=2,
            max_stalled_workers=2,
            max_pending_submissions=2,
        )


def test_ray_control_submissions_bound_stalled_workers_and_queued_owners():
    executor = ray_control_submission._RayControlSubmissionExecutor(
        idle_timeout_s=0.01,
        max_workers=3,
        max_pending_submissions=5,
        max_pending_per_owner=2,
    )
    release_submissions = threading.Event()
    all_workers_blocked = threading.Event()
    entered_lock = threading.Lock()
    entered = 0

    def blocked_submission():
        nonlocal entered
        with entered_lock:
            entered += 1
            if entered == 3:
                all_workers_blocked.set()
        if not release_submissions.wait(timeout=5.0):
            raise RuntimeError("timed out waiting to release bounded control worker")
        return "released"

    futures = [executor.submit(f"owner:blocked:{index}", blocked_submission) for index in range(5)]
    assert all_workers_blocked.wait(timeout=1.0)
    with pytest.raises(RuntimeError, match="queue is full"):
        executor.submit("owner:overflow", blocked_submission)

    with executor._condition:
        assert len(executor._workers) == 3
        assert executor._pending_submissions == 5
        assert len(executor._owners) == 5

    release_submissions.set()
    assert [future.result(timeout=1.0) for future in futures] == ["released"] * 5
    completion_deadline = time.monotonic() + 1.0
    while executor._workers and time.monotonic() < completion_deadline:
        time.sleep(0.005)
    assert executor._workers == {}
    assert executor._owners == {}
    assert executor._pending_submissions == 0


def test_stalled_control_owners_do_not_consume_the_schedulable_pool():
    executor = ray_control_submission._RayControlSubmissionExecutor(
        idle_timeout_s=0.01,
        stall_timeout_s=0.02,
        max_workers=32,
        max_stalled_workers=32,
        max_pending_submissions=64,
        max_pending_per_owner=1,
    )
    release_submissions = threading.Event()
    all_workers_blocked = threading.Event()
    entered_lock = threading.Lock()
    entered = 0

    def blocked_submission():
        nonlocal entered
        with entered_lock:
            entered += 1
            if entered == 32:
                all_workers_blocked.set()
        if not release_submissions.wait(timeout=5.0):
            raise RuntimeError("timed out waiting to release stalled control owner")
        return "released"

    blocked = [executor.submit(f"owner:stalled:{index}", blocked_submission) for index in range(32)]
    assert all_workers_blocked.wait(timeout=1.0)
    stall_deadline = time.monotonic() + 1.0
    while time.monotonic() < stall_deadline:
        with executor._condition:
            stalled_workers = sum(worker.stalled for worker in executor._workers.values())
        if stalled_workers == 32:
            break
        time.sleep(0.005)

    healthy = executor.submit("owner:healthy-after-stalls", lambda: "healthy")
    assert healthy.result(timeout=1.0) == "healthy"
    with executor._condition:
        assert sum(worker.stalled for worker in executor._workers.values()) == 32
        assert len(executor._workers) <= 64
        assert executor._pending_submissions == 32

    release_submissions.set()
    assert [future.result(timeout=1.0) for future in blocked] == ["released"] * 32
    completion_deadline = time.monotonic() + 1.0
    while executor._workers and time.monotonic() < completion_deadline:
        time.sleep(0.005)
    assert executor._workers == {}
    assert executor._owners == {}
    assert executor._pending_submissions == 0


def test_blocked_future_callbacks_do_not_consume_the_schedulable_pool():
    executor = ray_control_submission._RayControlSubmissionExecutor(
        idle_timeout_s=0.01,
        stall_timeout_s=0.2,
        max_workers=2,
        max_stalled_workers=2,
        max_pending_submissions=8,
        max_pending_per_owner=1,
    )
    callbacks_may_return = threading.Event()
    release_completion_callbacks = threading.Event()
    submissions_entered = threading.Event()
    completion_callbacks_entered = threading.Event()
    counter_lock = threading.Lock()
    entered_submissions = 0
    entered_callbacks = 0

    def submission():
        nonlocal entered_submissions
        with counter_lock:
            entered_submissions += 1
            if entered_submissions == 2:
                submissions_entered.set()
        assert callbacks_may_return.wait(timeout=5.0)
        return "completed"

    def block_completion(_future):
        nonlocal entered_callbacks
        with counter_lock:
            entered_callbacks += 1
            if entered_callbacks == 2:
                completion_callbacks_entered.set()
        assert release_completion_callbacks.wait(timeout=5.0)

    blocked = [executor.submit(f"owner:blocked-completion:{index}", submission) for index in range(2)]
    assert submissions_entered.wait(timeout=1.0)
    for future in blocked:
        future.add_done_callback(block_completion)
    callbacks_may_return.set()
    assert completion_callbacks_entered.wait(timeout=1.0)

    healthy = executor.submit("owner:healthy-after-completion-stalls", lambda: "healthy")
    assert healthy.result(timeout=1.0) == "healthy"
    with executor._condition:
        assert sum(worker.stalled for worker in executor._workers.values()) >= 1
        assert len(executor._workers) <= 4

    release_completion_callbacks.set()
    assert [future.result(timeout=1.0) for future in blocked] == ["completed"] * 2
    completion_deadline = time.monotonic() + 1.0
    while executor._workers and time.monotonic() < completion_deadline:
        time.sleep(0.005)
    assert executor._workers == {}
    assert executor._owners == {}
    assert executor._pending_submissions == 0


def test_control_deadline_uses_lock_free_completion_timestamps():
    executor = ray_control_submission._RayControlSubmissionExecutor(
        idle_timeout_s=0.5,
        stall_timeout_s=0.1,
        max_workers=1,
        max_stalled_workers=1,
        max_pending_submissions=3,
        max_pending_per_owner=1,
    )
    submission_entered = threading.Event()
    callback_may_return = threading.Event()

    def submission():
        submission_entered.set()
        assert callback_may_return.wait(timeout=5.0)
        return "first"

    first = executor.submit("owner:lock-contention:first", submission)
    assert submission_entered.wait(timeout=1.0)
    healthy = executor.submit("owner:lock-contention:healthy", lambda: "healthy")
    unrelated_deadline_ran = threading.Event()

    with executor._condition:
        original_worker = next(iter(executor._workers.values()))
        callback_may_return.set()
        timestamp_deadline = time.monotonic() + 1.0
        while original_worker.completion_finished_at is None and time.monotonic() < timestamp_deadline:
            time.sleep(0.001)
        assert original_worker.callback_finished_at is not None
        assert original_worker.completion_finished_at is not None
        # Hold the bookkeeping lock beyond both phase deadlines. The worker's
        # already-published timestamps must prevent a false stall classification
        # without delaying an unrelated deadline behind this lock.
        time.sleep(0.12)
        unrelated_call = ray_control_submission.create_ray_control_deadline(
            "owner:unrelated-lock-contention",
            unrelated_deadline_ran.set,
        )
        ray_control_deadline.RAY_CONTROL_DEADLINE_SCHEDULER.schedule(
            unrelated_call,
            0,
        )
        assert unrelated_deadline_ran.wait(timeout=0.2)

    assert first.result(timeout=1.0) == "first"
    assert healthy.result(timeout=1.0) == "healthy"
    assert original_worker.stalled is False


def test_stalled_control_owner_retains_fifo_while_healthy_owner_progresses():
    executor = ray_control_submission._RayControlSubmissionExecutor(
        idle_timeout_s=0.01,
        stall_timeout_s=0.02,
        max_workers=1,
        max_stalled_workers=1,
        max_pending_submissions=3,
        max_pending_per_owner=2,
    )
    release_submission = threading.Event()
    blocked_submission_entered = threading.Event()
    same_owner_followup_entered = threading.Event()

    def blocked_submission():
        blocked_submission_entered.set()
        if not release_submission.wait(timeout=5.0):
            raise RuntimeError("timed out waiting to release FIFO control owner")
        return "first"

    first = executor.submit("owner:stalled-fifo", blocked_submission)
    assert blocked_submission_entered.wait(timeout=1.0)
    second = executor.submit(
        "owner:stalled-fifo",
        lambda: same_owner_followup_entered.set() or "second",
    )
    healthy = executor.submit("owner:healthy-beside-stall", lambda: "healthy")

    assert healthy.result(timeout=1.0) == "healthy"
    assert same_owner_followup_entered.is_set() is False
    assert second.done() is False

    release_submission.set()
    assert first.result(timeout=1.0) == "first"
    assert second.result(timeout=1.0) == "second"
    assert same_owner_followup_entered.is_set()


def test_control_submissions_fail_fast_at_stalled_worker_bound_and_recover():
    executor = ray_control_submission._RayControlSubmissionExecutor(
        idle_timeout_s=0.01,
        stall_timeout_s=0.02,
        max_workers=2,
        max_stalled_workers=1,
        max_pending_submissions=4,
        max_pending_per_owner=2,
    )
    release_submissions = threading.Event()
    all_workers_blocked = threading.Event()
    entered_lock = threading.Lock()
    entered = 0

    def blocked_submission():
        nonlocal entered
        with entered_lock:
            entered += 1
            if entered == 2:
                all_workers_blocked.set()
        if not release_submissions.wait(timeout=5.0):
            raise RuntimeError("timed out waiting to release bounded stalled worker")
        return "released"

    blocked = [executor.submit(f"owner:bounded-stall:{index}", blocked_submission) for index in range(2)]
    assert all_workers_blocked.wait(timeout=1.0)
    queued = executor.submit("owner:bounded-stall:0", lambda: "unexpected")

    with pytest.raises(RuntimeError, match="stalled callback capacity is exhausted"):
        queued.result(timeout=1.0)
    with pytest.raises(RuntimeError, match="stalled callback capacity is exhausted"):
        executor.submit("owner:rejected-while-exhausted", lambda: "unexpected")
    with executor._condition:
        assert len(executor._workers) == 2
        assert executor._pending_submissions == 2

    release_submissions.set()
    assert [future.result(timeout=1.0) for future in blocked] == ["released"] * 2
    recovery = executor.submit("owner:recovered", lambda: "recovered")
    assert recovery.result(timeout=1.0) == "recovered"


def test_stalled_capacity_failures_do_not_run_future_callbacks_on_the_clock():
    executor = ray_control_submission._RayControlSubmissionExecutor(
        idle_timeout_s=0.01,
        stall_timeout_s=0.02,
        max_workers=2,
        max_stalled_workers=1,
        max_pending_submissions=6,
        max_pending_per_owner=2,
    )
    release_submissions = threading.Event()
    all_workers_blocked = threading.Event()
    failure_callback_entered = threading.Event()
    release_failure_callback = threading.Event()
    entered_lock = threading.Lock()
    entered = 0

    def blocked_submission():
        nonlocal entered
        with entered_lock:
            entered += 1
            if entered == 2:
                all_workers_blocked.set()
        assert release_submissions.wait(timeout=5.0)
        return "released"

    def block_failure_callback(_future):
        assert threading.current_thread().name.startswith("vane-ray-control-completion")
        failure_callback_entered.set()
        assert release_failure_callback.wait(timeout=5.0)

    running = [
        executor.submit("owner:isolated-failure:0", blocked_submission),
        executor.submit("owner:isolated-failure:1", blocked_submission),
    ]
    assert all_workers_blocked.wait(timeout=1.0)
    first_rejected = executor.submit("owner:isolated-failure:0", lambda: "unexpected")
    second_rejected = executor.submit("owner:isolated-failure:1", lambda: "unexpected")
    first_rejected.add_done_callback(block_failure_callback)

    try:
        assert failure_callback_entered.wait(timeout=1.0)
        # A callback blocked by the first set_exception must not prevent the
        # second rejected Future from becoming terminal.
        with pytest.raises(RuntimeError, match="stalled callback capacity is exhausted"):
            second_rejected.result(timeout=1.0)
    finally:
        release_failure_callback.set()
        release_submissions.set()

    with pytest.raises(RuntimeError, match="stalled callback capacity is exhausted"):
        first_rejected.result(timeout=1.0)
    assert [future.result(timeout=1.0) for future in running] == ["released"] * 2


def test_exhausted_completion_callbacks_do_not_block_control_state(monkeypatch):
    callback_executor = ray_control_submission._RayControlSubmissionExecutor(
        idle_timeout_s=0.01,
        stall_timeout_s=0.02,
        max_workers=1,
        max_stalled_workers=1,
        max_pending_submissions=3,
        max_pending_per_owner=1,
        thread_name_prefix="test-ray-control-completion",
    )
    monkeypatch.setattr(
        ray_control_submission,
        "_RAY_CONTROL_COMPLETION_CALLBACK_EXECUTOR",
        callback_executor,
    )
    all_callback_workers_blocked = threading.Event()
    release_callbacks = threading.Event()
    state_callback_ran = threading.Event()
    dropped_abandonment_ran = threading.Event()
    entered_lock = threading.Lock()
    entered = 0

    def blocked_callback():
        nonlocal entered
        with entered_lock:
            entered += 1
            if entered == 2:
                all_callback_workers_blocked.set()
        assert release_callbacks.wait(timeout=5.0)

    for index in range(2):
        ray_control_submission.dispatch_ray_control_abandonment(
            f"owner:blocked-abandonment:{index}",
            blocked_callback,
        )
    assert all_callback_workers_blocked.wait(timeout=1.0)
    stalled_deadline = time.monotonic() + 1.0
    while time.monotonic() < stalled_deadline:
        with callback_executor._condition:
            if callback_executor._stalled_worker_count_locked() > 1:
                break
        time.sleep(0.005)
    else:
        raise AssertionError("completion callback capacity was not exhausted")

    class _DroppedAbandonment:
        def cancel(self):
            dropped_abandonment_ran.set()

    dropped_abandonment = _DroppedAbandonment()
    dropped_ref = weakref.ref(dropped_abandonment)
    ray_control_submission.dispatch_ray_control_abandonment(
        "owner:dropped-abandonment",
        dropped_abandonment.cancel,
    )
    del dropped_abandonment
    gc.collect()
    assert dropped_ref() is None

    state_call = ray_control_submission.create_ray_control_deadline(
        "owner:state-beside-exhausted-completions",
        state_callback_ran.set,
    )
    ray_control_deadline.RAY_CONTROL_DEADLINE_SCHEDULER.schedule(state_call, 0)
    try:
        assert state_callback_ran.wait(timeout=1.0)
        with callback_executor._condition:
            assert any(worker.stalled for worker in callback_executor._workers.values())
            assert len(callback_executor._workers) <= 2
    finally:
        release_callbacks.set()
    assert dropped_abandonment_ran.is_set() is False
    completion_deadline = time.monotonic() + 1.0
    while callback_executor._workers and time.monotonic() < completion_deadline:
        time.sleep(0.005)
    assert callback_executor._workers == {}
    assert callback_executor._owners == {}


def test_ray_control_submission_deadline_start_failure_does_not_admit_work(monkeypatch):
    scheduler = ray_control_deadline.RayControlDeadlineScheduler()
    executor = ray_control_submission._RayControlSubmissionExecutor(
        idle_timeout_s=0.01,
        deadline_scheduler=scheduler,
    )
    original_start = threading.Thread.start
    starts = 0

    def fail_first_deadline_start(thread):
        nonlocal starts
        if thread.name == "vane-ray-control-deadlines":
            starts += 1
            if starts == 1:
                raise RuntimeError("control deadline clock unavailable")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_first_deadline_start)
    with pytest.raises(RuntimeError, match="control deadline clock unavailable"):
        executor.submit("owner:deadline-start-failure", lambda: "unexpected")

    assert scheduler._thread is None
    assert executor._workers == {}
    assert executor._owners == {}
    assert executor._pending_submissions == 0

    retry = executor.submit("owner:deadline-start-retry", lambda: "recovered")
    assert retry.result(timeout=1.0) == "recovered"


def test_ray_control_submissions_bound_one_owner_queue():
    executor = ray_control_submission._RayControlSubmissionExecutor(
        idle_timeout_s=0.01,
        max_workers=1,
        max_pending_submissions=3,
        max_pending_per_owner=2,
    )
    release_submission = threading.Event()
    submission_entered = threading.Event()

    def blocked_submission():
        submission_entered.set()
        if not release_submission.wait(timeout=5.0):
            raise RuntimeError("timed out waiting to release owner queue")
        return "released"

    first = executor.submit("owner:bounded", blocked_submission)
    assert submission_entered.wait(timeout=1.0)
    second = executor.submit("owner:bounded", lambda: "second")
    with pytest.raises(RuntimeError, match="owner queue is full"):
        executor.submit("owner:bounded", lambda: "overflow")

    release_submission.set()
    assert first.result(timeout=1.0) == "released"
    assert second.result(timeout=1.0) == "second"


def test_cancelled_queued_control_submissions_release_capacity_while_worker_is_blocked():
    executor = ray_control_submission._RayControlSubmissionExecutor(
        idle_timeout_s=0.01,
        max_workers=1,
        max_pending_submissions=3,
        max_pending_per_owner=1,
    )
    release_submission = threading.Event()
    submission_entered = threading.Event()

    def blocked_submission():
        submission_entered.set()
        if not release_submission.wait(timeout=5.0):
            raise RuntimeError("timed out waiting to release control worker")
        return "released"

    running = executor.submit("owner:blocked", blocked_submission)
    assert submission_entered.wait(timeout=1.0)
    cancelled = [
        executor.submit("owner:cancelled:1", lambda: "unexpected"),
        executor.submit("owner:cancelled:2", lambda: "unexpected"),
    ]

    assert all(future.cancel() for future in cancelled)
    with executor._condition:
        assert executor._pending_submissions == 1
        assert set(executor._owners) == {"owner:blocked"}

    replacement = executor.submit("owner:replacement", lambda: "replacement")
    release_submission.set()
    assert running.result(timeout=1.0) == "released"
    assert replacement.result(timeout=1.0) == "replacement"


def test_ray_control_submission_does_not_retain_worker_when_thread_start_fails(
    monkeypatch,
):
    executor = ray_control_submission._RayControlSubmissionExecutor()

    def fail_start(_worker):
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr(
        ray_control_submission._RayControlSubmissionWorker,
        "start",
        fail_start,
    )
    with pytest.raises(RuntimeError, match="thread unavailable"):
        executor.submit("owner:start-failure", lambda: None)

    assert executor._workers == {}
    assert executor._owners == {}
    assert executor._pending_submissions == 0


def test_local_slot_admission_owns_concrete_slots_and_wakes_one_waiter():
    authority = LocalSlotAdmissionAuthority(
        max_slots=2,
        execution_slot_prefix="subprocess",
    )
    wakeups = []
    authority.register_wakeup(lambda: wakeups.append("ready"))

    assert authority.request(11)
    first = authority.take(11)
    assert first.execution_slot_id == "subprocess:0"

    assert authority.request(22)
    second = authority.take(22)
    assert second.execution_slot_id == "subprocess:1"

    assert authority.request(33)
    assert authority.state() == {
        "state": "requested",
        "available": False,
        "retained_input_bytes": 33,
    }

    first.release()

    assert wakeups == ["ready"]
    assert authority.state() == {
        "state": "ready",
        "available": True,
        "retained_input_bytes": 33,
    }
    third = authority.take(33)
    assert third.execution_slot_id == "subprocess:0"

    second.release()
    third.release()
    assert authority.active_lease_count == 0


def test_local_slot_admission_release_is_idempotent_and_close_rejects_new_work():
    authority = LocalSlotAdmissionAuthority(max_slots=1, execution_slot_prefix="local")
    assert authority.request(7)
    lease = authority.take(7)

    lease.release()
    lease.release()
    assert authority.active_lease_count == 0

    authority.close()
    assert authority.state()["state"] == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        authority.request(8)


def test_local_slot_pool_is_shared_across_executor_authorities():
    pool = LocalExecutionSlotPool(
        max_slots=1,
        execution_slot_prefix="shared-subprocess",
    )
    first_authority = pool.create_authority()
    second_authority = pool.create_authority()
    second_wakeups = []
    second_authority.register_wakeup(lambda: second_wakeups.append("ready"))

    assert first_authority.request(10)
    first = first_authority.take(10)
    assert first.execution_slot_id == "shared-subprocess:0"

    assert second_authority.request(20)
    assert second_authority.state()["state"] == "requested"
    assert pool.active_lease_count == 1

    first.release()

    assert second_wakeups == ["ready"]
    second = second_authority.take(20)
    assert second.execution_slot_id == "shared-subprocess:0"
    assert pool.active_lease_count == 1
    second.release()
    assert pool.active_lease_count == 0
