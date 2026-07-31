# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import threading
import time
import weakref
from concurrent.futures import Future

import pytest

import duckdb.execution.udf_task_admission as task_admission
from duckdb.execution.udf_admission import (
    LocalExecutionSlotPool,
    LocalSlotAdmissionAuthority,
)
from duckdb.execution.udf_task_admission import TaskAdmissionController


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
        "stage_id": "stage:q:node:3:udf",
        "cpus": 1.0,
        "gpus": 0.0,
        "memory_bytes": 256,
        "udf_task_input_max_bytes": 128,
    }


def test_task_admission_requires_explicit_query_driver_handle():
    with pytest.raises(ValueError, match="query driver handle"):
        TaskAdmissionController(_payload(), driver=None)


@pytest.mark.parametrize("callback", [False, True])
def test_task_admission_setup_failure_cancels_the_remote_request(callback):
    driver = _Driver()
    driver.acquire_query_task_lease = _RemoteMethod(
        lambda _request: _FailingObjectRef(callback=callback),
    )
    controller = TaskAdmissionController(_payload(), driver=driver)

    expected = "callback registration" if callback else "Future factory"
    with pytest.raises(RuntimeError, match=expected):
        controller.request(64)

    request = driver.acquire_query_task_lease.calls[0][0][0]
    assert controller.state()["state"] == "failed"
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
            "stage_id": request["stage_id"],
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
    controller = TaskAdmissionController(_payload(), driver=driver)
    controller.register_wakeup(lambda: wakeups.append("ready"))

    assert controller.request(64)
    assert not controller.request(64)
    assert len(driver.acquire_query_task_lease.calls) == 1
    request = driver.acquire_query_task_lease.calls[0][0][0]
    assert request["retained_input_bytes"] == 64
    assert request["resources"]["object_store_bytes"] == 128
    assert controller.state() == {
        "state": "requested",
        "available": False,
        "retained_input_bytes": 64,
    }

    driver.requests[0].resolve(_grant(request))

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
    controller = TaskAdmissionController(_payload(), driver=driver)
    assert controller.request(64)
    request = driver.acquire_query_task_lease.calls[0][0][0]
    driver.requests[0].resolve(_grant(request))

    with pytest.raises(RuntimeError, match="retained input bytes"):
        controller.take(32)

    assert controller.state()["state"] == "ready"
    admission = controller.take(64)
    assert admission.lease["lease_id"] == "lease-1"
    admission.handoff()


def test_task_admission_preserves_async_denial_reason():
    driver = _Driver()
    controller = TaskAdmissionController(_payload(), driver=driver)
    assert controller.request(16)

    driver.requests[0].resolve(
        {
            "granted": False,
            "blocked_reason": "query_not_registered",
            "fatal": True,
        }
    )

    state = controller.state()
    assert state["state"] == "failed"
    assert "query_not_registered" in state["error"]
    with pytest.raises(RuntimeError, match="query_not_registered"):
        controller.request(16)


def test_task_admission_close_cancels_pending_and_ready_leases():
    pending_driver = _Driver()
    pending = TaskAdmissionController(_payload(), driver=pending_driver)
    assert pending.request(32)
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
    ready = TaskAdmissionController(_payload(), driver=ready_driver)
    assert ready.request(48)
    ready_request = ready_driver.acquire_query_task_lease.calls[0][0][0]
    ready_driver.requests[0].resolve(_grant(ready_request))

    ready.close()

    assert ready.state()["state"] == "closed"
    _wait_for_remote_calls(ready_driver.cancel_query_task_lease_request, 1)
    assert ready_driver.cancel_query_task_lease_request.calls == [((ready_request,), {})]


def test_pending_admission_future_does_not_retain_closed_controller():
    driver = _Driver()
    controller = TaskAdmissionController(_payload(), driver=driver)
    assert controller.request(32)
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
    controller = TaskAdmissionController(_payload(), driver=driver)
    assert controller.request(24)
    request = driver.acquire_query_task_lease.calls[0][0][0]
    driver.requests[0].resolve(_grant(request))

    admission = controller.take(24)
    admission.release()
    admission.release()

    _wait_for_remote_calls(driver.cancel_query_task_lease_request, 1)
    assert driver.cancel_query_task_lease_request.calls == [((request,), {})]


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
    controller = TaskAdmissionController(_payload(), driver=driver)
    assert controller.request(24)

    controller.close()

    assert acknowledged.wait(timeout=1.0)
    assert len(driver.cancel_query_task_lease_request.calls) == 2


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
    controller = TaskAdmissionController(_payload(), driver=driver)
    assert controller.request(24)

    controller.close()

    assert acknowledged.wait(timeout=1.0)
    assert len(driver.cancel_query_task_lease_request.calls) == 2
    assert first_response._future.cancelled()


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

    def cancel(_request):
        response = _ObjectRef()

        def acknowledge():
            if not response._future.cancelled():
                response.resolve({"cancelled": True})

        timer = threading.Timer(0.015, acknowledge)
        timer.daemon = True
        timer.start()
        response_timers.append(timer)
        return response

    driver.cancel_query_task_lease_request = _RemoteMethod(cancel)
    cancellation = task_admission._TaskAdmissionCancellation(
        driver=driver,
        request={"request_id": "slow-cancel", "resources": {}},
    )

    cancellation.start()

    deadline = time.monotonic() + 1.0
    while not cancellation._done and time.monotonic() < deadline:
        time.sleep(0.005)
    assert cancellation._done
    assert len(driver.cancel_query_task_lease_request.calls) == 2
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
                "resources": {},
            },
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
    scheduler_threads = [thread for thread in threading.enumerate() if thread.name == "vane-task-admission-cleanup"]
    assert len(scheduler_threads) == 1

    for response in responses:
        response.resolve({"cancelled": True})
    deadline = time.monotonic() + 1.0
    while not all(cancellation._done for cancellation in cancellations) and time.monotonic() < deadline:
        time.sleep(0.005)
    assert all(cancellation._done for cancellation in cancellations)


def test_task_admission_blocked_rpc_submissions_use_bounded_workers_without_blocking_deadlines():
    driver = _Driver()
    release_submissions = threading.Event()
    all_workers_blocked = threading.Event()
    entered_lock = threading.Lock()
    entered = 0

    def cancel(_request):
        nonlocal entered
        with entered_lock:
            entered += 1
            if entered == task_admission._TASK_ADMISSION_CLEANUP_SUBMISSION_WORKERS:
                all_workers_blocked.set()
        release_submissions.wait(timeout=5.0)
        return _resolved_ref({"cancelled": True})

    driver.cancel_query_task_lease_request = _RemoteMethod(cancel)
    cancellation_count = task_admission._TASK_ADMISSION_CLEANUP_SUBMISSION_WORKERS + 64
    cancellations = [
        task_admission._TaskAdmissionCancellation(
            driver=driver,
            request={
                "request_id": f"blocked-submit:{index}",
                "resources": {},
            },
        )
        for index in range(cancellation_count)
    ]

    started_at = time.monotonic()
    for cancellation in cancellations:
        cancellation.start()
    start_elapsed = time.monotonic() - started_at

    deadline_fired = threading.Event()
    deadline = task_admission._TASK_ADMISSION_CANCELLATION_SCHEDULER.create(
        deadline_fired.set,
    )
    task_admission._TASK_ADMISSION_CANCELLATION_SCHEDULER.schedule(
        deadline,
        0.01,
    )
    try:
        assert start_elapsed < 1.0
        assert all_workers_blocked.wait(timeout=2.0)
        assert deadline_fired.wait(timeout=1.0)
        assert (
            len(driver.cancel_query_task_lease_request.calls)
            == task_admission._TASK_ADMISSION_CLEANUP_SUBMISSION_WORKERS
        )
        submission_threads = [
            thread for thread in threading.enumerate() if thread.name.startswith("vane-task-admission-submit-")
        ]
        assert len(submission_threads) == task_admission._TASK_ADMISSION_CLEANUP_SUBMISSION_WORKERS
    finally:
        release_submissions.set()

    _wait_for_remote_calls(
        driver.cancel_query_task_lease_request,
        cancellation_count,
    )
    completion_deadline = time.monotonic() + 1.0
    while not all(cancellation._done for cancellation in cancellations) and time.monotonic() < completion_deadline:
        time.sleep(0.005)
    assert all(cancellation._done for cancellation in cancellations)


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
