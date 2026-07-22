# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import Future

import pytest

from vane.execution.udf_admission import (
    LocalExecutionSlotPool,
    LocalSlotAdmissionAuthority,
)
from vane.execution.udf_task_admission import TaskAdmissionController


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


class _Driver:
    def __init__(self):
        self.requests: list[_ObjectRef] = []
        self.acquire_query_task_lease = _RemoteMethod(self._acquire)
        self.cancel_query_task_lease_request = _RemoteMethod(lambda *_args, **_kwargs: None)

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
    assert pending_driver.cancel_query_task_lease_request.calls == [
        ((pending_request["request_id"],), {"submitted": False})
    ]

    pending_driver.requests[0].resolve(_grant(pending_request))
    assert pending_driver.cancel_query_task_lease_request.calls == [
        ((pending_request["request_id"],), {"submitted": False}),
        ((pending_request["request_id"],), {"submitted": False}),
    ]

    ready_driver = _Driver()
    ready = TaskAdmissionController(_payload(), driver=ready_driver)
    assert ready.request(48)
    ready_request = ready_driver.acquire_query_task_lease.calls[0][0][0]
    ready_driver.requests[0].resolve(_grant(ready_request))

    ready.close()

    assert ready.state()["state"] == "closed"
    assert ready_driver.cancel_query_task_lease_request.calls == [
        ((ready_request["request_id"],), {"submitted": False})
    ]


def test_taken_task_admission_abandons_if_submission_never_takes_ownership():
    driver = _Driver()
    controller = TaskAdmissionController(_payload(), driver=driver)
    assert controller.request(24)
    request = driver.acquire_query_task_lease.calls[0][0][0]
    driver.requests[0].resolve(_grant(request))

    admission = controller.take(24)
    admission.release()
    admission.release()

    assert driver.cancel_query_task_lease_request.calls == [((request["request_id"],), {"submitted": False})]


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
