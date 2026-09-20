# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from vane.execution import request_admission as admission
from vane.execution.request_admission import (
    RequestAdmissionLimits,
    RequestCancelled,
    RequestQueueFull,
    RequestQueueTimeout,
    RuntimeRequestAdmission,
)


@pytest.mark.parametrize(
    "changes",
    [{"max_active_requests": value} for value in (0, -1, True, 1.5)]
    + [{"max_queued_requests": value} for value in (-1, True, 1.5)]
    + [{"queue_timeout": value} for value in (-1, float("inf"), float("nan"), True, "1")],
)
def test_request_limits_reject_invalid_values(changes):
    options = {"max_active_requests": 1, "max_queued_requests": 2, **changes}
    with pytest.raises(ValueError):
        RequestAdmissionLimits(**options)


@pytest.mark.parametrize("capacity", [1, 2])
def test_fifo_waiters_cannot_be_bypassed_by_new_requests(capacity):
    runtime = RuntimeRequestAdmission(RequestAdmissionLimits(capacity, 3))
    active = [runtime.request().take() for _ in range(capacity)]
    first, second = runtime.request(), runtime.request()
    assert first.state == second.state == "queued"
    active[0].release()
    assert first.state == "ready"
    newer = runtime.request()
    assert second.state == newer.state == "queued"
    first.take().release()
    assert second.state == "ready" and newer.state == "queued"
    second.take().release()
    newer.take().release()
    for lease in active:
        lease.release()
    runtime.close()
    assert runtime.snapshot()["completed_requests"] == capacity + 3


@pytest.mark.parametrize("queued", [0, 1, 3])
def test_queue_capacity_counts_waiters_and_rejection_reserves_nothing(queued):
    runtime = RuntimeRequestAdmission(RequestAdmissionLimits(1, queued))
    lease = runtime.request().take()
    tickets = [runtime.request() for _ in range(queued)]
    with pytest.raises(RequestQueueFull):
        runtime.request()
    state = runtime.snapshot()
    assert state["active_requests"] == 1 and state["queued_requests"] == queued
    assert state["rejected_requests"] == 1
    for ticket in tickets:
        assert ticket.cancel()
        assert not ticket.cancel()
    lease.release()
    runtime.close()


def test_expiration_removes_non_head_waiters_and_preserves_fifo(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(admission, "time", SimpleNamespace(monotonic=lambda: now[0]))
    runtime = RuntimeRequestAdmission(RequestAdmissionLimits(1, 3))
    lease = runtime.request().take()
    oldest = runtime.request(queue_timeout=10)
    expired = runtime.request(queue_timeout=1)
    newest = runtime.request(queue_timeout=10)
    now[0] = 12
    with pytest.raises(RequestQueueTimeout):
        expired.take()
    assert runtime.snapshot()["queued_requests"] == 2
    lease.release()
    assert oldest.state == "ready" and newest.state == "queued"
    oldest.take().release()
    newest.take().release()
    assert runtime.snapshot()["queue_wait_seconds"] == 4
    runtime.close()


def test_queue_deadline_does_not_expire_an_admitted_request(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(admission, "time", SimpleNamespace(monotonic=lambda: now[0]))
    runtime = RuntimeRequestAdmission(RequestAdmissionLimits(1, 1, queue_timeout=1))
    first = runtime.request()
    now[0] = 12
    lease = first.take()
    with pytest.raises(RequestQueueTimeout):
        runtime.request(queue_timeout=0)
    lease.release()
    runtime.request(queue_timeout=0).take().release()
    runtime.close()


def test_waiting_deadline_wakes_without_a_release_or_poll():
    runtime = RuntimeRequestAdmission(RequestAdmissionLimits(1, 1))
    lease = runtime.request().take()
    queued = runtime.request(queue_timeout=0.03)
    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(queued.take)
        with pytest.raises(RequestQueueTimeout):
            future.result(timeout=3)
    assert runtime.snapshot()["queued_requests"] == 0
    lease.release()
    runtime.close()


@pytest.mark.parametrize("operation", ["cancel", "drain"])
def test_cancellation_wakes_waiters_and_never_releases_a_running_lease(operation):
    runtime = RuntimeRequestAdmission(RequestAdmissionLimits(1, 2))
    active = runtime.request()
    lease = active.take()
    queued = runtime.request(queue_timeout=1e100)
    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(queued.take)
        if operation == "cancel":
            assert queued.cancel()
        else:
            runtime.drain()
        with pytest.raises(RequestCancelled):
            future.result(timeout=3)
    assert not active.cancel()
    with pytest.raises(TimeoutError, match="active execution"):
        runtime.close()
    lease.release()
    runtime.close()


def test_drain_retires_unclaimed_grants_and_close_waits_for_claimed_execution():
    runtime = RuntimeRequestAdmission(RequestAdmissionLimits(2, 1))
    claimed = runtime.request().take()
    ready, queued = runtime.request(), runtime.request()
    entered = threading.Event()

    def close():
        runtime.drain()
        entered.set()
        runtime.close(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as threads:
        closing = threads.submit(close)
        assert entered.wait(3)
        assert not closing.done()
        for ticket in (ready, queued):
            with pytest.raises(RequestCancelled, match="drain"):
                ticket.take()
        with pytest.raises(RuntimeError, match="draining"):
            runtime.request()
        claimed.release()
        closing.result(timeout=3)
    assert runtime.snapshot()["active_requests"] == runtime.snapshot()["queued_requests"] == 0
    assert runtime.snapshot()["drained_requests"] == 2
    assert runtime.snapshot()["closed"]


def test_queued_failure_exceptions_are_fresh_without_cached_tracebacks(monkeypatch):
    now = [1.0]
    monkeypatch.setattr(admission, "time", SimpleNamespace(monotonic=lambda: now[0]))
    runtime = RuntimeRequestAdmission(RequestAdmissionLimits(1, 1, queue_timeout=1))
    lease = runtime.request().take()
    queued = runtime.request()
    now[0] = 3.0
    errors = []
    for _ in range(20):
        with pytest.raises(RequestQueueTimeout) as info:
            queued.take()
        errors.append(info.value)
    assert len({id(error) for error in errors}) == 20
    assert runtime.snapshot()["timed_out_requests"] == 1
    lease.release()
    runtime.close()


def test_simultaneous_claim_and_cancel_have_one_owner():
    runtime = RuntimeRequestAdmission(RequestAdmissionLimits(1, 1))
    with ThreadPoolExecutor(max_workers=2) as threads:
        for _ in range(50):
            ticket = runtime.request()
            barrier = threading.Barrier(2)

            def take():
                barrier.wait(timeout=3)
                try:
                    return ticket.take()
                except RequestCancelled:
                    return None

            def cancel():
                barrier.wait(timeout=3)
                return ticket.cancel()

            claimed, cancelled = threads.submit(take), threads.submit(cancel)
            lease = claimed.result(timeout=3)
            assert cancelled.result(timeout=3) == (lease is None)
            if lease is not None:
                assert runtime.snapshot()["active_requests"] == 1
                lease.release()
            assert runtime.snapshot()["active_requests"] == 0
    runtime.close()


def test_concurrent_waiters_start_in_fifo_order():
    runtime = RuntimeRequestAdmission(RequestAdmissionLimits(1, 3))
    blocker = runtime.request().take()
    tickets = [runtime.request() for _ in range(3)]
    entered = [threading.Event() for _ in tickets]
    release = [threading.Event() for _ in tickets]
    order = []

    def execute(index):
        lease = tickets[index].take()
        try:
            order.append(index)
            entered[index].set()
            assert release[index].wait(5)
        finally:
            lease.release()

    with ThreadPoolExecutor(max_workers=3) as threads:
        futures = [threads.submit(execute, i) for i in range(3)]
        try:
            blocker.release()
            for index in range(3):
                assert entered[index].wait(3)
                assert order == list(range(index + 1))
                assert runtime.snapshot()["active_requests"] == 1
                release[index].set()
            for future in futures:
                future.result(timeout=3)
        finally:
            blocker.release()
            for event in release:
                event.set()
    runtime.close()


def test_preparation_requires_a_live_claim_from_the_same_runtime():
    runtime = RuntimeRequestAdmission(RequestAdmissionLimits(1, 1))
    other = RuntimeRequestAdmission(RequestAdmissionLimits(1, 1))
    ready, queued = runtime.request(), runtime.request()
    foreign = other.request()
    foreign_lease = foreign.take()
    for ticket in (None, ready, queued, foreign):
        with pytest.raises(RuntimeError, match="live claim"):
            runtime.require_claimed(ticket)
    lease = ready.take()
    runtime.require_claimed(ready)
    runtime.drain()
    runtime.require_claimed(ready)
    with pytest.raises(RuntimeError, match="live claim"):
        runtime.require_claimed(queued)
    lease.release()
    with pytest.raises(RuntimeError, match="live claim"):
        runtime.require_claimed(ready)
    foreign_lease.release()
    runtime.close()
    other.close()
