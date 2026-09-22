# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from vane.execution import ref_bundle
from vane.execution.local_resource_graph import LocalResourceUnitContext
from vane.execution.udf_admission import LocalExecutionCapacity, LocalExecutionSlotPool
from vane.execution.udf_data_admission import (
    DataAdmissionLimits,
    DataAdmissionProgressError,
    DataAdmissionQueueFull,
    DataAdmissionTimeout,
    DataAdmissionWaitLimits,
)
from vane.execution.udf_data_lease import DataAllocation, RuntimeDataLedger
from vane.execution.udf_data_wait import WaitingDataAdmissionAuthority
from vane.execution.udf_runtime_admission import RuntimeTaskAdmission, TaskAdmissionLimits


def _wait(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.005)
    raise TimeoutError("byte admission did not transition")


class _Harness:
    def __init__(self, monkeypatch, *, budget=600, slots=1, queued=8, timeout=5, limited=False, ratio=None):
        self.transport = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: budget)
        monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", self.transport)
        self.ledger = RuntimeDataLedger(
            DataAdmissionLimits(
                budget, 100, 100, unit_reservation_ratio=ratio, wait=DataAdmissionWaitLimits(queued, timeout)
            )
        )
        self.capacity = LocalExecutionCapacity(max_slots=slots)
        self.task_runtime = RuntimeTaskAdmission(TaskAdmissionLimits(slots, 32)) if limited else None
        self.queries, self.task_queries, self.pools, self.authorities, self.leases, self.outputs = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

    def query(self, count=1):
        name = str(len(self.queries))
        units = [
            LocalResourceUnitContext(name, f"resource:{name}:udf:{i}", str(i), "subprocess_task") for i in range(count)
        ]
        query = self.ledger.open_query(resource_units=units)
        self.queries.append(query)
        return query, units

    def authority(self, query=None, unit=None, pool=None):
        if query is None:
            query, units = self.query()
            unit = units[0]
        if pool is None:
            pool = LocalExecutionSlotPool(
                max_slots=2, execution_slot_prefix=str(len(self.pools)), execution_capacity=self.capacity
            )
            self.pools.append(pool)
        task_query = None
        if self.task_runtime is not None:
            task_query = self.task_runtime.open_query()
            self.task_queries.append(task_query)
        authority = WaitingDataAdmissionAuthority(
            pool.create_authority(), query, resource_unit=unit, task_query=task_query
        )
        self.authorities.append(authority)
        return authority

    def take(self, authority):
        assert authority.state()["state"] == "ready"
        lease = authority.take(8)
        self.leases.append(lease)
        return lease

    def output(self, query, unit, *, size=100):
        task = query.open_task(query.reserve_task(resource_unit=unit), resource_unit=unit)
        output = task.own_output(DataAllocation("local_shm", str(len(self.outputs)), size))
        self.outputs.append(output)
        task.finish()
        return output

    def close(self):
        for authority in self.authorities:
            authority.close()
        for lease in self.leases:
            lease.release()
        for query in self.queries:
            query.shutdown()
        for query in self.task_queries:
            query.shutdown()
        if self.task_runtime:
            self.task_runtime.close()
        self.ledger.close()
        for output in self.outputs:
            output.release()
        for pool in self.pools:
            pool.close()
        assert self.capacity.reserved_slots == 0
        assert self.ledger.snapshot()["queued_byte_admissions"] == 0
        assert self.ledger.snapshot()["usage_bytes"] == self.transport.snapshot()["usage_bytes"] == 0
        assert self.ledger._wait_admission.snapshot()["queries"] == 0


@pytest.fixture
def harness(monkeypatch):
    instances = []

    def make(**kwargs):
        h = _Harness(monkeypatch, **kwargs)
        instances.append(h)
        return h

    yield make
    for h in instances:
        h.close()


@pytest.mark.parametrize("queued", [-1, True, 1.5, None])
def test_wait_queue_requires_a_bound(queued):
    with pytest.raises(ValueError, match="non-negative integer"):
        DataAdmissionWaitLimits(queued)


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan"), None])
def test_wait_deadline_is_finite_and_positive(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        DataAdmissionWaitLimits(2, timeout)


@pytest.mark.parametrize("limited", [False, True])
def test_transport_wait_holds_no_execution_or_task_capacity_and_wakes_on_release(harness, limited):
    h = harness(limited=limited)
    authority = h.authority()
    states = []
    authority.register_wakeup(lambda: states.append(authority.state()["state"]))
    occupied = h.transport.reserve_task_bytes(300, 300)
    try:
        assert authority.request(8)
        assert authority.state()["state"] == "waiting_bytes"
        assert h.capacity.reserved_slots == 0
        assert h.ledger.snapshot()["reserved_bytes"] == 0
        if h.task_runtime:
            assert h.task_runtime.snapshot()["ready_tasks"] == 0
        occupied.release()
        assert authority.state()["state"] == "ready"
        assert "ready" in states
        assert h.capacity.reserved_slots == 1
        h.take(authority).release()
    finally:
        occupied.release()


@pytest.mark.parametrize("limited", [False, True])
def test_ledger_only_release_wakes_waiter_without_transport_notification(harness, limited):
    h = harness(budget=200, limited=limited)
    query, (unit,) = h.query()
    output = h.output(query, unit)
    authority = h.authority(query, unit)
    assert authority.request(8)
    assert authority.state()["state"] == "waiting_bytes"
    before = h.ledger.snapshot()
    for _ in range(10):
        assert authority.diagnostic_state() == "waiting_bytes"
        assert h.ledger.snapshot() == before
    output.release()
    h.take(authority).release()


@pytest.mark.parametrize("ratio", [None, 0, 0.5, 1])
def test_producer_cannot_consume_the_downstream_envelope(harness, ratio):
    h = harness(budget=400, ratio=ratio)
    query, (producer, consumer) = h.query(2)
    output = h.output(query, producer)
    upstream, downstream = h.authority(query, producer), h.authority(query, consumer)
    assert upstream.request(8)
    assert upstream.state()["state"] == "waiting_bytes"
    assert downstream.request(8)
    lease = h.take(downstream)
    task = query.open_task(lease.lease["local_data_reservation"], resource_unit=consumer)
    task.hold_inputs([DataAllocation("local_shm", "0", 100)])
    output.release()
    assert upstream.state()["state"] == "waiting_bytes"
    task.finish()
    lease.release()
    h.take(upstream).release()


def test_impossible_plan_fails_before_any_reservation(harness):
    h = harness(budget=200)
    with pytest.raises(DataAdmissionProgressError, match="one complete envelope per plan UDF"):
        h.query(2)
    assert h.ledger.snapshot()["queries"] == 0
    assert h.transport.snapshot()["usage_bytes"] == 0


@pytest.mark.parametrize("limited", [False, True])
def test_timeout_wakes_without_release_and_retires_pending_task_request(harness, limited):
    h = harness(timeout=0.05, limited=limited)
    authority = h.authority()
    woke = threading.Event()
    authority.register_wakeup(lambda: woke.set() if authority.state()["state"] == "failed" else None)
    occupied = h.transport.reserve_task_bytes(300, 300)
    try:
        authority.request(8)
        assert woke.wait(5)
        assert "deadline expired" in authority.state()["error"]
        assert h.ledger.snapshot()["queued_byte_admissions"] == 0
        assert h.capacity.reserved_slots == 0
        if h.task_runtime:
            assert h.task_runtime.snapshot()["queued_tasks"] == 0
        occupied.release()
        assert authority.state()["state"] == "failed"
    finally:
        occupied.release()


@pytest.mark.parametrize("limited", [False, True])
def test_queue_full_cannot_leave_a_hidden_task_request(harness, limited):
    h = harness(queued=1, limited=limited)
    first, second = h.authority(), h.authority()
    occupied = h.transport.reserve_task_bytes(300, 300)
    try:
        first.request(8)
        with pytest.raises(DataAdmissionQueueFull, match="queue is full"):
            second.request(8)
        assert h.ledger.snapshot()["queued_byte_admissions"] == 1
        occupied.release()
        h.take(first).release()
        assert h.capacity.reserved_slots == 0
        assert second.state()["state"] in {"closed", "failed"}
    finally:
        occupied.release()


def test_drain_preserves_admitted_query_and_shutdown_removes_its_waiters(harness):
    h = harness()
    query, (unit,) = h.query()
    authority = h.authority(query, unit)
    occupied = h.transport.reserve_task_bytes(300, 300)
    try:
        authority.request(8)
        h.ledger.drain()
        with pytest.raises(RuntimeError, match="draining"):
            h.query()
        query.shutdown()
        assert authority.state()["state"] == "closed"
        h.ledger.close()
        occupied.release()
        assert h.capacity.reserved_slots == 0
    finally:
        occupied.release()


def test_callback_removal_survives_transport_and_deadline_notifications(harness):
    h = harness(timeout=0.05)
    authority = h.authority()
    calls = []
    authority.register_wakeup(lambda: calls.append(1))
    authority.register_wakeup(None)
    occupied = h.transport.reserve_task_bytes(300, 300)
    try:
        authority.request(8)
        _wait(lambda: authority.state()["state"] == "failed")
        occupied.release()
        assert calls == []
        assert "deadline expired" in authority.state()["error"]
    finally:
        occupied.release()


def test_output_release_notifies_after_publishing_owner_state(harness):
    h = harness(budget=200)
    query, (unit,) = h.query()
    output = h.output(query, unit)
    authority = h.authority(query, unit)
    observed = []
    authority.register_wakeup(lambda: observed.append((output.state, output.release())))
    authority.request(8)
    output.release()
    assert observed and all(state == "released" and released is False for state, released in observed)
    h.take(authority).release()


def test_deadline_crossed_inside_transport_reservation_never_grants_a_worker(harness, monkeypatch):
    from vane.execution.request_deadline import MonotonicDeadline

    h = harness()
    query, (unit,) = h.query()
    authority = h.authority(query, unit)
    reserved, proceed = threading.Event(), threading.Event()
    original = query.reserve_task
    expired = False

    def reserve(**kwargs):
        reservation = original(**kwargs)
        reserved.set()
        assert proceed.wait(5)
        return reservation

    monkeypatch.setattr(query, "reserve_task", reserve)
    monkeypatch.setattr(MonotonicDeadline, "expired", lambda self: expired)
    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(authority.request, 8)
        try:
            assert reserved.wait(5)
            expired = True
        finally:
            proceed.set()
        with pytest.raises(DataAdmissionTimeout):
            future.result(timeout=5)
    assert h.capacity.reserved_slots == 0
    assert h.ledger.snapshot()["usage_bytes"] == 0


def test_failed_unused_reservation_cleanup_keeps_a_retry_owner(harness, monkeypatch):
    h = harness()
    authority = h.authority()
    authority.request(8)
    reservation = authority._reservation

    def fail_release():
        raise OSError("planned transport reservation cleanup failure")

    with monkeypatch.context() as fault:
        fault.setattr(reservation.transport, "release", fail_release)
        with pytest.raises(OSError, match="cleanup failure"):
            authority.close()
        assert h.ledger.snapshot()["usage_bytes"] == 200
        assert h.capacity.reserved_slots == 0
        assert authority._reservation is reservation
    authority.close()
    assert h.ledger.snapshot()["usage_bytes"] == 0


@pytest.mark.parametrize("same_pool", [False, True])
@pytest.mark.parametrize("limited", [False, True])
def test_older_byte_waiter_keeps_its_turn_among_newer_ordinary_work(harness, same_pool, limited):
    h = harness(limited=limited)
    waiting = h.authority()
    pool = h.pools[0]
    if not same_pool:
        pool = LocalExecutionSlotPool(max_slots=2, execution_slot_prefix="ordinary", execution_capacity=h.capacity)
        h.pools.append(pool)
    ordinary = pool.create_authority()
    occupied = h.transport.reserve_task_bytes(300, 300)
    try:
        waiting.request(8)
        ordinary.request(8)
        lease = ordinary.take(8)
        h.leases.append(lease)
        # Bytes become free while a legacy caller holds the only global thread.
        occupied.release()
        assert waiting.state()["state"] in {"requested", "waiting_bytes"}
        ordinary.request(8)
        lease.release()
        # Pool/source round robin can admit the ordinary source once first,
        # but continuously replacing it must not suppress the older request.
        for _ in range(2):
            if waiting.state()["state"] == "ready":
                break
            assert ordinary.state()["state"] == "ready"
            lease = ordinary.take(8)
            h.leases.append(lease)
            ordinary.request(8)
            lease.release()
        h.take(waiting).release()
    finally:
        ordinary.close()
        occupied.release()


def test_busy_pool_does_not_park_bytes_needed_by_a_different_pool(harness):
    h = harness(budget=400, slots=2)
    busy = h.authority()
    ordinary = h.pools[0].create_authority()
    second_ordinary = h.pools[0].create_authority()
    try:
        for caller in (ordinary, second_ordinary):
            caller.request(8)
            h.leases.append(caller.take(8))
        busy.request(8)
        assert h.ledger.snapshot()["reserved_bytes"] == 0
        assert h.capacity.reserved_slots == 2
        # Return execution-only capacity, retaining both busy pool/result slots.
        h.leases[-1].complete_execution()
        other = h.authority()
        other.request(8)
        h.take(other).release()
        assert busy.state()["state"] == "requested"
    finally:
        ordinary.close()
        second_ordinary.close()


def test_ready_grant_racing_request_snapshot_is_not_requeued(harness, monkeypatch):
    h = harness()
    authority = h.authority()
    occupied = h.transport.reserve_task_bytes(300, 300)
    original = authority._base.state
    released = False

    def state():
        nonlocal released
        snapshot = original()
        if snapshot["state"] == "requested" and not released:
            released = True
            occupied.release()
        return snapshot

    monkeypatch.setattr(authority._base, "state", state)
    try:
        authority.request(8)
        assert released
        assert h.ledger.snapshot()["queued_byte_admissions"] == 0
        h.take(authority).release()
        assert h.ledger.snapshot()["queued_byte_admissions"] == 0
    finally:
        occupied.release()


def test_query_shutdown_during_callback_subscription_removes_the_callback(harness, monkeypatch):
    h = harness()
    query, (unit,) = h.query()
    before = set(ref_bundle._local_shm_budget_wakeup_callbacks)
    original = ref_bundle.register_local_shm_ref_budget_wakeup

    def register(callback):
        unregister = original(callback)
        query.shutdown()
        return unregister

    monkeypatch.setattr(ref_bundle, "register_local_shm_ref_budget_wakeup", register)
    with pytest.raises(RuntimeError, match="query data scope is closed"):
        h.authority(query, unit)
    assert set(ref_bundle._local_shm_budget_wakeup_callbacks) == before
    h.ledger.close()


def test_queue_rejection_preserves_primary_error_when_cleanup_also_fails(harness, monkeypatch):
    h = harness(queued=0)
    authority = h.authority()
    occupied = h.transport.reserve_task_bytes(300, 300)

    def fail_close():
        raise OSError("planned capacity cleanup failure")

    try:
        with monkeypatch.context() as fault:
            fault.setattr(authority._base, "close", fail_close)
            with pytest.raises(DataAdmissionQueueFull) as caught:
                authority.request(8)
            assert isinstance(caught.value.__cause__, OSError)
        authority.close()
        assert h.ledger.snapshot()["queued_byte_admissions"] == 0
    finally:
        occupied.release()


def test_rejected_guard_cannot_spin_in_another_pools_completion_callback(harness):
    h = harness(budget=400)
    query, (producer, consumer) = h.query(2)
    output = h.output(query, producer)
    upstream, downstream = h.authority(query, producer), h.authority(query, consumer)
    upstream.request(8)
    downstream.request(8)
    lease = h.take(downstream)
    # The byte-gated upstream is probed on both pools' turns. With no grant,
    # completion must return promptly, even though the waiter stays pending.
    with ThreadPoolExecutor(max_workers=1) as threads:
        completion = threads.submit(lease.release)
        try:
            completion.result(timeout=1)
            assert upstream.state()["state"] == "waiting_bytes"
        finally:
            output.release()
    h.take(upstream).release()
