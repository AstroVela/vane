# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from vane.execution.ref_bundle import LocalShmBudgetManager
from vane.execution.udf_admission import AdmissionLease, LocalExecutionCapacity, LocalExecutionSlotPool
from vane.execution.udf_lifecycle import ExecutionCancellationScope
from vane.execution.udf_runtime_admission import RuntimeTaskAdmission, TaskAdmissionLimits, TaskAdmissionQueueFull


class _Harness:
    def __init__(self, running, queued):
        self.runtime = RuntimeTaskAdmission(TaskAdmissionLimits(running, queued))
        self.queries = []
        self.pools = []
        self.leases = []

    def query(self):
        query = self.runtime.open_query()
        self.queries.append(query)
        return query

    def pool(self, slots=1, execution_capacity=None):
        pool = LocalExecutionSlotPool(
            max_slots=slots,
            execution_slot_prefix=f"model:{len(self.pools)}",
            execution_capacity=execution_capacity,
        )
        self.pools.append(pool)
        return pool

    def authority(self, query=None, pool=None):
        return (query or self.query()).create_authority((pool or self.pool()).create_authority())

    def take(self, authority, retained=8):
        lease = authority.take(retained)
        self.leases.append(lease)
        return lease

    def close(self):
        for query in self.queries:
            query.shutdown()
        for lease in self.leases:
            lease.release()
        self.runtime.close()
        for pool in self.pools:
            assert pool.active_lease_count == 0
            pool.close()


@pytest.fixture
def admission():
    harnesses = []

    def create(running=1, queued=10):
        harness = _Harness(running, queued)
        harnesses.append(harness)
        return harness

    yield create
    for harness in harnesses:
        harness.close()


@pytest.mark.parametrize("running,queued", [(0, 0), (-1, 1), (True, 1), (1.0, 1), (1, -1), (1, False), (1, 2.0)])
def test_limits_reject_ambiguous_or_unbounded_values(running, queued):
    with pytest.raises(ValueError, match="integer"):
        TaskAdmissionLimits(running, queued)


def test_busy_pool_never_holds_runtime_capacity_and_does_not_block_other_models(admission):
    h = admission()
    pool = h.pool()
    external = pool.create_authority()
    external.request(8)
    held = h.take(external)
    query = h.query()
    busy = h.authority(query, pool)
    other = h.authority(query)
    assert busy.request(8)
    assert busy.state()["state"] == "requested"
    assert h.runtime.snapshot()["ready_tasks"] == 0
    assert other.request(8)
    assert other.state()["state"] == "ready"
    h.take(other).release()
    assert busy.state()["state"] == "requested"
    held.release()
    assert busy.state()["state"] == "ready"
    h.take(busy).release()
    external.close()


def test_pending_tasks_rotate_by_query_instead_of_fifo_favoring_many_udf_nodes(admission):
    h = admission()
    blocker = h.authority()
    blocker.request(8)
    held = h.take(blocker)
    first_query, second_query = h.query(), h.query()
    authorities = {}
    order = []
    for name, query in [
        ("a1", first_query),
        ("a2", first_query),
        ("a3", first_query),
        ("b1", second_query),
        ("b2", second_query),
    ]:
        authority = h.authority(query)
        authority.register_wakeup(lambda name=name: order.append(name))
        authorities[name] = authority
        authority.request(8)
    assert h.runtime.snapshot()["queued_tasks"] == 5
    held.release()
    for name in ["a1", "b1", "a2", "b2", "a3"]:
        assert order[-1] == name
        h.take(authorities[name]).release()
    assert order == ["a1", "b1", "a2", "b2", "a3"]


@pytest.mark.parametrize("queued", [0, 1])
def test_full_queue_rejects_without_reserving_capacity_and_can_be_retried(admission, queued):
    h = admission(queued=queued)
    busy = h.authority()
    busy.request(8)
    held = h.take(busy)
    pending_query = h.query()
    pending = h.authority(pending_query)
    if queued:
        pending.request(8)
    refused = h.authority()
    for _ in range(20):
        with pytest.raises(TaskAdmissionQueueFull):
            refused.request(8)
        assert refused.state() == {"state": "idle", "available": False, "retained_input_bytes": 0}
        assert h.runtime.snapshot()["queued_tasks"] == queued
    pending_query.shutdown()
    held.release()
    assert refused.request(8)
    h.take(refused).release()


def test_runtime_execution_completion_does_not_release_buffered_result_slot(admission):
    h = admission()
    pool = h.pool()
    first, same_pool, other_pool = h.authority(pool=pool), h.authority(pool=pool), h.authority()
    first.request(8)
    lease = h.take(first)
    same_pool.request(8)
    other_pool.request(8)
    lease.complete_execution()
    lease.complete_execution()
    assert pool.active_lease_count == 1
    assert same_pool.state()["state"] == "requested"
    assert other_pool.state()["state"] == "ready"
    h.take(other_pool).release()
    assert h.runtime.snapshot()["running_tasks"] == 0
    assert h.runtime.snapshot()["ready_tasks"] == 0
    lease.release()
    assert same_pool.state()["state"] == "ready"
    h.take(same_pool).release()


def test_query_cancel_releases_waiting_and_ready_but_retains_running_tasks(admission):
    h = admission(running=2)
    active_query, ready_query, pending_query = h.query(), h.query(), h.query()
    active, ready, pending = [h.authority(query) for query in (active_query, ready_query, pending_query)]
    active.request(8)
    lease = h.take(active)
    ready.request(8)
    pending.request(8)
    pending_query.shutdown()
    assert not pending_query.cleanup_pending()
    assert h.runtime.snapshot()["queued_tasks"] == 0
    ready_query.shutdown()
    assert not ready_query.cleanup_pending()
    assert h.runtime.snapshot()["ready_tasks"] == 0
    active_query.shutdown(kill=True)
    assert active_query.cleanup_pending()
    assert h.runtime.snapshot()["running_tasks"] == 1
    with pytest.raises(TimeoutError, match="active queries"):
        h.runtime.close()
    with pytest.raises(RuntimeError, match="draining"):
        h.runtime.open_query()
    lease.release()
    assert not active_query.cleanup_pending()
    h.runtime.close()
    assert h.runtime.snapshot()["closed"]


def test_drain_allows_existing_queries_to_finish_and_timed_close_waits(admission):
    h = admission()
    authority = h.authority()
    h.runtime.drain()
    assert authority.request(8)
    lease = h.take(authority)
    h.queries[0].shutdown()
    with ThreadPoolExecutor(max_workers=1) as executor:
        closing = executor.submit(h.runtime.close, timeout=5)
        assert not closing.done()
        lease.release()
        closing.result(timeout=5)


def test_backend_close_wakes_pending_queries_even_while_runtime_is_full(admission):
    h = admission()
    active = h.authority()
    active.request(8)
    lease = h.take(active)
    pool = h.pool()
    pending = h.authority(pool=pool)
    wakeup = threading.Event()
    pending.register_wakeup(wakeup.set)
    pending.request(8)
    pool.close()
    assert wakeup.is_set()
    with pytest.raises(RuntimeError, match="capacity is closed"):
        pending.state()
    assert h.runtime.snapshot()["queued_tasks"] == 0
    lease.release()


def test_callback_failure_does_not_skip_other_queries_or_call_exception_str(admission):
    h = admission(running=2)
    pool = h.pool(2)
    external = pool.create_authority()
    external.request(8)
    held = h.take(external)
    external.request(8)
    second = h.take(external)
    failed, success = h.authority(pool=pool), h.authority(pool=pool)

    class CallbackError(RuntimeError):
        def __str__(self):
            raise AssertionError("failure diagnostics must not call exception str")

    def broken_wakeup():
        raise CallbackError("callback broke")

    failed.register_wakeup(broken_wakeup)
    successful_wakeup = threading.Event()
    success.register_wakeup(successful_wakeup.set)
    failed.request(8)
    success.request(8)
    held.release()
    second.release()
    assert successful_wakeup.is_set()
    with pytest.raises(RuntimeError, match="callback broke"):
        failed.state()
    failed.close()
    h.take(success).release()
    external.close()


def test_concurrent_pool_wakeups_respect_global_limit_and_return_each_lease_once(admission):
    h = admission(running=4, queued=32)
    authorities = [h.authority(pool=h.pool(4)) for _ in range(16)]
    barrier = threading.Barrier(len(authorities))
    running = 0
    maximum = 0
    lock = threading.Lock()

    def execute(authority):
        nonlocal running, maximum
        ready = threading.Event()
        authority.register_wakeup(ready.set)
        barrier.wait(timeout=5)
        for _ in range(10):
            ready.clear()
            authority.request(8)
            assert ready.wait(5)
            lease = authority.take(8)
            with lock:
                running += 1
                maximum = max(maximum, running)
                assert running <= 4
            with lock:
                running -= 1
            lease.complete_execution()
            lease.release()
            lease.release()

    with ThreadPoolExecutor(max_workers=len(authorities)) as executor:
        list(executor.map(execute, authorities))
    assert 1 <= maximum <= 4
    assert h.runtime.snapshot()["running_tasks"] == 0
    assert h.runtime.snapshot()["queued_tasks"] == 0
    assert h.runtime.snapshot()["ready_tasks"] == 0


def test_lease_cleanup_finishes_once_and_releases_backend_even_on_completion_error():
    events = []

    def finish():
        events.append("finished")
        raise RuntimeError("completion failed")

    lease = AdmissionLease(
        "test", 8, {}, _release_callback=lambda: events.append("released"), _execution_finished_callback=finish
    )
    with pytest.raises(RuntimeError, match="local execution cleanup"):
        lease.handoff()
    with pytest.raises(RuntimeError, match="completion failed"):
        lease.release()
    lease.complete_execution()
    lease.release()
    assert events == ["finished", "released"]


def test_suspended_task_retains_global_worker_until_completion_but_not_result_consumption(admission):
    h = admission()
    capacity = LocalExecutionCapacity(max_slots=1)
    pool = h.pool(execution_capacity=capacity)
    first = h.authority(pool=pool)
    second = h.authority(pool=h.pool(execution_capacity=capacity))
    first.request(8)
    lease = h.take(first)
    with lease.suspend_for_wait(ExecutionCancellationScope("first", 1)):
        assert second.request(8)
        assert second.state()["state"] == "requested"
        assert h.runtime.snapshot()["ready_tasks"] == 0
        assert capacity.reserved_slots == 1
    assert h.runtime.snapshot()["running_tasks"] == 1
    lease.complete_execution()
    assert second.state()["available"]
    assert pool.active_lease_count == 1
    assert capacity.reserved_slots == 1
    lease.release()
    lease.complete_execution()
    assert capacity.reserved_slots == 1
    h.take(second).release()
    assert capacity.reserved_slots == 0


def test_task_without_runtime_limit_cannot_steal_reserved_global_worker(admission):
    h = admission()
    capacity = LocalExecutionCapacity(max_slots=1)
    first = h.authority(pool=h.pool(execution_capacity=capacity))
    external = h.pool(execution_capacity=capacity).create_authority()
    third = h.authority(pool=h.pool(execution_capacity=capacity))
    external_wakeup = threading.Event()
    external.register_wakeup(external_wakeup.set)

    first.request(8)  # Reserve before submission, including against unbounded queries.
    external.request(8)
    assert external.state()["state"] == "requested"
    lease = h.take(first)
    lease.complete_execution()
    assert external_wakeup.is_set()
    assert external.state()["available"]
    third.request(8)
    assert third.state()["state"] == "requested"
    assert h.runtime.snapshot()["ready_tasks"] == 0
    external.close()  # An unused grant must wake waiters in other pools.
    assert third.state()["available"]
    h.take(third).release()
    assert capacity.reserved_slots == 0


@pytest.mark.parametrize("close_pool", [False, True])
def test_closing_global_capacity_owner_returns_ready_but_retains_running_grants(admission, close_pool):
    h = admission(running=3)
    capacity = LocalExecutionCapacity(max_slots=2)
    pool = h.pool(slots=2, execution_capacity=capacity)
    owner = pool.create_authority()
    owner.request(8)
    running = h.take(owner)
    owner.request(8)
    pending = h.authority(pool=h.pool(slots=2, execution_capacity=capacity))
    pending.request(8)
    assert pending.state()["state"] == "requested"
    (pool if close_pool else owner).close()
    assert pending.state()["available"]
    assert capacity.reserved_slots == 2
    h.take(pending).release()
    assert capacity.reserved_slots == 1
    running.complete_execution()
    running.release()
    assert capacity.reserved_slots == 0


def test_global_capacity_release_failure_still_returns_runtime_allowance(admission):
    h = admission()
    capacity = LocalExecutionCapacity(max_slots=1)
    pool = h.pool(execution_capacity=capacity)
    first = h.authority(pool=pool)
    other = h.authority(pool=h.pool(execution_capacity=capacity))
    first.request(8)
    lease = h.take(first)
    other.request(8)
    observer = pool.create_authority()
    fail = False

    def wakeup():
        if fail:
            raise RuntimeError("planned capacity wakeup failure")

    observer.register_capacity_wakeup(wakeup)
    fail = True
    try:
        with pytest.raises(RuntimeError, match="planned capacity wakeup failure"):
            lease.complete_execution()
    finally:
        fail = False
    assert h.runtime.snapshot()["running_tasks"] == 0
    assert other.state()["available"]
    lease.release()
    h.take(other).release()
    assert capacity.reserved_slots == 0


def test_multiple_runtimes_acquire_pool_and_global_capacity_atomically(admission):
    capacity = LocalExecutionCapacity(max_slots=2)
    harnesses = [admission(running=4, queued=32) for _ in range(2)]
    pairs = [(h, h.authority(pool=h.pool(execution_capacity=capacity))) for h in harnesses for _ in range(8)]
    start = threading.Barrier(len(pairs))
    running = 0
    lock = threading.Lock()

    def execute(pair):
        nonlocal running
        h, authority = pair
        ready = threading.Event()
        authority.register_wakeup(ready.set)
        start.wait(timeout=5)
        authority.request(8)
        assert ready.wait(5)
        lease = h.take(authority)
        with lock:
            running += 1
            assert running <= 2
        assert capacity.reserved_slots <= 2
        time.sleep(0.01)
        with lock:
            running -= 1
        lease.complete_execution()
        lease.release()

    with ThreadPoolExecutor(max_workers=len(pairs)) as threads:
        list(threads.map(execute, pairs))
    assert capacity.reserved_slots == 0


@pytest.mark.parametrize("limited", [(False, False), (True, True), (False, True), (True, False)])
def test_busy_global_task_pool_cannot_starve_another_pool(admission, limited):
    capacity = LocalExecutionCapacity(max_slots=1)
    harnesses = [admission(), admission()]
    pools = [h.pool(slots=2, execution_capacity=capacity) for h in harnesses]
    # Select the old dispatcher's first pool so the regression also fails with
    # its unordered pool set, independently of allocation addresses.
    first_pool = next(iter(capacity._pools))
    pools.sort(key=lambda pool: pool is not first_pool)
    owners = [
        h.authority(pool=pool) if enabled else pool.create_authority()
        for h, pool, enabled in zip(harnesses, pools, limited, strict=True)
    ]
    a, b = owners
    try:
        a.request(8)
        current = harnesses[0].take(a)
        b.request(8)
        for _ in range(20):
            a.request(8)
            current.release()
            assert b.state()["available"], "older B request was bypassed by busy pool A"
            assert a.state()["state"] == "requested"
            current = harnesses[1].take(b)
            b.request(8)
            current.release()
            assert a.state()["available"]
            assert b.state()["state"] == "requested"
            current = harnesses[0].take(a)
        b.close()
        current.release()
        assert capacity.reserved_slots == 0
    finally:
        for owner in owners:
            owner.close()


@pytest.mark.parametrize("limited", [(False, False, False), (True, True, True), (False, True, False)])
def test_bulk_global_capacity_return_gives_each_pool_one_turn(admission, limited):
    h = admission(running=3)
    capacity = LocalExecutionCapacity(max_slots=3)
    blocker = h.pool(slots=3, execution_capacity=capacity)
    blockers = [blocker.create_authority() for _ in range(3)]
    for owner in blockers:
        owner.request(8)
    pools = [h.pool(slots=3, execution_capacity=capacity) for _ in range(3)]
    pools.sort(key=lambda pool: list(capacity._pools).index(pool))
    owners = [
        [h.authority(pool=pool) if enabled else pool.create_authority() for _ in range(3)]
        for pool, enabled in zip(pools, limited, strict=True)
    ]
    try:
        for group in owners:
            for owner in group:
                owner.request(8)
        blocker.close()
        assert capacity.reserved_slots == 3
        assert [sum(owner.state()["available"] for owner in group) for group in owners] == [1, 1, 1]
    finally:
        for group in owners:
            for owner in group:
                owner.close()
    assert capacity.reserved_slots == 0


@pytest.mark.parametrize("concurrent", [False, True])
@pytest.mark.parametrize("event", ["request", "actor_completion"])
def test_request_becoming_eligible_during_another_pools_turn_is_not_stranded(admission, concurrent, event):
    h = admission()
    capacity = LocalExecutionCapacity(max_slots=1)
    target = h.authority(pool=h.pool(execution_capacity=capacity))
    observer = h.pool(execution_capacity=capacity).create_authority()
    blocker = h.pool(execution_capacity=capacity).create_authority()
    blocker.request(8)
    held = None
    if event == "actor_completion":
        actor = h.authority()
        actor.request(8)
        held = h.take(actor)
        target.request(8)
    armed = False

    def make_eligible():
        if held is None:
            target.request(8)
        else:
            held.complete_execution()

    def publish():
        nonlocal armed
        if not armed:
            return
        armed = False
        if concurrent:
            with ThreadPoolExecutor(max_workers=1) as threads:
                threads.submit(make_eligible).result(timeout=5)
        else:
            make_eligible()
        # No grant may bypass the currently selected pool, even from another
        # thread. The dispatcher must revisit this request before it quiesces.
        assert target.state()["state"] == "requested"

    observer.register_capacity_wakeup(publish)
    armed = True
    blocker.close()
    assert target.state()["available"]
    h.take(target).release()
    if held is not None:
        held.release()
    observer.close()
    assert capacity.reserved_slots == 0


def test_new_request_cannot_bypass_global_dispatch_waiting_to_start(admission, monkeypatch):
    h = admission()
    capacity = LocalExecutionCapacity(max_slots=1)
    a = h.pool(slots=2, execution_capacity=capacity).create_authority()
    b = h.pool(slots=2, execution_capacity=capacity).create_authority()
    a.request(8)
    held = h.take(a)
    b.request(8)
    dispatch = capacity._dispatch
    entered, proceed = threading.Event(), threading.Event()

    def delayed_dispatch():
        entered.set()
        assert proceed.wait(5)
        dispatch()

    try:
        with ThreadPoolExecutor(max_workers=1) as threads:
            with monkeypatch.context() as patch:
                patch.setattr(capacity, "_dispatch", delayed_dispatch)
                completion = threads.submit(held.complete_execution)
                try:
                    assert entered.wait(5)
                    # Publish A in the interval between returning the global
                    # slot and entering the arbiter; B must keep the next turn.
                    a.request(8)
                    assert a.state()["state"] == "requested"
                finally:
                    proceed.set()
                completion.result(timeout=5)
        assert b.state()["available"]
        assert a.state()["state"] == "requested"
    finally:
        a.close()
        b.close()
        held.release()
    assert capacity.reserved_slots == 0


def _wait_until(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "admission did not progress"
        time.sleep(0.01)


def test_concurrent_transport_resumes_obey_execution_limit(admission):
    h = admission(running=2, queued=32)
    authorities = [h.authority() for _ in range(16)]
    parked = threading.Barrier(len(authorities))
    running = 0
    maximum = 0
    lock = threading.Lock()

    def execute(authority):
        nonlocal running, maximum
        ready = threading.Event()
        authority.register_wakeup(ready.set)
        authority.request(8)
        assert ready.wait(5)
        lease = authority.take(8)
        scope = ExecutionCancellationScope(lease.request_id, 1)
        try:
            with lease.suspend_for_wait(scope):
                parked.wait(timeout=5)
            with lock:
                running += 1
                maximum = max(maximum, running)
                assert running <= 2
            snapshot = h.runtime.snapshot()
            assert snapshot["ready_tasks"] + snapshot["running_tasks"] <= 2
            time.sleep(0.005)
            with lock:
                running -= 1
        finally:
            lease.complete_execution()
            lease.release()

    with ThreadPoolExecutor(max_workers=len(authorities)) as threads:
        list(threads.map(execute, authorities))
    assert 1 <= maximum <= 2
    snapshot = h.runtime.snapshot()
    assert snapshot["running_tasks"] == snapshot["waiting_tasks"] == snapshot["resuming_tasks"] == 0


@pytest.mark.parametrize("kind", ["input", "output"])
def test_memory_wait_yields_capacity_and_resumes_before_new_work_without_releasing_owners(admission, kind):
    h = admission()
    budget = LocalShmBudgetManager(limit_factory=lambda: 100)
    budget.acquire_allocation(70)
    producer, consumer, fresh = h.authority(), h.authority(), h.authority()
    producer.request(8)
    lease = h.take(producer)
    consumer.request(8)
    scope = ExecutionCancellationScope("producer", 1)
    claim = budget.acquire_allocation if kind == "input" else budget.request_output_grant
    release = budget.release_allocation if kind == "input" else budget.release_output_grant
    with ThreadPoolExecutor(max_workers=1) as threads:
        waiting = threads.submit(claim, 70, cancel_event=scope, wait_context=lambda: lease.suspend_for_wait(scope))
        try:
            _wait_until(lambda: consumer.state()["state"] == "ready")
            assert h.runtime.snapshot()["waiting_tasks"] == 1
            assert producer._capacity.active_lease_count == 1
            consumer_lease = h.take(consumer)
            budget.release_allocation(70)
            _wait_until(lambda: h.runtime.snapshot()["resuming_tasks"] == 1)
            assert budget.snapshot()["usage_bytes"] == 0
            assert not waiting.done()
            fresh.request(8)
            producer._query.shutdown()
            assert producer._query.cleanup_pending()
            consumer_lease.complete_execution()
            claimed = waiting.result(timeout=5)
            assert h.runtime.snapshot()["running_tasks"] == 1
            assert h.runtime.snapshot()["waiting_tasks"] == 0
            assert fresh.state()["state"] == "requested"
            assert budget.snapshot()["usage_bytes"] == 70
            release(claimed)
            lease.complete_execution()
            assert not producer._query.cleanup_pending()
            assert producer._capacity.active_lease_count == 1
            assert fresh.state()["state"] == "ready"
        finally:
            scope.cancel()
            budget.wake_waiters()
            budget.release_allocation(70)
            for owner in h.leases:
                owner.complete_execution()


@pytest.mark.parametrize("kind", ["input", "output"])
@pytest.mark.parametrize("phase", ["memory", "resume"])
def test_cancelled_memory_wait_retains_cleanup_owner_and_never_spends_bytes_or_capacity(admission, kind, phase):
    h = admission()
    budget = LocalShmBudgetManager(limit_factory=lambda: 100)
    budget.acquire_allocation(70)
    producer, consumer = h.authority(), h.authority()
    producer.request(8)
    lease = h.take(producer)
    consumer.request(8)
    scope = ExecutionCancellationScope("cancelled-producer", 1)
    unregister = scope.register_cancel_wakeup(budget.wake_waiters)
    claim = budget.acquire_allocation if kind == "input" else budget.request_output_grant
    with ThreadPoolExecutor(max_workers=1) as threads:
        waiting = threads.submit(claim, 70, cancel_event=scope, wait_context=lambda: lease.suspend_for_wait(scope))
        try:
            _wait_until(lambda: consumer.state()["state"] == "ready")
            consumer_lease = h.take(consumer)
            if phase == "resume":
                budget.release_allocation(70)
                _wait_until(lambda: h.runtime.snapshot()["resuming_tasks"] == 1)
            scope.cancel("test cancellation")
            with pytest.raises(RuntimeError, match="cancel"):
                waiting.result(timeout=5)
            assert budget.snapshot()["usage_bytes"] == (70 if phase == "memory" else 0)
            assert budget.snapshot()["waiting_output_grants"] == 0
            assert h.runtime.snapshot()["resuming_tasks"] == 0
            assert h.runtime.snapshot()["running_tasks"] == 1
            assert h.runtime.snapshot()["waiting_tasks"] == 1
            producer._query.shutdown(kill=True)
            assert producer._query.cleanup_pending()
            lease.complete_execution()
            assert not producer._query.cleanup_pending()
            assert h.runtime.snapshot()["waiting_tasks"] == 0
            assert producer._capacity.active_lease_count == 1
            consumer_lease.complete_execution()
        finally:
            scope.cancel()
            budget.wake_waiters()
            budget.release_allocation(70)
            for owner in h.leases:
                owner.complete_execution()
            unregister()


@pytest.mark.parametrize("kind", ["input", "output"])
def test_memory_available_during_resume_is_rechecked_before_reserving(admission, kind):
    h = admission()
    budget = LocalShmBudgetManager(limit_factory=lambda: 100)
    budget.acquire_allocation(70)
    producer, consumer, other = h.authority(), h.authority(), h.authority()
    producer.request(8)
    lease = h.take(producer)
    consumer.request(8)
    scope = ExecutionCancellationScope("racing-producer", 1)
    claim = budget.acquire_allocation if kind == "input" else budget.request_output_grant
    release = budget.release_allocation if kind == "input" else budget.release_output_grant
    with ThreadPoolExecutor(max_workers=1) as threads:
        waiting = threads.submit(claim, 70, cancel_event=scope, wait_context=lambda: lease.suspend_for_wait(scope))
        try:
            _wait_until(lambda: consumer.state()["state"] == "ready")
            consumer_lease = h.take(consumer)
            budget.release_allocation(70)
            _wait_until(lambda: h.runtime.snapshot()["resuming_tasks"] == 1)
            budget.acquire_allocation(70)
            other.request(8)
            consumer_lease.complete_execution()
            # The producer's resume sees that the bytes have been used again,
            # yields once more, and lets the next consumer release them.
            _wait_until(lambda: other.state()["state"] == "ready")
            assert not waiting.done()
            assert budget.snapshot()["usage_bytes"] == 70
            other_lease = h.take(other)
            budget.release_allocation(70)
            other_lease.complete_execution()
            claimed = waiting.result(timeout=5)
            assert budget.snapshot()["usage_bytes"] == 70
            release(claimed)
        finally:
            scope.cancel()
            budget.wake_waiters()
            budget.release_allocation(70)
            for owner in h.leases:
                owner.complete_execution()
