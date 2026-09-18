# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from vane.execution.udf_admission import AdmissionLease, LocalExecutionSlotPool
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

    def pool(self, slots=1):
        pool = LocalExecutionSlotPool(max_slots=slots, execution_slot_prefix=f"model:{len(self.pools)}")
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
