# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Released inputs cannot become phantom output reservations."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from vane.execution.ref_bundle import LocalShmBudgetManager


def test_repeated_input_slices_do_not_multiply_reserved_bytes():
    manager = LocalShmBudgetManager(limit_factory=lambda: 1000)

    class SharedInput:
        name = "shared-input"
        size = 400
        released = False

        def release_budget(self):
            if not self.released:
                self.released = True
                manager.release_allocation(self.size)

    source = SharedInput()
    manager.acquire_allocation(500)
    manager.acquire_allocation(source.size)
    leases = []
    for _ in range(10):
        lease = manager.create_input_lease([source], source.size)
        leases.append(lease)
        manager.consume_input_lease(lease)
        assert manager.snapshot()["usage_bytes"] == 500
    grant = manager.request_output_grant(700, priority="consumer", input_lease_id=leases[-1])
    assert manager.snapshot()["usage_bytes"] == 1200
    manager.release_output_grant(grant)
    for lease in leases:
        manager.cancel_input_lease(lease)
    manager.release_allocation(500)
    assert manager.snapshot()["usage_bytes"] == 0


def test_consumer_escape_admits_one_complete_block_then_waits_for_release():
    manager = LocalShmBudgetManager(limit_factory=lambda: 1000)
    manager.acquire_allocation(900)
    first = manager.request_output_grant(350, priority="consumer")
    cancel = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as threads:
        waiting = threads.submit(manager.request_output_grant, 350, priority="consumer", cancel_event=cancel)
        try:
            with pytest.raises(TimeoutError):
                waiting.result(timeout=0.1)
            allocation = manager.convert_output_grant_to_allocation(first)
            with pytest.raises(TimeoutError):
                waiting.result(timeout=0.1)
            manager.release_allocation(allocation)
            second = waiting.result(timeout=5)
            manager.release_output_grant(second)
        finally:
            cancel.set()
            manager.wake_waiters()
            manager.release_output_grant(first)
            manager.release_allocation(900)
    assert manager.snapshot()["usage_bytes"] == 0


@pytest.mark.parametrize("strict", [False, True])
def test_producer_and_explicit_reservations_do_not_use_consumer_escape(strict):
    manager = LocalShmBudgetManager(limit_factory=lambda: 1000)
    reservation = manager.reserve_task_bytes(10, 10) if strict else None
    manager.acquire_allocation(900)
    cancel = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as threads:
        pending = threads.submit(
            manager.request_output_grant, 350, priority="consumer" if strict else "producer", cancel_event=cancel
        )
        try:
            with pytest.raises(TimeoutError):
                pending.result(timeout=0.1)
            assert manager.snapshot()["usage_bytes"] == (920 if strict else 900)
        finally:
            cancel.set()
            manager.wake_waiters()
            with pytest.raises(RuntimeError, match="cancelled"):
                pending.result(timeout=5)
            if reservation is not None:
                reservation.release()
            manager.release_allocation(900)
    assert manager.snapshot()["usage_bytes"] == 0


def test_pinned_consumer_can_progress_past_unrelated_inputs_once(monkeypatch):
    from vane.execution import ref_bundle as refs
    from vane.execution.udf_shm_store import LocalShmStore

    manager = LocalShmBudgetManager(limit_factory=lambda: 1000)
    monkeypatch.setattr(refs, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    store = LocalShmStore(8192)
    store.add_client("consumer-progress")

    def block(size, budget):
        lease = store.allocate(size)
        return refs.LocalShmBlockRef(lease.allocation.identity, size, allocation_lease=lease, budget_bytes=budget)

    inputs = [block(1024, manager.acquire_allocation(1024, block=False)) for _ in range(2)]
    lease = manager.create_input_lease([inputs[0]], 1024)
    manager.consume_input_lease(lease)
    output = None
    cancel = threading.Event()
    try:
        # Both upstream inputs remain charged. An unrelated input must not
        # prevent this consumer from producing the output that drains them.
        grant = manager.request_output_grant(512, priority="consumer", input_lease_id=lease)
        output = block(512, manager.convert_output_grant_to_allocation(grant))
        manager.record_consuming_output(lease, [output])
        with ThreadPoolExecutor(max_workers=1) as threads:
            waiting = threads.submit(
                manager.request_output_grant, 512, priority="consumer", input_lease_id=lease, cancel_event=cancel
            )
            try:
                with pytest.raises(TimeoutError):
                    waiting.result(timeout=0.1)
                output.release()
                next_grant = waiting.result(timeout=5)
                manager.release_output_grant(next_grant)
            finally:
                cancel.set()
                manager.wake_waiters()
    finally:
        if output is not None:
            output.release()
        manager.cancel_input_lease(lease)
        for ref in inputs:
            ref.release()
        store.remove_client("consumer-progress")
    assert manager.snapshot()["usage_bytes"] == 0
