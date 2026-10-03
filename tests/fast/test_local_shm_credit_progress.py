# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest

from vane.execution.ref_bundle import LocalShmBudgetManager


@pytest.fixture
def shared_credits(request):
    manager = LocalShmBudgetManager(limit_factory=lambda: 1000)
    reservation = manager.reserve_task_bytes(10, 10) if getattr(request, "param", False) else None

    class SharedInput:
        name = "shared-image-batch"
        size = 400
        released = False

        def release_budget(self):
            if not self.released:
                self.released = True
                manager.release_allocation(self.size)

    source = SharedInput()
    manager.acquire_allocation(500, name="upstream-backlog")
    manager.acquire_allocation(source.size, name=source.name)
    leases = []
    grants = []
    threads = []
    errors = []
    cancel = threading.Event()
    try:
        # Successive compute batches consume slices of one upstream buffer.
        for _ in range(3):
            lease = manager.create_input_lease([source], source.size)
            leases.append(lease)
            manager.consume_input_lease(lease)
        assert manager.snapshot()["usage_bytes"] == 1700 + (20 if reservation is not None else 0)

        def request(index, *, size=350, priority="consumer"):
            def run():
                try:
                    grants.append(
                        manager.request_output_grant(
                            size, priority=priority, input_lease_id=leases[index], cancel_event=cancel
                        )
                    )
                except RuntimeError as error:
                    errors.append(error)

            thread = threading.Thread(target=run, daemon=True)
            threads.append(thread)
            thread.start()
            return thread

        yield manager, grants, errors, request
    finally:
        cancel.set()
        for lease in leases:
            manager.cancel_input_lease(lease)
        manager.release_allocation(500)
        for thread in threads:
            thread.join(timeout=2)
        for grant in grants:
            manager.release_output_grant(grant)
        if reservation is not None:
            reservation.release()
        assert all(not thread.is_alive() for thread in threads)
        assert manager.snapshot()["usage_bytes"] == 0


def test_shared_input_credits_allow_only_bounded_consumer_progress(shared_credits):
    manager, grants, errors, request = shared_credits
    first = request(0)
    first.join(timeout=2)
    assert not first.is_alive(), "consumer cannot drain an overcommitted input-credit backlog"
    assert not errors
    second = request(1)
    second.join(timeout=2)
    assert not second.is_alive()
    snapshot = manager.snapshot()
    assert snapshot["allocated_bytes"] + snapshot["output_grant_bytes"] == 1200
    assert snapshot["usage_bytes"] == 1600  # Existing credit was converted, not duplicated.

    third = request(2)
    third.join(timeout=0.1)
    assert third.is_alive(), "another output must not increase materialized-memory debt"
    # Moving a grant into a live allocation does not make capacity available.
    allocation = manager.convert_output_grant_to_allocation(grants.pop(0))
    try:
        third.join(timeout=0.1)
        assert third.is_alive()
    finally:
        manager.release_allocation(allocation)
    third.join(timeout=2)
    assert not third.is_alive()
    assert not errors
    snapshot = manager.snapshot()
    assert snapshot["output_credit_bytes"] == 0
    assert snapshot["usage_bytes"] == 1200


@pytest.mark.parametrize("priority,size", [("producer", 350), ("consumer", 450), ("consumer", 1100)])
def test_progress_exception_requires_consumer_credit_and_a_bounded_output(shared_credits, priority, size):
    manager, grants, errors, request = shared_credits
    thread = request(0, priority=priority, size=size)
    thread.join(timeout=0.1)
    assert thread.is_alive()
    assert grants == [] and errors == []
    assert manager.snapshot()["usage_bytes"] == 1700


@pytest.mark.parametrize("shared_credits", [True], indirect=True)
def test_credit_progress_does_not_overcommit_an_explicit_task_envelope(shared_credits):
    manager, grants, errors, request = shared_credits
    thread = request(0)
    thread.join(timeout=0.1)
    assert thread.is_alive()
    assert grants == [] and errors == []
    assert manager.snapshot()["task_reserved_bytes"] == 20
    assert manager.snapshot()["usage_bytes"] == 1720
