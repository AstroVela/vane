# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Materialized input publication shares the bounded arena and lease protocol."""

from __future__ import annotations

import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager, nullcontext

import pyarrow as pa
import pytest

from vane.execution import ref_bundle as refs
from vane.execution import udf_shm_store as storage
from vane.execution.udf_lifecycle import ExecutionCancellationScope, ExecutionCancelledError


@pytest.fixture
def arena(monkeypatch):
    store = storage.LocalShmStore(128 * 1024)
    store.add_client("materialized-test")
    manager = refs.LocalShmBudgetManager(limit_factory=lambda: 1024 * 1024)
    monkeypatch.setattr(storage, "_current_store", store)
    monkeypatch.setattr(refs, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    yield store, manager
    gc.collect()
    store.remove_client("materialized-test")
    assert store.snapshot()["live_allocations"] == 0
    assert manager.snapshot()["usage_bytes"] == 0


def publish(table, **kwargs):
    return refs.make_local_shm_ref_bundle_result(table, **kwargs)[1][0]


def test_materialized_input_writes_directly_and_reuses_only_released_regions(arena, monkeypatch):
    store, manager = arena

    def forbid_buffer(*args, **kwargs):
        pytest.fail("input publication allocated an intermediate IPC buffer")

    monkeypatch.setattr(pa, "BufferOutputStream", forbid_buffer)
    table = pa.table({"x": pa.chunked_array([[1, None], [3]]), "s": ["a", "b", "c"]})
    first = publish(table)
    slot = first._allocation_lease.allocation
    view = first.to_table()
    first.release()
    assert manager.snapshot()["allocated_bytes"] == slot.size
    second = publish(table)
    assert second._allocation_lease.allocation.shm_name == slot.shm_name
    assert second._allocation_lease.allocation.offset != slot.offset
    assert view.equals(table)
    del view
    gc.collect()
    third = publish(table)
    try:
        assert third._allocation_lease.allocation.offset == slot.offset
        assert third._allocation_lease.allocation.generation != slot.generation
        assert third.to_table().equals(table)
        assert store.snapshot()["reused_allocations"] >= 1
    finally:
        second.release()
        third.release()


def test_input_ack_returns_progress_while_retained_views_remain_charged(arena, monkeypatch):
    _, manager = arena
    monkeypatch.setattr(manager, "_limit_factory", lambda: 1)
    table = pa.table({"x": [b"x" * 1024]})
    first = publish(table, input_owner_id="actor")
    view = first.to_table()
    lease_id = manager.create_input_lease([first], first.size, owns_refs=True)
    manager.consume_input_lease(lease_id)
    assert manager.snapshot()["allocated_bytes"] == first.size
    assert manager._owned_input_allocations == {}
    scope = ExecutionCancellationScope("actor", 2)
    try:
        with ThreadPoolExecutor(max_workers=1) as threads:
            pending = threads.submit(publish, table, input_owner_id="actor", cancel_event=scope)
            try:
                second = pending.result(timeout=5)
                try:
                    assert manager.snapshot()["allocated_bytes"] == first.size + second.size
                    assert view.equals(table)
                finally:
                    second.release()
            finally:
                scope.cancel()
                manager.wake_waiters()
    finally:
        manager.cancel_input_lease(lease_id)
        first.release()
        del view


@pytest.mark.parametrize("cancel", [False, True])
def test_input_capacity_wait_returns_budget_and_wakes_on_release_or_cancel(arena, cancel):
    store, manager = arena
    occupied = store.allocate(96 * 1024)
    scope = ExecutionCancellationScope("source", 1)
    waiting = threading.Event()

    @contextmanager
    def wait_context():
        waiting.set()
        yield

    with ThreadPoolExecutor(max_workers=1) as threads:
        pending = threads.submit(
            publish,
            pa.table({"x": [b"x" * (64 * 1024)]}),
            cancel_event=scope,
            input_owner_id="source",
            wait_context=wait_context,
        )
        try:
            assert waiting.wait(5)
            assert manager.snapshot()["allocated_bytes"] == 0
            if cancel:
                scope.cancel("test cancellation")
                with pytest.raises(ExecutionCancelledError):
                    pending.result(timeout=5)
            else:
                with pytest.raises(FutureTimeoutError):
                    pending.result(timeout=0.05)
                occupied.release()
                pending.result(timeout=5).release()
        finally:
            scope.cancel()
            occupied.release()


@pytest.mark.parametrize("failure", ["write", "cancel", "adopt"])
def test_failed_input_publication_returns_all_owners(arena, monkeypatch, failure):
    store, manager = arena
    scope = ExecutionCancellationScope("source", 1)
    original = refs.PreparedLocalShmBlock.write_to

    def write(block, region):
        original(block, region)
        if failure == "write":
            raise ValueError("input copy failed")
        if failure == "cancel":
            scope.cancel("cancel during copy")

    def adopt(*args, **kwargs):
        raise ValueError("input adoption failed")

    monkeypatch.setattr(refs.PreparedLocalShmBlock, "write_to", write)
    if failure == "adopt":
        monkeypatch.setattr(refs, "LocalShmBlockRef", adopt)
    with pytest.raises((ValueError, ExecutionCancelledError)) as error:
        publish(pa.table({"x": [1, 2]}), cancel_event=scope, input_owner_id="source")
    # Keep the traceback alive while checking that its mapping can be reused.
    assert error.value is not None
    assert store.snapshot()["live_allocations"] == 0
    assert manager.snapshot()["usage_bytes"] == 0
    assert manager._owned_input_allocations == {}
    check = store.allocate(store.capacity)
    check.release()


def test_empty_views_preserve_dependency_on_queued_materialized_input(arena):
    store, manager = arena
    current = publish(pa.table({"x": pa.nulls(1)}), input_owner_id="first-query", input_producer_id="replica")
    lease_id = manager.create_input_lease([current], current.size, owns_refs=True)
    manager.consume_input_lease(lease_id)
    queued = publish(pa.table({"x": [b"x" * (64 * 1024)]}), input_owner_id="second-query", input_producer_id="replica")
    try:
        queued_bytes = store.check_write_size(queued.size)
        with pytest.raises(storage.LocalShmStoreCapacityError, match=f"queued_input_bytes={queued_bytes}"):
            store.wait_for_capacity(
                64 * 1024,
                (),
                ExecutionCancellationScope("consumer", 1),
                nullcontext,
                manager.consuming_input_producers(lease_id),
            )
    finally:
        manager.cancel_input_lease(lease_id)
        current.release()
        queued.release()


def test_prefetched_input_does_not_take_another_replicas_progress_slot(arena, monkeypatch):
    _, manager = arena
    monkeypatch.setattr(manager, "_limit_factory", lambda: 1)
    table = pa.table({"x": [1]})
    queued = publish(table, input_owner_id="operator", input_producer_id="replica-0")
    scope = ExecutionCancellationScope("operator", 2)
    try:
        with ThreadPoolExecutor(max_workers=1) as threads:
            pending = threads.submit(
                publish, table, input_owner_id="operator", input_producer_id="replica-1", cancel_event=scope
            )
            try:
                # Replica 0 may be occupied by a prior call. Its prefetched
                # input cannot prevent an independent idle replica from running.
                pending.result(timeout=2).release()
            finally:
                scope.cancel()
                manager.wake_waiters()
    finally:
        queued.release()
