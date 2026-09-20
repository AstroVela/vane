# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory

import pyarrow as pa
import pytest

from vane.execution import ref_bundle
from vane.execution.udf_data_admission import DataAdmissionLimits
from vane.execution.udf_data_lease import DataAllocation, RuntimeDataLedger


@pytest.mark.parametrize("limited", [False, True])
def test_failed_input_cleanup_retries_partial_release_without_touching_other_queries(monkeypatch, limited):
    gc.collect()
    budget = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", budget)
    ledger = RuntimeDataLedger(DataAdmissionLimits(12288, 2048, 2048) if limited else None)
    first, other = ledger.open_query(), ledger.open_query()
    task = first.open_task(first.reserve_task() if limited else None)
    other_task = other.open_task(other.reserve_task() if limited else None)
    refs = [ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": [i]}))[1][0] for i in range(3)]
    ref_bundle.track_local_shm_inputs(task, refs[:2])
    ref_bundle.track_local_shm_inputs(other_task, refs[2:])
    first_lease = ref_bundle.create_local_shm_input_lease(refs[:2], reserve_output_credit=False)
    other_lease = ref_bundle.create_local_shm_input_lease(refs[2:], reserve_output_credit=False)
    task.hold_input_transport(budget, first_lease)
    other_task.hold_input_transport(budget, other_lease)
    release = budget._release_input_ack_ref
    calls = []

    def fail_one(ref):
        calls.append(ref.name)
        if ref is refs[0]:
            raise RuntimeError("planned partial input cleanup failure")
        release(ref)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(budget, "_release_input_ack_ref", fail_one)
            with pytest.raises(RuntimeError, match="planned partial input cleanup failure"):
                task.finish()
            assert calls == [refs[0].name, refs[1].name]
            assert budget.snapshot()["allocated_bytes"] == refs[0].size + refs[2].size
            assert ledger.snapshot()["input_bytes"] == sum(ref.size for ref in refs)
            task_ref = weakref.ref(task)
            del task
            gc.collect()
            assert task_ref() is not None  # The query owns retry, not the failed request.
            with pytest.raises(RuntimeError, match="planned partial input cleanup failure"):
                first.shutdown()
            assert calls == [refs[0].name, refs[1].name, refs[0].name]
        first.shutdown()
        gc.collect()
        assert task_ref() is None
        assert budget.input_lease_pending(other_lease)
        assert budget.snapshot()["input_lease_bytes"] == ledger.snapshot()["input_bytes"] == refs[2].size
        assert ledger.snapshot()["tasks"] == ledger.snapshot()["queries"] == 1
    finally:
        first.shutdown()
        other_task.finish()
        other.shutdown()
        for ref in refs:
            ref.release()
        ledger.close()
    assert budget.snapshot()["usage_bytes"] == ledger.snapshot()["retained_bytes"] == 0


def test_concurrent_transport_cleanup_keeps_input_accounted_until_release_finishes(monkeypatch):
    budget = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", budget)
    ledger = RuntimeDataLedger(DataAdmissionLimits(4096, 2048, 2048))
    query = ledger.open_query()
    task = query.open_task(query.reserve_task())
    ref = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": [1]}))[1][0]
    ref_bundle.track_local_shm_inputs(task, [ref])
    lease = ref_bundle.create_local_shm_input_lease([ref], reserve_output_credit=False)
    task.hold_input_transport(budget, lease)
    entered, proceed = threading.Event(), threading.Event()
    release = budget._release_input_ack_ref

    def blocked_release(ref):
        entered.set()
        assert proceed.wait(timeout=5)
        release(ref)

    try:
        with monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=1) as threads:
            patch.setattr(budget, "_release_input_ack_ref", blocked_release)
            cleanup = threads.submit(budget.consume_input_lease, lease)
            try:
                assert entered.wait(timeout=5)
                with pytest.raises(RuntimeError, match="cleanup is still in progress"):
                    task.finish()
                assert ledger.snapshot()["input_bytes"] == ref.size
                with pytest.raises(TimeoutError):
                    ledger.close()
            finally:
                proceed.set()
            cleanup.result(timeout=5)
        query.shutdown()
        ledger.close()
        assert ledger.snapshot()["usage_bytes"] == 0
    finally:
        proceed.set()
        task.finish()
        query.shutdown()
        ref.release()
        ledger.close()


@pytest.mark.parametrize("first", ["consume_input_lease", "cancel_input_lease"])
@pytest.mark.parametrize("second", ["consume_input_lease", "cancel_input_lease"])
@pytest.mark.parametrize("fails", [False, True])
def test_overlapping_input_releases_keep_cancellation_terminal(monkeypatch, first, second, fails):
    budget = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", budget)
    ref = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": [1]}))[1][0]
    lease = ref_bundle.create_local_shm_input_lease([ref])
    entered, proceed = threading.Event(), threading.Event()
    release = budget._release_input_ack_ref
    calls = []

    def blocked_release(ref):
        calls.append(ref.name)
        entered.set()
        assert proceed.wait(timeout=5)
        if fails:
            raise RuntimeError("planned concurrent input release failure")
        release(ref)

    expected_credit = ref.size if first == second == "consume_input_lease" else 0
    try:
        with monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=1) as threads:
            patch.setattr(budget, "_release_input_ack_ref", blocked_release)
            cleanup = threads.submit(getattr(budget, first), lease)
            try:
                assert entered.wait(timeout=5)
                assert getattr(budget, second)(lease) is None
                assert budget.input_lease_pending(lease)
                assert budget.snapshot()["input_lease_bytes"] == ref.size
                assert budget.snapshot()["output_credit_bytes"] == expected_credit
                assert calls == [ref.name]
            finally:
                proceed.set()
            if fails:
                with pytest.raises(RuntimeError, match="planned concurrent input release failure"):
                    cleanup.result(timeout=5)
            else:
                assert cleanup.result(timeout=5) == ref.size
        assert budget.input_lease_pending(lease) == fails
        if fails:
            # Even a late ACK must not revive a cancelled lease's credit.
            assert budget.consume_input_lease(lease) == ref.size
        assert not budget.input_lease_pending(lease)
        snapshot = budget.snapshot()
        assert snapshot["input_lease_bytes"] == snapshot["allocated_bytes"] == 0
        assert snapshot["output_credit_bytes"] == expected_credit
    finally:
        proceed.set()
        budget.cancel_input_lease(lease)
        ref.release()
    assert budget.snapshot()["usage_bytes"] == 0


def test_reentrant_input_cancellation_revokes_credit_without_waiting(monkeypatch):
    budget = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", budget)
    ref = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": [1]}))[1][0]
    lease = ref_bundle.create_local_shm_input_lease([ref])
    pending = []
    with ThreadPoolExecutor(max_workers=1) as threads:

        def cancel_on_release():
            if budget.input_lease_pending(lease):
                pending.append(budget.cancel_input_lease(lease))
                # Notifications must not hold the manager lock on this thread.
                assert threads.submit(budget.snapshot).result(timeout=5)["output_credit_bytes"] == 0

        unregister = ref_bundle.register_local_shm_ref_budget_wakeup(cancel_on_release)
        try:
            assert budget.consume_input_lease(lease) == ref.size
            assert pending and all(result is None for result in pending)
            assert not budget.input_lease_pending(lease)
            assert budget.snapshot()["usage_bytes"] == 0
        finally:
            unregister()
            budget.cancel_input_lease(lease)
            ref.release()


@pytest.fixture
def tracked_transport(monkeypatch):
    budget = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", budget)
    ledger = RuntimeDataLedger()
    query = ledger.open_query()
    task = query.open_task()
    result = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": list(range(128))}))
    ref_bundle.track_local_shm_output(task, result)
    try:
        yield ledger, query, task, result, budget
    finally:
        for ref in result[1]:
            ref.release()
        task.finish()
        query.shutdown()
        ledger.close()


@pytest.mark.parametrize("view_kind", ["table", "slice", "array", "buffer", "numpy"])
def test_zero_copy_consumers_keep_data_accounted_after_query_and_runtime_close(tracked_transport, view_kind):
    ledger, query, task, result, budget = tracked_transport
    ref = result[1][0]
    size = ref.size
    table = ref.to_table()
    if view_kind == "table":
        view = table.select([0]).rename_columns(["renamed"])
    elif view_kind == "slice":
        view = table.slice(3, 2)
    elif view_kind == "array":
        view = table.column(0).chunk(0).slice(3, 2)
    elif view_kind == "buffer":
        view = table.column(0).chunk(0).buffers()[1]
    else:
        view = table.column(0).chunk(0).to_numpy(zero_copy_only=True)
    del table
    ref_bundle.transition_local_shm_output(result, "unit_queue")
    task.finish()
    query.shutdown()
    ledger.close()
    ref.release()
    gc.collect()
    assert budget.snapshot()["usage_bytes"] == 0
    assert ledger.snapshot()["retained_bytes"] == size
    assert ledger.snapshot()["output_state_bytes"]["external_consumer"] == size
    if view_kind in {"table", "slice"}:
        assert view.column(0).to_pylist() == (list(range(128)) if view_kind == "table" else [3, 4])
    elif view_kind == "array":
        assert view.to_pylist() == [3, 4]
    elif view_kind == "numpy":
        assert view.tolist() == list(range(128))
    else:
        assert len(view) == 128 * 8
    del view
    gc.collect()
    assert ledger.snapshot()["retained_bytes"] == ledger.snapshot()["leases"] == 0


def test_input_ack_does_not_release_data_borrows_or_double_count_shared_output(tracked_transport):
    ledger, _, producer, result, budget = tracked_transport
    ref = result[1][0]
    size = ref.size
    producer.finish()
    consumer_query = ledger.open_query()
    consumer = consumer_query.open_task()
    ref_bundle.track_local_shm_inputs(consumer, [ref, ref])
    assert ledger.snapshot()["retained_bytes"] == size
    assert ledger.snapshot()["input_bytes"] == ledger.snapshot()["output_bytes"] == size
    transport_lease = ref_bundle.create_local_shm_input_lease([ref], reserve_output_credit=False)
    try:
        ref_bundle.consume_local_shm_input_lease(transport_lease)
        assert budget.snapshot()["allocated_bytes"] == 0
        assert ledger.snapshot()["input_bytes"] == size
        ref.release()
        assert ledger.snapshot()["output_bytes"] == 0
        assert ledger.snapshot()["retained_bytes"] == size
    finally:
        ref_bundle.cancel_local_shm_input_lease(transport_lease)
        consumer.finish()
        consumer_query.shutdown()
    assert ledger.snapshot()["retained_bytes"] == 0


@pytest.mark.parametrize("failure", ["open", "decode"])
def test_failed_materialization_returns_only_the_new_view_lease(tracked_transport, monkeypatch, failure):
    ledger, _, _, result, _ = tracked_transport
    ref = result[1][0]

    def fail(*_args, **_kwargs):
        raise RuntimeError("planned materialization failure")

    with monkeypatch.context() as patch:
        patch.setattr(ref_bundle, "_open_existing_shm" if failure == "open" else "_ipc_payload_bounds", fail)
        with pytest.raises(RuntimeError, match="planned materialization failure"):
            ref.to_table()
    assert ledger.snapshot()["retained_bytes"] == ref.size
    assert ledger.snapshot()["leases"] == 1
    table = ref.to_table()
    assert table.num_rows == 128
    del table
    gc.collect()
    assert ledger.snapshot()["leases"] == 1


def test_output_tracking_rolls_back_new_owner_when_ref_is_already_owned(tracked_transport):
    ledger, _, task, result, _ = tracked_transport
    before = ledger.snapshot()
    with pytest.raises(RuntimeError, match="already has a data lease"):
        ref_bundle.track_local_shm_output(task, result)
    assert ledger.snapshot() == before


def test_deferred_mapping_close_keeps_data_owner_until_retry_succeeds(monkeypatch):
    ledger = RuntimeDataLedger()
    query = ledger.open_query()
    task = query.open_task()
    monkeypatch.setattr(ref_bundle, "_deferred_shm_closes", [])
    mapping = shared_memory.SharedMemory(create=True, size=64)
    lease = task.own_output(DataAllocation("local_shm", mapping.name, 64))
    exported = mapping.buf[:]
    owner = ref_bundle._LocalShmBufferOwner(mapping, lease)
    task.finish()
    query.shutdown()
    ledger.close()
    try:
        owner.close()
        owner.close()
        assert ledger.snapshot()["retained_bytes"] == 64
        ref_bundle._retry_deferred_shm_closes()
        assert ledger.snapshot()["retained_bytes"] == 64
        assert len(ref_bundle._deferred_shm_closes) == 1
        exported.release()
        ref_bundle._retry_deferred_shm_closes()
        assert ledger.snapshot()["retained_bytes"] == 0
        assert not ref_bundle._deferred_shm_closes
    finally:
        exported.release()
        ref_bundle._retry_deferred_shm_closes()
        mapping.close()
        mapping.unlink()
        lease.release()


def test_untracked_ref_materialization_does_not_create_runtime_owners():
    result = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": [1, 2]}))
    ref = result[1][0]
    try:
        table = ref.to_table()
        ref.release()
        assert table.to_pydict() == {"x": [1, 2]}
        assert ref._data_lease is None
    finally:
        ref.release()
