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


@pytest.mark.parametrize("release_method", ["release", "release_budget"])
@pytest.mark.parametrize("finish_method", ["consume_input_lease", "cancel_input_lease"])
@pytest.mark.parametrize("retry_first", [False, True])
def test_failed_shared_input_hold_transfers_only_unreleased_owners(release_method, finish_method, retry_first):
    budget = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)

    class Ref:
        name, size = "shared-retry-input", 400

        def __init__(self, fail=False):
            self.calls, self.fail = 0, fail
            setattr(self, release_method, self.cleanup)

        def cleanup(self):
            self.calls += 1
            if self.fail:
                raise RuntimeError("planned shared input release failure")

    failed, succeeded, later, newest = Ref(fail=True), Ref(), Ref(), Ref()
    first = budget.create_input_lease([failed, succeeded], 400, reserve_output_credit=False)
    finish = getattr(budget, finish_method)
    with pytest.raises(RuntimeError, match="planned shared input release failure"):
        finish(first)
    assert (failed.calls, succeeded.calls) == (1, 1)
    assert budget.snapshot()["active_input_ref_holds"] == 1
    second = budget.create_input_lease([later], 400, reserve_output_credit=False)
    failed.fail = False
    if retry_first:
        assert finish(first) == 400
        assert (failed.calls, succeeded.calls, later.calls) == (1, 1, 0)
    assert finish(second) == 400
    assert (failed.calls, succeeded.calls, later.calls) == (2, 1, 1)
    # An old retry must neither repeat completed releases nor erase a newer
    # generation's hold when the shared-memory name is reused.
    third = budget.create_input_lease([newest], 400, reserve_output_credit=False)
    assert finish(first) == (0 if retry_first else 400)
    assert newest.calls == 0
    assert budget.snapshot()["active_input_ref_hold_count"] == 1
    assert finish(third) == 400
    assert (failed.calls, succeeded.calls, later.calls, newest.calls) == (2, 1, 1, 1)
    assert budget.snapshot()["active_input_ref_holds"] == budget.snapshot()["active_input_leases"] == 0


@pytest.mark.parametrize("fails", [False, True])
def test_new_input_borrow_is_atomic_while_owner_release_is_running(fails):
    budget = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    entered, proceed = threading.Event(), threading.Event()

    class Ref:
        size = 400

        def __init__(self, name):
            self.name, self.calls = name, 0

        def release(self):
            self.calls += 1
            if self is owner and self.calls == 1:
                # A reentrant borrow must fail without waiting for itself.
                with pytest.raises(RuntimeError, match="cleanup is still in progress"):
                    budget.create_input_lease([alias], 400)
                entered.set()
                assert proceed.wait(timeout=5)
                if fails:
                    raise RuntimeError("planned blocked input release failure")

    owner, alias, unrelated = Ref("shared"), Ref("shared"), Ref("other")
    first = budget.create_input_lease([owner], 400, reserve_output_credit=False)
    with ThreadPoolExecutor(max_workers=1) as threads:
        cleanup = threads.submit(budget.cancel_input_lease, first)
        try:
            assert entered.wait(timeout=5)
            with pytest.raises(RuntimeError, match="cleanup is still in progress"):
                budget.create_input_lease([unrelated, alias], 800)
            snapshot = budget.snapshot()
            assert snapshot["active_input_leases"] == snapshot["active_input_ref_holds"] == 1
            assert snapshot["active_input_ref_hold_count"] == 0
            assert snapshot["input_lease_bytes"] == 400
        finally:
            proceed.set()
        if fails:
            with pytest.raises(RuntimeError, match="planned blocked input release failure"):
                cleanup.result(timeout=5)
            # Once the failed release returns, later borrowing is safe again.
            second = budget.create_input_lease([alias], 400, reserve_output_credit=False)
            assert budget.cancel_input_lease(first) == 400
            assert owner.calls == 1
            assert budget.cancel_input_lease(second) == 400
            assert (owner.calls, alias.calls) == (2, 1)
        else:
            assert cleanup.result(timeout=5) == 400
            assert (owner.calls, alias.calls) == (1, 0)
    assert unrelated.calls == 0
    assert budget.snapshot()["active_input_leases"] == budget.snapshot()["active_input_ref_holds"] == 0


@pytest.mark.parametrize("fails", [False, True])
def test_shared_cleanup_retries_preserve_the_running_release_owner(fails):
    budget = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    entered, proceed = threading.Event(), threading.Event()

    class Ref:
        name, size = "shared", 400
        calls = 0

        def release(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("planned initial release failure")
            if self.calls == 2:
                entered.set()
                assert proceed.wait(timeout=5)
                if fails:
                    raise RuntimeError("planned second release failure")

    owner = Ref()
    first = budget.create_input_lease([owner], 400, reserve_output_credit=False)
    with pytest.raises(RuntimeError, match="planned initial release failure"):
        budget.cancel_input_lease(first)
    second = budget.create_input_lease([owner], 400, reserve_output_credit=False)
    with ThreadPoolExecutor(max_workers=1) as threads:
        cleanup = threads.submit(budget.cancel_input_lease, second)
        try:
            assert entered.wait(timeout=5)
            assert budget.cancel_input_lease(first) is None
            assert budget.input_lease_pending(first)
            assert budget.snapshot()["input_lease_bytes"] == 800
            assert owner.calls == 2
        finally:
            proceed.set()
        if fails:
            with pytest.raises(RuntimeError, match="planned second release failure"):
                cleanup.result(timeout=5)
        else:
            assert cleanup.result(timeout=5) == 400
    assert budget.cancel_input_lease(first) == 400
    assert budget.cancel_input_lease(second) == (400 if fails else 0)
    assert owner.calls == (3 if fails else 2)
    assert budget.snapshot()["active_input_leases"] == budget.snapshot()["active_input_ref_holds"] == 0


def test_input_release_rechecks_later_holds_after_reentrant_borrow():
    budget = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    later_leases = []

    class Ref:
        size = 400

        def __init__(self, name):
            self.name, self.calls = name, 0

        def release(self):
            self.calls += 1
            if self is first:
                later_leases.append(budget.create_input_lease([second], 400, reserve_output_credit=False))

    first, second = Ref("first"), Ref("second")
    original = budget.create_input_lease([first, second], 800, reserve_output_credit=False)
    assert budget.cancel_input_lease(original) == 800
    assert (first.calls, second.calls) == (1, 0)
    assert budget.snapshot()["input_lease_bytes"] == 400
    assert budget.cancel_input_lease(later_leases[0]) == 400
    assert (first.calls, second.calls) == (1, 1)
    assert budget.snapshot()["active_input_leases"] == budget.snapshot()["active_input_ref_holds"] == 0


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


@pytest.mark.parametrize("limited", [False, True])
def test_input_accounting_matches_metadata_fallback_and_descriptor_precedence(monkeypatch, limited):
    budget = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", budget)
    ledger = RuntimeDataLedger(DataAdmissionLimits(8192, 2048, 2048) if limited else None)
    query = ledger.open_query()
    task = query.open_task(query.reserve_task() if limited else None)
    first = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": [1, 2]}))
    second = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": [3]}))
    refs = [first[1][0], second[2][0], object()]
    metadata = [second[2][0], first[2][0], second[2][0]]
    try:
        payload = ref_bundle.make_local_ref_bundle_worker_payload(refs, metadata=metadata)
        assert [desc["shm_name"] for desc in payload["block_refs"]] == [
            first[1][0].name,
            second[1][0].name,
            second[1][0].name,
        ]
        ref_bundle.track_local_shm_inputs(task, refs, metadata)
        data = ledger.snapshot()
        assert data["input_bytes"] == first[1][0].size + second[1][0].size
        assert data["allocations"] == data["leases"] == 2
    finally:
        task.finish()
        query.shutdown()
        first[1][0].release()
        second[1][0].release()
        ledger.close()
    assert budget.snapshot()["usage_bytes"] == ledger.snapshot()["retained_bytes"] == 0


@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("invalid", ["length", "missing"])
def test_invalid_input_metadata_does_not_publish_partial_accounting(monkeypatch, limited, invalid):
    budget = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", budget)
    ledger = RuntimeDataLedger(DataAdmissionLimits(8192, 2048, 2048) if limited else None)
    query = ledger.open_query()
    task = query.open_task(query.reserve_task() if limited else None)
    original = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": [1]}))
    metadata = original[2] if invalid == "length" else [original[2][0], {}]
    message = (
        "ref/metadata length mismatch" if invalid == "length" else "requires local shared-memory input descriptors"
    )
    before = ledger.snapshot()
    try:
        with pytest.raises(ValueError, match=message):
            ref_bundle.track_local_shm_inputs(task, [object(), object()], metadata)
        assert ledger.snapshot() == before
    finally:
        task.finish()
        query.shutdown()
        original[1][0].release()
        ledger.close()
    assert budget.snapshot()["usage_bytes"] == ledger.snapshot()["retained_bytes"] == 0


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
