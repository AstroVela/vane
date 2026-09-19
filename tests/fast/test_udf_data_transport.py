# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
from multiprocessing import shared_memory

import pyarrow as pa
import pytest

from vane.execution import ref_bundle
from vane.execution.udf_data_lease import DataAllocation, RuntimeDataLedger


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
