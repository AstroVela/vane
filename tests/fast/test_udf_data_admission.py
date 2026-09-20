# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from vane.execution import ref_bundle
from vane.execution.byte_budget import ByteBudgetUsage, byte_budget_block_reason
from vane.execution.udf_admission import LocalExecutionSlotPool
from vane.execution.udf_data_admission import (
    DataAdmissionAuthority,
    DataAdmissionCapacityError,
    DataAdmissionLimits,
    DataBatchTooLarge,
)
from vane.execution.udf_data_lease import DataAllocation, RuntimeDataLedger
from vane.execution.udf_runtime_admission import RuntimeTaskAdmission, TaskAdmissionLimits


@pytest.fixture
def transport(monkeypatch):
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 10_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    yield manager
    assert manager.snapshot()["task_reserved_bytes"] == 0


@pytest.mark.parametrize("field", ["max_bytes", "max_task_input_bytes", "max_task_output_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_data_limits_require_positive_integer_bytes(field, value):
    args = dict(max_bytes=1000, max_task_input_bytes=100, max_task_output_bytes=200)
    args[field] = value
    with pytest.raises(ValueError, match="positive integer"):
        DataAdmissionLimits(**args)


def test_limit_must_fit_complete_task_envelope():
    with pytest.raises(ValueError, match="input and output"):
        DataAdmissionLimits(299, 100, 200)


def test_concurrent_reservations_are_atomic_and_capacity_refusal_is_retryable(transport):
    ledger = RuntimeDataLedger(DataAdmissionLimits(1200, 100, 200))
    queries = [ledger.open_query() for _ in range(12)]
    barrier = threading.Barrier(len(queries))

    def acquire(query):
        barrier.wait(timeout=5)
        try:
            return query.reserve_task()
        except DataAdmissionCapacityError as error:
            assert error.owner == "runtime"
            return None

    with ThreadPoolExecutor(max_workers=len(queries)) as threads:
        reservations = list(threads.map(acquire, queries))
    winners = [r for r in reservations if r is not None]
    assert len(winners) == 4
    assert ledger.snapshot()["usage_bytes"] == transport.snapshot()["usage_bytes"] == 1200
    winners.pop().release()
    retry = queries[reservations.index(None)].reserve_task()
    assert ledger.snapshot()["usage_bytes"] == 1200
    retry.release()
    for reservation in winners:
        reservation.release()
        reservation.release()
    for query in queries:
        query.shutdown()
    ledger.close()
    assert ledger.snapshot()["usage_bytes"] == transport.snapshot()["usage_bytes"] == 0


def test_conversion_deduplicates_shared_inputs_and_retains_output_after_close(transport):
    ledger = RuntimeDataLedger(DataAdmissionLimits(1000, 200, 200))
    producer = ledger.open_query()
    task = producer.open_task(producer.reserve_task())
    allocation = DataAllocation("local_shm", "shared", 200)
    output = task.own_output(allocation)
    task.finish()
    consumer = ledger.open_query()
    consuming = consumer.open_task(consumer.reserve_task())
    consuming.hold_inputs([allocation, allocation])
    snapshot = ledger.snapshot()
    assert snapshot["retained_bytes"] == snapshot["input_bytes"] == snapshot["output_bytes"] == 200
    assert snapshot["usage_bytes"] == 600
    output.release()
    assert ledger.snapshot()["retained_bytes"] == 200
    result = consuming.own_output(DataAllocation("local_shm", "result", 150))
    assert ledger.snapshot()["usage_bytes"] == 600
    consuming.finish()
    producer.shutdown()
    consumer.shutdown()
    ledger.close()
    assert ledger.snapshot()["usage_bytes"] == 150
    fork = result.fork()
    result.release()
    assert ledger.snapshot()["usage_bytes"] == 150
    fork.release()
    assert ledger.snapshot()["usage_bytes"] == 0


def test_input_batch_validation_is_atomic(transport):
    ledger = RuntimeDataLedger(DataAdmissionLimits(300, 100, 200))
    query = ledger.open_query()
    task = query.open_task(query.reserve_task())
    task.hold_inputs([DataAllocation("shm", "a", 60)])
    before = ledger.snapshot()
    with pytest.raises(DataBatchTooLarge, match="input.*requested=101"):
        task.hold_inputs([DataAllocation("shm", "a", 60), DataAllocation("shm", "b", 41)])
    assert ledger.snapshot() == before
    task.hold_inputs([DataAllocation("shm", "b", 40)])
    assert ledger.snapshot()["input_bytes"] == 100
    assert ledger.snapshot()["input_reserved_bytes"] == 0
    task.finish()
    query.shutdown()
    ledger.close()


def test_transport_refusal_rolls_back_runtime_reservation(transport):
    occupied = transport.reserve_task_bytes(5000, 5000)
    ledger = RuntimeDataLedger(DataAdmissionLimits(300, 100, 200))
    query = ledger.open_query()
    with pytest.raises(DataAdmissionCapacityError, match="transport"):
        query.reserve_task()
    assert ledger.snapshot()["reserved_bytes"] == 0
    occupied.release()
    query.reserve_task().release()
    query.shutdown()
    ledger.close()


def test_output_grant_uses_protected_bytes_even_when_transport_is_full(monkeypatch):
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 300)
    reservation = manager.reserve_task_bytes(100, 200)
    assert not manager.can_claim_output(1)
    assert reservation.allocate(100) == 100
    with pytest.raises(DataBatchTooLarge, match="output"):
        reservation.output_grant(201, name="oversized")
    grant = reservation.output_grant(200, name="completion")
    assert manager.snapshot()["usage_bytes"] == 300
    assert manager.snapshot()["waiting_output_grants"] == 0
    assert manager.convert_output_grant_to_allocation(grant) == 200
    reservation.release()
    manager.release_allocation(300)
    assert manager.snapshot()["usage_bytes"] == 0


@pytest.mark.parametrize("runtime_limited", [False, True])
def test_byte_refusal_returns_task_and_pool_capacity(transport, runtime_limited):
    ledger = RuntimeDataLedger(DataAdmissionLimits(300, 100, 200))
    queries = [ledger.open_query(), ledger.open_query()]
    pool = LocalExecutionSlotPool(max_slots=1, execution_slot_prefix="data")
    task_runtime = RuntimeTaskAdmission(TaskAdmissionLimits(1, 2)) if runtime_limited else None
    task_query = task_runtime.open_query() if task_runtime else None
    base = pool.create_authority()
    authority = DataAdmissionAuthority(task_query.create_authority(base) if task_query else base, queries[1])
    occupied = queries[0].reserve_task()
    try:
        with pytest.raises(DataAdmissionCapacityError):
            authority.request(8)
        assert base.active_lease_count == 0
        if task_runtime:
            assert task_runtime.snapshot()["running_tasks"] == task_runtime.snapshot()["ready_tasks"] == 0
        occupied.release()
        assert authority.request(8)
        assert authority.state()["available"]
        lease = authority.take(8)
        task = queries[1].open_task(lease.lease["local_data_reservation"])
        output = task.own_output(DataAllocation("shm", "output", 50))
        task.finish()
        lease.complete_execution()
        assert ledger.snapshot()["usage_bytes"] == 50
        lease.release()
        output.release()
    finally:
        occupied.release()
        authority.close()
        for query in queries:
            query.shutdown()
        if task_query:
            task_query.shutdown()
        if task_runtime:
            task_runtime.close()
        ledger.close()
        pool.close()


def test_unused_ready_bytes_are_released_by_query_shutdown(transport):
    ledger = RuntimeDataLedger(DataAdmissionLimits(300, 100, 200))
    query = ledger.open_query()
    pool = LocalExecutionSlotPool(max_slots=1, execution_slot_prefix="ready")
    authority = DataAdmissionAuthority(pool.create_authority(), query)
    authority.request(0)
    assert ledger.snapshot()["reservations"] == 1
    query.shutdown()
    ledger.close()
    assert ledger.snapshot()["usage_bytes"] == 0
    authority.close()
    pool.close()


@pytest.mark.parametrize("kind,amount,reason", [("task", 2, "total_bytes"), ("output", 2, None)])
def test_shared_policy_protects_output_capacity(kind, amount, reason):
    # A protected output share can progress even when other usage exceeds a
    # soft Ray budget. Strict local admission never overcommits its shares.
    assert (
        byte_budget_block_reason(
            ByteBudgetUsage(3, 2, 3, 0),
            amount,
            request_kind=kind,
            usage_bytes=6,
            limit_bytes=5,
            shared_used_bytes=0,
            shared_pool_bytes=0,
        )
        == reason
    )


def test_ready_wakeup_can_observe_byte_admission_from_another_thread(transport):
    ledger = RuntimeDataLedger(DataAdmissionLimits(300, 100, 200))
    query = ledger.open_query()
    runtime = RuntimeTaskAdmission(TaskAdmissionLimits(1, 2))
    task_query = runtime.open_query()
    pool = LocalExecutionSlotPool(max_slots=1, execution_slot_prefix="callback")
    authority = DataAdmissionAuthority(task_query.create_authority(pool.create_authority()), query)
    with ThreadPoolExecutor(max_workers=1) as threads:
        observed = []

        def wake():
            observed.append(threads.submit(authority.state).result(timeout=2))

        authority.register_wakeup(wake)
        try:
            authority.request(8)
            assert observed and observed[0]["available"]
            assert ledger.snapshot()["reservations"] == 1
        finally:
            authority.close()
            query.shutdown()
            task_query.shutdown()
            runtime.close()
            ledger.close()
            pool.close()


def test_runtime_close_waits_for_transport_reservation_cleanup(transport, monkeypatch):
    ledger = RuntimeDataLedger(DataAdmissionLimits(300, 100, 200))
    query = ledger.open_query()
    reservation = query.reserve_task()
    task = query.open_task(reservation)
    entered, proceed = threading.Event(), threading.Event()
    release = reservation.transport.release

    def blocked_release():
        entered.set()
        assert proceed.wait(timeout=5)
        release()

    monkeypatch.setattr(reservation.transport, "release", blocked_release)
    query.shutdown()
    with ThreadPoolExecutor(max_workers=1) as threads:
        finished = threads.submit(task.finish)
        try:
            assert entered.wait(timeout=5)
            with pytest.raises(TimeoutError, match="active queries"):
                ledger.close()
            assert ledger.snapshot()["reserved_bytes"] == 300
        finally:
            proceed.set()
        finished.result(timeout=5)
    ledger.close()
    assert ledger.snapshot()["usage_bytes"] == 0


def test_failed_transport_cleanup_remains_owned_until_explicit_retry(transport, monkeypatch):
    ledger = RuntimeDataLedger(DataAdmissionLimits(300, 100, 200))
    query = ledger.open_query()
    reservation = query.reserve_task()
    task = query.open_task(reservation)

    def fail():
        raise RuntimeError("planned transport cleanup failure")

    with monkeypatch.context() as patch:
        patch.setattr(reservation.transport, "release", fail)
        with pytest.raises(RuntimeError, match="planned transport cleanup failure"):
            task.finish()
        assert ledger.snapshot()["reservations"] == 1
        assert ledger.snapshot()["tasks"] == 0
        with pytest.raises(TimeoutError):
            ledger.close()
    query.shutdown()
    ledger.close()
    assert ledger.snapshot()["usage_bytes"] == 0
