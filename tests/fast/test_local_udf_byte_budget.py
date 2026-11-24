# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from vane.execution import ref_bundle
from vane.execution.local_resource_graph import LocalResourceUnitContext
from vane.execution.udf_admission import LocalExecutionSlotPool
from vane.execution.udf_data_admission import DataAdmissionAuthority, DataAdmissionCapacityError, DataAdmissionLimits
from vane.execution.udf_data_lease import DataAllocation, RuntimeDataLedger
from vane.execution.udf_runtime_admission import RuntimeTaskAdmission, TaskAdmissionLimits


def _unit(name):
    return LocalResourceUnitContext(f"query-{name}", name, f"node-{name}", "subprocess_task")


def _budget(ledger):
    snapshot = ledger.snapshot()
    budget = snapshot["unit_budget"]
    assert budget["usage_bytes"] == snapshot["usage_bytes"]
    assert sum(unit["usage_bytes"] for unit in budget["units"]) == snapshot["usage_bytes"]
    assert snapshot["usage_bytes"] <= snapshot["limit_bytes"]
    return {unit["resource_unit_id"]: unit for unit in budget["units"]}


@pytest.fixture
def transport(monkeypatch):
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 10_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    yield manager
    assert manager.snapshot()["usage_bytes"] == 0
    assert manager.snapshot()["task_reserved_bytes"] == 0


@pytest.mark.parametrize("ratio", [-1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_unit_budget_ratio_is_explicit_and_finite(ratio):
    with pytest.raises(ValueError, match="unit_reservation_ratio"):
        DataAdmissionLimits(1000, 100, 100, unit_reservation_ratio=ratio)


@pytest.mark.parametrize("input_bytes,output_bytes", [(1, 999), (999, 1), (500, 500)])
@pytest.mark.parametrize("ratio", [0, 0.5, 1])
def test_one_complete_envelope_progresses_with_many_prepared_units(transport, input_bytes, output_bytes, ratio):
    ledger = RuntimeDataLedger(DataAdmissionLimits(1000, input_bytes, output_bytes, unit_reservation_ratio=ratio))
    units = [_unit(str(index)) for index in range(24)]
    query = ledger.open_query(resource_units=units)
    assert not any(unit["eligible"] for unit in _budget(ledger).values())
    reservation = query.reserve_task(resource_unit=units[0])
    budget = _budget(ledger)
    assert budget["0"]["protected_task_bytes"] == input_bytes
    assert budget["0"]["protected_output_bytes"] == output_bytes
    assert sum(unit["eligible"] for unit in budget.values()) == 1
    with pytest.raises(DataAdmissionCapacityError):
        query.reserve_task(resource_unit=units[1])
    reservation.release()
    query.reserve_task(resource_unit=units[1]).release()
    query.shutdown()
    ledger.close()
    assert _budget(ledger) == {}


@pytest.mark.parametrize("ratio", [0.5, 1])
def test_busy_unit_preserves_another_units_share_with_hard_capacity_remaining(transport, ratio):
    ledger = RuntimeDataLedger(DataAdmissionLimits(600, 50, 50, unit_reservation_ratio=ratio))
    first, second = _unit("a"), _unit("b")
    query = ledger.open_query(resource_units=[first, second])
    for _ in range(3):
        query.reserve_task(resource_unit=first)
    query.reserve_task(resource_unit=second)
    if ratio == 0.5:
        query.reserve_task(resource_unit=first)
    before = ledger.snapshot()
    assert before["usage_bytes"] + 100 <= 600
    with pytest.raises(DataAdmissionCapacityError) as caught:
        query.reserve_task(resource_unit=first)
    assert caught.value.reason.startswith("unit_")
    assert caught.value.resource_unit_id == "a"
    assert ledger.snapshot() == before
    query.reserve_task(resource_unit=second)
    assert _budget(ledger)["b"]["usage_bytes"] == 200
    query.shutdown()
    ledger.close()


def test_shared_allocation_keeps_one_charge_after_its_first_owner_retires(transport):
    ledger = RuntimeDataLedger(DataAdmissionLimits(600, 100, 100, unit_reservation_ratio=1))
    first, second = _unit("producer"), _unit("consumer")
    producer = ledger.open_query(resource_units=[first])
    consumer = ledger.open_query(resource_units=[second])
    task = producer.open_task(producer.reserve_task(resource_unit=first), resource_unit=first)
    allocation = DataAllocation("shm", "shared", 80)
    output = task.own_output(allocation)
    view = output.fork()
    task.finish()
    producer.shutdown()
    borrower = consumer.open_task(consumer.reserve_task(resource_unit=second), resource_unit=second)
    borrower.hold_inputs([allocation, allocation])
    associated = ledger.unit_snapshots()
    assert associated["producer"]["usage"]["output_bytes"] == 80
    assert associated["consumer"]["usage"]["input_bytes"] == 80
    budget = _budget(ledger)
    assert budget["producer"]["usage_bytes"] == 80
    assert not budget["producer"]["eligible"]
    assert budget["consumer"]["usage_bytes"] == 200  # The unused input envelope is still reserved.
    assert budget["consumer"]["protected_task_bytes"] + budget["consumer"]["protected_output_bytes"] == 520
    output.release()
    view.release()
    assert _budget(ledger)["producer"]["usage_bytes"] == 80
    with pytest.raises(ValueError, match="already belongs"):
        ledger.open_query(resource_units=[first])
    borrower.finish()
    consumer.shutdown()
    ledger.close()
    assert _budget(ledger) == {}


def test_input_origin_charge_survives_another_units_output_views_after_close(transport):
    ledger = RuntimeDataLedger(DataAdmissionLimits(600, 100, 100, unit_reservation_ratio=1))
    first, second = _unit("input"), _unit("output")
    source = ledger.open_query(resource_units=[first])
    sink = ledger.open_query(resource_units=[second])
    reader = source.open_task(source.reserve_task(resource_unit=first), resource_unit=first)
    allocation = DataAllocation("shm", "aliased", 80)
    reader.hold_inputs([allocation])
    writer = sink.open_task(sink.reserve_task(resource_unit=second), resource_unit=second)
    output = writer.own_output(allocation)
    writer.finish()
    reader.finish()
    source.shutdown()
    sink.shutdown()
    ledger.close()
    view = output.fork()
    output.release()
    budget = _budget(ledger)
    assert set(budget) == {"input"}
    assert budget["input"]["usage_bytes"] == ledger.snapshot()["output_bytes"] == 80
    assert budget["input"]["output_usage_bytes"] == 0
    assert not budget["input"]["eligible"]
    view.release()
    assert _budget(ledger) == {}


@pytest.mark.parametrize("failure", ["input", "reservation"])
def test_cleanup_failures_keep_unique_charges_and_retire_protected_shares(transport, monkeypatch, failure):
    ledger = RuntimeDataLedger(DataAdmissionLimits(600, 100, 100, unit_reservation_ratio=1))
    unit = _unit("cleanup")
    query = ledger.open_query(resource_units=[unit])
    reservation = query.reserve_task(resource_unit=unit)
    task = query.open_task(reservation, resource_unit=unit)
    task.hold_inputs([DataAllocation("shm", "input", 60)])

    def fail(*args, **kwargs):
        raise RuntimeError("planned cleanup failure")

    with monkeypatch.context() as fault:
        if failure == "input":
            task.hold_input_transport(transport, 1)
            fault.setattr(transport, "cancel_input_lease", fail)
        else:
            fault.setattr(reservation.transport, "release", fail)
        with pytest.raises(RuntimeError, match="planned cleanup failure"):
            task.finish()
        usage = 60 if failure == "input" else 140
        assert _budget(ledger)["cleanup"]["usage_bytes"] == usage
        assert ledger.snapshot()["unit_budget"]["inactive_usage_bytes"] == usage
        assert not _budget(ledger)["cleanup"]["eligible"]
        with pytest.raises(TimeoutError):
            ledger.close()
    query.shutdown()
    ledger.close()
    assert _budget(ledger) == {}


def test_transport_refusal_publishes_no_unit_budget_reservation(transport):
    ledger = RuntimeDataLedger(DataAdmissionLimits(600, 100, 100, unit_reservation_ratio=0.5))
    unit = _unit("transport")
    query = ledger.open_query(resource_units=[unit])
    occupied = transport.acquire_allocation(9900)
    try:
        with pytest.raises(DataAdmissionCapacityError) as caught:
            query.reserve_task(resource_unit=unit)
        assert caught.value.owner == "transport"
        assert _budget(ledger)["transport"]["usage_bytes"] == 0
        assert not _budget(ledger)["transport"]["eligible"]
    finally:
        transport.release_allocation(occupied)
    query.reserve_task(resource_unit=unit).release()
    query.shutdown()
    ledger.close()


@pytest.mark.parametrize("ratio", [0, 0.5, 1])
def test_concurrent_unit_admission_and_snapshots_share_one_atomic_ledger(transport, ratio):
    ledger = RuntimeDataLedger(DataAdmissionLimits(800, 50, 50, unit_reservation_ratio=ratio))
    units = [_unit(str(index)) for index in range(16)]
    queries = [ledger.open_query(resource_units=[unit]) for unit in units]
    barrier = threading.Barrier(len(units))

    def reserve(index):
        barrier.wait(timeout=10)
        try:
            reservation = queries[index].reserve_task(resource_unit=units[index])
        except DataAdmissionCapacityError:
            reservation = None
        _budget(ledger)
        return reservation

    with ThreadPoolExecutor(max_workers=len(units)) as workers:
        reservations = list(workers.map(reserve, range(len(units))))
    assert sum(reservation is not None for reservation in reservations) == 8
    assert ledger.snapshot()["usage_bytes"] == transport.snapshot()["usage_bytes"] == 800
    for query in queries:
        query.shutdown()
        _budget(ledger)
    ledger.close()


def test_unit_budget_rejects_missing_and_foreign_bindings(transport):
    ledger = RuntimeDataLedger(DataAdmissionLimits(600, 100, 100, unit_reservation_ratio=1))
    first, second = _unit("one"), _unit("two")
    query = ledger.open_query(resource_units=[first])
    for unit in (None, second):
        with pytest.raises(ValueError, match="bound to this data query"):
            query.reserve_task(resource_unit=unit)
    with pytest.raises(ValueError, match="duplicate"):
        ledger.open_query(resource_units=[second, second])
    assert ledger.snapshot()["usage_bytes"] == 0
    query.shutdown()
    ledger.close()


@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("reentrant", [False, True])
def test_unit_refusal_returns_execution_capacity_and_preserves_retry_details(transport, limited, reentrant):
    ledger = RuntimeDataLedger(DataAdmissionLimits(600, 50, 50, unit_reservation_ratio=1))
    first, second = _unit("busy"), _unit("other")
    query = ledger.open_query(resource_units=[first, second])
    reservations = [query.reserve_task(resource_unit=first) for _ in range(3)]
    query.reserve_task(resource_unit=second)
    pool = LocalExecutionSlotPool(max_slots=1, execution_slot_prefix="unit-bytes")
    runtime = RuntimeTaskAdmission(TaskAdmissionLimits(1, 2)) if limited else None
    task_query = runtime.open_query() if runtime else None
    base = pool.create_authority()
    authority = DataAdmissionAuthority(
        task_query.create_authority(base) if task_query else base, query, resource_unit=first
    )
    if reentrant:
        authority.register_wakeup(lambda: authority.state())
    try:
        with pytest.raises(DataAdmissionCapacityError) as caught:
            authority.request(0)
        assert caught.value.reason.startswith("unit_")
        assert caught.value.resource_unit_id == "busy"
        assert base.active_lease_count == 0
        if runtime:
            snapshot = runtime.snapshot()
            assert snapshot["ready_tasks"] == snapshot["running_tasks"] == 0
        reservations.pop().release()
        assert authority.request(0)
        lease = authority.take(0)
        task = query.open_task(lease.lease["local_data_reservation"], resource_unit=first)
        task.finish()
        lease.complete_execution()
        lease.release()
    finally:
        authority.close()
        query.shutdown()
        if task_query:
            task_query.shutdown()
            runtime.close()
        pool.close()
        ledger.close()
