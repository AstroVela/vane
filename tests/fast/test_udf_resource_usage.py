# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import weakref
from contextlib import contextmanager

import pytest

from vane.execution import ref_bundle
from vane.execution.local_resource_graph import LocalResourceUnitContext
from vane.execution.udf_admission import LocalExecutionSlotPool
from vane.execution.udf_data_admission import DataAdmissionAuthority, DataAdmissionCapacityError, DataAdmissionLimits
from vane.execution.udf_data_lease import DataAllocation, RuntimeDataLedger
from vane.execution.udf_resource_usage import UnitResourceActivity, observe_transport_wait, unit_usage_snapshot
from vane.execution.udf_runtime_admission import RuntimeTaskAdmission, TaskAdmissionLimits


def _unit(query="query", node="one"):
    return LocalResourceUnitContext(query, f"{query}:{node}", node, "subprocess_task")


@pytest.mark.parametrize("limited", [False, True])
def test_admission_snapshots_are_passive_even_when_byte_capacity_is_full(monkeypatch, limited):
    transport = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 10_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", transport)
    ledger = RuntimeDataLedger(DataAdmissionLimits(300, 100, 200))
    query = ledger.open_query()
    occupied = query.reserve_task()
    pool = LocalExecutionSlotPool(max_slots=1, execution_slot_prefix="passive")
    runtime = RuntimeTaskAdmission(TaskAdmissionLimits(1, 2)) if limited else None
    task_query = runtime.open_query() if runtime else None
    base = pool.create_authority()
    if task_query:
        base = task_query.create_authority(base)
    activity = UnitResourceActivity(_unit().to_dict())
    authority = DataAdmissionAuthority(base, query, resource_unit=_unit(), activity=activity)
    activity.bind_admission(authority)
    try:
        # Make only the underlying task grant ready. state() on the wrapper
        # would try a byte reservation and fail; diagnostics must not do that.
        assert base.request(0)
        before = ledger.snapshot(), transport.snapshot()
        for _ in range(3):
            assert activity.snapshot()["ready_tasks"] == 1
            assert activity.snapshot()["byte_refusals"]["runtime_bytes"] == 0
            assert (ledger.snapshot(), transport.snapshot()) == before
        with pytest.raises(DataAdmissionCapacityError):
            authority.state()
        assert activity.snapshot()["ready_tasks"] == 0
        assert activity.snapshot()["byte_refusals"]["runtime_bytes"] == 1
        occupied.release()
        assert authority.request(0)
        assert ledger.unit_snapshots()[_unit().resource_unit_id]["usage"]["reserved_bytes"] == 300
    finally:
        occupied.release()
        authority.close()
        query.shutdown()
        ledger.close()
        if task_query:
            task_query.shutdown()
            runtime.close()
        pool.close()


def test_activities_report_queued_tasks_without_retaining_closed_authorities():
    pool = LocalExecutionSlotPool(max_slots=1, execution_slot_prefix="queued")
    first, second = pool.create_authority(), pool.create_authority()
    activity = UnitResourceActivity(_unit().to_dict())
    activity.bind_admission(first)
    activity.bind_admission(second)
    first.request(0)
    second.request(0)
    assert activity.snapshot()["ready_tasks"] == activity.snapshot()["queued_tasks"] == 1
    assert activity.snapshot()["waiting_by_reason"]["task_capacity"] == 1
    first.close()
    assert activity.snapshot()["ready_tasks"] == 1
    assert activity.snapshot()["queued_tasks"] == 0
    second.close()
    ref = weakref.ref(second)
    del second
    gc.collect()
    assert ref() is None
    assert activity.snapshot()["ready_tasks"] == 0
    pool.close()


@pytest.mark.parametrize("wrapped", [False, True])
def test_custom_pool_authorities_remain_valid_without_passive_diagnostics(wrapped):
    class Authority:
        __slots__ = ()  # The existing admission contract does not require weakrefs.

        def state(self):
            raise AssertionError("diagnostics must not call the active state method")

    ledger = RuntimeDataLedger()
    query = ledger.open_query()
    activity = UnitResourceActivity(_unit().to_dict())
    authority = DataAdmissionAuthority(Authority(), query) if wrapped else Authority()
    activity.bind_admission(authority)
    snapshot = activity.snapshot()
    assert snapshot["queued_tasks"] is snapshot["ready_tasks"] is None
    assert snapshot["waiting_by_reason"]["task_capacity"] is None
    query.shutdown()
    ledger.close()


def test_nonweakrefable_passive_authority_does_not_gain_a_diagnostic_owner():
    class Authority:
        __slots__ = ()

        def diagnostic_state(self):
            return "ready"

    activity = UnitResourceActivity(_unit().to_dict())
    activity.bind_admission(Authority())
    assert activity.snapshot()["ready_tasks"] is None


@pytest.mark.parametrize("reason", ["shared_memory_input", "shared_memory_output"])
@pytest.mark.parametrize("finish_during_wait", [False, True])
def test_transport_wait_reports_resume_capacity_without_resurrecting_finished_tasks(reason, finish_during_wait):
    activity = UnitResourceActivity(_unit().to_dict())
    task = activity.open_task()
    task.transition("running")

    @contextmanager
    def release_and_reacquire():
        assert activity.snapshot()["waiting_by_reason"][reason] == 1
        yield
        assert activity.snapshot()["waiting_by_reason"]["execution_capacity"] == (0 if finish_during_wait else 1)

    with task.activate(), observe_transport_wait(release_and_reacquire(), reason):
        snapshot = activity.snapshot()
        assert snapshot["running_tasks"] == 0 and snapshot["waiting_tasks"] == 1
        if finish_during_wait:
            task.finish()
    assert activity.snapshot()["running_tasks"] == (0 if finish_during_wait else 1)
    assert activity.snapshot()["waiting_tasks"] == 0
    task.finish()
    assert not unit_usage_snapshot({_unit().resource_unit_id: activity}, None, prepared_query_ids=set())


@pytest.mark.parametrize("consumer_identity", [_unit("second"), None])
def test_data_attribution_deduplicates_shared_allocations_and_preserves_output_views(consumer_identity):
    ledger = RuntimeDataLedger()
    producer_query, consumer_query = ledger.open_query(), ledger.open_query()
    producer = producer_query.open_task(resource_unit=_unit())
    consumer = consumer_query.open_task(resource_unit=consumer_identity)
    allocation = DataAllocation("local_shm", "shared", 296)
    output = producer.own_output(allocation)
    view = output.fork()
    consumer.hold_inputs([allocation, allocation])
    units = ledger.unit_snapshots()
    usage = units[_unit().resource_unit_id]["usage"]
    assert usage["retained_bytes"] == usage["output_bytes"] == usage["shared_retained_bytes"] == 296
    assert usage["allocations"] == 1 and usage["leases"] == 2
    if consumer_identity:
        assert units[consumer_identity.resource_unit_id]["usage"]["input_bytes"] == 296
    assert ledger.snapshot()["retained_bytes"] == 296
    producer.finish()
    producer_query.shutdown()
    output.release()
    consumer.finish()
    consumer_query.shutdown()
    ledger.close()
    (unit,) = unit_usage_snapshot({}, ledger.unit_snapshots(), prepared_query_ids=set())
    assert unit["resource_unit_id"] == _unit().resource_unit_id
    assert unit["data"]["retained_bytes"] == 296
    assert unit["data"]["shared_retained_bytes"] == 0
    view.release()
    assert not ledger.unit_snapshots()
    assert ledger.snapshot()["retained_bytes"] == 0


def test_byte_reservation_requires_the_same_unit_and_retains_failed_cleanup(monkeypatch):
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 1000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    ledger = RuntimeDataLedger(DataAdmissionLimits(300, 100, 200))
    query = ledger.open_query()
    reservation = query.reserve_task(resource_unit=_unit())
    with pytest.raises(RuntimeError, match="own live byte admission"):
        query.open_task(reservation, resource_unit=_unit(node="other"))
    task = query.open_task(reservation, resource_unit=_unit())
    task.hold_inputs([DataAllocation("shm", "input", 80)])
    output = task.own_output(DataAllocation("shm", "output", 120))
    usage = ledger.unit_snapshots()[_unit().resource_unit_id]["usage"]
    assert usage["retained_bytes"] == 200
    assert usage["input_reserved_bytes"] == 20 and usage["output_reserved_bytes"] == 80
    release = reservation.transport.release

    def fail():
        raise RuntimeError("planned reservation cleanup failure")

    monkeypatch.setattr(reservation.transport, "release", fail)
    try:
        with pytest.raises(RuntimeError, match="planned reservation cleanup failure"):
            task.finish()
        usage = ledger.unit_snapshots()[_unit().resource_unit_id]["usage"]
        assert usage["reservations"] == 1 and usage["reserved_bytes"] == 100
        assert usage["output_bytes"] == 120
        with pytest.raises(TimeoutError):
            ledger.close()
    finally:
        monkeypatch.setattr(reservation.transport, "release", release)
        query.shutdown()
        output.release()
        ledger.close()
    assert not ledger.unit_snapshots()


def test_failed_input_cleanup_keeps_unit_accounting_until_success():
    ledger = RuntimeDataLedger()
    query = ledger.open_query()
    task = query.open_task(resource_unit=_unit())
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 1000)

    class Owner:
        size = 296
        fail = True

        def release(self):
            if self.fail:
                raise RuntimeError("planned input cleanup failure")

    owner = Owner()
    task.hold_inputs([DataAllocation("local_shm", "input", owner.size)])
    lease = manager.create_input_lease([owner], owner.size, reserve_output_credit=False)
    task.hold_input_transport(manager, lease)
    try:
        with pytest.raises(RuntimeError, match="planned input cleanup failure"):
            task.finish()
        usage = ledger.unit_snapshots()[_unit().resource_unit_id]["usage"]
        assert usage["input_bytes"] == 296 and usage["cleanup_pending_tasks"] == 1
    finally:
        owner.fail = False
        query.shutdown()
        ledger.close()
    assert not ledger.unit_snapshots()


def test_retained_output_preserves_available_refusal_history_after_graph_release():
    ledger = RuntimeDataLedger()
    query = ledger.open_query()
    activity = UnitResourceActivity(_unit().to_dict())
    activity.refuse_bytes("runtime")
    task = query.open_task(resource_unit=_unit())
    output = task.own_output(DataAllocation("local_shm", "output", 20))
    task.finish()
    query.shutdown()
    ledger.close()
    try:
        (snapshot,) = unit_usage_snapshot(
            {_unit().resource_unit_id: activity}, ledger.unit_snapshots(), prepared_query_ids=set()
        )
        assert snapshot["byte_refusals"]["runtime_bytes"] == 1
        (data_only,) = unit_usage_snapshot({}, ledger.unit_snapshots(), prepared_query_ids=set())
        assert data_only["byte_refusals"] is None
        assert data_only["data"]["retained_bytes"] == 20
    finally:
        output.release()
