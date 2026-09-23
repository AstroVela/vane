# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager

import pyarrow as pa
import pytest

from vane.execution import ref_bundle
from vane.execution.local_resource_graph import LocalResourceUnitContext
from vane.execution.udf_admission import LocalExecutionCapacity, LocalExecutionSlotPool
from vane.execution.udf_data_admission import DataAdmissionLimits, DataAdmissionWaitLimits
from vane.execution.udf_data_lease import RuntimeDataLedger
from vane.execution.udf_data_wait import WaitingDataAdmissionAuthority
from vane.execution.udf_runtime_admission import RuntimeTaskAdmission, TaskAdmissionLimits


class _Workload:
    def __init__(self, monkeypatch, *, runtime_blocks, transport_blocks, output_blocks, ratio, limited):
        self.table = pa.table({"blob": [b"x" * 4096]})
        self.block_bytes = ref_bundle._IPC_HEADER_SIZE + len(ref_bundle._arrow_table_to_ipc_bytes(self.table))
        self.transport = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: transport_blocks * self.block_bytes)
        monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", self.transport)
        self.ledger = RuntimeDataLedger(
            DataAdmissionLimits(
                runtime_blocks * self.block_bytes,
                self.block_bytes,
                output_blocks * self.block_bytes,
                unit_reservation_ratio=ratio,
                wait=DataAdmissionWaitLimits(8, 10),
            )
        )
        self.capacity = LocalExecutionCapacity(max_slots=1)
        self.task_runtime = RuntimeTaskAdmission(TaskAdmissionLimits(1, 8)) if limited else None
        self.queries, self.task_queries, self.pools, self.authorities, self.outputs = [], [], [], [], []

    def query(self):
        name = str(len(self.queries))
        units = [LocalResourceUnitContext(name, f"{name}:{role}", role, "subprocess_task") for role in ("p", "c")]
        query = self.ledger.open_query(resource_units=units)
        self.queries.append(query)
        task_query = self.task_runtime.open_query() if self.task_runtime else None
        if task_query is not None:
            self.task_queries.append(task_query)
        authorities = []
        for unit in units:
            pool = LocalExecutionSlotPool(
                max_slots=2, execution_slot_prefix=unit.resource_unit_id, execution_capacity=self.capacity
            )
            self.pools.append(pool)
            authority = WaitingDataAdmissionAuthority(
                pool.create_authority(), query, resource_unit=unit, task_query=task_query
            )
            self.authorities.append(authority)
            authorities.append(authority)
        return query, units, authorities

    @contextmanager
    def task(self, plan, index, *, request=True):
        query, units, authorities = plan
        authority = authorities[index]
        if request:
            assert authority.request(8)
        assert authority.state()["state"] == "ready"
        lease = authority.take(8)
        task = query.open_task(lease.lease["local_data_reservation"], resource_unit=units[index])
        try:
            yield task
        finally:
            try:
                task.finish()
            finally:
                lease.release()

    def produce(self, task, blocks):
        outputs = []
        for _ in range(blocks):
            result = ref_bundle.make_local_shm_ref_bundle_result(
                self.table, task_reservation=task.reservation.transport, allocation_role="output"
            )
            self.outputs.extend(result[1])
            ref_bundle.track_local_shm_output(task, result)
            outputs.extend(result[1])
        return outputs

    def consume(self, task, output):
        assert output.to_table().equals(self.table)
        ref_bundle.track_local_shm_inputs(task, [output])
        lease = ref_bundle.create_local_shm_input_lease([output], reserve_output_credit=False)
        task.hold_input_transport(self.transport, lease)

    def close(self):
        for authority in self.authorities:
            authority.close()
        for query in self.queries:
            query.shutdown()
        for query in self.task_queries:
            query.shutdown()
        if self.task_runtime:
            self.task_runtime.close()
        for output in self.outputs:
            output.release()
        self.ledger.close()
        for pool in self.pools:
            pool.close()
        assert self.ledger.snapshot()["usage_bytes"] == self.transport.snapshot()["usage_bytes"] == 0
        assert self.capacity.reserved_slots == 0
        assert self.ledger.snapshot()["queued_byte_admissions"] == 0


@pytest.fixture
def workload(monkeypatch):
    instances = []

    def make(**kwargs):
        instance = _Workload(monkeypatch, **kwargs)
        instances.append(instance)
        return instance

    yield make
    for instance in instances:
        instance.close()


@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("ratio,batches", [(None, 5), (0, 5), (0.5, 4), (1, 3)])
def test_activation_waits_for_existing_usage_above_repartitioned_shares(workload, ratio, batches, limited):
    w = workload(runtime_blocks=20, transport_blocks=20, output_blocks=3, ratio=ratio, limited=limited)
    first, later = w.query(), w.query()
    outputs = []
    for _ in range(batches):
        with w.task(first, 0) as task:
            outputs.extend(w.produce(task, 3))
    before = w.ledger.snapshot()
    assert before["usage_bytes"] == w.transport.snapshot()["usage_bytes"] == 3 * batches * w.block_bytes
    # A full envelope fits the hard limit, and all four minimum envelopes fit.
    # The existing outputs still exceed the shares available after activation.
    assert before["usage_bytes"] + w.ledger.limits.task_bytes <= w.ledger.limits.max_bytes
    assert later[2][0].request(8)
    assert later[2][0].state()["state"] == "waiting_bytes"
    assert w.ledger.snapshot()["unit_budget"] == before["unit_budget"]
    assert w.ledger.snapshot()["reserved_bytes"] == 0
    assert w.capacity.reserved_slots == 0
    if w.task_runtime:
        assert w.task_runtime.snapshot()["ready_tasks"] == 0

    # The older query's consumer bypasses the refused activation and drains a
    # real shared-memory input even with a single task allowance/worker slot.
    consumed = outputs.pop()
    with w.task(first, 1) as task:
        w.consume(task, consumed)
    consumed.release()
    for output in outputs:
        output.release()
    # Release notifications can activate the later query before the first one
    # shuts down; the later query's own downstream must also retain headroom.
    with w.task(later, 0, request=False) as task:
        later_outputs = w.produce(task, 3)
    for output in later_outputs:
        with w.task(later, 1) as task:
            w.consume(task, output)
        output.release()


@pytest.mark.parametrize("limited", [False, True])
def test_activation_wakes_when_existing_shared_usage_exactly_fits(workload, limited):
    w = workload(runtime_blocks=20, transport_blocks=20, output_blocks=3, ratio=0, limited=limited)
    first, later = w.query(), w.query()
    outputs = []
    for _ in range(3):
        with w.task(first, 0) as task:
            outputs.extend(w.produce(task, 3))
    assert later[2][0].request(8)
    assert later[2][0].state()["state"] == "waiting_bytes"
    outputs.pop().release()
    with w.task(later, 0, request=False):
        budget = w.ledger.snapshot()["unit_budget"]
        assert budget["shared_used_bytes"] == budget["shared_pool_bytes"] == 4 * w.block_bytes
    with w.task(first, 1) as task:
        w.consume(task, outputs.pop())


@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("ratio", [None, 0, 0.5, 1])
def test_downstream_shares_use_the_smaller_transport_capacity(workload, limited, ratio):
    w = workload(runtime_blocks=6, transport_blocks=4, output_blocks=1, ratio=ratio, limited=limited)
    plan = w.query()
    with w.task(plan, 0) as task:
        output = w.produce(task, 1)[0]
    producer, consumer = plan[2]
    assert producer.request(8)
    assert producer.state()["state"] == "waiting_bytes"
    # No task needs external memory release: its own consumer can progress.
    with w.task(plan, 1) as task:
        w.consume(task, output)
    output.release()
    with w.task(plan, 0, request=False) as task:
        output = w.produce(task, 1)[0]
    with w.task(plan, 1) as task:
        w.consume(task, output)
    output.release()
    assert w.ledger.snapshot()["unit_budget"]["limit_bytes"] == 4 * w.block_bytes


@pytest.mark.parametrize("transport_blocks", [0, 6, 8])
def test_unlimited_or_larger_transport_preserves_the_runtime_capacity(workload, transport_blocks):
    w = workload(runtime_blocks=6, transport_blocks=transport_blocks, output_blocks=1, ratio=0, limited=False)
    plan = w.query()
    for _ in range(3):
        with w.task(plan, 0) as task:
            output = w.produce(task, 1)[0]
    assert plan[2][0].request(8)
    assert plan[2][0].state()["state"] == "waiting_bytes"
    with w.task(plan, 1) as task:
        w.consume(task, output)
    output.release()
    with w.task(plan, 0, request=False):
        pass
    assert w.ledger.snapshot()["unit_budget"]["limit_bytes"] == 6 * w.block_bytes
