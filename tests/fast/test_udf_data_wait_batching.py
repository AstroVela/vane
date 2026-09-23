# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest

import vane
from vane.execution import ref_bundle, udf_subprocess
from vane.execution.request_admission import RequestAdmissionLimits, RequestCancelled
from vane.execution.udf_data_admission import DataAdmissionLimits, DataAdmissionWaitLimits
from vane.execution.udf_data_wait import WaitingDataAdmissionAuthority
from vane.execution.udf_local_model import LocalModelRuntime
from vane.execution.udf_runtime_admission import TaskAdmissionLimits


def _run_chain(
    monkeypatch, *, limited, actor=False, batch_size=1, minimum=2, wait=True, budget=420_000, termination=None
):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: budget)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    with monkeypatch.context() as cpus:
        cpus.setattr(udf_subprocess.os, "cpu_count", lambda: 1)
        task_runtime = udf_subprocess._GlobalSubprocessTaskRuntime()
    monkeypatch.setattr(udf_subprocess, "_GLOBAL_TASK_RUNTIME", task_runtime)
    submitted_rows = []
    submit = udf_subprocess.UDFExecutor.submit_ref_bundle_with_id

    def record_submit(self, submit_id, refs, slices, metadata, names):
        # The source is materialized; only the downstream UDF receives refs.
        submitted_rows.append(ref_bundle._estimate_ref_bundle_num_rows(slices, metadata))
        return submit(self, submit_id, refs, slices, metadata, names)

    monkeypatch.setattr(udf_subprocess.UDFExecutor, "submit_ref_bundle_with_id", record_submit)
    observed = threading.Event()
    byte_waiters = []
    if termination:
        state = WaitingDataAdmissionAuthority.state

        def delay_partial_flush(self):
            snapshot = state(self)
            if snapshot["state"] == "idle" and snapshot.get("flush_partial_input"):
                with self._ledger._condition:
                    byte_waiters[:] = [
                        (authority, authority._generation)
                        for authority in self._ledger._byte_waiters
                        if authority._reason in {"runtime_bytes", "transport_bytes"}
                    ]
                observed.set()
                return {**snapshot, "flush_partial_input": False}
            return snapshot

        monkeypatch.setattr(WaitingDataAdmissionAuthority, "state", delay_partial_flush)

    def expand(table):
        return pa.table({"blob": [b"x" * 65_536 for _ in range(len(table))]})

    class Expand:
        def __call__(self, table):
            return expand(table)

    def consume(table):
        return pa.table({"size": [len(value.as_py()) for value in table.column(0)]})

    try:
        with vane.connect() as connection:
            relation = (
                connection.sql("SELECT unnest([0, 1, 2, 3])::BIGINT AS x")
                .map_batches(
                    Expand if actor else expand,
                    schema={"blob": vane.sqltypes.BLOB},
                    execution_backend="subprocess_actor" if actor else "subprocess_task",
                    actor_number=1 if actor else None,
                    batch_size=1,
                    min_task_batch_size=1,
                    task_input_max_bytes=8,
                )
                .map_batches(
                    consume,
                    schema={"size": vane.sqltypes.BIGINT},
                    execution_backend="subprocess_task",
                    batch_size=batch_size,
                    min_task_batch_size=minimum,
                    task_input_max_bytes=140_000,
                )
            )
            plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(
                connection
            )
            with LocalModelRuntime(
                session_id=plan.session_id(),
                session_config=plan.session_config(),
                request_limit=RequestAdmissionLimits(1, 1),
                task_limit=TaskAdmissionLimits(1, 8) if limited else None,
                data_limit=DataAdmissionLimits(
                    budget,
                    140_000,
                    70_000,
                    wait=DataAdmissionWaitLimits(8, 15) if wait else None,
                ),
            ) as runtime:
                request = runtime.request()
                if termination:
                    with ThreadPoolExecutor(max_workers=1) as threads:
                        future = threads.submit(request.execute, plan, {}, conn=connection)
                        try:
                            assert observed.wait(10)
                            assert submitted_rows == []
                            if termination == "cancel":
                                assert request.cancel()
                                with pytest.raises(RequestCancelled):
                                    future.result(timeout=15)
                            else:
                                # Deliver the deadline callback at this exact
                                # ownership boundary, independent of worker startup.
                                authority, generation = byte_waiters[0]
                                authority._expire(generation)
                                with pytest.raises(Exception, match="byte-admission queue deadline expired"):
                                    future.result(timeout=15)
                        finally:
                            request.shutdown(kill=True)
                    # Keep the physical plan and failure traceback alive: neither
                    # may retain the cancelled native input after cleanup.
                    assert manager.snapshot()["usage_bytes"] == 0
                else:
                    result = request.execute(plan, {}, conn=connection)
                    assert [value for table in result.partition_payloads for value in table.column(0).to_pylist()] == [
                        65_536
                    ] * 4
                    del result
                assert runtime.resource_snapshot()["data"]["queries"] == 0
    finally:
        task_runtime.close(kill=True)
        gc.collect()
        assert manager.snapshot()["usage_bytes"] == 0
    return submitted_rows


@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("actor", [False, True])
@pytest.mark.parametrize("batch_size,minimum", [(1, 2), (2, 2), (2, None)])
def test_byte_pressure_drains_a_short_downstream_batch(monkeypatch, limited, actor, batch_size, minimum):
    submitted_rows = _run_chain(monkeypatch, limited=limited, actor=actor, batch_size=batch_size, minimum=minimum)
    assert submitted_rows == [1] * 4


@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("wait,budget", [(False, 420_000), (True, 630_000)])
def test_without_byte_pressure_native_batch_coalescing_is_preserved(monkeypatch, limited, wait, budget):
    submitted_rows = _run_chain(monkeypatch, limited=limited, batch_size=2, wait=wait, budget=budget)
    assert submitted_rows == [2, 2]


@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("actor", [False, True])
@pytest.mark.parametrize("termination", ["cancel", "timeout"])
def test_termination_releases_partial_input_without_discarding_the_plan(monkeypatch, limited, actor, termination):
    assert _run_chain(monkeypatch, limited=limited, actor=actor, termination=termination) == []
