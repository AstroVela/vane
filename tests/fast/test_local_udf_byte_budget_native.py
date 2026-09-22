# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import time

import pyarrow as pa
import pytest

from vane import pickle as vane_pickle
from vane.execution import ref_bundle, udf_subprocess
from vane.execution.local_resource_graph import LocalResourceUnitContext
from vane.execution.udf import build_executor
from vane.execution.udf_data_admission import DataAdmissionCapacityError, DataAdmissionLimits
from vane.execution.udf_data_lease import RuntimeDataLedger
from vane.execution.udf_runtime_admission import RuntimeTaskAdmission, TaskAdmissionLimits


def _wait(predicate):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise TimeoutError("byte budget subprocess condition did not become ready")


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
@pytest.mark.parametrize("limited", [False, True])
def test_real_shared_pool_preserves_other_units_share_and_retries_after_completion(
    monkeypatch, tmp_path, backend, limited
):
    gc.collect()
    transport = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", transport)
    go = tmp_path / "go"

    def identity(table):
        import time

        value = table.column(0)[0].as_py()
        (tmp_path / str(value)).touch()
        deadline = time.monotonic() + 60
        while not go.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("test worker was not released")
            time.sleep(0.01)
        return table

    class Identity:
        def __call__(self, table):
            return identity(table)

    payload = dict(
        function_pickle=vane_pickle.dumps(Identity if backend == "subprocess_actor" else identity),
        call_mode="map_batches",
        execution_backend=backend,
        actor_number=4,
        udf_worker_slots=4,
        produce_ref_bundle_output=True,
        streaming_output_mode="local_shm_ref_bundle",
    )
    # The test needs four physical threads even on a smaller developer machine.
    # Keep this runtime separate from cached pools used by other tests.
    task_pool_runtime = None
    if backend == "subprocess_task":
        with monkeypatch.context() as cpu_count:
            cpu_count.setattr(udf_subprocess.os, "cpu_count", lambda: 4)
            task_pool_runtime = udf_subprocess._GlobalSubprocessTaskRuntime()
        monkeypatch.setattr(udf_subprocess, "_GLOBAL_TASK_RUNTIME", task_pool_runtime)
    actor_pool = udf_subprocess.LocalSubprocessActorPool(payload, 4) if backend == "subprocess_actor" else None
    ledger = RuntimeDataLedger(DataAdmissionLimits(16_384, 2048, 2048, unit_reservation_ratio=1))
    task_runtime = RuntimeTaskAdmission(TaskAdmissionLimits(4, 8)) if limited else None
    queries, task_queries, executors, outputs = [], [], [], []
    try:
        for name in ("busy", "other"):
            unit = LocalResourceUnitContext(name, f"resource:{name}:udf:one", "one", backend)
            query = ledger.open_query(resource_units=[unit])
            queries.append(query)
            options = {"local_data_scope": query, "local_resource_unit": unit}
            if task_runtime:
                task_query = task_runtime.open_query()
                task_queries.append(task_query)
                options["local_task_admission"] = task_query
            if actor_pool:
                options["local_actor_pool"] = actor_pool
            executors.append(build_executor(payload, options))
        busy, other = executors
        if backend == "subprocess_task":
            assert busy._task_pool is other._task_pool

        def submit(executor, value):
            assert executor.request_task_admission(8)
            executor.submit_with_id(value, pa.table({"x": [value]}))
            _wait(lambda: (tmp_path / str(value)).exists())

        submit(busy, 0)
        submit(busy, 1)
        submit(other, 2)
        before = ledger.snapshot()
        assert before["usage_bytes"] == 12_288
        assert before["usage_bytes"] + 4096 == before["limit_bytes"]
        with pytest.raises(DataAdmissionCapacityError) as caught:
            busy.request_task_admission(8)
        assert caught.value.reason.startswith("unit_")
        assert caught.value.resource_unit_id == "resource:busy:udf:one"
        assert busy.task_admission_state()["state"] == "idle"
        assert ledger.snapshot() == before
        if task_runtime:
            assert task_runtime.snapshot()["running_tasks"] == 3
            assert task_runtime.snapshot()["ready_tasks"] == 0
        submit(other, 3)
        assert ledger.snapshot()["usage_bytes"] == 16_384
        assert transport.snapshot()["waiting_output_grants"] == 0
        go.touch()
        for executor in executors:
            for _ in range(2):
                result = _wait(executor.take_ready_result)
                assert not isinstance(result[2], BaseException), result[2]
                outputs.extend(result[2][1])
        _wait(lambda: ledger.snapshot()["reserved_bytes"] == 0)
        budget = ledger.snapshot()["unit_budget"]
        assert budget["usage_bytes"] == budget["inactive_usage_bytes"] > 0
        assert sum(unit["usage_bytes"] for unit in budget["units"]) == budget["usage_bytes"]
        submit(busy, 4)
        result = _wait(busy.take_ready_result)
        assert not isinstance(result[2], BaseException), result[2]
        outputs.extend(result[2][1])
        assert sorted(ref.to_table().column(0)[0].as_py() for ref in outputs) == list(range(5))
    finally:
        go.touch()
        for executor in executors:
            executor.close(kill=True)
        for query in queries:
            query.shutdown(kill=True)
        for query in task_queries:
            query.shutdown(kill=True)
        if task_runtime:
            task_runtime.close()
        if actor_pool:
            actor_pool.shutdown(kill=True)
        if task_pool_runtime:
            task_pool_runtime.close(kill=True)
        ledger.close()
        for ref in outputs:
            ref.release()
        gc.collect()
    assert ledger.snapshot()["usage_bytes"] == transport.snapshot()["usage_bytes"] == 0
