# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest

import vane
from vane.execution import ref_bundle, udf_subprocess
from vane.execution.request_admission import RequestAdmissionLimits, RequestCancelled
from vane.execution.udf_data_admission import DataAdmissionLimits, DataAdmissionProgressError, DataAdmissionWaitLimits
from vane.execution.udf_local_model import LocalModelRuntime
from vane.execution.udf_runtime_admission import TaskAdmissionLimits


def _wait(predicate, future=None):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if future is not None and future.done():
            future.result()
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError("native byte-admission condition did not become ready")


@pytest.fixture
def native_environment(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    gc.collect()
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 280_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    with monkeypatch.context() as cpu_count:
        cpu_count.setattr(udf_subprocess.os, "cpu_count", lambda: 1)
        task_runtime = udf_subprocess._GlobalSubprocessTaskRuntime()
    monkeypatch.setattr(udf_subprocess, "_GLOBAL_TASK_RUNTIME", task_runtime)
    yield manager, task_runtime
    task_runtime.close(kill=True)
    gc.collect()
    assert manager.snapshot()["usage_bytes"] == 0
    assert manager.snapshot()["waiting_output_grants"] == 0


def _plan(connection, *, actor=False, chained=True, rows=1):
    def expand(table):
        return pa.table({"blob": [b"x" * 65_536 for _ in range(len(table))]})

    class Expand:
        def __call__(self, table):
            return expand(table)

    def consume(table):
        return pa.table({"size": [len(value.as_py()) for value in table.column(0)]})

    source = "SELECT 0::BIGINT AS x" if rows == 1 else f"SELECT unnest({list(range(rows))})::BIGINT AS x"
    relation = connection.sql(source).map_batches(
        Expand if actor else expand,
        schema={"blob": vane.sqltypes.BLOB},
        execution_backend="subprocess_actor" if actor else "subprocess_task",
        actor_number=1 if actor else None,
        batch_size=1,
        min_task_batch_size=1,
        task_input_max_bytes=8,
    )
    if chained:
        relation = relation.map_batches(
            consume,
            schema={"size": vane.sqltypes.BIGINT},
            execution_backend="subprocess_task",
            batch_size=1,
            min_task_batch_size=1,
            task_input_max_bytes=70_000,
        )
    return vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(connection)


def _runtime(plan, *, limited=False, budget=280_000, timeout=10):
    return LocalModelRuntime(
        session_id=plan.session_id(),
        session_config=plan.session_config(),
        request_limit=RequestAdmissionLimits(2, 2),
        task_limit=TaskAdmissionLimits(1, 8) if limited else None,
        data_limit=DataAdmissionLimits(budget, 70_000, 70_000, wait=DataAdmissionWaitLimits(8, timeout)),
    )


@pytest.mark.parametrize("actor", [False, True])
@pytest.mark.parametrize("limited", [False, True])
def test_native_chain_drains_repeated_large_outputs_with_one_worker(native_environment, actor, limited):
    manager, _ = native_environment
    with vane.connect() as connection:
        plan = _plan(connection, actor=actor, rows=8)
        with _runtime(plan, limited=limited) as runtime:
            result = runtime.request().execute(plan, {}, conn=connection)
            assert [value for table in result.partition_payloads for value in table.column(0).to_pylist()] == [
                65_536
            ] * 8
            assert runtime.resource_snapshot()["data"]["queued_byte_admissions"] == 0
            del result
        gc.collect()
        assert manager.snapshot()["usage_bytes"] == runtime.resource_snapshot()["data"]["usage_bytes"] == 0


@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("action", ["release", "cancel", "timeout"])
def test_native_transport_wait_is_woken_or_cancelled_without_starting_a_worker(
    native_environment, limited, action, monkeypatch, tmp_path
):
    manager, _ = native_environment
    marker = str(tmp_path / "entered")

    def model(table):
        from pathlib import Path

        Path(marker).touch()
        return table

    with vane.connect() as connection:
        relation = connection.sql("SELECT 7::BIGINT AS x").map_batches(
            model, schema={"x": vane.sqltypes.BIGINT}, execution_backend="subprocess_task"
        )
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(connection)
        with _runtime(plan, limited=limited, timeout=1 if action == "timeout" else 10) as runtime:
            request = runtime.request()
            occupied = manager.reserve_task_bytes(140_000, 140_000)
            try:
                with ThreadPoolExecutor(max_workers=1) as threads:
                    future = threads.submit(request.execute, plan, {}, conn=connection)
                    try:
                        _wait(lambda: runtime.resource_snapshot()["data"]["queued_byte_admissions"] == 1, future)
                        snapshot = runtime.resource_snapshot()
                        assert snapshot["data"]["reserved_bytes"] == 0
                        if limited:
                            assert snapshot["task_admission"]["ready_tasks"] == 0
                        assert not (tmp_path / "entered").exists()
                        if action == "cancel":
                            assert request.cancel()
                            with pytest.raises(RequestCancelled):
                                future.result(timeout=15)
                        elif action == "timeout":
                            with pytest.raises(Exception, match="byte-admission queue deadline expired"):
                                future.result(timeout=15)
                        else:
                            occupied.release()
                            result = future.result(timeout=20)
                            assert result.partition_payloads[0].column(0).to_pylist() == [7]
                            del result
                    finally:
                        occupied.release()
                        request.shutdown(kill=True)
                assert (tmp_path / "entered").exists() == (action == "release")
                assert runtime.resource_snapshot()["data"]["queued_byte_admissions"] == 0
            finally:
                occupied.release()


def test_native_plan_without_downstream_headroom_fails_before_preparation_publishes(native_environment):
    with vane.connect() as connection:
        plan = _plan(connection)
        with _runtime(plan, budget=140_000) as runtime:
            with pytest.raises(DataAdmissionProgressError, match="one complete envelope per plan UDF"):
                runtime.request().execute(plan, {}, conn=connection)
            snapshot = runtime.resource_snapshot()
            assert snapshot["data"]["queries"] == 0
            assert snapshot["prepared_query_graphs"] == []
            assert snapshot["data"]["usage_bytes"] == 0


@pytest.mark.parametrize("limited", [False, True])
def test_native_later_query_waits_for_a_slow_consumer_view(native_environment, limited, monkeypatch):
    manager, _ = native_environment
    views = []
    take_result = udf_subprocess.UDFExecutor.take_ready_result

    def retain_view(executor):
        result = take_result(executor)
        if result is not None and not isinstance(result[2], BaseException):
            views.extend(ref.to_table() for ref in result[2][1])
        return result

    with vane.connect() as connection:
        first_cursor, second_cursor = connection.cursor(), connection.cursor()
        first, second = _plan(first_cursor, chained=False), _plan(second_cursor)
        with _runtime(first, limited=limited) as runtime:
            with monkeypatch.context() as consumer:
                consumer.setattr(udf_subprocess.UDFExecutor, "take_ready_result", retain_view)
                result = runtime.request().execute(first, {}, conn=first_cursor)
            retained = runtime.resource_snapshot()["data"]["usage_bytes"]
            assert retained > 0
            request = runtime.request()
            with ThreadPoolExecutor(max_workers=1) as threads:
                future = threads.submit(request.execute, second, {}, conn=second_cursor)
                try:
                    _wait(lambda: runtime.resource_snapshot()["data"]["queued_byte_admissions"] > 0, future)
                    assert runtime.resource_snapshot()["data"]["usage_bytes"] == retained
                    assert not future.done()
                    assert views[0].column(0)[0].as_py() == b"x" * 65_536
                    del result
                    views.clear()
                    gc.collect()
                    completed = future.result(timeout=20)
                    assert completed.partition_payloads[0].column(0).to_pylist() == [65_536]
                    del completed
                finally:
                    views.clear()
                    request.shutdown(kill=True)
            gc.collect()
            assert manager.snapshot()["usage_bytes"] == 0
        first_cursor.close()
        second_cursor.close()


def test_wait_configuration_accepts_a_native_query_without_udfs(native_environment):
    with vane.connect() as connection:
        relation = connection.sql("SELECT 7::BIGINT AS x")
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(connection)
        with _runtime(plan) as runtime:
            result = runtime.request().execute(plan, {}, conn=connection)
            assert result.partition_payloads[0].column(0).to_pylist() == [7]
            assert runtime.resource_snapshot()["data"]["usage_bytes"] == 0
            assert runtime.resource_snapshot()["data"]["queries"] == 0


def test_byte_wait_cleanup_preserves_the_registered_model_for_the_next_request(native_environment):
    manager, _ = native_environment
    with vane.connect() as connection:
        plan = _plan(connection, actor=True, chained=False)
        node = plan.collect_udf_nodes(conn=connection)[0]
        with _runtime(plan, limited=True) as runtime:
            model = runtime.register("model", version="1", payload=node["payload"])
            model.prewarm()
            with model.acquire() as borrow:
                pids = borrow.pool.worker_pids()
            request = runtime.request()
            occupied = manager.reserve_task_bytes(140_000, 140_000)
            try:
                with ThreadPoolExecutor(max_workers=1) as threads:
                    future = threads.submit(request.execute, plan, {str(node["node_id"]): "model"}, conn=connection)
                    try:
                        _wait(lambda: runtime.resource_snapshot()["data"]["queued_byte_admissions"] == 1, future)
                    finally:
                        occupied.release()
                    result = future.result(timeout=20)
                    assert len(result.partition_payloads[0].column(0)[0].as_py()) == 65_536
                    del result
            finally:
                occupied.release()
                request.shutdown(kill=True)
            result = runtime.request().execute(plan, {str(node["node_id"]): "model"}, conn=connection)
            assert len(result.partition_payloads[0].column(0)[0].as_py()) == 65_536
            del result
            with model.acquire() as borrow:
                assert borrow.pool.worker_pids() == pids
