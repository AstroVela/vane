# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

import vane
from vane.execution import local_result_delivery
from vane.execution.request_admission import RequestAdmissionLimits, RequestExecutionTimeout
from vane.execution.result_delivery import (
    ResultDeliveryClosed,
    ResultDeliveryFull,
    ResultDeliveryLimits,
    ResultDeliveryTimeout,
)
from vane.execution.udf_local_model import LocalModelRuntime


def plan(conn, value=7, function=None, *, actor=False):
    relation = conn.sql(value if isinstance(value, str) else f"SELECT {value}::BIGINT AS x")
    if function is not None:
        relation = relation.map_batches(
            function,
            schema={"x": vane.sqltypes.BIGINT},
            execution_backend="subprocess_actor" if actor else "subprocess_task",
            actor_number=1 if actor else None,
        )
    return vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(conn)


def runtime(bound, *, results=2, size=1_000_000, **kwargs):
    return LocalModelRuntime(
        session_id=bound.session_id(),
        session_config=bound.session_config(),
        request_limit=RequestAdmissionLimits(1, 2),
        result_limit=ResultDeliveryLimits(results, size),
        **kwargs,
    )


@pytest.mark.parametrize("track_data", [False, True])
@pytest.mark.parametrize("actor", [False, True])
def test_native_managed_results_release_request_slots_and_preserve_exported_views(monkeypatch, track_data, actor):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    class Identity:
        def __call__(self, table):
            return table

    with vane.connect() as conn:
        bound = plan(conn, function=Identity if actor else lambda table: table, actor=actor)
        with runtime(bound, track_data=track_data) as models:
            request, queued = models.request(), models.request()
            result = request.execute_result(bound, {}, conn=conn, delivery_timeout=10)
            assert request.state == "finished" and queued.state == "ready"
            assert models.resource_snapshot()["result_delivery"]["active_results"] == 1
            before = models.resource_snapshot()["result_delivery"]["usage_bytes"]
            table = result.take()
            values = table.column(0).chunk(0).to_numpy(zero_copy_only=True)
            del table
            result.close()
            assert result.state == "delivered"
            queued.cancel()
            models.close()
            assert values.tolist() == [7]
            assert models.resource_snapshot()["result_delivery"]["usage_bytes"] == before
            del values
            gc.collect()
            assert models.resource_snapshot()["result_delivery"]["usage_bytes"] == 0


def test_native_nested_embedding_and_binary_results_roundtrip(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as conn:
        bound = plan(
            conn, "SELECT [1.0::DOUBLE, 2.0] AS embedding, 'image bytes'::BLOB AS image, {'text': 'caption'} AS context"
        )
        with runtime(bound) as models:
            with models.request().execute_result(bound, {}, conn=conn) as result:
                table = result.take()
                assert [column.to_pylist() for column in table.columns] == [
                    [[1.0, 2.0]],
                    [b"image bytes"],
                    [{"text": "caption"}],
                ]
                assert result.result_schema["types"] == ["DOUBLE[]", "BLOB", 'STRUCT("text" VARCHAR)']


def test_full_result_slots_refuse_before_native_udf_side_effects(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    marker = str(tmp_path / "udf-ran")

    def process(table):
        from pathlib import Path

        Path(marker).touch()
        return table

    with vane.connect() as conn:
        first, second = plan(conn), plan(conn, function=process)
        with runtime(first, results=1) as models:
            result = models.request().execute_result(first, {}, conn=conn)
            request = models.request()
            with pytest.raises(ResultDeliveryFull, match="slots"):
                request.execute_result(second, {}, conn=conn)
            assert request.state == "ready" and not (tmp_path / "udf-ran").exists()
            result.close()
            with request.execute_result(second, {}, conn=conn) as delivered:
                assert delivered.take().column(0).to_pylist() == [7]
            assert (tmp_path / "udf-ran").exists()


def test_native_output_too_large_releases_request_and_result_capacity(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as conn:
        bound = plan(conn)
        with runtime(bound, size=1) as models:
            request = models.request()
            with pytest.raises(ResultDeliveryFull, match="byte capacity"):
                request.execute_result(bound, {}, conn=conn)
            assert request.state == "finished"
            snapshot = models.resource_snapshot()
            assert snapshot["request_admission"]["active_requests"] == 0
            assert snapshot["result_delivery"]["active_results"] == snapshot["result_delivery"]["usage_bytes"] == 0


@pytest.mark.parametrize("stage", ["execution", "delivery"])
def test_native_execution_and_delivery_deadlines_are_distinct(monkeypatch, stage):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as conn:
        bound = plan(conn)
        with runtime(bound) as models:
            request = models.request()
            expected = RequestExecutionTimeout if stage == "execution" else ResultDeliveryTimeout
            with pytest.raises(expected):
                request.execute_result(
                    bound,
                    {},
                    conn=conn,
                    execution_timeout=0 if stage == "execution" else None,
                    delivery_timeout=0 if stage == "delivery" else None,
                )
            snapshot = models.resource_snapshot()
            assert request.state == ("execution_timed_out" if stage == "execution" else "finished")
            assert snapshot["request_admission"]["execution_timed_out_requests"] == int(stage == "execution")
            assert snapshot["result_delivery"]["timed_out_results"] == int(stage == "delivery")
            assert snapshot["result_delivery"]["active_results"] == snapshot["result_delivery"]["usage_bytes"] == 0


def test_native_delivery_cancellation_keeps_shared_model_and_other_results(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    marker = str(tmp_path / "initializations")

    class Model:
        def __init__(self):
            from pathlib import Path

            with Path(marker).open("a") as out:
                out.write("initialized\n")

        def __call__(self, table):
            return table

    with vane.connect() as conn:
        plans = [plan(conn, value, Model, actor=True) for value in (1, 2, 3)]
        nodes = [bound.collect_udf_nodes(conn=conn)[0] for bound in plans]
        with runtime(plans[0]) as models:
            model = models.register("model", version="v1", payload=nodes[0]["payload"])
            results = [
                models.request().execute_result(bound, {str(node["node_id"]): "model"}, conn=conn)
                for bound, node in zip(plans[:2], nodes[:2])
            ]
            with model.acquire() as borrow:
                pids = borrow.pool.worker_pids()
            assert results[0].cancel()
            assert results[1].take().column(0).to_pylist() == [2]
            with models.request().execute_result(plans[2], {str(nodes[2]["node_id"]): "model"}, conn=conn) as result:
                assert result.take().column(0).to_pylist() == [3]
            with model.acquire() as borrow:
                assert borrow.pool.worker_pids() == pids
            assert (tmp_path / "initializations").read_text().splitlines() == ["initialized"]


def test_runtime_close_retains_a_result_still_being_prepared(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    entered, proceed = threading.Event(), threading.Event()
    original = local_result_delivery.prepare_local_result

    def prepare(*args):
        entered.set()
        assert proceed.wait(10)
        return original(*args)

    monkeypatch.setattr(local_result_delivery, "prepare_local_result", prepare)
    with vane.connect() as conn:
        bound = plan(conn)
        models = runtime(bound)
        try:
            with ThreadPoolExecutor(max_workers=1) as threads:
                request = models.request()
                future = threads.submit(request.execute_result, bound, {}, conn=conn)
                try:
                    assert entered.wait(5)
                    assert request.state == "finished"
                    with pytest.raises(RuntimeError, match="result cleanup"):
                        models.close()
                    assert models.resource_snapshot()["result_delivery"]["active_results"] == 1
                finally:
                    proceed.set()
                with pytest.raises(ResultDeliveryClosed):
                    future.result(timeout=5)
            models.close()
            assert models.resource_snapshot()["result_delivery"]["active_results"] == 0
        finally:
            proceed.set()
            models.close()


@pytest.mark.parametrize("operation", ["take", "close"])
def test_native_result_cleanup_failure_is_owned_until_runtime_retry(monkeypatch, operation):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    original = local_result_delivery._ArrowResultPayload.close
    failing = [True]

    def close(self):
        if failing[0]:
            raise OSError("injected result cleanup failure")
        original(self)

    monkeypatch.setattr(local_result_delivery._ArrowResultPayload, "close", close)
    with vane.connect() as conn:
        bound = plan(conn)
        models = runtime(bound)
        try:
            result = models.request().execute_result(bound, {}, conn=conn)
            with pytest.raises((OSError, RuntimeError), match="cleanup failure|cleanup failed"):
                (result.take if operation == "take" else result.close)()
            snapshot = models.resource_snapshot()["result_delivery"]
            assert snapshot["usage_bytes"] > 0 and snapshot["cleanup_pending_results"] == 1
            with pytest.raises(RuntimeError, match="result cleanup"):
                models.close()
            failing[0] = False
            models.close()
            assert models.resource_snapshot()["result_delivery"]["usage_bytes"] == 0
            assert models.resource_snapshot()["result_delivery"]["active_results"] == 0
        finally:
            failing[0] = False
            models.close()


def test_ready_result_expiry_allows_a_later_native_request(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as conn:
        first, second = plan(conn, 1), plan(conn, 2)
        with runtime(first, results=1) as models:
            result = models.request().execute_result(first, {}, conn=conn, delivery_timeout=1)
            request = models.request()
            with pytest.raises(ResultDeliveryFull):
                request.execute_result(second, {}, conn=conn)
            gate = models._result_delivery
            with gate._condition:
                assert gate._condition.wait_for(lambda: result.state == "delivery_timed_out", timeout=5)
            with request.execute_result(second, {}, conn=conn) as delivered:
                assert delivered.take().column(0).to_pylist() == [2]
            assert models.resource_snapshot()["result_delivery"]["timed_out_results"] == 1


def test_drain_allows_ready_result_consumption_and_close_preserves_metadata(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as conn:
        bound = plan(conn)
        with runtime(bound) as models:
            result = models.request().execute_result(bound, {}, conn=conn)
            models.drain()
            with pytest.raises(RuntimeError, match="draining"):
                models.request()
            table = result.take()
            schema, status = result.result_schema, result.completion_status
            models.close()
            assert table.column(0).to_pylist() == [7]
            assert result.result_schema == schema and result.completion_status == status
