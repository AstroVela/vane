# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real Ray control actors and native Flight query/result data planes."""

import gc
import time
from dataclasses import replace

import pytest

import vane
from vane.execution.direct_exchange import DirectExchangeLimits

pytestmark = [pytest.mark.real_ray, pytest.mark.usefixtures("ray_local")]


def resources(**changes):
    return replace(vane.RayResources(), exchange=DirectExchangeLimits(256, 128, 3, 1), **changes)


def assert_idle(connection):
    import ray

    runtime = connection.query_runtime
    assert runtime.resource_snapshot()["queries"] == {}
    assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
    for worker in runtime.pool.workers:
        assert ray.get(worker.resources_snapshot.remote())["reservations"] == {}


def test_two_worker_public_query_and_empty_result():
    with vane.connect(backend="ray", resources=resources()) as connection:
        with connection.query("select range as value from range(20)", rows_per_batch=2) as result:
            assert isinstance(result, vane.QueryResult)
            first = result.read_batch()
            assert 0 < first.num_rows <= 2
            values = first.column(0).to_pylist()
            del first
            values += result.collect().column(0).to_pylist()
            assert sorted(values) == list(range(20))
            assert result.execution_state == "SUCCEEDED"
        assert_idle(connection)
        with connection.query("select range::bigint as value from range(0)") as result:
            table = result.collect()
            assert table.num_rows == 0
            assert table.column_names == ["value"]
        assert_idle(connection)


def test_slow_client_backpressure_and_early_close():
    import ray

    with vane.connect(backend="ray", resources=resources()) as connection:
        result = connection.query("select range as value from range(1000000)")
        first = result.read_batch()
        assert first.num_rows > 0
        context = result.context
        assert not context.production_done
        scheduler = context._reader
        time.sleep(0.1)
        snapshots = [
            ray.get(worker.status.remote(epoch, result.query_id))
            for worker, epoch in zip(connection.query_runtime.pool.workers, connection.query_runtime.pool.epochs)
        ]
        assert any(not state["production_done"] for state in snapshots)
        assert all(state["owned_bytes"] <= 4096 for state in snapshots)
        result.close()
        assert scheduler.closed
        assert first.column(0).to_pylist()
        assert connection.query_runtime.resource_snapshot()["result_delivery"]["usage_bytes"] > 0
        del first
        gc.collect()
        assert_idle(connection)


@pytest.mark.parametrize("victim", ["worker", "result"])
def test_process_loss_after_partial_delivery(victim):
    import ray

    with vane.connect(backend="ray", resources=resources()) as connection:
        result = connection.query("select range from range(1000000)")
        first = result.read_batch()
        del first
        scheduler = result.context._reader
        actor = scheduler.relay if victim == "result" else connection.query_runtime.pool.workers[0]
        ray.kill(actor, no_restart=True)
        with pytest.raises(Exception):
            while True:
                batch = result.read_batch()
                del batch
        assert result.execution_state == "FAILED"
        result.close()
        assert connection.query_runtime.resource_snapshot()["queries"] == {}


def test_group_capacity_failure_rolls_back_before_start():
    with vane.connect(backend="ray", resources=resources(io_concurrency=2)) as connection:
        with pytest.raises(Exception, match="capacity"):
            connection.query("select range from range(20)")
        assert len(connection.query_runtime.pool.workers) == 2
        assert_idle(connection)


def test_interrupt_and_delivery_timeout():
    from vane.execution.request_admission import RequestCancelled
    from vane.execution.result_delivery import ResultDeliveryTimeout

    with vane.connect(backend="ray", resources=resources()) as connection:
        result = connection.query("select range from range(1000000)")
        batch = result.read_batch()
        del batch
        connection.interrupt()
        with pytest.raises(RequestCancelled):
            result.read_batch()
        result.close()
        assert_idle(connection)
        options = vane.QueryExecutionOptions(vane.RayExecution(), 30, 30, 0.1)
        result = connection.query("select range from range(1000000)", options=options)
        time.sleep(0.2)
        with pytest.raises(ResultDeliveryTimeout):
            result.read_batch()
        result.close()
        assert_idle(connection)


def test_native_hash_routing_and_parquet(tmp_path, monkeypatch):
    from vane.execution import pipelined_runtime
    from vane.execution.compiler import FragmentCompileOptions

    source = tmp_path / "input.parquet"
    with vane.connect(backend="local") as local:
        local.execute(f"copy (select range % 7 as k, 'value-' || range::varchar as v from range(40)) to '{source}'")
    prepare = pipelined_runtime.prepare_ray_query

    def hash_plan(*args, **kwargs):
        kwargs["compile_options"] = FragmentCompileOptions(2, (0,))
        return prepare(*args, **kwargs)

    monkeypatch.setattr(pipelined_runtime, "prepare_ray_query", hash_plan)
    with vane.connect(backend="ray", resources=resources()) as connection:
        with connection.query(f"select k, v from read_parquet('{source}') where k > 2") as result:
            actual = result.collect().to_pylist()
        assert sorted((row["k"], row["v"]) for row in actual) == sorted(
            (i % 7, f"value-{i}") for i in range(40) if i % 7 > 2
        )
        assert_idle(connection)


@pytest.mark.parametrize("poll_status", [False, True])
def test_completed_production_survives_execution_deadline(monkeypatch, poll_status):
    from vane.execution.pipelined_runtime import PipelinedScheduler

    with vane.connect(backend="ray", resources=resources()) as connection:
        connection.query("select 1").collect()  # Warm up the fixed worker pool.
        if not poll_status:
            monkeypatch.setattr(PipelinedScheduler, "_monitor", lambda owner: owner.stop.wait())
        options = vane.QueryExecutionOptions(vane.RayExecution(), 30, 2, 30)
        result = connection.query("select 42 as value", options=options)
        if poll_status:
            deadline = time.monotonic() + 1.5
            while not result.context.production_done:
                assert time.monotonic() < deadline
                time.sleep(0.01)
        time.sleep(2.1)
        assert result.collect().to_pylist() == [{"value": 42}]
        assert result.execution_state == "SUCCEEDED"
        assert_idle(connection)


def test_execution_timeout_and_shared_session_admission():
    from vane.execution.request_admission import RequestExecutionTimeout, RequestQueueTimeout

    with (
        vane.connect(backend="ray", resources=resources(max_active_queries=1)) as connection,
        connection.cursor() as cursor,
    ):
        connection.query("select 1").collect()
        options = vane.QueryExecutionOptions(vane.RayExecution(), 30, 2, 30)
        result = connection.query("select range from range(1000000)", options=options)
        batch = result.read_batch()
        del batch
        queued = vane.QueryExecutionOptions(vane.RayExecution(), 0.1, 30, 30)
        with pytest.raises(RequestQueueTimeout):
            cursor.query("select 2", options=queued)
        time.sleep(2.1)
        with pytest.raises(RequestExecutionTimeout):
            result.read_batch()
        result.close()
        assert_idle(connection)


def test_failure_wakes_client_waiting_for_result_capacity():
    from concurrent.futures import ThreadPoolExecutor

    import ray

    with vane.connect(backend="ray", resources=resources(result_buffer_bytes=512)) as connection:
        result = connection.query("select range from range(1000000)")
        first = result.read_batch()
        expected = first.column(0).to_pylist()
        with ThreadPoolExecutor(max_workers=1) as executor:
            waiting = executor.submit(result.read_batch)
            deadline = time.monotonic() + 5
            while not result._waiting_bytes:
                assert not waiting.done()
                assert time.monotonic() < deadline
                time.sleep(0.005)
            ray.kill(connection.query_runtime.pool.workers[0], no_restart=True)
            with pytest.raises(Exception):
                waiting.result(timeout=10)
        result.close()
        assert result.execution_state == "FAILED"
        assert first.column(0).to_pylist() == expected
        del first
        gc.collect()
        assert connection.query_runtime.resource_snapshot()["result_delivery"]["usage_bytes"] == 0


@pytest.mark.parametrize("execution_timeout", [60, 2])
def test_long_native_filter_uses_execution_deadline(execution_timeout):
    from vane.execution.request_admission import RequestExecutionTimeout

    # A supported filter with no output can hold the pump execution lock for
    # longer than a status RPC's five-second deadline. Keep the real monitor
    # enabled: its liveness checks must not become an extra execution deadline.
    limits = vane.RayResources(worker_count=1, partitions=1, max_active_queries=1)
    options = vane.QueryExecutionOptions(vane.RayExecution(), 30, execution_timeout, 60)
    sql = "select range from range(30000) where hash(upper('" + "ß" * 16000 + "' || range::varchar)) = 0"
    with vane.connect(backend="ray", resources=limits) as connection:
        connection.query("select 1").collect()  # Exclude actor startup from the execution deadline.
        if execution_timeout == 60:
            with connection.query(sql, options=options) as result:
                assert result.collect().num_rows == 0
                assert result.execution_state == "SUCCEEDED"
        else:
            with pytest.raises(RequestExecutionTimeout):
                with connection.query(sql, options=options) as result:
                    result.collect()
        assert_idle(connection)
