# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest

import vane
from vane.execution.query_runtime import QueryContext
from vane.execution.request_admission import RequestCancelled, RequestExecutionTimeout, RequestQueueTimeout
from vane.execution.result_delivery import ResultDeliveryCancelled, ResultDeliveryFull, ResultDeliveryTimeout


def options(*, admission=5.0, execution=30.0, delivery=30.0):
    return vane.QueryExecutionOptions(vane.LocalExecution(), admission, execution, delivery)


def limits(*, active=1, results=1, size=20_000):
    return vane.QueryResources(active, 4, results, size)


def wait(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "query did not reach the expected phase"
        time.sleep(0.005)


def idle(runtime, *, bytes_=0):
    snapshot = runtime.resource_snapshot()
    assert snapshot["queries"] == {}
    assert snapshot["request_admission"]["active_requests"] == 0
    assert snapshot["request_admission"]["queued_requests"] == 0
    assert snapshot["result_delivery"]["active_results"] == 0
    assert snapshot["result_delivery"]["usage_bytes"] == bytes_


def source_view(connection, monkeypatch, batches):
    import builtins

    from vane.datasource import DataSource, DataSourceTask

    monkeypatch.setattr(builtins, "_vane_p1_query_batches", batches, raising=False)

    class Task(DataSourceTask):
        def execute(self):
            import builtins

            return builtins._vane_p1_query_batches()

    class Source(DataSource):
        @property
        def schema(self):
            return {"x": "BIGINT"}

        def get_tasks(self):
            yield Task()

    connection.from_datasource(Source()).create_view("input")


@pytest.mark.parametrize("runner", ["ray", "local", "local-fast", "invalid-runner"])
def test_local_query_uses_native_execution_without_model_or_fragment_runtime(monkeypatch, runner):
    from vane.execution import compiler, submission, udf_local_model, udf_local_request

    monkeypatch.setenv("VANE_RUNNER", runner)

    def forbidden(*args, **kwargs):
        raise AssertionError("local query entered a model request or distributed planner")

    monkeypatch.setattr(compiler, "compile_fragment_graph", forbidden)
    monkeypatch.setattr(submission, "prepare_ray_query", forbidden)
    monkeypatch.setattr(udf_local_model.LocalModelRuntime, "__init__", forbidden)
    monkeypatch.setattr(udf_local_request.LocalModelRequest, "__init__", forbidden)
    with vane.connect(backend="local") as connection:
        with connection.query("SELECT sum(i) AS n FROM range(10) t(i)") as result:
            assert isinstance(result, vane.QueryResult)
            assert isinstance(result.context, QueryContext)
            assert result.query_id == result.context.query_id
            assert result.collect().to_pylist() == [{"n": 45}]
            assert result.execution_state == "SUCCEEDED"
        idle(connection.query_runtime)


@pytest.mark.parametrize(
    ("sql", "parameters", "expected"),
    [
        ("SELECT 42 AS x", None, [{"x": 42}]),
        ("SELECT ?::BIGINT AS x", [7], [{"x": 7}]),
        ("SELECT $x::BIGINT AS x", {"x": 9}, [{"x": 9}]),
        ("SELECT [1, 2, NULL] AS x, {'a': 'ok'} AS s", None, [{"x": [1, 2, None], "s": {"a": "ok"}}]),
        ("SELECT NULL::BIGINT AS x", None, [{"x": None}]),
        ("SELECT 1 AS x WHERE FALSE", None, []),
    ],
)
def test_native_query_parameters_schema_and_empty_result(sql, parameters, expected):
    with vane.connect(backend="local") as connection:
        with connection.query(sql, parameters) as result:
            assert result.schema is not None
            table = result.collect()
            assert table.schema == result.schema
            assert table.to_pylist() == expected
            assert result.completion_status == ("ok" if expected else "empty")
            assert result.state == "delivered"
        idle(connection.query_runtime)


def test_module_query_uses_the_same_result_contract():
    with vane.connect(backend="local") as connection:
        with vane.query("SELECT $x::BIGINT AS x", {"x": 42}, connection=connection) as result:
            assert isinstance(result, vane.QueryResult)
            assert result.collect().to_pylist() == [{"x": 42}]
        with pytest.raises(ValueError, match="execution"):
            vane.query("SELECT 1", connection=connection, execution="pipelined")
        idle(connection.query_runtime)
    with vane.connect() as unselected:
        with pytest.raises(vane.InvalidInputException, match="backend='local'"):
            unselected.query("SELECT 1")


@pytest.mark.parametrize("value", [None, "pipelined", "fte"])
def test_local_rejects_any_execution_override(value):
    with pytest.raises(ValueError, match="execution"):
        vane.connect(backend="local", execution=value)
    with vane.connect(backend="local") as connection:
        with pytest.raises(ValueError, match="execution"):
            connection.query("SELECT 1", execution=value)
        idle(connection.query_runtime)


@pytest.mark.parametrize("batch_size", [True, 0, -1, 2**32, 1.5])
def test_batch_size_validation_does_not_start_execution(batch_size):
    with vane.connect(backend="local") as connection:
        with pytest.raises(ValueError, match="rows_per_batch"):
            connection.query("SELECT 1", rows_per_batch=batch_size)
        idle(connection.query_runtime)


def test_collect_larger_than_window_copies_before_fetching_next_batch():
    with vane.connect(backend="local", resources=limits()) as connection:
        with connection.query("SELECT i FROM range(20000) t(i)") as result:
            first = result.read_batch()
            assert isinstance(first, pa.RecordBatch)
            assert first.num_rows == 2048
            del first
            table = result.collect()
            assert table.column(0).to_pylist() == list(range(2048, 20000))
            idle(connection.query_runtime)
        assert table.num_rows == 20000 - 2048


def test_retained_numpy_view_applies_backpressure_and_survives_result_close():
    with vane.connect(backend="local", resources=limits()) as connection, ThreadPoolExecutor(1) as threads:
        runtime = connection.query_runtime
        result = connection.query("SELECT i FROM range(10000) t(i)")
        batch = result.read_batch()
        view = batch.column(0).to_numpy(zero_copy_only=True)
        del batch
        future = threads.submit(result.read_batch)
        wait(lambda: runtime.resource_snapshot()["result_delivery"]["waiting_byte_results"] == 1)
        assert not future.done()
        assert view.tolist() == list(range(2048))
        del view
        batch = future.result(timeout=5)
        result.close()
        usage = runtime.resource_snapshot()["result_delivery"]["usage_bytes"]
        assert usage > 0
        idle(runtime, bytes_=usage)
        assert batch.column(0).to_pylist() == list(range(2048, 4096))
        assert result.execution_state == "CANCELED"
        del batch, future
        gc.collect()
        idle(runtime)


def test_arrow_slice_retains_lease_after_connection_close():
    connection = vane.connect(backend="local", resources=limits())
    runtime = connection.query_runtime
    result = connection.query("SELECT i FROM range(10000) t(i)")
    batch = next(result)
    view = batch.slice(10, 4).column(0)
    del batch
    connection.close()
    assert view.to_pylist() == [10, 11, 12, 13]
    usage = runtime.resource_snapshot()["result_delivery"]["usage_bytes"]
    assert usage > 0
    idle(runtime, bytes_=usage)
    del view
    gc.collect()
    idle(runtime)


def test_queued_query_reserves_result_slot_only_after_admission():
    with vane.connect(backend="local", resources=limits()) as connection, ThreadPoolExecutor(1) as threads:
        runtime = connection.query_runtime
        child = connection.cursor()
        assert child.query_runtime is runtime
        first = connection.query("SELECT i FROM range(10000) t(i)")
        future = threads.submit(child.query, "SELECT 9 AS x")
        wait(lambda: runtime.resource_snapshot()["request_admission"]["queued_requests"] == 1)
        assert runtime.resource_snapshot()["result_delivery"]["active_results"] == 1
        first.close()
        with future.result(timeout=5) as second:
            assert second.collect().to_pylist() == [{"x": 9}]
        idle(runtime)


def test_admission_timeout_retires_unstarted_query():
    with vane.connect(backend="local", resources=limits()) as connection:
        child = connection.cursor()
        with connection.query("SELECT 1"):
            with pytest.raises(RequestQueueTimeout):
                child.query("SELECT 2", options=options(admission=0.02))
        idle(connection.query_runtime)


def test_full_result_slot_refuses_before_native_execution():
    with vane.connect(backend="local", resources=limits(active=2)) as connection:
        connection.execute("CREATE SEQUENCE query_sequence START 1")
        child = connection.cursor()
        with connection.query("SELECT 1"):
            with pytest.raises(ResultDeliveryFull) as caught:
                child.query("SELECT nextval('query_sequence')")
            assert caught.value.execution_started is False
            assert child.execute("SELECT nextval('query_sequence')").fetchone() == (1,)
        idle(connection.query_runtime)


@pytest.mark.parametrize("timeout", ["execution", "delivery"])
def test_deadline_wakes_result_byte_waiter(timeout):
    with vane.connect(backend="local", resources=limits()) as connection, ThreadPoolExecutor(1) as threads:
        result = connection.query("SELECT i FROM range(10000) t(i)", options=options(**{timeout: 0.2}))
        batch = result.read_batch()
        future = threads.submit(result.read_batch)
        error = RequestExecutionTimeout if timeout == "execution" else ResultDeliveryTimeout
        with pytest.raises(error):
            future.result(timeout=5)
        result.close()
        del batch, future
        gc.collect()
        idle(connection.query_runtime)


def test_execution_deadline_interrupts_native_computation_and_cursor_is_reusable():
    with vane.connect(backend="local") as connection:
        with pytest.raises(RequestExecutionTimeout):
            connection.query("SELECT sum(i * 0.1) FROM range(10000000000) t(i)", options=options(execution=0.03))
        idle(connection.query_runtime)
        assert connection.query("SELECT 7 AS x").collect().to_pylist() == [{"x": 7}]


def test_interrupt_wakes_byte_waiter_and_fences_later_query():
    with vane.connect(backend="local", resources=limits()) as connection, ThreadPoolExecutor(1) as threads:
        result = connection.query("SELECT i FROM range(10000) t(i)")
        batch = result.read_batch()
        future = threads.submit(result.read_batch)
        wait(lambda: connection.query_runtime.resource_snapshot()["result_delivery"]["waiting_byte_results"] == 1)
        connection.interrupt()
        with pytest.raises(RequestCancelled):
            future.result(timeout=5)
        result.close()
        del batch, future
        gc.collect()
        idle(connection.query_runtime)
        later = connection.query("SELECT 9 AS x")
        assert result.context.cancel() is False
        assert later.collect().to_pylist() == [{"x": 9}]


def test_cleanup_failure_retains_admission_and_retries(monkeypatch):
    with vane.connect(backend="local", resources=limits()) as connection:
        runtime = connection.query_runtime
        result = connection.query("SELECT i FROM range(10000) t(i)")
        close = result.context._close_native
        attempts = []

        def fail_once(retire):
            attempts.append(retire)
            if len(attempts) == 1:
                raise RuntimeError("injected cleanup failure")
            close(retire)

        monkeypatch.setattr(result.context, "_close_native", fail_once)
        with pytest.raises(RuntimeError, match="cleanup"):
            result.close()
        assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 1
        assert runtime.resource_snapshot()["result_delivery"]["cleanup_pending_results"] == 1
        result.close()
        idle(runtime)
        assert connection.query("SELECT 1 AS x").collect().to_pylist() == [{"x": 1}]


def test_startup_reader_failure_releases_query_and_cursor(monkeypatch):
    original = QueryContext.install_reader

    def fail(*args):
        raise RuntimeError("reader startup failed")

    with vane.connect(backend="local") as connection:
        monkeypatch.setattr(QueryContext, "install_reader", fail)
        with pytest.raises(RuntimeError, match="reader startup failed"):
            connection.query("SELECT 1")
        idle(connection.query_runtime)
        monkeypatch.setattr(QueryContext, "install_reader", original)
        assert connection.query("SELECT 2 AS x").collect().to_pylist() == [{"x": 2}]


def test_failed_sql_releases_query_and_preserves_later_execution():
    with vane.connect(backend="local") as connection:
        with pytest.raises(vane.BinderException):
            connection.query("SELECT missing_column")
        idle(connection.query_runtime)
        assert connection.query("SELECT 3 AS x").collect().to_pylist() == [{"x": 3}]


def test_active_result_rejects_same_cursor_reentry():
    with vane.connect(backend="local") as connection:
        with connection.query("SELECT i FROM range(10000) t(i)"):
            with pytest.raises(vane.InvalidInputException, match="reentrant"):
                connection.query("SELECT 1")
            with pytest.raises(vane.InvalidInputException, match="reentrant"):
                connection.execute("SELECT 1")
        idle(connection.query_runtime)


def test_parquet_query_supports_native_aggregate_and_join(tmp_path):
    import pyarrow.parquet as pq

    path = tmp_path / "input.parquet"
    pq.write_table(pa.table({"x": [1, 2, 3]}), path)
    sql = f"SELECT sum(a.x + b.x) AS n FROM read_parquet('{path}') a JOIN range(4) b(x) USING (x)"
    with vane.connect(backend="local") as connection:
        expected = connection.execute(sql).fetchone()[0]
        assert connection.query(sql).collect().to_pylist() == [{"n": expected}]
        idle(connection.query_runtime)


def test_first_batch_arrives_before_source_completion(monkeypatch):
    gate = threading.Event()
    completed = threading.Event()

    def batches():
        # Native streaming prefetches up to its own buffer threshold. Supply
        # enough chunks to fill that buffer, then keep the source unfinished.
        yield pa.record_batch({"x": list(range(8192))})
        assert gate.wait(5)
        completed.set()
        yield pa.record_batch({"x": list(range(8192, 10240))})

    with vane.connect(backend="local", config={"threads": 1}) as connection:
        connection.execute("SET streaming_buffer_size = '8KB'")
        source_view(connection, monkeypatch, batches)
        try:
            with connection.query("SELECT * FROM input", rows_per_batch=1024) as result:
                batch = result.read_batch()
                assert batch.num_rows == 1024
                assert not completed.is_set()
                del batch
                gate.set()
                assert result.collect().num_rows == 9216
        finally:
            gate.set()
        idle(connection.query_runtime)


def test_native_failure_after_partial_result_is_not_eof():
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        connection.execute("SET streaming_buffer_size = '8KB'")
        result = connection.query(
            "SELECT CASE WHEN i >= 8192 THEN error('late failure') ELSE i END AS x FROM range(10000) t(i)"
        )
        batch = result.read_batch()
        assert batch.column(0).to_pylist() == list(range(2048))
        del batch
        with pytest.raises(Exception, match="late failure"):
            result.collect()
        assert result.execution_state == "FAILED"
        assert result.state == "failed"
        idle(connection.query_runtime)
        assert connection.query("SELECT 8 AS x").collect().to_pylist() == [{"x": 8}]


def test_oversized_batch_refuses_without_retaining_buffers():
    with vane.connect(backend="local", resources=limits(size=64)) as connection:
        result = connection.query("SELECT i FROM range(10) t(i)")
        with pytest.raises(ResultDeliveryFull) as caught:
            result.read_batch()
        assert caught.value.execution_started is True
        assert result.execution_state == "FAILED"
        result.close()
        idle(connection.query_runtime)


@pytest.mark.parametrize("failure", ["deadline", "buffer"])
def test_allocation_and_watcher_failures_keep_cleanup_owner(monkeypatch, failure):
    from vane.execution.request_deadline import MonotonicDeadline

    def fail(*args):
        raise RuntimeError("injected preparation failure")

    with vane.connect(backend="local") as connection:
        with monkeypatch.context() as patch:
            if failure == "deadline":
                patch.setattr(MonotonicDeadline, "start", fail)
                with pytest.raises(RuntimeError, match="injected preparation failure"):
                    connection.query("SELECT 1")
            else:
                patch.setattr(vane.BatchLease, "build", fail)
                result = connection.query("SELECT 1")
                with pytest.raises(RuntimeError, match="injected preparation failure"):
                    result.read_batch()
                result.close()
        idle(connection.query_runtime)
        assert connection.query("SELECT 2 AS x").collect().to_pylist() == [{"x": 2}]


def test_result_cancel_is_idempotent_and_preserves_retained_batch():
    with vane.connect(backend="local", resources=limits()) as connection:
        result = connection.query("SELECT i FROM range(10000) t(i)")
        batch = result.read_batch()
        assert result.cancel() is True
        assert result.cancel() is False
        with pytest.raises(ResultDeliveryCancelled):
            result.read_batch()
        assert result.execution_state == "CANCELED"
        assert batch.num_rows == 2048
        del batch
        gc.collect()
        idle(connection.query_runtime)


def test_connection_close_wakes_active_and_queued_queries():
    connection = vane.connect(backend="local", resources=limits())
    runtime = connection.query_runtime
    child = connection.cursor()
    result = connection.query("SELECT i FROM range(10000) t(i)")
    batch = result.read_batch()
    with ThreadPoolExecutor(2) as threads:
        reading = threads.submit(result.read_batch)
        queued = threads.submit(child.query, "SELECT 1")
        wait(lambda: runtime.resource_snapshot()["request_admission"]["queued_requests"] == 1)
        wait(lambda: runtime.resource_snapshot()["result_delivery"]["waiting_byte_results"] == 1)
        connection.close()
        with pytest.raises(Exception, match="closed|cancelled"):
            reading.result(timeout=5)
        with pytest.raises(RequestCancelled):
            queued.result(timeout=5)
    assert batch.num_rows == 2048
    del batch, reading, queued
    gc.collect()
    idle(runtime)


@pytest.mark.parametrize("target", ["same", "sibling", "module"])
def test_input_callback_rejects_query_before_connection_lock(monkeypatch, target):
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        child = connection.cursor()
        called = []

        def batches():
            selected = connection if target == "same" else child
            with pytest.raises(vane.InvalidInputException, match="Python input callback"):
                if target == "module":
                    vane.query("SELECT 42", connection=selected)
                else:
                    selected.query("SELECT 42")
            called.append(True)
            yield pa.record_batch({"x": [7]})

        source_view(connection, monkeypatch, batches)
        assert connection.query("SELECT * FROM input").collect().to_pylist() == [{"x": 7}]
        assert called == [True]
        assert child.query("SELECT 9 AS x").collect().to_pylist() == [{"x": 9}]
        idle(connection.query_runtime)


def test_local_options_snapshot_and_mode_validation():
    from vane.execution.query_options import RayExecution

    snapshot = options()
    with vane.connect(backend="local") as connection:
        result = connection.query("SELECT 1", options=snapshot)
        assert result.context.options is snapshot
        result.close()
        with pytest.raises(ValueError, match="LocalExecution"):
            connection.query("SELECT 1", options=vane.QueryExecutionOptions(RayExecution(), 1, 1, 1))
        idle(connection.query_runtime)
    with pytest.raises(ValueError, match="backend"):
        vane.connect(backend="ray")
    with pytest.raises(TypeError, match="QueryResources"):
        vane.connect(backend="local", resources={})


def test_query_requires_single_read_only_select_and_auto_commit():
    with vane.connect(backend="local") as connection:
        for sql in ("", "SELECT 1; SELECT 2", "CREATE TABLE t AS SELECT 1"):
            with pytest.raises(vane.InvalidInputException, match="one SELECT"):
                connection.query(sql)
        connection.execute("CREATE SEQUENCE s START 1")
        with pytest.raises(vane.InvalidInputException, match="read-only"):
            connection.query("SELECT nextval('s')")
        assert connection.execute("SELECT nextval('s')").fetchone() == (1,)
        connection.execute("BEGIN")
        with pytest.raises(vane.InvalidInputException, match="auto-commit"):
            connection.query("SELECT 1")
        connection.execute("ROLLBACK")
        idle(connection.query_runtime)
