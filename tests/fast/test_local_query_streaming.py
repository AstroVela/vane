# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import gc
import os
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

import vane
from vane.datasource import DataSource, DataSourceTask
from vane.execution.request_admission import RequestAdmissionLimits, RequestCancelled, RequestExecutionTimeout
from vane.execution.resources import ResourceVector
from vane.execution.result_delivery import (
    ResultDeliveryCancelled,
    ResultDeliveryClosed,
    ResultDeliveryFull,
    ResultDeliveryLimits,
    ResultDeliveryTimeout,
)


@pytest.fixture(autouse=True)
def native_environment(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")


def runtime(connection, *, size=20_000, **options):
    return connection.configure_local_runtime(
        request_limit=RequestAdmissionLimits(1, 4),
        result_limit=ResultDeliveryLimits(2, size),
        **options,
    )


def execute(connection, api, sql="SELECT i FROM range(10000) t(i)", **options):
    if api == "sql":
        return connection.execute_result(sql, stream=True, rows_per_batch=2048, **options)
    return connection.sql(sql).execute_result(stream=True, rows_per_batch=2048, **options)


def wait(predicate):
    deadline = time.monotonic() + 10
    while not predicate():
        assert time.monotonic() < deadline, "stream did not reach its expected phase"
        time.sleep(0.01)


def register_source(connection, batches, monkeypatch):
    # Serialized tasks must use the caller's iterator rather than a pickled
    # copy of its callback state. Vane owns and guards this DataSource stream.
    monkeypatch.setattr(builtins, "_vane_stream_test_batches", batches, raising=False)

    class Task(DataSourceTask):
        def execute(self):
            import builtins

            return builtins._vane_stream_test_batches()

    class Source(DataSource):
        @property
        def schema(self):
            return {"x": "BIGINT"}

        def get_tasks(self):
            yield Task()

    connection.from_datasource(Source()).create_view("source")


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_stream_reads_larger_than_delivery_budget_one_batch_at_a_time(api):
    with vane.connect() as connection:
        state = runtime(connection)
        result = execute(connection, api)
        assert result.result_schema == {"names": ["i"], "types": ["BIGINT"]}
        assert state.resource_snapshot()["result_delivery"]["usage_bytes"] == 0
        assert state.resource_snapshot()["request_admission"]["active_requests"] == 1
        rows = []
        while True:
            try:
                table = result.take()
            except StopIteration:
                break
            assert 0 < table.num_rows <= 2048
            rows.extend(table.column(0).to_pylist())
            assert 0 < state.resource_snapshot()["result_delivery"]["usage_bytes"] <= 20_000
            del table
        assert rows == list(range(10000))
        assert result.completion_status == "ok"
        snapshot = state.resource_snapshot()
        assert snapshot["request_admission"]["active_requests"] == 0
        assert snapshot["result_delivery"]["active_results"] == 0
        assert snapshot["result_delivery"]["usage_bytes"] == 0
        assert connection.execute("SELECT 7").fetchone() == (7,)


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_retained_view_blocks_next_batch_and_releases_capacity(api):
    with vane.connect() as connection, ThreadPoolExecutor(1) as threads:
        state = runtime(connection)
        result = execute(connection, api)
        table = result.take()
        view = table.column(0).chunk(0).to_numpy(zero_copy_only=True)
        del table
        future = threads.submit(result.take)
        wait(lambda: state.resource_snapshot()["result_delivery"]["waiting_byte_results"] == 1)
        with pytest.raises(FutureTimeoutError):
            future.result(timeout=0.05)
        assert view.tolist() == list(range(2048))
        del view
        table = future.result(timeout=10)
        assert table.column(0).to_pylist() == list(range(2048, 4096))
        result.close()
        snapshot = state.resource_snapshot()
        assert snapshot["request_admission"]["active_requests"] == 0
        assert snapshot["result_delivery"]["active_results"] == 0
        assert snapshot["result_delivery"]["exported_bytes"] > 0
        assert table.num_rows == 2048
        del table, future
        gc.collect()
        assert state.resource_snapshot()["result_delivery"]["usage_bytes"] == 0


@pytest.mark.parametrize("api", ["sql", "relation"])
@pytest.mark.timeout(20)
def test_delivery_deadline_wakes_a_byte_waiter(api):
    with vane.connect() as connection, ThreadPoolExecutor(1) as threads:
        state = runtime(connection)
        result = execute(connection, api, delivery_timeout=0.3)
        table = result.take()
        future = threads.submit(result.take)
        with pytest.raises(ResultDeliveryTimeout):
            future.result(timeout=10)
        result.close()
        assert state.resource_snapshot()["request_admission"]["active_requests"] == 0
        assert table.num_rows == 2048
        del table, future
        gc.collect()
        assert state.resource_snapshot()["result_delivery"]["usage_bytes"] == 0


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_idle_stream_execution_expiry_returns_request_capacity(api):
    with vane.connect() as connection:
        state = runtime(connection, execution_timeout=0.15)
        result = execute(connection, api)
        wait(lambda: state.resource_snapshot()["request_admission"]["active_requests"] == 0)
        with pytest.raises(RequestExecutionTimeout):
            result.take()
        result.close()
        assert connection.execute("SELECT 42").fetchone() == (42,)


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_runtime_close_cancels_stream_before_waiting_for_admission(api):
    with vane.connect() as connection:
        state = runtime(connection)
        result = execute(connection, api)
        table = result.take()
        state.close(timeout=2)
        assert state.resource_snapshot()["request_admission"]["active_requests"] == 0
        assert state.resource_snapshot()["result_delivery"]["active_results"] == 0
        assert table.num_rows == 2048
        del table
        result.close()
        assert state.resource_snapshot()["result_delivery"]["usage_bytes"] == 0


@pytest.mark.parametrize("api", ["sql", "relation"])
@pytest.mark.parametrize("phase", ["before_source", "before_ready"])
@pytest.mark.parametrize("close_times_out", [False, True])
@pytest.mark.timeout(20)
def test_runtime_close_fences_streams_still_being_prepared(monkeypatch, api, phase, close_times_out):
    from vane.execution import local_result_delivery

    entered, proceed, fenced = threading.Event(), threading.Event(), threading.Event()
    prepare = local_result_delivery.prepare_native_query_stream

    def pause():
        entered.set()
        assert proceed.wait(5)

    def prepare_stream(*args):
        if phase == "before_source":
            pause()
        source = prepare(*args)
        if phase == "before_ready":
            pause()
        return source

    monkeypatch.setattr(local_result_delivery, "prepare_native_query_stream", prepare_stream)
    with vane.connect() as connection, ThreadPoolExecutor(2) as clients:
        state = runtime(connection)
        delivery = state._runtime._result_delivery
        cancel_streams = delivery.cancel_streams

        def fence():
            cancel_streams()
            fenced.set()

        monkeypatch.setattr(delivery, "cancel_streams", fence)
        pending = clients.submit(execute, connection, api)
        try:
            assert entered.wait(5)
            snapshot = state.resource_snapshot()
            assert snapshot["request_admission"]["active_requests"] == 1
            assert snapshot["result_delivery"]["preparing_results"] == 1
            assert snapshot["result_delivery"]["streaming_results"] == int(phase == "before_ready")
            closing = clients.submit(state.close, timeout=0 if close_times_out else 3)
            assert fenced.wait(5)
            if close_times_out:
                with pytest.raises(TimeoutError):
                    closing.result(timeout=5)
            else:
                with pytest.raises(FutureTimeoutError):
                    closing.result(timeout=0.05)
        finally:
            proceed.set()
        try:
            result = pending.result(timeout=5)
        except (RequestCancelled, ResultDeliveryClosed):
            pass
        else:
            result.close()
            pytest.fail("runtime close allowed a newly prepared stream to reach the consumer")
        if not close_times_out:
            closing.result(timeout=5)
        # Even a timed-out close must fence late streams without requiring
        # the caller to close a result it never received or retry the runtime.
        snapshot = state.resource_snapshot()
        assert snapshot["request_admission"]["active_requests"] == 0
        assert snapshot["result_delivery"]["active_results"] == 0
        assert snapshot["result_delivery"]["usage_bytes"] == 0
        state.close()


@pytest.mark.parametrize("api", ["sql", "relation"])
@pytest.mark.timeout(20)
def test_late_stream_keeps_admission_until_failed_native_cleanup_is_retried(monkeypatch, api):
    from vane.execution import local_result_delivery

    entered, proceed, fault = threading.Event(), threading.Event(), threading.Event()
    fault.set()
    prepare = local_result_delivery.prepare_native_query_stream

    def prepare_stream(*args):
        entered.set()
        assert proceed.wait(5)
        source = prepare(*args)
        close_native = source._close_native

        def close(retire):
            if fault.is_set():
                raise RuntimeError("injected late stream cleanup failure")
            close_native(retire)

        source._close_native = close
        return source

    monkeypatch.setattr(local_result_delivery, "prepare_native_query_stream", prepare_stream)
    with vane.connect() as connection, ThreadPoolExecutor(1) as clients:
        state = runtime(connection)
        pending = clients.submit(execute, connection, api)
        try:
            assert entered.wait(5)
            with pytest.raises(TimeoutError):
                state.close()
        finally:
            proceed.set()
        try:
            with pytest.raises(RequestCancelled) as failure:
                pending.result(timeout=5)
            assert "result cleanup failed" in str(failure.value.__cause__)
            snapshot = state.resource_snapshot()
            assert snapshot["request_admission"]["active_requests"] == 1
            assert snapshot["result_delivery"]["cleanup_pending_results"] == 1
            with pytest.raises(RuntimeError, match="stream cleanup failed"):
                state.close()
        finally:
            fault.clear()
            state.close(timeout=3)
        snapshot = state.resource_snapshot()
        assert snapshot["request_admission"]["active_requests"] == 0
        assert snapshot["result_delivery"]["active_results"] == 0
        assert snapshot["result_delivery"]["usage_bytes"] == 0


@pytest.mark.timeout(20)
def test_runtime_close_waits_for_the_consumer_after_native_cleanup(monkeypatch):
    with vane.connect() as connection, ThreadPoolExecutor(2) as clients:
        state = runtime(connection)
        result = execute(connection, "sql", "SELECT i FROM range(0) t(i)")
        source = result._stream
        close = source.close
        retired = threading.Event()
        proceed = threading.Event()

        def hold():
            close()
            if not retired.is_set():
                retired.set()
                assert proceed.wait(5)

        monkeypatch.setattr(source, "close", hold)
        consumer = clients.submit(result.take)
        try:
            assert retired.wait(5)
            assert state.resource_snapshot()["request_admission"]["active_requests"] == 1
            closing = clients.submit(state.close, timeout=3)
            with pytest.raises(FutureTimeoutError):
                closing.result(timeout=0.05)
        finally:
            proceed.set()
        closing.result(timeout=5)
        from vane.execution.result_delivery import ResultDeliveryClosed

        with pytest.raises(ResultDeliveryClosed):
            consumer.result(timeout=5)
        assert state.resource_snapshot()["result_delivery"]["active_results"] == 0


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_empty_stream_finishes_at_eof_without_buffer_reservations(api):
    with vane.connect() as connection:
        state = runtime(connection)
        result = execute(connection, api, "SELECT 42::BIGINT AS x WHERE false")
        assert list(result) == []
        assert result.completion_status == "empty"
        assert state.resource_snapshot()["request_admission"]["active_requests"] == 0
        assert state.resource_snapshot()["result_delivery"]["usage_bytes"] == 0


@pytest.mark.parametrize("api", ["sql", "relation"])
@pytest.mark.parametrize("phase", ["read", "cleanup"])
@pytest.mark.parametrize("overdue", [False, True])
def test_eof_checks_delivery_deadline_even_when_watcher_is_delayed(monkeypatch, api, phase, overdue):
    from vane.execution import request_deadline, result_delivery

    now = [10.0]
    timer = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(result_delivery, "time", timer)
    monkeypatch.setattr(request_deadline, "time", timer)
    monkeypatch.setattr(request_deadline.MonotonicDeadline, "start", lambda self: None)
    with vane.connect() as connection:
        state = runtime(connection)
        result = execute(connection, api, "SELECT 42 AS x WHERE false", delivery_timeout=1)
        source = result._stream
        method = "_read_native" if phase == "read" else "close"
        operation = getattr(source, method)

        def delayed():
            try:
                return operation()
            finally:
                now[0] = 12.0 if overdue else 10.5

        monkeypatch.setattr(source, method, delayed)
        with pytest.raises(ResultDeliveryTimeout if overdue else StopIteration):
            result.take()
        result.close()
        snapshot = state.resource_snapshot()
        assert snapshot["request_admission"]["active_requests"] == 0
        delivery = snapshot["result_delivery"]
        assert delivery["active_results"] == delivery["usage_bytes"] == 0
        assert delivery["timed_out_results"] == int(overdue)
        assert delivery["delivered_results"] == int(not overdue)


@pytest.mark.parametrize("api", ["sql", "relation"])
@pytest.mark.parametrize("termination", ["eof", "close", "cancel"])
@pytest.mark.timeout(20)
def test_stream_retires_result_slot_before_waking_queued_request(monkeypatch, api, termination):
    with vane.connect() as connection:
        state = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(1, 4), result_limit=ResultDeliveryLimits(1, 20_000)
        )
        with connection.cursor() as first, connection.cursor() as second, ThreadPoolExecutor(2) as clients:
            result = execute(first, api, "SELECT 42 AS x WHERE false")
            source = result._stream
            close = source.close
            cleaned = threading.Event()
            proceed = threading.Event()

            def hold():
                close()
                if not cleaned.is_set():
                    cleaned.set()
                    assert proceed.wait(5)

            monkeypatch.setattr(source, "close", hold)
            following = clients.submit(execute, second, api, "SELECT 7 AS x")
            wait(lambda: state.resource_snapshot()["request_admission"]["queued_requests"] == 1)
            consumer = clients.submit(result.take if termination == "eof" else getattr(result, termination))
            try:
                assert cleaned.wait(5)
                assert state.resource_snapshot()["request_admission"]["active_requests"] == 1
                # Even a direct cleanup retry cannot release this request while
                # the stream still owns its result admission slot.
                source._query.request.shutdown()
                with pytest.raises(FutureTimeoutError):
                    following.result(timeout=0.05)
            finally:
                proceed.set()
            if termination == "eof":
                with pytest.raises(StopIteration):
                    consumer.result(timeout=5)
            else:
                consumer.result(timeout=5)
            result.close()
            with following.result(timeout=5) as next_result:
                table = next_result.take()
                assert table.column(0).to_pylist() == [7]
                del table
            snapshot = state.resource_snapshot()
            assert snapshot["request_admission"]["active_requests"] == 0
            assert snapshot["result_delivery"]["active_results"] == 0
            assert snapshot["result_delivery"]["rejected_results"] == 0


def test_stream_retirement_failure_retains_both_admission_owners(monkeypatch):
    with vane.connect() as connection:
        state = runtime(connection)
        result = execute(connection, "sql")
        source = result._stream
        retire = source.retire

        def fail(_release_result):
            raise RuntimeError("injected result retirement failure")

        monkeypatch.setattr(source, "retire", fail)
        with pytest.raises(RuntimeError, match="retirement failure"):
            result.close()
        source._query.request.shutdown()
        snapshot = state.resource_snapshot()
        assert snapshot["request_admission"]["active_requests"] == 1
        assert snapshot["result_delivery"]["cleanup_pending_results"] == 1
        monkeypatch.setattr(source, "retire", retire)
        result.close()
        snapshot = state.resource_snapshot()
        assert snapshot["request_admission"]["active_requests"] == 0
        assert snapshot["result_delivery"]["active_results"] == 0


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_single_oversized_batch_is_not_retried(api):
    with vane.connect() as connection:
        state = runtime(connection, size=100)
        result = execute(connection, api)
        with pytest.raises(ResultDeliveryFull) as error:
            result.take()
        assert error.value.reason == "bytes"
        assert error.value.execution_started is True
        result.close()
        assert state.resource_snapshot()["request_admission"]["active_requests"] == 0
        assert state.resource_snapshot()["result_delivery"]["usage_bytes"] == 0


def test_stream_preserves_caller_replacement_scan():
    with vane.connect() as connection:
        runtime(connection)
        items = pa.table({"x": [7, 8]})
        result = connection.execute_result("SELECT * FROM items", stream=True)
        table = result.take()
        assert table.equals(items)
        del table
        assert list(result) == []


@pytest.mark.parametrize("retirement", [False, True])
def test_pending_native_cleanup_retains_request_and_result(monkeypatch, retirement):
    with vane.connect() as connection:
        state = runtime(connection)
        result = execute(connection, "sql")
        source = result._stream
        close = source._close_native

        def fail(retire):
            if retire is retirement:
                raise RuntimeError("injected native cleanup failure")
            close(retire)

        monkeypatch.setattr(source, "_close_native", fail)
        with pytest.raises(RuntimeError, match="result cleanup failed"):
            result.close()
        assert state.resource_snapshot()["request_admission"]["active_requests"] == 1
        assert state.resource_snapshot()["result_delivery"]["cleanup_pending_results"] == 1
        monkeypatch.setattr(source, "_close_native", close)
        result.close()
        assert state.resource_snapshot()["request_admission"]["active_requests"] == 0
        assert connection.execute("SELECT 7").fetchone() == (7,)


@pytest.mark.parametrize("api", ["sql", "relation"])
@pytest.mark.timeout(20)
def test_cancellation_wakes_byte_wait_and_keeps_exported_view(api):
    with vane.connect() as connection, ThreadPoolExecutor(1) as threads:
        state = runtime(connection)
        result = execute(connection, api)
        table = result.take()
        future = threads.submit(result.take)
        wait(lambda: state.resource_snapshot()["result_delivery"]["waiting_byte_results"] == 1)
        try:
            result.cancel()
        except RuntimeError as error:
            assert "cleanup is still in progress" in str(error)
        with pytest.raises(ResultDeliveryCancelled):
            future.result(timeout=10)
        result.close()
        assert table.num_rows == 2048
        assert state.resource_snapshot()["request_admission"]["active_requests"] == 0
        assert connection.execute("SELECT 7").fetchone() == (7,)
        del table, future
        gc.collect()
        assert state.resource_snapshot()["result_delivery"]["usage_bytes"] == 0


@pytest.mark.parametrize("reason", ["cancelled", "execution_timeout"])
@pytest.mark.timeout(20)
def test_recorded_cancellation_fences_handoff_before_notification(monkeypatch, reason):
    with vane.connect() as connection, ThreadPoolExecutor(1) as clients:
        state = runtime(connection)
        result = execute(connection, "sql")
        source = result._stream
        request = source._query.request
        check = source.check
        accepted = threading.Event()

        def record():
            check()
            if result._payloads and result._payloads[0]._owner is None and not accepted.is_set():
                # A watcher can be descheduled after recording cancellation,
                # before notifying its scope or the result handle.
                with request._lock:
                    assert request._begin_cancellation_locked(reason)
                accepted.set()

        monkeypatch.setattr(source, "check", record)
        future = clients.submit(result.take)
        try:
            assert accepted.wait(5)
            with pytest.raises(FutureTimeoutError):
                future.result(timeout=0.05)
        finally:
            if accepted.is_set():
                request._dispatch_cancellation(reason)
        error = RequestExecutionTimeout if reason == "execution_timeout" else RequestCancelled
        with pytest.raises(error):
            future.result(timeout=5)
        result.close()
        assert state.resource_snapshot()["request_admission"]["active_requests"] == 0
        assert state.resource_snapshot()["result_delivery"]["usage_bytes"] == 0
        assert connection.execute("SELECT 7").fetchone() == (7,)


@pytest.mark.parametrize("api", ["sql", "relation"])
@pytest.mark.parametrize("reason", ["cancelled", "execution_timeout"])
@pytest.mark.timeout(20)
def test_request_cancellation_after_batch_commit_retires_admission(monkeypatch, api, reason):
    with vane.connect() as connection:
        state = runtime(connection, execution_timeout=60 if reason == "execution_timeout" else None)
        with connection.cursor() as first, connection.cursor() as second, ThreadPoolExecutor(2) as clients:
            result = execute(first, api)
            request = result._stream._query.request
            cleanup = result._cleanup
            committed, proceed = threading.Event(), threading.Event()

            def pause_after_cleanup(*, consumer=False):
                cleanup(consumer=consumer)
                if consumer and not committed.is_set():
                    # The batch is committed and consumer cleanup has returned,
                    # but take() has not yet relinquished its consumer claim.
                    committed.set()
                    assert proceed.wait(5)

            monkeypatch.setattr(result, "_cleanup", pause_after_cleanup)
            consumer = clients.submit(result.take)
            following = clients.submit(execute, second, api, "SELECT 7 AS x")
            try:
                assert committed.wait(5)
                wait(lambda: state.resource_snapshot()["request_admission"]["queued_requests"] == 1)
                if reason == "execution_timeout":
                    monkeypatch.setattr(request._deadline, "expired", lambda: True)
                    request._expire_deadline()
                else:
                    assert request.cancel()
                assert result.state == "closing"
                assert state.resource_snapshot()["request_admission"]["active_requests"] == 1
            finally:
                proceed.set()
            table = consumer.result(timeout=5)
            try:
                assert result.state == "failed"
                next_result = following.result(timeout=5)
                assert next_result.take().column(0).to_pylist() == [7]
                assert list(next_result) == []
                error = RequestExecutionTimeout if reason == "execution_timeout" else RequestCancelled
                with pytest.raises(error):
                    result.take()
                snapshot = state.resource_snapshot()
                assert snapshot["request_admission"]["active_requests"] == 0
                assert snapshot["result_delivery"]["active_results"] == 0
                assert snapshot["result_delivery"]["usage_bytes"] > 0
                assert table.column(0).to_pylist() == list(range(2048))
            finally:
                # A failing regression must still unblock its queued control.
                result.close()
                following.result(timeout=5).close()
            del table, consumer
            gc.collect()
            assert state.resource_snapshot()["result_delivery"]["usage_bytes"] == 0


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_registered_model_reuse_and_shared_memory_cleanup(tmp_path, api):
    from vane.execution import ref_bundle
    from vane.execution.udf_data_admission import DataAdmissionLimits
    from vane.execution.udf_runtime_admission import TaskAdmissionLimits

    directory = str(tmp_path)

    class Model:
        def __init__(self):
            with Path(directory, "initializations").open("a") as output:
                output.write(f"{os.getpid()}\n")

        def __call__(self, values):
            return values

    with vane.connect() as connection:
        state = runtime(
            connection,
            resident_limit=ResourceVector(cpu=1),
            task_limit=TaskAdmissionLimits(1, 4),
            data_limit=DataAdmissionLimits(420_000, 140_000, 70_000),
        )
        model = state.register_model(
            "stream_model",
            vane.cls.batch(actor_number=1, return_dtype="BIGINT")(Model)(),
            version="v1",
            parameters=["BIGINT"],
        )
        vane.attach_function(model, "stream_encode", connection=connection)
        model.prewarm()
        for _ in range(2):
            result = execute(connection, api, "SELECT stream_encode(i) AS x FROM range(4096) t(i)")
            rows = []
            while True:
                try:
                    table = result.take()
                except StopIteration:
                    break
                rows.extend(table.column(0).to_pylist())
                del table
            assert rows == list(range(4096))
            assert state.resource_snapshot()["active_borrows"] == 0
            assert state.resource_snapshot()["data"]["usage_bytes"] == 0
        assert len(Path(directory, "initializations").read_text().splitlines()) == 1
        assert ref_bundle.local_shm_ref_budget_snapshot()["usage_bytes"] == 0


@pytest.mark.parametrize("boundary", ["projection", "sort", "aggregate", "join"])
@pytest.mark.timeout(30)
def test_stream_progresses_under_shared_udf_byte_wait(boundary):
    from vane.execution import ref_bundle
    from vane.execution.udf_data_admission import DataAdmissionLimits, DataAdmissionWaitLimits
    from vane.execution.udf_runtime_admission import TaskAdmissionLimits

    def produce(table):
        return pa.table({"blob": [b"x" * (65_536 + value) for value in table.column(0).to_pylist()]})

    def consume(table):
        return table

    with vane.connect(config={"threads": 2}) as connection:
        state = runtime(
            connection,
            task_limit=TaskAdmissionLimits(1, 8),
            data_limit=DataAdmissionLimits(420_000, 140_000, 70_000, wait=DataAdmissionWaitLimits(8, 5)),
        )
        relation = connection.sql("SELECT i FROM range(8) t(i)").map_batches(
            produce,
            schema={"blob": vane.sqltypes.BLOB},
            execution_backend="subprocess_task",
            batch_size=1,
            min_task_batch_size=1,
            task_input_max_bytes=8,
        )
        if boundary == "sort":
            relation = relation.order("octet_length(blob) DESC")
        elif boundary == "aggregate":
            relation = relation.aggregate("min(blob) AS blob, octet_length(blob)%2 AS k", "octet_length(blob)%2")
        elif boundary == "join":
            relation = relation.join(
                connection.sql("SELECT i AS k FROM range(65536, 65544) t(i)"), "octet_length(blob)=k"
            )
        relation = relation.project("octet_length(blob) AS n").map_batches(
            consume,
            schema={"n": vane.sqltypes.BIGINT},
            execution_backend="subprocess_task",
            batch_size=1,
            min_task_batch_size=1,
            task_input_max_bytes=8,
        )
        result = relation.execute_result(stream=True, rows_per_batch=1, delivery_timeout=10)
        rows = []
        with result:
            while True:
                try:
                    table = result.take()
                except StopIteration:
                    break
                rows.extend(table.column(0).to_pylist())
                del table
        expected = [65_536, 65_537] if boundary == "aggregate" else list(range(65_536, 65_544))
        assert sorted(rows) == expected
        snapshot = state.resource_snapshot()
        assert snapshot["active_borrows"] == snapshot["data"]["usage_bytes"] == 0
        assert snapshot["task_admission"]["running_tasks"] == 0
        assert snapshot["result_delivery"]["usage_bytes"] == 0
        assert ref_bundle.local_shm_ref_budget_snapshot()["usage_bytes"] == 0


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_open_stream_fences_cursor_queries_until_closed(api):
    with vane.connect() as connection:
        runtime(connection)
        result = execute(connection, api)
        with pytest.raises(vane.InvalidInputException, match="same cursor"):
            connection.execute("SELECT 1")
        result.close()
        assert connection.execute("SELECT 7").fetchone() == (7,)


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_connection_close_cancels_its_live_stream(api):
    connection = vane.connect()
    state = runtime(connection)
    result = execute(connection, api)
    table = result.take()
    connection.close()
    assert state.resource_snapshot()["request_admission"]["active_requests"] == 0
    assert state.resource_snapshot()["result_delivery"]["active_results"] == 0
    assert table.num_rows == 2048
    del table
    result.close()
    assert state.resource_snapshot()["result_delivery"]["usage_bytes"] == 0


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_stream_does_not_eagerly_exhaust_native_input(api, monkeypatch):
    fetched = []
    schema = pa.schema([("x", pa.int64())])

    def batches():
        for index in range(256):
            fetched.append(index)
            yield pa.record_batch([pa.array(range(index * 2048, (index + 1) * 2048))], schema=schema)

    with vane.connect(config={"threads": 1}) as connection:
        state = runtime(connection)
        register_source(connection, batches, monkeypatch)
        result = execute(connection, api, "SELECT x FROM source")
        assert len(fetched) < 256
        table = result.take()
        assert table.column(0).to_pylist() == list(range(2048))
        assert len(fetched) < 256
        result.close()
        assert state.resource_snapshot()["request_admission"]["active_requests"] == 0


@pytest.mark.timeout(20)
def test_connection_close_wakes_a_waiting_consumer():
    connection = vane.connect()
    state = runtime(connection)
    with ThreadPoolExecutor(1) as threads:
        result = execute(connection, "sql")
        table = result.take()
        future = threads.submit(result.take)
        wait(lambda: state.resource_snapshot()["result_delivery"]["waiting_byte_results"] == 1)
        connection.close()
        from vane.execution.result_delivery import ResultDeliveryClosed

        with pytest.raises(ResultDeliveryClosed):
            future.result(timeout=5)
        assert table.num_rows == 2048
        assert state.resource_snapshot()["request_admission"]["active_requests"] == 0
        assert state.resource_snapshot()["result_delivery"]["active_results"] == 0
        result.close()


def test_stream_cleanup_survives_connection_garbage_collection():
    connection = vane.connect()
    state = runtime(connection)
    result = execute(connection, "sql")
    owner = weakref.ref(connection)
    del connection
    gc.collect()
    assert owner() is None
    assert state.resource_snapshot()["request_admission"]["active_requests"] == 0
    assert state.resource_snapshot()["result_delivery"]["active_results"] == 0
    result.close()


def test_stream_setup_failure_cleans_native_execution_and_request(monkeypatch):
    from vane.execution import local_result_delivery

    original = local_result_delivery.prepare_native_query_stream

    def fail(*args):
        original(*args)
        raise OSError("injected Arrow export failure")

    with vane.connect() as connection:
        state = runtime(connection)
        monkeypatch.setattr(local_result_delivery, "prepare_native_query_stream", fail)
        with pytest.raises(OSError, match="Arrow export failure"):
            execute(connection, "sql")
        assert state.resource_snapshot()["request_admission"]["active_requests"] == 0
        assert state.resource_snapshot()["result_delivery"]["active_results"] == 0
        assert connection.execute("SELECT 7").fetchone() == (7,)


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_stream_options_validate_before_request_admission(api):
    with vane.connect() as connection:
        state = runtime(connection)
        with pytest.raises(vane.InvalidInputException, match="rows_per_batch"):
            if api == "sql":
                connection.execute_result("SELECT 7", stream=True, rows_per_batch=0)
            else:
                connection.sql("SELECT 7").execute_result(stream=True, rows_per_batch=0)
        assert state.resource_snapshot()["request_admission"]["active_requests"] == 0
        assert state.resource_snapshot()["result_delivery"]["active_results"] == 0
        assert connection.execute("SELECT 7").fetchone() == (7,)


def test_input_callback_cannot_consume_or_close_another_stream(monkeypatch):
    with vane.connect() as connection, vane.connect() as scanner:
        state = runtime(connection)
        result = execute(connection, "sql")
        schema = pa.schema([("x", pa.int64())])
        calls = []

        def batches():
            for operation in (result.take, result.cancel, result.close):
                with pytest.raises(vane.InvalidInputException, match="Python input callback"):
                    operation()
                calls.append(operation.__name__)
            yield pa.record_batch([pa.array([7])], schema=schema)

        register_source(scanner, batches, monkeypatch)
        assert scanner.sql("SELECT * FROM source").fetchall() == [(7,)]
        assert calls == ["take", "cancel", "close"]
        assert result.state == "ready"
        assert state.resource_snapshot()["request_admission"]["active_requests"] == 1
        result.close()


@pytest.mark.timeout(20)
def test_queued_stream_does_not_reserve_result_slot_before_execution():
    with vane.connect() as connection:
        state = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(1, 4), result_limit=ResultDeliveryLimits(1, 20_000)
        )
        with connection.cursor() as first, connection.cursor() as second, ThreadPoolExecutor(1) as clients:
            result = execute(first, "sql")
            future = clients.submit(execute, second, "sql", "SELECT 7::BIGINT AS x")
            try:
                wait(lambda: state.resource_snapshot()["request_admission"]["queued_requests"] == 1)
                assert state.resource_snapshot()["result_delivery"]["active_results"] == 1
                while True:
                    try:
                        table = result.take()
                    except StopIteration:
                        break
                    del table
                following = future.result(timeout=5)
                table = following.take()
                assert table.column(0).to_pylist() == [7]
                del table
                assert list(following) == []
                assert state.resource_snapshot()["request_admission"]["active_requests"] == 0
            finally:
                result.close()
                second.interrupt()
