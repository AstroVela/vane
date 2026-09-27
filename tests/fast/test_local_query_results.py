# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import os
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import vane
from vane.execution import local_result_delivery, ref_bundle
from vane.execution.request_admission import RequestAdmissionLimits, RequestCancelled, RequestExecutionTimeout
from vane.execution.resources import ResourceVector
from vane.execution.result_delivery import (
    ResultDeliveryCancelled,
    ResultDeliveryClosed,
    ResultDeliveryFull,
    ResultDeliveryLimits,
    ResultDeliveryTimeout,
)
from vane.execution.udf_data_admission import DataAdmissionLimits
from vane.execution.udf_runtime_admission import TaskAdmissionLimits


@pytest.fixture(autouse=True)
def native_environment(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 420_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    yield
    gc.collect()
    assert manager.snapshot()["usage_bytes"] == 0


def _runtime(connection, *, results=2, size=1_000_000, **options):
    return connection.configure_local_runtime(
        request_limit=RequestAdmissionLimits(1, 4),
        resident_limit=ResourceVector(cpu=1),
        result_limit=ResultDeliveryLimits(results, size),
        **options,
    )


def _execute(connection, api, sql="SELECT 42::BIGINT AS x", parameters=None, **options):
    if api == "sql":
        return connection.execute_result(sql, parameters, **options)
    return connection.sql(sql, params=parameters).execute_result(**options)


def _idle(runtime, *, results=0, bytes_=0):
    state = runtime.resource_snapshot()
    assert state["request_admission"]["active_requests"] == 0
    assert state["request_admission"]["queued_requests"] == 0
    assert state["active_borrows"] == 0
    assert state["result_delivery"]["active_results"] == results
    if bytes_ is not None:
        assert state["result_delivery"]["usage_bytes"] == bytes_


def _wait(predicate):
    deadline = time.monotonic() + 15
    while not predicate():
        assert time.monotonic() < deadline, "query did not reach its expected state"
        time.sleep(0.01)


def _model(connection, runtime, tmp_path, *, batch=False):
    directory = str(tmp_path)

    class Model:
        def __init__(self):
            with Path(directory, "initializations").open("a") as out:
                out.write(f"{os.getpid()}\n")

        def __call__(self, value):
            scalar = value[0].as_py() if batch else value
            with Path(directory, "calls").open("a") as out:
                out.write(f"{scalar}\n")
            if scalar == -1:
                Path(directory, "entered").touch()
                deadline = time.monotonic() + 20
                while not Path(directory, "release").exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("model gate was not released")
                    time.sleep(0.01)
            if scalar == -2:
                raise ValueError("managed query fixture failure")
            if scalar == -3:
                os._exit(23)
            return value

    decorate = vane.cls.batch if batch else vane.cls
    model = runtime.register_model(
        "encoder", decorate(actor_number=1, return_dtype="BIGINT")(Model)(), version="v1", parameters=["BIGINT"]
    )
    vane.attach_function(model, "managed_encode", connection=connection)
    return model


@pytest.mark.parametrize("api", ["sql", "relation"])
@pytest.mark.parametrize("parameters", [[7], {"n": 7}])
def test_managed_native_queries_preserve_parameters_and_nested_schema(api, parameters):
    parameter = "$n" if isinstance(parameters, dict) else "?"
    sql = f"SELECT {parameter}::BIGINT AS x, [1.0::DOUBLE, 2.0] AS embedding, 'rgb'::BLOB AS image"
    with vane.connect() as connection:
        runtime = _runtime(connection)
        with _execute(connection, api, sql, parameters) as result:
            _idle(runtime, results=1, bytes_=None)
            assert result.result_schema == {
                "names": ["x", "embedding", "image"],
                "types": ["BIGINT", "DOUBLE[]", "BLOB"],
            }
            table = result.take()
            assert table.to_pylist() == [{"x": 7, "embedding": [1.0, 2.0], "image": b"rgb"}]
            assert result.completion_status == "ok"
            del table
        _idle(runtime)


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_empty_managed_query_keeps_schema_without_delivery_owners(api):
    with vane.connect() as connection:
        runtime = _runtime(connection)
        result = _execute(connection, api, "SELECT 42::BIGINT AS x WHERE false")
        assert result.result_schema == {"names": ["x"], "types": ["BIGINT"]}
        assert result.completion_status == "empty"
        assert list(result) == []
        _idle(runtime)


@pytest.mark.parametrize("api", ["sql", "relation"])
@pytest.mark.parametrize("batch", [False, True])
def test_managed_queries_reuse_models_and_return_execution_capacity(tmp_path, api, batch):
    with vane.connect() as connection:
        runtime = _runtime(
            connection, task_limit=TaskAdmissionLimits(1, 4), data_limit=DataAdmissionLimits(420_000, 140_000, 70_000)
        )
        model = _model(connection, runtime, tmp_path, batch=batch)
        model.prewarm()
        for value in range(3):
            with connection.cursor() as cursor:
                result = (
                    cursor.execute_result("SELECT managed_encode(?) AS value", [value])
                    if api == "sql"
                    else cursor.sql(f"SELECT {value} AS x")
                    .project(model(vane.col("x")).alias("value"))
                    .execute_result()
                )
                _idle(runtime, results=1, bytes_=None)
                table = result.take()
                assert table.to_pylist() == [{"value": value}]
                del table
                result.close()
                _idle(runtime)
        assert len((tmp_path / "initializations").read_text().splitlines()) == 1
        assert runtime.resource_snapshot()["reserved_resources"] == ResourceVector(cpu=1).to_dict()
    assert runtime.resource_snapshot()["reserved_resources"] == ResourceVector().to_dict()


@pytest.mark.parametrize("actor", [False, True])
def test_unregistered_subprocess_udfs_use_managed_delivery(actor):
    def process(table):
        return table

    class Model:
        def __call__(self, table):
            return process(table)

    with vane.connect() as connection:
        runtime = _runtime(
            connection, task_limit=TaskAdmissionLimits(1, 4), data_limit=DataAdmissionLimits(420_000, 140_000, 70_000)
        )
        relation = connection.sql("SELECT i::BIGINT AS x FROM range(3) t(i)").map_batches(
            Model if actor else process,
            schema={"x": "BIGINT"},
            execution_backend="subprocess_actor" if actor else "subprocess_task",
            actor_number=1 if actor else None,
        )
        with relation.execute_result() as result:
            assert result.take().column(0).to_pylist() == [0, 1, 2]
        gc.collect()
        _idle(runtime)
        assert runtime.resource_snapshot()["registered_models"] == 0


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_full_result_slot_refuses_before_udf_execution_without_leaking_request(tmp_path, api):
    with vane.connect() as connection:
        runtime = _runtime(connection, results=1)
        _model(connection, runtime, tmp_path)
        first = _execute(connection, api)
        for _ in range(3):
            with pytest.raises(ResultDeliveryFull, match="slots") as caught:
                _execute(connection, api, "SELECT managed_encode(8)")
            assert (caught.value.reason, caught.value.requested, caught.value.used, caught.value.limit) == (
                "slots",
                1,
                1,
                1,
            )
            assert caught.value.execution_started is False
            _idle(runtime, results=1, bytes_=None)
        assert not (tmp_path / "calls").exists()
        assert not (tmp_path / "initializations").exists()
        first.close()
        result = _execute(connection, api, "SELECT managed_encode(8)")
        result.close()
        assert (tmp_path / "calls").read_text().splitlines() == ["8"]
        _idle(runtime)


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_queued_native_query_does_not_occupy_result_capacity(tmp_path, api):
    with vane.connect() as connection:
        runtime = _runtime(connection, results=1)
        _model(connection, runtime, tmp_path)
        with connection.cursor() as first, connection.cursor() as second, ThreadPoolExecutor(2) as clients:
            running = clients.submit(_execute, first, api, "SELECT managed_encode(-1)")
            try:
                _wait(lambda: (tmp_path / "entered").exists())
                queued = clients.submit(_execute, second, api, "SELECT managed_encode(2)")
                _wait(lambda: runtime.resource_snapshot()["request_admission"]["queued_requests"] == 1)
                delivery = runtime.resource_snapshot()["result_delivery"]
                assert delivery["active_results"] == delivery["preparing_results"] == 1
            finally:
                (tmp_path / "release").touch()
            result = running.result(timeout=15)
            with pytest.raises(ResultDeliveryFull):
                queued.result(timeout=15)
            result.close()
        assert (tmp_path / "calls").read_text().splitlines() == ["-1"]
        _idle(runtime)


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_exported_views_keep_delivery_bytes_after_connection_close(api):
    connection = vane.connect()
    runtime = _runtime(connection, size=512)
    result = _execute(connection, api)
    table = result.take()
    array = table.column(0).chunk(0).to_numpy(zero_copy_only=True)
    del table
    result.close()
    charged = runtime.resource_snapshot()["result_delivery"]["usage_bytes"]
    assert charged > 0
    with pytest.raises(ResultDeliveryFull, match="byte") as caught:
        _execute(connection, api)
    assert caught.value.reason == "bytes" and caught.value.execution_started is True
    assert caught.value.requested == caught.value.used == charged
    assert caught.value.limit == 512
    _idle(runtime, bytes_=charged)
    connection.close()
    assert array.tolist() == [42]
    assert runtime.resource_snapshot()["result_delivery"]["usage_bytes"] == charged
    del array
    gc.collect()
    _idle(runtime)


@pytest.mark.parametrize("api", ["sql", "relation"])
@pytest.mark.parametrize("action", ["cancel", "timeout", "close", "gc"])
def test_managed_result_termination_releases_buffers(api, action):
    connection = vane.connect()
    runtime = _runtime(connection, results=1)
    try:
        result = _execute(connection, api, delivery_timeout=0.02 if action == "timeout" else None)
    except ResultDeliveryTimeout:
        # Expiry may win the race with publication, especially under CI load.
        assert action == "timeout"
        _idle(runtime)
        connection.close()
        return
    error = ResultDeliveryClosed
    if action == "cancel":
        result.cancel()
        error = ResultDeliveryCancelled
    elif action == "timeout":
        _wait(lambda: runtime.resource_snapshot()["result_delivery"]["active_results"] == 0)
        error = ResultDeliveryTimeout
    elif action == "close":
        connection.close()
    else:
        owner = weakref.ref(connection)
        del connection
        gc.collect()
        assert owner() is None
    with pytest.raises(error):
        result.take()
    _idle(runtime)
    if action != "gc":
        connection.close()


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_delivery_byte_refusal_never_replays_a_model_call(tmp_path, api):
    with vane.connect() as connection:
        runtime = _runtime(connection, size=1)
        _model(connection, runtime, tmp_path)
        with pytest.raises(ResultDeliveryFull, match="byte") as caught:
            _execute(connection, api, "SELECT managed_encode(3)")
        assert caught.value.reason == "bytes" and caught.value.execution_started is True
        assert caught.value.requested > caught.value.limit == 1
        assert caught.value.used == 0
        assert (tmp_path / "calls").read_text().splitlines() == ["3"]
        _idle(runtime)
        assert connection.execute("SELECT managed_encode(4)").fetchall() == [(4,)]


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_zero_delivery_deadline_retires_unpublished_result(api):
    with vane.connect() as connection:
        runtime = _runtime(connection)
        with pytest.raises(ResultDeliveryTimeout):
            _execute(connection, api, delivery_timeout=0)
        _idle(runtime)


@pytest.mark.parametrize("api", ["sql", "relation"])
@pytest.mark.parametrize("failure", [-2, -3])
def test_managed_failure_retains_model_for_next_request(tmp_path, api, failure):
    with vane.connect() as connection:
        runtime = _runtime(connection)
        _model(connection, runtime, tmp_path)
        with pytest.raises(Exception):
            _execute(connection, api, f"SELECT managed_encode({failure})")
        _idle(runtime)
        with _execute(connection, api, "SELECT managed_encode(5)") as result:
            assert result.take().column(0).to_pylist() == [5]
        gc.collect()
        _idle(runtime)


@pytest.mark.parametrize("api", ["sql", "relation"])
@pytest.mark.parametrize("cause", ["interrupt", "deadline", "close"])
def test_managed_execution_cancellation_retires_result_reservation(tmp_path, api, cause):
    with vane.connect() as connection:
        runtime = _runtime(connection, execution_timeout=0.5 if cause == "deadline" else None)
        _model(connection, runtime, tmp_path)
        with connection.cursor() as cursor, ThreadPoolExecutor(2) as clients:
            future = clients.submit(_execute, cursor, api, "SELECT managed_encode(-1)")
            try:
                _wait(lambda: (tmp_path / "entered").exists() or future.done())
                closing = None
                if cause == "interrupt":
                    cursor.interrupt()
                elif cause == "close":
                    closing = clients.submit(cursor.close)
                with pytest.raises(RequestExecutionTimeout if cause == "deadline" else RequestCancelled):
                    future.result(timeout=15)
                if closing is not None:
                    closing.result(timeout=15)
            finally:
                (tmp_path / "release").touch()
        _idle(runtime)


def test_result_cleanup_failure_keeps_runtime_retry_owner(monkeypatch):
    with vane.connect() as connection:
        runtime = _runtime(connection)
        result = connection.execute_result("SELECT 1")
        original = local_result_delivery._ArrowResultPayload.close

        def fail(self):
            raise RuntimeError("result cleanup fixture failure")

        monkeypatch.setattr(local_result_delivery._ArrowResultPayload, "close", fail)
        with pytest.raises(RuntimeError, match="cleanup"):
            result.close()
        assert runtime.resource_snapshot()["result_delivery"]["cleanup_pending_results"] == 1
        del result
        with pytest.raises(RuntimeError, match="cleanup"):
            connection.close()
        monkeypatch.setattr(local_result_delivery._ArrowResultPayload, "close", original)
        connection.close()
        _idle(runtime)


@pytest.mark.parametrize("sql", ["", "SELECT 1; SELECT 2", "CREATE TABLE t(x INTEGER)", "EXPLAIN SELECT 1", "BEGIN"])
def test_execute_result_rejects_unsupported_statements(sql):
    with vane.connect() as connection:
        runtime = _runtime(connection)
        with pytest.raises(vane.InvalidInputException, match="one read-only SELECT"):
            connection.execute_result(sql)
        _idle(runtime)


@pytest.mark.parametrize("api", ["sql", "relation"])
@pytest.mark.parametrize("configured", [False, True])
def test_managed_queries_require_result_configuration(api, configured):
    with vane.connect() as connection:
        if configured:
            connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
        with pytest.raises((RuntimeError, vane.InvalidInputException), match="result_limit"):
            _execute(connection, api)


@pytest.mark.parametrize("api", ["sql", "relation"])
@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf")])
def test_invalid_delivery_timeout_does_not_admit_work(api, timeout):
    with vane.connect() as connection:
        runtime = _runtime(connection)
        with pytest.raises(ValueError, match="delivery_timeout"):
            _execute(connection, api, delivery_timeout=timeout)
        _idle(runtime)


def test_execute_result_rejects_a_relation_with_an_open_result():
    with vane.connect() as connection:
        runtime = _runtime(connection)
        relation = connection.sql("SELECT 1").execute()
        with pytest.raises(vane.InvalidInputException, match="open result"):
            relation.execute_result()
        assert relation.fetchall() == [(1,)]
        _idle(runtime)


def test_closed_native_result_allows_an_explicit_managed_execution():
    with vane.connect() as connection:
        runtime = _runtime(connection)
        relation = connection.sql("SELECT 1 AS x").execute()
        relation.close()
        with relation.execute_result() as result:
            assert result.take().column(0).to_pylist() == [1]
        _idle(runtime)


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_managed_query_entry_obeys_input_callback_guard(api):
    import numpy as np

    with vane.connect() as connection, vane.connect() as other:
        runtime = _runtime(connection)
        _runtime(other)
        relation = other.sql("SELECT 42")
        calls = []

        class Value:
            def __str__(self):
                calls.append(True)
                with pytest.raises(vane.InvalidInputException, match="Python input callback"):
                    if api == "sql":
                        other.execute_result("SELECT 42")
                    else:
                        relation.execute_result()
                return "ok"

        connection.register("items", {"x": np.array([Value()], dtype=object)})
        with connection.execute_result("SELECT x FROM items") as result:
            assert result.take().column(0).to_pylist() == ["ok"]
        assert calls
        gc.collect()
        _idle(runtime)


@pytest.mark.parametrize("argument", ["query", "parameters", "timeout"])
def test_managed_argument_callbacks_cannot_enter_other_connections(argument):
    with vane.connect() as connection, vane.connect() as other:
        runtime = _runtime(connection)
        calls = []

        def callback():
            calls.append(True)
            with pytest.raises(vane.InvalidInputException, match="Python input callback"):
                other.execute("SELECT 7")

        class Query(str):
            def __str__(self):
                callback()
                return super().__str__()

        class Parameters(list):
            def __len__(self):
                callback()
                return super().__len__()

        class Timeout(float):
            def __float__(self):
                callback()
                return super().__float__()

        with connection.execute_result(
            Query("SELECT ?::BIGINT") if argument == "query" else "SELECT ?::BIGINT",
            Parameters([7]) if argument == "parameters" else [7],
            delivery_timeout=Timeout(10) if argument == "timeout" else None,
        ) as result:
            assert result.take().column(0).to_pylist() == [7]
        # Native string conversion reads Unicode data directly. Parameter and
        # timeout conversion do call their guarded Python methods.
        assert bool(calls) == (argument != "query")
        gc.collect()
        _idle(runtime)


@pytest.mark.parametrize("api", ["sql", "relation"])
@pytest.mark.parametrize("empty", [False, True])
def test_runtime_close_fences_result_preparation_after_request_cleanup(monkeypatch, api, empty):
    entered, release = threading.Event(), threading.Event()
    original = local_result_delivery.prepare_native_query_result

    def prepare(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)

    monkeypatch.setattr(local_result_delivery, "prepare_native_query_result", prepare)
    with vane.connect() as connection:
        runtime = _runtime(connection)
        with ThreadPoolExecutor(1) as clients:
            future = clients.submit(_execute, connection, api, "SELECT 1" + (" WHERE false" if empty else ""))
            try:
                assert entered.wait(10)
                _idle(runtime, results=1)
                with pytest.raises(RuntimeError, match="cleanup"):
                    runtime.close()
                assert runtime.resource_snapshot()["result_delivery"]["cleanup_pending_results"] == 1
            finally:
                release.set()
            with pytest.raises(ResultDeliveryClosed):
                future.result(timeout=10)
        runtime.close()
        _idle(runtime)


@pytest.mark.parametrize("api", ["sql", "relation"])
def test_encoding_failure_preserves_primary_error_and_retry_owner(monkeypatch, api):
    original_build = local_result_delivery._ArrowResultPayload.build
    original_close = local_result_delivery._ArrowResultPayload.close

    def build(self, *args):
        original_build(self, *args)
        raise ValueError("encoding fixture failure")

    def close(self):
        raise RuntimeError("buffer cleanup fixture failure")

    with vane.connect() as connection:
        runtime = _runtime(connection)
        monkeypatch.setattr(local_result_delivery._ArrowResultPayload, "build", build)
        monkeypatch.setattr(local_result_delivery._ArrowResultPayload, "close", close)
        with pytest.raises(ValueError, match="encoding fixture failure") as failure:
            _execute(connection, api)
        assert failure.value.__cause__ is not None
        _idle(runtime, results=1, bytes_=None)
        delivery = runtime.resource_snapshot()["result_delivery"]
        assert delivery["cleanup_pending_results"] == 1 and delivery["usage_bytes"] > 0
        monkeypatch.setattr(local_result_delivery._ArrowResultPayload, "close", original_close)
        runtime.close()
        _idle(runtime)


@pytest.mark.parametrize("outcome", ["success", "slot_refused", "bytes_refused"])
def test_relation_close_does_not_execute_again_after_managed_delivery(tmp_path, outcome):
    with vane.connect() as connection:
        runtime = _runtime(connection, results=1, size=1 if outcome == "bytes_refused" else 1_000_000)
        _model(connection, runtime, tmp_path)
        occupied = connection.execute_result("SELECT 1") if outcome == "slot_refused" else None
        relation = connection.sql("SELECT managed_encode(9)")
        if outcome == "success":
            result = relation.execute_result()
            result.close()
        else:
            with pytest.raises(ResultDeliveryFull):
                relation.execute_result()
        relation.close()
        calls = tmp_path / "calls"
        assert (calls.read_text().splitlines() if calls.exists() else []) == (
            [] if outcome == "slot_refused" else ["9"]
        )
        if occupied is not None:
            occupied.close()
        _idle(runtime)


@pytest.mark.parametrize("kind", ["arrow", "pandas", "numpy"])
@pytest.mark.parametrize("statement", [False, True])
def test_managed_sql_preserves_caller_replacement_scans(kind, statement):
    import pyarrow as pa

    with vane.connect() as connection:
        runtime = _runtime(connection)
        if kind == "arrow":
            items = pa.table({"x": [7, 8]})
        elif kind == "pandas":
            items = pytest.importorskip("pandas").DataFrame({"x": [7, 8]})
        else:
            items = {"x": pytest.importorskip("numpy").array([7, 8])}  # noqa: F841 - replacement scan
        assert connection.execute("SELECT * FROM items ORDER BY x").fetchall() == [(7,), (8,)]
        query = "SELECT * FROM items WHERE x > ?"
        if statement:
            query = connection.extract_statements(query)[0]
        with connection.execute_result(query, [7]) as result:
            assert result.take().column(0).to_pylist() == [8]
        _idle(runtime)


def test_managed_sql_preserves_global_lookup_and_local_precedence(monkeypatch):
    import pyarrow as pa

    monkeypatch.setitem(globals(), "managed_global_items", pa.table({"x": [9]}))
    with vane.connect() as connection:
        runtime = _runtime(connection)

        def from_globals():
            return connection.execute_result("SELECT * FROM managed_global_items")

        with from_globals() as result:
            assert result.take().column(0).to_pylist() == [9]
        managed_global_items = pa.table({"x": [7]})  # noqa: F841 - shadows the global replacement scan
        with connection.execute_result("SELECT * FROM managed_global_items") as result:
            assert result.take().column(0).to_pylist() == [7]
        _idle(runtime)


@pytest.mark.parametrize("all_frames", [False, True])
def test_managed_sql_preserves_replacement_scan_frame_limit(all_frames):
    import pyarrow as pa

    caller_outer_items = pa.table({"x": [7]})  # noqa: F841 - only in the outer frame
    with vane.connect() as connection:
        connection.execute(f"SET python_scan_all_frames={str(all_frames).lower()}")
        runtime = _runtime(connection)

        def inner():
            return connection.execute_result("SELECT * FROM caller_outer_items")

        if all_frames:
            with inner() as result:
                assert result.take().column(0).to_pylist() == [7]
        else:
            with pytest.raises(vane.CatalogException, match="caller_outer_items"):
                inner()
        _idle(runtime)


def test_managed_sql_honors_disabled_replacement_scans():
    import pyarrow as pa

    items = pa.table({"x": [7]})  # noqa: F841 - disabled replacement scan
    with vane.connect() as connection:
        connection.execute("SET python_enable_replacements=false")
        runtime = _runtime(connection)
        with pytest.raises(vane.CatalogException, match="items"):
            connection.execute_result("SELECT * FROM items")
        _idle(runtime)


def test_queued_managed_sql_keeps_each_callers_replacement_frame(tmp_path):
    import pyarrow as pa

    def execute(cursor, value):
        items = pa.table({"x": [value]})  # noqa: F841 - each client has its own input
        return cursor.execute_result("SELECT * FROM items")

    with vane.connect() as connection:
        runtime = _runtime(connection, results=3)
        _model(connection, runtime, tmp_path)
        with (
            connection.cursor() as first,
            connection.cursor() as second,
            connection.cursor() as third,
            ThreadPoolExecutor(3) as clients,
        ):
            running = clients.submit(first.execute_result, "SELECT managed_encode(-1)")
            try:
                _wait(lambda: (tmp_path / "entered").exists())
                queued = [clients.submit(execute, cursor, value) for cursor, value in [(second, 7), (third, 8)]]
                _wait(lambda: runtime.resource_snapshot()["request_admission"]["queued_requests"] == 2)
            finally:
                (tmp_path / "release").touch()
            running.result(timeout=15).close()
            for future, value in zip(queued, [7, 8]):
                with future.result(timeout=15) as result:
                    assert result.take().column(0).to_pylist() == [value]
        _idle(runtime)


@pytest.mark.parametrize("outcome", ["success", "binding_failure", "slot_refused"])
def test_managed_sql_releases_caller_frames_after_return(outcome):
    import pyarrow as pa

    class Marker:
        pass

    with vane.connect() as connection:
        runtime = _runtime(connection, results=1)
        occupied = connection.execute_result("SELECT 1") if outcome == "slot_refused" else None
        references = []

        def execute():
            marker = Marker()
            references.append(weakref.ref(marker))
            items = pa.table({"x": [7]})  # noqa: F841 - replacement scan
            if outcome == "success":
                with connection.execute_result("SELECT * FROM items") as result:
                    assert result.take().column(0).to_pylist() == [7]
            elif outcome == "binding_failure":
                with pytest.raises(vane.CatalogException):
                    connection.execute_result("SELECT * FROM missing_managed_input")
            else:
                with pytest.raises(ResultDeliveryFull):
                    connection.execute_result("SELECT * FROM items")

        execute()
        if occupied is not None:
            occupied.close()
        gc.collect()
        assert references[0]() is None
        # A failed call must not leave its frame as the next call's lookup scope.
        items = pa.table({"x": [9]})  # noqa: F841 - new caller input
        with connection.execute_result("SELECT * FROM items") as result:
            assert result.take().column(0).to_pylist() == [9]
        _idle(runtime)


@pytest.mark.parametrize("refused_first", [False, True])
def test_managed_replacement_scan_callbacks_run_after_admission_with_reentry_guard(refused_first):
    import numpy as np

    with vane.connect() as connection, vane.connect() as other:
        runtime = _runtime(connection, results=1)
        calls = []

        class Value:
            def __str__(self):
                calls.append(True)
                with pytest.raises(vane.InvalidInputException, match="Python input callback"):
                    other.execute("SELECT 9")
                return "7"

        items = {"x": np.array([Value()], dtype=object)}  # noqa: F841 - replacement scan
        if refused_first:
            occupied = connection.execute_result("SELECT 1")
            with pytest.raises(ResultDeliveryFull):
                connection.execute_result("SELECT * FROM items")
            assert not calls
            occupied.close()
        with connection.execute_result("SELECT * FROM items") as result:
            assert result.take().column(0).to_pylist() == ["7"]
        assert calls
        _idle(runtime)


@pytest.mark.parametrize("fetch_method", ["fetchone", "fetchmany"])
@pytest.mark.parametrize("rows", [0, 1, 5])
def test_managed_execution_accepts_relation_at_first_observed_eof(fetch_method, rows):
    with vane.connect() as connection:
        runtime = _runtime(connection)
        relation = connection.sql(f"SELECT i AS x FROM range({rows}) t(i)").execute()
        consumed = []
        while True:
            batch = relation.fetchone() if fetch_method == "fetchone" else relation.fetchmany()
            if not batch:
                break
            consumed.extend([batch] if fetch_method == "fetchone" else batch)
        assert consumed == [(i,) for i in range(rows)]
        with relation.execute_result() as result:
            if rows:
                assert result.take().column(0).to_pylist() == list(range(rows))
            else:
                assert list(result) == []
                assert result.completion_status == "empty"
        _idle(runtime)


def test_managed_execution_accepts_eof_discovered_inside_a_partial_batch():
    with vane.connect() as connection:
        runtime = _runtime(connection)
        relation = connection.sql("SELECT 1 AS x").execute()
        assert relation.fetchmany(2) == [(1,)]
        with relation.execute_result() as result:
            assert result.take().column(0).to_pylist() == [1]
        _idle(runtime)


@pytest.mark.parametrize("consumption", ["unread", "partial", "zero_batch"])
def test_managed_execution_preserves_unread_relation_rows(consumption):
    with vane.connect() as connection:
        runtime = _runtime(connection)
        relation = connection.sql("SELECT i AS x FROM range(3) t(i)").execute()
        if consumption == "partial":
            assert relation.fetchone() == (0,)
        elif consumption == "zero_batch":
            assert relation.fetchmany(0) == []
        with pytest.raises(vane.InvalidInputException, match="open result"):
            relation.execute_result()
        assert relation.fetchall() == [(i,) for i in range(1 if consumption == "partial" else 0, 3)]
        _idle(runtime)
