# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Async UDF contracts across eager, runtime, SQL and process boundaries."""

from __future__ import annotations

import asyncio
import functools
import pickle
import threading
import uuid

import cloudpickle
import pyarrow as pa
import pytest

import vane
from vane.execution._udf_async import callable_payload_options, configure_udf_callable
from vane.execution._udf_runtime import UDFExecutor
from vane.execution._udf_validation import validate_udf_callable


def _payload(target, mode="map", *, concurrency=3, timeout=None, actor=False, **fields):
    target = configure_udf_callable(target, mode, concurrency, timeout)
    return {
        "payload_version": 2,
        "function_pickle": cloudpickle.dumps(target),
        "call_mode": mode,
        "execution_backend": "subprocess_actor" if actor else "subprocess_task",
        **({"actor_number": 1} if actor else {}),
        **callable_payload_options(target, mode),
        **fields,
    }


def _drain(executor):
    return pa.concat_tables(executor.drain_outputs()).to_pydict()


def test_runtime_row_concurrency_order_nulls_and_loop_lifecycle():
    class Counter:
        def __init__(self):
            self.loop = asyncio.get_running_loop()
            self.thread = threading.get_ident()
            self.active = self.peak = self.opens = self.closes = 0
            self.calls = []

        async def aopen(self):
            self.opens += 1
            assert asyncio.get_running_loop() is self.loop

        async def __call__(self, value):
            assert asyncio.get_running_loop() is self.loop
            assert threading.get_ident() == self.thread
            self.calls.append(value)
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep((10 - value) * 0.001)
                return value * 10
            finally:
                self.active -= 1

        async def aclose(self):
            assert asyncio.get_running_loop() is self.loop
            assert not self.active
            self.closes += 1

    executor = UDFExecutor(_payload(Counter, actor=True))
    instance = executor._map_fn
    try:
        for _ in range(2):
            executor.submit(pa.table({"x": [0, None, 1, 2, 3, 4, 5]}))
            assert _drain(executor) == {"value": [0, None, 10, 20, 30, 40, 50]}
        assert instance.peak == 3
        assert instance.opens == 1
        assert instance.calls.count(0) == 2
    finally:
        executor.close()
    executor.close()
    assert instance.closes == 1
    assert instance.loop.is_closed()


@pytest.mark.parametrize("concurrency", [1, 2, 32])
def test_runtime_row_empty_and_small_inputs(concurrency):
    async def identity(value):
        await asyncio.sleep(0)
        return value

    executor = UDFExecutor(_payload(identity, concurrency=concurrency))
    try:
        executor.submit(pa.table({"x": pa.array([], type=pa.int64())}))
        assert executor.drain_outputs() == []
        executor.submit(pa.table({"x": [4, None]}))
        assert _drain(executor) == {"value": [4, None]}
    finally:
        executor.close()


def test_row_failure_cancels_siblings_before_returning():
    class Fails:
        async def aopen(self):
            self.started = asyncio.Event()
            self.cancelled = False

        async def __call__(self, value):
            if value == 0:
                self.started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.cancelled = True
            await self.started.wait()
            raise ValueError("row failure")

    executor = UDFExecutor(_payload(Fails, actor=True, concurrency=2))
    instance = executor._map_fn
    try:
        with pytest.raises(ValueError, match="row failure"):
            executor.submit(pa.table({"x": [0, 1, 2]}))
        assert instance.cancelled
        assert not asyncio.all_tasks(executor._async_runtime.loop)
        assert executor.drain_outputs() == []
        with pytest.raises(RuntimeError, match="aborted after ValueError: row failure"):
            executor.submit(pa.table({"x": [3]}))
    finally:
        executor.close()


@pytest.mark.parametrize("stream", [False, True])
def test_repeated_submissions_preserve_bounded_first_failure(stream):
    class Cancelled:
        def __init__(self):
            self.calls = 0

        async def __call__(self, table):
            self.calls += 1
            raise asyncio.CancelledError("call cancelled " + "x" * 100_000)

    executor = UDFExecutor(_payload(Cancelled, "map_batches", actor=True, stream_output=stream))
    instance = executor._map_fn

    def submit():
        table = pa.table({"x": [1]})
        return list(executor.iter_submit(table)) if stream else executor.submit(table)

    try:
        with pytest.raises(asyncio.CancelledError, match="call cancelled"):
            submit()
        executor.abort(RuntimeError("secondary failure"))
        for _ in range(2):
            with pytest.raises(RuntimeError, match="aborted after CancelledError: call cancelled") as exc:
                submit()
            assert len(str(exc.value).encode()) < 4500
            assert exc.value.__cause__ is None
            assert exc.value.__context__ is None
        assert instance.calls == 1
    finally:
        executor.close()


def test_call_timeout_drains_cancellation():
    class TimesOut:
        def __init__(self):
            self.cancelled = 0

        async def __call__(self, value):
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled += 1

    executor = UDFExecutor(_payload(TimesOut, actor=True, timeout=0.01))
    instance = executor._map_fn
    try:
        with pytest.raises(asyncio.TimeoutError):
            executor.submit(pa.table({"x": [1, 2, 3]}))
        assert instance.cancelled == 3
        assert not asyncio.all_tasks(executor._async_runtime.loop)
    finally:
        executor.close()


def test_return_null_does_not_hide_async_result_protocol_errors():
    async def inner():
        return 1

    async def invalid(value):
        return inner()

    executor = UDFExecutor(_payload(invalid, exception_handling=int(vane.PythonExceptionHandling.RETURN_NULL)))
    try:
        with pytest.raises(TypeError, match="received an awaitable"):
            executor.submit(pa.table({"x": [1]}))
    finally:
        executor.close()


@pytest.mark.parametrize("stream", [False, True])
def test_async_batch_window_bounds_calls_and_restores_order(stream):
    class BatchCounter:
        def __init__(self):
            self.active = self.peak = 0

        async def __call__(self, table):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                first = table.column(0)[0].as_py()
                await asyncio.sleep((12 - first) * 0.001)
                return table
            finally:
                self.active -= 1

    executor = UDFExecutor(_payload(BatchCounter, "map_batches", actor=True, batch_size=2, stream_output=stream))
    instance = executor._map_fn
    table = pa.table({"x": list(range(11))})
    try:
        if stream:
            output = pa.concat_tables(list(executor.iter_submit(table)))
        else:
            executor.submit(table)
            output = pa.concat_tables(executor.drain_outputs())
        assert output.equals(table)
        assert instance.peak == 3
        assert instance.active == 0
    finally:
        executor.close()


def test_later_batch_failure_cancels_slow_first_batch():
    class Fails:
        async def aopen(self):
            self.cancelled = False
            self.started = asyncio.Event()

        async def __call__(self, table):
            if table.column(0)[0].as_py() == 0:
                self.started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.cancelled = True
            await self.started.wait()
            raise ValueError("batch failure")

    executor = UDFExecutor(_payload(Fails, "map_batches", actor=True, batch_size=1))
    instance = executor._map_fn
    try:
        with pytest.raises(ValueError, match="batch failure"):
            list(executor.iter_submit(pa.table({"x": [0, 1, 2]})))
        assert instance.cancelled
        assert not asyncio.all_tasks(executor._async_runtime.loop)
    finally:
        executor.close()


def test_aborted_executor_does_not_execute_buffered_tail():
    async def fail_if_called(table):
        raise AssertionError("cancelled tail was executed")

    executor = UDFExecutor(_payload(fail_if_called, "map_batches", batch_size=4096))
    executor.submit(pa.table({"x": [1, 2]}))
    assert not executor.drain_outputs()
    executor.abort()
    executor.close()


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_async_concurrency_validation(value):
    async def identity(value):
        return value

    with pytest.raises(ValueError, match="max_concurrency"):
        vane.func(identity, return_dtype="INTEGER", max_concurrency=value)


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "2"])
def test_async_timeout_validation(value):
    async def identity(value):
        return value

    with pytest.raises(ValueError, match="timeout_s"):
        vane.func(identity, return_dtype="INTEGER", timeout_s=value)


@pytest.mark.parametrize("options", [{"max_concurrency": 2}, {"timeout_s": 1}])
def test_sync_function_rejects_async_options(options):
    with pytest.raises(ValueError, match="require an async UDF"):
        vane.func(lambda value: value, return_dtype="INTEGER", **options)


def test_wrapped_async_and_bound_method_preserve_configuration():
    class Methods:
        def __init__(self, offset):
            self.offset = offset

        @vane.func(return_dtype="INTEGER", max_concurrency=7, timeout_s=2)
        async def add(self, value, *, extra=0):
            return self.offset + value + extra

    bound = Methods(10).add
    assert asyncio.run(bound(2, extra=3)) == 15
    options = callable_payload_options(bound.python_function, "map")
    assert options["max_concurrency"] == 7
    assert options["timeout_s"] == 2
    with vane.connect() as con:
        rows = con.sql("SELECT 2 AS x").select(bound(vane.col("x"), extra=3)).fetchall()
    assert rows == [(15,)]


def test_sync_wrapper_of_async_function_keeps_its_behavior():
    async def original(value):
        return value + 1

    @functools.wraps(original)
    def wrapper(value):
        return original(value * 10)

    udf = vane.func(wrapper, return_dtype="INTEGER")
    assert asyncio.run(udf(2)) == 21


@pytest.mark.parametrize("hook", ["aopen", "aclose"])
def test_class_lifecycle_hooks_must_be_async_instance_methods(hook):
    class Invalid:
        async def __call__(self, value):
            return value

    setattr(Invalid, hook, lambda self: None)
    with pytest.raises(TypeError, match=f"{hook} must be an async instance method"):
        vane.cls(Invalid, actor_number=1, return_dtype="INTEGER")


def test_partial_open_failure_preserves_primary_and_closes_loop():
    from vane.execution._async_runtime import AsyncRuntime
    from vane.execution._udf_async import AsyncClassInstance

    class Fails:
        async def aopen(self):
            self.loop = asyncio.get_running_loop()
            self.closes = 0
            raise ValueError("open failed")

        async def aclose(self):
            self.closes += 1
            raise RuntimeError("cleanup failed")

    managed = AsyncClassInstance(Fails)
    runtime = AsyncRuntime()
    try:
        with pytest.raises(ValueError, match="open failed") as caught:
            runtime.run(managed.open())
        assert "cleanup failed" in str(caught.value.__cause__)
        assert managed.instance.closes == 1
        assert not asyncio.all_tasks(runtime.loop)
    finally:
        runtime.close()


def test_async_close_failure_is_retried_on_the_same_loop():
    class TransientClose:
        async def aopen(self):
            self.loop = asyncio.get_running_loop()
            self.closes = 0

        async def __call__(self, value):
            return value

        async def aclose(self):
            assert asyncio.get_running_loop() is self.loop
            self.closes += 1
            if self.closes == 1:
                raise ValueError("transient close failure")

    executor = UDFExecutor(_payload(TransientClose, actor=True))
    instance = executor._map_fn
    with pytest.raises(RuntimeError, match="transient close failure"):
        executor.close()
    assert not instance.loop.is_closed()
    executor.close()
    assert instance.closes == 2
    assert instance.loop.is_closed()


def test_eager_class_cleanup_does_not_replace_body_failure():
    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class Fails:
        async def __call__(self, value):
            raise ValueError("body failed")

        async def aclose(self):
            raise RuntimeError("cleanup failed")

    async def use():
        async with Fails() as udf:
            await udf(1)

    with pytest.raises(ValueError, match="body failed") as caught:
        asyncio.run(use())
    assert "cleanup failed" in str(caught.value.__cause__)


def test_cleanup_summary_is_bounded_and_survives_ray_error_envelope():
    from vane.execution._diagnostics import attach_cleanup_error
    from vane.execution.udf_ray_stream_protocol import make_stream_error_pair

    primary = ValueError("primary failure")
    cleanup = RuntimeError("cleanup head" + "x" * 100_000 + "cleanup tail")
    attach_cleanup_error(primary, cleanup)
    payload = {"query_id": "q", "resource_unit_id": "u", "task_lease_id": "t", "attempt_id": "a"}
    _, metadata = make_stream_error_pair(payload, primary)
    assert metadata["exception_type"] == "ValueError"
    message = metadata["exception_message"]
    assert "primary failure" in message
    assert "cleanup head" in message and "cleanup tail" in message
    assert len(message.encode()) < 4096
    assert primary.__cause__.__context__ is None


def test_abandoned_batch_stream_drains_pending_calls():
    class Streaming:
        async def aopen(self):
            self.active = 0

        async def __call__(self, table):
            self.active += 1
            try:
                if table.column(0)[0].as_py() == 0:
                    await asyncio.sleep(0)
                    return table
                await asyncio.Event().wait()
            finally:
                self.active -= 1

    executor = UDFExecutor(
        _payload(Streaming, "map_batches", actor=True, batch_size=1, stream_output=True, output_batch_size=1)
    )
    instance = executor._map_fn
    try:
        stream = iter(executor.iter_submit(pa.table({"x": [0, 1, 2, 3]})))
        assert next(stream).to_pydict() == {"x": [0]}
        assert instance.active == 2
        stream.close()
        assert instance.active == 0
        assert not asyncio.all_tasks(executor._async_runtime.loop)
        assert executor._aborted
    finally:
        executor.close()


@pytest.mark.parametrize("batch", [False, True])
def test_eager_call_honors_timeout(batch):
    decorator = vane.func.batch if batch else vane.func

    @decorator(return_dtype="BIGINT", timeout_s=0.01)
    async def slow(value):
        await asyncio.Event().wait()

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(slow(pa.array([1]) if batch else 1))


def test_async_flat_map_and_iterator_results_are_rejected():
    async def identity(row):
        return row

    with pytest.raises(TypeError, match="flat_map does not support"):
        configure_udf_callable(identity, "flat_map")

    async def returns_iterator(table):
        return iter([table])

    with UDFExecutor(_payload(returns_iterator, "map_batches")) as executor:
        with pytest.raises(TypeError, match="one materialized"):
            executor.submit(pa.table({"x": [1]}))


def test_async_self_cancellation_is_not_converted_to_null():
    async def cancelled(value):
        raise asyncio.CancelledError()

    executor = UDFExecutor(_payload(cancelled, exception_handling=int(vane.PythonExceptionHandling.RETURN_NULL)))
    try:
        with pytest.raises(asyncio.CancelledError):
            executor.submit(pa.table({"x": [1]}))
        assert not executor.drain_outputs()
    finally:
        executor.close()


def test_async_tasks_do_not_reuse_cached_callable_instances():
    class Owner:
        async def add(self, value):
            self.calls = getattr(self, "calls", 0) + 1
            return self.calls

    payload = _payload(Owner().add, concurrency=1)
    for _ in range(2):
        with UDFExecutor(payload, cache_callable=True) as executor:
            executor.submit(pa.table({"x": [1]}))
            assert _drain(executor) == {"value": [1]}


def test_default_concurrency_and_decorators_leave_user_definitions_unchanged():
    async def identity(value):
        return value

    class Identity:
        async def __call__(self, value):
            return value

    for target, mode, expected in [
        (identity, "map", 32),
        (identity, "map_batches", 1),
        (Identity, "map", 1),
        (Identity, "map_batches", 1),
    ]:
        configured = configure_udf_callable(target, mode)
        assert callable_payload_options(configured, mode)["max_concurrency"] == expected
        assert not hasattr(target, "_vane_udf_call_options")
        assert validate_udf_callable(configured) == "async"


def test_async_batch_eager_and_class_context():
    @vane.func.batch(return_dtype=pa.int64())
    async def identity(values):
        return values

    values = pa.array([1, None, 2])
    assert asyncio.run(identity(values)).equals(values)

    @vane.cls(actor_number=1, return_dtype="INTEGER", max_concurrency=3)
    class Stateful:
        async def aopen(self):
            self.loop = asyncio.get_running_loop()
            self.closed = False

        async def __call__(self, value):
            assert asyncio.get_running_loop() is self.loop
            return value + 1

        async def aclose(self):
            self.closed = True

    decorated = Stateful()
    with pytest.raises(RuntimeError, match="async with"):
        decorated(1)

    async def use():
        async with decorated as local:
            instance = local._eager_async.instance
            assert await local(5) == 6
        assert instance.closed

    asyncio.run(use())


def test_eager_class_rejects_overlapping_context_initialization():
    async def use():
        ready = asyncio.Event()

        @vane.cls(actor_number=1, return_dtype="INTEGER")
        class Delayed:
            async def aopen(self):
                await ready.wait()

            async def __call__(self, value):
                return value

        udf = Delayed()
        opening = asyncio.create_task(udf.__aenter__())
        await asyncio.sleep(0)
        try:
            with pytest.raises(RuntimeError, match="context is already open"):
                await udf.__aenter__()
        finally:
            ready.set()
            await opening
            await udf.__aexit__(None, None, None)

    asyncio.run(use())


def _expression(kind):
    if kind == "func":

        @vane.func(return_dtype="BIGINT", max_concurrency=4)
        async def udf(value):
            await asyncio.sleep(0)
            return value + 10

        return udf
    if kind == "func_batch":

        @vane.func.batch(return_dtype=pa.int64(), batch_size=2, max_concurrency=3)
        async def udf(values):
            await asyncio.sleep(0)
            return pa.array([None if value is None else value + 10 for value in values.to_pylist()])

        return udf
    if kind == "cls":

        @vane.cls(actor_number=1, return_dtype="BIGINT", max_concurrency=4)
        class UDF:
            async def aopen(self):
                self.offset = 10
                self.loop = asyncio.get_running_loop()

            async def __call__(self, value):
                assert self.loop is asyncio.get_running_loop()
                await asyncio.sleep(0)
                return value + self.offset

        return UDF()

    @vane.cls.batch(actor_number=1, return_dtype=pa.int64(), batch_size=2, max_concurrency=3)
    class UDF:
        async def aopen(self):
            self.offset = 10
            self.loop = asyncio.get_running_loop()

        async def __call__(self, values):
            assert self.loop is asyncio.get_running_loop()
            await asyncio.sleep(0)
            return pa.array([None if value is None else value + self.offset for value in values.to_pylist()])

    return UDF()


@pytest.mark.parametrize("kind", ["func", "func_batch", "cls", "cls_batch"])
def test_async_expression_and_sql_local(kind):
    udf = _expression(kind)
    with vane.connect() as con:
        source = con.sql("SELECT i AS id, CASE WHEN i=3 THEN NULL ELSE i END AS x FROM range(9) t(i)")
        expected = [(i, None if i == 3 else i + 10) for i in range(9)]
        relation = source.select(vane.col("id"), udf(vane.col("x")).alias("y"))
        assert sorted(relation.fetchall()) == expected
        vane.attach_function(udf, alias="async_test", parameters=["BIGINT"], connection=con)
        actual = con.sql("SELECT i, async_test(CASE WHEN i=3 THEN NULL ELSE i END) FROM range(9) t(i)")
        assert sorted(actual.fetchall()) == expected


@pytest.mark.real_ray
@pytest.mark.parametrize("kind", ["func", "func_batch", "cls", "cls_batch"])
def test_async_expression_ray(kind, ray_local, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    udf = _expression(kind)
    with vane.connect() as con:
        source = con.sql("SELECT i AS x FROM range(12) t(i)")
        result = source.select(vane.col("x"), udf(vane.col("x")).alias("y"))
        assert sorted(result.fetchall()) == [(i, i + 10) for i in range(12)]
        vane.attach_function(udf, alias="ray_async", parameters=["BIGINT"], connection=con)
        assert sorted(con.sql("SELECT ray_async(i) FROM range(4) t(i)").fetchall()) == [(i + 10,) for i in range(4)]


@pytest.mark.parametrize("actor", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_relation_raw_async_callables(actor, batch):
    async def identity(value):
        await asyncio.sleep(0)
        return value

    class Identity:
        async def __call__(self, value):
            return await identity(value)

    kwargs = {"max_concurrency": 3, "timeout_s": 5}
    if actor:
        kwargs.update(actor_number=1, execution_backend="subprocess_actor")
    else:
        kwargs["execution_backend"] = "subprocess_task"
    target = Identity if actor else identity
    with vane.connect() as con:
        source = con.sql("SELECT i AS x FROM range(9) t(i)")
        if batch:
            output = source.map_batches(target, schema={"x": vane.sqltypes.BIGINT}, batch_size=2, **kwargs)
            assert sorted(output.fetchall()) == [(i,) for i in range(9)]
        else:
            output = source.map(target, return_type=vane.sqltypes.BIGINT, **kwargs)
            assert sorted(output.fetchall()) == [(i, i) for i in range(9)]


def test_async_payload_survives_plan_replay():
    udf = _expression("func_batch")
    with vane.connect() as source, vane.connect() as target:
        relation = source.sql("SELECT 1::BIGINT AS x").select(udf(vane.col("x")))
        logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, str(uuid.uuid4()))
        restored = pickle.loads(pickle.dumps(logical))
        physical = restored.to_physical_plan(target)
        payload = physical.collect_udf_nodes()[0]["payload"]
        assert payload["payload_version"] == 2
        assert payload["execution_kind"] == "async"
        assert payload["invocation_granularity"] == "batch"
        assert payload["max_concurrency"] == 3
        assert payload["min_task_batch_size"] == 6


def test_version_one_payload_and_actor_rpc_concurrency_are_rejected():
    from vane.execution.udf import normalize_options

    async def identity(value):
        return value

    payload = _payload(identity)
    payload["payload_version"] = 1
    with pytest.raises(ValueError, match="payload_version"):
        UDFExecutor(payload)
    with pytest.raises(ValueError, match="max_concurrency"):
        normalize_options({"ray_options": {"max_concurrency": 3}})


def test_configuring_async_options_does_not_change_callable_shape_rules():
    class Callable:
        async def __call__(self, value):
            return value

    with pytest.raises(TypeError, match="Python function, bound method, or callable class"):
        configure_udf_callable(Callable(), "map")


def test_raw_async_sql_configuration_and_override_rejection():
    async def identity(value):
        return value

    with vane.connect() as con:
        vane.attach_function(
            identity,
            alias="raw_async",
            parameters=["INTEGER"],
            return_dtype="INTEGER",
            max_concurrency=2,
            timeout_s=3,
            connection=con,
        )
        assert con.sql("SELECT raw_async(7::INTEGER)").fetchall() == [(7,)]
        with pytest.raises(vane.InvalidInputException, match="decorator"):
            vane.attach_function(
                _expression("func"), alias="invalid", parameters=["BIGINT"], max_concurrency=2, connection=con
            )


@pytest.mark.parametrize("batch", [False, True])
def test_async_file_values_keep_the_governed_type_contract(batch):
    if batch:

        @vane.func.batch(return_dtype=vane.file_type(), batch_size=2, max_concurrency=2)
        async def identity(value):
            await asyncio.sleep(0)
            return value
    else:

        @vane.func(return_dtype=vane.file_type(), max_concurrency=2)
        async def identity(value):
            assert isinstance(value, vane.File)
            await asyncio.sleep(0)
            return value

    with vane.connect() as con:
        source = con.sql(
            "SELECT * FROM (VALUES (0, file('memory://async', NULL, NULL, NULL, NULL)), (1, NULL::FILE)) t(id, value)"
        )
        result = source.select(vane.col("id"), identity(vane.col("value")).alias("value"))
        assert result.types[1].is_file()
        assert sorted(result.fetchall()) == [(0, vane.File("memory://async")), (1, None)]


def test_async_struct_outputs_and_batch_length_errors():
    dtype = pa.struct([("original", pa.int64()), ("doubled", pa.int64())])

    @vane.func.batch(return_dtype=dtype, batch_size=2, max_concurrency=3)
    async def pairs(values):
        await asyncio.sleep(0)
        return pa.array([{"original": value, "doubled": value * 2} for value in values.to_pylist()], type=dtype)

    @vane.func.batch(return_dtype=pa.int64())
    async def wrong_length(values):
        return pa.array([], type=pa.int64())

    with vane.connect() as con:
        source = con.sql("SELECT i AS x FROM range(7) t(i)")
        rows = source.select(vane.col("x"), pairs(vane.col("x"))).fetchall()
        assert sorted(rows) == [(i, {"original": i, "doubled": i * 2}) for i in range(7)]
        with pytest.raises(Exception, match="returned 0 rows for"):
            source.select(wrong_length(vane.col("x"))).fetchall()


def test_raw_async_batch_sql_inherits_batch_options():
    async def identity(table):
        await asyncio.sleep(0)
        return table

    with vane.connect() as con:
        vane.attach_function(
            identity,
            alias="raw_async_batch",
            parameters=["BIGINT"],
            input_names=["x"],
            schema={"x": "BIGINT"},
            batch_size=2,
            max_concurrency=3,
            connection=con,
        )
        assert sorted(con.sql("SELECT raw_async_batch(i) FROM range(9) t(i)").fetchall()) == [(i,) for i in range(9)]


def test_worker_rejects_mismatched_serialized_concurrency():
    async def identity(value):
        return value

    payload = _payload(identity, concurrency=2)
    payload["max_concurrency"] = 3
    with pytest.raises(ValueError, match="configuration does not match"):
        UDFExecutor(payload)


def test_async_subprocess_cancellation_reports_an_error_to_the_query():
    @vane.func(return_dtype="INTEGER")
    async def cancelled(value):
        raise asyncio.CancelledError("call cancelled")

    with vane.connect() as con:
        with pytest.raises(Exception, match="CancelledError.*call cancelled"):
            con.sql("SELECT 1 AS x").select(cancelled(vane.col("x"))).fetchall()


@pytest.mark.real_ray
@pytest.mark.parametrize("kind", ["func", "func_batch", "cls", "cls_batch"])
def test_async_ray_cancellation_reports_an_error_to_the_query(ray_local, monkeypatch, kind):
    monkeypatch.setenv("VANE_RUNNER", "ray")

    async def cancelled(value):
        raise asyncio.CancelledError("call cancelled")

    class Cancelled:
        async def __call__(self, value):
            raise asyncio.CancelledError("call cancelled")

    decorator = {
        "func": vane.func,
        "func_batch": vane.func.batch,
        "cls": vane.cls,
        "cls_batch": vane.cls.batch,
    }[kind]
    if kind.startswith("cls"):
        udf = decorator(actor_number=1, return_dtype="INTEGER")(Cancelled)()
    else:
        udf = decorator(return_dtype="INTEGER")(cancelled)

    with vane.connect() as con:
        with pytest.raises(Exception, match="CancelledError.*call cancelled"):
            con.sql("SELECT 1 AS x").select(udf(vane.col("x"))).fetchall()
