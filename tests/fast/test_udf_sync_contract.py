# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Generic Python UDFs reject async callables at every public boundary."""

from __future__ import annotations

import functools
import gc
import types
import warnings

import cloudpickle
import pyarrow as pa
import pytest

import duckdb
import vane
from duckdb.execution._udf_runtime import UDFExecutor
from duckdb.execution._udf_validation import ensure_synchronous_udf_result, validate_synchronous_udf_callable

_ASYNC_CALLABLE_ERROR = "generic UDF callables must be synchronous"
_AWAITABLE_RESULT_ERROR = "generic UDF callables must return values synchronously"
_FUNCTION_SHAPE_ERROR = r"vane\.func requires a Python function or bound method"


async def _async_value(value):
    return value


async def _async_values(value):
    yield value


@functools.wraps(_async_value)
def _wrapped_async_value(value):
    return _async_value(value)


def _returns_awaitable(value):
    return _async_value(value)


def _returns_async_iterable(value):
    return _async_values(value)


class _AsyncCallable:
    async def __call__(self, value):
        return value


class _AsyncConstructionMeta(type):
    async def __call__(cls):
        return super().__call__()


class _AsyncConstructedCallable(metaclass=_AsyncConstructionMeta):
    def __call__(self, value):
        return value


class _AsyncNewCallable:
    async def __new__(cls):
        return super().__new__(cls)

    def __call__(self, value):
        return value


class _AsyncInitCallable:
    async def __init__(self):
        pass

    def __call__(self, value):
        return value


class _PartialMethodAsyncCallable:
    async def _call(self, value, *, offset):
        return value + offset

    __call__ = functools.partialmethod(_call, offset=0)


class _PartialAsyncCallableDescriptor:
    __call__ = functools.partial(_AsyncCallable())


class _AwaitableConstructionMeta(type):
    def __call__(cls):
        return _async_value(super().__call__())


class _AwaitablyConstructedCallable(metaclass=_AwaitableConstructionMeta):
    def __call__(self, value):
        return value


class _SyncReturnsAwaitable:
    def __call__(self, value):
        return _async_value(value)


class _SyncCallableWithShadowedAsyncCall:
    def __init__(self):
        self.__call__ = _async_value

    def __call__(self, value):
        return value


@types.coroutine
def _generator_coroutine():
    yield None


def _batch_yields_awaitable(table):
    yield _async_value(table)


def _flat_map_yields_awaitable(row):
    yield _async_value(row)


def _runtime_payload(target, call_mode, *, execution_backend="subprocess_task"):
    payload = {
        "function_pickle": cloudpickle.dumps(target),
        "call_mode": call_mode,
        "execution_backend": execution_backend,
        "udf_worker_slots": 1,
    }
    if execution_backend.endswith("actor"):
        payload["actor_number"] = 1
    return payload


def test_vane_decorators_reject_async_callables():
    with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
        vane.func(_async_value, return_dtype="INTEGER")

    with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
        vane.func.batch(return_dtype=pa.int32())(_async_value)

    with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
        vane.cls(_AsyncCallable, actor_number=1, return_dtype="INTEGER")

    with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
        vane.cls.batch(actor_number=1, return_dtype=pa.int32())(_AsyncCallable)


@pytest.mark.parametrize(
    "target",
    [
        pytest.param(_async_values, id="async-generator"),
        pytest.param(_wrapped_async_value, id="wrapped-async"),
        pytest.param(_generator_coroutine, id="generator-coroutine"),
    ],
)
def test_vane_func_rejects_indirect_async_functions(target):
    with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
        vane.func(target, return_dtype="INTEGER")


@pytest.mark.parametrize(
    "target",
    [
        pytest.param(_AsyncCallable(), id="async-callable-instance"),
        pytest.param(functools.partial(_async_value, 1), id="partial-async"),
        pytest.param(functools.partial(_AsyncCallable()), id="partial-async-callable-instance"),
    ],
)
def test_vane_func_rejects_non_function_async_callables_by_shape(target):
    with pytest.raises(TypeError, match=_FUNCTION_SHAPE_ERROR):
        vane.func(target, return_dtype="INTEGER", name="invalid_callable")


@pytest.mark.parametrize(
    "target",
    [
        _AsyncConstructedCallable,
        _AsyncNewCallable,
        _AsyncInitCallable,
        _PartialMethodAsyncCallable,
        _PartialAsyncCallableDescriptor,
    ],
)
def test_vane_class_rejects_async_construction_and_call_descriptors(target):
    with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
        vane.cls(target, actor_number=1, return_dtype="INTEGER")


def test_sync_validator_uses_the_effective_type_call_method():
    target = _SyncCallableWithShadowedAsyncCall()

    validate_synchronous_udf_callable(target)

    assert target(1) == 1


@pytest.mark.parametrize("method", ["map", "map_batches", "flat_map"])
def test_relation_task_udfs_reject_async_functions(method):
    with duckdb.connect() as connection:
        source = connection.sql("SELECT 1::INTEGER AS value")
        kwargs = {"execution_backend": "subprocess_task"}
        if method == "map":
            kwargs["return_type"] = duckdb.sqltypes.INTEGER
        else:
            kwargs["schema"] = {"value": duckdb.sqltypes.INTEGER}

        with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
            getattr(source, method)(_async_value, **kwargs)


@pytest.mark.parametrize("method", ["map", "map_batches", "flat_map"])
def test_relation_actor_udfs_reject_async_call_methods(method):
    with duckdb.connect() as connection:
        source = connection.sql("SELECT 1::INTEGER AS value")
        kwargs = {
            "execution_backend": "subprocess_actor",
            "actor_number": 1,
            "gpus": 0,
        }
        if method == "map":
            kwargs["return_type"] = duckdb.sqltypes.INTEGER
        else:
            kwargs["schema"] = {"value": duckdb.sqltypes.INTEGER}

        with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
            getattr(source, method)(_AsyncCallable, **kwargs)


def test_connection_udf_registration_rejects_async_functions():
    with duckdb.connect() as connection:
        with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
            connection.create_function(
                "async_scalar",
                _async_value,
                [duckdb.sqltypes.INTEGER],
                duckdb.sqltypes.INTEGER,
            )

        with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
            connection.create_table_function(
                "async_batches",
                _async_value,
                schema={"value": duckdb.sqltypes.INTEGER},
            )


def test_attach_function_rejects_raw_async_callables():
    connection = vane.connect()
    try:
        with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
            vane.attach_function(
                _async_value,
                alias="async_scalar",
                connection=connection,
                parameters=["INTEGER"],
                return_dtype="INTEGER",
            )

        with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
            vane.attach_function(
                _async_value,
                alias="async_batch",
                connection=connection,
                parameters=["INTEGER"],
                input_names=["value"],
                schema={"value": "INTEGER"},
            )

        with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
            vane.attach_function(
                _AsyncCallable,
                alias="async_actor",
                connection=connection,
                parameters=["INTEGER"],
                input_names=["value"],
                schema={"value": "INTEGER"},
                actor_number=1,
                gpus=0,
            )
    finally:
        connection.close()


@pytest.mark.parametrize("udf_type", ["native", "arrow"])
@pytest.mark.parametrize("exception_handling", ["default", "return_null"])
@pytest.mark.parametrize(
    "target",
    [
        pytest.param(_returns_awaitable, id="awaitable"),
        pytest.param(_returns_async_iterable, id="async-iterable"),
    ],
)
def test_scalar_registration_rejects_async_results_without_leaking_coroutines(udf_type, exception_handling, target):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with duckdb.connect() as connection:
            connection.create_function(
                "returns_async_result",
                target,
                [duckdb.sqltypes.INTEGER],
                duckdb.sqltypes.INTEGER,
                type=udf_type,
                exception_handling=exception_handling,
            )
            with pytest.raises(duckdb.Error, match=_AWAITABLE_RESULT_ERROR):
                connection.execute("SELECT returns_async_result(1)").fetchall()
        gc.collect()

    assert not [warning for warning in caught if "was never awaited" in str(warning.message)]


@pytest.mark.parametrize("call_mode", ["map", "map_batches", "flat_map"])
@pytest.mark.parametrize(
    "target",
    [
        pytest.param(_async_value, id="coroutine-function"),
        pytest.param(_async_values, id="async-generator-function"),
        pytest.param(_generator_coroutine, id="generator-coroutine-function"),
    ],
)
def test_runtime_rejects_serialized_async_functions(call_mode, target):
    with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
        UDFExecutor(_runtime_payload(target, call_mode))


@pytest.mark.parametrize("call_mode", ["map", "map_batches", "flat_map"])
@pytest.mark.parametrize(
    "target",
    [
        _AsyncCallable,
        _AsyncConstructedCallable,
        _AsyncNewCallable,
        _AsyncInitCallable,
        _PartialMethodAsyncCallable,
        _PartialAsyncCallableDescriptor,
    ],
)
def test_runtime_rejects_serialized_async_actor_methods(call_mode, target):
    with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
        UDFExecutor(
            _runtime_payload(
                target,
                call_mode,
                execution_backend="subprocess_actor",
            )
        )


@pytest.mark.parametrize("call_mode", ["map", "map_batches", "flat_map"])
@pytest.mark.parametrize(
    "target",
    [
        pytest.param(_returns_awaitable, id="awaitable"),
        pytest.param(_returns_async_iterable, id="async-iterable"),
    ],
)
def test_runtime_rejects_async_results_without_leaking_coroutines(call_mode, target):
    executor = UDFExecutor(_runtime_payload(target, call_mode))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with pytest.raises(TypeError, match=_AWAITABLE_RESULT_ERROR):
                executor.submit(pa.table({"value": [1]}))
        finally:
            executor.close()
        gc.collect()

    assert not [warning for warning in caught if "was never awaited" in str(warning.message)]


def test_result_validation_closes_generator_based_coroutines():
    result = _generator_coroutine()

    with pytest.raises(TypeError, match=_AWAITABLE_RESULT_ERROR):
        ensure_synchronous_udf_result(result)

    assert result.gi_frame is None


@pytest.mark.parametrize(
    ("call_mode", "target"),
    [
        pytest.param("map_batches", _batch_yields_awaitable, id="map-batches"),
        pytest.param("flat_map", _flat_map_yields_awaitable, id="flat-map"),
    ],
)
def test_runtime_rejects_awaitables_yielded_by_streaming_results(call_mode, target):
    executor = UDFExecutor(_runtime_payload(target, call_mode))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with pytest.raises(TypeError, match=_AWAITABLE_RESULT_ERROR):
                executor.submit(pa.table({"value": [1]}))
        finally:
            executor.close()
        gc.collect()

    assert not [warning for warning in caught if "was never awaited" in str(warning.message)]


def test_actor_construction_rejects_awaitable_results_without_leaking_coroutines():
    row_class = vane.cls(_AwaitablyConstructedCallable, actor_number=1, return_dtype="INTEGER")()
    batch_class = vane.cls.batch(actor_number=1, return_dtype=pa.int32())(_AwaitablyConstructedCallable)()
    row_actor = row_class.actor_class(["value"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match=_AWAITABLE_RESULT_ERROR):
            row_class(1)
        with pytest.raises(TypeError, match=_AWAITABLE_RESULT_ERROR):
            batch_class(pa.array([1], type=pa.int32()))
        with pytest.raises(TypeError, match=_AWAITABLE_RESULT_ERROR):
            UDFExecutor(
                _runtime_payload(
                    _AwaitablyConstructedCallable,
                    "map",
                    execution_backend="subprocess_actor",
                )
            )
        with pytest.raises(TypeError, match=_AWAITABLE_RESULT_ERROR):
            UDFExecutor(
                _runtime_payload(
                    row_actor,
                    "map_batches",
                    execution_backend="subprocess_actor",
                )
            )
        gc.collect()

    assert not [warning for warning in caught if "was never awaited" in str(warning.message)]


def test_vane_eager_udfs_reject_awaitable_results_without_leaking_coroutines():
    scalar = vane.func(_returns_awaitable, return_dtype="INTEGER")
    batch = vane.func.batch(return_dtype=pa.int32())(_returns_awaitable)
    row_class = vane.cls(_SyncReturnsAwaitable, actor_number=1, return_dtype="INTEGER")()
    batch_class = vane.cls.batch(actor_number=1, return_dtype=pa.int32())(_SyncReturnsAwaitable)()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match=_AWAITABLE_RESULT_ERROR):
            scalar(1)
        with pytest.raises(TypeError, match=_AWAITABLE_RESULT_ERROR):
            batch(pa.array([1], type=pa.int32()))
        with pytest.raises(TypeError, match=_AWAITABLE_RESULT_ERROR):
            row_class(1)
        with pytest.raises(TypeError, match=_AWAITABLE_RESULT_ERROR):
            batch_class(pa.array([1], type=pa.int32()))
        gc.collect()

    assert not [warning for warning in caught if "was never awaited" in str(warning.message)]
