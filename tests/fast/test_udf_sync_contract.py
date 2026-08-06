# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Generic Python UDFs reject async callables at every public boundary."""

from __future__ import annotations

import functools
import gc
import warnings

import cloudpickle
import pyarrow as pa
import pytest

import duckdb
import vane
from duckdb.execution._udf_runtime import UDFExecutor

_ASYNC_CALLABLE_ERROR = "generic UDF callables must be synchronous"
_AWAITABLE_RESULT_ERROR = "generic UDF callables must return values synchronously"


async def _async_value(value):
    return value


async def _async_values(value):
    yield value


@functools.wraps(_async_value)
def _wrapped_async_value(value):
    return _async_value(value)


def _returns_awaitable(value):
    return _async_value(value)


class _AsyncCallable:
    async def __call__(self, value):
        return value


class _SyncReturnsAwaitable:
    def __call__(self, value):
        return _async_value(value)


class _SyncCallableWithShadowedAsyncCall:
    def __init__(self):
        self.__qualname__ = type(self).__qualname__
        self.__call__ = _async_value

    def __call__(self, value):
        return value


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

    with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
        vane.func(_AsyncCallable(), return_dtype="INTEGER")


@pytest.mark.parametrize(
    "target",
    [
        pytest.param(_async_values, id="async-generator"),
        pytest.param(_wrapped_async_value, id="wrapped-async"),
        pytest.param(functools.partial(_async_value, 1), id="partial-async"),
    ],
)
def test_vane_func_rejects_indirect_async_callables(target):
    with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
        vane.func(target, return_dtype="INTEGER")


def test_vane_func_validates_the_effective_callable_object_call_method():
    fn = vane.func(_SyncCallableWithShadowedAsyncCall(), return_dtype="INTEGER", name="sync_callable")

    assert fn(1) == 1


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
def test_scalar_registration_rejects_awaitable_results_without_leaking_coroutines(udf_type, exception_handling):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with duckdb.connect() as connection:
            connection.create_function(
                "returns_awaitable",
                _returns_awaitable,
                [duckdb.sqltypes.INTEGER],
                duckdb.sqltypes.INTEGER,
                type=udf_type,
                exception_handling=exception_handling,
            )
            with pytest.raises(duckdb.Error, match=_AWAITABLE_RESULT_ERROR):
                connection.execute("SELECT returns_awaitable(1)").fetchall()
        gc.collect()

    assert not [warning for warning in caught if "was never awaited" in str(warning.message)]


@pytest.mark.parametrize("call_mode", ["map", "map_batches", "flat_map"])
def test_runtime_rejects_serialized_async_functions(call_mode):
    with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
        UDFExecutor(_runtime_payload(_async_value, call_mode))


@pytest.mark.parametrize("call_mode", ["map", "map_batches", "flat_map"])
def test_runtime_rejects_serialized_async_actor_methods(call_mode):
    with pytest.raises(TypeError, match=_ASYNC_CALLABLE_ERROR):
        UDFExecutor(
            _runtime_payload(
                _AsyncCallable,
                call_mode,
                execution_backend="subprocess_actor",
            )
        )


@pytest.mark.parametrize("call_mode", ["map", "map_batches", "flat_map"])
def test_runtime_rejects_awaitable_results_without_leaking_coroutines(call_mode):
    executor = UDFExecutor(_runtime_payload(_returns_awaitable, call_mode))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with pytest.raises(TypeError, match=_AWAITABLE_RESULT_ERROR):
                executor.submit(pa.table({"value": [1]}))
        finally:
            executor.close()
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
