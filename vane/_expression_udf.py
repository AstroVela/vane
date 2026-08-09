# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Expression-level Python UDF decorators for Vane."""

from __future__ import annotations

import functools
import inspect
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from numbers import Real
from types import CodeType
from typing import Any, Protocol

import vane
from vane import _native
from vane._expressions import as_expression, is_expression
from vane.config import current_config
from vane.execution._udf_validation import ensure_synchronous_udf_result, validate_synchronous_udf_callable


class _PythonFunction(Protocol):
    __name__: str
    __qualname__: str
    __code__: CodeType

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def __get__(self, instance: Any, owner: Any | None = None) -> _PythonFunction: ...


def _invalid_input(message: str) -> vane.InvalidInputException:
    return vane.InvalidInputException(message)


def _unsupported_dtype(dtype: Any) -> vane.InvalidInputException:
    return _invalid_input(f"dtype {str(dtype)!r} is not supported for expression UDF output")


def _arrow_to_duckdb_type(dtype: Any, *, original: Any) -> Any:
    import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]

    primitive_types = (
        (pa.types.is_boolean, "BOOLEAN"),
        (pa.types.is_int8, "TINYINT"),
        (pa.types.is_uint8, "UTINYINT"),
        (pa.types.is_int16, "SMALLINT"),
        (pa.types.is_uint16, "USMALLINT"),
        (pa.types.is_int32, "INTEGER"),
        (pa.types.is_uint32, "UINTEGER"),
        (pa.types.is_int64, "BIGINT"),
        (pa.types.is_uint64, "UBIGINT"),
        (pa.types.is_float32, "FLOAT"),
        (pa.types.is_float64, "DOUBLE"),
        (pa.types.is_string, "VARCHAR"),
        (pa.types.is_binary, "BLOB"),
        (pa.types.is_date32, "DATE"),
    )
    for predicate, duckdb_name in primitive_types:
        if predicate(dtype):
            return vane.sqltype(duckdb_name)

    if pa.types.is_decimal128(dtype):
        return vane.sqltype(f"DECIMAL({dtype.precision},{dtype.scale})")
    if pa.types.is_list(dtype):
        child_type = _arrow_to_duckdb_type(dtype.value_type, original=original)
        return vane.list_type(child_type)
    if pa.types.is_fixed_size_list(dtype):
        child_type = _arrow_to_duckdb_type(dtype.value_type, original=original)
        return vane.array_type(child_type, dtype.list_size)
    if pa.types.is_struct(dtype):
        field_names = [field.name for field in dtype]
        if len(field_names) != len({name.casefold() for name in field_names}):
            raise _invalid_input(
                f"dtype {str(original)!r} is not supported; Struct field names must be unique case-insensitively"
            )
        return vane.struct_type({field.name: _arrow_to_duckdb_type(field.type, original=original) for field in dtype})
    if pa.types.is_timestamp(dtype):
        if dtype.tz is not None:
            raise _invalid_input(
                f"dtype {str(original)!r} is not supported; timezone-aware Arrow timestamps require "
                "a supported TIMESTAMPTZ contract"
            )
        timestamp_types = {
            "s": "TIMESTAMP_S",
            "ms": "TIMESTAMP_MS",
            "us": "TIMESTAMP",
            "ns": "TIMESTAMP_NS",
        }
        try:
            return vane.sqltype(timestamp_types[dtype.unit])
        except KeyError as exc:
            raise _unsupported_dtype(original) from exc
    raise _unsupported_dtype(original)


def _duckdb_to_arrow_type(dtype: Any, *, original: Any) -> Any:
    import pyarrow as pa

    primitive_types: dict[str, Callable[[], Any]] = {
        "boolean": pa.bool_,
        "tinyint": pa.int8,
        "utinyint": pa.uint8,
        "smallint": pa.int16,
        "usmallint": pa.uint16,
        "integer": pa.int32,
        "uinteger": pa.uint32,
        "bigint": pa.int64,
        "ubigint": pa.uint64,
        "float": pa.float32,
        "double": pa.float64,
        "varchar": pa.string,
        "blob": pa.binary,
        "date": pa.date32,
        "timestamp": lambda: pa.timestamp("us"),
        "timestamp_s": lambda: pa.timestamp("s"),
        "timestamp_ms": lambda: pa.timestamp("ms"),
        "timestamp_ns": lambda: pa.timestamp("ns"),
    }
    type_id = str(dtype.id)
    factory = primitive_types.get(type_id)
    if factory is not None:
        return factory()
    if type_id == "decimal":
        children = dict(dtype.children)
        return pa.decimal128(int(children["precision"]), int(children["scale"]))
    if type_id == "list":
        child_type = dict(dtype.children)["child"]
        return pa.list_(_duckdb_to_arrow_type(child_type, original=original))
    if type_id == "array":
        children = dict(dtype.children)
        return pa.list_(
            _duckdb_to_arrow_type(children["child"], original=original),
            int(children["size"]),
        )
    if type_id == "struct":
        return pa.struct(
            [
                pa.field(name, _duckdb_to_arrow_type(child_type, original=original))
                for name, child_type in dtype.children
            ]
        )
    raise _unsupported_dtype(original)


def _canonicalize_dtype(dtype: Any) -> tuple[Any, Any]:
    """Return one validated DuckDB/Arrow representation of an output type."""
    import pyarrow as pa

    if isinstance(dtype, pa.DataType):
        duckdb_type = _arrow_to_duckdb_type(dtype, original=dtype)
        return duckdb_type, _duckdb_to_arrow_type(duckdb_type, original=dtype)

    if isinstance(dtype, str):
        if not dtype.strip():
            raise _invalid_input("dtype must not be empty")
        duckdb_type = vane.sqltype(dtype)
    elif isinstance(dtype, vane.sqltypes.DuckDBPyType):
        duckdb_type = dtype
    else:
        raise _invalid_input(
            f"dtype must be a SQL type string, DuckDBPyType, or supported pyarrow.DataType; got {type(dtype).__name__}"
        )
    return duckdb_type, _duckdb_to_arrow_type(duckdb_type, original=dtype)


def _duckdb_type(dtype: Any) -> Any:
    return _canonicalize_dtype(dtype)[0]


def _dtype_to_arrow(dtype: Any) -> Any:
    return _canonicalize_dtype(dtype)[1]


def _runner_to_task_backend() -> str:
    runner = str(current_config().runner or "").strip().lower()
    return "ray_task" if runner == "ray" else "subprocess_task"


def _runner_to_actor_backend() -> str:
    runner = str(current_config().runner or "").strip().lower()
    return "ray_actor" if runner == "ray" else "subprocess_actor"


def _qualified_name(value: Any, *, kind: str) -> str:
    name = getattr(value, "__qualname__", None)
    if not isinstance(name, str) or not name:
        raise _invalid_input(f"{kind} must expose a non-empty __qualname__")
    return name


def _callable_name(fn: Callable[..., Any]) -> str:
    return _qualified_name(fn, kind="UDF callable")


def _validate_function_udf_callable(fn: Any, *, api: str) -> None:
    if not (inspect.isfunction(fn) or inspect.ismethod(fn)):
        raise TypeError(f"{api} requires a Python function or bound method")
    validate_synchronous_udf_callable(fn)


def _class_name(cls: type) -> str:
    return _qualified_name(cls, kind="UDF class")


def _resolve_udf_name(name: Any, default_name: Callable[[], str]) -> str:
    if name is None:
        return default_name()
    if not isinstance(name, str) or not name:
        raise _invalid_input("name must be a non-empty string")
    return name


def _has_expression(values: Any) -> bool:
    return any(is_expression(value) for value in values)


def _validate_positive_actor_number(actor_number: Any) -> int:
    if isinstance(actor_number, bool):
        raise _invalid_input("actor_number must be a positive integer; bool is not accepted")
    if type(actor_number) is not int or actor_number <= 0:
        raise _invalid_input("actor_number must be a positive integer")
    return int(actor_number)


def _bind_literal_kwargs(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> Callable[..., Any]:
    if not kwargs:
        return fn

    @functools.wraps(fn)
    def call_with_literal_kwargs(*args: Any) -> Any:
        return fn(*args, **kwargs)

    return call_with_literal_kwargs


def _call_or_build_expression(
    *,
    has_expression: bool,
    call_immediately: Callable[[], Any],
    build_expression: Callable[[], vane.Expression],
) -> Any:
    if has_expression:
        return build_expression()
    return call_immediately()


@dataclass(frozen=True)
class _BatchCallLayout:
    positional_input_names: tuple[str, ...]
    keyword_input_names: tuple[str, ...]

    @property
    def input_names(self) -> tuple[str, ...]:
        return self.positional_input_names + self.keyword_input_names


def _batch_function_signature(fn: Callable[..., Any]) -> inspect.Signature:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise _invalid_input("batch UDF signature cannot be inspected") from exc
    variadic = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if variadic:
        names = ", ".join(variadic)
        raise _invalid_input(f"batch UDF signatures cannot use *args or **kwargs: {names}")
    return signature


def _batch_call_layout(
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    context: str,
) -> _BatchCallLayout:
    try:
        signature.bind(*args, **dict(kwargs))
    except TypeError as exc:
        raise _invalid_input(f"{context} does not match batch UDF signature: {exc}") from exc

    positional_parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    positional_input_names = tuple(parameter.name for parameter in positional_parameters[: len(args)])
    keyword_input_names = tuple(kwargs)
    input_names = positional_input_names + keyword_input_names
    if not input_names:
        raise _invalid_input("batch UDFs require at least one input column")
    folded_names = [name.casefold() for name in input_names]
    if len(folded_names) != len(set(folded_names)):
        raise _invalid_input("batch UDF input names must be unique (case-insensitive)")
    return _BatchCallLayout(positional_input_names, keyword_input_names)


def _batch_input_mapping(
    layout: _BatchCallLayout,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    inputs = dict(zip(layout.positional_input_names, args, strict=True))
    inputs.update((name, kwargs[name]) for name in layout.keyword_input_names)
    return inputs


def _arrow_batch_column(table: Any, name: str) -> Any:
    column = table.column(name)
    if column.num_chunks == 1:
        return column.chunk(0)
    return column


def _invoke_batch_callable(
    fn: Callable[..., Any],
    layout: _BatchCallLayout,
    columns: list[Any],
) -> Any:
    positional_count = len(layout.positional_input_names)
    positional = columns[:positional_count]
    keyword = dict(zip(layout.keyword_input_names, columns[positional_count:], strict=True))
    return fn(*positional, **keyword)


def _normalize_batch_result(
    result: Any,
    *,
    output_arrow_type: Any,
    expected_length: int,
    udf_name: str,
) -> Any:
    import pyarrow as pa

    if not isinstance(result, (pa.Array, pa.ChunkedArray)):
        raise _invalid_input(
            f"batch UDF {udf_name!r} must return pyarrow.Array or pyarrow.ChunkedArray; got {type(result).__name__}"
        )
    if len(result) != expected_length:
        raise _invalid_input(f"batch UDF {udf_name!r} returned {len(result)} rows for {expected_length} input rows")
    if not result.type.equals(output_arrow_type):
        try:
            result = result.cast(output_arrow_type)
        except Exception as exc:
            raise _invalid_input(
                f"batch UDF {udf_name!r} returned Arrow type {result.type}, "
                f"which cannot be cast to declared return_dtype {output_arrow_type}"
            ) from exc
    return result


def _execute_batch_callable(
    fn: Callable[..., Any],
    layout: _BatchCallLayout,
    columns: list[Any],
    *,
    output_arrow_type: Any,
    output_column: str,
    udf_name: str,
) -> Any:
    import pyarrow as pa

    expected_length = len(columns[0])
    if any(len(column) != expected_length for column in columns[1:]):
        raise _invalid_input(f"batch UDF {udf_name!r} input columns must have equal lengths")
    result = ensure_synchronous_udf_result(_invoke_batch_callable(fn, layout, columns))
    normalized = _normalize_batch_result(
        result,
        output_arrow_type=output_arrow_type,
        expected_length=expected_length,
        udf_name=udf_name,
    )
    return pa.table({output_column: normalized})


def _call_batch_eager(
    fn: Callable[..., Any],
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    output_arrow_type: Any,
    output_column: str,
    udf_name: str,
) -> Any:
    import pyarrow as pa

    layout = _batch_call_layout(signature, args, kwargs, context=f"batch UDF {udf_name!r} call")
    inputs = _batch_input_mapping(layout, args, kwargs)
    columns = list(inputs.values())
    for index, column in enumerate(columns):
        if not isinstance(column, (pa.Array, pa.ChunkedArray)):
            raise _invalid_input(
                f"eager batch UDF inputs must be pyarrow.Array or pyarrow.ChunkedArray; got {type(column).__name__}"
            )
        if isinstance(column, pa.ChunkedArray) and column.num_chunks == 1:
            columns[index] = column.chunk(0)
    result_table = _execute_batch_callable(
        fn,
        layout,
        columns,
        output_arrow_type=output_arrow_type,
        output_column=output_column,
        udf_name=udf_name,
    )
    result = result_table.column(output_column)
    if result.num_chunks == 1:
        return result.chunk(0)
    return result


def _build_batch_function_adapter(
    fn: Callable[..., Any],
    layout: _BatchCallLayout,
    output_column: str,
    output_arrow_type: Any,
    udf_name: str,
) -> Callable[[Any], Any]:
    captured_input_names = tuple(layout.input_names)

    @functools.wraps(fn)
    def batch_adapter(table: Any) -> Any:
        columns = [_arrow_batch_column(table, name) for name in captured_input_names]
        return _execute_batch_callable(
            fn,
            layout,
            columns,
            output_arrow_type=output_arrow_type,
            output_column=output_column,
            udf_name=udf_name,
        )

    return batch_adapter


def _expand_batch_expression(expression: vane.Expression, *, unnest: bool) -> vane.Expression:
    if not unnest:
        return expression
    return vane.FunctionExpression("unnest", expression)


def _build_map_expression(
    fn: Callable[..., Any],
    name: str,
    return_dtype: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> vane.Expression:
    if _has_expression(kwargs.values()):
        raise _invalid_input(
            "Expression keyword arguments are not supported for vane.func; pass UDF input expressions positionally"
        )
    if return_dtype is None:
        raise _invalid_input("return_dtype is required for expression UDF")

    bound_fn = _bind_literal_kwargs(fn, kwargs)
    expr_args = tuple(as_expression(arg) for arg in args)
    return _native._VaneUDFMapExpression(
        bound_fn,
        name,
        _duckdb_type(return_dtype),
        _runner_to_task_backend(),
        uuid.uuid4().hex,
        *expr_args,
    )


@dataclass(frozen=True)
class _ClassCallContract:
    signature: inspect.Signature
    positional_parameters: tuple[inspect.Parameter, ...]
    min_required_positional: int
    max_positional: int | None
    has_varargs: bool
    required_keyword_only: tuple[str, ...]


def _class_call_contract(user_class: type) -> _ClassCallContract:
    try:
        descriptor = inspect.getattr_static(user_class, "__call__")
    except AttributeError as exc:
        raise _invalid_input(
            "class __call__ signature cannot be inspected; define an ordinary method, staticmethod, or classmethod"
        ) from exc

    drop_receiver = False
    if isinstance(descriptor, staticmethod):
        callable_object = descriptor.__func__
    elif isinstance(descriptor, classmethod):
        callable_object = descriptor.__func__
        drop_receiver = True
    elif inspect.isfunction(descriptor):
        callable_object = descriptor
        drop_receiver = True
    else:
        raise _invalid_input(
            "class __call__ uses an unsupported descriptor; define an ordinary method, staticmethod, or classmethod"
        )

    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError) as exc:
        raise _invalid_input(
            "class __call__ signature cannot be inspected; define an ordinary method, staticmethod, or classmethod"
        ) from exc

    parameters = list(signature.parameters.values())
    if drop_receiver:
        if not parameters or parameters[0].kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise _invalid_input("class __call__ descriptor does not define a positional receiver")
        parameters = parameters[1:]
        signature = signature.replace(parameters=parameters)

    positional_parameters = tuple(
        parameter
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    return _ClassCallContract(
        signature=signature,
        positional_parameters=positional_parameters,
        min_required_positional=sum(
            parameter.default is inspect.Parameter.empty for parameter in positional_parameters
        ),
        max_positional=None
        if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters)
        else len(positional_parameters),
        has_varargs=any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters),
        required_keyword_only=tuple(
            parameter.name
            for parameter in parameters
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default is inspect.Parameter.empty
        ),
    )


def _bind_class_call(
    contract: _ClassCallContract,
    arg_count: int,
    call_kwargs: Mapping[str, Any],
    *,
    context: str,
) -> None:
    try:
        contract.signature.bind(*([object()] * arg_count), **dict(call_kwargs))
    except TypeError as exc:
        raise _invalid_input(f"{context} does not match class __call__ signature: {exc}") from exc


def _expression_class_input_names(
    user_class: type,
    arg_count: int,
    call_kwargs: Mapping[str, Any],
) -> list[str]:
    contract = _class_call_contract(user_class)
    if contract.has_varargs:
        raise _invalid_input(
            "input_names cannot be inferred from class __call__ with *args; "
            "SQL registration may pass explicit input_names"
        )
    _bind_class_call(contract, arg_count, call_kwargs, context="vane.cls expression call")
    return [parameter.name for parameter in contract.positional_parameters[:arg_count]]


def _sql_class_input_names(
    user_class: type,
    arg_count: int,
    explicit_input_names: list[str] | None,
) -> list[str]:
    if arg_count == 0:
        raise _invalid_input(
            "zero-input vane.cls SQL UDFs are not supported; eager zero-argument calls remain available"
        )

    contract = _class_call_contract(user_class)
    if contract.required_keyword_only:
        names = ", ".join(contract.required_keyword_only)
        raise _invalid_input(
            f"SQL vane.cls registration cannot satisfy required keyword-only class __call__ parameter(s): {names}"
        )
    if explicit_input_names is None and contract.has_varargs:
        raise _invalid_input(
            "input_names cannot be inferred from class __call__ with *args; "
            "pass explicit input_names to attach_function"
        )
    if arg_count < contract.min_required_positional:
        raise _invalid_input(
            "SQL vane.cls registration requires at least "
            f"{contract.min_required_positional} positional input(s); received {arg_count}"
        )
    if contract.max_positional is not None and arg_count > contract.max_positional:
        raise _invalid_input(
            "SQL vane.cls registration accepts at most "
            f"{contract.max_positional} positional input(s); received {arg_count}"
        )
    _bind_class_call(contract, arg_count, {}, context="SQL vane.cls registration")
    if explicit_input_names is not None:
        return explicit_input_names
    return [parameter.name for parameter in contract.positional_parameters[:arg_count]]


def _sql_batch_function_input_names(
    signature: inspect.Signature,
    arg_count: int,
    explicit_input_names: list[str] | None,
) -> list[str]:
    if arg_count == 0:
        raise _invalid_input("zero-input vane.func.batch SQL UDFs are not supported")
    parameters = list(signature.parameters.values())
    required_keyword_only = [
        parameter.name
        for parameter in parameters
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default is inspect.Parameter.empty
    ]
    if required_keyword_only:
        names = ", ".join(required_keyword_only)
        raise _invalid_input(
            f"SQL vane.func.batch registration cannot satisfy required keyword-only parameter(s): {names}"
        )
    try:
        signature.bind(*([object()] * arg_count))
    except TypeError as exc:
        raise _invalid_input(f"SQL vane.func.batch registration does not match UDF signature: {exc}") from exc
    if explicit_input_names is not None:
        return explicit_input_names
    positional_parameters = [
        parameter
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return [parameter.name for parameter in positional_parameters[:arg_count]]


def _build_row_actor_class(
    user_class: type,
    init_args: tuple[Any, ...],
    init_kwargs: Mapping[str, Any],
    input_names: list[str],
    output_column: str,
    output_arrow_type: Any,
    call_kwargs: Mapping[str, Any] | None = None,
) -> type:
    import pyarrow as pa

    captured_init_kwargs = dict(init_kwargs)
    captured_call_kwargs = dict(call_kwargs or {})
    captured_input_names = list(input_names)

    class _VaneRowActorAdapter:
        def __init__(self) -> None:
            self._instance = ensure_synchronous_udf_result(user_class(*init_args, **captured_init_kwargs))

        def __call__(self, table: Any) -> Any:
            columns = [table.column(name).to_pylist() for name in captured_input_names]
            out: list[Any] = []
            for row in zip(*columns, strict=True):
                if any(value is None for value in row):
                    out.append(None)
                    continue
                out.append(ensure_synchronous_udf_result(self._instance(*row, **captured_call_kwargs)))
            if pa.types.is_timestamp(output_arrow_type):
                for value in out:
                    if isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None:
                        raise _invalid_input(
                            "TIMESTAMP is timezone-naive; use a naive datetime or a supported TIMESTAMPTZ contract"
                        )
            return pa.table({output_column: pa.array(out, type=output_arrow_type)})

    _VaneRowActorAdapter.__name__ = f"_{_class_name(user_class)}RowActor"
    _VaneRowActorAdapter.__qualname__ = _VaneRowActorAdapter.__name__
    return _VaneRowActorAdapter


def _build_batch_actor_class(
    user_class: type,
    init_args: tuple[Any, ...],
    init_kwargs: Mapping[str, Any],
    layout: _BatchCallLayout,
    output_column: str,
    output_arrow_type: Any,
    udf_name: str,
) -> type:
    captured_init_kwargs = dict(init_kwargs)
    captured_input_names = tuple(layout.input_names)

    class _VaneBatchActorAdapter:
        def __init__(self) -> None:
            self._instance = ensure_synchronous_udf_result(user_class(*init_args, **captured_init_kwargs))

        def __call__(self, table: Any) -> Any:
            columns = [_arrow_batch_column(table, name) for name in captured_input_names]
            return _execute_batch_callable(
                self._instance,
                layout,
                columns,
                output_arrow_type=output_arrow_type,
                output_column=output_column,
                udf_name=udf_name,
            )

    _VaneBatchActorAdapter.__name__ = f"_{_class_name(user_class)}BatchActor"
    _VaneBatchActorAdapter.__qualname__ = _VaneBatchActorAdapter.__name__
    return _VaneBatchActorAdapter


def _validate_sql_actor_callable(fn: Any) -> None:
    if not inspect.isclass(fn):
        raise _invalid_input("actor UDF backends require a callable class")
    instance_call_descriptor = next(
        (base.__dict__["__call__"] for base in fn.__mro__ if "__call__" in base.__dict__),
        None,
    )
    if instance_call_descriptor is None or not (
        callable(instance_call_descriptor) or hasattr(instance_call_descriptor, "__get__")
    ):
        raise _invalid_input("actor UDF backends require a callable class whose instances implement __call__")
    if inspect.isabstract(fn):
        raise _invalid_input("actor UDF backends require a concrete callable class")
    try:
        constructor_signature = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise _invalid_input(
            "callable class UDF constructor signature cannot be inspected; "
            "use vane.cls or vane.cls.batch for explicit constructor configuration"
        ) from exc
    try:
        constructor_signature.bind()
    except TypeError as exc:
        raise _invalid_input(
            "callable class UDF constructors must be zero-argument; "
            "use vane.cls or vane.cls.batch to capture constructor arguments"
        ) from exc


class VaneFunction:
    """Synchronous Python function or bound-method scalar UDF wrapper.

    When used as an instance-method descriptor, the serialized callable
    captures that instance's current snapshot. Reusable instance-local state is
    provided by :func:`vane.cls`; it is neither shared nor durable and may be
    reset by Actor reconstruction.
    """

    def __init__(
        self,
        fn: _PythonFunction,
        *,
        return_dtype: Any | None = None,
        name: str | None = None,
    ):
        _validate_function_udf_callable(fn, api="vane.func")
        self._fn = fn
        self._return_dtype = return_dtype
        self._name = _resolve_udf_name(name, lambda: _callable_name(fn))
        functools.update_wrapper(self, fn)

    @property
    def return_dtype(self) -> Any | None:
        return self._return_dtype

    @property
    def python_function(self) -> _PythonFunction:
        return self._fn

    @property
    def sql_name(self) -> str:
        return self._name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _call_or_build_expression(
            has_expression=_has_expression(args) or _has_expression(kwargs.values()),
            call_immediately=lambda: ensure_synchronous_udf_result(self._fn(*args, **kwargs)),
            build_expression=lambda: _build_map_expression(self._fn, self._name, self._return_dtype, args, kwargs),
        )

    def __get__(self, instance: Any, owner: Any | None = None) -> Any:
        if instance is None:
            return self
        bound_fn = self._fn.__get__(instance, owner)
        return VaneFunction(bound_fn, return_dtype=self._return_dtype, name=self._name)


class VaneBatchFunction:
    """Synchronous Python function or bound-method Arrow column batch UDF wrapper."""

    def __init__(
        self,
        fn: _PythonFunction,
        *,
        return_dtype: Any,
        name: str | None = None,
        batch_size: int | None = None,
        unnest: bool = False,
        gpus: float | None = None,
    ) -> None:
        _validate_function_udf_callable(fn, api="vane.func.batch")
        if return_dtype is None:
            raise _invalid_input("return_dtype is required for vane.func.batch")
        normalized_return_dtype, return_arrow_dtype = _canonicalize_dtype(return_dtype)
        if unnest:
            import pyarrow as pa

            if not pa.types.is_struct(return_arrow_dtype):
                raise _invalid_input("unnest=True requires a Struct return_dtype")
        self._fn = fn
        self._signature = _batch_function_signature(fn)
        self._return_dtype = normalized_return_dtype
        self._return_arrow_dtype = return_arrow_dtype
        self._name = _resolve_udf_name(name, lambda: _callable_name(fn))
        self._batch_size = _normalize_batch_size(batch_size)
        self._unnest = bool(unnest)
        self._gpus = _normalize_gpus(gpus)
        functools.update_wrapper(self, fn)

    @property
    def python_function(self) -> _PythonFunction:
        return self._fn

    @property
    def signature(self) -> inspect.Signature:
        return self._signature

    @property
    def return_dtype(self) -> Any:
        return self._return_dtype

    @property
    def return_arrow_dtype(self) -> Any:
        return self._return_arrow_dtype

    @property
    def sql_name(self) -> str:
        return self._name

    @property
    def batch_size(self) -> int | None:
        return self._batch_size

    @property
    def unnest(self) -> bool:
        return self._unnest

    @property
    def gpus(self) -> float | None:
        return self._gpus

    def adapter(self, layout: _BatchCallLayout) -> Callable[[Any], Any]:
        return _build_batch_function_adapter(
            self.python_function,
            layout,
            self.sql_name,
            self.return_arrow_dtype,
            self.sql_name,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _call_or_build_expression(
            has_expression=_has_expression(args) or _has_expression(kwargs.values()),
            call_immediately=lambda: _call_batch_eager(
                self.python_function,
                self.signature,
                args,
                kwargs,
                output_arrow_type=self.return_arrow_dtype,
                output_column=self.sql_name,
                udf_name=self.sql_name,
            ),
            build_expression=lambda: self._build_expression(args, kwargs),
        )

    def _build_expression(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> vane.Expression:
        layout = _batch_call_layout(
            self.signature,
            args,
            kwargs,
            context=f"vane.func.batch {self.sql_name!r} expression call",
        )
        expression = _build_map_batches_expression(
            self.adapter(layout),
            name=self.sql_name,
            inputs=_batch_input_mapping(layout, args, kwargs),
            schema={self.sql_name: self.return_dtype},
            batch_size=self.batch_size,
            row_preserving=True,
            gpus=self.gpus,
        )
        return _expand_batch_expression(expression, unnest=self.unnest)

    def __get__(self, instance: Any, owner: Any | None = None) -> Any:
        if instance is None:
            return self
        bound_fn = self._fn.__get__(instance, owner)
        return VaneBatchFunction(
            bound_fn,
            return_dtype=self.return_dtype,
            name=self.sql_name,
            batch_size=self.batch_size,
            unnest=self.unnest,
            gpus=self.gpus,
        )


class VaneClass:
    def __init__(
        self,
        class_: type,
        *,
        actor_number: int | None,
        return_dtype: Any | None,
        name: str | None = None,
        gpus: float | None = 0,
    ) -> None:
        if not inspect.isclass(class_):
            raise TypeError("vane.cls requires a class")
        validate_synchronous_udf_callable(class_)
        if return_dtype is None:
            raise _invalid_input("return_dtype is required for vane.cls")
        normalized_return_dtype, return_arrow_dtype = _canonicalize_dtype(return_dtype)
        self._class = class_
        self._actor_number = _validate_positive_actor_number(actor_number)
        self._return_dtype = normalized_return_dtype
        self._return_arrow_dtype = return_arrow_dtype
        self._name = _resolve_udf_name(name, lambda: _class_name(class_))
        self._gpus = 0.0 if gpus is None else _normalize_gpus(gpus)
        functools.update_wrapper(self, class_, updated=())

    @property
    def user_class(self) -> type:
        return self._class

    @property
    def actor_number(self) -> int:
        return self._actor_number

    @property
    def return_dtype(self) -> Any:
        return self._return_dtype

    @property
    def return_arrow_dtype(self) -> Any:
        return self._return_arrow_dtype

    @property
    def sql_name(self) -> str:
        return self._name

    @property
    def gpus(self) -> float | int | None:
        return self._gpus

    def __call__(self, *args: Any, **kwargs: Any) -> VaneClassInstance:
        return VaneClassInstance(self, args, kwargs)


class VaneClassInstance:
    def __init__(self, decorator: VaneClass, init_args: tuple[Any, ...], init_kwargs: Mapping[str, Any]) -> None:
        self._decorator = decorator
        self._init_args = tuple(init_args)
        self._init_kwargs = dict(init_kwargs)
        self._eager_instance: Any | None = None

    @property
    def sql_name(self) -> str:
        return self._decorator.sql_name

    @property
    def actor_number(self) -> int:
        return self._decorator.actor_number

    @property
    def gpus(self) -> float | int | None:
        return self._decorator.gpus

    @property
    def return_dtype(self) -> Any:
        return self._decorator.return_dtype

    @property
    def return_arrow_dtype(self) -> Any:
        return self._decorator.return_arrow_dtype

    @property
    def user_class(self) -> type:
        return self._decorator.user_class

    def _instance(self) -> Any:
        if self._eager_instance is None:
            self._eager_instance = ensure_synchronous_udf_result(self.user_class(*self._init_args, **self._init_kwargs))
        return self._eager_instance

    def input_names_for_expression(self, arg_count: int, call_kwargs: Mapping[str, Any]) -> list[str]:
        return _expression_class_input_names(self.user_class, arg_count, call_kwargs)

    def actor_class(self, input_names: list[str], call_kwargs: Mapping[str, Any] | None = None) -> type:
        return _build_row_actor_class(
            self.user_class,
            self._init_args,
            self._init_kwargs,
            input_names,
            self.sql_name,
            self.return_arrow_dtype,
            call_kwargs,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _call_or_build_expression(
            has_expression=_has_expression(args) or _has_expression(kwargs.values()),
            call_immediately=lambda: ensure_synchronous_udf_result(self._instance()(*args, **kwargs)),
            build_expression=lambda: self._build_expression(args, kwargs),
        )

    def _build_expression(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> vane.Expression:
        if _has_expression(kwargs.values()):
            raise _invalid_input(
                "Expression keyword arguments are not supported for vane.cls; pass UDF input expressions positionally"
            )
        input_names = self.input_names_for_expression(len(args), kwargs)
        actor_class = self.actor_class(input_names, kwargs)
        return _build_actor_map_batches_expression(
            actor_class,
            name=self.sql_name,
            inputs=dict(zip(input_names, args, strict=True)),
            schema={self.sql_name: self.return_dtype},
            batch_size=None,
            row_preserving=True,
            actor_number=self.actor_number,
            gpus=self.gpus,
        )


class VaneClassBatch:
    def __init__(
        self,
        class_: type,
        *,
        actor_number: int | None,
        return_dtype: Any,
        name: str | None = None,
        batch_size: int | None = None,
        unnest: bool = False,
        gpus: float | None = 0,
    ) -> None:
        if not inspect.isclass(class_):
            raise TypeError("vane.cls.batch requires a class")
        validate_synchronous_udf_callable(class_)
        if return_dtype is None:
            raise _invalid_input("return_dtype is required for vane.cls.batch")
        normalized_return_dtype, return_arrow_dtype = _canonicalize_dtype(return_dtype)
        if unnest:
            import pyarrow as pa

            if not pa.types.is_struct(return_arrow_dtype):
                raise _invalid_input("unnest=True requires a Struct return_dtype")
        signature = _class_call_contract(class_).signature
        variadic = [
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        if variadic:
            names = ", ".join(variadic)
            raise _invalid_input(f"batch UDF signatures cannot use *args or **kwargs: {names}")
        self._class = class_
        self._signature = signature
        self._actor_number = _validate_positive_actor_number(actor_number)
        self._return_dtype = normalized_return_dtype
        self._return_arrow_dtype = return_arrow_dtype
        self._name = _resolve_udf_name(name, lambda: _class_name(class_))
        self._batch_size = _normalize_batch_size(batch_size)
        self._unnest = bool(unnest)
        self._gpus = 0.0 if gpus is None else _normalize_gpus(gpus)
        functools.update_wrapper(self, class_, updated=())

    @property
    def user_class(self) -> type:
        return self._class

    @property
    def actor_number(self) -> int:
        return self._actor_number

    @property
    def signature(self) -> inspect.Signature:
        return self._signature

    @property
    def return_dtype(self) -> Any:
        return self._return_dtype

    @property
    def return_arrow_dtype(self) -> Any:
        return self._return_arrow_dtype

    @property
    def sql_name(self) -> str:
        return self._name

    @property
    def batch_size(self) -> int | None:
        return self._batch_size

    @property
    def unnest(self) -> bool:
        return self._unnest

    @property
    def gpus(self) -> float | int | None:
        return self._gpus

    def __call__(self, *args: Any, **kwargs: Any) -> VaneClassBatchInstance:
        return VaneClassBatchInstance(self, args, kwargs)


class VaneClassBatchInstance:
    def __init__(self, decorator: VaneClassBatch, init_args: tuple[Any, ...], init_kwargs: Mapping[str, Any]) -> None:
        self._decorator = decorator
        self._init_args = tuple(init_args)
        self._init_kwargs = dict(init_kwargs)
        self._eager_instance: Any | None = None

    @property
    def sql_name(self) -> str:
        return self._decorator.sql_name

    @property
    def actor_number(self) -> int:
        return self._decorator.actor_number

    @property
    def signature(self) -> inspect.Signature:
        return self._decorator.signature

    @property
    def return_dtype(self) -> Any:
        return self._decorator.return_dtype

    @property
    def return_arrow_dtype(self) -> Any:
        return self._decorator.return_arrow_dtype

    @property
    def batch_size(self) -> int | None:
        return self._decorator.batch_size

    @property
    def unnest(self) -> bool:
        return self._decorator.unnest

    @property
    def gpus(self) -> float | int | None:
        return self._decorator.gpus

    @property
    def user_class(self) -> type:
        return self._decorator.user_class

    def _instance(self) -> Any:
        if self._eager_instance is None:
            self._eager_instance = ensure_synchronous_udf_result(self.user_class(*self._init_args, **self._init_kwargs))
        return self._eager_instance

    def actor_class(self, layout: _BatchCallLayout) -> type:
        return _build_batch_actor_class(
            self.user_class,
            self._init_args,
            self._init_kwargs,
            layout,
            self.sql_name,
            self.return_arrow_dtype,
            self.sql_name,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _call_or_build_expression(
            has_expression=_has_expression(args) or _has_expression(kwargs.values()),
            call_immediately=lambda: _call_batch_eager(
                self._instance(),
                self.signature,
                args,
                kwargs,
                output_arrow_type=self.return_arrow_dtype,
                output_column=self.sql_name,
                udf_name=self.sql_name,
            ),
            build_expression=lambda: self._build_expression(args, kwargs),
        )

    def _build_expression(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> vane.Expression:
        layout = _batch_call_layout(
            self.signature,
            args,
            kwargs,
            context=f"vane.cls.batch {self.sql_name!r} expression call",
        )
        expression = _build_actor_map_batches_expression(
            self.actor_class(layout),
            name=self.sql_name,
            inputs=_batch_input_mapping(layout, args, kwargs),
            schema={self.sql_name: self.return_dtype},
            batch_size=self.batch_size,
            row_preserving=True,
            actor_number=self.actor_number,
            gpus=self.gpus,
        )
        return _expand_batch_expression(expression, unnest=self.unnest)


def _normalize_schema(schema: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(schema, Mapping) or not schema:
        raise _invalid_input("schema must be a non-empty mapping")
    if len(schema) != 1:
        raise _invalid_input("map_batches expression requires exactly one output column")
    name, dtype = next(iter(schema.items()))
    if not isinstance(name, str) or not name:
        raise _invalid_input("schema output name must be a non-empty string")
    return {name: _duckdb_type(dtype)}


def _unwrap_vane_function(fn: Any) -> tuple[Callable[..., Any], Any | None, str | None]:
    if isinstance(fn, VaneFunction):
        return fn.python_function, fn.return_dtype, fn.sql_name
    if callable(fn):
        return fn, None, _callable_name(fn)
    raise TypeError("vane.attach_function requires a callable or vane.func object")


def _normalize_sql_type_list(parameters: Any) -> list[Any] | None:
    import pyarrow as pa

    if parameters is None:
        return None
    if not isinstance(parameters, (list, tuple)):
        raise _invalid_input("parameters must be a list or tuple of DuckDB types")
    normalized: list[Any] = []
    for parameter in parameters:
        if isinstance(parameter, pa.DataType):
            normalized.append(_canonicalize_dtype(parameter)[0])
        elif isinstance(parameter, str):
            normalized.append(vane.sqltype(parameter))
        elif isinstance(parameter, vane.sqltypes.DuckDBPyType):
            normalized.append(parameter)
        else:
            raise _invalid_input(
                "parameters entries must be SQL type strings, DuckDBPyType, or supported pyarrow.DataType; "
                f"got {type(parameter).__name__}"
            )
    return normalized


def _require_sql_type_list(parameters: Any, message: str) -> list[Any]:
    normalized = _normalize_sql_type_list(parameters)
    if normalized is None:
        raise _invalid_input(message)
    return normalized


def _normalize_sql_input_names(input_names: Any) -> list[str]:
    if not isinstance(input_names, (list, tuple)) or not input_names:
        raise _invalid_input("input_names must be a non-empty list or tuple")
    normalized: list[str] = []
    folded_names: set[str] = set()
    for name in input_names:
        if not isinstance(name, str) or not name:
            raise _invalid_input("input_names must contain only non-empty strings")
        folded = name.casefold()
        if folded in folded_names:
            raise _invalid_input(f"input_names must be unique (case-insensitive); duplicate name: {name!r}")
        normalized.append(name)
        folded_names.add(folded)
    return normalized


def _normalize_batch_size(batch_size: Any) -> int | None:
    if batch_size is None:
        return None
    if isinstance(batch_size, bool) or type(batch_size) is not int or batch_size <= 0:
        raise _invalid_input("batch_size must be a positive integer")
    return int(batch_size)


def _normalize_gpus(gpus: Any) -> float | None:
    if gpus is None:
        return None
    if isinstance(gpus, bool) or not isinstance(gpus, Real):
        raise _invalid_input("gpus must be a non-negative number")
    value = float(gpus)
    if not isfinite(value) or value < 0:
        raise _invalid_input("gpus must be a non-negative number")
    return value


def _resolve_alias(alias: Any, default_name: Any) -> str:
    resolved = default_name if alias is None else alias
    if not isinstance(resolved, str) or not resolved:
        raise _invalid_input("alias is required and must be a non-empty string")
    return resolved


def _normalize_input_mapping(inputs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(inputs, Mapping) or not inputs:
        raise _invalid_input("inputs must be a non-empty mapping")
    return {str(name): value for name, value in inputs.items()}


def _normalize_inputs(inputs: Mapping[str, Any]) -> tuple[list[str], tuple[vane.Expression, ...]]:
    normalized = _normalize_input_mapping(inputs)
    names: list[str] = []
    exprs: list[vane.Expression] = []
    for name, value in normalized.items():
        names.append(name)
        exprs.append(as_expression(value))
    return names, tuple(exprs)


def _build_map_batches_expression(
    fn: Callable[[Any], Any],
    *,
    name: str | None,
    inputs: Mapping[str, Any],
    schema: Mapping[str, Any] | None,
    batch_size: int | None,
    row_preserving: bool,
    gpus: float | None,
    execution_backend: str | None = None,
    actor_number: int | None = None,
) -> vane.Expression:
    input_names, exprs = _normalize_inputs(inputs)
    normalized_schema = _normalize_schema(schema)
    resolved_backend = _runner_to_task_backend() if execution_backend is None else execution_backend
    return _native._VaneUDFMapBatchesExpression(
        fn,
        _resolve_udf_name(name, lambda: _callable_name(fn)),
        normalized_schema,
        resolved_backend,
        input_names,
        batch_size,
        bool(row_preserving),
        gpus,
        actor_number,
        uuid.uuid4().hex,
        *exprs,
    )


def _build_actor_map_batches_expression(
    fn: Callable[[Any], Any],
    *,
    name: str | None,
    inputs: Mapping[str, Any],
    schema: Mapping[str, Any] | None,
    batch_size: int | None,
    row_preserving: bool,
    actor_number: int,
    gpus: float | None,
) -> vane.Expression:
    normalized_actor_number = _validate_positive_actor_number(actor_number)
    return _build_map_batches_expression(
        fn,
        name=name,
        inputs=inputs,
        schema=schema,
        batch_size=batch_size,
        row_preserving=row_preserving,
        gpus=0 if gpus is None else gpus,
        execution_backend=_runner_to_actor_backend(),
        actor_number=normalized_actor_number,
    )


def func(
    fn: _PythonFunction | None = None,
    *,
    return_dtype: Any | None = None,
    name: str | None = None,
) -> VaneFunction | Callable[[_PythonFunction], VaneFunction]:
    """Decorate a synchronous Python function or bound method as a scalar expression UDF."""
    if fn is None:
        return lambda actual_fn: VaneFunction(actual_fn, return_dtype=return_dtype, name=name)
    return VaneFunction(fn, return_dtype=return_dtype, name=name)


def _func_batch(
    *,
    return_dtype: Any,
    name: str | None = None,
    batch_size: int | None = None,
    unnest: bool = False,
    gpus: float | None = None,
) -> Callable[[_PythonFunction], VaneBatchFunction]:
    """Decorate a synchronous Python function or bound method as an Arrow column batch UDF.

    Each decorated input is delivered as ``pyarrow.Array`` or
    ``pyarrow.ChunkedArray``. The function must return one of those column
    types with the same row count. Use a Struct ``return_dtype`` for a logical
    value with multiple fields and ``unnest=True`` to project those fields as
    separate columns.
    """
    return lambda actual_fn: VaneBatchFunction(
        actual_fn,
        return_dtype=return_dtype,
        name=name,
        batch_size=batch_size,
        unnest=unnest,
        gpus=gpus,
    )


func.batch = _func_batch  # type: ignore[attr-defined]


def _cls(
    class_: type | None = None,
    *,
    actor_number: int | None = None,
    return_dtype: Any | None = None,
    name: str | None = None,
    gpus: float | None = 0,
) -> VaneClass | Callable[[type], VaneClass]:
    """Decorate a callable class as an Actor-backed scalar expression UDF.

    ``actor_number`` creates that many independent class instances. Batches
    are assigned to available Actors without affinity or global ordering, so
    instance state is local and must not be used as shared query state. An
    Actor may be reconstructed and an in-flight call may be retried after a
    failure, which can reset local state or replay user code. External side
    effects must therefore be idempotent; exactly-once execution is not
    provided.
    """
    if class_ is None:
        return lambda actual_class: VaneClass(
            actual_class,
            actor_number=actor_number,
            return_dtype=return_dtype,
            name=name,
            gpus=gpus,
        )
    return VaneClass(
        class_,
        actor_number=actor_number,
        return_dtype=return_dtype,
        name=name,
        gpus=gpus,
    )


def _cls_batch(
    *,
    actor_number: int | None = None,
    return_dtype: Any,
    name: str | None = None,
    batch_size: int | None = None,
    unnest: bool = False,
    gpus: float | None = 0,
) -> Callable[[type], VaneClassBatch]:
    """Decorate an Actor-backed row-preserving Arrow column batch class.

    Each Actor owns an independent class instance. Batches have no Actor
    affinity or global ordering, and failures may reconstruct an Actor and
    retry an in-flight call. Instance state is consequently local,
    reconstructible cache state rather than durable query state; external
    side effects must be idempotent because execution is not exactly once.
    """
    return lambda actual_class: VaneClassBatch(
        actual_class,
        actor_number=actor_number,
        return_dtype=return_dtype,
        name=name,
        batch_size=batch_size,
        unnest=unnest,
        gpus=gpus,
    )


cls = _cls
cls.batch = _cls_batch  # type: ignore[attr-defined]


@dataclass(frozen=True)
class _PreparedScalarSQLRegistration:
    alias: str
    udf: Callable[..., Any]
    parameters: list[Any]
    return_type: Any
    replace: bool

    def apply(self, connection: Any) -> None:
        connection._create_vane_function(
            self.alias,
            self.udf,
            parameters=self.parameters,
            return_type=self.return_type,
            replace=self.replace,
        )


@dataclass(frozen=True)
class _PreparedBatchSQLRegistration:
    alias: str
    udf: Callable[..., Any]
    input_names: list[str]
    schema: dict[str, Any]
    parameters: list[Any]
    batch_size: int | None
    gpus: float | None
    actor_number: int | None
    row_preserving: bool
    replace: bool

    def apply(self, connection: Any) -> None:
        connection._create_vane_batch_function(
            self.alias,
            self.udf,
            input_names=self.input_names,
            schema=self.schema,
            parameters=self.parameters,
            batch_size=self.batch_size,
            gpus=self.gpus,
            actor_number=self.actor_number,
            row_preserving=self.row_preserving,
            replace=self.replace,
        )


_PreparedSQLRegistration = _PreparedScalarSQLRegistration | _PreparedBatchSQLRegistration


def _reject_attach_override(option: str, value: Any, *, kind: str, decorator: str) -> None:
    if value is not None:
        raise _invalid_input(
            f"{option} cannot override {kind} instance configuration during SQL registration; "
            f"configure it on @{decorator}"
        )


def _preflight_vane_class_instance(
    fn_or_instance: VaneClassInstance,
    *,
    alias: str | None,
    parameters: Any,
    input_names: Any,
    return_dtype: Any,
    schema: Any,
    batch_size: int | None,
    gpus: Any,
    actor_number: Any,
    replace: bool,
) -> _PreparedBatchSQLRegistration:
    validate_synchronous_udf_callable(fn_or_instance.user_class)
    _reject_attach_override(
        "return_dtype",
        return_dtype,
        kind="vane.cls",
        decorator="vane.cls",
    )
    if schema is not None:
        raise _invalid_input("schema is not valid for SQL vane.cls registration; use return_dtype on @vane.cls")
    _reject_attach_override("gpus", gpus, kind="vane.cls", decorator="vane.cls")
    _reject_attach_override("actor_number", actor_number, kind="vane.cls", decorator="vane.cls")
    normalized_parameters = _require_sql_type_list(
        parameters,
        "parameters is required for SQL vane.cls registration",
    )
    explicit_input_names = None if input_names is None else _normalize_sql_input_names(input_names)
    if explicit_input_names is not None and len(explicit_input_names) != len(normalized_parameters):
        raise _invalid_input("input_names count must match parameters count")
    normalized_input_names = _sql_class_input_names(
        fn_or_instance.user_class,
        len(normalized_parameters),
        explicit_input_names,
    )
    normalized_return_dtype, _ = _canonicalize_dtype(fn_or_instance.return_dtype)
    return _PreparedBatchSQLRegistration(
        alias=_resolve_alias(alias, fn_or_instance.sql_name),
        udf=fn_or_instance.actor_class(normalized_input_names),
        input_names=normalized_input_names,
        schema={fn_or_instance.sql_name: normalized_return_dtype},
        parameters=normalized_parameters,
        batch_size=_normalize_batch_size(batch_size),
        gpus=_normalize_gpus(fn_or_instance.gpus),
        actor_number=_validate_positive_actor_number(fn_or_instance.actor_number),
        row_preserving=True,
        replace=replace,
    )


def _preflight_vane_class_batch_instance(
    fn_or_instance: VaneClassBatchInstance,
    *,
    alias: str | None,
    parameters: Any,
    input_names: Any,
    return_dtype: Any,
    schema: Mapping[str, Any] | None,
    batch_size: int | None,
    gpus: Any,
    actor_number: Any,
    replace: bool,
) -> _PreparedBatchSQLRegistration:
    validate_synchronous_udf_callable(fn_or_instance.user_class)
    _reject_attach_override(
        "return_dtype",
        return_dtype,
        kind="vane.cls.batch",
        decorator="vane.cls.batch",
    )
    _reject_attach_override("gpus", gpus, kind="vane.cls.batch", decorator="vane.cls.batch")
    _reject_attach_override(
        "actor_number",
        actor_number,
        kind="vane.cls.batch",
        decorator="vane.cls.batch",
    )
    if schema is not None:
        raise _invalid_input(
            "schema is not valid for SQL vane.cls.batch registration; use return_dtype on @vane.cls.batch"
        )
    _reject_attach_override(
        "batch_size",
        batch_size,
        kind="vane.cls.batch",
        decorator="vane.cls.batch",
    )
    normalized_parameters = _require_sql_type_list(
        parameters,
        "parameters is required for SQL vane.cls.batch registration",
    )
    explicit_input_names = None if input_names is None else _normalize_sql_input_names(input_names)
    if explicit_input_names is not None and len(explicit_input_names) != len(normalized_parameters):
        raise _invalid_input("input_names count must match parameters count")
    normalized_input_names = _sql_class_input_names(
        fn_or_instance.user_class,
        len(normalized_parameters),
        explicit_input_names,
    )
    layout = _BatchCallLayout(tuple(normalized_input_names), ())
    return _PreparedBatchSQLRegistration(
        alias=_resolve_alias(alias, fn_or_instance.sql_name),
        udf=fn_or_instance.actor_class(layout),
        input_names=normalized_input_names,
        schema={fn_or_instance.sql_name: fn_or_instance.return_dtype},
        parameters=normalized_parameters,
        batch_size=fn_or_instance.batch_size,
        gpus=_normalize_gpus(fn_or_instance.gpus),
        actor_number=_validate_positive_actor_number(fn_or_instance.actor_number),
        row_preserving=True,
        replace=replace,
    )


def _preflight_vane_batch_function(
    fn_or_function: VaneBatchFunction,
    *,
    alias: str | None,
    parameters: Any,
    return_dtype: Any,
    input_names: Any,
    schema: Any,
    batch_size: Any,
    gpus: Any,
    actor_number: Any,
    replace: bool,
) -> _PreparedBatchSQLRegistration:
    validate_synchronous_udf_callable(fn_or_function.python_function)
    _reject_attach_override(
        "return_dtype",
        return_dtype,
        kind="vane.func.batch",
        decorator="vane.func.batch",
    )
    if schema is not None:
        raise _invalid_input(
            "schema is not valid for SQL vane.func.batch registration; use return_dtype on @vane.func.batch"
        )
    _reject_attach_override(
        "batch_size",
        batch_size,
        kind="vane.func.batch",
        decorator="vane.func.batch",
    )
    _reject_attach_override("gpus", gpus, kind="vane.func.batch", decorator="vane.func.batch")
    if actor_number is not None:
        raise _invalid_input("actor_number is not valid for SQL vane.func.batch registration; use vane.cls.batch")

    normalized_parameters = _require_sql_type_list(
        parameters,
        "parameters is required for SQL vane.func.batch registration",
    )
    explicit_input_names = None if input_names is None else _normalize_sql_input_names(input_names)
    if explicit_input_names is not None and len(explicit_input_names) != len(normalized_parameters):
        raise _invalid_input("input_names count must match parameters count")
    normalized_input_names = _sql_batch_function_input_names(
        fn_or_function.signature,
        len(normalized_parameters),
        explicit_input_names,
    )
    layout = _BatchCallLayout(tuple(normalized_input_names), ())
    return _PreparedBatchSQLRegistration(
        alias=_resolve_alias(alias, fn_or_function.sql_name),
        udf=fn_or_function.adapter(layout),
        input_names=normalized_input_names,
        schema={fn_or_function.sql_name: fn_or_function.return_dtype},
        parameters=normalized_parameters,
        batch_size=fn_or_function.batch_size,
        gpus=_normalize_gpus(fn_or_function.gpus),
        actor_number=None,
        row_preserving=True,
        replace=replace,
    )


def _preflight_vane_function(
    fn_or_function: VaneFunction,
    *,
    alias: str | None,
    parameters: Any,
    return_dtype: Any,
    input_names: Any,
    schema: Any,
    batch_size: Any,
    gpus: Any,
    actor_number: Any,
    replace: bool,
) -> _PreparedScalarSQLRegistration:
    validate_synchronous_udf_callable(fn_or_function.python_function)
    invalid_batch_options = [
        name
        for name, value in (
            ("input_names", input_names),
            ("schema", schema),
            ("batch_size", batch_size),
            ("gpus", gpus),
            ("actor_number", actor_number),
        )
        if value is not None
    ]
    if invalid_batch_options:
        joined = ", ".join(invalid_batch_options)
        raise _invalid_input(f"{joined} is not valid for SQL vane.func registration; vane.func is scalar-only")
    resolved_return_dtype = return_dtype if return_dtype is not None else fn_or_function.return_dtype
    if resolved_return_dtype is None:
        raise _invalid_input("return_dtype is required for SQL vane.func registration")
    normalized_parameters = _require_sql_type_list(
        parameters,
        "parameters is required for SQL vane.func registration",
    )
    return _PreparedScalarSQLRegistration(
        alias=_resolve_alias(alias, fn_or_function.sql_name),
        udf=fn_or_function.python_function,
        parameters=normalized_parameters,
        return_type=_duckdb_type(resolved_return_dtype),
        replace=replace,
    )


def _preflight_raw_callable(
    fn: Any,
    *,
    alias: str | None,
    parameters: Any,
    return_dtype: Any,
    input_names: Any,
    schema: Any,
    batch_size: Any,
    gpus: Any,
    actor_number: Any,
    replace: bool,
) -> _PreparedSQLRegistration:
    if not callable(fn):
        raise TypeError("vane.attach_function requires a callable or vane.func object")
    validate_synchronous_udf_callable(fn)

    has_input_names = input_names is not None
    has_schema = schema is not None
    if has_input_names != has_schema:
        raise _invalid_input("raw batch SQL registration requires input_names and schema together")

    default_name = _callable_name(fn)
    resolved_alias = _resolve_alias(alias, default_name)
    if has_input_names:
        if return_dtype is not None:
            raise _invalid_input("return_dtype is not valid for raw batch SQL registration; use schema")
        normalized_parameters = _require_sql_type_list(
            parameters,
            "parameters is required for raw batch SQL registration",
        )
        normalized_input_names = _normalize_sql_input_names(input_names)
        if len(normalized_input_names) != len(normalized_parameters):
            raise _invalid_input("input_names count must match parameters count")
        normalized_actor_number = None
        if actor_number is not None:
            normalized_actor_number = _validate_positive_actor_number(actor_number)
            _validate_sql_actor_callable(fn)
        return _PreparedBatchSQLRegistration(
            alias=resolved_alias,
            udf=fn,
            input_names=normalized_input_names,
            schema=_normalize_schema(schema),
            parameters=normalized_parameters,
            batch_size=_normalize_batch_size(batch_size),
            gpus=_normalize_gpus(gpus),
            actor_number=normalized_actor_number,
            row_preserving=True,
            replace=replace,
        )

    invalid_batch_options = [
        name
        for name, value in (
            ("batch_size", batch_size),
            ("gpus", gpus),
            ("actor_number", actor_number),
        )
        if value is not None
    ]
    if invalid_batch_options:
        joined = ", ".join(invalid_batch_options)
        raise _invalid_input(f"{joined} is not valid for raw scalar SQL registration")
    if return_dtype is None:
        raise _invalid_input("return_dtype is required for SQL raw scalar registration")
    normalized_parameters = _require_sql_type_list(
        parameters,
        "parameters is required for SQL raw scalar registration",
    )
    return _PreparedScalarSQLRegistration(
        alias=resolved_alias,
        udf=fn,
        parameters=normalized_parameters,
        return_type=_duckdb_type(return_dtype),
        replace=replace,
    )


def _preflight_attach_function(
    fn_or_function: Any,
    alias: str | None,
    *,
    replace: bool,
    parameters: Any,
    return_dtype: Any,
    input_names: Any,
    schema: Any,
    batch_size: Any,
    gpus: Any,
    actor_number: Any,
) -> _PreparedSQLRegistration:
    if isinstance(fn_or_function, VaneClass):
        class_name = fn_or_function.user_class.__name__
        raise _invalid_input(f"SQL registration for vane.cls requires an instantiated class; use {class_name}()")
    if isinstance(fn_or_function, VaneClassBatch):
        class_name = fn_or_function.user_class.__name__
        raise _invalid_input(f"SQL registration for vane.cls.batch requires an instantiated class; use {class_name}()")
    if isinstance(fn_or_function, VaneClassInstance):
        return _preflight_vane_class_instance(
            fn_or_function,
            alias=alias,
            parameters=parameters,
            input_names=input_names,
            return_dtype=return_dtype,
            schema=schema,
            batch_size=batch_size,
            gpus=gpus,
            actor_number=actor_number,
            replace=replace,
        )
    if isinstance(fn_or_function, VaneClassBatchInstance):
        return _preflight_vane_class_batch_instance(
            fn_or_function,
            alias=alias,
            parameters=parameters,
            input_names=input_names,
            return_dtype=return_dtype,
            schema=schema,
            batch_size=batch_size,
            gpus=gpus,
            actor_number=actor_number,
            replace=replace,
        )
    if isinstance(fn_or_function, VaneBatchFunction):
        return _preflight_vane_batch_function(
            fn_or_function,
            alias=alias,
            parameters=parameters,
            return_dtype=return_dtype,
            input_names=input_names,
            schema=schema,
            batch_size=batch_size,
            gpus=gpus,
            actor_number=actor_number,
            replace=replace,
        )
    if isinstance(fn_or_function, VaneFunction):
        return _preflight_vane_function(
            fn_or_function,
            alias=alias,
            parameters=parameters,
            return_dtype=return_dtype,
            input_names=input_names,
            schema=schema,
            batch_size=batch_size,
            gpus=gpus,
            actor_number=actor_number,
            replace=replace,
        )
    return _preflight_raw_callable(
        fn_or_function,
        alias=alias,
        parameters=parameters,
        return_dtype=return_dtype,
        input_names=input_names,
        schema=schema,
        batch_size=batch_size,
        gpus=gpus,
        actor_number=actor_number,
        replace=replace,
    )


def attach_function(
    fn_or_function: Any,
    alias: str | None = None,
    *,
    connection: Any | None = None,
    replace: bool = False,
    parameters: Any = None,
    return_dtype: Any | None = None,
    input_names: Any = None,
    schema: Mapping[str, Any] | None = None,
    batch_size: int | None = None,
    gpus: float | None = None,
    actor_number: int | None = None,
) -> None:
    """Attach an expression UDF callable to a DuckDB connection.

    Decorated UDFs require ``parameters`` and infer SQL argument names from
    their Python signatures. Raw scalar callables additionally require an
    effective ``return_dtype``. Raw batch callables retain the low-level
    ``pa.Table`` protocol and require ``input_names`` plus ``schema``.

    Decorator return types, GPU/actor settings, and decorator-configured batch
    sizes cannot be overridden at attach time. A row-oriented ``vane.cls``
    instance may set ``batch_size`` during attachment. Row-oriented class
    expressions propagate SQL NULL without invoking user code and accept only
    timezone-naive ``TIMESTAMP`` output; eager calls retain ordinary Python
    semantics.

    ``replace=True`` atomically replaces an existing Vane alias owned by the
    same connection. Builtins, aliases owned by another connection, and
    different SQL signatures are never overwritten. The active transaction
    may be cancelled by DuckDB while registering a function.
    """
    prepared = _preflight_attach_function(
        fn_or_function,
        alias,
        replace=replace,
        parameters=parameters,
        return_dtype=return_dtype,
        input_names=input_names,
        schema=schema,
        batch_size=batch_size,
        gpus=gpus,
        actor_number=actor_number,
    )
    conn = connection if connection is not None else vane.default_connection()
    prepared.apply(conn)


def detach_function(alias: str, *, connection: Any | None = None) -> None:
    conn = connection if connection is not None else vane.default_connection()
    conn.remove_function(alias)


__all__ = [
    "VaneClass",
    "VaneClassBatch",
    "VaneClassBatchInstance",
    "VaneClassInstance",
    "VaneBatchFunction",
    "VaneFunction",
    "_build_actor_map_batches_expression",
    "_build_batch_actor_class",
    "_build_row_actor_class",
    "attach_function",
    "cls",
    "detach_function",
    "func",
]
