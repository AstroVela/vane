# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Expression-level Python UDF decorators for Vane."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from numbers import Real
from typing import Any

import _vane_duckdb

import vane
from vane._expressions import as_expression, is_expression
from vane.config import current_config


def _invalid_input(message: str) -> vane.InvalidInputException:
    return vane.InvalidInputException(message)


def _unsupported_dtype(dtype: Any) -> vane.InvalidInputException:
    return _invalid_input(f"dtype {str(dtype)!r} is not supported for expression UDF output")


def _arrow_to_duckdb_type(dtype: Any, *, original: Any) -> Any:
    import pyarrow as pa

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


def _class_name(cls: type) -> str:
    return _qualified_name(cls, kind="UDF class")


def _resolve_udf_name(name: Any, default_name: str) -> str:
    if name is None:
        return default_name
    if not isinstance(name, str) or not name:
        raise _invalid_input("name must be a non-empty string")
    return name


def _has_expression(values: Any) -> bool:
    return any(is_expression(value) for value in values)


def _validate_actor_number(actor_number: Any) -> int:
    if isinstance(actor_number, bool):
        raise _invalid_input(
            "actor_number must be exactly 1 for stateful vane.cls UDFs; bool is not accepted and "
            "multi-actor state semantics are not defined"
        )
    if type(actor_number) is not int or actor_number != 1:
        raise _invalid_input(
            "actor_number must be exactly 1 for stateful vane.cls UDFs; multi-actor state semantics are not defined"
        )
    return int(actor_number)


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
    return _vane_duckdb._VaneUDFMapExpression(
        bound_fn,
        name,
        _duckdb_type(return_dtype),
        _runner_to_task_backend(),
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
            self._instance = user_class(*init_args, **captured_init_kwargs)

        def __call__(self, table: Any) -> Any:
            columns = [table.column(name).to_pylist() for name in captured_input_names]
            out: list[Any] = []
            for row in zip(*columns, strict=True):
                if any(value is None for value in row):
                    out.append(None)
                    continue
                out.append(self._instance(*row, **captured_call_kwargs))
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
) -> type:
    captured_init_kwargs = dict(init_kwargs)

    class _VaneBatchActorAdapter:
        def __init__(self) -> None:
            self._instance = user_class(*init_args, **captured_init_kwargs)

        def __call__(self, table: Any) -> Any:
            return self._instance(table)

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
    """Expression UDF decorator wrapper.

    When used as an instance-method descriptor, the serialized callable
    captures that instance's current snapshot. Persistent cross-batch state is
    intentionally provided only by :func:`vane.cls`.
    """

    def __init__(self, fn: Callable[..., Any], *, return_dtype: Any | None = None, name: str | None = None):
        if not callable(fn):
            raise TypeError("vane.func requires a callable")
        self._fn = fn
        self._return_dtype = return_dtype
        self._name = _resolve_udf_name(name, _callable_name(fn))
        functools.update_wrapper(self, fn)

    @property
    def return_dtype(self) -> Any | None:
        return self._return_dtype

    @property
    def python_function(self) -> Callable[..., Any]:
        return self._fn

    @property
    def sql_name(self) -> str:
        return self._name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _call_or_build_expression(
            has_expression=_has_expression(args) or _has_expression(kwargs.values()),
            call_immediately=lambda: self._fn(*args, **kwargs),
            build_expression=lambda: _build_map_expression(self._fn, self._name, self._return_dtype, args, kwargs),
        )

    def __get__(self, instance: Any, owner: Any | None = None) -> Any:
        if instance is None:
            return self
        bound_fn = self._fn.__get__(instance, owner)
        return VaneFunction(bound_fn, return_dtype=self._return_dtype, name=self._name)


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
        if return_dtype is None:
            raise _invalid_input("return_dtype is required for vane.cls")
        normalized_return_dtype, return_arrow_dtype = _canonicalize_dtype(return_dtype)
        self._class = class_
        self._actor_number = _validate_actor_number(actor_number)
        self._return_dtype = normalized_return_dtype
        self._return_arrow_dtype = return_arrow_dtype
        self._name = _resolve_udf_name(name, _class_name(class_))
        self._gpus = 0 if gpus is None else gpus
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
            self._eager_instance = self.user_class(*self._init_args, **self._init_kwargs)
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
            call_immediately=lambda: self._instance()(*args, **kwargs),
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
            stateful=True,
        )


class VaneClassBatch:
    def __init__(
        self,
        class_: type,
        *,
        actor_number: int | None,
        schema: Mapping[str, Any] | None,
        name: str | None = None,
        batch_size: int | None = None,
        row_preserving: bool = False,
        gpus: float | None = 0,
    ) -> None:
        if not inspect.isclass(class_):
            raise TypeError("vane.cls.batch requires a class")
        self._schema = _normalize_schema(schema)
        self._class = class_
        self._actor_number = _validate_actor_number(actor_number)
        self._name = _resolve_udf_name(name, _class_name(class_))
        self._batch_size = batch_size
        self._row_preserving = bool(row_preserving)
        self._gpus = 0 if gpus is None else gpus
        functools.update_wrapper(self, class_, updated=())

    @property
    def user_class(self) -> type:
        return self._class

    @property
    def actor_number(self) -> int:
        return self._actor_number

    @property
    def schema(self) -> dict[str, Any]:
        return dict(self._schema)

    @property
    def sql_name(self) -> str:
        return self._name

    @property
    def batch_size(self) -> int | None:
        return self._batch_size

    @property
    def row_preserving(self) -> bool:
        return self._row_preserving

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
    def schema(self) -> dict[str, Any]:
        return self._decorator.schema

    @property
    def batch_size(self) -> int | None:
        return self._decorator.batch_size

    @property
    def row_preserving(self) -> bool:
        return self._decorator.row_preserving

    @property
    def gpus(self) -> float | int | None:
        return self._decorator.gpus

    @property
    def user_class(self) -> type:
        return self._decorator.user_class

    def _instance(self) -> Any:
        if self._eager_instance is None:
            self._eager_instance = self.user_class(*self._init_args, **self._init_kwargs)
        return self._eager_instance

    def actor_class(self) -> type:
        return _build_batch_actor_class(self.user_class, self._init_args, self._init_kwargs)

    def __call__(self, **inputs: Any) -> Any:
        normalized_inputs = _normalize_input_mapping(inputs)
        return _call_or_build_expression(
            has_expression=_has_expression(normalized_inputs.values()),
            call_immediately=lambda: _call_batch_immediately(self._instance(), normalized_inputs),
            build_expression=lambda: _build_actor_map_batches_expression(
                self.actor_class(),
                name=self.sql_name,
                inputs=normalized_inputs,
                schema=self.schema,
                batch_size=self.batch_size,
                row_preserving=self.row_preserving,
                actor_number=self.actor_number,
                gpus=self.gpus,
                stateful=True,
            ),
        )


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


def _call_batch_immediately(fn: Callable[[Any], Any], inputs: Mapping[str, Any]) -> Any:
    import pyarrow as pa

    return fn(pa.table(_normalize_input_mapping(inputs)))


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
    stateful: bool = False,
) -> vane.Expression:
    input_names, exprs = _normalize_inputs(inputs)
    normalized_schema = _normalize_schema(schema)
    resolved_backend = _runner_to_task_backend() if execution_backend is None else execution_backend
    return _vane_duckdb._VaneUDFMapBatchesExpression(
        fn,
        _resolve_udf_name(name, _callable_name(fn)),
        normalized_schema,
        resolved_backend,
        input_names,
        batch_size,
        bool(row_preserving),
        gpus,
        actor_number,
        stateful,
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
    stateful: bool = False,
) -> vane.Expression:
    normalized_actor_number = (
        _validate_actor_number(actor_number) if stateful else _validate_positive_actor_number(actor_number)
    )
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
        stateful=stateful,
    )


def _map_batches(
    fn: Callable[[Any], Any],
    *,
    inputs: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
    name: str | None = None,
    batch_size: int | None = None,
    row_preserving: bool = False,
    gpus: float | None = None,
) -> Any:
    if not callable(fn):
        raise TypeError("vane.func.batch requires a callable")
    normalized_inputs = _normalize_input_mapping(inputs)
    return _call_or_build_expression(
        has_expression=_has_expression(normalized_inputs.values()),
        call_immediately=lambda: _call_batch_immediately(fn, normalized_inputs),
        build_expression=lambda: _build_map_batches_expression(
            fn,
            name=name,
            inputs=normalized_inputs,
            schema=schema,
            batch_size=batch_size,
            row_preserving=row_preserving,
            gpus=gpus,
        ),
    )


def func(
    fn: Callable[..., Any] | None = None,
    *,
    return_dtype: Any | None = None,
    name: str | None = None,
) -> VaneFunction | Callable[[Callable[..., Any]], VaneFunction]:
    if fn is None:
        return lambda actual_fn: VaneFunction(actual_fn, return_dtype=return_dtype, name=name)
    return VaneFunction(fn, return_dtype=return_dtype, name=name)


func.batch = _map_batches  # type: ignore[attr-defined]


def _cls(
    class_: type | None = None,
    *,
    actor_number: int | None = None,
    return_dtype: Any | None = None,
    name: str | None = None,
    gpus: float | None = 0,
) -> VaneClass | Callable[[type], VaneClass]:
    if class_ is None:
        return lambda actual_class: VaneClass(
            actual_class,
            actor_number=actor_number,
            return_dtype=return_dtype,
            name=name,
            gpus=gpus,
        )
    return VaneClass(class_, actor_number=actor_number, return_dtype=return_dtype, name=name, gpus=gpus)


def _cls_batch(
    class_: type | None = None,
    *,
    actor_number: int | None = None,
    schema: Mapping[str, Any] | None = None,
    name: str | None = None,
    batch_size: int | None = None,
    row_preserving: bool = False,
    gpus: float | None = 0,
) -> VaneClassBatch | Callable[[type], VaneClassBatch]:
    if class_ is None:
        return lambda actual_class: VaneClassBatch(
            actual_class,
            actor_number=actor_number,
            schema=schema,
            name=name,
            batch_size=batch_size,
            row_preserving=row_preserving,
            gpus=gpus,
        )
    return VaneClassBatch(
        class_,
        actor_number=actor_number,
        schema=schema,
        name=name,
        batch_size=batch_size,
        row_preserving=row_preserving,
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
    stateful: bool
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
            stateful=self.stateful,
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
        actor_number=_validate_actor_number(fn_or_instance.actor_number),
        stateful=True,
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
    if not fn_or_instance.row_preserving:
        raise _invalid_input(
            "row_preserving=False is supported by the expression API, but SQL attach v1 requires "
            "row-preserving batch UDFs"
        )
    normalized_parameters = _require_sql_type_list(
        parameters,
        "parameters is required for SQL vane.cls.batch registration",
    )
    if input_names is None:
        raise _invalid_input("input_names is required for SQL vane.cls.batch registration")
    normalized_input_names = _normalize_sql_input_names(input_names)
    if len(normalized_input_names) != len(normalized_parameters):
        raise _invalid_input("input_names count must match parameters count")
    resolved_schema = _normalize_schema(schema) if schema is not None else fn_or_instance.schema
    resolved_batch_size = batch_size if batch_size is not None else fn_or_instance.batch_size
    return _PreparedBatchSQLRegistration(
        alias=_resolve_alias(alias, fn_or_instance.sql_name),
        udf=fn_or_instance.actor_class(),
        input_names=normalized_input_names,
        schema=resolved_schema,
        parameters=normalized_parameters,
        batch_size=_normalize_batch_size(resolved_batch_size),
        gpus=_normalize_gpus(fn_or_instance.gpus),
        actor_number=_validate_actor_number(fn_or_instance.actor_number),
        stateful=True,
        row_preserving=fn_or_instance.row_preserving,
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
            stateful=False,
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

    ``VaneFunction`` and raw scalar callables require ``parameters`` and an
    effective ``return_dtype``. Raw batch callables require ``parameters``,
    ``input_names``, and ``schema``; they additionally accept ``batch_size``
    and ``gpus``, while ``actor_number`` requires a zero-argument callable
    class. Instantiated ``vane.cls`` requires ``parameters``, may infer
    ``input_names``, and accepts a ``batch_size`` override. Instantiated
    ``vane.cls.batch`` requires ``parameters`` and ``input_names`` and accepts
    ``schema`` and ``batch_size`` overrides. Uninstantiated class decorator
    wrappers are rejected.

    The row-class return type and all class GPU/actor settings belong to the
    decorator/instance and cannot be overridden at attach time. Batch-class
    ``schema`` and ``batch_size`` use the explicit override rules above. SQL
    v1 rejects non-row-preserving class-batch registrations. Row-oriented
    class expressions propagate SQL NULL without invoking user code and accept
    only timezone-naive ``TIMESTAMP`` output; eager calls retain ordinary
    Python semantics.

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
    "VaneFunction",
    "_build_actor_map_batches_expression",
    "_build_batch_actor_class",
    "_build_row_actor_class",
    "attach_function",
    "cls",
    "detach_function",
    "func",
]
