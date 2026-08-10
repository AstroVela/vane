# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Strict batch-format adapters for relation-level ``map_batches`` UDFs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]
from numpy.typing import NDArray

from vane.execution._udf_validation import ensure_synchronous_udf_result
from vane.execution.udf_output_schema import arrow_type_from_output_schema_entry, normalize_output_schema_entries

VALID_BATCH_FORMATS = frozenset({"pyarrow", "numpy", "pandas", "cudf"})


@dataclass(frozen=True)
class _OutputColumnSchema:
    name: str
    tensor_type: pa.DataType | None


_OutputSchema = tuple[_OutputColumnSchema, ...]


def normalize_batch_format(value: Any) -> str:
    if not isinstance(value, str) or value not in VALID_BATCH_FORMATS:
        choices = ", ".join(sorted(VALID_BATCH_FORMATS))
        raise ValueError(f"batch_format must be one of: {choices}")
    return value


def format_udf_input(table: pa.Table, batch_format: str) -> Any:
    """Convert an internal Arrow table to the exact format requested by a UDF."""
    batch_format = normalize_batch_format(batch_format)
    if batch_format == "pyarrow":
        return table

    _require_unique_column_names(table.schema.names, batch_format)
    if batch_format == "numpy":
        return {name: _arrow_column_to_numpy(table.column(index)) for index, name in enumerate(table.schema.names)}
    if batch_format == "pandas":
        return _arrow_table_to_pandas(table)
    return _arrow_table_to_cudf(table)


def iter_udf_output_tables(
    result: Any,
    *,
    batch_format: str,
    output_schema: Any = None,
    resolved_output_schema: _OutputSchema | None = None,
) -> Iterable[pa.Table]:
    """Normalize UDF output batches back to Arrow without cross-format fallback."""
    batch_format = normalize_batch_format(batch_format)
    if output_schema is not None and resolved_output_schema is not None:
        raise ValueError("provide output_schema or resolved_output_schema, not both")
    if resolved_output_schema is None:
        resolved_output_schema = resolve_udf_output_schema(batch_format, output_schema)
    yield from _iter_udf_output_tables(result, batch_format=batch_format, output_schema=resolved_output_schema)


def resolve_udf_output_schema(batch_format: str, output_schema: Any) -> _OutputSchema | None:
    """Resolve the declared output schema once for a worker-side format adapter."""
    batch_format = normalize_batch_format(batch_format)
    if batch_format == "pyarrow":
        return None
    columns: list[_OutputColumnSchema] = []
    for name, entry in normalize_output_schema_entries(output_schema):
        kind = str(entry.get("kind") or "").strip().lower()
        tensor_type = arrow_type_from_output_schema_entry(entry) if kind == "tensor" else None
        columns.append(_OutputColumnSchema(name=name, tensor_type=tensor_type))
    return tuple(columns)


def _iter_udf_output_tables(
    result: Any,
    *,
    batch_format: str,
    output_schema: _OutputSchema | None,
) -> Iterable[pa.Table]:
    result = ensure_synchronous_udf_result(result)
    if result is None:
        return

    if batch_format == "pyarrow":
        if isinstance(result, pa.RecordBatchReader):
            raise TypeError("pyarrow map_batches output must be materialized; RecordBatchReader is not supported")
        if isinstance(result, pa.Table):
            yield result
            return
        if isinstance(result, pa.RecordBatch):
            yield pa.Table.from_batches([result])
            return
    elif batch_format == "numpy":
        if type(result) is dict:
            assert output_schema is not None
            yield _numpy_batch_to_arrow(result, output_schema)
            return
    elif batch_format == "pandas":
        pandas = _import_pandas()
        if isinstance(result, pandas.DataFrame):
            assert output_schema is not None
            yield _pandas_batch_to_arrow(result, output_schema)
            return
    else:
        cudf = _import_cudf()
        if isinstance(result, cudf.DataFrame):
            assert output_schema is not None
            yield _cudf_batch_to_arrow(result, output_schema)
            return

    if _is_batch_like(result):
        raise TypeError(
            f"map_batches(batch_format={batch_format!r}) UDF must return {_output_type_name(batch_format)}, "
            f"got {type(result)}"
        )
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes, bytearray)):
        for item in result:
            if item is None:
                continue
            yield from _iter_udf_output_tables(item, batch_format=batch_format, output_schema=output_schema)
        return

    raise TypeError(
        f"map_batches(batch_format={batch_format!r}) UDF must return {_output_type_name(batch_format)} "
        f"or an iterator yielding that type, got {type(result)}"
    )


def _is_batch_like(value: Any) -> bool:
    if isinstance(value, (pa.Table, pa.RecordBatch, pa.RecordBatchReader, np.ndarray, Mapping)):
        return True
    module = type(value).__module__.partition(".")[0]
    return module in {"pandas", "cudf"}


def _output_type_name(batch_format: str) -> str:
    return {
        "pyarrow": "pyarrow.Table or pyarrow.RecordBatch",
        "numpy": "dict[str, numpy.ndarray]",
        "pandas": "pandas.DataFrame",
        "cudf": "cudf.DataFrame",
    }[batch_format]


def _require_unique_column_names(names: list[str], batch_format: str) -> None:
    seen = set()
    duplicates = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        rendered = ", ".join(repr(name) for name in sorted(duplicates))
        raise ValueError(f"batch_format={batch_format!r} requires unique column names; duplicates: {rendered}")


def _is_fixed_shape_tensor(data_type: pa.DataType) -> bool:
    return getattr(data_type, "extension_name", None) == "arrow.fixed_shape_tensor"


def _arrow_column_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    array = column.combine_chunks()
    if not _is_fixed_shape_tensor(array.type):
        if array.null_count == 0:
            return np.array(array.to_numpy(zero_copy_only=False), copy=True)
        valid = array.is_valid().to_numpy(zero_copy_only=False)
        valid_array = array.filter(pa.array(valid))
        valid_values = valid_array.to_numpy(zero_copy_only=False)
        nullable_values: NDArray[Any] = np.zeros(len(array), dtype=valid_values.dtype)
        nullable_values[valid] = valid_values
        return np.ma.MaskedArray(  # type: ignore[no-untyped-call]
            nullable_values,
            mask=np.logical_not(valid),
            copy=False,
        )

    dense = array.to_numpy_ndarray()
    if array.null_count == 0:
        return np.array(dense, copy=True)

    valid = array.is_valid().to_numpy(zero_copy_only=False)
    nullable: NDArray[np.object_] = np.empty(len(array), dtype=object)
    for index, is_valid in enumerate(valid):
        nullable[index] = np.array(dense[index], copy=True) if is_valid else None
    return nullable


def _arrow_table_to_pandas(table: pa.Table) -> Any:
    pandas = _import_pandas()
    tensor_indices = [index for index, field in enumerate(table.schema) if _is_fixed_shape_tensor(field.type)]
    if not tensor_indices:
        return _arrow_regular_table_to_pandas(table, pandas)

    tensor_index_set = set(tensor_indices)
    regular_columns = [index for index in range(table.num_columns) if index not in tensor_index_set]
    if regular_columns:
        frame = _arrow_regular_table_to_pandas(table.select(regular_columns), pandas)
    else:
        frame = pandas.DataFrame(index=pandas.RangeIndex(table.num_rows))
    for index, field in enumerate(table.schema):
        if not _is_fixed_shape_tensor(field.type):
            continue
        tensor_values = _tensor_column_to_object_array(table.column(index))
        frame.insert(index, field.name, pandas.Series(tensor_values, index=frame.index, dtype=object))
    return frame


def _arrow_regular_table_to_pandas(table: pa.Table, pandas: Any) -> Any:
    def types_mapper(data_type: pa.DataType) -> Any:
        if isinstance(data_type, pa.BaseExtensionType) or pa.types.is_dictionary(data_type):
            return None
        return pandas.ArrowDtype(data_type)

    return table.to_pandas(types_mapper=types_mapper)


def _tensor_column_to_object_array(column: pa.ChunkedArray) -> np.ndarray:
    array = column.combine_chunks()
    dense = array.to_numpy_ndarray()
    valid = array.is_valid().to_numpy(zero_copy_only=False) if array.null_count else None
    values: NDArray[np.object_] = np.empty(len(array), dtype=object)
    for index in range(len(array)):
        values[index] = None if valid is not None and not valid[index] else np.array(dense[index], copy=True)
    return values


def _arrow_table_to_cudf(table: pa.Table) -> Any:
    cudf = _import_cudf()
    return cudf.DataFrame.from_arrow(table)


def _numpy_batch_to_arrow(batch: dict[Any, Any], schema: _OutputSchema) -> pa.Table:
    _validate_output_names(list(batch.keys()), schema)
    arrays: list[pa.Array] = []
    row_count: int | None = None
    for column_schema in schema:
        value = batch[column_schema.name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"numpy batch column {column_schema.name!r} must be numpy.ndarray, got {type(value)}")
        if value.ndim == 0:
            raise ValueError(f"numpy batch column {column_schema.name!r} must include a row dimension")
        if row_count is None:
            row_count = len(value)
        elif len(value) != row_count:
            raise ValueError(
                f"numpy batch columns must have the same row count; {column_schema.name!r} has {len(value)}, "
                f"expected {row_count}"
            )
        arrays.append(_numpy_column_to_arrow(value, column_schema))
    return pa.Table.from_arrays(arrays, names=[column.name for column in schema])


def _numpy_column_to_arrow(value: np.ndarray, column_schema: _OutputColumnSchema) -> pa.Array:
    if column_schema.tensor_type is not None:
        return _tensor_values_to_arrow(value, column_schema)
    if value.ndim != 1:
        return pa.array(value.tolist())
    return pa.array(value)


def _pandas_batch_to_arrow(frame: Any, schema: _OutputSchema) -> pa.Table:
    _validate_output_names(list(frame.columns), schema)
    arrays = []
    for column_schema in schema:
        series = frame[column_schema.name]
        if column_schema.tensor_type is not None:
            arrays.append(_tensor_values_to_arrow(series.tolist(), column_schema))
        else:
            arrays.append(pa.array(series, from_pandas=True))
    return pa.Table.from_arrays(arrays, names=[column.name for column in schema])


def _cudf_batch_to_arrow(frame: Any, schema: _OutputSchema) -> pa.Table:
    table = frame.to_arrow(preserve_index=False)
    if not isinstance(table, pa.Table):
        raise TypeError(f"cudf.DataFrame.to_arrow() must return pyarrow.Table, got {type(table)}")
    _validate_output_names(table.schema.names, schema)
    return table.select([column.name for column in schema])


def _validate_output_names(actual_names: list[Any], schema: _OutputSchema) -> None:
    if not all(isinstance(name, str) for name in actual_names):
        raise TypeError("map_batches output column names must be strings")
    names = list(actual_names)
    if len(names) != len(set(names)):
        raise ValueError("map_batches output column names must be unique")
    expected = [column.name for column in schema]
    missing = [name for name in expected if name not in names]
    extra = [name for name in names if name not in expected]
    if missing or extra:
        raise ValueError(f"map_batches output columns do not match schema; missing={missing}, extra={extra}")


def _tensor_values_to_arrow(values: Any, column_schema: _OutputColumnSchema) -> pa.Array:
    tensor_type = column_schema.tensor_type
    assert tensor_type is not None
    shape = tuple(int(dim) for dim in tensor_type.shape)
    if np.ma.isMaskedArray(values):  # type: ignore[no-untyped-call]
        return _masked_tensor_values_to_arrow(values, column_schema, shape)
    if isinstance(values, np.ndarray) and values.ndim == len(shape) + 1 and values.dtype != object:
        return _dense_tensor_values_to_arrow(values, column_schema, shape)

    rows = list(values)
    tensor_rows: list[np.ndarray | None] = []
    dense_rows: list[np.ndarray] = []
    has_null = False
    for index, value in enumerate(rows):
        if np.ma.isMaskedArray(value):  # type: ignore[no-untyped-call]
            row_mask = np.ma.getmaskarray(value)  # type: ignore[no-untyped-call]
            if row_mask.all():
                tensor_rows.append(None)
                has_null = True
                continue
            if row_mask.any():
                raise ValueError(
                    f"tensor output column {column_schema.name!r} row {index} has a partial mask; "
                    "only whole tensor rows may be null"
                )
            value = value.data
        if _is_null_tensor_value(value):
            tensor_rows.append(None)
            has_null = True
            continue
        tensor = np.asarray(value)
        if tensor.shape != shape:
            raise ValueError(
                f"tensor output column {column_schema.name!r} row {index} has shape {tensor.shape}, expected {shape}"
            )
        tensor_rows.append(tensor)
        dense_rows.append(tensor)
    if rows and not has_null:
        return _dense_tensor_values_to_arrow(np.stack(dense_rows), column_schema, shape)
    storage_rows = [None if tensor is None else tensor.reshape(-1).tolist() for tensor in tensor_rows]
    storage = pa.array(storage_rows, type=tensor_type.storage_type, safe=True)
    return pa.ExtensionArray.from_storage(tensor_type, storage)


def _masked_tensor_values_to_arrow(
    values: np.ma.MaskedArray,
    column_schema: _OutputColumnSchema,
    shape: tuple[int, ...],
) -> pa.Array:
    if len(values) == 0:
        return _dense_tensor_values_to_arrow(np.asarray(values.data), column_schema, shape)
    if values.ndim == 1 and values.dtype == object:
        row_mask = np.ma.getmaskarray(values)  # type: ignore[no-untyped-call]
        rows = [None if row_mask[index] else values.data[index] for index in range(len(values))]
        return _tensor_values_to_arrow(rows, column_schema)

    if values.ndim != len(shape) + 1 or tuple(values.shape[1:]) != shape:
        raise ValueError(f"tensor output column {column_schema.name!r} has shape {values.shape[1:]}, expected {shape}")

    element_mask = np.ma.getmaskarray(values).reshape(  # type: ignore[no-untyped-call]
        len(values),
        -1,
    )
    masked_rows = element_mask.all(axis=1)
    partially_masked_rows = element_mask.any(axis=1) & ~masked_rows
    if partially_masked_rows.any():
        first_row = int(np.flatnonzero(partially_masked_rows)[0])
        raise ValueError(
            f"tensor output column {column_schema.name!r} row {first_row} has a partial mask; "
            "only whole tensor rows may be null"
        )
    if not masked_rows.any():
        return _dense_tensor_values_to_arrow(np.asarray(values.data), column_schema, shape)

    rows = [None if masked_rows[index] else np.asarray(values.data[index]) for index in range(len(values))]
    return _tensor_values_to_arrow(rows, column_schema)


def _dense_tensor_values_to_arrow(
    values: np.ndarray,
    column_schema: _OutputColumnSchema,
    shape: tuple[int, ...],
) -> pa.Array:
    tensor_type = column_schema.tensor_type
    assert tensor_type is not None
    if tuple(values.shape[1:]) != shape:
        raise ValueError(f"tensor output column {column_schema.name!r} has shape {values.shape[1:]}, expected {shape}")
    if len(values) == 0:
        storage = pa.array([], type=tensor_type.storage_type)
        return pa.ExtensionArray.from_storage(tensor_type, storage)
    contiguous = np.ascontiguousarray(values)
    try:
        source_value_type = pa.from_numpy_dtype(contiguous.dtype)
    except pa.ArrowNotImplementedError:
        source_value_type = None
    if contiguous.dtype.isnative and source_value_type == tensor_type.value_type:
        return pa.FixedShapeTensorArray.from_numpy_ndarray(contiguous)

    flattened = pa.array(contiguous.reshape(-1).tolist(), type=tensor_type.value_type, safe=True)
    storage = pa.FixedSizeListArray.from_arrays(flattened, tensor_type.storage_type.list_size)
    return pa.ExtensionArray.from_storage(tensor_type, storage)


def _is_null_tensor_value(value: Any) -> bool:
    if value is None:
        return True
    if value is np.ma.masked:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return type(value).__name__ == "NAType" and type(value).__module__.partition(".")[0] == "pandas"


def _import_pandas() -> Any:
    try:
        import pandas
    except ImportError as exc:
        raise ImportError("batch_format='pandas' requires pandas to be installed") from exc
    return pandas


def _import_cudf() -> Any:
    try:
        import cudf  # type: ignore[import-not-found, import-untyped, unused-ignore]
    except ImportError as exc:
        raise ImportError("batch_format='cudf' requires cuDF and a CUDA-capable environment") from exc
    return cudf


__all__ = [
    "VALID_BATCH_FORMATS",
    "format_udf_input",
    "iter_udf_output_tables",
    "normalize_batch_format",
    "resolve_udf_output_schema",
]
