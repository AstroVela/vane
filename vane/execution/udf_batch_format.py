# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Strict batch-format adapters for relation-level ``map_batches`` UDFs."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]
from numpy.typing import NDArray

from vane._image import _MODE_CHANNELS, _MODE_DTYPES, _image_arrow_scalar_to_numpy, _ImageArrowType
from vane.execution._udf_validation import ensure_synchronous_udf_result
from vane.execution.udf_file_contract import (
    _map_array_from_offsets,
    _native_outputs_to_arrow_array,
    validate_file_arrow_array,
)
from vane.execution.udf_output_schema import (
    _arrow_type_from_output_schema_entry,
    _canonicalize_struct_field_names,
    _fixed_shape_tensor_array,
    normalize_output_schema_entries,
)

VALID_BATCH_FORMATS = frozenset({"pyarrow", "numpy", "pandas", "cudf"})


@dataclass(frozen=True)
class _OutputColumnSchema:
    name: str
    tensor_type: pa.DataType | None
    image_dtype: Any = None
    map_container_type: pa.DataType | None = None


_OutputSchema = tuple[_OutputColumnSchema, ...]


def normalize_batch_format(value: Any) -> str:
    if not isinstance(value, str) or value not in VALID_BATCH_FORMATS:
        choices = ", ".join(sorted(VALID_BATCH_FORMATS))
        raise ValueError(f"batch_format must be one of: {choices}")
    return value


def format_udf_input(table: pa.Table, batch_format: str, *, zero_copy_batch: bool = True) -> Any:
    """Convert an internal Arrow table to the exact format requested by a UDF."""
    batch_format = normalize_batch_format(batch_format)
    if type(zero_copy_batch) is not bool:
        raise TypeError("zero_copy_batch must be a bool")
    if not zero_copy_batch and batch_format != "numpy":
        raise ValueError("zero_copy_batch=False requires batch_format='numpy'")
    if batch_format == "pyarrow":
        return table

    _require_unique_column_names(table.schema.names, batch_format)
    if batch_format == "numpy":
        return {
            name: _numpy_input_buffers(_arrow_column_to_numpy(table.column(index)), zero_copy_batch)
            for index, name in enumerate(table.schema.names)
        }
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
    """Normalize selected-format output; Arrow keeps its existing dict support."""
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
        tensor_type = _arrow_type_from_output_schema_entry(entry) if kind == "tensor" else None
        if tensor_type is not None and not isinstance(tensor_type, pa.FixedShapeTensorType):
            raise TypeError(f"batch_format={batch_format!r} supports only fixed-shape tensor outputs")
        image_dtype = None
        map_container_type = None
        if kind == "duckdb_type":
            import vane

            dtype = vane.type(str(entry["type"]))
            if dtype.is_image():
                image_dtype = dtype
            elif _contains_map(dtype):
                map_container_type = _arrow_type_from_output_schema_entry(entry)
        columns.append(
            _OutputColumnSchema(
                name=name, tensor_type=tensor_type, image_dtype=image_dtype, map_container_type=map_container_type
            )
        )
    return tuple(columns)


def _contains_map(dtype: Any) -> bool:
    if dtype.id == "map":
        return True
    if dtype.id in ("list", "array"):
        return _contains_map(dict(dtype.children)["child"])
    if dtype.id == "struct":
        return any(_contains_map(child) for _, child in dtype.children)
    return False


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
        if isinstance(result, dict):
            yield pa.table(result)
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
        "pyarrow": "pyarrow.Table, pyarrow.RecordBatch or dict",
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


def _numpy_input_buffers(values: np.ndarray, zero_copy_batch: bool) -> np.ndarray:
    """Expose read-only buffers, or detach all buffers for in-place UDF mutation."""
    if not zero_copy_batch:
        # Object columns can contain array views (lists, nullable tensors, images).
        # A shallow ndarray.copy() would retain those shared, read-only buffers.
        return copy.deepcopy(values) if values.dtype.hasobject else values.copy()
    _freeze_numpy_buffers(values)
    return values


def _freeze_numpy_buffers(value: Any) -> None:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            for item in value.flat:
                _freeze_numpy_buffers(item)
        if isinstance(value, np.ma.MaskedArray):
            np.ma.getmaskarray(value).setflags(write=False)  # type: ignore[no-untyped-call]
        value.setflags(write=False)
    elif isinstance(value, dict):
        for item in value.values():
            _freeze_numpy_buffers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _freeze_numpy_buffers(item)


def _single_arrow_array(column: pa.ChunkedArray) -> pa.Array:
    return column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()


def _tensor_numpy_view(array: pa.FixedShapeTensorArray) -> np.ndarray:
    # Read the primitive array, including its validity and offset. Arrow's Tensor
    # view drops validity and does not support Boolean or temporal elements.
    storage = array.storage
    size = storage.type.list_size
    flat = storage.values.slice(storage.offset * size, len(storage) * size)
    dense = _arrow_values_to_numpy(flat).reshape((len(array), *array.type.shape))
    if array.type.permutation:
        dense = dense.transpose((0, *(axis + 1 for axis in array.type.permutation)))
    return dense


def _arrow_values_to_numpy(array: pa.Array) -> np.ndarray:
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type) or pa.types.is_fixed_size_list(array.type):
        rows: NDArray[np.object_] = np.empty(len(array), dtype=object)
        for index, scalar in enumerate(array):
            rows[index] = _arrow_values_to_numpy(scalar.values) if scalar.is_valid else None
        return rows
    if pa.types.is_nested(array.type) or array.null_count:
        # NumPy's floating NaN cannot represent SQL NULL without losing NaNs or
        # integer precision. Object elements retain None at every nesting level.
        objects: NDArray[np.object_] = np.empty(len(array), dtype=object)
        objects[:] = array.to_pylist()
        return objects
    return array.to_numpy(zero_copy_only=False)


def _arrow_column_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    if isinstance(column.type, _ImageArrowType):
        return _image_column_to_numpy(column)
    array = _single_arrow_array(column)
    if not _is_fixed_shape_tensor(array.type):
        if array.null_count == 0:
            return _arrow_values_to_numpy(array)
        valid = array.is_valid().to_numpy(zero_copy_only=False)
        valid_array = array.filter(pa.array(valid))
        valid_values = _arrow_values_to_numpy(valid_array)
        nullable_values: NDArray[Any] = np.zeros(len(array), dtype=valid_values.dtype)
        nullable_values[valid] = valid_values
        return np.ma.MaskedArray(  # type: ignore[no-untyped-call]
            nullable_values,
            mask=np.logical_not(valid),
            copy=False,
        )

    dense = _tensor_numpy_view(array)
    if array.null_count == 0:
        return dense

    valid = array.is_valid().to_numpy(zero_copy_only=False)
    nullable: NDArray[np.object_] = np.empty(len(array), dtype=object)
    for index, is_valid in enumerate(valid):
        nullable[index] = dense[index] if is_valid else None
    return nullable


def _image_column_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    import vane

    image_type = column.type
    dtype = vane.image_type(image_type.mode, image_type.height, image_type.width)
    for chunk in column.chunks:
        validate_file_arrow_array(chunk, dtype, boundary="map_batches IMAGE input")
    if image_type.height is not None and column.null_count == 0:
        shape = (len(column), image_type.height, image_type.width, _MODE_CHANNELS[image_type.mode])
        if len(column) == 0:
            return np.empty(shape, dtype=_MODE_DTYPES[image_type.mode])
        storage = _single_arrow_array(column).storage
        size = storage.type.list_size
        pixels = storage.values.slice(storage.offset * size, len(storage) * size)
        return pixels.to_numpy().reshape(shape)

    # Variable-size images cannot share one dense batch shape. Keep each row's
    # HWC shape and mode-specific dtype, including mixed modes in generic IMAGE.
    rows: NDArray[np.object_] = np.empty(len(column), dtype=object)
    index = 0
    for chunk in column.chunks:
        for scalar in chunk:
            rows[index] = _image_arrow_scalar_to_numpy(scalar, dtype, copy=False) if scalar.is_valid else None
            index += 1
    return rows


def _arrow_table_to_pandas(table: pa.Table) -> Any:
    pandas = _import_pandas()
    tensor_indices = [
        index
        for index, field in enumerate(table.schema)
        if _is_fixed_shape_tensor(field.type) or isinstance(field.type, _ImageArrowType)
    ]
    if not tensor_indices:
        return _arrow_regular_table_to_pandas(table, pandas)

    tensor_index_set = set(tensor_indices)
    regular_columns = [index for index in range(table.num_columns) if index not in tensor_index_set]
    if regular_columns:
        frame = _arrow_regular_table_to_pandas(table.select(regular_columns), pandas)
    else:
        frame = pandas.DataFrame(index=pandas.RangeIndex(table.num_rows))
    for index, field in enumerate(table.schema):
        if index not in tensor_index_set:
            continue
        if isinstance(field.type, _ImageArrowType):
            values = _image_column_to_numpy(table.column(index))
            tensor_values: NDArray[np.object_] = np.empty(len(values), dtype=object)
            for row, value in enumerate(values):
                tensor_values[row] = None if value is None else value.copy()
        else:
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
    array = _single_arrow_array(column)
    dense = _tensor_numpy_view(array)
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
    if column_schema.image_dtype is not None:
        return _image_values_to_arrow(value, column_schema)
    if column_schema.tensor_type is not None:
        return _tensor_values_to_arrow(value, column_schema)
    if column_schema.map_container_type is not None:
        return _map_container_values_to_arrow(value, column_schema.map_container_type)
    if value.ndim != 1:
        return pa.array(value.tolist())
    return pa.array(value)


def _map_container_values_to_arrow(values: Any, dtype: pa.DataType) -> pa.Array:
    """Preserve MAP containers while leaving ordinary leaf casts to DuckDB."""
    rows = [None if value is np.ma.masked else value for value in values]
    if all(value is None for value in rows):
        return pa.array(rows, type=dtype)
    mask = pa.array([value is None for value in rows])
    if pa.types.is_map(dtype):
        offsets = [0]
        keys, items = [], []
        for row in rows:
            if row is not None:
                if not isinstance(row, (Mapping, Sequence, np.ndarray)) or isinstance(row, (str, bytes, bytearray)):
                    raise TypeError("MAP output requires mappings or sequences of key/value pairs")
                for pair in row.items() if isinstance(row, Mapping) else row:
                    if (
                        not isinstance(pair, (Sequence, np.ndarray))
                        or isinstance(pair, (str, bytes, bytearray))
                        or len(pair) != 2
                    ):
                        raise TypeError("MAP output requires key/value pairs")
                    key, item = pair
                    keys.append(key)
                    items.append(item)
            offsets.append(len(keys))
        return _map_array_from_offsets(
            offsets,
            _map_container_values_to_arrow(keys, dtype.key_type),
            _map_container_values_to_arrow(items, dtype.item_type),
            mask=mask,
        )
    if pa.types.is_list(dtype) or pa.types.is_fixed_size_list(dtype):
        offsets = [0]
        flattened: list[Any] = []
        for row in rows:
            if pa.types.is_fixed_size_list(dtype):
                if row is not None and len(row) != dtype.list_size:
                    raise ValueError(f"MAP container output requires rows of size {dtype.list_size}")
                flattened.extend([None] * dtype.list_size if row is None else row)
            elif row is not None:
                flattened.extend(row)
            offsets.append(len(flattened))
        child = _map_container_values_to_arrow(flattened, dtype.value_type)
        if pa.types.is_fixed_size_list(dtype):
            return pa.FixedSizeListArray.from_arrays(child, dtype.list_size, mask=mask)
        return pa.ListArray.from_arrays(offsets, child, mask=mask)
    if pa.types.is_struct(dtype):
        canonical = [_canonicalize_struct_field_names(row, dtype, boundary="map_batches output") for row in rows]
        children = [
            _map_container_values_to_arrow([None if row is None else row[field.name] for row in canonical], field.type)
            for field in dtype
        ]
        return pa.StructArray.from_arrays(children, names=[field.name for field in dtype], mask=mask)
    return pa.array(rows)


def _pandas_batch_to_arrow(frame: Any, schema: _OutputSchema) -> pa.Table:
    _validate_output_names(list(frame.columns), schema)
    arrays = []
    for column_schema in schema:
        series = frame[column_schema.name]
        if column_schema.image_dtype is not None:
            arrays.append(_image_values_to_arrow(series.tolist(), column_schema))
        elif column_schema.tensor_type is not None:
            arrays.append(_tensor_values_to_arrow(series.tolist(), column_schema))
        else:
            arrays.append(pa.array(series, from_pandas=True))
    return pa.Table.from_arrays(arrays, names=[column.name for column in schema])


def _image_values_to_arrow(values: Any, column_schema: _OutputColumnSchema) -> pa.Array:
    if isinstance(values, np.ma.MaskedArray):
        raise TypeError("IMAGE output must use HWC ndarrays or None, not MaskedArray")
    return _native_outputs_to_arrow_array(
        list(values), column_schema.image_dtype, boundary=f"map_batches IMAGE output column {column_schema.name!r}"
    )


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
    storage_rows: list[pa.Array] = []
    for index, value in enumerate(rows):
        if np.ma.isMaskedArray(value):  # type: ignore[no-untyped-call]
            row_mask = np.ma.getmaskarray(value)  # type: ignore[no-untyped-call]
            if row_mask.all():
                storage_rows.append(pa.nulls(1, type=tensor_type.storage_type))
                continue
            if row_mask.any():
                raise ValueError(
                    f"tensor output column {column_schema.name!r} row {index} has a partial mask; "
                    "only whole tensor rows may be null"
                )
            value = value.data
        if _is_null_tensor_value(value):
            storage_rows.append(pa.nulls(1, type=tensor_type.storage_type))
            continue
        tensor = np.asarray(value)
        if tensor.shape != shape:
            raise ValueError(
                f"tensor output column {column_schema.name!r} row {index} has shape {tensor.shape}, expected {shape}"
            )
        storage_rows.append(_dense_tensor_values_to_arrow(tensor[np.newaxis], column_schema, shape).storage)
    storage = pa.concat_arrays(storage_rows) if storage_rows else pa.array([], type=tensor_type.storage_type)
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
        return _fixed_shape_tensor_array(contiguous, tensor_type)

    if not contiguous.dtype.isnative:
        contiguous = contiguous.astype(contiguous.dtype.newbyteorder("="))
    # Infer the source type first: constructing a typed Arrow array from Python
    # floats can truncate even with safe=True. Casting checks the original data.
    flat = contiguous.reshape(-1)
    if (
        flat.dtype == object
        and pa.types.is_integer(tensor_type.value_type)
        and all(value is None or isinstance(value, (int, np.integer)) for value in flat)
    ):
        # Python integer inference defaults to int64, which cannot hold valid
        # UBIGINT values. Integer-only input can be range-checked directly.
        flattened = pa.array(flat, type=tensor_type.value_type, safe=True)
    else:
        flattened = pa.array(flat).cast(tensor_type.value_type, safe=True)
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
