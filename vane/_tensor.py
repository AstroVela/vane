# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Variable shape Tensor values and canonical Arrow transport."""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pyarrow as pa

_MAX_DIMENSION = (1 << 31) - 1
_MAX_RANK = 32
_EXTENSION_NAME = "arrow.variable_shape_tensor"


def is_variable_tensor(dtype: Any) -> bool:
    return str(dtype.id) == "tensor" and any(dim is None for dim in dict(dtype.children)["shape"])


def _shape(shape: Sequence[int | None]) -> tuple[int | None, ...]:
    normalized: list[int | None] = []
    for dim in shape:
        if dim is None:
            normalized.append(None)
        else:
            if isinstance(dim, (bool, np.bool_)):
                raise ValueError("Tensor dimensions must be integers or None")
            dimension = operator.index(dim)
            if not 0 <= dimension <= _MAX_DIMENSION:
                raise ValueError("Tensor dimensions must fit in nonnegative int32")
            normalized.append(dimension)
    if not 1 <= len(normalized) <= _MAX_RANK:
        raise ValueError("Variable Tensor rank must be between 1 and 32")
    return tuple(normalized)


def _check_dtype(dtype: pa.DataType) -> None:
    if not (pa.types.is_boolean(dtype) or pa.types.is_integer(dtype) or dtype in (pa.float32(), pa.float64())):
        raise ValueError("Variable Tensor requires Boolean or 8/16/32/64-bit numeric elements")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate Tensor metadata key: {name}")
        result[name] = value
    return result


class VariableShapeTensorType(pa.ExtensionType):
    """Python binding for Arrow's canonical type, preserving its metadata.

    Arrow's C++ type has no corresponding Python type/scalar or pickle binding.
    Vane's restrictions are checked when values enter Vane, not by this Arrow
    registry binding: other Arrow clients can retain layouts and dimension names.
    """

    def __init__(
        self,
        value_type: pa.DataType,
        uniform_shape: Sequence[int | None],
        *,
        _storage_type: pa.DataType | None = None,
        _metadata: dict[str, Any] | None = None,
    ) -> None:
        self._uniform_shape = tuple(uniform_shape)
        self._metadata = {"uniform_shape": self._uniform_shape} if _metadata is None else _metadata
        self._value_type = value_type
        storage = pa.struct(
            [pa.field("data", pa.list_(value_type)), pa.field("shape", pa.list_(pa.int32(), len(self._uniform_shape)))]
        )
        super().__init__(storage if _storage_type is None else _storage_type, _EXTENSION_NAME)

    @property
    def value_type(self) -> pa.DataType:
        return self._value_type

    @property
    def uniform_shape(self) -> tuple[int | None, ...]:
        return self._uniform_shape

    def __arrow_ext_serialize__(self) -> bytes:
        return json.dumps(self._metadata, separators=(",", ":")).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type: pa.DataType, serialized: bytes) -> VariableShapeTensorType:
        if not pa.types.is_struct(storage_type) or storage_type.names != ["data", "shape"]:
            raise ValueError("Variable Tensor requires STRUCT(data LIST, shape INTEGER[rank])")
        data_type, shape_type = storage_type[0].type, storage_type[1].type
        if (
            not pa.types.is_list(data_type)
            or not pa.types.is_fixed_size_list(shape_type)
            or shape_type.value_type != pa.int32()
        ):
            raise ValueError("Variable Tensor requires a list and a fixed-size int32 shape")
        rank = shape_type.list_size
        metadata = json.loads(serialized, object_pairs_hook=_object) if serialized else {}
        if not isinstance(metadata, dict):
            raise ValueError("Invalid variable Tensor metadata")
        shape = metadata.get("uniform_shape")
        if shape is None:
            shape = [None] * rank
        if not isinstance(shape, list) or len(shape) != rank:
            raise ValueError("Variable Tensor uniform shape must match its rank")
        if any(dim is not None and (type(dim) is not int or not 0 <= dim <= _MAX_DIMENSION) for dim in shape):
            raise ValueError("Tensor uniform dimensions must be nonnegative int32 or NULL")
        return cls(data_type.value_type, shape, _storage_type=storage_type, _metadata=metadata)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return self.__arrow_ext_deserialize__, (self.storage_type, self.__arrow_ext_serialize__())


# Replace the C++-only registry entry with its Python binding. The binding
# preserves canonical metadata and supplies pickling for Arrow schemas/arrays.
pa.unregister_extension_type(_EXTENSION_NAME)
pa.register_extension_type(VariableShapeTensorType(pa.float64(), (None,)))


def _validate_arrow_tensor_type(dtype: pa.DataType) -> VariableShapeTensorType:
    if not isinstance(dtype, pa.BaseExtensionType) or dtype.extension_name != _EXTENSION_NAME:
        raise ValueError("Tensor batch values must carry arrow.variable_shape_tensor")
    # The C Data interface also accepts canonical C++ types created before Vane
    # installed its Python binding, without relying on private type attributes.
    canonical = pa.DataType._import_from_c_capsule(dtype.__arrow_c_schema__())
    serialized = canonical.__arrow_ext_serialize__()
    if len(serialized) > 16 * 1024:
        raise ValueError("Variable Tensor metadata exceeds 16 KiB")
    actual = VariableShapeTensorType.__arrow_ext_deserialize__(dtype.storage_type, serialized)
    _check_dtype(actual.value_type)
    shape = _shape(actual.uniform_shape)
    if all(dim is not None for dim in shape):
        raise ValueError("Fully uniform tensors must use the fixed shape Tensor type")
    metadata = actual._metadata
    if set(metadata) - {"uniform_shape", "permutation", "dim_names"}:
        raise ValueError("Invalid variable Tensor metadata")
    if metadata.get("dim_names") is not None:
        raise ValueError("Variable Tensor dimension names are not supported")
    permutation = metadata.get("permutation")
    if permutation is not None and (
        not isinstance(permutation, list)
        or any(type(dim) is not int for dim in permutation)
        or permutation != list(range(len(shape)))
    ):
        raise ValueError("Variable Tensor requires row-major identity permutation")
    return actual


def tensor_arrow_type(dtype: pa.DataType, shape: Sequence[int | None]) -> pa.DataType:
    if any(dim is None for dim in shape):
        return VariableShapeTensorType(dtype, shape)
    return pa.fixed_shape_tensor(dtype, cast(tuple[int, ...], tuple(shape)))


def _validate_shape(actual: Sequence[int], declared: Sequence[int | None], count: int) -> tuple[int, ...]:
    shape = _shape(actual)
    if len(shape) != len(declared) or any(dim is None for dim in shape):
        raise ValueError("Tensor actual shape must have the declared rank and no NULL dimensions")
    dimensions = cast(tuple[int, ...], shape)
    for dim, expected in zip(shape, declared, strict=True):
        if expected is not None and dim != expected:
            raise ValueError("Tensor actual shape does not match its uniform dimensions")
    size = 0 if 0 in shape else 1
    if size:
        for dim in dimensions:
            size *= dim
            if size > _MAX_DIMENSION:
                raise ValueError("Tensor element count exceeds signed 32-bit Arrow list offsets")
    if size != count:
        raise ValueError("Tensor shape product does not match its element count")
    return dimensions


def _storage_to_numpy(value: Mapping[str, Any], dtype: VariableShapeTensorType) -> np.ndarray:
    if set(value) != {"data", "shape"} or value["data"] is None or value["shape"] is None:
        raise ValueError("Non-NULL Tensor requires data and shape")
    data = value["data"]
    if any(element is None for element in data):
        raise ValueError("Tensor elements cannot be NULL")
    shape = _validate_shape(value["shape"], dtype.uniform_shape, len(data))
    return np.array(data, dtype=dtype.value_type.to_pandas_dtype()).reshape(shape)


def _numpy_to_storage(value: Any, dtype: VariableShapeTensorType) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, np.ndarray):
        raise ValueError("Tensor Python values must be NumPy arrays or None")
    if value.dtype != np.dtype(dtype.value_type.to_pandas_dtype()):
        raise ValueError(f"Tensor dtype must be {dtype.value_type}, got {value.dtype}")
    shape = _validate_shape(value.shape, dtype.uniform_shape, value.size)
    return {"data": value.ravel(order="C").tolist(), "shape": list(shape)}


def _native_value_storage(value: Any, dtype: Any) -> dict[str, Any] | None:
    from vane.execution.udf_output_schema import _arrow_type_from_duckdb_pytype

    return _numpy_to_storage(value, _arrow_type_from_duckdb_pytype(dtype))


def _native_value_to_numpy(value: Mapping[str, Any], dtype: Any) -> np.ndarray:
    from vane.execution.udf_output_schema import _arrow_type_from_duckdb_pytype

    return _storage_to_numpy(value, _arrow_type_from_duckdb_pytype(dtype))


def tensor_array(values: Sequence[np.ndarray | None], dtype: Any) -> pa.ExtensionArray:
    """Build a validated Arrow Tensor column from NumPy row values and a Vane Tensor dtype."""
    from vane.execution.udf_output_schema import _arrow_type_from_duckdb_pytype

    if not is_variable_tensor(dtype):
        raise ValueError("tensor_array requires a variable shape Tensor dtype")
    arrow_type = _arrow_type_from_duckdb_pytype(dtype)
    storage = pa.array([_numpy_to_storage(value, arrow_type) for value in values], type=arrow_type.storage_type)
    return pa.ExtensionArray.from_storage(arrow_type, storage)


def validate_tensor_array(array: Any, dtype: Any, *, boundary: str, allow_untyped_null: bool = False) -> Any:
    from vane.execution.udf_output_schema import _arrow_type_from_duckdb_pytype

    expected = _arrow_type_from_duckdb_pytype(dtype)
    if isinstance(array, pa.ChunkedArray):
        return pa.chunked_array(
            [
                validate_tensor_array(chunk, dtype, boundary=boundary, allow_untyped_null=allow_untyped_null)
                for chunk in array.chunks
            ],
            type=expected,
        )
    if allow_untyped_null and pa.types.is_null(array.type):
        return pa.nulls(len(array), type=expected)
    if not isinstance(array, pa.ExtensionArray) or array.type.extension_name != _EXTENSION_NAME:
        raise ValueError(f"{boundary}: Tensor batch values must carry arrow.variable_shape_tensor")
    actual = _validate_arrow_tensor_type(array.type)
    if actual.value_type != expected.value_type or actual.uniform_shape != expected.uniform_shape:
        raise ValueError(f"{boundary}: Tensor dtype or declared shape does not match its contract")
    storage = array.storage
    data = storage.field("data")
    shapes = storage.field("shape")
    for row in range(len(storage)):
        if not storage[row].is_valid:
            continue
        if not data[row].is_valid or not shapes[row].is_valid:
            raise ValueError(f"{boundary}: non-NULL Tensor requires data and shape")
        elements = data[row].values
        if elements.null_count:
            raise ValueError(f"{boundary}: Tensor elements cannot be NULL")
        _validate_shape(shapes[row].as_py(), expected.uniform_shape, len(elements))
    if storage.type != expected.storage_type:
        storage = pa.StructArray.from_arrays([data, shapes], fields=expected.storage_type, mask=storage.is_null())
    return pa.ExtensionArray.from_storage(expected, storage)


def tensor(data: Any, shape: Any) -> Any:
    """Construct a validated variable shape Tensor expression without I/O."""
    import vane
    from vane._expressions import as_expression

    return vane.FunctionExpression("tensor", as_expression(data), as_expression(shape))


def tensor_data(value: Any) -> Any:
    """Return the flattened element list of a variable Tensor expression."""
    import vane
    from vane._expressions import as_expression

    return vane.FunctionExpression("tensor_data", as_expression(value))


def tensor_shape(value: Any) -> Any:
    """Return the actual shape of each variable Tensor value."""
    import vane
    from vane._expressions import as_expression

    return vane.FunctionExpression("tensor_shape", as_expression(value))
