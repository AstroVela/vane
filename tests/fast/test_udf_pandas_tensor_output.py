# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime

import numpy as np
import pyarrow as pa
import pytest

import vane
from vane.execution.udf_batch_format import iter_udf_output_tables


def _encode(series, dtype="BIGINT", shape=(2, 2), container="plain"):
    pd = pytest.importorskip("pandas")
    if container == "plain":
        entry = {"name": "x", "kind": "tensor", "dtype": dtype, "shape": list(shape)}
    else:
        declaration = f"TENSOR({dtype}, {list(shape)})"
        if container == "struct":
            declaration = f"STRUCT(value {declaration})"
        elif container == "map":
            declaration = f"MAP(VARCHAR, {declaration})"
        else:
            declaration += "[]"
        entry = {"name": "x", "kind": "duckdb_type", "type": declaration}
    (table,) = iter_udf_output_tables(pd.DataFrame({"x": series}), batch_format="pandas", output_schema=[entry])
    return table["x"]


@pytest.mark.parametrize("container", ["plain", "struct", "list", "map"])
@pytest.mark.parametrize(
    "dtype,first,last",
    [
        ("DOUBLE", 1.5, np.nan),
        ("BIGINT", 2**60 + 1, 7),
        ("UBIGINT", 2**63 + 1, 7),
        ("BOOLEAN", True, False),
        ("DATE", np.datetime64("2026-01-01"), np.datetime64("2026-01-02")),
    ],
)
def test_pandas_tensor_missing_elements_preserve_values_and_row_validity(container, dtype, first, last):
    pd = pytest.importorskip("pandas")
    tensor = np.array([[first, pd.NA], [pd.NaT, last]], dtype=object)
    rows = [tensor, pd.NA, np.full((2, 2), pd.NA, dtype=object)]
    if container in ("struct", "map"):
        rows = [{"value": row} for row in rows]
    elif container == "list":
        rows = [[row] for row in rows]
    result = _encode(pd.Series(rows, dtype=object), dtype, container=container).combine_chunks()
    if container == "struct":
        result = result.field("value")
    elif container == "map":
        result = result.items
    elif container == "list":
        result = result.values

    assert result.type.shape == [2, 2]
    assert result.is_valid().to_pylist() == [True, False, True]
    actual = result.storage.to_pylist()
    assert actual[0][1:3] == [None, None]
    assert actual[1:] == [None, [None] * 4]
    if dtype == "DATE":
        first, last = datetime.date(2026, 1, 1), datetime.date(2026, 1, 2)
    assert actual[0][0] == first
    if dtype == "DOUBLE":
        assert np.isnan(actual[0][3])
    else:
        assert actual[0][3] == last


@pytest.mark.parametrize("container", ["plain", "struct"])
@pytest.mark.parametrize("value", [1.5, float(2**63)])
def test_pandas_tensor_missing_elements_do_not_disable_checked_integer_cast(container, value):
    pd = pytest.importorskip("pandas")
    row = np.array([[value, pd.NA], [None, 0]], dtype=object)
    if container == "struct":
        row = {"value": row}
    with pytest.raises(pa.ArrowInvalid, match="truncated|out of bounds|not in range"):
        _encode(pd.Series([row], dtype=object), container=container)


@pytest.mark.parametrize("shape", [(4,), (2, 2), (1, 2, 2)])
@pytest.mark.parametrize("layout", ["sliced", "chunked", "empty"])
def test_pandas_arrow_tensor_output_keeps_shape_validity_and_buffers(shape, layout):
    pd = pytest.importorskip("pandas")
    dtype = pa.fixed_shape_tensor(pa.float64(), shape)
    child = pa.array([-1, 0, 0, 0, 0, 1.5, None, np.nan, 4, 0, 0, 0, 0], from_pandas=False).slice(1)
    storage = pa.FixedSizeListArray.from_arrays(child, 4, mask=pa.array([False, False, True])).slice(1)
    array = pa.ExtensionArray.from_storage(dtype, storage)
    if layout == "empty":
        chunks = []
    elif layout == "chunked":
        chunks = [array.slice(0, 1), array.slice(1)]
    else:
        chunks = [array]
    source = pa.chunked_array(chunks, type=dtype)
    result = _encode(pd.Series(source, dtype=pd.ArrowDtype(dtype)), "DOUBLE", shape)

    assert result.type == dtype
    assert result.num_chunks == source.num_chunks
    assert len(result) == len(source)
    for expected, actual in zip(source.chunks, result.chunks, strict=True):
        assert actual.offset == expected.offset
        assert actual.storage.values.offset == expected.storage.values.offset
        assert actual.is_valid().to_pylist() == expected.is_valid().to_pylist()
        assert [None if b is None else b.address for b in actual.buffers()] == [
            None if b is None else b.address for b in expected.buffers()
        ]
    if layout != "empty":
        rows = result.combine_chunks().storage.to_pylist()
        assert rows[0][:2] == [1.5, None] and np.isnan(rows[0][2]) and rows[0][3] == 4
        assert rows[1] is None


@pytest.mark.parametrize("value", [1.5, float(2**63)])
def test_pandas_arrow_tensor_output_rejects_lossy_integer_cast(value):
    pd = pytest.importorskip("pandas")
    dtype = pa.fixed_shape_tensor(pa.float64(), (2, 2))
    array = pa.ExtensionArray.from_storage(dtype, pa.array([[value, None, 0, 0]], type=dtype.storage_type))
    with pytest.raises(pa.ArrowInvalid, match="truncated|out of bounds|not in range"):
        _encode(pd.Series(array, dtype=pd.ArrowDtype(dtype)))


@pytest.mark.parametrize("empty", [False, True])
def test_pandas_arrow_tensor_output_checked_cast_preserves_chunks_and_nulls(empty):
    pd = pytest.importorskip("pandas")
    dtype = pa.fixed_shape_tensor(pa.float64(), (2, 2), permutation=[0, 1])
    child = pa.array([1.5] * 4 + [1, None, 3, 4] + [1.5] * 8)
    storage = pa.FixedSizeListArray.from_arrays(child, 4, mask=pa.array([False, False, True, False])).slice(1, 2)
    array = pa.ExtensionArray.from_storage(dtype, storage)
    source = pa.chunked_array([] if empty else [array.slice(0, 1), array.slice(1)], type=dtype)
    result = _encode(pd.Series(source, dtype=pd.ArrowDtype(dtype)))
    assert result.type == pa.fixed_shape_tensor(pa.int64(), (2, 2))
    assert result.num_chunks == source.num_chunks
    assert result.combine_chunks().storage.to_pylist() == ([] if empty else [[1, None, 3, 4], None])


@pytest.mark.parametrize(
    "dtype",
    [
        pa.fixed_shape_tensor(pa.int64(), (4,)),
        pa.fixed_shape_tensor(pa.int64(), (2, 2), permutation=[1, 0]),
        pa.fixed_shape_tensor(pa.int64(), (2, 2), dim_names=["height", "width"]),
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_pandas_arrow_tensor_output_requires_declared_metadata(dtype, empty):
    pd = pytest.importorskip("pandas")
    array = pa.ExtensionArray.from_storage(dtype, pa.array([] if empty else [[1, 2, 3, 4]], type=dtype.storage_type))
    with pytest.raises(ValueError, match="declared Arrow tensor metadata"):
        _encode(pd.Series(array, dtype=pd.ArrowDtype(dtype)))


@pytest.mark.parametrize("backend", ["subprocess_task", pytest.param("ray_actor", marks=pytest.mark.real_ray)])
def test_public_pandas_tensor_output_preserves_missing_elements_and_arrow_shape(request, monkeypatch, backend):
    pytest.importorskip("pandas")
    if backend == "ray_actor":
        request.getfixturevalue("ray_local")
    monkeypatch.setenv("VANE_RUNNER", "ray" if backend == "ray_actor" else "local-fast")

    class Output:
        def __call__(self, batch):
            import pandas as pd

            tensor_type = pa.fixed_shape_tensor(pa.int64(), (2, 2))
            array = pa.ExtensionArray.from_storage(
                tensor_type, pa.array([[1, None, 3, 4], None], type=tensor_type.storage_type)
            )
            rows = [np.array([1.5, pd.NA], dtype=object), None]
            return pd.DataFrame(
                {
                    "tensor": pd.Series(rows, dtype=object),
                    "nested": pd.Series([{"value": value} for value in rows], dtype=object),
                    "arrow_tensor": pd.Series(array, dtype=pd.ArrowDtype(tensor_type)),
                }
            )

    with vane.connect(config={"threads": 2}) as con:
        tensor_type = vane.tensor_type(vane.sqltypes.DOUBLE, [2])
        result = (
            con.sql("SELECT 1 AS input")
            .map_batches(
                Output if backend == "ray_actor" else lambda batch: Output()(batch),
                schema={
                    "tensor": tensor_type,
                    "nested": vane.struct_type({"value": tensor_type}),
                    "arrow_tensor": vane.tensor_type(vane.sqltypes.BIGINT, [2, 2]),
                },
                batch_format="pandas",
                execution_backend=backend,
                actor_number=1 if backend == "ray_actor" else None,
            )
            .to_arrow_table()
        )
    assert result["tensor"].combine_chunks().storage.to_pylist() == [[1.5, None], None]
    assert result["nested"].combine_chunks().field("value").storage.to_pylist() == [[1.5, None], None]
    assert result["arrow_tensor"].type.shape == [2, 2]
    assert result["arrow_tensor"].combine_chunks().storage.to_pylist() == [[1, None, 3, 4], None]
