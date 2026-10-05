# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import vane
from vane.execution.udf_batch_format import format_udf_input, iter_udf_output_tables


def _round_trip(array, declaration, batch_format, writable=False):
    batch = format_udf_input(pa.table({"x": array}), batch_format, zero_copy_batch=not writable)
    (result,) = iter_udf_output_tables(
        batch,
        batch_format=batch_format,
        output_schema=[{"name": "x", "kind": "duckdb_type", "type": declaration}],
    )
    return batch, result.column("x").combine_chunks()


def _wrap_leaf(leaf, declaration, container):
    if container == "list":
        return pa.ListArray.from_arrays([0, len(leaf), len(leaf)], leaf), f"{declaration}[]"
    if container == "array":
        return pa.FixedSizeListArray.from_arrays(leaf, len(leaf)), f"{declaration}[{len(leaf)}]"
    if container == "struct":
        return pa.StructArray.from_arrays([leaf], names=["value"]), f"STRUCT(value {declaration})"
    if container == "map":
        return (
            pa.MapArray.from_arrays([0, len(leaf)], pa.array([str(i) for i in range(len(leaf))]), leaf),
            f"MAP(VARCHAR, {declaration})",
        )
    return leaf, declaration


@pytest.mark.parametrize("batch_format,writable", [("numpy", False), ("numpy", True), ("pandas", False)])
@pytest.mark.parametrize("container", ["plain", "list", "array", "struct", "map"])
@pytest.mark.parametrize("unit", ["us", "ns"])
@pytest.mark.parametrize("timezone", [None, "Asia/Shanghai", "America/New_York"])
def test_timestamps_keep_timezone_units_and_nulls(batch_format, writable, container, unit, timezone):
    dtype = pa.timestamp(unit, tz=timezone)
    leaf = pa.array([123, None, -877], type=dtype)
    declaration = "TIMESTAMPTZ" if timezone else "TIMESTAMP_NS" if unit == "ns" else "TIMESTAMP"
    array, declaration = _wrap_leaf(leaf, declaration, container)
    _, result = _round_trip(array, declaration, batch_format, writable)
    assert result.equals(array)


@pytest.mark.parametrize("batch_format,writable", [("numpy", False), ("numpy", True), ("pandas", False)])
@pytest.mark.parametrize("container", ["plain", "list", "array", "struct", "map", "dictionary"])
@pytest.mark.parametrize("logical_type", ["bit", "hugeint", "uhugeint"])
def test_opaque_logical_values_keep_extension_metadata(batch_format, writable, container, logical_type):
    if logical_type == "bit":
        dtype = pa.opaque(pa.binary(), logical_type, "DuckDB")
        values = [b"\x05\xfd", None, b"\x00\xa5"]
    else:
        dtype = pa.opaque(pa.binary(16), logical_type, "DuckDB")
        signed = logical_type == "hugeint"
        numbers = [-(2**127) if signed else 2**128 - 1, None, 2**100 + 123]
        values = [None if value is None else value.to_bytes(16, "little", signed=signed) for value in numbers]
    leaf = pa.ExtensionArray.from_storage(dtype, pa.array(values, type=dtype.storage_type))
    declaration = logical_type.upper()
    if container == "dictionary":
        array = pa.DictionaryArray.from_arrays(pa.array([2, 0, None]), leaf)
        expected = array.dictionary_decode()
    else:
        array, declaration = _wrap_leaf(leaf, declaration, container)
        expected = array
    _, result = _round_trip(array, declaration, batch_format, writable)
    assert result.equals(expected)


@pytest.mark.parametrize("batch_format,writable", [("numpy", False), ("numpy", True), ("pandas", False)])
@pytest.mark.parametrize("container", ["list", "array", "struct", "map"])
@pytest.mark.parametrize("empty", [False, True])
def test_nested_tensor_values_keep_shape_nulls_and_offsets(batch_format, writable, container, empty):
    dtype = pa.fixed_shape_tensor(pa.int64(), [2, 2])
    storage = pa.array([[0, 0, 0, 0], [1, None, 3, 4], None, [5, 6, 7, 8]], type=dtype.storage_type)
    leaf = pa.ExtensionArray.from_storage(dtype, storage).slice(1)
    array, declaration = _wrap_leaf(leaf, "TENSOR(BIGINT, [2, 2])", container)
    if empty:
        array = array.slice(0, 0)
    _, result = _round_trip(array, declaration, batch_format, writable)
    assert result.equals(array)


@pytest.mark.parametrize("batch_format", ["numpy", "pandas"])
def test_nested_tensor_permutation_and_buffer_ownership(batch_format):
    dtype = pa.fixed_shape_tensor(pa.int64(), [2, 2], permutation=[1, 0])
    storage = pa.FixedSizeListArray.from_arrays(pa.array([-1, 1, 2, 3, 4]).slice(1), 4)
    leaf = pa.ExtensionArray.from_storage(dtype, storage)
    array, declaration = _wrap_leaf(leaf, "TENSOR(BIGINT, [2, 2])", "list")
    batch, result = _round_trip(array, declaration, batch_format)
    tensor = batch["x"][0][0]
    np.testing.assert_array_equal(tensor, [[1, 3], [2, 4]])
    assert tensor.flags.writeable == (batch_format == "pandas")
    assert result.values.storage.to_pylist() == [[1, 3, 2, 4]]
    if batch_format == "pandas":
        tensor[0, 0] = 99
        assert leaf.storage.to_pylist() == [[1, 2, 3, 4]]


@pytest.mark.parametrize("batch_format", ["numpy", "pandas"])
@pytest.mark.parametrize("container", ["list", "array", "struct", "map"])
def test_null_nested_tensor_containers_use_declared_extension_type(batch_format, container):
    leaf = pa.nulls(2, type=pa.fixed_shape_tensor(pa.float32(), [2]))
    array, declaration = _wrap_leaf(leaf, "TENSOR(FLOAT, [2])", container)
    array = pa.nulls(len(array), type=array.type)
    _, result = _round_trip(array, declaration, batch_format)
    assert result.equals(array)


@pytest.mark.parametrize("batch_format", ["pyarrow", "numpy", "pandas"])
@pytest.mark.parametrize("backend", ["subprocess_task", pytest.param("ray_actor", marks=pytest.mark.real_ray)])
def test_public_identity_preserves_nested_extensions_and_timestamp_values(request, monkeypatch, batch_format, backend):
    if backend == "ray_actor":
        request.getfixturevalue("ray_local")
    monkeypatch.setenv("VANE_RUNNER", "ray" if backend == "ray_actor" else "local-fast")

    class Identity:
        def __call__(self, batch):
            return batch

    expressions = {
        "tensor_list": "[[1., 2.]::TENSOR(FLOAT, [2]), NULL::TENSOR(FLOAT, [2])]",
        "tensor_array": "[[1., NULL]::TENSOR(FLOAT, [2])]::TENSOR(FLOAT, [2])[1]",
        "tensor_struct": "{'value': [1., 2.]::TENSOR(FLOAT, [2])}",
        "tensor_map": "map(['a'], [[1., 2.]::TENSOR(FLOAT, [2])])",
        "zoned": "'2026-01-01 00:00:00+00'::TIMESTAMPTZ",
        "timestamp_list": "['2026-01-01 00:00:00.000000123'::TIMESTAMP_NS, NULL]",
        "timestamp_struct": "{'value': '2026-01-01 00:00:00.000000123'::TIMESTAMP_NS}",
        "timestamp_map": "map(['a'], ['2026-01-01 00:00:00.000000123'::TIMESTAMP_NS])",
        "bits": "'101'::BIT",
        "huge": "'1267650600228229401496703205499'::HUGEINT",
        "unsigned_huge": "'340282366920938463463374607431768211455'::UHUGEINT",
        "binary": r"'\x05\xFD'::BLOB",
    }
    columns = ", ".join(
        f"CASE WHEN id = 1 THEN NULL ELSE {expression} END AS {name}" for name, expression in expressions.items()
    )
    with vane.connect(config={"threads": 2, "TimeZone": "Asia/Shanghai", "arrow_lossless_conversion": True}) as con:
        relation = con.sql(f"SELECT id, {columns} FROM range(2) t(id)")
        expected = relation.to_arrow_table().sort_by("id")
        result = (
            relation.map_batches(
                Identity if backend == "ray_actor" else lambda batch: batch,
                schema=dict(zip(relation.columns, relation.types, strict=True)),
                batch_format=batch_format,
                execution_backend=backend,
                actor_number=1 if backend == "ray_actor" else None,
            )
            .to_arrow_table()
            .sort_by("id")
        )
    assert result.equals(expected)


@pytest.mark.parametrize("batch_format", ["numpy", "pandas"])
def test_logical_extension_casts_keep_source_provenance(monkeypatch, batch_format):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(config={"threads": 2, "arrow_lossless_conversion": True}) as con:
        relation = con.sql("SELECT '101'::BIT AS x, '-123'::HUGEINT AS y")
        expected = relation.project("x::VARCHAR AS x, y::VARCHAR AS y").fetchall()
        actual = relation.map_batches(
            lambda batch: batch,
            schema={"x": vane.sqltypes.VARCHAR, "y": vane.sqltypes.VARCHAR},
            batch_format=batch_format,
            execution_backend="subprocess_task",
        ).fetchall()
    assert actual == expected
