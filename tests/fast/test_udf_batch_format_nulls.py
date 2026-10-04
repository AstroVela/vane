# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime

import numpy as np
import pyarrow as pa
import pytest

import vane
from vane.execution.udf_batch_format import format_udf_input, iter_udf_output_tables


def _schema(dtype, shape=None):
    return [
        {"name": "x", "kind": "tensor" if shape else "duckdb_type", "dtype": dtype, "type": dtype, "shape": shape or []}
    ]


def _round_trip(array, batch_format, schema, *, writable=False):
    table = pa.table({"x": array})
    batch = format_udf_input(table, batch_format, zero_copy_batch=not writable)
    (result,) = iter_udf_output_tables(batch, batch_format=batch_format, output_schema=schema)
    return batch, result.column("x").combine_chunks()


@pytest.mark.parametrize("batch_format,writable", [("numpy", False), ("numpy", True), ("pandas", False)])
@pytest.mark.parametrize(
    "dtype,arrow_type,first,last",
    [
        ("FLOAT", pa.float32(), 1.0, float("nan")),
        ("BIGINT", pa.int64(), 2**60 + 1, 7),
        ("UBIGINT", pa.uint64(), 2**63 + 1, 7),
        ("BOOLEAN", pa.bool_(), True, False),
        ("DATE", pa.date32(), datetime.date(2026, 1, 1), datetime.date(2026, 1, 2)),
    ],
)
@pytest.mark.parametrize("sliced", [False, True])
def test_tensor_child_nulls_remain_distinct_from_null_rows_and_nan(
    batch_format, writable, dtype, arrow_type, first, last, sliced
):
    tensor_type = pa.fixed_shape_tensor(arrow_type, (2,))
    rows = [[first, None], [None, None], None, [first, last]]
    storage = pa.array(([rows[-1]] if sliced else []) + rows, type=tensor_type.storage_type)
    array = pa.ExtensionArray.from_storage(tensor_type, storage).slice(int(sliced))

    batch, result = _round_trip(array, batch_format, _schema(dtype, [2]), writable=writable)

    column = batch["x"]
    assert column[0][1] is None and column[1].tolist() == [None, None] and column[2] is None
    if batch_format == "numpy":
        assert column[0].flags.writeable == writable
    actual = result.storage.to_pylist()
    assert actual[:3] == rows[:3]
    assert actual[3][0] == first
    if isinstance(last, float) and np.isnan(last):
        assert np.isnan(actual[3][1])
    else:
        assert actual[3][1] == last


@pytest.mark.parametrize("batch_format", ["numpy", "pandas"])
@pytest.mark.parametrize(
    "dtype,arrow_type,values",
    [
        ("BOOLEAN", pa.bool_(), [True, False]),
        ("DATE", pa.date32(), [datetime.date(2026, 1, 1), datetime.date(2026, 1, 2)]),
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_boolean_and_date_tensor_inputs_keep_shape_and_element_type(batch_format, dtype, arrow_type, values, empty):
    tensor_type = pa.fixed_shape_tensor(arrow_type, (2,))
    array = pa.ExtensionArray.from_storage(
        tensor_type, pa.array([] if empty else [values], type=tensor_type.storage_type)
    )
    batch, result = _round_trip(array, batch_format, _schema(dtype, [2]))
    assert result.storage.to_pylist() == ([] if empty else [values])
    if batch_format == "numpy":
        assert batch["x"].shape == (0 if empty else 1, 2)
        assert batch["x"].dtype == np.dtype(bool if dtype == "BOOLEAN" else "datetime64[D]")


@pytest.mark.parametrize("permutation", [(1, 0), (2, 0, 1)])
def test_tensor_child_nulls_follow_permutation_and_both_offsets(permutation):
    shape = (2, 3) if len(permutation) == 2 else (2, 3, 4)
    size = int(np.prod(shape))
    values = list(range(size * 3))
    values[size + 1] = None
    child = pa.array([-1, *values], type=pa.int64()).slice(1)
    storage = pa.FixedSizeListArray.from_arrays(child, size).slice(1, 1)
    array = pa.ExtensionArray.from_storage(pa.fixed_shape_tensor(pa.int64(), shape, permutation=permutation), storage)
    batch = format_udf_input(pa.table({"x": array}), "numpy")
    expected = np.array(values[size : size * 2], dtype=object).reshape(shape).transpose(permutation)
    np.testing.assert_array_equal(batch["x"][0], expected)
    assert not batch["x"].flags.writeable


@pytest.mark.parametrize("writable", [False, True])
@pytest.mark.parametrize("container", ["list", "fixed_list", "struct"])
def test_nested_numpy_nulls_preserve_integer_precision_and_nan(writable, container):
    numbers = [2**60 + 1, None, 7]
    floating = [1.0, None, float("nan")]
    if container == "struct":
        dtype = pa.struct([("ints", pa.list_(pa.int64())), ("floats", pa.list_(pa.float64()))])
        array = pa.array([{"ints": numbers, "floats": floating}, None], type=dtype)
        declaration = "STRUCT(ints BIGINT[], floats DOUBLE[])"
    else:
        child_type = pa.list_(pa.int64(), 3) if container == "fixed_list" else pa.list_(pa.int64())
        array = pa.array([[numbers, None], None], type=pa.list_(child_type))
        declaration = "BIGINT[3][]" if container == "fixed_list" else "BIGINT[][]"
    batch, result = _round_trip(array.slice(0), "numpy", _schema(declaration), writable=writable)
    assert isinstance(batch["x"], np.ma.MaskedArray)
    assert batch["x"].mask.tolist() == [False, True]
    actual = result.to_pylist()
    assert actual[1] is None
    if container == "struct":
        assert actual[0]["ints"] == numbers
        assert actual[0]["floats"][:2] == floating[:2]
        assert np.isnan(actual[0]["floats"][2])
    else:
        assert actual[0] == [numbers, None]


@pytest.mark.parametrize("form", ["dense", "non_native", "object", "nullable", "masked", "pandas"])
@pytest.mark.parametrize("values", [[1.9, 2.1], [float(2**63), 0]])
def test_tensor_integer_cast_rejects_loss_in_every_output_form(form, values):
    dense = np.array([values], dtype=np.float64)
    if form == "dense":
        output = dense
    elif form == "non_native":
        output = dense.astype(">f8")
    elif form == "object":
        output = dense.astype(object)
    elif form == "masked":
        output = np.ma.MaskedArray(np.vstack([dense, [0, 0]]), mask=[[False, False], [True, True]])
    else:
        output = np.empty(2, dtype=object)
        output[:] = [dense[0], None]
    batch = {"x": output}
    if form == "pandas":
        pd = pytest.importorskip("pandas")
        batch = pd.DataFrame(batch)
    with pytest.raises(pa.ArrowInvalid, match="truncated|out of bounds|not in range"):
        list(
            iter_udf_output_tables(
                batch, batch_format="pandas" if form == "pandas" else "numpy", output_schema=_schema("BIGINT", [2])
            )
        )


@pytest.mark.parametrize("container", ["map", "list", "array", "struct"])
@pytest.mark.parametrize("empty", [False, True])
def test_numpy_map_containers_preserve_nulls_and_inferred_leaf_types(container, empty):
    row = [("a", 1.75), ("b", None)]
    map_type = pa.map_(pa.string(), pa.float64())
    declaration = "MAP(VARCHAR, BIGINT)"
    if container == "list":
        dtype, rows = pa.list_(map_type), [[row, None], None, []]
        declaration += "[]"
    elif container == "array":
        dtype, rows = pa.list_(map_type, 2), [[row, None], None, [[], []]]
        declaration += "[2]"
    elif container == "struct":
        dtype, rows = pa.struct([("items", map_type)]), [{"items": row}, None, {"items": []}]
        declaration = "STRUCT(items MAP(VARCHAR, BIGINT))"
    else:
        dtype, rows = map_type, [row, None, []]
    array = pa.array([] if empty else rows, type=dtype)
    _, result = _round_trip(array, "numpy", _schema(declaration))
    assert result.to_pylist() == ([] if empty else rows)
    if not empty:
        assert result.type == dtype


@pytest.mark.parametrize("batch_format", ["pyarrow", "numpy", "pandas"])
@pytest.mark.parametrize("backend", ["subprocess_task", pytest.param("ray_actor", marks=pytest.mark.real_ray)])
def test_public_identity_preserves_tensor_nested_nulls_and_maps(request, monkeypatch, batch_format, backend):
    if backend == "ray_actor":
        request.getfixturevalue("ray_local")
    monkeypatch.setenv("VANE_RUNNER", "ray" if backend == "ray_actor" else "local-fast")

    class Identity:
        def __call__(self, batch):
            return batch

    sql = """
        SELECT [1.0, NULL]::TENSOR(FLOAT, [2]) AS tensor_null,
               [true, false]::TENSOR(BOOLEAN, [2]) AS tensor_bool,
               [DATE '2026-01-01', DATE '2026-01-02']::TENSOR(DATE, [2]) AS tensor_date,
               [1., NULL, 'NaN'::DOUBLE]::DOUBLE[] AS list_float,
               [1152921504606846977, NULL]::BIGINT[] AS list_int,
               map(['a'], [1]) AS mapping,
               [map(['a'], [1])] AS nested_mapping
    """
    with vane.connect(config={"threads": 2}) as con:
        relation = con.sql(sql)
        expected = relation.fetchall()[0]
        actual = relation.map_batches(
            Identity if backend == "ray_actor" else lambda batch: batch,
            schema=dict(zip(relation.columns, relation.types, strict=True)),
            batch_format=batch_format,
            execution_backend=backend,
            actor_number=1 if backend == "ray_actor" else None,
        ).fetchone()
    assert actual[:3] == expected[:3]
    assert actual[3][:2] == [1.0, None] and np.isnan(actual[3][2])
    assert actual[4:] == expected[4:]


@pytest.mark.parametrize(
    "expression,target", [("map(['a'], [1.75])", "BIGINT"), (r"map(['a'], ['\xFF'::BLOB])", "VARCHAR")]
)
def test_public_numpy_map_values_keep_duckdb_cast_semantics(monkeypatch, expression, target):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(config={"threads": 2}) as con:
        relation = con.sql(f"SELECT {expression} AS x")
        expected = con.sql(f"SELECT {expression}::MAP(VARCHAR, {target})").fetchall()
        actual = relation.map_batches(
            lambda batch: batch,
            schema={"x": vane.type(f"MAP(VARCHAR, {target})")},
            batch_format="numpy",
            execution_backend="subprocess_task",
        ).fetchall()
    assert actual == expected


@pytest.mark.parametrize("row,error", [([("a", 1), (None, 2)], "NULL"), (["ab"], "key/value"), ("ab", "mappings")])
def test_numpy_map_output_rejects_invalid_keys_and_pairs(row, error):
    values = np.empty(1, dtype=object)
    values[0] = row
    with pytest.raises((ValueError, TypeError), match=error):
        list(iter_udf_output_tables({"x": values}, batch_format="numpy", output_schema=_schema("MAP(VARCHAR, BIGINT)")))
