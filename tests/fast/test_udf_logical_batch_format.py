# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from decimal import Decimal

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


@pytest.mark.parametrize("writable", [False, True])
@pytest.mark.parametrize("container", ["plain", "list", "array", "struct", "map"])
@pytest.mark.parametrize("chunked", [False, True])
def test_numpy_hugeint_keeps_decimal128_source_type_at_signed_boundaries(writable, container, chunked):
    # DuckDB exports HUGEINT as decimal128(38, 0), whose physical buffer can
    # contain 39-digit integers outside the nominal decimal precision.
    values = [-1, -(2**127), 0, 2**127 - 1, 10**38 - 1]
    data = b"".join(value.to_bytes(16, "little", signed=True) for value in values)
    leaf = pa.Array.from_buffers(pa.decimal128(38, 0), 5, [pa.py_buffer(b"\x1b"), pa.py_buffer(data)]).slice(1)
    array, declaration = _wrap_leaf(leaf, "HUGEINT", container)
    source = pa.chunked_array([array.slice(0, 1), array.slice(1)]) if chunked else array
    batch, result = _round_trip(source, declaration, "numpy", writable)
    assert result.equals(array)
    if container == "plain":
        assert isinstance(batch["x"][0], pa.Decimal128Scalar)
        assert batch["x"][0].as_py() == Decimal(-(2**127))
        assert batch["x"].flags.writeable == writable


@pytest.mark.parametrize("container", ["plain", "list", "array", "struct", "map"])
def test_numpy_decimal_retains_precision_and_scale(container):
    leaf = pa.array([Decimal("1.2500"), None, Decimal("-123456789.9876")], type=pa.decimal128(18, 4))
    array, declaration = _wrap_leaf(leaf, "DECIMAL(18, 4)", container)
    _, result = _round_trip(array, declaration, "numpy")
    assert result.equals(array)


@pytest.mark.parametrize("mode", ["pyarrow", "numpy", "numpy_writable", "pandas"])
@pytest.mark.parametrize("backend", ["subprocess_task", pytest.param("ray_actor", marks=pytest.mark.real_ray)])
def test_public_hugeint_identity_with_default_arrow_conversion(request, monkeypatch, mode, backend):
    if backend == "ray_actor":
        request.getfixturevalue("ray_local")
    monkeypatch.setenv("VANE_RUNNER", "ray" if backend == "ray_actor" else "local-fast")

    class Identity:
        def __call__(self, batch):
            return batch

    values = [-(2**127), 2**127 - 1, 10**38 - 1, 0, None]
    rows = ", ".join(
        f"({index}, {'NULL' if value is None else repr(str(value))}::HUGEINT)" for index, value in enumerate(values)
    )
    with vane.connect(config={"threads": 2}) as con:
        relation = con.sql(
            "SELECT id, x, [x, NULL::HUGEINT] AS list_value, [x, NULL::HUGEINT]::HUGEINT[2] AS array_value, "
            "{'value': x} AS struct_value, map(['key'], [x]) AS map_value "
            f"FROM (VALUES {rows}) t(id, x)"
        )
        # Ray's final result transport rejects HUGEINT when lossless Arrow is
        # disabled. Keep HUGEINT through the UDF, then compare exact SQL text
        # after a downstream projection, without changing the default input.
        projection = "id, " + ", ".join(f"{name}::VARCHAR AS {name}" for name in relation.columns[1:])
        expected = relation.project(projection).order("id").fetchall()
        actual = (
            relation.map_batches(
                Identity if backend == "ray_actor" else lambda batch: batch,
                schema=dict(zip(relation.columns, relation.types, strict=True)),
                batch_format="numpy" if mode == "numpy_writable" else mode,
                zero_copy_batch=mode != "numpy_writable",
                execution_backend=backend,
                actor_number=1 if backend == "ray_actor" else None,
            )
            .project(projection)
            .order("id")
            .fetchall()
        )
    assert actual == expected


@pytest.mark.parametrize("batch_format,writable", [("numpy", False), ("numpy", True), ("pandas", False)])
@pytest.mark.parametrize("container", ["plain", "list", "array", "struct", "map"])
def test_intervals_preserve_calendar_components_nanoseconds_and_nulls(batch_format, writable, container):
    leaf = pa.array([(1, 2, 123), None, (-2, -3, -987)], type=pa.month_day_nano_interval())
    array, declaration = _wrap_leaf(leaf, "INTERVAL", container)
    batch, result = _round_trip(array, declaration, batch_format, writable)
    assert result.equals(array)
    if batch_format == "numpy" and container == "plain":
        assert isinstance(batch["x"][0], pa.MonthDayNanoIntervalScalar)
        assert batch["x"][0].as_py() == pa.MonthDayNano((1, 2, 123))
        assert batch["x"].flags.writeable == writable


@pytest.mark.parametrize("route", ["adapter", "public"])
@pytest.mark.parametrize("mode", ["pyarrow", "numpy", "numpy_writable"])
def test_interval_batches_without_optional_pandas(tmp_path, route, mode):
    # sitecustomize applies before Arrow imports in both this fresh interpreter
    # and the public API's subprocess worker. A prior pandas import cannot mask
    # the missing-dependency crash in Arrow's INTERVAL NumPy bridge.
    (tmp_path / "sitecustomize.py").write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys

        class NoPandas(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "pandas" or fullname.startswith("pandas."):
                    raise ModuleNotFoundError("pandas is not installed", name=fullname)

        sys.meta_path.insert(0, NoPandas())
        if sys.platform != "win32":
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    """)
    )
    script = textwrap.dedent('''
        import sys
        import pyarrow as pa
        import vane
        from vane.execution.udf_batch_format import format_udf_input, iter_udf_output_tables

        def identity(batch):
            assert "pandas" not in sys.modules
            try:
                import pandas
            except ModuleNotFoundError:
                return batch
            raise AssertionError("pandas must be unavailable in the worker")

        route, mode = sys.argv[1:]
        fmt = "numpy" if mode.startswith("numpy") else "pyarrow"
        zero_copy = mode != "numpy_writable"
        with vane.connect(config={"threads": 2}) as con:
            relation = con.sql("""
                SELECT x, [x, NULL::INTERVAL] AS items,
                       [x, NULL::INTERVAL]::INTERVAL[2] AS fixed_items,
                       {'value': x} AS record, map(['key'], [x]) AS mapping
                FROM (VALUES (INTERVAL '1 month 2 days 3 microseconds'),
                             (NULL), (INTERVAL '-2 months -3 days -4 microseconds')) t(x)
            """)
            expected = relation.to_arrow_table()
            if route == "public":
                result = relation.map_batches(
                    identity, schema=dict(zip(relation.columns, relation.types)),
                    batch_format=fmt, zero_copy_batch=zero_copy,
                    execution_backend="subprocess_task",
                ).to_arrow_table()
                assert result.equals(expected)
            else:
                schema = [{"name": name, "kind": "duckdb_type", "type": str(dtype)}
                          for name, dtype in zip(relation.columns, relation.types)]
                for table in (expected, expected.slice(1), expected.slice(0, 0),
                              pa.concat_tables([expected.slice(0, 1), expected.slice(1)])):
                    batch = identity(format_udf_input(table, fmt, zero_copy_batch=zero_copy))
                    result, = iter_udf_output_tables(batch, batch_format=fmt, output_schema=schema)
                    assert result.to_pydict() == table.to_pydict()
                    if len(table):
                        assert result.schema == table.schema
            assert "pandas" not in sys.modules
    ''')
    environment = {
        **os.environ,
        "VANE_RUNNER": "local-fast",
        "PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    completed = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script, route, mode],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


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
@pytest.mark.parametrize("container", ["plain", "list", "array", "struct", "map"])
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_times_keep_units_and_sub_microsecond_values(batch_format, writable, container, unit):
    dtype = pa.time32(unit) if unit in ("s", "ms") else pa.time64(unit)
    scale = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}[unit]
    leaf = pa.array([0, None, 86_400 * scale - 1], type=dtype)
    declaration = "TIME_NS" if unit == "ns" else "TIME"
    array, declaration = _wrap_leaf(leaf, declaration, container)
    _, result = _round_trip(array, declaration, batch_format, writable)
    assert result.equals(array)


@pytest.mark.parametrize("batch_format,writable", [("numpy", False), ("numpy", True), ("pandas", False)])
@pytest.mark.parametrize("container", ["plain", "list", "array", "struct", "map"])
@pytest.mark.parametrize("mode", ["sparse", "dense"])
@pytest.mark.parametrize("same_member_type", [False, True])
def test_union_identity_preserves_tags_and_tagged_nulls(batch_format, writable, container, mode, same_member_type):
    second_type = pa.int32() if same_member_type else pa.int64()
    second_declaration = "INTEGER" if same_member_type else "BIGINT"
    codes = pa.array([5, 9, 5, 9, 5, 9], type=pa.int8())
    if mode == "sparse":
        leaf = pa.UnionArray.from_sparse(
            codes,
            [
                pa.array([7, None, 42, None, None, None], type=pa.int32()),
                pa.array([None, 42, None, None, None, 8], type=second_type),
            ],
            field_names=["a", "b"],
            type_codes=[5, 9],
        )
    else:
        leaf = pa.UnionArray.from_dense(
            codes,
            pa.array([0, 0, 1, 1, 2, 2], type=pa.int32()),
            [pa.array([7, 42, None], type=pa.int32()), pa.array([42, None, 8], type=second_type)],
            field_names=["a", "b"],
            type_codes=[5, 9],
        )
    leaf = leaf.slice(1, 4)
    array, declaration = _wrap_leaf(leaf, f"UNION(a INTEGER, b {second_declaration})", container)
    _, result = _round_trip(array, declaration, batch_format, writable)
    assert result.equals(array)


def test_numpy_union_rows_keep_nested_members_and_reordered_rows():
    nested = pa.UnionArray.from_sparse(
        pa.array([0, 1], type=pa.int8()),
        [
            pa.array([[1, None], None], type=pa.list_(pa.int32())),
            pa.array([None, 45_296_123_456_789], type=pa.time64("ns")),
        ],
        field_names=["items", "clock"],
    )
    leaf = pa.UnionArray.from_sparse(
        pa.array([0, 1], type=pa.int8()),
        [nested, pa.array([None, 42], type=pa.int64())],
        field_names=["nested", "number"],
    )
    batch = format_udf_input(pa.table({"x": leaf}), "numpy", zero_copy_batch=False)
    batch["x"] = batch["x"][[1, 0, 1]]
    (result,) = iter_udf_output_tables(
        batch,
        batch_format="numpy",
        output_schema=[
            {
                "name": "x",
                "kind": "duckdb_type",
                "type": "UNION(nested UNION(items INTEGER[], clock TIME_NS), number BIGINT)",
            }
        ],
    )
    assert result["x"].combine_chunks().equals(leaf.take([1, 0, 1]))


@pytest.mark.parametrize("batch_format", ["numpy", "pandas"])
@pytest.mark.parametrize("empty", [False, True])
def test_union_output_retains_declared_type_without_values(batch_format, empty):
    dtype = pa.union([pa.field("a", pa.int32()), pa.field("b", pa.int64())], mode="sparse")
    array = pa.nulls(0 if empty else 2, type=dtype)
    _, result = _round_trip(array, "UNION(a INTEGER, b BIGINT)", batch_format)
    assert result.equals(array)


def test_union_output_rejects_mixing_tagged_and_untagged_values():
    leaf = pa.UnionArray.from_sparse(
        pa.array([0], type=pa.int8()),
        [pa.array([42], type=pa.int32()), pa.array([None], type=pa.int64())],
        field_names=["a", "b"],
    )
    values = np.empty(2, dtype=object)
    values[0] = leaf.slice(0, 1)
    values[1] = 42
    with pytest.raises(TypeError, match="one-row Arrow union arrays of one type"):
        list(
            iter_udf_output_tables(
                {"x": values},
                batch_format="numpy",
                output_schema=[{"name": "x", "kind": "duckdb_type", "type": "UNION(a INTEGER, b BIGINT)"}],
            )
        )


@pytest.mark.parametrize("batch_format,writable", [("numpy", False), ("numpy", True), ("pandas", False)])
@pytest.mark.parametrize("container", ["plain", "list", "array", "map"])
@pytest.mark.parametrize("float_type", [pa.float32(), pa.float64()])
def test_nested_tensor_siblings_keep_nan_separate_from_null(batch_format, writable, container, float_type):
    tensor_type = pa.fixed_shape_tensor(pa.float32(), [2])
    tensors = pa.ExtensionArray.from_storage(
        tensor_type, pa.array([[1, 2], None, [3, 4]], type=tensor_type.storage_type)
    )
    floating = pa.array([np.nan, None, 1.5], type=float_type, from_pandas=False)
    times = pa.array([45_296_123_456_789, None, 1], type=pa.time64("ns"))
    unions = pa.UnionArray.from_sparse(
        pa.array([0, 1, 1], type=pa.int8()),
        [pa.array([42, None, None], type=pa.int32()), pa.array([None, None, 42], type=pa.int64())],
        field_names=["a", "b"],
    )
    leaf = pa.StructArray.from_arrays(
        [tensors, floating, times, unions], names=["tensor", "floating", "clock", "tagged"]
    )
    float_declaration = "FLOAT" if float_type == pa.float32() else "DOUBLE"
    array, declaration = _wrap_leaf(
        leaf,
        f"STRUCT(tensor TENSOR(FLOAT, [2]), floating {float_declaration}, clock TIME_NS, tagged UNION(a INTEGER, b BIGINT))",
        container,
    )
    _, result = _round_trip(array, declaration, batch_format, writable)
    assert result.type == array.type
    if container in ("list", "array"):
        result = result.values
    elif container == "map":
        result = result.items
    assert result.field("tensor").equals(tensors)
    assert result.field("clock").equals(times)
    assert result.field("tagged").equals(unions)
    result_floating = result.field("floating")
    assert result_floating.is_valid().to_pylist() == [True, False, True]
    assert np.isnan(result_floating[0].as_py())
    assert result_floating[2].as_py() == 1.5


@pytest.mark.parametrize("container", ["plain", "list", "struct", "map"])
def test_pandas_object_float_output_keeps_nan_and_explicit_missing_values(container):
    pd = pytest.importorskip("pandas")
    rows = [np.nan, pd.NA, pd.NaT, np.datetime64("NaT"), np.ma.masked, None, 1.5]
    declaration = "DOUBLE"
    if container == "list":
        rows, declaration = [[value] for value in rows], "DOUBLE[]"
    elif container == "struct":
        rows, declaration = [{"value": value} for value in rows], "STRUCT(value DOUBLE)"
    elif container == "map":
        rows, declaration = [{"value": value} for value in rows], "MAP(VARCHAR, DOUBLE)"
    (result,) = iter_udf_output_tables(
        pd.DataFrame({"x": pd.Series(rows, dtype=object)}),
        batch_format="pandas",
        output_schema=[{"name": "x", "kind": "duckdb_type", "type": declaration}],
    )
    values = result["x"].combine_chunks()
    if container == "list":
        values = values.values
    elif container == "struct":
        values = values.field("value")
    elif container == "map":
        values = values.items
    assert values.is_valid().to_pylist() == [True, False, False, False, False, False, True]
    assert np.isnan(values[0].as_py())
    assert values[-1].as_py() == 1.5


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
        "clock": "'12:34:56.123456789'::TIME_NS",
        "clock_list": "['12:34:56.123456789'::TIME_NS, NULL]",
        "tagged": "union_value(a := 42)::UNION(a INTEGER, b BIGINT)",
        "tagged_list": "[union_value(a := 42)::UNION(a INTEGER, b BIGINT), union_value(b := 42::BIGINT)]",
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


@pytest.mark.parametrize("batch_format", ["pyarrow", "numpy", "pandas"])
@pytest.mark.parametrize("backend", ["subprocess_task", pytest.param("ray_actor", marks=pytest.mark.real_ray)])
def test_public_identity_keeps_nan_sibling_of_nested_tensor(request, monkeypatch, batch_format, backend):
    if backend == "ray_actor":
        request.getfixturevalue("ray_local")
    monkeypatch.setenv("VANE_RUNNER", "ray" if backend == "ray_actor" else "local-fast")

    class Identity:
        def __call__(self, batch):
            return batch

    with vane.connect(config={"threads": 2}) as con:
        relation = con.sql(
            "SELECT id, {'t': [1., 2.]::TENSOR(FLOAT, [2]), "
            "'f': CASE WHEN id = 0 THEN 'NaN'::DOUBLE ELSE NULL END} AS x FROM range(2) t(id)"
        )
        result = (
            relation.map_batches(
                Identity if backend == "ray_actor" else lambda batch: batch,
                schema=dict(zip(relation.columns, relation.types, strict=True)),
                batch_format=batch_format,
                execution_backend=backend,
                actor_number=1 if backend == "ray_actor" else None,
            )
            .project("id, x.f IS NULL, isnan(x.f)")
            .order("id")
            .fetchall()
        )
    assert result == [(0, False, True), (1, True, None)]


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
