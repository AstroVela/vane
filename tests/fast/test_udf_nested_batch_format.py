# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
from collections import UserList

import numpy as np
import pyarrow as pa
import pytest

import vane
from vane.execution.udf_batch_format import format_udf_input, iter_udf_output_tables


def _encode(values, declaration, batch_format):
    schema = [{"name": "x", "kind": "duckdb_type", "type": declaration}]
    if batch_format == "numpy":
        column = np.empty(len(values), dtype=object)
        for index, value in enumerate(values):
            column[index] = value
        batch = {"x": column}
    else:
        pd = pytest.importorskip("pandas")
        batch = pd.DataFrame({"x": values})
    (result,) = iter_udf_output_tables(batch, batch_format=batch_format, output_schema=schema)
    return result.column("x").combine_chunks()


@pytest.mark.parametrize("batch_format", ["numpy", "pandas"])
@pytest.mark.parametrize("container", ["map", "list", "array", "struct"])
def test_object_map_output_preserves_each_rows_keys(batch_format, container):
    maps = [{"a": 1}, {"b": 2}, {}, None, {"c": None}]
    rows = maps
    expected = [[("a", 1)], [("b", 2)], [], None, [("c", None)]]
    declaration = "MAP(VARCHAR, BIGINT)"
    if container in ("list", "array"):
        rows = [[row, None] for row in maps]
        expected = [[row, None] for row in expected]
        declaration += "[]" if container == "list" else "[2]"
    elif container == "struct":
        rows = [{"ITEMS": row} for row in maps]
        expected = [{"items": row} for row in expected]
        declaration = "STRUCT(items MAP(VARCHAR, BIGINT))"
    result = _encode(rows, declaration, batch_format)
    assert result.to_pylist() == expected


@pytest.mark.parametrize("container", ["map", "list", "struct", "struct_list"])
def test_pandas_object_containers_preserve_missing_values(container):
    pd = pytest.importorskip("pandas")
    rows = [{"a": pd.NA}, pd.NA, pd.NaT, np.nan, None, {"b": 2}]
    expected = [[("a", None)], None, None, None, None, [("b", 2)]]
    declaration = "MAP(VARCHAR, BIGINT)"
    if container == "list":
        rows = [[row] for row in rows]
        expected = [[row] for row in expected]
        declaration += "[]"
    elif container in ("struct", "struct_list"):
        if container == "struct_list":
            rows = [[row] if index % 2 == 0 else pd.NA for index, row in enumerate(rows)]
            expected = [[row] if index % 2 == 0 else None for index, row in enumerate(expected)]
            declaration += "[]"
        rows = [{"items": row} for row in rows]
        expected = [{"items": row} for row in expected]
        declaration = f"STRUCT(items {declaration})"
    assert _encode(rows, declaration, "pandas").to_pylist() == expected


@pytest.mark.parametrize("batch_format", ["numpy", "pandas"])
@pytest.mark.parametrize("extra_field", [False, True])
def test_nested_struct_fields_are_checked_and_matched_without_case(batch_format, extra_field):
    record = {"VALUE": 7}
    if extra_field:
        record["extra"] = 9
    rows = [UserList([record])]
    if extra_field:
        with pytest.raises(vane.InvalidInputException, match="exactly the declared fields"):
            _encode(rows, "STRUCT(value BIGINT)[]", batch_format)
    else:
        assert _encode(rows, "STRUCT(value BIGINT)[]", batch_format).to_pylist() == [[{"value": 7}]]


@pytest.mark.parametrize("batch_format,writable", [("numpy", False), ("numpy", True), ("pandas", False)])
@pytest.mark.parametrize("leaf", ["unsigned", "date"])
@pytest.mark.parametrize("container", ["list", "array", "struct", "map", "nested"])
@pytest.mark.parametrize("empty", [False, True])
def test_nested_identity_preserves_leaf_types_and_nulls(batch_format, writable, leaf, container, empty):
    if leaf == "unsigned":
        dtype, declaration = pa.uint64(), "UBIGINT"
        first, last = 2**63 + 1, 2**64 - 1
    else:
        dtype, declaration = pa.date32(), "DATE"
        first, last = datetime.date(2026, 1, 1), datetime.date(2026, 1, 2)
    if container in ("list", "array"):
        dtype = pa.list_(dtype, 2) if container == "array" else pa.list_(dtype)
        declaration += "[2]" if container == "array" else "[]"
        rows = [[first, last], [first, None], [None, None], None]
        if container == "list":
            rows.append([])
    elif container == "struct":
        dtype = pa.struct([("value", dtype), ("missing", dtype)])
        declaration = f"STRUCT(value {declaration}, missing {declaration})"
        rows = [{"value": first, "missing": None}, {"value": last, "missing": first}, None]
    elif container == "map":
        dtype = pa.map_(pa.string(), dtype)
        declaration = f"MAP(VARCHAR, {declaration})"
        rows = [[("a", first), ("b", None)], [("c", last)], [], None]
    else:
        dtype = pa.struct([("items", pa.list_(pa.map_(pa.uint64(), pa.list_(dtype, 2))))])
        declaration = f"STRUCT(items MAP(UBIGINT, {declaration}[2])[])"
        rows = [
            {"items": [[(2**63 + 1, [first, None]), (2**64 - 1, [last, first])], None, []]},
            {"items": None},
            None,
        ]
    array = pa.array([] if empty else [None, *rows, None], type=dtype)
    if not empty:
        array = array.slice(1, len(rows))
    batch = format_udf_input(pa.table({"x": array}), batch_format, zero_copy_batch=not writable)
    schema = [{"name": "x", "kind": "duckdb_type", "type": declaration}]
    (result,) = iter_udf_output_tables(batch, batch_format=batch_format, output_schema=schema)
    assert result.column("x").combine_chunks().equals(array)


def test_numpy_date_lists_keep_dates_outside_python_datetime_range():
    values = np.array(["10000-01-01", "-10000-01-01"], dtype="datetime64[D]")
    result = _encode([values], "DATE[]", "numpy")
    assert result.type == pa.list_(pa.date32())
    assert result.values.equals(pa.array(values))


@pytest.mark.parametrize("batch_format", ["numpy", "pandas"])
@pytest.mark.parametrize("container", ["list", "array", "struct", "map"])
def test_nested_output_leaves_keep_duckdb_cast_semantics(monkeypatch, batch_format, container):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    if container == "list":
        rows, declaration = [[1.75, None], [2.1, 3.8]], "BIGINT[]"
        expected = [([2, None],), ([2, 4],)]
    elif container == "array":
        rows, declaration = [[1.75, None], [2.1, 3.8]], "BIGINT[2]"
        expected = [((2, None),), ((2, 4),)]
    elif container == "struct":
        rows, declaration = [{"VALUE": 1.75}, {"VALUE": 2.1}], "STRUCT(value BIGINT)"
        expected = [({"value": 2},), ({"value": 2},)]
    else:
        rows, declaration = [{"a": 1.75}, {"b": 2.1}], "MAP(VARCHAR, BIGINT)"
        expected = [({"a": 2},), ({"b": 2},)]

    def output(batch):
        if batch_format == "pandas":
            import pandas as pd

            return pd.DataFrame({"x": rows})
        values = np.empty(len(rows), dtype=object)
        for index, row in enumerate(rows):
            values[index] = row
        return {"x": values}

    with vane.connect(config={"threads": 2}) as con:
        actual = (
            con.sql("SELECT 1")
            .map_batches(
                output,
                schema={"x": vane.type(declaration)},
                batch_format=batch_format,
                execution_backend="subprocess_task",
            )
            .fetchall()
        )
    assert actual == expected


@pytest.mark.parametrize("batch_format", ["numpy", "pandas"])
def test_pandas_and_numpy_map_blob_values_keep_duckdb_cast_semantics(monkeypatch, batch_format):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    def output(batch):
        rows = [{"a": b"\xff"}, {"b": b"\x00"}]
        if batch_format == "pandas":
            import pandas as pd

            return pd.DataFrame({"x": rows})
        values = np.empty(2, dtype=object)
        values[:] = rows
        return {"x": values}

    with vane.connect(config={"threads": 2}) as con:
        actual = (
            con.sql("SELECT 1")
            .map_batches(
                output,
                schema={"x": vane.type("MAP(VARCHAR, VARCHAR)")},
                batch_format=batch_format,
                execution_backend="subprocess_task",
            )
            .fetchall()
        )
    assert actual == [({"a": r"\xFF"},), ({"b": r"\x00"},)]
