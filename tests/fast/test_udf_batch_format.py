# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import sys
import types

import numpy as np
import pyarrow as pa
import pytest

from vane.execution._udf_runtime import UDFExecutor
from vane.execution.udf_batch_format import format_udf_input, iter_udf_output_tables


def _duckdb_field(name: str, type_name: str) -> dict[str, object]:
    return {"name": name, "kind": "duckdb_type", "type": type_name, "dtype": "", "shape": []}


def _tensor_field(name: str, dtype: str, shape: tuple[int, ...]) -> dict[str, object]:
    return {"name": name, "kind": "tensor", "type": "", "dtype": dtype, "shape": list(shape)}


def _runtime_payload(fn, *, batch_format: str, output_schema: list[dict[str, object]]) -> dict[str, object]:
    import cloudpickle

    return {
        "function_pickle": cloudpickle.dumps(fn),
        "call_mode": "map_batches",
        "execution_backend": "subprocess_task",
        "batch_format": batch_format,
        "output_schema": output_schema,
        "input_names": ["x", "embedding"],
    }


def _tensor_input_table(*, nullable: bool = False) -> pa.Table:
    dense = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    tensor_type = pa.fixed_shape_tensor(pa.float32(), (2, 2))
    if nullable:
        storage = pa.array(
            [dense[0].reshape(-1).tolist(), None, dense[2].reshape(-1).tolist()],
            type=tensor_type.storage_type,
        )
        tensors = pa.ExtensionArray.from_storage(tensor_type, storage)
    else:
        tensors = pa.FixedShapeTensorArray.from_numpy_ndarray(dense)
    return pa.table({"x": pa.array([1, 2, 3], type=pa.int64()), "embedding": tensors})


def test_numpy_batch_format_runtime_round_trip_preserves_tensor_shape():
    def transform(batch):
        assert set(batch) == {"x", "embedding"}
        assert all(isinstance(column, np.ndarray) for column in batch.values())
        assert all(column.flags.writeable for column in batch.values())
        assert batch["embedding"].shape == (3, 2, 2)
        return {"x": batch["x"] + 10, "embedding": batch["embedding"] * np.float32(2)}

    executor = UDFExecutor(
        _runtime_payload(
            transform,
            batch_format="numpy",
            output_schema=[_duckdb_field("x", "BIGINT"), _tensor_field("embedding", "FLOAT", (2, 2))],
        )
    )
    executor.submit(_tensor_input_table())
    result = executor.take_ready_result()

    assert result is not None
    assert result.column("x").to_pylist() == [11, 12, 13]
    tensor = result.column("embedding").combine_chunks()
    assert tensor.type == pa.fixed_shape_tensor(pa.float32(), (2, 2))
    np.testing.assert_array_equal(
        tensor.to_numpy_ndarray(),
        np.arange(12, dtype=np.float32).reshape(3, 2, 2) * np.float32(2),
    )


def test_numpy_batch_format_runtime_builds_declared_empty_output():
    def drop_batch(_batch):
        return None

    executor = UDFExecutor(
        _runtime_payload(
            drop_batch,
            batch_format="numpy",
            output_schema=[_duckdb_field("x", "BIGINT"), _tensor_field("embedding", "FLOAT", (2, 2))],
        )
    )
    executor.submit(_tensor_input_table())
    result = executor.take_ready_result()

    assert result is not None
    assert result.num_rows == 0
    assert result.schema == pa.schema(
        [
            pa.field("x", pa.int64()),
            pa.field("embedding", pa.fixed_shape_tensor(pa.float32(), (2, 2))),
        ]
    )


def test_numpy_batch_format_preserves_nullable_tensor_rows_as_object_array():
    batch = format_udf_input(_tensor_input_table(nullable=True), "numpy")

    assert batch["embedding"].dtype == object
    assert batch["embedding"][1] is None
    np.testing.assert_array_equal(batch["embedding"][0], np.arange(4, dtype=np.float32).reshape(2, 2))


def test_numpy_batch_format_round_trips_null_mask_separately_from_nan():
    table = pa.table(
        {
            "integer": pa.array([1, None, 3], type=pa.int64()),
            "floating": pa.array([1.0, np.nan, None], type=pa.float64()),
        }
    )

    batch = format_udf_input(table, "numpy")
    assert isinstance(batch["integer"], np.ma.MaskedArray)
    assert isinstance(batch["floating"], np.ma.MaskedArray)
    assert batch["integer"].dtype == np.dtype(np.int64)
    assert batch["integer"].mask.tolist() == [False, True, False]
    assert batch["floating"].mask.tolist() == [False, False, True]
    assert np.isnan(batch["floating"][1])

    output = list(
        iter_udf_output_tables(
            batch,
            batch_format="numpy",
            output_schema=[_duckdb_field("integer", "BIGINT"), _duckdb_field("floating", "DOUBLE")],
        )
    )[0]
    assert output.column("integer").to_pylist() == [1, None, 3]
    floating = output.column("floating")
    assert floating.null_count == 1
    assert np.isnan(floating[1].as_py())


def test_numpy_batch_output_requires_ndarrays_and_declared_columns():
    schema = [_duckdb_field("x", "BIGINT")]

    with pytest.raises(TypeError, match="must be numpy.ndarray"):
        list(iter_udf_output_tables({"x": [1, 2]}, batch_format="numpy", output_schema=schema))
    with pytest.raises(ValueError, match="missing=.*x.*extra=.*y"):
        list(
            iter_udf_output_tables(
                {"y": np.array([1, 2], dtype=np.int64)},
                batch_format="numpy",
                output_schema=schema,
            )
        )


def test_non_tensor_output_type_is_left_for_native_schema_validation():
    identifier = "550e8400-e29b-41d4-a716-446655440000"
    output = list(
        iter_udf_output_tables(
            {"identifier": np.array([identifier], dtype=object)},
            batch_format="numpy",
            output_schema=[_duckdb_field("identifier", "UUID")],
        )
    )

    assert output[0].schema.field("identifier").type == pa.string()
    assert output[0].column("identifier").to_pylist() == [identifier]


def test_numpy_output_iterator_yields_multiple_batches():
    def batches():
        yield {"x": np.array([1, 2], dtype=np.int64)}
        yield {"x": np.array([3], dtype=np.int64)}

    output = list(
        iter_udf_output_tables(
            batches(),
            batch_format="numpy",
            output_schema=[_duckdb_field("x", "BIGINT")],
        )
    )

    assert [table.column("x").to_pylist() for table in output] == [[1, 2], [3]]


def test_numpy_tensor_output_safely_casts_to_declared_dtype():
    output = list(
        iter_udf_output_tables(
            {"embedding": np.arange(8, dtype=np.float64).reshape(2, 2, 2)},
            batch_format="numpy",
            output_schema=[_tensor_field("embedding", "FLOAT", (2, 2))],
        )
    )[0]

    tensor = output.column("embedding").combine_chunks()
    assert tensor.type == pa.fixed_shape_tensor(pa.float32(), (2, 2))
    np.testing.assert_array_equal(tensor.to_numpy_ndarray(), np.arange(8, dtype=np.float32).reshape(2, 2, 2))


def test_numpy_tensor_output_does_not_confuse_date_with_timestamp_dtype():
    values = np.array(
        [["2026-01-01", "2026-01-02"], ["2026-01-03", "2026-01-04"]],
        dtype="datetime64[ms]",
    )

    output = list(
        iter_udf_output_tables(
            {"dates": values},
            batch_format="numpy",
            output_schema=[_tensor_field("dates", "DATE", (2,))],
        )
    )[0]

    tensor = output.column("dates").combine_chunks()
    assert tensor.type == pa.fixed_shape_tensor(pa.date32(), (2,))
    assert tensor.storage.to_pylist() == [
        [datetime.date(2026, 1, 1), datetime.date(2026, 1, 2)],
        [datetime.date(2026, 1, 3), datetime.date(2026, 1, 4)],
    ]


def test_numpy_tensor_output_converts_non_native_byte_order():
    values = np.arange(8, dtype=np.float32).astype(">f4").reshape(2, 2, 2)

    output = list(
        iter_udf_output_tables(
            {"embedding": values},
            batch_format="numpy",
            output_schema=[_tensor_field("embedding", "FLOAT", (2, 2))],
        )
    )[0]

    tensor = output.column("embedding").combine_chunks()
    assert tensor.type == pa.fixed_shape_tensor(pa.float32(), (2, 2))
    np.testing.assert_array_equal(tensor.to_numpy_ndarray(), np.arange(8, dtype=np.float32).reshape(2, 2, 2))


def test_numpy_object_tensor_output_safely_casts_to_declared_dtype():
    output = list(
        iter_udf_output_tables(
            {"embedding": np.arange(8).reshape(2, 2, 2).astype(object)},
            batch_format="numpy",
            output_schema=[_tensor_field("embedding", "FLOAT", (2, 2))],
        )
    )[0]

    tensor = output.column("embedding").combine_chunks()
    assert tensor.type == pa.fixed_shape_tensor(pa.float32(), (2, 2))
    np.testing.assert_array_equal(tensor.to_numpy_ndarray(), np.arange(8, dtype=np.float32).reshape(2, 2, 2))


@pytest.mark.parametrize("masked", [False, True])
def test_numpy_empty_tensor_output_preserves_declared_type(masked):
    values = np.empty((0, 2, 2), dtype=np.float32)
    if masked:
        values = np.ma.MaskedArray(values, mask=np.empty((0, 2, 2), dtype=bool))

    output = list(
        iter_udf_output_tables(
            {"embedding": values},
            batch_format="numpy",
            output_schema=[_tensor_field("embedding", "FLOAT", (2, 2))],
        )
    )[0]

    assert output.num_rows == 0
    assert output.schema.field("embedding").type == pa.fixed_shape_tensor(pa.float32(), (2, 2))


def test_numpy_masked_tensor_output_preserves_whole_row_nulls():
    values = np.ma.MaskedArray(
        np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        mask=np.array([np.zeros((2, 2), dtype=bool), np.ones((2, 2), dtype=bool)]),
    )

    output = list(
        iter_udf_output_tables(
            {"embedding": values},
            batch_format="numpy",
            output_schema=[_tensor_field("embedding", "FLOAT", (2, 2))],
        )
    )[0]

    tensor = output.column("embedding").combine_chunks()
    assert tensor.null_count == 1
    np.testing.assert_array_equal(tensor.to_numpy_ndarray()[0], np.arange(4, dtype=np.float32).reshape(2, 2))


def test_numpy_masked_tensor_output_rejects_partial_row_masks():
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[1, 0, 0] = True
    values = np.ma.MaskedArray(np.arange(8, dtype=np.float32).reshape(2, 2, 2), mask=mask)

    with pytest.raises(ValueError, match="row 1 has a partial mask"):
        list(
            iter_udf_output_tables(
                {"embedding": values},
                batch_format="numpy",
                output_schema=[_tensor_field("embedding", "FLOAT", (2, 2))],
            )
        )


def test_numpy_object_tensor_output_rejects_partial_row_masks():
    values = np.empty(1, dtype=object)
    values[0] = np.ma.MaskedArray(
        np.arange(4, dtype=np.float32).reshape(2, 2),
        mask=[[True, False], [False, False]],
    )

    with pytest.raises(ValueError, match="row 0 has a partial mask"):
        list(
            iter_udf_output_tables(
                {"embedding": values},
                batch_format="numpy",
                output_schema=[_tensor_field("embedding", "FLOAT", (2, 2))],
            )
        )


def test_batch_output_does_not_fall_back_across_formats():
    schema = [_duckdb_field("x", "BIGINT")]

    with pytest.raises(TypeError, match=r"batch_format='numpy'.*dict\[str, numpy.ndarray\]"):
        list(iter_udf_output_tables(pa.table({"x": [1]}), batch_format="numpy", output_schema=schema))
    with pytest.raises(TypeError, match="batch_format='pyarrow'.*pyarrow.Table"):
        list(
            iter_udf_output_tables(
                {"x": np.array([1], dtype=np.int64)},
                batch_format="pyarrow",
                output_schema=schema,
            )
        )


def test_batch_format_rejects_duplicate_input_column_names():
    table = pa.Table.from_arrays([pa.array([1]), pa.array([2])], names=["x", "x"])

    with pytest.raises(ValueError, match="requires unique column names"):
        format_udf_input(table, "numpy")


def test_pandas_batch_format_round_trip_preserves_tensor_cells():
    pandas = pytest.importorskip("pandas")
    table = _tensor_input_table(nullable=True)

    frame = format_udf_input(table, "pandas")
    assert isinstance(frame, pandas.DataFrame)
    assert frame.loc[1, "embedding"] is None
    assert frame.loc[0, "embedding"].shape == (2, 2)

    frame["x"] += 5
    output = list(
        iter_udf_output_tables(
            frame,
            batch_format="pandas",
            output_schema=[_duckdb_field("x", "BIGINT"), _tensor_field("embedding", "FLOAT", (2, 2))],
        )
    )
    assert len(output) == 1
    assert output[0].column("x").to_pylist() == [6, 7, 8]
    assert output[0].column("embedding").null_count == 1
    np.testing.assert_array_equal(
        output[0].column("embedding").combine_chunks().to_numpy_ndarray()[0],
        np.arange(4, dtype=np.float32).reshape(2, 2),
    )


def test_pandas_batch_format_preserves_nulls_separately_from_nan():
    pytest.importorskip("pandas")
    table = pa.table(
        {
            "integer": pa.array([1, None, 3], type=pa.int64()),
            "floating": pa.array([1.0, np.nan, None], type=pa.float64()),
        }
    )

    frame = format_udf_input(table, "pandas")
    assert str(frame["integer"].dtype) == "int64[pyarrow]"
    assert frame["integer"].isna().tolist() == [False, True, False]
    assert frame["floating"].isna().tolist() == [False, False, True]
    assert np.isnan(frame.loc[1, "floating"])

    output = list(
        iter_udf_output_tables(
            frame,
            batch_format="pandas",
            output_schema=[_duckdb_field("integer", "BIGINT"), _duckdb_field("floating", "DOUBLE")],
        )
    )[0]
    assert output.column("integer").to_pylist() == [1, None, 3]
    assert output.column("floating").null_count == 1
    assert np.isnan(output.column("floating")[1].as_py())


def test_cudf_batch_format_uses_arrow_boundary(monkeypatch):
    class FakeDataFrame:
        def __init__(self, table):
            self.table = table

        @classmethod
        def from_arrow(cls, table):
            return cls(table)

        def to_arrow(self, *, preserve_index):
            assert preserve_index is False
            return self.table

    fake_cudf = types.ModuleType("cudf")
    fake_cudf.DataFrame = FakeDataFrame
    monkeypatch.setitem(sys.modules, "cudf", fake_cudf)

    frame = format_udf_input(pa.table({"x": [1, 2]}), "cudf")
    assert isinstance(frame, FakeDataFrame)
    output = list(
        iter_udf_output_tables(
            frame,
            batch_format="cudf",
            output_schema=[_duckdb_field("x", "BIGINT")],
        )
    )
    assert output[0].to_pydict() == {"x": [1, 2]}


@pytest.mark.gpu
def test_cudf_batch_format_round_trip():
    cudf = pytest.importorskip("cudf")
    table = pa.table({"x": pa.array([1, 2, 3], type=pa.int64())})

    frame = format_udf_input(table, "cudf")
    assert isinstance(frame, cudf.DataFrame)
    frame["x"] += 4
    output = list(
        iter_udf_output_tables(
            frame,
            batch_format="cudf",
            output_schema=[_duckdb_field("x", "BIGINT")],
        )
    )
    assert output[0].column("x").to_pylist() == [5, 6, 7]


def test_unknown_batch_format_is_rejected():
    with pytest.raises(ValueError, match="batch_format must be one of"):
        format_udf_input(pa.table({"x": [1]}), "polars")
