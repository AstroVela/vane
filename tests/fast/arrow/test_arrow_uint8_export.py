# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

pa = pytest.importorskip("pyarrow")
np = pytest.importorskip("numpy")


@pytest.mark.parametrize("batch_size", [7, 2051])
def test_uint8_export_preserves_slices_and_appends(duckdb_cursor, batch_size):
    # The small batches slice inside engine chunks; 2051 also appends across
    # the usual 2048-row chunk boundary into an already populated Arrow buffer.
    values = [None if i % 19 == 0 else (i * 17) % 256 for i in range(4123)]
    table = pa.table({"value": pa.array(values, type=pa.uint8())}).slice(3, 4103)
    duckdb_cursor.register("byte_input", table)
    result = duckdb_cursor.execute("SELECT value FROM byte_input").to_arrow_reader(batch_size).read_all()
    assert result.schema == table.schema
    assert result.to_pydict() == table.to_pydict()


@pytest.mark.parametrize("batch_size", [7, 2051])
def test_uint8_export_preserves_filtered_and_constant_values(duckdb_cursor, batch_size):
    duckdb_cursor.execute("CREATE TABLE byte_input AS SELECT i, (i % 256)::UTINYINT AS value FROM range(4123) t(i)")
    result = (
        duckdb_cursor.execute(
            "SELECT value, 173::UTINYINT AS repeated, NULL::UTINYINT AS missing FROM byte_input WHERE i % 3 != 1"
        )
        .to_arrow_reader(batch_size)
        .read_all()
    )
    expected = [i % 256 for i in range(4123) if i % 3 != 1]
    assert result.to_pydict() == {
        "value": expected,
        "repeated": [173] * len(expected),
        "missing": [None] * len(expected),
    }
    assert all(field.type == pa.uint8() for field in result.schema)


def test_uint8_export_preserves_dictionary_selection(duckdb_cursor):
    dictionary = pa.array([255, None, 17, 0, 128], type=pa.uint8())
    indices = pa.array([4, 2, 0, None, 1, 3] * 701, type=pa.int16())
    values = pa.DictionaryArray.from_arrays(indices, dictionary).slice(5, 4099)
    duckdb_cursor.register("byte_input", pa.table({"value": values}))
    result = duckdb_cursor.execute("SELECT value FROM byte_input").to_arrow_reader(2051).read_all()
    assert result.column("value").type == pa.uint8()
    assert result.column("value").to_pylist() == values.to_pylist()


@pytest.mark.parametrize("batch_size", [7, 2051])
def test_uint8_tensor_export_preserves_values_shape_and_offsets(duckdb_cursor, batch_size):
    pixels = (np.arange(4123 * 30, dtype=np.uint32) * 17 % 256).astype(np.uint8).reshape(4123, 2, 5, 3)
    frames = pa.FixedShapeTensorArray.from_numpy_ndarray(pixels).slice(3, 4103)
    table = pa.table({"frame": frames})
    duckdb_cursor.register("frame_input", table)
    result = duckdb_cursor.execute("SELECT frame FROM frame_input").to_arrow_reader(batch_size).read_all()
    assert result.column("frame").type == frames.type
    np.testing.assert_array_equal(result.column("frame").combine_chunks().to_numpy_ndarray(), pixels[3:4106])


def test_uint8_fixed_array_export_preserves_parent_and_child_nulls(duckdb_cursor):
    values = [None if i % 11 == 0 else i % 256 for i in range(4123 * 6)]
    arrays = pa.FixedSizeListArray.from_arrays(
        pa.array(values, type=pa.uint8()),
        6,
        mask=pa.array([i % 13 == 0 for i in range(4123)]),
    ).slice(3, 4103)
    duckdb_cursor.register("frame_input", pa.table({"frame": arrays}))
    result = duckdb_cursor.execute("SELECT frame FROM frame_input").to_arrow_reader(2051).read_all()
    assert result.column("frame").type == arrays.type
    assert result.column("frame").to_pylist() == arrays.to_pylist()


def test_uint8_constant_array_export_preserves_values(duckdb_cursor):
    result = (
        duckdb_cursor.execute("SELECT [7, 255, NULL]::UTINYINT[3] AS frame FROM range(4103)")
        .to_arrow_reader(2051)
        .read_all()
    )
    assert result.column("frame").type == pa.list_(pa.uint8(), 3)
    assert result.column("frame").to_pylist() == [[7, 255, None]] * 4103
