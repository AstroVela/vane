# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc

import numpy as np
import pyarrow as pa
import pytest

import vane
from vane._image import _MODE_CHANNELS, _MODE_DTYPES, _image_native_storage, image_arrow_type
from vane.execution.udf_batch_format import format_udf_input, iter_udf_output_tables


def _format(column, *, zero_copy_batch=True):
    return format_udf_input(pa.table({"value": column}), "numpy", zero_copy_batch=zero_copy_batch)["value"]


def _image_array(dtype, rows):
    arrow_type = image_arrow_type(dtype)
    storage = pa.array(
        [None if row is None else _image_native_storage(row, dtype) for row in rows], type=arrow_type.storage_type
    )
    return pa.ExtensionArray.from_storage(arrow_type, storage)


def _image_schema(dtype):
    return [{"name": "image", "kind": "duckdb_type", "type": str(dtype), "dtype": "", "shape": []}]


@pytest.mark.parametrize("zero_copy_batch", [True, False])
def test_numpy_primitive_slice_shares_only_readonly_input(zero_copy_batch):
    original = np.arange(8, dtype=np.int64)
    result = _format(pa.chunked_array([pa.array(original).slice(2, 3)]), zero_copy_batch=zero_copy_batch)

    np.testing.assert_array_equal(result, [2, 3, 4])
    assert np.shares_memory(result, original) == zero_copy_batch
    assert result.flags.writeable != zero_copy_batch
    if zero_copy_batch:
        with pytest.raises(ValueError, match="read-only"):
            result[0] = 99
    else:
        result[0] = 99
        assert original[2] == 2


@pytest.mark.parametrize("zero_copy_batch", [True, False])
@pytest.mark.parametrize("parent_offset", [0, 1])
@pytest.mark.parametrize("child_offset", [0, 1])
@pytest.mark.parametrize("multiple_chunks", [False, True])
def test_numpy_tensor_offsets_preserve_values_and_storage(
    zero_copy_batch, parent_offset, child_offset, multiple_chunks
):
    pixels = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    flat = np.concatenate([np.full(child_offset, -1, dtype=np.float32), pixels.reshape(-1)])
    values = pa.array(flat).slice(child_offset)
    storage = pa.FixedSizeListArray.from_arrays(values, 6)
    array = pa.ExtensionArray.from_storage(pa.fixed_shape_tensor(pa.float32(), (2, 3)), storage)
    if multiple_chunks:
        column = pa.chunked_array([array.slice(parent_offset, 1), array.slice(parent_offset + 1, 1)])
    else:
        column = pa.chunked_array([array.slice(parent_offset, 2)])
    result = _format(column, zero_copy_batch=zero_copy_batch)

    np.testing.assert_array_equal(result, pixels[parent_offset : parent_offset + 2])
    assert result.dtype == np.float32 and result.flags.c_contiguous
    assert result.flags.writeable != zero_copy_batch
    assert np.shares_memory(result, flat) == (zero_copy_batch and not multiple_chunks)
    if not zero_copy_batch:
        result[:] = 99
        np.testing.assert_array_equal(values.to_numpy(), pixels.reshape(-1))


@pytest.mark.parametrize("child_offset", [0, 1])
def test_numpy_tensor_permutation_and_lifetime(child_offset):
    expected = np.arange(24, dtype=np.float32).reshape(3, 2, 4).transpose(0, 2, 1)[1:]
    array = pa.FixedShapeTensorArray.from_numpy_ndarray(expected)
    if child_offset:
        values = pa.concat_arrays([pa.array([-1], type=pa.float32()), array.storage.values]).slice(1)
        storage = pa.FixedSizeListArray.from_arrays(values, array.storage.type.list_size)
        array = pa.ExtensionArray.from_storage(array.type, storage)
    result = _format(array)
    del array
    gc.collect()

    np.testing.assert_array_equal(result, expected)
    assert not result.flags.writeable


@pytest.mark.parametrize("num_chunks", [0, 1])
def test_numpy_empty_tensor_keeps_declared_shape(num_chunks):
    dtype = pa.fixed_shape_tensor(pa.float32(), (2, 3))
    array = pa.ExtensionArray.from_storage(dtype, pa.array([], type=dtype.storage_type))
    result = _format(pa.chunked_array([array] * num_chunks, type=dtype))
    assert result.shape == (0, 2, 3)
    assert result.dtype == np.float32 and not result.flags.writeable


@pytest.mark.parametrize("zero_copy_batch", [True, False])
def test_numpy_nullable_buffers_preserve_masks_and_mutation_isolation(zero_copy_batch):
    source = pa.table({"integer": pa.array([1, None, 3]), "nested": pa.array([[1, 2], None, [3]])})
    batch = format_udf_input(source, "numpy", zero_copy_batch=zero_copy_batch)
    integer, nested = batch["integer"], batch["nested"]

    assert isinstance(integer, np.ma.MaskedArray)
    assert integer.mask.tolist() == [False, True, False]
    assert integer.flags.writeable != zero_copy_batch
    assert integer.mask.flags.writeable != zero_copy_batch
    assert nested[0].flags.writeable != zero_copy_batch
    if zero_copy_batch:
        with pytest.raises(ValueError, match="read-only"):
            nested[0][0] = 99
        with pytest.raises(ValueError, match="read-only"):
            integer.mask[0] = True
    else:
        integer[0] = 99
        integer.mask[1] = False
        nested[0][0] = 99
    assert source.to_pydict() == {"integer": [1, None, 3], "nested": [[1, 2], None, [3]]}


@pytest.mark.parametrize("zero_copy_batch", [True, False])
def test_numpy_nullable_tensor_rows_follow_copy_policy(zero_copy_batch):
    tensor_type = pa.fixed_shape_tensor(pa.int64(), (1, 2))
    array = pa.ExtensionArray.from_storage(tensor_type, pa.array([[1, 2], None, [3, 4]], tensor_type.storage_type))
    result = _format(array, zero_copy_batch=zero_copy_batch)

    assert result.dtype == object and result[1] is None
    np.testing.assert_array_equal(result[0], [[1, 2]])
    assert result.flags.writeable != zero_copy_batch
    assert result[0].flags.writeable != zero_copy_batch
    if not zero_copy_batch:
        result[0][:] = 99
        assert array.storage[0].as_py() == [1, 2]


@pytest.mark.parametrize("form", ["fixed", "variable", "generic"])
@pytest.mark.parametrize("mode", list(_MODE_DTYPES))
@pytest.mark.parametrize("zero_copy_batch", [True, False])
def test_numpy_image_modes_and_copy_policy(form, mode, zero_copy_batch):
    dtype = (
        vane.image_type(mode, 2, 3)
        if form == "fixed"
        else vane.image_type(mode)
        if form == "variable"
        else vane.image_type()
    )
    channels = _MODE_CHANNELS[mode]
    pixels = np.arange(3 * 2 * 3 * channels, dtype=_MODE_DTYPES[mode]).reshape(3, 2, 3, channels)
    array = _image_array(dtype, pixels).slice(1)
    result = _format(array, zero_copy_batch=zero_copy_batch)

    if form == "fixed":
        assert result.shape == (2, 2, 3, channels)
        source_pixels = array.storage.values.to_numpy()
        assert np.shares_memory(result, source_pixels) == zero_copy_batch
    else:
        assert result.shape == (2,) and result.dtype == object
        source_pixels = array.storage.field("data")[0].values.to_numpy()
        assert np.shares_memory(result[0], source_pixels) == (
            zero_copy_batch and (form == "variable" or _MODE_DTYPES[mode] == np.float32)
        )
    for row, expected in zip(result, pixels[1:], strict=True):
        np.testing.assert_array_equal(row, expected)
        assert row.dtype == expected.dtype and row.flags.c_contiguous
        assert row.flags.writeable != zero_copy_batch
    if not zero_copy_batch:
        result[0][:] = 0
        np.testing.assert_array_equal(_format(array)[0], pixels[1])


@pytest.mark.parametrize("parent_offset", [0, 1])
@pytest.mark.parametrize("child_offset", [0, 1])
@pytest.mark.parametrize("multiple_chunks", [False, True])
def test_numpy_fixed_image_offsets_and_chunk_consolidation(parent_offset, child_offset, multiple_chunks):
    pixels = np.arange(72, dtype=np.uint8).reshape(4, 2, 3, 3)
    flat = np.concatenate([np.full(child_offset, 255, dtype=np.uint8), pixels.reshape(-1)])
    values = pa.array(flat).slice(child_offset)
    storage = pa.FixedSizeListArray.from_arrays(values, 18)
    array = pa.ExtensionArray.from_storage(image_arrow_type(vane.image_type("RGB", 2, 3)), storage)
    chunks = (
        [array.slice(parent_offset, 1), array.slice(parent_offset + 1, 1)]
        if multiple_chunks
        else [array.slice(parent_offset, 2)]
    )
    result = _format(pa.chunked_array(chunks))

    np.testing.assert_array_equal(result, pixels[parent_offset : parent_offset + 2])
    assert np.shares_memory(result, flat) == (not multiple_chunks)
    assert not result.flags.writeable


@pytest.mark.parametrize("form", ["fixed", "variable", "generic"])
@pytest.mark.parametrize("zero_copy_batch", [True, False])
def test_numpy_image_null_rows_round_trip(form, zero_copy_batch):
    dtype = (
        vane.image_type("RGB", 2, 3)
        if form == "fixed"
        else vane.image_type("RGB")
        if form == "variable"
        else vane.image_type()
    )
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    array = _image_array(dtype, [pixels, None, pixels + 1])
    result = _format(pa.chunked_array([array.slice(0, 1), array.slice(1)]), zero_copy_batch=zero_copy_batch)
    assert result[1] is None
    assert result[0].flags.writeable != zero_copy_batch
    (output,) = iter_udf_output_tables({"image": result}, batch_format="numpy", output_schema=_image_schema(dtype))

    assert output.column("image").type == array.type
    assert output.column("image").combine_chunks().equals(array)


def test_numpy_generic_image_preserves_mixed_shapes_modes_and_nulls():
    dtype = vane.image_type()
    rows = [
        np.arange(6, dtype=np.uint8).reshape(1, 2, 3),
        None,
        np.arange(8, dtype=np.uint16).reshape(2, 1, 4),
        np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 4,
    ]
    array = _image_array(dtype, rows)
    result = _format(array)
    del array
    gc.collect()

    assert result.shape == (4,) and result[1] is None
    for actual, expected in zip(result, rows, strict=True):
        if expected is not None:
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == expected.dtype and not actual.flags.writeable


@pytest.mark.parametrize("dtype", [vane.image_type(), vane.image_type("RGB"), vane.image_type("RGB", 2, 3)])
def test_numpy_empty_image_chunks_preserve_declared_shape(dtype):
    column = pa.chunked_array([], type=image_arrow_type(dtype))
    result = _format(column)
    expected_shape = (0, 2, 3, 3) if dtype.is_fixed_shape_image() else (0,)
    assert result.shape == expected_shape
    assert result.dtype == (np.uint8 if dtype.is_fixed_shape_image() else object)
    assert not result.flags.writeable
    (output,) = iter_udf_output_tables({"image": result}, batch_format="numpy", output_schema=_image_schema(dtype))
    assert output.num_rows == 0 and output.column("image").type == column.type


@pytest.mark.parametrize(
    "row,message",
    [
        ({"data": [1, None, 3], "channel": 3, "height": 1, "width": 1, "mode": 3}, "NULL pixels"),
        ({"data": [1, 2, 3], "channel": 4, "height": 1, "width": 1, "mode": 3}, "channel count"),
        ({"data": [1, 2, 3], "channel": 3, "height": 2, "width": 1, "mode": 3}, "pixel values"),
        ({"data": [1, 2, 3], "channel": 3, "height": None, "width": 1, "mode": 3}, "NULL fields"),
        ({"data": [1.5, 2, 3], "channel": 3, "height": 1, "width": 1, "mode": 3}, "representable"),
    ],
)
def test_numpy_image_rejects_invalid_storage(row, message):
    arrow_type = image_arrow_type(vane.image_type())
    array = pa.ExtensionArray.from_storage(arrow_type, pa.array([row], type=arrow_type.storage_type))
    with pytest.raises(vane.InvalidInputException, match=message):
        _format(array)


@pytest.mark.parametrize(
    "value,message",
    [
        (np.nan, "HWC ndarray"),
        (np.zeros((2, 3), dtype=np.uint8), "HWC pixels"),
        (np.zeros((2, 3, 3), dtype=np.uint16), "mode"),
        (np.zeros((1, 3, 3), dtype=np.uint8), "shape"),
        (np.ma.array(np.zeros((2, 3, 3), dtype=np.uint8), mask=True), "HWC ndarray"),
    ],
)
def test_numpy_image_output_rejects_invalid_rows(value, message):
    rows = np.empty(1, dtype=object)
    rows[0] = value
    with pytest.raises(vane.InvalidInputException, match=message):
        list(
            iter_udf_output_tables(
                {"image": rows}, batch_format="numpy", output_schema=_image_schema(vane.image_type("RGB", 2, 3))
            )
        )


def test_numpy_fixed_image_rejects_null_pixels():
    arrow_type = image_arrow_type(vane.image_type("L", 1, 2))
    array = pa.ExtensionArray.from_storage(arrow_type, pa.array([[1, None]], type=arrow_type.storage_type))
    with pytest.raises(vane.InvalidInputException, match="NULL pixels"):
        _format(array)


@pytest.mark.parametrize("fixed", [True, False])
def test_pandas_image_cells_round_trip(fixed):
    pd = pytest.importorskip("pandas")
    dtype = vane.image_type("RGB", 2, 3) if fixed else vane.image_type()
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    array = _image_array(dtype, [pixels, None])
    frame = format_udf_input(pa.table({"image": array}), "pandas")

    assert isinstance(frame, pd.DataFrame)
    np.testing.assert_array_equal(frame["image"].iloc[0], pixels)
    assert frame["image"].iloc[1] is None
    assert frame["image"].iloc[0].flags.writeable
    (output,) = iter_udf_output_tables(frame, batch_format="pandas", output_schema=_image_schema(dtype))
    assert output.column("image").combine_chunks().equals(array)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_zero_copy_batch_requires_bool(value):
    with pytest.raises(TypeError, match="zero_copy_batch must be a bool"):
        format_udf_input(pa.table({"x": [1]}), "numpy", zero_copy_batch=value)
    with vane.connect() as con, pytest.raises(vane.InvalidInputException, match="zero_copy_batch must be a bool"):
        con.sql("select 1 as x").map_batches(
            lambda batch: batch, schema={"x": vane.sqltypes.INTEGER}, batch_format="numpy", zero_copy_batch=value
        )


@pytest.mark.parametrize("batch_format", ["pyarrow", "pandas", "cudf"])
def test_writable_batch_requires_numpy_format(batch_format):
    with pytest.raises(ValueError, match="requires batch_format='numpy'"):
        format_udf_input(pa.table({"x": [1]}), batch_format, zero_copy_batch=False)
    with vane.connect() as con, pytest.raises(vane.InvalidInputException, match="requires batch_format='numpy'"):
        con.sql("select 1 as x").map_batches(
            lambda batch: batch,
            schema={"x": vane.sqltypes.INTEGER},
            batch_format=batch_format,
            zero_copy_batch=False,
        )


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("zero_copy_batch", [True, False])
@pytest.mark.parametrize("backend", ["subprocess_actor", pytest.param("ray_actor", marks=pytest.mark.real_ray)])
def test_public_numpy_image_udf_handles_pixels_and_copy_policy(request, monkeypatch, fixed, zero_copy_batch, backend):
    if backend == "ray_actor":
        request.getfixturevalue("ray_local")
    monkeypatch.setenv("VANE_RUNNER", "ray" if backend == "ray_actor" else "local-fast")
    dtype = vane.image_type("RGB", 2, 3) if fixed else vane.image_type("RGB")
    pixels = np.arange(36, dtype=np.uint8).reshape(2, 2, 3, 3)

    class Model:
        def __call__(self, batch):
            images = batch["image"]
            assert images.shape == ((len(images), 2, 3, 3) if fixed else (len(images),))
            assert all(image.flags.writeable != zero_copy_batch for image in images)
            if not zero_copy_batch:
                for image in images:
                    image += 1
            return {"image": images}

    with vane.connect(config={"threads": 2}) as con:
        result = (
            con.from_arrow(pa.table({"image": _image_array(dtype, pixels)}))
            .map_batches(
                Model,
                schema={"image": dtype},
                batch_format="numpy",
                zero_copy_batch=zero_copy_batch,
                batch_size=2,
                execution_backend=backend,
                actor_number=1,
            )
            .fetchall()
        )
    assert len(result) == 2
    actual = sorted([image for (image,) in result], key=lambda image: int(image[0, 0, 0]))
    for image, expected in zip(actual, pixels if zero_copy_batch else pixels + 1, strict=True):
        np.testing.assert_array_equal(image, expected)
