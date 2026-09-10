# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Image mode, pixel dtype and logical transport contracts."""

import pickle

import numpy as np
import pyarrow as pa
import pytest

import vane
from tests.fast.test_native_media_extensions import _connect
from vane._image import image_arrow_type

MODES = [
    ("L", 1, np.uint8),
    ("LA", 2, np.uint8),
    ("RGB", 3, np.uint8),
    ("RGBA", 4, np.uint8),
    ("L16", 1, np.uint16),
    ("LA16", 2, np.uint16),
    ("RGB16", 3, np.uint16),
    ("RGBA16", 4, np.uint16),
    ("RGB32F", 3, np.float32),
    ("RGBA32F", 4, np.float32),
]


def pixels_for(mode, channels, pixel_type, height=3, width=5):
    values = np.arange(height * width * channels).reshape(height, width, channels)
    if pixel_type == np.float32:
        values = values / max(1, values.size - 1)
    elif pixel_type == np.uint16:
        values = values * 509
    return values.astype(pixel_type)


def assert_pixels(actual, expected):
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mode,channels,pixel_type", MODES)
@pytest.mark.parametrize("form", ["generic", "mode", "fixed"])
def test_all_modes_values_sql_arrow_ipc_and_storage(tmp_path, mode, channels, pixel_type, form):
    dtype = (
        vane.image_type()
        if form == "generic"
        else vane.image_type(mode)
        if form == "mode"
        else vane.image_type(mode, 3, 5)
    )
    pixels = pixels_for(mode, channels, pixel_type)[:, ::-1]
    assert dtype == pickle.loads(pickle.dumps(dtype)) == vane.sqltype(str(dtype))
    storage_dtype = pa.float32() if form == "generic" else pa.from_numpy_dtype(pixel_type)
    arrow_type = image_arrow_type(dtype)
    assert (
        arrow_type.storage_type.value_type if form == "fixed" else arrow_type.storage_type.field("data").type.value_type
    ) == storage_dtype
    database = str(tmp_path / "images.db")
    with vane.connect(database) as con:
        value = vane.Value(pixels, dtype)
        con.execute(f"CREATE TABLE images(id INTEGER, value {dtype})")
        con.execute("INSERT INTO images VALUES (0,$1),(1,NULL),(2,$1)", [value])
        con.execute("CHECKPOINT")
        result = con.sql("SELECT value FROM images WHERE id<>1 ORDER BY id DESC")
        assert result.types == [dtype]
        table = result.to_arrow_table()
        assert table.column(0).type == arrow_type
        with pa.BufferOutputStream() as sink:
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
            restored = pa.ipc.open_stream(sink.getvalue()).read_all()
        for row in con.from_arrow(restored).fetchall():
            assert_pixels(row[0], pixels)
        # Pixel bytes in the rendered SQL literal retain the original mode dtype.
        rendered = str(vane.ConstantExpression(value))
        assert_pixels(con.sql(f"SELECT {rendered}").fetchone()[0], pixels)
    with vane.connect(database) as con:
        rows = con.sql("SELECT value FROM images ORDER BY id").fetchall()
        assert_pixels(rows[0][0], pixels)
        assert rows[1] == (None,)
        assert_pixels(rows[2][0], pixels)


@pytest.mark.parametrize("mode,channels,pixel_type", MODES[4:])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_wide_images_registered_row_and_batch_udfs(mode, channels, pixel_type, batch, fixed):
    dtype = vane.image_type(mode, 3, 5) if fixed else vane.image_type(mode)
    pixels = pixels_for(mode, channels, pixel_type)
    arrow_type = image_arrow_type(dtype)

    def identity(value):
        if batch:
            assert value.type == arrow_type
        elif value is not None:
            assert_pixels(value, pixels)
        return value

    udf = (vane.func.batch if batch else vane.func)(return_dtype=dtype)(identity)
    with vane.connect() as con:
        vane.attach_function(udf, connection=con, alias="image_identity", parameters=[dtype])
        source = con.sql("SELECT $1 AS image UNION ALL SELECT NULL", params=[vane.Value(pixels, dtype)])
        for result, expected_count in (
            (source.select(udf(vane.col("image"))), 2),
            (con.sql("SELECT image_identity($1)", params=[vane.Value(pixels, dtype)]), 1),
        ):
            assert result.types == [dtype]
            rows = result.fetchall()
            assert len(rows) == expected_count
            present = [row[0] for row in rows if row[0] is not None]
            assert len(present) == 1
            assert_pixels(present[0], pixels)


def test_generic_mixed_modes_preserve_pixels_and_tensor_storage():
    examples = [
        np.array([[[255]]], dtype=np.uint8),
        np.array([[[1, 256, 65535]]], dtype=np.uint16),
        np.array([[[-2.5, 0.125, 1000]]], dtype=np.float32),
    ]
    values = [
        vane.Value(value, vane.image_type(mode)) for value, mode in zip(examples, ("L", "RGB16", "RGB32F"), strict=True)
    ]
    with vane.connect() as con:
        result = con.sql("SELECT $1 AS value UNION ALL SELECT $2 UNION ALL SELECT $3", params=values)
        assert result.types == [vane.image_type()]
        table = result.to_arrow_table()
        assert table.column(0).type.storage_type.field("data").type == pa.list_(pa.float32())
        for (actual,), expected in zip(con.from_arrow(table).fetchall(), examples, strict=True):
            assert_pixels(actual, expected)
        con.register("images", table)
        tensors = con.sql("SELECT image_to_tensor(value) FROM images")
        assert tensors.types == [vane.tensor_type(vane.sqltypes.FLOAT, (None, None, None))]
        for (actual,), expected in zip(tensors.fetchall(), examples, strict=True):
            assert_pixels(actual, expected.astype(np.float32))


@pytest.mark.parametrize("mode,channels,pixel_type", MODES[4:])
@pytest.mark.parametrize("backend", ["python", "native"])
def test_wide_crop_resize_convert_and_tensor(mode, channels, pixel_type, backend):
    pixels = pixels_for(mode, channels, pixel_type)
    with _connect("image") if backend == "native" else vane.connect() as con:
        value = vane.Value(pixels, vane.image_type(mode))
        assert_pixels(con.sql("SELECT crop($1,[1,1,2,2])", params=[value]).fetchone()[0], pixels[1:3, 1:3])
        resized = con.sql("SELECT resize($1,5,3)", params=[value])
        assert resized.types == [vane.image_type(mode, 3, 5)]
        assert_pixels(resized.fetchone()[0], pixels)
        constant = np.full((2, 2, channels), 0.5 if pixel_type == np.float32 else 65535, dtype=pixel_type)
        upscaled = con.sql("SELECT resize($1,3,4)", params=[vane.Value(constant, vane.image_type(mode))]).fetchone()[0]
        np.testing.assert_allclose(upscaled, np.broadcast_to(constant[0, 0], (4, 3, channels)), rtol=0, atol=1e-7)
        tensor = con.sql("SELECT image_to_tensor($1)", params=[value]).fetchone()[0]
        assert_pixels(tensor, pixels)
        converted = con.sql("SELECT convert_image($1,'RGB32F')", params=[value]).fetchone()[0]
        assert converted.dtype == np.float32 and converted.shape == (3, 5, 3)
        with pytest.raises(vane.InvalidInputException, match="layout"):
            con.sql("SELECT CAST($1 AS IMAGE('RGB'))", params=[value]).fetchall()


@pytest.mark.parametrize(
    "pixels",
    [
        np.full((1, 1, 3), np.nan, np.float32),
        np.full((1, 1, 3), np.inf, np.float32),
        np.zeros((1, 1, 2), np.float32),
        np.zeros((1, 1, 3), np.float64),
    ],
)
def test_invalid_wide_pixels_are_rejected(pixels):
    with pytest.raises(vane.InvalidInputException):
        vane.ConstantExpression(vane.Value(pixels, vane.image_type()))


@pytest.mark.parametrize("mode,code,maximum", [("L", 1, 255), ("L16", 5, 65535)])
@pytest.mark.parametrize("bad", [-1.0, 0.5, "overflow", float("nan"), float("inf")])
def test_generic_arrow_rejects_pixels_invalid_for_row_mode(mode, code, maximum, bad):
    dtype = vane.image_type()
    arrow_type = image_arrow_type(dtype)
    # The invalid value lies after multiple validation chunks.
    pixels = np.full(131075, maximum, np.float32)
    pixels[-1] = maximum + 1 if bad == "overflow" else bad
    storage = pa.array(
        [{"data": pixels, "channel": 1, "height": 1, "width": pixels.size, "mode": code}], type=arrow_type.storage_type
    )
    from vane.execution.udf_file_contract import normalize_file_arrow_array

    with pytest.raises(vane.InvalidInputException, match="finite|representable"):
        normalize_file_arrow_array(pa.ExtensionArray.from_storage(arrow_type, storage), dtype, boundary="test")


@pytest.mark.parametrize(
    "mode,pixel_type",
    [("L", np.uint8), ("L16", np.uint16), ("L", np.float32), ("L16", np.float32), ("RGB32F", np.float32)],
)
@pytest.mark.parametrize("strided", [False, True])
def test_pixel_validation_bounds_numerical_scratch(monkeypatch, mode, pixel_type, strided):
    from vane._image import _validate_pixels

    pixels = np.ones((512, 1024, 3), pixel_type)
    if strided:
        pixels = pixels[:, ::-1]

    def bounded(function):
        def check(values, *args, **kwargs):
            # Numerical validation must not allocate an image-sized temporary.
            assert values.nbytes <= 1024 * 1024
            return function(values, *args, **kwargs)

        return check

    with monkeypatch.context() as patch:
        for name in ("isfinite", "floor"):
            patch.setattr(np, name, bounded(getattr(np, name)))
        _validate_pixels(pixels, mode)


def test_uint16_packed_values_compare_numerically():
    with vane.connect() as con:
        values = [
            vane.Value(np.full((1, 1, 1), number, np.uint16), vane.image_type("L16")) for number in (1, 256, 65535)
        ]
        assert con.sql("SELECT $1 < $2, $2 < $3, $3 > $1", params=values).fetchone() == (True, True, True)
