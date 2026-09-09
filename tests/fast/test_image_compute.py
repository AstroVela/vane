# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Byte codecs, fixed-width perceptual hashes and explicit backend dispatch."""

import io
import struct
import zlib

import numpy as np
import pyarrow as pa
import pytest

import vane
from tests.fast.test_image_modes import MODES, assert_pixels, pixels_for
from tests.fast.test_native_media_extensions import _connect

METHODS = ("ahash", "dhash", "dhash_vertical", "phash", "phash_simple", "whash", "colorhash", "crop_resistant")


@pytest.fixture(params=["python", "native"])
def image_connection(request):
    with _connect("image") if request.param == "native" else vane.connect() as con:
        yield con


@pytest.mark.parametrize("mode,channels,pixel_type", MODES)
@pytest.mark.parametrize("image_format", ["PNG", "TIFF"])
def test_lossless_codec_matrix(image_connection, mode, channels, pixel_type, image_format):
    con = image_connection
    pixels = pixels_for(mode, channels, pixel_type)
    value = vane.Value(pixels, vane.image_type(mode))
    if image_format == "PNG" and pixel_type == np.float32:
        with pytest.raises(vane.InvalidInputException, match="convert_image"):
            con.sql("SELECT encode_image($1,$2)", params=[value, image_format]).fetchall()
        return
    encoded = con.sql("SELECT encode_image($1,$2)", params=[value, image_format]).fetchone()[0]
    result = con.sql("SELECT decode_image($1,mode=>NULL)", params=[encoded])
    assert result.types == [vane.image_type()]
    assert_pixels(result.fetchone()[0], pixels)
    # Functional and method API bind to the same typed output as SQL.
    for expression in (vane.decode_image(vane.lit(encoded), mode=mode), vane.lit(encoded).decode_image(mode=mode)):
        result = con.sql("SELECT 1").select(expression)
        assert result.types == [vane.image_type(mode)]
        assert_pixels(result.fetchone()[0], pixels)
    default = con.sql("SELECT decode_image($1)", params=[encoded])
    assert default.types == [vane.image_type("RGB")]
    assert default.fetchone()[0].shape == (3, 5, 3)


@pytest.mark.parametrize("image_format", ["JPEG", "GIF", "BMP"])
@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_standard_codec_outputs_are_readable(image_connection, image_format, mode):
    pil = pytest.importorskip("PIL.Image")
    channels = 1 if mode == "L" else 3
    pixels = np.full((16, 17, channels), 119, np.uint8)
    value = vane.Value(pixels, vane.image_type(mode))
    encoded = image_connection.sql("SELECT encode_image($1,$2)", params=[value, image_format]).fetchone()[0]
    with pil.open(io.BytesIO(encoded)) as image:
        assert image.format == image_format and image.size == (17, 16)
        expected = np.asarray(image.convert(mode)).reshape(pixels.shape)
    actual = image_connection.sql("SELECT decode_image($1,mode=>$2)", params=[encoded, mode]).fetchone()[0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2 if image_format == "JPEG" else 0)
    if image_format == "BMP" or mode == "L":
        np.testing.assert_allclose(actual, pixels, rtol=0, atol=2 if image_format == "JPEG" else 0)


@pytest.mark.parametrize("mode", ["1", "L", "P"])
@pytest.mark.parametrize("image_format", ["PNG", "BMP"])
def test_palette_and_grayscale_decode_preserve_pixels(image_connection, mode, image_format, tmp_path):
    pil = pytest.importorskip("PIL.Image")
    source = pil.new(mode, (3, 2))
    if mode == "P":
        source.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * (768 - 9))
        source.putdata([0, 1, 2, 2, 1, 0])
        if image_format == "PNG":
            source.info["transparency"] = 1
    else:
        source.putdata([0, 255, 0, 255, 0, 255])
    encoded = io.BytesIO()
    source.save(encoded, format=image_format)
    expected_mode = "L" if mode in ("1", "L") else "RGBA"
    expected = np.asarray(source.convert(expected_mode)).reshape(2, 3, 1 if expected_mode == "L" else 4)
    actual = image_connection.sql("SELECT decode_image($1,mode=>NULL)", params=[encoded.getvalue()]).fetchone()[0]
    assert_pixels(actual, expected)
    path = tmp_path / ("palette." + image_format.lower())
    path.write_bytes(encoded.getvalue())
    metadata = image_connection.sql("SELECT image_file_metadata(image_file($1))", params=[str(path)]).fetchone()[0]
    assert metadata["mode"] == mode


@pytest.mark.parametrize("mode", ["LA", "RGBA", "L16", "RGB16", "RGB32F"])
@pytest.mark.parametrize("image_format", ["JPEG", "GIF", "BMP"])
def test_encoders_require_explicit_supported_pixel_mode(image_connection, mode, image_format):
    mode, channels, dtype = next(item for item in MODES if item[0] == mode)
    value = vane.Value(pixels_for(mode, channels, dtype), vane.image_type(mode))
    with pytest.raises(vane.InvalidInputException, match="convert_image"):
        image_connection.sql("SELECT encode_image($1,$2)", params=[value, image_format]).fetchall()


def test_decode_nulls_errors_named_arguments_and_reuse(image_connection):
    con = image_connection
    assert con.sql(
        "SELECT decode_image(NULL),decode_image('bad'::BLOB,on_error=>'null'),decode_image('bad'::BLOB,on_error=>NULL)"
    ).fetchone() == (None, None, None)
    with pytest.raises(vane.InvalidInputException):
        con.sql("SELECT decode_image('bad'::BLOB)").fetchall()
    with pytest.raises(vane.InvalidInputException, match="on_error"):
        con.sql("SELECT decode_image('bad'::BLOB,on_error=>'ignore')").fetchall()
    with pytest.raises(vane.BinderException, match="BINARY"):
        con.sql("SELECT decode_image('a path')").fetchall()
    encoded = con.sql("SELECT encode_image(image('abc'::BLOB,1,1,3,'RGB'),'PNG')").fetchone()[0]
    con.execute("PREPARE decoder AS SELECT decode_image($1,mode=>'RGBA',on_error=>'null')")
    con.register("encoded", pa.table({"id": range(4101), "bytes": [encoded if i % 3 else None for i in range(4101)]}))
    result = con.sql("SELECT id,decode_image(bytes,mode=>'RGBA') FROM encoded WHERE id%7=1 ORDER BY id DESC")
    assert result.types[1] == vane.image_type("RGBA")
    for index, image in result.fetchall():
        if index % 3 == 0:
            assert image is None
        else:
            assert_pixels(image, np.array([[[97, 98, 99, 255]]], np.uint8))
    assert con.sql("SELECT 42").fetchone() == (42,)


def test_oversized_decode_is_never_suppressed(image_connection):
    def chunk(name, value):
        return struct.pack(">I", len(value)) + name + value + struct.pack(">I", zlib.crc32(name + value))

    encoded = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 100_000_001, 1, 8, 2, 0, 0, 0))
    encoded += chunk(b"IDAT", zlib.compress(b"\0")) + chunk(b"IEND", b"")
    with pytest.raises(vane.OutOfRangeException, match="pixel|limit"):
        image_connection.sql("SELECT decode_image($1,on_error=>'null')", params=[encoded]).fetchall()


@pytest.mark.parametrize(
    "error", [MemoryError("allocation"), ImportError("codec dependency"), RuntimeError("system failure")]
)
def test_python_decode_does_not_suppress_system_errors(monkeypatch, error):
    import vane._image_compute as helpers

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(helpers, "_decode_image_bytes", fail)
    with vane.connect() as con, pytest.raises(vane.Error):
        con.sql("SELECT decode_image('bad'::BLOB,on_error=>'null')").fetchall()


@pytest.mark.parametrize("mode,channels,pixel_type", [MODES[6], MODES[9]])
def test_imagefile_decode_uses_logical_window_and_preserves_wide_pixels(
    image_connection, tmp_path, mode, channels, pixel_type
):
    pixels = pixels_for(mode, channels, pixel_type)
    encoded = image_connection.sql(
        "SELECT encode_image($1,'TIFF')", params=[vane.Value(pixels, vane.image_type(mode))]
    ).fetchone()[0]
    path = tmp_path / "window.bin"
    path.write_bytes(b"prefix" + encoded + b"suffix")
    value = vane.ImageFile(str(path), "image/tiff", 6, len(encoded))
    metadata, decoded = image_connection.sql(
        "SELECT image_file_metadata($1),decode_image_file($1)", params=[value]
    ).fetchone()
    assert metadata == {"width": 5, "height": 3, "format": "TIFF", "mode": mode}
    assert_pixels(decoded, pixels)
    wrong = vane.ImageFile(str(path), "image/png", 6, len(encoded))
    assert image_connection.sql("SELECT decode_image_file($1,NULL,'null')", params=[wrong]).fetchone() == (None,)


@pytest.mark.parametrize(
    "layout",
    ["tiled", "orientation", "associated_alpha", "unspecified_alpha", "palette", "cmyk", "volume", "planar"],
)
def test_tiff_metadata_and_decode_reject_unsupported_layouts(image_connection, tmp_path, layout):
    tifffile = pytest.importorskip("tifffile")
    pixels = np.zeros((16, 16, 3), np.uint8)
    options = {"photometric": "rgb"}
    if layout == "tiled":
        options["tile"] = (16, 16)
    elif layout == "orientation":
        options["extratags"] = [(274, "H", 1, 6, False)]
    elif layout in ("associated_alpha", "unspecified_alpha"):
        pixels = np.zeros((16, 16, 4), np.uint8)
        options["extrasamples"] = ["assocalpha" if layout == "associated_alpha" else "unspecified"]
    elif layout == "palette":
        pixels = np.zeros((16, 16), np.uint8)
        options = {"photometric": "palette", "colormap": np.zeros((3, 256), np.uint16)}
    elif layout == "cmyk":
        pixels = np.zeros((16, 16, 4), np.uint8)
        options["photometric"] = "separated"
    elif layout == "volume":
        pixels = np.zeros((2, 16, 16, 3), np.uint8)
        options["volumetric"] = True
    path = tmp_path / "unsupported.tiff"
    tifffile.imwrite(path, pixels, metadata=None, **options)
    if layout == "planar":
        with tifffile.TiffFile(path, mode="r+") as tiff:
            tiff.pages[0].tags["PlanarConfiguration"].overwrite(3)

    value = vane.ImageFile(str(path), "image/tiff")
    with pytest.raises(vane.InvalidInputException):
        image_connection.sql("SELECT image_file_metadata($1)", params=[value]).fetchall()
    with pytest.raises(vane.ImageFileFormatError):
        value.metadata()
    for function, argument in (("decode_image", path.read_bytes()), ("decode_image_file", value)):
        with pytest.raises(vane.InvalidInputException):
            image_connection.sql(f"SELECT {function}($1)", params=[argument]).fetchall()
        assert image_connection.sql(f"SELECT {function}($1,on_error=>'null')", params=[argument]).fetchone() == (None,)


@pytest.mark.parametrize("layout", ["separate", "miniswhite"])
def test_tiff_metadata_and_decode_keep_supported_layouts(image_connection, tmp_path, layout):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "supported.tiff"
    if layout == "separate":
        pixels = np.arange(2 * 5 * 3, dtype=np.uint16).reshape(2, 5, 3)
        tifffile.imwrite(path, np.moveaxis(pixels, -1, 0), photometric="rgb", planarconfig="separate", metadata=None)
        mode = "RGB16"
    else:
        source = np.arange(2 * 5, dtype=np.uint8).reshape(2, 5)
        tifffile.imwrite(path, source, photometric="miniswhite", metadata=None)
        pixels = (255 - source)[:, :, None]
        mode = "L"
    value = vane.ImageFile(str(path), "image/tiff")
    metadata, decoded = image_connection.sql(
        "SELECT image_file_metadata($1),decode_image_file($1)", params=[value]
    ).fetchone()
    assert metadata == {"width": 5, "height": 2, "format": "TIFF", "mode": mode}
    assert_pixels(decoded, pixels)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("size", [3, 8])
def test_hash_function_method_sql_and_fixed_width_arrow(image_connection, method, size):
    if method == "whash" and size == 3:
        return
    pixels = np.random.default_rng(17).integers(0, 256, (19, 23, 3), dtype=np.uint8)
    value = vane.Value(pixels, vane.image_type("RGB"))
    bits = 14 * 3 if method == "colorhash" else size * size * (9 if method == "crop_resistant" else 1)
    byte_width = (bits + 7) // 8
    dtype = vane.sqltype(f"FIXEDBINARY({byte_width})")
    source = image_connection.sql("SELECT $1 AS image UNION ALL SELECT NULL", params=[value])
    expressions = (
        vane.image_hash(vane.col("image"), method=method, hash_size=size),
        vane.col("image").image_hash(method=method, hash_size=size),
    )
    expected = image_connection.sql(
        "SELECT image_hash($1,hash_size=>$2,method=>$3)", params=[value, size, method]
    ).fetchone()[0]
    assert len(expected) == byte_width
    if bits % 8:
        assert expected[-1] & ((1 << (8 - bits % 8)) - 1) == 0
    for expression in expressions:
        result = source.select(expression.alias("hash"))
        assert result.types == [dtype]
        table = result.to_arrow_table()
        assert table.column(0).type == pa.binary(byte_width)
        assert table.column(0).to_pylist() == [expected, None]
        assert image_connection.from_arrow(table).types == [dtype]
        assert image_connection.from_arrow(table).fetchall() == [(expected,), (None,)]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("mode,channels,pixel_type", [MODES[0], MODES[3], MODES[6], MODES[9]])
def test_native_hash_matches_python(method, mode, channels, pixel_type):
    pixels = pixels_for(mode, channels, pixel_type, 13, 17)
    if pixel_type == np.uint16:
        pixels = np.random.default_rng(819).integers(0, 65536, pixels.shape, dtype=np.uint16)
    value = vane.Value(pixels, vane.image_type(mode))
    with vane.connect() as python, _connect("image") as native:
        sql = "SELECT image_hash($1,method=>$2)"
        assert native.sql(sql, params=[value, method]).fetchone() == python.sql(sql, params=[value, method]).fetchone()


@pytest.mark.parametrize("method", ["ahash", "dhash", "dhash_vertical", "phash_simple", "whash"])
def test_constant_black_hash_is_zero(image_connection, method):
    value = vane.Value(np.zeros((16, 16, 3), np.uint8), vane.image_type("RGB"))
    assert image_connection.sql("SELECT image_hash($1,method=>$2)", params=[value, method]).fetchone() == (bytes(8),)


def test_hash_known_gradient_and_histogram_bits(image_connection):
    horizontal = np.broadcast_to(np.arange(9, dtype=np.uint8)[None, :, None] * 20, (8, 9, 1)).copy()
    value = vane.Value(horizontal, vane.image_type("L"))
    assert image_connection.sql("SELECT image_hash($1,method=>'dhash')", params=[value]).fetchone() == (b"\xff" * 8,)
    # All pixels are black: first 3-bit histogram count is 111, all others zero.
    black = vane.Value(np.zeros((8, 8, 3), np.uint8), vane.image_type("RGB"))
    assert image_connection.sql("SELECT image_hash($1,method=>'colorhash')", params=[black]).fetchone() == (
        b"\xe0" + bytes(5),
    )


@pytest.mark.parametrize(
    "options",
    [
        "method=>'unknown'",
        "hash_size=>1",
        "hash_size=>65",
        "hash_size=>2.5",
        "hash_size=>true",
        "binbits=>0",
        "segments=>17",
        "method=>'whash',hash_size=>3",
        "method=>NULL",
    ],
)
def test_hash_rejects_invalid_options_at_bind(image_connection, options):
    with pytest.raises(vane.Error):
        image_connection.sql(f"SELECT image_hash(NULL::IMAGE,{options}) WHERE FALSE")


def test_hash_requires_constant_shape_options(image_connection):
    with pytest.raises(vane.BinderException, match="constant"):
        image_connection.sql("SELECT image_hash(NULL::IMAGE,hash_size=>i) FROM range(2,5) t(i)")


def test_fixed_binary_cast_storage_and_udf(image_connection, tmp_path):
    con = image_connection
    dtype = vane.sqltype("FIXEDBINARY(2)")
    assert con.sql("SELECT 'ab'::BLOB::FIXEDBINARY(2),TRY_CAST('a'::BLOB AS FIXEDBINARY(2))").fetchone() == (
        b"ab",
        None,
    )
    with pytest.raises(vane.InvalidInputException, match="exactly 2 bytes"):
        con.sql("SELECT 'a'::BLOB::FIXEDBINARY(2)").fetchall()

    @vane.func.batch(return_dtype=dtype)
    def identity(values):
        assert values.type == pa.binary(2)
        return values

    vane.attach_function(identity, connection=con, alias="hash_identity", parameters=[dtype])
    con.register("hash_values", pa.table({"value": pa.array([b"ab", None, b"cd"], type=pa.binary(2))}))
    assert con.sql("SELECT hash_identity(value) FROM hash_values").fetchall() == [(b"ab",), (None,), (b"cd",)]
    con.execute("CREATE TABLE hashes AS SELECT * FROM hash_values")
    assert con.table("hashes").to_arrow_table().column(0).type == pa.binary(2)


def test_native_codec_and_hash_never_enter_python(monkeypatch):
    import vane._image_compute as helpers

    def forbidden(*args, **kwargs):
        pytest.fail("native Image computation entered Python")

    with _connect("image") as con:
        for name in ("_decode_image_bytes", "_encode_image_bytes", "_image_hash"):
            monkeypatch.setattr(helpers, name, forbidden)
        for image_format in ("PNG", "JPEG", "TIFF", "GIF", "BMP"):
            assert con.sql(
                "SELECT octet_length(image_hash(decode_image(encode_image(image('abc'::BLOB,1,1,3,'RGB'),$1))))",
                params=[image_format],
            ).fetchone() == (8,)
