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
def test_standard_codec_outputs_are_readable(image_connection, image_format, mode, tmp_path):
    pil = pytest.importorskip("PIL.Image")
    channels = 1 if mode == "L" else 3
    pixels = np.full((16, 17, channels), 119, np.uint8)
    value = vane.Value(pixels, vane.image_type(mode))
    encoded = image_connection.sql("SELECT encode_image($1,$2)", params=[value, image_format]).fetchone()[0]
    with pil.open(io.BytesIO(encoded)) as image:
        assert image.format == image_format and image.size == (17, 16)
        if image_format == "JPEG":
            assert image.mode == mode
        expected = np.asarray(image.convert(mode)).reshape(pixels.shape)
    actual = image_connection.sql("SELECT decode_image($1,mode=>$2)", params=[encoded, mode]).fetchone()[0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2 if image_format == "JPEG" else 0)
    if image_format == "BMP" or mode == "L":
        np.testing.assert_allclose(actual, pixels, rtol=0, atol=2 if image_format == "JPEG" else 0)
    if image_format == "JPEG":
        inferred = image_connection.sql("SELECT decode_image($1,mode=>NULL)", params=[encoded]).fetchone()[0]
        assert inferred.shape == pixels.shape
        path = tmp_path / "roundtrip.jpg"
        path.write_bytes(encoded)
        metadata = image_connection.sql("SELECT image_file_metadata(image_file($1))", params=[str(path)]).fetchone()[0]
        assert metadata["mode"] == mode


def test_grayscale_jpeg_streams_multiple_output_buffers_and_recovers_from_codec_errors(image_connection):
    pil = pytest.importorskip("PIL.Image")
    pixels = np.random.default_rng(37).integers(0, 256, (512, 513, 1), dtype=np.uint8)
    encoded = image_connection.sql(
        "SELECT encode_image($1,'JPEG')", params=[vane.Value(pixels, vane.image_type("L"))]
    ).fetchone()[0]
    assert len(encoded) > 65536
    with pil.open(io.BytesIO(encoded)) as image:
        assert image.mode == "L"
        expected = np.asarray(image).reshape(pixels.shape)
    actual = image_connection.sql("SELECT decode_image($1,mode=>NULL)", params=[encoded]).fetchone()[0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2)
    # The JPEG dimension ceiling is lower than the Image type's limit. A
    # library error must release the encoder and leave the connection usable.
    oversized = vane.Value(np.zeros((1, 70000, 1), np.uint8), vane.image_type("L"))
    with pytest.raises(vane.Error):
        image_connection.sql("SELECT encode_image($1,'JPEG')", params=[oversized]).fetchall()
    assert image_connection.sql("SELECT 42").fetchone() == (42,)


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


@pytest.mark.parametrize(
    "bits,colors,indices",
    [
        (1, 2, [0, 1, 1, 0, 1]),
        (4, 16, [1, 15, 2, 0, 3]),
        (4, 2, [0, 1, 1, 0, 1]),
        (4, 4, [0, 1, 2, 3, 1]),
        (8, 2, [0, 1, 1, 0, 1]),
        (8, 16, [1, 15, 2, 0, 3]),
        (8, 256, [1, 255, 2, 0, 3]),
    ],
)
@pytest.mark.parametrize("header_size,top_down", [(40, False), (40, True), (124, True)])
def test_bmp_compact_gray_palettes_preserve_index_depth(
    image_connection, tmp_path, bits, colors, indices, header_size, top_down
):
    rows = [indices, indices[::-1]]
    raw = bytearray()
    stride = ((5 * bits + 31) // 32) * 4
    for row in rows if top_down else rows[::-1]:
        packed = bytearray(stride)
        for x, index in enumerate(row):
            packed[x * bits // 8] |= index << (8 - bits - (x * bits % 8))
        raw.extend(packed)
    palette = b"".join(bytes((i * 255 if colors == 2 else i,) * 3 + (0,)) for i in range(colors))
    dib = struct.pack("<IiiHHIIiiII", header_size, 5, -2 if top_down else 2, 1, bits, 0, len(raw), 0, 0, colors, 0)
    dib += bytes(header_size - len(dib))
    offset = 14 + len(dib) + len(palette)
    encoded = struct.pack("<2sIHHI", b"BM", offset + len(raw), 0, 0, offset) + dib + palette + raw
    path = tmp_path / "compact-palette.bin"
    path.write_bytes(b"prefix" + encoded + b"suffix")
    value = vane.ImageFile(str(path), "image/bmp", 6, len(encoded))
    expected = np.array(rows, np.uint8)[:, :, None] * (255 if colors == 2 else 1)
    metadata, decoded, file_decoded, rgb = image_connection.sql(
        "SELECT image_file_metadata($1),decode_image($2,mode=>NULL),decode_image_file($1),decode_image($2,mode=>'RGB')",
        params=[value, encoded],
    ).fetchone()
    assert metadata == {"width": 5, "height": 2, "format": "BMP", "mode": "1" if colors == 2 else "L"}
    assert_pixels(decoded, expected)
    assert_pixels(file_decoded, expected)
    assert_pixels(rgb, np.repeat(expected, 3, axis=2))
    with value.decode() as image:
        assert image.mode == metadata["mode"]
        with image.convert("L") as gray:
            assert_pixels(np.asarray(gray)[:, :, None], expected)


@pytest.mark.parametrize("bits,compression", [(4, 2), (8, 1)])
def test_bmp_rle_black_white_palette_preserves_intensities(image_connection, tmp_path, bits, compression):
    raw = bytes([2, 0x01, 0, 0, 0, 1]) if bits == 4 else bytes([1, 0, 1, 1, 0, 0, 0, 1])
    palette = bytes(4) + bytes([255, 255, 255, 0])
    dib = struct.pack("<IiiHHIIiiII", 40, 2, 1, 1, bits, compression, len(raw), 0, 0, 2, 0)
    offset = 14 + len(dib) + len(palette)
    encoded = struct.pack("<2sIHHI", b"BM", offset + len(raw), 0, 0, offset) + dib + palette + raw
    path = tmp_path / "rle-palette.bmp"
    path.write_bytes(encoded)
    value = vane.ImageFile(str(path), "image/bmp")
    decoded, file_decoded = image_connection.sql(
        "SELECT decode_image($1,mode=>NULL),decode_image_file($2)", params=[encoded, value]
    ).fetchone()
    expected = np.array([[[0], [255]]], np.uint8)
    assert_pixels(decoded, expected)
    assert_pixels(file_decoded, expected)
    with value.decode("L") as image:
        assert_pixels(np.asarray(image)[:, :, None], expected)


def test_bmp_core_header_gray_palette_preserves_four_bit_indices(image_connection, tmp_path):
    palette = b"".join(bytes((i, i, i)) for i in range(16))
    dib = struct.pack("<IHHHH", 12, 2, 1, 1, 4)
    offset = 14 + len(dib) + len(palette)
    encoded = struct.pack("<2sIHHI", b"BM", offset + 4, 0, 0, offset) + dib + palette + bytes([0x1F, 0, 0, 0])
    path = tmp_path / "core-palette.bmp"
    path.write_bytes(encoded)
    value = vane.ImageFile(str(path), "image/bmp")
    decoded, file_decoded = image_connection.sql(
        "SELECT decode_image($1,mode=>NULL),decode_image_file($2)", params=[encoded, value]
    ).fetchone()
    expected = np.array([[[1], [15]]], np.uint8)
    assert_pixels(decoded, expected)
    assert_pixels(file_decoded, expected)
    with value.decode() as image:
        assert_pixels(np.asarray(image)[:, :, None], expected)


@pytest.mark.parametrize("mode", ["1", "L", "RGB"])
def test_bmp_metadata_accepts_exact_header_budget(image_connection, tmp_path, mode):
    pil = pytest.importorskip("PIL.Image")
    output = io.BytesIO()
    pil.new(mode, (3, 2)).save(output, format="BMP")
    encoded = output.getvalue()
    header_size = int.from_bytes(encoded[10:14], "little")
    path = tmp_path / "header-window.bin"
    path.write_bytes(b"prefix" + encoded + b"suffix")
    value = vane.ImageFile(str(path), "image/bmp", 6, len(encoded))
    assert value.metadata(max_bytes=header_size) == vane.ImageMetadata(3, 2, "BMP", mode)
    assert image_connection.sql(
        "SELECT image_file_metadata($1,max_bytes=>$2::UBIGINT)", params=[value, header_size]
    ).fetchone()[0] == {"width": 3, "height": 2, "format": "BMP", "mode": mode}
    # Native probing skips two reserved bytes, so this budget is below both
    # the prefix size and the actual unique bytes needed by either backend.
    with pytest.raises(vane.Error, match="max_bytes"):
        image_connection.sql(
            "SELECT image_file_metadata($1,max_bytes=>$2::UBIGINT)", params=[value, header_size - 3]
        ).fetchall()


@pytest.mark.parametrize("bits", [1, 4, 8])
@pytest.mark.parametrize("header_size", [12, 40, 124])
def test_bmp_rejects_palette_overlapping_pixels(image_connection, tmp_path, bits, header_size):
    colors = 1 << bits
    stride = 3 if header_size == 12 else 4
    palette = b"".join(bytes((i, i, i)) + bytes(stride - 3) for i in range(colors))
    if header_size == 12:
        dib = struct.pack("<IHHHH", header_size, 2, 1, 1, bits)
    else:
        dib = struct.pack("<IiiHHIIiiII", header_size, 2, 1, 1, bits, 0, 4, 0, 0, colors, 0)
        dib += bytes(header_size - len(dib))
    offset = 14 + len(dib) + len(palette)
    # The physical file contains the whole table: only the declared raster
    # boundary is invalid, so an EOF-only check cannot detect the overlap.
    encoded = struct.pack("<2sIHHI", b"BM", offset + 4, 0, 0, offset - 1) + dib + palette + bytes(4)
    path = tmp_path / "overlapping-palette.bmp"
    path.write_bytes(encoded)
    value = vane.ImageFile(str(path), "image/bmp")
    with pytest.raises(vane.ImageFileFormatError, match="overlaps"):
        value.metadata()
    with pytest.raises(vane.ImageFileFormatError, match="overlaps"):
        value.decode()
    with pytest.raises(vane.InvalidInputException, match="overlaps"):
        image_connection.sql("SELECT image_file_metadata($1)", params=[value]).fetchall()
    for function, argument in (("decode_image", encoded), ("decode_image_file", value)):
        with pytest.raises(vane.InvalidInputException, match="overlaps"):
            image_connection.sql(f"SELECT {function}($1)", params=[argument]).fetchall()
        assert image_connection.sql(f"SELECT {function}($1,on_error=>'null')", params=[argument]).fetchone() == (None,)


@pytest.mark.parametrize("mode", ["1", "L"])
def test_bmp_decoder_accepts_pillow_tile_tuple_protocol(monkeypatch, duckdb_cursor, tmp_path, mode):
    pil = pytest.importorskip("PIL.Image")
    source = pil.new(mode, (3, 2))
    source.putdata([0, 255, 0, 255, 0, 255])
    encoded = io.BytesIO()
    source.save(encoded, format="BMP")
    expected = np.asarray(source.convert("L")).reshape(2, 3, 1)
    original_open = pil.open

    def open_with_plain_tiles(*args, **kwargs):
        image = original_open(*args, **kwargs)
        if image.format == "BMP":
            image.tile = [tuple(tile) for tile in image.tile]
        return image

    monkeypatch.setattr(pil, "open", open_with_plain_tiles)
    path = tmp_path / "tuple-tiles.bmp"
    path.write_bytes(encoded.getvalue())
    value = vane.ImageFile(str(path), "image/bmp")
    for function, argument in (("decode_image", encoded.getvalue()), ("decode_image_file", value)):
        actual = duckdb_cursor.sql(f"SELECT {function}($1,mode=>'L')", params=[argument]).fetchone()[0]
        assert_pixels(actual, expected)
    with value.decode("L") as image:
        assert_pixels(np.asarray(image)[:, :, None], expected)


@pytest.mark.parametrize(
    "compression,bits,height",
    [(99, 24, 1), (4, 24, 1), (5, 24, 1), (1, 24, 1), (2, 8, 1), (3, 8, 1), (3, 24, 1), (1, 8, -1), (2, 4, -1)],
)
def test_bmp_metadata_and_decoding_reject_invalid_compression(image_connection, tmp_path, compression, bits, height):
    palette = b"".join(bytes((i, i, i, 0)) for i in range(1 << bits)) if bits <= 8 else b""
    masks = struct.pack("<III", 0xFF0000, 0xFF00, 0xFF) if compression == 3 else b""
    dib = struct.pack("<IiiHHIIiiII", 40, 2, height, 1, bits, compression, 8, 0, 0, 0, 0)
    offset = 14 + len(dib) + len(masks) + len(palette)
    encoded = struct.pack("<2sIHHI", b"BM", offset + 8, 0, 0, offset) + dib + masks + palette + bytes(8)
    path = tmp_path / "invalid-compression.bmp"
    path.write_bytes(encoded)
    value = vane.ImageFile(str(path), "image/bmp")
    with pytest.raises(vane.InvalidInputException, match="BMP|supported encoded image"):
        image_connection.sql("SELECT image_file_metadata($1)", params=[value]).fetchall()
    assert image_connection.sql(
        "SELECT decode_image($1,on_error=>'null'),decode_image_file($2,on_error=>'null')", params=[encoded, value]
    ).fetchone() == (None, None)


@pytest.mark.parametrize("bits,compression", [(4, 2), (8, 1), (16, 3)])
def test_bmp_supported_compression_metadata_and_pixels(image_connection, tmp_path, bits, compression):
    if bits <= 8:
        palette = b"".join(bytes((i, i, i, 0)) for i in range(1 << bits))
        raw = bytes([2, 0x11 if bits == 4 else 1, 0, 0, 0, 1])
        expected = np.ones((1, 2, 1), np.uint8)
        mode = "L"
    else:
        palette = struct.pack("<III", 0xF800, 0x7E0, 0x1F)
        raw = struct.pack("<HH", 0xFFFF, 0)
        expected = np.array([[[255, 255, 255], [0, 0, 0]]], np.uint8)
        mode = "RGB"
    dib = struct.pack("<IiiHHIIiiII", 40, 2, 1, 1, bits, compression, len(raw), 0, 0, 0, 0)
    offset = 14 + len(dib) + len(palette)
    encoded = struct.pack("<2sIHHI", b"BM", offset + len(raw), 0, 0, offset) + dib + palette + raw
    path = tmp_path / "supported-compression.bmp"
    path.write_bytes(encoded)
    metadata, decoded, file_decoded = image_connection.sql(
        "SELECT image_file_metadata($1),decode_image($2,mode=>NULL),decode_image_file($1)",
        params=[vane.ImageFile(str(path), "image/bmp"), encoded],
    ).fetchone()
    assert metadata == {"width": 2, "height": 1, "format": "BMP", "mode": mode}
    assert_pixels(decoded, expected)
    assert_pixels(file_decoded, expected)


@pytest.mark.parametrize(
    "header_size,bits,masks",
    [
        (40, 32, (0, 0, 0, 0)),
        (52, 32, (0xFF0000, 0xFF00, 0xFF, 0)),
        (56, 32, (0, 0, 0, 0)),
        (56, 32, (0, 0xFF00, 0xFF, 0xFF000000)),
        (56, 32, (0xFF0000, 0xFF0000, 0xFF, 0xFF000000)),
        (56, 32, (0xFF0000, 0xFF00, 0x55, 0xFF000000)),
        (56, 16, (0x10000, 0x7E0, 0x1F, 0)),
        (56, 32, (0xFF0000, 0xFF00, 0xFF, 0xFF0000)),
        (56, 16, (0xF00, 0xF0, 0xF, 0)),
        (56, 32, (0xFF0000, 0xFF00, 0xFF, 0x0F000000)),
        (56, 16, (0x7C00, 0x3E0, 0x1F, 0x8000)),
    ],
)
def test_bmp_rejects_invalid_or_unsupported_bitfields(image_connection, tmp_path, header_size, bits, masks):
    dib = struct.pack("<IiiHHIIiiII", header_size, 1, 1, 1, bits, 3, 4, 0, 0, 0, 0)
    dib += struct.pack("<III", *masks[:3]) if header_size < 56 else struct.pack("<IIII", *masks)
    offset = 14 + len(dib)
    encoded = struct.pack("<2sIHHI", b"BM", offset + 4, 0, 0, offset) + dib + bytes(4)
    path = tmp_path / "invalid-masks.bmp"
    path.write_bytes(encoded)
    value = vane.ImageFile(str(path), "image/bmp")
    with pytest.raises(vane.InvalidInputException, match="BMP|supported encoded image"):
        image_connection.sql("SELECT image_file_metadata($1)", params=[value]).fetchall()
    assert image_connection.sql(
        "SELECT decode_image($1,on_error=>'null'),decode_image_file($2,on_error=>'null')", params=[encoded, value]
    ).fetchone() == (None, None)


@pytest.mark.parametrize(
    "header_size,masks",
    [
        (40, (0xFF0000, 0xFF00, 0xFF, 0)),
        (40, (0xFF000000, 0xFF0000, 0xFF00, 0)),
        (64, (0xFF0000, 0xFF00, 0xFF, 0)),
        (56, (0xFF0000, 0xFF00, 0xFF, 0xFF000000)),
        (56, (0xFF000000, 0xFF0000, 0xFF00, 0xFF)),
        (56, (0xFF, 0xFF00, 0xFF0000, 0xFF000000)),
        (124, (0xFF000000, 0xFF0000, 0xFF00, 0xFF)),
    ],
)
def test_bmp_supported_bitfield_layouts_preserve_channels(image_connection, tmp_path, header_size, masks):
    expected = np.array([[[10, 20, 30, 40], [50, 60, 70, 80]]], np.uint8)
    packed = []
    for pixel in expected[0]:
        value = sum(int(channel) * (mask & -mask) for channel, mask in zip(pixel, masks, strict=True) if mask)
        packed.append(value)
    raw = struct.pack("<II", *packed)
    dib = struct.pack("<IiiHHIIiiII", header_size, 2, 1, 1, 32, 3, len(raw), 0, 0, 0, 0)
    dib += struct.pack("<III", *masks[:3]) if header_size < 56 else struct.pack("<IIII", *masks)
    dib += bytes(max(0, header_size - len(dib)))
    offset = 14 + len(dib)
    encoded = struct.pack("<2sIHHI", b"BM", offset + len(raw), 0, 0, offset) + dib + raw
    path = tmp_path / "valid-masks.bmp"
    path.write_bytes(encoded)
    metadata, decoded, file_decoded = image_connection.sql(
        "SELECT image_file_metadata($1),decode_image($2,mode=>NULL),decode_image_file($1)",
        params=[vane.ImageFile(str(path), "image/bmp"), encoded],
    ).fetchone()
    mode = "RGBA" if masks[3] else "RGB"
    assert metadata == {"width": 2, "height": 1, "format": "BMP", "mode": mode}
    assert_pixels(decoded, expected if masks[3] else expected[:, :, :3])
    assert_pixels(file_decoded, expected if masks[3] else expected[:, :, :3])


@pytest.mark.parametrize("compression", [3, 6])
def test_bmp_alpha_is_preserved_or_explicitly_rejected(image_connection, tmp_path, compression):
    pixels = np.array([[[10, 20, 30, 0], [40, 50, 60, 64]], [[70, 80, 90, 128], [100, 110, 120, 255]]], np.uint8)
    dib = struct.pack("<IiiHHIIiiII", 108, 2, -2, 1, 32, compression, pixels.size, 0, 0, 0, 0)
    dib += struct.pack("<IIII", 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000) + bytes(52)
    encoded = struct.pack("<2sIHHI", b"BM", 14 + len(dib) + pixels.size, 0, 0, 14 + len(dib))
    encoded += dib + pixels[:, :, [2, 1, 0, 3]].tobytes()
    path = tmp_path / "alpha.bmp"
    path.write_bytes(encoded)
    value = vane.ImageFile(str(path), "image/bmp")
    if compression == 6:
        for query, parameter in (
            ("SELECT decode_image($1,mode=>NULL)", encoded),
            ("SELECT decode_image_file($1)", value),
            ("SELECT image_file_metadata($1)", value),
        ):
            with pytest.raises(vane.InvalidInputException):
                image_connection.sql(query, params=[parameter]).fetchall()
        assert image_connection.sql(
            "SELECT decode_image($1,on_error=>'null'),decode_image_file($2,on_error=>'null')",
            params=[encoded, value],
        ).fetchone() == (None, None)
    else:
        decoded, file_decoded, metadata = image_connection.sql(
            "SELECT decode_image($1,mode=>NULL),decode_image_file($2),image_file_metadata($2)",
            params=[encoded, value],
        ).fetchone()
        assert_pixels(decoded, pixels)
        assert_pixels(file_decoded, pixels)
        assert metadata["mode"] == "RGBA"


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


def test_oversized_decode_is_never_suppressed(image_connection, tmp_path):
    def chunk(name, value):
        return struct.pack(">I", len(value)) + name + value + struct.pack(">I", zlib.crc32(name + value))

    encoded = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 100_000_001, 1, 8, 2, 0, 0, 0))
    encoded += chunk(b"IDAT", zlib.compress(b"\0")) + chunk(b"IEND", b"")
    with pytest.raises(vane.OutOfRangeException, match="pixel|limit"):
        image_connection.sql("SELECT decode_image($1,on_error=>'null')", params=[encoded]).fetchall()
    path = tmp_path / "oversized.png"
    path.write_bytes(encoded)
    value = vane.ImageFile(str(path))
    maximum = (1 << 64) - 1
    assert value.metadata(max_pixels=maximum).width == 100_000_001
    assert image_connection.sql(
        "SELECT (image_file_metadata($1,max_pixels=>$2::UBIGINT)).width", params=[value, maximum]
    ).fetchone() == (100_000_001,)
    with pytest.raises(vane.Error, match="pixel|limit"):
        image_connection.sql(
            "SELECT decode_image_file($1,on_error=>'null',max_pixels=>$2::UBIGINT,max_decoded_bytes=>$2::UBIGINT)",
            params=[value, maximum],
        ).fetchall()


@pytest.mark.parametrize("compressed", [b"invalid zlib stream", zlib.compress(b"\0")])
def test_corrupt_wide_png_is_a_content_error(image_connection, tmp_path, compressed):
    def chunk(name, value):
        return struct.pack(">I", len(value)) + name + value + struct.pack(">I", zlib.crc32(name + value))

    encoded = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 1, 16, 2, 0, 0, 0))
    encoded += chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    path = tmp_path / "corrupt-wide.png"
    path.write_bytes(encoded)
    for function, argument in (("decode_image", encoded), ("decode_image_file", vane.ImageFile(str(path)))):
        with pytest.raises(vane.InvalidInputException):
            image_connection.sql(f"SELECT {function}($1)", params=[argument]).fetchall()
        assert image_connection.sql(f"SELECT {function}($1,on_error=>'null')", params=[argument]).fetchone() == (None,)


@pytest.mark.parametrize(
    "message",
    [
        "png_create_read_struct returned NULL",
        "png_create_info_struct returned NULL",
        "Out of memory",
        "IDAT: insufficient memory",
        "invalid allocation size",
        "internal error",
        "unknown error",
    ],
)
def test_wide_png_preserves_codec_resource_and_unknown_failures(monkeypatch, tmp_path, message):
    imagecodecs = pytest.importorskip("imagecodecs")
    encoded = imagecodecs.png_encode(np.zeros((2, 3, 3), np.uint16))
    path = tmp_path / "wide.png"
    path.write_bytes(encoded)

    def fail(*args, **kwargs):
        raise imagecodecs.PngError(message)

    monkeypatch.setattr(imagecodecs, "png_decode", fail)
    with vane.connect() as con:
        for function, argument in (("decode_image", encoded), ("decode_image_file", vane.ImageFile(str(path)))):
            with pytest.raises(vane.Error, match=message):
                con.sql(f"SELECT {function}($1,on_error=>'null')", params=[argument]).fetchall()


def test_byte_decode_uses_a_separate_working_budget_and_keeps_the_payload_limit(image_connection):
    pil = pytest.importorskip("PIL.Image")
    encoded = io.BytesIO()
    with pil.new("RGB", (4800, 4800), (10, 20, 30)) as source:
        source.save(encoded, format="PNG")
    payload = encoded.getvalue()
    # RGB output is about 66 MiB, while its decode working set exceeds
    # 256 MiB. Generic Float32 output exceeds the separate 256 MiB payload cap.
    assert image_connection.sql("SELECT image_width(decode_image($1))", params=[payload]).fetchone() == (4800,)
    with pytest.raises(vane.OutOfRangeException, match="pixel or byte limit"):
        image_connection.sql(
            "SELECT image_width(decode_image($1,mode=>NULL,on_error=>'null'))", params=[payload]
        ).fetchall()


@pytest.mark.parametrize("length", [10, 12, 13, 18])
def test_gif_metadata_requires_the_screen_descriptor_and_global_palette(image_connection, tmp_path, length):
    encoded = (b"GIF89a" + struct.pack("<HHBBB", 2, 1, 0x80, 0, 0) + bytes(6))[:length]
    path = tmp_path / "truncated.gif"
    path.write_bytes(encoded)
    value = vane.ImageFile(str(path), "image/gif")
    with pytest.raises(vane.InvalidInputException):
        image_connection.sql("SELECT image_file_metadata($1)", params=[value]).fetchall()
    assert image_connection.sql(
        "SELECT decode_image($1,on_error=>'null'),decode_image_file($2,on_error=>'null')", params=[encoded, value]
    ).fetchone() == (None, None)


def test_grayscale_gif_retains_palette_metadata_and_decodes_as_rgba(image_connection, tmp_path):
    pil = pytest.importorskip("PIL.Image")
    pixels = np.arange(6, dtype=np.uint8).reshape(2, 3)
    encoded = io.BytesIO()
    with pil.fromarray(pixels) as source:
        source.save(encoded, format="GIF", optimize=False)
    with pil.open(io.BytesIO(encoded.getvalue())) as probe:
        assert probe.mode == "L"  # Pillow's identity-palette optimization.
    path = tmp_path / "grayscale.gif"
    path.write_bytes(encoded.getvalue())
    value = vane.ImageFile(str(path), "image/gif")
    decoded, file_decoded, metadata = image_connection.sql(
        "SELECT decode_image($1,mode=>NULL),decode_image_file($2),image_file_metadata($2)",
        params=[encoded.getvalue(), value],
    ).fetchone()
    expected = np.empty((2, 3, 4), np.uint8)
    expected[:, :, :3] = pixels[:, :, None]
    expected[:, :, 3] = 255
    assert_pixels(decoded, expected)
    assert_pixels(file_decoded, expected)
    assert metadata == {"width": 3, "height": 2, "format": "GIF", "mode": "P"}


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
@pytest.mark.parametrize("content_type", ["image/tiff", "image/x-tiff"])
def test_imagefile_decode_uses_logical_window_and_preserves_wide_pixels(
    image_connection, tmp_path, mode, channels, pixel_type, content_type
):
    pixels = pixels_for(mode, channels, pixel_type)
    encoded = image_connection.sql(
        "SELECT encode_image($1,'TIFF')", params=[vane.Value(pixels, vane.image_type(mode))]
    ).fetchone()[0]
    path = tmp_path / "window.bin"
    path.write_bytes(b"prefix" + encoded + b"suffix")
    value = vane.ImageFile(str(path), content_type, 6, len(encoded))
    metadata, decoded = image_connection.sql(
        "SELECT image_file_metadata($1),decode_image_file($1)", params=[value]
    ).fetchone()
    assert metadata == {"width": 5, "height": 3, "format": "TIFF", "mode": mode}
    assert_pixels(decoded, pixels)
    wrong = vane.ImageFile(str(path), "image/png", 6, len(encoded))
    assert image_connection.sql("SELECT decode_image_file($1,NULL,'null')", params=[wrong]).fetchone() == (None,)


@pytest.mark.parametrize(
    "option,limit",
    [
        ("max_decoded_bytes", 512 * 1024**2 + 1),
        ("max_decoded_bytes", (1 << 64) - 1),
        ("max_input_bytes", (1 << 64) - 1),
        ("max_pixels", (1 << 64) - 1),
    ],
)
def test_imagefile_decode_accepts_raised_budgets(image_connection, tmp_path, option, limit):
    tifffile = pytest.importorskip("tifffile")
    pixels = np.arange(18, dtype=np.uint16).reshape(2, 3, 3)
    path = tmp_path / "rgb16.tiff"
    tifffile.imwrite(path, pixels, photometric="rgb", metadata=None)
    value = vane.ImageFile(str(path), "image/tiff")
    decoded = image_connection.sql(
        f"SELECT decode_image_file($1,{option}=>$2::UBIGINT)", params=[value, limit]
    ).fetchone()[0]
    assert_pixels(decoded, pixels)


def test_imagefile_decode_can_raise_working_budget_above_default(image_connection, tmp_path):
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("imagecodecs")
    path = tmp_path / "large-rgb16.tiff"
    # The generic Float32 output fits 256 MiB, while source/converted pixels
    # and column storage together require 600 MB of working budget.
    tifffile.imwrite(
        path,
        np.zeros((4000, 5000, 3), np.uint16),
        photometric="rgb",
        metadata=None,
        compression="deflate",
    )
    value = vane.ImageFile(str(path), "image/tiff")
    with pytest.raises(vane.Error, match="max_decoded_bytes"):
        image_connection.sql("SELECT decode_image_file($1,on_error=>'null')", params=[value]).fetchall()
    assert image_connection.sql(
        "SELECT image_width(decode_image_file($1,max_decoded_bytes=>600000000))", params=[value]
    ).fetchone() == (5000,)


@pytest.mark.parametrize("image_format", ["TIFF", "PNG"])
def test_imagefile_working_pixels_can_exceed_output_byte_cap(image_connection, tmp_path, image_format):
    imagecodecs = pytest.importorskip("imagecodecs")
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / ("large-source." + image_format.lower())
    # 270 MB of source RGB16 pixels can produce a 180 MB generic L column.
    # The complete working set fits 1 GiB, while the default budget does not.
    pixels = np.zeros((4500, 10000, 3), np.uint16)
    if image_format == "TIFF":
        tifffile.imwrite(path, pixels, photometric="rgb", metadata=None, compression="deflate")
    else:
        path.write_bytes(imagecodecs.png_encode(pixels))
    del pixels
    assert path.stat().st_size < 256 * 1024 * 1024
    value = vane.ImageFile(str(path), "image/" + image_format.lower())
    with pytest.raises(vane.Error, match="max_decoded_bytes"):
        image_connection.sql("SELECT decode_image_file($1,mode=>'L',on_error=>'null')", params=[value]).fetchall()
    assert image_connection.sql(
        "SELECT image_width(decoded),image_height(decoded),image_channel(decoded),image_mode(decoded),"
        "decoded.data[1],decoded.data[-1] FROM "
        "(SELECT decode_image_file($1,mode=>'L',max_decoded_bytes=>1073741824) AS decoded)",
        params=[value],
    ).fetchone() == (10000, 4500, 1, 1, 0.0, 0.0)
    # A raised working budget does not permit an oversized output column.
    with pytest.raises(vane.Error, match="pixel or byte limit"):
        image_connection.sql(
            "SELECT decode_image_file($1,mode=>'RGB',max_decoded_bytes=>2147483648,on_error=>'null')",
            params=[value],
        ).fetchall()


@pytest.mark.parametrize(
    "mode,limit,channels,dtype",
    [(None, 126, 3, np.uint8), ("RGBA16", 180, 4, np.uint16), ("RGBA32F", 228, 4, np.float32)],
)
def test_imagefile_decode_budget_covers_converted_and_generic_storage(
    image_connection, tmp_path, mode, limit, channels, dtype
):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "rgb.tiff"
    tifffile.imwrite(path, np.arange(18, dtype=np.uint8).reshape(2, 3, 3), photometric="rgb", metadata=None)
    value = vane.ImageFile(str(path), "image/tiff")
    query = "SELECT decode_image_file($1,mode=>$2,on_error=>'null',max_decoded_bytes=>$3::UBIGINT)"
    with pytest.raises(vane.Error, match="max_decoded_bytes"):
        image_connection.sql(query, params=[value, mode, limit - 1]).fetchall()
    result = image_connection.sql(query, params=[value, mode, limit])
    assert result.types == [vane.image_type()]
    decoded = result.fetchone()[0]
    assert decoded.shape == (2, 3, channels)
    assert decoded.dtype == dtype


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


@pytest.mark.parametrize("mode,pixel_type", [("L", np.uint8), ("L16", np.uint16)])
def test_single_sample_planar_tiff_preserves_axes(image_connection, tmp_path, mode, pixel_type):
    pil = pytest.importorskip("PIL.Image")
    tifffile = pytest.importorskip("tifffile")
    pixels = np.arange(10, dtype=pixel_type).reshape(2, 5, 1)
    path = tmp_path / "single-sample-planar.tiff"
    # Pillow retains an explicit separate-planar tag even for one sample.
    pil.fromarray(pixels[:, :, 0]).save(path, format="TIFF", tiffinfo={284: 2})
    with tifffile.TiffFile(path) as tiff:
        assert tiff.pages[0].planarconfig == 2
        assert tiff.pages[0].asarray().shape == (2, 5)
    value = vane.ImageFile(str(path), "image/tiff")
    metadata, decoded = image_connection.sql(
        "SELECT image_file_metadata($1),decode_image_file($1)", params=[value]
    ).fetchone()
    assert metadata == {"width": 5, "height": 2, "format": "TIFF", "mode": mode}
    assert_pixels(decoded, pixels)
    decoded_bytes = image_connection.sql("SELECT decode_image($1,mode=>NULL)", params=[path.read_bytes()]).fetchone()[0]
    assert_pixels(decoded_bytes, pixels)


@pytest.mark.parametrize("channels,mode", [(1, "L16"), (3, "RGB16")])
@pytest.mark.parametrize("precision", [12, 16])
def test_native_jpeg_metadata_preserves_sample_precision(tmp_path, channels, mode, precision):
    imagecodecs = pytest.importorskip("imagecodecs")
    pixels = np.arange(15 * channels, dtype=np.uint16).reshape(3, 5, channels) * 32
    encoded = imagecodecs.jpeg_encode(
        pixels[:, :, 0] if channels == 1 else pixels,
        bitspersample=precision,
        lossless=precision == 16,
        subsampling="444",
    )
    path = tmp_path / "wide.jpg"
    path.write_bytes(encoded)
    with _connect("image") as con:
        value = vane.ImageFile(str(path), "image/jpeg")
        metadata = con.sql("SELECT image_file_metadata($1)", params=[value]).fetchone()[0]
        assert metadata == {"width": 5, "height": 3, "format": "JPEG", "mode": mode}
        # Header inspection does not require decoder support for the coding
        # process. FFmpeg supports these 12-bit DCT fixtures; its lossless
        # 16-bit JPEG support depends on the linked codec implementation.
        if precision == 12:
            decoded = con.sql("SELECT decode_image_file($1)", params=[value]).fetchone()[0]
            assert decoded.shape == pixels.shape
            assert decoded.dtype == np.uint16
            assert decoded.max() > 255
            decoded_bytes = con.sql("SELECT decode_image($1,mode=>NULL)", params=[encoded]).fetchone()[0]
            assert_pixels(decoded_bytes, decoded)


@pytest.mark.parametrize(
    "marker,precision,channels",
    [(0xC0, 12, 1), (0xC1, 0, 1), (0xC1, 16, 1), (0xC3, 1, 1), (0xC3, 17, 1), (0xC1, 12, 4)],
)
def test_native_jpeg_metadata_rejects_unsupported_sample_precision(tmp_path, marker, precision, channels):
    # Complete first frame headers with invalid precision or unsupported wide CMYK.
    encoded = (
        b"\xff\xd8\xff"
        + bytes([marker])
        + struct.pack(">HBHHB", 8 + 3 * channels, precision, 3, 5, channels)
        + b"".join(bytes([channel + 1, 0x11, 0]) for channel in range(channels))
    )
    path = tmp_path / "invalid-precision.jpg"
    path.write_bytes(encoded)
    with _connect("image") as con:
        value = vane.ImageFile(str(path), "image/jpeg")
        with pytest.raises(vane.InvalidInputException, match="JPEG sample precision"):
            con.sql("SELECT image_file_metadata($1)", params=[value]).fetchall()
        assert con.sql("SELECT decode_image_file($1,on_error=>'null')", params=[value]).fetchone() == (None,)


def test_imagefile_sql_named_options_and_defaults(image_connection, tmp_path):
    encoded = image_connection.sql("SELECT encode_image(image('abc'::BLOB,1,1,3,'RGB'),'PNG')").fetchone()[0]
    path = tmp_path / "named.png"
    path.write_bytes(encoded)
    value = vane.ImageFile(str(path))
    metadata = image_connection.sql("SELECT image_file_metadata($1,max_pixels=>1)", params=[value]).fetchone()[0]
    assert metadata == {"width": 1, "height": 1, "format": "PNG", "mode": "RGB"}
    decoded = image_connection.sql(
        "SELECT decode_image_file($1,on_error=>'null',max_pixels=>1,mode=>'RGBA')", params=[value]
    ).fetchone()[0]
    assert_pixels(decoded, np.array([[[97, 98, 99, 255]]], np.uint8))
    for query in (
        "SELECT image_file_metadata($1,max_bytes=>1)",
        "SELECT decode_image_file($1,on_error=>'null',max_input_bytes=>1)",
    ):
        with pytest.raises(vane.Error, match="max_bytes|max_input_bytes"):
            image_connection.sql(query, params=[value]).fetchall()
    assert image_connection.sql("SELECT decode_image_file($1,on_error=>NULL)", params=[value]).fetchone() == (None,)


def test_tiff_metadata_reads_first_directory_within_budget(image_connection, tmp_path):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "multipage.tiff"
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((128, 128), np.uint8), photometric="minisblack", metadata=None)
        writer.write(np.zeros((2, 3), np.uint8), photometric="minisblack", metadata=None)
    value = vane.ImageFile(str(path))
    assert path.stat().st_size > 1024
    metadata = image_connection.sql("SELECT image_file_metadata($1,max_bytes=>1024)", params=[value]).fetchone()[0]
    assert metadata == {"width": 128, "height": 128, "format": "TIFF", "mode": "L"}
    assert value.metadata(max_bytes=1024).width == 128
    decoded = image_connection.sql("SELECT decode_image($1,mode=>NULL)", params=[path.read_bytes()]).fetchone()[0]
    assert_pixels(decoded, np.zeros((128, 128, 1), np.uint8))


@pytest.mark.parametrize("window", ["directory", "tag_array"])
@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_tiff_metadata_budget_exhaustion_retains_limit_error(image_connection, tmp_path, window, byteorder):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "metadata-budget.tiff"
    tifffile.imwrite(path, np.ones((2, 3, 4), np.uint16), photometric="rgb", metadata=None, byteorder=byteorder)
    with tifffile.TiffFile(path) as tiff:
        limit = 8 if window == "directory" else tiff.pages[0].tags["BitsPerSample"].valueoffset + 2
    value = vane.ImageFile(str(path), "image/tiff")
    with pytest.raises(vane.ImageFileLimitError, match=f"max_bytes={limit}"):
        value.metadata(max_bytes=limit)
    with pytest.raises(vane.Error, match="max_bytes|read byte budget"):
        image_connection.sql("SELECT image_file_metadata($1,max_bytes=>$2::UBIGINT)", params=[value, limit]).fetchall()
    assert value.metadata().mode == "RGBA16"
    assert image_connection.sql("SELECT image_file_metadata($1)", params=[value]).fetchone()[0]["mode"] == "RGBA16"
    # A complete FILE window containing the same short header is malformed,
    # not a caller budget failure: increasing max_bytes cannot complete it.
    path.write_bytes(path.read_bytes()[:limit])
    with pytest.raises(vane.ImageFileFormatError):
        value.metadata()


@pytest.mark.parametrize("window", ["directory", "tag_array"])
@pytest.mark.parametrize("bigtiff,byteorder", [(False, "<"), (False, ">"), (True, "<"), (True, ">")])
def test_tiff_metadata_offsets_beyond_window_retain_limit_error(duckdb_cursor, tmp_path, window, bigtiff, byteorder):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "distant-metadata.tiff"
    tifffile.imwrite(
        path,
        np.ones((2, 3, 4), np.uint16),
        photometric="rgb",
        metadata=None,
        bigtiff=bigtiff,
        byteorder=byteorder,
        rowsperstrip=1,
    )
    data = bytearray(path.read_bytes())
    with tifffile.TiffFile(path) as tiff:
        page = tiff.pages[0]
        offset_width = 8 if bigtiff else 4
        count_width, entry_width = (8, 20) if bigtiff else (2, 12)
        if window == "directory":
            source = page.offset
            size = count_width + len(page.tags) * entry_width + offset_width
            pointer = 8 if bigtiff else 4
        else:
            tag = page.tags["StripOffsets" if bigtiff else "BitsPerSample"]
            source, size = tag.valueoffset, tag.valuebytecount
            pointer = tag.offset + entry_width - offset_width
        payload = data[source : source + size]
    # Move the required metadata after a gap, leaving the whole initial IFD
    # within the budget. The parser otherwise skips it on a filesize check
    # without attempting an out-of-window read.
    limit = len(data)
    target = len(data) + 512
    data[pointer : pointer + offset_width] = target.to_bytes(offset_width, "little" if byteorder == "<" else "big")
    data.extend(bytes(512) + payload)
    path.write_bytes(data)
    value = vane.ImageFile(str(path), "image/tiff")
    with pytest.raises(vane.ImageFileLimitError, match=f"max_bytes={limit}"):
        value.metadata(max_bytes=limit)
    with pytest.raises(vane.Error, match=f"max_bytes={limit}"):
        duckdb_cursor.sql("SELECT image_file_metadata($1,max_bytes=>$2::UBIGINT)", params=[value, limit]).fetchall()
    assert value.metadata().mode == "RGBA16"


@pytest.mark.parametrize("bigtiff,byteorder", [(False, "<"), (False, ">"), (True, "<"), (True, ">")])
def test_tiff_metadata_accepts_exact_window_boundary(image_connection, tmp_path, bigtiff, byteorder):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "exact-metadata.tiff"
    tifffile.imwrite(
        path,
        np.ones((2, 3, 4), np.uint16),
        photometric="rgb",
        metadata=None,
        software=False,
        align=1,
        bigtiff=bigtiff,
        byteorder=byteorder,
    )
    with tifffile.TiffFile(path) as tiff:
        page = tiff.pages[0]
        layout = tiff.tiff
        limit = max(
            page.offset + layout.tagnosize + len(page.tags) * layout.tagsize + layout.offsetsize,
            *(tag.valueoffset + tag.valuebytecount for tag in page.tags.values()),
        )
    encoded = path.read_bytes()
    assert limit < len(encoded)
    path.write_bytes(b"prefix" + encoded + b"suffix")
    value = vane.ImageFile(str(path), "image/tiff", 6, len(encoded))
    assert value.metadata(max_bytes=limit).mode == "RGBA16"
    assert (
        image_connection.sql("SELECT image_file_metadata($1,max_bytes=>$2::UBIGINT)", params=[value, limit]).fetchone()[
            0
        ]["mode"]
        == "RGBA16"
    )
    with pytest.raises(vane.ImageFileLimitError, match=f"max_bytes={limit - 1}"):
        value.metadata(max_bytes=limit - 1)
    with pytest.raises(vane.Error, match="max_bytes|read byte budget"):
        image_connection.sql(
            "SELECT image_file_metadata($1,max_bytes=>$2::UBIGINT)", params=[value, limit - 1]
        ).fetchall()


@pytest.mark.parametrize("window", ["directory", "tag_array", "strip_byte_counts"])
@pytest.mark.parametrize("bigtiff,byteorder", [(False, "<"), (False, ">"), (True, "<"), (True, ">")])
def test_tiff_offsets_beyond_logical_eof_are_content_errors(image_connection, tmp_path, window, bigtiff, byteorder):
    tifffile = pytest.importorskip("tifffile")
    output = io.BytesIO()
    tifffile.imwrite(
        output,
        np.ones((5, 3, 4), np.uint16),
        photometric="rgb",
        metadata=None,
        bigtiff=bigtiff,
        byteorder=byteorder,
        rowsperstrip=1,
    )
    data = bytearray(output.getvalue())
    with tifffile.TiffFile(io.BytesIO(data)) as tiff:
        offset_width = tiff.tiff.offsetsize
        pointer = (
            (8 if bigtiff else 4)
            if window == "directory"
            else (
                tiff.pages[0].tags["StripOffsets" if window == "tag_array" else "StripByteCounts"].offset
                + tiff.tiff.tagsize
                - offset_width
            )
        )
    target = len(data) + 64
    data[pointer : pointer + offset_width] = target.to_bytes(offset_width, "little" if byteorder == "<" else "big")
    path = tmp_path / "bad-offset-in-file-view.bin"
    path.write_bytes(b"prefix" + data + bytes(1024))
    value = vane.ImageFile(str(path), "image/tiff", 6, len(data))
    for limit in (len(data) - 1, len(data)):
        with pytest.raises(vane.ImageFileFormatError, match="logical FILE size"):
            value.metadata(max_bytes=limit)
        with pytest.raises(vane.InvalidInputException, match="TIFF|logical FILE size|header"):
            image_connection.sql(
                "SELECT image_file_metadata($1,max_bytes=>$2::UBIGINT)", params=[value, limit]
            ).fetchall()


def test_tiff_metadata_strip_arrays_respect_read_budget(image_connection, tmp_path):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "strip-arrays.tiff"
    tifffile.imwrite(
        path,
        np.ones((2048, 3, 4), np.uint16),
        photometric="rgb",
        metadata=None,
        bigtiff=True,
        rowsperstrip=1,
    )
    value = vane.ImageFile(str(path), "image/tiff")
    with pytest.raises(vane.ImageFileLimitError, match="max_bytes=1024"):
        value.metadata(max_bytes=1024)
    with pytest.raises(vane.Error, match="max_bytes|read byte budget"):
        image_connection.sql("SELECT image_file_metadata($1,max_bytes=>1024)", params=[value]).fetchall()
    assert image_connection.sql("SELECT image_file_metadata($1)", params=[value]).fetchone()[0]["height"] == 2048


@pytest.mark.parametrize("on_error", ["raise", "null"])
def test_image_file_encoded_hard_limit_precedes_header_parsing(image_connection, tmp_path, on_error):
    path = tmp_path / "oversized-invalid-image.bin"
    logical_size = 256 * 1024**2 + 1
    with path.open("wb") as stream:
        stream.write(b"not an image")
        stream.truncate(logical_size)
    value = vane.ImageFile(str(path))
    with pytest.raises(vane.Error, match="input byte limit|256 MiB"):
        image_connection.sql(
            "SELECT decode_image_file($1,on_error=>$2,max_input_bytes=>$3::UBIGINT,max_decoded_bytes=>$3::UBIGINT)",
            params=[value, on_error, (1 << 64) - 1],
        ).fetchall()


@pytest.mark.parametrize("header", [b"II*\0" + bytes(4), b"II+\0\x08\0\0\0" + bytes(8), b"II+\0\x04\0\0\0" + bytes(8)])
def test_tiff_invalid_complete_header_is_not_a_budget_error(tmp_path, header):
    path = tmp_path / "invalid-header.tiff"
    path.write_bytes(header + bytes(64))
    with pytest.raises(vane.ImageFileFormatError):
        vane.ImageFile(str(path)).metadata(max_bytes=len(header))


@pytest.mark.parametrize("compression", ["deflate", "jpeg", "lzw"])
def test_corrupt_tiff_strips_follow_content_error_policy(image_connection, tmp_path, compression):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "compressed.tiff"
    tifffile.imwrite(
        path, np.zeros((16, 16), np.uint8), photometric="minisblack", compression=compression, metadata=None
    )
    with tifffile.TiffFile(path) as tiff:
        offset, count = tiff.pages[0].dataoffsets[0], tiff.pages[0].databytecounts[0]
    # Confirm that the backend supports this compression before corrupting it.
    assert image_connection.sql("SELECT decode_image($1)", params=[path.read_bytes()]).fetchone()[0].shape == (
        16,
        16,
        3,
    )
    data = bytearray(path.read_bytes())
    data[offset : offset + count] = bytes(count)
    path.write_bytes(data)
    for function, argument in (("decode_image", bytes(data)), ("decode_image_file", vane.ImageFile(str(path)))):
        with pytest.raises(vane.InvalidInputException):
            image_connection.sql(f"SELECT {function}($1)", params=[argument]).fetchall()
        assert image_connection.sql(f"SELECT {function}($1,on_error=>'null')", params=[argument]).fetchone() == (None,)


@pytest.mark.parametrize(
    "failure", ["memory", "import", "runtime", "zlib_memory", "deflate_alloc", "jpeg_memory", "imcd_alloc"]
)
def test_tiff_decode_preserves_system_and_codec_allocation_failures(monkeypatch, failure):
    tifffile = pytest.importorskip("tifffile")
    imagecodecs = pytest.importorskip("imagecodecs")
    encoded = io.BytesIO()
    tifffile.imwrite(encoded, np.zeros((2, 3), np.uint8), photometric="minisblack", metadata=None)
    errors = {
        "memory": MemoryError("allocation"),
        "import": ImportError("dependency"),
        "runtime": RuntimeError("unexpected failure"),
        "zlib_memory": imagecodecs.ZlibError("uncompress", -4),
        "deflate_alloc": imagecodecs.DeflateError("libdeflate_alloc_decompressor", "NULL"),
        "jpeg_memory": imagecodecs.Jpeg8Error("Insufficient memory (case 0)"),
        "imcd_alloc": imagecodecs.LzwError("imcd_lzw_new", None),
    }

    def fail(*args, **kwargs):
        raise errors[failure]

    monkeypatch.setattr(tifffile.TiffPage, "asarray", fail)
    with vane.connect() as con, pytest.raises(vane.Error):
        con.sql("SELECT decode_image($1,on_error=>'null')", params=[encoded.getvalue()]).fetchall()


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
    assert con.sql("SELECT hash_identity(value) FROM hash_values").types == [dtype]
    assert con.sql("SELECT hash_identity(value) FROM hash_values").to_arrow_table().column(0).type == pa.binary(2)
    con.execute("CREATE TABLE hashes AS SELECT * FROM hash_values")
    assert con.table("hashes").to_arrow_table().column(0).type == pa.binary(2)


@pytest.mark.parametrize("left_width,right_width", [(2, None), (2, 1), (2, 2), (0, None), (0, 0), (0, 1)])
@pytest.mark.parametrize("reverse", [False, True])
def test_arrow_binary_common_type_widens_mixed_widths(duckdb_cursor, left_width, right_width, reverse):
    con = duckdb_cursor
    left = b"a" * left_width
    right = left if right_width is None else b"a" * right_width
    left_type, right_type = pa.binary(left_width), pa.binary() if right_width is None else pa.binary(right_width)
    con.register("fixed_side", pa.table({"id": [1, 2], "v": pa.array([left, None], type=left_type)}))
    con.register(
        "other_side",
        pa.table(
            {"id": [1, 2, 3], "v": pa.array([right, None, b"c" if right_width is None else right], type=right_type)}
        ),
    )
    first, second = ("other_side", "fixed_side") if reverse else ("fixed_side", "other_side")
    assert con.sql(f"SELECT count(*) FROM {first} a JOIN {second} b ON a.v=b.v").fetchone() == (
        (1 if right_width is None else 2) if left == right else 0,
    )
    expected_type = left_type if left_width == right_width else pa.binary()
    union = con.sql(f"SELECT v FROM {first} UNION ALL SELECT v FROM {second}").to_arrow_table()
    assert union.column(0).type == expected_type
    first_values = [right, None, b"c" if right_width is None else right] if reverse else [left, None]
    second_values = [left, None] if reverse else [right, None, b"c" if right_width is None else right]
    assert union.column(0).to_pylist() == first_values + second_values
    conditional = con.sql(
        f"SELECT CASE WHEN a.id=1 THEN a.v ELSE b.v END AS v FROM {first} a JOIN {second} b USING(id) ORDER BY id"
    ).to_arrow_table()
    assert conditional.column(0).type == expected_type
    assert conditional.column(0).to_pylist() == [right if reverse else left, None]


def test_zero_width_fixed_binary_arrow_cast_and_udf(image_connection):
    con = image_connection
    dtype = vane.sqltype("FIXEDBINARY(0)")
    chunks = [pa.array([b"", None, b""], type=pa.binary(0)).slice(1), pa.array([None, b""], type=pa.binary(0))]
    table = pa.table({"id": range(4), "value": pa.chunked_array(chunks)})
    con.register("zero_width", table)
    assert con.table("zero_width").types == [vane.sqltype("BIGINT"), dtype]
    expected = [None, b"", None, b""]
    assert con.sql("SELECT value FROM zero_width ORDER BY id").to_arrow_table().column(0).to_pylist() == expected

    @vane.func.batch(return_dtype=dtype)
    def identity(values):
        assert values.type == pa.binary(0)
        return values

    vane.attach_function(identity, connection=con, alias="zero_identity", parameters=[dtype])
    result = con.sql("SELECT zero_identity(value) AS value FROM zero_width ORDER BY id").to_arrow_table()
    assert result.column(0).type == pa.binary(0)
    assert result.column(0).to_pylist() == expected
    con.execute("CREATE TABLE zero_values AS SELECT * FROM zero_width")
    assert con.sql("SELECT count(DISTINCT value),count(value) FROM zero_values").fetchone() == (1, 2)
    assert con.sql("SELECT ''::BLOB::FIXEDBINARY(0),TRY_CAST('a'::BLOB AS FIXEDBINARY(0))").fetchone() == (b"", None)
    with pytest.raises(vane.InvalidInputException, match="exactly 0 bytes"):
        con.sql("SELECT 'a'::BLOB::FIXEDBINARY(0)").fetchall()


@pytest.mark.parametrize("width", [0, 2])
def test_fixed_binary_udf_validates_width_from_arrow_declaration(image_connection, width):
    @vane.func.batch(return_dtype=pa.binary(width))
    def wrong_width(values):
        return pa.array([b"x"] * len(values), type=pa.binary())

    vane.attach_function(
        wrong_width, connection=image_connection, alias="wrong_width", parameters=[vane.sqltype("BIGINT")]
    )
    assert image_connection.sql("SELECT wrong_width(1)").types == [vane.sqltype(f"FIXEDBINARY({width})")]
    with pytest.raises(vane.Error, match="length|bytes|size|width|cast"):
        image_connection.sql("SELECT wrong_width(1)").fetchall()


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
