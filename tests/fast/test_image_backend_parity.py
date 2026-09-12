# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Same-input regressions for codec fidelity and optional antialias filtering."""

import io

import numpy as np
import pytest

import vane
from tests.fast.test_image_modes import MODES
from tests.fast.test_native_media_extensions import _connect


@pytest.fixture(params=["python", "native"])
def connection(request):
    with _connect("image") if request.param == "native" else vane.connect() as con:
        yield con


def _encoded(pixels, image_format, mode=None, **options):
    Image = pytest.importorskip("PIL.Image")

    with (
        Image.fromarray(pixels)
        if mode is None
        else Image.frombytes(mode, pixels.shape[1::-1], pixels.tobytes()) as image
    ):
        with io.BytesIO() as stream:
            image.save(stream, image_format, **options)
            return stream.getvalue()


def _reference(encoded, mode):
    Image = pytest.importorskip("PIL.Image")

    with Image.open(io.BytesIO(encoded)) as image, image.convert(mode) as converted:
        pixels = np.asarray(converted).copy()
        return pixels[:, :, None] if pixels.ndim == 2 else pixels


@pytest.mark.usefixtures("ray_query")
def test_bmp_preserves_every_gray_level(connection, tmp_path):
    source = np.tile(np.arange(256, dtype=np.uint8), (3, 1))
    encoded = _encoded(source, "BMP")
    path = tmp_path / "gray.bmp"
    path.write_bytes(encoded)
    actual, file_actual, rgb = connection.sql(
        "SELECT decode_image($1,mode=>NULL),decode_image_file($2),decode_image($1)",
        params=[encoded, vane.ImageFile(str(path))],
    ).fetchone()
    np.testing.assert_array_equal(actual, source[:, :, None])
    np.testing.assert_array_equal(file_actual, actual)
    np.testing.assert_array_equal(rgb, np.repeat(actual, 3, axis=2))


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("mode", ["L", "RGB", "CMYK"])
@pytest.mark.parametrize("subsampling", [0, 1, 2])
@pytest.mark.parametrize("progressive", [False, True])
def test_jpeg_decode_matches_reference(connection, tmp_path, mode, subsampling, progressive):
    channels = {"L": 1, "RGB": 3, "CMYK": 4}[mode]
    source = np.random.default_rng(731).integers(0, 256, (19, 23, channels), dtype=np.uint8)
    encoded = _encoded(source, "JPEG", mode, quality=91, subsampling=subsampling, progressive=progressive)
    expected = _reference(encoded, "L" if mode == "L" else "RGB")
    path = tmp_path / "jpeg.bin"
    path.write_bytes(b"prefix" + encoded + b"suffix")
    value = vane.ImageFile(str(path), "image/jpeg", 6, len(encoded))
    actual, file_actual = connection.sql(
        "SELECT decode_image($1,mode=>NULL),decode_image_file($2)", params=[encoded, value]
    ).fetchone()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(file_actual, expected)


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("polarity", ["inverted", "ordinary"])
@pytest.mark.parametrize("progressive", [False, True])
def test_cmyk_jpeg_without_adobe_marker_matches_pillow(connection, tmp_path, polarity, progressive):
    Image = pytest.importorskip("PIL.Image")

    source = np.tile(np.array([20, 60, 100, 40], dtype=np.uint8), (7, 11, 1))
    # Pillow's encoder inverts CMYK samples. Invert its input to also exercise
    # ordinary-polarity samples, which its decoder still treats as inverted.
    if polarity == "ordinary":
        source = 255 - source
    encoded = _encoded(source, "JPEG", "CMYK", quality=100, subsampling=0, progressive=progressive)
    with Image.open(io.BytesIO(encoded)) as image:
        assert image.info["adobe_transform"] == 0  # Direct CMYK, not YCCK.
    offset = 2
    while offset + 4 <= len(encoded):
        assert encoded[offset] == 0xFF
        marker = encoded[offset + 1]
        assert marker not in (0xDA, 0xD9), "expected Adobe APP14 before the first scan"
        length = int.from_bytes(encoded[offset + 2 : offset + 4], "big")
        assert length >= 2 and offset + 2 + length <= len(encoded)
        if marker == 0xEE and encoded[offset + 4 : offset + 9] == b"Adobe":
            markerless = encoded[:offset] + encoded[offset + 2 + length :]
            break
        offset += 2 + length
    else:
        pytest.fail("Pillow did not write an Adobe APP14 marker")

    with Image.open(io.BytesIO(markerless)) as image:
        assert image.mode == "CMYK"
        assert "adobe" not in image.info
    expected = _reference(encoded, "RGB")
    expected_color = [198, 164, 131] if polarity == "inverted" else [3, 9, 16]
    np.testing.assert_array_equal(expected, np.broadcast_to(expected_color, expected.shape))
    np.testing.assert_array_equal(_reference(markerless, "RGB"), expected)

    path = tmp_path / "markerless-jpeg.bin"
    path.write_bytes(b"prefix" + markerless + b"suffix")
    value = vane.ImageFile(str(path), "image/jpeg", 6, len(markerless))
    actual, inferred, file_actual, metadata = connection.sql(
        "SELECT decode_image($1),decode_image($1,mode=>NULL),decode_image_file($2),image_file_metadata($2)",
        params=[markerless, value],
    ).fetchone()
    for pixels in (actual, inferred, file_actual):
        np.testing.assert_array_equal(pixels, expected)
    assert metadata == {"width": 11, "height": 7, "format": "JPEG", "mode": "CMYK"}


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_jpeg_encode_uses_quality_95_and_444(connection, mode):
    Image = pytest.importorskip("PIL.Image")

    source = np.random.default_rng(117).integers(0, 256, (37, 41, 1 if mode == "L" else 3), dtype=np.uint8)
    expected = _encoded(source, "JPEG", mode, quality=95, subsampling=0)
    actual = connection.sql(
        "SELECT encode_image($1,'JPEG')", params=[vane.Value(source, vane.image_type(mode))]
    ).fetchone()[0]
    with Image.open(io.BytesIO(actual)) as encoded, Image.open(io.BytesIO(expected)) as reference:
        assert encoded.mode == mode
        assert encoded.quantization == reference.quantization
        assert encoded.layer == reference.layer
    np.testing.assert_array_equal(_reference(actual, mode), _reference(expected, mode))


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("colors", [1, 2, 16, 256])
def test_gif_preserves_low_color_images(connection, colors):
    # Include the original RGB332 regression color and distinct colors sharing
    # a 5-bit histogram bin, which must still take the exact-color path.
    palette = np.column_stack((np.arange(colors), np.arange(colors) // 2, np.arange(colors) // 3)).astype(np.uint8)
    palette[0] = 127
    source = palette[np.arange(1025) % colors].reshape(25, 41, 3)
    actual = connection.sql(
        "SELECT encode_image($1,'GIF')", params=[vane.Value(source, vane.image_type("RGB"))]
    ).fetchone()[0]
    np.testing.assert_array_equal(_reference(actual, "RGB"), source)


@pytest.mark.usefixtures("ray_query")
def test_quantized_gif_backends_agree_and_improve_rgb332():
    pytest.importorskip("PIL.Image")
    source = np.random.default_rng(991).integers(0, 256, (79, 83, 3), dtype=np.uint8)
    value = vane.Value(source, vane.image_type("RGB"))
    with vane.connect() as python, _connect("image") as native:
        outputs = [con.sql("SELECT encode_image($1,'GIF')", params=[value]).fetchone()[0] for con in (python, native)]
    actual = _reference(outputs[0], "RGB")
    np.testing.assert_array_equal(actual, _reference(outputs[1], "RGB"))
    rgb332 = (source.astype(np.int32) >> [5, 5, 6]) * 255 // [7, 7, 3]
    assert np.mean((actual.astype(float) - source) ** 2) < np.mean((rgb332 - source) ** 2)


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
@pytest.mark.parametrize("lossless", [False, True])
@pytest.mark.parametrize("animated", [False, True])
def test_webp_bytes_file_and_metadata(connection, tmp_path, mode, lossless, animated):
    Image = pytest.importorskip("PIL.Image")

    source = np.random.default_rng(81).integers(0, 256, (17, 23, len(mode)), dtype=np.uint8)
    with Image.fromarray(source) as first, Image.fromarray(255 - source) as second, io.BytesIO() as stream:
        first.save(
            stream,
            "WEBP",
            lossless=lossless,
            exact=True,
            save_all=animated,
            append_images=[second] if animated else [],
            duration=100,
        )
        encoded = stream.getvalue()
    path = tmp_path / "webp.bin"
    path.write_bytes(b"prefix" + encoded + b"suffix")
    value = vane.ImageFile(str(path), "image/webp", 6, len(encoded))
    metadata, pixels, file_pixels = connection.sql(
        "SELECT image_file_metadata($1),decode_image($2,mode=>NULL),decode_image_file($1)", params=[value, encoded]
    ).fetchone()
    assert metadata == {"width": 23, "height": 17, "format": "WEBP", "mode": mode}
    np.testing.assert_array_equal(pixels, _reference(encoded, mode))
    np.testing.assert_array_equal(file_pixels, pixels)
    value_metadata = value.metadata()
    assert metadata == {field: getattr(value_metadata, field) for field in metadata}


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
@pytest.mark.parametrize("lossless", [False, True])
def test_webp_metadata_needs_only_headers(connection, tmp_path, monkeypatch, mode, lossless):
    plugin = pytest.importorskip("PIL.WebPImagePlugin")
    encoded = _encoded(np.full((17, 23, len(mode)), 127, np.uint8), "WEBP", lossless=lossless)
    path = tmp_path / "header.webp"
    path.write_bytes(b"prefix" + encoded + b"suffix")
    value = vane.ImageFile(str(path), "image/webp", 6, len(encoded))
    required = 25 if encoded[12:16] == b"VP8L" else 30

    def forbidden(*args, **kwargs):
        raise AssertionError("WebP metadata must not allocate a pixel decoder")

    monkeypatch.setattr(plugin._webp, "WebPAnimDecoder", forbidden)
    metadata = connection.sql(
        "SELECT image_file_metadata($1,max_bytes=>$2::UBIGINT)", params=[value, required]
    ).fetchone()[0]
    assert metadata == {"width": 23, "height": 17, "format": "WEBP", "mode": mode}
    value_metadata = value.metadata(max_bytes=required)
    assert metadata == {field: getattr(value_metadata, field) for field in metadata}
    for option in (f"max_bytes=>{required - 1}", "max_pixels=>1"):
        with pytest.raises(vane.Error, match="max_bytes|max_pixels|pixels"):
            connection.sql(f"SELECT image_file_metadata($1,{option})", params=[value]).fetchall()


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("option", ["max_pixels=>1", "max_decoded_bytes=>32"])
def test_webp_limits_precede_decoder_allocation(connection, tmp_path, monkeypatch, option):
    plugin = pytest.importorskip("PIL.WebPImagePlugin")
    encoded = _encoded(np.full((17, 23, 3), 127, np.uint8), "WEBP")
    path = tmp_path / "limited.webp"
    path.write_bytes(encoded)

    def forbidden(*args, **kwargs):
        raise AssertionError("WebP must validate resource limits before allocating its decoder")

    monkeypatch.setattr(plugin._webp, "WebPAnimDecoder", forbidden)
    with pytest.raises(vane.Error, match="max_pixels|pixels|max_decoded_bytes|byte"):
        connection.sql(
            f"SELECT decode_image_file($1,on_error=>'null',{option})", params=[vane.ImageFile(str(path))]
        ).fetchall()


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("damage", ["riff_size", "chunk_size", "extended_flags", "lossy_signature", "lossless_version"])
def test_webp_invalid_headers_remain_content_errors(connection, tmp_path, damage):
    channels = 4 if damage == "extended_flags" else 3
    encoded = bytearray(
        _encoded(np.full((17, 23, channels), 127, np.uint8), "WEBP", lossless=damage == "lossless_version")
    )
    if damage in ("riff_size", "chunk_size"):
        offset = 4 if damage == "riff_size" else 16
        encoded[offset : offset + 4] = b"\xff" * 4
    elif damage == "extended_flags":
        encoded[20] |= 1
    elif damage == "lossy_signature":
        encoded[23] = 0
    else:
        encoded[24] |= 0x20
    path = tmp_path / "invalid.webp"
    path.write_bytes(encoded)
    value = vane.ImageFile(str(path))
    with pytest.raises(vane.Error, match="WebP"):
        connection.sql("SELECT image_file_metadata($1)", params=[value]).fetchall()
    assert connection.sql(
        "SELECT decode_image_file($1,on_error=>'null'),decode_image($2,on_error=>'null')",
        params=[value, bytes(encoded)],
    ).fetchone() == (None, None)
    with pytest.raises(vane.ImageFileFormatError):
        value.metadata()


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("image_format", ["JPEG", "WEBP"])
def test_new_codecs_preserve_error_and_limit_contract(connection, tmp_path, image_format):
    encoded = _encoded(np.full((17, 23, 3), 127, np.uint8), image_format)
    path = tmp_path / "image.bin"
    path.write_bytes(encoded)
    value = vane.ImageFile(str(path))
    for option in ("max_pixels=>1", "max_decoded_bytes=>32"):
        with pytest.raises(vane.Error, match="max_pixels|pixels|max_decoded_bytes|byte"):
            connection.sql(f"SELECT decode_image_file($1,on_error=>'null',{option})", params=[value]).fetchall()
    for truncated in (encoded[:30], encoded[:-20]):
        assert connection.sql("SELECT decode_image($1,on_error=>'null')", params=[truncated]).fetchone() == (None,)
    assert connection.sql("SELECT decode_image(NULL)").fetchone() == (None,)


@pytest.mark.usefixtures("ray_query")
def test_antialias_downsampling_alpha_and_default(connection):
    source = np.array([[[0], [0], [0], [240]]], np.uint8)
    value = vane.Value(source, vane.image_type("L"))
    result = connection.sql("SELECT resize($1,1,1),resize($1,1,1,true)", params=[value])
    assert result.types == [vane.image_type("L", 1, 1)] * 2
    ordinary, filtered = result.fetchone()
    np.testing.assert_array_equal(ordinary, [[[0]]])
    np.testing.assert_array_equal(filtered, [[[50]]])
    source = np.array([[[255, 0, 0, 255], [0, 0, 255, 0]]], np.uint8)
    value = vane.Value(source, vane.image_type("RGBA"))
    for expression in (vane.resize(value, 1, 1, antialias=True), vane.lit(value).resize(1, 1, antialias=True)):
        np.testing.assert_array_equal(connection.sql("SELECT 1").select(expression).fetchone()[0], [[[255, 0, 0, 128]]])


@pytest.mark.usefixtures("ray_query")
def test_antialias_per_row_null_and_argument_validation(connection):
    value = vane.Value(np.array([[[0], [0], [0], [240]]], np.uint8), vane.image_type("L"))
    rows = connection.sql("SELECT resize($1,1,1,a) FROM (VALUES (true),(false),(NULL)) t(a)", params=[value]).fetchall()
    assert [None if row[0] is None else row[0].item() for row in rows] == [50, 0, None]
    for option in (1, "true", 0.5):
        with pytest.raises(TypeError, match="boolean"):
            vane.resize(value, 1, 1, antialias=option)
        with pytest.raises(vane.Error, match="boolean"):
            connection.sql("SELECT resize($1,1,1,$2)", params=[value, option]).fetchall()


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("mode,channels,dtype", MODES)
@pytest.mark.parametrize("shape", [(3, 5), (31, 4), (4, 31)])
def test_antialias_backends_agree(mode, channels, dtype, shape):
    source = np.random.default_rng(51).random((13, 17, channels))
    source = (source * (1 if dtype == np.float32 else np.iinfo(dtype).max)).astype(dtype)
    value = vane.Value(source, vane.image_type(mode))
    with vane.connect() as python, _connect("image") as native:
        outputs = [
            con.sql("SELECT resize($1,$2,$3,true)", params=[value, shape[1], shape[0]]).fetchone()[0]
            for con in (python, native)
        ]
    np.testing.assert_array_equal(*outputs)


@pytest.mark.usefixtures("ray_query")
def test_antialias_native_avoids_python(monkeypatch):
    import vane._image_operators as helpers

    def forbidden(*args, **kwargs):
        raise AssertionError("native antialias called the Python implementation")

    value = vane.Value(np.array([[[0], [0], [0], [240]]], np.uint8), vane.image_type("L"))
    with _connect("image") as con:
        con.execute("PRAGMA disable_optimizer")
        monkeypatch.setattr(helpers, "_resize_image", forbidden)
        relation = con.sql("SELECT resize($1,1,1,true)", params=[value])
        assert "native_resize" in relation.explain()
        np.testing.assert_array_equal(relation.fetchone()[0], [[[50]]])
