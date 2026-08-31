# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import importlib
import io
import threading

import pytest
from PIL import Image

import vane
from vane import _image_file


def _encoded_image(image_format: str, *, size: tuple[int, int] = (7, 5), color: str = "red") -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGB", size, color)
    try:
        image.save(buffer, format=image_format)
    except OSError as error:
        pytest.skip(f"Pillow build does not support {image_format}: {error}")
    finally:
        image.close()
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("image_format", "expected_mode", "expected_mime"),
    [
        ("PNG", "RGB", "image/png"),
        ("JPEG", "RGB", "image/jpeg"),
        ("WEBP", "RGB", "image/webp"),
        ("GIF", "P", "image/gif"),
    ],
)
def test_image_file_metadata_sql_and_python_value(
    duckdb_cursor,
    tmp_path,
    image_format,
    expected_mode,
    expected_mime,
):
    path = tmp_path / f"image.{image_format.lower()}"
    path.write_bytes(_encoded_image(image_format))
    value = vane.ImageFile(str(path), expected_mime)

    result_type, metadata, null_metadata = duckdb_cursor.execute(
        """
        SELECT
            typeof(image_file_metadata($1)),
            image_file_metadata($1),
            image_file_metadata(NULL::IMAGEFILE)
        """,
        [value],
    ).fetchone()

    assert result_type == 'STRUCT(width UINTEGER, height UINTEGER, format VARCHAR, "mode" VARCHAR)'
    assert metadata == {"width": 7, "height": 5, "format": image_format, "mode": expected_mode}
    assert null_metadata is None
    assert value.metadata(connection=duckdb_cursor) == vane.ImageMetadata(7, 5, image_format, expected_mode)


def test_image_file_metadata_facades(duckdb_cursor, tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(_encoded_image("PNG", size=(3, 2)))
    value = vane.ImageFile(str(path), "image/png")

    function_result = (
        duckdb_cursor.sql("SELECT 1")
        .select(vane.image_file_metadata(value, max_bytes=4096, max_pixels=6))
        .fetchone()[0]
    )
    method_result = duckdb_cursor.sql("SELECT 1").select(vane.image_file(value).image_file_metadata()).fetchone()[0]

    expected = {"width": 3, "height": 2, "format": "PNG", "mode": "RGB"}
    assert function_result == expected
    assert method_result == expected


def test_image_file_accepts_raw_jpeg2000_mime(duckdb_cursor, tmp_path):
    buffer = io.BytesIO()
    image = Image.new("L", (3, 2), 100)
    try:
        image.save(buffer, format="JPEG2000", no_jp2=True)
    except OSError as error:
        pytest.skip(f"Pillow build does not support JPEG 2000: {error}")
    finally:
        image.close()
    payload = buffer.getvalue()
    assert payload.startswith(b"\xff\x4f\xff\x51")
    path = tmp_path / "image.j2c"
    path.write_bytes(payload)
    value = vane.ImageFile(str(path), "image/j2c")

    assert value.metadata(connection=duckdb_cursor).format == "JPEG2000"
    assert duckdb_cursor.execute("SELECT image_file_metadata($1)", [value]).fetchone()[0]["mode"] == "L"
    decoded = value.decode(connection=duckdb_cursor)
    assert decoded.size == (3, 2)
    decoded.close()
    with pytest.raises(vane.ImageFileFormatError, match="detected MIME type"):
        vane.ImageFile(str(path), "image/jp2").metadata(connection=duckdb_cursor)


def test_image_file_accepts_precise_and_family_portable_anymap_mimes(duckdb_cursor, tmp_path):
    path = tmp_path / "image.pgm"
    path.write_bytes(b"P5\n2 1\n255\n\x00\xff")

    precise = vane.ImageFile(str(path), "image/x-portable-graymap")
    family = vane.ImageFile(str(path), "image/x-portable-anymap")
    assert precise.metadata(connection=duckdb_cursor).mode == "L"
    assert duckdb_cursor.execute("SELECT image_file_metadata($1)", [family]).fetchone()[0]["format"] == "PPM"
    with pytest.raises(vane.ImageFileFormatError, match="detected MIME type"):
        vane.ImageFile(str(path), "image/x-portable-pixmap").metadata(connection=duckdb_cursor)


def test_image_file_preserves_high_bit_depth_mode(duckdb_cursor, tmp_path):
    path = tmp_path / "high-depth.png"
    image = Image.new("I;16", (2, 3), 1000)
    try:
        image.save(path, format="PNG")
    finally:
        image.close()
    value = vane.ImageFile(str(path), "image/png")

    assert value.metadata(connection=duckdb_cursor) == vane.ImageMetadata(2, 3, "PNG", "I;16")
    decoded = value.decode(connection=duckdb_cursor)
    assert decoded.mode == "I;16"
    assert decoded.getpixel((0, 0)) == 1000
    decoded.close()


def test_image_file_metadata_and_decode_honor_logical_range(duckdb_cursor, tmp_path):
    payload = _encoded_image("PNG", size=(4, 3), color="blue")
    prefix = b"not-an-image-prefix"
    suffix = b"not-an-image-suffix"
    path = tmp_path / "ranged.bin"
    path.write_bytes(prefix + payload + suffix)
    value = vane.ImageFile(str(path), "image/png", len(prefix), len(payload))

    assert duckdb_cursor.execute("SELECT image_file_metadata($1)", [value]).fetchone()[0] == {
        "width": 4,
        "height": 3,
        "format": "PNG",
        "mode": "RGB",
    }
    assert value.metadata(connection=duckdb_cursor) == vane.ImageMetadata(4, 3, "PNG", "RGB")
    decoded = value.decode(connection=duckdb_cursor)
    assert decoded.size == (4, 3)
    assert decoded.mode == "RGB"
    assert decoded.getpixel((0, 0)) == (0, 0, 255)
    decoded.close()


def test_image_file_decode_returns_detached_image_and_converts_mode(duckdb_cursor, tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(_encoded_image("PNG", size=(2, 3)))
    value = vane.ImageFile(str(path), "image/png")

    decoded = value.decode("L", buffer_size=64, connection=duckdb_cursor)
    path.unlink()

    assert decoded.size == (2, 3)
    assert decoded.mode == "L"
    assert isinstance(decoded.getpixel((0, 0)), int)
    decoded.close()


def test_image_file_decode_uses_first_animated_frame(duckdb_cursor, tmp_path):
    path = tmp_path / "animated.gif"
    first = Image.new("RGB", (2, 2), "red")
    second = Image.new("RGB", (2, 2), "blue")
    try:
        first.save(path, format="GIF", save_all=True, append_images=[second], duration=10, loop=0)
    finally:
        first.close()
        second.close()

    decoded = vane.ImageFile(str(path), "image/gif").decode("RGB", connection=duckdb_cursor)

    assert decoded.getpixel((0, 0)) == (255, 0, 0)
    decoded.close()


def test_image_file_metadata_limits_are_enforced(duckdb_cursor, tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(_encoded_image("PNG", size=(4, 3)))
    value = vane.ImageFile(str(path), "image/png")

    with pytest.raises(vane.ImageFileLimitError, match="max_bytes=8"):
        value.metadata(max_bytes=8, connection=duckdb_cursor)
    with pytest.raises(vane.ImageFileLimitError, match="max_pixels=11"):
        value.metadata(max_pixels=11, connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match="max_bytes=8"):
        duckdb_cursor.execute("SELECT image_file_metadata($1, 8, 100)", [value]).fetchone()
    with pytest.raises(vane.InvalidInputException, match="max_pixels=11"):
        duckdb_cursor.execute("SELECT image_file_metadata($1, 1024, 11)", [value]).fetchone()


def test_image_file_per_call_pixel_limit_does_not_change_pillow_global_limit(
    duckdb_cursor,
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "image.png"
    path.write_bytes(_encoded_image("PNG", size=(4, 3)))
    value = vane.ImageFile(str(path), "image/png")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    assert value.metadata(max_pixels=12, connection=duckdb_cursor).width == 4
    assert Image.MAX_IMAGE_PIXELS == 1
    assert duckdb_cursor.execute("SELECT image_file_metadata($1, 1024, 12)", [value]).fetchone()[0]["height"] == 3
    assert Image.MAX_IMAGE_PIXELS == 1
    # TIFF repeats Pillow's bomb check while allocating its decode tile, so
    # this also verifies that the per-call context covers the complete load.
    decode_path = tmp_path / "image.tiff"
    decode_path.write_bytes(_encoded_image("TIFF", size=(4, 3)))
    decoded = vane.ImageFile(str(decode_path), "image/tiff").decode(
        max_pixels=12,
        connection=duckdb_cursor,
    )
    assert decoded.size == (4, 3)
    decoded.close()
    assert Image.MAX_IMAGE_PIXELS == 1


def test_image_file_pixel_limit_is_isolated_from_unrelated_pillow_threads(monkeypatch):
    payload = _encoded_image("PNG", size=(4, 3))
    entered = threading.Event()
    release = threading.Event()
    worker_sizes: list[tuple[int, int]] = []
    worker_errors: list[BaseException] = []
    original_check = Image._decompression_bomb_check
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    class BlockingBytesIO(io.BytesIO):
        def read(self, size=-1, /):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release the ImageFile reader")
            return super().read(size)

    def open_with_vane_limit():
        try:
            with _image_file._open_image_with_limit(Image, BlockingBytesIO(payload), max_pixels=12) as image:
                image.load()
                worker_sizes.append(image.size)
        except BaseException as error:
            worker_errors.append(error)

    worker = threading.Thread(target=open_with_vane_limit)
    worker.start()
    try:
        assert entered.wait(timeout=5)
        with pytest.raises(Image.DecompressionBombError):
            with Image.open(io.BytesIO(payload)) as unrelated:
                unrelated.load()
    finally:
        release.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert worker_errors == []
    assert worker_sizes == [(4, 3)]
    assert Image.MAX_IMAGE_PIXELS == 1
    assert Image._decompression_bomb_check is original_check


def test_image_file_decode_limits_are_enforced(duckdb_cursor, tmp_path):
    path = tmp_path / "image.png"
    payload = _encoded_image("PNG", size=(4, 3))
    path.write_bytes(payload)
    value = vane.ImageFile(str(path), "image/png")

    with pytest.raises(vane.ImageFileLimitError, match="max_input_bytes"):
        value.decode(max_input_bytes=len(payload) - 1, connection=duckdb_cursor)
    with pytest.raises(vane.ImageFileLimitError, match="max_pixels=11"):
        value.decode(max_pixels=11, connection=duckdb_cursor)
    for mode in (None, "LA", "YCbCr"):
        with pytest.raises(
            vane.ImageFileLimitError,
            match="requires up to 96 bytes, exceeding max_decoded_bytes=95",
        ):
            value.decode(mode, max_decoded_bytes=95, connection=duckdb_cursor)
        decoded = value.decode(mode, max_decoded_bytes=96, connection=duckdb_cursor)
        decoded.close()


@pytest.mark.parametrize(
    ("content_type", "message"),
    [("audio/mpeg", "contradicts"), ("image/jpeg", "detected MIME type")],
)
def test_image_file_rejects_contradictory_content_type(duckdb_cursor, tmp_path, content_type, message):
    path = tmp_path / "image.png"
    path.write_bytes(_encoded_image("PNG"))
    value = vane.ImageFile(str(path), content_type)

    with pytest.raises(vane.ImageFileFormatError, match=message):
        value.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.ImageFileFormatError, match=message):
        value.decode(connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match=message):
        duckdb_cursor.execute("SELECT image_file_metadata($1)", [value]).fetchone()


def test_image_file_classifies_invalid_media_but_propagates_io(duckdb_cursor, tmp_path):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    corrupt_value = vane.ImageFile(str(corrupt), "image/png")

    with pytest.raises(vane.ImageFileFormatError, match="supported encoded image"):
        corrupt_value.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.ImageFileFormatError, match="supported encoded image"):
        corrupt_value.decode(connection=duckdb_cursor)

    large_corrupt = tmp_path / "large-corrupt.png"
    large_corrupt.write_bytes(b"not an image" * 1024)
    # Once every allowed header byte has been consumed, more bytes could still
    # contain a late format marker; the bounded probe must report the budget,
    # not claim that it inspected the complete logical view.
    with pytest.raises(vane.ImageFileLimitError, match="max_bytes=1024"):
        vane.ImageFile(str(large_corrupt), "image/png").metadata(max_bytes=1024, connection=duckdb_cursor)

    missing = vane.ImageFile(str(tmp_path / "missing.png"), "image/png")
    with pytest.raises(vane.IOException):
        missing.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.IOException):
        missing.decode(connection=duckdb_cursor)
    with pytest.raises(vane.IOException):
        duckdb_cursor.execute("SELECT image_file_metadata($1)", [missing]).fetchone()


def test_image_file_metadata_requires_imagefile(duckdb_cursor):
    with pytest.raises(vane.BinderException, match="requires IMAGEFILE, not FILE"):
        duckdb_cursor.sql("SELECT image_file_metadata(file('memory://generic', NULL, NULL, NULL, NULL))")


@pytest.mark.parametrize(
    ("method", "kwargs", "error_type", "message"),
    [
        ("metadata", {"max_bytes": True}, TypeError, "max_bytes must be int"),
        ("metadata", {"max_bytes": 0}, ValueError, "greater than zero"),
        ("metadata", {"max_bytes": 64 * 1024 * 1024 + 1}, ValueError, "at most"),
        ("decode", {"mode": 1}, TypeError, "mode must be str or None"),
        ("decode", {"mode": "XYZ"}, ValueError, "unsupported image decode mode"),
        ("decode", {"max_pixels": 0}, ValueError, "greater than zero"),
    ],
)
def test_image_file_python_argument_validation(method, kwargs, error_type, message):
    value = vane.ImageFile("memory://not-opened")

    with pytest.raises(error_type, match=message):
        getattr(value, method)(**kwargs)


def test_image_file_optional_dependency_is_lazy(monkeypatch):
    original_import = importlib.import_module

    def fail_pillow(name, package=None):
        if name == "PIL.Image":
            raise ImportError("missing pillow")
        return original_import(name, package)

    monkeypatch.setattr(_image_file.importlib, "import_module", fail_pillow)

    value = vane.ImageFile("memory://not-opened")
    with pytest.raises(ImportError, match=r"vane-ai\[image\]"):
        value.metadata()
    with pytest.raises(ImportError, match=r"vane-ai\[image\]"):
        value.decode()
