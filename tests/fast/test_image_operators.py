# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import struct
import subprocess
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pytest

import vane
from tests.fast.test_native_media_extensions import _connect


@pytest.fixture(params=["python", "native"])
def image_connection(request):
    if request.param == "python":
        pytest.importorskip("PIL.Image")
    with _connect("image") if request.param == "native" else vane.connect() as con:
        yield con


def _pixels(mode):
    channels = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}[mode]
    return np.arange(4 * 5 * channels, dtype=np.uint8).reshape(4, 5, channels)


def _expected_crop(image, bbox):
    x, y, width, height = bbox
    result = np.zeros((height, width, image.shape[2]), dtype=np.uint8)
    for row in range(height):
        for column in range(width):
            if 0 <= y + row < image.shape[0] and 0 <= x + column < image.shape[1]:
                result[row, column] = image[y + row, x + column]
    return result


def _check_png(encoded, expected, mode):
    assert encoded[:8] == b"\x89PNG\r\n\x1a\n"
    offset, names = 8, []
    while offset < len(encoded):
        size = struct.unpack_from(">I", encoded, offset)[0]
        name = encoded[offset + 4 : offset + 8]
        data = encoded[offset + 8 : offset + 8 + size]
        checksum = struct.unpack_from(">I", encoded, offset + 8 + size)[0]
        assert checksum == zlib.crc32(name + data)
        names.append(name)
        offset += size + 12
    assert offset == len(encoded)
    assert names[0] == b"IHDR" and names[-1] == b"IEND" and b"IDAT" in names
    pil = pytest.importorskip("PIL.Image")
    with pil.open(io.BytesIO(encoded)) as image:
        assert image.mode == mode
        restored = np.asarray(image).reshape(expected.shape)
        np.testing.assert_array_equal(restored, expected)


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
@pytest.mark.parametrize("form", ["generic", "mode", "fixed"])
def test_crop_and_png_python_expression_and_sql(image_connection, mode, form):
    con = image_connection
    image = _pixels(mode)
    dtype = (
        vane.image_type()
        if form == "generic"
        else vane.image_type(mode)
        if form == "mode"
        else vane.image_type(mode, 4, 5)
    )
    value = vane.Value(image, dtype)
    source = con.sql("SELECT $1 AS image", params=[value])
    for box in [(1, 1, 3, 2), (-1, -2, 4, 4), (4, 3, 3, 2), (99, 99, 2, 3)]:
        expected = _expected_crop(image, box)
        expressions = [vane.crop(vane.col("image"), box), vane.col("image").crop(bbox=box)]
        for expression in expressions:
            relation = source.select(expression.alias("cropped"))
            assert relation.types == [vane.image_type(None if form == "generic" else mode)]
            actual = relation.fetchone()[0]
            np.testing.assert_array_equal(actual, expected)
            assert actual.flags.c_contiguous and actual.flags.writeable
            actual[:] = 255
            for encoded in [
                relation.select(vane.encode_image(vane.col("cropped"), vane.ImageFormat.PNG)).fetchone()[0],
                relation.select(vane.col("cropped").encode_image(image_format="png")).fetchone()[0],
            ]:
                _check_png(encoded, expected, mode)
        cropped, encoded = con.execute(
            "SELECT crop($1, $2), encode_image(crop($1, $2), 'PnG')", [value, list(box)]
        ).fetchone()
        np.testing.assert_array_equal(cropped, expected)
        _check_png(encoded, expected, mode)
    np.testing.assert_array_equal(image, _pixels(mode))


def test_crop_accepts_strided_arrays_and_pil(image_connection):
    pil = pytest.importorskip("PIL.Image")
    original = _pixels("RGB")
    for value in (original[::-1, ::2], pil.fromarray(original)):
        expected = np.asarray(value)
        result = image_connection.sql("SELECT 1").select(vane.crop(value, (0, 0, 2, 2))).fetchone()[0]
        np.testing.assert_array_equal(result, expected[:2, :2])


@pytest.mark.parametrize("height,width,mode", [(257, 259, "LA"), (2, 400000, "RGBA")])
def test_image_operators_cross_copy_and_png_chunk_boundaries(image_connection, height, width, mode):
    channels = 2 if mode == "LA" else 4
    pixels = np.random.default_rng(17).integers(0, 256, (height, width, channels), dtype=np.uint8)
    result, encoded = image_connection.execute(
        "SELECT crop($1,[0,0,$2,$3]), encode_image(crop($1,[0,0,$2,$3]),'PNG')",
        [vane.Value(pixels, vane.image_type(mode, height, width)), width, height],
    ).fetchone()
    np.testing.assert_array_equal(result, pixels)
    assert len(encoded) > 64 * 1024
    _check_png(encoded, pixels, mode)


@pytest.mark.parametrize("form", ["generic", "fixed"])
def test_image_operators_selected_rows_nulls_and_arrow(image_connection, form):
    con = image_connection
    dtype = "IMAGE('RGB', 2, 4)" if form == "fixed" else "IMAGE"
    con.execute(
        f"""CREATE TABLE images AS SELECT i,
        CASE WHEN i % 11 = 0 THEN NULL ELSE
          image(repeat(chr((65 + i % 20)::INTEGER), 24)::BLOB, 4, 2, 3, 'RGB') END::{dtype} AS image
        FROM range(4099) t(i)"""
    )
    relation = con.sql(
        """SELECT a.i, crop(a.image, [a.i % 3 - 1, 0, 3, 2]) AS cropped,
                  encode_image(a.image, 'PNG') AS encoded, a.image
           FROM images a JOIN images b ON a.i = b.i
           WHERE b.i % 7 = 2 ORDER BY a.i DESC"""
    )
    expected_type = vane.image_type("RGB") if form == "fixed" else vane.image_type()
    assert relation.types[1] == expected_type
    arrow = relation.to_arrow_table()
    assert arrow.schema.field("cropped").type.extension_name == "vane.image"
    # Slice canonical Arrow storage, then read it through the engine again.
    result = con.from_arrow(arrow.slice(1, 550)).fetchall()
    assert len(result) == 550
    for index, cropped, encoded, source in result:
        if index % 11 == 0:
            assert cropped is encoded is source is None
        else:
            expected = np.full((2, 4, 3), 65 + index % 20, dtype=np.uint8)
            np.testing.assert_array_equal(source, expected)
            np.testing.assert_array_equal(cropped, _expected_crop(expected, (index % 3 - 1, 0, 3, 2)))
            _check_png(encoded, expected, "RGB")


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("optimizer", [False, True])
def test_constant_images_with_varying_boxes_and_formats(image_connection, fixed, optimizer):
    con = image_connection
    if not optimizer:
        con.execute("PRAGMA disable_optimizer")
    pixels = _pixels("RGBA")
    dtype = vane.image_type("RGBA", 4, 5) if fixed else vane.image_type()
    rows = con.execute(
        """SELECT i, crop($1, [i % 3 - 1, 1, 2, 1]),
               encode_image($1, CASE WHEN i % 7 = 0 THEN NULL WHEN i % 2 = 0 THEN 'PNG' ELSE 'png' END)
           FROM range(4099) t(i)""",
        [vane.Value(pixels, dtype)],
    ).fetchall()
    assert len(rows) == 4099
    samples = {}
    for index, image, encoded in rows:
        np.testing.assert_array_equal(image, _expected_crop(pixels, (index % 3 - 1, 1, 2, 1)))
        assert (encoded is None) == (index % 7 == 0)
        if encoded is not None:
            samples[encoded] = True
    assert len(samples) == 1
    _check_png(next(iter(samples)), pixels, "RGBA")


def test_image_operator_null_empty_and_array_bbox(image_connection):
    con = image_connection
    assert con.execute("SELECT crop(NULL, [0, 0, 1, 1]), encode_image(NULL, 'PNG')").fetchone() == (None, None)
    assert con.execute("SELECT crop(NULL, [0,0,-1,1]), encode_image(NULL, 'JPEG')").fetchone() == (None, None)
    value = vane.Value(_pixels("L"), vane.image_type("L", 4, 5))
    assert con.execute("SELECT crop($1, NULL), encode_image($1, NULL)", [value]).fetchone() == (None, None)
    array = con.execute("SELECT crop($1, [1, 1, 2, 2]::INTEGER[4])", [value]).fetchone()[0]
    np.testing.assert_array_equal(array, _pixels("L")[1:3, 1:3])
    empty = con.sql("SELECT crop($1, [0, 0, 1, 1]) AS image WHERE false", params=[value])
    assert empty.types == [vane.image_type("L")]
    assert empty.to_arrow_table().num_rows == 0


@pytest.mark.parametrize("bbox", ["[0,0,1]", "[0,0,NULL,1]", "[0,0,0,1]", "[0,0,-1,1]", "[0,0,4294967296,1]"])
def test_crop_rejects_invalid_boxes(image_connection, bbox):
    with pytest.raises(vane.InvalidInputException, match="crop"):
        image_connection.execute(f"SELECT crop($1, {bbox})", [vane.Value(_pixels("RGB"), vane.image_type())]).fetchall()


@pytest.mark.parametrize("bbox", ["[0.1,0,1,1]", "[true,false,true,true]", "'0,0,1,1'", "[0,0,1]::INTEGER[3]"])
def test_crop_rejects_implicit_coordinate_conversion(image_connection, bbox):
    with pytest.raises(vane.BinderException, match="integer"):
        image_connection.execute(f"SELECT crop($1, {bbox})", [vane.Value(_pixels("RGB"), vane.image_type())]).fetchall()


@pytest.mark.parametrize("bbox", [(0.5, 0, 1, 1), (True, 0, 1, 1), (0, 0, 1)])
def test_crop_python_argument_validation(bbox):
    with pytest.raises((TypeError, ValueError), match="crop bbox"):
        vane.crop(vane.col("image"), bbox)


@pytest.mark.parametrize("value", [b"pixels", [1, 2, 3], vane.ImageFile("unused://image.png")])
def test_image_operators_require_decoded_image(image_connection, value):
    for query in ("SELECT crop($1, [0,0,1,1])", "SELECT encode_image($1, 'PNG')"):
        with pytest.raises(vane.BinderException, match="requires IMAGE"):
            image_connection.execute(query, [value]).fetchall()


@pytest.mark.parametrize("image_format", ["JPEG", "TIFF", "GIF", "BMP", "", "png "])
def test_encode_unsupported_format_is_explicit(image_connection, image_format):
    with pytest.raises(vane.NotImplementedException, match="PNG only"):
        image_connection.execute(
            "SELECT encode_image($1, $2)", [vane.Value(_pixels("RGBA"), vane.image_type()), image_format]
        ).fetchall()


def test_crop_checks_output_limits_before_allocation(image_connection):
    value = vane.Value(_pixels("RGBA"), vane.image_type())
    for bbox in ([0, 0, 100_000_001, 1], [0, 0, 100_000_000, 1]):
        with pytest.raises(vane.OutOfRangeException, match="pixel or byte limit"):
            image_connection.execute("SELECT crop($1, $2)", [value, bbox]).fetchall()
    for origin in (-(2**63), 2**63 - 1):
        pixels = image_connection.execute("SELECT crop($1, $2)", [value, [origin, origin, 2, 2]]).fetchone()[0]
        np.testing.assert_array_equal(pixels, np.zeros((2, 2, 4), dtype=np.uint8))


def test_png_enforces_cumulative_batch_limit(image_connection):
    con = image_connection
    con.execute("SET threads=1")
    # Each incompressible image is about 1 MiB. Every row fits individually;
    # the complete vector exceeds the 256 MiB encoded-output budget.
    pixels = np.random.default_rng(23).integers(0, 256, (512, 512, 4), dtype=np.uint8)
    with pytest.raises(vane.OutOfRangeException, match="byte limit"):
        con.execute(
            """SELECT sum(octet_length(encode_image($1,
                   CASE WHEN i % 2 = 0 THEN 'PNG' ELSE 'png' END))) FROM range(260) t(i)""",
            [vane.Value(pixels, vane.image_type("RGBA", 512, 512))],
        ).fetchone()


def test_native_image_operators_do_not_call_python_helpers(monkeypatch):
    import vane._image_operators as helpers

    def forbidden(*args):
        pytest.fail("native Image operator called a Python helper")

    with _connect("image") as con:
        monkeypatch.setattr(helpers, "_crop_image", forbidden)
        monkeypatch.setattr(helpers, "_encode_image_png", forbidden)
        pixels = _pixels("LA")
        encoded = con.execute(
            "SELECT encode_image(crop($1, [1,1,2,2]), 'PNG')",
            [vane.Value(pixels, vane.image_type("LA", 4, 5))],
        ).fetchone()[0]
        _check_png(encoded, pixels[1:3, 1:3], "LA")


@pytest.mark.parametrize("batch", [False, True])
def test_crop_png_through_registered_image_udf(image_connection, batch):
    dtype = vane.image_type("RGB")

    def identity(image):
        if batch:
            assert image.type.extension_name == "vane.image"
        else:
            assert image.dtype == np.uint8 and image.shape == (2, 2, 3)
        return image

    function = (vane.func.batch if batch else vane.func)(return_dtype=dtype)(identity)
    vane.attach_function(function, connection=image_connection, alias="crop_identity", parameters=[dtype])
    pixels = _pixels("RGB")
    result = image_connection.execute(
        "SELECT encode_image(crop_identity(crop($1,[1,1,2,2])), 'PNG')",
        [vane.Value(pixels, vane.image_type("RGB", 4, 5))],
    ).fetchone()[0]
    _check_png(result, pixels[1:3, 1:3], "RGB")


@pytest.mark.parametrize("backend", ["python", "native"])
def test_image_pixel_execution_without_pillow(backend):
    from tests.fast.test_native_media_extensions import _artifact

    program = r"""
import importlib.abc
import sys
class NoPillow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'PIL' or fullname.startswith('PIL.'):
            raise AssertionError('native Image execution imported Pillow')
sys.meta_path.insert(0, NoPillow())
import vane
backend, artifact = sys.argv[1:]
with vane.connect(config={'allow_unsigned_extensions': 'true', 'image_backend': backend}) as con:
    if artifact:
        con.load_extension(artifact)
    cropped = con.execute("SELECT crop(image('abcdef'::BLOB,3,1,2,'LA'),[1,0,2,1])").fetchone()[0]
    assert cropped.shape == (1,2,2) and cropped.tobytes() == b'cdef'
    if backend == 'native':
        encoded = con.execute("SELECT encode_image(crop(image('abcdef'::BLOB,3,1,2,'LA'),[1,0,2,1]),'PNG')").fetchone()[0]
        assert encoded.startswith(b'\x89PNG\r\n\x1a\n')
assert 'PIL' not in sys.modules
"""
    artifact = str(_artifact("image")) if backend == "native" else ""
    subprocess.run([sys.executable, "-I", "-c", program, backend, artifact], check=True, timeout=30)


def test_image_backend_selection_and_bound_plan():
    pytest.importorskip("PIL.Image")
    query = "SELECT encode_image(crop(image(repeat(chr((65+i)::INTEGER), 12)::BLOB, 2,2,3,'RGB'), [0,0,1,1]), 'PNG') FROM range(2) t(i)"
    with vane.connect(config={"image_backend": "native"}) as unloaded:
        with pytest.raises(vane.BinderException, match="requires the image extension"):
            unloaded.sql(query)
    with _connect("image") as con:
        relation = con.sql(query)
        assert "native_crop" in relation.explain() and "native_encode_image" in relation.explain()
        con.execute("PREPARE pixels AS " + query)
        con.execute("SET image_backend='python'")
        assert "native_crop" not in con.sql(query).explain()
        assert len(con.execute("EXECUTE pixels").fetchall()) == 2
        assert len(relation.fetchall()) == 2


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux address-space accounting")
@pytest.mark.parametrize("backend", ["python", "native"])
def test_hd_constant_image_does_not_expand_input_batch(backend):
    if backend == "python":
        pytest.importorskip("PIL.Image")
    artifact = ""
    if backend == "native":
        from tests.fast.test_native_media_extensions import _artifact

        artifact = str(_artifact("image"))
    program = r"""
import resource
import sys
from pathlib import Path
import numpy as np
import vane
backend, artifact = sys.argv[1:]
with vane.connect(config={'allow_unsigned_extensions': 'true', 'threads': 1}) as con:
    if artifact:
        con.load_extension(artifact)
    con.execute("SET image_backend='" + backend + "'")
    con.execute("PRAGMA disable_optimizer")
    warm = vane.Value(np.zeros((1,1,4), dtype=np.uint8), vane.image_type('RGBA',1,1))
    con.execute("SELECT encode_image(crop($1,[0,0,1,1]),'PNG')", [warm]).fetchone()
    pixels = np.full((2160,3840,4), 97, dtype=np.uint8)
    for dtype in (vane.image_type(), vane.image_type('RGBA',2160,3840)):
        value = vane.Value(pixels, dtype)
        vm = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                      if line.startswith('VmSize:'))) * 1024
        old, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = vm + 192 * 1024**2
        resource.setrlimit(resource.RLIMIT_AS, (limit if hard < 0 else min(limit, hard), hard))
        try:
            assert con.execute('SELECT sum(image_width(crop($1,[0,0,3840,2160]))) FROM range(4099)',
                               [value]).fetchone() == (3840 * 4099,)
            rows = con.execute('SELECT i, crop($1, [i % 3,0,1,1]) FROM range(4099) t(i)', [value]).fetchall()
            assert len(rows) == 4099
            for i, image in rows:
                assert image.shape == (1,1,4) and image.dtype == np.uint8
                assert (image == 97).all()
        finally:
            resource.setrlimit(resource.RLIMIT_AS, (old, hard))
"""
    subprocess.run([sys.executable, "-I", "-c", program, backend, artifact], check=True, timeout=120)


def test_image_operator_cancellation(image_connection):
    con = image_connection
    started = threading.Event()

    def execute():
        started.set()
        return con.execute(
            """SELECT sum(octet_length(encode_image(
               image(repeat(chr((65 + i % 20)::INTEGER),4096)::BLOB,64,64,1,'L'),'PNG')))
               FROM range(1000000) t(i)"""
        ).fetchone()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(execute)
        assert started.wait(5)
        deadline = time.monotonic() + 5
        # Repeat until completion so scheduler delays before query admission
        # cannot cause the only interrupt to be cleared by query startup.
        while not future.done() and time.monotonic() < deadline:
            con.interrupt()
            time.sleep(0.02)
        with pytest.raises(vane.InterruptException):
            future.result(timeout=15)
    assert con.execute("SELECT 42").fetchone() == (42,)


def _benchmark_frames(count):
    from multimodal_inference_benchmarks.video_object_detection.vane_image_pipeline import FRAME_TYPE
    from vane._image import image_arrow_type

    pixels = np.arange(count * 640 * 640 * 3, dtype=np.uint8).reshape(count, 640, 640, 3)
    storage = pa.FixedSizeListArray.from_arrays(pa.array(pixels.reshape(-1)), 640 * 640 * 3)
    return pixels, pa.ExtensionArray.from_storage(image_arrow_type(FRAME_TYPE), storage)


def test_video_benchmark_uses_dense_image_batches():
    from multimodal_inference_benchmarks.video_object_detection.vane_image_pipeline import frame_batch

    expected, images = _benchmark_frames(3)
    for column in (images.slice(1, 2), pa.chunked_array([images.slice(1, 1), images.slice(2, 1)])):
        actual = frame_batch(column)
        assert actual.dtype == np.uint8 and actual.flags.c_contiguous
        np.testing.assert_array_equal(actual, expected[1:])
    np.testing.assert_array_equal(frame_batch(images.slice(0, 0)), expected[:0])
    with pytest.raises(ValueError, match="requires IMAGE"):
        frame_batch(images.storage)
    with pytest.raises(ValueError, match="NULL frames"):
        frame_batch(images.take(pa.array([None], type=pa.int64())))


def test_video_benchmark_native_crop_pipeline():
    from multimodal_inference_benchmarks.video_object_detection.vane_image_pipeline import crop_objects

    pixels, images = _benchmark_frames(3)
    feature_type = pa.list_(
        pa.struct([("label", pa.int64()), ("confidence", pa.float64()), ("bbox", pa.list_(pa.float64()))])
    )
    features = [
        {"label": 2, "confidence": 0.9, "bbox": [-1.9, 1.9, 2.9, 3.9]},
        {"label": 3, "confidence": 0.7, "bbox": [639.1, 638.1, 642.8, 641.8]},
    ]
    table = pa.table(
        {"frame_index": [7, 8, 9], "frame": images, "features": pa.array([features, [], None], type=feature_type)}
    )
    with _connect("image") as con:
        relation = crop_objects(con.from_arrow(table))
        assert "native_crop" in relation.explain() and "native_encode_image" in relation.explain()
        rows = relation.order("features.label").fetchall()
    assert len(rows) == 2
    for (index, feature, encoded), expected_feature in zip(rows, features, strict=True):
        assert index == 7 and feature == expected_feature
        left, top, right, bottom = map(int, expected_feature["bbox"])
        _check_png(encoded, _expected_crop(pixels[0], (left, top, right - left, bottom - top)), "RGB")
