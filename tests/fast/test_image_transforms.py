# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pytest

import vane
from tests.fast.test_native_media_extensions import _artifact, _connect
from vane._image import image_arrow_type

MODES = ("L", "LA", "RGB", "RGBA")


@pytest.fixture(params=["python", "native"])
def transform_connection(request):
    with _connect("image") if request.param == "native" else vane.connect() as con:
        yield con


def _dtype(mode, form, height=2, width=2):
    return (
        vane.image_type()
        if form == "generic"
        else vane.image_type(mode)
        if form == "mode"
        else vane.image_type(mode, height, width)
    )


def _ramp(mode, values):
    base = np.asarray(values, dtype=np.uint8)
    channels = MODES.index(mode) + 1
    pixels = np.stack([base + 10 * c for c in range(channels)], axis=-1)
    if channels in (2, 4):
        pixels[:, :, -1] = 255
    return pixels


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("form", ["generic", "mode", "fixed"])
def test_resize_python_expression_and_sql(transform_connection, mode, form):
    con = transform_connection
    pixels = _ramp(mode, [[0, 40], [80, 120]])
    expected = _ramp(mode, [[0, 20, 40], [40, 60, 80], [80, 100, 120]])
    value = vane.Value(pixels, _dtype(mode, form))
    result_type = vane.image_type() if form == "generic" else vane.image_type(mode, 3, 3)
    source = con.sql("SELECT $1 AS image", params=[value])
    for expression in (vane.resize(vane.col("image"), 3, 3), vane.col("image").resize(w=3, h=np.int64(3))):
        relation = source.select(expression.alias("image"))
        assert relation.types == [result_type]
        actual = relation.fetchone()[0]
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.uint8 and actual.flags.c_contiguous and actual.flags.writeable
        actual[:] = 0
        np.testing.assert_array_equal(relation.execute().fetchone()[0], expected)
    relation = con.sql("SELECT resize($1, 1+2, 3) AS image", params=[value])
    assert relation.types == [result_type]
    table = relation.to_arrow_table()
    assert table.schema.field(0).type.equals(image_arrow_type(result_type))
    with pa.BufferOutputStream() as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        restored = pa.ipc.open_stream(sink.getvalue()).read_all()
    np.testing.assert_array_equal(con.from_arrow(restored).fetchone()[0], expected)
    np.testing.assert_array_equal(con.execute("SELECT resize($1,1,1)", [value]).fetchone()[0], _ramp(mode, [[60]]))
    np.testing.assert_array_equal(pixels, _ramp(mode, [[0, 40], [80, 120]]))


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("source_mode", MODES)
@pytest.mark.parametrize("target_mode", MODES)
@pytest.mark.parametrize("form", ["generic", "mode", "fixed"])
def test_convert_all_modes_and_result_types(transform_connection, source_mode, target_mode, form):
    rgb = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    gray = np.array([[76, 150], [29, 255]], dtype=np.uint8)
    alpha = np.array([[0, 1], [128, 255]], dtype=np.uint8)
    source = gray[:, :, None] if source_mode in ("L", "LA") else rgb
    if source_mode in ("LA", "RGBA"):
        source = np.concatenate([source, alpha[:, :, None]], axis=-1)
    expected = (
        gray[:, :, None]
        if target_mode in ("L", "LA")
        else (np.repeat(gray[:, :, None], 3, axis=2) if source_mode in ("L", "LA") else rgb)
    )
    if target_mode in ("LA", "RGBA"):
        target_alpha = alpha if source_mode in ("LA", "RGBA") else np.full((2, 2), 255, dtype=np.uint8)
        expected = np.concatenate([expected, target_alpha[:, :, None]], axis=-1)
    value = vane.Value(source, _dtype(source_mode, form))
    result_type = vane.image_type(target_mode, 2, 2) if form == "fixed" else vane.image_type(target_mode)
    con = transform_connection
    relation = con.sql("SELECT $1 AS image", params=[value])
    for expression in (
        vane.convert_image(vane.col("image"), vane.ImageMode(target_mode)),
        vane.col("image").convert_image(mode=target_mode.lower()),
    ):
        output = relation.select(expression)
        assert output.types == [result_type]
        np.testing.assert_array_equal(output.fetchone()[0], expected)
    output = con.sql("SELECT convert_image($1, $2)", params=[value, target_mode])
    assert output.types == [result_type]
    np.testing.assert_array_equal(output.fetchone()[0], expected)


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("mode", ["LA", "RGBA"])
def test_resize_premultiplies_alpha_and_preserves_identity(transform_connection, mode):
    if mode == "RGBA":
        pixels = np.array([[[255, 0, 0, 255], [0, 0, 255, 0]]], dtype=np.uint8)
        expected = np.array([[[255, 0, 0, 255], [255, 0, 0, 128], [0, 0, 0, 0]]], dtype=np.uint8)
    else:
        pixels = np.array([[[64, 255], [192, 0]]], dtype=np.uint8)
        expected = np.array([[[64, 255], [64, 128], [0, 0]]], dtype=np.uint8)
    value = vane.Value(pixels, vane.image_type(mode, 1, 2))
    actual, identity = transform_connection.execute("SELECT resize($1,3,1), resize($1,2,1)", [value]).fetchone()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(identity, pixels)


@pytest.mark.usefixtures("ray_query")
def test_resize_edges_rounding_and_downsampling_contract(transform_connection):
    con = transform_connection
    pixels = np.array([[[0], [1]]], dtype=np.uint8)
    value = vane.Value(pixels, vane.image_type("L", 1, 2))
    np.testing.assert_array_equal(con.execute("SELECT resize($1,3,1)", [value]).fetchone()[0], [[[0], [1], [1]]])
    pixels = np.array([[[0], [0], [0], [240]]], dtype=np.uint8)
    # Half-pixel bilinear samples the middle two pixels; it has no low-pass
    # antialiasing prefilter when shrinking.
    value = vane.Value(pixels, vane.image_type("L", 1, 4))
    np.testing.assert_array_equal(con.execute("SELECT resize($1,1,1)", [value]).fetchone()[0], [[[0]]])
    value = vane.Value(np.array([[[17, 29, 250]]], dtype=np.uint8), vane.image_type("RGB", 1, 1))
    np.testing.assert_array_equal(
        con.execute("SELECT resize($1,7,5)", [value]).fetchone()[0], np.tile([[[17, 29, 250]]], (5, 7, 1))
    )
    half = vane.Value(np.array([[[0, 0, 250]]], dtype=np.uint8), vane.image_type("RGB", 1, 1))
    assert con.execute("SELECT convert_image($1,'L')", [half]).fetchone()[0].item() == 29


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("fixed", [False, True])
def test_per_row_options_nulls_and_selected_batches(transform_connection, fixed):
    con = transform_connection
    image_type = "IMAGE('RGB',2,2)" if fixed else "IMAGE('RGB')"
    source = con.sql(f"""SELECT i, (CASE WHEN i % 7 = 0 THEN NULL ELSE
        image(from_hex(repeat(to_hex(40+i%100),12)),2,2,3,'RGB') END)::{image_type} AS image,
        CASE WHEN i % 11 = 0 THEN NULL ELSE i%3+1 END AS w,
        CASE WHEN i % 13 = 0 THEN NULL WHEN i%2=0 THEN 'L' ELSE 'rgba' END AS mode
        FROM range(4099) t(i)""")
    source.create_view("transform_inputs")
    relation = con.sql("""SELECT a.i, resize(a.image,a.w,2) AS resized,
        convert_image(a.image,a.mode) AS converted FROM transform_inputs a
        JOIN range(4099) b(i) ON a.i=b.i WHERE a.i%5<>1 ORDER BY a.i DESC""")
    assert relation.types == [vane.sqltypes.BIGINT, vane.image_type("RGB"), vane.image_type()]
    table = relation.to_arrow_table()
    assert table.schema.field(1).type.equals(image_arrow_type(vane.image_type("RGB")))
    assert table.schema.field(2).type.equals(image_arrow_type(vane.image_type()))
    for i, resized, converted in con.from_arrow(table).fetchall():
        value = 40 + i % 100
        if i % 7 == 0 or i % 11 == 0:
            assert resized is None
        else:
            np.testing.assert_array_equal(resized, np.full((2, i % 3 + 1, 3), value, dtype=np.uint8))
        if i % 7 == 0 or i % 13 == 0:
            assert converted is None
        else:
            expected = np.full((2, 2, 1 if i % 2 == 0 else 4), value, dtype=np.uint8)
            if i % 2:
                expected[:, :, -1] = 255
            np.testing.assert_array_equal(converted, expected)


@pytest.mark.usefixtures("ray_query")
def test_null_empty_and_fixed_result_padding(transform_connection):
    con = transform_connection
    assert con.execute("SELECT resize(NULL,2,3), convert_image(NULL,'RGB')").fetchone() == (None, None)
    value = vane.Value(_ramp("RGB", [[0, 40], [80, 120]]), vane.image_type("RGB", 2, 2))
    assert con.execute("SELECT resize($1,NULL,3), resize($1,3,NULL), convert_image($1,NULL)", [value]).fetchone() == (
        None,
        None,
        None,
    )
    empty = con.sql("SELECT convert_image(resize($1,3,2),'LA') AS image WHERE false", params=[value])
    assert empty.types == [vane.image_type("LA", 2, 3)]
    assert empty.to_arrow_table().num_rows == 0
    result = con.sql(
        """SELECT resize(CASE WHEN i%2=0 THEN NULL ELSE $1 END,3,2),
                        convert_image(CASE WHEN i%2=0 THEN NULL ELSE $1 END,'RGBA') FROM range(9) t(i)""",
        params=[value],
    )
    assert result.types == [vane.image_type("RGB", 2, 3), vane.image_type("RGBA", 2, 2)]
    for i, row in enumerate(result.fetchall()):
        if i % 2 == 0:
            assert row == (None, None)
        else:
            assert row[0].shape == (2, 3, 3) and row[1].shape == (2, 2, 4)


@pytest.mark.parametrize("argument", ["true", "1.1", "'2'", "[2]", "2::DECIMAL(3,0)"])
def test_resize_rejects_implicit_dimension_casts(transform_connection, argument):
    with pytest.raises(vane.BinderException, match="integers"):
        transform_connection.sql(f"SELECT resize(NULL, {argument}, 2)")


@pytest.mark.parametrize("argument", ["0", "-1", "4294967296"])
def test_resize_rejects_invalid_dimensions(transform_connection, argument):
    with pytest.raises(vane.InvalidInputException, match="positive UINTEGER"):
        transform_connection.sql(f"SELECT resize(NULL, {argument}, 2)")


@pytest.mark.parametrize("mode", ["'CMYK'", "''", "'RGB64'", "'rgb '"])
def test_convert_rejects_unsupported_modes(transform_connection, mode):
    with pytest.raises(vane.InvalidInputException, match="mode must be"):
        transform_connection.sql(f"SELECT convert_image(NULL, {mode})")


def test_transforms_reject_wrong_input_types(transform_connection):
    for query in ("SELECT resize('pixels'::BLOB,2,2)", "SELECT convert_image([1,2,3], 'RGB')"):
        with pytest.raises(vane.BinderException, match="requires IMAGE"):
            transform_connection.sql(query)
    with pytest.raises(vane.BinderException, match="mode must be a string"):
        transform_connection.sql("SELECT convert_image(NULL,1)")


@pytest.mark.parametrize("dimension", [True, np.bool_(False), 1.2, "2", 0, -1, 1 << 32])
def test_resize_python_dimension_validation(dimension):
    with pytest.raises((TypeError, ValueError), match="resize"):
        vane.resize(vane.col("image"), dimension, 2)
    with pytest.raises((TypeError, ValueError), match="resize"):
        vane.col("image").resize(2, dimension)


@pytest.mark.usefixtures("ray_query")
def test_transform_expression_arguments_and_strided_input(transform_connection):
    con = transform_connection
    pixels = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)[:, ::2]
    result = con.sql("SELECT 3 AS w, 'rgba' AS mode").select(
        vane.resize(pixels, vane.col("w"), 3).convert_image(vane.col("mode"))
    )
    assert result.types == [vane.image_type()]
    actual = result.fetchone()[0]
    assert actual.shape == (3, 3, 4) and (actual[:, :, -1] == 255).all()
    with pytest.raises(TypeError, match="mode must be"):
        vane.convert_image(vane.col("image"), 3)


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("operation", ["resize", "convert_image"])
@pytest.mark.parametrize("batch", [False, True])
def test_transform_outputs_through_image_udfs(transform_connection, operation, batch):
    con = transform_connection
    dtype = vane.image_type("RGB", 3, 3) if operation == "resize" else vane.image_type("LA", 2, 2)

    def identity(image):
        if batch:
            assert image.type.equals(image_arrow_type(dtype))
        else:
            assert image.dtype == np.uint8 and image.shape == (*dtype.shape, 3 if operation == "resize" else 2)
        return image

    function = (vane.func.batch if batch else vane.func)(return_dtype=dtype)(identity)
    vane.attach_function(function, connection=con, alias="transform_identity", parameters=[dtype])
    pixels = _ramp("RGB", [[0, 40], [80, 120]])
    value = vane.Value(pixels, vane.image_type("RGB", 2, 2))
    expression = "resize($1,3,3)" if operation == "resize" else "convert_image($1,'LA')"
    expected = con.execute("SELECT " + expression, [value]).fetchone()[0]
    actual = con.execute(f"SELECT transform_identity({expression})", [value]).fetchone()[0]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.usefixtures("ray_query")
def test_native_transforms_select_backend_and_avoid_python(monkeypatch):
    import vane._image_operators as helpers

    query = "SELECT convert_image(resize(image(repeat(chr((65+i)::INTEGER),12)::BLOB,2,2,3,'RGB'),3,3),'LA') FROM range(3) t(i)"
    with vane.connect(config={"image_backend": "native"}) as unloaded:
        with pytest.raises(vane.BinderException, match="requires the image extension"):
            unloaded.sql(query)

    def forbidden(*args):
        raise AssertionError("native transform called a Python pixel helper")

    with _connect("image") as con:
        plan = con.sql(query).explain()
        assert "native_resize" in plan and "native_convert_image" in plan
        con.execute("PREPARE transform AS " + query)
        with monkeypatch.context() as trap:
            trap.setattr(helpers, "_resize_image", forbidden)
            trap.setattr(helpers, "_convert_image", forbidden)
            assert len(con.execute("EXECUTE transform").fetchall()) == 3
            con.execute("SET image_backend='python'")
            assert len(con.execute("EXECUTE transform").fetchall()) == 3
        assert "native_resize" not in con.sql(query).explain()
        assert len(con.sql(query).fetchall()) == 3


@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.local_fast(reason="Native image transform dependency isolation")
def test_transforms_execute_without_pillow(backend):
    program = r"""
import importlib.abc
import sys
class NoPillow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'PIL' or fullname.startswith('PIL.'):
            raise AssertionError('pixel transform imported Pillow')
sys.meta_path.insert(0, NoPillow())
import vane
backend, artifact = sys.argv[1:]
with vane.connect(config={'allow_unsigned_extensions': 'true', 'image_backend': backend}) as con:
    if artifact:
        con.load_extension(artifact)
    value = con.execute("SELECT convert_image(resize(image('abcd'::BLOB,2,2,1,'L'),3,3),'RGBA')").fetchone()[0]
    assert value.shape == (3,3,4) and (value[:,:,-1] == 255).all()
assert 'PIL' not in sys.modules
"""
    artifact = str(_artifact("image")) if backend == "native" else ""
    subprocess.run([sys.executable, "-I", "-c", program, backend, artifact], check=True, timeout=30)


@pytest.mark.usefixtures("ray_query")
def test_transform_limits_include_fixed_null_padding(transform_connection):
    con = transform_connection
    con.execute("SET threads=1")
    value = vane.Value(np.ones((1, 1, 4), dtype=np.uint8), vane.image_type("RGBA", 1, 1))
    with pytest.raises(vane.OutOfRangeException, match="pixel or byte limit"):
        con.sql("SELECT resize($1,100000001,1)", params=[value])
    with pytest.raises(vane.OutOfRangeException, match="pixel or byte limit"):
        con.sql("SELECT resize($1,100000000,1)", params=[value])
    with pytest.raises(vane.OutOfRangeException, match="pixel or byte limit"):
        con.sql("SELECT convert_image(NULL::IMAGE('L',100000000,1),'RGBA')")
    # Nonconstant NULL inputs still occupy a fixed ARRAY row. Charge their
    # padding; skipping NULL rows must not evade the batch payload budget.
    with pytest.raises(vane.OutOfRangeException, match="pixel or byte limit"):
        con.execute(
            """SELECT sum(image_width(resize(
            CASE WHEN i%2=0 THEN NULL ELSE $1 END,1024,1024))) FROM range(65) t(i)""",
            [value],
        ).fetchone()


@pytest.mark.usefixtures("ray_query")
def test_parameter_rebinding_and_cast_do_not_convert_colors(transform_connection):
    con = transform_connection
    con.execute("PREPARE transform_size AS SELECT resize(image('a'::BLOB,1,1,1,'L')::IMAGE('L'),$1,$2)")
    assert con.execute("EXECUTE transform_size(3,2)").fetchone()[0].shape == (2, 3, 1)
    assert con.execute("EXECUTE transform_size(1,4)").fetchone()[0].shape == (4, 1, 1)
    actual = con.execute("""SELECT TRY_CAST(image('abc'::BLOB,1,1,3,'RGB') AS IMAGE('RGBA')),
                            convert_image(image('abc'::BLOB,1,1,3,'RGB'),'RGBA')""").fetchone()
    assert actual[0] is None
    np.testing.assert_array_equal(actual[1], [[[97, 98, 99, 255]]])


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("operation", ["_resize_image", "_convert_image"])
@pytest.mark.parametrize(
    "error,expected", [(MemoryError, vane.OutOfMemoryException), (KeyboardInterrupt, vane.InterruptException)]
)
def test_python_transform_system_errors_propagate(monkeypatch, operation, error, expected):
    import vane._image_operators as helpers

    def fail(*args):
        raise error()

    monkeypatch.setattr(helpers, operation, fail)
    expression = (
        "resize(image('a'::BLOB,1,1,1,'L'),2,2)"
        if operation == "_resize_image"
        else "convert_image(image('a'::BLOB,1,1,1,'L'),'RGB')"
    )
    with vane.connect() as con:
        with pytest.raises(expected):
            con.execute("SELECT " + expression).fetchall()
        assert con.execute("SELECT 42").fetchone() == (42,)


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux address-space accounting")
@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.local_fast(reason="Native image buffer allocation under a process memory limit")
def test_transforms_keep_large_constant_input_and_output_once(backend):
    program = r"""
import resource
import sys
from pathlib import Path
import numpy as np
import vane
backend, artifact = sys.argv[1:]
with vane.connect(config={'allow_unsigned_extensions': 'true', 'threads': 1, 'image_backend': backend}) as con:
    if artifact:
        con.load_extension(artifact)
    con.execute('PRAGMA disable_optimizer')
    con.execute("SELECT convert_image(resize(image('abc'::BLOB,1,1,3,'RGB'),2,2),'RGBA')").fetchone()
    pixels = np.full((2160,3840,4), 97, dtype=np.uint8)
    for dtype in (vane.image_type(), vane.image_type('RGBA',2160,3840)):
        value = vane.Value(pixels, dtype)
        constant = con.sql("SELECT sum(image_width(convert_image(resize($1,3840,2160),'RGB'))) FROM range(4099)", params=[value])
        varying = con.sql('SELECT i, resize($1,i%3+1,1) FROM range(4099) t(i)', params=[value])
        vm = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmSize:'))) * 1024
        old, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = vm + 224 * 1024**2
        resource.setrlimit(resource.RLIMIT_AS, (limit if hard < 0 else min(limit, hard), hard))
        try:
            assert constant.fetchone() == (3840 * 4099,)
            rows = varying.fetchall()
            assert len(rows) == 4099
            for i, image in rows:
                assert image.shape == (1,i%3+1,4) and (image == 97).all()
        finally:
            resource.setrlimit(resource.RLIMIT_AS, (old, hard))
"""
    artifact = str(_artifact("image")) if backend == "native" else ""
    subprocess.run([sys.executable, "-I", "-c", program, backend, artifact], check=True, timeout=120)


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("operation", ["resize", "convert_image"])
def test_transform_interruption_and_connection_reuse(transform_connection, operation):
    con = transform_connection
    started = threading.Event()
    transform = "resize(image,512,512)" if operation == "resize" else "convert_image(image,'RGBA')"

    def execute():
        started.set()
        return con.execute(f"""SELECT sum(image_width({transform})) FROM (
            SELECT image(repeat(chr((65+i%20)::INTEGER),49152)::BLOB,128,128,3,'RGB') AS image
            FROM range(1000000) t(i))""").fetchone()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(execute)
        assert started.wait(5)
        deadline = time.monotonic() + 5
        while not future.done() and time.monotonic() < deadline:
            con.interrupt()
            time.sleep(0.02)
        with pytest.raises(vane.InterruptException):
            future.result(timeout=15)
    assert con.execute("SELECT 42").fetchone() == (42,)


@pytest.mark.parametrize("operation", ["resize", "convert_image"])
def test_python_transforms_batch_narrow_rows_and_check_cancellation(operation):
    import vane._image_operators as helpers

    source = np.full((1_000_003, 1, 3), 97, dtype=np.uint8)
    target = np.full(
        (source.shape[0], 2 if operation == "resize" else 1, 3 if operation == "resize" else 4), 0xCC, dtype=np.uint8
    )
    calls = 0

    def check():
        nonlocal calls
        calls += 1

    def run(callback):
        args = [memoryview(source), 1, source.shape[0], 3]
        if operation == "resize":
            helpers._resize_image(*args, 2, source.shape[0], memoryview(target), callback)
        else:
            helpers._convert_image(*args, 4, memoryview(target), callback, "RGB", "RGBA")

    run(check)
    assert 1 < calls < 150
    assert (target[:, :, :3] == 97).all()
    if operation == "convert_image":
        assert (target[:, :, -1] == 255).all()
    target[:] = 0xCC

    def cancel():
        if target.reshape(-1)[0] == 97:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run(cancel)
    assert target.reshape(-1)[-1] == 0xCC
