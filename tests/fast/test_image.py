# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pickle
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pytest

import vane
from tests.image_helpers import assert_image_equal
from vane._image import _ImageArrowType, image_arrow_type
from vane.execution.udf_file_contract import FileUDFContract


@pytest.mark.parametrize("mode,channels", [("L", 1), ("LA", 2), ("RGB", 3), ("RGBA", 4)])
def test_image_type_forms_and_uint8_storage(mode, channels):
    generic = vane.image_type()
    variable = vane.image_type(mode)
    fixed = vane.image_type(mode, 2, 3)
    assert str(variable) == f"IMAGE('{mode}')"
    assert str(fixed) == f"IMAGE('{mode}', 2, 3)"
    assert generic.image_mode is None
    assert variable.image_mode is vane.ImageMode(mode)
    assert fixed.image_mode is vane.ImageMode(mode)
    assert fixed.shape == (2, 3)
    assert fixed.id == "array"
    assert fixed.children == [("child", vane.sqltypes.UTINYINT), ("size", 6 * channels)]
    assert generic.children == [
        ("data", vane.list_type(vane.sqltypes.UTINYINT)),
        ("channel", vane.sqltypes.USMALLINT),
        ("height", vane.sqltypes.UINTEGER),
        ("width", vane.sqltypes.UINTEGER),
        ("mode", vane.sqltypes.UTINYINT),
    ]
    assert generic != variable != fixed
    for dtype in (generic, variable, fixed):
        assert dtype.is_image() and not dtype.is_file()
        assert dtype.is_fixed_shape_image() == (dtype == fixed)
        assert dtype == vane.sqltype(str(dtype)) == pickle.loads(pickle.dumps(dtype))
    with pytest.raises(vane.InvalidInputException, match="fixed-shape"):
        _ = variable.shape


@pytest.mark.parametrize("enum", [vane.ImageMode, vane.ImageFormat, vane.ImageProperty])
def test_image_enum_string_roundtrip(enum):
    for member in enum:
        assert enum(str(member)) is member
        assert pickle.loads(pickle.dumps(member)) is member
    with pytest.raises(ValueError):
        enum("unsupported")


@pytest.mark.parametrize("dtype", [vane.image_type(), vane.image_type("RGB"), vane.image_type("RGB", 2, 3)])
def test_image_hwc_numpy_materialization_and_detached_pixels(duckdb_cursor, dtype):
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    parameter = vane.Value(pixels, dtype)
    relation = duckdb_cursor.sql("SELECT $1 AS image", params=[parameter])
    assert relation.types == [dtype]
    output = relation.fetchone()[0]
    assert_image_equal(output, pixels)
    assert output.flags.c_contiguous
    output[0, 0, 0] = 255
    assert pixels[0, 0, 0] == 0
    for consumer in ("fetchone", "fetchnumpy", "df"):
        fresh = duckdb_cursor.sql("SELECT $1 AS image", params=[parameter])
        values = getattr(fresh, consumer)()
        assert_image_equal(values[0] if consumer == "fetchone" else values["image"][0], pixels)
    rendered = str(vane.ConstantExpression(parameter))
    assert_image_equal(duckdb_cursor.sql(f"SELECT {rendered}").fetchone()[0], pixels)


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_image_typed_strided_numpy_and_optional_pil_inference(duckdb_cursor, channels):
    mode = list(vane.ImageMode)[channels - 1]
    pixels = np.arange(4 * 5 * channels, dtype=np.uint8).reshape(4, 5, channels)[::-1, ::2, :]
    dtype = vane.image_type(mode, 4, 3)
    assert_image_equal(duckdb_cursor.execute("SELECT $1", [vane.Value(pixels, dtype)]).fetchone()[0], pixels)
    pil = pytest.importorskip("PIL.Image")
    source = pixels[:, :, 0] if channels == 1 else pixels
    image = pil.fromarray(source, str(mode))
    relation = duckdb_cursor.sql("SELECT $1 AS image", params=[image])
    assert relation.types == [vane.image_type()]
    assert_image_equal(relation.fetchone()[0], pixels)
    pandas = pytest.importorskip("pandas")
    assert_image_equal(
        duckdb_cursor.from_df(pandas.DataFrame({"image": [image, None]})).fetchall(), [(pixels,), (None,)]
    )


def test_untyped_numpy_is_not_inferred_as_image(duckdb_cursor):
    pixels = np.zeros((1, 2, 3), dtype=np.uint8)
    assert not duckdb_cursor.sql("SELECT $1", params=[pixels]).types[0].is_image()


@pytest.mark.parametrize(
    "pixels,dtype",
    [
        (np.zeros((1, 1, 3), dtype=np.float32), vane.image_type()),
        (np.zeros((1, 1), dtype=np.uint8), vane.image_type()),
        (np.zeros((1, 1, 5), dtype=np.uint8), vane.image_type()),
        (np.zeros((0, 1, 3), dtype=np.uint8), vane.image_type()),
        (np.zeros((1, 1, 3), dtype=np.uint8), vane.image_type("RGBA")),
        (np.zeros((2, 1, 3), dtype=np.uint8), vane.image_type("RGB", 1, 2)),
        (np.ma.array(np.zeros((1, 1, 3), dtype=np.uint8), mask=True), vane.image_type()),
        ({"data": [1], "channel": 1, "height": 1, "width": 1, "mode": 1}, vane.image_type()),
    ],
)
def test_declared_image_input_rejects_invalid_pixels(pixels, dtype):
    with pytest.raises(vane.InvalidInputException):
        vane.ConstantExpression(vane.Value(pixels, dtype))


@pytest.mark.parametrize("dtype", [vane.image_type(), vane.image_type("RGB"), vane.image_type("RGB", 2, 3)])
@pytest.mark.parametrize("nested", [False, True])
def test_image_arrow_ipc_and_parquet_keep_mode_and_shape(duckdb_cursor, tmp_path, dtype, nested):
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    declared = vane.struct_type({"items": vane.list_type(dtype)}) if nested else dtype
    value = {"items": [pixels, None]} if nested else pixels
    table = duckdb_cursor.sql(
        "SELECT $1 AS image UNION ALL SELECT NULL", params=[vane.Value(value, declared)]
    ).to_arrow_table()
    leaf = table.schema.field(0).type.field("items").type.value_type if nested else table.schema.field(0).type
    assert leaf == image_arrow_type(dtype)
    assert pickle.loads(pickle.dumps(leaf)) == leaf
    if dtype.is_fixed_shape_image():
        assert leaf.storage_type == pa.list_(pa.uint8(), 18)
    else:
        assert leaf.storage_type.field("data").type == pa.list_(pa.uint8())
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    restored = pa.ipc.open_stream(sink.getvalue()).read_all()
    relation = duckdb_cursor.from_arrow(restored)
    assert relation.types == [declared]
    assert_image_equal(relation.fetchall(), [(value,), (None,)])
    parquet = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "images.parquet"
    if dtype.is_fixed_shape_image():
        # PyArrow's Parquet reader cannot reconstruct NULL FixedSizeList
        # values (including NULL ancestors). IPC above covers those rows.
        value = {"items": [pixels]} if nested else pixels
        table = duckdb_cursor.sql("SELECT $1 AS image", params=[vane.Value(value, declared)]).to_arrow_table()
        expected = [(value,)]
    else:
        expected = [(value,), (None,)]
    parquet.write_table(table, path)
    assert_image_equal(duckdb_cursor.from_arrow(parquet.read_table(path)).fetchall(), expected)


@pytest.mark.parametrize(
    "left_type,left_shape,right_type,right_shape",
    [
        (vane.image_type("RGB", 1, 2), (1, 2, 3), vane.image_type("RGB", 2, 1), (2, 1, 3)),
        (vane.image_type("RGB", 1, 4), (1, 4, 3), vane.image_type("RGBA", 1, 3), (1, 3, 4)),
        (vane.image_type("RGB"), (1, 2, 3), vane.image_type("RGBA"), (1, 2, 4)),
        (vane.image_type(), (1, 2, 3), vane.image_type("RGB"), (1, 2, 3)),
    ],
    ids=["fixed-dimensions", "fixed-modes", "dynamic-modes", "generic-mode"],
)
@pytest.mark.parametrize("nested", [False, True])
def test_image_arrow_concatenation_rejects_different_layouts(
    duckdb_cursor, left_type, left_shape, right_type, right_shape, nested
):
    tables = []
    for dtype, shape in ((left_type, left_shape), (right_type, right_shape)):
        pixels = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
        declared = vane.struct_type({"items": vane.list_type(dtype)}) if nested else dtype
        value = {"items": [pixels, None]} if nested else pixels
        tables.append(duckdb_cursor.sql("SELECT $1 AS image", params=[vane.Value(value, declared)]).to_arrow_table())
    left, right = tables
    left_leaf, right_leaf = image_arrow_type(left_type), image_arrow_type(right_type)
    assert left_leaf.storage_type == right_leaf.storage_type
    assert left_leaf != right_leaf and right_leaf != left_leaf
    assert not left_leaf.equals(right_leaf) and not right_leaf.equals(left_leaf)
    assert len({left_leaf: 1, right_leaf: 2}) == 2
    assert left.schema != right.schema
    assert not left.schema.equals(right.schema)
    for first, second in ((left, right), (right, left)):
        arrays = [first.column(0).combine_chunks(), second.column(0).combine_chunks()]
        with pytest.raises(pa.ArrowInvalid):
            pa.concat_tables([first, second])
        with pytest.raises(pa.ArrowInvalid):
            pa.concat_arrays(arrays)
        with pytest.raises(pa.ArrowTypeError):
            pa.chunked_array(arrays).combine_chunks()


@pytest.mark.parametrize("dtype", [vane.image_type(), vane.image_type("RGB"), vane.image_type("RGB", 2, 3)])
def test_image_arrow_concatenation_preserves_equal_types_after_serialization(duckdb_cursor, dtype):
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    table = (
        duckdb_cursor.sql(
            "SELECT NULL::" + str(dtype) + " AS image UNION ALL SELECT $1 UNION ALL SELECT NULL",
            params=[vane.Value(pixels, dtype)],
        )
        .to_arrow_table()
        .slice(1, 2)
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    restored = pa.ipc.open_stream(sink.getvalue()).read_all()
    types = [image_arrow_type(dtype), table.column(0).type, restored.column(0).type]
    types.append(pickle.loads(pickle.dumps(types[0])))
    for arrow_type in types:
        assert arrow_type == types[0] and not arrow_type != types[0]
        assert arrow_type.equals(types[0])
        assert hash(arrow_type) == hash(types[0])
        assert arrow_type != arrow_type.storage_type
    assert len(set(types)) == 1
    arrays = [table.column(0).combine_chunks(), restored.column(0).combine_chunks()]
    combined = [
        pa.concat_tables([table, restored]).combine_chunks(),
        pa.table({"image": pa.concat_arrays(arrays)}),
        pa.table({"image": pa.chunked_array(arrays).combine_chunks()}),
    ]
    for result in combined:
        relation = duckdb_cursor.from_arrow(result)
        assert relation.types == [dtype]
        assert_image_equal(relation.fetchall(), [(pixels,), (None,)] * 2)


@pytest.mark.parametrize("mode,channels", [("L", 1), ("LA", 2), ("RGB", 3), ("RGBA", 4)])
def test_image_attributes_sql_functions_and_methods(duckdb_cursor, mode, channels):
    pixels = np.zeros((2, 3, channels), dtype=np.uint8)
    for dtype in (vane.image_type(), vane.image_type(mode), vane.image_type(mode, 2, 3)):
        relation = duckdb_cursor.sql("SELECT $1 AS image UNION ALL SELECT NULL", params=[vane.Value(pixels, dtype)])
        names = ["height", "width", "channel", "mode"]
        columns = [getattr(vane, f"image_{name}")(vane.col("image")) for name in names]
        columns += [getattr(vane.col("image"), f"image_{name}")() for name in names]
        columns += [vane.col("image").image_attribute(name) for name in names]
        assert relation.select(*columns).fetchall() == [(2, 3, channels, channels) * 3, (None,) * 12]
        assert relation.query("images", "SELECT image_attribute(image, 'width') FROM images").fetchall() == [
            (3,),
            (None,),
        ]
    assert duckdb_cursor.sql("SELECT 1").select(vane.image_width(pixels)).fetchone() == (3,)
    with pytest.raises(vane.BinderException, match="requires IMAGE"):
        duckdb_cursor.sql("SELECT image_width([1, 2, 3])")
    with pytest.raises(vane.InvalidInputException, match="property"):
        duckdb_cursor.execute("SELECT image_attribute($1, 'bad')", [vane.Value(pixels, vane.image_type())])


def test_expression_as_image_validates_layout_without_color_conversion(duckdb_cursor):
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    relation = duckdb_cursor.sql("SELECT $1 AS image", params=[vane.Value(pixels, vane.image_type())])
    assert relation.select(vane.col("image").as_image("RGB")).types == [vane.image_type("RGB")]
    fixed = relation.select(vane.col("image").as_image(vane.ImageMode.RGB, 2, 3))
    assert fixed.types == [vane.image_type("RGB", 2, 3)]
    assert_image_equal(fixed.fetchone()[0], pixels)
    with pytest.raises(vane.InvalidInputException, match="does not match"):
        relation.select(vane.col("image").as_image("RGBA")).fetchall()


@pytest.mark.parametrize("dtype", [vane.image_type(), vane.image_type("RGB"), vane.image_type("RGB", 2, 3)])
@pytest.mark.parametrize("batch", [False, True])
def test_image_registered_udf_keeps_logical_type_and_nulls(duckdb_cursor, dtype, batch):
    def identity(value):
        if batch:
            assert value.type == image_arrow_type(dtype)
        else:
            assert isinstance(value, np.ndarray)
            assert value.dtype == np.uint8 and value.shape == (2, 3, 3)
        return value

    function = (vane.func.batch if batch else vane.func)(return_dtype=dtype)(identity)
    vane.attach_function(function, connection=duckdb_cursor, alias="identity_image", parameters=[dtype])
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    relation = duckdb_cursor.sql(
        "SELECT identity_image($1) AS image UNION ALL SELECT identity_image(NULL)", params=[vane.Value(pixels, dtype)]
    )
    assert relation.types == [dtype]
    assert_image_equal(relation.fetchall(), [(pixels,), (None,)])


@pytest.mark.parametrize("fixed", [False, True])
def test_image_arrow_rejects_null_pixels_but_ignores_null_rows(duckdb_cursor, fixed):
    dtype = vane.image_type("L", 1, 2) if fixed else vane.image_type("L")
    arrow_type = image_arrow_type(dtype)
    bad = [1, None] if fixed else {"data": [1, None], "channel": 1, "height": 1, "width": 2, "mode": 1}
    array = pa.ExtensionArray.from_storage(arrow_type, pa.array([bad], type=arrow_type.storage_type))
    with pytest.raises(vane.InvalidInputException, match="NULL"):
        duckdb_cursor.from_arrow(pa.table({"image": array})).fetchall()
    parent = pa.StructArray.from_arrays([array], names=["image"], mask=pa.array([True]))
    assert duckdb_cursor.from_arrow(pa.table({"row": parent})).fetchall() == [(None,)]


@pytest.mark.parametrize(
    "metadata",
    [
        b"{}",
        b'{"mode":"RGB","height":1,"width":null}',
        b'{"mode":"RGB","height":true,"width":1}',
        b'{"mode":null,"height":1,"width":1}',
        b'{"mode":null,"mode":null,"height":null}',
    ],
)
def test_image_arrow_metadata_rejects_malformed_layout(metadata):
    with pytest.raises(ValueError):
        _ImageArrowType.__arrow_ext_deserialize__(image_arrow_type(vane.image_type()).storage_type, metadata)


def test_image_base_materialization_does_not_require_pillow():
    program = """
import importlib.abc
import sys
class RejectPIL(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'PIL' or fullname.startswith('PIL.'):
            raise ModuleNotFoundError('Pillow is intentionally absent')
sys.meta_path.insert(0, RejectPIL())
import numpy as np
import vane
image = np.arange(6, dtype=np.uint8).reshape(1, 2, 3)
for dtype in (vane.image_type(), vane.image_type('RGB'), vane.image_type('RGB', 1, 2)):
    with vane.connect() as con:
        np.testing.assert_array_equal(con.execute('SELECT $1', [vane.Value(image, dtype)]).fetchone()[0], image)
assert 'PIL' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-c", program], check=True, capture_output=True, text=True)


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux address-space accounting")
def test_hd_image_scalars_keep_dense_pixel_buffers():
    pytest.importorskip("PIL.Image")
    program = """
import resource
from pathlib import Path
import numpy as np
from PIL import Image
import vane
with vane.connect(config={'threads': 1}) as con:
    con.execute('SELECT $1', [vane.Value(np.zeros((1, 1, 3), dtype=np.uint8), vane.image_type())]).fetchone()
    pixels = np.arange(1080 * 1920 * 3, dtype=np.uint8).reshape(1080, 1920, 3)
    pil = Image.fromarray(pixels)
    vm_kib = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                      if line.startswith('VmSize:')))
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    ceiling = vm_kib * 1024 + 192 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (ceiling if hard < 0 else min(ceiling, hard), hard))
    # ConstantExpression binds a native scalar, including fixed Image inputs.
    for dtype in (vane.image_type(), vane.image_type('RGB'), vane.image_type('RGB', 1080, 1920)):
        expression = vane.ConstantExpression(vane.Value(pixels, dtype))
        assert isinstance(expression, vane.Expression)
        del expression
    expression = vane.ConstantExpression(pil)
    del expression
    for image in (pixels, pixels[:, ::-1, :]):
        result = con.execute('SELECT $1', [vane.Value(image, vane.image_type())]).fetchone()[0]
        assert result.shape == image.shape and result.dtype == image.dtype
        # Compare every pixel without allocating full-image assertion temporaries.
        for row in range(0, image.shape[0], 16):
            np.testing.assert_array_equal(result[row:row + 16], image[row:row + 16])
        assert result.flags.c_contiguous
        del result
"""
    completed = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux address-space accounting")
@pytest.mark.parametrize("declared", ["IMAGE", "IMAGE('RGBA')", "IMAGE('RGBA', 2160, 3840)"])
def test_4k_image_udf_inputs_and_outputs_keep_dense_pixel_buffers(declared):
    pytest.importorskip("PIL.Image")
    program = """
import resource
import sys
from pathlib import Path
import numpy as np
import pyarrow as pa
from PIL import Image
import vane
from vane.execution.udf_file_contract import FileUDFContract
warm = FileUDFContract('warm', (), (vane.image_type(),))
warm.scalar_outputs_to_array([np.zeros((1, 1, 4), dtype=np.uint8)])
dtype = vane.sqltype(sys.argv[1])
contract = FileUDFContract('image_roundtrip', (dtype,), (dtype,))
pixels = np.arange(2160 * 3840 * 4, dtype=np.uint8).reshape(2160, 3840, 4)
pil = Image.fromarray(pixels)
vm_kib = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                  if line.startswith('VmSize:')))
_, hard = resource.getrlimit(resource.RLIMIT_AS)
ceiling = vm_kib * 1024 + 192 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (ceiling if hard < 0 else min(ceiling, hard), hard))
for image, expected in ((pixels, pixels), (pixels[:, ::-1, :], pixels[:, ::-1, :]), (pil, pixels)):
    output = contract.normalize_scalar_arrow_output(contract.scalar_outputs_to_array([image]))
    storage = output.storage if dtype.is_fixed_shape_image() else output.storage.field('data')
    actual = storage[0].values.to_numpy().reshape(expected.shape)
    assert actual.dtype == expected.dtype
    for row in range(0, expected.shape[0], 16):
        np.testing.assert_array_equal(actual[row:row + 16], expected[row:row + 16])
    del actual, storage
    restored = contract.materialize_scalar_inputs(pa.table({'image': output}))[0][0]
    assert restored.shape == expected.shape and restored.dtype == expected.dtype
    for row in range(0, expected.shape[0], 16):
        np.testing.assert_array_equal(restored[row:row + 16], expected[row:row + 16])
    assert restored.flags.c_contiguous and restored.flags.writeable
    del restored, output
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program, declared], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_image_udf_buffers_detach_and_preserve_nested_nulls(duckdb_cursor, fixed, nested, strided):
    dtype = vane.image_type("RGBA", 2, 3) if fixed else vane.image_type("RGBA")
    declared = vane.struct_type({"images": vane.list_type(dtype)}) if nested else dtype
    pixels = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    if strided:
        pixels = pixels[:, ::-1, :]
    expected = pixels.copy()
    value = {"images": [pixels, None]} if nested else pixels
    contract = FileUDFContract("image_roundtrip", (declared,), (declared,))
    output = contract.normalize_scalar_arrow_output(contract.scalar_outputs_to_array([value, None]))
    pixels[:] = 255
    table = pa.table({"image": output})
    materialized = contract.materialize_scalar_inputs(table)
    expected_value = {"images": [expected, None]} if nested else expected
    assert_image_equal(materialized, [[expected_value, None]])
    image_input = materialized[0][0]["images"][0] if nested else materialized[0][0]
    image_input[:] = 99
    relation = duckdb_cursor.from_arrow(table)
    assert relation.types == [declared]
    assert_image_equal(relation.fetchall(), [({"images": [expected, None]} if nested else expected,), (None,)])


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("kind", ["array", "map_key", "map_value", "struct"])
def test_image_udf_inputs_preserve_sliced_nested_containers(fixed, kind):
    image_type = vane.image_type("RGB", 1, 2) if fixed else vane.image_type()
    pixels = np.arange(6, dtype=np.uint8).reshape(1, 2, 3)
    if kind == "array":
        dtype = vane.array_type(image_type, 2)
        value = expected = (pixels, None)
    elif kind == "map_key":
        dtype = vane.map_type(image_type, vane.sqltypes.INTEGER)
        value = expected = {"key": [pixels], "value": [7]}
    elif kind == "map_value":
        dtype = vane.map_type(vane.sqltypes.VARCHAR, image_type)
        value = expected = {"present": pixels, "missing": None}
    else:
        dtype = vane.struct_type(
            {"image": image_type, "file": vane.file_type(), "ordinary": vane.array_type(vane.sqltypes.INTEGER, 2)}
        )
        value = {"image": pixels, "file": vane.File("missing.png"), "ordinary": (1, 2)}
        expected = {**value, "ordinary": [1, 2]}
    contract = FileUDFContract("nested_images", (dtype,), (dtype,))
    array = contract.scalar_outputs_to_array([None, value, None, value])
    chunks = pa.chunked_array([array.slice(1, 2), array.slice(3, 1)])
    assert_image_equal(contract.materialize_scalar_inputs(pa.table({"value": chunks})), [[expected, None, expected]])


@pytest.mark.parametrize("fixed", [False, True])
def test_image_udf_inputs_accept_validated_canonical_arrow_storage(fixed):
    dtype = vane.image_type("RGB", 1, 2) if fixed else vane.image_type()
    pixels = np.arange(6, dtype=np.uint8).reshape(1, 2, 3)
    contract = FileUDFContract("canonical_image", (dtype,), (dtype,))
    storage = contract.scalar_outputs_to_array([None, pixels, None]).storage.slice(1, 2)
    assert_image_equal(contract.materialize_scalar_inputs(pa.table({"image": storage})), [[pixels, None]])


@pytest.mark.parametrize("mode", list(vane.ImageMode))
@pytest.mark.parametrize("fixed", [False, True])
def test_image_batch_preserves_sliced_chunks_and_nulls(duckdb_cursor, mode, fixed):
    channels = list(vane.ImageMode).index(mode) + 1
    dtype = vane.image_type(mode, 1, 2) if fixed else vane.image_type(mode)
    arrow_type = image_arrow_type(dtype)
    pixels = list(range(2 * channels))
    storage = pixels if fixed else {"data": pixels, "channel": channels, "height": 1, "width": 2, "mode": channels}
    values = pa.ExtensionArray.from_storage(
        arrow_type, pa.array([None, storage, None, storage], type=arrow_type.storage_type)
    )
    chunks = pa.chunked_array([values.slice(1, 2), values.slice(3, 1)], type=arrow_type)

    @vane.func.batch(return_dtype=dtype)
    def identity(value):
        assert value.type == arrow_type
        return value

    vane.attach_function(identity, connection=duckdb_cursor, alias="sliced_image", parameters=[dtype])
    duckdb_cursor.register("sliced_images", pa.table({"ordinal": [0, 1, 2], "image": chunks}))
    result = duckdb_cursor.sql("SELECT sliced_image(image) FROM sliced_images ORDER BY ordinal")
    assert result.types == [dtype]
    expected = np.array(pixels, dtype=np.uint8).reshape(1, 2, channels)
    assert_image_equal(result.fetchall(), [(expected,), (None,), (expected,)])


def test_image_cast_and_attributes_across_vector_boundaries(duckdb_cursor):
    relation = duckdb_cursor.sql("""
        SELECT i, image_width(value), image_height(value), image_channel(value),
               image_mode(value), TRY_CAST(value AS IMAGE('RGB', 1, 1)) AS fixed
        FROM (
            SELECT i, image(CASE WHEN i % 2 = 0 THEN 'abc'::BLOB ELSE 'abcdef'::BLOB END,
                            (CASE WHEN i % 2 = 0 THEN 1 ELSE 2 END)::UINTEGER, 1, 3, 'RGB') AS value
            FROM range(4101) t(i)
        )
        WHERE i % 3 != 0 ORDER BY i
    """)
    for i, width, height, channels, mode, image in relation.fetchall():
        assert (width, height, channels, mode) == (1 if i % 2 == 0 else 2, 1, 3, 3)
        if i % 2:
            assert image is None
        else:
            assert_image_equal(image, np.array([97, 98, 99], dtype=np.uint8).reshape(1, 1, 3))


@pytest.mark.parametrize(
    "metadata", [b"{}", b'{"mode":"L","height":1,"width":2}', b'{"mode":"L","height":null,"width":null,"extra":0}']
)
def test_native_image_arrow_import_validates_metadata(duckdb_cursor, metadata):
    # Field metadata exercises the native C Data importer directly, without
    # invoking the registered Python extension deserializer first.
    storage_type = image_arrow_type(vane.image_type()).storage_type
    field = pa.field(
        "image", storage_type, metadata={b"ARROW:extension:name": b"vane.image", b"ARROW:extension:metadata": metadata}
    )
    table = pa.Table.from_arrays([pa.array([], type=storage_type)], schema=pa.schema([field]))
    with pytest.raises(vane.InvalidInputException, match="Image"):
        duckdb_cursor.from_arrow(table).fetchall()


@pytest.mark.parametrize("nested", [False, True])
def test_fixed_image_case_preserves_selected_rows_and_nulls(duckdb_cursor, nested):
    dtype = vane.image_type("RGB", 64, 64)
    left = np.full((64, 64, 3), 11, dtype=np.uint8)
    right = np.full((64, 64, 3), 29, dtype=np.uint8)
    if nested:
        dtype = vane.array_type(dtype, 2)
        left, right = (left, None), (right, None)
    relation = duckdb_cursor.sql(
        "SELECT i, CASE WHEN i % 3 = 0 THEN NULL WHEN i % 3 = 1 THEN $1 ELSE $2 END AS image "
        "FROM range(21) t(i) WHERE i % 2 = 0 ORDER BY i DESC",
        params=[vane.Value(left, dtype), vane.Value(right, dtype)],
    )
    assert relation.types[1] == dtype
    assert_image_equal(
        relation.fetchall(),
        [(i, None if i % 3 == 0 else left if i % 3 == 1 else right) for i in range(20, -1, -2)],
    )


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("consumer", ["fetchall", "to_arrow_table", "fetchnumpy"])
def test_empty_fixed_image_query_does_not_allocate_pixel_capacity(nested, consumer):
    image = "IMAGE('RGB', 5000, 5000)"
    dtype = f"STRUCT(image {image})" if nested else image
    with vane.connect(config={"memory_limit": "32MB"}) as con:
        con.execute(f"CREATE TABLE images(value {dtype})")
        relation = con.sql("SELECT * FROM images LIMIT 0")
        assert relation.types[0] == (
            vane.struct_type({"image": vane.image_type("RGB", 5000, 5000)})
            if nested
            else vane.image_type("RGB", 5000, 5000)
        )
        output = getattr(relation, consumer)()
        if consumer == "fetchall":
            assert output == []
        elif consumer == "to_arrow_table":
            assert output.num_rows == 0
        else:
            assert len(output["value"]) == 0
