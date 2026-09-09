# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pytest

import vane
from tests.fast.test_native_media_extensions import _connect
from vane._tensor import tensor_arrow_type

MODES = ("L", "LA", "RGB", "RGBA")


def _types(mode, form, height=2, width=3):
    channels = MODES.index(mode) + 1
    image = (
        vane.image_type()
        if form == "generic"
        else vane.image_type(mode)
        if form == "mode"
        else vane.image_type(mode, height, width)
    )
    shape = (height, width, channels) if form == "fixed" else (None, None, None if form == "generic" else channels)
    return image, vane.tensor_type(vane.sqltypes.UTINYINT, shape), tensor_arrow_type(pa.uint8(), shape)


def _assert_cell(value, expected, fixed):
    if expected is None:
        assert value is None
    elif fixed:
        # Existing fixed Tensor scalar materialization is a flat ARRAY tuple;
        # its HWC dimensions are carried by the logical/Arrow type.
        assert isinstance(value, tuple)
        np.testing.assert_array_equal(value, expected.ravel())
    else:
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.uint8
        assert value.shape == expected.shape
        np.testing.assert_array_equal(value, expected)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("form", ["generic", "mode", "fixed"])
def test_image_to_tensor_function_method_sql_and_arrow(mode, form):
    image_type, tensor_type, arrow_type = _types(mode, form)
    pixels = np.arange(6 * (MODES.index(mode) + 1), dtype=np.uint8).reshape(2, 3, -1)
    with vane.connect() as con:
        source = con.sql("SELECT $1 AS image", params=[vane.Value(pixels, image_type)])
        for expression in (vane.image_to_tensor(vane.col("image")), vane.col("image").image_to_tensor()):
            result = source.select(expression.alias("tensor"))
            assert result.types == [tensor_type]
            _assert_cell(result.fetchone()[0], pixels, form == "fixed")
        result = con.sql("SELECT image_to_tensor($1) AS tensor", params=[vane.Value(pixels, image_type)])
        assert result.types == [tensor_type]
        table = result.to_arrow_table()
        assert table.schema.field(0).type.equals(arrow_type)
        if form == "fixed":
            np.testing.assert_array_equal(table.column(0).combine_chunks().to_numpy_ndarray(), pixels[None, ...])
        with pa.BufferOutputStream() as sink:
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
            restored = pa.ipc.open_stream(sink.getvalue()).read_all()
        del source, result, table
        scanned = con.from_arrow(restored)
        assert scanned.types == [tensor_type]
        _assert_cell(scanned.fetchone()[0], pixels, form == "fixed")


@pytest.mark.parametrize("form", ["generic", "mode", "fixed"])
def test_image_to_tensor_null_empty_and_prepared_inputs(form):
    image_type, tensor_type, arrow_type = _types("LA", form)
    with vane.connect() as con:
        for tail, count in (("FROM range(2051)", 2051), ("WHERE FALSE", 0)):
            result = con.sql(f"SELECT image_to_tensor(NULL::{image_type}) AS value {tail}")
            assert result.types == [tensor_type]
            table = result.to_arrow_table()
            assert table.num_rows == count
            assert table.column(0).null_count == count
            assert table.column(0).type.equals(arrow_type)
        assert con.sql("SELECT image_to_tensor(NULL)").types == [_types("L", "generic")[1]]
        assert con.sql("SELECT image_to_tensor(NULL)").fetchone() == (None,)
        con.execute("PREPARE to_tensor AS SELECT image_to_tensor($1)")
        result = con.sql(f"EXECUTE to_tensor(image('abcd'::BLOB,2,1,2,'LA')::{_types('LA', form, 1, 2)[0]})")
        assert result.types == [_types("LA", form, 1, 2)[1]]
        _assert_cell(result.fetchone()[0], np.arange(97, 101, dtype=np.uint8).reshape(1, 2, 2), form == "fixed")


@pytest.mark.parametrize(
    "sql", ["1", "'bytes'::BLOB", "[1,2,3]", "file('missing.png',NULL,NULL,NULL,NULL)", "{'data': [1]}"]
)
def test_image_to_tensor_requires_an_image(sql):
    with vane.connect() as con, pytest.raises(vane.BinderException, match="requires IMAGE"):
        con.sql(f"SELECT image_to_tensor({sql}) WHERE FALSE")


def test_image_to_tensor_accepts_strided_numpy_and_pil():
    pixels = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)[:, ::-2, :]
    with vane.connect() as con:
        _assert_cell(con.sql("SELECT 1").select(vane.image_to_tensor(pixels)).fetchone()[0], pixels, False)
        pil = pytest.importorskip("PIL.Image").fromarray(pixels)
        _assert_cell(con.sql("SELECT 1").select(vane.image_to_tensor(pil)).fetchone()[0], pixels, False)


def test_image_to_tensor_is_base_cpp_for_both_backend_settings():
    program = """
import importlib.abc
import sys
class NoCodecs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PIL', 'av', 'soundfile', 'soxr'}:
            raise AssertionError('image_to_tensor imported a codec: ' + fullname)
sys.meta_path.insert(0, NoCodecs())
import vane
for backend in ('python', 'native'):
    with vane.connect(config={'image_backend': backend}) as con:
        assert con.sql("SELECT count(*) FROM duckdb_extensions() WHERE extension_name='image' AND loaded").fetchone()[0] == 0
        for image_type in ("IMAGE", "IMAGE('RGB')", "IMAGE('RGB',1,1)"):
            result = con.sql("SELECT image_to_tensor(image('abc'::BLOB,1,1,3,'RGB')::" + image_type + ")")
            assert str(result.types[0].id) == 'tensor'
            assert result.to_arrow_table().num_rows == 1
"""
    result = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("form", ["generic", "mode", "fixed"])
def test_image_to_tensor_survives_selection_storage_and_nested_growth(tmp_path, form):
    image_type, tensor_type, arrow_type = _types("RGB", form, 1, 1)
    database = str(tmp_path / "image-tensors.db")
    with vane.connect(database) as con:
        con.execute(f"""CREATE TABLE tensors AS
            SELECT i, image_to_tensor((CASE WHEN i%5=0 THEN NULL ELSE
                image(from_hex(repeat(lpad(to_hex(i%256),2,'0'),3)),1,1,3,'RGB') END)::{image_type}) AS value
            FROM range(4099) t(i)""")
        con.execute("DELETE FROM tensors WHERE i=1")
        con.execute(f"INSERT INTO tensors VALUES (1, NULL::{tensor_type})")
        con.execute("CHECKPOINT")
    with vane.connect(database) as con:
        result = con.sql("""SELECT a.i,a.value FROM tensors a JOIN tensors b USING(i)
                             WHERE a.i%7=1 ORDER BY a.i DESC""")
        assert result.types[-1] == tensor_type
        for index, value in result.fetchall():
            expected = None if index % 5 == 0 or index == 1 else np.full((1, 1, 3), index % 256, dtype=np.uint8)
            _assert_cell(value, expected, form == "fixed")
        rows = con.sql("SELECT list(value ORDER BY i) FROM tensors").fetchone()[0]
        assert len(rows) == 4099
        for index in (0, 1, 2047, 2048, 2050, 4098):
            expected = None if index % 5 == 0 or index == 1 else np.full((1, 1, 3), index % 256, dtype=np.uint8)
            _assert_cell(rows[index], expected, form == "fixed")
        arrow = con.sql("SELECT value FROM tensors ORDER BY i").to_arrow_table().slice(2045, 9)
        assert arrow.column(0).type.equals(arrow_type)
        rescanned = con.from_arrow(arrow)
        assert rescanned.types == [tensor_type]
        for index, (value,) in enumerate(rescanned.fetchall(), start=2045):
            expected = None if index % 5 == 0 else np.full((1, 1, 3), index % 256, dtype=np.uint8)
            _assert_cell(value, expected, form == "fixed")


def test_image_to_tensor_preserves_per_row_mode_and_dimensions():
    with vane.connect() as con:
        result = con.sql("""SELECT i, image_to_tensor(value) FROM (
            SELECT i, CASE WHEN i%5=0 THEN NULL ELSE
                image(from_hex(repeat('f0',((i%3+1)*(i%4+1))::INTEGER)),(i%3+1)::UINTEGER,1,
                      (i%4+1)::UTINYINT,['L','LA','RGB','RGBA'][i%4+1]) END AS value
            FROM range(4099) t(i)) WHERE i%7=1 ORDER BY i DESC""")
        assert result.types[1] == _types("L", "generic")[1]
        for index, value in result.fetchall():
            expected = None if index % 5 == 0 else np.full((1, index % 3 + 1, index % 4 + 1), 240, dtype=np.uint8)
            _assert_cell(value, expected, False)


def test_fixed_image_tensors_survive_case_and_coalesce():
    _, dtype, arrow_type = _types("RGB", "fixed", 1, 1)
    with vane.connect() as con:
        result = con.sql("""WITH tensors AS (
            SELECT i, image_to_tensor(image(repeat(chr((65+i%26)::INTEGER),3)::BLOB,
                1,1,3,'RGB')::IMAGE('RGB',1,1)) AS value,
                image_to_tensor(image('xyz'::BLOB,1,1,3,'RGB')::IMAGE('RGB',1,1)) AS other
            FROM range(4099) t(i)
        ) SELECT i,
            CASE WHEN i%3=0 THEN value WHEN i%3=1 THEN other END AS chosen,
            coalesce(CASE WHEN i%3=0 THEN value END, CASE WHEN i%3=1 THEN other END) AS combined
          FROM tensors WHERE i%7<>0 ORDER BY i DESC""")
        assert result.types == [vane.sqltypes.BIGINT, dtype, dtype]
        table = result.to_arrow_table()
        assert table.num_rows == 3513
        assert table.column(1).type.equals(arrow_type)
        assert table.column(2).type.equals(arrow_type)
        for index, chosen, combined in con.from_arrow(table).fetchall():
            expected = (
                np.full((1, 1, 3), 65 + index % 26, dtype=np.uint8)
                if index % 3 == 0
                else np.array([[[120, 121, 122]]], dtype=np.uint8)
                if index % 3 == 1
                else None
            )
            _assert_cell(chosen, expected, True)
            _assert_cell(combined, expected, True)


@pytest.mark.parametrize("form", ["generic", "mode", "fixed"])
@pytest.mark.parametrize("batch", [False, True])
def test_image_to_tensor_python_and_registered_sql_udfs(form, batch):
    image_type, tensor_type, arrow_type = _types("RGB", form, 1, 2)

    def identity(value):
        if batch:
            assert value.type.equals(arrow_type)
        elif value is not None:
            _assert_cell(value, np.arange(97, 103, dtype=np.uint8).reshape(1, 2, 3), form == "fixed")
        return value

    udf = (vane.func.batch if batch else vane.func)(return_dtype=tensor_type)(identity)
    with vane.connect() as con:
        vane.attach_function(udf, connection=con, alias="tensor_identity", parameters=[tensor_type])
        source = con.sql(f"""SELECT (CASE WHEN i=1 THEN NULL ELSE
            image('abcdef'::BLOB,2,1,3,'RGB') END)::{image_type} AS image FROM range(3) t(i)""")
        for result in (
            source.select(udf(vane.col("image").image_to_tensor())),
            source.select(vane.FunctionExpression("tensor_identity", vane.col("image").image_to_tensor())),
        ):
            assert result.types == [tensor_type]
            for index, (value,) in enumerate(result.fetchall()):
                expected = None if index == 1 else np.arange(97, 103, dtype=np.uint8).reshape(1, 2, 3)
                _assert_cell(value, expected, form == "fixed")


@pytest.mark.parametrize("backend", ["python", "native"])
def test_decode_crop_resize_convert_to_tensor_pipeline(tmp_path, backend):
    pil = pytest.importorskip("PIL.Image")
    pixels = np.full((3, 4, 3), [64, 128, 192], dtype=np.uint8)
    path = str(tmp_path / "source.png")
    pil.fromarray(pixels).save(path)
    with _connect("image") if backend == "native" else vane.connect() as con:
        result = con.sql(
            """SELECT image_to_tensor(convert_image(
            resize(crop(decode_image_file(image_file($1),'RGB')::IMAGE('RGB'),[1,1,2,1]),3,2),'RGBA'))""",
            params=[path],
        )
        assert result.types == [_types("RGBA", "fixed", 2, 3)[1]]
        table = result.to_arrow_table()
    expected = np.full((1, 2, 3, 4), [64, 128, 192, 255], dtype=np.uint8)
    np.testing.assert_array_equal(table.column(0).combine_chunks().to_numpy_ndarray(), expected)


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux address-space accounting")
@pytest.mark.parametrize("form", ["fixed", "mode", "generic"])
def test_image_to_tensor_allocates_for_actual_rows(form):
    # A full vector of 4K RGBA values exceeds 60 GiB. One converted value,
    # including the Arrow export, must fit in the bounded extra working memory.
    program = """
import resource
import sys
from pathlib import Path
import numpy as np
import vane
from vane.execution.udf_file_contract import FileUDFContract
with vane.connect(config={'threads':1}) as con:
    con.sql("SELECT image_to_tensor(image('abc'::BLOB,1,1,3,'RGB')::IMAGE('RGB',1,1))").to_arrow_table()
    pixels = np.full((2160,3840,4), [10,20,30,255], dtype=np.uint8)
    form = sys.argv[1]
    dtype = (vane.image_type('RGBA',2160,3840) if form == 'fixed' else
             vane.image_type('RGBA') if form == 'mode' else vane.image_type())
    value = vane.Value(pixels, dtype)
    vm = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                  if line.startswith('VmSize:'))) * 1024
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    ceiling = vm + 512 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (ceiling if hard < 0 else min(ceiling,hard), hard))
    result = con.sql('SELECT image_to_tensor($1) AS value', params=[value])
    table = result.to_arrow_table()
    if form == 'fixed':
        contract = FileUDFContract.from_payload({
            'udf_name': 'large_tensor',
            'output_schema': [{'kind': 'tensor', 'dtype': 'UTINYINT', 'shape': [2160,3840,4]}],
        })
        assert contract.output_types == (result.types[0],)
        normalized = contract.normalize_output_table(table)
        assert normalized.schema.equals(table.schema)
        assert (normalized.column(0).chunk(0).storage.values.buffers()[1].address ==
                table.column(0).chunk(0).storage.values.buffers()[1].address)
    column = table.column(0).combine_chunks()
    array = (column.to_numpy_ndarray() if form == 'fixed' else
             column.storage.field('data').values.to_numpy().reshape(1,2160,3840,4))
    assert array.dtype == np.uint8 and array.shape == (1,2160,3840,4)
    for row in range(0,2160,16):
        np.testing.assert_array_equal(array[0,row:row+16], pixels[row:row+16])
    empty = con.sql('SELECT image_to_tensor(NULL::' + str(dtype) + ') WHERE FALSE').to_arrow_table()
    assert empty.num_rows == 0
"""
    result = subprocess.run([sys.executable, "-I", "-c", program, form], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("element", "arrow_element"),
    [
        ("BOOLEAN", pa.bool_()),
        ("TINYINT", pa.int8()),
        ("SMALLINT", pa.int16()),
        ("INTEGER", pa.int32()),
        ("BIGINT", pa.int64()),
        ("UTINYINT", pa.uint8()),
        ("USMALLINT", pa.uint16()),
        ("UINTEGER", pa.uint32()),
        ("UBIGINT", pa.uint64()),
        ("FLOAT", pa.float32()),
        ("DOUBLE", pa.float64()),
    ],
)
@pytest.mark.parametrize("annotated", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_nullable_fixed_tensor_normalization_preserves_sliced_child_buffers(element, arrow_element, annotated, nested):
    from vane.execution.udf_file_contract import normalize_file_arrow_array

    dtype = vane.tensor_type(vane.type(element), (2, 3))
    arrow_type = tensor_arrow_type(arrow_element, (2, 3))
    zero, one = (False, True) if element == "BOOLEAN" else (0, 1)
    row = [zero, one, None, zero, one, zero]
    storage = pa.array([[zero] * 6, row, None, [one] * 6, [zero] * 6], type=arrow_type.storage_type)
    source = (
        pa.ExtensionArray.from_storage(
            pa.fixed_shape_tensor(arrow_type.value_type, (2, 3), permutation=(0, 1)), storage
        )
        if annotated
        else storage
    )
    if nested:
        source = pa.StructArray.from_arrays(
            [source], names=["PIXELS"], mask=pa.array([False, False, False, True, False])
        )
        dtype = vane.struct_type({"pixels": dtype})
    # Include a mixed-validity chunk, a fully inactive nested chunk and an
    # empty slice. Both parent and child validity have nonzero source offsets.
    column = pa.chunked_array([source.slice(1, 2), source.slice(3, 1), source.slice(2, 0)])
    normalized = normalize_file_arrow_array(column, dtype, boundary="test output")
    expected = [{"pixels": row}, {"pixels": None}, None] if nested else [row, None, [one] * 6]
    assert normalized.to_pylist() == expected
    for before, after in zip(column.chunks, normalized.chunks, strict=True):
        if nested:
            before, after = before.field("PIXELS"), after.field("pixels")
        before = before.storage if annotated else before
        assert after.type.equals(arrow_type)
        values = after.storage.values
        assert values.offset == before.values.offset + before.offset * 6
        assert len(values) == len(before) * 6
        for original, reused in zip(before.values.buffers(), values.buffers(), strict=True):
            assert (original.address if original else None) == (reused.address if reused else None)


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux address-space accounting")
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("execution", ["contract", "subprocess"])
def test_nullable_large_tensor_udf_normalization_has_bounded_memory(nested, execution):
    program = """
import resource
import sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import vane
from vane.execution.udf_file_contract import FileUDFContract

nested = sys.argv[1] == 'True'
shape = (2160,3840,4)
width = int(np.prod(shape))
arrow_type = pa.fixed_shape_tensor(pa.uint8(), shape)
dtype = vane.tensor_type(vane.sqltypes.UTINYINT, shape)
output_type = vane.struct_type({'pixels': dtype}) if nested else dtype

def make_table(_):
    pixels = pa.array(np.full(3 * width, 7, dtype=np.uint8))
    # The first row is outside the returned slice. For nested output, the
    # second returned Tensor is valid but hidden by its NULL STRUCT parent.
    storage = pa.Array.from_buffers(
        arrow_type.storage_type, 3,
        [None if nested else pa.py_buffer(b'\\x03')], children=[pixels])
    values = pa.ExtensionArray.from_storage(arrow_type, storage)
    if nested:
        values = pa.StructArray.from_arrays(
            [values], names=['PIXELS'], mask=pa.array([False,False,True]))
    table = pa.table({'value': values.slice(1,2)})
    vm = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                  if line.startswith('VmSize:'))) * 1024
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    ceiling = vm + 512 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (ceiling if hard < 0 else min(ceiling,hard), hard))
    return table

if sys.argv[2] == 'contract':
    contract = FileUDFContract.from_payload({
        'udf_name': 'nullable_large_tensor',
        'output_schema': [{'name': 'value', 'kind': 'duckdb_type', 'type': str(output_type)}],
    })
    source = make_table(None)
    output = contract.normalize_output_table(source)
    before = source.column(0).chunk(0)
    after = output.column(0).chunk(0)
    if nested:
        before, after = before.field('PIXELS'), after.field('pixels')
    assert before.storage.values.buffers()[1].address == after.storage.values.buffers()[1].address
else:
    with vane.connect(config={'threads':1}) as con:
        result = con.sql('SELECT i FROM range(2) t(i)').map_batches(
            make_table, schema={'value': output_type}, batch_size=2,
            execution_backend='subprocess_task')
        assert result.types == [output_type]
        output = result.to_arrow_table()
assert output.num_rows == 2
column = output.column(0).chunk(0)
assert column.is_valid().to_pylist() == [True,False]
if nested:
    column = column.field('pixels')
assert column.type.equals(arrow_type)
storage = column.storage
valid_pixels = storage.values.slice(storage.offset * width, width)
assert pc.min_max(valid_pixels).as_py() == {'min': 7, 'max': 7}
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program, str(nested), execution],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("malformed", ["shape", "length", "permutation", "names"])
def test_fixed_tensor_udf_rejects_mismatched_outputs(malformed):
    dtype = _types("RGB", "fixed", 1, 2)[1]
    if malformed != "length":

        @vane.func.batch(return_dtype=dtype)
        def invalid(values):
            metadata = (
                {"permutation": [2, 1, 0]}
                if malformed == "permutation"
                else {"dim_names": ["height", "width", "channel"]}
                if malformed == "names"
                else {}
            )
            shape = (6,) if malformed == "shape" else (1, 2, 3)
            return pa.ExtensionArray.from_storage(pa.fixed_shape_tensor(pa.uint8(), shape, **metadata), values.storage)

    else:

        @vane.func(return_dtype=dtype)
        def invalid(values):
            return (1, 2)

    with vane.connect() as con, pytest.raises(vane.Error, match="tensor metadata|fixed size|length|size 6"):
        source = con.sql("SELECT image_to_tensor(image('abcdef'::BLOB,2,1,3,'RGB')::IMAGE('RGB',1,2)) AS value")
        source.select(invalid(vane.col("value"))).fetchall()


@pytest.mark.parametrize(
    "element",
    [
        "BOOLEAN",
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "FLOAT",
        "DOUBLE",
    ],
)
def test_fixed_numeric_tensor_storage_after_deferred_allocation(element):
    dtype = vane.tensor_type(vane.type(element), (2, 2))
    with vane.connect() as con:
        # JSON's typed ARRAY writer must also reserve deferred Tensor storage.
        parsed = con.sql("SELECT from_json('[0,1,null,1]', $1)", params=[json.dumps(str(dtype))])
        assert parsed.types == [dtype]
        assert parsed.fetchone()[0] == (0, 1, None, 1)
        con.execute(f"""CREATE TABLE values_table AS SELECT i,
            (CASE WHEN i%3=0 THEN NULL ELSE [0,1,NULL,1] END)::{dtype} AS value FROM range(4099) t(i)""")
        rows = con.sql("SELECT list(value ORDER BY i) FROM values_table").fetchone()[0]
        assert len(rows) == 4099
        assert rows[0] is None and rows[1] == rows[4097] == (0, 1, None, 1)
        result = con.sql("SELECT a.value FROM values_table a JOIN values_table b USING(i) WHERE i%7=1 ORDER BY i")
        assert result.types == [dtype]
        arrow = result.to_arrow_table()
        assert arrow.column(0).type.extension_name == "arrow.fixed_shape_tensor"
        assert con.from_arrow(arrow).fetchall() == [(rows[i],) for i in range(1, 4099, 7)]
