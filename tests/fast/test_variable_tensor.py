# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pickle
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pytest

import vane
from vane._tensor import VariableShapeTensorType, _validate_arrow_tensor_type
from vane.sqltypes import BOOLEAN, DOUBLE, FLOAT, INTEGER


def _dtype(shape=(None, None)):
    return vane.tensor_type(DOUBLE, shape)


def _rows():
    return [np.arange(6, dtype=np.float64).reshape(3, 2), np.ones((5, 1)), np.empty((0, 2)), None]


def _assert_rows(actual, expected):
    assert len(actual) == len(expected)
    for value, wanted in zip(actual, expected, strict=True):
        if wanted is None:
            assert value is None
        else:
            assert isinstance(value, np.ndarray)
            assert value.dtype == wanted.dtype
            assert value.shape == wanted.shape
            np.testing.assert_array_equal(value, wanted)


def test_variable_tensor_type_contract_and_pickle():
    dtype = _dtype((None, 2))
    assert str(dtype) == "TENSOR(DOUBLE, [NULL, 2])"
    assert str(dtype.id) == "tensor"
    assert dict(dtype.children) == {"dtype": DOUBLE, "shape": (None, 2)}
    assert vane.type(str(dtype)) == dtype
    assert pickle.loads(pickle.dumps(dtype)) == dtype
    assert dtype != _dtype()
    assert dtype != vane.tensor_type(DOUBLE, (3, 2))
    with vane.connect() as connection:
        assert connection.sql("SELECT typeof(NULL::TENSOR(DOUBLE, [NULL, 2]))").fetchone()[0] == str(dtype)


@pytest.mark.parametrize(
    "shape", [(), (True, None), (np.bool_(True), None), (-1, None), (2**31, None), (None,) * 33, (1.5, None)]
)
def test_invalid_shape_declarations(shape):
    with pytest.raises((vane.InvalidInputException, TypeError, ValueError, OverflowError)):
        _dtype(shape)


@pytest.mark.parametrize("dtype", [vane.sqltypes.VARCHAR, vane.sqltypes.HUGEINT, vane.file_type()])
def test_variable_tensor_rejects_unsupported_elements(dtype):
    with pytest.raises(vane.InvalidInputException, match="numeric element"):
        vane.tensor_type(dtype, (None,))


def test_sql_constructor_preserves_variable_shapes_empty_and_null():
    with vane.connect() as connection:
        result = connection.sql(
            """
            SELECT tensor(data, shape) AS waveform FROM (VALUES
                ([0.,1.,2.,3.,4.,5.]::DOUBLE[], [3,2]::INTEGER[2]),
                ([1.,1.,1.,1.,1.]::DOUBLE[], [5,1]::INTEGER[2]),
                ([]::DOUBLE[], [0,2]::INTEGER[2]),
                (NULL::DOUBLE[], [9,1]::INTEGER[2])
            ) t(data, shape)
            """
        )
        assert result.types == [_dtype()]
        _assert_rows([row[0] for row in result.fetchall()], _rows())
        assert connection.sql("SELECT tensor_shape(tensor([1.,2.]::DOUBLE[], [2,1]))").fetchone()[0] == (2, 1)
        assert connection.sql("SELECT tensor_data(tensor([1.,2.]::DOUBLE[], [2,1]))").fetchone()[0] == [1.0, 2.0]


@pytest.mark.parametrize(
    ("data", "shape", "error"),
    [
        ("[1.,2.]", "[2,2]", "element count"),
        ("[1.,NULL]", "[2,1]", "cannot be NULL"),
        ("[1.]", "[-1,1]", "nonnegative"),
        ("[1.]", "[NULL,1]", "dimensions cannot be NULL"),
        ("[]", "[2147483647,2147483647]", "element count"),
    ],
)
def test_constructor_rejects_invalid_values(data, shape, error):
    with vane.connect() as connection, pytest.raises(vane.InvalidInputException, match=error):
        connection.sql(f"SELECT tensor({data}::DOUBLE[], {shape})").fetchall()


def test_expression_constructor_and_no_implicit_struct_cast():
    with vane.connect() as connection:
        expression = vane.tensor([1.0, 2.0], [2, 1])
        _assert_rows([connection.sql("SELECT 1").select(expression).fetchone()[0]], [np.array([[1.0], [2.0]])])
        with pytest.raises(vane.BinderException):
            connection.sql(
                "SELECT {'data': [1.]::DOUBLE[], 'shape': [1,1]::INTEGER[2]}::TENSOR(DOUBLE, [NULL,NULL])"
            ).fetchall()


@pytest.mark.parametrize(
    ("dtype", "numpy_type"), [(DOUBLE, np.float64), (FLOAT, np.float32), (INTEGER, np.int32), (BOOLEAN, np.bool_)]
)
def test_numpy_arrow_ipc_round_trip(dtype, numpy_type):
    tensor_type = vane.tensor_type(dtype, (None, None))
    rows = [np.arange(6).astype(numpy_type).reshape(2, 3), np.ones((4, 1), dtype=numpy_type), None]
    column = vane.tensor_array(rows, tensor_type)
    assert column.type.extension_name == "arrow.variable_shape_tensor"
    assert column.type.storage_type == pa.struct(
        [("data", pa.list_(pa.from_numpy_dtype(numpy_type))), ("shape", pa.list_(pa.int32(), 2))]
    )
    table = pa.table({"value": column})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    restored = pa.ipc.open_stream(sink.getvalue()).read_all()
    assert restored.schema == table.schema
    assert restored.column(0).to_pylist() == column.to_pylist()
    assert pickle.loads(pickle.dumps(restored)).schema == table.schema
    with vane.connect() as connection:
        result = connection.from_arrow(restored)
        assert result.types == [tensor_type]
        _assert_rows([row[0] for row in result.fetchall()], rows)
        output = result.to_arrow_table()
        assert output.schema == table.schema
        assert output.column(0).to_pylist() == column.to_pylist()


def test_uniform_dimensions_typed_parameters_and_persistence(tmp_path):
    dtype = _dtype((None, 2))
    value = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::3]
    assert not value.flags.c_contiguous
    database = str(tmp_path / "tensors.db")
    with vane.connect(database) as connection:
        connection.execute("CREATE TABLE waves(value TENSOR(DOUBLE, [NULL,2]))")
        connection.execute("INSERT INTO waves VALUES (?)", [vane.Value(value, dtype)])
        connection.execute("INSERT INTO waves VALUES (NULL)")
        with pytest.raises((ValueError, vane.InvalidInputException), match="uniform dimensions"):
            connection.execute("INSERT INTO waves VALUES (?)", [vane.Value(np.ones((2, 1)), dtype)])
    with vane.connect(database) as connection:
        rows = connection.table("waves").fetchall()
        _assert_rows([row[0] for row in rows], [value, None])
        assert connection.table("waves").types == [dtype]
        rows[0][0][0, 0] = 99
        assert connection.table("waves").fetchone()[0][0, 0] == value[0, 0]


def test_arrow_options_do_not_change_canonical_tensor_storage():
    with vane.connect() as connection:
        connection.execute("SET arrow_large_buffer_size = true")
        connection.execute("SET arrow_output_list_view = true")
        connection.execute("SET arrow_lossless_conversion = true")
        result = connection.sql("SELECT tensor([TRUE,FALSE], [2]) AS value").to_arrow_table()
        assert result.column(0).type.storage_type == VariableShapeTensorType(pa.bool_(), (None,)).storage_type
        assert result.column(0)[0].as_py() == {"data": [True, False], "shape": [2]}


@pytest.mark.parametrize(
    "record",
    [
        {"data": [1.0], "shape": [2, 1]},
        {"data": [None], "shape": [1, 1]},
        {"data": [1.0], "shape": [-1, 1]},
        {"data": None, "shape": [0, 1]},
    ],
)
def test_arrow_import_rejects_malformed_tensor_values(record):
    dtype = VariableShapeTensorType(pa.float64(), (None, None))
    storage = pa.array([record], type=dtype.storage_type)
    array = pa.ExtensionArray.from_storage(dtype, storage)
    with vane.connect() as connection, pytest.raises(vane.InvalidInputException):
        connection.from_arrow(pa.table({"value": array})).fetchall()


@pytest.mark.parametrize(
    "metadata",
    [
        b'{"uniform_shape":[null,true]}',
        b'{"uniform_shape":[null,-1]}',
        b'{"permutation":[1,0]}',
        b'{"uniform_shape":[null]}',
        b'{"uniform_shape":[null,null],"uniform_shape":[null,null]}',
        b'{"dim_names":["x","y"]}',
    ],
)
def test_arrow_metadata_is_strict(metadata):
    storage = VariableShapeTensorType(pa.float64(), (None, None)).storage_type

    class RawTensorType(pa.ExtensionType):
        def __init__(self):
            super().__init__(storage, "arrow.variable_shape_tensor")

        def __arrow_ext_serialize__(self):
            return metadata

    with pytest.raises((ValueError, TypeError)):
        _validate_arrow_tensor_type(RawTensorType())
    array = pa.ExtensionArray.from_storage(RawTensorType(), pa.array([], type=storage))
    with vane.connect() as connection, pytest.raises((vane.InvalidInputException, vane.NotImplementedException)):
        connection.from_arrow(pa.table({"value": array})).fetchall()


def test_arrow_binding_preserves_foreign_layouts_without_accepting_them_in_vane():
    storage = VariableShapeTensorType(pa.float64(), (None, None)).storage_type
    metadata = b'{"uniform_shape":[null,2],"permutation":[1,0],"dim_names":["x","y"]}'
    dtype = VariableShapeTensorType.__arrow_ext_deserialize__(storage, metadata)
    array = pa.ExtensionArray.from_storage(dtype, pa.array([{"data": [1.0, 2.0], "shape": [1, 2]}], type=storage))
    restored = pickle.loads(pickle.dumps(array))
    assert restored.type.__arrow_ext_serialize__() == metadata
    assert restored.to_pylist() == array.to_pylist()
    with pytest.raises(ValueError, match="dimension names"):
        _validate_arrow_tensor_type(restored.type)
    with vane.connect() as connection, pytest.raises(vane.NotImplementedException):
        connection.from_arrow(pa.table({"value": restored})).fetchall()


def test_sliced_chunked_empty_and_all_null_tensors():
    dtype = _dtype()
    source = vane.tensor_array(_rows(), dtype)
    table = pa.table({"wave": pa.chunked_array([source.slice(1, 2), source.slice(3, 1)])})
    with vane.connect() as connection:
        _assert_rows([row[0] for row in connection.from_arrow(table).fetchall()], _rows()[1:])
        for column in (vane.tensor_array([], dtype), vane.tensor_array([None, None], dtype)):
            result = connection.from_arrow(pa.table({"wave": column}))
            assert result.types == [dtype]
            assert result.to_arrow_table().column(0).type == source.type


def test_ray_partition_materialization_releases_gil_for_arrow_tensor_callbacks(tmp_path):
    # A subprocess deadline also detects a native GIL deadlock, which cannot be
    # interrupted by pytest's Python signal handler in the same process.
    code = """
import numpy as np
import pyarrow as pa
import vane

dtype = vane.tensor_type(vane.sqltypes.DOUBLE, (None, None))
table = pa.table({'wave': vane.tensor_array([np.ones((2, 1)), np.empty((0, 2)), None], dtype)})
partition = vane.ray_cxx._RayBackedResultPartitionForTest(table)
assert partition.materialize() == 3
assert partition.materialize() == 3
"""
    subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, timeout=30, check=True, capture_output=True)


@pytest.mark.parametrize("container", ["list", "array", "struct", "dictionary"])
def test_inactive_tensor_payloads_are_not_validated(container):
    dtype = VariableShapeTensorType(pa.float64(), (None, None))
    storage = pa.array([{"data": [1.0], "shape": [9, 9]}, {"data": [2.0], "shape": [1, 1]}], type=dtype.storage_type)
    tensors = pa.ExtensionArray.from_storage(dtype, storage)
    mask = pa.array([True, False])
    if container == "list":
        column = pa.ListArray.from_arrays([0, 1, 2], tensors, mask=mask)
    elif container == "array":
        column = pa.FixedSizeListArray.from_arrays(tensors, 1, mask=mask)
    elif container == "struct":
        column = pa.StructArray.from_arrays([tensors], names=["wave"], mask=mask)
    else:
        column = pa.DictionaryArray.from_arrays(pa.array([None, 1], type=pa.int32()), tensors)
    with vane.connect() as connection:
        rows = connection.from_arrow(pa.table({"value": column})).fetchall()
        assert rows[0][0] is None
        assert len(rows) == 2
        value = rows[1][0]
        if container in ("list", "array"):
            value = value[0]
        elif container == "struct":
            value = value["wave"]
        np.testing.assert_array_equal(value, [[2.0]])


def test_scalar_and_batch_udfs_preserve_tensor_values():
    dtype = _dtype()

    @vane.func(return_dtype=dtype)
    def reverse_frames(value):
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.float64
        return value[::-1].copy()

    @vane.func.batch(return_dtype=dtype, batch_size=2)
    def identity(values):
        assert values.type.extension_name == "arrow.variable_shape_tensor"
        return values

    with vane.connect() as connection:
        source = connection.from_arrow(pa.table({"value": vane.tensor_array(_rows(), dtype)}))
        result = source.select(identity(reverse_frames(vane.col("value"))).alias("value"))
        assert result.types == [dtype]
        _assert_rows(
            [row[0] for row in result.fetchall()], [None if value is None else value[::-1] for value in _rows()]
        )


def test_udf_output_dtype_and_shape_are_validated():
    @vane.func(return_dtype=_dtype((None, 1)))
    def invalid(value):
        return np.ones((2, 2), dtype=np.float64)

    with vane.connect() as connection, pytest.raises(Exception, match="uniform dimensions"):
        connection.sql("SELECT 1 AS value").select(invalid(vane.col("value"))).fetchall()


@pytest.mark.parametrize("malformed", ["shape", "storage", "dtype"])
def test_batch_udf_rejects_invalid_tensor_outputs(malformed):
    @vane.func.batch(return_dtype=_dtype())
    def invalid(values):
        dtype = VariableShapeTensorType(pa.float32() if malformed == "dtype" else pa.float64(), (None, None))
        storage = pa.array([{"data": [1.0], "shape": [2 if malformed == "shape" else 1, 1]}], type=dtype.storage_type)
        return storage if malformed == "storage" else pa.ExtensionArray.from_storage(dtype, storage)

    with vane.connect() as connection, pytest.raises(Exception, match="Tensor"):
        connection.sql("SELECT 1 AS value").select(invalid(vane.col("value"))).fetchall()


def test_typed_tensor_parameter_rejects_wrong_dtype():
    with vane.connect() as connection, pytest.raises((ValueError, vane.InvalidInputException), match="dtype"):
        connection.execute("SELECT ?", [vane.Value(np.ones((2, 1), dtype=np.float32), _dtype())])


def test_structured_output_schema_keeps_variable_tensor_type():
    from vane.datasource import _schema_entry_to_arrow
    from vane.execution.udf_file_contract import _parse_tensor_type
    from vane.execution.udf_output_schema import _arrow_type_from_output_schema_entry

    entry = {"kind": "tensor", "dtype": "DOUBLE", "shape": [None, 2]}
    expected = VariableShapeTensorType(pa.float64(), (None, 2))
    assert _schema_entry_to_arrow(entry) == expected
    assert _arrow_type_from_output_schema_entry(entry) == expected
    assert _parse_tensor_type(entry, field="output_schema", governed_only=True) == _dtype((None, 2))


@pytest.mark.parametrize("boundary", ["datasource", "udf_output"])
@pytest.mark.parametrize(
    "dtype,shape",
    [
        ("DOUBLE", [None, -1]),
        ("DOUBLE", [None, True]),
        ("DOUBLE", [None, 1.5]),
        ("DOUBLE", [None, 2**31]),
        ("DOUBLE", [None] * 33),
        ("VARCHAR", [None, 2]),
    ],
)
def test_structured_variable_tensor_schema_rejects_invalid_contract(boundary, dtype, shape):
    from vane.datasource import _schema_entry_to_arrow
    from vane.execution.udf_output_schema import _arrow_type_from_output_schema_entry

    convert = _schema_entry_to_arrow if boundary == "datasource" else _arrow_type_from_output_schema_entry
    with pytest.raises((TypeError, ValueError)):
        convert({"kind": "tensor", "dtype": dtype, "shape": shape})


@pytest.mark.parametrize("shape", [[None, 0], [None, 2**31 - 1], [None] * 32])
def test_structured_variable_tensor_schema_accepts_dimension_and_rank_boundaries(shape):
    from vane.datasource import _schema_entry_to_arrow
    from vane.execution.udf_output_schema import _arrow_type_from_output_schema_entry

    entry = {"kind": "tensor", "dtype": "DOUBLE", "shape": shape}
    expected = VariableShapeTensorType(pa.float64(), shape)
    assert _schema_entry_to_arrow(entry) == expected
    assert _arrow_type_from_output_schema_entry(entry) == expected


@pytest.mark.parametrize("batch", [False, True])
def test_registered_sql_tensor_udfs(batch):
    dtype = _dtype()
    decorator = vane.func.batch if batch else vane.func

    @decorator(return_dtype=dtype)
    def identity(value):
        return value

    with vane.connect() as connection:
        vane.attach_function(identity, connection=connection, alias="tensor_identity", parameters=[dtype])
        result = connection.sql("SELECT tensor_identity(tensor([1.,2.]::DOUBLE[], [2,1]))")
        assert result.types == [dtype]
        np.testing.assert_array_equal(result.fetchone()[0], [[1.0], [2.0]])


def test_nested_tensors_and_nonfinite_values():
    dtype = _dtype()
    value = np.array([[np.nan, np.inf], [-np.inf, 1.0]], dtype=np.float64)
    with vane.connect() as connection:
        row = connection.execute("SELECT ?", [vane.Value(value, dtype)]).fetchone()[0]
        np.testing.assert_array_equal(row, value)
        array = vane.tensor_array([value, None], dtype)
        lists = pa.ListArray.from_arrays([0, 2], array)
        source = connection.from_arrow(pa.table({"values": lists}))
        assert source.types == [vane.list_type(dtype)]

        @vane.func(return_dtype=vane.list_type(dtype))
        def identity(values):
            assert isinstance(values[0], np.ndarray)
            return values

        row = source.select(identity(vane.col("values"))).fetchone()[0]
        _assert_rows(row, [value, None])


@pytest.mark.real_ray
@pytest.mark.usefixtures("ray_local")
def test_ray_tensor_udf_and_flight_shuffle(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    vane.teardown_runner()
    vane.set_runner_ray(noop_if_initialized=True)
    dtype = _dtype()

    @vane.func(return_dtype=dtype)
    def reverse_frames(value):
        assert isinstance(value, np.ndarray)
        return value[::-1].copy()

    with vane.connect() as connection:
        source = connection.sql(
            """SELECT i, tensor(range(i % 3)::DOUBLE[], [i % 3,1]::INTEGER[2]) AS value
               FROM range(32) t(i)"""
        )
        result = source.select(vane.col("i"), reverse_frames(vane.col("value")).alias("value")).order("i")
        assert result.types[1] == dtype
        rows = result.fetchall()
        assert len(rows) == 32
        for index, value in rows:
            np.testing.assert_array_equal(value, np.arange(index % 3, dtype=np.float64)[::-1].reshape(-1, 1))
    vane.teardown_runner()


def test_tensor_values_survive_more_than_one_vector_and_hash_join():
    with vane.connect() as connection:
        result = connection.sql(
            """WITH waves AS (
                SELECT i, tensor(list_transform(range((i % 3) + 1), x -> x::DOUBLE),
                                 [(i % 3) + 1, 1]::INTEGER[2]) AS wave
                FROM range(4101) t(i)
            ) SELECT a.i, a.wave FROM waves a JOIN range(4101) b(i) USING(i) ORDER BY a.i"""
        ).fetchall()
    assert len(result) == 4101
    for index, value in result:
        np.testing.assert_array_equal(value, np.arange((index % 3) + 1, dtype=np.float64).reshape(-1, 1))
