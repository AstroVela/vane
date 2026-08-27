# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pickle

import pandas as pd
import pytest

import vane

FILE_FIELDS = ("url", "content_type", "position", "size", "checksum")


def test_file_value_contract():
    minimal = vane.File("s3://bucket/object")
    assert vane.File is vane._native.File
    assert tuple(getattr(minimal, field) for field in FILE_FIELDS) == (
        "s3://bucket/object",
        None,
        None,
        None,
        None,
    )
    assert str(minimal) == "s3://bucket/object"
    assert repr(minimal) == "File(url='s3://bucket/object', content_type=None, position=None, size=None, checksum=None)"
    assert {name for name in dir(minimal) if not name.startswith("_")} == set(FILE_FIELDS)

    complete = vane.File("memory://part", "text/plain", 2, 4, "sha256:abcd")
    assert tuple(getattr(complete, field) for field in FILE_FIELDS) == (
        "memory://part",
        "text/plain",
        2,
        4,
        "sha256:abcd",
    )
    assert complete == vane.File("memory://part", "text/plain", 2, 4, "sha256:abcd")
    assert complete != minimal
    assert complete != "memory://part"
    assert hash(complete) == hash(pickle.loads(pickle.dumps(complete)))
    assert pickle.loads(pickle.dumps(complete)) == complete


def test_file_value_is_immutable_and_final():
    value = vane.File("memory://value")
    for field in FILE_FIELDS:
        with pytest.raises(AttributeError):
            setattr(value, field, None)
    with pytest.raises(AttributeError):
        value.secret = "credential"
    with pytest.raises(TypeError):

        class DerivedFile(vane.File):
            pass


@pytest.mark.parametrize("url", [None, 42, b"path"])
def test_file_value_requires_string_url(url):
    with pytest.raises(TypeError, match=r"File\.url must be str"):
        vane.File(url)


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"content_type": 1}, "content_type"),
        ({"checksum": b"sha256:a"}, "checksum"),
        ({"position": True, "size": 0}, "position"),
        ({"position": 0, "size": False}, "size"),
        ({"position": 0.0, "size": 1}, "position"),
        ({"position": 0, "size": "1"}, "size"),
    ],
)
def test_file_value_requires_exact_optional_types(kwargs, field):
    with pytest.raises(TypeError, match=rf"File\.{field} must be"):
        vane.File("memory://value", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": 0},
        {"size": 0},
        {"position": -1, "size": 0},
        {"position": 0, "size": -1},
        {"position": 2**63 - 1, "size": 1},
    ],
)
def test_file_value_rejects_invalid_ranges(kwargs):
    with pytest.raises(ValueError):
        vane.File("memory://value", **kwargs)


@pytest.mark.parametrize("field", ["position", "size"])
def test_file_value_rejects_bigint_overflow(field):
    kwargs = {"position": 0, "size": 0, field: 2**63}
    with pytest.raises(OverflowError, match="signed 64-bit"):
        vane.File("memory://value", **kwargs)


@pytest.mark.parametrize("checksum", ["", "sha256", ":digest", "sha256:", "sha256:a:b", "sha256:a\0b"])
def test_file_value_rejects_invalid_checksums(checksum):
    with pytest.raises(ValueError, match=r"<algorithm>:<digest>"):
        vane.File("memory://value", checksum=checksum)


def test_file_value_rejects_nul_url():
    with pytest.raises(ValueError, match="NUL"):
        vane.File("memory://bad\0url")


def test_file_data_type_contract():
    dtype = vane.file_type()
    plain_struct = vane.struct_type(
        {
            "url": vane.sqltypes.VARCHAR,
            "content_type": vane.sqltypes.VARCHAR,
            "position": vane.sqltypes.BIGINT,
            "size": vane.sqltypes.BIGINT,
            "checksum": vane.sqltypes.VARCHAR,
        }
    )

    assert str(dtype) == "FILE"
    assert dtype.id == "struct"
    assert dtype.is_file()
    assert dtype == vane.sqltype("FILE")
    assert not plain_struct.is_file()
    assert dtype.children == [
        ("url", vane.sqltypes.VARCHAR),
        ("content_type", vane.sqltypes.VARCHAR),
        ("position", vane.sqltypes.BIGINT),
        ("size", vane.sqltypes.BIGINT),
        ("checksum", vane.sqltypes.VARCHAR),
    ]
    assert dtype.url == vane.sqltypes.VARCHAR
    assert pickle.loads(pickle.dumps(dtype)).is_file()


def test_local_file_results_materialize_recursively(duckdb_cursor):
    value = vane.File("memory://nested", "text/plain", 1, 2, "sha256:nested")
    row = duckdb_cursor.sql(
        """
        SELECT
            file('memory://nested', 'text/plain', 1, 2, 'sha256:nested'),
            [file('memory://nested', 'text/plain', 1, 2, 'sha256:nested'), NULL::FILE],
            struct_pack(item := file('memory://nested', 'text/plain', 1, 2, 'sha256:nested')),
            map(['item'], [file('memory://nested', 'text/plain', 1, 2, 'sha256:nested')]),
            union_value(item := file('memory://nested', 'text/plain', 1, 2, 'sha256:nested'))
        """
    ).fetchone()

    assert row == (value, [value, None], {"item": value}, {"item": value}, value)


def test_file_dataframe_results_use_file_values(duckdb_cursor):
    frame = duckdb_cursor.sql(
        """
        SELECT value
        FROM (
            SELECT 0 AS sort_key, file('memory://a', 'text/plain', 0, 4, 'sha256:a') AS value
            UNION ALL
            SELECT 1, NULL::FILE
        )
        ORDER BY sort_key
        """
    ).df()

    values = frame["value"].tolist()
    assert values[0] == vane.File("memory://a", "text/plain", 0, 4, "sha256:a")
    assert pd.isna(values[1])


def test_pandas_file_columns_preserve_file_values(duckdb_cursor):
    values = [vane.File("memory://first"), None, vane.File("memory://second", "text/plain")]
    relation = duckdb_cursor.from_df(pd.DataFrame({"value": values}))

    assert relation.types[0].is_file()
    assert [row[0] for row in relation.fetchall()] == values


def test_plain_file_shaped_struct_remains_a_dictionary(duckdb_cursor):
    query = """
        SELECT struct_pack(
            url := 'memory://struct',
            content_type := NULL::VARCHAR,
            "position" := NULL::BIGINT,
            size := NULL::BIGINT,
            checksum := NULL::VARCHAR
        ) AS value
    """
    expected = {
        "url": "memory://struct",
        "content_type": None,
        "position": None,
        "size": None,
        "checksum": None,
    }

    value = duckdb_cursor.sql(query).fetchone()[0]
    columnar_value = duckdb_cursor.sql(query).df()["value"].iloc[0]
    assert value == expected
    assert columnar_value == expected
    assert not isinstance(value, vane.File)
    assert not isinstance(columnar_value, vane.File)


@pytest.mark.parametrize(
    ("value", "dtype", "expected"),
    [
        (None, vane.file_type(), None),
        (None, vane.list_type(vane.file_type()), None),
        ([], vane.list_type(vane.file_type()), []),
        ([None], vane.list_type(vane.file_type()), [None]),
        ([None], vane.array_type(vane.file_type(), 1), (None,)),
        ({"file": None}, vane.struct_type({"file": vane.file_type()}), {"file": None}),
        ({}, vane.map_type(vane.sqltypes.VARCHAR, vane.file_type()), {}),
        (
            {"key": ["one"], "value": [None]},
            vane.map_type(vane.sqltypes.VARCHAR, vane.file_type()),
            {"one": None},
        ),
        ([[]], vane.list_type(vane.list_type(vane.file_type())), [[]]),
    ],
)
def test_declared_types_preserve_file_through_empty_and_null_values(duckdb_cursor, value, dtype, expected):
    actual_type, actual = duckdb_cursor.execute("SELECT typeof($1), $1", [vane.Value(value, dtype)]).fetchone()
    assert actual_type == str(dtype)
    assert actual == expected


def test_file_values_cross_explicit_python_boundaries(duckdb_cursor):
    value = vane.File("memory://parameter", "text/plain", 0, 3, "sha256:parameter")

    relation = duckdb_cursor.values(vane.ConstantExpression(value))
    assert relation.types[0].is_file()
    assert relation.fetchone() == (value,)

    duckdb_cursor.execute("CREATE TABLE file_parameters(value FILE)")
    duckdb_cursor.execute("INSERT INTO file_parameters VALUES (?)", [value])
    assert duckdb_cursor.sql("SELECT value FROM file_parameters").fetchone() == (value,)
    assert duckdb_cursor.execute("SELECT ?", [[value, None]]).fetchone() == ([value, None],)


def test_typed_union_selects_file_and_composite_members(duckdb_cursor):
    value = vane.File("memory://union")
    plain_struct_type = vane.struct_type({"item": vane.file_type()})
    list_type = vane.list_type(vane.file_type())
    array_type = vane.array_type(vane.file_type(), 1)
    map_type = vane.map_type(vane.file_type(), vane.sqltypes.INTEGER)
    cases = [
        (
            value,
            vane.union_type({"text": vane.sqltypes.VARCHAR, "file": vane.file_type()}),
            "file",
            value,
        ),
        (
            [value],
            vane.union_type({"text": vane.sqltypes.VARCHAR, "array": array_type}),
            "array",
            (value,),
        ),
        (
            [value, None],
            vane.union_type({"text": vane.sqltypes.VARCHAR, "files": list_type}),
            "files",
            [value, None],
        ),
        (
            {"item": value},
            vane.union_type({"text": vane.sqltypes.VARCHAR, "record": plain_struct_type}),
            "record",
            {"item": value},
        ),
        (
            {value: 7},
            vane.union_type({"text": vane.sqltypes.VARCHAR, "mapping": map_type}),
            "mapping",
            {value: 7},
        ),
    ]

    for python_value, dtype, expected_tag, expected_value in cases:
        actual_type, actual_tag, actual_value = duckdb_cursor.execute(
            "SELECT typeof($1), union_tag($1), $1", [vane.Value(python_value, dtype)]
        ).fetchone()
        assert actual_type == str(dtype)
        assert actual_tag == expected_tag
        assert actual_value == expected_value


def test_file_map_keys_materialize_as_hashable_values(duckdb_cursor):
    key = vane.File("memory://map-key", "text/plain", 0, 3, "sha256:key")
    value = vane.Value({key: 42}, vane.map_type(vane.file_type(), vane.sqltypes.BIGINT))

    fetched = duckdb_cursor.execute("SELECT $1", [value]).fetchone()[0]
    columnar = duckdb_cursor.execute("SELECT $1", [value]).df().iloc[0, 0]
    assert fetched == {key: 42}
    assert columnar == {key: 42}


@pytest.mark.parametrize(
    "fallback",
    [
        {
            "url": "memory://dict",
            "content_type": None,
            "position": None,
            "size": None,
            "checksum": None,
        },
        ("memory://tuple", None, None, None, None),
        b"memory://blob",
    ],
)
def test_explicit_file_conversion_rejects_structural_fallbacks(fallback):
    with pytest.raises(vane.InvalidInputException, match="Only vane.File or NULL"):
        vane.ConstantExpression(vane.Value(fallback, vane.file_type()))
    with pytest.raises(vane.InvalidInputException, match="Only vane.File or NULL"):
        vane.ConstantExpression(vane.Value([fallback], vane.list_type(vane.file_type())))


def test_file_does_not_implicitly_convert_to_plain_struct():
    plain_struct = vane.struct_type(
        {
            "url": vane.sqltypes.VARCHAR,
            "content_type": vane.sqltypes.VARCHAR,
            "position": vane.sqltypes.BIGINT,
            "size": vane.sqltypes.BIGINT,
            "checksum": vane.sqltypes.VARCHAR,
        }
    )
    with pytest.raises(vane.InvalidInputException, match="only be converted to FILE"):
        vane.ConstantExpression(vane.Value(vane.File("memory://value"), plain_struct))


def test_plain_struct_does_not_select_file_union_member():
    dtype = vane.union_type({"file": vane.file_type(), "text": vane.sqltypes.VARCHAR})
    value = {
        "url": "memory://struct",
        "content_type": None,
        "position": None,
        "size": None,
        "checksum": None,
    }
    with pytest.raises(vane.InvalidInputException, match="no compatible composite member"):
        vane.ConstantExpression(vane.Value(value, dtype))


def test_invalid_stored_file_is_rejected_during_materialization(duckdb_cursor):
    duckdb_cursor.execute("CREATE TABLE invalid_file(value FILE)")
    duckdb_cursor.execute(
        """
        INSERT INTO invalid_file
        SELECT ROW(NULL::VARCHAR, NULL::VARCHAR, NULL::BIGINT, NULL::BIGINT, NULL::VARCHAR)
        """
    )
    with pytest.raises(vane.InvalidInputException, match="url cannot be NULL"):
        duckdb_cursor.sql("SELECT value FROM invalid_file").fetchone()


def test_file_expression_api_is_pure_and_defers_validation(duckdb_cursor):
    expression = vane.file(
        "s3://bucket/not-accessed",
        content_type="application/octet-stream",
        position=5,
        size=7,
        checksum="sha256:expression",
    )
    assert duckdb_cursor.values(expression).fetchone() == (
        vane.File("s3://bucket/not-accessed", "application/octet-stream", 5, 7, "sha256:expression"),
    )
    assert duckdb_cursor.values(
        expression.url,
        expression.content_type,
        expression.position,
        expression.size,
        expression.checksum,
    ).fetchone() == ("s3://bucket/not-accessed", "application/octet-stream", 5, 7, "sha256:expression")

    relation = duckdb_cursor.values(vane.ConstantExpression("memory://column").alias("source"))
    assert relation.select(vane.col("source").as_file()).fetchone() == (vane.File("memory://column"),)

    invalid = vane.file("memory://invalid-range", position=0)
    assert isinstance(invalid, vane.Expression)
    with pytest.raises(vane.InvalidInputException, match="position and size"):
        duckdb_cursor.values(invalid).fetchone()
