# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pickle

import pandas as pd
import pytest

import vane

FILE_FIELDS = ("url", "content_type", "position", "size", "checksum")


def test_file_public_class_is_the_native_value_type():
    assert vane.File is vane._native.File


def test_file_value_defaults_and_complete_metadata():
    minimal = vane.File("s3://bucket/missing.bin")
    assert tuple(getattr(minimal, field) for field in FILE_FIELDS) == (
        "s3://bucket/missing.bin",
        None,
        None,
        None,
        None,
    )
    assert str(minimal) == "s3://bucket/missing.bin"
    assert repr(minimal) == (
        "File(url='s3://bucket/missing.bin', content_type=None, position=None, size=None, checksum=None)"
    )

    complete = vane.File(
        "https://example.test/image.png",
        content_type="image/png",
        position=10,
        size=20,
        checksum="sha256:abcdef",
    )
    assert tuple(getattr(complete, field) for field in FILE_FIELDS) == (
        "https://example.test/image.png",
        "image/png",
        10,
        20,
        "sha256:abcdef",
    )
    assert {name for name in dir(complete) if not name.startswith("_")} == set(FILE_FIELDS)


def test_file_value_is_immutable_and_final():
    value = vane.File("memory://value")

    for field in FILE_FIELDS:
        with pytest.raises(AttributeError):
            setattr(value, field, None)
    with pytest.raises(AttributeError):
        value.extra = "credential"
    with pytest.raises(TypeError):

        class DerivedFile(vane.File):
            pass


@pytest.mark.parametrize("url", [None, 42, b"path"])
def test_file_value_requires_a_string_url(url):
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
def test_file_value_requires_exact_optional_field_types(kwargs, field):
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
def test_file_value_rejects_invalid_byte_ranges(kwargs):
    with pytest.raises(ValueError):
        vane.File("memory://value", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": 2**63, "size": 0},
        {"position": 0, "size": 2**63},
        {"position": -(2**63) - 1, "size": 0},
    ],
)
def test_file_value_rejects_integer_fields_outside_bigint(kwargs):
    with pytest.raises(OverflowError, match="signed 64-bit"):
        vane.File("memory://value", **kwargs)


@pytest.mark.parametrize("checksum", ["", "sha256", ":digest", "sha256:", "sha256:a:b"])
def test_file_value_rejects_invalid_checksums(checksum):
    with pytest.raises(ValueError, match=r"<algorithm>:<digest>"):
        vane.File("memory://value", checksum=checksum)


def test_file_value_has_value_equality_hashing_and_pickle_support():
    value = vane.File("s3://bucket/object", "application/octet-stream", 4, 8, "sha256:abc")
    equal = vane.File("s3://bucket/object", "application/octet-stream", 4, 8, "sha256:abc")
    different = vane.File("s3://bucket/other", "application/octet-stream", 4, 8, "sha256:abc")

    assert value == equal
    assert hash(value) == hash(equal)
    assert value != different
    assert value != "s3://bucket/object"
    assert len({value, equal, different}) == 2
    assert pickle.loads(pickle.dumps(value)) == value


def test_file_data_type_contract():
    dtype = vane.file_type()

    assert str(dtype) == "FILE"
    assert dtype.id == "struct"
    assert dtype.is_file()
    assert dtype == vane.sqltype("FILE")
    assert not vane.struct_type(
        {
            "url": vane.sqltypes.VARCHAR,
            "content_type": vane.sqltypes.VARCHAR,
            "position": vane.sqltypes.BIGINT,
            "size": vane.sqltypes.BIGINT,
            "checksum": vane.sqltypes.VARCHAR,
        }
    ).is_file()
    assert dtype.children == [
        ("url", vane.sqltypes.VARCHAR),
        ("content_type", vane.sqltypes.VARCHAR),
        ("position", vane.sqltypes.BIGINT),
        ("size", vane.sqltypes.BIGINT),
        ("checksum", vane.sqltypes.VARCHAR),
    ]
    assert dtype.url == vane.sqltypes.VARCHAR
    assert pickle.loads(pickle.dumps(dtype)).is_file()


def test_file_query_results_materialize_as_file_values(duckdb_cursor):
    duckdb_cursor.execute("CREATE TABLE file_results(value FILE)")
    duckdb_cursor.execute(
        """
        INSERT INTO file_results VALUES
            (file('memory://a', 'text/plain', 0, 4, 'sha256:a')),
            (file('memory://b', NULL, NULL, NULL, NULL)),
            (NULL)
        """
    )
    rows = duckdb_cursor.sql("SELECT value FROM file_results ORDER BY rowid").fetchall()

    assert rows == [
        (vane.File("memory://a", "text/plain", 0, 4, "sha256:a"),),
        (vane.File("memory://b"),),
        (None,),
    ]


def test_file_columnar_results_materialize_as_file_values(duckdb_cursor):
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


def test_nested_file_query_results_materialize_recursively(duckdb_cursor):
    file_value = vane.File("memory://nested", "application/octet-stream", 1, 2, "sha256:nested")
    row = duckdb_cursor.sql(
        """
        SELECT
            [file('memory://nested', 'application/octet-stream', 1, 2, 'sha256:nested'), NULL::FILE],
            struct_pack(item := file('memory://nested', 'application/octet-stream', 1, 2, 'sha256:nested'))
        """
    ).fetchone()

    assert row == ([file_value, None], {"item": file_value})


def test_plain_struct_with_file_fields_remains_a_dictionary(duckdb_cursor):
    query = """
        SELECT struct_pack(
            url := 'memory://struct',
            content_type := NULL::VARCHAR,
            "position" := NULL::BIGINT,
            size := NULL::BIGINT,
            checksum := NULL::VARCHAR
        ) AS value
    """
    value = duckdb_cursor.sql(query).fetchone()[0]
    columnar_value = duckdb_cursor.sql(query).df()["value"].iloc[0]

    expected = {
        "url": "memory://struct",
        "content_type": None,
        "position": None,
        "size": None,
        "checksum": None,
    }
    assert value == expected
    assert columnar_value == expected
    assert not isinstance(value, vane.File)
    assert not isinstance(columnar_value, vane.File)


def test_file_values_cross_explicit_python_boundaries(duckdb_cursor):
    value = vane.File("memory://parameter", "text/plain", 0, 3, "sha256:parameter")

    assert duckdb_cursor.values(vane.ConstantExpression(value)).types[0].is_file()
    assert duckdb_cursor.values(vane.ConstantExpression(value)).fetchone() == (value,)
    assert duckdb_cursor.values(vane.ConstantExpression(vane.Value(value, vane.file_type()))).fetchone() == (value,)
    typed_null = duckdb_cursor.values(vane.ConstantExpression(vane.Value(None, vane.file_type())))
    assert typed_null.types[0].is_file()
    assert typed_null.fetchone() == (None,)

    duckdb_cursor.execute("CREATE TABLE file_parameters(value FILE)")
    duckdb_cursor.execute("INSERT INTO file_parameters VALUES (?)", [value])
    assert duckdb_cursor.sql("SELECT value FROM file_parameters").fetchone() == (value,)

    nested = duckdb_cursor.execute("SELECT ?", [[value, None]]).fetchone()[0]
    assert nested == [value, None]


@pytest.mark.parametrize(
    ("value", "dtype", "expected_value"),
    [
        pytest.param(None, vane.list_type(vane.file_type()), None, id="null-list"),
        pytest.param([], vane.list_type(vane.file_type()), [], id="empty-list"),
        pytest.param([None], vane.list_type(vane.file_type()), [None], id="null-only-list"),
        pytest.param([None], vane.array_type(vane.file_type(), 1), (None,), id="null-only-array"),
        pytest.param(
            {"file": None},
            vane.struct_type({"file": vane.file_type()}),
            {"file": None},
            id="null-only-struct",
        ),
        pytest.param({}, vane.map_type(vane.sqltypes.VARCHAR, vane.file_type()), {}, id="empty-map"),
        pytest.param(
            {"key": ["one"], "value": [None]},
            vane.map_type(vane.sqltypes.VARCHAR, vane.file_type()),
            {"one": None},
            id="null-only-map-values",
        ),
        pytest.param(
            [[]],
            vane.list_type(vane.list_type(vane.file_type())),
            [[]],
            id="nested-empty-list",
        ),
    ],
)
def test_explicit_nested_file_types_survive_null_only_values(duckdb_cursor, value, dtype, expected_value):
    actual_type, actual_value = duckdb_cursor.execute("SELECT typeof($1), $1", [vane.Value(value, dtype)]).fetchone()

    assert actual_type == str(dtype)
    assert actual_value == expected_value


def test_pandas_file_columns_preserve_file_identity(duckdb_cursor):
    values = [vane.File("memory://first"), float("nan"), vane.File("memory://second", "text/plain"), None]
    relation = duckdb_cursor.from_df(pd.DataFrame({"value": values}))

    assert relation.types[0].is_file()
    assert [row[0] for row in relation.fetchall()] == [values[0], None, values[2], None]


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
def test_python_file_conversion_rejects_fallbacks(fallback):
    with pytest.raises(vane.InvalidInputException, match="Only vane.File or NULL"):
        vane.ConstantExpression(vane.Value(fallback, vane.file_type()))
    with pytest.raises(vane.InvalidInputException, match="Only vane.File or NULL"):
        vane.ConstantExpression(vane.Value([fallback], vane.list_type(vane.file_type())))


def test_python_file_conversion_rejects_plain_struct_targets():
    struct_type = vane.struct_type(
        {
            "url": vane.sqltypes.VARCHAR,
            "content_type": vane.sqltypes.VARCHAR,
            "position": vane.sqltypes.BIGINT,
            "size": vane.sqltypes.BIGINT,
            "checksum": vane.sqltypes.VARCHAR,
        }
    )

    with pytest.raises(vane.InvalidInputException, match="can only be converted to FILE"):
        vane.ConstantExpression(vane.Value(vane.File("memory://value"), struct_type))

    nested_type = vane.struct_type({"value": struct_type})
    nested_value = {"value": vane.Value(vane.File("memory://value"), vane.file_type())}
    with pytest.raises(vane.InvalidInputException, match="can only be converted to FILE"):
        vane.ConstantExpression(vane.Value(nested_value, nested_type))


def test_python_sequence_cannot_infer_file_from_mixed_file_and_struct_values():
    struct_value = {
        "url": "memory://dict",
        "content_type": None,
        "position": None,
        "size": None,
        "checksum": None,
    }

    file_value = vane.File("memory://file")
    mixed_values = [
        [file_value, struct_value],
        [struct_value, file_value],
        [[file_value], [struct_value]],
        [[struct_value], [file_value]],
        [{"item": file_value}, {"item": struct_value}],
        [{"item": struct_value}, {"item": file_value}],
    ]
    for values in mixed_values:
        with pytest.raises(vane.InvalidInputException, match="Cannot mix vane.File and non-FILE"):
            vane.ConstantExpression(values)


def test_python_sequence_allows_nulls_at_file_positions(duckdb_cursor):
    file_value = vane.File("memory://file")
    expression = vane.ConstantExpression([{"item": file_value}, {"item": None}])

    assert duckdb_cursor.values(expression).fetchone() == ([{"item": file_value}, {"item": None}],)


@pytest.mark.parametrize(
    ("values_sql", "message"),
    [
        (
            "NULL::VARCHAR, NULL::VARCHAR, NULL::BIGINT, NULL::BIGINT, NULL::VARCHAR",
            r"File\.url cannot be None",
        ),
        (
            "'memory://range', NULL::VARCHAR, 0::BIGINT, NULL::BIGINT, NULL::VARCHAR",
            "position and File.size",
        ),
        (
            "'memory://negative', NULL::VARCHAR, -1::BIGINT, 0::BIGINT, NULL::VARCHAR",
            "must be non-negative",
        ),
        (
            "'memory://overflow', NULL::VARCHAR, 9223372036854775807::BIGINT, 1::BIGINT, NULL::VARCHAR",
            "exceeds signed 64-bit",
        ),
        (
            "'memory://checksum', NULL::VARCHAR, NULL::BIGINT, NULL::BIGINT, 'sha256'::VARCHAR",
            r"<algorithm>:<digest>",
        ),
    ],
)
def test_file_materialization_rejects_invalid_alias_values(duckdb_cursor, values_sql, message):
    duckdb_cursor.execute("CREATE TABLE invalid_file_results(value FILE)")
    duckdb_cursor.execute(
        f"""
        INSERT INTO invalid_file_results
        SELECT ROW({values_sql})
        """
    )

    with pytest.raises(ValueError, match=message):
        duckdb_cursor.sql("SELECT value FROM invalid_file_results").fetchone()


def test_file_expression_constructor_and_fields_are_pure(duckdb_cursor):
    expression = vane.file(
        "s3://bucket/not-accessed",
        content_type="application/octet-stream",
        position=5,
        size=7,
        checksum="sha256:expression",
    )

    assert isinstance(expression, vane.Expression)
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


def test_expression_as_file_uses_the_expression_as_url(duckdb_cursor):
    relation = duckdb_cursor.values(vane.ConstantExpression("memory://column").alias("source"))
    result = relation.select(vane.col("source").as_file()).fetchone()

    assert result == (vane.File("memory://column"),)


def test_file_expression_validation_is_deferred_until_execution(duckdb_cursor):
    expression = vane.file("memory://invalid-range", position=0)

    assert isinstance(expression, vane.Expression)
    with pytest.raises(vane.InvalidInputException, match="position and size"):
        duckdb_cursor.values(expression).fetchone()
