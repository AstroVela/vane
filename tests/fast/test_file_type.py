# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

import vane


FILE = "file('s3://bucket/missing.bin', 'application/octet-stream', 10, 20, 'sha256:abcdef')"
FILE_WITH_NULL_METADATA = "file('s3://bucket/missing.bin', NULL, NULL, NULL, NULL)"


@pytest.fixture
def connection():
    with vane.connect() as conn:
        yield conn


def test_file_constructor_is_pure_and_exposes_canonical_fields(connection):
    row = connection.execute(
        f"""
        SELECT typeof(value), value.url, value.content_type, value.position, value.size, value.checksum
        FROM (SELECT {FILE} AS value)
        """
    ).fetchone()

    assert row == (
        "FILE",
        "s3://bucket/missing.bin",
        "application/octet-stream",
        10,
        20,
        "sha256:abcdef",
    )


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("file(NULL, NULL, NULL, NULL, NULL)", "url cannot be NULL"),
        ("file('x', NULL, 0, NULL, NULL)", "position and size"),
        ("file('x', NULL, NULL, 0, NULL)", "position and size"),
        ("file('x', NULL, -1, 0, NULL)", "non-negative"),
        ("file('x', NULL, 0, -1, NULL)", "non-negative"),
        ("file('x', NULL, 9223372036854775807, 1, NULL)", "exceeds BIGINT"),
        ("file('x', NULL, NULL, NULL, 'sha256')", "<algorithm>:<digest>"),
        ("file('x', NULL, NULL, NULL, ':abcdef')", "<algorithm>:<digest>"),
        ("file('x', NULL, NULL, NULL, 'sha256:')", "<algorithm>:<digest>"),
        ("file('x', NULL, NULL, NULL, 'sha256:abc:def')", "<algorithm>:<digest>"),
    ],
)
def test_file_constructor_rejects_invalid_values(connection, expression, message):
    with pytest.raises(vane.InvalidInputException, match=message):
        connection.execute(f"SELECT {expression}").fetchone()


def test_file_constructor_accepts_zero_length_ranges(connection):
    assert connection.execute("SELECT file('x', NULL, 0, 0, NULL).size").fetchone() == (0,)


def test_file_constructor_supports_vectorized_inputs(connection):
    rows = connection.execute(
        """
        SELECT value.url, value.position, value.size
        FROM (
            SELECT file('object-' || i, NULL, i, i + 1, NULL) AS value
            FROM range(3) AS input(i)
        )
        ORDER BY value.position
        """
    ).fetchall()

    assert rows == [("object-0", 0, 1), ("object-1", 1, 2), ("object-2", 2, 3)]


def test_file_alias_survives_unrelated_nested_null_resolution(connection):
    row = connection.execute(
        f"""
        SELECT typeof(value.file), value.file.url, value.missing
        FROM (SELECT struct_pack(file := {FILE}, missing := NULL) AS value)
        """
    ).fetchone()

    assert row == ("FILE", "s3://bucket/missing.bin", None)


def test_file_comparison_uses_fieldwise_sql_three_value_logic(connection):
    row = connection.execute(
        f"""
        SELECT
            {FILE} = {FILE},
            {FILE} != {FILE},
            {FILE_WITH_NULL_METADATA} = {FILE_WITH_NULL_METADATA},
            {FILE_WITH_NULL_METADATA} != {FILE_WITH_NULL_METADATA},
            {FILE_WITH_NULL_METADATA} = file('other', NULL, NULL, NULL, NULL),
            {FILE_WITH_NULL_METADATA} != file('other', NULL, NULL, NULL, NULL),
            CAST(NULL AS FILE) = {FILE},
            CAST(NULL AS FILE) != {FILE}
        """
    ).fetchone()

    assert row == (True, False, None, None, False, True, None, None)


@pytest.mark.parametrize(
    "other",
    [
        "file('other', 'application/octet-stream', 10, 20, 'sha256:abcdef')",
        "file('s3://bucket/missing.bin', 'text/plain', 10, 20, 'sha256:abcdef')",
        "file('s3://bucket/missing.bin', 'application/octet-stream', 11, 20, 'sha256:abcdef')",
        "file('s3://bucket/missing.bin', 'application/octet-stream', 10, 21, 'sha256:abcdef')",
        "file('s3://bucket/missing.bin', 'application/octet-stream', 10, 20, 'sha256:fedcba')",
    ],
)
def test_file_comparison_includes_every_field(connection, other):
    assert connection.execute(f"SELECT {FILE} = {other}, {FILE} != {other}").fetchone() == (False, True)


def test_file_comparison_keeps_three_value_logic_in_joins(connection):
    count = connection.execute(
        f"""
        SELECT count(*)
        FROM (VALUES ({FILE}), ({FILE_WITH_NULL_METADATA})) AS left_side(value)
        JOIN (VALUES ({FILE}), ({FILE_WITH_NULL_METADATA})) AS right_side(value)
          ON left_side.value = right_side.value
        """
    ).fetchone()[0]

    assert count == 1


def test_file_comparison_propagates_stored_root_nulls(connection):
    connection.execute("CREATE TEMP TABLE file_values(value FILE)")
    connection.execute(f"INSERT INTO file_values VALUES ({FILE}), (NULL)")

    rows = connection.execute(
        f"""
        SELECT value = {FILE}, value != {FILE}
        FROM file_values
        ORDER BY value IS NULL
        """
    ).fetchall()

    assert rows == [(True, False), (None, None)]


def test_file_comparison_cannot_be_shadowed(connection):
    connection.execute("CREATE MACRO __vane_file_equal(left_value, right_value) AS TRUE")

    assert connection.execute(f"SELECT {FILE_WITH_NULL_METADATA} = {FILE_WITH_NULL_METADATA}").fetchone() == (None,)


@pytest.mark.parametrize(
    "predicate",
    [
        f"{FILE} = ROW('x', NULL, NULL, NULL, NULL)",
        f"{FILE} != 's3://bucket/missing.bin'",
        f"{FILE} < {FILE}",
        f"{FILE} IS DISTINCT FROM {FILE}",
        f"{FILE} IN ({FILE})",
        f"{FILE} IN (SELECT {FILE})",
        f"{FILE} BETWEEN {FILE} AND {FILE}",
        f"[{FILE}] = [{FILE}]",
        f"struct_pack(value := {FILE}) = struct_pack(value := {FILE})",
        f"[{FILE}] IN (SELECT [{FILE}])",
    ],
)
def test_file_rejects_unsupported_comparisons(connection, predicate):
    with pytest.raises(vane.BinderException):
        connection.execute(f"SELECT {predicate}").fetchone()


@pytest.mark.parametrize(
    "expression",
    [
        "CAST(ROW('x', NULL, NULL, NULL, NULL) AS FILE)",
        f"CAST({FILE} AS STRUCT(url VARCHAR, content_type VARCHAR, position BIGINT, size BIGINT, checksum VARCHAR))",
        f"CAST({FILE} AS VARCHAR)",
    ],
)
def test_file_rejects_casts_that_bypass_the_constructor(connection, expression):
    with pytest.raises(vane.BinderException):
        connection.execute(f"SELECT {expression}").fetchone()


@pytest.mark.parametrize(
    "expression",
    [
        """
        remap_struct(
            {
                'url': NULL::VARCHAR,
                'content_type': NULL::VARCHAR,
                'position': 0::BIGINT,
                'size': NULL::BIGINT,
                'checksum': NULL::VARCHAR
            },
            NULL::FILE,
            NULL,
            NULL
        )
        """,
        f"""
        remap_struct(
            {FILE},
            NULL::STRUCT(
                url VARCHAR,
                content_type VARCHAR,
                position BIGINT,
                size BIGINT,
                checksum VARCHAR
            ),
            NULL,
            NULL
        )
        """,
        f"""
        remap_struct(
            {{'unused': 1}},
            NULL::STRUCT(
                value STRUCT(
                    url VARCHAR,
                    content_type VARCHAR,
                    position BIGINT,
                    size BIGINT,
                    checksum VARCHAR
                )
            ),
            NULL,
            {{'value': {FILE}}}
        )
        """,
    ],
)
def test_file_rejects_struct_remapping_bypasses(connection, expression):
    with pytest.raises(vane.BinderException, match="remap_struct does not support FILE"):
        connection.execute(f"SELECT {expression}").fetchone()


@pytest.mark.parametrize("column_type", ["FILE", "FILE[]", "STRUCT(value FILE)"])
def test_file_rejects_untyped_parameters_that_bypass_the_constructor(connection, column_type):
    with pytest.raises(vane.BinderException, match=r"construct it with file\(\.\.\.\)"):
        connection.execute(f"PREPARE file_parameter AS SELECT CAST($1 AS {column_type})")


def test_file_output_is_rejected_by_expression_udfs_until_file_materialization_is_supported(connection):
    @vane.func(return_dtype="FILE")
    def make_file(value):
        return value

    relation = connection.sql("SELECT 1 AS value")
    with pytest.raises(vane.InvalidInputException, match="FILE inputs and outputs are not supported"):
        relation.select(make_file(vane.col("value"))).fetchall()


def test_file_input_is_rejected_by_expression_udfs_until_file_materialization_is_supported(connection):
    @vane.func(return_dtype="VARCHAR")
    def consume_file(value):
        return str(value)

    relation = connection.sql(f"SELECT {FILE} AS value")
    with pytest.raises(vane.BinderException, match="FILE inputs and outputs are not supported"):
        relation.select(consume_file(vane.col("value"))).fetchall()


def test_file_output_is_rejected_by_attached_python_udfs(connection):
    with pytest.raises(vane.InvalidInputException, match="FILE inputs and outputs are not supported"):
        vane.attach_function(
            lambda: None,
            alias="file_output_udf",
            connection=connection,
            parameters=[],
            return_dtype="FILE",
        )


def test_file_input_is_rejected_by_attached_python_udfs(connection):
    vane.attach_function(
        lambda value: str(value),
        alias="file_input_udf",
        connection=connection,
        parameters=["FILE"],
        return_dtype="VARCHAR",
    )

    with pytest.raises(vane.BinderException, match="FILE inputs and outputs are not supported"):
        connection.execute(f"SELECT file_input_udf({FILE})").fetchone()


def test_file_only_compares_with_the_same_logical_type(connection):
    connection.execute(
        """
        CREATE TYPE other_file AS STRUCT(
            url VARCHAR,
            content_type VARCHAR,
            position BIGINT,
            size BIGINT,
            checksum VARCHAR
        )
        """
    )
    with pytest.raises(vane.BinderException):
        connection.execute(f"SELECT {FILE} = CAST(ROW('x', NULL, NULL, NULL, NULL) AS other_file)").fetchone()


def test_typed_null_file_is_supported(connection):
    assert connection.execute("SELECT typeof(CAST(NULL AS FILE)), CAST(NULL AS FILE) IS NULL").fetchone() == (
        "FILE",
        True,
    )


def test_file_type_round_trips_through_storage(tmp_path: Path):
    database = tmp_path / "files.db"
    with vane.connect(str(database)) as connection:
        connection.execute("CREATE TABLE files(value FILE)")
        connection.execute(f"INSERT INTO files VALUES ({FILE}), (NULL)")

    with vane.connect(str(database)) as connection:
        rows = connection.execute("SELECT typeof(value), value.url FROM files ORDER BY value IS NULL").fetchall()

    assert rows == [("FILE", "s3://bucket/missing.bin"), ("FILE", None)]
