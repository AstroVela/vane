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


def test_file_equality_remains_a_hash_join_condition(connection):
    plan = connection.execute(
        f"""
        EXPLAIN
        SELECT *
        FROM (VALUES ({FILE}), ({FILE_WITH_NULL_METADATA})) AS left_side(value)
        JOIN (VALUES ({FILE}), ({FILE_WITH_NULL_METADATA})) AS right_side(value)
          ON left_side.value = right_side.value
        """
    ).fetchone()[1]

    assert "HASH_JOIN" in plan
    assert "BLOCKWISE_NL_JOIN" not in plan


def test_file_comparison_supports_scalar_subqueries(connection):
    assert connection.execute(f"SELECT (SELECT {FILE}) = {FILE}, (SELECT {FILE}) != {FILE}").fetchone() == (
        True,
        False,
    )


def test_file_comparison_does_not_duplicate_volatile_operands(connection):
    connection.execute("CREATE SEQUENCE file_comparison_sequence START 1")

    assert connection.execute(
        """
        SELECT
            file(nextval('file_comparison_sequence')::VARCHAR, NULL, NULL, NULL, NULL)
            = file(nextval('file_comparison_sequence')::VARCHAR, NULL, NULL, NULL, NULL)
        """
    ).fetchone() == (False,)
    assert connection.execute("SELECT currval('file_comparison_sequence')").fetchone() == (2,)


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
    "function_name",
    [
        "list_contains",
        "array_contains",
        "list_has",
        "array_has",
        "contains",
        "list_position",
        "list_indexof",
        "array_position",
        "array_indexof",
    ],
)
def test_file_rejects_list_search_comparison_bypasses(connection, function_name):
    with pytest.raises(vane.BinderException, match="Collection search functions do not support FILE"):
        connection.execute(f"SELECT {function_name}([{FILE}], {FILE})").fetchone()


def test_file_rejects_nested_list_search_comparison_bypasses(connection):
    value = f"struct_pack(value := {FILE})"
    with pytest.raises(vane.BinderException, match="Collection search functions do not support FILE"):
        connection.execute(f"SELECT list_contains([{value}], {value})").fetchone()


@pytest.mark.parametrize(
    "expression",
    [
        f"map_contains(map([{FILE_WITH_NULL_METADATA}], [1]), {FILE_WITH_NULL_METADATA})",
        f"map_extract(map([{FILE_WITH_NULL_METADATA}], [1]), {FILE_WITH_NULL_METADATA})",
        f"element_at(map([{FILE_WITH_NULL_METADATA}], [1]), {FILE_WITH_NULL_METADATA})",
        f"map_extract_value(map([{FILE_WITH_NULL_METADATA}], [1]), {FILE_WITH_NULL_METADATA})",
        f"map([{FILE_WITH_NULL_METADATA}], [1])[{FILE_WITH_NULL_METADATA}]",
        f"list_has_any([{FILE_WITH_NULL_METADATA}], [{FILE_WITH_NULL_METADATA}])",
        f"array_has_any([{FILE_WITH_NULL_METADATA}], [{FILE_WITH_NULL_METADATA}])",
        f"[{FILE_WITH_NULL_METADATA}] && [{FILE_WITH_NULL_METADATA}]",
        f"list_has_all([{FILE_WITH_NULL_METADATA}], [{FILE_WITH_NULL_METADATA}])",
        f"array_has_all([{FILE_WITH_NULL_METADATA}], [{FILE_WITH_NULL_METADATA}])",
        f"[{FILE_WITH_NULL_METADATA}] @> [{FILE_WITH_NULL_METADATA}]",
        f"[{FILE_WITH_NULL_METADATA}] <@ [{FILE_WITH_NULL_METADATA}]",
        f"list_intersect([{FILE_WITH_NULL_METADATA}], [{FILE_WITH_NULL_METADATA}])",
        f"array_intersect([{FILE_WITH_NULL_METADATA}], [{FILE_WITH_NULL_METADATA}])",
        f"struct_contains(row({FILE_WITH_NULL_METADATA}), {FILE_WITH_NULL_METADATA})",
        f"struct_has(row({FILE_WITH_NULL_METADATA}), {FILE_WITH_NULL_METADATA})",
        f"struct_position(row({FILE_WITH_NULL_METADATA}), {FILE_WITH_NULL_METADATA})",
        f"struct_indexof(row({FILE_WITH_NULL_METADATA}), {FILE_WITH_NULL_METADATA})",
    ],
)
def test_file_rejects_collection_search_comparison_bypasses(connection, expression):
    with pytest.raises(vane.BinderException, match="Collection search functions do not support FILE"):
        connection.execute(f"SELECT {expression}").fetchone()


def test_file_remains_usable_as_a_map_value(connection):
    row = connection.execute(
        f"""
        WITH input(value) AS (VALUES (map(['key'], [{FILE}])))
        SELECT
            map_contains(value, 'key'),
            typeof(map_extract_value(value, 'key')),
            (map_extract_value(value, 'key')).url,
            typeof(map_extract(value, 'key')[1]),
            (map_extract(value, 'key')[1]).url,
            typeof(value['key']),
            (value['key']).url
        FROM input
        """
    ).fetchone()

    assert row == (
        True,
        "FILE",
        "s3://bucket/missing.bin",
        "FILE",
        "s3://bucket/missing.bin",
        "FILE",
        "s3://bucket/missing.bin",
    )


@pytest.mark.parametrize(
    "query",
    [
        f"SELECT value FROM (VALUES ({FILE})) AS input(value) ORDER BY value",
        f"SELECT value FROM (VALUES ({FILE})) AS input(value) ORDER BY struct_pack(value := value)",
        f"SELECT list(value ORDER BY value) FROM (VALUES ({FILE})) AS input(value)",
        f"SELECT percentile_disc(0.5) WITHIN GROUP (ORDER BY value) FROM (VALUES ({FILE})) AS input(value)",
        f"SELECT row_number() OVER (ORDER BY value) FROM (VALUES ({FILE})) AS input(value)",
        f"SELECT first_value(value ORDER BY value) OVER () FROM (VALUES ({FILE})) AS input(value)",
    ],
)
def test_file_rejects_ordering_bypasses(connection, query):
    with pytest.raises(
        vane.BinderException,
        match=r"(?:ORDER BY does not|ordering functions do not) support FILE values",
    ):
        connection.execute(query).fetchall()


@pytest.mark.parametrize(
    "expression",
    [
        f"list_sort([{FILE}])",
        f"array_sort([{FILE}])",
        f"list_reverse_sort([{FILE}])",
        f"array_reverse_sort([{FILE}])",
        f"list_grade_up([{FILE}])",
        f"list_sort([struct_pack(value := {FILE})])",
        f"least({FILE}, {FILE})",
        f"greatest({FILE}, {FILE})",
        f"create_sort_key({FILE}, 'ASC NULLS LAST')",
    ],
)
def test_file_rejects_generic_ordering_functions(connection, expression):
    with pytest.raises(vane.BinderException, match="support FILE values"):
        connection.execute(f"SELECT {expression}").fetchone()


@pytest.mark.parametrize(
    "aggregate",
    [
        "min(value)",
        "max(value)",
        "min(value, 1)",
        "max(value, 1)",
        "min(struct_pack(value := value))",
        "max(struct_pack(value := value))",
    ],
)
def test_file_rejects_min_max_bypasses(connection, aggregate):
    with pytest.raises(vane.BinderException, match="does not support FILE values"):
        connection.execute(f"SELECT {aggregate} FROM (VALUES ({FILE})) AS input(value)").fetchone()


@pytest.mark.parametrize(
    "aggregate",
    [
        "arg_min(1, value)",
        "arg_max(1, value)",
        "arg_min(1, value, 1)",
        "arg_max(1, value, 1)",
        "median(value)",
        "quantile_disc(value, 0.5)",
        "histogram(value)",
        "histogram(value, [value])",
        "histogram_exact(value, [value])",
        "histogram(struct_pack(value := value))",
    ],
)
def test_file_rejects_other_ordering_aggregate_bypasses(connection, aggregate):
    with pytest.raises(vane.BinderException, match="support FILE values"):
        connection.execute(f"SELECT {aggregate} FROM (VALUES ({FILE})) AS input(value)").fetchone()


def test_file_remains_usable_as_an_arg_min_result(connection):
    row = connection.execute(
        f"""
        SELECT typeof(arg_min(value, key)), (arg_min(value, key)).url
        FROM (
            VALUES
                (2, {FILE}),
                (1, file('other', NULL, NULL, NULL, NULL))
        ) AS input(key, value)
        """
    ).fetchone()

    assert row == ("FILE", "other")


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


def test_file_type_round_trips_through_arrow(connection):
    pytest.importorskip("pyarrow")

    arrow_table = connection.sql(
        f"""
        SELECT 0 AS row_id, {FILE} AS value, struct_pack(value := {FILE}) AS nested
        UNION ALL
        SELECT 1 AS row_id, NULL::FILE AS value, struct_pack(value := NULL::FILE) AS nested
        """
    ).to_arrow_table()
    extension_name = b"ARROW:extension:name"
    assert arrow_table.schema.field("value").metadata[extension_name] == b"vane.file"
    assert arrow_table.schema.field("nested").type.field("value").metadata[extension_name] == b"vane.file"

    rows = (
        connection.from_arrow(arrow_table)
        .project("row_id, typeof(value), typeof(nested.value), value.url, nested.value.url")
        .order("row_id")
        .fetchall()
    )
    assert rows == [
        (0, "FILE", "FILE", "s3://bucket/missing.bin", "s3://bucket/missing.bin"),
        (1, "FILE", "FILE", None, None),
    ]


def test_file_type_rejects_invalid_arrow_values(connection):
    pa = pytest.importorskip("pyarrow")

    file_storage_type = pa.struct(
        [
            pa.field("url", pa.string()),
            pa.field("content_type", pa.string()),
            pa.field("position", pa.int64()),
            pa.field("size", pa.int64()),
            pa.field("checksum", pa.string()),
        ]
    )
    file_field = pa.field(
        "value",
        file_storage_type,
        metadata={
            b"ARROW:extension:name": b"vane.file",
            b"ARROW:extension:metadata": b"",
        },
    )
    invalid_file = pa.array(
        [
            {
                "url": None,
                "content_type": None,
                "position": None,
                "size": None,
                "checksum": None,
            }
        ],
        type=file_storage_type,
    )
    arrow_table = pa.Table.from_arrays([invalid_file], schema=pa.schema([file_field]))

    with pytest.raises(vane.InvalidInputException, match="Arrow FILE url cannot be NULL"):
        connection.from_arrow(arrow_table).fetchall()


def test_file_type_round_trips_through_local_exchange(connection):
    pytest.importorskip("pyarrow")

    relation = connection.sql(
        """
        SELECT
            i,
            file('object-' || i, NULL, NULL, NULL, NULL) AS value,
            struct_pack(value := file('nested-' || i, NULL, NULL, NULL, NULL)) AS nested
        FROM range(2) AS input(i)
        """
    )
    rows = (
        relation.local_exchange(1)
        .project("i, typeof(value), typeof(nested.value), value.url, nested.value.url")
        .order("i")
        .fetchall()
    )

    assert rows == [
        (0, "FILE", "FILE", "object-0", "nested-0"),
        (1, "FILE", "FILE", "object-1", "nested-1"),
    ]


def test_file_type_round_trips_through_storage(tmp_path: Path):
    database = tmp_path / "files.db"
    with vane.connect(str(database)) as connection:
        connection.execute("CREATE TABLE files(value FILE)")
        connection.execute(f"INSERT INTO files VALUES ({FILE}), (NULL)")

    with vane.connect(str(database)) as connection:
        rows = connection.execute("SELECT typeof(value), value.url FROM files ORDER BY value IS NULL").fetchall()

    assert rows == [("FILE", "s3://bucket/missing.bin"), ("FILE", None)]
