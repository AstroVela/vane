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


def test_file_supports_all_struct_extraction_forms(connection):
    rows = connection.execute(
        f"""
        SELECT
            struct_extract(value, 'url'),
            struct_extract(value, 1),
            struct_extract_at(value, 1),
            array_extract(value, 'url'),
            array_extract(value, 1),
            value['url'],
            value[1]
        FROM (VALUES ({FILE}), (NULL::FILE)) AS input(value)
        ORDER BY value IS NULL
        """
    ).fetchall()

    url = "s3://bucket/missing.bin"
    assert rows == [(url, url, url, url, url, url, url), (None, None, None, None, None, None, None)]


def test_file_extract_overloads_preserve_untyped_null_binding(connection):
    row = connection.execute(
        """
        SELECT
            struct_extract(NULL, 'url'),
            struct_extract(NULL, 1),
            struct_extract_at(NULL, 1),
            array_extract(NULL, 'url')
        """
    ).fetchone()

    assert row == (None, None, None, None)


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


def test_file_alias_survives_unrelated_recursive_type_replacement(connection):
    row = connection.execute(
        f"""
        SELECT typeof(value.file), typeof(value.number), value.file.url, value.number
        FROM (
            SELECT replace_type(
                struct_pack(file := {FILE}, number := 1::INTEGER),
                NULL::INTEGER,
                NULL::BIGINT
            ) AS value
        )
        """
    ).fetchone()

    assert row == ("FILE", "BIGINT", "s3://bucket/missing.bin", 1)


def test_nested_file_casts_allow_only_null_introduction(connection):
    row = connection.execute(
        """
        SELECT
            typeof(CAST([NULL] AS FILE[])[1]),
            CAST([NULL] AS FILE[])[1] IS NULL,
            typeof(CAST(struct_pack(number := 1) AS STRUCT(number INTEGER, file FILE)).file),
            CAST(struct_pack(number := 1) AS STRUCT(number INTEGER, file FILE)).file IS NULL
        """
    ).fetchone()

    assert row == ("FILE", True, "FILE", True)


def test_file_union_casts_preserve_file_members(connection):
    row = connection.execute(
        f"""
        SELECT union_tag(value), typeof(value.f), (value.f).url, value.n
        FROM (
            SELECT CAST(
                union_value(f := {FILE})
                AS UNION(f FILE, n BIGINT)
            ) AS value
        )
        """
    ).fetchone()

    assert row == ("f", "FILE", "s3://bucket/missing.bin", None)


def test_file_union_all_resolves_a_file_preserving_common_type(connection):
    rows = connection.execute(
        f"""
        SELECT union_tag(value), typeof(value.f), (value.f).url, value.n
        FROM (
            SELECT union_value(f := {FILE}) AS value
            UNION ALL
            SELECT CAST(
                42::BIGINT
                AS UNION(f FILE, n BIGINT)
            ) AS value
        )
        ORDER BY union_tag(value)
        """
    ).fetchall()

    assert rows == [
        ("f", "FILE", "s3://bucket/missing.bin", None),
        ("n", "FILE", None, 42),
    ]


def test_file_implicit_cast_reporting_obeys_file_identity_rules(connection):
    row = connection.execute(
        f"""
        SELECT
            can_cast_implicitly(
                {FILE},
                NULL::STRUCT(
                    url VARCHAR,
                    content_type VARCHAR,
                    position BIGINT,
                    size BIGINT,
                    checksum VARCHAR
                )
            ),
            can_cast_implicitly(
                union_value(f := {FILE}),
                NULL::UNION(f FILE, n BIGINT)
            ),
            can_cast_implicitly(
                union_value(f := {FILE}),
                NULL::UNION(
                    f STRUCT(
                        url VARCHAR,
                        content_type VARCHAR,
                        position BIGINT,
                        size BIGINT,
                        checksum VARCHAR
                    ),
                    n BIGINT
                )
            )
        """
    ).fetchone()

    assert row == (False, True, False)


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


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (
            "file('A', 'application/octet-stream', 0, 1, 'sha256:abcdef')",
            "file('a', 'application/octet-stream', 0, 1, 'sha256:abcdef')",
        ),
        (
            "file('x', 'TEXT/PLAIN', 0, 1, 'sha256:abcdef')",
            "file('x', 'text/plain', 0, 1, 'sha256:abcdef')",
        ),
        (
            "file('x', 'application/octet-stream', 0, 1, 'sha256:ABCDEF')",
            "file('x', 'application/octet-stream', 0, 1, 'sha256:abcdef')",
        ),
    ],
)
def test_file_comparison_uses_binary_string_semantics(connection, left, right):
    connection.execute("SET default_collation = 'nocase'")

    assert connection.execute(
        f"SELECT {left} = {right}, (SELECT {left}) = {right}, {left} != {right}, (SELECT {left}) != {right}"
    ).fetchone() == (False, False, True, True)


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


def test_file_stored_root_null_masks_hidden_children_and_comparisons(tmp_path: Path):
    database = tmp_path / "hidden-file-children.db"
    stored_file = "file('x', 'application/octet-stream', 0, 0, 'sha256:abcdef')"

    with vane.connect(str(database)) as connection:
        connection.execute("CREATE TABLE files(value FILE)")
        connection.execute(
            f"""
            INSERT INTO files
            SELECT constant_or_null({stored_file}, marker)
            FROM (VALUES (NULL::INTEGER)) AS input(marker)
            """
        )

    with vane.connect(str(database)) as connection:
        row = connection.execute(
            f"""
            SELECT
                value IS NULL,
                value.url,
                value.content_type,
                value.position,
                value.size,
                value.checksum,
                struct_extract_at(value, 1),
                value.url IS NULL,
                value = {stored_file},
                {stored_file} = value,
                value != {stored_file},
                {stored_file} != value
            FROM files
            """
        ).fetchone()

    assert row == (True, None, None, None, None, None, None, True, None, None, None, None)

    with vane.connect(str(database)) as connection:
        assert connection.execute("SELECT count(*) FROM files WHERE value.url = 'x'").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM files WHERE value.url IS NULL").fetchone() == (1,)


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
        f"map_contains(value, {FILE_WITH_NULL_METADATA}) FROM (VALUES (NULL::MAP(FILE, INTEGER))) input(value)",
        f"map_extract(value, {FILE_WITH_NULL_METADATA}) FROM (VALUES (NULL::MAP(FILE, INTEGER))) input(value)",
        f"element_at(value, {FILE_WITH_NULL_METADATA}) FROM (VALUES (NULL::MAP(FILE, INTEGER))) input(value)",
        f"map_extract_value(value, {FILE_WITH_NULL_METADATA}) FROM (VALUES (NULL::MAP(FILE, INTEGER))) input(value)",
        f"value[{FILE_WITH_NULL_METADATA}] FROM (VALUES (NULL::MAP(FILE, INTEGER))) input(value)",
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


@pytest.mark.parametrize(
    "expression",
    [
        f"list_distinct([{FILE_WITH_NULL_METADATA}, {FILE_WITH_NULL_METADATA}])",
        f"array_distinct([{FILE_WITH_NULL_METADATA}, {FILE_WITH_NULL_METADATA}])",
        f"list_unique([{FILE_WITH_NULL_METADATA}, {FILE_WITH_NULL_METADATA}])",
        f"array_unique([{FILE_WITH_NULL_METADATA}, {FILE_WITH_NULL_METADATA}])",
        (
            "list_distinct(["
            f"struct_pack(value := {FILE_WITH_NULL_METADATA}), "
            f"struct_pack(value := {FILE_WITH_NULL_METADATA})"
            "])"
        ),
    ],
)
def test_file_rejects_list_hash_comparison_bypasses(connection, expression):
    with pytest.raises(vane.BinderException, match="does not support FILE values"):
        connection.execute(f"SELECT {expression}").fetchone()


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        (
            f"map([{FILE_WITH_NULL_METADATA}, {FILE_WITH_NULL_METADATA}], [1, 2])",
            "map does not support FILE keys",
        ),
        (
            "map(["
            f"struct_pack(value := {FILE_WITH_NULL_METADATA}), "
            f"struct_pack(value := {FILE_WITH_NULL_METADATA})"
            "], [1, 2])",
            "map does not support FILE keys",
        ),
        (
            f"map_from_entries([row({FILE_WITH_NULL_METADATA}, 1), row({FILE_WITH_NULL_METADATA}, 2)])",
            "map_from_entries does not support FILE keys",
        ),
        (
            f"map_from_entries(array_value(row({FILE_WITH_NULL_METADATA}, 1), row({FILE_WITH_NULL_METADATA}, 2)))",
            "map_from_entries does not support FILE keys",
        ),
    ],
)
def test_file_rejects_map_key_hash_comparison_bypasses(connection, expression, message):
    with pytest.raises(vane.BinderException, match=message):
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

    row = connection.execute(
        f"""
        SELECT
            typeof(map_from_entries([row('key', {FILE})])['key']),
            (map_from_entries([row('key', {FILE})])['key']).url
        """
    ).fetchone()
    assert row == ("FILE", "s3://bucket/missing.bin")


@pytest.mark.parametrize(
    "key_type",
    [
        "FILE",
        "STRUCT(value FILE)",
    ],
)
def test_file_rejects_map_concat_keys(connection, key_type):
    with pytest.raises(vane.BinderException, match="MAP_CONCAT does not support FILE map keys"):
        connection.execute(
            f"SELECT map_concat(NULL::MAP({key_type}, INTEGER), NULL::MAP({key_type}, INTEGER))"
        ).fetchone()


@pytest.mark.parametrize(
    ("key", "key_type"),
    [
        (FILE_WITH_NULL_METADATA, "FILE"),
        (f"struct_pack(value := {FILE_WITH_NULL_METADATA})", "STRUCT(value FILE)"),
    ],
)
def test_file_rejects_switch_keys(connection, key, key_type):
    with pytest.raises(vane.BinderException, match="SWITCH does not support FILE map keys"):
        connection.execute(
            f"SELECT switch({key}, value) FROM (VALUES (NULL::MAP({key_type}, INTEGER))) input(value)"
        ).fetchone()


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
        "is_histogram_other_bin(value)",
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
        f"CAST([{FILE}] AS STRUCT(url VARCHAR, content_type VARCHAR, position BIGINT, size BIGINT, checksum VARCHAR)[])",
        f"CAST([{FILE}] AS VARCHAR)",
        "CAST([ROW('x', NULL, NULL, NULL, NULL)] AS FILE[])",
        (
            f"CAST(struct_pack(value := {FILE}) AS STRUCT(value STRUCT("
            "url VARCHAR, content_type VARCHAR, position BIGINT, size BIGINT, checksum VARCHAR)))"
        ),
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


@pytest.mark.parametrize(
    ("expression", "function_name"),
    [
        (f"struct_update({FILE}, url := 'other')", "struct_update"),
        (f"struct_insert({FILE}, extra := 1)", "struct_insert"),
        (f"struct_concat({FILE}, struct_pack(extra := 1))", "struct_concat"),
        (f"struct_concat(struct_pack(extra := 1), {FILE})", "struct_concat"),
    ],
)
def test_file_rejects_struct_alias_stripping_bypasses(connection, expression, function_name):
    with pytest.raises(vane.BinderException, match=rf"{function_name} does not support FILE values"):
        connection.execute(f"SELECT {expression}").fetchone()


def test_file_does_not_shadow_struct_values_null_binding(connection):
    with pytest.raises(vane.BinderException):
        connection.execute(f"SELECT struct_values({FILE})").fetchone()

    assert connection.execute("SELECT struct_values(NULL)").fetchone() == (None,)


@pytest.mark.parametrize(
    "expression",
    [
        "json_transform('{}', '\"FILE\"')",
        "json_transform('{}', '{\"value\":\"FILE\"}')",
    ],
)
def test_file_rejects_dynamic_type_materialization_bypasses(connection, expression):
    with pytest.raises(vane.BinderException, match="does not support FILE"):
        connection.execute(f"SELECT {expression}").fetchone()


def test_file_rejects_json_scan_materialization_bypass(connection, tmp_path: Path):
    source = tmp_path / "file.json"
    source.write_text('{"value":{"url":"hidden"}}')

    with pytest.raises(vane.BinderException, match="read_json does not support FILE column types"):
        connection.execute(f"SELECT * FROM read_json('{source}', columns={{value: 'FILE'}})").fetchall()

    connection.execute("CREATE TABLE json_files(value FILE)")
    with pytest.raises(vane.BinderException, match="COPY FROM JSON does not support FILE column types"):
        connection.execute(f"COPY json_files FROM '{source}' (FORMAT JSON)")


@pytest.mark.parametrize(
    "query",
    [
        f"SELECT DISTINCT value FROM (VALUES ({FILE_WITH_NULL_METADATA})) AS input(value)",
        f"SELECT DISTINCT struct_pack(value := value) FROM (VALUES ({FILE})) AS input(value)",
        f"SELECT DISTINCT ON (value) value FROM (VALUES ({FILE})) AS input(value)",
        f"SELECT count(*) FROM (VALUES ({FILE})) AS input(value) GROUP BY value",
        f"SELECT count(*) FROM (VALUES ({FILE})) AS input(value) GROUP BY struct_pack(value := value)",
        f"SELECT value FROM (VALUES ({FILE})) AS input(value) GROUP BY ALL",
        f"SELECT {FILE_WITH_NULL_METADATA} UNION SELECT {FILE_WITH_NULL_METADATA}",
        f"SELECT {FILE_WITH_NULL_METADATA} INTERSECT SELECT {FILE_WITH_NULL_METADATA}",
        f"SELECT {FILE_WITH_NULL_METADATA} INTERSECT ALL SELECT {FILE_WITH_NULL_METADATA}",
        f"SELECT {FILE_WITH_NULL_METADATA} EXCEPT SELECT {FILE_WITH_NULL_METADATA}",
        f"SELECT {FILE_WITH_NULL_METADATA} EXCEPT ALL SELECT {FILE_WITH_NULL_METADATA}",
        (f"SELECT row_number() OVER (PARTITION BY value) FROM (VALUES ({FILE_WITH_NULL_METADATA})) AS input(value)"),
        (f"SELECT row_number() OVER (PARTITION BY struct_pack(value := value)) FROM (VALUES ({FILE})) AS input(value)"),
        f"SELECT count(DISTINCT value) FROM (VALUES ({FILE})) AS input(value)",
        f"SELECT count(DISTINCT value) OVER () FROM (VALUES ({FILE})) AS input(value)",
        f"PIVOT (SELECT {FILE} AS key, 1 AS value) ON key IN (NULL::FILE) USING sum(value)",
        (
            "WITH RECURSIVE input(value, depth) AS ("
            f"SELECT {FILE}, 0 "
            "UNION "
            "SELECT value, depth + 1 FROM input WHERE depth < 0"
            ") SELECT * FROM input"
        ),
        (
            "WITH RECURSIVE input(value, depth) USING KEY (value) AS ("
            f"SELECT {FILE}, 0 "
            "UNION ALL "
            "SELECT value, depth + 1 FROM input WHERE depth < 0"
            ") SELECT * FROM input"
        ),
        (
            f"SELECT 1 = ANY (SELECT 1 WHERE outer_input.value.url IS NOT NULL) "
            f"FROM (VALUES ({FILE})) AS outer_input(value)"
        ),
    ],
)
def test_file_rejects_query_hash_comparison_bypasses(connection, query):
    with pytest.raises(vane.BinderException, match=r"(?:does|do) not support FILE values"):
        connection.execute(query).fetchall()


@pytest.mark.parametrize("value", [FILE, f"struct_pack(value := {FILE})"])
def test_file_rejects_relation_hash_comparison_bypasses(connection, value):
    relation = connection.sql(f"SELECT {value} AS value")

    with pytest.raises(vane.BinderException, match="DISTINCT does not support FILE values"):
        relation.local_exchange(1).distinct().fetchall()

    with pytest.raises(vane.BinderException, match="Repartition keys do not support FILE values"):
        relation.repartition(2, "value").fetchall()


@pytest.mark.parametrize("value", [FILE, f"struct_pack(value := {FILE})"])
def test_file_rejects_copy_partition_hash_comparison_bypasses(connection, tmp_path: Path, value):
    destination = tmp_path / "partitioned"

    with pytest.raises(vane.BinderException, match="PARTITION_BY does not support FILE values"):
        connection.execute(
            f"COPY (SELECT {value} AS value, 1 AS payload) TO '{destination}' (FORMAT CSV, PARTITION_BY (value))"
        )


def test_file_remains_usable_outside_query_hash_keys(connection):
    rows = connection.execute(
        f"""
        SELECT 1 AS key, {FILE} AS value
        UNION ALL
        SELECT 2 AS key, file('other', NULL, NULL, NULL, NULL) AS value
        """
    ).fetchall()
    assert [row[1]["url"] for row in rows] == ["s3://bucket/missing.bin", "other"]

    row = connection.execute(
        f"""
        SELECT DISTINCT ON (key) key, value
        FROM (VALUES (1, {FILE}), (1, {FILE})) AS input(key, value)
        ORDER BY key
        """
    ).fetchone()
    assert row[0] == 1
    assert row[1]["url"] == "s3://bucket/missing.bin"

    row = connection.execute(
        f"""
        SELECT key, typeof(any_value(value)), any_value(value)
        FROM (VALUES (1, {FILE}), (1, {FILE})) AS input(key, value)
        GROUP BY key
        """
    ).fetchone()
    assert row[:2] == (1, "FILE")
    assert row[2]["url"] == "s3://bucket/missing.bin"

    rows = connection.execute(
        f"""
        WITH RECURSIVE files(depth, value) AS (
            SELECT 0, {FILE}
            UNION ALL
            SELECT depth + 1, value FROM files WHERE depth < 1
        )
        SELECT depth, value FROM files ORDER BY depth
        """
    ).fetchall()
    assert [row[0] for row in rows] == [0, 1]
    assert [row[1]["url"] for row in rows] == ["s3://bucket/missing.bin"] * 2

    rows = connection.execute(
        f"""
        SELECT (SELECT outer_input.value.url)
        FROM (VALUES ({FILE}), ({FILE})) AS outer_input(value)
        """
    ).fetchall()
    assert rows == [("s3://bucket/missing.bin",)] * 2


def test_file_array_alias_survives_tuple_collection_payloads(connection):
    rows = connection.execute(
        f"""
        SELECT key, typeof(files[1]), files[1].url
        FROM (
            VALUES
                (2, array_value(file('other', NULL, NULL, NULL, NULL))),
                (1, array_value({FILE}))
        ) AS input(key, files)
        ORDER BY key
        """
    ).fetchall()

    assert rows == [
        (1, "FILE", "s3://bucket/missing.bin"),
        (2, "FILE", "other"),
    ]


@pytest.mark.parametrize(
    "expression",
    [
        "hash(value)",
        "hash(struct_pack(value := value))",
        "approx_count_distinct(value)",
        "mode(value)",
        "entropy(value)",
        "approx_top_k(value, 2)",
        "list_approx_count_distinct([value, value])",
        "list_mode([value, value])",
        "list_entropy([value, value])",
    ],
)
def test_file_rejects_explicit_hash_consumers(connection, expression):
    with pytest.raises(vane.BinderException, match="does not support FILE values"):
        connection.execute(f"SELECT {expression} FROM (VALUES ({FILE_WITH_NULL_METADATA})) AS input(value)").fetchone()


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


def test_file_type_arrow_parent_null_hides_child_values(connection):
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
    hidden_file = pa.StructArray.from_arrays(
        [
            pa.array(["hidden-url"]),
            pa.array(["application/hidden"]),
            pa.array([1], type=pa.int64()),
            pa.array([2], type=pa.int64()),
            pa.array(["sha256:hidden"]),
        ],
        type=file_storage_type,
        mask=pa.array([True]),
    )
    file_field = pa.field(
        "value",
        file_storage_type,
        metadata={
            b"ARROW:extension:name": b"vane.file",
            b"ARROW:extension:metadata": b"",
        },
    )
    arrow_table = pa.Table.from_arrays([hidden_file], schema=pa.schema([file_field]))

    row = (
        connection.from_arrow(arrow_table)
        .project("value IS NULL, value.url, value.content_type, value.position, value.size, value.checksum")
        .fetchone()
    )
    assert row == (True, None, None, None, None, None)


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


def test_file_type_arrow_dictionary_validates_only_referenced_values(connection):
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

    class FileExtensionType(pa.ExtensionType):
        def __init__(self):
            pa.ExtensionType.__init__(self, file_storage_type, "vane.file")

        def __arrow_ext_serialize__(self):
            return b""

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            return cls()

    file_values = FileExtensionType().wrap_array(
        pa.array(
            [
                {
                    "url": "active",
                    "content_type": None,
                    "position": None,
                    "size": None,
                    "checksum": None,
                },
                {
                    "url": None,
                    "content_type": None,
                    "position": None,
                    "size": None,
                    "checksum": None,
                },
            ],
            type=file_storage_type,
        )
    )

    unreferenced_invalid = pa.DictionaryArray.from_arrays(pa.array([0], type=pa.int8()), file_values)
    rows = (
        connection.from_arrow(pa.table({"value": unreferenced_invalid})).project("typeof(value), value.url").fetchall()
    )
    assert rows == [("FILE", "active")]

    referenced_invalid = pa.DictionaryArray.from_arrays(pa.array([1], type=pa.int8()), file_values)
    with pytest.raises(vane.InvalidInputException, match="Arrow FILE url cannot be NULL"):
        connection.from_arrow(pa.table({"value": referenced_invalid})).fetchall()


def test_file_type_arrow_ignores_inactive_sparse_union_slots(connection):
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
        "f",
        file_storage_type,
        metadata={
            b"ARROW:extension:name": b"vane.file",
            b"ARROW:extension:metadata": b"",
        },
    )
    file_values = pa.array(
        [
            {
                "url": None,
                "content_type": None,
                "position": 0,
                "size": None,
                "checksum": None,
            },
            {
                "url": "active",
                "content_type": None,
                "position": None,
                "size": None,
                "checksum": None,
            },
        ],
        type=file_storage_type,
    )
    union_type = pa.sparse_union([file_field, pa.field("n", pa.int64())], type_codes=[0, 1])
    union_values = pa.UnionArray.from_sparse(
        pa.array([1, 0], type=pa.int8()),
        [file_values, pa.array([7, 8], type=pa.int64())],
        field_names=["f", "n"],
        type_codes=[0, 1],
    )
    arrow_table = pa.Table.from_arrays(
        [union_values],
        schema=pa.schema([pa.field("value", union_type)]),
    )

    rows = (
        connection.from_arrow(arrow_table)
        .project("union_tag(value), typeof(value.f), (value.f).url, value.n")
        .fetchall()
    )
    assert rows == [
        ("n", "FILE", None, 7),
        ("f", "FILE", "active", None),
    ]


def test_file_type_arrow_ignores_children_of_null_lists(connection):
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
        "item",
        file_storage_type,
        metadata={
            b"ARROW:extension:name": b"vane.file",
            b"ARROW:extension:metadata": b"",
        },
    )
    file_values = pa.array(
        [
            {
                "url": None,
                "content_type": None,
                "position": None,
                "size": None,
                "checksum": None,
            },
            {
                "url": "active",
                "content_type": None,
                "position": None,
                "size": None,
                "checksum": None,
            },
        ],
        type=file_storage_type,
    )
    list_type = pa.list_(file_field)
    list_values = pa.ListArray.from_arrays(
        pa.array([0, 1, 2], type=pa.int32()),
        file_values,
        type=list_type,
        mask=pa.array([True, False]),
    )
    arrow_table = pa.Table.from_arrays(
        [list_values],
        schema=pa.schema([pa.field("value", list_type)]),
    )

    rows = connection.from_arrow(arrow_table).project("value IS NULL, typeof(value[1]), (value[1]).url").fetchall()
    assert rows == [
        (True, "FILE", None),
        (False, "FILE", "active"),
    ]


def test_file_type_arrow_list_view_validates_sparse_child_span(connection):
    pa = pytest.importorskip("pyarrow")
    if not hasattr(pa, "ListViewArray"):
        pytest.skip("The PyArrow version does not support ListViewArray")

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
        "item",
        file_storage_type,
        metadata={
            b"ARROW:extension:name": b"vane.file",
            b"ARROW:extension:metadata": b"",
        },
    )
    file_values = [
        {
            "url": None,
            "content_type": None,
            "position": None,
            "size": None,
            "checksum": None,
        }
        for _ in range(101)
    ]
    file_values[0]["url"] = "first"
    file_values[100]["url"] = "last"
    list_type = pa.list_view(file_field)
    list_values = pa.ListViewArray.from_arrays(
        offsets=pa.array([0, 100], type=pa.int32()),
        sizes=pa.array([1, 1], type=pa.int32()),
        values=pa.array(file_values, type=file_storage_type),
        type=list_type,
    )
    arrow_table = pa.Table.from_arrays([list_values], schema=pa.schema([pa.field("value", list_type)]))

    rows = connection.from_arrow(arrow_table).project("typeof(value[1]), (value[1]).url").fetchall()
    assert rows == [("FILE", "first"), ("FILE", "last")]


def test_file_type_arrow_list_view_ignores_offsets_of_null_rows(connection):
    pa = pytest.importorskip("pyarrow")
    if not hasattr(pa, "ListViewArray"):
        pytest.skip("The PyArrow version does not support ListViewArray")

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
        "item",
        file_storage_type,
        metadata={
            b"ARROW:extension:name": b"vane.file",
            b"ARROW:extension:metadata": b"",
        },
    )
    list_type = pa.list_view(file_field)
    list_values = pa.ListViewArray.from_arrays(
        offsets=pa.array([0, 100], type=pa.int32()),
        sizes=pa.array([1, 1], type=pa.int32()),
        values=pa.array(
            [
                {
                    "url": "active",
                    "content_type": None,
                    "position": None,
                    "size": None,
                    "checksum": None,
                }
            ],
            type=file_storage_type,
        ),
        type=list_type,
        mask=pa.array([False, True]),
    )
    arrow_table = pa.Table.from_arrays([list_values], schema=pa.schema([pa.field("value", list_type)]))

    rows = connection.from_arrow(arrow_table).project("value IS NULL, typeof(value[1]), (value[1]).url").fetchall()
    assert rows == [
        (False, "FILE", "active"),
        (True, "FILE", None),
    ]

    nested_list_values = pa.ListViewArray.from_arrays(
        offsets=pa.array([0, 100], type=pa.int32()),
        sizes=pa.array([1, 1], type=pa.int32()),
        values=list_values.values,
        type=list_type,
    )
    parent_type = pa.struct([pa.field("items", list_type)])
    parent_values = pa.StructArray.from_arrays(
        [nested_list_values],
        fields=list(parent_type),
        mask=pa.array([False, True]),
    )
    parent_table = pa.Table.from_arrays(
        [parent_values],
        schema=pa.schema([pa.field("value", parent_type)]),
    )

    rows = (
        connection.from_arrow(parent_table)
        .project("value IS NULL, typeof(value.items[1]), (value.items[1]).url")
        .fetchall()
    )
    assert rows == [
        (False, "FILE", "active"),
        (True, "FILE", None),
    ]


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


def test_file_type_round_trips_through_parquet(connection, tmp_path: Path):
    parquet_path = tmp_path / "files.parquet"
    connection.execute(
        f"""
        COPY (
            SELECT
                {FILE} AS direct,
                struct_pack(value := {FILE}) AS nested,
                struct_pack(file_values := [{FILE}]) AS deeply_nested,
                [{FILE}] AS files,
                array_value({FILE}) AS fixed_files,
                MAP {{'value': {FILE}}} AS file_map,
                {{
                    'url': 'ordinary',
                    'content_type': NULL::VARCHAR,
                    'position': NULL::BIGINT,
                    'size': NULL::BIGINT,
                    'checksum': NULL::VARCHAR
                }} AS ordinary,
                CAST(union_value(f := {FILE}) AS UNION(f FILE, n BIGINT)) AS choice
        ) TO '{parquet_path}' (FORMAT PARQUET)
        """
    )

    row = connection.execute(
        f"""
        SELECT
            typeof(direct),
            typeof(nested.value),
            typeof(deeply_nested.file_values[1]),
            typeof(files[1]),
            typeof(fixed_files[1]),
            typeof(map_extract_value(file_map, 'value')),
            typeof(ordinary) <> 'FILE',
            union_tag(choice),
            typeof(choice.f),
            direct.url,
            (deeply_nested.file_values[1]).url,
            (map_extract_value(file_map, 'value')).url,
            choice.f.url
        FROM read_parquet('{parquet_path}')
        """
    ).fetchone()

    assert row == (
        "FILE",
        "FILE",
        "FILE",
        "FILE",
        "FILE",
        "FILE",
        True,
        "f",
        "FILE",
        "s3://bucket/missing.bin",
        "s3://bucket/missing.bin",
        "s3://bucket/missing.bin",
        "s3://bucket/missing.bin",
    )


def test_file_type_survives_parquet_database_export_import(connection, tmp_path: Path):
    export_path = tmp_path / "file_export"
    connection.execute(
        """
        CREATE TABLE exported_files(
            id INTEGER,
            direct FILE,
            nested STRUCT(value FILE),
            deeply_nested STRUCT(file_values FILE[]),
            files FILE[],
            fixed_files FILE[1],
            file_map MAP(VARCHAR, FILE),
            choice UNION(f FILE, n BIGINT)
        )
        """
    )
    connection.execute(
        f"""
        INSERT INTO exported_files VALUES (
            1,
            {FILE},
            struct_pack(value := {FILE}),
            struct_pack(file_values := [{FILE}]),
            [{FILE}],
            array_value({FILE}),
            MAP {{'value': {FILE}}},
            CAST(union_value(f := {FILE}) AS UNION(f FILE, n BIGINT))
        )
        """
    )

    connection.execute(f"EXPORT DATABASE '{export_path}' (FORMAT PARQUET)")
    connection.execute("DROP TABLE exported_files")
    connection.execute(f"IMPORT DATABASE '{export_path}'")

    row = connection.execute(
        """
        SELECT
            typeof(direct),
            typeof(nested.value),
            typeof(deeply_nested.file_values[1]),
            typeof(files[1]),
            typeof(fixed_files[1]),
            typeof(map_extract_value(file_map, 'value')),
            typeof(choice.f),
            direct.url,
            (deeply_nested.file_values[1]).url,
            (map_extract_value(file_map, 'value')).url,
            choice.f.url
        FROM exported_files
        """
    ).fetchone()
    assert row == (
        "FILE",
        "FILE",
        "FILE",
        "FILE",
        "FILE",
        "FILE",
        "FILE",
        "s3://bucket/missing.bin",
        "s3://bucket/missing.bin",
        "s3://bucket/missing.bin",
        "s3://bucket/missing.bin",
    )
