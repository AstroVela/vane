# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0


import pytest

import vane

_WHEN_CLAUSES = [
    "WHEN MATCHED THEN UPDATE SET value = source.value",
    "WHEN NOT MATCHED THEN INSERT (id, value) VALUES (source.id, source.value)",
]


def _merge_connection(database, *, backend="local"):
    with vane.connect(str(database)) as connection:
        connection.execute("CREATE TABLE merge_target (id INTEGER PRIMARY KEY, value VARCHAR)")
        connection.execute("INSERT INTO merge_target VALUES (1, 'old'), (3, 'keep')")
        connection.execute("CREATE TABLE merge_source (id INTEGER, value VARCHAR)")
        connection.execute("INSERT INTO merge_source VALUES (1, 'new'), (2, 'inserted')")
    return vane.connect(str(database), backend=backend)


def _merge(source, **kwargs):
    return source.merge_into(
        "merge_target",
        "target.id = source.id",
        _WHEN_CLAUSES,
        **kwargs,
    )


def test_merge_relation_runs_with_explicit_local(tmp_path):
    connection = _merge_connection(tmp_path / "merge.duckdb")
    try:
        assert _merge(connection.table("merge_source")) is None
        assert connection.execute("SELECT * FROM merge_target ORDER BY id").fetchall() == [
            (1, "new"),
            (2, "inserted"),
            (3, "keep"),
        ]
    finally:
        connection.close()


def test_merge_relation_supports_using_columns(tmp_path):
    connection = _merge_connection(tmp_path / "merge.duckdb")
    try:
        connection.table("merge_source").merge_into(
            "merge_target",
            ["id"],
            _WHEN_CLAUSES,
        )
        assert connection.execute("SELECT * FROM merge_target ORDER BY id").fetchall() == [
            (1, "new"),
            (2, "inserted"),
            (3, "keep"),
        ]
    finally:
        connection.close()


def test_merge_relation_supports_expression_condition_and_custom_aliases(tmp_path):
    connection = _merge_connection(tmp_path / "merge.duckdb")
    try:
        condition = vane.ColumnExpression("destination.id") == vane.ColumnExpression("changes.id")
        connection.table("merge_source").merge_into(
            "merge_target",
            condition,
            [
                "WHEN MATCHED THEN UPDATE SET value = changes.value",
                "WHEN NOT MATCHED THEN INSERT (id, value) VALUES (changes.id, changes.value)",
            ],
            target_alias="destination",
            source_alias="changes",
        )
        assert connection.execute("SELECT * FROM merge_target ORDER BY id").fetchall() == [
            (1, "new"),
            (2, "inserted"),
            (3, "keep"),
        ]
    finally:
        connection.close()


def test_merge_relation_separates_line_comment_clauses(tmp_path):
    connection = _merge_connection(tmp_path / "merge.duckdb")
    try:
        connection.table("merge_source").merge_into(
            "merge_target",
            "target.id = source.id",
            [
                "WHEN MATCHED THEN UPDATE SET value = source.value -- update existing rows",
                "WHEN NOT MATCHED THEN INSERT (id, value) VALUES (source.id, source.value)",
            ],
        )
        assert connection.execute("SELECT * FROM merge_target ORDER BY id").fetchall() == [
            (1, "new"),
            (2, "inserted"),
            (3, "keep"),
        ]
    finally:
        connection.close()


def test_merge_relation_accepts_sql_whitespace_after_when(tmp_path):
    connection = _merge_connection(tmp_path / "merge.duckdb")
    try:
        connection.table("merge_source").merge_into(
            "merge_target",
            "target.id = source.id",
            [
                "WHEN\tMATCHED THEN UPDATE SET value = source.value",
                "WHEN\nNOT MATCHED THEN INSERT (id, value) VALUES (source.id, source.value)",
            ],
        )
        assert connection.execute("SELECT * FROM merge_target ORDER BY id").fetchall() == [
            (1, "new"),
            (2, "inserted"),
            (3, "keep"),
        ]
    finally:
        connection.close()


def test_merge_relation_accepts_sql_comments_after_when(tmp_path):
    connection = _merge_connection(tmp_path / "merge.duckdb")
    try:
        connection.table("merge_source").merge_into(
            "merge_target",
            "target.id = source.id",
            [
                "WHEN/* update existing rows */MATCHED THEN UPDATE SET value = source.value",
                "WHEN-- insert missing rows\nNOT MATCHED THEN INSERT (id, value) VALUES (source.id, source.value)",
            ],
        )
        assert connection.execute("SELECT * FROM merge_target ORDER BY id").fetchall() == [
            (1, "new"),
            (2, "inserted"),
            (3, "keep"),
        ]
    finally:
        connection.close()


def test_merge_relation_rejects_ray_backend(tmp_path):
    database = tmp_path / "merge.duckdb"
    with _merge_connection(database, backend="ray") as connection:
        with pytest.raises(vane.NotImplementedException, match="require backend='local'"):
            _merge(connection.table("merge_source"))
    with vane.connect(str(database)) as connection:
        assert connection.execute("SELECT * FROM merge_target ORDER BY id").fetchall() == [
            (1, "old"),
            (3, "keep"),
        ]


@pytest.mark.parametrize(
    ("condition", "when_clauses", "kwargs", "message"),
    [
        ("", _WHEN_CLAUSES, {}, "non-empty MERGE condition"),
        ([], _WHEN_CLAUSES, {}, "at least one MERGE USING column"),
        (["id", 2], _WHEN_CLAUSES, {}, "MERGE USING columns as strings"),
        ("target.id = source.id", [], {}, "at least one MERGE WHEN clause"),
        ("target.id = source.id", "WHEN MATCHED THEN DELETE", {}, "sequence of SQL strings"),
        ("target.id = source.id", ["UPDATE SET value = source.value"], {}, "must start with WHEN"),
        ("target.id = source.id", ["WHENEVER MATCHED THEN DELETE"], {}, "must start with WHEN"),
        ("target.id = source.id", _WHEN_CLAUSES, {"source_alias": "target"}, "must be different"),
    ],
)
def test_merge_relation_validates_api_inputs(condition, when_clauses, kwargs, message, tmp_path):
    connection = _merge_connection(tmp_path / "merge.duckdb")
    try:
        with pytest.raises(vane.InvalidInputException, match=message):
            connection.table("merge_source").merge_into(
                "merge_target",
                condition,
                when_clauses,
                **kwargs,
            )
    finally:
        connection.close()
