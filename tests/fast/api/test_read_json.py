# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from io import StringIO
from pathlib import Path

import pytest

import vane


@pytest.fixture(params=["connection", "module"])
def read_json(request, duckdb_cursor):
    if request.param == "connection":
        return duckdb_cursor.read_json
    return lambda source, **options: vane.read_json(source, connection=duckdb_cursor, **options)


def test_json_multifile_options(read_json, tmp_path):
    directory = tmp_path / "partition=7"
    directory.mkdir()
    (directory / "first.json").write_text('{"value": 1}\n', encoding="utf-8")
    (directory / "second.json").write_text('{"value": 2, "extra": "two"}\n', encoding="utf-8")

    relation = read_json(
        str(directory / "*.json"),
        format="newline_delimited",
        filename=True,
        hive_partitioning=True,
        hive_types={"partition": "INTEGER"},
        union_by_name=True,
    )
    rows = relation.project("value, extra, partition, filename").order("value").fetchall()
    assert [row[:3] for row in rows] == [(1, None, 7), (2, "two", 7)]
    assert [Path(row[3]).name for row in rows] == ["first.json", "second.json"]


@pytest.mark.parametrize("option", ["hive_partitioning", "union_by_name", "hive_types_autocast"])
def test_json_multifile_options_require_booleans(read_json, tmp_path, option):
    source = tmp_path / "values.json"
    source.write_text('[{"value": 1}]', encoding="utf-8")
    with pytest.raises(vane.BinderException, match=f"read_json only accepts '{option}' as a boolean"):
        read_json(source, **{option: "true"})


def test_json_buffer_with_explicit_types(read_json):
    pytest.importorskip("fsspec")
    source = StringIO('[{"value": "42"}]')
    relation = read_json(source, columns={"value": "INTEGER"}, format="array")
    assert relation.fetchall() == [(42,)]
