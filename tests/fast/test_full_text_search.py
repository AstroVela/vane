# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import json

import pyarrow as pa
import pytest

from vane.search import FullTextIndex

pytest.importorskip("tantivy")


def data(ids, titles, bodies=None):
    return pa.table(
        {"id": ids, "title": titles, "body": bodies or [None] * len(ids)},
        schema=pa.schema([(name, pa.string()) for name in ("id", "title", "body")]),
    )


@pytest.fixture
def index(tmp_path):
    return FullTextIndex.create(tmp_path / "search", id_column="id", text_columns=["title", "body"])


def ids(result):
    return result.column("id").to_pylist()


def test_ranking_queries_and_projection(index):
    index.upsert(
        data(
            ["a", "b", "c"],
            ["rust rust rust", "rust engine with many unrelated words in this title", "python engine"],
            ["fast search", "distributed search", None],
        )
    )
    hits = index.search("rust")
    assert ids(hits) == ["a", "b"]
    assert hits["_score"][0].as_py() > hits["_score"][1].as_py() > 0
    assert hits.column_names == ["id", "_score", "title", "body"]
    assert ids(index.search('body:"fast search"')) == ["a"]
    assert ids(index.search("rust AND distributed")) == ["b"]
    assert ids(index.search("rust NOT distributed")) == ["a"]
    assert set(ids(index.search("rust OR python"))) == {"a", "b", "c"}
    assert ids(index.search("rust distributed", conjunction=True)) == ["b"]
    assert ids(index.search("search", fields=["title"])) == []
    assert set(ids(index.search("search", fields=["body"]))) == {"a", "b"}
    assert ids(index.search("rust", limit=1, offset=1)) == ["b"]


def test_empty_result_has_stable_schema(index):
    empty = index.search("absent")
    assert empty.schema == pa.schema(
        [("id", pa.string()), ("_score", pa.float32()), ("title", pa.string()), ("body", pa.string())]
    )
    assert empty.num_rows == 0
    index.upsert(data(["a"], ["rust"]))
    assert index.search("rust")["body"].to_pylist() == [None]
    assert index.search("rust", offset=10).schema == empty.schema


def test_upsert_delete_and_reopen(index):
    index.upsert(data(["a", "b"], ["before", "keep"]))
    assert index.upsert(data(["a", "a"], ["middle", "after"]), batch_size=1) == 2
    reopened = FullTextIndex.open(index.path)
    assert reopened.id_column == "id"
    assert reopened.text_columns == ("title", "body")
    assert ids(reopened.search("after")) == ["a"]
    assert ids(reopened.search("before OR middle")) == []
    assert reopened.delete(["a", "missing", "a"]) == 3
    assert ids(index.search("after")) == []  # another handle sees the commit
    assert ids(index.search("keep")) == ["b"]
    assert index.delete([]) == 0
    assert index.upsert(data([], [])) == 0
    assert index.upsert(data(["a"], ["again"])) == 1  # writer lock released


def test_exact_string_ids(index):
    values = ["A", "a", "001", "1", "有 空格", "x:y"]
    index.upsert(data(values, ["match"] * len(values)))
    index.delete(["a", "1", "有 空格"])
    assert set(ids(index.search("match"))) == {"A", "001", "x:y"}


@pytest.mark.parametrize("bad_id", [None, "", "a\x00b", "界" * 171])
def test_invalid_id_rolls_back_all_batches(index, bad_id):
    index.upsert(data(["a"], ["original"]))
    with pytest.raises(ValueError, match="document IDs"):
        index.upsert(data(["a", "b", bad_id], ["changed", "added", "invalid"]), batch_size=1)
    assert ids(index.search("original")) == ["a"]
    assert ids(index.search("changed OR added")) == []
    index.upsert(data(["c"], ["recovered"]))
    assert ids(index.search("recovered")) == ["c"]


def test_stream_failure_rolls_back(index):
    index.upsert(data(["a"], ["original"]))

    def batches():
        yield data(["a"], ["changed"]).to_batches()[0]
        raise RuntimeError("source failed")

    reader = pa.RecordBatchReader.from_batches(data([], []).schema, batches())
    with pytest.raises(RuntimeError, match="source failed"):
        index.upsert(reader)
    assert ids(index.search("original")) == ["a"]
    assert ids(index.search("changed")) == []


def test_delete_failure_rolls_back(index):
    index.upsert(data(["a", "b"], ["match", "match"]))
    with pytest.raises(ValueError, match="document IDs"):
        index.delete(["a", None])
    assert set(ids(index.search("match"))) == {"a", "b"}
    with pytest.raises(TypeError, match="single string"):
        index.delete("a")


@pytest.mark.parametrize("kind", ["table", "batch", "reader"])
def test_arrow_inputs(index, kind):
    table = data(["a", "b"], ["needle", "other"]).append_column("ignored", pa.array([1, 2]))
    source = {"table": table, "batch": table.to_batches()[0], "reader": table.to_reader()}[kind]
    assert index.upsert(source, batch_size=1) == 2
    assert ids(index.search("needle")) == ["a"]


def test_large_string_and_null_text(index):
    table = data(["a"], ["needle"]).cast(pa.schema([(name, pa.large_string()) for name in ("id", "title", "body")]))
    index.upsert(table)
    assert index.search("needle").to_pylist()[0]["body"] is None


@pytest.mark.parametrize(
    "source, message",
    [
        (pa.table({"id": [1], "title": ["x"], "body": ["y"]}), "must have Arrow string"),
        (pa.table({"id": ["a"], "title": [1], "body": ["y"]}), "must have Arrow string"),
        (pa.table({"id": ["a"], "title": ["x"]}), "missing column"),
        (pa.table([["a"], ["x"], ["y"]], names=["id", "title", "title"]), "unique names"),
    ],
)
def test_source_schema_validation(index, source, message):
    with pytest.raises(ValueError, match=message):
        index.upsert(source)
    assert index.upsert(data(["a"], ["works"])) == 1


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_size(index, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        index.upsert(data([], []), batch_size=batch_size)


@pytest.mark.parametrize(
    "options",
    [
        {"limit": 0},
        {"limit": True},
        {"limit": 1.5},
        {"limit": 10001},
        {"limit": 10, "offset": 9991},
        {"offset": -1},
        {"offset": True},
        {"fields": []},
        {"fields": "title"},
        {"fields": ["title", "title"]},
        {"fields": ["unknown"]},
    ],
)
def test_search_argument_validation(index, options):
    with pytest.raises(ValueError):
        index.search("needle", **options)


@pytest.mark.parametrize("query", ["", "  ", None, 'title:"unterminated', "unknown:foo"])
def test_invalid_query(index, query):
    with pytest.raises(ValueError):
        index.search(query)


@pytest.mark.parametrize(
    "id_column, columns",
    [
        ("id", []),
        ("id", "title"),
        ("id", ["id"]),
        ("id", ["x", "x"]),
        ("_score", ["x"]),
        ("id", ["_vane_id"]),
        ("id", ["x:y"]),
    ],
)
def test_invalid_index_columns(tmp_path, id_column, columns):
    with pytest.raises(ValueError):
        FullTextIndex.create(tmp_path / "search", id_column=id_column, text_columns=columns)
    assert not (tmp_path / "search").exists()


@pytest.mark.parametrize(
    "tokenizer, text, query",
    [
        ("default", "Hello world", "HELLO"),
        ("en_stem", "running", "run"),
        ("whitespace", "中文 搜索", "搜索"),
        ("raw", "Exact", "Exact"),
    ],
)
def test_tokenizers_survive_reopen(tmp_path, tokenizer, text, query):
    index = FullTextIndex.create(tmp_path / "search", id_column="id", text_columns=["title"], tokenizer=tokenizer)
    index.upsert(data(["a"], [text]))
    assert ids(FullTextIndex.open(index.path).search(query)) == ["a"]


def test_create_never_overwrites_existing_directory(index):
    index.upsert(data(["a"], ["preserved"]))
    with pytest.raises(FileExistsError):
        FullTextIndex.create(index.path, id_column="id", text_columns=["title"])
    assert ids(index.search("preserved")) == ["a"]


@pytest.mark.parametrize(
    "change", [{"format_version": 2}, {"format_version": True}, {"tokenizer": "unknown"}, {"text_columns": ["other"]}]
)
def test_manifest_validation(index, change):
    path = index.path / "vane-search.json"
    metadata = json.loads(path.read_text())
    metadata.update(change)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        FullTextIndex.open(index.path)


def test_writer_lock_failure_preserves_index(index):
    import tantivy

    index.upsert(data(["a"], ["original"]))
    other = tantivy.Index.open(str(index.path / "index"))
    writer = other.writer(heap_size=15_000_000, num_threads=1)
    try:
        with pytest.raises(ValueError, match="lock|Lock"):
            index.upsert(data(["b"], ["added"]))
        assert ids(index.search("original")) == ["a"]
    finally:
        writer.rollback()
        writer.wait_merging_threads()
    index.upsert(data(["b"], ["added"]))
    assert ids(index.search("added")) == ["b"]


def test_missing_optional_dependency_error(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "tantivy":
            raise ModuleNotFoundError("No module named 'tantivy'", name="tantivy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"vane-ai\[search\]"):
        FullTextIndex.create(tmp_path / "search", id_column="id", text_columns=["title"])
    assert not (tmp_path / "search").exists()


def test_vane_relation_roundtrip(index):
    import vane

    with vane.connect() as connection:
        relation = connection.sql("SELECT 'a' AS id, 'Tantivy search' AS title, NULL::VARCHAR AS body")
        assert index.upsert(relation, batch_size=1) == 1
        hits = index.search("tantivy")
        connection.register("search_hits", hits)
        assert connection.sql("SELECT id, title FROM search_hits WHERE _score > 0").fetchall() == [
            ("a", "Tantivy search")
        ]
        assert connection.from_arrow(hits).project("id").fetchall() == [("a",)]
