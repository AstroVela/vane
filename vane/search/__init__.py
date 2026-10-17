# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Optional persistent Tantivy full-text indexes for Vane and Arrow data.

See docs/search/USER_GUIDE.md for installation and the consistency contract.
Importing this module does not import Tantivy until an index is created/opened.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]

__all__ = ["FullTextIndex"]

_FORMAT_VERSION = 1
_ID = "_vane_id"
_SCORE = "_score"
_TOKENIZERS = frozenset({"default", "en_stem", "whitespace", "raw"})
_MAX_WINDOW = 10_000


def _load_tantivy() -> Any:
    try:
        import tantivy  # type: ignore[import-not-found, import-untyped, unused-ignore]
    except ModuleNotFoundError as error:
        if error.name != "tantivy":
            raise
        raise ImportError("Full-text search requires Tantivy; install 'vane-ai[search]'") from error
    return tantivy


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _columns(id_column: str, text_columns: Sequence[str]) -> tuple[str, ...]:
    if isinstance(text_columns, (str, bytes)) or not isinstance(text_columns, Sequence) or not text_columns:
        raise ValueError("text_columns must be a non-empty sequence of column names")
    columns = (id_column, *text_columns)
    for name in columns:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("column names must be ASCII identifiers")
        if name in {_ID, _SCORE}:
            raise ValueError(f"{name!r} is reserved for full-text search")
    if len(set(columns)) != len(columns):
        raise ValueError("id_column and text_columns must be distinct")
    return tuple(text_columns)


def _document_id(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 512 or "\x00" in value:
        raise ValueError("document IDs must be non-empty UTF-8 strings of at most 512 bytes, without NUL")
    return value


def _schema(tantivy: Any, columns: tuple[str, ...], tokenizer: str) -> Any:
    builder = tantivy.SchemaBuilder()
    builder.add_text_field(_ID, stored=True, tokenizer_name="raw", index_option="basic")
    for name in columns:
        builder.add_text_field(name, stored=True, tokenizer_name=tokenizer, index_option="position")
    return builder.build()


class FullTextIndex:
    """A local on-disk index, with one atomic commit per upsert/delete call.

    Use :meth:`create` or :meth:`open`. Indexes are independent of DuckDB
    transactions and are not automatically synchronized with source tables.
    Each operation acquires Tantivy's exclusive writer lock; other processes
    can read concurrently, but competing writers fail instead of retrying.
    """

    def __init__(self, path: Path, index: Any, id_column: str, columns: tuple[str, ...], tokenizer: str) -> None:
        self._path = path
        self._index = index
        self._id_column = id_column
        self._text_columns = columns
        self._tokenizer = tokenizer
        self._lock = Lock()
        index.config_reader(reload_policy="Manual")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def id_column(self) -> str:
        return self._id_column

    @property
    def text_columns(self) -> tuple[str, ...]:
        return self._text_columns

    @property
    def tokenizer(self) -> str:
        return self._tokenizer

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        id_column: str,
        text_columns: Sequence[str],
        tokenizer: str = "default",
    ) -> FullTextIndex:
        """Create an index in a new directory; never overwrite an existing path.

        ``default`` lowercases words, ``en_stem`` adds English stemming,
        ``whitespace`` preserves case and splits on whitespace, and ``raw``
        treats the entire field as one case-sensitive token.
        """
        columns = _columns(id_column, text_columns)
        if not isinstance(tokenizer, str) or tokenizer not in _TOKENIZERS:
            raise ValueError(f"tokenizer must be one of {sorted(_TOKENIZERS)}")
        tantivy = _load_tantivy()
        schema = _schema(tantivy, columns, tokenizer)
        root = Path(path).resolve()
        root.mkdir(parents=True, exist_ok=False)
        data_path = root / "index"
        data_path.mkdir()
        index = tantivy.Index(schema, path=str(data_path), reuse=False)
        # Publish the manifest last. A failed initialization is intentionally
        # left on disk for inspection and cannot be mistaken for a valid index.
        metadata = {
            "format_version": _FORMAT_VERSION,
            "id_column": id_column,
            "text_columns": list(columns),
            "tokenizer": tokenizer,
        }
        temporary = root / "vane-search.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temporary.replace(root / "vane-search.json")
        return cls(root, index, id_column, columns, tokenizer)

    @classmethod
    def open(cls, path: str | Path) -> FullTextIndex:
        """Open a Vane index and validate its manifest and Tantivy schema."""
        root = Path(path).resolve()
        metadata = json.loads((root / "vane-search.json").read_text(encoding="utf-8"))
        if (
            not isinstance(metadata, dict)
            or set(metadata) != {"format_version", "id_column", "text_columns", "tokenizer"}
            or type(metadata["format_version"]) is not int
            or metadata["format_version"] != _FORMAT_VERSION
        ):
            raise ValueError("Unsupported Vane full-text index manifest")
        columns = _columns(metadata["id_column"], metadata["text_columns"])
        tokenizer = metadata["tokenizer"]
        if not isinstance(tokenizer, str) or tokenizer not in _TOKENIZERS:
            raise ValueError("Unsupported tokenizer in Vane full-text index manifest")
        tantivy = _load_tantivy()
        index = tantivy.Index.open(str(root / "index"))
        if index.schema != _schema(tantivy, columns, tokenizer):
            raise ValueError("Tantivy schema does not match the Vane full-text index manifest")
        return cls(root, index, metadata["id_column"], columns, tokenizer)

    @contextmanager
    def _writer(self) -> Iterator[Any]:
        # One thread makes memory usage predictable and avoids spawning a
        # machine-sized pool for every bounded ingestion operation.
        with self._lock:
            writer = self._index.writer(heap_size=64_000_000, num_threads=1)
            try:
                yield writer
                writer.commit()
            except BaseException:
                writer.rollback()
                raise
            finally:
                writer.wait_merging_threads()

    @contextmanager
    def _batches(self, source: Any, batch_size: int) -> Iterator[Iterable[Any]]:
        owned_reader = None
        if isinstance(source, pa.Table):
            schema, batches = source.schema, source.to_batches(max_chunksize=batch_size)
        elif isinstance(source, pa.RecordBatch):
            schema, batches = source.schema, [source]
        elif isinstance(source, pa.RecordBatchReader):
            schema, batches = source.schema, source
        elif callable(getattr(source, "to_arrow_reader", None)):
            owned_reader = source.to_arrow_reader(batch_size=batch_size)
            schema, batches = owned_reader.schema, owned_reader
        else:
            raise TypeError("source must be a Vane Relation, Arrow Table, RecordBatch, or RecordBatchReader")
        try:
            names = (self._id_column, *self._text_columns)
            if len(set(schema.names)) != len(schema.names):
                raise ValueError("source columns must have unique names")
            for name in names:
                if name not in schema.names:
                    raise ValueError(f"source is missing column {name!r}")
                data_type = schema.field(name).type
                if not (pa.types.is_string(data_type) or pa.types.is_large_string(data_type)):
                    raise ValueError(f"source column {name!r} must have Arrow string or large_string type")
            yield batches
        finally:
            if owned_reader is not None:
                owned_reader.close()

    def upsert(self, source: Any, *, batch_size: int = 8192) -> int:
        """Stream rows into one commit; duplicate IDs use the last input row.

        All configured columns must exist and be strings. IDs cannot be null;
        text may be null. Extra columns are ignored. A validation or stream
        failure before commit rolls back the whole call. Returns input rows,
        including repeated IDs. The caller owns any supplied Arrow reader.
        """
        _positive_int("batch_size", batch_size)
        tantivy = _load_tantivy()
        count = 0
        names = [self._id_column, *self._text_columns]
        with self._batches(source, batch_size) as batches, self._writer() as writer:
            for batch in batches:
                selected = batch.select(names)
                for start in range(0, selected.num_rows, batch_size):
                    for row in selected.slice(start, batch_size).to_pylist():
                        key = _document_id(row[self._id_column])
                        values = {_ID: key}
                        values.update({name: row[name] for name in self._text_columns if row[name] is not None})
                        document = tantivy.Document.from_dict(values, self._index.schema)
                        writer.delete_documents(_ID, key)
                        writer.add_document(document)
                        count += 1
        return count

    def delete(self, ids: Iterable[str]) -> int:
        """Delete exact string IDs in one commit; missing IDs are harmless.

        Returns the number of supplied IDs, not the number of removed documents.
        """
        if isinstance(ids, (str, bytes)):
            raise TypeError("ids must be an iterable of strings, not a single string")
        count = 0
        with self._writer() as writer:
            for value in ids:
                writer.delete_documents(_ID, _document_id(value))
                count += 1
        return count

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
        fields: Sequence[str] | None = None,
        conjunction: bool = False,
    ) -> Any:
        """Return an Arrow Table ordered by descending BM25 score.

        Columns are the configured ID, ``_score`` (float32), and stored text.
        ``fields`` chooses default fields for unqualified query terms; explicit
        field qualifiers still use the whole index schema. Tantivy query syntax
        supports phrases, AND/OR/NOT and field qualifiers; invalid syntax raises
        ValueError. Every search reloads the latest committed snapshot.
        """
        _positive_int("limit", limit)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if limit + offset > _MAX_WINDOW:
            raise ValueError(f"limit + offset must not exceed {_MAX_WINDOW}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(conjunction, bool):
            raise TypeError("conjunction must be a bool")
        if fields is None:
            fields = self._text_columns
        if (
            isinstance(fields, (str, bytes))
            or not isinstance(fields, Sequence)
            or not fields
            or any(name not in self._text_columns for name in fields)
            or len(set(fields)) != len(fields)
        ):
            raise ValueError("fields must be a non-empty sequence of distinct indexed text columns")
        with self._lock:
            self._index.reload()
            parsed = self._index.parse_query(query, list(fields), conjunction_by_default=conjunction)
            searcher = self._index.searcher()
        # Keep this exact searcher alive for both addresses and stored values:
        # a concurrent commit/merge must not mix different reader generations.
        result = searcher.search(parsed, limit=limit, offset=offset, count=False)
        rows = []
        for score, address in result.hits:
            document = searcher.doc(address).to_dict()
            row = {self._id_column: document[_ID][0], _SCORE: score}
            row.update({name: document.get(name, [None])[0] for name in self._text_columns})
            rows.append(row)
        schema = pa.schema(
            [(self._id_column, pa.string()), (_SCORE, pa.float32())]
            + [(name, pa.string()) for name in self._text_columns]
        )
        return pa.Table.from_pylist(rows, schema=schema)
