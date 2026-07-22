#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare Common Crawl text chunks and embeddings with Vane.

This example adapts Daft's Common Crawl tutorial:
https://docs.daft.ai/en/stable/examples/common-crawl-daft-tutorial/

The Daft tutorial reads Common Crawl WET text records, filters converted web
page text, parses WARC headers for language, keeps English pages, chunks text
into sentences, embeds each chunk, and writes the result.

This Vane version keeps the same shape:

1. Load sample Common Crawl-shaped records, a local WET/WARC file, or a WET URL.
2. Filter ``WARC-Type = conversion`` and decode ``warc_content`` as UTF-8.
3. Parse ``warc_headers`` and keep English records.
4. Sentence-chunk text and embed chunks with ``vane.ai.embed_text``.
5. Write page, chunk, and embedding outputs.

The default source is a built-in sample so the example can run without Common
Crawl network access. Use ``--source wet-file`` or ``--source wet-url`` for real
Common Crawl WET files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

import vane
from vane.ai import embed_text

DEFAULT_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUTPUT_DIR = Path("examples/output/common_crawl")


SAMPLE_RECORDS = [
    {
        "record_id": "<urn:uuid:vane-sample-0>",
        "warc_type": "conversion",
        "date": "2026-06-04T00:00:00Z",
        "uri": "https://example.com/vane",
        "language": "eng",
        "text": (
            "Vane is an analytical engine for AI data workflows. It can load "
            "tables, invoke models in batches, and write structured outputs. "
            "This makes it useful for preparing retrieval datasets from web text."
        ),
    },
    {
        "record_id": "<urn:uuid:vane-sample-1>",
        "warc_type": "conversion",
        "date": "2026-06-04T00:01:00Z",
        "uri": "https://example.com/common-crawl",
        "language": "eng",
        "text": (
            "Common Crawl contains web pages from many domains and languages. "
            "A useful preprocessing pipeline filters pages, extracts metadata, "
            "chunks text into sentences, and creates embeddings for search."
        ),
    },
    {
        "record_id": "<urn:uuid:vane-sample-2>",
        "warc_type": "conversion",
        "date": "2026-06-04T00:02:00Z",
        "uri": "https://example.com/multilingual",
        "language": "fra",
        "text": (
            "Cette page est un exemple de contenu non anglais. Elle doit etre "
            "filtree lorsque nous gardons uniquement la langue eng."
        ),
    },
    {
        "record_id": "<urn:uuid:vane-sample-3>",
        "warc_type": "metadata",
        "date": "2026-06-04T00:03:00Z",
        "uri": "https://example.com/metadata",
        "language": "eng",
        "text": "This metadata record is not converted web page text.",
    },
    {
        "record_id": "<urn:uuid:vane-sample-4>",
        "warc_type": "conversion",
        "date": "2026-06-04T00:04:00Z",
        "uri": "https://example.com/release",
        "language": "eng",
        "text": (
            "Before publishing open source software, keep packaging metadata, "
            "license text, README examples, and generated artifacts easy to "
            "audit. Small runnable examples help users validate the package."
        ),
    },
]


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def relation_from_rows(
    conn: Any,
    rows: list[dict[str, Any]],
    schema: dict[str, str],
) -> Any:
    """Build a VALUES relation whose data can travel with a Ray plan."""
    if not rows:
        raise RuntimeError("Cannot create a relation from zero rows.")
    columns = list(schema)
    constant = vane.ConstantExpression
    raw = conn.values(
        *(tuple(constant(row[column]) for column in columns) for row in rows),
    )
    projections = [
        f"{quote_ident(source)}::{schema[column]} AS {quote_ident(column)}"
        for source, column in zip(raw.columns, columns, strict=True)
    ]
    return raw.query("input_rows", f"select {', '.join(projections)} from input_rows")


def collect_relation(rel: Any) -> pa.Table:
    """Materialize a relation through the configured default runner."""
    tables = list(vane.runners.get_or_create_runner().run_iter_tables(rel))
    if not tables:
        return pa.table({column: pa.array([]) for column in rel.columns})
    table = pa.concat_tables(tables)
    expected_columns = list(rel.columns)
    if table.column_names != expected_columns:
        table = table.rename_columns(expected_columns)
    return table


WARC_SCHEMA = {
    "WARC-Record-ID": "VARCHAR",
    "WARC-Type": "VARCHAR",
    "WARC-Target-URI": "VARCHAR",
    "WARC-Date": "VARCHAR",
    "Content-Length": "BIGINT",
    "warc_content": "BLOB",
    "warc_headers": "VARCHAR",
}


def sample_relation(conn: Any, limit: int) -> Any:
    rows = []
    for i in range(limit):
        record = SAMPLE_RECORDS[i % len(SAMPLE_RECORDS)]
        text = record["text"]
        headers = {
            "Content-Type": "text/plain",
            "WARC-Block-Digest": f"sha1:SAMPLE{i:08d}",
            "WARC-Identified-Content-Language": record["language"],
            "WARC-Refers-To": record["record_id"],
            "WARC-Target-URI": record["uri"],
        }
        suffix = "" if i < len(SAMPLE_RECORDS) else f"-copy-{i // len(SAMPLE_RECORDS)}"
        rows.append(
            {
                "WARC-Record-ID": record["record_id"].replace(">", f"{suffix}>"),
                "WARC-Type": record["warc_type"],
                "WARC-Target-URI": record["uri"],
                "WARC-Date": record["date"],
                "Content-Length": len(text.encode("utf-8")),
                "warc_content": text.encode("utf-8"),
                "warc_headers": json.dumps(headers),
            }
        )
    return relation_from_rows(conn, rows, WARC_SCHEMA)


def parse_warc_header_lines(header_text: str) -> dict[str, str]:
    headers = {}
    for line in header_text.splitlines():
        if not line or line.startswith("WARC/"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip()] = value.strip()
    return headers


def parse_wet_bytes(data: bytes, limit: int) -> list[dict[str, Any]]:
    parts = re.split(rb"(?m)^WARC/1\.0\r?\n", data)
    rows = []
    for part in parts:
        if not part.strip():
            continue
        if b"\r\n\r\n" in part:
            header_bytes, content = part.split(b"\r\n\r\n", 1)
        elif b"\n\n" in part:
            header_bytes, content = part.split(b"\n\n", 1)
        else:
            continue

        headers = parse_warc_header_lines("WARC/1.0\n" + header_bytes.decode("utf-8", errors="replace"))
        record_id = headers.get("WARC-Record-ID", f"record-{len(rows)}")
        warc_type = headers.get("WARC-Type", "")
        uri = headers.get("WARC-Target-URI", "")
        warc_date = headers.get("WARC-Date", "")
        content_length = int(headers.get("Content-Length", len(content)) or 0)
        rows.append(
            {
                "WARC-Record-ID": record_id,
                "WARC-Type": warc_type,
                "WARC-Target-URI": uri,
                "WARC-Date": warc_date,
                "Content-Length": content_length,
                "warc_content": content.strip(b"\r\n"),
                "warc_headers": json.dumps(headers),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def read_wet_file(path: str, limit: int) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    if path.endswith(".gz"):
        raw = gzip.decompress(raw)
    return parse_wet_bytes(raw, limit)


def read_wet_url(url: str, limit: int) -> list[dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": "vane-example/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    if url.endswith(".gz"):
        raw = gzip.decompress(raw)
    return parse_wet_bytes(raw, limit)


def wet_relation(conn: Any, rows: list[dict[str, Any]]) -> Any:
    if not rows:
        raise RuntimeError("No WET/WARC records were parsed.")
    return relation_from_rows(conn, rows, WARC_SCHEMA)


class DecodeWarcBatch:
    """Decode WARC content bytes and parse WARC headers."""

    def __call__(self, batch: pa.Table) -> pa.Table:
        record_ids = batch["WARC-Record-ID"].to_pylist()
        target_uris = batch["WARC-Target-URI"].to_pylist()
        dates = batch["WARC-Date"].to_pylist()
        lengths = batch["Content-Length"].to_pylist()
        content_values = batch["warc_content"].to_pylist()
        header_values = batch["warc_headers"].to_pylist()

        texts = []
        languages = []
        for content, raw_headers in zip(content_values, header_values, strict=True):
            try:
                text = bytes(content or b"").decode("utf-8")
            except UnicodeDecodeError:
                text = None
            try:
                headers = json.loads(raw_headers or "{}")
            except json.JSONDecodeError:
                headers = {}
            languages.append(str(headers.get("WARC-Identified-Content-Language") or ""))
            texts.append(text)

        return pa.table(
            {
                "record_id": pa.array(record_ids, type=pa.string()),
                "target_uri": pa.array(target_uris, type=pa.string()),
                "warc_date": pa.array(dates, type=pa.string()),
                "content_length": pa.array(lengths, type=pa.int64()),
                "language": pa.array(languages, type=pa.string()),
                "text": pa.array(texts, type=pa.string()),
            }
        )


def regex_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", normalized)
    return [piece.strip() for piece in pieces if piece.strip()]


def split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            split_at = text.rfind(" ", start, end)
            if split_at > start + max_chars // 2:
                end = split_at
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


class ChunkTextBatch:
    """Split decoded web page text into embedding-sized chunks."""

    def __init__(self, *, max_doc_chars: int, max_chunk_chars: int):
        self.max_doc_chars = max_doc_chars
        self.max_chunk_chars = max_chunk_chars

    def __call__(self, batch: pa.Table) -> pa.Table:
        output = {
            "record_id": [],
            "target_uri": [],
            "warc_date": [],
            "language": [],
            "chunk_id": [],
            "text": [],
        }

        rows = batch.to_pylist()
        for row in rows:
            text = str(row["text"] or "")
            if self.max_doc_chars and len(text) > self.max_doc_chars:
                text = text[: self.max_doc_chars]

            sentence_id = 0
            for sentence in regex_sentences(text):
                for chunk in split_long_text(sentence, self.max_chunk_chars):
                    output["record_id"].append(row["record_id"])
                    output["target_uri"].append(row["target_uri"])
                    output["warc_date"].append(row["warc_date"])
                    output["language"].append(row["language"])
                    output["chunk_id"].append(sentence_id)
                    output["text"].append(chunk)
                    sentence_id += 1

        return pa.table(
            {
                "record_id": pa.array(output["record_id"], type=pa.string()),
                "target_uri": pa.array(output["target_uri"], type=pa.string()),
                "warc_date": pa.array(output["warc_date"], type=pa.string()),
                "language": pa.array(output["language"], type=pa.string()),
                "chunk_id": pa.array(output["chunk_id"], type=pa.int64()),
                "text": pa.array(output["text"], type=pa.string()),
            }
        )


def append_embedding(base: pa.Table, embedding_rel: Any) -> pa.Table:
    embeddings = collect_relation(embedding_rel)
    if base.num_rows != embeddings.num_rows:
        raise RuntimeError(f"Embedding row count mismatch: {embeddings.num_rows} vs {base.num_rows}")
    if "embedding" not in embeddings.column_names:
        raise RuntimeError("Embedding output column was not returned.")
    return base.append_column("embedding", embeddings["embedding"])


def embedding_dims(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def save_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(
    *,
    pages_table: pa.Table,
    chunks_table: pa.Table,
    embedded_table: pa.Table | None,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    page_rows = []
    for row in pages_table.to_pylist():
        text = str(row["text"] or "")
        page_rows.append(
            {
                "record_id": row["record_id"],
                "target_uri": row["target_uri"],
                "warc_date": row["warc_date"],
                "language": row["language"],
                "content_length": row["content_length"],
                "text_chars": len(text),
                "text_preview": text[:240],
            }
        )
    save_csv(
        page_rows,
        output_dir / "filtered_pages.csv",
        [
            "record_id",
            "target_uri",
            "warc_date",
            "language",
            "content_length",
            "text_chars",
            "text_preview",
        ],
    )

    chunk_rows = chunks_table.to_pylist()
    save_csv(
        chunk_rows,
        output_dir / "chunks.csv",
        ["record_id", "target_uri", "warc_date", "language", "chunk_id", "text"],
    )

    if embedded_table is None:
        return

    pq.write_table(embedded_table, output_dir / "chunk_embeddings.parquet")
    embedding_rows = []
    for row in embedded_table.to_pylist():
        embedding_rows.append(
            {
                "record_id": row["record_id"],
                "target_uri": row["target_uri"],
                "warc_date": row["warc_date"],
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "embedding_dim": embedding_dims(row["embedding"]),
            }
        )
    save_csv(
        embedding_rows,
        output_dir / "chunk_embeddings.csv",
        [
            "record_id",
            "target_uri",
            "warc_date",
            "chunk_id",
            "text",
            "embedding_dim",
        ],
    )


def load_source_relation(conn: Any, args: argparse.Namespace) -> Any:
    if args.source == "sample":
        return sample_relation(conn, args.limit)
    if args.source == "wet-file":
        if not args.wet_path:
            raise SystemExit("--wet-path is required when --source wet-file.")
        return wet_relation(conn, read_wet_file(args.wet_path, args.limit))
    if args.source == "wet-url":
        if not args.wet_url:
            raise SystemExit("--wet-url is required when --source wet-url.")
        return wet_relation(conn, read_wet_url(args.wet_url, args.limit))
    raise ValueError(f"Unsupported source: {args.source}")


def run(args: argparse.Namespace) -> None:
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1.")
    if args.max_doc_chars < 1:
        raise SystemExit("--max-doc-chars must be at least 1.")
    if args.max_chunk_chars < 1:
        raise SystemExit("--max-chunk-chars must be at least 1.")

    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    conn = vane.connect()
    rel = load_source_relation(conn, args)
    filtered = rel.query(
        "cc",
        """
        select *
        from cc
        where "WARC-Type" = 'conversion'
        """,
    )

    decoder = DecodeWarcBatch()
    pages = filtered.map_batches(
        decoder.__call__,
        schema={
            "record_id": vane.sqltypes.VARCHAR,
            "target_uri": vane.sqltypes.VARCHAR,
            "warc_date": vane.sqltypes.VARCHAR,
            "content_length": vane.sqltypes.BIGINT,
            "language": vane.sqltypes.VARCHAR,
            "text": vane.sqltypes.VARCHAR,
        },
        batch_size=args.batch_size,
    ).query(
        "pages",
        f"""
        select *
        from pages
        where text is not null
          and language = {sql_literal(args.language)}
        """,
    )
    pages_table = collect_relation(pages)
    if pages_table.num_rows == 0:
        raise RuntimeError("No decoded pages matched the requested language.")

    chunker = ChunkTextBatch(
        max_doc_chars=args.max_doc_chars,
        max_chunk_chars=args.max_chunk_chars,
    )
    chunks = pages.map_batches(
        chunker.__call__,
        schema={
            "record_id": vane.sqltypes.VARCHAR,
            "target_uri": vane.sqltypes.VARCHAR,
            "warc_date": vane.sqltypes.VARCHAR,
            "language": vane.sqltypes.VARCHAR,
            "chunk_id": vane.sqltypes.BIGINT,
            "text": vane.sqltypes.VARCHAR,
        },
        batch_size=args.batch_size,
    )
    chunks_table = collect_relation(chunks)
    if chunks_table.num_rows == 0:
        raise RuntimeError("No text chunks were produced.")

    embedded_table = None
    if not args.skip_embeddings:
        embedded_only = embed_text(
            chunks,
            "text",
            provider="transformers",
            model=args.embedding_model_id,
            output_column="embedding",
            max_chunk_chars=args.max_chunk_chars,
            batch_size=args.embedding_batch_size,
        )
        embedded_table = append_embedding(chunks_table, embedded_only)

    output_dir = Path(args.output_dir)
    save_outputs(
        pages_table=pages_table,
        chunks_table=chunks_table,
        embedded_table=embedded_table,
        output_dir=output_dir,
    )

    print(f"\nSource records scanned: {collect_relation(rel).num_rows}")
    print(f"Decoded English pages: {pages_table.num_rows}")
    print(f"Text chunks: {chunks_table.num_rows}")
    if embedded_table is not None:
        first = embedded_table.to_pylist()[0]
        print(f"Embedding dim: {embedding_dims(first['embedding'])}")
    else:
        print("Embeddings: skipped")
    print(f"Output directory: {output_dir}")

    preview_rows = []
    for row in (embedded_table if embedded_table is not None else chunks_table).to_pylist():
        preview_row = {
            "record_id": row["record_id"],
            "target_uri": row["target_uri"],
            "chunk_id": row["chunk_id"],
            "text": row["text"],
        }
        if embedded_table is not None:
            preview_row["embedding_dim"] = embedding_dims(row["embedding"])
        preview_rows.append(preview_row)
    preview_schema = {
        "record_id": "VARCHAR",
        "target_uri": "VARCHAR",
        "chunk_id": "BIGINT",
        "text": "VARCHAR",
    }
    select_embedding = ""
    if embedded_table is not None:
        preview_schema["embedding_dim"] = "BIGINT"
        select_embedding = ", embedding_dim"
    preview_rel = relation_from_rows(conn, preview_rows, preview_schema)
    preview_rel.query(
        "chunks",
        f"""
        select
            left(record_id, 36) as record_id,
            left(target_uri, 52) as target_uri,
            chunk_id,
            left(text, 88) as text
            {select_embedding}
        from chunks
        order by record_id, chunk_id
        limit {int(args.preview_rows)}
        """,
    ).show(max_width=180)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Common Crawl text chunks and embeddings with Vane.",
    )
    parser.add_argument(
        "--source",
        choices=["sample", "wet-file", "wet-url"],
        default="sample",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--wet-path", default="")
    parser.add_argument("--wet-url", default="")
    parser.add_argument("--language", default="eng")
    parser.add_argument("--max-doc-chars", type=int, default=1000)
    parser.add_argument("--max-chunk-chars", type=int, default=1024)
    parser.add_argument("--embedding-model-id", default=DEFAULT_EMBEDDING_MODEL_ID)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1 so embeddings load only from local cache.",
    )
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--preview-rows", type=int, default=10)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
