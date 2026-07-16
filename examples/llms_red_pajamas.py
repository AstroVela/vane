#!/usr/bin/env python3
"""Semantic search over Red Pajamas StackExchange text with Vane.

This example adapts Daft's Red Pajamas LLM tutorial:
https://docs.daft.ai/en/stable/examples/llms-red-pajamas/

The Daft tutorial reads a StackExchange slice from Red Pajamas, embeds the
question text with SentenceTransformers, then uses semantic search to match
low-scoring questions to related high-scoring questions.

This Vane version keeps the same workflow:

1. Load sample StackExchange-style questions, or the same S3 JSONL dataset.
2. Embed question text with ``vane.ai.embed_text``.
3. Compute cosine similarity between low-score and high-score questions.
4. Print and optionally save the top matches.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

import vane
from vane.ai import embed_text

DEFAULT_REDJAMA_PATH = "s3://daft-oss-public-data/redpajama-1t-sample/stackexchange_sample.jsonl"
DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


SAMPLE_ROWS = [
    {
        "id": 0,
        "text": (
            "How can I speed up a SQL query that scans a large Parquet dataset and filters on a timestamp column?"
        ),
        "url": "https://example.com/questions/0",
        "question_score": 0,
    },
    {
        "id": 1,
        "text": (
            "What is the best way to optimize analytical SQL over partitioned Parquet files with predicate pushdown?"
        ),
        "url": "https://example.com/questions/1",
        "question_score": 18,
    },
    {
        "id": 2,
        "text": ("Why does my Python model inference loop run slowly when I process one image at a time on the GPU?"),
        "url": "https://example.com/questions/2",
        "question_score": 1,
    },
    {
        "id": 3,
        "text": ("How do I batch image inference requests efficiently so the GPU is kept busy?"),
        "url": "https://example.com/questions/3",
        "question_score": 22,
    },
    {
        "id": 4,
        "text": ("How can I inspect the schema of a nested JSON field in a data lake before loading it into a table?"),
        "url": "https://example.com/questions/4",
        "question_score": 2,
    },
    {
        "id": 5,
        "text": (
            "How can nested JSON logs be converted into typed columns for analytics without losing optional fields?"
        ),
        "url": "https://example.com/questions/5",
        "question_score": 15,
    },
]


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sample_relation(conn: Any, limit: int) -> Any:
    rows = [SAMPLE_ROWS[i % len(SAMPLE_ROWS)] for i in range(limit)]
    ids = list(range(len(rows)))
    table = pa.table(
        {
            "id": pa.array(ids, type=pa.int64()),
            "text": pa.array([row["text"] for row in rows], type=pa.string()),
            "url": pa.array([row["url"] for row in rows], type=pa.string()),
            "question_score": pa.array(
                [row["question_score"] for row in rows],
                type=pa.int64(),
            ),
        }
    )
    return conn.from_arrow(table)


def load_redpajama_relation(conn: Any, path: str, limit: int) -> Any:
    try:
        conn.execute("INSTALL httpfs")
        conn.execute("LOAD httpfs")
    except Exception:
        pass
    try:
        conn.execute("SET s3_region='us-west-2'")
        conn.execute("SET s3_url_style='path'")
    except Exception:
        pass

    path_sql = sql_literal(path)
    return conn.sql(
        f"""
        with raw as (
            select
                text,
                to_json(meta) as meta_json
            from read_json_auto({path_sql}, maximum_object_size=16777216)
            where text is not null
        ),
        parsed as (
            select
                row_number() over () - 1 as id,
                text,
                coalesce(json_extract_string(meta_json, '$.url'), '') as url,
                try_cast(
                    json_extract_string(meta_json, '$.question_score') as bigint
                ) as question_score
            from raw
        )
        select id, text, url, question_score
        from parsed
        where question_score is not null
        limit {int(limit)}
        """
    )


def append_embedding(conn: Any, rel: Any, embedding_rel: Any) -> Any:
    base = rel.to_arrow_table()
    embeddings = embedding_rel.to_arrow_table()
    if base.num_rows != embeddings.num_rows:
        raise RuntimeError(f"Embedding row count mismatch: {embeddings.num_rows} vs {base.num_rows}")
    if "embedding" not in embeddings.column_names:
        raise RuntimeError("Embedding output column was not returned.")
    return conn.from_arrow(base.append_column("embedding", embeddings["embedding"]))


def normalize_embedding(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(array)
    if norm == 0:
        return array
    return array / norm


def semantic_matches(
    table: pa.Table,
    *,
    query_score_max: int,
    candidate_score_min: int,
    top_k: int,
) -> list[dict[str, Any]]:
    rows = table.to_pylist()
    queries = [row for row in rows if row["question_score"] <= query_score_max]
    candidates = [row for row in rows if row["question_score"] >= candidate_score_min]

    if not queries:
        raise RuntimeError("No low-score query rows matched --query-score-max.")
    if not candidates:
        raise RuntimeError("No high-score candidate rows matched --candidate-score-min.")

    candidate_vectors = [normalize_embedding(candidate["embedding"]) for candidate in candidates]
    results: list[dict[str, Any]] = []

    for query in queries:
        query_vector = normalize_embedding(query["embedding"])
        scored = [
            (float(np.dot(query_vector, candidate_vector)), candidate)
            for candidate, candidate_vector in zip(
                candidates,
                candidate_vectors,
                strict=True,
            )
            if candidate["id"] != query["id"]
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        for rank, (similarity, candidate) in enumerate(scored[:top_k], start=1):
            results.append(
                {
                    "query_id": query["id"],
                    "query_score": query["question_score"],
                    "query_text": query["text"],
                    "match_rank": rank,
                    "match_id": candidate["id"],
                    "match_score": candidate["question_score"],
                    "similarity": similarity,
                    "match_text": candidate["text"],
                    "match_url": candidate["url"],
                }
            )
    if not results:
        raise RuntimeError("No semantic matches were produced; check score thresholds.")
    return results


def save_matches(matches: list[dict[str, Any]], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(matches[0].keys()))
        writer.writeheader()
        writer.writerows(matches)


def run(args: argparse.Namespace) -> None:
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1.")
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1.")

    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    if args.runner:
        vane.configure(runner=args.runner)

    conn = vane.connect()
    rel = (
        sample_relation(conn, args.limit)
        if args.source == "sample"
        else load_redpajama_relation(conn, args.redpajama_path, args.limit)
    )

    if args.max_text_chars:
        rel = rel.query(
            "questions",
            f"""
            select
                id,
                substr(text, 1, {int(args.max_text_chars)}) as text,
                url,
                question_score
            from questions
            """,
        )

    embedded_only = embed_text(
        rel,
        "text",
        provider="transformers",
        model=args.model_id,
        output_column="embedding",
        execution_backend=args.execution_backend,
        max_chunk_chars=args.max_chunk_chars,
        batch_size=args.batch_size,
    )
    embedded = append_embedding(conn, rel, embedded_only)
    embedded_table = embedded.to_arrow_table()

    matches = semantic_matches(
        embedded_table,
        query_score_max=args.query_score_max,
        candidate_score_min=args.candidate_score_min,
        top_k=args.top_k,
    )

    if args.output_csv:
        save_matches(matches, args.output_csv)

    matches_rel = conn.from_arrow(
        pa.table(
            {
                "query_id": pa.array([row["query_id"] for row in matches], pa.int64()),
                "query_score": pa.array(
                    [row["query_score"] for row in matches],
                    pa.int64(),
                ),
                "query_text": pa.array(
                    [row["query_text"] for row in matches],
                    pa.string(),
                ),
                "match_rank": pa.array(
                    [row["match_rank"] for row in matches],
                    pa.int64(),
                ),
                "match_id": pa.array([row["match_id"] for row in matches], pa.int64()),
                "match_score": pa.array(
                    [row["match_score"] for row in matches],
                    pa.int64(),
                ),
                "similarity": pa.array(
                    [row["similarity"] for row in matches],
                    pa.float64(),
                ),
                "match_text": pa.array(
                    [row["match_text"] for row in matches],
                    pa.string(),
                ),
                "match_url": pa.array(
                    [row["match_url"] for row in matches],
                    pa.string(),
                ),
            }
        )
    )

    print(f"\nEmbedded rows: {embedded_table.num_rows}")
    print(f"Matches: {len(matches)}")
    if args.output_csv:
        print(f"Output CSV: {args.output_csv}")
    matches_rel.query(
        "matches",
        """
        select
            query_id,
            query_score,
            left(query_text, 72) as query_text,
            match_rank,
            match_id,
            match_score,
            round(similarity, 4) as similarity,
            left(match_text, 72) as match_text
        from matches
        order by query_id, match_rank
        """,
    ).show(max_width=180)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed Red Pajamas StackExchange text and find semantic matches.",
    )
    parser.add_argument("--source", choices=["sample", "redpajama"], default="sample")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--redpajama-path", default=DEFAULT_REDJAMA_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1 so embeddings load only from local cache.",
    )
    parser.add_argument("--query-score-max", type=int, default=2)
    parser.add_argument("--candidate-score-min", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-text-chars", type=int, default=1024)
    parser.add_argument("--max-chunk-chars", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--execution-backend",
        choices=["local", "ray_task", "ray_actor"],
        default="local",
    )
    parser.add_argument("--runner", choices=["", "ray"], default="")
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
