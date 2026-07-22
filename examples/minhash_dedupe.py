#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""MinHash text deduplication with Vane.

This example adapts Daft's Common Crawl MinHash tutorial:
https://docs.daft.ai/en/stable/examples/minhash-dedupe/

The Daft tutorial extracts text blocks from web pages, normalizes text,
computes MinHash signatures, applies LSH banding, finds connected components,
and keeps one representative per duplicate component.

This Vane version keeps the same shape while staying dependency-light:

1. Load sample text blocks, local CSV text, or local HTML files.
2. Normalize and MinHash text blocks with a batch UDF.
3. Generate LSH candidate pairs.
4. Build connected components and write deduped outputs.

The default source is a small built-in sample, so the example can run without
Common Crawl or extra Python packages.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import html
import json
import random
import re
import unicodedata
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import pyarrow as pa

import vane

DEFAULT_OUTPUT_DIR = Path("examples/output/minhash_dedupe")
HASH_PRIME = 2_305_843_009_213_693_951


SAMPLE_ROWS = [
    {
        "block_id": "home-0",
        "block": (
            "Vane is a fast analytical engine for local and distributed AI "
            "pipelines. It can read tables, call models in batches, and write "
            "structured outputs for downstream systems."
        ),
    },
    {
        "block_id": "home-1",
        "block": (
            "Vane is a fast analytical engine for local and distributed AI "
            "pipelines. It can read tables, call models in batches, and write "
            "structured outputs for downstream systems!"
        ),
    },
    {
        "block_id": "home-2",
        "block": (
            "VANE is a fast analytical engine for local and distributed AI "
            "pipelines; it can read tables, call models in batches, and write "
            "structured outputs for downstream systems."
        ),
    },
    {
        "block_id": "image-0",
        "block": (
            "The image generation workflow reads prompts, calls a diffusion "
            "model in batches, and saves generated images with metadata."
        ),
    },
    {
        "block_id": "image-1",
        "block": (
            "The image generation workflow reads prompts, calls a diffusion "
            "model in batches, and saves generated images with metadata."
        ),
    },
    {
        "block_id": "voice-0",
        "block": (
            "Voice analytics turns audio recordings into timestamped transcript "
            "segments, summaries, and searchable embeddings."
        ),
    },
    {
        "block_id": "voice-1",
        "block": (
            "Voice analytics turns audio recordings into timestamped transcript "
            "segments, summaries, and searchable embeddings."
        ),
    },
    {
        "block_id": "unique-0",
        "block": (
            "A release checklist should keep license notes, README updates, "
            "package metadata, and examples separate enough to review safely."
        ),
    },
    {
        "block_id": "boilerplate-0",
        "block": "Sign up for updates.",
    },
    {
        "block_id": "boilerplate-1",
        "block": "Sign up for updates!",
    },
]


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def relation_from_rows(
    conn: Any,
    rows: list[dict[str, Any]],
    schema: dict[str, str] | None = None,
) -> Any:
    """Build a VALUES relation whose data can travel with a Ray plan."""
    if not rows:
        raise RuntimeError("Cannot create a relation from zero rows.")
    columns = list(schema or rows[0])
    constant = vane.ConstantExpression
    raw = conn.values(
        *(tuple(constant(row[column]) for column in columns) for row in rows),
    )
    projections = []
    for source, column in zip(raw.columns, columns, strict=True):
        expression = quote_ident(source)
        if schema is not None:
            expression += f"::{schema[column]}"
        projections.append(f"{expression} AS {quote_ident(column)}")
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


def sample_relation(conn: Any, limit: int) -> Any:
    rows = [SAMPLE_ROWS[i % len(SAMPLE_ROWS)] for i in range(limit)]
    input_rows = [
        {
            "node_id": i,
            "block_id": (
                row["block_id"] if i < len(SAMPLE_ROWS) else f"{row['block_id']}-copy-{i // len(SAMPLE_ROWS)}"
            ),
            "block": row["block"],
        }
        for i, row in enumerate(rows)
    ]
    return relation_from_rows(
        conn,
        input_rows,
        {"node_id": "BIGINT", "block_id": "VARCHAR", "block": "VARCHAR"},
    )


def csv_relation(
    conn: Any,
    *,
    csv_path: str,
    text_column: str,
    id_column: str,
    limit: int,
    min_block_chars: int,
) -> Any:
    text_expr = quote_ident(text_column)
    id_expr = (
        f"coalesce(cast({quote_ident(id_column)} as varchar), 'row-' || cast(node_id as varchar))"
        if id_column
        else "'row-' || cast(node_id as varchar)"
    )
    return conn.sql(
        f"""
        with raw as (
            select *
            from read_csv_auto({sql_literal(csv_path)})
        ),
        filtered as (
            select
                row_number() over () - 1 as node_id,
                *
            from raw
            where {text_expr} is not null
              and length(cast({text_expr} as varchar)) >= {int(min_block_chars)}
        )
        select
            node_id,
            {id_expr} as block_id,
            cast({text_expr} as varchar) as block
        from filtered
        limit {int(limit)}
        """
    )


def extract_html_blocks(raw_html: str, min_block_chars: int) -> list[str]:
    cleaned = re.sub(
        r"(?is)<(script|style|noscript).*?</\1>",
        " ",
        raw_html,
    )
    cleaned = re.sub(
        r"(?i)</?(p|h[1-6]|li|article|main|section|div|blockquote|pre|tr|br)\b[^>]*>",
        "\n",
        cleaned,
    )
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    blocks = []
    for line in cleaned.splitlines():
        block = re.sub(r"\s+", " ", line).strip()
        if len(block) >= min_block_chars:
            blocks.append(block)
    return blocks


def html_glob_relation(
    conn: Any,
    *,
    html_glob: str,
    limit: int,
    min_block_chars: int,
) -> Any:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(html_glob)):
        raw_html = Path(path).read_text(encoding="utf-8", errors="ignore")
        for block_idx, block in enumerate(extract_html_blocks(raw_html, min_block_chars)):
            rows.append(
                {
                    "node_id": len(rows),
                    "block_id": f"{path}#{block_idx}",
                    "block": block,
                }
            )
            if len(rows) >= limit:
                return relation_from_rows(
                    conn,
                    rows,
                    {"node_id": "BIGINT", "block_id": "VARCHAR", "block": "VARCHAR"},
                )
    if not rows:
        raise RuntimeError(f"No text blocks matched --html-glob={html_glob!r}.")
    return relation_from_rows(
        conn,
        rows,
        {"node_id": "BIGINT", "block_id": "VARCHAR", "block": "VARCHAR"},
    )


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def word_shingles(normalized: str, ngram_size: int) -> list[str]:
    tokens = normalized.split()
    if not tokens:
        return []
    if len(tokens) <= ngram_size:
        return [" ".join(tokens)]
    return [" ".join(tokens[i : i + ngram_size]) for i in range(len(tokens) - ngram_size + 1)]


def stable_hash_u64(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") % HASH_PRIME


def permutation_coefficients(num_hashes: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [
        (
            rng.randrange(1, HASH_PRIME - 1),
            rng.randrange(0, HASH_PRIME - 1),
        )
        for _ in range(num_hashes)
    ]


def minhash_signature(
    shingles: list[str],
    coefficients: list[tuple[int, int]],
) -> list[int]:
    if not shingles:
        return [0] * len(coefficients)

    values = [stable_hash_u64(shingle) for shingle in set(shingles)]
    signature = [HASH_PRIME] * len(coefficients)
    for hashed in values:
        for i, (a, b) in enumerate(coefficients):
            candidate = (a * hashed + b) % HASH_PRIME
            if candidate < signature[i]:
                signature[i] = candidate
    return signature


class NormalizeMinHashBatch:
    """Batch UDF that normalizes text and computes MinHash signatures."""

    def __init__(self, *, num_hashes: int, ngram_size: int, seed: int):
        self.num_hashes = num_hashes
        self.ngram_size = ngram_size
        self.coefficients = permutation_coefficients(num_hashes, seed)

    def __call__(self, batch: pa.Table) -> pa.Table:
        node_ids = batch["node_id"].to_pylist()
        block_ids = batch["block_id"].to_pylist()
        blocks = [str(value or "") for value in batch["block"].to_pylist()]

        normalized_values = []
        minhash_values = []
        shingle_values = []
        for block in blocks:
            normalized = normalize_text(block)
            shingles = word_shingles(normalized, self.ngram_size)
            normalized_values.append(normalized)
            shingle_values.append(json.dumps(shingles, ensure_ascii=False))
            minhash_values.append(
                json.dumps(
                    minhash_signature(shingles, self.coefficients),
                    separators=(",", ":"),
                )
            )

        return pa.table(
            {
                "node_id": pa.array(node_ids, type=pa.int64()),
                "block_id": pa.array(block_ids, type=pa.string()),
                "block": pa.array(blocks, type=pa.string()),
                "content_normalized": pa.array(
                    normalized_values,
                    type=pa.string(),
                ),
                "minhashes_json": pa.array(minhash_values, type=pa.string()),
                "shingles_json": pa.array(shingle_values, type=pa.string()),
            }
        )


def integrate_probability(
    *,
    threshold: float,
    bands: int,
    rows_per_band: int,
    false_positive: bool,
    steps: int = 512,
) -> float:
    start, end = (0.0, threshold) if false_positive else (threshold, 1.0)
    if end <= start:
        return 0.0
    width = (end - start) / steps
    area = 0.0
    for i in range(steps):
        s = start + (i + 0.5) * width
        probability = 1.0 - (1.0 - s**rows_per_band) ** bands
        y = probability if false_positive else 1.0 - probability
        area += y * width
    return area


def optimal_lsh_params(
    threshold: float,
    num_hashes: int,
    *,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
) -> tuple[int, int]:
    best_error = float("inf")
    best = (1, num_hashes)
    for bands in range(1, num_hashes + 1):
        if num_hashes % bands != 0:
            continue
        rows_per_band = num_hashes // bands
        fp = integrate_probability(
            threshold=threshold,
            bands=bands,
            rows_per_band=rows_per_band,
            false_positive=True,
        )
        fn = integrate_probability(
            threshold=threshold,
            bands=bands,
            rows_per_band=rows_per_band,
            false_positive=False,
        )
        error = fp * false_positive_weight + fn * false_negative_weight
        if error < best_error:
            best_error = error
            best = (bands, rows_per_band)
    return best


def band_key(values: list[int]) -> str:
    payload = ",".join(str(value) for value in values)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def lsh_candidates(
    rows: list[dict[str, Any]],
    *,
    bands: int,
    rows_per_band: int,
    threshold: float,
    exact_jaccard: bool,
    max_bucket_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[tuple[int, str], set[int]] = defaultdict(set)
    minhash_by_node: dict[int, list[int]] = {}
    shingles_by_node: dict[int, set[str]] = {}
    block_by_node: dict[int, str] = {}

    for row in rows:
        node_id = int(row["node_id"])
        signature = [int(value) for value in json.loads(row["minhashes_json"])]
        minhash_by_node[node_id] = signature
        shingles_by_node[node_id] = set(json.loads(row["shingles_json"]))
        block_by_node[node_id] = str(row["block_id"])

        for band in range(bands):
            start = band * rows_per_band
            end = start + rows_per_band
            buckets[(band, band_key(signature[start:end]))].add(node_id)

    raw_pairs: set[tuple[int, int]] = set()
    bucket_rows = []
    for (band, digest), members in buckets.items():
        if len(members) < 2:
            continue
        nodes = sorted(members)
        bucket_rows.append(
            {
                "band": band,
                "bucket_hash": digest,
                "member_count": len(nodes),
                "members": "|".join(str(node) for node in nodes),
            }
        )
        if len(nodes) > max_bucket_size:
            rep = nodes[0]
            raw_pairs.update((rep, node) for node in nodes[1:])
        else:
            raw_pairs.update(combinations(nodes, 2))

    candidate_rows = []
    for u, v in sorted(raw_pairs):
        score = jaccard(shingles_by_node[u], shingles_by_node[v])
        if exact_jaccard and score < threshold:
            continue
        candidate_rows.append(
            {
                "u": u,
                "v": v,
                "u_block_id": block_by_node[u],
                "v_block_id": block_by_node[v],
                "jaccard": score,
            }
        )
    return candidate_rows, bucket_rows


class UnionFind:
    def __init__(self, nodes: list[int]):
        self.parent = {node: node for node in nodes}

    def find(self, node: int) -> int:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def build_components(
    rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    nodes = [int(row["node_id"]) for row in rows]
    uf = UnionFind(nodes)
    for pair in candidate_rows:
        uf.union(int(pair["u"]), int(pair["v"]))

    row_by_node = {int(row["node_id"]): row for row in rows}
    members_by_rep: dict[int, list[int]] = defaultdict(list)
    for node in nodes:
        members_by_rep[uf.find(node)].append(node)

    annotated_rows = []
    for node in nodes:
        rep = uf.find(node)
        row = row_by_node[node]
        rep_row = row_by_node[rep]
        annotated_rows.append(
            {
                "node_id": node,
                "block_id": row["block_id"],
                "component_node_id": rep,
                "component_block_id": rep_row["block_id"],
                "is_duplicate": node != rep,
                "block": row["block"],
                "content_normalized": row["content_normalized"],
            }
        )

    cluster_rows = []
    for rep, members in sorted(members_by_rep.items()):
        if len(members) < 2:
            continue
        rep_row = row_by_node[rep]
        cluster_rows.append(
            {
                "component_node_id": rep,
                "component_block_id": rep_row["block_id"],
                "member_count": len(members),
                "member_node_ids": "|".join(str(node) for node in sorted(members)),
                "member_block_ids": "|".join(str(row_by_node[node]["block_id"]) for node in sorted(members)),
                "representative_text": str(rep_row["block"])[:360],
            }
        )

    kept_rows = [row for row in annotated_rows if not row["is_duplicate"]]
    duplicate_rows = [row for row in annotated_rows if row["is_duplicate"]]
    return annotated_rows, kept_rows, duplicate_rows, cluster_rows


def save_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(
    *,
    annotated_rows: list[dict[str, Any]],
    kept_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    common_fields = [
        "node_id",
        "block_id",
        "component_node_id",
        "component_block_id",
        "is_duplicate",
        "block",
        "content_normalized",
    ]
    save_csv(annotated_rows, output_dir / "annotated.csv", common_fields)
    save_csv(kept_rows, output_dir / "deduped.csv", common_fields)
    save_csv(duplicate_rows, output_dir / "duplicates.csv", common_fields)
    save_csv(
        cluster_rows,
        output_dir / "clusters.csv",
        [
            "component_node_id",
            "component_block_id",
            "member_count",
            "member_node_ids",
            "member_block_ids",
            "representative_text",
        ],
    )
    save_csv(
        candidate_rows,
        output_dir / "candidate_pairs.csv",
        ["u", "v", "u_block_id", "v_block_id", "jaccard"],
    )
    save_csv(
        bucket_rows,
        output_dir / "lsh_buckets.csv",
        ["band", "bucket_hash", "member_count", "members"],
    )


def run(args: argparse.Namespace) -> None:
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1.")
    if args.num_hashes < 1:
        raise SystemExit("--num-hashes must be at least 1.")
    if args.ngram_size < 1:
        raise SystemExit("--ngram-size must be at least 1.")
    if not 0.0 < args.threshold <= 1.0:
        raise SystemExit("--threshold must be in (0, 1].")

    conn = vane.connect()
    if args.source == "sample":
        rel = sample_relation(conn, args.limit)
    elif args.source == "csv":
        if not args.csv_path:
            raise SystemExit("--csv-path is required when --source csv.")
        rel = csv_relation(
            conn,
            csv_path=args.csv_path,
            text_column=args.text_column,
            id_column=args.id_column,
            limit=args.limit,
            min_block_chars=args.min_block_chars,
        )
    else:
        rel = html_glob_relation(
            conn,
            html_glob=args.html_glob,
            limit=args.limit,
            min_block_chars=args.min_block_chars,
        )

    minhasher = NormalizeMinHashBatch(
        num_hashes=args.num_hashes,
        ngram_size=args.ngram_size,
        seed=args.seed,
    )
    processed = rel.map_batches(
        minhasher.__call__,
        schema={
            "node_id": vane.sqltypes.BIGINT,
            "block_id": vane.sqltypes.VARCHAR,
            "block": vane.sqltypes.VARCHAR,
            "content_normalized": vane.sqltypes.VARCHAR,
            "minhashes_json": vane.sqltypes.VARCHAR,
            "shingles_json": vane.sqltypes.VARCHAR,
        },
        batch_size=args.batch_size,
    )
    processed_table = collect_relation(processed)
    processed_rows = processed_table.to_pylist()

    if args.bands or args.rows_per_band:
        if not args.bands or not args.rows_per_band:
            raise SystemExit("--bands and --rows-per-band must be provided together.")
        bands = int(args.bands)
        rows_per_band = int(args.rows_per_band)
        if bands * rows_per_band != args.num_hashes:
            raise SystemExit("--bands * --rows-per-band must equal --num-hashes.")
    else:
        bands, rows_per_band = optimal_lsh_params(args.threshold, args.num_hashes)

    candidate_rows, bucket_rows = lsh_candidates(
        processed_rows,
        bands=bands,
        rows_per_band=rows_per_band,
        threshold=args.threshold,
        exact_jaccard=not args.skip_exact_jaccard,
        max_bucket_size=args.max_bucket_size,
    )
    annotated_rows, kept_rows, duplicate_rows, cluster_rows = build_components(
        processed_rows,
        candidate_rows,
    )

    output_dir = Path(args.output_dir)
    save_outputs(
        annotated_rows=annotated_rows,
        kept_rows=kept_rows,
        duplicate_rows=duplicate_rows,
        cluster_rows=cluster_rows,
        candidate_rows=candidate_rows,
        bucket_rows=bucket_rows,
        output_dir=output_dir,
    )

    kept_pct = 100.0 * len(kept_rows) / max(1, len(processed_rows))
    print(f"\nInput rows: {len(processed_rows)}")
    print(f"LSH bands: {bands}")
    print(f"Rows per band: {rows_per_band}")
    print(f"LSH buckets with collisions: {len(bucket_rows)}")
    print(f"Candidate duplicate pairs: {len(candidate_rows)}")
    print(f"Duplicate rows removed: {len(duplicate_rows)}")
    print(f"Rows kept: {len(kept_rows)} ({kept_pct:.2f}%)")
    print(f"Output directory: {output_dir}")

    if cluster_rows:
        relation_from_rows(conn, cluster_rows).query(
            "clusters",
            """
            select
                component_block_id,
                member_count,
                member_block_ids,
                left(representative_text, 96) as representative_text
            from clusters
            order by member_count desc, component_block_id
            """,
        ).show(max_width=180)

    if candidate_rows:
        relation_from_rows(conn, candidate_rows).query(
            "pairs",
            """
            select
                u_block_id,
                v_block_id,
                round(jaccard, 4) as jaccard
            from pairs
            order by jaccard desc, u_block_id, v_block_id
            """,
        ).show(max_width=120)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate text blocks with MinHash and LSH in Vane.",
    )
    parser.add_argument("--source", choices=["sample", "csv", "html-glob"], default="sample")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--csv-path", default="")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="")
    parser.add_argument("--html-glob", default="examples/data/html/*.html")
    parser.add_argument("--min-block-chars", type=int, default=12)
    parser.add_argument("--num-hashes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ngram-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--bands", type=int, default=0)
    parser.add_argument("--rows-per-band", type=int, default=0)
    parser.add_argument(
        "--skip-exact-jaccard",
        action="store_true",
        help="Use raw LSH candidates without exact shingle Jaccard verification.",
    )
    parser.add_argument("--max-bucket-size", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
