#!/usr/bin/env python3
"""Generate TPC-H Parquet data with parallel chunk generation.

Uses dbgen(children=N, step=i) with multiprocessing to generate chunks in parallel,
then exports to sharded Parquet files.

Each child process runs dbgen for one chunk independently, so all CPU cores
are utilized during both generation and export phases.

Usage:
    python3 benchmarking/tpch/generate_tpch_data_parallel.py --scale_factor 100 --output_dir data/tpch100
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time

TABLES = ["lineitem", "orders", "partsupp", "customer", "part", "supplier", "nation", "region"]
TARGET_FILE_SIZE_MB = 200


def _generate_chunk(args_tuple):
    """Worker function: generate one chunk and export to Parquet."""
    scale_factor, children, step, output_dir, threads = args_tuple

    sys.path.insert(0, "/tmp/duckdb_stock")
    import vane

    # Each worker uses limited threads to avoid oversubscription
    worker_threads = max(1, (threads or os.cpu_count() or 4) // children)

    con = vane.connect()
    con.execute(f"SET threads={worker_threads}")
    con.execute("LOAD tpch;")

    t0 = time.time()
    con.execute(f"CALL dbgen(sf={scale_factor}, children={children}, step={step})")
    gen_time = time.time() - t0

    # Get memory usage
    try:
        mem = con.execute(
            "SELECT SUM(memory_usage_bytes) FROM duckdb_memory() WHERE memory_usage_bytes > 0"
        ).fetchone()[0]
        mem_gb = (mem or 0) / (1024**3)
    except Exception:
        mem_gb = 0

    print(
        f"  [chunk {step}] dbgen done in {gen_time:.1f}s, memory: {mem_gb:.1f} GB, threads: {worker_threads}",
        flush=True,
    )

    # Export each table
    chunk_stats = {}
    for table in TABLES:
        table_dir = os.path.join(output_dir, table)
        os.makedirs(table_dir, exist_ok=True)

        row_count = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        if row_count == 0:
            continue

        t1 = time.time()
        if row_count < 100_000:
            # Small table: single file with chunk suffix to avoid collisions
            out_path = os.path.join(table_dir, f"{table}_chunk{step}.parquet")
            con.execute(f"""
                COPY {table} TO '{out_path}'
                (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 1000000)
            """)
        else:
            # Large table: sharded output with unique prefix per chunk
            # Use FILENAME_PATTERN to avoid file name collisions between chunks
            con.execute(f"""
                COPY {table} TO '{table_dir}'
                (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 1000000,
                 PER_THREAD_OUTPUT TRUE, FILE_SIZE_BYTES '{TARGET_FILE_SIZE_MB}MB',
                 FILENAME_PATTERN 'chunk{step}_{{i}}',
                 OVERWRITE_OR_IGNORE TRUE)
            """)
        export_time = time.time() - t1

        parquet_files = [f for f in os.listdir(table_dir) if f.endswith(".parquet") and f.startswith(f"chunk{step}")]
        if not parquet_files:
            # Single file case
            parquet_files = [f for f in os.listdir(table_dir) if f.endswith(".parquet") and f"{table}_chunk{step}" in f]
        total_size_mb = sum(os.path.getsize(os.path.join(table_dir, f)) / (1024 * 1024) for f in parquet_files)
        chunk_stats[table] = (row_count, len(parquet_files), total_size_mb, export_time)
        print(
            f"  [chunk {step}] {table:>12}: {row_count:>12,} rows | {len(parquet_files):>3} files | "
            f"{total_size_mb:>8.1f} MB | {export_time:.1f}s",
            flush=True,
        )

    con.close()
    return step, chunk_stats


def main():
    parser = argparse.ArgumentParser(description="Generate TPC-H Parquet data (parallel)")
    parser.add_argument("--scale_factor", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="data/tpch100")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--children",
        type=int,
        default=4,
        help="Number of parallel chunks. Peak memory ≈ full_size * 2/children (2 chunks run concurrently). Default: 4",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=2,
        help="Max parallel chunk workers. Default: 2 (safe for memory). Set to 1 for sequential (lowest memory).",
    )
    args = parser.parse_args()

    total_threads = args.threads or os.cpu_count() or 36
    print("DuckDB version: ", end="")
    sys.path.insert(0, "/tmp/duckdb_stock")
    import vane

    print(vane.__duckdb_version__)
    print(f"Scale factor: {args.scale_factor}")
    print(f"Output directory: {args.output_dir}")
    print(f"Chunks: {args.children}, Parallel workers: {args.parallel}")
    print(f"Total CPU threads: {total_threads}, per worker: ~{total_threads // args.children}")
    print(f"Estimated peak memory: ~{154 * args.parallel / args.children:.0f} GB (for SF100)")

    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()
    worker_args = [
        (args.scale_factor, args.children, step, args.output_dir, total_threads) for step in range(args.children)
    ]

    # Run chunks in parallel with bounded concurrency
    if args.parallel >= args.children:
        # All chunks in parallel
        with multiprocessing.Pool(args.children) as pool:
            results = pool.map(_generate_chunk, worker_args)
    elif args.parallel == 1:
        # Sequential
        results = [_generate_chunk(a) for a in worker_args]
    else:
        # Bounded parallelism
        with multiprocessing.Pool(args.parallel) as pool:
            results = pool.map(_generate_chunk, worker_args)

    # Aggregate stats
    print(f"\n{'=' * 70}")
    print(f"  TPC-H SF{args.scale_factor} Generation Complete")
    print(f"{'=' * 70}")

    grand_total_size = 0.0
    grand_total_files = 0
    for table in TABLES:
        total_rows = 0
        total_files = 0
        total_size = 0.0
        for _, chunk_stats in results:
            if table in chunk_stats:
                r, f, s, _ = chunk_stats[table]
                total_rows += r
                total_files += f
                total_size += s
        if total_rows > 0:
            # Count actual files on disk (may differ from per-chunk count)
            table_dir = os.path.join(args.output_dir, table)
            actual_files = len([f for f in os.listdir(table_dir) if f.endswith(".parquet")])
            actual_size = sum(
                os.path.getsize(os.path.join(table_dir, f)) / (1024 * 1024)
                for f in os.listdir(table_dir)
                if f.endswith(".parquet")
            )
            print(f"  {table:>12}: {total_rows:>12,} rows | {actual_files:>4} files | {actual_size:>8.1f} MB")
            grand_total_size += actual_size
            grand_total_files += actual_files

    print(f"  {'TOTAL':>12}: {'':>14} | {grand_total_files:>4} files | {grand_total_size:>8.1f} MB")
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
