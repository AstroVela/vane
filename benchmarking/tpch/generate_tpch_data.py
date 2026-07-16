#!/usr/bin/env python3
"""Generate TPC-H Parquet data with per-table sharding.

Uses stock DuckDB (at /tmp/duckdb_stock) with built-in tpch extension.
Each table is exported as multiple Parquet files (~200MB each) for distributed scans.

Memory safety: uses dbgen(children=N, step=i) to generate data in N chunks,
each chunk uses ~1/N of the full memory. For SF100 on a 94GB machine with
~82GB available, children=4 keeps peak memory at ~39GB (safe).

Industry standard: 128-256MB per file, Snappy compression, row_group_size ~1M rows.

Usage:
    python3 benchmarking/tpch/generate_tpch_data.py --scale_factor 100 --output_dir data/tpch100
    python3 benchmarking/tpch/generate_tpch_data.py --scale_factor 10  --output_dir data/tpch10_sharded
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

# Use stock DuckDB for tpch extension support
sys.path.insert(0, "/tmp/duckdb_stock")
import vane

TABLES = ["lineitem", "orders", "partsupp", "customer", "part", "supplier", "nation", "region"]
TARGET_FILE_SIZE_MB = 200


def _export_table(con, table: str, table_dir: str) -> tuple[int, int, float, float]:
    """Export a single table to sharded Parquet. Returns (row_count, num_files, size_mb, time_s)."""
    os.makedirs(table_dir, exist_ok=True)
    row_count = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    if row_count == 0:
        return 0, 0, 0.0, 0.0

    t1 = time.time()
    if row_count < 100_000:
        out_path = os.path.join(table_dir, f"{table}.parquet")
        con.execute(f"""
            COPY {table} TO '{out_path}'
            (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 1000000)
        """)
    else:
        con.execute(f"""
            COPY {table} TO '{table_dir}'
            (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 1000000,
             PER_THREAD_OUTPUT TRUE, FILE_SIZE_BYTES '{TARGET_FILE_SIZE_MB}MB')
        """)
    export_time = time.time() - t1

    parquet_files = [f for f in os.listdir(table_dir) if f.endswith(".parquet")]
    total_size_mb = sum(os.path.getsize(os.path.join(table_dir, f)) / (1024 * 1024) for f in parquet_files)
    return row_count, len(parquet_files), total_size_mb, export_time


def generate_tpch(scale_factor: int, output_dir: str, threads: int | None, children: int):
    """Generate TPC-H data in `children` chunks to bound peak memory.

    Each chunk calls dbgen(sf=SF, children=N, step=i) which generates ~1/N
    of each table. Tables are exported to Parquet after each chunk and then
    dropped to free memory before the next chunk.

    Peak memory ≈ full_dataset_size / children.
    """
    print(f"DuckDB version: {vane.__duckdb_version__}")
    print(f"Scale factor: {scale_factor}")
    print(f"Output directory: {output_dir}")
    print(f"Chunks (children): {children}")
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    total_rows = dict.fromkeys(TABLES, 0)
    total_files = dict.fromkeys(TABLES, 0)
    total_size = dict.fromkeys(TABLES, 0.0)

    for step in range(children):
        print(f"\n--- Chunk {step + 1}/{children} ---")
        con = vane.connect()
        if threads:
            con.execute(f"SET threads={threads}")
        thread_count = con.execute("SELECT current_setting('threads')").fetchone()[0]
        if step == 0:
            print(f"Threads: {thread_count}")

        con.execute("LOAD tpch;")
        t_gen = time.time()
        con.execute(f"CALL dbgen(sf={scale_factor}, children={children}, step={step})")
        gen_time = time.time() - t_gen
        print(f"dbgen chunk {step} completed in {gen_time:.1f}s")

        # Show memory usage
        try:
            mem = con.execute(
                "SELECT tag, memory_usage_bytes FROM duckdb_memory() WHERE memory_usage_bytes > 0"
            ).fetchall()
            if mem:
                total_mem = sum(m[1] for m in mem)
                print(f"Memory usage: {total_mem / (1024**3):.1f} GB")
        except Exception:
            pass

        # Export each table's chunk
        for table in TABLES:
            table_dir = os.path.join(output_dir, table)
            row_count, num_files, size_mb, export_time = _export_table(con, table, table_dir)

            if row_count > 0:
                total_rows[table] += row_count
                total_files[table] += num_files
                total_size[table] += size_mb
                print(
                    f"  {table:>12}: {row_count:>12,} rows | {num_files:>3} files | "
                    f"{size_mb:>8.1f} MB | {export_time:.1f}s"
                )

        # Close connection to release all memory before next chunk
        con.close()
        gc.collect()

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  TPC-H SF{scale_factor} Generation Complete")
    print(f"{'=' * 70}")
    grand_total_size = 0.0
    grand_total_files = 0
    for table in TABLES:
        if total_rows[table] > 0:
            print(
                f"  {table:>12}: {total_rows[table]:>12,} rows | {total_files[table]:>4} files | "
                f"{total_size[table]:>8.1f} MB"
            )
            grand_total_size += total_size[table]
            grand_total_files += total_files[table]
    print(f"  {'TOTAL':>12}: {'':>12} {'':>1} | {grand_total_files:>4} files | {grand_total_size:>8.1f} MB")
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TPC-H Parquet data with sharding")
    parser.add_argument("--scale_factor", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="data/tpch100")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--children",
        type=int,
        default=4,
        help="Number of chunks to split generation into (controls peak memory). "
        "Peak memory ≈ full_size / children. Default: 4 (~39GB for SF100)",
    )
    args = parser.parse_args()

    generate_tpch(args.scale_factor, args.output_dir, args.threads, args.children)
