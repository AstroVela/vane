#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run DuckDB TPC-H benchmarks against local parquet data.

Usage:
    python benchmarking/tpch/run_tpch_duckdb.py \
        --parquet_folder data/tpch10 \
        --questions 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

# Use stable official DuckDB from /tmp/duckdb_release if available, otherwise system.
try:
    sys.path.insert(0, "/tmp/duckdb_release")
    import duckdb
except ImportError:
    sys.path.pop(0)
    import duckdb

print(f"DuckDB version: {duckdb.__version__}")

TABLES = ["nation", "region", "supplier", "customer", "part", "partsupp", "orders", "lineitem"]
QUERY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries")


def setup_views(con: duckdb.DuckDBPyConnection, parquet_folder: str):
    """Create views for each TPC-H table pointing to parquet files."""
    for table in TABLES:
        parquet_path = os.path.join(parquet_folder, table, "*.parquet")
        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{parquet_path}')")


def run_query(con: duckdb.DuckDBPyConnection, qnum: int) -> tuple[float, int]:
    """Run a single TPC-H query and return (elapsed_seconds, row_count)."""
    query_file = os.path.join(QUERY_DIR, f"{qnum:02d}.sql")
    with open(query_file) as f:
        sql = f.read()

    t0 = time.time()
    result = con.execute(sql).fetchall()
    elapsed = time.time() - t0
    return elapsed, len(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DuckDB TPC-H benchmarks")
    parser.add_argument("--parquet_folder", required=True, help="Path to parquet data root (e.g. data/tpch10)")
    parser.add_argument("--questions", default=None, help="Comma-separated query numbers (default: all 1-22)")
    parser.add_argument("--threads", default=None, type=int, help="Number of DuckDB threads (default: all cores)")
    parser.add_argument("--iterations", default=3, type=int, help="Number of iterations per query (default: 3)")
    args = parser.parse_args()

    if args.questions:
        questions = sorted({int(q) for q in args.questions.split(",")})
    else:
        questions = list(range(1, 23))

    con = duckdb.connect()
    if args.threads:
        con.execute(f"SET threads={args.threads}")

    thread_count = con.execute("SELECT current_setting('threads')").fetchone()[0]
    print(f"Threads: {thread_count}")

    setup_views(con, args.parquet_folder)

    iterations = args.iterations
    print(f"\nRunning TPC-H queries {questions} on data: {args.parquet_folder}")
    print(f"Iterations per query: {iterations}\n")
    print(f"{'Query':<10} {'Status':<10} {'Min (s)':<12} {'Avg (s)':<12} {'Med (s)':<12} {'Rows':<12}")
    print("-" * 74)

    total_time = 0.0
    failed = []
    for qnum in questions:
        iter_times = []
        rows = 0
        query_failed = False
        for _ in range(iterations):
            try:
                elapsed, rows = run_query(con, qnum)
                iter_times.append(elapsed)
            except Exception as e:
                query_failed = True
                failed.append(qnum)
                print(f"Q{qnum:<9} {'FAIL':<10} {'—':<12} {'—':<12} {'—':<12} {str(e)[:40]}")
                break
        if not query_failed and iter_times:
            min_t = min(iter_times)
            avg_t = sum(iter_times) / len(iter_times)
            med_t = statistics.median(iter_times)
            total_time += sum(iter_times)
            times_str = ", ".join(f"{t:.3f}" for t in iter_times)
            print(f"Q{qnum:<9} {'OK':<10} {min_t:<12.3f} {avg_t:<12.3f} {med_t:<12.3f} {rows:<12}")
            print(f"{'':>10} iters: [{times_str}]")

    print("-" * 74)
    print(
        f"Total: {total_time:.3f}s | Passed: {len(questions) - len(failed)}/{len(questions)} | Iterations: {iterations}"
    )
    if failed:
        print(f"Failed queries: {failed}")

    con.close()
