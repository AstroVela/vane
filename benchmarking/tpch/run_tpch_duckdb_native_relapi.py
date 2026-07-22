#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run DuckDB TPC-H benchmarks using the NativeRunner (Relation API).

This is the "native" counterpart of `run_tpch_duckdb_ray_relapi.py`.

Key differences vs `run_tpch_duckdb.py`:
  - Uses the Relation API (`con.sql(sql)`) instead of `con.execute(sql)`.
  - Executes via `vane.runners.NativeRunner` by iterating
    `runner.run_iter_tables(relation)`, which yields PyArrow tables.
  - Each query runs in a subprocess (spawn) to enforce a hard timeout and to
    isolate hangs/crashes.

This does NOT use Ray and does not use the distributed planner.

Usage:
    # Run a single query with a 120s timeout
    python benchmarking/tpch/run_tpch_duckdb_native_relapi.py \
        --parquet_folder data/tpch10 \
        --question 9 \
        --timeout 120

    # Run multiple queries
    python benchmarking/tpch/run_tpch_duckdb_native_relapi.py \
        --parquet_folder data/tpch10 \
        --questions 1,6,9 \
        --threads 8
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import statistics
import sys
import time

TABLES = ["nation", "region", "supplier", "customer", "part", "partsupp", "orders", "lineitem"]
QUERY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries")


def _setup_views(con, parquet_folder: str) -> None:
    for table in TABLES:
        parquet_path = os.path.join(parquet_folder, table, "*.parquet")
        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{parquet_path}')")


def _run_query_native_in_subprocess(
    queue: multiprocessing.Queue,
    parquet_folder: str,
    qnum: int,
    threads: int | None,
    results_buffer_size: int | None,
    child_verbose: bool,
) -> None:
    """Subprocess target: run a single query via NativeRunner and return row count.

    Important: set env vars BEFORE importing Vane so editable-build chatter and
    runner selection take effect.
    """
    try:
        # Keep benchmark output clean by default: the runner stack prints a few
        # debug lines on import in this repo. Enable `--child-verbose` to see
        # all child-process stdout/stderr.
        if not child_verbose:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")

        # Silence scikit-build-core editable rebuild output by default.
        # Users can override by exporting SKBUILD_EDITABLE_VERBOSE=1.
        os.environ.setdefault("SKBUILD_EDITABLE_VERBOSE", "0")

        # Force runner type. This does not change DuckDB's normal local execution
        # by itself; we explicitly call into vane.runners below.
        os.environ["VANE_RUNNER"] = "native"

        import vane
        import vane.runners

        # Instantiate NativeRunner with optional thread hint.
        # Note: set_runner_native() can only be called once per process.
        runner = vane.runners.set_runner_native(threads)
        if getattr(runner, "name", None) != "native":
            raise RuntimeError(f"Expected native runner but got {getattr(runner, 'name', None)!r}")

        con = vane.connect()
        if threads:
            con.execute(f"SET threads={threads}")

        _setup_views(con, parquet_folder)

        query_file = os.path.join(QUERY_DIR, f"{qnum:02d}.sql")
        with open(query_file) as f:
            sql = f.read()

        rel = con.sql(sql)

        t0 = time.time()
        row_count = 0
        for table in runner.run_iter_tables(rel, results_buffer_size):
            row_count += table.num_rows
        elapsed = time.time() - t0

        con.close()
        queue.put(("OK", elapsed, row_count))
    except Exception as e:
        queue.put(("FAIL", 0.0, str(e)[:200]))


def run_query_native_with_timeout(
    parquet_folder: str,
    qnum: int,
    threads: int | None,
    results_buffer_size: int | None,
    timeout: int,
    child_verbose: bool,
) -> tuple[str, float, int | str]:
    """Run a query in a subprocess with timeout. Returns (status, elapsed, rows_or_error)."""
    queue: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_run_query_native_in_subprocess,
        args=(queue, parquet_folder, qnum, threads, results_buffer_size, child_verbose),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        return "TIMEOUT", float(timeout), f"subprocess killed after {timeout}s"

    if not queue.empty():
        return queue.get()

    return "FAIL", 0.0, f"subprocess exited without result (exitcode={proc.exitcode})"


def verify_row_counts(parquet_folder: str, qnum: int, threads: int | None) -> int:
    """Run query locally (no runner) to get a reference row count."""
    import vane

    con = vane.connect()
    if threads:
        con.execute(f"SET threads={threads}")

    _setup_views(con, parquet_folder)

    query_file = os.path.join(QUERY_DIR, f"{qnum:02d}.sql")
    with open(query_file) as f:
        sql = f.read()

    result = con.execute(sql).fetchall()
    con.close()
    return len(result)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Run TPC-H benchmarks using DuckDB NativeRunner (Relation API) with per-query timeouts"
    )
    parser.add_argument("--parquet_folder", required=True, help="Path to parquet data root (e.g. data/tpch10)")
    parser.add_argument(
        "--questions",
        "--question",
        default=None,
        help="Comma-separated query numbers (default: all 1-22). `--question` is an alias.",
    )
    parser.add_argument("--threads", default=None, type=int, help="Number of DuckDB threads (default: all cores)")
    parser.add_argument(
        "--timeout", default=300, type=int, help="Per-query subprocess timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--results-buffer-size",
        default=None,
        type=int,
        help="Optional results buffer size passed to runner.run_iter_tables() (default: None)",
    )
    parser.add_argument(
        "--child-verbose",
        action="store_true",
        help="Do not suppress child-process stdout/stderr (useful for debugging)",
    )
    parser.add_argument("--verify", action="store_true", help="Verify native runner row counts match local execution")
    parser.add_argument("--iterations", default=3, type=int, help="Number of iterations per query (default: 3)")
    args = parser.parse_args()

    # Silence scikit-build-core editable rebuild output by default for the driver
    # process too (subprocesses inherit the env).
    os.environ.setdefault("SKBUILD_EDITABLE_VERBOSE", "0")

    import vane

    print(f"DuckDB version: {vane.__duckdb_version__}")
    print("Runner: native (Relation API via vane.runners.NativeRunner)")
    print(f"  Subprocess timeout: {args.timeout}s")
    print(f"  Verify: {'enabled' if args.verify else 'disabled'}")
    print(f"  Iterations: {args.iterations}")

    if args.questions:
        questions = sorted({int(q) for q in args.questions.split(",")})
    else:
        questions = list(range(1, 23))

    # Show configured thread count (DuckDB setting, not runner internal threads).
    con = vane.connect()
    if args.threads:
        con.execute(f"SET threads={args.threads}")
    thread_count = con.execute("SELECT current_setting('threads')").fetchone()[0]
    con.close()
    print(f"  Threads: {thread_count}")

    iterations = args.iterations
    print(f"\nRunning TPC-H queries {questions} on data: {args.parquet_folder}\n")
    print(f"{'Query':<10} {'Status':<18} {'Min (s)':<12} {'Avg (s)':<12} {'Med (s)':<12} {'Rows':<12}")
    print("-" * 82)

    total_time = 0.0
    ok = []
    timed_out = []
    failed = []
    verification_mismatches: list[tuple[int, int, int]] = []

    for qnum in questions:
        iter_times = []
        last_rows = 0
        query_failed = False
        for _ in range(iterations):
            status, elapsed, detail = run_query_native_with_timeout(
                args.parquet_folder,
                qnum,
                args.threads,
                args.results_buffer_size,
                args.timeout,
                args.child_verbose,
            )
            if status == "OK":
                iter_times.append(float(elapsed))
                last_rows = detail
            elif status == "TIMEOUT":
                timed_out.append(qnum)
                failed.append(qnum)
                print(f"Q{qnum:<9} {'TIMEOUT':<18} {'—':<12} {'—':<12} {'—':<12} {detail}")
                query_failed = True
                break
            else:
                failed.append(qnum)
                print(f"Q{qnum:<9} {'FAIL':<18} {'—':<12} {'—':<12} {'—':<12} {str(detail)[:60]}")
                query_failed = True
                break

        if not query_failed and iter_times:
            min_t = min(iter_times)
            avg_t = sum(iter_times) / len(iter_times)
            med_t = statistics.median(iter_times)
            total_time += sum(iter_times)
            ok.append(qnum)
            times_str = ", ".join(f"{t:.3f}" for t in iter_times)
            print(f"Q{qnum:<9} {'OK(native)':<18} {min_t:<12.3f} {avg_t:<12.3f} {med_t:<12.3f} {last_rows:<12}")
            print(f"{'':>10} iters: [{times_str}]")

            if args.verify:
                try:
                    expected_rows = verify_row_counts(args.parquet_folder, qnum, args.threads)
                    if int(last_rows) != int(expected_rows):
                        verification_mismatches.append((qnum, int(last_rows), int(expected_rows)))
                        print(f"  VERIFY MISMATCH: native_runner={last_rows}, local={expected_rows}")
                except Exception as e:
                    print(f"  verify error: {str(e)[:120]}")

    print("-" * 82)
    print(f"\nTotal time: {total_time:.3f}s | Passed: {len(ok)}/{len(questions)} | Iterations: {iterations}")
    print(f"  OK:      {len(ok)}")
    print(f"  Timeout: {len(timed_out)}")
    print(f"  Failed:  {len(failed)}")
    if ok:
        print(f"  OK queries: {ok}")
    if timed_out:
        print(f"  Timed-out queries: {timed_out}")
    if failed:
        print(f"  Failed queries: {failed}")
    if verification_mismatches:
        print("\n  Verification mismatches:")
        for qn, got, expected in verification_mismatches:
            print(f"    Q{qn}: native_runner={got}, local={expected}")
