#!/usr/bin/env python3
"""Run DuckDB TPC-H benchmarks on Ray with per-query timeout and local fallback.

Uses the project's built-in DuckDB distributed execution engine (Relation API + Ray runner).
Each query runs in a separate subprocess to isolate hangs. Queries that timeout on Ray
are automatically retried with local DuckDB execution.

Usage:
    # Ray runner with default 120s timeout per query
    python benchmarking/tpch/run_tpch_duckdb_ray.py \
        --parquet_folder data/tpch10 --runner ray

    # Ray runner, 60s timeout, no local fallback
    python benchmarking/tpch/run_tpch_duckdb_ray.py \
        --parquet_folder data/tpch10 --runner ray \
        --timeout 60 --no-fallback

    # Local only (same as run_tpch_duckdb.py)
    python benchmarking/tpch/run_tpch_duckdb_ray.py \
        --parquet_folder data/tpch10
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import statistics
import time

import vane

print(f"DuckDB version: {vane.__duckdb_version__}")

TABLES = ["nation", "region", "supplier", "customer", "part", "partsupp", "orders", "lineitem"]
QUERY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries")


def _run_query_in_subprocess(queue, parquet_folder, qnum, threads, use_ray, ray_address=None):
    """Target function for multiprocessing.Process. Runs a single query and puts result in queue."""
    try:
        if use_ray:
            os.environ["VANE_RUNNER"] = "ray"
            import ray

            if not ray.is_initialized():
                if ray_address:
                    ray.init(address=ray_address, ignore_reinit_error=True)
                else:
                    ray.init(ignore_reinit_error=True)

        con = vane.connect()
        if threads:
            con.execute(f"SET threads={threads}")

        for table in TABLES:
            parquet_path = os.path.join(parquet_folder, table, "*.parquet")
            con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{parquet_path}')")

        query_file = os.path.join(QUERY_DIR, f"{qnum:02d}.sql")
        with open(query_file) as f:
            sql = f.read()

        t0 = time.time()
        result = con.execute(sql).fetchall()
        elapsed = time.time() - t0
        con.close()
        queue.put(("OK", elapsed, len(result)))
    except Exception as e:
        queue.put(("FAIL", 0.0, str(e)[:80]))


def run_query_with_timeout(parquet_folder, qnum, threads, use_ray, timeout, ray_address=None):
    """Run a query in a subprocess with timeout. Returns (status, elapsed, rows_or_error)."""
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_run_query_in_subprocess,
        args=(queue, parquet_folder, qnum, threads, use_ray, ray_address),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        return "TIMEOUT", timeout, f"killed after {timeout}s"

    if not queue.empty():
        return queue.get()

    return "FAIL", 0.0, "subprocess exited without result"


def run_query_local(parquet_folder, qnum, threads):
    """Run a query directly in the current process (local DuckDB, no Ray)."""
    con = vane.connect()
    if threads:
        con.execute(f"SET threads={threads}")

    for table in TABLES:
        parquet_path = os.path.join(parquet_folder, table, "*.parquet")
        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{parquet_path}')")

    query_file = os.path.join(QUERY_DIR, f"{qnum:02d}.sql")
    with open(query_file) as f:
        sql = f.read()

    t0 = time.time()
    result = con.execute(sql).fetchall()
    elapsed = time.time() - t0
    con.close()
    return elapsed, len(result)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Run DuckDB TPC-H benchmarks (local or Ray with timeout)")
    parser.add_argument("--parquet_folder", required=True, help="Path to parquet data root (e.g. data/tpch10)")
    parser.add_argument("--questions", default=None, help="Comma-separated query numbers (default: all 1-22)")
    parser.add_argument("--threads", default=None, type=int, help="Number of DuckDB threads (default: all cores)")
    parser.add_argument(
        "--runner", default="local", choices=["local", "ray"], help="Execution mode: 'local' (default) or 'ray'"
    )
    parser.add_argument("--timeout", default=120, type=int, help="Per-query timeout in seconds (default: 120)")
    parser.add_argument(
        "--no-fallback", action="store_true", help="Disable local fallback for timed-out/failed Ray queries"
    )
    parser.add_argument("--iterations", default=3, type=int, help="Number of iterations per query (default: 3)")
    args = parser.parse_args()

    use_ray = args.runner == "ray"
    use_fallback = not args.no_fallback

    ray_address = None
    if use_ray:
        import ray

        ray.init()
        ray_address = ray.get_runtime_context().gcs_address
        print(f"Runner: ray (nodes={len(ray.nodes())}, cpus={int(ray.cluster_resources().get('CPU', 0))})")
        print(f"Ray address: {ray_address}")
        print(f"Timeout: {args.timeout}s per query | Fallback: {'enabled' if use_fallback else 'disabled'}")
    else:
        print("Runner: local")

    if args.questions:
        questions = sorted({int(q) for q in args.questions.split(",")})
    else:
        questions = list(range(1, 23))

    # Show thread count
    con = vane.connect()
    if args.threads:
        con.execute(f"SET threads={args.threads}")
    thread_count = con.execute("SELECT current_setting('threads')").fetchone()[0]
    print(f"Threads: {thread_count}")
    con.close()

    iterations = args.iterations
    print(f"\nRunning TPC-H queries {questions} on data: {args.parquet_folder}")
    print(f"Iterations per query: {iterations}\n")
    print(f"{'Query':<10} {'Status':<16} {'Min (s)':<12} {'Avg (s)':<12} {'Med (s)':<12} {'Rows':<12}")
    print("-" * 80)

    total_time = 0.0
    ray_ok = []
    fallback_ok = []
    failed = []
    timed_out = []

    for qnum in questions:
        if use_ray:
            iter_times = []
            last_rows = 0
            query_failed = False
            for _ in range(iterations):
                status, elapsed, detail = run_query_with_timeout(
                    args.parquet_folder, qnum, args.threads, True, args.timeout, ray_address
                )
                if status == "OK":
                    iter_times.append(elapsed)
                    last_rows = detail
                elif status == "TIMEOUT" and use_fallback:
                    timed_out.append(qnum)
                    print(f"Q{qnum:<9} {'TIMEOUT->local':<16} {'—':<12} {'—':<12} {'—':<12} {detail}")
                    try:
                        elapsed_local, rows_local = run_query_local(args.parquet_folder, qnum, args.threads)
                        total_time += elapsed_local
                        fallback_ok.append(qnum)
                        print(
                            f"Q{qnum:<9} {'OK(local)':<16} {elapsed_local:<12.3f} {'—':<12} {'—':<12} {rows_local:<12}"
                        )
                    except Exception as e:
                        failed.append(qnum)
                        print(f"Q{qnum:<9} {'FAIL(local)':<16} {'—':<12} {'—':<12} {'—':<12} {str(e)[:40]}")
                    query_failed = True
                    break
                elif status == "FAIL" and use_fallback:
                    print(f"Q{qnum:<9} {'FAIL->local':<16} {'—':<12} {'—':<12} {'—':<12} {detail}")
                    try:
                        elapsed_local, rows_local = run_query_local(args.parquet_folder, qnum, args.threads)
                        total_time += elapsed_local
                        fallback_ok.append(qnum)
                        print(
                            f"Q{qnum:<9} {'OK(local)':<16} {elapsed_local:<12.3f} {'—':<12} {'—':<12} {rows_local:<12}"
                        )
                    except Exception as e:
                        failed.append(qnum)
                        print(f"Q{qnum:<9} {'FAIL(local)':<16} {'—':<12} {'—':<12} {'—':<12} {str(e)[:40]}")
                    query_failed = True
                    break
                else:
                    failed.append(qnum)
                    if status == "TIMEOUT":
                        timed_out.append(qnum)
                    print(f"Q{qnum:<9} {status:<16} {'—':<12} {'—':<12} {'—':<12} {detail}")
                    query_failed = True
                    break

            if not query_failed and iter_times:
                min_t = min(iter_times)
                avg_t = sum(iter_times) / len(iter_times)
                med_t = statistics.median(iter_times)
                total_time += sum(iter_times)
                ray_ok.append(qnum)
                times_str = ", ".join(f"{t:.3f}" for t in iter_times)
                print(f"Q{qnum:<9} {'OK(ray)':<16} {min_t:<12.3f} {avg_t:<12.3f} {med_t:<12.3f} {last_rows:<12}")
                print(f"{'':>10} iters: [{times_str}]")
        else:
            # Local mode: run directly, no subprocess needed
            iter_times = []
            rows = 0
            query_failed = False
            for _ in range(iterations):
                try:
                    elapsed, rows = run_query_local(args.parquet_folder, qnum, args.threads)
                    iter_times.append(elapsed)
                except Exception as e:
                    query_failed = True
                    failed.append(qnum)
                    print(f"Q{qnum:<9} {'FAIL':<16} {'—':<12} {'—':<12} {'—':<12} {str(e)[:40]}")
                    break
            if not query_failed and iter_times:
                min_t = min(iter_times)
                avg_t = sum(iter_times) / len(iter_times)
                med_t = statistics.median(iter_times)
                total_time += sum(iter_times)
                ray_ok.append(qnum)
                times_str = ", ".join(f"{t:.3f}" for t in iter_times)
                print(f"Q{qnum:<9} {'OK':<16} {min_t:<12.3f} {avg_t:<12.3f} {med_t:<12.3f} {rows:<12}")
                print(f"{'':>10} iters: [{times_str}]")

    print("-" * 80)
    passed = len(ray_ok) + len(fallback_ok)
    print(f"Total: {total_time:.3f}s | Passed: {passed}/{len(questions)} | Iterations: {iterations}")
    if use_ray:
        print(
            f"  Ray OK: {len(ray_ok)} | Fallback OK: {len(fallback_ok)} | "
            f"Timeout: {len(timed_out)} | Failed: {len(failed)}"
        )
        if ray_ok:
            print(f"  Ray queries: {ray_ok}")
        if fallback_ok:
            print(f"  Fallback queries: {fallback_ok}")
        if timed_out:
            print(f"  Timed-out queries: {timed_out}")
    if failed:
        print(f"  Failed queries: {failed}")

    if use_ray:
        ray.shutdown()
