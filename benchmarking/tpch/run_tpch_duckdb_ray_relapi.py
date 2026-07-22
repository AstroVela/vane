#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run DuckDB TPC-H benchmarks with true Ray distributed execution via Relation API.

Unlike run_tpch_duckdb_ray.py (which uses con.execute(sql).fetchall() — always local),
this script uses con.sql(sql).write_parquet(path) which triggers the VANE_RUNNER=ray
dispatch path: pyrelation.cpp → runner.run_write() → PyLogicalPlan →
to_physical_plan() → RayQueryDriverClient on Ray.

Each query runs in a subprocess (spawn) for isolation. Failures are classified as:
  TIMEOUT_ERROR  — query raised a timeout error
  UNSUPPORTED   — distributed translator hit unsupported operator (e.g. NESTED_LOOP_JOIN)
  TIMEOUT       — subprocess killed after --timeout seconds
  FAIL          — other error

This script does not fall back to local execution. Failed queries are reported and the run continues.

Usage:
    # Simple smoke test (Q1, Q6 are most likely to succeed distributed)
    python benchmarking/tpch/run_tpch_duckdb_ray_relapi.py \
        --parquet_folder data/tpch10 --questions 1,6

    # Full run, all 22 queries
    python benchmarking/tpch/run_tpch_duckdb_ray_relapi.py \
        --parquet_folder data/tpch10

    # With verification and custom timeouts
    python benchmarking/tpch/run_tpch_duckdb_ray_relapi.py \
        --parquet_folder data/tpch10 --verify --timeout 300
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import statistics
import sys
import tempfile
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TABLES = ["nation", "region", "supplier", "customer", "part", "partsupp", "orders", "lineitem"]
QUERY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries")

# Daft shuffle/exchange defaults — required for distributed plans with shuffle
# (GROUP BY, JOIN, etc.). Without these, RayQueryDriverClient hangs waiting for exchange.
DAFT_SHUFFLE_DEFAULTS = {
    "DAFT_SHUFFLE_ALGORITHM": "flight_shuffle",
    "DAFT_EXCHANGE_FLIGHT_PORT": "35010",
    "DAFT_EXCHANGE_FLIGHT_BIND_HOST": "127.0.0.1",
    "DAFT_EXCHANGE_FLIGHT_LOCATION_TEMPLATE": "grpc+tcp://127.0.0.1:35010",
    "DAFT_SHUFFLE_LOCAL_DIRS": "/tmp/duckdb_shuffle",
}


def _clean_output(path: str):
    """Remove output path whether it's a file or directory (distributed writes a dir of shards)."""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)


def _read_row_count(path: str) -> int:
    """Read row count from output parquet (handles both single file and directory of shards)."""
    import vane as _vane

    if os.path.isdir(path):
        glob = os.path.join(path, "*.parquet")
        return _vane.sql(f"SELECT count(*) FROM read_parquet('{glob}')").fetchone()[0]
    return _vane.sql(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]


def _classify_error(exc_str: str) -> str:
    """Classify an exception string into a failure category."""
    lowered = exc_str.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return "TIMEOUT_ERROR"
    if "not implemented" in lowered or "unsupported" in lowered or "nested_loop_join" in lowered:
        return "UNSUPPORTED"
    return "FAIL"


def _run_query_distributed_in_subprocess(queue, parquet_folder, qnum, threads, ray_address, output_path, mode="write"):
    """Subprocess target: run a single query via write_parquet or fetchall with VANE_RUNNER=ray.

    Must set env vars BEFORE importing Vane so the runner dispatch is active.
    """
    try:
        # gRPC workaround: disable fork support and use epoll1 polling
        # to prevent SETTINGS frame timeouts on this machine.
        os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
        os.environ["GRPC_POLL_STRATEGY"] = "epoll1"

        # Set env vars before importing Vane
        os.environ["VANE_RUNNER"] = "ray"

        # Set Daft shuffle/exchange env vars (use existing env if already set, else defaults)
        for key, default in DAFT_SHUFFLE_DEFAULTS.items():
            if key not in os.environ:
                os.environ[key] = default

        import ray

        import vane

        if not ray.is_initialized():
            ray.init(address=ray_address, ignore_reinit_error=True)

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
        if mode == "fetch":
            # Explicit distributed execution via runner.run_iter()
            import vane.runners

            runner = vane.runners.get_or_create_runner()
            tables = [r.partition() for r in runner.run_iter(con.sql(sql))]
            elapsed = time.time() - t0
            non_empty = [t for t in tables if t.num_rows > 0]
            row_count = sum(t.num_rows for t in non_empty) if non_empty else 0
        else:
            # Distributed write via runner.run_write()
            import vane.runners

            runner = vane.runners.get_or_create_runner()
            runner.run_write(con.sql(sql))
            elapsed = time.time() - t0
            try:
                row_count = _read_row_count(output_path)
            except Exception:
                row_count = -1

        con.close()
        queue.put(("OK", elapsed, row_count))
    except Exception as e:
        import traceback

        traceback.print_exc()
        error_str = str(e)
        category = _classify_error(error_str)
        queue.put((category, 0.0, error_str[:120]))


def run_query_distributed_with_timeout(parquet_folder, qnum, threads, timeout, ray_address, output_path, mode="write"):
    """Launch distributed query in subprocess with timeout. Returns (status, elapsed, detail)."""
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_run_query_distributed_in_subprocess,
        args=(queue, parquet_folder, qnum, threads, ray_address, output_path, mode),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        # Kill stuck query driver actor so ray.shutdown() won't hang
        try:
            import ray as _ray

            for actor_name in ("ray-query-driver-actor",):
                try:
                    actor = _ray.get_actor(actor_name, namespace="vane")
                except Exception:
                    continue
                _ray.kill(actor, no_restart=True)
                print(
                    f"  [diag] killed stuck actor '{actor_name}' after subprocess timeout",
                    flush=True,
                )
                break
        except Exception:
            pass
        return "TIMEOUT", timeout, f"subprocess killed after {timeout}s"

    if not queue.empty():
        return queue.get()

    exit_code = proc.exitcode
    return "FAIL", 0.0, f"subprocess exited without result (exitcode={exit_code})"


def verify_row_counts(parquet_folder, qnum, threads):
    """Run query locally with fetchall to get reference row count."""
    import vane

    con = vane.connect()
    if threads:
        con.execute(f"SET threads={threads}")

    for table in TABLES:
        parquet_path = os.path.join(parquet_folder, table, "*.parquet")
        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{parquet_path}')")

    query_file = os.path.join(QUERY_DIR, f"{qnum:02d}.sql")
    with open(query_file) as f:
        sql = f.read()

    result = con.execute(sql).fetchall()
    con.close()
    return len(result)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Run TPC-H benchmarks with true Ray distributed execution (Relation API)"
    )
    parser.add_argument("--parquet_folder", required=True, help="Path to parquet data root (e.g. data/tpch10)")
    parser.add_argument("--questions", default=None, help="Comma-separated query numbers (default: all 1-22)")
    parser.add_argument("--threads", default=None, type=int, help="Number of DuckDB threads (default: all cores)")
    parser.add_argument(
        "--timeout", default=300, type=int, help="Per-query subprocess timeout in seconds (default: 300)"
    )
    parser.add_argument("--verify", action="store_true", help="Verify distributed row counts match local execution")
    parser.add_argument("--output-dir", default=None, help="Directory for temp parquet output (default: auto tmpdir)")
    parser.add_argument(
        "--mode",
        default="fetch",
        choices=["write", "fetch"],
        help="Execution mode: 'write' (write_parquet, default) or 'fetch' (fetchall, no file I/O)",
    )
    parser.add_argument("--iterations", default=3, type=int, help="Number of iterations per query (default: 3)")
    args = parser.parse_args()

    import vane

    print(f"DuckDB version: {vane.__version__}")

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        cleanup_output_dir = False
    else:
        output_dir = tempfile.mkdtemp(prefix="tpch_relapi_")
        cleanup_output_dir = True

    # Set Daft shuffle/exchange env vars in main process (inherited by spawn subprocesses)
    for key, default in DAFT_SHUFFLE_DEFAULTS.items():
        if key not in os.environ:
            os.environ[key] = default

    # gRPC workaround: disable fork support and use epoll1 polling
    # to prevent SETTINGS frame timeouts on this machine.
    os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
    os.environ["GRPC_POLL_STRATEGY"] = "epoll1"

    # Initialize Ray in the main process to get address
    import ray

    ray.init()
    ray_address = ray.get_runtime_context().gcs_address
    print("Runner: ray (distributed via Relation API)")
    print(f"  Nodes: {len(ray.nodes())}, CPUs: {int(ray.cluster_resources().get('CPU', 0))}")
    print(f"  Ray address: {ray_address}")
    print(f"  Subprocess timeout: {args.timeout}s")
    print(f"  Verify: {'enabled' if args.verify else 'disabled'}")
    print(f"  Mode: {args.mode}")
    print(f"  Output dir: {output_dir}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Shuffle: {os.environ['DAFT_SHUFFLE_ALGORITHM']} (port={os.environ['DAFT_EXCHANGE_FLIGHT_PORT']})")

    if args.questions:
        questions = sorted({int(q) for q in args.questions.split(",")})
    else:
        questions = list(range(1, 23))

    # Show thread count
    con = vane.connect()
    if args.threads:
        con.execute(f"SET threads={args.threads}")
    thread_count = con.execute("SELECT current_setting('threads')").fetchone()[0]
    print(f"  Threads: {thread_count}")
    con.close()

    iterations = args.iterations
    print(f"\nRunning TPC-H queries {questions} on data: {args.parquet_folder}\n")
    print(f"{'Query':<10} {'Status':<24} {'Min (s)':<12} {'Avg (s)':<12} {'Med (s)':<12} {'Rows':<12}")
    print("-" * 86)

    wall_t0 = time.time()
    total_time = 0.0
    distributed_ok = []
    timeout_error_queries = []
    unsupported_queries = []
    timeout_queries = []
    failed_queries = []
    verification_mismatches = []
    ok_times_s: dict[int, list[float]] = {}  # qnum -> list of iteration times

    for qnum in questions:
        output_path = os.path.join(output_dir, f"q{qnum:02d}.parquet")

        iter_times = []
        last_rows = 0
        query_failed = False
        for _ in range(iterations):
            # Clean up any previous output (distributed writes a directory of shards)
            _clean_output(output_path)

            status, elapsed, detail = run_query_distributed_with_timeout(
                args.parquet_folder,
                qnum,
                args.threads,
                args.timeout,
                ray_address,
                output_path,
                args.mode,
            )

            if status == "OK":
                iter_times.append(elapsed)
                last_rows = detail
            else:
                # Categorize and track the failure
                if status == "TIMEOUT_ERROR":
                    timeout_error_queries.append(qnum)
                elif status == "UNSUPPORTED":
                    unsupported_queries.append(qnum)
                elif status == "TIMEOUT":
                    timeout_queries.append(qnum)

                failed_queries.append(qnum)
                print(f"Q{qnum:<9} {status:<24} {'—':<12} {'—':<12} {'—':<12} {str(detail)[:50]}")
                query_failed = True
                break

        if not query_failed and iter_times:
            min_t = min(iter_times)
            avg_t = sum(iter_times) / len(iter_times)
            med_t = statistics.median(iter_times)
            total_time += sum(iter_times)
            distributed_ok.append(qnum)
            ok_times_s[qnum] = iter_times
            times_str = ", ".join(f"{t:.3f}" for t in iter_times)
            print(f"Q{qnum:<9} {'OK(distributed)':<24} {min_t:<12.3f} {avg_t:<12.3f} {med_t:<12.3f} {last_rows:<12}")
            print(f"{'':>10} iters: [{times_str}]")

            # Verify row count if requested
            if args.verify:
                try:
                    expected_rows = verify_row_counts(args.parquet_folder, qnum, args.threads)
                    if last_rows != expected_rows:
                        verification_mismatches.append((qnum, last_rows, expected_rows))
                        print(f"  ⚠ VERIFY MISMATCH: distributed={last_rows}, local={expected_rows}")
                    else:
                        print(f"  ✓ verified: {expected_rows} rows")
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    print(f"  ⚠ verify error: {str(e)[:60]}")

    # Summary
    print("-" * 86)
    passed = len(distributed_ok)
    wall_time = time.time() - wall_t0
    print(
        f"\nTotal time (OK only): {total_time:.3f}s | Wall time: {wall_time:.3f}s | Passed: {passed}/{len(questions)} | Iterations: {iterations}"
    )
    print(f"  Distributed OK: {len(distributed_ok)}")
    print(f"  Timeout error:   {len(timeout_error_queries)}")
    print(f"  Unsupported:    {len(unsupported_queries)}")
    print(f"  Subprocess timeout: {len(timeout_queries)}")
    print(f"  Failed:         {len(failed_queries)}")

    # Timing summary: print again at the end so users don't need to scroll.
    if ok_times_s:
        # Use min time per query for the summary stats
        ok_min_times = {q: min(times) for q, times in ok_times_s.items()}
        min_times_list = [ok_min_times[q] for q in questions if q in ok_min_times]
        min_q = min(ok_min_times.items(), key=lambda kv: kv[1])[0]
        max_q = max(ok_min_times.items(), key=lambda kv: kv[1])[0]
        sum_s = sum(min_times_list)
        avg_s = sum_s / len(min_times_list)
        median_s = statistics.median(sorted(min_times_list))

        print("\nTiming summary (successful distributed queries, min times):")
        print(f"  Count:  {len(min_times_list)}")
        print(f"  Sum:    {sum_s:.3f}s")
        print(f"  Avg:    {avg_s:.3f}s")
        print(f"  Median: {median_s:.3f}s")
        print(f"  Min:    {ok_min_times[min_q]:.3f}s (Q{min_q})")
        print(f"  Max:    {ok_min_times[max_q]:.3f}s (Q{max_q})")

        # Show the slowest OK queries for quick triage.
        slowest = sorted(ok_min_times.items(), key=lambda kv: kv[1], reverse=True)[: min(10, len(ok_min_times))]
        print("\n  Slowest OK queries (min time):")
        print(f"  {'Query':<10} {'Min (s)':<12} {'All iters'}")
        for qn, t in slowest:
            ts = ", ".join(f"{x:.3f}" for x in ok_times_s[qn])
            print(f"  Q{qn:<9} {t:<12.3f} [{ts}]")

        print("\n  Per-query OK times (in run order):")
        print(f"  {'Query':<10} {'Min (s)':<12} {'All iters'}")
        for qn in questions:
            if qn in ok_times_s:
                ts = ", ".join(f"{x:.3f}" for x in ok_times_s[qn])
                print(f"  Q{qn:<9} {ok_min_times[qn]:<12.3f} [{ts}]")
    else:
        print("\nTiming summary: no successful distributed queries.")

    if distributed_ok:
        print(f"\n  Distributed queries: {distributed_ok}")
    if timeout_error_queries:
        print(f"  Timeout error queries: {timeout_error_queries}")
    if unsupported_queries:
        print(f"  Unsupported queries:  {unsupported_queries}")
    if timeout_queries:
        print(f"  Timed-out queries:    {timeout_queries}")
    if failed_queries:
        print(f"  Failed queries:       {failed_queries}")
    if verification_mismatches:
        print("\n  Verification mismatches:")
        for qn, got, expected in verification_mismatches:
            print(f"    Q{qn}: distributed={got}, local={expected}")

    # Cleanup
    if cleanup_output_dir:
        shutil.rmtree(output_dir, ignore_errors=True)

    print("[diag] summary printed, about to call ray.shutdown()...", flush=True)
    ray.shutdown()
    print("[diag] ray.shutdown() completed", flush=True)
