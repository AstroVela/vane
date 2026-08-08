#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the unified UDF Ray task backend.

This is a small CPU-only benchmark for the final unified Python UDF executor
layer, without building a full SQL plan.

Example:
    source .venv-system/bin/activate
    python benchmarking/bench_function_udf_ray_task_backend.py --num-cpus 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class BenchmarkResult:
    """Measured result for one executor backend."""

    backend: str
    rows: int
    input_batches: int
    output_rows: int
    wall_s: float
    rows_per_s: float
    inflight_limit: int


def _parse_backends(raw: str) -> list[str]:
    valid = {"local", "ray_task"}
    backends = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(backends) - valid)
    if unknown:
        message = f"unknown backend(s): {', '.join(unknown)}"
        raise ValueError(message)
    return backends


def _make_payload(work_iterations: int, execution_backend: str) -> dict:
    from vane import pickle as duckdb_pickle

    def cpu_udf(value: object) -> int:
        acc = int(value)
        for idx in range(work_iterations):
            acc = ((acc ^ idx) * 1103515245 + 12345) & 0x7FFFFFFF
        return acc

    return {
        "function_pickle": duckdb_pickle.dumps(cpu_udf),
        "call_mode": "map",
        "execution_backend": execution_backend,
    }


def _make_tables(input_batches: int, rows_per_batch: int) -> list[pa.Table]:
    tables = []
    start = 0
    for _ in range(input_batches):
        stop = start + rows_per_batch
        tables.append(pa.table({"x": list(range(start, stop))}))
        start = stop
    return tables


def _drain_executor(executor: object, tables: list[pa.Table], poll_interval_s: float) -> int:
    output_rows = 0
    for table in tables:
        executor.submit(table)
        output = executor.poll()
        if output is not None:
            output_rows += int(output.num_rows)

    executor.finished_submitting()
    while True:
        output = executor.poll()
        if output is not None:
            output_rows += int(output.num_rows)
            continue
        if executor.all_tasks_finished():
            return output_rows
        time.sleep(poll_interval_s)


def _build_executor(
    backend: str,
    payload: dict,
    bundle_rows: int,
    bundle_bytes: int,
) -> object:
    from vane.execution.unified_executor import build_unified_executor

    if backend in {"local", "ray_task"}:
        return build_unified_executor(
            payload,
            {
                "ray_task_bundle_rows": max(1, bundle_rows),
                "ray_task_max_bundle_rows": max(1, bundle_rows),
                "ray_task_max_bundle_bytes": max(1, bundle_bytes),
            },
        )
    message = f"unknown backend: {backend}"
    raise ValueError(message)


def _init_ray(num_cpus: int) -> None:
    import ray

    with suppress(Exception):
        ray.shutdown()
    ray.init(
        address="local",
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        num_cpus=num_cpus,
    )


def _shutdown_ray() -> None:
    with suppress(Exception):
        import ray

        ray.shutdown()


def _ray_object_store_bytes() -> int | None:
    with suppress(Exception):
        import ray

        info = ray._private.internal_api.memory_summary(stats_only=True)
        if not isinstance(info, str):
            return None
        for line in info.splitlines():
            if "Plasma memory usage" in line:
                fields = line.replace(",", "").split()
                for idx, field in enumerate(fields):
                    if field == "usage" and idx + 1 < len(fields):
                        return int(float(fields[idx + 1]))
    return None


def run_backend(
    backend: str,
    tables: list[pa.Table],
    work_iterations: int,
    bundle_rows: int,
    bundle_bytes: int,
    poll_interval_s: float,
) -> BenchmarkResult:
    """Run one backend and return throughput plus Ray task counters."""
    payload = _make_payload(work_iterations, backend)
    executor = _build_executor(backend, payload, bundle_rows, bundle_bytes)
    start = time.perf_counter()
    output_rows = _drain_executor(executor, tables, poll_interval_s)
    wall_s = time.perf_counter() - start

    rows = sum(int(table.num_rows) for table in tables)
    return BenchmarkResult(
        backend=backend,
        rows=rows,
        input_batches=len(tables),
        output_rows=output_rows,
        wall_s=wall_s,
        rows_per_s=(rows / wall_s) if wall_s > 0 else 0.0,
        inflight_limit=int(executor.get_inflight_batch_limit()),
    )


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results as a compact table."""
    print()
    print("=" * 104)
    print("Unified UDF Ray Task Backend Benchmark")
    print("=" * 104)
    print(f"{'backend':>10} {'rows':>8} {'batches':>8} {'wall(s)':>10} {'rows/s':>12} {'out_rows':>10} {'inflight':>8}")
    print("-" * 104)
    for result in results:
        print(
            f"{result.backend:>10} {result.rows:>8} {result.input_batches:>8} "
            f"{result.wall_s:>10.3f} {result.rows_per_s:>12.1f} "
            f"{result.output_rows:>10} {result.inflight_limit:>8}"
        )
    print("=" * 104)


def main() -> None:
    """Run the benchmark from command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark unified UDF Ray task backend")
    parser.add_argument("--backends", default="local,ray_task")
    parser.add_argument("--num-cpus", type=int, default=4)
    parser.add_argument("--input-batches", type=int, default=64)
    parser.add_argument("--rows-per-batch", type=int, default=512)
    parser.add_argument("--work-iterations", type=int, default=1000)
    parser.add_argument("--bundle-rows", type=int, default=8192)
    parser.add_argument("--bundle-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--poll-interval-ms", type=float, default=1.0)
    parser.add_argument("--json", action="store_true", help="Print JSON records after the table")
    args = parser.parse_args()

    backends = _parse_backends(args.backends)
    needs_ray = any(backend != "local" for backend in backends)
    tables = _make_tables(args.input_batches, args.rows_per_batch)

    if needs_ray:
        _init_ray(args.num_cpus)

    try:
        results = [
            run_backend(
                backend=backend,
                tables=tables,
                work_iterations=args.work_iterations,
                bundle_rows=args.bundle_rows,
                bundle_bytes=args.bundle_bytes,
                poll_interval_s=args.poll_interval_ms / 1000.0,
            )
            for backend in backends
        ]
        print_results(results)
        object_store_bytes = _ray_object_store_bytes() if needs_ray else None
        if object_store_bytes is not None:
            print(f"Ray object store bytes after benchmark: {object_store_bytes}")
        if args.json:
            for result in results:
                print(json.dumps(result.__dict__, sort_keys=True))
    finally:
        if needs_ray:
            _shutdown_ray()


if __name__ == "__main__":
    main()
