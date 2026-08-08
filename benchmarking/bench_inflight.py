# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark map_batches latency for a slow batch UDF.

Usage:
    python benchmarking/bench_inflight.py [--rows N] [--batch-size N] [--latency-ms N]
    python benchmarking/bench_inflight.py --backend ray_actor --actor-number 2
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import pyarrow as pa

import vane


class SlowUDF:
    """Simulates an API/GPU UDF with fixed per-batch latency."""

    def __init__(self, latency_ms: float = 100):
        self._latency_s = latency_ms / 1000.0

    def __call__(self, table: pa.Table) -> pa.Table:
        time.sleep(self._latency_s)
        return pa.table({"result": [f"row_{i}" for i in range(table.num_rows)]})


def _make_udf(backend: str, latency_ms: float):
    if backend in ("subprocess_actor", "ray_actor"):

        class SlowActorUDF(SlowUDF):
            def __init__(self) -> None:
                super().__init__(latency_ms)

        return SlowActorUDF

    latency_s = latency_ms / 1000.0

    def slow_udf(table: pa.Table) -> pa.Table:
        time.sleep(latency_s)
        return pa.table({"result": [f"row_{i}" for i in range(table.num_rows)]})

    return slow_udf


def run_benchmark(
    num_rows: int,
    batch_size: int,
    latency_ms: float,
    backend: str,
    actor_number: int | None = None,
) -> dict[str, Any]:
    conn = vane.connect()
    try:
        rel = conn.sql(f"SELECT i::VARCHAR AS text FROM range({num_rows}) t(i)")
        udf = _make_udf(backend, latency_ms)
        num_batches = (num_rows + batch_size - 1) // batch_size
        theoretical_serial = num_batches * latency_ms / 1000.0

        kwargs: dict[str, Any] = {
            "schema": {"result": "VARCHAR"},
            "batch_size": batch_size,
            "execution_backend": backend,
        }
        if actor_number is not None:
            if backend not in ("subprocess_actor", "ray_actor"):
                raise ValueError("actor_number is only supported for actor backends")
            kwargs["actor_number"] = actor_number

        start = time.perf_counter()
        out = rel.map_batches(udf, **kwargs).fetchall()
        elapsed = time.perf_counter() - start

        return {
            "backend": backend,
            "actor_number": actor_number,
            "rows": num_rows,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "latency_ms": latency_ms,
            "elapsed_s": round(elapsed, 3),
            "theoretical_serial_s": round(theoretical_serial, 3),
            "speedup": round(theoretical_serial / elapsed, 2) if elapsed > 0 else 0,
            "output_rows": len(out),
        }
    finally:
        conn.close()


def print_result(result: dict[str, Any]) -> None:
    print()
    print("=" * 80)
    print("map_batches Latency Benchmark")
    print("=" * 80)
    for key in (
        "backend",
        "actor_number",
        "rows",
        "batch_size",
        "num_batches",
        "latency_ms",
        "theoretical_serial_s",
        "elapsed_s",
        "speedup",
        "output_rows",
    ):
        print(f"{key:>22}: {result[key]}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark map_batches latency")
    parser.add_argument("--rows", type=int, default=200, help="Number of input rows")
    parser.add_argument("--batch-size", type=int, default=50, help="Rows per batch")
    parser.add_argument("--latency-ms", type=float, default=100, help="Simulated per-batch latency")
    parser.add_argument(
        "--backend",
        default="subprocess_task",
        choices=["subprocess_task", "subprocess_actor", "ray_task", "ray_actor"],
    )
    parser.add_argument("--actor-number", type=int, default=None)
    args = parser.parse_args()

    actor_number = args.actor_number
    if args.backend in ("subprocess_actor", "ray_actor") and actor_number is None:
        parser.error("--actor-number is required for actor backends")
    if args.backend in ("subprocess_task", "ray_task") and actor_number is not None:
        parser.error("--actor-number is only supported for actor backends")

    result = run_benchmark(
        num_rows=args.rows,
        batch_size=args.batch_size,
        latency_ms=args.latency_ms,
        backend=args.backend,
        actor_number=actor_number,
    )
    print_result(result)


if __name__ == "__main__":
    main()
