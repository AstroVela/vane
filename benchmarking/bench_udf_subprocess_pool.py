#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark UDF execution backends and subprocess worker-pool sizing.

The default run only uses local subprocess backends and does not start Ray.
Pass --include-ray to add ray_task and ray_actor rows to the same matrix.

Examples:
    source .venv-system/bin/activate
    python benchmarking/bench_udf_subprocess_pool.py --workloads sleep --rows 16 --batch-size 1
    python benchmarking/bench_udf_subprocess_pool.py --include-ray --actor-number 1,2,4
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vane  # noqa: E402

RAY_BACKENDS = ("ray_task",)


class SleepUDF:
    def __init__(self, sleep_ms: float) -> None:
        self.sleep_s = float(sleep_ms) / 1000.0

    def __call__(self, table: pa.Table) -> pa.Table:
        time.sleep(self.sleep_s)
        return pa.table({"y": table.column("x")})


class CpuUDF:
    def __init__(self, iterations: int) -> None:
        self.iterations = int(iterations)

    def __call__(self, table: pa.Table) -> pa.Table:
        out: list[int] = []
        for value in table.column("x").to_pylist():
            acc = int(value)
            for idx in range(self.iterations):
                acc = ((acc ^ idx) * 1103515245 + 12345) & 0x7FFFFFFF
            out.append(acc)
        return pa.table({"y": pa.array(out, type=pa.int64())})


class ArrowIdentityUDF:
    def __call__(self, table: pa.Table) -> pa.Table:
        return pa.table({name: table.column(name) for name in table.schema.names})


@dataclass
class BenchResult:
    workload: str
    backend: str
    actor_number: int | None
    rows: int
    batch_size: int
    batches: int
    streaming_breaker: bool
    repeat: int
    wall_s_mean: float
    wall_s_min: float
    rows_per_s_mean: float
    output_rows: int
    ok: bool
    error: str | None = None


def _parse_csv(value: str, *, cast=str) -> list[Any]:
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def _backend_supports_actor_number(backend: str) -> bool:
    return backend in {"subprocess_actor", "ray_actor"}


def _uses_actor_backend(backend: str) -> bool:
    return backend in {"subprocess_actor", "ray_actor"}


def _make_udf_and_schema(workload: str, backend: str, args: argparse.Namespace):
    if workload == "sleep":
        if _uses_actor_backend(backend):
            sleep_ms = args.sleep_ms

            class SleepActorUDF(SleepUDF):
                def __init__(self) -> None:
                    super().__init__(sleep_ms)

            return SleepActorUDF, {"y": vane.sqltypes.BIGINT}

        sleep_s = float(args.sleep_ms) / 1000.0

        def sleep_udf(table: pa.Table) -> pa.Table:
            time.sleep(sleep_s)
            return pa.table({"y": table.column("x")})

        return sleep_udf, {"y": vane.sqltypes.BIGINT}
    if workload == "cpu":
        if _uses_actor_backend(backend):
            iterations = args.cpu_iterations

            class CpuActorUDF(CpuUDF):
                def __init__(self) -> None:
                    super().__init__(iterations)

            return CpuActorUDF, {"y": vane.sqltypes.BIGINT}

        iterations = int(args.cpu_iterations)

        def cpu_udf(table: pa.Table) -> pa.Table:
            out: list[int] = []
            for value in table.column("x").to_pylist():
                acc = int(value)
                for idx in range(iterations):
                    acc = ((acc ^ idx) * 1103515245 + 12345) & 0x7FFFFFFF
                out.append(acc)
            return pa.table({"y": pa.array(out, type=pa.int64())})

        return cpu_udf, {"y": vane.sqltypes.BIGINT}
    if workload == "arrow":
        schema = {
            "x": vane.sqltypes.BIGINT,
            "payload": vane.sqltypes.VARCHAR,
        }
        if _uses_actor_backend(backend):
            return ArrowIdentityUDF, schema

        def arrow_identity_udf(table: pa.Table) -> pa.Table:
            return pa.table({name: table.column(name) for name in table.schema.names})

        return arrow_identity_udf, schema
    raise ValueError(f"unknown workload: {workload}")


def _make_relation(con: vane.DuckDBPyConnection, workload: str, rows: int, payload_bytes: int):
    if workload == "arrow":
        return con.sql(
            "select i::BIGINT as x, repeat('x', %d) as payload from range(%d) t(i)" % (int(payload_bytes), int(rows))
        )
    return con.sql("select i::BIGINT as x from range(%d) t(i)" % int(rows))


def _run_once(
    workload: str,
    backend: str,
    actor_number: int | None,
    args: argparse.Namespace,
) -> tuple[float, int]:
    con = vane.connect()
    try:
        rel = _make_relation(con, workload, args.rows, args.payload_bytes)
        udf, schema = _make_udf_and_schema(workload, backend, args)
        kwargs: dict[str, Any] = {
            "schema": schema,
            "batch_size": args.batch_size,
            "execution_backend": backend,
        }
        if _backend_supports_actor_number(backend) and actor_number is not None:
            kwargs["actor_number"] = int(actor_number)
        if args.streaming_breaker:
            kwargs["streaming_breaker"] = True
        if backend.startswith("ray_") and args.ray_cpus is not None:
            kwargs["cpus"] = float(args.ray_cpus)
        start = time.perf_counter()
        output = rel.map_batches(udf, **kwargs).fetchall()
        return time.perf_counter() - start, len(output)
    finally:
        con.close()


def _run_case(
    workload: str,
    backend: str,
    actor_number: int | None,
    args: argparse.Namespace,
) -> BenchResult:
    batches = (args.rows + args.batch_size - 1) // args.batch_size
    timings: list[float] = []
    output_rows = 0
    try:
        for _ in range(args.repeat):
            wall_s, output_rows = _run_once(workload, backend, actor_number, args)
            timings.append(wall_s)
        mean_s = statistics.mean(timings)
        return BenchResult(
            workload=workload,
            backend=backend,
            actor_number=actor_number,
            rows=args.rows,
            batch_size=args.batch_size,
            batches=batches,
            streaming_breaker=bool(args.streaming_breaker),
            repeat=args.repeat,
            wall_s_mean=mean_s,
            wall_s_min=min(timings),
            rows_per_s_mean=(args.rows / mean_s) if mean_s > 0 else 0.0,
            output_rows=output_rows,
            ok=True,
        )
    except Exception as exc:
        return BenchResult(
            workload=workload,
            backend=backend,
            actor_number=actor_number,
            rows=args.rows,
            batch_size=args.batch_size,
            batches=batches,
            streaming_breaker=bool(args.streaming_breaker),
            repeat=args.repeat,
            wall_s_mean=0.0,
            wall_s_min=0.0,
            rows_per_s_mean=0.0,
            output_rows=output_rows,
            ok=False,
            error=str(exc),
        )


def _maybe_init_ray(args: argparse.Namespace) -> bool:
    if not args.include_ray:
        return False
    import ray

    if ray.is_initialized():
        return False
    init_kwargs: dict[str, Any] = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "log_to_driver": False,
    }
    if args.ray_address:
        init_kwargs["address"] = args.ray_address
    else:
        init_kwargs["num_cpus"] = int(args.ray_num_cpus)
    ray.init(**init_kwargs)
    return True


def _shutdown_ray(started: bool) -> None:
    if not started:
        return
    with suppress(Exception):
        import ray

        ray.shutdown()


def _print_table(results: list[BenchResult]) -> None:
    print()
    print("=" * 118)
    print("UDF Backend Microbenchmark")
    print("=" * 118)
    print(
        f"{'workload':>8} {'backend':>18} {'actors':>6} {'batches':>7} "
        f"{'mean(s)':>9} {'min(s)':>9} {'rows/s':>12} {'out':>8} {'ok':>4} error"
    )
    print("-" * 118)
    for result in results:
        conc = "-" if result.actor_number is None else str(result.actor_number)
        print(
            f"{result.workload:>8} {result.backend:>18} {conc:>6} {result.batches:>7} "
            f"{result.wall_s_mean:>9.3f} {result.wall_s_min:>9.3f} "
            f"{result.rows_per_s_mean:>12.1f} {result.output_rows:>8} "
            f"{result.ok!s:>4} {result.error or ''}"
        )
    print("=" * 118)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark UDF backend worker-pool overhead")
    parser.add_argument("--workloads", default="sleep,cpu,arrow")
    parser.add_argument("--backends", default="subprocess_task,subprocess_actor")
    parser.add_argument("--include-ray", action="store_true", help="Add ray_task to the backend matrix")
    parser.add_argument(
        "--include-ray-actor",
        action="store_true",
        help="Also add ray_actor. This requires a driver-precreated actor pool in current Vane.",
    )
    parser.add_argument("--actor-number", default="1,2,4")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--sleep-ms", type=float, default=50.0)
    parser.add_argument("--cpu-iterations", type=int, default=20000)
    parser.add_argument("--payload-bytes", type=int, default=4096)
    parser.add_argument("--streaming-breaker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ray-address", default="")
    parser.add_argument("--ray-num-cpus", type=int, default=8)
    parser.add_argument("--ray-cpus", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    workloads = _parse_csv(args.workloads)
    backends = _parse_csv(args.backends)
    if args.include_ray:
        for backend in RAY_BACKENDS:
            if backend not in backends:
                backends.append(backend)
    if args.include_ray_actor and "ray_actor" not in backends:
        backends.append("ray_actor")
    actor_numbers = _parse_csv(args.actor_number, cast=int)

    ray_started = _maybe_init_ray(args)
    try:
        results: list[BenchResult] = []
        for workload in workloads:
            for backend in backends:
                if _backend_supports_actor_number(backend):
                    for actor_number in actor_numbers:
                        results.append(_run_case(workload, backend, actor_number, args))
                else:
                    results.append(_run_case(workload, backend, None, args))
        _print_table(results)
        if args.json:
            for result in results:
                print(json.dumps(asdict(result), sort_keys=True))
    finally:
        _shutdown_ray(ray_started)


if __name__ == "__main__":
    main()
