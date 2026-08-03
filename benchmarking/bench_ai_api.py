#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""No-network release benchmark for the public Prompt relation API.

The benchmark records planning cost, mock-provider throughput, row preservation,
parent-process peak allocations, and subprocess actor scaling.  Use
``VANE_BENCH_REPO_ROOT`` to run this script against another checkout while
keeping the exact same benchmark implementation.

Example::

    VANE_BENCH_REPO_ROOT=. python benchmarking/bench_ai_api.py --repeat 5
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(os.environ.get("VANE_BENCH_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vane  # noqa: E402
from vane.ai.protocols import PrompterDescriptor  # noqa: E402
from vane.ai.provider import Provider  # noqa: E402
from vane.ai.typing import UDFOptions  # noqa: E402


class MockPrompter:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s

    async def prompt(self, messages: tuple[Any, ...]) -> str:
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return f"mock:{messages[0]}"


@dataclass
class MockPrompterDescriptor(PrompterDescriptor):
    delay_s: float

    def get_provider(self) -> str:
        return "benchmark-mock"

    def get_model(self) -> str:
        return "benchmark-mock"

    def get_options(self) -> dict[str, Any]:
        return {}

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(num_gpus=0)

    def instantiate(self) -> MockPrompter:
        return MockPrompter(self.delay_s)


@dataclass
class MockProvider(Provider):
    delay_s: float = 0.0

    @property
    def name(self) -> str:
        return "benchmark-mock"

    def get_prompter(self, model: str | None = None, **options: Any) -> PrompterDescriptor:
        return MockPrompterDescriptor(self.delay_s)


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
        "repo_root": str(REPO_ROOT),
    }


def _planning_samples(repeat: int, iterations: int) -> list[float]:
    connection = vane.connect()
    try:
        source = connection.sql("SELECT 'planning'::VARCHAR AS text")

        def plan_once() -> None:
            planned = source.select(
                vane.ai.prompt(
                    vane.col("text"),
                    provider=MockProvider(),
                    max_retries=0,
                ).alias("response")
            )
            assert [str(value) for value in planned.types] == ["VARCHAR"]

        for _ in range(3):
            plan_once()
        samples: list[float] = []
        for _ in range(repeat):
            started = time.perf_counter()
            for _ in range(iterations):
                plan_once()
            samples.append((time.perf_counter() - started) / iterations)
        return samples
    finally:
        connection.close()


def _actor_sample(rows: int, batch_size: int, actor_number: int, delay_s: float) -> tuple[float, int]:
    connection = vane.connect()
    try:
        source = connection.sql(
            "SELECT i::BIGINT AS row_id, concat('row-', i::VARCHAR)::VARCHAR AS text "
            f"FROM range({rows}) AS source(i)"
        ).repartition(actor_number)
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        output = (
            vane.ai.prompt(
                source,
                vane.col("text"),
                provider=MockProvider(delay_s),
                execution_backend="subprocess_actor",
                actor_number=actor_number,
                max_concurrency_per_actor=1,
                batch_size=batch_size,
                max_retries=0,
            )
            .project("row_id, response")
            .fetchall()
        )
        elapsed = time.perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        connection.close()

    expected = {(index, f"mock:row-{index}") for index in range(rows)}
    if set(output) != expected:
        raise AssertionError("Prompt benchmark output violated the row-preserving result contract")
    return elapsed, peak_bytes


def _actor_result(
    *,
    rows: int,
    batch_size: int,
    actor_number: int,
    delay_s: float,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        _actor_sample(rows, batch_size, actor_number, delay_s)
    samples = [_actor_sample(rows, batch_size, actor_number, delay_s) for _ in range(repeat)]
    elapsed = [sample[0] for sample in samples]
    peak_bytes = [sample[1] for sample in samples]
    median_elapsed = statistics.median(elapsed)
    return {
        "actor_number": actor_number,
        "wall_s_median": median_elapsed,
        "rows_per_s_median": rows / median_elapsed,
        "parent_peak_bytes_median": statistics.median(peak_bytes),
        "output_rows": rows,
        "wall_s_samples": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--planning-iterations", type=int, default=25)
    parser.add_argument("--delay-ms", type=float, default=2.0)
    parser.add_argument("--actors", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    if min(args.rows, args.batch_size, args.repeat, args.planning_iterations, *args.actors) <= 0:
        parser.error("rows, batch-size, repeat, planning-iterations, and actors must be positive")
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    if args.delay_ms < 0:
        parser.error("delay-ms must be non-negative")

    vane.configure(runner="local")
    planning = _planning_samples(args.repeat, args.planning_iterations)
    actor_results = [
        _actor_result(
            rows=args.rows,
            batch_size=args.batch_size,
            actor_number=actor_number,
            delay_s=args.delay_ms / 1000.0,
            repeat=args.repeat,
            warmup=args.warmup,
        )
        for actor_number in args.actors
    ]
    report = {
        "source": _git_metadata(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "vane": getattr(vane, "__version__", None),
        },
        "config": vars(args),
        "results": {
            "planning_us_per_call_median": statistics.median(planning) * 1_000_000,
            "planning_s_per_call_samples": planning,
            "mock_prompt_actor_runs": actor_results,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
