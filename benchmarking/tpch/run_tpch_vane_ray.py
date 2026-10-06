#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Measure supported TPC-H SQL through query(), with explicit execution mode.

Unsupported queries are reported as failures. Results never fall back to a
second backend. Data and the optional FTE store must be visible to all workers.
This harness measures end-to-end latency; it is not the P5 performance gate.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import vane

TABLES = ("nation", "region", "supplier", "customer", "part", "partsupp", "orders", "lineitem")


def query_sql(folder: Path, query: int) -> str:
    definitions = []
    for table in TABLES:
        path = str(folder.resolve() / table / "*.parquet").replace("'", "''")
        definitions.append(f"{table} AS (SELECT * FROM read_parquet('{path}'))")
    sql = (Path(__file__).parent / "queries" / f"{query:02d}.sql").read_text().strip().removesuffix(";")
    return f"WITH {', '.join(definitions)} SELECT * FROM ({sql}) benchmark_query"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-folder", type=Path, required=True)
    parser.add_argument("--mode", choices=("local", "pipelined", "fte"), default="local")
    parser.add_argument("--questions", default="1,3,6", help="Comma-separated TPC-H query numbers")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--store", type=Path, help="Shared FTE store root, outside the source dataset")
    args = parser.parse_args()
    if args.iterations < 1 or args.threads < 1 or args.timeout <= 0:
        parser.error("iterations, threads and timeout must be positive")
    if args.mode == "fte" and args.store is None:
        parser.error("FTE requires --store")
    backend = "local" if args.mode == "local" else "ray"
    settings = {"backend": backend, "config": {"threads": args.threads}}
    if backend == "ray":
        import ray

        ray.init(address=args.ray_address)
        settings["execution"] = args.mode
        resources = vane.RayResources()
        if args.store is not None:
            resources = replace(
                resources, exchange_stores=(vane.ExchangeStore("benchmark", str(args.store.resolve())),)
            )
        settings["resources"] = resources
    execution = (
        vane.LocalExecution()
        if backend == "local"
        else vane.RayExecution(args.mode, vane.FteOptions("benchmark", 3, 0.1) if args.mode == "fte" else None)
    )
    options = vane.QueryExecutionOptions(execution, args.timeout, args.timeout, args.timeout)
    failed = False
    try:
        with vane.connect(**settings) as connection:
            for number in map(int, args.questions.split(",")):
                for iteration in range(args.iterations):
                    record = {"query": number, "iteration": iteration, "mode": args.mode}
                    started = time.perf_counter()
                    try:
                        with connection.query(query_sql(args.parquet_folder, number), options=options) as result:
                            record["rows"] = result.collect().num_rows
                        record["status"] = "ok"
                    except Exception as error:
                        record.update(status="failed", error=str(error))
                        failed = True
                    record["seconds"] = time.perf_counter() - started
                    print(json.dumps(record), flush=True)
    finally:
        if backend == "ray":
            ray.shutdown()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
