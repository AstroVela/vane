#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT

"""Run a Relation through Ray, then stream its explicit bound-plan transport.

Run from this directory with the installed Vane package:
    cd external/duckdb/examples
    python driver_demo.py

Connect to an existing Ray cluster without starting a local one:
    python driver_demo.py --ray-address auto
"""

import argparse
import logging
import os

import pyarrow as pa
import ray
import vane

logger = logging.getLogger("driver_demo")


def run_demo(ray_address: str | None = None, verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(address=ray_address)
    # Connection policy is fixed when the explicit connection is created.
    os.environ["VANE_RUNNER"] = "ray"
    try:
        runner = vane.set_runner_ray()
        with vane.connect() as connection:
            source = pa.table({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
            relation = connection.from_arrow(source).project("a, b, a + b AS total")
            logger.info("Rows from the shared Relation execution entry: %s", relation.fetchall())
            # Low-level runner methods consume a bound plan, with the source
            # relation retained until its result stream has closed.
            plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, None)
            partitions = runner.run_iter_tables(plan)
            try:
                for table in partitions:
                    print(table.to_pydict())
            finally:
                partitions.close()
    finally:
        vane.teardown_runner()
        if owns_ray:
            ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", help="Ray cluster address; use 'auto' to require an existing cluster")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    run_demo(ray_address=args.ray_address, verbose=args.verbose)
