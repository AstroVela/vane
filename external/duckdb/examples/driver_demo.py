#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT

"""
Minimal demo that uses `driver.py` via the Ray runner to execute a small plan.

This demo will:
- initialize ray (or rely on an existing Ray cluster)
- set the DuckDB Runner to Ray
- create a small DataFrame using `duckdb.from_pydict`
- perform a small transformation (select + expression)
- collect / show the results which will trigger the DistributedPhysicalPlanRunner used by Driver

Usage:
  python examples/driver_demo.py

Optional args:
  --no-ray-init    Don't call ray.init() and expect a Ray cluster to be running
  --verbose        Enable debug logging
"""

import argparse
import logging
import os
import sys

# Ensure our local repo is discoverable as a module when running the demo from the repository root
# Not strictly necessary if you installed the package to your venv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray
import duckdb

from duckdb import from_pydict, col

logger = logging.getLogger("driver_demo")


def run_demo(no_ray_init: bool = False, verbose: bool = False) -> None:
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if not no_ray_init:
        logger.info("Initializing ray (local head)...")
        ray.init(ignore_reinit_error=True)
    else:
        logger.info("Not initializing ray; using existing cluster")

    # Use the Ray runner. Ray is the default when `VANE_RUNNER` is unset or empty, but we call it
    # explicitly to ensure it uses the Ray runner.
    duckdb.set_runner_ray()
    logger.info("Current runner type: %s", duckdb.get_or_create_runner().name)

    # Small data set
    data = {"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}

    # Create DataFrame from plain dict
    df = from_pydict(data)

    # Make a simple expression and a projection
    df = df.select("a", "b", (col("a") + col("b")).alias("sum"))

    logger.info("Running collect() which will execute the plan through the configured runner (Ray/Driver)...")
    # collect() blocks and materializes the result; as we set the runner to Ray, this should use Driver
    collected = df.collect()

    logger.info("Result: (Preview)")
    collected.show()

    # For additional traceability, print raw partition content using run_iter_tables.
    logger.info("Printing partitions using runner.run_iter_tables for explicit iteration:")
    runner = duckdb.get_or_create_runner()
    builder = df._builder
    for table in runner.run_iter_tables(builder, results_buffer_size=1):
        try:
            print(table.to_pandas())
        except Exception:
            # to keep the demo robust, fallback to printing a string repr
            print(repr(table))
    # Show how RayQueryDriverClient can be used directly to stream a DistributedPhysicalPlan
    try:
        from duckdb.runners.ray.driver import RayQueryDriverClient

        if duckdb.get_or_create_runner().name == "ray":
            logger.info("Instantiating RayQueryDriverClient directly to show the remote actor handle.")
            driver = RayQueryDriverClient()
            logger.info("Created RayQueryDriverClient actor: %s", driver.runner)
    except Exception as e:
        logger.debug("Unable to instantiate RayQueryDriverClient directly: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ray-init", action="store_true", help="Don't call ray.init(); assume cluster exists")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    run_demo(no_ray_init=args.no_ray_init, verbose=args.verbose)
