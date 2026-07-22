#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Print the distributed physical plan (repr_ascii) for each TPC-H query.

Usage:
    python benchmarking/tpch/show_distributed_plans.py \
        --parquet_folder data/tpch10 --questions 1,2,3,4,5,6
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import uuid

TABLES = ["nation", "region", "supplier", "customer", "part", "partsupp", "orders", "lineitem"]
QUERY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries")


def show_plan(parquet_folder, qnum, threads, plan_timeout):
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

    # Build the relation
    relation = con.sql(sql)

    # Build logical plan
    PyLogicalPlan = vane.ray_cxx.PyLogicalPlan
    query_id = str(uuid.uuid4())

    logical_plan = PyLogicalPlan.from_duckdb_relation(relation, query_id)

    # Convert to distributed physical plan
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(logical_plan.to_physical_plan)
        try:
            distributed_plan = future.result(timeout=plan_timeout)
        except concurrent.futures.TimeoutError:
            return f"TIMEOUT (>{plan_timeout}s)"
        except Exception as e:
            return f"ERROR: {str(e)[:200]}"

    # Get ASCII representation
    try:
        return distributed_plan.repr_ascii(False)
    except Exception as e:
        return f"repr_ascii ERROR: {str(e)[:200]}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show distributed physical plans for TPC-H queries")
    parser.add_argument("--parquet_folder", required=True)
    parser.add_argument("--questions", default=None, help="Comma-separated query numbers (default: all 1-22)")
    parser.add_argument("--threads", default=None, type=int)
    parser.add_argument("--plan-timeout", default=30, type=int)
    args = parser.parse_args()

    # Need to set env before certain imports
    os.environ.setdefault("VANE_DISTRIBUTED_NODE_COUNT", "1")

    if args.questions:
        questions = sorted({int(q) for q in args.questions.split(",")})
    else:
        questions = list(range(1, 23))

    for qnum in questions:
        print(f"\n{'=' * 80}")
        print(f"  Q{qnum} — Distributed Physical Plan")
        print(f"{'=' * 80}")
        try:
            result = show_plan(args.parquet_folder, qnum, args.threads, args.plan_timeout)
            print(result)
        except Exception as e:
            print(f"EXCEPTION: {e}")
        sys.stdout.flush()
