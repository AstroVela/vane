# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest

try:
    from vane import ray_cxx
except Exception:
    import _vane_duckdb

    ray_cxx = _vane_duckdb.ray_cxx


def test_run_plan_uninitialized_plan_repeated_fail_fast_no_crash():
    for _ in range(20):
        runner = ray_cxx.DistributedPhysicalPlanRunner()
        plan = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)

        with pytest.raises(Exception, match="uninitialized"):
            runner.run_plan(plan)

        del plan, runner
        gc.collect()


def test_run_copy_plan_uninitialized_plan_repeated_fail_fast_no_crash():
    for _ in range(20):
        runner = ray_cxx.DistributedPhysicalPlanRunner()
        plan = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)

        with pytest.raises(Exception, match="uninitialized"):
            runner.run_copy_plan(plan)

        del plan, runner
        gc.collect()
