# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from vane import ray_cxx


def test_run_plan_sync_uninitialized_plan_fail_fast():
    runner = ray_cxx.DistributedPhysicalPlanRunner()
    plan = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)

    with pytest.raises(Exception, match="uninitialized"):
        runner.run_plan(plan)


def test_run_copy_plan_uninitialized_plan_fail_fast():
    runner = ray_cxx.DistributedPhysicalPlanRunner()
    plan = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)

    with pytest.raises(Exception, match="uninitialized"):
        runner.run_copy_plan(plan)
