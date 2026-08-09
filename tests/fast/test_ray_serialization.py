# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Test that LogicalPlan can be serialized and deserialized across Ray workers.
This is a focused test that validates the serialization fix without
requiring full execution pipeline.
"""

import pytest

try:
    import ray
except Exception:
    ray = None

import pickle

import vane


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_logical_plan_serialization():
    """Test that LogicalPlan serialization works in same process."""
    # Create a simple SQL relation (no pandas to avoid serialization issues)
    sql = "SELECT * FROM (VALUES (0,0), (1,10), (2,20)) AS t(a, b)"
    df = vane.sql(sql)

    # vane.sql(...) returns a DuckDB relation
    rel = df

    ray_cxx = getattr(vane, "ray_cxx", None)
    if ray_cxx is None or not hasattr(ray_cxx, "PyLogicalPlan"):
        pytest.skip("vane.ray_cxx.PyLogicalPlan not available in this environment")
    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(rel, "test-query-id")

    assert plan.idx() == "test-query-id"
    assert plan.idx() == "test-query-id"

    # Serialize and deserialize
    serialized = pickle.dumps(plan)
    assert len(serialized) > 0

    restored_plan = pickle.loads(serialized)
    assert restored_plan.idx() == "test-query-id"
    assert restored_plan.idx() == "test-query-id"

    print("✅ LogicalPlan serialization test passed!")


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_logical_plan_serialization_across_ray_workers():
    """Test that LogicalPlan can be serialized and sent to Ray workers."""
    # Create a simple SQL relation
    sql = "SELECT * FROM (VALUES (0,0), (1,10), (2,20)) AS t(a, b)"
    df = vane.sql(sql)

    # vane.sql(...) returns a DuckDB relation
    rel = df

    ray_cxx = getattr(vane, "ray_cxx", None)
    if ray_cxx is None or not hasattr(ray_cxx, "PyLogicalPlan"):
        pytest.skip("vane.ray_cxx.PyLogicalPlan not available in this environment")
    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(rel, "test-ray-serialization")

    # Define a remote function that receives the plan
    @ray.remote
    def verify_plan_in_worker(plan):
        """Verify that plan can be received and accessed in worker."""
        import vane

        conn = vane.connect()
        dist_plan = plan.to_physical_plan(conn)
        # Plan should be deserialized successfully
        assert plan.idx() == "test-ray-serialization"
        assert dist_plan.num_partitions() >= 1
        return {"success": True, "idx": plan.idx(), "num_partitions": dist_plan.num_partitions()}

    # Send plan to Ray worker
    result_ref = verify_plan_in_worker.remote(plan)
    result = ray.get(result_ref)

    assert result["success"] is True
    assert result["idx"] == "test-ray-serialization"
    assert result["num_partitions"] >= 1

    print("✅ Cross-worker serialization test passed!")
