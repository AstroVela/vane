# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Modified Ray e2e test that focuses on serialization verification.
Tests the core serialization functionality without requiring full execution pipeline.
"""

import pytest

try:
    import ray
except Exception:
    ray = None

import vane


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_ray_plan_serialization_core():
    """Test that plans can be created and serialized for Ray distribution."""
    # Use pure SQL to avoid pandas_scan serialization issues
    n = 12
    values_list = [f"({i}, {i * 10})" for i in range(n)]
    values_clause = ", ".join(values_list)
    sql = f"SELECT * FROM (VALUES {values_clause}) AS t(a, b)"
    df = vane.sql(sql)

    # vane.sql(...) returns a DuckDB relation
    rel = df

    # Create PyLogicalPlan (this triggers LogicalPlan serialization)
    ray_cxx = getattr(vane, "ray_cxx", None)
    if ray_cxx is None or not hasattr(ray_cxx, "PyLogicalPlan"):
        pytest.skip("vane.ray_cxx.PyLogicalPlan not available in this environment")
    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(rel, "test-e2e-query")

    assert plan.idx() == "test-e2e-query"
    assert plan.idx() == "test-e2e-query"

    # Test pickling (this is what Ray uses for serialization)
    import pickle

    serialized = pickle.dumps(plan)
    assert len(serialized) > 0

    # Test unpickling in same process
    restored_plan = pickle.loads(serialized)
    assert restored_plan.idx() == "test-e2e-query"
    conn = vane.connect()
    restored_dist_plan = restored_plan.to_physical_plan(conn)
    assert restored_dist_plan.num_partitions() >= 1

    # Test cross-worker serialization
    @ray.remote
    def verify_plan_in_worker(plan):
        """Verify plan can be received and accessed in Ray worker."""
        import vane

        assert plan.idx() == "test-e2e-query"
        conn = vane.connect()
        dist_plan = plan.to_physical_plan(conn)
        assert dist_plan.num_partitions() >= 1
        return {"success": True, "idx": plan.idx(), "num_partitions": dist_plan.num_partitions()}

    result_ref = verify_plan_in_worker.remote(plan)
    result = ray.get(result_ref, timeout=10)

    assert result["success"] is True
    assert result["idx"] == "test-e2e-query"
    assert result["num_partitions"] >= 1


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_streaming_metadata_and_rows_full_execution(tmp_path):
    """Execute a serialized Parquet-backed plan through the real Ray runner."""
    import pyarrow as pa

    input_path = tmp_path / "ray_serialization_execution.parquet"
    connection = vane.connect()
    connection.execute(
        f"""
        COPY (
            SELECT i::INTEGER AS value, (i * 10)::INTEGER AS scaled
            FROM range(12) AS t(i)
        ) TO '{input_path}' (FORMAT PARQUET)
        """
    )
    relation = connection.sql(f"SELECT value, scaled FROM read_parquet('{input_path}') WHERE value % 2 = 0")

    from vane import runners

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    parts = list(runner.run_iter_tables(relation, results_buffer_size=1))
    tables = [part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts]
    result = pa.concat_tables(tables)

    assert sorted(zip(result.column(0).to_pylist(), result.column(1).to_pylist())) == [
        (0, 0),
        (2, 20),
        (4, 40),
        (6, 60),
        (8, 80),
        (10, 100),
    ]
