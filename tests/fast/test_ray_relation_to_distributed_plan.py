# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

try:
    import ray
except Exception:
    ray = None

import vane
from vane import runners as _runners


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_relation_to_distributed_plan():
    try:
        _runners.set_runner_ray()
    except Exception:
        pytest.skip("duckdb runner API not available in this environment")

    # Use pure SQL relation to avoid pandas_scan serialization limitations.
    df = vane.sql("SELECT * FROM (VALUES (1,4), (2,5), (3,6)) AS t(x, y)")

    # instantiate the Python RayRunner directly (compiled runner wrapper may not expose
    # the Python-only helper method)
    from vane.runners.ray.runner import RayRunner

    runner = RayRunner(address=None, max_task_backlog=None)
    assert getattr(runner, "name", None) == "ray"

    # vane.sql(...) returns a DuckDB relation
    relation = df

    # Build PyLogicalPlan from relation, then materialize DistributedPhysicalPlan on the driver.
    ray_cxx = getattr(vane, "ray_cxx", None)
    if ray_cxx is None or not hasattr(ray_cxx, "PyLogicalPlan"):
        pytest.skip("vane.ray_cxx.PyLogicalPlan not available in this environment")
    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "test-relation-plan")
    assert plan is not None
    assert plan.idx() == "test-relation-plan"

    conn = vane.connect()
    distributed_plan = plan.to_physical_plan(conn)
    assert distributed_plan is not None
    assert hasattr(distributed_plan, "num_partitions")
    assert distributed_plan.num_partitions() >= 1


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_run_write_parquet_does_not_crash_without_transient_relation_owner(tmp_path):
    try:
        _runners.set_runner_ray(noop_if_initialized=True)
    except Exception:
        pytest.skip("duckdb runner API not available in this environment")

    output_path = tmp_path / "write_output.parquet"
    con = vane.connect()
    try:
        con.sql("SELECT 1 AS a").write_parquet(str(output_path))
        assert output_path.exists()
        assert con.sql(f"SELECT * FROM read_parquet('{output_path}')").fetchall() == [(1,)]
    finally:
        con.close()


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_scan_tasks_grouped_by_distributed_worker_slots(monkeypatch, tmp_path):
    try:
        _runners.set_runner_ray()
    except Exception:
        pytest.skip("duckdb runner API not available in this environment")

    ray_cxx = getattr(vane, "ray_cxx", None)
    if ray_cxx is None or not hasattr(ray_cxx, "PyLogicalPlan"):
        pytest.skip("vane.ray_cxx.PyLogicalPlan not available in this environment")

    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "3")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "6")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MIN_BYTES", "1")

    source_path = tmp_path / "scan_grouping_input"
    con = vane.connect()
    try:
        con.execute(f"""
            COPY (
                SELECT
                    i::INTEGER AS id,
                    (i % 12)::INTEGER AS file_id
                FROM range(0, 120) tbl(i)
            ) TO '{source_path}' (FORMAT PARQUET, PARTITION_BY (file_id))
        """)

        relation = con.sql(f"SELECT id FROM read_parquet('{source_path}/**/*.parquet')")
        logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "scan-grouping-plan")
        distributed_plan = logical_plan.to_physical_plan(con)
        scan_task_descriptors = distributed_plan.scan_task_descriptor_map()
        assert scan_task_descriptors, "expected non-empty scan task descriptor map"

        descriptor_counts = [len(descriptors) for descriptors in scan_task_descriptors.values()]
        assert descriptor_counts == [6], (
            f"expected exactly one scan node with 6 scan task descriptors, got {descriptor_counts}"
        )
    finally:
        con.close()


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_scan_tasks_can_disable_size_grouping(monkeypatch, tmp_path):
    try:
        _runners.set_runner_ray()
    except Exception:
        pytest.skip("duckdb runner API not available in this environment")

    ray_cxx = getattr(vane, "ray_cxx", None)
    if ray_cxx is None or not hasattr(ray_cxx, "PyLogicalPlan"):
        pytest.skip("vane.ray_cxx.PyLogicalPlan not available in this environment")

    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "3")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "6")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MIN_BYTES", "1GB")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MAX_BYTES", "2GB")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_SIZE_GROUPING", "0")

    source_path = tmp_path / "scan_disable_size_grouping_input"
    con = vane.connect()
    try:
        con.execute(f"""
            COPY (
                SELECT
                    i::INTEGER AS id,
                    (i % 12)::INTEGER AS file_id
                FROM range(0, 120) tbl(i)
            ) TO '{source_path}' (FORMAT PARQUET, PARTITION_BY (file_id))
        """)

        relation = con.sql(f"SELECT id FROM read_parquet('{source_path}/**/*.parquet')")
        logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "scan-disable-size-grouping-plan")
        distributed_plan = logical_plan.to_physical_plan(con)
        scan_task_descriptors = distributed_plan.scan_task_descriptor_map()
        assert scan_task_descriptors, "expected non-empty scan task descriptor map"

        descriptor_counts = [len(descriptors) for descriptors in scan_task_descriptors.values()]
        assert descriptor_counts == [6], (
            f"expected count-based grouping when VANE_RAY_SCAN_TASK_SIZE_GROUPING=0, got {descriptor_counts}"
        )
    finally:
        con.close()


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_scan_task_min_partition_num_can_exceed_worker_slots(monkeypatch, tmp_path):
    try:
        _runners.set_runner_ray()
    except Exception:
        pytest.skip("duckdb runner API not available in this environment")

    ray_cxx = getattr(vane, "ray_cxx", None)
    if ray_cxx is None or not hasattr(ray_cxx, "PyLogicalPlan"):
        pytest.skip("vane.ray_cxx.PyLogicalPlan not available in this environment")

    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "1")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "2")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MIN_PARTITION_NUM", "8")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_SIZE_GROUPING", "0")

    source_path = tmp_path / "scan_min_partition_input"
    con = vane.connect()
    try:
        con.execute(f"""
            COPY (
                SELECT
                    i::INTEGER AS id,
                    (i % 8)::INTEGER AS file_id
                FROM range(0, 80) tbl(i)
            ) TO '{source_path}' (FORMAT PARQUET, PARTITION_BY (file_id))
        """)

        relation = con.sql(f"SELECT id FROM read_parquet('{source_path}/**/*.parquet')")
        logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "scan-min-partition-plan")
        distributed_plan = logical_plan.to_physical_plan(con)
        scan_task_descriptors = distributed_plan.scan_task_descriptor_map()
        assert scan_task_descriptors, "expected non-empty scan task descriptor map"

        descriptor_counts = [len(descriptors) for descriptors in scan_task_descriptors.values()]
        assert descriptor_counts == [8], (
            f"expected min partition count above worker slots to produce 8 descriptors, got {descriptor_counts}"
        )
    finally:
        con.close()


@pytest.mark.skipif(ray is None, reason="ray not installed")
def test_ray_runner_sets_scan_task_backlog_env(monkeypatch):
    from vane.runners.ray import runner as ray_runner_mod

    monkeypatch.delenv("VANE_RAY_MAX_TASK_BACKLOG", raising=False)
    monkeypatch.setattr(ray_runner_mod.ray, "is_initialized", lambda: True)

    ray_runner_mod.RayRunner(address=None, max_task_backlog=7)

    assert os.environ["VANE_RAY_MAX_TASK_BACKLOG"] == "7"
