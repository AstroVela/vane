# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pickle

import pytest

import vane


def _require_ray_cxx():
    ray_cxx = getattr(vane, "ray_cxx", None)
    if ray_cxx is None or not hasattr(ray_cxx, "PyLogicalPlan"):
        pytest.skip("vane.ray_cxx.PyLogicalPlan not available in this environment")
    return ray_cxx


def _table_from_native_result(result):
    pa = pytest.importorskip("pyarrow")

    payloads = list(result.partition_payloads)
    assert payloads
    if len(payloads) == 1:
        return payloads[0]
    return pa.concat_tables(payloads)


def test_logical_plan_replays_connection_snapshot_on_to_physical_plan():
    ray_cxx = _require_ray_cxx()

    source_conn = vane.connect()
    source_conn.execute("SET threads=3")
    source_conn.execute("SET TimeZone='UTC'")
    relation = source_conn.sql("SELECT * FROM (VALUES (1), (2), (3)) AS t(a)")

    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "snapshot-to-physical")

    target_conn = vane.connect()
    assert target_conn.execute("SELECT current_setting('threads')").fetchone()[0] != 3
    assert target_conn.execute("SELECT current_setting('TimeZone')").fetchone()[0] != "UTC"

    plan.to_physical_plan(target_conn)

    assert target_conn.execute("SELECT current_setting('threads')").fetchone()[0] == 3
    assert target_conn.execute("SELECT current_setting('TimeZone')").fetchone()[0] == "UTC"


def test_pickled_physical_plan_replays_connection_snapshot_on_execute_native():
    ray_cxx = _require_ray_cxx()

    source_conn = vane.connect()
    source_conn.execute("SET threads=3")
    source_conn.execute("SET TimeZone='UTC'")
    relation = source_conn.sql("SELECT * FROM (VALUES (1), (2), (3)) AS t(a)")

    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "snapshot-execute-native")
    physical_plan = plan.to_physical_plan(vane.connect())
    restored_plan = pickle.loads(pickle.dumps(physical_plan))

    worker_cursor = vane.connect().cursor()
    assert worker_cursor.execute("SELECT current_setting('threads')").fetchone()[0] != 3
    assert worker_cursor.execute("SELECT current_setting('TimeZone')").fetchone()[0] != "UTC"

    result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(worker_cursor, restored_plan)
    table = _table_from_native_result(result)

    assert table.num_rows == 3
    assert worker_cursor.execute("SELECT current_setting('threads')").fetchone()[0] == 3
    assert worker_cursor.execute("SELECT current_setting('TimeZone')").fetchone()[0] == "UTC"


def test_pickled_physical_plan_replays_bootstrap_and_runtime_connection_snapshot():
    ray_cxx = _require_ray_cxx()

    source_conn = vane.connect(config={"custom_user_agent": "snapshot-test"})
    source_conn.execute("SET TimeZone='UTC'")
    relation = source_conn.sql(
        "SELECT current_setting('custom_user_agent') AS user_agent, current_setting('TimeZone') AS timezone"
    )

    target_conn = vane.connect()
    assert target_conn.execute("SELECT current_setting('custom_user_agent')").fetchone()[0] == ""
    assert target_conn.execute("SELECT current_setting('TimeZone')").fetchone()[0] != "UTC"

    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "snapshot-bootstrap-runtime")
    physical_plan = plan.to_physical_plan(target_conn)
    restored_plan = pickle.loads(pickle.dumps(physical_plan))

    worker_cursor = vane.connect().cursor()
    result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(worker_cursor, restored_plan)
    table = _table_from_native_result(result)

    assert table.column(0).to_pylist() == ["snapshot-test"]
    assert table.column(1).to_pylist() == ["UTC"]
