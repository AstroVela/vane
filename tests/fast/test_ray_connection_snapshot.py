# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import gc
import pickle

import pytest

import duckdb


def _require_ray_cxx():
    ray_cxx = getattr(duckdb, "ray_cxx", None)
    if ray_cxx is None or not hasattr(ray_cxx, "PyLogicalPlan"):
        pytest.skip("duckdb.ray_cxx.PyLogicalPlan not available in this environment")
    return ray_cxx


def _table_from_native_result(result):
    pa = pytest.importorskip("pyarrow")

    payloads = list(result.partition_payloads)
    assert payloads
    if len(payloads) == 1:
        return payloads[0]
    return pa.concat_tables(payloads)


def test_logical_plan_captures_connection_scoped_vane_session(monkeypatch):
    ray_cxx = _require_ray_cxx()

    monkeypatch.setenv("AWS_REVIEW_SESSION_SECRET_75", "session-a-secret")
    connection_a = duckdb.connect()
    cursor_a = connection_a.cursor()

    monkeypatch.delenv("AWS_REVIEW_SESSION_SECRET_75")
    connection_b = duckdb.connect()

    plan_a = ray_cxx.PyLogicalPlan.from_duckdb_relation(connection_a.sql("SELECT 1"), "session-a")
    cursor_plan_a = ray_cxx.PyLogicalPlan.from_duckdb_relation(cursor_a.sql("SELECT 1"), "session-a-cursor")
    plan_b = ray_cxx.PyLogicalPlan.from_duckdb_relation(connection_b.sql("SELECT 1"), "session-b")

    assert plan_a.session_id()
    assert plan_a.session_id() == cursor_plan_a.session_id()
    assert plan_a.session_id() != plan_b.session_id()
    assert plan_a.session_config()["AWS_REVIEW_SESSION_SECRET_75"] == "session-a-secret"
    assert "AWS_REVIEW_SESSION_SECRET_75" not in plan_b.session_config()

    restored_plan = pickle.loads(pickle.dumps(plan_a.to_physical_plan(duckdb.connect())))
    assert restored_plan.session_id() == plan_a.session_id()
    assert restored_plan.session_config() == plan_a.session_config()


def test_datasource_relation_retains_connection_scoped_vane_session(monkeypatch):
    from duckdb.datasource import DataSource, DataSourceTask, read_datasource

    ray_cxx = _require_ray_cxx()

    class SnapshotTask(DataSourceTask):
        def execute(self):
            return iter(())

    class SnapshotSource(DataSource):
        @property
        def schema(self):
            return {"value": "INTEGER"}

        def get_tasks(self):
            return iter((SnapshotTask(),))

    monkeypatch.setenv("AWS_DATASOURCE_SESSION_SECRET_75", "datasource-secret")
    connection = duckdb.connect()
    relation = read_datasource(SnapshotSource(), con=connection)
    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "datasource-session")

    assert plan.session_id()
    assert plan.session_config()["AWS_DATASOURCE_SESSION_SECRET_75"] == "datasource-secret"


def test_session_aws_settings_replay_only_on_the_target_connection_context(monkeypatch):
    ray_cxx = _require_ray_cxx()

    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://minio-a.internal:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "session-a-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "session-a-secret")
    connection_a = duckdb.connect()
    plan_a = ray_cxx.PyLogicalPlan.from_duckdb_relation(connection_a.sql("SELECT 1"), "session-a-settings")

    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://minio-b.internal:9443")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "session-b-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "session-b-secret")
    connection_b = duckdb.connect()
    plan_b = ray_cxx.PyLogicalPlan.from_duckdb_relation(connection_b.sql("SELECT 1"), "session-b-settings")

    for key in ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(key)
    target_root = duckdb.connect()
    target_a = target_root.cursor()
    target_b = target_root.cursor()

    plan_a.to_physical_plan(target_a)
    plan_b.to_physical_plan(target_b)

    assert target_a.execute("SELECT current_setting('s3_endpoint')").fetchone()[0] == "minio-a.internal:9000"
    assert target_a.execute("SELECT current_setting('s3_access_key_id')").fetchone()[0] == "session-a-key"
    assert target_a.execute("SELECT current_setting('s3_use_ssl')").fetchone()[0] is False
    assert target_b.execute("SELECT current_setting('s3_endpoint')").fetchone()[0] == "minio-b.internal:9443"
    assert target_b.execute("SELECT current_setting('s3_access_key_id')").fetchone()[0] == "session-b-key"
    assert target_b.execute("SELECT current_setting('s3_use_ssl')").fetchone()[0] is True


def test_cursor_plan_marks_owning_connection_session_for_close(monkeypatch):
    ray_cxx = _require_ray_cxx()
    closed_session_ids = []
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setattr(
        "duckdb.runners.ray.runner.notify_connection_closed",
        closed_session_ids.append,
    )

    connection = duckdb.connect()
    cursor = connection.cursor()
    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(cursor.sql("SELECT 1"), "cursor-session-close")
    session_id = plan.session_id()

    cursor.close()
    assert closed_session_ids == []

    connection.close()
    assert closed_session_ids == [session_id]


def test_cursor_keeps_vane_session_alive_after_root_connection_is_collected(monkeypatch):
    ray_cxx = _require_ray_cxx()
    closed_session_ids = []
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setattr(
        "duckdb.runners.ray.runner.notify_connection_closed",
        closed_session_ids.append,
    )

    connection = duckdb.connect()
    cursor = connection.cursor()
    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(cursor.sql("SELECT 1"), "cursor-session-gc")
    session_id = plan.session_id()

    del connection
    gc.collect()
    assert closed_session_ids == []

    cursor.close()
    assert closed_session_ids == [session_id]


def test_connection_close_notification_failure_remains_retryable(monkeypatch):
    ray_cxx = _require_ray_cxx()
    closed_session_ids = []
    monkeypatch.setenv("VANE_RUNNER", "ray")

    def _notify(session_id):
        closed_session_ids.append(session_id)
        if len(closed_session_ids) == 1:
            raise RuntimeError("planned session close notification failure")

    monkeypatch.setattr(
        "duckdb.runners.ray.runner.notify_connection_closed",
        _notify,
    )

    connection = duckdb.connect()
    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(connection.sql("SELECT 1"), "session-close-retry")
    session_id = plan.session_id()

    with pytest.raises(RuntimeError, match="planned session close notification failure"):
        connection.close()
    connection.close()

    assert closed_session_ids == [session_id, session_id]


def test_local_plan_snapshot_does_not_open_a_ray_session(monkeypatch):
    ray_cxx = _require_ray_cxx()
    closed_session_ids = []
    monkeypatch.setenv("VANE_RUNNER", "local")
    monkeypatch.setattr(
        "duckdb.runners.ray.runner.notify_connection_closed",
        closed_session_ids.append,
    )

    connection = duckdb.connect()
    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(connection.sql("SELECT 1"), "local-session")

    assert plan.session_id()
    connection.close()
    assert closed_session_ids == []


def test_deserialized_plan_rejects_missing_session_config():
    ray_cxx = _require_ray_cxx()
    connection = duckdb.connect()
    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(connection.sql("SELECT 1"), "missing-session-config")
    state = list(plan.__getstate__())
    state[3] = {"vane_session": {"id": plan.session_id()}}

    malformed_plan = ray_cxx.PyLogicalPlan.__new__(ray_cxx.PyLogicalPlan)
    with pytest.raises(Exception, match="Vane session is missing config"):
        malformed_plan.__setstate__(tuple(state))


def test_plan_pickles_reject_pre_session_state_shapes():
    ray_cxx = _require_ray_cxx()
    connection = duckdb.connect()
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(connection.sql("SELECT 1"), "strict-session-state")
    logical_state = logical_plan.__getstate__()
    malformed_logical = ray_cxx.PyLogicalPlan.__new__(ray_cxx.PyLogicalPlan)

    with pytest.raises(Exception, match="Invalid state for PyLogicalPlan"):
        malformed_logical.__setstate__(logical_state[:3])

    physical_plan = logical_plan.to_physical_plan(duckdb.connect())
    physical_state = physical_plan.__getstate__()
    malformed_physical = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)

    with pytest.raises(Exception, match="Invalid state for PyPhysicalPlanWrapper pickle"):
        malformed_physical.__setstate__(physical_state[:6])

    missing_resource_owner = list(physical_state)
    missing_resource_owner[3] = ""
    with pytest.raises(Exception, match="resource_query_id must not be empty"):
        malformed_physical.__setstate__(tuple(missing_resource_owner))


def test_physical_plan_replay_state_has_query_lifecycle(monkeypatch):
    ray_cxx = _require_ray_cxx()
    query_id = "query-replay-lifecycle"

    monkeypatch.setenv("AWS_QUERY_REPLAY_SECRET", "session-a")
    connection_a = duckdb.connect()
    plan_a = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        connection_a.sql("SELECT 1"),
        "plan-replay-source-a",
    ).to_physical_plan(duckdb.connect())

    monkeypatch.setenv("AWS_QUERY_REPLAY_SECRET", "session-b")
    connection_b = duckdb.connect()
    plan_b = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        connection_b.sql("SELECT 1"),
        "plan-replay-source-b",
    ).to_physical_plan(duckdb.connect())

    restored_plan_a = pickle.loads(pickle.dumps(plan_a))
    assert restored_plan_a.idx() != query_id
    assert restored_plan_a.resource_query_id() == plan_a.idx()
    assert ray_cxx._lookup_query_connection_snapshot(query_id) is None

    try:
        assert ray_cxx._register_query_python_replay_state(query_id, restored_plan_a) is True
        assert ray_cxx._register_query_python_replay_state(query_id, restored_plan_a) is False
        assert (
            ray_cxx._lookup_query_connection_snapshot(query_id)["vane_session"]["config"]["AWS_QUERY_REPLAY_SECRET"]
            == "session-a"
        )

        with pytest.raises(Exception, match="different Vane session"):
            ray_cxx._register_query_python_replay_state(query_id, plan_b)
    finally:
        ray_cxx._cleanup_query_python_replay_state(query_id)

    assert ray_cxx._lookup_query_connection_snapshot(query_id) is None


def test_query_replay_state_rejects_different_python_runtime_fields():
    ray_cxx = _require_ray_cxx()
    connection = duckdb.connect()
    source_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        connection.sql("SELECT 1"),
        "plan-replay-runtime-fields",
    ).to_physical_plan(duckdb.connect())
    source_state = list(source_plan.__getstate__())

    def plan_with(*, registrations, actor_handles):
        state = list(source_state)
        state[4] = registrations
        state[5] = actor_handles
        plan = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)
        plan.__setstate__(tuple(state))
        return plan

    registrations_query_id = "query-replay-registration-conflict"
    registrations_a = plan_with(registrations=[{"digest": "a"}], actor_handles=None)
    registrations_b = plan_with(registrations=[{"digest": "b"}], actor_handles=None)
    try:
        assert ray_cxx._register_query_python_replay_state(registrations_query_id, registrations_a) is True
        with pytest.raises(Exception, match="different Python UDF registrations"):
            ray_cxx._register_query_python_replay_state(registrations_query_id, registrations_b)
    finally:
        ray_cxx._cleanup_query_python_replay_state(registrations_query_id)

    handles_query_id = "query-replay-actor-handle-conflict"
    handles_a = plan_with(registrations=None, actor_handles={"node": {"handle": "a"}})
    handles_b = plan_with(registrations=None, actor_handles={"node": {"handle": "b"}})
    try:
        assert ray_cxx._register_query_python_replay_state(handles_query_id, handles_a) is True
        with pytest.raises(Exception, match="different Python UDF actor handles"):
            ray_cxx._register_query_python_replay_state(handles_query_id, handles_b)
    finally:
        ray_cxx._cleanup_query_python_replay_state(handles_query_id)


def test_logical_plan_replays_connection_snapshot_on_to_physical_plan():
    ray_cxx = _require_ray_cxx()

    source_conn = duckdb.connect()
    source_conn.execute("SET threads=3")
    source_conn.execute("SET TimeZone='UTC'")
    relation = source_conn.sql("SELECT * FROM (VALUES (1), (2), (3)) AS t(a)")

    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "snapshot-to-physical")

    target_conn = duckdb.connect()
    assert target_conn.execute("SELECT current_setting('threads')").fetchone()[0] != 3
    assert target_conn.execute("SELECT current_setting('TimeZone')").fetchone()[0] != "UTC"

    plan.to_physical_plan(target_conn)

    assert target_conn.execute("SELECT current_setting('threads')").fetchone()[0] == 3
    assert target_conn.execute("SELECT current_setting('TimeZone')").fetchone()[0] == "UTC"


def test_pickled_physical_plan_replays_connection_snapshot_on_execute_native():
    ray_cxx = _require_ray_cxx()

    source_conn = duckdb.connect()
    source_conn.execute("SET threads=3")
    source_conn.execute("SET TimeZone='UTC'")
    relation = source_conn.sql("SELECT * FROM (VALUES (1), (2), (3)) AS t(a)")

    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "snapshot-execute-native")
    physical_plan = plan.to_physical_plan(duckdb.connect())
    restored_plan = pickle.loads(pickle.dumps(physical_plan))

    worker_cursor = duckdb.connect().cursor()
    assert worker_cursor.execute("SELECT current_setting('threads')").fetchone()[0] != 3
    assert worker_cursor.execute("SELECT current_setting('TimeZone')").fetchone()[0] != "UTC"

    result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(worker_cursor, restored_plan)
    table = _table_from_native_result(result)

    assert table.num_rows == 3
    assert worker_cursor.execute("SELECT current_setting('threads')").fetchone()[0] == 3
    assert worker_cursor.execute("SELECT current_setting('TimeZone')").fetchone()[0] == "UTC"


def test_pickled_physical_plan_replays_bootstrap_and_runtime_connection_snapshot():
    ray_cxx = _require_ray_cxx()

    source_conn = duckdb.connect(config={"custom_user_agent": "snapshot-test"})
    source_conn.execute("SET TimeZone='UTC'")
    relation = source_conn.sql(
        "SELECT current_setting('custom_user_agent') AS user_agent, current_setting('TimeZone') AS timezone"
    )

    target_conn = duckdb.connect()
    assert target_conn.execute("SELECT current_setting('custom_user_agent')").fetchone()[0] == ""
    assert target_conn.execute("SELECT current_setting('TimeZone')").fetchone()[0] != "UTC"

    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "snapshot-bootstrap-runtime")
    physical_plan = plan.to_physical_plan(target_conn)
    restored_plan = pickle.loads(pickle.dumps(physical_plan))

    worker_cursor = duckdb.connect().cursor()
    result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(worker_cursor, restored_plan)
    table = _table_from_native_result(result)

    assert table.column(0).to_pylist() == ["snapshot-test"]
    assert table.column(1).to_pylist() == ["UTC"]
