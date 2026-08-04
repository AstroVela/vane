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


def test_vllm_named_actor_pool_identity_includes_connection_session():
    from vane.ai.providers.vllm import _build_native_vllm_options_argument

    ray_cxx = _require_ray_cxx()
    connection_a = duckdb.connect()
    connection_b = duckdb.connect()
    query_id = "reused-query-id"
    options = _build_native_vllm_options_argument({"use_ray": True})

    def build_relation(connection):
        source = connection.sql("SELECT 'hello' AS prompt")
        generated = duckdb.FunctionExpression(
            "vllm",
            duckdb.ColumnExpression("prompt"),
            duckdb.ConstantExpression("test-model"),
            duckdb.ConstantExpression(options),
        ).alias("generated")
        return source.select(generated)

    plan_a = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        build_relation(connection_a),
        query_id,
    ).to_physical_plan(connection_a)
    plan_b = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        build_relation(connection_b),
        query_id,
    ).to_physical_plan(connection_b)

    nodes_a = plan_a.collect_vllm_nodes(conn=connection_a)
    nodes_b = plan_b.collect_vllm_nodes(conn=connection_b)

    assert len(nodes_a) == 1
    assert len(nodes_b) == 1
    assert nodes_a[0]["pool_name"] != nodes_b[0]["pool_name"]
    assert plan_a.session_id() in nodes_a[0]["pool_name"]
    assert plan_b.session_id() in nodes_b[0]["pool_name"]


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


@pytest.mark.parametrize(
    ("configured_runner", "environment_runner", "expects_close"),
    [
        ("ray", "local", True),
        ("local", "ray", False),
    ],
)
def test_plan_snapshot_uses_configured_runner_for_session_lifecycle(
    monkeypatch,
    configured_runner,
    environment_runner,
    expects_close,
):
    ray_cxx = _require_ray_cxx()
    closed_session_ids = []
    monkeypatch.setenv("VANE_RUNNER", environment_runner)
    monkeypatch.setattr(
        duckdb.vane_runners_cpp,
        "get_or_infer_runner_type",
        lambda: configured_runner,
    )
    monkeypatch.setattr(
        "duckdb.runners.ray.runner.notify_connection_closed",
        closed_session_ids.append,
    )

    connection = duckdb.connect()
    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(connection.sql("SELECT 1"), "configured-runner-session")
    session_id = plan.session_id()

    connection.close()

    assert closed_session_ids == ([session_id] if expects_close else [])


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


def test_snapshot_replay_rejects_non_static_extensions_without_installing(tmp_path):
    ray_cxx = _require_ray_cxx()

    source_conn = duckdb.connect()
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_conn.sql("SELECT 1"),
        "snapshot-missing-extension",
    )
    physical_plan = logical_plan.to_physical_plan(duckdb.connect())
    state = list(physical_plan.__getstate__())
    snapshot = dict(state[6])
    snapshot["extensions"] = ["sqlite_scanner"]
    state[6] = snapshot

    replay_plan = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)
    replay_plan.__setstate__(tuple(state))

    extension_directory = tmp_path / "extensions"
    worker_connection = duckdb.connect(
        config={
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
            "extension_directory": str(extension_directory),
        }
    )
    worker_connection.execute("SET custom_extension_repository = 'http://127.0.0.1:9'")

    with pytest.raises(Exception) as exc_info:
        ray_cxx.DistributedPhysicalPlanRunner().execute_native(
            worker_connection.cursor(),
            replay_plan,
        )

    message = str(exc_info.value)
    assert "supports only statically linked extensions" in message
    assert "sqlite_scanner" in message
    assert "Failed to download extension" not in message
    assert not extension_directory.exists()


def test_snapshot_bootstrap_is_sanitized_before_connect(tmp_path):
    ray_cxx = _require_ray_cxx()

    source_conn = duckdb.connect()
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_conn.sql("SELECT 1"),
        "snapshot-bootstrap-extension-security",
    )
    physical_plan = logical_plan.to_physical_plan(duckdb.connect())
    state = list(physical_plan.__getstate__())
    snapshot = dict(state[6])
    extension_directory = tmp_path / "bootstrap-extensions"
    snapshot["bootstrap"] = {
        "database": ":memory:",
        "read_only": False,
        "config": {
            "allow_unsigned_extensions": "true",
            "autoinstall_known_extensions": "true",
            "autoload_known_extensions": "true",
            "custom_extension_repository": "http://127.0.0.1:9",
            "extension_directory": str(extension_directory),
            "sqlite_all_varchar": "true",
        },
    }
    snapshot["extensions"] = ["sqlite_scanner"]
    state[6] = snapshot

    replay_plan = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)
    replay_plan.__setstate__(tuple(state))

    with pytest.raises(Exception) as exc_info:
        ray_cxx.DistributedPhysicalPlanRunner().execute_native(
            duckdb.connect().cursor(),
            replay_plan,
        )

    assert not extension_directory.exists()
    message = str(exc_info.value)
    assert "sqlite_all_varchar" in message
    assert "sqlite_scanner" in message
    assert "not statically linked" in message


def test_snapshot_bootstrap_applies_static_extension_settings_after_connect(tmp_path):
    ray_cxx = _require_ray_cxx()

    source_conn = duckdb.connect()
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_conn.sql(
            """
            SELECT
                CAST(current_setting('http_timeout') AS BIGINT) AS timeout,
                CAST(current_setting('allow_unsigned_extensions') AS BOOLEAN) AS allow_unsigned,
                CAST(current_setting('autoinstall_known_extensions') AS BOOLEAN) AS autoinstall,
                CAST(current_setting('autoload_known_extensions') AS BOOLEAN) AS autoload
            """
        ),
        "snapshot-bootstrap-static-extension-setting",
    )
    state = list(logical_plan.__getstate__())
    snapshot = dict(state[3])
    extension_directory = tmp_path / "bootstrap-extensions"
    snapshot["bootstrap"] = {
        "database": ":memory:",
        "read_only": False,
        "config": {
            "allow_unsigned_extensions": "true",
            "autoinstall_known_extensions": "true",
            "autoload_known_extensions": "true",
            "custom_extension_repository": "http://127.0.0.1:9",
            "extension_directory": str(extension_directory),
            "http_timeout": "41",
        },
    }
    snapshot["settings"] = [
        setting for setting in snapshot.get("settings", []) if setting.get("name", "").lower() != "http_timeout"
    ]
    state[3] = snapshot

    replay_logical_plan = ray_cxx.PyLogicalPlan.__new__(ray_cxx.PyLogicalPlan)
    replay_logical_plan.__setstate__(tuple(state))
    physical_plan = replay_logical_plan.to_physical_plan(duckdb.connect())
    restored_plan = pickle.loads(pickle.dumps(physical_plan))
    result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        duckdb.connect().cursor(),
        restored_plan,
    )

    table = _table_from_native_result(result)
    assert [table.column(index).to_pylist() for index in range(4)] == [
        [41],
        [False],
        [False],
        [False],
    ]
    assert not extension_directory.exists()


def test_snapshot_replay_keeps_extension_security_settings_disabled():
    ray_cxx = _require_ray_cxx()
    snapshot_setting_names = (
        "autoinstall_known_extensions",
        "autoload_known_extensions",
    )
    setting_names = (
        "allow_unsigned_extensions",
        *snapshot_setting_names,
    )
    settings_query = f"""
        SELECT name, value
        FROM duckdb_settings()
        WHERE name IN ({", ".join(repr(name) for name in setting_names)})
        ORDER BY name
    """

    source_connection = duckdb.connect()
    for name in snapshot_setting_names:
        source_connection.execute(f"SET {name} = true")
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_connection.sql("SELECT 1"),
        "snapshot-extension-security-settings",
    )
    physical_plan = logical_plan.to_physical_plan(duckdb.connect())
    state = list(physical_plan.__getstate__())
    snapshot = dict(state[6])
    captured_setting_names = {setting["name"].lower() for setting in snapshot["settings"]}
    assert captured_setting_names.isdisjoint(setting_names)
    snapshot["settings"] = [
        *snapshot["settings"],
        *({"name": name, "value": "true", "input_type": "BOOLEAN"} for name in setting_names),
    ]
    state[6] = snapshot
    replay_plan = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)
    replay_plan.__setstate__(tuple(state))

    worker_connection = duckdb.connect(config={name: "true" for name in setting_names})
    assert dict(worker_connection.execute(settings_query).fetchall()) == {name: "true" for name in setting_names}

    ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        worker_connection.cursor(),
        replay_plan,
    )

    assert dict(worker_connection.execute(settings_query).fetchall()) == {name: "false" for name in setting_names}


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


def test_logical_plan_capture_planning_and_execution_preserve_file_database_security_config(tmp_path):
    ray_cxx = _require_ray_cxx()
    database_path = str(tmp_path / "capture-bootstrap.duckdb")
    setting_names = (
        "allow_unsigned_extensions",
        "autoinstall_known_extensions",
        "autoload_known_extensions",
    )
    settings_query = f"""
        SELECT name, value
        FROM duckdb_settings()
        WHERE name IN ({", ".join(repr(name) for name in setting_names)})
        ORDER BY name
    """
    source_conn = duckdb.connect(
        database_path,
        config={name: "true" for name in setting_names},
    )
    source_settings = dict(source_conn.execute(settings_query).fetchall())

    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_conn.sql("SELECT 1 AS value"),
        "file-security-setting",
    )

    assert logical_plan.idx() == "file-security-setting"
    assert source_settings == {name: "true" for name in setting_names}

    physical_plan = logical_plan.to_physical_plan(duckdb.connect())

    assert physical_plan.idx() == "file-security-setting"
    restored_plan = pickle.loads(pickle.dumps(physical_plan))
    assert dict(source_conn.execute(settings_query).fetchall()) == source_settings
    source_conn.close()
    del logical_plan
    del physical_plan
    gc.collect()

    worker_conn = duckdb.connect(config={name: "true" for name in setting_names})
    assert dict(worker_conn.execute(settings_query).fetchall()) == {name: "true" for name in setting_names}

    restored_result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        worker_conn.cursor(),
        restored_plan,
    )

    assert _table_from_native_result(restored_result).column(0).to_pylist() == [1]


def test_file_database_table_scan_reopens_bootstrap_on_worker(tmp_path):
    ray_cxx = _require_ray_cxx()
    database_path = str(tmp_path / "table-scan-bootstrap.duckdb")
    source_conn = duckdb.connect(database_path)
    source_conn.execute("CREATE TABLE numbers AS SELECT i AS value FROM range(10) data(i)")

    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_conn.sql("SELECT sum(value) AS total FROM numbers"),
        "snapshot-file-table-scan",
    )
    physical_plan = logical_plan.to_physical_plan(duckdb.connect())
    restored_plan = pickle.loads(pickle.dumps(physical_plan))

    result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        duckdb.connect().cursor(),
        restored_plan,
    )

    assert _table_from_native_result(result).column(0).to_pylist() == [45]


def test_effective_session_config_reaches_nondefault_bootstrap_connection(tmp_path):
    ray_cxx = _require_ray_cxx()
    database_path = str(tmp_path / "session-bootstrap.duckdb")
    source_conn = duckdb.connect(database_path)
    source_conn.execute("LOAD httpfs")
    source_conn.execute("SET GLOBAL s3_access_key_id='database-key'")
    source_conn.execute("SET GLOBAL s3_secret_access_key='database-secret'")
    source_conn.execute("SET GLOBAL s3_session_token='database-token'")
    relation = source_conn.sql(
        "SELECT current_setting('s3_access_key_id') AS access_key, "
        "current_setting('s3_secret_access_key') AS secret_key, "
        "current_setting('s3_session_token') AS session_token"
    )
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        "snapshot-effective-session-config",
    )
    assert logical_plan.has_explicit_s3_credentials() is False
    snapshot_setting_names = {setting["name"].lower() for setting in logical_plan.__getstate__()[3]["settings"]}
    assert snapshot_setting_names.isdisjoint(
        {
            "s3_access_key_id",
            "s3_secret_access_key",
            "s3_session_token",
        }
    )
    effective_config = {
        "AWS_ACCESS_KEY_ID": "resolved-profile-key",
        "AWS_SECRET_ACCESS_KEY": "resolved-profile-secret",
        "AWS_SESSION_TOKEN": "resolved-profile-token",
    }

    physical_plan = logical_plan.to_physical_plan(
        duckdb.connect(),
        effective_session_config=effective_config,
    )
    restored_plan = pickle.loads(pickle.dumps(physical_plan))
    result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        duckdb.connect().cursor(),
        restored_plan,
        effective_session_config=effective_config,
    )

    table = _table_from_native_result(result)
    assert [table.column(index).to_pylist() for index in range(3)] == [
        ["resolved-profile-key"],
        ["resolved-profile-secret"],
        ["resolved-profile-token"],
    ]


def test_file_database_snapshot_baseline_ignores_database_target_settings(tmp_path):
    ray_cxx = _require_ray_cxx()
    database_path = str(tmp_path / "read-only-bootstrap.duckdb")
    duckdb.connect(database_path).close()
    source_conn = duckdb.connect(database_path, config={"access_mode": "READ_ONLY"})

    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_conn.sql("SELECT 1 AS value"),
        "snapshot-file-target-settings",
    )
    snapshot_setting_names = {setting["name"].lower() for setting in logical_plan.__getstate__()[3]["settings"]}

    assert snapshot_setting_names.isdisjoint({"access_mode", "temp_directory"})
    physical_plan = logical_plan.to_physical_plan(duckdb.connect())
    result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        duckdb.connect().cursor(),
        physical_plan,
    )
    assert _table_from_native_result(result).column(0).to_pylist() == [1]


def test_explicit_connection_s3_settings_override_effective_session_config():
    ray_cxx = _require_ray_cxx()
    source_conn = duckdb.connect()
    source_conn.execute("LOAD httpfs")
    source_conn.execute("SET s3_access_key_id='explicit-key'")
    source_conn.execute("SET s3_secret_access_key='explicit-secret'")
    source_conn.execute("SET s3_session_token=''")
    relation = source_conn.sql(
        "SELECT current_setting('s3_access_key_id') AS access_key, "
        "current_setting('s3_secret_access_key') AS secret_key, "
        "current_setting('s3_session_token') AS session_token"
    )
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        "snapshot-explicit-s3-precedence",
    )
    assert logical_plan.has_explicit_s3_credentials() is True
    effective_config = {
        "AWS_ACCESS_KEY_ID": "environment-key",
        "AWS_SECRET_ACCESS_KEY": "environment-secret",
        "AWS_SESSION_TOKEN": "environment-token",
    }

    planning_conn = duckdb.connect()
    physical_plan = logical_plan.to_physical_plan(
        planning_conn,
        effective_session_config=effective_config,
    )
    assert physical_plan.has_explicit_s3_credentials() is True
    assert planning_conn.execute("SELECT current_setting('s3_access_key_id')").fetchone()[0] == "explicit-key"

    restored_plan = pickle.loads(pickle.dumps(physical_plan))
    worker_cursor = duckdb.connect().cursor()
    result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        worker_cursor,
        restored_plan,
        effective_session_config=effective_config,
    )

    table = _table_from_native_result(result)
    assert table.column(0).to_pylist() == ["explicit-key"]
    assert table.column(1).to_pylist() == ["explicit-secret"]
    assert table.column(2).to_pylist() == [""]
    assert worker_cursor.execute("SELECT current_setting('s3_access_key_id')").fetchone()[0] == "explicit-key"
    assert worker_cursor.execute("SELECT current_setting('s3_session_token')").fetchone()[0] == ""


def test_connection_snapshot_requires_both_explicit_s3_credential_settings():
    ray_cxx = _require_ray_cxx()
    source_conn = duckdb.connect()
    source_conn.execute("LOAD httpfs")
    source_conn.execute("SET s3_access_key_id='explicit-key'")

    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_conn.sql("SELECT 1"),
        "snapshot-partial-explicit-s3",
    )

    with pytest.raises(
        Exception,
        match="must set both s3_access_key_id and s3_secret_access_key",
    ):
        logical_plan.has_explicit_s3_credentials()


@pytest.mark.parametrize(
    ("access_key", "secret_key"),
    [
        ("explicit-key", ""),
        ("", "explicit-secret"),
    ],
)
def test_connection_snapshot_rejects_explicit_s3_key_pair_with_one_empty_value(
    access_key,
    secret_key,
):
    ray_cxx = _require_ray_cxx()
    source_conn = duckdb.connect()
    source_conn.execute("LOAD httpfs")
    source_conn.execute(f"SET s3_access_key_id='{access_key}'")
    source_conn.execute(f"SET s3_secret_access_key='{secret_key}'")

    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_conn.sql("SELECT 1"),
        "snapshot-empty-partial-explicit-s3",
    )

    with pytest.raises(
        Exception,
        match="must set both s3_access_key_id and s3_secret_access_key",
    ):
        logical_plan.has_explicit_s3_credentials()


def test_connection_snapshot_rejects_explicit_s3_session_token_without_key_pair():
    ray_cxx = _require_ray_cxx()
    source_conn = duckdb.connect()
    source_conn.execute("LOAD httpfs")
    source_conn.execute("SET s3_session_token='explicit-token'")

    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_conn.sql("SELECT 1"),
        "snapshot-partial-explicit-s3-token",
    )

    with pytest.raises(
        Exception,
        match="must set both s3_access_key_id and s3_secret_access_key",
    ):
        logical_plan.has_explicit_s3_credentials()


def test_connection_snapshot_recognizes_explicit_empty_s3_credentials():
    ray_cxx = _require_ray_cxx()
    source_conn = duckdb.connect()
    source_conn.execute("LOAD httpfs")
    source_conn.execute("SET s3_access_key_id=''")
    source_conn.execute("SET s3_secret_access_key=''")

    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_conn.sql("SELECT 1"),
        "snapshot-empty-explicit-s3",
    )

    assert logical_plan.has_explicit_s3_credentials() is True


def test_effective_session_config_overrides_stale_captured_environment(monkeypatch):
    ray_cxx = _require_ray_cxx()
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "stale-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "stale-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "stale-token")

    source_conn = duckdb.connect()
    source_conn.execute("LOAD httpfs")
    relation = source_conn.sql(
        "SELECT current_setting('s3_access_key_id') AS access_key, "
        "current_setting('s3_secret_access_key') AS secret_key, "
        "current_setting('s3_session_token') AS session_token"
    )
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        "snapshot-refreshed-s3-precedence",
    )
    effective_config = {
        "AWS_ACCESS_KEY_ID": "refreshed-key",
        "AWS_SECRET_ACCESS_KEY": "refreshed-secret",
        "AWS_SESSION_TOKEN": "refreshed-token",
    }

    planning_conn = duckdb.connect()
    physical_plan = logical_plan.to_physical_plan(
        planning_conn,
        effective_session_config=effective_config,
    )
    assert planning_conn.execute("SELECT current_setting('s3_access_key_id')").fetchone()[0] == "refreshed-key"

    restored_plan = pickle.loads(pickle.dumps(physical_plan))
    worker_cursor = duckdb.connect().cursor()
    result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        worker_cursor,
        restored_plan,
        effective_session_config=effective_config,
    )

    table = _table_from_native_result(result)
    assert table.column(0).to_pylist() == ["refreshed-key"]
    assert table.column(1).to_pylist() == ["refreshed-secret"]
    assert table.column(2).to_pylist() == ["refreshed-token"]
