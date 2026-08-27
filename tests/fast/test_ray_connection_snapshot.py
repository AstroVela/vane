# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import gc
import hashlib
import pickle
import threading

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


def _sql_string_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def test_distributed_plan_executes_storage_facing_file_scalars_after_transport(tmp_path):
    ray_cxx = _require_ray_cxx()
    payload = b"\x89PNG\r\n\x1a\n" + b"distributed-file-payload"
    payload_path = tmp_path / "distributed-file.bin"
    payload_path.write_bytes(payload)
    missing_path = tmp_path / "missing.bin"
    range_position = 8
    range_size = 12
    range_digest = hashlib.sha256(payload[range_position : range_position + range_size]).hexdigest()
    path_sql = _sql_string_literal(payload_path)
    missing_sql = _sql_string_literal(missing_path)

    connection = vane.connect()
    try:
        relation = connection.sql(
            f"""
            SELECT
                file_size(to_file({path_sql})) AS converted_size,
                file_size(try_to_file({path_sql})) AS optional_size,
                file_exists(file({path_sql}, NULL, NULL, NULL, NULL)) AS exists,
                file_stat(file({path_sql}, NULL, NULL, NULL, NULL)).object_size AS object_size,
                file_mime_type(file({path_sql}, NULL, NULL, NULL, NULL), 'content') AS mime_type,
                file_content_id(file_enrich(
                    file({path_sql}, NULL, {range_position}, {range_size}, NULL),
                    ['checksum']
                )) AS content_id,
                try_to_file({missing_sql}) IS NULL AS missing_is_null
            FROM range(2)
            """
        )
        logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            "storage-facing-file-scalars",
        )
        snapshot = logical_plan.__getstate__()[3]
        extension_names = {extension["name"] for extension in snapshot["extensions"]}
        assert {"file", "httpfs"}.issubset(extension_names)

        transported_logical_plan = pickle.loads(pickle.dumps(logical_plan))
    finally:
        connection.close()

    planning_connection = vane.connect()
    try:
        physical_plan = transported_logical_plan.to_physical_plan(planning_connection)
        transported_physical_plan = pickle.loads(pickle.dumps(physical_plan))
    finally:
        planning_connection.close()

    worker_connection = vane.connect()
    try:
        result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(
            worker_connection.cursor(),
            transported_physical_plan,
        )
        table = _table_from_native_result(result)
    finally:
        worker_connection.close()

    expected = (
        len(payload),
        len(payload),
        True,
        len(payload),
        "image/png",
        f"file-content-v1:checksum:sha256:{range_digest}",
        True,
    )
    assert [tuple(row.values()) for row in table.to_pylist()] == [expected, expected]


def test_distributed_plan_allows_io_free_file_scalars():
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    try:
        file_value = "file('file:///driver-only/value.txt', 'text/plain', 1, 2, 'sha256:abcd')"
        relation = connection.sql(
            f"""
            SELECT
                file_path({file_value}),
                file_mime_type({file_value}),
                guess_mime_type('GIF89a'::BLOB),
                file_same_location({file_value}, {file_value}),
                file_same_content({file_value}, {file_value}),
                file_locator_id({file_value}),
                file_content_id({file_value}),
                {file_value} = {file_value}
            FROM range(2)
            """
        )

        plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "io-free-file-scalars")

        assert plan is not None
    finally:
        connection.close()


@pytest.mark.parametrize("write_expression", ["check", "default"])
def test_distributed_write_snapshot_covers_write_owned_file_io_expressions(
    monkeypatch,
    tmp_path,
    write_expression,
):
    ray_cxx = _require_ray_cxx()
    payload_path = tmp_path / f"write-{write_expression}.bin"
    payload_path.write_bytes(b"write-owned-file-expression")
    database_path = tmp_path / f"write-{write_expression}.duckdb"
    path_sql = _sql_string_literal(payload_path)
    captured_plans = []

    class CapturingRunner:
        def run_write(self, relation):
            captured_plans.append(
                ray_cxx.PyLogicalPlan.from_duckdb_write_relation(
                    relation,
                    f"write-owned-file-{write_expression}",
                )
            )
            return {"ok": True}

    import vane.runners as runners_module

    monkeypatch.setattr(runners_module, "set_runner_ray", lambda *_args, **_kwargs: CapturingRunner())
    connection = vane.connect(str(database_path))
    try:
        if write_expression == "check":
            connection.execute("CREATE TABLE file_target (value FILE, CHECK (file_exists(value)))")
        else:
            connection.execute(f"CREATE TABLE file_target (value FILE DEFAULT try_to_file({path_sql}))")
            connection.execute(f"INSERT INTO file_target VALUES (file({path_sql}, NULL, NULL, NULL, NULL))")

        monkeypatch.setenv("VANE_RUNNER", "ray")
        if write_expression == "check":
            connection.sql(f"SELECT file({path_sql}, NULL, NULL, NULL, NULL) AS value").insert_into("file_target")
        else:
            connection.table("file_target").update({"value": vane.DefaultExpression()})

        assert len(captured_plans) == 1
        logical_plan = captured_plans[0]
        snapshot = logical_plan.__getstate__()[3]
        extension_names = {extension["name"] for extension in snapshot["extensions"]}
        assert {"file", "httpfs"}.issubset(extension_names)
        transported_logical_plan = pickle.loads(pickle.dumps(logical_plan))
    finally:
        connection.close()

    planning_connection = vane.connect()
    try:
        physical_plan = transported_logical_plan.to_physical_plan(planning_connection)
        assert physical_plan is not None
    finally:
        planning_connection.close()
    del physical_plan
    gc.collect()


def test_logical_plan_captures_connection_scoped_vane_session(monkeypatch):
    ray_cxx = _require_ray_cxx()

    monkeypatch.setenv("AWS_REVIEW_SESSION_SECRET_75", "session-a-secret")
    connection_a = vane.connect()
    cursor_a = connection_a.cursor()

    monkeypatch.delenv("AWS_REVIEW_SESSION_SECRET_75")
    connection_b = vane.connect()

    plan_a = ray_cxx.PyLogicalPlan.from_duckdb_relation(connection_a.sql("SELECT 1"), "session-a")
    cursor_plan_a = ray_cxx.PyLogicalPlan.from_duckdb_relation(cursor_a.sql("SELECT 1"), "session-a-cursor")
    plan_b = ray_cxx.PyLogicalPlan.from_duckdb_relation(connection_b.sql("SELECT 1"), "session-b")

    assert plan_a.session_id()
    assert plan_a.session_id() == cursor_plan_a.session_id()
    assert plan_a.session_id() != plan_b.session_id()
    assert plan_a.session_config()["AWS_REVIEW_SESSION_SECRET_75"] == "session-a-secret"
    assert "AWS_REVIEW_SESSION_SECRET_75" not in plan_b.session_config()

    restored_plan = pickle.loads(pickle.dumps(plan_a.to_physical_plan(vane.connect())))
    assert restored_plan.session_id() == plan_a.session_id()
    assert restored_plan.session_config() == plan_a.session_config()


def test_vllm_named_actor_pool_identity_includes_connection_session():
    from vane.ai.providers.vllm import _build_native_vllm_options_argument

    ray_cxx = _require_ray_cxx()
    connection_a = vane.connect()
    connection_b = vane.connect()
    query_id = "reused-query-id"
    options = _build_native_vllm_options_argument({"use_ray": True})

    def build_relation(connection):
        source = connection.sql("SELECT 'hello' AS prompt")
        generated = vane.FunctionExpression(
            "vllm",
            vane.ColumnExpression("prompt"),
            vane.ConstantExpression("test-model"),
            vane.ConstantExpression(options),
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
    from vane.datasource import DataSource, DataSourceTask, read_datasource

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
    connection = vane.connect()
    relation = read_datasource(SnapshotSource(), con=connection)
    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "datasource-session")

    assert plan.session_id()
    assert plan.session_config()["AWS_DATASOURCE_SESSION_SECRET_75"] == "datasource-secret"


def test_session_aws_settings_replay_only_on_the_target_connection_context(monkeypatch):
    ray_cxx = _require_ray_cxx()

    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://minio-a.internal:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "session-a-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "session-a-secret")
    connection_a = vane.connect()
    plan_a = ray_cxx.PyLogicalPlan.from_duckdb_relation(connection_a.sql("SELECT 1"), "session-a-settings")

    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://minio-b.internal:9443")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "session-b-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "session-b-secret")
    connection_b = vane.connect()
    plan_b = ray_cxx.PyLogicalPlan.from_duckdb_relation(connection_b.sql("SELECT 1"), "session-b-settings")

    for key in ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(key)
    target_root = vane.connect()
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
        "vane.runners.ray.runner.notify_connection_closed",
        closed_session_ids.append,
    )

    connection = vane.connect()
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
        "vane.runners.ray.runner.notify_connection_closed",
        closed_session_ids.append,
    )

    connection = vane.connect()
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
        "vane.runners.ray.runner.notify_connection_closed",
        _notify,
    )

    connection = vane.connect()
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
        "vane.runners.ray.runner.notify_connection_closed",
        closed_session_ids.append,
    )

    connection = vane.connect()
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
        vane._native,
        "get_or_infer_runner_type",
        lambda: configured_runner,
    )
    monkeypatch.setattr(
        "vane.runners.ray.runner.notify_connection_closed",
        closed_session_ids.append,
    )

    connection = vane.connect()
    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(connection.sql("SELECT 1"), "configured-runner-session")
    session_id = plan.session_id()

    connection.close()

    assert closed_session_ids == ([session_id] if expects_close else [])


def test_deserialized_plan_rejects_missing_session_config():
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(connection.sql("SELECT 1"), "missing-session-config")
    state = list(plan.__getstate__())
    state[3] = {"vane_session": {"id": plan.session_id()}}

    malformed_plan = ray_cxx.PyLogicalPlan.__new__(ray_cxx.PyLogicalPlan)
    with pytest.raises(Exception, match="Vane session is missing config"):
        malformed_plan.__setstate__(tuple(state))


def test_plan_pickles_reject_pre_session_state_shapes():
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(connection.sql("SELECT 1"), "strict-session-state")
    logical_state = logical_plan.__getstate__()
    malformed_logical = ray_cxx.PyLogicalPlan.__new__(ray_cxx.PyLogicalPlan)

    with pytest.raises(Exception, match="Invalid state for PyLogicalPlan"):
        malformed_logical.__setstate__(logical_state[:3])

    physical_plan = logical_plan.to_physical_plan(vane.connect())
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
    connection_a = vane.connect()
    plan_a = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        connection_a.sql("SELECT 1"),
        "plan-replay-source-a",
    ).to_physical_plan(vane.connect())

    monkeypatch.setenv("AWS_QUERY_REPLAY_SECRET", "session-b")
    connection_b = vane.connect()
    plan_b = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        connection_b.sql("SELECT 1"),
        "plan-replay-source-b",
    ).to_physical_plan(vane.connect())

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
    connection = vane.connect()
    source_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        connection.sql("SELECT 1"),
        "plan-replay-runtime-fields",
    ).to_physical_plan(vane.connect())
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


def test_transport_replays_attached_catalog_on_isolated_planning_connection(tmp_path):
    ray_cxx = _require_ray_cxx()
    attached_path = tmp_path / "attached-catalog.duckdb"

    setup_conn = vane.connect(str(attached_path))
    setup_conn.execute("CREATE TABLE items AS SELECT 42 AS value")
    setup_conn.close()

    source_conn = vane.connect()
    source_conn.execute(f"ATTACH '{attached_path.as_posix()}' AS attached_catalog")
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_conn.sql("SELECT * FROM attached_catalog.main.items"),
        "snapshot-attached-catalog",
    )
    snapshot = logical_plan.__getstate__()[3]
    assert len(snapshot["attached_databases"]) == 1
    assert "attached_catalog" in snapshot["attached_databases"][0]

    transported_plan = pickle.loads(pickle.dumps(logical_plan))
    source_conn.close()

    target_conn = vane.connect()
    physical_plan = transported_plan.to_physical_plan(target_conn)

    assert physical_plan.idx() == "snapshot-attached-catalog"
    assert "attached_databases" not in physical_plan.__getstate__()[6]
    assert target_conn.execute(
        "SELECT count(*) FROM duckdb_databases() WHERE database_name = 'attached_catalog'"
    ).fetchone() == (0,)


def test_connection_snapshot_does_not_transport_duckdb_secrets():
    ray_cxx = _require_ray_cxx()

    source_conn = vane.connect()
    source_conn.execute("LOAD httpfs")
    source_conn.execute(
        "CREATE SECRET source_only_secret (TYPE HTTP, BEARER_TOKEN 'source-only-token', SCOPE 'https://example.com')"
    )
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_conn.sql("SELECT 1 AS value"),
        "snapshot-with-source-secret",
    )

    snapshot = logical_plan.__getstate__()[3]
    assert "secrets" not in snapshot

    transported_plan = pickle.loads(pickle.dumps(logical_plan))
    source_conn.close()
    target_conn = vane.connect()
    transported_plan.to_physical_plan(target_conn)

    assert target_conn.execute(
        "SELECT count(*) FROM duckdb_secrets() WHERE name = 'source_only_secret'"
    ).fetchone() == (0,)


def test_worker_nondefault_snapshot_reuses_database(monkeypatch):
    from vane.runners.ray import worker as worker_module

    ray_cxx = _require_ray_cxx()
    query_id = "snapshot-nondefault-worker-database"
    source_connection = vane.connect(":memory:", config={"threads": "2"})
    source_connection.execute("SET threads=3")
    source_connection.execute("SET memory_limit='1GB'")
    source_connection.execute("SET local_exchange_buffer_bytes='64MB'")
    source_connection.execute("SET local_exchange_streaming=false")
    source_connection.execute("SET arrow_large_buffer_size=false")
    source_connection.execute("SET allow_persistent_secrets=false")
    source_memory_limit = source_connection.execute("SELECT current_setting('memory_limit')").fetchone()
    source_exchange_buffer = source_connection.execute(
        "SELECT current_setting('local_exchange_buffer_bytes')"
    ).fetchone()
    monkeypatch.setenv("VANE_DUCKDB_THREADS", "1")
    bootstrap_connection = vane.connect()
    actor_class = worker_module.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    actor._snapshot_connections = {}
    actor._snapshot_connections_lock = threading.Lock()
    actor._native_execution_condition = threading.Condition()
    actor._active_snapshot_execution_cursors = 0
    actor._closing_native_queries = set()
    actor._closing_native_tasks = set()
    actor._shutdown_started = False
    actor._duckdb_memory_bytes = 256 * 1024**2
    first_cursor = None
    second_cursor = None
    try:
        physical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
            source_connection.sql("SELECT 1 AS value"), query_id
        ).to_physical_plan(source_connection)
        worker_plan = pickle.loads(pickle.dumps(physical_plan))
        assert ray_cxx._register_query_python_replay_state(query_id, worker_plan) is True
        worker_snapshot = ray_cxx._lookup_query_connection_snapshot(query_id)
        worker_setting_names = {setting["name"].lower() for setting in worker_snapshot["settings"]}
        assert worker_setting_names.isdisjoint(
            {
                "max_memory",
                "memory_limit",
                "threads",
                "worker_threads",
                "local_exchange_streaming",
                "local_exchange_buffer_bytes",
                "arrow_large_buffer_size",
            }
        )

        database_identity = worker_module._query_worker_snapshot_database_identity(
            query_id,
            session_id="test-session",
            effective_s3_config={},
            use_session_credentials=True,
        )
        first_cursor = actor_class._get_snapshot_execution_cursor(
            actor,
            bootstrap_connection,
            query_id,
            database_identity=database_identity,
        )
        second_cursor = actor_class._get_snapshot_execution_cursor(
            actor,
            bootstrap_connection,
            query_id,
            database_identity=database_identity,
        )
        assert len(actor._snapshot_connections) == 1
        assert first_cursor.execute("SELECT current_setting('allow_persistent_secrets')").fetchone() == (False,)
        configured_memory_limit = first_cursor.execute("SELECT current_setting('memory_limit')").fetchone()
        configured_exchange_buffer = first_cursor.execute(
            "SELECT current_setting('local_exchange_buffer_bytes')"
        ).fetchone()
        assert configured_memory_limit != source_memory_limit
        assert configured_exchange_buffer != source_exchange_buffer
        assert first_cursor.execute("SELECT current_setting('threads')").fetchone() == (1,)
        assert first_cursor.execute("SELECT current_setting('local_exchange_streaming')").fetchone() == (True,)
        assert first_cursor.execute("SELECT current_setting('arrow_large_buffer_size')").fetchone() == (True,)

        result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(first_cursor, worker_plan)
        assert _table_from_native_result(result).column(0).to_pylist() == [1]
        assert first_cursor.execute("SELECT current_setting('memory_limit')").fetchone() == configured_memory_limit
        assert (
            first_cursor.execute("SELECT current_setting('local_exchange_buffer_bytes')").fetchone()
            == configured_exchange_buffer
        )
        assert first_cursor.execute("SELECT current_setting('threads')").fetchone() == (1,)
        assert first_cursor.execute("SELECT current_setting('local_exchange_streaming')").fetchone() == (True,)
        assert first_cursor.execute("SELECT current_setting('arrow_large_buffer_size')").fetchone() == (True,)

        first_cursor.execute("CREATE TABLE cached_snapshot_table AS SELECT 42 AS value")
        assert second_cursor.execute("SELECT value FROM cached_snapshot_table").fetchall() == [(42,)]
        with pytest.raises(Exception, match="cached_snapshot_table"):
            bootstrap_connection.execute("SELECT * FROM cached_snapshot_table")
    finally:
        if second_cursor is not None:
            actor_class._close_snapshot_execution_cursor(actor, second_cursor)
        if first_cursor is not None:
            actor_class._close_snapshot_execution_cursor(actor, first_cursor)
        for snapshot_connection in actor._snapshot_connections.values():
            snapshot_connection.close()
        ray_cxx._cleanup_query_python_replay_state(query_id)
        bootstrap_connection.close()
        source_connection.close()


def test_worker_file_snapshot_identities_use_isolated_read_only_instances(tmp_path, monkeypatch):
    from vane.runners.ray import worker as worker_module

    ray_cxx = _require_ray_cxx()
    database_path = tmp_path / "identity-source.duckdb"
    source_connection = vane.connect(str(database_path), config={"threads": "2"})
    source_connection.execute("CREATE TABLE snapshot_items AS SELECT 42 AS value")
    source_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_connection.sql("SELECT * FROM snapshot_items"),
        "snapshot-file-identity-source",
    ).to_physical_plan(source_connection)
    source_payload = pickle.dumps(source_plan)
    source_connection.close()
    del source_connection
    del source_plan
    gc.collect()

    first_plan = pickle.loads(source_payload)
    second_state = list(first_plan.__getstate__())
    second_state[3] = "snapshot-file-identity-second-resource"
    second_snapshot = dict(second_state[6])
    second_bootstrap = dict(second_snapshot["bootstrap"])
    second_bootstrap["config"] = {**second_bootstrap["config"], "threads": "3"}
    second_snapshot["bootstrap"] = second_bootstrap
    second_state[6] = second_snapshot
    second_plan = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)
    second_plan.__setstate__(tuple(second_state))

    actor_class = worker_module.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    actor._snapshot_connections = {}
    actor._snapshot_connections_lock = threading.Lock()
    actor._native_execution_condition = threading.Condition()
    actor._active_snapshot_execution_cursors = 0
    actor._shutdown_started = False
    actor._duckdb_memory_bytes = 256 * 1024**2
    monkeypatch.setenv("VANE_DUCKDB_THREADS", "1")
    bootstrap_connection = vane.connect()
    first_cursor = None
    second_cursor = None
    first_query_id = "snapshot-file-identity-first"
    second_query_id = "snapshot-file-identity-second"
    try:
        assert ray_cxx._register_query_python_replay_state(first_query_id, first_plan) is True
        assert ray_cxx._register_query_python_replay_state(second_query_id, second_plan) is True

        first_identity = worker_module._query_worker_snapshot_database_identity(
            first_query_id,
            session_id="test-session",
            effective_s3_config={},
            use_session_credentials=True,
        )
        second_identity = worker_module._query_worker_snapshot_database_identity(
            second_query_id,
            session_id="test-session",
            effective_s3_config={},
            use_session_credentials=True,
        )
        first_cursor = actor_class._get_snapshot_execution_cursor(
            actor,
            bootstrap_connection,
            first_query_id,
            database_identity=first_identity,
        )
        second_cursor = actor_class._get_snapshot_execution_cursor(
            actor,
            bootstrap_connection,
            second_query_id,
            database_identity=second_identity,
        )

        assert len(actor._snapshot_connections) == 2
        plan_runner = ray_cxx.DistributedPhysicalPlanRunner()
        first_result = plan_runner.execute_native(first_cursor, first_plan)
        second_result = plan_runner.execute_native(second_cursor, second_plan)
        assert _table_from_native_result(first_result).column(0).to_pylist() == [42]
        assert _table_from_native_result(second_result).column(0).to_pylist() == [42]
        assert first_cursor.execute("SELECT current_setting('access_mode'), current_setting('threads')").fetchone() == (
            "read_only",
            1,
        )
        assert second_cursor.execute(
            "SELECT current_setting('access_mode'), current_setting('threads')"
        ).fetchone() == (
            "read_only",
            1,
        )
        assert first_cursor.execute("SELECT * FROM snapshot_items").fetchall() == [(42,)]
        assert second_cursor.execute("SELECT * FROM snapshot_items").fetchall() == [(42,)]
    finally:
        if second_cursor is not None:
            actor_class._close_snapshot_execution_cursor(actor, second_cursor)
        if first_cursor is not None:
            actor_class._close_snapshot_execution_cursor(actor, first_cursor)
        for snapshot_connection in actor._snapshot_connections.values():
            snapshot_connection.close()
        ray_cxx._cleanup_query_python_replay_state(first_query_id)
        ray_cxx._cleanup_query_python_replay_state(second_query_id)
        bootstrap_connection.close()


def test_worker_file_snapshot_disables_persistent_secrets_before_first_use(tmp_path):
    ray_cxx = _require_ray_cxx()
    query_id = "snapshot-file-disable-persistent-secrets"
    database_path = tmp_path / "source.duckdb"
    source_connection = vane.connect(
        str(database_path),
        config={
            "allow_persistent_secrets": True,
            "secret_directory": str(tmp_path / "source-secrets"),
        },
    )
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_connection.sql("SELECT 1 AS value"),
        query_id,
    )
    physical_plan = logical_plan.to_physical_plan(source_connection)
    restored_plan = pickle.loads(pickle.dumps(physical_plan))
    del physical_plan
    del logical_plan
    source_connection.close()
    del source_connection
    gc.collect()

    bootstrap_connection = vane.connect()
    resolved_connection = None
    try:
        assert ray_cxx._register_query_python_replay_state(query_id, restored_plan) is True
        resolved_connection = ray_cxx._resolve_query_snapshot_connection(bootstrap_connection, query_id)

        assert resolved_connection.execute("SELECT current_setting('allow_persistent_secrets')").fetchone() == (False,)
        identity_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
            resolved_connection.sql("SELECT 1 AS value"),
            f"{query_id}-identity",
        )
        identity_config = identity_plan.__getstate__()[3]["bootstrap"]["config"]
        assert identity_config["allow_persistent_secrets"] is True
        assert identity_config["secret_directory"] == str(tmp_path / "source-secrets")
        with pytest.raises(Exception, match="Persistent secrets are disabled"):
            resolved_connection.execute(
                "CREATE PERSISTENT SECRET forbidden_worker_secret (TYPE HTTP, BEARER_TOKEN 'token')"
            )
    finally:
        ray_cxx._cleanup_query_python_replay_state(query_id)
        if resolved_connection is not None:
            resolved_connection.close()
        bootstrap_connection.close()


def test_pickled_physical_plan_replays_connection_snapshot_on_execute_native():
    ray_cxx = _require_ray_cxx()

    worker_cursor = vane.connect().cursor()
    worker_threads = worker_cursor.execute("SELECT current_setting('threads')").fetchone()[0]
    source_threads = 2 if worker_threads == 1 else 1
    source_conn = vane.connect()
    source_conn.execute(f"SET threads={source_threads}")
    source_conn.execute("SET TimeZone='UTC'")
    relation = source_conn.sql("SELECT * FROM (VALUES (1), (2), (3)) AS t(a)")

    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "snapshot-execute-native")
    physical_plan = plan.to_physical_plan(vane.connect())
    assert "threads" not in {setting["name"].lower() for setting in physical_plan.__getstate__()[6]["settings"]}
    restored_plan = pickle.loads(pickle.dumps(physical_plan))

    assert worker_cursor.execute("SELECT current_setting('TimeZone')").fetchone()[0] != "UTC"

    result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(worker_cursor, restored_plan)
    table = _table_from_native_result(result)

    assert table.num_rows == 3
    assert worker_cursor.execute("SELECT current_setting('threads')").fetchone()[0] == worker_threads
    assert worker_cursor.execute("SELECT current_setting('TimeZone')").fetchone()[0] == "UTC"


def test_snapshot_replay_error_does_not_echo_sensitive_setting_value():
    ray_cxx = _require_ray_cxx()

    source_connection = vane.connect()
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_connection.sql("SELECT 1"),
        "snapshot-redacted-query-error",
    )
    physical_plan = logical_plan.to_physical_plan(vane.connect())
    state = list(physical_plan.__getstate__())
    snapshot = dict(state[6])
    sensitive_value = "vane_snapshot_secret_must_not_reach_logs"
    snapshot["settings"] = [
        *snapshot["settings"],
        {"name": "threads", "value": sensitive_value, "input_type": "BIGINT"},
    ]
    state[6] = snapshot

    replay_plan = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)
    replay_plan.__setstate__(tuple(state))
    worker_connection = vane.connect()
    worker_cursor = worker_connection.cursor()
    try:
        with pytest.raises(Exception) as exc_info:
            ray_cxx.DistributedPhysicalPlanRunner().execute_native(worker_cursor, replay_plan)
        assert "Connection snapshot query failed (" in str(exc_info.value)
        assert sensitive_value not in str(exc_info.value)
    finally:
        worker_cursor.close()
        worker_connection.close()
        source_connection.close()


def test_connection_snapshot_captures_exact_extension_contract():
    ray_cxx = _require_ray_cxx()

    source_connection = vane.connect()
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_connection.sql("SELECT 1"),
        "snapshot-extension-contract",
    )
    snapshot = logical_plan.__getstate__()[3]

    assert snapshot["duckdb_source_id"]
    assert isinstance(snapshot["extensions"], list)
    assert all(set(extension) >= {"name", "version"} for extension in snapshot["extensions"])
    assert snapshot["distributed_extension_contracts"] == [
        "json{table_function:json_each(JSON)@1,"
        "table_function:json_each(JSON, VARCHAR)@1,"
        "table_function:json_each(VARCHAR)@1,"
        "table_function:json_each(VARCHAR, VARCHAR)@1,"
        "table_function:json_tree(JSON)@1,"
        "table_function:json_tree(JSON, VARCHAR)@1,"
        "table_function:json_tree(VARCHAR)@1,"
        "table_function:json_tree(VARCHAR, VARCHAR)@1}",
        "vane_core{table_function:datasource_scan(POINTER, POINTER, BLOB, BLOB[])@1,"
        "table_function:generate_series(BIGINT)@1,"
        "table_function:generate_series(BIGINT, BIGINT)@1,"
        "table_function:generate_series(BIGINT, BIGINT, BIGINT)@1,"
        "table_function:generate_series(TIMESTAMP, TIMESTAMP, INTERVAL)@1,"
        "table_function:range(BIGINT)@1,"
        "table_function:range(BIGINT, BIGINT)@1,"
        "table_function:range(BIGINT, BIGINT, BIGINT)@1,"
        "table_function:range(TIMESTAMP, TIMESTAMP, INTERVAL)@1,"
        "table_function:read_csv(VARCHAR)@1,"
        "table_function:read_csv(VARCHAR[])@1,"
        "table_function:read_csv_auto(VARCHAR)@1,"
        "table_function:read_csv_auto(VARCHAR[])@1,"
        "table_function:repeat(ANY, BIGINT)@1,"
        "table_function:repeat_row([ANY...])@1,"
        "table_function:unnest(ANY)@1}",
    ]


def test_snapshot_replay_rejects_different_duckdb_source_id():
    ray_cxx = _require_ray_cxx()

    source_connection = vane.connect()
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_connection.sql("SELECT 1"),
        "snapshot-source-id-mismatch",
    )
    physical_plan = logical_plan.to_physical_plan(vane.connect())
    state = list(physical_plan.__getstate__())
    snapshot = dict(state[6])
    snapshot["duckdb_source_id"] = "different-worker-build"
    state[6] = snapshot

    replay_plan = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)
    replay_plan.__setstate__(tuple(state))
    with pytest.raises(Exception, match="SourceID differs between coordinator and worker"):
        ray_cxx.DistributedPhysicalPlanRunner().execute_native(vane.connect().cursor(), replay_plan)


def test_snapshot_replay_rejects_different_static_extension_version():
    ray_cxx = _require_ray_cxx()

    source_connection = vane.connect()
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_connection.sql("SELECT 1"),
        "snapshot-extension-version-mismatch",
    )
    physical_plan = logical_plan.to_physical_plan(vane.connect())
    state = list(physical_plan.__getstate__())
    snapshot = dict(state[6])
    assert snapshot["extensions"]
    extensions = [dict(extension) for extension in snapshot["extensions"]]
    extensions[0]["version"] = "different-extension-version"
    snapshot["extensions"] = extensions
    state[6] = snapshot

    replay_plan = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)
    replay_plan.__setstate__(tuple(state))
    with pytest.raises(Exception, match="Static extension identities differ"):
        ray_cxx.DistributedPhysicalPlanRunner().execute_native(vane.connect().cursor(), replay_plan)


def test_snapshot_replay_rejects_different_distributed_extension_contract():
    ray_cxx = _require_ray_cxx()

    source_connection = vane.connect()
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_connection.sql("SELECT 1"),
        "snapshot-distributed-extension-mismatch",
    )
    physical_plan = logical_plan.to_physical_plan(vane.connect())
    state = list(physical_plan.__getstate__())
    snapshot = dict(state[6])
    snapshot["distributed_extension_contracts"] = [
        *snapshot["distributed_extension_contracts"],
        "missing_test_extension{table_function:scan(BIGINT)@1}",
    ]
    state[6] = snapshot

    replay_plan = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)
    replay_plan.__setstate__(tuple(state))
    with pytest.raises(Exception, match="contracts differ between coordinator and worker"):
        ray_cxx.DistributedPhysicalPlanRunner().execute_native(vane.connect().cursor(), replay_plan)


def test_logical_snapshot_validates_manifest_before_applying_effective_s3_config():
    ray_cxx = _require_ray_cxx()

    source_connection = vane.connect()
    source_connection.execute("LOAD httpfs")
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_connection.sql("SELECT 1"),
        "logical-snapshot-extension-before-s3",
    )
    state = list(logical_plan.__getstate__())
    snapshot = dict(state[3])
    assert any(extension["name"] == "httpfs" for extension in snapshot["extensions"])
    snapshot["distributed_extension_contracts"] = [
        *snapshot["distributed_extension_contracts"],
        "missing_test_extension{}",
    ]
    state[3] = snapshot

    replay_logical_plan = ray_cxx.PyLogicalPlan.__new__(ray_cxx.PyLogicalPlan)
    replay_logical_plan.__setstate__(tuple(state))
    planning_connection = vane.connect()
    with pytest.raises(Exception, match="contracts differ between coordinator and worker"):
        replay_logical_plan.to_physical_plan(
            planning_connection,
            effective_session_config={
                "AWS_ACCESS_KEY_ID": "must-not-be-applied",
                "AWS_SECRET_ACCESS_KEY": "must-not-be-applied",
            },
        )

    assert planning_connection.execute(
        "SELECT name, value FROM duckdb_settings() "
        "WHERE name IN ('s3_access_key_id', 's3_secret_access_key') ORDER BY name"
    ).fetchall() == [
        ("s3_access_key_id", None),
        ("s3_secret_access_key", None),
    ]


@pytest.mark.parametrize(
    ("field_name", "error_match"),
    [
        ("duckdb_source_id", "missing duckdb_source_id"),
        ("extensions", "extensions must be a list"),
        ("distributed_extension_contracts", "distributed_extension_contracts must be a list"),
    ],
)
def test_snapshot_replay_rejects_missing_extension_contract_field(field_name: str, error_match: str):
    ray_cxx = _require_ray_cxx()

    source_connection = vane.connect()
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_connection.sql("SELECT 1"),
        f"snapshot-missing-extension-contract-{field_name}",
    )
    physical_plan = logical_plan.to_physical_plan(vane.connect())
    state = list(physical_plan.__getstate__())
    snapshot = dict(state[6])
    del snapshot[field_name]
    state[6] = snapshot

    replay_plan = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)
    replay_plan.__setstate__(tuple(state))
    with pytest.raises(Exception, match=error_match):
        ray_cxx.DistributedPhysicalPlanRunner().execute_native(vane.connect().cursor(), replay_plan)


def test_snapshot_replay_rejects_non_string_distributed_extension_contract():
    ray_cxx = _require_ray_cxx()

    source_connection = vane.connect()
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_connection.sql("SELECT 1"),
        "snapshot-non-string-distributed-contract",
    )
    physical_plan = logical_plan.to_physical_plan(vane.connect())
    state = list(physical_plan.__getstate__())
    snapshot = dict(state[6])
    snapshot["distributed_extension_contracts"] = [True]
    state[6] = snapshot

    replay_plan = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)
    replay_plan.__setstate__(tuple(state))
    with pytest.raises(Exception, match="distributed extension contract must be a string"):
        ray_cxx.DistributedPhysicalPlanRunner().execute_native(vane.connect().cursor(), replay_plan)


def test_snapshot_replay_rejects_legacy_extension_name_list():
    ray_cxx = _require_ray_cxx()

    source_connection = vane.connect()
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_connection.sql("SELECT 1"),
        "snapshot-legacy-extension-list",
    )
    physical_plan = logical_plan.to_physical_plan(vane.connect())
    state = list(physical_plan.__getstate__())
    snapshot = dict(state[6])
    snapshot["extensions"] = ["core_functions"]
    state[6] = snapshot

    replay_plan = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)
    replay_plan.__setstate__(tuple(state))
    with pytest.raises(Exception, match="extension entry must be a dict"):
        ray_cxx.DistributedPhysicalPlanRunner().execute_native(vane.connect().cursor(), replay_plan)


def test_snapshot_replay_rejects_non_static_extensions_without_installing(tmp_path):
    ray_cxx = _require_ray_cxx()

    source_conn = vane.connect()
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_conn.sql("SELECT 1"),
        "snapshot-missing-extension",
    )
    physical_plan = logical_plan.to_physical_plan(vane.connect())
    state = list(physical_plan.__getstate__())
    snapshot = dict(state[6])
    snapshot["extensions"] = [{"name": "sqlite_scanner", "version": ""}]
    state[6] = snapshot

    replay_plan = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)
    replay_plan.__setstate__(tuple(state))

    extension_directory = tmp_path / "extensions"
    worker_connection = vane.connect(
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

    source_conn = vane.connect()
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_conn.sql("SELECT 1"),
        "snapshot-bootstrap-extension-security",
    )
    physical_plan = logical_plan.to_physical_plan(vane.connect())
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
    snapshot["extensions"] = [{"name": "sqlite_scanner", "version": ""}]
    state[6] = snapshot

    replay_plan = ray_cxx.DistributedPhysicalPlan.__new__(ray_cxx.DistributedPhysicalPlan)
    replay_plan.__setstate__(tuple(state))

    with pytest.raises(Exception) as exc_info:
        ray_cxx.DistributedPhysicalPlanRunner().execute_native(
            vane.connect().cursor(),
            replay_plan,
        )

    assert not extension_directory.exists()
    message = str(exc_info.value)
    assert "sqlite_all_varchar" in message
    assert "sqlite_scanner" in message
    assert "not statically linked" in message


def test_snapshot_bootstrap_applies_static_extension_settings_after_connect(tmp_path):
    ray_cxx = _require_ray_cxx()

    source_conn = vane.connect()
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
    physical_plan = replay_logical_plan.to_physical_plan(vane.connect())
    restored_plan = pickle.loads(pickle.dumps(physical_plan))
    result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        vane.connect().cursor(),
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

    source_connection = vane.connect()
    for name in snapshot_setting_names:
        source_connection.execute(f"SET {name} = true")
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_connection.sql("SELECT 1"),
        "snapshot-extension-security-settings",
    )
    physical_plan = logical_plan.to_physical_plan(vane.connect())
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

    worker_connection = vane.connect(config={name: "true" for name in setting_names})
    assert dict(worker_connection.execute(settings_query).fetchall()) == {name: "true" for name in setting_names}

    ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        worker_connection.cursor(),
        replay_plan,
    )

    assert dict(worker_connection.execute(settings_query).fetchall()) == {name: "false" for name in setting_names}


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
    source_conn = vane.connect(
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

    physical_plan = logical_plan.to_physical_plan(vane.connect())

    assert physical_plan.idx() == "file-security-setting"
    restored_plan = pickle.loads(pickle.dumps(physical_plan))
    assert dict(source_conn.execute(settings_query).fetchall()) == source_settings
    source_conn.close()
    del logical_plan
    del physical_plan
    gc.collect()

    worker_conn = vane.connect(config={name: "true" for name in setting_names})
    assert dict(worker_conn.execute(settings_query).fetchall()) == {name: "true" for name in setting_names}

    restored_result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        worker_conn.cursor(),
        restored_plan,
    )

    assert _table_from_native_result(restored_result).column(0).to_pylist() == [1]


def test_file_database_table_scan_reopens_bootstrap_on_worker(tmp_path):
    ray_cxx = _require_ray_cxx()
    database_path = str(tmp_path / "table-scan-bootstrap.duckdb")
    source_conn = vane.connect(database_path)
    source_conn.execute("CREATE TABLE numbers AS SELECT i AS value FROM range(10) data(i)")

    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_conn.sql("SELECT sum(value) AS total FROM numbers"),
        "snapshot-file-table-scan",
    )
    physical_plan = logical_plan.to_physical_plan(vane.connect())
    restored_plan = pickle.loads(pickle.dumps(physical_plan))

    result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        vane.connect().cursor(),
        restored_plan,
    )

    assert _table_from_native_result(result).column(0).to_pylist() == [45]


def test_effective_session_config_reaches_nondefault_bootstrap_connection(tmp_path):
    ray_cxx = _require_ray_cxx()
    database_path = str(tmp_path / "session-bootstrap.duckdb")
    source_conn = vane.connect(database_path)
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
        vane.connect(),
        effective_session_config=effective_config,
    )
    restored_plan = pickle.loads(pickle.dumps(physical_plan))
    result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        vane.connect().cursor(),
        restored_plan,
        effective_session_config=effective_config,
    )

    table = _table_from_native_result(result)
    assert [table.column(index).to_pylist() for index in range(3)] == [
        ["resolved-profile-key"],
        ["resolved-profile-secret"],
        ["resolved-profile-token"],
    ]


def test_effective_session_config_does_not_load_undeclared_httpfs():
    ray_cxx = _require_ray_cxx()
    source_conn = vane.connect()
    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_conn.sql("SELECT 1 AS value"),
        "snapshot-session-config-without-httpfs",
    )
    state = list(logical_plan.__getstate__())
    snapshot = dict(state[3])
    snapshot["extensions"] = [extension for extension in snapshot["extensions"] if extension["name"] != "httpfs"]
    state[3] = snapshot
    replay_logical_plan = ray_cxx.PyLogicalPlan.__new__(ray_cxx.PyLogicalPlan)
    replay_logical_plan.__setstate__(tuple(state))

    planning_conn = vane.connect()
    with pytest.raises(Exception, match="Static extension identities differ"):
        replay_logical_plan.to_physical_plan(
            planning_conn,
            effective_session_config={
                "AWS_ACCESS_KEY_ID": "unused-key",
                "AWS_SECRET_ACCESS_KEY": "unused-secret",
                "AWS_REGION": "us-east-2",
            },
        )

    assert planning_conn.execute(
        "SELECT name, value FROM duckdb_settings() "
        "WHERE name IN ('s3_access_key_id', 's3_secret_access_key') ORDER BY name"
    ).fetchall() == [
        ("s3_access_key_id", None),
        ("s3_secret_access_key", None),
    ]


def test_file_database_snapshot_baseline_ignores_database_target_settings(tmp_path):
    ray_cxx = _require_ray_cxx()
    database_path = str(tmp_path / "read-only-bootstrap.duckdb")
    vane.connect(database_path).close()
    source_conn = vane.connect(database_path, config={"access_mode": "READ_ONLY"})

    logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        source_conn.sql("SELECT 1 AS value"),
        "snapshot-file-target-settings",
    )
    snapshot_setting_names = {setting["name"].lower() for setting in logical_plan.__getstate__()[3]["settings"]}

    assert snapshot_setting_names.isdisjoint({"access_mode", "temp_directory"})
    physical_plan = logical_plan.to_physical_plan(vane.connect())
    result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        vane.connect().cursor(),
        physical_plan,
    )
    assert _table_from_native_result(result).column(0).to_pylist() == [1]


def test_explicit_connection_s3_settings_override_effective_session_config():
    ray_cxx = _require_ray_cxx()
    source_conn = vane.connect()
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

    planning_conn = vane.connect()
    physical_plan = logical_plan.to_physical_plan(
        planning_conn,
        effective_session_config=effective_config,
    )
    assert physical_plan.has_explicit_s3_credentials() is True
    assert planning_conn.execute("SELECT current_setting('s3_access_key_id')").fetchone()[0] == "explicit-key"

    restored_plan = pickle.loads(pickle.dumps(physical_plan))
    direct_result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        planning_conn,
        physical_plan,
        effective_session_config=effective_config,
    )
    direct_table = _table_from_native_result(direct_result)
    assert direct_table.column(0).to_pylist() == ["explicit-key"]
    assert direct_table.column(1).to_pylist() == ["explicit-secret"]
    assert direct_table.column(2).to_pylist() == [""]
    assert planning_conn.execute("SELECT current_setting('s3_access_key_id')").fetchone()[0] == "explicit-key"
    assert planning_conn.execute("SELECT current_setting('s3_session_token')").fetchone()[0] == ""

    worker_cursor = vane.connect().cursor()
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
    source_conn = vane.connect()
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
    source_conn = vane.connect()
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
    source_conn = vane.connect()
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
    source_conn = vane.connect()
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

    source_conn = vane.connect()
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

    planning_conn = vane.connect()
    physical_plan = logical_plan.to_physical_plan(
        planning_conn,
        effective_session_config=effective_config,
    )
    assert planning_conn.execute("SELECT current_setting('s3_access_key_id')").fetchone()[0] == "refreshed-key"

    restored_plan = pickle.loads(pickle.dumps(physical_plan))
    worker_cursor = vane.connect().cursor()
    result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        worker_cursor,
        restored_plan,
        effective_session_config=effective_config,
    )

    table = _table_from_native_result(result)
    assert table.column(0).to_pylist() == ["refreshed-key"]
    assert table.column(1).to_pylist() == ["refreshed-secret"]
    assert table.column(2).to_pylist() == ["refreshed-token"]
