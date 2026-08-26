# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import pytest

import vane
from vane.runners.ray import worker as worker_module


def _worker_actor():
    actor_class = worker_module.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    actor._native_execution_condition = threading.Condition()
    actor._active_snapshot_execution_cursors = 0
    actor._active_snapshot_cursors = set()
    actor._closing_native_queries = set()
    actor._closing_native_tasks = set()
    actor._shutdown_started = False
    actor._snapshot_connections = {}
    actor._snapshot_connections_lock = threading.Lock()
    actor._snapshot_connection_active_cursors = {}
    actor._snapshot_cursor_database_identities = {}
    actor._retired_snapshot_database_identities = set()
    actor._retired_snapshot_session_ids = set()
    actor._configure_snapshot_conn = lambda _connection: None
    return actor_class, actor


def _snapshot_database_identity(
    snapshot,
    *,
    effective_s3_config=None,
    use_session_credentials=True,
    session_id="test-session",
):
    return worker_module._worker_snapshot_database_identity(
        snapshot,
        session_id=session_id,
        effective_s3_config=effective_s3_config or {},
        use_session_credentials=use_session_credentials,
    )


def _dynamic_descriptor(*, name="root", sha256="1" * 64, dependencies=None):
    connection = vane.connect()
    try:
        platform = connection.execute("SELECT platform FROM pragma_platform()").fetchone()[0]
    finally:
        connection.close()
    return {
        "format_version": 1,
        "name": name,
        "extension_version": "test-version",
        "abi_type": "CPP",
        "duckdb_source_id": vane.__git_revision__,
        "vane_version": vane.__version__,
        "platform": platform,
        "sha256": sha256,
        "trust_identity": "local-tests",
        "dependencies": list(dependencies or []),
    }


def test_worker_shared_database_disables_persistent_secrets_at_connect(monkeypatch):
    actor_class, actor = _worker_actor()
    actor._shared_conn = None
    actor._shared_conn_lock = threading.Lock()
    actor._duckdb_memory_bytes = 4096
    configured_connections = []
    connect_calls = []
    connection = object()
    actor._configure_conn = configured_connections.append

    def connect(*args, **kwargs):
        connect_calls.append((args, kwargs))
        return connection

    monkeypatch.setattr(vane, "connect", connect)

    assert actor_class._get_shared_conn(actor) is connection
    assert connect_calls == [((), {"config": {"allow_persistent_secrets": False}})]
    assert configured_connections == [connection]


def test_worker_snapshot_execution_cursor_caches_nondefault_database(monkeypatch):
    actor_class, actor = _worker_actor()
    database_identity = worker_module.WorkerSnapshotDatabaseIdentity(
        "/tmp/vane-worker-snapshot.duckdb",
        False,
        (("threads", "2"),),
        (),
        "test-source-id",
        (),
        (),
        (),
        "",
        "",
        False,
    )
    cursors = []
    prepare_calls = []
    lifecycle = []

    class Cursor:
        def close(self):
            return None

    class ResolvedConnection:
        def cursor(self):
            lifecycle.append("cursor")
            cursor = Cursor()
            cursors.append(cursor)
            return cursor

    resolved_connection = ResolvedConnection()
    configured_connections = []

    def configure(connection):
        configured_connections.append(connection)
        lifecycle.append("configure")

    actor._configure_snapshot_conn = configure
    monkeypatch.setattr(
        worker_module,
        "require_ray_cxx_attr",
        lambda name, *, hint: (
            lambda query_id: (
                prepare_calls.append(query_id),
                lifecycle.append("prepare"),
                resolved_connection,
            )[2]
        ),
    )

    actor_class._prepare_snapshot_database(
        actor,
        "query-a",
        database_identity=database_identity,
    )
    first = actor_class._get_snapshot_execution_cursor(
        actor,
        "query-a",
        database_identity=database_identity,
    )
    actor_class._prepare_snapshot_database(
        actor,
        "query-b",
        database_identity=database_identity,
    )
    second = actor_class._get_snapshot_execution_cursor(
        actor,
        "query-b",
        database_identity=database_identity,
    )

    assert first is cursors[0]
    assert second is cursors[1]
    assert prepare_calls == ["query-a"]
    assert configured_connections == [resolved_connection]
    assert lifecycle == ["prepare", "configure", "cursor", "cursor"]
    assert actor._snapshot_connections == {database_identity: resolved_connection}
    assert actor._active_snapshot_cursors == {first, second}
    actor_class._close_snapshot_execution_cursor(actor, second)
    actor_class._close_snapshot_execution_cursor(actor, first)
    assert actor._active_snapshot_execution_cursors == 0
    assert actor._active_snapshot_cursors == set()


@pytest.mark.parametrize(
    ("old_credential_source", "new_credential_source"),
    (
        ("session", "session"),
        ("explicit", "explicit"),
        ("session", "explicit"),
    ),
    ids=("session-to-session", "explicit-to-explicit", "session-to-explicit"),
)
def test_worker_snapshot_execution_cursor_retires_rotated_s3_database(
    monkeypatch,
    old_credential_source,
    new_credential_source,
):
    actor_class, actor = _worker_actor()
    base_snapshot = {
        "duckdb_source_id": "test-source-id",
        "extensions": [{"name": "httpfs", "version": "test-version"}],
        "dynamic_extensions": [],
        "distributed_extension_contracts": [],
        "settings": [],
    }

    def identity(credential_source, access_key, secret_key):
        snapshot = base_snapshot
        effective_s3_config = {
            "AWS_ACCESS_KEY_ID": access_key,
            "AWS_SECRET_ACCESS_KEY": secret_key,
        }
        use_session_credentials = credential_source == "session"
        if not use_session_credentials:
            snapshot = {
                **base_snapshot,
                "settings": [
                    {"name": "s3_access_key_id", "value": access_key, "input_type": "VARCHAR"},
                    {"name": "s3_secret_access_key", "value": secret_key, "input_type": "VARCHAR"},
                    {"name": "s3_session_token", "value": "", "input_type": "VARCHAR"},
                ],
            }
            effective_s3_config = {}
        return _snapshot_database_identity(
            snapshot,
            session_id="session-a",
            effective_s3_config=effective_s3_config,
            use_session_credentials=use_session_credentials,
        )

    old_identity = identity(old_credential_source, "old-key", "old-secret")
    new_identity = identity(new_credential_source, "new-key", "new-secret")
    assert old_identity != new_identity
    assert old_identity.replaces_s3_identity(new_identity) is True
    assert new_identity.replaces_s3_identity(old_identity) is True
    assert "old-key" not in repr(old_identity)
    assert "old-secret" not in repr(old_identity)
    assert "new-key" not in repr(new_identity)
    assert "new-secret" not in repr(new_identity)
    connections = []

    class Cursor:
        def close(self):
            return None

    class ResolvedConnection:
        def __init__(self):
            self.closed = False

        def cursor(self):
            return Cursor()

        def close(self):
            self.closed = True

    def resolve(_query_id):
        connection = ResolvedConnection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(worker_module, "require_ray_cxx_attr", lambda name, *, hint: resolve)

    actor_class._prepare_snapshot_database(
        actor,
        "old-query",
        database_identity=old_identity,
    )
    old_cursor = actor_class._get_snapshot_execution_cursor(
        actor,
        "old-query",
        database_identity=old_identity,
    )
    actor_class._prepare_snapshot_database(
        actor,
        "new-query",
        database_identity=new_identity,
    )
    new_cursor = actor_class._get_snapshot_execution_cursor(
        actor,
        "new-query",
        database_identity=new_identity,
    )

    assert len(actor._snapshot_connections) == 2
    assert connections[0].closed is False
    actor_class._close_snapshot_execution_cursor(actor, old_cursor)
    assert connections[0].closed is True
    assert actor._snapshot_connections == {new_identity: connections[1]}

    actor_class._close_snapshot_execution_cursor(actor, new_cursor)
    actor_class._retire_snapshot_databases_for_session(actor, "session-a")
    assert connections[1].closed is True
    assert actor._snapshot_connections == {}

    actor_class._prepare_snapshot_database(
        actor,
        "late-cleanup",
        database_identity=new_identity,
    )
    late_cleanup_cursor = actor_class._get_snapshot_execution_cursor(
        actor,
        "late-cleanup",
        database_identity=new_identity,
    )
    assert actor._snapshot_connections == {new_identity: connections[2]}
    actor_class._close_snapshot_execution_cursor(actor, late_cleanup_cursor)
    assert connections[2].closed is True
    assert actor._snapshot_connections == {}


def test_worker_snapshot_configuration_failure_closes_uncached_database(monkeypatch):
    actor_class, actor = _worker_actor()
    database_identity = worker_module.WorkerSnapshotDatabaseIdentity(
        ":memory:",
        False,
        (),
        (),
        "test-source-id",
        (),
        (),
        (),
        "",
        "",
        False,
    )

    class ResolvedConnection:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

        def cursor(self):
            raise AssertionError("cursor must not be created after configuration failure")

    resolved_connection = ResolvedConnection()

    def fail_configuration(_connection):
        raise RuntimeError("snapshot configuration failed")

    actor._configure_snapshot_conn = fail_configuration
    monkeypatch.setattr(
        worker_module,
        "require_ray_cxx_attr",
        lambda name, *, hint: lambda _query_id: resolved_connection,
    )

    with pytest.raises(RuntimeError, match="snapshot configuration failed"):
        actor_class._prepare_snapshot_database(
            actor,
            "query-a",
            database_identity=database_identity,
        )

    assert resolved_connection.closed is True
    assert actor._snapshot_connections == {}
    assert actor._active_snapshot_execution_cursors == 0


def test_worker_query_snapshot_preparation_retires_database_when_session_closes_during_publish(monkeypatch):
    actor_class, actor = _worker_actor()
    actor._session_connections_lock = threading.RLock()
    actor._closed_session_ids = worker_module.BoundedReplayMap(capacity=16)
    actor._session_s3_configs = {"session-a": {}}
    session_connection = object()
    actor._session_connections = {"session-a": ({}, session_connection)}
    operation_lock = threading.Lock()
    actor._get_session_operation_lock = lambda _session_id: operation_lock
    actor._get_session_conn = lambda _session_id, _config, *, use_session_credentials: session_connection
    actor._refresh_session_s3_config_locked = lambda _session_id, _config, _connection, *, use_session_credentials: {}
    database_identity = worker_module.WorkerSnapshotDatabaseIdentity(
        ":memory:",
        False,
        (),
        (),
        "test-source-id",
        (("httpfs", ""),),
        (),
        (),
        "session-a",
        "test-s3-identity",
        True,
    )
    closed = []

    class _SnapshotConnection:
        def close(self):
            closed.append("snapshot")

    snapshot_connection = _SnapshotConnection()

    def _prepare(_query_id, *, database_identity):
        actor._snapshot_connections[database_identity] = snapshot_connection
        actor._closed_session_ids["session-a"] = True

    actor._prepare_snapshot_database = _prepare
    monkeypatch.setattr(
        worker_module,
        "_query_worker_snapshot_database_identity",
        lambda _query_id, **_kwargs: database_identity,
    )

    class _Plan:
        @staticmethod
        def session_id():
            return "session-a"

        @staticmethod
        def session_config():
            return {}

        @staticmethod
        def has_explicit_s3_credentials():
            return False

        @staticmethod
        def resource_query_id():
            return "resource-query"

    with pytest.raises(RuntimeError, match="session closed during snapshot preparation"):
        actor_class._prepare_query_snapshot_database(actor, _Plan())

    assert closed == ["snapshot"]
    assert actor._snapshot_connections == {}


def test_worker_snapshot_execution_cursor_isolates_exact_extension_identities(monkeypatch):
    actor_class, actor = _worker_actor()
    base_snapshot = {
        "duckdb_source_id": "test-source-id",
        "extensions": [],
        "dynamic_extensions": [],
        "distributed_extension_contracts": [],
        "settings": [],
    }
    httpfs_snapshot = {
        **base_snapshot,
        "extensions": [{"name": "httpfs", "version": "test-version"}],
    }
    identities = {
        "plain-a": _snapshot_database_identity(base_snapshot),
        "plain-b": _snapshot_database_identity(base_snapshot),
        "httpfs": _snapshot_database_identity(httpfs_snapshot),
    }
    assert identities["plain-a"] == identities["plain-b"]
    assert identities["plain-a"] != identities["httpfs"]

    created_connections = []

    class Cursor:
        def __init__(self, connection):
            self.connection = connection

        def close(self):
            return None

    class ResolvedConnection:
        def __init__(self, query_id):
            self.query_id = query_id

        def cursor(self):
            return Cursor(self)

    def resolve(query_id):
        connection = ResolvedConnection(query_id)
        created_connections.append(connection)
        return connection

    monkeypatch.setattr(
        worker_module,
        "require_ray_cxx_attr",
        lambda name, *, hint: resolve,
    )

    actor_class._prepare_snapshot_database(
        actor,
        "plain-a",
        database_identity=identities["plain-a"],
    )
    plain_a = actor_class._get_snapshot_execution_cursor(
        actor,
        "plain-a",
        database_identity=identities["plain-a"],
    )
    actor_class._prepare_snapshot_database(
        actor,
        "plain-b",
        database_identity=identities["plain-b"],
    )
    plain_b = actor_class._get_snapshot_execution_cursor(
        actor,
        "plain-b",
        database_identity=identities["plain-b"],
    )
    actor_class._prepare_snapshot_database(
        actor,
        "httpfs",
        database_identity=identities["httpfs"],
    )
    httpfs = actor_class._get_snapshot_execution_cursor(
        actor,
        "httpfs",
        database_identity=identities["httpfs"],
    )

    assert plain_a.connection is plain_b.connection
    assert httpfs.connection is not plain_a.connection
    assert len(created_connections) == 2
    assert len(actor._snapshot_connections) == 2

    actor_class._close_snapshot_execution_cursor(actor, httpfs)
    actor_class._close_snapshot_execution_cursor(actor, plain_b)
    actor_class._close_snapshot_execution_cursor(actor, plain_a)
    assert actor._active_snapshot_execution_cursors == 0


def test_worker_snapshot_database_identity_includes_exact_dynamic_manifest():
    base_snapshot = {
        "duckdb_source_id": "test-source-id",
        "extensions": [],
        "dynamic_extensions": [_dynamic_descriptor()],
        "distributed_extension_contracts": [],
        "settings": [],
    }
    first = _snapshot_database_identity(base_snapshot)
    second = _snapshot_database_identity(
        {
            **base_snapshot,
            "dynamic_extensions": [_dynamic_descriptor(sha256="2" * 64)],
        }
    )

    assert first != second
    assert first.dynamic_extensions[0][0] == "root"
    assert first.has_extension("root") is True
    assert first.has_extension("httpfs") is False


def test_worker_snapshot_execution_cursor_isolates_replayed_settings(monkeypatch):
    actor_class, actor = _worker_actor()
    base_snapshot = {
        "duckdb_source_id": "test-source-id",
        "extensions": [],
        "dynamic_extensions": [],
        "distributed_extension_contracts": [],
        "settings": [],
    }
    proxy_snapshot = {
        **base_snapshot,
        "settings": [
            {
                "name": "http_proxy_username",
                "value": "query-a",
                "input_type": "VARCHAR",
            }
        ],
    }
    identities = {
        "default": _snapshot_database_identity(base_snapshot),
        "proxy": _snapshot_database_identity(proxy_snapshot),
    }
    assert identities["default"] != identities["proxy"]
    assert identities["proxy"] == _snapshot_database_identity(
        {
            **base_snapshot,
            "settings": [
                {
                    "name": "HTTP_PROXY_USERNAME",
                    "value": "query-a",
                    "input_type": "varchar",
                }
            ],
        }
    )

    created_connections = []

    class Cursor:
        def __init__(self, connection):
            self.connection = connection

        def close(self):
            return None

    class ResolvedConnection:
        def __init__(self, query_id):
            self.query_id = query_id

        def cursor(self):
            return Cursor(self)

    def resolve(query_id):
        connection = ResolvedConnection(query_id)
        created_connections.append(connection)
        return connection

    monkeypatch.setattr(
        worker_module,
        "require_ray_cxx_attr",
        lambda name, *, hint: resolve,
    )

    actor_class._prepare_snapshot_database(
        actor,
        "default",
        database_identity=identities["default"],
    )
    default_cursor = actor_class._get_snapshot_execution_cursor(
        actor,
        "default",
        database_identity=identities["default"],
    )
    actor_class._prepare_snapshot_database(
        actor,
        "proxy",
        database_identity=identities["proxy"],
    )
    proxy_cursor = actor_class._get_snapshot_execution_cursor(
        actor,
        "proxy",
        database_identity=identities["proxy"],
    )

    assert default_cursor.connection is not proxy_cursor.connection
    assert len(created_connections) == 2
    assert len(actor._snapshot_connections) == 2

    actor_class._close_snapshot_execution_cursor(actor, proxy_cursor)
    actor_class._close_snapshot_execution_cursor(actor, default_cursor)
    assert actor._active_snapshot_execution_cursors == 0


def test_worker_snapshot_database_identity_normalizes_bootstrap_config_values():
    snapshot = {
        "bootstrap": {
            "database": ":memory:",
            "read_only": False,
            "config": {"threads": 2},
        },
        "duckdb_source_id": "test-source-id",
        "extensions": [],
        "dynamic_extensions": [],
        "distributed_extension_contracts": [],
        "settings": [],
    }
    string_config_snapshot = {
        **snapshot,
        "bootstrap": {**snapshot["bootstrap"], "config": {"threads": "2"}},
    }

    assert _snapshot_database_identity(snapshot) == _snapshot_database_identity(string_config_snapshot)


def test_worker_snapshot_database_identity_isolates_effective_s3_configuration():
    snapshot = {
        "duckdb_source_id": "test-source-id",
        "extensions": [{"name": "httpfs", "version": "test-version"}],
        "dynamic_extensions": [],
        "distributed_extension_contracts": [],
        "settings": [],
    }
    first = _snapshot_database_identity(
        snapshot,
        effective_s3_config={
            "AWS_ACCESS_KEY_ID": "key-a",
            "AWS_SECRET_ACCESS_KEY": "secret-a",
            "AWS_REGION": "region-a",
        },
    )
    second = _snapshot_database_identity(
        snapshot,
        effective_s3_config={
            "AWS_ACCESS_KEY_ID": "key-b",
            "AWS_SECRET_ACCESS_KEY": "secret-b",
            "AWS_REGION": "region-a",
        },
    )
    explicit_snapshot_credentials = _snapshot_database_identity(
        snapshot,
        effective_s3_config={"AWS_REGION": "region-a"},
        use_session_credentials=False,
    )
    first_with_different_refresh_deadline = _snapshot_database_identity(
        snapshot,
        effective_s3_config={
            "AWS_ACCESS_KEY_ID": "key-a",
            "AWS_SECRET_ACCESS_KEY": "secret-a",
            "AWS_REGION": "region-a",
            worker_module._AWS_CREDENTIAL_REFRESH_AT_KEY: "12345",
        },
    )

    assert first != second
    assert first != explicit_snapshot_credentials
    assert first == first_with_different_refresh_deadline
    assert "key-a" not in repr(first)
    assert "secret-a" not in repr(first)


def test_worker_snapshot_database_identity_hashes_explicit_s3_credentials():
    def identity(access_key, secret_key):
        return _snapshot_database_identity(
            {
                "duckdb_source_id": "test-source-id",
                "extensions": [{"name": "httpfs", "version": "test-version"}],
                "dynamic_extensions": [],
                "distributed_extension_contracts": [],
                "settings": [
                    {"name": "s3_access_key_id", "value": access_key, "input_type": "VARCHAR"},
                    {"name": "s3_secret_access_key", "value": secret_key, "input_type": "VARCHAR"},
                    {"name": "s3_session_token", "value": "", "input_type": "VARCHAR"},
                ],
            },
            use_session_credentials=False,
        )

    first = identity("key-a", "secret-a")
    second = identity("key-b", "secret-b")

    assert first != second
    assert first.settings == second.settings == ()
    assert first.effective_s3_config_identity != second.effective_s3_config_identity
    assert "key-a" not in repr(first)
    assert "secret-a" not in repr(first)


def test_worker_snapshot_database_identity_omits_s3_state_without_httpfs():
    snapshot = {
        "duckdb_source_id": "test-source-id",
        "extensions": [],
        "dynamic_extensions": [],
        "distributed_extension_contracts": [],
        "settings": [],
    }
    first = _snapshot_database_identity(
        snapshot,
        session_id="session-a",
        effective_s3_config={
            "AWS_ACCESS_KEY_ID": "key-a",
            "AWS_SECRET_ACCESS_KEY": "secret-a",
        },
    )
    second = _snapshot_database_identity(
        snapshot,
        session_id="session-b",
        effective_s3_config={
            "AWS_ACCESS_KEY_ID": "key-b",
            "AWS_SECRET_ACCESS_KEY": "secret-b",
        },
        use_session_credentials=False,
    )

    assert first == second
    assert first.s3_session_id == ""
    assert first.effective_s3_config_identity == ""
    assert first.use_session_credentials is False


@pytest.mark.parametrize(
    ("snapshot", "message"),
    [
        ({"extensions": [], "distributed_extension_contracts": [], "settings": []}, "duckdb_source_id"),
        (
            {
                "duckdb_source_id": "test-source-id",
                "extensions": [
                    {"name": "httpfs", "version": "test-version"},
                    {"name": "httpfs", "version": "test-version"},
                ],
                "dynamic_extensions": [],
                "distributed_extension_contracts": [],
                "settings": [],
            },
            "duplicate extension name",
        ),
    ],
)
def test_worker_snapshot_database_identity_rejects_ambiguous_contract(snapshot, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _snapshot_database_identity(snapshot)


def test_worker_snapshot_database_identity_requires_dynamic_extension_manifest():
    with pytest.raises(TypeError, match="dynamic_extensions must be a list"):
        _snapshot_database_identity(
            {
                "duckdb_source_id": "test-source-id",
                "extensions": [],
                "distributed_extension_contracts": [],
                "settings": [],
            }
        )


def test_worker_task_admission_rejects_unprepared_snapshot_database():
    actor_class, actor = _worker_actor()
    database_identity = worker_module.WorkerSnapshotDatabaseIdentity(
        ":memory:",
        False,
        (),
        (),
        "test-source-id",
        (),
        (),
        (),
        "",
        "",
        False,
    )

    with pytest.raises(RuntimeError, match="was not prepared before task admission"):
        actor_class._get_snapshot_execution_cursor(
            actor,
            "query-a",
            database_identity=database_identity,
        )

    assert actor._active_snapshot_execution_cursors == 0


def test_worker_snapshot_cursor_reserves_shutdown_fence_before_cursor_creation():
    actor_class, actor = _worker_actor()
    database_identity = worker_module.WorkerSnapshotDatabaseIdentity(
        ":memory:",
        False,
        (),
        (),
        "test-source-id",
        (),
        (),
        (),
        "",
        "",
        False,
    )

    class Cursor:
        def close(self):
            return None

    class Connection:
        def cursor(self):
            assert actor._active_snapshot_execution_cursors == 1
            return Cursor()

    resolved_connection = Connection()
    actor._snapshot_connections[database_identity] = resolved_connection

    cursor = actor_class._get_snapshot_execution_cursor(
        actor,
        "query-a",
        database_identity=database_identity,
    )
    actor_class._close_snapshot_execution_cursor(actor, cursor)

    assert actor._active_snapshot_execution_cursors == 0


def test_worker_snapshot_cursor_creation_failure_releases_shutdown_fence():
    actor_class, actor = _worker_actor()
    database_identity = worker_module.WorkerSnapshotDatabaseIdentity(
        ":memory:",
        False,
        (),
        (),
        "test-source-id",
        (),
        (),
        (),
        "",
        "",
        False,
    )

    class Connection:
        def cursor(self):
            raise RuntimeError("cursor creation failed")

    resolved_connection = Connection()
    actor._snapshot_connections[database_identity] = resolved_connection

    try:
        actor_class._get_snapshot_execution_cursor(
            actor,
            "query-a",
            database_identity=database_identity,
        )
    except RuntimeError as exc:
        assert str(exc) == "cursor creation failed"
    else:
        raise AssertionError("cursor creation failure was not propagated")

    assert actor._active_snapshot_execution_cursors == 0
