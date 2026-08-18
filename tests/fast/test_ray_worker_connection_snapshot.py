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
        {"dynamic_extensions": [], **snapshot},
        session_id=session_id,
        effective_s3_config=effective_s3_config or {},
        use_session_credentials=use_session_credentials,
    )


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
        "",
        "",
        False,
    )
    cursors = []
    resolve_calls = []
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

    bootstrap_connection = object()
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
            lambda connection, query_id: (
                resolve_calls.append((connection, query_id)),
                lifecycle.append("resolve"),
                resolved_connection,
            )[2]
        ),
    )

    first = actor_class._get_snapshot_execution_cursor(
        actor,
        bootstrap_connection,
        "query-a",
        database_identity=database_identity,
    )
    second = actor_class._get_snapshot_execution_cursor(
        actor,
        bootstrap_connection,
        "query-b",
        database_identity=database_identity,
    )

    assert first is cursors[0]
    assert second is cursors[1]
    assert resolve_calls == [(bootstrap_connection, "query-a")]
    assert configured_connections == [resolved_connection]
    assert lifecycle == ["resolve", "configure", "cursor", "cursor"]
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

    def resolve(_connection, _query_id):
        connection = ResolvedConnection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(worker_module, "require_ray_cxx_attr", lambda name, *, hint: resolve)

    old_cursor = actor_class._get_snapshot_execution_cursor(
        actor,
        object(),
        "old-query",
        database_identity=old_identity,
    )
    new_cursor = actor_class._get_snapshot_execution_cursor(
        actor,
        object(),
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

    late_cleanup_cursor = actor_class._get_snapshot_execution_cursor(
        actor,
        object(),
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
        lambda name, *, hint: lambda _connection, _query_id: resolved_connection,
    )

    with pytest.raises(RuntimeError, match="snapshot configuration failed"):
        actor_class._get_snapshot_execution_cursor(
            actor,
            object(),
            "query-a",
            database_identity=database_identity,
        )

    assert resolved_connection.closed is True
    assert actor._snapshot_connections == {}
    assert actor._active_snapshot_execution_cursors == 0


def test_worker_snapshot_execution_cursor_isolates_exact_extension_identities(monkeypatch):
    actor_class, actor = _worker_actor()
    base_snapshot = {
        "duckdb_source_id": "test-source-id",
        "extensions": [],
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

    def resolve(_connection, query_id):
        connection = ResolvedConnection(query_id)
        created_connections.append(connection)
        return connection

    monkeypatch.setattr(
        worker_module,
        "require_ray_cxx_attr",
        lambda name, *, hint: resolve,
    )

    plain_a = actor_class._get_snapshot_execution_cursor(
        actor,
        object(),
        "plain-a",
        database_identity=identities["plain-a"],
    )
    plain_b = actor_class._get_snapshot_execution_cursor(
        actor,
        object(),
        "plain-b",
        database_identity=identities["plain-b"],
    )
    httpfs = actor_class._get_snapshot_execution_cursor(
        actor,
        object(),
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


def test_worker_snapshot_database_identity_includes_dynamic_artifact_identity():
    base_snapshot = {
        "duckdb_source_id": "test-source-id",
        "extensions": [],
        "dynamic_extensions": [],
        "distributed_extension_contracts": [],
        "settings": [],
    }

    def dynamic_snapshot(digest):
        return {
            **base_snapshot,
            "dynamic_extensions": [
                {
                    "descriptor": {
                        "format_version": 1,
                        "name": "dynamic_test",
                        "extension_version": "test-version",
                        "abi_type": "CPP",
                        "duckdb_source_id": "a" * 40,
                        "vane_version": "test-vane-version",
                        "platform": "linux_amd64",
                        "sha256": digest,
                        "trust_identity": "test-provider",
                        "dependencies": [],
                    },
                    "dependency_order": [],
                }
            ],
        }

    no_dynamic_identity = _snapshot_database_identity(base_snapshot)
    first_dynamic_identity = _snapshot_database_identity(dynamic_snapshot("1" * 64))
    same_dynamic_identity = _snapshot_database_identity(dynamic_snapshot("1" * 64))
    altered_dynamic_identity = _snapshot_database_identity(dynamic_snapshot("2" * 64))

    assert first_dynamic_identity == same_dynamic_identity
    assert no_dynamic_identity != first_dynamic_identity
    assert first_dynamic_identity != altered_dynamic_identity
    assert first_dynamic_identity.dynamic_extensions


def test_worker_snapshot_execution_cursor_isolates_replayed_settings(monkeypatch):
    actor_class, actor = _worker_actor()
    base_snapshot = {
        "duckdb_source_id": "test-source-id",
        "extensions": [],
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

    def resolve(_connection, query_id):
        connection = ResolvedConnection(query_id)
        created_connections.append(connection)
        return connection

    monkeypatch.setattr(
        worker_module,
        "require_ray_cxx_attr",
        lambda name, *, hint: resolve,
    )

    default_cursor = actor_class._get_snapshot_execution_cursor(
        actor,
        object(),
        "default",
        database_identity=identities["default"],
    )
    proxy_cursor = actor_class._get_snapshot_execution_cursor(
        actor,
        object(),
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


def test_worker_snapshot_database_identity_requires_dynamic_extension_snapshot():
    with pytest.raises(TypeError, match="dynamic_extensions must be a list"):
        worker_module._worker_snapshot_database_identity(
            {
                "duckdb_source_id": "test-source-id",
                "extensions": [],
                "distributed_extension_contracts": [],
                "settings": [],
            },
            session_id="test-session",
            effective_s3_config={},
            use_session_credentials=True,
        )


def test_worker_snapshot_cursor_reserves_shutdown_fence_before_cursor_creation(monkeypatch):
    actor_class, actor = _worker_actor()
    database_identity = worker_module.WorkerSnapshotDatabaseIdentity(
        ":memory:",
        False,
        (),
        (),
        "test-source-id",
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
    monkeypatch.setattr(
        worker_module,
        "require_ray_cxx_attr",
        lambda name, *, hint: lambda _connection, _query_id: resolved_connection,
    )

    cursor = actor_class._get_snapshot_execution_cursor(
        actor,
        object(),
        "query-a",
        database_identity=database_identity,
    )
    actor_class._close_snapshot_execution_cursor(actor, cursor)

    assert actor._active_snapshot_execution_cursors == 0


def test_worker_snapshot_cursor_creation_failure_releases_shutdown_fence(monkeypatch):
    actor_class, actor = _worker_actor()
    database_identity = worker_module.WorkerSnapshotDatabaseIdentity(
        ":memory:",
        False,
        (),
        (),
        "test-source-id",
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
    monkeypatch.setattr(
        worker_module,
        "require_ray_cxx_attr",
        lambda name, *, hint: lambda _connection, _query_id: resolved_connection,
    )

    try:
        actor_class._get_snapshot_execution_cursor(
            actor,
            object(),
            "query-a",
            database_identity=database_identity,
        )
    except RuntimeError as exc:
        assert str(exc) == "cursor creation failed"
    else:
        raise AssertionError("cursor creation failure was not propagated")

    assert actor._active_snapshot_execution_cursors == 0
