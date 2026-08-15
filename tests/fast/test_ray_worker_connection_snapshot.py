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
    actor._configure_snapshot_conn = lambda _connection: None
    return actor_class, actor


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
        "_query_worker_snapshot_database_identity",
        lambda _query_id: database_identity,
    )
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

    first = actor_class._get_snapshot_execution_cursor(actor, bootstrap_connection, "query-a")
    second = actor_class._get_snapshot_execution_cursor(actor, bootstrap_connection, "query-b")

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
        "_query_worker_snapshot_database_identity",
        lambda _query_id: database_identity,
    )
    monkeypatch.setattr(
        worker_module,
        "require_ray_cxx_attr",
        lambda name, *, hint: lambda _connection, _query_id: resolved_connection,
    )

    with pytest.raises(RuntimeError, match="snapshot configuration failed"):
        actor_class._get_snapshot_execution_cursor(actor, object(), "query-a")

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
        "plain-a": worker_module._worker_snapshot_database_identity(base_snapshot),
        "plain-b": worker_module._worker_snapshot_database_identity(base_snapshot),
        "httpfs": worker_module._worker_snapshot_database_identity(httpfs_snapshot),
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
        "_query_worker_snapshot_database_identity",
        lambda query_id: identities[query_id],
    )
    monkeypatch.setattr(
        worker_module,
        "require_ray_cxx_attr",
        lambda name, *, hint: resolve,
    )

    plain_a = actor_class._get_snapshot_execution_cursor(actor, object(), "plain-a")
    plain_b = actor_class._get_snapshot_execution_cursor(actor, object(), "plain-b")
    httpfs = actor_class._get_snapshot_execution_cursor(actor, object(), "httpfs")

    assert plain_a.connection is plain_b.connection
    assert httpfs.connection is not plain_a.connection
    assert len(created_connections) == 2
    assert len(actor._snapshot_connections) == 2

    actor_class._close_snapshot_execution_cursor(actor, httpfs)
    actor_class._close_snapshot_execution_cursor(actor, plain_b)
    actor_class._close_snapshot_execution_cursor(actor, plain_a)
    assert actor._active_snapshot_execution_cursors == 0


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
        "default": worker_module._worker_snapshot_database_identity(base_snapshot),
        "proxy": worker_module._worker_snapshot_database_identity(proxy_snapshot),
    }
    assert identities["default"] != identities["proxy"]
    assert identities["proxy"] == worker_module._worker_snapshot_database_identity(
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
        "_query_worker_snapshot_database_identity",
        lambda query_id: identities[query_id],
    )
    monkeypatch.setattr(
        worker_module,
        "require_ray_cxx_attr",
        lambda name, *, hint: resolve,
    )

    default_cursor = actor_class._get_snapshot_execution_cursor(actor, object(), "default")
    proxy_cursor = actor_class._get_snapshot_execution_cursor(actor, object(), "proxy")

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

    assert worker_module._worker_snapshot_database_identity(
        snapshot
    ) == worker_module._worker_snapshot_database_identity(string_config_snapshot)


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
        worker_module._worker_snapshot_database_identity(snapshot)


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
    )
    monkeypatch.setattr(
        worker_module,
        "_query_worker_snapshot_database_identity",
        lambda _query_id: database_identity,
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

    cursor = actor_class._get_snapshot_execution_cursor(actor, object(), "query-a")
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
    )
    monkeypatch.setattr(
        worker_module,
        "_query_worker_snapshot_database_identity",
        lambda _query_id: database_identity,
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
        actor_class._get_snapshot_execution_cursor(actor, object(), "query-a")
    except RuntimeError as exc:
        assert str(exc) == "cursor creation failed"
    else:
        raise AssertionError("cursor creation failure was not propagated")

    assert actor._active_snapshot_execution_cursors == 0
