# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

from vane.runners.ray import worker as worker_module


def _worker_actor():
    actor_class = worker_module.RayWorkerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    actor._native_execution_condition = threading.Condition()
    actor._active_secret_snapshot_identity = None
    actor._active_secret_snapshot_leases = 0
    actor._active_secret_snapshot_initialized = False
    actor._active_snapshot_execution_cursors = 0
    actor._closing_native_queries = set()
    actor._closing_native_tasks = set()
    actor._shutdown_started = False
    actor._snapshot_connections = {}
    actor._snapshot_connections_lock = threading.Lock()
    return actor_class, actor


def test_worker_secret_snapshot_identity_includes_bootstrap_and_exact_secrets(monkeypatch):
    snapshots = {
        "shared": {
            "secrets": [
                {"storage": "memory", "name": "z", "payload": b"z-payload"},
                {"storage": "memory", "name": "a", "payload": b"a-payload"},
            ]
        },
        "isolated": {
            "bootstrap": {
                "database": ":memory:",
                "read_only": False,
                "config": {"threads": "2"},
            },
            "secrets": [],
        },
    }
    monkeypatch.setattr(
        worker_module,
        "require_ray_cxx_attr",
        lambda name, *, hint: lambda query_id: snapshots[query_id],
    )

    shared_identity = worker_module._query_worker_secret_snapshot_identity("shared")
    assert shared_identity is not None
    assert shared_identity.secret_count == 2
    assert len(shared_identity.digest) == 32
    empty_identity = worker_module._query_worker_secret_snapshot_identity(
        "shared",
        include_snapshot_secrets=False,
    )
    assert empty_identity is not None
    assert empty_identity.secret_count == 0
    assert len(empty_identity.digest) == 32
    assert shared_identity != empty_identity
    isolated_identity = worker_module._query_worker_secret_snapshot_identity("isolated")
    assert isolated_identity is not None
    assert isolated_identity.secret_count == 0
    assert isolated_identity != empty_identity


def test_worker_snapshot_execution_cursor_caches_nondefault_database(monkeypatch):
    actor_class, actor = _worker_actor()
    database_identity = worker_module.WorkerSnapshotDatabaseIdentity(
        "/tmp/vane-worker-snapshot.duckdb",
        False,
        (("threads", "2"),),
    )
    cursors = []
    resolve_calls = []

    class Cursor:
        def close(self):
            return None

    class ResolvedConnection:
        def cursor(self):
            cursor = Cursor()
            cursors.append(cursor)
            return cursor

    bootstrap_connection = object()
    resolved_connection = ResolvedConnection()
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
                resolved_connection,
            )[1]
        ),
    )

    first = actor_class._get_snapshot_execution_cursor(actor, bootstrap_connection, "query-a")
    second = actor_class._get_snapshot_execution_cursor(actor, bootstrap_connection, "query-b")

    assert first is cursors[0]
    assert second is cursors[1]
    assert resolve_calls == [(bootstrap_connection, "query-a")]
    assert actor._snapshot_connections == {database_identity: resolved_connection}
    actor_class._close_snapshot_execution_cursor(actor, second)
    actor_class._close_snapshot_execution_cursor(actor, first)
    assert actor._active_snapshot_execution_cursors == 0


def test_worker_snapshot_cursor_reserves_shutdown_fence_before_cursor_creation(monkeypatch):
    actor_class, actor = _worker_actor()
    monkeypatch.setattr(
        worker_module,
        "_query_worker_snapshot_database_identity",
        lambda _query_id: worker_module.WorkerSnapshotDatabaseIdentity(":memory:", False, ()),
    )

    class Cursor:
        def close(self):
            return None

    class Connection:
        def cursor(self):
            assert actor._active_snapshot_execution_cursors == 1
            return Cursor()

    cursor = actor_class._get_snapshot_execution_cursor(actor, Connection(), "query-a")
    actor_class._close_snapshot_execution_cursor(actor, cursor)

    assert actor._active_snapshot_execution_cursors == 0


def test_worker_snapshot_cursor_creation_failure_releases_shutdown_fence(monkeypatch):
    actor_class, actor = _worker_actor()
    monkeypatch.setattr(
        worker_module,
        "_query_worker_snapshot_database_identity",
        lambda _query_id: worker_module.WorkerSnapshotDatabaseIdentity(":memory:", False, ()),
    )

    class Connection:
        def cursor(self):
            raise RuntimeError("cursor creation failed")

    try:
        actor_class._get_snapshot_execution_cursor(actor, Connection(), "query-a")
    except RuntimeError as exc:
        assert str(exc) == "cursor creation failed"
    else:
        raise AssertionError("cursor creation failure was not propagated")

    assert actor._active_snapshot_execution_cursors == 0


def test_worker_secret_snapshot_initializes_once_for_concurrent_equal_snapshots(monkeypatch):
    actor_class, actor = _worker_actor()
    identity = worker_module.WorkerSecretSnapshotIdentity(b"a" * 32, 1)
    prepare_started = threading.Event()
    allow_prepare = threading.Event()
    prepare_calls = []
    acquired = []
    errors = []

    monkeypatch.setattr(worker_module, "_query_worker_secret_snapshot_identity", lambda *_args, **_kwargs: identity)

    def prepare(*_args, **_kwargs):
        prepare_calls.append("prepare")
        prepare_started.set()
        assert allow_prepare.wait(timeout=5)

    monkeypatch.setattr(worker_module, "_prepare_query_worker_secret_snapshot", prepare)

    def acquire(label):
        try:
            acquired.append(
                (
                    label,
                    actor_class._acquire_worker_secret_snapshot(
                        actor,
                        object(),
                        f"resource-{label}",
                        native_query_id=f"query-{label}",
                    ),
                )
            )
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=acquire, args=("a",))
    second = threading.Thread(target=acquire, args=("b",))
    first.start()
    assert prepare_started.wait(timeout=5)
    second.start()
    assert acquired == []
    allow_prepare.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert sorted(acquired) == [("a", identity), ("b", identity)]
    assert prepare_calls == ["prepare"]
    assert actor._active_secret_snapshot_leases == 2

    actor_class._release_worker_secret_snapshot(actor, identity)
    actor_class._release_worker_secret_snapshot(actor, identity)
    assert actor._active_secret_snapshot_leases == 0


def test_worker_secret_snapshot_switch_waits_for_previous_domain(monkeypatch):
    actor_class, actor = _worker_actor()
    identities = {
        "resource-a": worker_module.WorkerSecretSnapshotIdentity(b"a" * 32, 1),
        "resource-b": worker_module.WorkerSecretSnapshotIdentity(b"b" * 32, 1),
    }
    prepare_calls = []
    second_acquired = threading.Event()
    errors = []

    monkeypatch.setattr(
        worker_module,
        "_query_worker_secret_snapshot_identity",
        lambda query_id, **_kwargs: identities[query_id],
    )
    monkeypatch.setattr(
        worker_module,
        "_prepare_query_worker_secret_snapshot",
        lambda _connection, query_id, **_kwargs: prepare_calls.append(query_id),
    )

    first_identity = actor_class._acquire_worker_secret_snapshot(
        actor,
        object(),
        "resource-a",
        native_query_id="query-a",
    )

    def acquire_second():
        try:
            identity = actor_class._acquire_worker_secret_snapshot(
                actor,
                object(),
                "resource-b",
                native_query_id="query-b",
            )
            second_acquired.set()
            actor_class._release_worker_secret_snapshot(actor, identity)
        except BaseException as exc:
            errors.append(exc)

    second = threading.Thread(target=acquire_second)
    second.start()
    assert not second_acquired.wait(timeout=0.1)
    assert prepare_calls == ["resource-a"]

    actor_class._release_worker_secret_snapshot(actor, first_identity)
    second.join(timeout=5)

    assert not second.is_alive()
    assert errors == []
    assert second_acquired.is_set()
    assert prepare_calls == ["resource-a", "resource-b"]
    assert actor._active_secret_snapshot_leases == 0
