# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Persistent result contexts: concurrent cleanup and late control requests."""

import gc
import subprocess
import sys
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import vane
from vane.execution import pipelined_runtime, pipelined_worker
from vane.execution.compiler import compile_fragment_graph


def test_release_fences_delayed_creation_without_retaining_query_history():
    actor = pipelined_worker.ResultService(2)
    limits = vane.RayResources()
    actor.create("held", limits, 1)
    for sequence in range(2, 1002):
        actor.release(str(sequence), sequence)
        with pytest.raises(RuntimeError, match="already retired"):
            actor.create(str(sequence), limits, sequence)
    assert tuple(actor.contexts) == ("held",)
    assert actor.retired.ranges == [(2, 1001)]
    actor.release("held", 1)
    assert actor.retired.ranges == [(1, 1001)]
    actor.create("new", limits, 1002)
    actor.release("new", 1002)


def test_out_of_order_release_preserves_live_creation_gaps():
    actor = pipelined_worker.ResultService(4)
    for sequence in (6, 2, 4, 5, 6):
        actor.release(str(sequence), sequence)
    for sequence in (1, 3, 7):
        actor.create(str(sequence), vane.RayResources(), sequence)
    for sequence in (2, 4, 5, 6):
        with pytest.raises(RuntimeError, match="already retired"):
            actor.create(str(sequence), vane.RayResources(), sequence)
    for sequence in (3, 7, 1):
        actor.release(str(sequence), sequence)
    assert actor.retired.ranges == [(1, 7)]


@pytest.mark.parametrize("installed", [False, True])
def test_creation_submit_error_keeps_owner_until_fresh_release(monkeypatch, installed):
    ray = pytest.importorskip("ray")
    limits = vane.RayResources(max_results=1)
    server = pipelined_worker.ResultService(1)

    def create(identity, resources, sequence):
        if installed:
            server.create(identity, resources, sequence)
        raise ray.exceptions.ActorUnavailableError("ambiguous submit", None)

    client = pipelined_runtime.ResultServiceClient(limits)
    client.actor = SimpleNamespace(
        create=SimpleNamespace(remote=create), release=SimpleNamespace(remote=server.release)
    )
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(pipelined_runtime, "_get", lambda *args, **kwargs: None)
    with pytest.raises(ray.exceptions.ActorUnavailableError):
        client.create("query", limits)
    assert client.snapshot()["active_contexts"] == 1
    sequence = client.sequences["query"]
    client.release(client.actor, "query")
    assert client.snapshot()["active_contexts"] == 0
    assert server.contexts == {}
    with pytest.raises(RuntimeError, match="already retired"):
        server.create("query", limits, sequence)


@pytest.mark.parametrize("operation", ["prepare", "connect", "connect_materialized"])
def test_inflight_rpc_retains_its_closed_context(monkeypatch, operation):
    monkeypatch.setattr(pipelined_worker, "_host", lambda: "127.0.0.1")
    with vane.connect() as connection:
        graph = compile_fragment_graph(connection, "select 42", query_id="schema")
        root = next(f for f in graph.fragments if f.fragment_id == graph.result.fragment_id)
        schema = root.outputs[0].schema
    actor = pipelined_worker.ResultService(2)
    actor.create("old", vane.RayResources(), 1)
    actor.create("other", vane.RayResources(), 2)
    actor.prepare("other", schema, "other-ticket")
    entered, proceed = threading.Event(), threading.Event()
    original = getattr(pipelined_worker.ResultContext, operation)

    def delayed(context, *args):
        if context.query_id == "old":
            entered.set()
            assert proceed.wait(5)
        return original(context, *args)

    monkeypatch.setattr(pipelined_worker.ResultContext, operation, delayed)
    args = {"prepare": (schema, "old-ticket"), "connect": ("", "", 1), "connect_materialized": ({}, {})}[operation]
    with ThreadPoolExecutor(1) as threads:
        pending = threads.submit(getattr(actor, operation), "old", *args)
        try:
            assert entered.wait(5)
            actor.release("old", 1)
            actor.create("new", vane.RayResources(), 3)
            actor.prepare("new", schema, "new-ticket")
            proceed.set()
            with pytest.raises(RuntimeError, match="context|canceled|no longer accepts"):
                pending.result(timeout=5)
            actor.cancel("old", "late cancellation")
            actor.release("old", 1)
            for name in ("other", "new"):
                assert actor.status(name)["error"] == ""
                assert not actor.contexts[name].stop.is_set()
        finally:
            proceed.set()
            for sequence, name in enumerate(("old", "other", "new"), 1):
                actor.release(name, sequence)


def test_failed_context_cleanup_retains_capacity_and_other_queries(monkeypatch):
    actor = pipelined_worker.ResultService(2)
    actor.create("a", vane.RayResources(), 1)
    actor.create("b", vane.RayResources(), 2)
    original = actor.contexts["a"].release

    def fail():
        raise RuntimeError("watchdog still running")

    monkeypatch.setattr(actor.contexts["a"], "release", fail)
    with pytest.raises(RuntimeError, match="watchdog"):
        actor.release("a", 1)
    with pytest.raises(RuntimeError, match="capacity"):
        actor.create("c", vane.RayResources(), 3)
    assert not actor.contexts["b"].stop.is_set()
    monkeypatch.setattr(actor.contexts["a"], "release", original)
    actor.release("a", 1)
    actor.create("c", vane.RayResources(), 3)
    actor.release("b", 2)
    actor.release("c", 3)


@pytest.mark.parametrize("error", [TimeoutError("cleanup timeout"), RuntimeError("native cleanup failed")])
def test_failed_release_retains_shared_process_and_context(monkeypatch, error):
    ray = pytest.importorskip("ray")
    actor = SimpleNamespace(release=SimpleNamespace(remote=lambda identity, sequence: identity))
    service = pipelined_runtime.ResultServiceClient(vane.RayResources())
    service.actor = actor
    service.contexts.update(a="created-a", b="created-b")
    service.sequences.update(a=1, b=2)
    killed = []
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "kill", lambda actor, **kwargs: killed.append(actor))

    def get(reference, **kwargs):
        if reference == "a":
            raise error

    monkeypatch.setattr(pipelined_runtime, "_get", get)
    with pytest.raises(type(error), match=str(error)):
        service.release(actor, "a")
    assert service.snapshot()["active_contexts"] == 2
    assert killed == []
    service.release(actor, "b")
    assert service.snapshot()["active_contexts"] == 1
    monkeypatch.setattr(pipelined_runtime, "_get", lambda *args, **kwargs: None)
    service.release(actor, "a")
    service.close()
    assert killed == [actor]


@pytest.mark.parametrize("failure", ["unavailable", "unknown"])
def test_result_service_outage_retains_capacity_until_release_is_confirmed(monkeypatch, failure):
    ray = pytest.importorskip("ray")
    resources = vane.RayResources(max_results=1)
    server = pipelined_worker.ResultService(1)
    server.create("query", resources, 1)
    actor = SimpleNamespace(release=SimpleNamespace(remote=lambda identity, sequence: ("release", identity)))
    client = pipelined_runtime.ResultServiceClient(resources)
    client.actor = actor
    client.contexts["query"] = ("create", "query")
    client.sequences["query"] = 1
    error = (
        ray.exceptions.ActorUnavailableError("temporary transport outage", None)
        if failure == "unavailable"
        else ray.exceptions.RayActorError(error_msg="actor outcome is unknown")
    )
    unavailable = True

    def get(reference, **kwargs):
        operation, query_id = reference
        assert operation == "release"
        if unavailable:
            raise error
        server.release(query_id, 1)

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(pipelined_runtime, "_get", get)
    try:
        for _ in range(2):
            with pytest.raises(type(error), match=str(error)):
                client.release(actor, "query")
            assert client.snapshot()["active_contexts"] == 1
            assert tuple(server.contexts) == ("query",)
            with pytest.raises(RuntimeError, match="capacity is full"):
                client.create("next", resources)
        unavailable = False
        client.release(actor, "query")
        assert client.snapshot()["active_contexts"] == 0
        assert server.contexts == {}
        client.release(actor, "query")
        server.create("next", resources, 2)
    finally:
        for query_id in tuple(server.contexts):
            server.release(query_id, 1 if query_id == "query" else 2)


def test_confirmed_result_actor_death_retires_cleanup_ownership(monkeypatch):
    ray = pytest.importorskip("ray")
    actor = SimpleNamespace(release=SimpleNamespace(remote=lambda identity, sequence: ("release", identity)))
    client = pipelined_runtime.ResultServiceClient(vane.RayResources())
    client.actor = actor
    client.contexts["query"] = ("create", "query")
    client.sequences["query"] = 1
    calls = []

    def get(reference, **kwargs):
        operation, query_id = reference
        calls.append(operation)
        raise ray.exceptions.ActorDiedError()

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(pipelined_runtime, "_get", get)
    client.release(actor, "query")
    assert client.snapshot()["active_contexts"] == 0
    client.release(actor, "query")
    assert calls == ["release"]


def test_service_shutdown_waits_for_inflight_release(monkeypatch):
    ray = pytest.importorskip("ray")
    actor = SimpleNamespace(release=SimpleNamespace(remote=lambda identity, sequence: identity))
    service = pipelined_runtime.ResultServiceClient(vane.RayResources())
    service.actor = actor
    service.contexts["query"] = "created"
    service.sequences["query"] = 1
    entered, proceed = threading.Event(), threading.Event()
    killed = []
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "kill", lambda actor, **kwargs: killed.append(actor))

    def delayed(reference, **kwargs):
        if reference == "query":
            entered.set()
            assert proceed.wait(5)

    monkeypatch.setattr(pipelined_runtime, "_get", delayed)
    with ThreadPoolExecutor(1) as threads:
        pending = threads.submit(service.release, actor, "query")
        try:
            assert entered.wait(5)
            with pytest.raises(RuntimeError, match="cleanup is pending"):
                service.close()
            assert killed == []
        finally:
            proceed.set()
        pending.result(timeout=5)
    service.close()
    assert killed == [actor]


@pytest.mark.parametrize("error_type", [RuntimeError, TimeoutError])
def test_failed_service_termination_retains_retry_and_rejects_new_contexts(monkeypatch, error_type):
    ray = pytest.importorskip("ray")
    service = pipelined_runtime.ResultServiceClient(vane.RayResources())
    actor = service.actor = object()
    attempts = []
    monkeypatch.setattr(ray, "is_initialized", lambda: True)

    def kill(candidate, *, no_restart):
        assert candidate is actor
        assert no_restart
        attempts.append(candidate)
        if len(attempts) <= 2:
            raise error_type("temporary termination failure")

    monkeypatch.setattr(ray, "kill", kill)
    for attempt in (1, 2):
        with pytest.raises(error_type, match="temporary termination failure"):
            service.close()
        assert len(attempts) == attempt
        assert not service.closed
        assert service.actor is actor
        with pytest.raises(RuntimeError, match="closing"):
            service.create("new", vane.RayResources())
        assert service.snapshot()["active_contexts"] == 0
    service.close()
    assert service.closed
    service.close()
    assert attempts == [actor] * 3


def test_runtime_is_lazy_and_connections_have_explicit_service_ownership(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "ray", None)
    with vane.Runtime() as application:
        assert not application.resource_snapshot()["started"]
        with application.connect() as first, application.connect() as second:
            assert first.query_runtime.pool is second.query_runtime.pool
            assert first.query_runtime.session_id != second.query_runtime.session_id
            assert first.query_runtime.pool.workers == []
            first.close()
            assert not second.query_runtime.resource_snapshot()["request_admission"]["draining"]
        assert application.resource_snapshot()["service"]["sessions"] == {}
    assert application.resource_snapshot()["service"]["closed"]
    with pytest.raises(RuntimeError, match="closed"):
        application.connect()


def test_runtime_rejects_implicit_owners_and_invalid_session_capacities():
    with pytest.raises(ValueError, match="Runtime"):
        vane.connect(backend="ray")
    with vane.Runtime(vane.RayResources(max_active_queries=1)) as application:
        with pytest.raises(ValueError, match="exceeds"):
            application.connect(resources=vane.QueryResources(max_active_queries=2))
        with pytest.raises(TypeError, match="QueryResources"):
            application.connect(resources=vane.RayResources())
        with pytest.raises(vane.InvalidInputException):
            application.connect(":default:")
        assert application.resource_snapshot()["service"]["sessions"] == {}
        with pytest.raises(ValueError, match="local"):
            vane.connect(backend="local", runtime=application)


def test_close_racing_connection_creation_cannot_publish_a_new_session(monkeypatch):
    application = vane.Runtime()
    original = application._new_session
    entered, proceed = threading.Event(), threading.Event()

    def create(*args):
        session = original(*args)
        entered.set()
        assert proceed.wait(5)
        return session

    monkeypatch.setattr(application, "_new_session", create)
    with ThreadPoolExecutor(1) as threads:
        pending = threads.submit(application.connect)
        try:
            assert entered.wait(5)
            application.close()
        finally:
            proceed.set()
        with pytest.raises(RuntimeError, match="closed while opening"):
            pending.result(timeout=5)
    assert application.resource_snapshot()["service"]["sessions"] == {}


@pytest.mark.parametrize("factory", ["runtime", "native"])
@pytest.mark.parametrize("drop_intermediate", [False, True])
@pytest.mark.parametrize("closer", ["runtime", "connection"])
def test_runtime_close_releases_native_connections_and_database_lock(tmp_path, factory, drop_intermediate, closer):
    path = str(tmp_path / "session.duckdb")
    application = vane.Runtime()
    connection = (
        application.connect(path) if factory == "runtime" else vane.connect(path, backend="ray", runtime=application)
    )
    cursor = connection.cursor()
    nested = cursor.cursor()
    if drop_intermediate:
        reference = weakref.ref(cursor)
        cursor = None
        gc.collect()
        assert reference() is None
    close = application.close if closer == "runtime" else connection.close
    try:
        close()
        close()
        snapshot = application.resource_snapshot()["service"]
        assert snapshot["sessions"] == {}
        assert snapshot["closed"] == (closer == "runtime")
        for handle in (connection, nested, *((cursor,) if cursor is not None else ())):
            with pytest.raises(vane.ConnectionException, match="closed"):
                handle.execute("select 1")
        opened = subprocess.run(
            [sys.executable, "-I", "-c", "import sys, vane; vane.connect(sys.argv[1]).close()", path],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert opened.returncode == 0, opened.stderr
    finally:
        nested.close()
        if cursor is not None:
            cursor.close()
        connection.close()
        application.close()


def test_orphaned_cursor_close_failure_keeps_session_available_for_retry(tmp_path, monkeypatch):
    application = vane.Runtime()
    connection = application.connect(tmp_path / "session.duckdb")
    nested = connection.cursor().cursor().cursor()
    session = connection.query_runtime
    closed = session._connection_closed
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("native session retirement failed")
        closed()

    try:
        monkeypatch.setattr(session, "_connection_closed", fail_once)
        with pytest.raises(RuntimeError, match="cleanup is pending"):
            application.close()
        snapshot = application.resource_snapshot()["service"]
        assert not snapshot["closed"]
        assert session.session_id in snapshot["sessions"]
        assert connection.query_runtime is session
        assert calls == 1
        reference = weakref.ref(nested)
        nested = None
        gc.collect()
        assert reference() is None
        application.close()
        assert calls == 2
        snapshot = application.resource_snapshot()["service"]
        assert snapshot["closed"] and snapshot["sessions"] == {}
    finally:
        monkeypatch.setattr(session, "_connection_closed", closed)
        if nested is not None:
            nested.close()
        connection.close()
        application.close()


def test_runtime_close_deadline_bounds_drain_and_serializes_retries(monkeypatch):
    application = vane.Runtime()
    connection = application.connect()
    session = connection.query_runtime
    entered, proceed = threading.Event(), threading.Event()
    drain = session.drain
    calls = 0

    def blocked_drain():
        nonlocal calls
        calls += 1
        entered.set()
        assert proceed.wait(10), "test did not release session drain"
        drain()

    try:
        with monkeypatch.context() as patch:
            patch.setattr(session, "drain", blocked_drain)
            with ThreadPoolExecutor(1) as threads:
                started = time.monotonic()
                pending = threads.submit(application.close, timeout=0.2)
                try:
                    assert entered.wait(5)
                    with pytest.raises(TimeoutError, match="retry Runtime.close"):
                        pending.result(timeout=1)
                    assert time.monotonic() - started < 1
                    attempt = application._service.close_attempt
                    assert not attempt.done.is_set()
                    with pytest.raises(TimeoutError, match="retry Runtime.close"):
                        application.close(timeout=0.01)
                    assert application._service.close_attempt is attempt
                    assert calls == 1
                    snapshot = application.resource_snapshot()["service"]
                    assert session.session_id in snapshot["sessions"]
                    assert snapshot["closing"] and not snapshot["closed"]
                    with pytest.raises(RuntimeError, match="closed"):
                        application.connect()
                    with pytest.raises(RuntimeError, match="draining"):
                        connection.query("select 1")
                finally:
                    proceed.set()
            assert attempt.done.wait(5)
        application.close()
        assert application.resource_snapshot()["service"]["sessions"] == {}
        assert application.resource_snapshot()["service"]["closed"]
    finally:
        proceed.set()
        application.close()
        connection.close()


def test_runtime_close_fences_sessions_while_waiting_for_runtime_lock():
    application = vane.Runtime()
    connection = application.connect()
    entered, proceed = threading.Event(), threading.Event()

    def hold_lock():
        with application._lock:
            entered.set()
            assert proceed.wait(5), "test did not release Runtime lock"

    try:
        with ThreadPoolExecutor(1) as threads:
            pending = threads.submit(hold_lock)
            try:
                assert entered.wait(5)
                with pytest.raises(TimeoutError, match="retry Runtime.close"):
                    application.close(timeout=0.01)
                with pytest.raises(RuntimeError, match="draining"):
                    connection.query("select 1")
            finally:
                proceed.set()
            pending.result(timeout=5)
        application.close()
        assert application.resource_snapshot()["service"]["sessions"] == {}
    finally:
        proceed.set()
        application.close()
        connection.close()


def test_materialized_manifest_must_belong_to_its_result_context(monkeypatch):
    from vane.execution.materialized_exchange import ResultManifest

    context = pipelined_worker.ResultContext("expected-query", vane.RayResources())
    monkeypatch.setattr(
        ResultManifest, "from_dict", lambda value: SimpleNamespace(stage=SimpleNamespace(query_id="another-query"))
    )
    with pytest.raises(ValueError, match="another query context"):
        context.connect_materialized("expected-query", {}, {})
    assert context.store_lease is None
    assert context.watchdog is None
