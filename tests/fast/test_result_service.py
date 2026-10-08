# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Persistent result contexts: concurrent cleanup and late control requests."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import vane
from vane.execution import pipelined_runtime, pipelined_worker
from vane.execution.compiler import compile_fragment_graph


@pytest.mark.parametrize("operation", ["prepare", "connect", "connect_materialized"])
def test_inflight_rpc_retains_its_closed_context(monkeypatch, operation):
    monkeypatch.setattr(pipelined_worker, "_host", lambda: "127.0.0.1")
    with vane.connect() as connection:
        graph = compile_fragment_graph(connection, "select 42", query_id="schema")
        root = next(f for f in graph.fragments if f.fragment_id == graph.result.fragment_id)
        schema = root.outputs[0].schema
    actor = pipelined_worker.ResultService(2)
    actor.create("old", vane.RayResources())
    actor.create("other", vane.RayResources())
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
            actor.release("old")
            actor.create("new", vane.RayResources())
            actor.prepare("new", schema, "new-ticket")
            proceed.set()
            with pytest.raises(RuntimeError, match="context|canceled|no longer accepts"):
                pending.result(timeout=5)
            actor.cancel("old", "late cancellation")
            actor.release("old")
            for name in ("other", "new"):
                assert actor.status(name)["error"] == ""
                assert not actor.contexts[name].stop.is_set()
        finally:
            proceed.set()
            for name in ("old", "other", "new"):
                actor.release(name)


def test_failed_context_cleanup_retains_capacity_and_other_queries(monkeypatch):
    actor = pipelined_worker.ResultService(2)
    actor.create("a", vane.RayResources())
    actor.create("b", vane.RayResources())
    original = actor.contexts["a"].release

    def fail():
        raise RuntimeError("watchdog still running")

    monkeypatch.setattr(actor.contexts["a"], "release", fail)
    with pytest.raises(RuntimeError, match="watchdog"):
        actor.release("a")
    with pytest.raises(RuntimeError, match="capacity"):
        actor.create("c", vane.RayResources())
    assert not actor.contexts["b"].stop.is_set()
    monkeypatch.setattr(actor.contexts["a"], "release", original)
    actor.release("a")
    actor.create("c", vane.RayResources())
    actor.release("b")
    actor.release("c")


@pytest.mark.parametrize("error", [TimeoutError("cleanup timeout"), RuntimeError("native cleanup failed")])
def test_failed_release_retains_shared_process_and_context(monkeypatch, error):
    ray = pytest.importorskip("ray")
    actor = SimpleNamespace(release=SimpleNamespace(remote=lambda identity: identity))
    service = pipelined_runtime.ResultServiceClient(vane.RayResources())
    service.actor = actor
    service.contexts.update(a="created-a", b="created-b")
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


def test_service_shutdown_waits_for_inflight_release(monkeypatch):
    ray = pytest.importorskip("ray")
    actor = SimpleNamespace(release=SimpleNamespace(remote=lambda identity: identity))
    service = pipelined_runtime.ResultServiceClient(vane.RayResources())
    service.actor = actor
    service.contexts["query"] = "created"
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
