# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic result lease races and cleanup failures."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import vane
from vane.execution import pipelined_runtime, pipelined_worker
from vane.execution.compiler import compile_fragment_graph


@pytest.mark.parametrize("operation", ["prepare", "connect", "connect_materialized"])
def test_inflight_rpc_retains_the_old_session_after_release(monkeypatch, operation):
    monkeypatch.setattr(pipelined_worker, "_host", lambda: "127.0.0.1")
    with vane.connect() as connection:
        graph = compile_fragment_graph(connection, "select 42", query_id="schema")
        root = next(f for f in graph.fragments if f.fragment_id == graph.result.fragment_id)
        schema = root.outputs[0].schema
    actor = pipelined_worker.ResultService()
    actor.reserve("old", vane.RayResources())
    entered, proceed = threading.Event(), threading.Event()
    original = getattr(pipelined_worker._ResultSession, operation)

    def delayed(session, *args):
        if session.epoch == "old":
            entered.set()
            assert proceed.wait(5)
        return original(session, *args)

    monkeypatch.setattr(pipelined_worker._ResultSession, operation, delayed)
    args = {"prepare": (schema, "old-ticket"), "connect": ("", "", 1), "connect_materialized": ({}, {})}[operation]
    with ThreadPoolExecutor(1) as threads:
        pending = threads.submit(getattr(actor, operation), "old", *args)
        try:
            assert entered.wait(5)
            actor.release("old")
            actor.reserve("new", vane.RayResources())
            actor.prepare("new", schema, "new-ticket")
            proceed.set()
            with pytest.raises(RuntimeError, match="epoch|canceled|no longer accepts"):
                pending.result(timeout=5)
            actor.cancel("old", "late cancellation")
            actor.release("old")
            assert actor.status("new")["error"] == ""
            assert not actor.session.stop.is_set()
        finally:
            proceed.set()
            actor.release("old")
            actor.release("new")


@pytest.mark.parametrize("error", [TimeoutError("cleanup timeout"), RuntimeError("native cleanup failed")])
def test_failed_release_evicts_the_actor_instead_of_caching_it(monkeypatch, error):
    ray = pytest.importorskip("ray")
    actor = SimpleNamespace(release=SimpleNamespace(remote=lambda epoch: epoch))
    pool = pipelined_runtime.ResultServicePool(vane.RayResources())
    pool.leased["epoch"] = actor
    killed = []
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "kill", lambda actor, **kwargs: killed.append(actor))

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(pipelined_runtime, "_get", fail)
    with pytest.raises(type(error), match=str(error)):
        pool.release(actor, "epoch")
    assert pool.snapshot() == {"leased": 0, "idle": 0, "capacity": 4}
    assert killed == [actor]
    pool.release(actor, "epoch")
    assert killed == [actor]


def test_session_close_cannot_recache_an_inflight_release(monkeypatch):
    ray = pytest.importorskip("ray")
    actor = SimpleNamespace(release=SimpleNamespace(remote=lambda epoch: epoch))
    pool = pipelined_runtime.ResultServicePool(vane.RayResources())
    pool.leased["epoch"] = actor
    entered, proceed = threading.Event(), threading.Event()
    killed = []
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "kill", lambda actor, **kwargs: killed.append(actor))

    def delayed(*args, **kwargs):
        entered.set()
        assert proceed.wait(5)

    monkeypatch.setattr(pipelined_runtime, "_get", delayed)
    with ThreadPoolExecutor(1) as threads:
        pending = threads.submit(pool.release, actor, "epoch")
        try:
            assert entered.wait(5)
            pool.close()
        finally:
            proceed.set()
        pending.result(timeout=5)
    assert pool.snapshot() == {"leased": 0, "idle": 0, "capacity": 4}
    assert killed == [actor]
