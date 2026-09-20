# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from vane.execution import udf_local_request as local
from vane.execution.request_admission import RequestAdmissionLimits, RequestCancelled, RequestQueueFull
from vane.execution.udf_actor_pool_lifecycle import OwnedActorPoolsError
from vane.execution.udf_data_lease import QueryDataScope
from vane.execution.udf_local_model import LocalModelRuntime
from vane.execution.udf_runtime_admission import TaskAdmissionLimits


class Resource:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.pending = True
        self.calls = 0

    def shutdown(self, *, kill=False):
        self.calls += 1
        if self.fail:
            raise RuntimeError("planned request resource cleanup failure")
        self.pending = False

    def cleanup_pending(self):
        return self.pending


def make_runtime(**options):
    return LocalModelRuntime(
        session_id="session", session_config={}, request_limit=RequestAdmissionLimits(1, 2), **options
    )


def model_plan(monkeypatch, runtime, create):
    from vane import pickle as vane_pickle
    from vane.execution import udf_subprocess

    monkeypatch.setattr(udf_subprocess, "LocalSubprocessActorPool", lambda *a, **k: create())
    payload = {
        "function_pickle": vane_pickle.dumps(lambda table: table),
        "execution_backend": "subprocess_actor",
        "call_mode": "map_batches",
        "actor_number": 1,
    }
    published = {}
    plan = SimpleNamespace(
        session_id=lambda: "session",
        session_config=lambda: {},
        collect_udf_nodes=lambda **k: [{"node_id": "1", "payload": payload}],
        set_udf_actor_handles=lambda options, **k: published.update(options),
    )
    model = runtime.register("model", version="v1", payload=payload)
    return model, plan, published


@pytest.mark.parametrize("operation", ["acquire", "prewarm"])
def test_public_model_initialization_crossing_drain_cannot_publish_borrow(monkeypatch, operation):
    runtime = make_runtime()
    entered, proceed = threading.Event(), threading.Event()
    pool = Resource()

    def create():
        entered.set()
        assert proceed.wait(5)
        return pool

    model, _, _ = model_plan(monkeypatch, runtime, create)
    borrow = None
    try:
        with ThreadPoolExecutor(max_workers=1) as threads:
            future = threads.submit(getattr(model, operation))
            try:
                assert entered.wait(3)
                runtime.drain()
                assert runtime.resource_snapshot()["reserved_models"] == 1
            finally:
                proceed.set()
            with pytest.raises(RuntimeError, match="draining"):
                borrow = future.result(timeout=5)
        assert runtime.resource_snapshot()["active_borrows"] == 0
        assert pool.pending  # The initialized pool still belongs to the runtime.
    finally:
        if borrow is not None:
            borrow.release()
        runtime.close()
    assert not pool.pending


@pytest.mark.parametrize("drain", [False, True])
def test_prepared_model_handle_permission_expires_with_request(monkeypatch, drain):
    runtime = make_runtime()
    model, plan, published = model_plan(monkeypatch, runtime, Resource)

    def execute(*args):
        if drain:
            runtime.drain()
            with pytest.raises(RuntimeError, match="draining"):
                model.prewarm()
        with published["1"]["local_model_pool"].acquire():
            assert runtime.resource_snapshot()["active_borrows"] == 2
        return "result"

    monkeypatch.setattr(local, "_execute_native", execute)
    try:
        assert runtime.request().execute(plan, {"1": "model"}, conn=object()) == "result"
        prepared_model = published["1"]["local_model_pool"]
        assert prepared_model is not model
        for operation in (prepared_model.acquire, prepared_model.prewarm):
            with pytest.raises(RuntimeError, match="live claim"):
                operation()
        assert runtime.resource_snapshot()["active_borrows"] == 0
        if not drain:
            model.prewarm()
    finally:
        runtime.close()


def test_drain_preserves_existing_public_borrow_until_its_owner_releases_it(monkeypatch):
    runtime = make_runtime()
    pool = Resource()
    model, _, _ = model_plan(monkeypatch, runtime, lambda: pool)
    borrow = model.acquire()
    try:
        runtime.drain()
        assert borrow.pool is pool
        with pytest.raises(RuntimeError, match="draining"):
            model.acquire()
        with pytest.raises(TimeoutError, match="active borrows"):
            runtime.close()
    finally:
        borrow.release()
        runtime.close()
    assert not pool.pending


def test_request_gate_is_opt_in_and_cannot_be_bypassed_through_prepare():
    with LocalModelRuntime(session_id="session", session_config={}) as runtime:
        with pytest.raises(RuntimeError, match="configured request_limit"):
            runtime.request()
    with make_runtime() as runtime:
        with pytest.raises(RuntimeError, match="request.*execute"):
            runtime.prepare(object(), {})


def test_unstarted_requests_do_not_prepare_or_execute(monkeypatch):
    with make_runtime() as runtime:
        monkeypatch.setattr(runtime, "_prepare", lambda *a, **k: pytest.fail("unadmitted request prepared"))
        monkeypatch.setattr(local, "_execute_native", lambda *a: pytest.fail("unadmitted request executed"))
        first, second, third = runtime.request(), runtime.request(), runtime.request()
        with pytest.raises(RequestQueueFull):
            runtime.request()
        assert second.cancel()
        with pytest.raises(RequestCancelled):
            second.execute(object(), {}, conn=object())
        first.shutdown()
        assert third.state == "ready"
        third.shutdown()


@pytest.mark.parametrize("fails", [False, True])
def test_execution_cleans_resources_once_and_allows_the_next_request(monkeypatch, fails):
    with make_runtime() as runtime:
        resources = [Resource(), Resource()]
        monkeypatch.setattr(runtime, "_prepare", lambda *a, **k: resources)
        calls = []

        def execute(conn, plan):
            calls.append(plan)
            if fails:
                raise ValueError("planned execution failure")
            return "result"

        monkeypatch.setattr(local, "_execute_native", execute)
        first, queued = runtime.request(), runtime.request()
        if fails:
            with pytest.raises(ValueError, match="planned execution failure"):
                first.execute("plan", {}, conn="cursor")
        else:
            assert first.execute("plan", {}, conn="cursor") == "result"
        first.shutdown()
        assert [resource.calls for resource in resources] == [1, 1]
        assert first.state == "finished" and queued.state == "ready"
        with pytest.raises(RuntimeError, match="only execute once"):
            first.execute("plan", {}, conn="cursor")
        assert calls == ["plan"]
        queued.shutdown()


@pytest.mark.parametrize("stage", ["preparation", "execution", "cleanup"])
def test_failed_cleanup_retains_request_and_runtime_owns_retry(monkeypatch, stage):
    runtime = make_runtime()
    owner = Resource(fail=True)

    def prepare(*args, **kwargs):
        if stage == "preparation":
            raise OwnedActorPoolsError(
                "planned preparation failure", owned_actor_pools=[owner], creation_error=ValueError("constructor")
            )
        return [owner]

    def execute(*args):
        if stage == "execution":
            raise ValueError("planned execution failure")
        return "result"

    monkeypatch.setattr(runtime, "_prepare", prepare)
    monkeypatch.setattr(local, "_execute_native", execute)
    request = runtime.request()
    try:
        expected = ValueError if stage == "execution" else RuntimeError
        with pytest.raises(expected):
            request.execute(object(), {}, conn=object())
        queued = runtime.request()
        assert queued.state == "queued"
        snapshot = runtime.resource_snapshot()["request_admission"]
        assert snapshot["running_requests"] == snapshot["cleanup_pending_requests"] == 1
        weak_request = weakref.ref(request)
        del request
        gc.collect()
        assert weak_request() is not None
        with pytest.raises(RuntimeError, match="cleanup failed during runtime close"):
            runtime.close()
        assert queued.state == "drained"
        owner.fail = False
        runtime.close()
        gc.collect()
        assert weak_request() is None
        assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
    finally:
        owner.fail = False
        runtime.close()


def test_concurrent_cleanup_cannot_return_request_capacity_early(monkeypatch):
    runtime = make_runtime()
    entered, proceed = threading.Event(), threading.Event()
    owner = Resource()

    def shutdown(*, kill=False):
        entered.set()
        assert proceed.wait(5)
        owner.pending = False

    monkeypatch.setattr(owner, "shutdown", shutdown)
    monkeypatch.setattr(runtime, "_prepare", lambda *a, **k: [owner])
    monkeypatch.setattr(local, "_execute_native", lambda *a: "result")
    request, queued = runtime.request(), runtime.request()
    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(request.execute, object(), {}, conn=object())
        try:
            assert entered.wait(3)
            with pytest.raises(RuntimeError, match="cleanup is still in progress"):
                request.shutdown()
            assert not request.cancel()
            assert queued.state == "queued"
            with pytest.raises(TimeoutError, match="active execution or cleanup"):
                runtime.close()
        finally:
            proceed.set()
        assert future.result(timeout=5) == "result"
    runtime.close()


@pytest.mark.parametrize("scope", ["request", "runtime", "both"])
def test_context_preserves_primary_error_when_cleanup_retry_fails(monkeypatch, scope):
    runtime = make_runtime()
    owner = Resource(fail=True)
    monkeypatch.setattr(runtime, "_prepare", lambda *a, **k: [owner])

    def execute(*args):
        raise ValueError("primary execution error")

    monkeypatch.setattr(local, "_execute_native", execute)
    try:
        with pytest.raises(ValueError, match="primary execution error") as info:
            with ExitStack() as contexts:
                if scope in {"runtime", "both"}:
                    contexts.enter_context(runtime)
                request = runtime.request()
                if scope in {"request", "both"}:
                    contexts.enter_context(request)
                request.execute(object(), {}, conn=object())
        assert "request cleanup failed" in str(info.value.__cause__)
        state = runtime.resource_snapshot()["request_admission"]
        assert state["active_requests"] == state["cleanup_pending_requests"] == 1
        assert owner.calls == (3 if scope == "both" else 2)
    finally:
        owner.fail = False
        runtime.close()


def test_drain_allows_claimed_preparation_to_finish_and_cancels_waiters(monkeypatch):
    runtime = make_runtime()
    entered, proceed = threading.Event(), threading.Event()

    def prepare(*args, **kwargs):
        entered.set()
        assert proceed.wait(5)
        assert not runtime._registry.resource_snapshot()["draining"]
        return [Resource()]

    monkeypatch.setattr(runtime, "_prepare", prepare)
    monkeypatch.setattr(local, "_execute_native", lambda *a: "result")
    request, queued = runtime.request(), runtime.request()
    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(request.execute, object(), {}, conn=object())
        try:
            assert entered.wait(3)
            runtime.drain()
            with pytest.raises(RequestCancelled):
                queued.execute(object(), {}, conn=object())
            with pytest.raises(RuntimeError, match="draining"):
                runtime.request()
            with pytest.raises(RuntimeError, match="draining"):
                runtime.prewarm("unused")
            with pytest.raises(RuntimeError, match="draining"):
                runtime.register("unused", version="v1", payload={})
            with pytest.raises(RuntimeError, match="execution must finish"):
                request.shutdown()
        finally:
            proceed.set()
        assert future.result(timeout=5) == "result"
    runtime.close()


def test_successful_shutdown_that_reports_pending_still_retains_capacity(monkeypatch):
    runtime = make_runtime()
    owner = Resource()
    monkeypatch.setattr(owner, "shutdown", lambda **kwargs: None)
    monkeypatch.setattr(runtime, "_prepare", lambda *a, **k: [owner])
    monkeypatch.setattr(local, "_execute_native", lambda *a: None)
    request = runtime.request()
    with pytest.raises(RuntimeError, match="request cleanup failed"):
        request.execute(object(), {}, conn=object())
    assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 1
    owner.pending = False
    runtime.close()


@pytest.mark.parametrize("status_fails", [False, True])
def test_cleanup_status_controls_retirement_after_shutdown_error(monkeypatch, status_fails):
    runtime = make_runtime()
    owner = Resource()

    def cleanup(**kwargs):
        owner.pending = False
        raise RuntimeError("failure after owner was closed")

    def unknown_status():
        raise RuntimeError("cleanup status unavailable")

    monkeypatch.setattr(owner, "shutdown", cleanup)
    if status_fails:
        monkeypatch.setattr(owner, "cleanup_pending", unknown_status)
    monkeypatch.setattr(runtime, "_prepare", lambda *a, **k: [owner])
    monkeypatch.setattr(local, "_execute_native", lambda *a: None)
    with pytest.raises(RuntimeError, match="request cleanup failed"):
        runtime.request().execute(object(), {}, conn=object())
    assert runtime.resource_snapshot()["request_admission"]["active_requests"] == int(status_fails)
    monkeypatch.setattr(owner, "shutdown", lambda **kwargs: None)
    monkeypatch.setattr(owner, "cleanup_pending", lambda: False)
    runtime.close()


def test_runtime_close_attempts_every_pending_request(monkeypatch):
    runtime = LocalModelRuntime(session_id="session", session_config={}, request_limit=RequestAdmissionLimits(2, 1))
    owners = [Resource(fail=True), Resource(fail=True)]
    monkeypatch.setattr(runtime, "_prepare", lambda plan, *a, **k: [owners[plan]])
    monkeypatch.setattr(local, "_execute_native", lambda *a: None)
    for index in range(2):
        with pytest.raises(RuntimeError, match="request cleanup failed"):
            runtime.request().execute(index, {}, conn=object())
    owners[1].fail = False
    with pytest.raises(RuntimeError, match="cleanup failed during runtime close"):
        runtime.close()
    assert [owner.calls for owner in owners] == [2, 2]
    assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 1
    owners[0].fail = False
    runtime.close()


def test_failed_publication_keeps_query_cleanup_owners(monkeypatch):
    from vane import pickle as vane_pickle

    class Plan:
        def session_id(self):
            return "session"

        def session_config(self):
            return {}

        def collect_udf_nodes(self, conn=None):
            return [
                {
                    "node_id": "one",
                    "payload": {
                        "function_pickle": vane_pickle.dumps(lambda table: table),
                        "execution_backend": "subprocess_task",
                        "call_mode": "map_batches",
                    },
                }
            ]

        def set_udf_actor_handles(self, options, conn=None):
            raise ValueError("planned publication failure")

    runtime = make_runtime(track_data=True, task_limit=TaskAdmissionLimits(1, 1))
    shutdown = QueryDataScope.shutdown

    def fail(*args, **kwargs):
        raise RuntimeError("planned query cleanup failure")

    with monkeypatch.context() as patch:
        patch.setattr(QueryDataScope, "shutdown", fail)
        with pytest.raises(OwnedActorPoolsError) as info:
            runtime.request().execute(Plan(), {}, conn=object())
        assert isinstance(info.value.creation_error, ValueError)
        assert runtime.resource_snapshot()["data"]["queries"] == 1
        assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 1
    assert QueryDataScope.shutdown is shutdown
    runtime.close()
    assert runtime.resource_snapshot()["data"]["queries"] == 0


def test_reentrant_cleanup_runs_outside_request_and_runtime_locks(monkeypatch):
    with make_runtime() as runtime, ThreadPoolExecutor(max_workers=1) as threads:
        owner = Resource()
        shutdown = owner.shutdown

        def cleanup(**kwargs):
            state = threads.submit(runtime.resource_snapshot).result(timeout=3)
            assert state["request_admission"]["active_requests"] == 1
            shutdown(**kwargs)

        monkeypatch.setattr(owner, "shutdown", cleanup)
        monkeypatch.setattr(runtime, "_prepare", lambda *a, **k: [owner])
        monkeypatch.setattr(local, "_execute_native", lambda *a: "result")
        assert runtime.request().execute(object(), {}, conn=object()) == "result"
