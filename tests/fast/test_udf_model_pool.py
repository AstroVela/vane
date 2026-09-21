# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vane.execution.resources import ResourceVector
from vane.execution.udf_actor_pool_lifecycle import OwnedActorPoolsError
from vane.execution.udf_lifecycle import ExecutionCancellationScope, ExecutionCancelledError
from vane.execution.udf_model_pool import ModelPoolIdentity, ModelPoolRegistry


def _identity(**changes):
    return replace(ModelPoolIdentity("session", "encoder", "v1", "subprocess_actor", "weights-a", "cpu-1"), **changes)


@pytest.mark.parametrize("cancel_initializer", [False, True])
def test_cancelled_borrow_does_not_cancel_shared_initialization(cancel_initializer):
    registry = ModelPoolRegistry()
    pool = _Pool()
    entered, proceed = threading.Event(), threading.Event()
    cancellation = ExecutionCancellationScope("request", 1)

    def create():
        entered.set()
        assert proceed.wait(5)
        return pool

    registry.register(_identity(), create)
    with ThreadPoolExecutor(max_workers=2) as threads:
        initial = threads.submit(
            registry.acquire, _identity(), cancellation=cancellation if cancel_initializer else None
        )
        try:
            assert entered.wait(3)
            waiter = threads.submit(
                registry.acquire, _identity(), cancellation=None if cancel_initializer else cancellation
            )
            cancellation.cancel()
            if not cancel_initializer:
                with pytest.raises(ExecutionCancelledError):
                    waiter.result(timeout=3)
                assert not initial.done()
        finally:
            proceed.set()
        with pytest.raises(ExecutionCancelledError):
            (initial if cancel_initializer else waiter).result(timeout=3)
        borrow = (waiter if cancel_initializer else initial).result(timeout=3)
        assert borrow.pool is pool
        borrow.release()
        with registry.acquire(_identity()) as later:
            assert later.pool is pool
    registry.close()


class _Pool:
    def __init__(self):
        self.closed = False
        self.calls = []
        self.fail = False

    def shutdown(self, *, kill=False):
        self.calls.append(kill)
        if self.fail:
            raise RuntimeError("worker cleanup failed")
        self.closed = True

    def cleanup_pending(self):
        return not self.closed


def test_explicit_registration_prewarm_and_idempotent_borrow_release():
    registry = ModelPoolRegistry()
    pools = []

    def create():
        pool = _Pool()
        pools.append(pool)
        return pool

    registry.register(_identity(), create)
    assert not pools
    registry.prewarm(_identity())
    first = registry.acquire(_identity())
    second = registry.acquire(_identity())
    assert first.pool is second.pool is pools[0]
    first.shutdown(kill=True)
    first.release()
    assert not first.cleanup_pending()
    assert second.cleanup_pending()
    assert not pools[0].calls
    with pytest.raises(RuntimeError, match="released"):
        _ = first.pool
    with pytest.raises(TimeoutError, match="active borrows"):
        registry.close(kill=True)
    assert not pools[0].calls
    with pytest.raises(RuntimeError, match="draining"):
        registry.acquire(_identity())
    second.release()
    registry.close()
    registry.close()
    assert pools[0].calls == [False]


@pytest.mark.parametrize(
    "changes",
    [
        {"session_id": "another-session"},
        {"model": "another-model"},
        {"version": "v2"},
        {"backend": "ray_actor"},
        {"initialization": "weights-b"},
        {"configuration": "cpu-2"},
    ],
)
def test_model_identity_does_not_alias_distinct_owners_or_configuration(changes):
    with ModelPoolRegistry() as registry:
        registry.register(_identity(), _Pool)
        registry.register(_identity(**changes), _Pool)
        with registry.acquire(_identity()) as first, registry.acquire(_identity(**changes)) as second:
            assert first.pool is not second.pool
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_identity(), _Pool)


def test_concurrent_initialization_runs_once_and_keeps_other_models_available():
    registry = ModelPoolRegistry()
    initializing = threading.Event()
    finish_init = threading.Event()
    pools = []

    def create():
        initializing.set()
        assert finish_init.wait(5)
        pool = _Pool()
        pools.append(pool)
        return pool

    registry.register(_identity(), create)
    registry.register(_identity(model="other"), _Pool)
    with ThreadPoolExecutor(max_workers=8) as threads:
        futures = [threads.submit(registry.acquire, _identity()) for _ in range(8)]
        try:
            assert initializing.wait(5)
            with registry.acquire(_identity(model="other")):
                pass
        finally:
            finish_init.set()
        borrows = [future.result(timeout=5) for future in futures]
    assert len(pools) == 1
    assert all(borrow.pool is pools[0] for borrow in borrows)
    for borrow in borrows:
        borrow.release()
    registry.close()
    assert pools[0].closed


def test_drain_racing_initialization_retains_owner_without_publishing_a_borrow():
    registry = ModelPoolRegistry()
    initializing = threading.Event()
    finish_init = threading.Event()
    pool = _Pool()

    def create():
        initializing.set()
        assert finish_init.wait(5)
        return pool

    registry.register(_identity(), create)
    with ThreadPoolExecutor(max_workers=2) as threads:
        future = threads.submit(registry.acquire, _identity())
        try:
            assert initializing.wait(5)
            registry.drain()
            with pytest.raises(TimeoutError):
                registry.close()
        finally:
            finish_init.set()
        with pytest.raises(RuntimeError, match="draining"):
            future.result(timeout=5)
    registry.close()
    assert pool.calls == [False]


def test_close_waits_for_borrow_release_without_holding_the_registry_lock():
    registry = ModelPoolRegistry()
    registry.register(_identity(), _Pool)
    borrow = registry.acquire(_identity())
    pool = borrow.pool
    registry.drain()
    with ThreadPoolExecutor(max_workers=1) as threads:
        close = threads.submit(registry.close, timeout=5)
        borrow.release()
        close.result(timeout=5)
    assert pool.closed


def test_initialization_failure_is_sticky_and_partial_owners_stay_with_runtime():
    registry = ModelPoolRegistry()
    pool = _Pool()
    original = ValueError("model initialization failed")
    calls = []

    def create():
        calls.append(True)
        raise OwnedActorPoolsError("partial construction", owned_actor_pools=[pool], creation_error=original)

    registry.register(_identity(), create)
    for attempt in range(2):
        with pytest.raises(ValueError) as error:
            registry.prewarm(_identity())
        assert (error.value is original) == (attempt == 0)
        assert error.value.args == original.args
        assert not hasattr(error.value, "owned_actor_pools")
    assert calls == [True]
    assert not pool.closed
    registry.close(kill=True)
    assert pool.calls == [True]


@pytest.mark.parametrize("workers", [1, 8], ids=["sequential", "concurrent"])
def test_cached_initialization_failure_does_not_retain_failed_requests(workers):
    calls = []

    class Request:
        pass

    def create():
        calls.append(True)
        try:
            raise RuntimeError("weights unavailable")
        except RuntimeError as cause:
            raise ValueError("model initialization failed") from cause

    with ModelPoolRegistry() as registry:
        registry.register(_identity(), create)

        def request(_):
            owner = Request()
            reference = weakref.ref(owner)
            try:
                registry.prewarm(_identity())
            except ValueError as error:
                assert str(error) == "model initialization failed"
                assert isinstance(error.__cause__, RuntimeError)
                assert str(error.__cause__) == "weights unavailable"
                # Caller annotations must not mutate the cached failure either.
                error.request = owner
            else:
                pytest.fail("initialization should fail")
            return reference

        with ThreadPoolExecutor(max_workers=workers) as threads:
            references = list(threads.map(request, range(100)))
        gc.collect()
        # Requests must be collectable while the failed registration stays open.
        assert not any(reference() is not None for reference in references)
        assert calls == [True]


@pytest.mark.parametrize("failure_kind", ["local_class", "unsupported_state", "broken_reducer"])
def test_cached_initialization_failure_falls_back_without_retaining_custom_errors(failure_kind):
    calls = []
    references = []

    class Request:
        pass

    class LocalError(ValueError):
        def __reduce__(self):
            if failure_kind == "broken_reducer":
                raise TypeError("cannot serialize this exception")
            return super().__reduce__()

    def create():
        calls.append(True)
        owner = Request()
        references.append(weakref.ref(owner))
        error = LocalError("weights unavailable")
        if failure_kind == "unsupported_state":
            error.request = owner
        raise error

    with ModelPoolRegistry() as registry:
        registry.register(_identity(), create)

        def request(first):
            owner = Request()
            references.append(weakref.ref(owner))
            try:
                registry.prewarm(_identity())
            except BaseException as error:
                assert type(error) is (LocalError if first else RuntimeError)
                assert "weights unavailable" in str(error)
                if not first:
                    assert "LocalError" in str(error)
                assert error.__cause__ is None
                assert error.__context__ is None
            else:
                pytest.fail("initialization should fail")

        request(True)
        for _ in range(100):
            request(False)
        gc.collect()
        assert not any(reference() is not None for reference in references)
        assert calls == [True]


def test_failed_close_keeps_only_retry_owners_and_closes_other_models():
    registry = ModelPoolRegistry()
    pools = [_Pool(), _Pool()]
    for index, pool in enumerate(pools):
        identity = _identity(model=str(index))
        registry.register(identity, lambda pool=pool: pool)
        registry.prewarm(identity)
    pools[1].fail = True
    with pytest.raises(OwnedActorPoolsError) as error:
        registry.close()
    assert error.value.owned_actor_pools == [pools[1]]
    assert pools[0].closed
    pools[1].fail = False
    registry.close(kill=True)
    assert pools[0].calls == [False]
    assert pools[1].calls == [False, True]


@pytest.mark.parametrize("status_raises", [False, True])
def test_successful_shutdown_must_prove_ownership_was_released(status_raises):
    registry = ModelPoolRegistry()
    pool = _Pool()
    uncertain = True

    def status():
        if uncertain:
            if status_raises:
                raise RuntimeError("status unavailable")
            return True
        return False

    pool.cleanup_pending = status
    registry.register(_identity(), lambda: pool)
    registry.prewarm(_identity())
    with pytest.raises(OwnedActorPoolsError) as error:
        registry.close()
    assert error.value.owned_actor_pools == [pool]
    uncertain = False
    registry.close()
    assert len(pool.calls) == 2


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan")])
def test_invalid_close_timeout_does_not_drain_runtime(timeout):
    with ModelPoolRegistry() as registry:
        with pytest.raises(ValueError, match="finite and non-negative"):
            registry.close(timeout=timeout)
        registry.register(_identity(), _Pool)
        registry.prewarm(_identity())


@pytest.mark.parametrize("backend", ["subprocess_actor", "ray_actor"])
def test_local_and_ray_pool_adapters_share_runtime_ownership(monkeypatch, backend):
    import sys

    import vane.execution.udf_ray_actor_pool as ray_pool
    import vane.execution.udf_subprocess as local_pool

    closed = []
    if backend == "subprocess_actor":

        class Worker:
            _proc = SimpleNamespace(pid=123)

            def close(self, kill=False):
                closed.append(kill)

            def cleanup_pending(self):
                return not closed

        monkeypatch.setattr(local_pool, "_SingleSubprocessExecutor", lambda *args, **kwargs: Worker())
        pool = local_pool.LocalSubprocessActorPool({"execution_backend": "subprocess_actor"}, 1)
    else:
        actor = object()
        monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(kill=lambda handle, **kwargs: closed.append(handle)))
        pool = ray_pool.UDFActorPoolBase.__new__(ray_pool.UDFActorPoolBase)
        pool._owns_actors = True
        pool.actors = [actor]
        pool._init_refs = []
        pool._payload_ref = None
    resources = ResourceVector(cpu=1, heap_bytes=100)
    registry = ModelPoolRegistry(resident_limit=resources)
    key = _identity(backend=backend)
    registry.register(key, lambda: pool, resources=resources)
    with registry.acquire(key) as first, registry.acquire(key) as second:
        assert first.pool is second.pool is pool
        first.shutdown(kill=True)
        assert not closed
        assert registry.resource_snapshot()["reserved_resources"] == resources.to_dict()
    assert registry.resource_snapshot()["reserved_resources"] == resources.to_dict()
    registry.close(kill=True)
    assert len(closed) == 1
    assert not pool.cleanup_pending()
    assert registry.resource_snapshot()["reserved_resources"] == ResourceVector().to_dict()
