# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vane.execution.udf_actor_pool_lifecycle import OwnedActorPoolsError
from vane.execution.udf_model_pool import ModelPoolIdentity, ModelPoolRegistry


def _identity(**changes):
    return replace(ModelPoolIdentity("session", "encoder", "v1", "subprocess_actor", "weights-a", "cpu-1"), **changes)


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
    for _ in range(2):
        with pytest.raises(ValueError) as error:
            registry.prewarm(_identity())
        assert error.value is original
        assert not hasattr(error.value, "owned_actor_pools")
    assert calls == [True]
    assert not pool.closed
    registry.close(kill=True)
    assert pool.calls == [True]


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
    registry = ModelPoolRegistry()
    key = _identity(backend=backend)
    registry.register(key, lambda: pool)
    with registry.acquire(key) as first, registry.acquire(key) as second:
        assert first.pool is second.pool is pool
        first.shutdown(kill=True)
        assert not closed
    registry.close(kill=True)
    assert len(closed) == 1
    assert not pool.cleanup_pending()
