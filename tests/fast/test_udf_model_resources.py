# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from vane import pickle as vane_pickle
from vane.execution.resources import ResourceVector, udf_process_resources
from vane.execution.udf_actor_pool_lifecycle import OwnedActorPoolsError
from vane.execution.udf_local_model import LocalModelRuntime
from vane.execution.udf_model_pool import ModelPoolCapacityError, ModelPoolIdentity, ModelPoolRegistry


def _identity(name="model"):
    return ModelPoolIdentity("session", name, "v1", "subprocess_actor", "weights", "config")


class _Pool:
    def __init__(self):
        self.closed = False
        self.fail_close = False
        self.calls = 0

    def shutdown(self, *, kill=False):
        self.calls += 1
        if self.fail_close:
            raise RuntimeError("cleanup failed")
        self.closed = True

    def cleanup_pending(self):
        return not self.closed


def _resources(snapshot, field="reserved_resources"):
    return ResourceVector.from_dict(snapshot[field])


def test_shared_resources_preserve_ray_type_identity_and_pickle_compatibility():
    from vane.runners.ray.query_resource_graph import ResourceVector as RayResourceVector

    assert RayResourceVector is ResourceVector
    # A protocol-zero reference written before the class moved still resolves.
    assert vane_pickle.loads(b"cvane.runners.ray.query_resource_graph\nResourceVector\n.") is ResourceVector
    value = ResourceVector(cpu=0.25, heap_bytes=123)
    assert vane_pickle.loads(vane_pickle.dumps(value)) == value


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"cpus": -1}, "cpus"),
        ({"cpus": float("nan")}, "cpus"),
        ({"gpus": float("inf")}, "gpus"),
        ({"gpus": -1}, "gpus"),
        ({"cpus": 0, "gpus": 0}, "CPU or GPU"),
        ({"memory_bytes": 0}, "memory_bytes"),
        ({"memory_bytes": -1}, "memory_bytes"),
        ({"memory_bytes": float("inf")}, "memory_bytes"),
    ],
)
def test_invalid_process_declarations_are_rejected(payload, message):
    with pytest.raises(ValueError, match=message):
        udf_process_resources(payload)


def test_missing_heap_is_unreserved_and_declared_bytes_keep_integer_precision():
    assert udf_process_resources({}) == ResourceVector(cpu=1)
    assert udf_process_resources({"cpus": 0.25, "memory_bytes": 2**53 + 1}) == ResourceVector(
        cpu=0.25, heap_bytes=2**53 + 1
    )


def test_lazy_reservation_counts_a_pool_once_across_queries_and_releases_only_on_close():
    request = ResourceVector(cpu=0.5, heap_bytes=100)
    pool = _Pool()
    registry = ModelPoolRegistry(resident_limit=request)
    registry.register(_identity(), lambda: pool, resources=request)
    assert _resources(registry.resource_snapshot()).is_zero()
    assert _resources(registry.resource_snapshot(), "registered_resources") == request
    first = registry.acquire(_identity())
    second = registry.acquire(_identity())
    assert first.pool is second.pool
    assert registry.resource_snapshot()["active_borrows"] == 2
    assert _resources(registry.resource_snapshot()) == request
    first.release()
    first.release()
    second.release()
    assert _resources(registry.resource_snapshot(), "resident_resources") == request
    assert registry.resource_snapshot()["reserved_models"] == 1
    registry.close()
    registry.close()
    assert _resources(registry.resource_snapshot()).is_zero()
    assert pool.calls == 1


@pytest.mark.parametrize("demand", [ResourceVector(cpu=2), ResourceVector(heap_bytes=101)])
def test_oversized_registration_is_rejected_without_publishing_an_entry(demand):
    limit = ResourceVector(cpu=1, heap_bytes=100)
    with ModelPoolRegistry(resident_limit=limit) as registry:
        with pytest.raises(ModelPoolCapacityError, match="request exceeds resident limit") as error:
            registry.register(_identity(), _Pool, resources=demand)
        assert error.value.oversized
        assert error.value.requested == demand
        assert error.value.reserved.is_zero()
        assert registry.resource_snapshot()["registered_models"] == 0
        # A refused registration has not consumed its identity.
        registry.register(_identity(), _Pool, resources=limit)
        registry.prewarm(_identity())


def test_concurrent_models_cannot_each_acquire_the_full_runtime_capacity():
    count = 8
    barrier = threading.Barrier(count)
    limit = ResourceVector(cpu=1, heap_bytes=100)
    constructed = []

    def create(name):
        constructed.append(name)
        return _Pool()

    with ModelPoolRegistry(resident_limit=limit) as registry:
        for index in range(count):
            registry.register(_identity(str(index)), lambda i=index: create(i), resources=limit)

        def acquire(index):
            barrier.wait(timeout=5)
            try:
                registry.prewarm(_identity(str(index)))
                return True
            except ModelPoolCapacityError as error:
                assert not error.oversized
                assert error.reserved == limit
                return False

        with ThreadPoolExecutor(max_workers=count) as threads:
            results = list(threads.map(acquire, range(count)))
        assert sum(results) == 1
        assert len(constructed) == 1
        assert _resources(registry.resource_snapshot()) == limit


def test_initializing_reservation_blocks_other_models_and_clean_failure_allows_retry():
    entered = threading.Event()
    finish = threading.Event()
    request = ResourceVector(cpu=1, heap_bytes=10)
    attempts = []

    def fail():
        entered.set()
        assert finish.wait(5)
        raise ValueError("weights unavailable")

    def create():
        attempts.append(True)
        return _Pool()

    with ModelPoolRegistry(resident_limit=request) as registry:
        registry.register(_identity("failing"), fail, resources=request)
        registry.register(_identity("other"), create, resources=request)
        with ThreadPoolExecutor(max_workers=1) as threads:
            pending = threads.submit(registry.prewarm, _identity("failing"))
            try:
                assert entered.wait(5)
                assert _resources(registry.resource_snapshot(), "initializing_resources") == request
                with pytest.raises(ModelPoolCapacityError, match="capacity is in use"):
                    registry.prewarm(_identity("other"))
                assert not attempts
            finally:
                finish.set()
            with pytest.raises(ValueError, match="weights unavailable"):
                pending.result(timeout=5)
        assert _resources(registry.resource_snapshot()).is_zero()
        registry.prewarm(_identity("other"))
        # Cached initialization failures neither reacquire nor release capacity.
        for _ in range(2):
            with pytest.raises(ValueError, match="weights unavailable"):
                registry.prewarm(_identity("failing"))
        assert attempts == [True]
        assert _resources(registry.resource_snapshot()) == request


def test_partial_initialization_keeps_full_reservation_until_every_owner_is_cleaned():
    request = ResourceVector(cpu=2, heap_bytes=200)
    pools = [_Pool(), _Pool()]
    pools[1].fail_close = True

    def create():
        raise OwnedActorPoolsError("partial init", owned_actor_pools=pools, creation_error=ValueError("init failed"))

    registry = ModelPoolRegistry(resident_limit=request)
    registry.register(_identity(), create, resources=request)
    registry.register(_identity("other"), _Pool, resources=request)
    with pytest.raises(ValueError, match="init failed"):
        registry.prewarm(_identity())
    assert _resources(registry.resource_snapshot(), "retained_failure_resources") == request
    with pytest.raises(ModelPoolCapacityError):
        registry.prewarm(_identity("other"))
    with pytest.raises(OwnedActorPoolsError):
        registry.close()
    assert pools[0].closed
    assert _resources(registry.resource_snapshot()) == request
    pools[1].fail_close = False
    registry.close()
    assert _resources(registry.resource_snapshot()).is_zero()
    assert [pool.calls for pool in pools] == [1, 2]


@pytest.mark.parametrize("uncertain_status", [False, True])
def test_close_releases_only_confirmed_model_reservations(uncertain_status):
    request = ResourceVector(cpu=1, heap_bytes=10)
    registry = ModelPoolRegistry(resident_limit=request.scale(2))
    pools = [_Pool(), _Pool()]
    pending = True

    def status():
        if pending:
            if uncertain_status:
                raise RuntimeError("cleanup status unavailable")
            return True
        return False

    pools[1].cleanup_pending = status
    for index, pool in enumerate(pools):
        registry.register(_identity(str(index)), lambda pool=pool: pool, resources=request)
        registry.prewarm(_identity(str(index)))
    with pytest.raises(OwnedActorPoolsError):
        registry.close()
    assert _resources(registry.resource_snapshot()) == request
    pending = False
    registry.close()
    registry.close()
    assert _resources(registry.resource_snapshot()).is_zero()
    assert [pool.calls for pool in pools] == [1, 2]


def test_drain_racing_constructor_keeps_reservation_through_close_timeout():
    started = threading.Event()
    finish = threading.Event()
    request = ResourceVector(cpu=1)
    registry = ModelPoolRegistry(resident_limit=request)

    def create():
        started.set()
        assert finish.wait(5)
        return _Pool()

    registry.register(_identity(), create, resources=request)
    with ThreadPoolExecutor(max_workers=1) as threads:
        pending = threads.submit(registry.prewarm, _identity())
        try:
            assert started.wait(5)
            with pytest.raises(TimeoutError):
                registry.close()
            assert _resources(registry.resource_snapshot()) == request
        finally:
            finish.set()
        with pytest.raises(RuntimeError, match="draining"):
            pending.result(timeout=5)
    assert _resources(registry.resource_snapshot(), "resident_resources") == request
    registry.close()
    assert _resources(registry.resource_snapshot()).is_zero()


@pytest.mark.parametrize("limit", [ResourceVector(gpu=1), ResourceVector(object_store_bytes=1)])
def test_local_limits_reject_dimensions_with_different_ownership(limit):
    with pytest.raises(ValueError, match="CPU and declared heap only"):
        LocalModelRuntime(session_id="session", session_config={}, resident_limit=limit)


def test_unbounded_registry_still_reports_declared_reservations():
    with ModelPoolRegistry() as registry:
        request = ResourceVector(cpu=2, heap_bytes=100)
        registry.register(_identity(), _Pool, resources=request)
        registry.prewarm(_identity())
        assert registry.resource_snapshot()["limit"] is None
        assert _resources(registry.resource_snapshot()) == request


def test_fractional_reservations_follow_remaining_owners_without_subtraction_drift():
    registry = ModelPoolRegistry()
    small = _Pool()
    small.fail_close = True
    for name, cpu, pool in [("large", 1_000_000, _Pool()), ("small", 0.1, small)]:
        registry.register(_identity(name), lambda pool=pool: pool, resources=ResourceVector(cpu=cpu))
        registry.prewarm(_identity(name))
    with pytest.raises(OwnedActorPoolsError):
        registry.close()
    assert _resources(registry.resource_snapshot()) == ResourceVector(cpu=0.1)
    small.fail_close = False
    registry.close()
    assert _resources(registry.resource_snapshot()).is_zero()
