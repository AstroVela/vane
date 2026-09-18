# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Ownership contracts shared by local and Ray UDF actor preparation."""

from __future__ import annotations

import sys
import types

import pytest

from vane.execution.udf_actor_pool_lifecycle import (
    OwnedActorPoolsError,
    actor_pool_cleanup_pending,
    rollback_actor_pools,
)


def test_rollback_recovers_constructor_ownership_once_in_reverse_order():
    events = []
    errors = []

    class Pool:
        def __init__(self, name):
            self.name = name

        def __eq__(self, other):
            raise AssertionError("pool ownership must use identity, not user equality")

        def shutdown(self):
            events.append(self.name)
            raise RuntimeError(self.name)

    first, second = Pool("first"), Pool("second")
    cause = ValueError("constructor failed")
    error = OwnedActorPoolsError(
        "constructor cleanup failed",
        owned_actor_pools=[first, second, second],
        creation_error=cause,
    )

    pending = rollback_actor_pools(
        [first],
        error,
        shutdown=lambda pool: pool.shutdown(),
        cleanup_pending=actor_pool_cleanup_pending,
        record_error=errors.append,
    )

    assert events == ["second", "first"]
    assert len(pending) == 2
    assert pending[0] is first and pending[1] is second
    assert [str(exc) for exc in errors] == ["second", "first"]
    assert error.creation_error is cause


def test_rollback_status_failure_retains_pool_and_continues_after_interrupt():
    events = []
    errors = []

    class Pool:
        def __init__(self, name):
            self.name = name

        def shutdown(self):
            events.append(self.name)
            if self.name == "second":
                raise KeyboardInterrupt("shutdown interrupted")

        @property
        def cleanup_pending(self):
            raise RuntimeError("status unavailable")

    first, second = Pool("first"), Pool("second")
    pending = rollback_actor_pools(
        [first, second],
        RuntimeError("prepare failed"),
        shutdown=lambda pool: pool.shutdown(),
        cleanup_pending=actor_pool_cleanup_pending,
        record_error=errors.append,
    )

    assert events == ["second", "first"]
    assert pending == [second]
    assert isinstance(errors[0], KeyboardInterrupt)
    assert str(errors[1]) == "status unavailable"


def test_rollback_does_not_retain_terminated_pool_after_cleanup_diagnostic():
    errors = []

    class Pool:
        def shutdown(self):
            raise RuntimeError("callable cleanup failed, process already terminated")

        def cleanup_pending(self):
            return False

    assert (
        rollback_actor_pools(
            [Pool()],
            RuntimeError("prepare failed"),
            shutdown=lambda pool: pool.shutdown(),
            cleanup_pending=actor_pool_cleanup_pending,
            record_error=errors.append,
        )
        == []
    )
    assert len(errors) == 1


def _prepare_actor_pools(monkeypatch, backend, pool_factory, *, set_handles=None):
    nodes = [
        {
            "node_id": str(index),
            "actor_pool_size": 1,
            "payload": {
                "execution_backend": "subprocess_actor" if backend == "local" else "ray_actor",
                "actor_number": 1,
                "actor_pool_size": 1,
                "query_id": "query",
                "resource_unit_id": f"unit-{index}",
            },
        }
        for index in range(2)
    ]
    if backend == "local":
        from vane.execution import udf_subprocess

        monkeypatch.setattr(udf_subprocess, "LocalSubprocessActorPool", pool_factory)
        return udf_subprocess.ensure_local_subprocess_actor_pools_for_nodes(nodes, set_handles=set_handles)

    from vane.execution import udf_ray_actor_pool

    monkeypatch.setitem(sys.modules, "ray", types.SimpleNamespace(is_initialized=lambda: True))
    return udf_ray_actor_pool.prepare_actor_pools_for_nodes(
        nodes,
        query_driver_handle=object(),
        query_generation_capability="generation",
        session_config={},
        set_handles=set_handles,
        actor_pool_cls=pool_factory,
        is_vane_worker_process=lambda: False,
        requires_actor_pool_fn=lambda payload: True,
        normalize_actor_pool_payload=dict,
        payload_num_gpus=lambda payload: 0.0,
        required_positive_int=lambda payload, key: payload[key],
        resolve_actor_num_cpus=lambda payload: 1.0,
        build_udf_executor_options=lambda **options: options,
    )


@pytest.mark.parametrize("backend", ["local", "ray"])
@pytest.mark.parametrize("shutdown_fails", [False, True])
def test_preparation_rollback_preserves_failure_and_only_unclosed_pools(monkeypatch, backend, shutdown_fails):
    events = []
    created = []
    preparation_error = RuntimeError("handle injection failed")

    class Pool:
        def __init__(self, *args, **kwargs):
            self.index = len(created)
            self.actors = [object()]
            self.actor_node_ids = ["node"]
            self._confirmed_ready = {0}
            self._init_refs = []
            created.append(self)

        def shutdown(self, *, kill=False):
            events.append((self.index, kill))
            if shutdown_fails and self.index == 1:
                raise RuntimeError("worker still alive")
            self.actors.clear()

        def cleanup_pending(self):
            return bool(self.actors)

    def fail_injection(handles):
        assert set(handles) == {"0", "1"}
        raise preparation_error

    with pytest.raises(RuntimeError) as exc_info:
        _prepare_actor_pools(monkeypatch, backend, Pool, set_handles=fail_injection)

    assert events == [(1, backend == "local"), (0, backend == "local")]
    if shutdown_fails:
        assert isinstance(exc_info.value, OwnedActorPoolsError)
        assert exc_info.value.creation_error is preparation_error
        assert exc_info.value.owned_actor_pools == [created[1]]
    else:
        assert exc_info.value is preparation_error


@pytest.mark.parametrize("backend", ["local", "ray"])
def test_preparation_recovers_partial_constructor_pool_and_original_error(monkeypatch, backend):
    events = []
    constructor_error = ValueError("model loading failed")

    class PartialPool:
        def shutdown(self, *, kill=False):
            events.append(kill)

    pending_pool = PartialPool()

    def fail_constructor(*args, **kwargs):
        raise OwnedActorPoolsError(
            "partial constructor cleanup failed",
            owned_actor_pools=[pending_pool],
            creation_error=constructor_error,
        )

    with pytest.raises(ValueError) as exc_info:
        _prepare_actor_pools(monkeypatch, backend, fail_constructor)

    assert exc_info.value is constructor_error
    assert events == [backend == "local"]


def test_ray_readiness_failure_retains_uncertain_pool_and_closes_other_pools(monkeypatch):
    from vane.execution import udf_ray_actor_pool

    events = []
    readiness_error = RuntimeError("warmup failed")

    class Pool:
        def __init__(self, name):
            self.name = name

        def shutdown(self):
            events.append(self.name)
            if self.name == "second":
                raise RuntimeError("shutdown failed")

        @property
        def actors(self):
            raise RuntimeError("actor status unavailable")

    first, second = Pool("first"), Pool("second")

    def fail_readiness(ray, pool):
        raise readiness_error

    monkeypatch.setattr(udf_ray_actor_pool, "_resolve_actor_pool_init_refs", fail_readiness)
    with pytest.raises(OwnedActorPoolsError) as exc_info:
        udf_ray_actor_pool.wait_for_actor_pools_ready([first, second])

    assert events == ["second", "first"]
    assert exc_info.value.creation_error is readiness_error
    assert exc_info.value.owned_actor_pools == [second]
    assert "actor status unavailable" in str(exc_info.value)


def test_local_preparation_failure_leaves_borrowed_model_pool_with_its_owner(monkeypatch):
    from vane.execution import udf_subprocess

    def unexpected(*args, **kwargs):
        raise AssertionError("preparation must not execute or shut down a borrowed pool")

    borrowed = types.SimpleNamespace(
        pool_size=1,
        session_config=None,
        submit=unexpected,
        create_admission_authority=unexpected,
        stats=unexpected,
        cancel_output_grants=unexpected,
        abort_scopes=unexpected,
        first_proc=unexpected,
        worker_pids=unexpected,
        shutdown=unexpected,
    )
    shutdowns = []
    owned = types.SimpleNamespace(shutdown=lambda *, kill: shutdowns.append(kill))
    monkeypatch.setattr(udf_subprocess, "LocalSubprocessActorPool", lambda *args, **kwargs: owned)
    failure = RuntimeError("handle injection failed")

    def fail_injection(handles):
        assert handles["borrowed"]["local_actor_pool"] is borrowed
        assert handles["owned"]["local_actor_pool"] is owned
        raise failure

    payload = {"execution_backend": "subprocess_actor", "actor_number": 1}
    nodes = [
        {"node_id": "borrowed", "payload": payload, "executor_options": {"local_actor_pool": borrowed}},
        {"node_id": "owned", "payload": payload},
    ]
    with pytest.raises(RuntimeError) as exc_info:
        udf_subprocess.ensure_local_subprocess_actor_pools_for_nodes(nodes, set_handles=fail_injection)

    assert exc_info.value is failure
    assert shutdowns == [True]
