# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from duckdb.runners.ray.fragment_worker_lifecycle import FteWorkerLifecycleMixin


class _RemoteMethod:
    def __init__(self, events: list[str], result: object) -> None:
        self._events = events
        self._result = result

    def remote(self) -> object:
        self._events.append("shutdown-rpc")
        return self._result


class _Actor:
    def __init__(self, events: list[str], result: object) -> None:
        self.shutdown = _RemoteMethod(events, result)


def _lifecycle(actor: object) -> FteWorkerLifecycleMixin:
    lifecycle = FteWorkerLifecycleMixin()
    lifecycle.worker_id = ""
    lifecycle.actor_handle = actor
    return lifecycle


def test_worker_shutdown_waits_for_graceful_rpc_before_kill(monkeypatch):
    from duckdb.runners.ray import fragment_worker_lifecycle as lifecycle_module

    events: list[str] = []
    result = object()
    lifecycle = _lifecycle(_Actor(events, result))

    def resolve(ref, **kwargs):
        assert ref is result
        assert kwargs == {"timeout": 30, "honor_query_deadline": False}
        events.append("shutdown-resolved")

    monkeypatch.setattr(lifecycle_module, "resolve_object_refs_blocking", resolve)
    monkeypatch.setattr(lifecycle_module.ray, "kill", lambda actor: events.append("kill"))

    lifecycle.shutdown()

    assert events == ["shutdown-rpc", "shutdown-resolved", "kill"]


def test_worker_shutdown_kills_actor_when_graceful_rpc_fails(monkeypatch):
    from duckdb.runners.ray import fragment_worker_lifecycle as lifecycle_module

    events: list[str] = []
    result = object()
    lifecycle = _lifecycle(_Actor(events, result))

    def fail_resolve(ref, **kwargs):
        assert ref is result
        events.append("shutdown-failed")
        raise RuntimeError("stop failed")

    monkeypatch.setattr(lifecycle_module, "resolve_object_refs_blocking", fail_resolve)
    monkeypatch.setattr(lifecycle_module.ray, "kill", lambda actor: events.append("kill"))

    with pytest.raises(RuntimeError, match="graceful shutdown failed"):
        lifecycle.shutdown()

    assert events == ["shutdown-rpc", "shutdown-failed", "kill"]
