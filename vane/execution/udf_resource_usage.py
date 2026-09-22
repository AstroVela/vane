# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Passive UDF activity diagnostics, separate from admission and ownership."""

from __future__ import annotations

import threading
import weakref
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import Any

_TASK_STATES = ("preparing", "submitted", "running", "completing")
_WAIT_REASONS = ("shared_memory_input", "shared_memory_output", "execution_capacity")
_active_task: ContextVar[UnitTaskActivity | None] = ContextVar("vane_udf_unit_activity", default=None)


class UnitResourceActivity:
    """Keep scalar activity only; never keep executors, futures or input buffers."""

    def __init__(self, identity: Mapping[str, str]) -> None:
        self._identity = dict(identity)
        self._lock = threading.Lock()
        self._authorities: weakref.WeakSet[Any] = weakref.WeakSet()
        self._unobservable_admission = False
        self._tasks: dict[object, str] = {}
        self._refusals = {"runtime_bytes": 0, "transport_bytes": 0}

    @property
    def identity(self) -> dict[str, str]:
        return dict(self._identity)

    def bind_admission(self, authority: Any) -> None:
        with self._lock:
            # Custom actor pools may implement only the existing admission
            # contract, or expose non-weakrefable authorities. Diagnostics
            # must neither change their ownership nor call an active state().
            if not callable(getattr(authority, "diagnostic_state", None)):
                self._unobservable_admission = True
            else:
                try:
                    self._authorities.add(authority)
                except TypeError:
                    self._unobservable_admission = True

    def refuse_bytes(self, owner: str) -> None:
        with self._lock:
            self._refusals[f"{owner}_bytes"] += 1

    def open_task(self) -> UnitTaskActivity:
        token = object()
        with self._lock:
            self._tasks[token] = "preparing"
        return UnitTaskActivity(self, token)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            authorities = tuple(self._authorities)
            states = tuple(self._tasks.values())
            refusals = dict(self._refusals)
            unobservable = self._unobservable_admission
        # These reads must not acquire admission, notify callbacks, or retry
        # byte reservations. In particular, never call authority.state() here.
        admissions = [authority.diagnostic_state() for authority in authorities]
        observable = not unobservable and "unavailable" not in admissions
        queued = sum(state == "requested" for state in admissions) if observable else None
        return {
            **self.identity,
            **{f"{state}_tasks": states.count(state) for state in _TASK_STATES},
            "queued_tasks": queued,
            "ready_tasks": admissions.count("ready") if observable else None,
            "waiting_tasks": sum(states.count(reason) for reason in _WAIT_REASONS),
            "waiting_by_reason": {
                "task_capacity": queued,
                **{reason: states.count(reason) for reason in _WAIT_REASONS},
            },
            "byte_refusals": refusals,
        }


class UnitTaskActivity:
    def __init__(self, unit: UnitResourceActivity, token: object) -> None:
        self._unit = unit
        self._token = token

    def transition(self, state: str) -> None:
        with self._unit._lock:
            if self._token in self._unit._tasks:
                self._unit._tasks[self._token] = state

    def finish(self) -> None:
        with self._unit._lock:
            self._unit._tasks.pop(self._token, None)

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _active_task.set(self)
        try:
            yield
        finally:
            _active_task.reset(token)


@contextmanager
def observe_transport_wait(base: AbstractContextManager[None], reason: str) -> Iterator[None]:
    task = _active_task.get()
    if task is None:
        with base:
            yield
        return
    with task._unit._lock:
        previous = task._unit._tasks.get(task._token, "running")
    try:
        task.transition(reason)
        with base:
            try:
                yield
            finally:
                # The transport is ready, but exiting the underlying context
                # can still wait to reacquire the runtime execution allowance.
                task.transition("execution_capacity")
    finally:
        task.transition(previous)


def unit_usage_snapshot(
    activities: Mapping[str, UnitResourceActivity],
    data: Mapping[str, dict[str, Any]] | None,
    *,
    prepared_query_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Combine independently locked diagnostic snapshots without owning resources."""
    units = {}
    for key, activity in activities.items():
        snapshot = activity.snapshot()
        active = any(value for name, value in snapshot.items() if name.endswith("_tasks"))
        if prepared_query_ids is None or snapshot["query_id"] in prepared_query_ids or active or key in (data or {}):
            units[key] = snapshot
    for key, values in (data or {}).items():
        if key not in units:
            units[key] = UnitResourceActivity(values["identity"]).snapshot()
            # Data views can outlive their diagnostic scope. Do not invent a
            # zero historical refusal count once that scope has been collected.
            units[key]["byte_refusals"] = None
    for key, snapshot in units.items():
        snapshot["data"] = None if data is None else (data.get(key) or {}).get("usage", empty_unit_data())
    return [units[key] for key in sorted(units)]


def empty_unit_data() -> dict[str, int]:
    return {
        "retained_bytes": 0,
        "input_bytes": 0,
        "output_bytes": 0,
        "shared_retained_bytes": 0,
        "allocations": 0,
        "leases": 0,
        "input_reserved_bytes": 0,
        "output_reserved_bytes": 0,
        "reserved_bytes": 0,
        "reservations": 0,
        "cleanup_pending_tasks": 0,
    }
