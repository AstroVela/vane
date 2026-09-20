# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Query-owned input cleanup when optional UDF data accounting is disabled."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vane.execution.ref_bundle import LocalShmBudgetManager


class QueryInputCleanup:
    """Retain running tasks and failed input releases without a byte ledger."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._closed = False
        self._tasks: set[TaskInputCleanup] = set()

    def open_task(self) -> TaskInputCleanup:
        with self._lock:
            if self._closed:
                raise RuntimeError("query input cleanup scope is closed")
            task = TaskInputCleanup(self)
            self._tasks.add(task)
            return task

    def shutdown(self, *, kill: bool = False) -> None:
        with self._lock:
            self._closed = True
            # Executor cancellation owns running work. Its input must remain
            # valid until the worker future (including cleanup) completes.
            pending = [task for task in self._tasks if task._finished]
        error: BaseException | None = None
        for task in pending:
            try:
                task.finish()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def cleanup_pending(self) -> bool:
        with self._lock:
            return bool(self._tasks)


_active_input_cleanup: ContextVar[TaskInputCleanup | None] = ContextVar("vane_udf_input_cleanup", default=None)


def current_input_cleanup() -> TaskInputCleanup | None:
    return _active_input_cleanup.get()


class TaskInputCleanup:
    def __init__(self, query: QueryInputCleanup) -> None:
        self._query = query
        self._transports: dict[tuple[LocalShmBudgetManager, int], None] = {}
        self._finished = False
        self._finishing = False

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _active_input_cleanup.set(self)
        try:
            yield
        finally:
            _active_input_cleanup.reset(token)

    def hold_input_transport(self, manager: LocalShmBudgetManager, lease_id: int) -> None:
        with self._query._lock:
            if self._finished:
                raise RuntimeError("task input cleanup scope is finished")
            self._transports[manager, lease_id] = None

    def finish(self) -> None:
        with self._query._lock:
            if self._finishing:
                return
            self._finishing = True
            self._finished = True
        error: BaseException | None = None
        try:
            # Retry only this task's leases through the transport's shared
            # ownership protocol. Never release raw refs or sweep other queries.
            for manager, lease_id in tuple(self._transports):
                try:
                    manager.cancel_input_lease(lease_id, name="task-input-cleanup")
                    if manager.input_lease_pending(lease_id):
                        raise RuntimeError("task input transport cleanup is still in progress")
                except BaseException as exc:
                    if error is None:
                        error = exc
                else:
                    del self._transports[manager, lease_id]
        finally:
            with self._query._lock:
                self._finishing = False
                if not self._transports:
                    self._query._tasks.discard(self)
        if error is not None:
            raise error
