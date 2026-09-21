# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Request-owned executor cleanup, independent of task and byte accounting."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vane.execution.udf_subprocess import UDFExecutor


class QueryExecutorCleanup:
    """Keep executors reachable until their completion callbacks and close finish."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._closed = False
        self._executors: set[UDFExecutor] = set()

    def hold(self, executor: UDFExecutor) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("query executor cleanup scope is closed")
            self._executors.add(executor)

    def shutdown(self, *, kill: bool = False) -> None:
        with self._lock:
            self._closed = True
            pending = tuple(self._executors)
        error: BaseException | None = None
        for executor in pending:
            try:
                executor.close(kill=kill)
                if executor.cleanup_pending():
                    raise RuntimeError("query executor cleanup is still in progress")
            except BaseException as exc:
                if error is None:
                    error = exc
            else:
                with self._lock:
                    self._executors.discard(executor)
        if error is not None:
            raise error

    def cleanup_pending(self) -> bool:
        with self._lock:
            return bool(self._executors)
