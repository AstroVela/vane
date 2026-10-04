# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Fence native interruption to one query's lifetime."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from vane.execution.udf_lifecycle import ExecutionCancellationScope


class NativeQueryCancellation:
    """Bind interruption after query startup, and fence callbacks before reuse."""

    def __init__(self, cancellation: ExecutionCancellationScope) -> None:
        self._cancellation = cancellation
        self._lock = threading.Lock()
        self._interrupt_native: Callable[[], None] | None = None
        self._active = True
        self._unregister = cancellation.register_cancel_wakeup(self._interrupt)

    def _interrupt(self) -> None:
        with self._lock:
            if self._active and self._interrupt_native is not None:
                self._interrupt_native()

    def started(self, conn: Any) -> None:
        self.started_callback(conn.interrupt)

    def started_callback(self, interrupt: Callable[[], None]) -> None:
        with self._lock:
            if not self._active:
                return
            self._interrupt_native = interrupt
            # DuckDB resets the interrupt flag during startup. Replay an
            # earlier cancellation only after that reset, on the actual cursor.
            if self._cancellation.is_set():
                interrupt()

    def close(self) -> None:
        with self._lock:
            # Unregister alone cannot fence a callback already copied by cancel.
            self._active = False
            self._interrupt_native = None
        self._unregister()
