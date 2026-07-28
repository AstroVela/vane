# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral task cancellation primitives for UDF executors."""

from __future__ import annotations

import threading
from collections.abc import Callable


class ExecutionCancelledError(RuntimeError):
    """Raised when one logical UDF execution scope is cancelled."""


class ExecutionCancellationScope:
    """One immutable executor/task generation with event-driven cancellation."""

    def __init__(self, owner_id: str, generation: int) -> None:
        parsed_owner_id = str(owner_id).strip()
        if not parsed_owner_id:
            raise ValueError("execution cancellation owner_id must be non-empty")
        parsed_generation = int(generation)
        if parsed_generation <= 0:
            raise ValueError("execution cancellation generation must be positive")
        self.owner_id = parsed_owner_id
        self.generation = parsed_generation
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._cancel_reason = ""
        self._finished = False
        self._next_callback_id = 0
        self._cancel_wakeups: dict[int, Callable[[], None]] = {}

    @property
    def identity(self) -> tuple[str, int]:
        return self.owner_id, self.generation

    @property
    def cancel_reason(self) -> str:
        with self._lock:
            return self._cancel_reason

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    def is_set(self) -> bool:
        """Match ``threading.Event``'s cancellation predicate."""
        return self._event.is_set()

    def raise_if_cancelled(self, operation: str = "UDF execution") -> None:
        if not self.is_set():
            return
        reason = self.cancel_reason or "cancelled"
        raise ExecutionCancelledError(f"{operation} cancelled: {reason}")

    def register_cancel_wakeup(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Wake one blocking authority when this scope is cancelled."""
        if not callable(callback):
            raise TypeError("execution cancellation wakeup must be callable")
        callback_id: int | None = None
        wake_now = False
        with self._lock:
            if self._event.is_set():
                wake_now = True
            elif not self._finished:
                self._next_callback_id += 1
                callback_id = self._next_callback_id
                self._cancel_wakeups[callback_id] = callback
        if wake_now:
            callback()

        def unregister() -> None:
            if callback_id is None:
                return
            with self._lock:
                self._cancel_wakeups.pop(callback_id, None)

        return unregister

    def cancel(self, reason: str = "cancelled") -> bool:
        """Cancel this generation exactly once and wake every registered wait."""
        wakeups: list[Callable[[], None]] = []
        with self._lock:
            if self._finished or self._event.is_set():
                return False
            self._cancel_reason = str(reason).strip() or "cancelled"
            self._event.set()
            wakeups = list(self._cancel_wakeups.values())
            self._cancel_wakeups.clear()
        for wakeup in wakeups:
            try:
                wakeup()
            except Exception:
                # Cancellation must remain observable even if a diagnostic
                # wakeup fails. The blocked owner also checks ``is_set()``.
                pass
        return True

    def finish(self) -> None:
        """Retire a completed generation so later owner close cannot affect it."""
        with self._lock:
            self._finished = True
            self._cancel_wakeups.clear()


__all__ = ["ExecutionCancellationScope", "ExecutionCancelledError"]
