# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Engine-agnostic execution contract shared by every inference backend.

``LLMExecutor`` is the base class implemented by each backend (vLLM,
SGLang, ...). Besides the submit/result lifecycle, it owns the one-shot wakeup
protocol used by DuckDB's native scheduler: a scheduler callback is armed only
while no result or terminal state is ready, then consumed by the next relevant
state change so a blocked pipeline task can be scheduled again.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Callable

import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]


class LLMExecutor(ABC):
    """Common execution contract shared by every inference backend executor."""

    def _ensure_wakeup_state(self) -> None:
        """Lazily initialize callback state for subclasses and test doubles."""
        if not hasattr(self, "_wakeup_lock"):
            self._wakeup_lock = threading.Lock()
        if not hasattr(self, "_wakeup_callbacks"):
            self._wakeup_callbacks: list[Callable[[], None]] = []

    def _wakeup_ready(self) -> bool:
        """Return whether the native scheduler should resume without arming."""
        return False

    def register_wakeup_callback(self, callback: Callable[[], None]) -> bool:
        """Arm a one-shot native wakeup unless work is already actionable.

        True means the callback is stored and the scheduler may safely block;
        False means it must immediately recheck results or terminal state.
        """
        if not callable(callback):
            raise TypeError("llm wakeup callback must be callable")
        self._ensure_wakeup_state()
        with self._wakeup_lock:
            if self._wakeup_ready():
                return False
            self._wakeup_callbacks.append(callback)
            return True

    def _notify_state_change(self, *, force: bool = False) -> None:
        """Wake condition waiters and consume actionable native callbacks.

        Condition waiters are always notified.  Native callbacks are one-shot
        and run only when `_wakeup_ready()` is true, unless `force` requests an
        unconditional scheduler recheck after a state transition.
        """
        self._ensure_wakeup_state()
        callbacks: list[Callable[[], None]] = []
        with self._wakeup_lock:
            if force or self._wakeup_ready():
                callbacks = self._wakeup_callbacks
                self._wakeup_callbacks = []
        result_cv = getattr(self, "_result_cv", None)
        if result_cv is not None:
            with result_cv:
                result_cv.notify_all()
        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass

    @abstractmethod
    def submit(self, _prefix: str | None, prompts: list[str], rows: pa.Table) -> None:
        pass

    @abstractmethod
    def take_ready_result(self) -> tuple[list[str | None], pa.Table] | None:
        pass

    @abstractmethod
    def finished_submitting(self) -> None:
        pass

    @abstractmethod
    def all_tasks_finished(self) -> bool:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass
