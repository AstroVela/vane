# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral, bounded accounting for observed worker terminal outcomes.

Adapters report structured events where they observe them. A lifecycle belongs
to one physical worker generation, not a request or a pool slot. The collector
retains only fixed counters; lifecycles never retain exceptions or callbacks.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from enum import Enum


class WorkerOutcome(str, Enum):
    INITIALIZATION_FAILURE = "initialization_failures"
    EXECUTION_ERROR = "execution_errors"
    WORKER_LOSS = "worker_losses"
    RUNTIME_ERROR = "runtime_errors"
    CANCELLED = "cancelled_workers"
    SHUTDOWN = "shutdown_workers"


class WorkerMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts = {outcome.value: 0 for outcome in WorkerOutcome}

    def _record(self, outcome: WorkerOutcome) -> None:
        with self._lock:
            self._counts[outcome.value] += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


class WorkerLifecycle:
    """Record the first terminal observation, including intentional retirement.

    Cached task workers bind to the current borrower and unbind before becoming
    idle. Model workers remain bound to their owning runtime. Binding a different
    collector never republishes an already observed outcome.
    """

    def __init__(self, metrics: WorkerMetrics | None = None) -> None:
        self._lock = threading.Lock()
        self._metrics = metrics
        self._ready = False
        self._outcome: WorkerOutcome | None = None

    def bind(self, metrics: WorkerMetrics | None) -> None:
        with self._lock:
            self._metrics = metrics

    def ready(self) -> None:
        with self._lock:
            self._ready = True

    def finish(self, outcome: WorkerOutcome) -> None:
        with self._lock:
            self._finish_locked(outcome, self._metrics)

    def capture_observer(self) -> Callable[[WorkerOutcome], None]:
        """Keep a deferred result attributed to its producing borrower.

        The observer retains only this lifecycle and the fixed-size collector,
        never a worker, request or runtime owner. It shares the same first-event
        rule as immediate observations, even after task-pool rebinding.
        """
        with self._lock:
            metrics = self._metrics

        def observe(outcome: WorkerOutcome) -> None:
            with self._lock:
                self._finish_locked(outcome, metrics)

        return observe

    def _finish_locked(self, outcome: WorkerOutcome, metrics: WorkerMetrics | None) -> None:
        if not isinstance(outcome, WorkerOutcome):
            raise TypeError("worker outcome must be WorkerOutcome")
        if self._outcome is not None:
            return
        if not self._ready and outcome not in {WorkerOutcome.CANCELLED, WorkerOutcome.SHUTDOWN}:
            outcome = WorkerOutcome.INITIALIZATION_FAILURE
        self._outcome = outcome
        # No callbacks or adapter locks are acquired by this collector.
        if metrics is not None:
            metrics._record(outcome)
