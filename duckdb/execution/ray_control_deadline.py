# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import heapq
import math
import threading
import time
from collections.abc import Callable


class RayControlScheduledCall:
    """One cancellable, one-shot callback owned by the deadline clock."""

    def __init__(
        self,
        scheduler: RayControlDeadlineScheduler,
        callback: Callable[[], None],
    ) -> None:
        self._scheduler = scheduler
        self._callback: Callable[[], None] | None = callback
        self._scheduled = False
        self._cancelled = False

    def cancel(self) -> None:
        self._scheduler.cancel(self)

    def cancelled(self) -> bool:
        with self._scheduler._condition:
            return self._cancelled


class RayControlDeadlineScheduler:
    """One process clock that runs only bounded, non-blocking bookkeeping."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._queue: list[tuple[float, int, RayControlScheduledCall]] = []
        self._next_sequence = 0
        self._thread: threading.Thread | None = None

    def create_inline(self, callback: Callable[[], None]) -> RayControlScheduledCall:
        """Create an internal callback that is safe to run on the clock.

        Potentially blocking owner work must use
        ``create_ray_control_deadline`` from ``ray_control_submission`` so the
        clock only hands that work to the bounded callback executor.
        """
        if not callable(callback):
            raise TypeError("Ray control deadline callback must be callable")
        return RayControlScheduledCall(self, callback)

    def ensure_started(self) -> None:
        """Start before a caller transfers work that requires a deadline."""
        with self._condition:
            if self._thread is not None:
                return
            thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="vane-ray-control-deadlines",
            )
            # Publish only a successfully started thread so a later caller can
            # retry a transient start failure.
            thread.start()
            self._thread = thread

    def schedule(self, call: RayControlScheduledCall, delay_s: float) -> None:
        delay = float(delay_s)
        if not math.isfinite(delay):
            raise ValueError("Ray control deadline delay must be finite")
        self.schedule_at(call, time.monotonic() + max(0.0, delay))

    def schedule_at(self, call: RayControlScheduledCall, deadline: float) -> None:
        target = float(deadline)
        if not math.isfinite(target):
            raise ValueError("Ray control deadline must be finite")
        self.ensure_started()
        with self._condition:
            if call._scheduler is not self:
                raise RuntimeError("scheduled call belongs to a different deadline scheduler")
            if call._callback is None or call._cancelled:
                return
            if call._scheduled:
                raise RuntimeError("scheduled call is already armed")
            call._scheduled = True
            self._next_sequence += 1
            heapq.heappush(
                self._queue,
                (target, self._next_sequence, call),
            )
            self._condition.notify()

    def cancel(self, call: RayControlScheduledCall) -> None:
        with self._condition:
            if call._scheduler is not self:
                raise RuntimeError("scheduled call belongs to a different deadline scheduler")
            # Clearing the callback also releases everything captured by a
            # cancelled heap entry before that entry reaches the queue head.
            call._cancelled = True
            call._callback = None
            call._scheduled = False
            self._condition.notify()

    def _run(self) -> None:
        while True:
            callback: Callable[[], None] | None = None
            with self._condition:
                while callback is None:
                    while self._queue and self._queue[0][2]._callback is None:
                        heapq.heappop(self._queue)
                    if not self._queue:
                        self._condition.wait()
                        continue
                    deadline, _sequence, call = self._queue[0]
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        self._condition.wait(timeout=remaining)
                        continue
                    heapq.heappop(self._queue)
                    callback = call._callback
                    call._callback = None
                    call._scheduled = False
            if callback is not None:
                try:
                    callback()
                except BaseException:
                    # Deadline owners provide their own terminal/retry policy.
                    # One malformed callback must not terminate the clock.
                    pass


RAY_CONTROL_DEADLINE_SCHEDULER = RayControlDeadlineScheduler()


__all__ = [
    "RAY_CONTROL_DEADLINE_SCHEDULER",
    "RayControlDeadlineScheduler",
    "RayControlScheduledCall",
]
