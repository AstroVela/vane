# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Monotonic execution deadlines independent of the execution adapter."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from vane.execution.request_admission import _timeout


class RequestExecutionDeadline:
    """One deadline watcher per timed execution, bounded by request admission.

    Callbacks run independently so one request's slow cancellation cannot delay
    another deadline. The adapter arbitrates expiration against completion and
    fences a callback already copied by the watcher before reusing its cursor.
    """

    def __init__(self, started_at: float, timeout: float, on_expire: Callable[[], None]) -> None:
        self.expires_at = started_at + _timeout(timeout, "execution_timeout")
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._callback: Callable[[], None] | None = on_expire
        self._started = False

    def expired(self) -> bool:
        return time.monotonic() >= self.expires_at

    def start(self) -> None:
        with self._lock:
            if self._started or self._stopped.is_set():
                return
            self._started = True
            thread = threading.Thread(target=self._wait, name="vane-request-deadline", daemon=True)
            thread.start()

    def _wait(self) -> None:
        try:
            while not self._stopped.is_set():
                remaining = self.expires_at - time.monotonic()
                if remaining <= 0:
                    with self._lock:
                        callback, self._callback = self._callback, None
                    if callback is not None:
                        callback()
                    return
                self._stopped.wait(min(remaining, threading.TIMEOUT_MAX))
        finally:
            self.close()

    def close(self) -> None:
        # Do not join: the adapter may be finishing on the callback's thread,
        # or holding its own completion lock while a copied callback waits.
        with self._lock:
            self._callback = None
            self._stopped.set()
