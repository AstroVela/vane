# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, TypeVar

_RAY_CONTROL_SUBMISSION_WORKERS = 4

_T = TypeVar("_T")


class _RayControlSubmissionExecutor:
    """Fixed daemon workers for Ray calls whose submission may block."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue: queue.SimpleQueue[tuple[Future[Any], Callable[[], Any]]] = queue.SimpleQueue()
        self._threads: list[threading.Thread] = []

    def submit(self, callback: Callable[[], _T]) -> Future[_T]:
        with self._lock:
            while len(self._threads) < _RAY_CONTROL_SUBMISSION_WORKERS:
                index = len(self._threads)
                thread = threading.Thread(
                    target=self._run,
                    daemon=True,
                    name=f"vane-ray-control-submit-{index}",
                )
                thread.start()
                self._threads.append(thread)
        future: Future[_T] = Future()
        self._queue.put((future, callback))
        return future

    def _run(self) -> None:
        while True:
            future, callback = self._queue.get()
            result: Any = None
            try:
                try:
                    if not future.set_running_or_notify_cancel():
                        continue
                    result = callback()
                except BaseException as exc:
                    if not future.done():
                        try:
                            future.set_exception(exc)
                        except BaseException:
                            pass
                else:
                    try:
                        future.set_result(result)
                    except BaseException:
                        # The Future becomes terminal before callbacks run. A
                        # malformed callback must not reduce the fixed worker set.
                        pass
            finally:
                # A worker blocks indefinitely in the next queue.get(). Do not
                # let its frame retain the previous stream owner or ObjectRef.
                result = None
                del callback
                del future


_RAY_CONTROL_SUBMISSION_EXECUTOR = _RayControlSubmissionExecutor()


def submit_ray_control(callback: Callable[[], _T]) -> Future[_T]:
    """Submit one Ray control call without occupying its owner thread."""
    return _RAY_CONTROL_SUBMISSION_EXECUTOR.submit(callback)


__all__ = ["submit_ray_control"]
