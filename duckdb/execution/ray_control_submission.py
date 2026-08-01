# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, TypeVar

_RAY_CONTROL_SUBMISSION_IDLE_TIMEOUT_S = 30.0

_T = TypeVar("_T")


class _RayControlSubmissionWorker:
    """Serialize potentially blocking submissions for one ownership scope."""

    def __init__(
        self,
        executor: _RayControlSubmissionExecutor,
        owner_scope: str,
        *,
        sequence: int,
        idle_timeout_s: float,
    ) -> None:
        self.executor = executor
        self.owner_scope = str(owner_scope)
        self.accepting = True
        self.queue: queue.Queue[tuple[Future[Any], Callable[[], Any]]] = queue.Queue()
        self.idle_timeout_s = float(idle_timeout_s)
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"vane-ray-control-submit-{sequence}",
        )

    def start(self) -> None:
        self.thread.start()

    def submit(self, callback: Callable[[], _T]) -> Future[_T]:
        if not self.accepting:
            raise RuntimeError("Ray control submission worker is retired")
        future: Future[_T] = Future()
        self.queue.put((future, callback))
        return future

    def _run(self) -> None:
        while True:
            try:
                future, callback = self.queue.get(timeout=self.idle_timeout_s)
            except queue.Empty:
                if self.executor._retire_if_idle(self):
                    return
                continue
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
                        # malformed callback must not terminate this scope's
                        # submission worker.
                        pass
            finally:
                # The next queue wait must not retain the previous stream owner
                # or ObjectRef through this worker's frame.
                result = None
                del callback
                del future


class _RayControlSubmissionExecutor:
    """One independently progressing worker per admitted control owner."""

    def __init__(
        self,
        *,
        idle_timeout_s: float = _RAY_CONTROL_SUBMISSION_IDLE_TIMEOUT_S,
    ) -> None:
        if not math.isfinite(idle_timeout_s) or idle_timeout_s <= 0:
            raise ValueError("Ray control submission idle timeout must be finite and positive")
        self._lock = threading.Lock()
        self._workers: dict[str, _RayControlSubmissionWorker] = {}
        self._next_sequence = 0
        self._idle_timeout_s = float(idle_timeout_s)

    def submit(self, owner_scope: str, callback: Callable[[], _T]) -> Future[_T]:
        owner_key = str(owner_scope or "").strip()
        if not owner_key:
            raise ValueError("Ray control submission requires an explicit owner scope")
        if not callable(callback):
            raise TypeError("Ray control submission callback must be callable")
        with self._lock:
            worker = self._workers.get(owner_key)
            if worker is None or not worker.accepting:
                worker = _RayControlSubmissionWorker(
                    self,
                    owner_key,
                    sequence=self._next_sequence,
                    idle_timeout_s=self._idle_timeout_s,
                )
                self._next_sequence += 1
                self._workers[owner_key] = worker
                try:
                    worker.start()
                except BaseException:
                    worker.accepting = False
                    if self._workers.get(owner_key) is worker:
                        self._workers.pop(owner_key, None)
                    raise
            return worker.submit(callback)

    def _retire_if_idle(self, worker: _RayControlSubmissionWorker) -> bool:
        with self._lock:
            owner_key = worker.owner_scope
            if self._workers.get(owner_key) is not worker or not worker.queue.empty():
                return False
            worker.accepting = False
            self._workers.pop(owner_key, None)
            worker.owner_scope = ""
            return True


_RAY_CONTROL_SUBMISSION_EXECUTOR = _RayControlSubmissionExecutor()


def submit_ray_control(owner_scope: str, callback: Callable[[], _T]) -> Future[_T]:
    """Submit one Ray control call on its admitted owner's isolated worker."""
    return _RAY_CONTROL_SUBMISSION_EXECUTOR.submit(owner_scope, callback)


__all__ = ["submit_ray_control"]
