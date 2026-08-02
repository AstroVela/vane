# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, TypeVar

_RAY_CONTROL_SUBMISSION_IDLE_TIMEOUT_S = 30.0
_RAY_CONTROL_SUBMISSION_STALL_TIMEOUT_S = 1.0
_RAY_CONTROL_SUBMISSION_MAX_WORKERS = 32
_RAY_CONTROL_SUBMISSION_MAX_PENDING = 4096
_RAY_CONTROL_SUBMISSION_MAX_PENDING_PER_OWNER = 256

_T = TypeVar("_T")
_Submission = tuple[Future[Any], Callable[[], Any]]


class _RayControlSubmissionOwner:
    """FIFO submission state for one ownership scope."""

    def __init__(self, owner_scope: str) -> None:
        self.owner_scope = str(owner_scope)
        self.queue: deque[_Submission] = deque()
        self.running = False
        self.ready = False


class _RayControlSubmissionWorker:
    """One daemon in the bounded process-wide control submission pool."""

    def __init__(
        self,
        executor: _RayControlSubmissionExecutor,
        *,
        sequence: int,
    ) -> None:
        self.executor = executor
        self.sequence = int(sequence)
        # These fields are protected by the executor condition. A stalled
        # worker remains tracked until its callback actually returns, but no
        # longer consumes one of the schedulable worker slots.
        self.running_future: Future[Any] | None = None
        self.stall_deadline: float | None = None
        self.stalled = False
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"vane-ray-control-submit-{sequence}",
        )

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        self.executor._run_worker(self)


class _RayControlSubmissionExecutor:
    """Bounded active/stalled workers with per-owner FIFO ordering."""

    def __init__(
        self,
        *,
        idle_timeout_s: float = _RAY_CONTROL_SUBMISSION_IDLE_TIMEOUT_S,
        stall_timeout_s: float = _RAY_CONTROL_SUBMISSION_STALL_TIMEOUT_S,
        max_workers: int = _RAY_CONTROL_SUBMISSION_MAX_WORKERS,
        max_stalled_workers: int | None = None,
        max_pending_submissions: int = _RAY_CONTROL_SUBMISSION_MAX_PENDING,
        max_pending_per_owner: int = _RAY_CONTROL_SUBMISSION_MAX_PENDING_PER_OWNER,
    ) -> None:
        if not math.isfinite(idle_timeout_s) or idle_timeout_s <= 0:
            raise ValueError("Ray control submission idle timeout must be finite and positive")
        if not math.isfinite(stall_timeout_s) or stall_timeout_s <= 0:
            raise ValueError("Ray control submission stall timeout must be finite and positive")
        resolved_max_stalled_workers = max_workers if max_stalled_workers is None else max_stalled_workers
        for name, value in (
            ("max_workers", max_workers),
            ("max_stalled_workers", resolved_max_stalled_workers),
            ("max_pending_submissions", max_pending_submissions),
            ("max_pending_per_owner", max_pending_per_owner),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"Ray control submission {name} must be a positive integer")
        if max_pending_submissions < max_workers:
            raise ValueError("Ray control submission pending capacity must be at least max_workers")
        if max_pending_submissions <= resolved_max_stalled_workers:
            raise ValueError("Ray control submission pending capacity must exceed max_stalled_workers")
        self._condition = threading.Condition()
        self._owners: dict[str, _RayControlSubmissionOwner] = {}
        self._ready_owners: deque[str] = deque()
        self._workers: dict[int, _RayControlSubmissionWorker] = {}
        self._next_sequence = 0
        self._watchdog_thread: threading.Thread | None = None
        self._next_watchdog_sequence = 0
        self._pending_submissions = 0
        self._idle_timeout_s = float(idle_timeout_s)
        self._stall_timeout_s = float(stall_timeout_s)
        self._max_workers = int(max_workers)
        self._max_stalled_workers = int(resolved_max_stalled_workers)
        self._max_pending_submissions = int(max_pending_submissions)
        self._max_pending_per_owner = int(max_pending_per_owner)

    def submit(self, owner_scope: str, callback: Callable[[], _T]) -> Future[_T]:
        """Admit one callback atomically or raise before it can run."""
        owner_key = str(owner_scope or "").strip()
        if not owner_key:
            raise ValueError("Ray control submission requires an explicit owner scope")
        if not callable(callback):
            raise TypeError("Ray control submission callback must be callable")
        future: Future[_T] = Future()
        executor_ref = weakref.ref(self)

        def discard_cancelled_submission(done: Future[_T]) -> None:
            if not done.cancelled():
                return
            executor = executor_ref()
            if executor is not None:
                executor._discard_cancelled_submission(owner_key, done)

        future.add_done_callback(discard_cancelled_submission)
        with self._condition:
            if self._stalled_worker_count_locked() > self._max_stalled_workers:
                raise RuntimeError(
                    "Ray control submission stalled callback capacity is exhausted: "
                    f"capacity={self._max_stalled_workers}"
                )
            owner = self._owners.get(owner_key)
            owner_pending = 0 if owner is None else len(owner.queue) + int(owner.running)
            if owner_pending >= self._max_pending_per_owner:
                raise RuntimeError(
                    f"Ray control submission owner queue is full: owner={owner_key} "
                    f"capacity={self._max_pending_per_owner}"
                )
            if self._pending_submissions >= self._max_pending_submissions:
                raise RuntimeError(f"Ray control submission queue is full: capacity={self._max_pending_submissions}")
            if owner is None:
                owner = _RayControlSubmissionOwner(owner_key)
                self._owners[owner_key] = owner
            owner.queue.append((future, callback))
            self._pending_submissions += 1
            if not owner.running and not owner.ready:
                owner.ready = True
                self._ready_owners.append(owner_key)
            try:
                self._start_workers_if_needed_locked()
            except BaseException:
                queued_future, _queued_callback = owner.queue.pop()
                assert queued_future is future
                self._pending_submissions -= 1
                if not owner.queue and not owner.running:
                    if owner.ready:
                        self._ready_owners = deque(
                            queued_owner for queued_owner in self._ready_owners if queued_owner != owner_key
                        )
                        owner.ready = False
                    if self._owners.get(owner_key) is owner:
                        self._owners.pop(owner_key, None)
                raise
            self._condition.notify_all()
        return future

    def _discard_cancelled_submission(
        self,
        owner_key: str,
        future: Future[Any],
    ) -> None:
        """Reclaim a queued cancellation even if every worker is blocked."""
        discarded_callback: Callable[[], Any] | None = None
        with self._condition:
            owner = self._owners.get(owner_key)
            if owner is None:
                return
            for index, (queued_future, callback) in enumerate(owner.queue):
                if queued_future is not future:
                    continue
                del owner.queue[index]
                discarded_callback = callback
                self._pending_submissions -= 1
                break
            if discarded_callback is None:
                return
            if not owner.queue and not owner.running:
                if owner.ready:
                    self._ready_owners = deque(
                        queued_owner for queued_owner in self._ready_owners if queued_owner != owner_key
                    )
                    owner.ready = False
                if self._owners.get(owner_key) is owner:
                    self._owners.pop(owner_key, None)
            self._condition.notify_all()

    def _stalled_worker_count_locked(self) -> int:
        return sum(worker.stalled for worker in self._workers.values())

    def _active_worker_count_locked(self) -> int:
        return sum(not worker.stalled for worker in self._workers.values())

    def _active_running_callback_count_locked(self) -> int:
        return sum(worker.running_future is not None and not worker.stalled for worker in self._workers.values())

    def _ensure_watchdog_started_locked(self) -> None:
        if self._watchdog_thread is not None:
            return
        sequence = self._next_watchdog_sequence
        self._next_watchdog_sequence += 1
        thread = threading.Thread(
            target=self._run_watchdog,
            daemon=True,
            name=f"vane-ray-control-submit-watchdog-{sequence}",
        )
        # Publish only a successfully started watchdog. Starting it while the
        # condition is held ensures it observes the published reference before
        # it can inspect executor state.
        thread.start()
        self._watchdog_thread = thread

    def _start_workers_if_needed_locked(self) -> None:
        if self._stalled_worker_count_locked() > self._max_stalled_workers:
            return
        desired_workers = min(
            self._max_workers,
            self._active_running_callback_count_locked() + len(self._ready_owners),
        )
        if desired_workers <= 0:
            return
        self._ensure_watchdog_started_locked()
        while self._active_worker_count_locked() < desired_workers:
            sequence = self._next_sequence
            self._next_sequence += 1
            worker = _RayControlSubmissionWorker(
                self,
                sequence=sequence,
            )
            self._workers[sequence] = worker
            try:
                worker.start()
            except BaseException:
                self._workers.pop(sequence, None)
                raise

    def _claim_submission_locked(
        self,
        worker: _RayControlSubmissionWorker,
    ) -> tuple[_RayControlSubmissionOwner, Future[Any], Callable[[], Any]] | None:
        if worker.stalled:
            return None
        while self._ready_owners:
            owner_key = self._ready_owners.popleft()
            owner = self._owners.get(owner_key)
            if owner is None or owner.running or not owner.ready:
                continue
            owner.ready = False
            while owner.queue:
                future, callback = owner.queue.popleft()
                if future.set_running_or_notify_cancel():
                    owner.running = True
                    worker.running_future = future
                    worker.stall_deadline = time.monotonic() + self._stall_timeout_s
                    self._condition.notify_all()
                    return owner, future, callback
                self._pending_submissions -= 1
                del callback
            if self._owners.get(owner_key) is owner:
                self._owners.pop(owner_key, None)
        return None

    def _run_worker(self, worker: _RayControlSubmissionWorker) -> None:
        idle_deadline = time.monotonic() + self._idle_timeout_s
        while True:
            rejected: list[Future[Any]] = []
            rejection_message = ""
            with self._condition:
                claimed = self._claim_submission_locked(worker)
                while claimed is None:
                    if worker.stalled:
                        return
                    remaining = idle_deadline - time.monotonic()
                    if remaining <= 0:
                        if self._workers.get(worker.sequence) is worker:
                            self._workers.pop(worker.sequence, None)
                        self._condition.notify_all()
                        return
                    self._condition.wait(timeout=remaining)
                    claimed = self._claim_submission_locked(worker)
                owner, future, callback = claimed

            result: Any = None
            error: BaseException | None = None
            try:
                result = callback()
            except BaseException as exc:
                error = exc
            finally:
                # Only callback invocation time determines whether this worker
                # is stalled. Future callbacks may perform arbitrary owner-side
                # bookkeeping and must not consume the submission deadline.
                with self._condition:
                    worker.stall_deadline = None
                    self._condition.notify_all()
            try:
                if error is not None:
                    if not future.done():
                        try:
                            future.set_exception(error)
                        except BaseException:
                            pass
                else:
                    try:
                        future.set_result(result)
                    except BaseException:
                        # The Future becomes terminal before callbacks run. A
                        # malformed callback must not terminate this pool worker.
                        pass
            finally:
                result = None
                error = None
                del callback
                del future
                with self._condition:
                    worker.running_future = None
                    owner.running = False
                    self._pending_submissions -= 1
                    if owner.queue:
                        owner.ready = True
                        self._ready_owners.append(owner.owner_scope)
                    elif self._owners.get(owner.owner_scope) is owner:
                        self._owners.pop(owner.owner_scope, None)
                    retire_stalled_worker = worker.stalled
                    if retire_stalled_worker and self._workers.get(worker.sequence) is worker:
                        self._workers.pop(worker.sequence, None)
                    try:
                        self._start_workers_if_needed_locked()
                    except BaseException as exc:
                        if self._active_worker_count_locked() == 0 and self._ready_owners:
                            rejected = self._reject_queued_submissions_locked()
                            rejection_message = (
                                f"Ray control submission replacement worker is unavailable: {type(exc).__name__}: {exc}"
                            )
                    self._condition.notify_all()
                self._fail_submissions(rejected, rejection_message)
                if retire_stalled_worker:
                    return
                idle_deadline = time.monotonic() + self._idle_timeout_s

    def _reject_queued_submissions_locked(self) -> list[Future[Any]]:
        """Fail admitted-but-unclaimed work when stalled capacity is spent."""
        rejected: list[Future[Any]] = []
        self._ready_owners.clear()
        for owner_key, owner in tuple(self._owners.items()):
            owner.ready = False
            while owner.queue:
                future, callback = owner.queue.popleft()
                rejected.append(future)
                self._pending_submissions -= 1
                del callback
            if not owner.running and self._owners.get(owner_key) is owner:
                self._owners.pop(owner_key, None)
        return rejected

    @staticmethod
    def _fail_submissions(futures: list[Future[Any]], message: str) -> None:
        for future in futures:
            if future.done():
                continue
            try:
                future.set_exception(RuntimeError(message))
            except BaseException:
                pass

    def _run_watchdog(self) -> None:
        current = threading.current_thread()
        while True:
            rejected: list[Future[Any]] = []
            rejection_message = ""
            with self._condition:
                deadlines = [
                    worker.stall_deadline
                    for worker in self._workers.values()
                    if worker.running_future is not None and not worker.stalled and worker.stall_deadline is not None
                ]
                if not deadlines:
                    # Keep one watchdog while a schedulable worker can claim
                    # already-admitted FIFO work without another submit call.
                    # Once those workers retire, the watchdog may retire too.
                    if self._active_worker_count_locked() == 0:
                        if self._watchdog_thread is current:
                            self._watchdog_thread = None
                        self._condition.notify_all()
                        return
                    self._condition.wait(timeout=self._idle_timeout_s)
                    continue
                remaining = min(deadlines) - time.monotonic()
                if remaining > 0:
                    self._condition.wait(timeout=remaining)
                    continue
                now = time.monotonic()
                for worker in self._workers.values():
                    deadline = worker.stall_deadline
                    if (
                        worker.running_future is not None
                        and not worker.stalled
                        and deadline is not None
                        and deadline <= now
                    ):
                        worker.stalled = True
                        worker.stall_deadline = None
                if self._stalled_worker_count_locked() > self._max_stalled_workers:
                    # At most max_workers callbacks can cross the threshold in
                    # one watchdog pass. Stop admitting more until enough of
                    # those callbacks return, bounding live callback workers by
                    # max_stalled_workers + max_workers.
                    rejected = self._reject_queued_submissions_locked()
                    rejection_message = (
                        "Ray control submission stalled callback capacity "
                        f"is exhausted: capacity={self._max_stalled_workers}"
                    )
                else:
                    try:
                        self._start_workers_if_needed_locked()
                    except BaseException as exc:
                        if self._active_worker_count_locked() == 0 and self._ready_owners:
                            rejected = self._reject_queued_submissions_locked()
                            rejection_message = (
                                f"Ray control submission replacement worker is unavailable: {type(exc).__name__}: {exc}"
                            )
                self._condition.notify_all()
            self._fail_submissions(rejected, rejection_message)


_RAY_CONTROL_SUBMISSION_EXECUTOR = _RayControlSubmissionExecutor()


def submit_ray_control(owner_scope: str, callback: Callable[[], _T]) -> Future[_T]:
    """Submit one Ray control call with bounded, owner-ordered execution."""
    return _RAY_CONTROL_SUBMISSION_EXECUTOR.submit(owner_scope, callback)


__all__ = ["submit_ray_control"]
