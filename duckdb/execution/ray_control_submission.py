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

from duckdb.execution.ray_control_deadline import (
    RAY_CONTROL_DEADLINE_SCHEDULER,
    RayControlDeadlineScheduler,
    RayControlScheduledCall,
)

_RAY_CONTROL_SUBMISSION_IDLE_TIMEOUT_S = 30.0
_RAY_CONTROL_SUBMISSION_STALL_TIMEOUT_S = 1.0
_RAY_CONTROL_SUBMISSION_MAX_WORKERS = 32
_RAY_CONTROL_SUBMISSION_MAX_PENDING = 4096
_RAY_CONTROL_SUBMISSION_MAX_PENDING_PER_OWNER = 256
_RAY_CONTROL_DEADLINE_CALLBACK_MAX_WORKERS = 32
_RAY_CONTROL_DEADLINE_CALLBACK_MAX_STALLED_WORKERS = 32
_RAY_CONTROL_DEADLINE_CALLBACK_RETRY_INITIAL_DELAY_S = 0.01
_RAY_CONTROL_DEADLINE_CALLBACK_RETRY_MAX_DELAY_S = 1.0
_RAY_CONTROL_DEADLINE_LOCK_RETRY_DELAY_S = 0.001

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
        # Identity/deadline fields are protected by the executor condition. The
        # two timestamps are deliberately published before taking that lock so
        # deadline handling cannot mistake lock contention for a stalled call.
        self.running_future: Future[Any] | None = None
        self.deadline_call: RayControlScheduledCall | None = None
        self.callback_finished_at: float | None = None
        self.completion_finished_at: float | None = None
        self.stalled = False
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"{executor._thread_name_prefix}-{sequence}",
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
        deadline_scheduler: RayControlDeadlineScheduler | None = None,
        thread_name_prefix: str = "vane-ray-control-submit",
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
        resolved_thread_name_prefix = str(thread_name_prefix or "").strip()
        if not resolved_thread_name_prefix:
            raise ValueError("Ray control submission thread name prefix must not be empty")
        self._condition = threading.Condition()
        self._owners: dict[str, _RayControlSubmissionOwner] = {}
        self._ready_owners: deque[str] = deque()
        self._workers: dict[int, _RayControlSubmissionWorker] = {}
        self._next_sequence = 0
        self._pending_submissions = 0
        self._deadline_scheduler = RAY_CONTROL_DEADLINE_SCHEDULER if deadline_scheduler is None else deadline_scheduler
        self._idle_timeout_s = float(idle_timeout_s)
        self._stall_timeout_s = float(stall_timeout_s)
        self._max_workers = int(max_workers)
        self._max_stalled_workers = int(resolved_max_stalled_workers)
        self._max_pending_submissions = int(max_pending_submissions)
        self._max_pending_per_owner = int(max_pending_per_owner)
        self._thread_name_prefix = resolved_thread_name_prefix

    def submit(self, owner_scope: str, callback: Callable[[], _T]) -> Future[_T]:
        """Admit one callback atomically or raise before it can run."""
        owner_key = str(owner_scope or "").strip()
        if not owner_key:
            raise ValueError("Ray control submission requires an explicit owner scope")
        if not callable(callback):
            raise TypeError("Ray control submission callback must be callable")
        # A submission cannot be allowed to run until the clock that owns its
        # invocation and completion deadlines is available.
        self._deadline_scheduler.ensure_started()
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

    def _start_workers_if_needed_locked(self) -> None:
        if self._stalled_worker_count_locked() > self._max_stalled_workers:
            return
        desired_workers = min(
            self._max_workers,
            self._active_running_callback_count_locked() + len(self._ready_owners),
        )
        if desired_workers <= 0:
            return
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
                    callback_deadline = time.monotonic() + self._stall_timeout_s
                    deadline_call: RayControlScheduledCall

                    def deadline_elapsed() -> None:
                        self._worker_deadline_elapsed(
                            worker,
                            deadline_call,
                            phase="callback",
                            deadline=callback_deadline,
                        )

                    deadline_call = self._deadline_scheduler.create_inline(deadline_elapsed)
                    self._deadline_scheduler.schedule_at(deadline_call, callback_deadline)
                    owner.running = True
                    worker.running_future = future
                    worker.deadline_call = deadline_call
                    worker.callback_finished_at = None
                    worker.completion_finished_at = None
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
            # Publish progress before any executor lock acquisition. The
            # deadline clock can then distinguish a returned callback from a
            # callback stalled behind unrelated condition-lock contention.
            worker.callback_finished_at = time.monotonic()
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
                # ``Future.set_result`` and ``set_exception`` invoke registered
                # callbacks inline. They are part of the bounded work phase as
                # well: a blocked completion callback must release a schedulable
                # slot for unrelated owners.
                worker.completion_finished_at = time.monotonic()
                result = None
                error = None
                del callback
                del future
                with self._condition:
                    deadline_call = worker.deadline_call
                    worker.deadline_call = None
                    if deadline_call is not None:
                        deadline_call.cancel()
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

            def fail_submission(
                rejected_future: Future[Any] = future,
                rejection_message: str = message,
            ) -> None:
                if rejected_future.done():
                    return
                try:
                    rejected_future.set_exception(RuntimeError(rejection_message))
                except BaseException:
                    pass

            def submission_is_pending(rejected_future: Future[Any] = future) -> bool:
                return not rejected_future.done()

            _dispatch_ray_control_callback(
                f"submission-failure:{id(future):x}",
                fail_submission,
                still_owned=submission_is_pending,
            )

    def _worker_deadline_elapsed(
        self,
        worker: _RayControlSubmissionWorker,
        call: RayControlScheduledCall,
        *,
        phase: str,
        deadline: float,
    ) -> None:
        """Replace a genuinely stalled worker without occupying the clock."""
        if not self._condition.acquire(blocking=False):
            # Executor bookkeeping can briefly contend with a worker that has
            # already published its lock-free phase timestamps. Retry the
            # inspection instead of delaying every unrelated process deadline.
            retry_call = self._deadline_scheduler.create_inline(
                lambda: self._worker_deadline_elapsed(
                    worker,
                    call,
                    phase=phase,
                    deadline=deadline,
                )
            )
            self._deadline_scheduler.schedule(
                retry_call,
                _RAY_CONTROL_DEADLINE_LOCK_RETRY_DELAY_S,
            )
            return
        try:
            rejected, rejection_message = self._worker_deadline_elapsed_locked(
                worker,
                call,
                phase=phase,
                deadline=deadline,
            )
        finally:
            self._condition.release()
        self._fail_submissions(rejected, rejection_message)

    def _worker_deadline_elapsed_locked(
        self,
        worker: _RayControlSubmissionWorker,
        call: RayControlScheduledCall,
        *,
        phase: str,
        deadline: float,
    ) -> tuple[list[Future[Any]], str]:
        """Apply one deadline transition while ``_condition`` is held."""
        rejected: list[Future[Any]] = []
        rejection_message = ""
        if (
            self._workers.get(worker.sequence) is not worker
            or worker.running_future is None
            or worker.stalled
            or worker.deadline_call is not call
        ):
            return rejected, rejection_message

        finished_at = worker.callback_finished_at if phase == "callback" else worker.completion_finished_at
        phase_finished_in_time = finished_at is not None and finished_at <= deadline
        if phase == "callback" and phase_finished_in_time:
            assert finished_at is not None
            completion_deadline = finished_at + self._stall_timeout_s
            completion_finished_at = worker.completion_finished_at
            if completion_finished_at is not None and completion_finished_at <= completion_deadline:
                worker.deadline_call = None
                self._condition.notify_all()
                return rejected, rejection_message
            if completion_finished_at is None:
                completion_call: RayControlScheduledCall

                def completion_deadline_elapsed() -> None:
                    self._worker_deadline_elapsed(
                        worker,
                        completion_call,
                        phase="completion",
                        deadline=completion_deadline,
                    )

                completion_call = self._deadline_scheduler.create_inline(completion_deadline_elapsed)
                worker.deadline_call = completion_call
                try:
                    self._deadline_scheduler.schedule_at(
                        completion_call,
                        completion_deadline,
                    )
                except BaseException:
                    # Losing deadline supervision is equivalent to losing a
                    # schedulable worker: retire it after completion and let a
                    # replacement own unrelated queued work.
                    worker.deadline_call = None
                else:
                    self._condition.notify_all()
                    return rejected, rejection_message
        elif phase == "completion" and phase_finished_in_time:
            worker.deadline_call = None
            self._condition.notify_all()
            return rejected, rejection_message

        worker.stalled = True
        worker.deadline_call = None
        if self._stalled_worker_count_locked() > self._max_stalled_workers:
            # At most max_workers callbacks can cross the threshold at one
            # deadline. Stop admitting more until they return, bounding live
            # callback workers by max_stalled_workers + max_workers.
            rejected = self._reject_queued_submissions_locked()
            rejection_message = (
                f"Ray control submission stalled callback capacity is exhausted: capacity={self._max_stalled_workers}"
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
        return rejected, rejection_message


_RAY_CONTROL_DEADLINE_CALLBACK_EXECUTOR = _RayControlSubmissionExecutor(
    max_workers=_RAY_CONTROL_DEADLINE_CALLBACK_MAX_WORKERS,
    max_stalled_workers=_RAY_CONTROL_DEADLINE_CALLBACK_MAX_STALLED_WORKERS,
    max_pending_per_owner=1,
    thread_name_prefix="vane-ray-control-deadline-callback",
)
_RAY_CONTROL_SUBMISSION_EXECUTOR = _RayControlSubmissionExecutor()


def _dispatch_ray_control_callback(
    owner_scope: str,
    callback: Callable[[], None],
    *,
    still_owned: Callable[[], bool] = lambda: True,
    retry_delay_s: float = _RAY_CONTROL_DEADLINE_CALLBACK_RETRY_INITIAL_DELAY_S,
) -> None:
    """Hand potentially blocking work to the bounded, recoverable pool."""
    if not still_owned():
        return

    def invoke() -> None:
        if not still_owned():
            return
        try:
            callback()
        except BaseException:
            # Scheduled owners implement their own terminal/retry policy. Keep
            # a malformed owner callback from consuming a reusable worker.
            pass

    next_retry_delay = min(
        max(float(retry_delay_s), _RAY_CONTROL_DEADLINE_CALLBACK_RETRY_INITIAL_DELAY_S) * 2,
        _RAY_CONTROL_DEADLINE_CALLBACK_RETRY_MAX_DELAY_S,
    )

    def retry_dispatch() -> None:
        if not still_owned():
            return
        retry_call = RAY_CONTROL_DEADLINE_SCHEDULER.create_inline(
            lambda: _dispatch_ray_control_callback(
                owner_scope,
                callback,
                still_owned=still_owned,
                retry_delay_s=next_retry_delay,
            )
        )
        RAY_CONTROL_DEADLINE_SCHEDULER.schedule(
            retry_call,
            retry_delay_s,
        )

    try:
        dispatch_future = _RAY_CONTROL_DEADLINE_CALLBACK_EXECUTOR.submit(
            owner_scope,
            invoke,
        )
    except BaseException:
        retry_dispatch()
        return

    def resubmit_rejected_dispatch(done: Future[Any]) -> None:
        try:
            done.result()
        except BaseException:
            retry_dispatch()

    # This is an executor-owned ``concurrent.futures.Future``; the callback is
    # internal and cannot be replaced by an owner implementation.
    dispatch_future.add_done_callback(resubmit_rejected_dispatch)


def create_ray_control_deadline(
    owner_scope: str,
    callback: Callable[[], None],
) -> RayControlScheduledCall:
    """Create a cancellable deadline whose owner work never runs on the clock."""
    owner_key = str(owner_scope or "").strip()
    if not owner_key:
        raise ValueError("Ray control deadline requires an explicit owner scope")
    if not callable(callback):
        raise TypeError("Ray control deadline callback must be callable")
    scheduled_call: RayControlScheduledCall

    def deadline_elapsed() -> None:
        _dispatch_ray_control_callback(
            f"{owner_key}:deadline:{id(scheduled_call):x}",
            callback,
            still_owned=lambda: not scheduled_call.cancelled(),
        )

    scheduled_call = RAY_CONTROL_DEADLINE_SCHEDULER.create_inline(deadline_elapsed)
    return scheduled_call


def submit_ray_control(owner_scope: str, callback: Callable[[], _T]) -> Future[_T]:
    """Submit one Ray control call with bounded, owner-ordered execution."""
    return _RAY_CONTROL_SUBMISSION_EXECUTOR.submit(owner_scope, callback)


__all__ = ["create_ray_control_deadline", "submit_ray_control"]
