# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from vane.runners.common import QueryDeadlineExceeded
from vane.runners.fte.fte_events import TaskStatusChanged, WorkerFailed
from vane.runners.fte.fte_failures import _safe_failure_message
from vane.runners.fte.fte_state import FteTaskState
from vane.runners.fte.fte_types import validate_fte_status_identity
from vane.runners.ray.safe_get import configured_ray_get_timeout_s

if TYPE_CHECKING:
    from vane.runners.fte.fte_events import FteEvent
    from vane.runners.fte.fte_scheduler import FteQueryScheduler


class FteStatusObservationRuntime:
    """One event loop and bounded dispatch capacity for driver-side status waits."""

    def __init__(self, *, capacity: int = 256, dispatch_workers: int = 16) -> None:
        if capacity <= 0 or dispatch_workers <= 0:
            raise ValueError("status observation capacity and worker count must be positive")
        # Leave room in the worker's 512-entry control group for mutations and
        # teardown, even when every admitted native attempt is waiting.
        self.slots = asyncio.Semaphore(capacity)
        self.dispatch_slots = asyncio.Semaphore(dispatch_workers)
        self.executor = ThreadPoolExecutor(max_workers=dispatch_workers, thread_name_prefix="vane-fte-events")
        self._lock = threading.Lock()
        self._active: set[Any] = set()
        self._closed = False
        self.loop = asyncio.new_event_loop()
        ready = threading.Event()

        def run() -> None:
            asyncio.set_event_loop(self.loop)
            self.loop.call_soon(ready.set)
            self.loop.run_forever()

        self.thread = threading.Thread(target=run, name="vane-fte-status-loop", daemon=True)
        self.thread.start()
        if not ready.wait(5.0):
            raise RuntimeError("FTE status observation loop did not start")

    def start(self, watcher: Any) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("FTE status observation runtime is closed")
            self._active.add(watcher)
            try:
                self.loop.call_soon_threadsafe(self._start, watcher)
            except BaseException:
                self._active.discard(watcher)
                raise

    def _start(self, watcher: Any) -> None:
        self.loop.create_task(self._run(watcher))

    async def _run(self, watcher: Any) -> None:
        try:
            await watcher._run()
        finally:
            try:
                if callable(watcher.on_exit):
                    watcher.on_exit(watcher)
            finally:
                with self._lock:
                    self._active.discard(watcher)
                watcher._settled.set()

    async def dispatch(self, function: Any, *args: Any) -> Any:
        # Bound executor submissions until the callback actually returns,
        # including after an observation has passed its own event barrier.
        async with self.dispatch_slots:
            return await self.loop.run_in_executor(self.executor, function, *args)

    def close(self, timeout_s: float = 5.0) -> None:
        with self._lock:
            if self._active:
                raise RuntimeError("cannot close FTE status observation runtime with live watchers")
            self._closed = True
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout_s)
        if self.thread.is_alive():
            raise RuntimeError("FTE status observation loop did not stop")
        self.loop.close()
        self.executor.shutdown(wait=True)


_RUNTIME_LOCK = threading.Lock()
_RUNTIME: FteStatusObservationRuntime | None = None


def get_status_observation_runtime() -> FteStatusObservationRuntime:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = FteStatusObservationRuntime()
        return _RUNTIME


def shutdown_status_observation_runtime(timeout_s: float = 5.0) -> None:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            _RUNTIME.close(timeout_s)
            _RUNTIME = None


class FteAttemptStatusWatcher:
    """Share one versioned status stream with the scheduler and result consumer."""

    def __init__(
        self,
        *,
        scheduler: FteQueryScheduler,
        attempt_id: Any,
        worker: Any,
        wait_timeout_s: float = 1.0,
        completion: Future[dict[str, Any]] | None = None,
        runtime: FteStatusObservationRuntime | None = None,
        on_exit: Any = None,
    ) -> None:
        self.scheduler = scheduler
        self.attempt_id = attempt_id
        self.worker = worker
        self.wait_timeout_s = max(0.0, float(wait_timeout_s))
        self.completion = Future() if completion is None else completion
        self.on_exit = on_exit
        self._runtime = runtime
        self._stop = threading.Event()
        self._settled = threading.Event()
        self._start_lock = threading.Lock()
        self._started = False
        self._pending_wait: asyncio.Task[Any] | None = None
        self._slot_owned = False
        self._drains: set[asyncio.Task[None]] = set()

    def start(self) -> bool:
        if not callable(getattr(self.worker, "fte_wait_task_status_async", None)):
            raise TypeError("FTE worker handle fte_wait_task_status_async must be callable")
        with self._start_lock:
            if self._started:
                return True
            if self._runtime is None:
                self._runtime = get_status_observation_runtime()
            self._started = True
            try:
                self._runtime.start(self)
            except BaseException:
                self._started = False
                raise
        return True

    def stop(self) -> None:
        # Cancel only the wait, never an executor callback that may still own
        # scheduler mutations or terminal publication. join() observes that
        # callback's real completion, including remote-wait cancellation.
        with self._start_lock:
            if self._stop.is_set():
                return
            self._stop.set()
            runtime = self._runtime
            if runtime is not None and self.is_alive():
                runtime.loop.call_soon_threadsafe(self._cancel_pending_wait)

    def _cancel_pending_wait(self) -> None:
        if self._pending_wait is not None:
            self._pending_wait.cancel()

    def join(self, timeout_s: float | None = None) -> None:
        if self._started:
            self._settled.wait(timeout_s)

    def is_alive(self) -> bool:
        return self._started and not self._settled.is_set()

    def shutdown_timeout_s(self) -> float:
        return max(5.0, self.wait_timeout_s + 5.0)

    def _enqueue_and_drain(self, event: FteEvent, barrier_ready: Future[Future[None]]) -> None:
        try:
            self.scheduler.enqueue(event)
            completion = self.scheduler.enqueue_drain_barrier()
            barrier_ready.set_result(completion)
            self.scheduler.drain()
        except Exception as exc:
            self.scheduler.fail(f"FTE status watcher failed while handling {event.event_type}: {exc}")
            raise

    async def _dispatch(self, event: FteEvent, barrier_ready: Future[Future[None]]) -> None:
        assert self._runtime is not None
        try:
            await self._runtime.dispatch(self._enqueue_and_drain, event, barrier_ready)
        except BaseException as exc:
            if not barrier_ready.done():
                barrier_ready.set_exception(exc)
            # Scheduler drain failures settle pending barriers and fail the
            # scheduler. A failure after this barrier cannot revoke a status
            # that has already been delivered.

    async def _publish(self, event: FteEvent) -> None:
        barrier_ready: Future[Future[None]] = Future()
        drain = asyncio.create_task(self._dispatch(event, barrier_ready))
        self._drains.add(drain)
        drain.add_done_callback(self._drains.discard)
        completion = await asyncio.wrap_future(barrier_ready)
        # Follow this event's causal barrier, even when our own drain keeps
        # processing later events. Retain that drain separately for teardown.
        await asyncio.wrap_future(completion)

    @staticmethod
    def _is_soft_status_wait_timeout(exc: BaseException) -> bool:
        message = _safe_failure_message(exc)
        if issubclass(type(exc), QueryDeadlineExceeded) or "query deadline expired" in message.lower():
            return False
        if issubclass(type(exc), TimeoutError):
            return True
        name = type(exc).__name__
        if name in {"TimeoutError", "GetTimeoutError"}:
            return True
        return "did not complete within" in message or "timed out" in message.lower()

    async def _wait_status(self, min_version: int | None) -> Any:
        assert self._runtime is not None

        async def acquire() -> None:
            assert self._runtime is not None
            await self._runtime.slots.acquire()
            # Set ownership inside the acquired coroutine, so cancellation of
            # wait_for after acquisition cannot leak an observation slot.
            self._slot_owned = True

        budget = configured_ray_get_timeout_s(None)
        try:
            # Semaphore.acquire() does not suspend when capacity is available.
            # wait_for(..., 0) would cancel it before even trying to acquire.
            if budget is None or not self._runtime.slots.locked():
                await acquire()
            else:
                await asyncio.wait_for(acquire(), timeout=budget)
        except asyncio.TimeoutError:
            configured_ray_get_timeout_s(None)
            raise
        return await self.worker.fte_wait_task_status_async(self.attempt_id.to_dict(), min_version, self.wait_timeout_s)

    async def _fail(self, error: BaseException) -> None:
        assert self._runtime is not None
        try:
            await self._publish(
                WorkerFailed(
                    query_id=self.scheduler.query_id,
                    worker_id=str(self.worker.worker_id),
                    worker_incarnation_id=str(self.worker.worker_incarnation_id),
                    manager_instance_id=str(self.worker.manager_instance_id),
                    error=error,
                ),
            )
        except BaseException as publication_error:
            error = RuntimeError(
                f"{_safe_failure_message(error)}; worker failure publication failed: "
                f"{_safe_failure_message(publication_error)}"
            )
        if not self.completion.done():
            self.completion.set_exception(error)

    async def _run(self) -> None:
        assert self._runtime is not None
        min_version = None
        try:
            while not self._stop.is_set():
                try:
                    self._pending_wait = asyncio.create_task(self._wait_status(min_version))
                    try:
                        status = await self._pending_wait
                    except Exception as exc:
                        if self._is_soft_status_wait_timeout(exc):
                            # Back off a failed RPC, without periodic wakeups
                            # while an ordinary remote wait remains pending.
                            await asyncio.sleep(0.05)
                            continue
                        raise
                    finally:
                        self._pending_wait = None
                    if self._stop.is_set():
                        break
                    if not isinstance(status, dict):
                        raise TypeError("fte_wait_task_status must return a dict")
                    validate_fte_status_identity(status, self.attempt_id)
                    version = status.get("version")
                    if version is not None:
                        try:
                            min_version = int(version) + 1
                        except (TypeError, ValueError):
                            min_version = None
                    raw_state = status.get("state")
                    try:
                        state = raw_state if isinstance(raw_state, FteTaskState) else FteTaskState(str(raw_state))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"unknown FTE task state: {raw_state!r}") from exc
                    await self._publish(
                        TaskStatusChanged.from_status(self.scheduler.query_id, self.attempt_id, status),
                    )
                    if state in {
                        FteTaskState.FINISHED,
                        FteTaskState.FAILED,
                        FteTaskState.CANCELED,
                        FteTaskState.ABORTED,
                    }:
                        if not self.completion.done():
                            self.completion.set_result(status)
                        return
                finally:
                    if self._slot_owned:
                        self._runtime.slots.release()
                        self._slot_owned = False
        except asyncio.CancelledError:
            if not self._stop.is_set():
                await self._fail(RuntimeError("FTE status observation was cancelled before teardown"))
        except BaseException as error:
            await self._fail(error)
        finally:
            if not self.completion.done():
                self.completion.set_exception(InterruptedError("FTE status observation stopped"))
            if self._drains:
                await asyncio.gather(*self._drains)
