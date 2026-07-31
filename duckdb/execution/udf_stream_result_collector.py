# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import heapq
import math
import os
import sys
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from duckdb.execution.ray_stream_adapter import (
    RAY_STREAM_CLEANUP_CANCELLATION,
    RAY_STREAM_CLEANUP_CONTROL,
    RAY_STREAM_CLEANUP_CORE,
    RAY_STREAM_CLEANUP_OUTPUT,
    RayStreamAdapter,
    RayStreamCleanupOperation,
    TaskLeaseObjectRefGenerator,
    resolve_ray_control_ack,
)
from duckdb.execution.udf_ray_actor_state import format_stateful_actor_loss
from duckdb.execution.udf_ray_config import REF_BUNDLE_RESULT_MARKER
from duckdb.execution.udf_ray_stream_protocol import (
    validate_stream_block_metadata,
    validate_stream_error_metadata,
)

_GENERATOR_READINESS_POLL_INITIAL_DELAY_S = 0.001
_GENERATOR_READINESS_POLL_MAX_DELAY_S = 0.01
_DEFAULT_CONTROL_CLEANUP_WORKERS = 4
_DEFAULT_OUTPUT_CLEANUP_WORKERS = 4
_DEFAULT_CANCELLATION_CLEANUP_WORKERS = 2
_DEFAULT_CORE_CLEANUP_WORKERS = 2
_CLEANUP_RETRY_INITIAL_DELAY_S = 0.01
_CLEANUP_RETRY_MAX_DELAY_S = 1.0


def _cleanup_worker_count(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0 or value > 256:
        raise ValueError(f"{name} must be between 1 and 256")
    return value


def _collector_debug_log(event: str, record: _StreamRecord, **fields: Any) -> None:
    value = os.environ.get("DUCKDB_DISTRIBUTED_DEBUG", "")
    if value.strip().lower() not in {"1", "true", "yes", "on"}:
        return
    parts = [
        f"event={event}",
        f"pid={os.getpid()}",
        f"t={time.monotonic():.6f}",
        f"slot={record.slot_id}",
        f"submit={record.submit_id}",
        f"sequence={record.sequence}",
        f"phase={record.phase}",
    ]
    parts.extend(f"{key}={value}" for key, value in fields.items())
    print("[vane-ray-stream-collector] " + " ".join(parts), file=sys.stderr, flush=True)


@dataclass(frozen=True)
class _DrainCapacity:
    rows: int
    bytes: int | None = None
    item_bytes: int | None = None


@dataclass
class _OutputLeaseToken:
    request_id: str
    lease_id: str
    query_id: str
    driver: Any
    slot_id: int
    submit_id: int
    size_bytes: int
    handed_off: bool = False
    handoff_pending: bool = False
    release_pending: bool = False
    handoff_response_ref: Any | None = None
    release_response_ref: Any | None = None


@dataclass
class _ReadyEvent:
    slot_id: int
    submit_id: int
    kind: str
    payload: Any
    size_bytes: int = 0
    output_token: _OutputLeaseToken | None = None

    def as_tuple(self) -> tuple[Any, ...]:
        if self.kind != "data":
            return (self.slot_id, self.submit_id, self.kind, self.payload)
        if self.output_token is None:
            raise RuntimeError("Ray UDF data event is missing its output lease")
        return (
            self.slot_id,
            self.submit_id,
            self.kind,
            self.payload,
            self.output_token.request_id,
            self.output_token.lease_id,
        )


@dataclass
class _StreamRecord:
    slot_id: int
    submit_id: int
    adapter: RayStreamAdapter
    sequence: int
    phase: str = "block"
    block_ref: Any | None = None
    metadata_ref: Any | None = None
    terminal_ref: Any | None = None
    metadata: dict[str, Any] | None = None
    block_item_capacity_bytes: int | None = None
    output_request_id: str = ""
    output_request: dict[str, Any] | None = None
    output_lease_ref: Any | None = None
    output_cancel_sent: bool = False
    producer_completed: bool = False
    terminal: bool = False
    error_context: dict[str, Any] | None = None
    wait_kind: str = ""
    wait_future: Any | None = None
    completion_future: Any | None = None
    terminal_signal_observed: bool = False
    next_ref_ready: bool = False
    ready_sequence: int | None = None
    cleanup_started: bool = False
    registration_accepted: bool = True
    registration_cancelled: bool = False


class _CleanupGroup:
    """Completion fence for one atomically accepted terminal cleanup plan."""

    def __init__(
        self,
        operation_count: int,
        *,
        on_complete: Callable[[BaseException | None], None],
    ) -> None:
        self._remaining = int(operation_count)
        self._errors: list[BaseException] = []
        self._on_complete = on_complete
        self._callback_thread: threading.Thread | None = None
        self._callback_complete = False
        self._cv = threading.Condition()

    def finish(self, error: BaseException | None) -> None:
        callback: Callable[[BaseException | None], None] | None = None
        first_error: BaseException | None = None
        with self._cv:
            if error is not None:
                self._errors.append(error)
            self._remaining -= 1
            if self._remaining < 0:
                raise RuntimeError("Ray UDF cleanup group count underflow")
            if self._remaining == 0:
                callback = self._on_complete
                self._on_complete = lambda _error: None
                first_error = self._errors[0] if self._errors else None
                self._callback_thread = threading.current_thread()
                self._cv.notify_all()
        if callback is None:
            return
        try:
            callback(first_error)
        finally:
            with self._cv:
                self._callback_thread = None
                self._callback_complete = True
                self._cv.notify_all()

    def wait(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        current = threading.current_thread()
        with self._cv:
            while self._remaining or (not self._callback_complete and self._callback_thread is not current):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(timeout=remaining)
            return True

    def callback_owned_by_current_thread(self) -> bool:
        with self._cv:
            return self._remaining == 0 and self._callback_thread is threading.current_thread()

    @property
    def errors(self) -> tuple[BaseException, ...]:
        with self._cv:
            return tuple(self._errors)


@dataclass
class _CleanupWorkItem:
    operation: Callable[[], Any]
    group: _CleanupGroup
    retry_on_error: bool = False
    retry_on_incomplete: bool = False
    retry_count: int = 0


class _DaemonCleanupPool:
    """Lazily scaled, fixed-size daemon bulkhead for blocking terminal calls."""

    def __init__(self, worker_count: int, *, name: str) -> None:
        self._worker_count = int(worker_count)
        self._name = str(name)
        self._cv = threading.Condition()
        self._queue: list[tuple[float, int, _CleanupWorkItem]] = []
        self._next_queue_sequence = 0
        self._threads: list[threading.Thread] = []
        self._started = False
        self._stop_when_idle = False
        self._idle_workers = 0

    def start(self) -> None:
        with self._cv:
            if self._started:
                return
            self._started = True

    def _ensure_worker_locked(self) -> None:
        if self._threads:
            return
        try:
            self._start_worker_locked()
        except BaseException as exc:
            raise RuntimeError(f"Ray UDF cleanup pool {self._name!r} did not start") from exc

    def _start_worker_locked(self) -> None:
        worker = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"{self._name}-{len(self._threads)}",
        )
        worker.start()
        self._threads.append(worker)

    def _scale_workers_locked(self) -> None:
        needed = max(0, len(self._queue) - self._idle_workers)
        available = self._worker_count - len(self._threads)
        for _index in range(min(needed, available)):
            try:
                self._start_worker_locked()
            except BaseException:
                # The already-running worker remains the durable queue owner.
                break

    def _enqueue_locked(
        self,
        operations: Sequence[Callable[[], Any]],
        group: _CleanupGroup,
    ) -> None:
        if not self._started or self._stop_when_idle:
            raise RuntimeError(f"Ray UDF cleanup pool {self._name!r} is not accepting work")
        ready_at = time.monotonic()
        items = [
            _CleanupWorkItem(
                operation,
                group,
                retry_on_error=(
                    operation.retry_on_error if isinstance(operation, RayStreamCleanupOperation) else False
                ),
                retry_on_incomplete=(
                    operation.retry_on_incomplete if isinstance(operation, RayStreamCleanupOperation) else False
                ),
            )
            for operation in operations
        ]
        for item in items:
            sequence = self._next_queue_sequence
            self._next_queue_sequence += 1
            heapq.heappush(self._queue, (ready_at, sequence, item))
        self._scale_workers_locked()
        self._cv.notify_all()

    def stop_when_idle(self) -> None:
        with self._cv:
            self._stop_when_idle = True
            self._cv.notify_all()

    def join(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cv:
            threads = tuple(self._threads)
        current = threading.current_thread()
        for worker in threads:
            if worker is current:
                continue
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
        return not any(worker is not current and worker.is_alive() for worker in threads)

    @property
    def threads(self) -> tuple[threading.Thread, ...]:
        with self._cv:
            return tuple(self._threads)

    def _take_ready_item_locked(self) -> tuple[_CleanupWorkItem | None, float | None]:
        if not self._queue:
            return None, None
        now = time.monotonic()
        ready_at, _sequence, item = self._queue[0]
        if ready_at <= now:
            heapq.heappop(self._queue)
            return item, None
        return None, max(0.0, ready_at - now)

    @staticmethod
    def _retry_delay(retry_count: int) -> float:
        exponent = min(max(0, int(retry_count) - 1), 30)
        return min(
            math.ldexp(_CLEANUP_RETRY_INITIAL_DELAY_S, exponent),
            _CLEANUP_RETRY_MAX_DELAY_S,
        )

    def _run(self) -> None:
        while True:
            with self._cv:
                item: _CleanupWorkItem | None = None
                while item is None:
                    item, retry_wait = self._take_ready_item_locked()
                    if item is not None:
                        break
                    if not self._queue and self._stop_when_idle:
                        return
                    self._idle_workers += 1
                    try:
                        self._cv.wait(timeout=retry_wait)
                    finally:
                        self._idle_workers -= 1
            error: BaseException | None = None
            retry = False
            result: Any = None
            try:
                result = item.operation()
            except BaseException as exc:
                if item.retry_on_error:
                    retry = True
                else:
                    error = exc
            else:
                retry = result is False and item.retry_on_incomplete
            if retry:
                item.retry_count += 1
                ready_at = time.monotonic() + self._retry_delay(item.retry_count)
                with self._cv:
                    sequence = self._next_queue_sequence
                    self._next_queue_sequence += 1
                    heapq.heappush(self._queue, (ready_at, sequence, item))
                    self._cv.notify_all()
                del item
                del error
                del result
                continue
            item.group.finish(error)
            del item
            del error
            del result


class AsyncResultCollector:
    """Capacity-aware scheduler for lease-owned Ray generator streams.

    ``_cv`` owns every local stream transition. A registering record is not
    schedulable until ``track_generator_ref`` commits its accepted bit, and a
    terminal record is removed only while ``_cleanup_handoff_lock`` prevents
    shutdown from observing an ownership gap.

    The asyncio thread is the sole generator readiness/read consumer. Terminal
    Ray, driver, and CoreWorker calls run in fixed-size lane-specific daemon
    pools; task admission bounds the queued plans, while lane isolation keeps a
    blocked cancellation from withholding stream deletion or unrelated control
    work. Cancellation acceptance intentionally precedes task-lease release so
    running work remains budgeted. User wakeups are invoked only after the
    complete terminal plan has an accepted cleanup owner.
    """

    def __init__(self, *, ray_module: Any | None = None) -> None:
        if ray_module is None:
            import ray as imported_ray

            active_ray_module = imported_ray
        else:
            active_ray_module = ray_module

        self._ray = active_ray_module
        self._shutdown_timeout_s = float(os.environ.get("VANE_UDF_STREAM_SHUTDOWN_TIMEOUT_S", "5"))
        if not math.isfinite(self._shutdown_timeout_s) or self._shutdown_timeout_s <= 0:
            raise ValueError("VANE_UDF_STREAM_SHUTDOWN_TIMEOUT_S must be positive")
        self._cv = threading.Condition()
        self._shutdown = False
        self._started = False
        self._thread_started = False
        self._thread_error: BaseException | None = None
        self._wakeup_fn: Any | None = None
        self._records: dict[tuple[int, int], _StreamRecord] = {}
        self._ready_by_slot: dict[int, deque[_ReadyEvent]] = defaultdict(deque)
        self._capacity_by_slot: dict[int, _DrainCapacity] = {}
        self._active_output_leases: dict[tuple[str, str], _OutputLeaseToken] = {}
        self._cancelled_slots: set[int] = set()
        self._next_sequence = 0
        self._next_ready_sequence = 0
        self._cleanup_handoff_lock = threading.RLock()
        self._cleanup_groups: set[_CleanupGroup] = set()
        self._cleanup_groups_by_slot: dict[int, set[_CleanupGroup]] = defaultdict(set)
        self._terminal_cleanup_errors: list[BaseException] = []
        self._cleanup_pools_stopping = False
        self._cleanup_pools = {
            RAY_STREAM_CLEANUP_CONTROL: _DaemonCleanupPool(
                _cleanup_worker_count(
                    "VANE_UDF_STREAM_CONTROL_CLEANUP_WORKERS",
                    _DEFAULT_CONTROL_CLEANUP_WORKERS,
                ),
                name="udf-ray-stream-control-cleanup",
            ),
            RAY_STREAM_CLEANUP_OUTPUT: _DaemonCleanupPool(
                _cleanup_worker_count(
                    "VANE_UDF_STREAM_OUTPUT_CLEANUP_WORKERS",
                    _DEFAULT_OUTPUT_CLEANUP_WORKERS,
                ),
                name="udf-ray-stream-output-cleanup",
            ),
            RAY_STREAM_CLEANUP_CANCELLATION: _DaemonCleanupPool(
                _cleanup_worker_count(
                    "VANE_UDF_STREAM_CANCELLATION_CLEANUP_WORKERS",
                    _DEFAULT_CANCELLATION_CLEANUP_WORKERS,
                ),
                name="udf-ray-stream-cancellation-cleanup",
            ),
            RAY_STREAM_CLEANUP_CORE: _DaemonCleanupPool(
                _cleanup_worker_count(
                    "VANE_UDF_STREAM_CORE_CLEANUP_WORKERS",
                    _DEFAULT_CORE_CLEANUP_WORKERS,
                ),
                name="udf-ray-stream-core-cleanup",
            ),
        }
        self._readiness_delay_s = _GENERATOR_READINESS_POLL_INITIAL_DELAY_S
        self._readiness_timer: asyncio.TimerHandle | None = None
        self._readiness_wakeup_pending = False
        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._start_lock = threading.RLock()
        self._thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="udf-ray-stream-multiplexer",
        )
        started_pools: list[_DaemonCleanupPool] = []
        try:
            for pool in self._cleanup_pools.values():
                pool.start()
                started_pools.append(pool)
        except BaseException:
            for pool in started_pools:
                pool.stop_when_idle()
            for pool in started_pools:
                pool.join(self._shutdown_timeout_s)
            raise

    # Public API called by the C++ dispatcher while it owns the GIL.

    def track_generator_ref(
        self,
        slot_id: int,
        submit_id: int,
        source: Any,
        error_context: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(source, TaskLeaseObjectRefGenerator):
            raise TypeError(
                f"distributed Ray UDF submission must return TaskLeaseObjectRefGenerator; got {type(source).__name__}"
            )
        key = (int(slot_id), int(submit_id))
        adapter: RayStreamAdapter | None = None
        record: _StreamRecord | None = None
        accepted = False
        try:
            # Serialize only registration/startup against shutdown. The record
            # stays invisible to the scheduler until the final accepted bit is
            # committed, so a ready stream cannot retire before C++ owns its
            # corresponding submit_id.
            with self._start_lock:
                adapter = RayStreamAdapter(source, ray_module=self._ray)
                source = None
                with self._cv:
                    self._raise_if_stopped_locked()
                    if key in self._records:
                        raise ValueError(f"duplicate Ray UDF stream identity slot={key[0]} submit={key[1]}")
                    if key[0] in self._cancelled_slots:
                        raise RuntimeError(f"Ray UDF slot {key[0]} is cancelled")
                    record = _StreamRecord(
                        slot_id=key[0],
                        submit_id=key[1],
                        adapter=adapter,
                        sequence=self._next_sequence,
                        error_context=dict(error_context) if error_context else None,
                        registration_accepted=False,
                    )
                    self._next_sequence += 1
                    self._records[key] = record
                    self._cv.notify_all()
                self._ensure_started()
                with self._cv:
                    if self._records.get(key) is record:
                        record.registration_accepted = True
                        accepted = True
                        self._signal_readiness_change_locked()
                    elif record.registration_cancelled:
                        raise RuntimeError(
                            f"Ray UDF stream slot={key[0]} submit={key[1]} was cancelled during registration"
                        )
                    else:
                        raise RuntimeError(f"Ray UDF stream slot={key[0]} submit={key[1]} lost registration ownership")
            assert record is not None
            _collector_debug_log("track", record)
            self._refresh_waiters()
        except BaseException as registration_error:
            cleanup_group: _CleanupGroup | None = None
            with self._cleanup_handoff_lock:
                cleanup_operations: Sequence[Callable[[], Any]] = ()
                with self._cv:
                    if record is not None and self._records.pop(key, None) is record:
                        record.terminal = True
                        record.cleanup_started = True
                        cleanup_operations = record.adapter.cancel_operations()
                        self._signal_readiness_change_locked()
                    elif record is None and adapter is not None:
                        cleanup_operations = adapter.cancel_operations()
                    elif record is None and source is not None:
                        cleanup_operations = source.cancel_operations()
                        source.retire_local_state()
                if cleanup_operations:
                    cleanup_group = self._submit_cleanup_operations(
                        self._fence_cleanup_operations(cleanup_operations),
                        store_error=False,
                        slot_ids=(key[0],),
                    )
            if cleanup_group is not None:
                cleanup_complete = cleanup_group.wait(self._shutdown_timeout_s)
                cleanup_errors = cleanup_group.errors
                if not cleanup_complete or cleanup_errors:
                    detail = (
                        "cleanup did not terminate"
                        if not cleanup_complete
                        else "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
                    )
                    add_note = getattr(registration_error, "add_note", None)
                    if callable(add_note):
                        add_note(f"submitted Ray stream cleanup failed: {detail}")
            raise
        if not accepted:
            raise RuntimeError(f"Ray UDF stream slot={key[0]} submit={key[1]} was not accepted")

    def drain_results(self, capacities: dict[Any, Any] | None = None) -> list[tuple[Any, ...]]:
        parsed = self._parse_capacities(capacities)
        results: list[tuple[Any, ...]] = []
        readiness_changed = False
        with self._cv:
            self._raise_if_stopped_locked()
            for slot_id, capacity in parsed.items():
                ready = self._ready_by_slot.get(slot_id)
                delivered_rows = 0
                delivered_bytes = 0
                while ready:
                    event = ready[0]
                    if event.kind == "data":
                        if delivered_rows >= capacity.rows:
                            break
                        if capacity.bytes is not None:
                            if capacity.bytes <= delivered_bytes:
                                break
                            if delivered_bytes + event.size_bytes > capacity.bytes:
                                break
                        if capacity.item_bytes is not None and capacity.item_bytes <= 0:
                            break
                        if capacity.item_bytes is not None and event.size_bytes > capacity.item_bytes:
                            break
                        delivered_rows += 1
                        delivered_bytes += event.size_bytes
                    results.append(ready.popleft().as_tuple())
                if delivered_rows:
                    readiness_changed = True
                if ready is not None and not ready:
                    self._ready_by_slot.pop(slot_id, None)
                remaining_bytes = None if capacity.bytes is None else max(0, capacity.bytes - delivered_bytes)
                updated_capacity = _DrainCapacity(
                    rows=max(0, capacity.rows - delivered_rows),
                    bytes=remaining_bytes,
                    item_bytes=capacity.item_bytes,
                )
                if self._capacity_by_slot.get(slot_id) != updated_capacity:
                    readiness_changed = True
                self._capacity_by_slot[slot_id] = updated_capacity
            if readiness_changed:
                self._signal_readiness_change_locked()
            else:
                self._cv.notify_all()
        self._refresh_waiters()
        return results

    def release_output_block_lease(self, request_id: str, lease_id: str) -> bool:
        key = (str(request_id), str(lease_id))
        with self._cv:
            token = self._active_output_leases.get(key)
            if token is None or token.release_pending:
                return False
            token.release_pending = True
            self._cv.notify_all()

        def finish_release(error: BaseException | None) -> None:
            with self._cv:
                if error is None:
                    if self._active_output_leases.get(key) is token:
                        self._active_output_leases.pop(key, None)
                else:
                    token.release_pending = False
                self._cv.notify_all()

        try:
            self._submit_cleanup_operations(
                (self._output_token_release_operation(token),),
                on_done=finish_release,
                slot_ids=(token.slot_id,),
            )
        except BaseException:
            with self._cv:
                token.release_pending = False
                self._cv.notify_all()
            raise
        return True

    def handoff_output_block_lease(self, request_id: str, lease_id: str) -> bool:
        """Move producer-side liveness ownership into the downstream pipeline.

        The token remains in ``_active_output_leases`` until the last C++
        descriptor owner releases it.  Handoff is deliberately a separate,
        idempotent transition: it must never drop the physical ObjectRef lease.
        """
        key = (str(request_id), str(lease_id))
        with self._cv:
            token = self._active_output_leases.get(key)
            if token is None or token.handed_off or token.handoff_pending:
                return False
            token.handoff_pending = True
            self._cv.notify_all()

        def finish_handoff(error: BaseException | None) -> None:
            with self._cv:
                token.handoff_pending = False
                if error is None and not token.release_pending and self._active_output_leases.get(key) is token:
                    token.handed_off = True
                self._cv.notify_all()

        try:
            self._submit_cleanup_operations(
                (self._output_token_handoff_operation(token),),
                on_done=finish_handoff,
                slot_ids=(token.slot_id,),
            )
        except BaseException:
            with self._cv:
                token.handoff_pending = False
                self._cv.notify_all()
            raise
        return True

    def _run_output_token_handoff(self, token: _OutputLeaseToken) -> bool:
        key = (token.request_id, token.lease_id)
        with self._cv:
            if token.release_pending or self._active_output_leases.get(key) is not token:
                return True
            response_ref = token.handoff_response_ref
        try:
            if response_ref is None:
                response_ref = token.driver.handoff_query_output_block_lease.remote(
                    token.request_id,
                    token.lease_id,
                    token.query_id,
                )
                with self._cv:
                    token.handoff_response_ref = response_ref
            resolve_ray_control_ack(response_ref, field="handed_off")
        except BaseException as exc:
            if not isinstance(exc, TimeoutError):
                with self._cv:
                    if token.handoff_response_ref is response_ref:
                        token.handoff_response_ref = None
            raise
        with self._cv:
            token.handoff_response_ref = None
        return True

    def _run_output_token_release(self, token: _OutputLeaseToken) -> bool:
        with self._cv:
            response_ref = token.release_response_ref
        try:
            if response_ref is None:
                response_ref = token.driver.release_query_output_block_lease.remote(
                    token.request_id,
                    token.lease_id,
                    token.query_id,
                )
                with self._cv:
                    token.release_response_ref = response_ref
            resolve_ray_control_ack(response_ref, field="released")
        except BaseException as exc:
            if not isinstance(exc, TimeoutError):
                with self._cv:
                    if token.release_response_ref is response_ref:
                        token.release_response_ref = None
            raise
        with self._cv:
            token.release_response_ref = None
        return True

    def _output_token_handoff_operation(
        self,
        token: _OutputLeaseToken,
    ) -> RayStreamCleanupOperation:
        return RayStreamCleanupOperation(
            RAY_STREAM_CLEANUP_OUTPUT,
            lambda: self._run_output_token_handoff(token),
            retry_on_error=True,
            retry_on_incomplete=True,
        )

    def _output_token_release_operation(
        self,
        token: _OutputLeaseToken,
    ) -> RayStreamCleanupOperation:
        return RayStreamCleanupOperation(
            RAY_STREAM_CLEANUP_OUTPUT,
            lambda: self._run_output_token_release(token),
            retry_on_error=True,
            retry_on_incomplete=True,
        )

    @staticmethod
    def _output_request_cancel_operation(
        driver: Any,
        request: dict[str, Any],
    ) -> RayStreamCleanupOperation:
        response_ref: Any | None = None

        def cancel() -> bool:
            nonlocal response_ref
            try:
                if response_ref is None:
                    response_ref = driver.cancel_query_output_block_lease_request.remote(request)
                resolve_ray_control_ack(response_ref, field="cancelled")
            except BaseException as exc:
                if not isinstance(exc, TimeoutError):
                    response_ref = None
                raise
            response_ref = None
            return True

        return RayStreamCleanupOperation(
            RAY_STREAM_CLEANUP_OUTPUT,
            cancel,
            retry_on_error=True,
            retry_on_incomplete=True,
        )

    def cancel_slot(self, slot_id: int) -> None:
        slot_key = int(slot_id)
        cleanup_groups: set[_CleanupGroup] = set()
        with self._cleanup_handoff_lock:
            with self._cv:
                if self._shutdown:
                    return
                cleanup_groups.update(self._cleanup_groups_by_slot.get(slot_key, ()))
                records = [record for record in self._records.values() if record.slot_id == slot_key]
                records_to_cancel = [record for record in records if not record.cleanup_started]
                cleanup_operations: list[Callable[[], Any]] = []
                for record in records:
                    self._records.pop((record.slot_id, record.submit_id), None)
                    record.terminal = True
                    if not record.registration_accepted:
                        record.registration_cancelled = True
                    if not record.cleanup_started:
                        record.cleanup_started = True
                    self._cancel_record_wait_locked(record)
                for record in records_to_cancel:
                    request = record.output_request
                    driver = record.adapter.driver
                    if request is not None and driver is not None and not record.output_cancel_sent:
                        record.output_cancel_sent = True
                        cleanup_operations.append(self._output_request_cancel_operation(driver, request))
                    cleanup_operations.extend(record.adapter.cancel_operations())
                ready = list(self._ready_by_slot.pop(slot_key, ()))
                tokens = [token for token in self._active_output_leases.values() if token.slot_id == slot_key]
                for token in tokens:
                    token.release_pending = True
                    self._active_output_leases.pop((token.request_id, token.lease_id), None)
                self._capacity_by_slot.pop(slot_key, None)
                self._cancelled_slots.add(slot_key)
                self._signal_readiness_change_locked()

            ready_tokens = [event.output_token for event in ready if event.output_token is not None]
            token_keys = {(token.request_id, token.lease_id) for token in tokens}
            for token in ready_tokens:
                if token is not None and (token.request_id, token.lease_id) not in token_keys:
                    tokens.append(token)
                    token_keys.add((token.request_id, token.lease_id))
            for token in tokens:
                token.release_pending = True
            cleanup_operations.extend(self._output_token_release_operation(token) for token in tokens)
            cleanup_group = self._submit_cleanup_operations(
                self._fence_cleanup_operations(cleanup_operations),
                store_error=False,
                slot_ids=(slot_key,),
            )
            if cleanup_group is not None:
                cleanup_groups.add(cleanup_group)

        cleanup_errors = self._wait_slot_cleanup(
            slot_key,
            timeout_message=f"Ray UDF slot {slot_key} cleanup did not terminate",
            initial_groups=tuple(cleanup_groups),
        )
        if cleanup_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            raise RuntimeError(f"Ray UDF slot {slot_key} cleanup failed: {details}") from cleanup_errors[0]

    def slot_has_pending(self, slot_id: int) -> bool:
        slot_key = int(slot_id)
        with self._cv:
            return (
                any(record.slot_id == slot_key for record in self._records.values())
                or bool(self._ready_by_slot.get(slot_key))
                or any(token.slot_id == slot_key for token in self._active_output_leases.values())
                or bool(self._cleanup_groups_by_slot.get(slot_key))
            )

    def set_wakeup_callback(self, fn: Any) -> None:
        with self._cv:
            self._wakeup_fn = fn

    def shutdown(self) -> None:
        deadline = time.monotonic() + self._shutdown_timeout_s
        shutdown_errors: list[BaseException] = []
        with self._start_lock:
            with self._cleanup_handoff_lock:
                with self._cv:
                    first_shutdown = not self._shutdown
                    cleanup_operations: list[Callable[[], Any]] = []
                    cleanup_slot_ids: set[int] = set()
                    if first_shutdown:
                        self._shutdown = True
                        self._wakeup_fn = None
                        records = list(self._records.values())
                        for record in records:
                            cleanup_slot_ids.add(record.slot_id)
                            self._cancel_record_wait_locked(record)
                            if not record.registration_accepted:
                                record.registration_cancelled = True
                            if record.cleanup_started:
                                continue
                            record.cleanup_started = True
                            request = record.output_request
                            driver = record.adapter.driver
                            if request is not None and driver is not None and not record.output_cancel_sent:
                                record.output_cancel_sent = True
                                cleanup_operations.append(self._output_request_cancel_operation(driver, request))
                            cleanup_operations.extend(record.adapter.cancel_operations())
                        self._records.clear()
                        ready = [event for events in self._ready_by_slot.values() for event in events]
                        tokens = list(self._active_output_leases.values())
                        token_keys = {(token.request_id, token.lease_id) for token in tokens}
                        for event in ready:
                            cleanup_slot_ids.add(event.slot_id)
                            token = event.output_token
                            if token is not None and (token.request_id, token.lease_id) not in token_keys:
                                tokens.append(token)
                                token_keys.add((token.request_id, token.lease_id))
                        self._active_output_leases.clear()
                        self._ready_by_slot.clear()
                        self._capacity_by_slot.clear()
                        for token in tokens:
                            token.release_pending = True
                        cleanup_operations.extend(self._output_token_release_operation(token) for token in tokens)
                        cleanup_slot_ids.update(token.slot_id for token in tokens)
                        self._signal_readiness_change_locked()
                    if self._thread_started and self._thread.is_alive():
                        try:
                            self._loop.call_soon_threadsafe(self._loop.stop)
                        except RuntimeError as exc:
                            if self._thread.is_alive():
                                shutdown_errors.append(exc)
                if first_shutdown and cleanup_operations:
                    try:
                        self._submit_cleanup_operations(
                            self._fence_cleanup_operations(cleanup_operations),
                            slot_ids=tuple(cleanup_slot_ids),
                        )
                    except BaseException as exc:
                        shutdown_errors.append(exc)

        if self._thread_started and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if self._thread.is_alive():
                shutdown_errors.append(RuntimeError("Ray UDF stream multiplexer did not terminate"))

        if not self._thread.is_alive():
            self._stop_cleanup_pools_when_idle()

        with self._cv:
            while True:
                pending = [group for group in self._cleanup_groups if not group.callback_owned_by_current_thread()]
                if not pending:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cv.wait(timeout=remaining)
            pending = [group for group in self._cleanup_groups if not group.callback_owned_by_current_thread()]
            if pending:
                shutdown_errors.append(RuntimeError("Ray UDF stream remote cleanup did not terminate"))
            shutdown_errors.extend(self._terminal_cleanup_errors)
        if not self._thread.is_alive():
            shutdown_errors.extend(self._join_cleanup_pools(deadline))
        if shutdown_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in shutdown_errors)
            raise RuntimeError(f"Ray UDF stream shutdown failed: {details}") from shutdown_errors[0]

    # Multiplexer internals.

    def _ensure_started(self) -> None:
        with self._start_lock:
            with self._cv:
                if self._started:
                    self._raise_if_stopped_locked()
                    return
                self._raise_if_stopped_locked()
                self._started = True
            thread_started = False
            try:
                self._thread.start()
                thread_started = True
                self._thread_started = True
                if not self._loop_ready.wait(timeout=self._shutdown_timeout_s):
                    raise RuntimeError("Ray UDF stream event loop did not start")
                with self._cv:
                    self._raise_if_stopped_locked()
                    self._signal_readiness_change_locked()
            except BaseException as exc:
                with self._cv:
                    if self._thread_error is None:
                        self._thread_error = exc
                    self._cv.notify_all()
                if not thread_started:
                    self._loop.close()
                raise

    def _raise_if_stopped_locked(self) -> None:
        if self._thread_error is not None:
            raise RuntimeError(f"Ray UDF stream multiplexer failed: {self._thread_error}") from self._thread_error
        if self._shutdown:
            raise RuntimeError("Ray UDF stream collector is shut down")

    @staticmethod
    def _parse_capacities(
        capacities: dict[Any, Any] | None,
    ) -> dict[int, _DrainCapacity]:
        if capacities is None:
            return {}
        parsed: dict[int, _DrainCapacity] = {}
        for raw_slot, raw in capacities.items():
            if not isinstance(raw, dict) or "rows" not in raw:
                raise ValueError(f"invalid Ray UDF drain capacity for slot {raw_slot!r}")
            rows = max(0, int(raw["rows"]))
            bytes_value = raw.get("bytes")
            item_value = raw.get("item_bytes")
            parsed[int(raw_slot)] = _DrainCapacity(
                rows=rows,
                bytes=None if bytes_value is None else max(0, int(bytes_value)),
                item_bytes=None if item_value is None else max(0, int(item_value)),
            )
        return parsed

    def _pending_data_counts_locked(self) -> dict[int, int]:
        """Count every admitted block once in a single linear ledger pass."""
        counts: dict[int, int] = defaultdict(int)
        for slot_id, ready in self._ready_by_slot.items():
            counts[slot_id] += sum(1 for event in ready if event.kind == "data")
        for record in self._records.values():
            if record.phase != "block" or (record.wait_kind == "data" and record.wait_future is not None):
                counts[record.slot_id] += 1
        return counts

    def _may_read_block_locked(
        self,
        record: _StreamRecord,
        *,
        pending_data_count: int,
    ) -> bool:
        if not record.next_ref_ready:
            return False
        capacity = self._capacity_by_slot.get(record.slot_id)
        if capacity is None or capacity.rows <= 0:
            return False
        if capacity.bytes is not None and capacity.bytes <= 0:
            return False
        if capacity.item_bytes is not None and capacity.item_bytes <= 0:
            return False
        return int(pending_data_count) < capacity.rows

    def _signal_readiness_change_locked(self) -> None:
        """Request an immediate scheduler pass without a remote wakeup RPC."""
        self._cv.notify_all()
        if (
            self._shutdown
            or self._thread_error is not None
            or not self._started
            or not self._loop_ready.is_set()
            or self._readiness_wakeup_pending
        ):
            return
        self._readiness_wakeup_pending = True
        try:
            self._loop.call_soon_threadsafe(self._wake_generator_readiness)
        except RuntimeError as exc:
            self._readiness_wakeup_pending = False
            if not self._shutdown:
                self._thread_error = exc
                self._cv.notify_all()

    def _readiness_waitables_locked(self) -> dict[Any, _StreamRecord]:
        waitables: dict[Any, _StreamRecord] = {}
        pending_data_counts = self._pending_data_counts_locked()
        admissible_slots = {
            slot_id
            for slot_id, capacity in self._capacity_by_slot.items()
            if capacity.rows > 0
            and (capacity.bytes is None or capacity.bytes > 0)
            and (capacity.item_bytes is None or capacity.item_bytes > 0)
            and pending_data_counts.get(slot_id, 0) < capacity.rows
        }
        for record in self._records.values():
            if (
                record.terminal
                or not record.registration_accepted
                or record.slot_id not in admissible_slots
                or record.phase != "block"
                or record.next_ref_ready
                or record.wait_future is not None
                or record.terminal_ref is not None
            ):
                continue
            waitable = record.adapter.waitable
            if waitable in waitables:
                raise RuntimeError("Ray UDF collector cannot observe one generator for multiple stream records")
            waitables[waitable] = record
        return waitables

    def _scheduler_stopped_locked(self) -> bool:
        return self._shutdown or self._thread_error is not None

    def _wake_generator_readiness(self) -> None:
        """Interrupt an idle backoff and run the central scheduler immediately."""
        try:
            with self._cv:
                self._readiness_wakeup_pending = False
                timer = self._readiness_timer
                self._readiness_timer = None
                if timer is not None:
                    timer.cancel()
                if self._shutdown or self._thread_error is not None:
                    return
                self._readiness_delay_s = _GENERATOR_READINESS_POLL_INITIAL_DELAY_S
            self._poll_generator_readiness()
        except BaseException as exc:
            self._report_scheduler_error(exc)

    def _poll_generator_readiness(self) -> None:
        """Run one Ray Data-style non-consuming readiness scheduling pass.

        The existing multiplexer event loop is the single owner of generator
        probes, reads, and terminal-state transitions.  A zero-time public
        ``ray.wait`` rebuilds the eligible set on every pass; local admission
        changes cancel the current idle timer, while exponential backoff bounds
        polling overhead for genuinely slow generators.
        """
        try:
            self._poll_generator_readiness_once()
        except BaseException as exc:
            self._report_scheduler_error(exc)

    def _poll_generator_readiness_once(self) -> None:
        with self._cv:
            self._readiness_timer = None
            if self._scheduler_stopped_locked():
                return
            waitables = self._readiness_waitables_locked()
        if not waitables:
            with self._cv:
                self._readiness_delay_s = _GENERATOR_READINESS_POLL_INITIAL_DELAY_S
            return

        try:
            ready, _ = self._ray.wait(
                list(waitables),
                num_returns=len(waitables),
                fetch_local=False,
                timeout=0,
            )
        except BaseException:
            # A single dead generator must not poison unrelated query slots.
            # Ray does not identify the failed waitable in a batch exception,
            # so isolate only on this exceptional path with zero-time probes.
            ready = []
            failed: list[tuple[_StreamRecord, BaseException]] = []
            for waitable, record in waitables.items():
                try:
                    one_ready, _ = self._ray.wait(
                        [waitable],
                        num_returns=1,
                        fetch_local=False,
                        timeout=0,
                    )
                except BaseException as exc:
                    failed.append((record, exc))
                else:
                    ready.extend(one_ready)
            if not failed or len(failed) == len(waitables):
                raise
            for failed_record, failure in failed:
                self._fail_record(failed_record, failure)

        ready_records: list[_StreamRecord] = []
        with self._cv:
            if self._scheduler_stopped_locked():
                return
            currently_waitable = self._readiness_waitables_locked()
            for waitable in ready:
                ready_record = waitables.get(waitable)
                if ready_record is None:
                    self._thread_error = RuntimeError("Ray returned an unknown generator readiness handle")
                    self._cv.notify_all()
                    break
                if currently_waitable.get(waitable) is ready_record:
                    ready_record.next_ref_ready = True
                    ready_record.ready_sequence = self._next_ready_sequence
                    self._next_ready_sequence += 1
                    ready_records.append(ready_record)
            if self._thread_error is not None:
                notify_error = True
            else:
                notify_error = False
                if ready_records:
                    self._readiness_delay_s = _GENERATOR_READINESS_POLL_INITIAL_DELAY_S
                    self._cv.notify_all()
                elif not self._readiness_wakeup_pending:
                    delay_s = self._readiness_delay_s
                    self._readiness_delay_s = min(
                        _GENERATOR_READINESS_POLL_MAX_DELAY_S,
                        delay_s * 2,
                    )
                    self._readiness_timer = self._loop.call_later(
                        delay_s,
                        self._poll_generator_readiness,
                    )

        if notify_error:
            self._notify_wakeup()
            return
        if ready_records:
            for record in ready_records:
                _collector_debug_log("generator_ready", record)
            self._refresh_waiters()

    def _report_scheduler_error(self, exc: BaseException) -> None:
        with self._cv:
            report = not self._shutdown and self._thread_error is None
            if report:
                self._thread_error = exc
            self._cv.notify_all()
        if report:
            self._notify_wakeup()

    def _run_scheduler_callback(
        self,
        callback: Callable[..., Any],
        *args: Any,
    ) -> None:
        """Turn every event-loop callback failure into visible collector state."""
        try:
            callback(*args)
        except BaseException as exc:
            self._report_scheduler_error(exc)

    def _submit_cleanup_operations(
        self,
        operations: Sequence[Callable[[], Any]],
        *,
        on_done: Callable[[BaseException | None], None] | None = None,
        store_error: bool = True,
        slot_ids: Sequence[int] = (),
    ) -> _CleanupGroup | None:
        """Atomically hand one terminal plan to bounded, isolated worker lanes."""
        actions = tuple(operations)
        if not actions:
            if on_done is not None:
                on_done(None)
            return None

        operations_by_pool: dict[_DaemonCleanupPool, list[Callable[[], Any]]] = defaultdict(list)
        for action in actions:
            lane = action.lane if isinstance(action, RayStreamCleanupOperation) else RAY_STREAM_CLEANUP_CONTROL
            operations_by_pool[self._cleanup_pools[lane]].append(action)
        pools = tuple(pool for pool in self._cleanup_pools.values() if pool in operations_by_pool)
        group_holder: list[_CleanupGroup] = []
        owner_slots = frozenset(int(slot_id) for slot_id in slot_ids)

        def complete(first_error: BaseException | None) -> None:
            callback_error: BaseException | None = None
            try:
                if on_done is not None:
                    on_done(first_error)
            except BaseException as exc:
                callback_error = exc
            finally:
                with self._cv:
                    if store_error:
                        if first_error is not None:
                            self._terminal_cleanup_errors.append(first_error)
                    if callback_error is not None:
                        self._terminal_cleanup_errors.append(callback_error)
                    self._cleanup_groups.discard(group_holder[0])
                    for slot_id in owner_slots:
                        groups = self._cleanup_groups_by_slot.get(slot_id)
                        if groups is not None:
                            groups.discard(group_holder[0])
                            if not groups:
                                self._cleanup_groups_by_slot.pop(slot_id, None)
                    self._cv.notify_all()

        with self._cleanup_handoff_lock:
            for pool in pools:
                pool._cv.acquire()
            try:
                for pool in pools:
                    if not pool._started or pool._stop_when_idle:
                        raise RuntimeError(f"Ray UDF cleanup pool {pool._name!r} is not accepting work")
                    pool._ensure_worker_locked()
                group = _CleanupGroup(len(actions), on_complete=complete)
                group_holder.append(group)
                with self._cv:
                    self._cleanup_groups.add(group)
                    for slot_id in owner_slots:
                        self._cleanup_groups_by_slot[slot_id].add(group)
                    self._cv.notify_all()
                for pool, pool_operations in operations_by_pool.items():
                    pool._enqueue_locked(pool_operations, group)
            finally:
                for pool in reversed(pools):
                    pool._cv.release()
        return group

    def _fence_cleanup_operations(
        self,
        operations: Sequence[Callable[[], Any]],
    ) -> tuple[RayStreamCleanupOperation, ...]:
        """Delay generator teardown until every earlier scheduler probe exits."""
        actions = tuple(operations)
        if not actions:
            return ()
        scheduler_safe = threading.Event()
        with self._cv:
            needs_turn = self._started and self._thread.is_alive() and self._thread is not threading.current_thread()
        if needs_turn:
            try:
                self._loop.call_soon_threadsafe(scheduler_safe.set)
            except RuntimeError:
                if not self._thread.is_alive():
                    scheduler_safe.set()
        else:
            scheduler_safe.set()

        fenced: list[RayStreamCleanupOperation] = []
        for action in actions:
            lane = action.lane if isinstance(action, RayStreamCleanupOperation) else RAY_STREAM_CLEANUP_CONTROL
            retry_on_error = action.retry_on_error if isinstance(action, RayStreamCleanupOperation) else False
            retry_on_incomplete = action.retry_on_incomplete if isinstance(action, RayStreamCleanupOperation) else False

            def run(action: Callable[[], Any] = action) -> Any:
                while not scheduler_safe.wait(timeout=0.1):
                    if not self._thread.is_alive():
                        break
                return action()

            fenced.append(
                RayStreamCleanupOperation(
                    lane,
                    run,
                    retry_on_error=retry_on_error,
                    retry_on_incomplete=retry_on_incomplete,
                )
            )
        return tuple(fenced)

    def _wait_cleanup_group(
        self,
        group: _CleanupGroup | None,
        *,
        timeout_message: str,
    ) -> None:
        if group is None:
            return
        if not group.wait(self._shutdown_timeout_s):
            raise RuntimeError(timeout_message)

    def _wait_slot_cleanup(
        self,
        slot_id: int,
        *,
        timeout_message: str,
        initial_groups: Sequence[_CleanupGroup] = (),
    ) -> tuple[BaseException, ...]:
        deadline = time.monotonic() + self._shutdown_timeout_s
        observed = set(initial_groups)
        while True:
            with self._cv:
                groups = tuple(self._cleanup_groups_by_slot.get(int(slot_id), ()))
            observed.update(groups)
            pending = [
                group for group in observed if not group.wait(0) and not group.callback_owned_by_current_thread()
            ]
            if not pending:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not pending[0].wait(remaining):
                raise RuntimeError(timeout_message)
        return tuple(error for group in observed for error in group.errors)

    def _stop_cleanup_pools_when_idle(self) -> None:
        with self._cleanup_handoff_lock:
            if self._cleanup_pools_stopping:
                return
            self._cleanup_pools_stopping = True
            for pool in self._cleanup_pools.values():
                pool.stop_when_idle()

    def _join_cleanup_pools(self, deadline: float) -> list[BaseException]:
        errors: list[BaseException] = []
        for lane, pool in self._cleanup_pools.items():
            if not pool.join(max(0.0, deadline - time.monotonic())):
                errors.append(RuntimeError(f"Ray UDF {lane} cleanup workers did not terminate"))
        return errors

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        try:
            self._loop.run_forever()
        except BaseException as exc:
            with self._cv:
                self._thread_error = exc
                self._cv.notify_all()
            self._notify_wakeup()
        finally:
            with self._cv:
                timer = self._readiness_timer
                self._readiness_timer = None
                self._readiness_wakeup_pending = False
            if timer is not None:
                timer.cancel()
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()
            with self._cv:
                shutting_down = self._shutdown
            if shutting_down:
                self._stop_cleanup_pools_when_idle()

    def _cancel_record_wait_locked(self, record: _StreamRecord) -> None:
        future = record.wait_future
        completion_future = record.completion_future
        record.wait_future = None
        record.completion_future = None
        record.wait_kind = ""
        if future is not None:
            future.cancel()
        if completion_future is not None:
            completion_future.cancel()

    @staticmethod
    def _object_ref_future(ref: Any) -> Any:
        future_factory = getattr(ref, "future", None)
        if not callable(future_factory):
            raise TypeError("Ray UDF control ObjectRef does not expose future()")
        future = future_factory()
        if not callable(getattr(future, "add_done_callback", None)):
            raise TypeError("Ray UDF control ObjectRef future does not support callbacks")
        return future

    def _schedule_record_wait_locked(
        self,
        record: _StreamRecord,
        *,
        pending_data_count: int,
    ) -> bool:
        if record.terminal or record.wait_future is not None:
            return False
        kind = ""
        future = None
        block_admitted = False
        if record.terminal_ref is not None:
            kind = "terminal"
            future = self._object_ref_future(record.terminal_ref)
        elif record.output_lease_ref is not None:
            kind = "output_lease"
            future = self._object_ref_future(record.output_lease_ref)
        elif record.metadata_ref is not None:
            kind = "metadata"
            future = self._object_ref_future(record.metadata_ref)
        elif record.phase == "metadata" or (
            record.phase == "block"
            and self._may_read_block_locked(
                record,
                pending_data_count=pending_data_count,
            )
        ):
            kind = "data"
            if record.phase == "block":
                capacity = self._capacity_by_slot.get(record.slot_id)
                if capacity is None:
                    raise RuntimeError("Ray UDF block read was scheduled without downstream capacity")
                # Consuming the block ObjectRef is the admission point for the
                # whole block/metadata pair. Keep its per-item limit stable;
                # downstream capacity may legitimately fall to zero before the
                # metadata ObjectRef becomes ready.
                record.block_item_capacity_bytes = capacity.item_bytes
                record.next_ref_ready = False
                record.ready_sequence = None
                block_admitted = True
                self._signal_readiness_change_locked()
            future = asyncio.run_coroutine_threadsafe(
                record.adapter.read_next_ref_async(),
                self._loop,
            )
        if future is None:
            return False
        record.wait_kind = kind
        record.wait_future = future

        def on_done(done: Any) -> None:
            with self._cv:
                if self._shutdown:
                    return
            try:
                self._loop.call_soon_threadsafe(
                    self._run_scheduler_callback,
                    self._complete_record_wait,
                    record,
                    kind,
                    done,
                )
            except RuntimeError as exc:
                with self._cv:
                    shutting_down = self._shutdown
                if not shutting_down:
                    self._report_scheduler_error(exc)

        future.add_done_callback(on_done)
        return block_admitted

    def _schedule_completion_wait_locked(self, record: _StreamRecord) -> None:
        if record.terminal or record.producer_completed or record.completion_future is not None:
            return
        future = self._object_ref_future(record.adapter.completion_ref)
        record.completion_future = future

        def on_done(done: Any) -> None:
            with self._cv:
                if self._shutdown:
                    return
            try:
                self._loop.call_soon_threadsafe(
                    self._run_scheduler_callback,
                    self._complete_producer_wait,
                    record,
                    done,
                )
            except RuntimeError as exc:
                with self._cv:
                    shutting_down = self._shutdown
                if not shutting_down:
                    self._report_scheduler_error(exc)

        future.add_done_callback(on_done)

    def _refresh_waiters(self) -> None:
        with self._cv:
            if self._shutdown or self._thread_error is not None or not self._started or not self._loop_ready.is_set():
                return
            records = sorted(
                (record for record in self._records.values() if record.registration_accepted),
                key=lambda record: (
                    record.phase == "block" and record.next_ref_ready,
                    record.ready_sequence if record.ready_sequence is not None else record.sequence,
                ),
            )
            pending_data_counts = self._pending_data_counts_locked()
            for record in records:
                self._schedule_completion_wait_locked(record)
                if self._schedule_record_wait_locked(
                    record,
                    pending_data_count=pending_data_counts.get(record.slot_id, 0),
                ):
                    pending_data_counts[record.slot_id] = pending_data_counts.get(record.slot_id, 0) + 1

    def _complete_producer_wait(self, record: _StreamRecord, future: Any) -> None:
        key = (record.slot_id, record.submit_id)
        with self._cv:
            if (
                self._shutdown
                or self._records.get(key) is not record
                or record.terminal
                or record.completion_future is not future
            ):
                return
            if future is None:
                return
        try:
            future.result()
            record.producer_completed = True
            if record.adapter.stream_finished():
                record.adapter.mark_drained()
            self._maybe_complete_record(record)
        except BaseException as exc:
            # A failed completion ObjectRef means Ray has already made the task
            # terminal.  Force-cancelling that task can race its normal reply.
            record.terminal_signal_observed = True
            with self._cv:
                shutting_down = self._shutdown
            if not shutting_down:
                self._fail_record(record, exc)
        finally:
            # Keep the completed future installed until its state transition is
            # fully applied. A concurrent dispatcher capacity refresh calls
            # _refresh_waiters(); exposing an empty slot earlier can register a
            # duplicate callback for the same completion ObjectRef.
            with self._cv:
                if record.completion_future is future:
                    record.completion_future = None
            self._refresh_waiters()

    def _complete_record_wait(self, record: _StreamRecord, kind: str, future: Any) -> None:
        key = (record.slot_id, record.submit_id)
        with self._cv:
            if (
                self._shutdown
                or self._records.get(key) is not record
                or record.terminal
                or record.wait_future is not future
            ):
                return
            if future is None:
                return
        try:
            value = future.result()
            _collector_debug_log(f"ready_{kind}", record)
            if kind == "data":
                self._accept_stream_ref(record, value)
            elif kind == "metadata":
                record.metadata_ref = None
                self._accept_metadata(record, value)
            elif kind == "output_lease":
                self._finish_output_lease(record, value)
                self._maybe_complete_record(record)
            elif kind == "terminal":
                record.terminal_ref = None
                self._finish_stream(record)
            else:
                raise RuntimeError(f"unknown Ray stream readiness kind {kind!r}")
        except StopAsyncIteration:
            try:
                self._finish_stream(record)
            except BaseException as exc:
                self._fail_record(record, exc)
        except BaseException as exc:
            if kind == "terminal":
                record.terminal_signal_observed = True
            with self._cv:
                shutting_down = self._shutdown
            if not shutting_down:
                self._fail_record(record, exc)
        finally:
            # wait_future is also the transition-in-progress fence. Clearing
            # it before block/metadata/output state is updated lets a capacity
            # refresh schedule a second waiter for the same ObjectRef. That
            # duplicate callback observes the next phase and corrupts the
            # strict block/metadata pairing.
            with self._cv:
                if record.wait_future is future:
                    record.wait_future = None
                    record.wait_kind = ""
                    self._signal_readiness_change_locked()
            self._refresh_waiters()

    def _accept_stream_ref(self, record: _StreamRecord, next_ref: Any) -> None:
        if record.adapter.is_terminal_ref(next_ref):
            record.terminal_ref = next_ref
            return
        if record.phase == "block":
            record.block_ref = next_ref
            record.phase = "metadata"
            return
        if record.phase != "metadata" or record.block_ref is None:
            raise RuntimeError("Ray UDF stream violated block/metadata pair ordering")
        record.metadata_ref = next_ref

    def _finish_stream(self, record: _StreamRecord) -> None:
        record.terminal_signal_observed = True
        if record.phase == "metadata" and record.block_ref is not None:
            raise RuntimeError(
                "Ray UDF generator terminated after a block without its metadata; "
                "the remote task failed or violated the block/metadata protocol"
            )
        record.adapter.mark_drained()
        record.producer_completed = True
        self._maybe_complete_record(record)

    def _accept_metadata(self, record: _StreamRecord, metadata: Any) -> None:
        if record.phase != "metadata" or record.block_ref is None:
            raise RuntimeError("Ray UDF stream metadata arrived without its block")
        if isinstance(metadata, dict) and metadata.get("event_kind") == "error":
            remote_error = validate_stream_error_metadata(metadata)
            self._validate_task_identity(record, remote_error)
            # Error pairs are the final two objects emitted by the task.  The
            # generator is already returning, even if Ray's completion reply
            # has not reached this worker yet.
            record.terminal_signal_observed = True
            raise RuntimeError(
                f"remote Ray UDF failed: {remote_error['exception_type']}: {remote_error['exception_message']}"
            )
        validated = validate_stream_block_metadata(metadata)
        self._validate_task_identity(record, validated)
        item_capacity_bytes = record.block_item_capacity_bytes
        if item_capacity_bytes is not None and int(validated["size_bytes"]) > item_capacity_bytes:
            raise RuntimeError(
                "Ray UDF block exceeds downstream item capacity: "
                f"query={validated['query_id']} "
                f"stage={validated['producer_stage_id']} "
                f"task_lease={validated['task_lease_id']} "
                f"block={validated['block_id']} "
                f"size_bytes={validated['size_bytes']} "
                f"item_capacity_bytes={item_capacity_bytes}"
            )
        driver = record.adapter.driver
        if driver is None:
            raise RuntimeError("Ray UDF stream has no query resource driver")
        request_id = f"output-request:{validated['block_id']}"
        request = {
            "request_id": request_id,
            "query_id": validated["query_id"],
            "producer_stage_id": validated["producer_stage_id"],
            "task_lease_id": validated["task_lease_id"],
            "attempt_id": validated["attempt_id"],
            "block_id": validated["block_id"],
            "size_bytes": validated["size_bytes"],
        }
        # Metadata is already consumed at this point.  Keep output admission as
        # its own state so producer completion cannot mistake a drained stream
        # for a block whose metadata never arrived.
        key = (record.slot_id, record.submit_id)
        with self._cv:
            if self._shutdown or self._records.get(key) is not record or record.terminal:
                return
            record.metadata = validated
            record.output_request_id = request_id
            record.output_request = request
            record.output_cancel_sent = False
            record.phase = "output_lease"

        output_lease_ref = driver.acquire_query_output_block_lease.remote(request)
        with self._cv:
            stale = self._shutdown or self._records.get(key) is not record or record.terminal
            if not stale:
                record.output_lease_ref = output_lease_ref
            cancel_stale_request = stale and not record.output_cancel_sent
            if cancel_stale_request:
                record.output_cancel_sent = True
        if stale:
            if cancel_stale_request:
                self._submit_cleanup_operations(
                    (self._output_request_cancel_operation(driver, request),),
                    slot_ids=(record.slot_id,),
                )
            return
        _collector_debug_log("output_lease_requested", record)

    @staticmethod
    def _validate_task_identity(record: _StreamRecord, metadata: dict[str, Any]) -> None:
        lease = record.adapter.task_lease
        if lease is None:
            raise RuntimeError("Ray UDF stream metadata arrived before task lease admission")
        expected = {
            "query_id": str(lease["query_id"]),
            "producer_stage_id": str(lease["stage_id"]),
            "task_lease_id": str(lease["lease_id"]),
            "attempt_id": str(lease["attempt_id"]),
        }
        mismatched = [name for name, value in expected.items() if metadata.get(name) != value]
        if mismatched:
            raise RuntimeError("stale or cross-task Ray UDF stream metadata: " + ", ".join(mismatched))

    def _finish_output_lease(self, record: _StreamRecord, grant: Any) -> None:
        if not isinstance(grant, dict) or not grant.get("granted"):
            reason = grant.get("blocked_reason") if isinstance(grant, dict) else "invalid_grant"
            raise RuntimeError(f"Ray UDF output block lease denied: {reason}")
        lease = grant.get("lease")
        if not isinstance(lease, dict) or not str(lease.get("lease_id") or ""):
            raise RuntimeError("Ray UDF output block lease grant is missing lease identity")
        metadata = record.metadata
        block_ref = record.block_ref
        if metadata is None or block_ref is None:
            raise RuntimeError("Ray UDF output lease completed without its block pair")
        if str(lease.get("block_id") or "") != metadata["block_id"]:
            raise RuntimeError("Ray UDF output lease block identity mismatch")
        driver = record.adapter.driver
        assert driver is not None
        token = _OutputLeaseToken(
            request_id=record.output_request_id,
            lease_id=str(lease["lease_id"]),
            query_id=str(metadata["query_id"]),
            driver=driver,
            slot_id=record.slot_id,
            submit_id=record.submit_id,
            size_bytes=int(metadata["size_bytes"]),
        )
        descriptor = {
            "query_id": metadata["query_id"],
            "producer_stage_id": metadata["producer_stage_id"],
            "task_lease_id": metadata["task_lease_id"],
            "attempt_id": metadata["attempt_id"],
            "block_id": metadata["block_id"],
            "output_block_lease_id": token.lease_id,
            "num_rows": int(metadata["num_rows"]),
            "size_bytes": int(metadata["size_bytes"]),
        }
        payload = (
            REF_BUNDLE_RESULT_MARKER,
            [block_ref],
            [descriptor],
            list(metadata["names"]),
        )
        event = _ReadyEvent(
            slot_id=record.slot_id,
            submit_id=record.submit_id,
            kind="data",
            payload=payload,
            size_bytes=token.size_bytes,
            output_token=token,
        )
        with self._cv:
            stale = self._records.get((record.slot_id, record.submit_id)) is not record
            if not stale:
                self._active_output_leases[(token.request_id, token.lease_id)] = token
                self._ready_by_slot[record.slot_id].append(event)
                record.phase = "block"
                record.block_ref = None
                record.metadata = None
                record.block_item_capacity_bytes = None
                record.output_request_id = ""
                record.output_request = None
                record.output_lease_ref = None
                record.output_cancel_sent = False
                self._signal_readiness_change_locked()
        if stale:
            token.release_pending = True
            self._submit_cleanup_operations(
                (self._output_token_release_operation(token),),
                slot_ids=(record.slot_id,),
            )
            return
        _collector_debug_log("output_lease_granted", record)
        self._notify_wakeup()

    def _maybe_complete_record(self, record: _StreamRecord) -> None:
        if not record.producer_completed:
            return
        if record.phase != "block" or record.output_lease_ref is not None:
            return
        if not record.adapter.stream_finished():
            return
        record.adapter.mark_drained()
        key = (record.slot_id, record.submit_id)

        def finish_retirement(error: BaseException | None) -> None:
            dropped_tokens: list[_OutputLeaseToken] = []
            with self._cleanup_handoff_lock:
                with self._cv:
                    if self._records.pop(key, None) is not record:
                        if error is not None:
                            self._terminal_cleanup_errors.append(error)
                            self._cv.notify_all()
                        return
                    if error is None:
                        event = _ReadyEvent(record.slot_id, record.submit_id, "complete", None)
                    else:
                        ready = self._ready_by_slot.get(record.slot_id)
                        if ready is not None:
                            kept: deque[_ReadyEvent] = deque()
                            for queued in ready:
                                if queued.submit_id == record.submit_id and queued.kind == "data":
                                    if queued.output_token is not None:
                                        dropped_tokens.append(queued.output_token)
                                    continue
                                kept.append(queued)
                            self._ready_by_slot[record.slot_id] = kept
                        for token in dropped_tokens:
                            token.release_pending = True
                            self._active_output_leases.pop((token.request_id, token.lease_id), None)
                        event = _ReadyEvent(
                            record.slot_id,
                            record.submit_id,
                            "error",
                            f"{type(error).__name__}: {error}",
                        )
                    self._ready_by_slot[record.slot_id].append(event)
                    self._cv.notify_all()
                if dropped_tokens:
                    self._submit_cleanup_operations(
                        tuple(self._output_token_release_operation(token) for token in dropped_tokens),
                        slot_ids=(record.slot_id,),
                    )
            _collector_debug_log("retired" if error is None else "retire_failed", record)
            self._notify_wakeup()

        with self._cleanup_handoff_lock:
            with self._cv:
                if self._records.get(key) is not record:
                    return
                record.terminal = True
                record.cleanup_started = True
                self._signal_readiness_change_locked()
                cleanup_operations = record.adapter.retire_operations()
            self._submit_cleanup_operations(
                cleanup_operations,
                on_done=finish_retirement,
                store_error=False,
                slot_ids=(record.slot_id,),
            )

    def _fail_record(self, record: _StreamRecord, exc: BaseException) -> None:
        exc = format_stateful_actor_loss(record.error_context, exc)
        key = (record.slot_id, record.submit_id)
        with self._cleanup_handoff_lock:
            with self._cv:
                if self._records.pop(key, None) is not record:
                    return
                record.terminal = True
                record.cleanup_started = True
                ready = self._ready_by_slot.get(record.slot_id)
                dropped_tokens: list[_OutputLeaseToken] = []
                if ready is not None:
                    kept: deque[_ReadyEvent] = deque()
                    for event in ready:
                        if event.submit_id == record.submit_id and event.kind == "data":
                            if event.output_token is not None:
                                dropped_tokens.append(event.output_token)
                            continue
                        kept.append(event)
                    self._ready_by_slot[record.slot_id] = kept
                for token in dropped_tokens:
                    token.release_pending = True
                    self._active_output_leases.pop((token.request_id, token.lease_id), None)
                self._ready_by_slot[record.slot_id].append(
                    _ReadyEvent(
                        record.slot_id,
                        record.submit_id,
                        "error",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                request = record.output_request
                driver = record.adapter.driver
                record.output_cancel_sent = request is not None and driver is not None
                terminal_signal_observed = record.terminal_signal_observed
                cleanup_operations: list[Callable[[], Any]] = [
                    self._output_token_release_operation(token) for token in dropped_tokens
                ]
                if request is not None and driver is not None:
                    cleanup_operations.append(self._output_request_cancel_operation(driver, request))
                if terminal_signal_observed:
                    cleanup_operations.extend(record.adapter.retire_failed_operations())
                else:
                    cleanup_operations.extend(record.adapter.cancel_operations())
                self._signal_readiness_change_locked()
            self._submit_cleanup_operations(
                cleanup_operations,
                slot_ids=(record.slot_id,),
            )

        # The entire terminal plan has a durable bounded-lane owner before an
        # arbitrary dispatcher callback can reenter shutdown.
        self._notify_wakeup()

    def _notify_wakeup(self) -> None:
        with self._cv:
            callback = self._wakeup_fn
        if callback is not None:
            try:
                callback()
            except BaseException as exc:
                with self._cv:
                    if not self._shutdown and self._thread_error is None:
                        self._thread_error = RuntimeError(
                            f"Ray UDF stream wakeup callback failed: {type(exc).__name__}: {exc}"
                        )
                    self._cv.notify_all()


__all__ = ["AsyncResultCollector"]
