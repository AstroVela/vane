# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import math
import os
import sys
import threading
import time
import weakref
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from duckdb.execution.ray_control_deadline import (
    RAY_CONTROL_DEADLINE_SCHEDULER,
    RayControlScheduledCall,
)
from duckdb.execution.ray_control_submission import (
    create_ray_control_deadline,
    dispatch_ray_control_abandonment,
    submit_ray_control,
)
from duckdb.execution.ray_stream_adapter import (
    RayStreamAdapter,
    RayStreamCleanupOperation,
    RayStreamCleanupWait,
    TaskLeaseObjectRefGenerator,
    ray_object_ref_future,
    validate_ray_control_ack,
)
from duckdb.execution.udf_ray_actor_state import format_stateful_actor_loss
from duckdb.execution.udf_ray_config import REF_BUNDLE_RESULT_MARKER
from duckdb.execution.udf_ray_stream_protocol import (
    validate_stream_block_metadata,
    validate_stream_error_metadata,
)

_GENERATOR_READINESS_POLL_INITIAL_DELAY_S = 0.001
_GENERATOR_READINESS_POLL_MAX_DELAY_S = 0.01
_CLEANUP_RETRY_INITIAL_DELAY_S = 0.01
_CLEANUP_RETRY_MAX_DELAY_S = 1.0
_CLEANUP_RESPONSE_TIMEOUT_S = 1.0
_CLEANUP_RESPONSE_TIMEOUT_MAX_S = 30.0


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
    submission_scope: str
    slot_id: int
    submit_id: int
    size_bytes: int
    handed_off: bool = False
    handoff_pending: bool = False
    release_pending: bool = False


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
    registered: bool = False
    phase: str = "block"
    block_ref: Any | None = None
    metadata_ref: Any | None = None
    terminal_ref: Any | None = None
    metadata: dict[str, Any] | None = None
    block_item_capacity_bytes: int | None = None
    output_request_id: str = ""
    output_request: dict[str, Any] | None = None
    output_submit_future: Any | None = None
    output_lease_ref: Any | None = None
    output_cancel_sent: bool = False
    producer_completed: bool = False
    terminal: bool = False
    error_context: dict[str, Any] | None = None
    wait_kind: str = ""
    wait_future: Any | None = None
    completion_future: Any | None = None
    detached_waits: tuple[Any, ...] = ()
    terminal_signal_observed: bool = False
    next_ref_ready: bool = False
    ready_sequence: int | None = None
    cleanup_started: bool = False
    cleanup_remaining: int = 0
    cleanup_error: BaseException | None = None


class _CleanupTicket:
    """Durable owner for one independently driven terminal operation."""

    def __init__(
        self,
        *,
        operation: RayStreamCleanupOperation,
        retry_on_error: bool,
        retry_on_incomplete: bool,
        on_complete: Callable[[BaseException | None], None],
    ) -> None:
        self.operation = operation
        self.retry_on_error = bool(retry_on_error)
        self.retry_on_incomplete = bool(retry_on_incomplete)
        self.retry_count = 0
        self.submitting = False
        self.operation_future: Any | None = None
        self.wait_future: Any | None = None
        self.wait_response: RayStreamCleanupWait | None = None
        self.retry_call: RayControlScheduledCall | None = None
        self.wait_timeout: RayControlScheduledCall | None = None
        self._done = False
        self._error: BaseException | None = None
        self._on_complete = on_complete
        self._callback_thread: threading.Thread | None = None
        self._callback_complete = False
        self._cv = threading.Condition()

    def finish(self, error: BaseException | None) -> bool:
        callback: Callable[[BaseException | None], None] | None = None
        with self._cv:
            if self._done:
                return False
            self._done = True
            self._error = error
            callback = self._on_complete
            self._on_complete = lambda _error: None
            self._callback_thread = threading.current_thread()
            self._cv.notify_all()
        try:
            callback(error)
        finally:
            with self._cv:
                self._callback_thread = None
                self._callback_complete = True
                self._cv.notify_all()
        return True

    def wait(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        current = threading.current_thread()
        with self._cv:
            while not self._done or (not self._callback_complete and self._callback_thread is not current):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(timeout=remaining)
            return True

    def callback_owned_by_current_thread(self) -> bool:
        with self._cv:
            return self._done and self._callback_thread is threading.current_thread()

    @property
    def errors(self) -> tuple[BaseException, ...]:
        with self._cv:
            return () if self._error is None else (self._error,)


class UDFStreamResultCollector:
    """Capacity-aware scheduler for lease-owned Ray generator streams.

    ``_cv`` owns every published stream transition. Registration first starts
    the scheduler, then publishes a complete record in one critical section.
    A terminal record is removed only while ``_cleanup_handoff_lock`` prevents
    shutdown from observing an ownership gap.

    The asyncio thread is the sole generator-readiness, read, and stream-state
    scheduler. Potentially blocking Ray control submissions and terminal
    operations run on admitted stream-owner daemon workers. Cleanup tickets use
    the process-wide control deadline clock, which hands state transitions and
    potentially blocking completion/abandonment work to separate bounded,
    recoverable domains. Loop shutdown and one blocked Future callback therefore
    cannot orphan unrelated retries or response timeouts.
    Driver RPCs are observed through public ObjectRef futures. Every terminal
    operation enters the ticket ledger before its source record is removed, so
    shutdown can never observe an ownership gap.

    The native dispatcher keeps this collector alive across empty slot
    intervals. ``shutdown`` is the process-owner fence and is called only after
    that dispatcher thread has stopped accepting and executing submissions.
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
        # Cleanup ownership can outlive the asyncio loop, so its process-wide
        # deadline owner must exist before this collector accepts any stream.
        RAY_CONTROL_DEADLINE_SCHEDULER.ensure_started()
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
        self._cleanup_tickets: set[_CleanupTicket] = set()
        self._cleanup_tickets_by_slot: dict[int, set[_CleanupTicket]] = defaultdict(set)
        self._slot_cancellation_tickets: dict[int, set[_CleanupTicket]] = defaultdict(set)
        self._terminal_cleanup_errors: list[BaseException] = []
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
        # Start before any submitted generator can be transferred into this
        # collector.  A collector that cannot own its event loop therefore
        # fails construction instead of creating a post-submit cleanup gap.
        self._ensure_started()

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
        detached_waits: tuple[Any, ...] = ()
        with self._start_lock:
            try:
                adapter = RayStreamAdapter(source, ray_module=self._ray)
                source = None
                self._ensure_started()
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
                    )
                    self._next_sequence += 1
                    self._records[key] = record
                    # Publish and install the first terminal observer under
                    # one lock. Readiness skips unregistered records, so a
                    # failing Future factory can never expose a half-owned
                    # stream to the scheduler.
                    self._schedule_completion_wait_locked(record)
                    record.registered = True
                    self._signal_readiness_change_locked()
                _collector_debug_log("track", record)
            except BaseException:
                with self._cleanup_handoff_lock:
                    cleanup_operations: Sequence[RayStreamCleanupOperation] = ()
                    with self._cv:
                        if record is not None and self._records.get(key) is record:
                            record.terminal = True
                            record.cleanup_started = True
                            self._detach_record_waits_locked(record)
                            cleanup_operations = record.adapter.cancel_operations()
                            self._signal_readiness_change_locked()
                        elif adapter is not None:
                            cleanup_operations = adapter.cancel_operations()
                        elif source is not None:
                            cleanup_operations = source.cancel_operations()
                            source.retire_local_state()
                    if cleanup_operations:
                        # Registration runs on the one native dispatcher. The
                        # durable ticket ledger owns remote cleanup from this
                        # point; waiting for its ACK here would stall every
                        # otherwise healthy slot.
                        self._submit_cleanup_operations(
                            cleanup_operations,
                            slot_ids=(key[0],),
                        )
                    with self._cv:
                        if record is not None and self._records.get(key) is record:
                            detached_waits = record.detached_waits
                            record.detached_waits = ()
                            self._records.pop(key, None)
                            self._signal_readiness_change_locked()
                self._abandon_record_waits(
                    detached_waits,
                    owner_scope=f"collector:{id(self):x}:track:{key[0]}:{key[1]}",
                )
                raise

    def discard_generator_ref(self, slot_id: int, source: Any) -> None:
        """Cancel a submitted stream rejected by the C++ slot-generation fence."""
        if not isinstance(source, TaskLeaseObjectRefGenerator):
            raise TypeError(
                f"distributed Ray UDF discard requires TaskLeaseObjectRefGenerator; got {type(source).__name__}"
            )
        slot_key = int(slot_id)
        with self._start_lock:
            try:
                adapter = RayStreamAdapter(source, ray_module=self._ray)
            except BaseException:
                cleanup_operations = source.cancel_operations()
                source.retire_local_state()
            else:
                cleanup_operations = adapter.cancel_operations()
            with self._cv:
                if self._shutdown:
                    raise RuntimeError("Ray UDF stream collector is shut down")
            with self._cleanup_handoff_lock:
                self._submit_cleanup_operations(
                    cleanup_operations,
                    store_error=True,
                    slot_ids=(slot_key,),
                )

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
        with self._cleanup_handoff_lock:
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
                    on_each_done=finish_release,
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
        with self._cleanup_handoff_lock:
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
                    on_each_done=finish_handoff,
                    slot_ids=(token.slot_id,),
                )
            except BaseException:
                with self._cv:
                    token.handoff_pending = False
                    self._cv.notify_all()
                raise
        return True

    def _run_output_token_handoff(
        self,
        token: _OutputLeaseToken,
    ) -> bool | RayStreamCleanupWait:
        key = (token.request_id, token.lease_id)
        with self._cv:
            if token.release_pending or self._active_output_leases.get(key) is not token:
                return True
        response_ref = token.driver.handoff_query_output_block_lease.remote(
            token.request_id,
            token.lease_id,
            token.query_id,
        )
        response_future = ray_object_ref_future(response_ref)
        return RayStreamCleanupWait(
            response_future,
            lambda response: validate_ray_control_ack(
                response,
                field="handed_off",
            ),
        )

    def _run_output_token_release(
        self,
        token: _OutputLeaseToken,
    ) -> RayStreamCleanupWait:
        response_ref = token.driver.release_query_output_block_lease.remote(
            token.request_id,
            token.lease_id,
            token.query_id,
        )
        response_future = ray_object_ref_future(response_ref)
        return RayStreamCleanupWait(
            response_future,
            lambda response: validate_ray_control_ack(
                response,
                field="released",
            ),
        )

    @staticmethod
    def _claim_output_token_releases_locked(
        tokens: Sequence[_OutputLeaseToken],
    ) -> tuple[_OutputLeaseToken, ...]:
        """Claim each physical output lease once before its ticket handoff."""
        claimed: list[_OutputLeaseToken] = []
        seen: set[tuple[str, str]] = set()
        for token in tokens:
            key = (token.request_id, token.lease_id)
            if key in seen:
                continue
            seen.add(key)
            if token.release_pending:
                continue
            token.release_pending = True
            claimed.append(token)
        return tuple(claimed)

    def _restore_output_token_release_claims(
        self,
        tokens: Sequence[_OutputLeaseToken],
    ) -> None:
        with self._cv:
            for token in tokens:
                token.release_pending = False
            self._cv.notify_all()

    def _output_token_handoff_operation(
        self,
        token: _OutputLeaseToken,
    ) -> RayStreamCleanupOperation:
        return RayStreamCleanupOperation(
            lambda: self._run_output_token_handoff(token),
            submission_scope=token.submission_scope,
            retry_on_error=True,
            retry_on_incomplete=True,
        )

    def _output_token_release_operation(
        self,
        token: _OutputLeaseToken,
    ) -> RayStreamCleanupOperation:
        return RayStreamCleanupOperation(
            lambda: self._run_output_token_release(token),
            submission_scope=token.submission_scope,
            retry_on_error=True,
            retry_on_incomplete=True,
        )

    @staticmethod
    def _output_request_cancel_operation(
        driver: Any,
        request: dict[str, Any],
        submission_scope: str,
    ) -> RayStreamCleanupOperation:
        def cancel() -> RayStreamCleanupWait:
            response_ref = driver.cancel_query_output_block_lease_request.remote(request)
            response_future = ray_object_ref_future(response_ref)
            return RayStreamCleanupWait(
                response_future,
                lambda response: validate_ray_control_ack(
                    response,
                    field="cancelled",
                ),
            )

        return RayStreamCleanupOperation(
            cancel,
            submission_scope=submission_scope,
            retry_on_error=True,
            retry_on_incomplete=True,
        )

    def cancel_slot(self, slot_id: int) -> None:
        """Fence a slot and atomically hand its remote cleanup to the event loop."""
        slot_key = int(slot_id)
        detached_waits: list[Any] = []
        with self._cleanup_handoff_lock:
            with self._cv:
                if self._shutdown:
                    return
                self._slot_cancellation_tickets[slot_key].update(self._cleanup_tickets_by_slot.get(slot_key, ()))
                records = [record for record in self._records.values() if record.slot_id == slot_key]
                records_to_cancel = [record for record in records if not record.cleanup_started]
                cleanup_operations: list[RayStreamCleanupOperation] = []
                output_cancel_records: list[_StreamRecord] = []
                for record in records:
                    record.terminal = True
                    if not record.cleanup_started:
                        record.cleanup_started = True
                    self._detach_record_waits_locked(record)
                for record in records_to_cancel:
                    request = record.output_request
                    driver = record.adapter.driver
                    if request is not None and driver is not None and not record.output_cancel_sent:
                        record.output_cancel_sent = True
                        output_cancel_records.append(record)
                        cleanup_operations.append(
                            self._output_request_cancel_operation(
                                driver,
                                request,
                                record.adapter.submission_scope,
                            )
                        )
                    cleanup_operations.extend(record.adapter.cancel_operations())
                ready = list(self._ready_by_slot.get(slot_key, ()))
                tokens = [token for token in self._active_output_leases.values() if token.slot_id == slot_key]
                token_keys = {(token.request_id, token.lease_id) for token in tokens}
                for event in ready:
                    token = event.output_token
                    if token is not None and (token.request_id, token.lease_id) not in token_keys:
                        tokens.append(token)
                        token_keys.add((token.request_id, token.lease_id))
                claimed_tokens = self._claim_output_token_releases_locked(tokens)
                cleanup_operations.extend(self._output_token_release_operation(token) for token in claimed_tokens)
                self._signal_readiness_change_locked()

            try:
                cleanup_tickets = self._submit_cleanup_operations(
                    cleanup_operations,
                    store_error=False,
                    slot_ids=(slot_key,),
                )
            except BaseException:
                self._restore_output_token_release_claims(claimed_tokens)
                with self._cv:
                    for record in records_to_cancel:
                        if self._records.get((record.slot_id, record.submit_id)) is record:
                            record.cleanup_started = False
                    for record in output_cancel_records:
                        if self._records.get((record.slot_id, record.submit_id)) is record:
                            record.output_cancel_sent = False
                    self._cv.notify_all()
                raise
            with self._cv:
                self._slot_cancellation_tickets[slot_key].update(cleanup_tickets)
                for record in records:
                    if self._records.get((record.slot_id, record.submit_id)) is record:
                        detached_waits.extend(record.detached_waits)
                        record.detached_waits = ()
                        self._records.pop((record.slot_id, record.submit_id), None)
                self._ready_by_slot.pop(slot_key, None)
                for token in tokens:
                    self._active_output_leases.pop((token.request_id, token.lease_id), None)
                self._capacity_by_slot.pop(slot_key, None)
                self._cancelled_slots.add(slot_key)
                self._signal_readiness_change_locked()
        self._abandon_record_waits(
            detached_waits,
            owner_scope=f"collector:{id(self):x}:cancel-slot:{slot_key}",
        )

    def slot_has_pending(self, slot_id: int) -> bool:
        slot_key = int(slot_id)
        with self._cv:
            return (
                any(record.slot_id == slot_key for record in self._records.values())
                or bool(self._ready_by_slot.get(slot_key))
                or any(token.slot_id == slot_key for token in self._active_output_leases.values())
                or bool(self._cleanup_tickets_by_slot.get(slot_key))
            )

    def retire_slot(self, slot_id: int) -> str | None:
        """Drop a quiescent slot fence and return any terminal cleanup failure."""
        slot_key = int(slot_id)
        with self._cv:
            if (
                any(record.slot_id == slot_key for record in self._records.values())
                or bool(self._ready_by_slot.get(slot_key))
                or any(token.slot_id == slot_key for token in self._active_output_leases.values())
                or bool(self._cleanup_tickets_by_slot.get(slot_key))
            ):
                raise RuntimeError(f"Ray UDF slot {slot_key} still owns collector state")
            cancellation_tickets = self._slot_cancellation_tickets.pop(slot_key, ())
            self._cancelled_slots.discard(slot_key)
            self._capacity_by_slot.pop(slot_key, None)
            self._cv.notify_all()
        cleanup_errors = tuple(error for ticket in cancellation_tickets for error in ticket.errors)
        if not cleanup_errors:
            return None
        return "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)

    def set_wakeup_callback(self, fn: Any) -> None:
        with self._cv:
            self._wakeup_fn = fn

    def fatal_error(self) -> BaseException | None:
        """Return the process-owner-visible scheduler failure, if any."""
        with self._cv:
            return self._thread_error

    def shutdown(self) -> None:
        """Stop the process-owned loop after the native submitter is quiescent."""
        deadline = time.monotonic() + self._shutdown_timeout_s
        shutdown_errors: list[BaseException] = []
        detached_waits: list[Any] = []
        with self._start_lock:
            with self._cleanup_handoff_lock:
                with self._cv:
                    self._shutdown = True
                    self._wakeup_fn = None
                    cleanup_operations: list[RayStreamCleanupOperation] = []
                    cleanup_slot_ids: set[int] = set()
                    claimed_records: list[_StreamRecord] = []
                    output_cancel_records: list[_StreamRecord] = []
                    records = list(self._records.values())
                    for record in records:
                        cleanup_slot_ids.add(record.slot_id)
                        record.terminal = True
                        self._detach_record_waits_locked(record)
                        if record.cleanup_started:
                            continue
                        record.cleanup_started = True
                        claimed_records.append(record)
                        request = record.output_request
                        driver = record.adapter.driver
                        if request is not None and driver is not None and not record.output_cancel_sent:
                            record.output_cancel_sent = True
                            output_cancel_records.append(record)
                            cleanup_operations.append(
                                self._output_request_cancel_operation(
                                    driver,
                                    request,
                                    record.adapter.submission_scope,
                                )
                            )
                        cleanup_operations.extend(record.adapter.cancel_operations())
                    ready = [event for events in self._ready_by_slot.values() for event in events]
                    tokens = list(self._active_output_leases.values())
                    token_keys = {(token.request_id, token.lease_id) for token in tokens}
                    for event in ready:
                        cleanup_slot_ids.add(event.slot_id)
                        token = event.output_token
                        if token is not None and (token.request_id, token.lease_id) not in token_keys:
                            tokens.append(token)
                            token_keys.add((token.request_id, token.lease_id))
                    claimed_tokens = self._claim_output_token_releases_locked(tokens)
                    cleanup_operations.extend(self._output_token_release_operation(token) for token in claimed_tokens)
                    cleanup_slot_ids.update(token.slot_id for token in tokens)
                    self._signal_readiness_change_locked()
                if cleanup_operations:
                    try:
                        self._submit_cleanup_operations(
                            cleanup_operations,
                            slot_ids=tuple(cleanup_slot_ids),
                        )
                    except BaseException as exc:
                        self._restore_output_token_release_claims(claimed_tokens)
                        with self._cv:
                            for record in claimed_records:
                                if self._records.get((record.slot_id, record.submit_id)) is record:
                                    record.cleanup_started = False
                            for record in output_cancel_records:
                                if self._records.get((record.slot_id, record.submit_id)) is record:
                                    record.output_cancel_sent = False
                            self._cv.notify_all()
                        shutdown_errors.append(exc)
                if not shutdown_errors:
                    with self._cv:
                        for record in records:
                            detached_waits.extend(record.detached_waits)
                            record.detached_waits = ()
                        self._records.clear()
                        self._active_output_leases.clear()
                        self._ready_by_slot.clear()
                        self._capacity_by_slot.clear()
                        stop_now = not self._cleanup_tickets
                        self._cv.notify_all()
                    if stop_now:
                        self._request_event_loop_stop()

        if not shutdown_errors:
            self._abandon_record_waits(
                detached_waits,
                owner_scope=f"collector:{id(self):x}:shutdown",
            )

        if self._thread is threading.current_thread():
            if shutdown_errors:
                details = "; ".join(f"{type(error).__name__}: {error}" for error in shutdown_errors)
                raise RuntimeError(f"Ray UDF stream shutdown failed: {details}") from shutdown_errors[0]
            return

        with self._cv:
            while True:
                pending = [ticket for ticket in self._cleanup_tickets if not ticket.callback_owned_by_current_thread()]
                if not pending:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cv.wait(timeout=remaining)
            pending = [ticket for ticket in self._cleanup_tickets if not ticket.callback_owned_by_current_thread()]
            if pending:
                shutdown_errors.append(RuntimeError("Ray UDF stream remote cleanup did not terminate"))
            recorded_errors = [
                *self._terminal_cleanup_errors,
                *(
                    error
                    for tickets in self._slot_cancellation_tickets.values()
                    for ticket in tickets
                    for error in ticket.errors
                ),
            ]
            known_error_ids = {id(error) for error in shutdown_errors}
            for error in recorded_errors:
                if id(error) not in known_error_ids:
                    known_error_ids.add(id(error))
                    shutdown_errors.append(error)
            owns_stream_state = bool(self._records or self._active_output_leases or self._ready_by_slot)

        if not pending and not owns_stream_state:
            self._request_event_loop_stop()
        if (
            self._thread_started
            and self._thread is not threading.current_thread()
            and not pending
            and not owns_stream_state
        ):
            self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if self._thread.is_alive():
                shutdown_errors.append(RuntimeError("Ray UDF stream multiplexer did not terminate"))
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
                if thread_started:
                    self._request_event_loop_stop()
                    if self._thread is not threading.current_thread():
                        # Construction has not transferred ownership to a
                        # caller, so returning while this thread is alive would
                        # orphan its loop and Ray references.  The production
                        # thread target publishes ``_loop_ready`` before doing
                        # any blocking work; once ``loop.stop`` is queued it
                        # therefore has a finite path to termination.
                        self._thread.join()
                else:
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
                or not record.registered
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
            # Even when every current probe fails, keep the process-owned
            # collector usable for future query slots.
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
        """Turn every asynchronous callback failure into visible collector state."""
        try:
            callback(*args)
        except BaseException as exc:
            self._report_scheduler_error(exc)

    def _submit_cleanup_operations(
        self,
        operations: Sequence[RayStreamCleanupOperation],
        *,
        on_each_done: Callable[[BaseException | None], None] | None = None,
        store_error: bool = True,
        slot_ids: Sequence[int] = (),
    ) -> tuple[_CleanupTicket, ...]:
        """Atomically transfer operations into the durable cleanup ledger."""
        actions = tuple(operations)
        if not actions:
            if on_each_done is not None:
                on_each_done(None)
            return ()

        owner_slots = frozenset(int(slot_id) for slot_id in slot_ids)
        tickets: list[_CleanupTicket] = []

        for action in actions:
            ticket_holder: list[_CleanupTicket] = []

            def complete(
                error: BaseException | None,
                *,
                holder: list[_CleanupTicket] = ticket_holder,
            ) -> None:
                callback_error: BaseException | None = None
                try:
                    if on_each_done is not None:
                        on_each_done(error)
                except BaseException as exc:
                    callback_error = exc
                finally:
                    ticket = holder[0]
                    with self._cv:
                        if store_error and error is not None:
                            self._terminal_cleanup_errors.append(error)
                        if callback_error is not None:
                            self._terminal_cleanup_errors.append(callback_error)
                        self._cleanup_tickets.discard(ticket)
                        for slot_id in owner_slots:
                            slot_tickets = self._cleanup_tickets_by_slot.get(slot_id)
                            if slot_tickets is not None:
                                slot_tickets.discard(ticket)
                                if not slot_tickets:
                                    self._cleanup_tickets_by_slot.pop(
                                        slot_id,
                                        None,
                                    )
                        self._cv.notify_all()
                    self._notify_wakeup()
                    self._stop_loop_after_final_cleanup_ticket()

            ticket = _CleanupTicket(
                operation=action,
                retry_on_error=action.retry_on_error,
                retry_on_incomplete=action.retry_on_incomplete,
                on_complete=complete,
            )
            ticket_holder.append(ticket)
            tickets.append(ticket)

        with self._cleanup_handoff_lock:
            # Publish durable ownership before any operation can run. Driving a
            # ticket only queues bounded control work, so it does not need an
            # asyncio handoff and remains viable after that loop has stopped.
            with self._cv:
                self._cleanup_tickets.update(tickets)
                for slot_id in owner_slots:
                    self._cleanup_tickets_by_slot[slot_id].update(tickets)
                self._cv.notify_all()
            for ticket in tickets:
                self._drive_cleanup_ticket(ticket)
        return tuple(tickets)

    @staticmethod
    def _cleanup_retry_delay(retry_count: int) -> float:
        exponent = min(max(0, int(retry_count) - 1), 30)
        return min(
            math.ldexp(_CLEANUP_RETRY_INITIAL_DELAY_S, exponent),
            _CLEANUP_RETRY_MAX_DELAY_S,
        )

    @staticmethod
    def _cleanup_response_timeout(retry_count: int) -> float:
        exponent = min(max(0, int(retry_count)), 30)
        return min(
            math.ldexp(_CLEANUP_RESPONSE_TIMEOUT_S, exponent),
            _CLEANUP_RESPONSE_TIMEOUT_MAX_S,
        )

    def _retry_cleanup_ticket(self, ticket: _CleanupTicket) -> None:
        with self._cv:
            if (
                ticket not in self._cleanup_tickets
                or ticket.submitting
                or ticket.operation_future is not None
                or ticket.wait_future is not None
                or ticket.retry_call is not None
            ):
                return
            ticket.retry_count += 1
            retry_call: RayControlScheduledCall

            def retry() -> None:
                self._run_cleanup_retry(ticket, retry_call)

            retry_call = create_ray_control_deadline(
                f"{ticket.operation.submission_scope}:cleanup-retry",
                retry,
            )
            ticket.retry_call = retry_call
            retry_delay = self._cleanup_retry_delay(ticket.retry_count)
        try:
            RAY_CONTROL_DEADLINE_SCHEDULER.schedule(retry_call, retry_delay)
        except BaseException as exc:
            with self._cv:
                if ticket.retry_call is retry_call:
                    ticket.retry_call = None
                    active = ticket in self._cleanup_tickets
                else:
                    active = False
            if active:
                ticket.finish(exc)

    def _run_cleanup_retry(
        self,
        ticket: _CleanupTicket,
        retry_call: RayControlScheduledCall,
    ) -> None:
        with self._cv:
            if ticket not in self._cleanup_tickets or ticket.retry_call is not retry_call:
                return
            ticket.retry_call = None
        self._drive_cleanup_ticket(ticket)

    def _resume_cleanup_ticket(
        self,
        ticket: _CleanupTicket,
        future: Any,
        accept_response: Callable[[Any], Any] | None,
    ) -> None:
        with self._cv:
            if ticket not in self._cleanup_tickets:
                return
            current = ticket.wait_future is future
            if current:
                ticket.wait_future = None
                ticket.wait_response = None
                timeout = ticket.wait_timeout
                ticket.wait_timeout = None
            else:
                timeout = None
        if timeout is not None:
            timeout.cancel()
        if accept_response is None:
            if current:
                self._drive_cleanup_ticket(ticket)
            return
        try:
            accept_response(future.result())
        except BaseException as exc:
            if current:
                self._finish_cleanup_operation(ticket, error=exc)
            return

        # A valid acknowledgement from any timed-out attempt proves that the
        # idempotent terminal transition completed. Retire a newer observer or
        # pending retry before publishing ticket completion.
        with self._cv:
            if ticket not in self._cleanup_tickets:
                return
            wait_timeout = ticket.wait_timeout
            retry_call = ticket.retry_call
            ticket.wait_future = None
            ticket.wait_response = None
            ticket.wait_timeout = None
            ticket.retry_call = None
        if wait_timeout is not None:
            wait_timeout.cancel()
        if retry_call is not None:
            retry_call.cancel()
        ticket.finish(None)

    def _timeout_cleanup_ticket(
        self,
        ticket: _CleanupTicket,
        future: Any,
        response: RayStreamCleanupWait,
        timeout_call: RayControlScheduledCall,
    ) -> None:
        with self._cv:
            if (
                ticket not in self._cleanup_tickets
                or ticket.wait_future is not future
                or ticket.wait_response is not response
                or ticket.wait_timeout is not timeout_call
            ):
                return
            ticket.wait_future = None
            ticket.wait_response = None
            ticket.wait_timeout = None
        # The Future callback retains this response attempt, so a later valid
        # acknowledgement can still complete the ticket while the retry owns
        # forward progress.
        self._retry_cleanup_ticket(ticket)

    def _drive_cleanup_ticket(self, ticket: _CleanupTicket) -> None:
        with self._cv:
            if (
                ticket not in self._cleanup_tickets
                or ticket.submitting
                or ticket.operation_future is not None
                or ticket.wait_future is not None
                or ticket.retry_call is not None
            ):
                return
            ticket.submitting = True
        try:
            operation_future = submit_ray_control(
                ticket.operation.submission_scope,
                ticket.operation,
            )
        except BaseException as exc:
            with self._cv:
                ticket.submitting = False
            self._finish_cleanup_operation(ticket, error=exc)
            return
        with self._cv:
            ticket.submitting = False
            if ticket not in self._cleanup_tickets:
                return
            ticket.operation_future = operation_future

        collector_ref = weakref.ref(self)
        ticket_ref = weakref.ref(ticket)

        def ready(future: Any) -> None:
            collector = collector_ref()
            active_ticket = ticket_ref()
            if collector is None or active_ticket is None:
                return
            collector._run_scheduler_callback(
                collector._complete_cleanup_operation,
                active_ticket,
                future,
            )

        try:
            operation_future.add_done_callback(ready)
        except BaseException as exc:
            with self._cv:
                if ticket in self._cleanup_tickets and ticket.operation_future is operation_future:
                    ticket.operation_future = None
                    active = True
                else:
                    active = False
            if active:
                self._finish_cleanup_operation(ticket, error=exc)

    def _complete_cleanup_operation(
        self,
        ticket: _CleanupTicket,
        operation_future: Any,
    ) -> None:
        with self._cv:
            if ticket not in self._cleanup_tickets or ticket.operation_future is not operation_future:
                return
            ticket.operation_future = None
        assert operation_future is not None
        try:
            result = operation_future.result()
        except BaseException as exc:
            self._finish_cleanup_operation(ticket, error=exc)
            return
        self._finish_cleanup_operation(ticket, result=result)

    def _finish_cleanup_operation(
        self,
        ticket: _CleanupTicket,
        *,
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        if error is not None:
            if ticket.retry_on_error:
                self._retry_cleanup_ticket(ticket)
                return
            ticket.finish(error)
            return

        response = result if isinstance(result, RayStreamCleanupWait) else None
        future = response.future if response is not None else result
        if all(callable(getattr(future, name, None)) for name in ("add_done_callback", "done", "result")):
            with self._cv:
                if ticket not in self._cleanup_tickets:
                    return
                ticket.wait_future = future
                ticket.wait_response = response

            collector_ref = weakref.ref(self)
            ticket_ref = weakref.ref(ticket)
            response_holder = [None if response is None else response.accept]

            def ready(done: Any) -> None:
                accept_response = response_holder.pop() if response_holder else None
                collector = collector_ref()
                active_ticket = ticket_ref()
                if collector is None or active_ticket is None:
                    return
                collector._run_scheduler_callback(
                    collector._resume_cleanup_ticket,
                    active_ticket,
                    done,
                    accept_response,
                )

            try:
                future.add_done_callback(ready)
            except BaseException as exc:
                with self._cv:
                    if ticket in self._cleanup_tickets and ticket.wait_future is future:
                        ticket.wait_future = None
                        ticket.wait_response = None
                        active = True
                    else:
                        active = False
                # Callback registration is part of the required public Future
                # contract, not a transient remote operation.
                if active:
                    ticket.finish(exc)
                return
            if response is not None:
                timeout_call: RayControlScheduledCall

                def response_timed_out() -> None:
                    self._timeout_cleanup_ticket(
                        ticket,
                        future,
                        response,
                        timeout_call,
                    )

                timeout_call = create_ray_control_deadline(
                    f"{ticket.operation.submission_scope}:cleanup-response-timeout",
                    response_timed_out,
                )
                with self._cv:
                    if (
                        ticket in self._cleanup_tickets
                        and ticket.wait_future is future
                        and ticket.wait_response is response
                    ):
                        ticket.wait_timeout = timeout_call
                        start_timeout = True
                    else:
                        start_timeout = False
                if start_timeout:
                    try:
                        RAY_CONTROL_DEADLINE_SCHEDULER.schedule(
                            timeout_call,
                            self._cleanup_response_timeout(ticket.retry_count),
                        )
                    except BaseException as exc:
                        with self._cv:
                            if (
                                ticket.wait_timeout is timeout_call
                                and ticket.wait_future is future
                                and ticket.wait_response is response
                            ):
                                ticket.wait_timeout = None
                                ticket.wait_future = None
                                ticket.wait_response = None
                                active = ticket in self._cleanup_tickets
                            else:
                                active = False
                        if active:
                            ticket.finish(exc)
                else:
                    timeout_call.cancel()
            return

        if result is False and ticket.retry_on_incomplete:
            self._retry_cleanup_ticket(ticket)
            return
        ticket.finish(None)

    def _stop_loop_after_final_cleanup_ticket(self) -> None:
        with self._cv:
            stop = self._shutdown and not self._cleanup_tickets
        if stop:
            self._request_event_loop_stop()

    def _request_event_loop_stop(self) -> None:
        """Idempotently wake a live loop without racing its final close."""
        if not self._thread.is_alive():
            return
        if self._thread is threading.current_thread():
            self._loop.stop()
            return
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            if not self._loop.is_closed() and self._thread.is_alive():
                raise
            # The loop completed between the liveness check and notification.

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

    @staticmethod
    def _detach_record_waits_locked(record: _StreamRecord) -> None:
        """Detach local observers while retaining handles until cleanup handoff."""
        futures = (
            *record.detached_waits,
            record.wait_future,
            record.completion_future,
            record.output_submit_future,
        )
        record.wait_future = None
        record.completion_future = None
        record.output_submit_future = None
        record.wait_kind = ""
        unique: list[Any] = []
        seen: set[int] = set()
        for future in futures:
            if future is None or id(future) in seen:
                continue
            seen.add(id(future))
            unique.append(future)
        record.detached_waits = tuple(unique)

    @staticmethod
    def _abandon_record_waits(
        futures: Sequence[Any],
        *,
        owner_scope: str,
    ) -> None:
        """Cancel detached observers only after durable cleanup owns the record."""
        for future in futures:
            cancel = getattr(future, "cancel", None)
            if not callable(cancel):
                continue
            try:
                dispatch_ray_control_abandonment(
                    f"{owner_scope}:future:{id(future):x}",
                    cancel,
                )
            except BaseException:
                # Remote ownership is already durable. Event-loop shutdown is
                # the final fallback for local observers if dispatch fails.
                pass

    @staticmethod
    def _object_ref_future(ref: Any) -> Any:
        return ray_object_ref_future(ref)

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

        collector_ref = weakref.ref(self)
        record_ref = weakref.ref(record)

        def on_done(done: Any) -> None:
            collector = collector_ref()
            active_record = record_ref()
            if collector is None or active_record is None:
                return
            with collector._cv:
                if collector._shutdown:
                    return
            try:
                collector._loop.call_soon_threadsafe(
                    collector._run_scheduler_callback,
                    collector._complete_record_wait,
                    active_record,
                    kind,
                    done,
                )
            except RuntimeError as exc:
                with collector._cv:
                    shutting_down = collector._shutdown
                if not shutting_down:
                    collector._report_scheduler_error(exc)

        future.add_done_callback(on_done)
        return block_admitted

    def _schedule_completion_wait_locked(self, record: _StreamRecord) -> None:
        if record.terminal or record.producer_completed or record.completion_future is not None:
            return
        future = self._object_ref_future(record.adapter.completion_ref)
        record.completion_future = future

        collector_ref = weakref.ref(self)
        record_ref = weakref.ref(record)

        def on_done(done: Any) -> None:
            collector = collector_ref()
            active_record = record_ref()
            if collector is None or active_record is None:
                return
            with collector._cv:
                if collector._shutdown:
                    return
            try:
                collector._loop.call_soon_threadsafe(
                    collector._run_scheduler_callback,
                    collector._complete_producer_wait,
                    active_record,
                    done,
                )
            except RuntimeError as exc:
                with collector._cv:
                    shutting_down = collector._shutdown
                if not shutting_down:
                    collector._report_scheduler_error(exc)

        future.add_done_callback(on_done)

    def _refresh_waiters(self) -> None:
        failures: list[tuple[_StreamRecord, BaseException]] = []
        with self._cv:
            if self._shutdown or self._thread_error is not None or not self._started or not self._loop_ready.is_set():
                return
            records = sorted(
                self._records.values(),
                key=lambda record: (
                    record.phase == "block" and record.next_ref_ready,
                    record.ready_sequence if record.ready_sequence is not None else record.sequence,
                ),
            )
            pending_data_counts = self._pending_data_counts_locked()
            for record in records:
                if not record.registered:
                    continue
                try:
                    self._schedule_completion_wait_locked(record)
                    if self._schedule_record_wait_locked(
                        record,
                        pending_data_count=pending_data_counts.get(record.slot_id, 0),
                    ):
                        pending_data_counts[record.slot_id] = pending_data_counts.get(record.slot_id, 0) + 1
                except BaseException as exc:
                    failures.append((record, exc))
        for record, failure in failures:
            self._fail_record(record, failure)

    def _complete_producer_wait(self, record: _StreamRecord, future: Any) -> None:
        key = (record.slot_id, record.submit_id)
        with self._cleanup_handoff_lock:
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
                # A failed completion ObjectRef means Ray has already made the
                # task terminal. Force-cancelling it can race its normal reply.
                record.terminal_signal_observed = True
                with self._cv:
                    shutting_down = self._shutdown
                if not shutting_down:
                    self._fail_record(record, exc)
            finally:
                # Keep the completed future installed until its state transition
                # is fully applied. A concurrent capacity refresh must not
                # register a duplicate callback for the completion ObjectRef.
                with self._cv:
                    if record.completion_future is future:
                        record.completion_future = None
        self._refresh_waiters()

    def _complete_record_wait(self, record: _StreamRecord, kind: str, future: Any) -> None:
        key = (record.slot_id, record.submit_id)
        with self._cleanup_handoff_lock:
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
                # it before block/metadata/output state is updated lets a
                # capacity refresh corrupt the strict block/metadata pairing.
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
            details = remote_error["exception_details"]
            if details is not None:
                from vane.ai.provider import ProviderCapabilityError

                raise ProviderCapabilityError._from_safe_summary(
                    details["provider"],
                    details["model"],
                    details["capability"],
                    details["original_error_summary"],
                ) from None
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

        output_submit_future = submit_ray_control(
            record.adapter.submission_scope,
            lambda: driver.acquire_query_output_block_lease.remote(
                dict(request),
            ),
        )
        with self._cv:
            stale = self._shutdown or self._records.get(key) is not record or record.terminal
            if not stale:
                record.output_submit_future = output_submit_future
            cancel_stale_request = stale and not record.output_cancel_sent
            if cancel_stale_request:
                record.output_cancel_sent = True
        if stale:
            if cancel_stale_request:
                self._submit_cleanup_operations(
                    (
                        self._output_request_cancel_operation(
                            driver,
                            request,
                            record.adapter.submission_scope,
                        ),
                    ),
                    slot_ids=(record.slot_id,),
                )
            return

        collector_ref = weakref.ref(self)
        record_ref = weakref.ref(record)

        def submitted(done: Any) -> None:
            collector = collector_ref()
            active_record = record_ref()
            if collector is None or active_record is None:
                return
            with collector._cv:
                if collector._shutdown:
                    return
            try:
                collector._loop.call_soon_threadsafe(
                    collector._run_scheduler_callback,
                    collector._complete_output_lease_submission,
                    active_record,
                    done,
                )
            except RuntimeError as exc:
                with collector._cv:
                    shutting_down = collector._shutdown
                if not shutting_down:
                    collector._report_scheduler_error(exc)

        try:
            output_submit_future.add_done_callback(submitted)
        except BaseException:
            with self._cv:
                if record.output_submit_future is output_submit_future:
                    record.output_submit_future = None
            raise
        _collector_debug_log("output_lease_requested", record)

    def _complete_output_lease_submission(
        self,
        record: _StreamRecord,
        output_submit_future: Any,
    ) -> None:
        key = (record.slot_id, record.submit_id)
        failure: BaseException | None = None
        with self._cleanup_handoff_lock:
            with self._cv:
                if record.output_submit_future is not output_submit_future:
                    return
                record.output_submit_future = None
                if self._shutdown or self._records.get(key) is not record or record.terminal:
                    return
            assert output_submit_future is not None
            try:
                output_lease_ref = output_submit_future.result()
            except BaseException as exc:
                failure = exc
            else:
                with self._cv:
                    if self._shutdown or self._records.get(key) is not record or record.terminal:
                        return
                    record.output_lease_ref = output_lease_ref
                    self._cv.notify_all()
        if failure is not None:
            self._fail_record(record, failure)
            return
        self._refresh_waiters()

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
            submission_scope=record.adapter.submission_scope,
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
        with self._cleanup_handoff_lock:
            with self._cv:
                stale = (
                    self._shutdown
                    or record.terminal
                    or self._records.get((record.slot_id, record.submit_id)) is not record
                )
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
                else:
                    claimed_tokens = self._claim_output_token_releases_locked((token,))
            if stale:
                try:
                    self._submit_cleanup_operations(
                        tuple(self._output_token_release_operation(item) for item in claimed_tokens),
                        slot_ids=(record.slot_id,),
                    )
                except BaseException:
                    self._restore_output_token_release_claims(claimed_tokens)
                    raise
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
            with self._cleanup_handoff_lock:
                with self._cv:
                    if self._records.get(key) is not record:
                        if error is not None:
                            self._terminal_cleanup_errors.append(error)
                            self._cv.notify_all()
                        return
                    dropped_tokens: list[_OutputLeaseToken] = []
                    if error is not None:
                        ready = self._ready_by_slot.get(record.slot_id)
                        if ready is not None:
                            for queued in ready:
                                if queued.submit_id == record.submit_id and queued.kind == "data":
                                    if queued.output_token is not None:
                                        dropped_tokens.append(queued.output_token)
                    claimed_tokens = self._claim_output_token_releases_locked(dropped_tokens)
                try:
                    self._submit_cleanup_operations(
                        tuple(self._output_token_release_operation(token) for token in claimed_tokens),
                        slot_ids=(record.slot_id,),
                    )
                except BaseException:
                    self._restore_output_token_release_claims(claimed_tokens)
                    raise
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
                            self._ready_by_slot[record.slot_id] = deque(
                                queued
                                for queued in ready
                                if not (queued.submit_id == record.submit_id and queued.kind == "data")
                            )
                        for token in dropped_tokens:
                            self._active_output_leases.pop((token.request_id, token.lease_id), None)
                        event = _ReadyEvent(
                            record.slot_id,
                            record.submit_id,
                            "error",
                            f"{type(error).__name__}: {error}",
                        )
                    self._ready_by_slot[record.slot_id].append(event)
                    self._cv.notify_all()
            _collector_debug_log("retired" if error is None else "retire_failed", record)
            self._notify_wakeup()

        def finish_cleanup_operation(error: BaseException | None) -> None:
            with self._cv:
                if error is not None and record.cleanup_error is None:
                    record.cleanup_error = error
                record.cleanup_remaining -= 1
                if record.cleanup_remaining < 0:
                    raise RuntimeError("Ray UDF record cleanup count underflow")
                if record.cleanup_remaining:
                    return
                cleanup_error = record.cleanup_error
            finish_retirement(cleanup_error)

        with self._cleanup_handoff_lock:
            with self._cv:
                if self._records.get(key) is not record:
                    return
                record.terminal = True
                record.cleanup_started = True
                self._signal_readiness_change_locked()
                cleanup_operations = record.adapter.retire_operations()
                record.cleanup_remaining = len(cleanup_operations)
                record.cleanup_error = None
            if cleanup_operations:
                try:
                    self._submit_cleanup_operations(
                        cleanup_operations,
                        on_each_done=finish_cleanup_operation,
                        store_error=False,
                        slot_ids=(record.slot_id,),
                    )
                except BaseException:
                    with self._cv:
                        if self._records.get(key) is record:
                            record.cleanup_started = False
                            record.cleanup_remaining = 0
                            record.cleanup_error = None
                            self._cv.notify_all()
                    raise
            else:
                finish_retirement(None)

    def _fail_record(self, record: _StreamRecord, exc: BaseException) -> None:
        exc = format_stateful_actor_loss(record.error_context, exc)
        key = (record.slot_id, record.submit_id)
        with self._cleanup_handoff_lock:
            with self._cv:
                if self._records.get(key) is not record:
                    return
                record.terminal = True
                cleanup_was_started = record.cleanup_started
                record.cleanup_started = True
                ready = self._ready_by_slot.get(record.slot_id)
                dropped_tokens: list[_OutputLeaseToken] = []
                if ready is not None:
                    for event in ready:
                        if event.submit_id == record.submit_id and event.kind == "data":
                            if event.output_token is not None:
                                dropped_tokens.append(event.output_token)
                claimed_tokens = self._claim_output_token_releases_locked(dropped_tokens)
                request = record.output_request
                driver = record.adapter.driver
                output_cancel_operation: RayStreamCleanupOperation | None = None
                if request is not None and driver is not None and not record.output_cancel_sent:
                    record.output_cancel_sent = True
                    output_cancel_operation = self._output_request_cancel_operation(
                        driver,
                        request,
                        record.adapter.submission_scope,
                    )
                terminal_signal_observed = record.terminal_signal_observed
                cleanup_operations: list[RayStreamCleanupOperation] = [
                    self._output_token_release_operation(token) for token in claimed_tokens
                ]
                if output_cancel_operation is not None:
                    cleanup_operations.append(output_cancel_operation)
                if not cleanup_was_started:
                    if terminal_signal_observed:
                        cleanup_operations.extend(record.adapter.retire_failed_operations())
                    else:
                        cleanup_operations.extend(record.adapter.cancel_operations())
                self._signal_readiness_change_locked()
            try:
                self._submit_cleanup_operations(
                    cleanup_operations,
                    slot_ids=(record.slot_id,),
                )
            except BaseException:
                self._restore_output_token_release_claims(claimed_tokens)
                with self._cv:
                    if self._records.get(key) is record:
                        if not cleanup_was_started:
                            record.cleanup_started = False
                        if output_cancel_operation is not None:
                            record.output_cancel_sent = False
                        self._cv.notify_all()
                raise
            with self._cv:
                if self._records.get(key) is not record:
                    return
                self._records.pop(key, None)
                ready = self._ready_by_slot.get(record.slot_id)
                if ready is not None:
                    self._ready_by_slot[record.slot_id] = deque(
                        event for event in ready if not (event.submit_id == record.submit_id and event.kind == "data")
                    )
                for token in dropped_tokens:
                    self._active_output_leases.pop((token.request_id, token.lease_id), None)
                self._ready_by_slot[record.slot_id].append(
                    _ReadyEvent(
                        record.slot_id,
                        record.submit_id,
                        "error",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                self._signal_readiness_change_locked()

        # The complete terminal plan has a durable ticket owner before an
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


__all__ = ["UDFStreamResultCollector"]
