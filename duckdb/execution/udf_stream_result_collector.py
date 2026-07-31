# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import math
import os
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from duckdb.execution.ray_stream_adapter import (
    RayStreamAdapter,
    TaskLeaseObjectRefGenerator,
)
from duckdb.execution.udf_ray_actor_state import format_stateful_actor_loss
from duckdb.execution.udf_ray_config import REF_BUNDLE_RESULT_MARKER
from duckdb.execution.udf_ray_stream_protocol import (
    validate_stream_block_metadata,
    validate_stream_error_metadata,
)

_GENERATOR_READINESS_POLL_INITIAL_DELAY_S = 0.001
_GENERATOR_READINESS_POLL_MAX_DELAY_S = 0.01


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
    driver: Any
    slot_id: int
    submit_id: int
    size_bytes: int
    handed_off: bool = False


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


class AsyncResultCollector:
    """Event-driven multiplexer for lease-owned Ray generator streams."""

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
        self._cleanup_threads: set[threading.Thread] = set()
        self._cleanup_errors: list[BaseException] = []
        self._cleanup_starting = 0
        self._readiness_delay_s = _GENERATOR_READINESS_POLL_INITIAL_DELAY_S
        self._readiness_timer: asyncio.TimerHandle | None = None
        self._readiness_wakeup_pending = False
        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._start_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="udf-ray-stream-multiplexer",
        )

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
        with self._cv:
            self._raise_if_stopped_locked()
            if key in self._records:
                raise ValueError(f"duplicate Ray UDF stream identity slot={key[0]} submit={key[1]}")
            if key[0] in self._cancelled_slots:
                raise RuntimeError(f"Ray UDF slot {key[0]} is cancelled")
            adapter = RayStreamAdapter(source, ray_module=self._ray)
            record = _StreamRecord(
                slot_id=key[0],
                submit_id=key[1],
                adapter=adapter,
                sequence=self._next_sequence,
                error_context=dict(error_context) if error_context else None,
            )
            self._next_sequence += 1
            self._records[key] = record
            self._signal_readiness_change_locked()
        _collector_debug_log("track", record)
        try:
            self._ensure_started()
            with self._cv:
                if self._records.get(key) is not record:
                    self._raise_if_stopped_locked()
                    raise RuntimeError(f"Ray UDF stream slot={key[0]} submit={key[1]} was cancelled during startup")
        except BaseException as exc:
            with self._cv:
                owned = self._records.pop(key, None) is record
                record.terminal = True
                self._cv.notify_all()
            if owned:
                try:
                    record.adapter.cancel()
                except BaseException as cleanup_exc:
                    add_note = getattr(exc, "add_note", None)
                    if callable(add_note):
                        add_note(f"stream startup cleanup failed: {cleanup_exc}")
            raise
        self._refresh_waiters()

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
            token = self._active_output_leases.pop(key, None)
            self._cv.notify_all()
        if token is None:
            return False
        token.driver.release_query_output_block_lease.remote(
            token.request_id,
            token.lease_id,
        )
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
            if token is None or token.handed_off:
                return False
            token.handed_off = True
            self._cv.notify_all()
        token.driver.handoff_query_output_block_lease.remote(
            token.request_id,
            token.lease_id,
        )
        return True

    def cancel_slot(self, slot_id: int) -> None:
        slot_key = int(slot_id)
        with self._cv:
            if self._shutdown:
                return
            records = [record for record in self._records.values() if record.slot_id == slot_key]
            records_to_cancel = [record for record in records if not record.cleanup_started]
            control_requests: list[tuple[Any, dict[str, Any]]] = []
            for record in records:
                self._records.pop((record.slot_id, record.submit_id), None)
                record.terminal = True
                if not record.cleanup_started:
                    record.cleanup_started = True
                self._cancel_record_wait_locked(record)
            for record in records_to_cancel:
                request = record.output_request
                driver = record.adapter.driver
                if request is not None and driver is not None and not record.output_cancel_sent:
                    record.output_cancel_sent = True
                    control_requests.append((driver, request))
            ready = list(self._ready_by_slot.pop(slot_key, ()))
            tokens = [token for token in self._active_output_leases.values() if token.slot_id == slot_key]
            for token in tokens:
                self._active_output_leases.pop((token.request_id, token.lease_id), None)
            self._capacity_by_slot.pop(slot_key, None)
            self._cancelled_slots.add(slot_key)
            self._cleanup_starting += 1
            self._signal_readiness_change_locked()

        ready_tokens = [event.output_token for event in ready if event.output_token is not None]
        token_keys = {(token.request_id, token.lease_id) for token in tokens}
        for token in ready_tokens:
            if token is not None and (token.request_id, token.lease_id) not in token_keys:
                tokens.append(token)
                token_keys.add((token.request_id, token.lease_id))

        def cleanup_slot() -> None:
            actions: list[Any] = [
                lambda driver=driver, request=request: driver.cancel_query_output_block_lease_request.remote(request)
                for driver, request in control_requests
            ]
            for record in records_to_cancel:
                actions.append(record.adapter.cancel)
            for token in tokens:
                actions.append(
                    lambda token=token: token.driver.release_query_output_block_lease.remote(
                        token.request_id,
                        token.lease_id,
                    )
                )
            self._run_isolated_cleanup_actions(
                actions,
                name="udf-ray-stream-cancel-action",
            )

        def cleanup_registered() -> None:
            with self._cv:
                self._cleanup_starting -= 1
                self._cv.notify_all()

        self._run_cleanup_after_scheduler_turn(
            cleanup_slot,
            timeout_message=f"Ray UDF slot {slot_key} cleanup did not terminate",
            on_registered=cleanup_registered,
        )

    def slot_has_pending(self, slot_id: int) -> bool:
        slot_key = int(slot_id)
        with self._cv:
            return (
                any(record.slot_id == slot_key for record in self._records.values())
                or bool(self._ready_by_slot.get(slot_key))
                or any(token.slot_id == slot_key for token in self._active_output_leases.values())
            )

    def set_wakeup_callback(self, fn: Any) -> None:
        with self._cv:
            self._wakeup_fn = fn

    def shutdown(self) -> None:
        deadline = time.monotonic() + self._shutdown_timeout_s
        shutdown_from_scheduler = self._thread is threading.current_thread()
        shutdown_errors: list[BaseException] = []
        with self._start_lock:
            with self._cv:
                if self._shutdown:
                    return
                self._shutdown = True
                self._wakeup_fn = None
                records = list(self._records.values())
                records_to_cancel = [record for record in records if not record.cleanup_started]
                control_requests: list[tuple[Any, dict[str, Any]]] = []
                for record in records:
                    self._cancel_record_wait_locked(record)
                for record in records_to_cancel:
                    request = record.output_request
                    driver = record.adapter.driver
                    if request is not None and driver is not None and not record.output_cancel_sent:
                        record.output_cancel_sent = True
                        control_requests.append((driver, request))
                self._records.clear()
                tokens = list(self._active_output_leases.values())
                self._active_output_leases.clear()
                self._ready_by_slot.clear()
                self._capacity_by_slot.clear()
                self._signal_readiness_change_locked()
            if self._thread_started:
                try:
                    self._loop.call_soon_threadsafe(self._loop.stop)
                except RuntimeError as exc:
                    if self._thread.is_alive():
                        shutdown_errors.append(exc)
        for driver, request in control_requests:
            self._start_tracked_cleanup(
                name="udf-ray-stream-shutdown-control",
                operation=lambda driver=driver, request=request: driver.cancel_query_output_block_lease_request.remote(
                    request
                ),
            )

        for record in records_to_cancel:

            def cancel_adapter(record: _StreamRecord = record) -> None:
                # A stalled zero-time Ray probe still owns the generator
                # handle. Keep this daemon task as its durable cleanup owner.
                if self._thread.is_alive() and not shutdown_from_scheduler:
                    self._thread.join()
                record.adapter.cancel()

            self._start_tracked_cleanup(
                name="udf-ray-stream-shutdown-adapter",
                operation=cancel_adapter,
            )
        for token in tokens:
            self._start_tracked_cleanup(
                name="udf-ray-stream-shutdown-lease",
                operation=lambda token=token: token.driver.release_query_output_block_lease.remote(
                    token.request_id,
                    token.lease_id,
                ),
            )

        if self._thread_started and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if self._thread.is_alive():
                shutdown_errors.append(RuntimeError("Ray UDF stream multiplexer did not terminate"))

        with self._cv:
            while self._cleanup_starting or self._cleanup_threads:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cv.wait(timeout=remaining)
            if self._cleanup_starting or self._cleanup_threads:
                shutdown_errors.append(RuntimeError("Ray UDF stream remote cleanup did not terminate"))
            shutdown_errors.extend(self._cleanup_errors)
            self._cleanup_errors.clear()
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

    def _pending_data_count_locked(self, slot_id: int) -> int:
        ready = sum(1 for event in self._ready_by_slot.get(slot_id, ()) if event.kind == "data")
        in_progress = sum(
            1
            for record in self._records.values()
            if record.slot_id == slot_id
            and (record.phase != "block" or (record.wait_kind == "data" and record.wait_future is not None))
        )
        return ready + in_progress

    def _may_read_block_locked(self, record: _StreamRecord) -> bool:
        if not record.next_ref_ready:
            return False
        capacity = self._capacity_by_slot.get(record.slot_id)
        if capacity is None or capacity.rows <= 0:
            return False
        if capacity.bytes is not None and capacity.bytes <= 0:
            return False
        if capacity.item_bytes is not None and capacity.item_bytes <= 0:
            return False
        return self._pending_data_count_locked(record.slot_id) < capacity.rows

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
        admissible_slots = {
            slot_id
            for slot_id, capacity in self._capacity_by_slot.items()
            if capacity.rows > 0
            and (capacity.bytes is None or capacity.bytes > 0)
            and (capacity.item_bytes is None or capacity.item_bytes > 0)
            and self._pending_data_count_locked(slot_id) < capacity.rows
        }
        for record in self._records.values():
            if (
                record.terminal
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
        probes, reads, and terminal retirement.  A zero-time public
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

        ready, _ = self._ray.wait(
            list(waitables),
            num_returns=len(waitables),
            fetch_local=False,
            timeout=0,
        )

        ready_records: list[_StreamRecord] = []
        with self._cv:
            if self._scheduler_stopped_locked():
                return
            currently_waitable = self._readiness_waitables_locked()
            for waitable in ready:
                record = waitables.get(waitable)
                if record is None:
                    self._thread_error = RuntimeError("Ray returned an unknown generator readiness handle")
                    self._cv.notify_all()
                    break
                if currently_waitable.get(waitable) is record:
                    record.next_ref_ready = True
                    record.ready_sequence = self._next_ready_sequence
                    self._next_ready_sequence += 1
                    ready_records.append(record)
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

    def _start_tracked_cleanup(
        self,
        *,
        name: str,
        operation: Any,
        on_done: Any | None = None,
        store_error: bool = True,
    ) -> tuple[threading.Thread | None, list[BaseException]]:
        """Run a possibly blocking Ray cleanup without occupying the scheduler."""
        errors: list[BaseException] = []
        thread: threading.Thread

        def cleanup() -> None:
            operation_error: BaseException | None = None
            try:
                operation()
            except BaseException as exc:
                operation_error = exc
                errors.append(exc)
            try:
                if on_done is not None:
                    on_done(operation_error)
                elif operation_error is not None and store_error:
                    with self._cv:
                        self._cleanup_errors.append(operation_error)
                        self._cv.notify_all()
            except BaseException as exc:
                errors.append(exc)
                if store_error:
                    with self._cv:
                        self._cleanup_errors.append(exc)
                        self._cv.notify_all()
            finally:
                with self._cv:
                    self._cleanup_threads.discard(thread)
                    self._cv.notify_all()

        thread = threading.Thread(
            target=cleanup,
            daemon=True,
            name=name,
        )
        with self._cv:
            self._cleanup_threads.add(thread)
        try:
            thread.start()
        except BaseException:
            with self._cv:
                self._cleanup_threads.discard(thread)
                self._cv.notify_all()
            cleanup()
            return None, errors
        return thread, errors

    def _run_isolated_cleanup_actions(
        self,
        actions: list[Any],
        *,
        name: str,
    ) -> None:
        """Attempt every independent cleanup even when one Ray call blocks."""
        handles: list[threading.Thread] = []
        action_errors: list[list[BaseException]] = []
        for action in actions:
            thread, errors = self._start_tracked_cleanup(
                name=name,
                operation=action,
                store_error=False,
            )
            action_errors.append(errors)
            if thread is not None:
                handles.append(thread)
        for thread in handles:
            thread.join()
        errors = [error for group in action_errors for error in group]
        if errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            raise RuntimeError(f"Ray UDF stream cleanup failed: {details}") from errors[0]

    def _run_cleanup_after_scheduler_turn(
        self,
        operation: Any,
        *,
        timeout_message: str,
        on_registered: Any | None = None,
    ) -> None:
        """Run terminal cleanup only after every earlier generator probe."""
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

        def cleanup() -> None:
            while not scheduler_safe.wait(timeout=0.1):
                if not self._thread.is_alive():
                    break
            operation()

        cleanup_thread, cleanup_errors = self._start_tracked_cleanup(
            name="udf-ray-stream-cancel",
            operation=cleanup,
            store_error=False,
        )
        if on_registered is not None:
            on_registered()
        if cleanup_thread is None:
            if cleanup_errors:
                raise RuntimeError(f"{timeout_message}: {cleanup_errors[0]}") from cleanup_errors[0]
            return
        cleanup_thread.join(timeout=self._shutdown_timeout_s)
        if cleanup_thread.is_alive():
            raise RuntimeError(timeout_message)
        if cleanup_errors:
            raise RuntimeError(f"{timeout_message}: {cleanup_errors[0]}") from cleanup_errors[0]

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

    def _schedule_record_wait_locked(self, record: _StreamRecord) -> None:
        if record.terminal or record.wait_future is not None:
            return
        kind = ""
        future = None
        if record.terminal_ref is not None:
            kind = "terminal"
            future = self._object_ref_future(record.terminal_ref)
        elif record.output_lease_ref is not None:
            kind = "output_lease"
            future = self._object_ref_future(record.output_lease_ref)
        elif record.metadata_ref is not None:
            kind = "metadata"
            future = self._object_ref_future(record.metadata_ref)
        elif record.phase == "metadata" or (record.phase == "block" and self._may_read_block_locked(record)):
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
                self._signal_readiness_change_locked()
            future = asyncio.run_coroutine_threadsafe(
                record.adapter.read_next_ref_async(),
                self._loop,
            )
        if future is None:
            return
        record.wait_kind = kind
        record.wait_future = future

        def on_done(done: Any) -> None:
            with self._cv:
                if self._shutdown:
                    return
            try:
                self._loop.call_soon_threadsafe(
                    self._complete_record_wait,
                    record,
                    kind,
                    done,
                )
            except RuntimeError as exc:
                with self._cv:
                    if not self._shutdown:
                        self._thread_error = exc
                        self._cv.notify_all()

        future.add_done_callback(on_done)

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
                    self._complete_producer_wait,
                    record,
                    done,
                )
            except RuntimeError as exc:
                with self._cv:
                    if not self._shutdown:
                        self._thread_error = exc
                        self._cv.notify_all()

        future.add_done_callback(on_done)

    def _refresh_waiters(self) -> None:
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
            for record in records:
                self._schedule_completion_wait_locked(record)
                self._schedule_record_wait_locked(record)

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
                self._start_tracked_cleanup(
                    name="udf-ray-stream-stale-control",
                    operation=lambda: driver.cancel_query_output_block_lease_request.remote(request),
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
            self._start_tracked_cleanup(
                name="udf-ray-stream-stale-lease",
                operation=lambda: driver.release_query_output_block_lease.remote(
                    token.request_id,
                    token.lease_id,
                ),
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
            with self._cv:
                if self._records.pop(key, None) is not record:
                    if error is not None:
                        self._cleanup_errors.append(error)
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
                        self._active_output_leases.pop((token.request_id, token.lease_id), None)
                    event = _ReadyEvent(
                        record.slot_id,
                        record.submit_id,
                        "error",
                        f"{type(error).__name__}: {error}",
                    )
                self._ready_by_slot[record.slot_id].append(event)
                for token in dropped_tokens:
                    self._start_tracked_cleanup(
                        name="udf-ray-stream-retire-leases",
                        operation=lambda token=token: token.driver.release_query_output_block_lease.remote(
                            token.request_id,
                            token.lease_id,
                        ),
                    )
                self._cv.notify_all()
            _collector_debug_log("retired" if error is None else "retire_failed", record)
            self._notify_wakeup()

        with self._cv:
            if self._records.get(key) is not record:
                return
            record.terminal = True
            record.cleanup_started = True
            self._signal_readiness_change_locked()
            self._start_tracked_cleanup(
                name="udf-ray-stream-retire",
                operation=record.adapter.retire,
                on_done=finish_retirement,
            )

    def _cancel_record_control(self, record: _StreamRecord) -> None:
        with self._cv:
            request = record.output_request
            driver = record.adapter.driver
            if request is None or driver is None or record.output_cancel_sent:
                return
            record.output_cancel_sent = True
        driver.cancel_query_output_block_lease_request.remote(request)

    def _fail_record(self, record: _StreamRecord, exc: BaseException) -> None:
        exc = format_stateful_actor_loss(record.error_context, exc)
        key = (record.slot_id, record.submit_id)
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
            self._cleanup_starting += 1
            self._signal_readiness_change_locked()
        # Publish the terminal event before any Ray cancellation/control work.
        # The C++ dispatcher is event-driven; delaying this callback until after
        # ray.cancel() can leave the slot asleep forever when cancellation is
        # slow or the failed generator is being reconstructed.
        try:
            self._notify_wakeup()
        finally:
            try:
                for token in dropped_tokens:
                    self._start_tracked_cleanup(
                        name="udf-ray-stream-failed-lease",
                        operation=lambda token=token: token.driver.release_query_output_block_lease.remote(
                            token.request_id,
                            token.lease_id,
                        ),
                    )
                if request is not None and driver is not None:
                    self._start_tracked_cleanup(
                        name="udf-ray-stream-failed-control",
                        operation=lambda: driver.cancel_query_output_block_lease_request.remote(request),
                    )
                self._start_tracked_cleanup(
                    name="udf-ray-stream-failed-adapter",
                    operation=record.adapter.retire_failed if terminal_signal_observed else record.adapter.cancel,
                )
            finally:
                with self._cv:
                    self._cleanup_starting -= 1
                    self._cv.notify_all()

    def _notify_wakeup(self) -> None:
        with self._cv:
            callback = self._wakeup_fn
        if callback is not None:
            callback()


__all__ = ["AsyncResultCollector"]
