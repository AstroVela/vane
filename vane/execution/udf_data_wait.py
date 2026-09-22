# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded byte waiting using the existing query and physical-slot arbiters."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vane.execution.request_deadline import MonotonicDeadline
from vane.execution.udf_admission import AdmissionLease, LocalSlotAdmissionAuthority
from vane.execution.udf_data_admission import (
    DataAdmissionCapacityError,
    DataAdmissionQueueFull,
    DataAdmissionTimeout,
)
from vane.execution.udf_runtime_admission import QueryTaskAdmission, _failure_message

if TYPE_CHECKING:
    from vane.execution.local_resource_graph import LocalResourceUnitContext
    from vane.execution.udf_data_lease import DataTaskReservation, QueryDataScope
    from vane.execution.udf_resource_usage import UnitResourceActivity


class _ByteCapacity:
    def __init__(self, owner: WaitingDataAdmissionAuthority) -> None:
        self._owner = owner

    def try_acquire(self, retained_input_bytes: int) -> AdmissionLease | None:
        return self._owner._capacity.try_acquire_if(retained_input_bytes, self._owner._reserve)

    def register_capacity_wakeup(self, callback: Callable[[], None]) -> None:
        self._owner._capacity.register_capacity_wakeup(callback)

    def state(self) -> dict[str, Any]:
        return self._owner._capacity.state()

    def close(self) -> None:
        self._owner._capacity.close()


class WaitingDataAdmissionAuthority:
    """Acquire bytes and physical capacity before publishing a task allowance.

    The runtime ledger lock protects the byte request, deadline and reservation.
    Guards run under the physical pool lock and never notify other authorities.
    All calls back into task/pool arbitration happen outside the ledger lock.
    """

    def __init__(
        self,
        capacity: LocalSlotAdmissionAuthority,
        query: QueryDataScope,
        *,
        task_query: QueryTaskAdmission | None = None,
        resource_unit: LocalResourceUnitContext,
        activity: UnitResourceActivity | None = None,
    ) -> None:
        from vane.execution.ref_bundle import register_local_shm_ref_budget_wakeup

        self._capacity = capacity
        self._query = query
        self._ledger = query._ledger
        self._resource_unit = resource_unit
        self._activity = activity
        self._reservation: DataTaskReservation | None = None
        self._pending = False
        self._closed = False
        self._generation = 0
        self._reason: str | None = None
        self._error: tuple[type[Exception], str] | None = None
        self._deadline: MonotonicDeadline | None = None
        self._wakeup: Callable[[], None] | None = None
        self._unregister: Callable[[], None] | None = None
        self._base = (task_query or query.wait_task_query()).create_authority(_ByteCapacity(self))
        try:
            self._base.register_wakeup(self._notify)
            unregister = register_local_shm_ref_budget_wakeup(self.wake)
            with self._ledger._condition:
                self._unregister = unregister
                state = self._ledger._queries.get(query.query_id)
                if state is None or state.closed:
                    raise RuntimeError("query data scope is closed")
                state.authorities.add(self)
        except BaseException:
            self.close()
            raise

    def _stop_wait_locked(self) -> None:
        self._ledger._byte_waiters.pop(self, None)
        self._reason = None
        if self._deadline is not None:
            self._deadline.close()
            self._deadline = None

    def _fail_locked(self, kind: type[Exception], message: str) -> str:
        self._error = kind, message
        self._stop_wait_locked()
        return message

    def _wait_locked(self, reason: str) -> None:
        assert self._query.limits is not None and self._query.limits.wait is not None
        if self not in self._ledger._byte_waiters:
            if len(self._ledger._byte_waiters) >= self._query.limits.wait.max_queued_tasks:
                raise DataAdmissionQueueFull(
                    self._fail_locked(DataAdmissionQueueFull, "runtime pending byte-admission queue is full")
                )
            self._ledger._byte_waiters[self] = None
        self._reason = reason

    def _reserve(self) -> bool:
        # Called only after the shared arbiter has a physical slot and fair turn.
        with self._ledger._condition:
            if self._closed or not self._pending:
                return False
            if self._error is not None:
                raise self._error[0](self._error[1])
            if self._deadline is not None and self._deadline.expired():
                raise DataAdmissionTimeout(
                    self._fail_locked(DataAdmissionTimeout, "UDF byte-admission queue deadline expired")
                )
            try:
                reservation = self._query.reserve_task(resource_unit=self._resource_unit)
            except DataAdmissionCapacityError as exc:
                if self._activity is not None and self._reason != f"{exc.owner}_bytes":
                    self._activity.refuse_bytes(exc.owner)
                self._wait_locked(f"{exc.owner}_bytes")
                return False
            except Exception as exc:
                self._fail_locked(RuntimeError, _failure_message(exc))
                raise
            self._reservation = reservation
            if self._deadline is not None and self._deadline.expired():
                # Reservation can consult a transport adapter. If it crossed
                # the deadline, keep its cleanup owner but publish no worker.
                raise DataAdmissionTimeout(
                    self._fail_locked(DataAdmissionTimeout, "UDF byte-admission queue deadline expired")
                )
            self._stop_wait_locked()
            return True

    def request(self, retained_input_bytes: int) -> bool:
        retained = int(retained_input_bytes)
        if retained < 0:
            raise ValueError("retained_input_bytes must be >= 0")
        with self._ledger._condition:
            if self._closed:
                raise RuntimeError("byte admission authority is closed")
            if self._error is not None:
                raise self._error[0](self._error[1])
            if self._pending:
                return False
            self._pending = True
            self._generation += 1
            generation = self._generation
            assert self._query.limits is not None and self._query.limits.wait is not None
            deadline = MonotonicDeadline(
                time.monotonic(),
                self._query.limits.wait.queue_timeout,
                lambda: self._expire(generation),
                timeout_name="byte queue_timeout",
                thread_name="vane-byte-admission-deadline",
            )
            self._deadline = deadline
        try:
            accepted = self._base.request(retained)
            with self._ledger._condition:
                if self._error is not None:
                    raise self._error[0](self._error[1])
            state = self._base.state()
            with self._ledger._condition:
                if self._error is not None:
                    raise self._error[0](self._error[1])
                if state["state"] == "requested" and self._reservation is None and self._pending and not self._closed:
                    self._wait_locked(self._reason or "execution_capacity")
            # Immediate grants close the watcher before start(), avoiding a
            # thread for batches that never actually queue.
            deadline.start()
            return accepted
        except BaseException as error:
            # Queue rejection is terminal for this executor; close its queued
            # task request too, without discarding previously submitted leases.
            try:
                self.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def state(self) -> dict[str, Any]:
        with self._ledger._condition:
            if self._error is not None:
                return {"state": "failed", "available": False, "error": self._error[1], "retained_input_bytes": 0}
            if self._closed:
                return {"state": "closed", "available": False, "retained_input_bytes": 0}
            reason = self._reason
            # Native inputs can retain upstream output while coalescing a
            # preferred batch, before asking for their own task allowance.
            # Let these consumers spend their protected envelopes under byte
            # pressure. Reading this signal never requests execution capacity.
            flush_input = any(
                authority._reason in {"runtime_bytes", "transport_bytes"} for authority in self._ledger._byte_waiters
            )
        state = self._base.state()
        if state["state"] == "requested" and reason in {"runtime_bytes", "transport_bytes"}:
            state = {**state, "state": "waiting_bytes", "waiting_reason": reason}
        return {**state, "flush_partial_input": flush_input}

    def diagnostic_state(self) -> str:
        with self._ledger._condition:
            if self._closed:
                return "closed"
            if self._error is not None:
                return "failed"
            if self._reason in {"runtime_bytes", "transport_bytes"}:
                return "waiting_bytes"
        return self._base.diagnostic_state()

    def take(self, retained_input_bytes: int) -> AdmissionLease:
        base = self._base.take(retained_input_bytes)
        with self._ledger._condition:
            reservation, self._reservation = self._reservation, None
            self._pending = False
        if reservation is None:
            base.release()
            raise RuntimeError("byte admission requires a ready reservation")

        def complete() -> None:
            try:
                reservation.release()
            finally:
                base.complete_execution()

        return AdmissionLease(
            request_id=base.request_id,
            retained_input_bytes=base.retained_input_bytes,
            lease={**base.lease, "local_data_reservation": reservation},
            driver=base.driver,
            _release_callback=base.release,
            _execution_finished_callback=complete,
            _capacity_wait_context=base.suspend_for_wait,
        )

    def _expire(self, generation: int) -> None:
        with self._ledger._condition:
            if self._closed or generation != self._generation or self._deadline is None:
                return
            self._fail_locked(DataAdmissionTimeout, "UDF byte-admission queue deadline expired")
        # Fence the recorded timeout before a racing capacity notification.
        try:
            self._base.close()
        except BaseException as exc:
            with self._ledger._condition:
                self._error = (
                    DataAdmissionTimeout,
                    (f"UDF byte-admission queue deadline expired; capacity cleanup failed: {_failure_message(exc)}"),
                )
        finally:
            self._notify()
            self._ledger.wake_byte_waiters()

    def register_wakeup(self, callback: Callable[[], None] | None) -> None:
        with self._ledger._condition:
            self._wakeup = callback

    def wrap_wakeup(self, callback: Callable[[], None] | None) -> Callable[[], None] | None:
        # state() publishes failures as values, including from reentrant wakes.
        return callback

    def _notify(self) -> None:
        with self._ledger._condition:
            callback = self._wakeup
        if callback is not None:
            try:
                callback()
            except BaseException as exc:
                with self._ledger._condition:
                    self._fail_locked(RuntimeError, f"byte admission wakeup failed: {_failure_message(exc)}")

    def wake(self) -> None:
        with self._ledger._condition:
            if self._closed:
                return
        try:
            self._capacity.notify_capacity()
        except BaseException as exc:
            with self._ledger._condition:
                self._fail_locked(RuntimeError, f"byte admission capacity wakeup failed: {_failure_message(exc)}")
        self._notify()

    def close(self) -> None:
        with self._ledger._condition:
            self._closed = True
            self._pending = False
            self._stop_wait_locked()
            reservation = self._reservation
            self._wakeup = None
            unregister, self._unregister = self._unregister, None
        if unregister is not None:
            unregister()
        try:
            self._base.close()
        finally:
            if reservation is not None:
                reservation.release()
        with self._ledger._condition:
            if self._reservation is reservation:
                self._reservation = None
            state = self._ledger._queries.get(self._query.query_id)
            if state is not None:
                state.authorities.discard(self)
                self._ledger._retire_query_locked(self._query.query_id)
