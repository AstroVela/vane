# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Byte admission limits and the immediate-refusal admission adapter."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vane.execution.udf_admission import AdmissionAuthority, AdmissionLease
from vane.execution.udf_resource_usage import UnitResourceActivity

if TYPE_CHECKING:
    from collections.abc import Callable

    from vane.execution.local_resource_graph import LocalResourceUnitContext
    from vane.execution.udf_data_lease import DataTaskReservation, QueryDataScope


@dataclass(frozen=True)
class DataAdmissionWaitLimits:
    max_queued_tasks: int
    queue_timeout: float = 30.0

    def __post_init__(self) -> None:
        if type(self.max_queued_tasks) is not int or self.max_queued_tasks < 0:
            raise ValueError("max_queued_tasks must be a non-negative integer")
        value = self.queue_timeout
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("queue_timeout must be finite and positive")


@dataclass(frozen=True)
class DataAdmissionLimits:
    max_bytes: int
    max_task_input_bytes: int
    max_task_output_bytes: int
    unit_reservation_ratio: float | None = None
    wait: DataAdmissionWaitLimits | None = None

    def __post_init__(self) -> None:
        if self.wait is not None and not isinstance(self.wait, DataAdmissionWaitLimits):
            raise TypeError("wait must be DataAdmissionWaitLimits")
        for name in ("max_bytes", "max_task_input_bytes", "max_task_output_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.task_bytes > self.max_bytes:
            raise ValueError("data limit must fit one task's input and output reservations")
        ratio = self.unit_reservation_ratio
        if ratio is not None and (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or not 0 <= ratio <= 1
        ):
            raise ValueError("unit_reservation_ratio must be a finite number between zero and one")

    @property
    def task_bytes(self) -> int:
        return self.max_task_input_bytes + self.max_task_output_bytes


class DataAdmissionCapacityError(RuntimeError):
    """No complete task envelope fits; retry after releasing query resources."""

    def __init__(
        self,
        *,
        requested: int,
        usage: int,
        limit: int,
        owner: str,
        reason: str | None = None,
        resource_unit_id: str | None = None,
    ) -> None:
        self.requested = requested
        self.usage = usage
        self.limit = limit
        self.owner = owner
        self.reason = reason
        self.resource_unit_id = resource_unit_id
        self._admission_request: tuple[int, int] | None = None
        super().__init__(
            f"{owner} data admission capacity exceeded: requested={requested}, usage={usage}, limit={limit}; "
            "release retained results and query resources before retrying"
            + (f"; reason={reason}, resource_unit_id={resource_unit_id}" if reason is not None else "")
        )


class DataBatchTooLarge(ValueError):
    """The exact IPC bytes exceed a task's declared input or output bound."""

    def __init__(self, role: str, requested: int, limit: int) -> None:
        super().__init__(f"UDF {role} batch exceeds data limit: requested={requested}, limit={limit}")


class DataAdmissionQueueFull(RuntimeError):
    """The bounded byte-admission queue has no remaining entry."""


class DataAdmissionTimeout(TimeoutError):
    """A complete task envelope could not be admitted before its deadline."""


class DataAdmissionProgressError(ValueError):
    """The hard budget cannot protect one complete envelope per plan UDF."""


class DataAdmissionAuthority:
    """Reserve bytes only after fair task/pool arbitration succeeds.

    A refusal gives back the unused execution grant immediately. Never queue a
    byte waiter holding execution capacity or inputs needed by another query.
    The authority contains no failed-request exception/traceback cache.
    """

    def __init__(
        self,
        base: AdmissionAuthority,
        query: QueryDataScope,
        *,
        resource_unit: LocalResourceUnitContext | None = None,
        activity: UnitResourceActivity | None = None,
    ) -> None:
        self._base = base
        self._query = query
        self._resource_unit = resource_unit
        self._activity = activity
        self._lock = threading.RLock()
        self._reservation: DataTaskReservation | None = None
        self._closed = False
        self._request_generation = 0
        self._wakeup_refusal: tuple[int, int, int, str, str | None, str | None] | None = None

    def request(self, retained_input_bytes: int) -> bool:
        with self._lock:
            self._request_generation += 1
            self._wakeup_refusal = None
        # Backend callbacks may inspect this authority on another thread.
        # Never invoke them under the byte authority's lock.
        accepted = self._base.request(retained_input_bytes)
        self._raise_wakeup_refusal()
        if accepted:
            self._reserve_ready()
        return accepted

    def _raise_wakeup_refusal(self) -> None:
        with self._lock:
            refusal, self._wakeup_refusal = self._wakeup_refusal, None
            generation = self._request_generation
        if refusal is not None:
            requested, usage, limit, owner, reason, unit_id = refusal
            error = DataAdmissionCapacityError(
                requested=requested, usage=usage, limit=limit, owner=owner, reason=reason, resource_unit_id=unit_id
            )
            error._admission_request = (id(self), generation)
            raise error

    def _reserve_ready(self) -> None:
        unused = None
        try:
            with self._lock:
                if self._closed or self._reservation is not None:
                    return
                state = self._base.state()
                if not state["available"]:
                    return
                try:
                    self._reservation = self._query.reserve_task(resource_unit=self._resource_unit)
                except BaseException as exc:
                    if isinstance(exc, DataAdmissionCapacityError):
                        exc._admission_request = (id(self), self._request_generation)
                        if self._activity is not None:
                            self._activity.refuse_bytes(exc.owner)
                    # Take only to retire an unused grant; never submit a worker.
                    unused = self._base.take(int(state["retained_input_bytes"]))
                    raise
        finally:
            if unused is not None:
                unused.release()

    def state(self) -> dict[str, Any]:
        self._raise_wakeup_refusal()
        self._reserve_ready()
        return self._base.state()

    def diagnostic_state(self) -> str:
        # Passive observation must not reserve bytes or consume a cached refusal.
        observe = getattr(self._base, "diagnostic_state", None)
        return observe() if callable(observe) else "unavailable"

    def take(self, retained_input_bytes: int) -> AdmissionLease:
        self._raise_wakeup_refusal()
        self._reserve_ready()
        with self._lock:
            base = self._base.take(retained_input_bytes)
            reservation = self._reservation
            self._reservation = None
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

    def register_wakeup(self, callback: Callable[[], None] | None) -> None:
        self._base.register_wakeup(self.wrap_wakeup(callback))

    def wrap_wakeup(self, callback: Callable[[], None] | None) -> Callable[[], None] | None:
        """Apply the same refusal boundary to pool and transport notifications."""
        if callback is None:
            return None

        def wake() -> None:
            with self._lock:
                generation = self._request_generation
            try:
                callback()
            except DataAdmissionCapacityError as exc:
                # state() may refuse bytes during a reentrant notification.
                # Keep only scalar details for the caller; task admission must
                # not cache this temporary refusal as a permanent callback error.
                with self._lock:
                    # A callback can start before a new request but inspect it
                    # afterward. Attribute state() failures at the point of
                    # refusal, not at the beginning of the notification.
                    request = exc._admission_request or (id(self), generation)
                    if not self._closed and request == (id(self), self._request_generation):
                        self._wakeup_refusal = (
                            exc.requested,
                            exc.usage,
                            exc.limit,
                            exc.owner,
                            exc.reason,
                            exc.resource_unit_id,
                        )

        return wake

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._wakeup_refusal = None
            reservation, self._reservation = self._reservation, None
        try:
            if reservation is not None:
                reservation.release()
        finally:
            self._base.close()
