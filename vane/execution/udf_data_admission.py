# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Strict, non-waiting byte admission composed with task/pool arbitration."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vane.execution.udf_admission import AdmissionAuthority, AdmissionLease

if TYPE_CHECKING:
    from collections.abc import Callable

    from vane.execution.udf_data_lease import DataTaskReservation, QueryDataScope


@dataclass(frozen=True)
class DataAdmissionLimits:
    max_bytes: int
    max_task_input_bytes: int
    max_task_output_bytes: int

    def __post_init__(self) -> None:
        for name in ("max_bytes", "max_task_input_bytes", "max_task_output_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.task_bytes > self.max_bytes:
            raise ValueError("data limit must fit one task's input and output reservations")

    @property
    def task_bytes(self) -> int:
        return self.max_task_input_bytes + self.max_task_output_bytes


class DataAdmissionCapacityError(RuntimeError):
    """No complete task envelope fits; retry after releasing query resources."""

    def __init__(self, *, requested: int, usage: int, limit: int, owner: str) -> None:
        self.requested = requested
        self.usage = usage
        self.limit = limit
        self.owner = owner
        super().__init__(
            f"{owner} data admission capacity exceeded: requested={requested}, usage={usage}, limit={limit}; "
            "release retained results and query resources before retrying"
        )


class DataBatchTooLarge(ValueError):
    """The exact IPC bytes exceed a task's declared input or output bound."""

    def __init__(self, role: str, requested: int, limit: int) -> None:
        super().__init__(f"UDF {role} batch exceeds data limit: requested={requested}, limit={limit}")


class DataAdmissionAuthority:
    """Reserve bytes only after fair task/pool arbitration succeeds.

    A refusal gives back the unused execution grant immediately. Never queue a
    byte waiter holding execution capacity or inputs needed by another query.
    The authority contains no failed-request exception/traceback cache.
    """

    def __init__(self, base: AdmissionAuthority, query: QueryDataScope) -> None:
        self._base = base
        self._query = query
        self._lock = threading.RLock()
        self._reservation: DataTaskReservation | None = None
        self._closed = False
        self._request_generation = 0
        self._wakeup_refusal: tuple[int, int, int, str] | None = None

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
        if refusal is not None:
            requested, usage, limit, owner = refusal
            raise DataAdmissionCapacityError(requested=requested, usage=usage, limit=limit, owner=owner)

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
                    self._reservation = self._query.reserve_task()
                except BaseException:
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

    def register_wakeup(self, callback: Callable[[], None]) -> None:
        self._base.register_wakeup(self.wrap_wakeup(callback))

    def wrap_wakeup(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Apply the same refusal boundary to pool and transport notifications."""

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
                    if not self._closed and generation == self._request_generation:
                        self._wakeup_refusal = (exc.requested, exc.usage, exc.limit, exc.owner)

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
