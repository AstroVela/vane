# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral admission leases for Python UDF executors.

The C++ dispatcher only observes a readiness snapshot.  Resource ownership and
state transitions live in one authority implementation and are transferred to
the executor as an opaque :class:`AdmissionLease`.
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol

from vane.execution.udf_lifecycle import ExecutionCancellationScope


@dataclass
class AdmissionLease:
    """One concrete execution slot owned until ``release`` is called."""

    request_id: str
    retained_input_bytes: int
    lease: dict[str, Any]
    driver: Any | None = None
    _release_callback: Callable[[], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _release_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )
    _released: bool = field(default=False, init=False, repr=False, compare=False)
    _execution_finished_callback: Callable[[], None] | None = field(default=None, repr=False, compare=False)
    _capacity_wait_context: Callable[[ExecutionCancellationScope], AbstractContextManager[None]] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def execution_slot_id(self) -> str:
        return str(self.lease.get("execution_slot_id") or "")

    def complete_execution(self) -> None:
        """Return execution-only capacity while retaining buffered-result ownership."""
        with self._release_lock:
            callback = self._execution_finished_callback
            self._execution_finished_callback = None
            self._capacity_wait_context = None
        if callback is not None:
            callback()

    def suspend_for_wait(self, scope: ExecutionCancellationScope) -> AbstractContextManager[None]:
        """Yield execution capacity during transport waits, retaining ownership.

        The backend must stop user execution before entering and reacquire
        capacity before continuing. Cancellation may skip reacquisition, but
        backend completion still owns the final release.
        """
        with self._release_lock:
            callback = self._capacity_wait_context
        return callback(scope) if callback is not None else nullcontext()

    def release(self) -> None:
        callback: Callable[[], None] | None = None
        with self._release_lock:
            if self._released:
                return
            self._released = True
            callback = self._release_callback
            self._release_callback = None
        try:
            self.complete_execution()
        finally:
            if callback is not None:
                callback()

    def handoff(self) -> None:
        """Transfer cleanup ownership to the submitted execution object."""
        with self._release_lock:
            if self._released:
                raise RuntimeError("cannot hand off an already released admission lease")
            if self._execution_finished_callback is not None or self._capacity_wait_context is not None:
                raise RuntimeError("cannot hand off a lease with local execution cleanup")
            self._released = True
            self._release_callback = None


class AdmissionAuthority(Protocol):
    """The sole owner of admission state for one executor."""

    def request(self, retained_input_bytes: int) -> bool: ...

    def state(self) -> dict[str, Any]: ...

    def take(self, retained_input_bytes: int) -> AdmissionLease: ...

    def register_wakeup(self, callback: Callable[[], None]) -> None: ...

    def close(self) -> None: ...


class AdmissionCapacity(Protocol):
    """Nonblocking backend capacity used by a shared admission policy.

    Capacity notifications must run outside the backend's ledger lock. An
    unsuccessful acquisition must neither reserve capacity nor enqueue work.
    """

    def try_acquire(self, retained_input_bytes: int) -> AdmissionLease | None: ...

    def register_capacity_wakeup(self, callback: Callable[[], None]) -> None: ...

    def state(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _notify_slot_wakeups(wakeups: list[Callable[[], None]]) -> None:
    error: BaseException | None = None
    seen: set[int] = set()
    for wakeup in wakeups:
        if id(wakeup) in seen:
            continue
        seen.add(id(wakeup))
        try:
            wakeup()
        except BaseException as exc:
            if error is None:
                error = exc
    if error is not None:
        raise error


class LocalExecutionSlotPool:
    """The single physical-slot ledger shared by every executor using a pool."""

    def __init__(self, *, max_slots: int, execution_slot_prefix: str) -> None:
        slot_count = int(max_slots)
        if slot_count <= 0:
            raise ValueError("max_slots must be positive")
        prefix = str(execution_slot_prefix).strip()
        if not prefix:
            raise ValueError("execution_slot_prefix must be non-empty")
        self._prefix = prefix
        self._lock = threading.Lock()
        self._available_slots = deque(range(slot_count))
        self._active_slots: dict[str, tuple[int, LocalSlotAdmissionAuthority]] = {}
        self._waiters: deque[LocalSlotAdmissionAuthority] = deque()
        self._authorities: set[LocalSlotAdmissionAuthority] = set()
        self._closed = False

    @property
    def active_lease_count(self) -> int:
        with self._lock:
            return len(self._active_slots)

    def create_authority(self) -> LocalSlotAdmissionAuthority:
        return LocalSlotAdmissionAuthority(slot_pool=self)

    def _assign_slot_locked(self, slot: int) -> Callable[[], None] | None:
        while self._waiters:
            authority = self._waiters.popleft()
            if authority._state != "requested":
                continue
            authority._ready_slot = int(slot)
            authority._state = "ready"
            return authority._wakeup
        if not self._closed:
            self._available_slots.append(int(slot))
        return None

    def _release(self, lease_id: str) -> None:
        wakeup: Callable[[], None] | None = None
        with self._lock:
            owned = self._active_slots.pop(str(lease_id), None)
            if owned is None:
                return
            slot, authority = owned
            authority._active_lease_ids.discard(str(lease_id))
            wakeup = self._assign_slot_locked(slot)
            wakeups = self._capacity_wakeups_locked()
        _notify_slot_wakeups(([wakeup] if wakeup is not None else []) + wakeups)

    def _capacity_wakeups_locked(self) -> list[Callable[[], None]]:
        return [a._capacity_wakeup for a in self._authorities if a._capacity_wakeup is not None]

    def _close_authority(self, authority: LocalSlotAdmissionAuthority) -> None:
        wakeup: Callable[[], None] | None = None
        with self._lock:
            if authority._state == "closed":
                return
            if authority._state == "requested":
                self._waiters = deque(item for item in self._waiters if item is not authority)
            elif authority._state == "ready" and authority._ready_slot is not None:
                wakeup = self._assign_slot_locked(authority._ready_slot)
            authority._state = "closed"
            authority._request_id = ""
            authority._retained_input_bytes = 0
            authority._ready_slot = None
            authority._wakeup = None
            capacity_wakeup = authority._capacity_wakeup
            authority._capacity_wakeup = None
            self._authorities.discard(authority)
            wakeups = self._capacity_wakeups_locked()
            if capacity_wakeup is not None:
                wakeups.append(capacity_wakeup)
        _notify_slot_wakeups(([wakeup] if wakeup is not None else []) + wakeups)

    def close(self) -> None:
        wakeups: list[Callable[[], None]] = []
        with self._lock:
            if self._closed:
                return
            self._closed = True
            wakeups.extend(self._capacity_wakeups_locked())
            authorities = list(self._authorities)
            self._waiters.clear()
            for authority in authorities:
                authority._state = "closed"
                authority._request_id = ""
                authority._retained_input_bytes = 0
                authority._ready_slot = None
                if authority._wakeup is not None:
                    wakeups.append(authority._wakeup)
                authority._wakeup = None
                authority._capacity_wakeup = None
            self._authorities.clear()
            self._available_slots.clear()
        _notify_slot_wakeups(wakeups)


class LocalSlotAdmissionAuthority:
    """Per-dispatcher request state backed by one shared physical-slot pool."""

    def __init__(
        self,
        *,
        max_slots: int | None = None,
        execution_slot_prefix: str | None = None,
        slot_pool: LocalExecutionSlotPool | None = None,
    ) -> None:
        if slot_pool is None:
            if max_slots is None or execution_slot_prefix is None:
                raise ValueError("max_slots and execution_slot_prefix are required without slot_pool")
            slot_pool = LocalExecutionSlotPool(
                max_slots=max_slots,
                execution_slot_prefix=execution_slot_prefix,
            )
        elif max_slots is not None or execution_slot_prefix is not None:
            raise ValueError("slot_pool cannot be combined with max_slots or execution_slot_prefix")
        self._pool = slot_pool
        self._state = "idle"
        self._request_id = ""
        self._retained_input_bytes = 0
        self._ready_slot: int | None = None
        self._sequence = 0
        self._wakeup: Callable[[], None] | None = None
        self._capacity_wakeup: Callable[[], None] | None = None
        self._active_lease_ids: set[str] = set()
        with self._pool._lock:
            if self._pool._closed:
                raise RuntimeError("local execution slot pool is closed")
            self._pool._authorities.add(self)

    @property
    def active_lease_count(self) -> int:
        with self._pool._lock:
            return len(self._active_lease_ids)

    def register_wakeup(self, callback: Callable[[], None]) -> None:
        with self._pool._lock:
            self._wakeup = callback

    def register_capacity_wakeup(self, callback: Callable[[], None]) -> None:
        with self._pool._lock:
            if self._state != "closed":
                self._capacity_wakeup = callback
        # Also covers capacity returned or closed immediately before subscribing.
        callback()

    def try_acquire(self, retained_input_bytes: int) -> AdmissionLease | None:
        retained = int(retained_input_bytes)
        if retained < 0:
            raise ValueError("retained_input_bytes must be >= 0")
        with self._pool._lock:
            if self._state == "closed" or self._pool._closed:
                raise RuntimeError("local admission authority is closed")
            if self._state != "idle":
                raise RuntimeError("cannot combine capacity acquisition with a pending local request")
            if not self._pool._available_slots:
                return None
            self._sequence += 1
            return self._lease_locked(
                self._pool._available_slots.popleft(),
                f"request:local:{self._pool._prefix}:{self._sequence}",
                retained,
            )

    def request(self, retained_input_bytes: int) -> bool:
        retained = int(retained_input_bytes)
        if retained < 0:
            raise ValueError("retained_input_bytes must be >= 0")
        with self._pool._lock:
            if self._state == "closed" or self._pool._closed:
                raise RuntimeError("local admission authority is closed")
            if self._state != "idle":
                return False
            self._sequence += 1
            self._request_id = f"request:local:{self._pool._prefix}:{self._sequence}"
            self._retained_input_bytes = retained
            if self._pool._available_slots:
                self._ready_slot = self._pool._available_slots.popleft()
                self._state = "ready"
            else:
                self._state = "requested"
                self._pool._waiters.append(self)
            return True

    def state(self) -> dict[str, Any]:
        with self._pool._lock:
            return {
                "state": self._state,
                "available": self._state == "ready",
                "retained_input_bytes": self._retained_input_bytes,
            }

    def take(self, retained_input_bytes: int) -> AdmissionLease:
        retained = int(retained_input_bytes)
        with self._pool._lock:
            if self._state != "ready" or self._ready_slot is None:
                raise RuntimeError("local admission lease is not ready")
            if retained != self._retained_input_bytes:
                raise RuntimeError(
                    "local admission retained input bytes do not match: "
                    f"requested={self._retained_input_bytes} consumed={retained}"
                )
            slot = self._ready_slot
            request_id = self._request_id
            self._state = "idle"
            self._request_id = ""
            self._retained_input_bytes = 0
            self._ready_slot = None
            return self._lease_locked(slot, request_id, retained)

    def _lease_locked(self, slot: int, request_id: str, retained: int) -> AdmissionLease:
        lease_id = uuid.uuid4().hex
        execution_slot_id = f"{self._pool._prefix}:{slot}"
        self._pool._active_slots[lease_id] = (slot, self)
        self._active_lease_ids.add(lease_id)
        return AdmissionLease(
            request_id=request_id,
            retained_input_bytes=retained,
            lease={
                "lease_id": lease_id,
                "execution_slot_id": execution_slot_id,
                "slot_index": slot,
            },
            _release_callback=lambda: self._pool._release(lease_id),
        )

    def close(self) -> None:
        self._pool._close_authority(self)


class AdmissionExecutorMixin:
    """Stable wire API consumed by the C++ dispatcher."""

    def _initialize_admission(self, authority: AdmissionAuthority) -> None:
        self._admission_authority = authority

    def request_task_admission(self, retained_input_bytes: int) -> bool:
        return self._admission_authority.request(retained_input_bytes)

    def task_admission_state(self) -> dict[str, Any]:
        return self._admission_authority.state()

    def _take_task_admission(self) -> AdmissionLease:
        state = self._admission_authority.state()
        return self._admission_authority.take(int(state["retained_input_bytes"]))

    def register_wakeup(self, callback: Callable[[], None]) -> None:
        self._admission_authority.register_wakeup(callback)

    def _close_admission(self) -> None:
        self._admission_authority.close()


__all__ = [
    "AdmissionAuthority",
    "AdmissionCapacity",
    "AdmissionExecutorMixin",
    "AdmissionLease",
    "LocalExecutionSlotPool",
    "LocalSlotAdmissionAuthority",
]
