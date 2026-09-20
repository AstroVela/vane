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

    def register_wakeup(self, callback: Callable[[], None] | None) -> None: ...

    def close(self) -> None: ...


class AdmissionCapacity(Protocol):
    """Nonblocking backend capacity used by a shared admission policy.

    Capacity notifications must run outside the backend's ledger lock. An
    unsuccessful acquisition must neither reserve capacity nor enqueue work.
    Reuse one callback object across a policy's authorities so adapters can
    identify that policy when arbitrating between competing sources.
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


class LocalExecutionCapacity:
    """Execution-only capacity shared by pools using the same worker executor.

    Ready grants reserve capacity before submission. Suspended tasks retain it
    until backend completion, whereas buffered results only retain a pool slot.
    All pools use this ledger's lock to acquire both resources atomically.
    """

    def __init__(self, *, max_slots: int) -> None:
        if int(max_slots) <= 0:
            raise ValueError("max_slots must be positive")
        self._max_slots = int(max_slots)
        self._reserved = 0
        self._lock = threading.Lock()
        # Move a pool to the back after every grant, including direct grants.
        self._pools: dict[LocalExecutionSlotPool, None] = {}
        self._dispatching = False
        self._dispatch_requested = False
        self._turn_pool: LocalExecutionSlotPool | None = None
        self._turn_remaining = 0

    @property
    def reserved_slots(self) -> int:
        with self._lock:
            return self._reserved

    def _dispatch_locked(self) -> list[Callable[[], None]]:
        self._dispatch_requested = True
        return [self._dispatch]

    def _dispatch(self) -> None:
        with self._lock:
            if self._dispatching:
                return
            self._dispatching = True
            self._dispatch_requested = False
        error: BaseException | None = None
        finished = False
        try:
            while True:
                with self._lock:
                    pools = list(self._pools)
                granted = False
                for pool in pools:
                    with self._lock:
                        if pool._closed:
                            continue
                        self._turn_pool = pool
                        self._turn_remaining = 1
                        pool._dispatch_requested = True
                    # Each pool arbitrates between ordinary requests and runtime
                    # policies before consuming its global turn.
                    try:
                        pool._dispatch()
                    except BaseException as exc:
                        if error is None:
                            error = exc
                    with self._lock:
                        granted |= self._turn_remaining == 0
                with self._lock:
                    self._turn_pool = None
                    self._turn_remaining = 0
                    if self._reserved >= self._max_slots or (not granted and not self._dispatch_requested):
                        self._dispatching = False
                        finished = True
                        break
                    self._dispatch_requested = False
        finally:
            if not finished:
                with self._lock:
                    self._dispatching = False
                    self._turn_pool = None
                    self._turn_remaining = 0
        if error is not None:
            raise error

    def _complete_execution(self) -> None:
        with self._lock:
            self._reserved -= 1
            wakeups = self._dispatch_locked()
        _notify_slot_wakeups(wakeups)


class LocalExecutionSlotPool:
    """The single physical-slot ledger shared by every executor using a pool."""

    def __init__(
        self,
        *,
        max_slots: int,
        execution_slot_prefix: str,
        execution_capacity: LocalExecutionCapacity | None = None,
    ) -> None:
        slot_count = int(max_slots)
        if slot_count <= 0:
            raise ValueError("max_slots must be positive")
        prefix = str(execution_slot_prefix).strip()
        if not prefix:
            raise ValueError("execution_slot_prefix must be non-empty")
        self._prefix = prefix
        self._execution_capacity = execution_capacity
        self._lock = execution_capacity._lock if execution_capacity is not None else threading.Lock()
        self._available_slots = deque(range(slot_count))
        self._active_slots: dict[str, tuple[int, LocalSlotAdmissionAuthority]] = {}
        self._waiters: deque[LocalSlotAdmissionAuthority] = deque()
        self._authorities: set[LocalSlotAdmissionAuthority] = set()
        # Source 0 is the ordinary FIFO. A shared callback identifies one
        # runtime policy, regardless of how many queries it has in this pool.
        self._sources: dict[int, Callable[[], None] | None] = {0: None}
        self._dispatching = False
        self._dispatch_requested = False
        self._turn_source: int | None = None
        self._turn_remaining = 0
        self._closed = False
        if execution_capacity is not None:
            with self._lock:
                execution_capacity._pools[self] = None

    @property
    def active_lease_count(self) -> int:
        with self._lock:
            return len(self._active_slots)

    def create_authority(self) -> LocalSlotAdmissionAuthority:
        return LocalSlotAdmissionAuthority(slot_pool=self)

    def _try_take_slot_locked(self, source: int = 0) -> int | None:
        capacity = self._execution_capacity
        if not self._available_slots or (capacity is not None and capacity._reserved >= capacity._max_slots):
            return None
        if capacity is not None:
            if capacity._dispatch_requested and not capacity._dispatching:
                return None
            if capacity._dispatching and (
                capacity._turn_pool is not self or capacity._turn_remaining == 0 or not self._dispatching
            ):
                # A request or policy allowance can become eligible after its
                # pool's turn. Revisit it before the arbiter goes idle.
                capacity._dispatch_requested = True
                return None
        if self._dispatch_requested and not self._dispatching:
            return None
        if self._dispatching and (source != self._turn_source or self._turn_remaining == 0):
            self._dispatch_requested = True
            return None
        if self._dispatching:
            self._turn_remaining -= 1
        self._sources[source] = self._sources.pop(source)
        if capacity is not None:
            capacity._reserved += 1
            if capacity._dispatching:
                capacity._turn_remaining -= 1
            capacity._pools.pop(self)
            capacity._pools[self] = None
        return self._available_slots.popleft()

    def _dispatch_waiters_locked(self) -> list[Callable[[], None]]:
        wakeups: list[Callable[[], None]] = []
        while self._waiters and not self._closed:
            slot = self._try_take_slot_locked()
            if slot is None:
                break
            authority = self._waiters.popleft()
            authority._ready_slot = int(slot)
            authority._state = "ready"
            if authority._wakeup is not None:
                wakeups.append(authority._wakeup)
        return wakeups

    def _dispatch_capacity_locked(self) -> list[Callable[[], None]]:
        if self._execution_capacity is not None:
            return self._execution_capacity._dispatch_locked()
        self._dispatch_requested = True
        return [self._dispatch]

    def _dispatch(self) -> None:
        with self._lock:
            if self._closed or self._dispatching:
                return
            self._dispatching = True
            self._dispatch_requested = False
        error: BaseException | None = None
        finished = False
        try:
            while True:
                with self._lock:
                    sources = list(self._sources.items())
                granted = False
                for source, callback in sources:
                    with self._lock:
                        if self._closed or source not in self._sources:
                            continue
                        self._turn_source = source
                        self._turn_remaining = 1
                        wakeups = self._dispatch_waiters_locked() if callback is None else [callback]
                    # Runtime callbacks acquire their policy locks and may try
                    # other pools. Neither ledger lock may be held here.
                    try:
                        _notify_slot_wakeups(wakeups)
                    except BaseException as exc:
                        if error is None:
                            error = exc
                    with self._lock:
                        granted |= self._turn_remaining == 0
                with self._lock:
                    self._turn_source = None
                    self._turn_remaining = 0
                    capacity = self._execution_capacity
                    global_turn_finished = capacity is not None and (
                        capacity._reserved >= capacity._max_slots or capacity._turn_remaining == 0
                    )
                    if (
                        self._closed
                        or not self._available_slots
                        or global_turn_finished
                        or (not granted and not self._dispatch_requested)
                    ):
                        self._dispatching = False
                        finished = True
                        break
                    self._dispatch_requested = False
        finally:
            if not finished:
                with self._lock:
                    self._dispatching = False
                    self._turn_source = None
                    self._turn_remaining = 0
        if error is not None:
            raise error

    def _release(self, lease_id: str) -> None:
        with self._lock:
            owned = self._active_slots.pop(str(lease_id), None)
            if owned is None:
                return
            slot, authority = owned
            authority._active_lease_ids.discard(str(lease_id))
            if not self._closed:
                self._available_slots.append(slot)
            wakeups = self._dispatch_capacity_locked()
        _notify_slot_wakeups(wakeups)

    def _capacity_wakeups_locked(self) -> list[Callable[[], None]]:
        return [callback for callback in self._sources.values() if callback is not None]

    def _retire_source_locked(self, callback: Callable[[], None] | None) -> None:
        if callback is not None and not any(a._capacity_wakeup is callback for a in self._authorities):
            self._sources.pop(id(callback), None)

    def _close_authority(self, authority: LocalSlotAdmissionAuthority) -> None:
        with self._lock:
            if authority._state == "closed":
                return
            if authority._state == "requested":
                self._waiters = deque(item for item in self._waiters if item is not authority)
            elif authority._state == "ready" and authority._ready_slot is not None:
                self._available_slots.append(authority._ready_slot)
                if self._execution_capacity is not None:
                    self._execution_capacity._reserved -= 1
            authority._state = "closed"
            authority._request_id = ""
            authority._retained_input_bytes = 0
            authority._ready_slot = None
            authority._wakeup = None
            capacity_wakeup = authority._capacity_wakeup
            authority._capacity_wakeup = None
            self._authorities.discard(authority)
            self._retire_source_locked(capacity_wakeup)
            wakeups = self._dispatch_capacity_locked()
            if capacity_wakeup is not None:
                wakeups.append(capacity_wakeup)
        _notify_slot_wakeups(wakeups)

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
                if authority._ready_slot is not None and self._execution_capacity is not None:
                    self._execution_capacity._reserved -= 1
                authority._state = "closed"
                authority._request_id = ""
                authority._retained_input_bytes = 0
                authority._ready_slot = None
                if authority._wakeup is not None:
                    wakeups.append(authority._wakeup)
                authority._wakeup = None
                authority._capacity_wakeup = None
            self._authorities.clear()
            self._sources.clear()
            self._available_slots.clear()
            if self._execution_capacity is not None:
                self._execution_capacity._pools.pop(self, None)
                wakeups.extend(self._execution_capacity._dispatch_locked())
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

    def register_wakeup(self, callback: Callable[[], None] | None) -> None:
        with self._pool._lock:
            self._wakeup = callback

    def register_capacity_wakeup(self, callback: Callable[[], None]) -> None:
        with self._pool._lock:
            if self._state != "closed":
                previous = self._capacity_wakeup
                self._capacity_wakeup = callback
                self._pool._retire_source_locked(previous)
                self._pool._sources.setdefault(id(callback), callback)
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
            source = id(self._capacity_wakeup) if self._capacity_wakeup is not None else 0
            slot = self._pool._try_take_slot_locked(source)
            if slot is None:
                return None
            self._sequence += 1
            return self._lease_locked(
                slot,
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
            self._ready_slot = self._pool._try_take_slot_locked()
            if self._ready_slot is not None:
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
            _execution_finished_callback=(
                self._pool._execution_capacity._complete_execution
                if self._pool._execution_capacity is not None
                else None
            ),
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

    def register_wakeup(self, callback: Callable[[], None] | None) -> None:
        self._admission_authority.register_wakeup(callback)

    def _close_admission(self) -> None:
        self._admission_authority.close()


__all__ = [
    "AdmissionAuthority",
    "AdmissionCapacity",
    "AdmissionExecutorMixin",
    "AdmissionLease",
    "LocalExecutionCapacity",
    "LocalExecutionSlotPool",
    "LocalSlotAdmissionAuthority",
]
