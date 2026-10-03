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
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol

from vane.execution.resources import ResourceVector
from vane.execution.udf_lifecycle import ExecutionCancellationScope
from vane.execution.udf_local_resources import LocalProcessCapacityError


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
    """Process resources shared by local task pools and resident actors.

    Ready grants reserve resources before submission. Tasks waiting on transport
    yield CPU but retain declared heap and their physical owner. Completion
    returns process resources; buffered results retain only their pool slot.
    All pools use this ledger's lock to acquire slots and resources atomically.
    """

    def __init__(self, *, max_slots: int | None, resource_limit: ResourceVector | None = None) -> None:
        if max_slots is not None and int(max_slots) <= 0:
            raise ValueError("max_slots must be positive")
        if resource_limit is None and max_slots is None:
            raise ValueError("execution capacity requires a slot or resource limit")
        self._max_slots = None if max_slots is None else int(max_slots)
        self.resource_limit = resource_limit
        self._resources = ResourceVector()
        self._task_resources = ResourceVector()
        self._residents: dict[str, ResourceVector] = {}
        self._progress: set[LocalTaskProgress] = set()
        self._resuming: deque[_LocalExecutionReservation] = deque()
        self._reserved = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        # Move a pool to the back after every grant, including direct grants.
        self._pools: dict[LocalExecutionSlotPool, None] = {}
        self._dispatching = False
        self._dispatch_requested = False
        self._turn_pool: LocalExecutionSlotPool | None = None
        self._turn_remaining = 0
        self._deferred_guards: set[Callable[[], bool]] = set()

    @property
    def reserved_slots(self) -> int:
        with self._lock:
            return self._reserved

    def _slots_full_locked(self) -> bool:
        return self._max_slots is not None and self._reserved >= self._max_slots

    def _fits_locked(self, resources: ResourceVector) -> bool:
        return self.resource_limit is None or (self._resources + resources).fits_within(self.resource_limit)

    def _protected_heap_locked(self) -> int:
        return sum(query.remaining_heap for query in self._progress)

    def _task_cpu_floor_locked(self) -> float:
        return max((query.cpu_floor for query in self._progress), default=0.0)

    def reserve_task_progress(self, requirements: dict[str, ResourceVector]) -> LocalTaskProgress:
        """Protect one heap reservation per task stage until query cleanup.

        Transport waiters cannot return heap. Every prepared task stage therefore
        needs a protected minimum; extra producer grants may use only surplus.
        CPU is reusable across waits, so only the largest task CPU is protected
        against new resident actors.
        """
        progress = LocalTaskProgress(self, requirements)
        with self._lock:
            resident = sum(self._residents.values(), ResourceVector())
            needed = ResourceVector(
                cpu=resident.cpu + max(self._task_cpu_floor_locked(), progress.cpu_floor),
                heap_bytes=self._resources.heap_bytes + self._protected_heap_locked() + progress.remaining_heap,
            )
            if self.resource_limit is not None and not needed.fits_within(self.resource_limit):
                raise LocalProcessCapacityError("local task progress exceeds available node CPU/heap resource capacity")
            self._progress.add(progress)
        return progress

    def _can_acquire_locked(self, resources: ResourceVector, progress: LocalTaskProgressBinding | None = None) -> bool:
        if progress is not None and progress.query.closed:
            return False
        credit = 0 if progress is None else progress.heap_credit
        protected = ResourceVector(heap_bytes=self._protected_heap_locked() - credit)
        return not self._resuming and not self._slots_full_locked() and self._fits_locked(resources + protected)

    def _return_ready_locked(self, resources: ResourceVector, progress: LocalTaskProgressBinding | None = None) -> None:
        self._reserved -= 1
        self._resources -= resources
        self._task_resources -= resources
        if progress is not None:
            progress.release_locked()
        self._condition.notify_all()

    def reserve_resident(
        self, resources: ResourceVector, cancellation: ExecutionCancellationScope | None = None
    ) -> Callable[[], None]:
        """Reserve actor processes once, until their physical owner closes."""
        with self._condition:
            while True:
                resident = sum(self._residents.values(), ResourceVector())
                if self.resource_limit is None:
                    break
                if not resources.fits_within(self.resource_limit):
                    raise ValueError("local actor processes exceed the node CPU/heap resource capacity")
                protected = ResourceVector(
                    cpu=self._task_cpu_floor_locked(),
                    heap_bytes=sum(query.total_heap for query in self._progress),
                )
                if not (resident + resources + protected).fits_within(self.resource_limit):
                    raise LocalProcessCapacityError(
                        "local actor processes exceed available node CPU/heap resource capacity"
                    )
                # A new resident must not claim CPU temporarily yielded by a
                # task which still needs that CPU to finish and release input.
                if (
                    resident
                    + self._task_resources
                    + resources
                    + ResourceVector(heap_bytes=self._protected_heap_locked())
                ).fits_within(self.resource_limit):
                    break
                if cancellation is not None:
                    cancellation.raise_if_cancelled("local actor process resources")
                self._condition.wait(timeout=0.1)
            if cancellation is not None:
                cancellation.raise_if_cancelled("local actor process resources")
            token = uuid.uuid4().hex
            self._residents[token] = resources
            self._resources += resources

        def release() -> None:
            with self._condition:
                owned = self._residents.pop(token, None)
                if owned is None:
                    return
                self._resources -= owned
                self._condition.notify_all()
                wakeups = self._dispatch_locked()
            _notify_slot_wakeups(wakeups)

        return release

    def resource_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "limit": None if self.resource_limit is None else self.resource_limit.to_dict(),
                "usage": self._resources.to_dict(),
                "resident": sum(self._residents.values(), ResourceVector()).to_dict(),
                "resuming_tasks": len(self._resuming),
                "task_progress": {
                    "queries": len(self._progress),
                    "protected_heap_bytes": self._protected_heap_locked(),
                    "cpu_floor": self._task_cpu_floor_locked(),
                },
            }

    def _resume_locked(self) -> None:
        while self._resuming:
            reservation = self._resuming[0]
            if not self._fits_locked(reservation.cpu):
                break
            self._resuming.popleft()
            self._resources += reservation.cpu
            reservation.suspended = False
        self._condition.notify_all()

    def _dispatch_locked(self) -> list[Callable[[], None]]:
        self._resume_locked()
        self._dispatch_requested = True
        self._deferred_guards.clear()
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
                    if self._slots_full_locked() or (not granted and not self._dispatch_requested):
                        self._dispatching = False
                        self._deferred_guards.clear()
                        finished = True
                        break
                    self._dispatch_requested = False
        finally:
            if not finished:
                with self._lock:
                    self._dispatching = False
                    self._turn_pool = None
                    self._turn_remaining = 0
                    self._deferred_guards.clear()
        if error is not None:
            raise error

    def _complete_execution(self, reservation: _LocalExecutionReservation) -> None:
        with self._lock:
            if reservation.finished:
                return
            reservation.finished = True
            if reservation in self._resuming:
                self._resuming.remove(reservation)
            self._reserved -= 1
            self._task_resources -= reservation.resources
            if reservation.progress is not None:
                reservation.progress.release_locked()
            self._resources -= (
                reservation.resources - reservation.cpu if reservation.suspended else reservation.resources
            )
            self._condition.notify_all()
            wakeups = self._dispatch_locked()
        _notify_slot_wakeups(wakeups)


class LocalTaskProgress:
    """Query-owned minimum heap for all prepared subprocess task stages."""

    def __init__(self, capacity: LocalExecutionCapacity, requirements: dict[str, ResourceVector]) -> None:
        self.capacity = capacity
        self.requirements = dict(requirements)
        self.active = {node: 0 for node in requirements}
        self.total_heap = sum(resources.heap_bytes for resources in requirements.values())
        self.cpu_floor = max((resources.cpu for resources in requirements.values()), default=0.0)
        self.closed = False

    @property
    def remaining_heap(self) -> int:
        return sum(resources.heap_bytes for node, resources in self.requirements.items() if not self.active[node])

    def bind(self, node_id: str) -> LocalTaskProgressBinding:
        if node_id not in self.requirements:
            raise ValueError("task progress binding requires a prepared task node")
        return LocalTaskProgressBinding(self, node_id)

    def shutdown(self, *, kill: bool = False) -> None:
        with self.capacity._lock:
            if self.closed:
                return
            self.closed = True
            self.capacity._progress.remove(self)
            wakeups = self.capacity._dispatch_locked()
        _notify_slot_wakeups(wakeups)

    def cleanup_pending(self) -> bool:
        return False


@dataclass(frozen=True)
class LocalTaskProgressBinding:
    query: LocalTaskProgress
    node_id: str

    @property
    def heap_credit(self) -> int:
        return (
            0
            if self.query.closed or self.query.active[self.node_id]
            else self.query.requirements[self.node_id].heap_bytes
        )

    def acquire_locked(self) -> None:
        self.query.active[self.node_id] += 1

    def release_locked(self) -> None:
        self.query.active[self.node_id] -= 1


class _LocalExecutionReservation:
    def __init__(
        self,
        capacity: LocalExecutionCapacity,
        resources: ResourceVector,
        progress: LocalTaskProgressBinding | None = None,
    ) -> None:
        self.capacity = capacity
        self.resources = resources
        self.progress = progress
        self.cpu = ResourceVector(cpu=resources.cpu)
        self.suspended = False
        self.finished = False

    def complete(self) -> None:
        self.capacity._complete_execution(self)

    @contextmanager
    def suspend(self, scope: ExecutionCancellationScope) -> Iterator[None]:
        capacity = self.capacity
        with capacity._condition:
            if self.finished or self.suspended:
                raise RuntimeError("cannot suspend inactive local execution resources")
            self.suspended = True
            capacity._resources -= self.cpu
            capacity._condition.notify_all()
            wakeups = capacity._dispatch_locked()
        _notify_slot_wakeups(wakeups)
        yield
        # Failed transport goes directly to cleanup without reclaiming CPU.
        try:
            with capacity._condition:
                if not self.finished and not scope.is_set():
                    capacity._resuming.append(self)
                    capacity._resume_locked()
                try:
                    while not self.finished and self.suspended and not scope.is_set():
                        capacity._condition.wait(timeout=0.1)
                finally:
                    if self in capacity._resuming:
                        capacity._resuming.remove(self)
                    wakeups = capacity._dispatch_locked()
            _notify_slot_wakeups(wakeups)
            scope.raise_if_cancelled("local execution resource resumption")
            if self.finished:
                raise RuntimeError("cannot resume completed local execution resources")
        finally:
            with capacity._condition:
                if self in capacity._resuming:
                    capacity._resuming.remove(self)


class LocalExecutionSlotPool:
    """The single physical-slot ledger shared by every executor using a pool."""

    def __init__(
        self,
        *,
        max_slots: int,
        execution_slot_prefix: str,
        execution_capacity: LocalExecutionCapacity | None = None,
        resources: ResourceVector = ResourceVector(),
    ) -> None:
        slot_count = int(max_slots)
        if slot_count <= 0:
            raise ValueError("max_slots must be positive")
        prefix = str(execution_slot_prefix).strip()
        if not prefix:
            raise ValueError("execution_slot_prefix must be non-empty")
        self._prefix = prefix
        self._execution_capacity = execution_capacity
        self._resources = resources
        if execution_capacity is not None and execution_capacity.resource_limit is not None:
            if not resources.fits_within(execution_capacity.resource_limit):
                raise ValueError("local task exceeds the node CPU/heap resource capacity")
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
        self._deferred_guards: set[Callable[[], bool]] = set()
        self._closed = False
        if execution_capacity is not None:
            with self._lock:
                execution_capacity._pools[self] = None

    @property
    def active_lease_count(self) -> int:
        with self._lock:
            return len(self._active_slots)

    def create_authority(self) -> LocalSlotAdmissionAuthority:
        return self.create_task_authority()

    def create_task_authority(self, progress: LocalTaskProgressBinding | None = None) -> LocalSlotAdmissionAuthority:
        return LocalSlotAdmissionAuthority(slot_pool=self, progress=progress)

    def _try_take_slot_locked(
        self, source: int = 0, guard: Callable[[], bool] | None = None, progress: LocalTaskProgressBinding | None = None
    ) -> int | None:
        capacity = self._execution_capacity
        if not self._available_slots or (
            capacity is not None and not capacity._can_acquire_locked(self._resources, progress)
        ):
            return None
        if capacity is not None:
            if capacity._dispatch_requested and not capacity._dispatching:
                return None
            if capacity._dispatching and (
                capacity._turn_pool is not self or capacity._turn_remaining == 0 or not self._dispatching
            ):
                # A request or policy allowance can become eligible after its
                # pool's turn. Revisit it before the arbiter goes idle.
                # A byte-gated source can remain ineligible on its own turn.
                # Repeated off-turn probes must not keep the dispatcher alive
                # forever. Revisit once per guard until a real capacity event
                # or successful grant changes the scheduling state.
                if guard is None or guard not in capacity._deferred_guards:
                    if guard is not None:
                        capacity._deferred_guards.add(guard)
                    capacity._dispatch_requested = True
                return None
        if self._dispatch_requested and not self._dispatching:
            return None
        if self._dispatching and (source != self._turn_source or self._turn_remaining == 0):
            if guard is None or guard not in self._deferred_guards:
                if guard is not None:
                    self._deferred_guards.add(guard)
                self._dispatch_requested = True
            return None
        # A nonblocking resource guard commits only once the physical slot and
        # this source's fair turn are available. It must not invoke callbacks.
        if guard is not None and not guard():
            return None
        if self._dispatching:
            self._turn_remaining -= 1
        self._deferred_guards.clear()
        self._sources[source] = self._sources.pop(source)
        if capacity is not None:
            capacity._deferred_guards.clear()
            capacity._reserved += 1
            capacity._resources += self._resources
            capacity._task_resources += self._resources
            if progress is not None:
                progress.acquire_locked()
            if capacity._dispatching:
                capacity._turn_remaining -= 1
            capacity._pools.pop(self)
            capacity._pools[self] = None
        return self._available_slots.popleft()

    def _dispatch_waiters_locked(self) -> list[Callable[[], None]]:
        wakeups: list[Callable[[], None]] = []
        for authority in tuple(self._waiters):
            if self._closed:
                break
            slot = self._try_take_slot_locked(progress=authority._progress)
            if slot is None:
                continue
            self._waiters.remove(authority)
            authority._ready_slot = int(slot)
            authority._state = "ready"
            if authority._wakeup is not None:
                wakeups.append(authority._wakeup)
        return wakeups

    def _dispatch_capacity_locked(self) -> list[Callable[[], None]]:
        self._deferred_guards.clear()
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
                        capacity._resuming
                        or capacity._slots_full_locked()
                        or not capacity._fits_locked(self._resources)
                        or capacity._turn_remaining == 0
                    )
                    if (
                        self._closed
                        or not self._available_slots
                        or global_turn_finished
                        or (not granted and not self._dispatch_requested)
                    ):
                        self._dispatching = False
                        self._deferred_guards.clear()
                        finished = True
                        break
                    self._dispatch_requested = False
        finally:
            if not finished:
                with self._lock:
                    self._dispatching = False
                    self._turn_source = None
                    self._turn_remaining = 0
                    self._deferred_guards.clear()
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
                    self._execution_capacity._return_ready_locked(self._resources, authority._progress)
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
                    self._execution_capacity._return_ready_locked(self._resources, authority._progress)
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
        progress: LocalTaskProgressBinding | None = None,
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
        self._progress = progress
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
            if progress is not None and (
                progress.query.capacity is not self._pool._execution_capacity
                or progress.query.closed
                or progress.query not in progress.query.capacity._progress
                or progress.query.requirements.get(progress.node_id) != self._pool._resources
            ):
                raise ValueError("task progress binding does not match its pool resources")
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
        return self.try_acquire_if(retained_input_bytes, None)

    def try_acquire_if(self, retained_input_bytes: int, guard: Callable[[], bool] | None) -> AdmissionLease | None:
        """Commit an additional reservation together with physical capacity."""
        retained = int(retained_input_bytes)
        if retained < 0:
            raise ValueError("retained_input_bytes must be >= 0")
        with self._pool._lock:
            if self._state == "closed" or self._pool._closed:
                raise RuntimeError("local admission authority is closed")
            if self._state != "idle":
                raise RuntimeError("cannot combine capacity acquisition with a pending local request")
            source = id(self._capacity_wakeup) if self._capacity_wakeup is not None else 0
            slot = self._pool._try_take_slot_locked(source, guard, self._progress)
            if slot is None:
                return None
            self._sequence += 1
            return self._lease_locked(
                slot,
                f"request:local:{self._pool._prefix}:{self._sequence}",
                retained,
            )

    def notify_capacity(self) -> None:
        """Recheck all sources through the existing pool/global fair arbiter."""
        with self._pool._lock:
            wakeups = self._pool._dispatch_capacity_locked()
        _notify_slot_wakeups(wakeups)

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
            self._ready_slot = self._pool._try_take_slot_locked(progress=self._progress)
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

    def diagnostic_state(self) -> str:
        with self._pool._lock:
            return self._state

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
        capacity = self._pool._execution_capacity
        reservation = (
            None if capacity is None else _LocalExecutionReservation(capacity, self._pool._resources, self._progress)
        )
        return AdmissionLease(
            request_id=request_id,
            retained_input_bytes=retained,
            lease={
                "lease_id": lease_id,
                "execution_slot_id": execution_slot_id,
                "slot_index": slot,
            },
            _release_callback=lambda: self._pool._release(lease_id),
            _execution_finished_callback=None if reservation is None else reservation.complete,
            _capacity_wait_context=None if reservation is None else reservation.suspend,
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
