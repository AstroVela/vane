# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded local output storage, independent of transport admission credits.

The parent allocates immutable blocks in one arena. A worker receives a write
grant or a pre-registered read lease; neither an input ACK nor task completion
releases that lease. A separate socket delivers last-buffer notifications even
while the task control socket is waiting for output admission.
"""

from __future__ import annotations

import atexit
import gc
import mmap
import os
import queue
import select
import socket
import struct
import threading
import uuid
import weakref
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from itertools import count
from typing import Any

from vane.execution.udf_lifecycle import ExecutionCancellationScope

_RELEASE = struct.Struct("!Q")
_ALIGNMENT = 64
_registry_pid = os.getpid()
_registry_lock = threading.RLock()
_stores: dict[str, LocalShmStore] = {}
_current_store: LocalShmStore | None = None
_worker_client: WorkerShmClient | None = None
_gc_lock = threading.RLock()
_gc_holds = 0
_gc_was_enabled = False


def _pause_automatic_gc() -> None:
    global _gc_holds, _gc_was_enabled
    _gc_lock.acquire()
    try:
        if not _gc_holds:
            _gc_was_enabled = gc.isenabled()
            gc.disable()
        # Count waiters as well as owners: releasing one critical section must
        # not re-enable collection inside another thread's critical section.
        _gc_holds += 1
    finally:
        _gc_lock.release()


def _resume_automatic_gc() -> None:
    global _gc_holds
    # Bind release while GC is still paused. Re-enabling it must be followed
    # only by the primitive unlock, not an allocating context-manager exit.
    unlock = _gc_lock.release
    _gc_lock.acquire()
    try:
        _gc_holds -= 1
        if not _gc_holds and _gc_was_enabled:
            gc.enable()
    finally:
        unlock()


class _ForkLock:
    """Keep automatic cyclic finalizers outside the allocation mutation lock."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def acquire(self) -> None:
        # GC can release Arrow owners and acquire store/lease locks. Pause it
        # before taking this inner lock, including recursive acquisitions.
        _pause_automatic_gc()
        try:
            self._lock.acquire()
        except BaseException:
            _resume_automatic_gc()
            raise

    def release(self) -> None:
        self._lock.release()
        _resume_automatic_gc()

    def __enter__(self) -> _ForkLock:
        self.acquire()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.release()

    def _is_owned(self) -> bool:
        # CPython exposes this ownership probe outside the public type stub.
        return self._lock._is_owned()  # type: ignore[attr-defined]


# Always acquire this after registry/store locks. Fork preparation takes only
# this lock, so a different thread holding a public store lock cannot block it.
_fork_lock = _ForkLock()
_fork_holds: set[_ForkHold] = set()
_pending_fork: _ForkHold | None = None
_inherited_fork_writers: set[int] = set()
_worker_clients: weakref.WeakSet[WorkerShmClient] = weakref.WeakSet()


class _ForkHold:
    def __init__(self) -> None:
        self.stores: list[LocalShmStore] = []
        self.clients: list[WorkerShmClient] = []
        self.read_fd: int | None = None
        self.write_fd: int | None = None
        self.alive = True
        self.watching = False

    def is_alive(self) -> bool:
        # Called under _fork_lock. An incomplete fork notification setup must
        # retain its pins: at-fork callback errors do not cancel os.fork().
        if not self.alive or self.read_fd is None:
            return self.alive
        poller = select.poll()
        poller.register(self.read_fd, select.POLLIN | select.POLLHUP)
        if poller.poll(0):
            if os.read(self.read_fd, 1) != b"":
                raise RuntimeError("unexpected fork lifetime notification")
            self.alive = False
            if not self.watching:
                self.close_reader()
        return self.alive

    def close_reader(self) -> None:
        if self.read_fd is not None:
            os.close(self.read_fd)
            self.read_fd = None
        _fork_holds.discard(self)


def _pin_stores_before_fork() -> None:
    global _pending_fork
    _fork_lock.acquire()
    hold = _ForkHold()
    _pending_fork = hold
    for store in _stores.values():
        pinned = False
        for generation, entry in store._live.items():
            if entry.refs:
                entry.forks.append(hold)
                store._forked[generation] = entry
                pinned = True
        if pinned:
            hold.stores.append(store)
    for client in _worker_clients:
        if client._live_reads:
            for token in client._live_reads:
                client._fork_reads.setdefault(token, set()).add(hold)
            hold.clients.append(client)
    if hold.stores or hold.clients:
        _fork_holds.add(hold)
        # Pins are installed before allocating descriptors. If pipe creation
        # fails, they remain owned until process exit instead of risking reuse.
        hold.read_fd, hold.write_fd = os.pipe()


def _watch_fork_exit(hold: _ForkHold) -> None:
    assert hold.read_fd is not None
    if os.read(hold.read_fd, 1) != b"":
        raise RuntimeError("unexpected fork lifetime notification")
    with _fork_lock:
        hold.alive = False
        hold.close_reader()
        for client in hold.clients:
            client._collect_fork_reads_locked()
    error: Exception | None = None
    for store in hold.stores:
        try:
            store.drain_if_idle()
        except Exception as exc:
            # Stores retain failed cleanup for retry; try every owned store.
            if error is None:
                error = exc
    if error is not None:
        raise error


def _start_fork_watcher(hold: _ForkHold) -> None:
    with _fork_lock:
        if hold.watching or hold.read_fd is None:
            return
        hold.watching = True
        try:
            threading.Thread(target=_watch_fork_exit, args=(hold,), name="vane-shm-fork-exit", daemon=True).start()
        except BaseException:
            # Allocation/drain and input-release paths can still observe EOF.
            hold.watching = False
            raise


def _resume_parent_after_fork() -> None:
    global _pending_fork
    try:
        hold, _pending_fork = _pending_fork, None
        if hold is not None and hold.write_fd is not None:
            os.close(hold.write_fd)
            hold.write_fd = None
            _start_fork_watcher(hold)
    finally:
        _fork_lock.release()


def _reset_registry_after_fork() -> None:
    global _registry_pid, _registry_lock, _stores, _current_store, _fork_lock, _fork_holds, _pending_fork
    global _worker_clients, _gc_lock, _gc_holds
    # Other parent threads may have been waiting for the mutation lock, or
    # holding the GC-state lock. Neither their locks nor their hold count
    # survives in the child. Keep one pause until registry reset completes.
    _gc_lock = threading.RLock()
    _gc_holds = 1
    _fork_lock = _ForkLock()
    try:
        if _pending_fork is not None and _pending_fork.write_fd is not None:
            # Keep the writer through normal exit, _exit, signals and further
            # forks. CLOEXEC releases it when exec discards inherited mappings.
            _inherited_fork_writers.add(_pending_fork.write_fd)
        for hold in _fork_holds:
            if hold.read_fd is not None:
                os.close(hold.read_fd)
        _fork_holds = set()
        _pending_fork = None
        _worker_clients = weakref.WeakSet()
        # Free lists and locks are process-local even though their backing mmap
        # is shared. Discard inherited state without closing parent arenas.
        _registry_lock = threading.RLock()
        _stores = {}
        _current_store = None
        _registry_pid = os.getpid()
    finally:
        _resume_automatic_gc()


def _require_registry_owner() -> None:
    # Reject unregistered fork paths before touching an inherited lock.
    if _registry_pid != os.getpid():
        raise RuntimeError("shared-memory registry belongs to a different process")


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_pin_stores_before_fork,
        after_in_parent=_resume_parent_after_fork,
        after_in_child=_reset_registry_after_fork,
    )


class LocalShmStoreCapacityError(RuntimeError):
    """Live physical buffers, rather than transport credits, fill the store."""


def _allocation_capacity(size: int) -> int:
    if type(size) is not int or size <= 0:
        raise ValueError("shared-memory allocation size must be positive")
    # Sixteen size classes per power-of-two range absorb small IPC metadata
    # changes without stranding an almost-large-enough free slot.
    alignment = 1 << max(6, (size - 1).bit_length() - 5)
    return (size + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class ShmAllocation:
    store_id: str
    shm_name: str
    offset: int
    size: int
    generation: int

    @property
    def identity(self) -> str:
        return f"{self.shm_name}:{self.offset}:{self.generation}"

    def descriptor(self) -> dict[str, Any]:
        return {
            "store_id": self.store_id,
            "shm_name": self.shm_name,
            "offset": self.offset,
            "size": self.size,
            "generation": self.generation,
        }

    @classmethod
    def parse(cls, value: dict[str, Any]) -> ShmAllocation:
        for field in ("store_id", "shm_name"):
            if not isinstance(value.get(field), str) or not value[field]:
                raise ValueError(f"shared-memory allocation requires {field}")
        for field in ("offset", "size", "generation"):
            if type(value.get(field)) is not int or value[field] < (0 if field == "offset" else 1):
                raise ValueError(f"invalid shared-memory allocation {field}")
        return cls(**{key: value[key] for key in ("store_id", "shm_name", "offset", "size", "generation")})


@dataclass
class _LiveAllocation:
    allocation: ShmAllocation
    capacity: int
    refs: int = 1
    forks: list[_ForkHold] = dataclass_field(default_factory=list)
    budget_releases: list[Callable[[], None]] = dataclass_field(default_factory=list)
    retained: bool = True


class StoreLease:
    def __init__(self, store: LocalShmStore, allocation: ShmAllocation) -> None:
        self.store = store
        self.allocation = allocation
        self._lifetime = store._require_locked(allocation)
        self._lock = threading.Lock()
        self._released = False

    def fork(self) -> StoreLease:
        self.store._require_owner()
        with self._lock:
            if self._released:
                raise RuntimeError("shared-memory allocation lease is released")
            return self.store.acquire(self.allocation)

    def release(self) -> None:
        # A fork inherits both locks and a stale allocation table. Its
        # finalizers must not reclaim pages or names owned by the parent.
        if self.store._owner_pid != os.getpid():
            return
        callbacks = []
        with self._lock:
            if not self._released:
                callbacks = self.store.release(self.allocation)
                self._released = True
        self.store.return_budgets(callbacks)
        self.store.drain_if_idle()

    def __del__(self) -> None:
        self.release()


class LocalShmStore:
    def __init__(self, capacity: int) -> None:
        _require_registry_owner()
        if type(capacity) is not int or capacity < _ALIGNMENT:
            raise ValueError("shared-memory store capacity must be at least 64 bytes")
        self.capacity = capacity
        self.store_id = uuid.uuid4().hex
        self._owner_pid = os.getpid()
        self._lock = threading.RLock()
        self._capacity_condition = threading.Condition(self._lock)
        self._waiting_writes: dict[tuple[str, int], tuple[int, set[int]]] = {}
        self._shm: Any = None
        self._free = [(0, capacity)]
        self._live: dict[int, _LiveAllocation] = {}
        self._forked: dict[int, _LiveAllocation] = {}
        self._generations = count(1)
        self._clients: set[str] = set()
        self._closed = False
        self._unlinked = False
        self._allocations = 0
        self._reused_allocations = 0
        self._high_water = 0
        self._budget_releases: list[Callable[[], None]] = []
        with _registry_lock, _fork_lock:
            _stores[self.store_id] = self

    def _require_owner(self) -> None:
        if self._owner_pid != os.getpid():
            raise RuntimeError("shared-memory store belongs to a different process")

    def add_client(self, client_id: str) -> bool:
        """Register atomically, or let the caller replace a retired arena."""
        self._require_owner()
        with self._lock:
            if self._closed:
                return False
            self._clients.add(client_id)
            return True

    def remove_client(self, client_id: str) -> None:
        if self._owner_pid != os.getpid():
            return
        with self._mutation():
            self._collect_fork_pins_locked()
            self._clients.discard(client_id)
            if not self._clients:
                self._drain_locked()

    def allocate(self, size: int) -> StoreLease:
        self._require_owner()
        required = self.check_write_size(size)
        with self._mutation():
            self._collect_fork_pins_locked()
            if self._closed:
                raise RuntimeError("shared-memory store is closed")
            index = next((i for i, (_, length) in enumerate(self._free) if length >= required), None)
            if index is None:
                live = sum(entry.capacity for entry in self._live.values())
                raise LocalShmStoreCapacityError(
                    f"shared-memory store cannot allocate {size} bytes: "
                    f"live={live}, capacity={self.capacity}, "
                    f"largest_free={max((length for _, length in self._free), default=0)}; "
                    "live buffers must be released before retry"
                )
            if self._shm is None:
                from vane.execution.ref_bundle import _create_shm

                self._shm = _create_shm(self.capacity, track=False)
            offset, length = self._free.pop(index)
            if length > required:
                self._free.insert(index, (offset + required, length - required))
            allocation = ShmAllocation(self.store_id, self._shm.name, offset, size, next(self._generations))
            self._live[allocation.generation] = _LiveAllocation(allocation, required)
            self._allocations += 1
            if offset < self._high_water:
                self._reused_allocations += 1
            self._high_water = max(self._high_water, offset + required)
            return StoreLease(self, allocation)

    def check_write_size(self, size: int) -> int:
        required = _allocation_capacity(size)
        if required > self.capacity:
            raise LocalShmStoreCapacityError(
                f"shared-memory output block exceeds arena capacity: requested={size}, capacity={self.capacity}"
            )
        return required

    def wait_for_capacity(
        self,
        size: int,
        inputs: tuple[ShmAllocation, ...],
        scope: ExecutionCancellationScope,
        wait_context: Callable[[], AbstractContextManager[None]],
    ) -> None:
        """Wait for a physical release without owning a grant or a worker CPU.

        A wait cannot resolve when every live region is an input of a writer
        waiting for output space. Report that capacity boundary instead of
        letting the producers and consumers wait on each other indefinitely.
        """
        self._require_owner()
        required = self.check_write_size(size)
        generations = {allocation.generation for allocation in inputs if allocation.store_id == self.store_id}

        def wake() -> None:
            with self._capacity_condition:
                self._capacity_condition.notify_all()

        unregister = scope.register_cancel_wakeup(wake)
        try:
            # Admission transitions can acquire unrelated locks. Keep them
            # outside the store lock, including CPU reacquisition on exit.
            with wait_context():
                with self._capacity_condition:
                    self._waiting_writes[scope.identity] = (required, generations)
                    self._capacity_condition.notify_all()
                    try:
                        while True:
                            scope.raise_if_cancelled("shared-memory output capacity")
                            if self._closed:
                                raise RuntimeError("shared-memory store is closed")
                            largest = max((length for _, length in self._free), default=0)
                            if largest >= required:
                                return
                            pinned = sum(self._live[g].capacity for g in generations if g in self._live)
                            blocked_inputs = set().union(*(g for _, g in self._waiting_writes.values()))
                            all_blocked = self._live.keys() <= blocked_inputs and all(
                                requested > largest for requested, _ in self._waiting_writes.values()
                            )
                            if pinned + required > self.capacity or all_blocked:
                                raise LocalShmStoreCapacityError(
                                    f"shared-memory output cannot make progress: requested={size}, "
                                    f"pinned_input_bytes={pinned}, capacity={self.capacity}, "
                                    f"largest_free={largest}; reduce task input or output block size"
                                )
                            self._capacity_condition.wait()
                    finally:
                        del self._waiting_writes[scope.identity]
                        self._capacity_condition.notify_all()
        finally:
            unregister()

    def _require_locked(self, allocation: ShmAllocation) -> _LiveAllocation:
        entry = self._live.get(allocation.generation)
        if entry is None or entry.allocation != allocation:
            raise ValueError("stale or foreign shared-memory allocation")
        return entry

    def acquire(self, allocation: ShmAllocation) -> StoreLease:
        self._require_owner()
        with self._lock, _fork_lock:
            entry = self._require_locked(allocation)
            entry.refs += 1
            return StoreLease(self, allocation)

    def release(self, allocation: ShmAllocation) -> list[Callable[[], None]]:
        if self._owner_pid != os.getpid():
            return []
        with self._lock, _fork_lock:
            self._collect_fork_pins_locked()
            entry = self._require_locked(allocation)
            if not entry.refs:
                raise ValueError("shared-memory allocation lease already released")
            entry.refs -= 1
            if not entry.refs and not entry.forks:
                self._free_allocation_locked(entry)
            callbacks, self._budget_releases = self._budget_releases, []
            return callbacks

    def retain_budget(self, allocation: ShmAllocation, release: Callable[[], None]) -> None:
        self._require_owner()
        with self._lock:
            self._require_locked(allocation).budget_releases.append(release)

    @staticmethod
    def return_budgets(callbacks: list[Callable[[], None]]) -> None:
        error: BaseException | None = None
        for callback in callbacks:
            try:
                callback()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    @contextmanager
    def _mutation(self) -> Iterator[None]:
        callbacks = []
        try:
            with self._lock, _fork_lock:
                try:
                    yield
                finally:
                    callbacks, self._budget_releases = self._budget_releases, []
        except BaseException as error:
            try:
                self.return_budgets(callbacks)
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise
        else:
            self.return_budgets(callbacks)

    def _free_allocation_locked(self, entry: _LiveAllocation) -> None:
        del self._live[entry.allocation.generation]
        entry.retained = False
        self._budget_releases.extend(entry.budget_releases)
        entry.budget_releases.clear()
        self._free.append((entry.allocation.offset, entry.capacity))
        merged: list[tuple[int, int]] = []
        for start, length in sorted(self._free):
            if merged and sum(merged[-1]) == start:
                previous, capacity = merged.pop()
                merged.append((previous, capacity + length))
            else:
                merged.append((start, length))
        self._free = merged
        self._capacity_condition.notify_all()

    def _collect_fork_pins_locked(self) -> bool:
        if not self._forked:
            return False
        freed = False
        for generation, entry in list(self._forked.items()):
            entry.forks = [hold for hold in entry.forks if hold.is_alive()]
            if not entry.forks:
                del self._forked[generation]
                if not entry.refs:
                    self._free_allocation_locked(entry)
                    freed = True
        return freed

    def drain_if_idle(self) -> None:
        if self._owner_pid != os.getpid():
            return
        with self._mutation():
            self._collect_fork_pins_locked()
            if not self._clients:
                self._drain_locked()

    def buffer(self, allocation: ShmAllocation) -> memoryview:
        self._require_owner()
        with self._lock:
            self._require_locked(allocation)
            return self._shm.buf[allocation.offset : allocation.offset + allocation.size]

    def _drain_locked(self) -> None:
        if self._live:
            # A returned Arrow view may outlive every worker. Decommit only
            # wholly free pages; keep its live allocation and mapping valid.
            if self._shm is not None:
                page = mmap.PAGESIZE
                for offset, length in self._free:
                    start = (offset + page - 1) // page * page
                    end = (offset + length) // page * page
                    if end > start:
                        self._shm._mmap.madvise(mmap.MADV_REMOVE, start, end - start)
            return
        if self._shm is not None:
            self._unlink_locked()
            self._shm.close()
            self._shm = None
        self._closed = True
        self._capacity_condition.notify_all()
        # Registry entries remain until cleanup succeeds, so close can retry.

    def _unlink_locked(self) -> None:
        if self._shm is not None and not self._unlinked:
            from vane.execution.ref_bundle import _unlink_shm

            try:
                _unlink_shm(self._shm, track=False)
            except FileNotFoundError:
                pass
            self._unlinked = True

    def _unlink_at_exit(self) -> None:
        # Forked children inherit the registry and its locks, but do not own
        # these names. Check the owner before touching an inherited lock.
        if self._owner_pid != os.getpid():
            return
        with self._lock:
            # Unlinking preserves mapped buffers, including views accessed by
            # later exit callbacks. The OS releases mappings on process exit.
            self._unlink_locked()

    def close(self) -> None:
        if self._owner_pid != os.getpid():
            return
        with self._mutation():
            self._collect_fork_pins_locked()
            if self._clients:
                raise RuntimeError("shared-memory store still has live worker clients")
            self._drain_locked()

    def snapshot(self) -> dict[str, int]:
        self._require_owner()
        with self._mutation():
            if self._collect_fork_pins_locked() and not self._clients:
                self._drain_locked()
            live = sum(entry.capacity for entry in self._live.values())
            return {
                "capacity_bytes": self.capacity,
                "mapped_capacity_bytes": self.capacity if self._shm is not None else 0,
                "live_bytes": live,
                "free_bytes": self.capacity - live,
                "live_allocations": len(self._live),
                "clients": len(self._clients),
                "allocations": self._allocations,
                "reused_allocations": self._reused_allocations,
            }


def current_store(client_id: str) -> LocalShmStore:
    global _current_store
    _require_registry_owner()
    with _registry_lock:
        if _current_store is not None:
            _current_store._require_owner()
            with _current_store._lock:
                if not _current_store._clients and not _current_store._live:
                    _current_store.close()
        with _fork_lock:
            for key, store in list(_stores.items()):
                if store._closed:
                    del _stores[key]
        # Retirement can race with this lookup. add_client() checks and pins
        # under the store lock, so failure means we must create a new arena.
        if _current_store is not None and _current_store.add_client(client_id):
            return _current_store
        from vane.execution.ref_bundle import _auto_local_shm_store_capacity_bytes, _parse_byte_size

        raw = os.environ.get("VANE_LOCAL_SHM_STORE_BYTES", "auto").strip().lower()
        capacity = _auto_local_shm_store_capacity_bytes() if raw == "auto" else _parse_byte_size(raw)
        _current_store = LocalShmStore(capacity)
        _current_store.add_client(client_id)
        return _current_store


class LocalQueryShmStore:
    """Pin the physical arena used to size a query, before task workers start."""

    def __init__(self) -> None:
        self._owner_pid = os.getpid()
        self._client_id = uuid.uuid4().hex
        self.store = current_store(self._client_id)
        self._closed = False

    def shutdown(self, *, kill: bool = False) -> None:
        if self._owner_pid != os.getpid() or self._closed:
            return
        self.store.remove_client(self._client_id)
        self._closed = True

    def cleanup_pending(self) -> bool:
        return self._owner_pid == os.getpid() and not self._closed


def _unlink_owned_stores_at_exit() -> None:
    # Do not acquire the registry lock: a forked child can inherit it from a
    # thread that no longer exists. Each arena checks its owning PID first.
    error: Exception | None = None
    for store in list(_stores.values()):
        try:
            store._unlink_at_exit()
        except Exception as exc:
            # A failed unlink must not prevent cleanup of the other arenas.
            if error is None:
                error = exc
    if error is not None:
        raise error


atexit.register(_unlink_owned_stores_at_exit)


def acquire_allocation(value: dict[str, Any]) -> StoreLease:
    _require_registry_owner()
    allocation = ShmAllocation.parse(value)
    with _registry_lock:
        store = _stores.get(allocation.store_id)
    if store is None:
        raise ValueError("unknown shared-memory store")
    return store.acquire(allocation)


def local_shm_store_snapshot() -> dict[str, int]:
    """Diagnostic totals for current arenas and retained views from older ones."""
    _require_registry_owner()
    with _registry_lock:
        snapshots = [store.snapshot() for store in _stores.values()]
    fields = (
        "capacity_bytes",
        "mapped_capacity_bytes",
        "live_bytes",
        "free_bytes",
        "live_allocations",
        "clients",
        "allocations",
        "reused_allocations",
    )
    return {field: sum(snapshot[field] for snapshot in snapshots) for field in fields}


class ParentShmPeer:
    """Own a worker's write grants and remote read pins until release or death."""

    def __init__(self) -> None:
        self._owner_pid = os.getpid()
        self.client_id = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._ids = count(1)
        self._reads: dict[int, StoreLease] = {}
        self._writes: dict[int, StoreLease] = {}
        self._closed = False
        self._reader_error: BaseException | None = None
        self._lifetime = _ForkHold()
        self.store = current_store(self.client_id)
        try:
            self.sock, self.child_sock = socket.socketpair()
            # A separate lifetime descriptor survives release-channel shutdown
            # and the worker's death when fork descendants still retain inputs.
            # Pass it once at startup, without adding per-batch descriptors.
            with _fork_lock:
                self._lifetime.read_fd, write_fd = os.pipe()
                _fork_holds.add(self._lifetime)
                try:
                    socket.send_fds(self.sock, [b"L"], [write_fd])
                finally:
                    os.close(write_fd)
            self._reader = threading.Thread(target=self._receive_releases, name="vane-shm-releases", daemon=True)
            self._reader.start()
        except BaseException:
            for name in ("sock", "child_sock"):
                if (sock := getattr(self, name, None)) is not None:
                    sock.close()
            with _fork_lock:
                self._lifetime.close_reader()
            self.store.remove_client(self.client_id)
            raise

    def _receive_releases(self) -> None:
        try:
            pending = b""
            while True:
                part = self.sock.recv(4096)
                if not part:
                    raise EOFError("worker shared-memory release channel closed")
                pending += part
                while len(pending) >= _RELEASE.size:
                    token = _RELEASE.unpack(pending[: _RELEASE.size])[0]
                    pending = pending[_RELEASE.size :]
                    self.release_read(token)
        except BaseException as error:
            self._reader_error = error

    def check(self) -> None:
        if self._closed:
            raise RuntimeError("shared-memory worker peer is closed")
        if self._reader_error is not None:
            raise RuntimeError("shared-memory release channel failed") from self._reader_error

    def borrow_inputs(self, payload: dict[str, Any]) -> list[int]:
        tokens: list[int] = []
        try:
            with self._lock:
                self.check()
                for descriptor in payload["block_refs"]:
                    allocation = descriptor.get("allocation")
                    if allocation is None:
                        continue
                    lease = acquire_allocation(allocation)
                    token = next(self._ids)
                    self._reads[token] = lease
                    descriptor["borrow_id"] = token
                    tokens.append(token)
        except BaseException:
            for token in tokens:
                self.release_read(token)
            raise
        return tokens

    def release_read(self, token: int) -> None:
        with self._lock:
            lease = self._reads.get(token)
            if lease is not None:
                lease.release()
                del self._reads[token]

    def reserve_write(self, grant_id: int, size: int) -> dict[str, Any]:
        with self._lock:
            self.check()
            if grant_id in self._writes:
                raise ValueError("duplicate shared-memory write grant")
            lease = self.store.allocate(size)
            self._writes[grant_id] = lease
            return lease.allocation.descriptor()

    def validate_write(self, grant_id: int, value: dict[str, Any]) -> None:
        with self._lock:
            lease = self._writes.get(grant_id)
            if lease is None or lease.allocation != ShmAllocation.parse(value):
                raise ValueError("output descriptor does not match its shared-memory write grant")

    def finish_write(self, grant_id: int) -> None:
        with self._lock:
            lease = self._writes.get(grant_id)
            if lease is not None:
                lease.release()
                del self._writes[grant_id]

    def close_after_exit(self) -> None:
        # Forked executor finalizers inherit this peer. They must neither wait
        # on an inherited lock nor shut down the parent's shared socket.
        if os.getpid() != self._owner_pid:
            return
        # The caller must confirm process death. A socket disconnect alone
        # cannot prove that the process stopped reading or writing the arena.
        with self._lock:
            if self._closed:
                return
            self.child_sock.close()
            with _fork_lock:
                descendants_alive = self._lifetime.is_alive()
            if descendants_alive and (self._reads or self._lifetime.stores):
                # Transfer outstanding input ownership to process-lifetime
                # pins before retiring this peer. The worker can die while its
                # children or grandchildren still read the inherited mapping.
                for lease in self._reads.values():
                    store = lease.store
                    with store._lock, _fork_lock:
                        entry = store._require_locked(lease.allocation)
                        if self._lifetime not in entry.forks:
                            entry.forks.append(self._lifetime)
                            store._forked[lease.allocation.generation] = entry
                        if store not in self._lifetime.stores:
                            self._lifetime.stores.append(store)
                _start_fork_watcher(self._lifetime)
            elif not self._lifetime.watching:
                with _fork_lock:
                    self._lifetime.close_reader()
            for leases in (self._reads, self._writes):
                for token, lease in list(leases.items()):
                    lease.release()
                    del leases[token]
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()
            self.store.remove_client(self.client_id)
            self._closed = True


class _WorkerMapping:
    def __init__(self, name: str) -> None:
        from vane.execution.ref_bundle import _open_existing_shm

        self.shm: Any = None
        self.shm = _open_existing_shm(name, track=False)

    def __del__(self) -> None:
        if self.shm is not None:
            self.shm.close()


class _RemoteLease:
    def __init__(self, client: WorkerShmClient, token: int, allocation: ShmAllocation) -> None:
        self.client = client
        self.token = token
        self.allocation = allocation

    def __del__(self) -> None:
        self.client.release(self.token)


class WorkerShmClient:
    def __init__(self, fd: int) -> None:
        self._owner_pid = os.getpid()
        self._lifetime_fd: int | None = None
        self.sock = socket.socket(fileno=fd)
        self._mappings: dict[str, _WorkerMapping] = {}
        self._reads: dict[int, _RemoteLease] = {}
        self._live_reads: set[int] = set()
        self._fork_reads: dict[int, set[_ForkHold]] = {}
        self._queue: queue.SimpleQueue[int | None] = queue.SimpleQueue()
        self._closed = False
        self._error: BaseException | None = None
        try:
            with _fork_lock:
                message, fds, flags, _ = socket.recv_fds(self.sock, 1, 1)
                if message != b"L" or len(fds) != 1 or flags & socket.MSG_CTRUNC:
                    for received_fd in fds:
                        os.close(received_fd)
                    raise RuntimeError("invalid shared-memory worker lifetime descriptor")
                self._lifetime_fd = fds[0]
                os.set_inheritable(self._lifetime_fd, False)
                _worker_clients.add(self)
            self._writer = threading.Thread(target=self._send_releases, name="vane-shm-release-writer", daemon=True)
            self._writer.start()
        except BaseException:
            if self._lifetime_fd is not None:
                os.close(self._lifetime_fd)
                self._lifetime_fd = None
            self.sock.close()
            raise

    def _send_releases(self) -> None:
        try:
            while (token := self._queue.get()) is not None:
                self.sock.sendall(_RELEASE.pack(token))
        except BaseException as error:
            self._error = error

    def check(self) -> None:
        if self._owner_pid != os.getpid():
            raise RuntimeError("shared-memory worker client belongs to a different process")
        if self._closed or self._error is not None:
            raise RuntimeError("shared-memory release channel is unavailable") from self._error

    def begin_input(self, payload: dict[str, Any]) -> None:
        self.check()
        try:
            with _fork_lock:
                self._collect_fork_reads_locked()
                for descriptor in payload["block_refs"]:
                    if "allocation" not in descriptor:
                        continue
                    token = descriptor.get("borrow_id")
                    if type(token) is not int or token <= 0 or token in self._live_reads or token in self._fork_reads:
                        raise ValueError("shared-memory input requires a unique borrow token")
                    self._reads[token] = _RemoteLease(self, token, ShmAllocation.parse(descriptor["allocation"]))
                    self._live_reads.add(token)
        except BaseException:
            self.end_input()
            raise

    def end_input(self) -> None:
        if self._owner_pid != os.getpid():
            return
        with _fork_lock:
            self._reads.clear()

    def read_lease(self, value: dict[str, Any], token: int) -> _RemoteLease:
        self.check()
        lease = self._reads.get(token)
        if lease is None or lease.allocation != ShmAllocation.parse(value):
            raise ValueError("shared-memory input has no live borrow lease")
        return lease

    def mapping(self, allocation: ShmAllocation) -> _WorkerMapping:
        self.check()
        mapping = self._mappings.get(allocation.shm_name)
        if mapping is None:
            mapping = _WorkerMapping(allocation.shm_name)
            self._mappings[allocation.shm_name] = mapping
        if allocation.offset + allocation.size > mapping.shm.size:
            raise BufferError("shared-memory allocation exceeds its arena")
        return mapping

    def release(self, token: int) -> None:
        # Inherited finalizers must neither queue a parent's token nor wait on
        # a lock copied from a vanished thread.
        if self._owner_pid != os.getpid():
            return
        with _fork_lock:
            self._collect_fork_reads_locked()
            if token in self._live_reads:
                self._live_reads.remove(token)
                if token not in self._fork_reads and not self._closed:
                    self._queue.put(token)
            self._close_lifetime_if_idle_locked()

    def _collect_fork_reads_locked(self) -> None:
        for token, holds in list(self._fork_reads.items()):
            live = {hold for hold in holds if hold.is_alive()}
            if live:
                self._fork_reads[token] = live
            else:
                del self._fork_reads[token]
                if token not in self._live_reads and not self._closed:
                    self._queue.put(token)

    def _close_lifetime_if_idle_locked(self) -> None:
        # Late exit callbacks can still fork retained views after close().
        # Keep the inheritable lifetime until those local views are gone;
        # descriptors already inherited by children live until their exit.
        if self._closed and not self._live_reads and self._lifetime_fd is not None:
            os.close(self._lifetime_fd)
            self._lifetime_fd = None

    def __del__(self) -> None:
        if self._owner_pid == os.getpid() and self._lifetime_fd is not None:
            os.close(self._lifetime_fd)
            self._lifetime_fd = None

    def close(self) -> None:
        if self._owner_pid != os.getpid():
            return
        with _fork_lock:
            if not self._closed:
                self.end_input()
                self._closed = True
                _worker_clients.discard(self)
                self._queue.put(None)
                self._mappings.clear()
                self._close_lifetime_if_idle_locked()
        # Late Python finalizers remain pinned in the parent until process exit.
        self._writer.join(timeout=1)
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


def initialize_worker_shm_client(fd: int) -> WorkerShmClient:
    global _worker_client
    _worker_client = WorkerShmClient(fd)
    return _worker_client


def worker_shm_client() -> WorkerShmClient:
    if _worker_client is None:
        raise RuntimeError("shared-memory worker client is not initialized")
    return _worker_client
