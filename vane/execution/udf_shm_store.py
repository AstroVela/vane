# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded local output storage, independent of transport admission credits.

The parent allocates immutable blocks in one arena. A worker receives a write
grant or a pre-registered read lease; neither an input ACK nor task completion
releases that lease. A separate socket delivers last-buffer notifications even
while the task control socket is waiting for output admission.
"""

from __future__ import annotations

import mmap
import os
import queue
import socket
import struct
import threading
import uuid
from dataclasses import dataclass
from itertools import count
from typing import Any

_RELEASE = struct.Struct("!Q")
_ALIGNMENT = 64
_registry_lock = threading.RLock()
_stores: dict[str, LocalShmStore] = {}
_current_store: LocalShmStore | None = None
_worker_client: WorkerShmClient | None = None


class LocalShmStoreCapacityError(RuntimeError):
    """Live physical buffers, rather than transport credits, fill the store."""


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


class StoreLease:
    def __init__(self, store: LocalShmStore, allocation: ShmAllocation) -> None:
        self.store = store
        self.allocation = allocation
        self._lock = threading.Lock()
        self._released = False

    def fork(self) -> StoreLease:
        with self._lock:
            if self._released:
                raise RuntimeError("shared-memory allocation lease is released")
            return self.store.acquire(self.allocation)

    def release(self) -> None:
        with self._lock:
            if not self._released:
                self.store.release(self.allocation)
                self._released = True
        self.store.drain_if_idle()

    def __del__(self) -> None:
        self.release()


class LocalShmStore:
    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity < _ALIGNMENT:
            raise ValueError("shared-memory store capacity must be at least 64 bytes")
        self.capacity = capacity
        self.store_id = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._shm: Any = None
        self._free = [(0, capacity)]
        self._live: dict[int, _LiveAllocation] = {}
        self._generations = count(1)
        self._clients: set[str] = set()
        self._closed = False
        self._unlinked = False
        self._allocations = 0
        self._reused_allocations = 0
        self._high_water = 0
        with _registry_lock:
            _stores[self.store_id] = self

    def add_client(self, client_id: str) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("shared-memory store is closed")
            self._clients.add(client_id)

    def remove_client(self, client_id: str) -> None:
        with self._lock:
            self._clients.discard(client_id)
            if not self._clients:
                self._drain_locked()

    def allocate(self, size: int) -> StoreLease:
        if type(size) is not int or size <= 0:
            raise ValueError("shared-memory allocation size must be positive")
        # Sixteen size classes per power-of-two range keep large-block slack
        # below 6.25%, while absorbing small IPC metadata changes between
        # batches. Exact-size free holes otherwise strand almost an entire
        # image batch when its successor grows by only a few bytes.
        alignment = 1 << max(6, (size - 1).bit_length() - 5)
        required = (size + alignment - 1) // alignment * alignment
        with self._lock:
            if self._closed:
                raise RuntimeError("shared-memory store is closed")
            index = next((i for i, (_, length) in enumerate(self._free) if length >= required), None)
            if index is None:
                # Waiting with input views pinned can deadlock a pipeline. This
                # is a physical capacity error and must not wait for an ACK.
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

    def _require_locked(self, allocation: ShmAllocation) -> _LiveAllocation:
        entry = self._live.get(allocation.generation)
        if entry is None or entry.allocation != allocation:
            raise ValueError("stale or foreign shared-memory allocation")
        return entry

    def acquire(self, allocation: ShmAllocation) -> StoreLease:
        with self._lock:
            entry = self._require_locked(allocation)
            entry.refs += 1
            return StoreLease(self, allocation)

    def release(self, allocation: ShmAllocation) -> None:
        with self._lock:
            entry = self._require_locked(allocation)
            if entry.refs > 1:
                entry.refs -= 1
                return
            del self._live[allocation.generation]
            self._free.append((allocation.offset, entry.capacity))
            merged: list[tuple[int, int]] = []
            for start, length in sorted(self._free):
                if merged and sum(merged[-1]) == start:
                    previous, capacity = merged.pop()
                    merged.append((previous, capacity + length))
                else:
                    merged.append((start, length))
            self._free = merged

    def drain_if_idle(self) -> None:
        with self._lock:
            if not self._clients:
                self._drain_locked()

    def buffer(self, allocation: ShmAllocation) -> memoryview:
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
            from vane.execution.ref_bundle import _unlink_shm

            if not self._unlinked:
                _unlink_shm(self._shm, track=False)
                self._unlinked = True
            self._shm.close()
            self._shm = None
        self._closed = True
        # Registry entries remain until cleanup succeeds, so close can retry.

    def close(self) -> None:
        with self._lock:
            if self._clients:
                raise RuntimeError("shared-memory store still has live worker clients")
            self._drain_locked()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
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
    with _registry_lock:
        if _current_store is not None:
            with _current_store._lock:
                if not _current_store._clients and not _current_store._live:
                    _current_store.close()
        for key, store in list(_stores.items()):
            if store._closed:
                del _stores[key]
        if _current_store is None or _current_store._closed:
            from vane.execution.ref_bundle import _auto_local_shm_store_capacity_bytes, _parse_byte_size

            raw = os.environ.get("VANE_LOCAL_SHM_STORE_BYTES", "auto").strip().lower()
            capacity = _auto_local_shm_store_capacity_bytes() if raw == "auto" else _parse_byte_size(raw)
            _current_store = LocalShmStore(capacity)
        _current_store.add_client(client_id)
        return _current_store


def acquire_allocation(value: dict[str, Any]) -> StoreLease:
    allocation = ShmAllocation.parse(value)
    with _registry_lock:
        store = _stores.get(allocation.store_id)
    if store is None:
        raise ValueError("unknown shared-memory store")
    return store.acquire(allocation)


def local_shm_store_snapshot() -> dict[str, int]:
    """Diagnostic totals for current arenas and retained views from older ones."""
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
        self.client_id = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._ids = count(1)
        self._reads: dict[int, StoreLease] = {}
        self._writes: dict[int, StoreLease] = {}
        self._closed = False
        self._reader_error: BaseException | None = None
        self.store = current_store(self.client_id)
        try:
            self.sock, self.child_sock = socket.socketpair()
            self._reader = threading.Thread(target=self._receive_releases, name="vane-shm-releases", daemon=True)
            self._reader.start()
        except BaseException:
            for name in ("sock", "child_sock"):
                if (sock := getattr(self, name, None)) is not None:
                    sock.close()
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
        # The caller must confirm process death. A socket disconnect alone
        # cannot prove that the process stopped reading or writing the arena.
        with self._lock:
            if self._closed:
                return
            for leases in (self._reads, self._writes):
                for token, lease in list(leases.items()):
                    lease.release()
                    del leases[token]
            self.child_sock.close()
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
        self.sock = socket.socket(fileno=fd)
        self._mappings: dict[str, _WorkerMapping] = {}
        self._reads: dict[int, _RemoteLease] = {}
        self._queue: queue.SimpleQueue[int | None] = queue.SimpleQueue()
        self._closed = False
        self._error: BaseException | None = None
        self._writer = threading.Thread(target=self._send_releases, name="vane-shm-release-writer", daemon=True)
        self._writer.start()

    def _send_releases(self) -> None:
        try:
            while (token := self._queue.get()) is not None:
                self.sock.sendall(_RELEASE.pack(token))
        except BaseException as error:
            self._error = error

    def check(self) -> None:
        if self._closed or self._error is not None:
            raise RuntimeError("shared-memory release channel is unavailable") from self._error

    def begin_input(self, payload: dict[str, Any]) -> None:
        self.check()
        try:
            for descriptor in payload["block_refs"]:
                if "allocation" not in descriptor:
                    continue
                token = descriptor.get("borrow_id")
                if type(token) is not int or token <= 0 or token in self._reads:
                    raise ValueError("shared-memory input requires a unique borrow token")
                self._reads[token] = _RemoteLease(self, token, ShmAllocation.parse(descriptor["allocation"]))
        except BaseException:
            self.end_input()
            raise

    def end_input(self) -> None:
        self._reads.clear()

    def read_lease(self, value: dict[str, Any], token: int) -> _RemoteLease:
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
        if not self._closed:
            self._queue.put(token)

    def close(self) -> None:
        self.end_input()
        self._closed = True
        self._queue.put(None)
        self._mappings.clear()
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
