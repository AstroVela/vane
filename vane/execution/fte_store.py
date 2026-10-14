# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared storage admission and finite query leases across coordinator processes.

The mount must survive compute worker loss and honor advisory file locks,
atomic publication and sync. Lock ownership outlives a heartbeat: an expired
query is collected only after all native participants have released their locks.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vane.execution.materialized_exchange import _hex, _label, canonical
from vane.execution.materialized_store import (
    SharedDirectoryStore,
    StorageCleanupPending,
    StoreDescriptor,
    _publish,
    _sync_directory,
)
from vane.execution.plan import _fields, _items
from vane.execution.query_options import _seconds
from vane.execution.resource_demand import _capacity


@dataclass(frozen=True)
class ExchangeStore:
    """A named shared mount whose failure domain is independent of compute.

    Byte capacities cover source and exchange data. Protocol metadata has
    separate fixed count/size bounds. All sessions using a root must agree on
    capacity_bytes; each admitted query reserves its complete query_bytes.
    Optional source_cache_bytes reserves a separate version cache per Runtime;
    its immutable copies remain charged until Runtime.close().
    """

    name: str
    root: str
    capacity_bytes: int = 16 << 30
    query_bytes: int = 1 << 30
    object_bytes: int = 64 << 20
    source_bytes: int = 256 << 20
    lease_seconds: float = 60
    source_cache_bytes: int = 0

    def __post_init__(self) -> None:
        _label(self.name, "exchange store name")
        if not isinstance(self.root, str) or not Path(self.root).is_absolute() or "=" in self.root:
            raise ValueError("FTE store root must be absolute and cannot contain '='")
        for name in ("capacity_bytes", "query_bytes", "object_bytes", "source_bytes"):
            _capacity(getattr(self, name), name)
        _capacity(self.source_cache_bytes, "source_cache_bytes", minimum=0)
        if self.query_bytes > self.capacity_bytes or max(self.object_bytes, self.source_bytes) > self.query_bytes:
            raise ValueError("FTE storage reservations exceed their containing capacity")
        if max(self.object_bytes, self.source_bytes) > 1 << 40:
            raise ValueError("FTE files are limited to 1 TiB")
        if self.source_cache_bytes > 1 << 40 or self.query_bytes + self.source_cache_bytes > self.capacity_bytes:
            raise ValueError("FTE query and source cache reservations exceed store capacity")
        object.__setattr__(self, "lease_seconds", _seconds(self.lease_seconds, "lease_seconds"))


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or path.stat().st_size > 16384:
        raise ValueError("invalid store lease metadata")
    value = json.loads(path.read_bytes())
    _fields(value, {"query_id", "lease_id", "namespace", "bytes", "expires", "seconds"}, "store allocation")
    _label(value["query_id"], "query_id")
    _hex(value["lease_id"], 32, "lease_id")
    _hex(value["namespace"], 32, "namespace")
    _capacity(value["bytes"], "storage reservation")
    _seconds(value["expires"], "lease expiry")
    _seconds(value["seconds"], "lease duration")
    return value


def replace_metadata(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{uuid.uuid4().hex}.pending")
    try:
        handle = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(handle, "wb") as output:
            output.write(canonical(value))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class _StoreLockTimeout(TimeoutError):
    """Metadata lock contention, distinct from storage publication failures."""


@contextmanager
def store_lock(path: Path, *, timeout: float = 5) -> Iterator[Any]:
    from vane._native import execution_runtime as native

    end = time.monotonic() + timeout
    while True:
        guard = native.StoreGuard.acquire(str(path), True, True)
        if guard is not None:
            break
        if time.monotonic() >= end:
            raise _StoreLockTimeout("shared storage control lock timed out")
        time.sleep(0.005)
    try:
        yield guard
    finally:
        guard.close()


class ActiveStoreLease:
    """A worker/result-service pin, released only after its native I/O stops."""

    def __init__(self, wire: Mapping[str, Any]) -> None:
        from vane._native import execution_runtime as native

        fields = {"store", "namespace", "lease_id"}
        if "source_leases" in wire:
            fields.add("source_leases")
        _fields(wire, fields, "active store lease")
        sources = _items(wire.get("source_leases", ()), "source leases")
        if len(sources) > 1 or any(not isinstance(s, Mapping) or "source_leases" in s for s in sources):
            raise ValueError("invalid source cache leases")
        self.source_leases: list[ActiveStoreLease] = []
        self.store = StoreDescriptor.from_dict(wire["store"])
        self.store.check()
        self.namespace = wire["namespace"]
        self.lease_id = wire["lease_id"]
        _hex(self.namespace, 32, "namespace")
        _hex(self.lease_id, 32, "lease_id")
        self.directory = Path(self.store.root) / "queries" / self.namespace
        if self.directory.resolve() != self.directory:
            raise ValueError("query lease path contains a symlink")
        allocations = Path(self.store.root) / "allocations"
        if allocations.resolve() != allocations:
            raise ValueError("allocation root contains a symlink")
        self.guard = native.StoreGuard.acquire(str(allocations / f"{self.namespace}.lock"), False, False)
        if self.guard is None:
            raise RuntimeError("query store is being cleaned")
        try:
            self.check()
            for source in sources:
                _fields(source, {"store", "namespace", "lease_id"}, "source cache lease")
                if StoreDescriptor.from_dict(source["store"]) != self.store or source["namespace"] == self.namespace:
                    raise ValueError("source cache lease belongs to another store or refers to itself")
                self.source_leases.append(ActiveStoreLease(source))
        except BaseException:
            self.close()
            raise

    def check(self) -> dict[str, Any]:
        value = _read(Path(self.store.root) / "allocations" / f"{self.namespace}.json")
        if value["lease_id"] != self.lease_id or value["namespace"] != self.namespace:
            raise RuntimeError("query store lease changed")
        if value["expires"] <= time.time():
            raise RuntimeError("query store lease expired")
        for source in self.source_leases:
            source.check()
        return value

    def to_dict(self) -> dict[str, Any]:
        value = {"store": self.store.to_dict(), "namespace": self.namespace, "lease_id": self.lease_id}
        if self.source_leases:
            value["source_leases"] = [source.to_dict() for source in self.source_leases]
        return value

    def close(self) -> None:
        for source in self.source_leases:
            source.close()
        if self.guard is not None:
            self.guard.close()
            self.guard = None


class StorePool:
    def __init__(self, config: ExchangeStore) -> None:
        self.config = config
        self.store = SharedDirectoryStore(config.root)
        self.root = Path(self.store.descriptor.root)
        self.allocations = self.root / "allocations"
        self.allocations.mkdir(mode=0o700, exist_ok=True)
        if self.allocations.is_symlink():
            raise ValueError("allocation root contains a symlink")
        _publish(self.root / "capacity.json", {"bytes": config.capacity_bytes, "protocol": 1})
        self._collection_lock = threading.Lock()
        self._collection_thread: threading.Thread | None = None
        self._collection_error: BaseException | None = None
        self._cache_lock = threading.Lock()
        self._cache: Any = None
        self._cache_lease: QueryStoreLease | None = None
        self._cache_stop = threading.Event()
        self._cache_thread: threading.Thread | None = None
        self._cache_error: BaseException | None = None
        self._cache_closing = False

    def _records(self) -> list[tuple[Path, dict[str, Any]]]:
        if self.allocations.resolve() != self.allocations:
            raise ValueError("allocation root contains a symlink")
        result = []
        for path in self.allocations.glob("*.json"):
            value = _read(path)
            if path.name != f"{value['namespace']}.json":
                raise ValueError("store allocation namespace mismatch")
            result.append((path, value))
        return result

    def _check(self) -> None:
        self.store.descriptor.check()
        queries = self.root / "queries"
        if queries.resolve() != queries:
            raise ValueError("query directory contains a symlink")
        if self.allocations.resolve() != self.allocations:
            raise ValueError("allocation root contains a symlink")

    def _candidates(self, *, timeout: float = 5) -> list[Path]:
        with store_lock(self.root / "allocation.lock", timeout=timeout):
            self._check()
            return [record for record, value in self._records() if value["expires"] <= time.time()]

    def _retire(self, record: Path, value: Mapping[str, Any], guard: Any) -> None:
        # The directory is gone, and its external lock still excludes native
        # participants and other cleaners. Retire lock + quota together under
        # the metadata lock, releasing the old inode before another claim.
        with store_lock(self.root / "allocation.lock"):
            try:
                self._check()
                current = _read(record)
                if (current["namespace"], current["lease_id"]) != (value["namespace"], value["lease_id"]):
                    raise RuntimeError("query storage lease changed during cleanup")
                (self.allocations / f"{value['namespace']}.lock").unlink()
                record.unlink()
                _sync_directory(self.allocations)
            finally:
                guard.close()

    def _collect(self, candidates: list[Path]) -> None:
        from vane import IOException
        from vane._native import execution_runtime as native

        for record in candidates:
            guard = None
            try:
                with store_lock(self.root / "allocation.lock"):
                    self._check()
                    if not record.exists():
                        continue
                    value = _read(record)
                    if record.name != f"{value['namespace']}.json":
                        raise ValueError("store allocation namespace mismatch")
                    # A lease may have renewed since the candidate scan.
                    if value["expires"] > time.time():
                        continue
                    directory = self.root / "queries" / value["namespace"]
                    if directory.resolve() != directory:
                        raise ValueError("query directory contains a symlink")
                    guard = native.StoreGuard.acquire(str(self.allocations / f"{value['namespace']}.lock"), True, True)
                    if guard is None:
                        continue
                # Neither recursive deletion nor query cleanup callbacks may
                # hold allocation.lock. The external lease lock survives even
                # a partial rmtree, and the allocation keeps charging quota.
                if directory.exists():
                    shutil.rmtree(directory)
                self._retire(record, value, guard)
            except (OSError, IOException):
                # A failed orphan cleanup must not fail another query's
                # heartbeat or admission. Keep its external allocation (and
                # quota) until a later collection can complete the deletion.
                continue
            finally:
                if guard is not None:
                    guard.close()

    def collect_expired(self) -> None:
        self._collect(self._candidates())

    def request_collection(self) -> None:
        # One background collector per pool: a heartbeat must keep renewing
        # even when an unrelated deletion exceeds that query's lease duration.
        with self._collection_lock:
            if self._collection_error is not None:
                error, self._collection_error = self._collection_error, None
                raise error
            if self._collection_thread is not None and self._collection_thread.is_alive():
                return
            try:
                candidates = self._candidates(timeout=0)
            except _StoreLockTimeout:
                # Maintenance is opportunistic. Admission still takes the
                # same lock and counts every allocation until it is retired.
                return
            if not candidates:
                return

            def collect() -> None:
                try:
                    self._collect(candidates)
                except BaseException as error:
                    with self._collection_lock:
                        self._collection_error = error

            self._collection_thread = threading.Thread(target=collect, name="vane-fte-orphan-cleanup", daemon=True)
            self._collection_thread.start()

    def reserve(self, query_id: str) -> QueryStoreLease:
        return self._reserve(query_id, self.config.query_bytes)

    def _reserve(self, query_id: str, reserved_bytes: int) -> QueryStoreLease:
        from vane._native import execution_runtime as native

        _label(query_id, "query_id")
        self.request_collection()
        with store_lock(self.root / "allocation.lock"):
            self._check()
            records = self._records()
            if (
                len(records) >= 4096
                or sum(v["bytes"] for _, v in records) + reserved_bytes > self.config.capacity_bytes
            ):
                raise RuntimeError("insufficient shared storage capacity")
            namespace = uuid.uuid4().hex
            value = {
                "query_id": query_id,
                "namespace": namespace,
                "lease_id": uuid.uuid4().hex,
                "bytes": reserved_bytes,
                "seconds": self.config.lease_seconds,
                "expires": time.time() + self.config.lease_seconds,
            }
            directory = self.root / "queries" / namespace
            directory.mkdir(mode=0o700)
            lock_path = self.allocations / f"{namespace}.lock"
            guard = native.StoreGuard.acquire(str(lock_path), False, True)
            assert guard is not None
            try:
                _publish(self.allocations / f"{namespace}.json", value)
            except BaseException:
                guard.close()
                directory.rmdir()  # Not published to any native participant.
                lock_path.unlink()
                raise
            return QueryStoreLease(self, value, guard)

    def source_cache(self, query: QueryStoreLease) -> Any:
        """Borrow a version cache whose files have their own charged store lease."""
        if not self.config.source_cache_bytes:
            return None
        from vane._native import execution_plan

        with self._cache_lock:
            if self._cache_closing:
                raise RuntimeError("source snapshot cache is closing")
            if self._cache_error is not None:
                raise RuntimeError("source snapshot cache lease failed") from self._cache_error
            if self._cache_lease is None:
                owner = self._reserve(f"source-cache-{uuid.uuid4().hex}", self.config.source_cache_bytes)
                try:
                    cache = execution_plan.SourceSnapshotCache(
                        str(owner.directory / "sources"), self.config.source_cache_bytes
                    )
                    thread = threading.Thread(
                        target=self._renew_source_cache, args=(owner,), name="vane-source-cache-lease", daemon=True
                    )
                    thread.start()
                except BaseException:
                    owner.close(lambda: None)
                    raise
                self._cache, self._cache_lease, self._cache_thread = cache, owner, thread
            query.pin_source(self._cache_lease.to_dict())
            return self._cache

    def _renew_source_cache(self, owner: QueryStoreLease) -> None:
        try:
            while not self._cache_stop.wait(min(1.0, self.config.lease_seconds / 4)):
                owner.renew()
        except BaseException as error:
            with self._cache_lock:
                self._cache_error = error.with_traceback(None)

    def close(self) -> None:
        """Retire the cache only after query/worker pins and its heartbeat end."""
        from vane.execution.cleanup_deadline import cleanup_timeout

        with self._cache_lock:
            self._cache_closing = True
            self._cache_stop.set()
            thread = self._cache_thread
        if thread is not None:
            thread.join(timeout=cleanup_timeout(10))
            if thread.is_alive():
                raise StorageCleanupPending("source cache heartbeat cleanup is pending")
        with self._cache_lock:
            if self._cache_lease is not None:
                self._cache_lease.close(lambda: None)
                self._cache_lease = None
                self._cache = None

    def snapshot(self) -> dict[str, int]:
        with store_lock(self.root / "allocation.lock"):
            values = self._records()
            return {"queries": len(values), "reserved_bytes": sum(v["bytes"] for _, v in values)}


class QueryStoreLease:
    def __init__(self, pool: StorePool, value: dict[str, Any], guard: Any) -> None:
        self.pool, self.value, self.guard = pool, value, guard
        self.directory = pool.root / "queries" / value["namespace"]
        self.lock_path = pool.allocations / f"{value['namespace']}.lock"
        self._close_lock = threading.Lock()
        self.exclusive = False
        self.closing = False
        self.closed = False
        self._source_pins: list[ActiveStoreLease] = []

    def to_dict(self) -> dict[str, Any]:
        value = {
            "store": self.pool.store.descriptor.to_dict(),
            "namespace": self.value["namespace"],
            "lease_id": self.value["lease_id"],
        }
        if self._source_pins:
            value["source_leases"] = [source.to_dict() for source in self._source_pins]
        return value

    def pin_source(self, wire: Mapping[str, Any]) -> None:
        with self._close_lock:
            if self.closed or self.closing or self._source_pins:
                raise RuntimeError("query source lease is already fixed or closing")
            if StoreDescriptor.from_dict(wire["store"]) != self.pool.store.descriptor:
                raise ValueError("source cache lease belongs to another store")
            self._source_pins.append(ActiveStoreLease(wire))

    def renew(self) -> None:
        if self.closed or self.closing:
            return
        # This owner's shared lease lock excludes orphan retirement. Serialize
        # with its own close, but never hold the global quota lock across sync:
        # renewal atomically changes expiry only, not the reserved byte count.
        with self._close_lock:
            if self.closed or self.closing:
                # Another thread may close while this one waits for the lock.
                return  # type: ignore[unreachable]
            record = self.pool.allocations / f"{self.value['namespace']}.json"
            current = _read(record)
            if current["lease_id"] != self.value["lease_id"] or current["expires"] <= time.time():
                raise RuntimeError("query storage lease expired or changed")
            current["expires"] = time.time() + self.value["seconds"]
            replace_metadata(record, current)

    def close(self, cleanup: Callable[[], None]) -> None:
        from vane._native import execution_runtime as native

        with self._close_lock:
            if self.closed:
                return
            collected = False
            with store_lock(self.pool.root / "allocation.lock"):
                self.pool._check()
                record = self.pool.allocations / f"{self.value['namespace']}.json"
                if not record.exists() and not self.directory.exists():
                    self.guard.close()
                    collected = True
                else:
                    current = _read(record)
                    if current["lease_id"] != self.value["lease_id"]:
                        raise RuntimeError("query storage lease changed before cleanup")
                    # Fence late native arrivals, including after a failed
                    # retirement releases the lock for later retries.
                    self.closing = True
                    current["expires"] = min(current["expires"], time.time())
                    replace_metadata(record, current)
                    if not self.exclusive:
                        self.guard.close()
                        guard = native.StoreGuard.acquire(str(self.lock_path), True, True)
                        if guard is None:
                            shared = native.StoreGuard.acquire(str(self.lock_path), False, False)
                            if shared is not None:
                                self.guard = shared
                            # An orphan collector may already own the exclusive
                            # lock; keep the closed handle valid for close retry.
                            raise StorageCleanupPending("native storage participants still hold query leases")
                        self.guard = guard
                        self.exclusive = True
            # Even if an orphan collector finished first, release this
            # coordinator's in-memory accounting through its cleanup callback.
            cleanup()
            if not collected:
                if self.directory.exists():
                    shutil.rmtree(self.directory)
                try:
                    self.pool._retire(record, self.value, self.guard)
                finally:
                    self.guard.close()
                    self.exclusive = False
            for source in self._source_pins:
                source.close()
            self.closed = True
