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
from vane.execution.plan import _fields
from vane.execution.query_options import _seconds
from vane.execution.resource_demand import _capacity


@dataclass(frozen=True)
class ExchangeStore:
    """A named shared mount whose failure domain is independent of compute.

    Byte capacities cover source and exchange data. Protocol metadata has
    separate fixed count/size bounds. All sessions using a root must agree on
    capacity_bytes; each admitted query reserves its complete query_bytes.
    """

    name: str
    root: str
    capacity_bytes: int = 16 << 30
    query_bytes: int = 1 << 30
    object_bytes: int = 64 << 20
    source_bytes: int = 256 << 20
    lease_seconds: float = 60

    def __post_init__(self) -> None:
        _label(self.name, "exchange store name")
        if not isinstance(self.root, str) or not Path(self.root).is_absolute() or "=" in self.root:
            raise ValueError("FTE store root must be absolute and cannot contain '='")
        for name in ("capacity_bytes", "query_bytes", "object_bytes", "source_bytes"):
            _capacity(getattr(self, name), name)
        if self.query_bytes > self.capacity_bytes or max(self.object_bytes, self.source_bytes) > self.query_bytes:
            raise ValueError("FTE storage reservations exceed their containing capacity")
        if max(self.object_bytes, self.source_bytes) > 1 << 40:
            raise ValueError("FTE files are limited to 1 TiB")
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


@contextmanager
def store_lock(path: Path, *, timeout: float = 5) -> Iterator[Any]:
    from vane._native import execution_runtime as native

    end = time.monotonic() + timeout
    while True:
        guard = native.StoreGuard.acquire(str(path), True, True)
        if guard is not None:
            break
        if time.monotonic() >= end:
            raise TimeoutError("shared storage control lock timed out")
        time.sleep(0.005)
    try:
        yield guard
    finally:
        guard.close()


class ActiveStoreLease:
    """A worker/result-service pin, released only after its native I/O stops."""

    def __init__(self, wire: Mapping[str, Any]) -> None:
        from vane._native import execution_runtime as native

        _fields(wire, {"store", "namespace", "lease_id"}, "active store lease")
        self.store = StoreDescriptor.from_dict(wire["store"])
        self.store.check()
        self.namespace = wire["namespace"]
        self.lease_id = wire["lease_id"]
        _hex(self.namespace, 32, "namespace")
        _hex(self.lease_id, 32, "lease_id")
        self.directory = Path(self.store.root) / "queries" / self.namespace
        if self.directory.resolve() != self.directory:
            raise ValueError("query lease path contains a symlink")
        self.guard = native.StoreGuard.acquire(str(self.directory / "lease.lock"), False, False)
        if self.guard is None:
            raise RuntimeError("query store is being cleaned")
        try:
            self.check()
        except BaseException:
            self.close()
            raise

    def check(self) -> dict[str, Any]:
        value = _read(Path(self.store.root) / "allocations" / f"{self.namespace}.json")
        if value["lease_id"] != self.lease_id or value["namespace"] != self.namespace:
            raise RuntimeError("query store lease changed")
        if value["expires"] <= time.time():
            raise RuntimeError("query store lease expired")
        return value

    def close(self) -> None:
        if self.guard is not None:
            self.guard.close()


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

    def _collect(self) -> None:
        from vane import IOException
        from vane._native import execution_runtime as native

        self.store.descriptor.check()
        queries = self.root / "queries"
        if queries.resolve() != queries:
            raise ValueError("query directory contains a symlink")
        for record, value in self._records():
            if value["expires"] > time.time():
                continue
            directory = self.root / "queries" / value["namespace"]
            guard = None
            try:
                if directory.resolve() != directory:
                    raise ValueError("query directory contains a symlink")
                if directory.exists():
                    guard = native.StoreGuard.acquire(str(directory / "lease.lock"), True, True)
                    if guard is None:
                        continue
                    shutil.rmtree(directory)
                record.unlink()
            except (OSError, IOException):
                # A failed orphan cleanup must not fail another query's
                # heartbeat or admission. Keep its external allocation (and
                # quota) until a later collection can complete the deletion.
                continue
            finally:
                if guard is not None:
                    guard.close()

    def collect_expired(self) -> None:
        with store_lock(self.root / "allocation.lock"):
            self._collect()

    def reserve(self, query_id: str) -> QueryStoreLease:
        from vane._native import execution_runtime as native

        _label(query_id, "query_id")
        with store_lock(self.root / "allocation.lock"):
            self._collect()
            records = self._records()
            if (
                len(records) >= 4096
                or sum(v["bytes"] for _, v in records) + self.config.query_bytes > self.config.capacity_bytes
            ):
                raise RuntimeError("insufficient shared storage capacity")
            namespace = uuid.uuid4().hex
            value = {
                "query_id": query_id,
                "namespace": namespace,
                "lease_id": uuid.uuid4().hex,
                "bytes": self.config.query_bytes,
                "seconds": self.config.lease_seconds,
                "expires": time.time() + self.config.lease_seconds,
            }
            directory = self.root / "queries" / namespace
            directory.mkdir(mode=0o700)
            guard = native.StoreGuard.acquire(str(directory / "lease.lock"), False, True)
            assert guard is not None
            try:
                _publish(self.allocations / f"{namespace}.json", value)
            except BaseException:
                guard.close()
                shutil.rmtree(directory)
                raise
            return QueryStoreLease(self, value, guard)

    def snapshot(self) -> dict[str, int]:
        with store_lock(self.root / "allocation.lock"):
            values = self._records()
            return {"queries": len(values), "reserved_bytes": sum(v["bytes"] for _, v in values)}


class QueryStoreLease:
    def __init__(self, pool: StorePool, value: dict[str, Any], guard: Any) -> None:
        self.pool, self.value, self.guard = pool, value, guard
        self.directory = pool.root / "queries" / value["namespace"]
        self.exclusive = False
        self.closed = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "store": self.pool.store.descriptor.to_dict(),
            "namespace": self.value["namespace"],
            "lease_id": self.value["lease_id"],
        }

    def renew(self) -> None:
        if self.closed:
            return
        with store_lock(self.pool.root / "allocation.lock"):
            record = self.pool.allocations / f"{self.value['namespace']}.json"
            current = _read(record)
            if current["lease_id"] != self.value["lease_id"] or current["expires"] <= time.time():
                raise RuntimeError("query storage lease expired or changed")
            current["expires"] = time.time() + self.value["seconds"]
            replace_metadata(record, current)

    def close(self, cleanup: Callable[[], None]) -> None:
        from vane._native import execution_runtime as native

        if self.closed:
            return
        with store_lock(self.pool.root / "allocation.lock"):
            record = self.pool.allocations / f"{self.value['namespace']}.json"
            if not record.exists() and not self.directory.exists():
                self.guard.close()
                self.closed = True
                return
            if _read(record)["lease_id"] != self.value["lease_id"]:
                raise RuntimeError("query storage lease changed before cleanup")
            if not self.exclusive and self.directory.exists():
                self.guard.close()
                self.guard = native.StoreGuard.acquire(str(self.directory / "lease.lock"), True, True)
                if self.guard is None:
                    self.guard = native.StoreGuard.acquire(str(self.directory / "lease.lock"), False, False)
                    assert self.guard is not None
                    raise StorageCleanupPending("native storage participants still hold query leases")
                self.exclusive = True
            cleanup()
            if self.directory.exists():
                shutil.rmtree(self.directory)
            record.unlink()
            _sync_directory(self.pool.allocations)
            self.guard.close()
            self.closed = True
