# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded, exclusive database instances shared by pipelined and FTE owners."""

from __future__ import annotations

import threading
from typing import Any

from vane.execution.pipelined_plan import RayResources


class WorkerDatabasePool:
    def __init__(self, resources: RayResources) -> None:
        self.operator_bytes = resources.operator_memory_bytes // resources.max_active_queries
        # Match worker admission's integer division, including a remainder.
        self.capacity = resources.operator_memory_bytes // self.operator_bytes
        self.config: dict[str, Any] = {"threads": resources.cpus_per_worker, "memory_limit": f"{self.operator_bytes}B"}
        self.lock = threading.Lock()
        self.idle: list[Any] = []
        self.leases: dict[str, WorkerDatabaseLease] = {}
        self.created = 0
        self.reused = 0

    def acquire(self, owner: str, *, reusable: bool) -> WorkerDatabaseLease:
        with self.lock:
            if owner in self.leases:
                raise ValueError("database owner already admitted")
            if len(self.leases) >= self.capacity:
                raise RuntimeError("worker has insufficient database capacity")
            lease = WorkerDatabaseLease(self, owner, reusable, self.idle.pop() if self.idle else None)
            self.leases[owner] = lease
            return lease

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "capacity": self.capacity,
                "operator_bytes_per_slot": self.operator_bytes,
                "owners": list(self.leases),
                "idle_instances": len(self.idle),
                "created_instances": self.created,
                "reused_instances": self.reused,
            }


class WorkerDatabaseLease:
    """Operations are serialized by the query/attempt lifecycle lock.

    The owner closes its query connection before release. A failed reset keeps
    this lease and the worker reservation until cleanup is retried successfully.
    """

    def __init__(self, pool: WorkerDatabasePool, owner: str, reusable: bool, anchor: Any) -> None:
        self.pool, self.owner, self.reusable = pool, owner, reusable
        self.anchor = anchor
        self.released = False
        self.opened = False

    def open(self) -> Any:
        import vane
        from vane._native import execution_runtime as native

        if self.released or self.opened:
            raise RuntimeError("database lease no longer accepts a query connection")
        # FTE shares these slots but still starts each attempt on a fresh DB.
        # Retire the idle instance before allocating its replacement.
        if self.anchor is not None and not self.reusable:
            self.anchor.close()
            self.anchor = None
        reused = self.anchor is not None
        if self.anchor is None:
            self.anchor = vane.connect(backend="local", config=self.pool.config)
            with self.pool.lock:
                self.pool.created += 1
        connection = native.worker_connection(self.anchor)
        self.opened = True
        if reused:
            with self.pool.lock:
                self.pool.reused += 1
        return connection

    def release(self) -> None:
        from vane._native import execution_runtime as native

        if self.released:
            return
        if self.anchor is not None:
            try:
                reusable = native.release_worker_database(self.anchor, self.reusable)
            except BaseException:
                # Cleanup may have only partly reset caches. Keep ownership,
                # and retire the instance once a retry confirms quiescence.
                self.reusable = False
                raise
            if not reusable:
                self.anchor.close()
                self.anchor = None
        with self.pool.lock:
            if self.anchor is not None:
                self.pool.idle.append(self.anchor)
                self.anchor = None
            del self.pool.leases[self.owner]
            self.released = True
