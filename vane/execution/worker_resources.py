# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Atomic worker reservations shared by pipelined graphs and FTE attempts."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any


def worker_capacity(resources: Any) -> dict[str, int]:
    return {
        "contexts": resources.task_contexts_per_worker,
        "exchange": resources.exchange_buffer_bytes,
        "staging": resources.staging_buffer_bytes,
        "io": resources.io_concurrency,
        "operator": resources.operator_memory_bytes,
    }


def _demand(resources: Any, contexts: int, links: int, staging_per_link: int) -> dict[str, int]:
    return {
        "contexts": contexts,
        "exchange": links * resources.exchange.window_bytes,
        "staging": links * staging_per_link,
        "io": links,
        "operator": resources.operator_memory_bytes // resources.max_active_queries,
    }


def pipelined_demand(resources: Any, index: int, tasks: list[str], routes: list[dict[str, Any]]) -> dict[str, int]:
    from vane._native import execution_runtime as native

    links = sum((r["source_worker"] == index) + (r["target_worker"] == index) for r in routes)
    return _demand(resources, len(tasks), links, native.DirectFlight.staging_per_link(resources.exchange.frame_bytes))


def materialized_demand(resources: Any, binding: Any) -> dict[str, int]:
    from vane._native import execution_runtime as native

    links = sum(len(objects) for objects in binding.inputs.values()) + len(binding.task.outputs)
    return _demand(resources, 1, links, native.MaterializedIO.staging_bytes(resources.exchange.frame_bytes))


class WorkerResourceManager:
    """FIFO admission for an atomic set of worker reservations.

    Pipelined graphs hold their entire placement until cleanup. FTE releases
    each attempt before requesting its next quantum. A queued graph therefore
    cannot be overtaken indefinitely by a long sequence of FTE tasks. Worker
    actors independently enforce the same capacities as a final ownership check.
    """

    def __init__(self, capacity: dict[str, int], worker_count: int) -> None:
        self.capacity = dict(capacity)
        self.worker_count = worker_count
        self.condition = threading.Condition()
        self.waiting: OrderedDict[str, tuple[str, dict[int, dict[str, int]], float]] = OrderedDict()
        self.reservations: dict[str, tuple[str, dict[int, dict[str, int]]]] = {}
        self.closed = False

    def _used(self, index: int, name: str) -> int:
        return sum(demands.get(index, {}).get(name, 0) for _, demands in self.reservations.values())

    def try_acquire(self, token: str, query: str, demands: dict[int, dict[str, int]]) -> bool:
        with self.condition:
            if self.closed:
                raise RuntimeError("worker resource manager is closed")
            for index, demand in demands.items():
                if type(index) is not int or not 0 <= index < self.worker_count or set(demand) != set(self.capacity):
                    raise ValueError("invalid worker resource declaration")
                for name, amount in demand.items():
                    if type(amount) is not int or amount < 0 or amount > self.capacity[name]:
                        raise ValueError(f"query demand exceeds worker {index} {name} capacity")
            if not demands:
                raise ValueError("worker reservation must contain at least one worker")
            if token in self.reservations:
                if self.reservations[token] != (query, demands):
                    raise ValueError("worker reservation token reused with different demand")
                return True
            if token not in self.waiting:
                self.waiting[token] = (query, {i: dict(d) for i, d in demands.items()}, time.monotonic())
            elif self.waiting[token][:2] != (query, demands):
                raise ValueError("worker admission token reused with different demand")
            if next(iter(self.waiting)) != token:
                return False
            if any(self._used(i, n) + v > self.capacity[n] for i, d in demands.items() for n, v in d.items()):
                return False
            _, owned, _ = self.waiting.pop(token)
            self.reservations[token] = (query, owned)
            self.condition.notify_all()
            return True

    def acquire(
        self,
        token: str,
        query: str,
        demands: dict[int, dict[str, int]],
        check: Callable[[], None],
        timeout: float | None = None,
    ) -> None:
        from vane.execution.request_admission import RequestQueueTimeout

        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            while True:
                check()
                with self.condition:
                    if self.try_acquire(token, query, demands):
                        return
                    if deadline is not None and time.monotonic() >= deadline:
                        raise RequestQueueTimeout("worker resource admission deadline exceeded")
                    self.condition.wait(0.02)
        except BaseException:
            self.release(token)
            raise

    def release(self, token: str) -> None:
        with self.condition:
            self.waiting.pop(token, None)
            self.reservations.pop(token, None)
            self.condition.notify_all()

    def cancel_waiting(self, query: str) -> None:
        with self.condition:
            for token, (owner, _, _) in tuple(self.waiting.items()):
                if owner == query:
                    self.waiting.pop(token)
            self.condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self.condition:
            now = time.monotonic()
            return {
                "capacity_per_worker": dict(self.capacity),
                "used": {i: {n: self._used(i, n) for n in self.capacity} for i in range(self.worker_count)},
                "reservations": {
                    t: {"query_id": q, "workers": {i: dict(d) for i, d in ds.items()}}
                    for t, (q, ds) in self.reservations.items()
                },
                "waiting": [{"token": t, "query_id": q, "seconds": now - at} for t, (q, _, at) in self.waiting.items()],
            }

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()
