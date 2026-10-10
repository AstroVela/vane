# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Local lifetime adapter for the same query-budget allocator used by Ray."""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING

from vane.execution.cluster_resource_policy import (
    ClusterQueryResourceCoordinator,
    NodeCapacity,
    query_coordinator_timing,
)
from vane.execution.query_resource_demand import build_query_demand
from vane.execution.resources import ResourceVector

if TYPE_CHECKING:
    from vane.execution.local_query_admission import LocalQueryAdmission, LocalQueryAdmissionAuthority


class LocalQueryCoordinator:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._queries: dict[str, LocalQueryAdmission] = {}
        self._capacities: dict[str, tuple[ResourceVector, frozenset[str]]] = {}
        heartbeat_timeout, self.refresh_interval_s = query_coordinator_timing(os.environ)
        self._coordinator = ClusterQueryResourceCoordinator((), heartbeat_timeout_s=heartbeat_timeout)
        self._last_refresh = 0.0
        self._nodes: tuple[NodeCapacity, ...] = ()

    def register(self, query: LocalQueryAdmission, capacity: ResourceVector) -> None:
        with self._lock:
            key = query.graph.query_id
            if key in self._queries:
                raise ValueError(f"local query already registered: {key}")
            try:
                self._capacities[key] = (capacity, query.gpu_devices)
                self._update_nodes()
                self._coordinator.register_query(build_query_demand(query.graph, self._nodes))
                self._queries[key] = query
                self._refresh()
            except BaseException as error:
                self._queries.pop(key, None)
                self._capacities.pop(key, None)
                state = self._coordinator.snapshot()["queries"].get(key)
                try:
                    if state is not None:
                        self._coordinator.release_query(key, state["allocation"]["generation"])
                    self._update_nodes()
                    self._refresh()
                except BaseException as cleanup_error:
                    raise error from cleanup_error
                raise

    def _update_nodes(self) -> None:
        capacities = tuple(self._capacities.values())
        # Observations describe the same node. Distinct GPU inventories can
        # overlap, so count device identities instead of adding pool sizes or
        # taking the largest query's device count.
        resources = ResourceVector(
            gpu=len({device for _, devices in capacities for device in devices}),
            **{
                name: max((getattr(value, name) for value, _ in capacities), default=0)
                for name in ("cpu", "heap_bytes", "object_store_bytes")
            },
        )
        self._nodes = (NodeCapacity("local", resources),)
        self._coordinator.update_node_capacities(self._nodes)

    def refresh(self, query: LocalQueryAdmission) -> None:
        with self._lock:
            if query.graph.query_id not in self._queries:
                return
            if (
                time.monotonic() - self._last_refresh >= self.refresh_interval_s
                or query.manager.pending_allocation_frontier() is not None
            ):
                self._refresh()

    def drive(self, query: LocalQueryAdmission) -> set[LocalQueryAdmissionAuthority]:
        # Like the Ray driver's serialized admission turn, keep shared actor
        # load observation, policy selection and backend reservation together.
        # Backend requests are nonblocking; execution never runs under this lock.
        with self._lock:
            query._sync_actor_slots()
            self.refresh(query)
            return query._pump()

    def _refresh(self) -> None:
        states = self._coordinator.snapshot()["queries"]
        frontiers = {key: query.manager.current_allocation_frontier() for key, query in self._queries.items()}
        allocations = self._coordinator.refresh_queries(
            observed_usage_by_query={
                key: ResourceVector.from_dict(query.manager.snapshot()["soft_allocation_usage"])
                for key, query in self._queries.items()
            },
            generations={key: state["allocation"]["generation"] for key, state in states.items()},
            demands_by_query={
                key: build_query_demand(
                    query.graph,
                    self._nodes,
                    eligible_unit_ids=frontiers[key][0],
                )
                for key, query in self._queries.items()
            },
        )
        for key, allocation in allocations.items():
            manager = self._queries[key].manager
            # Reopen only the frontier whose demand produced this allocation.
            # Completion can publish a newer phase while snapshots are read.
            manager.update_allocation(allocation, reopen_fence_epoch=frontiers[key][1])
        self._last_refresh = time.monotonic()

    def remove(self, query: LocalQueryAdmission) -> None:
        with self._lock:
            key = query.graph.query_id
            if self._queries.pop(key, None) is None:
                return
            generation = self._coordinator.snapshot()["queries"][key]["allocation"]["generation"]
            self._coordinator.release_query(key, generation)
            self._capacities.pop(key)
            self._update_nodes()
            self._refresh()
