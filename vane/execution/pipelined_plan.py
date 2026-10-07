# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Fixed pipelined placement and finite per-worker capacity declarations."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, field
from typing import Any

from vane.execution.direct_exchange import DirectExchangeLimits
from vane.execution.fte_store import ExchangeStore
from vane.execution.query_runtime import QueryResources
from vane.execution.resource_demand import _capacity
from vane.execution.submission import RayQuerySpec


@dataclass(frozen=True)
class RayResources(QueryResources):
    """Service worker pool capacities, shared by all sessions and queries.

    Ray reserves each worker's CPU and operator memory. Native exchange and
    encoding windows are additionally reserved on that worker before start.
    Pipelined scheduling admits the whole graph; FTE admits bounded stage
    attempts. Both charge the same worker ledger.
    """

    worker_count: int = 2
    cpus_per_worker: int = 1
    task_contexts_per_worker: int = 16
    operator_memory_bytes: int = 256 << 20
    exchange_buffer_bytes: int = 16 << 20
    staging_buffer_bytes: int = 32 << 20
    io_concurrency: int = 32
    partitions: int = 2
    exchange: DirectExchangeLimits = field(default_factory=DirectExchangeLimits)
    exchange_stores: tuple[ExchangeStore, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "worker_count",
            "cpus_per_worker",
            "task_contexts_per_worker",
            "operator_memory_bytes",
            "exchange_buffer_bytes",
            "staging_buffer_bytes",
            "io_concurrency",
            "partitions",
        ):
            _capacity(getattr(self, name), name)
        if not isinstance(self.exchange, DirectExchangeLimits):
            raise TypeError("exchange must be DirectExchangeLimits")
        if not isinstance(self.exchange_stores, (tuple, list)) or any(
            not isinstance(s, ExchangeStore) for s in self.exchange_stores
        ):
            raise TypeError("exchange_stores must contain ExchangeStore registrations")
        if len({s.name for s in self.exchange_stores}) != len(self.exchange_stores):
            raise ValueError("exchange store names must be unique")
        object.__setattr__(self, "exchange_stores", tuple(self.exchange_stores))
        if self.io_concurrency > 4096 or self.worker_count > 256:
            raise ValueError("Ray capacities exceed the supported worker/link limits")
        if self.operator_memory_bytes < self.max_active_queries:
            raise ValueError("operator memory must provide a positive share for every admitted query")


@dataclass(frozen=True)
class DirectTicket:
    query_id: str
    attempt_id: str
    producer_epoch: str
    consumer_epoch: str
    exchange_id: str
    producer_task: str
    consumer_task: str
    partition: int
    schema_id: str
    capability: str = field(repr=False)
    protocol: int = 1
    routing_version: int = 1

    def encode(self) -> str:
        from dataclasses import asdict

        for name in (
            "query_id",
            "attempt_id",
            "producer_epoch",
            "consumer_epoch",
            "exchange_id",
            "producer_task",
            "consumer_task",
            "schema_id",
            "capability",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value) > 256:
                raise ValueError(f"invalid direct ticket {name}")
        if type(self.partition) is not int or self.partition < 0 or self.protocol != 1 or self.routing_version != 1:
            raise ValueError("invalid direct ticket version or partition")
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def task_id(fragment: str, partition: int) -> str:
    return f"{fragment}/{partition}"


def placement(spec: RayQuerySpec, epochs: list[str], result_epoch: str) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Freeze all members and routing before any native task is started."""
    if not epochs or len(set(epochs)) != len(epochs) or result_epoch in epochs:
        raise ValueError("worker epochs must be distinct")
    fragments = {f.fragment_id: f for f in spec.graph.fragments}
    tasks = [
        task_id(fragment_id, part)
        for fragment_id in spec.graph.topological_fragment_ids()
        for part in range(fragments[fragment_id].partition_count)
    ]
    workers = {identity: index % len(epochs) for index, identity in enumerate(tasks)}
    routes: list[dict[str, Any]] = []

    def add(edge: str, source: str, target: str, port: str, part: int, schema: bytes) -> None:
        producer_worker = workers[source]
        consumer_worker = workers.get(target, -1)
        ticket = DirectTicket(
            spec.query_id,
            "0",
            epochs[producer_worker],
            result_epoch if consumer_worker == -1 else epochs[consumer_worker],
            edge,
            source,
            target,
            part,
            hashlib.sha256(schema).hexdigest(),
            secrets.token_urlsafe(32),
        )
        routes.append(
            {
                "id": f"{edge}/{source}/{part}",
                "source": source,
                "target": target,
                "port": port,
                "source_worker": producer_worker,
                "target_worker": consumer_worker,
                "partition": part,
                "edge": edge,
                "schema": schema,
                "ticket": ticket.encode(),
            }
        )

    for edge in spec.graph.exchanges:
        producer, consumer = fragments[edge.producer_fragment_id], fragments[edge.consumer_fragment_id]
        schema = next(port.schema for port in producer.outputs if port.port_id == edge.producer_port)
        for upstream in range(producer.partition_count):
            for downstream in range(consumer.partition_count):
                add(
                    edge.exchange_id,
                    task_id(producer.fragment_id, upstream),
                    task_id(consumer.fragment_id, downstream),
                    edge.consumer_port,
                    downstream,
                    schema,
                )
    add("result", task_id(spec.graph.result.fragment_id, 0), "result-service", "result", 0, spec.result_schema)
    return workers, routes
