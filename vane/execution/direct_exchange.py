# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""In-process contract harness for the distributed native task runtime.

Python binds metadata and drives control operations. Fragment data stays in
native, bounded channels. This module is internal: it is neither a local query
backend nor a Ray scheduler/transport. Result batches expose finite test views.
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from typing import Any

from vane.execution.plan import Distribution
from vane.execution.resource_demand import _capacity
from vane.execution.submission import RayQuerySpec, prepare_worker_plan


@dataclass(frozen=True)
class DirectExchangeLimits:
    """Per-consumer owned buffers, including borrowed batches and their slices.

    Frame metadata is separately bounded by frame_slots and fixed membership.
    Native operator input allocations belong to the operator memory domain.
    In-process exchange has no encoding/decoding staging allocation.
    """

    window_bytes: int = 1 << 20
    frame_bytes: int = 1 << 16
    frame_rows: int = 1024
    frame_slots: int = 16

    def __post_init__(self) -> None:
        for name in ("window_bytes", "frame_bytes", "frame_rows", "frame_slots"):
            _capacity(getattr(self, name), name)
        if self.frame_bytes > self.window_bytes:
            raise ValueError("frame_bytes must fit window_bytes")
        if self.frame_rows > 2048:
            raise ValueError("frame_rows must be at most 2048")


def _expire(reference: weakref.ReferenceType[InProcessTaskService]) -> None:
    service = reference()
    if service is not None:
        service.native.expire()


class InProcessTaskService:
    """Prepare real fragment attempts, then advance their native PendingQuery tasks.

    All task contexts and channel windows are reserved before the first start.
    Starts are idempotent by token. ``pump`` is cooperative even with one native
    execution thread. Cancellation interrupts outside the pump/control lock.
    """

    def __init__(
        self,
        connection: Any,
        spec: RayQuerySpec,
        limits: DirectExchangeLimits = DirectExchangeLimits(),
    ) -> None:
        from vane._native import execution_runtime as native

        if not isinstance(spec, RayQuerySpec) or spec.requires_replay:
            raise ValueError("DirectExchange requires a pipelined RayQuerySpec")
        if not isinstance(limits, DirectExchangeLimits):
            raise ValueError("limits must be DirectExchangeLimits")
        fragments = {fragment.fragment_id: fragment for fragment in spec.graph.fragments}
        exchange_windows = sum(fragments[edge.consumer_fragment_id].partition_count for edge in spec.graph.exchanges)
        self.exchange_reservation = exchange_windows * limits.window_bytes
        self.result_reservation = limits.window_bytes
        if self.exchange_reservation > spec.resources.memory.exchange_bytes:
            raise ValueError("exchange windows exceed the query exchange reservation")
        if self.result_reservation > spec.resources.memory.result_bytes:
            raise ValueError("result window exceeds the query result reservation")
        # Use an exclusive cursor: worker preparation restores the captured
        # session and must not change the planning connection's settings.
        with connection.cursor() as validation:
            prepare_worker_plan(validation, spec)

        self.spec = spec
        self.limits = limits
        self.native = native.TaskService(connection)
        self.channels: dict[str, Any] = {}
        self._timer: threading.Timer | None = None
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self._started = False
        self._canceled = False
        self._order: list[str] = []
        native_limits = native.DirectLimits(
            limits.window_bytes, limits.frame_bytes, limits.frame_rows, limits.frame_slots
        )
        incoming: dict[tuple[str, int], dict[str, list[tuple[Any, str]]]] = {}
        outgoing: dict[str, list[dict[str, Any]]] = {}
        try:
            for edge in spec.graph.exchanges:
                producer = fragments[edge.producer_fragment_id]
                consumer = fragments[edge.consumer_fragment_id]
                schema = next(port.schema for port in producer.outputs if port.port_id == edge.producer_port)
                consumers = [self.task_id(consumer.fragment_id, part) for part in range(consumer.partition_count)]
                targets = []
                if edge.distribution is Distribution.BROADCAST:
                    channel = native.DirectChannel(schema, native_limits, producer.partition_count, consumers)
                    self.channels[edge.exchange_id] = channel
                    targets.append(channel)
                    for part, identity in enumerate(consumers):
                        incoming.setdefault((consumer.fragment_id, part), {})[edge.consumer_port] = [
                            (channel, identity)
                        ]
                else:
                    for part, identity in enumerate(consumers):
                        channel = native.DirectChannel(schema, native_limits, producer.partition_count, [identity])
                        self.channels[f"{edge.exchange_id}/{part}"] = channel
                        targets.append(channel)
                        incoming.setdefault((consumer.fragment_id, part), {})[edge.consumer_port] = [
                            (channel, identity)
                        ]
                for part in range(producer.partition_count):
                    identity = self.task_id(producer.fragment_id, part)
                    for channel in targets:
                        channel.add_producer(identity)
                    outgoing.setdefault(identity, []).append(
                        {"channels": targets, "producer": identity, "partitioning": edge.partitioning}
                    )
                for channel in targets:
                    channel.seal_producers()

            root_id = self.task_id(spec.graph.result.fragment_id, 0)
            self.result = native.DirectChannel(spec.result_schema, native_limits, 1, ["client"])
            self.result.add_producer(root_id)
            self.result.seal_producers()
            self.channels["result"] = self.result
            outgoing.setdefault(root_id, []).append({"channels": [self.result], "producer": root_id})

            snapshots = {source.fragment_id: source.payload for source in spec.source_snapshots}
            # Reverse topological order installs consumers first, including all
            # hash partitions. Preparation loads plans but creates no executor.
            for fragment_id in reversed(spec.graph.topological_fragment_ids()):
                fragment = fragments[fragment_id]
                for part in range(fragment.partition_count):
                    identity = self.task_id(fragment_id, part)
                    assignments = {
                        source.source_id: [split.split_id for split in source.splits[part :: fragment.partition_count]]
                        for source in fragment.sources
                    }
                    self.native.prepare(
                        identity,
                        fragment.native_plan,
                        spec.connection_snapshot,
                        snapshots[fragment_id],
                        assignments,
                        incoming.get((fragment_id, part), {}),
                        outgoing[identity],
                    )
                    self._order.append(identity)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def task_id(fragment_id: str, partition: int) -> str:
        return f"{fragment_id}/{partition}"

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self._order)

    def start(self, task_id: str | None = None, *, token: str = "initial") -> None:
        from vane._native.execution_runtime import check_entry

        check_entry()
        if not isinstance(token, str) or not token:
            raise ValueError("start token must be a nonempty string")
        if task_id is not None and task_id not in self._order:
            raise ValueError("unknown task id")
        with self._lifecycle_lock:
            if self._closed or self._canceled:
                raise RuntimeError("task service is closed or canceled")
            if not self._started:
                self._started = True
                self._timer = threading.Timer(self.spec.options.execution_timeout, _expire, (weakref.ref(self),))
                self._timer.daemon = True
                weakref.finalize(self, self._timer.cancel)
                self._timer.start()
        for identity in self._order if task_id is None else (task_id,):
            self.native.start(identity, token)

    def pump(self, steps: int = 1) -> int:
        _capacity(steps, "steps")
        result: int = self.native.pump(steps)
        self._stop_finished_timer()
        return result

    def poll_result(self) -> tuple[str, Any]:
        status, batch = self.result.poll("client")
        return str(status), batch

    def _stop_finished_timer(self) -> None:
        if all(task["state"] not in {"PREPARED", "RUNNING"} for task in self.native.status()):
            with self._lifecycle_lock:
                if self._timer is not None:
                    self._timer.cancel()

    def snapshot(self) -> dict[str, Any]:
        tasks = self.native.status()
        channels = {name: channel.snapshot() for name, channel in self.channels.items()}
        return {
            "tasks": tasks,
            "channels": channels,
            "active_contexts": sum(not task["released"] for task in tasks),
            "owned_bytes": sum(channel["bytes"] for channel in channels.values()),
            "leased_bytes": sum(channel["leased_bytes"] for channel in channels.values()),
            "exchange_reservation": self.exchange_reservation,
            "result_reservation": self.result_reservation,
        }

    def cancel(self, reason: str = "canceled") -> None:
        from vane._native.execution_runtime import check_entry

        check_entry()
        with self._lifecycle_lock:
            self._canceled = True
            if self._timer is not None:
                self._timer.cancel()
        self.native.cancel(reason)

    def close(self) -> None:
        self.cancel("task service closed")
        self.native.release()
        for channel in self.channels.values():
            channel.abort("task service closed")
        with self._lifecycle_lock:
            self._closed = True
            timer = self._timer
        if timer is not None and timer is not threading.current_thread():
            timer.join()

    def __enter__(self) -> InProcessTaskService:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
