# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Ray control actors; all fragment and relay data movement stays in native."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import replace
from typing import Any

from vane.execution.pipelined_plan import RayResources, task_id
from vane.execution.submission import RayQuerySpec, native_plan_capabilities, prepare_worker_plan


def _host() -> str:
    import ray

    return str(ray.util.get_node_ip_address())


def _channel(schema: bytes, resources: RayResources, producer: str, consumer: str) -> Any:
    from vane._native import execution_runtime as native

    limits = resources.exchange
    channel = native.DirectChannel(
        schema,
        native.DirectLimits(limits.window_bytes, limits.frame_bytes, limits.frame_rows, limits.frame_slots),
        1,
        [consumer],
    )
    channel.add_producer(producer)
    channel.seal_producers()
    return channel


class _Query:
    def __init__(
        self,
        spec: RayQuerySpec,
        resources: RayResources,
        index: int,
        routes: list[dict[str, Any]],
    ) -> None:
        self.spec = spec
        self.resources = resources
        self.index = index
        self.routes = routes
        self.channels: dict[str, Any] = {}
        self.connection: Any = None
        self.service: Any = None
        self.flight: Any = None
        self.thread: threading.Thread | None = None
        self.stop = threading.Event()
        self.started = False
        self.started_at = 0.0
        self.error = ""
        self.order: list[str] = []
        self.lifecycle = threading.RLock()
        self.closed = False
        self.control = threading.Lock()

    def prepare(self, tasks: list[str], operator_bytes: int) -> None:
        import vane
        from vane._native import execution_runtime as native

        spec, resources, index, routes = self.spec, self.resources, self.index, self.routes
        relevant = [r for r in routes if index in (r["source_worker"], r["target_worker"])]
        links = sum((r["source_worker"] == index) + (r["target_worker"] == index) for r in relevant)
        with self.lifecycle:
            if self.stop.is_set():
                raise RuntimeError("query preparation was canceled")
            self.connection = vane.connect(
                backend="local",
                config={
                    "threads": resources.cpus_per_worker,
                    "memory_limit": f"{operator_bytes}B",
                },
            )
            prepare_worker_plan(self.connection, spec)
            if self.stop.is_set():
                raise RuntimeError("query preparation was canceled")
            self.service = native.TaskService(self.connection)
            self.flight = native.DirectFlight(
                "0.0.0.0",
                _host(),
                links,
                links * native.DirectFlight.staging_per_link(resources.exchange.frame_bytes),
                resources.exchange.frame_bytes,
            )
            incoming: dict[str, dict[str, list[Any]]] = {}
            outgoing: dict[str, dict[str, list[Any]]] = {}
            for route in relevant:
                if route["source_worker"] == index:
                    channel = _channel(route["schema"], resources, route["source"], route["target"])
                    self.channels[f"out/{route['id']}"] = channel
                    self.flight.publish(route["ticket"], channel, route["target"])
                    outgoing.setdefault(route["source"], {}).setdefault(route["edge"], []).append(channel)
                if route["target_worker"] == index:
                    channel = _channel(route["schema"], resources, route["source"], route["target"])
                    self.channels[f"in/{route['id']}"] = channel
                    incoming.setdefault(route["target"], {}).setdefault(route["port"], []).append(
                        (channel, route["target"])
                    )
            fragments = {f.fragment_id: f for f in spec.graph.fragments}
            snapshots = {s.fragment_id: s.payload for s in spec.source_snapshots}
            edges = {e.exchange_id: e for e in spec.graph.exchanges}
            for identity in reversed(spec.graph.topological_fragment_ids()):
                fragment = fragments[identity]
                for part in range(fragment.partition_count):
                    task = task_id(identity, part)
                    if task not in tasks:
                        continue
                    assignments = {
                        source.source_id: [split.split_id for split in source.splits[part :: fragment.partition_count]]
                        for source in fragment.sources
                    }
                    outputs = [
                        {
                            "channels": channels,
                            "producer": task,
                            "partitioning": edges[edge].partitioning if edge in edges else None,
                        }
                        for edge, channels in outgoing[task].items()
                    ]
                    self.service.prepare(
                        task,
                        fragment.native_plan,
                        spec.connection_snapshot,
                        snapshots[identity],
                        assignments,
                        incoming.get(task, {}),
                        outputs,
                    )
                    self.order.append(task)
            if self.stop.is_set():
                raise RuntimeError("query preparation was canceled")

    def connect(self, locations: dict[int, str]) -> None:
        with self.lifecycle:
            if self.stop.is_set() or self.started:
                raise RuntimeError("query no longer accepts input bindings")
            for route in self.routes:
                if route["target_worker"] == self.index:
                    self.flight.subscribe(
                        locations[route["source_worker"]],
                        route["ticket"],
                        self.channels[f"in/{route['id']}"],
                        route["source"],
                        self.spec.options.execution_timeout + self.spec.options.delivery_timeout,
                    )

    def start(self, tasks: list[str]) -> None:
        with self.lifecycle:
            if self.stop.is_set():
                raise RuntimeError("query is canceled")
            for task in tasks:
                if task not in self.order:
                    raise ValueError("task does not belong to this worker")
                self.service.start(task, "initial")
            if not self.started:
                self.started = True
                self.started_at = time.monotonic()
                self.thread = threading.Thread(target=self._run, name="vane-pipelined-worker", daemon=True)
                self.thread.start()

    def _run(self) -> None:
        try:
            while not self.stop.is_set():
                self.service.pump(max(1, len(self.order)))
                status = self.service.status()
                error = next((task["error"] for task in status if task["state"] in {"FAILED", "CANCELED"}), "")
                error = error or self.flight.error
                if error:
                    self.cancel(error)
                    return
                if time.monotonic() - self.started_at >= self.spec.options.execution_timeout:
                    self.service.expire()
                self.stop.wait(0.002)
        except BaseException as error:
            self.cancel(str(error))

    def snapshot(self) -> dict[str, Any]:
        tasks = self.service.diagnostics()
        production = self.service.production_status()
        channels = {name: channel.snapshot() for name, channel in self.channels.items()}
        error = self.error or self.flight.error or next((c["error"] for c in channels.values() if c["error"]), "")
        return {
            "tasks": tasks,
            "error": error,
            "production_done": production["finished"] and not production["error"],
            "active_contexts": sum(not t["released"] for t in tasks),
            "owned_bytes": sum(c["bytes"] for c in channels.values()),
            "channels": channels,
            "cleanup_complete": self.closed,
        }

    def cancel(self, reason: str) -> None:
        # Interrupt outside lifecycle/pump locks. Failure remains sticky in native.
        if self.closed:
            return
        self.error = self.error or reason
        self.stop.set()
        with self.control:
            if self.service is None and self.connection is not None:
                self.connection.interrupt()
        if self.service is not None:
            self.service.cancel(reason)
        if self.flight is not None:
            self.flight.cancel(reason)

    def close(self) -> None:
        self.cancel("worker query released")
        with self.lifecycle:
            if self.closed:
                return
            if self.thread is not None and self.thread is not threading.current_thread():
                self.thread.join(timeout=5)
                if self.thread.is_alive():
                    raise RuntimeError("native task cleanup is still pending")
            if self.service is not None:
                self.service.release()
            if self.flight is not None:
                self.flight.close()
            with self.control:
                if self.connection is not None:
                    self.connection.close()
                    self.connection = None
            self.closed = True


class PipelinedWorker:
    """One fixed worker epoch and an atomically accounted query registry."""

    def __init__(self, resources: RayResources) -> None:
        import vane

        self.resources = resources
        self.epoch = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.queries: dict[str, _Query] = {}
        self.attempts: dict[str, Any] = {}
        self.reservations: dict[str, dict[str, int]] = {}
        with vane.connect(backend="local") as connection:
            self.engine = native_plan_capabilities(connection).engine_identity

    def describe(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "engine": self.engine}

    def _check(self, epoch: str) -> None:
        if epoch != self.epoch:
            raise RuntimeError("worker epoch changed; a new attempt requires scheduler admission")

    def prepare(
        self,
        epoch: str,
        encoded: dict[str, Any],
        index: int,
        tasks: list[str],
        routes: list[dict[str, Any]],
        frame_rows: int,
    ) -> str:

        self._check(epoch)
        spec = RayQuerySpec.from_dict(encoded, expected_engine_identity=self.engine)
        if spec.requires_replay:
            raise ValueError("pipelined worker does not accept FTE submissions")
        from vane.execution.worker_resources import pipelined_demand, worker_capacity

        reservation = pipelined_demand(self.resources, index, tasks, routes)
        capacity = worker_capacity(self.resources)
        with self.lock:
            if spec.query_id in self.reservations:
                raise ValueError("query already prepared")
            for name, amount in reservation.items():
                if amount + sum(r[name] for r in self.reservations.values()) > capacity[name]:
                    raise RuntimeError(f"worker has insufficient {name} capacity")
            resources = replace(self.resources, exchange=replace(self.resources.exchange, frame_rows=frame_rows))
            query = _Query(spec, resources, index, routes)
            self.reservations[spec.query_id] = reservation
            self.queries[spec.query_id] = query
        try:
            query.prepare(tasks, reservation["operator"])
            return str(query.flight.location)
        except BaseException as primary:
            try:
                self.release(epoch, spec.query_id)
            except BaseException as cleanup:
                # Retain the registered owner and its reservation for release retry.
                raise primary from cleanup
            raise

    def connect(self, epoch: str, query: str, locations: dict[int, str]) -> None:
        self._check(epoch)
        self.queries[query].connect(locations)

    def start(self, epoch: str, query: str, tasks: list[str]) -> None:
        self._check(epoch)
        self.queries[query].start(tasks)

    def status(self, epoch: str, query: str) -> dict[str, Any]:
        self._check(epoch)
        return {"epoch": self.epoch, **self.queries[query].snapshot()}

    def production(self, epoch: str, query: str) -> dict[str, Any]:
        self._check(epoch)
        value: dict[str, Any] = self.queries[query].service.production_status()
        return value

    def ready(self, epoch: str, query: str) -> bool:
        self._check(epoch)
        owner = self.queries[query]
        if owner.flight.error:
            raise RuntimeError(owner.flight.error)
        return bool(owner.flight.ready)

    def cancel(self, epoch: str, query: str, reason: str) -> None:
        self._check(epoch)
        with self.lock:
            owner = self.queries.get(query)
        if owner is not None:
            owner.cancel(reason)

    def release(self, epoch: str, query: str) -> None:
        self._check(epoch)
        with self.lock:
            owner = self.queries.get(query)
        if owner is not None:
            owner.close()
        with self.lock:
            if self.queries.get(query) is owner:
                self.queries.pop(query, None)
                self.reservations.pop(query, None)

    def resources_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {"epoch": self.epoch, "reservations": dict(self.reservations)}

    def check_store(self, epoch: str, descriptor: dict[str, str]) -> None:
        from pathlib import Path

        from vane.execution.materialized_store import StoreDescriptor

        self._check(epoch)
        store = StoreDescriptor.from_dict(descriptor)
        if not any(str(Path(c.root).resolve()) == store.root for c in self.resources.exchange_stores):
            raise ValueError("worker has no registration for this exchange store")
        store.check()

    def prepare_materialized(
        self,
        epoch: str,
        encoded: dict[str, Any],
        fragment_id: str,
        partition: int,
        upstream: dict[str, Any],
        reserved: dict[str, Any],
        lease: dict[str, Any],
        frame_rows: int,
    ) -> None:
        from pathlib import Path

        from vane.execution.fte_plan import bind_task
        from vane.execution.fte_worker import MaterializedAttempt
        from vane.execution.materialized_exchange import StageManifest
        from vane.execution.materialized_store import AttemptReservation

        self._check(epoch)
        spec = RayQuerySpec.from_dict(encoded, expected_engine_identity=self.engine)
        if not spec.requires_replay:
            raise ValueError("materialized worker requires FTE execution")
        reservation = AttemptReservation.from_dict(reserved)
        token = reservation.token
        if reservation.engine_identity != self.engine or token.worker_epoch != epoch or token.query_id != spec.query_id:
            raise ValueError("attempt engine, epoch or query identity mismatch")
        options = spec.options.target
        assert hasattr(options, "fte_options") and options.fte_options is not None
        registered = next(
            (s for s in self.resources.exchange_stores if s.name == options.fte_options.exchange_store), None
        )
        if registered is None or str(Path(registered.root).resolve()) != reservation.store.root:
            raise ValueError("worker exchange store registration mismatch")
        self.check_store(epoch, reservation.store.to_dict())
        fragment = next(f for f in spec.graph.fragments if f.fragment_id == fragment_id)
        binding = bind_task(
            spec, fragment, partition, {name: StageManifest.from_dict(s) for name, s in upstream.items()}
        )
        if (token.task_id, token.stage_id, token.input_id) != (
            binding.task.task_id,
            fragment_id,
            binding.task.input_id,
        ):
            raise ValueError("attempt does not have the declared immutable input")
        if tuple(o.output for o in reservation.objects) != binding.task.outputs:
            raise ValueError("attempt output partitions differ from the fragment graph")
        if type(frame_rows) is not int or not 0 < frame_rows <= self.resources.exchange.frame_rows:
            raise ValueError("invalid materialized frame row capacity")
        from vane.execution.worker_resources import materialized_demand, worker_capacity

        demand = materialized_demand(self.resources, binding)
        capacity = worker_capacity(self.resources)
        key = f"fte/{spec.query_id}/{token.fence}"
        resources = replace(self.resources, exchange=replace(self.resources.exchange, frame_rows=frame_rows))

        def orphan() -> None:
            from vane.execution.fte_store import StorePool

            self.release_materialized(epoch, key)
            StorePool(registered).collect_expired()

        with self.lock:
            if key in self.reservations:
                raise ValueError("attempt already prepared")
            if any(
                amount + sum(r[name] for r in self.reservations.values()) > capacity[name]
                for name, amount in demand.items()
            ):
                raise RuntimeError("worker has insufficient materialized capacity")
            owner = MaterializedAttempt(spec, resources, binding, reservation, lease, orphan)
            self.reservations[key] = demand
            self.attempts[key] = owner
        try:
            owner.prepare(demand["operator"])
        except BaseException as primary:
            try:
                self.release_materialized(epoch, key)
            except BaseException as cleanup:
                raise primary from cleanup
            raise

    def materialized_status(self, epoch: str, key: str) -> dict[str, Any]:
        self._check(epoch)
        with self.lock:
            owner = self.attempts[key]
        return {"epoch": self.epoch, **owner.status()}

    def cancel_materialized(self, epoch: str, key: str, reason: str) -> None:
        self._check(epoch)
        with self.lock:
            owner = self.attempts.get(key)
        if owner is not None:
            owner.cancel(reason)

    def release_materialized(self, epoch: str, key: str) -> None:
        self._check(epoch)
        with self.lock:
            owner = self.attempts.get(key)
        if owner is not None:
            owner.close()
        with self.lock:
            if self.attempts.get(key) is owner:
                self.attempts.pop(key, None)
                self.reservations.pop(key, None)


class _ResultSession:
    """One query's native relay, never reused across result leases."""

    def __init__(self, epoch: str, resources: RayResources) -> None:
        self.epoch = epoch
        self.resources = resources
        self.channel: Any = None
        self.flight: Any = None
        self.schema = b""
        self.materialized: Any = None
        self.store_lease: Any = None
        self.stop = threading.Event()
        self.lifecycle = threading.RLock()
        self.watchdog: threading.Thread | None = None

    def prepare(self, epoch: str, schema: bytes, ticket: str) -> str:
        from vane._native import execution_runtime as native

        with self.lifecycle:
            if epoch != self.epoch or self.flight is not None or self.stop.is_set():
                raise RuntimeError("invalid result service epoch or duplicate preparation")
            self.schema = schema
            self.channel = _channel(schema, self.resources, "root", "client")
            self.flight = native.DirectFlight(
                "0.0.0.0",
                _host(),
                2,
                2 * native.DirectFlight.staging_per_link(self.resources.exchange.frame_bytes),
                self.resources.exchange.frame_bytes,
            )
            self.flight.publish(ticket, self.channel, "client")
            return str(self.flight.location)

    def connect(self, location: str, ticket: str, timeout: float) -> None:
        with self.lifecycle:
            if self.stop.is_set():
                raise RuntimeError("result service is canceled")
            self.flight.subscribe(location, ticket, self.channel, "root", timeout)

    def status(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "channel": self.channel.snapshot(),
            "error": self.flight.error or (self.materialized.status()["error"] if self.materialized else ""),
            "ready": self.flight.ready,
        }

    def cancel(self, reason: str) -> None:
        self.stop.set()
        if self.materialized is not None:
            self.materialized.cancel(reason)
        if self.flight is not None:
            self.flight.cancel(reason)

    def release(self) -> None:
        self.cancel("result service released")
        with self.lifecycle:
            if self.materialized is not None:
                self.materialized.close()
            if self.flight is not None:
                self.flight.close()
            if self.store_lease is not None:
                self.store_lease.close()
        if self.watchdog is not None and self.watchdog is not threading.current_thread():
            self.watchdog.join(timeout=5)
            if self.watchdog.is_alive():
                raise RuntimeError("result service watchdog cleanup is pending")

    def connect_materialized(self, epoch: str, manifest: dict[str, Any], lease: dict[str, Any]) -> None:
        import json

        from vane._native import execution_plan
        from vane._native import execution_runtime as native
        from vane.execution.fte_store import ActiveStoreLease
        from vane.execution.materialized_exchange import ResultManifest

        with self.lifecycle:
            if epoch != self.epoch or self.materialized is not None or self.stop.is_set():
                raise RuntimeError("result service no longer accepts a manifest")
            result = ResultManifest.from_dict(manifest)
            if (
                result.stage.engine_identity != execution_plan.engine_identity()
                or result.output.output.schema != self.schema
            ):
                raise ValueError("result manifest native engine or schema mismatch")
            self.store_lease = ActiveStoreLease(lease)
            # Own cleanup as soon as the lease is pinned, including failures
            # while validating a manifest whose coordinator subsequently exits.
            self.watchdog = threading.Thread(target=self._watch_materialized, name="vane-fte-result-lease", daemon=True)
            self.watchdog.start()
            value = self.store_lease.check()
            if (
                value["query_id"] != result.stage.query_id
                or result.output.key.split("/")[0] != self.store_lease.namespace
            ):
                raise ValueError("result belongs to another query storage lease")
            path = self.store_lease.directory / "result.json"
            if path.is_symlink() or path.stat().st_size > 32 << 20 or json.loads(path.read_bytes()) != manifest:
                raise ValueError("result manifest has not been published by its coordinator")
            obj = result.output
            self.materialized = native.MaterializedIO.read(
                str(self.store_lease.store.path(obj.key)),
                self.channel,
                "root",
                obj.metadata.bytes,
                native.MaterializedIO.staging_bytes(self.resources.exchange.frame_bytes),
                obj.metadata.to_dict(),
            )

    def _watch_materialized(self) -> None:
        while not self.stop.is_set():
            try:
                value = self.store_lease.check()
            except BaseException as error:
                self.cancel(str(error))
                self.release()
                return
            self.stop.wait(min(0.5, value["seconds"] / 4))


class ResultService:
    """Reusable process with one exclusive, epoch-fenced result lease at a time."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.session: _ResultSession | None = None

    def reserve(self, epoch: str, resources: RayResources) -> None:
        with self.lock:
            if self.session is not None:
                raise RuntimeError("result service already has a lease")
            self.session = _ResultSession(epoch, resources)

    def _session(self, epoch: str) -> _ResultSession:
        with self.lock:
            if self.session is None or self.session.epoch != epoch:
                raise RuntimeError("stale result service lease")
            return self.session

    def prepare(self, epoch: str, schema: bytes, ticket: str) -> str:
        return self._session(epoch).prepare(epoch, schema, ticket)

    def connect(self, epoch: str, location: str, ticket: str, timeout: float) -> None:
        self._session(epoch).connect(location, ticket, timeout)

    def connect_materialized(self, epoch: str, manifest: dict[str, Any], lease: dict[str, Any]) -> None:
        self._session(epoch).connect_materialized(epoch, manifest, lease)

    def status(self, epoch: str) -> dict[str, Any]:
        return self._session(epoch).status()

    def cancel(self, epoch: str, reason: str) -> None:
        with self.lock:
            session = self.session
        if session is not None and session.epoch == epoch:
            session.cancel(reason)

    def release(self, epoch: str) -> None:
        with self.lock:
            session = self.session
        if session is None or session.epoch != epoch:
            return
        # Close the native endpoints and join the lease watchdog before the
        # driver can return this actor to its idle pool. RPCs already in flight
        # retain the old session; later RPCs must present the new lease epoch.
        session.release()
        with self.lock:
            if self.session is session:
                self.session = None
