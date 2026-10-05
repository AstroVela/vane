# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""One native fragment attempt between immutable objects and private outputs."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from typing import Any

from vane.execution.fte_plan import TaskBinding
from vane.execution.fte_store import ActiveStoreLease
from vane.execution.materialized_exchange import ObjectMeta
from vane.execution.materialized_store import AttemptReservation
from vane.execution.pipelined_plan import RayResources
from vane.execution.submission import RayQuerySpec, prepare_worker_plan


class MaterializedAttempt:
    def __init__(
        self,
        spec: RayQuerySpec,
        resources: RayResources,
        binding: TaskBinding,
        reservation: AttemptReservation,
        lease: dict[str, Any],
        on_orphan: Callable[[], None],
    ) -> None:
        self.spec, self.resources, self.binding = spec, resources, binding
        self.reservation, self.lease_wire, self.on_orphan = reservation, lease, on_orphan
        self.lease: ActiveStoreLease | None = None
        self.attempt_guard: Any = None
        self.connection: Any = None
        self.service: Any = None
        self.readers: list[Any] = []
        self.writers: dict[str, Any] = {}
        self.channels: dict[str, Any] = {}
        self.thread: threading.Thread | None = None
        self.watchdog: threading.Thread | None = None
        self.stop = threading.Event()
        self.lifecycle = threading.RLock()
        self.control = threading.Lock()
        self.started = False
        self.closed = False
        self.error = ""

    def prepare(self, operator_bytes: int) -> None:
        import vane
        from vane._native import execution_runtime as native
        from vane.execution.pipelined_worker import _channel

        with self.lifecycle:
            if self.stop.is_set():
                raise RuntimeError("attempt preparation canceled")
            self.lease = ActiveStoreLease(self.lease_wire)
            if self.lease.store != self.reservation.store or any(
                o.key.split("/")[0] != self.lease.namespace for o in self.reservation.objects
            ):
                raise ValueError("attempt does not belong to its query storage lease")
            token = self.reservation.token
            self.attempt_guard = native.StoreGuard.acquire(
                str(self.lease.directory / token.fence / "io.lock"),
                False,
                False,
            )
            if self.attempt_guard is None:
                raise RuntimeError("attempt storage is being cleaned")
            self._check_fence()
            self.watchdog = threading.Thread(target=self._watch, name="vane-fte-lease", daemon=True)
            self.watchdog.start()
            with self.control:
                if self.stop.is_set():
                    raise RuntimeError(self.error or "attempt preparation canceled")
                self.connection = vane.connect(
                    backend="local",
                    config={
                        "threads": self.resources.cpus_per_worker,
                        "memory_limit": f"{operator_bytes}B",
                    },
                )
            prepare_worker_plan(self.connection, self.spec)
            if self.stop.is_set():
                raise RuntimeError(self.error or "attempt preparation canceled")
            self.service = native.TaskService(self.connection)
            inputs: dict[str, list[Any]] = {}
            for port, objects in self.binding.inputs.items():
                for index, obj in enumerate(objects):
                    if obj.key.split("/")[0] != self.lease.namespace:
                        raise ValueError("input object belongs to another query namespace")
                    producer = f"{port}/{index}"
                    channel = _channel(obj.output.schema, self.resources, producer, token.task_id)
                    self.channels[f"input/{producer}"] = channel
                    self.readers.append(
                        native.MaterializedIO.read(
                            str(self.reservation.store.path(obj.key)),
                            channel,
                            producer,
                            obj.metadata.bytes,
                            native.MaterializedIO.staging_bytes(self.resources.exchange.frame_bytes),
                            obj.metadata.to_dict(),
                        )
                    )
                    inputs.setdefault(port, []).append((channel, token.task_id))
            outputs: dict[str, list[Any]] = {}
            for output in self.reservation.objects:
                channel = _channel(output.output.schema, self.resources, token.task_id, "store")
                self.channels[f"output/{output.key}"] = channel
                self.writers[output.key] = native.MaterializedIO.write(
                    str(self.reservation.store.path(output.key)),
                    channel,
                    "store",
                    output.max_bytes,
                    native.MaterializedIO.staging_bytes(self.resources.exchange.frame_bytes),
                )
                outputs.setdefault(output.output.exchange_id, []).append(channel)
            fragment = next(f for f in self.spec.graph.fragments if f.fragment_id == token.stage_id)
            snapshot = next(s.payload for s in self.spec.source_snapshots if s.fragment_id == token.stage_id)
            edges = {e.exchange_id: e for e in self.spec.graph.exchanges}
            self.service.prepare(
                token.task_id,
                fragment.native_plan,
                self.spec.connection_snapshot,
                snapshot,
                self.binding.assignments,
                inputs,
                [
                    {
                        "channels": channels,
                        "producer": token.task_id,
                        "partitioning": edges[edge].partitioning if edge in edges else None,
                    }
                    for edge, channels in outputs.items()
                ],
            )
            self._check_fence()
            if self.stop.is_set():
                raise RuntimeError(self.error or "attempt canceled before start")
            self.service.start(token.task_id, token.fence)
            self.started = True
            self.thread = threading.Thread(target=self._run, name="vane-fte-native-task", daemon=True)
            self.thread.start()

    def _check_fence(self) -> dict[str, Any]:
        assert self.lease is not None
        value = self.lease.check()
        name = hashlib.sha256(self.reservation.token.task_id.encode()).hexdigest()
        path = self.lease.directory / f"task-{name}.json"
        if path.is_symlink() or path.stat().st_size > 1024:
            raise ValueError("invalid task fence metadata")
        if json.loads(path.read_bytes()) != {"fence": self.reservation.token.fence}:
            raise RuntimeError("materialized attempt was fenced")
        return value

    def _watch(self) -> None:
        while not self.closed:
            try:
                value = self._check_fence()
            except BaseException as error:
                self.cancel(str(error))
                # A blocked cleanup retains both its registry reservation and
                # file locks. Retry until native exits; never free live objects.
                while True:
                    try:
                        self.on_orphan()
                    except BaseException:
                        if bool(self.closed):
                            # Native resources are gone. A store outage can be
                            # retried by the shared janitor/next admission.
                            return
                        time.sleep(0.05)
                    else:
                        return
            time.sleep(min(0.5, value["seconds"] / 4))

    def _run(self) -> None:
        try:
            while not self.stop.is_set():
                self.service.pump(1)
                status = self.status()
                if status["error"]:
                    self.cancel(status["error"])
                    return
                if status["sealed"]:
                    return
                self.stop.wait(0.002)
        except BaseException as error:
            self.cancel(str(error))

    def status(self) -> dict[str, Any]:
        # This probe never waits for pump's executor lock.
        production = self.service.production_status() if self.service is not None else {"finished": False, "error": ""}
        writers = {key: io.status() for key, io in tuple(self.writers.items())}
        readers = [io.status() for io in tuple(self.readers)]
        error = (
            self.error
            or production["error"]
            or next((s["error"] for s in [*writers.values(), *readers] if s["error"]), "")
        )
        sealed = (
            self.started
            and production["finished"]
            and len(writers) == len(self.reservation.objects)
            and all(s["done"] for s in [*writers.values(), *readers])
            and not error
        )
        manifest = (
            self.reservation.seal({key: ObjectMeta.from_dict(value["object"]) for key, value in writers.items()})
            if sealed
            else None
        )
        return {
            "error": error,
            "sealed": bool(sealed),
            "manifest": manifest.to_dict() if manifest else None,
            "tasks": self.service.diagnostics() if self.service is not None else [],
            "channels": {key: c.snapshot() for key, c in tuple(self.channels.items())},
            "cleanup_complete": self.closed,
        }

    def cancel(self, reason: str) -> None:
        if self.closed:
            return
        self.error = self.error or reason
        self.stop.set()
        with self.control:
            if self.service is None and self.connection is not None:
                self.connection.interrupt()
        if self.service is not None:
            self.service.cancel(reason)
        for io in [*tuple(self.readers), *tuple(self.writers.values())]:
            io.cancel(reason)

    def close(self) -> None:
        self.cancel("attempt released")
        with self.lifecycle:
            if self.closed:
                return
            if self.thread is not None and self.thread is not threading.current_thread():
                self.thread.join(timeout=5)
                if self.thread.is_alive():
                    raise RuntimeError("native attempt cleanup is pending")
            for io in [*self.readers, *self.writers.values()]:
                io.close()
            if self.service is not None:
                self.service.release()
            with self.control:
                if self.connection is not None:
                    self.connection.close()
                    self.connection = None
            if self.attempt_guard is not None:
                self.attempt_guard.close()
            if self.lease is not None:
                self.lease.close()
            self.closed = True
