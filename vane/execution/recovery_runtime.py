# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage barriers, fenced task retries, and committed native result delivery."""

from __future__ import annotations

import errno
import hashlib
import secrets
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, TypeVar

from vane.execution.cleanup_deadline import cleanup_timeout
from vane.execution.compiler import FragmentCompileOptions
from vane.execution.fte_plan import TaskBinding, bind_task
from vane.execution.fte_store import QueryStoreLease, StorePool
from vane.execution.materialized_exchange import AttemptManifest, AttemptToken, ResultManifest, StageManifest
from vane.execution.materialized_store import AttemptReservation, CommitCoordinator, ReadLease, StorageCleanupPending
from vane.execution.pipelined_plan import DirectTicket
from vane.execution.pipelined_runtime import PipelinedContext, WorkerPool, _get
from vane.execution.query_options import QueryExecutionOptions, RayExecution
from vane.execution.resource_demand import ResourceDemand
from vane.execution.result_consumer import NativeResultConsumer
from vane.execution.submission import RayQuerySpec, stage_ray_query
from vane.execution.worker_resources import materialized_demand

T = TypeVar("T")


@dataclass
class RunningAttempt:
    index: int
    worker: Any
    epoch: str
    partition: int
    binding: TaskBinding
    reservation: AttemptReservation
    sequence: int
    prepare: Any = None
    capacity_token: str = ""

    @property
    def key(self) -> str:
        token = self.reservation.token
        return f"fte/{token.query_id}/{token.fence}"


class RecoveryScheduler:
    def __init__(
        self,
        pool: WorkerPool,
        store: StorePool,
        context: PipelinedContext,
        rows_per_batch: int,
        consumer: NativeResultConsumer | None = None,
    ) -> None:
        self.pool, self.store, self.context = pool, store, context
        self.resources = replace(
            pool.resources,
            exchange=replace(
                pool.resources.exchange, frame_rows=min(rows_per_batch, pool.resources.exchange.frame_rows)
            ),
        )
        self.spec: RayQuerySpec
        self.schema: Any = None
        self.lease: QueryStoreLease | None = None
        self.coordinator: CommitCoordinator | None = None
        self.read_lease: ReadLease | None = None
        self.relay: Any = None
        self.result_id = ""
        self.consumer = NativeResultConsumer() if consumer is None else consumer
        self.planning_connection: Any = None
        self.result_endpoint: dict[str, str] = {}
        self.result_manifest: ResultManifest | None = None
        self.completed_at: float | None = None
        self.thread: threading.Thread | None = None
        self.heartbeat: threading.Thread | None = None
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.cleanup_lock = threading.Lock()
        self.active: dict[int, RunningAttempt] = {}
        self.retired: list[AttemptToken] = []
        self.history: deque[AttemptToken] = deque(maxlen=4096)
        self.closed = False

    def prepare(
        self,
        connection: Any,
        sql: str,
        options: QueryExecutionOptions,
        demand: ResourceDemand,
        compile_options: FragmentCompileOptions,
    ) -> None:
        from vane._native import execution_plan
        from vane._native import execution_runtime as native

        self.planning_connection = connection
        try:
            self.context.check()
            self.lease = self.store.reserve(self.context.query_id)
            self.heartbeat = threading.Thread(target=self._heartbeat, name="vane-fte-store-heartbeat", daemon=True)
            self.heartbeat.start()
            engine = execution_plan.engine_identity()
            self.coordinator = CommitCoordinator(
                self.store.store,
                self.context.query_id,
                engine,
                (),
                max_bytes=self.store.config.query_bytes,
                namespace=self.lease.value["namespace"],
            )
            self.pool.ensure(self.context, engine)
            for index in range(len(self.pool.workers)):
                import ray

                worker, epoch = self.pool.workers[index], self.pool.epochs[index]
                try:
                    _get(worker.check_store.remote(epoch, self.store.store.descriptor.to_dict()), self.context)
                except ray.exceptions.RayActorError:
                    self.pool.replace(index, epoch, self.context, engine)
                    _get(
                        self.pool.workers[index].check_store.remote(
                            self.pool.epochs[index], self.store.store.descriptor.to_dict()
                        ),
                        self.context,
                    )
            self.spec, source_bytes = stage_ray_query(
                connection,
                sql,
                query_id=self.context.query_id,
                options=options,
                resources=demand,
                source_directory=str(self.lease.directory / "sources"),
                source_budget=self.store.config.source_bytes,
                compile_options=compile_options,
            )
            self.coordinator.source_bytes(source_bytes)
            fragments = {f.fragment_id: f for f in self.spec.graph.fragments}
            outputs = 1 + sum(
                fragments[e.producer_fragment_id].partition_count * fragments[e.consumer_fragment_id].partition_count
                for e in self.spec.graph.exchanges
            )
            if outputs > 4096 or sum(f.partition_count for f in fragments.values()) > 4096:
                raise ValueError("FTE graph exceeds the task/partition metadata limit")
            self.schema = native.arrow_schema(self.spec.result_schema, list(self.spec.result_names))
            self._prepare_delivery()
            self.context.deadline_probe = self.production_status
            self.context.check()
            # thread.start() may run the first attempt before returning. Its
            # failure cleanup must not interrupt an already finished planner.
            self.planning_connection = None
            self.thread = threading.Thread(target=self._run, name="vane-recovery-scheduler", daemon=True)
            self.thread.start()
        finally:
            self.planning_connection = None

    def _prepare_delivery(self) -> None:
        resources = self.resources
        self.relay, self.result_id = self.pool.results.create(self.context.query_id, resources)
        self.pool.results.prepared(self.result_id, self.context)
        ticket = DirectTicket(
            self.spec.query_id,
            "committed-result",
            self.result_id,
            uuid.uuid4().hex,
            "client-result",
            "result-service",
            "client",
            0,
            hashlib.sha256(self.spec.result_schema).hexdigest(),
            secrets.token_urlsafe(32),
        ).encode()
        location = _get(self.relay.prepare.remote(self.result_id, self.spec.result_schema, ticket), self.context)
        self.result_endpoint = {"location": location, "ticket": ticket}
        self.consumer.attach(self.spec, resources, location, ticket)
        deadline = time.monotonic() + self.spec.options.admission_timeout
        while True:
            self.context.check()
            status = _get(self.relay.status.remote(self.result_id), self.context)
            if status["error"] or self.consumer.error:
                raise RuntimeError(status["error"] or self.consumer.error)
            if status["ready"] and self.consumer.ready:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("FTE result consumer admission timed out")
            self.stop.wait(0.01)

    def _heartbeat(self) -> None:
        assert self.lease is not None
        interval = min(1.0, self.store.config.lease_seconds / 4)
        try:
            while not self.stop.wait(interval):
                self.lease.renew()
                self.store.request_collection()
        except BaseException as error:
            if not self.stop.is_set():
                self.context.failed(str(error))
                self.cancel(str(error))

    def production_status(self) -> bool:
        return self.completed_at is not None and self.completed_at <= (
            self.context._ticket.claimed_at + self.context.options.execution_timeout
        )

    def _dispatch(self, index: int, partition: int, binding: TaskBinding, upstream: dict[str, StageManifest]) -> bool:
        assert self.coordinator is not None and self.lease is not None
        token = f"fte/{self.spec.query_id}/{binding.task.task_id}/{index}"
        with self.lock:
            self.context.check()
            if self.stop.is_set():
                raise RuntimeError("query canceled before materialized task dispatch")
            # Enqueue and cancellation share the lock: after cancel_waiting()
            # clears this query, no dispatch may leave a new orphan FIFO head.
            if not self.pool.admission.try_acquire(
                token, self.spec.query_id, {index: materialized_demand(self.resources, binding)}
            ):
                return False
            try:
                worker, epoch = self.pool.workers[index], self.pool.epochs[index]
                reserved = self.coordinator.begin(
                    binding.task.task_id, epoch, object_bytes=self.store.config.object_bytes
                )
                attempt = RunningAttempt(
                    index,
                    worker,
                    epoch,
                    partition,
                    binding,
                    reserved,
                    self.pool.next_call(index, worker, epoch),
                    capacity_token=token,
                )
                self.active[index] = attempt
                self.history.append(reserved.token)
                attempt.prepare = worker.prepare_materialized.remote(
                    epoch,
                    self.spec.to_dict(),
                    binding.task.stage_id,
                    partition,
                    {name: stage.to_dict() for name, stage in upstream.items()},
                    reserved.to_dict(),
                    self.lease.to_dict(),
                    self.resources.exchange.frame_rows,
                    attempt.sequence,
                )
                return True
            except BaseException:
                if index not in self.active:
                    self.pool.admission.release(token)
                raise

    def _release(self, attempt: RunningAttempt) -> None:
        import ray

        try:
            timeout = cleanup_timeout(5)
            _get(
                attempt.worker.release_materialized.remote(attempt.epoch, attempt.key, attempt.sequence),
                timeout=timeout,
            )
        except ray.exceptions.ActorDiedError:
            pass
        with self.lock:
            if self.active.get(attempt.index) is attempt:
                self.active.pop(attempt.index)
            self.pool.admission.release(attempt.capacity_token)

    def _discard_retired(self) -> None:
        assert self.coordinator is not None
        remaining = []
        for token in self.retired:
            try:
                self.coordinator.discard(token)
            except StorageCleanupPending:
                remaining.append(token)
        self.retired = remaining

    def _retry(self, attempt: RunningAttempt, error: BaseException) -> None:
        assert isinstance(self.spec.options.target, RayExecution)
        policy = self.spec.options.target.fte_options
        assert policy is not None
        self.context.check()
        self._release(attempt)
        self.retired.append(attempt.reservation.token)
        self._discard_retired()
        if attempt.reservation.token.attempt >= policy.max_attempts:
            raise RuntimeError(f"FTE attempt limit exhausted for {attempt.binding.task.task_id}") from error
        self.pool.replace(attempt.index, attempt.epoch, self.context, self.spec.graph.engine_identity)
        end = time.monotonic() + policy.retry_backoff_seconds
        while time.monotonic() < end:
            self.context.check()
            self.stop.wait(min(0.05, end - time.monotonic()))
        self.context.check()

    def _publish(self, operation: Callable[[], T]) -> T:
        """Retry only idempotent metadata publication, on the query deadline."""
        for retry in range(3):
            self.context.check()
            try:
                return operation()
            except OSError as error:
                if retry == 2 or error.errno not in {errno.EAGAIN, errno.EINTR, errno.ETIMEDOUT, errno.ECONNRESET}:
                    raise
                self.stop.wait(0.01 * (retry + 1))
        raise AssertionError("publication retry did not terminate")

    def _stage(self, fragment: Any, upstream: dict[str, StageManifest]) -> StageManifest:
        import ray

        assert self.coordinator is not None
        coordinator = self.coordinator
        bindings = [bind_task(self.spec, fragment, part, upstream) for part in range(fragment.partition_count)]
        self.coordinator.declare_stage(tuple(binding.task for binding in bindings))
        pending = deque(range(fragment.partition_count))
        waiting_worker: int | None = None
        while pending or self.active:
            self.context.check()
            self._discard_retired()
            for index in range(len(self.pool.workers)):
                if waiting_worker is not None and index != waiting_worker:
                    continue
                if pending and index not in self.active:
                    partition = pending[0]
                    if self._dispatch(index, partition, bindings[partition], upstream):
                        pending.popleft()
                        waiting_worker = None
                    else:
                        waiting_worker = index
                        break
            for attempt in tuple(self.active.values()):
                self.context.check()
                try:
                    # Preparation can hash large frozen sources. It obeys the
                    # query deadline instead of a short liveness RPC timeout.
                    if attempt.prepare is not None:
                        _get(attempt.prepare, self.context, timeout=self.spec.options.execution_timeout)
                        attempt.prepare = None
                    status = _get(
                        attempt.worker.materialized_status.remote(attempt.epoch, attempt.key), self.context, timeout=5
                    )
                except ray.exceptions.RayActorError as error:
                    self._retry(attempt, error)
                    # Preserve a queued admission's partition/token. Moving a
                    # retry ahead of it would leave the old FIFO head orphaned.
                    pending.append(attempt.partition)
                    continue
                if status["epoch"] != attempt.epoch:
                    raise RuntimeError("materialized attempt worker epoch changed")
                if status["error"]:
                    # SQL/capacity failures and missing or corrupt committed
                    # input are permanent. They cannot trigger recomputation.
                    raise RuntimeError(status["error"])
                if status["sealed"]:
                    manifest = AttemptManifest.from_dict(status["manifest"])
                    self._publish(lambda: coordinator.commit(manifest))
                    self._release(attempt)
            self.stop.wait(0.005)
        return self._publish(lambda: coordinator.seal_stage(fragment.fragment_id))

    def _run(self) -> None:
        assert self.coordinator is not None and self.lease is not None
        coordinator = self.coordinator
        try:
            stages: dict[str, StageManifest] = {}
            fragments = {f.fragment_id: f for f in self.spec.graph.fragments}
            for identity in self.spec.graph.topological_fragment_ids():
                incoming = {
                    e.producer_fragment_id: stages[e.producer_fragment_id]
                    for e in self.spec.graph.exchanges
                    if e.consumer_fragment_id == identity
                }
                stages[identity] = self._stage(fragments[identity], incoming)
            self.context.check()
            root = stages[self.spec.graph.result.fragment_id]
            self.read_lease = self.coordinator.retain(root)
            self.result_manifest = self._publish(lambda: coordinator.publish_result(root, self.spec.result_names))
            self.completed_at = time.monotonic()
            self.context.check()
            self.context.produced()
            _get(
                self.relay.connect_materialized.remote(
                    self.result_id, self.result_manifest.to_dict(), self.lease.to_dict()
                ),
                self.context,
            )
            while not self.stop.is_set():
                self.context.check()
                status = _get(self.relay.status.remote(self.result_id), self.context, timeout=5)
                if status["query_id"] != self.result_id:
                    raise RuntimeError("result context identity changed")
                error = status["error"] or status["channel"]["error"] or self.consumer.error
                if error:
                    raise RuntimeError(error)
                self.stop.wait(0.02)
        except BaseException as error:
            if not self.stop.is_set():
                self.context.failed(str(error))
                self.cancel(str(error))

    def delivery_complete(self) -> bool:
        if not self.consumer.delivered() or self.result_manifest is None or not self.production_status():
            return False
        status = _get(self.relay.status.remote(self.result_id), self.context)
        if status["query_id"] != self.result_id:
            raise RuntimeError("result context identity changed")
        error = status["error"] or status["channel"]["error"] or self.consumer.error
        if error:
            raise RuntimeError(error)
        return True

    def read_next_batch(self) -> Any:
        while True:
            self.context.check()
            state, batch = self.consumer.channel.poll("client")
            if state == "data":
                if self.result_manifest is None or not self.production_status():
                    batch.close()
                    raise RuntimeError("FTE data arrived before root result commit")
                try:
                    return batch.to_arrow(list(self.spec.result_names))
                finally:
                    batch.close()
            if state == "end":
                if self.result_manifest is None or not self.production_status():
                    raise RuntimeError("FTE result ended before commit")
                status = _get(self.relay.status.remote(self.result_id), self.context)
                error = status["error"] or status["channel"]["error"] or self.consumer.error
                if error:
                    raise RuntimeError(error)
                self.context.check()
                raise StopIteration
            if state == "closed":
                raise RuntimeError("FTE result channel closed before completion")
            self.stop.wait(0.002)

    def cancel(self, reason: str = "query canceled") -> None:
        import ray

        with self.lock:
            if self.closed:
                return
            self.stop.set()
            self.pool.admission.cancel_waiting(self.context.query_id)
            if self.coordinator is not None:
                self.coordinator.cancel(reason)
            if self.planning_connection is not None:
                self.planning_connection.interrupt()
            if self.consumer.flight is not None:
                self.consumer.cancel(reason)
            if ray.is_initialized():
                for attempt in tuple(self.active.values()):
                    attempt.worker.cancel_materialized.remote(attempt.epoch, attempt.key, reason)
                if self.relay is not None:
                    self.relay.cancel.remote(self.result_id, reason)

    def close(self) -> None:
        import ray

        with self.cleanup_lock:
            if self.closed:
                return
            self.context.deadline_probe = None
            self.cancel("query result released")
            for thread in (self.thread, self.heartbeat):
                if thread is not None and thread is not threading.current_thread():
                    thread.join(timeout=cleanup_timeout(10))
                    if thread.is_alive():
                        raise RuntimeError("FTE query cleanup is pending")
            if self.consumer.flight is not None:
                self.consumer.close()
            errors = []
            for attempt in tuple(self.active.values()):
                try:
                    if ray.is_initialized():
                        self._release(attempt)
                    else:
                        self.active.pop(attempt.index, None)
                        self.pool.admission.release(attempt.capacity_token)
                except BaseException as error:
                    errors.append(error)
            try:
                self.pool.results.release(self.pool.results.actor, self.context.query_id)
            except BaseException as error:
                errors.append(error)
            if errors:
                raise RuntimeError("FTE native cleanup is pending; retry result.close()") from errors[0]
            if self.read_lease is not None:
                self.read_lease.close()
            # Ray can report actor death before the process has actually
            # released its file locks. Give normal process teardown a bounded
            # grace period; keep quota owned if native I/O still has not exited.
            deadline = time.monotonic() + cleanup_timeout(5)
            while True:
                try:
                    if self.coordinator is not None:
                        for reservation in tuple(self.coordinator._reservations.values()):
                            committed = self.coordinator._committed.get(reservation.token.task_id)
                            if committed is None or committed.token != reservation.token:
                                self.coordinator.discard(reservation.token)
                    if self.lease is not None:
                        self.lease.close(self.coordinator.close if self.coordinator is not None else lambda: None)
                    break
                except StorageCleanupPending:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
            self.closed = True

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "active": {a.key: a.reservation.token.to_dict() for a in self.active.values()},
                "history": [t.to_dict() for t in self.history],
                "committed": self.result_manifest is not None,
                "store": self.coordinator.snapshot() if self.coordinator is not None else None,
            }

    def diagnostics(self) -> dict[str, Any]:
        with self.lock:
            active = tuple(self.active.values())
        attempts: dict[str, Any] = {}
        for attempt in active:
            try:
                attempts[attempt.key] = _get(
                    attempt.worker.materialized_status.remote(attempt.epoch, attempt.key), timeout=2
                )
            except Exception as error:
                attempts[attempt.key] = {"unavailable": str(error)}
        return {
            "mode": "fte",
            **self.snapshot(),
            "attempts": attempts,
            "result_channel": self.consumer.channel.snapshot() if self.consumer.channel is not None else None,
            "resources": self.pool.admission.snapshot(),
            "cleanup_complete": self.closed,
        }
