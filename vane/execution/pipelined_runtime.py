# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared Ray query runtime, pipelined scheduling and native result delivery."""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from vane.execution.cleanup_deadline import cleanup_deadline, cleanup_timeout
from vane.execution.compiler import FragmentCompileOptions
from vane.execution.pipelined_plan import DirectTicket, RayResources, placement, task_id
from vane.execution.query_options import DistributedMode, FteOptions, QueryExecutionOptions, RayExecution
from vane.execution.query_runtime import QueryContext, QueryRuntime
from vane.execution.resource_demand import MemoryDemand, ResourceDemand
from vane.execution.result_consumer import NativeResultConsumer
from vane.execution.submission import prepare_ray_query
from vane.execution.worker_resources import WorkerResourceManager, pipelined_demand, worker_capacity


def _get(reference: Any, context: QueryContext | None = None, timeout: float = 30) -> Any:
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray session is no longer connected")
    deadline = time.monotonic() + cleanup_timeout(timeout)
    while True:
        if context is not None:
            context.check()
        ready, _ = ray.wait([reference], timeout=min(0.05, max(0.0, deadline - time.monotonic())))
        if ready:
            return ray.get(reference)
        if time.monotonic() >= deadline:
            raise TimeoutError("Ray control operation timed out")


class PipelinedContext(QueryContext):
    """Execution deadlines stop at production completion; delivery stays bounded."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.production_done = False
        self.failure = ""
        self.deadline_probe: Callable[[], bool] | None = None
        self._expiry_lock = threading.RLock()

    def begin(self, *, defer_execution: bool = False) -> Any:
        assert isinstance(self.options.target, RayExecution)
        return super().begin(defer_execution=self.options.target.mode is DistributedMode.PIPELINED)

    def produced(self) -> None:
        with self._lock:
            self.production_done = True
            if self._deadline is not None:
                self._deadline.close()
                self._deadline = None

    def _expire(self) -> None:
        if self.failure or self._ticket.cancellation_reason is not None or self._done:
            return
        deadline = self._deadline
        if self._admission_deadline is not None:
            super()._expire()
            return
        if bool(self.production_done) or deadline is None or not deadline.expired():
            return
        with self._expiry_lock:
            if self.production_done or self.failure or self._ticket.cancellation_reason is not None or self._done:
                return
            if self.deadline_probe is not None:
                try:
                    if self.deadline_probe():
                        self.produced()
                        return
                except BaseException as error:
                    self.failed(str(error))
                    return
            super()._expire()

    def check(self) -> None:
        if self.failure:
            raise RuntimeError(self.failure)
        super().check()
        if self.failure:
            raise RuntimeError(self.failure)

    def failed(self, message: str) -> None:
        with self._lock:
            if self._done or self.failure or self._ticket.cancellation_reason is not None:
                return
            self.failure = message
            self._state = "FAILED"
            if self._deadline is not None:
                self._deadline.close()
                self._deadline = None
        self._cancellation.cancel(message)
        result = self._result
        if result is not None:
            result.request_cancelled(lambda: RuntimeError(message))


class ResultServiceClient:
    """One service process with many independently owned query contexts.

    Failed cleanup retains its context and capacity. Process failure invalidates
    all resident results; this owner never silently replaces the service.
    """

    def __init__(self, resources: RayResources) -> None:
        self.resources = resources
        self.lock = threading.Lock()
        self.actor: Any = None
        self.contexts: dict[str, Any] = {}
        self.sequences: dict[str, int] = {}
        self.next_sequence = 0
        self.closing = False
        self.closed = False

    def create(self, query_id: str, resources: RayResources) -> tuple[Any, str]:
        import ray

        from vane._native import execution_runtime as native
        from vane.execution.pipelined_worker import ResultService

        with self.lock:
            if self.closing:
                raise RuntimeError("result service is closing")
            if not ray.is_initialized():
                raise RuntimeError("Runtime queries require ray.init()")
            if query_id in self.contexts:
                raise RuntimeError("duplicate result context")
            if len(self.contexts) >= self.resources.max_results:
                raise RuntimeError("result service capacity is full")
            if self.actor is None:
                limits = self.resources.exchange
                memory = 2 * limits.window_bytes + 2 * native.DirectFlight.staging_per_link(limits.frame_bytes)
                if self.resources.exchange_stores:
                    memory += native.MaterializedIO.staging_bytes(limits.frame_bytes)
                self.actor = (
                    ray.remote(
                        max_restarts=0,
                        max_task_retries=0,
                        num_cpus=0,
                        max_concurrency=2 * self.resources.max_results + 4,
                    )(ResultService)
                    .options(memory=memory * self.resources.max_results)
                    .remote(self.resources.max_results)
                )
            # Own cleanup before dispatch, including an ambiguous submit error.
            # A fresh release fences this call even if its receipt never succeeds.
            self.next_sequence += 1
            self.sequences[query_id] = self.next_sequence
            self.contexts[query_id] = None
            self.contexts[query_id] = self.actor.create.remote(query_id, resources, self.next_sequence)
            return self.actor, query_id

    def prepared(self, query_id: str, context: QueryContext) -> None:
        with self.lock:
            reference = self.contexts[query_id]
        _get(reference, context)

    def release(self, actor: Any, query_id: str) -> None:
        import ray

        with self.lock:
            if query_id not in self.contexts or actor is not self.actor:
                return
            sequence = self.sequences[query_id]
        if ray.is_initialized():
            try:
                timeout = cleanup_timeout(10)
                _get(actor.release.remote(query_id, sequence), timeout=timeout)
            except ray.exceptions.ActorDiedError:
                pass  # Only confirmed death makes remote cleanup unnecessary.
        with self.lock:
            self.contexts.pop(query_id, None)
            self.sequences.pop(query_id, None)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "active_contexts": len(self.contexts),
                "capacity": self.resources.max_results,
                "started": self.actor is not None,
            }

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closing = True
            if self.contexts:
                raise RuntimeError("result context cleanup is pending")
            if self.actor is not None:
                import ray

                if ray.is_initialized():
                    cleanup_timeout(10)
                    ray.kill(self.actor, no_restart=True)
            # Reject new contexts once shutdown starts, but keep termination
            # retryable if ray.kill raises. The lock serializes close attempts.
            self.closed = True


class WorkerPool:
    def __init__(self, resources: RayResources) -> None:
        self.resources = resources
        self.admission = WorkerResourceManager(worker_capacity(resources), resources.worker_count)
        self.results = ResultServiceClient(resources)
        self.lock = threading.Lock()
        self.workers: list[Any] = []
        self.epochs: list[str] = []
        self.call_sequences: list[int] = []
        self.closed = False

    def ensure(self, context: QueryContext, engine: str) -> None:
        import ray

        from vane.execution.pipelined_worker import PipelinedWorker

        if not ray.is_initialized():
            raise RuntimeError("backend='ray' requires ray.init() before query submission")
        with self.lock:
            if self.closed:
                raise RuntimeError("worker pool is closed")
            if self.workers:
                return
            resources = self.resources
            actor = ray.remote(max_restarts=0, max_task_retries=0, max_concurrency=16)(PipelinedWorker)
            try:
                for _ in range(resources.worker_count):
                    self.workers.append(
                        actor.options(
                            num_cpus=resources.cpus_per_worker,
                            memory=resources.operator_memory_bytes
                            + resources.exchange_buffer_bytes
                            + resources.staging_buffer_bytes,
                        ).remote(resources)
                    )
                    self.call_sequences.append(0)
                descriptions = [worker.describe.remote() for worker in self.workers]
                for reference in descriptions:
                    value = _get(reference, context)
                    if value["engine"] != engine:
                        raise RuntimeError("Ray worker native engine identity does not match planner")
                    self.epochs.append(value["epoch"])
            except BaseException:
                for worker in self.workers:
                    ray.kill(worker, no_restart=True)
                self.workers.clear()
                self.epochs.clear()
                self.call_sequences.clear()
                raise

    def next_call(self, index: int, worker: Any, epoch: str) -> int:
        with self.lock:
            if self.closed or self.workers[index] is not worker or self.epochs[index] != epoch:
                raise RuntimeError("worker epoch changed before preparation")
            self.call_sequences[index] += 1
            return self.call_sequences[index]

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self.admission.close()
            if self.workers:
                import ray

                if ray.is_initialized():
                    for worker in self.workers:
                        cleanup_timeout(10)
                        ray.kill(worker, no_restart=True)
            self.workers.clear()
            self.epochs.clear()
            self.call_sequences.clear()
        self.results.close()

    def replace(self, index: int, epoch: str, context: QueryContext, engine: str) -> None:
        import ray

        from vane.execution.pipelined_worker import PipelinedWorker

        with self.lock:
            if self.closed:
                raise RuntimeError("worker pool is closed")
            if self.epochs[index] != epoch:
                return
            ray.kill(self.workers[index], no_restart=True)
            resources = self.resources
            actor = ray.remote(max_restarts=0, max_task_retries=0, max_concurrency=16)(PipelinedWorker)
            replacement = actor.options(
                num_cpus=resources.cpus_per_worker,
                memory=resources.operator_memory_bytes
                + resources.exchange_buffer_bytes
                + resources.staging_buffer_bytes,
            ).remote(resources)
            try:
                value = _get(replacement.describe.remote(), context)
                if value["engine"] != engine or value["epoch"] == epoch:
                    raise RuntimeError("replacement worker has an incompatible native engine or epoch")
            except BaseException:
                ray.kill(replacement, no_restart=True)
                raise
            self.workers[index] = replacement
            self.epochs[index] = value["epoch"]
            self.call_sequences[index] = 0


class PipelinedScheduler:
    def __init__(
        self,
        pool: WorkerPool,
        context: PipelinedContext,
        spec: Any,
        rows_per_batch: int = 2048,
        consumer: NativeResultConsumer | None = None,
    ) -> None:
        self.pool = pool
        self.context = context
        self.spec = spec
        self.resources = replace(
            pool.resources,
            exchange=replace(
                pool.resources.exchange, frame_rows=min(rows_per_batch, pool.resources.exchange.frame_rows)
            ),
        )
        self.relay: Any = None
        self.result_id = ""
        self.consumer = NativeResultConsumer() if consumer is None else consumer
        self.monitor: threading.Thread | None = None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.cleanup_lock = threading.Lock()
        self.prepared: set[int] = set()
        self.prepare_calls: list[Any] = []
        self.worker_calls: dict[int, tuple[Any, str, int]] = {}
        self.closed = False
        self.failure = ""
        self.schema: Any = None
        self.result_endpoint: dict[str, str] = {}
        self.reservation = f"pipelined/{spec.query_id}"

    def prepare(self) -> None:
        from vane._native import execution_runtime as native

        self.pool.ensure(self.context, self.spec.graph.engine_identity)
        self.context.check()
        resources = self.resources
        self.relay, self.result_id = self.pool.results.create(self.context.query_id, resources)
        self.pool.results.prepared(self.result_id, self.context)
        assignments, routes = placement(self.spec, self.pool.epochs, self.result_id)
        demands = {
            index: pipelined_demand(resources, index, [t for t, host in assignments.items() if host == index], routes)
            for index in sorted(set(assignments.values()))
        }
        self.pool.admission.acquire(
            self.reservation,
            self.spec.query_id,
            demands,
            self.context.check,
            max(0.0, self.context._ticket.admission_deadline - time.monotonic()),
        )
        self.context.start_execution()
        client_epoch = uuid.uuid4().hex
        ticket = DirectTicket(
            self.spec.query_id,
            "0",
            self.result_id,
            client_epoch,
            "client-result",
            "result-service",
            "client",
            0,
            hashlib.sha256(self.spec.result_schema).hexdigest(),
            secrets.token_urlsafe(32),
        ).encode()
        location = _get(self.relay.prepare.remote(self.result_id, self.spec.result_schema, ticket), self.context)
        self.result_endpoint = {"location": location, "ticket": ticket}
        for index, worker in enumerate(self.pool.workers):
            owned = [task for task, host in assignments.items() if host == index]
            if not owned:
                continue
            self.context.check()
            with self.lock:
                if self.stop.is_set():
                    raise RuntimeError("query canceled before worker preparation")
                epoch = self.pool.epochs[index]
                sequence = self.pool.next_call(index, worker, epoch)
                self.worker_calls[index] = (worker, epoch, sequence)
                self.prepared.add(index)  # Rollback includes an in-flight prepare RPC.
                self.prepare_calls.append(
                    worker.prepare.remote(
                        epoch,
                        self.spec.to_dict(),
                        index,
                        owned,
                        routes,
                        self.resources.exchange.frame_rows,
                        sequence,
                    )
                )
        locations = {
            index: _get(reference, self.context) for index, reference in zip(sorted(self.prepared), self.prepare_calls)
        }
        for index in self.prepared:
            _get(
                self.pool.workers[index].connect.remote(self.pool.epochs[index], self.spec.query_id, locations),
                self.context,
            )
        root_route = next(route for route in routes if route["target_worker"] == -1)
        timeout = self.spec.options.execution_timeout + self.spec.options.delivery_timeout
        _get(
            self.relay.connect.remote(
                self.result_id, locations[root_route["source_worker"]], root_route["ticket"], timeout
            ),
            self.context,
        )
        self.consumer.attach(self.spec, resources, location, ticket)
        self.schema = native.arrow_schema(self.spec.result_schema, list(self.spec.result_names))
        while True:
            self.context.check()
            ready = [
                _get(self.pool.workers[index].ready.remote(self.pool.epochs[index], self.spec.query_id), self.context)
                for index in sorted(self.prepared)
            ]
            relay_status = _get(self.relay.status.remote(self.result_id), self.context)
            error = relay_status["error"] or relay_status["channel"]["error"] or self.consumer.error
            if error:
                raise RuntimeError(error)
            if all(ready) and relay_status["ready"] and self.consumer.ready:
                break
            self.stop.wait(0.01)
        self.context.deadline_probe = self.production_status
        # Every context, input and endpoint now exists. Start consumers before
        # their producers, retaining the same fixed attempt and split assignment.
        fragments = {f.fragment_id: f for f in self.spec.graph.fragments}
        for identity in reversed(self.spec.graph.topological_fragment_ids()):
            by_worker: dict[int, list[str]] = {}
            for part in range(fragments[identity].partition_count):
                task = task_id(identity, part)
                by_worker.setdefault(assignments[task], []).append(task)
            calls = [
                self.pool.workers[index].start.remote(self.pool.epochs[index], self.spec.query_id, tasks)
                for index, tasks in by_worker.items()
            ]
            for reference in calls:
                _get(reference, self.context)
        self.monitor = threading.Thread(target=self._monitor, name="vane-pipelined-query", daemon=True)
        self.monitor.start()

    def production_status(self, *, timeout: float = 2) -> bool:
        calls = [
            self.pool.workers[index].production.remote(self.pool.epochs[index], self.spec.query_id)
            for index in sorted(self.prepared)
        ]
        values = [_get(reference, timeout=timeout) for reference in calls]
        error = next((value["error"] for value in values if value["error"]), "")
        relay = _get(self.relay.status.remote(self.result_id), timeout=timeout)
        if relay["query_id"] != self.result_id:
            raise RuntimeError("result context identity changed")
        error = error or relay["error"] or relay["channel"]["error"] or self.consumer.error
        if error:
            raise RuntimeError(error)
        return bool(values) and all(value["finished"] for value in values)

    def _monitor(self) -> None:
        try:
            while not self.stop.is_set():
                # Detailed task status waits for the execution lock held by
                # pump(). Monitor the independent native production/error
                # probe so a long ExecuteTask is not mistaken for worker loss.
                if self.production_status(timeout=5):
                    self.context.produced()
                self.stop.wait(0.02)
        except BaseException as error:
            if not self.stop.is_set():
                self.context.failed(str(error))
                self.cancel(str(error))

    def delivery_complete(self) -> bool:
        return self.consumer.delivered() and self.production_status()

    def read_next_batch(self) -> Any:
        while True:
            self.context.check()
            state, batch = self.consumer.channel.poll("client")
            if state == "data":
                try:
                    return batch.to_arrow(list(self.spec.result_names))
                finally:
                    batch.close()
            if state == "end" and self.production_status():
                # Success requires the control-plane production outcome as well
                # as FINISH. A transport EOF alone never commits query success.
                self.context.produced()
                self.context.check()
                raise StopIteration
            if state == "closed":
                self.context.check()
                raise RuntimeError("result channel closed before query completion")
            self.stop.wait(0.002)

    def cancel(self, reason: str = "query canceled") -> None:
        import ray

        with self.lock:
            if self.closed:
                return
            self.failure = self.failure or reason
            self.stop.set()
            self.pool.admission.cancel_waiting(self.spec.query_id)
            if self.consumer.flight is not None:
                self.consumer.cancel(reason)
            if not ray.is_initialized():
                return
            for index in self.prepared:
                self.pool.workers[index].cancel.remote(self.pool.epochs[index], self.spec.query_id, reason)
            if self.relay is not None:
                self.relay.cancel.remote(self.result_id, reason)

    def diagnostics(self) -> dict[str, Any]:
        calls: dict[int, Any] = {}
        workers: dict[int, Any] = {}
        with self.lock:
            if not self.closed:
                for index in sorted(self.prepared):
                    try:
                        calls[index] = self.pool.workers[index].status.remote(
                            self.pool.epochs[index], self.spec.query_id
                        )
                    except Exception as error:
                        workers[index] = {"unavailable": str(error)}
        deadline = time.monotonic() + 2
        for index, reference in calls.items():
            try:
                workers[index] = _get(reference, timeout=max(0.001, deadline - time.monotonic()))
            except Exception as error:
                # Diagnostics are observational; a delayed or lost probe must
                # never change a query's execution outcome.
                workers[index] = {"unavailable": str(error)}
        return {
            "mode": "pipelined",
            "workers": workers,
            "result_channel": self.consumer.channel.snapshot() if self.consumer.channel is not None else None,
            "resources": self.pool.admission.snapshot(),
            "cleanup_complete": self.closed,
        }

    def close(self) -> None:
        import ray

        with self.cleanup_lock:
            if self.closed:
                return
            self.context.deadline_probe = None
            self.cancel("query result released")
            if self.monitor is not None and self.monitor is not threading.current_thread():
                self.monitor.join(timeout=cleanup_timeout(10))
                if self.monitor.is_alive():
                    raise RuntimeError("Ray query monitor cleanup is pending")
            if self.consumer.flight is not None:
                self.consumer.close()
            if not ray.is_initialized():
                self.pool.results.release(self.pool.results.actor, self.context.query_id)
                self.pool.admission.release(self.reservation)
                self.closed = True
                return
            # Creation receipts may be permanently failed or still pending. A
            # fresh release fences late preparation and confirms native cleanup.
            errors = []
            for worker, epoch, sequence in self.worker_calls.values():
                try:
                    timeout = cleanup_timeout(10)
                    _get(
                        worker.release.remote(epoch, self.spec.query_id, sequence),
                        timeout=timeout,
                    )
                except ray.exceptions.ActorDiedError:
                    pass  # Dead epoch owns no usable native reservation.
                except BaseException as error:
                    errors.append(error)
            try:
                self.pool.results.release(self.pool.results.actor, self.context.query_id)
            except BaseException as error:
                errors.append(error)
            if errors:
                raise RuntimeError("worker release failed; retry result.close()") from errors[0]
            self.pool.admission.release(self.reservation)
            self.closed = True


class RayQueryRuntime(QueryRuntime):
    backend = "ray"
    context_type: type[QueryContext] = PipelinedContext

    def __init__(self, service: Any, execution: str, resources: Any) -> None:
        self.service = service
        self.session_id = uuid.uuid4().hex
        self._connection: Any = None
        try:
            self.execution = DistributedMode(execution)
        except (ValueError, TypeError) as error:
            raise ValueError("Ray execution must be 'pipelined' or 'fte'") from error
        if self.execution is DistributedMode.FTE and not service.resources.exchange_stores:
            raise ValueError("Ray FTE requires a registered ExchangeStore")
        super().__init__(resources)
        from vane.execution.request_admission import RequestAdmissionLimits
        from vane.execution.result_delivery import ResultDeliveryLimits, RuntimeResultDelivery

        self._admission = service.admission.scope(
            RequestAdmissionLimits(resources.max_active_queries, resources.max_queued_queries)
        )
        self._delivery = RuntimeResultDelivery(
            ResultDeliveryLimits(resources.max_results, resources.result_buffer_bytes), parent=service.delivery
        )
        self.ray_resources = service.resources
        self.pool = service.pool
        self.stores = service.stores

    def exchange_store(self, name: str) -> Any:
        return self.service.exchange_store(name)

    def _attach_connection(self, connection: Any) -> None:
        with self.service.lock:
            if self.service.closing or self.session_id not in self.service.sessions:
                raise RuntimeError("Runtime closed while opening a session")
            if self._connection is not None:
                raise RuntimeError("session already has a native connection")
            self._connection = connection

    def _connection_closed(self) -> None:
        # The native session calls this only after its last connection has
        # released the database and query cleanup has succeeded.
        with self.service.lock:
            self._connection = None
            self.service.retire(self)

    def close_session(self, *, timeout: float) -> None:
        with cleanup_deadline(time.monotonic() + timeout):
            self.close(timeout=cleanup_timeout(timeout))
            with self.service.lock:
                connection = self._connection
            if connection is not None:
                cleanup_timeout(timeout)
                connection.close()

    def resource_snapshot(self) -> dict[str, Any]:
        return {
            **super().resource_snapshot(),
            "workers": self.pool.admission.snapshot(),
            "result_service": self.pool.results.snapshot(),
            "session_id": self.session_id,
            "service_id": self.service.service_id,
        }

    def submit(
        self,
        connection: Any,
        sql: str,
        parameters: Any,
        options: QueryExecutionOptions | None,
        rows_per_batch: int,
        overrides: dict[str, Any],
        publish: Any,
        retire: Any,
        *,
        consumer: NativeResultConsumer | None = None,
    ) -> Any:
        if self.service.closing:
            raise RuntimeError("query service is draining")
        if parameters is not None:
            raise NotImplementedError("Ray query parameters are not supported by the fragment compiler")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("Ray query() requires SQL text")
        if set(overrides) - {"execution"}:
            raise ValueError("unknown Ray query options")
        if type(rows_per_batch) is not int or not 0 < rows_per_batch <= 2048:
            raise ValueError("Ray rows_per_batch must be between 1 and 2048")
        if options is not None and (
            not isinstance(options, QueryExecutionOptions) or not isinstance(options.target, RayExecution)
        ):
            raise ValueError("Ray queries require RayExecution options")
        default_mode = self.execution
        if options is not None:
            assert isinstance(options.target, RayExecution)
            default_mode = options.target.mode
        mode = DistributedMode(overrides.get("execution", default_mode))
        if options is None:
            policy = None
            if mode is DistributedMode.FTE:
                if len(self.ray_resources.exchange_stores) != 1:
                    raise ValueError("FTE requires explicit options naming a registered ExchangeStore")
                policy = FteOptions(self.ray_resources.exchange_stores[0].name, 3, 0.1)
            options = QueryExecutionOptions(RayExecution(mode, policy), 30, 300, 300)
        if not isinstance(options.target, RayExecution) or options.target.mode is not mode:
            raise ValueError("execution override disagrees with the immutable query options")

        def execute(context: QueryContext) -> None:
            assert isinstance(context, PipelinedContext)
            resources = self.ray_resources
            from vane.execution.recovery_runtime import RecoveryScheduler

            scheduler: PipelinedScheduler | RecoveryScheduler | None = None

            def close(retired: bool) -> None:
                if retired:
                    retire()
                elif scheduler is not None:
                    scheduler.close()

            from vane._native.execution_runtime import check_entry

            context.install_cleanup(close, check_entry)
            demand = ResourceDemand(
                resources.worker_count * resources.cpus_per_worker / resources.max_active_queries,
                resources.worker_count * resources.task_contexts_per_worker,
                MemoryDemand(
                    resources.worker_count * (resources.operator_memory_bytes // resources.max_active_queries),
                    resources.result_buffer_bytes,
                    resources.worker_count * resources.exchange_buffer_bytes,
                    resources.worker_count * resources.staging_buffer_bytes,
                ),
                resources.worker_count * resources.io_concurrency,
            )
            if mode is DistributedMode.FTE:
                assert isinstance(options.target, RayExecution) and options.target.fte_options is not None
                store = self.exchange_store(options.target.fte_options.exchange_store)
                scheduler = RecoveryScheduler(self.pool, store, context, rows_per_batch, consumer)
                context.started(scheduler.cancel)
                scheduler.prepare(connection, sql, options, demand, FragmentCompileOptions(resources.partitions))
                spec = scheduler.spec
            else:
                spec = prepare_ray_query(
                    connection,
                    sql,
                    query_id=context.query_id,
                    options=options,
                    resources=demand,
                    compile_options=FragmentCompileOptions(resources.partitions),
                )
                scheduler = PipelinedScheduler(self.pool, context, spec, rows_per_batch, consumer)
                context.started(scheduler.cancel)
                scheduler.prepare()
            context.install_reader(
                scheduler,
                scheduler.read_next_batch,
                {"names": list(spec.result_names), "types": [str(t) for t in scheduler.schema.types]},
            )

        return self.run(execute, publish, options)

    def close(self, *, timeout: float = 5.0) -> None:
        with cleanup_deadline(time.monotonic() + timeout):
            super().close(timeout=cleanup_timeout(timeout))
        with self.service.lock:
            if self._connection is None:
                self.service.retire(self)
