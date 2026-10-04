# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Whole-graph Ray pipelined scheduling and native Flight result delivery."""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from vane.execution.compiler import FragmentCompileOptions
from vane.execution.pipelined_plan import DirectTicket, RayResources, placement, task_id
from vane.execution.query_options import DistributedMode, QueryExecutionOptions, RayExecution
from vane.execution.query_runtime import QueryContext, QueryRuntime
from vane.execution.resource_demand import MemoryDemand, ResourceDemand
from vane.execution.submission import prepare_ray_query


def _get(reference: Any, context: QueryContext | None = None, timeout: float = 30) -> Any:
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray session is no longer connected")
    deadline = time.monotonic() + timeout
    while True:
        if context is not None:
            context.check()
        ready, _ = ray.wait([reference], timeout=0.05)
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


class WorkerPool:
    def __init__(self, resources: RayResources) -> None:
        self.resources = resources
        self.lock = threading.Lock()
        self.workers: list[Any] = []
        self.epochs: list[str] = []
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
                raise

    def close(self) -> None:
        import ray

        with self.lock:
            self.closed = True
            if ray.is_initialized():
                for worker in self.workers:
                    ray.kill(worker, no_restart=True)
            self.workers.clear()
            self.epochs.clear()


class PipelinedScheduler:
    def __init__(self, pool: WorkerPool, context: PipelinedContext, spec: Any, rows_per_batch: int = 2048) -> None:
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
        self.relay_epoch = ""
        self.client: Any = None
        self.channel: Any = None
        self.monitor: threading.Thread | None = None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.cleanup_lock = threading.Lock()
        self.prepared: set[int] = set()
        self.prepare_calls: list[Any] = []
        self.closed = False
        self.failure = ""
        self.schema: Any = None
        self.result_endpoint: dict[str, str] = {}

    def prepare(self) -> None:
        import ray

        from vane._native import execution_runtime as native
        from vane.execution.pipelined_worker import ResultService, _channel

        self.pool.ensure(self.context, self.spec.graph.engine_identity)
        self.context.check()
        resources = self.resources
        relay_type = ray.remote(max_restarts=0, max_task_retries=0, num_cpus=0, max_concurrency=4)(ResultService)
        self.relay = relay_type.options(
            memory=2 * resources.exchange.window_bytes
            + 2 * native.DirectFlight.staging_per_link(resources.exchange.frame_bytes)
        ).remote(resources)
        self.relay_epoch = _get(self.relay.describe.remote(), self.context)
        assignments, routes = placement(self.spec, self.pool.epochs, self.relay_epoch)
        client_epoch = uuid.uuid4().hex
        ticket = DirectTicket(
            self.spec.query_id,
            "0",
            self.relay_epoch,
            client_epoch,
            "client-result",
            "result-service",
            "client",
            0,
            hashlib.sha256(self.spec.result_schema).hexdigest(),
            secrets.token_urlsafe(32),
        ).encode()
        location = _get(self.relay.prepare.remote(self.relay_epoch, self.spec.result_schema, ticket), self.context)
        self.result_endpoint = {"location": location, "ticket": ticket}
        for index, worker in enumerate(self.pool.workers):
            owned = [task for task, host in assignments.items() if host == index]
            if not owned:
                continue
            self.context.check()
            with self.lock:
                if self.stop.is_set():
                    raise RuntimeError("query canceled before worker preparation")
                self.prepared.add(index)  # Rollback includes an in-flight prepare RPC.
                self.prepare_calls.append(
                    worker.prepare.remote(
                        self.pool.epochs[index],
                        self.spec.to_dict(),
                        index,
                        owned,
                        routes,
                        self.resources.exchange.frame_rows,
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
            self.relay.connect.remote(locations[root_route["source_worker"]], root_route["ticket"], timeout),
            self.context,
        )
        self.channel = _channel(self.spec.result_schema, resources, "result-service", "client")
        self.client = native.DirectFlight(
            "127.0.0.1",
            "127.0.0.1",
            1,
            native.DirectFlight.staging_per_link(resources.exchange.frame_bytes),
            resources.exchange.frame_bytes,
        )
        self.client.subscribe(location, ticket, self.channel, "result-service", timeout)
        self.schema = native.arrow_schema(self.spec.result_schema, list(self.spec.result_names))
        ready_deadline = time.monotonic() + self.spec.options.admission_timeout
        while True:
            self.context.check()
            ready = [
                _get(self.pool.workers[index].ready.remote(self.pool.epochs[index], self.spec.query_id), self.context)
                for index in sorted(self.prepared)
            ]
            relay_status = _get(self.relay.status.remote(), self.context)
            error = relay_status["error"] or relay_status["channel"]["error"] or self.client.error
            if error:
                raise RuntimeError(error)
            if all(ready) and relay_status["ready"] and self.client.ready:
                break
            if time.monotonic() >= ready_deadline:
                raise TimeoutError("native Flight consumers did not become ready before admission timeout")
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
        relay = _get(self.relay.status.remote(), timeout=timeout)
        if relay["epoch"] != self.relay_epoch:
            raise RuntimeError("result service epoch changed")
        error = error or relay["error"] or relay["channel"]["error"] or self.client.error
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

    def read_next_batch(self) -> Any:
        while True:
            self.context.check()
            state, batch = self.channel.poll("client")
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
            if self.client is not None:
                self.client.cancel(reason)
            if not ray.is_initialized():
                return
            for index in self.prepared:
                self.pool.workers[index].cancel.remote(self.pool.epochs[index], self.spec.query_id, reason)
            if self.relay is not None:
                self.relay.cancel.remote(reason)

    def close(self) -> None:
        import ray

        with self.cleanup_lock:
            if self.closed:
                return
            self.context.deadline_probe = None
            self.cancel("query result released")
            if self.monitor is not None and self.monitor is not threading.current_thread():
                self.monitor.join(timeout=10)
                if self.monitor.is_alive():
                    raise RuntimeError("Ray query monitor cleanup is pending")
            if self.client is not None:
                self.client.close()
            if not ray.is_initialized():
                self.closed = True
                return
            # Do not release a reservation before its prepare call has settled.
            # Even if the caller stopped waiting, the remote preparation owns it.
            for reference in self.prepare_calls:
                try:
                    _get(reference, timeout=10)
                except ray.exceptions.RayError:
                    pass
            errors = []
            for index in self.prepared:
                try:
                    _get(
                        self.pool.workers[index].release.remote(self.pool.epochs[index], self.spec.query_id), timeout=10
                    )
                except ray.exceptions.RayActorError:
                    pass  # Dead epoch owns no usable native reservation.
                except BaseException as error:
                    errors.append(error)
            if self.relay is not None:
                try:
                    _get(self.relay.release.remote(), timeout=10)
                except ray.exceptions.RayActorError:
                    pass
                finally:
                    ray.kill(self.relay, no_restart=True)
            if errors:
                raise RuntimeError("worker release failed; retry result.close()") from errors[0]
            self.closed = True


class RayQueryRuntime(QueryRuntime):
    backend = "ray"
    context_type: type[QueryContext] = PipelinedContext

    def __init__(self, resources: RayResources | None = None, execution: str = "pipelined") -> None:
        if resources is None:
            resources = RayResources()
        if not isinstance(resources, RayResources):
            raise TypeError("Ray connections require RayResources")
        if execution != "pipelined":
            raise NotImplementedError("Ray FTE execution is scheduled for P3")
        super().__init__(resources)
        self.ray_resources = resources
        self.pool = WorkerPool(resources)

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
    ) -> Any:
        if parameters is not None:
            raise NotImplementedError("Ray query parameters are not supported by the fragment compiler")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("Ray query() requires SQL text")
        if set(overrides) - {"execution"}:
            raise ValueError("unknown Ray query options")
        execution = overrides.get("execution", "pipelined")
        if execution != "pipelined":
            raise NotImplementedError("Ray FTE execution is scheduled for P3")
        if type(rows_per_batch) is not int or not 0 < rows_per_batch <= 2048:
            raise ValueError("Ray rows_per_batch must be between 1 and 2048")
        if options is None:
            options = QueryExecutionOptions(RayExecution(), 30, 300, 300)
        if (
            not isinstance(options, QueryExecutionOptions)
            or not isinstance(options.target, RayExecution)
            or options.target.mode is not DistributedMode.PIPELINED
        ):
            raise ValueError("Ray pipelined queries require pipelined RayExecution options")

        def execute(context: QueryContext) -> None:
            assert isinstance(context, PipelinedContext)
            resources = self.ray_resources
            scheduler: PipelinedScheduler | None = None

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
            spec = prepare_ray_query(
                connection,
                sql,
                query_id=context.query_id,
                options=options,
                resources=demand,
                compile_options=FragmentCompileOptions(resources.partitions),
            )
            scheduler = PipelinedScheduler(self.pool, context, spec, rows_per_batch)
            context.started(scheduler.cancel)
            scheduler.prepare()
            context.install_reader(
                scheduler,
                scheduler.read_next_batch,
                {"names": list(spec.result_names), "types": [str(t) for t in scheduler.schema.types]},
            )

        return self.run(execute, publish, options)

    def close(self, *, timeout: float = 5.0) -> None:
        super().close(timeout=timeout)
        self.pool.close()
