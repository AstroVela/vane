# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Query admission and lifetime, independent of model requests and schedulers."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vane.execution.batch_lease import BatchLease
from vane.execution.cleanup_deadline import cleanup_timeout
from vane.execution.native_cancellation import NativeQueryCancellation
from vane.execution.query_options import LocalExecution, QueryExecutionOptions
from vane.execution.request_admission import (
    RequestAdmissionLimits,
    RequestAdmissionScope,
    RequestCancellationReason,
    RequestCancelled,
    RequestTicket,
    RuntimeRequestAdmission,
)
from vane.execution.request_deadline import MonotonicDeadline
from vane.execution.result_delivery import QueryResult, ResultDeliveryFull, ResultDeliveryLimits, RuntimeResultDelivery
from vane.execution.udf_admission import AdmissionLease
from vane.execution.udf_lifecycle import ExecutionCancellationScope


@dataclass(frozen=True)
class QueryResources:
    """Session capacities; result bytes exclude native operators and collect()."""

    max_active_queries: int = 4
    max_queued_queries: int = 64
    max_results: int = 4
    result_buffer_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        RequestAdmissionLimits(self.max_active_queries, self.max_queued_queries)
        ResultDeliveryLimits(self.max_results, self.result_buffer_bytes)


class QueryContext:
    """One execution owner, also the pull source for its QueryResult.

    Query completion and delivery completion are separate. Admission stays
    owned until native cleanup and result retirement have both succeeded.
    """

    track_graph = False

    def __init__(self, runtime: QueryRuntime, ticket: RequestTicket, options: QueryExecutionOptions) -> None:
        self.query_id = uuid.uuid4().hex
        self.options = options
        self._runtime = runtime
        self._ticket = ticket
        self._lock = threading.RLock()
        self._state = "ADMISSION_WAIT"
        self._done = False
        self._eof = False
        self._had_rows = False
        self._native_closed = False
        self._cursor_retired = False
        self._lease: AdmissionLease | None = None
        self._deadline: MonotonicDeadline | None = None
        self._admission_deadline: MonotonicDeadline | None = None
        self._cancellation = ExecutionCancellationScope(self.query_id, 1)
        self._binding = NativeQueryCancellation(self._cancellation)
        self._result: QueryResult | None = None
        self._reader: Any = None
        self._read_native: Callable[[], Any] | None = None
        self._close_native: Callable[[bool], None] | None = None
        self._guard_native: Callable[[], None] | None = None
        self._execution_diagnostics: dict[str, Any] = {"mode": runtime.backend}

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            reader = self._reader
            value: dict[str, Any] = {
                "query_id": self.query_id,
                "execution_state": self._state,
                "cleanup": {
                    "native_closed": self._native_closed,
                    "cursor_retired": self._cursor_retired,
                    "complete": self._done,
                },
            }
        probe = getattr(reader, "diagnostics", None)
        value["execution"] = probe() if probe is not None else self._execution_diagnostics
        value["session_resources"] = self._runtime.resource_snapshot()
        return value

    def begin(self, *, defer_execution: bool = False) -> QueryResult:
        def reserve() -> None:
            try:
                result = self._runtime._delivery.begin()
            except ResultDeliveryFull as error:
                error._execution_started = False
                raise
            self._result = result
            result.query_id = self.query_id
            result.context = self
            result.start_stream(self)

            def cancel() -> None:
                self.cancel()

            result._cancellation.register_cancel_wakeup(cancel)

        self._lease = self._ticket.take(before_claim=reserve, defer_execution=defer_execution)
        if defer_execution:
            with self._lock:
                self._admission_deadline = MonotonicDeadline(
                    self._ticket.admission_deadline - self.options.admission_timeout,
                    self.options.admission_timeout,
                    self._expire,
                    timeout_name="admission_timeout",
                )
            self._admission_deadline.start()
        else:
            self.start_execution()
        self.check()
        assert self._result is not None
        return self._result

    def start_execution(self) -> None:
        self.check()
        with self._lock:
            if self._state != "ADMISSION_WAIT":
                return
            started = self._ticket.start_execution()
            if self._admission_deadline is not None:
                self._admission_deadline.close()
                self._admission_deadline = None
            self._state = "RUNNING"
            self._deadline = MonotonicDeadline(started, self.options.execution_timeout, self._expire)
        self._deadline.start()
        self.check()

    def prepare(self, nodes: list[Any], graph: Any) -> dict[str, Any]:
        if nodes:
            raise NotImplementedError("query() model UDF execution is not implemented")
        return {}

    def started(self, interrupt: Callable[[], None]) -> None:
        self._binding.started_callback(interrupt)
        self.check()

    def install_cleanup(self, close_native: Callable[[bool], None], guard_native: Callable[[], None]) -> None:
        self._close_native = close_native
        self._guard_native = guard_native

    def install_reader(self, reader: Any, read_native: Callable[[], Any], schema: dict[str, Any]) -> None:
        self._reader = reader
        self._read_native = read_native
        assert self._result is not None
        self._result.result_schema = schema
        self._result.schema = reader.schema
        self._result.completion_status = "streaming"

    def _cancel(self, reason: RequestCancellationReason) -> bool:
        with self._lock:
            # Failure cleanup can interrupt a still-published connection.
            # That wakeup must not replace an already recorded failure.
            if self._done or self._state == "FAILED":
                return False
            if reason == "admission_timeout" and (
                self._admission_deadline is None or not self._admission_deadline.expired()
            ):
                return False
            accepted = self._ticket.cancel() if reason == "cancelled" else False
            if not accepted:
                accepted = self._ticket.cancel_running(reason=reason)
            if not accepted:
                return False
            self._state = "CANCELED" if reason == "cancelled" else "FAILED"
        self._cancellation.cancel(reason)
        result = self._result
        if result is not None:
            result.request_cancelled(self._ticket.cancellation_error)
        return True

    def cancel(self) -> bool:
        return self._cancel("cancelled")

    def _expire(self) -> None:
        # _cancel arbitrates a copied admission callback against execution
        # start under the state lock, then dispatches wakeups outside that lock.
        if self._admission_deadline is not None:
            self._cancel("admission_timeout")
        if self._deadline is not None and self._deadline.expired():
            self._cancel("execution_timeout")

    def check(self) -> None:
        self._expire()
        if self._ticket.cancellation_reason is not None:
            raise self._ticket.cancellation_error()

    def guard(self) -> None:
        if self._guard_native is not None:
            self._guard_native()

    def commit(self, operation: Callable[[], None]) -> None:
        while True:
            self.check()
            with self._lock:
                if self._ticket.cancellation_reason is not None or (
                    self._deadline is not None and self._deadline.expired()
                ):
                    continue
                operation()
                if self._deadline is None or not self._deadline.expired():
                    return

    def read(self, result: QueryResult) -> bool:
        batch = None
        try:
            self.check()
            assert self._read_native is not None
            batch = self._read_native()
            self.check()
            self._had_rows = self._had_rows or bool(batch.num_rows)
            size = BatchLease.size(batch)
            payload = BatchLease(result.own_buffer(size))
            result.hold(payload)
            result.check_preparation()
            payload.build(batch, size)
            return True
        except StopIteration:
            self.check()
            self._eof = True
            self.close()
            self.check()
            result.completion_status = "ok" if self._had_rows else "empty"
            return False
        except BaseException as error:
            # Arrow can wrap native interruption as OSError. Restore the
            # recorded cancellation/deadline outcome before reporting failure.
            self.check()
            if isinstance(error, ResultDeliveryFull):
                error._execution_started = True
            with self._lock:
                if self._state == "RUNNING":
                    self._state = "FAILED"
            raise
        finally:
            batch = None

    def complete_external(self, result: QueryResult) -> bool:
        self.check()
        reader = self._reader
        if reader is None or not reader.delivery_complete():
            return False
        self.check()

        def mark_eof() -> None:
            self._eof = True
            result.completion_status = "ok"

        result.complete_external(mark_eof)
        return True

    def fail(self, error: BaseException) -> None:
        with self._lock:
            if self._state not in {"CANCELED", "FAILED"}:
                self._state = "CANCELED" if isinstance(error, RequestCancelled) else "FAILED"

    def close(self) -> None:
        self.guard()
        if not self._native_closed:
            if not self._eof and self.state == "RUNNING":
                self.cancel()
            self._binding.close()
            if self._close_native is not None:
                self._close_native(False)
            if self._reader is not None:
                self._reader.close()
                probe = getattr(self._reader, "diagnostics", None)
                if probe is not None:
                    try:
                        self._execution_diagnostics = probe()
                    except Exception as error:
                        self._execution_diagnostics = {"unavailable": str(error)}
            self._reader = None
            self._read_native = None
            self._native_closed = True
        self._expire()
        with self._lock:
            if not self._done:
                if self._state == "RUNNING":
                    self._state = "SUCCEEDED" if self._eof else "CANCELED"
                self._ticket.finish_execution(failed=self._state == "FAILED")
                self._done = True
                if self._deadline is not None:
                    self._deadline.close()
                if self._admission_deadline is not None:
                    self._admission_deadline.close()
                self._cancellation.finish()
        if not self._cursor_retired:
            if self._close_native is not None:
                self._close_native(True)
            self._cursor_retired = True
            self._close_native = None
            self._guard_native = None

    def cleanup_pending(self) -> bool:
        return not self._native_closed or not self._cursor_retired

    def retire(self, release_result: Callable[[], None]) -> None:
        with self._lock:
            if self.cleanup_pending():
                raise RuntimeError("native query cleanup must finish before retirement")
            release_result()
            self._result = None
            if self._lease is not None:
                self._lease.release()
            else:
                self._ticket.cancel()
        self._runtime._retire(self)


class QueryRuntime:
    """Session-owned capacities shared by its independent native cursors."""

    backend = "local"
    context_type: type[QueryContext] = QueryContext

    def __init__(self, resources: QueryResources | None = None) -> None:
        if resources is None:
            resources = QueryResources()
        if not isinstance(resources, QueryResources):
            raise TypeError("resources must be QueryResources")
        self.resources = resources
        self._admission: RuntimeRequestAdmission | RequestAdmissionScope = RuntimeRequestAdmission(
            RequestAdmissionLimits(resources.max_active_queries, resources.max_queued_queries)
        )
        self._delivery = RuntimeResultDelivery(
            ResultDeliveryLimits(resources.max_results, resources.result_buffer_bytes)
        )
        self._lock = threading.Lock()
        self._contexts: dict[str, QueryContext] = {}
        self._draining = False

    @staticmethod
    def options(
        options: QueryExecutionOptions | None, rows_per_batch: int, overrides: dict[str, Any]
    ) -> QueryExecutionOptions:
        if overrides:
            raise ValueError("local query() does not accept execution overrides or unknown options")
        if type(rows_per_batch) is not int or not 0 < rows_per_batch <= 2**31 - 1:
            raise ValueError("rows_per_batch must be a positive 32-bit integer")
        if options is None:
            options = QueryExecutionOptions(LocalExecution(), 30.0, 300.0, 300.0)
        if not isinstance(options, QueryExecutionOptions) or not isinstance(options.target, LocalExecution):
            raise ValueError("local query() requires QueryExecutionOptions with LocalExecution")
        return options

    def run(
        self, operation: Callable[[QueryContext], None], publish: Callable[[Any], None], options: QueryExecutionOptions
    ) -> QueryResult:
        with self._lock:
            if self._draining:
                raise RuntimeError("query runtime is draining")
            ticket = self._admission.request(queue_timeout=options.admission_timeout)
            context = self.context_type(self, ticket, options)
            self._contexts[context.query_id] = context
        try:
            publish(context)
            result = context.begin()
            operation(context)
            context.check()
            result.ready(delivery_timeout=options.delivery_timeout)
            return result
        except BaseException as error:
            context.fail(error)
            if isinstance(error, ResultDeliveryFull):
                error._execution_started = context._lease is not None
            primary = context._ticket.cancellation_error() if context._ticket.cancellation_reason else error
            try:
                if context._result is not None:
                    context._result.abort_preparation()
                else:
                    context.close()
                    context.retire(lambda: None)
            except BaseException as cleanup_error:
                raise primary from cleanup_error
            raise primary
        finally:
            if not context.cleanup_pending():
                publish(None)

    def _retire(self, context: QueryContext) -> None:
        with self._lock:
            self._contexts.pop(context.query_id, None)

    def resource_snapshot(self) -> dict[str, Any]:
        with self._lock:
            contexts = tuple(self._contexts.values())
        return {
            "queries": {context.query_id: context.state for context in contexts},
            "request_admission": self._admission.snapshot(),
            "result_delivery": self._delivery.snapshot(),
        }

    def drain(self) -> None:
        with self._lock:
            self._draining = True
            contexts = tuple(self._contexts.values())
        self._admission.drain()
        for context in contexts:
            context.cancel()

    def close(self, *, timeout: float = 5.0) -> None:
        self.drain()
        self._delivery.close(timeout=cleanup_timeout(timeout))
        self._admission.close(timeout=cleanup_timeout(timeout))
