# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded result ownership and delivery, independent of execution backends."""

from __future__ import annotations

import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol

from vane.execution.cleanup_deadline import cleanup_timeout
from vane.execution.data_lifecycle import _OUTPUT_STATES, OutputBlockLeaseOwner
from vane.execution.request_admission import _timeout
from vane.execution.request_deadline import MonotonicDeadline
from vane.execution.udf_actor_pool_lifecycle import rollback_actor_pools
from vane.execution.udf_admission import AdmissionLease
from vane.execution.udf_lifecycle import ExecutionCancellationScope, ExecutionCancelledError

if TYPE_CHECKING:
    import pyarrow as pa

    from vane.execution.query_runtime import QueryContext


@dataclass(frozen=True)
class ResultDeliveryLimits:
    max_results: int
    max_bytes: int

    def __post_init__(self) -> None:
        for name in ("max_results", "max_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class ResultDeliveryFull(RuntimeError):
    """A result slot or buffer would exceed the delivery capacity.

    ``reason`` is ``"slots"`` or ``"bytes"`` for runtime capacity refusals;
    ``requested``, ``used`` and ``limit`` use that reason's unit. Legacy or
    manually constructed errors can leave these fields unknown (``None``).
    Messages and positional exception arguments keep their existing behavior.
    """

    def __init__(
        self,
        *args: object,
        reason: Literal["slots", "bytes"] | None = None,
        requested: int | None = None,
        used: int | None = None,
        limit: int | None = None,
    ) -> None:
        super().__init__(*args)
        self.reason = reason
        self.requested = requested
        self.used = used
        self.limit = limit
        self._execution_started: bool | None = None

    @property
    def execution_started(self) -> bool | None:
        """Whether a managed request entered execution, including preparation.

        Only its request adapter can confirm ``False`` for slot refusal before
        execution admission. ``True`` does not imply user code completed;
        ``None`` provides no execution/retry guarantee. Earlier caller-side
        binding/conversion callbacks are outside this execution boundary.
        """
        return self._execution_started


class ResultDeliveryTimeout(TimeoutError):
    """The ready result exceeded its total delivery deadline."""


class ResultDeliveryCancelled(ExecutionCancelledError):
    """The consumer cancelled delivery of the remaining result."""


class ResultDeliveryClosed(RuntimeError):
    """The result no longer accepts consumers."""


class _ResultCleanupPending(RuntimeError):
    pass


class ResultPayload(Protocol):
    """An adapter owns pending data; exported views own their own references."""

    def export(self, cancellation: ExecutionCancellationScope) -> Any: ...

    def close(self) -> None: ...

    def cleanup_pending(self) -> bool: ...


class ResultStream(Protocol):
    """Pull one payload at a time; retain execution until EOF or cleanup."""

    def read(self, result: QueryResult) -> bool: ...

    def check(self) -> None: ...

    def commit(self, operation: Callable[[], None]) -> None: ...

    def guard(self) -> None: ...

    def close(self) -> None: ...

    def retire(self, release_result: Callable[[], None]) -> None: ...

    def cleanup_pending(self) -> bool: ...


def _close_payload(payload: ResultPayload) -> None:
    payload.close()
    if payload.cleanup_pending():
        raise RuntimeError("result payload cleanup is still in progress")


@dataclass
class _BufferLease:
    lease_id: str
    size_bytes: int
    state: str = "unit_queue"


class RuntimeResultDelivery:
    """Reserve result slots before execution; charge buffers through their views.

    The registry keeps abandoned and failed-cleanup results alive. Buffer lease
    records carry only metadata and survive runtime close while consumers retain
    exported views. Transport callbacks never run under the registry condition.
    """

    def __init__(self, limits: ResultDeliveryLimits, *, parent: RuntimeResultDelivery | None = None) -> None:
        if not isinstance(limits, ResultDeliveryLimits):
            raise TypeError("result_limit must be ResultDeliveryLimits")
        self.limits = limits
        self._condition: threading.Condition = threading.Condition() if parent is None else parent._condition
        self._budgets: tuple[RuntimeResultDelivery, ...] = (self,) if parent is None else (self, *parent._budgets)
        self._results: dict[str, QueryResult] = {}
        self._buffers: dict[str, _BufferLease] = {}
        self._usage_bytes = 0
        self._closed = False
        self._streams_closed = False
        self._completed: Counter[str] = Counter()
        self._rejected = 0
        self._delivery_seconds = 0.0
        self._delivery_samples = 0

    def begin(self) -> QueryResult:
        with self._condition:
            for budget in self._budgets:
                if budget._closed:
                    raise ResultDeliveryClosed("result delivery runtime is closed")
                if len(budget._results) >= budget.limits.max_results:
                    budget._rejected += 1
                    raise ResultDeliveryFull(
                        f"runtime result slots are full: used={len(budget._results)}, limit={budget.limits.max_results}",
                        reason="slots",
                        requested=1,
                        used=len(budget._results),
                        limit=budget.limits.max_results,
                    )
            result = QueryResult(self, uuid.uuid4().hex)
            for budget in self._budgets:
                budget._results[result.result_id] = result
            return result

    def _release_result(self, result: QueryResult) -> None:
        with self._condition:
            for budget in self._budgets:
                if budget._results.pop(result.result_id, None) is not None:
                    assert result._outcome is not None
                    result._released_at = time.monotonic()
                    if result._ready_at is not None:
                        budget._delivery_seconds += max(0.0, result._released_at - result._ready_at)
                        budget._delivery_samples += 1
                    budget._completed[result._outcome] += 1
            self._condition.notify_all()

    def transition_output_block(self, lease_id: str, state: str) -> bool:
        with self._condition:
            lease = self._buffers.get(lease_id)
            if lease is None:
                return False
            if state not in _OUTPUT_STATES[:-1] or _OUTPUT_STATES.index(state) != _OUTPUT_STATES.index(lease.state) + 1:
                raise ValueError("result buffer leases must advance one state at a time")
            lease.state = state
            return True

    def release_output_block(self, lease_id: str) -> bool:
        with self._condition:
            lease = self._buffers.get(lease_id)
            if lease is None:
                return False
            for budget in self._budgets:
                budget._buffers.pop(lease_id)
                budget._usage_bytes -= lease.size_bytes
            self._condition.notify_all()
            return True

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "max_results": self.limits.max_results,
                "limit_bytes": self.limits.max_bytes,
                "active_results": len(self._results),
                "preparing_results": sum(r._preparing for r in self._results.values()),
                "ready_results": sum(not r._preparing and r._outcome is None for r in self._results.values()),
                "streaming_results": sum(r._stream is not None for r in self._results.values()),
                "waiting_byte_results": sum(bool(r._waiting_bytes) for r in self._results.values()),
                "waiting_bytes": sum(r._waiting_bytes for r in self._results.values()),
                "cleanup_pending_results": sum(r._outcome is not None for r in self._results.values()),
                "usage_bytes": self._usage_bytes,
                "buffers": len(self._buffers),
                "exported_bytes": sum(b.size_bytes for b in self._buffers.values() if b.state == "external_consumer"),
                "delivered_results": self._completed["delivered"],
                "closed_results": self._completed["closed"],
                "cancelled_results": self._completed["cancelled"],
                "timed_out_results": self._completed["delivery_timed_out"],
                "failed_results": self._completed["failed"],
                "rejected_results": self._rejected,
                "delivery_seconds": self._delivery_seconds,
                "delivery_samples": self._delivery_samples,
                "closed": self._closed,
            }

    def close(self, *, timeout: float = 0.0) -> None:
        deadline = time.monotonic() + _timeout(timeout, "result close timeout")
        with self._condition:
            self._closed = True
            results = tuple(self._results.values())
            # Fence every consumer before dispatching any adapter cleanup.
            dispatch = {result for result in results if result._finish_locked("closed")}
        errors = []
        for result in results:
            try:
                if result in dispatch:
                    result._dispatch_cancellation()
                result._close_stream(timeout=max(0.0, deadline - time.monotonic()))
            except BaseException as error:
                errors.append(error)
        if errors:
            raise RuntimeError(
                "result cleanup failed or is in progress; retry result.close() or runtime.close()"
            ) from errors[0]

    def cancel_streams(self) -> None:
        """Fence late streams and wake live execution before waiting for requests."""
        with self._condition:
            # Claimed requests may still be executing or preparing their
            # readers. Their eventual ownership transfer must observe close,
            # even if this call's request-admission wait later times out.
            self._streams_closed = True
            streams = tuple(r for r in self._results.values() if r._stream is not None)
        errors = []
        for result in streams:
            try:
                result.close()
            except _ResultCleanupPending:
                # A concurrent consumer owns cleanup until its read returns.
                pass
            except BaseException as error:
                errors.append(error)
        if errors:
            raise RuntimeError("stream cleanup failed; retry result.close() or runtime.close()") from errors[0]


class QueryResult:
    """A single-consumer iterator whose remaining output has a separate lifetime.

    ``take()`` transfers one payload to the caller. Its underlying buffers remain
    charged until the last exported view is gone, even after this handle closes.
    """

    def __init__(self, runtime: RuntimeResultDelivery, result_id: str) -> None:
        self._runtime = runtime
        self.result_id = result_id
        self.query_id: str | None = None
        self.context: QueryContext | None = None
        self.schema: pa.Schema | None = None
        self._payloads: deque[ResultPayload] = deque()
        self._stream: ResultStream | None = None
        self._stream_error: Callable[[], BaseException] | None = None
        self._waiting_bytes = 0
        self._preparing = True
        self._taking = False
        self._cleaning = False
        self._outcome: str | None = None
        self._ready_at: float | None = None
        self._released_at: float | None = None
        self._deadline: MonotonicDeadline | None = None
        self._cancellation = ExecutionCancellationScope(result_id, 1)
        self._cancel_finished = threading.Event()
        self._cancel_finished.set()
        self._lease = AdmissionLease(result_id, 0, {}, _release_callback=lambda: runtime._release_result(self))
        self.result_schema: Any = None
        self.completion_status: Any = None
        self.stats: Any = None
        self.task_stats: Any = None

    @property
    def state(self) -> str:
        with self._runtime._condition:
            if self._outcome is not None:
                return "closing" if self.result_id in self._runtime._results else self._outcome
            return "preparing" if self._preparing else "ready"

    def timing_snapshot(self) -> dict[str, float | None]:
        """Ready-to-retirement time, including cleanup, excluding exported views."""
        with self._runtime._condition:
            return {
                "delivery_seconds": (
                    max(0.0, self._released_at - self._ready_at)
                    if self._released_at is not None and self._ready_at is not None
                    else None
                )
            }

    def diagnostics(self) -> dict[str, Any]:
        """Observe query, task, channel and cleanup ownership without advancing execution."""
        return {
            **(self.context.diagnostics() if self.context is not None else {}),
            "result_id": self.result_id,
            "delivery_state": self.state,
            "timing": self.timing_snapshot(),
        }

    def _finish_locked(self, outcome: str) -> bool:
        if self._outcome is not None:
            return False
        self._outcome = outcome
        if outcome in {"closed", "cancelled", "delivery_timed_out"}:
            self._cancel_finished.clear()
        if self._deadline is not None:
            self._deadline.close()
        return True

    def _check_locked(self) -> None:
        if self._outcome == "failed" and self._stream_error is not None:
            raise self._stream_error()
        if self._outcome == "delivery_timed_out":
            raise ResultDeliveryTimeout("result delivery deadline exceeded")
        if self._outcome == "cancelled":
            raise ResultDeliveryCancelled("result delivery cancelled")
        if self._outcome == "delivered":
            raise StopIteration
        if self._outcome is not None:
            raise ResultDeliveryClosed(f"result delivery is {self._outcome}")

    def own_buffer(self, size_bytes: int) -> OutputBlockLeaseOwner:
        """Reserve exact adapter-buffer capacity before allocating it."""
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError("result buffer size must be a non-negative integer")
        runtime = self._runtime
        with runtime._condition:
            self._check_locked()
            if not self._preparing and self._stream is None:
                raise RuntimeError("result preparation has finished")

            def full() -> RuntimeResultDelivery | None:
                return next((b for b in runtime._budgets if b._usage_bytes + size_bytes > b.limits.max_bytes), None)

            if self._stream is not None and all(size_bytes <= b.limits.max_bytes for b in runtime._budgets):
                try:
                    while full() is not None:
                        self._waiting_bytes = size_bytes
                        self._check_locked()
                        runtime._condition.wait()
                    self._check_locked()
                finally:
                    self._waiting_bytes = 0
            budget = full()
            if budget is not None:
                budget._rejected += 1
                raise ResultDeliveryFull(
                    "result buffers exceed runtime delivery byte capacity: "
                    f"requested={size_bytes}, used={budget._usage_bytes}, limit={budget.limits.max_bytes}",
                    reason="bytes",
                    requested=size_bytes,
                    used=budget._usage_bytes,
                    limit=budget.limits.max_bytes,
                )
            lease = _BufferLease(uuid.uuid4().hex, size_bytes)
            for budget in runtime._budgets:
                budget._buffers[lease.lease_id] = lease
                budget._usage_bytes += size_bytes
            return OutputBlockLeaseOwner(runtime, lease)

    def check_preparation(self) -> None:
        with self._runtime._condition:
            self._check_locked()
            if not self._preparing and self._stream is None:
                raise RuntimeError("result preparation has finished")

    def hold(self, payload: ResultPayload) -> None:
        with self._runtime._condition:
            # Preparation must transfer each owner before any fallible build.
            if not self._preparing and self._stream is None:
                raise RuntimeError("result preparation has finished")
            self._payloads.append(payload)

    def start_stream(self, stream: ResultStream) -> None:
        with self._runtime._condition:
            self._check_locked()
            if not self._preparing or self._stream is not None:
                raise RuntimeError("result stream has already been prepared")
            self._stream = stream
            closing = any(b._streams_closed for b in self._runtime._budgets) and self._finish_locked("closed")
        if closing:
            # Install the cleanup owner before cancellation. Preparation still
            # owns the result, so let the adapter finish its ownership transfer;
            # its readiness check will reject delivery and abort preparation.
            self._dispatch_cancellation()

    def request_cancelled(self, error: Callable[[], BaseException]) -> None:
        """Record a native request outcome without retaining its traceback."""
        # Claiming execution takes the request gate before this result gate.
        # Read the request outcome outside our condition to preserve that order.
        failure = error()
        error_type, arguments = type(failure), failure.args
        with self._runtime._condition:
            self._stream_error = lambda: error_type(*arguments)
            self._finish_locked("failed")
            self._runtime._condition.notify_all()
        try:
            self._cleanup()
        except Exception:
            # The consumer or explicit close retains the pending owner.
            pass

    def ready(self, *, delivery_timeout: float | None) -> None:
        timeout = None if delivery_timeout is None else _timeout(delivery_timeout, "delivery_timeout")
        with self._runtime._condition:
            self._check_locked()
            if not self._preparing:
                raise RuntimeError("result preparation has finished")
            self._preparing = False
            self._ready_at = time.monotonic()
            if not self._payloads and self._stream is None:
                self._finish_locked("delivered")
            elif timeout is not None:
                self._deadline = MonotonicDeadline(
                    self._ready_at,
                    timeout,
                    self._expire,
                    timeout_name="delivery_timeout",
                    thread_name="vane-result-deadline",
                )
        if self._deadline is not None:
            self._deadline.start()
            self._expire()
        with self._runtime._condition:
            if self._outcome != "delivered":
                self._check_locked()
            delivered = self._outcome == "delivered"
        if delivered:
            self._cleanup()

    def abort_preparation(self) -> None:
        with self._runtime._condition:
            self._preparing = False
            self._finish_locked("failed")
        self._cleanup()

    def _expire(self) -> None:
        with self._runtime._condition:
            if not self._expire_locked():
                return
        self._dispatch_cancellation()
        try:
            self._cleanup()
        except Exception:
            # Keep the owner for explicit retry, without caching a traceback.
            pass

    def _expire_locked(self) -> bool:
        return self._deadline is not None and self._deadline.expired() and self._finish_locked("delivery_timed_out")

    def _dispatch_cancellation(self) -> None:
        try:
            self._cancellation.cancel(f"result delivery {self._outcome}")
        finally:
            self._cancel_finished.set()
            with self._runtime._condition:
                self._runtime._condition.notify_all()

    def _cleanup(self, *, consumer: bool = False) -> None:
        with self._runtime._condition:
            if self._outcome is None or self.result_id not in self._runtime._results:
                return
            if (
                self._preparing
                or (self._taking and not consumer)
                or self._cleaning
                or not self._cancel_finished.is_set()
            ):
                raise _ResultCleanupPending("result cleanup is still in progress")
            self._cleaning = True
            stream = self._stream
            pending = (*self._payloads, *((stream,) if stream is not None else ()))
        errors: list[BaseException] = []
        try:
            remaining = rollback_actor_pools(
                pending,
                RuntimeError("result cleanup"),
                shutdown=_close_payload,
                cleanup_pending=lambda payload: payload.cleanup_pending(),
                record_error=errors.append,
            )
            with self._runtime._condition:
                self._payloads = deque(p for p in remaining if p is not stream)
            if not remaining:
                self._cancellation.finish()
                if stream is not None:
                    stream.retire(self._lease.release)
                    with self._runtime._condition:
                        self._stream = None
                else:
                    self._lease.release()
            if errors:
                raise RuntimeError("result cleanup failed; retry result.close() or runtime.close()") from errors[0]
        finally:
            with self._runtime._condition:
                self._cleaning = False
                self._runtime._condition.notify_all()

    def complete_external(self, mark_eof: Callable[[], None]) -> None:
        """Finish a native external stream after its owner validated FINISH.

        The context arbitrates cancellation; the result condition arbitrates
        the delivery deadline. No batch is fetched through Python here.
        """
        stream = self._stream
        if stream is None:
            self._cleanup()
            return

        accepted_expiry = False

        def commit() -> None:
            nonlocal accepted_expiry
            with self._runtime._condition:
                accepted_expiry = self._expire_locked()
                self._check_locked()
                if self._preparing or self._taking or self._payloads:
                    raise RuntimeError("external delivery cannot share a Python consumer")
                mark_eof()
                self._finish_locked("delivered")

        try:
            stream.commit(commit)
        finally:
            if accepted_expiry:
                self._dispatch_cancellation()

    def take(self) -> Any:
        """Export one payload, fencing cancellation and expiry before handoff."""
        self._guard_stream()
        with self._runtime._condition:
            if self._preparing:
                raise RuntimeError("result is not ready")
            if self._taking:
                raise RuntimeError("concurrent result consumers are not supported")
            self._taking = True
        value = None
        closing_payload = False
        deferred = False
        accepted_expiry = False
        primary: BaseException | None = None
        try:
            self._expire()
            with self._runtime._condition:
                accepted_expiry = self._expire_locked()
                self._check_locked()
                stream = self._stream
                needs_read = not self._payloads and stream is not None
            if needs_read:
                assert stream is not None
                if not stream.read(self):
                    with self._runtime._condition:
                        accepted_expiry = self._expire_locked()
                        self._check_locked()
                        self._finish_locked("delivered")
                    self._cleanup(consumer=True)
                    raise StopIteration
            with self._runtime._condition:
                self._check_locked()
                payload = self._payloads[0]
            value = payload.export(self._cancellation)
            closing_payload = True
            _close_payload(payload)
            closing_payload = False
            self._expire()

            def commit() -> None:
                nonlocal accepted_expiry
                with self._runtime._condition:
                    # Arbitrate delivery expiry in this same lock. Native
                    # streams additionally fence recorded request cancellation.
                    accepted_expiry = self._expire_locked()
                    self._payloads.popleft()
                    self._check_locked()
                    if not self._payloads and self._stream is None:
                        self._finish_locked("delivered")

            if stream is None:
                commit()
            else:
                stream.commit(commit)
            self._cleanup(consumer=True)
            return value
        except BaseException as error:
            value = None
            if accepted_expiry:
                self._dispatch_cancellation()
            with self._runtime._condition:
                self._finish_locked("failed")
                primary = error
                if self._outcome in {"delivery_timed_out", "cancelled", "closed"}:
                    try:
                        self._check_locked()
                    except (ResultDeliveryTimeout, ResultDeliveryCancelled, ResultDeliveryClosed) as cancelled:
                        if not isinstance(error, type(cancelled)):
                            cancelled.__cause__ = error
                            primary = cancelled
            if not closing_payload:
                try:
                    self._cleanup(consumer=True)
                except _ResultCleanupPending as cleanup_error:
                    deferred = True
                    raise primary from cleanup_error
                except BaseException as cleanup_error:
                    raise primary from cleanup_error
            raise primary
        finally:
            # A traceback points back into this frame. In particular, normal
            # StopIteration must not retain a generator and its last Arrow
            # view in a cycle that only a later GC pass can release.
            primary = None
            with self._runtime._condition:
                self._taking = False
                self._runtime._condition.notify_all()
                retry = deferred or (
                    not closing_payload
                    and (
                        self._outcome in {"closed", "cancelled", "delivery_timed_out"}
                        # Request cancellation/expiry preserves its own error.
                        or (self._outcome == "failed" and self._stream_error is not None)
                    )
                )
            if retry:
                # The last of the consumer and cancellation dispatcher to
                # finish must retry their handoff, even if both saw it busy.
                # Cancellation can also arrive after a successful partial
                # handoff, before the consumer relinquishes its claim.
                try:
                    self._cleanup()
                except Exception:
                    pass

    def cancel(self) -> bool:
        self._guard_stream()
        with self._runtime._condition:
            if not self._finish_locked("cancelled"):
                return False
        self._dispatch_cancellation()
        self._cleanup()
        return True

    @property
    def execution_state(self) -> str | None:
        """Execution outcome, independent of the delivery handle's state."""
        return None if self.context is None else self.context.state

    def read_batch(self) -> Any:
        """Read the next batch; EOF raises StopIteration and errors propagate."""
        return self.take()

    def collect(self) -> pa.Table:
        """Collect remaining rows into caller-owned memory outside the stream budget.

        Copy each batch before requesting another: retaining every stream lease
        would exhaust its window and prevent the collection from making progress.
        """
        import pyarrow as pa

        tables: list[pa.Table] = []
        value = None
        table = None
        try:
            while True:
                try:
                    value = self.take()
                except StopIteration:
                    break
                table = pa.Table.from_batches([value]) if isinstance(value, pa.RecordBatch) else value
                if not isinstance(table, pa.Table):
                    raise TypeError("collect() requires Arrow result batches")
                with pa.BufferOutputStream() as output:
                    with pa.ipc.new_stream(output, table.schema) as writer:
                        writer.write_table(table)
                    copied = output.getvalue()
                with pa.ipc.open_stream(copied) as reader:
                    tables.append(reader.read_all())
                value = table = None
            return pa.concat_tables(tables) if tables else pa.Table.from_batches([], schema=self.schema)
        except BaseException as error:
            value = table = None
            try:
                self.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def close(self) -> None:
        self._guard_stream()
        with self._runtime._condition:
            accepted = self._finish_locked("closed")
        if accepted:
            self._dispatch_cancellation()
        self._cleanup()

    def _guard_stream(self) -> None:
        with self._runtime._condition:
            stream = self._stream
        if stream is not None:
            stream.guard()

    def _close_stream(self, timeout: float = 5.0) -> None:
        """A connection may wait for its interrupted consumer before teardown."""
        deadline = time.monotonic() + cleanup_timeout(timeout)
        while True:
            try:
                self.close()
                return
            except _ResultCleanupPending:
                with self._runtime._condition:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise
                    if self._preparing or self._taking or self._cleaning or not self._cancel_finished.is_set():
                        self._runtime._condition.wait(remaining)

    def __iter__(self) -> QueryResult:
        return self

    def __next__(self) -> Any:
        return self.take()

    def __enter__(self) -> QueryResult:
        return self

    def __exit__(self, _type: object, error: BaseException | None, _traceback: object) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if error is not None:
                raise error from cleanup_error
            raise
