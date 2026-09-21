# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded result ownership and delivery, independent of execution backends."""

from __future__ import annotations

import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Protocol

from vane.execution.data_lifecycle import _OUTPUT_STATES, OutputBlockLeaseOwner
from vane.execution.request_admission import _timeout
from vane.execution.request_deadline import MonotonicDeadline
from vane.execution.udf_actor_pool_lifecycle import rollback_actor_pools
from vane.execution.udf_admission import AdmissionLease
from vane.execution.udf_lifecycle import ExecutionCancellationScope, ExecutionCancelledError


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
    """A result slot or result buffer would exceed the delivery capacity."""


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

    def __init__(self, limits: ResultDeliveryLimits) -> None:
        if not isinstance(limits, ResultDeliveryLimits):
            raise TypeError("result_limit must be ResultDeliveryLimits")
        self.limits = limits
        self._condition = threading.Condition()
        self._results: dict[str, ManagedResult] = {}
        self._buffers: dict[str, _BufferLease] = {}
        self._usage_bytes = 0
        self._closed = False
        self._completed: Counter[str] = Counter()
        self._rejected = 0

    def begin(self) -> ManagedResult:
        with self._condition:
            if self._closed:
                raise ResultDeliveryClosed("result delivery runtime is closed")
            if len(self._results) >= self.limits.max_results:
                self._rejected += 1
                raise ResultDeliveryFull(
                    f"runtime result slots are full: used={len(self._results)}, limit={self.limits.max_results}"
                )
            result = ManagedResult(self, uuid.uuid4().hex)
            self._results[result.result_id] = result
            return result

    def _release_result(self, result: ManagedResult) -> None:
        with self._condition:
            if self._results.pop(result.result_id, None) is not None:
                assert result._outcome is not None
                self._completed[result._outcome] += 1
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
            lease = self._buffers.pop(lease_id, None)
            if lease is None:
                return False
            self._usage_bytes -= lease.size_bytes
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
                "closed": self._closed,
            }

    def close(self) -> None:
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
                result.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise RuntimeError(
                "result cleanup failed or is in progress; retry result.close() or runtime.close()"
            ) from errors[0]


class ManagedResult:
    """A single-consumer iterator whose remaining output has a separate lifetime.

    ``take()`` transfers one payload to the caller. Its underlying buffers remain
    charged until the last exported view is gone, even after this handle closes.
    """

    def __init__(self, runtime: RuntimeResultDelivery, result_id: str) -> None:
        self._runtime = runtime
        self.result_id = result_id
        self._payloads: deque[ResultPayload] = deque()
        self._preparing = True
        self._taking = False
        self._cleaning = False
        self._outcome: str | None = None
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
            if not self._preparing:
                raise RuntimeError("result preparation has finished")
            if runtime._usage_bytes + size_bytes > runtime.limits.max_bytes:
                runtime._rejected += 1
                raise ResultDeliveryFull(
                    "result buffers exceed runtime delivery byte capacity: "
                    f"requested={size_bytes}, used={runtime._usage_bytes}, limit={runtime.limits.max_bytes}"
                )
            lease = _BufferLease(uuid.uuid4().hex, size_bytes)
            runtime._buffers[lease.lease_id] = lease
            runtime._usage_bytes += size_bytes
            return OutputBlockLeaseOwner(runtime, lease)

    def check_preparation(self) -> None:
        with self._runtime._condition:
            self._check_locked()
            if not self._preparing:
                raise RuntimeError("result preparation has finished")

    def hold(self, payload: ResultPayload) -> None:
        with self._runtime._condition:
            # Preparation must transfer each owner before any fallible build.
            if not self._preparing:
                raise RuntimeError("result preparation has finished")
            self._payloads.append(payload)

    def ready(self, *, delivery_timeout: float | None) -> None:
        timeout = None if delivery_timeout is None else _timeout(delivery_timeout, "delivery_timeout")
        with self._runtime._condition:
            self._check_locked()
            if not self._preparing:
                raise RuntimeError("result preparation has finished")
            self._preparing = False
            if not self._payloads:
                self._finish_locked("delivered")
            elif timeout is not None:
                self._deadline = MonotonicDeadline(
                    time.monotonic(),
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
            pending = tuple(self._payloads)
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
                self._payloads = deque(remaining)
            if not remaining:
                self._cancellation.finish()
                self._lease.release()
            if errors:
                raise RuntimeError("result cleanup failed; retry result.close() or runtime.close()") from errors[0]
        finally:
            with self._runtime._condition:
                self._cleaning = False

    def take(self) -> Any:
        """Export one payload, fencing cancellation and expiry before handoff."""
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
                payload = self._payloads[0]
            value = payload.export(self._cancellation)
            closing_payload = True
            _close_payload(payload)
            closing_payload = False
            self._expire()
            with self._runtime._condition:
                # The poll may precede lock acquisition by an arbitrary
                # delay. Arbitrate expiry and delivery in this same lock.
                accepted_expiry = self._expire_locked()
                self._payloads.popleft()
                self._check_locked()
                if not self._payloads:
                    self._finish_locked("delivered")
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
                retry = deferred or (
                    not closing_payload and self._outcome in {"closed", "cancelled", "delivery_timed_out"}
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
        with self._runtime._condition:
            if not self._finish_locked("cancelled"):
                return False
        self._dispatch_cancellation()
        self._cleanup()
        return True

    def close(self) -> None:
        with self._runtime._condition:
            accepted = self._finish_locked("closed")
        if accepted:
            self._dispatch_cancellation()
        self._cleanup()

    def __iter__(self) -> ManagedResult:
        return self

    def __next__(self) -> Any:
        return self.take()

    def __enter__(self) -> ManagedResult:
        return self

    def __exit__(self, _type: object, error: BaseException | None, _traceback: object) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if error is not None:
                raise error from cleanup_error
            raise
