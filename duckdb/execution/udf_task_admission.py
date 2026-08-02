# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import heapq
import os
import threading
import time
import uuid
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from duckdb.execution.ray_control_submission import submit_ray_control
from duckdb.execution.ray_stream_adapter import (
    ray_object_ref_future,
    validate_ray_control_ack,
)
from duckdb.execution.udf_admission import (
    AdmissionExecutorMixin,
    AdmissionLease,
)

_DEFAULT_RAY_TASK_HEAP_BYTES = 2 * 1024**3
_DEFAULT_RAY_ACTOR_HEAP_BYTES = 4 * 1024**3
_TASK_ADMISSION_CLEANUP_RETRY_INITIAL_DELAY_S = 0.01
_TASK_ADMISSION_CLEANUP_RETRY_MAX_DELAY_S = 1.0
_TASK_ADMISSION_CLEANUP_RESPONSE_TIMEOUT_S = 1.0
_TASK_ADMISSION_CLEANUP_RESPONSE_TIMEOUT_MAX_S = 30.0


class _ScheduledCall:
    """One cancellable deadline owned by the shared cleanup scheduler."""

    def __init__(
        self,
        scheduler: _TaskAdmissionCancellationScheduler,
        callback: Callable[[], None],
    ) -> None:
        self._scheduler = scheduler
        self._callback: Callable[[], None] | None = callback
        self._scheduled = False

    def cancel(self) -> None:
        self._scheduler.cancel(self)


class _TaskAdmissionCancellationScheduler:
    """Process-wide deadline scheduler for every admission cancellation."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._queue: list[tuple[float, int, _ScheduledCall]] = []
        self._next_sequence = 0
        self._thread: threading.Thread | None = None

    def create(self, callback: Callable[[], None]) -> _ScheduledCall:
        return _ScheduledCall(self, callback)

    def ensure_started(self) -> None:
        """Start before remote work can transfer cleanup ownership to this clock."""
        with self._cv:
            if self._thread is not None:
                return
            thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="vane-task-admission-cleanup",
            )
            # Publish only a successfully started thread.  If start() raises,
            # a later controller can retry instead of inheriting a permanently
            # poisoned scheduler with no worker.
            thread.start()
            self._thread = thread

    def schedule(self, call: _ScheduledCall, delay_s: float) -> None:
        self.ensure_started()
        with self._cv:
            if call._scheduler is not self:
                raise RuntimeError("scheduled call belongs to a different scheduler")
            if call._callback is None:
                return
            if call._scheduled:
                raise RuntimeError("scheduled call is already armed")
            call._scheduled = True
            self._next_sequence += 1
            heapq.heappush(
                self._queue,
                (
                    time.monotonic() + max(0.0, float(delay_s)),
                    self._next_sequence,
                    call,
                ),
            )
            self._cv.notify()

    def cancel(self, call: _ScheduledCall) -> None:
        with self._cv:
            if call._scheduler is not self:
                raise RuntimeError("scheduled call belongs to a different scheduler")
            call._callback = None
            self._cv.notify()

    def _run(self) -> None:
        while True:
            callback: Callable[[], None] | None = None
            with self._cv:
                while callback is None:
                    while self._queue and self._queue[0][2]._callback is None:
                        heapq.heappop(self._queue)
                    if not self._queue:
                        self._cv.wait()
                        continue
                    deadline, _sequence, call = self._queue[0]
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        self._cv.wait(timeout=remaining)
                        continue
                    heapq.heappop(self._queue)
                    callback = call._callback
                    call._callback = None
            if callback is not None:
                try:
                    callback()
                except BaseException:
                    # Cancellation methods own their retry/error policy. One
                    # malformed callback must not terminate the shared clock.
                    pass


_TASK_ADMISSION_CANCELLATION_SCHEDULER = _TaskAdmissionCancellationScheduler()


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def ray_udf_task_memory_bytes(payload: dict[str, Any]) -> int:
    backend = str(payload.get("execution_backend") or "").strip()
    if backend == "ray_task":
        env_name = "VANE_UDF_TASK_HEAP_BYTES"
        default = _DEFAULT_RAY_TASK_HEAP_BYTES
    elif backend == "ray_actor":
        env_name = "VANE_UDF_ACTOR_HEAP_BYTES"
        default = _DEFAULT_RAY_ACTOR_HEAP_BYTES
    else:
        raise ValueError(f"task admission requires a Ray UDF backend, got {backend!r}")
    raw = payload.get("memory_bytes")
    if raw is None:
        raw = os.environ.get(env_name, str(default))
    return _positive_int(raw, "memory_bytes")


def ray_udf_task_resource_spec(payload: dict[str, Any]) -> dict[str, Any]:
    backend = str(payload.get("execution_backend") or "").strip()
    resident = backend == "ray_actor"
    return {
        "cpu": 0.0 if resident else float(payload.get("cpus", 1.0)),
        "gpu": 0.0 if resident else float(payload.get("gpus", 0.0)),
        "heap_bytes": 0 if resident else ray_udf_task_memory_bytes(payload),
        "object_store_bytes": _positive_int(
            payload.get("udf_task_input_max_bytes"),
            "udf_task_input_max_bytes",
        ),
    }


@dataclass(kw_only=True)
class TaskAdmission(AdmissionLease):
    """Ray task admission plus its long-lived local control owner."""

    submission_scope: str

    def __post_init__(self) -> None:
        self.submission_scope = str(self.submission_scope or "").strip()
        if not self.submission_scope:
            raise ValueError("Ray task admission requires an explicit submission scope")


class _TaskAdmissionCancellation:
    """Durably drive one task-admission cancellation to acknowledgement."""

    def __init__(
        self,
        *,
        driver: Any,
        request: dict[str, Any],
        submission_scope: str,
    ) -> None:
        self._driver = driver
        self._request = {
            **request,
            "resources": dict(request["resources"]),
        }
        self._submission_scope = str(submission_scope or "").strip()
        if not self._submission_scope:
            raise ValueError("Ray task admission cancellation requires an explicit submission scope")
        self._lock = threading.Lock()
        self._response_future: Any | None = None
        self._response_deadline: _ScheduledCall | None = None
        self._retry_call: _ScheduledCall | None = None
        self._submitting = False
        self._retry_count = 0
        self._done = False

    @staticmethod
    def _retry_delay(retry_count: int) -> float:
        exponent = min(max(0, int(retry_count) - 1), 30)
        return min(
            _TASK_ADMISSION_CLEANUP_RETRY_INITIAL_DELAY_S * (2**exponent),
            _TASK_ADMISSION_CLEANUP_RETRY_MAX_DELAY_S,
        )

    @staticmethod
    def _response_timeout(retry_count: int) -> float:
        exponent = min(max(0, int(retry_count)), 30)
        return min(
            _TASK_ADMISSION_CLEANUP_RESPONSE_TIMEOUT_S * (2**exponent),
            _TASK_ADMISSION_CLEANUP_RESPONSE_TIMEOUT_MAX_S,
        )

    def start(self) -> None:
        """Start once; callbacks and shared deadlines retain cleanup ownership."""
        with self._lock:
            if self._done or self._submitting or self._response_future is not None or self._retry_call is not None:
                return
            self._submitting = True
        try:
            submission_future = submit_ray_control(
                self._submission_scope,
                self._submit,
            )
        except BaseException:
            with self._lock:
                self._submitting = False
            self._schedule_retry()
            return
        try:
            submission_future.add_done_callback(self._finish_submission)
        except BaseException:
            submission_future.cancel()
            with self._lock:
                self._submitting = False
            self._schedule_retry()

    def _finish_submission(self, submission_future: Any) -> None:
        """Retry if admitted control work is rejected before invocation."""
        try:
            submission_future.result()
        except BaseException:
            with self._lock:
                if self._done or not self._submitting:
                    return
                self._submitting = False
            self._schedule_retry()

    def _submit(self) -> None:
        """Submit one driver RPC without occupying its caller or deadline thread."""
        try:
            response_ref = self._driver.cancel_query_task_lease_request.remote(
                dict(self._request),
            )
            response_future = ray_object_ref_future(response_ref)
        except BaseException:
            with self._lock:
                self._submitting = False
            self._schedule_retry()
            return
        response_deadline = _TASK_ADMISSION_CANCELLATION_SCHEDULER.create(
            lambda: self._response_timed_out(response_future),
        )
        discard_response = self._publish_response_future(
            response_future,
            response_deadline,
        )
        if discard_response:
            try:
                response_future.cancel()
            except BaseException:
                pass
            return
        owner_ref = weakref.ref(self)

        def finish(done: Any) -> None:
            owner = owner_ref()
            if owner is not None:
                owner._finish(done)

        try:
            response_future.add_done_callback(finish)
        except BaseException:
            with self._lock:
                if self._response_future is response_future:
                    self._response_future = None
                    self._response_deadline = None
            response_deadline.cancel()
            self._schedule_retry()
            return
        with self._lock:
            start_deadline = (
                not self._done
                and self._response_future is response_future
                and self._response_deadline is response_deadline
            )
        if start_deadline:
            _TASK_ADMISSION_CANCELLATION_SCHEDULER.schedule(
                response_deadline,
                self._response_timeout(self._retry_count),
            )

    def _publish_response_future(
        self,
        response_future: Any,
        response_deadline: _ScheduledCall,
    ) -> bool:
        """Publish a response unless a concurrent acknowledgement finished."""
        with self._lock:
            self._submitting = False
            if self._done:
                return True
            self._response_future = response_future
            self._response_deadline = response_deadline
            return False

    def _finish(self, response_future: Any) -> None:
        try:
            validate_ray_control_ack(
                response_future.result(),
                field="cancelled",
            )
        except BaseException:
            with self._lock:
                if self._done or self._response_future is not response_future:
                    return
                self._response_future = None
                response_deadline = self._response_deadline
                self._response_deadline = None
            if response_deadline is not None:
                response_deadline.cancel()
            self._schedule_retry()
            return
        with self._lock:
            if self._done:
                return
            # A late acknowledgement from a timed-out attempt is still a valid
            # proof for this idempotent cancellation request.
            self._done = True
            current_response = self._response_future
            self._response_future = None
            response_deadline = self._response_deadline
            self._response_deadline = None
            retry_call = self._retry_call
            self._retry_call = None
        if current_response is not None and current_response is not response_future:
            try:
                current_response.cancel()
            except BaseException:
                pass
        if response_deadline is not None:
            response_deadline.cancel()
        if retry_call is not None:
            retry_call.cancel()

    def _response_timed_out(self, response_future: Any) -> None:
        with self._lock:
            if self._done or self._response_future is not response_future:
                return
            self._response_future = None
            self._response_deadline = None
        try:
            cancel = getattr(response_future, "cancel", None)
            if callable(cancel):
                cancel()
        except BaseException:
            pass
        self._schedule_retry()

    def _schedule_retry(self) -> None:
        with self._lock:
            if self._done or self._submitting or self._response_future is not None or self._retry_call is not None:
                return
            self._retry_count += 1
            retry_call: _ScheduledCall

            def retry() -> None:
                self._retry(retry_call)

            retry_call = _TASK_ADMISSION_CANCELLATION_SCHEDULER.create(retry)
            self._retry_call = retry_call
            retry_delay = self._retry_delay(self._retry_count)
        _TASK_ADMISSION_CANCELLATION_SCHEDULER.schedule(
            retry_call,
            retry_delay,
        )

    def _retry(self, retry_call: _ScheduledCall) -> None:
        with self._lock:
            if self._retry_call is not retry_call:
                return
            self._retry_call = None
        self.start()


class TaskAdmissionController:
    """One-lookahead query task admission that never blocks its dispatcher."""

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        driver: Any,
        query_generation_capability: str,
    ) -> None:
        self._payload = dict(payload)
        self._query_id = str(self._payload.get("query_id") or "").strip()
        self._stage_id = str(self._payload.get("stage_id") or "").strip()
        if not self._query_id or not self._stage_id:
            raise ValueError("distributed Ray UDF task admission requires query_id and stage_id")
        if driver is None:
            raise ValueError("distributed Ray UDF task admission requires an explicit query driver handle")
        self._query_generation_capability = str(query_generation_capability or "").strip()
        if not self._query_generation_capability:
            raise ValueError("distributed Ray UDF task admission requires a query generation capability")
        # The cancellation deadline owner must exist before this controller can
        # submit any remote lease request.  Startup failure is therefore a
        # pre-submit construction error, never a post-submit ownership gap.
        _TASK_ADMISSION_CANCELLATION_SCHEDULER.ensure_started()
        self._resources = ray_udf_task_resource_spec(self._payload)
        self._driver = driver
        self._executor_id = uuid.uuid4().hex
        self._sequence = 0
        self._lock = threading.Lock()
        self._state = "idle"
        self._request_id = ""
        self._retained_input_bytes = 0
        self._request_submit_future: Any | None = None
        self._request_ref: Any | None = None
        self._ready_lease: dict[str, Any] | None = None
        self._cancellation: _TaskAdmissionCancellation | None = None
        self._error: BaseException | None = None
        self._wakeup: Callable[[], None] | None = None

    def _driver_actor(self) -> Any:
        return self._driver

    def register_wakeup(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._wakeup = callback

    def _build_request(self, retained_input_bytes: int) -> dict[str, Any]:
        self._sequence += 1
        identity = f"executor:{self._executor_id}:admission:{self._sequence}"
        return {
            "request_id": f"request:task:{self._stage_id}:{identity}",
            "query_id": self._query_id,
            "stage_id": self._stage_id,
            "task_id": f"task:{self._stage_id}:{identity}",
            "attempt_id": f"attempt:{self._executor_id}:{self._sequence}",
            "node_id": None,
            "retained_input_bytes": retained_input_bytes,
            "resources": dict(self._resources),
            "query_generation_capability": self._query_generation_capability,
        }

    @staticmethod
    def _submission_scope_for_request(request_id: str) -> str:
        request_key = str(request_id or "").strip()
        if not request_key:
            raise ValueError("Ray task admission submission scope requires a request identity")
        return f"udf-stream:{request_key}"

    def request(self, retained_input_bytes: int) -> bool:
        retained = int(retained_input_bytes)
        if retained < 0:
            raise ValueError("retained_input_bytes must be >= 0")
        with self._lock:
            self._raise_if_failed_locked()
            if self._state == "closed":
                raise RuntimeError("task admission controller is closed")
            if self._state != "idle":
                return False
            request = self._build_request(retained)
            self._state = "requested"
            self._request_id = str(request["request_id"])
            self._retained_input_bytes = retained
            submission_scope = self._submission_scope_for_request(self._request_id)
            cancellation = _TaskAdmissionCancellation(
                driver=self._driver_actor(),
                request=request,
                submission_scope=submission_scope,
            )
            self._cancellation = cancellation
        request_id = str(request["request_id"])
        driver = self._driver_actor()
        submitted_request = {
            **request,
            "resources": dict(request["resources"]),
        }
        try:
            submission = submit_ray_control(
                submission_scope,
                lambda: driver.acquire_query_task_lease.remote(submitted_request),
            )
        except BaseException as exc:
            with self._lock:
                if self._state == "requested" and self._request_id == request_id:
                    self._state = "failed"
                    self._error = exc
                    self._cancellation = None
                    wakeup = self._wakeup
                else:
                    wakeup = None
            # submit_ray_control rejects atomically, so this request never
            # crossed the remote boundary and owns nothing to cancel.
            if wakeup is not None:
                wakeup()
            raise
        with self._lock:
            publish = self._state == "requested" and self._request_id == request_id
            if publish:
                self._request_submit_future = submission
        if not publish:
            submission.cancel()
            cancellation.start()
            return True

        controller_ref = weakref.ref(self)

        def finish_submission(done: Any) -> None:
            controller = controller_ref()
            if controller is None:
                cancellation.start()
                return
            controller._finish_request_submission(
                request_id,
                cancellation,
                done,
            )

        try:
            submission.add_done_callback(finish_submission)
        except BaseException as exc:
            submission.cancel()
            with self._lock:
                if (
                    self._state == "requested"
                    and self._request_id == request_id
                    and self._request_submit_future is submission
                ):
                    self._request_submit_future = None
                    self._state = "failed"
                    self._error = exc
                    wakeup = self._wakeup
                else:
                    wakeup = None
            cancellation.start()
            if wakeup is not None:
                wakeup()
            raise
        return True

    def _finish_request_submission(
        self,
        request_id: str,
        cancellation: _TaskAdmissionCancellation,
        submission: Any,
    ) -> None:
        try:
            request_ref = submission.result()
            future = ray_object_ref_future(request_ref)
        except BaseException as exc:
            with self._lock:
                active = (
                    self._state == "requested"
                    and self._request_id == request_id
                    and self._request_submit_future is submission
                )
                if self._request_submit_future is submission:
                    self._request_submit_future = None
                if active:
                    self._state = "failed"
                    self._error = exc
                    wakeup = self._wakeup
                else:
                    wakeup = None
            cancellation.start()
            if wakeup is not None:
                wakeup()
            return

        with self._lock:
            active = (
                self._state == "requested"
                and self._request_id == request_id
                and self._request_submit_future is submission
            )
            if self._request_submit_future is submission:
                self._request_submit_future = None
            if active:
                self._request_ref = request_ref
        if not active:
            cancellation.start()
            return

        controller_ref = weakref.ref(self)

        def finish(done: Any) -> None:
            controller = controller_ref()
            if controller is None:
                cancellation.start()
                return
            controller._finish_request(
                request_id,
                cancellation,
                done,
            )

        try:
            future.add_done_callback(finish)
        except BaseException as exc:
            with self._lock:
                active = (
                    self._state == "requested" and self._request_id == request_id and self._request_ref is request_ref
                )
                if active:
                    self._request_ref = None
                    self._state = "failed"
                    self._error = exc
                    wakeup = self._wakeup
                else:
                    wakeup = None
            cancellation.start()
            if wakeup is not None:
                wakeup()

    def _finish_request(
        self,
        request_id: str,
        cancellation: _TaskAdmissionCancellation,
        future: Any,
    ) -> None:
        wakeup: Callable[[], None] | None = None
        abandon = False
        cancel = False
        try:
            grant = future.result()
            if not isinstance(grant, dict) or not grant.get("granted"):
                reason = grant.get("blocked_reason") if isinstance(grant, dict) else "invalid_grant"
                raise RuntimeError(f"Ray UDF task admission failed: {reason}")
            lease = grant.get("lease")
            if not isinstance(lease, dict) or not str(lease.get("lease_id") or ""):
                raise RuntimeError("Ray UDF task admission grant is missing lease identity")
            with self._lock:
                if self._state != "requested" or self._request_id != request_id:
                    abandon = True
                else:
                    self._request_ref = None
                    self._ready_lease = dict(lease)
                    self._state = "ready"
                    wakeup = self._wakeup
        except BaseException as exc:
            with self._lock:
                if self._state != "requested" or self._request_id != request_id:
                    return
                self._request_ref = None
                self._state = "failed"
                self._error = exc
                wakeup = self._wakeup
                cancel = True
        if abandon:
            cancellation.start()
        elif cancel:
            cancellation.start()
        if wakeup is not None:
            wakeup()

    def take(self, retained_input_bytes: int) -> TaskAdmission:
        retained = int(retained_input_bytes)
        with self._lock:
            self._raise_if_failed_locked()
            if self._state != "ready" or self._ready_lease is None:
                raise RuntimeError("task admission is not ready")
            if retained != self._retained_input_bytes:
                raise RuntimeError(
                    "task admission retained input bytes do not match: "
                    f"requested={self._retained_input_bytes} consumed={retained}"
                )
            driver = self._driver_actor()
            request_id = self._request_id
            cancellation = self._cancellation
            if cancellation is None:
                raise RuntimeError("task admission is missing cancellation ownership")
            admission = TaskAdmission(
                driver=driver,
                request_id=request_id,
                retained_input_bytes=self._retained_input_bytes,
                lease=dict(self._ready_lease),
                submission_scope=self._submission_scope_for_request(request_id),
                _release_callback=cancellation.start,
            )
            self._state = "idle"
            self._request_id = ""
            self._retained_input_bytes = 0
            self._ready_lease = None
            self._cancellation = None
            return admission

    def state(self) -> dict[str, Any]:
        with self._lock:
            state = {
                "state": self._state,
                "available": self._state == "ready",
                "retained_input_bytes": self._retained_input_bytes,
            }
            if self._state == "failed" and self._error is not None:
                state["error"] = str(self._error)
            return state

    def _raise_if_failed_locked(self) -> None:
        if self._state != "failed":
            return
        assert self._error is not None
        raise RuntimeError(f"task admission failed: {self._error}") from self._error

    def close(self) -> None:
        cancellation: _TaskAdmissionCancellation | None = None
        submission: Any | None = None
        with self._lock:
            if self._state == "closed" and self._cancellation is None:
                return
            cancellation = self._cancellation
            submission = self._request_submit_future
            self._state = "closed"
            self._request_id = ""
            self._retained_input_bytes = 0
            self._request_submit_future = None
            self._request_ref = None
            self._ready_lease = None
            self._cancellation = None
            self._wakeup = None
        if submission is not None:
            submission.cancel()
        if cancellation is not None:
            cancellation.start()


class TaskAdmissionExecutorMixin(AdmissionExecutorMixin):
    """Executor-facing task-admission API consumed by the C++ dispatcher."""

    def _initialize_task_admission(
        self,
        payload: dict[str, Any],
        *,
        driver: Any,
        query_generation_capability: str,
    ) -> None:
        authority = TaskAdmissionController(
            payload,
            driver=driver,
            query_generation_capability=query_generation_capability,
        )
        self._task_admission = authority
        self._initialize_admission(authority)

    def _close_task_admission(self) -> None:
        self._close_admission()


__all__ = [
    "TaskAdmission",
    "TaskAdmissionController",
    "TaskAdmissionExecutorMixin",
    "ray_udf_task_memory_bytes",
    "ray_udf_task_resource_spec",
]
