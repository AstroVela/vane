# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
import uuid
import weakref
from collections.abc import Callable
from typing import Any, TypeAlias

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


TaskAdmission: TypeAlias = AdmissionLease


class _TaskAdmissionCancellation:
    """Durably drive one pre-submission lease cancellation to acknowledgement."""

    def __init__(self, *, driver: Any, request: dict[str, Any]) -> None:
        self._driver = driver
        self._request = {
            **request,
            "resources": dict(request["resources"]),
        }
        self._lock = threading.Lock()
        self._response_future: Any | None = None
        self._response_watchdog: threading.Timer | None = None
        self._retry_timer: threading.Timer | None = None
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

    def start(self) -> None:
        """Start once; callbacks and bounded timers retain cancellation ownership."""
        with self._lock:
            if self._done or self._submitting or self._response_future is not None or self._retry_timer is not None:
                return
            self._submitting = True
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
        watchdog = threading.Timer(
            _TASK_ADMISSION_CLEANUP_RESPONSE_TIMEOUT_S,
            self._response_timed_out,
            args=(response_future,),
        )
        watchdog.daemon = True
        discard_response = False
        with self._lock:
            self._submitting = False
            if self._done:
                discard_response = True
            else:
                self._response_future = response_future
                self._response_watchdog = watchdog
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
                    self._response_watchdog = None
            watchdog.cancel()
            self._schedule_retry()
            return
        with self._lock:
            start_watchdog = (
                not self._done and self._response_future is response_future and self._response_watchdog is watchdog
            )
        if start_watchdog:
            watchdog.start()

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
                watchdog = self._response_watchdog
                self._response_watchdog = None
            if watchdog is not None:
                watchdog.cancel()
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
            watchdog = self._response_watchdog
            self._response_watchdog = None
            retry_timer = self._retry_timer
            self._retry_timer = None
        if current_response is not None and current_response is not response_future:
            try:
                current_response.cancel()
            except BaseException:
                pass
        if watchdog is not None:
            watchdog.cancel()
        if retry_timer is not None:
            retry_timer.cancel()

    def _response_timed_out(self, response_future: Any) -> None:
        with self._lock:
            if self._done or self._response_future is not response_future:
                return
            self._response_future = None
            self._response_watchdog = None
        try:
            response_future.cancel()
        except BaseException:
            pass
        self._schedule_retry()

    def _schedule_retry(self) -> None:
        with self._lock:
            if self._done or self._submitting or self._response_future is not None or self._retry_timer is not None:
                return
            self._retry_count += 1
            timer = threading.Timer(
                self._retry_delay(self._retry_count),
                self._retry,
            )
            timer.daemon = True
            self._retry_timer = timer
        timer.start()

    def _retry(self) -> None:
        with self._lock:
            self._retry_timer = None
        self.start()


class TaskAdmissionController:
    """One-lookahead, non-blocking query task-lease admission."""

    def __init__(self, payload: dict[str, Any], *, driver: Any) -> None:
        self._payload = dict(payload)
        self._query_id = str(self._payload.get("query_id") or "").strip()
        self._stage_id = str(self._payload.get("stage_id") or "").strip()
        if not self._query_id or not self._stage_id:
            raise ValueError("distributed Ray UDF task admission requires query_id and stage_id")
        if driver is None:
            raise ValueError("distributed Ray UDF task admission requires an explicit query driver handle")
        self._resources = ray_udf_task_resource_spec(self._payload)
        self._driver = driver
        self._executor_id = uuid.uuid4().hex
        self._sequence = 0
        self._lock = threading.Lock()
        self._state = "idle"
        self._request_id = ""
        self._retained_input_bytes = 0
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
        }

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
            cancellation = _TaskAdmissionCancellation(
                driver=self._driver_actor(),
                request=request,
            )
            self._cancellation = cancellation
        request_id = str(request["request_id"])
        try:
            request_ref = self._driver_actor().acquire_query_task_lease.remote(request)
            future = ray_object_ref_future(request_ref)
            with self._lock:
                if self._state == "requested" and self._request_id == request_id:
                    self._request_ref = request_ref
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

            future.add_done_callback(finish)
        except BaseException as exc:
            with self._lock:
                if self._state == "requested" and self._request_id == request_id:
                    self._request_ref = None
                    self._state = "failed"
                    self._error = exc
            cancellation.start()
            raise
        return True

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
        with self._lock:
            if self._state == "closed" and self._cancellation is None:
                return
            cancellation = self._cancellation
            self._state = "closed"
            self._request_id = ""
            self._retained_input_bytes = 0
            self._request_ref = None
            self._ready_lease = None
            self._cancellation = None
            self._wakeup = None
        if cancellation is not None:
            cancellation.start()


class TaskAdmissionExecutorMixin(AdmissionExecutorMixin):
    """Executor-facing task-admission API consumed by the C++ dispatcher."""

    def _initialize_task_admission(
        self,
        payload: dict[str, Any],
        *,
        driver: Any,
    ) -> None:
        authority = TaskAdmissionController(payload, driver=driver)
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
