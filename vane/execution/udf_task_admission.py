# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Callable
from typing import Any

from vane.execution.udf_admission import (
    AdmissionExecutorMixin,
    AdmissionLease,
)

_DEFAULT_RAY_TASK_HEAP_BYTES = 2 * 1024**3
_DEFAULT_RAY_ACTOR_HEAP_BYTES = 4 * 1024**3


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


TaskAdmission = AdmissionLease


class TaskAdmissionController:
    """One-lookahead, non-blocking query task-lease admission."""

    def __init__(self, payload: dict[str, Any], *, driver: Any | None = None) -> None:
        self._payload = dict(payload)
        self._query_id = str(self._payload.get("query_id") or "").strip()
        self._stage_id = str(self._payload.get("stage_id") or "").strip()
        if not self._query_id or not self._stage_id:
            raise ValueError("distributed Ray UDF task admission requires query_id and stage_id")
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
        self._error: BaseException | None = None
        self._wakeup: Callable[[], None] | None = None

    def _driver_actor(self):
        if self._driver is not None:
            return self._driver
        import ray

        from vane.runners.ray.driver import (
            RAY_QUERY_DRIVER_ACTOR_NAME,
            RAY_QUERY_DRIVER_ACTOR_NAMESPACE,
        )

        self._driver = ray.get_actor(
            RAY_QUERY_DRIVER_ACTOR_NAME,
            namespace=RAY_QUERY_DRIVER_ACTOR_NAMESPACE,
        )
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
        try:
            request_ref = self._driver_actor().acquire_query_task_lease.remote(request)
            future = request_ref.future()
        except BaseException as exc:
            with self._lock:
                self._state = "failed"
                self._error = exc
            raise
        with self._lock:
            if self._state == "requested" and self._request_id == request["request_id"]:
                self._request_ref = request_ref
        future.add_done_callback(
            lambda done, request_id=str(request["request_id"]): self._finish_request(
                request_id,
                done,
            )
        )
        return True

    def _finish_request(self, request_id: str, future: Any) -> None:
        wakeup: Callable[[], None] | None = None
        abandon = False
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
        if abandon:
            self._driver_actor().cancel_query_task_lease_request.remote(
                request_id,
                submitted=False,
            )
            return
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
            admission = TaskAdmission(
                driver=driver,
                request_id=request_id,
                retained_input_bytes=self._retained_input_bytes,
                lease=dict(self._ready_lease),
                _release_callback=lambda: driver.cancel_query_task_lease_request.remote(
                    request_id,
                    submitted=False,
                ),
            )
            self._state = "idle"
            self._request_id = ""
            self._retained_input_bytes = 0
            self._ready_lease = None
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
        request_id = ""
        with self._lock:
            if self._state == "closed":
                return
            if self._state in {"requested", "ready"}:
                request_id = self._request_id
            self._state = "closed"
            self._request_id = ""
            self._retained_input_bytes = 0
            self._request_ref = None
            self._ready_lease = None
            self._wakeup = None
        if request_id:
            self._driver_actor().cancel_query_task_lease_request.remote(
                request_id,
                submitted=False,
            )


class TaskAdmissionExecutorMixin(AdmissionExecutorMixin):
    """Executor-facing task-admission API consumed by the C++ dispatcher."""

    def _initialize_task_admission(
        self,
        payload: dict[str, Any],
        *,
        driver: Any | None = None,
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
