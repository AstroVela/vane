# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

_REQUIRED_GENERATOR_METHODS = ("__anext__", "completed")
_REQUIRED_CORE_WORKER_METHODS = (
    "async_delete_object_ref_stream",
    "is_object_ref_stream_finished",
)


def validate_ray_stream_contract(ray_module: Any) -> None:
    version = str(getattr(ray_module, "__version__", "")).strip() or "unknown"
    object_ref_generator = getattr(ray_module, "ObjectRefGenerator", None)
    if object_ref_generator is None:
        raise RuntimeError(f"Ray {version!r} ObjectRefGenerator implementation is unavailable")
    missing = [name for name in _REQUIRED_GENERATOR_METHODS if not callable(getattr(object_ref_generator, name, None))]
    if missing:
        raise RuntimeError(f"Ray {version!r} ObjectRefGenerator contract is missing: " + ", ".join(missing))


class TaskLeaseObjectRefGenerator:
    """Ray stream started immediately from an already granted task admission."""

    def __init__(
        self,
        *,
        admission: Any,
        submitter: Callable[[dict[str, Any]], Any],
        ray_module: Any | None = None,
    ) -> None:
        if ray_module is None:
            import ray as imported_ray

            active_ray_module = imported_ray
        else:
            active_ray_module = ray_module
        try:
            validate_ray_stream_contract(active_ray_module)
        except BaseException:
            admission.release()
            raise
        request_id = str(admission.request_id or "").strip()
        lease = dict(admission.lease)
        if not request_id or not str(lease.get("lease_id") or ""):
            admission.release()
            raise ValueError("pregranted Ray UDF task admission is missing identity")
        self._ray = active_ray_module
        self._driver = admission.driver
        self._request_id = request_id
        self._generator: Any | None = None
        self._lease: dict[str, Any] | None = lease
        self._submitted = False
        self._released = False
        self._cancelled = False
        self._lock = threading.Lock()
        try:
            self._generator = submitter(dict(lease))
        except BaseException:
            self._lease = None
            admission.release()
            raise
        admission.handoff()
        self._submitted = True
        self._driver.mark_query_task_lease_submitted.remote(
            self._request_id,
            str(lease["lease_id"]),
        )

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def driver(self) -> Any:
        return self._driver

    @property
    def lease(self) -> dict[str, Any] | None:
        return None if self._lease is None else dict(self._lease)

    @property
    def generator(self) -> Any | None:
        return self._generator

    @property
    def start(self) -> Any:
        with self._lock:
            if self._cancelled or self._generator is None:
                raise RuntimeError("Ray UDF task stream is not active")
            return self._generator

    def release_task(self) -> None:
        with self._lock:
            if self._released or self._lease is None:
                return
            self._released = True
            lease = dict(self._lease)
        self._driver.release_query_task_lease.remote(
            self._request_id,
            str(lease["lease_id"]),
            str(lease["attempt_id"]),
        )

    def cancel(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            generator = self._generator
            submitted = self._submitted
            self._released = True
            self._generator = None
        if generator is not None:
            self._ray.cancel(generator, force=True, recursive=True)
        self._driver.cancel_query_task_lease_request.remote(
            self._request_id,
            submitted=bool(submitted),
        )

    def retire_failed(self) -> None:
        """Retire a terminally failed task without cancelling completed work."""
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            submitted = self._submitted
            self._released = True
            self._generator = None
        self._driver.cancel_query_task_lease_request.remote(
            self._request_id,
            submitted=bool(submitted),
        )

    def retire_local_state(self) -> None:
        """Drop client-side task payload ownership after terminal cleanup."""
        with self._lock:
            self._generator = None
            self._lease = None


class RayStreamAdapter:
    """Capability-validated, non-blocking access to one Ray streaming generator."""

    def __init__(self, source: Any, *, ray_module: Any | None = None) -> None:
        if ray_module is None:
            import ray as imported_ray

            active_ray_module = imported_ray
        else:
            active_ray_module = ray_module

        validate_ray_stream_contract(active_ray_module)
        self._ray = active_ray_module
        self._source = source
        self._leased = source if isinstance(source, TaskLeaseObjectRefGenerator) else None
        self._generator = self._leased.generator if self._leased is not None else source
        self._completion_ref: Any | None = None
        self._drained = False
        self._task_released = False
        self._retired = False
        self._validate_generator_if_started()

    def _validate_generator_if_started(self) -> None:
        if self._generator is None:
            return
        version = str(getattr(self._ray, "__version__", "")).strip() or "unknown"
        missing = [name for name in _REQUIRED_GENERATOR_METHODS if not callable(getattr(self._generator, name, None))]
        if missing:
            raise TypeError(
                f"Ray {version!r} stream source is not an ObjectRefGenerator; missing " + ", ".join(missing)
            )
        worker = getattr(self._generator, "worker", None)
        core_worker = getattr(worker, "core_worker", None)
        missing_core = [
            name for name in _REQUIRED_CORE_WORKER_METHODS if not callable(getattr(core_worker, name, None))
        ]
        if missing_core:
            raise TypeError(
                f"Ray {version!r} stream source is missing required CoreWorker contract: " + ", ".join(missing_core)
            )
        self._completion_ref = self._generator.completed()

    @property
    def task_lease(self) -> dict[str, Any] | None:
        return None if self._leased is None else self._leased.lease

    @property
    def task_request_id(self) -> str:
        return "" if self._leased is None else self._leased.request_id

    @property
    def driver(self) -> Any | None:
        return None if self._leased is None else self._leased.driver

    async def read_next_ref_async(self) -> Any:
        if self._generator is None:
            raise RuntimeError("Ray stream has not acquired its task lease")
        if self._drained:
            raise StopAsyncIteration
        ref = await self._generator.__anext__()
        if hasattr(ref, "is_nil") and ref.is_nil():
            raise RuntimeError("Ray generator returned a nil ObjectRef")
        return ref

    @property
    def completion_ref(self) -> Any:
        if self._completion_ref is None:
            raise RuntimeError("Ray stream completion ObjectRef is unavailable")
        return self._completion_ref

    def stream_finished(self) -> bool:
        if self._generator is None:
            return self._drained
        return bool(self._generator.worker.core_worker.is_object_ref_stream_finished(self.completion_ref))

    def is_terminal_ref(self, ref: Any) -> bool:
        return self._generator is not None and ref == self._completion_ref

    def mark_drained(self) -> None:
        self._drained = True

    def release_task(self) -> None:
        if self._task_released:
            return
        self._task_released = True
        if self._leased is not None:
            self._leased.release_task()

    @staticmethod
    def _delete_object_ref_stream(generator: Any, completion_ref: Any) -> None:
        worker = generator.worker
        core_worker = worker.core_worker
        try:
            core_worker.async_delete_object_ref_stream(completion_ref)
        finally:
            # Ray 2.55's ObjectRefGenerator.__del__ unconditionally deletes the
            # stream through ``generator.worker``.  We retire streams
            # explicitly, so leaving that worker attached performs a second
            # delete when the last local reference is dropped.  Under sustained
            # streaming load that duplicate CoreWorker call can block the
            # collector thread and, in a RayWorkerActor, the whole FTE consumer.
            # Detach only after capturing the pinned CoreWorker/ref above; the
            # adapter is terminal at every call site and must not be reused.
            generator.worker = None

    def retire(self) -> None:
        """Deterministically retire a fully drained Ray generator stream.

        Ray keeps task metadata and dependency lineage until the language
        frontend deletes the ObjectRef stream.  Relying on ObjectRefGenerator's
        ``__del__`` makes that cleanup depend on incidental Python references in
        this long-lived multiplexer, so terminal streams are retired explicitly.
        """
        if self._retired:
            return
        self._retired = True
        leased = self._leased
        generator = self._generator
        completion_ref = self._completion_ref
        self.release_task()
        self._source = None
        self._leased = None
        self._generator = None
        self._completion_ref = None
        if leased is not None:
            leased.retire_local_state()
        if generator is not None and completion_ref is not None:
            self._delete_object_ref_stream(generator, completion_ref)

    def cancel(self) -> None:
        if self._retired:
            return
        self._retired = True
        leased = self._leased
        generator = self._generator
        completion_ref = self._completion_ref
        self._source = None
        self._leased = None
        self._generator = None
        self._completion_ref = None
        if leased is not None:
            leased.cancel()
            leased.retire_local_state()
        elif generator is not None:
            self._ray.cancel(generator, force=True, recursive=True)
        if generator is not None and completion_ref is not None:
            self._delete_object_ref_stream(generator, completion_ref)

    def retire_failed(self) -> None:
        """Retire a terminal failure without racing it with ``ray.cancel``.

        Once Ray has published task completion, or a Vane stream has emitted
        its terminal error pair, force-cancelling the generator can race the
        normal task reply in Ray's CoreWorker.  The task lease still follows
        the failure path, but the already-ending remote work is left alone.
        """
        if self._retired:
            return
        self._retired = True
        leased = self._leased
        generator = self._generator
        completion_ref = self._completion_ref
        self._source = None
        self._leased = None
        self._generator = None
        self._completion_ref = None
        if leased is not None:
            leased.retire_failed()
            leased.retire_local_state()
        if generator is not None and completion_ref is not None:
            self._delete_object_ref_stream(generator, completion_ref)


__all__ = [
    "RayStreamAdapter",
    "TaskLeaseObjectRefGenerator",
    "validate_ray_stream_contract",
]
