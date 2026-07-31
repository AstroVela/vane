# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

_REQUIRED_GENERATOR_METHODS = ("__anext__", "completed")
_REQUIRED_RAY_METHODS = ("wait",)
_REQUIRED_CORE_WORKER_METHODS = (
    "async_delete_object_ref_stream",
    "is_object_ref_stream_finished",
)


@dataclass(frozen=True)
class RayStreamCleanupOperation:
    """One non-blocking terminal transition driven by the collector loop."""

    operation: Callable[[], Any]
    retry_on_error: bool = False
    retry_on_incomplete: bool = False

    def __call__(self) -> Any:
        return self.operation()


def validate_ray_control_ack(response: Any, *, field: str) -> dict[str, Any]:
    """Require driver acceptance before completing a cleanup ticket."""
    if not isinstance(response, dict) or response.get(field) is not True:
        raise RuntimeError(f"Ray task lease control handoff was not accepted: {field}")
    return response


def ray_object_ref_future(response_ref: Any) -> Any:
    """Return the public concurrent Future used by the cleanup event loop."""
    future_method = getattr(response_ref, "future", None)
    if not callable(future_method):
        raise TypeError("Ray control ObjectRef does not expose future()")
    future = future_method()
    missing = [name for name in ("add_done_callback", "done", "result") if not callable(getattr(future, name, None))]
    if missing:
        raise TypeError("Ray control ObjectRef future is missing: " + ", ".join(missing))
    return future


def validate_ray_stream_contract(ray_module: Any) -> None:
    version = str(getattr(ray_module, "__version__", "")).strip() or "unknown"
    missing_ray = [name for name in _REQUIRED_RAY_METHODS if not callable(getattr(ray_module, name, None))]
    if missing_ray:
        raise RuntimeError(f"Ray {version!r} runtime contract is missing: " + ", ".join(missing_ray))
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
        self._execution_backend = "ray_actor" if lease.get("actor_index") is not None else "ray_task"
        self._generator: Any | None = None
        self._completion_ref: Any | None = None
        self._lease: dict[str, Any] | None = lease
        self._lease_identity = dict(lease)
        self._released = False
        self._release_response_future: Any | None = None
        self._cancelled = False
        self._task_cleanup_handed_off = False
        self._task_cleanup_response_future: Any | None = None
        self._cancel_generator: Any | None = None
        self._generator_cancel_submitted = False
        self._lock = threading.Lock()
        try:
            self._generator = submitter(dict(lease))
        except BaseException:
            self._lease = None
            admission.release()
            raise
        # Stream-contract validation belongs to RayStreamAdapter, which is
        # constructed by AsyncResultCollector under its durable cleanup-ticket
        # owner.  Once remote work exists, this constructor must not discover a
        # post-submit error that has no scheduler available to retain an
        # asynchronous driver handoff.
        admission.handoff()

    def _bind_completion_ref(self, completion_ref: Any) -> None:
        if completion_ref is None:
            raise RuntimeError("Ray stream completion ObjectRef is unavailable")
        with self._lock:
            if self._cancelled or self._generator is None:
                raise RuntimeError("cannot bind completion ownership to an inactive Ray stream")
            if self._completion_ref is not None and self._completion_ref is not completion_ref:
                raise RuntimeError("Ray stream completion ObjectRef changed after binding")
            self._completion_ref = completion_ref

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def driver(self) -> Any:
        return self._driver

    @property
    def lease(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._lease is None else dict(self._lease)

    @property
    def generator(self) -> Any | None:
        with self._lock:
            return self._generator

    def _release_task(self) -> bool | Any:
        with self._lock:
            if self._released or self._task_cleanup_handed_off:
                return True
            if self._cancelled:
                return False
            lease = self._lease_identity
            driver = self._driver
            request_id = self._request_id
            response_future = self._release_response_future
        try:
            if response_future is None:
                response_ref = driver.release_query_task_lease.remote(
                    request_id,
                    str(lease["lease_id"]),
                    str(lease["attempt_id"]),
                )
                response_future = ray_object_ref_future(response_ref)
                with self._lock:
                    self._release_response_future = response_future
            if not response_future.done():
                return response_future
            validate_ray_control_ack(response_future.result(), field="released")
        except BaseException:
            with self._lock:
                if self._release_response_future is response_future:
                    self._release_response_future = None
            raise
        with self._lock:
            self._release_response_future = None
            self._released = True
        return True

    def release_task_operations(self) -> tuple[RayStreamCleanupOperation, ...]:
        return (
            RayStreamCleanupOperation(
                self._release_task,
                retry_on_error=True,
                retry_on_incomplete=True,
            ),
        )

    def _begin_cancellation(self, *, cancel_generator: bool) -> None:
        with self._lock:
            if not self._cancelled:
                self._cancelled = True
                if cancel_generator:
                    self._cancel_generator = self._generator
                self._generator = None

    def _submit_generator_cancel(self) -> bool:
        with self._lock:
            if self._cancel_generator is None:
                return self._generator_cancel_submitted or not self._cancelled
            generator = self._cancel_generator
            ray_module = self._ray
            force = self._execution_backend == "ray_task"
        try:
            ray_module.cancel(generator, force=force, recursive=True)
        except BaseException:
            raise
        with self._lock:
            self._generator_cancel_submitted = True
            if self._cancel_generator is generator:
                self._cancel_generator = None
        return True

    def _handoff_task_completion(self) -> bool | Any:
        with self._lock:
            if self._released or self._task_cleanup_handed_off:
                return True
            lease = self._lease_identity
            driver = self._driver
            request_id = self._request_id
            completion_ref = self._completion_ref
            response_future = self._task_cleanup_response_future
            if completion_ref is None:
                if not self._generator_cancel_submitted:
                    raise RuntimeError(
                        "Ray task lease cannot be handed to query teardown before generator cancellation is submitted"
                    )
                ack_field = "handed_off"
            else:
                ack_field = "scheduled"
        try:
            if response_future is None:
                if completion_ref is None:
                    response_ref = driver.handoff_query_task_lease_to_teardown.remote(
                        request_id,
                        str(lease["lease_id"]),
                        str(lease["attempt_id"]),
                    )
                else:
                    # Ray resolves only top-level ObjectRef arguments. Nesting
                    # the completion ref transfers ownership immediately
                    # without making this RPC wait for the remote task itself.
                    response_ref = driver.release_query_task_lease_after_completion.remote(
                        request_id,
                        str(lease["lease_id"]),
                        str(lease["attempt_id"]),
                        [completion_ref],
                    )
                response_future = ray_object_ref_future(response_ref)
                with self._lock:
                    self._task_cleanup_response_future = response_future
            if not response_future.done():
                return response_future
            validate_ray_control_ack(response_future.result(), field=ack_field)
        except BaseException:
            with self._lock:
                if self._task_cleanup_response_future is response_future:
                    self._task_cleanup_response_future = None
            raise
        with self._lock:
            self._task_cleanup_response_future = None
            self._task_cleanup_handed_off = True
            self._completion_ref = None
        return True

    @property
    def cancellation_started(self) -> bool:
        with self._lock:
            return self._cancelled

    def _generator_cancellation_complete_locked(self) -> bool:
        return not self._cancelled or self._generator_cancel_submitted or self._cancel_generator is None

    @property
    def generator_cancellation_complete(self) -> bool:
        with self._lock:
            return self._generator_cancellation_complete_locked()

    @property
    def terminal_cleanup_complete(self) -> bool:
        with self._lock:
            task_complete = self._released or self._task_cleanup_handed_off
            generator_complete = self._generator_cancellation_complete_locked()
            return task_complete and generator_complete

    def cancel_operations(self) -> tuple[RayStreamCleanupOperation, ...]:
        self._begin_cancellation(cancel_generator=True)
        return (
            RayStreamCleanupOperation(
                self._submit_generator_cancel,
                retry_on_error=True,
                retry_on_incomplete=True,
            ),
            RayStreamCleanupOperation(
                self._handoff_task_completion,
                retry_on_error=True,
                retry_on_incomplete=True,
            ),
        )

    def retire_failed_operations(self) -> tuple[RayStreamCleanupOperation, ...]:
        self._begin_cancellation(cancel_generator=False)
        return (
            RayStreamCleanupOperation(
                self._handoff_task_completion,
                retry_on_error=True,
                retry_on_incomplete=True,
            ),
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
        self._task_request_identity = "" if self._leased is None else self._leased.request_id
        self._generator = self._leased.generator if self._leased is not None else source
        self._completion_ref: Any | None = None
        self._drained = False
        self._retired = False
        self._release_owner: TaskLeaseObjectRefGenerator | None = None
        self._cleanup_generator: Any | None = None
        self._cleanup_completion_ref: Any | None = None
        self._stream_cleanup_active = False
        self._terminal_kind = ""
        self._cancel_generator: Any | None = None
        self._generator_cancelled = False
        self._lifecycle = threading.Condition()
        self._active_reads = 0
        self._active_core_calls = 0
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
        completion_ref = self._generator.completed()
        if completion_ref is None:
            raise RuntimeError("Ray stream completion ObjectRef is unavailable")
        if self._leased is not None:
            self._leased._bind_completion_ref(completion_ref)
        self._completion_ref = completion_ref

    @property
    def task_lease(self) -> dict[str, Any] | None:
        with self._lifecycle:
            leased = self._leased
        return None if leased is None else leased.lease

    @property
    def task_request_id(self) -> str:
        return self._task_request_identity

    @property
    def driver(self) -> Any | None:
        with self._lifecycle:
            leased = self._leased
        return None if leased is None else leased.driver

    @property
    def waitable(self) -> Any:
        """Return the generator used for non-consuming readiness probes."""
        with self._lifecycle:
            if self._retired or self._generator is None:
                raise RuntimeError("Ray stream has no active ObjectRefGenerator")
            return self._generator

    async def read_next_ref_async(self) -> Any:
        with self._lifecycle:
            generator = self._generator
            if self._retired or generator is None:
                raise RuntimeError("Ray stream has not acquired its task lease")
            if self._drained:
                raise StopAsyncIteration
            if self._active_reads:
                raise RuntimeError("Ray ObjectRefGenerator does not support concurrent reads")
            self._active_reads += 1
        try:
            ref = await generator.__anext__()
            if hasattr(ref, "is_nil") and ref.is_nil():
                raise RuntimeError("Ray generator returned a nil ObjectRef")
            return ref
        finally:
            with self._lifecycle:
                self._active_reads -= 1
                if self._active_reads < 0:
                    raise RuntimeError("Ray stream active read count underflow")
                self._lifecycle.notify_all()

    @property
    def completion_ref(self) -> Any:
        with self._lifecycle:
            completion_ref = self._completion_ref
        if completion_ref is None:
            raise RuntimeError("Ray stream completion ObjectRef is unavailable")
        return completion_ref

    def stream_finished(self) -> bool:
        with self._lifecycle:
            generator = self._generator
            completion_ref = self._completion_ref
            drained = self._drained
            if generator is None or completion_ref is None:
                return drained
            self._active_core_calls += 1
        try:
            return bool(generator.worker.core_worker.is_object_ref_stream_finished(completion_ref))
        finally:
            with self._lifecycle:
                self._active_core_calls -= 1
                if self._active_core_calls < 0:
                    raise RuntimeError("Ray stream active CoreWorker call count underflow")
                self._lifecycle.notify_all()

    def is_terminal_ref(self, ref: Any) -> bool:
        with self._lifecycle:
            return self._generator is not None and ref == self._completion_ref

    def mark_drained(self) -> None:
        with self._lifecycle:
            self._drained = True

    @staticmethod
    def _delete_object_ref_stream(generator: Any, completion_ref: Any) -> None:
        worker = generator.worker
        core_worker = worker.core_worker
        core_worker.async_delete_object_ref_stream(completion_ref)
        # Ray's ObjectRefGenerator.__del__ unconditionally deletes through
        # ``generator.worker``. Detach only after the explicit delete succeeds;
        # a synchronous failure retains the worker for a bounded retry.
        generator.worker = None

    def _run_pending_stream_cleanup(self) -> bool:
        with self._lifecycle:
            if self._stream_cleanup_active:
                return False
            generator = self._cleanup_generator
            completion_ref = self._cleanup_completion_ref
            if generator is None or completion_ref is None:
                return True
            release_owner = self._release_owner
            if self._terminal_kind == "cancel":
                if release_owner is not None and not release_owner.generator_cancellation_complete:
                    return False
                if release_owner is None and not self._generator_cancelled and self._cancel_generator is not None:
                    return False
            if self._active_reads or self._active_core_calls:
                # The event loop retries after the in-flight call exits;
                # retirement prevents any new call from starting.
                return False
            self._stream_cleanup_active = True
        try:
            self._delete_object_ref_stream(generator, completion_ref)
        except BaseException:
            with self._lifecycle:
                self._stream_cleanup_active = False
                self._lifecycle.notify_all()
            raise
        with self._lifecycle:
            self._stream_cleanup_active = False
            if self._cleanup_generator is generator and self._cleanup_completion_ref is completion_ref:
                self._cleanup_generator = None
                self._cleanup_completion_ref = None
            self._lifecycle.notify_all()
        return True

    def _run_unleased_generator_cancel(self) -> bool:
        with self._lifecycle:
            generator = self._cancel_generator
            if generator is None:
                return True
        try:
            self._ray.cancel(generator, force=True, recursive=True)
        except BaseException:
            raise
        with self._lifecycle:
            self._generator_cancelled = True
            if self._cancel_generator is generator:
                self._cancel_generator = None
            self._lifecycle.notify_all()
        return True

    def _drop_release_owner_if_complete(self, leased: TaskLeaseObjectRefGenerator) -> None:
        with self._lifecycle:
            if self._release_owner is leased and leased.terminal_cleanup_complete:
                self._release_owner = None
            self._lifecycle.notify_all()

    def _leased_operation(
        self,
        leased: TaskLeaseObjectRefGenerator,
        operation: Callable[[], Any],
    ) -> Any:
        result = operation()
        self._drop_release_owner_if_complete(leased)
        return result

    def retire_operations(self) -> tuple[RayStreamCleanupOperation, ...]:
        """Build deterministic cleanup for a fully drained generator stream."""
        return self._prepare_terminal_operations("retire")

    def cancel_operations(self) -> tuple[RayStreamCleanupOperation, ...]:
        return self._prepare_terminal_operations("cancel")

    def retire_failed_operations(self) -> tuple[RayStreamCleanupOperation, ...]:
        return self._prepare_terminal_operations("failed")

    def _prepare_terminal_operations(
        self,
        requested_kind: str,
    ) -> tuple[RayStreamCleanupOperation, ...]:
        if requested_kind not in {"retire", "failed", "cancel"}:
            raise ValueError(f"invalid Ray stream terminal kind: {requested_kind}")
        with self._lifecycle:
            if not self._retired:
                self._retired = True
                self._terminal_kind = requested_kind
                leased = self._leased
                self._release_owner = leased
                self._cleanup_generator = self._generator
                self._cleanup_completion_ref = self._completion_ref
                if requested_kind == "cancel" and leased is None:
                    self._cancel_generator = self._generator
                self._source = None
                self._leased = None
                self._generator = None
                self._completion_ref = None
            else:
                leased = self._release_owner
                # Never escalate normal completion into force-cancelling a Ray
                # task whose terminal reply may already be in flight.
                if self._terminal_kind == "retire" and requested_kind != "retire":
                    self._terminal_kind = "failed"
            terminal_kind = self._terminal_kind

        leased_operations: list[RayStreamCleanupOperation] = []
        if leased is not None:
            if terminal_kind == "cancel":
                source_operations = leased.cancel_operations()
            elif terminal_kind == "failed":
                source_operations = leased.retire_failed_operations()
            elif leased.cancellation_started:
                source_operations = leased.cancel_operations()
            else:
                source_operations = leased.release_task_operations()
            leased.retire_local_state()
            leased_operations = [
                RayStreamCleanupOperation(
                    partial(self._leased_operation, leased, operation.operation),
                    retry_on_error=operation.retry_on_error,
                    retry_on_incomplete=operation.retry_on_incomplete,
                )
                for operation in source_operations
            ]

        stream_operation = RayStreamCleanupOperation(
            self._run_pending_stream_cleanup,
            retry_on_error=True,
            retry_on_incomplete=True,
        )
        operations: list[RayStreamCleanupOperation] = [*leased_operations]
        if terminal_kind == "cancel" and leased is None:
            operations.append(
                RayStreamCleanupOperation(
                    self._run_unleased_generator_cancel,
                    retry_on_error=True,
                    retry_on_incomplete=True,
                )
            )
        operations.append(stream_operation)
        return tuple(operations)

    @property
    def terminal_cleanup_complete(self) -> bool:
        with self._lifecycle:
            stream_complete = self._cleanup_generator is None and not self._stream_cleanup_active
            release_complete = self._release_owner is None
            generator_complete = (
                self._terminal_kind != "cancel" or self._generator_cancelled or self._cancel_generator is None
            )
            return self._retired and stream_complete and release_complete and generator_complete


__all__ = [
    "RayStreamAdapter",
    "RayStreamCleanupOperation",
    "TaskLeaseObjectRefGenerator",
    "ray_object_ref_future",
    "validate_ray_stream_contract",
    "validate_ray_control_ack",
]
