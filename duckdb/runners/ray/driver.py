# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import math
import os
import sys
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Mapping
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from duckdb._ray_cxx import require_ray_cxx_attr
from duckdb._ray_errors import restore_remote_ray_exception
from duckdb._ray_progress_env import configure_ray_progress_logging_defaults

RayResultPartitionRef = require_ray_cxx_attr(
    "RayResultPartitionRef",
    hint="Ensure the C++ ray extension is built and importable in the driver process.",
)
RayTaskResult = require_ray_cxx_attr(
    "RayTaskResult",
    hint="Ensure the C++ ray extension is built and importable in the driver process.",
)

configure_ray_progress_logging_defaults()

from duckdb.event_loop import set_event_loop
from duckdb.runners.copy_outcome import CopyOutcomeUnknownError
from duckdb.runners.fte import (
    FteTaskAttemptId,
    FteTaskState,
    fte_status_wait_timeout_s,
    validate_fte_status_identity,
)
from duckdb.runners.fte.fte_failures import _normalize_failure_payload, _safe_failure_message
from duckdb.runners.progress import ProgressRenderer, progress_enabled
from duckdb.runners.ray.admission_ledger import BoundedReplayMap
from duckdb.runners.ray.ray_env import (
    collect_vane_env_overrides,
    reject_node_local_ray_runtime_env,
    scrub_shared_runtime_session_env,
)
from duckdb.runners.ray.safe_get import QueryDeadlineExceeded, resolve_object_refs_blocking
from duckdb.runners.ray.worker import WorkerTaskMetadata

_LEASE_REQUEST_REPLAY_CAPACITY = 65_536
_SESSION_CLOSE_REPLAY_CAPACITY = 65_536
_CLIENT_DETACH_REPLAY_CAPACITY = 65_536
_COPY_OPERATION_REPLAY_CAPACITY = 1_024
_DEFAULT_PROGRESS_TOPOLOGY_INIT_TIMEOUT_S = 60.0
_DEFAULT_COPY_RECONCILIATION_TIMEOUT_S = 30.0
_RUNTIME_ATTACH_TIMEOUT_S = 300.0
_RUNTIME_ATTACH_RETRY_INTERVAL_S = 0.05
_RUNTIME_ATTACH_CLEANUP_ATTEMPTS = 3
_RUNTIME_ATTACH_CLEANUP_TIMEOUT_S = 300.0
_DEFAULT_CLIENT_LEASE_TIMEOUT_S = 60.0
_DEFAULT_CLIENT_HEARTBEAT_INTERVAL_S = 10.0
_TASK_COMPLETION_CLEANUP_RETRY_MIN_S = 0.01
_TASK_COMPLETION_CLEANUP_RETRY_MAX_S = 1.0
RAY_QUERY_RUNTIME_ACTOR_NAMESPACE = "vane"
RAY_QUERY_RUNTIME_ACTOR_NAME_PREFIX = "vane-query-runtime-"

if TYPE_CHECKING:
    from collections.abc import Iterator

    import duckdb
    from duckdb.runners.ray.partition_metadata import RayMaterializedResult

import ray


class QueryTeardownOwnershipError(RuntimeError):
    """Teardown stopped before owner release and must remain retryable."""


class QueryFteRegistryQuiesceError(QueryTeardownOwnershipError):
    """Local FTE threads still own query state; teardown must remain retryable."""


class RayRuntimeShuttingDownError(RuntimeError):
    """The named job runtime cannot accept a new owner while it exits."""


class VaneSessionOpenRejectedError(RuntimeError):
    """A session-open request was rejected before the actor recorded it."""


class QueryExecutionCleanupError(RuntimeError):
    """A primary query failure accompanied by one or more cleanup failures."""

    def __init__(
        self,
        message: str,
        *,
        primary_error: BaseException | None = None,
        cleanup_errors: tuple[BaseException, ...] = (),
    ) -> None:
        super().__init__(str(message))
        self.primary_error = primary_error
        self.cleanup_errors = tuple(cleanup_errors)

    @classmethod
    def from_errors(
        cls,
        context: str,
        primary_error: BaseException,
        cleanup_errors: list[BaseException] | tuple[BaseException, ...],
    ) -> QueryExecutionCleanupError:
        cleanup = tuple(cleanup_errors)
        if not cleanup:
            raise ValueError("QueryExecutionCleanupError requires at least one cleanup error")
        message = f"{context}: primary={type(primary_error).__name__}: {primary_error}; cleanup=" + "; ".join(
            f"{type(error).__name__}: {error}" for error in cleanup
        )
        return cls(
            message,
            primary_error=primary_error,
            cleanup_errors=cleanup,
        )


@dataclass(frozen=True, slots=True)
class _TaskTerminalControl:
    """Bounded payload-replay identity for one terminal task request."""

    identity_fingerprint: bytes
    query_fingerprint: bytes
    status: str
    blocked_reason: str


@dataclass(frozen=True, slots=True)
class _OutputTerminalControl:
    """Bounded payload-replay identity for one terminal output request."""

    identity_fingerprint: bytes
    query_fingerprint: bytes
    status: str
    blocked_reason: str


def _client_owner_lease_settings() -> tuple[float, float]:
    lease_timeout = float(
        os.environ.get(
            "VANE_RAY_CLIENT_LEASE_TIMEOUT_S",
            str(_DEFAULT_CLIENT_LEASE_TIMEOUT_S),
        )
    )
    heartbeat_interval = float(
        os.environ.get(
            "VANE_RAY_CLIENT_HEARTBEAT_INTERVAL_S",
            str(_DEFAULT_CLIENT_HEARTBEAT_INTERVAL_S),
        )
    )
    if not math.isfinite(lease_timeout) or lease_timeout <= 0:
        raise ValueError("VANE_RAY_CLIENT_LEASE_TIMEOUT_S must be finite and > 0")
    if not math.isfinite(heartbeat_interval) or heartbeat_interval <= 0 or heartbeat_interval > lease_timeout / 3.0:
        raise ValueError(
            "VANE_RAY_CLIENT_HEARTBEAT_INTERVAL_S must be finite, > 0, and no greater than one third of "
            "VANE_RAY_CLIENT_LEASE_TIMEOUT_S"
        )
    return lease_timeout, heartbeat_interval


def _copy_reconciliation_timeout_s() -> float:
    value = float(
        os.environ.get(
            "VANE_RAY_COPY_RECONCILIATION_TIMEOUT_S",
            str(_DEFAULT_COPY_RECONCILIATION_TIMEOUT_S),
        )
    )
    if not math.isfinite(value) or value <= 0:
        raise ValueError("VANE_RAY_COPY_RECONCILIATION_TIMEOUT_S must be finite and > 0")
    return value


def _runtime_error_candidates(error: BaseException) -> list[BaseException]:
    candidates: list[BaseException] = [error]
    candidate_ids = {id(error)}
    candidate_index = 0
    while candidate_index < len(candidates) and candidate_index < 16:
        candidate = candidates[candidate_index]
        candidate_index += 1
        converted = getattr(candidate, "as_instanceof_cause", None)
        if callable(converted):
            try:
                converted_cause = converted()
            except BaseException:
                converted_cause = None
            if isinstance(converted_cause, BaseException) and id(converted_cause) not in candidate_ids:
                candidate_ids.add(id(converted_cause))
                candidates.append(converted_cause)
        for attribute in ("cause", "__cause__"):
            cause = getattr(candidate, attribute, None)
            if isinstance(cause, BaseException) and id(cause) not in candidate_ids:
                candidate_ids.add(id(cause))
                candidates.append(cause)
    return candidates


def _raise_copy_outcome_unknown(
    operation_id: str,
    operation_error: BaseException,
) -> NoReturn:
    unknown_error = next(
        (
            candidate
            for candidate in _runtime_error_candidates(operation_error)
            if isinstance(candidate, CopyOutcomeUnknownError)
        ),
        None,
    )
    if unknown_error is not None:
        raise unknown_error
    raise CopyOutcomeUnknownError(operation_id) from operation_error


def _runtime_error_contains(error: BaseException, error_type: type[BaseException]) -> bool:
    return any(isinstance(candidate, error_type) for candidate in _runtime_error_candidates(error))


def _runtime_error_is_wait_timeout(error: BaseException) -> bool:
    candidates = _runtime_error_candidates(error)
    ray_task_error_type = getattr(ray.exceptions, "RayTaskError", None)
    if isinstance(ray_task_error_type, type) and any(
        isinstance(candidate, ray_task_error_type) for candidate in candidates
    ):
        # A completed actor RPC can itself raise TimeoutError. That mutation
        # failed terminally and must be retried with a new ObjectRef rather than
        # mistaken for an ambiguous local wait on the existing ObjectRef.
        return False
    if any(getattr(candidate, "remote_exception_type", None) for candidate in candidates):
        return False
    return any(
        isinstance(candidate, (TimeoutError, FutureTimeoutError))
        or type(candidate).__name__ in {"GetTimeoutError", "FutureTimeoutError"}
        for candidate in candidates
    )


def _runtime_actor_is_unavailable(error: BaseException) -> bool:
    actor_error_type = getattr(ray.exceptions, "RayActorError", None)
    return isinstance(actor_error_type, type) and any(
        isinstance(candidate, actor_error_type) for candidate in _runtime_error_candidates(error)
    )


def _runtime_actor_is_being_replaced(error: BaseException) -> bool:
    candidates = _runtime_error_candidates(error)
    if any(isinstance(candidate, RayRuntimeShuttingDownError) for candidate in candidates):
        return True
    actor_error_type = getattr(ray.exceptions, "RayActorError", None)
    return isinstance(actor_error_type, type) and any(
        isinstance(candidate, actor_error_type) and not bool(getattr(candidate, "actor_init_failed", False))
        for candidate in candidates
    )


@dataclass(frozen=True)
class CopyPlanOutcome:
    """Internal actor-to-client COPY result captured before query teardown."""

    result: dict[str, Any]
    final_progress_snapshot: dict[str, Any] | None
    operation_id: str = ""
    write_state: Literal["committed"] = "committed"
    cleanup_state: Literal["complete", "pending"] = "complete"
    cleanup_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CopyPlanRecovery:
    """Read-only reconciliation response for one submitted COPY operation."""

    operation_id: str
    outcome: CopyPlanOutcome | None = None
    error: BaseException | None = None


@dataclass(frozen=True)
class _CopyOperationInFlight:
    owner_id: str
    session_id: str
    task: asyncio.Task[CopyPlanOutcome]


@dataclass(frozen=True)
class _CopyOperationTerminal:
    owner_id: str
    session_id: str
    outcome: CopyPlanOutcome | None = None
    error: BaseException | None = None


@dataclass(frozen=True)
class _PreparedQueryResourceRegistration:
    """Immutable registration input prepared away from the actor owner loop."""

    graph: Any
    node_capacities: tuple[Any, ...]
    capacity_snapshot_started_at: float


@dataclass(frozen=True)
class _DeferredQueryAllocationTeardown:
    """Generation-fenced fragment teardown discovered during a local commit."""

    query_id: str
    generation: int
    reason: str


class _PlanStartupOwnership:
    """Hand registered resources to exactly one startup or cancellation path."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner = "caller"

    def claim_for_worker(self) -> bool:
        with self._lock:
            if self._owner != "caller":
                return False
            self._owner = "worker"
            return True

    def cancel_before_worker_claim(self) -> bool:
        with self._lock:
            if self._owner != "caller":
                return False
            self._owner = "cancelled"
            return True


@dataclass
class _DriverSession:
    owner_id: str
    config: dict[str, str]
    connection: Any
    s3_config: dict[str, str]
    plan_ids: set[str] = field(default_factory=set)
    lock: threading.RLock = field(default_factory=threading.RLock)
    operation_lock: threading.Lock = field(default_factory=threading.Lock)
    closing: bool = False
    close_in_progress: bool = False
    closed: bool = False
    active_operations: int = 0
    active_operation_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    condition: threading.Condition = field(init=False)

    def __post_init__(self) -> None:
        self.condition = threading.Condition(self.lock)


@dataclass
class _ClientOwnerLease:
    lease_token: str
    expires_at: float
    state: str = "active"
    cleanup_attempts: int = 0
    next_cleanup_at: float = 0.0
    last_cleanup_error: str = ""


@dataclass
class _AdmissionLoopFence:
    completed: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


def _normalize_session_config(config: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(config, Mapping):
        raise TypeError("Vane session config must be a mapping")
    normalized: dict[str, str] = {}
    for raw_key, raw_value in config.items():
        key = str(raw_key)
        if not key:
            raise ValueError("Vane session config keys must not be empty")
        if raw_value is None:
            raise ValueError(f"Vane session config value must not be None: {key}")
        normalized[key] = str(raw_value)
    return normalized


def _ray_runtime_job_id(runtime_context: Any) -> str:
    job_id_obj = runtime_context.get_job_id()
    job_id_hex = getattr(job_id_obj, "hex", None)
    job_id = str(job_id_hex() if callable(job_id_hex) else job_id_obj).strip()
    if not job_id:
        raise RuntimeError("Ray runtime context returned an empty job identity")
    return job_id


async def _to_thread_with_owned_side_effects(callback: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Do not expose cancellation until a thread-owned mutation has finished."""
    thread_task = asyncio.create_task(asyncio.to_thread(callback, *args, **kwargs))
    cancellation: asyncio.CancelledError | None = None
    while not thread_task.done():
        try:
            await asyncio.shield(thread_task)
        except asyncio.CancelledError as error:
            cancellation = error
    result = thread_task.result()
    if cancellation is not None:
        raise cancellation
    return result


def _progress_topology_init_timeout_s() -> float:
    value = float(
        os.environ.get(
            "VANE_PROGRESS_TOPOLOGY_INIT_TIMEOUT_S",
            str(_DEFAULT_PROGRESS_TOPOLOGY_INIT_TIMEOUT_S),
        )
    )
    if not math.isfinite(value) or value <= 0:
        raise ValueError("VANE_PROGRESS_TOPOLOGY_INIT_TIMEOUT_S must be finite and > 0")
    return value


def _log_resource_debug(event: str, **fields: Any) -> None:
    value = os.getenv("DUCKDB_DISTRIBUTED_DEBUG", "").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return
    parts = [
        f"event={event}",
        f"pid={os.getpid()}",
        f"t={time.monotonic():.6f}",
    ]
    parts.extend(f"{key}={value}" for key, value in fields.items())
    print("[vane-query-resource-driver] " + " ".join(parts), file=sys.stderr, flush=True)


def _log_copy_result_debug(query_id: str, result: Any) -> None:
    value = os.getenv("DUCKDB_DISTRIBUTED_DEBUG", "").strip().lower()
    if value in ("", "0", "false", "no", "off") or not isinstance(result, dict):
        return
    files = result.get("files")
    file_count = len(files) if isinstance(files, list) else -1
    print(
        "[vane-copy-result] "
        f"query_id={query_id} "
        f"rows_copied={int(result.get('rows_copied') or 0)} "
        f"file_count={file_count} "
        f"selected_file_count={int(result.get('copy_selected_file_count') or 0)} "
        f"duplicate_file_count={int(result.get('copy_duplicate_file_count') or 0)}",
        file=sys.stderr,
        flush=True,
    )


def _safe_remote_error_message(exc: BaseException) -> str:
    cause = getattr(exc, "cause", None)
    if cause is not None:
        args = getattr(cause, "args", None)
        if isinstance(args, tuple):
            for arg in args:
                if isinstance(arg, str) and arg.strip():
                    return arg.strip()

    traceback_str = getattr(exc, "traceback_str", None)
    if isinstance(traceback_str, str) and traceback_str.strip():
        lines = [line.strip() for line in traceback_str.splitlines() if line.strip()]
        if lines:
            return lines[-1]

    args = getattr(exc, "args", None)
    if isinstance(args, tuple):
        for arg in args:
            if isinstance(arg, str) and arg.strip():
                return arg.strip()

    return f"{type(exc).__name__} from remote Ray task"


def _ray_progress_snapshot_or_none(
    runner: Any,
    owner_id: str,
    session_id: str,
    plan_id: Any,
    started_at: float,
) -> dict[str, Any] | None:
    # Short hard timeout: progress must never stall query execution or pin the
    # driver actor long enough to block UDF lease admission/release.
    try:
        return resolve_object_refs_blocking(
            runner.progress_snapshot.remote(owner_id, session_id, plan_id, started_at),
            timeout=0.1,
        )
    except Exception:
        return None


class _RayProgressSession:
    """Drive best-effort progress updates from synchronous Ray waits."""

    def __init__(
        self,
        runner: Any,
        owner_id: str,
        session_id: str,
        plan_id: Any,
        started_at: float,
    ) -> None:
        self._renderer = None
        self._renderer_failed = False
        self._finished = False
        try:
            if progress_enabled():
                self._renderer = ProgressRenderer(
                    lambda: _ray_progress_snapshot_or_none(
                        runner,
                        owner_id,
                        session_id,
                        plan_id,
                        started_at,
                    )
                )
        except Exception:
            self._renderer_failed = True

    def _update(self) -> None:
        if self._renderer is None or self._renderer_failed:
            return
        try:
            self._renderer.update()
        except Exception:
            self._renderer_failed = True

    def resolve(self, object_ref: Any) -> Any:
        if self._renderer is None or self._renderer_failed:
            return resolve_object_refs_blocking(object_ref)
        return resolve_object_refs_blocking(
            object_ref,
            on_wait=self._update,
            wait_interval_s=self._renderer.interval_s,
        )

    def finish(
        self,
        *,
        final_state: str | None = None,
        final_snapshot: dict[str, Any] | None = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        if self._renderer is None or self._renderer_failed:
            return
        try:
            self._renderer.finish(
                final_state=final_state,
                final_snapshot=final_snapshot,
            )
        except Exception:
            self._renderer_failed = True


_GLOBAL_EVENT_LOOP: asyncio.AbstractEventLoop | None = None
_GLOBAL_EVENT_LOOP_LOCK = threading.Lock()
_GLOBAL_EVENT_LOOP_THREAD: threading.Thread | None = None


def _set_global_event_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    global _GLOBAL_EVENT_LOOP
    if loop is not None:
        _GLOBAL_EVENT_LOOP = loop


def _get_global_event_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return _GLOBAL_EVENT_LOOP


def _ensure_background_event_loop() -> asyncio.AbstractEventLoop:
    global _GLOBAL_EVENT_LOOP_THREAD

    loop = _get_global_event_loop()
    if loop is not None and loop.is_running() and not loop.is_closed():
        return loop

    with _GLOBAL_EVENT_LOOP_LOCK:
        loop = _GLOBAL_EVENT_LOOP
        if loop is not None and loop.is_running() and not loop.is_closed():
            return loop

        ready = threading.Event()

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            set_event_loop(loop)
            _set_global_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(
            target=_run_loop,
            name="duckdb-ray-driver-loop",
            daemon=True,
        )
        thread.start()
        ready.wait(timeout=5.0)
        _GLOBAL_EVENT_LOOP_THREAD = thread

        loop = _GLOBAL_EVENT_LOOP
        if loop is None or not loop.is_running() or loop.is_closed():
            raise RuntimeError("failed to start background asyncio event loop")
        return loop


def shutdown_background_event_loop(timeout_s: float = 5.0) -> None:
    """Stop the driver helper loop so Ray can be reinitialized cleanly.

    Test suites and notebooks commonly call ``ray.shutdown()`` and then start a
    new local Ray session in the same Python process.  Ray ObjectRef awaiters
    scheduled on the old helper loop can otherwise survive across sessions and
    leave later distributed tasks waiting on stale state.
    """
    global _GLOBAL_EVENT_LOOP
    global _GLOBAL_EVENT_LOOP_THREAD

    with _GLOBAL_EVENT_LOOP_LOCK:
        loop = _GLOBAL_EVENT_LOOP
        thread = _GLOBAL_EVENT_LOOP_THREAD
        _GLOBAL_EVENT_LOOP = None
        _GLOBAL_EVENT_LOOP_THREAD = None

    if loop is None or loop.is_closed():
        return

    if loop.is_running():
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            return

    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=timeout_s)

    if not loop.is_running() and not loop.is_closed():
        try:
            loop.close()
        except RuntimeError:
            pass


def _collect_vane_env_overrides() -> dict[str, str]:
    return collect_vane_env_overrides()


def _apply_duckdb_thread_setting(conn: Any) -> None:
    raw_value = os.environ.get("VANE_DUCKDB_THREADS", "").strip()
    if not raw_value:
        return
    threads = int(raw_value)
    if threads <= 0:
        raise ValueError("VANE_DUCKDB_THREADS must be positive")
    conn.execute(f"SET threads={threads}")


@dataclass
class FteWorkerTaskHandle:
    """Handle for worker-side FTE task attempts.

    FTE worker handles use versioned long-poll status waits to avoid polling
    worker status in a tight loop.
    """

    task_id: FteTaskAttemptId | str | dict[str, Any]
    worker_handle: Any
    status_wait_timeout_s: float = 0.0
    task_context_info: dict[str, Any] | None = None
    query_task_lease: dict[str, Any] | None = None
    _result: Any = None
    _error: Exception | None = None
    _future: Any = None
    task: Any = None
    _is_done: bool = False
    _last_status_version: int = -1
    _terminal_recorded_task_id: str | None = None
    _worker_failure_publish_started: bool = False
    _acked: bool = False
    _released: bool = False
    _unacked_output_owners: list[Any] = field(default_factory=list)
    _lifecycle_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.task_id = FteTaskAttemptId.coerce(self.task_id)
        self.worker_id = getattr(self.worker_handle, "worker_id", None)
        self.task_context_info = dict(self.task_context_info or {})
        self.query_task_lease = dict(self.query_task_lease or {})
        for method_name in (
            "fte_wait_task_status",
            "fte_cancel_task",
            "mark_fte_worker_failed",
            "record_fte_task_terminal",
            "finish_fte_task_with_outputs",
            "enqueue_fte_ack_task_result",
            "enqueue_fte_release_task_result",
        ):
            if not callable(getattr(self.worker_handle, method_name, None)):
                raise RuntimeError(f"FTE worker handle must provide {method_name}")
        if not str(self.worker_id or ""):
            raise RuntimeError("FTE worker handle must provide non-empty worker_id")
        if self.status_wait_timeout_s <= 0:
            self.status_wait_timeout_s = fte_status_wait_timeout_s()

    def _attempt_id(self) -> FteTaskAttemptId:
        return FteTaskAttemptId.coerce(self.task_id)

    def _terminal_state_is_set(self) -> bool:
        """Reload terminal state after another thread or awaited RPC may mutate it."""
        return self._is_done

    def _ensure_started(self) -> None:
        with self._lifecycle_lock:
            if self._is_done or self._result is not None or self._error is not None:
                return
            if self._future is not None or self.task is not None:
                return
            loop = _get_global_event_loop()
            if loop is None or not loop.is_running() or loop.is_closed():
                loop = _ensure_background_event_loop()
            self._future = asyncio.run_coroutine_threadsafe(
                self._watch_status(),
                loop,
            )

    async def _wait_task_status(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            min_version = self._last_status_version
        return await asyncio.to_thread(
            self.worker_handle.fte_wait_task_status,
            self._attempt_id().to_dict(),
            min_version,
            self.status_wait_timeout_s,
        )

    @staticmethod
    def _is_soft_status_wait_timeout(exc: BaseException) -> bool:
        message = _safe_failure_message(exc)
        if issubclass(type(exc), QueryDeadlineExceeded) or "query deadline expired" in message.lower():
            return False
        if issubclass(type(exc), TimeoutError):
            return True
        name = type(exc).__name__
        if name in {"TimeoutError", "GetTimeoutError"}:
            return True
        return "did not complete within" in message or "timed out" in message.lower()

    async def _watch_status(self) -> Any:
        while True:
            with self._lifecycle_lock:
                if self._is_done:
                    if self._error is not None:
                        raise self._error
                    if self._result is None:
                        raise RuntimeError("FTE task completed without result")
                    return self._result
            try:
                status = await self._wait_task_status()
            except Exception as exc:
                # Soft timeouts are expected when workers are busy or no status
                # version advanced; never mark the worker failed for that.
                if self._is_soft_status_wait_timeout(exc):
                    continue
                with self._lifecycle_lock:
                    if self._terminal_state_is_set():
                        continue
                terminal_error = await self._finalize_status_watch_failure(
                    exc,
                    failure_kind="status wait failed",
                )
                if terminal_error is None:
                    continue
                raise terminal_error
            apply_failure = await asyncio.to_thread(self._apply_status, status)
            if apply_failure is not None:
                terminal_error = await self._finalize_status_watch_failure(
                    apply_failure,
                    failure_kind="status protocol failed",
                )
                if terminal_error is None:
                    continue
                raise terminal_error

    async def _publish_worker_failure(
        self,
        exc: BaseException,
        *,
        failure_kind: str,
    ) -> None:
        with self._lifecycle_lock:
            if self._worker_failure_publish_started:
                return
            self._worker_failure_publish_started = True
        contextual_error = RuntimeError(f"{failure_kind} for {self.task_id}: {_safe_failure_message(exc)}")
        contextual_error.__cause__ = exc
        await asyncio.to_thread(
            self.worker_handle.mark_fte_worker_failed,
            self.worker_id,
            contextual_error,
        )

    async def _finalize_status_watch_failure(
        self,
        exc: BaseException,
        *,
        failure_kind: str,
    ) -> Exception | None:
        with self._lifecycle_lock:
            if self._is_done:
                return None
        cleanup_errors: list[str] = []
        try:
            await self._publish_worker_failure(
                exc,
                failure_kind=failure_kind,
            )
        except Exception as cleanup_exc:
            cleanup_errors.append(f"worker failure publication failed: {_safe_failure_message(cleanup_exc)}")
        try:
            await asyncio.to_thread(self._record_fte_task_terminal)
        except Exception as cleanup_exc:
            cleanup_errors.append(f"terminal record failed: {_safe_failure_message(cleanup_exc)}")
        message = _safe_failure_message(exc)
        if cleanup_errors:
            message += "; cleanup also failed: " + "; ".join(cleanup_errors)
        terminal_error = exc if not cleanup_errors and isinstance(exc, Exception) else RuntimeError(message)
        with self._lifecycle_lock:
            if self._terminal_state_is_set():
                return None
            self._error = terminal_error
            self._result = None
            self._is_done = True
        return terminal_error

    def done(self) -> bool:
        with self._lifecycle_lock:
            if self._is_done:
                return True
            if self._result is not None or self._error is not None:
                self._is_done = True
                return True
            self._ensure_started()
            completed = None
            if self._future is not None and self._future.done():
                completed = self._future
            elif self.task is not None and self.task.done():
                completed = self.task
            if completed is not None:
                try:
                    self._result = completed.result()
                except Exception as exc:
                    self._error = exc
                self._is_done = True
            return self._is_done

    def get_result_sync(self) -> Any:
        with self._lifecycle_lock:
            if not self.done():
                raise RuntimeError("FTE task result not ready")
            if self._error is not None:
                raise self._error
            if self._result is None:
                raise RuntimeError("FTE task completed without result")
            self.ack()
            return self._result

    async def get_result(self) -> Any:
        with self._lifecycle_lock:
            self._ensure_started()
            future = self._future
        if future is None:
            return self.get_result_sync()
        result = await asyncio.wrap_future(future)
        self.ack()
        return result

    def cancel(self) -> None:
        terminal_error: Exception | None = None
        with self._lifecycle_lock:
            if self._is_done:
                return
            errors: list[str] = []
            try:
                self.worker_handle.fte_cancel_task(self._attempt_id().to_dict())
            except Exception as exc:
                errors.append(f"cancel failed: {exc}")
            try:
                self.release_result_payload()
            except Exception as exc:
                errors.append(f"result release failed: {exc}")
            self._acked = True
            try:
                self._record_fte_task_terminal()
            except Exception as exc:
                errors.append(f"terminal record failed: {exc}")
            if errors:
                terminal_error = RuntimeError(f"failed to cancel FTE task {self.task_id}: " + "; ".join(errors))
                self._error = terminal_error
                self._result = None
            else:
                self._result = RayTaskResult.success([], [], None)
                self._error = None
            self._is_done = True
        if terminal_error is not None:
            raise terminal_error

    def _record_fte_task_terminal(self) -> None:
        with self._lifecycle_lock:
            task_key = str(self.task_id)
            if self._terminal_recorded_task_id == task_key:
                return
            self.worker_handle.record_fte_task_terminal(self.task_id)
            self._terminal_recorded_task_id = task_key

    def ack(self) -> None:
        with self._lifecycle_lock:
            if self._acked:
                return
            self.worker_handle.enqueue_fte_ack_task_result(self._attempt_id().to_dict())
            self._acked = True

    def release_result_payload(self) -> None:
        with self._lifecycle_lock:
            if self._released:
                return
            self.worker_handle.enqueue_fte_release_task_result(self._attempt_id().to_dict())
            self._released = True

    def _apply_status(self, status: dict[str, Any]) -> Exception | None:
        with self._lifecycle_lock:
            if self._is_done:
                return None
            try:
                self._apply_status_locked(status)
            except Exception as exc:
                prior_error = self._error
                cleanup_errors = self._discard_unacked_finished_result()
                message = f"failed to apply FTE task status for {self.task_id}: {exc}"
                if prior_error is not None:
                    message = f"{prior_error}; {message}"
                if cleanup_errors:
                    message += "; cleanup also failed: " + "; ".join(cleanup_errors)
                self._result = None
                return RuntimeError(message)
        return None

    def _apply_status_locked(self, status: dict[str, Any]) -> None:
        validate_fte_status_identity(status, self.task_id)
        version = status.get("version")
        if version is not None:
            self._last_status_version = max(self._last_status_version, int(version))
        state = self._state_from_status(status)
        if state == FteTaskState.FINISHED:
            # The scheduler's FteAttemptStatusWatcher is the single owner of
            # attempt selection.  A result handle only adopts the immutable
            # terminal payload; selected-attempt filtering happens after the
            # query-level selection set is committed.  Publishing FINISHED a
            # second time here used to race that watcher and silently discard
            # a winning payload when the scheduler drain was already active.
            raw_result = status.get("result")
            if isinstance(raw_result, dict):
                raw_result = raw_result.get("result", raw_result)
            output_stats = self._output_stats_from_status(status)
            if raw_result is None and output_stats is None and self._requires_final_output_stats():
                self._error = RuntimeError(
                    f"FTE task {self.task_id} FINISHED without final task info or spooling output stats"
                )
                self._record_fte_task_terminal()
                self._is_done = True
                return
            if raw_result is not None:
                try:
                    self._result = self._normalize_raw_result(raw_result)
                except Exception as exc:
                    cleanup_errors = self._discard_unacked_finished_result()
                    message = f"failed to adopt FTE task result for {self.task_id}: {exc}"
                    if cleanup_errors:
                        message += "; cleanup also failed: " + "; ".join(cleanup_errors)
                    self._error = RuntimeError(message)
                    self._record_fte_task_terminal()
                    self._is_done = True
                    return
            else:
                self._finish_task_output_ownership([])
                self._result = RayTaskResult.success(
                    [],
                    self._stats_from_status(status),
                    None,
                    0,
                    (self.task_context_info or {}).get("exchange_sink_instance"),
                )
            self._record_fte_task_terminal()
            try:
                self.ack()
            except Exception as exc:
                cleanup_errors = self._discard_unacked_finished_result()
                message = f"failed to ack FTE task result for {self.task_id}: {exc}"
                if cleanup_errors:
                    message += "; cleanup also failed: " + "; ".join(cleanup_errors)
                self._error = RuntimeError(message)
                self._is_done = True
                return
            self._unacked_output_owners.clear()
            self._is_done = True
            return
        if state in (FteTaskState.FAILED, FteTaskState.CANCELED, FteTaskState.ABORTED):
            # FteAttemptStatusWatcher is the sole scheduler/status writer for
            # every terminal state.  The result handle only converts the
            # observed transport terminal into its local consumer result.
            failure = _normalize_failure_payload(status.get("failure"))
            message = failure.get("message") or failure["error_code"]
            self._error = RuntimeError(f"FTE task {self.task_id} {state.value}: {message}")
            self._record_fte_task_terminal()
            self._is_done = True

    @staticmethod
    def _state_from_status(status: dict[str, Any]) -> FteTaskState:
        state = status.get("state")
        if isinstance(state, FteTaskState):
            return state
        return FteTaskState(str(state))

    def _finish_task_output_ownership(
        self,
        outputs: list[dict[str, Any]],
    ) -> list[Any]:
        owners = self.worker_handle.finish_fte_task_with_outputs(
            self._attempt_id().to_dict(),
            dict(self.query_task_lease or {}),
            outputs,
        )
        if not isinstance(owners, list) or len(owners) != len(outputs):
            cleanup_errors = self._release_output_owners(
                [owner for owner in owners if callable(getattr(owner, "release", None))]
                if isinstance(owners, list)
                else []
            )
            message = (
                "FTE output ownership transfer must return one owner per output: "
                f"outputs={len(outputs)} owners={len(owners) if isinstance(owners, list) else 'invalid'}"
            )
            if cleanup_errors:
                message += "; partial owner cleanup failed: " + "; ".join(cleanup_errors)
            raise RuntimeError(message)
        if any(not callable(getattr(owner, "release", None)) for owner in owners):
            cleanup_errors = self._release_output_owners(
                [owner for owner in owners if callable(getattr(owner, "release", None))]
            )
            message = "FTE output lease owner must provide release()"
            if cleanup_errors:
                message += "; partial owner cleanup failed: " + "; ".join(cleanup_errors)
            raise TypeError(message)
        return owners

    @staticmethod
    def _release_output_owners(owners: list[Any]) -> list[str]:
        rollback_errors: list[str] = []
        for owner in owners:
            try:
                owner.release()
            except Exception as exc:
                rollback_errors.append(str(exc))
        return rollback_errors

    @classmethod
    def _rollback_output_owners(cls, owners: list[Any], *, cause: Exception) -> None:
        rollback_errors = cls._release_output_owners(owners)
        if rollback_errors:
            raise RuntimeError(
                f"{cause}; output ownership rollback also failed: " + "; ".join(rollback_errors)
            ) from cause

    def _discard_unacked_finished_result(self) -> list[str]:
        owners = self._unacked_output_owners
        self._unacked_output_owners = []
        cleanup_errors = self._release_output_owners(owners)
        try:
            self.release_result_payload()
        except Exception as exc:
            cleanup_errors.append(str(exc))
        self._result = None
        return cleanup_errors

    def _normalize_raw_result(self, raw_result: Any) -> Any:
        if isinstance(raw_result, RayTaskResult):
            self._finish_task_output_ownership([])
            return raw_result
        if isinstance(raw_result, tuple):
            parts = raw_result[0] if len(raw_result) >= 1 else []
            metas = raw_result[1] if len(raw_result) >= 2 else []
            result_schema = raw_result[2] if len(raw_result) >= 3 else None
            stats_payload = raw_result[3] if len(raw_result) >= 4 else []
            flight_port = int(raw_result[4] or 0) if len(raw_result) >= 5 else 0
            exchange_sink_instance = raw_result[5] if len(raw_result) >= 6 else None
            normalized_parts: list[tuple[Any, int, int, Any | None]] = []
            output_specs: list[dict[str, Any]] = []
            task_lease_id = str((self.query_task_lease or {}).get("lease_id") or "").strip()
            for idx, part in enumerate(parts):
                num_rows = 0
                size_bytes = 0
                if idx < len(metas):
                    meta = metas[idx]
                    if isinstance(meta, dict):
                        num_rows = int(meta.get("num_rows") or 0)
                        size_bytes = int(meta.get("size_bytes") or 0)
                    elif isinstance(meta, (tuple, list)) and len(meta) >= 2:
                        num_rows = int(meta[0])
                        size_bytes = int(meta[1])
                previous_owner = None
                if isinstance(part, RayResultPartitionRef):
                    if num_rows == 0:
                        num_rows = int(part.num_rows)
                    if size_bytes == 0:
                        size_bytes = int(part.size_bytes)
                    previous_owner = part.lease_owner
                    part = part.object_ref
                output_specs.append(
                    {
                        "block_id": f"fte-block:{task_lease_id}:{idx}",
                        "size_bytes": size_bytes,
                    }
                )
                normalized_parts.append((part, num_rows, size_bytes, previous_owner))
            lease_owners = self._finish_task_output_ownership(output_specs)
            self._unacked_output_owners = list(lease_owners)
            try:
                partition_refs = [
                    RayResultPartitionRef(part, num_rows, size_bytes, lease_owner)
                    for (part, num_rows, size_bytes, _), lease_owner in zip(
                        normalized_parts,
                        lease_owners,
                        strict=True,
                    )
                ]
                for _, _, _, previous_owner in normalized_parts:
                    if previous_owner is not None:
                        previous_owner.release()
            except Exception as exc:
                self._unacked_output_owners = []
                self._rollback_output_owners(lease_owners, cause=exc)
                raise
            return RayTaskResult.success(
                partition_refs,
                self._stats_from_payload(stats_payload),
                result_schema,
                flight_port,
                exchange_sink_instance,
            )
        return RayTaskResult.success([], [], None)

    def _requires_final_output_stats(self) -> bool:
        task_context_info = self.task_context_info or {}
        if task_context_info.get("exchange_sink_instance") is not None:
            return True
        return bool(task_context_info.get("requires_spooling_output_stats"))

    @staticmethod
    def _output_stats_from_status(status: dict[str, Any]) -> Any:
        output_stats = None
        if status.get("spooling_output_stats") is not None:
            output_stats = status.get("spooling_output_stats")
        elif status.get("output_stats") is not None:
            output_stats = status.get("output_stats")
        task_stats = status.get("task_stats")
        if isinstance(task_stats, dict):
            if isinstance(output_stats, dict):
                merged = dict(task_stats)
                merged.update(output_stats)
                return merged
            if output_stats is None:
                return dict(task_stats)
        return output_stats

    @staticmethod
    def _stats_from_status(status: dict[str, Any]) -> list[int]:
        stats = status.get("stats")
        if stats is None:
            stats = status.get("stats_serialized")
        return FteWorkerTaskHandle._stats_from_payload(stats)

    @staticmethod
    def _stats_from_payload(stats: Any) -> list[int]:
        if stats is None:
            return []
        if isinstance(stats, (bytes, bytearray)):
            return list(stats)
        if isinstance(stats, memoryview):
            return list(stats.tobytes())
        if isinstance(stats, (list, tuple)):
            return [int(value) for value in stats]
        return []


def batch_wait_ready(handles: list[Any]) -> list[int]:
    """Batch-check which worker task handle instances are ready.

    Called from C++ RayTaskResultPoller under a single GIL acquisition.  Each
    handle owns its terminal transition under its lifecycle lock, so the batch
    path delegates to that same state machine instead of mutating fields.

    Returns a list of indices into `handles` that are done.
    """
    return [index for index, handle in enumerate(handles) if handle.done()]


def _udf_test_hooks_enabled() -> bool:
    value = os.environ.get("VANE_ENABLE_UDF_TEST_HOOKS", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


@ray.remote(
    num_cpus=0,
)
class RayQueryDriverActor:
    def __init__(
        self,
        runtime_config: dict[str, str],
        duckdb_memory_bytes: int,
    ) -> None:
        scrub_shared_runtime_session_env()
        duckdb_memory_bytes = int(duckdb_memory_bytes)
        if duckdb_memory_bytes <= 0:
            raise ValueError("Ray driver duckdb_memory_bytes must be positive")
        self._runtime_config = _normalize_session_config(runtime_config)
        (
            self._client_lease_timeout_s,
            self._client_heartbeat_interval_s,
        ) = _client_owner_lease_settings()
        self._client_ids: set[str] = set()
        self._client_leases: dict[str, _ClientOwnerLease] = {}
        self._detaching_client_ids: set[str] = set()
        self._detached_client_results = BoundedReplayMap[str, bool](capacity=_CLIENT_DETACH_REPLAY_CAPACITY)
        self._client_detach_locks: dict[str, asyncio.Lock] = {}
        self._expired_client_cleanup_tasks: dict[str, asyncio.Task[Any]] = {}
        self._client_lease_maintenance_task: asyncio.Task[Any] | None = None
        self._client_lease_maintenance_stop: asyncio.Event | None = None
        self._client_lease_maintenance_error = ""
        self._client_lease_maintenance_failures = 0
        self._sessions: dict[str, _DriverSession] = {}
        self._closed_session_owners = BoundedReplayMap[str, str](capacity=_SESSION_CLOSE_REPLAY_CAPACITY)
        self._session_lock = threading.RLock()
        self._plan_session_ids: dict[str, str] = {}
        self._plan_connections: dict[str, Any] = {}
        self._plan_teardown_condition = threading.Condition(self._session_lock)
        self._plan_teardowns_in_progress: set[str] = set()
        self._copy_operations_inflight: dict[str, _CopyOperationInFlight] = {}
        self._copy_operation_terminal = BoundedReplayMap[str, _CopyOperationTerminal](
            capacity=_COPY_OPERATION_REPLAY_CAPACITY
        )
        # Outcome payloads are bounded, but an admitted identity must remain
        # known for the owner's lifetime. Evicting a payload may make recovery
        # impossible; it must never authorize the same write to run again.
        self._copy_operation_identities: dict[str, tuple[str, str]] = {}
        self._copy_cleanup_tasks: dict[str, asyncio.Task[CopyPlanOutcome]] = {}
        self._driver_handle = ray.get_runtime_context().current_actor
        # Avoid referencing C++ types at import time; store opaque plan objects and
        # lazily instantiate the plan runner when first required.
        self.curr_plans: dict[str, Any] = {}
        self.curr_streams: dict[str, Any] = {}  # C++ ResultPartitionStream objects
        self._plan_query_ids: dict[str, str] = {}
        self._query_terminal_errors: dict[str, str] = {}
        self._leased_result_partition_refs: dict[str, dict[str, Any]] = {}
        self._result_partition_ref_counters: dict[str, int] = {}
        self.plan_runner: Any | None = None
        self._active_udf_actors: list[Any] = []
        self._active_udf_actors_by_plan: dict[str, list[Any]] = {}
        self._active_vllm_actors: list[Any] = []
        self._active_vllm_actors_by_plan: dict[str, list[Any]] = {}
        self._query_resource_lock = threading.RLock()
        self._query_graphs: dict[str, Any] = {}
        self._query_allocations: dict[str, Any] = {}
        self._query_node_capacities: tuple[Any, ...] = ()
        self._query_allocation_teardowns_pending: dict[
            str,
            _DeferredQueryAllocationTeardown,
        ] = {}
        self._query_allocation_teardowns_claimed: set[str] = set()
        self._query_allocation_teardown_futures: set[asyncio.Future[Any]] = set()
        self._query_task_lease_requests: dict[str, dict[str, Any]] = {}
        self._query_output_lease_requests: dict[str, dict[str, Any]] = {}
        self._query_task_lease_request_tombstones = BoundedReplayMap[str, dict[str, Any]](
            capacity=_LEASE_REQUEST_REPLAY_CAPACITY
        )
        # Request payload replay is bounded and query-scoped. Granted task
        # lease IDs are signed capabilities, so cleanup ACK replay does not
        # depend on retaining an entry in either compact denial ledger.
        self._query_task_terminal_controls = BoundedReplayMap[bytes, _TaskTerminalControl](
            capacity=_LEASE_REQUEST_REPLAY_CAPACITY
        )
        self._query_task_terminal_attempts = BoundedReplayMap[bytes, bytes](capacity=_LEASE_REQUEST_REPLAY_CAPACITY)
        self._query_output_lease_request_tombstones = BoundedReplayMap[str, dict[str, Any]](
            capacity=_LEASE_REQUEST_REPLAY_CAPACITY
        )
        # Request payload replay is bounded and query-scoped. Granted output
        # lease IDs are signed capabilities, so cleanup ACK replay does not
        # depend on retaining an entry in this map.
        self._query_output_terminal_controls = BoundedReplayMap[bytes, _OutputTerminalControl](
            capacity=_LEASE_REQUEST_REPLAY_CAPACITY
        )
        self._query_lease_control_key = os.urandom(32)
        self._query_lease_generation_ids: dict[str, str] = {}
        self._query_task_request_owner_by_identity: dict[tuple[str, str, str], str] = {}
        self._query_output_request_owner_by_identity: dict[tuple[str, str], str] = {}
        self._query_resource_closing_queries: set[str] = set()
        self._query_execution_quiesced_queries: set[str] = set()
        self._query_pending_execution_teardowns: dict[str, set[str]] = {}
        self._query_task_admission_pumps: set[str] = set()
        self._query_task_lease_condition = threading.Condition()
        self._query_task_completion_waiters: dict[str, tuple[str, asyncio.Task[None]]] = {}
        self._query_output_admission_pumps: set[str] = set()
        self._query_fte_admission_pumps: dict[str, asyncio.Task[Any]] = {}
        self._query_fte_admission_dirty_queries: set[str] = set()
        self._query_fte_admission_done_events: dict[str, threading.Event] = {}
        self._query_resource_admission_signal_lock = threading.Lock()
        self._query_resource_admission_dirty_queries: set[str] = set()
        self._query_resource_admission_signal_scheduled = False
        self._query_resource_admission_loop: asyncio.AbstractEventLoop | None = None
        self._query_resource_admission_fence_lock = threading.Lock()
        self._query_resource_admission_uncertain_fences: dict[str, _AdmissionLoopFence] = {}
        self._query_resource_maintenance_task: asyncio.Task[Any] | None = None
        self._query_resource_maintenance_stop: asyncio.Event | None = None
        self._query_resource_maintenance_error = ""
        self._query_resource_maintenance_failures = 0
        self._query_resource_last_maintenance_at = 0.0
        self._query_resource_last_capacity_refresh_at = 0.0
        self._progress_snapshot_lock = threading.Lock()
        self._progress_snapshot_builds: dict[tuple[str, float | None], asyncio.Task[Any]] = {}
        self._progress_snapshot_cache: dict[tuple[str, float | None], dict[str, Any]] = {}
        self._plan_runner_lifecycle_lock = threading.RLock()
        self._driver_shutdown_lock = asyncio.Lock()
        self._driver_shutdown_started = False
        self._driver_shutdown_complete = False
        self._driver_duckdb_memory_bytes = duckdb_memory_bytes

        self._query_resource_coordinator = self._create_query_resource_coordinator()

        # Set event loop - create one if not running
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, get or create the event loop
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        set_event_loop(loop)
        _set_global_event_loop(loop)
        self._query_resource_admission_loop = loop

        # Eagerly create DuckDB connection during __init__ so the ~2s
        # cold-start cost overlaps with actor creation instead of
        # blocking the first query.
        self._duckdb_conn: Any = None
        self._ensure_duckdb_conn()

        # Eagerly create PlanRunner + warm up Worker workers so the
        # ~2.8s cold-start cost overlaps with GPU model loading.
        self._get_plan_runner()
        self._start_client_lease_maintenance()
        self._start_query_resource_maintenance()

    def _drop_query_fragments_sync(self, query_id: str) -> None:
        self._drop_query_fragments_after_admission_fence_sync(
            query_id,
            release_resources=True,
        )

    def _drop_query_fragments_after_admission_fence_sync(
        self,
        query_id: str,
        *,
        release_resources: bool,
    ) -> None:
        if not query_id:
            return
        try:
            self._fence_query_resource_admission_for_teardown(query_id)
        except BaseException as exc:
            if isinstance(exc, QueryTeardownOwnershipError):
                raise
            raise QueryTeardownOwnershipError(
                f"failed to fence query teardown before fragment drop for {query_id}: {type(exc).__name__}: {exc}"
            ) from exc
        errors: list[BaseException] = []
        pending_execution_teardowns = getattr(self, "_query_pending_execution_teardowns", None)
        if pending_execution_teardowns is None:
            pending_execution_teardowns = {}
            self._query_pending_execution_teardowns = pending_execution_teardowns
        remembered_execution_query_ids = set(pending_execution_teardowns.get(str(query_id), ()))
        try:
            plan_runner = self._get_plan_runner()
            from duckdb.runners.ray.fte_fragment_scheduler import (
                fte_execution_query_ids_for_resource,
            )

            execution_query_ids = set(fte_execution_query_ids_for_resource(query_id))
            execution_query_ids.add(str(query_id))
        except BaseException as exc:
            errors.append(exc)
            execution_query_ids = {str(query_id)}
            plan_runner = None
        execution_query_ids.update(remembered_execution_query_ids)
        pending_execution_teardowns.setdefault(str(query_id), set()).update(execution_query_ids)
        resource_query_id = str(query_id)
        teardown_query_ids = sorted(
            execution_query_id for execution_query_id in execution_query_ids if execution_query_id != resource_query_id
        )
        successful_execution_query_ids: set[str] = set()
        if plan_runner is not None:
            # The resource query owns query-scoped replay and datasource state.
            # Drain every internal execution before releasing that outer owner.
            for execution_query_id in teardown_query_ids:
                try:
                    plan_runner.drop_query_fragments(execution_query_id)
                except BaseException as exc:
                    errors.append(exc)
                else:
                    successful_execution_query_ids.add(execution_query_id)
        from duckdb.runners.ray.fte_fragment_scheduler import (
            fte_query_remote_teardown_blockers,
        )

        teardown_blockers_by_query = {
            execution_query_id: blockers
            for execution_query_id in teardown_query_ids
            if (blockers := fte_query_remote_teardown_blockers(execution_query_id))
        }
        if plan_runner is not None and not errors and not teardown_blockers_by_query:
            try:
                plan_runner.drop_query_fragments(resource_query_id)
            except BaseException as exc:
                errors.append(exc)
            else:
                successful_execution_query_ids.add(resource_query_id)
        blockers = fte_query_remote_teardown_blockers(resource_query_id)
        if blockers:
            teardown_blockers_by_query[resource_query_id] = blockers
        pending_for_query = pending_execution_teardowns[str(query_id)]
        pending_for_query.difference_update(
            execution_query_id
            for execution_query_id in successful_execution_query_ids
            if execution_query_id not in teardown_blockers_by_query
        )
        if errors or teardown_blockers_by_query or pending_for_query:
            details = [
                *(f"{type(error).__name__}: {error}" for error in errors),
                *(
                    f"owners[{execution_query_id}]=" + ", ".join(blockers)
                    for execution_query_id, blockers in sorted(teardown_blockers_by_query.items())
                ),
            ]
            if pending_for_query:
                details.append("pending_execution_queries=" + ", ".join(sorted(pending_for_query)))
            raise QueryTeardownOwnershipError(
                f"distributed query teardown retains remote ownership for {query_id}: " + "; ".join(details)
            ) from (errors[0] if errors else None)
        pending_execution_teardowns.pop(str(query_id), None)
        try:
            # The remote fan-out is best effort, but local FTE ownership must
            # be fully quiesced before QRM/coordinator release. This second,
            # idempotent barrier also distinguishes dead-worker RPC failures
            # from a live local watcher that still depends on query state.
            from duckdb.runners.ray.fte_fragment_scheduler import (
                _drop_fte_registry_for_query,
            )

            _drop_fte_registry_for_query(query_id)
        except BaseException as exc:
            quiesce_errors: list[BaseException] = [*errors, exc]
            raise QueryFteRegistryQuiesceError(
                f"local FTE registry did not quiesce for {query_id}; "
                "query remains fenced for retry: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in quiesce_errors)
            ) from exc
        if release_resources:
            try:
                self._release_query_resources(
                    query_id,
                    reason="query_fragments_dropped",
                    admission_fenced=True,
                    execution_quiesced=True,
                )
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(
                f"failed to drop distributed query fragments for {query_id}: "
                + "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
            ) from errors[0]

    async def fragment_stats(self) -> dict[str, Any]:
        plan_runner = self._get_plan_runner()
        stats = await asyncio.to_thread(plan_runner.fragment_stats)
        if isinstance(stats, dict):
            return stats
        raise TypeError("DistributedPhysicalPlanRunner.fragment_stats() must return a dict")

    def _ensure_progress_snapshot_state(self) -> None:
        if not hasattr(self, "_progress_snapshot_lock"):
            self._progress_snapshot_lock = threading.Lock()
        if not hasattr(self, "_progress_snapshot_builds"):
            self._progress_snapshot_builds = {}
        if not hasattr(self, "_progress_snapshot_cache"):
            self._progress_snapshot_cache = {}

    def _drop_progress_snapshot_state(self, query_id: str) -> None:
        self._ensure_progress_snapshot_state()
        query_key = str(query_id or "").strip()
        if not query_key:
            return
        with self._progress_snapshot_lock:
            keys = {
                key for key in (*self._progress_snapshot_builds, *self._progress_snapshot_cache) if key[0] == query_key
            }
            builds = [self._progress_snapshot_builds.pop(key) for key in keys if key in self._progress_snapshot_builds]
            for key in keys:
                self._progress_snapshot_cache.pop(key, None)
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        for build in builds:
            if build.done():
                continue
            build_loop = build.get_loop()
            if current_loop is build_loop:
                build.cancel()
            elif build_loop.is_running():
                build_loop.call_soon_threadsafe(build.cancel)

    def _build_local_progress_snapshot(
        self,
        query_id: str,
        started_at: float | None,
    ) -> dict[str, Any]:
        from duckdb.runners.progress import build_progress_snapshot
        from duckdb.runners.ray.fte_fragment_scheduler import (
            fte_progress_registry_snapshot,
        )
        from duckdb.runners.ray.query_resource_runtime import query_resource_manager_snapshot

        manager_snapshot: dict[str, Any] = {}
        try:
            manager_snapshot = query_resource_manager_snapshot(str(query_id))
        except Exception:
            pass
        registry_snapshot = fte_progress_registry_snapshot(str(query_id))
        query_snapshot = registry_snapshot["queries"][str(query_id)]
        if manager_snapshot:
            query_snapshot["query_resource_manager"] = manager_snapshot
        snapshot = build_progress_snapshot(
            registry_snapshot,
            str(query_id),
            started_at=started_at,
        )
        snapshot["query_resource_manager"] = manager_snapshot
        return snapshot

    async def progress_snapshot(
        self,
        owner_id: str,
        session_id: str,
        plan_id: str,
        started_at: float | None = None,
    ) -> dict[str, Any]:
        """Build one local-only progress view without blocking actor admission."""
        self._require_plan_session(owner_id, session_id, plan_id)
        query_id = self._plan_query_ids.get(str(plan_id))
        if not query_id:
            raise RuntimeError(f"query plan {plan_id} is missing its registered query resource owner")
        self._ensure_progress_snapshot_state()
        key = (
            str(query_id),
            None if started_at is None else float(started_at),
        )
        with self._progress_snapshot_lock:
            cached = self._progress_snapshot_cache.get(key)
            build = self._progress_snapshot_builds.get(key)
            if build is None:
                build = asyncio.create_task(
                    asyncio.to_thread(
                        self._build_local_progress_snapshot,
                        key[0],
                        key[1],
                    ),
                    name=f"vane-progress-snapshot:{key[0]}",
                )
                self._progress_snapshot_builds[key] = build

                def _complete(completed: asyncio.Task[Any]) -> None:
                    try:
                        snapshot = completed.result()
                    except BaseException:
                        snapshot = None
                    with self._progress_snapshot_lock:
                        if snapshot is not None:
                            self._progress_snapshot_cache[key] = snapshot
                        if self._progress_snapshot_builds.get(key) is completed:
                            self._progress_snapshot_builds.pop(key, None)

                build.add_done_callback(_complete)
            if cached is not None:
                return dict(cached)
        return await asyncio.shield(build)

    def _ensure_duckdb_conn(self) -> Any:
        """Eagerly create the DuckDB connection."""
        if self._duckdb_conn is not None:
            return self._duckdb_conn

        import duckdb
        from duckdb.runners.fte.memory_config import apply_duckdb_memory_limit

        self._duckdb_conn = duckdb.connect()
        _apply_duckdb_thread_setting(self._duckdb_conn)
        apply_duckdb_memory_limit(self._duckdb_conn, self._driver_duckdb_memory_bytes)
        return self._duckdb_conn

    def _ensure_client_lease_state(self) -> None:
        if not hasattr(self, "_client_lease_timeout_s"):
            self._client_lease_timeout_s = _DEFAULT_CLIENT_LEASE_TIMEOUT_S
        if not hasattr(self, "_client_heartbeat_interval_s"):
            self._client_heartbeat_interval_s = _DEFAULT_CLIENT_HEARTBEAT_INTERVAL_S
        if not hasattr(self, "_client_leases"):
            self._client_leases = {}
        if not hasattr(self, "_client_detach_locks"):
            self._client_detach_locks = {}
        if not hasattr(self, "_expired_client_cleanup_tasks"):
            self._expired_client_cleanup_tasks = {}
        # Local test shells historically populated only ``_client_ids``.
        # Treat those owners as non-expiring unless a test installs a lease.
        for owner_key in self._client_ids:
            self._client_leases.setdefault(
                owner_key,
                _ClientOwnerLease(
                    lease_token="",
                    expires_at=math.inf,
                ),
            )

    def _owner_is_active_locked(self, owner_key: str, *, now: float | None = None) -> bool:
        self._ensure_client_lease_state()
        lease = self._client_leases.get(owner_key)
        if owner_key not in self._client_ids or lease is None:
            return False
        timestamp = time.monotonic() if now is None else float(now)
        if lease.state in {"active", "detaching"} and timestamp >= lease.expires_at:
            lease.state = "expired"
            lease.next_cleanup_at = timestamp
            self._detaching_client_ids.add(owner_key)
        return lease.state == "active" and owner_key not in self._detaching_client_ids

    def _require_owner(self, owner_id: str) -> None:
        owner_key = str(owner_id).strip()
        with self._session_lock:
            if not self._owner_is_active_locked(owner_key):
                raise PermissionError("Ray query runtime operation requires an attached client owner")

    def attach_client(
        self,
        owner_id: str,
        runtime_config: dict[str, str],
        lease_token: str,
    ) -> bool:
        owner_key = str(owner_id).strip()
        if not owner_key:
            raise ValueError("Ray query runtime owner_id must not be empty")
        token_key = str(lease_token).strip()
        if not token_key:
            raise ValueError("Ray query runtime lease_token must not be empty")
        normalized_config = _normalize_session_config(runtime_config)
        with self._session_lock:
            self._ensure_client_lease_state()
            if self._driver_shutdown_started:
                raise RayRuntimeShuttingDownError("Ray query runtime is shutting down")
            if owner_key in self._detaching_client_ids or owner_key in self._detached_client_results:
                raise RuntimeError("Ray query runtime owner identity was already detached")
            if normalized_config != self._runtime_config:
                raise RuntimeError("Ray query runtime configuration differs from the existing job runtime")
            existing = self._client_leases.get(owner_key)
            if existing is not None and existing.lease_token != token_key:
                raise RuntimeError("Ray query runtime owner identity has a different lease generation")
            expires_at = time.monotonic() + float(self._client_lease_timeout_s)
            if existing is None:
                self._client_leases[owner_key] = _ClientOwnerLease(
                    lease_token=token_key,
                    expires_at=expires_at,
                )
            else:
                if existing.state != "active":
                    raise RuntimeError("Ray query runtime owner lease is no longer active")
                existing.expires_at = expires_at
            self._client_ids.add(owner_key)
        return True

    def heartbeat_client(self, owner_id: str, lease_token: str) -> bool:
        """Renew exactly one live client generation without resurrecting it."""
        owner_key = str(owner_id).strip()
        token_key = str(lease_token).strip()
        with self._session_lock:
            self._ensure_client_lease_state()
            lease = self._client_leases.get(owner_key)
            if (
                owner_key not in self._client_ids
                or lease is None
                or lease.lease_token != token_key
                or lease.state == "expired"
            ):
                raise PermissionError("Ray query runtime client lease is no longer active")
            timestamp = time.monotonic()
            if lease.state in {"active", "detaching"} and timestamp >= lease.expires_at:
                lease.state = "expired"
                lease.next_cleanup_at = timestamp
                self._detaching_client_ids.add(owner_key)
                raise PermissionError("Ray query runtime client lease expired")
            if lease.state not in {"active", "detaching"}:
                raise PermissionError("Ray query runtime client lease is no longer active")
            lease.expires_at = timestamp + float(self._client_lease_timeout_s)
        return True

    def _require_session(
        self,
        owner_id: str,
        session_id: str,
        *,
        allow_closing: bool = False,
    ) -> _DriverSession:
        self._require_owner(owner_id)
        session_key = str(session_id).strip()
        if not session_key:
            raise ValueError("Vane session_id must not be empty")
        with self._session_lock:
            session = self._sessions.get(session_key)
        if session is None:
            raise RuntimeError(f"Vane session is not open: {session_key}")
        if session.owner_id != str(owner_id).strip():
            raise PermissionError("Vane session operation requires its owning runtime client")
        with session.condition:
            if session.closed:
                raise RuntimeError(f"Vane session is closed: {session_key}")
            if session.closing and not allow_closing:
                raise RuntimeError(f"Vane session is closing: {session_key}")
        return session

    def _require_plan_session(self, owner_id: str, session_id: str, plan_id: str) -> _DriverSession:
        session = self._require_session(owner_id, session_id)
        plan_key = str(plan_id)
        with self._session_lock:
            plan_session_id = self._plan_session_ids.get(plan_key)
        with session.condition:
            owns_plan = plan_key in session.plan_ids
        if plan_session_id != str(session_id) or not owns_plan:
            raise PermissionError("query plan does not belong to the requested Vane session")
        return session

    @staticmethod
    def _begin_session_operation(session: _DriverSession, session_id: str) -> None:
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        with session.condition:
            if session.closed:
                raise RuntimeError(f"Vane session is closed: {session_id}")
            if session.closing:
                raise RuntimeError(f"Vane session is closing: {session_id}")
            session.active_operations += 1
            if current_task is not None:
                session.active_operation_tasks.add(current_task)

    @staticmethod
    def _end_session_operation(session: _DriverSession) -> None:
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        with session.condition:
            if session.active_operations <= 0:
                raise RuntimeError("Vane session operation accounting underflow")
            session.active_operations -= 1
            if current_task is not None:
                session.active_operation_tasks.discard(current_task)
            session.condition.notify_all()

    async def open_session(self, owner_id: str, session_id: str, config: dict[str, str]) -> bool:
        self._require_owner(owner_id)
        owner_key = str(owner_id).strip()
        session_key = str(session_id).strip()
        if not session_key:
            raise ValueError("Vane session_id must not be empty")
        normalized_config = _normalize_session_config(config)
        with self._session_lock:
            if not self._owner_is_active_locked(owner_key):
                raise PermissionError("Ray query runtime operation requires an attached client owner")
            if self._driver_shutdown_started:
                raise VaneSessionOpenRejectedError("Ray query runtime is shutting down")
            if session_key in self._closed_session_owners:
                raise VaneSessionOpenRejectedError(f"Vane session identity was already closed: {session_key}")
            existing = self._sessions.get(session_key)
            if existing is not None:
                if existing.owner_id != owner_key or existing.config != normalized_config:
                    raise VaneSessionOpenRejectedError(f"Vane session identity collision: {session_key}")
                return False

        session_connection = self._ensure_duckdb_conn().cursor()
        with self._session_lock:
            if not self._owner_is_active_locked(owner_key):
                session_connection.close()
                raise PermissionError("Ray query runtime operation requires an attached client owner")
            if bool(getattr(self, "_driver_shutdown_started")):
                session_connection.close()
                raise VaneSessionOpenRejectedError("Ray query runtime is shutting down")
            if session_key in self._closed_session_owners:
                session_connection.close()
                raise VaneSessionOpenRejectedError(f"Vane session identity was already closed: {session_key}")
            existing = self._sessions.get(session_key)
            if existing is not None:
                session_connection.close()
                if existing.owner_id != owner_key or existing.config != normalized_config:
                    raise VaneSessionOpenRejectedError(f"Vane session identity collision: {session_key}")
                return False
            self._sessions[session_key] = _DriverSession(
                owner_id=owner_key,
                config=normalized_config,
                connection=session_connection,
                s3_config={},
            )
        return True

    def _validate_plan_session(self, session_id: str, plan: Any, session: _DriverSession) -> None:
        plan_session_id = str(plan.session_id()).strip()
        if plan_session_id != str(session_id):
            raise ValueError(f"logical plan session mismatch: plan={plan_session_id!r} requested={str(session_id)!r}")
        plan_config = _normalize_session_config(plan.session_config())
        if plan_config != session.config:
            raise ValueError(f"logical plan session config changed after session open: {session_id}")

    def ping(self, owner_id: str) -> bool:
        """Health-check: returns True if the actor is alive."""
        self._require_owner(owner_id)
        if self._driver_shutdown_started:
            raise RuntimeError("Ray query driver is shutting down")
        return True

    def runtime_replacement_ready(self) -> bool:
        """Return whether a new client may retire this drained named runtime."""
        with self._session_lock:
            return self._driver_shutdown_complete and not self._client_ids

    def runtime_lifecycle_snapshot(self, owner_id: str) -> dict[str, Any]:
        """Return aggregate lifecycle state to an attached diagnostic caller."""
        self._require_owner(owner_id)
        with self._session_lock:
            self._ensure_client_lease_state()
            lease_states: dict[str, int] = {}
            cleanup_errors = 0
            for lease in self._client_leases.values():
                lease_states[lease.state] = lease_states.get(lease.state, 0) + 1
                cleanup_errors += bool(lease.last_cleanup_error)
            return {
                "client_count": len(self._client_ids),
                "session_count": len(self._sessions),
                "plan_count": len(self._plan_session_ids),
                "lease_states": lease_states,
                "cleanup_errors": cleanup_errors,
                "lease_maintenance": {
                    "failures": int(getattr(self, "_client_lease_maintenance_failures", 0)),
                    "error": str(getattr(self, "_client_lease_maintenance_error", "")),
                },
            }

    async def _shutdown_runtime(self) -> None:
        """Stop background maintenance and gracefully drain every worker actor."""
        async with self._driver_shutdown_lock:
            if self._driver_shutdown_complete:
                return
            with self._plan_runner_lifecycle_lock:
                self._driver_shutdown_started = True
                plan_runner = self.plan_runner
            await self.stop_client_lease_maintenance()
            await self.stop_query_resource_maintenance()
            self._ensure_query_allocation_teardown_state()
            teardown_futures = tuple(self._query_allocation_teardown_futures)
            if teardown_futures:
                await asyncio.gather(*teardown_futures, return_exceptions=True)
            if plan_runner is not None:
                await _to_thread_with_owned_side_effects(plan_runner.shutdown)
            self._driver_shutdown_complete = True

    def _client_detach_lock_for(self, owner_key: str) -> asyncio.Lock:
        self._ensure_client_lease_state()
        lock = self._client_detach_locks.get(owner_key)
        if lock is None:
            lock = asyncio.Lock()
            self._client_detach_locks[owner_key] = lock
        return lock

    def _mark_expired_client_leases(
        self,
        *,
        now: float | None = None,
    ) -> tuple[str, ...]:
        timestamp = time.monotonic() if now is None else float(now)
        expired_sessions: list[_DriverSession] = []
        with self._session_lock:
            self._ensure_client_lease_state()
            for owner_key in sorted(self._client_ids):
                lease = self._client_leases[owner_key]
                if lease.state in {"active", "detaching"} and timestamp >= lease.expires_at:
                    lease.state = "expired"
                    lease.next_cleanup_at = timestamp
                    self._detaching_client_ids.add(owner_key)
                    expired_sessions.extend(
                        session for session in self._sessions.values() if session.owner_id == owner_key
                    )
            due = tuple(
                owner_key
                for owner_key in sorted(self._client_ids)
                if self._client_leases[owner_key].state == "expired"
                and self._client_leases[owner_key].next_cleanup_at <= timestamp
                and owner_key not in self._expired_client_cleanup_tasks
            )
        for session in expired_sessions:
            with session.condition:
                session.closing = True
        return due

    def _schedule_expired_client_reclamations(self) -> None:
        for owner_key in self._mark_expired_client_leases():
            with self._session_lock:
                sessions = tuple(session for session in self._sessions.values() if session.owner_id == owner_key)
            for session in sessions:
                with session.condition:
                    active_operation_tasks = tuple(session.active_operation_tasks)
                for active_operation in active_operation_tasks:
                    active_operation.cancel()
            task = asyncio.create_task(
                self._detach_client_owner(owner_key, expired=True),
                name=f"vane-expired-client-cleanup:{owner_key}",
            )
            self._expired_client_cleanup_tasks[owner_key] = task

            def _complete(
                completed: asyncio.Task[Any],
                *,
                expected_owner: str = owner_key,
            ) -> None:
                if self._expired_client_cleanup_tasks.get(expected_owner) is completed:
                    self._expired_client_cleanup_tasks.pop(expected_owner, None)
                try:
                    completed.result()
                except BaseException:
                    pass
                with self._session_lock:
                    owner_detached = expected_owner not in self._client_ids
                lock = self._client_detach_locks.get(expected_owner)
                if owner_detached and lock is not None and not lock.locked():
                    self._client_detach_locks.pop(expected_owner, None)

            task.add_done_callback(_complete)

    def _start_client_lease_maintenance(self) -> None:
        task = getattr(self, "_client_lease_maintenance_task", None)
        if task is not None and not task.done():
            return
        loop = asyncio.get_running_loop()
        self._client_lease_maintenance_stop = asyncio.Event()
        self._client_lease_maintenance_task = loop.create_task(
            self._client_lease_maintenance_loop(),
            name="vane-client-lease-maintenance",
        )

    async def _client_lease_maintenance_loop(self) -> None:
        stop = self._client_lease_maintenance_stop
        if stop is None:
            raise RuntimeError("client lease maintenance stop event is not initialized")
        interval = min(1.0, float(self._client_heartbeat_interval_s))
        while not stop.is_set():
            try:
                self._schedule_expired_client_reclamations()
                self._client_lease_maintenance_error = ""
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                self._client_lease_maintenance_failures += 1
                self._client_lease_maintenance_error = f"{type(error).__name__}: {error}"
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except (TimeoutError, asyncio.TimeoutError):
                continue

    async def stop_client_lease_maintenance(self) -> None:
        stop = getattr(self, "_client_lease_maintenance_stop", None)
        task = getattr(self, "_client_lease_maintenance_task", None)
        if stop is not None:
            stop.set()
        if task is not None and task is not asyncio.current_task():
            await task
        self._client_lease_maintenance_task = None

    async def detach_client(self, owner_id: str) -> bool:
        owner_key = str(owner_id).strip()
        if not owner_key:
            raise ValueError("Ray query runtime owner_id must not be empty")
        result = await self._detach_client_owner(owner_key, expired=False)
        lock = self._client_detach_locks.get(owner_key)
        if lock is not None and not lock.locked():
            self._client_detach_locks.pop(owner_key, None)
        return result

    async def _detach_client_owner(self, owner_key: str, *, expired: bool) -> bool:
        async with self._client_detach_lock_for(owner_key):
            with self._session_lock:
                detached_result = self._detached_client_results.get(owner_key)
                if detached_result is not None:
                    return detached_result
                if owner_key not in self._client_ids:
                    raise PermissionError("Ray query runtime operation requires an attached client owner")
                lease = self._client_leases[owner_key]
                expired_cleanup = expired or lease.state == "expired"
                if expired and lease.state != "expired":
                    return False
                if not expired_cleanup:
                    lease.state = "detaching"
                self._detaching_client_ids.add(owner_key)
                session_ids = [
                    session_id for session_id, session in self._sessions.items() if session.owner_id == owner_key
                ]
            try:
                for session_id in session_ids:
                    await self._close_session_for_owner(
                        owner_key,
                        session_id,
                        cancel_active=expired_cleanup,
                    )
                with self._session_lock:
                    last_owner = len(self._client_ids) == 1
                    if last_owner:
                        with self._plan_runner_lifecycle_lock:
                            self._driver_shutdown_started = True
                    else:
                        self._client_ids.remove(owner_key)
                if last_owner:
                    await self._shutdown_runtime()
                    with self._session_lock:
                        self._client_ids.remove(owner_key)
                self._discard_copy_operation_state_for_owner(owner_key)
            except BaseException as error:
                with self._session_lock:
                    lease = self._client_leases[owner_key]
                    expired_cleanup = expired_cleanup or lease.state == "expired"
                    if expired_cleanup:
                        lease.state = "expired"
                        lease.cleanup_attempts += 1
                        retry_base = max(
                            0.05,
                            min(5.0, float(self._client_heartbeat_interval_s)),
                        )
                        retry_delay = min(
                            30.0,
                            retry_base * (2 ** min(lease.cleanup_attempts - 1, 5)),
                        )
                        lease.next_cleanup_at = time.monotonic() + retry_delay
                        lease.last_cleanup_error = f"{type(error).__name__}: {error}"
                    else:
                        lease.state = "active"
                        lease.expires_at = time.monotonic() + float(self._client_lease_timeout_s)
                        lease.cleanup_attempts = 0
                        lease.next_cleanup_at = 0.0
                        lease.last_cleanup_error = ""
                        self._detaching_client_ids.discard(owner_key)
                if self._driver_shutdown_started and not self._driver_shutdown_complete:
                    # Shutdown stops maintenance before draining the native
                    # runner. If that drain fails, lease expiry still needs a
                    # live reaper so an abandoned retry cannot become immortal.
                    self._start_client_lease_maintenance()
                raise
            with self._session_lock:
                self._detaching_client_ids.discard(owner_key)
                self._client_leases.pop(owner_key, None)
                self._detached_client_results[owner_key] = last_owner
            return last_owner

    async def close_session(self, owner_id: str, session_id: str) -> None:
        self._require_owner(owner_id)
        owner_key = str(owner_id).strip()
        await self._close_session_for_owner(owner_key, session_id)

    async def _close_session_for_owner(
        self,
        owner_key: str,
        session_id: str,
        *,
        cancel_active: bool = False,
    ) -> None:
        session_key = str(session_id).strip()
        if not session_key:
            raise ValueError("Vane session_id must not be empty")
        with self._session_lock:
            session = self._sessions.get(session_key)
            closed_owner = self._closed_session_owners.get(session_key)
            if session is None and closed_owner is None:
                self._closed_session_owners[session_key] = owner_key
                return
        if session is None:
            if closed_owner != owner_key:
                raise PermissionError("Vane session operation requires its owning runtime client")
            return
        if session.owner_id != owner_key:
            raise PermissionError("Vane session operation requires its owning runtime client")
        with session.condition:
            if session.closed:
                return
            session.closing = True
            active_operation_tasks = tuple(session.active_operation_tasks)
        if cancel_active:
            for task in active_operation_tasks:
                task.cancel()

        def _close() -> None:
            with session.condition:
                while session.close_in_progress:
                    session.condition.wait()
                if session.closed:
                    return
                session.close_in_progress = True
            try:
                with session.condition:
                    while session.active_operations > 0:
                        session.condition.wait()
                    plan_ids = tuple(session.plan_ids)
                for plan_id in plan_ids:
                    self._cleanup_finished_plan(plan_id)
                with session.condition:
                    if session.plan_ids:
                        raise RuntimeError(f"Vane session still owns query plans after teardown: {session_key}")
                plan_runner = self.plan_runner
                if plan_runner is not None:
                    plan_runner.close_session(session_key)
                session.connection.close()
                with self._session_lock:
                    current = self._sessions.get(session_key)
                    if current is session:
                        self._closed_session_owners[session_key] = owner_key
                        self._sessions.pop(session_key, None)
                with session.condition:
                    session.closed = True
            finally:
                with session.condition:
                    session.close_in_progress = False
                    session.condition.notify_all()

        await _to_thread_with_owned_side_effects(_close)

    @staticmethod
    def _sum_node_capacity(node_capacities: tuple[Any, ...]) -> Any:
        from duckdb.runners.ray.query_execution_graph import ResourceVector

        total = ResourceVector()
        for node in node_capacities:
            total = total + node.resources
        return total

    def _create_query_resource_coordinator(self) -> Any:
        from duckdb.runners.ray.cluster_resource_coordinator import (
            ClusterQueryResourceCoordinator,
            read_ray_node_capacities,
        )

        object_store_fraction = float(os.environ.get("VANE_QUERY_OBJECT_STORE_FRACTION", "0.5"))
        heap_reserve = int(os.environ.get("VANE_QUERY_HEAP_RESERVE_BYTES_PER_NODE", "0"))
        heartbeat_timeout = float(os.environ.get("VANE_QUERY_HEARTBEAT_TIMEOUT_S", "30"))
        if heartbeat_timeout <= 0:
            raise ValueError("VANE_QUERY_HEARTBEAT_TIMEOUT_S must be positive")
        self._query_resource_heartbeat_timeout_s = heartbeat_timeout
        default_interval = min(5.0, heartbeat_timeout / 3.0)
        maintenance_interval = float(os.environ.get("VANE_QUERY_RESOURCE_REFRESH_INTERVAL_S", str(default_interval)))
        if maintenance_interval <= 0 or maintenance_interval >= heartbeat_timeout:
            raise ValueError(
                "VANE_QUERY_RESOURCE_REFRESH_INTERVAL_S must be positive and less than VANE_QUERY_HEARTBEAT_TIMEOUT_S"
            )
        self._query_resource_maintenance_interval_s = maintenance_interval
        capacities = read_ray_node_capacities(
            ray,
            object_store_fraction=object_store_fraction,
            heap_reserve_bytes_per_node=heap_reserve,
        )
        if not capacities:
            raise RuntimeError("Ray reports no alive nodes with schedulable query resources")
        self._query_node_capacities = capacities
        return ClusterQueryResourceCoordinator(
            capacities,
            heartbeat_timeout_s=heartbeat_timeout,
        )

    def _ensure_query_allocation_teardown_state(self) -> None:
        if not hasattr(self, "_query_allocation_teardowns_pending"):
            self._query_allocation_teardowns_pending = {}
        if not hasattr(self, "_query_allocation_teardowns_claimed"):
            self._query_allocation_teardowns_claimed = set()
        if not hasattr(self, "_query_allocation_teardown_futures"):
            self._query_allocation_teardown_futures = set()

    def _synchronize_query_allocations(
        self,
    ) -> tuple[_DeferredQueryAllocationTeardown, ...]:
        """Commit local allocation state and return teardown work.

        The caller owns ``_query_resource_lock``. Returned work may perform
        remote calls and must run only after that lock is released.
        """
        from duckdb.runners.ray.query_execution_graph import QueryAllocation
        from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

        self._ensure_query_allocation_teardown_state()
        snapshot = self._query_resource_coordinator.snapshot()
        coordinator_queries = snapshot["queries"]
        deferred_teardowns = dict(self._query_allocation_teardowns_pending)
        terminal_errors = getattr(self, "_query_terminal_errors", None)
        if terminal_errors is None:
            terminal_errors = {}
            self._query_terminal_errors = terminal_errors
        for query_id, graph in self._query_graphs.items():
            query_snapshot = coordinator_queries.get(query_id)
            if query_snapshot is None:
                raise RuntimeError(f"coordinator lost registered query {query_id}")
            allocation = QueryAllocation.from_dict(query_snapshot["allocation"])
            graph.validate_allocation(
                allocation,
                require_full_minimum=False,
            )
            current = self._query_allocations[query_id]
            if allocation.generation > current.generation:
                state = str(query_snapshot.get("state") or "")
                placement_changed = (
                    bool(current.actor_placements) and current.actor_placements != allocation.actor_placements
                )
                placement_lost = state == "ACTOR_PLACEMENT_LOST" or placement_changed
                manager = get_query_resource_manager(query_id)
                manager.update_allocation(
                    allocation,
                    admission_open=(state == "RUNNING" and not placement_lost),
                )
                self._query_allocations[query_id] = allocation
                if placement_lost and query_id not in terminal_errors:
                    reason = (
                        f"query {query_id} lost its fixed Ray actor placement; "
                        "the running actor-backed execution cannot migrate in place"
                    )
                    terminal_errors[query_id] = reason
                    manager.cancel("ray_actor_placement_lost")
                    teardown = _DeferredQueryAllocationTeardown(
                        query_id=query_id,
                        generation=allocation.generation,
                        reason=reason,
                    )
                    self._query_allocation_teardowns_pending[query_id] = teardown
                    deferred_teardowns[query_id] = teardown
        return tuple(deferred_teardowns[query_id] for query_id in sorted(deferred_teardowns))

    def _run_query_allocation_teardowns(
        self,
        teardowns: tuple[_DeferredQueryAllocationTeardown, ...],
    ) -> None:
        """Run remote fragment teardown after releasing resource-state locks."""
        self._ensure_query_allocation_teardown_state()
        for teardown in teardowns:
            query_id = teardown.query_id
            with self._query_resource_lock:
                if self._query_allocation_teardowns_pending.get(query_id) != teardown:
                    continue
                if query_id in self._query_allocation_teardowns_claimed:
                    continue
                self._query_allocation_teardowns_claimed.add(query_id)
            succeeded = False
            try:
                self._drop_query_fragments_after_admission_fence_sync(
                    query_id,
                    release_resources=False,
                )
                succeeded = True
            except BaseException as exc:
                with self._query_resource_lock:
                    if (
                        self._query_allocation_teardowns_pending.get(query_id) == teardown
                        and query_id in self._query_graphs
                    ):
                        self._query_terminal_errors[query_id] = (
                            f"{teardown.reason}; fragment cancellation failed: {type(exc).__name__}: {exc}"
                        )
            finally:
                with self._query_resource_lock:
                    self._query_allocation_teardowns_claimed.discard(query_id)
                    if succeeded and self._query_allocation_teardowns_pending.get(query_id) == teardown:
                        self._query_allocation_teardowns_pending.pop(query_id, None)

    def _schedule_query_allocation_teardowns(
        self,
        teardowns: tuple[_DeferredQueryAllocationTeardown, ...],
    ) -> None:
        """Submit teardown work without adding a post-commit cancellation point."""
        if not teardowns:
            return
        self._ensure_query_allocation_teardown_state()
        future = asyncio.get_running_loop().run_in_executor(
            None,
            self._run_query_allocation_teardowns,
            teardowns,
        )
        self._query_allocation_teardown_futures.add(future)

        def _retire(done: asyncio.Future[Any]) -> None:
            self._query_allocation_teardown_futures.discard(done)
            if not done.cancelled():
                done.exception()

        future.add_done_callback(_retire)

    def _apply_query_capacity_snapshot(
        self,
        capacities: tuple[Any, ...],
        *,
        snapshot_started_at: float,
    ) -> Any:
        """Apply a complete GCS snapshot while the caller owns the state lock.

        Request start time orders overlapping reads so a slow older request
        cannot overwrite a later request that already committed.
        """
        if not capacities:
            raise RuntimeError("Ray reports no alive nodes with schedulable query resources")
        last_snapshot_started_at = float(getattr(self, "_query_resource_last_capacity_refresh_at", 0.0))
        cached_capacities = tuple(getattr(self, "_query_node_capacities", ()))
        if cached_capacities and float(snapshot_started_at) < last_snapshot_started_at:
            return self._sum_node_capacity(cached_capacities)
        self._query_resource_coordinator.update_node_capacities(capacities)
        self._query_node_capacities = capacities
        self._query_resource_last_capacity_refresh_at = float(snapshot_started_at)
        return self._sum_node_capacity(capacities)

    @staticmethod
    def _read_query_node_capacities() -> tuple[Any, ...]:
        from duckdb.runners.ray.cluster_resource_coordinator import read_ray_node_capacities

        return read_ray_node_capacities(
            ray,
            object_store_fraction=float(os.environ.get("VANE_QUERY_OBJECT_STORE_FRACTION", "0.5")),
            heap_reserve_bytes_per_node=int(os.environ.get("VANE_QUERY_HEAP_RESERVE_BYTES_PER_NODE", "0")),
        )

    def _maintain_query_resources_once(
        self,
        *,
        capacities: tuple[Any, ...] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Refresh Ray capacity, query usage, allocations, and heartbeats once.

        Capacity discovery happens before the driver lock.  If the GCS query is
        transiently unavailable, the last complete Ray snapshot remains valid
        for this cycle while active query heartbeats and usage still advance.
        """
        from duckdb.runners.ray.query_execution_graph import ResourceVector
        from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

        timestamp = time.monotonic() if now is None else float(now)
        capacity_error: BaseException | None = None
        if capacities is None:
            try:
                capacities = self._read_query_node_capacities()
            except BaseException as exc:
                capacity_error = exc
                capacities = ()
        capacities = tuple(capacities)
        if capacity_error is None and not capacities:
            raise RuntimeError("Ray reports no alive nodes with schedulable query resources")

        deferred_teardowns: list[_DeferredQueryAllocationTeardown] = []
        try:
            with self._query_resource_lock:
                if capacity_error is None:
                    self._apply_query_capacity_snapshot(
                        capacities,
                        snapshot_started_at=timestamp,
                    )
                else:
                    cached_capacities = tuple(self._query_node_capacities)
                    if not cached_capacities:
                        raise RuntimeError(
                            f"failed to refresh Ray capacity and no cached snapshot exists: {capacity_error}"
                        ) from capacity_error
                    self._query_resource_coordinator.update_node_capacities(
                        cached_capacities,
                        now=timestamp,
                    )
                deferred_teardowns.extend(self._synchronize_query_allocations())

                observed_usage: dict[str, ResourceVector] = {}
                generations: dict[str, int] = {}
                for query_id in sorted(self._query_graphs):
                    manager = get_query_resource_manager(query_id)
                    observed_usage[query_id] = ResourceVector.from_dict(manager.snapshot()["usage"])
                    generations[query_id] = self._query_allocations[query_id].generation
                self._query_resource_coordinator.refresh_queries(
                    observed_usage_by_query=observed_usage,
                    generations=generations,
                    now=timestamp,
                )
                expired = self._query_resource_coordinator.expire_queries(now=timestamp)
                if expired:
                    raise RuntimeError("active query resource heartbeats expired unexpectedly: " + ", ".join(expired))
                deferred_teardowns.extend(self._synchronize_query_allocations())
                self._query_resource_last_maintenance_at = timestamp
                result = {
                    "query_count": len(observed_usage),
                    "capacity_cached": capacity_error is not None,
                    "capacity_error": "" if capacity_error is None else str(capacity_error),
                }
        finally:
            self._run_query_allocation_teardowns(tuple(deferred_teardowns))
        return result

    def _start_query_resource_maintenance(self) -> None:
        task = self._query_resource_maintenance_task
        if task is not None and not task.done():
            return
        loop = asyncio.get_running_loop()
        self._query_resource_maintenance_stop = asyncio.Event()
        self._query_resource_maintenance_task = loop.create_task(
            self._query_resource_maintenance_loop(),
            name="vane-query-resource-maintenance",
        )

    async def _query_resource_maintenance_loop(self) -> None:
        stop = self._query_resource_maintenance_stop
        if stop is None:
            raise RuntimeError("query resource maintenance stop event is not initialized")
        interval = float(self._query_resource_maintenance_interval_s)
        while not stop.is_set():
            try:
                await asyncio.to_thread(self._maintain_query_resources_once)
                self._query_resource_maintenance_error = ""
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                self._query_resource_maintenance_failures += 1
                self._query_resource_maintenance_error = f"{type(exc).__name__}: {exc}"
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except (TimeoutError, asyncio.TimeoutError):
                continue

    async def stop_query_resource_maintenance(self) -> None:
        stop = self._query_resource_maintenance_stop
        task = self._query_resource_maintenance_task
        if stop is not None:
            stop.set()
        if task is not None and task is not asyncio.current_task():
            await task
        self._query_resource_maintenance_task = None

    def _ensure_query_resource_admission_state(self) -> None:
        """Initialize serialized admission state for construction and test shells."""
        if not hasattr(self, "_query_task_lease_requests"):
            self._query_task_lease_requests = {}
        if not hasattr(self, "_query_output_lease_requests"):
            self._query_output_lease_requests = {}
        if not hasattr(self, "_query_task_lease_request_tombstones"):
            self._query_task_lease_request_tombstones = BoundedReplayMap(
                capacity=_LEASE_REQUEST_REPLAY_CAPACITY,
            )
        if not hasattr(self, "_query_task_terminal_controls"):
            self._query_task_terminal_controls = BoundedReplayMap(
                capacity=_LEASE_REQUEST_REPLAY_CAPACITY,
            )
        if not hasattr(self, "_query_task_terminal_attempts"):
            self._query_task_terminal_attempts = BoundedReplayMap(
                capacity=_LEASE_REQUEST_REPLAY_CAPACITY,
            )
        if not hasattr(self, "_query_output_lease_request_tombstones"):
            self._query_output_lease_request_tombstones = BoundedReplayMap(
                capacity=_LEASE_REQUEST_REPLAY_CAPACITY,
            )
        if not hasattr(self, "_query_output_terminal_controls"):
            self._query_output_terminal_controls = BoundedReplayMap(
                capacity=_LEASE_REQUEST_REPLAY_CAPACITY,
            )
        if not hasattr(self, "_query_lease_control_key"):
            self._query_lease_control_key = os.urandom(32)
        if not hasattr(self, "_query_lease_generation_ids"):
            self._query_lease_generation_ids = {}
        if not hasattr(self, "_query_task_request_owner_by_identity"):
            self._query_task_request_owner_by_identity = {}
        if not hasattr(self, "_query_output_request_owner_by_identity"):
            self._query_output_request_owner_by_identity = {}
        if not hasattr(self, "_query_resource_closing_queries"):
            self._query_resource_closing_queries = set()
        if not hasattr(self, "_query_execution_quiesced_queries"):
            self._query_execution_quiesced_queries = set()
        if not hasattr(self, "_query_task_admission_pumps"):
            self._query_task_admission_pumps = set()
        if not hasattr(self, "_query_task_lease_condition"):
            self._query_task_lease_condition = threading.Condition()
        if not hasattr(self, "_query_task_completion_waiters"):
            self._query_task_completion_waiters = {}
        if not hasattr(self, "_query_terminal_errors"):
            self._query_terminal_errors = {}
        if not hasattr(self, "_query_output_admission_pumps"):
            self._query_output_admission_pumps = set()
        if not hasattr(self, "_query_fte_admission_pumps"):
            self._query_fte_admission_pumps = {}
        if not hasattr(self, "_query_fte_admission_dirty_queries"):
            self._query_fte_admission_dirty_queries = set()
        if not hasattr(self, "_query_fte_admission_done_events"):
            self._query_fte_admission_done_events = {}
        if not hasattr(self, "_query_resource_admission_signal_lock"):
            self._query_resource_admission_signal_lock = threading.Lock()
        if not hasattr(self, "_query_resource_admission_dirty_queries"):
            self._query_resource_admission_dirty_queries = set()
        if not hasattr(self, "_query_resource_admission_signal_scheduled"):
            self._query_resource_admission_signal_scheduled = False
        if not hasattr(self, "_query_resource_admission_loop"):
            try:
                self._query_resource_admission_loop = asyncio.get_running_loop()
            except RuntimeError:
                self._query_resource_admission_loop = None
        if not hasattr(self, "_query_resource_admission_fence_lock"):
            self._query_resource_admission_fence_lock = threading.Lock()
        if not hasattr(self, "_query_resource_admission_uncertain_fences"):
            self._query_resource_admission_uncertain_fences = {}

    def _settle_query_resource_admission_fence(self, query_id: str) -> None:
        query_key = str(query_id or "").strip()
        if not query_key:
            raise ValueError("query admission owner-loop fence requires a query_id")
        with self._query_resource_admission_fence_lock:
            fence = self._query_resource_admission_uncertain_fences.get(query_key)
            if fence is None:
                return
            if not fence.completed.is_set():
                raise QueryTeardownOwnershipError(f"query admission owner-loop fence is still settling for {query_key}")
            self._query_resource_admission_uncertain_fences.pop(query_key, None)
        if fence.error is not None:
            raise QueryTeardownOwnershipError(
                f"late query admission owner-loop mutation failed for {query_key}: "
                f"{type(fence.error).__name__}: {fence.error}"
            ) from fence.error

    def _run_on_query_resource_admission_loop_sync(
        self,
        callback: Callable[[], None],
        *,
        query_id: str,
        timeout_s: float = 30.0,
    ) -> None:
        """Run one query-scoped admission mutation on its owner loop."""
        self._ensure_query_resource_admission_state()
        query_key = str(query_id or "").strip()
        self._settle_query_resource_admission_fence(query_key)
        loop = self._query_resource_admission_loop
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            callback()
            return
        if loop is None or loop.is_closed() or not loop.is_running():
            raise RuntimeError("query admission owner loop is not running")

        fence = _AdmissionLoopFence()
        invocation_lock = threading.Lock()
        invocation_state = "queued"

        def invoke() -> None:
            nonlocal invocation_state
            with invocation_lock:
                if invocation_state == "cancelled":
                    fence.completed.set()
                    return
                invocation_state = "started"
            try:
                callback()
            except BaseException as exc:
                fence.error = exc
            finally:
                with invocation_lock:
                    invocation_state = "finished"
                fence.completed.set()

        loop.call_soon_threadsafe(invoke)
        if not fence.completed.wait(max(0.001, float(timeout_s))):
            with invocation_lock:
                if invocation_state == "queued":
                    invocation_state = "cancelled"
                    timeout_message = f"timed out waiting for query admission owner-loop fence for {query_key}"
                elif invocation_state == "finished":
                    timeout_message = ""
                else:
                    # A started callback cannot be revoked safely. Fence only
                    # this query generation until the callback settles; other
                    # queries keep using the shared actor.
                    with self._query_resource_admission_fence_lock:
                        self._query_resource_admission_uncertain_fences[query_key] = fence
                    timeout_message = (
                        f"query admission owner-loop callback started but did not finish for {query_key}; "
                        "query generation remains fenced"
                    )
            if timeout_message:
                raise QueryTeardownOwnershipError(timeout_message)
        if fence.error is not None:
            raise fence.error

    def _discard_query_admission_replay(self, query_id: str) -> None:
        """Drop bounded replay caches that must never cross a query generation."""
        query_key = str(query_id)
        query_fingerprint = self._query_fingerprint(query_key)
        self._query_task_lease_request_tombstones.discard_where(
            lambda _request_id, state: state.get("identity", {}).get("query_id") == query_key
        )
        self._query_output_lease_request_tombstones.discard_where(
            lambda _request_id, state: state.get("identity", {}).get("query_id") == query_key
        )
        self._query_task_terminal_controls.discard_where(
            lambda _request_fingerprint, control: control.query_fingerprint == query_fingerprint
        )
        self._query_task_terminal_attempts.discard_where(
            lambda _attempt_fingerprint, owner_query_fingerprint: owner_query_fingerprint == query_fingerprint
        )
        self._query_output_terminal_controls.discard_where(
            lambda _request_fingerprint, control: control.query_fingerprint == query_fingerprint
        )

    def _close_query_resource_admission(self, query_id: str) -> None:
        """Fence new requests and resolve all old requests on the owner loop."""
        from duckdb.runners.ray.fte_fragment_scheduler import close_fte_registry_for_query
        from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

        self._ensure_query_resource_admission_state()
        query_key = str(query_id)
        close_fte_registry_for_query(query_key)
        try:
            get_query_resource_manager(query_key).close_admission()
        except KeyError:
            pass
        self._query_resource_closing_queries.add(query_key)
        self._fail_query_admission_requests(query_key)
        # Granted task records are the execution ledger. Keep them until their
        # terminal RPC releases the physical lease. Native query teardown cannot
        # prove that a dispatcher call which crossed its bounded unregister
        # deadline did not submit remote work afterward.
        self._query_output_lease_requests = {
            request_id: state
            for request_id, state in self._query_output_lease_requests.items()
            if state.get("identity", {}).get("query_id") != query_key
        }
        self._discard_query_admission_replay(query_key)
        self._query_output_request_owner_by_identity = {
            identity: request_id
            for identity, request_id in self._query_output_request_owner_by_identity.items()
            if identity[0] != query_key
        }
        self._query_task_admission_pumps.discard(query_key)
        self._query_output_admission_pumps.discard(query_key)
        self._query_fte_admission_dirty_queries.discard(query_key)
        with self._query_resource_admission_signal_lock:
            self._query_resource_admission_dirty_queries.discard(query_key)

    def _open_query_resource_admission(self, query_id: str) -> None:
        """Open a newly registered generation after the old generation is gone."""
        from duckdb.runners.ray.fte_fragment_scheduler import open_fte_registry_for_query

        self._ensure_query_resource_admission_state()
        query_key = str(query_id)
        self._settle_query_resource_admission_fence(query_key)
        old_state_exists = any(
            state.get("identity", {}).get("query_id") == query_key
            for table in (
                self._query_task_lease_requests,
                self._query_output_lease_requests,
            )
            for state in table.values()
        )
        old_owner_exists = any(
            identity[0] == query_key
            for owners in (
                self._query_task_request_owner_by_identity,
                self._query_output_request_owner_by_identity,
            )
            for identity in owners
        )
        with self._query_task_lease_condition:
            old_completion_waiter_exists = any(
                owner_query_id == query_key for owner_query_id, _task in self._query_task_completion_waiters.values()
            )
            old_execution_quiesced = query_key in self._query_execution_quiesced_queries
        if old_state_exists or old_owner_exists or old_completion_waiter_exists or old_execution_quiesced:
            raise RuntimeError(f"cannot reopen query admission with old generation state: {query_key}")
        fte_pump = self._query_fte_admission_pumps.get(query_key)
        if fte_pump is not None and not fte_pump.done():
            raise RuntimeError(f"cannot reopen query admission while old FTE pump is active: {query_key}")
        self._query_fte_admission_pumps.pop(query_key, None)
        self._query_fte_admission_done_events.pop(query_key, None)
        self._query_fte_admission_dirty_queries.discard(query_key)
        self._discard_query_admission_replay(query_key)
        open_fte_registry_for_query(query_key)
        self._query_lease_generation_ids[query_key] = uuid.uuid4().hex
        self._query_resource_closing_queries.discard(query_key)

    def _signal_query_resource_change(self, query_id: str) -> None:
        """Coalesce a resource mutation into one task and output pump run.

        This callback is intentionally Ray-free and wait-free because resource
        managers may invoke it while holding their internal accounting lock.
        """
        self._ensure_query_resource_admission_state()
        query_key = str(query_id)
        loop = self._query_resource_admission_loop
        if loop is None or loop.is_closed() or not loop.is_running():
            return
        with self._query_resource_admission_signal_lock:
            self._query_resource_admission_dirty_queries.add(query_key)
            if self._query_resource_admission_signal_scheduled:
                return
            self._query_resource_admission_signal_scheduled = True
        try:
            loop.call_soon_threadsafe(self._drain_query_resource_admission_signals)
        except RuntimeError:
            with self._query_resource_admission_signal_lock:
                self._query_resource_admission_signal_scheduled = False
            raise

    def _drain_query_resource_admission_signals(self) -> None:
        with self._query_resource_admission_signal_lock:
            query_ids = tuple(sorted(self._query_resource_admission_dirty_queries))
            self._query_resource_admission_dirty_queries.clear()
            self._query_resource_admission_signal_scheduled = False
        _log_resource_debug(
            "signal_batch",
            query_count=len(query_ids),
        )
        for query_id in query_ids:
            self._schedule_query_task_admission_pump(query_id)
            self._schedule_query_output_admission_pump(query_id)
            self._schedule_query_fte_admission_pump(query_id)

    def _schedule_query_fte_admission_pump(self, query_id: str) -> None:
        """Wake the FTE ownership domain without blocking the actor loop."""
        from duckdb.runners.ray.fte_fragment_scheduler import (
            has_fte_resource_admission_waiter,
        )

        query_key = str(query_id)
        if query_key in self._query_resource_closing_queries:
            return
        if not has_fte_resource_admission_waiter(query_key):
            return
        self._query_fte_admission_dirty_queries.add(query_key)
        existing = self._query_fte_admission_pumps.get(query_key)
        if existing is not None and not existing.done():
            return
        loop = self._query_resource_admission_loop
        if loop is None or loop.is_closed():
            raise RuntimeError("query FTE admission pump has no live event loop")
        done = threading.Event()
        self._query_fte_admission_done_events[query_key] = done
        task = loop.create_task(
            self._run_query_fte_admission_pump(query_key, done),
            name=f"vane-query-fte-admission:{query_key}",
        )
        self._query_fte_admission_pumps[query_key] = task

    async def _run_query_fte_admission_pump(
        self,
        query_id: str,
        done: threading.Event,
    ) -> None:
        from duckdb.runners.ray.fte_fragment_scheduler import (
            drain_fte_resource_admission_change,
        )

        query_key = str(query_id)
        current = asyncio.current_task()
        try:
            while (
                query_key in self._query_fte_admission_dirty_queries
                and query_key not in self._query_resource_closing_queries
            ):
                self._query_fte_admission_dirty_queries.discard(query_key)
                await asyncio.to_thread(
                    drain_fte_resource_admission_change,
                    query_key,
                )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._query_fte_admission_dirty_queries.discard(query_key)
            message = f"FTE resource admission wake failed for {query_key}: {type(exc).__name__}: {exc}"
            self._query_terminal_errors.setdefault(query_key, message)
            try:
                from duckdb.runners.ray.query_resource_runtime import (
                    get_query_resource_manager,
                )

                get_query_resource_manager(query_key).cancel(message)
            except KeyError:
                pass
            self._fail_query_admission_requests(query_key)
        finally:
            if self._query_fte_admission_pumps.get(query_key) is current:
                self._query_fte_admission_pumps.pop(query_key, None)
            done.set()
            if (
                query_key in self._query_fte_admission_dirty_queries
                and query_key not in self._query_resource_closing_queries
            ):
                self._schedule_query_fte_admission_pump(query_key)

    def _wait_for_query_fte_admission_pump(
        self,
        query_id: str,
        *,
        timeout_s: float = 30.0,
    ) -> None:
        done = self._query_fte_admission_done_events.get(str(query_id))
        if done is None or done.is_set():
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self._query_resource_admission_loop:
            raise RuntimeError(
                f"cannot synchronously fence active query FTE admission pump on its owner loop: {query_id}"
            )
        if not done.wait(max(0.001, float(timeout_s))):
            raise RuntimeError(f"timed out waiting for query FTE admission pump: {query_id}")

    def _purge_completed_query_fte_admission_pump(self, query_id: str) -> None:
        query_key = str(query_id)
        task = self._query_fte_admission_pumps.get(query_key)
        if task is not None and not task.done():
            raise RuntimeError(f"cannot purge active query FTE admission pump: {query_key}")
        self._query_fte_admission_pumps.pop(query_key, None)
        self._query_fte_admission_done_events.pop(query_key, None)
        self._query_fte_admission_dirty_queries.discard(query_key)

    @staticmethod
    def _complete_lease_request(state: dict[str, Any], result: dict[str, Any]) -> None:
        future = state.get("future")
        if future is None:
            return

        def _complete() -> None:
            if not future.done():
                future.set_result(result)

        target_loop = future.get_loop()
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is target_loop:
            _complete()
        elif not target_loop.is_closed():
            target_loop.call_soon_threadsafe(_complete)

    @staticmethod
    def _task_identity_fingerprint(identity: dict[str, Any]) -> bytes:
        encoded = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).digest()

    @staticmethod
    def _task_request_fingerprint(request_id: str) -> bytes:
        return hashlib.sha256(str(request_id).encode("utf-8")).digest()

    @classmethod
    def _task_attempt_fingerprint(cls, identity: dict[str, Any]) -> bytes:
        return cls._task_identity_fingerprint(
            {
                "query_id": str(identity["query_id"]),
                "task_id": str(identity["task_id"]),
                "attempt_id": str(identity["attempt_id"]),
            }
        )

    @staticmethod
    def _task_resource_identity(identity: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(identity["query_id"]),
            str(identity["task_id"]),
            str(identity["attempt_id"]),
        )

    def _store_task_terminal_control(
        self,
        *,
        request_id: str,
        identity: dict[str, Any],
        status: str,
        result: dict[str, Any],
    ) -> None:
        request_key = str(request_id)
        request_fingerprint = self._task_request_fingerprint(request_key)
        control = _TaskTerminalControl(
            identity_fingerprint=self._task_identity_fingerprint(identity),
            query_fingerprint=self._query_fingerprint(
                str(identity["query_id"]),
            ),
            status=str(status),
            blocked_reason=str(result.get("blocked_reason") or "task_request_terminal"),
        )
        existing = self._query_task_terminal_controls.get(request_fingerprint)
        if existing is not None and existing != control:
            raise RuntimeError(f"task terminal control identity changed: {request_key}")
        self._query_task_terminal_controls[request_fingerprint] = control
        if str(status) != "failed":
            self._query_task_terminal_attempts[self._task_attempt_fingerprint(identity)] = control.query_fingerprint

    def _task_terminal_control_for_identity(
        self,
        request_id: str,
        identity: dict[str, Any],
    ) -> _TaskTerminalControl | None:
        control = self._query_task_terminal_controls.get(self._task_request_fingerprint(request_id))
        if control is None:
            return None
        if control.identity_fingerprint != self._task_identity_fingerprint(identity):
            raise ValueError(f"task lease request_id reused with different identity: {request_id}")
        return control

    def _query_lease_generation_id(self, query_id: str) -> str:
        self._ensure_query_resource_admission_state()
        query_key = str(query_id)
        generation_id = self._query_lease_generation_ids.get(query_key)
        if generation_id is None:
            generation_id = uuid.uuid4().hex
            self._query_lease_generation_ids[query_key] = generation_id
        return generation_id

    def _capability_targets_inactive_query_generation(
        self,
        query_id: str,
        generation_id: str,
    ) -> bool:
        active_generation_id = self._query_lease_generation_ids.get(str(query_id))
        return active_generation_id != str(generation_id)

    def _issue_lease_capability(
        self,
        *,
        kind: str,
        fields: tuple[str, ...],
    ) -> str:
        self._ensure_query_resource_admission_state()
        payload = json.dumps(
            [f"vane-{kind}-lease-v1", *fields],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        encoded = base64.urlsafe_b64encode(payload).rstrip(b"=")
        signature = hmac.new(
            self._query_lease_control_key,
            encoded,
            hashlib.sha256,
        ).hexdigest()
        return f"v1.{encoded.decode('ascii')}.{signature}"

    def _decode_lease_capability(
        self,
        lease_id: str,
        *,
        kind: str,
        field_count: int,
    ) -> tuple[str, ...] | None:
        self._ensure_query_resource_admission_state()
        try:
            version, encoded_text, supplied_signature = str(lease_id).split(".", 2)
            if version != "v1":
                return None
            encoded = encoded_text.encode("ascii")
            expected_signature = hmac.new(
                self._query_lease_control_key,
                encoded,
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(supplied_signature, expected_signature):
                return None
            padding = b"=" * (-len(encoded) % 4)
            values = json.loads(base64.urlsafe_b64decode(encoded + padding).decode("utf-8"))
        except (UnicodeError, ValueError, TypeError, binascii.Error):
            return None
        if not isinstance(values, list) or len(values) != field_count + 1 or values[0] != f"vane-{kind}-lease-v1":
            return None
        fields = tuple(str(value) for value in values[1:])
        if any(not value for value in fields):
            return None
        return fields

    def _issue_query_task_admission_capability(self, query_id: str) -> str:
        query_key = str(query_id)
        return self._issue_lease_capability(
            kind="task-admission",
            fields=(
                query_key,
                self._query_lease_generation_id(query_key),
            ),
        )

    def _verify_query_task_admission_capability(
        self,
        *,
        capability: str,
        query_id: str,
    ) -> str | None:
        fields = self._decode_lease_capability(
            capability,
            kind="task-admission",
            field_count=2,
        )
        if fields is None:
            return None
        capability_query_id, capability_generation_id = fields
        if capability_query_id != str(query_id):
            return None
        return capability_generation_id

    def _issue_task_lease_capability(
        self,
        *,
        request_id: str,
        manager_lease_id: str,
        attempt_id: str,
        query_id: str,
    ) -> str:
        query_key = str(query_id)
        return self._issue_lease_capability(
            kind="task",
            fields=(
                str(request_id),
                str(manager_lease_id),
                str(attempt_id),
                query_key,
                self._query_lease_generation_id(query_key),
            ),
        )

    def _verify_task_lease_capability(
        self,
        *,
        lease_id: str,
        request_id: str = "",
        attempt_id: str = "",
        query_id: str = "",
    ) -> tuple[str, str, str, str, str] | None:
        fields = self._decode_lease_capability(
            lease_id,
            kind="task",
            field_count=5,
        )
        if fields is None:
            return None
        (
            capability_request_id,
            manager_lease_id,
            capability_attempt_id,
            capability_query_id,
            capability_generation_id,
        ) = fields
        if (
            (request_id and capability_request_id != str(request_id))
            or (attempt_id and capability_attempt_id != str(attempt_id))
            or (query_id and capability_query_id != str(query_id))
        ):
            return None
        return (
            manager_lease_id,
            capability_attempt_id,
            capability_query_id,
            capability_request_id,
            capability_generation_id,
        )

    @staticmethod
    def _output_identity_fingerprint(identity: dict[str, Any]) -> bytes:
        encoded = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).digest()

    @staticmethod
    def _output_request_fingerprint(request_id: str) -> bytes:
        return hashlib.sha256(str(request_id).encode("utf-8")).digest()

    @staticmethod
    def _query_fingerprint(query_id: str) -> bytes:
        return hashlib.sha256(str(query_id).encode("utf-8")).digest()

    def _issue_output_lease_capability(
        self,
        *,
        request_id: str,
        manager_lease_id: str,
        query_id: str,
    ) -> str:
        query_key = str(query_id)
        return self._issue_lease_capability(
            kind="output",
            fields=(
                str(request_id),
                str(manager_lease_id),
                query_key,
                self._query_lease_generation_id(query_key),
            ),
        )

    def _verify_output_lease_capability(
        self,
        *,
        request_id: str,
        lease_id: str,
        query_id: str = "",
    ) -> tuple[str, str, str] | None:
        fields = self._decode_lease_capability(
            lease_id,
            kind="output",
            field_count=4,
        )
        if fields is None:
            return None
        (
            capability_request_id,
            manager_lease_id,
            capability_query_id,
            capability_generation_id,
        ) = fields
        if capability_request_id != str(request_id) or (query_id and capability_query_id != str(query_id)):
            return None
        return manager_lease_id, capability_query_id, capability_generation_id

    def _store_output_terminal_control(
        self,
        *,
        request_id: str,
        identity: dict[str, Any],
        status: str,
        result: dict[str, Any],
    ) -> None:
        request_key = str(request_id)
        request_fingerprint = self._output_request_fingerprint(request_key)
        control = _OutputTerminalControl(
            identity_fingerprint=self._output_identity_fingerprint(identity),
            query_fingerprint=self._query_fingerprint(
                str(identity["query_id"]),
            ),
            status=str(status),
            blocked_reason=str(result.get("blocked_reason") or "output_request_terminal"),
        )
        existing = self._query_output_terminal_controls.get(request_fingerprint)
        if existing is not None and existing != control:
            raise RuntimeError(f"output terminal control identity changed: {request_key}")
        self._query_output_terminal_controls[request_fingerprint] = control

    def _output_terminal_control_for_identity(
        self,
        request_id: str,
        identity: dict[str, Any],
    ) -> _OutputTerminalControl | None:
        control = self._query_output_terminal_controls.get(self._output_request_fingerprint(request_id))
        if control is None:
            return None
        if control.identity_fingerprint != self._output_identity_fingerprint(identity):
            raise ValueError(f"output lease request_id reused with different identity: {request_id}")
        return control

    def _retire_query_lease_request(
        self,
        *,
        request_id: str,
        state: dict[str, Any],
        result: dict[str, Any],
        status: str,
        active: dict[str, dict[str, Any]],
        tombstones: BoundedReplayMap[str, dict[str, Any]],
    ) -> None:
        """Publish one terminal result and retain only compact replay state."""
        terminal_result = dict(result)
        state["status"] = str(status)
        self._complete_lease_request(state, terminal_result)
        active.pop(str(request_id), None)
        tombstone = {
            "identity": dict(state["identity"]),
            "status": str(status),
            "result": terminal_result,
        }
        lease = state.get("lease")
        if isinstance(lease, dict):
            tombstone["lease"] = dict(lease)
        tombstones[str(request_id)] = tombstone
        identity = state["identity"]
        if "block_id" in identity:
            self._store_output_terminal_control(
                request_id=str(request_id),
                identity=dict(identity),
                status=str(status),
                result=terminal_result,
            )
            output_owner_key = (str(identity["query_id"]), str(identity["block_id"]))
            output_owners = self._query_output_request_owner_by_identity
            if output_owners.get(output_owner_key) == str(request_id):
                output_owners.pop(output_owner_key, None)
        else:
            self._store_task_terminal_control(
                request_id=str(request_id),
                identity=dict(identity),
                status=str(status),
                result=terminal_result,
            )
            task_owner_key = (
                str(identity["query_id"]),
                str(identity["task_id"]),
                str(identity["attempt_id"]),
            )
            task_owners = self._query_task_request_owner_by_identity
            if task_owners.get(task_owner_key) == str(request_id):
                task_owners.pop(task_owner_key, None)

    def _fail_query_admission_requests(self, query_id: str) -> None:
        """Resolve every pending request before its query manager disappears."""
        from duckdb.runners.ray.query_resource_manager import (
            OutputBlockGrant,
            TaskGrant,
        )

        query_key = str(query_id)
        denials = (
            (
                self._query_task_lease_requests,
                self._query_task_lease_request_tombstones,
                self._grant_payload(
                    TaskGrant(
                        False,
                        blocked_reason="query_not_registered",
                        fatal=True,
                    )
                ),
            ),
            (
                self._query_output_lease_requests,
                self._query_output_lease_request_tombstones,
                self._grant_payload(
                    OutputBlockGrant(
                        False,
                        blocked_reason="query_not_registered",
                        fatal=True,
                    )
                ),
            ),
        )
        for requests, tombstones, denial in denials:
            for request_id, state in list(requests.items()):
                if state.get("status") == "pending" and state.get("identity", {}).get("query_id") == query_key:
                    self._retire_query_lease_request(
                        request_id=request_id,
                        state=state,
                        result=denial,
                        status="failed",
                        active=requests,
                        tombstones=tombstones,
                    )

    def _schedule_query_task_admission_pump(self, query_id: str) -> None:
        self._ensure_query_resource_admission_state()
        query_key = str(query_id)
        if query_key in self._query_task_admission_pumps:
            return
        self._query_task_admission_pumps.add(query_key)
        loop = self._query_resource_admission_loop
        if loop is None or loop.is_closed():
            self._query_task_admission_pumps.discard(query_key)
            raise RuntimeError("query task admission pump has no live event loop")
        loop.call_soon(self._run_query_task_admission_pump, query_key)

    def _run_query_task_admission_pump(self, query_id: str) -> None:
        from duckdb.runners.ray.query_resource_manager import TaskGrant
        from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

        query_key = str(query_id)
        try:
            try:
                manager = get_query_resource_manager(query_key)
            except KeyError:
                denial = self._grant_payload(TaskGrant(False, blocked_reason="query_not_registered", fatal=True))
                for request_id, state in list(self._query_task_lease_requests.items()):
                    if state.get("status") == "pending" and state.get("identity", {}).get("query_id") == query_key:
                        self._retire_query_lease_request(
                            request_id=request_id,
                            state=state,
                            result=denial,
                            status="failed",
                            active=self._query_task_lease_requests,
                            tombstones=self._query_task_lease_request_tombstones,
                        )
                return

            pending_by_key: dict[
                tuple[str, str],
                tuple[str, dict[str, Any]],
            ] = {}
            for request_id, state in self._query_task_lease_requests.items():
                request = state.get("request")
                if state.get("status") != "pending" or request is None or str(request.query_id) != query_key:
                    continue
                key = (str(request.task_id), str(request.attempt_id))
                pending_by_key.setdefault(key, (request_id, state))
            if not pending_by_key:
                return
            request, grant = manager.try_acquire_next_queued_task(set(pending_by_key))
            if request is None or grant is None:
                return
            selected_key = (str(request.task_id), str(request.attempt_id))
            owned_request = pending_by_key.get(selected_key)
            if owned_request is None:
                _log_resource_debug(
                    "task_admission_yield",
                    query_id=query_key,
                    stage_id=request.stage_id,
                    task_id=request.task_id,
                    reason=grant.blocked_reason,
                )
                return
            request_id, state = owned_request
            if grant.granted:
                lease_payload = asdict(grant.lease)
                manager_lease_id = str(lease_payload["lease_id"])
                public_lease_id = self._issue_task_lease_capability(
                    request_id=request_id,
                    manager_lease_id=manager_lease_id,
                    attempt_id=lease_payload["attempt_id"],
                    query_id=lease_payload["query_id"],
                )
                lease_payload["lease_id"] = public_lease_id
                if lease_payload.get("actor_index") is None:
                    # A task execution slot is a public per-lease identity.
                    # Keep the QRM's raw lease ID from being used as an
                    # authoritative external control token and bind the
                    # worker-visible slot to the signed capability.
                    lease_payload["execution_slot_id"] = f"ray_task:{lease_payload['stage_id']}:{public_lease_id}"
                # This request record is both the admission result and the
                # durable execution owner. Publish it before completing the
                # caller's Future so teardown can never miss a visible grant.
                with self._query_task_lease_condition:
                    state["manager_lease_id"] = manager_lease_id
                    state["lease"] = lease_payload
                    state["status"] = "granted"
                    self._query_task_lease_condition.notify_all()
                _log_resource_debug(
                    "task_granted",
                    query_id=request.query_id,
                    stage_id=request.stage_id,
                    request_id=request_id,
                    lease_id=lease_payload["lease_id"],
                )
                result = self._grant_payload(grant)
                result["lease"] = lease_payload
                self._complete_lease_request(
                    state,
                    result,
                )
                return
            if grant.fatal:
                manager.remove_task_waiter(request.task_id, request.attempt_id)
                self._retire_query_lease_request(
                    request_id=request_id,
                    state=state,
                    result=self._grant_payload(grant),
                    status="failed",
                    active=self._query_task_lease_requests,
                    tombstones=self._query_task_lease_request_tombstones,
                )
                return
            _log_resource_debug(
                "task_blocked",
                query_id=request.query_id,
                stage_id=request.stage_id,
                request_id=request_id,
                reason=grant.blocked_reason,
            )
        finally:
            self._query_task_admission_pumps.discard(query_key)

    def _schedule_query_output_admission_pump(self, query_id: str) -> None:
        self._ensure_query_resource_admission_state()
        query_key = str(query_id)
        if query_key in self._query_output_admission_pumps:
            return
        self._query_output_admission_pumps.add(query_key)
        loop = self._query_resource_admission_loop
        if loop is None or loop.is_closed():
            self._query_output_admission_pumps.discard(query_key)
            raise RuntimeError("query output admission pump has no live event loop")
        loop.call_soon(self._run_query_output_admission_pump, query_key)

    def _run_query_output_admission_pump(self, query_id: str) -> None:
        from duckdb.runners.ray.query_resource_manager import OutputBlockGrant
        from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

        query_key = str(query_id)
        try:
            try:
                manager = get_query_resource_manager(query_key)
            except KeyError:
                denial = self._grant_payload(
                    OutputBlockGrant(
                        False,
                        blocked_reason="query_not_registered",
                        fatal=True,
                    )
                )
                for request_id, state in list(self._query_output_lease_requests.items()):
                    if state.get("status") == "pending" and state.get("identity", {}).get("query_id") == query_key:
                        self._retire_query_lease_request(
                            request_id=request_id,
                            state=state,
                            result=denial,
                            status="failed",
                            active=self._query_output_lease_requests,
                            tombstones=self._query_output_lease_request_tombstones,
                        )
                return

            pending_by_block: dict[
                str,
                tuple[str, dict[str, Any]],
            ] = {}
            for request_id, state in self._query_output_lease_requests.items():
                request = state.get("request")
                if state.get("status") != "pending" or request is None or str(request.query_id) != query_key:
                    continue
                pending_by_block.setdefault(
                    str(request.block_id),
                    (request_id, state),
                )
            if not pending_by_block:
                return
            request, grant = manager.try_acquire_next_queued_output_block(set(pending_by_block))
            if request is None or grant is None:
                return
            request_id, state = pending_by_block[str(request.block_id)]
            if grant.granted:
                lease = grant.lease
                assert lease is not None
                if lease.state != "stage_queue":
                    raise RuntimeError("queued output admission must atomically grant stage_queue ownership")
                lease_payload = asdict(lease)
                manager_lease_id = str(lease_payload["lease_id"])
                lease_payload["lease_id"] = self._issue_output_lease_capability(
                    request_id=request_id,
                    manager_lease_id=manager_lease_id,
                    query_id=lease_payload["query_id"],
                )
                state["manager_lease_id"] = manager_lease_id
                state["status"] = "granted"
                state["lease"] = lease_payload
                _log_resource_debug(
                    "output_granted",
                    query_id=request.query_id,
                    stage_id=request.producer_stage_id,
                    request_id=request_id,
                    lease_id=lease_payload["lease_id"],
                )
                self._complete_lease_request(
                    state,
                    {
                        "granted": True,
                        "lease": lease_payload,
                        "blocked_reason": "",
                        "fatal": False,
                        "liveness": bool(lease_payload["liveness"]),
                    },
                )
                return
            if grant.fatal:
                manager.remove_output_waiter(request.block_id)
                self._retire_query_lease_request(
                    request_id=request_id,
                    state=state,
                    result=self._grant_payload(grant),
                    status="failed",
                    active=self._query_output_lease_requests,
                    tombstones=self._query_output_lease_request_tombstones,
                )
                return
            _log_resource_debug(
                "output_blocked",
                query_id=request.query_id,
                stage_id=request.producer_stage_id,
                request_id=request_id,
                reason=grant.blocked_reason,
            )
        finally:
            self._query_output_admission_pumps.discard(query_key)

    def _run_query_resource_admission_pumps(self, query_id: str) -> None:
        """Drain admission immediately when already executing on the owner loop."""
        query_key = str(query_id)
        self._run_query_task_admission_pump(query_key)
        self._run_query_output_admission_pump(query_key)

    @staticmethod
    def _grant_payload(grant: Any) -> dict[str, Any]:
        return asdict(grant)

    @staticmethod
    def _strict_lease_request(
        payload: dict[str, Any],
        *,
        fields: set[str],
        kind: str,
    ) -> dict[str, Any]:
        values = dict(payload)
        unknown = sorted(set(values) - fields)
        missing = sorted(fields - set(values))
        if unknown or missing:
            details = []
            if unknown:
                details.append(f"unknown={','.join(unknown)}")
            if missing:
                details.append(f"missing={','.join(missing)}")
            raise ValueError(f"invalid {kind} lease request: {' '.join(details)}")
        request_id = str(values["request_id"] or "").strip()
        if not request_id:
            raise ValueError(f"{kind} lease request_id must be non-empty")
        values["request_id"] = request_id
        return values

    def _parse_output_block_lease_request(
        self,
        payload: dict[str, Any],
        *,
        allow_inactive_generation: bool = False,
    ) -> tuple[str, Any, dict[str, Any], bool]:
        from duckdb.runners.ray.query_resource_manager import OutputBlockRequest

        values = self._strict_lease_request(
            payload,
            fields={
                "request_id",
                "query_id",
                "producer_stage_id",
                "task_lease_id",
                "attempt_id",
                "block_id",
                "size_bytes",
            },
            kind="output block",
        )
        request_id = values.pop("request_id")
        task_lease_id = str(values["task_lease_id"])
        capability = self._verify_task_lease_capability(
            lease_id=task_lease_id,
            attempt_id=str(values["attempt_id"]),
            query_id=str(values["query_id"]),
        )
        if capability is None:
            raise ValueError("output block request has an invalid task lease capability")
        (
            manager_task_lease_id,
            _attempt_id,
            capability_query_id,
            _task_request_id,
            capability_generation_id,
        ) = capability
        generation_is_active = not self._capability_targets_inactive_query_generation(
            capability_query_id,
            capability_generation_id,
        )
        if not generation_is_active and not allow_inactive_generation:
            raise ValueError("output block request belongs to an inactive query generation")
        values["task_lease_id"] = manager_task_lease_id
        request = OutputBlockRequest(**values)
        identity = asdict(request)
        identity["task_lease_id"] = task_lease_id
        return request_id, request, identity, generation_is_active

    def _parse_task_lease_request(
        self,
        payload: dict[str, Any],
    ) -> tuple[str, Any, dict[str, Any], str]:
        from duckdb.runners.ray.query_resource_manager import TaskRequest

        raw_values = dict(payload)
        generation_capability = str(raw_values.pop("query_generation_capability", "") or "").strip()
        values = self._strict_lease_request(
            raw_values,
            fields={
                "request_id",
                "query_id",
                "stage_id",
                "task_id",
                "attempt_id",
                "node_id",
                "retained_input_bytes",
                "resources",
            },
            kind="task",
        )
        request_id = values.pop("request_id")
        raw_resources = values.pop("resources")
        request = TaskRequest(**values)
        identity = {**asdict(request), "resources": dict(raw_resources)}
        return request_id, request, identity, generation_capability

    async def acquire_query_task_lease(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Resolve one task lease through the query's serialized admission pump."""
        from duckdb.runners.ray.query_resource_manager import TaskGrant
        from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

        request_id, request, identity, generation_capability = self._parse_task_lease_request(payload)
        raw_resources = identity["resources"]
        self._ensure_query_resource_admission_state()
        generation_id = self._verify_query_task_admission_capability(
            capability=generation_capability,
            query_id=request.query_id,
        )
        if generation_id is None:
            return self._grant_payload(
                TaskGrant(
                    False,
                    blocked_reason="invalid_query_generation_capability",
                    fatal=True,
                )
            )
        if self._capability_targets_inactive_query_generation(
            request.query_id,
            generation_id,
        ):
            return self._grant_payload(
                TaskGrant(
                    False,
                    blocked_reason="query_generation_inactive",
                    fatal=True,
                )
            )
        terminal_control = self._task_terminal_control_for_identity(
            request_id,
            identity,
        )
        if terminal_control is not None:
            return self._grant_payload(
                TaskGrant(
                    False,
                    blocked_reason=terminal_control.blocked_reason,
                    fatal=True,
                )
            )
        if self._task_attempt_fingerprint(identity) in self._query_task_terminal_attempts:
            return self._grant_payload(
                TaskGrant(
                    False,
                    blocked_reason="attempt_terminal",
                    fatal=True,
                )
            )
        tombstone = self._query_task_lease_request_tombstones.get(request_id)
        if tombstone is not None and tombstone["identity"] != identity:
            raise ValueError(f"task lease request_id reused with different identity: {request_id}")
        if tombstone is not None:
            return dict(tombstone["result"])
        if str(request.query_id) in self._query_resource_closing_queries:
            return self._grant_payload(
                TaskGrant(
                    False,
                    blocked_reason="query_not_registered",
                    fatal=True,
                )
            )
        existing = self._query_task_lease_requests.get(request_id)
        if existing is not None and existing["identity"] != identity:
            raise ValueError(f"task lease request_id reused with different identity: {request_id}")
        if existing is not None:
            status = existing["status"]
            if status in {"granted", "completion_wait", "teardown_wait"}:
                return {
                    "granted": True,
                    "lease": dict(existing["lease"]),
                    "blocked_reason": "",
                    "fatal": False,
                    "liveness": bool(existing["lease"]["liveness"]),
                }
            if status != "pending":
                raise RuntimeError(f"invalid task lease request state: {status}")
            return await asyncio.shield(existing["future"])

        resource_identity = self._task_resource_identity(identity)
        conflicting_owner = self._query_task_request_owner_by_identity.get(resource_identity)
        if conflicting_owner is not None:
            raise ValueError(
                "task attempt is already owned by request_id "
                f"{conflicting_owner}: query_id={request.query_id} "
                f"task_id={request.task_id} attempt_id={request.attempt_id}"
            )

        try:
            manager = get_query_resource_manager(request.query_id)
        except KeyError:
            return self._grant_payload(TaskGrant(False, blocked_reason="query_not_registered", fatal=True))
        expected_resources = manager.graph.stage_by_id(request.stage_id).per_task.to_dict()
        if dict(raw_resources) != expected_resources:
            denial = self._grant_payload(TaskGrant(False, blocked_reason="task_resource_spec_mismatch", fatal=True))
            self._query_task_lease_request_tombstones[request_id] = {
                "identity": identity,
                "status": "failed",
                "result": denial,
            }
            self._store_task_terminal_control(
                request_id=request_id,
                identity=identity,
                status="failed",
                result=denial,
            )
            return dict(denial)

        future = asyncio.get_running_loop().create_future()
        existing = {
            "identity": identity,
            "request": request,
            "status": "pending",
            "lease": None,
            "future": future,
        }
        self._query_task_request_owner_by_identity[resource_identity] = request_id
        self._query_task_lease_requests[request_id] = existing
        _log_resource_debug(
            "task_request",
            query_id=request.query_id,
            stage_id=request.stage_id,
            request_id=request_id,
        )
        try:
            manager.note_task_waiting(request)
            self._run_query_task_admission_pump(request.query_id)
        except BaseException:
            manager.remove_task_waiter(request.task_id, request.attempt_id)
            self._query_task_lease_requests.pop(request_id, None)
            if self._query_task_request_owner_by_identity.get(resource_identity) == request_id:
                self._query_task_request_owner_by_identity.pop(
                    resource_identity,
                    None,
                )
            raise
        return await asyncio.shield(future)

    @staticmethod
    def _query_task_lease_matches(
        state: dict[str, Any],
        lease_id: str,
        attempt_id: str,
    ) -> bool:
        lease = state.get("lease") or {}
        return str(lease.get("lease_id") or "") == str(lease_id) and str(lease.get("attempt_id") or "") == str(
            attempt_id
        )

    def _retire_query_task_lease_record(
        self,
        request_id: str,
        state: dict[str, Any],
        *,
        result: dict[str, Any],
        status: str,
    ) -> bool:
        request_key = str(request_id)
        with self._query_task_lease_condition:
            if self._query_task_lease_requests.get(request_key) is not state:
                return False
            self._retire_query_lease_request(
                request_id=request_key,
                state=state,
                result=result,
                status=status,
                active=self._query_task_lease_requests,
                tombstones=self._query_task_lease_request_tombstones,
            )
            self._query_task_lease_condition.notify_all()
            return True

    def _mark_query_execution_quiesced(self, query_id: str) -> None:
        """Retire only task leases explicitly transferred to query teardown."""
        from duckdb.runners.ray.query_resource_manager import TaskGrant

        self._ensure_query_resource_admission_state()
        query_key = str(query_id)
        with self._query_task_lease_condition:
            self._query_execution_quiesced_queries.add(query_key)
            teardown_waiters = [
                (request_id, state)
                for request_id, state in self._query_task_lease_requests.items()
                if state.get("identity", {}).get("query_id") == query_key and state.get("status") == "teardown_wait"
            ]
            for request_id, state in teardown_waiters:
                self._retire_query_lease_request(
                    request_id=request_id,
                    state=state,
                    result=self._grant_payload(
                        TaskGrant(
                            False,
                            blocked_reason="task_lease_owned_by_query_teardown",
                            fatal=True,
                        )
                    ),
                    status="released_by_teardown",
                    active=self._query_task_lease_requests,
                    tombstones=self._query_task_lease_request_tombstones,
                )
            self._query_task_lease_condition.notify_all()

    def _finish_query_task_completion_wait(
        self,
        request_id: str,
        lease_id: str,
        attempt_id: str,
    ) -> dict[str, Any]:
        """Release one lease only after its submitted Ray call is terminal."""
        from duckdb.runners.ray.query_resource_manager import TaskGrant
        from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

        request_key = str(request_id)
        with self._query_task_lease_condition:
            state = self._query_task_lease_requests.get(request_key)
            if (
                state is None
                or state.get("status") != "completion_wait"
                or not self._query_task_lease_matches(
                    state,
                    lease_id,
                    attempt_id,
                )
            ):
                return {"released": False}
            identity = dict(state["identity"])
            manager_lease_id = str(state["manager_lease_id"])
        query_id = str(identity["query_id"])
        try:
            released = get_query_resource_manager(query_id).release_task_lease(
                manager_lease_id,
                attempt_id=str(attempt_id),
            )
        except KeyError:
            released = False
        try:
            get_query_resource_manager(query_id).mark_task_attempt_terminal(
                str(identity["task_id"]),
                str(identity["attempt_id"]),
            )
        except KeyError:
            pass
        retired = self._retire_query_task_lease_record(
            request_key,
            state,
            result=self._grant_payload(
                TaskGrant(
                    False,
                    blocked_reason="task_lease_remote_work_completed",
                    fatal=True,
                )
            ),
            status="released_after_completion",
        )
        if not retired:
            raise RuntimeError(f"task completion lease changed while finishing request {request_key}")
        if query_id not in self._query_resource_closing_queries:
            self._run_query_resource_admission_pumps(query_id)
        return {"released": bool(released)}

    async def _wait_for_query_task_completion(
        self,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        completion_future: ConcurrentFuture[Any],
    ) -> None:
        """Observe terminal work without occupying a Ray actor RPC slot."""
        request_key = str(request_id)
        current = asyncio.current_task()
        try:
            try:
                await asyncio.wrap_future(completion_future)
            except asyncio.CancelledError:
                raise
            except BaseException:
                # A failed completion ObjectRef is still terminal proof.
                pass
            retry_delay_s = _TASK_COMPLETION_CLEANUP_RETRY_MIN_S
            while True:
                try:
                    self._finish_query_task_completion_wait(
                        request_key,
                        lease_id,
                        attempt_id,
                    )
                    return
                except BaseException as exc:
                    with self._query_task_lease_condition:
                        state = self._query_task_lease_requests.get(request_key)
                        if state is None or state.get("status") != "completion_wait":
                            return
                        state["completion_cleanup_attempts"] = int(state.get("completion_cleanup_attempts", 0)) + 1
                        state["completion_cleanup_error"] = f"{type(exc).__name__}: {exc}"
                        self._query_task_lease_condition.notify_all()
                    await asyncio.sleep(retry_delay_s)
                    retry_delay_s = min(
                        _TASK_COMPLETION_CLEANUP_RETRY_MAX_S,
                        retry_delay_s * 2.0,
                    )
        finally:
            with self._query_task_lease_condition:
                record = self._query_task_completion_waiters.get(request_key)
                if record is not None and record[1] is current:
                    self._query_task_completion_waiters.pop(request_key, None)
                    self._query_task_lease_condition.notify_all()

    async def release_query_task_lease_after_completion(
        self,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        completion_refs: list[Any],
    ) -> dict[str, Any]:
        """Transfer task accounting to a driver-owned completion waiter."""
        self._ensure_query_resource_admission_state()
        request_key = str(request_id)
        capability = self._verify_task_lease_capability(
            request_id=request_key,
            lease_id=str(lease_id),
            attempt_id=str(attempt_id),
        )
        if capability is None:
            return {"scheduled": False}
        (
            manager_lease_id,
            _capability_attempt_id,
            capability_query_id,
            _capability_request_id,
            capability_generation_id,
        ) = capability
        if self._capability_targets_inactive_query_generation(
            capability_query_id,
            capability_generation_id,
        ):
            return {"scheduled": True}
        with self._query_task_lease_condition:
            state = self._query_task_lease_requests.get(request_key)
            if state is None:
                return {"scheduled": True}
            if (
                not self._query_task_lease_matches(
                    state,
                    lease_id,
                    attempt_id,
                )
                or str(state.get("manager_lease_id") or "") != manager_lease_id
            ):
                return {"scheduled": False}
            if str(state["identity"]["query_id"]) != capability_query_id:
                return {"scheduled": False}
            if state.get("status") == "teardown_wait":
                return {"scheduled": False}
            waiter = self._query_task_completion_waiters.get(request_key)
            if waiter is not None and not waiter[1].done():
                return {"scheduled": True}
            query_id = str(state["identity"]["query_id"])

        if not isinstance(completion_refs, list) or len(completion_refs) != 1:
            raise ValueError("task completion handoff requires exactly one nested ObjectRef")
        completion_ref = completion_refs[0]
        future_method = getattr(completion_ref, "future", None)
        if not callable(future_method):
            raise TypeError("task completion handoff requires an ObjectRef with future()")
        completion_future = future_method()
        if not isinstance(completion_future, ConcurrentFuture):
            raise TypeError("task completion ObjectRef.future() must return a concurrent Future")
        with self._query_task_lease_condition:
            if self._query_task_lease_requests.get(request_key) is not state:
                return {"scheduled": False}
            if state.get("status") == "teardown_wait":
                return {"scheduled": False}
            waiter = self._query_task_completion_waiters.get(request_key)
            if waiter is not None and not waiter[1].done():
                return {"scheduled": True}
            task = asyncio.create_task(
                self._wait_for_query_task_completion(
                    request_key,
                    str(lease_id),
                    str(attempt_id),
                    completion_future,
                ),
                name=f"vane-query-task-completion:{request_key}",
            )
            state["status"] = "completion_wait"
            state.pop("completion_cleanup_attempts", None)
            state.pop("completion_cleanup_error", None)
            self._query_task_completion_waiters[request_key] = (query_id, task)
            self._query_task_lease_condition.notify_all()
        return {"scheduled": True}

    async def handoff_query_task_lease_to_teardown(
        self,
        request_id: str,
        lease_id: str,
        attempt_id: str,
    ) -> dict[str, Any]:
        """Transfer a cancelled task without a completion ref to query teardown."""
        from duckdb.runners.ray.query_resource_manager import TaskGrant

        self._ensure_query_resource_admission_state()
        request_key = str(request_id)
        capability = self._verify_task_lease_capability(
            request_id=request_key,
            lease_id=str(lease_id),
            attempt_id=str(attempt_id),
        )
        if capability is None:
            return {"handed_off": False}
        (
            manager_lease_id,
            _capability_attempt_id,
            capability_query_id,
            _capability_request_id,
            capability_generation_id,
        ) = capability
        if self._capability_targets_inactive_query_generation(
            capability_query_id,
            capability_generation_id,
        ):
            return {"handed_off": True}
        with self._query_task_lease_condition:
            state = self._query_task_lease_requests.get(request_key)
            if state is None:
                return {"handed_off": True}
            if (
                not self._query_task_lease_matches(
                    state,
                    lease_id,
                    attempt_id,
                )
                or str(state.get("manager_lease_id") or "") != manager_lease_id
            ):
                return {"handed_off": False}
            if str(state["identity"]["query_id"]) != capability_query_id:
                return {"handed_off": False}
            status = str(state.get("status") or "")
            if status == "completion_wait":
                return {"handed_off": False}
            if status == "teardown_wait":
                return {"handed_off": True}
            if status != "granted":
                return {"handed_off": False}
            state["status"] = "teardown_wait"
            query_id = str(state["identity"]["query_id"])
            if query_id in self._query_execution_quiesced_queries:
                self._retire_query_lease_request(
                    request_id=request_key,
                    state=state,
                    result=self._grant_payload(
                        TaskGrant(
                            False,
                            blocked_reason="task_lease_owned_by_query_teardown",
                            fatal=True,
                        )
                    ),
                    status="released_by_teardown",
                    active=self._query_task_lease_requests,
                    tombstones=self._query_task_lease_request_tombstones,
                )
            self._query_task_lease_condition.notify_all()
        return {"handed_off": True}

    def _wait_for_query_task_leases(
        self,
        query_id: str,
        *,
        timeout_s: float = 30.0,
    ) -> None:
        """Fence QRM release behind every granted task record."""
        self._ensure_query_resource_admission_state()
        query_key = str(query_id)
        deadline = time.monotonic() + max(0.001, float(timeout_s))
        with self._query_task_lease_condition:
            while True:
                owners = sorted(
                    (request_id, str(state.get("status") or "unknown"))
                    for request_id, state in self._query_task_lease_requests.items()
                    if state.get("identity", {}).get("query_id") == query_key
                )
                if not owners:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise QueryTeardownOwnershipError(
                        f"query task leases did not quiesce for {query_key}: "
                        + ", ".join(f"{request_id}[{status}]" for request_id, status in owners)
                    )
                self._query_task_lease_condition.wait(remaining)

    async def cancel_query_task_lease_request(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        from duckdb.runners.ray.query_resource_manager import TaskGrant
        from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

        request_id, request, identity, generation_capability = self._parse_task_lease_request(payload)
        self._ensure_query_resource_admission_state()
        generation_id = self._verify_query_task_admission_capability(
            capability=generation_capability,
            query_id=request.query_id,
        )
        if generation_id is None or self._capability_targets_inactive_query_generation(
            request.query_id,
            generation_id,
        ):
            # A stale or unsigned cleanup request owns no state in the active
            # generation. Acknowledge the no-op so its durable retry owner can
            # retire without touching a reopened query with the same ID.
            return {"cancelled": True, "released": False}
        request_key = str(request_id)
        with self._query_task_lease_condition:
            terminal_control = self._task_terminal_control_for_identity(
                request_key,
                identity,
            )
            if terminal_control is not None:
                return {"cancelled": True, "released": False}
            if self._task_attempt_fingerprint(identity) in self._query_task_terminal_attempts:
                raise ValueError(
                    "task attempt is already terminal under another request_id: "
                    f"query_id={request.query_id} task_id={request.task_id} "
                    f"attempt_id={request.attempt_id}"
                )
            state = self._query_task_lease_requests.get(request_key)
            if state is None:
                resource_identity = self._task_resource_identity(identity)
                conflicting_owner = self._query_task_request_owner_by_identity.get(resource_identity)
                if conflicting_owner is not None:
                    raise ValueError(
                        "task attempt is already owned by request_id "
                        f"{conflicting_owner}: query_id={request.query_id} "
                        f"task_id={request.task_id} attempt_id={request.attempt_id}"
                    )
                denial = self._grant_payload(
                    TaskGrant(
                        False,
                        blocked_reason="task_lease_request_cancelled",
                        fatal=True,
                    )
                )
                try:
                    manager = get_query_resource_manager(request.query_id)
                    manager.remove_task_waiter(
                        str(request.task_id),
                        str(request.attempt_id),
                    )
                    manager.mark_task_attempt_terminal(
                        str(request.task_id),
                        str(request.attempt_id),
                    )
                except KeyError:
                    pass
                self._query_task_lease_request_tombstones[request_key] = {
                    "identity": identity,
                    "status": "cancelled",
                    "result": denial,
                }
                self._store_task_terminal_control(
                    request_id=request_key,
                    identity=identity,
                    status="cancelled",
                    result=denial,
                )
                if str(request.query_id) not in self._query_resource_closing_queries:
                    self._run_query_resource_admission_pumps(request.query_id)
                return {"cancelled": True, "released": False}
            if state["identity"] != identity:
                raise ValueError(f"task lease request_id reused with different identity: {request_key}")
            status = str(state.get("status") or "")
            if status in {"completion_wait", "teardown_wait"}:
                # A submitted task can only retire through its selected
                # completion or query-teardown owner.
                return {"cancelled": False, "released": False}
            identity = dict(state["identity"])
            lease = state.get("lease")

        query_id = str(identity["query_id"])
        released = False
        try:
            manager = get_query_resource_manager(query_id)
            if isinstance(lease, dict):
                released = manager.abandon_task_lease(
                    str(state["manager_lease_id"]),
                    attempt_id=str(lease["attempt_id"]),
                )
            else:
                manager.remove_task_waiter(
                    str(identity["task_id"]),
                    str(identity["attempt_id"]),
                )
            manager.mark_task_attempt_terminal(
                str(identity["task_id"]),
                str(identity["attempt_id"]),
            )
        except KeyError:
            pass

        retired = self._retire_query_task_lease_record(
            request_key,
            state,
            result=self._grant_payload(
                TaskGrant(
                    False,
                    blocked_reason="task_lease_request_cancelled",
                    fatal=True,
                )
            ),
            status="cancelled",
        )
        if not retired:
            return {"cancelled": True, "released": False}
        if query_id not in self._query_resource_closing_queries:
            self._run_query_resource_admission_pumps(query_id)
        return {"cancelled": True, "released": bool(released)}

    async def release_query_task_lease(
        self,
        request_id: str,
        lease_id: str,
        attempt_id: str,
    ) -> dict[str, Any]:
        from duckdb.runners.ray.query_resource_manager import TaskGrant
        from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

        self._ensure_query_resource_admission_state()
        request_key = str(request_id)
        capability = self._verify_task_lease_capability(
            request_id=request_key,
            lease_id=str(lease_id),
            attempt_id=str(attempt_id),
        )
        if capability is None:
            return {"released": False}
        (
            manager_lease_id,
            _capability_attempt_id,
            capability_query_id,
            _capability_request_id,
            capability_generation_id,
        ) = capability
        if self._capability_targets_inactive_query_generation(
            capability_query_id,
            capability_generation_id,
        ):
            return {"released": True}
        with self._query_task_lease_condition:
            state = self._query_task_lease_requests.get(request_key)
            if state is None:
                return {"released": True}
            if state.get("status") in {"completion_wait", "teardown_wait"}:
                return {"released": False}
            if (
                not self._query_task_lease_matches(
                    state,
                    lease_id,
                    attempt_id,
                )
                or str(state.get("manager_lease_id") or "") != manager_lease_id
            ):
                return {"released": False}
            if str(state["identity"]["query_id"]) != capability_query_id:
                return {"released": False}
            identity = dict(state["identity"])
        query_id = str(identity["query_id"])
        try:
            _log_resource_debug(
                "task_release_start",
                query_id=query_id,
                stage_id=identity.get("stage_id", ""),
                request_id=request_key,
                lease_id=lease_id,
            )
            released = get_query_resource_manager(query_id).release_task_lease(
                manager_lease_id,
                attempt_id=str(attempt_id),
            )
        except KeyError:
            released = False
        try:
            get_query_resource_manager(query_id).mark_task_attempt_terminal(
                str(identity["task_id"]),
                str(identity["attempt_id"]),
            )
        except KeyError:
            pass
        retired = self._retire_query_task_lease_record(
            request_key,
            state,
            result=self._grant_payload(
                TaskGrant(
                    False,
                    blocked_reason="task_lease_request_released",
                    fatal=True,
                )
            ),
            status="released",
        )
        if not retired:
            return {"released": False}
        _log_resource_debug(
            "task_release_done",
            query_id=query_id,
            stage_id=identity.get("stage_id", ""),
            request_id=request_key,
            lease_id=lease_id,
            released=bool(released),
        )
        if query_id not in self._query_resource_closing_queries:
            self._run_query_resource_admission_pumps(query_id)
        return {"released": True}

    async def acquire_query_output_block_lease(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Resolve one produced block through the query's admission pump."""
        from duckdb.runners.ray.query_resource_manager import OutputBlockGrant
        from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

        request_id, request, identity, _generation_is_active = self._parse_output_block_lease_request(payload)
        self._ensure_query_resource_admission_state()
        terminal_control = self._output_terminal_control_for_identity(
            request_id,
            identity,
        )
        if terminal_control is not None:
            return self._grant_payload(
                OutputBlockGrant(
                    False,
                    blocked_reason=terminal_control.blocked_reason,
                    fatal=True,
                )
            )
        tombstone = self._query_output_lease_request_tombstones.get(request_id)
        if tombstone is not None and tombstone["identity"] != identity:
            raise ValueError(f"output lease request_id reused with different identity: {request_id}")
        if tombstone is not None:
            return dict(tombstone["result"])
        if str(request.query_id) in self._query_resource_closing_queries:
            return self._grant_payload(
                OutputBlockGrant(
                    False,
                    blocked_reason="query_not_registered",
                    fatal=True,
                )
            )
        existing = self._query_output_lease_requests.get(request_id)
        if existing is not None and existing["identity"] != identity:
            raise ValueError(f"output lease request_id reused with different identity: {request_id}")
        if existing is not None:
            status = existing["status"]
            if status == "granted":
                return {
                    "granted": True,
                    "lease": dict(existing["lease"]),
                    "blocked_reason": "",
                    "fatal": False,
                    "liveness": bool(existing["lease"]["liveness"]),
                }
            return await asyncio.shield(existing["future"])

        resource_identity = (
            str(identity["query_id"]),
            str(identity["block_id"]),
        )
        conflicting_owner = self._query_output_request_owner_by_identity.get(resource_identity)
        if conflicting_owner is not None:
            raise ValueError(
                "output block is already owned by request_id "
                f"{conflicting_owner}: query_id={request.query_id} "
                f"block_id={request.block_id}"
            )

        try:
            manager = get_query_resource_manager(request.query_id)
        except KeyError:
            return self._grant_payload(
                OutputBlockGrant(
                    False,
                    blocked_reason="query_not_registered",
                    fatal=True,
                )
            )
        future = asyncio.get_running_loop().create_future()
        existing = {
            "identity": identity,
            "request": request,
            "status": "pending",
            "lease": None,
            "future": future,
        }
        self._query_output_request_owner_by_identity[resource_identity] = request_id
        self._query_output_lease_requests[request_id] = existing
        try:
            manager.note_output_waiting(request)
            self._run_query_output_admission_pump(request.query_id)
        except BaseException:
            manager.remove_output_waiter(request.block_id)
            self._query_output_lease_requests.pop(request_id, None)
            if self._query_output_request_owner_by_identity.get(resource_identity) == request_id:
                self._query_output_request_owner_by_identity.pop(
                    resource_identity,
                    None,
                )
            raise
        return await asyncio.shield(future)

    async def handoff_query_output_block_lease(
        self,
        request_id: str,
        lease_id: str,
        query_id: str = "",
    ) -> dict[str, Any]:
        """Transfer a live output from the producer queue to downstream input.

        This returns only the producer-side liveness credit.  The output lease
        itself remains active, so its bytes continue to count against query,
        stage, and node object-store limits until the final descriptor owner
        releases it.
        """
        from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

        self._ensure_query_resource_admission_state()
        request_key = str(request_id)
        capability = self._verify_output_lease_capability(
            request_id=request_key,
            lease_id=str(lease_id),
            query_id=str(query_id),
        )
        if capability is None:
            return {"handed_off": False}
        manager_lease_id, capability_query_id, capability_generation_id = capability
        if self._capability_targets_inactive_query_generation(
            capability_query_id,
            capability_generation_id,
        ):
            return {"handed_off": True}
        state = self._query_output_lease_requests.get(request_key)
        if state is None:
            return {"handed_off": True}
        lease = state.get("lease") or {}
        if str(lease.get("lease_id") or "") != str(lease_id):
            return {"handed_off": False}
        if str(lease.get("query_id") or "") != capability_query_id:
            return {"handed_off": False}
        current_state = str(lease.get("state") or "")
        if current_state == "downstream_input":
            return {"handed_off": True}
        if current_state != "stage_queue":
            raise RuntimeError(f"output lease handoff requires stage_queue state, got {current_state!r}")
        try:
            handed_off = get_query_resource_manager(capability_query_id).transition_output_block(
                manager_lease_id,
                "downstream_input",
            )
        except KeyError:
            handed_off = False
        if handed_off:
            lease["state"] = "downstream_input"
            self._run_query_resource_admission_pumps(lease["query_id"])
        return {
            "handed_off": bool(handed_off or str(lease.get("query_id") or "") in self._query_resource_closing_queries)
        }

    async def release_query_output_block_lease(
        self,
        request_id: str,
        lease_id: str,
        query_id: str = "",
    ) -> dict[str, Any]:
        from duckdb.runners.ray.query_resource_manager import OutputBlockGrant
        from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

        self._ensure_query_resource_admission_state()
        request_key = str(request_id)
        capability = self._verify_output_lease_capability(
            request_id=request_key,
            lease_id=str(lease_id),
            query_id=str(query_id),
        )
        if capability is None:
            return {"released": False}
        manager_lease_id, capability_query_id, capability_generation_id = capability
        if self._capability_targets_inactive_query_generation(
            capability_query_id,
            capability_generation_id,
        ):
            return {"released": True}
        state = self._query_output_lease_requests.get(request_key)
        if state is None:
            # Query close drops request payload state before the QRM itself can
            # be released. The signed capability still authorizes this exact
            # internal lease in the current generation, so reclaim it
            # idempotently instead of retaining object-store accounting behind
            # an unrelated task-teardown fence.
            try:
                get_query_resource_manager(capability_query_id).release_output_block(
                    manager_lease_id,
                )
            except KeyError:
                pass
            return {"released": True}
        lease = state.get("lease") or {}
        if str(lease.get("lease_id") or "") != str(lease_id):
            return {"released": False}
        if str(lease.get("query_id") or "") != capability_query_id:
            return {"released": False}
        try:
            get_query_resource_manager(capability_query_id).release_output_block(manager_lease_id)
        except KeyError:
            pass
        self._retire_query_lease_request(
            request_id=request_key,
            state=state,
            result=self._grant_payload(
                OutputBlockGrant(
                    False,
                    blocked_reason="output_lease_request_released",
                    fatal=True,
                )
            ),
            status="released",
            active=self._query_output_lease_requests,
            tombstones=self._query_output_lease_request_tombstones,
        )
        self._run_query_resource_admission_pumps(lease["query_id"])
        return {"released": True}

    async def cancel_query_output_block_lease_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        from duckdb.runners.ray.query_resource_manager import OutputBlockGrant
        from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

        request_id, request, identity, generation_is_active = self._parse_output_block_lease_request(
            payload,
            allow_inactive_generation=True,
        )
        self._ensure_query_resource_admission_state()
        # A valid task capability from an older query generation proves that
        # this cancellation cannot own anything in the current manager.  ACK it
        # without publishing a tombstone or terminal block into the reopened
        # generation; the remote cleanup owner can then retire durably.
        if not generation_is_active:
            return {"cancelled": True, "released": False}
        request_key = request_id
        terminal_control = self._output_terminal_control_for_identity(
            request_key,
            identity,
        )
        if terminal_control is not None:
            return {"cancelled": True, "released": False}
        denial = self._grant_payload(
            OutputBlockGrant(
                False,
                blocked_reason="output_lease_request_cancelled",
                fatal=True,
            )
        )
        tombstone = self._query_output_lease_request_tombstones.get(request_key)
        if tombstone is not None and tombstone["identity"] != identity:
            raise ValueError(f"output lease request_id reused with different identity: {request_key}")
        if tombstone is not None:
            return {"cancelled": True, "released": False}
        if str(request.query_id) in self._query_resource_closing_queries:
            self._store_output_terminal_control(
                request_id=request_key,
                identity=identity,
                status="cancelled",
                result=denial,
            )
            return {"cancelled": True, "released": False}
        state = self._query_output_lease_requests.get(request_key)
        if state is not None and state["identity"] != identity:
            raise ValueError(f"output lease request_id reused with different identity: {request_key}")
        if state is None:
            resource_identity = (
                str(identity["query_id"]),
                str(identity["block_id"]),
            )
            conflicting_owner = self._query_output_request_owner_by_identity.get(resource_identity)
            if conflicting_owner is not None:
                raise ValueError(
                    "output block is already owned by request_id "
                    f"{conflicting_owner}: query_id={request.query_id} "
                    f"block_id={request.block_id}"
                )
            try:
                get_query_resource_manager(request.query_id).mark_output_block_terminal(request.block_id)
            except KeyError:
                pass
            self._query_output_lease_request_tombstones[request_key] = {
                "identity": identity,
                "status": "cancelled",
                "result": denial,
            }
            self._store_output_terminal_control(
                request_id=request_key,
                identity=identity,
                status="cancelled",
                result=denial,
            )
            self._run_query_resource_admission_pumps(request.query_id)
            return {"cancelled": True, "released": False}
        lease = state.get("lease")
        released = False
        if lease is not None:
            try:
                released = get_query_resource_manager(lease["query_id"]).release_output_block(
                    str(state["manager_lease_id"])
                )
            except KeyError:
                released = False
        else:
            identity = state["identity"]
            try:
                manager = get_query_resource_manager(identity["query_id"])
                manager.remove_output_waiter(identity["block_id"])
                manager.mark_output_block_terminal(identity["block_id"])
            except KeyError:
                pass
        if lease is not None:
            try:
                get_query_resource_manager(state["identity"]["query_id"]).mark_output_block_terminal(
                    state["identity"]["block_id"]
                )
            except KeyError:
                pass
        query_id = state["identity"]["query_id"]
        self._retire_query_lease_request(
            request_id=request_key,
            state=state,
            result=denial,
            status="cancelled",
            active=self._query_output_lease_requests,
            tombstones=self._query_output_lease_request_tombstones,
        )
        self._run_query_resource_admission_pumps(query_id)
        return {"cancelled": True, "released": bool(released)}

    def _prepare_query_resource_registration(
        self,
        plan: Any,
        query_connection: Any,
        expected_plan_id: str | None = None,
    ) -> _PreparedQueryResourceRegistration:
        from duckdb.runners.ray.query_graph_builder import (
            build_query_execution_graph,
        )

        metadata = plan.collect_execution_stages(conn=query_connection)
        graph = build_query_execution_graph(metadata)
        # The session preparation step already compared the logical and
        # physical IDs. Reuse that immutable identity after handing the plan to
        # the registration thread instead of re-entering the physical wrapper.
        plan_id = str(plan.idx()) if expected_plan_id is None else str(expected_plan_id)
        if graph.query_id != plan_id:
            raise ValueError(f"execution graph query_id mismatch: graph={graph.query_id!r} plan={plan_id!r}")
        capacity_snapshot_started_at = time.monotonic()
        capacities = self._read_query_node_capacities()
        if not capacities:
            raise RuntimeError("Ray reports no alive nodes with schedulable query resources")
        return _PreparedQueryResourceRegistration(
            graph=graph,
            node_capacities=tuple(capacities),
            capacity_snapshot_started_at=capacity_snapshot_started_at,
        )

    async def _register_query_resources(
        self,
        plan: Any,
        *,
        query_connection: Any,
        expected_plan_id: str | None = None,
    ) -> tuple[Any, Any]:
        """Prepare registration off-loop, then atomically publish local state."""
        self._ensure_query_resource_admission_state()
        owner_loop = self._query_resource_admission_loop
        running_loop = asyncio.get_running_loop()
        if owner_loop is None:
            self._query_resource_admission_loop = running_loop
        elif owner_loop is not running_loop:
            raise RuntimeError("query resource registration must run on the admission owner loop")

        prepared = await _to_thread_with_owned_side_effects(
            self._prepare_query_resource_registration,
            plan,
            query_connection,
            expected_plan_id,
        )
        deferred_teardowns: list[_DeferredQueryAllocationTeardown] = []
        try:
            result = self._commit_query_resource_registration(
                prepared,
                deferred_teardowns=deferred_teardowns,
            )
        finally:
            self._schedule_query_allocation_teardowns(
                tuple(deferred_teardowns),
            )
        return result

    def _commit_query_resource_registration(
        self,
        prepared: _PreparedQueryResourceRegistration,
        *,
        deferred_teardowns: list[_DeferredQueryAllocationTeardown],
    ) -> tuple[Any, Any]:
        from duckdb.runners.ray.query_graph_builder import build_query_demand
        from duckdb.runners.ray.query_resource_runtime import (
            register_query_graph,
            release_query_resource_manager,
        )

        graph = prepared.graph
        with self._query_resource_lock:
            if graph.query_id in self._query_graphs:
                raise ValueError(f"query graph is already registered: {graph.query_id}")
            self._ensure_query_allocation_teardown_state()
            if graph.query_id in self._query_allocation_teardowns_pending:
                raise RuntimeError(f"query {graph.query_id} still has a pending allocation-loss teardown")
            cluster_capacity = self._apply_query_capacity_snapshot(
                prepared.node_capacities,
                snapshot_started_at=prepared.capacity_snapshot_started_at,
            )
            deferred_teardowns.extend(self._synchronize_query_allocations())
            demand = build_query_demand(
                graph,
                cluster_capacity,
            )
            allocation = self._query_resource_coordinator.register_query(demand)
            manager_registered = False
            try:
                deferred_teardowns.extend(self._synchronize_query_allocations())
                manager = register_query_graph(
                    graph,
                    allocation,
                    reservation_ratio=float(os.environ.get("VANE_QUERY_RESOURCE_RESERVATION_RATIO", "0.5")),
                    on_change=lambda: self._signal_query_resource_change(graph.query_id),
                )
                manager_registered = True
                for stage in graph.stages:
                    manager.update_stage_state(
                        stage.stage_id,
                        runnable=not stage.input_stage_ids,
                        actor_ready=stage.backend != "ray_actor",
                    )
                self._query_graphs[graph.query_id] = graph
                self._query_allocations[graph.query_id] = allocation
                self._open_query_resource_admission(graph.query_id)
                return graph, allocation
            except BaseException as registration_error:
                cleanup_errors: list[BaseException] = []
                coordinator_released = False
                try:
                    released = self._query_resource_coordinator.release_query(
                        graph.query_id,
                        allocation.generation,
                    )
                    coordinator_released = True
                    if not released:
                        cleanup_errors.append(
                            RuntimeError("coordinator allocation disappeared during query registration rollback")
                        )
                except BaseException as exc:
                    cleanup_errors.append(exc)
                if coordinator_released and manager_registered:
                    try:
                        release_query_resource_manager(
                            graph.query_id,
                            reason="query_registration_failed",
                        )
                    except BaseException as exc:
                        cleanup_errors.append(exc)
                if coordinator_released:
                    self._query_graphs.pop(graph.query_id, None)
                    self._query_allocations.pop(graph.query_id, None)
                else:
                    self._query_graphs[graph.query_id] = graph
                    self._query_allocations[graph.query_id] = allocation
                try:
                    deferred_teardowns.extend(self._synchronize_query_allocations())
                except BaseException as exc:
                    cleanup_errors.append(exc)
                if cleanup_errors:
                    raise RuntimeError(
                        f"query resource registration failed for {graph.query_id} and rollback had "
                        f"{len(cleanup_errors)} error(s): "
                        + "; ".join(f"{type(exc).__name__}: {exc}" for exc in cleanup_errors)
                    ) from registration_error
                raise

    def _mark_query_actor_stages_ready(self, graph: Any) -> None:
        from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

        manager = get_query_resource_manager(graph.query_id)
        for stage in graph.stages:
            if stage.backend == "ray_actor":
                manager.set_stage_actor_ready(stage.stage_id, True)

    def _fence_query_resource_admission_for_teardown(self, query_id: str) -> None:
        from duckdb.runners.ray.fte_fragment_scheduler import quiesce_fte_registry_for_query

        query_key = str(query_id or "").strip()
        if not query_key:
            return
        self._ensure_query_resource_admission_state()
        loop = self._query_resource_admission_loop
        if loop is None or loop.is_closed() or not loop.is_running():
            pump = self._query_fte_admission_pumps.get(query_key)
            if pump is not None and not pump.done():
                raise RuntimeError(f"query {query_key} has an active FTE admission pump without a live owner loop")
            self._close_query_resource_admission(query_key)
            self._purge_completed_query_fte_admission_pump(query_key)
            quiesce_fte_registry_for_query(query_key)
            return
        self._run_on_query_resource_admission_loop_sync(
            lambda: self._close_query_resource_admission(query_key),
            query_id=query_key,
        )
        # A resource-change callback may already be draining the FTE scheduler
        # in a worker thread. Keep every fragment registry alive until that
        # drain exits; the close fence prevents another drain from starting.
        self._wait_for_query_fte_admission_pump(query_key)
        self._run_on_query_resource_admission_loop_sync(
            lambda: self._purge_completed_query_fte_admission_pump(query_key),
            query_id=query_key,
        )
        quiesce_fte_registry_for_query(query_key)

    def _release_query_resources(
        self,
        query_id: str,
        *,
        reason: str,
        admission_fenced: bool = False,
        execution_quiesced: bool = False,
    ) -> None:
        from duckdb.runners.ray.query_resource_runtime import release_query_resource_manager

        query_key = str(query_id or "").strip()
        if not query_key:
            return
        self._drop_progress_snapshot_state(query_key)
        if not admission_fenced:
            try:
                self._fence_query_resource_admission_for_teardown(query_key)
            except BaseException as exc:
                if isinstance(exc, QueryTeardownOwnershipError):
                    raise
                raise QueryTeardownOwnershipError(
                    f"failed to fence query resource release for {query_key}: {type(exc).__name__}: {exc}"
                ) from exc
        if execution_quiesced:
            # Successful execution teardown is terminal evidence only for a
            # cancelled task that explicitly selected teardown ownership.
            # Ordinary grants and completion waiters remain fenced.
            self._mark_query_execution_quiesced(query_key)
        # A granted request remains in the same admission ledger until its
        # explicit terminal transition, including completion handoffs that
        # arrive after query teardown is already fenced.
        self._wait_for_query_task_leases(query_key)
        cleanup_errors: list[BaseException] = []
        deferred_teardowns: tuple[_DeferredQueryAllocationTeardown, ...] = ()
        try:
            with self._query_resource_lock:
                allocation = self._query_allocations.get(query_key)
                coordinator_released = allocation is None
                if allocation is not None:
                    try:
                        released = self._query_resource_coordinator.release_query(
                            query_key,
                            allocation.generation,
                        )
                        coordinator_released = True
                        if not released:
                            cleanup_errors.append(
                                RuntimeError(
                                    f"coordinator allocation was already absent for query {query_key} "
                                    f"at generation {allocation.generation}"
                                )
                            )
                    except BaseException as exc:
                        cleanup_errors.append(exc)
                if coordinator_released:
                    try:
                        release_query_resource_manager(query_key, reason=reason)
                    except BaseException as exc:
                        cleanup_errors.append(exc)
                    self._query_graphs.pop(query_key, None)
                    self._query_allocations.pop(query_key, None)
                try:
                    deferred_teardowns = self._synchronize_query_allocations()
                except BaseException as exc:
                    cleanup_errors.append(exc)
        finally:
            self._run_query_allocation_teardowns(deferred_teardowns)
        if cleanup_errors:
            raise RuntimeError(
                f"query resource cleanup for {query_key} had "
                f"{len(cleanup_errors)} error(s): "
                + "; ".join(f"{type(exc).__name__}: {exc}" for exc in cleanup_errors)
            ) from cleanup_errors[0]
        with self._query_task_lease_condition:
            self._query_execution_quiesced_queries.discard(query_key)
        self._query_lease_generation_ids.pop(query_key, None)

    def query_resource_snapshot(
        self,
        owner_id: str,
        session_id: str,
        query_id: str,
    ) -> dict[str, Any]:
        from duckdb.runners.ray.query_resource_runtime import query_resource_manager_snapshot

        session = self._require_session(owner_id, session_id)
        query_key = str(query_id or "").strip()
        with session.condition:
            plan_ids = tuple(session.plan_ids)
        with self._session_lock:
            owned_query_ids = {self._plan_query_ids.get(plan_id) for plan_id in plan_ids}
        if query_key not in owned_query_ids:
            raise PermissionError("query resources do not belong to the requested Vane session")
        coordinator = self._query_resource_coordinator.snapshot()
        return {
            "manager": query_resource_manager_snapshot(query_key),
            "coordinator": coordinator["queries"].get(query_key, {}),
            "maintenance": {
                "last_run_at": float(getattr(self, "_query_resource_last_maintenance_at", 0.0)),
                "last_capacity_refresh_at": float(getattr(self, "_query_resource_last_capacity_refresh_at", 0.0)),
                "failures": int(getattr(self, "_query_resource_maintenance_failures", 0)),
                "error": str(getattr(self, "_query_resource_maintenance_error", "")),
            },
        }

    def get_test_udf_actor_handle(self, owner_id: str, plan_id: str, udf_name: str) -> Any:
        """Return one query-owned stateful actor for deterministic fault tests.

        This hook is deliberately environment-gated and lives on the Driver;
        it is not part of Vane's public UDF API.
        """
        if not _udf_test_hooks_enabled():
            raise RuntimeError(
                "UDF actor test hooks are disabled; set VANE_ENABLE_UDF_TEST_HOOKS=1 before starting the Driver"
            )

        plan_key = str(plan_id)
        with self._session_lock:
            session_id = self._plan_session_ids.get(plan_key)
        if session_id is None:
            raise RuntimeError(f"query plan is not active: {plan_key}")
        self._require_plan_session(owner_id, session_id, plan_key)
        pools = self._active_udf_actors_by_plan.get(plan_key)
        if pools is None:
            raise RuntimeError(f"no active UDF actor pools found for plan {plan_key!r}")

        matches: list[Any] = []
        for pool in pools:
            payload = getattr(pool, "_payload", None)
            if not isinstance(payload, dict) or not payload.get("stateful"):
                continue
            if str(payload.get("udf_name") or "") != str(udf_name):
                continue
            matches.append(pool)

        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one active stateful UDF pool for plan {plan_key!r} "
                f"and udf {udf_name!r}, found {len(matches)}"
            )
        actors = list(getattr(matches[0], "actors", ()))
        if len(actors) != 1:
            raise RuntimeError(
                f"stateful UDF pool for {udf_name!r} must contain exactly one actor, found {len(actors)}"
            )
        return actors[0]

    def _get_plan_runner(self) -> Any:
        with self._plan_runner_lifecycle_lock:
            if self._driver_shutdown_started:
                raise RuntimeError("Ray query driver is shutting down")
            if self.plan_runner is None:
                DistributedPhysicalPlanRunner = require_ray_cxx_attr(
                    "DistributedPhysicalPlanRunner",
                    hint="Ensure the C++ ray extension is built and importable in this process.",
                )
                self.plan_runner = DistributedPhysicalPlanRunner()
                self.plan_runner.warm_up()
            return self.plan_runner

    def _precreate_udf_actors(
        self,
        plan: Any,
        graph: Any,
        allocation: Any,
        *,
        query_connection: Any,
        session_config: dict[str, str],
    ) -> list[Any]:
        """Create Ray actors and inject handles without waiting for model init."""
        from duckdb.execution.udf_ray import prepare_actor_pools_for_plan

        plan_id = str(plan.idx())
        actor_node_ids_by_stage = {
            stage.stage_id: allocation.actor_node_ids_for_stage(stage.stage_id)
            for stage in graph.stages
            if stage.backend == "ray_actor"
        }
        try:
            created, _ = prepare_actor_pools_for_plan(
                plan,
                actor_node_ids_by_stage=actor_node_ids_by_stage,
                query_driver_handle=self._driver_handle,
                query_generation_capability=self._issue_query_task_admission_capability(
                    graph.query_id,
                ),
                session_config=session_config,
                conn=query_connection,
            )
        except BaseException as creation_error:
            owned_pools = list(getattr(creation_error, "owned_actor_pools", ()))
            for pool in owned_pools:
                if all(existing is not pool for existing in self._active_udf_actors):
                    self._active_udf_actors.append(pool)
            if owned_pools:
                self._active_udf_actors_by_plan[plan_id] = owned_pools
            raise
        self._active_udf_actors.extend(created)
        if created:
            self._active_udf_actors_by_plan[plan_id] = list(created)
        return created

    @staticmethod
    def _wait_for_udf_actors_ready(actor_pools: list[Any]) -> None:
        from duckdb.execution.udf_ray import wait_for_actor_pools_ready

        wait_for_actor_pools_ready(actor_pools)

    def _precreate_vllm_actors(
        self,
        plan: Any,
        *,
        query_connection: Any,
        session_config: dict[str, str],
    ) -> list[Any]:
        """Pre-create Ray actor pools for vLLM nodes on the driver."""
        from duckdb.execution.vllm import ensure_named_vllm_pools_for_plan

        plan_id = str(plan.idx())
        try:
            created, _ = ensure_named_vllm_pools_for_plan(
                plan,
                conn=query_connection,
                session_config=session_config,
            )
        except BaseException as creation_error:
            owned_pools = list(getattr(creation_error, "owned_actor_pools", ()))
            for pool in owned_pools:
                if all(existing is not pool for existing in self._active_vllm_actors):
                    self._active_vllm_actors.append(pool)
            if owned_pools:
                self._active_vllm_actors_by_plan[plan_id] = owned_pools
            raise
        self._active_vllm_actors.extend(created)
        if created:
            self._active_vllm_actors_by_plan[plan_id] = list(created)
        return created

    def _cleanup_udf_actor_pools(self, plan_id: str) -> None:
        plan_key = str(plan_id)
        pools = list(self._active_udf_actors_by_plan.get(plan_key, ()))
        if not pools:
            return
        errors: list[str] = []
        remaining_pools: list[Any] = []
        for pool in pools:
            try:
                pool.shutdown()
            except BaseException as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                remaining_pools.append(pool)
            else:
                try:
                    self._active_udf_actors.remove(pool)
                except ValueError:
                    pass
        if remaining_pools:
            self._active_udf_actors_by_plan[plan_key] = remaining_pools
        else:
            self._active_udf_actors_by_plan.pop(plan_key, None)
        if errors:
            raise RuntimeError(f"failed to shut down {len(errors)} query-owned UDF actor pool(s): " + "; ".join(errors))

    def _cleanup_vllm_actor_pools(self, plan_id: str) -> None:
        plan_key = str(plan_id)
        pools = list(self._active_vllm_actors_by_plan.get(plan_key, ()))
        if not pools:
            return
        errors: list[str] = []
        remaining_pools: list[Any] = []
        for pool in pools:
            try:
                pool.shutdown()
            except BaseException as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                remaining_pools.append(pool)
            else:
                try:
                    self._active_vllm_actors.remove(pool)
                except ValueError:
                    pass
        if remaining_pools:
            self._active_vllm_actors_by_plan[plan_key] = remaining_pools
        else:
            self._active_vllm_actors_by_plan.pop(plan_key, None)
        if errors:
            raise RuntimeError(
                f"failed to shut down {len(errors)} query-owned vLLM actor pool(s): " + "; ".join(errors)
            )

    def _release_plan_session_state(self, plan_id: str) -> None:
        plan_key = str(plan_id)
        with self._session_lock:
            session_id = self._plan_session_ids.get(plan_key)
            query_connection = self._plan_connections.get(plan_key)
            session = self._sessions.get(session_id) if session_id is not None else None
        if query_connection is not None:
            query_connection.close()
        with self._session_lock:
            if self._plan_session_ids.get(plan_key) == session_id:
                self._plan_session_ids.pop(plan_key, None)
            if self._plan_connections.get(plan_key) is query_connection:
                self._plan_connections.pop(plan_key, None)
        if session is not None:
            with session.condition:
                session.plan_ids.discard(plan_key)

    def _teardown_plan_resources(
        self,
        plan_id: str,
        query_id: str,
        *,
        drop_fragments: bool,
    ) -> None:
        plan_key = str(plan_id)
        with self._plan_teardown_condition:
            while plan_key in self._plan_teardowns_in_progress:
                self._plan_teardown_condition.wait()
            if plan_key not in self._plan_session_ids:
                return
            self._plan_teardowns_in_progress.add(plan_key)
        try:
            self._teardown_plan_resources_once(
                plan_key,
                query_id,
                drop_fragments=drop_fragments,
            )
        finally:
            with self._plan_teardown_condition:
                self._plan_teardowns_in_progress.discard(plan_key)
                self._plan_teardown_condition.notify_all()

    def _teardown_plan_resources_once(
        self,
        plan_id: str,
        query_id: str,
        *,
        drop_fragments: bool,
    ) -> None:
        errors: list[BaseException] = []
        execution_owner_errors: list[BaseException] = []
        self.curr_plans.pop(plan_id, None)
        self.curr_streams.pop(plan_id, None)
        with self._plan_teardown_condition:
            leased_refs = getattr(self, "_leased_result_partition_refs", None)
            records = {} if leased_refs is None else dict(leased_refs.get(str(plan_id), {}))
        if records:
            for release_token, record in records.items():
                _, output_lease_owner = record
                try:
                    output_lease_owner.release()
                except BaseException as exc:
                    errors.append(exc)
                    execution_owner_errors.append(exc)
                else:
                    with self._plan_teardown_condition:
                        current_refs = self._leased_result_partition_refs.get(str(plan_id))
                        if current_refs is not None and current_refs.get(release_token) is record:
                            current_refs.pop(release_token, None)
                            if not current_refs:
                                self._leased_result_partition_refs.pop(str(plan_id), None)
        with self._plan_teardown_condition:
            leased_refs = getattr(self, "_leased_result_partition_refs", None)
            remaining_refs = None if leased_refs is None else leased_refs.get(str(plan_id))
            if not remaining_refs:
                counters = getattr(self, "_result_partition_ref_counters", None)
                if counters is not None:
                    counters.pop(str(plan_id), None)
        try:
            self._cleanup_udf_actor_pools(str(plan_id))
        except BaseException as exc:
            errors.append(exc)
            execution_owner_errors.append(exc)
        try:
            self._cleanup_vllm_actor_pools(str(plan_id))
        except BaseException as exc:
            errors.append(exc)
            execution_owner_errors.append(exc)
        query_key = str(query_id or "").strip()
        if query_key:
            try:
                if drop_fragments:
                    if execution_owner_errors:
                        # Quiesce independently owned fragments, but keep the
                        # QRM/coordinator allocation intact while an output or
                        # actor owner failed to terminate.
                        self._drop_query_fragments_after_admission_fence_sync(
                            query_key,
                            release_resources=False,
                        )
                    else:
                        self._drop_query_fragments_sync(query_key)
                elif not execution_owner_errors:
                    self._release_query_resources(
                        query_key,
                        reason="query_ended_before_fragment_start",
                        execution_quiesced=True,
                    )
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(
                f"query plan {plan_id} teardown failed: " + "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
            ) from errors[0]
        self._plan_query_ids.pop(plan_id, None)
        terminal_errors = getattr(self, "_query_terminal_errors", None)
        if terminal_errors is not None:
            terminal_errors.pop(str(query_id or ""), None)
        self._release_plan_session_state(plan_id)

    def _cleanup_finished_plan(self, plan_id: str) -> None:
        query_id = self._plan_query_ids.get(plan_id, "")
        self._teardown_plan_resources(
            plan_id,
            query_id,
            drop_fragments=bool(query_id),
        )

    def _finish_terminal_query(self, plan_id: str, query_id: str) -> None:
        reason = str(getattr(self, "_query_terminal_errors", {}).get(str(query_id), ""))
        if not reason:
            return
        primary_error = RuntimeError(reason)
        try:
            self._teardown_plan_resources(
                plan_id,
                query_id,
                drop_fragments=True,
            )
        except BaseException as teardown_error:
            aggregate_error = QueryExecutionCleanupError.from_errors(
                f"terminal query {query_id} failed and deterministic teardown also failed",
                primary_error,
                [teardown_error],
            )
            raise aggregate_error from primary_error
        raise primary_error

    def _lease_result_partition_ref(
        self,
        session: _DriverSession,
        session_id: str,
        plan_id: str,
        object_ref: Any,
        output_lease_owner: Any,
    ) -> str:
        plan_key = str(plan_id)
        with session.condition:
            if session.closing or session.closed:
                raise RuntimeError(f"Vane session closed while publishing query result: {session_id}")
            with self._plan_teardown_condition:
                if plan_key in self._plan_teardowns_in_progress or self._plan_session_ids.get(plan_key) != str(
                    session_id
                ):
                    raise RuntimeError(f"query plan closed while publishing result: {plan_key}")
                if not hasattr(self, "_leased_result_partition_refs"):
                    self._leased_result_partition_refs = {}
                if not hasattr(self, "_result_partition_ref_counters"):
                    self._result_partition_ref_counters = {}
                if not output_lease_owner.transition_to("external_consumer"):
                    raise RuntimeError(f"query result lease was released before external publication: plan={plan_key}")
                next_id = int(self._result_partition_ref_counters.get(plan_key, 0))
                self._result_partition_ref_counters[plan_key] = next_id + 1
                release_token = str(next_id)
                self._leased_result_partition_refs.setdefault(plan_key, {})[release_token] = (
                    object_ref,
                    output_lease_owner,
                )
                return release_token

    def release_result_partition_ref(
        self,
        owner_id: str,
        session_id: str,
        plan_id: str,
        release_token: str,
    ) -> None:
        owner_key = str(owner_id).strip()
        session_key = str(session_id).strip()
        if not owner_key:
            raise ValueError("Ray query runtime owner_id must not be empty")
        if not session_key:
            raise ValueError("Vane session_id must not be empty")
        plan_key = str(plan_id)
        with self._session_lock:
            session = self._sessions.get(session_key)
            closed_owner = self._closed_session_owners.get(session_key)
            plan_session_id = self._plan_session_ids.get(plan_key)
        if session is None:
            if closed_owner is None:
                raise RuntimeError(f"Vane session is not open: {session_key}")
            if closed_owner != owner_key:
                raise PermissionError("Vane session operation requires its owning runtime client")
            return
        if session.owner_id != owner_key:
            raise PermissionError("Vane session operation requires its owning runtime client")
        if plan_session_id is None:
            return
        if plan_session_id != session_key:
            raise PermissionError("query plan does not belong to the requested Vane session")
        with self._plan_teardown_condition:
            if plan_key in self._plan_teardowns_in_progress:
                return
            current_plan_session_id = self._plan_session_ids.get(plan_key)
            if current_plan_session_id is None:
                return
            if current_plan_session_id != session_key:
                raise PermissionError("query plan does not belong to the requested Vane session")
            if not hasattr(self, "_leased_result_partition_refs"):
                self._leased_result_partition_refs = {}
            refs = self._leased_result_partition_refs.get(plan_key)
            if not refs:
                return
            token_key = str(release_token)
            record = refs.get(token_key)
            if record is None:
                return
            _, output_lease_owner = record
            output_lease_owner.release()
            if refs.get(token_key) is record:
                refs.pop(token_key, None)
                if not refs:
                    self._leased_result_partition_refs.pop(plan_key, None)

    async def close_plan(self, owner_id: str, session_id: str, plan_id: str) -> None:
        owner_key = str(owner_id).strip()
        session_key = str(session_id).strip()
        plan_key = str(plan_id)
        if not owner_key:
            raise ValueError("Ray query runtime owner_id must not be empty")
        if not session_key:
            raise ValueError("Vane session_id must not be empty")
        with self._session_lock:
            session = self._sessions.get(session_key)
            closed_owner = self._closed_session_owners.get(session_key)
            plan_session_id = self._plan_session_ids.get(plan_key)
        if session is None:
            if closed_owner is None:
                raise RuntimeError(f"Vane session is not open: {session_key}")
            if closed_owner != owner_key:
                raise PermissionError("Vane session operation requires its owning runtime client")
            return
        if session.owner_id != owner_key:
            raise PermissionError("Vane session operation requires its owning runtime client")
        with session.condition:
            owns_plan = plan_key in session.plan_ids
        if plan_session_id is None and not owns_plan:
            return
        if plan_session_id is not None and plan_session_id != session_key:
            raise PermissionError("query plan does not belong to the requested Vane session")
        if plan_session_id is None or not owns_plan:
            raise RuntimeError(f"query plan ownership state is inconsistent: {plan_key}")
        await _to_thread_with_owned_side_effects(self._cleanup_finished_plan, plan_key)

    async def run_plan(
        self,
        owner_id: str,
        session_id: str,
        plan: duckdb.ray_cxx.PyLogicalPlan,
    ) -> None:
        """Run a plan without blocking the driver actor's control event loop."""
        session = self._require_session(owner_id, session_id)
        self._validate_plan_session(session_id, plan, session)
        _set_global_event_loop(asyncio.get_running_loop())
        plan_id = str(plan.idx())
        self._begin_session_operation(session, session_id)
        try:
            try:
                plan, plan_id, query_connection = await _to_thread_with_owned_side_effects(
                    self._prepare_plan_sync,
                    session_id,
                    session,
                    plan,
                    plan_id,
                )
            except asyncio.CancelledError as cancellation_error:
                try:
                    await _to_thread_with_owned_side_effects(
                        self._teardown_plan_resources,
                        plan_id,
                        "",
                        drop_fragments=False,
                    )
                except asyncio.CancelledError:
                    pass
                except BaseException as teardown_error:
                    aggregate_error = QueryExecutionCleanupError.from_errors(
                        f"query plan {plan_id} was cancelled during preparation and teardown failed",
                        cancellation_error,
                        [teardown_error],
                    )
                    raise aggregate_error from cancellation_error
                raise cancellation_error

            self._plan_query_ids[plan_id] = plan_id
            try:
                graph, allocation = await self._register_query_resources(
                    plan,
                    query_connection=query_connection,
                    expected_plan_id=plan_id,
                )
            except BaseException as registration_error:
                try:
                    await _to_thread_with_owned_side_effects(
                        self._teardown_plan_resources,
                        plan_id,
                        plan_id,
                        drop_fragments=False,
                    )
                except asyncio.CancelledError:
                    pass
                except BaseException as teardown_error:
                    aggregate_error = QueryExecutionCleanupError.from_errors(
                        f"query plan {plan_id} registration failed and teardown also failed",
                        registration_error,
                        [teardown_error],
                    )
                    raise aggregate_error from registration_error
                raise
            self._plan_query_ids[plan_id] = str(graph.query_id)

            startup_ownership = _PlanStartupOwnership()
            startup = asyncio.create_task(
                asyncio.to_thread(
                    self._run_plan_sync_with_ownership,
                    plan,
                    plan_id,
                    graph,
                    allocation,
                    query_connection,
                    session.config,
                    startup_ownership,
                ),
                name=f"vane-plan-startup:{plan_id}",
            )
            try:
                await asyncio.shield(startup)
            except asyncio.CancelledError as cancellation_error:
                if startup_ownership.cancel_before_worker_claim():
                    startup.cancel()
                    await asyncio.gather(startup, return_exceptions=True)
                    try:
                        await _to_thread_with_owned_side_effects(
                            self._teardown_plan_resources,
                            plan_id,
                            str(graph.query_id),
                            drop_fragments=False,
                        )
                    except asyncio.CancelledError:
                        pass
                    except BaseException as teardown_error:
                        aggregate_error = QueryExecutionCleanupError.from_errors(
                            f"query plan {plan_id} was cancelled before startup and teardown failed",
                            cancellation_error,
                            [teardown_error],
                        )
                        raise aggregate_error from cancellation_error
                else:
                    try:
                        await asyncio.shield(startup)
                    except BaseException as startup_error:
                        raise startup_error from cancellation_error
                    try:
                        await _to_thread_with_owned_side_effects(
                            self._teardown_plan_resources,
                            plan_id,
                            str(graph.query_id),
                            drop_fragments=True,
                        )
                    except asyncio.CancelledError:
                        pass
                    except BaseException as teardown_error:
                        aggregate_error = QueryExecutionCleanupError.from_errors(
                            f"query plan {plan_id} was cancelled during startup and teardown failed",
                            cancellation_error,
                            [teardown_error],
                        )
                        raise aggregate_error from cancellation_error
                raise
        finally:
            self._end_session_operation(session)

    def _prepare_plan_sync(
        self,
        session_id: str,
        session: _DriverSession,
        plan: duckdb.ray_cxx.PyLogicalPlan,
        plan_id: str,
    ) -> tuple[Any, str, Any]:
        """Prepare a physical plan on an owned worker thread."""
        if os.environ.get("VANE_WORKER") == "1":
            raise RuntimeError(
                "RayQueryDriverActor.run_plan() called inside a Worker Worker. "
                "Nested distributed execution is not supported."
            )
        with session.operation_lock:
            logical_plan = plan
            query_connection = session.connection.cursor()
            with self._session_lock:
                if self._sessions.get(str(session_id)) is not session:
                    query_connection.close()
                    raise RuntimeError(f"Vane session closed during query startup: {session_id}")
                if plan_id in self._plan_session_ids:
                    query_connection.close()
                    raise RuntimeError(f"query plan identity is already active: {plan_id}")
                session.plan_ids.add(plan_id)
                self._plan_session_ids[plan_id] = str(session_id)
                self._plan_connections[plan_id] = query_connection

            try:
                from duckdb.runners.ray.worker import (
                    _configure_duckdb_s3,
                    _refresh_effective_duckdb_s3_config,
                )

                use_session_credentials = not bool(logical_plan.has_explicit_s3_credentials())
                refreshed_s3_config = _refresh_effective_duckdb_s3_config(
                    session.config,
                    session.s3_config,
                    use_session_credentials=use_session_credentials,
                )
                session.s3_config = _configure_duckdb_s3(
                    query_connection,
                    refreshed_s3_config,
                    use_session_credentials=use_session_credentials,
                )
                physical_plan = logical_plan.to_physical_plan(
                    query_connection,
                    session.s3_config,
                )
                physical_plan_id = str(physical_plan.idx())
                if physical_plan_id != plan_id:
                    raise RuntimeError(
                        f"logical/physical query plan identity changed: logical={plan_id!r} "
                        f"physical={physical_plan_id!r}"
                    )
                self._validate_plan_session(session_id, physical_plan, session)
            except BaseException:
                self._release_plan_session_state(plan_id)
                raise
        return physical_plan, plan_id, query_connection

    def _run_plan_sync(
        self,
        plan: Any,
        plan_id: str,
        graph: Any,
        allocation: Any,
        query_connection: Any,
        session_config: dict[str, str],
    ) -> None:
        """Start an already registered streaming plan on a worker thread."""
        plan_runner_started = False
        try:
            udf_actors = self._precreate_udf_actors(
                plan,
                graph,
                allocation,
                query_connection=query_connection,
                session_config=session_config,
            )
            if udf_actors:
                self._active_udf_actors_by_plan[plan_id] = list(udf_actors)
            self._precreate_vllm_actors(
                plan,
                query_connection=query_connection,
                session_config=session_config,
            )

            self.curr_plans[plan_id] = plan
            self._plan_query_ids[plan_id] = str(graph.query_id)
            plan_runner = self._get_plan_runner()
            plan_runner_started = True
            self.curr_streams[plan_id] = plan_runner.run_plan(plan, query_connection)
            # Native FTE execution is intentionally started before the actor
            # readiness fence opens.  It can initialize DuckDB's real
            # Fragment/Pipeline topology, but its Ray-actor UDF dispatcher
            # cannot obtain a QRM lease until model initialization succeeds.
            self._wait_for_udf_actors_ready(udf_actors)
            self._mark_query_actor_stages_ready(graph)
        except BaseException as start_error:
            query_id = self._plan_query_ids.get(plan_id, "")
            try:
                self._teardown_plan_resources(
                    plan_id,
                    query_id,
                    drop_fragments=plan_runner_started,
                )
            except BaseException as teardown_error:
                aggregate_error = QueryExecutionCleanupError.from_errors(
                    f"query plan {plan_id} failed to start and teardown also failed",
                    start_error,
                    [teardown_error],
                )
                raise aggregate_error from start_error
            raise

    def _run_plan_sync_with_ownership(
        self,
        plan: Any,
        plan_id: str,
        graph: Any,
        allocation: Any,
        query_connection: Any,
        session_config: dict[str, str],
        startup_ownership: _PlanStartupOwnership,
    ) -> None:
        """Start only after atomically claiming the registered resources."""
        if not startup_ownership.claim_for_worker():
            return
        self._run_plan_sync(
            plan,
            plan_id,
            graph,
            allocation,
            query_connection,
            session_config,
        )

    def _ensure_copy_operation_state(self) -> None:
        if not isinstance(getattr(self, "_copy_operations_inflight", None), dict):
            self._copy_operations_inflight = {}
        if not isinstance(getattr(self, "_copy_operation_terminal", None), BoundedReplayMap):
            self._copy_operation_terminal = BoundedReplayMap(capacity=_COPY_OPERATION_REPLAY_CAPACITY)
        if not isinstance(getattr(self, "_copy_operation_identities", None), dict):
            self._copy_operation_identities = {}
        if not isinstance(getattr(self, "_copy_cleanup_tasks", None), dict):
            self._copy_cleanup_tasks = {}

    def _discard_copy_operation_state_for_owner(self, owner_id: str) -> None:
        self._ensure_copy_operation_state()
        operation_ids = {
            operation_id
            for operation_id, (record_owner_id, _) in self._copy_operation_identities.items()
            if record_owner_id == owner_id
        }
        for operation_id in operation_ids:
            self._copy_operation_identities.pop(operation_id, None)
        self._copy_operation_terminal.discard_where(
            lambda operation_id, record: operation_id in operation_ids and record.owner_id == owner_id
        )

    @staticmethod
    def _copy_operation_id(plan: Any) -> str:
        operation_id = str(plan.idx()).strip()
        if not operation_id:
            raise ValueError("COPY operation identity must not be empty")
        return operation_id

    @staticmethod
    def _validate_copy_operation_identity(
        record: _CopyOperationInFlight | _CopyOperationTerminal,
        *,
        owner_id: str,
        session_id: str | None,
        operation_id: str,
    ) -> None:
        if record.owner_id != owner_id or (session_id is not None and record.session_id != session_id):
            raise PermissionError(
                f"COPY operation {operation_id} does not belong to the requested runtime owner and session"
            )

    @staticmethod
    def _validate_copy_identity_tombstone(
        identity: tuple[str, str],
        *,
        owner_id: str,
        session_id: str | None,
        operation_id: str,
    ) -> None:
        record_owner_id, record_session_id = identity
        if record_owner_id != owner_id or (session_id is not None and record_session_id != session_id):
            raise PermissionError(
                f"COPY operation {operation_id} does not belong to the requested runtime owner and session"
            )

    @staticmethod
    def _copy_terminal_result(record: _CopyOperationTerminal) -> CopyPlanOutcome:
        if record.error is not None:
            raise record.error
        if record.outcome is None:
            raise RuntimeError("COPY terminal replay record has no result")
        return record.outcome

    @staticmethod
    def _copy_terminal_recovery(
        operation_id: str,
        record: _CopyOperationTerminal,
    ) -> CopyPlanRecovery:
        if (record.outcome is None) == (record.error is None):
            raise RuntimeError("COPY terminal replay record must contain exactly one result")
        return CopyPlanRecovery(
            operation_id=operation_id,
            outcome=record.outcome,
            error=record.error,
        )

    @staticmethod
    async def _await_copy_operation(task: asyncio.Task[CopyPlanOutcome]) -> CopyPlanOutcome:
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                # The write owns its operation identity until a terminal
                # outcome is recorded. Do not let cancellation of one waiter
                # cancel the write or interrupt observation of its outcome.
                cancellation = error
            except BaseException:
                break
        if cancellation is not None:
            try:
                task.result()
            except BaseException:
                pass
            raise cancellation
        return task.result()

    @staticmethod
    async def _drain_copy_startup_tasks(tasks: list[asyncio.Task[Any]]) -> None:
        if not tasks:
            return
        drain = asyncio.gather(*tasks, return_exceptions=True)
        while not drain.done():
            try:
                await asyncio.shield(drain)
            except asyncio.CancelledError:
                # Once native COPY may have committed, cancellation cannot
                # interrupt observation of the remaining startup tasks.
                continue
        drain.result()

    @staticmethod
    def _copy_cleanup_warning(stage: str, error: BaseException) -> str:
        try:
            message = str(error)
        except BaseException:
            message = "<error message unavailable>"
        return f"{stage} failed: {type(error).__name__}: {message}"

    @staticmethod
    def _copy_native_cleanup_warnings(result: Mapping[str, Any]) -> tuple[str, ...]:
        raw_warnings = result.get("copy_runner_cleanup_warnings", ())
        if isinstance(raw_warnings, str):
            return (raw_warnings,) if raw_warnings else ()
        if not isinstance(raw_warnings, (list, tuple)):
            return ()
        return tuple(str(warning) for warning in raw_warnings if str(warning))

    @staticmethod
    def _committed_copy_outcome(
        operation_id: str,
        result: Mapping[str, Any],
        final_progress_snapshot: dict[str, Any] | None,
        *,
        cleanup_state: Literal["complete", "pending"],
        cleanup_warnings: tuple[str, ...],
    ) -> CopyPlanOutcome:
        public_result = dict(result)
        public_result["copy_operation_id"] = operation_id
        public_result["copy_write_state"] = "committed"
        public_result["copy_cleanup_state"] = cleanup_state
        public_result["copy_cleanup_warnings"] = list(cleanup_warnings)
        return CopyPlanOutcome(
            result=public_result,
            final_progress_snapshot=final_progress_snapshot,
            operation_id=operation_id,
            write_state="committed",
            cleanup_state=cleanup_state,
            cleanup_warnings=cleanup_warnings,
        )

    async def _execute_copy_operation(
        self,
        owner_id: str,
        session_id: str,
        session: _DriverSession,
        plan: Any,
        operation_id: str,
    ) -> CopyPlanOutcome:
        try:
            self._begin_session_operation(session, session_id)
            try:
                outcome = await self._run_copy_plan_for_session(
                    session_id,
                    session,
                    plan,
                )
            finally:
                self._end_session_operation(session)
        except BaseException as error:
            self._copy_operation_terminal[operation_id] = _CopyOperationTerminal(
                owner_id=owner_id,
                session_id=session_id,
                error=error,
            )
            raise
        else:
            self._copy_operation_terminal[operation_id] = _CopyOperationTerminal(
                owner_id=owner_id,
                session_id=session_id,
                outcome=outcome,
            )
            return outcome
        finally:
            current = self._copy_operations_inflight.get(operation_id)
            if current is not None and current.task is asyncio.current_task():
                self._copy_operations_inflight.pop(operation_id, None)

    async def run_copy_plan(
        self,
        owner_id: str,
        session_id: str,
        plan: Any,
    ) -> CopyPlanOutcome:
        """Run one idempotently identified COPY and replay its terminal outcome."""
        self._ensure_copy_operation_state()
        owner_key = str(owner_id).strip()
        session_key = str(session_id).strip()
        operation_id = self._copy_operation_id(plan)
        session = self._require_session(owner_id, session_id)
        self._validate_plan_session(session_id, plan, session)
        terminal = self._copy_operation_terminal.get(operation_id)
        if terminal is not None:
            self._validate_copy_operation_identity(
                terminal,
                owner_id=owner_key,
                session_id=session_key,
                operation_id=operation_id,
            )
            self._copy_operation_identities.setdefault(
                operation_id,
                (terminal.owner_id, terminal.session_id),
            )
            return self._copy_terminal_result(terminal)
        in_flight = self._copy_operations_inflight.get(operation_id)
        if in_flight is not None:
            self._validate_copy_operation_identity(
                in_flight,
                owner_id=owner_key,
                session_id=session_key,
                operation_id=operation_id,
            )
            self._copy_operation_identities.setdefault(
                operation_id,
                (in_flight.owner_id, in_flight.session_id),
            )
            return await self._await_copy_operation(in_flight.task)
        identity = self._copy_operation_identities.get(operation_id)
        if identity is not None:
            self._validate_copy_identity_tombstone(
                identity,
                owner_id=owner_key,
                session_id=session_key,
                operation_id=operation_id,
            )
            raise CopyOutcomeUnknownError(operation_id)
        self._copy_operation_identities[operation_id] = (
            owner_key,
            session_key,
        )
        task = asyncio.create_task(
            self._execute_copy_operation(
                owner_key,
                session_key,
                session,
                plan,
                operation_id,
            ),
            name=f"vane-copy-operation:{operation_id}",
        )
        self._copy_operations_inflight[operation_id] = _CopyOperationInFlight(
            owner_id=owner_key,
            session_id=session_key,
            task=task,
        )
        return await self._await_copy_operation(task)

    async def _recover_copy_operation(
        self,
        owner_id: str,
        operation_id: str,
        *,
        session_id: str | None,
        wait_timeout_s: float | None = None,
    ) -> CopyPlanRecovery:
        self._ensure_copy_operation_state()
        owner_key = str(owner_id).strip()
        session_key = None if session_id is None else str(session_id).strip()
        operation_key = str(operation_id).strip()
        self._require_owner(owner_key)
        if session_id is not None and not session_key:
            raise ValueError("Vane session_id must not be empty")
        if not operation_key:
            raise ValueError("COPY operation identity must not be empty")
        terminal = self._copy_operation_terminal.get(operation_key)
        if terminal is not None:
            self._validate_copy_operation_identity(
                terminal,
                owner_id=owner_key,
                session_id=session_key,
                operation_id=operation_key,
            )
            return self._copy_terminal_recovery(operation_key, terminal)
        in_flight = self._copy_operations_inflight.get(operation_key)
        if in_flight is not None:
            self._validate_copy_operation_identity(
                in_flight,
                owner_id=owner_key,
                session_id=session_key,
                operation_id=operation_key,
            )
            if wait_timeout_s is not None:
                # asyncio.wait leaves a pending task running. Reconciliation
                # timing out must never cancel a COPY that may still commit.
                done, _ = await asyncio.wait(
                    {in_flight.task},
                    timeout=wait_timeout_s,
                )
                if not done:
                    terminal = self._copy_operation_terminal.get(operation_key)
                    if terminal is not None:
                        self._validate_copy_operation_identity(
                            terminal,
                            owner_id=owner_key,
                            session_id=session_key,
                            operation_id=operation_key,
                        )
                        return self._copy_terminal_recovery(operation_key, terminal)
                    raise CopyOutcomeUnknownError(operation_key)
            try:
                if wait_timeout_s is None:
                    outcome = await self._await_copy_operation(in_flight.task)
                else:
                    outcome = in_flight.task.result()
            except asyncio.CancelledError:
                raise
            except BaseException as operation_error:
                terminal = self._copy_operation_terminal.get(operation_key)
                if terminal is not None:
                    self._validate_copy_operation_identity(
                        terminal,
                        owner_id=owner_key,
                        session_id=session_key,
                        operation_id=operation_key,
                    )
                    return self._copy_terminal_recovery(operation_key, terminal)
                return CopyPlanRecovery(
                    operation_id=operation_key,
                    error=operation_error,
                )
            return CopyPlanRecovery(
                operation_id=operation_key,
                outcome=outcome,
            )
        identity = self._copy_operation_identities.get(operation_key)
        if identity is not None:
            self._validate_copy_identity_tombstone(
                identity,
                owner_id=owner_key,
                session_id=session_key,
                operation_id=operation_key,
            )
        raise CopyOutcomeUnknownError(operation_key)

    async def recover_copy_plan(
        self,
        owner_id: str,
        session_id: str,
        operation_id: str,
    ) -> CopyPlanRecovery:
        """Read or briefly await an existing COPY outcome without submitting a write."""
        return await self._recover_copy_operation(
            owner_id,
            operation_id,
            session_id=session_id,
            wait_timeout_s=_copy_reconciliation_timeout_s(),
        )

    async def _retry_copy_cleanup_for_operation(
        self,
        operation_id: str,
        terminal: _CopyOperationTerminal,
        outcome: CopyPlanOutcome,
    ) -> CopyPlanOutcome:
        cleanup_warnings = outcome.cleanup_warnings
        cleanup_state: Literal["complete", "pending"] = "complete"
        try:
            await _to_thread_with_owned_side_effects(
                self._cleanup_finished_plan,
                operation_id,
            )
        except asyncio.CancelledError:
            # The owned cleanup call completed before waiter cancellation was
            # exposed, so there is no cleanup work left to retry.
            pass
        except BaseException as cleanup_error:
            cleanup_state = "pending"
            cleanup_warnings += (self._copy_cleanup_warning("COPY cleanup retry", cleanup_error),)
        updated = self._committed_copy_outcome(
            operation_id,
            outcome.result,
            outcome.final_progress_snapshot,
            cleanup_state=cleanup_state,
            cleanup_warnings=cleanup_warnings,
        )
        current = self._copy_operation_terminal.get(operation_id)
        if current is terminal:
            self._copy_operation_terminal[operation_id] = replace(
                terminal,
                outcome=updated,
            )
        return updated

    async def retry_copy_cleanup(
        self,
        owner_id: str,
        operation_id: str,
    ) -> CopyPlanOutcome:
        """Retry only post-commit cleanup for an existing COPY operation."""
        recovery = await self._recover_copy_operation(
            owner_id,
            operation_id,
            session_id=None,
        )
        if recovery.error is not None:
            raise recovery.error
        outcome = recovery.outcome
        if outcome is None:
            raise CopyOutcomeUnknownError(str(operation_id).strip())
        if outcome.cleanup_state == "complete":
            return outcome
        operation_key = str(operation_id).strip()
        terminal = self._copy_operation_terminal.get(operation_key)
        if terminal is None or terminal.outcome is None:
            raise CopyOutcomeUnknownError(operation_key)
        cleanup_task = self._copy_cleanup_tasks.get(operation_key)
        if cleanup_task is None:
            cleanup_task = asyncio.create_task(
                self._retry_copy_cleanup_for_operation(
                    operation_key,
                    terminal,
                    outcome,
                ),
                name=f"vane-copy-cleanup:{operation_key}",
            )
            self._copy_cleanup_tasks[operation_key] = cleanup_task

            def discard_completed_cleanup(completed: asyncio.Task[CopyPlanOutcome]) -> None:
                if self._copy_cleanup_tasks.get(operation_key) is completed:
                    self._copy_cleanup_tasks.pop(operation_key, None)

            cleanup_task.add_done_callback(discard_completed_cleanup)
        return await self._await_copy_operation(cleanup_task)

    async def _run_copy_plan_for_session(
        self,
        session_id: str,
        session: _DriverSession,
        plan: Any,
    ) -> CopyPlanOutcome:
        logical_plan = plan
        plan_id = str(logical_plan.idx())
        graph = None
        plan_runner_started = False
        plan_execution: asyncio.Task[Any] | None = None
        startup_tasks: list[asyncio.Task[Any]] = []
        try:
            plan, query_connection = await _to_thread_with_owned_side_effects(
                self._prepare_copy_plan_sync,
                session_id,
                session,
                logical_plan,
                plan_id,
            )
            self._plan_query_ids[plan_id] = plan_id
            graph, allocation = await self._register_query_resources(
                plan,
                query_connection=query_connection,
                expected_plan_id=plan_id,
            )
            self._plan_query_ids[plan_id] = str(graph.query_id)
            udf_actors = await _to_thread_with_owned_side_effects(
                self._precreate_udf_actors,
                plan,
                graph,
                allocation,
                query_connection=query_connection,
                session_config=session.config,
            )
            await _to_thread_with_owned_side_effects(
                self._precreate_vllm_actors,
                plan,
                query_connection=query_connection,
                session_config=session.config,
            )
            plan_runner = self._get_plan_runner()
            plan_runner_started = True
            plan_execution = asyncio.create_task(
                asyncio.to_thread(
                    plan_runner.run_copy_plan,
                    plan,
                    query_connection,
                ),
                name=f"vane-copy-plan:{plan_id}",
            )
            if udf_actors:
                from duckdb.runners.ray.fte_fragment_scheduler import (
                    wait_for_fte_query_progress_topology,
                )

                topology_ready = asyncio.create_task(
                    asyncio.to_thread(
                        wait_for_fte_query_progress_topology,
                        graph.query_id,
                        timeout_s=_progress_topology_init_timeout_s(),
                    ),
                    name=f"vane-copy-topology-ready:{plan_id}",
                )
                actors_ready = asyncio.create_task(
                    asyncio.to_thread(self._wait_for_udf_actors_ready, udf_actors),
                    name=f"vane-copy-actors-ready:{plan_id}",
                )
                startup_tasks.extend((topology_ready, actors_ready))
                pending_startup = set(startup_tasks)
                plan_execution_observed = False
                while pending_startup:
                    wait_tasks = set(pending_startup)
                    if not plan_execution_observed:
                        wait_tasks.add(plan_execution)
                    done, _ = await asyncio.wait(
                        wait_tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if plan_execution in done:
                        # Propagate a native-plan failure immediately. A
                        # successful zero-row COPY may legitimately finish
                        # before a slow actor has initialized, so keep waiting
                        # for the startup barriers instead of rewriting that
                        # success as an execution error.
                        await self._await_copy_operation(plan_execution)
                        plan_execution_observed = True
                    for task in done:
                        if task is plan_execution:
                            continue
                        await task
                        pending_startup.discard(task)
            self._mark_query_actor_stages_ready(graph)
            # Cancellation must reach the exception path promptly so teardown
            # can interrupt a still-running native write, but it must not
            # cancel the asyncio wrapper around that thread. The exception path
            # waits for the wrapper's terminal result and checks the commit
            # marker before classifying the operation.
            result = await asyncio.shield(plan_execution)
            if not isinstance(result, Mapping):
                raise TypeError(f"native COPY returned {type(result).__name__}, expected a mapping")
            if result.get("copy_output_outcome_unknown") is True:
                raise CopyOutcomeUnknownError.from_native_result(plan_id, result)
            if result.get("copy_output_committed") is not True:
                raise RuntimeError(f"native COPY plan {plan_id} returned without a committed output marker")
        except BaseException as execution_error:
            query_id = self._plan_query_ids.get(plan_id, "")
            teardown_error: BaseException | None = None
            try:
                await _to_thread_with_owned_side_effects(
                    self._teardown_plan_resources,
                    plan_id,
                    query_id,
                    drop_fragments=plan_runner_started,
                )
            except asyncio.CancelledError:
                # The owned teardown finished before cancellation was exposed.
                pass
            except BaseException as error:
                teardown_error = error
            # Teardown interrupts the native runner. Always retrieve its
            # terminal result so a concurrent failure cannot escape as an
            # unobserved asyncio task exception.
            committed_result: Mapping[str, Any] | None = None
            unknown_outcome_error = execution_error if isinstance(execution_error, CopyOutcomeUnknownError) else None
            if plan_execution is not None:
                try:
                    await self._await_copy_operation(plan_execution)
                except BaseException:
                    pass
                try:
                    observed_result = plan_execution.result()
                except BaseException:
                    pass
                else:
                    if isinstance(observed_result, Mapping):
                        if observed_result.get("copy_output_outcome_unknown") is True:
                            if unknown_outcome_error is None:
                                unknown_outcome_error = CopyOutcomeUnknownError.from_native_result(
                                    plan_id,
                                    observed_result,
                                )
                        elif observed_result.get("copy_output_committed") is True:
                            committed_result = observed_result
            await self._drain_copy_startup_tasks(startup_tasks)
            if committed_result is not None:
                cleanup_warnings = self._copy_native_cleanup_warnings(committed_result) + (
                    self._copy_cleanup_warning(
                        "COPY post-commit execution finalization",
                        execution_error,
                    ),
                )
                cleanup_state: Literal["complete", "pending"] = "complete"
                if teardown_error is not None:
                    cleanup_state = "pending"
                    cleanup_warnings += (
                        self._copy_cleanup_warning(
                            "COPY post-commit teardown",
                            teardown_error,
                        ),
                    )
                try:
                    _log_copy_result_debug(str(query_id or plan_id), committed_result)
                except BaseException as logging_error:
                    cleanup_warnings += (
                        self._copy_cleanup_warning(
                            "COPY result logging",
                            logging_error,
                        ),
                    )
                return self._committed_copy_outcome(
                    plan_id,
                    committed_result,
                    None,
                    cleanup_state=cleanup_state,
                    cleanup_warnings=cleanup_warnings,
                )
            if unknown_outcome_error is not None:
                if unknown_outcome_error is not execution_error:
                    unknown_outcome_error.add_cleanup_warnings(
                        self._copy_cleanup_warning(
                            "COPY outcome-unknown execution finalization",
                            execution_error,
                        )
                    )
                if teardown_error is not None:
                    unknown_outcome_error.add_cleanup_warnings(
                        self._copy_cleanup_warning("COPY outcome-unknown teardown", teardown_error)
                    )
                if unknown_outcome_error is execution_error:
                    raise
                raise unknown_outcome_error from execution_error
            if teardown_error is not None:
                aggregate_error = QueryExecutionCleanupError.from_errors(
                    f"COPY query plan {plan_id} failed and teardown also failed",
                    execution_error,
                    [teardown_error],
                )
                raise aggregate_error from execution_error
            raise
        query_id = str(graph.query_id)
        cleanup_warnings = self._copy_native_cleanup_warnings(result)
        try:
            _log_copy_result_debug(query_id, result)
        except BaseException as logging_error:
            cleanup_warnings += (
                self._copy_cleanup_warning(
                    "COPY result logging",
                    logging_error,
                ),
            )
        terminal_reason = str(getattr(self, "_query_terminal_errors", {}).get(query_id, ""))
        if terminal_reason:
            cleanup_warnings += (f"COPY post-commit terminal query state: {terminal_reason}",)
        try:
            final_progress_snapshot = await _to_thread_with_owned_side_effects(
                self._build_local_progress_snapshot,
                query_id,
                None,
            )
        except BaseException as progress_error:
            final_progress_snapshot = None
            cleanup_warnings += (
                self._copy_cleanup_warning(
                    "COPY progress finalization",
                    progress_error,
                ),
            )
        final_cleanup_state: Literal["complete", "pending"] = "complete"
        try:
            await _to_thread_with_owned_side_effects(
                self._teardown_plan_resources,
                plan_id,
                query_id,
                drop_fragments=True,
            )
        except asyncio.CancelledError:
            # The owned teardown completed before cancellation was exposed.
            pass
        except BaseException as teardown_error:
            final_cleanup_state = "pending"
            cleanup_warnings += (
                self._copy_cleanup_warning(
                    "COPY post-commit teardown",
                    teardown_error,
                ),
            )
        return self._committed_copy_outcome(
            plan_id,
            result,
            final_progress_snapshot,
            cleanup_state=final_cleanup_state,
            cleanup_warnings=cleanup_warnings,
        )

    def _prepare_copy_plan_sync(
        self,
        session_id: str,
        session: _DriverSession,
        logical_plan: Any,
        plan_id: str,
    ) -> tuple[Any, Any]:
        """Build a COPY plan on an owned thread without blocking actor control RPCs."""
        with session.operation_lock:
            query_connection = session.connection.cursor()
            with self._session_lock:
                if self._sessions.get(str(session_id)) is not session:
                    query_connection.close()
                    raise RuntimeError(f"Vane session closed during COPY startup: {session_id}")
                if plan_id in self._plan_session_ids:
                    query_connection.close()
                    raise RuntimeError(f"query plan identity is already active: {plan_id}")
                session.plan_ids.add(plan_id)
                self._plan_session_ids[plan_id] = str(session_id)
                self._plan_connections[plan_id] = query_connection
            try:
                from duckdb.runners.ray.worker import (
                    _configure_duckdb_s3,
                    _refresh_effective_duckdb_s3_config,
                )

                use_session_credentials = not bool(logical_plan.has_explicit_s3_credentials())
                refreshed_s3_config = _refresh_effective_duckdb_s3_config(
                    session.config,
                    session.s3_config,
                    use_session_credentials=use_session_credentials,
                )
                session.s3_config = _configure_duckdb_s3(
                    query_connection,
                    refreshed_s3_config,
                    use_session_credentials=use_session_credentials,
                )
                plan = logical_plan.to_physical_plan(
                    query_connection,
                    session.s3_config,
                )
                physical_plan_id = str(plan.idx())
                if physical_plan_id != plan_id:
                    raise RuntimeError(
                        f"logical/physical query plan identity changed: logical={plan_id!r} "
                        f"physical={physical_plan_id!r}"
                    )
                self._validate_plan_session(session_id, plan, session)
            except BaseException:
                self._release_plan_session_state(plan_id)
                raise
        return plan, query_connection

    async def get_next_partition(
        self,
        owner_id: str,
        session_id: str,
        plan_id: str,
        release_owner: Any | None = None,
    ) -> RayMaterializedResult | None:
        session = self._require_plan_session(owner_id, session_id, plan_id)
        self._begin_session_operation(session, session_id)
        try:
            return await self._get_next_partition_for_session(
                session,
                owner_id,
                session_id,
                plan_id,
                release_owner,
            )
        finally:
            self._end_session_operation(session)

    async def _get_next_partition_for_session(
        self,
        session: _DriverSession,
        owner_id: str,
        session_id: str,
        plan_id: str,
        release_owner: Any | None,
    ) -> RayMaterializedResult | None:
        from duckdb.runners.ray.partition_metadata import (
            PartitionMetadata,
            PartitionMetadataAccessor,
            RayMaterializedResult,
        )

        if plan_id not in self.curr_streams:
            raise ValueError(f"Plan {plan_id} not found in DriverPlanRunner")

        query_id = self._plan_query_ids.get(str(plan_id))
        if not query_id:
            raise RuntimeError(f"Plan {plan_id} is missing its registered query resource owner")
        if query_id in getattr(self, "_query_terminal_errors", {}):
            await _to_thread_with_owned_side_effects(
                self._finish_terminal_query,
                str(plan_id),
                query_id,
            )
        from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

        manager = get_query_resource_manager(query_id)
        manager.set_external_consumer_waiting(True)
        try:
            try:
                # C++ ResultPartitionStream path: use blocking_next in a thread.
                # IMPORTANT: blocking_next raises StopIteration when stream is exhausted.
                # Python 3.12 forbids StopIteration inside asyncio Futures, so we must
                # catch it in the thread and convert to a sentinel before it reaches
                # the event loop.
                stream = self.curr_streams[plan_id]

                def _safe_blocking_next() -> Any | None:
                    try:
                        return stream.blocking_next()
                    except StopIteration:
                        return None
                    except RuntimeError as e:
                        if "StopIteration" in str(e):
                            return None
                        raise

                next_item = await asyncio.to_thread(_safe_blocking_next)
            except (StopIteration, StopAsyncIteration):
                await _to_thread_with_owned_side_effects(self._cleanup_finished_plan, plan_id)
                return None
            except RuntimeError as e:
                # pybind11 wraps StopIteration in RuntimeError
                if "StopIteration" in str(e):
                    await _to_thread_with_owned_side_effects(self._cleanup_finished_plan, plan_id)
                    return None
                raise
        finally:
            manager.set_external_consumer_waiting(False)
        if query_id in getattr(self, "_query_terminal_errors", {}):
            await _to_thread_with_owned_side_effects(
                self._finish_terminal_query,
                str(plan_id),
                query_id,
            )
        if next_item is None:
            await _to_thread_with_owned_side_effects(self._cleanup_finished_plan, plan_id)
            return None

        if isinstance(next_item, RayMaterializedResult):
            return next_item
        if isinstance(next_item, WorkerTaskMetadata):
            await _to_thread_with_owned_side_effects(self._cleanup_finished_plan, plan_id)
            return None

        if hasattr(next_item, "object_ref"):
            object_ref = next_item.object_ref
            output_lease_owner = next_item.lease_owner
            if not callable(getattr(output_lease_owner, "transition_to", None)):
                raise TypeError("metadata-aware Ray result is missing its output lease owner")
            try:
                release_token = self._lease_result_partition_ref(
                    session,
                    session_id,
                    str(plan_id),
                    object_ref,
                    output_lease_owner,
                )
            except BaseException:
                output_lease_owner.release()
                raise
            metadata_accessor = PartitionMetadataAccessor.from_metadata_list(
                [PartitionMetadata(next_item.num_rows, next_item.size_bytes)]
            )
            return RayMaterializedResult(
                partition=object_ref,
                metadatas=metadata_accessor,
                metadata_idx=0,
                release_owner=release_owner,
                release_owner_id=str(owner_id),
                release_session_id=str(session_id),
                release_plan_id=str(plan_id),
                release_token=release_token,
            )
        raise TypeError(f"expected metadata-aware fragment from stream, got {type(next_item).__name__}")


def get_head_node() -> dict[str, Any] | None:
    for node in ray.nodes():
        if (
            "Resources" in node
            and "node:__internal_head__" in node["Resources"]
            and node["Resources"]["node:__internal_head__"] == 1
        ):
            return node
    return None


def get_head_node_id() -> str | None:
    node = get_head_node()
    return None if node is None else str(node["NodeID"])


def _maybe_set_distributed_cluster_capacity() -> None:
    """Auto-detect Ray cluster capacity and set distributed node/slot env vars."""
    need_node_count = not os.environ.get("VANE_DISTRIBUTED_NODE_COUNT")
    need_worker_slots = not os.environ.get("VANE_DISTRIBUTED_WORKER_SLOTS")
    if not need_node_count and not need_worker_slots:
        return
    try:
        nodes = ray.nodes()
    except Exception as exc:
        raise RuntimeError(f"Failed to query Ray cluster nodes: {exc}") from exc
    if not nodes:
        return

    def _usable(node: dict[str, Any]) -> bool:
        if not node.get("Alive", True):
            return False
        resources = node.get("Resources", {}) or {}
        return resources.get("CPU", 0) > 0 and resources.get("memory", 0) > 0

    usable_nodes = [node for node in nodes if _usable(node)]
    if not usable_nodes:
        return

    if need_node_count:
        os.environ["VANE_DISTRIBUTED_NODE_COUNT"] = str(len(usable_nodes))

    if need_worker_slots:
        try:
            min_cpu_per_task = max(1, int(os.environ.get("VANE_MIN_CPU_PER_TASK", "1")))
        except ValueError:
            min_cpu_per_task = 1

        total_slots = 0
        for node in usable_nodes:
            cpu = float((node.get("Resources", {}) or {}).get("CPU", 0))
            if cpu <= 0:
                continue
            slots = math.floor(cpu / min_cpu_per_task)
            total_slots += max(1, slots)

        if total_slots <= 0:
            total_slots = len(usable_nodes)
        os.environ["VANE_DISTRIBUTED_WORKER_SLOTS"] = str(total_slots)


class RayQueryDriverClient:
    """Client attachment to the Ray Job-scoped Vane runtime actor."""

    def __init__(self) -> None:
        self._owner_id = uuid.uuid4().hex
        self._lease_token = uuid.uuid4().hex
        (
            self._client_lease_timeout_s,
            self._client_heartbeat_interval_s,
        ) = _client_owner_lease_settings()
        self._client_heartbeat_rpc_timeout_s = max(
            0.001,
            min(
                self._client_heartbeat_interval_s,
                self._client_lease_timeout_s / 3.0,
            ),
        )
        self._client_heartbeat_stop = threading.Event()
        self._client_heartbeat_thread: threading.Thread | None = None
        self._client_heartbeat_error = ""
        self._opened_sessions: dict[str, dict[str, str]] = {}
        self._uncertain_sessions: dict[str, dict[str, str]] = {}
        self._opening_session_ids: set[str] = set()
        self._closing_session_ids: set[str] = set()
        self._closed_session_ids = BoundedReplayMap[str, bool](capacity=_SESSION_CLOSE_REPLAY_CAPACITY)
        self._session_closes_in_progress: set[str] = set()
        self._session_condition = threading.Condition()
        self._client_closing = False
        self._client_close_in_progress = False
        try:
            runtime_context = ray.get_runtime_context()
            self._ray_gcs_address = runtime_context.gcs_address
            self._ray_job_id = _ray_runtime_job_id(runtime_context)
        except Exception:
            self._ray_gcs_address = None
            raise RuntimeError("Ray runtime context is missing the current job identity") from None
        reject_node_local_ray_runtime_env(runtime_context)
        _maybe_set_distributed_cluster_capacity()
        from duckdb.runners.ray.worker_memory import build_ray_node_memory_layout

        head_node = get_head_node()
        if head_node is None:
            raise RuntimeError("Ray cluster has no alive internal head node")
        head_node_id = str(head_node.get("NodeID") or "").strip()
        head_memory_bytes = int(float((head_node.get("Resources") or {}).get("memory", 0) or 0))
        if not head_node_id or head_memory_bytes <= 0:
            raise RuntimeError("Ray head node has no positive logical memory capacity")
        head_memory_layout = build_ray_node_memory_layout(head_memory_bytes)
        driver_duckdb_memory_bytes = head_memory_layout.driver_duckdb_reserve_bytes
        runtime_config = _normalize_session_config(_collect_vane_env_overrides())
        scheduling = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        )
        runner_options = {
            "name": f"{RAY_QUERY_RUNTIME_ACTOR_NAME_PREFIX}{self._ray_job_id}",
            "namespace": RAY_QUERY_RUNTIME_ACTOR_NAMESPACE,
            "get_if_exists": True,
            "scheduling_strategy": scheduling,
            "memory": driver_duckdb_memory_bytes,
            # RayQueryDriverActor must receive PYTHONPATH/env overrides at actor
            # creation time; constructor args are too late for module import.
            "runtime_env": {"env_vars": runtime_config},
        }
        attach_deadline = time.monotonic() + _RUNTIME_ATTACH_TIMEOUT_S
        while True:
            runner = RayQueryDriverActor.options(**runner_options).remote(  # type: ignore[attr-defined]
                runtime_config,
                driver_duckdb_memory_bytes,
            )
            self.runner = runner
            try:
                attach_ref = runner.attach_client.remote(
                    self._owner_id,
                    runtime_config,
                    self._lease_token,
                )
                resolve_object_refs_blocking(
                    attach_ref,
                    timeout=300,
                )
                break
            except BaseException as error:
                self.runner = None
                replacing_runtime = _runtime_actor_is_being_replaced(error)
                if not replacing_runtime:
                    self._detach_after_failed_attach(runner, error)
                    raise
                try:
                    replacement_ready = bool(
                        resolve_object_refs_blocking(
                            runner.runtime_replacement_ready.remote(),
                            timeout=300,
                            honor_query_deadline=False,
                        )
                    )
                    if replacement_ready:
                        ray.kill(runner, no_restart=True)
                except BaseException:
                    pass
                if time.monotonic() >= attach_deadline:
                    raise TimeoutError("timed out waiting for the previous Ray query runtime to exit") from error
                time.sleep(_RUNTIME_ATTACH_RETRY_INTERVAL_S)
        self._start_client_heartbeat()

    @staticmethod
    def _client_heartbeat_worker(
        client_ref: weakref.ReferenceType[RayQueryDriverClient],
        stop: threading.Event,
        interval_s: float,
        rpc_timeout_s: float,
    ) -> None:
        retry_interval_s = min(
            1.0,
            max(0.001, interval_s / 4.0),
        )
        next_interval_s = interval_s
        while not stop.wait(next_interval_s):
            client = client_ref()
            if client is None:
                return
            with client._session_condition:
                runner = getattr(client, "runner", None)
                owner_id = client._owner_id
                lease_token = client._lease_token
            if runner is None:
                return
            terminal = False
            try:
                heartbeat_ref = runner.heartbeat_client.remote(
                    owner_id,
                    lease_token,
                )
                resolve_object_refs_blocking(
                    heartbeat_ref,
                    timeout=rpc_timeout_s,
                    honor_query_deadline=False,
                )
            except BaseException as error:
                terminal = (
                    _runtime_error_contains(error, PermissionError)
                    or _runtime_actor_is_unavailable(error)
                    or client._runtime_is_unavailable_or_replaced()
                )
                with client._session_condition:
                    client._client_heartbeat_error = f"{type(error).__name__}: {error}"
                next_interval_s = retry_interval_s
            else:
                with client._session_condition:
                    client._client_heartbeat_error = ""
                next_interval_s = interval_s
            del client
            if terminal:
                return

    def _start_client_heartbeat(self) -> None:
        thread = self._client_heartbeat_thread
        if thread is not None and thread.is_alive():
            return
        stop = self._client_heartbeat_stop
        client_ref = weakref.ref(self)
        thread = threading.Thread(
            target=type(self)._client_heartbeat_worker,
            args=(
                client_ref,
                stop,
                float(self._client_heartbeat_interval_s),
                float(self._client_heartbeat_rpc_timeout_s),
            ),
            name=f"vane-ray-client-heartbeat:{self._owner_id}",
            daemon=True,
        )
        self._client_heartbeat_thread = thread
        thread.start()

    def _stop_client_heartbeat(self) -> None:
        stop = getattr(self, "_client_heartbeat_stop", None)
        if stop is not None:
            stop.set()
        thread = getattr(self, "_client_heartbeat_thread", None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._client_heartbeat_thread = None

    def __del__(self) -> None:
        stop = getattr(self, "_client_heartbeat_stop", None)
        if stop is not None:
            stop.set()

    def _runtime_is_unavailable_or_replaced(self) -> bool:
        if not ray.is_initialized():
            return True
        try:
            runtime_context = ray.get_runtime_context()
            current_gcs_address = runtime_context.gcs_address
        except Exception:
            return False
        expected_gcs_address = getattr(self, "_ray_gcs_address", None)
        if (
            expected_gcs_address is not None
            and current_gcs_address is not None
            and current_gcs_address != expected_gcs_address
        ):
            return True
        expected_job_id = getattr(self, "_ray_job_id", None)
        if expected_job_id is None:
            return False
        try:
            return _ray_runtime_job_id(runtime_context) != expected_job_id
        except Exception:
            # A context lookup failure is not proof that the old runtime was
            # replaced; let the actor RPC produce the actionable error.
            return False

    def _detach_after_failed_attach(self, runner: Any, attach_error: BaseException) -> None:
        # A failed actor process cannot retain an owner, including constructor
        # failures that must not be classified as a replaceable old runtime.
        if _runtime_actor_is_unavailable(attach_error):
            return
        cleanup_error: BaseException | None = None
        cleanup_deadline = time.monotonic() + _RUNTIME_ATTACH_CLEANUP_TIMEOUT_S
        detach_ref: Any | None = None
        terminal_failures = 0
        while time.monotonic() < cleanup_deadline:
            try:
                if detach_ref is None:
                    detach_ref = runner.detach_client.remote(self._owner_id)
                last_owner = bool(
                    resolve_object_refs_blocking(
                        detach_ref,
                        timeout=max(0.0, cleanup_deadline - time.monotonic()),
                        honor_query_deadline=False,
                    )
                )
            except BaseException as error:
                if (
                    _runtime_error_contains(error, PermissionError)
                    or _runtime_actor_is_unavailable(error)
                    or self._runtime_is_unavailable_or_replaced()
                ):
                    return
                cleanup_error = error
                if _runtime_error_is_wait_timeout(error):
                    time.sleep(_RUNTIME_ATTACH_RETRY_INTERVAL_S)
                    continue
                terminal_failures += 1
                if terminal_failures >= _RUNTIME_ATTACH_CLEANUP_ATTEMPTS:
                    break
                # The detach method failed and restored owner state. Retry the
                # idempotent mutation with a new RPC. Ambiguous waits keep
                # resolving the original ObjectRef instead.
                detach_ref = None
                time.sleep(_RUNTIME_ATTACH_RETRY_INTERVAL_S)
                continue
            if last_owner:
                try:
                    ray.kill(runner, no_restart=True)
                except BaseException:
                    # The actor is drained and has no owners. A later attach can
                    # retire it through runtime_replacement_ready().
                    pass
            return
        assert cleanup_error is not None
        raise RuntimeError(
            "Ray query runtime attach failed and owner detach could not be confirmed: "
            f"attach={type(attach_error).__name__}: {attach_error}; "
            f"detach={type(cleanup_error).__name__}: {cleanup_error}"
        ) from attach_error

    def _complete_session_close_locally(self, session_key: str) -> None:
        with self._session_condition:
            self._session_closes_in_progress.discard(session_key)
            self._opened_sessions.pop(session_key, None)
            self._uncertain_sessions.pop(session_key, None)
            self._closed_session_ids[session_key] = True
            self._closing_session_ids.discard(session_key)
            self._session_condition.notify_all()

    def _complete_client_close_locally(self) -> None:
        with self._session_condition:
            self.runner = None
            self._opened_sessions.clear()
            self._uncertain_sessions.clear()
            self._closing_session_ids.clear()
            self._closed_session_ids.clear()
            self._client_closing = False
            self._client_close_in_progress = False
            self._session_condition.notify_all()
        self._stop_client_heartbeat()

    def _ensure_session(self, plan: Any) -> tuple[str, dict[str, str], Any]:
        session_id = str(plan.session_id()).strip()
        if not session_id:
            raise ValueError("distributed logical plan is missing its Vane session identity")
        session_config = _normalize_session_config(plan.session_config())
        with self._session_condition:
            while True:
                runner = getattr(self, "runner", None)
                if runner is None or self._client_closing:
                    raise RuntimeError("Ray query runtime client is closed")
                if session_id in self._closed_session_ids:
                    raise RuntimeError(f"Vane session is closed: {session_id}")
                if session_id in self._closing_session_ids:
                    raise RuntimeError(f"Vane session is closing: {session_id}")
                existing = self._opened_sessions.get(session_id)
                if existing is not None:
                    if existing != session_config:
                        raise RuntimeError(f"Vane session config changed after open: {session_id}")
                    return session_id, session_config, runner
                uncertain = self._uncertain_sessions.get(session_id)
                if uncertain is not None and uncertain != session_config:
                    raise RuntimeError(f"Vane session config changed after an ambiguous open: {session_id}")
                if session_id not in self._opening_session_ids:
                    self._opening_session_ids.add(session_id)
                    break
                self._session_condition.wait()
        try:
            resolve_object_refs_blocking(
                runner.open_session.remote(self._owner_id, session_id, session_config),
                timeout=300,
            )
        except BaseException as error:
            rejected_without_open = _runtime_error_contains(error, VaneSessionOpenRejectedError)
            with self._session_condition:
                self._opening_session_ids.remove(session_id)
                if not rejected_without_open:
                    self._uncertain_sessions[session_id] = session_config
                self._session_condition.notify_all()
            raise
        with self._session_condition:
            self._opening_session_ids.remove(session_id)
            self._opened_sessions[session_id] = session_config
            self._uncertain_sessions.pop(session_id, None)
            closing = self._client_closing or session_id in self._closing_session_ids
            self._session_condition.notify_all()
        if closing:
            raise RuntimeError(f"Vane session closed while it was opening: {session_id}")
        return session_id, session_config, runner

    def close_session(self, session_id: str) -> None:
        session_key = str(session_id).strip()
        if not session_key:
            return
        with self._session_condition:
            if session_key in self._closed_session_ids:
                return
            self._closing_session_ids.add(session_key)
            while session_key in self._opening_session_ids or session_key in self._session_closes_in_progress:
                self._session_condition.wait()
            if self._client_closing:
                self._closing_session_ids.discard(session_key)
                self._session_condition.notify_all()
                return
            if session_key not in self._opened_sessions and session_key not in self._uncertain_sessions:
                self._closed_session_ids[session_key] = True
                self._closing_session_ids.discard(session_key)
                self._session_condition.notify_all()
                return
            runner = getattr(self, "runner", None)
            if runner is None:
                self._complete_session_close_locally(session_key)
                return
            self._session_closes_in_progress.add(session_key)
        if self._runtime_is_unavailable_or_replaced():
            self._complete_session_close_locally(session_key)
            return
        try:
            resolve_object_refs_blocking(
                runner.close_session.remote(self._owner_id, session_key),
                timeout=300,
                honor_query_deadline=False,
            )
        except BaseException as error:
            if self._runtime_is_unavailable_or_replaced() or _runtime_actor_is_being_replaced(error):
                self._complete_session_close_locally(session_key)
                return
            with self._session_condition:
                self._session_closes_in_progress.remove(session_key)
                self._session_condition.notify_all()
            raise
        self._complete_session_close_locally(session_key)

    def close(self) -> None:
        with self._session_condition:
            while self._client_close_in_progress:
                self._session_condition.wait()
            runner = getattr(self, "runner", None)
            if runner is not None:
                self._client_close_in_progress = True
                self._client_closing = True
                while self._opening_session_ids or self._session_closes_in_progress:
                    self._session_condition.wait()
        if runner is None:
            self._stop_client_heartbeat()
            return
        # Closing is terminal intent. Stop renewal before detach so a hung
        # cleanup is eventually escalated to the actor-side expiry reaper.
        self._stop_client_heartbeat()
        if self._runtime_is_unavailable_or_replaced():
            self._complete_client_close_locally()
            return
        try:
            last_owner = bool(
                resolve_object_refs_blocking(
                    runner.detach_client.remote(self._owner_id),
                    timeout=300,
                    honor_query_deadline=False,
                )
            )
            if last_owner:
                ray.kill(runner, no_restart=True)
        except BaseException as error:
            if (
                _runtime_error_contains(error, PermissionError)
                or self._runtime_is_unavailable_or_replaced()
                or _runtime_actor_is_being_replaced(error)
            ):
                self._complete_client_close_locally()
                return
            with self._session_condition:
                self._client_close_in_progress = False
                self._session_condition.notify_all()
            raise
        self._complete_client_close_locally()

    shutdown = close

    def get_test_udf_actor_handle(self, plan_id: str, udf_name: str) -> Any:
        """Environment-gated client wrapper for the Driver fault-test hook."""
        runner = getattr(self, "runner", None)
        if runner is None:
            raise RuntimeError("Ray query Driver is closed")
        return resolve_object_refs_blocking(
            runner.get_test_udf_actor_handle.remote(
                self._owner_id,
                str(plan_id),
                str(udf_name),
            ),
            timeout=30,
        )

    @staticmethod
    def _cancel_and_settle_remote_call(future: Any) -> BaseException | None:
        """Wait for an async actor mutation to stop before local ownership moves on."""
        try:
            ray.cancel(future, force=False)
        except BaseException:
            pass
        try:
            resolve_object_refs_blocking(
                future,
                timeout=300,
                honor_query_deadline=False,
            )
        except BaseException as error:
            return error
        return None

    def _client_detach_completed_for_runner(
        self,
        runner: Any,
        error: BaseException,
    ) -> bool:
        with self._session_condition:
            while self._client_close_in_progress:
                self._session_condition.wait()
            if getattr(self, "runner", None) is not runner:
                return True
            # detach_client removes the owner only after every owned session and
            # plan was closed. A retryable post-detach actor-kill failure leaves
            # the local handle in place, so PermissionError is still terminal
            # for this plan teardown.
            return self._client_closing and _runtime_error_contains(error, PermissionError)

    def _close_plan_after_stream(
        self,
        runner: Any,
        *,
        session_id: str,
        plan_id: str,
    ) -> None:
        try:
            resolve_object_refs_blocking(
                runner.close_plan.remote(
                    self._owner_id,
                    session_id,
                    plan_id,
                ),
                timeout=300,
                honor_query_deadline=False,
            )
        except BaseException as error:
            if (
                self._client_detach_completed_for_runner(runner, error)
                or self._runtime_is_unavailable_or_replaced()
                or _runtime_actor_is_being_replaced(error)
            ):
                return
            raise

    def _teardown_failed_plan(
        self,
        runner: Any,
        future: Any,
        *,
        session_id: str,
        plan_id: str,
        operation_error: BaseException,
    ) -> None:
        settled_error = self._cancel_and_settle_remote_call(future)
        try:
            self._close_plan_after_stream(
                runner,
                session_id=session_id,
                plan_id=plan_id,
            )
        except BaseException as teardown_error:
            cleanup_errors = [
                *(() if settled_error is None else (settled_error,)),
                teardown_error,
            ]
            aggregate_error = QueryExecutionCleanupError.from_errors(
                f"Ray query plan {plan_id} failed and teardown also failed",
                operation_error,
                cleanup_errors,
            )
            raise aggregate_error from operation_error

    def _recover_copy_plan(
        self,
        runner: Any,
        *,
        session_id: str,
        operation_id: str,
        operation_error: BaseException,
        reconciliation_timeout_s: float,
    ) -> CopyPlanOutcome:
        """Resolve an ambiguous COPY response with one bounded read-only wait."""
        try:
            recovery_future = runner.recover_copy_plan.remote(
                self._owner_id,
                session_id,
                operation_id,
            )
        except BaseException:
            _raise_copy_outcome_unknown(operation_id, operation_error)
        try:
            recovery = resolve_object_refs_blocking(
                recovery_future,
                timeout=reconciliation_timeout_s,
                honor_query_deadline=False,
                honor_object_get_timeout=False,
            )
        except BaseException as recovery_error:
            unknown_error = next(
                (
                    candidate
                    for candidate in _runtime_error_candidates(recovery_error)
                    if isinstance(candidate, CopyOutcomeUnknownError)
                ),
                None,
            )
            if unknown_error is not None:
                raise unknown_error
            _raise_copy_outcome_unknown(operation_id, operation_error)
        if not isinstance(recovery, CopyPlanRecovery):
            _raise_copy_outcome_unknown(operation_id, operation_error)
        if recovery.operation_id != operation_id:
            _raise_copy_outcome_unknown(operation_id, operation_error)
        if (recovery.outcome is None) == (recovery.error is None):
            _raise_copy_outcome_unknown(operation_id, operation_error)
        if recovery.error is not None:
            if not isinstance(recovery.error, BaseException):
                _raise_copy_outcome_unknown(operation_id, operation_error)
            restored_error = restore_remote_ray_exception(recovery.error)
            raise recovery.error if restored_error is None else restored_error
        try:
            return self._validate_copy_outcome(
                recovery.outcome,
                operation_id,
            )
        except BaseException:
            _raise_copy_outcome_unknown(operation_id, operation_error)

    @staticmethod
    def _validate_copy_outcome(
        outcome: Any,
        operation_id: str,
    ) -> CopyPlanOutcome:
        if not isinstance(outcome, CopyPlanOutcome):
            raise TypeError(f"Ray COPY returned {type(outcome).__name__}, expected CopyPlanOutcome")
        if outcome.operation_id != operation_id:
            raise RuntimeError(
                f"Ray COPY outcome identity mismatch: expected={operation_id} actual={outcome.operation_id}"
            )
        if outcome.write_state != "committed":
            raise RuntimeError(
                f"Ray COPY operation {operation_id} returned non-committed write state: {outcome.write_state}"
            )
        if outcome.cleanup_state not in {"complete", "pending"}:
            raise RuntimeError(
                f"Ray COPY operation {operation_id} returned invalid cleanup state: {outcome.cleanup_state}"
            )
        if not isinstance(outcome.result, dict):
            raise TypeError(f"Ray COPY operation {operation_id} returned a non-dict result")
        return outcome

    def retry_copy_cleanup(self, operation_id: str) -> dict[str, Any]:
        """Retry post-commit cleanup using the operation id returned by run_copy_plan()."""
        operation_key = str(operation_id).strip()
        if not operation_key:
            raise ValueError("COPY operation identity must not be empty")
        with self._session_condition:
            runner = getattr(self, "runner", None)
            if runner is None or self._client_closing:
                raise RuntimeError("Ray query runtime client is closed")
        outcome = resolve_object_refs_blocking(
            runner.retry_copy_cleanup.remote(
                self._owner_id,
                operation_key,
            ),
            honor_query_deadline=False,
        )
        validated = self._validate_copy_outcome(
            outcome,
            operation_key,
        )
        return validated.result

    def stream_plan(
        self,
        plan: Any,
    ) -> Iterator[RayMaterializedResult]:
        """Stream results from a distributed plan execution."""
        import time as _time

        from duckdb.runners.ray.partition_metadata import RayMaterializedResult

        _t_stream_start = _time.time()

        session_id, _, runner = self._ensure_session(plan)
        plan_id = str(plan.idx())
        run_future = runner.run_plan.remote(self._owner_id, session_id, plan)
        try:
            resolve_object_refs_blocking(run_future)
        except BaseException as operation_error:
            self._teardown_failed_plan(
                runner,
                run_future,
                session_id=session_id,
                plan_id=plan_id,
                operation_error=operation_error,
            )
            raise
        progress = _RayProgressSession(
            runner,
            self._owner_id,
            session_id,
            plan_id,
            _t_stream_start,
        )

        completed = False
        try:
            while True:
                partition_future = runner.get_next_partition.remote(
                    self._owner_id,
                    session_id,
                    plan_id,
                    runner,
                )
                materialized_result = progress.resolve(partition_future)
                if materialized_result is None:
                    completed = True
                    break
                if not isinstance(materialized_result, RayMaterializedResult):
                    continue
                yield materialized_result
        except BaseException as operation_error:
            cleanup_errors: list[BaseException] = []
            try:
                if not completed:
                    self._close_plan_after_stream(
                        runner,
                        session_id=session_id,
                        plan_id=plan_id,
                    )
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                progress.finish(final_state="FINISHED" if completed else None)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                aggregate_error = QueryExecutionCleanupError.from_errors(
                    f"Ray query plan {plan_id} failed and cleanup also failed",
                    operation_error,
                    cleanup_errors,
                )
                raise aggregate_error from operation_error
            raise
        else:
            progress.finish(final_state="FINISHED" if completed else None)

    def run_copy_plan(
        self,
        plan: Any,
    ) -> dict[str, Any]:
        """Execute a COPY/write plan and return aggregated file info."""
        import time as _time

        _t0 = _time.time()
        reconciliation_timeout_s = _copy_reconciliation_timeout_s()

        try:
            session_id, _, runner = self._ensure_session(plan)
            plan_id = str(plan.idx())
            future = runner.run_copy_plan.remote(
                self._owner_id,
                session_id,
                plan,
            )
            progress = _RayProgressSession(
                runner,
                self._owner_id,
                session_id,
                plan_id,
                _t0,
            )
            completed = False
            outcome = None
            final_progress_snapshot = None
            try:
                try:
                    outcome = progress.resolve(future)
                except BaseException as operation_error:
                    outcome = self._recover_copy_plan(
                        runner,
                        session_id=session_id,
                        operation_id=plan_id,
                        operation_error=operation_error,
                        reconciliation_timeout_s=reconciliation_timeout_s,
                    )
                else:
                    try:
                        outcome = self._validate_copy_outcome(
                            outcome,
                            plan_id,
                        )
                    except BaseException as response_error:
                        outcome = self._recover_copy_plan(
                            runner,
                            session_id=session_id,
                            operation_id=plan_id,
                            operation_error=response_error,
                            reconciliation_timeout_s=reconciliation_timeout_s,
                        )
                final_progress_snapshot = outcome.final_progress_snapshot
                completed = True
            finally:
                progress.finish(
                    final_state="FINISHED" if completed else None,
                    final_snapshot=final_progress_snapshot,
                )
            return outcome.result
        except Exception as e:
            unknown_outcome = next(
                (
                    candidate
                    for candidate in _runtime_error_candidates(e)
                    if isinstance(candidate, CopyOutcomeUnknownError)
                ),
                None,
            )
            if unknown_outcome is not None:
                raise unknown_outcome
            cleanup_failure = next(
                (
                    candidate
                    for candidate in _runtime_error_candidates(e)
                    if isinstance(candidate, QueryExecutionCleanupError)
                ),
                None,
            )
            if cleanup_failure is not None:
                primary_error = getattr(cleanup_failure, "primary_error", None)
                if isinstance(primary_error, BaseException):
                    raise cleanup_failure from primary_error
                raise cleanup_failure
            safe_message = _safe_remote_error_message(e)
            if hasattr(e, "traceback_str") or hasattr(e, "cause"):
                raise RuntimeError(safe_message) from None
            raise
