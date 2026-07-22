# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import math
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from vane._ray_cxx import require_ray_cxx_attr
from vane._ray_progress_env import configure_ray_progress_logging_defaults

RayResultPartitionRef = require_ray_cxx_attr(
    "RayResultPartitionRef",
    hint="Ensure the C++ ray extension is built and importable in the driver process.",
)
RayTaskResult = require_ray_cxx_attr(
    "RayTaskResult",
    hint="Ensure the C++ ray extension is built and importable in the driver process.",
)

configure_ray_progress_logging_defaults()

from vane.event_loop import set_event_loop
from vane.runners.fte import (
    FteTaskAttemptId,
    FteTaskState,
    fte_status_wait_timeout_s,
    validate_fte_status_identity,
)
from vane.runners.progress import ProgressRenderer, progress_enabled
from vane.runners.ray.admission_ledger import BoundedReplayMap
from vane.runners.ray.ray_env import collect_vane_env_overrides
from vane.runners.ray.safe_get import QueryDeadlineExceeded, resolve_object_refs_blocking
from vane.runners.ray.worker import WorkerTaskMetadata

_LEASE_REQUEST_REPLAY_CAPACITY = 65_536
_DEFAULT_PROGRESS_TOPOLOGY_INIT_TIMEOUT_S = 60.0

if TYPE_CHECKING:
    from collections.abc import Iterator

    import vane
    from vane.runners.ray.partition_metadata import RayMaterializedResult

import ray


class QueryTeardownOwnershipError(RuntimeError):
    """Teardown stopped before owner release and must remain retryable."""


class QueryFteRegistryQuiesceError(QueryTeardownOwnershipError):
    """Local FTE threads still own query state; teardown must remain retryable."""


@dataclass(frozen=True)
class CopyPlanOutcome:
    """Internal actor-to-client COPY result captured before query teardown."""

    result: dict[str, Any]
    final_progress_snapshot: dict[str, Any]


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


def _ray_progress_snapshot_or_none(runner: Any, plan_id: Any, started_at: float) -> dict[str, Any] | None:
    # Short hard timeout: progress must never stall query execution or pin the
    # driver actor long enough to block UDF lease admission/release.
    try:
        return resolve_object_refs_blocking(
            runner.progress_snapshot.remote(plan_id, started_at),
            timeout=0.1,
        )
    except Exception:
        return None


class _RayProgressSession:
    """Drive best-effort progress updates from synchronous Ray waits."""

    def __init__(self, runner: Any, plan_id: Any, started_at: float) -> None:
        self._renderer = None
        self._renderer_failed = False
        self._finished = False
        try:
            if progress_enabled():
                self._renderer = ProgressRenderer(lambda: _ray_progress_snapshot_or_none(runner, plan_id, started_at))
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


def _apply_env_overrides(env_overrides: dict[str, str] | None) -> None:
    if not env_overrides:
        return
    for key, value in env_overrides.items():
        if value is None:
            continue
        os.environ[key] = str(value)


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
    _result: RayTaskResult | None = None
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
            self.task_id.to_dict(),
            min_version,
            self.status_wait_timeout_s,
        )

    @staticmethod
    def _is_soft_status_wait_timeout(exc: BaseException) -> bool:
        if isinstance(exc, QueryDeadlineExceeded) or "query deadline expired" in str(exc).lower():
            return False
        if isinstance(exc, TimeoutError):
            return True
        name = exc.__class__.__name__
        if name in {"TimeoutError", "GetTimeoutError"}:
            return True
        message = str(exc)
        return "did not complete within" in message or "timed out" in message.lower()

    async def _watch_status(self) -> RayTaskResult:
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
                    if self._is_done:
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
        await asyncio.to_thread(
            self.worker_handle.mark_fte_worker_failed,
            self.worker_id,
            f"{failure_kind} for {self.task_id}: {exc}",
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
            cleanup_errors.append(f"worker failure publication failed: {cleanup_exc}")
        try:
            await asyncio.to_thread(self._record_fte_task_terminal)
        except Exception as cleanup_exc:
            cleanup_errors.append(f"terminal record failed: {cleanup_exc}")
        message = str(exc)
        if cleanup_errors:
            message += "; cleanup also failed: " + "; ".join(cleanup_errors)
        terminal_error = exc if not cleanup_errors and isinstance(exc, Exception) else RuntimeError(message)
        with self._lifecycle_lock:
            if self._is_done:
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

    def get_result_sync(self) -> RayTaskResult:
        with self._lifecycle_lock:
            if not self.done():
                raise RuntimeError("FTE task result not ready")
            if self._error is not None:
                raise self._error
            if self._result is None:
                raise RuntimeError("FTE task completed without result")
            self.ack()
            return self._result

    async def get_result(self) -> RayTaskResult:
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
                self.worker_handle.fte_cancel_task(self.task_id.to_dict())
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
            self.worker_handle.enqueue_fte_ack_task_result(self.task_id.to_dict())
            self._acked = True

    def release_result_payload(self) -> None:
        with self._lifecycle_lock:
            if self._released:
                return
            self.worker_handle.enqueue_fte_release_task_result(self.task_id.to_dict())
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
                    self.task_context_info.get("exchange_sink_instance"),
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
            failure = status.get("failure") or {}
            if isinstance(failure, dict):
                message = failure.get("message") or failure.get("type") or state.value
            else:
                message = str(failure)
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
            self.task_id.to_dict(),
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

    def _normalize_raw_result(self, raw_result: Any) -> RayTaskResult:
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
        if self.task_context_info.get("exchange_sink_instance") is not None:
            return True
        return bool(self.task_context_info.get("requires_spooling_output_stats"))

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


def batch_wait_ready(handles: list) -> list[int]:
    """Batch-check which worker task handle instances are ready.

    Called from C++ RayTaskResultPoller under a single GIL acquisition.  Each
    handle owns its terminal transition under its lifecycle lock, so the batch
    path delegates to that same state machine instead of mutating fields.

    Returns a list of indices into `handles` that are done.
    """
    return [index for index, handle in enumerate(handles) if handle.done()]


RAY_QUERY_DRIVER_ACTOR_NAMESPACE = "vane"
RAY_QUERY_DRIVER_ACTOR_NAME = "ray-query-driver-actor"


def _udf_test_hooks_enabled() -> bool:
    value = os.environ.get("VANE_ENABLE_UDF_TEST_HOOKS", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


@ray.remote(
    num_cpus=0,
)
class RayQueryDriverActor:
    def __init__(
        self,
        env_overrides: dict[str, str] | None,
        duckdb_memory_bytes: int,
    ) -> None:
        duckdb_memory_bytes = int(duckdb_memory_bytes)
        if duckdb_memory_bytes <= 0:
            raise ValueError("Ray driver duckdb_memory_bytes must be positive")
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
        self._query_resource_lock = threading.RLock()
        self._query_graphs: dict[str, Any] = {}
        self._query_allocations: dict[str, Any] = {}
        self._query_node_capacities: tuple[Any, ...] = ()
        self._query_task_lease_requests: dict[str, dict[str, Any]] = {}
        self._query_output_lease_requests: dict[str, dict[str, Any]] = {}
        self._query_task_lease_request_tombstones = BoundedReplayMap[str, dict[str, Any]](
            capacity=_LEASE_REQUEST_REPLAY_CAPACITY
        )
        self._query_output_lease_request_tombstones = BoundedReplayMap[str, dict[str, Any]](
            capacity=_LEASE_REQUEST_REPLAY_CAPACITY
        )
        self._query_task_request_owner_by_identity: dict[tuple[str, str, str], str] = {}
        self._query_output_request_owner_by_identity: dict[tuple[str, str], str] = {}
        self._query_resource_closing_queries: set[str] = set()
        self._query_task_admission_pumps: set[str] = set()
        self._query_output_admission_pumps: set[str] = set()
        self._query_fte_admission_pumps: dict[str, asyncio.Task[Any]] = {}
        self._query_fte_admission_dirty_queries: set[str] = set()
        self._query_fte_admission_done_events: dict[str, threading.Event] = {}
        self._query_resource_admission_signal_lock = threading.Lock()
        self._query_resource_admission_dirty_queries: set[str] = set()
        self._query_resource_admission_signal_scheduled = False
        self._query_resource_admission_loop: asyncio.AbstractEventLoop | None = None
        self._query_resource_admission_bridge_poisoned = False
        self._query_resource_maintenance_task: asyncio.Task[Any] | None = None
        self._query_resource_maintenance_stop: asyncio.Event | None = None
        self._query_resource_maintenance_error = ""
        self._query_resource_maintenance_failures = 0
        self._query_resource_last_maintenance_at = 0.0
        self._query_resource_last_capacity_refresh_at = 0.0
        self._progress_snapshot_lock = threading.Lock()
        self._progress_snapshot_builds: dict[tuple[str, float | None], asyncio.Task[Any]] = {}
        self._progress_snapshot_cache: dict[tuple[str, float | None], dict[str, Any]] = {}
        self._driver_duckdb_memory_bytes = duckdb_memory_bytes
        self._env_overrides = env_overrides or {}
        _apply_env_overrides(self._env_overrides)

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
        self._duckdb_conn = None
        self._ensure_duckdb_conn()

        # Eagerly create PlanRunner + warm up Worker workers so the
        # ~2.8s cold-start cost overlaps with GPU model loading.
        self._get_plan_runner()
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
        try:
            plan_runner = self._get_plan_runner()
            from vane.runners.ray.fte_fragment_scheduler import (
                fte_execution_query_ids_for_resource,
            )

            execution_query_ids = set(fte_execution_query_ids_for_resource(query_id))
            execution_query_ids.add(str(query_id))
        except BaseException as exc:
            errors.append(exc)
            execution_query_ids = {str(query_id)}
            plan_runner = None
        if plan_runner is not None:
            for execution_query_id in sorted(execution_query_ids):
                try:
                    plan_runner.drop_query_fragments(execution_query_id)
                except BaseException as exc:
                    errors.append(exc)
        from vane.runners.ray.fte_fragment_scheduler import (
            fte_query_remote_teardown_blockers,
        )

        teardown_blockers = fte_query_remote_teardown_blockers(query_id)
        if teardown_blockers:
            details = [
                *(f"{type(error).__name__}: {error}" for error in errors),
                "owners=" + ", ".join(teardown_blockers),
            ]
            raise QueryTeardownOwnershipError(
                f"distributed query teardown retains remote ownership for {query_id}: " + "; ".join(details)
            ) from (errors[0] if errors else None)
        try:
            # The remote fan-out is best effort, but local FTE ownership must
            # be fully quiesced before QRM/coordinator release. This second,
            # idempotent barrier also distinguishes dead-worker RPC failures
            # from a live local watcher that still depends on query state.
            from vane.runners.ray.fte_fragment_scheduler import (
                _drop_fte_registry_for_query,
            )

            _drop_fte_registry_for_query(query_id)
        except BaseException as exc:
            self._query_resource_admission_bridge_poisoned = True
            details = [*errors, exc]
            raise QueryFteRegistryQuiesceError(
                f"local FTE registry did not quiesce for {query_id}; "
                "driver admission bridge poisoned: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in details)
            ) from exc
        if release_resources:
            try:
                self._release_query_resources(
                    query_id,
                    reason="query_fragments_dropped",
                    admission_fenced=True,
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
        from vane.runners.progress import build_progress_snapshot
        from vane.runners.ray.fte_fragment_scheduler import (
            fte_progress_registry_snapshot,
        )
        from vane.runners.ray.query_resource_runtime import query_resource_manager_snapshot

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
        query_id: str,
        started_at: float | None = None,
    ) -> dict[str, Any]:
        """Build one local-only progress view without blocking actor admission."""
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

    def _ensure_duckdb_conn(self):
        """Eagerly create the DuckDB connection."""
        if self._duckdb_conn is not None:
            return self._duckdb_conn

        import vane
        from vane.runners.fte.memory_config import apply_duckdb_memory_limit
        from vane.runners.ray.worker import _configure_duckdb_s3

        self._duckdb_conn = vane.connect()
        _apply_duckdb_thread_setting(self._duckdb_conn)
        apply_duckdb_memory_limit(self._duckdb_conn, self._driver_duckdb_memory_bytes)
        _configure_duckdb_s3(self._duckdb_conn)
        return self._duckdb_conn

    def ping(self) -> bool:
        """Health-check: returns True if the actor is alive."""
        return True

    @staticmethod
    def _sum_node_capacity(node_capacities: tuple[Any, ...]) -> Any:
        from vane.runners.ray.query_execution_graph import ResourceVector

        total = ResourceVector()
        for node in node_capacities:
            total = total + node.resources
        return total

    def _create_query_resource_coordinator(self) -> Any:
        from vane.runners.ray.cluster_resource_coordinator import (
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

    def _synchronize_query_allocations(self) -> None:
        from vane.runners.ray.query_execution_graph import QueryAllocation
        from vane.runners.ray.query_resource_runtime import get_query_resource_manager

        snapshot = self._query_resource_coordinator.snapshot()
        coordinator_queries = snapshot["queries"]
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
                    try:
                        self._drop_query_fragments_after_admission_fence_sync(
                            query_id,
                            release_resources=False,
                        )
                    except BaseException as exc:
                        terminal_errors[query_id] = (
                            f"{reason}; fragment cancellation failed: {type(exc).__name__}: {exc}"
                        )

    def _refresh_query_capacity(self) -> Any:
        capacities = self._read_query_node_capacities()
        if not capacities:
            raise RuntimeError("Ray reports no alive nodes with schedulable query resources")
        self._query_resource_coordinator.update_node_capacities(capacities)
        self._query_node_capacities = capacities
        self._query_resource_last_capacity_refresh_at = time.monotonic()
        self._synchronize_query_allocations()
        return self._sum_node_capacity(capacities)

    @staticmethod
    def _read_query_node_capacities() -> tuple[Any, ...]:
        from vane.runners.ray.cluster_resource_coordinator import read_ray_node_capacities

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
        from vane.runners.ray.query_execution_graph import ResourceVector
        from vane.runners.ray.query_resource_runtime import get_query_resource_manager

        timestamp = time.monotonic() if now is None else float(now)
        capacity_error: BaseException | None = None
        if capacities is None:
            try:
                capacities = self._read_query_node_capacities()
            except BaseException as exc:
                capacity_error = exc
                capacities = tuple(self._query_node_capacities)
        capacities = tuple(capacities)
        if not capacities:
            if capacity_error is not None:
                raise RuntimeError(
                    f"failed to refresh Ray capacity and no cached snapshot exists: {capacity_error}"
                ) from capacity_error
            raise RuntimeError("Ray reports no alive nodes with schedulable query resources")

        with self._query_resource_lock:
            self._query_resource_coordinator.update_node_capacities(
                capacities,
                now=timestamp,
            )
            self._query_node_capacities = capacities
            if capacity_error is None:
                self._query_resource_last_capacity_refresh_at = timestamp
            self._synchronize_query_allocations()

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
            self._synchronize_query_allocations()
            self._query_resource_last_maintenance_at = timestamp
            return {
                "query_count": len(observed_usage),
                "capacity_cached": capacity_error is not None,
                "capacity_error": "" if capacity_error is None else str(capacity_error),
            }

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
            except TimeoutError:
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
        task_tombstones = getattr(self, "_query_task_lease_request_tombstones", ())
        if not isinstance(task_tombstones, BoundedReplayMap):
            self._query_task_lease_request_tombstones = BoundedReplayMap(
                getattr(task_tombstones, "items", lambda: ())(),
                capacity=_LEASE_REQUEST_REPLAY_CAPACITY,
            )
        output_tombstones = getattr(self, "_query_output_lease_request_tombstones", ())
        if not isinstance(output_tombstones, BoundedReplayMap):
            self._query_output_lease_request_tombstones = BoundedReplayMap(
                getattr(output_tombstones, "items", lambda: ())(),
                capacity=_LEASE_REQUEST_REPLAY_CAPACITY,
            )
        if not hasattr(self, "_query_task_request_owner_by_identity"):
            self._query_task_request_owner_by_identity = {}
        if not hasattr(self, "_query_output_request_owner_by_identity"):
            self._query_output_request_owner_by_identity = {}
        if not hasattr(self, "_query_resource_closing_queries"):
            self._query_resource_closing_queries = set()
        if not hasattr(self, "_query_task_admission_pumps"):
            self._query_task_admission_pumps = set()
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
        if not hasattr(self, "_query_resource_admission_bridge_poisoned"):
            self._query_resource_admission_bridge_poisoned = False

    def _run_on_query_resource_admission_loop_sync(
        self,
        callback: Callable[[], None],
        *,
        timeout_s: float = 30.0,
    ) -> None:
        """Run one admission-state mutation on its owner loop with a fence."""
        self._ensure_query_resource_admission_state()
        if self._query_resource_admission_bridge_poisoned:
            raise RuntimeError("query admission owner-loop fence is poisoned; restart the driver actor")
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

        completed = threading.Event()
        error: list[BaseException] = []
        invocation_lock = threading.Lock()
        invocation_state = "queued"

        def invoke() -> None:
            nonlocal invocation_state
            with invocation_lock:
                if invocation_state == "cancelled":
                    completed.set()
                    return
                invocation_state = "started"
            try:
                callback()
            except BaseException as exc:
                error.append(exc)
            finally:
                with invocation_lock:
                    invocation_state = "finished"
                completed.set()

        loop.call_soon_threadsafe(invoke)
        if not completed.wait(max(0.001, float(timeout_s))):
            with invocation_lock:
                if invocation_state == "queued":
                    invocation_state = "cancelled"
                    timeout_message = "timed out waiting for query admission owner-loop fence"
                elif invocation_state == "finished":
                    timeout_message = ""
                else:
                    # A started callback cannot be revoked safely. Poison the
                    # bridge so teardown/rollback cannot continue into a new
                    # query generation while a late mutation is possible.
                    self._query_resource_admission_bridge_poisoned = True
                    timeout_message = "query admission owner-loop callback started but did not finish; bridge poisoned"
            if timeout_message:
                raise RuntimeError(timeout_message)
        if error:
            raise error[0]

    def _close_query_resource_admission(self, query_id: str) -> None:
        """Fence new requests and resolve all old requests on the owner loop."""
        from vane.runners.ray.fte_fragment_scheduler import close_fte_registry_for_query
        from vane.runners.ray.query_resource_runtime import get_query_resource_manager

        self._ensure_query_resource_admission_state()
        query_key = str(query_id)
        close_fte_registry_for_query(query_key)
        try:
            get_query_resource_manager(query_key).close_admission()
        except KeyError:
            pass
        self._query_resource_closing_queries.add(query_key)
        self._fail_query_admission_requests(query_key)
        self._query_task_lease_requests = {
            request_id: state
            for request_id, state in self._query_task_lease_requests.items()
            if state.get("identity", {}).get("query_id") != query_key
        }
        self._query_output_lease_requests = {
            request_id: state
            for request_id, state in self._query_output_lease_requests.items()
            if state.get("identity", {}).get("query_id") != query_key
        }
        self._query_task_lease_request_tombstones.discard_where(
            lambda _request_id, state: state.get("identity", {}).get("query_id") == query_key
        )
        self._query_output_lease_request_tombstones.discard_where(
            lambda _request_id, state: state.get("identity", {}).get("query_id") == query_key
        )
        self._query_task_request_owner_by_identity = {
            identity: request_id
            for identity, request_id in self._query_task_request_owner_by_identity.items()
            if identity[0] != query_key
        }
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
        from vane.runners.ray.fte_fragment_scheduler import open_fte_registry_for_query

        self._ensure_query_resource_admission_state()
        query_key = str(query_id)
        old_state_exists = any(
            state.get("identity", {}).get("query_id") == query_key
            for table in (
                self._query_task_lease_requests,
                self._query_output_lease_requests,
                self._query_task_lease_request_tombstones,
                self._query_output_lease_request_tombstones,
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
        if old_state_exists or old_owner_exists:
            raise RuntimeError(f"cannot reopen query admission with old generation state: {query_key}")
        fte_pump = self._query_fte_admission_pumps.get(query_key)
        if fte_pump is not None and not fte_pump.done():
            raise RuntimeError(f"cannot reopen query admission while old FTE pump is active: {query_key}")
        self._query_fte_admission_pumps.pop(query_key, None)
        self._query_fte_admission_done_events.pop(query_key, None)
        self._query_fte_admission_dirty_queries.discard(query_key)
        open_fte_registry_for_query(query_key)
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
        from vane.runners.ray.fte_fragment_scheduler import (
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
        from vane.runners.ray.fte_fragment_scheduler import (
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
                from vane.runners.ray.query_resource_runtime import (
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
        tombstones[str(request_id)] = {
            "identity": dict(state["identity"]),
            "status": str(status),
            "result": terminal_result,
        }
        identity = state["identity"]
        if "block_id" in identity:
            owner_key = (str(identity["query_id"]), str(identity["block_id"]))
            owners = self._query_output_request_owner_by_identity
        else:
            owner_key = (
                str(identity["query_id"]),
                str(identity["task_id"]),
                str(identity["attempt_id"]),
            )
            owners = self._query_task_request_owner_by_identity
        if owners.get(owner_key) == str(request_id):
            owners.pop(owner_key, None)

    def _fail_query_admission_requests(self, query_id: str) -> None:
        """Resolve every pending request before its query manager disappears."""
        from vane.runners.ray.query_resource_manager import (
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
        from vane.runners.ray.query_resource_manager import TaskGrant
        from vane.runners.ray.query_resource_runtime import get_query_resource_manager

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
                state["status"] = "granted"
                state["lease"] = lease_payload
                _log_resource_debug(
                    "task_granted",
                    query_id=request.query_id,
                    stage_id=request.stage_id,
                    request_id=request_id,
                    lease_id=lease_payload["lease_id"],
                )
                self._complete_lease_request(
                    state,
                    self._grant_payload(grant),
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
        from vane.runners.ray.query_resource_manager import OutputBlockGrant
        from vane.runners.ray.query_resource_runtime import get_query_resource_manager

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

    async def acquire_query_task_lease(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Resolve one task lease through the query's serialized admission pump."""
        from vane.runners.ray.query_resource_manager import TaskGrant, TaskRequest
        from vane.runners.ray.query_resource_runtime import get_query_resource_manager

        values = self._strict_lease_request(
            payload,
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
        self._ensure_query_resource_admission_state()
        if str(request.query_id) in self._query_resource_closing_queries:
            return self._grant_payload(
                TaskGrant(
                    False,
                    blocked_reason="query_not_registered",
                    fatal=True,
                )
            )
        tombstone = self._query_task_lease_request_tombstones.get(request_id)
        if tombstone is not None and tombstone["identity"] != identity:
            raise ValueError(f"task lease request_id reused with different identity: {request_id}")
        if tombstone is not None:
            return dict(tombstone["result"])
        existing = self._query_task_lease_requests.get(request_id)
        if existing is not None and existing["identity"] != identity:
            raise ValueError(f"task lease request_id reused with different identity: {request_id}")
        if existing is not None:
            status = existing["status"]
            if status in {"granted", "submitted"}:
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
            str(identity["task_id"]),
            str(identity["attempt_id"]),
        )
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

    async def mark_query_task_lease_submitted(self, request_id: str, lease_id: str) -> dict[str, Any]:
        self._ensure_query_resource_admission_state()
        state = self._query_task_lease_requests.get(str(request_id))
        if state is None or state.get("status") not in {"granted", "submitted"}:
            return {"submitted": False}
        lease = state.get("lease") or {}
        if str(lease.get("lease_id") or "") != str(lease_id):
            return {"submitted": False}
        state["status"] = "submitted"
        return {"submitted": True}

    async def cancel_query_task_lease_request(self, request_id: str, *, submitted: bool) -> dict[str, Any]:
        from vane.runners.ray.query_resource_manager import TaskGrant
        from vane.runners.ray.query_resource_runtime import get_query_resource_manager

        self._ensure_query_resource_admission_state()
        request_key = str(request_id)
        state = self._query_task_lease_requests.get(request_key)
        if state is None:
            if request_key in self._query_task_lease_request_tombstones:
                return {"cancelled": True, "released": False}
            return {"cancelled": False, "released": False}
        denial = self._grant_payload(
            TaskGrant(
                False,
                blocked_reason="task_lease_request_cancelled",
                fatal=True,
            )
        )
        lease = state.get("lease")
        released = False
        if lease is not None:
            try:
                manager = get_query_resource_manager(lease["query_id"])
                if submitted:
                    released = manager.release_task_lease(lease["lease_id"], attempt_id=lease["attempt_id"])
                else:
                    released = manager.abandon_task_lease(lease["lease_id"], attempt_id=lease["attempt_id"])
            except KeyError:
                released = False
        identity = state["identity"]
        if lease is None:
            try:
                manager = get_query_resource_manager(identity["query_id"])
                manager.remove_task_waiter(
                    identity["task_id"],
                    identity["attempt_id"],
                )
                manager.mark_task_attempt_terminal(
                    identity["task_id"],
                    identity["attempt_id"],
                )
            except KeyError:
                pass
        else:
            try:
                get_query_resource_manager(identity["query_id"]).mark_task_attempt_terminal(
                    identity["task_id"],
                    identity["attempt_id"],
                )
            except KeyError:
                pass
        self._retire_query_lease_request(
            request_id=request_key,
            state=state,
            result=denial,
            status="cancelled",
            active=self._query_task_lease_requests,
            tombstones=self._query_task_lease_request_tombstones,
        )
        self._run_query_resource_admission_pumps(identity["query_id"])
        return {"cancelled": True, "released": bool(released)}

    async def release_query_task_lease(
        self,
        request_id: str,
        lease_id: str,
        attempt_id: str,
    ) -> dict[str, Any]:
        from vane.runners.ray.query_resource_manager import TaskGrant
        from vane.runners.ray.query_resource_runtime import get_query_resource_manager

        self._ensure_query_resource_admission_state()
        request_key = str(request_id)
        state = self._query_task_lease_requests.get(request_key)
        if state is None:
            return {"released": False}
        lease = state.get("lease") or {}
        if str(lease.get("lease_id") or "") != str(lease_id) or str(lease.get("attempt_id") or "") != str(attempt_id):
            return {"released": False}
        try:
            _log_resource_debug(
                "task_release_start",
                query_id=lease.get("query_id", ""),
                stage_id=lease.get("stage_id", ""),
                request_id=request_key,
                lease_id=lease_id,
            )
            released = get_query_resource_manager(lease["query_id"]).release_task_lease(
                str(lease_id), attempt_id=str(attempt_id)
            )
        except KeyError:
            released = False
        self._retire_query_lease_request(
            request_id=request_key,
            state=state,
            result=self._grant_payload(
                TaskGrant(
                    False,
                    blocked_reason="task_lease_request_released",
                    fatal=True,
                )
            ),
            status="released",
            active=self._query_task_lease_requests,
            tombstones=self._query_task_lease_request_tombstones,
        )
        _log_resource_debug(
            "task_release_done",
            query_id=lease.get("query_id", ""),
            stage_id=lease.get("stage_id", ""),
            request_id=request_key,
            lease_id=lease_id,
            released=bool(released),
        )
        self._run_query_resource_admission_pumps(lease["query_id"])
        return {"released": bool(released)}

    async def acquire_query_output_block_lease(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Resolve one produced block through the query's admission pump."""
        from vane.runners.ray.query_resource_manager import OutputBlockGrant, OutputBlockRequest
        from vane.runners.ray.query_resource_runtime import get_query_resource_manager

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
        request = OutputBlockRequest(**values)
        identity = asdict(request)
        self._ensure_query_resource_admission_state()
        if str(request.query_id) in self._query_resource_closing_queries:
            return self._grant_payload(
                OutputBlockGrant(
                    False,
                    blocked_reason="query_not_registered",
                    fatal=True,
                )
            )
        tombstone = self._query_output_lease_request_tombstones.get(request_id)
        if tombstone is not None and tombstone["identity"] != identity:
            raise ValueError(f"output lease request_id reused with different identity: {request_id}")
        if tombstone is not None:
            return dict(tombstone["result"])
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
    ) -> dict[str, Any]:
        """Transfer a live output from the producer queue to downstream input.

        This returns only the producer-side liveness credit.  The output lease
        itself remains active, so its bytes continue to count against query,
        stage, and node object-store limits until the final descriptor owner
        releases it.
        """
        from vane.runners.ray.query_resource_runtime import get_query_resource_manager

        self._ensure_query_resource_admission_state()
        request_key = str(request_id)
        state = self._query_output_lease_requests.get(request_key)
        if state is None or state.get("status") in {"released", "cancelled", "failed"}:
            return {"handed_off": False}
        lease = state.get("lease") or {}
        if str(lease.get("lease_id") or "") != str(lease_id):
            return {"handed_off": False}
        current_state = str(lease.get("state") or "")
        if current_state == "downstream_input":
            return {"handed_off": False}
        if current_state != "stage_queue":
            raise RuntimeError(f"output lease handoff requires stage_queue state, got {current_state!r}")
        try:
            handed_off = get_query_resource_manager(lease["query_id"]).transition_output_block(
                str(lease_id),
                "downstream_input",
            )
        except KeyError:
            handed_off = False
        if handed_off:
            lease["state"] = "downstream_input"
            self._run_query_resource_admission_pumps(lease["query_id"])
        return {"handed_off": bool(handed_off)}

    async def release_query_output_block_lease(
        self,
        request_id: str,
        lease_id: str,
    ) -> dict[str, Any]:
        from vane.runners.ray.query_resource_manager import OutputBlockGrant
        from vane.runners.ray.query_resource_runtime import get_query_resource_manager

        self._ensure_query_resource_admission_state()
        request_key = str(request_id)
        state = self._query_output_lease_requests.get(request_key)
        if state is None:
            return {"released": False}
        lease = state.get("lease") or {}
        if str(lease.get("lease_id") or "") != str(lease_id):
            return {"released": False}
        try:
            released = get_query_resource_manager(lease["query_id"]).release_output_block(str(lease_id))
        except KeyError:
            released = False
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
        return {"released": bool(released)}

    async def cancel_query_output_block_lease_request(self, request_id: str) -> dict[str, Any]:
        from vane.runners.ray.query_resource_manager import OutputBlockGrant
        from vane.runners.ray.query_resource_runtime import get_query_resource_manager

        self._ensure_query_resource_admission_state()
        request_key = str(request_id)
        state = self._query_output_lease_requests.get(request_key)
        if state is None:
            if request_key in self._query_output_lease_request_tombstones:
                return {"cancelled": True, "released": False}
            return {"cancelled": False, "released": False}
        denial = self._grant_payload(
            OutputBlockGrant(
                False,
                blocked_reason="output_lease_request_cancelled",
                fatal=True,
            )
        )
        lease = state.get("lease")
        released = False
        if lease is not None:
            try:
                released = get_query_resource_manager(lease["query_id"]).release_output_block(lease["lease_id"])
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

    def _register_query_resources(self, plan: Any) -> tuple[Any, Any]:
        from vane.runners.ray.query_graph_builder import (
            build_query_demand,
            build_query_execution_graph,
        )
        from vane.runners.ray.query_resource_runtime import (
            register_query_graph,
            release_query_resource_manager,
        )

        metadata = plan.collect_execution_stages(conn=self._duckdb_conn)
        graph = build_query_execution_graph(metadata)
        plan_id = str(plan.idx())
        if graph.query_id != plan_id:
            raise ValueError(f"execution graph query_id mismatch: graph={graph.query_id!r} plan={plan_id!r}")

        with self._query_resource_lock:
            if graph.query_id in self._query_graphs:
                raise ValueError(f"query graph is already registered: {graph.query_id}")
            cluster_capacity = self._refresh_query_capacity()
            demand = build_query_demand(
                graph,
                cluster_capacity,
            )
            allocation = self._query_resource_coordinator.register_query(demand)
            manager_registered = False
            try:
                self._synchronize_query_allocations()
                manager = register_query_graph(
                    graph,
                    allocation,
                    reservation_ratio=float(os.environ.get("VANE_QUERY_RESOURCE_RESERVATION_RATIO", "0.5")),
                    on_change=lambda query_id=graph.query_id: self._signal_query_resource_change(query_id),
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
                self._run_on_query_resource_admission_loop_sync(
                    lambda: self._open_query_resource_admission(graph.query_id)
                )
                return graph, allocation
            except BaseException as registration_error:
                cleanup_errors: list[BaseException] = []
                if manager_registered:
                    try:
                        release_query_resource_manager(
                            graph.query_id,
                            reason="query_registration_failed",
                        )
                    except BaseException as exc:
                        cleanup_errors.append(exc)
                self._query_graphs.pop(graph.query_id, None)
                self._query_allocations.pop(graph.query_id, None)
                try:
                    released = self._query_resource_coordinator.release_query(
                        graph.query_id,
                        allocation.generation,
                    )
                    if not released:
                        cleanup_errors.append(
                            RuntimeError("coordinator allocation disappeared during query registration rollback")
                        )
                except BaseException as exc:
                    cleanup_errors.append(exc)
                try:
                    self._synchronize_query_allocations()
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
        from vane.runners.ray.query_resource_runtime import get_query_resource_manager

        manager = get_query_resource_manager(graph.query_id)
        for stage in graph.stages:
            if stage.backend == "ray_actor":
                manager.set_stage_actor_ready(stage.stage_id, True)

    def _fence_query_resource_admission_for_teardown(self, query_id: str) -> None:
        from vane.runners.ray.fte_fragment_scheduler import quiesce_fte_registry_for_query

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
        self._run_on_query_resource_admission_loop_sync(lambda: self._close_query_resource_admission(query_key))
        # A resource-change callback may already be draining the FTE scheduler
        # in a worker thread. Keep every fragment registry alive until that
        # drain exits; the close fence prevents another drain from starting.
        self._wait_for_query_fte_admission_pump(query_key)
        self._run_on_query_resource_admission_loop_sync(
            lambda: self._purge_completed_query_fte_admission_pump(query_key)
        )
        quiesce_fte_registry_for_query(query_key)

    def _release_query_resources(
        self,
        query_id: str,
        *,
        reason: str,
        admission_fenced: bool = False,
    ) -> None:
        from vane.runners.ray.query_resource_runtime import release_query_resource_manager

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
        cleanup_errors: list[BaseException] = []
        with self._query_resource_lock:
            allocation = self._query_allocations.get(query_key)
            try:
                release_query_resource_manager(query_key, reason=reason)
            except BaseException as exc:
                cleanup_errors.append(exc)
            if allocation is not None:
                try:
                    released = self._query_resource_coordinator.release_query(
                        query_key,
                        allocation.generation,
                    )
                    if not released:
                        cleanup_errors.append(
                            RuntimeError(
                                f"coordinator allocation was already absent for query {query_key} "
                                f"at generation {allocation.generation}"
                            )
                        )
                except BaseException as exc:
                    cleanup_errors.append(exc)
            self._query_graphs.pop(query_key, None)
            self._query_allocations.pop(query_key, None)
            try:
                self._synchronize_query_allocations()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            raise RuntimeError(
                f"query resource cleanup for {query_key} had "
                f"{len(cleanup_errors)} error(s): "
                + "; ".join(f"{type(exc).__name__}: {exc}" for exc in cleanup_errors)
            ) from cleanup_errors[0]

    def query_resource_snapshot(self, query_id: str) -> dict[str, Any]:
        from vane.runners.ray.query_resource_runtime import query_resource_manager_snapshot

        query_key = str(query_id or "").strip()
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

    def get_test_udf_actor_handle(self, plan_id: str, udf_name: str) -> Any:
        """Return one query-owned stateful actor for deterministic fault tests.

        This hook is deliberately environment-gated and lives on the Driver;
        it is not part of Vane's public UDF API.
        """
        if not _udf_test_hooks_enabled():
            raise RuntimeError(
                "UDF actor test hooks are disabled; set VANE_ENABLE_UDF_TEST_HOOKS=1 before starting the Driver"
            )

        plan_key = str(plan_id)
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

    def install_env_overrides(self, env_overrides: dict[str, str] | None) -> None:
        self._env_overrides = env_overrides or {}
        _apply_env_overrides(self._env_overrides)
        if self._duckdb_conn is not None:
            _apply_duckdb_thread_setting(self._duckdb_conn)

    def _get_plan_runner(self) -> Any:
        if self.plan_runner is None:
            DistributedPhysicalPlanRunner = require_ray_cxx_attr(
                "DistributedPhysicalPlanRunner",
                hint="Ensure the C++ ray extension is built and importable in this process.",
            )
            self.plan_runner = DistributedPhysicalPlanRunner()
            self.plan_runner.warm_up()
        return self.plan_runner

    def _precreate_udf_actors(self, plan: Any, graph: Any, allocation: Any) -> list:
        """Create Ray actors and inject handles without waiting for model init."""
        from vane.execution.udf_ray import prepare_actor_pools_for_plan

        actor_node_ids_by_stage = {
            stage.stage_id: allocation.actor_node_ids_for_stage(stage.stage_id)
            for stage in graph.stages
            if stage.backend == "ray_actor"
        }
        created, _ = prepare_actor_pools_for_plan(
            plan,
            actor_node_ids_by_stage=actor_node_ids_by_stage,
            conn=self._duckdb_conn,
        )
        self._active_udf_actors.extend(created)
        return created

    @staticmethod
    def _wait_for_udf_actors_ready(actor_pools: list[Any]) -> None:
        from vane.execution.udf_ray import wait_for_actor_pools_ready

        wait_for_actor_pools_ready(actor_pools)

    def _precreate_vllm_actors(self, plan: Any) -> list:
        """Pre-create Ray actor pools for vLLM nodes on the driver."""
        from vane.execution.vllm import ensure_named_vllm_pools_for_plan

        created, _ = ensure_named_vllm_pools_for_plan(plan, conn=self._duckdb_conn)
        self._active_vllm_actors.extend(created)
        return created

    def _cleanup_udf_actor_pools(self, plan_id: str) -> None:
        pools = self._active_udf_actors_by_plan.pop(str(plan_id), [])
        if not pools:
            return
        errors: list[str] = []
        for pool in pools:
            try:
                pool.shutdown()
            except BaseException as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
            try:
                self._active_udf_actors.remove(pool)
            except ValueError:
                pass
        if errors:
            raise RuntimeError(f"failed to shut down {len(errors)} query-owned UDF actor pool(s): " + "; ".join(errors))

    def _teardown_plan_resources(
        self,
        plan_id: str,
        query_id: str,
        *,
        drop_fragments: bool,
    ) -> None:
        errors: list[BaseException] = []
        retain_query_owner = False
        self.curr_plans.pop(plan_id, None)
        self.curr_streams.pop(plan_id, None)
        leased_refs = getattr(self, "_leased_result_partition_refs", None)
        if leased_refs is not None:
            records = leased_refs.pop(str(plan_id), {})
            for _, output_lease_owner in records.values():
                try:
                    output_lease_owner.release()
                except BaseException as exc:
                    errors.append(exc)
        counters = getattr(self, "_result_partition_ref_counters", None)
        if counters is not None:
            counters.pop(str(plan_id), None)
        try:
            self._cleanup_udf_actor_pools(str(plan_id))
        except BaseException as exc:
            errors.append(exc)
        query_key = str(query_id or "").strip()
        if query_key:
            try:
                if drop_fragments:
                    self._drop_query_fragments_sync(query_key)
                else:
                    self._release_query_resources(
                        query_key,
                        reason="query_ended_before_fragment_start",
                    )
            except BaseException as exc:
                errors.append(exc)
                retain_query_owner = isinstance(
                    exc,
                    QueryTeardownOwnershipError,
                )
        if errors:
            if not retain_query_owner:
                self._plan_query_ids.pop(plan_id, None)
                terminal_errors = getattr(self, "_query_terminal_errors", None)
                if terminal_errors is not None:
                    terminal_errors.pop(str(query_id or ""), None)
            raise RuntimeError(
                f"query plan {plan_id} teardown failed: " + "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
            ) from errors[0]
        self._plan_query_ids.pop(plan_id, None)
        terminal_errors = getattr(self, "_query_terminal_errors", None)
        if terminal_errors is not None:
            terminal_errors.pop(str(query_id or ""), None)

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
        try:
            self._teardown_plan_resources(
                plan_id,
                query_id,
                drop_fragments=True,
            )
        except BaseException as teardown_error:
            raise RuntimeError(
                f"{reason}; deterministic teardown failed: {type(teardown_error).__name__}: {teardown_error}"
            ) from teardown_error
        raise RuntimeError(reason)

    def _lease_result_partition_ref(
        self,
        plan_id: str,
        object_ref: Any,
        output_lease_owner: Any,
    ) -> str:
        if not hasattr(self, "_leased_result_partition_refs"):
            self._leased_result_partition_refs = {}
        if not hasattr(self, "_result_partition_ref_counters"):
            self._result_partition_ref_counters = {}
        plan_key = str(plan_id)
        next_id = int(self._result_partition_ref_counters.get(plan_key, 0))
        self._result_partition_ref_counters[plan_key] = next_id + 1
        release_token = str(next_id)
        self._leased_result_partition_refs.setdefault(plan_key, {})[release_token] = (
            object_ref,
            output_lease_owner,
        )
        return release_token

    def release_result_partition_ref(self, plan_id: str, release_token: str) -> None:
        if not hasattr(self, "_leased_result_partition_refs"):
            self._leased_result_partition_refs = {}
        refs = self._leased_result_partition_refs.get(str(plan_id))
        if not refs:
            return
        record = refs.pop(str(release_token), None)
        if record is not None:
            _, output_lease_owner = record
            output_lease_owner.release()
        if not refs:
            self._leased_result_partition_refs.pop(str(plan_id), None)

    async def close_plan(self, plan_id: str) -> None:
        await asyncio.to_thread(self._cleanup_finished_plan, plan_id)

    async def run_plan(
        self,
        plan: vane.ray_cxx.PyLogicalPlan,
    ) -> None:
        """Run a plan without blocking the driver actor's control event loop."""
        _set_global_event_loop(asyncio.get_running_loop())
        await asyncio.to_thread(self._run_plan_sync, plan)

    def _run_plan_sync(
        self,
        plan: vane.ray_cxx.PyLogicalPlan,
    ) -> None:
        """Blocking plan startup executed by ``run_plan`` on an owned worker thread."""
        _apply_env_overrides(self._env_overrides)

        if os.environ.get("VANE_WORKER") == "1":
            raise RuntimeError(
                "RayQueryDriverActor.run_plan() called inside a Worker Worker. "
                "Nested distributed execution is not supported."
            )

        self._ensure_duckdb_conn()

        plan = plan.to_physical_plan(self._duckdb_conn)
        graph = None
        plan_id = str(plan.idx())
        plan_runner_started = False
        try:
            graph, allocation = self._register_query_resources(plan)
            udf_actors = self._precreate_udf_actors(plan, graph, allocation)
            if udf_actors:
                self._active_udf_actors_by_plan[plan_id] = list(udf_actors)
            self._precreate_vllm_actors(plan)

            self.curr_plans[plan_id] = plan
            self._plan_query_ids[plan_id] = graph.query_id
            plan_runner = self._get_plan_runner()
            plan_runner_started = True
            self.curr_streams[plan_id] = plan_runner.run_plan(plan, self._duckdb_conn)
            # Native FTE execution is intentionally started before the actor
            # readiness fence opens.  It can initialize DuckDB's real
            # Fragment/Pipeline topology, but its Ray-actor UDF dispatcher
            # cannot obtain a QRM lease until model initialization succeeds.
            self._wait_for_udf_actors_ready(udf_actors)
            self._mark_query_actor_stages_ready(graph)
        except BaseException as start_error:
            query_id = "" if graph is None else str(graph.query_id)
            try:
                self._teardown_plan_resources(
                    plan_id,
                    query_id,
                    drop_fragments=plan_runner_started,
                )
            except BaseException as teardown_error:
                raise RuntimeError(
                    f"query plan {plan_id} failed to start and teardown also failed: "
                    f"start={type(start_error).__name__}: {start_error}; "
                    f"teardown={type(teardown_error).__name__}: {teardown_error}"
                ) from start_error
            raise

    async def run_copy_plan(
        self,
        plan: Any,
    ) -> CopyPlanOutcome:
        """Run COPY and capture its final progress state before teardown."""
        _apply_env_overrides(self._env_overrides)

        self._ensure_duckdb_conn()

        plan = plan.to_physical_plan(self._duckdb_conn)
        plan_id = str(plan.idx())
        graph = None
        plan_runner_started = False
        plan_execution: asyncio.Task[Any] | None = None
        startup_tasks: list[asyncio.Task[Any]] = []
        try:
            graph, allocation = self._register_query_resources(plan)
            udf_actors = await asyncio.to_thread(
                self._precreate_udf_actors,
                plan,
                graph,
                allocation,
            )
            if udf_actors:
                self._active_udf_actors_by_plan[plan_id] = list(udf_actors)
            await asyncio.to_thread(self._precreate_vllm_actors, plan)
            plan_runner = self._get_plan_runner()
            plan_runner_started = True
            plan_execution = asyncio.create_task(
                asyncio.to_thread(
                    plan_runner.run_copy_plan,
                    plan,
                    self._duckdb_conn,
                ),
                name=f"vane-copy-plan:{plan_id}",
            )
            if udf_actors:
                from vane.runners.ray.fte_fragment_scheduler import (
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
                        await plan_execution
                        plan_execution_observed = True
                    for task in done:
                        if task is plan_execution:
                            continue
                        await task
                        pending_startup.discard(task)
            self._mark_query_actor_stages_ready(graph)
            result = await plan_execution
        except BaseException as execution_error:
            query_id = "" if graph is None else str(graph.query_id)
            teardown_error: BaseException | None = None
            try:
                await asyncio.to_thread(
                    self._teardown_plan_resources,
                    plan_id,
                    query_id,
                    drop_fragments=plan_runner_started,
                )
            except BaseException as error:
                teardown_error = error
            # Teardown interrupts the native runner. Always retrieve its
            # terminal result so a concurrent failure cannot escape as an
            # unobserved asyncio task exception.
            if plan_execution is not None:
                try:
                    await plan_execution
                except BaseException:
                    pass
            if startup_tasks:
                await asyncio.gather(*startup_tasks, return_exceptions=True)
            if teardown_error is not None:
                raise RuntimeError(
                    f"COPY query plan {plan_id} failed and teardown also failed: "
                    f"execution={type(execution_error).__name__}: {execution_error}; "
                    f"teardown={type(teardown_error).__name__}: {teardown_error}"
                ) from execution_error
            raise
        query_id = str(graph.query_id)
        _log_copy_result_debug(query_id, result)
        if query_id in getattr(self, "_query_terminal_errors", {}):
            await asyncio.to_thread(
                self._finish_terminal_query,
                plan_id,
                query_id,
            )
        try:
            final_progress_snapshot = await asyncio.to_thread(
                self._build_local_progress_snapshot,
                query_id,
                None,
            )
        except BaseException as progress_error:
            try:
                await asyncio.to_thread(
                    self._teardown_plan_resources,
                    plan_id,
                    query_id,
                    drop_fragments=True,
                )
            except BaseException as teardown_error:
                raise RuntimeError(
                    f"COPY query plan {plan_id} progress finalization failed and teardown also failed: "
                    f"progress={type(progress_error).__name__}: {progress_error}; "
                    f"teardown={type(teardown_error).__name__}: {teardown_error}"
                ) from progress_error
            raise
        await asyncio.to_thread(
            self._teardown_plan_resources,
            plan_id,
            query_id,
            drop_fragments=True,
        )
        return CopyPlanOutcome(
            result=result,
            final_progress_snapshot=final_progress_snapshot,
        )

    async def get_next_partition(
        self,
        plan_id: str,
        release_owner: Any | None = None,
    ) -> RayMaterializedResult | None:
        from vane.runners.ray.partition_metadata import (
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
            await asyncio.to_thread(
                self._finish_terminal_query,
                str(plan_id),
                query_id,
            )
        from vane.runners.ray.query_resource_runtime import get_query_resource_manager

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

                def _safe_blocking_next():
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
                await asyncio.to_thread(self._cleanup_finished_plan, plan_id)
                return None
            except RuntimeError as e:
                # pybind11 wraps StopIteration in RuntimeError
                if "StopIteration" in str(e):
                    await asyncio.to_thread(self._cleanup_finished_plan, plan_id)
                    return None
                raise
        finally:
            manager.set_external_consumer_waiting(False)
        if query_id in getattr(self, "_query_terminal_errors", {}):
            await asyncio.to_thread(
                self._finish_terminal_query,
                str(plan_id),
                query_id,
            )
        if next_item is None:
            await asyncio.to_thread(self._cleanup_finished_plan, plan_id)
            return None

        if isinstance(next_item, RayMaterializedResult):
            return next_item
        if isinstance(next_item, WorkerTaskMetadata):
            await asyncio.to_thread(self._cleanup_finished_plan, plan_id)
            return None

        if hasattr(next_item, "object_ref"):
            object_ref = next_item.object_ref
            output_lease_owner = next_item.lease_owner
            if not callable(getattr(output_lease_owner, "transition_to", None)):
                raise TypeError("metadata-aware Ray result is missing its output lease owner")
            output_lease_owner.transition_to("external_consumer")
            release_token = self._lease_result_partition_ref(
                str(plan_id),
                object_ref,
                output_lease_owner,
            )
            metadata_accessor = PartitionMetadataAccessor.from_metadata_list(
                [PartitionMetadata(next_item.num_rows, next_item.size_bytes)]
            )
            return RayMaterializedResult(
                partition=object_ref,
                metadatas=metadata_accessor,
                metadata_idx=0,
                release_owner=release_owner,
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

    def _usable(node: dict) -> bool:
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
    """Client wrapper for the Ray query driver actor."""

    def __init__(self) -> None:
        try:
            self._ray_gcs_address = ray.get_runtime_context().gcs_address
        except Exception:
            self._ray_gcs_address = None
        _maybe_set_distributed_cluster_capacity()
        from vane.runners.ray.worker_memory import build_ray_node_memory_layout

        head_node = get_head_node()
        if head_node is None:
            raise RuntimeError("Ray cluster has no alive internal head node")
        head_node_id = str(head_node.get("NodeID") or "").strip()
        head_memory_bytes = int(float((head_node.get("Resources") or {}).get("memory", 0) or 0))
        if not head_node_id or head_memory_bytes <= 0:
            raise RuntimeError("Ray head node has no positive logical memory capacity")
        head_memory_layout = build_ray_node_memory_layout(head_memory_bytes)
        driver_duckdb_memory_bytes = head_memory_layout.driver_duckdb_reserve_bytes
        env_overrides = _collect_vane_env_overrides()
        scheduling = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        )
        runner_options = {
            "name": RAY_QUERY_DRIVER_ACTOR_NAME,
            "namespace": RAY_QUERY_DRIVER_ACTOR_NAMESPACE,
            "scheduling_strategy": scheduling,
            "memory": driver_duckdb_memory_bytes,
            # RayQueryDriverActor must receive PYTHONPATH/env overrides at actor
            # creation time; constructor args are too late for module import.
            "runtime_env": {"env_vars": env_overrides},
        }
        self.runner = RayQueryDriverActor.options(  # type: ignore[attr-defined]
            get_if_exists=True,
            **runner_options,
        ).remote(env_overrides, driver_duckdb_memory_bytes)

        # Health-check: verify the actor is alive.  If a stale actor from a
        # previous session is returned by get_if_exists, the ping will fail
        # and we recreate the actor from scratch.
        try:
            resolve_object_refs_blocking(self.runner.ping.remote(), timeout=300)
        except Exception:
            try:
                ray.kill(self.runner)
            except Exception:
                pass
            self.runner = RayQueryDriverActor.options(  # type: ignore[attr-defined]
                get_if_exists=False,
                **runner_options,
            ).remote(env_overrides, driver_duckdb_memory_bytes)
            # Verify the fresh actor is alive
            resolve_object_refs_blocking(self.runner.ping.remote(), timeout=300)

        resolve_object_refs_blocking(self.runner.install_env_overrides.remote(env_overrides))

    def close(self) -> None:
        runner = getattr(self, "runner", None)
        if runner is None:
            return
        self.runner = None
        if not ray.is_initialized():
            return
        try:
            current_gcs_address = ray.get_runtime_context().gcs_address
        except Exception:
            current_gcs_address = None
        if (
            self._ray_gcs_address is not None
            and current_gcs_address is not None
            and current_gcs_address != self._ray_gcs_address
        ):
            return
        try:
            ray.kill(runner, no_restart=True)
        except TypeError:
            try:
                ray.kill(runner)
            except Exception:
                pass
        except Exception:
            pass

    shutdown = close

    def get_test_udf_actor_handle(self, plan_id: str, udf_name: str) -> Any:
        """Environment-gated client wrapper for the Driver fault-test hook."""
        runner = getattr(self, "runner", None)
        if runner is None:
            raise RuntimeError("Ray query Driver is closed")
        return resolve_object_refs_blocking(
            runner.get_test_udf_actor_handle.remote(str(plan_id), str(udf_name)),
            timeout=30,
        )

    def stream_plan(
        self,
        plan: Any,
    ) -> Iterator[RayMaterializedResult]:
        """Stream results from a distributed plan execution."""
        import time as _time

        from vane.runners.ray.partition_metadata import RayMaterializedResult

        _t_stream_start = _time.time()

        env_overrides = _collect_vane_env_overrides()
        resolve_object_refs_blocking(self.runner.install_env_overrides.remote(env_overrides))
        plan_id = plan.idx()
        resolve_object_refs_blocking(self.runner.run_plan.remote(plan))
        progress = _RayProgressSession(self.runner, plan_id, _t_stream_start)

        completed = False
        try:
            while True:
                partition_future = self.runner.get_next_partition.remote(plan_id, self.runner)
                materialized_result = progress.resolve(partition_future)
                if materialized_result is None:
                    completed = True
                    break
                if not isinstance(materialized_result, RayMaterializedResult):
                    continue
                yield materialized_result
        finally:
            try:
                if not completed:
                    resolve_object_refs_blocking(self.runner.close_plan.remote(plan_id), timeout=30)
            finally:
                progress.finish(final_state="FINISHED" if completed else None)

    def run_copy_plan(
        self,
        plan: Any,
    ) -> dict[str, Any]:
        """Execute a COPY/write plan and return aggregated file info."""
        import time as _time

        _t0 = _time.time()

        env_overrides = _collect_vane_env_overrides()
        resolve_object_refs_blocking(self.runner.install_env_overrides.remote(env_overrides))

        try:
            plan_id = str(plan.idx())
            future = self.runner.run_copy_plan.remote(plan)
            progress = _RayProgressSession(self.runner, plan_id, _t0)
            completed = False
            outcome = None
            final_progress_snapshot = None
            try:
                outcome = progress.resolve(future)
                if not isinstance(outcome, CopyPlanOutcome):
                    raise TypeError(f"Ray COPY returned {type(outcome).__name__}, expected CopyPlanOutcome")
                final_progress_snapshot = outcome.final_progress_snapshot
                completed = True
            finally:
                progress.finish(
                    final_state="FINISHED" if completed else None,
                    final_snapshot=final_progress_snapshot,
                )
            return outcome.result
        except Exception as e:
            safe_message = _safe_remote_error_message(e)
            if hasattr(e, "traceback_str") or hasattr(e, "cause"):
                raise RuntimeError(safe_message) from None
            raise
