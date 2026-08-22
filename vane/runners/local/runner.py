# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from numbers import Integral
from typing import TYPE_CHECKING, Any

from vane._ray_cxx import require_ray_cxx_attr
from vane._vane_session import ensure_vane_session_dir
from vane.runners.copy_outcome import CopyOutcomeUnknownError
from vane.runners.fte import FteTaskAttemptId
from vane.runners.fte.backends.native import NativeFteWorkerManagerBackend
from vane.runners.fte.memory_config import apply_duckdb_memory_limit
from vane.runners.progress import ProgressRenderer, build_progress_snapshot, progress_enabled
from vane.runners.runner import Runner

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]


_ARROW_DATASET_PRELOAD_LOCK = threading.Lock()
_ARROW_DATASET_PRELOADED: bool = False


def _arrow_dataset_is_preloaded() -> bool:
    return _ARROW_DATASET_PRELOADED


def _preload_arrow_dataset_imports() -> None:
    global _ARROW_DATASET_PRELOADED
    if _arrow_dataset_is_preloaded():
        return
    with _ARROW_DATASET_PRELOAD_LOCK:
        if _arrow_dataset_is_preloaded():
            return
        # DuckDB may lazily import pyarrow.dataset while native worker threads
        # are submitting local-shm ref bundles. Do the import once on the caller
        # thread so pyarrow/pandas import locks are not first hit inside execution.
        import pyarrow.dataset  # type: ignore[import-not-found, import-untyped, unused-ignore]  # noqa: F401

        _ARROW_DATASET_PRELOADED = True


def _normalize_num_workers(num_workers: Any) -> int:
    if num_workers is None:
        return 1
    if isinstance(num_workers, bool) or not isinstance(num_workers, Integral):
        raise ValueError("num_workers must be a positive integer")
    workers = int(num_workers)
    if workers <= 0:
        raise ValueError("num_workers must be a positive integer")
    return workers


def _normalize_execution_mode(execution_mode: str | None) -> str:
    mode = str(execution_mode or "in_process").strip().lower().replace("-", "_")
    if mode != "in_process":
        raise ValueError("local currently supports execution_mode='in_process'")
    return mode


def _normalize_max_running_tasks(max_running_tasks: Any) -> int | None:
    if max_running_tasks is None:
        return None
    if isinstance(max_running_tasks, bool) or not isinstance(max_running_tasks, Integral):
        raise ValueError("max_running_tasks must be a positive integer or None")
    value = int(max_running_tasks)
    if value <= 0:
        raise ValueError("max_running_tasks must be a positive integer or None")
    return value


def _copy_output_info_from_context(context: dict[str, Any] | None) -> dict[str, str] | None:
    if not context:
        return None
    base = context.get("copy_output_base")
    run_id = context.get("copy_output_run_id")
    remote_base = context.get("copy_output_remote_base")
    if base is None and run_id is None and remote_base is None:
        return None
    return {
        "base": str(base or ""),
        "run_id": str(run_id or ""),
        "remote_base": str(remote_base or ""),
    }


def _require_known_copy_outcome(operation_id: str, result: dict[str, Any]) -> dict[str, Any]:
    if result.get("copy_output_outcome_unknown") is True:
        raise CopyOutcomeUnknownError.from_native_result(operation_id, result)
    return result


def _record_unknown_copy_cleanup_errors(
    primary_error: BaseException | None,
    stage: str,
    cleanup_errors: list[BaseException],
) -> bool:
    if not isinstance(primary_error, CopyOutcomeUnknownError) or not cleanup_errors:
        return False
    warnings: list[str] = []
    for error in cleanup_errors:
        try:
            message = str(error)
        except BaseException:
            message = "<error message unavailable>"
        warnings.append(f"{stage} failed: {type(error).__name__}: {message}")
    primary_error.add_cleanup_warnings(*warnings)
    return True


def _shutdown_udf_actor_pools(actor_pools: list[Any], *, kill: bool) -> list[BaseException]:
    errors: list[BaseException] = []
    for pool in reversed(actor_pools):
        try:
            pool.shutdown(kill=kill)
        except BaseException as exc:
            errors.append(exc)
    return errors


def _shutdown_local_write_resources(
    backend: Any,
    fragment_executor: Any,
    conn: Any,
    actor_pools: list[Any],
    *,
    kill_actor_pools: bool,
    timeout_s: float,
) -> list[BaseException]:
    """Stop execution before releasing any resource a fragment may still use."""
    timeout_s = float(timeout_s)
    if not math.isfinite(timeout_s) or timeout_s < 0:
        raise ValueError("local write resource shutdown timeout must be finite and non-negative")
    deadline = time.monotonic() + timeout_s
    errors: list[BaseException] = []
    try:
        backend.request_shutdown()
    except BaseException as exc:
        errors.append(exc)

    try:
        fragment_executor.request_shutdown()
    except BaseException as exc:
        errors.append(exc)

    try:
        backend.shutdown(timeout_s=max(0.0, deadline - time.monotonic()))
    except BaseException as exc:
        errors.append(exc)
        return errors

    try:
        fragment_executor.close(timeout_s=max(0.0, deadline - time.monotonic()))
    except BaseException as exc:
        errors.append(exc)
        return errors

    errors.extend(_shutdown_udf_actor_pools(actor_pools, kill=kill_actor_pools))
    try:
        conn.close()
    except BaseException as exc:
        errors.append(exc)
    return errors


def _native_task_maps_from_context(context: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    scan_split_batch_map: dict[str, Any] = {}
    exchange_source_task_map: dict[str, Any] = {}
    for key, value in (context or {}).items():
        if key.startswith("scan_split_batch:"):
            node_id = key.split(":", 1)[1]
            if node_id:
                scan_split_batch_map[node_id] = value
        elif key.startswith("exchange_source_task:"):
            node_id = key.split(":", 1)[1]
            if node_id:
                exchange_source_task_map[node_id] = value
    return scan_split_batch_map, exchange_source_task_map


class _InProcessFragmentExecutor:
    def __init__(self, *, close_timeout_s: float = 30.0) -> None:
        close_timeout_s = float(close_timeout_s)
        if not math.isfinite(close_timeout_s) or close_timeout_s <= 0:
            raise ValueError("local fragment executor close timeout must be finite and positive")
        self._close_timeout_s = close_timeout_s
        self._local = threading.local()
        self._resources_lock = threading.RLock()
        self._resources_condition = threading.Condition(self._resources_lock)
        self._plan_clone_lock = threading.Lock()
        self._connections: list[Any] = []
        self._plan_runners: list[Any] = []
        self._retained_resources: list[Any] = []
        self._active_cursors: set[Any] = set()
        self._in_flight = 0
        self._closing = False
        self._closed = False

    @property
    def close_timeout_s(self) -> float:
        return self._close_timeout_s

    def retain_resources(self, *resources: Any) -> None:
        with self._resources_condition:
            if self._closing or self._closed:
                raise RuntimeError("local fragment executor is closing")
            self._retained_resources.extend(resource for resource in resources if resource is not None)

    def _begin_execution(self) -> None:
        with self._resources_condition:
            if self._closing or self._closed:
                raise RuntimeError("local fragment executor is closing")
            self._in_flight += 1

    def _end_execution(self) -> None:
        with self._resources_condition:
            if self._in_flight <= 0:
                raise RuntimeError("local fragment executor execution ownership underflow")
            self._in_flight -= 1
            self._resources_condition.notify_all()

    def _register_cursor(self, cursor: Any) -> bool:
        with self._resources_condition:
            self._active_cursors.add(cursor)
            return not self._closing

    def _unregister_cursor(self, cursor: Any) -> None:
        with self._resources_condition:
            self._active_cursors.discard(cursor)
            self._resources_condition.notify_all()

    def request_shutdown(self) -> None:
        """Fence new fragment calls and interrupt cursors currently in native execution."""
        interrupt_errors: list[BaseException] = []
        with self._resources_condition:
            if self._closed:
                return
            self._closing = True
            # Keep cursor lifecycle ownership until every interrupt returns.
            # The fragment finally block must unregister under this condition
            # before it can close the cursor, so Close() cannot clear the
            # native connection while Interrupt() is reading it.
            for cursor in list(self._active_cursors):
                try:
                    cursor.interrupt()
                except BaseException as exc:
                    if cursor in self._active_cursors:
                        interrupt_errors.append(exc)
        if interrupt_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in interrupt_errors)
            raise RuntimeError(f"failed to interrupt local fragment execution during close: {details}") from (
                interrupt_errors[0]
            )

    def close(self, *, timeout_s: float | None = None) -> None:
        timeout_s = self._close_timeout_s if timeout_s is None else float(timeout_s)
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("local fragment executor close timeout must be finite and non-negative")
        deadline = time.monotonic() + timeout_s
        request_error: BaseException | None = None
        with self._resources_condition:
            if self._closed:
                return
            shutdown_requested = self._closing
        if not shutdown_requested:
            try:
                self.request_shutdown()
            except BaseException as exc:
                request_error = exc
        with self._resources_condition:
            while self._in_flight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"local fragment executor did not drain before close: active_executions={self._in_flight}"
                    )
                self._resources_condition.wait(timeout=remaining)
            connections = list(self._connections)
            self._connections.clear()
            self._plan_runners.clear()
            retained_resources = self._retained_resources
            self._retained_resources = []
            self._active_cursors.clear()
            self._closed = True
        for conn in connections:
            try:
                conn.close()
            except Exception:
                pass
        retained_resources.clear()
        if request_error is not None:
            raise request_error

    @staticmethod
    def _configure_conn(conn: Any) -> None:
        duckdb_memory_limit = os.environ.get("VANE_DUCKDB_MEMORY_BUDGET_BYTES")
        if duckdb_memory_limit:
            apply_duckdb_memory_limit(conn, int(duckdb_memory_limit))
        duckdb_threads = os.environ.get("VANE_DUCKDB_THREADS")
        if duckdb_threads:
            conn.execute(f"SET threads={int(duckdb_threads)}")
        conn.execute("SET local_exchange_streaming=true")
        le_buf = os.environ.get("VANE_LOCAL_EXCHANGE_BUFFER", "32MB")
        conn.execute(f"SET local_exchange_buffer_bytes = '{le_buf}'")
        conn.execute("SET arrow_large_buffer_size=true")

    def _get_conn(self) -> Any:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        import vane

        conn = vane.connect()
        self._configure_conn(conn)
        self._local.conn = conn
        with self._resources_lock:
            self._connections.append(conn)
        return conn

    def _get_plan_runner(self) -> Any:
        plan_runner = getattr(self._local, "plan_runner", None)
        if plan_runner is not None:
            return plan_runner
        DistributedPhysicalPlanRunner = require_ray_cxx_attr("DistributedPhysicalPlanRunner")
        plan_runner = DistributedPhysicalPlanRunner()
        self._local.plan_runner = plan_runner
        with self._resources_lock:
            self._plan_runners.append(plan_runner)
        return plan_runner

    def __call__(self, request: Mapping[str, Any]) -> Any:
        self._begin_execution()
        cursor = None
        cursor_registered = False
        try:
            request_payload = dict(request)
            task_attempt_id = FteTaskAttemptId.coerce(request_payload.get("task_id"))
            context = NativeFteWorkerManagerBackend.materialize_task_context(
                request_payload,
                merge_scan_split_batches=require_ray_cxx_attr("merge_scan_split_batches"),
            )
            scan_split_batch_map, exchange_source_task_map = _native_task_maps_from_context(context)
            plan = request_payload.get("fragment_plan")
            if plan is None:
                raise RuntimeError("local fragment execution requires fragment_plan")

            conn = self._get_conn()
            if hasattr(plan, "clone"):
                with self._plan_clone_lock:
                    plan = plan.clone(conn)
            cursor = conn.cursor()
            accepting_work = self._register_cursor(cursor)
            cursor_registered = True
            if not accepting_work:
                try:
                    cursor.interrupt()
                except Exception:
                    pass
                raise RuntimeError("local fragment executor is closing")
            return self._get_plan_runner().execute_native(
                cursor,
                plan,
                scan_split_batch_map or None,
                exchange_source_task_map or None,
                _copy_output_info_from_context(context),
                request_payload.get("exchange_sink_instance"),
                request_payload.get("fte_scan_source_queues"),
                request_payload.get("fte_exchange_source_queues"),
                request_payload.get("dynamic_filter_domains"),
                request_payload.get("native_progress_callback"),
                {"task_id": str(task_attempt_id)},
            )
        finally:
            try:
                if cursor_registered:
                    self._unregister_cursor(cursor)
            finally:
                try:
                    if cursor is not None:
                        cursor.close()
                except Exception:
                    pass
                finally:
                    self._end_execution()


class LocalRunner(Runner):
    name = "local"

    def __init__(
        self,
        *,
        num_workers: int | None = 1,
        max_running_tasks: Any = None,
        execution_mode: str | None = "in_process",
    ) -> None:
        ensure_vane_session_dir()
        self.num_workers = _normalize_num_workers(num_workers)
        self.max_running_tasks = _normalize_max_running_tasks(max_running_tasks)
        self.execution_mode = _normalize_execution_mode(execution_mode)
        os.environ["VANE_LOCAL_FTE_WORKERS"] = str(self.num_workers)
        os.environ["VANE_LOCAL_FTE_EXECUTION_MODE"] = self.execution_mode

    def run_iter(self, relation: Any) -> Iterator[Any]:
        raise NotImplementedError("local FTE run_iter is not implemented yet")

    def run_iter_tables(self, relation: Any) -> Iterator[pa.Table]:
        raise NotImplementedError("local FTE run_iter_tables is not implemented yet")

    @staticmethod
    def _progress_snapshot(
        backend: NativeFteWorkerManagerBackend,
        query_id: str,
        started_at: float,
    ) -> dict[str, Any]:
        return build_progress_snapshot(
            {"queries": {query_id: backend.fte_query_status(query_id)}},
            query_id,
            started_at=started_at,
        )

    def run_write(self, relation: Any) -> dict[str, Any]:
        import vane

        _preload_arrow_dataset_imports()

        PyLogicalPlan = require_ray_cxx_attr("PyLogicalPlan")
        DistributedPhysicalPlanRunner = require_ray_cxx_attr("DistributedPhysicalPlanRunner")

        query_id = str(uuid.uuid4())
        logical_plan = PyLogicalPlan.from_duckdb_write_relation(relation, query_id)
        conn = vane.connect()
        fragment_executor = _InProcessFragmentExecutor()
        backend = NativeFteWorkerManagerBackend(
            execute_fn=fragment_executor,
            num_workers=self.num_workers,
            max_running_tasks=self.max_running_tasks,
        )
        udf_actor_pools: list[Any] = []
        renderer = None
        write_succeeded = False
        try:
            physical_plan = logical_plan.to_physical_plan(conn)
            from vane.execution.udf_subprocess import ensure_local_subprocess_actor_pools_for_plan

            udf_actor_pools, _ = ensure_local_subprocess_actor_pools_for_plan(physical_plan, conn=conn)
            # If a bounded backend shutdown ever times out, an in-flight native
            # call still owns this executor. Keep its driver and actor
            # dependencies reachable until the explicit fragment drain
            # succeeds instead of letting local-variable teardown destroy them.
            fragment_executor.retain_resources(conn, *udf_actor_pools)
            plan_runner = DistributedPhysicalPlanRunner(backend)

            started_at = time.time()
            if progress_enabled("local"):
                renderer = ProgressRenderer(lambda: self._progress_snapshot(backend, query_id, started_at))

            def execute_write() -> dict[str, Any]:
                result = plan_runner.run_copy_plan(physical_plan, conn)
                if not isinstance(result, dict):
                    raise TypeError("DistributedPhysicalPlanRunner.run_copy_plan() must return a dict")
                return result

            write_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vane-local-fte-write")
            try:
                future = write_executor.submit(execute_write)
                if renderer is None:
                    result = _require_known_copy_outcome(query_id, future.result())
                    write_succeeded = True
                    return result
                while True:
                    try:
                        result = _require_known_copy_outcome(
                            query_id,
                            future.result(timeout=renderer.interval_s),
                        )
                        write_succeeded = True
                        break
                    except TimeoutError:
                        renderer.update()
                renderer.update(force=True)
                return result
            except Exception:
                if renderer is not None:
                    try:
                        renderer.update(force=True)
                    except Exception:
                        # Progress is diagnostic and must not replace the
                        # write's terminal error, especially UNKNOWN.
                        pass
                raise
            finally:
                primary_error = sys.exc_info()[1]
                progress_error: Exception | None = None
                if renderer is not None:
                    try:
                        renderer.finish(final_state="FINISHED" if write_succeeded else None)
                    except Exception as error:
                        if (
                            not _record_unknown_copy_cleanup_errors(
                                primary_error,
                                "progress finalization",
                                [error],
                            )
                            and primary_error is None
                        ):
                            progress_error = error
                shutdown_error: Exception | None = None
                try:
                    write_executor.shutdown(wait=True)
                except Exception as error:
                    if (
                        not _record_unknown_copy_cleanup_errors(
                            primary_error,
                            "write executor shutdown",
                            [error],
                        )
                        and primary_error is None
                    ):
                        shutdown_error = error
                if shutdown_error is not None:
                    raise shutdown_error
                if progress_error is not None:
                    raise progress_error
        finally:
            primary_error = sys.exc_info()[1]
            cleanup_errors = _shutdown_local_write_resources(
                backend,
                fragment_executor,
                conn,
                udf_actor_pools,
                kill_actor_pools=not write_succeeded,
                timeout_s=fragment_executor.close_timeout_s,
            )
            _record_unknown_copy_cleanup_errors(
                primary_error,
                "local write resource shutdown",
                cleanup_errors,
            )
            if write_succeeded and primary_error is None and cleanup_errors:
                details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
                raise RuntimeError(f"failed to shut down local write resources: {details}") from cleanup_errors[0]

    def run_datasink(self, relation: Any) -> dict[str, Any]:
        """Execute one immediate, idempotent DataSink with the local FTE backend."""

        import vane

        _preload_arrow_dataset_imports()
        PyLogicalPlan = require_ray_cxx_attr("PyLogicalPlan")
        DistributedPhysicalPlanRunner = require_ray_cxx_attr("DistributedPhysicalPlanRunner")

        query_id = str(uuid.uuid4())
        logical_plan = PyLogicalPlan.from_duckdb_relation(relation, query_id)
        conn = vane.connect()
        fragment_executor = _InProcessFragmentExecutor()
        backend = NativeFteWorkerManagerBackend(
            execute_fn=fragment_executor,
            num_workers=self.num_workers,
            max_running_tasks=self.max_running_tasks,
        )
        udf_actor_pools: list[Any] = []
        renderer = None
        result: dict[str, Any] | None = None
        progress_diagnostics: list[tuple[str, BaseException]] = []

        def format_warning(stage: str, error: BaseException) -> str:
            try:
                message = str(error)
            except BaseException:
                message = "<error message unavailable>"
            return f"{stage} failed: {type(error).__name__}: {message}"

        def append_warning(stage: str, error: BaseException) -> None:
            if result is None:
                return
            raw_warnings = result.get("data_sink_cleanup_warnings", ())
            warnings = [raw_warnings] if isinstance(raw_warnings, str) and raw_warnings else []
            if not isinstance(raw_warnings, str) and isinstance(raw_warnings, (list, tuple)):
                warnings.extend(str(warning) for warning in raw_warnings)
            warnings.append(format_warning(stage, error))
            result["data_sink_cleanup_warnings"] = warnings

        try:
            physical_plan = logical_plan.to_physical_plan(conn)
            from vane.execution.udf_subprocess import ensure_local_subprocess_actor_pools_for_plan

            udf_actor_pools, _ = ensure_local_subprocess_actor_pools_for_plan(physical_plan, conn=conn)
            fragment_executor.retain_resources(conn, *udf_actor_pools)
            plan_runner = DistributedPhysicalPlanRunner(backend)
            started_at = time.time()
            if progress_enabled("local"):
                renderer = ProgressRenderer(lambda: self._progress_snapshot(backend, query_id, started_at))

            def execute_datasink() -> dict[str, Any]:
                native_result = plan_runner.run_datasink_plan(physical_plan, conn)
                if not isinstance(native_result, dict):
                    raise TypeError("DistributedPhysicalPlanRunner.run_datasink_plan() must return a dict")
                return native_result

            write_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vane-local-fte-datasink")
            try:
                future = write_executor.submit(execute_datasink)
                if renderer is None:
                    result = future.result()
                else:
                    progress_updates_enabled = True
                    while True:
                        try:
                            result = future.result(timeout=renderer.interval_s)
                            break
                        except TimeoutError:
                            if future.done():
                                result = future.result()
                                break
                            if progress_updates_enabled:
                                try:
                                    renderer.update()
                                except BaseException as progress_error:
                                    progress_diagnostics.append(("DataSink progress update", progress_error))
                                    progress_updates_enabled = False
                    if progress_updates_enabled:
                        try:
                            renderer.update(force=True)
                        except BaseException as progress_error:
                            progress_diagnostics.append(("DataSink progress update", progress_error))
                    for stage, progress_error in progress_diagnostics:
                        append_warning(stage, progress_error)
            finally:
                primary_error = sys.exc_info()[1]
                if result is None and primary_error is not None and hasattr(primary_error, "add_note"):
                    for stage, progress_error in progress_diagnostics:
                        primary_error.add_note(format_warning(stage, progress_error))
                finalization_error: BaseException | None = None
                try:
                    write_executor.shutdown(wait=True)
                except BaseException as shutdown_error:
                    if result is not None:
                        append_warning("DataSink executor shutdown", shutdown_error)
                    elif primary_error is not None:
                        if hasattr(primary_error, "add_note"):
                            primary_error.add_note(format_warning("DataSink executor shutdown", shutdown_error))
                    else:
                        finalization_error = shutdown_error
                if renderer is not None:
                    try:
                        renderer.finish(final_state="FINISHED" if result is not None else None)
                    except BaseException as progress_error:
                        if result is not None:
                            append_warning("DataSink progress finalization", progress_error)
                        elif primary_error is not None:
                            if hasattr(primary_error, "add_note"):
                                primary_error.add_note(format_warning("DataSink progress finalization", progress_error))
                        elif finalization_error is None:
                            finalization_error = progress_error
                        elif hasattr(finalization_error, "add_note"):
                            finalization_error.add_note(
                                format_warning("DataSink progress finalization", progress_error)
                            )
                if finalization_error is not None:
                    raise finalization_error
        finally:
            primary_error = sys.exc_info()[1]
            cleanup_errors = _shutdown_local_write_resources(
                backend,
                fragment_executor,
                conn,
                udf_actor_pools,
                kill_actor_pools=result is None or result.get("outcome_unknown") is True,
                timeout_s=fragment_executor.close_timeout_s,
            )
            if result is not None:
                for cleanup_error in cleanup_errors:
                    append_warning("DataSink local resource shutdown", cleanup_error)
            elif primary_error is not None and hasattr(primary_error, "add_note"):
                for cleanup_error in cleanup_errors:
                    primary_error.add_note(format_warning("DataSink local resource shutdown", cleanup_error))
            elif cleanup_errors:
                details = "; ".join(
                    format_warning("DataSink local resource shutdown", error) for error in cleanup_errors
                )
                raise RuntimeError(f"failed to shut down local DataSink resources: {details}") from cleanup_errors[0]

        if result is None:
            raise RuntimeError("local DataSink execution completed without a result")
        return result
