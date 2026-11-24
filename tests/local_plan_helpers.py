# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Mapping
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any

import pyarrow as pa

import vane
from tests.result_stream_helpers import collect_result_stream
from vane._ray_cxx import require_ray_cxx_attr
from vane.execution.udf_subprocess import ensure_local_subprocess_actor_pools_for_plan
from vane.runners.fte import FteTaskAttemptId
from vane.runners.fte.backends.native import NativeFteWorkerManagerBackend
from vane.runners.fte.memory_config import apply_duckdb_memory_limit

if TYPE_CHECKING:
    from vane.ai._errors import DiagnosticError


# Test support for exercising distributed FTE plans without a Ray cluster.
# Public local execution uses the native query path and has no FTE runner.
_ARROW_DATASET_PRELOAD_LOCK = threading.Lock()
_ARROW_DATASET_PRELOADED = False


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
        self._diagnostic: DiagnosticError | None = None

    @property
    def diagnostic(self) -> DiagnosticError | None:
        with self._resources_lock:
            return self._diagnostic

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

        conn = vane._native._connect_with_runner("local")
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
        execution_plan = None
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
            cursor = conn.cursor()
            if hasattr(plan, "clone"):
                with self._plan_clone_lock:
                    # Native execution uses the connection retained by the
                    # bound plan. Bind it to the cursor we will interrupt.
                    execution_plan = plan.clone(cursor)
            else:
                execution_plan = plan
            accepting_work = self._register_cursor(cursor)
            cursor_registered = True
            if not accepting_work:
                try:
                    cursor.interrupt()
                except Exception:
                    pass
                raise RuntimeError("local fragment executor is closing")
            plan_runner = self._get_plan_runner()
            try:
                return plan_runner.execute_native(
                    cursor,
                    execution_plan,
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
            except (RuntimeError, ValueError) as error:
                from vane.ai._errors import _native_query_diagnostic

                # Native FTE later stringifies/truncates fragment failures. Keep
                # the first typed diagnostic on this query's executor beforehand.
                diagnostic = _native_query_diagnostic(error)
                if diagnostic is not None:
                    with self._resources_lock:
                        if self._diagnostic is None:
                            self._diagnostic = diagnostic
                raise
        finally:
            try:
                # Operator state can retain cursor-owned resources. Destroy
                # the execution clone before closing its binding connection.
                execution_plan = None
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


def execute_local_physical_plan(connection, plan) -> pa.Table:
    """Execute a prepared coordinator plan through the local fragment scheduler."""
    _preload_arrow_dataset_imports()
    with ExitStack() as cleanup:
        pools, _ = ensure_local_subprocess_actor_pools_for_plan(plan, conn=connection)
        for pool in pools:
            cleanup.callback(pool.shutdown, kill=True)

        executor = _InProcessFragmentExecutor()
        cleanup.callback(executor.close)
        executor.retain_resources(connection, *pools)
        backend = NativeFteWorkerManagerBackend(execute_fn=executor, num_workers=1, max_running_tasks=1)
        cleanup.callback(backend.shutdown)
        runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
        cleanup.callback(runner.shutdown)
        cleanup.callback(runner.drop_query_fragments, plan.idx())

        partitions = collect_result_stream(runner.run_plan(plan, connection))
        return pa.concat_tables(partitions)
