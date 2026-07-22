# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from numbers import Integral
from typing import TYPE_CHECKING, Any

from vane._ray_cxx import require_ray_cxx_attr
from vane._vane_session import ensure_vane_session_dir
from vane.runners.fte.backends.native import NativeFteWorkerManagerBackend
from vane.runners.fte.memory_config import apply_duckdb_memory_limit
from vane.runners.progress import ProgressRenderer, build_progress_snapshot, progress_enabled
from vane.runners.runner import Runner

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import pyarrow as pa


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
        import pyarrow.dataset  # noqa: F401

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


def _native_task_maps_from_context(context: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    scan_task_map: dict[str, Any] = {}
    exchange_source_task_map: dict[str, Any] = {}
    for key, value in (context or {}).items():
        if key.startswith("scan_task:"):
            node_id = key.split(":", 1)[1]
            if node_id:
                scan_task_map[node_id] = value
        elif key.startswith("exchange_source_task:"):
            node_id = key.split(":", 1)[1]
            if node_id:
                exchange_source_task_map[node_id] = value
    return scan_task_map, exchange_source_task_map


class _InProcessFragmentExecutor:
    def __init__(self) -> None:
        self._local = threading.local()
        self._resources_lock = threading.Lock()
        self._plan_clone_lock = threading.Lock()
        self._connections: list[Any] = []
        self._plan_runners: list[Any] = []

    def close(self) -> None:
        with self._resources_lock:
            connections = list(self._connections)
            self._connections.clear()
            self._plan_runners.clear()
        for conn in connections:
            try:
                conn.close()
            except Exception:
                pass

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
        import vane as _duckdb

        conn = _duckdb.connect()
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
        request_payload = dict(request)
        context = NativeFteWorkerManagerBackend.materialize_task_context(
            request_payload,
            merge_scan_task_descriptors=require_ray_cxx_attr("merge_scan_task_descriptors"),
        )
        scan_task_map, exchange_source_task_map = _native_task_maps_from_context(context)
        plan = request_payload.get("fragment_plan")
        if plan is None:
            raise RuntimeError("local fragment execution requires fragment_plan")

        conn = self._get_conn()
        if hasattr(plan, "clone"):
            with self._plan_clone_lock:
                plan = plan.clone(conn)
        cursor = conn.cursor()
        try:
            return self._get_plan_runner().execute_native(
                cursor,
                plan,
                scan_task_map or None,
                exchange_source_task_map or None,
                _copy_output_info_from_context(context),
                request_payload.get("exchange_sink_instance"),
                request_payload.get("fte_scan_source_queues"),
                request_payload.get("fte_exchange_source_queues"),
                request_payload.get("dynamic_filter_domains"),
                request_payload.get("native_progress_callback"),
            )
        finally:
            try:
                cursor.close()
            except Exception:
                pass


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

    def run_iter(self, relation: Any, results_buffer_size: int | None = None) -> Iterator[Any]:
        raise NotImplementedError("local FTE run_iter is not implemented yet")

    def run_iter_tables(self, relation: Any, results_buffer_size: int | None = None) -> Iterator[pa.Table]:
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
        import vane as _duckdb

        _preload_arrow_dataset_imports()

        PyLogicalPlan = require_ray_cxx_attr("PyLogicalPlan")
        DistributedPhysicalPlanRunner = require_ray_cxx_attr("DistributedPhysicalPlanRunner")

        query_id = str(uuid.uuid4())
        logical_plan = PyLogicalPlan.from_duckdb_relation(relation, query_id)
        conn = _duckdb.connect()
        fragment_executor = _InProcessFragmentExecutor()
        backend = NativeFteWorkerManagerBackend(
            execute_fn=fragment_executor,
            num_workers=self.num_workers,
            max_running_tasks=self.max_running_tasks,
        )
        udf_actor_pools = []
        renderer = None
        try:
            physical_plan = logical_plan.to_physical_plan(conn)
            from vane.execution.udf_subprocess import ensure_local_subprocess_actor_pools_for_plan

            udf_actor_pools, _ = ensure_local_subprocess_actor_pools_for_plan(physical_plan, conn=conn)
            plan_runner = DistributedPhysicalPlanRunner(backend)

            started_at = time.time()
            if progress_enabled("local"):
                renderer = ProgressRenderer(lambda: self._progress_snapshot(backend, query_id, started_at))

            def execute_write() -> dict[str, Any]:
                return plan_runner.run_copy_plan(physical_plan, conn)

            write_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vane-local-fte-write")
            write_succeeded = False
            try:
                future = write_executor.submit(execute_write)
                if renderer is None:
                    result = future.result()
                    write_succeeded = True
                    return result
                while True:
                    try:
                        result = future.result(timeout=renderer.interval_s)
                        write_succeeded = True
                        break
                    except TimeoutError:
                        renderer.update()
                renderer.update(force=True)
                return result
            except Exception:
                if renderer is not None:
                    renderer.update(force=True)
                raise
            finally:
                if renderer is not None:
                    renderer.finish(final_state="FINISHED" if write_succeeded else None)
                write_executor.shutdown(wait=True)
        finally:
            for pool in reversed(udf_actor_pools):
                try:
                    pool.shutdown(kill=True)
                except Exception:
                    pass
            try:
                backend.shutdown()
            finally:
                fragment_executor.close()
                try:
                    conn.close()
                except Exception:
                    pass
