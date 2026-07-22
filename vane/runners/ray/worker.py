# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from typing import Any, NamedTuple

import ray

from vane._ray_cxx import require_ray_cxx_attr

# Avoid importing C++ bindings at module import time (may not be registered yet).
# Resolve `vane.ray_cxx` attributes lazily at use-time instead.
from vane.event_loop import set_event_loop
from vane.runners.common import PartitionMetadata
from vane.runners.fte import (
    FteTaskAttemptId,
    FteWorkerTaskManager,
    collect_spooling_output_stats,
    materialize_task_inputs,
    validate_fte_status_identity,
)
from vane.runners.fte.debug_memory import describe_result_payload, log_debug, process_memory_snapshot
from vane.runners.fte.fte_config import FteWorkerAdmissionConfig
from vane.runners.fte.memory_config import apply_duckdb_memory_limit


def _fte_applied_control_status(
    operation: str,
    task_id: str | dict[str, Any],
    status: dict[str, Any],
) -> dict[str, Any]:
    result = dict(status)
    expected = FteTaskAttemptId.coerce(task_id)
    try:
        validate_fte_status_identity(result, expected)
    except Exception as exc:
        raise RuntimeError(
            f"FTE control {operation} returned a mismatched task identity for {expected}: {exc}"
        ) from exc
    result["_fte_control_operation"] = str(operation)
    result["_fte_control_applied"] = str(result.get("state") or "").upper() != "UNKNOWN"
    return result


def _env_flag_enabled(*names: str) -> bool:
    for name in names:
        value = os.getenv(name, "")
        if value and value.strip().lower() not in ("0", "false", "no"):
            return True
    return False


def _ray_worker_memory_debug_enabled() -> bool:
    return _env_flag_enabled("VANE_RAY_WORKER_MEMORY_DEBUG", "VANE_FTE_RESULT_DEBUG", "DUCKDB_DISTRIBUTED_DEBUG")


def _ray_worker_memory_log(event: str, **fields: Any) -> None:
    if not _ray_worker_memory_debug_enabled():
        return
    payload = process_memory_snapshot()
    payload.update(fields)
    log_debug("vane-ray-worker-memory", event, **payload)


def _fte_worker_label() -> str:
    return os.getenv("VANE_WORKER_ID", "").strip() or os.getenv("VANE_FTE_WORKER_ID", "").strip() or "-"


def _ensure_python_datasource_runtime() -> None:
    import vane.datasource  # noqa: F401


def _chaos_worker_loss_matches(task_id: FteTaskAttemptId) -> bool:
    if not _env_flag_enabled("VANE_FTE_CHAOS_KILL_WORKER_ON_RUNNING"):
        return False
    target_worker_index = os.getenv("VANE_FTE_CHAOS_KILL_WORKER_INDEX", "").strip()
    if target_worker_index:
        worker_index = os.getenv("VANE_WORKER_INDEX", "").strip()
        target_worker_indices = {index.strip() for index in target_worker_index.split(",") if index.strip()}
        if worker_index not in target_worker_indices:
            return False
    target_attempt = os.getenv("VANE_FTE_CHAOS_KILL_ATTEMPT_ID", "0").strip()
    if target_attempt and int(target_attempt) != int(task_id.attempt_id):
        return False
    target_task = os.getenv("VANE_FTE_CHAOS_KILL_TASK_ID", "").strip()
    return not (target_task and target_task != str(task_id))


def _cleanup_flight_shuffle_for_query(query_id: str) -> dict[str, int]:
    query_id = str(query_id or "").strip()
    if not query_id:
        return {
            "registry_entries_removed": 0,
            "storage_entries_removed": 0,
            "cleanup_errors": 0,
        }
    try:
        cleanup_fn = require_ray_cxx_attr(
            "cleanup_flight_shuffle_for_query",
            hint="Ensure the C++ ray extension is built with Flight shuffle cleanup support.",
        )
        raw = cleanup_fn(query_id)
    except Exception:
        return {
            "registry_entries_removed": 0,
            "storage_entries_removed": 0,
            "cleanup_errors": 1,
        }
    if not isinstance(raw, dict):
        return {
            "registry_entries_removed": 0,
            "storage_entries_removed": 0,
            "cleanup_errors": 1,
        }
    return {
        "registry_entries_removed": int(raw.get("registry_entries_removed", 0)),
        "storage_entries_removed": int(raw.get("storage_entries_removed", 0)),
        "cleanup_errors": int(raw.get("cleanup_errors", 0)),
    }


def _maybe_chaos_kill_worker(task_id: FteTaskAttemptId) -> None:
    if not _chaos_worker_loss_matches(task_id):
        return
    delay_s = float(os.getenv("VANE_FTE_CHAOS_KILL_DELAY_S", "0") or "0")
    if delay_s > 0:
        import time

        time.sleep(delay_s)
    sys.stderr.flush()
    os._exit(88)


def _normalize_native_task_result(result: Any):
    native_type = require_ray_cxx_attr(
        "NativeDistributedTaskResult",
        hint="Ensure the C++ ray extension is built and importable in this process.",
    )
    if not isinstance(result, native_type):
        raise TypeError("execute_native must return NativeDistributedTaskResult")

    payloads = list(result.partition_payloads)
    partition_metadatas = [
        PartitionMetadata(int(metadata.num_rows), int(metadata.size_bytes)) for metadata in result.partition_metadatas
    ]
    result_schema = dict(result.result_schema) if result.result_schema is not None else None
    stats = list(result.stats)
    task_stats = result.task_stats
    if isinstance(task_stats, dict):
        task_stats = dict(task_stats)
    completion_status = result.completion_status
    flight_port = int(result.flight_port or 0)
    exchange_sink_instance = result.exchange_sink_instance
    return (
        payloads,
        partition_metadatas,
        result_schema,
        stats,
        completion_status,
        flight_port,
        exchange_sink_instance,
        task_stats,
    )


def _validate_fte_output_publication(
    partition_metadatas: list[PartitionMetadata],
    query_task_lease: dict[str, Any],
) -> tuple[int, ...]:
    """Validate all FTE result bytes before the worker publishes any ObjectRef."""
    lease_id = str(query_task_lease.get("lease_id") or "").strip()
    query_id = str(query_task_lease.get("query_id") or "").strip()
    stage_id = str(query_task_lease.get("stage_id") or "").strip()
    attempt_id = str(query_task_lease.get("attempt_id") or "").strip()
    target_bytes = int(query_task_lease.get("target_output_block_bytes") or 0)
    window_bytes = int(query_task_lease.get("output_window_bytes") or 0)
    if not lease_id or not query_id or not stage_id or not attempt_id:
        raise RuntimeError("FTE output publication requires a complete query task lease identity")
    if target_bytes <= 0 or window_bytes <= 0:
        raise RuntimeError("FTE output publication requires positive target_output_block_bytes and output_window_bytes")

    normalized_sizes: list[int] = []
    for index, metadata in enumerate(partition_metadatas):
        num_rows = int(metadata.num_rows)
        raw_size = int(metadata.size_bytes or 0)
        if raw_size <= 0 and num_rows > 0:
            raise RuntimeError(
                f"FTE output block {index} is missing positive size_bytes: "
                f"query={query_id} stage={stage_id} attempt={attempt_id}"
            )
        size_bytes = max(1, raw_size)
        if size_bytes > target_bytes:
            raise RuntimeError(
                f"FTE output block {index} size {size_bytes} exceeds target {target_bytes}: "
                f"query={query_id} stage={stage_id} task_lease={lease_id} attempt={attempt_id}"
            )
        normalized_sizes.append(size_bytes)

    total_bytes = sum(normalized_sizes)
    if total_bytes > window_bytes:
        raise RuntimeError(
            f"FTE total output bytes {total_bytes} exceed task window {window_bytes}: "
            f"query={query_id} stage={stage_id} task_lease={lease_id} attempt={attempt_id}"
        )
    return tuple(normalized_sizes)


def _normalize_stats_for_ray(stats_payload: Any) -> list[int]:
    if stats_payload is None:
        return []
    if isinstance(stats_payload, (bytes, bytearray)):
        return list(stats_payload)
    if isinstance(stats_payload, memoryview):
        return list(stats_payload.tobytes())
    if isinstance(stats_payload, (list, tuple)):
        values = [int(value) for value in stats_payload]
        for value in values:
            if value < 0 or value > 255:
                raise ValueError(f"Stats payload value out of range [0, 255]: {value}")
        return values
    return []


def _configure_duckdb_s3(conn: Any) -> None:
    """Configure S3 settings on a DuckDB connection from AWS environment variables.

    Ray actors create bare ``vane.connect()`` instances that lack S3 endpoint
    and credential settings.  The standard ``AWS_*`` env vars *are* propagated
    to workers, but DuckDB does not automatically map ``AWS_ENDPOINT_URL`` to
    its ``s3_endpoint`` setting, causing S3 reads to go to the real AWS instead
    of a local MinIO / compatible store.
    """
    from urllib.parse import urlparse

    endpoint_url = os.environ.get("AWS_ENDPOINT_URL", "").strip()
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
    session_token = os.environ.get("AWS_SESSION_TOKEN", "").strip()
    region = (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "").strip()

    if not endpoint_url and not access_key:
        return  # nothing to configure

    try:
        conn.execute("LOAD httpfs")
    except Exception:
        conn.execute("INSTALL httpfs; LOAD httpfs")

    def _q(s: str) -> str:
        return s.replace("'", "''")

    # Use GLOBAL so cursors (which create new ClientContexts) inherit settings.
    if region:
        conn.execute(f"SET GLOBAL s3_region='{_q(region)}'")
    if access_key:
        conn.execute(f"SET GLOBAL s3_access_key_id='{_q(access_key)}'")
    if secret_key:
        conn.execute(f"SET GLOBAL s3_secret_access_key='{_q(secret_key)}'")
    if session_token:
        conn.execute(f"SET GLOBAL s3_session_token='{_q(session_token)}'")
    if endpoint_url:
        parsed = urlparse(endpoint_url)
        endpoint = parsed.netloc or parsed.path
        use_ssl = parsed.scheme == "https"
        conn.execute(f"SET GLOBAL s3_endpoint='{_q(endpoint)}'")
        conn.execute(f"SET GLOBAL s3_use_ssl={'true' if use_ssl else 'false'}")
        conn.execute("SET GLOBAL s3_url_style='path'")

    # Keep-alive MUST stay enabled: disabling it creates a new TCP connection
    # per S3 request, which exhausts ephemeral ports via TIME_WAIT buildup
    # (55K+ TIME_WAIT sockets observed with keep_alive=false).
    conn.execute("SET GLOBAL http_keep_alive=true")
    # Increase retries to handle transient connection failures during
    # concurrent S3 access from many DuckDB threads.
    conn.execute("SET GLOBAL http_retries=10")
    conn.execute("SET GLOBAL http_retry_wait_ms=100")
    conn.execute("SET GLOBAL http_retry_backoff=1.5")

    # Create an explicit S3 secret so that credentials are available to the
    # DuckDB secret manager even when internal code paths use a null FileOpener.
    if endpoint_url and access_key and secret_key:
        use_ssl_str = "true" if (urlparse(endpoint_url).scheme == "https") else "false"
        endpoint_host = urlparse(endpoint_url).netloc or urlparse(endpoint_url).path
        region_val = region or "us-east-1"
        conn.execute(f"""
            CREATE SECRET IF NOT EXISTS __vane_s3 (
                TYPE S3,
                KEY_ID '{_q(access_key)}',
                SECRET '{_q(secret_key)}',
                ENDPOINT '{_q(endpoint_host)}',
                REGION '{_q(region_val)}',
                USE_SSL {use_ssl_str},
                URL_STYLE 'path'
            )
        """)


def _configure_ray_worker_conn(conn: Any, duckdb_memory_bytes: int) -> None:
    apply_duckdb_memory_limit(conn, duckdb_memory_bytes)
    duckdb_threads = os.environ.get("VANE_DUCKDB_THREADS")
    if duckdb_threads:
        try:
            conn.execute(f"SET threads={int(duckdb_threads)}")
        except Exception:
            pass
    try:
        _configure_duckdb_s3(conn)
    except Exception:
        pass
    try:
        conn.execute("SET local_exchange_streaming=true")
    except Exception:
        pass
    le_buf = os.environ.get("VANE_LOCAL_EXCHANGE_BUFFER", "32MB")
    try:
        conn.execute(f"SET local_exchange_buffer_bytes = '{le_buf}'")
    except Exception:
        pass
    try:
        conn.execute("SET arrow_large_buffer_size=true")
    except Exception:
        pass


def _warm_up_python_native_dependencies() -> None:
    """Import native Python dependencies before concurrent DuckDB tasks start.

    DuckDB's Python/Arrow bridge can lazily import PyArrow submodules from C++
    plan execution threads.  In a Ray async actor, several native tasks can
    enter that path at once, which can leave one thread inside PyArrow C
    extension initialization while another waits on DuckDB/Python locks.  Warm
    these modules on the actor init thread so task threads only use initialized
    modules.
    """
    try:
        __import__("pyarrow")
        __import__("pyarrow.compute")
        __import__("pyarrow.dataset")
        __import__("pyarrow.fs")
        __import__("pyarrow.parquet")
    except Exception:
        pass


def _apply_env_overrides(env_overrides: dict[str, str] | None) -> None:
    if not env_overrides:
        return
    for key, value in env_overrides.items():
        if value is None:
            continue
        os.environ[key] = str(value)


def _copy_output_info_from_context(context: dict[str, Any] | None) -> dict[str, str] | None:
    """Extract copy output info dict from task context for worker-driven path generation."""
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


def _extract_native_task_maps_from_context(
    context: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scan_task_map: dict[str, Any] = {}
    exchange_source_task_map: dict[str, Any] = {}
    if not context:
        return scan_task_map, exchange_source_task_map
    for key, value in context.items():
        if key.startswith("scan_task:"):
            node_id = key.split(":", 1)[1]
            if node_id:
                scan_task_map[node_id] = value
        elif key.startswith("exchange_source_task:"):
            node_id = key.split(":", 1)[1]
            if node_id:
                exchange_source_task_map[node_id] = value
    return scan_task_map, exchange_source_task_map


class WorkerTaskMetadata(NamedTuple):
    partition_metadatas: list[PartitionMetadata]
    result_schema: Any | None
    stats: Any
    flight_port: int = 0
    exchange_sink_instance: Any = None


@ray.remote(concurrency_groups={"execute": 128, "control": 512})
class RayWorkerActor:
    """RayWorkerActor is a ray actor that runs local physical plans on worker.

    It is a stateless, async actor, and can run multiple plans concurrently and is able to retry itself and it's tasks.
    """

    def __init__(
        self,
        num_cpus: int,
        num_gpus: int,
        duckdb_memory_bytes: int,
        task_heap_capacity_bytes: int,
        env_overrides: dict[str, str] | None = None,
    ) -> None:
        duckdb_memory_bytes = int(duckdb_memory_bytes)
        task_heap_capacity_bytes = int(task_heap_capacity_bytes)
        if duckdb_memory_bytes <= 0:
            raise ValueError("Ray worker duckdb_memory_bytes must be positive")
        if task_heap_capacity_bytes <= 0:
            raise ValueError("Ray worker task_heap_capacity_bytes must be positive")
        self._env_overrides = env_overrides or {}
        self._node_id = str(ray.get_runtime_context().get_node_id() or "").strip()
        if not self._node_id:
            raise RuntimeError("Ray worker runtime context is missing node_id")
        self._duckdb_memory_bytes = duckdb_memory_bytes
        self._task_heap_capacity_bytes = task_heap_capacity_bytes
        _apply_env_overrides(self._env_overrides)
        os.environ["VANE_WORKER"] = "1"
        if num_gpus > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
        try:
            loop = asyncio.get_running_loop()
            set_event_loop(loop)
            # Increase default thread pool to prevent starvation when many concurrent
            # tasks arrive via asyncio.to_thread() (e.g. 62 exchange tasks in Q2).
            import concurrent.futures

            loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=max(128, num_cpus * 8)))
        except RuntimeError:
            pass

        # Enable faulthandler to capture C++ crash signals (SIGSEGV, SIGABRT, etc.)
        import faulthandler

        faulthandler.enable(file=sys.stderr, all_threads=True)
        _warm_up_python_native_dependencies()
        _ensure_python_datasource_runtime()

        # Defer creation of the C++ plan runner until needed (avoids import-time failures)
        self._plan_runner: Any | None = None

        self._plan_fragments: dict[str, Any] = {}
        self._query_fragments: dict[str, set[str]] = {}
        self._fragment_query_ids: dict[str, str] = {}
        self._fragment_register_calls = 0
        self._fragment_registered_total = 0
        self._fragment_existing_total = 0
        self._fragment_lookup_hits = 0
        self._fragment_lookup_misses = 0
        self._fte_task_manager: FteWorkerTaskManager | None = None
        self._fte_admission_config = FteWorkerAdmissionConfig(
            max_running_tasks=max(1, int(num_cpus)),
            mode="lease",
            memory_budget_bytes=task_heap_capacity_bytes,
            task_memory_bytes=None,
        )

        # Shared DuckDB connection: all tasks share the same DatabaseInstance
        # (and thus the same TaskScheduler thread pool).  Each task creates a
        # lightweight cursor (new ClientContext) from this connection.
        # Eagerly create during __init__ so the ~2s startup cost overlaps
        # with actor creation instead of blocking the first task.
        self._shared_conn = None
        self._shared_conn_lock = threading.Lock()
        self._get_shared_conn()  # eagerly initialize
        _ray_worker_memory_log(
            "actor_initialized",
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            worker_label=_fte_worker_label(),
        )

    @ray.method(concurrency_group="control")
    def install_env_overrides(self, env_overrides: dict[str, str] | None) -> None:
        self._env_overrides = env_overrides or {}
        _apply_env_overrides(self._env_overrides)

    @ray.method(concurrency_group="control")
    async def register_fragments(self, fragments: list[dict[str, Any]]) -> dict[str, int]:
        """Register plan fragments in this actor.

        Each entry is expected to contain:
        - fragment_id: stable fragment id string
        - plan: plan object (PhysicalPlan / DistributedPhysicalPlan wrapper)
        - query_id: query identity for lifecycle cleanup
        """
        registered = 0
        existing = 0
        self._fragment_register_calls += 1
        pending_entries: list[dict[str, Any]] = []
        pending_refs: list[ray.ObjectRef] = []
        pending_ref_indexes: list[int] = []
        seen_new_fragment_ids: dict[str, str] = {}

        for entry in fragments:
            fragment_id = str(entry.get("fragment_id", "")).strip()
            if not fragment_id:
                raise ValueError("fragment registration requires non-empty fragment_id")
            query_id = str(entry.get("query_id", "")).strip()
            if not query_id:
                raise ValueError("fragment registration requires non-empty query_id")
            existing_owner = self._fragment_query_ids.get(fragment_id)
            if existing_owner is not None and existing_owner != query_id:
                raise RuntimeError(
                    "fragment registration query ownership mismatch: "
                    f"fragment={fragment_id} owner={existing_owner} requested={query_id}"
                )
            batch_owner = seen_new_fragment_ids.get(fragment_id)
            if batch_owner is not None and batch_owner != query_id:
                raise RuntimeError(
                    "fragment registration batch contains conflicting query ownership: "
                    f"fragment={fragment_id} owners={batch_owner},{query_id}"
                )
            if fragment_id in self._plan_fragments or batch_owner is not None:
                existing += 1
                continue
            plan = entry.get("plan")
            seen_new_fragment_ids[fragment_id] = query_id
            pending_entries.append(
                {
                    "fragment_id": fragment_id,
                    "plan": plan,
                    "query_id": query_id,
                }
            )
            if isinstance(plan, ray.ObjectRef):
                pending_refs.append(plan)
                pending_ref_indexes.append(len(pending_entries) - 1)

        if pending_refs:
            resolved_plans = await asyncio.gather(*pending_refs)
            for entry_index, resolved_plan in zip(pending_ref_indexes, resolved_plans, strict=False):
                pending_entries[entry_index]["plan"] = resolved_plan

        for entry in pending_entries:
            fragment_id = str(entry["fragment_id"])
            plan = entry.get("plan")
            if plan is None:
                raise ValueError(f"fragment {fragment_id} registration requires a physical plan")
            if fragment_id in self._plan_fragments:
                owner_query_id = self._fragment_query_ids[fragment_id]
                if owner_query_id != entry["query_id"]:
                    raise RuntimeError(
                        "fragment registration query ownership changed while awaiting plan: "
                        f"fragment={fragment_id} owner={owner_query_id} "
                        f"requested={entry['query_id']}"
                    )
                existing += 1
                continue
            self._plan_fragments[fragment_id] = plan
            query_id = str(entry.get("query_id", "")).strip()
            self._fragment_query_ids[fragment_id] = query_id
            self._query_fragments.setdefault(query_id, set()).add(fragment_id)
            registered += 1
        self._fragment_registered_total += registered
        self._fragment_existing_total += existing
        return {
            "registered": registered,
            "existing": existing,
            "total": len(self._plan_fragments),
        }

    @ray.method(concurrency_group="control")
    def drop_query_fragments(self, query_id: str) -> int:
        fragment_ids = self._query_fragments.pop(query_id, set())
        removed = 0
        for fragment_id in fragment_ids:
            if fragment_id in self._plan_fragments:
                self._plan_fragments.pop(fragment_id, None)
                self._fragment_query_ids.pop(fragment_id, None)
                removed += 1
        return removed

    @ray.method(concurrency_group="control")
    def stats_fragments(self) -> dict[str, int]:
        return {
            "fragments_total": len(self._plan_fragments),
            "queries_tracked": len(self._query_fragments),
            "register_calls": self._fragment_register_calls,
            "registered_total": self._fragment_registered_total,
            "existing_total": self._fragment_existing_total,
            "lookup_hits": self._fragment_lookup_hits,
            "lookup_misses": self._fragment_lookup_misses,
        }

    def _get_fte_task_manager(self) -> FteWorkerTaskManager:
        if self._fte_task_manager is None:
            self._fte_task_manager = FteWorkerTaskManager(
                self._execute_fte_request,
                admission_config=self._fte_admission_config,
                require_query_task_lease=True,
                worker_label=_fte_worker_label(),
            )
        return self._fte_task_manager

    async def _execute_fte_request(self, request: dict[str, Any]) -> Any:
        import vane

        await self._await_fragment_registration(request.get("fragment_registration_result"))

        query_task_lease = dict(request.get("query_task_lease") or {})
        leased_node_id = str(query_task_lease.get("node_id") or "").strip()
        if not leased_node_id:
            raise RuntimeError("FTE task lease is missing node_id")
        if leased_node_id != self._node_id:
            raise RuntimeError(
                "FTE task executed outside its query lease: "
                f"expected_node_id={leased_node_id} actual_node_id={self._node_id}"
            )

        context = materialize_task_inputs(
            request.get("context"),
            request.get("initial_splits"),
            merge_scan_task_descriptors=vane.ray_cxx.merge_scan_task_descriptors,
        )

        task_id = FteTaskAttemptId.coerce(request.get("task_id"))
        _maybe_chaos_kill_worker(task_id)
        query_id = str(request.get("query_id") or task_id.query_id or "").strip() or None
        fragment_id = str(request.get("fragment_id", "")).strip()
        if not fragment_id:
            raise ValueError("FTE create_task request requires fragment_id")

        template_plan = self._resolve_fragment_template(
            fragment_id,
            context,
            request.get("fragment_plan"),
            query_id,
        )
        plan = template_plan
        if template_plan.has_root():
            try:
                plan = ray.cloudpickle.loads(ray.cloudpickle.dumps(template_plan))
            except Exception as exc:
                raise RuntimeError(f"Failed to clone PlanFragment {fragment_id}: {exc}") from exc
        result = await self.run_plan_return(
            plan,
            context,
            query_task_lease,
            request.get("exchange_sink_instance"),
            request.get("fte_scan_source_queues"),
            request.get("fte_exchange_source_queues"),
            dynamic_filter_domains=request.get("dynamic_filter_domains"),
            native_progress_callback=request.get("native_progress_callback"),
            debug_context={
                "task_id": str(task_id),
                "query_id": query_id,
                "fragment_id": fragment_id,
                "worker_label": _fte_worker_label(),
            },
        )
        spooling_output_stats = collect_spooling_output_stats(request.get("exchange_sink_instance"))
        if spooling_output_stats is None:
            return result
        return {
            "result": result,
            "output_stats": spooling_output_stats,
            "spooling_output_stats": spooling_output_stats,
        }

    @ray.method(concurrency_group="control")
    async def fte_create_task(self, request: dict[str, Any]) -> dict[str, Any]:
        status = await self._get_fte_task_manager().create_task(request)
        return _fte_applied_control_status(
            "fte_create_task",
            request.get("task_id"),
            status,
        )

    @ray.method(concurrency_group="control")
    async def fte_add_splits(
        self,
        task_id: str | dict[str, Any],
        source_node_id: str,
        splits: list[dict[str, Any]],
        _fte_control_dependency: Any = None,
    ) -> dict[str, Any]:
        status = await self._get_fte_task_manager().add_splits(task_id, source_node_id, splits)
        return _fte_applied_control_status("fte_add_splits", task_id, status)

    @ray.method(concurrency_group="control")
    async def fte_no_more_splits(
        self,
        task_id: str | dict[str, Any],
        source_node_id: str,
        _fte_control_dependency: Any = None,
    ) -> dict[str, Any]:
        status = await self._get_fte_task_manager().no_more_splits(task_id, source_node_id)
        return _fte_applied_control_status("fte_no_more_splits", task_id, status)

    @ray.method(concurrency_group="control")
    async def fte_update_task(
        self,
        task_id: str | dict[str, Any],
        update: dict[str, Any],
        _fte_control_dependency: Any = None,
    ) -> dict[str, Any]:
        status = await self._get_fte_task_manager().update_task(task_id, update)
        return _fte_applied_control_status("fte_update_task", task_id, status)

    @ray.method(concurrency_group="control")
    async def fte_get_task_status(self, task_id: str | dict[str, Any]) -> dict[str, Any]:
        return await self._get_fte_task_manager().get_task_status(task_id)

    @ray.method(concurrency_group="control")
    async def fte_wait_task_status(
        self,
        task_id: str | dict[str, Any],
        min_version: int | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return await self._get_fte_task_manager().wait_task_status(
            task_id,
            min_version=min_version,
            timeout_s=timeout_s,
        )

    @ray.method(concurrency_group="control")
    async def fte_wait_split_queue_has_space(
        self,
        task_id: str | dict[str, Any],
        source_node_id: str | None = None,
        max_buffered_splits: int | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return await self._get_fte_task_manager().wait_split_queue_has_space(
            task_id,
            source_node_id=source_node_id,
            max_buffered_splits=max_buffered_splits,
            timeout_s=timeout_s,
        )

    @ray.method(concurrency_group="control")
    async def fte_get_task_info(self, task_id: str | dict[str, Any]) -> dict[str, Any]:
        return await self._get_fte_task_manager().get_task_info(task_id)

    @ray.method(concurrency_group="control")
    async def fte_ack_task_result(
        self,
        task_id: str | dict[str, Any],
        _fte_control_dependency: Any = None,
    ) -> dict[str, Any]:
        status = self._get_fte_task_manager().ack_task_result(task_id)
        return _fte_applied_control_status("fte_ack_task_result", task_id, status)

    @ray.method(concurrency_group="control")
    async def fte_release_task_result(
        self,
        task_id: str | dict[str, Any],
        _fte_control_dependency: Any = None,
    ) -> dict[str, Any]:
        status = self._get_fte_task_manager().release_task_result(task_id)
        return _fte_applied_control_status("fte_release_task_result", task_id, status)

    @ray.method(concurrency_group="control")
    async def fte_cancel_task(
        self,
        task_id: str | dict[str, Any],
        _fte_control_dependency: Any = None,
    ) -> dict[str, Any]:
        status = await self._get_fte_task_manager().cancel_task(task_id)
        return _fte_applied_control_status("fte_cancel_task", task_id, status)

    @ray.method(concurrency_group="control")
    async def fte_drop_query(self, query_id: str) -> dict[str, int]:
        fte_result = await self._get_fte_task_manager().drop_query(query_id)
        fragments_removed = self.drop_query_fragments(query_id)
        flight_shuffle_cleanup = _cleanup_flight_shuffle_for_query(query_id)
        return {
            "tasks_removed": int(fte_result["removed"]),
            "tasks_canceled": int(fte_result["canceled"]),
            "fragments_removed": int(fragments_removed),
            "flight_shuffle_registry_entries_removed": int(flight_shuffle_cleanup["registry_entries_removed"]),
            "flight_shuffle_storage_entries_removed": int(flight_shuffle_cleanup["storage_entries_removed"]),
            "flight_shuffle_cleanup_errors": int(flight_shuffle_cleanup["cleanup_errors"]),
        }

    def _get_plan_runner(self) -> Any:
        if self._plan_runner is None:
            DistributedPhysicalPlanRunner = require_ray_cxx_attr(
                "DistributedPhysicalPlanRunner",
                hint="Ensure the C++ ray extension is built and importable in this process.",
            )
            self._plan_runner = DistributedPhysicalPlanRunner()
        return self._plan_runner

    def _resolve_fragment_template(
        self,
        fragment_id: str,
        context: dict[str, str] | None,
        fragment_plan: Any | None = None,
        query_id: str | None = None,
    ) -> Any:
        resolved_query_id = str(query_id or "").strip()
        if not resolved_query_id and context:
            resolved_query_id = str(context.get("query_id", "")).strip()
        if not resolved_query_id:
            raise ValueError("fragment template lookup requires non-empty query_id")

        if fragment_id in self._plan_fragments:
            owner_query_id = self._fragment_query_ids.get(fragment_id)
            if owner_query_id != resolved_query_id:
                raise RuntimeError(
                    "fragment template query ownership mismatch: "
                    f"fragment={fragment_id} owner={owner_query_id} "
                    f"requested={resolved_query_id}"
                )
            template_plan = self._plan_fragments[fragment_id]
            self._fragment_lookup_hits += 1
            return template_plan

        if fragment_plan is None:
            self._fragment_lookup_misses += 1
            raise ValueError(f"PlanFragment not found in actor registry: {fragment_id}")

        self._plan_fragments[fragment_id] = fragment_plan
        self._fragment_query_ids[fragment_id] = resolved_query_id
        self._query_fragments.setdefault(resolved_query_id, set()).add(fragment_id)
        self._fragment_lookup_hits += 1
        return fragment_plan

    def _configure_conn(self, conn):
        """Apply standard DuckDB settings (S3, threading, etc.) to a connection."""
        _configure_ray_worker_conn(conn, self._duckdb_memory_bytes)

    def _get_shared_conn(self):
        """Return the shared DuckDB connection, creating it lazily on first use.

        All tasks executed by this actor share the same DatabaseInstance (and
        therefore the same TaskScheduler thread pool).  Individual tasks should
        call ``self._get_shared_conn().cursor()`` to obtain a lightweight cursor
        with its own ClientContext.
        """
        if self._shared_conn is not None:
            return self._shared_conn
        with self._shared_conn_lock:
            if self._shared_conn is not None:
                return self._shared_conn
            import vane

            conn = vane.connect()
            self._configure_conn(conn)
            self._shared_conn = conn
            return conn

    def __del__(self):
        """Cleanup method called when actor is being destroyed."""
        # Use a try-except to safely handle cleanup during Python shutdown
        try:
            import sys

            # Check if Python is shutting down
            if sys.meta_path is None:
                return

            conn = getattr(self, "_shared_conn", None)
            if conn is not None:
                self._shared_conn = None
                try:
                    conn.interrupt()
                except Exception:
                    pass
                try:
                    import time

                    time.sleep(0.5)
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception:
            pass

    def _execute_native_task(
        self,
        plan,
        scan_task_map: dict[str, str] | None,
        copy_output_info: dict[str, str] | None = None,
        exchange_source_task_map: dict[str, Any] | None = None,
        exchange_sink_instance: dict[str, Any] | bytes | None = None,
        fte_scan_source_queues: dict[str, Any] | None = None,
        fte_exchange_source_queues: dict[str, Any] | None = None,
        dynamic_filter_domains: dict[str, Any] | None = None,
        native_progress_callback: Any | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> Any:
        conn = self._get_shared_conn()
        cursor = conn.cursor()
        debug_context = dict(debug_context or {})
        start = time.monotonic()
        _ray_worker_memory_log(
            "native_execute_start",
            **debug_context,
            scan_task_map_count=len(scan_task_map or {}),
            exchange_source_task_map_count=len(exchange_source_task_map or {}),
            has_exchange_sink_instance=exchange_sink_instance is not None,
            has_dynamic_filter_domains=bool(dynamic_filter_domains),
        )

        try:
            plan_runner = self._get_plan_runner()
            scan_task_arg = scan_task_map or None
            result = plan_runner.execute_native(
                cursor,
                plan,
                scan_task_arg,
                exchange_source_task_map or None,
                copy_output_info,
                exchange_sink_instance,
                fte_scan_source_queues,
                fte_exchange_source_queues,
                dynamic_filter_domains or None,
                native_progress_callback,
                debug_context or None,
            )
            _ray_worker_memory_log(
                "native_execute_done",
                **debug_context,
                duration_ms=int((time.monotonic() - start) * 1000),
            )
            return result
        except BaseException as exc:
            _ray_worker_memory_log(
                "native_execute_error",
                **debug_context,
                duration_ms=int((time.monotonic() - start) * 1000),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        finally:
            try:
                cursor.close()
            except Exception:
                pass

    @staticmethod
    async def _await_fragment_registration(registration_result: Any | None) -> None:
        if registration_result is None:
            return
        if isinstance(registration_result, ray.ObjectRef):
            await registration_result

    async def run_plan_return(
        self,
        plan,  # DistributedPhysicalPlan from _vane_duckdb.ray_cxx
        context: dict[str, str] | None,
        query_task_lease: dict[str, Any],
        exchange_sink_instance: dict[str, Any] | bytes | None = None,
        fte_scan_source_queues: dict[str, Any] | None = None,
        fte_exchange_source_queues: dict[str, Any] | None = None,
        dynamic_filter_domains: dict[str, Any] | None = None,
        native_progress_callback: Any | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> Any:
        """Run a plan on worker and return a Ray-serializable result tuple."""
        _apply_env_overrides(self._env_overrides)
        debug_context = dict(debug_context or {})

        copy_output_info = _copy_output_info_from_context(context)
        scan_task_map, exchange_source_task_map = _extract_native_task_maps_from_context(context)
        run_start = time.monotonic()
        _ray_worker_memory_log("run_plan_return_start", **debug_context)
        result_list = await asyncio.to_thread(
            self._execute_native_task,
            plan,
            scan_task_map or None,
            copy_output_info=copy_output_info,
            exchange_source_task_map=exchange_source_task_map or None,
            exchange_sink_instance=exchange_sink_instance,
            fte_scan_source_queues=fte_scan_source_queues,
            fte_exchange_source_queues=fte_exchange_source_queues,
            dynamic_filter_domains=dynamic_filter_domains,
            native_progress_callback=native_progress_callback,
            debug_context=debug_context,
        )
        (
            payloads,
            partition_metadatas,
            result_schema,
            stats_payload,
            _completion_status,
            flight_port,
            native_exchange_sink_instance,
            task_stats,
        ) = _normalize_native_task_result(result_list)
        _ray_worker_memory_log(
            "native_result_normalized",
            **debug_context,
            duration_ms=int((time.monotonic() - run_start) * 1000),
            stats_len=len(stats_payload or []),
            **describe_result_payload(
                (
                    payloads,
                    [(metadata.num_rows, metadata.size_bytes or 0) for metadata in partition_metadatas],
                    result_schema,
                    stats_payload,
                )
            ),
        )
        if exchange_sink_instance is None:
            exchange_sink_instance = native_exchange_sink_instance
        if len(payloads) != len(partition_metadatas):
            raise RuntimeError(
                "execute_native returned mismatched payload/meta lengths: "
                f"payloads={len(payloads)} metas={len(partition_metadatas)}"
            )

        normalized_output_sizes = _validate_fte_output_publication(
            partition_metadatas,
            query_task_lease,
        )

        partition_payloads_for_ray: list[Any] = []
        partition_metas_for_ray: list[tuple[int, int]] = []
        stats_for_ray = _normalize_stats_for_ray(stats_payload)
        ray_put_count = 0

        for payload, metadata, size_bytes in zip(
            payloads,
            partition_metadatas,
            normalized_output_sizes,
            strict=True,
        ):
            obj_ref = payload if isinstance(payload, ray.ObjectRef) else None
            if obj_ref is None:
                obj_ref = ray.put(payload)
                ray_put_count += 1
            partition_payloads_for_ray.append(obj_ref)
            partition_metas_for_ray.append((metadata.num_rows, size_bytes))
        _ray_worker_memory_log(
            "ray_put_done",
            **debug_context,
            duration_ms=int((time.monotonic() - run_start) * 1000),
            ray_put_count=ray_put_count,
            **describe_result_payload(
                (partition_payloads_for_ray, partition_metas_for_ray, result_schema, stats_for_ray)
            ),
        )
        return (
            partition_payloads_for_ray,
            partition_metas_for_ray,
            result_schema,
            stats_for_ray,
            flight_port,
            exchange_sink_instance,
            task_stats,
        )
