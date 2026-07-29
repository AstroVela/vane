# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="untyped-decorator"

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, cast

import ray

from duckdb._ray_cxx import require_ray_cxx_attr

# Avoid importing C++ bindings at module import time (may not be registered yet).
# Resolve `duckdb.ray_cxx` attributes lazily at use-time instead.
from duckdb.event_loop import set_event_loop
from duckdb.runners.common import PartitionMetadata
from duckdb.runners.fte import (
    FteTaskAttemptId,
    FteWorkerTaskManager,
    collect_spooling_output_stats,
    materialize_task_inputs,
    validate_fte_status_identity,
)
from duckdb.runners.fte.debug_memory import describe_result_payload, log_debug, process_memory_snapshot
from duckdb.runners.fte.fte_config import FteWorkerAdmissionConfig
from duckdb.runners.fte.memory_config import apply_duckdb_memory_limit
from duckdb.runners.ray.admission_ledger import BoundedReplayMap
from duckdb.runners.ray.fte_scheduler_config import _fte_control_rpc_timeout_s
from duckdb.runners.ray.ray_env import build_explicit_session_process_env, scrub_shared_runtime_session_env

_SESSION_CLOSE_REPLAY_CAPACITY = 65_536
_AWS_CREDENTIAL_RESOLVER_TIMEOUT_S = 120
_AWS_CREDENTIAL_REFRESH_AT_KEY = "__vane_aws_credential_refresh_at_epoch_s"
_AWS_CREDENTIAL_PROVIDER_KEYS = (
    "AWS_CONFIG_FILE",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_DEFAULT_PROFILE",
    "AWS_PROFILE",
    "AWS_ROLE_ARN",
    "AWS_ROLE_SESSION_NAME",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
)
_DUCKDB_S3_SESSION_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_DEFAULT_REGION",
    "AWS_ENDPOINT_URL",
    "AWS_REGION",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
)


async def _to_thread_with_owned_side_effects(
    callback: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
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
    import duckdb.datasource  # noqa: F401


def _release_datasource_factories_for_query(query_id: str) -> int:
    import _duckdb  # type: ignore[import-not-found]

    return int(_duckdb._release_datasource_factories_for_query(str(query_id)))


def _clear_datasource_factory_registry() -> None:
    import _duckdb

    _duckdb._clear_datasource_factory_registry()


def _register_query_python_replay_state(query_id: str, plan: Any) -> bool:
    register = require_ray_cxx_attr(
        "_register_query_python_replay_state",
        hint="Ensure the C++ ray extension is built with query replay lifecycle support.",
    )
    return bool(register(str(query_id), plan))


def _plan_resource_query_id(plan: Any) -> str:
    resource_query_id = getattr(plan, "resource_query_id", None)
    if not callable(resource_query_id):
        raise TypeError("distributed physical plan is missing resource_query_id()")
    query_id = str(resource_query_id()).strip()
    if not query_id:
        raise ValueError("distributed physical plan resource_query_id must not be empty")
    return query_id


def _cleanup_query_python_replay_state(query_id: str) -> None:
    cleanup = require_ray_cxx_attr(
        "_cleanup_query_python_replay_state",
        hint="Ensure the C++ ray extension is built with query replay lifecycle support.",
    )
    cleanup(str(query_id))


def _cleanup_flight_shuffle_for_query(query_id: str) -> dict[str, Any]:
    query_id = str(query_id or "").strip()
    if not query_id:
        return {
            "registry_entries_removed": 0,
            "storage_entries_removed": 0,
            "cleanup_errors": 0,
            "cleanup_pending": 0,
            "active_executions": 0,
            "last_error": "",
        }
    cleanup_fn = require_ray_cxx_attr(
        "cleanup_flight_shuffle_for_query",
        hint="Ensure the C++ ray extension is built with Flight shuffle cleanup support.",
    )
    raw = cleanup_fn(query_id)
    if not isinstance(raw, dict):
        raise TypeError("Flight shuffle cleanup binding must return a dict")
    return {
        "registry_entries_removed": int(raw.get("registry_entries_removed", 0)),
        "storage_entries_removed": int(raw.get("storage_entries_removed", 0)),
        "cleanup_errors": int(raw.get("cleanup_errors", 0)),
        "cleanup_pending": int(raw.get("cleanup_pending", 0)),
        "active_executions": int(raw.get("active_executions", 0)),
        "last_error": str(raw.get("last_error", "")),
    }


def _close_flight_shuffle_query(query_id: str) -> None:
    close_fn = require_ray_cxx_attr(
        "close_flight_shuffle_query",
        hint="Ensure the C++ ray extension is built with Flight shuffle query fencing support.",
    )
    close_fn(str(query_id))


def _flight_shuffle_query_status(query_id: str) -> dict[str, int]:
    status_fn = require_ray_cxx_attr(
        "flight_shuffle_query_status",
        hint="Ensure the C++ ray extension is built with Flight shuffle query fencing support.",
    )
    raw = status_fn(str(query_id))
    if not isinstance(raw, dict):
        raise TypeError("Flight shuffle query status binding must return a dict")
    return {
        "cleanup_pending": int(raw.get("cleanup_pending", 0)),
        "active_executions": int(raw.get("active_executions", 0)),
    }


async def _wait_flight_shuffle_executions_for_query(query_id: str, *, timeout_s: float | None = None) -> None:
    if timeout_s is None:
        timeout_s = _fte_control_rpc_timeout_s() * 0.8
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    backoff_s = 0.01
    while True:
        status = await asyncio.to_thread(_flight_shuffle_query_status, query_id)
        if status["active_executions"] == 0:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "timed out waiting for Flight shuffle native executions "
                f"for {query_id}: active_executions={status['active_executions']}"
            )
        await asyncio.sleep(backoff_s)
        backoff_s = min(backoff_s * 2, 0.5)


def _retire_flight_shuffle_query(query_id: str) -> None:
    retire_fn = require_ray_cxx_attr(
        "retire_flight_shuffle_query",
        hint="Ensure the C++ ray extension is built with Flight shuffle query retirement support.",
    )
    retire_fn(str(query_id))


async def _drain_flight_shuffle_for_query(
    query_id: str,
    *,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    if timeout_s is None:
        timeout_s = _fte_control_rpc_timeout_s() * 0.8
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    total_registry_removed = 0
    total_storage_removed = 0
    backoff_s = 0.01
    while True:
        cleanup = await asyncio.to_thread(_cleanup_flight_shuffle_for_query, query_id)
        total_registry_removed += cleanup["registry_entries_removed"]
        total_storage_removed += cleanup["storage_entries_removed"]
        if cleanup["active_executions"] == 0 and cleanup["cleanup_pending"] == 0 and cleanup["cleanup_errors"] == 0:
            cleanup["registry_entries_removed"] = total_registry_removed
            cleanup["storage_entries_removed"] = total_storage_removed
            return cleanup
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "timed out draining Flight shuffle query "
                f"{query_id}: active_executions={cleanup['active_executions']} "
                f"cleanup_pending={cleanup['cleanup_pending']} "
                f"cleanup_errors={cleanup['cleanup_errors']} "
                f"last_error={cleanup['last_error']!r}"
            )
        await asyncio.sleep(backoff_s)
        backoff_s = min(backoff_s * 2, 0.5)


def _normalize_native_task_result(result: Any) -> tuple[Any, ...]:
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


def _resolve_session_aws_credentials(config: Mapping[str, str]) -> tuple[dict[str, str], float | None]:
    """Resolve one session credential chain without mutating shared process state."""
    environment = build_explicit_session_process_env(config)
    environment["VANE_RUNNER"] = "local"
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "duckdb.runners.ray.aws_credentials"],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
            timeout=_AWS_CREDENTIAL_RESOLVER_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("DuckDB AWS session credential resolver timed out") from exc
    if completed.returncode != 0:
        raise RuntimeError(f"DuckDB AWS session credential resolver failed with exit code {completed.returncode}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("DuckDB AWS session credential resolver returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("DuckDB AWS session credential resolver returned a non-mapping payload")
    raw_credentials = payload.get("credentials")
    if not isinstance(raw_credentials, dict):
        raise RuntimeError("DuckDB AWS session credential resolver returned invalid credentials")
    credentials = {
        str(key): str(value)
        for key, value in raw_credentials.items()
        if str(key) in _DUCKDB_S3_SESSION_KEYS and value is not None
    }
    if not credentials.get("AWS_ACCESS_KEY_ID") or not credentials.get("AWS_SECRET_ACCESS_KEY"):
        raise RuntimeError("DuckDB AWS session credential resolver returned incomplete credentials")
    expiration_epoch_s = None
    raw_expiration = payload.get("expiration_epoch_s")
    if raw_expiration is not None:
        try:
            expiration_epoch_s = float(raw_expiration)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("DuckDB AWS session credential resolver returned an invalid expiration") from exc
        if not math.isfinite(expiration_epoch_s):
            raise RuntimeError("DuckDB AWS session credential resolver returned a non-finite expiration")
    return credentials, expiration_epoch_s


def _effective_duckdb_s3_config(
    config: Mapping[str, str],
    *,
    use_session_credentials: bool = True,
) -> dict[str, str]:
    normalized = {str(key): str(value) for key, value in config.items()}
    effective = {key: normalized[key] for key in _DUCKDB_S3_SESSION_KEYS if normalized.get(key, "").strip()}
    if not use_session_credentials:
        return {
            key: value
            for key, value in effective.items()
            if key not in {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"}
        }
    refresh_at = normalized.get(_AWS_CREDENTIAL_REFRESH_AT_KEY, "").strip()
    if refresh_at:
        effective[_AWS_CREDENTIAL_REFRESH_AT_KEY] = refresh_at
    access_key = effective.get("AWS_ACCESS_KEY_ID")
    secret_key = effective.get("AWS_SECRET_ACCESS_KEY")
    session_token = effective.get("AWS_SESSION_TOKEN")
    has_static_credentials = bool(access_key and secret_key)
    if bool(access_key) != bool(secret_key) or (session_token and not has_static_credentials):
        raise ValueError("AWS static session credentials must provide both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
    needs_credential_chain = any(normalized.get(key, "").strip() for key in _AWS_CREDENTIAL_PROVIDER_KEYS)
    if not has_static_credentials and needs_credential_chain:
        resolved, expiration_epoch_s = _resolve_session_aws_credentials(normalized)
        resolved.update(effective)
        effective = resolved
        if expiration_epoch_s is not None:
            resolved_at = time.time()
            refresh_at_epoch_s = resolved_at + max(0.0, expiration_epoch_s - resolved_at) * 0.8
            effective[_AWS_CREDENTIAL_REFRESH_AT_KEY] = repr(refresh_at_epoch_s)
    return effective


def _refresh_effective_duckdb_s3_config(
    config: Mapping[str, str],
    effective_config: Mapping[str, str],
    *,
    use_session_credentials: bool = True,
) -> dict[str, str]:
    normalized = {str(key): str(value) for key, value in config.items()}
    if not use_session_credentials:
        return _effective_duckdb_s3_config(
            normalized,
            use_session_credentials=False,
        )
    has_static_credentials = bool(
        normalized.get("AWS_ACCESS_KEY_ID", "").strip() and normalized.get("AWS_SECRET_ACCESS_KEY", "").strip()
    )
    needs_credential_chain = any(normalized.get(key, "").strip() for key in _AWS_CREDENTIAL_PROVIDER_KEYS)
    current = {str(key): str(value) for key, value in effective_config.items()}
    if has_static_credentials or not needs_credential_chain:
        return _effective_duckdb_s3_config(normalized)
    refresh_at = current.get(_AWS_CREDENTIAL_REFRESH_AT_KEY)
    if refresh_at is None:
        if current.get("AWS_ACCESS_KEY_ID") and current.get("AWS_SECRET_ACCESS_KEY"):
            return current
        return _effective_duckdb_s3_config(normalized)
    try:
        refresh_at_epoch_s = float(refresh_at)
    except ValueError:
        return _effective_duckdb_s3_config(normalized)
    if math.isfinite(refresh_at_epoch_s) and time.time() < refresh_at_epoch_s:
        return current
    return _effective_duckdb_s3_config(normalized)


def _configure_duckdb_s3(
    conn: Any,
    config: Mapping[str, str],
    *,
    use_session_credentials: bool = True,
) -> dict[str, str]:
    """Configure one DuckDB context from explicit session AWS settings.

    Shared driver/worker processes must not read session credentials from their
    process environment. The caller owns the immutable connection-session
    snapshot.
    """
    from urllib.parse import urlparse

    effective_config = _effective_duckdb_s3_config(
        config,
        use_session_credentials=use_session_credentials,
    )
    endpoint_url = str(effective_config.get("AWS_ENDPOINT_URL", "")).strip()
    access_key = str(effective_config.get("AWS_ACCESS_KEY_ID", "")).strip()
    secret_key = str(effective_config.get("AWS_SECRET_ACCESS_KEY", "")).strip()
    session_token = str(effective_config.get("AWS_SESSION_TOKEN", "")).strip()
    region = str(effective_config.get("AWS_REGION") or effective_config.get("AWS_DEFAULT_REGION") or "").strip()

    if use_session_credentials and not any((endpoint_url, access_key, secret_key, session_token, region)):
        return effective_config

    try:
        conn.execute("LOAD httpfs")
    except Exception as exc:
        raise RuntimeError(
            "Ray S3 configuration requires the statically linked httpfs extension; "
            "runtime extension installation is disabled"
        ) from exc

    def _q(s: str) -> str:
        return s.replace("'", "''")

    if region:
        conn.execute(f"SET s3_region='{_q(region)}'")
    if use_session_credentials:
        if access_key:
            conn.execute(f"SET s3_access_key_id='{_q(access_key)}'")
        if secret_key:
            conn.execute(f"SET s3_secret_access_key='{_q(secret_key)}'")
    else:
        conn.execute("SET s3_access_key_id=''")
        conn.execute("SET s3_secret_access_key=''")
    # Task cursors inherit settings from the long-lived session connection.
    # Always overwrite the token so a refresh from temporary credentials to
    # credentials without a token cannot retain the previous value.
    conn.execute(f"SET s3_session_token='{_q(session_token)}'")
    if endpoint_url:
        parse_target = endpoint_url
        if "://" not in parse_target and not parse_target.startswith("//"):
            parse_target = f"//{parse_target}"
        parsed = urlparse(parse_target)
        endpoint = parsed.netloc or parsed.path
        use_ssl = parsed.scheme == "https"
        conn.execute(f"SET s3_endpoint='{_q(endpoint)}'")
        conn.execute(f"SET s3_use_ssl={'true' if use_ssl else 'false'}")
        conn.execute("SET s3_url_style='path'")

    # Keep-alive MUST stay enabled: disabling it creates a new TCP connection
    # per S3 request, which exhausts ephemeral ports via TIME_WAIT buildup
    # (55K+ TIME_WAIT sockets observed with keep_alive=false).
    conn.execute("SET http_keep_alive=true")
    # Increase retries to handle transient connection failures during
    # concurrent S3 access from many DuckDB threads.
    conn.execute("SET http_retries=10")
    conn.execute("SET http_retry_wait_ms=100")
    conn.execute("SET http_retry_backoff=1.5")
    return effective_config


def _configure_ray_worker_conn(conn: Any, duckdb_memory_bytes: int) -> None:
    apply_duckdb_memory_limit(conn, duckdb_memory_bytes)
    duckdb_threads = os.environ.get("VANE_DUCKDB_THREADS")
    if duckdb_threads:
        try:
            conn.execute(f"SET threads={int(duckdb_threads)}")
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
        ray_node_ip_address: str = "",
    ) -> None:
        scrub_shared_runtime_session_env()
        ray_node_ip_address = str(ray_node_ip_address or "").strip()
        if ray_node_ip_address and not os.environ.get("VANE_FLIGHT_ADVERTISE_HOST", "").strip():
            os.environ["VANE_FLIGHT_ADVERTISE_HOST"] = ray_node_ip_address
        duckdb_memory_bytes = int(duckdb_memory_bytes)
        task_heap_capacity_bytes = int(task_heap_capacity_bytes)
        if duckdb_memory_bytes <= 0:
            raise ValueError("Ray worker duckdb_memory_bytes must be positive")
        if task_heap_capacity_bytes <= 0:
            raise ValueError("Ray worker task_heap_capacity_bytes must be positive")
        self._node_id = str(ray.get_runtime_context().get_node_id() or "").strip()
        if not self._node_id:
            raise RuntimeError("Ray worker runtime context is missing node_id")
        self._duckdb_memory_bytes = duckdb_memory_bytes
        self._task_heap_capacity_bytes = task_heap_capacity_bytes
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
        self._shared_conn: Any | None = None
        self._shared_conn_lock = threading.Lock()
        self._session_connections: dict[str, tuple[dict[str, str], Any]] = {}
        self._session_s3_configs: dict[str, dict[str, str]] = {}
        self._session_operation_locks: dict[str, threading.Lock] = {}
        self._closed_session_ids = BoundedReplayMap[str, bool](capacity=_SESSION_CLOSE_REPLAY_CAPACITY)
        self._session_connections_lock = threading.RLock()
        self._shutdown_lock = threading.RLock()
        self._native_execution_condition = threading.Condition()
        self._native_execution_count = 0
        self._native_execution_counts_by_query: dict[str, int] = {}
        self._active_native_cursors: set[Any] = set()
        self._native_cursor_query_ids: dict[Any, str] = {}
        self._closing_native_queries: set[str] = set()
        self._shutdown_started = False
        self._shutdown_prepared = False
        self._shutdown_complete = False
        self._get_shared_conn()  # eagerly initialize
        _ray_worker_memory_log(
            "actor_initialized",
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            worker_label=_fte_worker_label(),
        )

    def _ensure_worker_runtime_running(self) -> None:
        shutdown_lock = getattr(self, "_shutdown_lock", None)
        if shutdown_lock is None:
            if getattr(self, "_shutdown_started", False):
                raise RuntimeError("Ray worker runtime is shutting down")
            return
        with shutdown_lock:
            if getattr(self, "_shutdown_started", False):
                raise RuntimeError("Ray worker runtime is shutting down")

    @ray.method(concurrency_group="control")
    def ping(self) -> bool:
        self._ensure_worker_runtime_running()
        return True

    @ray.method(concurrency_group="control")
    async def register_fragments(self, fragments: list[dict[str, Any]]) -> dict[str, int]:
        """Register plan fragments in this actor.

        Each entry is expected to contain:
        - fragment_id: stable fragment id string
        - plan: plan object (PhysicalPlan / DistributedPhysicalPlan wrapper)
        - query_id: query identity for lifecycle cleanup
        """
        self._ensure_worker_runtime_running()
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

        self._ensure_worker_runtime_running()
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
            query_id = str(entry.get("query_id", "")).strip()
            _register_query_python_replay_state(_plan_resource_query_id(plan), plan)
            self._plan_fragments[fragment_id] = plan
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
        self._ensure_worker_runtime_running()
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
        shutdown_lock = getattr(self, "_shutdown_lock", None)
        if shutdown_lock is None:
            shutdown_lock = threading.RLock()
            self._shutdown_lock = shutdown_lock
        with shutdown_lock:
            if getattr(self, "_shutdown_started", False):
                raise RuntimeError("Ray worker runtime is shutting down")
            if self._fte_task_manager is None:
                self._fte_task_manager = FteWorkerTaskManager(
                    self._execute_fte_request,
                    admission_config=self._fte_admission_config,
                    require_query_task_lease=True,
                    worker_label=_fte_worker_label(),
                )
            return self._fte_task_manager

    async def _execute_fte_request(self, request: dict[str, Any]) -> Any:
        import duckdb

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
            merge_scan_task_descriptors=duckdb.ray_cxx.merge_scan_task_descriptors,
        )

        task_id = FteTaskAttemptId.coerce(request.get("task_id"))
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
            cast(str | dict[str, Any], request.get("task_id")),
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
    async def fte_prepare_drop_query(self, query_id: str) -> dict[str, int]:
        _close_flight_shuffle_query(query_id)
        interrupt_errors = self._close_worker_native_query(query_id)
        try:
            fte_result = await self._get_fte_task_manager().drop_query(query_id)
            fragments_removed = self.drop_query_fragments(query_id)
        finally:
            native_drain_result, flight_drain_result = await asyncio.gather(
                self._wait_worker_native_executions_for_query(query_id),
                _wait_flight_shuffle_executions_for_query(query_id),
                return_exceptions=True,
            )
            drain_errors = [
                result for result in (native_drain_result, flight_drain_result) if isinstance(result, BaseException)
            ]
            if not isinstance(native_drain_result, BaseException):
                try:
                    _release_datasource_factories_for_query(query_id)
                except Exception as error:
                    drain_errors.append(error)
            if drain_errors:
                details = "; ".join(f"{type(error).__name__}: {error}" for error in drain_errors)
                raise RuntimeError(f"failed to prepare query teardown for {query_id}: {details}") from drain_errors[0]
        if interrupt_errors:
            raise RuntimeError(
                f"failed to interrupt {len(interrupt_errors)} native execution(s) for {query_id}: "
                + "; ".join(interrupt_errors)
            )
        return {
            "tasks_removed": int(fte_result["removed"]),
            "tasks_canceled": int(fte_result["canceled"]),
            "fragments_removed": int(fragments_removed),
        }

    @ray.method(concurrency_group="control")
    async def fte_cleanup_query(self, query_id: str) -> dict[str, int]:
        flight_shuffle_cleanup = await _drain_flight_shuffle_for_query(query_id)
        _retire_flight_shuffle_query(query_id)
        self._retire_worker_native_query(query_id)
        _cleanup_query_python_replay_state(query_id)
        return {
            "flight_shuffle_registry_entries_removed": int(flight_shuffle_cleanup["registry_entries_removed"]),
            "flight_shuffle_storage_entries_removed": int(flight_shuffle_cleanup["storage_entries_removed"]),
            "flight_shuffle_cleanup_errors": int(flight_shuffle_cleanup["cleanup_errors"]),
        }

    @ray.method(concurrency_group="control")
    async def fte_drop_query(self, query_id: str) -> dict[str, int]:
        result = await self.fte_prepare_drop_query(query_id)
        result.update(await self.fte_cleanup_query(query_id))
        return result

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

        _register_query_python_replay_state(_plan_resource_query_id(fragment_plan), fragment_plan)
        self._plan_fragments[fragment_id] = fragment_plan
        self._fragment_query_ids[fragment_id] = resolved_query_id
        self._query_fragments.setdefault(resolved_query_id, set()).add(fragment_id)
        self._fragment_lookup_hits += 1
        return fragment_plan

    def _configure_conn(self, conn: Any) -> None:
        """Apply standard DuckDB settings (S3, threading, etc.) to a connection."""
        _configure_ray_worker_conn(conn, self._duckdb_memory_bytes)

    def _get_shared_conn(self) -> Any:
        """Return the shared DuckDB connection, creating it lazily on first use.

        All tasks executed by this actor share the same DatabaseInstance (and
        therefore the same TaskScheduler thread pool).  Individual tasks should
        call ``self._get_shared_conn().cursor()`` to obtain a lightweight cursor
        with its own ClientContext.
        """
        with self._shared_conn_lock:
            if self._shutdown_started:
                raise RuntimeError("Ray worker runtime is shut down")
            if self._shared_conn is not None:
                return self._shared_conn
            import duckdb

            conn = duckdb.connect()
            self._configure_conn(conn)
            self._shared_conn = conn
            return conn

    def _get_session_operation_lock(
        self,
        session_id: str,
        *,
        allow_closed: bool = False,
    ) -> threading.Lock:
        with self._session_connections_lock:
            if not allow_closed:
                if self._shutdown_started:
                    raise RuntimeError("Ray worker runtime is shutting down")
                if session_id in self._closed_session_ids:
                    raise RuntimeError(f"Ray worker Vane session is closed: {session_id}")
            operation_locks = getattr(self, "_session_operation_locks", None)
            if operation_locks is None:
                operation_locks = {}
                self._session_operation_locks = operation_locks
            operation_lock = operation_locks.get(session_id)
            if operation_lock is None:
                operation_lock = threading.Lock()
                operation_locks[session_id] = operation_lock
            return operation_lock

    def _get_session_conn(
        self,
        session_id: str,
        config: Mapping[str, str],
        *,
        use_session_credentials: bool = True,
    ) -> Any:
        session_key = str(session_id).strip()
        if not session_key:
            raise ValueError("Ray worker execution requires a Vane session_id")
        normalized_config = {str(key): str(value) for key, value in config.items()}
        operation_lock = self._get_session_operation_lock(session_key)
        with operation_lock:
            return self._get_session_conn_locked(
                session_key,
                normalized_config,
                use_session_credentials=use_session_credentials,
            )

    def _get_session_conn_locked(
        self,
        session_key: str,
        normalized_config: dict[str, str],
        *,
        use_session_credentials: bool,
    ) -> Any:
        with self._session_connections_lock:
            session_s3_configs = getattr(self, "_session_s3_configs", None)
            if session_s3_configs is None:
                session_s3_configs = {}
                self._session_s3_configs = session_s3_configs
            if self._shutdown_started:
                raise RuntimeError("Ray worker runtime is shutting down")
            if session_key in self._closed_session_ids:
                raise RuntimeError(f"Ray worker Vane session is closed: {session_key}")
            existing = self._session_connections.get(session_key)
            if existing is not None:
                existing_config, connection = existing
                if existing_config != normalized_config:
                    raise RuntimeError(f"Ray worker Vane session config changed: {session_key}")
                return connection

        connection = self._get_shared_conn().cursor()
        try:
            effective_s3_config = _configure_duckdb_s3(
                connection,
                normalized_config,
                use_session_credentials=use_session_credentials,
            )
        except BaseException as config_error:
            try:
                connection.close()
            except BaseException as close_error:
                raise RuntimeError(
                    f"Ray worker Vane session {session_key} configuration failed and "
                    f"its connection could not be closed: {type(close_error).__name__}: {close_error}"
                ) from config_error
            raise

        try:
            with self._session_connections_lock:
                if self._shutdown_started:
                    raise RuntimeError("Ray worker runtime is shutting down")
                if session_key in self._closed_session_ids:
                    raise RuntimeError(f"Ray worker Vane session is closed: {session_key}")
                existing = self._session_connections.get(session_key)
                if existing is not None:
                    raise RuntimeError(f"Ray worker Vane session opened outside its operation lock: {session_key}")
                self._session_connections[session_key] = (normalized_config, connection)
                session_s3_configs[session_key] = effective_s3_config
                return connection
        except BaseException as state_error:
            try:
                connection.close()
            except BaseException as close_error:
                raise RuntimeError(
                    f"Ray worker Vane session {session_key} changed while opening and "
                    f"its connection could not be closed: {type(close_error).__name__}: {close_error}"
                ) from state_error
            raise

    def _refresh_session_s3_config(
        self,
        session_id: str,
        session_config: dict[str, str],
        connection: Any,
        *,
        use_session_credentials: bool,
    ) -> dict[str, str]:
        operation_lock = self._get_session_operation_lock(session_id)
        with operation_lock:
            with self._session_connections_lock:
                if self._shutdown_started:
                    raise RuntimeError("Ray worker runtime is shutting down")
                if session_id in self._closed_session_ids:
                    raise RuntimeError(f"Ray worker Vane session is closed: {session_id}")
                record = self._session_connections.get(session_id)
                if record is None or record[1] is not connection:
                    raise RuntimeError(f"Ray worker Vane session closed during task startup: {session_id}")
                effective_s3_config = dict(self._session_s3_configs.get(session_id, session_config))
            effective_s3_config = _refresh_effective_duckdb_s3_config(
                session_config,
                effective_s3_config,
                use_session_credentials=use_session_credentials,
            )
            with self._session_connections_lock:
                if self._shutdown_started:
                    raise RuntimeError("Ray worker runtime is shutting down")
                if session_id in self._closed_session_ids:
                    raise RuntimeError(f"Ray worker Vane session is closed: {session_id}")
                record = self._session_connections.get(session_id)
                if record is None or record[1] is not connection:
                    raise RuntimeError(f"Ray worker Vane session closed during task startup: {session_id}")
                self._session_s3_configs[session_id] = effective_s3_config
            return effective_s3_config

    @ray.method(concurrency_group="control")
    async def close_session(self, session_id: str) -> None:
        session_key = str(session_id).strip()
        if not session_key:
            raise ValueError("Ray worker close_session requires a Vane session_id")

        def _close() -> None:
            operation_lock = self._get_session_operation_lock(session_key, allow_closed=True)
            with self._session_connections_lock:
                self._closed_session_ids[session_key] = True
            with operation_lock:
                with self._session_connections_lock:
                    record = self._session_connections.get(session_key)
                if record is None:
                    with self._session_connections_lock:
                        operation_locks = getattr(self, "_session_operation_locks", {})
                        if operation_locks.get(session_key) is operation_lock:
                            operation_locks.pop(session_key, None)
                    return
                _, connection = record
                connection.close()
                with self._session_connections_lock:
                    if self._session_connections.get(session_key) is record:
                        self._session_connections.pop(session_key, None)
                        getattr(self, "_session_s3_configs", {}).pop(session_key, None)
                        getattr(self, "_session_operation_locks", {}).pop(session_key, None)

        await _to_thread_with_owned_side_effects(_close)

    def _begin_worker_native_execution(self, query_id: str) -> None:
        query_id = str(query_id or "").strip()
        if not query_id:
            raise ValueError("native execution admission requires a query_id")
        with self._native_execution_condition:
            if self._shutdown_started:
                raise RuntimeError("Ray worker runtime is shutting down")
            if query_id in self._closing_native_queries:
                raise RuntimeError(f"native query is closing: {query_id}")
            self._native_execution_count += 1
            self._native_execution_counts_by_query[query_id] = (
                self._native_execution_counts_by_query.get(query_id, 0) + 1
            )

    def _end_worker_native_execution(self, query_id: str) -> None:
        query_id = str(query_id or "").strip()
        with self._native_execution_condition:
            if self._native_execution_count <= 0:
                raise RuntimeError("Ray worker native execution ownership underflow")
            query_count = self._native_execution_counts_by_query.get(query_id, 0)
            if query_count <= 0:
                raise RuntimeError(f"Ray worker native query execution ownership underflow: {query_id}")
            self._native_execution_count -= 1
            if query_count == 1:
                self._native_execution_counts_by_query.pop(query_id, None)
            else:
                self._native_execution_counts_by_query[query_id] = query_count - 1
            self._native_execution_condition.notify_all()

    def _register_native_cursor(self, cursor: Any, query_id: str = "") -> bool:
        with self._native_execution_condition:
            self._active_native_cursors.add(cursor)
            self._native_cursor_query_ids[cursor] = str(query_id)
            return str(query_id) not in self._closing_native_queries

    def _unregister_native_cursor(self, cursor: Any) -> None:
        with self._native_execution_condition:
            self._active_native_cursors.discard(cursor)
            self._native_cursor_query_ids.pop(cursor, None)
            self._native_execution_condition.notify_all()

    def _worker_native_query_is_closing(self, query_id: str) -> bool:
        with self._native_execution_condition:
            return str(query_id) in self._closing_native_queries

    async def _wait_worker_native_executions_for_query(
        self,
        query_id: str,
        *,
        timeout_s: float | None = None,
    ) -> None:
        query_id = str(query_id or "").strip()
        if timeout_s is None:
            timeout_s = _fte_control_rpc_timeout_s() * 0.8
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        backoff_s = 0.01
        while True:
            with self._native_execution_condition:
                active_executions = self._native_execution_counts_by_query.get(query_id, 0)
            if active_executions == 0:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "timed out waiting for Ray worker native executions "
                    f"for {query_id}: active_executions={active_executions}"
                )
            await asyncio.sleep(backoff_s)
            backoff_s = min(backoff_s * 2, 0.5)

    def _close_worker_native_query(self, query_id: str) -> list[str]:
        query_id = str(query_id)
        with self._native_execution_condition:
            self._closing_native_queries.add(query_id)
            cursors = [
                cursor
                for cursor, cursor_query_id in self._native_cursor_query_ids.items()
                if cursor_query_id == query_id
            ]
        errors: list[str] = []
        for cursor in cursors:
            try:
                cursor.interrupt()
            except Exception as exc:
                errors.append(str(exc))
        return errors

    def _retire_worker_native_query(self, query_id: str) -> None:
        with self._native_execution_condition:
            query_id = str(query_id)
            active_executions = self._native_execution_counts_by_query.get(query_id, 0)
            if active_executions:
                raise RuntimeError(
                    "cannot retire Ray worker native query with active executions: "
                    f"{query_id} active_executions={active_executions}"
                )
            self._closing_native_queries.discard(query_id)

    def _prepare_worker_runtime_shutdown(self) -> None:
        shutdown_lock = getattr(self, "_shutdown_lock", None)
        if shutdown_lock is None:
            shutdown_lock = threading.RLock()
            self._shutdown_lock = shutdown_lock
        with shutdown_lock:
            if getattr(self, "_shutdown_prepared", False) or getattr(self, "_shutdown_complete", False):
                return
            errors: list[str] = []
            native_condition = getattr(self, "_native_execution_condition", None)
            if native_condition is None:
                native_condition = threading.Condition()
                self._native_execution_condition = native_condition
                self._native_execution_count = 0
                self._native_execution_counts_by_query = {}
                self._active_native_cursors = set()
                self._native_cursor_query_ids = {}
                self._closing_native_queries = set()
            with native_condition:
                self._shutdown_started = True
            task_manager = getattr(self, "_fte_task_manager", None)
            if task_manager is not None:
                try:
                    task_manager.shutdown()
                except Exception as exc:
                    errors.append(f"cancel FTE tasks: {exc}")
            shared_conn_lock = getattr(self, "_shared_conn_lock", None)
            if shared_conn_lock is None:
                shared_conn_lock = threading.Lock()
                self._shared_conn_lock = shared_conn_lock
            with shared_conn_lock:
                conn = getattr(self, "_shared_conn", None)
            session_connections_lock = getattr(self, "_session_connections_lock", None)
            if session_connections_lock is None:
                session_connections_lock = threading.RLock()
                self._session_connections_lock = session_connections_lock
            with session_connections_lock:
                session_connections = list(getattr(self, "_session_connections", {}).items())
            if conn is not None:
                try:
                    conn.interrupt()
                except Exception as exc:
                    errors.append(f"interrupt DuckDB connection: {exc}")
            deadline = time.monotonic() + 25.0
            cursor_interrupt_errors: set[str] = set()
            while True:
                with native_condition:
                    active_executions = int(getattr(self, "_native_execution_count", 0))
                    active_cursors = list(getattr(self, "_active_native_cursors", ()))
                if active_executions == 0:
                    break
                for cursor in active_cursors:
                    try:
                        cursor.interrupt()
                    except Exception as exc:
                        message = f"interrupt active DuckDB cursor: {exc}"
                        if message not in cursor_interrupt_errors:
                            cursor_interrupt_errors.add(message)
                            errors.append(message)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    errors.append(f"timed out waiting for {active_executions} native execution(s) to stop")
                    break
                with native_condition:
                    if self._native_execution_count > 0:
                        native_condition.wait(timeout=min(0.1, remaining))
            with native_condition:
                native_drained = self._native_execution_count == 0
            if not native_drained:
                raise RuntimeError("; ".join(errors))
            for session_id, record in session_connections:
                with session_connections_lock:
                    if getattr(self, "_session_connections", {}).get(session_id) is not record:
                        continue
                    if record is None:
                        continue
                    _, session_connection = record
                    try:
                        session_connection.close()
                    except Exception as exc:
                        errors.append(f"close DuckDB session connection {session_id}: {exc}")
                    else:
                        if self._session_connections.get(session_id) is record:
                            self._session_connections.pop(session_id, None)
                            getattr(self, "_session_s3_configs", {}).pop(session_id, None)
            with shared_conn_lock:
                conn = getattr(self, "_shared_conn", None)
            if conn is not None:
                try:
                    conn.close()
                except Exception as exc:
                    errors.append(f"close DuckDB connection: {exc}")
                else:
                    with shared_conn_lock:
                        if getattr(self, "_shared_conn", None) is conn:
                            self._shared_conn = None
            if errors:
                raise RuntimeError("; ".join(errors))
            self._shutdown_prepared = True

    def _finish_worker_runtime_shutdown(self) -> None:
        shutdown_lock = getattr(self, "_shutdown_lock", None)
        if shutdown_lock is None:
            shutdown_lock = threading.RLock()
            self._shutdown_lock = shutdown_lock
        with shutdown_lock:
            if getattr(self, "_shutdown_complete", False):
                return
            if not getattr(self, "_shutdown_prepared", False):
                raise RuntimeError("Ray worker runtime shutdown was not prepared")
            shutdown_flight = require_ray_cxx_attr(
                "shutdown_local_flight_service",
                hint="Ensure the C++ ray extension is built with Flight service lifecycle support.",
            )
            shutdown_flight()
            _clear_datasource_factory_registry()
            self._shutdown_complete = True

    @ray.method(concurrency_group="control")
    def prepare_shutdown(self) -> None:
        self._prepare_worker_runtime_shutdown()

    @ray.method(concurrency_group="control")
    def finish_shutdown(self) -> None:
        self._finish_worker_runtime_shutdown()

    def _shutdown_worker_runtime(self) -> None:
        self._prepare_worker_runtime_shutdown()
        self._finish_worker_runtime_shutdown()

    def __del__(self) -> None:
        """Cleanup method called when actor is being destroyed."""
        try:
            if sys.meta_path is None:
                return  # type: ignore[unreachable]
            self._shutdown_worker_runtime()
        except Exception:
            pass

    def _execute_native_task(
        self,
        plan: Any,
        scan_task_map: dict[str, str] | None,
        copy_output_info: dict[str, str] | None = None,
        exchange_source_task_map: dict[str, Any] | None = None,
        exchange_sink_instance: dict[str, Any] | bytes | None = None,
        fte_scan_source_queues: dict[str, Any] | None = None,
        fte_exchange_source_queues: dict[str, Any] | None = None,
        dynamic_filter_domains: dict[str, Any] | None = None,
        native_progress_callback: Any | None = None,
        debug_context: dict[str, Any] | None = None,
        native_query_id: str = "",
    ) -> Any:
        session_id = str(plan.session_id()).strip()
        session_config = {str(key): str(value) for key, value in dict(plan.session_config()).items()}
        has_explicit_s3_credentials = getattr(plan, "has_explicit_s3_credentials", None)
        if not callable(has_explicit_s3_credentials):
            raise TypeError("distributed physical plan is missing has_explicit_s3_credentials()")
        use_session_credentials = not bool(has_explicit_s3_credentials())
        conn = self._get_session_conn(
            session_id,
            session_config,
            use_session_credentials=use_session_credentials,
        )
        effective_s3_config = self._refresh_session_s3_config(
            session_id,
            session_config,
            conn,
            use_session_credentials=use_session_credentials,
        )
        cursor = None
        cursor_registered = False
        debug_context = dict(debug_context or {})
        start = time.monotonic()

        try:
            with self._session_connections_lock:
                if self._shutdown_started:
                    raise RuntimeError("Ray worker runtime is shutting down")
                if session_id in self._closed_session_ids:
                    raise RuntimeError(f"Ray worker Vane session is closed: {session_id}")
                record = self._session_connections.get(session_id)
                if record is None or record[1] is not conn:
                    raise RuntimeError(f"Ray worker Vane session closed during task startup: {session_id}")
                cursor = conn.cursor()
                query_admitted = self._register_native_cursor(cursor, native_query_id)
                cursor_registered = True
            if not query_admitted:
                raise RuntimeError(f"native query is closing: {native_query_id}")
            effective_s3_config = _configure_duckdb_s3(
                cursor,
                effective_s3_config,
                use_session_credentials=use_session_credentials,
            )
            _ray_worker_memory_log(
                "native_execute_start",
                **debug_context,
                scan_task_map_count=len(scan_task_map or {}),
                exchange_source_task_map_count=len(exchange_source_task_map or {}),
                has_exchange_sink_instance=exchange_sink_instance is not None,
                has_dynamic_filter_domains=bool(dynamic_filter_domains),
            )
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
                effective_s3_config,
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
                if cursor is not None:
                    cursor.close()
            except Exception:
                pass
            finally:
                if cursor_registered:
                    self._unregister_native_cursor(cursor)

    @staticmethod
    async def _await_fragment_registration(registration_result: Any | None) -> None:
        if registration_result is None:
            return
        if isinstance(registration_result, ray.ObjectRef):
            await registration_result

    async def run_plan_return(
        self,
        plan: Any,  # DistributedPhysicalPlan from _duckdb.ray_cxx
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
        debug_context = dict(debug_context or {})

        copy_output_info = _copy_output_info_from_context(context)
        scan_task_map, exchange_source_task_map = _extract_native_task_maps_from_context(context)
        run_start = time.monotonic()
        _ray_worker_memory_log("run_plan_return_start", **debug_context)
        # Native execution, shuffle publication, and actor teardown are owned by
        # the execution query. The resource query can differ for nested
        # executions and is only the owner of the admission lease.
        query_id = str(query_task_lease.get("execution_query_id") or "").strip()
        if not query_id:
            raise RuntimeError("native task execution requires an execution_query_id")

        begin_execution = require_ray_cxx_attr(
            "begin_flight_shuffle_query_execution",
            hint="Ensure the C++ ray extension is built with Flight shuffle query fencing support.",
        )
        end_execution = require_ray_cxx_attr(
            "end_flight_shuffle_query_execution",
            hint="Ensure the C++ ray extension is built with Flight shuffle query fencing support.",
        )

        self._begin_worker_native_execution(query_id)
        try:
            begin_execution(query_id)
        except BaseException:
            self._end_worker_native_execution(query_id)
            raise

        def execute_native_task() -> Any:
            try:
                if self._worker_native_query_is_closing(query_id):
                    raise RuntimeError(f"native query is closing: {query_id}")
                return self._execute_native_task(
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
                    native_query_id=query_id,
                )
            finally:
                try:
                    end_execution(query_id)
                finally:
                    self._end_worker_native_execution(query_id)

        try:
            native_future = asyncio.get_running_loop().run_in_executor(None, execute_native_task)
        except BaseException:
            try:
                end_execution(query_id)
            finally:
                self._end_worker_native_execution(query_id)
            raise
        result_list = await asyncio.shield(native_future)
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
        if native_exchange_sink_instance is not None:
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
