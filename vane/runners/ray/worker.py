# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="untyped-decorator"

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, NamedTuple, cast

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
from vane.runners.fte.debug_memory import (
    debug_flag_enabled,
    describe_result_payload,
    log_debug,
    process_memory_snapshot,
)
from vane.runners.fte.fte_config import FteWorkerAdmissionConfig
from vane.runners.fte.fte_failures import FteTaskTerminalControlError, _safe_failure_message
from vane.runners.fte.memory_config import apply_duckdb_memory_limit
from vane.runners.ray.admission_ledger import BoundedReplayMap, BoundedSet
from vane.runners.ray.fte_scheduler_config import _fte_control_rpc_timeout_s
from vane.runners.ray.ray_env import build_explicit_session_process_env, scrub_shared_runtime_session_env

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
_CONNECTION_SNAPSHOT_S3_CREDENTIAL_SETTINGS = frozenset(
    {
        "s3_access_key_id",
        "s3_secret_access_key",
        "s3_session_token",
    }
)
_CONNECTION_SNAPSHOT_SECURITY_SETTINGS = frozenset(
    {
        "allow_persistent_secrets",
        "allow_unsigned_extensions",
        "autoinstall_known_extensions",
        "autoload_known_extensions",
        "default_secret_storage",
        "secret_directory",
    }
)


async def _await_future_with_owned_side_effects(future: asyncio.Future[Any]) -> Any:
    """Do not expose cancellation until a future-owned mutation has finished."""
    cancellation: asyncio.CancelledError | None = None
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError as error:
            cancellation = error
    result = future.result()
    if cancellation is not None:
        raise cancellation
    return result


async def _to_thread_with_owned_side_effects(
    callback: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Do not expose cancellation until a thread-owned mutation has finished."""
    thread_task = asyncio.create_task(asyncio.to_thread(callback, *args, **kwargs))
    return await _await_future_with_owned_side_effects(thread_task)


async def _await_with_repeated_native_interrupts(
    operation: Awaitable[Any],
    interrupt_callback: Callable[[], list[str]],
    interrupt_errors: set[str],
) -> Any:
    """Keep an admitted native cursor interrupted until its owner is terminal."""
    operation_task = asyncio.ensure_future(operation)
    cancellation: asyncio.CancelledError | None = None
    while not operation_task.done():
        try:
            await asyncio.wait({operation_task}, timeout=0.01)
        except asyncio.CancelledError as error:
            # Retain this cancellation until the owned operation is terminal.
            # Catching the injection lets the next timed wait preserve pacing.
            cancellation = error
        if operation_task.done():
            break
        try:
            interrupt_errors.update(str(error) for error in interrupt_callback())
        except Exception as error:
            interrupt_errors.add(f"{type(error).__name__}: {error}")
    result = operation_task.result()
    if cancellation is not None:
        raise cancellation
    return result


def _fte_applied_control_status(
    operation: str,
    task_id: str | dict[str, Any],
    status: dict[str, Any],
    *,
    applied: bool | None = None,
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
    result["_fte_control_applied"] = (
        str(result.get("state") or "").upper() != "UNKNOWN" if applied is None else bool(applied)
    )
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


def _ray_worker_observability_log(event: str, **fields: Any) -> None:
    if not debug_flag_enabled(
        "VANE_RAY_WORKER_MEMORY_DEBUG",
        "VANE_FTE_RESULT_DEBUG",
        "VANE_FTE_ADMISSION_DEBUG",
        "DUCKDB_DISTRIBUTED_DEBUG",
    ):
        return
    log_debug("vane-ray-worker", event, **fields)


def _fte_worker_label() -> str:
    return os.getenv("VANE_WORKER_ID", "").strip() or os.getenv("VANE_FTE_WORKER_ID", "").strip() or "-"


def _ray_worker_log_fields(worker: Any) -> dict[str, str]:
    return {
        "worker_id": str(getattr(worker, "_worker_id", "") or _fte_worker_label()),
        "manager_instance_id": str(
            getattr(worker, "_manager_instance_id", "") or os.getenv("VANE_WORKER_MANAGER_INSTANCE_ID", "")
        ).strip(),
        "node_id": str(getattr(worker, "_node_id", "") or os.getenv("VANE_WORKER_NODE_ID", "")).strip(),
        "host": str(getattr(worker, "_host", "") or os.getenv("VANE_WORKER_HOST", "")).strip(),
    }


def _ensure_python_datasource_runtime() -> None:
    import vane.datasource  # noqa: F401


def _release_datasource_factories_for_query(query_id: str) -> int:
    from vane import _native

    return int(_native._release_datasource_factories_for_query(str(query_id)))


def _clear_datasource_factory_registry() -> None:
    from vane import _native

    _native._clear_datasource_factory_registry()


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


class CleanupConnectionSnapshotIdentity(NamedTuple):
    """Connection state that determines the cleanup filesystem."""

    bootstrap_database: str
    bootstrap_read_only: bool
    bootstrap_config: tuple[tuple[str, str], ...]
    settings: tuple[tuple[str, str, str], ...]


class WorkerSnapshotDatabaseIdentity(NamedTuple):
    """Exact identity for a worker-owned DatabaseInstance."""

    database: str
    read_only: bool
    config: tuple[tuple[str, str], ...]
    settings: tuple[tuple[str, str, str], ...]
    duckdb_source_id: str
    extensions: tuple[tuple[str, str], ...]
    distributed_extension_contracts: tuple[str, ...]
    s3_session_id: str
    effective_s3_config_identity: str
    use_session_credentials: bool

    def has_static_extension(self, extension_name: str) -> bool:
        normalized_name = str(extension_name).lower()
        return any(name.lower() == normalized_name for name, _version in self.extensions)

    def replaces_s3_identity(self, other: WorkerSnapshotDatabaseIdentity) -> bool:
        if not self.s3_session_id or self == other:
            return False
        return self._replace(
            effective_s3_config_identity="",
            use_session_credentials=False,
        ) == other._replace(
            effective_s3_config_identity="",
            use_session_credentials=False,
        )


def _snapshot_nonempty_string(entry: Mapping[str, Any], field: str, description: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value:
        raise TypeError(f"query connection snapshot {description} {field} must be a non-empty string")
    return value


def _worker_snapshot_database_identity(
    snapshot: Mapping[str, Any],
    *,
    session_id: str,
    effective_s3_config: Mapping[str, str],
    use_session_credentials: bool,
) -> WorkerSnapshotDatabaseIdentity:
    raw_bootstrap = snapshot.get("bootstrap")
    if raw_bootstrap is None:
        database = ":memory:"
        read_only = False
        raw_config: Mapping[str, Any] = {}
    else:
        if not isinstance(raw_bootstrap, Mapping):
            raise TypeError("query connection snapshot bootstrap must be a mapping")
        raw_database = raw_bootstrap.get("database", ":memory:")
        if not isinstance(raw_database, str) or not raw_database:
            raise TypeError("query connection snapshot bootstrap database must be a non-empty string")
        database = raw_database
        raw_read_only = raw_bootstrap.get("read_only", False)
        if not isinstance(raw_read_only, bool):
            raise TypeError("query connection snapshot bootstrap read_only must be a boolean")
        read_only = raw_read_only
        raw_config_value = raw_bootstrap.get("config", {})
        if not isinstance(raw_config_value, Mapping):
            raise TypeError("query connection snapshot bootstrap config must be a mapping")
        raw_config = raw_config_value
    config = tuple(sorted((str(key).lower(), str(value)) for key, value in raw_config.items()))
    if len({key for key, _value in config}) != len(config):
        raise ValueError("query connection snapshot bootstrap has duplicate case-insensitive config keys")

    raw_settings = snapshot.get("settings")
    if not isinstance(raw_settings, list):
        raise TypeError("query connection snapshot settings must be a list")
    settings: list[tuple[str, str, str]] = []
    explicit_s3_credentials: list[tuple[str, str, str]] = []
    setting_names: set[str] = set()
    for raw_setting in raw_settings:
        if not isinstance(raw_setting, Mapping):
            raise TypeError("query connection snapshot setting entry must be a mapping")
        name = _snapshot_nonempty_string(raw_setting, "name", "setting")
        value = raw_setting.get("value")
        if not isinstance(value, str):
            raise TypeError("query connection snapshot setting value must be a string")
        input_type = raw_setting.get("input_type")
        if not isinstance(input_type, str) or not input_type:
            raise TypeError("query connection snapshot setting input_type must be a non-empty string")
        normalized_name = name.lower()
        if normalized_name in setting_names:
            raise ValueError(f"query connection snapshot has duplicate setting name: {name}")
        setting_names.add(normalized_name)
        if normalized_name in _CONNECTION_SNAPSHOT_SECURITY_SETTINGS:
            continue
        normalized_setting = (normalized_name, value, input_type.upper())
        if normalized_name in _CONNECTION_SNAPSHOT_S3_CREDENTIAL_SETTINGS:
            explicit_s3_credentials.append(normalized_setting)
            continue
        settings.append(normalized_setting)
    settings.sort()
    explicit_s3_credentials.sort()

    duckdb_source_id = snapshot.get("duckdb_source_id")
    if not isinstance(duckdb_source_id, str) or not duckdb_source_id:
        raise TypeError("query connection snapshot duckdb_source_id must be a non-empty string")

    raw_extensions = snapshot.get("extensions")
    if not isinstance(raw_extensions, list):
        raise TypeError("query connection snapshot extensions must be a list")
    extensions: list[tuple[str, str]] = []
    extension_names: set[str] = set()
    for raw_extension in raw_extensions:
        if not isinstance(raw_extension, Mapping):
            raise TypeError("query connection snapshot extension entry must be a mapping")
        name = _snapshot_nonempty_string(raw_extension, "name", "extension")
        version = raw_extension.get("version")
        if not isinstance(version, str):
            raise TypeError("query connection snapshot extension version must be a string")
        if name in extension_names:
            raise ValueError(f"query connection snapshot has duplicate extension name: {name}")
        extension_names.add(name)
        extensions.append((name, version))
    extensions.sort()

    raw_distributed_contracts = snapshot.get("distributed_extension_contracts")
    if not isinstance(raw_distributed_contracts, list):
        raise TypeError("query connection snapshot distributed_extension_contracts must be a list")
    distributed_contracts: list[str] = []
    seen_distributed_contracts: set[str] = set()
    for raw_contract in raw_distributed_contracts:
        if not isinstance(raw_contract, str) or not raw_contract:
            raise TypeError("query connection snapshot distributed extension contract must be a non-empty string")
        if raw_contract in seen_distributed_contracts:
            raise ValueError("query connection snapshot has a duplicate distributed extension contract")
        seen_distributed_contracts.add(raw_contract)
        distributed_contracts.append(raw_contract)
    distributed_contracts.sort()
    has_httpfs = any(name.lower() == "httpfs" for name, _version in extensions)
    if has_httpfs:
        s3_session_id = str(session_id).strip()
        if not s3_session_id:
            raise ValueError("worker snapshot httpfs database identity requires a Vane session_id")
        if use_session_credentials:
            s3_config: tuple[Any, ...] = (
                "session",
                tuple(
                    (key, str(effective_s3_config[key]))
                    for key in _DUCKDB_S3_SESSION_KEYS
                    if str(effective_s3_config.get(key, "")).strip()
                ),
            )
        else:
            s3_config = ("explicit", tuple(explicit_s3_credentials))
        s3_config_identity = hashlib.sha256(
            json.dumps(s3_config, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        identity_uses_session_credentials = bool(use_session_credentials)
    else:
        s3_session_id = ""
        s3_config_identity = ""
        identity_uses_session_credentials = False
    return WorkerSnapshotDatabaseIdentity(
        database,
        read_only,
        config,
        tuple(settings),
        duckdb_source_id,
        tuple(extensions),
        tuple(distributed_contracts),
        s3_session_id,
        s3_config_identity,
        identity_uses_session_credentials,
    )


def _query_worker_snapshot_database_identity(
    connection_snapshot_query_id: str,
    *,
    session_id: str,
    effective_s3_config: Mapping[str, str],
    use_session_credentials: bool,
) -> WorkerSnapshotDatabaseIdentity:
    lookup = require_ray_cxx_attr(
        "_lookup_query_connection_snapshot",
        hint="Ensure the C++ ray extension is built with query replay lifecycle support.",
    )
    snapshot = lookup(str(connection_snapshot_query_id))
    if snapshot is None:
        raise RuntimeError(
            f"query connection snapshot is unavailable during worker connection resolution: "
            f"{connection_snapshot_query_id}"
        )
    if not isinstance(snapshot, Mapping):
        raise TypeError("query connection snapshot must be a mapping")
    return _worker_snapshot_database_identity(
        snapshot,
        session_id=session_id,
        effective_s3_config=effective_s3_config,
        use_session_credentials=use_session_credentials,
    )


def _query_cleanup_connection_identity(
    connection_snapshot_query_id: str,
    *,
    apply_snapshot_s3_credentials: bool,
) -> CleanupConnectionSnapshotIdentity:
    """Return the bootstrap and settings that cleanup replay will apply."""
    lookup = require_ray_cxx_attr(
        "_lookup_query_connection_snapshot",
        hint="Ensure the C++ ray extension is built with query replay lifecycle support.",
    )
    snapshot = lookup(str(connection_snapshot_query_id))
    if snapshot is None:
        raise RuntimeError(
            "query connection snapshot is unavailable during cleanup context registration: "
            f"{connection_snapshot_query_id}"
        )
    if not isinstance(snapshot, Mapping):
        raise TypeError("query connection snapshot must be a mapping")

    bootstrap_database = ":memory:"
    bootstrap_read_only = False
    bootstrap_config: tuple[tuple[str, str], ...] = ()
    raw_bootstrap = snapshot.get("bootstrap")
    if isinstance(raw_bootstrap, Mapping):
        bootstrap_database = str(raw_bootstrap.get("database") or ":memory:")
        bootstrap_read_only = bool(raw_bootstrap.get("read_only", False))
        raw_bootstrap_config = raw_bootstrap.get("config")
        if isinstance(raw_bootstrap_config, Mapping):
            bootstrap_config = tuple(sorted((str(key), str(value)) for key, value in raw_bootstrap_config.items()))

    raw_settings = snapshot.get("settings", [])
    if raw_settings is None:
        raw_settings = []
    if not isinstance(raw_settings, list):
        raw_settings = []

    settings: list[tuple[str, str, str]] = []
    for raw_setting in raw_settings:
        if not isinstance(raw_setting, Mapping) or "name" not in raw_setting or "value" not in raw_setting:
            continue
        name = str(raw_setting["name"])
        lower_name = name.lower()
        if (
            not name
            or lower_name in _CONNECTION_SNAPSHOT_SECURITY_SETTINGS
            or (not apply_snapshot_s3_credentials and lower_name in _CONNECTION_SNAPSHOT_S3_CREDENTIAL_SETTINGS)
        ):
            continue
        settings.append(
            (
                lower_name,
                str(raw_setting["value"]),
                str(raw_setting.get("input_type", "VARCHAR")).upper(),
            )
        )
    return CleanupConnectionSnapshotIdentity(
        bootstrap_database=bootstrap_database,
        bootstrap_read_only=bootstrap_read_only,
        bootstrap_config=bootstrap_config,
        settings=tuple(settings),
    )


def _cleanup_query_python_replay_state(query_id: str) -> None:
    cleanup = require_ray_cxx_attr(
        "_cleanup_query_python_replay_state",
        hint="Ensure the C++ ray extension is built with query replay lifecycle support.",
    )
    cleanup(str(query_id))


def _cleanup_flight_shuffle_for_query(
    query_id: str,
    cleanup_connection: Any | None = None,
    connection_snapshot_query_id: str = "",
    *,
    apply_snapshot_s3_credentials: bool = True,
    effective_session_config: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    query_id = str(query_id or "").strip()
    if not query_id:
        return {
            "registry_entries_removed": 0,
            "storage_entries_removed": 0,
            "cleanup_errors": 0,
            "cleanup_storage_required": 0,
            "cleanup_pending": 0,
            "active_executions": 0,
            "last_error": "",
        }
    cleanup_fn = require_ray_cxx_attr(
        "cleanup_flight_shuffle_for_query",
        hint="Ensure the C++ ray extension is built with Flight shuffle cleanup support.",
    )
    if cleanup_connection is None:
        raw = cleanup_fn(query_id)
    else:
        raw = cleanup_fn(
            query_id,
            cleanup_connection,
            str(connection_snapshot_query_id or ""),
            bool(apply_snapshot_s3_credentials),
            None
            if effective_session_config is None
            else {str(key): str(value) for key, value in effective_session_config.items()},
        )
    if not isinstance(raw, dict):
        raise TypeError("Flight shuffle cleanup binding must return a dict")
    return {
        "registry_entries_removed": int(raw.get("registry_entries_removed", 0)),
        "storage_entries_removed": int(raw.get("storage_entries_removed", 0)),
        "cleanup_errors": int(raw.get("cleanup_errors", 0)),
        "cleanup_storage_required": int(raw.get("cleanup_storage_required", 0)),
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
    cleanup_callback: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if timeout_s is None:
        timeout_s = _fte_control_rpc_timeout_s() * 0.8
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    total_registry_removed = 0
    total_storage_removed = 0
    backoff_s = 0.01
    cleanup_once = cleanup_callback or _cleanup_flight_shuffle_for_query
    while True:
        cleanup = await asyncio.to_thread(cleanup_once, query_id)
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
    """Validate FTE result metadata before the worker publishes ObjectRefs.

    ``target_output_block_bytes`` and ``output_window_bytes`` are pending-output
    estimates used by query admission. Native execution materializes its
    result before its exact size is known, so either value may be exceeded (or
    be zero for a sink). The manager replaces the estimate with these exact
    bytes after publication and applies soft backpressure to subsequent work.
    """
    lease_id = str(query_task_lease.get("lease_id") or "").strip()
    query_id = str(query_task_lease.get("query_id") or "").strip()
    resource_unit_id = str(query_task_lease.get("resource_unit_id") or "").strip()
    attempt_id = str(query_task_lease.get("attempt_id") or "").strip()
    if not lease_id or not query_id or not resource_unit_id or not attempt_id:
        raise RuntimeError("FTE output publication requires a complete query task lease identity")

    normalized_sizes: list[int] = []
    for index, metadata in enumerate(partition_metadatas):
        num_rows = int(metadata.num_rows)
        raw_size = int(metadata.size_bytes or 0)
        if raw_size <= 0 and num_rows > 0:
            raise RuntimeError(
                f"FTE output block {index} is missing positive size_bytes: "
                f"query={query_id} resource_unit={resource_unit_id} attempt={attempt_id}"
            )
        size_bytes = max(1, raw_size)
        normalized_sizes.append(size_bytes)
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
            [sys.executable, "-m", "vane.runners.ray.aws_credentials"],
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


class NativeQueryCleanupContext(NamedTuple):
    """Serializable inputs for rebuilding a query-scoped cleanup cursor."""

    session_id: str
    session_config: tuple[tuple[str, str], ...]
    use_session_credentials: bool
    connection_snapshot_query_id: str
    connection_snapshot_identity: CleanupConnectionSnapshotIdentity


@ray.remote(concurrency_groups={"execute": 128, "control": 512})  # type: ignore[call-overload]
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
        runtime_context = ray.get_runtime_context()
        self._node_id = str(runtime_context.get_node_id() or "").strip()
        if not self._node_id:
            raise RuntimeError("Ray worker runtime context is missing node_id")
        self._worker_id = _fte_worker_label()
        self._manager_instance_id = os.getenv("VANE_WORKER_MANAGER_INSTANCE_ID", "").strip()
        self._host = ray_node_ip_address or os.getenv("VANE_WORKER_HOST", "").strip() or self._node_id
        get_actor_id = getattr(runtime_context, "get_actor_id", None)
        self._actor_id = str(get_actor_id() or "").strip() if callable(get_actor_id) else ""
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
        self._snapshot_connections: dict[WorkerSnapshotDatabaseIdentity, Any] = {}
        self._snapshot_connections_lock = threading.Lock()
        self._snapshot_connection_active_cursors: dict[WorkerSnapshotDatabaseIdentity, int] = {}
        self._snapshot_cursor_database_identities: dict[Any, WorkerSnapshotDatabaseIdentity] = {}
        self._retired_snapshot_database_identities: set[WorkerSnapshotDatabaseIdentity] = set()
        self._retired_snapshot_session_ids = BoundedSet[str](capacity=_SESSION_CLOSE_REPLAY_CAPACITY)
        self._session_connections: dict[str, tuple[dict[str, str], Any]] = {}
        self._session_s3_configs: dict[str, dict[str, str]] = {}
        self._session_operation_locks: dict[str, threading.Lock] = {}
        self._closed_session_ids = BoundedReplayMap[str, bool](capacity=_SESSION_CLOSE_REPLAY_CAPACITY)
        self._session_connections_lock = threading.RLock()
        self._shutdown_lock = threading.RLock()
        self._native_execution_condition = threading.Condition()
        self._native_execution_count = 0
        self._native_execution_counts_by_query: dict[str, int] = {}
        self._native_execution_counts_by_task: dict[str, int] = {}
        self._native_task_query_ids: dict[str, str] = {}
        self._native_query_cleanup_contexts: dict[str, NativeQueryCleanupContext] = {}
        self._active_native_cursors: set[Any] = set()
        self._native_cursor_query_ids: dict[Any, str] = {}
        self._native_cursor_task_ids: dict[Any, str] = {}
        self._closing_native_queries: set[str] = set()
        self._closing_native_tasks: set[str] = set()
        self._active_snapshot_execution_cursors = 0
        self._active_snapshot_cursors: set[Any] = set()
        self._shutdown_started = False
        self._shutdown_prepared = False
        self._shutdown_complete = False
        self._get_shared_conn()  # eagerly initialize
        _ray_worker_observability_log(
            "worker_registered",
            **self._worker_log_fields(),
            actor_id=self._actor_id or "-",
        )
        _ray_worker_memory_log(
            "actor_initialized",
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            worker_label=self._worker_id,
            **self._worker_log_fields(),
        )

    def _worker_log_fields(self) -> dict[str, str]:
        return _ray_worker_log_fields(self)

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
        pending_refs: list[ray.ObjectRef[Any]] = []
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
                worker_log_fields = self._worker_log_fields()
                self._fte_task_manager = FteWorkerTaskManager(
                    self._execute_fte_request,
                    admission_config=self._fte_admission_config,
                    require_query_task_lease=True,
                    worker_label=worker_log_fields["worker_id"],
                    worker_log_fields=worker_log_fields,
                )
            return self._fte_task_manager

    async def _execute_fte_request(self, request: Mapping[str, Any]) -> Any:
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
                plan = ray.cloudpickle.loads(
                    ray.cloudpickle.dumps(template_plan)  # type: ignore[no-untyped-call]
                )
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
            native_task_id=str(task_id),
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
        splits: list[Mapping[str, Any]],
        _fte_control_dependency: Any = None,
    ) -> dict[str, Any]:
        try:
            status = await self._get_fte_task_manager().add_splits(task_id, source_node_id, splits)
        except FteTaskTerminalControlError as exc:
            if exc.operation != "add_splits":
                raise
            return _fte_applied_control_status(
                "fte_add_splits",
                task_id,
                exc.status,
                applied=False,
            )
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
        attempt_id = FteTaskAttemptId.coerce(task_id)
        task_key = str(attempt_id)
        interrupt_errors = set(self._close_worker_native_task(task_key))
        try:
            status = await _await_with_repeated_native_interrupts(
                self._get_fte_task_manager().cancel_task(task_id),
                lambda: self._close_worker_native_task(task_key),
                interrupt_errors,
            )
        except BaseException as exc:
            if interrupt_errors:
                details = "; ".join(sorted(interrupt_errors))
                raise RuntimeError(
                    f"failed to quiesce native FTE task {task_key} after interrupt errors: {details}"
                ) from exc
            raise
        if interrupt_errors:
            details = "; ".join(sorted(interrupt_errors))
            raise RuntimeError(f"failed to quiesce native FTE task {task_key} after interrupt errors: {details}")
        self._retire_worker_native_task(task_key)
        return _fte_applied_control_status("fte_cancel_task", task_id, status)

    @ray.method(concurrency_group="control")
    def fte_interrupt_query(self, query_id: str) -> dict[str, int]:
        query_id = str(query_id)
        _close_flight_shuffle_query(query_id)
        interrupt_errors = self._close_worker_native_query(query_id)
        if interrupt_errors:
            raise RuntimeError(
                f"failed to interrupt {len(interrupt_errors)} native execution(s) for {query_id}: "
                + "; ".join(interrupt_errors)
            )
        return {"native_interrupt_errors": 0}

    @ray.method(concurrency_group="control")
    async def fte_prepare_drop_query(self, query_id: str) -> dict[str, int]:
        _close_flight_shuffle_query(query_id)
        interrupt_errors = set(self._close_worker_native_query(query_id))

        async def prepare_query_drop() -> tuple[dict[str, Any], int]:
            try:
                fte_result = await self._get_fte_task_manager().drop_query(query_id)
                fragments_removed = self.drop_query_fragments(query_id)  # type: ignore[operator]
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
                    raise RuntimeError(f"failed to prepare query teardown for {query_id}: {details}") from drain_errors[
                        0
                    ]
            return fte_result, fragments_removed

        fte_result, fragments_removed = await _await_with_repeated_native_interrupts(
            prepare_query_drop(),
            lambda: self._close_worker_native_query(query_id),
            interrupt_errors,
        )
        if interrupt_errors:
            raise RuntimeError(
                f"failed to interrupt {len(interrupt_errors)} native execution(s) for {query_id}: "
                + "; ".join(sorted(interrupt_errors))
            )
        return {
            "tasks_removed": int(fte_result["removed"]),
            "tasks_canceled": int(fte_result["canceled"]),
            "fragments_removed": int(fragments_removed),
        }

    @ray.method(concurrency_group="control")
    async def fte_cleanup_query(self, query_id: str) -> dict[str, int]:
        cleanup_with_context = getattr(self, "_cleanup_flight_shuffle_for_query_with_context", None)
        if callable(cleanup_with_context):
            flight_shuffle_cleanup = await _drain_flight_shuffle_for_query(
                query_id,
                cleanup_callback=cleanup_with_context,
            )
        else:
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
        prepared = await self.fte_prepare_drop_query(query_id)  # type: ignore[operator]
        if not isinstance(prepared, Mapping):
            raise TypeError("fte_prepare_drop_query() must return a mapping")
        result: dict[str, int] = {str(key): int(value) for key, value in prepared.items()}
        result.update(await self.fte_cleanup_query(query_id))  # type: ignore[operator]
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

    def _configure_snapshot_conn(self, conn: Any) -> None:
        """Apply actor-owned resource limits to an isolated snapshot database."""
        _configure_ray_worker_conn(conn, self._duckdb_memory_bytes)

    def _get_shared_conn(self) -> Any:
        """Return the shared DuckDB connection, creating it lazily on first use.

        Default-bootstrap tasks executed by this actor share this
        DatabaseInstance (and therefore the same TaskScheduler thread pool).
        Non-default snapshot databases are cached separately.
        """
        with self._shared_conn_lock:
            if self._shutdown_started:
                raise RuntimeError("Ray worker runtime is shut down")
            if self._shared_conn is not None:
                return self._shared_conn
            import vane

            conn = vane.connect(config={"allow_persistent_secrets": False})
            self._configure_conn(conn)
            self._shared_conn = conn
            return conn

    def _close_retired_snapshot_databases_locked(self) -> None:
        snapshot_connections = getattr(self, "_snapshot_connections", {})
        active_cursors = getattr(self, "_snapshot_connection_active_cursors", {})
        retired_identities: set[WorkerSnapshotDatabaseIdentity] = getattr(
            self,
            "_retired_snapshot_database_identities",
            set(),
        )
        for database_identity in tuple(retired_identities):
            if active_cursors.get(database_identity, 0) > 0:
                continue
            connection = snapshot_connections.get(database_identity)
            if connection is None:
                retired_identities.discard(database_identity)
                active_cursors.pop(database_identity, None)
                continue
            connection.close()
            if snapshot_connections.get(database_identity) is connection:
                snapshot_connections.pop(database_identity, None)
                active_cursors.pop(database_identity, None)
                retired_identities.discard(database_identity)

    def _activate_snapshot_database_identity_locked(
        self,
        database_identity: WorkerSnapshotDatabaseIdentity,
    ) -> bool:
        active_cursors = getattr(self, "_snapshot_connection_active_cursors", None)
        if active_cursors is None:
            active_cursors = {}
            self._snapshot_connection_active_cursors = active_cursors
        retired_identities = getattr(self, "_retired_snapshot_database_identities", None)
        if retired_identities is None:
            retired_identities = set()
            self._retired_snapshot_database_identities = retired_identities
        retired_session_ids = getattr(self, "_retired_snapshot_session_ids", ())
        retire_after_use = bool(
            database_identity.s3_session_id and database_identity.s3_session_id in retired_session_ids
        )

        if database_identity in retired_identities and active_cursors.get(database_identity, 0) == 0:
            self._close_retired_snapshot_databases_locked()
        if not retire_after_use:
            retired_identities.discard(database_identity)
        for existing_identity in getattr(self, "_snapshot_connections", {}):
            if database_identity.replaces_s3_identity(existing_identity):
                retired_identities.add(existing_identity)
        self._close_retired_snapshot_databases_locked()
        return retire_after_use

    def _retire_snapshot_databases_for_session(self, session_id: str) -> None:
        session_key = str(session_id).strip()
        if not session_key:
            return
        snapshot_connections_lock = getattr(self, "_snapshot_connections_lock", None)
        if snapshot_connections_lock is None:
            snapshot_connections_lock = threading.Lock()
            self._snapshot_connections_lock = snapshot_connections_lock
        with snapshot_connections_lock:
            retired_session_ids = getattr(self, "_retired_snapshot_session_ids", None)
            if retired_session_ids is None:
                retired_session_ids = BoundedSet[str](capacity=_SESSION_CLOSE_REPLAY_CAPACITY)
                self._retired_snapshot_session_ids = retired_session_ids
            retired_session_ids.add(session_key)
            retired_identities = getattr(self, "_retired_snapshot_database_identities", None)
            if retired_identities is None:
                retired_identities = set()
                self._retired_snapshot_database_identities = retired_identities
            for database_identity in getattr(self, "_snapshot_connections", {}):
                if database_identity.s3_session_id == session_key:
                    retired_identities.add(database_identity)
            self._close_retired_snapshot_databases_locked()

    def _get_snapshot_execution_cursor(
        self,
        bootstrap_connection: Any,
        connection_snapshot_query_id: str,
        *,
        database_identity: WorkerSnapshotDatabaseIdentity,
    ) -> Any:
        """Return a task cursor on the snapshot's stable DatabaseInstance."""
        with self._native_execution_condition:
            if self._shutdown_started:
                raise RuntimeError("Ray worker runtime is shutting down")
            self._active_snapshot_execution_cursors = int(getattr(self, "_active_snapshot_execution_cursors", 0)) + 1
        cursor = None
        try:
            snapshot_connections_lock = getattr(self, "_snapshot_connections_lock", None)
            if snapshot_connections_lock is None:
                snapshot_connections_lock = threading.Lock()
                self._snapshot_connections_lock = snapshot_connections_lock
            with snapshot_connections_lock:
                snapshot_connections = getattr(self, "_snapshot_connections", None)
                if snapshot_connections is None:
                    snapshot_connections = {}
                    self._snapshot_connections = snapshot_connections
                retire_after_use = self._activate_snapshot_database_identity_locked(database_identity)
                connection: Any = snapshot_connections.get(database_identity)
                if connection is None:
                    resolve = require_ray_cxx_attr(
                        "_resolve_query_snapshot_connection",
                        hint="Ensure the C++ ray extension is built with worker snapshot resolution support.",
                    )
                    connection = cast(Any, resolve(bootstrap_connection, str(connection_snapshot_query_id)))
                    if connection is bootstrap_connection:
                        raise RuntimeError("non-default worker snapshot unexpectedly reused the bootstrap connection")
                    try:
                        self._configure_snapshot_conn(connection)
                    except BaseException as config_error:
                        close = getattr(connection, "close", None)
                        if callable(close):
                            try:
                                close()
                            except BaseException as close_error:
                                raise RuntimeError(
                                    "worker snapshot configuration failed and its isolated "
                                    "DuckDB connection could not be closed: "
                                    f"{type(close_error).__name__}: {close_error}"
                                ) from config_error
                        raise
                    snapshot_connections[database_identity] = connection
                cursor = connection.cursor()
                active_cursors = getattr(self, "_snapshot_connection_active_cursors", None)
                if active_cursors is None:
                    active_cursors = {}
                    self._snapshot_connection_active_cursors = active_cursors
                active_cursors[database_identity] = active_cursors.get(database_identity, 0) + 1
                cursor_identities = getattr(self, "_snapshot_cursor_database_identities", None)
                if cursor_identities is None:
                    cursor_identities = {}
                    self._snapshot_cursor_database_identities = cursor_identities
                cursor_identities[cursor] = database_identity
                if retire_after_use:
                    self._retired_snapshot_database_identities.add(database_identity)
            with self._native_execution_condition:
                active_snapshot_cursors = getattr(self, "_active_snapshot_cursors", None)
                if active_snapshot_cursors is None:
                    active_snapshot_cursors = set()
                    self._active_snapshot_cursors = active_snapshot_cursors
                active_snapshot_cursors.add(cursor)
            return cursor
        except BaseException:
            with self._native_execution_condition:
                if cursor is not None:
                    getattr(self, "_active_snapshot_cursors", set()).discard(cursor)
                active_cursors = int(getattr(self, "_active_snapshot_execution_cursors", 0))
                if active_cursors <= 0:
                    raise RuntimeError("Ray worker snapshot execution cursor ownership underflow")
                self._active_snapshot_execution_cursors = active_cursors - 1
                self._native_execution_condition.notify_all()
            raise

    def _close_snapshot_execution_cursor(self, cursor: Any) -> None:
        """Close a snapshot cursor and release its shutdown fence."""
        try:
            cursor.close()
        finally:
            try:
                snapshot_connections_lock = getattr(self, "_snapshot_connections_lock", None)
                if snapshot_connections_lock is None:
                    raise RuntimeError("Ray worker snapshot database identity lock is unavailable")
                with snapshot_connections_lock:
                    cursor_identities = getattr(self, "_snapshot_cursor_database_identities", {})
                    database_identity = cursor_identities.pop(cursor, None)
                    if database_identity is None:
                        raise RuntimeError("Ray worker snapshot cursor database identity is unavailable")
                    active_cursors_by_database = getattr(self, "_snapshot_connection_active_cursors", {})
                    database_active_cursors = active_cursors_by_database.get(database_identity, 0)
                    if database_active_cursors <= 0:
                        raise RuntimeError("Ray worker snapshot database cursor ownership underflow")
                    if database_active_cursors == 1:
                        active_cursors_by_database.pop(database_identity, None)
                    else:
                        active_cursors_by_database[database_identity] = database_active_cursors - 1
                    self._close_retired_snapshot_databases_locked()
            finally:
                with self._native_execution_condition:
                    getattr(self, "_active_snapshot_cursors", set()).discard(cursor)
                    active_cursors = int(getattr(self, "_active_snapshot_execution_cursors", 0))
                    if active_cursors <= 0:
                        raise RuntimeError("Ray worker snapshot execution cursor ownership underflow")
                    self._active_snapshot_execution_cursors = active_cursors - 1
                    self._native_execution_condition.notify_all()

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
            return self._refresh_session_s3_config_locked(
                session_id,
                session_config,
                connection,
                use_session_credentials=use_session_credentials,
            )

    def _refresh_session_s3_config_locked(
        self,
        session_id: str,
        session_config: dict[str, str],
        connection: Any,
        *,
        use_session_credentials: bool,
    ) -> dict[str, str]:
        """Refresh credentials while the caller holds the session operation lock."""
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

    def _register_native_query_cleanup_context(
        self,
        query_id: str,
        plan: Any,
        session_id: str,
        session_config: Mapping[str, str],
        *,
        use_session_credentials: bool,
    ) -> None:
        query_key = str(query_id or "").strip()
        if not query_key:
            return
        connection_snapshot_query_id = _plan_resource_query_id(plan)
        use_session_credentials = bool(use_session_credentials)
        cleanup_context = NativeQueryCleanupContext(
            session_id=str(session_id),
            session_config=tuple(sorted((str(key), str(value)) for key, value in session_config.items())),
            use_session_credentials=use_session_credentials,
            connection_snapshot_query_id=connection_snapshot_query_id,
            connection_snapshot_identity=_query_cleanup_connection_identity(
                connection_snapshot_query_id,
                apply_snapshot_s3_credentials=not use_session_credentials,
            ),
        )
        with self._native_execution_condition:
            contexts = getattr(self, "_native_query_cleanup_contexts", None)
            if contexts is None:
                contexts = {}
                self._native_query_cleanup_contexts = contexts
            existing = contexts.get(query_key)
            if existing is not None:
                # One execution query can contain independently materialized
                # fragment plans with different replay-registry keys. Their
                # effective cleanup settings must agree, but retaining the first
                # live snapshot is sufficient for the query-scoped storage context.
                if (
                    existing.session_id != cleanup_context.session_id
                    or existing.session_config != cleanup_context.session_config
                    or existing.use_session_credentials != cleanup_context.use_session_credentials
                    or existing.connection_snapshot_identity != cleanup_context.connection_snapshot_identity
                ):
                    raise RuntimeError(f"native query cleanup context changed: {query_key}")
                return
            contexts[query_key] = cleanup_context

    def _cleanup_flight_shuffle_for_query_with_context(self, query_id: str) -> dict[str, Any]:
        query_key = str(query_id or "").strip()
        with self._native_execution_condition:
            cleanup_context = getattr(self, "_native_query_cleanup_contexts", {}).get(query_key)
        if cleanup_context is None:
            return _cleanup_flight_shuffle_for_query(query_key)

        # Retained local caches need no connection context. Probe first so
        # unrelated AWS credential failures cannot block local-only teardown.
        probe = _cleanup_flight_shuffle_for_query(query_key)
        cleanup_storage_required = int(probe["cleanup_storage_required"])
        if cleanup_storage_required == 0:
            return probe

        session_config = dict(cleanup_context.session_config)
        operation_lock = self._get_session_operation_lock(
            cleanup_context.session_id,
            allow_closed=True,
        )
        with operation_lock:
            with self._session_connections_lock:
                cached_s3_config = dict(
                    getattr(self, "_session_s3_configs", {}).get(
                        cleanup_context.session_id,
                        session_config,
                    )
                )
            effective_s3_config = _refresh_effective_duckdb_s3_config(
                session_config,
                cached_s3_config,
                use_session_credentials=cleanup_context.use_session_credentials,
            )
            with self._session_connections_lock:
                if cleanup_context.session_id in getattr(self, "_session_connections", {}):
                    self._session_s3_configs[cleanup_context.session_id] = effective_s3_config
            database_identity = _query_worker_snapshot_database_identity(
                cleanup_context.connection_snapshot_query_id,
                session_id=cleanup_context.session_id,
                effective_s3_config=effective_s3_config,
                use_session_credentials=cleanup_context.use_session_credentials,
            )
            cleanup_cursor = self._get_snapshot_execution_cursor(
                self._get_shared_conn(),
                cleanup_context.connection_snapshot_query_id,
                database_identity=database_identity,
            )
        try:
            if database_identity.has_static_extension("httpfs"):
                _configure_duckdb_s3(
                    cleanup_cursor,
                    effective_s3_config,
                    use_session_credentials=cleanup_context.use_session_credentials,
                )
            apply_snapshot_s3_credentials = not cleanup_context.use_session_credentials
            cleanup = _cleanup_flight_shuffle_for_query(
                query_key,
                cleanup_cursor,
                cleanup_context.connection_snapshot_query_id,
                # Refreshed profile/role credentials must remain newer than
                # plan-time local settings captured in the snapshot. Explicit
                # connection credentials still replay when session credentials
                # are intentionally disabled.
                apply_snapshot_s3_credentials=apply_snapshot_s3_credentials,
                effective_session_config=effective_s3_config,
            )
            cleanup["registry_entries_removed"] += probe["registry_entries_removed"]
            cleanup["storage_entries_removed"] += probe["storage_entries_removed"]
            # The retry satisfies only the probe errors caused by missing
            # caller-owned storage; preserve any independent cleanup failures.
            unresolved_probe_errors = max(0, probe["cleanup_errors"] - cleanup_storage_required)
            cleanup["cleanup_errors"] += unresolved_probe_errors
            if unresolved_probe_errors and not cleanup["last_error"]:
                cleanup["last_error"] = probe["last_error"]
            return cleanup
        finally:
            try:
                self._close_snapshot_execution_cursor(cleanup_cursor)
            except Exception:
                # The cursor is no longer reachable after this callback. Keep
                # the storage/configuration error as the retry diagnostic.
                pass

    @ray.method(concurrency_group="control")
    async def close_session(self, session_id: str) -> None:
        session_key = str(session_id).strip()
        if not session_key:
            raise ValueError("Ray worker close_session requires a Vane session_id")

        def _close() -> None:
            operation_lock = self._get_session_operation_lock(session_key, allow_closed=True)
            with self._session_connections_lock:
                self._closed_session_ids[session_key] = True
            self._retire_snapshot_databases_for_session(session_key)
            with operation_lock:
                with self._session_connections_lock:
                    record = self._session_connections.get(session_key)
                if record is not None:
                    _, connection = record
                    connection.close()
                    with self._session_connections_lock:
                        if self._session_connections.get(session_key) is record:
                            self._session_connections.pop(session_key, None)
                            getattr(self, "_session_s3_configs", {}).pop(session_key, None)
                self._retire_snapshot_databases_for_session(session_key)
                with self._session_connections_lock:
                    operation_locks = getattr(self, "_session_operation_locks", {})
                    if operation_locks.get(session_key) is operation_lock:
                        getattr(self, "_session_operation_locks", {}).pop(session_key, None)

        await _to_thread_with_owned_side_effects(_close)

    def _begin_worker_native_execution(self, query_id: str, task_id: str = "") -> None:
        query_id = str(query_id or "").strip()
        task_id = str(task_id or "").strip()
        if not query_id:
            raise ValueError("native execution admission requires a query_id")
        with self._native_execution_condition:
            if self._shutdown_started:
                raise RuntimeError("Ray worker runtime is shutting down")
            if query_id in self._closing_native_queries:
                raise RuntimeError(f"native query is closing: {query_id}")
            if task_id and task_id in self._closing_native_tasks:
                raise RuntimeError(f"native task is closing: {task_id}")
            if task_id:
                existing_query_id = self._native_task_query_ids.get(task_id)
                if existing_query_id is not None and existing_query_id != query_id:
                    raise RuntimeError(
                        f"native task ownership query mismatch: task={task_id} "
                        f"expected={existing_query_id} actual={query_id}"
                    )
            self._native_execution_count += 1
            self._native_execution_counts_by_query[query_id] = (
                self._native_execution_counts_by_query.get(query_id, 0) + 1
            )
            if task_id:
                self._native_task_query_ids[task_id] = query_id
                self._native_execution_counts_by_task[task_id] = (
                    self._native_execution_counts_by_task.get(task_id, 0) + 1
                )

    def _end_worker_native_execution(self, query_id: str, task_id: str = "") -> None:
        query_id = str(query_id or "").strip()
        task_id = str(task_id or "").strip()
        with self._native_execution_condition:
            if self._native_execution_count <= 0:
                raise RuntimeError("Ray worker native execution ownership underflow")
            query_count = self._native_execution_counts_by_query.get(query_id, 0)
            if query_count <= 0:
                raise RuntimeError(f"Ray worker native query execution ownership underflow: {query_id}")
            task_count = 0
            if task_id:
                task_count = self._native_execution_counts_by_task.get(task_id, 0)
                if task_count <= 0:
                    raise RuntimeError(f"Ray worker native task execution ownership underflow: {task_id}")
            self._native_execution_count -= 1
            if query_count == 1:
                self._native_execution_counts_by_query.pop(query_id, None)
            else:
                self._native_execution_counts_by_query[query_id] = query_count - 1
            if task_id:
                if task_count == 1:
                    self._native_execution_counts_by_task.pop(task_id, None)
                    if task_id not in self._closing_native_tasks:
                        self._native_task_query_ids.pop(task_id, None)
                else:
                    self._native_execution_counts_by_task[task_id] = task_count - 1
            self._native_execution_condition.notify_all()

    def _register_native_cursor(self, cursor: Any, query_id: str = "", task_id: str = "") -> bool:
        query_id = str(query_id)
        task_id = str(task_id or "").strip()
        with self._native_execution_condition:
            self._active_native_cursors.add(cursor)
            self._native_cursor_query_ids[cursor] = query_id
            if task_id:
                self._native_cursor_task_ids[cursor] = task_id
            return query_id not in self._closing_native_queries and (
                not task_id or task_id not in self._closing_native_tasks
            )

    def _unregister_native_cursor(self, cursor: Any) -> None:
        with self._native_execution_condition:
            self._active_native_cursors.discard(cursor)
            self._native_cursor_query_ids.pop(cursor, None)
            self._native_cursor_task_ids.pop(cursor, None)
            self._native_execution_condition.notify_all()

    def _worker_native_query_is_closing(self, query_id: str) -> bool:
        with self._native_execution_condition:
            return str(query_id) in self._closing_native_queries

    def _worker_native_task_is_closing(self, task_id: str) -> bool:
        task_id = str(task_id or "").strip()
        if not task_id:
            return False
        with self._native_execution_condition:
            return task_id in self._closing_native_tasks

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

    def _close_worker_native_task(self, task_id: str) -> list[str]:
        attempt_id = FteTaskAttemptId.coerce(task_id)
        task_id = str(attempt_id)
        query_id = str(attempt_id.query_id)
        with self._native_execution_condition:
            existing_query_id = self._native_task_query_ids.get(task_id)
            if existing_query_id is not None and existing_query_id != query_id:
                raise RuntimeError(
                    f"native task ownership query mismatch: task={task_id} "
                    f"expected={existing_query_id} actual={query_id}"
                )
            self._native_task_query_ids[task_id] = query_id
            self._closing_native_tasks.add(task_id)
            cursors = [
                cursor for cursor, cursor_task_id in self._native_cursor_task_ids.items() if cursor_task_id == task_id
            ]
        errors: list[str] = []
        for cursor in cursors:
            try:
                cursor.interrupt()
            except Exception as exc:
                errors.append(str(exc))
        return errors

    def _retire_worker_native_task(self, task_id: str) -> None:
        task_id = str(FteTaskAttemptId.coerce(task_id))
        with self._native_execution_condition:
            active_executions = self._native_execution_counts_by_task.get(task_id, 0)
            if active_executions:
                raise RuntimeError(
                    "cannot retire Ray worker native task with active executions: "
                    f"{task_id} active_executions={active_executions}"
                )
            self._closing_native_tasks.discard(task_id)
            self._native_task_query_ids.pop(task_id, None)

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
            retired_task_ids = [
                task_id for task_id, task_query_id in self._native_task_query_ids.items() if task_query_id == query_id
            ]
            for task_id in retired_task_ids:
                self._closing_native_tasks.discard(task_id)
                self._native_task_query_ids.pop(task_id, None)
            getattr(self, "_native_query_cleanup_contexts", {}).pop(query_id, None)

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
                self._native_execution_counts_by_task = {}
                self._native_task_query_ids = {}
                self._active_native_cursors = set()
                self._native_cursor_query_ids = {}
                self._native_cursor_task_ids = {}
                self._closing_native_queries = set()
                self._closing_native_tasks = set()
                self._active_snapshot_execution_cursors = 0
                self._active_snapshot_cursors = set()
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
            snapshot_connections_lock = getattr(self, "_snapshot_connections_lock", None)
            if snapshot_connections_lock is None:
                snapshot_connections_lock = threading.Lock()
                self._snapshot_connections_lock = snapshot_connections_lock
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
                    active_snapshot_cursors = int(getattr(self, "_active_snapshot_execution_cursors", 0))
                    active_cursors = set(getattr(self, "_active_native_cursors", ()))
                    active_cursors.update(getattr(self, "_active_snapshot_cursors", ()))
                if active_executions == 0 and active_snapshot_cursors == 0:
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
                    errors.append(
                        "timed out waiting for worker database users to stop: "
                        f"native_executions={active_executions} snapshot_cursors={active_snapshot_cursors}"
                    )
                    break
                with native_condition:
                    if (
                        self._native_execution_count > 0
                        or int(getattr(self, "_active_snapshot_execution_cursors", 0)) > 0
                    ):
                        native_condition.wait(timeout=min(0.1, remaining))
            with native_condition:
                native_drained = (
                    self._native_execution_count == 0
                    and int(getattr(self, "_active_snapshot_execution_cursors", 0)) == 0
                )
            if not native_drained:
                raise RuntimeError("; ".join(errors))
            # A task admitted immediately before shutdown may finish resolving a
            # non-default DatabaseInstance while shutdown waits on its cursor
            # fence. Snapshot the cache only after every such user has drained.
            with snapshot_connections_lock:
                snapshot_connections: list[tuple[WorkerSnapshotDatabaseIdentity, Any]] = list(
                    cast(
                        dict[WorkerSnapshotDatabaseIdentity, Any],
                        getattr(self, "_snapshot_connections", {}),
                    ).items()
                )
            for database_identity, snapshot_connection in snapshot_connections:
                with snapshot_connections_lock:
                    if getattr(self, "_snapshot_connections", {}).get(database_identity) is not snapshot_connection:
                        continue
                connection_to_close = cast(Any, snapshot_connection)
                try:
                    connection_to_close.close()
                except Exception as exc:
                    errors.append(f"close DuckDB snapshot connection {database_identity.database}: {exc}")
                else:
                    with snapshot_connections_lock:
                        if self._snapshot_connections.get(database_identity) is snapshot_connection:
                            self._snapshot_connections.pop(database_identity, None)
                            getattr(self, "_snapshot_connection_active_cursors", {}).pop(
                                database_identity,
                                None,
                            )
                            getattr(self, "_retired_snapshot_database_identities", set()).discard(database_identity)
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
        worker_log_fields = self._worker_log_fields()
        _ray_worker_observability_log("worker_shutdown_prepare_start", **worker_log_fields)
        try:
            self._prepare_worker_runtime_shutdown()
        except BaseException as exc:
            _ray_worker_observability_log(
                "worker_shutdown_prepare_error",
                **worker_log_fields,
                error_type=type(exc).__name__,
                error=_safe_failure_message(exc),
            )
            raise
        _ray_worker_observability_log("worker_shutdown_prepare_done", **worker_log_fields)

    @ray.method(concurrency_group="control")
    def finish_shutdown(self) -> None:
        worker_log_fields = self._worker_log_fields()
        _ray_worker_observability_log("worker_shutdown_finish_start", **worker_log_fields)
        try:
            self._finish_worker_runtime_shutdown()
        except BaseException as exc:
            _ray_worker_observability_log(
                "worker_shutdown_finish_error",
                **worker_log_fields,
                error_type=type(exc).__name__,
                error=_safe_failure_message(exc),
            )
            raise
        _ray_worker_observability_log("worker_shutdown_finish_done", **worker_log_fields)

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
        native_task_id: str = "",
    ) -> Any:
        native_query_id = str(native_query_id or "").strip()
        native_task_id = str(native_task_id or "").strip()
        debug_context = dict(debug_context or {})
        runtime_context = dict(debug_context)
        debug_task_id = runtime_context.pop("task_id", None)
        if native_task_id:
            if debug_task_id is not None and str(debug_task_id) != native_task_id:
                raise RuntimeError(
                    "native runtime task identity differs from debug context: "
                    f"runtime={native_task_id} debug={debug_task_id}"
                )
            runtime_context["task_id"] = native_task_id
        elif debug_task_id is not None:
            raise RuntimeError("debug task identity requires an authoritative native runtime task identity")
        session_id = str(plan.session_id()).strip()
        session_config = {str(key): str(value) for key, value in dict(plan.session_config()).items()}
        has_explicit_s3_credentials = getattr(plan, "has_explicit_s3_credentials", None)
        if not callable(has_explicit_s3_credentials):
            raise TypeError("distributed physical plan is missing has_explicit_s3_credentials()")
        use_session_credentials = not bool(has_explicit_s3_credentials())
        connection_snapshot_query_id = _plan_resource_query_id(plan)
        conn = self._get_session_conn(
            session_id,
            session_config,
            use_session_credentials=use_session_credentials,
        )
        cursor = None
        cursor_registered = False
        worker_log_context = dict(debug_context)
        worker_log_context.update(_ray_worker_log_fields(self))
        start = time.monotonic()

        try:
            operation_lock = self._get_session_operation_lock(session_id)
            with operation_lock:
                effective_s3_config = self._refresh_session_s3_config_locked(
                    session_id,
                    session_config,
                    conn,
                    use_session_credentials=use_session_credentials,
                )
                with self._session_connections_lock:
                    if self._shutdown_started:
                        raise RuntimeError("Ray worker runtime is shutting down")
                    if session_id in self._closed_session_ids:
                        raise RuntimeError(f"Ray worker Vane session is closed: {session_id}")
                    record = self._session_connections.get(session_id)
                    if record is None or record[1] is not conn:
                        raise RuntimeError(f"Ray worker Vane session closed during task startup: {session_id}")
                database_identity = _query_worker_snapshot_database_identity(
                    connection_snapshot_query_id,
                    session_id=session_id,
                    effective_s3_config=effective_s3_config,
                    use_session_credentials=use_session_credentials,
                )
                cursor = self._get_snapshot_execution_cursor(
                    conn,
                    connection_snapshot_query_id,
                    database_identity=database_identity,
                )
                with self._session_connections_lock:
                    if self._shutdown_started:
                        raise RuntimeError("Ray worker runtime is shutting down")
                    if session_id in self._closed_session_ids:
                        raise RuntimeError(f"Ray worker Vane session is closed: {session_id}")
                    record = self._session_connections.get(session_id)
                    if record is None or record[1] is not conn:
                        raise RuntimeError(f"Ray worker Vane session closed during task startup: {session_id}")
                    query_admitted = self._register_native_cursor(
                        cursor,
                        native_query_id,
                        native_task_id,
                    )
                    cursor_registered = True
            if not query_admitted:
                if self._worker_native_query_is_closing(native_query_id):
                    raise RuntimeError(f"native query is closing: {native_query_id}")
                raise RuntimeError(f"native task is closing: {native_task_id}")
            if database_identity.has_static_extension("httpfs"):
                effective_s3_config = _configure_duckdb_s3(
                    cursor,
                    effective_s3_config,
                    use_session_credentials=use_session_credentials,
                )
            self._register_native_query_cleanup_context(
                native_query_id,
                plan,
                session_id,
                session_config,
                use_session_credentials=use_session_credentials,
            )
            _ray_worker_memory_log(
                "native_execute_start",
                **worker_log_context,
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
                runtime_context or None,
                effective_s3_config,
            )
            _ray_worker_memory_log(
                "native_execute_done",
                **worker_log_context,
                duration_ms=int((time.monotonic() - start) * 1000),
            )
            return result
        except BaseException as exc:
            _ray_worker_memory_log(
                "native_execute_error",
                **worker_log_context,
                duration_ms=int((time.monotonic() - start) * 1000),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        finally:
            if cursor is not None:
                try:
                    self._close_snapshot_execution_cursor(cursor)
                except Exception:
                    pass
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
        plan: Any,  # DistributedPhysicalPlan from _native.ray_cxx
        context: dict[str, str] | None,
        query_task_lease: dict[str, Any],
        exchange_sink_instance: dict[str, Any] | bytes | None = None,
        fte_scan_source_queues: dict[str, Any] | None = None,
        fte_exchange_source_queues: dict[str, Any] | None = None,
        dynamic_filter_domains: dict[str, Any] | None = None,
        native_progress_callback: Any | None = None,
        debug_context: dict[str, Any] | None = None,
        native_task_id: str = "",
    ) -> Any:
        """Run a plan on worker and return a Ray-serializable result tuple."""
        debug_context = dict(debug_context or {})
        worker_log_context = dict(debug_context)
        worker_log_context.update(_ray_worker_log_fields(self))

        copy_output_info = _copy_output_info_from_context(context)
        scan_task_map, exchange_source_task_map = _extract_native_task_maps_from_context(context)
        run_start = time.monotonic()
        _ray_worker_memory_log("run_plan_return_start", **worker_log_context)
        # Native execution, shuffle publication, and actor teardown are owned by
        # the execution query. The resource query can differ for nested
        # executions and is only the owner of the admission lease.
        query_id = str(query_task_lease.get("execution_query_id") or "").strip()
        native_task_id = str(native_task_id or "").strip()
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

        self._begin_worker_native_execution(query_id, native_task_id)
        try:
            begin_execution(query_id)
        except BaseException:
            self._end_worker_native_execution(query_id, native_task_id)
            raise

        def execute_native_task() -> Any:
            try:
                if self._worker_native_query_is_closing(query_id):
                    raise RuntimeError(f"native query is closing: {query_id}")
                if self._worker_native_task_is_closing(native_task_id):
                    raise RuntimeError(f"native task is closing: {native_task_id}")
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
                    native_task_id=native_task_id,
                )
            finally:
                try:
                    end_execution(query_id)
                finally:
                    self._end_worker_native_execution(query_id, native_task_id)

        try:
            native_future = asyncio.get_running_loop().run_in_executor(None, execute_native_task)
        except BaseException:
            try:
                end_execution(query_id)
            finally:
                self._end_worker_native_execution(query_id, native_task_id)
            raise
        result_list = await _await_future_with_owned_side_effects(native_future)
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
            **worker_log_context,
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
            **worker_log_context,
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
