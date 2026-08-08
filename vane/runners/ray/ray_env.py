# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping
from typing import Any

from vane._vane_session import ensure_vane_session_dir

_JOB_DUCKDB_ENV_KEYS = (
    "DUCKDB_DISTRIBUTED_DEBUG",
    "DUCKDB_FLIGHT_PORT",
    "DUCKDB_SHUFFLE_DIRS",
)

_NODE_LOCAL_RUNTIME_ENV_KEYS = frozenset({"VANE_FLIGHT_ADVERTISE_HOST"})

_SENSITIVE_ENV_MARKERS = (
    "ACCESS_KEY",
    "API_KEY",
    "AUTH",
    "CREDENTIAL",
    "ENDPOINT",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "SESSION",
    "TOKEN",
    "URL",
)

_SESSION_ENV_PREFIXES = (
    "AWS_",
    "DUCKDB_",
    "S3FS_",
)

_SESSION_ENV_PAYLOAD_KEY = "VANE_INTERNAL_SESSION_ENV_PAYLOAD"


def reject_node_local_ray_runtime_env(runtime_context: Any) -> None:
    """Reject node-local settings inherited through a Ray runtime environment."""
    runtime_env = getattr(runtime_context, "runtime_env", None)
    if not isinstance(runtime_env, Mapping):
        return
    env_vars = runtime_env.get("env_vars")
    if not isinstance(env_vars, Mapping):
        return

    inherited_keys = sorted(key for key in _NODE_LOCAL_RUNTIME_ENV_KEYS if key in env_vars)
    if inherited_keys:
        keys = ", ".join(inherited_keys)
        raise RuntimeError(
            f"{keys} is node-local and must not be set in a Ray Job or actor runtime_env; "
            "configure it in each Ray worker node's environment instead"
        )


def _is_session_environment_key(key: str) -> bool:
    normalized_key = str(key).upper()
    if normalized_key in _JOB_DUCKDB_ENV_KEYS:
        return False
    if normalized_key.startswith(_SESSION_ENV_PREFIXES):
        return True
    return normalized_key.startswith("VANE_") and any(marker in normalized_key for marker in _SENSITIVE_ENV_MARKERS)


def session_environment_overrides(config: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in config.items()
        if _is_session_environment_key(str(key)) and str(key) not in {"VANE_SESSION_DIR", _SESSION_ENV_PAYLOAD_KEY}
    }


def build_explicit_session_process_env(
    config: Mapping[str, Any],
    *,
    base_env: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Build a child-process environment containing exactly one session context."""
    source = os.environ if base_env is None else base_env
    environment = {
        str(key): str(value)
        for key, value in source.items()
        if not _is_session_environment_key(str(key)) or str(key) == "VANE_SESSION_DIR"
    }
    environment.update(session_environment_overrides(config))
    return environment


def scrub_shared_runtime_session_env() -> None:
    """Remove inherited connection/session values before shared runtime use."""
    for key in tuple(os.environ):
        if _is_session_environment_key(key) and key != "VANE_SESSION_DIR":
            os.environ.pop(key, None)


def build_session_runtime_env_vars(config: Mapping[str, Any]) -> dict[str, str]:
    """Encode one explicit session context for a dedicated task/actor process."""
    overrides = session_environment_overrides(config)
    payload = json.dumps(overrides, sort_keys=True, separators=(",", ":")).encode()
    return {
        _SESSION_ENV_PAYLOAD_KEY: base64.urlsafe_b64encode(payload).decode(),
    }


def install_explicit_session_runtime_env() -> None:
    """Scrub inherited values, then install exactly one encoded session context."""
    encoded_payload = os.environ.pop(_SESSION_ENV_PAYLOAD_KEY, "")
    if not encoded_payload:
        raise RuntimeError("Vane session runtime environment is missing its explicit payload")
    try:
        decoded = base64.urlsafe_b64decode(encoded_payload.encode())
        raw_overrides = json.loads(decoded)
    except Exception as exc:
        raise RuntimeError("Vane session runtime environment payload is invalid") from exc
    if not isinstance(raw_overrides, dict):
        raise RuntimeError("Vane session runtime environment payload must contain a mapping")
    overrides = session_environment_overrides(raw_overrides)
    if overrides != raw_overrides:
        raise RuntimeError("Vane session runtime environment payload contains unmanaged keys")
    scrub_shared_runtime_session_env()
    os.environ.update(overrides)


def collect_vane_env_overrides() -> dict[str, str]:
    """Collect immutable Ray Job runtime environment.

    Connection/session credentials are carried in the logical-plan session
    snapshot and must never be installed in a shared runtime process.
    Node-local service addresses remain in each worker's process environment
    rather than inheriting through Ray runtime environments.
    """
    ensure_vane_session_dir()

    overrides = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("VANE_") and key not in _NODE_LOCAL_RUNTIME_ENV_KEYS and not _is_session_environment_key(key)
    }
    for key in ("PYTHONPATH", "PYTHONWARNINGS"):
        value = os.environ.get(key)
        if value:
            overrides[key] = value
    overrides["VANE_SESSION_DIR"] = ensure_vane_session_dir()
    for key in _JOB_DUCKDB_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None:
            overrides[key] = value
    return overrides
