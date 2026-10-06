# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

RAY_QUERY_RUNTIME_ACTOR_NAMESPACE = "vane"
RAY_QUERY_RUNTIME_ACTOR_NAME_PREFIX = "vane-query-runtime-"

RAY_ACTOR_QUERY_ID_ENV = "VANE_INTERNAL_RAY_ACTOR_QUERY_ID"
RAY_ACTOR_RESOURCE_UNIT_ID_ENV = "VANE_INTERNAL_RAY_ACTOR_RESOURCE_UNIT_ID"
RAY_ACTOR_INDEX_ENV = "VANE_INTERNAL_RAY_ACTOR_INDEX"
RAY_ACTOR_POOL_NONCE_ENV = "VANE_INTERNAL_RAY_ACTOR_POOL_NONCE"
RAY_ACTOR_GENERATION_CAPABILITY_ENV = "VANE_INTERNAL_RAY_ACTOR_GENERATION_CAPABILITY"


def ray_runtime_job_id(runtime_context: Any) -> str:
    job_id_obj = runtime_context.get_job_id()
    job_id_hex = getattr(job_id_obj, "hex", None)
    job_id = str(job_id_hex() if callable(job_id_hex) else job_id_obj).strip()
    if not job_id:
        raise RuntimeError("Ray runtime context returned an empty job identity")
    return job_id


def query_runtime_actor_name(job_id: str) -> str:
    job_key = str(job_id or "").strip()
    if not job_key:
        raise ValueError("Ray query runtime actor name requires a job identity")
    return f"{RAY_QUERY_RUNTIME_ACTOR_NAME_PREFIX}{job_key}"


__all__ = [
    "RAY_ACTOR_GENERATION_CAPABILITY_ENV",
    "RAY_ACTOR_INDEX_ENV",
    "RAY_ACTOR_POOL_NONCE_ENV",
    "RAY_ACTOR_QUERY_ID_ENV",
    "RAY_ACTOR_RESOURCE_UNIT_ID_ENV",
    "RAY_QUERY_RUNTIME_ACTOR_NAME_PREFIX",
    "RAY_QUERY_RUNTIME_ACTOR_NAMESPACE",
    "query_runtime_actor_name",
    "ray_runtime_job_id",
]
