# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import sys
from typing import Any

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RAYON_NUM_THREADS",
)

_TORCH_THREADS_CONFIGURED = False

RAY_ACTOR_THREAD_POLICY_ENV = "VANE_RAY_ACTOR_THREAD_POLICY"
_MANAGED_THREAD_POLICY = "managed"
_RAY_NATIVE_THREAD_POLICY = "ray_native"
_DEFAULT_RAY_ACTOR_THREAD_POLICY = _RAY_NATIVE_THREAD_POLICY
_RAY_ACTOR_THREAD_POLICIES = frozenset({_MANAGED_THREAD_POLICY, _RAY_NATIVE_THREAD_POLICY})


def _positive_int(value: Any, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return parsed


def payload_cpu_thread_count(payload: dict[str, Any] | None) -> int:
    value = (payload or {}).get("cpus")
    if value is None:
        return 1
    if type(value) not in (int, float):
        raise ValueError(f"payload.cpus must be a finite non-negative number, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"payload.cpus must be a finite non-negative number, got {value!r}")
    return max(math.floor(parsed), 1)


def ray_actor_thread_policy(payload: dict[str, Any] | None = None) -> str:
    """Resolve the thread policy for one Ray actor UDF.

    The coordinator reads the policy from the UDF payload.  The actor process
    reads the payload-owned marker injected into its runtime environment so
    user callable constructors observe the same policy before execution.
    """
    value = None if payload is None else payload.get("ray_actor_thread_policy")
    if value is None:
        value = os.environ.get(RAY_ACTOR_THREAD_POLICY_ENV, _DEFAULT_RAY_ACTOR_THREAD_POLICY)
    policy = str(value).strip().lower()
    if policy not in _RAY_ACTOR_THREAD_POLICIES:
        raise ValueError(f"Ray actor thread policy must be one of {sorted(_RAY_ACTOR_THREAD_POLICIES)}, got {value!r}")
    return policy


def ray_actor_uses_native_threads(payload: dict[str, Any] | None = None) -> bool:
    return ray_actor_thread_policy(payload) == _RAY_NATIVE_THREAD_POLICY


def worker_thread_env(payload: dict[str, Any] | None) -> dict[str, str]:
    thread_count = str(payload_cpu_thread_count(payload))
    env: dict[str, str] = {}
    effective_thread_count = os.environ.get("OMP_NUM_THREADS") or thread_count
    for name in _THREAD_ENV_VARS:
        if name not in os.environ:
            env[name] = effective_thread_count
    if "VANE_TORCH_NUM_THREADS" not in os.environ:
        env["VANE_TORCH_NUM_THREADS"] = effective_thread_count
    if "VANE_TORCH_INTEROP_THREADS" not in os.environ:
        env["VANE_TORCH_INTEROP_THREADS"] = "1"
    return env


def ray_actor_thread_env(payload: dict[str, Any] | None) -> dict[str, str]:
    """Return payload-owned thread defaults for a fresh Ray actor process.

    Ray coordinator code can itself run inside a Ray worker whose thread
    variables were set for the coordinator's CPU allocation.  Those inherited
    values must not leak into a child UDF actor: the actor's own ``cpus``
    resource declaration is the source of truth.  Explicit actor runtime-env
    values are merged on top by the actor-pool builder.
    """
    if ray_actor_uses_native_threads(payload):
        # Leave OMP and library-specific variables absent. Ray will derive
        # OMP_NUM_THREADS from the actor's assigned CPU resource, while
        # PyTorch retains its platform-native inter-op thread-pool default.
        return {}

    thread_count = str(payload_cpu_thread_count(payload))
    env = dict.fromkeys(_THREAD_ENV_VARS, thread_count)
    env["VANE_TORCH_NUM_THREADS"] = thread_count
    env["VANE_TORCH_INTEROP_THREADS"] = "1"
    return env


def configure_ray_actor_loaded_torch_threads(payload: dict[str, Any] | None) -> bool:
    """Configure managed actors and leave ray-native actors untouched."""
    if ray_actor_uses_native_threads(payload):
        return False
    return configure_loaded_torch_threads()


def configure_loaded_torch_threads() -> bool:
    global _TORCH_THREADS_CONFIGURED
    if _TORCH_THREADS_CONFIGURED:
        return False

    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return False

    torch_threads = os.environ.get("VANE_TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS") or "1"
    interop_threads = os.environ.get("VANE_TORCH_INTEROP_THREADS") or "1"
    target_threads = _positive_int(torch_threads, name="VANE_TORCH_NUM_THREADS")
    target_interop_threads = _positive_int(interop_threads, name="VANE_TORCH_INTEROP_THREADS")
    if int(torch_module.get_num_threads()) != target_threads:
        torch_module.set_num_threads(target_threads)
    if int(torch_module.get_num_interop_threads()) != target_interop_threads:
        torch_module.set_num_interop_threads(target_interop_threads)
    _TORCH_THREADS_CONFIGURED = True
    return True
