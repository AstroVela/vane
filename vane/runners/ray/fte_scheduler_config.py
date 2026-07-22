# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os

import ray


def _fte_control_rpc_max_attempts() -> int:
    raw = os.getenv("VANE_FTE_CONTROL_RPC_MAX_ATTEMPTS", "3")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 3
    return max(1, value)


def _fte_control_rpc_initial_backoff_s() -> float:
    raw = os.getenv("VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S", "0.05")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.05
    return min(max(0.0, value), 5.0)


def _fte_control_rpc_timeout_s() -> float:
    value = float(os.getenv("VANE_FTE_CONTROL_RPC_TIMEOUT_S", "30"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("VANE_FTE_CONTROL_RPC_TIMEOUT_S must be finite and > 0")
    return value


def _is_ray_worker_context() -> bool:
    try:
        from ray._private import worker as ray_worker
    except Exception:
        return False
    try:
        return ray.is_initialized() and ray_worker.global_worker.mode == ray_worker.WORKER_MODE
    except Exception:
        return False


def _fte_event_source_high_watermark() -> int:
    raw = os.getenv("VANE_FTE_EVENT_SOURCE_HIGH_WATERMARK", "64")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 64


def _fte_event_source_low_watermark(high_watermark: int) -> int:
    raw = os.getenv("VANE_FTE_EVENT_SOURCE_LOW_WATERMARK", "")
    if not str(raw).strip():
        return max(0, int(high_watermark) // 2)
    try:
        return max(0, min(int(raw), int(high_watermark) - 1))
    except (TypeError, ValueError):
        return max(0, int(high_watermark) // 2)


def _fte_event_source_chunk_size() -> int:
    raw = os.getenv("VANE_FTE_EVENT_SOURCE_CHUNK_SIZE", "8")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 8


def _ray_fragment_plan_cache_session_key() -> str:
    if not ray.is_initialized():
        return "uninitialized"
    try:
        from ray._private import worker as ray_worker

        cluster_id, job_id = ray_worker.global_worker.current_cluster_and_job
        return f"{cluster_id}:{job_id}"
    except Exception:
        pass
    try:
        return str(ray.get_runtime_context().get_node_id())
    except Exception:
        return "unknown"


def _fte_exhausted_node_wait_period_s() -> float:
    raw = os.getenv("VANE_FTE_EXHAUSTED_NODE_WAIT_PERIOD_S", "120")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 120.0


def _fte_allowed_no_matching_node_period_s() -> float:
    raw = os.getenv("VANE_FTE_ALLOWED_NO_MATCHING_NODE_PERIOD_S", "600")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 600.0


def _fte_retry_initial_delay_s() -> float:
    raw = os.getenv("VANE_FTE_RETRY_INITIAL_DELAY_S", "10")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 10.0


def _fte_retry_max_delay_s() -> float:
    raw = os.getenv("VANE_FTE_RETRY_MAX_DELAY_S", "60")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 60.0


def _fte_retry_delay_scale_factor() -> float:
    raw = os.getenv("VANE_FTE_RETRY_DELAY_SCALE_FACTOR", "2.0")
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 2.0


def _chaos_host_loss_worker_indices() -> set[str]:
    if os.getenv("VANE_FTE_CHAOS_FAIL_HOST_ON_WORKER_LOSS", "").strip().lower() in (
        "",
        "0",
        "false",
        "no",
        "off",
    ):
        return set()
    return {index.strip() for index in os.getenv("VANE_FTE_CHAOS_KILL_WORKER_INDEX", "").split(",") if index.strip()}
