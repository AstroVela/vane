# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import ray

from vane._ray_cxx import require_ray_cxx_attr
from vane.runners.ray.fragment_worker_client import RayWorkerActorHandle
from vane.runners.ray.fte_fragment_scheduler import (
    _collect_vane_env_overrides,
)
from vane.runners.ray.fte_scheduler_config import _is_ray_worker_context
from vane.runners.ray.safe_get import resolve_object_refs_blocking
from vane.runners.ray.worker import RayWorkerActor
from vane.runners.ray.worker_memory import build_ray_node_memory_layout

RayWorkerRuntime = require_ray_cxx_attr(
    "RayWorkerRuntime",
    hint="Ensure the C++ ray extension is built and importable in the driver process.",
)


def start_ray_workers(existing_worker_ids: list[str]) -> list[RayWorkerRuntime]:
    env_overrides = _collect_vane_env_overrides()
    actors = []
    for node in ray.nodes():
        base_worker_id = str(node.get("NodeManagerAddress") or node.get("NodeID") or "")
        if (
            "Resources" in node
            and "CPU" in node["Resources"]
            and "memory" in node["Resources"]
            and node["Resources"]["CPU"] > 0
            and node["Resources"]["memory"] > 0
            and base_worker_id
        ):
            worker_id = base_worker_id
            if worker_id in existing_worker_ids:
                continue
            worker_env = dict(env_overrides)
            worker_env["VANE_WORKER_ID"] = worker_id
            worker_env["VANE_WORKER_INDEX"] = "0"
            memory_layout = build_ray_node_memory_layout(int(node["Resources"]["memory"]))
            # max_concurrency limits how many control/execute RPCs can queue
            # inside the actor. FTE backpressure is handled by task
            # status/split-queue feedback and the shared DuckDB TaskScheduler.
            _actor_max_conc = int(os.environ.get("VANE_RAY_ACTOR_MAX_CONCURRENCY", "256"))
            actor = RayWorkerActor.options(  # type: ignore[attr-defined]
                max_concurrency=_actor_max_conc,
                memory=(memory_layout.worker_duckdb_memory_bytes + memory_layout.runtime_reserve_bytes),
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"],
                    soft=False,
                ),
            ).remote(
                num_cpus=int(node["Resources"]["CPU"]),
                num_gpus=int(node["Resources"].get("GPU", 0)),
                duckdb_memory_bytes=memory_layout.worker_duckdb_memory_bytes,
                task_heap_capacity_bytes=memory_layout.task_heap_capacity_bytes,
                env_overrides=worker_env,
            )
            actors.append((node, worker_id, actor))

    # Pre-warm: wait for all actors to be fully initialized before returning.
    # This absorbs the ~2.5s actor cold-start so the first task dispatch is fast.
    # Matches upstream Vane's start_ray_workers() pattern.
    ACTOR_STARTUP_TIMEOUT = 120
    warmup_refs = [actor.install_env_overrides.remote(None) for _, _, actor in actors]
    # Nested distributed execution may call start_ray_workers() from inside a
    # Ray actor. Let actor startup proceed asynchronously there and rely on the
    # first task submission for backpressure.
    if not _is_ray_worker_context():
        try:
            resolve_object_refs_blocking(warmup_refs, timeout=ACTOR_STARTUP_TIMEOUT)
        except ray.exceptions.GetTimeoutError:
            raise RuntimeError(f"Failed to warm up Worker actors within {ACTOR_STARTUP_TIMEOUT}s")

    handles = []
    for node, worker_id, actor in actors:
        actor_handle = RayWorkerActorHandle(
            actor,
            worker_id=worker_id,
            node_id=str(node["NodeID"]),
            memory_capacity_bytes=build_ray_node_memory_layout(
                int(node["Resources"]["memory"])
            ).task_heap_capacity_bytes,
        )
        handles.append(
            RayWorkerRuntime(
                worker_id,
                actor_handle,
                int(node["Resources"]["CPU"]),
                int(node["Resources"].get("GPU", 0)),
                int(node["Resources"]["memory"]),
            )
        )

    return handles


def try_autoscale(bundles: list[dict[str, int]]) -> None:
    from ray.autoscaler.sdk import request_resources

    request_resources(
        bundles=bundles,
    )


__all__ = [name for name in globals() if not name.startswith("__")]
