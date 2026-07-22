# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from vane.runners.ray.safe_get import (
    configured_ray_get_timeout_s,
    resolve_object_refs_blocking,
)

_DEFAULT_RAY_ACTOR_INIT_TIMEOUT_S = 60.0

from vane.execution.udf_ray_config import (
    MAX_ACTOR_RESTARTS,
    MAX_ACTOR_TASK_RETRIES,
)
from vane.execution.udf_threading import (
    RAY_ACTOR_THREAD_POLICY_ENV,
    ray_actor_thread_env,
    ray_actor_thread_policy,
)


def _with_actor_thread_env(
    runtime_env: dict[str, Any] | None,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge CPU-allocation defaults below explicit actor environment values."""
    merged_runtime_env = dict(runtime_env or {})
    env_vars = ray_actor_thread_env(payload)
    env_vars.update(merged_runtime_env.get("env_vars") or {})
    # The payload owns this marker.  Besides preventing a job-level override,
    # it lets the user callable constructor observe the actor-specific policy.
    env_vars[RAY_ACTOR_THREAD_POLICY_ENV] = ray_actor_thread_policy(payload)
    merged_runtime_env["env_vars"] = env_vars
    return merged_runtime_env


def _validate_stateful_actor_pool_contract(
    payload: dict[str, Any] | None,
    concurrency: int,
    *,
    max_restarts: int | None = None,
    max_task_retries: int | None = None,
) -> None:
    if not payload or not payload.get("stateful"):
        return
    actor_number = payload.get("actor_number")
    if type(actor_number) is not int or actor_number != 1 or concurrency != 1:
        raise ValueError(
            "actor_number must be exactly 1 for stateful vane.cls UDFs; multi-actor state semantics are not defined"
        )
    if max_restarts is not None and max_restarts != 0:
        raise ValueError("stateful UDF actor pools require max_restarts=0")
    if max_task_retries is not None and max_task_retries != 0:
        raise ValueError("stateful UDF actor pools require max_task_retries=0")


class UDFActorPoolBase:
    def __init__(
        self,
        payload: dict[str, Any],
        concurrency: int,
        gpus_per_actor: float,
        actor_node_ids: list[str],
        ray_options: dict[str, Any] | None = None,
        max_restarts: int = MAX_ACTOR_RESTARTS,
        max_task_retries: int = MAX_ACTOR_TASK_RETRIES,
    ) -> None:
        _validate_stateful_actor_pool_contract(
            payload,
            concurrency,
            max_restarts=max_restarts,
            max_task_retries=max_task_retries,
        )
        Actor = self._actor_class(max_restarts, max_task_retries)
        options = dict(ray_options or {})
        if "scheduling_strategy" in options:
            raise ValueError("UDF actor scheduling_strategy is owned by the query coordinator")
        options["num_cpus"] = self._resolve_actor_num_cpus(payload)
        options["num_gpus"] = gpus_per_actor
        options["memory"] = self._resolve_actor_memory_bytes(payload)
        runtime_env = _with_actor_thread_env(
            self._build_actor_runtime_env(options),
            payload,
        )
        options["runtime_env"] = runtime_env
        normalized_node_ids = self._normalize_actor_node_ids(
            actor_node_ids,
            expected_count=concurrency,
        )
        if normalized_node_ids is None or any(not str(node_id).strip() for node_id in normalized_node_ids):
            raise ValueError("every UDF actor requires a coordinator-selected Ray node_id")
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        actor_options = []
        for node_id in normalized_node_ids:
            actor_options.append(
                {
                    **options,
                    "scheduling_strategy": NodeAffinitySchedulingStrategy(
                        node_id=str(node_id),
                        soft=False,
                    ),
                }
            )
        self._owns_actors = True
        self.actor_node_ids: list[str] | None = normalized_node_ids
        self._payload = payload

        import ray

        payload_ref = ray.put(payload)
        self._payload_ref = payload_ref

        self.actors = [Actor.options(**actor_options[idx]).remote() for idx in range(concurrency)]

        self._init_refs = [a.init_payload.remote(payload_ref) for a in self.actors]
        self._confirmed_ready: set[int] = set()

    @staticmethod
    def _actor_class(max_restarts: int, max_task_retries: int):
        raise NotImplementedError

    @staticmethod
    def _resolve_actor_num_cpus(payload: dict[str, Any]) -> float:
        raise NotImplementedError

    @staticmethod
    def _resolve_actor_memory_bytes(payload: dict[str, Any]) -> int:
        raise NotImplementedError

    @staticmethod
    def _build_actor_runtime_env(ray_options: dict[str, Any] | None) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _normalize_actor_node_ids(
        node_ids: list[str] | None,
        *,
        expected_count: int,
    ) -> list[str] | None:
        raise NotImplementedError

    @classmethod
    def _from_handles(
        cls,
        actors: list[Any],
        *,
        payload: dict[str, Any] | None = None,
        actor_node_ids: list[str] | None = None,
        actor_dispatch_indices: list[int] | tuple[int, ...] | set[int] | None = None,
    ) -> UDFActorPoolBase:
        _validate_stateful_actor_pool_contract(payload, len(actors))
        if actor_dispatch_indices is None:
            raise ValueError("pre-created Ray UDF actor handles require explicit actor_dispatch_indices")
        instance = cls.__new__(cls)
        instance.actors = actors
        instance._owns_actors = False
        parsed_dispatch_indices = list(actor_dispatch_indices)
        if any(type(idx) is not int for idx in parsed_dispatch_indices):
            raise ValueError("actor_dispatch_indices must contain integers")
        if len(set(parsed_dispatch_indices)) != len(parsed_dispatch_indices):
            raise ValueError("actor_dispatch_indices must not contain duplicates")
        invalid = [idx for idx in parsed_dispatch_indices if idx < 0 or idx >= len(actors)]
        if invalid:
            raise ValueError(f"actor_dispatch_indices contains out-of-range indices: {invalid}")
        instance._confirmed_ready = set(parsed_dispatch_indices)
        instance._payload = payload
        instance._payload_ref = None
        instance._init_refs = []
        instance.actor_node_ids = cls._normalize_actor_node_ids(actor_node_ids, expected_count=len(actors))
        return instance

    def shutdown(self, *, kill: bool = False) -> None:
        if not self._owns_actors:
            return

        import ray

        for actor in self.actors:
            ray.kill(actor, no_restart=True)
        self.actors = []


def apply_actor_node_options(
    actors: UDFActorPoolBase,
    *,
    options: dict[str, Any],
    normalize_actor_node_ids: Callable[..., list[str] | None],
) -> UDFActorPoolBase:
    normalized_node_ids = normalize_actor_node_ids(
        options.get("actor_node_ids"),
        expected_count=len(actors.actors),
    )
    actors.actor_node_ids = normalized_node_ids
    return actors


def requires_actor_pool(payload: dict[str, Any]) -> bool:
    return str(payload.get("execution_backend") or "").strip().lower() == "ray_actor"


def _positive_float_env(name: str, default: float | None = None) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    value = float(raw)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _actor_init_timeout_s() -> float | None:
    return configured_ray_get_timeout_s(
        _positive_float_env("VANE_RAY_ACTOR_INIT_TIMEOUT_S", _DEFAULT_RAY_ACTOR_INIT_TIMEOUT_S)
    )


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    return type(exc).__name__ in {"GetTimeoutError", "TimeoutError"}


def _shutdown_owned_actors(ray: Any, actors_obj: Any) -> None:
    if not bool(getattr(actors_obj, "_owns_actors", False)):
        return
    actors = list(getattr(actors_obj, "actors", []))
    for actor in actors:
        ray.kill(actor, no_restart=True)
    actors_obj.actors = []


def _resolve_actor_pool_init_refs(ray: Any, actors_obj: Any) -> None:
    init_refs = getattr(actors_obj, "_init_refs", None)
    if init_refs is None:
        raise RuntimeError("pre-created UDF actor pool is missing init readiness refs")
    refs = list(init_refs)
    if not refs:
        return
    actors = list(getattr(actors_obj, "actors", []))
    if len(refs) != len(actors) or any(ref is None for ref in refs):
        raise RuntimeError("pre-created UDF actor pool has incomplete init readiness refs")
    timeout_s: float | None = None
    try:
        timeout_s = _actor_init_timeout_s()
        if timeout_s is None:
            resolve_object_refs_blocking(refs)
        else:
            resolve_object_refs_blocking(refs, timeout=timeout_s)
    except Exception as exc:
        _shutdown_owned_actors(ray, actors_obj)
        if _is_timeout_error(exc):
            timeout_desc = "query deadline" if timeout_s is None else f"{timeout_s:.3f}s"
            raise RuntimeError(
                "UDF actor pool initialization timed out "
                f"after {timeout_desc} while waiting for {len(refs)} actor init refs"
            ) from exc
        raise RuntimeError(f"UDF actor pool initialization failed: {type(exc).__name__}: {exc}") from exc
    actors_obj._confirmed_ready.update(range(len(actors)))


def ensure_actor_pools_for_plan(
    plan: Any,
    conn: Any = None,
    *,
    actor_node_ids_by_stage: dict[str, tuple[str, ...]],
    actor_pool_cls: type[UDFActorPoolBase],
    is_vane_worker_process: Callable[[], bool],
    requires_actor_pool_fn: Callable[[dict[str, Any]], bool],
    normalize_actor_pool_payload: Callable[..., dict[str, Any]],
    payload_num_gpus: Callable[[dict[str, Any]], float],
    required_positive_int: Callable[[dict[str, Any], str], int],
    resolve_actor_num_cpus: Callable[[dict[str, Any]], float],
    build_udf_executor_options: Callable[..., dict[str, Any]],
) -> tuple[list[UDFActorPoolBase], dict[str, Any]]:
    udf_nodes = plan.collect_udf_nodes(conn=conn)
    return ensure_actor_pools_for_nodes(
        udf_nodes,
        actor_node_ids_by_stage=actor_node_ids_by_stage,
        set_handles=lambda actor_handles_map: plan.set_udf_actor_handles(actor_handles_map, conn=conn),
        actor_pool_cls=actor_pool_cls,
        is_vane_worker_process=is_vane_worker_process,
        requires_actor_pool_fn=requires_actor_pool_fn,
        normalize_actor_pool_payload=normalize_actor_pool_payload,
        payload_num_gpus=payload_num_gpus,
        required_positive_int=required_positive_int,
        resolve_actor_num_cpus=resolve_actor_num_cpus,
        build_udf_executor_options=build_udf_executor_options,
    )


def prepare_actor_pools_for_plan(
    plan: Any,
    conn: Any = None,
    *,
    actor_node_ids_by_stage: dict[str, tuple[str, ...]],
    actor_pool_cls: type[UDFActorPoolBase],
    is_vane_worker_process: Callable[[], bool],
    requires_actor_pool_fn: Callable[[dict[str, Any]], bool],
    normalize_actor_pool_payload: Callable[..., dict[str, Any]],
    payload_num_gpus: Callable[[dict[str, Any]], float],
    required_positive_int: Callable[[dict[str, Any], str], int],
    resolve_actor_num_cpus: Callable[[dict[str, Any]], float],
    build_udf_executor_options: Callable[..., dict[str, Any]],
) -> tuple[list[UDFActorPoolBase], dict[str, Any]]:
    """Create actors and publish immutable handles without waiting for init.

    The query coordinator keeps every Ray-actor QRM stage closed until
    :func:`wait_for_actor_pools_ready` succeeds.  Publishing the handles first
    lets native fragment executors initialize and expose their real pipeline
    topology while expensive user-model initialization is still running.
    """
    udf_nodes = plan.collect_udf_nodes(conn=conn)
    return _create_actor_pools_for_nodes(
        udf_nodes,
        actor_node_ids_by_stage=actor_node_ids_by_stage,
        set_handles=lambda actor_handles_map: plan.set_udf_actor_handles(actor_handles_map, conn=conn),
        actor_pool_cls=actor_pool_cls,
        is_vane_worker_process=is_vane_worker_process,
        requires_actor_pool_fn=requires_actor_pool_fn,
        normalize_actor_pool_payload=normalize_actor_pool_payload,
        payload_num_gpus=payload_num_gpus,
        required_positive_int=required_positive_int,
        resolve_actor_num_cpus=resolve_actor_num_cpus,
        build_udf_executor_options=build_udf_executor_options,
        wait_for_ready=False,
    )


def ensure_actor_pools_for_nodes(
    udf_nodes: Any,
    *,
    actor_node_ids_by_stage: dict[str, tuple[str, ...]],
    set_handles: Callable[[dict[str, Any]], None] | None = None,
    actor_pool_cls: type[UDFActorPoolBase],
    is_vane_worker_process: Callable[[], bool],
    requires_actor_pool_fn: Callable[[dict[str, Any]], bool],
    normalize_actor_pool_payload: Callable[..., dict[str, Any]],
    payload_num_gpus: Callable[[dict[str, Any]], float],
    required_positive_int: Callable[[dict[str, Any], str], int],
    resolve_actor_num_cpus: Callable[[dict[str, Any]], float],
    build_udf_executor_options: Callable[..., dict[str, Any]],
) -> tuple[list[UDFActorPoolBase], dict[str, Any]]:
    return _create_actor_pools_for_nodes(
        udf_nodes,
        actor_node_ids_by_stage=actor_node_ids_by_stage,
        set_handles=set_handles,
        actor_pool_cls=actor_pool_cls,
        is_vane_worker_process=is_vane_worker_process,
        requires_actor_pool_fn=requires_actor_pool_fn,
        normalize_actor_pool_payload=normalize_actor_pool_payload,
        payload_num_gpus=payload_num_gpus,
        required_positive_int=required_positive_int,
        resolve_actor_num_cpus=resolve_actor_num_cpus,
        build_udf_executor_options=build_udf_executor_options,
        wait_for_ready=True,
    )


def _create_actor_pools_for_nodes(
    udf_nodes: Any,
    *,
    actor_node_ids_by_stage: dict[str, tuple[str, ...]],
    set_handles: Callable[[dict[str, Any]], None] | None,
    actor_pool_cls: type[UDFActorPoolBase],
    is_vane_worker_process: Callable[[], bool],
    requires_actor_pool_fn: Callable[[dict[str, Any]], bool],
    normalize_actor_pool_payload: Callable[..., dict[str, Any]],
    payload_num_gpus: Callable[[dict[str, Any]], float],
    required_positive_int: Callable[[dict[str, Any], str], int],
    resolve_actor_num_cpus: Callable[[dict[str, Any]], float],
    build_udf_executor_options: Callable[..., dict[str, Any]],
    wait_for_ready: bool,
) -> tuple[list[UDFActorPoolBase], dict[str, Any]]:
    if is_vane_worker_process():
        return [], {}

    if not udf_nodes:
        return [], {}

    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray actor UDF creation requires an initialized RayRunner runtime")

    created: list[UDFActorPoolBase] = []
    actor_handles_map: dict[str, Any] = {}

    try:
        for node in udf_nodes:
            raw_payload = node.get("payload") or {}
            if not requires_actor_pool_fn(raw_payload):
                continue
            payload = normalize_actor_pool_payload(raw_payload)
            gpus = payload_num_gpus(payload)

            if not requires_actor_pool_fn(payload):
                continue

            node_id = str(node["node_id"])
            stage_id = str(payload.get("stage_id") or "").strip()
            if not stage_id:
                raise RuntimeError(f"Ray actor UDF node {node_id} is missing stage_id")
            concurrency = required_positive_int(node, "actor_pool_size")
            _validate_stateful_actor_pool_contract(payload, concurrency)
            assigned_node_ids = tuple(actor_node_ids_by_stage.get(stage_id, ()))
            if len(assigned_node_ids) != concurrency:
                raise RuntimeError(
                    f"Ray actor UDF stage {stage_id} requires {concurrency} coordinator placements, "
                    f"got {len(assigned_node_ids)}"
                )
            cpus = resolve_actor_num_cpus(payload)
            ray_options = {"num_cpus": cpus}

            stateful_options: dict[str, int] = {}
            if payload.get("stateful"):
                stateful_options = {
                    "max_restarts": 0,
                    "max_task_retries": 0,
                }

            actors_obj = actor_pool_cls(
                payload=payload,
                concurrency=concurrency,
                gpus_per_actor=gpus,
                actor_node_ids=list(assigned_node_ids),
                ray_options=ray_options,
                **stateful_options,
            )
            created.append(actors_obj)
            if wait_for_ready:
                _resolve_actor_pool_init_refs(ray, actors_obj)
            actor_node_ids = list(assigned_node_ids)
            actor_dispatch_indices = set(range(len(actors_obj.actors)))
            actor_handles_map[node_id] = build_udf_executor_options(
                actor_handles=list(actors_obj.actors),
                actor_node_ids=actor_node_ids,
                actor_dispatch_indices=actor_dispatch_indices,
            )

        if actor_handles_map and set_handles is not None:
            set_handles(actor_handles_map)
    except BaseException as execution_error:
        cleanup_errors = []
        for actors_obj in reversed(created):
            try:
                actors_obj.shutdown()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            raise RuntimeError(
                f"UDF actor pool creation failed and cleanup also failed: {cleanup_errors[0]}"
            ) from execution_error
        raise

    return created, actor_handles_map


def wait_for_actor_pools_ready(actor_pools: list[UDFActorPoolBase]) -> None:
    """Resolve all staged actor init refs before QRM actor admission opens."""
    if not actor_pools:
        return

    import ray

    try:
        for actors_obj in actor_pools:
            _resolve_actor_pool_init_refs(ray, actors_obj)
    except BaseException as readiness_error:
        cleanup_errors = []
        for actors_obj in reversed(actor_pools):
            try:
                actors_obj.shutdown()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            raise RuntimeError(
                f"UDF actor readiness failed and cleanup also failed: {cleanup_errors[0]}"
            ) from readiness_error
        raise


__all__ = [name for name in globals() if not name.startswith("__")]
