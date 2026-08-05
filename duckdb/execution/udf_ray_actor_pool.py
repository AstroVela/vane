# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import secrets
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from duckdb.execution.udf_ray_config import (
    MAX_ACTOR_RESTARTS,
    MAX_ACTOR_TASK_RETRIES,
)
from duckdb.execution.udf_threading import (
    RAY_ACTOR_THREAD_POLICY_ENV,
    ray_actor_thread_env,
    ray_actor_thread_policy,
)
from duckdb.runners.ray.query_runtime_protocol import (
    RAY_ACTOR_GENERATION_CAPABILITY_ENV,
    RAY_ACTOR_INDEX_ENV,
    RAY_ACTOR_POOL_NONCE_ENV,
    RAY_ACTOR_QUERY_ID_ENV,
    RAY_ACTOR_RESOURCE_UNIT_ID_ENV,
)
from duckdb.runners.ray.ray_env import build_session_runtime_env_vars
from duckdb.runners.ray.safe_get import (
    configured_ray_get_timeout_s,
    resolve_object_refs_blocking,
)


class _OwnedUDFActorPoolsError(RuntimeError):
    """Carry actor pools whose teardown must remain retryable by the driver."""

    def __init__(
        self,
        message: str,
        *,
        owned_actor_pools: list[Any],
        creation_error: BaseException,
    ) -> None:
        super().__init__(message)
        self.owned_actor_pools = list(owned_actor_pools)
        self.creation_error = creation_error


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
            raise ValueError("UDF actor scheduling_strategy must leave placement to Ray Core")
        options["num_cpus"] = self._resolve_actor_num_cpus(payload)
        options["num_gpus"] = gpus_per_actor
        options.pop("memory", None)
        memory_bytes = int(self._resolve_actor_memory_bytes(payload))
        if memory_bytes < 0:
            raise ValueError("memory_bytes must be non-negative")
        if memory_bytes > 0:
            options["memory"] = memory_bytes
        runtime_env = _with_actor_thread_env(
            self._build_actor_runtime_env(options),
            payload,
        )
        options["runtime_env"] = runtime_env
        # Submit the real actor requests without a Vane-selected node. Ray Core
        # owns placement, pending demand, autoscaling, and reconstruction.
        actor_options = [dict(options) for _ in range(concurrency)]
        for actor_index, actor_option in enumerate(actor_options):
            runtime_env = dict(actor_option.get("runtime_env") or {})
            env_vars = dict(runtime_env.get("env_vars") or {})
            env_vars[RAY_ACTOR_INDEX_ENV] = str(actor_index)
            runtime_env["env_vars"] = env_vars
            actor_option["runtime_env"] = runtime_env
        self._owns_actors = True
        self.actor_node_ids: list[str] = [""] * concurrency
        self._payload = payload

        import ray

        payload_ref = ray.put(payload)
        self._payload_ref = payload_ref

        self.actors: list[Any] = []
        self._init_refs: list[Any] = []
        try:
            for actor_index in range(concurrency):
                self.actors.append(Actor.options(**actor_options[actor_index]).remote())
            for actor in self.actors:
                self._init_refs.append(actor.init_payload.remote(payload_ref))
        except BaseException as creation_error:
            try:
                self.shutdown()
            except BaseException as cleanup_error:
                raise _OwnedUDFActorPoolsError(
                    "UDF actor pool creation failed and partial actor cleanup also failed: "
                    f"creation={type(creation_error).__name__}: {creation_error}; "
                    f"cleanup={type(cleanup_error).__name__}: {cleanup_error}",
                    owned_actor_pools=[self],
                    creation_error=creation_error,
                ) from creation_error
            raise
        self._confirmed_ready: set[int] = set()
        self._vane_retired = False
        self._vane_location_nonce = ""
        self._vane_readiness_futures: list[Any] = []
        self._vane_init_errors: dict[int, BaseException] = {}

    @staticmethod
    def _actor_class(max_restarts: int, max_task_retries: int) -> Any:
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

    @classmethod
    def _from_handles(
        cls,
        actors: list[Any],
        *,
        payload: dict[str, Any] | None = None,
        actor_node_ids: list[str] | None = None,
        actor_dispatch_indices: list[int] | tuple[int, ...] | set[int] | None = None,
        actor_init_refs: list[Any] | tuple[Any, ...] | None = None,
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
        instance._vane_retired = False
        instance._vane_location_nonce = ""
        instance._vane_readiness_futures = []
        instance._vane_init_errors = {}
        instance._payload = payload or {}
        instance._payload_ref = None
        init_refs = [] if actor_init_refs is None else list(actor_init_refs)
        if init_refs and len(init_refs) != len(actors):
            raise ValueError("actor_init_refs count must match actor handle count")
        instance._init_refs = init_refs
        if actor_node_ids is None:
            instance.actor_node_ids = [""] * len(actors)
        else:
            parsed_node_ids = [str(node_id or "").strip() for node_id in actor_node_ids]
            if len(parsed_node_ids) != len(actors):
                raise ValueError("actor node IDs count must match actor handle count")
            instance.actor_node_ids = parsed_node_ids
        return instance

    def shutdown(self, *, kill: bool = False) -> None:
        if not self._owns_actors:
            return

        import ray

        errors: list[str] = []
        remaining_actors: list[Any] = []
        for actor_index, actor in enumerate(self.actors):
            try:
                ray.kill(actor, no_restart=True)
            except BaseException as exc:
                errors.append(f"actor-{actor_index}: {type(exc).__name__}: {exc}")
                remaining_actors.append(actor)
        self.actors = remaining_actors
        if errors:
            raise RuntimeError("failed to shut down UDF actor pool: " + "; ".join(errors))


def apply_actor_node_options(
    actors: UDFActorPoolBase,
    *,
    options: dict[str, Any],
) -> UDFActorPoolBase:
    raw_node_ids = options.get("actor_node_ids")
    if not isinstance(raw_node_ids, list | tuple):
        raise TypeError("actor_node_ids must be a list or tuple")
    normalized_node_ids = [str(node_id or "").strip() for node_id in raw_node_ids]
    if len(normalized_node_ids) != len(actors.actors):
        raise ValueError("actor node IDs count must match actor handle count")
    actors.actor_node_ids = normalized_node_ids
    return actors


def requires_actor_pool(payload: dict[str, Any]) -> bool:
    return str(payload.get("execution_backend") or "").strip().lower() == "ray_actor"


def _positive_float_env(name: str, default: float | None = None) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _actor_init_timeout_s() -> float | None:
    return configured_ray_get_timeout_s(_positive_float_env("VANE_RAY_ACTOR_INIT_TIMEOUT_S"))


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    return type(exc).__name__ in {"GetTimeoutError", "TimeoutError"}


def _shutdown_owned_actors(ray: Any, actors_obj: Any) -> None:
    if not bool(getattr(actors_obj, "_owns_actors", False)):
        return
    actors = list(getattr(actors_obj, "actors", []))
    errors: list[str] = []
    remaining_actors: list[Any] = []
    for actor_index, actor in enumerate(actors):
        try:
            ray.kill(actor, no_restart=True)
        except BaseException as exc:
            errors.append(f"actor-{actor_index}: {type(exc).__name__}: {exc}")
            remaining_actors.append(actor)
    actors_obj.actors = remaining_actors
    if errors:
        raise RuntimeError("failed to shut down UDF actor pool: " + "; ".join(errors))


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
            resolved_node_ids = resolve_object_refs_blocking(refs)
        else:
            resolved_node_ids = resolve_object_refs_blocking(refs, timeout=timeout_s)
        node_ids = [str(node_id or "").strip() for node_id in resolved_node_ids]
        if len(node_ids) != len(actors) or any(not node_id for node_id in node_ids):
            raise RuntimeError("UDF actor initialization did not return one Ray node_id per actor")
    except Exception as exc:
        if _is_timeout_error(exc):
            timeout_desc = "query deadline" if timeout_s is None else f"{timeout_s:.3f}s"
            readiness_error = RuntimeError(
                "UDF actor pool initialization timed out "
                f"after {timeout_desc} while waiting for {len(refs)} actor init refs"
            )
        else:
            readiness_error = RuntimeError(f"UDF actor pool initialization failed: {type(exc).__name__}: {exc}")
        try:
            _shutdown_owned_actors(ray, actors_obj)
        except BaseException as cleanup_error:
            raise _OwnedUDFActorPoolsError(
                "UDF actor pool initialization failed and actor cleanup also failed: "
                f"initialization={readiness_error}; cleanup={type(cleanup_error).__name__}: {cleanup_error}",
                owned_actor_pools=[actors_obj],
                creation_error=readiness_error,
            ) from exc
        raise readiness_error from exc
    actors_obj.actor_node_ids = node_ids
    actors_obj._confirmed_ready.update(range(len(actors)))


def wait_for_first_actor_pool_ready(actors_obj: UDFActorPoolBase) -> dict[int, str]:
    """Wait until at least one actor is usable, leaving the rest pending.

    Actor CPU/GPU/declared heap are Ray Core resources.  A pool larger than
    the currently free cluster capacity therefore starts incrementally: the
    first successfully initialized actor opens execution while remaining
    handles stay pending and may join later.
    """

    refs = list(getattr(actors_obj, "_init_refs", ()))
    actors = list(getattr(actors_obj, "actors", ()))
    if len(refs) != len(actors) or not refs or any(ref is None for ref in refs):
        raise RuntimeError("UDF actor pool has incomplete init readiness refs")

    import ray

    pending = list(refs)
    actor_index_by_ref = {ref: actor_index for actor_index, ref in enumerate(refs)}
    timeout_s = _actor_init_timeout_s()
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    failures: list[BaseException] = []
    try:
        while pending:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            ready, pending = ray.wait(
                pending,
                num_returns=1,
                timeout=remaining,
            )
            if not ready:
                timeout_desc = "query deadline" if timeout_s is None else f"{timeout_s:.3f}s"
                raise RuntimeError(
                    "UDF actor pool initialization timed out "
                    f"after {timeout_desc} while waiting for its first ready actor"
                )
            ready_refs = list(ready)
            if pending:
                immediately_ready, pending = ray.wait(
                    pending,
                    num_returns=len(pending),
                    timeout=0,
                )
                ready_refs.extend(immediately_ready)
            ready_nodes: dict[int, str] = {}
            for ref in ready_refs:
                actor_index = actor_index_by_ref[ref]
                try:
                    node_id = str(resolve_object_refs_blocking(ref) or "").strip()
                    if not node_id:
                        raise RuntimeError("UDF actor initialization returned an empty Ray node_id")
                except BaseException as exc:
                    failures.append(exc)
                    continue
                actors_obj.actor_node_ids[actor_index] = node_id
                actors_obj._confirmed_ready.add(actor_index)
                ready_nodes[actor_index] = node_id
            if ready_nodes:
                return ready_nodes
        if failures:
            raise RuntimeError(
                "all UDF actors failed during initialization: "
                + "; ".join(f"{type(exc).__name__}: {exc}" for exc in failures[:3])
            ) from failures[0]
        raise RuntimeError("all UDF actors finished initialization without becoming ready")
    except BaseException as readiness_error:
        try:
            _shutdown_owned_actors(ray, actors_obj)
        except BaseException as cleanup_error:
            raise _OwnedUDFActorPoolsError(
                "UDF actor pool initialization failed and actor cleanup also failed: "
                f"initialization={readiness_error}; cleanup={type(cleanup_error).__name__}: {cleanup_error}",
                owned_actor_pools=[actors_obj],
                creation_error=readiness_error,
            ) from readiness_error
        raise


def ensure_actor_pools_for_plan(
    plan: Any,
    conn: Any = None,
    *,
    query_driver_handle: Any,
    query_generation_capability: str,
    session_config: dict[str, str],
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
        query_driver_handle=query_driver_handle,
        query_generation_capability=query_generation_capability,
        session_config=session_config,
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
    query_driver_handle: Any,
    query_generation_capability: str,
    session_config: dict[str, str],
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

    The query coordinator keeps every Ray-actor resource unit closed until
    :func:`wait_for_actor_pools_ready` succeeds.  Publishing the handles first
    lets native fragment executors initialize and expose their real pipeline
    topology while expensive user-model initialization is still running.
    """
    udf_nodes = plan.collect_udf_nodes(conn=conn)
    return _create_actor_pools_for_nodes(
        udf_nodes,
        query_driver_handle=query_driver_handle,
        query_generation_capability=query_generation_capability,
        session_config=session_config,
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
    query_driver_handle: Any,
    query_generation_capability: str,
    session_config: dict[str, str],
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
        query_driver_handle=query_driver_handle,
        query_generation_capability=query_generation_capability,
        session_config=session_config,
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


def prepare_actor_pools_for_nodes(
    udf_nodes: Any,
    *,
    query_driver_handle: Any,
    query_generation_capability: str,
    session_config: dict[str, str],
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
        query_driver_handle=query_driver_handle,
        query_generation_capability=query_generation_capability,
        session_config=session_config,
        set_handles=set_handles,
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


def _create_actor_pools_for_nodes(
    udf_nodes: Any,
    *,
    query_driver_handle: Any,
    query_generation_capability: str,
    session_config: dict[str, str],
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
    if query_driver_handle is None:
        raise ValueError("Ray UDF executor preparation requires an explicit query driver handle")
    generation_capability = str(query_generation_capability or "").strip()
    if not generation_capability:
        raise ValueError("Ray UDF executor preparation requires a query generation capability")
    normalized_session_config = {str(key): str(value) for key, value in session_config.items()}
    session_runtime_env_vars = build_session_runtime_env_vars(normalized_session_config)

    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray actor UDF creation requires an initialized RayRunner runtime")

    created: list[UDFActorPoolBase] = []
    actor_handles_map: dict[str, Any] = {}

    try:
        for node in udf_nodes:
            raw_payload = node.get("payload") or {}
            backend = str(raw_payload.get("execution_backend") or "").strip().lower()
            if backend not in {"ray_task", "ray_actor", "subprocess_task", "subprocess_actor"}:
                continue
            node_id = str(node["node_id"])
            executor_options: dict[str, Any] = {
                "session_config": normalized_session_config,
            }
            actor_handles_map[node_id] = executor_options
            if backend in {"subprocess_task", "subprocess_actor"}:
                continue
            executor_options["query_driver_handle"] = query_driver_handle
            executor_options["query_generation_capability"] = generation_capability
            if not requires_actor_pool_fn(raw_payload):
                continue
            payload = normalize_actor_pool_payload(raw_payload)
            gpus = payload_num_gpus(payload)

            if not requires_actor_pool_fn(payload):
                continue

            resource_unit_id = str(payload.get("resource_unit_id") or "").strip()
            if not resource_unit_id:
                raise RuntimeError(f"Ray actor UDF node {node_id} is missing resource_unit_id")
            query_id = str(payload.get("query_id") or "").strip()
            if not query_id:
                raise RuntimeError(f"Ray actor UDF node {node_id} is missing query_id")
            concurrency = required_positive_int(node, "actor_pool_size")
            _validate_stateful_actor_pool_contract(payload, concurrency)
            cpus = resolve_actor_num_cpus(payload)
            actor_location_nonce = secrets.token_urlsafe(32)
            ray_options = {
                "num_cpus": cpus,
                "runtime_env": {
                    "env_vars": {
                        **session_runtime_env_vars,
                        RAY_ACTOR_QUERY_ID_ENV: query_id,
                        RAY_ACTOR_RESOURCE_UNIT_ID_ENV: resource_unit_id,
                        RAY_ACTOR_POOL_NONCE_ENV: actor_location_nonce,
                        RAY_ACTOR_GENERATION_CAPABILITY_ENV: generation_capability,
                    },
                },
            }

            fault_tolerance_options: dict[str, int] = {}
            if payload.get("side_effects"):
                fault_tolerance_options["max_task_retries"] = 0
            if payload.get("stateful"):
                fault_tolerance_options["max_restarts"] = 0
                fault_tolerance_options["max_task_retries"] = 0

            actors_obj = actor_pool_cls(
                payload=payload,
                concurrency=concurrency,
                gpus_per_actor=gpus,
                ray_options=ray_options,
                **fault_tolerance_options,
            )
            actors_obj._vane_location_nonce = actor_location_nonce
            created.append(actors_obj)
            if wait_for_ready:
                _resolve_actor_pool_init_refs(ray, actors_obj)
            actor_node_ids = list(actors_obj.actor_node_ids)
            actor_dispatch_indices = set(actors_obj._confirmed_ready)
            executor_options.update(
                build_udf_executor_options(
                    actor_handles=list(actors_obj.actors),
                    actor_node_ids=actor_node_ids,
                    actor_dispatch_indices=actor_dispatch_indices,
                    actor_init_refs=list(actors_obj._init_refs),
                )
            )

        if actor_handles_map and set_handles is not None:
            set_handles(actor_handles_map)
    except BaseException as execution_error:
        for actors_obj in getattr(execution_error, "owned_actor_pools", ()):
            if actors_obj not in created:
                created.append(actors_obj)
        cleanup_errors: list[BaseException] = []
        remaining_owned: list[UDFActorPoolBase] = []
        for actors_obj in reversed(created):
            try:
                actors_obj.shutdown()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
                remaining_owned.append(actors_obj)
        if cleanup_errors:
            creation_error = getattr(execution_error, "creation_error", execution_error)
            raise _OwnedUDFActorPoolsError(
                f"UDF actor pool creation failed and cleanup also failed: {cleanup_errors[0]}",
                owned_actor_pools=list(reversed(remaining_owned)),
                creation_error=creation_error,
            ) from execution_error
        if isinstance(execution_error, _OwnedUDFActorPoolsError):
            raise execution_error.creation_error
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
        cleanup_errors: list[str] = []
        remaining_owned: list[UDFActorPoolBase] = []
        for actors_obj in reversed(actor_pools):
            try:
                actors_obj.shutdown()
            except BaseException as cleanup_error:
                cleanup_errors.append(f"{type(cleanup_error).__name__}: {cleanup_error}")
                remaining_owned.append(actors_obj)
        if cleanup_errors:
            creation_error = getattr(readiness_error, "creation_error", readiness_error)
            raise _OwnedUDFActorPoolsError(
                "UDF actor readiness failed and cleanup also failed: " + "; ".join(cleanup_errors),
                owned_actor_pools=list(reversed(remaining_owned)),
                creation_error=creation_error,
            ) from readiness_error
        if isinstance(readiness_error, _OwnedUDFActorPoolsError):
            raise readiness_error.creation_error
        raise


__all__ = [name for name in globals() if not name.startswith("__")]
