# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from typing import Any

import pyarrow as pa

from vane.execution._common import callable_cache_enabled as _callable_cache_enabled
from vane.execution._common import ensure_table as _ensure_table
from vane.execution._udf_runtime import UDFExecutor as RuntimeUDFExecutor
from vane.execution.ray_stream_adapter import TaskLeaseObjectRefGenerator
from vane.execution.udf_ray_actor_pool import (
    UDFActorPoolBase as _UDFActorPoolBase,
)
from vane.execution.udf_ray_actor_pool import (
    apply_actor_node_options as _apply_actor_node_options_impl,
)
from vane.execution.udf_ray_actor_pool import (
    ensure_actor_pools_for_nodes as _ensure_actor_pools_for_nodes_impl,
)
from vane.execution.udf_ray_actor_pool import (
    ensure_actor_pools_for_plan as _ensure_actor_pools_for_plan_impl,
)
from vane.execution.udf_ray_actor_pool import (
    prepare_actor_pools_for_plan as _prepare_actor_pools_for_plan_impl,
)
from vane.execution.udf_ray_actor_pool import (
    requires_actor_pool as _requires_actor_pool_impl,
)
from vane.execution.udf_ray_actor_pool import (
    wait_for_actor_pools_ready as _wait_for_actor_pools_ready_impl,
)
from vane.execution.udf_ray_actor_runtime import (
    _actor_class as _actor_runtime_class,
)
from vane.execution.udf_ray_actor_state import (
    build_stateful_actor_error_context as _build_stateful_actor_error_context,
)
from vane.execution.udf_ray_config import (
    MAX_ACTOR_TASK_RETRIES,
)
from vane.execution.udf_ray_config import (
    payload_num_cpus as _payload_num_cpus,
)
from vane.execution.udf_ray_config import (
    payload_num_gpus as _payload_num_gpus,
)
from vane.execution.udf_ray_config import (
    required_positive_int as _required_positive_int,
)
from vane.execution.udf_ray_env import (
    collect_actor_env_overrides as _collect_actor_env_overrides,
)
from vane.execution.udf_ray_env import (
    is_vane_worker_process as _is_vane_worker_process,
)
from vane.execution.udf_ray_env import (
    normalize_actor_node_ids as _normalize_actor_node_ids,
)
from vane.execution.udf_ray_env import (
    normalize_actor_pool_payload as _normalize_actor_pool_payload,
)
from vane.execution.udf_ray_env import (
    resolve_actor_num_cpus as _resolve_actor_num_cpus,
)
from vane.execution.udf_ray_ref_bundle import (
    apply_ref_bundle_slices as _apply_ref_bundle_slices,
)
from vane.execution.udf_ray_remote_lifecycle import (
    RemoteUDFLifecycleMixin,
)
from vane.execution.udf_ray_remote_readiness import RemoteUDFActorReadinessMixin
from vane.execution.udf_ray_remote_ref_bundle import (
    RemoteUDFRefBundleMixin,
    _resolve_ref_bundle_task_refs,
)
from vane.execution.udf_ray_remote_runtime import RemoteUDFRuntimeMixin
from vane.execution.udf_ray_remote_submit import (
    RemoteUDFSubmitMixin,
)
from vane.execution.udf_ray_scalar import (
    execute_scalar_map_layout as _execute_scalar_map_layout,
)
from vane.execution.udf_ray_stream_protocol import (
    RAY_UDF_GENERATOR_BACKPRESSURE_OBJECTS,
    RAY_UDF_STREAM_BUFFER_BLOCKS,
    iter_bounded_stream_blocks,
    make_stream_block_metadata,
    make_stream_error_pair,
    task_payload_with_lease,
    validate_task_runtime_node,
)
from vane.execution.udf_task_admission import (
    TaskAdmissionExecutorMixin,
    ray_udf_task_memory_bytes,
)
from vane.execution.udf_threading import configure_loaded_torch_threads
from vane.execution.unified_executor import UDFExecutor

DEFAULT_UDF_OUTPUT_TARGET_MAX_BYTES = 128 * 1024 * 1024
DEFAULT_GENERATOR_BACKPRESSURE_NUM_OBJECTS = RAY_UDF_GENERATOR_BACKPRESSURE_OBJECTS
DEFAULT_GENERATOR_STREAM_BUFFER_BLOCKS = RAY_UDF_STREAM_BUFFER_BLOCKS

_RAY_TASK_DEBUG_SEQ = 0


def _next_ray_task_debug_seq() -> int:
    global _RAY_TASK_DEBUG_SEQ
    _RAY_TASK_DEBUG_SEQ += 1
    return _RAY_TASK_DEBUG_SEQ


def _ray_payload_requires_block_stream(payload: dict[str, Any]) -> bool:
    if not bool(payload.get("produce_ray_block_stream", False)):
        raise RuntimeError("distributed Ray UDF output requires produce_ray_block_stream=True")
    if not str(payload.get("query_id") or "").strip():
        raise RuntimeError("distributed Ray UDF payload requires query_id")
    if not str(payload.get("stage_id") or "").strip():
        raise RuntimeError("distributed Ray UDF payload requires pre-registered stage_id")
    return True


def _submit_ray_remote(remote_fn: Any, node_id: str, *args: Any, **kwargs: Any) -> Any:
    node_key = str(node_id).strip()
    if not node_key:
        raise RuntimeError("Ray task submission requires its leased node_id")
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    return remote_fn.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=node_key,
            soft=False,
        )
    ).remote(*args, **kwargs)


def _ray_task_debug_enabled() -> bool:
    for name in ("VANE_RAY_TASK_DEBUG", "VANE_UDF_WORKER_SLOT_DEBUG", "DUCKDB_DISTRIBUTED_DEBUG"):
        value = os.environ.get(name, "")
        if value.strip().lower() not in ("", "0", "false", "no", "off"):
            return True
    return False


def _ray_task_log_every() -> int:
    value = os.environ.get("VANE_RAY_TASK_LOG_EVERY_N", "").strip()
    if not value:
        return 0
    parsed = int(value)
    if parsed < 0:
        raise ValueError("VANE_RAY_TASK_LOG_EVERY_N must be non-negative")
    return parsed


def _ray_task_should_log(seq: int) -> bool:
    if not _ray_task_debug_enabled():
        return False
    if seq <= 5:
        return True
    every = _ray_task_log_every()
    return every > 0 and seq % every == 0


def _process_thread_count() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("Threads:"):
                    return int(line.split(":", 1)[1].strip())
    except (OSError, ValueError):
        return -1
    return -1


def _env_value(name: str) -> str:
    value = os.environ.get(name)
    return "<unset>" if value is None else value


def _torch_thread_fields() -> str:
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return "torch_loaded=false"
    fields = ["torch_loaded=true"]
    try:
        fields.append(f"torch_num_threads={int(torch_module.get_num_threads())}")
    except Exception:
        fields.append("torch_num_threads=<error>")
    try:
        fields.append(f"torch_interop_threads={int(torch_module.get_num_interop_threads())}")
    except Exception:
        fields.append("torch_interop_threads=<error>")
    return " ".join(fields)


def _ray_task_debug_log(event: str, payload: dict[str, Any] | None = None, **fields: object) -> None:
    if not _ray_task_debug_enabled():
        return
    backend = str((payload or {}).get("execution_backend") or "-")
    parts = [
        f"event={event}",
        f"backend={backend}",
        f"process_threads={_process_thread_count()}",
        f"OMP_NUM_THREADS={_env_value('OMP_NUM_THREADS')}",
        f"MKL_NUM_THREADS={_env_value('MKL_NUM_THREADS')}",
        f"OPENBLAS_NUM_THREADS={_env_value('OPENBLAS_NUM_THREADS')}",
        f"NUMEXPR_NUM_THREADS={_env_value('NUMEXPR_NUM_THREADS')}",
        _torch_thread_fields(),
    ]
    parts.extend(f"{key}={value}" for key, value in fields.items())
    print(f"[vane-ray-task pid={os.getpid()}] " + " ".join(parts), file=sys.stderr, flush=True)


def _build_udf_executor_options(
    *,
    actor_handles: list[Any],
    actor_node_ids: list[str] | None,
    actor_dispatch_indices: set[int] | list[int] | tuple[int, ...] | None,
) -> dict[str, Any]:
    normalized_node_ids = _normalize_actor_node_ids(
        actor_node_ids,
        expected_count=len(actor_handles),
    )
    if normalized_node_ids is None:
        raise ValueError("actor node IDs are required for pre-created Ray UDF actors")
    if actor_dispatch_indices is None:
        raise ValueError("actor_dispatch_indices are required for pre-created Ray UDF actors")
    raw_dispatch_indices = list(actor_dispatch_indices)
    if any(type(item) is not int for item in raw_dispatch_indices):
        raise ValueError("actor_dispatch_indices must contain integers")
    if len(set(raw_dispatch_indices)) != len(raw_dispatch_indices):
        raise ValueError("actor_dispatch_indices must not contain duplicates")
    invalid = [idx for idx in raw_dispatch_indices if idx < 0 or idx >= len(actor_handles)]
    if invalid:
        raise ValueError(f"actor_dispatch_indices contains out-of-range indices: {invalid}")
    return {
        "actor_handles": list(actor_handles),
        "actor_node_ids": normalized_node_ids,
        "actor_dispatch_indices": sorted(raw_dispatch_indices),
    }


def _build_actor_runtime_env(ray_options: dict[str, Any] | None) -> dict[str, Any]:
    options = dict(ray_options or {})
    runtime_env = dict(options.get("runtime_env") or {})
    env_vars = dict(_collect_actor_env_overrides())
    env_vars.update(runtime_env.get("env_vars") or {})
    runtime_env["env_vars"] = env_vars
    return runtime_env


def _actor_class(max_restarts: int, max_task_retries: int):
    return _actor_runtime_class(max_restarts, max_task_retries)


class UDFActorPool(_UDFActorPoolBase):
    @staticmethod
    def _actor_class(max_restarts: int, max_task_retries: int):
        return _actor_class(max_restarts, max_task_retries)

    @staticmethod
    def _resolve_actor_num_cpus(payload: dict[str, Any]) -> float:
        return _resolve_actor_num_cpus(payload)

    @staticmethod
    def _resolve_actor_memory_bytes(payload: dict[str, Any]) -> int:
        return ray_udf_task_memory_bytes(payload)

    @staticmethod
    def _build_actor_runtime_env(ray_options: dict[str, Any] | None) -> dict[str, Any]:
        return _build_actor_runtime_env(ray_options)

    @staticmethod
    def _normalize_actor_node_ids(
        node_ids: list[str] | None,
        *,
        expected_count: int,
    ) -> list[str] | None:
        return _normalize_actor_node_ids(node_ids, expected_count=expected_count)


def _apply_actor_node_options(
    actors: UDFActorPool,
    *,
    options: dict[str, Any],
) -> UDFActorPool:
    return _apply_actor_node_options_impl(
        actors,
        options=options,
        normalize_actor_node_ids=_normalize_actor_node_ids,
    )


def ensure_actor_pools_for_plan(
    plan: Any,
    *,
    actor_node_ids_by_stage: dict[str, tuple[str, ...]],
    conn: Any = None,
) -> tuple[list[UDFActorPool], dict[str, Any]]:
    return _ensure_actor_pools_for_plan_impl(
        plan,
        conn=conn,
        actor_node_ids_by_stage=actor_node_ids_by_stage,
        actor_pool_cls=UDFActorPool,
        is_vane_worker_process=_is_vane_worker_process,
        requires_actor_pool_fn=_requires_actor_pool_impl,
        normalize_actor_pool_payload=_normalize_actor_pool_payload,
        payload_num_gpus=_payload_num_gpus,
        required_positive_int=_required_positive_int,
        resolve_actor_num_cpus=_resolve_actor_num_cpus,
        build_udf_executor_options=_build_udf_executor_options,
    )


def ensure_actor_pools_for_nodes(
    udf_nodes: Any,
    *,
    actor_node_ids_by_stage: dict[str, tuple[str, ...]],
    set_handles: Any = None,
) -> tuple[list[UDFActorPool], dict[str, Any]]:
    return _ensure_actor_pools_for_nodes_impl(
        udf_nodes,
        actor_node_ids_by_stage=actor_node_ids_by_stage,
        set_handles=set_handles,
        actor_pool_cls=UDFActorPool,
        is_vane_worker_process=_is_vane_worker_process,
        requires_actor_pool_fn=_requires_actor_pool_impl,
        normalize_actor_pool_payload=_normalize_actor_pool_payload,
        payload_num_gpus=_payload_num_gpus,
        required_positive_int=_required_positive_int,
        resolve_actor_num_cpus=_resolve_actor_num_cpus,
        build_udf_executor_options=_build_udf_executor_options,
    )


def prepare_actor_pools_for_plan(
    plan: Any,
    *,
    actor_node_ids_by_stage: dict[str, tuple[str, ...]],
    conn: Any = None,
) -> tuple[list[UDFActorPool], dict[str, Any]]:
    return _prepare_actor_pools_for_plan_impl(
        plan,
        conn=conn,
        actor_node_ids_by_stage=actor_node_ids_by_stage,
        actor_pool_cls=UDFActorPool,
        is_vane_worker_process=_is_vane_worker_process,
        requires_actor_pool_fn=_requires_actor_pool_impl,
        normalize_actor_pool_payload=_normalize_actor_pool_payload,
        payload_num_gpus=_payload_num_gpus,
        required_positive_int=_required_positive_int,
        resolve_actor_num_cpus=_resolve_actor_num_cpus,
        build_udf_executor_options=_build_udf_executor_options,
    )


def wait_for_actor_pools_ready(actor_pools: list[UDFActorPool]) -> None:
    _wait_for_actor_pools_ready_impl(actor_pools)


class RemoteUDFExecutor(
    TaskAdmissionExecutorMixin,
    RemoteUDFRuntimeMixin,
    RemoteUDFActorReadinessMixin,
    RemoteUDFRefBundleMixin,
    RemoteUDFSubmitMixin,
    RemoteUDFLifecycleMixin,
    UDFExecutor,
):
    def __init__(
        self,
        actors: UDFActorPool,
        payload: dict[str, Any],
    ) -> None:
        self._actors_obj = actors
        self._payload = payload
        self._initialize_task_admission(payload)
        self.actors = actors.actors
        # Actor readiness lifecycle:
        # - _pending_ready_refs tracks __ray_ready__ refs waiting to resolve.
        # - _ready_actor_indices is the dispatch set.
        self._pending_ready_refs: dict[Any, int] = {}
        self._ready_actor_indices: list[int] = []
        self._ready_actor_set: set[int] = set()
        self._actor_init_errors: dict[int, BaseException | str] = {}
        self._finished_submitting = False
        self._shutdown_called = False
        if bool(payload.get("async_mode", False)):
            raise ValueError("ray actor async_mode has been removed; use synchronous actor streaming")
        self._async_mode = False
        _ray_payload_requires_block_stream(payload)
        self._ready_refs_cv = threading.Condition()
        self._ready_probe_refs: deque[Any] = deque()
        self._ready_probe_ref_set: set[Any] = set()
        # Input column renaming: prewarm actors may not have input_names in
        # their payload, so we rename here before dispatching to the actor.
        raw_input_names = payload.get("input_names")
        self._input_names: list[str] | None = list(raw_input_names) if raw_input_names else None

        self._actor_node_ids = _normalize_actor_node_ids(
            self._actors_obj.actor_node_ids,
            expected_count=len(self.actors),
        )
        if not self._actor_node_ids or any(not node_id for node_id in self._actor_node_ids):
            raise RuntimeError("every Ray actor handle requires a coordinator-confirmed node identity")

        self._prime_actor_readiness()
        self._refresh_actor_readiness()

    def shutdown(self) -> None:
        self._close_task_admission()
        RemoteUDFLifecycleMixin.shutdown(self)

    def close(self) -> None:
        self.shutdown()

    def error_context(self) -> dict[str, Any] | None:
        """Describe the exact stateful actor for centralized loss errors."""
        if not self._payload.get("stateful"):
            return None
        return _build_stateful_actor_error_context(self._payload, list(self.actors))


def _build_ray_actor_executor(payload: dict[str, Any], options: dict[str, Any]) -> UDFExecutor:
    """Build a Ray-backed UDF executor using pre-created actor handles.

    Actor handles MUST be provided via ``options["actor_handles"]``.

    This function NEVER creates actors.  All actor creation must happen
    at the driver level.
    """
    if options.get("ray_actor_pool_name") is not None or payload.get("ray_actor_pool_name") is not None:
        raise RuntimeError("Ray UDF named actor pools have been removed; use driver pre-created actor_handles instead")

    payload = _normalize_actor_pool_payload(payload)
    _ray_payload_requires_block_stream(payload)

    pre_created_handles = options.get("actor_handles")
    if pre_created_handles is not None:
        actors = UDFActorPool._from_handles(
            pre_created_handles,
            payload=payload,
            actor_dispatch_indices=options.get("actor_dispatch_indices"),
        )
        actors = _apply_actor_node_options(actors, options=options)
        return RemoteUDFExecutor(actors, payload)

    raise RuntimeError(
        "build_ray_executor requires pre-created actor handles. Ray UDF named actor pools have been removed."
    )


class RayTaskUDFExecutor(TaskAdmissionExecutorMixin, UDFExecutor):
    def __init__(
        self,
        payload: dict[str, Any],
        run_bundle_stream: Any,
        run_ref_bundle_stream: Any,
    ) -> None:
        self.payload = payload
        self.run_bundle_stream = run_bundle_stream
        self.run_ref_bundle_stream = run_ref_bundle_stream
        self._finished_submitting = False
        self._initialize_task_admission(payload)
        _ray_payload_requires_block_stream(payload)
        _ray_task_debug_log(
            "executor_init",
            payload,
            cpus=payload.get("cpus"),
            gpus=payload.get("gpus"),
        )

    def shutdown(self) -> None:
        self._close_task_admission()

    def close(self) -> None:
        self.shutdown()

    def submit(self, _args: pa.Table) -> None:
        raise RuntimeError("RayTaskUDFExecutor.submit() direct submit path has been removed; use submit_with_id()")

    def submit_with_id(self, submit_id: int, args: pa.Table):
        seq = _next_ray_task_debug_seq()
        table = _ensure_table(args)
        if _ray_task_should_log(seq):
            _ray_task_debug_log(
                "driver_submit_table",
                self.payload,
                seq=seq,
                submit_id=int(submit_id),
                rows=table.num_rows,
                columns=table.num_columns,
            )
        admission = self._take_task_admission()
        return TaskLeaseObjectRefGenerator(
            admission=admission,
            submitter=lambda granted_lease: _submit_ray_remote(
                self.run_bundle_stream,
                granted_lease["node_id"],
                task_payload_with_lease(self.payload, granted_lease),
                [table],
            ),
        )

    def submit_ref_bundle_with_id(self, submit_id: int, block_refs, slices, metadata, names):
        seq = _next_ray_task_debug_seq()
        if _ray_task_should_log(seq):
            _ray_task_debug_log(
                "driver_submit_ref_bundle",
                self.payload,
                seq=seq,
                submit_id=int(submit_id),
                block_refs=len(block_refs or []),
                slices=len(slices or []),
                metadata=len(metadata or []),
                names=len(names or []),
            )
        task_block_refs = _resolve_ref_bundle_task_refs(block_refs)
        admission = self._take_task_admission()
        return TaskLeaseObjectRefGenerator(
            admission=admission,
            submitter=lambda granted_lease: _submit_ray_remote(
                self.run_ref_bundle_stream,
                granted_lease["node_id"],
                *task_block_refs,
                payload=task_payload_with_lease(self.payload, granted_lease),
                slices=list(slices or []),
                metadata=list(metadata or []),
                names=list(names or []),
            ),
        )

    def submit_ref_bundle(self, _block_refs, _slices, _metadata, _names) -> None:
        raise RuntimeError(
            "RayTaskUDFExecutor.submit_ref_bundle() direct ref-bundle submit path has been removed; "
            "use submit_ref_bundle_with_id()"
        )

    def take_ready_result(self) -> Any | None:
        return None

    def finished_submitting(self) -> None:
        self._finished_submitting = True

    def all_tasks_finished(self) -> bool:
        return self._finished_submitting


def _task_remote_options(
    num_cpus: float,
    num_gpus: float,
    memory_bytes: int,
    max_retries: int,
    ray_options: dict[str, Any],
) -> dict[str, Any]:
    options = dict(ray_options or {})
    options["num_cpus"] = num_cpus
    options["num_gpus"] = num_gpus
    options["memory"] = int(memory_bytes)
    options["max_retries"] = max_retries
    options["_generator_backpressure_num_objects"] = RAY_UDF_GENERATOR_BACKPRESSURE_OBJECTS
    return options


def _streaming_task_payload(payload: dict[str, Any]) -> dict[str, Any]:
    call_mode = str(payload.get("call_mode") or "")
    if call_mode == "map_batches_rows":
        return payload
    if call_mode not in {"map_batches", "flat_map"}:
        return payload
    stream_payload = dict(payload)
    stream_payload["stream_output"] = True
    stream_payload["prebatched_input"] = False
    return stream_payload


def _iter_materialized_task_outputs(
    payload: dict[str, Any],
    tables: list[pa.Table] | tuple[pa.Table, ...],
):
    """Yield direct block/metadata pairs for materialized Ray task input."""
    validate_task_runtime_node(payload)
    stream_payload = _streaming_task_payload(payload)
    executor = RuntimeUDFExecutor(
        stream_payload,
        cache_callable=_callable_cache_enabled(payload),
    )
    configure_loaded_torch_threads()
    output_index = 0

    def emit(output: pa.Table):
        nonlocal output_index
        for block in iter_bounded_stream_blocks(_ensure_table(output), stream_payload):
            metadata = make_stream_block_metadata(
                block,
                stream_payload,
                output_index=output_index,
            )
            output_index += 1
            yield block, metadata

    if str(stream_payload.get("call_mode") or "") == "map":
        for raw_table in tables:
            for block, metadata in emit(
                _execute_scalar_map_layout(
                    stream_payload,
                    _ensure_table(raw_table),
                    executor,
                )
            ):
                yield block
                yield metadata
        executor.finished_submitting()
        return

    if str(stream_payload.get("call_mode") or "") == "map_batches_rows":
        for raw_table in tables:
            for block, metadata in emit(
                _execute_row_preserving_batch_layout(
                    stream_payload,
                    _ensure_table(raw_table),
                    executor,
                )
            ):
                yield block
                yield metadata
        executor.finished_submitting()
        return

    for raw_table in tables:
        for output in executor.iter_submit(_ensure_table(raw_table)):
            for block, metadata in emit(output):
                yield block
                yield metadata
    executor.finished_submitting()
    for output in executor.drain_outputs():
        for block, metadata in emit(output):
            yield block
            yield metadata


def _execute_row_preserving_batch_layout(
    payload: dict[str, Any],
    table: pa.Table,
    executor: RuntimeUDFExecutor,
) -> pa.Table:
    from vane.execution.udf_row_preserving import (
        fuse_row_preserving_output,
        split_row_preserving_input,
    )

    args, passthrough = split_row_preserving_input(payload, _ensure_table(table))
    executor.submit(args)
    outputs = executor.drain_outputs()
    if len(outputs) != 1:
        raise RuntimeError("map_batches_rows task produced %d outputs, expected exactly 1" % len(outputs))
    return fuse_row_preserving_output(payload, passthrough, _ensure_table(outputs[0]))


def _build_bundle_stream_remote(
    num_cpus: float,
    num_gpus: float,
    memory_bytes: int,
    max_retries: int,
    ray_options: dict[str, Any],
) -> Any:
    import ray

    task_options = _task_remote_options(
        num_cpus,
        num_gpus,
        memory_bytes,
        max_retries,
        ray_options,
    )

    @ray.remote(**task_options)
    def run_bundle_stream(payload: dict[str, Any], tables: list[pa.Table]):
        seq = _next_ray_task_debug_seq()
        log_task = _ray_task_should_log(seq)
        start = time.perf_counter()
        if log_task:
            _ray_task_debug_log(
                "worker_bundle_start",
                payload,
                seq=seq,
                tables=len(tables),
                rows=sum(_ensure_table(table).num_rows for table in tables),
            )
        output_count = 0
        output_rows = 0
        if log_task:
            _ray_task_debug_log("worker_bundle_executor_ready", payload, seq=seq)
        try:
            for item in _iter_materialized_task_outputs(payload, tables):
                if isinstance(item, pa.Table):
                    output_count += 1
                    output_rows += item.num_rows
                yield item
        except Exception as exc:
            error_block, error_metadata = make_stream_error_pair(payload, exc)
            yield error_block
            yield error_metadata
        if log_task:
            _ray_task_debug_log(
                "worker_bundle_finished",
                payload,
                seq=seq,
                outputs=output_count,
                output_rows=output_rows,
                total_s=f"{time.perf_counter() - start:.6f}",
            )

    return run_bundle_stream


def _iter_ref_bundle_task_outputs(payload: dict[str, Any], blocks, slices, metadata, names):
    validate_task_runtime_node(payload)
    seq = _next_ray_task_debug_seq()
    log_task = _ray_task_should_log(seq)
    total_start = time.perf_counter()
    if log_task:
        _ray_task_debug_log(
            "worker_ref_bundle_start",
            payload,
            seq=seq,
            blocks=len(blocks or []),
            slices=len(slices or []),
            metadata=len(metadata or []),
            names=len(names or []),
        )
    call_mode = str(payload.get("call_mode") or "")
    materialize_start = time.perf_counter()
    table = _apply_ref_bundle_slices(blocks, slices, metadata=metadata, names=names)
    materialize_s = time.perf_counter() - materialize_start
    if log_task:
        _ray_task_debug_log(
            "worker_ref_bundle_materialized",
            payload,
            seq=seq,
            rows=table.num_rows,
            columns=table.num_columns,
            materialize_s=f"{materialize_s:.6f}",
        )
    if call_mode in {"map_batches", "flat_map"}:
        stream_payload = _streaming_task_payload(payload)
        executor = RuntimeUDFExecutor(stream_payload, cache_callable=_callable_cache_enabled(payload))
        configure_loaded_torch_threads()
        if log_task:
            _ray_task_debug_log("worker_ref_bundle_executor_ready", stream_payload, seq=seq)
        output_count = 0
        output_rows = 0

        def emit_output(output: pa.Table):
            nonlocal output_count, output_rows
            for output_table in iter_bounded_stream_blocks(_ensure_table(output), stream_payload):
                output_rows += output_table.num_rows
                output_metadata = make_stream_block_metadata(
                    output_table,
                    stream_payload,
                    output_index=output_count,
                )
                output_count += 1
                yield output_table, output_metadata

        for output in executor.iter_submit(table):
            for block, output_metadata in emit_output(output):
                yield block
                yield output_metadata
        executor.finished_submitting()
        for output in executor.drain_outputs():
            for block, output_metadata in emit_output(output):
                yield block
                yield output_metadata
        if log_task:
            _ray_task_debug_log(
                "worker_ref_bundle_finished",
                stream_payload,
                seq=seq,
                outputs=output_count,
                output_rows=output_rows,
                total_s=f"{time.perf_counter() - total_start:.6f}",
            )
        return

    if call_mode == "map_batches_rows":
        executor = RuntimeUDFExecutor(payload, cache_callable=_callable_cache_enabled(payload))
        configure_loaded_torch_threads()
        fused = _execute_row_preserving_batch_layout(payload, table, executor)
        executor.finished_submitting()
        for output_index, block in enumerate(iter_bounded_stream_blocks(fused, payload)):
            yield block
            yield make_stream_block_metadata(block, payload, output_index=output_index)
        return

    if call_mode != "map":
        raise RuntimeError("ray_task ref bundle input requires call_mode=map, map_batches, or flat_map")

    executor = RuntimeUDFExecutor(payload, cache_callable=_callable_cache_enabled(payload))
    configure_loaded_torch_threads()
    fused = _execute_scalar_map_layout(payload, table, executor)
    for output_index, block in enumerate(iter_bounded_stream_blocks(fused, payload)):
        yield block
        yield make_stream_block_metadata(block, payload, output_index=output_index)


def _build_ref_bundle_stream_remote(
    num_cpus: float,
    num_gpus: float,
    memory_bytes: int,
    max_retries: int,
    ray_options: dict[str, Any],
) -> Any:
    import ray

    task_options = _task_remote_options(
        num_cpus,
        num_gpus,
        memory_bytes,
        max_retries,
        ray_options,
    )

    @ray.remote(**task_options)
    def run_ref_bundle_stream(*blocks, payload: dict[str, Any], slices, metadata, names):
        try:
            yield from _iter_ref_bundle_task_outputs(payload, blocks, slices, metadata, names)
        except Exception as exc:
            error_block, error_metadata = make_stream_error_pair(payload, exc)
            yield error_block
            yield error_metadata

    return run_ref_bundle_stream


def _build_ray_task_executor(payload: dict[str, Any], options: dict[str, Any]) -> UDFExecutor:
    import ray

    # Graph registration and the direct block-stream contract are mandatory.
    # Reject standalone execution before starting or attaching to a Ray cluster.
    _ray_payload_requires_block_stream(payload)
    if not ray.is_initialized():
        raise RuntimeError("Ray task UDF execution requires an initialized RayRunner runtime")
    ray_options = dict(options.get("ray_options") or {})
    num_cpus = _payload_num_cpus(payload)
    num_gpus = _payload_num_gpus(payload)
    memory_bytes = ray_udf_task_memory_bytes(payload)
    configured_max_retries = options["max_task_retries"]
    max_retries = MAX_ACTOR_TASK_RETRIES if configured_max_retries is None else configured_max_retries
    if payload.get("side_effects"):
        max_retries = 0

    return RayTaskUDFExecutor(
        payload,
        _build_bundle_stream_remote(
            num_cpus,
            num_gpus,
            memory_bytes,
            max_retries,
            ray_options,
        ),
        _build_ref_bundle_stream_remote(
            num_cpus,
            num_gpus,
            memory_bytes,
            max_retries,
            ray_options,
        ),
    )


def build_ray_executor(payload: dict[str, Any], options: dict[str, Any]) -> UDFExecutor:
    backend = str(payload.get("execution_backend") or "").strip().lower()
    if backend == "ray_task":
        return _build_ray_task_executor(payload, options)
    if backend == "ray_actor":
        return _build_ray_actor_executor(payload, options)
    raise ValueError("Ray UDF executor requires execution_backend=ray_task or ray_actor")
