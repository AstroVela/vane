# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from vane.execution._common import ensure_table as _ensure_table
from vane.execution._udf_runtime import UDFExecutor as RuntimeUDFExecutor
from vane.execution.udf_ray_config import (
    eager_actor_warm_up_enabled as _eager_actor_warm_up_enabled,
)
from vane.execution.udf_ray_ref_bundle import (
    apply_ref_bundle_slices as _apply_ref_bundle_slices,
)
from vane.execution.udf_ray_ref_bundle import (
    empty_output_table_from_payload as _empty_output_table_from_payload,
)
from vane.execution.udf_ray_scalar import execute_scalar_map_layout
from vane.execution.udf_ray_stream_protocol import (
    iter_bounded_stream_blocks,
    make_stream_block_metadata,
    make_stream_error_pair,
)
from vane.execution.udf_threading import configure_ray_actor_loaded_torch_threads
from vane.runners.ray.query_runtime_protocol import (
    RAY_ACTOR_GENERATION_CAPABILITY_ENV,
    RAY_ACTOR_INDEX_ENV,
    RAY_ACTOR_POOL_NONCE_ENV,
    RAY_ACTOR_QUERY_ID_ENV,
    RAY_ACTOR_RESOURCE_UNIT_ID_ENV,
    RAY_QUERY_RUNTIME_ACTOR_NAMESPACE,
    query_runtime_actor_name,
    ray_runtime_job_id,
)
from vane.runners.ray.ray_env import install_explicit_session_runtime_env
from vane.runners.ray.safe_get import resolve_object_refs_blocking


def _debug_enabled() -> bool:
    value = os.environ.get("DUCKDB_DISTRIBUTED_DEBUG", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _actor_debug_log(event: str, payload: dict[str, Any] | None = None, **fields: Any) -> None:
    if not _debug_enabled():
        return
    payload = payload or {}
    parts = [
        f"event={event}",
        f"pid={os.getpid()}",
        f"t={time.monotonic():.3f}",
    ]
    udf_name = payload.get("name") or payload.get("udf_name") or payload.get("callable_name")
    if udf_name:
        parts.append(f"udf_name={udf_name}")
    query_id = payload.get("query_id")
    if query_id:
        parts.append(f"query_id={query_id}")
    resource_unit_id = payload.get("resource_unit_id")
    if resource_unit_id:
        parts.append(f"resource_unit_id={resource_unit_id}")
    backend = payload.get("backend")
    if backend:
        parts.append(f"backend={backend}")
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    print("[vane-ray-actor-udf] " + " ".join(parts), file=sys.stderr, flush=True)


def _materialize_ref_bundle(
    block_refs: tuple[Any, ...] | list[Any],
    slices: list[Any] | tuple[Any, ...] | None = None,
    metadata: list[Any] | tuple[Any, ...] | None = None,
    names: list[str] | tuple[str, ...] | None = None,
) -> pa.Table:
    """Resolve a ref bundle into one Arrow table, applying descriptor slices.

    This is the materialization barrier for Vane's execution-level
    LazyDataChunk.  The fast path still passes ObjectRefs directly to Ray
    actors; this helper is only used when a non-ref-aware boundary needs rows in
    DuckDB memory.
    """
    refs = list(block_refs)
    if not refs:
        raise ValueError("empty ref bundle input is not supported")
    if all(isinstance(ref, pa.Table) for ref in refs):
        blocks = refs
    else:
        blocks = resolve_object_refs_blocking(refs)
    return _apply_ref_bundle_slices(blocks, slices, metadata=metadata, names=names)


def _reconstructed_actor_location_report(
    runtime_context: Any,
    *,
    expected_payload: dict[str, Any] | None = None,
    wait: bool,
) -> Any | None:
    if not bool(runtime_context.was_current_actor_reconstructed):
        return None

    values = {
        "query_id": os.environ.get(RAY_ACTOR_QUERY_ID_ENV, "").strip(),
        "resource_unit_id": os.environ.get(RAY_ACTOR_RESOURCE_UNIT_ID_ENV, "").strip(),
        "pool_nonce": os.environ.get(RAY_ACTOR_POOL_NONCE_ENV, "").strip(),
        "generation_capability": os.environ.get(RAY_ACTOR_GENERATION_CAPABILITY_ENV, "").strip(),
        "actor_index": os.environ.get(RAY_ACTOR_INDEX_ENV, "").strip(),
    }
    missing = sorted(key for key, value in values.items() if not value)
    if missing:
        raise RuntimeError("reconstructed Ray actor is missing location protocol environment: " + ", ".join(missing))
    try:
        actor_index = int(values["actor_index"])
    except ValueError as exc:
        raise RuntimeError("reconstructed Ray actor has an invalid actor index") from exc
    if actor_index < 0:
        raise RuntimeError("reconstructed Ray actor has a negative actor index")
    if expected_payload is not None:
        expected_query_id = str(expected_payload.get("query_id") or "").strip()
        expected_resource_unit_id = str(expected_payload.get("resource_unit_id") or "").strip()
        expected_actor_index = expected_payload.get("actor_index")
        if (
            values["query_id"] != expected_query_id
            or values["resource_unit_id"] != expected_resource_unit_id
            or isinstance(expected_actor_index, bool)
            or not isinstance(expected_actor_index, int)
            or actor_index != expected_actor_index
        ):
            raise RuntimeError("reconstructed Ray actor location does not match its task lease")

    node_id = str(runtime_context.get_node_id() or "").strip()
    if not node_id:
        raise RuntimeError("reconstructed Ray actor runtime did not expose node_id")
    import ray

    driver = ray.get_actor(
        query_runtime_actor_name(ray_runtime_job_id(runtime_context)),
        namespace=RAY_QUERY_RUNTIME_ACTOR_NAMESPACE,
    )
    report_ref = driver.report_query_udf_actor_location.remote(
        {
            "query_id": values["query_id"],
            "resource_unit_id": values["resource_unit_id"],
            "actor_index": actor_index,
            "pool_nonce": values["pool_nonce"],
            "node_id": node_id,
        },
        values["generation_capability"],
    )
    if not wait:
        return report_ref
    result = resolve_object_refs_blocking(report_ref)
    if not isinstance(result, dict):
        raise RuntimeError("Ray actor location reconciliation returned an invalid result")
    if not bool(result.get("accepted")) or str(result.get("node_id") or "").strip() != node_id:
        raise RuntimeError("Ray query no longer accepts this reconstructed actor location")
    return report_ref


def _validate_actor_task_runtime_node(payload: dict[str, Any]) -> str:
    expected_node_id = str(payload.get("node_id") or "").strip()
    if not expected_node_id:
        raise RuntimeError("distributed Ray UDF task payload is missing leased node_id")
    import ray

    runtime_context = ray.get_runtime_context()
    actual_node_id = str(runtime_context.get_node_id() or "").strip()
    if not actual_node_id:
        raise RuntimeError("Ray runtime context did not expose the executing node_id")
    if actual_node_id == expected_node_id:
        return actual_node_id
    report_ref = _reconstructed_actor_location_report(
        runtime_context,
        expected_payload=payload,
        wait=True,
    )
    if report_ref is None:
        raise RuntimeError(
            "distributed Ray UDF executed outside its query lease: "
            f"expected_node_id={expected_node_id} actual_node_id={actual_node_id}"
        )
    return actual_node_id


def _payload_requires_ray_block_stream(payload: dict[str, Any] | None) -> bool:
    payload = payload or {}
    if not bool(payload.get("produce_ray_block_stream", False)):
        raise RuntimeError("distributed Ray UDF output requires produce_ray_block_stream=True")
    return True


def _actor_class(
    max_restarts: int,
    max_task_retries: int,
) -> Any:
    import ray

    # UDFExecutor and user callables are not thread-safe. QRM may pre-admit a
    # bounded next invocation, but Ray must execute actor methods serially.
    @ray.remote(  # type: ignore[call-overload]
        max_restarts=max_restarts,
        max_task_retries=max_task_retries,
        max_concurrency=1,
    )
    class UDFActor:
        def __init__(self) -> None:
            install_explicit_session_runtime_env()
            # No-arg constructor avoids Ray warning about constructor arguments
            # in the object store with max_restarts>0 (ray#53727).
            # Payload is injected via init_payload() immediately after creation.
            self._payload: dict[str, Any] | None = None
            self.executor: RuntimeUDFExecutor | None = None  # lazy init on first streaming submission
            self._vane_location_report_ref: Any | None = None
            self._vane_location_report_error: BaseException | None = None
            try:
                import ray

                self._vane_location_report_ref = _reconstructed_actor_location_report(
                    ray.get_runtime_context(),
                    wait=False,
                )
            except BaseException as exc:
                # Do not turn a transient driver lookup race into another Ray
                # actor reconstruction. The first retried invocation performs
                # the same report synchronously before executing user code.
                self._vane_location_report_error = exc

        def init_payload(self, payload: dict[str, Any]) -> str:
            """Inject payload after construction to avoid Ray object-store GC issues."""
            self._payload = payload
            _actor_debug_log("init_payload", self._payload)
            if self.executor is None:
                _actor_debug_log("executor_init_start", self._payload, path="init_payload")
                runtime_payload = dict(self._payload)
                runtime_payload["stream_output"] = True
                runtime_payload["prebatched_input"] = False
                executor = RuntimeUDFExecutor(runtime_payload)
                self.executor = executor
                configure_ray_actor_loaded_torch_threads(self._payload)
                if _eager_actor_warm_up_enabled(self._payload):
                    executor.warm_up()
                _actor_debug_log("executor_init_done", self._payload, path="init_payload")
            import ray

            node_id = str(ray.get_runtime_context().get_node_id() or "").strip()
            if not node_id:
                raise RuntimeError("Ray actor runtime did not expose node_id after initialization")
            return node_id

        def _ensure_executor(self, effective_payload: dict[str, Any]) -> RuntimeUDFExecutor:
            executor = self.executor
            if executor is not None:
                return executor
            if self._payload is None:
                self._payload = dict(effective_payload)
            runtime_payload = dict(self._payload)
            runtime_payload["stream_output"] = True
            runtime_payload["prebatched_input"] = False
            executor = RuntimeUDFExecutor(runtime_payload)
            self.executor = executor
            configure_ray_actor_loaded_torch_threads(self._payload)
            return executor

        def _run_row_preserving_batch(
            self,
            table: pa.Table,
            effective_payload: dict[str, Any],
        ) -> pa.Table:
            from vane.execution.udf_row_preserving import (
                fuse_row_preserving_output,
                split_row_preserving_input,
            )

            executor = self._ensure_executor(effective_payload)
            args, passthrough = split_row_preserving_input(effective_payload, table)
            if args.num_rows == 0:
                output = _empty_output_table_from_payload(effective_payload)
                return fuse_row_preserving_output(effective_payload, passthrough, output)
            executor.submit(args)
            outputs = executor.drain_outputs()
            if len(outputs) != 1:
                raise RuntimeError("map_batches_rows actor produced %d outputs, expected exactly 1" % len(outputs))
            return fuse_row_preserving_output(
                effective_payload,
                passthrough,
                _ensure_table(outputs[0]),
            )

        def _run_block_stream_impl(
            self,
            args: pa.Table,
            effective_payload: dict[str, Any],
        ) -> Iterator[Any]:
            _payload_requires_ray_block_stream(effective_payload)
            _validate_actor_task_runtime_node(effective_payload)
            executor = self.executor
            if executor is None:
                _actor_debug_log("executor_init_start", effective_payload, path="run_block_stream")
                executor = self._ensure_executor(effective_payload)
                _actor_debug_log("executor_init_done", effective_payload, path="run_block_stream")
            args = _ensure_table(args)
            _actor_debug_log(
                "run_block_stream_submit_start",
                effective_payload,
                rows=args.num_rows,
                columns=args.num_columns,
            )
            output_count = 0

            def emit(table: pa.Table) -> Iterator[Any]:
                nonlocal output_count
                for block in iter_bounded_stream_blocks(_ensure_table(table), effective_payload):
                    yield block
                    yield make_stream_block_metadata(
                        block,
                        effective_payload,
                        output_index=output_count,
                    )
                    output_count += 1

            if str(effective_payload.get("call_mode") or "") == "map":
                table = execute_scalar_map_layout(effective_payload, args, executor)
                yield from emit(table)
                _actor_debug_log(
                    "run_block_stream_submit_done",
                    effective_payload,
                    rows=args.num_rows,
                    outputs=1,
                )
                return
            if str(effective_payload.get("call_mode") or "") == "map_batches_rows":
                yield from emit(self._run_row_preserving_batch(args, effective_payload))
                _actor_debug_log(
                    "run_block_stream_submit_done",
                    effective_payload,
                    rows=args.num_rows,
                    outputs=1,
                )
                return
            for result in executor.iter_submit(args):
                table = _ensure_table(result)
                _actor_debug_log(
                    "run_block_stream_output",
                    effective_payload,
                    output_index=output_count,
                    rows=table.num_rows,
                    columns=table.num_columns,
                )
                yield from emit(table)
            if output_count == 0:
                table = _empty_output_table_from_payload(effective_payload)
                yield from emit(table)
            _actor_debug_log(
                "run_block_stream_submit_done",
                effective_payload,
                rows=args.num_rows,
                outputs=output_count,
            )

        def run_block_stream(
            self,
            args: pa.Table,
            payload: dict[str, Any] | None = None,
        ) -> Iterator[Any]:
            effective_payload = dict(self._payload or {})
            if payload is not None:
                effective_payload.update(payload)
            try:
                yield from self._run_block_stream_impl(args, effective_payload)
            except Exception as exc:
                error_block, error_metadata = make_stream_error_pair(effective_payload, exc)
                yield error_block
                yield error_metadata

        def run_ref_bundle_stream(
            self,
            *blocks: Any,
            payload: dict[str, Any] | None = None,
            slices: Any = None,
            metadata: Any = None,
            names: Any = None,
        ) -> Iterator[Any]:
            effective_payload = dict(payload or self._payload or {})
            try:
                # A reconstructed actor may have moved to another Ray node. Move
                # the resident slot and any prefetched task lease in QRM before
                # resolving input ObjectRefs on that node, not merely before the
                # user callable starts. Carry the reconciled node into the nested
                # block-stream call so it does not report the same move twice.
                effective_payload["node_id"] = _validate_actor_task_runtime_node(effective_payload)
                _actor_debug_log(
                    "run_ref_bundle_stream_start",
                    effective_payload,
                    blocks=len(blocks or []),
                    slices=len(slices or []),
                    metadata=len(metadata or []),
                    names=len(names or []),
                    block_types=",".join(type(block).__name__ for block in blocks[:4]),
                )
                materialize_start = time.perf_counter()
                args = _apply_ref_bundle_slices(blocks, slices, metadata=metadata, names=names)
                _actor_debug_log(
                    "run_ref_bundle_stream_materialized",
                    effective_payload,
                    rows=args.num_rows,
                    columns=args.num_columns,
                    materialize_s=f"{time.perf_counter() - materialize_start:.6f}",
                )
                for result in self.run_block_stream(
                    args,
                    payload=effective_payload,
                ):
                    yield result
            except Exception as exc:
                error_block, error_metadata = make_stream_error_pair(effective_payload, exc)
                yield error_block
                yield error_metadata

    return UDFActor


__all__ = [name for name in globals() if not name.startswith("__")]
