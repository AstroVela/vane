# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import time
from typing import Any

import pyarrow as pa

from vane.execution._common import ensure_table as _ensure_table
from vane.execution._udf_runtime import UDFExecutor as RuntimeUDFExecutor
from vane.execution.udf_output_schema import (
    empty_output_table_from_payload as _empty_output_table_from_payload,
)
from vane.execution.udf_ray_config import (
    eager_actor_warm_up_enabled as _eager_actor_warm_up_enabled,
)
from vane.execution.udf_ray_ref_bundle import (
    apply_ref_bundle_slices as _apply_ref_bundle_slices,
)
from vane.execution.udf_ray_scalar import execute_scalar_map_layout
from vane.execution.udf_ray_stream_protocol import (
    iter_bounded_stream_blocks,
    make_stream_block_metadata,
    make_stream_error_pair,
    validate_task_runtime_node,
)
from vane.execution.udf_threading import configure_ray_actor_loaded_torch_threads
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
    stage_id = payload.get("stage_id")
    if stage_id:
        parts.append(f"stage_id={stage_id}")
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


def _payload_requires_ray_block_stream(payload: dict[str, Any] | None) -> bool:
    payload = payload or {}
    if not bool(payload.get("produce_ray_block_stream", False)):
        raise RuntimeError("distributed Ray UDF output requires produce_ray_block_stream=True")
    return True


def _actor_class(
    max_restarts: int,
    max_task_retries: int,
):
    import ray

    # UDFExecutor and user callables are not thread-safe. QRM may pre-admit a
    # bounded next invocation, but Ray must execute actor methods serially.
    @ray.remote(max_restarts=max_restarts, max_task_retries=max_task_retries, max_concurrency=1)
    class UDFActor:
        def __init__(self) -> None:
            # No-arg constructor avoids Ray warning about constructor arguments
            # in the object store with max_restarts>0 (ray#53727).
            # Payload is injected via init_payload() immediately after creation.
            self._payload = None
            self.executor = None  # lazy init on first streaming submission

        def init_payload(self, payload: dict[str, Any]) -> None:
            """Inject payload after construction to avoid Ray object-store GC issues."""
            self._payload = payload
            _actor_debug_log("init_payload", self._payload)
            if self.executor is None:
                _actor_debug_log("executor_init_start", self._payload, path="init_payload")
                runtime_payload = dict(self._payload)
                runtime_payload["stream_output"] = True
                runtime_payload["prebatched_input"] = False
                self.executor = RuntimeUDFExecutor(runtime_payload)
                configure_ray_actor_loaded_torch_threads(self._payload)
                if _eager_actor_warm_up_enabled(self._payload):
                    self.executor.warm_up()
                _actor_debug_log("executor_init_done", self._payload, path="init_payload")

        def _ensure_executor(self, effective_payload: dict[str, Any]) -> None:
            if self.executor is not None:
                return
            if self._payload is None:
                self._payload = dict(effective_payload)
            runtime_payload = dict(self._payload)
            runtime_payload["stream_output"] = True
            runtime_payload["prebatched_input"] = False
            self.executor = RuntimeUDFExecutor(runtime_payload)
            configure_ray_actor_loaded_torch_threads(self._payload)

        def _run_row_preserving_batch(
            self,
            table: pa.Table,
            effective_payload: dict[str, Any],
        ) -> pa.Table:
            from vane.execution.udf_row_preserving import (
                fuse_row_preserving_output,
                split_row_preserving_input,
            )

            self._ensure_executor(effective_payload)
            args, passthrough = split_row_preserving_input(effective_payload, table)
            if args.num_rows == 0:
                output = _empty_output_table_from_payload(effective_payload)
                return fuse_row_preserving_output(effective_payload, passthrough, output)
            self.executor.submit(args)
            outputs = self.executor.drain_outputs()
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
        ):
            _payload_requires_ray_block_stream(effective_payload)
            validate_task_runtime_node(effective_payload)
            if self.executor is None:
                _actor_debug_log("executor_init_start", effective_payload, path="run_block_stream")
                self._ensure_executor(effective_payload)
                _actor_debug_log("executor_init_done", effective_payload, path="run_block_stream")
            args = _ensure_table(args)
            _actor_debug_log(
                "run_block_stream_submit_start",
                effective_payload,
                rows=args.num_rows,
                columns=args.num_columns,
            )
            output_count = 0

            def emit(table: pa.Table):
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
                table = execute_scalar_map_layout(effective_payload, args, self.executor)
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
            for result in self.executor.iter_submit(args):
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
        ):
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
            *blocks,
            payload: dict[str, Any] | None = None,
            slices=None,
            metadata=None,
            names=None,
        ):
            base_payload = payload or self._payload or {}
            try:
                _actor_debug_log(
                    "run_ref_bundle_stream_start",
                    base_payload,
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
                    base_payload,
                    rows=args.num_rows,
                    columns=args.num_columns,
                    materialize_s=f"{time.perf_counter() - materialize_start:.6f}",
                )
                for result in self.run_block_stream(
                    args,
                    payload=payload,
                ):
                    yield result
            except Exception as exc:
                error_block, error_metadata = make_stream_error_pair(base_payload, exc)
                yield error_block
                yield error_metadata

    return UDFActor


__all__ = [name for name in globals() if not name.startswith("__")]
