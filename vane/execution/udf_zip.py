# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Compose two colocated streaming UDFs with bounded, strict key alignment."""

from __future__ import annotations

import inspect
import math
import threading
import time
from collections.abc import Callable, Generator, Iterable
from contextlib import ExitStack
from typing import Any

_Schema = list[dict[str, str]]
_Lease = tuple[Any, Callable[[], None]]
_BatchFunction = Callable[[Any], Any]


def _plain_arrow_type(value: Any) -> None:
    from vane import sqltype

    dtype = sqltype(str(value))
    if str(dtype).partition("(")[0].strip().upper() in {"FILE", "IMAGE", "AUDIO", "VIDEO"}:
        raise ValueError("zip_map_batches requires plain Arrow values, not descriptor-bound FILE/media handles")
    if dtype.id in {"struct", "list", "array", "map", "union"}:
        for _, child in dtype.children:
            if hasattr(child, "id"):
                _plain_arrow_type(child)


def _schema(value: Any) -> _Schema:
    if not isinstance(value, dict) or not value or any(not isinstance(name, str) or not name for name in value):
        raise ValueError("zip_map_batches requires explicit, nonempty output schemas")
    for kind in value.values():
        _plain_arrow_type(kind)
    return [{"name": name, "kind": "duckdb_type", "type": str(kind)} for name, kind in value.items()]


def zip_map_batches(
    relation: Any,
    left: _BatchFunction,
    right: _BatchFunction,
    *,
    left_schema: dict[str, Any],
    right_schema: dict[str, Any],
    schema: dict[str, Any],
    on: str | Iterable[str],
    merge: Callable[[Any, Any], Any],
    cpus: tuple[float, float] = (1.0, 1.0),
    memory_bytes: tuple[int, int] = (1024**3, 1024**3),
    max_buffer_bytes: int = 256 * 1024**2,
    timeout: float = 60.0,
    execution_backend: str | None = None,
) -> Any:
    """Run two streaming batch functions per input row and zip ordered keys.

    Branch workers are colocated within one execution unit. Its declared CPU
    and memory are the sum of both branches. A branch retains one unread block;
    permits are returned only after the paired output is published. This is
    strict positional alignment with key validation, not a general SQL join.
    """
    from vane.udf import no_retries

    schemas = (_schema(left_schema), _schema(right_schema))
    _schema(schema)
    for dtype in relation.types if relation is not None else ():
        _plain_arrow_type(dtype)
    if not all(inspect.isfunction(value) or inspect.ismethod(value) for value in (left, right)) or not callable(merge):
        raise TypeError("zip_map_batches branches and merge must be callable")
    keys = (on,) if isinstance(on, str) else tuple(on)
    if not keys or len(set(keys)) != len(keys) or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("zip_map_batches requires distinct alignment column names")
    for key in keys:
        if key not in left_schema or key not in right_schema or str(left_schema[key]) != str(right_schema[key]):
            raise ValueError("zip_map_batches alignment columns must have matching declared types")
    if len(cpus) != 2 or any(not math.isfinite(float(value)) or value <= 0 for value in cpus):
        raise ValueError("zip_map_batches requires two positive CPU reservations")
    if len(memory_bytes) != 2 or any(type(value) is not int or value <= 0 for value in memory_bytes):
        raise ValueError("zip_map_batches requires two positive memory reservations")
    if type(max_buffer_bytes) is not int or max_buffer_bytes <= 0:
        raise ValueError("zip_map_batches buffer bytes must be positive")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("zip_map_batches timeout must be finite and positive")
    if execution_backend not in (None, "subprocess_task", "ray_task"):
        raise ValueError("zip_map_batches requires a task execution backend")

    @no_retries
    def combined(batch: Any) -> Generator[Any, None, None]:
        yield from _zip_outputs(
            batch, (left, right), schemas, keys, merge, cpus, memory_bytes, max_buffer_bytes, timeout
        )

    return relation.map_batches(
        combined,
        schema=schema,
        batch_size=1,
        min_task_batch_size=1,
        output_batch_size=1,
        cpus=sum(cpus),
        gpus=0,
        memory_bytes=sum(memory_bytes),
        execution_backend=execution_backend,
        target_max_batch_bytes=max_buffer_bytes,
        task_input_max_bytes=max_buffer_bytes,
        output_target_max_bytes=max_buffer_bytes,
    )


class _Wake:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.epoch = 0

    def notify(self) -> None:
        with self.condition:
            self.epoch += 1
            self.condition.notify_all()

    def version(self) -> int:
        with self.condition:
            return self.epoch

    def wait(self, version: int, deadline: float) -> None:
        with self.condition:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("zip_map_batches timed out waiting for aligned outputs")
            self.condition.wait_for(lambda: self.epoch != version, timeout=remaining)


def _unpack(lease: _Lease, budget: int) -> tuple[Any, list[Any]]:
    from vane.execution.ref_bundle import REF_BUNDLE_RESULT_MARKER, materialize_ref_bundle

    result = lease[0]
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], bool):
        result = result[0]
    if result is None:
        return None, []
    if isinstance(result, BaseException):
        raise result
    if not isinstance(result, tuple) or len(result) != 4 or result[0] != REF_BUNDLE_RESULT_MARKER:
        raise RuntimeError("zip_map_batches received an invalid output bundle")
    refs = result[1]
    try:
        table = materialize_ref_bundle(refs, metadata=result[2], names=result[3])
        if table.get_total_buffer_size() > budget:
            raise ValueError("zip_map_batches branch exceeds its buffer byte budget")
        return table, refs
    except BaseException:
        for ref in refs:
            ref.release()
        raise


def _release(lease: _Lease, refs: list[Any]) -> None:
    try:
        lease[1]()
    finally:
        for ref in refs:
            ref.release()


def _key(table: Any, keys: tuple[str, ...]) -> tuple[Any, ...]:
    if table.num_rows != 1:
        raise ValueError("zip_map_batches requires one row per branch output block")
    values = tuple(table[name][0].as_py() for name in keys)
    if any(value is None or isinstance(value, float) and not math.isfinite(value) for value in values):
        raise ValueError("zip_map_batches alignment keys must be non-null and finite")
    return values


def _zip_outputs(
    batch: Any,
    branches: tuple[_BatchFunction, _BatchFunction],
    schemas: tuple[_Schema, _Schema],
    keys: tuple[str, ...],
    merge: Callable[[Any, Any], Any],
    cpus: tuple[float, float],
    memory: tuple[int, int],
    budget: int,
    timeout: float,
) -> Generator[Any, None, None]:
    from vane import pickle
    from vane.execution.udf_subprocess import UDFExecutor

    wake = _Wake()
    executors = []
    successful = False
    leases: list[_Lease | None] = [None, None]
    refs: list[list[Any]] = [[], []]
    tables: list[Any] = [None, None]
    ended = [False, False]
    try:
        for index, function in enumerate(branches):
            payload = {
                "call_mode": "map_batches",
                "execution_backend": "subprocess_task",
                "function_pickle": pickle.dumps(function),
                "udf_name": f"zip_map_batches.branch{index}",
                "batch_size": 1,
                "output_batch_size": 1,
                "output_schema": schemas[index],
                "cpus": cpus[index],
                "gpus": 0,
                "memory_bytes": memory[index],
                "stream_buffer_blocks": 1,
                "produce_ref_bundle_output": True,
                "streaming_output_mode": "local_shm_ref_bundle",
            }
            executor = UDFExecutor(payload)
            executors.append(executor)
            executor.register_wakeup(wake.notify)
            executor.request_task_admission(batch.get_total_buffer_size())
        deadline = time.monotonic() + timeout
        submitted = [False, False]
        while not all(submitted):
            version = wake.version()
            for index, executor in enumerate(executors):
                if not submitted[index] and executor.task_admission_state()["available"]:
                    executor.submit(batch)
                    executor.finished_submitting()
                    submitted[index] = True
            if not all(submitted):
                wake.wait(version, deadline)
        previous = None
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("zip_map_batches timed out waiting for aligned outputs")
            version = wake.version()
            progressed = False
            for index, executor in enumerate(executors):
                if tables[index] is not None or ended[index]:
                    continue
                acquired = executor._take_ready_result_with_release()
                if acquired is None:
                    if executor.all_tasks_finished():
                        ended[index] = True
                    continue
                progressed = True
                leases[index] = acquired
                table, owned_refs = _unpack(acquired, budget)
                refs[index] = owned_refs
                if table is None or not table.num_rows:
                    _release(acquired, owned_refs)
                    leases[index], refs[index] = None, []
                else:
                    tables[index] = table
            if any(ended):
                if any(table is not None for table in tables):
                    raise ValueError("zip_map_batches branches ended at different output positions")
                if all(ended):
                    successful = True
                    return
            elif all(table is not None for table in tables):
                left_key, right_key = (_key(table, keys) for table in tables)
                if left_key != right_key or previous is not None and left_key <= previous:
                    raise ValueError("zip_map_batches branches have mismatched or non-increasing alignment keys")
                if sum(table.get_total_buffer_size() for table in tables) > budget:
                    raise ValueError("zip_map_batches aligned inputs exceed their combined byte budget")
                result = merge(*tables)
                if result.num_rows != 1 or result.get_total_buffer_size() > budget:
                    raise ValueError("zip_map_batches merged output exceeds its row or byte budget")
                yield result
                result = None
                previous = left_key
                tables[:] = [None, None]
                with ExitStack() as releases:
                    for index, lease in enumerate(leases):
                        assert lease is not None
                        releases.callback(_release, lease, refs[index])
                        leases[index], refs[index] = None, []
                deadline = time.monotonic() + timeout
                continue
            if progressed:
                continue
            wake.wait(version, deadline)
    finally:
        tables.clear()
        with ExitStack() as cleanup:
            for executor in executors:
                cleanup.callback(executor.close, kill=not successful)
            for lease, owned_refs in zip(leases, refs, strict=True):
                if lease is not None:
                    cleanup.callback(_release, lease, owned_refs)
            if not successful:
                for executor in executors:
                    cleanup.callback(executor._cancel_execution_scopes, "parallel UDF branch failed or closed")
