# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit actor/worker boundary for materialized BatchUDF callables."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pyarrow as pa

from vane.execution._udf_validation import ensure_synchronous_udf_result, validate_synchronous_udf_callable
from vane.execution.udf_output_schema import (
    _arrow_type_from_duckdb_pytype,
    batch_udf_output_schema,
    columns_to_output_table,
)
from vane.udf import BatchUDF


def validate_batch_udf_class(udf: type[BatchUDF], payload: dict[str, Any]) -> None:
    """Reject unsupported contracts before running the user's constructor."""
    if (
        payload.get("execution_backend") != "ray_actor"
        or payload.get("call_mode") != "map_batches"
        or payload.get("row_preserving", False)
    ):
        raise ValueError("BatchUDF requires ordinary map_batches with execution_backend='ray_actor'")
    if udf.__call__ is BatchUDF.__call__:
        raise TypeError("BatchUDF subclasses must implement __call__")
    for name in ("prepare_batch", "__call__", "warm_up", "_vane_close"):
        method = getattr(udf, name, None)
        if not callable(method):
            if name in ("prepare_batch", "__call__"):
                raise TypeError(f"BatchUDF.{name} must be callable")
            continue
        validate_synchronous_udf_callable(method)
        if inspect.isgeneratorfunction(method) or inspect.isgeneratorfunction(inspect.unwrap(method)):
            raise TypeError(f"BatchUDF.{name} must not be a generator function")
    if callable(getattr(udf, "bind_async_runtime", None)):
        raise TypeError("BatchUDF does not support bind_async_runtime")
    batch_udf_output_schema(payload)


class BatchCallableRuntime:
    """One serial worker, owned and called by the actor thread only."""

    def __init__(self, payload: dict[str, Any], *, output_contract_types: tuple[Any | None, ...]) -> None:
        self._owner = threading.get_ident()
        self._schema = batch_udf_output_schema(payload)
        # Only our restricted materialized encoder establishes canonical storage.
        # Schema equality alone cannot establish this for user-returned Arrow tables.
        try:
            self.output_is_canonical = len(output_contract_types) == len(self._schema) and all(
                dtype is None or field.type.equals(_arrow_type_from_duckdb_pytype(dtype))
                for field, dtype in zip(self._schema, output_contract_types, strict=True)
            )
        except (TypeError, ValueError):
            # A richer logical contract still uses the ordinary normalization path.
            self.output_is_canonical = False
        self._udf_name = str(payload.get("udf_name") or "<BatchUDF>")
        self._pool: ThreadPoolExecutor | None = None
        self._closed = False

    def check_owner(self) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError("BatchUDF must be called and closed by its owning actor thread")

    def __call__(self, udf: BatchUDF, table: pa.Table) -> pa.Table:
        self.check_owner()
        if self._closed:
            raise RuntimeError("BatchUDF worker is closed")
        prepared = ensure_synchronous_udf_result(udf.prepare_batch(table))
        if isinstance(prepared, Iterator):
            raise TypeError("BatchUDF.prepare_batch must return a materialized value")
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vane-batch-udf")
        future = self._pool.submit(udf, prepared)
        result = None
        try:
            result = ensure_synchronous_udf_result(future.result())
            return columns_to_output_table(result, self._schema, udf_name=self._udf_name)
        finally:
            # Do not retain raw results or Future exception cycles across a
            # caller's suspended output generator. Arrow owns shared buffers.
            del prepared, result, future

    def close(self) -> None:
        self.check_owner()
        self._closed = True
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
