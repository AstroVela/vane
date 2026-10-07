# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Arrow IPC buffers for managed local result delivery."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pyarrow as pa

from vane.execution.data_lifecycle import OutputBlockLeaseOwner
from vane.execution.result_delivery import QueryResult
from vane.execution.udf_lifecycle import ExecutionCancellationScope


class _NativeResultStream:
    """Own the reader, native cancellation fence and request through cleanup."""

    def __init__(self, result: QueryResult, reader: Any, query: Any, close_native: Any) -> None:
        self._reader = reader
        self._read_native: Any = None
        self._guard_native: Any = None
        self._query = query
        self._close_native = close_native
        self._native_closed = False
        self._execution_finished = False
        self._retired = False
        self._eof = False
        self._failed = False
        self._had_rows = False
        request = query.request

        def cancel() -> None:
            request.cancel()

        self._unregister = result._cancellation.register_cancel_wakeup(cancel)

    def check(self) -> None:
        self._query.request._check_cancelled()

    def guard(self) -> None:
        if self._guard_native is not None:
            self._guard_native()

    def commit(self, operation: Callable[[], None]) -> None:
        request = self._query.request
        while True:
            self.check()
            with request._lock:
                # Preserve request -> result lock order. Cancellation can be
                # recorded before its notification reaches the result handle.
                if request._ticket.cancellation_reason is not None or (
                    request._deadline is not None and request._deadline.expired()
                ):
                    continue
                operation()
                # Acquiring the result gate can be delayed. Check absolute
                # execution expiry again before committing the caller handoff.
                if request._deadline is None or not request._deadline.expired():
                    return

    def read(self, result: QueryResult) -> bool:
        batch = None
        try:
            self.check()
            result.schema = self._reader.schema
            batch = self._read_native()
            self.check()
            self._had_rows = self._had_rows or bool(batch.num_rows)
            _prepare_table(result, pa.Table.from_batches([batch]))
        except StopIteration:
            self._eof = True
            self.close()
            self.check()
            result.completion_status = "ok" if self._had_rows else "empty"
            return False
        except BaseException as error:
            from vane.execution.result_delivery import ResultDeliveryFull

            if isinstance(error, ResultDeliveryFull):
                error._execution_started = True
            self._failed = True
            raise
        finally:
            # A failed reservation's traceback must not keep native input
            # buffers alive after request cleanup tries to retire them.
            batch = None
        return True

    def close(self) -> None:
        self.guard()
        request = self._query.request
        if not self._native_closed:
            if not self._eof and not self._failed:
                request.cancel()
            # Fence native interrupts before permitting reuse of this cursor.
            self._query.close()
            self._close_native(False)
            if self._reader is not None:
                self._reader.close()
            self._native_closed = True
            request._cancel_cleanup = None
            self._unregister()
        if not self._execution_finished:
            request._finish_execution(failed=not self._eof)
            self._execution_finished = True
        if not self._retired:
            self._close_native(True)
            self._retired = True
        request.shutdown(kill=not self._eof)

    def retire(self, release_result: Callable[[], None]) -> None:
        self._query.request._release_result(release_result)

    def cleanup_pending(self) -> bool:
        request = self._query.request
        with request._lock:
            return (
                not self._native_closed
                or not self._retired
                or request._executing
                or request._cleaning
                or bool(request._resources)
            )


def prepare_native_query_stream(
    result: QueryResult, reader: Any, schema: dict[str, Any], query: Any, close_native: Any
) -> _NativeResultStream:
    result.result_schema = schema
    result.schema = reader.schema if reader is not None else None
    result.completion_status = "streaming"
    source = _NativeResultStream(result, reader, query, close_native)
    result.start_stream(source)
    request = query.request
    with request._lock:
        request._result_pending = True
    request._cancel_cleanup = lambda: result.request_cancelled(request._ticket.cancellation_error)
    return source


class _ArrowResultPayload:
    def __init__(self, owner: OutputBlockLeaseOwner) -> None:
        self._owner: OutputBlockLeaseOwner | None = owner
        self._buffer: pa.Buffer | None = None

    def build(self, table: pa.Table, size: int) -> None:
        # Reserve before allocating; a fixed-size writer cannot grow the body.
        try:
            self._buffer = pa.allocate_buffer(size)
            with pa.FixedSizeBufferWriter(self._buffer) as output:
                with pa.ipc.new_stream(output, table.schema) as writer:
                    writer.write_table(table)
        finally:
            del table

    def export(self, cancellation: ExecutionCancellationScope) -> pa.Table:
        cancellation.raise_if_cancelled("result delivery")
        if self._buffer is None or self._owner is None:
            raise RuntimeError("result payload is closed")
        self._owner.transition_to("external_consumer")
        # The foreign buffer retains both the allocation and its logical owner.
        # Tables, slices, arrays and NumPy views therefore keep the same charge.
        buffer = pa.foreign_buffer(self._buffer.address, self._buffer.size, base=(self._buffer, self._owner))
        return pa.ipc.open_stream(pa.BufferReader(buffer)).read_all()

    def close(self) -> None:
        self._buffer = None
        self._owner = None

    def cleanup_pending(self) -> bool:
        return self._buffer is not None or self._owner is not None


def prepare_native_query_result(result: QueryResult, table: pa.Table, schema: dict[str, Any]) -> None:
    """Adapt an ordinary, fully materialized native query without another request."""
    result.result_schema = schema
    result.schema = table.schema
    result.completion_status = "ok" if table.num_rows else "empty"
    if table.num_rows:
        _prepare_table(result, table)


def _prepare_table(result: QueryResult, table: pa.Table) -> None:
    try:
        result.check_preparation()
        if not isinstance(table, pa.Table):
            raise TypeError("managed local results require Arrow table partitions")
        result.schema = table.schema
        with pa.MockOutputStream() as sizing:
            with pa.ipc.new_stream(sizing, table.schema) as writer:
                writer.write_table(table)
            size = sizing.size()
        payload = _ArrowResultPayload(result.own_buffer(size))
        result.hold(payload)
        result.check_preparation()
        payload.build(table, size)
    finally:
        del table
