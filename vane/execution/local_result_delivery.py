# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Arrow IPC buffers for managed local result delivery."""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from vane.execution.data_lifecycle import OutputBlockLeaseOwner
from vane.execution.result_delivery import ManagedResult
from vane.execution.udf_lifecycle import ExecutionCancellationScope


class _ArrowResultPayload:
    def __init__(self, owner: OutputBlockLeaseOwner) -> None:
        self._owner: OutputBlockLeaseOwner | None = owner
        self._buffer: pa.Buffer | None = None

    def build(self, table: pa.Table, size: int) -> None:
        # Reserve before allocating; a fixed-size writer cannot grow the body.
        self._buffer = pa.allocate_buffer(size)
        with pa.FixedSizeBufferWriter(self._buffer) as output:
            with pa.ipc.new_stream(output, table.schema) as writer:
                writer.write_table(table)

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


def prepare_local_result(result: ManagedResult, native: Any) -> None:
    """Build exact-size delivery buffers from an already materialized result.

    Native collection and its peak memory precede this budget. Encoding creates
    one IPC copy per native partition; consumers then receive zero-copy views.
    """
    result.result_schema = native.result_schema
    result.completion_status = getattr(native, "completion_status", None)
    result.stats = getattr(native, "stats", None)
    result.task_stats = getattr(native, "task_stats", None)
    for table in native.partition_payloads:
        result.check_preparation()
        if not isinstance(table, pa.Table):
            raise TypeError("managed local results require Arrow table partitions")
        with pa.MockOutputStream() as sizing:
            with pa.ipc.new_stream(sizing, table.schema) as writer:
                writer.write_table(table)
            size = sizing.size()
        payload = _ArrowResultPayload(result.own_buffer(size))
        result.hold(payload)
        result.check_preparation()
        payload.build(table, size)
