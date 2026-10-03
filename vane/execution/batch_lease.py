# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Owned Arrow batches whose exported views retain their byte reservation."""

from __future__ import annotations

import pyarrow as pa

from vane.execution.data_lifecycle import OutputBlockLeaseOwner
from vane.execution.udf_lifecycle import ExecutionCancellationScope


class BatchLease:
    """One IPC allocation; closing this owner does not revoke exported views."""

    def __init__(self, owner: OutputBlockLeaseOwner) -> None:
        self._owner: OutputBlockLeaseOwner | None = owner
        self._buffer: pa.Buffer | None = None

    @staticmethod
    def size(batch: pa.RecordBatch) -> int:
        try:
            with pa.MockOutputStream() as output:
                with pa.ipc.new_stream(output, batch.schema) as writer:
                    writer.write_batch(batch)
                return output.size()
        finally:
            # A serialization error's traceback must not retain native input.
            del batch

    def build(self, batch: pa.RecordBatch, size: int) -> None:
        try:
            self._buffer = pa.allocate_buffer(size)
            with pa.FixedSizeBufferWriter(self._buffer) as output:
                with pa.ipc.new_stream(output, batch.schema) as writer:
                    writer.write_batch(batch)
        finally:
            del batch

    def export(self, cancellation: ExecutionCancellationScope) -> pa.RecordBatch:
        cancellation.raise_if_cancelled("query result delivery")
        if self._buffer is None or self._owner is None:
            raise RuntimeError("batch lease is closed")
        self._owner.transition_to("external_consumer")
        buffer = pa.foreign_buffer(self._buffer.address, self._buffer.size, base=(self._buffer, self._owner))
        with pa.ipc.open_stream(buffer) as reader:
            return reader.read_next_batch()

    def close(self) -> None:
        self._buffer = None
        self._owner = None

    def cleanup_pending(self) -> bool:
        return self._buffer is not None or self._owner is not None
