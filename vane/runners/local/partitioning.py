# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Local (non-distributed) materialized result type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vane.runners.common import MaterializedResult, PartitionMetadata

if TYPE_CHECKING:
    import pyarrow as pa


@dataclass
class LocalMaterializedResult(MaterializedResult):
    _partition: pa.Table
    _metadata: PartitionMetadata | None = None

    def partition(self) -> pa.Table:
        return self._partition

    def metadata(self) -> PartitionMetadata:
        if self._metadata is None:
            self._metadata = PartitionMetadata(
                num_rows=self._partition.num_rows,
                size_bytes=self._partition.nbytes,
            )
        return self._metadata

    def cancel(self) -> None:
        return None
