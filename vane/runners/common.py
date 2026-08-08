# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared types used by both local and ray runners."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa


class QueryDeadlineExceeded(TimeoutError):
    """The query-wide deadline expired, not an individual wait operation."""


@dataclass(frozen=True)
class PartitionMetadata:
    num_rows: int
    size_bytes: int | None = None


class MaterializedResult:
    """A protocol for accessing the result partition of a task."""

    @abstractmethod
    def partition(self) -> pa.Table: ...

    @abstractmethod
    def metadata(self) -> PartitionMetadata: ...

    @abstractmethod
    def cancel(self) -> None: ...
