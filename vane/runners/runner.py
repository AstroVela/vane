# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]

    from vane.runners.common import MaterializedResult


class Runner:
    name: ClassVar[Literal["ray", "local"]]

    @abstractmethod
    def run_iter(self, logical_plan: Any) -> Iterator[MaterializedResult]:
        """Yield individual partitions as they are completed.

        Args:
            logical_plan: an already-bound, serialized query plan
        """
        ...

    @abstractmethod
    def run_iter_tables(self, logical_plan: Any) -> Iterator[pa.Table]:
        """Similar to run_iter(), but always dereference and yield table objects.

        Args:
            logical_plan: an already-bound, serialized query plan
        """
        ...

    @abstractmethod
    def run_write(self, logical_plan: Any) -> dict[str, Any]:
        """Execute an already-bound write plan through the selected backend."""
        ...

    @abstractmethod
    def run_datasink(self, logical_plan: Any) -> dict[str, Any]:
        """Execute an already-bound Python DataSink plan through the selected backend."""
        ...
