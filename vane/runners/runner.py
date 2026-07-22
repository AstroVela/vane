# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pyarrow as pa

    from vane.runners.common import MaterializedResult


class Runner:
    name: ClassVar[Literal["ray", "local"]]

    @abstractmethod
    def run_iter(self, relation: Any, results_buffer_size: int | None = None) -> Iterator[MaterializedResult]:
        """Yield individual partitions as they are completed.

        Args:
            relation: a DuckDB relation-like object describing the query to execute
            results_buffer_size: if the plan is executed asynchronously, this is the maximum size of the number of results
                that can be buffered before execution should pause and wait.
        """
        ...

    @abstractmethod
    def run_iter_tables(self, relation: Any, results_buffer_size: int | None = None) -> Iterator[pa.Table]:
        """Similar to run_iter(), but always dereference and yield table objects.

        Args:
            relation: a DuckDB relation-like object describing the query to execute
            results_buffer_size: if the plan is executed asynchronously, this is the maximum size of the number of results
                that can be buffered before execution should pause and wait.
        """
        ...

    @abstractmethod
    def run_write(self, relation: Any) -> dict[str, Any]:
        """Execute a COPY/write relation through the runner."""
        ...
