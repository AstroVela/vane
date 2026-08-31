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
    def run_iter(self, relation: Any) -> Iterator[MaterializedResult]:
        """Yield individual partitions as they are completed.

        Args:
            relation: a DuckDB relation-like object describing the query to execute
        """
        ...

    @abstractmethod
    def run_iter_tables(self, relation: Any) -> Iterator[pa.Table]:
        """Similar to run_iter(), but always dereference and yield table objects.

        Args:
            relation: a DuckDB relation-like object describing the query to execute
        """
        ...

    @abstractmethod
    def run_write(self, relation: Any) -> dict[str, Any]:
        """Execute a write through the selected backend."""
        ...

    @abstractmethod
    def run_statement_write(self, connection: Any, statement: Any) -> dict[str, Any]:
        """Execute a statement write through the selected backend."""
        ...

    @abstractmethod
    def run_datasink(self, relation: Any) -> dict[str, Any]:
        """Execute a Python DataSink terminal through the selected backend."""
        ...
