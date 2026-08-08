# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Unified UDF Executor interface and factory.

This module defines ``UDFExecutor``, the single abstract base class for all
UDF executor types. It also provides ``build_unified_executor()``, a unified
factory that routes to the correct backend based on payload parameters.

The interface matches what the C++ physical operator actually calls::

    executor.submit(args: pa.Table)      # send a batch of input
    executor.take_ready_result() -> pa.Table | None
    executor.finished_submitting()       # no more input batches
    executor.all_tasks_finished() -> bool

Every executor uses event-driven wakeups::

    executor.register_wakeup(callback)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import pyarrow as pa


class UDFExecutor(ABC):
    """Unified executor interface for all UDF types.

    The interface uses the simplified calling convention that the C++
    physical operator expects:
    - ``submit(args)`` — args is a ``pa.Table`` with one column per UDF argument
    - ``take_ready_result()`` -> ``pa.Table | None`` returns one ready output table
    """

    @abstractmethod
    def submit(self, args: pa.Table) -> None:
        """Submit a batch of input arguments for processing."""
        ...

    @abstractmethod
    def take_ready_result(self) -> pa.Table | None:
        """Return the next ready output batch, or None if no result is ready."""
        ...

    @abstractmethod
    def finished_submitting(self) -> None:
        """Signal that no more batches will be submitted."""
        ...

    @abstractmethod
    def all_tasks_finished(self) -> bool:
        """Return True if all submitted work is complete and consumed."""
        ...

    @abstractmethod
    def register_wakeup(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when a result becomes available."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up resources. Called when executor is no longer needed."""
        ...


def build_unified_executor(
    payload: dict[str, Any],
    options: dict[str, Any] | None = None,
) -> UDFExecutor:
    """Build a UDF executor from the unified payload."""
    from vane.execution.udf import build_executor

    return build_executor(payload, options)


__all__ = ["UDFExecutor", "build_unified_executor"]
