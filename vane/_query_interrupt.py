# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Keep connection interruption scoped to one distributed result operation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextvars import ContextVar
from typing import Any

_active_interrupt_check: ContextVar[Callable[[], None] | None] = ContextVar("vane_query_interrupt_check", default=None)


def has_query_interrupt_check() -> bool:
    return _active_interrupt_check.get() is not None


def check_query_interrupted() -> None:
    check = _active_interrupt_check.get()
    if check is None:
        return
    check()


class QueryResultIterator:
    def __init__(self, iterator: Iterator[Any], check: Callable[[], None] | None) -> None:
        self._iterator: Iterator[Any] | None = iterator
        self._check = check

    def __iter__(self) -> QueryResultIterator:
        return self

    def __next__(self) -> Any:
        iterator = self._iterator
        if iterator is None:
            raise StopIteration
        token = _active_interrupt_check.set(self._check)
        try:
            check_query_interrupted()
            result = next(iterator)
            check_query_interrupted()
            return result
        except BaseException:
            self.close()
            raise
        finally:
            _active_interrupt_check.reset(token)

    def close(self) -> None:
        iterator = self._iterator
        self._iterator = None
        token = _active_interrupt_check.set(None)
        try:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        finally:
            _active_interrupt_check.reset(token)
