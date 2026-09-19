# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit readiness boundary between Python sources and native scan tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa


class _DataSourceWait(ABC):
    """A yielded wait owns its cancellation and one-shot readiness callback.

    subscribe() must atomically return ready or register the callback. Readiness
    can race the caller's descheduling; native interrupt epochs handle that
    race. Waiting must never block the thread calling subscribe().
    """

    @abstractmethod
    def subscribe(self, wakeup: Callable[[], None]) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...


_EMPTY = object()


class _DataSourceIterator:
    """Stage one batch for Arrow, keeping wait tokens out of its data stream."""

    def __init__(self, source: Iterator[pa.RecordBatch | _DataSourceWait]):
        self._source = iter(source)
        self._wait: _DataSourceWait | None = None
        self._item: object = _EMPTY
        self._error: BaseException | None = None
        self._done = False

    def poll(self, wakeup: Callable[[], None]) -> bool:
        if self._item is not _EMPTY or self._error is not None or self._done:
            return True
        try:
            while True:
                if self._wait is not None:
                    if not self._wait.subscribe(wakeup):
                        return False
                    self._wait = None
                item = next(self._source)
                if isinstance(item, _DataSourceWait):
                    self._wait = item
                else:
                    self._item = item
                    return True
        except StopIteration:
            self._done = True
        except BaseException as error:
            # Let Arrow report Python failures through its ordinary get_next
            # path; the native boundary restores classified video errors.
            self._error = error
        return True

    def __iter__(self) -> _DataSourceIterator:
        return self

    def __next__(self) -> pa.RecordBatch:
        if self._error is not None:
            error, self._error = self._error, None
            self._done = True
            raise error
        if self._done:
            raise StopIteration
        if self._item is _EMPTY:
            raise RuntimeError("DataSource Arrow read requires a ready poll")
        item, self._item = self._item, _EMPTY
        return item  # type: ignore[return-value]

    def close(self) -> None:
        self._done = True
        try:
            if self._wait is not None:
                self._wait.close()
        finally:
            self._wait = None
            self._item = _EMPTY
            self._error = None
            source, self._source = self._source, iter(())
            close = getattr(source, "close", None)
            if close is not None:
                close()
