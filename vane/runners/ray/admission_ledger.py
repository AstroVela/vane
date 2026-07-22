# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded replay and terminal-identity ledgers for query admission."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, MutableMapping, MutableSet
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


def _positive_capacity(capacity: int) -> int:
    value = int(capacity)
    if value <= 0:
        raise ValueError("ledger capacity must be positive")
    return value


class BoundedSet(MutableSet[K], Generic[K]):
    """Exact insertion-ordered set with a fixed replay horizon."""

    def __init__(self, values=(), *, capacity: int) -> None:
        self._capacity = _positive_capacity(capacity)
        self._values: OrderedDict[K, None] = OrderedDict()
        for value in values:
            self.add(value)

    def __contains__(self, value: object) -> bool:
        return value in self._values

    def __iter__(self) -> Iterator[K]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def add(self, value: K) -> None:
        if value in self._values:
            self._values.move_to_end(value)
            return
        self._values[value] = None
        if len(self._values) > self._capacity:
            self._values.popitem(last=False)

    def discard(self, value: K) -> None:
        self._values.pop(value, None)

    def clear(self) -> None:
        self._values.clear()


class BoundedReplayMap(MutableMapping[K, V], Generic[K, V]):
    """Dictionary-compatible terminal replay window with bounded memory."""

    def __init__(self, values=(), *, capacity: int) -> None:
        self._capacity = _positive_capacity(capacity)
        self._values: OrderedDict[K, V] = OrderedDict()
        for key, value in dict(values).items():
            self[key] = value

    def __getitem__(self, key: K) -> V:
        return self._values[key]

    def __setitem__(self, key: K, value: V) -> None:
        if key in self._values:
            self._values.move_to_end(key)
        self._values[key] = value
        if len(self._values) > self._capacity:
            self._values.popitem(last=False)

    def __delitem__(self, key: K) -> None:
        del self._values[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def clear(self) -> None:
        self._values.clear()

    def discard_where(self, predicate: Callable[[K, V], bool]) -> None:
        for key, value in list(self._values.items()):
            if predicate(key, value):
                self._values.pop(key, None)


__all__ = ["BoundedReplayMap", "BoundedSet"]
