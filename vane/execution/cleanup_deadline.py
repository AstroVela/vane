# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""One monotonic budget across nested native and Ray cleanup calls."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_deadline: ContextVar[float | None] = ContextVar("vane_cleanup_deadline", default=None)


def cleanup_timeout(limit: float) -> float:
    deadline = _deadline.get()
    if deadline is None:
        return limit
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("cleanup deadline exceeded; retry close()")
    return min(limit, remaining)


@contextmanager
def cleanup_deadline(deadline: float) -> Iterator[None]:
    current = _deadline.get()
    token = _deadline.set(deadline if current is None else min(current, deadline))
    try:
        yield
    finally:
        _deadline.reset(token)
