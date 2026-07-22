# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING, Any

from vane.runners.common import QueryDeadlineExceeded

if TYPE_CHECKING:
    from collections.abc import Callable


def _positive_float_env(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    value = float(raw)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def configured_ray_get_timeout_s(
    timeout: float | None = None,
    *,
    honor_query_deadline: bool = True,
) -> float | None:
    resolved_timeout = max(0.0, float(timeout)) if timeout is not None else None
    deadline = _positive_float_env("VANE_QUERY_DEADLINE_EPOCH_S") if honor_query_deadline else None
    if deadline is not None:
        remaining = deadline - time.time()
        if remaining <= 0.0:
            raise QueryDeadlineExceeded("query deadline expired before Ray ObjectRef get")
        resolved_timeout = remaining if resolved_timeout is None else min(resolved_timeout, remaining)
    configured = _positive_float_env("VANE_RAY_OBJECT_GET_TIMEOUT_S")
    if configured is not None:
        resolved_timeout = configured if resolved_timeout is None else min(resolved_timeout, configured)
    return resolved_timeout


def _reject_running_event_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        "blocking ObjectRef resolution cannot run on an event loop; await the ObjectRef at the async call site"
    )


def _object_ref_future(ref: Any) -> Any:
    future = getattr(ref, "future", None)
    if not callable(future):
        raise TypeError(f"expected Ray ObjectRef with future(), got {type(ref).__name__}")
    return future()


def _resolve_future(
    future: Any,
    *,
    deadline: float | None,
    on_wait: Callable[[], None] | None,
    wait_interval_s: float,
) -> Any:
    if on_wait is None:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        return future.result(timeout=remaining)

    while True:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        wait_timeout = wait_interval_s if remaining is None else min(wait_interval_s, remaining)
        try:
            return future.result(timeout=wait_timeout)
        except FutureTimeoutError:
            done = getattr(future, "done", None)
            if callable(done) and done():
                raise
            if deadline is not None and time.monotonic() >= deadline:
                raise
            on_wait()


def _resolve_object_refs(
    object_refs: Any,
    timeout: float | None,
    *,
    on_wait: Callable[[], None] | None,
    wait_interval_s: float,
) -> Any:
    if not isinstance(object_refs, list | tuple):
        future = _object_ref_future(object_refs)
        if on_wait is None:
            return future.result(timeout=timeout)
        deadline = None if timeout is None else time.monotonic() + timeout
        return _resolve_future(
            future,
            deadline=deadline,
            on_wait=on_wait,
            wait_interval_s=wait_interval_s,
        )
    if not object_refs:
        return []

    futures = [_object_ref_future(ref) for ref in object_refs]
    deadline = None if timeout is None else time.monotonic() + timeout
    results = []
    for future in futures:
        results.append(
            _resolve_future(
                future,
                deadline=deadline,
                on_wait=on_wait,
                wait_interval_s=wait_interval_s,
            )
        )
    return results


def resolve_object_refs_blocking(
    object_refs: Any,
    *,
    timeout: float | None = None,
    honor_query_deadline: bool = True,
    on_wait: Callable[[], None] | None = None,
    wait_interval_s: float = 0.5,
) -> Any:
    """Resolve Ray ObjectRefs only from a thread without a running event loop.

    This is a synchronous API. Async actor methods must await ObjectRefs at the
    call site; native pollers and other worker-owned threads wait through the
    ObjectRef concurrent-future bridge. When ``on_wait`` is provided, one total
    timeout is divided into bounded waits and the callback runs between them.
    """
    timeout = configured_ray_get_timeout_s(
        timeout,
        honor_query_deadline=honor_query_deadline,
    )
    wait_interval_s = float(wait_interval_s)
    if on_wait is not None and wait_interval_s <= 0:
        raise ValueError("wait_interval_s must be positive")

    _reject_running_event_loop()
    return _resolve_object_refs(
        object_refs,
        timeout,
        on_wait=on_wait,
        wait_interval_s=wait_interval_s,
    )
