# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import math
import os
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING, Any

from vane._ray_errors import restore_remote_ray_exception
from vane.runners.common import QueryDeadlineExceeded as QueryDeadlineExceeded

if TYPE_CHECKING:
    from collections.abc import Callable


def _positive_float_env(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _configured_ray_get_timeout(
    timeout: float | None = None,
    *,
    honor_query_deadline: bool = True,
    honor_object_get_timeout: bool = True,
) -> tuple[float | None, bool]:
    resolved_timeout = max(0.0, float(timeout)) if timeout is not None else None
    query_deadline_limited = False
    deadline = _positive_float_env("VANE_QUERY_DEADLINE_EPOCH_S") if honor_query_deadline else None
    if deadline is not None:
        remaining = deadline - time.time()
        if remaining <= 0.0:
            raise QueryDeadlineExceeded("query deadline expired before Ray ObjectRef get")
        if resolved_timeout is None or remaining <= resolved_timeout:
            resolved_timeout = remaining
            query_deadline_limited = True
    configured = _positive_float_env("VANE_RAY_OBJECT_GET_TIMEOUT_S") if honor_object_get_timeout else None
    if configured is not None and (resolved_timeout is None or configured < resolved_timeout):
        resolved_timeout = configured
        query_deadline_limited = False
    return resolved_timeout, query_deadline_limited


def configured_ray_get_timeout_s(
    timeout: float | None = None,
    *,
    honor_query_deadline: bool = True,
    honor_object_get_timeout: bool = True,
) -> float | None:
    resolved_timeout, _ = _configured_ray_get_timeout(
        timeout,
        honor_query_deadline=honor_query_deadline,
        honor_object_get_timeout=honor_object_get_timeout,
    )
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
    timeout: float | None,
    deadline: float | None,
    query_deadline_limited: bool,
    on_wait: Callable[[], None] | None,
    wait_interval_s: float,
) -> Any:
    if on_wait is None:
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as error:
            done = getattr(future, "done", None)
            if callable(done) and done():
                return future.result()
            if query_deadline_limited:
                raise QueryDeadlineExceeded("query deadline expired while waiting for Ray ObjectRef") from error
            raise

    while True:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        wait_timeout = wait_interval_s if remaining is None else min(wait_interval_s, remaining)
        try:
            return future.result(timeout=wait_timeout)
        except FutureTimeoutError as error:
            done = getattr(future, "done", None)
            if callable(done) and done():
                return future.result()
            if deadline is not None and time.monotonic() >= deadline:
                if query_deadline_limited:
                    raise QueryDeadlineExceeded("query deadline expired while waiting for Ray ObjectRef") from error
                raise
            on_wait()


def _resolve_object_refs(
    object_refs: Any,
    timeout: float | None,
    *,
    query_deadline_limited: bool,
    on_wait: Callable[[], None] | None,
    wait_interval_s: float,
) -> Any:
    deadline = None if timeout is None else time.monotonic() + timeout
    if not isinstance(object_refs, list | tuple):
        future = _object_ref_future(object_refs)
        return _resolve_future(
            future,
            timeout=timeout,
            deadline=deadline,
            query_deadline_limited=query_deadline_limited,
            on_wait=on_wait,
            wait_interval_s=wait_interval_s,
        )
    if not object_refs:
        return []

    futures = [_object_ref_future(ref) for ref in object_refs]
    results = []
    for future in futures:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        results.append(
            _resolve_future(
                future,
                timeout=remaining,
                deadline=deadline,
                query_deadline_limited=query_deadline_limited,
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
    honor_object_get_timeout: bool = True,
    on_wait: Callable[[], None] | None = None,
    wait_interval_s: float = 0.5,
) -> Any:
    """Resolve Ray ObjectRefs only from a thread without a running event loop.

    This is a synchronous API. Async actor methods must await ObjectRefs at the
    call site; native pollers and other worker-owned threads wait through the
    ObjectRef concurrent-future bridge. When ``on_wait`` is provided, one total
    timeout is divided into bounded waits and the callback runs between them.
    Callers with a dedicated hard timeout may opt out of the process-wide
    ObjectRef timeout without disabling that explicit bound.
    """
    timeout, query_deadline_limited = _configured_ray_get_timeout(
        timeout,
        honor_query_deadline=honor_query_deadline,
        honor_object_get_timeout=honor_object_get_timeout,
    )
    wait_interval_s = float(wait_interval_s)
    if on_wait is not None and wait_interval_s <= 0:
        raise ValueError("wait_interval_s must be positive")

    _reject_running_event_loop()
    restored: BaseException | None = None
    try:
        return _resolve_object_refs(
            object_refs,
            timeout,
            query_deadline_limited=query_deadline_limited,
            on_wait=on_wait,
            wait_interval_s=wait_interval_s,
        )
    except BaseException as exc:
        restored = restore_remote_ray_exception(exc)
        if restored is None:
            raise
    # Raising outside the handler keeps Python from replacing a restored
    # implicit remote context with the local RayTaskError.
    raise restored
