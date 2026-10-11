# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded request dispatch on an executor-owned event loop."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar, cast

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


async def map_async_ordered(
    items: Iterable[_Input],
    invoke: Callable[[_Input, Callable[[], None]], Awaitable[_Output]],
    *,
    max_concurrency: int,
) -> list[_Output]:
    """Run bounded work, retaining input order and awaiting cancellation cleanup.

    At most ``max_concurrency`` worker tasks exist. Each claims another input
    only after its previous invocation finishes. An invocation owns its own
    retries and provider cleanup; this dispatcher never replays work or creates
    an event loop. A failed worker stops admission before siblings can claim
    queued inputs, even before the coordinator observes the failure. ``invoke``
    receives a synchronous cancellation check to call before each request
    attempt, including retries and any split requests.
    """
    if type(max_concurrency) is not int or max_concurrency < 1:
        raise ValueError("max_concurrency must be a positive integer")
    iterator = iter(items)
    results: list[_Output] = []
    stopped = asyncio.Event()
    waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def is_stopped() -> bool:
        # Task.cancel() cancels its awaited Future synchronously. Workers can
        # observe this even when they resume before the coordinator does.
        return stopped.is_set() or waiter.cancelled()

    def check_stopped() -> None:
        if is_stopped():
            raise asyncio.CancelledError

    def claim() -> tuple[int, _Input] | None:
        if is_stopped():
            return None
        try:
            item = next(iterator)
        except StopIteration:
            return None
        index = len(results)
        results.append(cast(_Output, None))
        return index, item

    async def worker(current: tuple[int, _Input]) -> None:
        try:
            while not is_stopped():
                index, item = current
                results[index] = await invoke(item, check_stopped)
                following = claim()
                if following is None:
                    return
                current = following
        except BaseException:
            stopped.set()
            raise

    def complete(future: asyncio.Future[list[None]]) -> None:
        if waiter.done():
            return
        if future.cancelled():
            waiter.cancel()
        elif (error := future.exception()) is not None:
            waiter.set_exception(error)
        else:
            waiter.set_result(None)

    tasks: list[asyncio.Task[None]] = []
    joined: asyncio.Future[list[None]] | None = None
    try:
        for _ in range(max_concurrency):
            current = claim()
            if current is None:
                break
            tasks.append(asyncio.create_task(worker(current)))
        if not tasks:
            return []
        joined = asyncio.gather(*tasks)
        joined.add_done_callback(complete)
        # Await a separate Future so parent cancellation closes admission
        # immediately without forwarding repeated cancel() calls to workers.
        # The exception path cancels them once and shields their cleanup.
        await waiter
    except BaseException:
        stopped.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        drained = asyncio.gather(*tasks, return_exceptions=True)
        while not drained.done():
            try:
                await asyncio.shield(drained)
            except asyncio.CancelledError:
                continue
        # Observe the original gather too when the parent was cancelled before
        # it completed, otherwise its exception would be reported as unhandled.
        if joined is not None and joined.done() and not joined.cancelled():
            joined.exception()
        raise
    return results
