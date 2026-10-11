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
    invoke: Callable[[_Input], Awaitable[_Output]],
    *,
    max_concurrency: int,
) -> list[_Output]:
    """Run bounded work, retaining input order and awaiting cancellation cleanup.

    At most ``max_concurrency`` worker tasks exist. Each claims another input
    only after its previous invocation finishes. An invocation owns its own
    retries and provider cleanup; this dispatcher never replays work or creates
    an event loop. A failed worker stops admission before siblings can claim
    queued inputs, even before the coordinator observes the failure.
    """
    if type(max_concurrency) is not int or max_concurrency < 1:
        raise ValueError("max_concurrency must be a positive integer")
    iterator = iter(items)
    results: list[_Output] = []
    stopped = asyncio.Event()

    def claim() -> tuple[int, _Input] | None:
        try:
            item = next(iterator)
        except StopIteration:
            return None
        index = len(results)
        results.append(cast(_Output, None))
        return index, item

    async def worker(current: tuple[int, _Input]) -> None:
        try:
            while not stopped.is_set():
                index, item = current
                results[index] = await invoke(item)
                if stopped.is_set():
                    return
                following = claim()
                if following is None:
                    return
                current = following
        except BaseException:
            stopped.set()
            raise

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
        # Shield so cancellation is delivered to each worker only once. A
        # second Task.cancel() must not interrupt a provider's finally block.
        await asyncio.shield(joined)
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
