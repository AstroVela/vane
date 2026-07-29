"""Actor-owned event-loop runtime for async UDF callables.

UDF actors execute batches serially on a single thread, so a UDF callable
that drives async work (e.g. an AI provider's async SDK client) needs one
long-lived event loop for the actor's lifetime: driving each batch with a
fresh ``asyncio.run()`` strands loop-bound state (HTTP connection pools,
semaphores, tasks) on a closed loop and fails intermittently with
``RuntimeError: Event loop is closed`` from the second batch on.

The executor owns an :class:`AsyncRuntime` and hands its ``run`` method to
the callable (see ``UDFExecutor``); callables never create loops or threads
themselves. The loop is driven with ``run_until_complete`` on the calling
thread — no background thread — and is closed explicitly when the executor
shuts down.
"""

from __future__ import annotations

import asyncio
from typing import Any


class AsyncRuntime:
    """One lazily-created event loop, driven on the caller's thread.

    Not thread-safe: like the UDF executor and user callables, an instance
    must only be used from the single thread that executes actor batches.
    ``run()`` raises if that thread is already running an event loop —
    callers inside async code must ``await`` instead of using this bridge.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The owned loop, or ``None`` before first use / after close()."""
        return self._loop

    def run(self, coro: Any) -> Any:
        """Drive *coro* to completion on the owned loop and return its result."""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def close(self) -> None:
        """Cancel outstanding tasks and close the loop. Idempotent.

        Mirrors ``asyncio.run``'s teardown: background tasks a coroutine
        left behind are cancelled and awaited (so their ``finally`` blocks
        run) before async generators, the default executor, and the loop
        itself are shut down.
        """
        loop, self._loop = self._loop, None
        if loop is None or loop.is_closed():
            return
        try:
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()
            if tasks:
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            loop.close()
