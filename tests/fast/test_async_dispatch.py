# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Request admission, ordering and cleanup without model or transport I/O."""

from __future__ import annotations

import asyncio

import pytest

from vane.execution._async_dispatch import map_async_ordered


def test_dispatch_is_lazy_bounded_and_restores_completion_order():
    async def run():
        gates = [asyncio.Event() for _ in range(6)]
        claimed, started, finished = [], [], []
        baseline = asyncio.all_tasks()

        def inputs():
            for value in range(6):
                claimed.append(value)
                yield value

        async def invoke(value, check_stopped):
            started.append(value)
            await gates[value].wait()
            finished.append(value)
            return value * 10

        task = asyncio.create_task(map_async_ordered(inputs(), invoke, max_concurrency=2))
        while len(started) < 2:
            await asyncio.sleep(0)
        assert claimed == started == [0, 1]
        assert len(asyncio.all_tasks() - baseline) == 3  # coordinator and two workers
        for value in [1, 2, 3, 4, 5, 0]:
            gates[value].set()
            while value not in finished:
                await asyncio.sleep(0)
        assert await task == [0, 10, 20, 30, 40, 50]
        assert finished == [1, 2, 3, 4, 5, 0]
        assert asyncio.all_tasks() == baseline

    asyncio.run(asyncio.wait_for(run(), 5))


@pytest.mark.parametrize("failed", [0, 1])
@pytest.mark.parametrize("cancelled", [False, True])
def test_worker_failure_stops_dispatch_before_coordinator_wakes(failed, cancelled):
    async def run():
        gates = [asyncio.get_running_loop().create_future() for _ in range(2)]
        started = []

        async def invoke(value, check_stopped):
            started.append(value)
            await gates[value]
            return value

        task = asyncio.create_task(map_async_ordered(range(10), invoke, max_concurrency=2))
        while len(started) < 2:
            await asyncio.sleep(0)
        if cancelled:
            gates[failed].cancel()
        else:
            gates[failed].set_exception(ValueError("failed input"))
        gates[1 - failed].set_result(None)
        with pytest.raises(asyncio.CancelledError if cancelled else ValueError):
            await task
        assert started == [0, 1]

    asyncio.run(asyncio.wait_for(run(), 5))


@pytest.mark.parametrize("ready_worker", [0, 1])
def test_parent_cancellation_stops_ready_workers_before_coordinator_resumes(ready_worker):
    async def run():
        gates = [asyncio.get_running_loop().create_future() for _ in range(2)]
        claimed, started = [], []
        baseline = asyncio.all_tasks()

        def inputs():
            for value in range(8):
                claimed.append(value)
                yield value

        async def invoke(value, check_stopped):
            started.append(value)
            if value < 2:
                await gates[value]
            return value

        task = asyncio.create_task(map_async_ordered(inputs(), invoke, max_concurrency=2))
        while len(started) < 2:
            await asyncio.sleep(0)
        # Make a worker runnable before cancel() schedules the coordinator.
        # The remaining invocations would complete without yielding.
        gates[ready_worker].set_result(None)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert started == claimed == [0, 1]
        assert asyncio.all_tasks() == baseline

    asyncio.run(asyncio.wait_for(run(), 5))


def test_parent_cancellation_prevents_primed_workers_from_starting():
    async def run():
        started = []

        async def invoke(value, check_stopped):
            started.append(value)
            return value

        task = asyncio.create_task(map_async_ordered(range(8), invoke, max_concurrency=2))
        await asyncio.sleep(0)  # coordinator primes workers and awaits its waiter
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert started == []

    asyncio.run(asyncio.wait_for(run(), 5))


def test_repeated_parent_cancellation_waits_for_provider_cleanup():
    async def run():
        started, cleaning, cleaned = [], [], []
        release = asyncio.Event()
        baseline = asyncio.all_tasks()

        async def invoke(value, check_stopped):
            started.append(value)
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.append(value)
                await release.wait()
                cleaned.append(value)

        task = asyncio.create_task(map_async_ordered(range(10), invoke, max_concurrency=2))
        try:
            while len(started) < 2:
                await asyncio.sleep(0)
            task.cancel()
            while len(cleaning) < 2:
                await asyncio.sleep(0)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert sorted(cleaned) == started == [0, 1]
            assert asyncio.all_tasks() == baseline
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(asyncio.wait_for(run(), 5))


@pytest.mark.parametrize("failure_at", [1, 3])
def test_iterator_failure_cleans_up_created_workers(failure_at):
    async def run():
        active = 0
        baseline = asyncio.all_tasks()

        def inputs():
            yield from range(failure_at)
            raise ValueError("input iterator failed")

        async def invoke(value, check_stopped):
            nonlocal active
            active += 1
            try:
                await asyncio.sleep(0)
                return value
            finally:
                active -= 1

        with pytest.raises(ValueError, match="input iterator failed"):
            await map_async_ordered(inputs(), invoke, max_concurrency=2)
        assert active == 0
        assert asyncio.all_tasks() == baseline

    asyncio.run(run())


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, None])
def test_invalid_concurrency_is_rejected_before_consuming_inputs(limit):
    def inputs():
        pytest.fail("invalid configuration consumed an input")
        yield None

    async def invoke(value, check_stopped):
        return value

    with pytest.raises(ValueError, match="positive integer"):
        asyncio.run(map_async_ordered(inputs(), invoke, max_concurrency=limit))


def test_empty_inputs_and_none_results():
    async def invoke(value, check_stopped):
        return None

    async def run():
        assert await map_async_ordered([], invoke, max_concurrency=100) == []
        assert await map_async_ordered([1, 2], invoke, max_concurrency=100) == [None, None]

    asyncio.run(run())
