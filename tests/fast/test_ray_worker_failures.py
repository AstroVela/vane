import asyncio

import pytest


class BadTaskHandle:
    def get_result(self):
        async def coro():
            raise RuntimeError("simulated failure")

        return coro()


class WDTaskHandle:
    def get_result(self):
        async def coro():
            return ("WorkerDied", {"reason": "killed"})

        return coro()


class BadFormatHandle:
    def get_result(self):
        async def coro():
            return "not a tuple"

        return coro()


def test_task_handle_raises():
    h = BadTaskHandle()
    with pytest.raises(RuntimeError):
        asyncio.run(h.get_result())


def test_worker_died_format():
    h = WDTaskHandle()
    res = asyncio.run(h.get_result())
    assert isinstance(res, tuple)
    assert res[0] == "WorkerDied"


def test_invalid_format_not_tuple():
    res = asyncio.run(BadFormatHandle().get_result())
    assert not isinstance(res, tuple)
