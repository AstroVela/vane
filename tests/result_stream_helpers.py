# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any


async def collect_result_stream_async(stream: Any) -> list[Any]:
    """Consume the native one-shot readiness protocol from an asyncio loop."""

    loop = asyncio.get_running_loop()
    ready = asyncio.Event()
    stream.set_ready_callback(loop, ready.set)
    results = []
    try:
        while True:
            try:
                item = stream.next_nowait()
            except (StopIteration, StopAsyncIteration):
                return results
            except RuntimeError as error:
                # pybind11 can translate StopIteration through RuntimeError.
                if "StopIteration" in str(error):
                    return results
                raise
            if item is not None:
                results.append(item)
                continue
            ready.clear()
            stream.arm_ready_notification()
            await ready.wait()
    finally:
        stream.clear_ready_callback()


def collect_result_stream(stream: Any) -> list[Any]:
    return asyncio.run(collect_result_stream_async(stream))
