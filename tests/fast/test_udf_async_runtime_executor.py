# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""The UDF executor owns the async runtime it hands to async callables.

Covers vane#139 at the executor layer: ``UDFExecutor`` creates one
``AsyncRuntime`` per executor for callables that declare a
``bind_async_runtime`` hook, drives it with ``run_until_complete`` on the
actor's own execution thread (no background thread), and releases the
callable's resources plus the loop in ``close()``.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa


def _pickle(obj):
    cloudpickle = pytest.importorskip("cloudpickle")
    return cloudpickle.dumps(obj)


class AsyncBatchCallable:
    """Actor-style callable that drives its batch through the bound runtime."""

    def __init__(self) -> None:
        self.run_async = None
        self.seen_loops: list[asyncio.AbstractEventLoop] = []
        self.seen_threads: list[int] = []
        self.closed = False

    def bind_async_runtime(self, run_async) -> None:
        self.run_async = run_async

    def close(self) -> None:
        self.closed = True

    def __call__(self, table: pa.Table) -> pa.Table:
        async def _work() -> pa.Table:
            import threading

            self.seen_loops.append(asyncio.get_running_loop())
            self.seen_threads.append(threading.get_ident())
            return table

        return self.run_async(_work())


def _actor_map_batches_payload(cls) -> dict:
    return {
        "function_pickle": _pickle(cls),
        "call_mode": "map_batches",
        "execution_backend": "subprocess_actor",
        "actor_number": 1,
    }


# ---------------------------------------------------------------------------
# AsyncRuntime unit behavior
# ---------------------------------------------------------------------------


def test_async_runtime_lazy_loop_reused_across_runs():
    from duckdb.execution._async_runtime import AsyncRuntime

    runtime = AsyncRuntime()
    assert runtime.loop is None

    async def loop_of() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    first = runtime.run(loop_of())
    second = runtime.run(loop_of())

    assert first is second is runtime.loop
    runtime.close()


def test_async_runtime_runs_on_the_calling_thread():
    import threading

    from duckdb.execution._async_runtime import AsyncRuntime

    runtime = AsyncRuntime()

    async def thread_of() -> int:
        return threading.get_ident()

    assert runtime.run(thread_of()) == threading.get_ident()
    runtime.close()


def test_async_runtime_close_is_idempotent_and_closes_loop():
    from duckdb.execution._async_runtime import AsyncRuntime

    runtime = AsyncRuntime()

    async def nothing() -> None:
        return None

    runtime.run(nothing())
    loop = runtime.loop
    runtime.close()
    runtime.close()

    assert loop.is_closed()
    assert runtime.loop is None


def test_async_runtime_close_before_first_run_is_noop():
    from duckdb.execution._async_runtime import AsyncRuntime

    runtime = AsyncRuntime()
    runtime.close()
    assert runtime.loop is None


def test_async_runtime_close_finalizes_async_generators():
    from duckdb.execution._async_runtime import AsyncRuntime

    runtime = AsyncRuntime()
    finalized = []

    async def agen():
        try:
            yield 1
            yield 2
        finally:
            finalized.append(True)

    async def start() -> int:
        gen = agen()
        return await gen.__anext__()

    assert runtime.run(start()) == 1
    assert not finalized
    runtime.close()
    assert finalized == [True]


def test_async_runtime_rejects_nested_use_no_fallback():
    """Inside async code callers must await; there is no bridge thread."""
    from duckdb.execution._async_runtime import AsyncRuntime

    outer = AsyncRuntime()
    inner = AsyncRuntime()

    async def nested() -> None:
        async def noop() -> None:
            return None

        inner.run(noop())

    with pytest.raises(RuntimeError):
        outer.run(nested())
    outer.close()
    inner.close()


# ---------------------------------------------------------------------------
# Executor wiring
# ---------------------------------------------------------------------------


def test_executor_binds_runtime_and_batches_share_one_loop():
    from duckdb.execution._udf_runtime import UDFExecutor

    executor = UDFExecutor(_actor_map_batches_payload(AsyncBatchCallable))
    executor.submit(pa.table({"x": [1]}))
    executor.submit(pa.table({"x": [2, 3]}))
    executor.finished_submitting()
    outputs = executor.drain_outputs()

    callable_instance = executor._map_fn
    assert sum(t.num_rows for t in outputs) == 3
    assert len(callable_instance.seen_loops) == 2
    assert len(set(callable_instance.seen_loops)) == 1

    import threading

    assert set(callable_instance.seen_threads) == {threading.get_ident()}

    loop = callable_instance.seen_loops[0]
    executor.close()
    assert callable_instance.closed
    assert loop.is_closed()


def test_executor_close_is_idempotent():
    from duckdb.execution._udf_runtime import UDFExecutor

    executor = UDFExecutor(_actor_map_batches_payload(AsyncBatchCallable))
    executor.close()
    executor.close()


def test_executor_without_hook_binds_nothing():
    from duckdb.execution._udf_runtime import UDFExecutor

    def identity(table: pa.Table) -> pa.Table:
        return table

    executor = UDFExecutor(
        {
            "function_pickle": _pickle(identity),
            "call_mode": "map_batches",
            "execution_backend": "subprocess_task",
        }
    )

    assert executor._async_runtime is None
    executor.close()  # no-op without a bound runtime
