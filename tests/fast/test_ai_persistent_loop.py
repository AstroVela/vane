# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""AI batch wrappers drive every batch through one persistent event loop.

Covers vane#139: ``_EmbedTextBatch`` and ``_PromptBatch`` cache async SDK
clients across batches, so each wrapper owns a long-lived background event
loop instead of spinning up a fresh ``asyncio.run()`` per batch (which
stranded cached connection pools on closed loops). The loop must be reused
across batch calls, must not be pickled to actors (it is re-created lazily
after unpickling), must preserve semaphore-based concurrency limits and
result ordering, and must stay callable from a context that already runs an
event loop.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import pickle
import sys
import threading
import types

import pyarrow as pa
import pytest


def _load_functions() -> types.ModuleType:
    """Import the real ``vane.ai.functions``, even under the no-duckdb harness.

    The stub harness plugin registers a placeholder for ``vane.ai.functions``
    because the real module imports duckdb-backed modules at top level. When
    that placeholder is present, evict it and stub just the duckdb-importing
    dependencies so the real module under test can load. On CI (where duckdb
    imports fine) the real module imports normally and nothing is stubbed.
    """
    module = sys.modules.get("vane.ai.functions")
    if getattr(module, "__file__", None):
        return module  # real module already loaded
    if module is not None:
        sys.modules.pop("vane.ai.functions")
        stub_specs: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("vane._expressions", ("as_expression", "is_expression")),
            ("vane._expression_udf", ("_build_actor_map_batches_expression",)),
        )
        for name, attrs in stub_specs:
            if name not in sys.modules:
                stub = types.ModuleType(name)
                for attr in attrs:
                    setattr(stub, attr, lambda *a, **k: None)
                sys.modules[name] = stub
    return importlib.import_module("vane.ai.functions")


functions = _load_functions()

_EmbedTextBatch = functions._EmbedTextBatch
_PersistentLoopRunner = functions._PersistentLoopRunner
_PromptBatch = functions._PromptBatch
RetryAfterError = functions.RetryAfterError


def test_real_functions_module_is_under_test() -> None:
    """Guard: the harness must import the real module, not the plugin stub."""
    assert getattr(functions, "__file__", None)
    assert isinstance(_PromptBatch, type)


# ---------------------------------------------------------------------------
# Picklable fakes (module level so pickle can resolve them by reference)
# ---------------------------------------------------------------------------


class LoopRecordingPrompter:
    """Async prompter recording id() of the running loop for each call."""

    def __init__(self) -> None:
        self.loop_ids: list[int] = []

    async def prompt(self, messages: tuple) -> str:
        self.loop_ids.append(id(asyncio.get_running_loop()))
        return f"echo:{messages[0]}"


class LoopRecordingDescriptor:
    def instantiate(self) -> LoopRecordingPrompter:
        return LoopRecordingPrompter()


class ConcurrencyTrackingPrompter:
    """Async prompter tracking the peak number of in-flight calls."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def prompt(self, messages: tuple) -> str:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        return f"echo:{messages[0]}"


class ConcurrencyTrackingDescriptor:
    def instantiate(self) -> ConcurrencyTrackingPrompter:
        return ConcurrencyTrackingPrompter()


class FlakyOncePrompter:
    """Fails the first call with a zero-wait retryable error, then succeeds."""

    def __init__(self) -> None:
        self.calls = 0
        self.loop_ids: list[int] = []

    async def prompt(self, messages: tuple) -> str:
        self.calls += 1
        self.loop_ids.append(id(asyncio.get_running_loop()))
        if self.calls == 1:
            raise RetryAfterError(0.0, ValueError("transient"))
        return f"echo:{messages[0]}"


class FlakyOnceDescriptor:
    def instantiate(self) -> FlakyOncePrompter:
        return FlakyOncePrompter()


class AsyncLoopRecordingEmbedder:
    """Async embedder recording id() of the running loop for each call."""

    def __init__(self) -> None:
        self.loop_ids: list[int] = []

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        self.loop_ids.append(id(asyncio.get_running_loop()))
        # Unit vector, exactly representable in float32.
        return [[1.0, 0.0] for _ in texts]


class AsyncEmbedderDescriptor:
    def instantiate(self) -> AsyncLoopRecordingEmbedder:
        return AsyncLoopRecordingEmbedder()


def _messages_table(n: int) -> pa.Table:
    return pa.table({"messages": [f"m{i}" for i in range(n)]})


# ---------------------------------------------------------------------------
# _PersistentLoopRunner
# ---------------------------------------------------------------------------


class TestPersistentLoopRunner:
    def test_same_loop_serves_successive_runs(self) -> None:
        runner = _PersistentLoopRunner()
        try:

            async def which() -> asyncio.AbstractEventLoop:
                return asyncio.get_running_loop()

            first = runner.run(which())
            second = runner.run(which())
            assert first is second
            assert first is runner._loop
            assert not first.is_closed()
        finally:
            runner.close()

    def test_exception_propagates_and_loop_survives(self) -> None:
        runner = _PersistentLoopRunner()
        try:

            async def boom() -> None:
                raise ValueError("boom")

            async def ok() -> str:
                return "still-works"

            with pytest.raises(ValueError, match="boom"):
                runner.run(boom())
            assert runner.run(ok()) == "still-works"
        finally:
            runner.close()

    def test_close_stops_thread_and_closes_loop(self) -> None:
        runner = _PersistentLoopRunner()

        async def noop() -> None:
            return None

        runner.run(noop())
        loop, thread = runner._loop, runner._thread
        runner.close()
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert loop.is_closed()
        assert runner._loop is None
        assert runner._thread is None

    def test_close_before_first_use_and_twice_is_noop(self) -> None:
        runner = _PersistentLoopRunner()
        runner.close()
        runner.close()

    def test_reuse_after_close_creates_fresh_loop(self) -> None:
        runner = _PersistentLoopRunner()
        try:

            async def which() -> asyncio.AbstractEventLoop:
                return asyncio.get_running_loop()

            first = runner.run(which())
            runner.close()
            second = runner.run(which())
            assert second is not first
            assert not second.is_closed()
            assert first.is_closed()
        finally:
            runner.close()

    def test_pickle_round_trip_is_fresh_and_lazy(self) -> None:
        runner = _PersistentLoopRunner()
        try:

            async def which() -> asyncio.AbstractEventLoop:
                return asyncio.get_running_loop()

            active_loop = runner.run(which())
            restored = pickle.loads(pickle.dumps(runner))
            try:
                assert restored._loop is None
                assert restored._thread is None
                restored_loop = restored.run(which())
                assert restored_loop is restored._loop
                assert restored_loop is not active_loop
            finally:
                restored.close()
        finally:
            runner.close()

    def test_run_from_inside_a_running_loop(self) -> None:
        runner = _PersistentLoopRunner()
        try:

            async def inner() -> asyncio.AbstractEventLoop:
                return asyncio.get_running_loop()

            async def outer() -> tuple[asyncio.AbstractEventLoop, asyncio.AbstractEventLoop]:
                # Synchronous bridge from within a running loop must not
                # deadlock and must execute on the runner's own loop.
                return runner.run(inner()), asyncio.get_running_loop()

            ran_on, caller_loop = asyncio.run(outer())
            assert ran_on is runner._loop
            assert ran_on is not caller_loop
        finally:
            runner.close()


# ---------------------------------------------------------------------------
# _PromptBatch on the persistent loop
# ---------------------------------------------------------------------------


class TestPromptBatchPersistentLoop:
    def test_two_batches_share_one_loop(self) -> None:
        wrapper = _PromptBatch(LoopRecordingDescriptor(), "messages", "response", None)
        try:
            out1 = wrapper(_messages_table(4))
            out2 = wrapper(_messages_table(4))
            expected = [f"echo:m{i}" for i in range(4)]
            assert out1.column("response").to_pylist() == expected
            assert out2.column("response").to_pylist() == expected
            prompter = wrapper._prompter
            assert len(prompter.loop_ids) == 8
            assert len(set(prompter.loop_ids)) == 1
            assert prompter.loop_ids[0] == id(wrapper._loop_runner._loop)
        finally:
            wrapper._loop_runner.close()

    def test_pickle_round_trip_then_call_works(self) -> None:
        # Pickling before first use mirrors shipping the wrapper to an actor.
        wrapper = _PromptBatch(LoopRecordingDescriptor(), "messages", "response", None)
        restored = pickle.loads(pickle.dumps(wrapper))
        try:
            assert restored._loop_runner._loop is None
            assert restored._loop_runner._thread is None
            out = restored(_messages_table(3))
            assert out.column("response").to_pylist() == ["echo:m0", "echo:m1", "echo:m2"]
            assert restored._prompter.loop_ids[0] == id(restored._loop_runner._loop)
        finally:
            restored._loop_runner.close()

    def test_pickle_after_use_does_not_ship_loop_state(self) -> None:
        wrapper = _PromptBatch(LoopRecordingDescriptor(), "messages", "response", None)
        try:
            wrapper(_messages_table(2))
            restored = pickle.loads(pickle.dumps(wrapper))
            try:
                assert restored._loop_runner._loop is None
                assert restored._loop_runner._thread is None
                out = restored(_messages_table(2))
                assert out.column("response").to_pylist() == ["echo:m0", "echo:m1"]
                assert restored._prompter.loop_ids[-1] == id(restored._loop_runner._loop)
            finally:
                restored._loop_runner.close()
        finally:
            wrapper._loop_runner.close()

    def test_semaphore_limits_concurrent_submissions(self) -> None:
        wrapper = _PromptBatch(ConcurrencyTrackingDescriptor(), "messages", "response", 2)
        try:
            out = wrapper(_messages_table(8))
            assert out.column("response").to_pylist() == [f"echo:m{i}" for i in range(8)]
            assert wrapper._prompter.peak == 2
        finally:
            wrapper._loop_runner.close()

    def test_no_semaphore_runs_all_concurrently(self) -> None:
        wrapper = _PromptBatch(ConcurrencyTrackingDescriptor(), "messages", "response", None)
        try:
            out = wrapper(_messages_table(6))
            assert out.column("response").to_pylist() == [f"echo:m{i}" for i in range(6)]
            assert wrapper._prompter.peak == 6
        finally:
            wrapper._loop_runner.close()

    def test_call_from_inside_a_running_loop(self) -> None:
        wrapper = _PromptBatch(LoopRecordingDescriptor(), "messages", "response", None)
        try:

            async def driver() -> pa.Table:
                return wrapper(_messages_table(2))

            out = asyncio.run(driver())
            assert out.column("response").to_pylist() == ["echo:m0", "echo:m1"]
            assert wrapper._prompter.loop_ids[0] == id(wrapper._loop_runner._loop)
        finally:
            wrapper._loop_runner.close()

    def test_retry_runs_on_the_same_loop(self) -> None:
        wrapper = _PromptBatch(FlakyOnceDescriptor(), "messages", "response", None, max_retries=1)
        try:
            out = wrapper(pa.table({"messages": ["only"]}))
            assert out.column("response").to_pylist() == ["echo:only"]
            prompter = wrapper._prompter
            assert prompter.calls == 2
            assert len(set(prompter.loop_ids)) == 1
            assert prompter.loop_ids[0] == id(wrapper._loop_runner._loop)
        finally:
            wrapper._loop_runner.close()


# ---------------------------------------------------------------------------
# _EmbedTextBatch on the persistent loop
# ---------------------------------------------------------------------------


class TestEmbedTextBatchPersistentLoop:
    def test_two_batches_share_one_loop(self) -> None:
        wrapper = _EmbedTextBatch(AsyncEmbedderDescriptor(), "text", "embedding")
        try:
            table = pa.table({"text": ["a", "b"]})
            out1 = wrapper(table)
            out2 = wrapper(table)
            assert out1.column("embedding").to_pylist() == [[1.0, 0.0], [1.0, 0.0]]
            assert out2.column("embedding").to_pylist() == [[1.0, 0.0], [1.0, 0.0]]
            embedder = wrapper._embedder
            assert len(embedder.loop_ids) == 2
            assert len(set(embedder.loop_ids)) == 1
            assert embedder.loop_ids[0] == id(wrapper._loop_runner._loop)
        finally:
            wrapper._loop_runner.close()

    def test_chunked_embedding_uses_the_persistent_loop(self) -> None:
        wrapper = _EmbedTextBatch(
            AsyncEmbedderDescriptor(),
            "text",
            "embedding",
            max_chunk_chars=10,
            chunk_overlap_chars=2,
        )
        try:
            out1 = wrapper(pa.table({"text": ["short", "x" * 25]}))
            out2 = wrapper(pa.table({"text": ["short", "x" * 25]}))
            assert out1.column("embedding").to_pylist() == [[1.0, 0.0], [1.0, 0.0]]
            assert out2.column("embedding").to_pylist() == [[1.0, 0.0], [1.0, 0.0]]
            embedder = wrapper._embedder
            assert len(embedder.loop_ids) == 2
            assert len(set(embedder.loop_ids)) == 1
            assert embedder.loop_ids[0] == id(wrapper._loop_runner._loop)
        finally:
            wrapper._loop_runner.close()

    def test_pickle_round_trip_then_call_works(self) -> None:
        wrapper = _EmbedTextBatch(AsyncEmbedderDescriptor(), "text", "embedding")
        restored = pickle.loads(pickle.dumps(wrapper))
        try:
            assert restored._loop_runner._loop is None
            assert restored._loop_runner._thread is None
            out = restored(pa.table({"text": ["a"]}))
            assert out.column("embedding").to_pylist() == [[1.0, 0.0]]
            assert restored._embedder.loop_ids[0] == id(restored._loop_runner._loop)
        finally:
            restored._loop_runner.close()


# ---------------------------------------------------------------------------
# Loop re-creation: stale loops are closed, cached clients are rebuilt
# ---------------------------------------------------------------------------


class TestLoopRecreation:
    def test_stale_loop_is_closed_on_recreation(self) -> None:
        runner = _PersistentLoopRunner()
        try:

            async def _noop() -> None:
                return None

            runner.run(_noop())
            first_loop = runner._loop
            first_thread = runner._thread
            # Simulate a fork: the thread is gone but the loop object survives
            # un-closed (threads never cross fork; the loop's fds do).
            first_loop.call_soon_threadsafe(first_loop.stop)
            first_thread.join(timeout=5.0)
            assert not first_thread.is_alive()
            assert not first_loop.is_closed()

            runner.run(_noop())
            assert runner._loop is not first_loop
            assert first_loop.is_closed()
        finally:
            runner.close()

    def test_prompter_is_rebuilt_when_loop_is_recreated(self) -> None:
        wrapper = _PromptBatch(LoopRecordingDescriptor(), "messages", "response", None)
        try:
            wrapper(_messages_table(2))
            first_prompter = wrapper._prompter
            first_loop = wrapper._prompter_loop
            wrapper._loop_runner.close()

            out = wrapper(_messages_table(2))
            assert out.column("response").to_pylist() == ["echo:m0", "echo:m1"]
            assert wrapper._prompter is not first_prompter
            assert wrapper._prompter_loop is not first_loop
            assert wrapper._prompter.loop_ids[0] == id(wrapper._loop_runner._loop)
        finally:
            wrapper._loop_runner.close()

    def test_embedder_is_rebuilt_when_loop_is_recreated(self) -> None:
        wrapper = _EmbedTextBatch(AsyncEmbedderDescriptor(), "text", "embedding")
        try:
            wrapper(pa.table({"text": ["a"]}))
            first_embedder = wrapper._embedder
            wrapper._loop_runner.close()

            out = wrapper(pa.table({"text": ["a"]}))
            assert out.column("embedding").to_pylist() == [[1.0, 0.0]]
            assert wrapper._embedder is not first_embedder
            assert wrapper._embedder.loop_ids[0] == id(wrapper._loop_runner._loop)
        finally:
            wrapper._loop_runner.close()

    def test_pickle_after_use_clears_cached_client(self) -> None:
        wrapper = _PromptBatch(LoopRecordingDescriptor(), "messages", "response", None)
        try:
            wrapper(_messages_table(1))
            assert wrapper._prompter is not None
            restored = pickle.loads(pickle.dumps(wrapper))
            try:
                assert restored._prompter is None
                assert restored._prompter_loop is None
                out = restored(_messages_table(1))
                assert out.column("response").to_pylist() == ["echo:m0"]
            finally:
                restored._loop_runner.close()
        finally:
            wrapper._loop_runner.close()

    def test_fork_zombie_running_loop_is_skipped_not_closed(self) -> None:
        """A forked child's inherited loop can claim to be running with no
        live thread; recreation must skip closing it instead of raising."""

        class RunningZombieLoop:
            def __init__(self) -> None:
                self.close_calls = 0

            def is_closed(self) -> bool:
                return False

            def is_running(self) -> bool:
                return True

            def close(self) -> None:
                self.close_calls += 1
                raise RuntimeError("Cannot close a running event loop")

        runner = _PersistentLoopRunner()
        zombie = RunningZombieLoop()
        dead_thread = threading.Thread(target=lambda: None, name="vane-ai-batch-loop")
        dead_thread.start()
        dead_thread.join(timeout=5.0)
        assert not dead_thread.is_alive()
        runner._loop = zombie
        runner._thread = dead_thread
        try:

            async def _noop() -> str:
                return "recovered"

            assert runner.run(_noop()) == "recovered"
            assert zombie.close_calls == 0
            assert runner._loop is not zombie
            assert not runner._loop.is_closed()
        finally:
            runner.close()

    def test_close_cancels_inflight_run_instead_of_hanging(self) -> None:
        """close() racing an active run() must cancel the pending future so
        the blocked caller raises CancelledError instead of hanging."""
        runner = _PersistentLoopRunner()
        started = threading.Event()
        outcome: list[object] = []

        async def _hang() -> None:
            started.set()
            await asyncio.sleep(30)

        def _caller() -> None:
            try:
                runner.run(_hang())
            except BaseException as exc:  # noqa: BLE001 - the assertion target
                outcome.append(exc)
            else:
                outcome.append("returned")

        caller = threading.Thread(target=_caller, daemon=True)
        caller.start()
        try:
            assert started.wait(timeout=5.0)
            runner.close()
            caller.join(timeout=5.0)
            assert not caller.is_alive()
            assert len(outcome) == 1
            assert isinstance(outcome[0], (asyncio.CancelledError, concurrent.futures.CancelledError))
        finally:
            runner.close()

    def test_close_from_the_loop_thread_does_not_raise(self) -> None:
        runner = _PersistentLoopRunner()
        try:
            loop = runner._ensure_loop()
            done = threading.Event()
            errors: list[BaseException] = []

            def _close_from_inside() -> None:
                try:
                    runner.close()
                except BaseException as exc:  # noqa: BLE001 - the assertion target
                    errors.append(exc)
                finally:
                    done.set()

            loop.call_soon_threadsafe(_close_from_inside)
            assert done.wait(timeout=5.0)
            assert errors == []

            # The runner recovers with a fresh loop afterwards.
            async def _noop() -> None:
                return None

            runner.run(_noop())
        finally:
            runner.close()
