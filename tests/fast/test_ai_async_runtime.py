# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""AI batch wrappers are driven by an executor-bound async runtime.

Covers vane#139: ``_EmbedTextBatch`` and ``_PromptBatch`` cache async SDK
clients across batches, so the UDF executor binds one long-lived event loop
per actor (see ``UDFExecutor._bind_async_runtime``) and hands wrappers a
``run_async`` capability. Wrappers own no loops or threads: the provider
runtime is instantiated inside the bound loop, every batch (including
retries and chunked embeds) runs on that same loop, ``close()`` releases
the provider client on it, and unbound wrappers fail fast instead of
falling back to ``asyncio.run()``.
"""

from __future__ import annotations

import asyncio
import importlib
import pickle
import sys
import types

import pyarrow as pa
import pytest


def _load_functions() -> types.ModuleType:
    """Import the real ``vane.ai.functions``, even under the no-duckdb harness.

    On CI the real module imports normally and nothing is stubbed. Under the
    no-duckdb harness the import fails on dependencies that reach into
    ``duckdb.execution`` at module level; those get stubbed and the import is
    retried — the wrapper tests never touch the stubbed surfaces.
    """
    module = sys.modules.get("vane.ai.functions")
    if getattr(module, "__file__", None):
        return module  # real module already loaded
    if module is not None:
        sys.modules.pop("vane.ai.functions")
    try:
        return importlib.import_module("vane.ai.functions")
    except ImportError:
        stub_specs: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("vane._expressions", ("as_expression", "is_expression")),
            ("vane._expression_udf", ("_build_actor_map_batches_expression",)),
            # The real vllm provider module imports duckdb.execution at top
            # level; the wrapper tests never touch the native vLLM plan.
            ("vane.ai.providers.vllm", ("NativeVLLMPromptPlan", "_build_native_vllm_options_argument")),
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
_PromptBatch = functions._PromptBatch
RetryAfterError = functions.RetryAfterError


def test_real_functions_module_is_under_test() -> None:
    """Guard: the harness must import the real module, not the plugin stub."""
    assert getattr(functions, "__file__", None)
    assert isinstance(_PromptBatch, type)


class _Runtime:
    """Stand-in for the executor-owned AsyncRuntime: one loop, run_until_complete."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()

    def run(self, coro):
        return self.loop.run_until_complete(coro)

    def close(self) -> None:
        self.loop.close()


@pytest.fixture
def runtime():
    rt = _Runtime()
    yield rt
    rt.close()


# ---------------------------------------------------------------------------
# Picklable fakes (module level so pickle can resolve them by reference)
# ---------------------------------------------------------------------------


class LoopRecordingEmbedder:
    """Async embedder recording the running loop at construction and per call."""

    def __init__(self) -> None:
        self.created_on = asyncio.get_running_loop()
        self.seen_loops: list[asyncio.AbstractEventLoop] = []
        self.closed_on: asyncio.AbstractEventLoop | None = None
        self.close_calls = 0

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        self.seen_loops.append(asyncio.get_running_loop())
        return [[float(len(t))] * 2 for t in texts]

    async def aclose(self) -> None:
        self.close_calls += 1
        self.closed_on = asyncio.get_running_loop()


class LoopRecordingEmbedDescriptor:
    def __init__(self) -> None:
        self.instantiations = 0
        self.embedder: LoopRecordingEmbedder | None = None

    def instantiate(self) -> LoopRecordingEmbedder:
        self.instantiations += 1
        self.embedder = LoopRecordingEmbedder()
        return self.embedder


class PicklableEmbedder:
    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] * 2 for t in texts]


class PicklableEmbedDescriptor:
    def instantiate(self) -> PicklableEmbedder:
        return PicklableEmbedder()


class SyncEmbedder:
    """Synchronous embedder: no loop-bound state, no ``aclose``."""

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] * 2 for t in texts]


class CountingSyncEmbedDescriptor:
    def __init__(self) -> None:
        self.instantiations = 0

    def instantiate(self) -> SyncEmbedder:
        self.instantiations += 1
        return SyncEmbedder()


class AsyncNoAcloseEmbedder:
    """Async (loop-bound) embedder that does not expose ``aclose`` — the
    protocol permits this, and it must still be dropped at close (issue #139)."""

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] * 2 for t in texts]


class CountingAsyncEmbedDescriptor:
    def __init__(self) -> None:
        self.instantiations = 0

    def instantiate(self) -> AsyncNoAcloseEmbedder:
        self.instantiations += 1
        return AsyncNoAcloseEmbedder()


class AwaitableReturningEmbedder:
    """A plain ``def embed_text`` that RETURNS an awaitable, no ``aclose`` —
    static inspection cannot classify it, only observing the awaitable can."""

    def embed_text(self, texts: list[str]):
        async def _run() -> list[list[float]]:
            return [[float(len(t))] * 2 for t in texts]

        return _run()


class CountingAwaitableEmbedDescriptor:
    def __init__(self) -> None:
        self.instantiations = 0

    def instantiate(self) -> AwaitableReturningEmbedder:
        self.instantiations += 1
        return AwaitableReturningEmbedder()


class AsyncNoAclosePrompter:
    """Async (loop-bound) prompter that does not expose ``aclose``."""

    async def prompt(self, messages: tuple) -> str:
        return f"echo:{messages[0]}"


class CountingAsyncPromptDescriptor:
    def __init__(self) -> None:
        self.instantiations = 0

    def instantiate(self) -> AsyncNoAclosePrompter:
        self.instantiations += 1
        return AsyncNoAclosePrompter()


class AwaitableReturningPrompter:
    """A plain ``def prompt`` that RETURNS an awaitable, no ``aclose`` —
    static inspection cannot classify it, only observing the awaitable can."""

    def prompt(self, messages: tuple):
        async def _run() -> str:
            return f"echo:{messages[0]}"

        return _run()


class CountingAwaitablePromptDescriptor:
    def __init__(self) -> None:
        self.instantiations = 0

    def instantiate(self) -> AwaitableReturningPrompter:
        self.instantiations += 1
        return AwaitableReturningPrompter()


class LoopRecordingPrompter:
    """Async prompter recording the running loop for each call."""

    def __init__(self) -> None:
        self.seen_loops: list[asyncio.AbstractEventLoop] = []

    async def prompt(self, messages: tuple) -> str:
        self.seen_loops.append(asyncio.get_running_loop())
        return f"echo:{messages[0]}"


class LoopRecordingPromptDescriptor:
    def __init__(self) -> None:
        self.prompter: LoopRecordingPrompter | None = None

    def instantiate(self) -> LoopRecordingPrompter:
        self.prompter = LoopRecordingPrompter()
        return self.prompter


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


class ConcurrencyTrackingPromptDescriptor:
    def __init__(self) -> None:
        self.prompter: ConcurrencyTrackingPrompter | None = None

    def instantiate(self) -> ConcurrencyTrackingPrompter:
        self.prompter = ConcurrencyTrackingPrompter()
        return self.prompter


class FlakyOncePrompter:
    """Fails the first call with a zero-wait retryable error, then succeeds."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen_loops: list[asyncio.AbstractEventLoop] = []

    async def prompt(self, messages: tuple) -> str:
        self.calls += 1
        self.seen_loops.append(asyncio.get_running_loop())
        if self.calls == 1:
            raise RetryAfterError(0.0, RuntimeError("transient"))
        return f"echo:{messages[0]}"


class FlakyOncePromptDescriptor:
    def __init__(self) -> None:
        self.prompter: FlakyOncePrompter | None = None

    def instantiate(self) -> FlakyOncePrompter:
        self.prompter = FlakyOncePrompter()
        return self.prompter


# ---------------------------------------------------------------------------
# Loop ownership: client construction, batches, retries, chunking
# ---------------------------------------------------------------------------


def test_embed_client_created_and_used_on_the_bound_loop(runtime) -> None:
    descriptor = LoopRecordingEmbedDescriptor()
    wrapper = _EmbedTextBatch(descriptor, "text", "emb", 2)
    wrapper.bind_async_runtime(runtime.run)

    wrapper(pa.table({"text": ["a"]}))
    wrapper(pa.table({"text": ["bb", "ccc"]}))

    assert descriptor.instantiations == 1
    embedder = descriptor.embedder
    assert embedder.created_on is runtime.loop
    assert embedder.seen_loops == [runtime.loop, runtime.loop]


def test_prompt_batches_share_the_bound_loop(runtime) -> None:
    descriptor = LoopRecordingPromptDescriptor()
    wrapper = _PromptBatch(descriptor, ["text"], "response")
    wrapper.bind_async_runtime(runtime.run)

    first = wrapper(pa.table({"text": ["a", "b"]}))
    second = wrapper(pa.table({"text": ["c"]}))

    assert first.column("response").to_pylist() == ["echo:a", "echo:b"]
    assert second.column("response").to_pylist() == ["echo:c"]
    assert set(descriptor.prompter.seen_loops) == {runtime.loop}


def test_prompt_retry_reruns_on_the_same_loop(runtime) -> None:
    descriptor = FlakyOncePromptDescriptor()
    wrapper = _PromptBatch(descriptor, ["text"], "response", max_retries=2)
    wrapper.bind_async_runtime(runtime.run)

    result = wrapper(pa.table({"text": ["q"]}))

    assert result.column("response").to_pylist() == ["echo:q"]
    assert descriptor.prompter.calls == 2
    assert set(descriptor.prompter.seen_loops) == {runtime.loop}


def test_prompt_semaphore_and_ordering_on_bound_loop(runtime) -> None:
    descriptor = ConcurrencyTrackingPromptDescriptor()
    wrapper = _PromptBatch(descriptor, ["text"], "response", max_concurrency_per_actor=2)
    wrapper.bind_async_runtime(runtime.run)

    texts = [f"m{i}" for i in range(8)]
    result = wrapper(pa.table({"text": texts}))

    assert result.column("response").to_pylist() == [f"echo:{t}" for t in texts]
    assert descriptor.prompter.peak <= 2


def test_embed_chunking_drives_chunks_on_the_bound_loop(runtime) -> None:
    descriptor = LoopRecordingEmbedDescriptor()
    wrapper = _EmbedTextBatch(descriptor, "text", "emb", 2, max_chunk_chars=50, chunk_overlap_chars=10)
    wrapper.bind_async_runtime(runtime.run)

    result = wrapper(pa.table({"text": ["short", "a" * 200]}))

    assert result.num_rows == 2
    assert set(descriptor.embedder.seen_loops) == {runtime.loop}


# ---------------------------------------------------------------------------
# No fallback: unbound wrappers fail fast
# ---------------------------------------------------------------------------


def test_unbound_embed_wrapper_raises() -> None:
    wrapper = _EmbedTextBatch(LoopRecordingEmbedDescriptor(), "text", "emb", 2)
    with pytest.raises(RuntimeError, match="bind_async_runtime"):
        wrapper(pa.table({"text": ["a"]}))


def test_unbound_prompt_wrapper_raises() -> None:
    wrapper = _PromptBatch(LoopRecordingPromptDescriptor(), ["text"], "response")
    with pytest.raises(RuntimeError, match="bind_async_runtime"):
        wrapper(pa.table({"text": ["a"]}))


def test_unbound_wrapper_raises_even_with_on_error_ignore() -> None:
    """A missing runtime is a programming error, not a provider failure."""
    wrapper = _EmbedTextBatch(LoopRecordingEmbedDescriptor(), "text", "emb", 2, max_retries=3, on_error="ignore")
    with pytest.raises(RuntimeError, match="bind_async_runtime"):
        wrapper(pa.table({"text": ["a"]}))


def test_retry_call_requires_runtime_for_awaitables_and_never_retries_it() -> None:
    attempts = 0

    async def api() -> str:
        return "unreachable"

    def call() -> object:
        nonlocal attempts
        attempts += 1
        return api()

    with pytest.raises(RuntimeError, match="bind_async_runtime"):
        functions._retry_call(call, max_retries=5, on_error="ignore", default="fallback")
    assert attempts == 1


# ---------------------------------------------------------------------------
# close(): provider client released on the owning loop
# ---------------------------------------------------------------------------


def test_close_releases_client_on_the_bound_loop(runtime) -> None:
    descriptor = LoopRecordingEmbedDescriptor()
    wrapper = _EmbedTextBatch(descriptor, "text", "emb", 2)
    wrapper.bind_async_runtime(runtime.run)
    wrapper(pa.table({"text": ["a"]}))

    wrapper.close()
    wrapper.close()  # idempotent

    embedder = descriptor.embedder
    assert embedder.close_calls == 1
    assert embedder.closed_on is runtime.loop


def test_close_before_first_batch_is_a_noop(runtime) -> None:
    descriptor = LoopRecordingEmbedDescriptor()
    wrapper = _EmbedTextBatch(descriptor, "text", "emb", 2)
    wrapper.bind_async_runtime(runtime.run)

    wrapper.close()

    assert descriptor.instantiations == 0


def test_close_retains_sync_provider(runtime) -> None:
    """A proven-synchronous embedder (sync ``embed_text``, no ``aclose``) holds
    no loop-bound state and survives close: task backends close the executor
    per invocation while the callable cache keeps the wrapper, so dropping it
    would reload the model per task."""
    descriptor = CountingSyncEmbedDescriptor()
    wrapper = _EmbedTextBatch(descriptor, "text", "emb", 2)
    wrapper.bind_async_runtime(runtime.run)
    wrapper(pa.table({"text": ["a"]}))

    wrapper.close()  # sync provider stays cached, must not raise

    fresh = _Runtime()
    try:
        wrapper.bind_async_runtime(fresh.run)
        wrapper(pa.table({"text": ["ab"]}))
    finally:
        fresh.close()
    assert descriptor.instantiations == 1


def test_close_drops_async_embedder_without_aclose(runtime) -> None:
    """The maintainer's #139 reproduction: an async embedder without ``aclose``
    is loop-bound, so it is dropped at close and re-instantiated on the next
    (fresh-loop) batch rather than reused across loops."""
    descriptor = CountingAsyncEmbedDescriptor()
    wrapper = _EmbedTextBatch(descriptor, "text", "emb", 2)
    wrapper.bind_async_runtime(runtime.run)
    wrapper(pa.table({"text": ["a"]}))

    wrapper.close()

    fresh = _Runtime()
    try:
        wrapper.bind_async_runtime(fresh.run)
        wrapper(pa.table({"text": ["ab"]}))
    finally:
        fresh.close()
    assert descriptor.instantiations == 2


def test_close_drops_async_prompter_without_aclose(runtime) -> None:
    """``Prompter.prompt`` is async by protocol: a prompter without ``aclose``
    is still loop-bound and dropped at close, then re-instantiated."""
    descriptor = CountingAsyncPromptDescriptor()
    wrapper = _PromptBatch(descriptor, ["text"], "response")
    wrapper.bind_async_runtime(runtime.run)
    wrapper(pa.table({"text": ["a"]}))

    wrapper.close()

    fresh = _Runtime()
    try:
        wrapper.bind_async_runtime(fresh.run)
        wrapper(pa.table({"text": ["ab"]}))
    finally:
        fresh.close()
    assert descriptor.instantiations == 2


def test_close_drops_embedder_after_observed_awaitable(runtime) -> None:
    """A plain ``def embed_text`` that returns an awaitable is not a coroutine
    function, so static inspection cannot classify it; observing the awaitable
    at runtime still marks it loop-bound, so it is dropped at close."""
    descriptor = CountingAwaitableEmbedDescriptor()
    wrapper = _EmbedTextBatch(descriptor, "text", "emb", 2)
    wrapper.bind_async_runtime(runtime.run)
    wrapper(pa.table({"text": ["a"]}))  # observes the awaitable → marks loop-bound

    wrapper.close()

    fresh = _Runtime()
    try:
        wrapper.bind_async_runtime(fresh.run)
        wrapper(pa.table({"text": ["ab"]}))
    finally:
        fresh.close()
    assert descriptor.instantiations == 2


def test_close_drops_prompter_after_observed_awaitable(runtime) -> None:
    """The per-row prompt path (no ``prompt_batch``) also reports observed
    awaitables: a plain ``def prompt`` returning an awaitable is marked
    loop-bound and dropped at close."""
    descriptor = CountingAwaitablePromptDescriptor()
    wrapper = _PromptBatch(descriptor, ["text"], "response")
    wrapper.bind_async_runtime(runtime.run)
    result = wrapper(pa.table({"text": ["a"]}))  # observes the awaitable
    assert result.column("response").to_pylist() == ["echo:a"]

    wrapper.close()

    fresh = _Runtime()
    try:
        wrapper.bind_async_runtime(fresh.run)
        wrapper(pa.table({"text": ["ab"]}))
    finally:
        fresh.close()
    assert descriptor.instantiations == 2


# ---------------------------------------------------------------------------
# Pickling: client and capability never cross process boundaries
# ---------------------------------------------------------------------------


def test_pickle_clears_client_and_capability(runtime) -> None:
    wrapper = _EmbedTextBatch(PicklableEmbedDescriptor(), "text", "emb", 2)
    wrapper.bind_async_runtime(runtime.run)
    wrapper(pa.table({"text": ["warm"]}))

    restored = pickle.loads(pickle.dumps(wrapper))

    assert restored._run_async is None
    assert restored._embedder is None
    with pytest.raises(RuntimeError, match="bind_async_runtime"):
        restored(pa.table({"text": ["a"]}))

    fresh = _Runtime()
    try:
        restored.bind_async_runtime(fresh.run)
        result = restored(pa.table({"text": ["ab"]}))
        assert result.column("emb").to_pylist() == [[2.0, 2.0]]
    finally:
        fresh.close()


def test_prompt_pickle_clears_client_and_capability(runtime) -> None:
    descriptor = LoopRecordingPromptDescriptor()
    wrapper = _PromptBatch(descriptor, ["text"], "response", max_concurrency_per_actor=4)
    wrapper.bind_async_runtime(runtime.run)
    wrapper(pa.table({"text": ["warm"]}))

    state = wrapper.__getstate__()

    assert state["_run_async"] is None
    assert state["_prompter"] is None
    assert state["_max_concurrency_per_actor"] == 4


# ---------------------------------------------------------------------------
# Backend adapters forward the capability hooks
# ---------------------------------------------------------------------------


def test_actor_adapter_forwards_bind_and_close(runtime) -> None:
    descriptor = LoopRecordingEmbedDescriptor()
    wrapper = _EmbedTextBatch(descriptor, "text", "emb", 2)
    actor_cls = functions._adapt_batch_wrapper_for_backend(wrapper, "subprocess_actor")

    actor = actor_cls()
    actor.bind_async_runtime(runtime.run)
    actor(pa.table({"text": ["a"]}))
    actor.close()

    embedder = descriptor.embedder
    assert embedder.created_on is runtime.loop
    assert embedder.close_calls == 1
    assert embedder.closed_on is runtime.loop


def test_task_adapter_carries_capability_hooks(runtime) -> None:
    descriptor = LoopRecordingEmbedDescriptor()
    wrapper = _EmbedTextBatch(descriptor, "text", "emb", 2)
    fn = functions._adapt_batch_wrapper_for_backend(wrapper, "subprocess_task")

    fn.bind_async_runtime(runtime.run)
    fn(pa.table({"text": ["a"]}))
    fn.close()

    embedder = descriptor.embedder
    assert embedder.created_on is runtime.loop
    assert embedder.close_calls == 1


def test_task_adapter_leaves_plain_wrappers_without_hooks() -> None:
    class PlainWrapper:
        def __call__(self, table: pa.Table) -> pa.Table:
            return table

    fn = functions._adapt_batch_wrapper_for_backend(PlainWrapper(), "subprocess_task")

    assert not hasattr(fn, "bind_async_runtime")
    assert not hasattr(fn, "close")
