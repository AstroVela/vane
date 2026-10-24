# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared model ownership and the service's bounded execution lifecycle."""

import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from tests.fast.test_mlx_provider import Tokenizer
from vane.ai import _embedding_ipc as ipc
from vane.ai._audio_embedding import AudioClip
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec
from vane.ai._video_embedding import VideoClip
from vane.ai.provider import load_provider
from vane.ai.providers import _mlx_service as module
from vane.ai.providers.mlx import MODEL_ID

pytest_plugins = ("mlx_sdk_fixture",)


def content(*parts):
    return EmbeddingContent(tuple(EmbeddingPart(EmbeddingPartSpec(kind), value) for kind, value in parts))


def test_one_full_media_core_reuses_weights_processor_and_original_text_tokenizer(media_sdk, monkeypatch):
    from vane.ai.providers.mlx import _MLXModelCore

    tokenizer = Tokenizer()
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    image = np.zeros((3, 5, 3), np.uint8)
    audio = AudioClip(np.zeros((161, 1), np.float32), 16000)
    video = VideoClip((image,), (0.125,), (7,))
    cases = [
        ("text_document", content(("text", "same text"))),
        ("text_query", content(("text", "same text"))),
        ("image", content(("image", image))),
        ("audio", content(("audio", audio))),
        ("video", content(("video", video))),
        ("multimodal", content(("text", "scene"), ("image", image), ("audio", audio), ("video", video))),
    ]
    core = _MLXModelCore({})
    model, processor, options = core._model, core._processor, dict(core._options)
    try:
        for encoding, row in cases:
            result = core.encode(row, encoding)
            assert result.dtype == np.float32 and result.shape == (768,)
            assert core._model is model and core._processor is processor
            assert core._options == options
        assert tokenizer.texts == ["title: none | text: same text", "task: search result | query: same text"]
        assert media_sdk.loads == media_sdk.processor_loads == 1
        assert len(media_sdk.model_calls) == 6
    finally:
        core.close()
        core.close()
    assert core._model is core._processor is core._tokenizer is None
    assert media_sdk.clears == 1


@pytest.fixture
def fake_core(monkeypatch):
    state = SimpleNamespace(
        events=[],
        contexts=[],
        submissions=[],
        cores=[],
        active=0,
        peak=0,
        entered=threading.Event(),
        gate=None,
        load_entered=threading.Event(),
        load_gate=None,
        failures={},
    )

    def event(name, **values):
        state.events.append(dict(event=name, thread=threading.get_ident(), **values))

    class Core:
        def __init__(self, options):
            event("load")
            state.load_entered.set()
            if state.load_gate is not None:
                assert state.load_gate.wait(10)
            if "load" in state.failures:
                raise state.failures["load"]
            self.options = dict(options)
            self.closed = False
            state.cores.append(self)

        def encode(self, row, encoding):
            event("processor", model=id(self), encoding=encoding)
            if "processor" in state.failures:
                raise state.failures["processor"]
            state.active += 1
            state.peak = max(state.peak, state.active)
            event("forward_start", model=id(self), encoding=encoding)
            state.entered.set()
            try:
                if state.gate is not None:
                    assert state.gate.wait(10)
                if "forward" in state.failures:
                    raise state.failures["forward"]
                # Read only after the barrier, so cancellation cannot stand in
                # for actual consumption of the retained input.
                value = row.parts[0].value
                index = int(value) if isinstance(value, str) and value.isdigit() else int(encoding == "text_query")
                return np.eye(1, 768, index, dtype=np.float32)[0]
            finally:
                event("forward_end", model=id(self))
                state.active -= 1

        def close(self):
            assert not self.closed and state.active == 0
            self.closed = True
            event("close", model=id(self))
            if "close" in state.failures:
                raise state.failures["close"]

    class Executor(ThreadPoolExecutor):
        def submit(self, fn, *args, **kwargs):
            state.submissions.append(fn)
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr(module, "_create_core", Core)
    monkeypatch.setattr(module, "ThreadPoolExecutor", Executor)
    yield state
    for gate in (state.gate, state.load_gate):
        if gate is not None:
            gate.set()


async def entered(event):
    assert await asyncio.wait_for(asyncio.to_thread(event.wait, 8), 9)


def backend():
    return module.MLXServiceBackend(MODEL_ID, 768, {})


def context(owner, encoding="text_document"):
    return ipc.RequestContext(encoding, owner.contract)


def test_single_owner_thread_serializes_load_processing_forward_and_close(fake_core):
    async def run():
        owner = backend()
        assert owner.state == "CREATED" and owner._executor is None and not fake_core.events
        await owner.start()
        try:
            with pytest.raises(RuntimeError, match="once"):
                await owner.start()
            results = await asyncio.gather(
                *[owner.embed(content(("text", str(i))), context(owner)) for i in (2, 0, 1, 2)]
            )
            assert [int(np.argmax(row)) for row in results] == [2, 0, 1, 2]
        finally:
            await owner.aclose()
            await owner.aclose()
        assert owner.state == "CLOSED" and owner._executor is owner._core is None

    asyncio.run(run())
    assert fake_core.peak == 1 and len(fake_core.cores) == 1
    assert [e["event"] for e in fake_core.events].count("load") == 1
    assert [e["event"] for e in fake_core.events].count("close") == 1
    threads = {e["thread"] for e in fake_core.events}
    assert len(threads) == 1 and threading.get_ident() not in threads


def test_repeated_cancel_holds_owner_lock_and_never_submits_waiting_work(fake_core):
    fake_core.gate = threading.Event()

    async def run():
        owner = backend()
        await owner.start()
        first = asyncio.create_task(owner.embed(content(("text", "first")), context(owner)))
        await entered(fake_core.entered)
        second = asyncio.create_task(owner.embed(content(("text", "second")), context(owner)))
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        for _ in range(3):
            await asyncio.wait_for(asyncio.sleep(0), 1)  # main loop remains responsive
        assert not first.done() and not second.done() and fake_core.active == 1
        assert len(fake_core.submissions) == 2  # load + first forward; no executor backlog
        assert not any(e["event"] == "close" for e in fake_core.events)
        fake_core.gate.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        await owner.aclose()

    asyncio.run(run())
    assert fake_core.peak == 1
    assert [e["event"] for e in fake_core.events].count("forward_start") == 2


def test_cancelled_start_retains_loaded_core_for_owned_cleanup(fake_core):
    fake_core.load_gate = threading.Event()

    async def run():
        owner = backend()
        starting = asyncio.create_task(owner.start())
        await entered(fake_core.load_entered)
        starting.cancel()
        await asyncio.sleep(0)
        starting.cancel()
        assert not starting.done()
        fake_core.load_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await starting
        assert owner._core is fake_core.cores[0]
        await owner.aclose()
        assert owner._core is owner._executor is None

    asyncio.run(run())
    assert [e["event"] for e in fake_core.events] == ["load", "close"]


@pytest.mark.parametrize("phase", ["load", "processor", "forward", "close"])
@pytest.mark.parametrize("resource", [False, True])
def test_failure_phases_preserve_classification_and_release_once(fake_core, phase, resource):
    failure = MemoryError("fixture allocation") if resource else ValueError("fixture execution")
    fake_core.failures[phase] = failure

    async def run():
        owner = backend()
        if phase == "load":
            with pytest.raises(type(failure)):
                await owner.start()
        else:
            await owner.start()
            if phase != "close":
                with pytest.raises(type(failure)):
                    await owner.embed(content(("text", "input")), context(owner))
                assert owner.restart_required is resource
                if resource:
                    before = len(fake_core.events)
                    with pytest.raises(MemoryError, match="restart"):
                        await owner.embed(content(("text", "later")), context(owner))
                    assert len(fake_core.events) == before
        if phase == "close":
            for _ in range(2):
                with pytest.raises(type(failure)):
                    await owner.aclose()
        else:
            await owner.aclose()
        assert owner._executor is owner._core is None

    asyncio.run(run())
    assert [e["event"] for e in fake_core.events].count("close") == (0 if phase == "load" else 1)


def service_case(root):
    from vane.ai.embedding_service import EmbeddingService

    return EmbeddingService(
        path=str(root / "mlx.json"),
        service_id="mlx-fixture",
        family="mlx",
        model=MODEL_ID,
        dimensions=768,
        max_inflight=2,
        max_connections=4,
        max_bytes=1024,
    )


def service_client(root, *, query=False, **options):
    return (
        load_provider("mlx")
        .get_text_embedder(
            options={
                "runtime": "service",
                "service_path": str(root / "mlx.json"),
                "service_id": "mlx-fixture",
                "input_type": "query" if query else "document",
                "timeout": 5,
            }
            | options
        )
        .instantiate()
    )


def test_real_ipc_cancel_and_cancelled_shutdown_wait_for_actual_owner_work(fake_core, tmp_path, monkeypatch):
    pytest.importorskip("zmq")
    from vane.ai import _embedding_shared_memory as shm

    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    fake_core.gate = threading.Event()
    release = shm.release
    reclaimed = []

    def observe_release(memory):
        assert any(e["event"] == "forward_end" for e in fake_core.events)
        reclaimed.append(memory.name)
        release(memory)

    monkeypatch.setattr(shm, "release", observe_release)

    async def run():
        service = service_case(root)
        await service.start()
        client = service_client(root)
        pending = asyncio.create_task(client.embed_text(["retained input"]))
        closing = None
        try:
            await entered(fake_core.entered)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert service.credits.inflight == 1 and service.credits.bytes > 0
            assert service._zmq_server.buffers and reclaimed == []
            closing = asyncio.create_task(service.aclose())
            await asyncio.sleep(0)
            closing.cancel()
            await asyncio.sleep(0)
            closing.cancel()
            assert not closing.done() and not fake_core.cores[0].closed
            fake_core.gate.set()
            await closing
            assert service.credits.bytes == service.credits.inflight == 0
            assert not (root / "mlx.json").exists()
            await service.aclose()
        finally:
            fake_core.gate.set()
            await client.aclose()
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
            await service.aclose()

    asyncio.run(run())
    assert len(reclaimed) == 1
    assert [e["event"] for e in fake_core.events][-2:] == ["forward_end", "close"]


def test_real_ipc_multiple_clients_keep_roles_order_and_server_ownership(fake_core, tmp_path):
    pytest.importorskip("zmq")
    root = tmp_path / "private"
    root.mkdir(mode=0o700)

    async def run():
        service = service_case(root)
        original = service.backend._encode

        def observe(row, request):
            fake_core.contexts.append(request)
            return original(row, request)

        service.backend._encode = observe
        await service.start()
        document = service_client(root, max_concurrency_per_actor=2)
        query = service_client(root, query=True, max_concurrency_per_actor=2)
        try:
            documents, queries = await asyncio.gather(
                document.embed_text(["2", "0", "1", "2"]), query.embed_text(["query"])
            )
            assert np.argmax(documents, axis=1).tolist() == [2, 0, 1, 2]
            assert np.argmax(queries, axis=1).tolist() == [1]
            await document.aclose()
            assert not fake_core.cores[0].closed
            assert int(np.argmax((await query.embed_text(["again"]))[0])) == 1
        finally:
            await document.aclose()
            await query.aclose()
            await service.aclose()

    asyncio.run(run())
    assert len(fake_core.cores) == fake_core.peak == 1
    assert {c.encoding for c in fake_core.contexts} == {"text_document", "text_query"}
    assert len({c.request_id for c in fake_core.contexts}) == 6
    assert all(len(c.request_id) == 32 for c in fake_core.contexts)


def test_metal_oom_is_fatal_without_retry_and_failed_requests_allocate_no_new_buffer(fake_core, tmp_path, monkeypatch):
    pytest.importorskip("zmq")
    from vane.ai import _embedding_shared_memory as shm

    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    fake_core.failures["forward"] = RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory")
    allocations = []
    create = shm.create

    def count(size):
        allocations.append(size)
        return create(size)

    monkeypatch.setattr(shm, "create", count)

    async def run():
        service = service_case(root)
        await service.start()
        client = service_client(root)
        client.configure_execution(on_error="ignore", max_retries=9, validate=lambda value: value)
        try:
            for _ in range(2):
                with pytest.raises(MemoryError, match="restart"):
                    await client.embed_text(["input"])
            assert service.backend.state == "FAILED"
            assert service.credits.bytes == service.credits.inflight == 0
        finally:
            await client.aclose()
            await service.aclose()
        fake_core.failures.clear()
        restarted = service_case(root)
        await restarted.start()
        client = service_client(root)
        try:
            assert len(await client.embed_text(["fresh service"])) == 1
        finally:
            await client.aclose()
            await restarted.aclose()

    asyncio.run(run())
    assert len(fake_core.cores) == 2 and len(allocations) == 2
    assert [e["event"] for e in fake_core.events].count("forward_start") == 2


def test_loading_failure_cleans_endpoint_and_owner_thread(fake_core, tmp_path):
    pytest.importorskip("zmq")
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    fake_core.failures["load"] = MemoryError("fixture allocation")

    async def run():
        service = service_case(root)
        with pytest.raises(MemoryError):
            await service.start()
        assert service._zmq_server is None and service.backend._executor is None
        assert not (root / "mlx.json").exists()
        await service.aclose()

    asyncio.run(run())


def test_public_admission_bounds_waiters_without_an_executor_backlog(fake_core, tmp_path):
    pytest.importorskip("zmq")
    from vane.ai._embedding_inputs import EmbeddingConfigurationError

    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    fake_core.gate = threading.Event()

    async def run():
        service = service_case(root)
        admitted = asyncio.Event()
        calls = []
        infer = service._infer

        async def observe(row, request):
            calls.append(request)
            if len(calls) == 2:
                admitted.set()
            return await infer(row, request)

        service._infer = observe
        await service.start()
        client = service_client(root)
        wrong = service_client(root, max_length=32)
        pending = asyncio.create_task(client.embed_text(["one", "two", "three"]))
        try:
            await asyncio.wait_for(admitted.wait(), 5)
            # Admission and executor submission can precede the owner thread's
            # first instruction. Observe actual forward entry before asserting
            # its active count, independently of the host scheduler.
            await entered(fake_core.entered)
            assert service.credits.inflight == 2 and len(service._zmq_server.buffers) == 2
            assert len(fake_core.submissions) == 2 and fake_core.active == 1
            # Control traffic and contract refusal still progress while the
            # sole owner is blocked, without reserving a third input snapshot.
            with pytest.raises(EmbeddingConfigurationError):
                await wrong.embed_text(["mismatched client"])
            assert service.credits.inflight == 2 and len(calls) == 2
            fake_core.gate.set()
            assert len(await pending) == 3
            assert service.credits.bytes == service.credits.inflight == 0
        finally:
            fake_core.gate.set()
            await asyncio.gather(pending, return_exceptions=True)
            await client.aclose()
            await wrong.aclose()
            await service.aclose()

    asyncio.run(run())
    assert fake_core.peak == 1
    assert [e["event"] for e in fake_core.events].count("forward_start") == 3
