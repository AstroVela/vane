# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Persistent native service: PyZMQ, SharedMemory, process death and processors."""

from __future__ import annotations

import asyncio
import json
import mmap
import os
import pickle
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.fast.test_native_multimodal import install_sdk
from tests.fast.test_native_multimodal import make_real_video_processor as processor_fixture
from vane.ai import _embedding_ipc as ipc
from vane.ai import _embedding_shared_memory as shm
from vane.ai import _embedding_zmq as transport
from vane.ai._audio_embedding import AudioClip
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._video_embedding import VideoClip
from vane.ai.embedding_service import EmbeddingService
from vane.ai.provider import load_provider

make_real_video_processor = processor_fixture

# PyZMQ announces its supported Tornado selector bridge on Windows Proactor loops.
pytestmark = pytest.mark.filterwarnings(
    r"ignore:Proactor event loop does not implement add_reader family of methods required for zmq\.:RuntimeWarning"
)

pytest.importorskip("zmq")
import zmq.asyncio


def content(kind="text", value="query"):
    return EmbeddingContent((EmbeddingPart(EmbeddingPartSpec(kind), value),))


def video():
    frames = tuple(np.full((4, 4, 3), value, dtype=np.uint8) for value in (10, 20))
    return VideoClip(frames, (0.125, 1.375), (2, 5))


def client(root, family="sglang", kind="multimodal", **options):
    descriptor = getattr(load_provider(family), f"get_{kind}_embedder")(
        "fixture",
        2,
        options={"runtime": "service", "service_path": str(root / "engine.json"), "service_id": "fixture-v1"} | options,
    )
    return descriptor.instantiate()


async def until(predicate, timeout=8):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


@pytest.fixture
def process(tmp_path):
    processes = []

    def start(family="sglang", **config):
        root = tmp_path / str(len(processes))
        root.mkdir(mode=0o700)
        (root / "config.json").write_text(json.dumps(config))
        log = (root / "log.txt").open("w")
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("embedding_service_process.py")), str(root), family],
            stdout=log,
            stderr=log,
        )
        processes.append((proc, root, log))
        return proc, root

    yield start
    for proc, root, log in processes:
        (root / "hold").unlink(missing_ok=True)
        if proc.poll() is None:
            (root / "stop").touch()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
                pytest.fail("Embedding fixture failed to drain")
        log.close()
        assert proc.returncode == 0 or (root / "killed").exists(), (root / "log.txt").read_text()


def state(root):
    path = root / "state.json"
    return json.loads(path.read_text()) if path.exists() else {}


@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("kind", ["text", "image", "video", "audio", "multimodal"])
def test_service_planning_needs_no_sdk_and_claims_no_gpu(tmp_path, monkeypatch, family, kind):
    monkeypatch.setitem(sys.modules, "sglang", None)
    monkeypatch.setitem(sys.modules, "vllm", None)
    descriptor = client(
        tmp_path,
        family,
        kind,
        input_modalities=["text", "image", "video", "audio"],
        sample_rate=16000,
        max_samples=16000,
    ).descriptor
    clone = pickle.loads(pickle.dumps(descriptor))
    assert clone.is_async() and clone.get_udf_options().num_gpus == 0
    assert clone.instantiate()._zmq_client is None


@pytest.mark.parametrize(
    "options",
    [
        {"engine_args": {}},
        {"base_url": "http://localhost"},
        {"service_id": ""},
        {"service_path": "relative"},
        {"timeout": float("nan")},
        {"input_modalities": ["audio"]},
        {"max_input_bytes": 2**40},
        {"service_transport": "unix"},
    ],
)
def test_invalid_service_configuration_fails_without_connecting(tmp_path, options):
    with pytest.raises((EmbeddingConfigurationError, TypeError, ValueError)):
        client(tmp_path, **options)


@pytest.mark.parametrize("family", ["sglang", "vllm"])
def test_ingest_and_search_clients_share_one_engine(process, family):
    proc, root = process(family)

    async def run():
        await until(lambda: (root / "ready").exists() or proc.poll() is not None)
        assert proc.poll() is None
        ingest, search = client(root, family), client(root, family, "text")
        try:
            values = await asyncio.gather(
                ingest.embed_multimodal([EmbeddingContent(content("video", video()).parts + content().parts)] * 3),
                search.embed_text(["find clip", "find image"]),
            )
            for batch in values:
                np.testing.assert_equal(batch, [[3, 4]] * len(batch))
            await until(lambda: state(root).get("calls") == 5 and state(root).get("bytes") == 0)
            assert state(root)["engines"] == 1
        finally:
            await ingest.aclose()
            await search.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("family", ["sglang", "vllm"])
def test_serial_requests_reclaim_connection_before_client_eof(process, monkeypatch, family):
    proc, root = process(family, max_connections=1, max_inflight=1)

    async def run():
        await until(lambda: (root / "ready").exists() or proc.poll() is not None)
        assert proc.poll() is None
        instance = client(root, family, timeout=3, max_concurrency_per_actor=1)
        close = zmq.asyncio.Socket.close
        held = []

        def delay_client_eof(socket, *args, **kwargs):
            # Keep completed client sockets open to force the successor to
            # arrive before EOF, independently of libzmq's scheduling.
            held.append((socket, args, kwargs))

        try:
            with monkeypatch.context() as patch:
                patch.setattr(zmq.asyncio.Socket, "close", delay_client_eof)
                for index in range(20):
                    if index == 1:
                        (root / "memory_error").touch()
                        try:
                            with pytest.raises(MemoryError, match="resources"):
                                await instance.embed_text(["allocation failure"])
                        finally:
                            (root / "memory_error").unlink()
                    else:
                        np.testing.assert_equal(await instance.embed_text([f"serial request {index}"]), [[3, 4]])
                    # Keep at most one completed client open across calls.
                    while len(held) > 1:
                        socket, args, kwargs = held.pop(0)
                        close(socket, *args, **kwargs)
            await until(lambda: state(root).get("calls") == 20 and state(root).get("inflight") == 0)
            assert state(root)["engines"] == 1
        finally:
            for socket, args, kwargs in held:
                close(socket, *args, **kwargs)
            await instance.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("family", ["sglang", "vllm"])
def test_cancelled_mapped_input_holds_credit_until_model_finishes(process, family):
    proc, root = process(family, max_bytes=200, max_inflight=2)
    (root / "hold").touch()

    async def run():
        await until(lambda: (root / "ready").exists() or proc.poll() is not None)
        instance = client(root, family)
        first = asyncio.create_task(instance.embed_multimodal([content("video", video())]))
        await until(lambda: state(root).get("active") == 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(instance.embed_multimodal([content("video", video())]))
        await until(lambda: state(root).get("connections") == 2)
        assert state(root)["bytes"] == 192 and state(root)["calls"] == 1
        (root / "hold").unlink()
        np.testing.assert_equal(await second, [[3, 4]])
        await until(lambda: state(root).get("bytes") == state(root).get("inflight") == 0)
        assert state(root)["calls"] == 2
        await instance.aclose()

    asyncio.run(run())


def test_producer_process_death_does_not_release_mapped_credit(process):
    proc, root = process(max_inflight=1)
    (root / "hold").touch()

    async def run():
        await until(lambda: (root / "ready").exists() or proc.poll() is not None)
        code = "import asyncio,sys; import numpy as np; from vane.ai._video_embedding import VideoClip; from vane.ai.provider import load_provider; d=load_provider('sglang').get_video_embedder('fixture',2,options={'runtime':'service','service_path':sys.argv[1],'service_id':'fixture-v1'}); asyncio.run(d.instantiate().embed_video([VideoClip((np.zeros((4,4,3),np.uint8),),(0.0,),(0,))]))"
        producer = subprocess.Popen([sys.executable, "-c", code, str(root / "engine.json")])
        try:
            await until(lambda: state(root).get("active") == 1)
            producer.kill()
            producer.wait(timeout=5)
            assert state(root)["inflight"] == 1
            instance = client(root)
            query = asyncio.create_task(instance.embed_text(["query"]))
            await until(lambda: state(root).get("connections") == 2)
            assert not query.done() and state(root)["calls"] == 1
            (root / "hold").unlink()
            np.testing.assert_equal(await query, [[3, 4]])
            await until(lambda: state(root).get("inflight") == 0)
            await instance.aclose()
        finally:
            if producer.poll() is None:
                producer.kill()
                producer.wait(timeout=5)

    asyncio.run(run())


def test_server_death_fails_client_and_reclaims_owned_buffers(process):
    proc, root = process()
    (root / "hold").touch()

    def removed(names):
        for name in names:
            try:
                with shm.attach(name, 96):
                    return False
            except FileNotFoundError:
                pass
        return True

    async def run():
        await until(lambda: (root / "ready").exists() or proc.poll() is not None)
        instance = client(root, timeout=2)
        task = asyncio.create_task(instance.embed_multimodal([content("video", video())]))
        try:
            await until(lambda: state(root).get("active") == 1)
            names = state(root)["buffers"]
            assert len(names) == 1
            (root / "killed").touch()
            proc.kill()
            proc.wait(timeout=5)
            with pytest.raises((asyncio.TimeoutError, ConnectionError)):
                await task
            await until(lambda: removed(names))
            assert not instance._zmq_client.sockets
        finally:
            await instance.aclose()

    asyncio.run(run())


def test_disconnect_before_publication_releases_reservation(process):
    proc, root = process(publication_timeout=0.2)

    async def run():
        await until(lambda: (root / "ready").exists() or proc.poll() is not None)
        for disconnect in (True, False):
            peer = RawClient(root)
            try:
                response = await peer.reserve(content("video", video()))
                name = response["shm_name"]
                assert response["state"] == "reserved"
                if disconnect:
                    peer.close()
                else:
                    assert "error" in await peer.receive()

                def removed():
                    try:
                        with shm.attach(name, 96):
                            return False
                    except FileNotFoundError:
                        return True

                await until(removed)
                await until(lambda: state(root).get("inflight") == 0)
            finally:
                peer.close()
        assert state(root)["calls"] == 0

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["text", "video"])
@pytest.mark.parametrize("disconnect", ["eof", "protocol_rejection"])
def test_unpublished_disconnect_unblocks_successor(tmp_path, monkeypatch, kind, disconnect):
    sdk = install_sdk(monkeypatch.setitem)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    # Retain the default 30-second publication deadline: a short deadline would
    # conceal a handler that never learns about the disconnect.
    service = make_service(root, max_inflight=1, max_connections=1)

    async def run():
        await service.start()
        peer, successor = RawClient(root), client(root, timeout=2)
        try:
            reserved = await peer.reserve(content("text", "") if kind == "text" else content("video", video()))
            assert reserved["state"] == "reserved" and service.credits.inflight == 1
            if disconnect == "eof":
                peer.close()
            else:
                await peer.connection.send(b"invalid signed record")
            await until(lambda: not service._zmq_server.peers, timeout=2)
            np.testing.assert_equal(await successor.embed_text(["healthy successor"]), [[3, 4]])
            assert len(sdk.calls) == 1
            assert service.credits.bytes == service.credits.inflight == 0
            assert not service._zmq_server.pending and not service._zmq_server.buffers
            if reserved["shm_name"] is not None:
                with pytest.raises(FileNotFoundError), shm.attach(reserved["shm_name"], 96):
                    pass
        finally:
            peer.close()
            await successor.aclose()
            await service.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("release_before_eof", [False, True])
def test_disconnect_cancels_admission_wait_without_leaking_credits(tmp_path, monkeypatch, release_before_eof):
    sdk = install_sdk(monkeypatch.setitem)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    service = make_service(root, max_inflight=1)

    async def run():
        await service.start()
        peer = RawClient(root)
        locked = False
        try:
            await service.credits.condition.acquire()
            locked = True
            await peer.send_reservation(content("video", video()))
            await until(
                lambda: any(
                    getattr(task.get_coro(), "__qualname__", "") == "_Credits.acquire" for task in asyncio.all_tasks()
                )
            )
            if release_before_eof:
                # Race a newly available admission grant against client EOF.
                service.credits.condition.release()
                locked = False
            peer.close()
            await until(lambda: not service._tasks and not service._zmq_server.peers, timeout=2)
            assert not sdk.calls and not service._zmq_server.pending
            assert not service._zmq_server.buffers
            assert service.credits.bytes == service.credits.inflight == 0
            if locked:
                service.credits.condition.release()
                locked = False
            successor = client(root, timeout=2)
            try:
                np.testing.assert_equal(await successor.embed_text(["healthy successor"]), [[3, 4]])
            finally:
                await successor.aclose()
        finally:
            if locked:
                service.credits.condition.release()
            peer.close()
            await service.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("pause_at", ["publication", "inference"])
@pytest.mark.parametrize("interruption", ["eof", "handler_cancel", "shutdown", "timeout"])
def test_committed_publication_and_inference_keep_resources(tmp_path, monkeypatch, pause_at, interruption):
    sdk = install_sdk(monkeypatch.setitem)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    service = make_service(root, max_inflight=1, **({"publication_timeout": 0.2} if interruption == "timeout" else {}))

    async def run():
        entered, resume = asyncio.Event(), asyncio.Event()

        async def infer(_):
            if pause_at == "inference":
                entered.set()
                await resume.wait()
            return [3, 4]

        sdk.on_embedding = infer
        await service.start()
        server = service._zmq_server
        handle = server._handle

        async def pause_publication(identity, message, size, queue, request_context):
            get = queue.get

            async def paused_get():
                publication = await get()
                entered.set()
                await resume.wait()
                return publication

            queue.get = paused_get
            await handle(identity, message, size, queue, request_context)

        if pause_at == "publication":
            monkeypatch.setattr(server, "_handle", pause_publication)
        peer = RawClient(root)
        closing = None
        try:
            value = video()
            reserved = await peer.reserve(content("video", value))
            name = reserved["shm_name"]
            with shm.attach(name, 96) as memory:
                memory[:] = b"".join(frame.tobytes() for frame in value.frames)
            await peer.publish()
            await asyncio.wait_for(entered.wait(), 2)
            if interruption == "eof":
                peer.close()
                await until(lambda: not server.peers, timeout=2)
            elif interruption == "handler_cancel":
                handler = next(iter(service._tasks))
                for _ in range(3):
                    handler.cancel()
                    await asyncio.sleep(0)
            elif interruption == "shutdown":
                closing = asyncio.create_task(service.aclose())
                await until(lambda: server.receiver is None, timeout=2)
                assert not closing.done()
            else:
                await asyncio.sleep(0.3)
            assert service.credits.bytes == 192 and service.credits.inflight == 1
            assert server.buffers == {name}
            assert len(sdk.calls) == (1 if pause_at == "inference" else 0)
            resume.set()
            if closing is not None:
                await asyncio.wait_for(closing, 2)
            else:
                await until(lambda: not service._tasks, timeout=2)
            assert service.metrics["completed"] == 1 and len(sdk.calls) == 1
            assert service.credits.bytes == service.credits.inflight == 0
            assert not server.pending and not server.buffers
            with pytest.raises(FileNotFoundError), shm.attach(name, 96):
                pass
        finally:
            resume.set()
            peer.close()
            if closing is not None:
                await closing
            else:
                await service.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["text", "video"])
@pytest.mark.parametrize("interruption", ["handler_cancel", "shutdown"])
def test_cancellation_at_publication_commit_runs_inference_once(tmp_path, monkeypatch, kind, interruption):
    sdk = install_sdk(monkeypatch.setitem)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    service = make_service(root, max_inflight=1)

    async def run():
        entered, resume = asyncio.Event(), asyncio.Event()

        async def infer(_):
            entered.set()
            await resume.wait()
            return [3, 4]

        sdk.on_embedding = infer
        await service.start()
        server = service._zmq_server
        message = server._message
        closing = None

        async def cancel_at_commit(identity, remote, value):
            nonlocal closing
            await message(identity, remote, value)
            if value == {"request_id": remote.request_id, "state": "published"}:
                # Do not yield after enqueueing: the real queue consumer has
                # not resumed yet, even though the publication has committed.
                if interruption == "handler_cancel":
                    next(iter(service._tasks)).cancel()
                else:
                    closing = asyncio.create_task(service.aclose())

        monkeypatch.setattr(server, "_message", cancel_at_commit)
        peer = RawClient(root)
        try:
            value = video()
            reserved = await peer.reserve(content("text", "") if kind == "text" else content("video", value))
            name = reserved["shm_name"]
            if name is not None:
                with shm.attach(name, 96) as memory:
                    memory[:] = b"".join(frame.tobytes() for frame in value.frames)
            await peer.publish()
            await asyncio.wait_for(entered.wait(), 2)
            handler = next(iter(service._tasks))
            for _ in range(3):
                handler.cancel()
                await asyncio.sleep(0)
            assert service.credits.inflight == 1
            assert service.credits.bytes == (1 if kind == "text" else 192)
            assert len(sdk.calls) == 1 and service.metrics["completed"] == 0
            if name is not None:
                assert server.buffers == {name}
            resume.set()
            if closing is not None:
                await asyncio.wait_for(closing, 2)
            else:
                await until(lambda: not service._tasks, timeout=2)
            assert len(sdk.calls) == service.metrics["completed"] == 1
            assert service.credits.bytes == service.credits.inflight == 0
            assert not server.pending and not server.buffers
            if name is not None:
                with pytest.raises(FileNotFoundError), shm.attach(name, 96):
                    pass
        finally:
            resume.set()
            peer.close()
            if closing is not None:
                await closing
            else:
                await service.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["admission", "cleanup"])
def test_publication_outside_reservation_wait_is_rejected(tmp_path, monkeypatch, phase):
    sdk = install_sdk(monkeypatch.setitem)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    service = make_service(root, max_inflight=1)

    async def run():
        cancelled = asyncio.Event()
        await service.start()
        server = service._zmq_server
        handle = server._handle

        async def observe_publication_wait(identity, message, size, queue, request_context):
            get = queue.get

            async def observed_get():
                try:
                    return await get()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            queue.get = observed_get
            await handle(identity, message, size, queue, request_context)

        monkeypatch.setattr(server, "_handle", observe_publication_wait)
        peer = RawClient(root)
        locked = False
        try:
            if phase == "cleanup":
                assert (await peer.reserve(content("video", video())))["state"] == "reserved"
            await service.credits.condition.acquire()
            locked = True
            if phase == "admission":
                await peer.send_reservation(content("video", video()))
                await until(lambda: bool(service._tasks), timeout=2)
            else:
                next(iter(service._tasks)).cancel()
                await asyncio.wait_for(cancelled.wait(), 2)
            await peer.publish()
            assert (await asyncio.wait_for(peer.receive(), 2))["error"] == "configuration"
            peer.close()
            service.credits.condition.release()
            locked = False
            await until(lambda: not service._tasks, timeout=2)
            assert not sdk.calls and not server.pending and not server.buffers
            assert service.credits.bytes == service.credits.inflight == 0
        finally:
            if locked:
                service.credits.condition.release()
            peer.close()
            await service.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("bad", ["size", "offset", "dtype", "shape", "timeline"])
def test_wire_bounds_rejected_before_mapping(bad):
    spec = ipc.input_spec({})
    parts, _, size = ipc.describe(content("video", video()), spec)
    if bad == "size":
        size -= 1
    elif bad == "offset":
        parts[0]["buffers"][0]["offset"] = 4
    elif bad == "dtype":
        parts[0]["buffers"][0]["dtype"] = "O"
    elif bad == "shape":
        parts[0]["buffers"][0]["shape"] = [2**50, 4, 3]
    else:
        parts[0]["times"] = [1, 0]
    with pytest.raises(EmbeddingConfigurationError):
        ipc.validate_layout(parts, size, spec)


@pytest.mark.parametrize("times", [(0.0, 0.0), (0.5, 0.5), (0.0, 0.5, 0.5, 1.0)])
def test_video_transport_preserves_nondecreasing_times(times):
    frames = tuple(np.full((4, 4, 3), i + 10, np.uint8) for i in range(len(times)))
    original = VideoClip(frames, times, tuple(i * 3 for i in range(len(times))))
    parts, arrays, size = ipc.describe(content("video", original), ipc.input_spec({}))
    assert ipc.validate_layout(parts, size, ipc.input_spec({})) == size
    restored = ipc.restore(parts, snapshot(arrays, size)).parts[0].value
    assert restored.frame_times == original.frame_times
    assert restored.frame_indices == original.frame_indices
    np.testing.assert_equal(restored.frames, original.frames)


@pytest.mark.parametrize(
    "times,indices",
    [
        ((1.0, 0.5), (0, 1)),
        ((-0.5, 0.5), (0, 1)),
        ((0.5, float("nan")), (0, 1)),
        ((0.5, float("inf")), (0, 1)),
        ((False, 0.5), (0, 1)),
        ((0.5,), (0, 1)),
        ((0.5, 0.5), (0, 0)),
        ((0.5, 0.5), (1, 0)),
        ((0.5, 0.5), (False, 1)),
    ],
)
def test_invalid_video_timeline_rejected_on_both_sides(times, indices):
    spec = ipc.input_spec({})
    original = video()
    with pytest.raises(EmbeddingConfigurationError):
        ipc.describe(content("video", VideoClip(original.frames, times, indices)), spec)
    parts, _, size = ipc.describe(content("video", original), spec)
    parts[0].update(times=list(times), indices=list(indices))
    with pytest.raises(EmbeddingConfigurationError):
        ipc.validate_layout(parts, size, spec)


@pytest.mark.parametrize("value", ["", " \t\n", "\u3000"])
def test_text_transport_preserves_blanks_and_charges_admission(value):
    budget = max(1, len(value.encode("utf-8")))
    spec = ipc.input_spec({"max_input_bytes": budget})
    parts, arrays, size = ipc.describe(content("text", value), spec)
    assert size == len(value.encode("utf-8"))
    assert parts == [{"kind": "text", "offset": 0, "nbytes": size}]
    assert len(arrays) == 1 and arrays[0].tobytes() == value.encode("utf-8")
    assert ipc.validate_layout(parts, size, spec) == budget
    assert ipc.restore(parts, snapshot(arrays, size)) == content("text", value)


@pytest.mark.parametrize("value", [None, 1, b"", [], {}, " " * 5, "\u3000" * 2])
def test_text_transport_keeps_type_and_utf8_budget_validation(value):
    spec = ipc.input_spec({"max_input_bytes": 4})
    with pytest.raises(EmbeddingConfigurationError, match="string|byte budget"):
        ipc.describe(content("text", value), spec)


@pytest.mark.parametrize(
    "bad", ["negative", "boolean", "string", "hole", "overlap", "past_end", "extra", "trailing", "inline"]
)
def test_text_wire_bounds_rejected_before_mapping(bad):
    spec = ipc.input_spec({"max_input_bytes": 3})
    parts = [{"kind": "text", "offset": 0, "nbytes": 1}, {"kind": "text", "offset": 1, "nbytes": 2}]
    if bad in {"negative", "boolean", "string", "past_end"}:
        parts[1]["nbytes"] = {"negative": -1, "boolean": True, "string": "2", "past_end": 3}[bad]
    elif bad in {"hole", "overlap"}:
        parts[1]["offset"] = {"hole": 2, "overlap": 0}[bad]
    elif bad == "extra":
        parts[0]["text"] = "untrusted"
    elif bad == "trailing":
        parts.pop()
    else:
        parts = [{"kind": "text", "text": "old inline representation"}]
    with pytest.raises(EmbeddingConfigurationError):
        ipc.validate_layout(parts, 3, spec)


def test_shared_arrays_preserve_noncontiguous_pixels_and_audio():
    spec = ipc.input_spec({"input_modalities": ["text", "image", "audio"], "sample_rate": 16000, "max_samples": 8})
    original = EmbeddingContent(
        content("text", "中\x00\t").parts
        + content("image", np.zeros((8, 4, 3), np.uint8)[::2]).parts
        + content("audio", AudioClip(np.array([[0.5], [-0.5]], dtype=np.float64), 16000)).parts
        + content("text", "").parts
    )
    parts, arrays, size = ipc.describe(original, spec)
    assert ipc.validate_layout(parts, size, spec) == size
    restored = ipc.restore(parts, snapshot(arrays, size))
    np.testing.assert_equal(restored.parts[1].value, original.parts[1].value)
    np.testing.assert_equal(restored.parts[2].value.samples, original.parts[2].value.samples)
    assert restored.parts[2].value.sample_rate == 16000
    assert restored.parts[0].value == original.parts[0].value
    assert restored.parts[3].value == ""


@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("encoding", ["ascii", "unicode", "escaped"])
def test_long_text_uses_input_budget_and_matches_native_processor(tmp_path, monkeypatch, family, joint, encoding):
    sdk = install_sdk(monkeypatch.setitem)
    value = {
        "ascii": "x" * (ipc.MAX_CONTROL_BYTES + 1),
        "unicode": "\U0001f680" * 100_000,
        "escaped": "\n" * (ipc.MAX_CONTROL_BYTES // 2 + 1),
    }[encoding]
    original = EmbeddingContent(content("text", value).parts + (content("video", video()).parts if joint else ()))
    size = len(value.encode("utf-8")) + (96 if joint else 0)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    options = {"gpus_per_actor": 0, "max_input_bytes": size}
    descriptor = getattr(load_provider(family), f"get_{'multimodal' if joint else 'text'}_embedder")(
        "fixture", 2, options=options
    )
    service = EmbeddingService(
        path=str(root / "engine.json"),
        service_id="fixture-v1",
        family=family,
        model="fixture",
        dimensions=2,
        options=options,
        max_bytes=2 * size,
        max_inflight=1,
    )

    def prompt():
        return sdk.processing[-1]["text"] if family == "sglang" else sdk.calls[-1][1]["prompt"]

    async def run():
        await service.start()
        native, instance = descriptor.instantiate(), client(root, family, max_input_bytes=size)
        instance.configure_execution(on_error="ignore", max_retries=3, validate=lambda v: v)
        try:
            parts, _, length = ipc.describe(original, instance.spec)
            assert length == size
            assert len(ipc.encode(instance._reservation(parts, length, "0" * 32))) < 2048
            expected = await native.embed_multimodal([original]) if joint else await native.embed_text([value])
            expected_prompt = prompt()
            actual = await instance.embed_multimodal([original]) if joint else await instance.embed_text([value])
            np.testing.assert_equal(actual, expected)
            assert prompt() == expected_prompt
            assert sdk.messages[-1]["content"][0] == {"type": "text", "text": value}
            assert instance.metrics["input_bytes"] == size
            assert service.credits.bytes == service.credits.inflight == 0
            with pytest.raises(EmbeddingConfigurationError, match="byte budget"):
                await instance.embed_multimodal([EmbeddingContent(original.parts + content("text", "x").parts)])
            assert len(sdk.calls) == 2
        finally:
            await instance.aclose()
            await native.aclose()
            await service.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("family", ["sglang", "vllm"])
def test_invalid_utf8_releases_admission_without_model_work(tmp_path, monkeypatch, family):
    sdk = install_sdk(monkeypatch.setitem)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    service = make_service(root, family=family, max_inflight=1)

    async def run():
        await service.start()
        instance, peer = client(root, family), RawClient(root, family)
        try:
            response = await peer.reserve(content("text", "x"))
            with shm.attach(response["shm_name"], 1) as memory:
                memory[0] = 255
            await peer.publish()
            assert (await peer.receive())["error"] == "configuration"
            await until(lambda: not service._tasks)
            assert not sdk.calls
            assert service.credits.bytes == service.credits.inflight == 0
            np.testing.assert_equal(await instance.embed_text(["healthy"]), [[3, 4]])
        finally:
            peer.close()
            await instance.aclose()
            await service.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("family", ["sglang", "vllm"])
def test_input_and_identity_errors_are_not_ignored(process, family):
    proc, root = process(family)

    async def run():
        await until(lambda: (root / "ready").exists() or proc.poll() is not None)
        instance = client(root, family, service_id="wrong")
        instance.configure_execution(max_retries=3, on_error="ignore", validate=lambda v: v)
        with pytest.raises(EmbeddingConfigurationError):
            await instance.embed_text(["query"])
        with pytest.raises(EmbeddingConfigurationError):
            await instance.embed_multimodal([content(), content("video", "path.mp4")])
        assert state(root)["calls"] == 0
        await instance.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("transport", ["native", "service"])
def test_joint_audio_video_and_text_use_one_native_request(tmp_path, monkeypatch, family, transport):
    state = install_sdk(monkeypatch.setitem)
    options = {
        "gpus_per_actor": 0,
        "input_modalities": ["text", "image", "video", "audio"],
        "sample_rate": 16000,
        "max_samples": 16000,
    }
    descriptor = load_provider(family).get_multimodal_embedder("fixture", 2, options=options)
    row = EmbeddingContent(
        content().parts
        + content("image", np.full((4, 4, 3), 20, np.uint8)).parts
        + content("video", video()).parts
        + content("audio", AudioClip(np.zeros((16, 1), np.float32), 16000)).parts
    )

    async def run():
        service = None
        if transport == "service":
            root = tmp_path / "private"
            root.mkdir(mode=0o700)
            service = EmbeddingService(
                path=str(root / "engine.json"),
                service_id="fixture-v1",
                family=family,
                model="fixture",
                dimensions=2,
                options=options,
            )
            await service.start()
            runtime = client(
                root, family, input_modalities=["text", "image", "video", "audio"], sample_rate=16000, max_samples=16000
            )
        else:
            runtime = descriptor.instantiate()
        try:
            np.testing.assert_equal(await runtime.embed_multimodal([row]), [[3, 4]])
        finally:
            await runtime.aclose()
            if service is not None:
                await service.aclose()

    asyncio.run(run())
    assert len(state.calls) == 1
    if family == "sglang":
        assert state.processing[0]["audio_kwargs"]["sampling_rate"] == 16000
        assert len(state.processing[0]["audio"]) == len(state.processing[0]["videos"]) == 1
        request = state.calls[0][1]
        assert sum(bool(request.get(key)) for key in ("image_data", "video_data", "audio_data")) == 1
    else:
        assert set(state.calls[0][1]["multi_modal_data"]) == {"image", "video", "audio"}


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning:transformers.processing_utils"
)
@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("times", [(0.2, 0.6, 1.4), (0.5, 0.5), (0.0, 0.5, 0.5, 1.0)])
@pytest.mark.parametrize("joint", [False, True])
def test_real_processor_equal_for_direct_and_shared_frames(
    tmp_path, monkeypatch, family, times, joint, make_real_video_processor
):
    import transformers
    from transformers.video_utils import VideoMetadata

    processor = make_real_video_processor()
    state = install_sdk(monkeypatch.setitem, mock_transformers=False)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **kw: processor)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    options = {
        "gpus_per_actor": 0,
        "processor_kwargs": {"videos_kwargs": {"size": {"shortest_edge": 4096, "longest_edge": 8192}}},
    }
    clip = VideoClip(
        tuple(np.full((64, 64, 3), 10 * (i + 1), np.uint8) for i in range(len(times))),
        times,
        tuple(2 + i * 4 for i in range(len(times))),
    )
    row = EmbeddingContent(content("video", clip).parts + (content().parts if joint else ()))
    kind = "multimodal" if joint else "video"
    service = EmbeddingService(
        path=str(root / "engine.json"),
        service_id="fixture-v1",
        family=family,
        model="fixture",
        dimensions=2,
        options=options,
    )

    async def run():
        await service.start()
        instance = client(root, family, kind=kind)
        native = getattr(load_provider(family), f"get_{kind}_embedder")("fixture", 2, options=options).instantiate()
        for runtime in (native, instance):
            runtime.configure_execution(on_error="ignore", max_retries=0, validate=lambda v: v)
        try:
            for runtime in (native, instance):
                result = await runtime.embed_multimodal([row]) if joint else await runtime.embed_video([clip])
                np.testing.assert_equal(result, [[3, 4]])
            assert service.credits.bytes == service.credits.inflight == 0
        finally:
            await native.aclose()
            await instance.aclose()
            await service.aclose()

    asyncio.run(run())
    assert len(state.engines) == len(state.calls) == 2

    def tensors(call):
        request = call[1]
        if family == "sglang":
            return request["video_data"][0]
        frames, metadata = request["multi_modal_data"]["video"][0]
        return processor(
            text=[request["prompt"]],
            videos=[frames],
            video_metadata=[VideoMetadata(**{k: v for k, v in metadata.items() if k != "do_sample_frames"})],
            **request["mm_processor_kwargs"],
        )

    direct, shared = (tensors(call) for call in state.calls)
    expected = processor(
        text=["<|video_pad|>"],
        videos=[np.stack(clip.frames)],
        video_metadata=[
            VideoMetadata(
                fps=10, frames_indices=[round(t * 10) for t in times], total_num_frames=round(times[-1] * 10) + 1
            )
        ],
        videos_kwargs=options["processor_kwargs"]["videos_kwargs"],
        do_sample_frames=False,
        add_special_tokens=False,
        return_tensors="pt",
    )
    assert direct.keys() == shared.keys()
    for key in direct:
        np.testing.assert_equal(
            direct[key] if isinstance(direct[key], str) else np.asarray(direct[key]),
            shared[key] if isinstance(shared[key], str) else np.asarray(shared[key]),
        )
    for key in expected:
        np.testing.assert_equal(np.asarray(direct[key]), np.asarray(expected[key]))


@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
@pytest.mark.skipif(os.name != "posix", reason="Generic UDF subprocess launcher requires POSIX file descriptors")
def test_public_embed_runs_in_real_workers(process, family, backend):
    check_public_embed(process, family, backend)


@pytest.mark.real_ray
@pytest.mark.ray_cluster_owner
@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("backend", ["ray_task", "ray_actor"])
def test_public_embed_uses_service_on_ray(process, monkeypatch, family, backend):
    import ray

    import vane

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    try:
        ray.init(address="local", num_cpus=3, include_dashboard=False)
        check_public_embed(process, family, backend)
    finally:
        vane.teardown_runner()
        ray.shutdown()


def check_public_embed(process, family, backend):
    import pyarrow as pa

    import vane
    from tests.fast.test_ai_video_embedding import clip, clip_array
    from vane.ai import MultimodalInput, TextInput, VideoInput

    proc, root = process(family)
    asyncio.run(until(lambda: (root / "ready").exists() or proc.poll() is not None))
    with vane.connect() as connection:
        source = connection.from_arrow(
            pa.table({"id": [0, 1, 2], "text": ["ingest", "search", "skip"], "selected": [True, True, False]})
        )
        result = source.embed(
            MultimodalInput([TextInput(vane.col("text")), TextInput(vane.ConstantExpression(None))]),
            provider=family,
            model="fixture",
            dimensions=2,
            runtime="service",
            service_path=str(root / "engine.json"),
            service_id="fixture-v1",
            execution_backend=backend,
            where=vane.col("selected"),
            output_column="vector",
            normalize=False,
        )
        assert result.select("id", "vector").order("id").fetchall() == [(0, (3, 4)), (1, (3, 4)), (2, None)]
        texts = connection.from_arrow(
            pa.table({"id": list(range(6)), "text": ["healthy", "", " \t\n", "\u3000", None, "\U0001f680" * 100_000]})
        )
        result = texts.embed(
            TextInput(vane.col("text")),
            provider=family,
            model="fixture",
            dimensions=2,
            runtime="service",
            service_path=str(root / "engine.json"),
            service_id="fixture-v1",
            execution_backend=backend,
            on_error="ignore",
            output_column="vector",
            normalize=False,
        )
        assert result.select("id", "vector").order("id").fetchall() == [
            (0, (3, 4)),
            (1, (3, 4)),
            (2, (3, 4)),
            (3, (3, 4)),
            (4, None),
            (5, (3, 4)),
        ]
        records = clip()
        records[1]["frame_time"] = records[0]["frame_time"]
        videos = connection.from_arrow(pa.table({"frames": clip_array([records]), "text": ["query"]}))
        for embedding_input in (
            VideoInput(vane.col("frames")),
            MultimodalInput([VideoInput(vane.col("frames")), TextInput(vane.col("text"))]),
        ):
            result = videos.embed(
                embedding_input,
                provider=family,
                model="fixture",
                dimensions=2,
                runtime="service",
                service_path=str(root / "engine.json"),
                service_id="fixture-v1",
                execution_backend=backend,
                on_error="ignore",
                output_column="vector",
                normalize=False,
            )
            assert result.select("vector").fetchall() == [((3, 4),)]
    asyncio.run(
        until(lambda: state(root).get("calls") == 9 and state(root).get("bytes") == state(root).get("inflight") == 0)
    )
    assert state(root)["engines"] == 1


@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
@pytest.mark.skipif(os.name != "posix", reason="Generic UDF subprocess launcher requires POSIX file descriptors")
def test_public_embed_resource_failures_abort_without_retry(process, family, backend):
    check_public_resource_failure(process, family, backend)


@pytest.mark.real_ray
@pytest.mark.ray_cluster_owner
@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("backend", ["ray_task", "ray_actor"])
def test_public_embed_resource_failures_abort_on_ray(process, monkeypatch, family, backend):
    import ray

    import vane

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    try:
        ray.init(address="local", num_cpus=3, include_dashboard=False)
        check_public_resource_failure(process, family, backend)
    finally:
        vane.teardown_runner()
        ray.shutdown()


def check_public_resource_failure(process, family, backend):
    import pyarrow as pa

    import vane
    from vane.ai import MultimodalInput, TextInput

    proc, root = process(family)
    asyncio.run(until(lambda: (root / "ready").exists() or proc.poll() is not None))
    assert proc.poll() is None

    def query(joint=False):
        inputs = TextInput(vane.col("text"))
        if joint:
            inputs = MultimodalInput([inputs])
        with vane.connect() as connection:
            return (
                connection.from_arrow(pa.table({"text": ["query"]}))
                .embed(
                    inputs,
                    provider=family,
                    model="fixture",
                    dimensions=2,
                    runtime="service",
                    service_path=str(root / "engine.json"),
                    service_id="fixture-v1",
                    execution_backend=backend,
                    on_error="ignore",
                    max_retries=3,
                    output_column="vector",
                    normalize=False,
                )
                .select("vector")
                .fetchall()
            )

    for flag in ("memory_error", "wrapped_enospc"):
        (root / flag).touch()
        for joint in (False, True):
            with pytest.raises(Exception, match="MemoryError|resources") as failure:
                query(joint)
            assert "private model" not in str(failure.value)
        (root / flag).unlink()
    (root / "execution_error").touch()
    assert query() == [(None,)]
    (root / "execution_error").unlink()
    assert query() == [((3, 4),)]
    # Ray's existing fragment policy attempts a failed UDF four times. The
    # service client must not multiply those attempts by max_retries=3.
    failed_attempts = 4 if backend.startswith("ray_") else 1
    expected_calls = 4 * failed_attempts + 2
    asyncio.run(until(lambda: state(root).get("calls") == expected_calls and state(root).get("inflight") == 0))
    assert state(root)["engines"] == 1


@pytest.mark.parametrize("failure", ["memory", "torch", "wrapped", "cpu", "wrapped_cpu", "execution"])
def test_resource_failures_remain_fatal_with_ignore(tmp_path, monkeypatch, failure):
    from vane.ai import DiagnosticError

    torch = pytest.importorskip("torch") if failure == "torch" else None
    state = install_sdk(monkeypatch.setitem)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)

    async def fail(_):
        if "cpu" in failure:
            error = RuntimeError(
                "DefaultCPUAllocator: can't allocate memory: you tried to allocate 1073741824 bytes. "
                "Error code 12 (Cannot allocate memory)"
            )
            if failure == "wrapped_cpu":
                raise pickle.loads(pickle.dumps(DiagnosticError("fixture", error)))
            raise error
        if failure == "torch":
            raise torch.OutOfMemoryError("fixture without allocation")
        if failure == "wrapped":
            raise DiagnosticError("fixture", MemoryError("private data"))
        if failure == "memory":
            raise MemoryError("private data")
        raise RuntimeError("private data")

    state.on_embedding = fail
    service = EmbeddingService(
        path=str(root / "engine.json"),
        service_id="fixture-v1",
        family="sglang",
        model="fixture",
        dimensions=2,
        options={"gpus_per_actor": 0},
    )

    async def run():
        await service.start()
        instance = client(root)
        instance.configure_execution(max_retries=3, on_error="ignore", validate=lambda value: value)
        try:
            if failure == "execution":
                assert await instance.embed_text(["query"]) == [None]
            else:
                with pytest.raises(MemoryError, match="resources"):
                    await instance.embed_text(["query"])
            assert service.credits.bytes == service.credits.inflight == 0
            assert len(state.calls) == 1
        finally:
            await instance.aclose()
            await service.aclose()

    asyncio.run(run())


def test_handler_cancellation_joins_processor_before_unmapping(tmp_path, monkeypatch):
    state = install_sdk(monkeypatch.setitem)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def wait(_):
            entered.set()
            await release.wait()
            return [3, 4]

        state.on_embedding = wait
        service = EmbeddingService(
            path=str(root / "engine.json"),
            service_id="fixture-v1",
            family="sglang",
            model="fixture",
            dimensions=2,
            options={"gpus_per_actor": 0},
        )
        await service.start()
        instance = client(root)
        try:
            pending = asyncio.create_task(instance.embed_multimodal([content("video", video())]))
            await asyncio.wait_for(entered.wait(), 3)
            handler = next(iter(service._tasks))
            handler.cancel()
            await asyncio.sleep(0)
            handler.cancel()
            await asyncio.sleep(0)
            assert service.credits.bytes == 192 and not handler.done()
            release.set()
            np.testing.assert_equal(await pending, [[3, 4]])
            assert service.credits.bytes == 0
        finally:
            release.set()
            await instance.aclose()
            await service.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["queued", "handoff", "release"])
@pytest.mark.parametrize("kind", ["text", "video"])
def test_repeated_handler_cancellation_returns_admission_credits(tmp_path, monkeypatch, phase, kind):
    state = install_sdk(monkeypatch.setitem)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    service = make_service(root, max_inflight=1)

    async def run():
        await service.start()
        peer = RawClient(root)
        locked = False
        try:
            if phase != "release":
                await service.credits.condition.acquire()
                locked = True
            value = content() if kind == "text" else content("video", video())
            await peer.send_reservation(value)
            if phase == "release":
                assert (await peer.receive())["state"] == "reserved"
                await service.credits.condition.acquire()
                locked = True
            else:
                await until(
                    lambda: any(
                        getattr(task.get_coro(), "__qualname__", "") == "_Credits.acquire"
                        for task in asyncio.all_tasks()
                    )
                )
            handler = next(iter(service._tasks))
            if phase == "handoff":
                service.credits.condition.release()
                locked = False
            handler.cancel()
            await asyncio.sleep(0)
            handler.cancel()
            await asyncio.sleep(0)
            handler.cancel()
            if locked:
                service.credits.condition.release()
                locked = False
            await until(lambda: not service._tasks)
            assert service.credits.bytes == service.credits.inflight == 0
            assert not state.calls and not service._zmq_server.pending
            instance = client(root, timeout=3)
            try:
                np.testing.assert_equal(await instance.embed_text(["healthy successor"]), [[3, 4]])
                assert len(state.calls) == 1
            finally:
                await instance.aclose()
        finally:
            if locked:
                service.credits.condition.release()
            peer.close()
            await service.aclose()

    asyncio.run(run())


def test_unsigned_request_cannot_reserve_memory(process):
    proc, root = process()

    async def run():
        await until(lambda: (root / "ready").exists() or proc.poll() is not None)
        peer = RawClient(root)
        try:
            peer.key = b"wrong key"
            with pytest.raises(EmbeddingConfigurationError, match="authentication"):
                await peer.send_reservation(content())
            # The service authenticates every request before any allocation.
            await asyncio.sleep(0.1)
            assert state(root)["inflight"] == state(root)["calls"] == 0
        finally:
            peer.close()

    asyncio.run(run())


def test_model_reads_private_snapshot_after_publication(tmp_path, monkeypatch):
    install_sdk(monkeypatch.setitem)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    service = make_service(root, max_bytes=192, max_inflight=1)
    original = video()

    async def run():
        entered, finish = asyncio.Event(), asyncio.Event()

        async def infer(value, request_context):
            assert request_context.encoding == "default" and request_context.model_contract is None
            assert len(request_context.request_id) == 32
            entered.set()
            await finish.wait()
            np.testing.assert_equal(value.parts[0].value.frames, original.frames)
            return {"vector": [3, 4]}

        monkeypatch.setattr(service, "_infer", infer)
        await service.start()
        peer = RawClient(root)
        try:
            response = await peer.reserve(content("video", original))
            with shm.attach(response["shm_name"], 96) as writer:
                writer[:] = b"".join(frame.tobytes() for frame in original.frames)
                await peer.publish()
                await asyncio.wait_for(entered.wait(), 3)
                assert service.credits.bytes == 192
                writer[:] = bytes(96)
                finish.set()
                assert (await peer.receive())["vector"] == [3, 4]
            assert service.credits.bytes == service.credits.inflight == 0
            assert not service._zmq_server.buffers
        finally:
            finish.set()
            peer.close()
            await service.aclose()

    asyncio.run(run())


def test_borrowed_mapping_exposes_only_requested_bytes(monkeypatch):
    # Exercise Windows' rounded allocation size on every host.
    memory = shm.create(4096)
    try:
        monkeypatch.setattr(shm, "os", SimpleNamespace(name="nt"))
        with shm.attach(memory.name, 16) as borrowed:
            assert len(borrowed) == 16
            borrowed[:] = b"x" * 16
        data = bytes(memory.buf)
        assert data[:16] == b"x" * 16
        assert data[16:] == bytes(memory.size - 16)
    finally:
        shm.release(memory)


@pytest.mark.parametrize(
    "size", sorted({1, 9, 16, 4095, 4096, mmap.PAGESIZE - 1, mmap.PAGESIZE, mmap.PAGESIZE + 1, 2 * mmap.PAGESIZE + 1})
)
def test_shared_memory_page_boundary_round_trip(size):
    memory = shm.create(size)
    try:
        with shm.attach(memory.name, size) as writer:
            assert len(writer) == size
            writer[:] = b"x" * size
        data = bytes(memory.buf)
        assert data[:size] == b"x" * size
        assert data[size:] == bytes(memory.size - size)
        with shm.attach(memory.name, size) as reader:
            assert len(reader) == size
            assert bytes(reader[:]) == b"x" * size
    finally:
        shm.release(memory)


@pytest.fixture(params=["fd", "track_false"])
def posix_attachment_path(request, monkeypatch):
    if os.name != "posix":
        pytest.skip("POSIX shared-memory attachment branches")
    shared_memory = shm.shared_memory.SharedMemory
    attached = []

    def attach_existing(*, name, track):
        assert track is False
        options = {"track": False} if sys.version_info >= (3, 13) else {}
        memory = shared_memory(name=name, **options)
        attached.append(memory)
        return memory

    def activate():
        # Keep allocation native to the host, then exercise both attachment
        # paths even when the interpreter does not provide track=False.
        monkeypatch.setattr(
            shm,
            "shared_memory",
            SimpleNamespace(
                _posixshmem=shm.shared_memory._posixshmem,
                SharedMemory=attach_existing,
            ),
        )
        monkeypatch.setattr(
            shm,
            "sys",
            SimpleNamespace(
                platform="darwin",
                version_info=(3, 12) if request.param == "fd" else (3, 13),
            ),
        )
        monkeypatch.setattr(shm, "mmap", SimpleNamespace(PAGESIZE=16384, mmap=mmap.mmap))

    yield activate
    for memory in attached:
        assert memory.buf is None


@pytest.mark.parametrize(
    "size,allocation",
    [
        (1, 16384),
        (9, 16384),
        (16, 16384),
        (4095, 16384),
        (4096, 16384),
        (16383, 16384),
        (16384, 16384),
        (16385, 32768),
        (32769, 49152),
    ],
)
def test_darwin_shared_memory_rounding(size, allocation, posix_attachment_path):
    memory = shm.create(allocation)
    try:
        posix_attachment_path()
        with shm.attach(memory.name, size) as writer:
            assert len(writer) == size
            writer[:] = b"x" * size
        data = bytes(memory.buf)
        assert data[:size] == b"x" * size
        assert data[size:] == bytes(memory.size - size)
    finally:
        shm.release(memory)


@pytest.mark.parametrize(
    "platform,size,allocation",
    [
        ("darwin", 16385, 16384),
        ("darwin", 9, 32768),
        ("linux", 16385, 16384),
        ("linux", 9, 16384),
    ],
)
def test_shared_memory_rejects_unexpected_allocation(platform, size, allocation, posix_attachment_path, monkeypatch):
    memory = shm.create(allocation)
    try:
        posix_attachment_path()
        monkeypatch.setattr(shm.sys, "platform", platform)
        with (
            pytest.raises(EmbeddingConfigurationError, match="shared memory size mismatch"),
            shm.attach(memory.name, size),
        ):
            pytest.fail("Invalid allocation was exposed to the producer")
    finally:
        shm.release(memory)


def test_independent_client_exit_does_not_unlink_server_buffer():
    memory = shm.create(16)
    try:
        code = """
import sys
from vane.ai import _embedding_shared_memory as shm
with shm.attach(sys.argv[1], 16) as memory:
    memory[:] = b'x' * 16
"""
        result = subprocess.run([sys.executable, "-c", code, memory.name], capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        assert "resource_tracker" not in result.stderr
        with shm.attach(memory.name, 16) as reader:
            assert bytes(reader[:]) == b"x" * 16
    finally:
        shm.release(memory)


def test_shutdown_drains_accepted_request_before_closing_engine(tmp_path, monkeypatch):
    sdk = install_sdk(monkeypatch.setitem)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    service = make_service(root)

    async def run():
        entered, finish = asyncio.Event(), asyncio.Event()

        async def infer(_):
            entered.set()
            await finish.wait()
            return [3, 4]

        sdk.on_embedding = infer
        await service.start()
        instance = client(root, timeout=3)
        pending = asyncio.create_task(instance.embed_text(["query"]))
        closing = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            closing = asyncio.create_task(service.aclose())
            await asyncio.sleep(0.05)
            assert not closing.done() and service.credits.bytes == 10
            assert service.backend.runtime is not None
            finish.set()
            np.testing.assert_equal(await pending, [[3, 4]])
            await closing
            assert service.credits.bytes == service.credits.inflight == 0
            assert service.backend.runtime is None and not (root / "engine.json").exists()
        finally:
            finish.set()
            await instance.aclose()
            if closing is not None:
                await closing
            else:
                await service.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("failure", [TimeoutError, asyncio.CancelledError])
def test_failed_engine_close_keeps_runtime_for_shared_retry(tmp_path, monkeypatch, family, failure):
    sdk = install_sdk(monkeypatch.setitem)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    service = make_service(root, family)

    async def run():
        await service.start()
        runtime = service.backend.runtime
        engine = runtime.engine
        finish = asyncio.Event()
        attempts = 0
        pending = []

        async def shutdown():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise failure("authored shutdown failure")
            await finish.wait()
            engine.closed += 1

        monkeypatch.setattr(engine, "shutdown", shutdown)
        try:
            with pytest.raises(failure):
                await service.aclose()
            assert service.backend.runtime is runtime and runtime.engine is engine
            assert not (root / "engine.json").exists()
            with pytest.raises(RuntimeError, match="already started"):
                await service.start()
            pending = [asyncio.create_task(service.aclose()) for _ in range(3)]
            await until(lambda: attempts == 2)
            pending[0].cancel()
            await asyncio.sleep(0.02)
            assert attempts == 2 and all(not task.done() for task in pending)
            finish.set()
            await asyncio.gather(*pending)
            assert service.backend.runtime is None and engine.closed == 1
            assert len(sdk.engines) == 1
            await service.aclose()
            assert attempts == 2
        finally:
            finish.set()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await runtime.aclose()

    asyncio.run(run())


def test_failed_transport_close_keeps_transport_and_engine_until_retry(tmp_path, monkeypatch):
    install_sdk(monkeypatch.setitem)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    service = make_service(root)

    async def run():
        await service.start()
        runtime, server = service.backend.runtime, service._zmq_server
        engine = runtime.engine
        close = server.aclose
        attempts = 0

        async def interrupted():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TimeoutError("authored transport shutdown failure")
            await close()

        monkeypatch.setattr(server, "aclose", interrupted)
        try:
            with pytest.raises(TimeoutError, match="transport shutdown"):
                await service.aclose()
            assert service._zmq_server is server and service.backend.runtime is runtime
            assert engine.closed == 0
            await service.aclose()
            assert service._zmq_server is service.backend.runtime is None
            assert engine.closed == 1 and attempts == 2
            assert not (root / "engine.json").exists()
        finally:
            await close()
            await runtime.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("family", ["sglang", "vllm"])
def test_empty_request_round_trip_needs_no_shared_memory(tmp_path, monkeypatch, family):
    sdk = install_sdk(monkeypatch.setitem)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    service = make_service(root, family, max_bytes=1, max_inflight=1)

    async def infer(_):
        assert service.credits.bytes == service.credits.inflight == 1
        assert not service._zmq_server.buffers
        return [3, 4]

    sdk.on_embedding = infer

    async def run():
        await service.start()
        instance = client(root, family)
        try:
            np.testing.assert_equal(await instance.embed_text([""]), [[3, 4]])
            assert service.credits.bytes == service.credits.inflight == 0
        finally:
            await instance.aclose()
            await service.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    "bad", ["json", "protocol", "legacy_protocol", "remote", "key", "file_permissions", "directory_permissions"]
)
def test_invalid_rendezvous_is_fatal_even_with_ignore(tmp_path, monkeypatch, bad):
    if os.name != "posix" and "permissions" in bad:
        pytest.skip("POSIX permission bits; Windows uses the user directory ACL")
    install_sdk(monkeypatch.setitem)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    service = make_service(root)

    async def run():
        await service.start()
        path = root / "engine.json"
        data = json.loads(path.read_text())
        if bad == "json":
            path.write_text("{")
        elif bad == "file_permissions":
            path.chmod(0o644)
        elif bad == "directory_permissions":
            root.chmod(0o755)
        else:
            data[{"protocol": "protocol", "legacy_protocol": "protocol", "remote": "endpoint", "key": "key"}[bad]] = {
                "protocol": "unix",
                "legacy_protocol": "vane-embedding-stream-2",
                "remote": "tcp://example.com:1234",
                "key": "invalid",
            }[bad]
            path.write_text(json.dumps(data))
        instance = client(root)
        instance.configure_execution(on_error="ignore", max_retries=3, validate=lambda value: value)
        try:
            with pytest.raises(EmbeddingConfigurationError):
                await instance.embed_text(["query"])
            assert not service._tasks and service.metrics["accepted"] == 0
        finally:
            await instance.aclose()
            await service.aclose()

    asyncio.run(run())


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO rendezvous")
def test_fifo_rendezvous_fails_without_blocking(tmp_path):
    path = tmp_path / "engine.json"
    os.mkfifo(path, 0o600)
    code = """
import sys
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._embedding_zmq import _read_endpoint
try:
    _read_endpoint(sys.argv[1])
except EmbeddingConfigurationError:
    pass
else:
    raise AssertionError("A FIFO must not be accepted as a rendezvous file")
"""
    # A separate process makes a blocking open fail within a bounded deadline.
    result = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_missing_zmq_does_not_affect_planning_and_cleans_failed_start(tmp_path, monkeypatch):
    from vane.ai.provider import ProviderImportError

    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    monkeypatch.setitem(sys.modules, "zmq", None)
    monkeypatch.setitem(sys.modules, "zmq.asyncio", None)
    instance = client(root)
    assert instance._zmq_client is None
    service = make_service(root)

    async def run():
        with pytest.raises(ProviderImportError, match="embedding_service"):
            await service.start()
        assert not (root / "engine.json").exists()
        assert service._zmq_server is None and service.backend.runtime is None
        instance.configure_execution(on_error="ignore", max_retries=3, validate=lambda value: value)
        with pytest.raises(ProviderImportError, match="embedding_service"):
            await instance.embed_text(["query"])
        await instance.aclose()

    asyncio.run(run())


class RawClient:
    def __init__(self, root, family="sglang"):
        self.endpoint, self.key = transport._read_endpoint(str(root / "engine.json"))
        self.context = zmq.asyncio.Context()
        self.socket = transport._socket(self.context, zmq.STREAM)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.request_id = uuid.uuid4().hex
        self.socket.connect(self.endpoint)
        self.connection = None
        self.family = family

    async def send_reservation(self, value):
        parts, _, size = ipc.describe(value, ipc.input_spec({}))
        await self.send(
            {
                "version": ipc.VERSION,
                "service_id": "fixture-v1",
                "family": self.family,
                "model": "fixture",
                "dimensions": 2,
                "request_id": self.request_id,
                "parts": parts,
                "nbytes": size,
                "encoding": "default",
                "model_contract": None,
            }
        )

    async def reserve(self, value):
        await self.send_reservation(value)
        return await self.receive()

    async def send(self, message):
        if self.connection is None:
            self.connection = transport.stream.Connection(self.socket, self.key, self.request_id)
            await self.connection.authenticate()
        await self.connection.send(transport._pack(message, self.connection.key, b"request"))

    async def publish(self):
        await self.send({"request_id": self.request_id, "state": "published"})

    async def receive(self):
        return transport._unpack(await asyncio.wait_for(self.connection.receive(), 5), self.connection.key, b"response")

    def close(self):
        self.socket.close(linger=0)
        self.context.term()


def snapshot(arrays, size):
    if not size:
        return None
    memory = shm.create(size)
    try:
        with shm.attach(memory.name, size) as writer:
            offset = 0
            for value in arrays:
                target = np.ndarray(value.shape, dtype=value.dtype, buffer=writer, offset=offset)
                np.copyto(target, value)
                del target
                offset += value.nbytes
        return bytearray(memory.buf[:size])
    finally:
        shm.release(memory)


def make_service(root, family="sglang", **options):
    return EmbeddingService(
        path=str(root / "engine.json"),
        service_id="fixture-v1",
        family=family,
        model="fixture",
        dimensions=2,
        options={"gpus_per_actor": 0},
        **options,
    )
