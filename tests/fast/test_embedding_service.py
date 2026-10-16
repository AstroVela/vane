# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Persistent native service: real sockets, memfd, process death and processors."""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.fast.test_native_multimodal import install_sdk
from tests.fast.test_native_multimodal import make_real_video_processor as processor_fixture
from vane.ai import _embedding_ipc as ipc
from vane.ai._audio_embedding import AudioClip
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._video_embedding import VideoClip
from vane.ai.embedding_service import EmbeddingService
from vane.ai.provider import load_provider

make_real_video_processor = processor_fixture

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux memfd and Unix peer credentials")


def content(kind="text", value="query"):
    return EmbeddingContent((EmbeddingPart(EmbeddingPartSpec(kind), value),))


def video():
    frames = tuple(np.full((4, 4, 3), value, dtype=np.uint8) for value in (10, 20))
    return VideoClip(frames, (0.125, 1.375), (2, 5))


def client(root, family="sglang", kind="multimodal", **options):
    descriptor = getattr(load_provider(family), f"get_{kind}_embedder")(
        "fixture",
        2,
        options={"runtime": "service", "service_path": str(root / "engine.sock"), "service_id": "fixture-v1"} | options,
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
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
                pytest.fail("Embedding fixture failed to drain")
        log.close()
        assert proc.returncode in (0, -9), (root / "log.txt").read_text()


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
    assert clone.instantiate()._sockets == set()


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
def test_cancelled_mapped_input_holds_credit_until_model_finishes(process, family):
    proc, root = process(family, max_bytes=100, max_inflight=2)
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
        assert state(root)["bytes"] == 96 and state(root)["calls"] == 1
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
        producer = subprocess.Popen([sys.executable, "-c", code, str(root / "engine.sock")])
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


def test_server_death_closes_client_and_anonymous_buffer(process):
    proc, root = process()
    (root / "hold").touch()

    def buffers():
        return [
            name
            for name in Path("/proc/self/fd").iterdir()
            if name.exists() and "memfd:vane_embedding" in os.readlink(name)
        ]

    async def run():
        await until(lambda: (root / "ready").exists() or proc.poll() is not None)
        instance = client(root)
        task = asyncio.create_task(instance.embed_multimodal([content("video", video())]))
        await until(lambda: state(root).get("active") == 1)
        assert not buffers()  # Producer already handed off and closed its FD.
        proc.kill()
        proc.wait(timeout=5)
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(task, 3)
        assert not buffers() and not instance._sockets
        await instance.aclose()

    asyncio.run(run())


def test_disconnect_before_publication_releases_reservation(process):
    proc, root = process(publication_timeout=0.2)

    async def run():
        await until(lambda: (root / "ready").exists() or proc.poll() is not None)
        for disconnect in (True, False):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.setblocking(False)
            await asyncio.get_running_loop().sock_connect(sock, str(root / "engine.sock"))
            parts, _, size = ipc.describe(content("video", video()), ipc.input_spec({}))
            await ipc.send(
                sock,
                {
                    "version": 1,
                    "service_id": "fixture-v1",
                    "family": "sglang",
                    "model": "fixture",
                    "dimensions": 2,
                    "request_id": "0" * 32,
                    "parts": parts,
                    "nbytes": size,
                },
            )
            assert (await ipc.control(sock))["state"] == "reserved"
            if disconnect:
                sock.close()
            else:
                assert "error" in await ipc.control(sock)
                sock.close()
            await until(lambda: state(root).get("inflight") == 0)
        assert state(root)["calls"] == 0

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


@pytest.mark.parametrize("value", ["", " \t\n", "\u3000"])
def test_text_transport_preserves_blanks_and_charges_admission(value):
    budget = max(1, len(value.encode("utf-8")))
    spec = ipc.input_spec({"max_input_bytes": budget})
    parts, arrays, size = ipc.describe(content("text", value), spec)
    assert parts == [{"kind": "text", "text": value}]
    assert arrays == [] and size == 0
    assert ipc.validate_layout(parts, size, spec) == budget
    assert ipc.sealed_buffer(arrays, size) is None
    assert ipc.restore(parts, None) == content("text", value)


@pytest.mark.parametrize("value", [None, 1, b"", [], {}, " " * 5, "\u3000" * 2])
def test_text_transport_keeps_type_and_utf8_budget_validation(value):
    spec = ipc.input_spec({"max_input_bytes": 4})
    with pytest.raises(EmbeddingConfigurationError, match="string|byte budget"):
        ipc.describe(content("text", value), spec)
    with pytest.raises(EmbeddingConfigurationError, match="string|byte budget"):
        ipc.validate_layout([{"kind": "text", "text": value}], 0, spec)


def test_sealed_arrays_preserve_noncontiguous_pixels_and_audio():
    import fcntl
    import mmap

    spec = ipc.input_spec({"input_modalities": ["image", "audio"], "sample_rate": 16000, "max_samples": 8})
    original = EmbeddingContent(
        content("image", np.zeros((8, 4, 3), np.uint8)[::2]).parts
        + content("audio", AudioClip(np.array([[0.5], [-0.5]], dtype=np.float64), 16000)).parts
    )
    parts, arrays, size = ipc.describe(original, spec)
    assert ipc.validate_layout(parts, size, spec) == size
    fd = ipc.sealed_buffer(arrays, size)
    try:
        assert fcntl.fcntl(fd, fcntl.F_GET_SEALS) == ipc.required_seals()
        with pytest.raises(OSError):
            os.ftruncate(fd, 1)
        with mmap.mmap(fd, size, access=mmap.ACCESS_COPY) as memory:
            restored = ipc.restore(parts, memory)
            np.testing.assert_equal(restored.parts[0].value, original.parts[0].value)
            np.testing.assert_equal(restored.parts[1].value.samples, original.parts[1].value.samples)
            assert restored.parts[1].value.sample_rate == 16000
            del restored
    finally:
        os.close(fd)


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
                path=str(root / "engine.sock"),
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
def test_real_processor_equal_for_direct_and_shared_frames(tmp_path, monkeypatch, family, make_real_video_processor):
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
    row = EmbeddingContent(
        content(
            "video",
            VideoClip(tuple(np.full((64, 64, 3), v, np.uint8) for v in (10, 20, 30)), (0.2, 0.6, 1.4), (2, 6, 14)),
        ).parts
        + content().parts
    )
    service = EmbeddingService(
        path=str(root / "engine.sock"),
        service_id="fixture-v1",
        family=family,
        model="fixture",
        dimensions=2,
        options=options,
    )

    async def run():
        await service.start()
        instance = client(root, family)
        try:
            await service.runtime._embed_content(row)
            np.testing.assert_equal(await instance.embed_multimodal([row]), [[3, 4]])
        finally:
            await instance.aclose()
            await service.aclose()

    asyncio.run(run())
    assert len(state.engines) == 1

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
    assert direct.keys() == shared.keys()
    for key in direct:
        np.testing.assert_equal(
            direct[key] if isinstance(direct[key], str) else np.asarray(direct[key]),
            shared[key] if isinstance(shared[key], str) else np.asarray(shared[key]),
        )


@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
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
    from vane.ai import MultimodalInput, TextInput

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
            service_path=str(root / "engine.sock"),
            service_id="fixture-v1",
            execution_backend=backend,
            where=vane.col("selected"),
            output_column="vector",
            normalize=False,
        )
        assert result.select("id", "vector").order("id").fetchall() == [(0, (3, 4)), (1, (3, 4)), (2, None)]
        texts = connection.from_arrow(
            pa.table({"id": list(range(5)), "text": ["healthy", "", " \t\n", "\u3000", None]})
        )
        result = texts.embed(
            TextInput(vane.col("text")),
            provider=family,
            model="fixture",
            dimensions=2,
            runtime="service",
            service_path=str(root / "engine.sock"),
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
        ]
    asyncio.run(
        until(lambda: state(root).get("calls") == 6 and state(root).get("bytes") == state(root).get("inflight") == 0)
    )
    assert state(root)["engines"] == 1


@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
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
                    service_path=str(root / "engine.sock"),
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

    (root / "memory_error").touch()
    for joint in (False, True):
        with pytest.raises(Exception, match="MemoryError|resources") as failure:
            query(joint)
        assert "private model" not in str(failure.value)
    (root / "memory_error").unlink()
    (root / "execution_error").touch()
    assert query() == [(None,)]
    (root / "execution_error").unlink()
    assert query() == [((3, 4),)]
    # Ray's existing fragment policy attempts a failed UDF four times. The
    # service client must not multiply those attempts by max_retries=3.
    failed_attempts = 4 if backend.startswith("ray_") else 1
    expected_calls = 2 * failed_attempts + 2
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
        path=str(root / "engine.sock"),
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
            path=str(root / "engine.sock"),
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
            assert service.credits.bytes == 96 and not handler.done()
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
    service = EmbeddingService(
        path=str(root / "engine.sock"),
        service_id="fixture-v1",
        family="sglang",
        model="fixture",
        dimensions=2,
        options={"gpus_per_actor": 0},
        max_inflight=1,
    )

    async def run():
        await service.start()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.setblocking(False)
        locked = False
        try:
            if phase != "release":
                await service.credits.condition.acquire()
                locked = True
            await asyncio.get_running_loop().sock_connect(sock, service.path)
            value = content() if kind == "text" else content("video", video())
            parts, _, size = ipc.describe(value, service.spec)
            await ipc.send(
                sock,
                {
                    "version": 1,
                    "service_id": "fixture-v1",
                    "family": "sglang",
                    "model": "fixture",
                    "dimensions": 2,
                    "request_id": "0" * 32,
                    "parts": parts,
                    "nbytes": size,
                },
            )
            if phase == "release":
                assert (await ipc.control(sock))["state"] == "reserved"
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
            loop = asyncio.get_running_loop()
            if phase == "handoff":
                service.credits.condition.release()
                locked = False
                # First cancel races with completed admission; the second
                # interrupts joining the cancelled disconnect watcher.
                loop.call_soon(handler.cancel)
                loop.call_soon(lambda: loop.call_soon(handler.cancel))
                await asyncio.wait_for(ipc.control(sock), 3)
            else:
                handler.cancel()
                await asyncio.wait_for(ipc.control(sock), 3)
                handler.cancel()
                await asyncio.sleep(0)
                handler.cancel()
                await asyncio.sleep(0)
                service.credits.condition.release()
                locked = False
            await until(lambda: not service._tasks)
            assert service.credits.bytes == service.credits.inflight == 0
            assert not state.calls and not service._connections
            instance = client(root, timeout=3)
            try:
                np.testing.assert_equal(await instance.embed_text(["healthy successor"]), [[3, 4]])
                assert len(state.calls) == 1
                assert service.credits.bytes == service.credits.inflight == 0
            finally:
                await instance.aclose()
        finally:
            if locked:
                service.credits.condition.release()
            sock.close()
            await service.aclose()

    asyncio.run(run())


def test_unsealed_fd_rejected_without_model_call(process):
    proc, root = process()

    async def run():
        await until(lambda: (root / "ready").exists() or proc.poll() is not None)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.setblocking(False)
        await asyncio.get_running_loop().sock_connect(sock, str(root / "engine.sock"))
        parts, _, size = ipc.describe(content("video", video()), ipc.input_spec({}))
        await ipc.send(
            sock,
            {
                "version": 1,
                "service_id": "fixture-v1",
                "family": "sglang",
                "model": "fixture",
                "dimensions": 2,
                "request_id": "0" * 32,
                "parts": parts,
                "nbytes": size,
            },
        )
        assert (await ipc.control(sock))["state"] == "reserved"
        fd = os.memfd_create("unsealed-fixture", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            os.ftruncate(fd, size)
            await ipc.send(sock, {"request_id": "0" * 32, "state": "published"}, fd)
            assert (await ipc.control(sock))["error"] == "configuration"
        finally:
            os.close(fd)
            sock.close()
        await until(lambda: state(root).get("inflight") == 0)
        assert state(root)["calls"] == 0

    asyncio.run(run())
