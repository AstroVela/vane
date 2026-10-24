# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""LiteRT service identities, roles, sole-thread ownership and committed drain."""

import asyncio
import json
import pickle
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pytest

from tests.fast.test_litert_provider import options as options
from tests.fast.test_litert_provider import sdk as sdk
from vane.ai import AudioClip, EmbeddingContent, EmbeddingPart, EmbeddingPartSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._litert_options import LITERT_DEFAULT_MODEL
from vane.ai.embedding_service import EmbeddingService
from vane.ai.functions import _prepare_embed_call
from vane.ai.provider import _ProviderResultError, load_provider

pytest.importorskip("zmq")
pytestmark = pytest.mark.filterwarnings(
    r"ignore:Proactor event loop does not implement add_reader family of methods required for zmq\.:RuntimeWarning"
)


def client_options(tmp_path, options, **updates):
    return {
        "runtime": "service",
        "service_path": str(tmp_path / "service" / "engine.json"),
        "service_id": "litert-fixture",
        "model_sha256": options["model_sha256"],
        "max_length": 128,
    } | updates


def descriptor(tmp_path, options, kind="text", **updates):
    return getattr(load_provider("litert"), f"get_{kind}_embedder")(
        LITERT_DEFAULT_MODEL, 128, options=client_options(tmp_path, options, **updates)
    )


def service(tmp_path, options):
    root = tmp_path / "service"
    root.mkdir(mode=0o700)
    return EmbeddingService(
        path=str(root / "engine.json"),
        service_id="litert-fixture",
        family="litert",
        model=LITERT_DEFAULT_MODEL,
        dimensions=128,
        options=options | {"max_length": 128},
    )


@asynccontextmanager
async def running(tmp_path, options):
    instance = service(tmp_path, options)
    await instance.start()
    try:
        yield instance
    finally:
        await instance.aclose()


@pytest.mark.parametrize("kind", ["text", "image", "audio", "multimodal"])
@pytest.mark.parametrize("dimensions", [128, 256, 512, 768])
def test_service_planning_keeps_native_space_without_sdk_or_files(tmp_path, options, monkeypatch, kind, dimensions):
    monkeypatch.setitem(sys.modules, "litert_lm", None)
    provider = load_provider("litert")
    remote = getattr(provider, f"get_{kind}_embedder")(
        LITERT_DEFAULT_MODEL, dimensions, options=client_options(tmp_path, options)
    )
    native = getattr(provider, f"get_{kind}_embedder")(LITERT_DEFAULT_MODEL, dimensions, options=options)
    clone = pickle.loads(pickle.dumps(remote))
    assert clone.get_output_spec() == native.get_output_spec()
    assert clone.is_async() and clone.get_udf_options().num_gpus == 0
    assert clone.instantiate()._zmq_client is None
    assert "model_path" not in clone.get_options() and "backend" not in clone.get_options()
    assert not (tmp_path / "service").exists()
    if kind == "audio":
        assert clone.get_input_spec() == native.get_input_spec()
    if kind == "multimodal":
        assert clone.get_input_spec() == native.get_input_spec()


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor", "ray_task", "ray_actor"])
def test_service_allows_cpu_client_executors(tmp_path, options, backend):
    prepared = _prepare_embed_call(
        "litert",
        LITERT_DEFAULT_MODEL,
        128,
        "raise",
        client_options(tmp_path, options, execution_backend=backend),
        relation=True,
    )
    assert prepared[2].num_gpus == 0 and prepared[-1] == backend


@pytest.mark.parametrize(
    "invalid",
    [
        {"backend": "gpu"},
        {"model_path": "/private/model.litertlm"},
        {"num_threads": 4},
        {"cache_dir": "/private/cache"},
        {"input_modalities": ["video"]},
        {"sample_rate": 48000},
        {"vision_tokens_per_image": 71},
        {"max_length": 129},
        {"model_sha256": "bad"},
        {"service_id": "bad identity"},
        {"service_path": "relative.json"},
        {"timeout": False},
        {"max_concurrency_per_actor": 1025},
    ],
)
def test_service_rejects_engine_overrides_and_invalid_configuration(tmp_path, options, invalid):
    with pytest.raises((TypeError, ValueError)):
        descriptor(tmp_path, options, **invalid)


def test_service_uses_one_thread_and_model_for_query_documents_and_media(tmp_path, options, sdk, monkeypatch):
    events = []
    original = sdk.EmbeddingEngine

    class TracedEngine(original):
        def __init__(self, *args, **kwargs):
            events.append(("load", threading.get_ident()))
            super().__init__(*args, **kwargs)

        def compute_embedding_batch(self, *args, **kwargs):
            events.append(("infer", threading.get_ident()))
            return super().compute_embedding_batch(*args, **kwargs)

        def close(self):
            events.append(("close", threading.get_ident()))
            super().close()

    monkeypatch.setattr(sdk, "EmbeddingEngine", TracedEngine)

    async def scenario():
        async with running(tmp_path, options) as instance:
            document = descriptor(tmp_path, options).instantiate()
            query = descriptor(tmp_path, options, input_type="query").instantiate()
            audio = descriptor(tmp_path, options, kind="audio").instantiate()
            mixed = descriptor(tmp_path, options, kind="multimodal").instantiate()
            try:
                docs, queries = await asyncio.gather(document.embed_text(["same"] * 3), query.embed_text(["same"] * 2))
                assert not np.array_equal(docs[0], queries[0])
                await audio.embed_audio([AudioClip(np.zeros((160, 1), dtype=np.float32), 16000)])
                await mixed.embed_multimodal(
                    [
                        EmbeddingContent(
                            (
                                EmbeddingPart(EmbeddingPartSpec("text"), "sound"),
                                EmbeddingPart(
                                    EmbeddingPartSpec("audio"), AudioClip(np.zeros((160, 1), dtype=np.float32), 16000)
                                ),
                            )
                        )
                    ]
                )
                inputs = [event[1][0] for event in sdk.events if event[0] == "infer"]
                assert sum(row[0] == "title: none | text: same" for row in inputs) == 3
                assert sum(row[0] == "task: search result | query: same" for row in inputs) == 2
                assert instance.metrics["completed"] == 7
                assert instance.credits.bytes == instance.credits.inflight == 0
            finally:
                for client in (document, query, audio, mixed):
                    await client.aclose()

    asyncio.run(scenario())
    assert [event[0] for event in events].count("load") == [event[0] for event in events].count("close") == 1
    assert len({thread for _, thread in events}) == 1 and events[0][1] != threading.get_ident()
    assert not any(thread.name.startswith("vane-litert-service") for thread in threading.enumerate())


@pytest.mark.parametrize(
    "mismatch",
    [
        {"model_sha256": "b" * 64},
        {"max_length": 256},
        {"vision_tokens_per_image": 140},
    ],
)
def test_service_checks_file_and_encoding_identity_before_reserving(tmp_path, options, sdk, mismatch):
    async def scenario():
        async with running(tmp_path, options) as instance:
            client = descriptor(tmp_path, options, **mismatch).instantiate()
            try:
                with pytest.raises(EmbeddingConfigurationError):
                    await client.embed_text(["query"])
                assert instance.metrics["accepted"] == 0 and instance.credits.bytes == 0
                assert not any(event[0] == "infer" for event in sdk.events)
            finally:
                await client.aclose()

    asyncio.run(scenario())


def test_invalid_model_output_is_fatal_under_ignore(tmp_path, options, sdk):
    sdk.responses = [[1.0] * 128]

    async def scenario():
        async with running(tmp_path, options):
            client = descriptor(tmp_path, options).instantiate()
            client._request_on_error = "ignore"
            try:
                with pytest.raises(_ProviderResultError):
                    await client.embed_text(["query"])
            finally:
                await client.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("values", [[True] * 128, [0.0] * 128, [1.0] * 127])
def test_invalid_remote_vector_is_fatal_under_ignore(tmp_path, options, sdk, monkeypatch, values):
    async def invalid(*args, **kwargs):
        return {"vector": values}

    async def scenario():
        async with running(tmp_path, options) as instance:
            monkeypatch.setattr(instance, "_infer", invalid)
            client = descriptor(tmp_path, options).instantiate()
            client._request_on_error = "ignore"
            try:
                with pytest.raises(_ProviderResultError):
                    await client.embed_text(["query"])
            finally:
                await client.aclose()

    asyncio.run(scenario())


def test_query_role_cannot_be_used_for_media(tmp_path, options, sdk, monkeypatch):
    async def scenario():
        async with running(tmp_path, options) as instance:
            client = descriptor(tmp_path, options, kind="image").instantiate()
            reservation = client._reservation

            def changed(*args):
                message = reservation(*args)
                message["encoding"]["input_type"] = "query"
                return message

            monkeypatch.setattr(client, "_reservation", changed)
            try:
                with pytest.raises(EmbeddingConfigurationError):
                    await client.embed_image([np.zeros((4, 4, 3), dtype=np.uint8)])
                assert instance.metrics["accepted"] == 0
                assert not any(event[0] == "infer" for event in sdk.events)
            finally:
                await client.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_gpu_initialization_failure_does_not_try_cpu(tmp_path, options, sdk, failure):
    sdk.failure = failure("authored GPU initialization failure")

    async def scenario():
        instance = service(tmp_path, options | {"backend": "gpu"})
        with pytest.raises(failure, match="authored GPU"):
            await instance.start()
        assert instance.backend.runtime is None and not (tmp_path / "service/engine.json").exists()
        await instance.aclose()

    asyncio.run(scenario())
    loads = [event for event in sdk.events if event[0] == "load"]
    assert len(loads) == 1 and loads[0][2]["backend"].kind == "gpu"
    assert not any(thread.name.startswith("vane-litert-service") for thread in threading.enumerate())


def test_close_drains_committed_cancelled_inference(tmp_path, options, sdk, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    compute = sdk.EmbeddingEngine.compute_embedding_batch

    def held(engine, *args, **kwargs):
        entered.set()
        assert release.wait(8)
        return compute(engine, *args, **kwargs)

    monkeypatch.setattr(sdk.EmbeddingEngine, "compute_embedding_batch", held)

    async def scenario():
        instance = service(tmp_path, options)
        await instance.start()
        client = descriptor(tmp_path, options).instantiate()
        request = asyncio.create_task(client.embed_text(["held"]))
        closing = None
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            closing = asyncio.create_task(instance.aclose())
            await asyncio.sleep(0.03)
            closing.cancel()
            await asyncio.sleep(0.03)
            assert not closing.done() and instance.credits.inflight == 1
            assert not any(event[0] == "close" for event in sdk.events)
        finally:
            release.set()
            if closing is not None:
                await closing
            await instance.aclose()
            await client.aclose()
        assert instance.credits.bytes == instance.credits.inflight == 0

    asyncio.run(scenario())
    assert len([event for event in sdk.events if event[0] == "close"]) == 1


@pytest.mark.parametrize("failure", [TimeoutError, asyncio.CancelledError])
def test_failed_sdk_close_retries_on_the_original_live_thread(tmp_path, options, sdk, monkeypatch, failure):
    events = []
    entered, release = threading.Event(), threading.Event()
    original = sdk.EmbeddingEngine

    class Engine(original):
        def __init__(self, *args, **kwargs):
            events.append(("load", threading.get_ident()))
            super().__init__(*args, **kwargs)
            self.attempts = 0

        def close(self):
            self.attempts += 1
            events.append(("close", threading.get_ident()))
            if self.attempts == 1:
                raise failure("authored SDK shutdown failure")
            entered.set()
            assert release.wait(8)
            super().close()

    monkeypatch.setattr(sdk, "EmbeddingEngine", Engine)

    async def scenario():
        instance = service(tmp_path, options)
        await instance.start()
        runtime = instance.backend.runtime
        pending = []
        try:
            with pytest.raises(failure):
                await instance.aclose()
            assert instance.backend.runtime is runtime
            assert any(thread.name.startswith("vane-litert-service") for thread in threading.enumerate())
            with pytest.raises(RuntimeError, match="already started"):
                await instance.start()
            pending = [asyncio.create_task(instance.aclose()) for _ in range(3)]
            assert await asyncio.to_thread(entered.wait, 5)
            pending[0].cancel()
            await asyncio.sleep(0.02)
            assert all(not task.done() for task in pending)
            release.set()
            await asyncio.gather(*pending)
            assert instance.backend.runtime is None
            await instance.aclose()
            assert [event[0] for event in events] == ["load", "close", "close"]
        finally:
            release.set()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            # The native owner is retained here even if the service loses it.
            native = runtime.initialization.result()
            if not native._closed:
                await asyncio.to_thread(native.close)
            runtime.executor.shutdown(wait=True)

    asyncio.run(scenario())
    assert len({thread for _, thread in events}) == 1
    assert not any(thread.name.startswith("vane-litert-service") for thread in threading.enumerate())


def test_cancelled_startup_joins_the_single_sdk_thread(tmp_path, options, sdk, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = sdk.EmbeddingEngine

    class HeldEngine(original):
        def __init__(self, *args, **kwargs):
            entered.set()
            assert release.wait(8)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(sdk, "EmbeddingEngine", HeldEngine)

    async def scenario():
        instance = service(tmp_path, options)
        starting = asyncio.create_task(instance.start())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            starting.cancel()
            await asyncio.sleep(0.03)
            assert not starting.done()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await starting
            await instance.aclose()
        assert instance.backend.runtime is None and not (tmp_path / "service/engine.json").exists()

    asyncio.run(scenario())
    assert len([event for event in sdk.events if event[0] == "close"]) == 1
    assert not any(thread.name.startswith("vane-litert-service") for thread in threading.enumerate())


def test_close_during_initialization_cannot_publish_an_endpoint(tmp_path, options, sdk, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = sdk.EmbeddingEngine

    class HeldEngine(original):
        def __init__(self, *args, **kwargs):
            entered.set()
            assert release.wait(8)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(sdk, "EmbeddingEngine", HeldEngine)

    async def scenario():
        instance = service(tmp_path, options)
        starting = asyncio.create_task(instance.start())
        closing = None
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            closing = asyncio.create_task(instance.aclose())
            await asyncio.sleep(0.03)
            assert not closing.done()
        finally:
            release.set()
            with pytest.raises(RuntimeError, match="closed during startup"):
                await starting
            if closing is not None:
                await closing
        assert instance.backend.runtime is None and not (tmp_path / "service/engine.json").exists()

    asyncio.run(scenario())
    assert len([event for event in sdk.events if event[0] == "close"]) == 1


@pytest.mark.real_ray
@pytest.mark.ray_cluster_owner
@pytest.mark.parametrize("backend", ["ray_task", "ray_actor"])
def test_public_ray_clients_share_the_external_litert_engine(tmp_path, options, monkeypatch, backend):
    import pyarrow as pa
    import ray

    import vane
    from vane.ai import TextInput, embed

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("VANE_RUNNER", raising=False)

    root = tmp_path / "service"
    root.mkdir(mode=0o700)
    (root / "config.json").write_text(
        json.dumps(
            {
                "path": str(root / "engine.json"),
                "service_id": "litert-fixture",
                "family": "litert",
                "model": LITERT_DEFAULT_MODEL,
                "dimensions": 128,
                "options": options | {"max_length": 128},
            }
        )
    )
    log = (root / "log.txt").open("w")
    helper = Path(__file__).parents[1] / "ai/litert_service_process.py"
    process = subprocess.Popen([sys.executable, str(helper), str(root), "--fake"], stdout=log, stderr=log)
    try:
        ray.init(address="local", num_cpus=3, include_dashboard=False)
        deadline = time.monotonic() + 30
        while not (root / "ready").exists():
            if process.poll() is not None or time.monotonic() > deadline:
                pytest.fail((root / "log.txt").read_text())
            time.sleep(0.02)
        with vane.connect() as connection:
            source = connection.from_arrow(pa.table({"text": ["document", None, "query"]}))
            result = (
                embed(
                    source,
                    TextInput(vane.col("text")),
                    provider="litert",
                    model=LITERT_DEFAULT_MODEL,
                    dimensions=128,
                    execution_backend=backend,
                    batch_size=2,
                    max_retries=0,
                    **client_options(tmp_path, options),
                )
                .to_arrow_table()
                .to_pylist()
            )
        # Ray batches may complete in any order; match each output to its input.
        assert len(result) == 3
        embeddings = {row["text"]: row["embedding"] for row in result}
        assert embeddings.keys() == {"document", None, "query"}
        assert embeddings[None] is None
        for text in ("document", "query"):
            assert len(embeddings[text]) == 128
            np.testing.assert_allclose(np.linalg.norm(embeddings[text]), 1, atol=1e-5)
    finally:
        vane.teardown_runner()
        ray.shutdown()
        (root / "stop").touch()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            pytest.fail("LiteRT fixture process did not drain")
        log.close()
    assert process.returncode == 0, (root / "log.txt").read_text()
    events = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
    assert sum(event["event"] == "loaded" for event in events) == 1
    assert sum(event["event"] == "closed" for event in events) == 1
    assert len({event["thread"] for event in events}) == 1
