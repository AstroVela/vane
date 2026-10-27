# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared ONNX engine, owned calls, shutdown retries and real CPU Ray clients."""

import asyncio
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from tests.fast.test_onnx_provider import options as options
from tests.fast.test_onnx_provider import sdk as sdk
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._onnx_options import ONNX_MODEL
from vane.ai.embedding_service import EmbeddingService
from vane.ai.provider import load_provider

pytest.importorskip("zmq")
pytestmark = pytest.mark.filterwarnings(
    r"ignore:Proactor event loop does not implement add_reader family of methods required for zmq\.:RuntimeWarning"
)


def client_options(tmp_path, options, **updates):
    return {
        "runtime": "service",
        "service_path": str(tmp_path / "service/engine.json"),
        "service_id": "onnx-fixture",
        "model_sha256": options["model_sha256"],
        "max_length": options["max_length"],
    } | updates


def instance(tmp_path, options):
    path = tmp_path / "service"
    path.mkdir(mode=0o700)
    return EmbeddingService(
        path=str(path / "engine.json"),
        service_id="onnx-fixture",
        family="onnx",
        model=ONNX_MODEL,
        dimensions=128,
        options=options,
    )


@pytest.mark.parametrize("kind", ["text", "image", "audio", "video", "multimodal"])
def test_service_planning_retains_native_space_and_reserves_no_gpu(tmp_path, options, monkeypatch, kind):
    provider = load_provider("onnx")
    native = getattr(provider, f"get_{kind}_embedder")(dimensions=128, options=options | {"backend": "cuda"})
    for module in ("onnxruntime", "onnx", "transformers", "torch"):
        monkeypatch.setitem(sys.modules, module, None)
    client = getattr(provider, f"get_{kind}_embedder")(dimensions=128, options=client_options(tmp_path, options))
    assert native.get_output_spec() == client.get_output_spec()
    assert client.is_async() and client.get_udf_options().num_gpus == 0
    assert client.instantiate()._zmq_client is None
    assert not (tmp_path / "service").exists()


def test_concurrent_roles_use_one_external_engine_and_release_credits(tmp_path, options, sdk):
    async def scenario():
        service = instance(tmp_path, options)
        await service.start()
        provider = load_provider("onnx")
        document = provider.get_text_embedder(dimensions=128, options=client_options(tmp_path, options)).instantiate()
        query = provider.get_text_embedder(
            dimensions=128, options=client_options(tmp_path, options, input_type="query")
        ).instantiate()
        try:
            docs, queries = await asyncio.gather(document.embed_text(["same"] * 3), query.embed_text(["same"] * 2))
            assert not np.array_equal(docs[0], queries[0])
            assert len([event for event in sdk.events if event[0] == "load"]) == 3
            assert len({event[2] for event in sdk.events if event[0] in {"load", "run"}}) == 1
        finally:
            await document.aclose()
            await query.aclose()
            await service.aclose()
        assert service.credits.bytes == service.credits.inflight == 0
        assert sdk.active == 0 and service.backend.executor is None

    asyncio.run(scenario())


@pytest.mark.parametrize("update", [{"model_sha256": "b" * 64}, {"max_length": 129}, {"service_id": "wrong-service"}])
def test_identity_mismatch_is_rejected_before_inference(tmp_path, options, sdk, update):
    async def scenario():
        service = instance(tmp_path, options)
        await service.start()
        client = (
            load_provider("onnx")
            .get_text_embedder(dimensions=128, options=client_options(tmp_path, options, **update))
            .instantiate()
        )
        try:
            with pytest.raises(EmbeddingConfigurationError):
                await client.embed_text(["input"])
        finally:
            await client.aclose()
            await service.aclose()
        assert not any(event[0] == "run" for event in sdk.events)
        assert service.credits.bytes == service.credits.inflight == 0

    asyncio.run(scenario())


def test_shutdown_holds_committed_input_until_inference_finishes(tmp_path, options, sdk):
    entered, release = threading.Event(), threading.Event()
    sdk.entered, sdk.release = entered, release

    async def scenario():
        service = instance(tmp_path, options)
        await service.start()
        client = (
            load_provider("onnx")
            .get_text_embedder(dimensions=128, options=client_options(tmp_path, options))
            .instantiate()
        )
        request = asyncio.create_task(client.embed_text(["held"]))
        closing = None
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            closing = asyncio.create_task(service.aclose())
            await asyncio.sleep(0.03)
            assert not closing.done() and sdk.active == 3
            assert service.credits.inflight == 1
            closing.cancel()
            await asyncio.sleep(0.03)
            assert not closing.done()
        finally:
            release.set()
            await asyncio.gather(request, return_exceptions=True)
            if closing is not None:
                await closing
            await client.aclose()
            await service.aclose()
        assert sdk.active == 0 and service.credits.bytes == service.credits.inflight == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [TimeoutError, asyncio.CancelledError])
def test_failed_close_retries_on_the_same_owned_thread(tmp_path, options, sdk, monkeypatch, failure):
    from vane.ai.providers._onnx_model import ONNXEmbedder

    original = ONNXEmbedder.close
    attempts = []

    def flaky(self):
        attempts.append(threading.get_ident())
        if len(attempts) == 1:
            raise failure()
        original(self)

    monkeypatch.setattr(ONNXEmbedder, "close", flaky)

    async def scenario():
        service = instance(tmp_path, options)
        await service.start()
        backend = service.backend
        with pytest.raises(failure):
            await service.aclose()
        assert service.backend is backend and backend.executor is not None and sdk.active == 3
        await asyncio.gather(service.aclose(), service.aclose())
        assert sdk.active == 0 and backend.executor is None

    asyncio.run(scenario())
    assert len(attempts) == 2
    assert attempts[0] == attempts[1] == next(event[2] for event in sdk.events if event[0] == "load")


def test_cancelled_initialization_is_joined_and_closed_before_exit(tmp_path, options, sdk, monkeypatch):
    from vane.ai.providers.onnx import ONNXMultimodalDescriptor

    entered, release = threading.Event(), threading.Event()
    original = ONNXMultimodalDescriptor.instantiate

    def held(self):
        entered.set()
        assert release.wait(10)
        return original(self)

    monkeypatch.setattr(ONNXMultimodalDescriptor, "instantiate", held)

    async def scenario():
        service = instance(tmp_path, options)
        starting = asyncio.create_task(service.start())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            starting.cancel()
            await asyncio.sleep(0.03)
            assert not starting.done()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await starting
            await service.aclose()
        assert sdk.active == 0 and service.backend.executor is None
        assert not Path(service.path).exists()

    asyncio.run(scenario())


@pytest.mark.real_ray
@pytest.mark.ray_cluster_owner
@pytest.mark.parametrize("backend", ["ray_task", "ray_actor"])
def test_real_ray_clients_share_one_onnx_service(tmp_path, options, monkeypatch, backend):
    import pyarrow as pa
    import ray

    import vane
    from vane.ai import TextInput, embed

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    root = tmp_path / "service"
    root.mkdir(mode=0o700)
    config = {
        "path": str(root / "engine.json"),
        "service_id": "onnx-fixture",
        "family": "onnx",
        "model": ONNX_MODEL,
        "dimensions": 128,
        "options": options,
    }
    (root / "config.json").write_text(json.dumps(config))
    log = (root / "log.txt").open("w")
    helper = Path(__file__).parents[1] / "ai/onnx_service_process.py"
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
                    provider="onnx",
                    dimensions=128,
                    execution_backend=backend,
                    batch_size=2,
                    **client_options(tmp_path, options),
                )
                .to_arrow_table()
                .to_pylist()
            )
        assert len(result) == 3
        values = {row["text"]: row["embedding"] for row in result}
        assert values.keys() == {"document", None, "query"} and values[None] is None
        for text in ("document", "query"):
            assert len(values[text]) == 128
            np.testing.assert_allclose(np.linalg.norm(values[text]), 1, atol=1e-5)
    finally:
        vane.teardown_runner()
        ray.shutdown()
        (root / "stop").touch()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            pytest.fail("ONNX fixture process did not drain")
        log.close()
    assert process.returncode == 0, (root / "log.txt").read_text()
    events = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["loaded", "closed"]
    assert len({event["thread"] for event in events}) == 1
