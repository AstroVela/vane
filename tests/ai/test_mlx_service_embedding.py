# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Actual Metal/IPC parity and model ownership across independent OS clients."""

import asyncio
import gc
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from mlx_metal_reference import audio_inputs, infer, joint_inputs, parity, sdk_model, vectors, video_inputs
from mlx_service_fixture import ClientProcess, ServiceProcess, save_case

pytest_plugins = ("mlx_metal_reference",)
pytestmark = [pytest.mark.metal, pytest.mark.external_service, pytest.mark.usefixtures("prepared_metal")]


@pytest.fixture(scope="module")
def cases():
    from tests.ai.test_mlx_multimodal_embedding import cases as joint_cases
    from tests.ai.test_mlx_multimodal_embedding import content
    from tests.ai.test_mlx_text_embedding import DOCUMENTS, QUERIES
    from vane.ai import AudioClip, VideoClip

    joint = joint_cases.__wrapped__()
    result = [("text_document", content(("text", value))) for value in DOCUMENTS]
    result += [("text_query", content(("text", value))) for value in QUERIES]
    result += [("text_query", content(("text", DOCUMENTS[0]))), ("text_document", content(("text", " \t")))]
    for image in (joint[1].parts[0].value, joint[8].parts[1].value, joint[8].parts[4].value):
        result.append(("image", content(("image", image))))
    for clip in (joint[2].parts[0].value, joint[8].parts[3].value):
        result.append(("video", content(("video", clip))))
    frame = joint[2].parts[0].value.frames[0]
    result.append(("video", content(("video", VideoClip((frame,) * 12, (1.25,) * 12, tuple(range(7, 67, 5)))))))
    for clip in (
        AudioClip(np.zeros((161, 1), np.float32), 16000),
        joint[3].parts[0].value,
        joint[8].parts[0].value,
        AudioClip(np.zeros((480000, 1), np.float32), 16000),
    ):
        result.append(("audio", content(("audio", clip))))
    result += [("multimodal", row) for row in joint]
    return result


@pytest.fixture(scope="module")
def references(cases, tmp_path_factory, record_testsuite_property):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from vane.ai.provider import load_provider

    root = tmp_path_factory.mktemp("service-references")
    sdk_vectors = []
    with sdk_model() as sdk:
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_download(
                "mlx-community/embeddinggemma-2-bf16",
                revision="1a4ffddb7905d3f63486748deabe091a01fb6201",
                local_files_only=True,
            ),
            local_files_only=True,
            trust_remote_code=False,
        )
        for encoding, row in cases:
            value = row.parts[0].value
            if encoding in {"text_document", "text_query"}:
                prefix = "task: search result | query: " if encoding == "text_query" else "title: none | text: "
                encoded = tokenizer([prefix + value], return_tensors="np", padding=True, truncation=False)
                other = sdk.processor.tokenizer([prefix + value], return_tensors="np", padding=True, truncation=False)
                for key in ("input_ids", "attention_mask"):
                    np.testing.assert_array_equal(encoded[key], other[key])
                encoded = {key: encoded[key] for key in ("input_ids", "attention_mask")}
            elif encoding == "image":
                encoded = sdk.processor(
                    images=[[value.transpose(2, 0, 1)]],
                    images_kwargs={"input_data_format": "channels_first"},
                    return_tensors="np",
                    padding=True,
                    truncation=False,
                )
            elif encoding == "audio":
                encoded = audio_inputs(sdk.processor, [value.samples[:, 0]])
            elif encoding == "video":
                encoded = video_inputs(sdk.processor, [np.stack(value.frames)])
            else:
                encoded = joint_inputs(sdk.processor, [row])
            sdk_vectors.append(infer(sdk.model, encoded)[0])
        del tokenizer, encoded, other
    gc.collect()
    native = np.empty((len(cases), 768), np.float32)
    for encoding in ("text_document", "text_query", "image", "audio", "video", "multimodal"):
        kind = "text" if encoding.startswith("text_") else encoding
        options = {"input_type": "query" if encoding == "text_query" else "document"} if kind == "text" else {}
        encoder = getattr(load_provider("mlx"), "get_" + kind + "_embedder")(options=options).instantiate()
        try:
            for index, (actual, row) in enumerate(cases):
                if actual == encoding:
                    values = [row] if kind == "multimodal" else [row.parts[0].value]
                    native[index] = getattr(encoder, "embed_" + kind)(values)[0]
        finally:
            encoder.close()
        del encoder
        gc.collect()
    sdk_values = vectors(np.asarray(sdk_vectors, np.float32), len(cases))
    vectors(native, len(cases))
    metrics = parity(native, sdk_values)
    joint = [i for i, (encoding, _) in enumerate(cases) if encoding == "multimodal"]
    assert np.abs(native[joint] - sdk_values[joint]).max() <= 0.01
    np.savez(root / "references.npz", sdk=sdk_values, native=native)
    paths = []
    for i, (encoding, row) in enumerate(cases):
        path = root / f"case-{i}.npz"
        save_case(path, row, encoding)
        paths.append(str(path))
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                created_for_service_acceptance=True,
                frozen_before_implementation=False,
                sdk_native_parity=metrics,
                cases=len(cases),
            ),
            indent=2,
        )
        + "\n"
    )
    record_testsuite_property("references", str(root))
    return root, paths, sdk_values, native


def assert_closed(service):
    events = service.events()
    assert len([row for row in events if row["event"] == "load"]) == 1
    assert len([row for row in events if row["event"] == "processor_loaded"]) == 1
    assert len([row for row in events if row["event"] == "close"]) == 1
    work = [
        row for row in events if row["event"] in {"load", "processor_loaded", "forward_start", "forward_end", "close"}
    ]
    assert len({row["thread_id"] for row in work}) == 1
    assert len({row["model_instance_id"] for row in work}) == 1
    assert max(row.get("active_forward_count", 0) for row in events) == 1
    closed = [row for row in events if row["event"] == "closed"][-1]
    assert closed["bytes"] == closed["inflight"] == 0 and closed["buffers"] == [] and closed["executor_closed"]
    return events


def test_production_cli_uses_real_metal_with_a_no_sdk_public_client(tmp_path):
    service = ServiceProcess(tmp_path / "production", traced=False)
    client = None
    try:
        service.ready()
        client = ClientProcess(tmp_path / "client", service.options)
        output = tmp_path / "cli.npy"
        client.ask(action="public_text", texts=["A document about Mars."], role="document", output=str(output))
        vectors(np.load(output), 1)
    finally:
        if client is not None:
            client.close()
        service.stop()


def test_two_independent_public_clients_share_one_model_and_match_sdk_native(
    cases, references, tmp_path, record_property
):
    root, paths, sdk, native = references
    service = ServiceProcess(tmp_path / "shared")
    clients = []
    try:
        service.ready()
        clients = [ClientProcess(tmp_path / f"client-{i}", service.options) for i in range(2)]
        assert len({service.process.pid, *(client.process.pid for client in clients)}) == 3
        with ThreadPoolExecutor(max_workers=2) as pool:
            operations = [
                pool.submit(
                    clients[i].ask,
                    action="public_text",
                    texts=[cases[j][1].parts[0].value for j in range(i * 3, i * 3 + 3)],
                    role="document" if i == 0 else "query",
                    output=str(tmp_path / f"public-{i}.npy"),
                )
                for i in range(2)
            ]
            for future in operations:
                future.result(timeout=180)
        for i in range(2):
            values = vectors(np.load(tmp_path / f"public-{i}.npy"), 3)
            parity(values, sdk[i * 3 : i * 3 + 3], record_property, f"public-{i}")
        scores = np.load(tmp_path / "public-1.npy") @ np.load(tmp_path / "public-0.npy").T
        assert scores.argmax(axis=1).tolist() == [0, 1, 2]
        with ThreadPoolExecutor(max_workers=2) as pool:
            operations = [
                pool.submit(
                    clients[i].ask, action="direct", paths=paths[i::2], output=str(tmp_path / f"direct-{i}.npy")
                )
                for i in range(2)
            ]
            for future in operations:
                future.result(timeout=180)
        actual = np.empty_like(sdk)
        for i in range(2):
            actual[i::2] = vectors(np.load(tmp_path / f"direct-{i}.npy"), len(paths[i::2]))
        parity(actual, sdk, record_property, "service-sdk")
        parity(actual, native, record_property, "service-native")
        joint = [i for i, (encoding, _) in enumerate(cases) if encoding == "multimodal"]
        assert np.abs(actual[joint] - sdk[joint]).max() <= 0.01
        clients[0].close()
        clients[1].ask(
            action="public_text",
            texts=[cases[3][1].parts[0].value],
            role="query",
            output=str(tmp_path / "survivor.npy"),
        )
        parity(vectors(np.load(tmp_path / "survivor.npy"), 1), sdk[3:4], record_property, "surviving-client")
        assert service.process.poll() is None
        assert {row["encoding"] for row in service.events() if row["event"] == "forward_start"} == {
            "text_document",
            "text_query",
            "image",
            "audio",
            "video",
            "multimodal",
        }
        record_property("server_pid", service.process.pid)
        record_property("client_pids", json.dumps([client.process.pid for client in clients]))
    finally:
        for client in clients:
            client.close()
        service.stop()
    events = assert_closed(service)
    record_property("trace", str(service.trace))
    record_property("load_count", len([row for row in events if row["event"] == "load"]))
    record_property("forward_count", len([row for row in events if row["event"] == "forward_start"]))


def test_real_cancel_and_sigterm_keep_inputs_until_metal_and_owner_close(tmp_path, record_property):
    from vane.ai._embedding_inputs import EmbeddingConfigurationError
    from vane.ai.provider import load_provider

    service = ServiceProcess(tmp_path / "cancel")
    try:
        service.ready()
        (service.root / "gate").touch()

        async def run():
            client = load_provider("mlx").get_text_embedder(options=service.options).instantiate()
            wrong = load_provider("mlx").get_text_embedder(options=service.options | {"max_length": 32}).instantiate()
            pending = asyncio.create_task(client.embed_text(["input retained until actual Metal completion"]))
            try:
                await asyncio.to_thread(service.wait_event, "blocked")
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                with pytest.raises(EmbeddingConfigurationError):
                    await wrong.embed_text(["control still responds"])
                assert service.state()["inflight"] == 1 and service.state()["buffers"]
                service.process.terminate()
                await asyncio.sleep(0.1)
                assert service.process.poll() is None
                assert not any(row["event"] == "close" for row in service.events())
                (service.root / "release").touch()
                await asyncio.to_thread(service.process.wait, timeout=60)
            finally:
                (service.root / "release").touch()
                await client.aclose()
                await wrong.aclose()
                await asyncio.gather(pending, return_exceptions=True)

        asyncio.run(run())
    finally:
        service.stop()
    events = assert_closed(service)
    assert [row["event"] for row in events].index("forward_end") < [row["event"] for row in events].index("close")
    record_property("trace", str(service.trace))


def test_real_service_injected_metal_oom_is_fatal_and_explicit_restart_recovers(tmp_path, record_property):
    from vane.ai.provider import load_provider

    service = ServiceProcess(tmp_path / "oom")
    try:
        service.ready()
        (service.root / "oom").touch()

        async def run():
            client = load_provider("mlx").get_text_embedder(options=service.options).instantiate()
            client.configure_execution(max_retries=9, on_error="ignore", validate=lambda value: value)
            try:
                for _ in range(2):
                    with pytest.raises(MemoryError, match="restart"):
                        await client.embed_text(["injected allocation boundary"])
            finally:
                await client.aclose()

        asyncio.run(run())
        assert len([row for row in service.events() if row["event"] == "forward_start"]) == 1
    finally:
        service.stop()
    assert_closed(service)
    record_property("oom_trace", str(service.trace))
    (service.root / "oom").unlink()
    restarted = ServiceProcess(service.root)
    try:
        restarted.ready()

        async def recovered():
            client = load_provider("mlx").get_text_embedder(options=restarted.options).instantiate()
            try:
                vectors(await client.embed_text(["explicitly restarted service"]), 1)
            finally:
                await client.aclose()

        asyncio.run(recovered())
    finally:
        restarted.stop()
    assert_closed(restarted)
    record_property("restart_trace", str(restarted.trace))
