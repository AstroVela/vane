# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real SDK/HTTP contracts, independent of hosted models or GPUs."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import pickle
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pyarrow as pa
import pytest

import vane
from vane.ai import embed, embed_video, prompt
from vane.ai._media import PromptMedia
from vane.ai._video_embedding import VideoClip
from vane.ai.functions import _prepare_embed_call, _prepare_prompt_call
from vane.ai.protocols import NativeInferencePlan, PrompterDescriptor
from vane.ai.provider import load_provider
from vane.ai.providers.sglang import SGLangProvider
from vane.ai.providers.vllm import VLLMProvider
from vane.execution.udf_file_contract import FileUDFContract


@pytest.fixture
def endpoint():
    requests = []
    response_override = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            requests.append((self.path, dict(self.headers), body))
            if self.path.endswith("/embeddings"):
                response = {
                    "object": "list",
                    "model": body["model"],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    "data": [{"index": 0, "object": "embedding", "embedding": [3.0, 4.0]}],
                }
            else:
                response = {
                    "id": "fixture",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": '{"answer":"ok"}'},
                        }
                    ],
                }
            if response_override:
                response = response_override[0]
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (
            {"transport": "http", "base_url": f"http://127.0.0.1:{server.server_port}/v1"},
            requests,
            response_override,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def clip():
    return VideoClip(
        frames=(np.full((8, 8, 3), 20, dtype=np.uint8), np.full((8, 8, 3), 90, dtype=np.uint8)),
        frame_times=(0.125, 1.375),
        frame_indices=(4, 18),
    )


@pytest.mark.parametrize("family", ["vllm", "sglang"])
def test_planning_is_explicit_and_keeps_native_execution(family):
    provider = load_provider(family)
    assert isinstance(provider.get_prompter(), NativeInferencePlan)
    options = {"transport": "http", "base_url": "http://inference.invalid/v1", "media_mime_types": ["video/mp4"]}
    desc, udf, backend, _ = _prepare_prompt_call(
        provider,
        "served-vlm",
        None,
        None,
        False,
        "raise",
        {**options, "execution_backend": "subprocess_task", "max_retries": 1},
        relation=True,
    )
    assert isinstance(desc, PrompterDescriptor)
    assert desc.supported_media_mime_types() == {"video/mp4"}
    assert desc.get_options()["transport"] == "http"
    assert udf.num_gpus == 0 and udf.max_retries == 1 and backend == "subprocess_task"
    for invalid in (
        {"transport": "http"},
        {**options, "engine_args": {}},
        {**options, "transport": "magic"},
        {**options, "media_mime_types": ["audio/wav"]},
        {**options, "api_key": "private"},
    ):
        with pytest.raises((TypeError, ValueError)):
            provider.get_prompter(model="served-vlm", options=invalid)


@pytest.mark.parametrize("family", ["vllm", "sglang"])
def test_pinned_credentials_endpoint_and_media_after_serialization(endpoint, monkeypatch, family):
    options, requests, _ = endpoint
    monkeypatch.setenv(family.upper() + "_API_KEY", "server-key")
    provider = VLLMProvider("custom") if family == "vllm" else SGLangProvider("custom")
    descriptor = provider.get_prompter(model="served-vlm", options={**options, "media_mime_types": ["video/mp4"]})
    descriptor = pickle.loads(pickle.dumps(descriptor))
    monkeypatch.setenv(family.upper() + "_API_KEY", "worker-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://wrong.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-key")
    monkeypatch.setenv("OPENAI_ORG_ID", "wrong-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "wrong-project")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer wrong-key")

    async def run():
        runtime = descriptor.instantiate()
        try:
            assert (
                await runtime.prompt(("Find the event", PromptMedia(b"video-body", "video/mp4"))) == '{"answer":"ok"}'
            )
            with pytest.raises(ValueError, match="not declared"):
                await runtime.prompt((PromptMedia(b"png", "image/png"),))
        finally:
            await runtime.aclose()

    asyncio.run(run())
    path, headers, body = requests[0]
    assert len(requests) == 1 and path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer server-key"
    assert "OpenAI-Organization" not in headers and "OpenAI-Project" not in headers
    assert body["messages"][0]["content"][1] == {
        "type": "video_url",
        "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(b"video-body").decode()},
    }
    assert "server-key" not in repr(descriptor)


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("kind", ["text", "image"])
def test_embedding_dialects_preserve_one_input_per_vector(endpoint, family, kind):
    options, requests, _ = endpoint
    desc = getattr(load_provider(family), f"get_{kind}_embedder")(
        model="served-embedding", dimensions=2, options=options
    )
    value = "hello" if kind == "text" else clip().frames[0]

    async def run():
        runtime = desc.instantiate()
        try:
            assert await getattr(runtime, f"embed_{kind}")([value, value]) == [[3, 4], [3, 4]]
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert len(requests) == 2
    body = requests[0][2]
    assert body["dimensions"] == 2
    if family == "sglang":
        assert "messages" not in body and len(body["input"]) == 1
        sent = body["input"][0][kind]
    else:
        assert "input" not in body
        part = body["messages"][1]["content"][0]
        sent = part["text"] if kind == "text" else part["image_url"]["url"]
    if kind == "text":
        assert sent == value
    else:
        from PIL import Image

        np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(base64.b64decode(sent.split(",")[1])))), value)


def test_video_wire_keeps_all_frames_and_original_timestamps(endpoint):
    options, requests, _ = endpoint
    desc = load_provider("vllm").get_video_embedder(model="Qwen/Qwen3-VL-Embedding-2B", dimensions=128, options=options)
    assert desc.supports_image_queries() and desc.get_input_spec().max_frames == 64

    async def run():
        runtime = desc.instantiate()
        try:
            await runtime.embed_video([clip()])
        finally:
            await runtime.aclose()

    asyncio.run(run())
    body = requests[0][2]
    metadata = body["media_io_kwargs"]["video"]
    assert metadata["num_frames"] == -1 and metadata["do_sample_frames"] is False
    assert [index / metadata["fps"] for index in metadata["frames_indices"]] == list(clip().frame_times)
    parts = body["messages"][1]["content"][0]["video_url"]["url"].split(",")
    assert parts[0] == "data:video/jpeg;base64" and len(parts) == 3
    from PIL import Image

    assert [np.asarray(Image.open(io.BytesIO(base64.b64decode(data)))).mean() for data in parts[1:]] == [20, 90]
    with pytest.raises(ValueError, match="timestamps"):
        load_provider("sglang").get_video_embedder(model="served", dimensions=2, options=options)


@pytest.mark.parametrize("kind", ["text", "image", "video"])
def test_public_embedding_plans_validate_http_and_dimensions(kind):
    options = {"transport": "http", "base_url": "http://inference.invalid/v1"}
    desc, dims, udf, *_ = _prepare_embed_call("vllm", "served", 2, "raise", options, relation=True, input_kind=kind)
    assert dims == 2 and udf.num_gpus == 0 and desc.is_async()
    with pytest.raises(ValueError, match="dimensions"):
        _prepare_embed_call("vllm", "served", None, "ignore", options, relation=True, input_kind=kind)


def test_text_only_http_prompt_has_no_media_capability(endpoint):
    options, requests, _ = endpoint
    with vane.connect() as conn:
        row = (
            conn.sql("select 'hello' q")
            .select(prompt(vane.col("q"), provider="sglang", model="served", **options))
            .fetchone()
        )
        assert row == ('{"answer":"ok"}',)
    assert len(requests) == 1


def test_public_video_embedding_preserves_nulls_and_uses_http(endpoint):
    options, requests, _ = endpoint
    dtype = vane.list_type(
        vane.struct_type(
            {
                "frame_index": vane.sqltypes.BIGINT,
                "frame_time": vane.sqltypes.DOUBLE,
                "data": vane.sqltype("IMAGE('RGB')"),
            }
        )
    )
    value = clip()
    frames = [
        {"frame_index": index, "frame_time": time, "data": data}
        for index, time, data in zip(value.frame_indices, value.frame_times, value.frames)
    ]
    array = FileUDFContract("fixture", (), (dtype,)).scalar_outputs_to_array([None, frames, None])
    with vane.connect() as conn:
        conn.register("clips", pa.table({"frames": array}))
        rows = (
            conn.table("clips")
            .select(
                embed_video(
                    vane.col("frames"),
                    provider="vllm",
                    model="served",
                    dimensions=2,
                    normalize=True,
                    max_retries=0,
                    **options,
                )
            )
            .fetchall()
        )
    assert rows[0] == rows[2] == (None,)
    np.testing.assert_allclose(rows[1][0], [0.6, 0.8])
    assert len(requests) == 1


@pytest.mark.parametrize("entry", ["python", "sql"])
def test_public_prompt_and_embedding_execute_via_http(endpoint, tmp_path, entry):
    options, requests, _ = endpoint
    media = tmp_path / "input.mp4"
    media.write_bytes(b"fixture-video")
    with vane.connect() as conn:
        if entry == "python":
            result = (
                conn.sql("select 'find event' as q")
                .select(
                    prompt(
                        [vane.col("q"), vane.file(str(media), "video/mp4")],
                        provider="vllm",
                        model="served-vlm",
                        **options,
                        media_mime_types=["video/mp4"],
                        max_retries=0,
                    ),
                    embed(
                        vane.col("q"),
                        provider="vllm",
                        model="served-embedding",
                        dimensions=2,
                        normalize=True,
                        **options,
                    ),
                )
                .fetchone()
            )
        else:
            result = conn.sql(
                """select ai_prompt(q, file(?, 'video/mp4', NULL, NULL, NULL), provider => 'vllm', model => 'served-vlm',
                                  options => {transport: 'http', base_url: ?, media_mime_types: ['video/mp4']})
                                  from (select 'find event' q)""",
                params=[str(media), options["base_url"]],
            ).fetchone()
        assert result[0] == '{"answer":"ok"}'
        if entry == "python":
            np.testing.assert_allclose(result[1], [0.6, 0.8])
    assert any(path.endswith("chat/completions") for path, _, _ in requests)


@pytest.mark.parametrize("index", [True, 1, None, "0"])
def test_malformed_embedding_index_is_rejected(endpoint, index):
    options, requests, overrides = endpoint
    overrides.append({"data": [{"index": index, "embedding": [3, 4]}]})
    desc = load_provider("vllm").get_text_embedder(model="served", dimensions=2, options=options)

    async def run():
        runtime = desc.instantiate()
        try:
            with pytest.raises(TypeError, match="index zero"):
                await runtime.embed_text(["hello"])
        finally:
            await runtime.aclose()

    asyncio.run(run())


def test_http_provider_executes_in_existing_ray_worker(endpoint, ray_local, monkeypatch):
    import ray

    options, requests, _ = endpoint
    monkeypatch.setenv("VLLM_API_KEY", "driver-only-key")
    descriptor = load_provider("vllm").get_prompter(
        model="served-vlm", options={**options, "media_mime_types": ["video/mp4"]}
    )
    embedding = load_provider("vllm").get_text_embedder(model="served-embedding", dimensions=2, options=options)

    @ray.remote
    def execute(prompt_desc, embed_desc):
        import os

        os.environ["VLLM_API_KEY"] = "wrong-worker-key"
        os.environ["OPENAI_BASE_URL"] = "http://wrong.invalid"

        async def run():
            prompter, embedder = prompt_desc.instantiate(), embed_desc.instantiate()
            try:
                return await prompter.prompt(
                    ("question", PromptMedia(b"video", "video/mp4"))
                ), await embedder.embed_text(["query"])
            finally:
                await prompter.aclose()
                await embedder.aclose()

        return asyncio.run(run())

    assert ray.get(execute.remote(descriptor, embedding), timeout=30) == ('{"answer":"ok"}', [[3, 4]])
    assert len(requests) == 2 and all(
        headers["Authorization"] == "Bearer driver-only-key" for _, headers, _ in requests
    )
