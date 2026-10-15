# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Gemini media encoding and RPC cardinality without paid model calls."""

import asyncio
import io
import json
import pickle
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from vane.ai import AudioClip, VideoClip
from vane.ai.media import TimedContent
from vane.ai.media_models import GoogleJointEmbedder
from vane.ai.providers.google import GoogleProvider


def clip(frames=2):
    return VideoClip(
        tuple(np.full((16, 24, 3), (index * 100) % 256, np.uint8) for index in range(frames)),
        tuple(10.0 + index * 0.5 for index in range(frames)),
        tuple(range(frames)),
    )


def sound(start=10.0, rate=16000):
    return TimedContent(start, start + 0.5, AudioClip(np.full((rate // 2, 1), 0.25, np.float32), rate))


@pytest.fixture
def pillow():
    pytest.importorskip("PIL.Image", reason="Install the video extra to test image encoding")


@pytest.fixture
def sdk(monkeypatch):
    pytest.importorskip("google.genai")
    from vane.ai.providers import google

    calls = []
    state = SimpleNamespace(active=0, peak=0, closed=False, response=None)

    async def embed_content(**kwargs):
        calls.append(kwargs)
        state.active += 1
        state.peak = max(state.peak, state.active)
        try:
            await asyncio.sleep(0.01)
            if state.response is not None:
                return state.response
            dims = kwargs["config"]["output_dimensionality"]
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0] * dims) for _ in kwargs["contents"]])
        finally:
            state.active -= 1

    async def close():
        state.closed = True

    client = SimpleNamespace(
        vertexai=False, aio=SimpleNamespace(models=SimpleNamespace(embed_content=embed_content), aclose=close)
    )
    monkeypatch.setattr(google, "create_google_client", lambda *args: client)
    return calls, state, client


@pytest.mark.parametrize("kind", ["video", "audio", "image"])
def test_media_descriptors_are_lazy_serializable_and_cpu_only(kind, monkeypatch):
    descriptor = getattr(GoogleProvider(), f"get_{kind}_embedder")(dimensions=128)
    restored = pickle.loads(pickle.dumps(descriptor))
    assert restored.get_dimensions() == 128 and restored.is_async()
    assert restored.get_udf_options().num_gpus == 0
    with pytest.raises(ValueError, match="require gemini-embedding-2"):
        getattr(GoogleProvider(), f"get_{kind}_embedder")(model="gemini-embedding-001")
    with pytest.raises((ValueError, TypeError)):
        getattr(GoogleProvider(), f"get_{kind}_embedder")(options={"task_type": "RETRIEVAL_QUERY"})
    with pytest.raises(ValueError, match="only request concurrency"):
        getattr(GoogleProvider(), f"get_{kind}_embedder")(options={"request_batch_size": 2})


@pytest.mark.parametrize("kind", ["image", "video", "audio"])
def test_public_media_binding_accepts_google_concurrency_without_text_options(kind):
    from vane.ai.functions import _prepare_embed_call

    descriptor, dimensions, resources, *_ = _prepare_embed_call(
        provider=GoogleProvider(),
        model="gemini-embedding-2",
        dimensions=128,
        on_error="raise",
        options={"max_concurrency_per_actor": 2},
        input_kind=kind,
        relation=True,
    )
    assert dimensions == 128 and resources.num_gpus == 0
    assert descriptor.get_options()["max_concurrency_per_actor"] == 2


@pytest.mark.parametrize("vertex", [False, True])
def test_separate_media_preserves_rows_and_bounded_concurrency(sdk, vertex):
    calls, state, client = sdk
    client.vertexai = vertex
    descriptor = GoogleProvider().get_audio_embedder(dimensions=128, options={"max_concurrency_per_actor": 2})

    async def run():
        encoder = descriptor.instantiate()
        try:
            results = await encoder.embed_audio([sound().value] * 3)
            assert len(results) == 3 and all(len(value) == 128 for value in results)
        finally:
            await encoder.aclose()

    asyncio.run(run())
    assert state.peak == 2 and state.closed
    assert len(calls) == 3 and all(len(call["contents"]) == 1 for call in calls)
    for call in calls:
        assert call["config"].get("auto_truncate") is (False if vertex else None)
        payload = next(part.inline_data.data for part in call["contents"][0].parts if part.inline_data)
        with wave.open(io.BytesIO(payload)) as stream:
            assert (stream.getframerate(), stream.getnframes(), stream.getnchannels()) == (16000, 8000, 1)
            assert np.frombuffer(stream.readframes(8000), dtype="<i2").tolist() == [8192] * 8000


def test_joint_encodes_frames_audio_and_text_in_one_content_with_paired_query(sdk, pillow):
    calls, state, _ = sdk
    adapter = GoogleJointEmbedder(dimensions=128)
    assert adapter._runtime is None and adapter.input_modalities == {"video", "audio", "text"}

    async def run():
        try:
            assert len(await adapter.embed_joint(clip(), "spoken words", audio=(sound(rate=48000),))) == 128
            assert len(await adapter.embed_query("find spoken words")) == 128
        finally:
            await adapter.aclose()

    asyncio.run(run())
    assert state.closed and adapter._runtime is None
    assert len(calls) == 2 and all(len(call["contents"]) == 1 for call in calls)
    parts = calls[0]["contents"][0].parts
    assert [part.inline_data.mime_type for part in parts if part.inline_data] == ["image/png", "image/png", "audio/wav"]
    assert [part.text for part in parts if part.text] == [
        "spoken words",
        "frame_time_sec=10",
        "frame_time_sec=10.5",
        "audio_start_sec=10; audio_end_sec=10.5",
    ]
    assert calls[1]["contents"][0].parts[0].text == "task: search result | query: find spoken words"
    assert "task_type" not in calls[1]["config"]


def test_server_truncation_aborts_joint_encoding(sdk):
    from vane.ai._embedding_requests import _EmbeddingBatchError

    _, state, _ = sdk
    state.response = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[1.0] * 128, statistics=SimpleNamespace(truncated=True))]
    )

    async def run():
        adapter = GoogleJointEmbedder(dimensions=128)
        try:
            with pytest.raises(_EmbeddingBatchError, match="truncated the embedding input"):
                await adapter.embed_joint(None, "words")
        finally:
            await adapter.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("bad", ["frames", "empty", "nonfinite", "timeline", "overlap", "text"])
def test_invalid_joint_input_fails_before_rpc(sdk, pillow, bad):
    calls, _, _ = sdk
    video, text, audio = clip(), "words", (sound(),)
    if bad == "frames":
        video = clip(7)
    elif bad == "empty":
        video, text, audio = None, "", ()
    elif bad == "nonfinite":
        audio[0].value.samples[0] = float("nan")
    elif bad == "timeline":
        audio = (TimedContent(10, 11, sound().value),)
    elif bad == "overlap":
        audio = (sound(), sound(10.25))
    else:
        text = "x" * 8192

    async def run():
        adapter = GoogleJointEmbedder(dimensions=128)
        try:
            with pytest.raises(ValueError):
                await adapter.embed_joint(video, text, audio=audio)
        finally:
            await adapter.aclose()

    asyncio.run(run())
    assert not calls


@pytest.mark.parametrize("values", [[], [1.0], [float("nan")] * 128, [0.0] * 128])
def test_invalid_vectors_abort_joint_encoding(sdk, values):
    _, state, _ = sdk
    state.response = SimpleNamespace(embeddings=[SimpleNamespace(values=values)])

    async def run():
        adapter = GoogleJointEmbedder(dimensions=128)
        try:
            with pytest.raises(ValueError):
                await adapter.embed_joint(None, "words")
        finally:
            await adapter.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("vertex", [False, True])
def test_real_sdk_preserves_media_parts_and_independent_rows(monkeypatch, pillow, vertex):
    import httpx

    pytest.importorskip("google.genai")
    from vane.ai.providers import google

    bodies = []

    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        entries = body.get("requests", [body])
        assert len(entries) == 1
        assert entries[0]["content"]["parts"]
        vector = {"values": [float(len(bodies))] * 128}
        if vertex:
            assert body["embedContentConfig"]["autoTruncate"] is False
            assert body["embedContentConfig"]["outputDimensionality"] == 128
        return httpx.Response(200, json={"embedding": vector} if vertex else {"embeddings": [vector]})

    transport = httpx.MockTransport(handle)

    def client(factory, types, snapshot):
        return factory(
            api_key="fixture-key",
            vertexai=vertex,
            http_options=types.HttpOptions(
                base_url="http://fixture.invalid",
                client_args={"transport": transport},
                async_client_args={"transport": transport},
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    monkeypatch.setattr(google, "create_google_client", client)

    async def run():
        for kind, inputs in [
            ("audio", [sound().value] * 2),
            ("video", [clip()] * 2),
            ("image", [clip().frames[0]] * 2),
        ]:
            encoder = getattr(GoogleProvider(), f"get_{kind}_embedder")(dimensions=128).instantiate()
            try:
                result = await getattr(encoder, f"embed_{kind}")(inputs)
                assert len(result) == 2 and result[1][0] == result[0][0] + 1
            finally:
                await encoder.aclose()
        joint = GoogleJointEmbedder(dimensions=128)
        try:
            assert len(await joint.embed_joint(clip(), "words", audio=(sound(),))) == 128
        finally:
            await joint.aclose()

    asyncio.run(run())
    assert len(bodies) == 7
    from google.genai import types

    content = bodies[-1].get("requests", [bodies[-1]])[0]["content"]
    parts = types.Content.model_validate(content).parts
    assert [part.inline_data.mime_type for part in parts if part.inline_data] == [
        "image/png",
        "image/png",
        "audio/wav",
    ]
