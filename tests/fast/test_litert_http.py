# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Official LiteRT HTTP rows, media fidelity, budgets and worker contracts."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import pickle
import sys
import wave

import numpy as np
import pytest

from vane.ai import AudioClip, EmbeddingContent, EmbeddingPart, EmbeddingPartSpec, VideoClip
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._litert_options import LITERT_DEFAULT_MODEL, LITERT_MODELS
from vane.ai.functions import _prepare_embed_call
from vane.ai.provider import _ProviderResultError, load_provider

httpx = pytest.importorskip("httpx")


def descriptor(kind="multimodal", *, model=LITERT_DEFAULT_MODEL, dimensions=128, api_key=None, **options):
    return getattr(load_provider("litert", api_key=api_key), "get_" + kind + "_embedder")(
        model,
        dimensions,
        options={"runtime": "http", "base_url": "http://litert.invalid/v1", "model_sha256": "a" * 64} | options,
    )


def part(kind, value, mime=None):
    return EmbeddingPart(EmbeddingPartSpec(kind, mime), value)


def content(*parts):
    return EmbeddingContent(parts)


def clip():
    return VideoClip(
        tuple(np.full((3, 4, 3), value, dtype=np.uint8) for value in (10, 20, 30)),
        (0.0, 0.5, 0.5),
        (0, 1, 2),
    )


def audio():
    return AudioClip(np.array([[-1.0], [-0.25], [0], [0.25], [1]], np.float32), 16000)


def response(rows):
    return httpx.Response(
        200,
        json={
            "data": [{"index": i, "embedding": np.eye(1, 128, i).ravel().tolist()} for i in reversed(range(len(rows)))],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        },
    )


async def run(d, rows, handler, *, method="embed_multimodal", on_error="raise"):
    runtime = d.instantiate()
    runtime.configure_execution(max_retries=0, on_error=on_error, validate=lambda vector: vector)
    runtime._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    try:
        return await getattr(runtime, method)(rows)
    finally:
        await runtime.aclose()


@pytest.mark.parametrize("kind", ["text", "image", "video", "audio", "multimodal"])
def test_metadata_loader_snapshot_and_preparation_without_sdk_or_client(monkeypatch, kind):
    monkeypatch.setitem(sys.modules, "litert_lm", None)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: pytest.fail("HTTP client created during planning"))
    d = pickle.loads(pickle.dumps(descriptor(kind, api_key="private-sentinel")))
    assert "private-sentinel" not in repr(d)
    assert d.is_async() and d.get_udf_options().num_gpus == 0
    assert d.get_output_spec().outputs[0].space == LITERT_DEFAULT_MODEL + ":" + "a" * 64
    assert d.instantiate()._client is None
    call = _prepare_embed_call("litert", d.model, 128, "raise", d.get_options(), relation=True, input_kind=kind)
    assert call[0].get_output_spec() == d.get_output_spec()
    assert call[0].is_async()


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor", "ray_task", "ray_actor"])
def test_http_does_not_apply_native_engine_ownership_restrictions(backend):
    options = descriptor("text").get_options() | {"execution_backend": backend}
    if backend.endswith("actor"):
        options["actor_number"] = 2
    _prepare_embed_call("litert", LITERT_DEFAULT_MODEL, 128, "raise", options, relation=True)


def test_ordered_frames_and_audio_are_one_row_with_exact_pixels_and_pcm():
    from PIL import Image

    video, sound = clip(), audio()
    rows = [content(part("video", video), part("audio", sound))]

    def handler(req):
        assert str(req.url) == "http://litert.invalid/v1/embeddings"
        body = json.loads(req.content)
        assert body["normalize"] is True and body["insert_special_tokens"] is True
        assert body["dimensions"] == 128 and body["encoding_format"] == "float"
        assert len(body["input"]) == 1
        parts = body["input"][0]
        assert [item["type"] for item in parts] == ["image_url"] * 3 + ["input_audio"]
        for item, expected in zip(parts[:3], video.frames, strict=True):
            data = base64.b64decode(item["image_url"]["url"].split(",", 1)[1], validate=True)
            with Image.open(io.BytesIO(data)) as image:
                np.testing.assert_array_equal(np.asarray(image), expected)
        raw = base64.b64decode(parts[3]["input_audio"]["data"], validate=True)
        with wave.open(io.BytesIO(raw)) as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
            np.testing.assert_array_equal(np.frombuffer(wav.readframes(5), "<i2"), [-32768, -8192, 0, 8192, 32767])
        return response(body["input"])

    vectors = asyncio.run(run(descriptor(), rows, handler))
    assert len(vectors) == 1 and vectors[0].shape == (128,)


def test_native_batch_means_eight_independent_http_rows_with_index_attribution():
    sizes = []

    def handler(req):
        body = json.loads(req.content)
        sizes.append(len(body["input"]))
        assert all([part["type"] for part in row] == ["image_url"] * 3 + ["input_audio"] for row in body["input"])
        return response(body["input"])

    row = content(part("video", clip()), part("audio", audio()))
    result = asyncio.run(run(descriptor(request_batch_size=8), [row] * 9, handler))
    assert sizes == [8, 1]
    assert [int(np.argmax(vector)) for vector in result] == list(range(8)) + [0]


@pytest.mark.parametrize(
    "kind,method,value",
    [
        ("video", "embed_video", clip()),
        ("audio", "embed_audio", audio()),
        ("image", "embed_image", np.zeros((2, 2, 3), np.uint8)),
    ],
)
def test_unary_media_use_the_same_http_row_contract(kind, method, value):
    def handler(req):
        return response(json.loads(req.content)["input"])

    assert len(asyncio.run(run(descriptor(kind), [value], handler, method=method))) == 1


@pytest.mark.parametrize(
    "role,prefix", [("query", "task: search result | query: "), ("document", "title: none | text: ")]
)
@pytest.mark.parametrize("text", ["find a bicycle", "", "   "])
def test_unary_text_preserves_blank_semantics_and_exact_retrieval_prefix(role, prefix, text):
    def handler(req):
        body = json.loads(req.content)
        assert body["input"] == [[{"type": "text", "text": prefix + text}]]
        return response(body["input"])

    asyncio.run(run(descriptor("text", input_type=role), [text], handler, method="embed_text"))


def test_joint_optional_parts_and_repeated_text_keep_order_and_one_prefix():
    codec = descriptor()._codec()
    body = json.loads(
        codec.encode(
            content(
                part("audio", None),
                part("text", " "),
                part("text", "first"),
                part("image", np.zeros((1, 1, 3), np.uint8)),
                part("text", "last"),
            )
        )
    )
    assert [item["type"] for item in body] == ["text", "image_url", "text"]
    assert body[0]["text"] == "title: none | text: first" and body[2]["text"] == "last"


@pytest.mark.parametrize("on_error", ["raise", "ignore"])
@pytest.mark.parametrize(
    "bad",
    [
        content(part("text", " ")),
        content(part("audio", None)),
        content(part("video", b"mp4", "video/mp4")),
        content(part("video", VideoClip(clip().frames, (0, 2, 1), (0, 1, 2)))),
        content(part("video", VideoClip(clip().frames, (0, 1, 2), (0, 1, 1)))),
        content(part("image", np.zeros((2, 2, 3), np.float32))),
        content(part("audio", AudioClip(audio().samples, 48000))),
    ],
)
def test_invalid_input_finishes_before_client_creation_even_with_ignore(monkeypatch, on_error, bad):
    runtime = descriptor().instantiate()
    runtime.configure_execution(max_retries=0, on_error=on_error, validate=lambda value: value)
    monkeypatch.setattr(runtime, "_get_client", lambda: pytest.fail("invalid input reached client creation"))
    with pytest.raises(EmbeddingConfigurationError):
        asyncio.run(runtime.embed_multimodal([content(part("text", "valid")), bad]))


@pytest.mark.parametrize("on_error", ["raise", "ignore"])
def test_decoded_and_encoded_prefix_and_base64_json_budgets_are_not_ignored(monkeypatch, on_error):
    for options, row in [
        ({"max_input_bytes": 35}, content(part("video", clip()))),
        ({"max_input_bytes": 10}, content(part("text", "x"))),
        ({"max_input_bytes": 60}, content(part("image", np.zeros((1, 1, 3), np.uint8)))),
        ({"max_request_bytes": 300}, content(part("text", "x" * 300))),
    ]:
        runtime = descriptor(**options).instantiate()
        runtime.configure_execution(max_retries=0, on_error=on_error, validate=lambda value: value)
        monkeypatch.setattr(runtime, "_get_client", lambda: pytest.fail("invalid budget reached client creation"))
        with pytest.raises(EmbeddingConfigurationError, match="budget"):
            asyncio.run(runtime.embed_multimodal([row]))


def test_exact_serialized_limit_partitions_without_reencoding_or_merging_rows():
    d = descriptor("text")
    row = content(part("text", "example"))
    exact = len(d._codec().body([d._codec().encode(row)]))
    sizes = []

    def handler(req):
        sizes.append(len(req.content))
        return response(json.loads(req.content)["input"])

    asyncio.run(run(descriptor("text", max_request_bytes=exact), [row] * 3, handler))
    assert sizes == [exact] * 3
    with pytest.raises(EmbeddingConfigurationError, match="serialized"):
        asyncio.run(run(descriptor("text", max_request_bytes=exact - 1), [row], handler))


@pytest.mark.parametrize(
    "options",
    [
        {"max_length": 2048},
        {"vision_tokens_per_image": 70},
        {"backend": "cpu"},
        {"model_path": "/models/model.litertlm"},
        {"http_input_format": "text"},
        {"request_batch_size": 0},
        {"timeout": float("inf")},
        {"base_url": "http://private@host/v1"},
        {"model_sha256": "bad"},
    ],
)
def test_unsupported_server_settings_and_invalid_http_contracts_fail_during_planning(options):
    with pytest.raises((EmbeddingConfigurationError, TypeError)):
        descriptor(**options)


@pytest.mark.parametrize("model", list(LITERT_MODELS))
def test_model_specific_video_and_audio_contracts(model):
    kinds = LITERT_MODELS[model]
    spec = descriptor(model=model).get_input_spec()
    assert (EmbeddingPartSpec("video") in spec.supported_parts) == ("image" in kinds)
    assert (EmbeddingPartSpec("audio") in spec.supported_parts) == ("audio" in kinds)
    if "image" not in kinds:
        with pytest.raises(EmbeddingConfigurationError):
            descriptor("video", model=model)


@pytest.mark.parametrize("key", ["", " ", "bad\ncredential"])
def test_http_auth_validation_does_not_block_native_and_service_metadata(monkeypatch, key):
    monkeypatch.setenv("LITERT_API_KEY", key)
    provider = load_provider("litert")
    provider.get_text_embedder(options={"model_path": "/models/model.litertlm", "model_sha256": "a" * 64})
    provider.get_text_embedder(
        options={
            "runtime": "service",
            "service_path": "/models/service.json",
            "service_id": "fixture",
            "model_sha256": "a" * 64,
        }
    )
    with pytest.raises(ValueError, match="LITERT_API_KEY"):
        provider.get_text_embedder(options=descriptor("text", api_key="valid").get_options())


@pytest.mark.parametrize("on_error", ["raise", "ignore"])
@pytest.mark.parametrize("family", ["litert", "sglang", "vllm"])
def test_local_resource_failure_remains_fatal(on_error, family):
    def handler(req):
        raise MemoryError("private-resource-sentinel")

    d = (
        descriptor("text")
        if family == "litert"
        else load_provider(family).get_text_embedder(
            "fixture-model", 128, options={"runtime": "http", "base_url": "http://model.invalid/v1"}
        )
    )
    with pytest.raises(MemoryError, match="Embedding exhausted") as caught:
        asyncio.run(run(d, [content(part("text", "test"))], handler, on_error=on_error))
    assert "private-resource-sentinel" not in str(caught.value)


@pytest.mark.parametrize("bad", [[0] * 128, [1] * 128, [1] + [0] * 126])
def test_invalid_response_shape_and_nonunit_vectors_fail(bad):
    def handler(req):
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": bad}]})

    with pytest.raises(_ProviderResultError):
        asyncio.run(run(descriptor("text"), [content(part("text", "test"))], handler))
