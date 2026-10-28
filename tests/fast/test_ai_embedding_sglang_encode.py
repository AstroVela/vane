# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit SGLang /encode wire contracts; no model SDKs or weights."""

import asyncio
import base64
import json

import numpy as np
import pytest

from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai.options import validate_embed_options
from vane.ai.provider import _ProviderResultError

from .test_ai_embedding_http_runtime import _content, _descriptor, _part, _run, _text

httpx = pytest.importorskip("httpx")


def descriptor(**options):
    return _descriptor(
        **{
            "base_url": "http://embedding.invalid",
            "http_input_format": "encode",
            "http_media_tokens": {"video": "<v>", "audio": "<a>", "image": "<i>"},
            "media_mime_types": ["video/mp4", "audio/wav", "image/png"],
        }
        | options
    )


@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("override", [False, True])
def test_encode_transports_audio_video_and_text_without_reencoding(batch_size, override):
    calls = []
    video, audio = b"exact MP4 bytes", b"exact WAV bytes"

    def handler(request):
        assert str(request.url) == "http://embedding.invalid/encode"
        body = json.loads(request.content)
        assert "model" not in body and "input" not in body
        assert body.get("dimensions") == (2 if override else None)
        single = isinstance(body["text"], str)
        rows = [body]
        if not single:
            rows = [
                dict(zip(("text", "image_data", "video_data", "audio_data"), values, strict=True))
                for values in zip(body["text"], body["image_data"], body["video_data"], body["audio_data"], strict=True)
            ]
        calls.append(len(rows))
        results = []
        for row in rows:
            index = int(row["text"][-1])
            assert row["image_data"] is None
            assert row["text"] == ("<v><a>" if index % 2 == 0 else "") + str(index)
            if index % 2 == 0:
                assert base64.b64decode(row["video_data"].split(",", 1)[1], validate=True) == video
                assert row["audio_data"].startswith("data:audio/wav;base64,")
                assert base64.b64decode(row["audio_data"].split(",", 1)[1], validate=True) == audio
            else:
                assert row["audio_data"] is row["video_data"] is None
            results.append({"embedding": [float(index), 1.0], "meta_info": {"prompt_tokens": 3}})
        return httpx.Response(200, json=results[0] if single else results)

    rows = [
        _content(_part("video", video, "video/mp4"), _part("text", str(i)), _part("audio", audio, "audio/wav"))
        if i % 2 == 0
        else _text(str(i))
        for i in range(5)
    ]
    vectors = asyncio.run(
        _run(descriptor(request_batch_size=batch_size, supports_overriding_dimensions=override), rows, handler)
    )
    np.testing.assert_array_equal(vectors, [[i, 1] for i in range(5)])
    assert calls == ([1] * 5 if batch_size == 1 else [2, 2, 1] if batch_size == 2 else [5])


def test_encode_defaults_to_concurrent_single_row_requests():
    instance = descriptor().instantiate()
    assert instance._request_batch_size == 1 and instance._request_concurrency == 4


@pytest.mark.parametrize("on_error", ["raise", "ignore"])
@pytest.mark.parametrize("kind", ["bytes", "serialized", "marker", "duplicate"])
def test_encode_rejects_invalid_inputs_before_creating_a_client(monkeypatch, on_error, kind):
    row = {
        "bytes": _content(_part("audio", b"x" * 1025, "audio/wav")),
        "serialized": _content(_part("audio", b"x" * 768, "audio/wav")),
        "marker": _text("literal <a> marker"),
        "duplicate": _content(_part("audio", b"one", "audio/wav"), _part("audio", b"two", "audio/wav")),
    }[kind]
    instance = descriptor(max_input_bytes=1024, max_request_bytes=1024).instantiate()
    instance.configure_execution(max_retries=0, on_error=on_error, validate=lambda vector: vector)
    monkeypatch.setattr(instance, "_get_client", lambda: pytest.fail("invalid input created client"))
    with pytest.raises(EmbeddingConfigurationError):
        asyncio.run(instance.embed_multimodal([row]))


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_column_batch_encoding_respects_combined_serialized_budget(batch_size):
    instance = descriptor(request_batch_size=batch_size, max_request_bytes=600).instantiate()
    rows = [
        instance._codec.encode(_text("x" * 20))
        if i % 3
        else instance._codec.encode(_content(_part("audio", b"x" * 90, "audio/wav")))
        for i in range(32)
    ]
    assert all(len(instance._codec.body(parts)) <= 600 for _, parts in instance._batches(rows))


@pytest.mark.parametrize("value", [{}, [], {"embedding": [1]}, {"embedding": [float("nan"), 1]}])
def test_malformed_encode_response_is_not_an_embedding(value):
    with pytest.raises(_ProviderResultError):
        asyncio.run(
            _run(
                descriptor(),
                [_text("query")],
                lambda request: httpx.Response(
                    200, content=json.dumps(value), headers={"Content-Type": "application/json"}
                ),
            )
        )


@pytest.mark.parametrize(
    "family,options",
    [
        ("vllm", {"http_input_format": "encode"}),
        ("sglang", {"http_input_format": "encode", "media_mime_types": ["audio/wav"]}),
        ("sglang", {"http_input_format": "multimodal", "http_media_tokens": {"audio": "<a>"}}),
        ("sglang", {"http_input_format": "encode", "http_media_tokens": {"audio": ""}}),
        ("sglang", {"http_input_format": "encode", "http_media_tokens": {"audio": "same", "video": "same"}}),
    ],
)
def test_invalid_encode_deployment_is_rejected(family, options):
    with pytest.raises((ValueError, TypeError)):
        validate_embed_options(
            family, {"runtime": "http", "base_url": "http://example.invalid"} | options, relation=False
        )
