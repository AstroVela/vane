# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""SGLang / vLLM HTTP protocol, admission and lifecycle contracts, without weights."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import pickle

import numpy as np
import pytest

from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai.options import validate_embed_options
from vane.ai.provider import _ProviderResultError
from vane.ai.providers.sglang import SGLangProvider
from vane.ai.providers.vllm import VLLMProvider

httpx = pytest.importorskip("httpx")
PROVIDERS = {"sglang": SGLangProvider, "vllm": VLLMProvider}


def _descriptor(family="sglang", *, api_key=None, **options):
    return PROVIDERS[family](api_key=api_key).get_multimodal_embedder(
        "served-model",
        2,
        options={
            "runtime": "http",
            "base_url": "http://embedding.invalid/v1",
            "http_input_format": "multimodal",
            "media_mime_types": ["video/mp4", "image/png"],
        }
        | options,
    )


def _part(kind, value, mime=None):
    return EmbeddingPart(EmbeddingPartSpec(kind, mime), value)


def _content(*parts):
    return EmbeddingContent(parts)


def _text(value):
    return _content(_part("text", value))


def _inputs(request):
    payload = json.loads(request.content)
    rows = payload.get("input", payload.get("messages"))
    texts = []
    for row in rows:
        if isinstance(row, str):
            texts.append(row)
        elif isinstance(row, dict):
            texts.append(row.get("text", "0"))
        else:
            texts.append(next((part["text"] for part in row[0]["content"] if part["type"] == "text"), "0"))
    return payload, rows, texts


def _response(texts):
    return httpx.Response(
        200,
        json={"data": [{"index": i, "embedding": [float(text), 1.0]} for i, text in reversed(list(enumerate(texts)))]},
    )


async def _run(descriptor, contents, handler, *, on_error="raise", retries=0):
    instance = descriptor.instantiate()
    instance.configure_execution(max_retries=retries, on_error=on_error, validate=lambda vector: vector)
    instance._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    try:
        return await instance.embed_multimodal(contents)
    finally:
        await instance.aclose()


@pytest.mark.parametrize("family", PROVIDERS)
def test_planning_is_serializable_and_requires_no_runtime_or_client(monkeypatch, family):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: pytest.fail("client created during planning"))
    descriptor = _descriptor(family, api_key="private-key")
    assert "private-key" not in repr(descriptor)
    clone = pickle.loads(pickle.dumps(descriptor))
    assert clone.get_dimensions() == 2
    assert clone.get_udf_options().num_gpus == 0
    assert clone.is_async()
    clone.instantiate()


@pytest.mark.parametrize("family", PROVIDERS)
@pytest.mark.parametrize("key", ["", " ", "invalid\ncredential"])
@pytest.mark.parametrize("explicit", [False, True])
def test_unused_http_credentials_do_not_block_native_embedding_or_prompt(monkeypatch, family, key, explicit):
    from vane.ai.protocols import NativeInferencePlan, PrompterDescriptor
    from vane.ai.providers._native_inference import (
        NativeImageEmbedderDescriptor,
        NativeTextEmbedderDescriptor,
        NativeVideoEmbedderDescriptor,
    )

    monkeypatch.setenv(f"{family.upper()}_API_KEY", key)
    provider = PROVIDERS[family](**({"api_key": key} if explicit else {}))
    for options in (None, {"runtime": "native"}):
        for kind, expected in (
            ("text", NativeTextEmbedderDescriptor),
            ("image", NativeImageEmbedderDescriptor),
            ("video", NativeVideoEmbedderDescriptor),
        ):
            descriptor = getattr(provider, f"get_{kind}_embedder")("native-model", 2, options=options)
            assert isinstance(descriptor, expected)
    assert isinstance(provider.get_prompter("native-model"), NativeInferencePlan)
    assert isinstance(
        provider.get_prompter("native-model", options={"media_mime_types": ["image/png"]}), PrompterDescriptor
    )


@pytest.mark.parametrize("family", PROVIDERS)
@pytest.mark.parametrize("key", ["", " ", "invalid\ncredential"])
@pytest.mark.parametrize("kind", ["text", "image", "multimodal"])
def test_http_validates_the_captured_key_before_client_creation(monkeypatch, family, key, kind):
    monkeypatch.setenv(f"{family.upper()}_API_KEY", key)
    provider = pickle.loads(pickle.dumps(PROVIDERS[family]()))
    # Later application/worker state must not repair an invalid snapshot.
    monkeypatch.setenv(f"{family.upper()}_API_KEY", "later-valid-key")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: pytest.fail("client created before validation"))
    with pytest.raises(ValueError, match=f"{family.upper()}_API_KEY") as error:
        getattr(provider, f"get_{kind}_embedder")(
            "served-model",
            2,
            options={
                "runtime": "http",
                "base_url": "http://embedding.invalid/v1",
                "http_input_format": "multimodal",
                "media_mime_types": ["image/png"],
            },
        )
    assert "invalid\ncredential" not in str(error.value)


@pytest.mark.parametrize("family", PROVIDERS)
@pytest.mark.parametrize("key", [None, "application-key"])
def test_http_preserves_missing_or_valid_driver_key_across_serialization(monkeypatch, family, key):
    from vane.ai._redaction import unwrap_sensitive_options

    variable = f"{family.upper()}_API_KEY"
    monkeypatch.delenv(variable, raising=False)
    if key is not None:
        monkeypatch.setenv(variable, key)
    provider = pickle.loads(pickle.dumps(PROVIDERS[family]()))
    monkeypatch.setenv(variable, "different-worker-key")
    descriptor = provider.get_text_embedder(
        "served-model", 2, options={"runtime": "http", "base_url": "http://embedding.invalid/v1"}
    )
    assert unwrap_sensitive_options(descriptor.client_options)["api_key"] == key
    assert "application-key" not in repr(descriptor)


@pytest.mark.parametrize("family", PROVIDERS)
@pytest.mark.parametrize("credential", ["explicit", "environment", "missing"])
@pytest.mark.parametrize("kind", ["text", "multimodal"])
def test_public_loader_authenticates_with_the_serialized_application_key(monkeypatch, family, credential, kind):
    import sys

    from vane.ai import load_provider

    variable = f"{family.upper()}_API_KEY"
    monkeypatch.delenv(variable, raising=False)
    if credential != "missing":
        monkeypatch.setenv(variable, "environment-key")
    # Loading a remote provider must not require its native inference package.
    monkeypatch.setitem(sys.modules, family, None)
    provider = load_provider(
        family, name="my-deployment", **({"api_key": "explicit-key"} if credential == "explicit" else {})
    )
    provider = pickle.loads(pickle.dumps(provider))
    monkeypatch.setenv(variable, "different-worker-key")
    descriptor = getattr(provider, f"get_{kind}_embedder")(
        "served-model",
        2,
        options={
            "runtime": "http",
            "base_url": "http://embedding.invalid/v1",
            "http_input_format": "multimodal",
            "request_batch_size": 2,
        },
    )
    assert provider.name == descriptor.get_provider() == "my-deployment"
    assert all(key not in repr(descriptor) for key in ("explicit-key", "environment-key", "different-worker-key"))
    assert "api_key" not in descriptor.get_options()
    descriptor = pickle.loads(pickle.dumps(descriptor))
    expected = {"explicit": "Bearer explicit-key", "environment": "Bearer environment-key", "missing": None}[credential]
    calls = []

    def handler(request):
        calls.append(request)
        assert request.headers.get("authorization") == expected
        return _response(_inputs(request)[2])

    client_factory = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: client_factory(transport=httpx.MockTransport(handler), **kwargs)
    )

    async def run():
        client = descriptor.instantiate()
        try:
            result = await client.embed_text(["1", "2"])
            assert np.asarray(result).tolist() == [[1.0, 1.0], [2.0, 1.0]]
        finally:
            await client.aclose()

    asyncio.run(run())
    assert len(calls) == 1


@pytest.mark.parametrize("family", PROVIDERS)
def test_public_loader_preserves_native_credential_validation_boundary(family):
    from vane.ai import load_provider
    from vane.ai.protocols import NativeInferencePlan
    from vane.ai.providers._native_inference import NativeTextEmbedderDescriptor

    provider = load_provider(family, api_key="")
    assert isinstance(provider.get_text_embedder("native-model", 2), NativeTextEmbedderDescriptor)
    assert isinstance(provider.get_prompter("native-model"), NativeInferencePlan)
    with pytest.raises(ValueError, match=f"{family.upper()}_API_KEY"):
        provider.get_text_embedder(
            "served-model", 2, options={"runtime": "http", "base_url": "http://embedding.invalid/v1"}
        )


@pytest.mark.parametrize("family", PROVIDERS)
def test_public_loader_rejects_unknown_client_settings(family):
    from vane.ai import load_provider

    with pytest.raises(TypeError, match="unexpected keyword argument 'unknown_setting'"):
        load_provider(family, unknown_setting="invalid")


@pytest.mark.parametrize("family", PROVIDERS)
@pytest.mark.parametrize(
    "options",
    [
        {"runtime": "automatic"},
        {"base_url": None},
        {"base_url": "file:///tmp/model"},
        {"base_url": "http://user:pass@localhost/v1"},
        {"base_url": "http://localhost/v1?key=secret"},
        {"base_url": "http://localhost/v1#fragment"},
        {"timeout": None},
        {"timeout": float("inf")},
        {"timeout": 0},
        {"request_batch_size": True},
        {"max_concurrency_per_actor": 1025},
        {"max_input_bytes": 0},
        {"max_request_bytes": 257 * 1024**2},
        {"http_input_format": "automatic"},
        {"http_input_format": "text"},  # media declarations require multimodal format
        {"media_mime_types": "video/mp4"},
        {"media_mime_types": ["video/unknown"]},
        {"engine_args": {}},
        {"gpus_per_actor": 1},
        {"chat_template": "guessed"},
        {"supports_overriding_dimensions": 1},
    ],
)
def test_http_options_reject_ambiguous_or_native_settings(family, options):
    with pytest.raises((ValueError, TypeError)):
        _descriptor(family, **options)


@pytest.mark.parametrize("family", PROVIDERS)
def test_native_is_explicit_and_http_endpoint_does_not_select_it(family):
    provider = PROVIDERS[family]()
    for options in ({}, {"runtime": "native"}):
        assert provider.get_text_embedder("model", 2, options=options).get_udf_options().num_gpus == 1
    with pytest.raises(TypeError, match="base_url"):
        validate_embed_options(family, {"base_url": "http://localhost/v1"}, relation=False)
    for method in (provider.get_video_embedder, provider.get_audio_embedder):
        with pytest.raises(EmbeddingConfigurationError, match="encoded"):
            method("model", 2, options={"runtime": "http", "base_url": "http://localhost/v1"})


@pytest.mark.parametrize("family", PROVIDERS)
@pytest.mark.parametrize("input_format", ["text", "multimodal"])
def test_default_dispatch_preserves_protocol_batching_and_bounded_media_concurrency(family, input_format):
    single_rows = family == "sglang" and input_format == "multimodal"
    expected_rows, expected_concurrency = (1, 4) if single_rows else (8, 1)
    media = b"\x00\xffMP4 bytes\x00"
    contents = [
        _content(_part("text", str(i)), _part("video", media, "video/mp4"))
        if input_format == "multimodal"
        else _text(str(i))
        for i in range(8)
    ]

    async def run():
        active, peak = 0, 0
        calls = []
        first_wave = asyncio.Event()

        async def handler(request):
            nonlocal active, peak
            _, rows, texts = _inputs(request)
            assert len(rows) == expected_rows
            if input_format == "multimodal":
                for row in rows:
                    url = row["video"] if family == "sglang" else row[0]["content"][1]["video_url"]["url"]
                    assert base64.b64decode(url.split(",", 1)[1], validate=True) == media
            calls.append(texts)
            active += 1
            peak = max(peak, active)
            if active == expected_concurrency:
                first_wave.set()
            try:
                await first_wave.wait()
                await asyncio.sleep(0)
                return _response(texts)
            finally:
                active -= 1

        result = await asyncio.wait_for(
            _run(
                _descriptor(
                    family,
                    http_input_format=input_format,
                    media_mime_types=["video/mp4"] if input_format == "multimodal" else [],
                ),
                contents,
                handler,
            ),
            timeout=2,
        )
        assert [vector.tolist() for vector in result] == [[i, 1] for i in range(8)]
        assert len(calls) == 8 // expected_rows
        assert peak == expected_concurrency
        assert active == 0

    asyncio.run(run())


@pytest.mark.parametrize("family", PROVIDERS)
@pytest.mark.parametrize("override", [True, False])
def test_exact_media_bytes_batch_identity_and_dimensions(family, override):
    calls = []
    media = b"\x00\xff\x80MP4\x00" * 21

    def handler(request):
        payload, rows, texts = _inputs(request)
        calls.append(payload)
        assert str(request.url) == "http://embedding.invalid/v1/embeddings"
        assert payload.get("dimensions") == (2 if override else None)
        for row in rows:
            url = row["video"] if family == "sglang" else row[0]["content"][1]["video_url"]["url"]
            assert url.startswith("data:video/mp4;base64,")
            assert base64.b64decode(url.split(",", 1)[1], validate=True) == media
        return _response(texts)

    result = asyncio.run(
        _run(
            _descriptor(family, request_batch_size=2, supports_overriding_dimensions=override),
            [_content(_part("text", str(i)), _part("video", media, "video/mp4")) for i in range(5)],
            handler,
        )
    )
    assert [vector.tolist() for vector in result] == [[i, 1] for i in range(5)]
    assert [len(payload.get("input", payload.get("messages"))) for payload in calls] == [2, 2, 1]


def test_vllm_audio_and_ordered_multiple_parts():
    audio = b"unchanged WAV bytes"

    def handler(request):
        payload = json.loads(request.content)
        content = payload["messages"][0][0]["content"]
        assert [part["type"] for part in content] == ["text", "input_audio", "text"]
        assert content[1]["input_audio"]["format"] == "wav"
        assert base64.b64decode(content[1]["input_audio"]["data"]) == audio
        return _response(["1"])

    asyncio.run(
        _run(
            _descriptor("vllm", media_mime_types=["audio/wav"]),
            [_content(_part("text", "1"), _part("audio", audio, "audio/wav"), _part("text", "tail"))],
            handler,
        )
    )
    with pytest.raises(EmbeddingConfigurationError, match="unsupported"):
        _descriptor("sglang", media_mime_types=["audio/wav"])


@pytest.mark.parametrize("family", PROVIDERS)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_decoded_image_uses_lossless_png(family, channels):
    image_module = pytest.importorskip("PIL.Image")
    pixels = np.arange(12 * 16 * channels, dtype=np.uint8).reshape(12, 16, channels)

    def handler(request):
        _, rows, _ = _inputs(request)
        url = rows[0]["image"] if family == "sglang" else rows[0][0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        decoded = np.asarray(image_module.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))))
        np.testing.assert_array_equal(decoded, pixels[:, :, 0] if channels == 1 else pixels)
        return _response(["0"])

    asyncio.run(_run(_descriptor(family), [_content(_part("image", pixels))], handler))


@pytest.mark.parametrize("family", PROVIDERS)
@pytest.mark.parametrize("on_error", ["raise", "ignore"])
@pytest.mark.parametrize("invalid", ["empty", "raw_bytes", "base64_bytes", "unicode", "representation"])
def test_local_validation_finishes_before_any_client_or_request(monkeypatch, family, on_error, invalid):
    descriptor = _descriptor(family, max_input_bytes=1024, max_request_bytes=1024)
    instance = descriptor.instantiate()
    instance.configure_execution(max_retries=3, on_error=on_error, validate=lambda vector: vector)
    monkeypatch.setattr(instance, "_get_client", lambda: pytest.fail("created a client for invalid input"))
    bad = {
        "empty": _content(_part("text", "  "), _part("video", None, "video/mp4")),
        "raw_bytes": _text("x" * 1025),
        "base64_bytes": _content(_part("video", b"x" * 768, "video/mp4")),
        "unicode": _text("\ud800"),
        "representation": _content(_part("video", b"bytes")),
    }[invalid]
    with pytest.raises(EmbeddingConfigurationError):
        asyncio.run(instance.embed_multimodal([_text("1"), bad]))


def test_sglang_never_drops_duplicate_parts():
    with pytest.raises(EmbeddingConfigurationError, match="at most one"):
        asyncio.run(
            _run(_descriptor(), [_content(_part("text", "first"), _part("text", "second"))], lambda req: pytest.fail())
        )


@pytest.mark.parametrize("family", PROVIDERS)
def test_exact_serialized_budget_and_multiple_row_partition(family):
    row = _text("1" + " " * 40)
    descriptor = _descriptor(family)
    codec = descriptor._codec()
    exact = len(codec.body([codec.encode(row)]))
    sizes = []

    def handler(request):
        sizes.append(len(request.content))
        _, _, texts = _inputs(request)
        return _response(texts)

    results = asyncio.run(_run(_descriptor(family, max_request_bytes=exact), [row] * 3, handler))
    assert len(results) == 3
    assert sizes == [exact] * 3
    with pytest.raises(EmbeddingConfigurationError, match="serialized"):
        asyncio.run(_run(_descriptor(family, max_request_bytes=exact - 1), [row], handler))
    assert sizes == [exact] * 3


@pytest.mark.parametrize("family", PROVIDERS)
def test_text_query_uses_the_same_multimodal_template_contract(family):
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        return _response(["1"])

    async def run():
        instance = _descriptor(family).instantiate()
        instance._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            await instance.embed_text(["1"])
            await instance.embed_multimodal([_text("1")])
        finally:
            await instance.aclose()

    asyncio.run(run())
    assert payloads[0] == payloads[1]


@pytest.mark.parametrize("family", PROVIDERS)
def test_text_protocol_sends_a_batch_of_plain_strings(family):
    def handler(request):
        payload, _, _ = _inputs(request)
        assert payload["input"] == ["1", "2"]
        return _response(["1", "2"])

    asyncio.run(
        _run(_descriptor(family, http_input_format="text", media_mime_types=[]), [_text("1"), _text("2")], handler)
    )


@pytest.mark.parametrize("status", [401, 403, 400, 413, 429, 503])
def test_account_errors_and_exhausted_transient_errors_do_not_fan_out(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            status,
            headers={"Retry-After": "0"},
            json={"error": {"code": "INVALID_API_KEY"}} if status in {400, 413} else {},
        )

    result = asyncio.run(
        _run(_descriptor(request_batch_size=2), [_text("1"), _text("2")], handler, on_error="ignore", retries=1)
    )
    assert result == [None, None]
    assert len(calls) == (2 if status in {429, 503} else 1)


def test_server_413_splits_only_the_rejected_batch():
    calls = []

    def handler(request):
        _, _, texts = _inputs(request)
        calls.append(texts)
        if len(texts) > 1 and texts[0] == "1":
            return httpx.Response(413, json={})
        return _response(texts)

    result = asyncio.run(
        _run(
            _descriptor(request_batch_size=2, max_concurrency_per_actor=1),
            [_text(str(i)) for i in range(1, 5)],
            handler,
            on_error="ignore",
        )
    )
    assert [vector.tolist() for vector in result] == [[i, 1] for i in range(1, 5)]
    assert calls == [["1", "2"], ["1"], ["2"], ["3", "4"]]


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        {"data": []},
        {"data": [{"embedding": [1, 2]}]},
        {"data": [{"index": True, "embedding": [1, 2]}]},
        {"data": [{"index": 1, "embedding": [1, 2]}]},
    ],
)
def test_malformed_row_attribution_is_rejected(invalid):
    with pytest.raises(_ProviderResultError, match="HTTP embedding response"):
        asyncio.run(_run(_descriptor(), [_text("1")], lambda req: httpx.Response(200, json=invalid)))


@pytest.mark.parametrize("bad", [[], [1], ["1", 2], [True, 2], [None, 2], [[1], 2], [1e300, 2]])
def test_row_attributable_bad_vector_does_not_replay_successful_neighbors(bad):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"data": [{"index": 1, "embedding": bad}, {"index": 0, "embedding": [1, 2]}]})

    result = asyncio.run(_run(_descriptor(request_batch_size=2), [_text("1"), _text("2")], handler, on_error="ignore"))
    assert result[0].tolist() == [1, 2]
    assert result[1] is None
    assert len(calls) == 1


def test_bounded_concurrency_cancellation_drains_requests():
    async def run():
        active, peak, admitted = 0, 0, 0
        two_started = asyncio.Event()
        instance = _descriptor(request_batch_size=1, max_concurrency_per_actor=2).instantiate()

        async def handler(request):
            nonlocal active, peak, admitted
            admitted += 1
            active += 1
            peak = max(peak, active)
            if active == 2:
                two_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                active -= 1

        client = instance._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        task = asyncio.create_task(instance.embed_multimodal([_text(str(i)) for i in range(10)]))
        await asyncio.wait_for(two_started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await instance.aclose()
        assert active == 0
        assert peak == admitted == 2
        assert client.is_closed

    asyncio.run(run())


@pytest.mark.parametrize("family", PROVIDERS)
def test_credentials_are_driver_snapshots_and_redirects_are_disabled(monkeypatch, family):
    monkeypatch.setenv(f"{family.upper()}_API_KEY", "driver-key")
    descriptor = _descriptor(family)
    monkeypatch.setenv(f"{family.upper()}_API_KEY", "worker-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")

    async def run():
        instance = descriptor.instantiate()
        client = instance._get_client()
        try:
            assert client.headers["Authorization"] == "Bearer driver-key"
            assert client.follow_redirects is False
        finally:
            await instance.aclose()

    asyncio.run(run())
