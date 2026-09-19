# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Embedding contracts using in-process fake models and SDKs, without downloads."""

from __future__ import annotations

import asyncio
import base64
import json
import pickle
import sys
from collections import Counter
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pyarrow as pa
import pytest

from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai.functions import _EmbedTextBatch
from vane.ai.options import validate_embed_options
from vane.ai.providers.openai import OpenAITextEmbedder, OpenAITextEmbedderDescriptor


def _openai(request, *, concurrency=1, request_size=2, policy=None):
    embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
    embedder._provider_name = "openai"
    embedder._model = "fake"
    embedder._dimensions = 2
    embedder._request_concurrency = concurrency
    embedder._request_batch_size = request_size
    embedder._batch_token_limit = 100
    embedder._input_text_token_limit = 100
    embedder._estimate_tokens = len
    embedder._overlength = policy
    embedder._embed_batch = request
    embedder._client = SimpleNamespace(close=AsyncMock())
    return embedder


def _wrapper(embedder, **options):
    descriptor = SimpleNamespace(
        instantiate=lambda: embedder,
        get_provider=lambda: "fake",
        get_model=lambda: "fake",
    )
    return _EmbedTextBatch(descriptor, "text", "embedding", 2, **options)


def _drive(wrapper, texts):
    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        return wrapper(pa.table({"text": pa.array(texts, type=pa.string())}))["embedding"].to_pylist()
    finally:
        wrapper.close()
        loop.close()


def test_bounded_requests_restore_order_without_replaying_success():
    from vane.ai.functions import RetryAfterError

    calls = Counter()
    active = 0
    peak = 0
    two_started = asyncio.Event()

    async def request(texts):
        nonlocal active, peak
        calls[tuple(texts)] += 1
        active += 1
        peak = max(peak, active)
        try:
            if active == 2:
                two_started.set()
            await asyncio.wait_for(two_started.wait(), timeout=2)
            if texts[0] == "2" and calls[tuple(texts)] == 1:
                raise RetryAfterError(0, status=429)
            return [np.array([float(text), 1]) for text in texts]
        finally:
            active -= 1

    embedder = _openai(request, concurrency=2)
    result = _drive(_wrapper(embedder, max_retries=1), ["0", None, "1", "2", "3", "4"])
    assert result == [[0, 1], None, [1, 1], [2, 1], [3, 1], [4, 1]]
    assert peak == 2
    assert calls == {("0", "1"): 1, ("2", "3"): 2, ("4",): 1}
    assert embedder.metrics.retries == 1
    assert embedder.metrics.requests == 4


@pytest.mark.parametrize("status,retries,expected", [(401, 3, 1), (429, 1, 2), (503, 1, 2)])
def test_terminal_service_failures_do_not_fan_out(status, retries, expected):
    from vane.ai.functions import RetryAfterError

    class HTTPError(Exception):
        status_code = status

    error = RetryAfterError(0, status=status) if status != 401 else HTTPError()
    error.details = {
        "error": {"status": {401: "UNAUTHENTICATED", 429: "RESOURCE_EXHAUSTED", 503: "SERVICE_UNAVAILABLE"}[status]}
    }
    request = AsyncMock(side_effect=error)
    embedder = _openai(request, request_size=64)
    assert _drive(_wrapper(embedder, max_retries=retries, on_error="ignore"), ["a", "b", "c"]) == [None] * 3
    assert request.await_count == expected


@pytest.mark.parametrize("on_error", ["raise", "ignore"])
@pytest.mark.parametrize(
    "status,attributes",
    [
        (400, {"details": {"error": {"status": "INVALID_ARGUMENT", "details": [{"reason": "API_KEY_INVALID"}]}}}),
        (422, {"body": {"error": {"code": "invalid_api_key", "type": "authentication_error"}}}),
        (400, {"status": "FAILED_PRECONDITION"}),
        (400, {"details": [{"reason": "BILLING_DISABLED"}]}),
        (422, {"body": {"code": "insufficient_quota"}}),
        (400, {"details": {"error": {"details": [{"reason": "API_KEY_SERVICE_BLOCKED"}]}}}),
        (400, {"details": {"errors": [{"reason": "CONSUMER_SUSPENDED"}]}}),
        (400, {"status": "UNAUTHENTICATED"}),
        (422, {"body": {"error": {"param": "api_key"}}}),
        (413, {"body": {"error": {"code": "invalid_api_key"}}}),
    ],
)
def test_structured_account_errors_do_not_bisect_or_retry(status, attributes, on_error):
    class AccountError(Exception):
        status_code = status

    error = AccountError("private account diagnostic")
    error.__dict__.update(attributes)
    error.__cause__ = ConnectionError("a transport cause must not make an account error retryable")
    request = AsyncMock(side_effect=error)
    embedder = _openai(request, request_size=64)
    wrapper = _wrapper(embedder, max_retries=3, on_error=on_error)
    if on_error == "raise":
        with pytest.raises(RuntimeError):
            _drive(wrapper, ["a"] * 64)
    else:
        assert _drive(wrapper, ["a"] * 64) == [None] * 64
    assert request.await_count == 1
    assert embedder.metrics.retries == 0


@pytest.mark.parametrize("provider", ["openai", "google"])
@pytest.mark.parametrize("on_error", ["raise", "ignore"])
@pytest.mark.parametrize("terminal", [True, False])
@pytest.mark.parametrize("retry_header", [True, False])
def test_sdk_quota_classification_survives_retry_conversion(monkeypatch, provider, on_error, terminal, retry_header):
    from vane.ai import functions

    # Test the default-header conversion without waiting for its production delay.
    monkeypatch.setattr(functions, "_DEFAULT_RETRY_AFTER_SECONDS", 0.0)

    class SDKError(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"Retry-After": "0"} if retry_header else {})

    secret = "private quota diagnostic sk-test-do-not-expose"
    error = SDKError(secret)
    if provider == "openai":
        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAIError=SDKError))
        error.body = {"code": "insufficient_quota" if terminal else "rate_limit_exceeded", "message": secret}
        response = SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[i, 1]) for i in range(2)], usage=None)
        request = AsyncMock(side_effect=error if terminal else [error, response])
        embedder = _openai(None)
        del embedder._embed_batch  # Exercise the real adapter's exception conversion.
        embedder._client = SimpleNamespace(embeddings=SimpleNamespace(create=request), close=AsyncMock())
    else:
        from vane.ai.providers.google import GoogleTextEmbedder

        types = SimpleNamespace(
            Content=lambda **kwargs: SimpleNamespace(**kwargs),
            Part=SimpleNamespace(from_text=lambda **kwargs: SimpleNamespace(**kwargs)),
        )
        genai = SimpleNamespace(types=types)
        monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=genai))
        monkeypatch.setitem(sys.modules, "google.genai", genai)
        error.code = 429
        error.details = {
            "error": {
                "status": "RESOURCE_EXHAUSTED",
                "details": [{"reason": "BILLING_DISABLED" if terminal else "RATE_LIMIT_EXCEEDED"}],
                "message": secret,
            }
        }
        response = SimpleNamespace(embeddings=[SimpleNamespace(values=[i, 1]) for i in range(2)])
        request = AsyncMock(side_effect=error if terminal else [error, response])
        embedder = GoogleTextEmbedder.__new__(GoogleTextEmbedder)
        embedder._client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(embed_content=request), aclose=AsyncMock())
        )
        embedder._model = "fake"
        embedder._dimensions = 2
        embedder._options = {}

    wrapper = _wrapper(embedder, max_retries=3, on_error=on_error)
    if terminal and on_error == "raise":
        with pytest.raises(RuntimeError) as caught:
            _drive(wrapper, ["first", "second"])
        assert secret not in str(caught.value)
        assert caught.value.__context__ is None
        assert caught.value.__cause__ is None
    else:
        assert _drive(wrapper, ["first", "second"]) == ([None, None] if terminal else [[0, 1], [1, 1]])
    assert request.await_count == (1 if terminal else 2)
    assert embedder.metrics.retries == (0 if terminal else 1)


def test_input_error_isolation_preserves_good_rows():
    class InputError(Exception):
        status_code = 400
        details = {"error": {"message": "Input contains API_KEY_INVALID", "details": [{"reason": "INPUT_TOO_LONG"}]}}

    calls = []

    async def request(texts):
        calls.append(tuple(texts))
        if "bad" in texts:
            raise InputError()
        return [np.array([len(text), 1]) for text in texts]

    result = _drive(_wrapper(_openai(request, request_size=4), on_error="ignore"), ["ok", "bad", "yes"])
    assert result == [[2, 1], None, [3, 1]]
    assert calls.count(("ok",)) == 1
    assert calls.count(("yes",)) == 1


@pytest.mark.parametrize("oversized_row", [False, True])
def test_payload_too_large_splits_requests_and_preserves_recoverable_rows(oversized_row):
    class PayloadTooLarge(Exception):
        status_code = 413

    calls = []

    async def request(texts):
        calls.append(tuple(texts))
        if len(texts) > 1 or (oversized_row and texts == ["large"]):
            raise PayloadTooLarge()
        return [np.array([len(texts[0]), 1])]

    embedder = _openai(request, request_size=2)
    result = _drive(_wrapper(embedder, on_error="ignore", max_retries=0), ["ok", "large", "yes"])
    assert result == [[2, 1], None if oversized_row else [5, 1], [3, 1]]
    assert Counter(calls) == {("ok", "large"): 1, ("ok",): 1, ("large",): 1, ("yes",): 1}
    assert embedder.metrics.failed_inputs == int(oversized_row)


def test_invalid_vector_nulls_only_its_row_without_reissue():
    request = AsyncMock(return_value=[np.array([1, 2]), np.array([float("inf"), 2])])
    assert _drive(_wrapper(_openai(request), on_error="ignore"), ["a", "b"]) == [[1, 2], None]
    assert request.await_count == 1


@pytest.mark.parametrize("on_error", ["raise", "ignore"])
@pytest.mark.parametrize(
    "provider,failure",
    [(provider, failure) for provider in ("managed", "openai", "google") for failure in ("short", "extra", "empty")]
    + [
        ("openai", failure)
        for failure in (
            "duplicate",
            "missing",
            "out_of_range",
            "string",
            "bool",
            "null_data",
            "missing_data",
            "object_data",
            "string_data",
            "number_data",
            "bool_data",
            "invalid_json",
            "invalid_utf8",
        )
    ],
)
def test_batch_response_shape_errors_recover_rows_without_retrying(monkeypatch, provider, failure, on_error):
    from vane.ai.provider import _ProviderResultError

    calls = []

    async def request(texts):
        calls.append(texts)
        if provider == "openai" and "bad" in texts:
            if failure == "invalid_json":
                raise json.JSONDecodeError("private decoder diagnostic", '"private response', 0)
            if failure == "invalid_utf8":
                raise UnicodeDecodeError("utf-8", b"\xffprivate response", 0, 1, "private decoder diagnostic")
        if provider == "openai" and failure.endswith("_data") and "bad" in texts:
            if failure == "missing_data":
                return SimpleNamespace(usage=None)
            return SimpleNamespace(
                data={
                    "null_data": None,
                    "object_data": {"private diagnostic": "bad"},
                    "string_data": "private diagnostic",
                    "number_data": 1,
                    "bool_data": True,
                }[failure],
                usage=None,
            )
        data = [
            SimpleNamespace(index=i, embedding=[0 if text == "bad" else int(text), 1]) for i, text in enumerate(texts)
        ]
        if "bad" in texts:
            if failure == "short" or (failure == "missing" and len(texts) == 1):
                data = data[:-1]
            elif failure == "extra":
                data.append(SimpleNamespace(index=len(texts), embedding=[99, 1]))
            elif failure == "empty":
                data = []
            else:
                data[-1].index = {
                    "duplicate": 0 if len(texts) > 1 else -1,
                    "missing": None,
                    "out_of_range": len(texts),
                    "string": "private index diagnostic",
                    "bool": True,
                }[failure]
        if provider == "managed":
            return [np.array(item.embedding) for item in data]
        if provider == "openai":
            return SimpleNamespace(data=data, usage=None)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=item.embedding) for item in data])

    if provider == "google":
        from vane.ai.providers.google import GoogleTextEmbedder

        fake_types = SimpleNamespace(
            Content=lambda **kwargs: SimpleNamespace(**kwargs),
            Part=SimpleNamespace(from_text=lambda **kwargs: SimpleNamespace(**kwargs)),
        )
        genai = SimpleNamespace(types=fake_types)
        monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=genai))
        monkeypatch.setitem(sys.modules, "google.genai", genai)

        async def sdk_request(**kwargs):
            return await request([content.parts[0].text for content in kwargs["contents"]])

        embedder = GoogleTextEmbedder.__new__(GoogleTextEmbedder)
        embedder._model = "fake"
        embedder._dimensions = 2
        embedder._options = {}
        embedder._request_batch_size = 3
        embedder._client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(embed_content=sdk_request), aclose=AsyncMock())
        )
    else:
        embedder = _openai(request, request_size=3)
        if provider == "openai":
            monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAIError=type("SDKError", (Exception,), {})))

            async def sdk_request(**kwargs):
                return await request(kwargs["input"])

            del embedder._embed_batch
            embedder._client = SimpleNamespace(embeddings=SimpleNamespace(create=sdk_request), close=AsyncMock())

    wrapper = _wrapper(embedder, on_error=on_error, max_retries=3)
    texts = ["0", None, "bad", "3", "4"]
    if on_error == "raise":
        with pytest.raises(_ProviderResultError) as caught:
            _drive(wrapper, texts)
        assert "private" not in str(caught.value)
        assert caught.value.__context__ is None
        assert caught.value.__cause__ is None
        assert b"private" not in pickle.dumps(caught.value)
        assert calls == [["0", "bad", "3"]]
    else:
        assert _drive(wrapper, texts) == [[0, 1], None, None, [3, 1], [4, 1]]
        assert calls == [["0", "bad", "3"], ["0"], ["bad", "3"], ["bad"], ["3"], ["4"]]
        assert embedder.metrics.failed_inputs == 1
    assert embedder.metrics.retries == 0


def test_cancel_drains_workers_and_does_not_dispatch_queued_requests():
    async def run():
        active = 0
        started = asyncio.Event()
        calls = []

        async def request(texts):
            nonlocal active
            calls.append(texts)
            active += 1
            try:
                if active == 2:
                    started.set()
                await asyncio.Event().wait()
            finally:
                active -= 1

        embedder = _openai(request, concurrency=2, request_size=1)
        task = asyncio.create_task(embedder.embed_text(["a", "b", "c", "d"]))
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert active == 0
        assert len(calls) == 2

    asyncio.run(run())


@pytest.mark.parametrize("failed_worker", [0, 1])
@pytest.mark.parametrize("failure", ["request", "validation", "cancellation"])
def test_worker_failure_stops_queued_dispatch_before_gather_cancels_siblings(failed_worker, failure):
    from vane.ai.provider import _ProviderResultError

    async def run():
        gates = [asyncio.get_running_loop().create_future() for _ in range(2)]
        started = asyncio.Event()
        calls = []
        active = 0

        async def request(texts):
            nonlocal active
            calls.append(texts)
            active += 1
            if active == 2:
                started.set()
            try:
                if texts[0] in {"0", "1"}:
                    return await gates[int(texts[0])]
                await asyncio.Event().wait()
            finally:
                active -= 1

        embedder = _openai(request, concurrency=2, request_size=1)
        task = asyncio.create_task(embedder.embed_text(["0", "1", "queued"]))
        await asyncio.wait_for(started.wait(), timeout=2)
        expected = RuntimeError
        if failure == "request":
            gates[failed_worker].set_exception(RuntimeError("request failed"))
        elif failure == "validation":
            gates[failed_worker].set_result([_ProviderResultError("invalid vector")])
            expected = _ProviderResultError
        else:
            gates[failed_worker].cancel()
            expected = asyncio.CancelledError
        # Both workers are runnable before gather sees the failed task. The
        # failure is observed first, then the successful sibling resumes.
        gates[1 - failed_worker].set_result([np.array([1, 2])])
        with pytest.raises(expected):
            await asyncio.wait_for(task, timeout=2)
        assert calls == [["0"], ["1"]]
        assert active == 0

    asyncio.run(run())


def test_all_null_does_not_instantiate_provider():
    descriptor = SimpleNamespace(instantiate=lambda: pytest.fail("NULL rows must not instantiate"))
    assert _drive(_EmbedTextBatch(descriptor, "text", "embedding", 2), [None, None]) == [None, None]


def test_dimension_declaration_is_picklable_and_omits_request_override(monkeypatch):
    from vane.ai.providers import openai as provider

    calls = []
    monkeypatch.setattr(provider, "OpenAITextEmbedder", lambda **kwargs: calls.append(kwargs))
    descriptor = OpenAITextEmbedderDescriptor(
        model_name="fixed-model",
        dimensions=2,
        options={"base_url": "http://localhost:8000/v1", "supports_overriding_dimensions": False},
    )
    descriptor = pickle.loads(pickle.dumps(descriptor))
    assert descriptor.get_dimensions() == 2
    assert descriptor.request_dimensions is None
    assert calls == []
    descriptor.instantiate()
    assert calls[0]["dimensions"] is None


def test_fixed_endpoint_request_omits_dimensions_and_restores_response_indices(monkeypatch):
    response = SimpleNamespace(
        data=[SimpleNamespace(index=1, embedding=[3, 4]), SimpleNamespace(index=0, embedding=[1, 2])],
        usage=SimpleNamespace(prompt_tokens=7),
    )
    client = SimpleNamespace(embeddings=SimpleNamespace(create=AsyncMock(return_value=response)), close=AsyncMock())
    client_options = []

    def create_client(**options):
        client_options.append(options)
        return client

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=create_client, OpenAIError=type("SDKError", (Exception,), {})),
    )
    embedder = OpenAITextEmbedderDescriptor(
        model_name="fixed",
        dimensions=2,
        options={"base_url": "http://localhost:8000/v1", "supports_overriding_dimensions": False},
    ).instantiate()
    assert _drive(_wrapper(embedder), ["abc", "d"]) == [[1, 2], [3, 4]]
    assert "dimensions" not in client.embeddings.create.await_args.kwargs
    assert client_options[0]["max_retries"] == 0
    assert embedder.metrics.input_tokens == 7


def test_google_request_options_stay_out_of_sdk_config_and_batch_order(monkeypatch):
    from vane.ai.providers.google import GoogleTextEmbedder

    fake_types = SimpleNamespace(
        Content=lambda **kwargs: SimpleNamespace(**kwargs),
        Part=SimpleNamespace(from_text=lambda **kwargs: SimpleNamespace(**kwargs)),
        HttpOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        HttpRetryOptions=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    calls = []

    async def request(**kwargs):
        calls.append(kwargs)
        # Complete later-numbered requests first.
        await asyncio.sleep(0.01 if kwargs["contents"][0].parts[0].text == "0" else 0)
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[int(item.parts[0].text), 1]) for item in kwargs["contents"]]
        )

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(embed_content=request), aclose=AsyncMock()))
    genai = SimpleNamespace(types=fake_types, Client=lambda **kwargs: client)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=genai))
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    embedder = GoogleTextEmbedder(
        model="fake",
        dimensions=2,
        options={"task_type": "RETRIEVAL_QUERY", "request_batch_size": 2, "max_concurrency_per_actor": 2},
    )
    result = _drive(_wrapper(embedder), ["0", "1", "2", "3", "4"])
    assert result == [[i, 1] for i in range(5)]
    assert [len(call["contents"]) for call in calls] == [2, 2, 1]
    assert all(call["config"] == {"task_type": "RETRIEVAL_QUERY", "output_dimensionality": 2} for call in calls)


@pytest.mark.parametrize("vertexai", [False, True])
@pytest.mark.parametrize(
    "model,single_input",
    [
        ("gemini-embedding-001", True),
        ("gemini-embedding-2", True),
        ("gemini-embedding-2-preview", True),
        ("models/gemini-embedding-2", True),
        ("google/gemini-embedding-2", True),
        ("publishers/google/models/gemini-embedding-2", True),
        ("projects/test/locations/global/publishers/google/models/gemini-embedding-2", True),
        ("text-embedding-005", False),
        ("publishers/other/models/gemini-embedding-2", False),
        ("projects/test/locations/global/models/gemini-embedding-2", False),
    ],
)
def test_google_vertex_request_limit_applied_before_sdk_call(monkeypatch, vertexai, model, single_input):
    from vane.ai.providers.google import GoogleTextEmbedder

    fake_types = SimpleNamespace(
        Content=lambda **kwargs: SimpleNamespace(**kwargs),
        Part=SimpleNamespace(from_text=lambda **kwargs: SimpleNamespace(**kwargs)),
        HttpOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        HttpRetryOptions=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    calls = []

    async def request(**kwargs):
        assert kwargs["model"] == model
        texts = [content.parts[0].text for content in kwargs["contents"]]
        calls.append(texts)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[int(text), 1]) for text in texts])

    client = SimpleNamespace(
        vertexai=vertexai,
        aio=SimpleNamespace(models=SimpleNamespace(embed_content=request), aclose=AsyncMock()),
    )
    genai = SimpleNamespace(types=fake_types, Client=lambda **kwargs: client)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=genai))
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    embedder = GoogleTextEmbedder(model=model, dimensions=2, options={"request_batch_size": 2})
    assert _drive(_wrapper(embedder), ["0", None, "1", "2"]) == [[0, 1], None, [1, 1], [2, 1]]
    assert calls == ([["0"], ["1"], ["2"]] if vertexai and single_input else [["0", "1"], ["2"]])


@pytest.mark.parametrize("error_kind", ["validation", "runtime", "account", "http"])
@pytest.mark.parametrize("on_error", ["raise", "ignore"])
def test_google_statusless_sdk_validation_isolation(monkeypatch, error_kind, on_error):
    from vane.ai.provider import _ProviderResultError
    from vane.ai.providers.google import GoogleTextEmbedder

    fake_types = SimpleNamespace(
        Content=lambda **kwargs: SimpleNamespace(**kwargs),
        Part=SimpleNamespace(from_text=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    genai = SimpleNamespace(types=fake_types)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=genai))
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    error = RuntimeError("private SDK diagnostic") if error_kind == "runtime" else ValueError("private SDK diagnostic")
    if error_kind == "account":
        error.reason = "API_KEY_INVALID"
    elif error_kind == "http":
        error.code = 403
    elif error_kind == "validation":
        error.__cause__ = ConnectionError("private cause must not make validation retryable")
    calls = []

    async def request(**kwargs):
        texts = [content.parts[0].text for content in kwargs["contents"]]
        calls.append(texts)
        if "bad" in texts:
            raise error
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[int(text), 1]) for text in texts])

    embedder = GoogleTextEmbedder.__new__(GoogleTextEmbedder)
    embedder._client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(embed_content=request)))
    embedder._model = "fake"
    embedder._dimensions = 2
    embedder._options = {}
    embedder.configure_execution(max_retries=3, on_error=on_error, validate=lambda value: value)
    if on_error == "raise":
        expected = _ProviderResultError if error_kind == "validation" else type(error)
        with pytest.raises(expected) as caught:
            asyncio.run(embedder.embed_text(["0", "1", "bad", "3"]))
        if error_kind == "validation":
            assert "private" not in str(caught.value)
            assert caught.value.__cause__ is None
            assert caught.value.__context__ is None
        assert len(calls) == 1
    else:
        result = asyncio.run(embedder.embed_text(["0", "1", "bad", "3"]))
        result = [value.tolist() if value is not None else None for value in result]
        if error_kind == "validation":
            assert result == [[0, 1], [1, 1], None, [3, 1]]
            assert calls == [["0", "1", "bad", "3"], ["0", "1"], ["bad", "3"], ["bad"], ["3"]]
            assert embedder.metrics.failed_inputs == 1
        else:
            assert result == [None] * 4
            assert len(calls) == 1
    assert embedder.metrics.retries == 0


@pytest.mark.parametrize("model,dimensions", [("text-embedding-ada-002", 2), ("text-embedding-3-small", 1024)])
def test_dimension_declaration_cannot_disguise_official_model_mismatch(model, dimensions):
    with pytest.raises(ValueError):
        OpenAITextEmbedderDescriptor(
            model_name=model, dimensions=dimensions, options={"supports_overriding_dimensions": False}
        )
    assert (
        OpenAITextEmbedderDescriptor(
            model_name="text-embedding-ada-002", dimensions=1536, options={"supports_overriding_dimensions": False}
        ).request_dimensions
        is None
    )


@pytest.mark.parametrize(
    "family,options",
    [
        ("openai", {"request_batch_size": 0}),
        ("openai", {"max_concurrency_per_actor": True}),
        ("openai", {"supports_overriding_dimensions": "false"}),
        ("openai", {"overlength": "guess"}),
        ("openai", {"overlength": "truncate", "max_chunk_chars": 300}),
        ("openai", {"input_type": "query"}),
        ("google", {"overlength": "truncate"}),
        ("google", {"input_type": "query", "task_type": "RETRIEVAL_QUERY"}),
        ("transformers", {"input_type": "query", "prompt": "query: "}),
        ("transformers", {"prompt_name": ""}),
        ("transformers", {"max_concurrency_per_actor": 2}),
        ("custom", {"request_batch_size": 2}),
    ],
)
def test_invalid_or_unsupported_options_rejected_before_execution(family, options):
    with pytest.raises((ValueError, TypeError)):
        validate_embed_options(family, options, relation=True)


_GOOGLE_MODEL_PREFIXES = (
    "",
    "models/",
    "google/",
    "publishers/google/models/",
    "projects/test/locations/global/publishers/google/models/",
)


@pytest.mark.parametrize("prefix", _GOOGLE_MODEL_PREFIXES)
@pytest.mark.parametrize("model", ["gemini-embedding-001", "gemini-embedding-2"])
def test_google_resource_names_share_embedding_metadata(prefix, model):
    from vane.ai.providers.google import GoogleProvider, GoogleTextEmbedderDescriptor

    name = prefix + model
    # Fallback metadata for unknown models must not override a known model.
    descriptor = GoogleProvider(embedding_dimensions=1).get_text_embedder(model=name)
    assert descriptor.get_dimensions() == 3072
    assert descriptor.request_dimensions is None
    assert descriptor.get_model() == name
    assert GoogleTextEmbedderDescriptor(model_name=name).get_dimensions() == 3072
    assert GoogleProvider().get_text_embedder(model=name, dimensions=128).get_dimensions() == 128


@pytest.mark.parametrize("prefix", _GOOGLE_MODEL_PREFIXES)
@pytest.mark.parametrize(
    "dimensions,options",
    [
        (1, {}),
        (3073, {}),
        (128, {"input_type": "query"}),
        (128, {"task_type": "RETRIEVAL_DOCUMENT", "title": "document"}),
    ],
)
def test_google_resource_names_reject_unsupported_embedding_options(prefix, dimensions, options):
    from vane.ai.providers.google import GoogleProvider, GoogleTextEmbedderDescriptor

    name = prefix + "gemini-embedding-2"
    with pytest.raises(ValueError, match="output dimensionality|does not support embedding"):
        GoogleProvider().get_text_embedder(model=name, dimensions=dimensions, options=options)
    with pytest.raises(ValueError, match="output dimensionality|does not support embedding"):
        GoogleTextEmbedderDescriptor(model_name=name, dimensions=dimensions, options=options)


@pytest.mark.parametrize("prefix", _GOOGLE_MODEL_PREFIXES)
def test_google_resource_names_share_prompt_and_task_capabilities(prefix):
    from vane.ai.providers.google import GoogleProvider

    provider = GoogleProvider()
    descriptor = provider.get_text_embedder(model=prefix + "gemini-embedding-001", options={"input_type": "query"})
    assert descriptor.options == {"task_type": "RETRIEVAL_QUERY"}
    with pytest.raises(ValueError, match="supports Embed, not Prompt"):
        provider.get_prompter(model=prefix + "gemini-embedding-2")
    with pytest.raises(ValueError, match="supports Prompt, not Embed"):
        provider.get_text_embedder(model=prefix + "gemini-3.6-flash", dimensions=128)
    with pytest.raises(ValueError, match="does not support options"):
        provider.get_prompter(model=prefix + "gemini-3.6-flash", options={"temperature": 0.5})


@pytest.mark.parametrize(
    "name",
    [
        "publishers/other/models/gemini-embedding-2",
        "projects/test/locations/global/publishers/other/models/gemini-embedding-2",
        "projects/test/locations/global/models/gemini-embedding-2",
        "tunedModels/gemini-embedding-2",
        "custom/gemini-embedding-2",
        "models/custom/gemini-embedding-2",
        "projects//locations/global/publishers/google/models/gemini-embedding-2",
    ],
)
def test_google_custom_resources_do_not_inherit_known_model_metadata(name):
    from vane.ai.providers.google import GoogleProvider

    with pytest.raises(ValueError, match="Cannot derive embedding dimensions"):
        GoogleProvider().get_text_embedder(model=name)
    descriptor = GoogleProvider().get_text_embedder(model=name, dimensions=1, options={"input_type": "query"})
    assert descriptor.get_dimensions() == 1
    assert descriptor.get_model() == name
    assert descriptor.options == {"task_type": "RETRIEVAL_QUERY"}


def test_google_query_mapping_respects_model_capability():
    from vane.ai.providers.google import GoogleProvider

    descriptor = GoogleProvider().get_text_embedder(model="gemini-embedding-001", options={"input_type": "query"})
    assert descriptor.options == {"task_type": "RETRIEVAL_QUERY"}
    with pytest.raises(ValueError, match="task_type"):
        GoogleProvider().get_text_embedder(model="gemini-embedding-2", options={"input_type": "query"})


@pytest.mark.parametrize("policy", [None, "error", "truncate", "chunk_mean"])
def test_explicit_long_text_and_legacy_normalization(policy):
    async def request(texts):
        return [np.array([len(text), 0]) for text in texts]

    embedder = _openai(request, policy=policy)
    embedder._input_text_token_limit = 3
    result = _drive(_wrapper(embedder, on_error="ignore"), ["abcdef", "xy"])
    expected = {None: [1, 0], "error": None, "truncate": [3, 0], "chunk_mean": [3, 0]}
    assert result == [expected[policy], [2, 0]]


def test_failed_chunk_nulls_document_without_replaying_other_chunks():
    calls = []

    class AuthError(Exception):
        status_code = 401

    async def request(texts):
        calls.extend(texts)
        if texts == ["def"]:
            raise AuthError()
        return [np.array([1, 2]) for text in texts]

    embedder = _openai(request, request_size=1, policy="chunk_mean")
    embedder._input_text_token_limit = 3
    assert _drive(_wrapper(embedder, on_error="ignore"), ["abcdef", "xy"]) == [None, [1, 2]]
    assert calls == ["abc", "def", "xy"]


def test_compatible_endpoint_cannot_claim_an_unknown_tokenizer():
    with pytest.raises(ValueError, match="tokenizer"):
        OpenAITextEmbedderDescriptor(
            model_name="text-embedding-3-small",
            dimensions=2,
            options={"base_url": "http://localhost:8000/v1", "overlength": "truncate"},
        )


@pytest.mark.parametrize(
    "text,fitting",
    [
        ("abcd", {"a", "abc", "d"}),
        ("abcde", {"a", "ab", "bc", "de"}),
        ("abcdef", {"ab", "abcde", "c", "cd", "def"}),
        ("a🙂bc", {"a", "a🙂b", "c"}),
    ],
)
def test_chunking_backtracks_when_nonmonotonic_prefix_strands_the_tail(text, fitting):
    from vane.ai._embedding_inputs import split_text

    pieces = split_text(text, 1, lambda value: 1 if value in fitting else 2)
    assert "".join(pieces) == text
    assert all(piece in fitting for piece in pieces)


def test_nonmonotonic_chunking_matches_exhaustive_small_partitions():
    import itertools
    import random

    from vane.ai._embedding_inputs import split_text

    text = "abcde"
    spans = [text[start:end] for start in range(len(text)) for end in range(start + 1, len(text) + 1)]
    generator = random.Random(0)
    for _ in range(64):
        fitting = {span for span in spans if generator.random() < 0.25}
        partitions = (
            [text[start:end] for start, end in zip((0, *cuts), (*cuts, len(text)), strict=True)]
            for count in range(len(text))
            for cuts in itertools.combinations(range(1, len(text)), count)
        )
        if any(all(piece in fitting for piece in pieces) for pieces in partitions):
            pieces = split_text(text, 1, lambda value: 1 if value in fitting else 2)
            assert "".join(pieces) == text
            assert all(piece in fitting for piece in pieces)
        else:
            with pytest.raises(ValueError, match="cannot fit an input chunk"):
                split_text(text, 1, lambda value: 1 if value in fitting else 2)


def _fake_text_module(tokenizer, max_seq_length):
    def preprocess(texts):
        return [tokenizer.encode(text) for text in texts]

    # Represent the supported upstream implementation without an optional SDK.
    preprocess.__module__ = "sentence_transformers.base.modules.transformer"
    preprocess.__qualname__ = "Transformer.preprocess"
    return SimpleNamespace(tokenizer=tokenizer, max_seq_length=max_seq_length, preprocess=preprocess)


def _fake_transformers(monkeypatch):
    class Tokenizer:
        def encode(self, text, *, add_special_tokens=True, **kwargs):
            return list(text) + (["BOS", "EOS"] if add_special_tokens else [])

    class Model:
        prompts = {"query": "q:", "document": "d:", "custom": "c:"}
        default_prompt_name = "query"
        tokenizer = Tokenizer()
        max_seq_length = 7

        def __init__(self, *args, **kwargs):
            self.calls = []

        def eval(self):
            pass

        def _first_module(self):
            return _fake_text_module(self.tokenizer, self.max_seq_length)

        def encode(self, texts, **kwargs):
            self.calls.append((texts, kwargs))
            return np.array([[len(text), 1] for text in texts])

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    return Model


@pytest.mark.parametrize("policy", ["error", "truncate", "chunk_mean"])
def test_transformers_counts_prompt_and_special_tokens(monkeypatch, policy):
    from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

    _fake_transformers(monkeypatch)
    embedder = TransformersTextEmbedderDescriptor(
        model="fake", dimensions=2, options={"input_type": "document", "overlength": policy}
    ).instantiate()
    result = _drive(_wrapper(embedder, on_error="ignore"), ["abcdef", "xy"])
    assert result == [None if policy == "error" else [3, 1], [2, 1]]
    assert all(options["prompt_name"] == "document" for _, options in embedder.model.calls)
    assert all(len(text) <= 3 for texts, _ in embedder.model.calls for text in texts)


@pytest.mark.parametrize("on_error", ["raise", "ignore"])
def test_transformers_chunk_mean_retains_nonmonotonic_tail(monkeypatch, on_error):
    from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

    model = _fake_transformers(monkeypatch)
    model.max_seq_length = 1
    model.prompts = {"query": ""}
    model.tokenizer.encode = lambda text, **kwargs: [] if not text else [0] * (1 if text in {"a", "abc", "d"} else 2)
    embedder = TransformersTextEmbedderDescriptor(
        model="nonmonotonic", dimensions=2, options={"overlength": "chunk_mean"}
    ).instantiate()
    assert _drive(_wrapper(embedder, on_error=on_error), ["abcd"]) == [[2, 1]]
    assert embedder.model.calls[0][0] == ["abc", "d"]


def _fake_whitespace_transformers(monkeypatch, prefix):
    base = _fake_transformers(monkeypatch)

    class Tokenizer:
        def encode(self, text, *, add_special_tokens=True, **kwargs):
            # A leading-space BPE merge can make stripped text longer in tokens.
            tokens = [" ab", *text[3:]] if text.startswith(" ab") else list(text)
            return ["BOS", *tokens, "EOS"] if add_special_tokens else tokens

    class LegacyInput:
        tokenizer = Tokenizer()
        max_seq_length = 6
        do_lower_case = False

        def tokenize(self, texts):
            return [self.tokenizer.encode(text.strip()) for text in texts]

    class ModernInput(LegacyInput):
        def preprocess(self, texts):
            return [self.tokenizer.encode(text) for text in texts]

    # Represent the supported upstream implementations without an optional SDK.
    LegacyInput.tokenize.__module__ = "sentence_transformers.models.Transformer"
    LegacyInput.tokenize.__qualname__ = "Transformer.tokenize"
    ModernInput.preprocess.__module__ = "sentence_transformers.base.modules.transformer"
    ModernInput.preprocess.__qualname__ = "Transformer.preprocess"

    query, document = LegacyInput(), ModernInput()
    router = SimpleNamespace(
        sub_modules={"query": [query], "document": [document]},
        _resolve_route=lambda task, modality: task,
    )

    class Model(base):
        prompts = {"query": prefix, "document": prefix}

        def _first_module(self):
            return router

        def encode_query(self, texts, **kwargs):
            return self._encode_with(texts, query.tokenize)

        def encode_document(self, texts, **kwargs):
            return self._encode_with(texts, document.preprocess)

        def _encode_with(self, texts, preprocess):
            full = preprocess([prefix + text for text in texts])
            retained = [tokens[:6] for tokens in full]
            self.calls.append((list(texts), full, retained))
            return np.array([[len(tokens), 1] for tokens in retained])

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    return query, document


@pytest.mark.parametrize("policy", ["error", "truncate", "chunk_mean"])
@pytest.mark.parametrize("input_type", ["query", "document"])
@pytest.mark.parametrize("prefix", ["", " "])
def test_transformers_counts_selected_preprocessing_after_prompt(monkeypatch, policy, input_type, prefix):
    from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

    _fake_whitespace_transformers(monkeypatch, prefix)
    text = "abcde" if prefix else " abcde"
    embedder = TransformersTextEmbedderDescriptor(
        model="whitespace", dimensions=2, options={"input_type": input_type, "overlength": policy}
    ).instantiate()
    result = _drive(_wrapper(embedder, on_error="ignore"), [text])
    if input_type == "query" and policy == "error":
        assert result == [None]
        assert not embedder.model.calls
    else:
        expected_length = 5.4 if input_type == "query" and policy == "chunk_mean" else 6
        assert result[0] == pytest.approx([expected_length, 1])
        assert all(full == retained for _, full, retained in embedder.model.calls)
        submitted = "".join(text for texts, _, _ in embedder.model.calls for text in texts)
        if policy == "chunk_mean" or input_type == "document":
            assert submitted == text
        else:
            assert submitted.strip() == "abcd"


@pytest.mark.parametrize("method", ["tokenize", "preprocess"])
def test_unknown_text_preprocessing_rejects_explicit_policy(monkeypatch, method):
    from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

    query, document = _fake_whitespace_transformers(monkeypatch, "")
    module = query if method == "tokenize" else document
    setattr(module, method, lambda texts: texts)
    descriptor = TransformersTextEmbedderDescriptor(
        model="custom",
        dimensions=2,
        options={"input_type": "query" if method == "tokenize" else "document", "overlength": "error"},
    )
    with pytest.raises(EmbeddingConfigurationError):
        _drive(_EmbedTextBatch(descriptor, "text", "embedding", 2, on_error="ignore"), [" abcde"])


def _fake_preprocessing_transformers(monkeypatch, settings, actual_limit):
    base = _fake_transformers(monkeypatch)
    module = _fake_text_module(base.tokenizer, 10)
    module.processing_kwargs = {}
    module.__dict__.update(settings)

    class Model(base):
        prompts = {"query": "", "document": ""}
        input_module = module

        def _first_module(self):
            return self.input_module

        def encode_query(self, texts, **kwargs):
            return self.encode(texts, task="query", **kwargs)

        def encode_document(self, texts, **kwargs):
            return self.encode(texts, task="document", **kwargs)

        def encode(self, texts, **kwargs):
            retained = [text[: actual_limit - 2] for text in texts]
            self.calls.append((list(texts), retained))
            return np.array([[len(text), sum(map(ord, text))] for text in retained])

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    return Model


@pytest.mark.parametrize("policy", [None, "error", "truncate", "chunk_mean"])
@pytest.mark.parametrize("method", ["missing", "non_callable"])
def test_unavailable_text_preprocessing_rejects_explicit_policy_even_with_ignore(monkeypatch, method, policy):
    from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

    model = _fake_preprocessing_transformers(monkeypatch, {}, 6)
    del model.input_module.preprocess
    if method == "non_callable":
        model.input_module.preprocess = "private preprocessing diagnostic"
        model.input_module.tokenize = "private preprocessing diagnostic"
    descriptor = TransformersTextEmbedderDescriptor(
        model="custom", dimensions=2, options={} if policy is None else {"overlength": policy}
    )
    wrapper = _EmbedTextBatch(descriptor, "text", "embedding", 2, on_error="ignore")
    if policy is None:
        assert _drive(wrapper, ["abcd"]) == [[4, 394]]
    else:
        with pytest.raises(EmbeddingConfigurationError, match="effective preprocessing token budget") as caught:
            _drive(wrapper, ["abcd"])
        assert "private" not in str(caught.value)
        assert caught.value.__context__ is None


@pytest.mark.parametrize("policy", ["error", "truncate", "chunk_mean"])
@pytest.mark.parametrize(
    "input_type,settings,limit",
    [
        ("query", {"query_length": 6}, 6),
        ("document", {"document_length": 6}, 6),
        ("query", {"query_length": 4, "processing_kwargs": {"text": {"max_length": 6}}}, 6),
        (
            "query",
            {"query_length": 4, "processing_kwargs": {"text": {"max_length": 8}, "common": {"max_length": 6}}},
            6,
        ),
        ("query", {"query_length": 6, "processing_kwargs": {"text": {"max_length": None}}}, 10),
        ("query", {"query_length": 6, "processing_kwargs": {"common": {"max_length": None}}}, 10),
        (None, {"query_length": 6}, 10),
        ("query", {"max_seq_length": None, "query_length": 6}, 6),
    ],
)
def test_transformers_uses_effective_preprocessing_limit(monkeypatch, policy, input_type, settings, limit):
    from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

    _fake_preprocessing_transformers(monkeypatch, settings, limit)
    options = {"overlength": policy}
    if input_type is not None:
        options["input_type"] = input_type
    embedder = TransformersTextEmbedderDescriptor(model="configured", dimensions=2, options=options).instantiate()
    result = _drive(_wrapper(embedder, on_error="ignore"), ["abcdef"])
    if limit == 10:
        assert result == [[6, 597]]
    elif policy == "error":
        assert result == [None]
        assert not embedder.model.calls
    else:
        expected = [4, 394] if policy == "truncate" else [10 / 3, (394 * 4 + 203 * 2) / 6]
        assert result[0] == pytest.approx(expected)
        submitted = embedder.model.calls[0][0]
        assert "".join(submitted) == ("abcd" if policy == "truncate" else "abcdef")
    assert all(submitted == retained for submitted, retained in embedder.model.calls)


@pytest.mark.parametrize(
    "settings",
    [
        {"query_length": 0},
        {"query_length": True},
        {"processing_kwargs": {"text": {"max_length": "6"}}},
        {"processing_kwargs": {"common": None}},
        {"processing_kwargs": {"text": {"add_special_tokens": False}}},
        {"processing_kwargs": {"chat_template": {"max_length": 6}}},
        {"query_expansion": {"strategy": "fixed", "length": 6}},
        {"modality_config": {"text": {}, "message": {}}},
        {"processor": object()},
    ],
)
def test_unknown_preprocessing_budget_is_configuration_error_even_with_ignore(monkeypatch, settings):
    from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

    _fake_preprocessing_transformers(monkeypatch, settings, 6)
    descriptor = TransformersTextEmbedderDescriptor(
        model="configured", dimensions=2, options={"input_type": "query", "overlength": "chunk_mean"}
    )
    with pytest.raises(EmbeddingConfigurationError):
        _drive(_EmbedTextBatch(descriptor, "text", "embedding", 2, on_error="ignore"), ["abcdef"])
    TransformersTextEmbedderDescriptor(model="configured", dimensions=2, options={"input_type": "query"}).instantiate()


@pytest.mark.parametrize("forwards_task", [False, True])
def test_transformers_route_preprocessing_receives_task_only_when_forwarded(monkeypatch, forwards_task):
    from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

    model = _fake_preprocessing_transformers(monkeypatch, {"query_length": 6}, 6 if forwards_task else 10)
    router = SimpleNamespace(sub_modules={"query": [model.input_module]}, default_route="query")
    if forwards_task:
        router._resolve_route = lambda task, modality: "query"
    model._first_module = lambda self: router
    embedder = TransformersTextEmbedderDescriptor(
        model="routed", dimensions=2, options={"input_type": "query", "overlength": "error"}
    ).instantiate()
    assert _drive(_wrapper(embedder, on_error="ignore"), ["abcdef"]) == ([None] if forwards_task else [[6, 597]])


def _fake_routed_transformers(monkeypatch, layout="legacy"):
    model = _fake_transformers(monkeypatch)

    class ByteTokenizer:
        def encode(self, text, *, add_special_tokens=True, **kwargs):
            return list(text.encode("utf-8")) + (["BOS", "EOS"] if add_special_tokens else [])

    query = _fake_text_module(ByteTokenizer(), 6)
    document = _fake_text_module(model.tokenizer, 10)

    class Router:
        default_route = "document"

        def __init__(self, query_module):
            self.sub_modules = {"document": [document], "query": [query_module]}

        @property
        def tokenizer(self):
            return next(iter(self.sub_modules.values()))[0].tokenizer

        @property
        def max_seq_length(self):
            return max(route[0].max_seq_length for route in self.sub_modules.values())

    class MappedRouter(Router):
        route_mappings = {("query", "text"): "query_text"}

        def __init__(self, query_module=query):
            super().__init__(document)  # Direct task lookup selects the wrong budget.
            self.sub_modules["query_text"] = [query_module]

        def _resolve_route(self, task=None, modality=None):
            return self.route_mappings.get((task, modality), task or self.default_route)

    router = (
        MappedRouter(Router(query)) if layout == "nested" else MappedRouter() if layout == "mapped" else Router(query)
    )

    class RoutedModel(model):
        def _first_module(self):
            return self.router

        @property
        def tokenizer(self):
            return self.router.tokenizer

        @property
        def max_seq_length(self):
            return self.router.max_seq_length

        def encode_query(self, texts, **kwargs):
            return self.encode(texts, task="query", **kwargs)

        def encode_document(self, texts, **kwargs):
            return self.encode(texts, task="document", **kwargs)

        def encode(self, texts, task=None, **kwargs):
            route = self.router
            while hasattr(route, "sub_modules"):
                resolver = getattr(route, "_resolve_route", None)
                name = resolver(task=task, modality="text") if resolver else task or route.default_route
                route = route.sub_modules[name][0]
            prefix = kwargs.get("prompt", self.prompts[kwargs.get("prompt_name", self.default_prompt_name)])
            retained = []
            for text in texts:
                end = 0
                while end < len(text) and len(route.tokenizer.encode(prefix + text[: end + 1])) <= route.max_seq_length:
                    end += 1
                retained.append(text[:end])
            self.calls.append((list(texts), retained, task))
            return np.array([[sum(map(ord, text)), 1] for text in retained])

    RoutedModel.router = router
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=RoutedModel))
    return RoutedModel


@pytest.mark.parametrize("layout", ["legacy", "mapped", "nested"])
@pytest.mark.parametrize("input_type", ["query", "document"])
@pytest.mark.parametrize("policy", ["error", "truncate", "chunk_mean"])
def test_transformers_overlength_uses_selected_route(monkeypatch, layout, input_type, policy):
    from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

    _fake_routed_transformers(monkeypatch, layout)
    embedder = TransformersTextEmbedderDescriptor(
        model="routed", dimensions=2, options={"input_type": input_type, "overlength": policy}
    ).instantiate()
    result = _drive(_wrapper(embedder, on_error="ignore"), ["aébc"])
    if input_type == "document":
        assert result == [[527, 1]]
    elif policy == "error":
        assert result == [None]
        assert not embedder.model.calls
    else:
        assert result[0] == pytest.approx([97 if policy == "truncate" else 191.4, 1])
        submitted = embedder.model.calls[0][0]
        assert "".join(submitted) == ("a" if policy == "truncate" else "aébc")
    assert all(submitted == retained for submitted, retained, _ in embedder.model.calls)


@pytest.mark.parametrize("options", [{}, {"prompt_name": "query"}, {"input_type": "query"}])
def test_transformers_prompt_or_encode_fallback_does_not_select_a_query_route(monkeypatch, options):
    from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

    model = _fake_routed_transformers(monkeypatch)
    delattr(model, "encode_query")  # Older models fall back to encode with a prompt.
    embedder = TransformersTextEmbedderDescriptor(
        model="routed", dimensions=2, options={**options, "overlength": "error"}
    ).instantiate()
    assert _drive(_wrapper(embedder), ["aébc"]) == [[527, 1]]
    assert embedder.model.calls[0][2] is None


@pytest.mark.parametrize("missing", ["tokenizer", "limit", "route", "resolver", "structure", "nested_legacy", "cycle"])
def test_unresolved_transformers_route_is_configuration_error_even_with_ignore(monkeypatch, missing):
    from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

    model = _fake_routed_transformers(monkeypatch)
    router = model.router
    query = router.sub_modules["query"][0]
    if missing == "tokenizer":
        query.tokenizer = None
    elif missing == "limit":
        query.max_seq_length = None
    elif missing == "route":
        del router.sub_modules["query"]
    elif missing == "resolver":
        router.route_mappings = {("query", "text"): "query"}
    elif missing == "structure":
        model._first_module = None
    elif missing == "nested_legacy":
        router.sub_modules["query"] = [type(router)(query)]
    else:
        router.sub_modules["query"] = [router]
    descriptor = TransformersTextEmbedderDescriptor(
        model="routed", dimensions=2, options={"input_type": "query", "overlength": "error"}
    )
    with pytest.raises(EmbeddingConfigurationError, match="selected input route"):
        _drive(_EmbedTextBatch(descriptor, "text", "embedding", 2, on_error="ignore"), ["abc"])
    # Omitting a policy keeps the existing model initialization contract.
    TransformersTextEmbedderDescriptor(model="routed", dimensions=2, options={"input_type": "query"}).instantiate()


def test_unknown_template_is_configuration_error_even_with_ignore(monkeypatch):
    from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

    _fake_transformers(monkeypatch)
    descriptor = TransformersTextEmbedderDescriptor(model="fake", dimensions=2, options={"prompt_name": "missing"})
    with pytest.raises(EmbeddingConfigurationError):
        _drive(_EmbedTextBatch(descriptor, "text", "embedding", 2, on_error="ignore"), ["abc"])


def test_missing_tokenizer_is_configuration_error_even_with_ignore(monkeypatch):
    from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

    model = _fake_transformers(monkeypatch)
    model.tokenizer = None
    descriptor = TransformersTextEmbedderDescriptor(model="fake", dimensions=2, options={"overlength": "truncate"})
    with pytest.raises(EmbeddingConfigurationError):
        _drive(_EmbedTextBatch(descriptor, "text", "embedding", 2, on_error="ignore"), ["abc"])


@pytest.mark.parametrize("tokenized", [False, True])
@pytest.mark.parametrize("on_error", ["raise", "ignore"])
def test_openai_truncate_does_not_process_discarded_unicode_tail(tokenized, on_error):
    from vane.ai.providers.openai import _TokenEstimator

    class ByteTokenizer:
        n_vocab = 256

        def encode_ordinary(self, value):
            return list(value.encode("utf-8"))

        def decode_single_token_bytes(self, token):
            return bytes([token])

    request = AsyncMock(return_value=[np.array([1, 0])])
    embedder = _openai(request, policy="truncate")
    embedder._estimate_tokens = _TokenEstimator(ByteTokenizer() if tokenized else None)
    embedder._input_text_token_limit = 1
    assert _drive(_wrapper(embedder, on_error=on_error), ["a🙂"]) == [[1, 0]]
    assert request.await_args.args == (["a"],)


def test_transformers_truncate_does_not_process_discarded_unicode_tail(monkeypatch):
    from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

    model = _fake_transformers(monkeypatch)
    model.max_seq_length = 5  # q: plus two special tokens leaves one token.
    model.tokenizer.encode = lambda text, add_special_tokens=True, **kwargs: (
        list(text.encode("utf-8")) + (["BOS", "EOS"] if add_special_tokens else [])
    )
    embedder = TransformersTextEmbedderDescriptor(
        model="fake", dimensions=2, options={"overlength": "truncate"}
    ).instantiate()
    assert _drive(_wrapper(embedder), ["a🙂"]) == [[1, 1]]
    assert embedder.model.calls[0][0] == ["a"]


@pytest.mark.parametrize(
    "provider", ["openai", "openai-base64", "openai-base64-junk", "google", "google-null-item", "google-missing-item"]
)
@pytest.mark.parametrize("on_error", ["raise", "ignore"])
def test_malformed_sdk_vector_preserves_good_rows_without_reissuing_request(monkeypatch, provider, on_error):
    from vane.ai.provider import _ProviderResultError

    if provider.startswith("openai"):
        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAIError=type("SDKError", (Exception,), {})))
        values = [[1.0, 2.0], ["not-a-number"]]
        if provider.startswith("openai-base64"):
            values = [base64.b64encode(np.array([1, 2], dtype="<f4").tobytes()).decode(), "a"]
            if provider == "openai-base64-junk":
                values[1] = "!" + values[0]
        request = AsyncMock(
            return_value=SimpleNamespace(
                data=[SimpleNamespace(index=i, embedding=value) for i, value in enumerate(values)], usage=None
            )
        )
        embedder = _openai(None)
        del embedder._embed_batch
        embedder._encoding_format = "base64" if provider.startswith("openai-base64") else "float"
        embedder._client = SimpleNamespace(embeddings=SimpleNamespace(create=request), close=AsyncMock())
    else:
        from vane.ai.providers.google import GoogleTextEmbedder

        fake_types = SimpleNamespace(
            Content=lambda **kwargs: SimpleNamespace(**kwargs),
            Part=SimpleNamespace(from_text=lambda **kwargs: SimpleNamespace(**kwargs)),
        )
        genai = SimpleNamespace(types=fake_types)
        monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=genai))
        monkeypatch.setitem(sys.modules, "google.genai", genai)
        bad_item = SimpleNamespace(values=["not-a-number"])
        if provider == "google-null-item":
            bad_item = None
        elif provider == "google-missing-item":
            bad_item = SimpleNamespace()
        request = AsyncMock(return_value=SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 2.0]), bad_item]))
        embedder = GoogleTextEmbedder.__new__(GoogleTextEmbedder)
        embedder._client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(embed_content=request), aclose=AsyncMock())
        )
        embedder._model = "fake"
        embedder._dimensions = 2
        embedder._options = {}

    if on_error == "ignore":
        assert _drive(_wrapper(embedder, on_error="ignore"), ["ok", "bad"]) == [[1, 2], None]
    else:
        with pytest.raises(_ProviderResultError, match="cannot be decoded") as caught:
            _drive(_wrapper(embedder), ["ok", "bad"])
        assert caught.value.__context__ is None
    assert request.await_count == 1


@pytest.mark.parametrize("encoding", ["float", "base64"])
@pytest.mark.parametrize("on_error", ["raise", "ignore"])
@pytest.mark.parametrize("item", [None, SimpleNamespace(), "private response item", [], 42, True])
def test_malformed_openai_item_preserves_neighbors_without_reissuing_request(monkeypatch, encoding, on_error, item):
    from vane.ai.provider import _ProviderResultError

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAIError=type("SDKError", (Exception,), {})))
    vector = [1.0, 2.0]
    if encoding == "base64":
        vector = base64.b64encode(np.array(vector, dtype="<f4").tobytes()).decode()
    request = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(embedding=vector), item, SimpleNamespace(embedding=vector)])
    )
    embedder = _openai(None, request_size=3)
    del embedder._embed_batch
    embedder._encoding_format = encoding
    embedder._client = SimpleNamespace(embeddings=SimpleNamespace(create=request), close=AsyncMock())
    wrapper = _wrapper(embedder, on_error=on_error, max_retries=3)
    if on_error == "ignore":
        assert _drive(wrapper, ["ok", None, "bad", "yes"]) == [[1, 2], None, None, [1, 2]]
        assert embedder.metrics.failed_inputs == 1
    else:
        with pytest.raises(_ProviderResultError, match="cannot be decoded") as caught:
            _drive(wrapper, ["ok", None, "bad", "yes"])
        assert caught.value.__context__ is None
        assert "private" not in str(caught.value)
    assert request.await_count == 1
    assert embedder.metrics.retries == 0


def test_new_options_bind_consistently_without_sdk_or_model_loading(monkeypatch):
    import vane
    from vane.ai import embed
    from vane.ai._sql import build_ai_embed_sql_spec

    monkeypatch.setattr(OpenAITextEmbedderDescriptor, "instantiate", lambda _: pytest.fail("binding must not execute"))
    options = {
        "base_url": "http://localhost:8000/v1",
        "supports_overriding_dimensions": False,
        "request_batch_size": 2,
        "max_concurrency_per_actor": 3,
    }
    assert build_ai_embed_sql_spec(model="fixed", dimensions=2, options=options)["return_type"] == "FLOAT[2]"
    connection = vane.connect()
    try:
        relation = connection.sql("SELECT 'abc' AS text")
        expression = embed(vane.col("text"), model="fixed", dimensions=2, **options)
        assert str(relation.select(expression).types[0]) == "FLOAT[2]"
        assert str(relation.embed(vane.col("text"), model="fixed", dimensions=2, **options).types[-1]) == "FLOAT[2]"
        sql = """SELECT ai_embed('abc', model := 'fixed', dimensions := 2,
            options := {'base_url': 'http://localhost:8000/v1',
                        'supports_overriding_dimensions': false,
                        'request_batch_size': 2, 'max_concurrency_per_actor': 3})"""
        assert str(connection.sql(sql).types[0]) == "FLOAT[2]"
        connection.sql("EXPLAIN " + sql).fetchall()
    finally:
        connection.close()
