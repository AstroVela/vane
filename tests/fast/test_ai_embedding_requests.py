# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Embedding contracts using in-process fake models and SDKs, without downloads."""

from __future__ import annotations

import asyncio
import base64
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
    request = AsyncMock(side_effect=error)
    embedder = _openai(request, request_size=64)
    assert _drive(_wrapper(embedder, max_retries=retries, on_error="ignore"), ["a", "b", "c"]) == [None] * 3
    assert request.await_count == expected


def test_input_error_isolation_preserves_good_rows():
    class InputError(Exception):
        status_code = 400

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


def test_invalid_vector_nulls_only_its_row_without_reissue():
    request = AsyncMock(return_value=[np.array([1, 2]), np.array([float("inf"), 2])])
    assert _drive(_wrapper(_openai(request), on_error="ignore"), ["a", "b"]) == [[1, 2], None]
    assert request.await_count == 1


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


@pytest.mark.parametrize("provider", ["openai", "openai-base64", "openai-base64-junk", "google"])
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
        request = AsyncMock(
            return_value=SimpleNamespace(
                embeddings=[SimpleNamespace(values=[1.0, 2.0]), SimpleNamespace(values=["not-a-number"])]
            )
        )
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
