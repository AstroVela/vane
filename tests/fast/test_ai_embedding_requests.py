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

        def _first_module(self):
            return SimpleNamespace(tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)

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
    module = SimpleNamespace(tokenizer=base.tokenizer, max_seq_length=10, processing_kwargs={})
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

    query = SimpleNamespace(tokenizer=ByteTokenizer(), max_seq_length=6)
    document = SimpleNamespace(tokenizer=model.tokenizer, max_seq_length=10)

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
