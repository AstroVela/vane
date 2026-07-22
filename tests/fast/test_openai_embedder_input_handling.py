# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Input-handling tests for the OpenAI embedder (issue #144).

Covers empty-row splicing, conservative CJK token estimation, token-limited
batching of oversized-input chunks, float32 dtype consistency, token-limit
validation, and zero-usage metrics recording.
"""

import asyncio
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import vane.ai.metrics as ai_metrics
from vane.ai.providers.openai import OpenAIPrompter, OpenAITextEmbedder


class FakeOpenAIError(Exception):
    pass


class FakeEmbeddingServer:
    """Records embeddings.create requests; embeds text ``t`` as ``[len(t)] * dim``.

    Mirrors the real API's rejection of empty-string inputs: one empty input
    fails the entire request.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        texts = kwargs["input"]
        if any(t == "" for t in texts):
            raise FakeOpenAIError("'$.input' is invalid: empty string")
        dim = kwargs.get("dimensions", self.dim)
        data = [SimpleNamespace(embedding=[float(len(t))] * dim) for t in texts]
        return SimpleNamespace(data=data, usage=None)


def _install_fake_openai(monkeypatch, server):
    def fake_async_openai(**_kwargs):
        return SimpleNamespace(embeddings=SimpleNamespace(create=server.create))

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=fake_async_openai, OpenAIError=FakeOpenAIError),
    )


def _make_embedder(monkeypatch, server, **kwargs):
    _install_fake_openai(monkeypatch, server)
    return OpenAITextEmbedder(provider_options={"api_key": "test-key"}, **kwargs)


def test_empty_and_null_rows_are_spliced_not_sent(monkeypatch):
    server = FakeEmbeddingServer()
    embedder = _make_embedder(monkeypatch, server, model="text-embedding-3-small", dimensions=4)

    result = asyncio.run(embedder.embed_text(["alpha", "", "beta", None, "   \n"]))

    assert [req["input"] for req in server.requests] == [["alpha", "beta"]]
    assert len(result) == 5
    np.testing.assert_array_equal(result[0], np.full(4, 5.0, dtype=np.float32))
    np.testing.assert_array_equal(result[2], np.full(4, 4.0, dtype=np.float32))
    for position in (1, 3, 4):
        assert result[position].dtype == np.float32
        np.testing.assert_array_equal(result[position], np.zeros(4, dtype=np.float32))


def test_all_empty_rows_use_model_dimension_table(monkeypatch):
    server = FakeEmbeddingServer()
    embedder = _make_embedder(monkeypatch, server, model="text-embedding-3-small")

    result = asyncio.run(embedder.embed_text(["", None]))

    assert server.requests == []
    assert len(result) == 2
    for row in result:
        assert row.dtype == np.float32
        assert row.shape == (1536,)
        assert not row.any()


def test_all_empty_rows_unknown_model_fall_back_to_none(monkeypatch):
    server = FakeEmbeddingServer()
    embedder = _make_embedder(monkeypatch, server, model="custom-model")

    result = asyncio.run(embedder.embed_text(["", "  "]))

    assert server.requests == []
    assert result == [None, None]


def test_empty_row_dimension_inferred_from_first_embedding(monkeypatch):
    server = FakeEmbeddingServer(dim=8)
    embedder = _make_embedder(monkeypatch, server, model="custom-model")

    result = asyncio.run(embedder.embed_text(["", "hello"]))

    assert [req["input"] for req in server.requests] == [["hello"]]
    np.testing.assert_array_equal(result[0], np.zeros(8, dtype=np.float32))
    np.testing.assert_array_equal(result[1], np.full(8, 5.0, dtype=np.float32))


def test_cjk_text_estimate_triggers_conservative_chunking(monkeypatch):
    server = FakeEmbeddingServer(dim=4)
    embedder = _make_embedder(
        monkeypatch,
        server,
        model="custom-model",
        input_text_token_limit=100,
    )
    # CJK runs ~1 token per char: 250 real tokens. The old ``len // 3``
    # estimate said 83 tokens and sent it whole, over the limit.
    text = "汉" * 250

    result = asyncio.run(embedder.embed_text([text]))

    sent = [t for req in server.requests for t in req["input"]]
    assert len(sent) > 1  # chunked
    assert all(len(chunk) <= 100 for chunk in sent)  # each chunk within the limit
    assert "".join(sent) == text
    assert len(result) == 1
    assert result[0].dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(result[0]), 1.0, atol=1e-6)


def test_oversized_chunks_respect_batch_token_limit(monkeypatch):
    server = FakeEmbeddingServer(dim=4)
    embedder = _make_embedder(
        monkeypatch,
        server,
        model="custom-model",
        input_text_token_limit=10,
        batch_token_limit=15,
    )
    text = "a" * 90  # 30 estimated tokens: chunked, chunks must not share one request

    result = asyncio.run(embedder.embed_text([text]))

    assert len(server.requests) >= 2
    for req in server.requests:
        assert sum(len(t) for t in req["input"]) // 3 < 15
    assert "".join(t for req in server.requests for t in req["input"]) == text
    assert len(result) == 1


def test_inverted_limits_still_bound_each_request(monkeypatch):
    """input_text_token_limit above batch_token_limit must not produce over-limit requests."""
    server = FakeEmbeddingServer(dim=4)
    embedder = _make_embedder(
        monkeypatch,
        server,
        model="custom-model",
        input_text_token_limit=40,
        batch_token_limit=10,
    )
    text = "a" * 150  # 50 estimated tokens: oversized, but chunks must fit the batch limit

    result = asyncio.run(embedder.embed_text([text]))

    assert len(server.requests) >= 2
    for req in server.requests:
        assert sum(len(t) for t in req["input"]) // 3 <= 10
    assert "".join(t for req in server.requests for t in req["input"]) == text
    assert len(result) == 1


def test_oversized_average_is_float32(monkeypatch):
    server = FakeEmbeddingServer(dim=4)
    embedder = _make_embedder(
        monkeypatch,
        server,
        model="custom-model",
        input_text_token_limit=10,
    )

    result = asyncio.run(embedder.embed_text(["a" * 90]))

    assert len(result) == 1
    assert result[0].dtype == np.float32


def test_row_order_preserved_with_empty_oversized_and_normal_rows(monkeypatch):
    server = FakeEmbeddingServer(dim=4)
    embedder = _make_embedder(
        monkeypatch,
        server,
        model="text-embedding-3-small",
        dimensions=4,
        input_text_token_limit=100,
    )
    oversized = "汉" * 250

    result = asyncio.run(embedder.embed_text(["alpha", "", oversized, "beta"]))

    assert len(result) == 4
    np.testing.assert_array_equal(result[0], np.full(4, 5.0, dtype=np.float32))
    np.testing.assert_array_equal(result[1], np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(np.linalg.norm(result[2]), 1.0, atol=1e-6)
    np.testing.assert_array_equal(result[3], np.full(4, 4.0, dtype=np.float32))
    # Non-empty rows are sent exactly once, in order.
    sent = [t for req in server.requests for t in req["input"]]
    assert sent[0] == "alpha"
    assert sent[-1] == "beta"
    assert "".join(sent[1:-1]) == oversized


@pytest.mark.parametrize(
    "limits",
    [
        {"batch_token_limit": 0},
        {"batch_token_limit": -1},
        {"input_text_token_limit": 0},
        {"input_text_token_limit": -5},
    ],
)
def test_non_positive_token_limits_are_rejected(monkeypatch, limits):
    server = FakeEmbeddingServer()
    _install_fake_openai(monkeypatch, server)

    with pytest.raises(ValueError, match="token_limit"):
        OpenAITextEmbedder(provider_options={"api_key": "test-key"}, model="custom-model", **limits)


def test_prompter_records_legitimate_zero_usage(monkeypatch):
    _install_fake_openai(monkeypatch, FakeEmbeddingServer())
    recorded: list[dict] = []
    monkeypatch.setattr(ai_metrics, "record_token_metrics", lambda **kwargs: recorded.append(kwargs))
    prompter = OpenAIPrompter(provider_options={"api_key": "test-key"}, model="gpt-4o-mini")

    usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    prompter._record_usage(SimpleNamespace(usage=usage))

    assert recorded == [
        {
            "protocol": "prompt",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    ]


def test_prompter_usage_falls_back_to_responses_api_fields(monkeypatch):
    _install_fake_openai(monkeypatch, FakeEmbeddingServer())
    recorded: list[dict] = []
    monkeypatch.setattr(ai_metrics, "record_token_metrics", lambda **kwargs: recorded.append(kwargs))
    prompter = OpenAIPrompter(provider_options={"api_key": "test-key"}, model="gpt-4o-mini")

    usage = SimpleNamespace(input_tokens=30, output_tokens=15, total_tokens=45)
    prompter._record_usage(SimpleNamespace(usage=usage))

    assert recorded[0]["input_tokens"] == 30
    assert recorded[0]["output_tokens"] == 15
    assert recorded[0]["total_tokens"] == 45
