# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pytest

import vane
from vane.ai import provider as provider_registry
from vane.ai.protocols import PrompterDescriptor, TextEmbedderDescriptor
from vane.ai.provider import Provider
from vane.ai.typing import EmbeddingDimensions, UDFOptions


class MockTextEmbedder:
    def __init__(self, dim: int) -> None:
        self._dim = dim

    def embed_text(self, text: list[str]) -> list[np.ndarray]:
        return [np.ones(self._dim, dtype=np.float32) * float(len(item)) for item in text]


@dataclass
class MockTextEmbedderDescriptor(TextEmbedderDescriptor):
    dim: int
    actor_number: int | None = None
    max_api_concurrency: int | None = None

    def get_provider(self) -> str:
        return "mock"

    def get_model(self) -> str:
        return "mock-embedding"

    def get_options(self) -> dict[str, object]:
        return {
            "batch_size": 2,
            "actor_number": self.actor_number,
            "max_api_concurrency": self.max_api_concurrency,
        }

    def get_dimensions(self) -> EmbeddingDimensions:
        return EmbeddingDimensions(size=self.dim, dtype=pa.float32())

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(
            actor_number=self.actor_number,
            num_gpus=0,
            max_retries=0,
            on_error="raise",
            batch_size=2,
            max_api_concurrency=self.max_api_concurrency,
        )

    def instantiate(self) -> MockTextEmbedder:
        return MockTextEmbedder(self.dim)


class MockPrompter:
    def prompt_batch(self, text: list[str]) -> list[str]:
        return [f"topic:{item}" for item in text]

    async def prompt(self, messages: tuple[object, ...]) -> str:
        return f"topic:{messages[0]}"


@dataclass
class MockPrompterDescriptor(PrompterDescriptor):
    actor_number: int | None = None
    max_api_concurrency: int | None = None
    num_gpus: float | None = 0

    def get_provider(self) -> str:
        return "mock"

    def get_model(self) -> str:
        return "mock-prompt"

    def get_options(self) -> dict[str, object]:
        return {
            "batch_size": 1,
            "actor_number": self.actor_number,
            "max_api_concurrency": self.max_api_concurrency,
            "num_gpus": self.num_gpus,
        }

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(
            actor_number=self.actor_number,
            num_gpus=self.num_gpus,
            max_retries=0,
            on_error="raise",
            batch_size=1,
            max_api_concurrency=self.max_api_concurrency,
        )

    def instantiate(self) -> MockPrompter:
        return MockPrompter()


class MockProvider(Provider):
    @property
    def name(self) -> str:
        return "mock"

    def get_text_embedder(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        **options: object,
    ) -> TextEmbedderDescriptor:
        return MockTextEmbedderDescriptor(
            dim=dimensions or 4,
            actor_number=options.get("actor_number"),
            max_api_concurrency=options.get("max_api_concurrency"),
        )

    def get_prompter(self, model: str | None = None, **options: object) -> PrompterDescriptor:
        return MockPrompterDescriptor(
            actor_number=options.get("actor_number"),
            max_api_concurrency=options.get("max_api_concurrency"),
            num_gpus=options.get("num_gpus", options.get("gpus_per_actor", 0)),
        )


def test_ai_embed_is_public_expression_api():
    assert callable(vane.ai.embed)

    conn = vane.connect()
    rel = conn.sql("select 'abc'::VARCHAR as text union all select NULL::VARCHAR as text")

    expr = vane.ai.embed(
        vane.col("text"),
        provider=MockProvider(),
        dimensions=4,
    ).alias("embedding")

    rows = rel.select(vane.col("text"), expr).fetchall()
    assert {text: list(embedding) for text, embedding in rows} == {
        "abc": [3.0, 3.0, 3.0, 3.0],
        None: [0.0, 0.0, 0.0, 0.0],
    }


def test_ai_embed_normalize_returns_unit_vectors():
    conn = vane.connect()
    rel = conn.sql("select 'abc'::VARCHAR as text")

    expr = vane.ai.embed(
        vane.col("text"),
        provider=MockProvider(),
        dimensions=4,
        normalize=True,
    ).alias("embedding")

    vector = rel.select(expr).fetchone()[0]
    assert pytest.approx(math.sqrt(sum(item * item for item in vector)), rel=1e-6) == 1.0


def test_ai_embed_accepts_registered_embedding_provider_name(monkeypatch):
    monkeypatch.setitem(provider_registry.PROVIDERS, "mock_ai", lambda name=None, **options: MockProvider())

    expr = vane.ai.embed(vane.col("text"), provider="mock_ai")

    assert expr is not None


def test_ai_embed_rejects_provider_without_text_embedder():
    with pytest.raises((AttributeError, TypeError, ValueError), match=r"get_text_embedder|embedding provider"):
        vane.ai.embed(vane.col("text"), provider="vllm")


def test_embed_zero_fill_fallback_survives_dimension_probe_failure():
    from vane.ai.functions import _EmbedTextBatch

    class FailingEmbedder:
        def embed_text(self, texts):
            raise RuntimeError("endpoint down")

    class FailingDescriptor:
        def instantiate(self):
            return FailingEmbedder()

        def get_dimensions(self):
            raise RuntimeError("dimension probe requires network")

        def get_udf_options(self):
            return UDFOptions(max_retries=0, on_error="ignore")

    wrapper = _EmbedTextBatch(FailingDescriptor(), "text", "embedding", max_retries=0, on_error="ignore")
    out = wrapper(pa.table({"text": ["a", "b"]}))

    assert out.num_rows == 2
    assert out.column("embedding").to_pylist() == [None, None]


def test_ai_prompt_expression_basic():
    conn = vane.connect()
    rel = conn.sql(
        "select chunk from (values (0, 'search'::VARCHAR), (1, 'ranking'::VARCHAR)) t(ord, chunk) order by ord"
    )

    expr = vane.ai.prompt(
        vane.col("chunk"),
        provider=MockProvider(),
    ).alias("topic")

    assert rel.select(expr).fetchall() == [("topic:search",), ("topic:ranking",)]


def test_ai_prompt_keeps_existing_relation_api():
    conn = vane.connect()
    rel = conn.sql("select 'search'::VARCHAR as chunk")

    result = vane.ai.prompt(rel, "chunk", provider=MockProvider())

    assert result.fetchall() == [("topic:search",)]


def test_ai_prompt_rel_keyword_matches_positional_relation_api():
    conn = vane.connect()
    rel = conn.sql("select 'search'::VARCHAR as chunk")

    positional = vane.ai.prompt(rel, "chunk", provider=MockProvider())
    keyword = vane.ai.prompt(rel=rel, column="chunk", provider=MockProvider())

    assert keyword.fetchall() == positional.fetchall() == [("topic:search",)]


def test_ai_prompt_rejects_first_and_rel_together():
    conn = vane.connect()
    rel = conn.sql("select 'search'::VARCHAR as chunk")

    with pytest.raises(TypeError, match=r"first.*rel|rel.*first"):
        vane.ai.prompt(rel, "chunk", rel=rel, provider=MockProvider())


def test_ai_prompt_rel_keyword_requires_column():
    conn = vane.connect()
    rel = conn.sql("select 'search'::VARCHAR as chunk")

    with pytest.raises(TypeError, match="relation API requires a column name"):
        vane.ai.prompt(rel=rel, provider=MockProvider())


def test_ai_prompt_rel_keyword_accepts_relation_only_options():
    conn = vane.connect()
    rel = conn.sql("select 'search'::VARCHAR as chunk")

    result = vane.ai.prompt(
        rel=rel,
        column="chunk",
        output_column="answer",
        provider=MockProvider(),
    )

    assert result.fetchall() == [("topic:search",)]


def test_prompt_expression_rejects_relation_only_kwargs_with_guidance():
    with pytest.raises(TypeError, match="expression API does not support.*output_column.*alias"):
        vane.ai.prompt(vane.col("q"), output_column="answer", provider="openai")

    with pytest.raises(TypeError, match="return_format"):
        vane.ai.prompt(vane.col("q"), return_format=dict, provider="openai")


def test_ai_options_are_public_and_map_concurrency():
    openai_provider_options = vane.ai.OpenAIProviderOptions(concurrency=3, max_api_concurrency=7)
    vllm_provider_options = vane.ai.VLLMProviderOptions(concurrency=2, gpus_per_actor=1)
    anthropic_provider_options = vane.ai.AnthropicProviderOptions(concurrency=4, max_api_concurrency=9)
    google_provider_options = vane.ai.GoogleProviderOptions(concurrency=5, max_api_concurrency=11)

    assert openai_provider_options.to_descriptor_options() == {
        "actor_number": 3,
        "max_api_concurrency": 7,
    }
    assert vllm_provider_options.to_descriptor_options() == {
        "actor_number": 2,
        "gpus_per_actor": 1,
    }
    assert anthropic_provider_options.to_descriptor_options() == {
        "actor_number": 4,
        "max_api_concurrency": 9,
    }
    assert google_provider_options.to_descriptor_options() == {
        "actor_number": 5,
        "max_api_concurrency": 11,
    }


def test_openai_prompt_options_do_not_emit_unset_use_chat_completions():
    assert "use_chat_completions" not in vane.ai.OpenAIPromptOptions().to_descriptor_options()
    assert (
        vane.ai.OpenAIPromptOptions(use_chat_completions=False).to_descriptor_options()["use_chat_completions"] is False
    )


def test_anthropic_and_google_options_are_public_request_mappers():
    anthropic_prompt_options = vane.ai.AnthropicPromptOptions(max_tokens=64, temperature=0, on_error="log")
    google_prompt_options = vane.ai.GooglePromptOptions(max_output_tokens=32, temperature=0, on_error="ignore")
    google_embedding_options = vane.ai.GoogleEmbeddingOptions(task_type="RETRIEVAL_DOCUMENT", on_error="log")

    assert anthropic_prompt_options.to_descriptor_options() == {
        "max_tokens": 64,
        "temperature": 0,
        "on_error": "log",
    }
    assert google_prompt_options.to_descriptor_options() == {
        "max_output_tokens": 32,
        "temperature": 0,
        "on_error": "ignore",
    }
    assert google_embedding_options.to_descriptor_options() == {
        "task_type": "RETRIEVAL_DOCUMENT",
        "on_error": "log",
    }


def test_ai_prompt_expression_explain_uses_native_actor_backend(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    conn = vane.connect()
    rel = conn.sql("select 'search'::VARCHAR as chunk")

    expr = vane.ai.prompt(
        vane.col("chunk"),
        provider=MockProvider(),
        provider_options=vane.ai.OpenAIProviderOptions(concurrency=3, max_api_concurrency=5),
    ).alias("topic")

    plan = rel.select(expr).explain()

    assert "execution_backend:" in plan
    assert "subprocess_actor" in plan
    assert "actor_number:" in plan
    assert "3" in plan


def test_ai_prompt_expression_explain_uses_ray_actor_backend(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    vane.configure(runner="ray")

    conn = vane.connect()
    rel = conn.sql("select 'search'::VARCHAR as chunk")

    expr = vane.ai.prompt(
        vane.col("chunk"),
        provider=MockProvider(),
        provider_options=vane.ai.OpenAIProviderOptions(concurrency=2),
    ).alias("topic")

    plan = rel.select(expr).explain()

    assert "execution_backend:" in plan
    assert "ray_actor" in plan
    assert "actor_number:" in plan
    assert "2" in plan


def test_ai_prompt_vllm_options_map_to_actor_and_gpu_fields(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    conn = vane.connect()
    rel = conn.sql("select 'search'::VARCHAR as chunk")

    expr = vane.ai.prompt(
        vane.col("chunk"),
        provider=MockProvider(),
        provider_options=vane.ai.VLLMProviderOptions(concurrency=2, gpus_per_actor=1),
        prompt_options=vane.ai.VLLMPromptOptions(generate_args={"temperature": 0}),
    ).alias("topic")

    plan = rel.select(expr).explain()

    assert "ray_actor" in plan
    assert "actor_number:" in plan
    assert "2" in plan
    assert "gpus:" in plan
    assert "1" in plan


def test_ai_prompt_expression_rejects_gpu_actor_on_local_runner(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    conn = vane.connect()
    conn.sql("select 1 as id, 'search'::VARCHAR as chunk")

    with pytest.raises(vane.InvalidInputException, match="GPU resources require a Ray UDF backend"):
        vane.ai.prompt(
            vane.col("chunk"),
            provider=MockProvider(),
            provider_options=vane.ai.VLLMProviderOptions(concurrency=1, gpus_per_actor=1),
        ).alias("topic")
