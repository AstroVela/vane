# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from datetime import date

import numpy as np
import pyarrow as pa
import pytest

import duckdb
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

    def get_provider(self) -> str:
        return "mock"

    def get_model(self) -> str:
        return "mock-embedding"

    def get_options(self) -> dict[str, object]:
        return {
            "batch_size": 2,
            "actor_number": self.actor_number,
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
        )

    def get_prompter(self, model: str | None = None, **options: object) -> PrompterDescriptor:
        return MockPrompterDescriptor(
            actor_number=options.get("actor_number"),
            max_api_concurrency=options.get("max_api_concurrency"),
            num_gpus=options.get("num_gpus", options.get("gpus_per_actor", 0)),
        )


class _RecordingNativeVLLMExecutor:
    """Minimal executor used to exercise the native PhysicalVLLM bridge."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self.responses = responses
        self.submissions: list[tuple[str | None, tuple[str, ...]]] = []
        self.ready = deque()
        self.finished = False
        self.finished_count = 0
        self.shutdown_count = 0

    def submit(self, prefix, prompts, rows) -> None:
        prompt_values = tuple(prompts)
        self.submissions.append((prefix, prompt_values))
        output_values = (
            [f"generated:{prompt}" for prompt in prompt_values]
            if self.responses is None
            else [self.responses[prompt] for prompt in prompt_values]
        )
        self.ready.append((output_values, rows))

    def take_ready_result(self):
        try:
            return self.ready.popleft()
        except IndexError:
            return None

    def finished_submitting(self) -> None:
        self.finished = True
        self.finished_count += 1

    def all_tasks_finished(self) -> bool:
        return self.finished and not self.ready

    def wait_for_result(self) -> None:
        pass

    def register_wakeup_callback(self, _callback) -> bool:
        return False

    def shutdown(self) -> None:
        self.finished = True
        self.shutdown_count += 1


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
    assert {text: None if embedding is None else list(embedding) for text, embedding in rows} == {
        "abc": [3.0, 3.0, 3.0, 3.0],
        None: None,
    }


def test_ai_embed_literal_null_is_fixed_type_without_runtime_calls(monkeypatch):
    runtime_calls = []
    original_embed = MockTextEmbedder.embed_text

    def recording_embed(embedder, text):
        runtime_calls.append(list(text))
        return original_embed(embedder, text)

    monkeypatch.setattr(MockTextEmbedder, "embed_text", recording_embed)
    conn = vane.connect()
    source = conn.sql("SELECT 1 AS row_id")

    expression_result = source.select(
        vane.ai.embed(
            duckdb.ConstantExpression(None).cast("VARCHAR"),
            provider=MockProvider(),
            dimensions=4,
        ).alias("embedding")
    )
    relation_result = vane.ai.embed(
        source,
        duckdb.ConstantExpression(None).cast("VARCHAR"),
        provider=MockProvider(),
        dimensions=4,
    ).project("embedding")

    for result in (expression_result, relation_result):
        assert [str(dtype) for dtype in result.types] == ["FLOAT[4]"]
        assert result.fetchall() == [(None,)]
    assert runtime_calls == []


def test_ai_embed_four_public_entries_are_equivalent(monkeypatch):
    monkeypatch.setitem(provider_registry.PROVIDERS, "mock_embed_contract", lambda name=None, **options: MockProvider())
    conn = vane.connect()
    rel = conn.sql("SELECT * FROM (VALUES (1, 'abc'::VARCHAR), (2, NULL::VARCHAR)) AS t(row_id, text)")

    expression_rows = (
        rel.select(
            vane.col("row_id"),
            vane.ai.embed(
                vane.col("text"),
                provider="mock_embed_contract",
                dimensions=4,
            ).alias("embedding"),
        )
        .order("row_id")
        .fetchall()
    )
    functional_rows = (
        vane.ai.embed(
            rel,
            vane.col("text"),
            provider="mock_embed_contract",
            dimensions=4,
        )
        .project("row_id, embedding")
        .order("row_id")
        .fetchall()
    )
    method_rows = (
        rel.embed(
            vane.col("text"),
            provider="mock_embed_contract",
            dimensions=4,
        )
        .project("row_id, embedding")
        .order("row_id")
        .fetchall()
    )
    sql_rows = conn.sql("""
        SELECT row_id, ai_embed(
            text,
            provider := 'mock_embed_contract',
            dimensions := 4
        ) AS embedding
        FROM (VALUES (1, 'abc'::VARCHAR), (2, NULL::VARCHAR)) AS t(row_id, text)
        ORDER BY row_id
    """).fetchall()

    assert expression_rows == functional_rows == method_rows == sql_rows


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


@pytest.mark.parametrize(
    ("provider", "model", "expected_type"),
    [
        ("openai", None, "FLOAT[1536]"),
        ("google", "gemini-embedding-001", "FLOAT[3072]"),
        ("transformers", None, "FLOAT[384]"),
    ],
)
def test_ai_embed_uses_trusted_builtin_dimensions_without_execution(provider, model, expected_type):
    rel = vane.connect().sql("SELECT 'planning only'::VARCHAR AS text")
    expression = vane.ai.embed(vane.col("text"), provider=provider, model=model)

    assert [str(value) for value in rel.select(expression).types] == [expected_type]


def test_ai_embed_accepts_registered_embedding_provider_name(monkeypatch):
    monkeypatch.setitem(provider_registry.PROVIDERS, "mock_ai", lambda name=None, **options: MockProvider())

    expr = vane.ai.embed(vane.col("text"), provider="mock_ai")

    assert expr is not None


def test_ai_embed_rejects_provider_without_text_embedder():
    with pytest.raises((AttributeError, TypeError, ValueError), match=r"get_text_embedder|embedding provider"):
        vane.ai.embed(vane.col("text"), provider="vllm")


def test_ai_embed_rejects_non_provider_objects_and_undocumented_first_keyword():
    with pytest.raises(TypeError, match="Provider object"):
        vane.ai.embed(vane.col("text"), provider=object())
    with pytest.raises((TypeError, ValueError), match="first|text Expression"):
        vane.ai.embed(first=vane.col("text"), provider=MockProvider())


@pytest.mark.parametrize(
    "option",
    ["concurrency", "max_api_concurrency", "max_concurrency_per_actor", "embedding_options"],
)
def test_ai_embed_rejects_removed_or_unknown_options(option):
    with pytest.raises(TypeError, match=option):
        vane.ai.embed(vane.col("text"), provider=MockProvider(), **{option: 2})


@pytest.mark.parametrize("option", ["api_key", "authorization", "access_token"])
def test_ai_embed_rejects_inline_credentials(option):
    secret = "embed-secret-sentinel"
    with pytest.raises(ValueError, match=option) as exc_info:
        vane.ai.embed(vane.col("text"), provider=MockProvider(), **{option: secret})
    assert secret not in str(exc_info.value)


@pytest.mark.parametrize(
    ("provider_kind", "field"),
    [
        ("openai", "api_key"),
        ("openai", "organization"),
        ("google", "api_key"),
    ],
)
def test_ai_embed_rejects_credentials_from_builtin_provider_presets(provider_kind, field):
    from vane.ai.providers.google import GoogleProvider
    from vane.ai.providers.openai import OpenAIProvider

    secret = "embed-provider-secret-sentinel"
    if provider_kind == "openai":
        provider = OpenAIProvider(**{field: secret})
        model = None
        dimensions = 4
    else:
        provider = GoogleProvider(**{field: secret})
        model = "gemini-embedding-001"
        dimensions = 128

    with pytest.raises(ValueError, match=field) as exc_info:
        vane.ai.embed(
            vane.col("text"),
            provider=provider,
            model=model,
            dimensions=dimensions,
        )
    assert secret not in str(exc_info.value)


@pytest.mark.parametrize("dimensions", [True, 0, -1, 1.5, "4"])
def test_ai_embed_rejects_invalid_dimensions(dimensions):
    with pytest.raises(ValueError, match="dimensions"):
        vane.ai.embed(vane.col("text"), provider=MockProvider(), dimensions=dimensions)


def test_ai_embed_rejects_nonfinal_on_error_value():
    with pytest.raises(ValueError, match="on_error"):
        vane.ai.embed(vane.col("text"), provider=MockProvider(), on_error="log")


def test_ai_embed_expression_rejects_relation_only_options():
    with pytest.raises(TypeError, match="execution_backend"):
        vane.ai.embed(
            vane.col("text"),
            provider=MockProvider(),
            execution_backend="subprocess_task",
        )


def test_ai_embed_relation_rejects_actor_count_with_task_backend():
    rel = vane.connect().sql("select 'abc'::VARCHAR as text")
    with pytest.raises(ValueError, match="actor_number.*actor"):
        vane.ai.embed(
            rel,
            vane.col("text"),
            provider=MockProvider(),
            execution_backend="subprocess_task",
            actor_number=2,
        )


def test_ai_embed_relation_rejects_blank_output_column():
    rel = vane.connect().sql("select 'abc'::VARCHAR as text")

    with pytest.raises(ValueError, match="output_column"):
        vane.ai.embed(rel, vane.col("text"), provider=MockProvider(), output_column="   ")


def test_ai_embed_requires_static_dimensions_for_unknown_model():
    from vane.ai.providers.openai import OpenAIProvider

    with pytest.raises(ValueError, match="without network or model loading.*dimensions"):
        vane.ai.embed(
            vane.col("text"),
            provider=OpenAIProvider(),
            model="compatible-endpoint-model",
        )


def test_ai_embed_rejects_known_openai_dimension_above_model_maximum():
    from vane.ai.providers.openai import OpenAIProvider

    with pytest.raises(ValueError, match="at most 1536 dimensions"):
        vane.ai.embed(
            vane.col("text"),
            provider=OpenAIProvider(),
            model="text-embedding-3-small",
            dimensions=1537,
        )


def test_ai_embed_openai_base_url_allows_noncredential_query_but_rejects_credentials():
    from vane.ai.providers.openai import OpenAIProvider

    assert (
        vane.ai.embed(
            vane.col("text"),
            provider=OpenAIProvider(),
            dimensions=4,
            base_url="https://embedding.example.test/v1?region=cn",
        )
        is not None
    )

    secret = "embed-secret-sentinel"
    with pytest.raises(ValueError, match="credentials") as exc_info:
        vane.ai.embed(
            vane.col("text"),
            provider=OpenAIProvider(),
            dimensions=4,
            base_url=f"https://embedding.example.test/v1?api_key={secret}",
        )
    assert secret not in str(exc_info.value)


def test_ai_embed_transformers_trust_remote_code_requires_pinned_revision():
    from vane.ai.providers.transformers import TransformersProvider

    with pytest.raises(ValueError, match="trust_remote_code=True.*revision"):
        vane.ai.embed(
            vane.col("text"),
            provider=TransformersProvider(),
            trust_remote_code=True,
        )


def test_ai_embed_accepts_valid_provider_specific_options():
    from vane.ai.providers.google import GoogleProvider
    from vane.ai.providers.transformers import TransformersProvider

    assert (
        vane.ai.embed(
            vane.col("text"),
            provider=GoogleProvider(),
            model="gemini-embedding-001",
            task_type="RETRIEVAL_DOCUMENT",
            title="Document title",
        )
        is not None
    )
    assert (
        vane.ai.embed(
            vane.col("text"),
            provider=TransformersProvider(),
            revision="reviewed-commit",
            trust_remote_code=True,
        )
        is not None
    )


@pytest.mark.parametrize(
    ("provider", "model", "options", "offending"),
    [
        ("openai", None, {"task_type": "RETRIEVAL_QUERY"}, "task_type"),
        ("google", "gemini-embedding-001", {"encoding_format": "float"}, "encoding_format"),
        ("transformers", None, {"timeout": 3.0}, "timeout"),
    ],
)
def test_ai_embed_rejects_options_from_another_provider(provider, model, options, offending):
    with pytest.raises(TypeError, match=offending):
        vane.ai.embed(vane.col("text"), provider=provider, model=model, **options)


def test_ai_embed_google_title_requires_retrieval_document_task():
    from vane.ai.providers.google import GoogleProvider

    with pytest.raises(ValueError, match="RETRIEVAL_DOCUMENT"):
        vane.ai.embed(
            vane.col("text"),
            provider=GoogleProvider(),
            model="gemini-embedding-001",
            task_type="RETRIEVAL_QUERY",
            title="Invalid title context",
        )


def test_ai_embed_explicit_dimensions_skip_provider_dimension_lookup():
    class Descriptor(MockTextEmbedderDescriptor):
        def get_dimensions(self):
            raise AssertionError("dimension lookup must not run")

    class ExplicitProvider(MockProvider):
        def get_text_embedder(self, model=None, dimensions=None, **options):
            return Descriptor(dim=dimensions or 99)

    rel = vane.connect().sql("select 'abc'::VARCHAR as text")
    expr = vane.ai.embed(vane.col("text"), provider=ExplicitProvider(), dimensions=7)

    assert [str(value) for value in rel.select(expr).types] == ["FLOAT[7]"]


def test_embed_on_error_ignore_returns_fixed_type_nulls():
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

    wrapper = _EmbedTextBatch(FailingDescriptor(), "text", "embedding", 4, max_retries=0, on_error="ignore")
    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        out = wrapper(pa.table({"text": ["a", "b"]}))
    finally:
        loop.close()

    assert out.num_rows == 2
    assert out.column("embedding").to_pylist() == [None, None]
    assert out.schema.field("embedding").type == pa.list_(pa.float32(), 4)


def test_embed_on_error_ignore_isolates_only_failed_rows_and_skips_nulls():
    from vane.ai.functions import _EmbedTextBatch

    class SelectiveEmbedder:
        def embed_text(self, texts):
            if "bad" in texts:
                raise RuntimeError("row rejected")
            return [np.full(3, len(text), dtype=np.float32) for text in texts]

    class Descriptor:
        def get_provider(self):
            return "mock"

        def get_model(self):
            return "selective"

        def instantiate(self):
            return SelectiveEmbedder()

    wrapper = _EmbedTextBatch(Descriptor(), "text", "embedding", 3, max_retries=0, on_error="ignore")
    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        out = wrapper(pa.table({"text": ["ok", "bad", None, "fine"]}))
    finally:
        loop.close()

    assert out.column("embedding").to_pylist() == [
        [2.0, 2.0, 2.0],
        None,
        None,
        [4.0, 4.0, 4.0],
    ]


def test_embed_dynamic_capability_error_is_preserved_and_not_retried_per_row():
    from vane.ai.functions import _EmbedTextBatch

    calls = []
    original = RuntimeError("endpoint rejects embedding model")

    class UnsupportedEmbedder:
        def embed_text(self, texts):
            calls.append(list(texts))
            raise vane.ai.ProviderCapabilityError(
                "mock",
                "chat-only-model",
                "embedding endpoint/model",
                original_error=original,
            ) from original

    class Descriptor:
        def instantiate(self):
            return UnsupportedEmbedder()

    wrapper = _EmbedTextBatch(Descriptor(), "text", "embedding", 3, max_retries=3, on_error="ignore")
    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        out = wrapper(pa.table({"text": ["first", "second"]}))
    finally:
        loop.close()

    assert out.column("embedding").to_pylist() == [None, None]
    assert calls == [["first", "second"]]

    error = vane.ai.ProviderCapabilityError(
        "mock",
        "chat-only-model",
        "embedding endpoint/model",
        original_error=original,
    )
    assert (error.provider, error.model, error.capability, error.original_error) == (
        "mock",
        "chat-only-model",
        "embedding endpoint/model",
        original,
    )


def test_embed_dynamic_dimension_mismatch_is_a_result_type_error():
    from vane.ai.functions import _EmbedTextBatch

    class WrongDimensionEmbedder:
        def embed_text(self, texts):
            return [np.ones(2, dtype=np.float32) for _ in texts]

    class Descriptor:
        def get_provider(self):
            return "mock"

        def get_model(self):
            return "dynamic-model"

        def instantiate(self):
            return WrongDimensionEmbedder()

    wrapper = _EmbedTextBatch(Descriptor(), "text", "embedding", 3, max_retries=0)
    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        with pytest.raises(TypeError, match="length 2; expected 3"):
            wrapper(pa.table({"text": ["abc"]}))
    finally:
        loop.close()


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


def test_vllm_descriptor_builds_native_options_without_mutating_inputs():
    from vane.ai.providers.vllm import VLLMPrompterDescriptor

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    caller_options = {
        "actor_number": 4,
        "on_error": "ignore",
        "temperature": 0.25,
        "engine_args": {"max_model_len": 2048},
        "generate_args": {"sampling_params": '{"max_tokens":32}'},
    }
    descriptor = VLLMPrompterDescriptor(
        model_name="native-model",
        return_format=schema,
        vllm_options=caller_options,
    )

    native_options = descriptor.build_physical_vllm_options()

    assert native_options["concurrency"] == 4
    assert "actor_number" not in native_options
    assert native_options["use_threading"] is True
    assert native_options["_force_background_thread"] is True
    assert native_options["on_error"] == "null"
    sampling_params = native_options["generate_args"]["sampling_params"]
    assert sampling_params["max_tokens"] == 32
    assert sampling_params["temperature"] == 0.25
    assert sampling_params["structured_outputs"] == {"type": "json", "value": schema}
    assert caller_options == {
        "actor_number": 4,
        "on_error": "ignore",
        "temperature": 0.25,
        "engine_args": {"max_model_len": 2048},
        "generate_args": {"sampling_params": '{"max_tokens":32}'},
    }

    sampling_params["max_tokens"] = 1
    sampling_params["structured_outputs"]["value"]["required"].clear()
    assert caller_options["generate_args"]["sampling_params"] == '{"max_tokens":32}'
    assert schema["required"] == ["answer"]

    explicit_concurrency = VLLMPrompterDescriptor(
        vllm_options={"actor_number": 2, "concurrency": 7}
    ).build_physical_vllm_options()
    assert explicit_concurrency["concurrency"] == 7


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        (
            {"engine_args": {"dtype": object()}},
            TypeError,
            r"engine_args\.dtype.*JSON-compatible.*object",
        ),
        (
            {"generate_args": {"temperature": float("nan")}},
            ValueError,
            r"generate_args\.temperature.*finite",
        ),
    ],
)
def test_vllm_native_options_reject_invalid_json_values_early(options, error, message):
    from vane.ai.providers.vllm import VLLMPrompterDescriptor

    descriptor = VLLMPrompterDescriptor(model_name="native-model", vllm_options=options)

    with pytest.raises(error, match=message):
        descriptor.build_physical_vllm_options()


@pytest.mark.parametrize(
    "reserved_key",
    [
        "__vane_vllm_payload_version",
        "__vane_vllm_public_options_json",
        "__vane_vllm_secret_payload",
        "_vane_vllm_secret_payload",
    ],
)
def test_vllm_native_options_reject_reserved_protocol_fields(reserved_key):
    from vane.ai.providers.vllm import _build_native_vllm_options_argument

    with pytest.raises(ValueError, match="reserved protocol fields"):
        _build_native_vllm_options_argument({reserved_key: "user-value"})


def test_ai_prompt_vllm_rejects_udf_retry_policy():
    with pytest.raises(ValueError, match="native vLLM prompting does not support max_retries"):
        vane.ai.prompt(
            vane.col("chunk"),
            provider="vllm",
            prompt_options={"max_retries": 1},
        )


def test_ai_prompt_vllm_expression_plans_native_operator():
    from vane.ai.providers.vllm import VLLMProvider

    conn = vane.connect()
    rel = conn.sql("select 'search'::VARCHAR as chunk")

    expr = vane.ai.prompt(
        vane.col("chunk"),
        provider=VLLMProvider(),
        model="native-model",
        system_message="Answer briefly.",
    ).alias("answer")
    plan = rel.select(expr).explain()

    assert "VLLM_PROJECT" in plan
    assert "native-model" in plan
    assert "subprocess_actor" not in plan
    assert "ray_actor" not in plan


@pytest.mark.parametrize("api_shape", ["relation", "expression"])
def test_ai_prompt_vllm_keeps_inline_secrets_out_of_public_plan_options(api_shape):
    sentinel = "hf-NATIVE-PLAN-SECRET-SENTINEL"
    sampling_sentinel = "sk-NATIVE-SAMPLING-SECRET-SENTINEL"
    conn = vane.connect()
    source = conn.sql("select 'search'::VARCHAR as chunk")
    prompt_kwargs = {
        "provider": "vllm",
        "model": "native-model",
        "provider_options": {
            "engine_args": {
                "hf_token": sentinel,
                "max_model_len": 2048,
            },
            "generate_args": {"sampling_params": json.dumps({"api_key": sampling_sentinel, "max_tokens": 8})},
        },
    }
    if api_shape == "relation":
        relation = vane.ai.prompt(source, "chunk", **prompt_kwargs)
    else:
        expression = vane.ai.prompt(vane.col("chunk"), **prompt_kwargs).alias("response")
        relation = source.select(expression)

    assert sentinel not in relation.explain()
    assert sampling_sentinel not in relation.explain()
    physical = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        f"opaque-secret-{api_shape}",
    ).to_physical_plan(conn)
    nodes = physical.collect_vllm_nodes(conn=conn)

    assert len(nodes) == 1
    envelope = nodes[0]["options"]
    assert isinstance(envelope, dict)
    public_options_json = envelope["__vane_vllm_public_options_json"]
    assert sentinel not in public_options_json
    assert sampling_sentinel not in public_options_json
    public_options = json.loads(public_options_json)
    assert public_options["engine_args"] == {
        "hf_token": {"__vane_vllm_secret_ref": 0},
        "max_model_len": 2048,
    }
    assert public_options["generate_args"]["sampling_params"] == {
        "api_key": {"__vane_vllm_secret_ref": 1},
        "max_tokens": 8,
    }
    assert sentinel.encode() in envelope["__vane_vllm_secret_payload"]
    assert sampling_sentinel.encode() in envelope["__vane_vllm_secret_payload"]


def test_ai_prompt_vllm_relation_uses_one_native_lifecycle_and_returns_only_output(monkeypatch):
    import duckdb.execution.vllm as vllm_executor
    from vane.ai.providers.vllm import VLLMProvider

    executor = _RecordingNativeVLLMExecutor()
    captured: dict[str, object] = {}

    def build_executor(model, options):
        captured["model"] = model
        captured["options"] = options
        return executor

    monkeypatch.setattr(vllm_executor, "build_executor", build_executor)
    conn = vane.connect()
    rel = conn.sql(
        "select * from (values (1, 'question'::VARCHAR), (2, NULL::VARCHAR)) source(id, question) order by id"
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    result = vane.ai.prompt(
        rel,
        "question",
        provider=VLLMProvider(),
        model="native-model",
        system_message="Answer briefly.",
        return_format=schema,
        output_column="answer",
        provider_options=vane.ai.VLLMProviderOptions(concurrency=2),
        prompt_options={
            "do_prefix_routing": False,
            "generate_args": {"sampling_params": {"max_tokens": 32}},
        },
    )

    assert result.columns == ["answer"]
    assert "VLLM_PROJECT" in result.explain()
    assert "STREAMING_UDF" not in result.explain()
    assert Counter(result.fetchall()) == Counter(
        [
            ("generated:Answer briefly.\n\nquestion",),
            (None,),
        ]
    )
    assert captured["model"] == "native-model"
    options = captured["options"]
    assert isinstance(options, dict)
    assert options["concurrency"] == 2
    assert "actor_number" not in options
    assert options["generate_args"]["sampling_params"]["structured_outputs"] == {
        "type": "json",
        "value": schema,
    }
    assert executor.finished_count == 1
    assert executor.shutdown_count == 1
    assert [prompt for _prefix, prompts in executor.submissions for prompt in prompts] == [
        "Answer briefly.\n\nquestion",
    ]


def test_ai_prompt_vllm_relation_validates_pydantic_output_and_preserves_nulls(monkeypatch):
    pydantic = pytest.importorskip("pydantic")
    import duckdb.execution.vllm as vllm_executor

    class Answer(pydantic.BaseModel):
        answer: str
        validated: bool = True

        @pydantic.field_validator("answer")
        @classmethod
        def answer_must_be_approved(cls, value: str) -> str:
            if value != "approved":
                raise ValueError("answer must be approved")
            return value

    executor = _RecordingNativeVLLMExecutor(
        {
            "valid": '{"answer":"approved"}',
            "invalid": '{"answer":"rejected"}',
        }
    )
    monkeypatch.setattr(vllm_executor, "build_executor", lambda _model, _options: executor)
    conn = vane.connect()
    source = conn.sql(
        "select * from (values (1, 'valid'::VARCHAR), (2, 'invalid'::VARCHAR), "
        "(3, NULL::VARCHAR)) source(id, question) order by id"
    )

    result = vane.ai.prompt(
        source,
        "question",
        provider="vllm",
        return_format=Answer,
        prompt_options={"on_error": "ignore"},
    )

    plan = result.explain()
    assert "VLLM_PROJECT" in plan
    assert "STREAMING_UDF" in plan
    assert Counter(result.fetchall()) == Counter(
        [
            ('{"answer":"approved","validated":true}',),
            (None,),
            (None,),
        ]
    )
    assert executor.finished_count == 1
    assert executor.shutdown_count == 1
    assert Counter(prompt for _prefix, prompts in executor.submissions for prompt in prompts) == Counter(
        ["valid", "invalid"]
    )


def test_ai_prompt_vllm_relation_uses_pydantic_json_validation(monkeypatch):
    pydantic = pytest.importorskip("pydantic")
    import duckdb.execution.vllm as vllm_executor

    class Answer(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(strict=True)

        due: date

    executor = _RecordingNativeVLLMExecutor({"valid": '{"due":"2026-01-02"}'})
    monkeypatch.setattr(vllm_executor, "build_executor", lambda _model, _options: executor)
    conn = vane.connect()
    source = conn.sql("select 'valid'::VARCHAR as question")

    result = vane.ai.prompt(
        source,
        "question",
        provider="vllm",
        return_format=Answer,
    )

    assert result.fetchall() == [('{"due":"2026-01-02"}',)]
    assert executor.finished_count == 1
    assert executor.shutdown_count == 1


def test_ai_prompt_vllm_relation_rejects_pydantic_validation_failure(monkeypatch):
    pydantic = pytest.importorskip("pydantic")
    import duckdb.execution.vllm as vllm_executor

    class Answer(pydantic.BaseModel):
        answer: str

        @pydantic.field_validator("answer")
        @classmethod
        def answer_must_be_approved(cls, value: str) -> str:
            if value != "approved":
                raise ValueError("answer must be approved")
            return value

    executor = _RecordingNativeVLLMExecutor({"invalid": '{"answer":"rejected"}'})
    monkeypatch.setattr(vllm_executor, "build_executor", lambda _model, _options: executor)
    conn = vane.connect()
    source = conn.sql("select 'invalid'::VARCHAR as question")
    result = vane.ai.prompt(
        source,
        "question",
        provider="vllm",
        return_format=Answer,
    )

    with pytest.raises(Exception, match="answer must be approved"):
        result.fetchall()

    assert executor.finished_count == 1
    assert executor.shutdown_count == 1


def test_ai_prompt_vllm_relation_restores_opaque_secrets_at_local_executor_creation(monkeypatch):
    import duckdb.execution.vllm as vllm_executor

    engine_token = "hf-LOCAL-EXECUTOR-SECRET-SENTINEL"
    generate_key = "sk-LOCAL-EXECUTOR-SECRET-SENTINEL"
    executor = _RecordingNativeVLLMExecutor()
    captured = {}

    def create_local_executor(model, engine_args, generate_args, **kwargs):
        captured.update(
            model=model,
            engine_args=engine_args,
            generate_args=generate_args,
            kwargs=kwargs,
        )
        return executor

    monkeypatch.setattr(vllm_executor, "LocalVLLMExecutor", create_local_executor)
    conn = vane.connect()
    source = conn.sql("select 'question'::VARCHAR as question")

    result = vane.ai.prompt(
        source,
        "question",
        provider="vllm",
        model="native-secret-model",
        provider_options={
            "engine_args": {
                "hf_token": engine_token,
                "max_model_len": 1024,
            }
        },
        prompt_options={
            "generate_args": {
                "api_key": generate_key,
                "sampling_params": {"max_tokens": 8},
            }
        },
    )

    assert engine_token not in result.explain()
    assert generate_key not in result.explain()
    assert result.fetchall() == [("generated:question",)]
    assert captured["model"] == "native-secret-model"
    assert captured["engine_args"]["hf_token"] == engine_token
    assert captured["engine_args"]["max_model_len"] == 1024
    assert captured["generate_args"]["api_key"] == generate_key
    assert captured["generate_args"]["sampling_params"] == {"max_tokens": 8}


def test_ai_prompt_native_vllm_output_replaces_same_named_input(monkeypatch):
    import duckdb.execution.vllm as vllm_executor
    from vane.ai.providers.vllm import VLLMProvider

    monkeypatch.setattr(
        vllm_executor,
        "build_executor",
        lambda _model, _options: _RecordingNativeVLLMExecutor(),
    )
    rel = vane.connect().sql("select 'question'::VARCHAR as question, 'stale'::VARCHAR as answer")

    result = vane.ai.prompt(
        rel,
        "question",
        provider=VLLMProvider(),
        output_column="answer",
    )

    assert result.columns == ["answer"]
    assert result.fetchall() == [("generated:question",)]


@pytest.mark.parametrize("execution_backend", ["ray_actor", "subprocess_task", "definitely-invalid"])
def test_ai_prompt_native_vllm_rejects_explicit_execution_backend(execution_backend):
    from vane.ai.providers.vllm import VLLMProvider

    rel = vane.connect().sql("select 'question'::VARCHAR as question")

    with pytest.raises(
        ValueError,
        match="execution_backend applies only to Python UDF providers; native vLLM routing is derived from the query runner",
    ):
        vane.ai.prompt(
            rel,
            "question",
            provider=VLLMProvider(),
            execution_backend=execution_backend,
        )


def test_ai_prompt_native_vllm_rejects_image_columns():
    from vane.ai.providers.vllm import VLLMProvider

    conn = vane.connect()
    rel = conn.sql("select 'question'::VARCHAR as question, 'image'::BLOB as image")

    with pytest.raises(ValueError, match="native vLLM prompting does not support image_columns"):
        vane.ai.prompt(
            rel,
            "question",
            provider=VLLMProvider(),
            image_columns=["image"],
        )


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


def test_anthropic_and_google_prompt_options_are_public_request_mappers():
    anthropic_prompt_options = vane.ai.AnthropicPromptOptions(max_tokens=64, temperature=0, on_error="log")
    google_prompt_options = vane.ai.GooglePromptOptions(max_output_tokens=32, temperature=0, on_error="ignore")

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


def test_legacy_embed_public_symbols_are_deleted():
    assert not hasattr(vane.ai, "embed_text")
    assert not hasattr(vane.ai, "OpenAIEmbeddingOptions")
    assert not hasattr(vane.ai, "GoogleEmbeddingOptions")


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
    rel = conn.sql("select 1 as id, 'search'::VARCHAR as chunk")

    with pytest.raises(duckdb.InvalidInputException, match="GPU resources require a Ray UDF backend"):
        expr = vane.ai.prompt(
            vane.col("chunk"),
            provider=MockProvider(),
            provider_options=vane.ai.VLLMProviderOptions(concurrency=1, gpus_per_actor=1),
        ).alias("topic")
        rel.select(expr)
