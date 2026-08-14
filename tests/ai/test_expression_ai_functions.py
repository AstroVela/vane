# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import inspect
import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal, get_overloads, get_type_hints

import numpy as np
import pyarrow as pa
import pytest
from typing_extensions import Unpack

import vane
from vane.ai import provider as provider_registry
from vane.ai.options import EmbedOptions, PromptOptions
from vane.ai.protocols import PrompterDescriptor, TextEmbedderDescriptor
from vane.ai.provider import Provider
from vane.ai.typing import UDFOptions


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

    def get_dimensions(self) -> int:
        return self.dim

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
    def __init__(self, return_format: dict[str, Any] | None = None) -> None:
        self._return_format = return_format

    def prompt_batch(self, text: list[str]) -> list[str]:
        return [f"topic:{item}" for item in text]

    async def prompt(self, messages: tuple[object, ...]) -> str:
        result = f"topic:{messages[0]}"
        if self._return_format is not None:
            return json.dumps({"answer": result})
        return result


@dataclass
class MockPrompterDescriptor(PrompterDescriptor):
    actor_number: int | None = None
    max_concurrency_per_actor: int | None = None
    num_gpus: float | None = 0
    return_format: dict[str, Any] | None = None

    def get_provider(self) -> str:
        return "mock"

    def get_model(self) -> str:
        return "mock-prompt"

    def get_options(self) -> dict[str, object]:
        return {
            "batch_size": 1,
            "actor_number": self.actor_number,
            "max_concurrency_per_actor": self.max_concurrency_per_actor,
            "num_gpus": self.num_gpus,
        }

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(
            actor_number=self.actor_number,
            num_gpus=self.num_gpus,
            max_retries=0,
            on_error="raise",
            batch_size=1,
            max_concurrency_per_actor=self.max_concurrency_per_actor,
        )

    def instantiate(self) -> MockPrompter:
        return MockPrompter(self.return_format)


class MockProvider(Provider):
    @property
    def name(self) -> str:
        return "mock"

    def get_text_embedder(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        *,
        options: dict[str, object] | None = None,
    ) -> TextEmbedderDescriptor:
        return MockTextEmbedderDescriptor(dim=dimensions or 4)

    def get_prompter(
        self,
        model: str | None = None,
        system_message: str | None = None,
        return_format: dict[str, Any] | None = None,
        return_raw_response: bool = False,
        *,
        options: dict[str, object] | None = None,
    ) -> PrompterDescriptor:
        return MockPrompterDescriptor(return_format=return_format)


def test_prompt_and_embed_ignore_descriptor_execution_defaults():
    from vane.ai.functions import _prepare_embed_call, _prepare_prompt_call

    _, _, embed_options, _, _, _, _ = _prepare_embed_call(
        MockProvider(),
        None,
        None,
        "raise",
        {},
        relation=False,
    )
    _, prompt_options, _, _ = _prepare_prompt_call(
        MockProvider(),
        None,
        None,
        None,
        False,
        "raise",
        {},
        relation=False,
    )

    assert (embed_options.batch_size, embed_options.actor_number, embed_options.max_retries) == (64, 1, 3)
    assert (prompt_options.batch_size, prompt_options.actor_number, prompt_options.max_retries) == (32, 1, 3)


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


def test_ai_embed_runtime_signature_matches_public_contract():
    signature = inspect.signature(vane.ai.embed, eval_str=True)
    parameters = signature.parameters

    assert parameters["first"].annotation == vane.Expression | vane.Relation
    assert parameters["text"].annotation is vane.Expression
    assert parameters["rel"].annotation is vane.Relation
    assert parameters["provider"].annotation == str | Provider
    assert parameters["on_error"].annotation == Literal["raise", "ignore"]
    assert parameters["output_column"].annotation is str
    assert parameters["output_column"].default == "embedding"
    assert parameters["options"].kind is inspect.Parameter.VAR_KEYWORD

    hints = get_type_hints(vane.ai.embed, include_extras=True)
    assert hints["options"] == Unpack[EmbedOptions]
    assert hints["return"] == vane.Expression | vane.Relation
    assert len(get_overloads(vane.ai.embed)) == 4


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
            vane.ConstantExpression(None).cast("VARCHAR"),
            provider=MockProvider(),
            dimensions=4,
        ).alias("embedding")
    )
    relation_result = vane.ai.embed(
        source,
        vane.ConstantExpression(None).cast("VARCHAR"),
        provider=MockProvider(),
        dimensions=4,
    ).project("embedding")

    for result in (expression_result, relation_result):
        assert [str(dtype) for dtype in result.types] == ["FLOAT[4]"]
        assert result.fetchall() == [(None,)]
    assert runtime_calls == []


def test_ai_embed_four_public_entries_are_equivalent(monkeypatch):
    monkeypatch.setitem(provider_registry.PROVIDERS, "mock_embed_contract", lambda name=None: MockProvider())
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
        ("google", None, "FLOAT[3072]"),
        ("transformers", None, "FLOAT[384]"),
    ],
)
def test_ai_embed_uses_trusted_builtin_dimensions_without_execution(provider, model, expected_type):
    rel = vane.connect().sql("SELECT 'planning only'::VARCHAR AS text")
    expression = vane.ai.embed(vane.col("text"), provider=provider, model=model)

    assert [str(value) for value in rel.select(expression).types] == [expected_type]


def test_ai_embed_accepts_registered_embedding_provider_name(monkeypatch):
    monkeypatch.setitem(provider_registry.PROVIDERS, "mock_ai", lambda name=None: MockProvider())

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
def test_builtin_provider_constructors_reject_legacy_credentials(provider_kind, field):
    from vane.ai.providers.google import GoogleProvider
    from vane.ai.providers.openai import OpenAIProvider

    secret = "embed-provider-secret-sentinel"
    provider_type = OpenAIProvider if provider_kind == "openai" else GoogleProvider

    with pytest.raises(TypeError, match=field) as exc_info:
        provider_type(**{field: secret})
    assert secret not in str(exc_info.value)


@pytest.mark.parametrize("dimensions", [True, 0, -1, 1.5, "4"])
def test_ai_embed_rejects_invalid_dimensions(dimensions):
    with pytest.raises(ValueError, match="dimensions"):
        vane.ai.embed(vane.col("text"), provider=MockProvider(), dimensions=dimensions)


@pytest.mark.parametrize("on_error", ["log", ["ignore"]])
def test_ai_embed_rejects_nonfinal_on_error_value(on_error):
    with pytest.raises(ValueError, match="on_error"):
        vane.ai.embed(vane.col("text"), provider=MockProvider(), on_error=on_error)


def test_ai_embed_expression_rejects_relation_only_options():
    with pytest.raises(TypeError, match="execution_backend"):
        vane.ai.embed(
            vane.col("text"),
            provider=MockProvider(),
            execution_backend="subprocess_task",
        )


@pytest.mark.parametrize("output_column", ["embedding", "custom_embedding"])
def test_ai_embed_expression_rejects_explicit_output_column(output_column):
    with pytest.raises(TypeError, match="output_column"):
        vane.ai.embed(
            vane.col("text"),
            provider=MockProvider(),
            output_column=output_column,
        )


@pytest.mark.parametrize("output_column", [None, 7, True])
def test_ai_embed_relation_rejects_non_string_output_column(output_column):
    rel = vane.connect().sql("SELECT 'text'::VARCHAR AS text")

    with pytest.raises(ValueError, match="output_column must be a non-empty string"):
        vane.ai.embed(
            rel,
            vane.col("text"),
            provider=MockProvider(),
            dimensions=4,
            output_column=output_column,
        )


@pytest.mark.parametrize(
    "text_sql",
    ["1::INTEGER", "TRUE", "from_hex('00')", "[1, 2]", "NULL::INTEGER"],
)
@pytest.mark.parametrize("relation_api", [False, True])
def test_ai_embed_python_entries_reject_non_varchar_during_planning(relation_api, text_sql):
    rel = vane.connect().sql(f"SELECT {text_sql} AS text")

    with pytest.raises(vane.BinderException, match="ai SQL input argument must be VARCHAR"):
        if relation_api:
            vane.ai.embed(
                rel,
                vane.col("text"),
                provider=MockProvider(),
                dimensions=4,
                on_error="ignore",
            )
        else:
            expression = vane.ai.embed(
                vane.col("text"),
                provider=MockProvider(),
                dimensions=4,
                on_error="ignore",
            )
            rel.select(expression)


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
            revision="0123456789abcdef0123456789abcdef01234567",
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
        def get_text_embedder(self, model=None, dimensions=None, *, options=None):
            return Descriptor(dim=dimensions or 99)

    rel = vane.connect().sql("select 'abc'::VARCHAR as text")
    expr = vane.ai.embed(vane.col("text"), provider=ExplicitProvider(), dimensions=7)

    assert [str(value) for value in rel.select(expr).types] == ["FLOAT[7]"]


@pytest.mark.parametrize("value", [True, 0, -1, 3.0, "3"])
def test_ai_embed_rejects_invalid_provider_dimensions(value):
    class Descriptor(MockTextEmbedderDescriptor):
        def get_dimensions(self):
            return value

    class InvalidProvider(MockProvider):
        def get_text_embedder(self, model=None, dimensions=None, *, options=None):
            return Descriptor(dim=3)

    with pytest.raises(ValueError, match=r"get_dimensions\(\).*positive integer"):
        vane.ai.embed(vane.col("text"), provider=InvalidProvider())


def test_ai_embed_rejects_legacy_dimension_metadata_object():
    class LegacyDimensions:
        size = 3

    class Descriptor(MockTextEmbedderDescriptor):
        def get_dimensions(self):
            return LegacyDimensions()

    class LegacyProvider(MockProvider):
        def get_text_embedder(self, model=None, dimensions=None, *, options=None):
            return Descriptor(dim=3)

    with pytest.raises(ValueError, match=r"get_dimensions\(\).*positive integer"):
        vane.ai.embed(vane.col("text"), provider=LegacyProvider())


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
    assert (error.provider, error.model, error.capability) == (
        "mock",
        "chat-only-model",
        "embedding endpoint/model",
    )
    assert error.original_error is not original
    assert str(error.original_error) == "RuntimeError"


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


def _drive_embed_wrapper(wrapper: Any, texts: list[str | None]) -> pa.Table:
    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        return wrapper(pa.table({"text": texts}))
    finally:
        loop.close()


@pytest.mark.parametrize(
    "invalid_component",
    [float("nan"), float("inf"), float("-inf"), 1e100],
    ids=["nan", "positive-infinity", "negative-infinity", "float32-overflow"],
)
def test_embed_non_finite_result_raises_without_retry(invalid_component):
    from vane.ai.functions import _EmbedTextBatch
    from vane.ai.provider import _ProviderResultError

    calls = []

    class NonFiniteEmbedder:
        def embed_text(self, texts):
            calls.append(list(texts))
            return [[1.0, invalid_component, 2.0] for _ in texts]

    class Descriptor:
        def get_provider(self):
            return "mock"

        def get_model(self):
            return "non-finite-model"

        def instantiate(self):
            return NonFiniteEmbedder()

    wrapper = _EmbedTextBatch(Descriptor(), "text", "embedding", 3, max_retries=3)

    with pytest.raises(_ProviderResultError, match="embedding containing non-finite components"):
        _drive_embed_wrapper(wrapper, ["bad"])

    assert calls == [["bad"]]


@pytest.mark.parametrize(
    "invalid_embedding",
    [
        [1.0, float("nan"), 2.0],
        ["not", "numeric", "values"],
        [1.0, 2.0],
    ],
    ids=["non-finite", "non-numeric", "wrong-dimensions"],
)
def test_embed_on_error_ignore_nulls_invalid_result_row_without_reinvoking_provider(invalid_embedding):
    from vane.ai.functions import _EmbedTextBatch

    calls = []

    class MixedEmbedder:
        def embed_text(self, texts):
            calls.append(list(texts))
            vectors = {
                "first": [3.0, 0.0, 4.0],
                "bad": invalid_embedding,
                "last": [0.0, 5.0, 0.0],
            }
            return [vectors[text] for text in texts]

    class Descriptor:
        def instantiate(self):
            return MixedEmbedder()

    wrapper = _EmbedTextBatch(
        Descriptor(),
        "text",
        "embedding",
        3,
        max_retries=3,
        on_error="ignore",
        normalize=True,
    )

    result = _drive_embed_wrapper(wrapper, ["first", "bad", None, "last"])
    embeddings = result.column("embedding").to_pylist()

    np.testing.assert_allclose(embeddings[0], [0.6, 0.0, 0.8], rtol=1e-6)
    assert embeddings[1:3] == [None, None]
    np.testing.assert_allclose(embeddings[3], [0.0, 1.0, 0.0], rtol=1e-6)
    assert calls == [["first", "bad", "last"]]
    assert result.schema.field("embedding").type == pa.list_(pa.float32(), 3)


@pytest.mark.parametrize(
    "magnitude",
    [np.finfo(np.float32).max, np.finfo(np.float32).tiny],
    ids=["large", "small"],
)
def test_embed_normalization_uses_float64_for_extreme_finite_vector(magnitude):
    from vane.ai.functions import _EmbedTextBatch

    class ExtremeFiniteEmbedder:
        def embed_text(self, texts):
            return [np.array([magnitude, magnitude], dtype=np.float32) for _ in texts]

    class Descriptor:
        def instantiate(self):
            return ExtremeFiniteEmbedder()

    wrapper = _EmbedTextBatch(Descriptor(), "text", "embedding", 2, max_retries=0, normalize=True)
    embedding = _drive_embed_wrapper(wrapper, ["extreme"]).column("embedding")[0].as_py()

    assert all(math.isfinite(value) for value in embedding)
    np.testing.assert_allclose(embedding, [math.sqrt(0.5), math.sqrt(0.5)], rtol=1e-6)


def test_embed_normalization_preserves_finite_zero_vector():
    from vane.ai.functions import _EmbedTextBatch

    class ZeroEmbedder:
        def embed_text(self, texts):
            return [np.zeros(3, dtype=np.float32) for _ in texts]

    class Descriptor:
        def instantiate(self):
            return ZeroEmbedder()

    wrapper = _EmbedTextBatch(Descriptor(), "text", "embedding", 3, max_retries=0, normalize=True)

    assert _drive_embed_wrapper(wrapper, ["zero"]).column("embedding").to_pylist() == [[0.0, 0.0, 0.0]]


def test_embed_non_finite_chunk_nulls_parent_without_reinvoking_provider():
    from vane.ai.functions import _EmbedTextBatch

    calls = []

    class ChunkEmbedder:
        def embed_text(self, texts):
            calls.append(list(texts))
            return [[float("nan"), 0.0] if text == "bbbb" else [1.0, 0.0] for text in texts]

    class Descriptor:
        def instantiate(self):
            return ChunkEmbedder()

    wrapper = _EmbedTextBatch(
        Descriptor(),
        "text",
        "embedding",
        2,
        max_chunk_chars=4,
        chunk_overlap_chars=0,
        max_retries=3,
        on_error="ignore",
    )

    result = _drive_embed_wrapper(wrapper, ["aaaabbbb", "ccccdddd"])

    assert result.column("embedding").to_pylist() == [None, [1.0, 0.0]]
    assert calls == [["aaaa", "bbbb", "cccc", "dddd"]]


def test_embed_final_contract_rejects_non_finite_normalized_output(monkeypatch):
    from vane.ai.functions import _EmbedTextBatch

    class FiniteEmbedder:
        def embed_text(self, texts):
            return [[1.0, 2.0] for _ in texts]

    class Descriptor:
        def instantiate(self):
            return FiniteEmbedder()

    monkeypatch.setattr(
        "vane.ai.functions._normalize_embedding",
        lambda _embedding: np.array([1.0, float("inf")], dtype=np.float32),
    )
    wrapper = _EmbedTextBatch(
        Descriptor(),
        "text",
        "embedding",
        2,
        max_retries=0,
        on_error="ignore",
        normalize=True,
    )

    assert _drive_embed_wrapper(wrapper, ["row"]).column("embedding").to_pylist() == [None]


def test_ai_prompt_expression_basic():
    conn = vane.connect()
    rel = conn.sql(
        "select chunk from (values (0, 'search'::VARCHAR), (1, 'ranking'::VARCHAR)) t(ord, chunk) order by ord"
    )

    expr = vane.ai.prompt(vane.col("chunk"), provider=MockProvider()).alias("topic")

    assert rel.select(expr).fetchall() == [("topic:search",), ("topic:ranking",)]


def test_ai_prompt_structured_output_preserves_json_control_characters():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    relation = vane.connect().sql("SELECT concat('line1', chr(10), 'line2', chr(9), chr(1))::VARCHAR AS message")
    message = vane.col("message")
    expected = [({"answer": "topic:line1\nline2\t\x01"},)]

    expression_result = relation.select(
        vane.ai.prompt(message, provider=MockProvider(), return_format=schema).alias("response")
    )
    relation_result = vane.ai.prompt(
        relation,
        message,
        provider=MockProvider(),
        return_format=schema,
    ).project("response")
    method_result = relation.prompt(
        message,
        provider=MockProvider(),
        return_format=schema,
    ).project("response")

    assert expression_result.fetchall() == expected
    assert relation_result.fetchall() == expected
    assert method_result.fetchall() == expected


def test_ai_prompt_runtime_signature_matches_public_contract():
    signature = inspect.signature(vane.ai.prompt, eval_str=True)
    parameters = signature.parameters

    assert parameters["first"].annotation == vane.Expression | list[vane.Expression] | vane.Relation
    assert parameters["messages"].annotation == vane.Expression | list[vane.Expression]
    assert parameters["rel"].annotation is vane.Relation
    assert parameters["return_format"].annotation == type[Any] | vane.ai.JSONSchema | None
    assert parameters["provider"].annotation == str | Provider
    assert parameters["on_error"].annotation == Literal["raise", "ignore"]
    assert parameters["output_column"].annotation is str
    assert parameters["output_column"].default == "response"
    assert parameters["options"].kind is inspect.Parameter.VAR_KEYWORD

    hints = get_type_hints(vane.ai.prompt, include_extras=True)
    assert hints["options"] == Unpack[PromptOptions]
    assert hints["return"] == vane.Expression | vane.Relation
    assert len(get_overloads(vane.ai.prompt)) == 4


def test_anthropic_zero_tokens_rejects_structured_python_entries():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    relation = vane.connect().sql("select 'hello'::VARCHAR as message")
    message = vane.col("message")
    common = {
        "provider": "anthropic",
        "model": "claude-test",
        "return_format": schema,
        "max_tokens": 0,
    }

    with pytest.raises(ValueError, match="max_tokens=0.*structured"):
        vane.ai.prompt(message, **common)
    with pytest.raises(ValueError, match="max_tokens=0.*structured"):
        vane.ai.prompt(relation, message, **common)
    with pytest.raises(ValueError, match="max_tokens=0.*structured"):
        relation.prompt(message, **common)


@pytest.mark.parametrize(
    ("model", "options"),
    [
        ("gpt-4o", {}),
        ("gpt-4o-2024-08-06", {}),
        ("o3-mini", {}),
        ("gpt-5.1", {}),
        ("gpt-5.1-2025-11-13", {}),
        ("gpt-4o", {"base_url": "https://api.openai.com/v1/"}),
    ],
)
def test_openai_known_structured_output_models_enforce_strict_schema(model, options):
    from vane.ai._schema import SchemaValidationError
    from vane.ai.providers.openai import OpenAIProvider

    loose_schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }

    with pytest.raises(SchemaValidationError, match="additionalProperties"):
        vane.ai.prompt(
            vane.col("message"),
            provider=OpenAIProvider(),
            model=model,
            return_format=loose_schema,
            **options,
        )


def test_openai_environment_base_url_cannot_change_static_capability(monkeypatch):
    from vane.ai._schema import SchemaValidationError
    from vane.ai.providers.openai import OpenAIProvider

    monkeypatch.setenv("OPENAI_BASE_URL", "https://compatible.example.test/v1")
    loose_schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }

    with pytest.raises(SchemaValidationError, match="additionalProperties"):
        vane.ai.prompt(
            vane.col("message"),
            provider=OpenAIProvider(),
            model="gpt-4o",
            return_format=loose_schema,
        )


@pytest.mark.parametrize(
    ("model", "options"),
    [
        ("gpt-4o", {"base_url": "https://compatible.example.test/v1"}),
        ("gateway/gpt-4o", {}),
        ("compatible-unknown-model", {}),
        ("gpt-4o-audio-preview", {}),
        ("gpt-4o-realtime-preview", {}),
        ("gpt-4o-2024-05-13", {}),
        ("o3-deep-research", {}),
    ],
)
def test_openai_compatible_models_do_not_get_static_strict_schema_enforcement(model, options):
    from vane.ai.providers.openai import OpenAIProvider

    loose_schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }

    expression = vane.ai.prompt(
        vane.col("message"),
        provider=OpenAIProvider(),
        model=model,
        return_format=loose_schema,
        **options,
    )

    assert expression is not None


@pytest.mark.parametrize("model", ["o1-mini", "o1-mini-2024-09-12"])
def test_openai_known_unsupported_structured_output_models_fail_during_planning(model):
    from vane.ai.providers.openai import OpenAIProvider

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    with pytest.raises(ValueError, match="does not support structured Prompt output"):
        vane.ai.prompt(
            vane.col("message"),
            provider=OpenAIProvider(),
            model=model,
            return_format=schema,
        )


@pytest.mark.parametrize(
    ("model", "options", "message"),
    [
        ("o3-pro", {"use_chat_completions": True}, "Responses API only"),
        ("gpt-4o-search-preview", {}, "Chat Completions only"),
    ],
)
def test_openai_known_model_api_conflicts_fail_during_planning(model, options, message):
    from vane.ai.providers.openai import OpenAIProvider

    with pytest.raises(ValueError, match=message):
        vane.ai.prompt(
            vane.col("message"),
            provider=OpenAIProvider(),
            model=model,
            **options,
        )


@pytest.mark.parametrize(
    ("model", "options"),
    [
        ("o3-pro", {"use_chat_completions": True}),
        ("gpt-4o-search-preview", {}),
    ],
)
def test_openai_compatible_endpoints_do_not_inherit_official_api_path_restrictions(model, options):
    from vane.ai.providers.openai import OpenAIProvider

    expression = vane.ai.prompt(
        vane.col("message"),
        provider=OpenAIProvider(),
        model=model,
        base_url="https://compatible.example.test/v1",
        **options,
    )

    assert expression is not None


@pytest.mark.parametrize(
    ("model", "options"),
    [
        ("o1-mini", {}),
        ("o1-mini-2024-09-12", {}),
        ("o1-preview", {}),
        ("o1-preview-2024-09-12", {}),
        ("o3-mini", {}),
        ("o3-mini-2025-01-31", {}),
        ("gpt-4o-search-preview", {"use_chat_completions": True}),
        ("gpt-4o-search-preview-2025-03-11", {"use_chat_completions": True}),
        ("gpt-4o-mini-search-preview", {"use_chat_completions": True}),
        ("gpt-4o-mini-search-preview-2025-03-11", {"use_chat_completions": True}),
    ],
)
@pytest.mark.parametrize("entry_point", ["relation", "expression", "sql"])
def test_openai_known_text_only_models_reject_images_during_planning(model, options, entry_point):
    from vane.ai._sql import build_ai_prompt_sql_spec

    if entry_point == "sql":
        with pytest.raises(ValueError, match="does not support Prompt image inputs"):
            build_ai_prompt_sql_spec(model=model, image_input=True, options=options)
        return

    relation = vane.connect().sql("select 'question'::VARCHAR as question, '\\x89504e470d0a1a0a'::BLOB as image")
    if entry_point == "relation":
        with pytest.raises(ValueError, match="does not support Prompt image inputs"):
            vane.ai.prompt(
                relation,
                [vane.col("question"), vane.col("image")],
                provider="openai",
                model=model,
                **options,
            )
        return

    expression = vane.ai.prompt(
        [vane.col("question"), vane.col("image")],
        provider="openai",
        model=model,
        **options,
    )
    with pytest.raises(Exception, match="VARCHAR"):
        relation.select(expression).types


def test_openai_compatible_endpoint_does_not_inherit_official_text_only_capability():
    relation = vane.connect().sql("select 'question'::VARCHAR as question, '\\x89504e470d0a1a0a'::BLOB as image")

    result = vane.ai.prompt(
        relation,
        [vane.col("question"), vane.col("image")],
        provider="openai",
        model="o1-mini",
        base_url="https://compatible.example.test/v1",
    )

    assert str(result.types[-1]) == "VARCHAR"


def test_ai_prompt_four_public_entries_are_equivalent():
    conn = vane.connect()
    relation = conn.sql("select 'search'::VARCHAR as chunk")
    messages = vane.col("chunk")

    expression_positional = relation.select(vane.ai.prompt(messages, provider=MockProvider()).alias("response"))
    expression_keyword = relation.select(vane.ai.prompt(messages=messages, provider=MockProvider()).alias("response"))
    relation_positional = vane.ai.prompt(relation, messages, provider=MockProvider())
    relation_keyword = vane.ai.prompt(rel=relation, messages=messages, provider=MockProvider())
    relation_method = relation.prompt(messages, provider=MockProvider())

    expected = [("topic:search",)]
    assert expression_positional.fetchall() == expected
    assert expression_keyword.fetchall() == expected
    assert [row[-1:] for row in relation_positional.fetchall()] == expected
    assert [row[-1:] for row in relation_keyword.fetchall()] == expected
    assert [row[-1:] for row in relation_method.fetchall()] == expected


def _drive_prompt_wrapper(wrapper, table):
    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        return wrapper(table)
    finally:
        wrapper.close()
        loop.close()


def test_prompt_batch_preserves_part_order_flattens_blob_lists_and_skips_nulls():
    from vane.ai.functions import _PromptBatch

    png = b"\x89PNG\r\n\x1a\nimage"
    jpeg = b"\xff\xd8image"

    class RecordingPrompter:
        def __init__(self):
            self.calls = []

        async def prompt(self, messages):
            self.calls.append(messages)
            return f"row-{len(self.calls)}"

    class Descriptor:
        def __init__(self):
            self.prompter = RecordingPrompter()
            self.instantiate_count = 0

        def instantiate(self):
            self.instantiate_count += 1
            return self.prompter

    descriptor = Descriptor()
    wrapper = _PromptBatch(
        descriptor,
        ["message_0", "message_1"],
        "response",
        max_concurrency_per_actor=1,
        on_error="ignore",
        max_retries=0,
    )
    result = _drive_prompt_wrapper(
        wrapper,
        pa.table(
            {
                "message_0": ["first", None, None, "bad-image"],
                "message_1": [[png, None, jpeg], [png], [], [b""]],
            }
        ),
    )

    assert descriptor.prompter.calls == [
        ("first", png, jpeg),
        (png,),
    ]
    assert result.column("response").to_pylist() == ["row-1", "row-2", None, None]


def test_prompt_batch_single_message_null_short_circuits_even_with_image():
    from vane.ai.functions import _PromptBatch

    class Descriptor:
        instantiate_count = 0

        def instantiate(self):
            self.instantiate_count += 1
            raise AssertionError("NULL prompt rows must not instantiate a provider")

    descriptor = Descriptor()
    wrapper = _PromptBatch(
        descriptor,
        ["message_0", "message_1"],
        "response",
        single_message=True,
        max_retries=0,
    )
    result = _drive_prompt_wrapper(
        wrapper,
        pa.table({"message_0": [None], "message_1": [b"\x89PNG\r\n\x1a\nimage"]}),
    )

    assert result.column("response").to_pylist() == [None]
    assert descriptor.instantiate_count == 0


def test_prompt_batch_zero_length_blob_raises_or_isolates_by_row():
    from vane.ai.functions import _PromptBatch

    class Prompter:
        async def prompt(self, messages):
            return "ok"

    class Descriptor:
        def instantiate(self):
            return Prompter()

    table = pa.table({"message_0": [b""]})
    with pytest.raises(ValueError, match="zero length"):
        _drive_prompt_wrapper(
            _PromptBatch(Descriptor(), ["message_0"], "response", max_retries=0),
            table,
        )

    isolated = _drive_prompt_wrapper(
        _PromptBatch(
            Descriptor(),
            ["message_0"],
            "response",
            max_retries=0,
            on_error="ignore",
        ),
        table,
    )
    assert isolated.column("response").to_pylist() == [None]


def test_prompt_batch_rejects_non_text_provider_results():
    from vane.ai.functions import _PromptBatch

    class Prompter:
        async def prompt(self, messages):
            return {"not": "text"}

    class Descriptor:
        def instantiate(self):
            return Prompter()

    with pytest.raises(TypeError, match="must be text or NULL"):
        _drive_prompt_wrapper(
            _PromptBatch(Descriptor(), ["message_0"], "response", max_retries=0),
            pa.table({"message_0": ["hello"]}),
        )


@pytest.mark.parametrize(
    "messages",
    [
        [],
        ["not-an-expression"],
        [vane.col("text"), "not-an-expression"],
    ],
)
def test_ai_prompt_rejects_invalid_message_lists(messages):
    expected = ValueError if messages == [] else TypeError
    with pytest.raises(expected):
        vane.ai.prompt(messages, provider=MockProvider())


def test_ai_prompt_relation_rejects_unsupported_expression_type():
    relation = vane.connect().sql("select 1::INTEGER as value")
    with pytest.raises(TypeError, match="VARCHAR, BLOB, or BLOB"):
        vane.ai.prompt(relation, vane.col("value"), provider=MockProvider())


@pytest.mark.parametrize("on_error", ["raise", "ignore"])
def test_ai_prompt_expression_rejects_unsupported_type_during_planning(on_error):
    relation = vane.connect().sql("select 1::INTEGER as value")
    expression = vane.ai.prompt(vane.col("value"), provider=MockProvider(), on_error=on_error)

    with pytest.raises(Exception, match="VARCHAR, BLOB, or BLOB"):
        relation.select(expression).types


def test_ai_prompt_expression_accepts_supported_types_during_planning():
    relation = vane.connect().sql("""
        select
            'question'::VARCHAR as text,
            from_hex('89504e47') as image,
            [from_hex('ffd8ff')]::BLOB[] as images
    """)

    for column in ("text", "image", "images"):
        expression = vane.ai.prompt(vane.col(column), provider=MockProvider())
        assert [str(value) for value in relation.select(expression).types] == ["VARCHAR"]


@pytest.mark.parametrize(
    "option",
    [
        {"column": "text"},
        {"image_columns": ["image"]},
        {"provider_options": {}},
        {"prompt_options": {}},
        {"concurrency": 2},
        {"max_api_concurrency": 2},
    ],
)
def test_ai_prompt_rejects_removed_options(option):
    with pytest.raises(TypeError):
        vane.ai.prompt(vane.col("text"), provider=MockProvider(), **option)


@pytest.mark.parametrize(
    ("provider", "options", "offending"),
    [
        ("openai", {"max_tokens": 8}, "max_tokens"),
        ("anthropic", {"max_output_tokens": 8}, "max_output_tokens"),
        ("google", {"base_url": "https://example.test"}, "base_url"),
        ("vllm", {"top_p": 0.5}, "top_p"),
    ],
)
def test_ai_prompt_options_are_provider_closed(provider, options, offending):
    model = "claude-test" if provider == "anthropic" else "gemini-test" if provider == "google" else None
    if provider == "anthropic":
        options = {"max_tokens": 8, **options}
    with pytest.raises(TypeError, match=offending):
        vane.ai.prompt(vane.col("text"), provider=provider, model=model, **options)


@pytest.mark.parametrize(
    ("provider", "model", "required_options"),
    [
        ("openai", None, {}),
        ("anthropic", "claude-test", {"max_tokens": 8}),
        ("google", "gemini-test", {}),
        ("vllm", None, {}),
    ],
)
def test_builtin_prompt_rejects_negative_temperature_during_planning(provider, model, required_options):
    with pytest.raises(ValueError, match=r"Prompt option 'temperature' must be >= 0"):
        vane.ai.prompt(
            vane.col("text"),
            provider=provider,
            model=model,
            temperature=-0.1,
            **required_options,
        )


@pytest.mark.parametrize(
    ("provider", "model", "required_options"),
    [
        ("openai", None, {}),
        ("anthropic", "claude-test", {"max_tokens": 8}),
        ("google", "gemini-test", {}),
        ("vllm", None, {}),
    ],
)
def test_builtin_prompt_accepts_zero_temperature_during_planning(provider, model, required_options):
    vane.ai.prompt(
        vane.col("text"),
        provider=provider,
        model=model,
        temperature=0,
        **required_options,
    )


@pytest.mark.parametrize(
    "options",
    [
        {"api_key": "secret"},
        {"engine_args": {"hf_token": "secret"}},
        {"generate_args": {"nested": {"authorization": "secret"}}},
    ],
)
def test_ai_prompt_rejects_sensitive_options_recursively(options):
    provider = "vllm" if "engine_args" in options or "generate_args" in options else "openai"
    with pytest.raises(ValueError, match="sensitive"):
        vane.ai.prompt(vane.col("text"), provider=provider, **options)


def test_vllm_prompt_options_fold_without_mutating_callers():
    from vane.ai.providers.vllm import NativeVLLMPromptPlan

    engine_args = {"max_model_len": 2048}
    generate_args = {"sampling_params": {"top_p": 0.8}}
    plan = NativeVLLMPromptPlan(
        model_name="test-model",
        vllm_options={
            "actor_number": 3,
            "batch_size": 64,
            "max_retries": 0,
            "max_tokens": 17,
            "temperature": 0.25,
            "engine_args": engine_args,
            "generate_args": generate_args,
        },
    )

    physical = plan.build_physical_vllm_options()

    assert physical["concurrency"] == 3
    assert physical["batch_size"] == 64
    assert physical["engine_args"] == engine_args
    assert physical["generate_args"]["sampling_params"] == {
        "top_p": 0.8,
        "max_tokens": 17,
        "temperature": 0.25,
    }
    assert generate_args == {"sampling_params": {"top_p": 0.8}}


def test_vllm_prompt_injects_structured_schema_without_mutating_callers():
    from vane.ai.providers.vllm import NativeVLLMPromptPlan

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    plan = NativeVLLMPromptPlan(model_name="test-model", return_format=schema)

    physical = plan.build_physical_vllm_options()

    assert physical["generate_args"]["sampling_params"]["structured_outputs"] == {"json": schema}
    assert schema == {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }


@pytest.mark.parametrize("entry_point", ["expression", "relation"])
def test_ai_prompt_vllm_validates_structured_output_and_nulls_invalid_rows(monkeypatch, entry_point):
    import vane.execution.vllm as vllm_executor

    control_text = "line1\nline2\t\x01"
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    executor = _RecordingNativeVLLMExecutor({"valid": json.dumps({"answer": control_text}), "invalid": '{"answer":1}'})
    monkeypatch.setattr(vllm_executor, "build_executor", lambda _model, _options: executor)
    relation = vane.connect().sql(
        "select * from (values (1, 'valid'::VARCHAR), (2, 'invalid'::VARCHAR), "
        "(3, NULL::VARCHAR)) source(id, prompt) order by id"
    )

    if entry_point == "expression":
        result = relation.select(
            vane.col("id"),
            vane.col("prompt"),
            vane.ai.prompt(
                vane.col("prompt"),
                provider="vllm",
                return_format=schema,
                on_error="ignore",
            ).alias("response"),
        )
    else:
        result = vane.ai.prompt(
            relation,
            vane.col("prompt"),
            provider="vllm",
            return_format=schema,
            on_error="ignore",
        )

    assert str(result.types[-1]) == "STRUCT(answer VARCHAR)"
    assert sorted(result.fetchall()) == [
        (1, "valid", {"answer": control_text}),
        (2, "invalid", None),
        (3, None, None),
    ]


def test_ai_prompt_vllm_rejects_raw_response_during_planning():
    with pytest.raises(ValueError, match="does not support return_raw_response"):
        vane.ai.prompt(vane.col("prompt"), provider="vllm", return_raw_response=True)


@pytest.mark.parametrize(
    "options",
    [
        {"max_retries": 1},
        {"engine_args": {"dtype": object()}},
        {"generate_args": {"temperature": float("nan")}},
        {"generate_args": {"structured_outputs": {"type": "json"}}},
    ],
)
def test_vllm_prompt_rejects_invalid_options_during_planning(options):
    error = (TypeError, ValueError)
    with pytest.raises(error):
        vane.ai.prompt(vane.col("text"), provider="vllm", **options)


def test_ai_prompt_vllm_rejects_images_and_execution_backend():
    relation = vane.connect().sql("select 'question'::VARCHAR as question, '\\x89504e470d0a1a0a'::BLOB as image")
    with pytest.raises(ValueError, match="does not support Prompt image"):
        vane.ai.prompt(
            relation,
            [vane.col("question"), vane.col("image")],
            provider="vllm",
        )
    with pytest.raises(TypeError, match="execution_backend"):
        vane.ai.prompt(
            relation,
            vane.col("question"),
            provider="vllm",
            execution_backend="ray_actor",
        )


@pytest.mark.parametrize("image_type", ["BLOB", "BLOB[]"])
def test_ai_prompt_expression_vllm_rejects_mixed_image_input_during_planning(image_type):
    image = "'\\x89504e470d0a1a0a'::BLOB" if image_type == "BLOB" else "['\\x89504e470d0a1a0a'::BLOB]"
    relation = vane.connect().sql(f"select 'question'::VARCHAR as question, {image} as image")
    expression = vane.ai.prompt(
        [vane.col("question"), vane.col("image")],
        provider="vllm",
    )

    with pytest.raises(Exception, match="VARCHAR"):
        relation.select(expression).types


def test_ai_prompt_replaces_existing_output_column():
    relation = vane.connect().sql("select 'question'::VARCHAR as question, 'old'::VARCHAR as response")
    result = vane.ai.prompt(
        relation,
        vane.col("question"),
        provider=MockProvider(),
        output_column="response",
    )

    assert result.columns == ["question", "response"]
    assert result.fetchall() == [("question", "topic:question")]


def test_ai_prompt_public_option_exports_are_closed():
    assert vane.ai.PromptOptions is not None
    for old_name in (
        "OpenAIProviderOptions",
        "OpenAIPromptOptions",
        "AnthropicProviderOptions",
        "AnthropicPromptOptions",
        "GoogleProviderOptions",
        "GooglePromptOptions",
        "VLLMProviderOptions",
        "VLLMPromptOptions",
    ):
        assert not hasattr(vane.ai, old_name)
