# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for vane.ai high-level functions.

Uses mock models to avoid network/GPU dependencies. Tests verify the full
path: Provider → Descriptor → map_batches wrapper → DuckDB execution.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest

import duckdb
import vane
from vane.ai.protocols import (
    TextClassifierDescriptor,
    TextEmbedderDescriptor,
)
from vane.ai.provider import Provider
from vane.ai.typing import EmbeddingDimensions, UDFOptions

if TYPE_CHECKING:
    from vane.ai.protocols import TextClassifier, TextEmbedder
    from vane.ai.typing import Options


def _has_module(name: str) -> bool:
    """Check if a Python module is importable."""
    import importlib

    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def _drive(wrapper, table: pa.Table) -> pa.Table:
    """Call an AI batch wrapper the way a UDF executor drives it.

    Wrappers no longer own event loops: the executor binds a ``run_async``
    capability before the first batch. Tests that invoke wrappers directly
    must do the same.
    """
    import asyncio

    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        return wrapper(table)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Mock implementations
# ---------------------------------------------------------------------------


class MockTextEmbedder:
    """Returns fixed-dimension random embeddings."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed_text(self, text: list[str]) -> list[np.ndarray]:
        return [np.ones(self.dim, dtype=np.float32) * len(t) for t in text]


@dataclass
class MockTextEmbedderDescriptor(TextEmbedderDescriptor):
    dim: int = 4

    def get_provider(self) -> str:
        return "mock"

    def get_model(self) -> str:
        return "mock-embedder"

    def get_options(self) -> Options:
        return {"batch_size": 2}

    def get_dimensions(self) -> EmbeddingDimensions:
        return EmbeddingDimensions(size=self.dim, dtype=pa.float32())

    def instantiate(self) -> TextEmbedder:
        return MockTextEmbedder(dim=self.dim)


class MockTextClassifier:
    """Returns the first label for every input."""

    def classify_text(self, text: list[str], labels: list[str]) -> list[str]:
        return [labels[0] for _ in text]


@dataclass
class MockTextClassifierDescriptor(TextClassifierDescriptor):
    def get_provider(self) -> str:
        return "mock"

    def get_model(self) -> str:
        return "mock-classifier"

    def get_options(self) -> Options:
        return {"batch_size": 2}

    def instantiate(self) -> TextClassifier:
        return MockTextClassifier()


class MockProvider(Provider):
    """Provider that returns mock descriptors."""

    @property
    def name(self) -> str:
        return "mock"

    def get_text_embedder(self, model=None, dimensions=None, **_options) -> TextEmbedderDescriptor:
        return MockTextEmbedderDescriptor(dim=dimensions or 4)

    def get_text_classifier(self, model=None, **_options) -> TextClassifierDescriptor:
        return MockTextClassifierDescriptor()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmbed:
    def test_embed_basic(self):
        """embed preserves the source relation and appends an embedding."""
        from vane.ai.functions import embed

        conn = duckdb.connect()
        rel = conn.sql("SELECT 'hello' AS text UNION ALL SELECT 'world' AS text")

        result = embed(
            rel,
            vane.col("text"),
            provider=MockProvider(),
        )

        rows = result.fetchall()
        assert len(rows) == 2
        # Each embedding should be a list of 4 floats
        for row in rows:
            emb = row[1]
            assert len(emb) == 4

    def test_embed_custom_dimensions(self):
        """embed respects dimensions parameter."""
        from vane.ai.functions import embed

        conn = duckdb.connect()
        rel = conn.sql("SELECT 'test' AS text")

        result = embed(
            rel,
            vane.col("text"),
            provider=MockProvider(),
            dimensions=8,
        )

        rows = result.fetchall()
        assert len(rows[0][1]) == 8

    def test_embed_custom_output_column(self):
        """embed uses a custom output column name."""
        from vane.ai.functions import embed

        conn = duckdb.connect()
        rel = conn.sql("SELECT 'test' AS text")

        result = embed(
            rel,
            vane.col("text"),
            provider=MockProvider(),
            output_column="my_emb",
        )

        rows = result.fetchall()
        assert len(rows) == 1
        assert result.columns == ["text", "my_emb"]

    def test_embed_propagates_null(self):
        """embed propagates NULL without sending an empty-string request."""
        from vane.ai.functions import embed

        conn = duckdb.connect()
        rel = conn.sql("SELECT NULL::VARCHAR AS text")

        result = embed(
            rel,
            vane.col("text"),
            provider=MockProvider(),
        )

        rows = result.fetchall()
        assert len(rows) == 1
        assert rows[0][1] is None


class TestClassifyText:
    def test_classify_text_basic(self):
        """classify_text produces a relation with label column."""
        from vane.ai.functions import classify_text

        conn = duckdb.connect()
        rel = conn.sql("SELECT 'great product' AS text UNION ALL SELECT 'terrible' AS text")

        result = classify_text(
            rel,
            "text",
            labels=["positive", "negative"],
            provider=MockProvider(),
        )

        rows = result.fetchall()
        assert len(rows) == 2
        # MockTextClassifier always returns the first label
        for row in rows:
            assert row[0] == "positive"

    def test_classify_text_custom_output(self):
        from vane.ai.functions import classify_text

        conn = duckdb.connect()
        rel = conn.sql("SELECT 'test' AS text")

        result = classify_text(
            rel,
            "text",
            labels=["a", "b"],
            provider=MockProvider(),
            output_column="sentiment",
        )

        rows = result.fetchall()
        assert len(rows) == 1


class TestMockDescriptorPickle:
    """Verify mock descriptors are serializable (requirement for Ray)."""

    def test_embedder_descriptor_pickle(self):
        desc = MockTextEmbedderDescriptor(dim=16)
        restored = pickle.loads(pickle.dumps(desc))
        embedder = restored.instantiate()
        result = embedder.embed_text(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 16

    def test_classifier_descriptor_pickle(self):
        desc = MockTextClassifierDescriptor()
        restored = pickle.loads(pickle.dumps(desc))
        classifier = restored.instantiate()
        result = classifier.classify_text(["test"], ["a", "b"])
        assert result == ["a"]


class TestWrapperPickle:
    """Verify wrapper classes are picklable (critical for Ray execution)."""

    def test_embed_wrapper_pickle(self):
        from vane.ai.functions import _EmbedTextBatch

        wrapper = _EmbedTextBatch(MockTextEmbedderDescriptor(dim=4), "text", "emb", 4)
        restored = pickle.loads(pickle.dumps(wrapper))
        table = pa.table({"text": ["hello", "world"]})
        result = _drive(restored, table)
        assert result.num_rows == 2
        assert result.column_names == ["emb"]

    def test_classify_wrapper_pickle(self):
        from vane.ai.functions import _ClassifyTextBatch

        wrapper = _ClassifyTextBatch(MockTextClassifierDescriptor(), "text", "label", ["a", "b"])
        restored = pickle.loads(pickle.dumps(wrapper))
        table = pa.table({"text": ["hello"]})
        result = restored(table)
        assert result.num_rows == 1
        assert result.column("label").to_pylist() == ["a"]


# ---------------------------------------------------------------------------
# vLLM Provider tests
# ---------------------------------------------------------------------------


class TestVLLMProvider:
    """Tests for native vLLM provider planning metadata."""

    def test_provider_loads(self):
        from vane.ai.provider import PROVIDERS

        assert "vllm" in PROVIDERS

    def test_descriptor_creates(self):
        from vane.ai.providers.vllm import VLLMPrompterDescriptor

        descriptor = VLLMPrompterDescriptor(model_name="Qwen/Qwen3-1.7B")

        assert descriptor.get_provider() == "vllm"
        assert descriptor.get_model() == "Qwen/Qwen3-1.7B"

    def test_descriptor_pickle_roundtrip(self):
        from vane.ai.providers.vllm import VLLMPrompterDescriptor

        descriptor = VLLMPrompterDescriptor(
            model_name="meta-llama/Llama-3.1-8B",
            system_message="You are a helpful assistant.",
            vllm_options={
                "engine_args": {"max_model_len": 2048},
                "generate_args": {"sampling_params": {"max_tokens": 256}},
                "gpus_per_actor": 1,
            },
        )

        restored = pickle.loads(pickle.dumps(descriptor))

        assert restored.model_name == descriptor.model_name
        assert restored.system_message == descriptor.system_message
        assert restored.vllm_options == descriptor.vllm_options

    def test_provider_get_prompter(self):
        from vane.ai.providers.vllm import VLLMPrompterDescriptor, VLLMProvider

        descriptor = VLLMProvider().get_prompter(
            model="Qwen/Qwen3-1.7B",
            system_message="Be concise.",
            engine_args={"max_model_len": 1024},
        )

        assert isinstance(descriptor, VLLMPrompterDescriptor)
        assert descriptor.model_name == "Qwen/Qwen3-1.7B"
        assert descriptor.system_message == "Be concise."
        assert descriptor.vllm_options["engine_args"] == {"max_model_len": 1024}

    def test_vllm_prompt_plan_is_not_an_executable_descriptor(self):
        from vane.ai.protocols import NativePrompterPlan, PrompterDescriptor
        from vane.ai.providers.vllm import NativeVLLMPromptPlan, VLLMPrompterDescriptor

        plan = VLLMPrompterDescriptor(model_name="test-model")

        assert NativeVLLMPromptPlan is VLLMPrompterDescriptor
        assert isinstance(plan, NativePrompterPlan)
        assert not isinstance(plan, PrompterDescriptor)
        assert not hasattr(plan, "instantiate")


class TestVLLMStructuredOutput:
    """Tests for native vLLM structured-output configuration."""

    def test_json_schema_from_pydantic(self):
        BaseModel = pytest.importorskip("pydantic").BaseModel

        from vane.ai.providers.vllm import _json_schema_from_return_format

        class Person(BaseModel):
            name: str
            age: int

        schema = _json_schema_from_return_format(Person)

        assert schema["type"] == "object"
        assert {"name", "age"} <= schema["properties"].keys()

    def test_json_schema_from_dict(self):
        from vane.ai.providers.vllm import _json_schema_from_return_format

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}

        assert _json_schema_from_return_format(schema) is schema

    def test_json_schema_from_none(self):
        from vane.ai.providers.vllm import _json_schema_from_return_format

        assert _json_schema_from_return_format(None) == {}

    def test_json_schema_bad_type(self):
        from vane.ai.providers.vllm import _json_schema_from_return_format

        with pytest.raises(TypeError, match="return_format must be"):
            _json_schema_from_return_format("bad")

    def test_descriptor_with_return_format_survives_pickle(self):
        import cloudpickle

        BaseModel = pytest.importorskip("pydantic").BaseModel

        from vane.ai.providers.vllm import VLLMPrompterDescriptor

        class Score(BaseModel):
            value: float
            label: str

        descriptor = VLLMPrompterDescriptor(
            model_name="test-model",
            return_format=Score,
        )

        restored = pickle.loads(cloudpickle.dumps(descriptor))

        assert restored.model_name == "test-model"
        assert restored.return_format(value=1.0, label="test").value == 1.0

    def test_descriptor_injects_schema_into_native_sampling_params(self):
        from vane.ai.providers.vllm import VLLMPrompterDescriptor

        schema = {"type": "object", "properties": {"n": {"type": "number"}}}
        descriptor = VLLMPrompterDescriptor(
            model_name="test-model",
            return_format=schema,
            vllm_options={"generate_args": {"sampling_params": {"max_tokens": 100}}},
        )

        sampling_params = descriptor.build_physical_vllm_options()["generate_args"]["sampling_params"]

        assert sampling_params["max_tokens"] == 100
        assert sampling_params["structured_outputs"] == {"type": "json", "value": schema}
        assert "guided_json" not in sampling_params


class TestUDFExecutionOptions:
    """Tests for AI UDF execution option plumbing."""

    def test_actor_requires_explicit_num_gpus(self):
        from vane.ai.functions import _map_batches_kwargs

        with pytest.raises(ValueError, match="num_gpus is required"):
            _map_batches_kwargs(UDFOptions(actor_number=2, batch_size=8), None)

    def test_actor_preserves_explicit_num_gpus(self):
        from vane.ai.functions import _map_batches_kwargs

        kwargs = _map_batches_kwargs(UDFOptions(actor_number=2, num_gpus=1), None)

        assert kwargs["actor_number"] == 2
        assert kwargs["gpus"] == 1

    def test_prompt_descriptors_preserve_num_gpus_but_embed_descriptors_do_not(self):
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor
        from vane.ai.providers.google import GooglePrompterDescriptor, GoogleTextEmbedderDescriptor
        from vane.ai.providers.openai import OpenAIPrompterDescriptor, OpenAITextEmbedderDescriptor

        with pytest.raises(TypeError, match="num_gpus"):
            OpenAITextEmbedderDescriptor(embed_options={"num_gpus": 1})
        assert OpenAIPrompterDescriptor(prompt_options={"num_gpus": 2}).get_udf_options().num_gpus == 2
        assert (
            AnthropicPrompterDescriptor(
                model_name="claude-test-model", prompt_options={"max_tokens": 64, "num_gpus": 3}
            )
            .get_udf_options()
            .num_gpus
            == 3
        )
        with pytest.raises(TypeError, match="num_gpus"):
            GoogleTextEmbedderDescriptor(model_name="gemini-embedding-001", embed_options={"num_gpus": 4})
        assert (
            GooglePrompterDescriptor(model_name="gemini-3.6-flash", prompt_options={"num_gpus": 5})
            .get_udf_options()
            .num_gpus
            == 5
        )

    def test_prompt_relation_defaults_keep_task_fanout_and_batch_size_one(self, monkeypatch):
        import vane
        from vane.ai import provider as provider_registry
        from vane.ai.protocols import PrompterDescriptor

        class _DefaultsPrompter:
            def prompt_batch(self, text):
                return [f"r:{item}" for item in text]

        @dataclass
        class _DefaultsPrompterDescriptor(PrompterDescriptor):
            def get_provider(self) -> str:
                return "mock_prompt_defaults"

            def get_model(self) -> str:
                return "mock"

            def get_options(self) -> dict[str, object]:
                return {}

            def get_udf_options(self) -> UDFOptions:
                return UDFOptions(actor_number=None, num_gpus=None, max_retries=0, on_error="raise", batch_size=None)

            def instantiate(self) -> _DefaultsPrompter:
                return _DefaultsPrompter()

        class _DefaultsProvider(Provider):
            @property
            def name(self) -> str:
                return "mock_prompt_defaults"

            def get_prompter(self, model=None, **options):
                return _DefaultsPrompterDescriptor()

        class FakeRel:
            def __init__(self):
                self.map_batches_kwargs = None

            def map_batches(self, udf, **kwargs):
                self.map_batches_kwargs = kwargs
                return "mapped"

            def select(self, *args, **kwargs):
                raise NotImplementedError

        monkeypatch.setitem(
            provider_registry.PROVIDERS,
            "mock_prompt_defaults",
            lambda name=None, **options: _DefaultsProvider(),
        )

        rel = FakeRel()
        result = vane.ai.prompt(rel, "chunk", provider="mock_prompt_defaults")

        assert result == "mapped"
        kwargs = rel.map_batches_kwargs
        assert kwargs["batch_size"] == 1
        assert "actor_number" not in kwargs


# ---------------------------------------------------------------------------
# Prompt semaphore tests
# ---------------------------------------------------------------------------


class TestPromptSemaphore:
    """Tests for max_api_concurrency semaphore in _PromptBatch."""

    def test_udf_options_has_max_api_concurrency(self):
        """UDFOptions dataclass includes max_api_concurrency field."""
        opts = UDFOptions()
        assert opts.max_api_concurrency is None
        opts2 = UDFOptions(max_api_concurrency=16)
        assert opts2.max_api_concurrency == 16

    def test_openai_prompter_default_concurrency(self):
        """OpenAI prompter defaults to max_api_concurrency=32."""
        try:
            from vane.ai.providers.openai import OpenAIPrompterDescriptor

            desc = OpenAIPrompterDescriptor(
                provider_options={"api_key": "test"},
            )
            opts = desc.get_udf_options()
            assert opts.max_api_concurrency == 32
        except ImportError:
            pytest.skip("openai not installed")

    def test_concurrency_override(self):
        """User can override max_api_concurrency via prompt_options."""
        try:
            from vane.ai.providers.openai import OpenAIPrompterDescriptor

            desc = OpenAIPrompterDescriptor(
                provider_options={"api_key": "test"},
                prompt_options={"max_api_concurrency": 8},
            )
            opts = desc.get_udf_options()
            assert opts.max_api_concurrency == 8
        except ImportError:
            pytest.skip("openai not installed")

    def test_semaphore_limits_concurrency(self):
        """Semaphore actually limits the number of concurrent calls."""
        import asyncio

        from vane.ai.functions import _PromptBatch

        peak_concurrent = 0
        current_concurrent = 0

        async def fake_prompt(messages):
            nonlocal peak_concurrent, current_concurrent
            current_concurrent += 1
            if current_concurrent > peak_concurrent:
                peak_concurrent = current_concurrent
            await asyncio.sleep(0.01)
            current_concurrent -= 1
            return f"reply to {messages[0]}"

        mock_desc = MagicMock()
        mock_prompter = MagicMock()
        mock_prompter.prompt = fake_prompt
        # No prompt_batch → forces async gather path
        del mock_prompter.prompt_batch
        mock_desc.instantiate.return_value = mock_prompter

        batch = _PromptBatch(mock_desc, "text", "response", max_api_concurrency=2)
        table = pa.table({"text": [f"msg{i}" for i in range(10)]})
        result = _drive(batch, table)

        assert result.num_rows == 10
        assert peak_concurrent <= 2

    def test_no_semaphore_when_none(self):
        """Without max_api_concurrency, all tasks run concurrently."""
        import asyncio

        from vane.ai.functions import _PromptBatch

        peak_concurrent = 0
        current_concurrent = 0

        async def fake_prompt(messages):
            nonlocal peak_concurrent, current_concurrent
            current_concurrent += 1
            if current_concurrent > peak_concurrent:
                peak_concurrent = current_concurrent
            await asyncio.sleep(0.01)
            current_concurrent -= 1
            return f"reply to {messages[0]}"

        mock_desc = MagicMock()
        mock_prompter = MagicMock()
        mock_prompter.prompt = fake_prompt
        del mock_prompter.prompt_batch
        mock_desc.instantiate.return_value = mock_prompter

        batch = _PromptBatch(mock_desc, "text", "response", max_api_concurrency=None)
        table = pa.table({"text": [f"msg{i}" for i in range(10)]})
        result = _drive(batch, table)

        assert result.num_rows == 10
        # Without semaphore, all 10 should run concurrently
        assert peak_concurrent == 10

    def test_prompt_batch_pickle_with_semaphore(self):
        """_PromptBatch with max_api_concurrency survives pickle."""
        from vane.ai.functions import _PromptBatch

        # Use a real picklable descriptor (not MagicMock)
        from vane.ai.providers.vllm import VLLMPrompterDescriptor

        desc = VLLMPrompterDescriptor(model_name="test-model")
        batch = _PromptBatch(desc, "text", "response", max_api_concurrency=16)
        restored = pickle.loads(pickle.dumps(batch))
        assert restored._max_api_concurrency == 16


# ---------------------------------------------------------------------------
# Anthropic Provider tests
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    """Tests for the Anthropic provider and descriptor."""

    def test_provider_registered(self):
        """Anthropic provider is in the registry."""
        from vane.ai.provider import PROVIDERS

        assert "anthropic" in PROVIDERS

    def test_descriptor_creates(self):
        """AnthropicPrompterDescriptor can be created."""
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        desc = AnthropicPrompterDescriptor(
            model_name="claude-sonnet-4-20250514",
            system_message="Be concise.",
            prompt_options={"max_tokens": 64},
        )
        assert desc.get_provider() == "anthropic"
        assert desc.get_model() == "claude-sonnet-4-20250514"

    def test_descriptor_pickle_roundtrip(self):
        """AnthropicPrompterDescriptor survives pickle."""
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        desc = AnthropicPrompterDescriptor(
            provider_options={"api_key": "test-key"},
            model_name="claude-sonnet-4-20250514",
            system_message="You are helpful.",
            prompt_options={"max_tokens": 64, "temperature": 0.7},
        )
        restored = pickle.loads(pickle.dumps(desc))
        assert restored.model_name == desc.model_name
        assert restored.system_message == desc.system_message
        assert restored.prompt_options == desc.prompt_options

    def test_udf_options(self):
        """Anthropic descriptor produces correct UDFOptions."""
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        desc = AnthropicPrompterDescriptor(model_name="claude-test-model", prompt_options={"max_tokens": 64})
        opts = desc.get_udf_options()
        assert opts.max_api_concurrency == 16

    def test_provider_get_prompter(self):
        """AnthropicProvider.get_prompter returns descriptor."""
        from vane.ai.providers.anthropic import (
            AnthropicPrompterDescriptor,
            AnthropicProvider,
        )

        provider = AnthropicProvider(api_key="test-key")
        desc = provider.get_prompter(
            model="claude-sonnet-4-20250514",
            system_message="Be brief.",
            max_tokens=64,
            temperature=0.5,
        )
        assert isinstance(desc, AnthropicPrompterDescriptor)
        assert desc.model_name == "claude-sonnet-4-20250514"
        assert desc.system_message == "Be brief."

    def test_provider_get_prompter_splits_call_client_options(self):
        """Anthropic call-level client options go to provider_options only."""
        from vane.ai._redaction import Secret
        from vane.ai.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="ctor-key", base_url="https://ctor.example")
        desc = provider.get_prompter(
            model="claude-test-model",
            api_key="call-key",
            base_url="https://call.example",
            timeout=30,
            max_tokens=64,
            max_api_concurrency=6,
            temperature=0,
        )

        # Credentials are sealed on the descriptor (vane#105).
        assert desc.provider_options == {
            "api_key": Secret("call-key"),
            "base_url": "https://call.example",
            "timeout": 30,
        }
        assert "api_key" not in desc.prompt_options
        assert "base_url" not in desc.prompt_options
        assert desc.prompt_options["max_api_concurrency"] == 6
        assert desc.prompt_options["temperature"] == 0

    def test_descriptor_requires_model_name(self):
        """AnthropicPrompterDescriptor.model_name is required."""
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        with pytest.raises(TypeError):
            AnthropicPrompterDescriptor()

    def test_get_prompter_without_model_raises(self):
        """Missing model fails fast, naming both fix paths."""
        from vane.ai.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        with pytest.raises(ValueError, match="No prompt model configured") as excinfo:
            provider.get_prompter()
        message = str(excinfo.value)
        assert "model=" in message
        assert "AnthropicProvider(prompt_model=" in message

    def test_provider_prompt_model_flows_through(self):
        """Provider-level prompt_model configures the descriptor model."""
        from vane.ai.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key", prompt_model="claude-config-model", max_tokens=64)
        desc = provider.get_prompter()
        assert desc.model_name == "claude-config-model"
        # prompt_model is provider-level config, never a request option.
        assert "prompt_model" not in desc.prompt_options
        assert "prompt_model" not in desc.provider_options

    def test_call_model_overrides_provider_prompt_model(self):
        """Call-site model beats provider-level prompt_model."""
        from vane.ai.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key", prompt_model="claude-config-model", max_tokens=64)
        desc = provider.get_prompter(model="claude-call-model")
        assert desc.model_name == "claude-call-model"

    @pytest.mark.parametrize("model", ["", "   "])
    def test_blank_call_model_rejected(self, model):
        """A blank call-site model is an error, not a provider-config fallback."""
        from vane.ai.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key", prompt_model="claude-config-model", max_tokens=64)
        with pytest.raises(ValueError, match="non-empty string"):
            provider.get_prompter(model=model)

    def test_blank_provider_prompt_model_rejected(self):
        """A blank provider-level prompt_model fails at expression-build time."""
        from vane.ai.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key", prompt_model="", max_tokens=64)
        with pytest.raises(ValueError, match="non-empty string"):
            provider.get_prompter()

    def test_ctor_prompt_model_is_keyword_only(self):
        """prompt_model must be passed by name; positional use is a TypeError."""
        from vane.ai.providers.anthropic import AnthropicProvider

        with pytest.raises(TypeError):
            AnthropicProvider("anthropic", "claude-config-model")

    def test_ctor_typo_is_a_type_error(self):
        """A misspelled ctor kwarg raises immediately instead of leaking into request options."""
        from vane.ai.providers.anthropic import AnthropicProvider

        with pytest.raises(TypeError):
            AnthropicProvider(api_key="test-key", promt_model="claude-config-model")


class TestAnthropicStrictOptions:
    """Tests for the strict Anthropic option contract (vane#146)."""

    def _provider(self, **kwargs):
        from vane.ai.providers.anthropic import AnthropicProvider

        return AnthropicProvider(api_key="test-key", prompt_model="claude-test-model", **kwargs)

    # --- unknown options are rejected pre-dispatch ------------------------

    def test_unknown_option_rejected(self):
        """A typo'd request option raises before dispatch, naming the key."""
        with pytest.raises(ValueError, match="thinkng") as excinfo:
            self._provider(max_tokens=64).get_prompter(thinkng={"type": "adaptive"})
        message = str(excinfo.value)
        assert "claude-test-model" in message
        assert "thinking" in message  # the supported list points at the fix

    def test_unknown_option_rejected_on_direct_descriptor_construction(self):
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        with pytest.raises(ValueError, match="metdata"):
            AnthropicPrompterDescriptor(
                model_name="claude-test-model",
                prompt_options={"max_tokens": 64, "metdata": {"user_id": "u"}},
            )

    def test_ctor_typo_is_rejected_by_the_named_ctor(self):
        """The ctor has no **options channel; a misspelled kwarg is a TypeError."""
        with pytest.raises(TypeError):
            self._provider(max_tokens=64, promt_model="claude-other-model")

    # --- known options forward to the API ---------------------------------

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_known_options_forward_to_messages_create(self):
        """thinking/metadata/tool_choice/service_tier reach the recorded call."""
        import asyncio

        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        desc = AnthropicPrompterDescriptor(
            model_name="claude-test-model",
            provider_options={"api_key": "test-key"},
            prompt_options={
                "max_tokens": 64,
                "thinking": {"type": "adaptive"},
                "metadata": {"user_id": "user-1"},
                "tool_choice": {"type": "auto"},
                "service_tier": "auto",
                "max_api_concurrency": 4,
                "on_error": "raise",
            },
        )
        prompter = desc.instantiate()

        captured: dict = {}
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "ok"
        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_response.stop_reason = "end_turn"

        async def mock_create(**kwargs):
            captured.update(kwargs)
            return mock_response

        prompter._client.messages.create = mock_create
        assert asyncio.run(prompter.prompt(("Hi",))) == "ok"

        assert captured["max_tokens"] == 64
        assert captured["thinking"] == {"type": "adaptive"}
        assert captured["metadata"] == {"user_id": "user-1"}
        assert captured["tool_choice"] == {"type": "auto"}
        assert captured["service_tier"] == "auto"
        # Vane execution options never reach the API.
        assert "max_api_concurrency" not in captured
        assert "on_error" not in captured

    # --- per-model unsupported combinations -------------------------------

    def test_sonnet5_sampling_option_rejected(self):
        """Sonnet 5 documents rejecting non-default sampling parameters."""
        with pytest.raises(ValueError, match="temperature") as excinfo:
            self._provider(max_tokens=64).get_prompter(model="claude-sonnet-5", temperature=0.2)
        assert "claude-sonnet-5" in str(excinfo.value)

    def test_sonnet5_snapshot_id_matches_family_prefix(self):
        with pytest.raises(ValueError, match="top_p"):
            self._provider(max_tokens=64).get_prompter(model="claude-sonnet-5-20260601", top_p=0.9)

    def test_sonnet46_sampling_option_passes(self):
        """The same option is fine on a model without the restriction."""
        desc = self._provider(max_tokens=64).get_prompter(model="claude-sonnet-4-6", temperature=0.2)
        assert desc.prompt_options["temperature"] == 0.2

    @pytest.mark.parametrize("default", [1, 1.0])
    def test_sonnet5_default_temperature_passes(self, default):
        """The documented default temperature (1.0) is accepted by the API."""
        desc = self._provider(max_tokens=64).get_prompter(model="claude-sonnet-5", temperature=default)
        assert desc.prompt_options["temperature"] == default

    def test_sonnet5_boolean_temperature_rejected(self):
        """True == 1 in Python, but a bool is not the documented default."""
        with pytest.raises(ValueError, match="temperature"):
            self._provider(max_tokens=64).get_prompter(model="claude-sonnet-5", temperature=True)

    def test_sonnet5_default_like_top_p_still_rejected(self):
        """top_p has no documented default, so any explicit value is non-default."""
        with pytest.raises(ValueError, match="top_p"):
            self._provider(max_tokens=64).get_prompter(model="claude-sonnet-5", top_p=1.0)

    @pytest.mark.parametrize("model", ["claude-fable-5", "claude-mythos-5", "claude-mythos-preview"])
    def test_fable_mythos_sampling_option_rejected(self, model):
        """The Fable/Mythos families document the same non-default rejection."""
        with pytest.raises(ValueError, match="temperature") as excinfo:
            self._provider(max_tokens=64).get_prompter(model=model, temperature=0.2)
        assert model in str(excinfo.value)

    def test_fable_default_temperature_passes(self):
        """The documented default temperature is accepted on Fable too."""
        desc = self._provider(max_tokens=64).get_prompter(model="claude-fable-5", temperature=1.0)
        assert desc.prompt_options["temperature"] == 1.0

    def test_fable_top_p_rejected(self):
        with pytest.raises(ValueError, match="top_p"):
            self._provider(max_tokens=64).get_prompter(model="claude-fable-5", top_p=1.0)

    # --- per-model thinking.type restrictions -----------------------------

    @pytest.mark.parametrize(
        "model",
        ["claude-sonnet-5", "claude-opus-4-7", "claude-fable-5", "claude-sonnet-5-20260601"],
    )
    def test_manual_thinking_rejected_on_adaptive_only_models(self, model):
        """The removed manual form fails pre-dispatch on adaptive-only families."""
        with pytest.raises(ValueError, match="thinking.type") as excinfo:
            self._provider(max_tokens=64).get_prompter(model=model, thinking={"type": "enabled", "budget_tokens": 1024})
        assert model in str(excinfo.value)

    @pytest.mark.parametrize("model", ["claude-fable-5", "claude-mythos-5", "claude-mythos-preview"])
    def test_disabled_thinking_rejected_on_always_on_models(self, model):
        """Thinking cannot be turned off where it is always on."""
        with pytest.raises(ValueError, match="disabled"):
            self._provider(max_tokens=64).get_prompter(model=model, thinking={"type": "disabled"})

    @pytest.mark.parametrize("model", ["claude-mythos-preview", "claude-opus-4-6"])
    def test_manual_thinking_passes_where_still_supported(self, model):
        """Mythos Preview and the 4.6 models still accept the manual form."""
        desc = self._provider(max_tokens=64).get_prompter(
            model=model, thinking={"type": "enabled", "budget_tokens": 1024}
        )
        assert desc.prompt_options["thinking"]["type"] == "enabled"

    @pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-opus-4-5", "claude-sonnet-4-5"])
    def test_adaptive_thinking_rejected_on_extended_only_models(self, model):
        """Extended-thinking-only models reject the adaptive form."""
        with pytest.raises(ValueError, match="adaptive"):
            self._provider(max_tokens=64).get_prompter(model=model, thinking={"type": "adaptive"})

    def test_adaptive_thinking_passes_on_adaptive_models(self):
        desc = self._provider(max_tokens=64).get_prompter(model="claude-sonnet-5", thinking={"type": "adaptive"})
        assert desc.prompt_options["thinking"] == {"type": "adaptive"}

    # --- extra_body cannot bypass the option contract ---------------------

    @pytest.mark.parametrize(
        "smuggled",
        [
            {"thinking": {"type": "enabled", "budget_tokens": 1024}},
            {"temperature": 0.2},
            {"max_tokens": 5},
            {"model": "claude-other-model"},
            {"stream": True},
        ],
    )
    def test_extra_body_with_validated_field_rejected(self, smuggled):
        """Vane-owned and known request fields cannot ride in extra_body."""
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        with pytest.raises(ValueError, match="extra_body") as excinfo:
            AnthropicPrompterDescriptor(
                model_name="claude-test-model",
                prompt_options={"max_tokens": 64, "extra_body": smuggled},
            )
        assert next(iter(smuggled)) in str(excinfo.value)

    def test_extra_body_non_mapping_rejected(self):
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        with pytest.raises(ValueError, match="extra_body"):
            AnthropicPrompterDescriptor(
                model_name="claude-test-model",
                prompt_options={"max_tokens": 64, "extra_body": ["thinking"]},
            )

    def test_extra_body_with_unmodeled_field_passes(self):
        """extra_body keeps its purpose: fields the SDK does not model."""
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        desc = AnthropicPrompterDescriptor(
            model_name="claude-test-model",
            prompt_options={"max_tokens": 64, "extra_body": {"future_api_field": {"enabled": True}}},
        )
        assert desc.prompt_options["extra_body"] == {"future_api_field": {"enabled": True}}

    # --- explicit max_tokens chain ----------------------------------------

    def test_get_prompter_without_max_tokens_raises(self):
        """Missing max_tokens fails fast, naming both fix paths."""
        with pytest.raises(ValueError, match="No max_tokens configured") as excinfo:
            self._provider().get_prompter()
        message = str(excinfo.value)
        assert "max_tokens=" in message
        assert "AnthropicProvider(max_tokens=" in message

    def test_descriptor_requires_max_tokens(self):
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        with pytest.raises(ValueError, match="No max_tokens configured"):
            AnthropicPrompterDescriptor(model_name="claude-test-model")

    def test_call_max_tokens_flows_through(self):
        desc = self._provider().get_prompter(max_tokens=256)
        assert desc.prompt_options["max_tokens"] == 256

    def test_provider_max_tokens_flows_through(self):
        desc = self._provider(max_tokens=128).get_prompter()
        assert desc.prompt_options["max_tokens"] == 128

    def test_call_max_tokens_overrides_provider_config(self):
        desc = self._provider(max_tokens=128).get_prompter(max_tokens=512)
        assert desc.prompt_options["max_tokens"] == 512

    def test_none_max_tokens_falls_back_to_provider_config(self):
        """An explicit max_tokens=None is unconfigured, not a configured value."""
        desc = self._provider(max_tokens=128).get_prompter(max_tokens=None)
        assert desc.prompt_options["max_tokens"] == 128

    def test_none_max_tokens_without_config_raises(self):
        with pytest.raises(ValueError, match="No max_tokens configured"):
            self._provider().get_prompter(max_tokens=None)

    def test_none_max_tokens_rejected_on_direct_descriptor_construction(self):
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        with pytest.raises(ValueError, match="No max_tokens configured"):
            AnthropicPrompterDescriptor(
                model_name="claude-test-model",
                prompt_options={"max_tokens": None},
            )

    @pytest.mark.parametrize("bad", ["64", 64.0, False, -1])
    def test_non_integer_max_tokens_rejected(self, bad):
        """max_tokens must be a non-negative int; False must not satisfy == 0."""
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        with pytest.raises(ValueError, match="non-negative integer"):
            AnthropicPrompterDescriptor(
                model_name="claude-test-model",
                prompt_options={"max_tokens": bad},
            )

    def test_ctor_max_tokens_is_keyword_only(self):
        from vane.ai.providers.anthropic import AnthropicProvider

        with pytest.raises(TypeError):
            AnthropicProvider("anthropic", "claude-config-model", 64)

    # --- truncation surfaces as an error ----------------------------------

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_truncated_text_response_raises(self):
        import asyncio

        from vane.ai.providers.anthropic import AnthropicPrompter

        prompter = AnthropicPrompter(
            provider_options={"api_key": "test-key"},
            model="claude-test-model",
            max_tokens=64,
        )

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "partial answ"
        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_response.stop_reason = "max_tokens"

        async def mock_create(**kwargs):
            return mock_response

        prompter._client.messages.create = mock_create
        with pytest.raises(ValueError, match="truncated") as excinfo:
            asyncio.run(prompter.prompt(("Hi",)))
        message = str(excinfo.value)
        assert "claude-test-model" in message
        assert "max_tokens=64" in message

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_truncated_structured_response_raises_instead_of_none(self):
        import asyncio

        from vane.ai.providers.anthropic import AnthropicPrompter

        prompter = AnthropicPrompter(
            provider_options={"api_key": "test-key"},
            model="claude-test-model",
            max_tokens=64,
            return_format=dict,
        )

        # Truncation cut the turn before a complete tool_use block emerged.
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "max_tokens"

        async def mock_create(**kwargs):
            return mock_response

        prompter._client.messages.create = mock_create
        with pytest.raises(ValueError, match="truncated"):
            asyncio.run(prompter.prompt(("Extract data",)))

    # --- documented-invalid option combinations ---------------------------

    def test_return_format_with_caller_tool_choice_rejected(self):
        """A caller tool_choice would be overridden by extract_data — reject it."""
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        with pytest.raises(ValueError, match="tool_choice"):
            AnthropicPrompterDescriptor(
                model_name="claude-test-model",
                return_format=dict,
                prompt_options={"max_tokens": 64, "tool_choice": {"type": "none"}},
            )

    def test_return_format_with_manual_thinking_rejected(self):
        """Forced tool use plus manual extended thinking is API-rejected."""
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        with pytest.raises(ValueError, match="manual extended thinking"):
            AnthropicPrompterDescriptor(
                model_name="claude-test-model",
                return_format=dict,
                prompt_options={"max_tokens": 64, "thinking": {"type": "enabled", "budget_tokens": 1024}},
            )

    def test_return_format_with_adaptive_thinking_passes(self):
        """Adaptive thinking supports forced tool use and stays allowed."""
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        desc = AnthropicPrompterDescriptor(
            model_name="claude-test-model",
            return_format=dict,
            prompt_options={"max_tokens": 64, "thinking": {"type": "adaptive"}},
        )
        assert desc.return_format is dict

    # --- forced tool choice needs tool definitions ------------------------

    @pytest.mark.parametrize(
        "tool_choice",
        [{"type": "any"}, {"type": "tool", "name": "extract_data"}],
    )
    def test_forced_tool_choice_without_return_format_rejected(self, tool_choice):
        """Vane sends no tools without return_format, so forced tool use cannot be satisfied."""
        with pytest.raises(ValueError, match="tool_choice"):
            self._provider(max_tokens=64).get_prompter(tool_choice=tool_choice)

    @pytest.mark.parametrize("tool_choice", [{"type": "auto"}, {"type": "none"}])
    def test_no_tools_safe_tool_choice_passes(self, tool_choice):
        """auto/none remain valid without tool definitions."""
        desc = self._provider(max_tokens=64).get_prompter(tool_choice=tool_choice)
        assert desc.prompt_options["tool_choice"] == tool_choice

    # --- execution options flow into UDFOptions ---------------------------

    def test_concurrency_and_batch_size_flow_into_udf_options(self):
        """The raw-dict path carries concurrency/batch_size instead of dropping them."""
        desc = self._provider(max_tokens=64).get_prompter(concurrency=3, batch_size=7)
        udf_options = desc.get_udf_options()
        assert udf_options.actor_number == 3
        assert udf_options.batch_size == 7

    def test_actor_number_and_matching_concurrency_pass(self):
        """Equal values for the alias pair are not a conflict."""
        desc = self._provider(max_tokens=64).get_prompter(actor_number=3, concurrency=3)
        assert desc.get_udf_options().actor_number == 3

    def test_conflicting_actor_number_and_concurrency_rejected(self):
        """concurrency aliases actor_number; disagreeing values raise."""
        with pytest.raises(ValueError, match="alias"):
            self._provider(max_tokens=64).get_prompter(actor_number=2, concurrency=3)

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_structured_output_dispatch_kwargs(self):
        """The dispatched call carries the forced extract_data tool choice."""
        import asyncio

        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        desc = AnthropicPrompterDescriptor(
            model_name="claude-test-model",
            provider_options={"api_key": "test-key"},
            return_format=dict,
            prompt_options={"max_tokens": 64},
        )
        prompter = desc.instantiate()

        captured: dict = {}
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.input = {"answer": 42}
        mock_response = MagicMock()
        mock_response.content = [tool_block]
        mock_response.stop_reason = "end_turn"

        async def mock_create(**kwargs):
            captured.update(kwargs)
            return mock_response

        prompter._client.messages.create = mock_create
        assert asyncio.run(prompter.prompt(("Extract",))) == {"answer": 42}

        assert captured["tool_choice"] == {"type": "tool", "name": "extract_data"}
        assert captured["tools"][0]["name"] == "extract_data"
        assert captured["max_tokens"] == 64

    # --- max_tokens=0 cache prewarming ------------------------------------

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_zero_max_tokens_prewarm_returns_none(self):
        """A max_tokens=0 response with stop_reason=max_tokens is prewarming, not truncation."""
        import asyncio

        from vane.ai.providers.anthropic import AnthropicPrompter

        prompter = AnthropicPrompter(
            provider_options={"api_key": "test-key"},
            model="claude-test-model",
            max_tokens=0,
        )

        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "max_tokens"

        async def mock_create(**kwargs):
            return mock_response

        prompter._client.messages.create = mock_create
        assert asyncio.run(prompter.prompt(("cached prefix",))) is None

    def test_zero_max_tokens_descriptor_constructs(self):
        """Plain max_tokens=0 (cache prewarming) passes descriptor validation."""
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        desc = AnthropicPrompterDescriptor(
            model_name="claude-test-model",
            prompt_options={"max_tokens": 0, "cache_control": {"type": "ephemeral"}},
        )
        assert desc.prompt_options["max_tokens"] == 0

    @pytest.mark.parametrize(
        "options",
        [
            {"max_tokens": 0, "thinking": {"type": "enabled", "budget_tokens": 1024}},
            {"max_tokens": 0, "tool_choice": {"type": "any"}},
            {"max_tokens": 0, "tool_choice": {"type": "tool", "name": "extract_data"}},
            {"max_tokens": 0, "output_config": {"format": "json"}},
        ],
    )
    def test_zero_max_tokens_rejects_output_implying_options(self, options):
        """Options that imply output are rejected with max_tokens=0 pre-dispatch."""
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        with pytest.raises(ValueError, match="cache-prewarming"):
            AnthropicPrompterDescriptor(model_name="claude-test-model", prompt_options=options)

    def test_zero_max_tokens_rejects_return_format(self):
        """Structured output implies output and cannot ride a prewarming request."""
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        with pytest.raises(ValueError, match="cache-prewarming"):
            AnthropicPrompterDescriptor(
                model_name="claude-test-model",
                return_format=dict,
                prompt_options={"max_tokens": 0},
            )

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_positive_max_tokens_truncation_still_raises(self):
        """The prewarming carve-out does not weaken truncation detection."""
        import asyncio

        from vane.ai.providers.anthropic import AnthropicPrompter

        prompter = AnthropicPrompter(
            provider_options={"api_key": "test-key"},
            model="claude-test-model",
            max_tokens=1,
        )

        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "max_tokens"

        async def mock_create(**kwargs):
            return mock_response

        prompter._client.messages.create = mock_create
        with pytest.raises(ValueError, match="truncated"):
            asyncio.run(prompter.prompt(("Hi",)))


# ---------------------------------------------------------------------------
# Google Provider tests
# ---------------------------------------------------------------------------


class TestGoogleProvider:
    """Tests for the Google Generative AI provider and descriptor."""

    def test_provider_registered(self):
        """Google provider is in the registry."""
        from vane.ai.provider import PROVIDERS

        assert "google" in PROVIDERS

    def test_embedder_descriptor_creates(self):
        """GoogleTextEmbedderDescriptor derives dims for a known model."""
        from vane.ai.providers.google import GoogleTextEmbedderDescriptor

        desc = GoogleTextEmbedderDescriptor(
            model_name="gemini-embedding-001",
        )
        assert desc.get_provider() == "google"
        assert desc.get_model() == "gemini-embedding-001"
        dims = desc.get_dimensions()
        assert dims.size == 3072

    def test_embedder_descriptor_custom_dims(self):
        """GoogleTextEmbedderDescriptor supports custom dimensions."""
        from vane.ai.providers.google import GoogleTextEmbedderDescriptor

        desc = GoogleTextEmbedderDescriptor(
            model_name="gemini-embedding-001",
            dimensions=256,
        )
        dims = desc.get_dimensions()
        assert dims.size == 256

    def test_embedder_descriptor_pickle(self):
        """GoogleTextEmbedderDescriptor survives pickle."""
        from vane.ai.providers.google import GoogleTextEmbedderDescriptor

        desc = GoogleTextEmbedderDescriptor(
            provider_options={"api_key": "test"},
            model_name="gemini-embedding-001",
            dimensions=256,
        )
        restored = pickle.loads(pickle.dumps(desc))
        assert restored.model_name == desc.model_name
        assert restored.dimensions == desc.dimensions

    def test_prompter_descriptor_creates(self):
        """GooglePrompterDescriptor can be created."""
        from vane.ai.providers.google import GooglePrompterDescriptor

        desc = GooglePrompterDescriptor(
            model_name="gemini-3.6-flash",
            system_message="Be helpful.",
        )
        assert desc.get_provider() == "google"
        assert desc.get_model() == "gemini-3.6-flash"

    def test_prompter_descriptor_pickle(self):
        """GooglePrompterDescriptor survives pickle."""
        from vane.ai.providers.google import GooglePrompterDescriptor

        desc = GooglePrompterDescriptor(
            provider_options={"api_key": "test"},
            model_name="gemini-2.5-pro",
            system_message="Be concise.",
            prompt_options={"temperature": 0.5},
        )
        restored = pickle.loads(pickle.dumps(desc))
        assert restored.model_name == desc.model_name
        assert restored.system_message == desc.system_message

    def test_prompter_udf_options(self):
        """Google prompter descriptor produces correct UDFOptions."""
        from vane.ai.providers.google import GooglePrompterDescriptor

        desc = GooglePrompterDescriptor(model_name="gemini-3.6-flash")
        opts = desc.get_udf_options()
        assert opts.max_api_concurrency == 16

    def test_descriptor_model_name_required(self):
        """Descriptors cannot be constructed without an explicit model."""
        from vane.ai.providers.google import (
            GooglePrompterDescriptor,
            GoogleTextEmbedderDescriptor,
        )

        with pytest.raises(TypeError):
            GooglePrompterDescriptor()
        with pytest.raises(TypeError):
            GoogleTextEmbedderDescriptor()

    def test_provider_get_prompter(self):
        """GoogleProvider.get_prompter returns descriptor."""
        from vane.ai.providers.google import GooglePrompterDescriptor, GoogleProvider

        provider = GoogleProvider(api_key="test")
        desc = provider.get_prompter(
            model="gemini-3.6-flash",
            system_message="Summarize.",
        )
        assert isinstance(desc, GooglePrompterDescriptor)
        assert desc.model_name == "gemini-3.6-flash"

    def test_get_prompter_missing_model_raises(self):
        """No call-site model and no provider config fails fast."""
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test")
        with pytest.raises(ValueError) as excinfo:
            provider.get_prompter()
        message = str(excinfo.value)
        assert "No prompt model configured" in message
        assert "model=" in message
        assert "GoogleProvider(prompt_model=...)" in message

    def test_get_prompter_provider_config_flow_through(self):
        """Provider-level prompt_model is used when the call omits model."""
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test", prompt_model="gemini-3.6-flash")
        desc = provider.get_prompter()
        assert desc.model_name == "gemini-3.6-flash"

    def test_get_prompter_call_model_beats_provider_config(self):
        """Call-site model overrides the provider-level prompt_model."""
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test", prompt_model="gemini-2.5-pro")
        desc = provider.get_prompter(model="gemini-3.6-flash")
        assert desc.model_name == "gemini-3.6-flash"

    def test_get_text_embedder_missing_model_raises(self):
        """No call-site model and no provider config fails fast."""
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test")
        with pytest.raises(ValueError) as excinfo:
            provider.get_text_embedder()
        message = str(excinfo.value)
        assert "No embedding model configured" in message
        assert "model=" in message
        assert "GoogleProvider(embedding_model=...)" in message

    def test_get_text_embedder_provider_config_flow_through(self):
        """Provider-level embedding_model is used when the call omits model."""
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test", embedding_model="gemini-embedding-001")
        desc = provider.get_text_embedder()
        assert desc.model_name == "gemini-embedding-001"
        assert desc.get_dimensions().size == 3072

    def test_get_text_embedder_call_model_beats_provider_config(self):
        """Call-site model overrides the provider-level embedding_model."""
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test", embedding_model="gemini-embedding-2")
        desc = provider.get_text_embedder(model="gemini-embedding-001")
        assert desc.model_name == "gemini-embedding-001"

    def test_get_text_embedder_provider_dimensions_flow_through(self):
        """Provider-level embedding_dimensions covers models without metadata."""
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(
            api_key="test",
            embedding_model="custom-tuned-embedder",
            embedding_dimensions=1024,
        )
        desc = provider.get_text_embedder()
        assert desc.get_dimensions().size == 1024

    def test_get_text_embedder_call_dimensions_beat_provider_config(self):
        """Call-site dimensions override the provider-level configuration."""
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test", embedding_dimensions=1024)
        desc = provider.get_text_embedder(model="gemini-embedding-001", dimensions=256)
        assert desc.get_dimensions().size == 256

    def test_provider_ctor_typo_raises_type_error(self):
        """A mistyped constructor kwarg cannot leak into request options."""
        from vane.ai.providers.google import GoogleProvider

        with pytest.raises(TypeError):
            GoogleProvider(promt_model="gemini-3.6-flash")

    @pytest.mark.parametrize("model", ["gemini-3.6-flash", "gemini-3.5-flash-lite"])
    @pytest.mark.parametrize("option", ["temperature", "top_p", "top_k"])
    def test_unsupported_sampling_option_rejected(self, model, option):
        """Deprecated sampling params are rejected before dispatch."""
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test")
        with pytest.raises(ValueError) as excinfo:
            provider.get_prompter(model=model, **{option: 0.5})
        message = str(excinfo.value)
        assert option in message
        assert model in message

    def test_unsupported_option_rejected_on_direct_descriptor_construction(self):
        """Validation also covers direct descriptor construction."""
        from vane.ai.providers.google import GooglePrompterDescriptor

        with pytest.raises(ValueError, match="does not support options"):
            GooglePrompterDescriptor(
                model_name="gemini-3.6-flash",
                prompt_options={"temperature": 0.2},
            )

    def test_sampling_options_pass_for_untabled_model(self):
        """Models without a capability-table entry accept sampling params."""
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test")
        desc = provider.get_prompter(
            model="gemini-2.5-pro",
            temperature=0.2,
            top_p=0.9,
            top_k=40,
        )
        assert desc.prompt_options["temperature"] == 0.2
        assert desc.prompt_options["top_p"] == 0.9
        assert desc.prompt_options["top_k"] == 40

    def test_embedder_unknown_model_without_dimensions_raises(self):
        """Unknown embedding model without explicit dimensions fails fast."""
        from vane.ai.providers.google import GoogleProvider, GoogleTextEmbedderDescriptor

        provider = GoogleProvider(api_key="test")
        with pytest.raises(ValueError) as excinfo:
            provider.get_text_embedder(model="custom-tuned-embedder")
        message = str(excinfo.value)
        assert "Cannot derive embedding dimensions" in message
        assert "dimensions=" in message
        assert "GoogleProvider(embedding_dimensions=...)" in message

        with pytest.raises(ValueError, match="Cannot derive embedding dimensions"):
            GoogleTextEmbedderDescriptor(model_name="custom-tuned-embedder")

    def test_embedder_unknown_model_with_explicit_dimensions_flows(self):
        """Explicit dimensions make an unknown model usable."""
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test")
        desc = provider.get_text_embedder(model="custom-tuned-embedder", dimensions=512)
        assert desc.get_dimensions().size == 512

    def test_embedder_zero_dimensions_rejected(self):
        """dimensions=0 is rejected instead of silently treated as unset."""
        from vane.ai.providers.google import GoogleTextEmbedderDescriptor

        with pytest.raises(ValueError, match="positive integer"):
            GoogleTextEmbedderDescriptor(model_name="custom-tuned-embedder", dimensions=0)

    @pytest.mark.parametrize("dimensions", [64, 4096])
    def test_embedder_dimensions_outside_documented_range_rejected(self, dimensions):
        """Dimensions conflicting with trusted model metadata fail fast."""
        from vane.ai.providers.google import GoogleTextEmbedderDescriptor

        with pytest.raises(ValueError, match="output dimensionality"):
            GoogleTextEmbedderDescriptor(model_name="gemini-embedding-001", dimensions=dimensions)

    @pytest.mark.parametrize("dimensions", [256.5, True, "256"])
    def test_embedder_non_integer_dimensions_rejected(self, dimensions):
        """Non-integer dimensions raise the promised configuration error."""
        from vane.ai.providers.google import GoogleTextEmbedderDescriptor

        with pytest.raises(ValueError, match="positive integer"):
            GoogleTextEmbedderDescriptor(model_name="custom-tuned-embedder", dimensions=dimensions)

    @pytest.mark.parametrize("model", ["gemini-3.6-flash", "models/gemini-3.6-flash"])
    def test_unsupported_option_rejected_for_both_model_name_forms(self, model):
        """The models/ resource form cannot bypass the capability table."""
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test")
        with pytest.raises(ValueError, match="does not support options"):
            provider.get_prompter(model=model, temperature=0.2)

    @pytest.mark.parametrize("model", ["gemini-embedding-001", "models/gemini-embedding-001"])
    def test_embedder_trusted_dimensions_for_both_model_name_forms(self, model):
        """The models/ resource form still hits the trusted dimensions table."""
        from vane.ai.providers.google import GoogleProvider

        desc = GoogleProvider(api_key="test").get_text_embedder(model=model)
        assert desc.model_name == model
        assert desc.get_dimensions().size == 3072

    @pytest.mark.parametrize("model", ["", "   "])
    def test_get_prompter_blank_call_model_rejected(self, model):
        """A blank call-site model is an error, not a provider-config fallback."""
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test", prompt_model="gemini-2.5-pro")
        with pytest.raises(ValueError, match="non-empty string"):
            provider.get_prompter(model=model)

    def test_get_prompter_blank_provider_model_rejected(self):
        """A blank provider-level prompt_model fails at expression-build time."""
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test", prompt_model="")
        with pytest.raises(ValueError, match="non-empty string"):
            provider.get_prompter()

    @pytest.mark.parametrize("model", ["", "   "])
    def test_get_text_embedder_blank_call_model_rejected(self, model):
        """A blank call-site model is an error, not a provider-config fallback."""
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test", embedding_model="gemini-embedding-001")
        with pytest.raises(ValueError, match="non-empty string"):
            provider.get_text_embedder(model=model)

    def test_get_text_embedder_blank_provider_model_rejected(self):
        """A blank provider-level embedding_model fails at expression-build time."""
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test", embedding_model="")
        with pytest.raises(ValueError, match="non-empty string"):
            provider.get_text_embedder()

    def test_provider_get_prompter_splits_call_client_options(self):
        """Google prompt call-level client options go to provider_options only."""
        from vane.ai._redaction import Secret
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="ctor-key")
        desc = provider.get_prompter(
            model="gemini-2.5-pro",
            api_key="call-key",
            max_api_concurrency=7,
            temperature=0,
        )

        # Credentials are sealed on the descriptor (vane#105).
        assert desc.provider_options == {"api_key": Secret("call-key")}
        assert "api_key" not in desc.prompt_options
        assert desc.prompt_options["max_api_concurrency"] == 7
        assert desc.prompt_options["temperature"] == 0

    def test_provider_get_text_embedder(self):
        """GoogleProvider.get_text_embedder returns descriptor."""
        from vane.ai.providers.google import (
            GoogleProvider,
            GoogleTextEmbedderDescriptor,
        )

        provider = GoogleProvider(api_key="test")
        desc = provider.get_text_embedder(
            model="gemini-embedding-001",
            dimensions=256,
        )
        assert isinstance(desc, GoogleTextEmbedderDescriptor)
        assert desc.dimensions == 256

    def test_provider_get_text_embedder_splits_call_client_options(self):
        """Google embedding call-level client options go to provider_options only."""
        from vane.ai._redaction import Secret
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="ctor-key")
        desc = provider.get_text_embedder(
            model="gemini-embedding-001",
            api_key="call-key",
            task_type="RETRIEVAL_QUERY",
        )

        # Credentials are sealed on the descriptor (vane#105).
        assert desc.provider_options == {"api_key": Secret("call-key")}
        assert "api_key" not in desc.embed_options
        assert desc.embed_options["task_type"] == "RETRIEVAL_QUERY"

        with pytest.raises(TypeError, match="on_error"):
            provider.get_text_embedder(model="gemini-embedding-001", on_error="log")


class TestGoogleEmbeddingRowPreservation:
    """embed_text must return exactly one embedding per input row.

    gemini-embedding-2 aggregates multiple direct string inputs into a
    single embedding, so each input is sent as its own ``types.Content``
    and the result count is verified against the input count.
    """

    def _make_embedder(self, embeddings):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch

        from vane.ai.providers.google import GoogleTextEmbedder

        with patch("google.genai.Client"):
            embedder = GoogleTextEmbedder(
                provider_options={"api_key": "test"},
                model="gemini-embedding-2",
            )
        embed_content = AsyncMock(return_value=SimpleNamespace(embeddings=embeddings))
        embedder._client.aio.models.embed_content = embed_content
        return embedder, embed_content

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_two_inputs_yield_two_vectors_via_separate_contents(self):
        import asyncio
        from types import SimpleNamespace

        from google.genai import types

        embedder, embed_content = self._make_embedder(
            [
                SimpleNamespace(values=[0.1, 0.2]),
                SimpleNamespace(values=[0.3, 0.4]),
            ]
        )

        result = asyncio.run(embedder.embed_text(["first row", "second row"]))

        assert len(result) == 2
        contents = embed_content.call_args.kwargs["contents"]
        assert len(contents) == 2
        assert all(isinstance(c, types.Content) for c in contents)
        assert contents[0].parts[0].text == "first row"
        assert contents[1].parts[0].text == "second row"

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_embedding_count_mismatch_raises(self):
        import asyncio
        from types import SimpleNamespace

        embedder, _ = self._make_embedder([SimpleNamespace(values=[0.1, 0.2])])

        with pytest.raises(TypeError, match="preserve row count and order"):
            asyncio.run(embedder.embed_text(["first row", "second row"]))


class TestGoogleEmbeddingBatching:
    """Tests for Google embedding request chunking under the API batch cap."""

    def _recording_embedder(self, model, dimensions=None, options=None):
        """Build a GoogleTextEmbedder around a recording embed_content mock.

        The fake response derives each embedding value from the input text
        (``"txt-<n>"`` embeds to ``[float(n)]``), so result order can be
        asserted independently of request order.
        """
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.google import GoogleTextEmbedder

        calls: list[dict] = []

        async def fake_embed_content(**kwargs):
            calls.append(kwargs)
            response = MagicMock()
            response.embeddings = []
            for item in kwargs["contents"]:
                embedding = MagicMock()
                embedding.values = [float(item.parts[0].text.rsplit("-", 1)[1])]
                response.embeddings.append(embedding)
            return response

        embedder = GoogleTextEmbedder.__new__(GoogleTextEmbedder)
        embedder._client = MagicMock()
        embedder._client.aio.models.embed_content = AsyncMock(side_effect=fake_embed_content)
        embedder._model = model
        embedder._dimensions = dimensions
        embedder._options = dict(options or {})
        return embedder, calls

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_oversized_batch_chunked_under_cap(self):
        """250 inputs become 3 requests of <=100, concatenated in order."""
        import asyncio

        embedder, calls = self._recording_embedder("gemini-embedding-001")
        texts = [f"txt-{i}" for i in range(250)]

        result = asyncio.run(embedder.embed_text(texts))

        assert [len(call["contents"]) for call in calls] == [100, 100, 50]
        assert all(call["model"] == "gemini-embedding-001" for call in calls)
        # Chunks cover the inputs in order without overlap.
        sent = [item.parts[0].text for call in calls for item in call["contents"]]
        assert sent == texts
        # One embedding per input row, in input order.
        assert len(result) == len(texts)
        assert [e[0] for e in result] == [float(i) for i in range(250)]

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    @pytest.mark.parametrize("count", [1, 100])
    def test_batch_at_or_under_cap_is_single_request(self, count):
        """Batches within the cap go out as one request."""
        import asyncio

        embedder, calls = self._recording_embedder("gemini-embedding-001")
        texts = [f"txt-{i}" for i in range(count)]

        result = asyncio.run(embedder.embed_text(texts))

        assert len(calls) == 1
        assert len(calls[0]["contents"]) == count
        assert len(result) == count

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_empty_batch_makes_no_requests(self):
        import asyncio

        embedder, calls = self._recording_embedder("gemini-embedding-001")

        result = asyncio.run(embedder.embed_text([]))

        assert calls == []
        assert result == []

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_aggregating_model_batches_via_separate_contents(self):
        """gemini-embedding-2 aggregates multiple direct string inputs, so
        each input travels as its own Content and full-size chunks stay
        row-preserving without falling back to one request per input."""
        import asyncio

        embedder, calls = self._recording_embedder("gemini-embedding-2")
        texts = ["txt-0", "txt-1", "txt-2"]

        result = asyncio.run(embedder.embed_text(texts))

        assert len(calls) == 1
        assert [item.parts[0].text for item in calls[0]["contents"]] == texts
        assert len(result) == len(texts)
        assert [e[0] for e in result] == [0.0, 1.0, 2.0]

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_result_count_mismatch_raises(self):
        """A model that does not embed inputs individually is rejected."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.google import GoogleTextEmbedder

        async def aggregate_embed_content(**kwargs):
            response = MagicMock()
            embedding = MagicMock()
            embedding.values = [1.0]
            response.embeddings = [embedding]  # one embedding for N inputs
            return response

        embedder = GoogleTextEmbedder.__new__(GoogleTextEmbedder)
        embedder._client = MagicMock()
        embedder._client.aio.models.embed_content = AsyncMock(side_effect=aggregate_embed_content)
        embedder._model = "custom-aggregating-embedder"
        embedder._dimensions = None
        embedder._options = {}

        with pytest.raises(TypeError, match="returned 1 embeddings for 3 inputs"):
            asyncio.run(embedder.embed_text(["a", "b", "c"]))

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_config_forwarded_to_every_chunk(self):
        """Dimensions and request options reach each chunked request."""
        import asyncio

        embedder, calls = self._recording_embedder(
            "gemini-embedding-001",
            dimensions=256,
            options={"task_type": "RETRIEVAL_QUERY"},
        )
        texts = [f"txt-{i}" for i in range(150)]

        asyncio.run(embedder.embed_text(texts))

        assert len(calls) == 2
        for call in calls:
            assert call["config"]["output_dimensionality"] == 256
            assert call["config"]["task_type"] == "RETRIEVAL_QUERY"

    def test_udf_batch_size_defaults_to_request_cap(self):
        """The descriptor's UDF batch size defaults to the per-request cap."""
        from vane.ai.providers.google import GoogleTextEmbedderDescriptor

        desc = GoogleTextEmbedderDescriptor(model_name="gemini-embedding-001")
        assert desc.get_udf_options().batch_size == 100

    def test_udf_batch_size_cap_applies_to_aggregating_model(self):
        """Per-Content encoding keeps gemini-embedding-2 row-preserving at
        full chunk size, so it gets the same default cap as other models."""
        from vane.ai.providers.google import GoogleTextEmbedderDescriptor

        desc = GoogleTextEmbedderDescriptor(model_name="gemini-embedding-2")
        assert desc.get_udf_options().batch_size == 100

    def test_descriptor_rejects_legacy_batch_size_override(self):
        from vane.ai.providers.google import GoogleTextEmbedderDescriptor

        with pytest.raises(TypeError, match="batch_size"):
            GoogleTextEmbedderDescriptor(
                model_name="gemini-embedding-001",
                embed_options={"batch_size": 25},
            )


class TestGoogleConversationStructure:
    """Recording-client structural tests for Google role handling.

    Asserts the exact ``Content`` / ``system_instruction`` structures the
    prompter sends to ``generate_content``.
    """

    def _recording_prompter(self, system_message=None):
        from unittest.mock import AsyncMock, MagicMock, patch

        from vane.ai.providers.google import GooglePrompter

        with patch("google.genai.Client"):
            prompter = GooglePrompter(
                provider_options={"api_key": "test"},
                model="gemini-2.5-pro",
                system_message=system_message,
            )

        calls: list[dict] = []
        response = MagicMock()
        response.text = "ok"
        response.usage_metadata = None

        async def fake_generate_content(**kwargs):
            calls.append(kwargs)
            return response

        client = MagicMock()
        client.aio.models.generate_content = AsyncMock(side_effect=fake_generate_content)
        prompter._client = client
        return prompter, calls

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_system_plus_user_string(self):
        """Descriptor system_message routes through system_instruction."""
        import asyncio

        prompter, calls = self._recording_prompter(system_message="Be helpful.")

        result = asyncio.run(prompter.prompt(("hello",)))

        assert result == "ok"
        assert len(calls) == 1
        assert calls[0]["model"] == "gemini-2.5-pro"
        contents = calls[0]["contents"]
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert [part.text for part in contents[0].parts] == ["hello"]
        assert calls[0]["config"].system_instruction == "Be helpful."

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_plain_string_prompt_regression(self):
        """A plain-string prompt stays a single user turn with no config."""
        import asyncio

        prompter, calls = self._recording_prompter()

        asyncio.run(prompter.prompt(("hello",)))

        contents = calls[0]["contents"]
        assert [content.role for content in contents] == ["user"]
        assert [part.text for part in contents[0].parts] == ["hello"]
        assert calls[0]["config"] is None

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_alternating_turns_preserved_in_order(self):
        """Role-tagged dicts become ordered user/model Content turns."""
        import asyncio

        prompter, calls = self._recording_prompter()

        asyncio.run(
            prompter.prompt(
                (
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "q2"},
                )
            )
        )

        contents = calls[0]["contents"]
        assert [content.role for content in contents] == ["user", "model", "user"]
        assert [content.parts[0].text for content in contents] == ["q1", "a1", "q2"]

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_system_messages_combined_in_declaration_order(self):
        """system dicts merge with the descriptor system_message, in order,
        and never appear as conversation turns."""
        import asyncio

        prompter, calls = self._recording_prompter(system_message="A.")

        asyncio.run(
            prompter.prompt(
                (
                    {"role": "system", "content": "B."},
                    {"role": "user", "content": "hi"},
                )
            )
        )

        assert calls[0]["config"].system_instruction == "A.\n\nB."
        contents = calls[0]["contents"]
        assert [content.role for content in contents] == ["user"]
        assert contents[0].parts[0].text == "hi"

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    @pytest.mark.parametrize("role", ["tool", "function", "assistantt"])
    def test_unsupported_role_raises(self, role):
        """Unknown roles fail fast instead of degrading into user text."""
        import asyncio

        prompter, calls = self._recording_prompter()

        with pytest.raises(ValueError, match=f"Unsupported message role '{role}'"):
            asyncio.run(prompter.prompt(({"role": role, "content": "x"},)))
        assert calls == []

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_mixed_text_image_parts_preserved_in_order(self):
        """Structured part lists convert per-part, in order."""
        import asyncio

        prompter, calls = self._recording_prompter()
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10

        asyncio.run(prompter.prompt(({"role": "user", "content": ["look", png]},)))

        contents = calls[0]["contents"]
        assert len(contents) == 1
        assert contents[0].role == "user"
        parts = contents[0].parts
        assert len(parts) == 2
        assert parts[0].text == "look"
        assert parts[1].inline_data.mime_type == "image/png"
        assert parts[1].inline_data.data == png

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_untagged_parts_group_into_user_turns_between_role_dicts(self):
        """Loose parts flush into user turns around role-tagged messages."""
        import asyncio

        prompter, calls = self._recording_prompter()

        asyncio.run(
            prompter.prompt(
                (
                    "intro",
                    {"role": "assistant", "content": "a"},
                    "next",
                )
            )
        )

        contents = calls[0]["contents"]
        assert [content.role for content in contents] == ["user", "model", "user"]
        assert [content.parts[0].text for content in contents] == ["intro", "a", "next"]

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_dict_text_content_part_converts_without_repr(self):
        """OpenAI-style text part dicts become real text parts."""
        import asyncio

        prompter, calls = self._recording_prompter()

        asyncio.run(prompter.prompt(({"role": "user", "content": [{"type": "text", "text": "hi"}]},)))

        parts = calls[0]["contents"][0].parts
        assert [part.text for part in parts] == ["hi"]

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_unsupported_dict_content_part_raises(self):
        import asyncio

        prompter, calls = self._recording_prompter()

        with pytest.raises(ValueError, match="Unsupported dict content part"):
            asyncio.run(
                prompter.prompt(({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u"}}]},))
            )
        assert calls == []

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_non_string_system_content_raises(self):
        import asyncio

        prompter, calls = self._recording_prompter()

        with pytest.raises(ValueError, match="system messages must be plain text"):
            asyncio.run(prompter.prompt(({"role": "system", "content": ["a", "b"]},)))
        assert calls == []


# ---------------------------------------------------------------------------
# Anthropic Structured Output + Multimodal tests
# ---------------------------------------------------------------------------


class TestAnthropicStructuredOutput:
    """Tests for Anthropic structured output via tool_use."""

    def test_descriptor_has_return_format(self):
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        desc = AnthropicPrompterDescriptor(
            model_name="claude-test-model", return_format=dict, prompt_options={"max_tokens": 64}
        )
        assert desc.return_format is dict

    def test_descriptor_default_no_return_format(self):
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        desc = AnthropicPrompterDescriptor(model_name="claude-test-model", prompt_options={"max_tokens": 64})
        assert desc.return_format is None

    def test_descriptor_pickle_with_return_format(self):
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        desc = AnthropicPrompterDescriptor(
            model_name="claude-test-model", return_format=dict, prompt_options={"max_tokens": 64}
        )
        restored = pickle.loads(pickle.dumps(desc))
        assert restored.return_format is dict

    def test_provider_passes_return_format(self):
        from vane.ai.providers.anthropic import AnthropicProvider

        prov = AnthropicProvider(api_key="test")
        desc = prov.get_prompter(model="claude-test-model", max_tokens=64, return_format=dict)
        assert desc.return_format is dict

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_build_tool_schema_from_dict(self):
        from vane.ai.providers.anthropic import AnthropicPrompter

        p = AnthropicPrompter(
            provider_options={"api_key": "test"},
            model="claude-sonnet-4-20250514",
            return_format={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        tool = p._build_tool_schema()
        assert tool["name"] == "extract_data"
        assert tool["input_schema"]["type"] == "object"

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_build_tool_schema_from_pydantic(self):
        from vane.ai.providers.anthropic import AnthropicPrompter

        mock_model = MagicMock()
        mock_model.model_json_schema.return_value = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        p = AnthropicPrompter(
            provider_options={"api_key": "test"},
            model="claude-sonnet-4-20250514",
            return_format=mock_model,
        )
        tool = p._build_tool_schema()
        assert tool["input_schema"]["type"] == "object"

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_structured_output_extracts_tool_use_block(self):
        """Structured output extracts data from tool_use response block."""
        import asyncio

        from vane.ai.providers.anthropic import AnthropicPrompter

        p = AnthropicPrompter(
            provider_options={"api_key": "test"},
            model="claude-sonnet-4-20250514",
            return_format=dict,
        )

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.input = {"name": "Alice", "age": 30}

        mock_usage = MagicMock()
        mock_usage.input_tokens = 12
        mock_usage.output_tokens = 7

        mock_response = MagicMock()
        mock_response.content = [tool_block]
        mock_response.usage = mock_usage

        async def mock_create(**kwargs):
            assert "tools" in kwargs
            assert kwargs["tool_choice"] == {"type": "tool", "name": "extract_data"}
            return mock_response

        p._client.messages.create = mock_create
        result = asyncio.run(p.prompt(("Extract name and age",)))
        assert result == {"name": "Alice", "age": 30}

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_plain_text_response(self):
        """Without return_format, returns text content."""
        import asyncio

        from vane.ai.providers.anthropic import AnthropicPrompter

        p = AnthropicPrompter(
            provider_options={"api_key": "test"},
            model="claude-sonnet-4-20250514",
        )

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello world"

        mock_usage = MagicMock()
        mock_usage.input_tokens = 8
        mock_usage.output_tokens = 3

        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_response.usage = mock_usage

        async def mock_create(**kwargs):
            assert "tools" not in kwargs
            return mock_response

        p._client.messages.create = mock_create
        result = asyncio.run(p.prompt(("Hi",)))
        assert result == "Hello world"


class TestAnthropicMultimodal:
    """Tests for Anthropic multimodal message processing."""

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_process_str(self):
        from vane.ai.providers.anthropic import AnthropicPrompter

        p = AnthropicPrompter(
            provider_options={"api_key": "test"},
            model="claude-sonnet-4-20250514",
        )
        result = p._process_message("Hello")
        assert result == {"type": "text", "text": "Hello"}

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_process_bytes_png(self):
        from vane.ai.providers.anthropic import AnthropicPrompter

        p = AnthropicPrompter(
            provider_options={"api_key": "test"},
            model="claude-sonnet-4-20250514",
        )
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
        result = p._process_message(png)
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/png"
        assert len(result["source"]["data"]) > 0

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_process_bytes_jpeg(self):
        from vane.ai.providers.anthropic import AnthropicPrompter

        p = AnthropicPrompter(
            provider_options={"api_key": "test"},
            model="claude-sonnet-4-20250514",
        )
        jpeg = b"\xff\xd8\xff" + b"\x00" * 10
        result = p._process_message(jpeg)
        assert result["source"]["media_type"] == "image/jpeg"

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_process_ndarray(self):
        from vane.ai.providers.anthropic import AnthropicPrompter

        p = AnthropicPrompter(
            provider_options={"api_key": "test"},
            model="claude-sonnet-4-20250514",
        )
        arr = np.zeros((2, 2, 3), dtype=np.uint8)
        result = p._process_message(arr)
        assert result["type"] == "image"
        assert result["source"]["media_type"] == "image/png"

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_process_dict_passthrough(self):
        from vane.ai.providers.anthropic import AnthropicPrompter

        p = AnthropicPrompter(
            provider_options={"api_key": "test"},
            model="claude-sonnet-4-20250514",
        )
        part = {"type": "text", "text": "pre-built"}
        result = p._process_message(part)
        assert result is part

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_unsupported_type_raises(self):
        from vane.ai.providers.anthropic import AnthropicPrompter

        p = AnthropicPrompter(
            provider_options={"api_key": "test"},
            model="claude-sonnet-4-20250514",
        )
        with pytest.raises(ValueError, match="Unsupported multimodal"):
            p._process_message(42)

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_multimodal_prompt_assembly(self):
        """Text + image are assembled into content array."""
        import asyncio

        from vane.ai.providers.anthropic import AnthropicPrompter

        p = AnthropicPrompter(
            provider_options={"api_key": "test"},
            model="claude-sonnet-4-20250514",
        )

        captured_messages = []
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I see an image"

        mock_usage = MagicMock()
        mock_usage.input_tokens = 25
        mock_usage.output_tokens = 6

        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_response.usage = mock_usage

        async def mock_create(**kwargs):
            captured_messages.append(kwargs["messages"])
            return mock_response

        p._client.messages.create = mock_create
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
        asyncio.run(p.prompt(("Describe:", png)))

        msgs = captured_messages[0]
        assert msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_dict_with_role_as_complete_message(self):
        """Dict with 'role' key becomes a separate message."""
        import asyncio

        from vane.ai.providers.anthropic import AnthropicPrompter

        p = AnthropicPrompter(
            provider_options={"api_key": "test"},
            model="claude-sonnet-4-20250514",
        )

        captured_messages = []
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "ok"

        mock_usage = MagicMock()
        mock_usage.input_tokens = 15
        mock_usage.output_tokens = 2

        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_response.usage = mock_usage

        async def mock_create(**kwargs):
            captured_messages.append(kwargs["messages"])
            return mock_response

        p._client.messages.create = mock_create
        asyncio.run(
            p.prompt(
                (
                    {"role": "assistant", "content": "Previous"},
                    "Follow up",
                )
            )
        )

        msgs = captured_messages[0]
        assert msgs[0] == {"role": "assistant", "content": "Previous"}
        assert msgs[1]["role"] == "user"


class TestAnthropicGuessMediaType:
    """Tests for the Anthropic _guess_media_type helper."""

    def test_png(self):
        from vane.ai.providers.anthropic import _guess_media_type

        assert _guess_media_type(b"\x89PNG\r\n\x1a\n") == "image/png"

    def test_jpeg(self):
        from vane.ai.providers.anthropic import _guess_media_type

        assert _guess_media_type(b"\xff\xd8") == "image/jpeg"

    def test_gif(self):
        from vane.ai.providers.anthropic import _guess_media_type

        assert _guess_media_type(b"GIF89a") == "image/gif"

    def test_webp(self):
        from vane.ai.providers.anthropic import _guess_media_type

        assert _guess_media_type(b"RIFF\x00\x00\x00\x00WEBP") == "image/webp"

    def test_unknown(self):
        from vane.ai.providers.anthropic import _guess_media_type

        assert _guess_media_type(b"\x00\x01") == "application/octet-stream"


# ---------------------------------------------------------------------------
# Google Structured Output + Multimodal tests
# ---------------------------------------------------------------------------


class TestGoogleStructuredOutput:
    """Tests for Google Gemini structured output via response_schema."""

    def test_descriptor_has_return_format(self):
        from vane.ai.providers.google import GooglePrompterDescriptor

        desc = GooglePrompterDescriptor(model_name="gemini-3.6-flash", return_format=dict)
        assert desc.return_format is dict

    def test_descriptor_default_no_return_format(self):
        from vane.ai.providers.google import GooglePrompterDescriptor

        desc = GooglePrompterDescriptor(model_name="gemini-3.6-flash")
        assert desc.return_format is None

    def test_descriptor_pickle_with_return_format(self):
        from vane.ai.providers.google import GooglePrompterDescriptor

        desc = GooglePrompterDescriptor(model_name="gemini-3.6-flash", return_format=dict)
        restored = pickle.loads(pickle.dumps(desc))
        assert restored.return_format is dict

    def test_provider_passes_return_format(self):
        from vane.ai.providers.google import GoogleProvider

        prov = GoogleProvider(api_key="test")
        desc = prov.get_prompter(model="gemini-3.6-flash", return_format=dict)
        assert desc.return_format is dict


class TestGoogleMultimodal:
    """Tests for Google Gemini multimodal message processing."""

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_process_str(self):
        from unittest.mock import patch

        from vane.ai.providers.google import GooglePrompter

        with patch("google.genai.Client"):
            p = GooglePrompter(
                provider_options={"api_key": "test"},
                model="gemini-2.0-flash",
            )
        result = p._process_message("Hello")
        # Should return a Part object
        assert hasattr(result, "text") or result is not None

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_process_bytes(self):
        from unittest.mock import patch

        from vane.ai.providers.google import GooglePrompter

        with patch("google.genai.Client"):
            p = GooglePrompter(
                provider_options={"api_key": "test"},
                model="gemini-2.0-flash",
            )
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
        result = p._process_message(png)
        assert result is not None

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_process_ndarray(self):
        from unittest.mock import patch

        from vane.ai.providers.google import GooglePrompter

        with patch("google.genai.Client"):
            p = GooglePrompter(
                provider_options={"api_key": "test"},
                model="gemini-2.0-flash",
            )
        arr = np.zeros((2, 2, 3), dtype=np.uint8)
        result = p._process_ndarray(arr)
        assert result is not None

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_unsupported_type_raises(self):
        from unittest.mock import patch

        from vane.ai.providers.google import GooglePrompter

        with patch("google.genai.Client"):
            p = GooglePrompter(
                provider_options={"api_key": "test"},
                model="gemini-2.0-flash",
            )
        with pytest.raises(ValueError, match="Unsupported multimodal"):
            p._process_message(42)


class TestGoogleGuessMediaType:
    """Tests for the Google _guess_media_type helper."""

    def test_png(self):
        from vane.ai.providers.google import _guess_media_type

        assert _guess_media_type(b"\x89PNG\r\n\x1a\n") == "image/png"

    def test_jpeg(self):
        from vane.ai.providers.google import _guess_media_type

        assert _guess_media_type(b"\xff\xd8") == "image/jpeg"

    def test_unknown(self):
        from vane.ai.providers.google import _guess_media_type

        assert _guess_media_type(b"\x00\x01") == "application/octet-stream"


# ---------------------------------------------------------------------------
# Long-text chunking tests
# ---------------------------------------------------------------------------


class TestChunking:
    """Tests for chunk_text utility and _EmbedTextBatch chunking."""

    def test_chunk_text_short(self):
        """Short text returns single chunk."""
        from vane.ai.functions import chunk_text

        result = chunk_text("hello world", max_chars=100)
        assert result == ["hello world"]

    def test_chunk_text_exact_boundary(self):
        """Text at exactly max_chars returns single chunk."""
        from vane.ai.functions import chunk_text

        text = "a" * 100
        result = chunk_text(text, max_chars=100)
        assert result == [text]

    def test_chunk_text_splits(self):
        """Long text is split into overlapping chunks."""
        from vane.ai.functions import chunk_text

        text = "a" * 250
        result = chunk_text(text, max_chars=100, overlap_chars=20)
        assert len(result) == 3
        # First chunk: 0-100, second: 80-180, third: 160-250
        assert all(len(c) <= 100 for c in result)
        assert len(result[-1]) == 90  # 250-160

    def test_chunk_text_overlap_content(self):
        """Overlapping regions share the same content."""
        from vane.ai.functions import chunk_text

        text = "".join(str(i % 10) for i in range(300))
        result = chunk_text(text, max_chars=100, overlap_chars=30)
        # Check overlap between first two chunks
        assert result[0][-30:] == result[1][:30]

    def test_chunk_text_no_overlap(self):
        """Zero overlap produces non-overlapping chunks."""
        from vane.ai.functions import chunk_text

        text = "a" * 200
        result = chunk_text(text, max_chars=100, overlap_chars=0)
        assert len(result) == 2
        assert result[0] == "a" * 100
        assert result[1] == "a" * 100

    def test_weighted_average_embeddings(self):
        """Weighted average normalizes embeddings correctly."""
        from vane.ai.functions import _weighted_average_embeddings

        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        result = _weighted_average_embeddings([e1, e2], [1.0, 1.0])
        # Equal weights → 45 degree angle, normalized
        expected = np.array([1, 1, 0], dtype=np.float32) / np.sqrt(2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_weighted_average_unequal_weights(self):
        """Longer chunk gets more weight."""
        from vane.ai.functions import _weighted_average_embeddings

        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        # e1 has 3x the weight
        result = _weighted_average_embeddings([e1, e2], [3.0, 1.0])
        assert result[0] > result[1]  # x component dominant

    def test_embed_batch_with_chunking(self):
        """_EmbedTextBatch chunks long texts and averages embeddings."""
        from vane.ai.functions import _EmbedTextBatch

        dim = 4

        class FakeEmbedder:
            def embed_text(self, texts):
                # Return a distinct unit vector per chunk index
                return [np.random.RandomState(hash(t) % 2**31).randn(dim).astype(np.float32) for t in texts]

        @dataclass
        class FakeDescriptor:
            def get_provider(self):
                return "test"

            def get_model(self):
                return "test"

            def get_options(self):
                return {}

            def get_dimensions(self):
                return EmbeddingDimensions(size=dim)

            def instantiate(self):
                return FakeEmbedder()

        desc = FakeDescriptor()
        batch = _EmbedTextBatch(desc, "text", "embedding", dim, max_chunk_chars=50, chunk_overlap_chars=10)

        # Short text (no chunking) + long text (will be chunked)
        table = pa.table({"text": ["short", "a" * 200]})
        result = _drive(batch, table)

        assert result.num_rows == 2
        emb0 = result.column("embedding")[0].as_py()
        emb1 = result.column("embedding")[1].as_py()
        assert len(emb0) == dim
        assert len(emb1) == dim

    def test_embed_batch_no_chunking_by_default(self):
        """_EmbedTextBatch without max_chunk_chars doesn't chunk."""
        from vane.ai.functions import _EmbedTextBatch

        call_count = 0
        dim = 4

        class CountingEmbedder:
            def embed_text(self, texts):
                nonlocal call_count
                call_count += 1
                return [np.ones(dim, dtype=np.float32) for _ in texts]

        @dataclass
        class FakeDescriptor:
            def get_provider(self):
                return "test"

            def get_model(self):
                return "test"

            def get_options(self):
                return {}

            def get_dimensions(self):
                return EmbeddingDimensions(size=dim)

            def instantiate(self):
                return CountingEmbedder()

        desc = FakeDescriptor()
        batch = _EmbedTextBatch(desc, "text", "embedding", dim)  # no chunking

        table = pa.table({"text": ["a" * 5000]})
        result = _drive(batch, table)

        assert result.num_rows == 1
        assert call_count == 1  # single call, no chunking

    def test_embed_batch_chunking_params_stored(self):
        """_EmbedTextBatch stores chunking params correctly."""
        from vane.ai.functions import _EmbedTextBatch

        dim = 4

        class SimpleEmbedder:
            def embed_text(self, t):
                return [np.zeros(dim) for _ in t]

        @dataclass
        class FakeDescriptor:
            def get_provider(self):
                return "test"

            def get_model(self):
                return "test"

            def get_options(self):
                return {}

            def get_dimensions(self):
                return EmbeddingDimensions(size=dim)

            def instantiate(self):
                return SimpleEmbedder()

        desc = FakeDescriptor()
        batch = _EmbedTextBatch(desc, "text", "embedding", dim, max_chunk_chars=500, chunk_overlap_chars=50)
        assert batch._max_chunk_chars == 500
        assert batch._chunk_overlap_chars == 50


# ---------------------------------------------------------------------------
# Structured Output + Responses API tests
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    """Tests for OpenAI Structured Output and Responses API support."""

    def test_openai_prompter_descriptor_has_return_format(self):
        """Descriptor stores return_format field."""
        from vane.ai.providers.openai import OpenAIPrompterDescriptor

        desc = OpenAIPrompterDescriptor(return_format=dict)
        assert desc.return_format is dict

    def test_openai_prompter_descriptor_default_no_return_format(self):
        """Default return_format is None."""
        from vane.ai.providers.openai import OpenAIPrompterDescriptor

        desc = OpenAIPrompterDescriptor()
        assert desc.return_format is None

    def test_openai_prompter_descriptor_use_chat_completions_default(self):
        """Default use_chat_completions is True (backward compatible)."""
        from vane.ai.providers.openai import OpenAIPrompterDescriptor

        desc = OpenAIPrompterDescriptor()
        assert desc.use_chat_completions is True

    def test_openai_prompter_descriptor_use_chat_completions_false(self):
        """Can set use_chat_completions to False for Responses API."""
        from vane.ai.providers.openai import OpenAIPrompterDescriptor

        desc = OpenAIPrompterDescriptor(use_chat_completions=False)
        assert desc.use_chat_completions is False

    def test_openai_prompter_descriptor_pickle_with_return_format(self):
        """Descriptor with return_format survives pickle roundtrip."""
        from vane.ai.providers.openai import OpenAIPrompterDescriptor

        desc = OpenAIPrompterDescriptor(
            return_format=dict,  # use dict as a simple stand-in
            use_chat_completions=False,
        )
        restored = pickle.loads(pickle.dumps(desc))
        assert restored.return_format is dict
        assert restored.use_chat_completions is False

    def test_openai_provider_get_prompter_passes_return_format(self):
        """Provider.get_prompter forwards return_format to descriptor."""
        from vane.ai.providers.openai import OpenAIProvider

        prov = OpenAIProvider(api_key="test-key")
        desc = prov.get_prompter(return_format=dict, use_chat_completions=False)
        assert desc.return_format is dict
        assert desc.use_chat_completions is False

    def test_openai_provider_get_prompter_default_chat_completions(self):
        """Provider.get_prompter defaults to use_chat_completions=True."""
        from vane.ai.providers.openai import OpenAIProvider

        prov = OpenAIProvider(api_key="test-key")
        desc = prov.get_prompter()
        assert desc.use_chat_completions is True
        assert desc.return_format is None

    def test_prompt_batch_stores_return_format(self):
        """_PromptBatch stores return_format for serialization."""
        from vane.ai.functions import _PromptBatch

        desc = MagicMock()
        wrapper = _PromptBatch(desc, "text", "response", return_format=dict)
        assert wrapper._return_format is dict

    def test_prompt_batch_serialize_result_string(self):
        """_serialize_result returns strings as-is."""
        from vane.ai.functions import _PromptBatch

        wrapper = _PromptBatch(MagicMock(), "t", "r", return_format=dict)
        assert wrapper._serialize_result("hello") == "hello"

    def test_prompt_batch_serialize_result_none(self):
        """_serialize_result returns None for None."""
        from vane.ai.functions import _PromptBatch

        wrapper = _PromptBatch(MagicMock(), "t", "r", return_format=dict)
        assert wrapper._serialize_result(None) is None

    def test_prompt_batch_serialize_result_pydantic_model(self):
        """_serialize_result calls model_dump_json() on Pydantic models."""
        from vane.ai.functions import _PromptBatch

        mock_model = MagicMock()
        mock_model.model_dump_json.return_value = '{"name":"Alice","age":30}'
        wrapper = _PromptBatch(MagicMock(), "t", "r", return_format=dict)
        result = wrapper._serialize_result(mock_model)
        assert result == '{"name":"Alice","age":30}'
        mock_model.model_dump_json.assert_called_once()

    def test_prompt_batch_serialize_result_dict(self):
        """_serialize_result JSON-encodes dicts."""
        import json

        from vane.ai.functions import _PromptBatch

        wrapper = _PromptBatch(MagicMock(), "t", "r", return_format=dict)
        result = wrapper._serialize_result({"name": "Alice", "age": 30})
        parsed = json.loads(result)
        assert parsed == {"name": "Alice", "age": 30}

    def test_prompt_function_accepts_return_format(self):
        """prompt() accepts return_format and use_chat_completions params."""
        from vane.ai.functions import prompt as prompt_fn
        from vane.ai.providers.openai import OpenAIProvider

        captured = {}
        original_get_prompter = OpenAIProvider.get_prompter

        def patched_get_prompter(self, **kwargs):
            captured.update(kwargs)
            return original_get_prompter(self, **kwargs)

        conn = duckdb.connect()
        rel = conn.sql("SELECT 'Hello' AS text")
        with pytest.MonkeyPatch.context() as m:
            m.setattr(OpenAIProvider, "get_prompter", patched_get_prompter)
            # Just verify it doesn't error on param passing.
            # Will fail on actual API call, but we only test param propagation.
            try:
                prompt_fn(
                    rel,
                    "text",
                    provider=OpenAIProvider(api_key="test"),
                    return_format=dict,
                    use_chat_completions=False,
                )
            except Exception:
                pass  # Expected — no real API

        assert captured.get("return_format") is dict
        assert captured.get("use_chat_completions") is False


class TestStructuredOutputExecution:
    """Tests for structured output execution with mock OpenAI client."""

    def _make_prompter(self, return_format=None, use_chat_completions=True, **options):
        """Create an OpenAIPrompter with a mock client."""
        from vane.ai.providers.openai import OpenAIPrompter

        return OpenAIPrompter(
            provider_options={"api_key": "test-key"},
            model="gpt-4o-mini",
            return_format=return_format,
            use_chat_completions=use_chat_completions,
            **options,
        )

    def test_chat_completions_plain_text(self):
        """Chat Completions without return_format → plain text."""
        import asyncio

        prompter = self._make_prompter(return_format=None, use_chat_completions=True)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello world"

        async def mock_create(**_kwargs):
            return mock_response

        prompter._client.chat.completions.create = mock_create
        result = asyncio.run(prompter.prompt(("Hi",)))
        assert result == "Hello world"

    def test_chat_completions_omits_responses_only_token_option(self):
        """Chat Completions receives max_tokens, not Responses-only max_output_tokens."""
        import asyncio

        prompter = self._make_prompter(
            return_format=None,
            use_chat_completions=True,
            max_tokens=7,
            max_output_tokens=11,
        )
        captured = {}
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"

        async def mock_create(**kwargs):
            captured.update(kwargs)
            return mock_response

        prompter._client.chat.completions.create = mock_create
        asyncio.run(prompter.prompt(("Hi",)))

        assert captured["max_tokens"] == 7
        assert "max_output_tokens" not in captured

    def test_chat_completions_structured_output(self):
        """Chat Completions with return_format → calls parse(), returns .parsed."""
        import asyncio

        mock_parsed = MagicMock()
        mock_parsed.name = "Alice"

        prompter = self._make_prompter(return_format=dict, use_chat_completions=True)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = mock_parsed

        async def mock_parse(**kwargs):
            assert "response_format" in kwargs
            assert kwargs["response_format"] is dict
            return mock_response

        prompter._client.chat.completions.parse = mock_parse
        result = asyncio.run(prompter.prompt(("Describe Alice",)))
        assert result is mock_parsed

    def test_responses_api_plain_text(self):
        """Responses API without return_format → responses.create(), output_text."""
        import asyncio

        prompter = self._make_prompter(return_format=None, use_chat_completions=False)

        mock_response = MagicMock()
        mock_response.output_text = "Response from Responses API"

        async def mock_create(**kwargs):
            assert "input" in kwargs
            return mock_response

        prompter._client.responses.create = mock_create
        result = asyncio.run(prompter.prompt(("Hi",)))
        assert result == "Response from Responses API"

    def test_responses_api_omits_chat_only_token_option(self):
        """Responses API receives max_output_tokens, not Chat-only max_tokens."""
        import asyncio

        prompter = self._make_prompter(
            return_format=None,
            use_chat_completions=False,
            max_tokens=7,
            max_output_tokens=11,
        )
        captured = {}
        mock_response = MagicMock()
        mock_response.output_text = "ok"

        async def mock_create(**kwargs):
            captured.update(kwargs)
            return mock_response

        prompter._client.responses.create = mock_create
        asyncio.run(prompter.prompt(("Hi",)))

        assert captured["max_output_tokens"] == 11
        assert "max_tokens" not in captured

    def test_responses_api_structured_output(self):
        """Responses API with return_format → responses.parse(), output_parsed."""
        import asyncio

        mock_parsed = {"name": "Bob", "age": 25}
        prompter = self._make_prompter(return_format=dict, use_chat_completions=False)

        mock_response = MagicMock()
        mock_response.output_parsed = mock_parsed

        async def mock_parse(**kwargs):
            assert "text_format" in kwargs
            assert kwargs["text_format"] is dict
            return mock_response

        prompter._client.responses.parse = mock_parse
        result = asyncio.run(prompter.prompt(("Describe Bob",)))
        assert result == {"name": "Bob", "age": 25}

    def test_system_message_included_in_chat_completions(self):
        """System message is prepended in Chat Completions API."""
        import asyncio

        prompter = self._make_prompter(return_format=None, use_chat_completions=True)
        prompter._system_message = "You are helpful."

        captured_messages = []
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"

        async def mock_create(**kwargs):
            captured_messages.extend(kwargs["messages"])
            return mock_response

        prompter._client.chat.completions.create = mock_create
        asyncio.run(prompter.prompt(("Hi",)))

        assert captured_messages[0] == {"role": "system", "content": "You are helpful."}
        assert captured_messages[1] == {"role": "user", "content": "Hi"}

    def test_system_message_included_in_responses_api(self):
        """System message is included in Responses API input."""
        import asyncio

        prompter = self._make_prompter(return_format=None, use_chat_completions=False)
        prompter._system_message = "You are helpful."

        captured_input = []
        mock_response = MagicMock()
        mock_response.output_text = "ok"

        async def mock_create(**kwargs):
            captured_input.extend(kwargs["input"])
            return mock_response

        prompter._client.responses.create = mock_create
        asyncio.run(prompter.prompt(("Hi",)))

        assert captured_input[0] == {"role": "system", "content": "You are helpful."}
        assert captured_input[1] == {"role": "user", "content": "Hi"}

    def test_dict_messages_pass_through(self):
        """Dict messages in tuple are passed through as-is."""
        import asyncio

        prompter = self._make_prompter(return_format=None, use_chat_completions=True)
        captured_messages = []

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"

        async def mock_create(**kwargs):
            captured_messages.extend(kwargs["messages"])
            return mock_response

        prompter._client.chat.completions.create = mock_create
        asyncio.run(prompter.prompt(({"role": "assistant", "content": "I see"},)))
        assert captured_messages[0] == {"role": "assistant", "content": "I see"}

    def test_prompt_batch_with_structured_output_serializes(self):
        """_PromptBatch serializes structured output to JSON strings."""
        from vane.ai.functions import _PromptBatch

        mock_model = MagicMock()
        mock_model.model_dump_json.return_value = '{"answer":"42"}'

        mock_prompter = MagicMock(spec=[])  # spec=[] blocks auto-attributes
        mock_prompter.prompt_batch = None  # explicitly not available
        delattr(mock_prompter, "prompt_batch")

        async def mock_prompt(_msgs):
            return mock_model

        mock_prompter.prompt = mock_prompt
        mock_descriptor = MagicMock()
        mock_descriptor.instantiate.return_value = mock_prompter

        batch = _PromptBatch(
            mock_descriptor,
            "text",
            "response",
            max_api_concurrency=4,
            return_format=dict,
        )
        table = pa.table({"text": ["Hello", "World"]})
        result = _drive(batch, table)

        assert result.column("response").to_pylist() == [
            '{"answer":"42"}',
            '{"answer":"42"}',
        ]

    def test_prompt_batch_without_return_format_returns_strings(self):
        """_PromptBatch without return_format returns plain strings."""
        from vane.ai.functions import _PromptBatch

        class SimplePrompter:
            async def prompt(self, msgs):
                return f"reply to {msgs[0]}"

        mock_descriptor = MagicMock()
        mock_descriptor.instantiate.return_value = SimplePrompter()

        batch = _PromptBatch(mock_descriptor, "text", "response")
        table = pa.table({"text": ["Hello"]})
        result = _drive(batch, table)

        assert result.column("response").to_pylist() == ["reply to Hello"]

    def test_prompt_batch_structured_output_with_prompt_batch_method(self):
        """prompt_batch method results are also serialized with return_format."""
        from vane.ai.functions import _PromptBatch

        class BatchPrompter:
            def prompt_batch(self, _texts):
                return [
                    MagicMock(model_dump_json=MagicMock(return_value='{"a":1}')),
                    MagicMock(model_dump_json=MagicMock(return_value='{"a":2}')),
                ]

        mock_descriptor = MagicMock()
        mock_descriptor.instantiate.return_value = BatchPrompter()

        batch = _PromptBatch(
            mock_descriptor,
            "text",
            "out",
            return_format=dict,
        )
        table = pa.table({"text": ["x", "y"]})
        result = _drive(batch, table)
        assert result.column("out").to_pylist() == ['{"a":1}', '{"a":2}']


# ---------------------------------------------------------------------------
# Multimodal input tests
# ---------------------------------------------------------------------------


class TestMultimodalMessageProcessing:
    """Tests for OpenAIPrompter multimodal message dispatch."""

    def _make_prompter(self, use_chat_completions=True):
        from vane.ai.providers.openai import OpenAIPrompter

        return OpenAIPrompter(
            provider_options={"api_key": "test-key"},
            model="gpt-4o",
            use_chat_completions=use_chat_completions,
        )

    def test_process_str_chat_completions(self):
        p = self._make_prompter(use_chat_completions=True)
        result = p._process_str("Hello")
        assert result == {"type": "text", "text": "Hello"}

    def test_process_str_responses_api(self):
        p = self._make_prompter(use_chat_completions=False)
        result = p._process_str("Hello")
        assert result == {"type": "input_text", "text": "Hello"}

    def test_process_bytes_png_chat_completions(self):
        p = self._make_prompter(use_chat_completions=True)
        # Minimal PNG header
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
        result = p._process_bytes(png_bytes)
        assert result["type"] == "image_url"
        assert "image_url" in result
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_process_bytes_png_responses_api(self):
        p = self._make_prompter(use_chat_completions=False)
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
        result = p._process_bytes(png_bytes)
        assert result["type"] == "input_image"
        assert result["image_url"].startswith("data:image/png;base64,")

    def test_process_bytes_jpeg(self):
        p = self._make_prompter(use_chat_completions=True)
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 10
        result = p._process_bytes(jpeg_bytes)
        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_process_bytes_gif(self):
        p = self._make_prompter()
        gif_bytes = b"GIF89a" + b"\x00" * 10
        result = p._process_bytes(gif_bytes)
        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/gif;base64,")

    def test_process_bytes_webp(self):
        p = self._make_prompter()
        webp_bytes = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 10
        result = p._process_bytes(webp_bytes)
        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/webp;base64,")

    def test_process_bytes_pdf_chat_completions(self):
        p = self._make_prompter(use_chat_completions=True)
        pdf_bytes = b"%PDF-1.4" + b"\x00" * 10
        result = p._process_bytes(pdf_bytes)
        assert result["type"] == "file"
        assert result["file"]["file_data"].startswith("data:application/pdf;base64,")

    def test_process_bytes_pdf_responses_api(self):
        p = self._make_prompter(use_chat_completions=False)
        pdf_bytes = b"%PDF-1.4" + b"\x00" * 10
        result = p._process_bytes(pdf_bytes)
        assert result["type"] == "input_file"
        assert result["file_data"].startswith("data:application/pdf;base64,")

    def test_process_bytes_unknown_becomes_file(self):
        p = self._make_prompter()
        result = p._process_bytes(b"\x00\x01\x02\x03")
        assert result["type"] == "file"
        assert result["file"]["file_data"].startswith("data:application/octet-stream;base64,")

    def test_process_ndarray_creates_image(self):
        p = self._make_prompter(use_chat_completions=True)
        arr = np.zeros((2, 2, 3), dtype=np.uint8)
        result = p._process_ndarray(arr)
        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_process_ndarray_responses_api(self):
        p = self._make_prompter(use_chat_completions=False)
        arr = np.zeros((2, 2, 3), dtype=np.uint8)
        result = p._process_ndarray(arr)
        assert result["type"] == "input_image"
        assert result["image_url"].startswith("data:image/png;base64,")

    def test_process_message_dispatches_str(self):
        p = self._make_prompter()
        result = p._process_message("Hello")
        assert result == {"type": "text", "text": "Hello"}

    def test_process_message_dispatches_bytes(self):
        p = self._make_prompter()
        result = p._process_message(b"\xff\xd8\xff" + b"\x00" * 10)
        assert result["type"] == "image_url"

    def test_process_message_dispatches_ndarray(self):
        p = self._make_prompter()
        arr = np.zeros((2, 2, 3), dtype=np.uint8)
        result = p._process_message(arr)
        assert result["type"] == "image_url"

    def test_process_message_dispatches_dict_content_part(self):
        """A dict without 'role' is treated as content part passthrough."""
        p = self._make_prompter()
        part = {"type": "text", "text": "pre-built"}
        result = p._process_message(part)
        assert result is part

    def test_process_message_unsupported_type_raises(self):
        p = self._make_prompter()
        with pytest.raises(ValueError, match="Unsupported multimodal"):
            p._process_message(42)


class TestMultimodalPromptAssembly:
    """Tests for multimodal message assembly in prompt()."""

    def _make_prompter(self, use_chat_completions=True):
        from vane.ai.providers.openai import OpenAIPrompter

        return OpenAIPrompter(
            provider_options={"api_key": "test-key"},
            model="gpt-4o",
            use_chat_completions=use_chat_completions,
        )

    def test_single_text_stays_plain_string(self):
        """A single str message uses plain string content (backward compat)."""
        import asyncio

        p = self._make_prompter()
        captured = []

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"

        async def mock_create(**kwargs):
            captured.append(kwargs["messages"])
            return mock_resp

        p._client.chat.completions.create = mock_create
        asyncio.run(p.prompt(("Hello",)))

        msgs = captured[0]
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"  # plain string, not array

    def test_text_plus_image_becomes_content_array(self):
        """Text + bytes creates a multimodal content array."""
        import asyncio

        p = self._make_prompter()
        captured = []

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "I see an image"

        async def mock_create(**kwargs):
            captured.append(kwargs["messages"])
            return mock_resp

        p._client.chat.completions.create = mock_create
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
        asyncio.run(p.prompt(("Describe this image:", png)))

        msgs = captured[0]
        assert msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"

    def test_dict_with_role_preserved_as_message(self):
        """Dict with 'role' key is treated as a complete message."""
        import asyncio

        p = self._make_prompter()
        captured = []

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"

        async def mock_create(**kwargs):
            captured.append(kwargs["messages"])
            return mock_resp

        p._client.chat.completions.create = mock_create
        asyncio.run(
            p.prompt(
                (
                    {"role": "assistant", "content": "Previous response"},
                    "Follow up",
                )
            )
        )

        msgs = captured[0]
        assert msgs[0] == {"role": "assistant", "content": "Previous response"}
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Follow up"

    def test_system_message_plus_multimodal(self):
        """System message + text + image = 3-element messages array."""
        import asyncio

        p = self._make_prompter()
        p._system_message = "You are a vision expert."
        captured = []

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"

        async def mock_create(**kwargs):
            captured.append(kwargs["messages"])
            return mock_resp

        p._client.chat.completions.create = mock_create
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
        asyncio.run(p.prompt(("What is this?", png)))

        msgs = captured[0]
        assert len(msgs) == 2  # system + user
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert isinstance(msgs[1]["content"], list)


class TestMultimodalPromptBatch:
    """Tests for _PromptBatch with image_columns."""

    def test_prompt_batch_with_image_columns(self):
        """image_columns are packed into message tuples alongside text."""
        from vane.ai.functions import _PromptBatch

        captured_messages = []

        class MultimodalPrompter:
            async def prompt(self, msgs):
                captured_messages.append(msgs)
                return f"saw {len(msgs)} parts"

        mock_descriptor = MagicMock()
        mock_descriptor.instantiate.return_value = MultimodalPrompter()

        batch = _PromptBatch(
            mock_descriptor,
            "text",
            "response",
            image_columns=["image"],
        )
        table = pa.table(
            {
                "text": ["Describe this"],
                "image": [b"\x89PNG\r\n\x1a\n\x00\x00"],
            }
        )
        result = _drive(batch, table)

        assert len(captured_messages) == 1
        assert len(captured_messages[0]) == 2  # text + image bytes
        assert captured_messages[0][0] == "Describe this"
        assert captured_messages[0][1] == b"\x89PNG\r\n\x1a\n\x00\x00"
        assert result.column("response").to_pylist() == ["saw 2 parts"]

    def test_prompt_batch_skips_none_images(self):
        """None image values are excluded from the message tuple."""
        from vane.ai.functions import _PromptBatch

        captured_messages = []

        class SimplePrompter:
            async def prompt(self, msgs):
                captured_messages.append(msgs)
                return "ok"

        mock_descriptor = MagicMock()
        mock_descriptor.instantiate.return_value = SimplePrompter()

        batch = _PromptBatch(
            mock_descriptor,
            "text",
            "response",
            image_columns=["image"],
        )
        table = pa.table(
            {
                "text": ["No image here"],
                "image": pa.array([None], type=pa.binary()),
            }
        )
        _drive(batch, table)

        assert len(captured_messages[0]) == 1  # just text
        assert captured_messages[0][0] == "No image here"

    def test_prompt_batch_multiple_image_columns(self):
        """Multiple image columns produce multi-part messages."""
        from vane.ai.functions import _PromptBatch

        captured_messages = []

        class SimplePrompter:
            async def prompt(self, msgs):
                captured_messages.append(msgs)
                return "ok"

        mock_descriptor = MagicMock()
        mock_descriptor.instantiate.return_value = SimplePrompter()

        batch = _PromptBatch(
            mock_descriptor,
            "text",
            "response",
            image_columns=["img1", "img2"],
        )
        table = pa.table(
            {
                "text": ["Compare these"],
                "img1": [b"\x89PNG\r\n\x1a\n"],
                "img2": [b"\xff\xd8\xff"],
            }
        )
        _drive(batch, table)

        assert len(captured_messages[0]) == 3  # text + 2 images

    def test_prompt_batch_partitions_image_lists_and_preserves_row_order(self):
        """Only rows with actual images use the multimodal prompt path."""
        from vane.ai.functions import _PromptBatch

        captured_messages = []
        captured_batches = []

        class DualPathPrompter:
            def prompt_batch(self, texts):
                captured_batches.append(texts)
                return [f"batch:{text}" for text in texts]

            async def prompt(self, msgs):
                captured_messages.append(msgs)
                return f"multimodal:{msgs[0]}"

        mock_descriptor = MagicMock()
        mock_descriptor.instantiate.return_value = DualPathPrompter()

        batch = _PromptBatch(
            mock_descriptor,
            "text",
            "response",
            image_columns=["images"],
        )
        table = pa.table(
            {
                "text": ["Compare these", "No images", "Empty images"],
                "images": pa.array(
                    [
                        [b"\x89PNG\r\n\x1a\n", None, b"\xff\xd8\xff"],
                        None,
                        [],
                    ],
                    type=pa.list_(pa.binary()),
                ),
            }
        )
        result = _drive(batch, table)

        assert captured_messages == [
            ("Compare these", b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff"),
        ]
        assert captured_batches == [["No images", "Empty images"]]
        assert result.column("response").to_pylist() == [
            "multimodal:Compare these",
            "batch:No images",
            "batch:Empty images",
        ]

    def test_prompt_batch_uses_text_batch_api_when_all_image_parts_are_empty(self):
        """NULL, empty lists, and NULL list elements remain safely batchable."""
        from vane.ai.functions import _PromptBatch

        captured_batches = []

        class VLLMLikePrompter:
            def prompt_batch(self, texts):
                captured_batches.append(texts)
                return [f"response:{text}" for text in texts]

            async def prompt(self, _msgs):
                raise AssertionError("text-only rows must not use the shared single-row executor")

        mock_descriptor = MagicMock()
        mock_descriptor.instantiate.return_value = VLLMLikePrompter()

        batch = _PromptBatch(
            mock_descriptor,
            "text",
            "response",
            image_columns=["images"],
        )
        table = pa.table(
            {
                "text": ["alpha", "beta", "gamma"],
                "images": pa.array(
                    [
                        None,
                        [],
                        [None],
                    ],
                    type=pa.list_(pa.binary()),
                ),
            }
        )
        result = _drive(batch, table)

        assert captured_batches == [["alpha", "beta", "gamma"]]
        assert result.column("response").to_pylist() == [
            "response:alpha",
            "response:beta",
            "response:gamma",
        ]

    def test_prompt_batch_propagates_null_prompts_when_requested(self):
        """NULL prompts stay local while other rows retain their execution path."""
        from vane.ai.functions import _PromptBatch

        captured_messages = []
        captured_batches = []

        class DualPathPrompter:
            def prompt_batch(self, texts):
                captured_batches.append(texts)
                return [f"batch:{text}" for text in texts]

            async def prompt(self, msgs):
                captured_messages.append(msgs)
                return f"multimodal:{msgs[0]}"

        mock_descriptor = MagicMock()
        mock_descriptor.instantiate.return_value = DualPathPrompter()

        batch = _PromptBatch(
            mock_descriptor,
            "text",
            "response",
            image_columns=["image"],
            propagate_null_prompts=True,
        )
        table = pa.table(
            {
                "text": pa.array([None, "Text only", "With image"], type=pa.string()),
                "image": pa.array([b"ignored", None, b"\x89PNG"], type=pa.binary()),
            }
        )
        result = _drive(batch, table)

        assert captured_batches == [["Text only"]]
        assert captured_messages == [("With image", b"\x89PNG")]
        assert result.column("response").to_pylist() == [
            None,
            "batch:Text only",
            "multimodal:With image",
        ]

    def test_prompt_batch_does_not_instantiate_for_only_null_prompts(self):
        """An all-NULL propagated batch does not initialize a provider."""
        from vane.ai.functions import _PromptBatch

        mock_descriptor = MagicMock()
        batch = _PromptBatch(
            mock_descriptor,
            "text",
            "response",
            image_columns=["image"],
            propagate_null_prompts=True,
        )
        table = pa.table(
            {
                "text": pa.array([None], type=pa.string()),
                "image": pa.array([b"\x89PNG"], type=pa.binary()),
            }
        )

        result = batch(table)

        mock_descriptor.instantiate.assert_not_called()
        assert result.column("response").to_pylist() == [None]

    def test_prompt_batch_treats_zero_length_bytes_as_empty_images(self):
        """Zero-length BLOBs and BLOB[] elements remain safely batchable."""
        from vane.ai.functions import _PromptBatch

        captured_batches = []

        class VLLMLikePrompter:
            def prompt_batch(self, texts):
                captured_batches.append(texts)
                return [f"response:{text}" for text in texts]

            async def prompt(self, _msgs):
                raise AssertionError("empty image values must not use the multimodal path")

        mock_descriptor = MagicMock()
        mock_descriptor.instantiate.return_value = VLLMLikePrompter()

        batch = _PromptBatch(
            mock_descriptor,
            "text",
            "response",
            image_columns=["image", "images"],
        )
        table = pa.table(
            {
                "text": ["empty blob", "empty list item"],
                "image": pa.array([b"", None], type=pa.binary()),
                "images": pa.array([None, [b"", None]], type=pa.list_(pa.binary())),
            }
        )
        result = _drive(batch, table)

        assert captured_batches == [["empty blob", "empty list item"]]
        assert result.column("response").to_pylist() == [
            "response:empty blob",
            "response:empty list item",
        ]

    def test_prompt_batch_no_image_columns_text_only(self):
        """Without image_columns, behavior is identical to original."""
        from vane.ai.functions import _PromptBatch

        captured_messages = []

        class SimplePrompter:
            async def prompt(self, msgs):
                captured_messages.append(msgs)
                return "reply"

        mock_descriptor = MagicMock()
        mock_descriptor.instantiate.return_value = SimplePrompter()

        batch = _PromptBatch(mock_descriptor, "text", "response")
        table = pa.table({"text": ["Hello"]})
        result = _drive(batch, table)

        assert captured_messages[0] == ("Hello",)
        assert result.column("response").to_pylist() == ["reply"]


class TestGuesssMimeType:
    """Tests for the _guess_mime_type helper."""

    def test_png(self):
        from vane.ai.providers.openai import _guess_mime_type

        assert _guess_mime_type(b"\x89PNG\r\n\x1a\n") == "image/png"

    def test_jpeg(self):
        from vane.ai.providers.openai import _guess_mime_type

        assert _guess_mime_type(b"\xff\xd8\xff") == "image/jpeg"

    def test_gif(self):
        from vane.ai.providers.openai import _guess_mime_type

        assert _guess_mime_type(b"GIF89a") == "image/gif"

    def test_webp(self):
        from vane.ai.providers.openai import _guess_mime_type

        data = b"RIFF\x00\x00\x00\x00WEBP"
        assert _guess_mime_type(data) == "image/webp"

    def test_pdf(self):
        from vane.ai.providers.openai import _guess_mime_type

        assert _guess_mime_type(b"%PDF-1.4") == "application/pdf"

    def test_unknown(self):
        from vane.ai.providers.openai import _guess_mime_type

        assert _guess_mime_type(b"\x00\x01\x02") == "application/octet-stream"


# ---------------------------------------------------------------------------
# Token Metrics
# ---------------------------------------------------------------------------


class TestTokenMetrics:
    """Tests for vane.ai.metrics module."""

    def setup_method(self):
        from vane.ai.metrics import reset_token_metrics, set_token_metrics_callback

        reset_token_metrics()
        set_token_metrics_callback(None)

    def teardown_method(self):
        from vane.ai.metrics import reset_token_metrics, set_token_metrics_callback

        reset_token_metrics()
        set_token_metrics_callback(None)

    def test_record_and_get(self):
        from vane.ai.metrics import get_token_metrics, record_token_metrics

        record_token_metrics(
            protocol="prompt",
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        entries = get_token_metrics()
        assert len(entries) == 1
        e = entries[0]
        assert e.protocol == "prompt"
        assert e.model == "gpt-4o"
        assert e.provider == "openai"
        assert e.input_tokens == 100
        assert e.output_tokens == 50
        assert e.total_tokens == 150
        assert e.requests == 1

    def test_accumulation(self):
        from vane.ai.metrics import get_token_metrics, record_token_metrics

        record_token_metrics(
            protocol="prompt",
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
        )
        record_token_metrics(
            protocol="prompt",
            model="gpt-4o",
            provider="openai",
            input_tokens=200,
            output_tokens=80,
        )
        entries = get_token_metrics()
        assert len(entries) == 1
        e = entries[0]
        assert e.input_tokens == 300
        assert e.output_tokens == 130
        assert e.requests == 2

    def test_multiple_keys(self):
        from vane.ai.metrics import get_token_metrics, record_token_metrics

        record_token_metrics(
            protocol="prompt",
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
        )
        record_token_metrics(
            protocol="embed",
            model="text-embedding-3-small",
            provider="openai",
            input_tokens=500,
            total_tokens=500,
        )
        record_token_metrics(
            protocol="prompt",
            model="claude-3-5-sonnet",
            provider="anthropic",
            input_tokens=200,
            output_tokens=60,
        )
        entries = get_token_metrics()
        assert len(entries) == 3

    def test_none_tokens_ignored(self):
        from vane.ai.metrics import get_token_metrics, record_token_metrics

        record_token_metrics(
            protocol="prompt",
            model="m",
            provider="p",
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )
        e = get_token_metrics()[0]
        assert e.input_tokens == 0
        assert e.output_tokens == 0
        assert e.total_tokens == 0
        assert e.requests == 1

    def test_reset(self):
        from vane.ai.metrics import get_token_metrics, record_token_metrics, reset_token_metrics

        record_token_metrics(protocol="prompt", model="m", provider="p", input_tokens=10)
        assert len(get_token_metrics()) == 1
        reset_token_metrics()
        assert len(get_token_metrics()) == 0

    def test_summary(self):
        from vane.ai.metrics import get_token_metrics_summary, record_token_metrics

        record_token_metrics(
            protocol="prompt",
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        record_token_metrics(
            protocol="prompt",
            model="claude-3",
            provider="anthropic",
            input_tokens=200,
            output_tokens=80,
        )
        s = get_token_metrics_summary()
        assert s["total_input_tokens"] == 300
        assert s["total_output_tokens"] == 130
        assert s["total_requests"] == 2
        assert "openai" in s["by_provider"]
        assert "anthropic" in s["by_provider"]
        assert s["by_provider"]["openai"]["input_tokens"] == 100
        assert s["by_provider"]["anthropic"]["input_tokens"] == 200

    def test_callback(self):
        from vane.ai.metrics import record_token_metrics, set_token_metrics_callback

        received = []
        set_token_metrics_callback(lambda entry: received.append(entry))
        record_token_metrics(
            protocol="prompt",
            model="m",
            provider="p",
            input_tokens=10,
            output_tokens=5,
        )
        assert len(received) == 1
        assert received[0]["input_tokens"] == 10
        assert received[0]["output_tokens"] == 5
        assert received[0]["provider"] == "p"

    def test_callback_error_does_not_raise(self):
        from vane.ai.metrics import record_token_metrics, set_token_metrics_callback

        set_token_metrics_callback(lambda _: 1 / 0)
        # Should not raise
        record_token_metrics(protocol="prompt", model="m", provider="p", input_tokens=1)

    def test_remove_callback(self):
        from vane.ai.metrics import record_token_metrics, set_token_metrics_callback

        received = []
        set_token_metrics_callback(lambda entry: received.append(entry))
        record_token_metrics(protocol="prompt", model="m", provider="p", input_tokens=1)
        assert len(received) == 1
        set_token_metrics_callback(None)
        record_token_metrics(protocol="prompt", model="m", provider="p", input_tokens=1)
        assert len(received) == 1  # callback removed, no new entry

    def test_thread_safety(self):
        import threading

        from vane.ai.metrics import get_token_metrics, record_token_metrics

        def record_many():
            for _ in range(100):
                record_token_metrics(
                    protocol="prompt",
                    model="m",
                    provider="p",
                    input_tokens=1,
                    output_tokens=1,
                )

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        e = get_token_metrics()[0]
        assert e.requests == 400
        assert e.input_tokens == 400


class TestOpenAITokenMetrics:
    """Tests that OpenAI provider calls record_token_metrics."""

    def setup_method(self):
        from vane.ai.metrics import reset_token_metrics

        reset_token_metrics()

    def teardown_method(self):
        from vane.ai.metrics import reset_token_metrics

        reset_token_metrics()

    def test_chat_completions_records_usage(self):
        """Chat Completions response with usage triggers metrics."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.metrics import get_token_metrics
        from vane.ai.providers.openai import OpenAIPrompter

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 42
        mock_usage.completion_tokens = 18
        mock_usage.total_tokens = 60

        mock_choice = MagicMock()
        mock_choice.message.content = "hello"

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        prompter = OpenAIPrompter.__new__(OpenAIPrompter)
        prompter._client = mock_client
        prompter._model = "gpt-4o"
        prompter._use_chat_completions = True
        prompter._return_format = None
        prompter._options = {}
        prompter._system_message = None

        result = asyncio.run(prompter.prompt(("hi",)))
        assert result == "hello"
        entries = get_token_metrics()
        assert len(entries) == 1
        e = entries[0]
        assert e.provider == "openai"
        assert e.input_tokens == 42
        assert e.output_tokens == 18
        assert e.total_tokens == 60

    def test_responses_api_records_usage(self):
        """Responses API response with usage triggers metrics."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.metrics import get_token_metrics
        from vane.ai.providers.openai import OpenAIPrompter

        mock_usage = MagicMock()
        mock_usage.input_tokens = 30
        mock_usage.output_tokens = 15
        mock_usage.total_tokens = 45
        # Responses API doesn't have prompt_tokens/completion_tokens
        mock_usage.prompt_tokens = None
        mock_usage.completion_tokens = None

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.output_text = "world"

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        prompter = OpenAIPrompter.__new__(OpenAIPrompter)
        prompter._client = mock_client
        prompter._model = "gpt-4o"
        prompter._use_chat_completions = False
        prompter._return_format = None
        prompter._options = {}
        prompter._system_message = None

        result = asyncio.run(prompter.prompt(("hi",)))
        assert result == "world"
        entries = get_token_metrics()
        assert len(entries) == 1
        e = entries[0]
        assert e.input_tokens == 30
        assert e.output_tokens == 15

    def test_embed_records_usage(self):
        """Embedding response with usage triggers metrics."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.metrics import get_token_metrics
        from vane.ai.providers.openai import OpenAITextEmbedder

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 25
        mock_usage.total_tokens = 25

        mock_embedding = MagicMock()
        mock_embedding.index = 0
        mock_embedding.embedding = [0.1, 0.2, 0.3]

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.data = [mock_embedding]

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._client = mock_client
        embedder._model = "text-embedding-3-small"
        embedder._dimensions = None

        result = asyncio.run(embedder._embed_batch(["hello"]))
        assert len(result) == 1
        entries = get_token_metrics()
        assert len(entries) == 1
        e = entries[0]
        assert e.protocol == "embed"
        assert e.provider == "openai"
        assert e.input_tokens == 25

    def test_no_usage_no_error(self):
        """If response has no usage attribute, no metrics recorded, no error."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.metrics import get_token_metrics
        from vane.ai.providers.openai import OpenAIPrompter

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"

        mock_response = MagicMock(spec=[])  # spec=[] means no attributes
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        prompter = OpenAIPrompter.__new__(OpenAIPrompter)
        prompter._client = mock_client
        prompter._model = "gpt-4o"
        prompter._use_chat_completions = True
        prompter._return_format = None
        prompter._options = {}
        prompter._system_message = None

        result = asyncio.run(prompter.prompt(("test",)))
        assert result == "ok"
        assert len(get_token_metrics()) == 0


class TestOpenAITokenLimits:
    """Tests for per-model input token limits and oversized-text chunking."""

    def test_embedding_response_indices_restore_input_order(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.openai import OpenAITextEmbedder

        response = SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[20.0]),
                SimpleNamespace(index=0, embedding=[10.0]),
            ],
            usage=None,
        )
        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._client = MagicMock()
        embedder._client.embeddings.create = AsyncMock(return_value=response)
        embedder._model = "compatible-model"
        embedder._dimensions = 1

        result = asyncio.run(embedder._embed_batch(["row-0", "row-1"]))

        assert [embedding.tolist() for embedding in result] == [[10.0], [20.0]]

    def test_embedding_response_without_indices_preserves_physical_order(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.openai import OpenAITextEmbedder

        response = SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[10.0]),
                SimpleNamespace(embedding=[20.0]),
            ],
            usage=None,
        )
        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._client = MagicMock()
        embedder._client.embeddings.create = AsyncMock(return_value=response)
        embedder._model = "compatible-model"
        embedder._dimensions = 1

        result = asyncio.run(embedder._embed_batch(["row-0", "row-1"]))

        assert [embedding.tolist() for embedding in result] == [[10.0], [20.0]]

    @pytest.mark.parametrize("indices", [(0,), (0, 0), (0, 2), (0, "1")])
    def test_embedding_response_rejects_mixed_or_invalid_indices(self, indices):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.openai import OpenAITextEmbedder

        data = [SimpleNamespace(index=index, embedding=[float(position)]) for position, index in enumerate(indices)]
        data.extend(SimpleNamespace(embedding=[float(position)]) for position in range(len(data), 2))
        response = SimpleNamespace(data=data, usage=None)
        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._client = MagicMock()
        embedder._client.embeddings.create = AsyncMock(return_value=response)
        embedder._model = "compatible-model"
        embedder._dimensions = 1

        with pytest.raises(TypeError, match="invalid embedding indices"):
            asyncio.run(embedder._embed_batch(["row-0", "row-1"]))

    def test_get_input_token_limit_known_model(self):
        from vane.ai.providers.openai import _get_input_token_limit

        assert _get_input_token_limit("text-embedding-ada-002") == 8191
        assert _get_input_token_limit("text-embedding-3-small") == 8191
        assert _get_input_token_limit("text-embedding-3-large") == 8191

    def test_get_input_token_limit_unknown_model(self):
        from vane.ai.providers.openai import _get_input_token_limit

        assert _get_input_token_limit("custom-embed-model") == 8192

    def test_chunk_text_basic(self):
        from vane.ai.providers.openai import _chunk_text

        result = _chunk_text("abcdefgh", 3)
        assert result == ["abc", "def", "gh"]

    def test_chunk_text_exact(self):
        from vane.ai.providers.openai import _chunk_text

        result = _chunk_text("abcdef", 3)
        assert result == ["abc", "def"]

    def test_chunk_text_short(self):
        from vane.ai.providers.openai import _chunk_text

        result = _chunk_text("ab", 10)
        assert result == ["ab"]

    def test_oversized_input_gets_chunked(self):
        """An input exceeding input_text_token_limit is chunked and averaged."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.openai import OpenAITextEmbedder

        call_log: list[list[str]] = []

        async def mock_create(**kwargs):
            texts = kwargs["input"]
            call_log.append(texts)
            mock_response = MagicMock()
            mock_response.usage = None
            mock_response.data = []
            for index, _t in enumerate(texts):
                emb = MagicMock()
                emb.index = index
                emb.embedding = [1.0, 0.0, 0.0]
                mock_response.data.append(emb)
            return mock_response

        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._model = "text-embedding-3-small"
        embedder._dimensions = None
        embedder._batch_token_limit = 300_000
        embedder._input_text_token_limit = 10  # 10 tokens → 30 chars
        mock_client = AsyncMock()
        mock_client.embeddings.create = mock_create
        embedder._client = mock_client

        # "a" * 90 → 30 est_tokens > limit of 10 → chunked into 3 pieces of 30 chars
        result = asyncio.run(embedder.embed_text(["a" * 90]))

        assert len(result) == 1
        # Should have been chunked: one _embed_batch call with 3 chunks
        assert len(call_log) == 1
        assert len(call_log[0]) == 3
        # Result is L2-normalised
        norm = np.linalg.norm(result[0])
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)

    def test_oversized_input_chunks_still_respect_batch_token_limit(self):
        """Chunks from one long input are split across bounded requests."""
        import asyncio

        from vane.ai.providers.openai import OpenAITextEmbedder

        calls: list[list[str]] = []

        async def mock_embed_batch(texts):
            calls.append(list(texts))
            return [np.ones(2, dtype=np.float32) for _ in texts]

        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._model = "test-model"
        embedder._dimensions = 2
        embedder._batch_token_limit = 5
        embedder._input_text_token_limit = 3
        embedder._embed_batch = mock_embed_batch

        result = asyncio.run(embedder.embed_text(["a" * 60]))

        assert len(result) == 1
        assert len(calls) > 1
        assert all(sum((len(text) + 2) // 3 for text in call) <= 5 for call in calls)

    def test_normal_input_not_chunked(self):
        """Inputs within token limit are batched normally, not chunked."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.openai import OpenAITextEmbedder

        call_log: list[list[str]] = []

        async def mock_create(**kwargs):
            texts = kwargs["input"]
            call_log.append(texts)
            mock_response = MagicMock()
            mock_response.usage = None
            mock_response.data = []
            for index, _ in enumerate(texts):
                emb = MagicMock()
                emb.index = index
                emb.embedding = [0.5, 0.5]
                mock_response.data.append(emb)
            return mock_response

        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._model = "text-embedding-3-small"
        embedder._dimensions = None
        embedder._batch_token_limit = 300_000
        embedder._input_text_token_limit = 8191
        mock_client = AsyncMock()
        mock_client.embeddings.create = mock_create
        embedder._client = mock_client

        result = asyncio.run(embedder.embed_text(["hello", "world"]))

        assert len(result) == 2
        # Single batch call with both texts
        assert len(call_log) == 1
        assert call_log[0] == ["hello", "world"]

    def test_batch_splitting_still_works(self):
        """Batch token limit still triggers multi-call splitting."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.openai import OpenAITextEmbedder

        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            texts = kwargs["input"]
            mock_response = MagicMock()
            mock_response.usage = None
            mock_response.data = []
            for index, _ in enumerate(texts):
                emb = MagicMock()
                emb.index = index
                emb.embedding = [1.0]
                mock_response.data.append(emb)
            return mock_response

        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._model = "test-model"
        embedder._dimensions = None
        embedder._batch_token_limit = 5  # very small: 5 tokens ≈ 15 chars
        embedder._input_text_token_limit = 100
        mock_client = AsyncMock()
        mock_client.embeddings.create = mock_create
        embedder._client = mock_client

        # Each "a"*12 → 4 est tokens; two won't fit in one batch of limit 5
        result = asyncio.run(embedder.embed_text(["a" * 12, "b" * 12]))

        assert len(result) == 2
        assert call_count == 2  # split into 2 batches

    def test_mixed_oversized_and_normal(self):
        """Mix of oversized (chunked) and normal texts handled correctly."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.openai import OpenAITextEmbedder

        call_log: list[int] = []  # track number of texts per call

        async def mock_create(**kwargs):
            texts = kwargs["input"]
            call_log.append(len(texts))
            mock_response = MagicMock()
            mock_response.usage = None
            mock_response.data = []
            for i, _t in enumerate(texts):
                emb = MagicMock()
                emb.index = i
                emb.embedding = [float(i + 1), 0.0]
                mock_response.data.append(emb)
            return mock_response

        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._model = "test-model"
        embedder._dimensions = None
        embedder._batch_token_limit = 300_000
        embedder._input_text_token_limit = 10  # 10 tokens → 30 chars
        mock_client = AsyncMock()
        mock_client.embeddings.create = mock_create
        embedder._client = mock_client

        texts = [
            "short",  # normal
            "a" * 90,  # oversized → 3 chunks of 30 chars
            "also short",  # normal
        ]
        result = asyncio.run(embedder.embed_text(texts))

        assert len(result) == 3
        # First "short" is batched, then flushed before oversized
        # Oversized → separate _embed_batch with 3 chunks
        # "also short" → final flush
        assert len(call_log) == 3

    def test_descriptor_passes_token_limits(self):
        """OpenAITextEmbedderDescriptor passes token limits to embedder."""
        from unittest.mock import patch

        from vane.ai.providers.openai import OpenAITextEmbedderDescriptor

        desc = OpenAITextEmbedderDescriptor(
            provider_options={"api_key": "test"},
            model_name="text-embedding-3-small",
            embed_options={
                "batch_token_limit": 100_000,
                "input_text_token_limit": 4096,
            },
        )

        with patch("openai.AsyncOpenAI"):
            embedder = desc.instantiate()

        assert embedder._batch_token_limit == 100_000
        assert embedder._input_text_token_limit == 4096

    def test_descriptor_default_token_limits(self):
        """Default token limits when not specified in options."""
        from unittest.mock import patch

        from vane.ai.providers.openai import OpenAITextEmbedderDescriptor

        desc = OpenAITextEmbedderDescriptor(
            provider_options={"api_key": "test"},
            model_name="text-embedding-3-small",
        )

        with patch("openai.AsyncOpenAI"):
            embedder = desc.instantiate()

        assert embedder._batch_token_limit == 300_000
        assert embedder._input_text_token_limit == 8191  # model-specific


class TestAnthropicTokenMetrics:
    """Tests that Anthropic provider calls record_token_metrics."""

    def setup_method(self):
        from vane.ai.metrics import reset_token_metrics

        reset_token_metrics()

    def teardown_method(self):
        from vane.ai.metrics import reset_token_metrics

        reset_token_metrics()

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_prompt_records_usage(self):
        """Anthropic messages.create response records token metrics."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.metrics import get_token_metrics
        from vane.ai.providers.anthropic import AnthropicPrompter

        mock_usage = MagicMock()
        mock_usage.input_tokens = 55
        mock_usage.output_tokens = 20

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "answer"

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.content = [mock_text_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        prompter = AnthropicPrompter.__new__(AnthropicPrompter)
        prompter._client = mock_client
        prompter._model = "claude-3-5-sonnet-20241022"
        prompter._system_message = None
        prompter._return_format = None
        prompter._options = {}

        result = asyncio.run(prompter.prompt(("hello",)))
        assert result == "answer"
        entries = get_token_metrics()
        assert len(entries) == 1
        e = entries[0]
        assert e.provider == "anthropic"
        assert e.input_tokens == 55
        assert e.output_tokens == 20
        assert e.requests == 1


class TestGoogleTokenMetrics:
    """Tests that Google provider calls record_token_metrics."""

    def setup_method(self):
        from vane.ai.metrics import reset_token_metrics

        reset_token_metrics()

    def teardown_method(self):
        from vane.ai.metrics import reset_token_metrics

        reset_token_metrics()

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_prompt_records_usage(self):
        """Google generate_content response records token metrics."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.metrics import get_token_metrics
        from vane.ai.providers.google import GooglePrompter

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 33
        mock_usage.candidates_token_count = 12
        mock_usage.total_token_count = 45

        mock_response = MagicMock()
        mock_response.usage_metadata = mock_usage
        mock_response.text = "gemini says hi"

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        prompter = GooglePrompter.__new__(GooglePrompter)
        prompter._client = mock_client
        prompter._model = "gemini-2.0-flash"
        prompter._system_message = None
        prompter._return_format = None
        prompter._options = {}

        result = asyncio.run(prompter.prompt(("hello",)))
        assert result == "gemini says hi"
        entries = get_token_metrics()
        assert len(entries) == 1
        e = entries[0]
        assert e.provider == "google"
        assert e.input_tokens == 33
        assert e.output_tokens == 12
        assert e.total_tokens == 45


# ---------------------------------------------------------------------------
# Retry / on_error
# ---------------------------------------------------------------------------


class TestRetryAfterError:
    """Tests for RetryAfterError integration with retry helpers."""

    def test_retry_call_honors_retry_after(self):
        """_retry_call uses RetryAfterError.retry_after for wait time."""
        import time

        from vane.ai.functions import RetryAfterError, _retry_call

        calls = []

        def fn():
            calls.append(time.monotonic())
            if len(calls) < 2:
                raise RetryAfterError(retry_after=0.1, original=RuntimeError("429"))
            return "ok"

        result = _retry_call(fn, max_retries=2, on_error="raise")
        assert result == "ok"
        assert len(calls) == 2
        # Should have waited ~0.1s (the retry_after), not 1s (exponential backoff)
        gap = calls[1] - calls[0]
        assert gap >= 0.08  # allow timing slack
        assert gap < 0.5  # definitely not exponential backoff (1s)

    def test_retry_call_unwraps_original_on_exhaust(self):
        """When retries exhausted, the original exception is raised, not RetryAfterError."""
        from vane.ai.functions import RetryAfterError, _retry_call

        original = RuntimeError("rate limited")

        def fn():
            raise RetryAfterError(retry_after=0.01, original=original)

        with pytest.raises(RuntimeError, match="rate limited"):
            _retry_call(fn, max_retries=0, on_error="raise")

    def test_retry_call_async_honors_retry_after(self):
        import asyncio

        from vane.ai.functions import RetryAfterError, _retry_call_async

        calls = []

        async def fn():
            calls.append(1)
            if len(calls) < 2:
                raise RetryAfterError(retry_after=0.05, original=ValueError("503"))
            return "done"

        result = asyncio.run(_retry_call_async(fn, max_retries=2, on_error="raise"))
        assert result == "done"
        assert len(calls) == 2

    def test_retry_call_async_unwraps_original(self):
        import asyncio

        from vane.ai.functions import RetryAfterError, _retry_call_async

        async def fn():
            raise RetryAfterError(retry_after=0.01, original=ValueError("overloaded"))

        with pytest.raises(ValueError, match="overloaded"):
            asyncio.run(_retry_call_async(fn, max_retries=0, on_error="raise"))


class TestGoogleRetryHandling:
    """Tests for Google provider 429/503 → RetryAfterError conversion."""

    def test_google_429_raises_retry_after(self):
        """Google APIError with code=429 is converted to RetryAfterError."""
        from vane.ai.functions import RetryAfterError
        from vane.ai.providers.google import _raise_retry_after_on_google_error

        exc = Exception("rate limited")
        exc.code = 429
        exc.response = None

        with pytest.raises(RetryAfterError) as ctx:
            _raise_retry_after_on_google_error(exc)
        assert ctx.value.retry_after == 5.0  # default
        assert ctx.value.__cause__ is exc

    def test_google_503_raises_retry_after(self):
        """Google APIError with code=503 is converted to RetryAfterError."""
        from vane.ai.functions import RetryAfterError
        from vane.ai.providers.google import _raise_retry_after_on_google_error

        exc = Exception("service unavailable")
        exc.code = 503
        exc.response = None

        with pytest.raises(RetryAfterError) as ctx:
            _raise_retry_after_on_google_error(exc)
        assert ctx.value.retry_after == 5.0

    def test_google_429_with_retry_after_header(self):
        """Retry-After header from response is honoured."""
        from unittest.mock import MagicMock

        from vane.ai.functions import RetryAfterError
        from vane.ai.providers.google import _raise_retry_after_on_google_error

        mock_response = MagicMock()
        mock_response.headers = {"Retry-After": "10"}

        exc = Exception("rate limited")
        exc.code = 429
        exc.response = mock_response

        with pytest.raises(RetryAfterError) as ctx:
            _raise_retry_after_on_google_error(exc)
        assert ctx.value.retry_after == 10.0

    def test_google_400_not_retryable(self):
        """Non-retryable errors (400) are not converted."""
        from vane.ai.providers.google import _raise_retry_after_on_google_error

        exc = Exception("bad request")
        exc.code = 400
        exc.response = None

        # Should not raise — just returns
        _raise_retry_after_on_google_error(exc)

    def test_google_no_code_not_retryable(self):
        """Exceptions without .code attribute are not converted."""
        from vane.ai.providers.google import _raise_retry_after_on_google_error

        exc = RuntimeError("random error")
        # No .code attribute → should not raise
        _raise_retry_after_on_google_error(exc)


class TestEmbedProviderCapabilityErrors:
    @pytest.mark.parametrize("name", ["batch_size", "actor_number", "batch_token_limit"])
    def test_non_nullable_integer_options_reject_explicit_none_during_planning(self, name):
        with pytest.raises(ValueError, match=name):
            vane.ai.embed(vane.col("text"), dimensions=4, **{name: None})

    @pytest.mark.parametrize(
        "base_url",
        [
            "file:///tmp/embed",
            "https://user:provider-url-secret-sentinel@example.test/v1",
            "https://example.test/v1?organization=provider-url-secret-sentinel",
        ],
    )
    def test_openai_provider_preset_reuses_safe_base_url_validation(self, base_url):
        from vane.ai.providers.openai import OpenAIProvider

        with pytest.raises(ValueError) as exc_info:
            vane.ai.embed(vane.col("text"), provider=OpenAIProvider(base_url=base_url), dimensions=4)

        assert "provider-url-secret-sentinel" not in str(exc_info.value)

    @pytest.mark.parametrize(
        "options",
        [
            {"trust_remote_code": True},
            {"local_files_only": "yes"},
            {"revision": "   "},
        ],
    )
    def test_transformers_provider_preset_reuses_embed_value_validation(self, options):
        from vane.ai.providers.transformers import TransformersProvider

        with pytest.raises(ValueError):
            vane.ai.embed(vane.col("text"), provider=TransformersProvider(**options))

    @pytest.mark.parametrize("status_code", [404, 405, 501])
    def test_openai_dynamic_endpoint_error_preserves_capability_context(self, monkeypatch, status_code):
        import asyncio
        import sys
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.provider import ProviderCapabilityError
        from vane.ai.providers.openai import OpenAIProvider

        class EndpointError(Exception):
            pass

        original = EndpointError("model is unavailable on this endpoint")
        original.status_code = status_code
        client = MagicMock()
        client.embeddings.create = AsyncMock(side_effect=original)
        monkeypatch.setitem(
            sys.modules,
            "openai",
            SimpleNamespace(OpenAIError=EndpointError, AsyncOpenAI=MagicMock(return_value=client)),
        )

        descriptor = OpenAIProvider(name="openai-compatible-alias").get_text_embedder(
            model="endpoint-only-model",
            dimensions=4,
        )
        embedder = descriptor.instantiate()

        with pytest.raises(ProviderCapabilityError) as exc_info:
            asyncio.run(embedder._embed_batch(["hello"]))

        error = exc_info.value
        assert (error.provider, error.model, error.capability) == (
            "openai-compatible-alias",
            "endpoint-only-model",
            "embedding endpoint/model",
        )
        assert error.original_error is original
        assert error.__cause__ is original

    def test_openai_input_validation_error_is_not_misclassified_as_capability(self, monkeypatch):
        import asyncio
        import sys
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.openai import OpenAITextEmbedder

        class InputError(Exception):
            status_code = 400
            param = "input"
            code = "context_length_exceeded"

        original = InputError("one input is too long")
        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAIError=InputError))

        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._client = MagicMock()
        embedder._client.embeddings.create = AsyncMock(side_effect=original)
        embedder._model = "endpoint-model"
        embedder._dimensions = 4

        with pytest.raises(InputError) as exc_info:
            asyncio.run(embedder._embed_batch(["bad input"]))

        assert exc_info.value is original

    def test_openai_structured_model_error_is_a_capability_error(self, monkeypatch):
        import asyncio
        import sys
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.provider import ProviderCapabilityError
        from vane.ai.providers.openai import OpenAITextEmbedder

        class ModelError(Exception):
            status_code = 400
            param = "model"
            code = "model_not_found"

        original = ModelError("selected model cannot embed")
        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAIError=ModelError))

        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._client = MagicMock()
        embedder._client.embeddings.create = AsyncMock(side_effect=original)
        embedder._model = "chat-only-model"
        embedder._dimensions = 4

        with pytest.raises(ProviderCapabilityError) as exc_info:
            asyncio.run(embedder._embed_batch(["hello"]))

        assert exc_info.value.original_error is original

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    @pytest.mark.parametrize(
        ("code", "status"),
        [
            (404, None),
            (405, None),
            (501, None),
            (None, "NOT_FOUND"),
            (None, "UNIMPLEMENTED"),
        ],
    )
    def test_google_dynamic_endpoint_error_preserves_capability_context(self, monkeypatch, code, status):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from google import genai

        from vane.ai.provider import ProviderCapabilityError
        from vane.ai.providers.google import GoogleProvider

        class EndpointError(Exception):
            pass

        original = EndpointError("model does not implement embed_content")
        original.code = code
        original.status = status
        client = MagicMock()
        monkeypatch.setattr(genai, "Client", lambda **_kwargs: client)
        provider = GoogleProvider(
            name="google-alias",
            embedding_model="chat-only-model",
            embedding_dimensions=4,
        )
        descriptor = provider.get_text_embedder()
        embedder = descriptor.instantiate()
        embedder._client.aio.models.embed_content = AsyncMock(side_effect=original)

        with pytest.raises(ProviderCapabilityError) as exc_info:
            asyncio.run(embedder.embed_text(["hello"]))

        error = exc_info.value
        assert (error.provider, error.model, error.capability) == (
            "google-alias",
            "chat-only-model",
            "embedding endpoint/model",
        )
        assert error.original_error is original
        assert error.__cause__ is original

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_google_generic_invalid_input_is_not_misclassified_as_capability(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.google import GoogleTextEmbedder

        class InputError(Exception):
            code = 400
            status = "INVALID_ARGUMENT"
            details = {"error": {"message": "input is too long"}}

        original = InputError("input is too long")
        embedder = GoogleTextEmbedder.__new__(GoogleTextEmbedder)
        embedder._client = MagicMock()
        embedder._client.aio.models.embed_content = AsyncMock(side_effect=original)
        embedder._model = "embedding-model"
        embedder._dimensions = 4
        embedder._options = {}

        with pytest.raises(InputError) as exc_info:
            asyncio.run(embedder.embed_text(["bad input"]))

        assert exc_info.value is original

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_google_structured_model_field_error_is_a_capability_error(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.provider import ProviderCapabilityError
        from vane.ai.providers.google import GoogleTextEmbedder

        class ModelError(Exception):
            code = 400
            status = "INVALID_ARGUMENT"
            details = {"fieldViolations": [{"field": "request.model"}]}

        original = ModelError("selected model cannot embed")
        embedder = GoogleTextEmbedder.__new__(GoogleTextEmbedder)
        embedder._client = MagicMock()
        embedder._client.aio.models.embed_content = AsyncMock(side_effect=original)
        embedder._model = "chat-only-model"
        embedder._dimensions = 4
        embedder._options = {}

        with pytest.raises(ProviderCapabilityError) as exc_info:
            asyncio.run(embedder.embed_text(["hello"]))

        assert exc_info.value.original_error is original

    def test_transformers_not_implemented_model_is_a_capability_error(self, monkeypatch):
        import contextlib
        import sys
        from types import SimpleNamespace

        from vane.ai.provider import ProviderCapabilityError
        from vane.ai.providers.transformers import TransformersProvider

        class UnsupportedModel:
            def eval(self):
                return self

            def encode(self, *_args, **_kwargs):
                raise NotImplementedError("model has no sentence-embedding capability")

        model = UnsupportedModel()
        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            SimpleNamespace(SentenceTransformer=lambda *_args, **_kwargs: model),
        )
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=contextlib.nullcontext))
        provider = TransformersProvider(name="transformers-alias")
        descriptor = provider.get_text_embedder(model="chat-only-model", dimensions=4)
        embedder = descriptor.instantiate()

        with pytest.raises(ProviderCapabilityError) as exc_info:
            embedder.embed_text(["hello"])

        assert descriptor.get_provider() == "transformers-alias"
        assert exc_info.value.provider == "transformers-alias"
        assert exc_info.value.original_error is not None


class TestRetryCall:
    """Tests for _retry_call and _retry_call_async helpers."""

    def test_success_no_retry(self):
        from vane.ai.functions import _retry_call

        calls = []

        def fn():
            calls.append(1)
            return "ok"

        result = _retry_call(fn, max_retries=3, on_error="raise")
        assert result == "ok"
        assert len(calls) == 1

    def test_retry_then_success(self):
        from vane.ai.functions import _retry_call

        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("transient")
            return "recovered"

        result = _retry_call(fn, max_retries=3, on_error="raise")
        assert result == "recovered"
        assert len(calls) == 3

    def test_retry_exhausted_raises(self):
        from vane.ai.functions import _retry_call

        def fn():
            raise RuntimeError("permanent")

        with pytest.raises(RuntimeError, match="permanent"):
            _retry_call(fn, max_retries=1, on_error="raise")

    def test_on_error_log_returns_default(self):
        from vane.ai.functions import _retry_call

        def fn():
            raise RuntimeError("fail")

        result = _retry_call(fn, max_retries=0, on_error="log", default="fallback")
        assert result == "fallback"

    def test_on_error_ignore_returns_default(self):
        from vane.ai.functions import _retry_call

        def fn():
            raise RuntimeError("fail")

        result = _retry_call(fn, max_retries=0, on_error="ignore", default=42)
        assert result == 42

    def test_on_error_ignore_returns_none_by_default(self):
        from vane.ai.functions import _retry_call

        def fn():
            raise RuntimeError("fail")

        result = _retry_call(fn, max_retries=0, on_error="ignore")
        assert result is None

    def test_awaitable_result_handled(self):
        """_retry_call drives awaitables through the provided run_async."""
        import asyncio

        from vane.ai.functions import _retry_call

        async def async_fn():
            return "async_result"

        loop = asyncio.new_event_loop()
        try:
            result = _retry_call(async_fn, max_retries=0, on_error="raise", run_async=loop.run_until_complete)
        finally:
            loop.close()
        assert result == "async_result"

    def test_awaitable_result_without_run_async_raises(self):
        """Without a bound runtime an awaitable result is a hard error."""
        from vane.ai.functions import _retry_call

        async def async_fn():
            return "unreachable"

        with pytest.raises(RuntimeError, match="bind_async_runtime"):
            _retry_call(async_fn, max_retries=0, on_error="ignore")

    def test_retry_call_async(self):
        import asyncio

        from vane.ai.functions import _retry_call_async

        calls = []

        async def fn():
            calls.append(1)
            if len(calls) < 2:
                raise ValueError("transient")
            return "ok"

        result = asyncio.run(_retry_call_async(fn, max_retries=2, on_error="raise"))
        assert result == "ok"
        assert len(calls) == 2

    def test_retry_call_async_exhausted_raises(self):
        import asyncio

        from vane.ai.functions import _retry_call_async

        async def fn():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(_retry_call_async(fn, max_retries=1, on_error="raise"))

    def test_retry_call_async_on_error_log(self):
        import asyncio

        from vane.ai.functions import _retry_call_async

        async def fn():
            raise RuntimeError("fail")

        result = asyncio.run(_retry_call_async(fn, max_retries=0, on_error="log", default="safe"))
        assert result == "safe"


class TestWrapperRetry:
    """Tests that wrapper classes use retry/on_error correctly."""

    def _make_embed_descriptor(self, embed_fn):
        """Create a minimal descriptor + embedder for testing."""
        desc = MagicMock(spec=[])
        desc.get_dimensions = MagicMock(
            return_value=MagicMock(
                as_arrow_type=MagicMock(return_value=pa.list_(pa.float32(), 3)),
                list_size=3,
            )
        )
        embedder = MagicMock(spec=[])
        embedder.embed_text = embed_fn
        desc.instantiate = MagicMock(return_value=embedder)
        return desc

    def test_embed_retry_success(self, monkeypatch):
        from vane.ai.functions import _EmbedTextBatch

        monkeypatch.setattr("vane.ai.functions.time.sleep", lambda _seconds: None)
        calls = []

        class ServiceUnavailableError(RuntimeError):
            status_code = 503

        def embed(texts):
            calls.append(1)
            if len(calls) < 2:
                raise ServiceUnavailableError("API error")
            return [np.array([1.0, 2.0, 3.0])] * len(texts)

        desc = self._make_embed_descriptor(embed)
        wrapper = _EmbedTextBatch(desc, "text", "emb", 3, max_retries=3, on_error="raise")
        table = pa.table({"text": ["hello"]})
        result = _drive(wrapper, table)
        assert result.column("emb").length() == 1
        assert len(calls) == 2

    @pytest.mark.parametrize("status_code", [None, 400, 401, 403, 404, 405, 422, 501])
    def test_embed_does_not_retry_nontransient_http_error(self, status_code):
        from vane.ai.functions import _EmbedTextBatch

        calls = []

        class NonTransientError(RuntimeError):
            pass

        error = NonTransientError("invalid input")
        if status_code is not None:
            error.status_code = status_code

        def embed(texts):
            calls.append(list(texts))
            raise error

        desc = self._make_embed_descriptor(embed)
        wrapper = _EmbedTextBatch(desc, "text", "emb", 3, max_retries=3, on_error="raise")

        with pytest.raises(NonTransientError, match="invalid input"):
            _drive(wrapper, pa.table({"text": ["hello"]}))

        assert calls == [["hello"]]

    def test_embed_ignore_isolates_without_retrying_nontransient_error(self):
        from vane.ai.functions import _EmbedTextBatch

        calls = []

        class UnprocessableInputError(RuntimeError):
            status_code = 422

        def embed(texts):
            calls.append(list(texts))
            raise UnprocessableInputError("invalid input")

        desc = self._make_embed_descriptor(embed)
        wrapper = _EmbedTextBatch(desc, "text", "emb", 3, max_retries=3, on_error="ignore")
        result = _drive(wrapper, pa.table({"text": ["first", "second"]}))

        assert result.column("emb").to_pylist() == [None, None]
        assert calls == [["first", "second"], ["first"], ["second"]]

    def test_embed_on_error_ignore(self):
        from vane.ai.functions import _EmbedTextBatch

        def embed(_texts):
            raise RuntimeError("permanent failure")

        desc = self._make_embed_descriptor(embed)
        wrapper = _EmbedTextBatch(desc, "text", "emb", 3, max_retries=0, on_error="ignore")
        table = pa.table({"text": ["hello"]})
        result = _drive(wrapper, table)
        assert result.column("emb").to_pylist() == [None]

    def test_classify_on_error_log(self):
        from vane.ai.functions import _ClassifyTextBatch

        def classify(_texts, _labels):
            raise RuntimeError("fail")

        desc = MagicMock(spec=[])
        classifier = MagicMock(spec=[])
        classifier.classify_text = classify
        desc.instantiate = MagicMock(return_value=classifier)
        wrapper = _ClassifyTextBatch(desc, "text", "label", ["a", "b"], max_retries=0, on_error="log")
        table = pa.table({"text": ["hello"]})
        result = wrapper(table)
        assert result.column("label").to_pylist() == [None]

    def test_prompt_retry_per_row(self):
        """_PromptBatch retries each individual prompt call."""
        from vane.ai.functions import _PromptBatch

        call_count = 0

        class FakePrompter:
            async def prompt(self, _msgs):
                nonlocal call_count
                call_count += 1
                if call_count <= 1:
                    raise RuntimeError("rate limit")
                return f"answer-{call_count}"

        desc = MagicMock(spec=[])
        desc.instantiate = MagicMock(return_value=FakePrompter())
        wrapper = _PromptBatch(
            desc,
            "text",
            "response",
            max_api_concurrency=1,
            max_retries=2,
            on_error="raise",
        )
        table = pa.table({"text": ["q1"]})
        result = _drive(wrapper, table)
        assert result.column("response").to_pylist()[0] is not None
        assert call_count == 2

    def test_prompt_on_error_ignore(self):
        """_PromptBatch returns None on failure with on_error='ignore'."""
        from vane.ai.functions import _PromptBatch

        class FailPrompter:
            async def prompt(self, _msgs):
                raise RuntimeError("always fails")

        desc = MagicMock(spec=[])
        desc.instantiate = MagicMock(return_value=FailPrompter())
        wrapper = _PromptBatch(
            desc,
            "text",
            "response",
            max_api_concurrency=1,
            max_retries=0,
            on_error="ignore",
        )
        table = pa.table({"text": ["q1"]})
        result = _drive(wrapper, table)
        assert result.column("response").to_pylist() == [None]

    def test_prompt_batch_api_retry(self):
        """_PromptBatch retries prompt_batch() calls too."""
        from vane.ai.functions import _PromptBatch

        calls = []

        class FakeBatchPrompter:
            def prompt_batch(self, texts):
                calls.append(1)
                if len(calls) < 2:
                    raise RuntimeError("batch error")
                return ["ok"] * len(texts)

        desc = MagicMock(spec=[])
        desc.instantiate = MagicMock(return_value=FakeBatchPrompter())
        wrapper = _PromptBatch(
            desc,
            "text",
            "response",
            max_retries=2,
            on_error="raise",
        )
        table = pa.table({"text": ["q1", "q2"]})
        result = _drive(wrapper, table)
        assert result.column("response").to_pylist() == ["ok", "ok"]
        assert len(calls) == 2
