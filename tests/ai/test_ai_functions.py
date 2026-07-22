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


class TestEmbedText:
    def test_embed_text_basic(self):
        """embed_text produces a relation with embedding column."""
        from vane.ai.functions import embed_text

        conn = vane.connect()
        rel = conn.sql("SELECT 'hello' AS text UNION ALL SELECT 'world' AS text")

        result = embed_text(
            rel,
            "text",
            provider=MockProvider(),
        )

        rows = result.fetchall()
        assert len(rows) == 2
        # Each embedding should be a list of 4 floats
        for row in rows:
            emb = row[0]
            assert len(emb) == 4

    def test_embed_text_custom_dimensions(self):
        """embed_text respects dimensions parameter."""
        from vane.ai.functions import embed_text

        conn = vane.connect()
        rel = conn.sql("SELECT 'test' AS text")

        result = embed_text(
            rel,
            "text",
            provider=MockProvider(),
            dimensions=8,
        )

        rows = result.fetchall()
        assert len(rows[0][0]) == 8

    def test_embed_text_custom_output_column(self):
        """embed_text uses custom output column name."""
        from vane.ai.functions import embed_text

        conn = vane.connect()
        rel = conn.sql("SELECT 'test' AS text")

        result = embed_text(
            rel,
            "text",
            provider=MockProvider(),
            output_column="my_emb",
        )

        rows = result.fetchall()
        assert len(rows) == 1

    def test_embed_text_handles_null(self):
        """embed_text handles NULL values by converting to empty string."""
        from vane.ai.functions import embed_text

        conn = vane.connect()
        rel = conn.sql("SELECT NULL::VARCHAR AS text")

        result = embed_text(
            rel,
            "text",
            provider=MockProvider(),
        )

        rows = result.fetchall()
        assert len(rows) == 1
        # Empty string → len 0, so all zeros
        assert all(v == 0.0 for v in rows[0][0])


class TestClassifyText:
    def test_classify_text_basic(self):
        """classify_text produces a relation with label column."""
        from vane.ai.functions import classify_text

        conn = vane.connect()
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

        conn = vane.connect()
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

        wrapper = _EmbedTextBatch(MockTextEmbedderDescriptor(dim=4), "text", "emb")
        restored = pickle.loads(pickle.dumps(wrapper))
        table = pa.table({"text": ["hello", "world"]})
        result = restored(table)
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
    """Tests for the vLLM provider and descriptor."""

    def test_provider_loads(self):
        """Vllm provider is registered and loadable."""
        from vane.ai.provider import PROVIDERS

        assert "vllm" in PROVIDERS

    def test_descriptor_creates(self):
        """VLLMPrompterDescriptor can be created with default settings."""
        from vane.ai.providers.vllm import VLLMPrompterDescriptor

        desc = VLLMPrompterDescriptor(
            model_name="Qwen/Qwen3-1.7B",
        )
        assert desc.get_provider() == "vllm"
        assert desc.get_model() == "Qwen/Qwen3-1.7B"

    def test_descriptor_pickle_roundtrip(self):
        """VLLMPrompterDescriptor survives pickle (required for Ray)."""
        import pickle

        from vane.ai.providers.vllm import VLLMPrompterDescriptor

        desc = VLLMPrompterDescriptor(
            model_name="meta-llama/Llama-3.1-8B",
            system_message="You are a helpful assistant.",
            vllm_options={
                "engine_args": {"max_model_len": 2048},
                "generate_args": {"sampling_params": {"max_tokens": 256}},
                "gpus_per_actor": 1,
            },
        )
        restored = pickle.loads(pickle.dumps(desc))
        assert restored.model_name == desc.model_name
        assert restored.system_message == desc.system_message
        assert restored.vllm_options == desc.vllm_options

    def test_udf_options_defaults(self):
        """VLLMPrompterDescriptor produces correct UDFOptions."""
        from vane.ai.providers.vllm import VLLMPrompterDescriptor

        desc = VLLMPrompterDescriptor(
            model_name="Qwen/Qwen3-1.7B",
            vllm_options={"gpus_per_actor": 2},
        )
        opts = desc.get_udf_options()
        assert opts.num_gpus == 2
        assert opts.on_error == "raise"

    def test_provider_get_prompter(self):
        """VLLMProvider.get_prompter returns a VLLMPrompterDescriptor."""
        from vane.ai.providers.vllm import VLLMPrompterDescriptor, VLLMProvider

        provider = VLLMProvider()
        desc = provider.get_prompter(
            model="Qwen/Qwen3-1.7B",
            system_message="Be concise.",
            engine_args={"max_model_len": 1024},
        )
        assert isinstance(desc, VLLMPrompterDescriptor)
        assert desc.model_name == "Qwen/Qwen3-1.7B"
        assert desc.system_message == "Be concise."
        assert desc.vllm_options["engine_args"] == {"max_model_len": 1024}

    def test_prompt_batch_uses_batch_api(self):
        """_PromptBatch detects prompt_batch() method on vLLM prompter."""
        from unittest.mock import MagicMock, patch

        from vane.ai.providers.vllm import VLLMPrompterDescriptor

        desc = VLLMPrompterDescriptor(model_name="test-model")

        # Mock the instantiate to return a fake prompter with prompt_batch
        mock_prompter = MagicMock()
        mock_prompter.prompt_batch.return_value = ["Hello!", "World!"]

        with patch.object(desc, "instantiate", return_value=mock_prompter):
            from vane.ai.functions import _PromptBatch

            batch = _PromptBatch(desc, "text", "response")
            table = pa.table({"text": ["hi", "hey"]})
            result = batch(table)

        mock_prompter.prompt_batch.assert_called_once_with(["hi", "hey"])
        assert result.column("response").to_pylist() == ["Hello!", "World!"]

    def test_system_message_formatting(self):
        """VLLMPrompter prepends system message to prompts."""
        from vane.ai.providers.vllm import VLLMPrompter

        prompter = VLLMPrompter(
            model="test",
            system_message="Be brief.",
        )
        assert prompter._format_prompt("Hello") == "Be brief.\n\nHello"

    def test_no_system_message(self):
        """VLLMPrompter passes through prompts when no system_message."""
        from vane.ai.providers.vllm import VLLMPrompter

        prompter = VLLMPrompter(model="test")
        assert prompter._format_prompt("Hello") == "Hello"

    def test_prompter_uses_background_executor_for_sync_actor_calls(self, monkeypatch):
        """The synchronous prompter must not reuse the enclosing Ray actor loop."""
        import vane.execution.vllm as vllm_executor
        from vane.ai.providers.vllm import VLLMPrompter

        captured = {}
        sentinel = object()

        def fake_build_executor(model, options):
            captured.update(model=model, options=options)
            return sentinel

        monkeypatch.setattr(vllm_executor, "build_executor", fake_build_executor)
        prompter = VLLMPrompter(model="test", vllm_options={"use_threading": False})

        assert prompter._ensure_executor() is sentinel
        assert captured["model"] == "test"
        assert captured["options"]["use_threading"] is True
        assert captured["options"]["_force_background_thread"] is True

    def test_local_executor_can_force_background_loop_inside_ray_actor(self, monkeypatch):
        """Sync actor wrappers need a loop thread even when Ray reports actor context."""
        import sys
        import types

        import vane.execution.vllm as vllm_executor

        fake_vllm = types.ModuleType("vllm")

        class SamplingParams:
            pass

        fake_vllm.SamplingParams = SamplingParams
        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
        monkeypatch.setattr(vllm_executor.LocalVLLMExecutor, "_detect_ray_actor", staticmethod(lambda: True))

        def fake_run_event_loop(executor):
            executor.loop = object()
            executor.loop_ready.set()

        monkeypatch.setattr(vllm_executor.LocalVLLMExecutor, "_run_event_loop", fake_run_event_loop)

        executor = vllm_executor.LocalVLLMExecutor(
            "test-model",
            {},
            {},
            force_background_thread=True,
        )

        assert executor._ray_actor_mode is False
        assert executor.loop_ready.is_set()

    def test_prompt_batch_errors_when_wait_returns_without_result(self):
        """VLLMPrompter treats an empty wait wakeup as an executor contract error."""
        from vane.ai.providers.vllm import VLLMPrompter

        class EmptyWakeupExecutor:
            def __init__(self):
                self.wait_calls = 0

            def submit(self, _prefix, _prompts, _rows):
                pass

            def finished_submitting(self):
                pass

            def take_ready_result(self):
                return None

            def all_tasks_finished(self):
                return False

            def wait_for_result(self):
                self.wait_calls += 1

        prompter = VLLMPrompter(model="test")
        executor = EmptyWakeupExecutor()
        prompter._executor = executor

        with pytest.raises(RuntimeError, match="wait_for_result returned without a ready result"):
            prompter.prompt_batch(["a", "b"])
        assert executor.wait_calls == 1


# ---------------------------------------------------------------------------
# vLLM Structured Output tests
# ---------------------------------------------------------------------------


class TestVLLMStructuredOutput:
    """Tests for vLLM provider structured output via guided decoding."""

    def test_json_schema_from_pydantic(self):
        """_json_schema_from_return_format extracts schema from Pydantic model."""
        from pydantic import BaseModel

        from vane.ai.providers.vllm import _json_schema_from_return_format

        class Person(BaseModel):
            name: str
            age: int

        schema = _json_schema_from_return_format(Person)
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_json_schema_from_dict(self):
        """_json_schema_from_return_format passes dicts through."""
        from vane.ai.providers.vllm import _json_schema_from_return_format

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        assert _json_schema_from_return_format(schema) is schema

    def test_json_schema_from_none(self):
        """_json_schema_from_return_format returns empty dict for None."""
        from vane.ai.providers.vllm import _json_schema_from_return_format

        assert _json_schema_from_return_format(None) == {}

    def test_json_schema_bad_type(self):
        """_json_schema_from_return_format raises for unsupported types."""
        from vane.ai.providers.vllm import _json_schema_from_return_format

        with pytest.raises(TypeError, match="return_format must be"):
            _json_schema_from_return_format("bad")

    def test_parse_structured_output_pydantic(self):
        """_parse_structured_output validates JSON into Pydantic model."""
        from pydantic import BaseModel

        from vane.ai.providers.vllm import _parse_structured_output

        class Person(BaseModel):
            name: str
            age: int

        result = _parse_structured_output('{"name": "Alice", "age": 30}', Person)
        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30

    def test_parse_structured_output_dict_schema(self):
        """_parse_structured_output returns dict when schema is a dict."""
        from vane.ai.providers.vllm import _parse_structured_output

        schema = {"type": "object"}
        result = _parse_structured_output('{"x": 1}', schema)
        assert result == {"x": 1}

    def test_parse_structured_output_none(self):
        """_parse_structured_output returns None for None input."""
        from vane.ai.providers.vllm import _parse_structured_output

        assert _parse_structured_output(None, {"type": "object"}) is None

    def test_descriptor_with_return_format(self):
        """VLLMPrompterDescriptor stores return_format."""
        from pydantic import BaseModel

        from vane.ai.providers.vllm import VLLMPrompterDescriptor

        class Item(BaseModel):
            label: str

        desc = VLLMPrompterDescriptor(
            model_name="test-model",
            return_format=Item,
        )
        assert desc.return_format is Item

    def test_descriptor_pickle_with_return_format(self):
        """VLLMPrompterDescriptor with return_format survives pickle."""
        import pickle

        import cloudpickle
        from pydantic import BaseModel

        from vane.ai.providers.vllm import VLLMPrompterDescriptor

        class Score(BaseModel):
            value: float
            label: str

        desc = VLLMPrompterDescriptor(
            model_name="test-model",
            return_format=Score,
        )
        restored = pickle.loads(cloudpickle.dumps(desc))
        assert restored.return_format is not None
        assert restored.model_name == "test-model"
        # Validate the restored return_format still works
        obj = restored.return_format(value=1.0, label="test")
        assert obj.value == 1.0

    def test_provider_get_prompter_with_return_format(self):
        """VLLMProvider.get_prompter forwards return_format."""
        from pydantic import BaseModel

        from vane.ai.providers.vllm import VLLMPrompterDescriptor, VLLMProvider

        class Output(BaseModel):
            text: str

        provider = VLLMProvider()
        desc = provider.get_prompter(
            model="test-model",
            return_format=Output,
        )
        assert isinstance(desc, VLLMPrompterDescriptor)
        assert desc.return_format is Output

    def test_prompter_injects_structured_output(self):
        """VLLMPrompter injects structured output config into sampling_params."""
        from pydantic import BaseModel

        from vane.ai.providers.vllm import VLLMPrompter

        class Entity(BaseModel):
            name: str
            kind: str

        prompter = VLLMPrompter(
            model="test-model",
            return_format=Entity,
            vllm_options={"generate_args": {"sampling_params": {"max_tokens": 100}}},
        )
        sp = prompter._options["generate_args"]["sampling_params"]
        schema = sp["structured_outputs"]["value"]
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "guided_json" not in sp
        assert sp["max_tokens"] == 100

    def test_prompter_injects_structured_output_empty_options(self):
        """VLLMPrompter creates generate_args/sampling_params if absent."""
        from pydantic import BaseModel

        from vane.ai.providers.vllm import VLLMPrompter

        class Tag(BaseModel):
            value: str

        prompter = VLLMPrompter(
            model="test-model",
            return_format=Tag,
        )
        sp = prompter._options["generate_args"]["sampling_params"]
        assert "structured_outputs" in sp
        assert "guided_json" not in sp

    def test_prompter_no_return_format_no_structured_output(self):
        """VLLMPrompter without return_format does not inject structured output."""
        from vane.ai.providers.vllm import VLLMPrompter

        prompter = VLLMPrompter(model="test-model")
        gen_args = prompter._options.get("generate_args", {})
        sp = gen_args.get("sampling_params", {})
        if isinstance(sp, dict):
            assert "guided_json" not in sp
            assert "structured_outputs" not in sp

    def test_maybe_parse_with_return_format(self):
        """_maybe_parse parses JSON when return_format is set."""
        from pydantic import BaseModel

        from vane.ai.providers.vllm import VLLMPrompter

        class Item(BaseModel):
            x: int

        prompter = VLLMPrompter(model="test", return_format=Item)
        result = prompter._maybe_parse('{"x": 42}')
        assert isinstance(result, Item)
        assert result.x == 42

    def test_maybe_parse_without_return_format(self):
        """_maybe_parse returns raw text when no return_format."""
        from vane.ai.providers.vllm import VLLMPrompter

        prompter = VLLMPrompter(model="test")
        assert prompter._maybe_parse("hello") == "hello"

    def test_maybe_parse_none(self):
        """_maybe_parse returns None for None input."""
        from pydantic import BaseModel

        from vane.ai.providers.vllm import VLLMPrompter

        class Item(BaseModel):
            x: int

        prompter = VLLMPrompter(model="test", return_format=Item)
        assert prompter._maybe_parse(None) is None

    def test_dict_schema_structured_output(self):
        """VLLMPrompter accepts raw dict schema for structured output."""
        from vane.ai.providers.vllm import VLLMPrompter

        schema = {"type": "object", "properties": {"n": {"type": "number"}}}
        prompter = VLLMPrompter(model="test", return_format=schema)
        sp = prompter._options["generate_args"]["sampling_params"]
        assert sp["structured_outputs"]["value"] == schema
        assert "guided_json" not in sp


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

    def test_provider_descriptors_preserve_num_gpus(self):
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor
        from vane.ai.providers.google import GooglePrompterDescriptor, GoogleTextEmbedderDescriptor
        from vane.ai.providers.openai import OpenAIPrompterDescriptor, OpenAITextEmbedderDescriptor

        assert OpenAITextEmbedderDescriptor(embed_options={"num_gpus": 1}).get_udf_options().num_gpus == 1
        assert OpenAIPrompterDescriptor(prompt_options={"num_gpus": 2}).get_udf_options().num_gpus == 2
        assert AnthropicPrompterDescriptor(prompt_options={"num_gpus": 3}).get_udf_options().num_gpus == 3
        assert GoogleTextEmbedderDescriptor(embed_options={"num_gpus": 4}).get_udf_options().num_gpus == 4
        assert GooglePrompterDescriptor(prompt_options={"num_gpus": 5}).get_udf_options().num_gpus == 5

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
        result = batch(table)

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
        result = batch(table)

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
            prompt_options={"temperature": 0.7},
        )
        restored = pickle.loads(pickle.dumps(desc))
        assert restored.model_name == desc.model_name
        assert restored.system_message == desc.system_message
        assert restored.prompt_options == desc.prompt_options

    def test_udf_options(self):
        """Anthropic descriptor produces correct UDFOptions."""
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        desc = AnthropicPrompterDescriptor()
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
            api_key="call-key",
            base_url="https://call.example",
            timeout=30,
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
        """GoogleTextEmbedderDescriptor can be created."""
        from vane.ai.providers.google import GoogleTextEmbedderDescriptor

        desc = GoogleTextEmbedderDescriptor(
            model_name="text-embedding-004",
        )
        assert desc.get_provider() == "google"
        assert desc.get_model() == "text-embedding-004"
        dims = desc.get_dimensions()
        assert dims.size == 768

    def test_embedder_descriptor_custom_dims(self):
        """GoogleTextEmbedderDescriptor supports custom dimensions."""
        from vane.ai.providers.google import GoogleTextEmbedderDescriptor

        desc = GoogleTextEmbedderDescriptor(
            model_name="text-embedding-004",
            dimensions=256,
        )
        dims = desc.get_dimensions()
        assert dims.size == 256

    def test_embedder_descriptor_pickle(self):
        """GoogleTextEmbedderDescriptor survives pickle."""
        from vane.ai.providers.google import GoogleTextEmbedderDescriptor

        desc = GoogleTextEmbedderDescriptor(
            provider_options={"api_key": "test"},
            model_name="text-embedding-004",
            dimensions=256,
        )
        restored = pickle.loads(pickle.dumps(desc))
        assert restored.model_name == desc.model_name
        assert restored.dimensions == desc.dimensions

    def test_prompter_descriptor_creates(self):
        """GooglePrompterDescriptor can be created."""
        from vane.ai.providers.google import GooglePrompterDescriptor

        desc = GooglePrompterDescriptor(
            model_name="gemini-2.0-flash",
            system_message="Be helpful.",
        )
        assert desc.get_provider() == "google"
        assert desc.get_model() == "gemini-2.0-flash"

    def test_prompter_descriptor_pickle(self):
        """GooglePrompterDescriptor survives pickle."""
        from vane.ai.providers.google import GooglePrompterDescriptor

        desc = GooglePrompterDescriptor(
            provider_options={"api_key": "test"},
            model_name="gemini-2.0-flash",
            system_message="Be concise.",
            prompt_options={"temperature": 0.5},
        )
        restored = pickle.loads(pickle.dumps(desc))
        assert restored.model_name == desc.model_name
        assert restored.system_message == desc.system_message

    def test_prompter_udf_options(self):
        """Google prompter descriptor produces correct UDFOptions."""
        from vane.ai.providers.google import GooglePrompterDescriptor

        desc = GooglePrompterDescriptor()
        opts = desc.get_udf_options()
        assert opts.max_api_concurrency == 16

    def test_provider_get_prompter(self):
        """GoogleProvider.get_prompter returns descriptor."""
        from vane.ai.providers.google import GooglePrompterDescriptor, GoogleProvider

        provider = GoogleProvider(api_key="test")
        desc = provider.get_prompter(
            model="gemini-2.0-flash",
            system_message="Summarize.",
        )
        assert isinstance(desc, GooglePrompterDescriptor)
        assert desc.model_name == "gemini-2.0-flash"

    def test_provider_get_prompter_splits_call_client_options(self):
        """Google prompt call-level client options go to provider_options only."""
        from vane.ai._redaction import Secret
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="ctor-key")
        desc = provider.get_prompter(
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
            model="text-embedding-004",
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
            api_key="call-key",
            task_type="RETRIEVAL_QUERY",
            on_error="log",
        )

        # Credentials are sealed on the descriptor (vane#105).
        assert desc.provider_options == {"api_key": Secret("call-key")}
        assert "api_key" not in desc.embed_options
        assert desc.embed_options["task_type"] == "RETRIEVAL_QUERY"
        assert desc.embed_options["on_error"] == "log"


# ---------------------------------------------------------------------------
# Anthropic Structured Output + Multimodal tests
# ---------------------------------------------------------------------------


class TestAnthropicStructuredOutput:
    """Tests for Anthropic structured output via tool_use."""

    def test_descriptor_has_return_format(self):
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        desc = AnthropicPrompterDescriptor(return_format=dict)
        assert desc.return_format is dict

    def test_descriptor_default_no_return_format(self):
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        desc = AnthropicPrompterDescriptor()
        assert desc.return_format is None

    def test_descriptor_pickle_with_return_format(self):
        from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

        desc = AnthropicPrompterDescriptor(return_format=dict)
        restored = pickle.loads(pickle.dumps(desc))
        assert restored.return_format is dict

    def test_provider_passes_return_format(self):
        from vane.ai.providers.anthropic import AnthropicProvider

        prov = AnthropicProvider(api_key="test")
        desc = prov.get_prompter(return_format=dict)
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

        mock_response = MagicMock()
        mock_response.content = [tool_block]

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
        text_block.text = "Hello world"
        mock_response = MagicMock()
        mock_response.content = [text_block]

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
        text_block.text = "I see an image"
        mock_response = MagicMock()
        mock_response.content = [text_block]

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
        text_block.text = "ok"
        mock_response = MagicMock()
        mock_response.content = [text_block]

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

        desc = GooglePrompterDescriptor(return_format=dict)
        assert desc.return_format is dict

    def test_descriptor_default_no_return_format(self):
        from vane.ai.providers.google import GooglePrompterDescriptor

        desc = GooglePrompterDescriptor()
        assert desc.return_format is None

    def test_descriptor_pickle_with_return_format(self):
        from vane.ai.providers.google import GooglePrompterDescriptor

        desc = GooglePrompterDescriptor(return_format=dict)
        restored = pickle.loads(pickle.dumps(desc))
        assert restored.return_format is dict

    def test_provider_passes_return_format(self):
        from vane.ai.providers.google import GoogleProvider

        prov = GoogleProvider(api_key="test")
        desc = prov.get_prompter(return_format=dict)
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
        batch = _EmbedTextBatch(desc, "text", "embedding", max_chunk_chars=50, chunk_overlap_chars=10)

        # Short text (no chunking) + long text (will be chunked)
        table = pa.table({"text": ["short", "a" * 200]})
        result = batch(table)

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
        batch = _EmbedTextBatch(desc, "text", "embedding")  # no chunking

        table = pa.table({"text": ["a" * 5000]})
        result = batch(table)

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
        batch = _EmbedTextBatch(desc, "text", "embedding", max_chunk_chars=500, chunk_overlap_chars=50)
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

        conn = vane.connect()
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
        result = batch(table)

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
        result = batch(table)

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
        result = batch(table)
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
        result = batch(table)

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
        batch(table)

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
        batch(table)

        assert len(captured_messages[0]) == 3  # text + 2 images

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
        result = batch(table)

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
            for _t in texts:
                emb = MagicMock()
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
            for _ in texts:
                emb = MagicMock()
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
            for _ in texts:
                emb = MagicMock()
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
        """_retry_call correctly handles sync functions returning awaitables."""
        from vane.ai.functions import _retry_call

        async def async_fn():
            return "async_result"

        result = _retry_call(async_fn, max_retries=0, on_error="raise")
        assert result == "async_result"

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

    def test_embed_retry_success(self):
        from vane.ai.functions import _EmbedTextBatch

        calls = []

        def embed(texts):
            calls.append(1)
            if len(calls) < 2:
                raise RuntimeError("API error")
            return [np.array([1.0, 2.0, 3.0])] * len(texts)

        desc = self._make_embed_descriptor(embed)
        wrapper = _EmbedTextBatch(desc, "text", "emb", max_retries=3, on_error="raise")
        table = pa.table({"text": ["hello"]})
        result = wrapper(table)
        assert result.column("emb").length() == 1
        assert len(calls) == 2

    def test_embed_on_error_ignore(self):
        from vane.ai.functions import _EmbedTextBatch

        def embed(_texts):
            raise RuntimeError("permanent failure")

        desc = self._make_embed_descriptor(embed)
        wrapper = _EmbedTextBatch(desc, "text", "emb", max_retries=0, on_error="ignore")
        table = pa.table({"text": ["hello"]})
        result = wrapper(table)
        # Should return zero embeddings
        assert result.column("emb").length() == 1

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
        result = wrapper(table)
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
        result = wrapper(table)
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
        result = wrapper(table)
        assert result.column("response").to_pylist() == ["ok", "ok"]
        assert len(calls) == 2
