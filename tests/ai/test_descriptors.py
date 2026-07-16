"""Tests for vane.ai descriptor serialization and provider loading."""

from __future__ import annotations

import base64
import pickle

import numpy as np
import pyarrow as pa
import pytest

# ---------------------------------------------------------------------------
# Provider loading
# ---------------------------------------------------------------------------


class TestProviderLoading:
    def test_load_unknown_provider_raises(self):
        from vane.ai.provider import load_provider

        with pytest.raises(ValueError, match="not supported"):
            load_provider("nonexistent")

    def test_load_transformers_provider(self):
        """TransformersProvider can be instantiated (deps mocked if needed)."""
        from vane.ai.providers.transformers import TransformersProvider

        provider = TransformersProvider()
        assert provider.name == "transformers"

    def test_transformers_provider_merges_constructor_options(self):
        from vane.ai.providers.transformers import TransformersProvider

        provider = TransformersProvider(batch_size=16, max_retries=2)

        embedder = provider.get_text_embedder(max_retries=4)
        classifier = provider.get_text_classifier(on_error="ignore")

        assert embedder.embed_options == {"batch_size": 16, "max_retries": 4}
        assert classifier.classify_options == {"batch_size": 16, "max_retries": 2, "on_error": "ignore"}

    def test_load_openai_provider(self):
        from vane.ai.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        assert provider.name == "openai"

    def test_provider_registry_contains_expected(self):
        from vane.ai.provider import PROVIDERS

        assert "transformers" in PROVIDERS
        assert "openai" in PROVIDERS


# ---------------------------------------------------------------------------
# Descriptor serialization (pickle round-trip)
# ---------------------------------------------------------------------------


class TestTransformersDescriptorPickle:
    def test_text_embedder_descriptor_roundtrip(self):
        from vane.ai.providers.transformers import (
            TransformersTextEmbedderDescriptor,
        )

        desc = TransformersTextEmbedderDescriptor(
            model="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=128,
            embed_options={"batch_size": 32},
        )

        # Pickle round-trip
        data = pickle.dumps(desc)
        restored = pickle.loads(data)

        assert restored.model == desc.model
        assert restored.dimensions == desc.dimensions
        assert restored.embed_options == desc.embed_options
        assert restored.get_provider() == "transformers"
        assert restored.get_model() == "sentence-transformers/all-MiniLM-L6-v2"

    def test_text_classifier_descriptor_roundtrip(self):
        from vane.ai.providers.transformers import (
            TransformersTextClassifierDescriptor,
        )

        desc = TransformersTextClassifierDescriptor(
            model="facebook/bart-large-mnli",
            classify_options={"max_retries": 5},
        )

        data = pickle.dumps(desc)
        restored = pickle.loads(data)

        assert restored.model == desc.model
        assert restored.get_provider() == "transformers"


class TestOpenAIDescriptorPickle:
    def test_text_embedder_descriptor_roundtrip(self):
        from vane.ai.providers.openai import OpenAITextEmbedderDescriptor

        desc = OpenAITextEmbedderDescriptor(
            provider_name="openai",
            provider_options={"api_key": "test-key"},
            model_name="text-embedding-3-small",
            dimensions=512,
            embed_options={"batch_size": 32},
        )

        data = pickle.dumps(desc)
        restored = pickle.loads(data)

        assert restored.model_name == "text-embedding-3-small"
        assert restored.dimensions == 512
        assert restored.provider_options == {"api_key": "test-key"}
        assert restored.get_provider() == "openai"
        assert restored.is_async() is True

    def test_prompter_descriptor_roundtrip(self):
        from vane.ai.providers.openai import OpenAIPrompterDescriptor

        desc = OpenAIPrompterDescriptor(
            model_name="gpt-4o",
            system_message="You are a helpful assistant.",
            prompt_options={"temperature": 0.7},
        )

        data = pickle.dumps(desc)
        restored = pickle.loads(data)

        assert restored.model_name == "gpt-4o"
        assert restored.system_message == "You are a helpful assistant."
        assert restored.prompt_options == {"temperature": 0.7}

    def test_dimension_override_validation(self):
        from vane.ai.providers.openai import OpenAITextEmbedderDescriptor

        # ada-002 does not support custom dimensions
        with pytest.raises(ValueError, match="does not support custom dimensions"):
            OpenAITextEmbedderDescriptor(
                model_name="text-embedding-ada-002",
                dimensions=512,
            )

    def test_openai_embedding_base64_decodes_float32_vector(self):
        from vane.ai.providers.openai import _decode_openai_embedding_base64

        raw = np.array([1.5, -2.0, 0.25], dtype="<f4")
        encoded = base64.b64encode(raw.tobytes()).decode("ascii")

        decoded = _decode_openai_embedding_base64(encoded)

        assert decoded.dtype == np.float32
        assert decoded.tolist() == [1.5, -2.0, 0.25]


# ---------------------------------------------------------------------------
# Descriptor API contracts
# ---------------------------------------------------------------------------


class TestDescriptorAPI:
    def test_udf_options_from_transformers(self):
        from vane.ai.providers.transformers import (
            TransformersTextEmbedderDescriptor,
        )

        desc = TransformersTextEmbedderDescriptor(
            model="test-model",
            embed_options={"batch_size": 16, "max_retries": 5},
        )
        opts = desc.get_udf_options()
        assert opts.batch_size == 16
        assert opts.max_retries == 5

    def test_udf_options_from_openai(self):
        from vane.ai.providers.openai import OpenAITextEmbedderDescriptor

        desc = OpenAITextEmbedderDescriptor(
            model_name="text-embedding-3-small",
            embed_options={"batch_size": 128},
        )
        opts = desc.get_udf_options()
        assert opts.batch_size == 128
        assert opts.max_retries == 0  # OpenAI client retries internally

    def test_embedding_dimensions_arrow_type(self):
        from vane.ai.typing import EmbeddingDimensions

        dims = EmbeddingDimensions(size=384, dtype=pa.float32())
        arrow_type = dims.as_arrow_type()
        assert isinstance(arrow_type, pa.DataType)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class TestProtocols:
    def test_text_embedder_protocol_check(self):
        from vane.ai.protocols import TextEmbedder

        class MyEmbedder:
            def embed_text(self, text: list[str]) -> list:
                return [[] for _ in text]

        assert isinstance(MyEmbedder(), TextEmbedder)

    def test_text_classifier_protocol_check(self):
        from vane.ai.protocols import TextClassifier

        class MyClassifier:
            def classify_text(self, text, _labels):
                return ["pos" for _ in text]

        assert isinstance(MyClassifier(), TextClassifier)

    def test_prompter_protocol_check(self):
        from vane.ai.protocols import Prompter

        class MyPrompter:
            async def prompt(self, _messages):
                return "response"

        assert isinstance(MyPrompter(), Prompter)
