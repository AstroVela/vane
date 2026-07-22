# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

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
        from vane.ai._redaction import Secret
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
        # Credentials stay sealed on the descriptor (vane#105); the round-trip
        # preserves the real value, comparable only against another Secret.
        assert restored.provider_options == {"api_key": Secret("test-key")}
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


class TestGoogleProviderDefaults:
    """Google defaults must be current (non-shut-down) models (vane#141)."""

    def test_default_embed_model_flows_through(self):
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test")
        desc = provider.get_text_embedder()

        # gemini-embedding-001 replaces text-embedding-004 (shut down 2026-01-14).
        # It is stable, embeds each input separately (one vector per row), and
        # supports task_type, unlike gemini-embedding-2.
        assert desc.get_model() == "gemini-embedding-001"
        # Default output dimensionality documented at 3072.
        assert desc.get_dimensions().size == 3072

    def test_default_prompt_model_flows_through(self):
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key="test")
        desc = provider.get_prompter()

        # gemini-3.6-flash replaces gemini-2.0-flash (shut down 2026-06-01).
        assert desc.get_model() == "gemini-3.6-flash"

    def test_descriptor_dataclass_defaults_match_provider_defaults(self):
        from vane.ai.providers.google import (
            GooglePrompterDescriptor,
            GoogleProvider,
            GoogleTextEmbedderDescriptor,
        )

        assert GoogleTextEmbedderDescriptor().get_model() == GoogleProvider.DEFAULT_EMBED_MODEL
        assert GooglePrompterDescriptor().get_model() == GoogleProvider.DEFAULT_PROMPT_MODEL


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
