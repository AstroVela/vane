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

    def test_transformers_provider_accepts_only_call_level_embed_options(self):
        from vane.ai.providers.transformers import TransformersProvider

        provider = TransformersProvider()
        embedder = provider.get_text_embedder(options={"revision": "pinned-revision", "device": "cpu"})

        assert embedder.options == {"revision": "pinned-revision", "device": "cpu"}

        with pytest.raises(TypeError, match="batch_size"):
            provider.get_text_embedder(options={"batch_size": 16})

    @pytest.mark.parametrize(
        ("provider", "legacy_option"),
        [
            ("openai", {"base_url": "https://example.test/v1"}),
            ("google", {"api_key": "secret"}),
            ("anthropic", {"max_tokens": 64}),
            ("transformers", {"revision": "pinned"}),
        ],
    )
    def test_builtin_provider_constructors_reject_legacy_execution_options(self, provider, legacy_option):
        from vane.ai.providers.anthropic import AnthropicProvider
        from vane.ai.providers.google import GoogleProvider
        from vane.ai.providers.openai import OpenAIProvider
        from vane.ai.providers.transformers import TransformersProvider

        providers = {
            "openai": OpenAIProvider,
            "google": GoogleProvider,
            "anthropic": AnthropicProvider,
            "transformers": TransformersProvider,
        }
        with pytest.raises(TypeError):
            providers[provider](**legacy_option)

    def test_load_openai_provider(self):
        from vane.ai.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        assert provider.name == "openai"

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            pytest.param({"use_ray": True}, "Unsupported Prompt option", id="runtime-key"),
            pytest.param(
                {"generate_args": {"sampling_params": {"structured_outputs": {"json": {}}}}},
                "cannot configure structured_outputs directly",
                id="structured-output-override",
            ),
        ],
    )
    def test_vllm_provider_rejects_options_outside_closed_prompt_contract(self, options, message):
        from vane.ai.providers.vllm import VLLMProvider

        with pytest.raises((TypeError, ValueError), match=message):
            VLLMProvider().get_prompter(options=options)

    @pytest.mark.parametrize(
        ("model", "options", "message"),
        [
            ("o3-pro", {"use_chat_completions": True}, "Responses API only"),
            ("gpt-4o-search-preview", {}, "Chat Completions only"),
        ],
    )
    def test_openai_provider_factory_rejects_known_api_path_conflicts(self, model, options, message):
        from vane.ai.providers.openai import OpenAIProvider

        with pytest.raises(ValueError, match=message):
            OpenAIProvider().get_prompter(model=model, options=options)

    @pytest.mark.parametrize(
        ("operation", "options"),
        [
            ("embed", {"task_type": "BOGUS"}),
            ("embed", {"task_type": "RETRIEVAL_DOCUMENT", "title": 123}),
            ("prompt", {"temperature": "hot"}),
            ("prompt", {"top_p": 2}),
        ],
    )
    def test_google_provider_factories_apply_closed_option_value_validation(self, operation, options):
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider()
        with pytest.raises(ValueError):
            if operation == "embed":
                provider.get_text_embedder(model="gemini-embedding-001", options=options)
            else:
                provider.get_prompter(model="gemini-test", options=options)

    def test_known_models_cannot_be_used_with_the_wrong_operation(self):
        from vane.ai.providers.google import GoogleProvider
        from vane.ai.providers.openai import OpenAIProvider

        factories = [
            lambda: OpenAIProvider().get_prompter(model="text-embedding-3-small"),
            lambda: OpenAIProvider().get_text_embedder(model="gpt-4o", dimensions=8),
            lambda: GoogleProvider().get_prompter(model="gemini-embedding-2"),
            lambda: GoogleProvider().get_text_embedder(model="gemini-3.6-flash", dimensions=8),
        ]

        for factory in factories:
            with pytest.raises(ValueError, match="supports (Embed|Prompt), not (Prompt|Embed)"):
                factory()

    @pytest.mark.parametrize(
        ("model", "dimensions"),
        [
            ("text-embedding-ada-002", 256),
            ("text-embedding-3-small", 2048),
        ],
    )
    def test_openai_compatible_endpoint_does_not_inherit_official_dimension_limits(self, model, dimensions):
        from vane.ai.providers.openai import OpenAIProvider

        descriptor = OpenAIProvider().get_text_embedder(
            model=model,
            dimensions=dimensions,
            options={"base_url": "https://compatible.example.test/v1"},
        )

        assert descriptor.get_dimensions().size == dimensions

    def test_openai_compatible_endpoint_requires_explicit_dimensions(self):
        from vane.ai.providers.openai import OpenAIProvider

        with pytest.raises(ValueError, match="pass dimensions"):
            OpenAIProvider().get_text_embedder(
                model="text-embedding-3-small",
                options={"base_url": "https://compatible.example.test/v1"},
            )

    def test_openai_compatible_endpoint_can_remap_known_model_ids(self):
        from vane.ai.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        assert (
            provider.get_prompter(
                model="text-embedding-3-small",
                options={"base_url": "https://compatible.example.test/v1"},
            )
            is not None
        )
        assert (
            provider.get_text_embedder(
                model="gpt-4o",
                dimensions=8,
                options={"base_url": "https://compatible.example.test/v1"},
            )
            is not None
        )

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
            options={"revision": "pinned-revision"},
        )

        # Pickle round-trip
        data = pickle.dumps(desc)
        restored = pickle.loads(data)

        assert restored.model == desc.model
        assert restored.dimensions == desc.dimensions
        assert restored.options == desc.options
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
            model_name="text-embedding-3-small",
            dimensions=512,
            options={"encoding_format": "base64"},
        )

        data = pickle.dumps(desc)
        restored = pickle.loads(data)

        assert restored.model_name == "text-embedding-3-small"
        assert restored.dimensions == 512
        assert restored.options == desc.options
        assert restored.get_provider() == "openai"
        assert restored.is_async() is True

    def test_prompter_descriptor_roundtrip(self):
        from vane.ai.providers.openai import OpenAIPrompterDescriptor

        desc = OpenAIPrompterDescriptor(
            model_name="gpt-4o",
            system_message="You are a helpful assistant.",
            options={"temperature": 0.7},
        )

        data = pickle.dumps(desc)
        restored = pickle.loads(data)

        assert restored.model_name == "gpt-4o"
        assert restored.system_message == "You are a helpful assistant."
        assert restored.options == {"temperature": 0.7}

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
    @pytest.mark.parametrize("options", [{}, {"device": None}])
    def test_transformers_missing_or_null_device_is_explicit_cpu(self, options):
        from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

        desc = TransformersTextEmbedderDescriptor(model="test-model", options=options)

        assert desc.options["device"] == "cpu"
        assert desc.get_udf_options().num_gpus == 0

    def test_transformers_explicit_cuda_device_reserves_gpu(self):
        from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

        desc = TransformersTextEmbedderDescriptor(model="test-model", options={"device": "cuda:0"})
        opts = desc.get_udf_options()

        assert opts.batch_size is None
        assert opts.num_gpus == 1

    def test_udf_options_from_openai(self):
        from vane.ai.providers.openai import OpenAITextEmbedderDescriptor

        desc = OpenAITextEmbedderDescriptor(
            model_name="text-embedding-3-small",
            options={"encoding_format": "float"},
        )
        opts = desc.get_udf_options()
        assert opts.batch_size is None
        assert opts.num_gpus == 0

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
