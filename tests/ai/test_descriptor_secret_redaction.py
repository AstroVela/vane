# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Provider descriptors seal credentials at construction and unseal only at execution.

Covers vane#105: descriptors must never expose API keys through repr, str,
logging, exception rendering, or pickled copies, while ``instantiate()`` (and
the OpenAI dimension probe / local engine construction) still hands the real
plaintext to the SDK client or engine.
"""

from __future__ import annotations

import logging
import pickle
import sys
import traceback
import types
from types import SimpleNamespace

import pytest

from vane.ai._redaction import REDACTED_PLACEHOLDER, Secret

API_KEY = "sk-PLAINTEXT-API-KEY-SENTINEL-0123456789"
ORGANIZATION = "org-PLAINTEXT-ORG-SENTINEL-0123456789"
HUB_TOKEN = "hf_PLAINTEXT-HUB-TOKEN-SENTINEL-0123456789"
ALL_SENTINELS = (API_KEY, ORGANIZATION, HUB_TOKEN)


def _assert_no_plaintext(rendered: str) -> None:
    for sentinel in ALL_SENTINELS:
        assert sentinel not in rendered


# ---------------------------------------------------------------------------
# Descriptor factories
# ---------------------------------------------------------------------------


def _openai_embedder_descriptor():
    from vane.ai.providers.openai import OpenAITextEmbedderDescriptor

    return OpenAITextEmbedderDescriptor(
        provider_options={"api_key": API_KEY, "organization": ORGANIZATION, "base_url": "https://api.example"},
        model_name="text-embedding-3-small",
        dimensions=512,
        embed_options={"batch_size": 32, "auth_token": API_KEY},
    )


def _openai_prompter_descriptor():
    from vane.ai.providers.openai import OpenAIPrompterDescriptor

    return OpenAIPrompterDescriptor(
        provider_options={"api_key": API_KEY, "organization": ORGANIZATION},
        model_name="gpt-4o-mini",
        prompt_options={"temperature": 0.5, "auth_token": API_KEY},
    )


def _anthropic_prompter_descriptor():
    from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

    return AnthropicPrompterDescriptor(
        provider_options={"api_key": API_KEY, "base_url": "https://api.example"},
        model_name="claude-sonnet-4-20250514",
        prompt_options={"temperature": 0.2, "auth_token": API_KEY},
    )


def _google_embedder_descriptor():
    from vane.ai.providers.google import GoogleTextEmbedderDescriptor

    return GoogleTextEmbedderDescriptor(
        provider_options={"api_key": API_KEY},
        model_name="text-embedding-004",
        embed_options={"task_type": "RETRIEVAL_QUERY", "auth_token": API_KEY},
    )


def _google_prompter_descriptor():
    from vane.ai.providers.google import GooglePrompterDescriptor

    return GooglePrompterDescriptor(
        provider_options={"api_key": API_KEY},
        model_name="gemini-2.0-flash",
        prompt_options={"temperature": 0.1, "auth_token": API_KEY},
    )


def _vllm_prompter_descriptor():
    from vane.ai.providers.vllm import VLLMPrompterDescriptor

    return VLLMPrompterDescriptor(
        model_name="Qwen/Qwen3-1.7B",
        vllm_options={
            "engine_args": {"hf_token": HUB_TOKEN, "max_model_len": 2048},
            "generate_args": {"sampling_params": {"max_tokens": 64}, "api_key": API_KEY},
            "gpus_per_actor": 1,
        },
    )


def _transformers_embedder_descriptor():
    from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

    return TransformersTextEmbedderDescriptor(
        model="sentence-transformers/all-MiniLM-L6-v2",
        embed_options={"batch_size": 8, "revision": "pinned", "token": HUB_TOKEN},
    )


def _transformers_classifier_descriptor():
    from vane.ai.providers.transformers import TransformersTextClassifierDescriptor

    return TransformersTextClassifierDescriptor(
        model="facebook/bart-large-mnli",
        classify_options={"batch_size": 4, "token": HUB_TOKEN},
    )


ALL_DESCRIPTOR_FACTORIES = [
    pytest.param(_openai_embedder_descriptor, id="openai-embedder"),
    pytest.param(_openai_prompter_descriptor, id="openai-prompter"),
    pytest.param(_anthropic_prompter_descriptor, id="anthropic-prompter"),
    pytest.param(_google_embedder_descriptor, id="google-embedder"),
    pytest.param(_google_prompter_descriptor, id="google-prompter"),
    pytest.param(_vllm_prompter_descriptor, id="vllm-prompter"),
    pytest.param(_transformers_embedder_descriptor, id="transformers-embedder"),
    pytest.param(_transformers_classifier_descriptor, id="transformers-classifier"),
]


# ---------------------------------------------------------------------------
# Fake SDK modules
# ---------------------------------------------------------------------------


class _RecordingClient:
    """Records constructor kwargs; stands in for any SDK client class."""

    calls: list[dict] = []  # overridden per instance factory

    def __init__(self, **kwargs):
        type(self).calls.append(kwargs)


def _fresh_recording_client():
    return type("FakeClient", (_RecordingClient,), {"calls": []})


def _install_fake_openai(monkeypatch, async_client, sync_client=None):
    module = SimpleNamespace(
        AsyncOpenAI=async_client,
        OpenAI=sync_client or _fresh_recording_client(),
        OpenAIError=Exception,
    )
    monkeypatch.setitem(sys.modules, "openai", module)
    return module


def _install_fake_anthropic(monkeypatch, async_client):
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(AsyncAnthropic=async_client))


def _install_fake_google(monkeypatch, client):
    fake_genai = SimpleNamespace(Client=client)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)


def _install_fake_vllm_engine(monkeypatch, captured):
    def fake_build_executor(model, options):
        captured.append((model, options))
        return SimpleNamespace()

    vllm_module = types.ModuleType("vane.execution.vllm")
    vllm_module.build_executor = fake_build_executor
    execution_module = types.ModuleType("vane.execution")
    execution_module.vllm = vllm_module
    monkeypatch.setitem(sys.modules, "vane.execution", execution_module)
    monkeypatch.setitem(sys.modules, "vane.execution.vllm", vllm_module)


# ---------------------------------------------------------------------------
# repr / str redaction
# ---------------------------------------------------------------------------


class TestDescriptorReprRedaction:
    @pytest.mark.parametrize("factory", ALL_DESCRIPTOR_FACTORIES)
    def test_repr_and_str_contain_no_plaintext(self, factory):
        descriptor = factory()
        for rendered in (repr(descriptor), str(descriptor), f"{descriptor}", "{!r}".format(descriptor)):
            _assert_no_plaintext(rendered)
            assert REDACTED_PLACEHOLDER in rendered

    @pytest.mark.parametrize("factory", ALL_DESCRIPTOR_FACTORIES)
    def test_repr_keeps_non_sensitive_fields_readable(self, factory):
        descriptor = factory()
        assert descriptor.get_model() in repr(descriptor)

    def test_openai_organization_is_redacted(self):
        descriptor = _openai_prompter_descriptor()
        assert ORGANIZATION not in repr(descriptor)
        assert ORGANIZATION not in str(descriptor)

    def test_openai_nested_organization_header_is_redacted(self):
        from vane.ai.providers.openai import OpenAIProvider

        descriptor = OpenAIProvider(api_key=API_KEY).get_prompter(
            extra_headers={"OpenAI-Organization": ORGANIZATION},
        )
        assert ORGANIZATION not in repr(descriptor)
        assert isinstance(descriptor.prompt_options["extra_headers"]["OpenAI-Organization"], Secret)

    def test_credential_kwarg_landing_in_prompt_options_is_redacted(self):
        from vane.ai.providers.openai import OpenAIProvider

        descriptor = OpenAIProvider(api_key=API_KEY).get_prompter(auth_token=API_KEY, temperature=0.3)
        assert API_KEY not in repr(descriptor)
        assert isinstance(descriptor.prompt_options["auth_token"], Secret)
        assert descriptor.prompt_options["temperature"] == 0.3

    def test_credential_kwarg_landing_in_embed_options_is_redacted(self):
        from vane.ai.providers.google import GoogleProvider

        descriptor = GoogleProvider(api_key=API_KEY).get_text_embedder(auth_token=API_KEY, task_type="RETRIEVAL_QUERY")
        assert API_KEY not in repr(descriptor)
        assert isinstance(descriptor.embed_options["auth_token"], Secret)
        assert descriptor.embed_options["task_type"] == "RETRIEVAL_QUERY"

    def test_vllm_nested_engine_and_generate_args_are_redacted(self):
        descriptor = _vllm_prompter_descriptor()
        rendered = repr(descriptor)
        _assert_no_plaintext(rendered)
        assert "2048" in rendered  # non-sensitive engine arg stays readable
        assert isinstance(descriptor.vllm_options["engine_args"]["hf_token"], Secret)
        assert isinstance(descriptor.vllm_options["generate_args"]["api_key"], Secret)

    def test_transformers_hub_token_is_redacted(self):
        descriptor = _transformers_embedder_descriptor()
        assert HUB_TOKEN not in repr(descriptor)
        assert isinstance(descriptor.embed_options["token"], Secret)
        assert descriptor.embed_options["revision"] == "pinned"


# ---------------------------------------------------------------------------
# get_options() stays wrapped
# ---------------------------------------------------------------------------


class TestGetOptionsStaysWrapped:
    @pytest.mark.parametrize("factory", ALL_DESCRIPTOR_FACTORIES)
    def test_get_options_repr_has_no_plaintext(self, factory):
        options = factory().get_options()
        _assert_no_plaintext(repr(options))

    def test_openai_prompter_get_options_holds_secret(self):
        options = _openai_prompter_descriptor().get_options()
        assert isinstance(options["auth_token"], Secret)
        assert options["auth_token"].reveal() == API_KEY

    def test_vllm_get_options_holds_nested_secret(self):
        options = _vllm_prompter_descriptor().get_options()
        assert isinstance(options["engine_args"]["hf_token"], Secret)
        assert options["engine_args"]["hf_token"].reveal() == HUB_TOKEN


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLoggingRedaction:
    @pytest.mark.parametrize("factory", ALL_DESCRIPTOR_FACTORIES)
    def test_percent_s_and_percent_r_logging_emit_no_plaintext(self, factory, caplog):
        descriptor = factory()
        logger = logging.getLogger("vane.test.descriptor_redaction")
        with caplog.at_level(logging.INFO, logger=logger.name):
            logger.info("descriptor is %s", descriptor)
            logger.info("descriptor is %r", descriptor)
        assert len(caplog.records) == 2
        _assert_no_plaintext(caplog.text)
        assert REDACTED_PLACEHOLDER in caplog.text


# ---------------------------------------------------------------------------
# Exceptions during client construction
# ---------------------------------------------------------------------------


class TestExceptionRedaction:
    def _assert_exception_clean(self, excinfo):
        assert not any(sentinel in str(excinfo.value) for sentinel in ALL_SENTINELS)
        rendered = "".join(traceback.format_exception(excinfo.value))
        _assert_no_plaintext(rendered)

    def test_openai_client_construction_failure_carries_no_plaintext(self, monkeypatch):
        class ExplodingClient:
            def __init__(self, **kwargs):
                raise RuntimeError("client construction failed")

        _install_fake_openai(monkeypatch, ExplodingClient)
        with pytest.raises(RuntimeError, match="client construction failed") as excinfo:
            _openai_embedder_descriptor().instantiate()
        self._assert_exception_clean(excinfo)

    def test_anthropic_client_construction_failure_carries_no_plaintext(self, monkeypatch):
        class ExplodingClient:
            def __init__(self, **kwargs):
                raise RuntimeError("client construction failed")

        _install_fake_anthropic(monkeypatch, ExplodingClient)
        with pytest.raises(RuntimeError, match="client construction failed") as excinfo:
            _anthropic_prompter_descriptor().instantiate()
        self._assert_exception_clean(excinfo)


# ---------------------------------------------------------------------------
# Plaintext restored exactly at the execution boundary
# ---------------------------------------------------------------------------


class TestUnwrapAtExecutionBoundary:
    def test_openai_embedder_client_receives_plaintext(self, monkeypatch):
        client = _fresh_recording_client()
        _install_fake_openai(monkeypatch, client)
        _openai_embedder_descriptor().instantiate()
        assert client.calls == [{"api_key": API_KEY, "organization": ORGANIZATION, "base_url": "https://api.example"}]

    def test_openai_prompter_client_receives_plaintext_and_options_are_unwrapped(self, monkeypatch):
        client = _fresh_recording_client()
        _install_fake_openai(monkeypatch, client)
        prompter = _openai_prompter_descriptor().instantiate()
        assert client.calls == [{"api_key": API_KEY, "organization": ORGANIZATION}]
        # prompt-time options forwarded to the SDK must be plain again
        assert prompter._options == {"temperature": 0.5, "auth_token": API_KEY}

    def test_openai_get_dimensions_probe_receives_plaintext(self, monkeypatch):
        from vane.ai.providers.openai import OpenAITextEmbedderDescriptor

        probe_calls = []

        class FakeProbeClient:
            def __init__(self, **kwargs):
                probe_calls.append(kwargs)
                self.embeddings = SimpleNamespace(
                    create=lambda **_: SimpleNamespace(data=[SimpleNamespace(embedding=[0.0] * 7)])
                )

        _install_fake_openai(monkeypatch, _fresh_recording_client(), sync_client=FakeProbeClient)
        descriptor = OpenAITextEmbedderDescriptor(
            provider_options={"api_key": API_KEY, "base_url": "https://api.example"},
            model_name="custom-served-model",
        )
        assert descriptor.get_dimensions().size == 7
        assert probe_calls == [{"api_key": API_KEY, "base_url": "https://api.example"}]

    def test_anthropic_client_receives_plaintext(self, monkeypatch):
        client = _fresh_recording_client()
        _install_fake_anthropic(monkeypatch, client)
        _anthropic_prompter_descriptor().instantiate()
        assert client.calls == [{"api_key": API_KEY, "base_url": "https://api.example"}]

    def test_google_embedder_client_receives_plaintext(self, monkeypatch):
        client = _fresh_recording_client()
        _install_fake_google(monkeypatch, client)
        _google_embedder_descriptor().instantiate()
        assert client.calls == [{"api_key": API_KEY}]

    def test_google_prompter_client_receives_plaintext(self, monkeypatch):
        client = _fresh_recording_client()
        _install_fake_google(monkeypatch, client)
        _google_prompter_descriptor().instantiate()
        assert client.calls == [{"api_key": API_KEY}]

    def test_vllm_engine_receives_plaintext_nested_args(self, monkeypatch):
        captured = []
        _install_fake_vllm_engine(monkeypatch, captured)
        prompter = _vllm_prompter_descriptor().instantiate()
        prompter._ensure_executor()
        assert len(captured) == 1
        model, options = captured[0]
        assert model == "Qwen/Qwen3-1.7B"
        assert options["engine_args"] == {"hf_token": HUB_TOKEN, "max_model_len": 2048}
        assert options["generate_args"]["api_key"] == API_KEY
        assert options["generate_args"]["sampling_params"] == {"max_tokens": 64}
        assert options["use_threading"] is True

    def test_transformers_get_dimensions_receives_plaintext_token(self, monkeypatch):
        auto_config_calls = []

        class FakeAutoConfig:
            @staticmethod
            def from_pretrained(model, **options):
                auto_config_calls.append((model, options))
                return SimpleNamespace(hidden_size=384)

        monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoConfig=FakeAutoConfig))
        descriptor = _transformers_embedder_descriptor()
        assert descriptor.get_dimensions().size == 384
        assert auto_config_calls == [
            (
                "sentence-transformers/all-MiniLM-L6-v2",
                {"trust_remote_code": False, "revision": "pinned", "token": HUB_TOKEN},
            )
        ]

    def test_transformers_embedder_model_receives_plaintext_token(self, monkeypatch):
        calls = []

        class FakeSentenceTransformer:
            def __init__(self, model, **options):
                calls.append((model, options))

            def eval(self):
                return self

        monkeypatch.setitem(
            sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        )
        _transformers_embedder_descriptor().instantiate()
        assert calls == [
            (
                "sentence-transformers/all-MiniLM-L6-v2",
                {"trust_remote_code": False, "backend": "torch", "revision": "pinned", "token": HUB_TOKEN},
            )
        ]

    def test_transformers_classifier_pipeline_receives_plaintext_token(self, monkeypatch):
        pipeline_calls = []

        def fake_pipeline(task, **options):
            pipeline_calls.append((task, options))
            return object()

        monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=fake_pipeline))
        _transformers_classifier_descriptor().instantiate()
        assert pipeline_calls == [
            (
                "zero-shot-classification",
                {"model": "facebook/bart-large-mnli", "trust_remote_code": False, "token": HUB_TOKEN},
            )
        ]

    def test_runtime_classes_accept_plain_dicts_unchanged(self, monkeypatch):
        """Directly-constructed runtime objects with plain dicts keep working."""
        from vane.ai.providers.openai import OpenAITextEmbedder

        client = _fresh_recording_client()
        _install_fake_openai(monkeypatch, client)
        OpenAITextEmbedder(provider_options={"api_key": "plain-key"}, model="text-embedding-3-small")
        assert client.calls == [{"api_key": "plain-key"}]


# ---------------------------------------------------------------------------
# Pickle round-trips
# ---------------------------------------------------------------------------


class TestPickleRoundTrip:
    @pytest.mark.parametrize("factory", ALL_DESCRIPTOR_FACTORIES)
    def test_repr_stays_redacted_after_pickle(self, factory):
        restored = pickle.loads(pickle.dumps(factory()))
        _assert_no_plaintext(repr(restored))
        assert REDACTED_PLACEHOLDER in repr(restored)

    def test_openai_pickled_descriptor_still_builds_working_client(self, monkeypatch):
        client = _fresh_recording_client()
        _install_fake_openai(monkeypatch, client)
        restored = pickle.loads(pickle.dumps(_openai_embedder_descriptor()))
        restored.instantiate()
        assert client.calls == [{"api_key": API_KEY, "organization": ORGANIZATION, "base_url": "https://api.example"}]

    def test_anthropic_pickled_descriptor_still_builds_working_client(self, monkeypatch):
        client = _fresh_recording_client()
        _install_fake_anthropic(monkeypatch, client)
        restored = pickle.loads(pickle.dumps(_anthropic_prompter_descriptor()))
        restored.instantiate()
        assert client.calls == [{"api_key": API_KEY, "base_url": "https://api.example"}]

    def test_google_pickled_descriptor_still_builds_working_client(self, monkeypatch):
        client = _fresh_recording_client()
        _install_fake_google(monkeypatch, client)
        restored = pickle.loads(pickle.dumps(_google_prompter_descriptor()))
        restored.instantiate()
        assert client.calls == [{"api_key": API_KEY}]

    def test_vllm_pickled_descriptor_still_builds_engine_with_plaintext(self, monkeypatch):
        captured = []
        _install_fake_vllm_engine(monkeypatch, captured)
        restored = pickle.loads(pickle.dumps(_vllm_prompter_descriptor()))
        restored.instantiate()._ensure_executor()
        _model, options = captured[0]
        assert options["engine_args"]["hf_token"] == HUB_TOKEN
