# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Provider descriptors reject inline credentials and keep option rendering safe."""

from __future__ import annotations

import logging
import pickle
import sys
import traceback
from types import SimpleNamespace

import pytest

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
        model_name="text-embedding-3-small",
        dimensions=512,
        options={"base_url": "https://api.example", "encoding_format": "base64"},
    )


def _openai_embedder_descriptor_with_options(options):
    from vane.ai.providers.openai import OpenAITextEmbedderDescriptor

    return OpenAITextEmbedderDescriptor(options=options)


def _openai_provider_embedder_descriptor(options):
    from vane.ai.providers.openai import OpenAIProvider

    return OpenAIProvider().get_text_embedder(options=options)


def _openai_provider_prompt_descriptor(options):
    from vane.ai.providers.openai import OpenAIProvider

    return OpenAIProvider().get_prompter(options=options)


def _google_embedder_descriptor():
    from vane.ai.providers.google import GoogleTextEmbedderDescriptor

    return GoogleTextEmbedderDescriptor(
        model_name="gemini-embedding-001",
        options={"task_type": "RETRIEVAL_QUERY"},
    )


def _transformers_embedder_descriptor():
    from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

    return TransformersTextEmbedderDescriptor(
        model="sentence-transformers/all-MiniLM-L6-v2",
        options={"revision": "pinned"},
    )


def _openai_prompt_descriptor(options):
    from vane.ai.providers.openai import OpenAIPrompterDescriptor

    return OpenAIPrompterDescriptor(options=options)


def _anthropic_prompt_descriptor(options):
    from vane.ai.providers.anthropic import AnthropicPrompterDescriptor

    return AnthropicPrompterDescriptor(model_name="claude-test", options=options)


def _anthropic_provider_prompt_descriptor(options):
    from vane.ai.providers.anthropic import AnthropicProvider

    return AnthropicProvider(prompt_model="claude-test").get_prompter(options=options)


def _google_prompt_descriptor(options):
    from vane.ai.providers.google import GooglePrompterDescriptor

    return GooglePrompterDescriptor(model_name="gemini-test", options=options)


def _vllm_prompt_plan(options):
    from vane.ai.providers.vllm import NativeVLLMPromptPlan

    return NativeVLLMPromptPlan(vllm_options=options)


ALL_DESCRIPTOR_FACTORIES = [
    pytest.param(_openai_embedder_descriptor, id="openai-embedder"),
    pytest.param(_google_embedder_descriptor, id="google-embedder"),
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


def _install_fake_google(monkeypatch, client):
    class HttpRetryOptions:
        def __init__(self, *, attempts):
            self.attempts = attempts

    class HttpOptions:
        def __init__(self, *, retry_options):
            self.retry_options = retry_options

    fake_genai = SimpleNamespace(
        Client=client,
        types=SimpleNamespace(HttpOptions=HttpOptions, HttpRetryOptions=HttpRetryOptions),
    )
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)


# ---------------------------------------------------------------------------
# repr / str redaction
# ---------------------------------------------------------------------------


class TestDescriptorReprRedaction:
    @pytest.mark.parametrize("factory", ALL_DESCRIPTOR_FACTORIES)
    def test_repr_and_str_contain_no_plaintext(self, factory):
        descriptor = factory()
        for rendered in (repr(descriptor), str(descriptor), f"{descriptor}", "{!r}".format(descriptor)):
            _assert_no_plaintext(rendered)

    @pytest.mark.parametrize("factory", ALL_DESCRIPTOR_FACTORIES)
    def test_repr_keeps_non_sensitive_fields_readable(self, factory):
        descriptor = factory()
        assert descriptor.get_model() in repr(descriptor)

    @pytest.mark.parametrize(
        ("factory", "options"),
        [
            pytest.param(
                _openai_prompt_descriptor,
                {"auth_token": API_KEY},
                id="openai",
            ),
            pytest.param(
                _anthropic_prompt_descriptor,
                {"max_tokens": 64, "extra_headers": {"api_key": API_KEY}},
                id="anthropic",
            ),
            pytest.param(
                _google_prompt_descriptor,
                {"credentials": API_KEY},
                id="google",
            ),
        ],
    )
    def test_prompt_descriptors_reject_sensitive_options(self, factory, options):
        with pytest.raises((TypeError, ValueError), match="sensitive|Unsupported") as exc_info:
            factory(options)
        _assert_no_plaintext(str(exc_info.value))

    def test_internal_vllm_plan_seals_sensitive_values(self):
        plan = _vllm_prompt_plan({"engine_args": {"hf_token": HUB_TOKEN}})

        _assert_no_plaintext(repr(plan))
        _assert_no_plaintext(repr(plan.get_options()))

    def test_credential_kwarg_cannot_land_in_embed_options(self):
        from vane.ai.providers.google import GoogleProvider

        with pytest.raises(TypeError, match="api_key") as exc_info:
            GoogleProvider(api_key=API_KEY)
        assert API_KEY not in str(exc_info.value)

    def test_transformers_embedder_rejects_hub_token(self):
        from vane.ai.providers.transformers import TransformersTextEmbedderDescriptor

        with pytest.raises(TypeError, match="token") as exc_info:
            TransformersTextEmbedderDescriptor(
                model="sentence-transformers/all-MiniLM-L6-v2",
                options={"revision": "pinned", "token": HUB_TOKEN},
            )
        assert HUB_TOKEN not in str(exc_info.value)

    @pytest.mark.parametrize(
        ("factory", "base_options"),
        [
            pytest.param(
                _openai_provider_embedder_descriptor,
                {},
                id="openai-provider-embed",
            ),
            pytest.param(
                _openai_provider_prompt_descriptor,
                {},
                id="openai-provider-prompt",
            ),
            pytest.param(
                _openai_embedder_descriptor_with_options,
                {},
                id="openai-descriptor-embed",
            ),
            pytest.param(
                _openai_prompt_descriptor,
                {},
                id="openai-descriptor-prompt",
            ),
            pytest.param(
                _anthropic_provider_prompt_descriptor,
                {"max_tokens": 64},
                id="anthropic-provider",
            ),
            pytest.param(
                _anthropic_prompt_descriptor,
                {"max_tokens": 64},
                id="anthropic-descriptor",
            ),
        ],
    )
    def test_credential_bearing_base_url_is_rejected_at_descriptor_boundary(self, factory, base_options):
        credential_url = f"https://user:{API_KEY}@api.example/v1"
        options = {**base_options, "base_url": credential_url}

        with pytest.raises(ValueError, match="cannot contain credentials") as exc_info:
            factory(options)

        _assert_no_plaintext(str(exc_info.value))


# ---------------------------------------------------------------------------
# get_options() stays wrapped
# ---------------------------------------------------------------------------


class TestGetOptionsStaysWrapped:
    @pytest.mark.parametrize("factory", ALL_DESCRIPTOR_FACTORIES)
    def test_get_options_repr_has_no_plaintext(self, factory):
        options = factory().get_options()
        _assert_no_plaintext(repr(options))


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


# ---------------------------------------------------------------------------
# Plaintext restored exactly at the execution boundary
# ---------------------------------------------------------------------------


class TestOptionsAtExecutionBoundary:
    def test_openai_embedder_client_receives_non_sensitive_client_options(self, monkeypatch):
        client = _fresh_recording_client()
        _install_fake_openai(monkeypatch, client)
        _openai_embedder_descriptor().instantiate()
        assert client.calls == [
            {
                "base_url": "https://api.example",
                "max_retries": 0,
            }
        ]

    @pytest.mark.parametrize("kind", ["embed", "prompt"])
    def test_openai_default_endpoint_ignores_sdk_environment_override(self, monkeypatch, kind):
        from vane.ai.providers.openai import OpenAIPrompterDescriptor, OpenAITextEmbedderDescriptor

        client = _fresh_recording_client()
        _install_fake_openai(monkeypatch, client)
        monkeypatch.setenv("OPENAI_BASE_URL", "https://compatible.example.test/v1")

        descriptor = OpenAITextEmbedderDescriptor() if kind == "embed" else OpenAIPrompterDescriptor()
        descriptor.instantiate()

        assert client.calls == [{"base_url": "https://api.openai.com/v1", "max_retries": 0}]

    def test_openai_unknown_dimensions_do_not_probe_with_plaintext(self, monkeypatch):
        from vane.ai.providers.openai import OpenAITextEmbedderDescriptor

        probe_calls = []

        class FakeProbeClient:
            def __init__(self, **kwargs):
                probe_calls.append(kwargs)
                self.embeddings = SimpleNamespace(
                    create=lambda **_: SimpleNamespace(data=[SimpleNamespace(embedding=[0.0] * 7)])
                )

        _install_fake_openai(monkeypatch, _fresh_recording_client(), sync_client=FakeProbeClient)
        with pytest.raises(ValueError, match="pass dimensions"):
            OpenAITextEmbedderDescriptor(
                model_name="custom-served-model",
                options={"base_url": "https://api.example"},
            )
        assert probe_calls == []

    def test_google_embedder_client_uses_environment_credentials(self, monkeypatch):
        client = _fresh_recording_client()
        _install_fake_google(monkeypatch, client)
        _google_embedder_descriptor().instantiate()
        assert len(client.calls) == 1
        assert "api_key" not in client.calls[0]
        assert client.calls[0]["http_options"].retry_options.attempts == 1

    def test_transformers_get_dimensions_uses_static_metadata_without_loading_config(self, monkeypatch):
        auto_config_calls = []

        class FakeAutoConfig:
            @staticmethod
            def from_pretrained(model, **options):
                auto_config_calls.append((model, options))
                return SimpleNamespace(hidden_size=384)

        monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoConfig=FakeAutoConfig))
        descriptor = _transformers_embedder_descriptor()
        assert descriptor.get_dimensions() == 384
        assert auto_config_calls == []

    def test_transformers_embedder_model_receives_registered_loading_options(self, monkeypatch):
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
                {
                    "trust_remote_code": False,
                    "backend": "torch",
                    "revision": "pinned",
                    "device": "cpu",
                },
            )
        ]

    def test_runtime_classes_accept_non_sensitive_options_mapping(self, monkeypatch):
        from vane.ai.providers.openai import OpenAITextEmbedder

        client = _fresh_recording_client()
        _install_fake_openai(monkeypatch, client)
        OpenAITextEmbedder(
            options={"base_url": "https://api.example"},
            model="text-embedding-3-small",
        )
        assert client.calls == [{"base_url": "https://api.example", "max_retries": 0}]


# ---------------------------------------------------------------------------
# Pickle round-trips
# ---------------------------------------------------------------------------


class TestPickleRoundTrip:
    @pytest.mark.parametrize("factory", ALL_DESCRIPTOR_FACTORIES)
    def test_repr_stays_redacted_after_pickle(self, factory):
        restored = pickle.loads(pickle.dumps(factory()))
        _assert_no_plaintext(repr(restored))

    def test_openai_pickled_descriptor_still_builds_working_client(self, monkeypatch):
        client = _fresh_recording_client()
        _install_fake_openai(monkeypatch, client)
        restored = pickle.loads(pickle.dumps(_openai_embedder_descriptor()))
        restored.instantiate()
        assert client.calls == [
            {
                "base_url": "https://api.example",
                "max_retries": 0,
            }
        ]
