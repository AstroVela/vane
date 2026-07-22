# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Execution options must take effect or fail loudly on every AI surface.

Covers vane#147: API prompter/embedder descriptors read ``batch_size`` and
``concurrency`` from their stored options, declare ``num_gpus=0`` (pure HTTP
providers) so actor fan-out works on the relation path, OpenAIProvider routes
provider-level defaults through ``_split_options`` like Anthropic/Google,
``return_format`` smuggled inside option dicts is rejected with a clean
TypeError, the vLLM ``on_error`` vocabulary is translated at the engine
boundary, and ``image_columns`` on a batch-only prompter raises instead of
silently dropping images.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pyarrow as pa
import pytest


def _load_functions() -> types.ModuleType:
    """Import the real ``vane.ai.functions``, even under the no-duckdb harness.

    The stub harness plugin registers a placeholder for ``vane.ai.functions``
    because the real module imports duckdb-backed modules at top level. When
    that placeholder is present, evict it and stub just the duckdb-importing
    dependencies so the real module under test can load. On CI (where duckdb
    imports fine) the real module imports normally and nothing is stubbed.
    """
    module = sys.modules.get("vane.ai.functions")
    if getattr(module, "__file__", None):
        return module  # real module already loaded
    if module is not None:
        sys.modules.pop("vane.ai.functions")
        stub_specs: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("vane._expressions", ("as_expression", "is_expression")),
            ("vane._expression_udf", ("_build_actor_map_batches_expression",)),
        )
        for name, attrs in stub_specs:
            if name not in sys.modules:
                stub = types.ModuleType(name)
                for attr in attrs:
                    setattr(stub, attr, lambda *a, **k: None)
                sys.modules[name] = stub
    return importlib.import_module("vane.ai.functions")


functions = _load_functions()

from vane.ai.providers.anthropic import AnthropicPrompterDescriptor  # noqa: E402
from vane.ai.providers.google import (  # noqa: E402
    GooglePrompterDescriptor,
    GoogleTextEmbedderDescriptor,
)
from vane.ai.providers.openai import (  # noqa: E402
    OpenAIPrompterDescriptor,
    OpenAIProvider,
    OpenAITextEmbedderDescriptor,
)
from vane.ai.providers.vllm import VLLMPrompter, VLLMPrompterDescriptor  # noqa: E402
from vane.ai.typing import UDFOptions  # noqa: E402


def test_real_functions_module_is_under_test() -> None:
    """Guard: the harness must import the real module, not the plugin stub."""
    assert getattr(functions, "__file__", None)
    assert isinstance(functions._PromptBatch, type)


class _FakeRelation:
    """Captures ``map_batches`` kwargs for relation-path assertions."""

    def __init__(self) -> None:
        self.map_batches_kwargs: dict | None = None

    def map_batches(self, udf, **kwargs):
        self.map_batches_kwargs = kwargs
        return "mapped"

    def select(self, *args, **kwargs):
        raise NotImplementedError


class _StubProvider:
    """Provider stand-in returning a pre-built descriptor."""

    def __init__(self, descriptor) -> None:
        self._descriptor = descriptor
        self.get_prompter_kwargs: dict | None = None

    def get_prompter(self, **kwargs):
        self.get_prompter_kwargs = kwargs
        return self._descriptor


# ---------------------------------------------------------------------------
# 1. batch_size plumb-through
# ---------------------------------------------------------------------------


class TestBatchSizePlumbThrough:
    def test_api_prompter_descriptors_read_batch_size(self):
        assert OpenAIPrompterDescriptor(prompt_options={"batch_size": 16}).get_udf_options().batch_size == 16
        assert AnthropicPrompterDescriptor(prompt_options={"batch_size": 8}).get_udf_options().batch_size == 8
        assert GooglePrompterDescriptor(prompt_options={"batch_size": 4}).get_udf_options().batch_size == 4

    def test_google_embedder_descriptor_reads_batch_size(self):
        assert GoogleTextEmbedderDescriptor(embed_options={"batch_size": 12}).get_udf_options().batch_size == 12

    def test_prompter_batch_size_defaults_to_none_when_unset(self):
        """The relation-path default of 1 is applied downstream, not here."""
        assert OpenAIPrompterDescriptor().get_udf_options().batch_size is None
        assert AnthropicPrompterDescriptor().get_udf_options().batch_size is None
        assert GooglePrompterDescriptor().get_udf_options().batch_size is None

    def test_relation_prompt_explicit_batch_size_reaches_map_batches(self):
        descriptor = OpenAIPrompterDescriptor(prompt_options={"batch_size": 16})
        rel = _FakeRelation()

        assert functions.prompt(rel, "text", provider=_StubProvider(descriptor)) == "mapped"
        assert rel.map_batches_kwargs["batch_size"] == 16

    def test_relation_prompt_default_batch_size_stays_one(self):
        descriptor = OpenAIPrompterDescriptor()
        rel = _FakeRelation()

        functions.prompt(rel, "text", provider=_StubProvider(descriptor))
        assert rel.map_batches_kwargs["batch_size"] == 1


# ---------------------------------------------------------------------------
# 2. concurrency kwarg maps to actor_number on the Python path
# ---------------------------------------------------------------------------


class TestConcurrencyKwarg:
    def test_non_positive_concurrency_is_rejected(self):
        """Python path validates like SQL (_int_or_none): positive integers only."""
        with pytest.raises(ValueError, match="concurrency must be a positive integer"):
            OpenAIPrompterDescriptor(prompt_options={"concurrency": 0}).get_udf_options()
        with pytest.raises(ValueError, match="actor_number must be a positive integer"):
            OpenAIPrompterDescriptor(prompt_options={"actor_number": -1}).get_udf_options()

    def test_api_descriptors_map_concurrency_to_actor_number(self):
        assert OpenAITextEmbedderDescriptor(embed_options={"concurrency": 2}).get_udf_options().actor_number == 2
        assert OpenAIPrompterDescriptor(prompt_options={"concurrency": 3}).get_udf_options().actor_number == 3
        assert AnthropicPrompterDescriptor(prompt_options={"concurrency": 4}).get_udf_options().actor_number == 4
        assert GoogleTextEmbedderDescriptor(embed_options={"concurrency": 5}).get_udf_options().actor_number == 5
        assert GooglePrompterDescriptor(prompt_options={"concurrency": 6}).get_udf_options().actor_number == 6

    def test_explicit_actor_number_wins_over_concurrency(self):
        descriptor = OpenAIPrompterDescriptor(prompt_options={"concurrency": 3, "actor_number": 2})
        assert descriptor.get_udf_options().actor_number == 2

    def test_concurrency_is_int_coerced(self):
        descriptor = GooglePrompterDescriptor(prompt_options={"concurrency": 3.0})
        assert descriptor.get_udf_options().actor_number == 3


# ---------------------------------------------------------------------------
# 3. GPU guard coherence for pure HTTP providers
# ---------------------------------------------------------------------------


class TestApiProviderGpuDefaults:
    def test_api_descriptors_declare_zero_gpus_by_default(self):
        assert OpenAITextEmbedderDescriptor().get_udf_options().num_gpus == 0
        assert OpenAIPrompterDescriptor().get_udf_options().num_gpus == 0
        assert AnthropicPrompterDescriptor().get_udf_options().num_gpus == 0
        assert GoogleTextEmbedderDescriptor().get_udf_options().num_gpus == 0
        assert GooglePrompterDescriptor().get_udf_options().num_gpus == 0

    def test_explicit_num_gpus_is_preserved(self):
        assert OpenAIPrompterDescriptor(prompt_options={"num_gpus": 2}).get_udf_options().num_gpus == 2
        assert GoogleTextEmbedderDescriptor(embed_options={"num_gpus": 1}).get_udf_options().num_gpus == 1

    def test_api_provider_concurrency_does_not_trip_gpu_guard(self):
        """provider_options=...(concurrency=4) works on the relation path."""
        descriptor = OpenAIPrompterDescriptor(prompt_options={"actor_number": 4})

        kwargs = functions._map_batches_kwargs(descriptor.get_udf_options(), None)

        assert kwargs["actor_number"] == 4
        assert kwargs["gpus"] == 0

    def test_relation_prompt_concurrency_end_to_end(self):
        descriptor = AnthropicPrompterDescriptor(prompt_options={"concurrency": 4})
        rel = _FakeRelation()

        functions.prompt(rel, "text", provider=_StubProvider(descriptor))

        assert rel.map_batches_kwargs["actor_number"] == 4
        assert rel.map_batches_kwargs["gpus"] == 0

    def test_gpu_guard_still_protects_undeclared_num_gpus(self):
        """GPU-capable providers that leave num_gpus None keep the guard."""
        with pytest.raises(ValueError, match="num_gpus is required"):
            functions._map_batches_kwargs(UDFOptions(actor_number=2, batch_size=8), None)


# ---------------------------------------------------------------------------
# 4. return_format smuggled inside option dicts
# ---------------------------------------------------------------------------


class TestReturnFormatSmuggling:
    def _fake_provider(self):
        descriptor = SimpleNamespace(
            get_udf_options=lambda: UDFOptions(),
            get_provider=lambda: "fake",
            get_model=lambda: "fake-model",
            instantiate=lambda: None,
        )
        return _StubProvider(descriptor)

    def test_expression_prompt_rejects_return_format_in_prompt_options(self):
        with pytest.raises(TypeError, match="return_format"):
            functions.prompt(
                "messages-text",
                provider=self._fake_provider(),
                prompt_options={"return_format": dict},
            )

    def test_expression_prompt_rejects_return_format_in_provider_options(self):
        with pytest.raises(TypeError, match="return_format"):
            functions.prompt(
                "messages-text",
                provider=self._fake_provider(),
                provider_options={"return_format": dict},
            )

    def test_sql_prompt_spec_rejects_return_format(self, monkeypatch):
        from vane.ai import _sql
        from vane.ai import provider as provider_registry

        fake = self._fake_provider()
        monkeypatch.setitem(provider_registry.PROVIDERS, "mock_smuggle", lambda name=None, **options: fake)

        with pytest.raises(TypeError, match="return_format"):
            _sql.build_ai_prompt_sql_spec({"provider": "mock_smuggle", "return_format": {"type": "object"}})


# ---------------------------------------------------------------------------
# 5. OpenAIProvider _split_options
# ---------------------------------------------------------------------------


class TestOpenAISplitOptions:
    def test_provider_level_defaults_reach_prompt_options(self):
        provider = OpenAIProvider(api_key="k", max_api_concurrency=8, on_error="ignore")

        descriptor = provider.get_prompter()
        udf_opts = descriptor.get_udf_options()

        assert udf_opts.max_api_concurrency == 8
        assert udf_opts.on_error == "ignore"
        assert set(descriptor.provider_options) == {"api_key"}

    def test_provider_level_batch_size_reaches_embed_options(self):
        provider = OpenAIProvider(api_key="k", batch_size=16)

        descriptor = provider.get_text_embedder()

        assert descriptor.get_udf_options().batch_size == 16
        assert set(descriptor.provider_options) == {"api_key"}

    def test_call_level_options_override_provider_defaults(self):
        provider = OpenAIProvider(on_error="ignore", base_url="https://provider.example")

        descriptor = provider.get_prompter(on_error="raise", base_url="https://call.example")

        assert descriptor.get_udf_options().on_error == "raise"
        assert descriptor.provider_options["base_url"] == "https://call.example"

    def test_client_keys_still_split_from_call_options(self):
        provider = OpenAIProvider()

        descriptor = provider.get_prompter(api_key="call-key", temperature=0.3)

        assert set(descriptor.provider_options) == {"api_key"}
        assert descriptor.prompt_options["temperature"] == 0.3
        assert "api_key" not in descriptor.prompt_options

    def test_get_dimensions_probe_receives_only_client_kwargs(self, monkeypatch):
        probe_calls: list[dict] = []

        class FakeProbeClient:
            def __init__(self, **kwargs):
                probe_calls.append(kwargs)
                self.embeddings = SimpleNamespace(
                    create=lambda **_: SimpleNamespace(data=[SimpleNamespace(embedding=[0.0] * 5)])
                )

        monkeypatch.setitem(
            sys.modules,
            "openai",
            SimpleNamespace(OpenAI=FakeProbeClient, AsyncOpenAI=object, OpenAIError=Exception),
        )
        descriptor = OpenAITextEmbedderDescriptor(
            provider_options={"api_key": "k", "base_url": "https://api.example", "batch_size": 4},
            model_name="custom-served-model",
        )

        assert descriptor.get_dimensions().size == 5
        assert probe_calls == [{"api_key": "k", "base_url": "https://api.example"}]


# ---------------------------------------------------------------------------
# 6. vLLM on_error vocabulary translation
# ---------------------------------------------------------------------------


class TestVLLMOnErrorVocabulary:
    def test_engine_null_never_leaks_into_udf_options(self):
        descriptor = VLLMPrompterDescriptor(vllm_options={"on_error": "null"})
        assert descriptor.get_udf_options().on_error == "ignore"

    def test_vane_ignore_translates_to_engine_null(self):
        prompter = VLLMPrompter(model="test", vllm_options={"on_error": "ignore"})
        assert prompter._options["on_error"] == "null"

    def test_raise_and_log_pass_through_unchanged(self):
        assert VLLMPrompterDescriptor(vllm_options={"on_error": "raise"}).get_udf_options().on_error == "raise"
        assert VLLMPrompterDescriptor(vllm_options={"on_error": "log"}).get_udf_options().on_error == "log"
        assert VLLMPrompter(model="test", vllm_options={"on_error": "log"})._options["on_error"] == "log"

    def test_instantiate_translates_descriptor_ignore(self):
        descriptor = VLLMPrompterDescriptor(vllm_options={"on_error": "ignore"})
        prompter = descriptor.instantiate()
        assert prompter._options["on_error"] == "null"


# ---------------------------------------------------------------------------
# 7. image_columns on batch-only prompters
# ---------------------------------------------------------------------------


class TestImageColumnsBatchOnlyPrompter:
    class _BatchOnlyPrompter:
        def prompt_batch(self, texts):
            return [f"r:{t}" for t in texts]

    class _BatchOnlyDescriptor:
        def instantiate(self):
            return TestImageColumnsBatchOnlyPrompter._BatchOnlyPrompter()

    def test_image_columns_raise_instead_of_silent_drop(self):
        wrapper = functions._PromptBatch(
            self._BatchOnlyDescriptor(),
            "text",
            "response",
            image_columns=["img"],
        )
        table = pa.table({"text": ["hi"], "img": [b"\x89PNG\r\n\x1a\n"]})

        with pytest.raises(ValueError, match="image_columns"):
            wrapper(table)

    def test_text_only_batch_path_still_works(self):
        wrapper = functions._PromptBatch(self._BatchOnlyDescriptor(), "text", "response")
        table = pa.table({"text": ["a", "b"]})

        assert wrapper(table).column("response").to_pylist() == ["r:a", "r:b"]
