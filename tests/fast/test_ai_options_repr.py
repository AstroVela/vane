# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Public provider options never leak ``api_key`` through repr, str, or assertion diffs.

Covers vane#105: the three api_key-bearing options dataclasses (OpenAI,
Anthropic, Google) render a fixed placeholder instead of the plaintext key,
while construction, ``.api_key`` reads, equality, and the credential-free
options dataclasses stay untouched.
"""

from __future__ import annotations

import dataclasses

import pytest

from vane.ai._redaction import REDACTED_PLACEHOLDER
from vane.ai.options import (
    AnthropicPromptOptions,
    AnthropicProviderOptions,
    GoogleEmbeddingOptions,
    GooglePromptOptions,
    GoogleProviderOptions,
    OpenAIEmbeddingOptions,
    OpenAIPromptOptions,
    OpenAIProviderOptions,
    VLLMPromptOptions,
    VLLMProviderOptions,
)

API_KEY = "sk-PLAINTEXT-OPTIONS-KEY-SENTINEL-0123456789"

API_KEY_OPTIONS_CLASSES = [
    OpenAIProviderOptions,
    AnthropicProviderOptions,
    GoogleProviderOptions,
]

CREDENTIAL_FREE_OPTIONS_CLASSES = [
    VLLMProviderOptions,
    OpenAIPromptOptions,
    OpenAIEmbeddingOptions,
    AnthropicPromptOptions,
    GooglePromptOptions,
    GoogleEmbeddingOptions,
    VLLMPromptOptions,
]


def _default_dataclass_repr(instance) -> str:
    rendered = ", ".join(f"{field.name}={getattr(instance, field.name)!r}" for field in dataclasses.fields(instance))
    return f"{type(instance).__qualname__}({rendered})"


@pytest.mark.parametrize("options_cls", API_KEY_OPTIONS_CLASSES)
class TestApiKeyOptionsRedaction:
    def test_repr_and_str_mask_api_key(self, options_cls):
        options = options_cls(api_key=API_KEY)
        for rendered in (repr(options), str(options), f"{options}", "{}".format(options)):
            assert REDACTED_PLACEHOLDER in rendered
            assert API_KEY not in rendered

    def test_none_api_key_renders_as_none(self, options_cls):
        options = options_cls(api_key=None)
        assert "api_key=None" in repr(options)
        assert REDACTED_PLACEHOLDER not in repr(options)

    def test_repr_keeps_dataclass_shape(self, options_cls):
        options = options_cls(api_key=API_KEY)
        rendered = repr(options)
        assert rendered.startswith(f"{options_cls.__name__}(")
        assert rendered.endswith(")")
        for field in dataclasses.fields(options):
            assert f"{field.name}=" in rendered

    def test_construction_and_read_unchanged(self, options_cls):
        options = options_cls(api_key=API_KEY)
        assert options.api_key == API_KEY
        assert isinstance(options.api_key, str)
        assert options.to_descriptor_options()["api_key"] == API_KEY

    def test_equality_unchanged(self, options_cls):
        assert options_cls(api_key=API_KEY) == options_cls(api_key=API_KEY)
        assert options_cls(api_key=API_KEY) != options_cls(api_key="other")
        assert options_cls(api_key=None) == options_cls()

    def test_still_frozen(self, options_cls):
        options = options_cls(api_key=API_KEY)
        with pytest.raises(dataclasses.FrozenInstanceError):
            options.api_key = "changed"

    def test_field_type_stays_optional_str(self, options_cls):
        (api_key_field,) = [field for field in dataclasses.fields(options_cls) if field.name == "api_key"]
        assert api_key_field.type == "str | None"


class TestOtherFieldsRenderNormally:
    def test_openai_full_repr(self):
        options = OpenAIProviderOptions(
            base_url="https://api.example",
            api_key=API_KEY,
            organization="org-plain",
            timeout=30.0,
            concurrency=4,
            max_api_concurrency=8,
        )
        assert repr(options) == (
            "OpenAIProviderOptions(base_url='https://api.example', "
            f"api_key={REDACTED_PLACEHOLDER}, organization='org-plain', "
            "timeout=30.0, concurrency=4, max_api_concurrency=8)"
        )

    def test_anthropic_full_repr(self):
        options = AnthropicProviderOptions(
            api_key=API_KEY,
            base_url="https://api.example",
            timeout=15.5,
            max_retries=3,
            concurrency=2,
            max_api_concurrency=6,
        )
        assert repr(options) == (
            f"AnthropicProviderOptions(api_key={REDACTED_PLACEHOLDER}, "
            "base_url='https://api.example', timeout=15.5, max_retries=3, "
            "concurrency=2, max_api_concurrency=6)"
        )

    def test_google_full_repr(self):
        options = GoogleProviderOptions(api_key=API_KEY, concurrency=3, max_api_concurrency=9)
        assert repr(options) == (
            f"GoogleProviderOptions(api_key={REDACTED_PLACEHOLDER}, concurrency=3, max_api_concurrency=9)"
        )

    def test_all_none_repr_matches_default_dataclass_shape(self):
        for options_cls in API_KEY_OPTIONS_CLASSES:
            options = options_cls()
            assert repr(options) == _default_dataclass_repr(options)


class TestCredentialFreeOptionsUntouched:
    @pytest.mark.parametrize("options_cls", CREDENTIAL_FREE_OPTIONS_CLASSES)
    def test_keeps_generated_dataclass_repr(self, options_cls):
        options = options_cls()
        assert repr(options) == _default_dataclass_repr(options)
        assert options_cls.__repr__.__qualname__ == f"{options_cls.__qualname__}.__repr__"

    def test_regular_values_render_verbatim(self):
        options = OpenAIPromptOptions(max_tokens=128, temperature=0.5)
        assert repr(options) == (
            "OpenAIPromptOptions(use_chat_completions=None, max_output_tokens=None, "
            "max_tokens=128, temperature=0.5, on_error=None)"
        )


@pytest.mark.parametrize("options_cls", API_KEY_OPTIONS_CLASSES)
class TestAssertionDiffDoesNotLeak:
    def test_failing_comparison_with_equal_keys_hides_key(self, options_cls):
        left = options_cls(api_key=API_KEY, concurrency=1)
        right = options_cls(api_key=API_KEY, concurrency=2)
        with pytest.raises(AssertionError) as excinfo:
            assert left == right
        message = str(excinfo.value)
        assert API_KEY not in message
        assert "concurrency" in message
