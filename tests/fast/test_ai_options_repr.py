# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import vane
from vane.ai.options import PromptOptions, normalize_prompt_options


def test_prompt_options_is_the_only_public_prompt_option_type():
    assert vane.ai.PromptOptions is PromptOptions
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


def test_prompt_options_are_plain_closed_mappings():
    options: PromptOptions = {
        "temperature": 0.2,
        "actor_number": 2,
        "max_concurrency_per_actor": 7,
    }
    assert normalize_prompt_options("openai", options, relation=False) == options


def test_prompt_outer_normalization_leaves_provider_values_for_adapter_validation():
    options: PromptOptions = {"temperature": -0.1}

    assert normalize_prompt_options("openai", options, relation=False) == options


@pytest.mark.parametrize(
    "options",
    [
        {"api_key": "plaintext-secret"},
        {"engine_args": {"hf_token": "plaintext-secret"}},
        {"generate_args": {"headers": {"authorization": "plaintext-secret"}}},
    ],
)
def test_prompt_options_reject_sensitive_values_before_repr_or_planning(options):
    family = "vllm" if "engine_args" in options or "generate_args" in options else "openai"
    with pytest.raises(ValueError, match="sensitive") as error:
        normalize_prompt_options(family, options, relation=False)
    assert "plaintext-secret" not in str(error.value)
