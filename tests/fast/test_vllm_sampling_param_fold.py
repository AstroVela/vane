# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Top-level vLLM ``max_tokens``/``temperature`` fold into ``sampling_params``.

Covers vane#142: the executor only reads ``generate_args["sampling_params"]``,
so the vLLM descriptor folds the convenience options in at construction —
otherwise they are silently dropped and vLLM's ``SamplingParams`` default
(``max_tokens=16``) truncates output. Explicit ``sampling_params`` entries win
over the convenience fields on conflict, user input mappings are never
mutated, and nested credentials stay sealed through the fold.
"""

from __future__ import annotations

import copy
from decimal import Decimal

import pytest

from vane.ai._redaction import REDACTED_PLACEHOLDER, unwrap_sensitive_options, wrap_sensitive_options
from vane.ai.options import VLLMPromptOptions
from vane.ai.providers.vllm import VLLMPrompterDescriptor, VLLMProvider

API_KEY = "sk-PLAINTEXT-VLLM-FOLD-KEY-SENTINEL-0123456789"


def _descriptor_from_prompt_options(options: VLLMPromptOptions) -> VLLMPrompterDescriptor:
    """Mirror the ``prompt(..., prompt_options=...)`` flow into the descriptor."""
    return VLLMProvider().get_prompter(**options.to_descriptor_options())


def _unwrapped_options(descriptor: VLLMPrompterDescriptor) -> dict:
    return unwrap_sensitive_options(descriptor.get_options())


class TestFoldHappens:
    def test_top_level_max_tokens_reaches_sampling_params(self):
        descriptor = _descriptor_from_prompt_options(VLLMPromptOptions(max_tokens=512))
        options = _unwrapped_options(descriptor)
        assert options["generate_args"]["sampling_params"]["max_tokens"] == 512
        assert "max_tokens" not in options

    def test_top_level_temperature_reaches_sampling_params(self):
        descriptor = _descriptor_from_prompt_options(VLLMPromptOptions(temperature=0.25))
        options = _unwrapped_options(descriptor)
        assert options["generate_args"]["sampling_params"]["temperature"] == 0.25
        assert "temperature" not in options

    def test_fold_merges_with_existing_generate_args(self):
        descriptor = _descriptor_from_prompt_options(
            VLLMPromptOptions(
                generate_args={"sampling_params": {"top_p": 0.9}, "lora_request": "adapter"},
                max_tokens=256,
                temperature=0.5,
            )
        )
        options = _unwrapped_options(descriptor)
        assert options["generate_args"]["sampling_params"] == {
            "top_p": 0.9,
            "max_tokens": 256,
            "temperature": 0.5,
        }
        assert options["generate_args"]["lora_request"] == "adapter"

    def test_direct_descriptor_construction_folds(self):
        descriptor = VLLMPrompterDescriptor(vllm_options={"max_tokens": 512})
        options = _unwrapped_options(descriptor)
        assert options["generate_args"]["sampling_params"]["max_tokens"] == 512

    def test_prompter_receives_folded_options(self):
        descriptor = _descriptor_from_prompt_options(VLLMPromptOptions(max_tokens=512))
        prompter = descriptor.instantiate()
        assert prompter._options["generate_args"]["sampling_params"]["max_tokens"] == 512

    def test_on_error_stays_top_level(self):
        descriptor = _descriptor_from_prompt_options(VLLMPromptOptions(max_tokens=8, on_error="log"))
        options = _unwrapped_options(descriptor)
        assert options["on_error"] == "log"


class TestExplicitSamplingParamsPrecedence:
    def test_explicit_entry_wins_over_convenience_field(self):
        descriptor = _descriptor_from_prompt_options(
            VLLMPromptOptions(
                generate_args={"sampling_params": {"max_tokens": 64}},
                max_tokens=512,
            )
        )
        sampling_params = _unwrapped_options(descriptor)["generate_args"]["sampling_params"]
        assert sampling_params["max_tokens"] == 64

    def test_non_conflicting_field_still_folds_alongside_explicit_entry(self):
        descriptor = _descriptor_from_prompt_options(
            VLLMPromptOptions(
                generate_args={"sampling_params": {"max_tokens": 64}},
                max_tokens=512,
                temperature=0.1,
            )
        )
        sampling_params = _unwrapped_options(descriptor)["generate_args"]["sampling_params"]
        assert sampling_params == {"max_tokens": 64, "temperature": 0.1}


class TestNoInputMutation:
    def test_user_mappings_are_not_mutated(self):
        sampling_params = {"top_p": 0.9}
        generate_args = {"sampling_params": sampling_params}
        options = VLLMPromptOptions(generate_args=generate_args, max_tokens=128)
        _descriptor_from_prompt_options(options)
        assert generate_args == {"sampling_params": {"top_p": 0.9}}
        assert sampling_params == {"top_p": 0.9}

    def test_descriptor_input_dict_is_not_mutated(self):
        vllm_options = {"max_tokens": 512, "generate_args": {"sampling_params": {"seed": 7}}}
        snapshot = copy.deepcopy(vllm_options)
        VLLMPrompterDescriptor(vllm_options=vllm_options)
        assert vllm_options == snapshot


class TestSecretInterplay:
    def test_repr_still_redacts_nested_credentials(self):
        options = VLLMPromptOptions(generate_args={"api_key": API_KEY}, max_tokens=512)
        rendered = repr(options)
        assert API_KEY not in rendered
        assert REDACTED_PLACEHOLDER in rendered
        assert "max_tokens=512" in rendered

    def test_descriptor_repr_redacts_credentials_after_fold(self):
        descriptor = VLLMPrompterDescriptor(vllm_options={"generate_args": {"api_key": API_KEY}, "max_tokens": 512})
        rendered = repr(descriptor)
        assert API_KEY not in rendered
        assert REDACTED_PLACEHOLDER in rendered

    def test_fold_preserves_presealed_credentials(self):
        presealed = wrap_sensitive_options({"generate_args": {"api_key": API_KEY}, "max_tokens": 9})
        descriptor = VLLMPrompterDescriptor(vllm_options=presealed)
        options = _unwrapped_options(descriptor)
        assert options["generate_args"]["api_key"] == API_KEY
        assert options["generate_args"]["sampling_params"]["max_tokens"] == 9


class TestNonMappingContainers:
    def test_non_mapping_sampling_params_with_convenience_field_raises(self):
        with pytest.raises(TypeError, match="sampling_params"):
            VLLMPrompterDescriptor(
                vllm_options={"max_tokens": 5, "generate_args": {"sampling_params": '{"max_tokens": 3}'}}
            )

    def test_non_mapping_generate_args_with_convenience_field_raises(self):
        with pytest.raises(TypeError, match="generate_args"):
            VLLMPrompterDescriptor(vllm_options={"max_tokens": 5, "generate_args": "not-a-mapping"})

    def test_non_mapping_sampling_params_without_convenience_field_passes_through(self):
        descriptor = VLLMPrompterDescriptor(vllm_options={"generate_args": {"sampling_params": '{"max_tokens": 3}'}})
        options = _unwrapped_options(descriptor)
        assert options["generate_args"]["sampling_params"] == '{"max_tokens": 3}'


class TestNoOpCases:
    def test_none_convenience_values_are_stripped_without_fold(self):
        descriptor = VLLMPrompterDescriptor(vllm_options={"max_tokens": None, "temperature": None})
        options = _unwrapped_options(descriptor)
        assert "max_tokens" not in options
        assert "temperature" not in options
        assert "generate_args" not in options

    def test_options_without_convenience_fields_stay_unchanged(self):
        descriptor = VLLMPrompterDescriptor(vllm_options={"generate_args": {"lora_request": "adapter"}})
        options = _unwrapped_options(descriptor)
        assert options["generate_args"] == {"lora_request": "adapter"}
        assert "sampling_params" not in options["generate_args"]


class TestSQLPathFolds:
    """SQL struct_pack options flow through get_prompter into the same fold."""

    def test_sql_top_level_max_tokens_folds(self):
        from vane.ai._sql import _normalize_sql_options

        opts = _normalize_sql_options({"max_tokens": Decimal(512), "temperature": Decimal("0.5")})
        descriptor = VLLMProvider().get_prompter(**opts)
        sampling_params = _unwrapped_options(descriptor)["generate_args"]["sampling_params"]
        assert sampling_params == {"max_tokens": 512, "temperature": 0.5}

    def test_sql_explicit_generate_args_json_wins(self):
        from vane.ai._sql import _normalize_sql_options

        opts = _normalize_sql_options(
            {
                "max_tokens": Decimal(512),
                "generate_args_json": '{"sampling_params": {"max_tokens": 64}}',
            }
        )
        descriptor = VLLMProvider().get_prompter(**opts)
        sampling_params = _unwrapped_options(descriptor)["generate_args"]["sampling_params"]
        assert sampling_params["max_tokens"] == 64
