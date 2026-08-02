# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Top-level vLLM ``max_tokens``/``temperature`` fold into ``sampling_params``.

Regression tests for vane#142: the executor only reads
``generate_args["sampling_params"]`` from the native operator options, so
``NativeVLLMPromptPlan.build_physical_vllm_options()`` folds the top-level
convenience options in — otherwise they would be silently dropped and vLLM's
``SamplingParams`` default (``max_tokens=16``) would truncate output. These
tests pin the fold on the one path the executor consumes, for the Python
Prompt flow and the SQL option flow alike: explicit
``sampling_params`` entries win over the convenience fields on conflict, and
user input mappings are never mutated.
"""

from __future__ import annotations

import copy
from decimal import Decimal

import pytest

from vane.ai.providers.vllm import NativeVLLMPromptPlan, VLLMProvider


def _plan_from_prompt_options(options: dict) -> NativeVLLMPromptPlan:
    return VLLMProvider().get_prompter(**options)


def _native_sampling_params(plan: NativeVLLMPromptPlan) -> dict:
    """Return the ``sampling_params`` the executor will actually read."""
    return plan.build_physical_vllm_options()["generate_args"]["sampling_params"]


class TestFoldReachesNativeOptions:
    def test_top_level_max_tokens_reaches_sampling_params(self):
        plan = _plan_from_prompt_options({"max_tokens": 512})
        native = plan.build_physical_vllm_options()
        assert native["generate_args"]["sampling_params"]["max_tokens"] == 512
        assert "max_tokens" not in native

    def test_top_level_temperature_reaches_sampling_params(self):
        plan = _plan_from_prompt_options({"temperature": 0.25})
        native = plan.build_physical_vllm_options()
        assert native["generate_args"]["sampling_params"]["temperature"] == 0.25
        assert "temperature" not in native

    def test_fold_merges_with_existing_generate_args(self):
        plan = _plan_from_prompt_options(
            {
                "generate_args": {"sampling_params": {"top_p": 0.9}, "lora_request": "adapter"},
                "max_tokens": 256,
                "temperature": 0.5,
            }
        )
        native = plan.build_physical_vllm_options()
        assert native["generate_args"]["sampling_params"] == {
            "top_p": 0.9,
            "max_tokens": 256,
            "temperature": 0.5,
        }
        assert native["generate_args"]["lora_request"] == "adapter"

    def test_direct_plan_construction_folds(self):
        plan = NativeVLLMPromptPlan(vllm_options={"max_tokens": 512})
        assert _native_sampling_params(plan)["max_tokens"] == 512

    def test_on_error_stays_top_level(self):
        plan = _plan_from_prompt_options({"max_tokens": 8})
        plan.on_error = "ignore"
        native = plan.build_physical_vllm_options()
        assert native["on_error"] == "null"
        assert "on_error" not in native["generate_args"]["sampling_params"]


class TestExplicitSamplingParamsPrecedence:
    def test_explicit_entry_wins_over_convenience_field(self):
        plan = _plan_from_prompt_options({"generate_args": {"sampling_params": {"max_tokens": 64}}, "max_tokens": 512})
        assert _native_sampling_params(plan)["max_tokens"] == 64

    def test_non_conflicting_field_still_folds_alongside_explicit_entry(self):
        plan = _plan_from_prompt_options(
            {
                "generate_args": {"sampling_params": {"max_tokens": 64}},
                "max_tokens": 512,
                "temperature": 0.1,
            }
        )
        assert _native_sampling_params(plan) == {"max_tokens": 64, "temperature": 0.1}

    def test_json_string_sampling_params_entry_wins(self):
        plan = NativeVLLMPromptPlan(
            vllm_options={"max_tokens": 512, "generate_args": {"sampling_params": '{"max_tokens": 64}'}}
        )
        assert _native_sampling_params(plan)["max_tokens"] == 64

    def test_convenience_field_folds_into_json_string_sampling_params(self):
        plan = NativeVLLMPromptPlan(
            vllm_options={"max_tokens": 512, "generate_args": {"sampling_params": '{"top_p": 0.9}'}}
        )
        assert _native_sampling_params(plan) == {"top_p": 0.9, "max_tokens": 512}


class TestNoInputMutation:
    def test_user_mappings_are_not_mutated(self):
        sampling_params = {"top_p": 0.9}
        generate_args = {"sampling_params": sampling_params}
        plan = _plan_from_prompt_options({"generate_args": generate_args, "max_tokens": 128})
        plan.build_physical_vllm_options()
        assert generate_args == {"sampling_params": {"top_p": 0.9}}
        assert sampling_params == {"top_p": 0.9}

    def test_plan_input_dict_is_not_mutated(self):
        vllm_options = {"max_tokens": 512, "generate_args": {"sampling_params": {"seed": 7}}}
        snapshot = copy.deepcopy(vllm_options)
        plan = NativeVLLMPromptPlan(vllm_options=vllm_options)
        plan.build_physical_vllm_options()
        assert vllm_options == snapshot

    def test_build_is_repeatable(self):
        plan = NativeVLLMPromptPlan(vllm_options={"max_tokens": 512})
        assert plan.build_physical_vllm_options() == plan.build_physical_vllm_options()


def test_sensitive_nested_options_are_rejected():
    with pytest.raises(ValueError, match="sensitive"):
        NativeVLLMPromptPlan(vllm_options={"generate_args": {"api_key": "secret"}, "max_tokens": 9})


class TestNonMappingContainers:
    def test_non_mapping_generate_args_with_convenience_field_raises(self):
        with pytest.raises(TypeError, match="generate_args"):
            NativeVLLMPromptPlan(vllm_options={"max_tokens": 5, "generate_args": "not-a-mapping"})

    def test_invalid_sampling_params_json_raises(self):
        plan = NativeVLLMPromptPlan(
            vllm_options={"max_tokens": 5, "generate_args": {"sampling_params": '{"max_tokens":'}}
        )
        with pytest.raises(ValueError, match="sampling_params"):
            plan.build_physical_vllm_options()


class TestNoOpCases:
    def test_none_convenience_values_are_dropped_without_fold(self):
        plan = NativeVLLMPromptPlan(vllm_options={"max_tokens": None, "temperature": None})
        native = plan.build_physical_vllm_options()
        assert "max_tokens" not in native
        assert "temperature" not in native
        assert "generate_args" not in native

    def test_options_without_convenience_fields_pass_through(self):
        plan = NativeVLLMPromptPlan(vllm_options={"generate_args": {"lora_request": "adapter"}})
        native = plan.build_physical_vllm_options()
        assert native["generate_args"] == {"lora_request": "adapter"}


class TestSQLPathFolds:
    """SQL struct_pack options flow through get_prompter into the same fold."""

    def test_sql_top_level_options_fold(self):
        from vane.ai._sql import _normalize_sql_options

        opts = _normalize_sql_options({"max_tokens": Decimal(512), "temperature": Decimal("0.5")})
        plan = VLLMProvider().get_prompter(**opts)
        assert _native_sampling_params(plan) == {"max_tokens": 512, "temperature": 0.5}

    def test_sql_explicit_generate_args_wins(self):
        from vane.ai._sql import _normalize_sql_options

        opts = _normalize_sql_options(
            {
                "max_tokens": Decimal(512),
                "generate_args": {"sampling_params": {"max_tokens": Decimal(64)}},
            }
        )
        plan = VLLMProvider().get_prompter(**opts)
        assert _native_sampling_params(plan)["max_tokens"] == 64
