# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the engine-agnostic LLM executor base class."""

from __future__ import annotations


def test_llm_executor_contract_and_hierarchy():
    from vane.execution._llm_executor import LLMExecutor
    from vane.execution.vllm import (
        LocalVLLMExecutor,
        RayLocalVLLMExecutor,
        RemoteVLLMExecutor,
        VLLMExecutor,
    )

    assert issubclass(VLLMExecutor, LLMExecutor)
    assert issubclass(LocalVLLMExecutor, VLLMExecutor)
    assert issubclass(RayLocalVLLMExecutor, VLLMExecutor)
    assert issubclass(RemoteVLLMExecutor, VLLMExecutor)

    for name in (
        "submit",
        "take_ready_result",
        "finished_submitting",
        "all_tasks_finished",
        "shutdown",
    ):
        assert name in LLMExecutor.__abstractmethods__
