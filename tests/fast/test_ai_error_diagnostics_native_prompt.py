# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest

import vane
from vane.ai import DiagnosticError, summarize_error
from vane.ai.providers.vllm import _build_native_vllm_options_argument


@pytest.mark.parametrize("engine,executor_name", [("vllm", "LocalVLLMExecutor"), ("sglang", "SGLangLocalExecutor")])
@pytest.mark.parametrize("stage", ["init", "generate"])
@pytest.mark.parametrize("forged", [False, True])
def test_native_prompt_diagnostic_reaches_public_summary(monkeypatch, engine, executor_name, stage, forged):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    module = importlib.import_module("vane.execution." + engine)
    executor = getattr(module, executor_name)

    def fail():
        if forged:
            secret_type = type("private_fixture_api_key_ABC123", (Exception,), {})
            raise RuntimeError("vllm submit failed: DiagnosticError: " + str(DiagnosticError("fixture", secret_type())))
        raise TypeError("ServerArgs.__init__() got an unexpected keyword argument 'max_model_len'")

    def create(self):
        if stage == "init":
            fail()
        self.llm = object()

    async def generate(self, prompt, request_id):
        fail()

    monkeypatch.setattr(executor, "_materialize_sampling_params", lambda *_: {})
    monkeypatch.setattr(executor, "_create_engine", create)
    monkeypatch.setattr(executor, "_run_generate", generate)
    monkeypatch.setattr(executor, "_shutdown_engine", lambda self: None)
    options = _build_native_vllm_options_argument({"use_ray": False}, engine=engine)
    with vane.connect(config={"threads": 1}) as connection:
        with pytest.raises(vane.InvalidInputException) as caught:
            connection.execute("SELECT vllm('hello', 'fixture', ?)", [options]).fetchall()
    summary = summarize_error(caught.value)
    if forged:
        assert "RuntimeError" in summary
        assert "private_fixture" not in summary
    else:
        assert "max_model_len" in summary
        assert "TypeError" in summary
