# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import threading

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


@pytest.mark.parametrize("engine,executor_name", [("vllm", "LocalVLLMExecutor"), ("sglang", "SGLangLocalExecutor")])
@pytest.mark.parametrize("on_error", ["raise", "null", "log"])
@pytest.mark.parametrize("failed", [True, False])
def test_native_prompt_generation_settles_between_drain_and_completion(
    monkeypatch, engine, executor_name, on_error, failed
):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    module = importlib.import_module("vane.execution." + engine)
    executor = getattr(module, executor_name)
    release_generation = threading.Event()
    checked_completion = threading.Event()
    instances = []
    take_ready = executor.take_ready_result
    all_finished = executor.all_tasks_finished

    def create(self):
        self.llm = object()
        instances.append(self)

    async def generate(self, prompt, request_id):
        assert await asyncio.to_thread(release_generation.wait, 5), "native final drain was not reached"
        if failed:
            raise TypeError("ServerArgs.__init__() got an unexpected keyword argument 'max_model_len'")
        return "generated:hello"

    def take(self, *args):
        result = take_ready(self, *args)
        if result is None and self._finished_submitting and not release_generation.is_set():
            assert self.error_message is None
            assert self.running_task_count == 1
            release_generation.set()
        return result

    def finished(self):
        if release_generation.is_set():
            # Let the real generation path publish its result/error after the
            # final drain, before native code observes the terminal task count.
            with self._result_cv:
                assert self._result_cv.wait_for(lambda: self.running_task_count == 0, timeout=5)
            checked_completion.set()
        return all_finished(self)

    monkeypatch.setattr(executor, "_materialize_sampling_params", lambda *_: {})
    monkeypatch.setattr(executor, "_create_engine", create)
    monkeypatch.setattr(executor, "_run_generate", generate)
    monkeypatch.setattr(executor, "_shutdown_engine", lambda self: None)
    monkeypatch.setattr(executor, "take_ready_result", take)
    monkeypatch.setattr(executor, "all_tasks_finished", finished)
    options = _build_native_vllm_options_argument({"use_ray": False, "on_error": on_error}, engine=engine)
    try:
        with vane.connect(config={"threads": 1}) as connection:
            if failed and on_error == "raise":
                with pytest.raises(vane.InvalidInputException) as caught:
                    connection.execute("SELECT vllm('hello', 'fixture', ?)", [options]).fetchall()
                summary = summarize_error(caught.value)
                assert "TypeError" in summary
                assert "max_model_len" in summary
            else:
                rows = connection.execute("SELECT vllm('hello', 'fixture', ?)", [options]).fetchall()
                assert rows == [(None if failed else "generated:hello",)]
    finally:
        release_generation.set()

    assert checked_completion.is_set()
    assert len(instances) == 1
    assert instances[0]._shutdown_complete
    assert not instances[0].loop_thread.is_alive()
