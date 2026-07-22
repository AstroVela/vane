# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Executor lifecycle contract for ``VLLMPrompter`` (#143).

The prompter caches one executor for its lifetime.  ``finished_submitting()``
must NOT run per ``prompt()``/``prompt_batch()`` call: in Ray-remote mode it
permanently retires the executor id on every actor, so the second call would
fail with "vllm executor <id> is already finished".  It runs exactly once, at
``close()``/teardown, via ``executor.shutdown()``.

The fake executor below enforces the Ray-remote contract (submit after finish
raises), so these tests fail against the old per-call lifecycle.
"""

import asyncio

import pytest

from vane.ai.providers.vllm import VLLMPrompter


class FakeExecutor:
    """Records lifecycle calls; serves one result per submitted prompt.

    Mirrors the Ray-remote ``RemoteVLLMExecutor`` contract: once submission
    is finished (via ``finished_submitting`` or ``shutdown``), any further
    ``submit`` raises — exactly the failure mode of issue #143.
    """

    def __init__(self):
        self.calls = []
        self.pending = []
        self.finished = False
        self.shutdown_calls = 0

    def submit(self, prefix, prompts, rows):
        if self.finished:
            raise RuntimeError("vllm executor fake-id is already finished")
        self.calls.append(("submit", list(prompts)))
        for i in range(rows.num_rows):
            self.pending.append((prompts[i], rows.slice(i, 1)))

    def finished_submitting(self):
        self.calls.append(("finished_submitting",))
        self.finished = True

    def take_ready_result(self):
        if not self.pending:
            return None
        prompt, row = self.pending.pop(0)
        return [f"echo:{prompt}"], row

    def all_tasks_finished(self):
        return not self.pending

    def wait_for_result(self):
        self.calls.append(("wait_for_result",))

    def shutdown(self):
        self.calls.append(("shutdown",))
        self.shutdown_calls += 1
        self.finished = True


def _prompter_with_fake():
    prompter = VLLMPrompter(model="test-model")
    executor = FakeExecutor()
    prompter._executor = executor
    return prompter, executor


def test_two_consecutive_prompt_batch_calls_both_submit():
    """Second (and later) batches must submit on the same cached executor."""
    prompter, executor = _prompter_with_fake()

    first = prompter.prompt_batch(["a", "b"])
    second = prompter.prompt_batch(["c", "d"])

    assert first == ["echo:a", "echo:b"]
    assert second == ["echo:c", "echo:d"]
    submits = [call for call in executor.calls if call[0] == "submit"]
    assert submits == [("submit", ["a", "b"]), ("submit", ["c", "d"])]


def test_prompt_batch_does_not_finish_submitting_per_call():
    """finished_submitting must not run per call — only at teardown."""
    prompter, executor = _prompter_with_fake()

    prompter.prompt_batch(["a", "b"])
    prompter.prompt_batch(["c"])

    assert ("finished_submitting",) not in executor.calls
    assert ("shutdown",) not in executor.calls


def test_consecutive_single_prompt_calls_both_submit():
    """prompt() has the same deferred lifecycle as prompt_batch()."""
    prompter, executor = _prompter_with_fake()

    first = asyncio.run(prompter.prompt(("hello",)))
    second = asyncio.run(prompter.prompt(("again",)))

    assert first == "echo:hello"
    assert second == "echo:again"
    assert ("finished_submitting",) not in executor.calls


def test_close_shuts_down_executor_exactly_once():
    """close() finishes the executor via shutdown() and is idempotent."""
    prompter, executor = _prompter_with_fake()

    prompter.prompt_batch(["a"])
    prompter.close()

    assert executor.shutdown_calls == 1
    assert prompter._executor is None

    prompter.close()
    assert executor.shutdown_calls == 1


def test_del_shuts_down_executor():
    """GC teardown releases the executor without raising."""
    prompter, executor = _prompter_with_fake()

    prompter.__del__()

    assert executor.shutdown_calls == 1
    assert prompter._executor is None


def test_del_swallows_shutdown_errors():
    """Destructors must not raise even if shutdown fails."""
    prompter, executor = _prompter_with_fake()

    def boom():
        raise RuntimeError("shutdown failed")

    executor.shutdown = boom
    prompter.__del__()  # must not raise


def test_submit_after_close_surfaces_finished_error():
    """close() must drop the executor reference so a fresh one is built next call.

    Guards the invariant that close() clears the cache rather than leaving
    a finished executor behind that would poison later calls.
    """
    prompter, executor = _prompter_with_fake()
    prompter.close()

    # The cached (now finished) executor must not be reused.
    assert prompter._executor is None
    with pytest.raises(RuntimeError, match="already finished"):
        executor.submit(None, ["x"], _one_row_table())


def _one_row_table():
    import pyarrow as pa

    return pa.table({"_idx": [0]})


# ---------------------------------------------------------------------------
# _PromptBatch delegates teardown to the prompter (explicit close hook)
# ---------------------------------------------------------------------------


def _load_real_functions():
    """Import the real ``vane.ai.functions``, even under the no-duckdb harness.

    The stub harness registers a placeholder for ``vane.ai.functions``; evict
    it and stub just the duckdb-importing dependencies so the real module
    under test can load. On CI (where duckdb imports fine) nothing is stubbed.
    """
    import importlib
    import sys
    import types

    module = sys.modules.get("vane.ai.functions")
    if getattr(module, "__file__", None):
        return module
    if module is not None:
        sys.modules.pop("vane.ai.functions")
        stub_specs = (
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


class CloseRecordingPrompter:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def test_prompt_batch_close_closes_prompter_once():
    functions = _load_real_functions()
    wrapper = functions._PromptBatch(object(), "messages", "response", None)
    prompter = CloseRecordingPrompter()
    wrapper._prompter = prompter

    wrapper.close()
    assert prompter.close_calls == 1
    assert wrapper._prompter is None

    wrapper.close()  # idempotent
    assert prompter.close_calls == 1


def test_prompt_batch_close_tolerates_prompters_without_close():
    functions = _load_real_functions()
    wrapper = functions._PromptBatch(object(), "messages", "response", None)
    wrapper._prompter = object()
    wrapper.close()
    assert wrapper._prompter is None


def test_prompt_batch_close_before_first_use_is_noop():
    functions = _load_real_functions()
    wrapper = functions._PromptBatch(object(), "messages", "response", None)
    wrapper.close()
    assert wrapper._prompter is None


def test_prompt_batch_del_swallows_close_errors():
    functions = _load_real_functions()
    wrapper = functions._PromptBatch(object(), "messages", "response", None)

    class ExplodingPrompter:
        def close(self):
            raise RuntimeError("teardown boom")

    wrapper._prompter = ExplodingPrompter()
    wrapper.__del__()  # must not raise


# ---------------------------------------------------------------------------
# Actor adapter forwards close() to the wrapped batch object
#
# Actor backends own a _ConfiguredAIBatchActor instance, not the _PromptBatch
# itself, so _PromptBatch.close() is only reachable if the adapter forwards it.
# ---------------------------------------------------------------------------


def _actor_adapter_wrapping_prompt_batch():
    functions = _load_real_functions()
    batch = functions._PromptBatch(object(), "messages", "response", None)
    prompter = CloseRecordingPrompter()
    batch._prompter = prompter
    actor_cls = functions._adapt_batch_wrapper_for_backend(batch, "subprocess_actor")
    return actor_cls(), batch, prompter


def test_actor_adapter_close_delegates_to_wrapped_batch():
    adapter, batch, prompter = _actor_adapter_wrapping_prompt_batch()

    adapter.close()
    assert prompter.close_calls == 1
    assert batch._prompter is None

    adapter.close()  # idempotent through _PromptBatch.close()
    assert prompter.close_calls == 1


def test_actor_adapter_del_delegates_to_wrapped_batch():
    adapter, batch, prompter = _actor_adapter_wrapping_prompt_batch()

    adapter.__del__()
    assert prompter.close_calls == 1
    assert batch._prompter is None


def test_actor_adapter_close_tolerates_wrappers_without_close():
    functions = _load_real_functions()
    actor_cls = functions._adapt_batch_wrapper_for_backend(lambda table: table, "subprocess_actor")
    adapter = actor_cls()
    adapter.close()  # plain-function wrapper: no close attr, must not raise


def test_actor_adapter_del_swallows_close_errors():
    functions = _load_real_functions()

    class ExplodingWrapper:
        def close(self):
            raise RuntimeError("teardown boom")

    actor_cls = functions._adapt_batch_wrapper_for_backend(ExplodingWrapper(), "subprocess_actor")
    actor_cls().__del__()  # must not raise
