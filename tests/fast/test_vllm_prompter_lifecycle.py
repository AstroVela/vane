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
    """Using a prompter after close() recreates no fake — executor is gone.

    Guards the invariant that close() drops the cached executor reference
    rather than leaving a finished executor that would poison later calls.
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
