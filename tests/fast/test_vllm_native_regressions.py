# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections import deque

import pytest

pa = pytest.importorskip("pyarrow")


class _RecordingExecutor:
    def __init__(self) -> None:
        self.submissions: list[tuple[str | None, tuple[str, ...]]] = []
        self.ready = deque()
        self.finished = False
        self.finished_count = 0
        self.invalid_wait = False
        self.wakeup_callbacks = []
        self.wakeup_registrations = 0

    def submit(self, prefix, prompts, rows) -> None:
        prompt_values = tuple(prompts)
        self.submissions.append((prefix, prompt_values))
        self.ready.append(([f"generated:{prompt}" for prompt in prompt_values], rows))
        self._notify_wakeups()

    def take_ready_result(self):
        try:
            return self.ready.popleft()
        except IndexError:
            return None

    def finished_submitting(self) -> None:
        self.finished_count += 1
        self.finished = True
        self._notify_wakeups()

    def all_tasks_finished(self) -> bool:
        return self.finished and not self.ready

    def wait_for_result(self) -> None:
        if not self.ready and not self.finished:
            self.invalid_wait = True
            raise AssertionError("wait_for_result called with no inflight work")

    def register_wakeup_callback(self, callback) -> bool:
        self.wakeup_registrations += 1
        if self.ready or self.all_tasks_finished():
            return False
        self.wakeup_callbacks.append(callback)
        return True

    def _notify_wakeups(self) -> None:
        if not self.ready and not self.all_tasks_finished():
            return
        callbacks, self.wakeup_callbacks = self.wakeup_callbacks, []
        for callback in callbacks:
            callback()

    def shutdown(self) -> None:
        self.finished = True


class _DeferredWakeupExecutor(_RecordingExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.pending = deque()
        self.callback_armed = threading.Event()
        self.pending_ready = threading.Event()
        self.callback_invocations = 0

    def submit(self, prefix, prompts, rows) -> None:
        prompt_values = tuple(prompts)
        self.submissions.append((prefix, prompt_values))
        self.pending.append(([f"generated:{prompt}" for prompt in prompt_values], rows))
        self.pending_ready.set()

    def all_tasks_finished(self) -> bool:
        return self.finished and not self.pending and not self.ready

    def register_wakeup_callback(self, callback) -> bool:
        self.wakeup_registrations += 1
        if self.ready or self.all_tasks_finished():
            return False
        self.wakeup_callbacks.append(callback)
        self.callback_armed.set()
        return True

    def publish_results(self) -> None:
        self.ready.extend(self.pending)
        self.pending.clear()
        callbacks, self.wakeup_callbacks = self.wakeup_callbacks, []
        for callback in callbacks:
            self.callback_invocations += 1
            callback()


def _run_recording_sql(monkeypatch, prompts, options, *, executor=None, threads=1):
    import vane
    import vane.execution.vllm as vllm

    executor = executor or _RecordingExecutor()
    monkeypatch.setattr(vllm, "build_executor", lambda *_args, **_kwargs: executor)
    con = vane.connect()
    try:
        con.execute(f"PRAGMA threads={threads}")
        con.register(
            "vllm_input",
            pa.table(
                {
                    "id": pa.array(range(len(prompts)), type=pa.int64()),
                    "prompt": pa.array(list(prompts), type=pa.string()),
                }
            ),
        )
        encoded = json.dumps(options, separators=(",", ":"))
        rows = con.execute(
            "SELECT id, prompt, vllm(prompt, 'recording-model', '" + encoded + "') AS generated FROM vllm_input"
        ).fetchall()
        return executor, rows
    finally:
        con.close()


@pytest.mark.parametrize(
    ("prompts", "expected_prefix"),
    [
        (["abc1", "abc2"], "abc"),
        (["你好甲", "你好乙"], "你好"),
        (["🙂alpha", "🙂alpine"], "🙂alp"),
        (["same", "same"], "same"),
        (["alpha", "zulu"], None),
        (["", ""], None),
    ],
)
def test_native_bucket_prefix_ends_on_a_complete_utf8_character(monkeypatch, prompts, expected_prefix):
    executor, rows = _run_recording_sql(
        monkeypatch,
        prompts,
        {
            "do_prefix_routing": True,
            "max_buffer_size": 0,
            "min_bucket_size": 2,
            "prefix_match_threshold": 0.3,
            "batch_size": None,
            "inflight_limit": 0,
        },
    )

    assert [submission[0] for submission in executor.submissions] == [expected_prefix]
    assert {row[0]: row[2] for row in rows} == {index: f"generated:{prompt}" for index, prompt in enumerate(prompts)}


def test_native_bridge_rejects_zero_batch_size_even_if_python_normalization_is_bypassed(monkeypatch):
    import vane
    import vane.execution.vllm as vllm

    invalid = vllm.normalize_options({})
    invalid["batch_size"] = 0
    monkeypatch.setattr(vllm, "normalize_options", lambda _options: invalid)
    monkeypatch.setattr(vllm, "build_executor", lambda *_args, **_kwargs: _RecordingExecutor())

    con = vane.connect()
    try:
        with pytest.raises(Exception, match="batch_size"):
            con.execute("SELECT vllm('hello', 'recording-model')").fetchall()
    finally:
        con.close()


@pytest.mark.timeout(30)
def test_native_finalizer_blocks_and_resumes_through_a_one_shot_callback(monkeypatch):
    executor = _DeferredWakeupExecutor()
    publisher_errors = []

    def publish_after_arm() -> None:
        try:
            assert executor.callback_armed.wait(timeout=20), "native finalizer did not arm a wakeup callback"
            assert executor.pending_ready.wait(timeout=20), "native producer did not submit a pending result"
            executor.publish_results()
        except BaseException as exc:
            publisher_errors.append(exc)

    publisher = threading.Thread(target=publish_after_arm, name="vllm-test-result-publisher")
    publisher.start()
    try:
        _, rows = _run_recording_sql(
            monkeypatch,
            ["prefix-alpha", "prefix-beta"],
            {
                "do_prefix_routing": True,
                "max_buffer_size": 0,
                "min_bucket_size": 2,
                "batch_size": None,
                "inflight_limit": 0,
            },
            executor=executor,
            threads=2,
        )
    finally:
        publisher.join(timeout=5)

    assert not publisher.is_alive()
    assert publisher_errors == []
    assert {row[0]: row[2] for row in rows} == {
        0: "generated:prefix-alpha",
        1: "generated:prefix-beta",
    }
    assert executor.wakeup_registrations >= 1
    assert executor.callback_invocations >= 1
    assert executor.finished_count == 1
    assert not executor.pending
    assert not executor.invalid_wait


_PARALLEL_FINALIZER_SCRIPT = r"""
import json
import threading
from collections import deque

import vane
import vane.execution.vllm as vllm


class Fake:
    def __init__(self):
        self.condition = threading.Condition()
        self.pending = deque()
        self.ready = deque()
        self.finished = False
        self.finished_count = 0
        self.shutdown_count = 0
        self.wait_count = 0
        self.finished_event = threading.Event()
        self.waiters_ready = threading.Event()
        self.submit_threads = set()

    def submit(self, _prefix, prompts, rows):
        values = tuple(prompts)
        self.submit_threads.add(threading.get_ident())
        with self.condition:
            self.pending.append((["generated:" + value for value in values], rows))

    def take_ready_result(self):
        with self.condition:
            return self.ready.popleft() if self.ready else None

    def finished_submitting(self):
        self.finished_count += 1
        self.finished = True
        self.finished_event.set()

    def all_tasks_finished(self):
        with self.condition:
            return self.finished and not self.pending and not self.ready

    def wait_for_result(self):
        with self.condition:
            self.wait_count += 1
            wait_for_shutdown = self.wait_count == 1
            if self.wait_count >= 2:
                self.waiters_ready.set()
            predicate = (lambda: self.shutdown_count) if wait_for_shutdown else (lambda: self.ready or self.shutdown_count)
            ready = self.condition.wait_for(predicate, timeout=20)
            if not ready:
                raise AssertionError("legacy finalizer wait was not released")

    def publish_results(self):
        with self.condition:
            self.ready.extend(self.pending)
            self.pending.clear()
            self.condition.notify_all()

    def shutdown(self):
        with self.condition:
            self.shutdown_count += 1
            self.condition.notify_all()


executor = Fake()
vllm.build_executor = lambda *_args, **_kwargs: executor
publisher_errors = []


def publish_after_legacy_wait():
    try:
        assert executor.finished_event.wait(timeout=20), "native producers did not finish"
        assert executor.waiters_ready.wait(timeout=20), "native finalizers did not enter concurrent legacy waits"
        executor.publish_results()
    except BaseException as exc:
        publisher_errors.append(exc)


publisher = threading.Thread(target=publish_after_legacy_wait, name="vllm-legacy-result-publisher")
publisher.start()
options = json.dumps(
    {
        "do_prefix_routing": True,
        "max_buffer_size": 0,
        "min_bucket_size": 1,
        "batch_size": None,
        "inflight_limit": 0,
    },
    separators=(",", ":"),
)
con = vane.connect()
try:
    con.execute("PRAGMA threads=2")
    con.execute(
        "CREATE TABLE parallel_input AS SELECT i::BIGINT id, "
        "'prefix-' || i::VARCHAR prompt FROM range(300000) t(i)"
    )
    count = con.execute(
        "SELECT count(generated) FROM (SELECT vllm(prompt, 'model', '"
        + options
        + "') AS generated FROM parallel_input) q"
    ).fetchone()[0]
finally:
    con.close()
    publisher.join(timeout=5)

assert count == 300000
assert not publisher.is_alive()
assert publisher_errors == []
assert len(executor.submit_threads) == 2
assert executor.finished_count == 1
assert executor.wait_count >= 1
assert executor.shutdown_count == 1
"""


def test_parallel_legacy_finalizer_wait_survives_shutdown():
    completed = subprocess.run(
        [sys.executable, "-c", _PARALLEL_FINALIZER_SCRIPT],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
