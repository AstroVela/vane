# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
import uuid
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
    import duckdb
    import duckdb.execution.vllm as vllm

    executor = executor or _RecordingExecutor()
    monkeypatch.setattr(vllm, "build_executor", lambda *_args, **_kwargs: executor)
    con = duckdb.connect()
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
    import duckdb
    import duckdb.execution.vllm as vllm

    invalid = vllm.normalize_options({})
    invalid["batch_size"] = 0
    monkeypatch.setattr(vllm, "normalize_options", lambda _options: invalid)
    monkeypatch.setattr(vllm, "build_executor", lambda *_args, **_kwargs: _RecordingExecutor())

    con = duckdb.connect()
    try:
        with pytest.raises(Exception, match="batch_size"):
            con.execute("SELECT vllm('hello', 'recording-model')").fetchall()
    finally:
        con.close()


def test_native_bridge_normalizes_decimal_struct_options(monkeypatch):
    import duckdb
    import duckdb.execution.vllm as vllm

    executor = _RecordingExecutor()
    normalized = {}

    def build_executor(_model, options):
        normalized.update(options)
        return executor

    monkeypatch.setattr(vllm, "build_executor", build_executor)

    con = duckdb.connect()
    try:
        row = con.execute(
            """
            SELECT vllm(
                'hello',
                'recording-model',
                struct_pack(
                    prefix_match_threshold := 0.33,
                    gpus_per_actor := 0.25,
                    engine_init_timeout_s := 1.5
                )
            )
            """
        ).fetchone()
    finally:
        con.close()

    assert row == ("generated:hello",)
    assert normalized["prefix_match_threshold"] == pytest.approx(0.33)
    assert normalized["gpus_per_actor"] == pytest.approx(0.25)
    assert normalized["engine_init_timeout_s"] == pytest.approx(1.5)
    assert type(normalized["prefix_match_threshold"]) is float
    assert type(normalized["gpus_per_actor"]) is float
    assert type(normalized["engine_init_timeout_s"]) is float


def test_native_bridge_preserves_sql_boolean_struct_options(monkeypatch):
    import duckdb
    import duckdb.execution.vllm as vllm

    executor = _RecordingExecutor()
    normalized = {}

    def build_executor(_model, options):
        normalized.update(options)
        return executor

    monkeypatch.setattr(vllm, "build_executor", build_executor)
    option_names = (
        "do_prefix_routing",
        "use_ray",
        "use_threading",
        "require_ray_worker",
        "ray_worker_only",
    )

    con = duckdb.connect()
    try:
        row = con.execute(
            """
            SELECT vllm(
                'hello',
                'recording-model',
                struct_pack(
                    do_prefix_routing := FALSE,
                    use_ray := FALSE,
                    use_threading := FALSE,
                    require_ray_worker := FALSE,
                    ray_worker_only := FALSE
                )
            )
            """
        ).fetchone()
    finally:
        con.close()

    assert row == ("generated:hello",)
    assert {name: normalized[name] for name in option_names} == dict.fromkeys(option_names, False)
    assert all(type(normalized[name]) is bool for name in option_names)


@pytest.mark.parametrize(
    "name",
    [
        "use_ray",
        "use_threading",
        "require_ray_worker",
        "ray_worker_only",
        "_force_background_thread",
    ],
)
def test_native_bridge_rejects_non_boolean_execution_options(monkeypatch, name):
    import duckdb
    import duckdb.execution.vllm as vllm

    monkeypatch.setattr(vllm, "build_executor", lambda *_args, **_kwargs: _RecordingExecutor())

    con = duckdb.connect()
    try:
        with pytest.raises(Exception, match=rf"vllm {name} must be a boolean"):
            con.execute(f"SELECT vllm('hello', 'recording-model', struct_pack({name} := 'false'))").fetchall()
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


def test_distributed_collection_preserves_an_explicit_named_pool():
    import duckdb

    con = duckdb.connect()
    try:
        pool_name = "explicit-shared-vllm-pool"
        options = json.dumps({"use_ray": True, "ray_actor_pool_name": pool_name}, separators=(",", ":"))
        relation = con.sql(
            "SELECT vllm(prompt, 'model', '" + options + "') AS generated FROM (VALUES ('hello')) input(prompt)"
        )
        plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, str(uuid.uuid4())).to_physical_plan(con)

        nodes = plan.collect_vllm_nodes(conn=con)

        assert len(nodes) == 1
        assert nodes[0]["pool_name"] == pool_name
        assert json.loads(nodes[0]["options"])["ray_actor_pool_name"] == pool_name
    finally:
        con.close()


@pytest.mark.parametrize(
    "context",
    ["ctas", "insert_select", "scalar_subquery", "explain_analyze", "prepared"],
)
@pytest.mark.timeout(30)
def test_native_finalizer_has_scheduler_wakeup_in_materialized_and_nested_contexts(monkeypatch, context):
    import duckdb
    import duckdb.execution.vllm as vllm

    executor = _DeferredWakeupExecutor()
    monkeypatch.setattr(vllm, "build_executor", lambda *_args, **_kwargs: executor)
    publisher_errors = []

    def publish_after_arm() -> None:
        try:
            assert executor.pending_ready.wait(timeout=20), "native producer did not submit a pending result"
            assert executor.callback_armed.wait(timeout=20), "native finalizer did not arm a wakeup callback"
            executor.publish_results()
        except BaseException as exc:
            publisher_errors.append(exc)

    publisher = threading.Thread(target=publish_after_arm, name=f"vllm-{context}-result-publisher")
    publisher.start()
    con = duckdb.connect()
    try:
        con.register("vllm_input", pa.table({"prompt": ["hello"]}))
        expression = "vllm(prompt, 'recording-model')"
        if context == "ctas":
            con.execute(f"CREATE TABLE vllm_output AS SELECT {expression} AS generated FROM vllm_input")
        elif context == "insert_select":
            con.execute("CREATE TABLE vllm_output(generated VARCHAR)")
            con.execute(f"INSERT INTO vllm_output SELECT {expression} FROM vllm_input")
        elif context == "scalar_subquery":
            con.execute("SELECT (SELECT vllm('hello', 'recording-model'))").fetchall()
        elif context == "explain_analyze":
            con.execute(f"EXPLAIN ANALYZE SELECT {expression} FROM vllm_input").fetchall()
        else:
            con.execute("PREPARE vllm_statement AS SELECT vllm(CAST($1 AS VARCHAR), 'recording-model')")
            con.execute("EXECUTE vllm_statement('hello')").fetchall()
    finally:
        con.close()
        publisher.join(timeout=5)

    assert not publisher.is_alive()
    assert publisher_errors == []
    assert [submission[1] for submission in executor.submissions] == [("hello",)]
    assert executor.wakeup_registrations >= 1
    assert executor.callback_invocations >= 1
    assert executor.finished_count == 1
    assert not executor.pending
    assert not executor.invalid_wait


def test_native_bridge_rejects_executor_without_wakeup_callback(monkeypatch):
    import duckdb
    import duckdb.execution.vllm as vllm

    monkeypatch.setattr(vllm, "build_executor", lambda *_args, **_kwargs: object())

    con = duckdb.connect()
    try:
        with pytest.raises(Exception, match="vllm executor must implement register_wakeup_callback"):
            con.execute("SELECT vllm('hello', 'model')").fetchall()
    finally:
        con.close()
