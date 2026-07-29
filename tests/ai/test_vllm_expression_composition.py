# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque

import pytest

import duckdb
import vane


class _ImmediateVLLMExecutor:
    def __init__(self, model: str) -> None:
        self.model = model
        self.ready = deque()
        self.finished = False
        self.finished_count = 0
        self.shutdown_count = 0

    def submit(self, _prefix, prompts, rows) -> None:
        outputs = [f"{self.model}::{prompt}" for prompt in prompts]
        self.ready.append((outputs, rows))

    def take_ready_result(self):
        try:
            return self.ready.popleft()
        except IndexError:
            return None

    def finished_submitting(self) -> None:
        self.finished = True
        self.finished_count += 1

    def all_tasks_finished(self) -> bool:
        return self.finished and not self.ready

    def wait_for_result(self) -> None:
        raise AssertionError("immediate vLLM results must not block")

    def register_wakeup_callback(self, _callback) -> bool:
        return False

    def shutdown(self) -> None:
        self.finished = True
        self.shutdown_count += 1


def _record_native_executors(monkeypatch):
    import duckdb.execution.vllm as vllm_executor

    executors: list[_ImmediateVLLMExecutor] = []

    def build_executor(model, _options):
        executor = _ImmediateVLLMExecutor(model)
        executors.append(executor)
        return executor

    monkeypatch.setattr(vllm_executor, "build_executor", build_executor)
    return executors


def test_python_vllm_expressions_support_multiple_nested_outputs(monkeypatch):
    executors = _record_native_executors(monkeypatch)
    conn = vane.connect()
    conn.execute("PRAGMA threads=1")
    rel = conn.sql("SELECT * FROM (VALUES (1, 'Alpha'), (2, 'Beta')) source(id, chunk)")

    lower_prompt = duckdb.FunctionExpression(
        "lower",
        vane.ai.prompt(
            vane.col("chunk"),
            provider="vllm",
            model="PY-UPPER",
            prompt_options={"do_prefix_routing": False},
        ),
    ).alias("lowered")
    wrapped_prompt = duckdb.FunctionExpression(
        "concat",
        duckdb.ConstantExpression("["),
        vane.ai.prompt(
            vane.col("chunk"),
            provider="vllm",
            model="py-wrapped",
            prompt_options={"do_prefix_routing": False},
        ),
        duckdb.ConstantExpression("]"),
    ).alias("wrapped")

    result = rel.select(vane.col("id"), lower_prompt, wrapped_prompt).order("id")

    assert result.explain().count("VLLM_PROJECT") == 2
    assert result.fetchall() == [
        (1, "py-upper::alpha", "[py-wrapped::Alpha]"),
        (2, "py-upper::beta", "[py-wrapped::Beta]"),
    ]
    assert len(executors) == 2
    assert all(executor.finished_count == 1 for executor in executors)
    assert all(executor.shutdown_count == 1 for executor in executors)


def test_python_vllm_expressions_support_chained_native_prompts(monkeypatch):
    executors = _record_native_executors(monkeypatch)
    conn = vane.connect()
    conn.execute("PRAGMA threads=1")
    rel = conn.sql("SELECT * FROM (VALUES (1, 'Alpha'), (2, 'Beta')) source(id, chunk)")

    inner_prompt = vane.ai.prompt(
        vane.col("chunk"),
        provider="vllm",
        model="inner",
        prompt_options={"do_prefix_routing": False},
    )
    chained_prompt = vane.ai.prompt(
        inner_prompt,
        provider="vllm",
        model="outer",
        prompt_options={"do_prefix_routing": False},
    ).alias("chained")

    result = rel.select(vane.col("id"), chained_prompt).order("id")

    assert result.explain().count("VLLM_PROJECT") == 2
    assert result.fetchall() == [
        (1, "outer::inner::Alpha"),
        (2, "outer::inner::Beta"),
    ]
    assert len(executors) == 2
    assert all(executor.finished_count == 1 for executor in executors)
    assert all(executor.shutdown_count == 1 for executor in executors)


def test_sql_ai_prompt_supports_multiple_calls_nested_in_eager_expressions(monkeypatch):
    executors = _record_native_executors(monkeypatch)
    conn = vane.connect()
    conn.execute("PRAGMA threads=1")

    result = conn.sql("""
        SELECT id, concat(
            lower(ai_prompt(
                chunk,
                struct_pack(
                    provider := 'vllm',
                    model := 'SQL-UPPER',
                    do_prefix_routing := false
                )
            )),
            ' / ',
            ai_prompt(
                chunk,
                struct_pack(
                    provider := 'vllm',
                    model := 'sql-plain',
                    do_prefix_routing := false
                )
            )
        ) AS combined
        FROM (VALUES (1, 'Alpha'), (2, 'Beta')) source(id, chunk)
        ORDER BY id
    """)

    assert result.explain().count("VLLM_PROJECT") == 2
    rows = result.fetchall()
    assert rows == [
        (1, "sql-upper::alpha / sql-plain::Alpha"),
        (2, "sql-upper::beta / sql-plain::Beta"),
    ]
    assert len(executors) == 2
    assert all(executor.finished_count == 1 for executor in executors)
    assert all(executor.shutdown_count == 1 for executor in executors)


@pytest.mark.parametrize(
    ("expression", "source"),
    [
        (
            "CASE WHEN flag THEN vllm(chunk, 'conditional-model') ELSE 'skipped' END",
            "(VALUES (false, 'secret')) source(flag, chunk)",
        ),
        (
            "coalesce('ready', vllm(chunk, 'conditional-model'))",
            "(VALUES ('secret')) source(chunk)",
        ),
    ],
)
def test_vllm_expressions_reject_short_circuit_contexts_without_submitting(monkeypatch, expression, source):
    executors = _record_native_executors(monkeypatch)
    conn = vane.connect()

    with pytest.raises(
        Exception,
        match="vllm expressions are not supported inside CASE, AND/OR, or COALESCE short-circuit expressions",
    ):
        conn.sql(f"SELECT {expression} FROM {source}").fetchall()

    assert executors == []
