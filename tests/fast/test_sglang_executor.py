# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""SGLang backend executor tests using a fake in-process engine."""

from __future__ import annotations

import sys
import types

import pytest

pa = pytest.importorskip("pyarrow")


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeEngine:
    def __init__(self, model_path=None, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs

    def generate(self, prompt, sampling_params):
        del sampling_params
        return {"text": f"generated:{prompt}"}


def _fake_sglang():
    return types.SimpleNamespace(Engine=_FakeEngine, SamplingParams=_FakeSamplingParams)


def test_sglang_hierarchy_and_normalize():
    from vane.execution._llm_executor import LLMExecutor
    from vane.execution.sglang import SGLangExecutor, normalize_options

    assert issubclass(SGLangExecutor, LLMExecutor)

    from vane.ai.providers.vllm import _build_native_vllm_options_argument

    envelope = _build_native_vllm_options_argument({})
    envelope["engine"] = "sglang"
    normalized = normalize_options(envelope)
    assert normalized["do_prefix_routing"] is False
    assert normalized["engine_args"] == {}
    assert normalized["generate_args"] == {}


def test_sglang_local_executor_roundtrip(monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang", _fake_sglang())
    from vane.execution.sglang import SGLangLocalExecutor

    executor = SGLangLocalExecutor("test-model", {}, {"sampling_params": {"max_new_tokens": 8}})
    try:
        rows = pa.table({"prompt": ["hello"]})
        executor.submit(None, ["hello"], rows)
        executor.finished_submitting()
        assert executor.wait_for_result()
        outputs, out_rows = executor.take_ready_result()
        assert outputs == ["generated:hello"]
        assert out_rows.num_rows == 1
    finally:
        executor.shutdown()


def test_sglang_engine_dispatch_via_sql(monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang", _fake_sglang())
    import vane
    from vane.ai.providers.vllm import _build_native_vllm_options_argument

    envelope = _build_native_vllm_options_argument({})
    envelope["engine"] = "sglang"
    con = vane.connect()
    try:
        con.register("sglang_input", pa.table({"prompt": ["hello", "world"]}))
        rows = con.execute(
            "SELECT vllm(prompt, 'recording-model', ?) AS generated FROM sglang_input",
            [envelope],
        ).fetchall()
        assert [r[0] for r in rows] == ["generated:hello", "generated:world"]
    finally:
        con.close()


def test_sglang_prompt_expression_end_to_end(monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang", _fake_sglang())
    import vane

    conn = vane.connect()
    try:
        conn.execute("PRAGMA threads=1")
        rel = conn.sql("SELECT * FROM (VALUES (1, 'Alpha'), (2, 'Beta')) source(id, chunk)")
        result = rel.select(
            vane.col("id"),
            vane.ai.prompt(vane.col("chunk"), provider="sglang", model="test-model").alias("generated"),
        ).order("id")
        rows = result.fetchall()
        assert rows == [(1, "generated:Alpha"), (2, "generated:Beta")]
    finally:
        conn.close()


def test_sglang_driver_precreation_dispatches_by_engine(monkeypatch):
    import vane.execution.vllm as vllm
    from vane.ai.providers.vllm import _build_native_vllm_options_argument

    envelope = _build_native_vllm_options_argument({})
    envelope["engine"] = "sglang"
    envelope.update(use_ray=True, ray_worker_only=True, ray_actor_pool_name="sglang-pool")

    class Plan:
        def collect_vllm_nodes(self, conn=None):
            return [{"model": "test-model", "pool_name": "sglang-pool", "options": envelope}]

    fake_ray = types.ModuleType("ray")
    fake_ray.is_initialized = lambda: True
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.delenv("VANE_WORKER", raising=False)

    captured = {}
    actors = object()

    def get_or_create_named(_cls, **kwargs):
        captured.update(kwargs)
        return actors

    monkeypatch.setattr(vllm.LLMActors, "get_or_create_named", classmethod(get_or_create_named))

    created, _leases = vllm.ensure_named_vllm_pools_for_plan(Plan(), session_config={})

    from vane.execution.sglang import SGLangRayLocalExecutor

    assert created == [actors]
    assert captured["actor_cls"] is SGLangRayLocalExecutor
    assert captured["name_prefix"] == "sglang-pool"
    assert captured["engine_args"] == {}
    assert captured["generate_args"] == {}


def test_sglang_ray_local_executor_hierarchy():
    import inspect

    from vane.execution._llm_executor import RayActorExecutorMixin
    from vane.execution.sglang import SGLangLocalExecutor, SGLangRayLocalExecutor

    assert issubclass(SGLangRayLocalExecutor, SGLangLocalExecutor)
    assert issubclass(SGLangRayLocalExecutor, RayActorExecutorMixin)
    assert inspect.iscoroutinefunction(SGLangRayLocalExecutor.wait_for_result)
