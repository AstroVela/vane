# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""SGLang backend executor tests using a fake in-process engine."""

from __future__ import annotations

import asyncio
import sys
import threading
import types

import pytest

pa = pytest.importorskip("pyarrow")


class _FakeEngine:
    def __init__(self, model_path=None, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self.shutdown_called = False
        self.shutdown_calls = 0
        self.shutdown_loop = None
        self.creation_loop = asyncio.get_running_loop()
        self.generation_loops: list[asyncio.AbstractEventLoop] = []
        self.generate_calls: list[tuple[str, dict[str, object], dict[str, object]]] = []
        self.active_generations = 0
        self.max_active_generations = 0

    async def async_generate(self, prompt, sampling_params, **generate_args):
        assert isinstance(sampling_params, dict)
        self.generation_loops.append(asyncio.get_running_loop())
        self.generate_calls.append((prompt, dict(sampling_params), dict(generate_args)))
        self.active_generations += 1
        self.max_active_generations = max(self.max_active_generations, self.active_generations)
        try:
            await asyncio.sleep(0)
            return {"text": f"generated:{prompt}"}
        finally:
            self.active_generations -= 1

    def shutdown(self):
        self.shutdown_called = True
        self.shutdown_calls += 1
        self.shutdown_loop = asyncio.get_running_loop()


def _install_fake_sglang(monkeypatch):
    """Register a fake SGLang package exposing the public Engine API."""
    sglang_module = types.ModuleType("sglang")
    sglang_module.Engine = _FakeEngine
    monkeypatch.setitem(sys.modules, "sglang", sglang_module)


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
    _install_fake_sglang(monkeypatch)
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
        assert executor.sampling_params == {"max_new_tokens": 8}
        assert executor.llm.generation_loops == [executor.llm.creation_loop]
    finally:
        executor.shutdown()


def test_sglang_local_executor_shutdown_releases_engine(monkeypatch):
    _install_fake_sglang(monkeypatch)
    from vane.execution.sglang import SGLangLocalExecutor

    executor = SGLangLocalExecutor("test-model", {}, {})
    try:
        executor._wait_for_engine_ready_blocking()
        engine = executor.llm
        assert engine is not None
        assert engine.shutdown_called is False
        executor.shutdown()
        assert engine.shutdown_called is True
        assert engine.shutdown_calls == 1
        assert engine.shutdown_loop is engine.creation_loop
        assert executor.llm is None
        assert executor.loop_thread.is_alive() is False
        executor.shutdown()
        assert engine.shutdown_calls == 1
    finally:
        executor.shutdown()


def test_sglang_shutdown_waits_for_initialization_and_closes_late_engine(monkeypatch):
    init_started = threading.Event()
    allow_init = threading.Event()
    instances = []

    class SlowEngine(_FakeEngine):
        def __init__(self, model_path=None, **kwargs):
            init_started.set()
            assert allow_init.wait(timeout=2)
            super().__init__(model_path=model_path, **kwargs)
            instances.append(self)

    sglang_module = types.ModuleType("sglang")
    sglang_module.Engine = SlowEngine
    monkeypatch.setitem(sys.modules, "sglang", sglang_module)

    from vane.execution.sglang import SGLangLocalExecutor

    executor = SGLangLocalExecutor("test-model", {}, {})
    assert init_started.wait(timeout=2)

    shutdown_thread = threading.Thread(target=executor.shutdown)
    shutdown_thread.start()
    shutdown_thread.join(timeout=0.05)
    assert shutdown_thread.is_alive()

    allow_init.set()
    shutdown_thread.join(timeout=2)

    assert shutdown_thread.is_alive() is False
    assert len(instances) == 1
    engine = instances[0]
    assert engine.shutdown_calls == 1
    assert engine.shutdown_loop is engine.creation_loop
    assert executor.llm is None
    assert executor.loop_thread.is_alive() is False

    with pytest.raises(RuntimeError, match="executor is shutting down"):
        executor.submit(None, ["late"], pa.table({"prompt": ["late"]}))
    with pytest.raises(RuntimeError, match="executor is shutting down"):
        asyncio.run(executor.submit_async(["late"], pa.table({"prompt": ["late"]})))


def test_sglang_shutdown_cancels_generation_before_engine_teardown(monkeypatch):
    generation_started = threading.Event()
    generation_cancelled = threading.Event()

    class BlockingEngine(_FakeEngine):
        async def async_generate(self, prompt, sampling_params, **generate_args):
            del prompt, sampling_params, generate_args
            generation_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                generation_cancelled.set()

        def shutdown(self):
            assert generation_cancelled.is_set()
            super().shutdown()

    sglang_module = types.ModuleType("sglang")
    sglang_module.Engine = BlockingEngine
    monkeypatch.setitem(sys.modules, "sglang", sglang_module)

    from vane.execution.sglang import SGLangLocalExecutor

    executor = SGLangLocalExecutor("test-model", {}, {})
    executor.submit(None, ["blocked"], pa.table({"prompt": ["blocked"]}))
    assert generation_started.wait(timeout=2)
    engine = executor.llm

    executor.shutdown()

    assert generation_cancelled.is_set()
    assert engine.shutdown_calls == 1
    assert executor.running_task_count == 0
    assert executor.loop_thread.is_alive() is False


def test_sglang_generate_forwards_generate_args(monkeypatch):
    _install_fake_sglang(monkeypatch)
    from vane.execution.sglang import SGLangLocalExecutor

    executor = SGLangLocalExecutor(
        "test-model",
        {},
        {"sampling_params": {"max_new_tokens": 8}, "return_logprob": True},
    )
    try:
        rows = pa.table({"prompt": ["hello"]})
        executor.submit(None, ["hello"], rows)
        executor.finished_submitting()
        assert executor.wait_for_result()
        outputs, _out_rows = executor.take_ready_result()
        assert outputs == ["generated:hello"]
        assert executor.llm.generate_calls[0][1] == {"max_new_tokens": 8}
        assert executor.llm.generate_calls[0][2] == {"return_logprob": True}
    finally:
        executor.shutdown()


def test_sglang_async_generation_can_overlap_on_engine_loop(monkeypatch):
    _install_fake_sglang(monkeypatch)
    from vane.execution.sglang import SGLangLocalExecutor

    executor = SGLangLocalExecutor("test-model", {}, {})
    try:
        rows = pa.table({"prompt": ["hello", "world"]})
        executor.submit(None, ["hello", "world"], rows)
        executor.finished_submitting()

        outputs = []
        while not executor.all_tasks_finished():
            assert executor.wait_for_result()
            result = executor.take_ready_result()
            if result is not None:
                ready_outputs, _out_rows = result
                outputs.extend(ready_outputs)

        assert sorted(outputs) == ["generated:hello", "generated:world"]
        assert executor.llm.max_active_generations == 2
        assert executor.llm.generation_loops == [executor.llm.creation_loop, executor.llm.creation_loop]
    finally:
        executor.shutdown()


def test_sglang_engine_dispatch_via_sql(monkeypatch):
    _install_fake_sglang(monkeypatch)
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
    _install_fake_sglang(monkeypatch)
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


def test_sglang_plan_lowers_prompt_controls_into_sampling_params():
    from vane.ai.providers.sglang import NativeSGLangPromptPlan

    plan = NativeSGLangPromptPlan(
        sglang_options={
            "actor_number": 3,
            "batch_size": 7,
            "max_retries": 0,
            "max_tokens": 32,
            "max_new_tokens": 40,
            "temperature": 0.7,
        },
        return_format={"type": "object"},
        on_error="ignore",
    )
    options = plan.build_physical_vllm_options()

    assert options["use_threading"] is True
    assert options["_force_background_thread"] is True
    assert options["on_error"] == "null"
    assert options["concurrency"] == 3
    assert options["batch_size"] == 7
    assert "actor_number" not in options
    assert "max_retries" not in options

    sampling = options["generate_args"]["sampling_params"]
    assert sampling["max_new_tokens"] == 40  # max_new_tokens wins over max_tokens
    assert sampling["temperature"] == 0.7
    assert sampling["json_schema"] == '{"type": "object"}'


def test_sglang_builder_forwards_background_loop_controls(monkeypatch):
    import vane.execution.sglang as sglang
    from vane.ai.providers.vllm import _build_native_vllm_options_argument

    captured = {}
    executor = object()

    def create_local_executor(model, engine_args, generate_args, **kwargs):
        captured.update(model=model, engine_args=engine_args, generate_args=generate_args, kwargs=kwargs)
        return executor

    monkeypatch.setattr(sglang, "SGLangLocalExecutor", create_local_executor)
    envelope = _build_native_vllm_options_argument(
        {"use_threading": True, "_force_background_thread": True},
        engine="sglang",
    )

    assert sglang.build_executor("test-model", envelope) is executor
    assert captured["kwargs"]["use_threading"] is True
    assert captured["kwargs"]["force_background_thread"] is True


def test_sglang_secrets_restore_before_named_pool_creation(monkeypatch):
    import vane.execution.vllm as vllm
    from vane.ai._redaction import Secret
    from vane.ai.providers.vllm import _build_native_vllm_options_argument

    envelope = _build_native_vllm_options_argument(
        {
            "engine_args": {"hf_token": Secret("hf_SGLANG-ENGINE-TOKEN")},
            "generate_args": {"api_key": Secret("sk-SGLANG-GENERATE-KEY")},
        },
        engine="sglang",
    )
    envelope.update(use_ray=True, ray_worker_only=True, ray_actor_pool_name="sglang-secret-pool")

    class Plan:
        def collect_vllm_nodes(self, conn=None):
            return [{"model": "secret-model", "pool_name": "sglang-secret-pool", "options": envelope}]

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

    created, leases = vllm.ensure_named_vllm_pools_for_plan(Plan(), session_config={})

    assert created == [actors]
    assert leases == {}
    assert captured["engine_args"]["hf_token"] == "hf_SGLANG-ENGINE-TOKEN"
    assert captured["generate_args"]["api_key"] == "sk-SGLANG-GENERATE-KEY"
