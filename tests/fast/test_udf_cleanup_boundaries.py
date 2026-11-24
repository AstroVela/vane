# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real process, cached task and diagnostic boundaries after failed cleanup."""

from __future__ import annotations

import asyncio
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import psutil
import pyarrow as pa
import pytest

from vane import pickle as vane_pickle
from vane.ai import summarize_error


def _alive(pid):
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _engine_payload(marker, backend, detached=False):
    state = {}

    def execute(table):
        if "child" not in state:
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(120)", marker],
                start_new_session=detached,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            state["child"] = child
            Path(marker).write_text(str(child.pid))
        return table

    def close():
        raise OSError(5, "private engine cleanup path")

    class Actor:
        def __call__(self, table):
            return execute(table)

        def _vane_close(self):
            close()

    execute._vane_close = close
    return {
        "function_pickle": vane_pickle.dumps(Actor if backend == "subprocess_actor" else execute),
        "execution_backend": backend,
        "call_mode": "map_batches",
        "output_schema": [{"name": "text", "type": "VARCHAR"}],
    }


def _clean_child(marker):
    if Path(marker).exists():
        try:
            child = psutil.Process(int(Path(marker).read_text()))
            if child.cmdline()[-1:] == [marker]:
                child.kill()
                psutil.wait_procs([child], timeout=2)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass


@pytest.mark.parametrize("backend", ["subprocess_actor", "subprocess_task"])
@pytest.mark.parametrize("detached", [False, True])
@pytest.mark.parametrize("kill_fails", [False, True])
def test_native_children_remain_owned_until_process_cleanup_finishes(
    tmp_path, monkeypatch, backend, detached, kill_fails
):
    from vane.execution.udf_subprocess import _SingleSubprocessExecutor

    marker = str(tmp_path / "engine-child.pid")
    worker = _SingleSubprocessExecutor(_engine_payload(marker, backend, detached))
    group = getattr(worker, "_process_group", None)
    real_close = group.close if group is not None else None
    if kill_fails and group is not None:

        def fail_cleanup(timeout):
            raise OSError(13, "injected child termination refusal")

        monkeypatch.setattr(group, "close", fail_cleanup)
    # An unrelated child shares the application's session, never the worker's.
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        if backend == "subprocess_task":
            with pytest.raises(RuntimeError):
                worker.submit(pa.table({"text": ["hello"]}))
        else:
            worker.submit(pa.table({"text": ["hello"]}))
            with pytest.raises(RuntimeError):
                worker.close(kill=False)
        pid = int(Path(marker).read_text())
        assert unrelated.poll() is None
        if kill_fails:
            assert not worker._cleanup_finished
            assert worker._process_group is group
            assert _alive(pid)
            monkeypatch.setattr(group, "close", real_close)
            worker.close(kill=True)
        assert worker._cleanup_finished
        assert not _alive(pid)
        assert unrelated.poll() is None
    finally:
        if group is not None:
            monkeypatch.setattr(group, "close", real_close)
        try:
            worker.close(kill=True)
        finally:
            _clean_child(marker)
            unrelated.kill()
            unrelated.wait(timeout=5)


@pytest.mark.parametrize("backend", ["subprocess_actor", "subprocess_task"])
def test_runtime_cleanup_errno_survives_real_worker_transport(tmp_path, backend):
    from vane.execution.udf_subprocess import _SingleSubprocessExecutor

    marker = str(tmp_path / "diagnostic-child.pid")
    worker = _SingleSubprocessExecutor(_engine_payload(marker, backend))
    try:
        with pytest.raises(RuntimeError) as caught:
            worker.submit(pa.table({"text": ["hello"]}))
            worker.close(kill=False)
        for error in (caught.value, pickle.loads(pickle.dumps(caught.value))):
            summary = summarize_error(error)
            assert "OSError (errno=5)" in summary
            assert "private engine" not in summary
            assert len(summary) <= 512
    finally:
        try:
            worker.close(kill=True)
        finally:
            _clean_child(marker)


@pytest.mark.parametrize("kind", ["embed", "prompt"])
@pytest.mark.parametrize("clear_cache", [False, True])
def test_cached_ray_task_retries_cleanup_on_the_original_loop(monkeypatch, kind, clear_cache):
    import ray

    from vane.ai.functions import _adapt_batch_wrapper_for_backend, _EmbedTextBatch, _PromptBatch
    from vane.execution import _common
    from vane.execution.udf_ray import _iter_materialized_task_outputs

    _common.clear_udf_callable_cache()
    providers = []
    failures = [2]

    class Provider:
        def __init__(self):
            self.loop = asyncio.get_running_loop()
            self.calls = 0
            self.closes = 0
            self.closed = False

        async def embed_text(self, texts):
            assert asyncio.get_running_loop() is self.loop
            self.calls += 1
            return [[1.0, 2.0] for _ in texts]

        async def prompt(self, messages):
            assert asyncio.get_running_loop() is self.loop
            self.calls += 1
            return "ok"

        async def aclose(self):
            assert asyncio.get_running_loop() is self.loop
            self.closes += 1
            if failures[0]:
                failures[0] -= 1
                raise OSError(5, "private SDK cleanup path")
            self.closed = True

    class Descriptor:
        def instantiate(self):
            provider = Provider()
            providers.append(provider)
            return provider

    wrapper = (
        _EmbedTextBatch(Descriptor(), "text", "out", 2, max_retries=0)
        if kind == "embed"
        else _PromptBatch(Descriptor(), ["text"], "out", max_retries=0)
    )
    function = _adapt_batch_wrapper_for_backend(wrapper, "ray_task")
    # Keep the descriptor observable; the real LRU, task entry, executor,
    # async runtime, provider invocation and close hooks all run normally.
    monkeypatch.setattr(_common, "load_udf_from_payload", lambda _payload: function)
    monkeypatch.setattr(ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "test-node"))
    monkeypatch.setenv("VANE_CPU_UDF_CALLABLE_CACHE", "1")
    payload = {
        "function_pickle": vane_pickle.dumps(function),
        "execution_backend": "ray_task",
        "call_mode": "map_batches",
        "query_id": "cleanup-query",
        "resource_unit_id": "cleanup-udf",
        "task_lease_id": "cleanup-lease",
        "attempt_id": "cleanup-attempt",
        "udf_output_target_max_bytes": 131072,
        "output_window_bytes": 262144,
        "output_schema": [{"name": "out", "type": "FLOAT[2]" if kind == "embed" else "VARCHAR"}],
    }
    table = pa.table({"text": ["hello"]})
    try:
        with pytest.raises(RuntimeError, match="OSError"):
            list(_iter_materialized_task_outputs(payload, [table]))
        first = providers[0]
        assert first.calls == 1 and first.closes == 1
        if clear_cache:
            _common.clear_udf_callable_cache()
        with pytest.raises(RuntimeError, match="OSError"):
            list(_iter_materialized_task_outputs(payload, [table]))
        assert len(providers) == 1
        assert first.calls == 1 and first.closes == 2
        outputs = list(_iter_materialized_task_outputs(payload, [table]))
        assert len(outputs) == 2
        assert len(providers) == 2 and providers[1].calls == 1
        assert first.closes == 3
        assert all(provider.closed and provider.loop.is_closed() for provider in providers)
        assert first.loop is not providers[1].loop
    finally:
        failures[0] = 0
        # Complete retained executor cleanup even if an assertion failed.
        _common.load_udf_from_payload_cached(payload)
        for provider in providers:
            if not provider.loop.is_closed():
                provider.loop.run_until_complete(provider.aclose())
                provider.loop.close()
        _common.clear_udf_callable_cache()


@pytest.mark.parametrize("kind", ["embed", "prompt"])
def test_provider_rebind_rejects_a_live_failed_cleanup_owner(kind):
    from vane.ai.functions import _EmbedTextBatch, _PromptBatch
    from vane.execution._async_runtime import AsyncRuntime

    class Provider:
        async def aclose(self):
            raise OSError(5, "private cleanup path")

    class Descriptor:
        def instantiate(self):
            return Provider()

    old, new = AsyncRuntime(), AsyncRuntime()
    wrapper = (
        _EmbedTextBatch(Descriptor(), "text", "out", 2)
        if kind == "embed"
        else _PromptBatch(Descriptor(), ["text"], "out")
    )
    wrapper.bind_async_runtime(old.run)
    provider = wrapper._ensure_embedder() if kind == "embed" else wrapper._ensure_prompter()
    try:
        with pytest.raises(RuntimeError):
            wrapper.close()
        with pytest.raises(RuntimeError, match="cannot rebind"):
            wrapper.bind_async_runtime(new.run)
        assert wrapper._run_async == old.run
        assert (wrapper._embedder if kind == "embed" else wrapper._prompter) is provider
    finally:
        old.close()
        new.close()


def test_process_group_fork_child_cannot_signal_parent_workers(monkeypatch):
    from vane.execution.udf_process_group import WorkerProcessGroup

    group = WorkerProcessGroup(os.getpid())
    monkeypatch.setattr(group, "owner_pid", os.getpid() + 1)
    monkeypatch.setattr(psutil, "process_iter", lambda: pytest.fail("inherited process group must not be inspected"))
    group.capture()
    group.close(timeout=0.01)


def test_runtime_cleanup_diagnostic_survives_native_string_conversion():
    from vane._native import InvalidInputException
    from vane.execution._udf_runtime import MAX_CLOSE_ERROR_BYTES, UDFExecutor

    class FailingActor:
        def __call__(self, table):
            return table

        def _vane_close(self):
            raise OSError(5, "private cleanup path")

    executor = UDFExecutor(
        {
            "function_pickle": vane_pickle.dumps(FailingActor),
            "execution_backend": "subprocess_actor",
            "call_mode": "map_batches",
        }
    )
    with pytest.raises(RuntimeError) as caught:
        executor.close()
    for error in (caught.value, pickle.loads(pickle.dumps(caught.value))):
        native = InvalidInputException("Invalid Input Error: streaming UDF async error: DiagnosticError: " + str(error))
        assert "OSError (errno=5)" in summarize_error(native)
        assert "private" not in summarize_error(native)
        assert len(str(error).encode("utf-8")) <= MAX_CLOSE_ERROR_BYTES
        assert error.__cause__ is None and error.__context__ is None
    executor._map_fn._vane_close = lambda: None
    executor.close()
