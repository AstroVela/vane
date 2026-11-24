# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Native SDK contracts without model downloads or a serving endpoint."""

from __future__ import annotations

import asyncio
import pickle
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from vane.ai._media import PromptMedia
from vane.ai._video_embedding import VideoClip
from vane.ai.functions import _prepare_embed_call, _prepare_prompt_call
from vane.ai.protocols import NativeInferencePlan, PrompterDescriptor
from vane.ai.provider import load_provider


def install_sdk(setitem, *, mock_transformers=True):
    """Doubles for the documented Python APIs, installed inside each worker."""
    state = SimpleNamespace(
        loads=[], engines=[], calls=[], processing=[], finish="stop", fail=False, requests_closed=0, on_embedding=None
    )
    state.config = SimpleNamespace()

    class Processor:
        model_input_names = ["input_ids", "attention_mask"]

        def __init__(self):
            self.tokenizer = self

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            state.loads.append((model, kwargs))
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            state.messages = messages
            state.template = kwargs
            return "rendered:" + "\n".join(part.get("text", "") for item in messages for part in item["content"])

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return [10, 20, 30]

        def __call__(self, **kwargs):
            state.processing.append(kwargs)
            result = {"input_ids": np.array([[10, 20, 30]]), "attention_mask": np.array([[1, 1, 1]])}
            if "videos" in kwargs:
                result.update(pixel_values_videos=np.array([[1.0]]), video_grid_thw=np.array([[1, 2, 2]]))
            if "images" in kwargs:
                result.update(pixel_values=np.array([[1.0]]), image_grid_thw=np.array([[1, 2, 2]]))
            return result

    class EmbeddingRequest(SimpleNamespace):
        def __init__(self, *, input_ids, dimensions=None, image_data=None, video_data=None, audio_data=None):
            super().__init__(
                input_ids=input_ids,
                dimensions=dimensions,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
            )

    class Engine:
        def __init__(self, **kwargs):
            self.args = kwargs
            self.loop = asyncio.get_running_loop()
            self.closed = 0
            self.tokenizer_manager = self
            state.engines.append(self)

        @classmethod
        def from_engine_args(cls, args):
            return cls(**vars(args))

        async def generate(self, prompt, sampling_params, request_id, **kwargs):
            assert self.loop is asyncio.get_running_loop()
            state.calls.append(("generate", prompt, vars(sampling_params), kwargs))
            if state.fail:
                raise ValueError("engine failure")
            yield SimpleNamespace(
                finished=True, outputs=[SimpleNamespace(text='{"answer":"ok"}', finish_reason=state.finish)]
            )

        async def encode(self, prompt, pooling_params, request_id, **kwargs):
            assert self.loop is asyncio.get_running_loop()
            state.calls.append(("encode", prompt, vars(pooling_params), kwargs))
            try:
                if state.fail:
                    raise ValueError("engine failure")
                vector = [3.0, 4.0] if state.on_embedding is None else await state.on_embedding(prompt)
                yield SimpleNamespace(finished=True, outputs=SimpleNamespace(data=np.array(vector)))
            finally:
                state.requests_closed += 1

        async def async_generate(self, **kwargs):
            assert self.loop is asyncio.get_running_loop()
            state.calls.append(("generate", kwargs))
            if not kwargs.get("image_data") and not kwargs.get("video_data"):
                assert "input_ids" in kwargs and "prompt" not in kwargs
            if state.fail:
                raise ValueError("engine failure")
            return {"text": '{"answer":"ok"}', "meta_info": {"finish_reason": {"type": state.finish}}}

        async def generate_request(self, request, http_request):
            assert self.loop is asyncio.get_running_loop() and http_request is None
            assert isinstance(request, EmbeddingRequest)
            state.calls.append(("encode", vars(request)))
            try:
                if state.fail:
                    raise ValueError("engine failure")
                vector = [3.0, 4.0] if state.on_embedding is None else await state.on_embedding(request)
                yield {"embedding": vector}
            finally:
                state.requests_closed += 1

        def shutdown(self):
            assert self.loop is asyncio.get_running_loop()
            self.closed += 1

    class VideoIO:
        def __init__(self, image_io, num_frames):
            self.num_frames = num_frames

        def load_bytes(self, data):
            state.video_bytes = data
            return SimpleNamespace(media=(np.zeros((2, 4, 4, 3), dtype=np.uint8), {"fps": 2, "frames_indices": [0, 1]}))

    for name, entries in {
        "transformers": {
            "AutoProcessor": Processor,
            "AutoConfig": SimpleNamespace(from_pretrained=lambda *args, **kwargs: state.config),
            "PreTrainedTokenizerBase": type("TokenizerBase", (), {}),
        },
        "transformers.video_utils": {"VideoMetadata": SimpleNamespace},
        "vllm": {
            "AsyncEngineArgs": SimpleNamespace,
            "AsyncLLMEngine": Engine,
            "PoolingParams": SimpleNamespace,
            "SamplingParams": SimpleNamespace,
        },
        "vllm.sampling_params": {"StructuredOutputsParams": SimpleNamespace},
        "vllm.multimodal.media": {"ImageMediaIO": SimpleNamespace, "VideoMediaIO": VideoIO},
        "sglang": {"Engine": Engine},
        "sglang.srt.managers.io_struct": {"EmbeddingReqInput": EmbeddingRequest},
    }.items():
        if not mock_transformers and name.startswith("transformers"):
            continue
        module = ModuleType(name)
        module.__dict__.update(entries)
        setitem(sys.modules, name, module)
    return state


@pytest.fixture
def sdk(monkeypatch):
    return install_sdk(monkeypatch.setitem)


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("failure", ["sync", "async", "cancelled"])
def test_native_shutdown_failure_retains_engine_until_retry(family, failure, sdk):
    import tempfile
    from pathlib import Path

    async def run():
        runtime = load_provider(family).get_text_embedder("configured-encoder", 2).instantiate()
        engine = runtime._ensure_engine()
        processor = runtime.processor
        original_shutdown = engine.shutdown
        attempts = 0
        error_type = asyncio.CancelledError if failure == "cancelled" else TimeoutError
        directory = tempfile.TemporaryDirectory()
        runtime._model_directory = directory

        def shutdown():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise error_type("engine is still running")
            original_shutdown()

        async def async_shutdown():
            await asyncio.sleep(0)
            shutdown()

        engine.shutdown = shutdown if failure == "sync" else async_shutdown
        try:
            with pytest.raises(error_type, match="engine is still running"):
                await runtime.aclose()
            assert not runtime.closed and engine.closed == 0
            assert runtime.engine is engine and runtime.processor is processor
            assert runtime._model_directory is directory and Path(directory.name).is_dir()
            with pytest.raises(RuntimeError, match="closing or closed"):
                runtime._ensure_engine()
            await runtime.aclose()
            assert runtime.closed and engine.closed == 1 and attempts == 2
            assert runtime.engine is None and runtime.processor is None
            assert runtime._model_directory is None and not Path(directory.name).exists()
            await runtime.aclose()
            assert engine.closed == 1 and attempts == 2
        finally:
            engine.shutdown = original_shutdown
            await runtime.aclose()
            directory.cleanup()

    asyncio.run(run())


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("started", [False, True])
def test_native_model_directory_cleanup_retries_without_reclosing_engine(family, started, sdk):
    import tempfile
    from pathlib import Path

    async def run():
        runtime = load_provider(family).get_text_embedder("configured-encoder", 2).instantiate()
        engine = runtime._ensure_engine() if started else None
        directory = tempfile.TemporaryDirectory()
        runtime._model_directory = directory
        cleanup = directory.cleanup
        attempts = 0

        def cleanup_once_ready():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("model files are still busy")
            cleanup()

        directory.cleanup = cleanup_once_ready
        try:
            with pytest.raises(OSError, match="model files are still busy"):
                await runtime.aclose()
            assert not runtime.closed and runtime.engine is None
            assert runtime._model_directory is directory and Path(directory.name).is_dir()
            with pytest.raises(RuntimeError, match="closing or closed"):
                runtime._ensure_engine()
            await runtime.aclose()
            assert runtime.closed and runtime._model_directory is None
            assert attempts == 2 and not Path(directory.name).exists()
            assert engine is None or engine.closed == 1
            await runtime.aclose()
            assert attempts == 2
        finally:
            directory.cleanup = cleanup
            await runtime.aclose()
            cleanup()

    asyncio.run(run())


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("kind", ["embed", "prompt"])
@pytest.mark.parametrize("backend", ["subprocess_actor", "subprocess_task"])
@pytest.mark.parametrize("failures", [1, 2])
def test_worker_native_cleanup_acknowledges_only_a_stopped_engine(family, kind, backend, failures, sdk):
    import pyarrow as pa

    from vane.ai.functions import _adapt_batch_wrapper_for_backend, _EmbedTextBatch, _PromptBatch
    from vane.execution._udf_runtime import UDFExecutor
    from vane.execution.udf_subprocess_worker import _close_executor_with_retry
    from vane.pickle import dumps

    provider = load_provider(family)
    options = {"gpus_per_actor": 0}
    if kind == "embed":
        wrapper = _EmbedTextBatch(
            provider.get_text_embedder("configured-encoder", 2, options=options), "text", "out", 2
        )
    else:
        descriptor = provider.get_prompter(
            "configured-generator", options={**options, "media_mime_types": ["image/png"]}
        )
        wrapper = _PromptBatch(descriptor, ["text"], "out")
    executor = UDFExecutor(
        {
            "function_pickle": dumps(_adapt_batch_wrapper_for_backend(wrapper, backend)),
            "udf_name": "native_cleanup_retry",
            "call_mode": "map_batches",
            "execution_backend": backend,
            "batch_size": 1,
            "output_schema": [{"name": "out", "type": "FLOAT[2]" if kind == "embed" else "VARCHAR"}],
        }
    )
    engine = None
    try:
        assert list(executor.iter_submit(pa.table({"text": ["hello"]})))
        engine = sdk.engines[0]
        original_shutdown = engine.shutdown
        attempts = 0

        def shutdown_once_ready():
            nonlocal attempts
            attempts += 1
            if attempts <= failures:
                raise TimeoutError("native child processes are still running")
            original_shutdown()

        engine.shutdown = shutdown_once_ready
        error = _close_executor_with_retry(executor)
        if failures == 1:
            assert error is None and executor._closed
            assert engine.closed == 1 and attempts == 2
        else:
            assert error is not None and "TimeoutError" in str(error)
            assert not executor._closed and engine.closed == 0 and attempts == 2
            assert executor._async_runtime is not None and not engine.loop.is_closed()
            with pytest.raises(RuntimeError, match="closing or closed"):
                list(executor.iter_submit(pa.table({"text": ["new request"]})))
            assert _close_executor_with_retry(executor) is None
            assert executor._closed and engine.closed == 1 and attempts == 3
        assert engine.loop.is_closed()
        assert _close_executor_with_retry(executor) is None
        assert engine.closed == 1 and attempts == failures + 1
    finally:
        if engine is not None:
            engine.shutdown = original_shutdown
        executor.close()


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("backend", ["subprocess_actor", "subprocess_task"])
@pytest.mark.parametrize("entry,failures", [("worker", 1), ("worker", 3), ("worker", 5), ("public", 1)])
def test_real_native_worker_retains_failed_task_cleanup(tmp_path, monkeypatch, family, backend, entry, failures):
    import json
    from pathlib import Path

    import cloudpickle
    import psutil
    import pyarrow as pa

    import vane
    import vane.execution.udf_subprocess as subprocess_exec
    from vane.ai import embed
    from vane.ai.functions import _adapt_batch_wrapper_for_backend, _EmbedTextBatch
    from vane.ai.providers._native_inference import NativeTextEmbedderDescriptor
    from vane.pickle import dumps

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    marker = str(tmp_path / "engine-events.jsonl")

    class WorkerEmbedding(NativeTextEmbedderDescriptor):
        def instantiate(self):
            import subprocess

            install_sdk(lambda mapping, key, value: mapping.__setitem__(key, value))
            runtime = super().instantiate()
            ensure = runtime._ensure_engine

            def ensure_engine():
                engine = ensure()
                if hasattr(engine, "shutdown_attempts"):
                    return engine
                child = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(120)", marker],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                )
                engine.shutdown_attempts = 0
                shutdown = engine.shutdown

                def record(event):
                    with open(marker, "a") as output:
                        output.write(json.dumps({"event": event, "pid": child.pid}) + "\n")

                record("started")

                def shutdown_once_ready():
                    engine.shutdown_attempts += 1
                    record("shutdown_attempt")
                    if engine.shutdown_attempts <= failures:
                        raise TimeoutError("native engine child is still running")
                    shutdown()
                    child.terminate()
                    child.wait(timeout=5)
                    record("stopped")

                engine.shutdown = shutdown_once_ready
                return engine

            runtime._ensure_engine = ensure_engine
            return runtime

    descriptor = load_provider(family).get_text_embedder("configured-encoder", 2, options={"gpus_per_actor": 0})
    descriptor.__class__ = WorkerEmbedding
    worker = None
    try:
        if entry == "worker":
            replies = []
            recv = subprocess_exec._recv_message
            checked_pending_submit = False

            def observe_reply(sock):
                nonlocal checked_pending_submit
                msg_type, payload = recv(sock)
                replies.append(msg_type)
                if (
                    msg_type == subprocess_exec._MSG_ERROR
                    and backend == "subprocess_task"
                    and failures == 3
                    and not checked_pending_submit
                ):
                    import struct

                    checked_pending_submit = True
                    # Even an empty new request must not replace a task whose
                    # executor still owns failed cleanup resources.
                    empty = pa.table({"text": pa.array([], type=pa.string())})
                    sink = pa.BufferOutputStream()
                    with pa.ipc.new_stream(sink, empty.schema) as writer:
                        writer.write_table(empty)
                    assert worker is not None and worker._data_shm is not None
                    size = subprocess_exec._write_ipc_to_shm(worker._data_shm, sink.getvalue().to_pybytes())
                    subprocess_exec._send_message(sock, subprocess_exec._MSG_SUBMIT, struct.pack("<Q", size))
                    retry_type, _ = recv(sock)
                    assert retry_type == subprocess_exec._MSG_ERROR
                if msg_type == subprocess_exec._MSG_ACK:
                    assert '"event": "stopped"' in Path(marker).read_text()
                return msg_type, payload

            monkeypatch.setattr(subprocess_exec, "_recv_message", observe_reply)
            wrapper = _EmbedTextBatch(descriptor, "text", "out", 2, max_retries=0)
            worker = subprocess_exec._SingleSubprocessExecutor(
                {
                    "function_pickle": dumps(_adapt_batch_wrapper_for_backend(wrapper, backend)),
                    "udf_name": "native_worker_cleanup_retry",
                    "call_mode": "map_batches",
                    "execution_backend": backend,
                    "batch_size": 1,
                    "output_schema": [{"name": "out", "type": "FLOAT[2]"}],
                }
            )
            if backend == "subprocess_task":
                with pytest.raises(Exception, match="TimeoutError"):
                    worker.submit(pa.table({"text": ["hello"]}))
                worker.close(kill=False)
            else:
                worker.submit(pa.table({"text": ["hello"]}))
                if failures >= 3:
                    with pytest.raises(Exception, match="TimeoutError"):
                        worker.close(kill=False)
                else:
                    worker.close(kill=False)
            assert (subprocess_exec._MSG_ACK in replies) == (failures == 1)
            if failures >= 3:
                assert replies.count(subprocess_exec._MSG_ERROR) == (2 if backend == "subprocess_task" else 1)
        else:
            base = type(load_provider(family))

            class WorkerProvider(base):
                def get_text_embedder(self, *args, **kwargs):
                    return descriptor

            with vane.connect() as connection:
                relation = embed(
                    connection.sql("select 'hello' as text"),
                    vane.col("text"),
                    provider=WorkerProvider(),
                    model="configured-encoder",
                    dimensions=2,
                    gpus_per_actor=0,
                    execution_backend=backend,
                    max_retries=0,
                )
                if backend == "subprocess_task":
                    with pytest.raises(Exception, match="TimeoutError"):
                        relation.fetchall()
                else:
                    assert len(relation.fetchall()) == 1
        events = [json.loads(line) for line in Path(marker).read_text().splitlines()]
        exhausted = failures >= (5 if backend == "subprocess_task" else 4)
        attempts = (5 if backend == "subprocess_task" else 4) if exhausted else failures + 1
        assert [event["event"] for event in events] == ["started"] + ["shutdown_attempt"] * attempts + (
            [] if exhausted else ["stopped"]
        )
        if exhausted and psutil.pid_exists(events[0]["pid"]):
            assert psutil.Process(events[0]["pid"]).status() == psutil.STATUS_ZOMBIE
        else:
            assert not psutil.pid_exists(events[0]["pid"])
    finally:
        try:
            if worker is not None:
                worker.close(kill=True)
        finally:
            if Path(marker).exists():
                events = [json.loads(line) for line in Path(marker).read_text().splitlines()]
                for pid in {event["pid"] for event in events}:
                    try:
                        child = psutil.Process(pid)
                        if child.cmdline()[-1:] == [marker]:
                            child.kill()
                            psutil.wait_procs([child], timeout=5)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        pass


def clip():
    return VideoClip((np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8, 3), dtype=np.uint8)), (0.125, 1.375), (4, 18))


@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("on_error", ["raise", "ignore"])
def test_native_single_text_preserves_blanks_and_nulls(family, on_error, sdk):
    import pyarrow as pa

    from vane.ai._embedding_inputs import EmbeddingConfigurationError
    from vane.ai.functions import _EmbedTextBatch
    from vane.execution._async_runtime import AsyncRuntime

    descriptor = load_provider(family).get_text_embedder(
        "fixture", 2, options={"gpus_per_actor": 0, "max_input_bytes": 8}
    )
    wrapper = _EmbedTextBatch(descriptor, "text", "vector", 2, on_error=on_error, normalize=False)
    loop = AsyncRuntime()
    wrapper.bind_async_runtime(loop.run)
    values = ["healthy", "", " \t\n", "\u3000", None]
    try:
        result = wrapper(pa.table({"text": values}))
        assert result["vector"].to_pylist() == [[3.0, 4.0]] * 4 + [None]
        prompts = (
            [call["text"][0] for call in sdk.processing]
            if family == "sglang"
            else [call[1]["prompt"] for call in sdk.calls]
        )
        assert prompts == [f"rendered:{descriptor.options['instruction']}\n{value}" for value in values[:-1]]
        with pytest.raises(EmbeddingConfigurationError, match="max_input_bytes"):
            wrapper(pa.table({"text": ["\u3000" * 3]}))
        assert len(sdk.calls) == 4
    finally:
        wrapper.close()
        loop.close()
    assert len(sdk.engines) == 1 and sdk.engines[0].closed == 1


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("kind", ["text", "image", "video"])
def test_native_embedding_overlaps_requests_and_restores_order(family, kind, sdk):
    async def run():
        descriptor = getattr(load_provider(family), f"get_{kind}_embedder")(
            model="configured-encoder", dimensions=2, options={"max_concurrency_per_actor": 2}
        )
        runtime = descriptor.instantiate()
        active, peak, count = 0, 0, 0
        gates = [asyncio.Event(), asyncio.Event()]
        finished = []

        async def request(value):
            nonlocal active, peak, count
            index, count = count, count + 1
            active += 1
            peak = max(peak, active)
            try:
                if index % 2 == 0:
                    await gates[index // 2].wait()
                else:
                    gates[index // 2].set()
                finished.append(index)
                return [index, 1]
            finally:
                active -= 1

        sdk.on_embedding = request
        value = {"text": "query", "image": np.zeros((8, 8, 3), dtype=np.uint8), "video": clip()}[kind]
        try:
            result = await getattr(runtime, f"embed_{kind}")([value] * 4)
            assert result == [[0, 1], [1, 1], [2, 1], [3, 1]]
            assert peak == 2 and active == 0
            assert finished[0] == 1
            assert len(sdk.engines) == 1 and sdk.requests_closed == 4
        finally:
            await runtime.aclose()

    asyncio.run(asyncio.wait_for(run(), 5))


@pytest.mark.parametrize("family", ["vllm", "sglang"])
def test_native_embedding_retries_only_the_failed_request(family, sdk):
    import pyarrow as pa

    from vane.ai.functions import RetryAfterError, _EmbedTextBatch
    from vane.execution._async_runtime import AsyncRuntime

    calls = 0

    async def request(value):
        nonlocal calls
        index, calls = calls, calls + 1
        await asyncio.sleep(0)
        if index == 2:
            raise RetryAfterError(0, status=503)
        return [min(index, 2), 1]

    sdk.on_embedding = request
    descriptor = load_provider(family).get_text_embedder(
        model="configured-encoder", dimensions=2, options={"max_concurrency_per_actor": 2}
    )
    wrapper = _EmbedTextBatch(descriptor, "text", "embedding", 2, max_retries=1)
    runtime = AsyncRuntime()
    wrapper.bind_async_runtime(runtime.run)
    try:
        result = wrapper(pa.table({"text": ["first", None, "second", "retry"]}))
        assert result["embedding"].to_pylist() == [[0, 1], None, [1, 1], [2, 1]]
        assert calls == sdk.requests_closed == 4
        assert len(sdk.engines) == 1
    finally:
        wrapper.close()
        runtime.close()


@pytest.mark.parametrize("family", ["vllm", "sglang"])
def test_native_embedding_terminal_failure_stops_sibling_retries(family, sdk):
    import pyarrow as pa

    from vane.ai.functions import RetryAfterError, _EmbedTextBatch
    from vane.execution._async_runtime import AsyncRuntime

    calls = []
    ready = asyncio.Event()

    async def request(value):
        index = len(calls)
        calls.append(value)
        if index == 0:
            await ready.wait()
            raise ValueError("terminal input failure")
        ready.set()
        raise RetryAfterError(0, status=503)

    sdk.on_embedding = request
    descriptor = load_provider(family).get_text_embedder(
        model="configured-encoder", dimensions=2, options={"max_concurrency_per_actor": 2}
    )
    wrapper = _EmbedTextBatch(descriptor, "text", "embedding", 2, max_retries=3)
    runtime = AsyncRuntime()
    wrapper.bind_async_runtime(runtime.run)
    try:
        with pytest.raises(RuntimeError, match="upstream error: ValueError"):
            wrapper(pa.table({"text": ["fatal", "retry", "queued"]}))
        assert len(calls) == sdk.requests_closed == 2
    finally:
        wrapper.close()
        runtime.close()


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("failure", ["request", "vector"])
def test_native_embedding_ignore_preserves_successful_rows(family, failure, sdk):
    import pyarrow as pa

    from vane.ai.functions import _EmbedTextBatch
    from vane.execution._async_runtime import AsyncRuntime

    calls = 0

    async def request(value):
        nonlocal calls
        index, calls = calls, calls + 1
        await asyncio.sleep(0)
        if index == 1:
            if failure == "request":
                raise ValueError("invalid input")
            return [float("nan"), 1]
        return [index, 1]

    sdk.on_embedding = request
    descriptor = load_provider(family).get_text_embedder(
        model="configured-encoder", dimensions=2, options={"max_concurrency_per_actor": 2}
    )
    wrapper = _EmbedTextBatch(descriptor, "text", "embedding", 2, max_retries=2, on_error="ignore")
    runtime = AsyncRuntime()
    wrapper.bind_async_runtime(runtime.run)
    try:
        result = wrapper(pa.table({"text": ["first", "bad", "last"]}))
        assert result["embedding"].to_pylist() == [[0, 1], None, [2, 1]]
        assert calls == sdk.requests_closed == 3
    finally:
        wrapper.close()
        runtime.close()


@pytest.mark.parametrize("family", ["vllm", "sglang"])
def test_native_embedding_cancellation_drains_sdk_streams(family, sdk):
    async def run():
        descriptor = load_provider(family).get_video_embedder(
            model="configured-encoder", dimensions=2, options={"max_concurrency_per_actor": 2}
        )
        runtime = descriptor.instantiate()
        started, cleaning = [], []
        release = asyncio.Event()

        async def request(value):
            started.append(value)
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.append(value)
                await release.wait()

        sdk.on_embedding = request
        task = asyncio.create_task(runtime.embed_video([clip()] * 6))
        try:
            while len(started) < 2:
                await asyncio.sleep(0)
            task.cancel()
            while len(cleaning) < 2:
                await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and sdk.requests_closed == 0
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(started) == sdk.requests_closed == 2
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await runtime.aclose()

    asyncio.run(asyncio.wait_for(run(), 5))


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("kind", ["text", "image", "video"])
@pytest.mark.parametrize("limit", [0, True, 1.5, 1025])
def test_native_embedding_rejects_invalid_concurrency_before_loading(family, kind, limit, sdk):
    with pytest.raises(ValueError):
        getattr(load_provider(family), f"get_{kind}_embedder")(
            model="configured-encoder", dimensions=2, options={"max_concurrency_per_actor": limit}
        )
    assert sdk.loads == sdk.engines == []


@pytest.mark.parametrize("family", ["vllm", "sglang"])
def test_models_and_dimensions_are_explicit_and_planning_is_lazy(family, sdk):
    provider = load_provider(family)
    with pytest.raises(ValueError, match="model"):
        provider.get_prompter()
    for model in (None, "", " "):
        with pytest.raises(ValueError, match="model"):
            provider.get_prompter(model=model)
        with pytest.raises(ValueError, match="model"):
            provider.get_prompter(model=model, options={"media_mime_types": ["video/mp4"]})
    assert isinstance(provider.get_prompter(model="configured-text-model"), NativeInferencePlan)
    for kind in ("text", "image", "video"):
        for model in (None, "", " "):
            with pytest.raises(ValueError, match="model"):
                getattr(provider, f"get_{kind}_embedder")(model, dimensions=17)
        with pytest.raises(ValueError, match="dimensions"):
            getattr(provider, f"get_{kind}_embedder")("configured-encoder")
        desc, dims, udf, *_ = _prepare_embed_call(
            family, "/models/custom-encoder", 17, "raise", {"gpus_per_actor": 2}, relation=True, input_kind=kind
        )
        assert pickle.loads(pickle.dumps(desc)).get_model() == "/models/custom-encoder"
        assert dims == 17 and udf.num_gpus == 2
    assert sdk.loads == [] and sdk.engines == []


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("options", [{"transport": "http"}, {"base_url": "http://localhost/v1"}, {"timeout": 12}])
def test_http_options_are_rejected(family, options):
    provider = load_provider(family)
    with pytest.raises((TypeError, ValueError)):
        provider.get_text_embedder("configured", 2, options=options)
    with pytest.raises((TypeError, ValueError)):
        provider.get_prompter("configured", options={"media_mime_types": ["video/mp4"], **options})


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("kind", ["text", "image", "video", "prompt"])
def test_explicit_processing_source_is_used_on_the_first_load(family, kind, sdk):
    key = "tokenizer" if family == "vllm" else "tokenizer_path"
    args = {key: "separate-processing-source", "revision": "model-revision"}
    if family == "vllm":
        args["tokenizer_revision"] = "processing-revision"
    options = {"engine_args": args, "gpus_per_actor": 0}
    provider = load_provider(family)
    if kind == "prompt":
        descriptor = provider.get_prompter(
            "weights-only-checkpoint", options={**options, "media_mime_types": ["image/png"]}
        )
    else:
        descriptor = getattr(provider, f"get_{kind}_embedder")("weights-only-checkpoint", 2, options=options)
    runtime = descriptor.instantiate()
    try:
        assert sdk.loads == [
            (
                "separate-processing-source",
                {
                    "revision": "processing-revision" if family == "vllm" else "model-revision",
                    "trust_remote_code": False,
                    "config": sdk.config,
                },
            )
        ]
        assert descriptor.get_model() == "weights-only-checkpoint" and sdk.engines == []
    finally:
        asyncio.run(runtime.aclose())


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("processing", [False, True])
def test_native_text_embedding_formats_tokenizer_messages(family, processing, sdk, monkeypatch):
    transformers = sys.modules["transformers"]

    class Tokenizer(transformers.AutoProcessor, transformers.PreTrainedTokenizerBase):
        def __init__(self):
            # A text tokenizer does not have a nested .tokenizer attribute.
            pass

        def apply_chat_template(self, messages, **kwargs):
            sdk.messages = messages
            return "\n".join(message["role"] + ": " + message["content"] for message in messages)

    monkeypatch.setattr(transformers, "AutoProcessor", Tokenizer)
    options = {"gpus_per_actor": 0, "instruction": "Find related items"}
    if processing:
        options["processor_kwargs"] = {"add_special_tokens": False}
    descriptor = load_provider(family).get_text_embedder("configured-text-encoder", 2, options=options)

    async def run():
        runtime = descriptor.instantiate()
        try:
            assert await runtime.embed_text(["query"]) == [[3, 4]]
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert sdk.messages == [
        {"role": "system", "content": "Find related items"},
        {"role": "user", "content": "query"},
    ]
    if family == "sglang":
        assert sdk.calls[0][1]["input_ids"] == [10, 20, 30]
        assert sdk.processing[0]["text"] == ["system: Find related items\nuser: query"]
        assert sdk.processing[0]["add_special_tokens"] is False
    else:
        assert sdk.calls[0][1]["prompt"] == "system: Find related items\nuser: query"


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("kind", ["image", "video"])
def test_tokenizer_rejects_media_before_engine_initialization(family, kind, sdk, monkeypatch):
    transformers = sys.modules["transformers"]
    monkeypatch.setattr(
        transformers,
        "AutoProcessor",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: transformers.PreTrainedTokenizerBase()),
    )
    descriptor = getattr(load_provider(family), f"get_{kind}_embedder")(
        "configured-text-encoder", 2, options={"gpus_per_actor": 0}
    )

    async def run():
        runtime = descriptor.instantiate()
        try:
            with pytest.raises(ValueError, match="tokenizer cannot format"):
                await getattr(runtime, f"embed_{kind}")([clip().frames[0] if kind == "image" else clip()])
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert sdk.engines == []


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("kind", ["text", "image", "video"])
def test_native_embedding_enforces_input_bytes_before_preprocessing(family, kind, sdk):
    value = "中文" if kind == "text" else clip().frames[0] if kind == "image" else clip()
    size = 6 if kind == "text" else 192 if kind == "image" else 384

    async def run(limit):
        descriptor = getattr(load_provider(family), f"get_{kind}_embedder")(
            "configured-encoder", 2, options={"gpus_per_actor": 0, "max_input_bytes": limit}
        )
        runtime = descriptor.instantiate()
        try:
            return await getattr(runtime, f"embed_{kind}")([value])
        finally:
            await runtime.aclose()

    with pytest.raises(ValueError, match="max_input_bytes"):
        asyncio.run(run(1))
    with pytest.raises(ValueError, match="max_input_bytes"):
        asyncio.run(run(size - 1))
    assert sdk.engines == [] and sdk.processing == [] and not hasattr(sdk, "messages")
    assert asyncio.run(run(size)) == [[3, 4]]


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("kind", ["text", "image", "video"])
def test_native_embedding_preserves_inputs_and_reuses_engine(family, kind, sdk):
    options = {"gpus_per_actor": 0, "instruction": "Find related items", "paired_image_queries": True}
    desc = getattr(load_provider(family), f"get_{kind}_embedder")("configured-encoder", 2, options=options)
    value = "query" if kind == "text" else clip().frames[0] if kind == "image" else clip()

    async def run():
        runtime = desc.instantiate()
        try:
            assert await getattr(runtime, f"embed_{kind}")([value, value]) == [[3, 4], [3, 4]]
        finally:
            await runtime.aclose()
            await runtime.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            runtime._ensure_engine()

    asyncio.run(run())
    assert len(sdk.engines) == 1 and sdk.engines[0].closed == 1
    assert len(sdk.calls) == 2 and sdk.loads[0][0] == "configured-encoder"
    # Fixed-width models reject a dimensions override even at their natural width.
    assert (sdk.calls[0][2] if family == "vllm" else sdk.calls[0][1])["dimensions"] is None
    if kind == "video":
        if family == "vllm":
            frames, metadata = sdk.calls[0][1]["multi_modal_data"]["video"][0]
            assert frames.shape == (2, 8, 8, 3)
        else:
            metadata = vars(sdk.processing[0]["videos_kwargs"]["video_metadata"][0])
            assert sdk.processing[0]["do_sample_frames"] is False
            assert sdk.calls[0][1]["video_data"][0]["format"] == "processor_output"
            assert sdk.calls[0][1]["input_ids"] == sdk.calls[0][1]["video_data"][0]["input_ids"][0].tolist()
        assert metadata["frames_indices"] == [125000, 1375000] and metadata["fps"] == 1000000
    if family == "sglang":
        assert sdk.requests_closed == 2


@pytest.mark.parametrize("family", ["vllm", "sglang"])
def test_embedding_dimension_override_is_explicit(family, sdk):
    descriptor = load_provider(family).get_text_embedder(
        "matryoshka-model", 2, options={"gpus_per_actor": 0, "supports_overriding_dimensions": True}
    )

    async def run():
        runtime = descriptor.instantiate()
        try:
            assert await runtime.embed_text(["query"]) == [[3, 4]]
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert (sdk.calls[0][2] if family == "vllm" else sdk.calls[0][1])["dimensions"] == 2


@pytest.mark.parametrize("family", ["vllm", "sglang"])
def test_native_prompt_forwards_model_settings_and_structured_output(family, sdk):
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    options = {
        "media_mime_types": ["video/mp4"],
        "gpus_per_actor": 0,
        "max_tokens": 128,
        "engine_args": {"dtype": "float16"},
        "generate_args": {"sampling_params": {"top_k": 7}},
        "chat_template": "custom template",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    desc, udf, *_ = _prepare_prompt_call(
        family, "/models/custom-vlm", schema, "system", False, "raise", options, relation=True
    )
    assert isinstance(desc, PrompterDescriptor) and udf.num_gpus == 0

    async def run():
        runtime = desc.instantiate()
        try:
            assert await runtime.prompt(("question", PromptMedia(b"video-bytes", "video/mp4"))) == '{"answer":"ok"}'
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert sdk.engines[0].args["dtype"] == "float16" and sdk.engines[0].closed == 1
    assert sdk.template["chat_template"] == "custom template" and sdk.template["enable_thinking"] is False
    if family == "vllm":
        assert sdk.video_bytes == b"video-bytes"
        assert sdk.calls[0][2]["top_k"] == 7 and sdk.calls[0][2]["max_tokens"] == 128
        assert sdk.calls[0][2]["structured_outputs"].json == schema
        assert sdk.calls[0][3]["tokenization_kwargs"] == {"add_special_tokens": False}
    else:
        assert sdk.calls[0][1]["video_data"] == [b"video-bytes"]
        assert sdk.calls[0][1]["sampling_params"]["top_k"] == 7
        assert sdk.calls[0][1]["sampling_params"]["max_new_tokens"] == 128


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("mode", ["engine_error", "length", "budget", "mime"])
def test_failures_are_explicit_and_engine_is_released(family, mode, sdk):
    desc = load_provider(family).get_prompter(
        "model", options={"media_mime_types": ["video/mp4"], "gpus_per_actor": 0, "max_input_bytes": 32}
    )
    sdk.fail = mode == "engine_error"
    sdk.finish = "length" if mode == "length" else "stop"

    async def run():
        runtime = desc.instantiate()
        try:
            with pytest.raises(ValueError):
                await runtime.prompt(
                    (
                        "q" * (100 if mode == "budget" else 1),
                        PromptMedia(b"video", "image/png" if mode == "mime" else "video/mp4"),
                    )
                )
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert all(engine.closed == 1 for engine in sdk.engines)
    if mode in {"budget", "mime"}:
        assert not sdk.engines


@pytest.mark.parametrize("family", ["vllm", "sglang"])
def test_engine_resource_admission_and_input_ownership(family):
    options = {"engine_args": {"tensor_parallel_size" if family == "vllm" else "tp_size": 2}}
    desc = load_provider(family).get_text_embedder("configured", 2, options=options)
    assert desc.get_udf_options().num_gpus == 2
    with pytest.raises(ValueError, match="parallelism"):
        load_provider(family).get_text_embedder("configured", 2, options={**options, "gpus_per_actor": 1})
    for invalid in (
        {"processor_kwargs": {"videos_kwargs": {"do_sample_frames": True}}},
        {"processor_kwargs": {"text_kwargs": {"padding": True}}},
        {"processor_kwargs": {"text_kwargs": {"truncation": True}}},
        {"processor_kwargs": {"text_kwargs": {"return_tensors": "np"}}},
        {"processor_kwargs": {"images_kwargs": []}},
        {"engine_args": {"model": "other"}},
        {"pooling_args": {"dimensions": 9}},
    ):
        with pytest.raises((TypeError, ValueError)):
            load_provider(family).get_video_embedder("configured", 2, options=invalid)


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("processing", [{"add_special_tokens": True}, {"text_kwargs": {"add_special_tokens": True}}])
def test_native_chat_template_owns_special_tokens(family, processing):
    with pytest.raises(ValueError, match="chat templates own special tokens"):
        load_provider(family).get_text_embedder("configured", 2, options={"processor_kwargs": processing})


@pytest.mark.parametrize("tokenization", [{"add_special_tokens": True}, None])
def test_vllm_prompt_chat_template_owns_special_tokens(tokenization):
    with pytest.raises(ValueError, match="chat templates own special tokens"):
        load_provider("vllm").get_prompter(
            "configured",
            options={"media_mime_types": ["image/png"], "generate_args": {"tokenization_kwargs": tokenization}},
        )


@pytest.mark.parametrize("failure", [False, True])
def test_sglang_embedding_request_is_closed_on_success_or_failure(sdk, failure):
    sdk.fail = failure

    async def run():
        runtime = (
            load_provider("sglang").get_text_embedder("configured", 2, options={"gpus_per_actor": 0}).instantiate()
        )
        try:
            if failure:
                with pytest.raises(ValueError, match="engine failure"):
                    await runtime.embed_text(["query"])
            else:
                assert await runtime.embed_text(["query"]) == [[3, 4]]
            assert sdk.requests_closed == 1
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert sdk.engines[0].closed == 1


def test_sglang_text_token_budget_is_checked_before_engine_startup(sdk):
    async def run():
        runtime = (
            load_provider("sglang")
            .get_text_embedder("configured", 2, options={"gpus_per_actor": 0, "engine_args": {"context_length": 2}})
            .instantiate()
        )
        try:
            with pytest.raises(ValueError, match="context_length"):
                await runtime.embed_text(["query"])
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert sdk.engines == []


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize(
    "times,error",
    [
        ((0.0,), "at least two uniformly spaced video frames"),
        ((0.0, 0.5, 1.5, 2.0), "at least two uniformly spaced video frames"),
        ((0.0, 0.5, 0.5, 1.0), "at least two uniformly spaced video frames"),
        ((0.5, 0.5), "at least two uniformly spaced video frames"),
        ((1.0, 0.5), "timestamps and indices must be ordered and aligned"),
    ],
)
def test_scalar_video_timing_rejects_unrepresentable_clips_before_engine_startup(family, times, error, sdk):
    sys.modules["transformers"].AutoProcessor.model_input_names = ["input_ids", "second_per_grid_ts"]
    frames = tuple(np.zeros((8, 8, 3), dtype=np.uint8) for _ in times)

    async def run():
        runtime = load_provider(family).get_video_embedder("configured", 2, options={"gpus_per_actor": 0}).instantiate()
        runtime.configure_execution(on_error="ignore", max_retries=2, validate=lambda v: v)
        try:
            with pytest.raises(ValueError, match=error):
                await runtime.embed_video([VideoClip(frames, times, tuple(range(len(times))))])
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert sdk.engines == [] and sdk.calls == [] and sdk.processing == []


@pytest.mark.parametrize("kind", ["image", "video"])
def test_vllm_rejects_ambiguous_media_options_before_engine_startup(kind, sdk):
    descriptor = getattr(load_provider("vllm"), f"get_{kind}_embedder")(
        "configured",
        2,
        options={
            "gpus_per_actor": 0,
            "processor_kwargs": {"do_resize": True, f"{kind}s_kwargs": {"do_resize": False}},
        },
    )

    async def run():
        runtime = descriptor.instantiate()
        try:
            value = clip() if kind == "video" else clip().frames[0]
            with pytest.raises(ValueError, match="Duplicate flat and nested embedding media options"):
                await getattr(runtime, f"embed_{kind}")([value])
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert sdk.engines == [] and sdk.calls == []


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("processing", [False, True])
def test_native_embedding_with_real_text_tokenizer(family, processing, monkeypatch):
    transformers = pytest.importorskip("transformers", minversion="4.57.1")
    from tokenizers import Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import BPE

    vocabulary = ["<unk>", "<pad>"] + sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(BPE({word: index for index, word in enumerate(vocabulary)}, [], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        chat_template="{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}",
    )
    state = install_sdk(monkeypatch.setitem, mock_transformers=False)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: tokenizer)
    descriptor = load_provider(family).get_text_embedder(
        "configured-text-encoder",
        2,
        options={
            "gpus_per_actor": 0,
            "instruction": "Find related items",
            "processor_kwargs": {"add_special_tokens": False} if processing else {},
        },
    )

    async def run():
        runtime = descriptor.instantiate()
        try:
            assert await runtime.embed_text(["query"]) == [[3, 4]]
        finally:
            await runtime.aclose()

    asyncio.run(run())
    expected_text = "system: Find related items\nuser: query\n"
    if family == "sglang":
        assert state.calls[0][1]["input_ids"] == tokenizer.encode(expected_text, add_special_tokens=False)
        assert "prompt" not in state.calls[0][1]
    else:
        assert state.calls[0][1]["prompt"] == expected_text
        assert state.calls[0][3]["tokenization_kwargs"] == {"add_special_tokens": False}


@pytest.fixture
def bos_tokenizer():
    transformers = pytest.importorskip("transformers", minversion="4.57.1")
    from tokenizers import Tokenizer, pre_tokenizers, processors
    from tokenizers.models import WordLevel

    backend = Tokenizer(WordLevel({"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)]
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        chat_template="{{ bos_token }}{{ messages[-1]['content'] }}{{ eos_token }}",
    )


@pytest.mark.parametrize("processor_kind", ["tokenizer", "gemma3"])
def test_sglang_text_only_prompt_preserves_real_template_tokens(tmp_path, processor_kind, bos_tokenizer, monkeypatch):
    import transformers

    tokenizer = bos_tokenizer
    if processor_kind == "gemma3":
        pytest.importorskip("torchvision")
        tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_object=bos_tokenizer.backend_tokenizer,
            unk_token=bos_tokenizer.unk_token,
            bos_token=bos_tokenizer.bos_token,
            eos_token=bos_tokenizer.eos_token,
            chat_template=bos_tokenizer.chat_template,
            extra_special_tokens={
                "image_token": "<image_soft_token>",
                "boi_token": "<start_of_image>",
                "eoi_token": "<end_of_image>",
            },
        )
        processor = transformers.Gemma3Processor(
            image_processor=transformers.Gemma3ImageProcessor(),
            tokenizer=tokenizer,
            chat_template="{{ bos_token }}{{ messages[-1]['content'][0]['text'] }}{{ eos_token }}",
        )
        transformers.Gemma3Config().save_pretrained(tmp_path)
        processor.save_pretrained(tmp_path)
    else:
        transformers.LlamaConfig().save_pretrained(tmp_path)
        tokenizer.save_pretrained(tmp_path)
    state = install_sdk(monkeypatch.setitem, mock_transformers=False)
    descriptor = load_provider("sglang").get_prompter(
        str(tmp_path), options={"media_mime_types": ["image/png"], "gpus_per_actor": 0, "max_tokens": 64}
    )

    async def run():
        runtime = descriptor.instantiate()
        try:
            assert await runtime.prompt(("hello",)) == '{"answer":"ok"}'
        finally:
            await runtime.aclose()

    asyncio.run(run())
    expected = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}], tokenize=True, return_dict=False, add_generation_prompt=True
    )
    assert expected == [1, 3, 2]
    assert tokenizer.encode("<s>hello</s>") == [1, 1, 3, 2, 2]
    request = state.calls[0][1]
    assert request["input_ids"] == expected and "prompt" not in request
    assert request["image_data"] is None and request["video_data"] is None
    assert request["sampling_params"]["max_new_tokens"] == 64
    assert state.engines[0].closed == 1


@pytest.mark.parametrize("family", ["vllm", "sglang"])
def test_real_standalone_tokenizer_loads_with_weights_only_checkpoint(tmp_path, family, bos_tokenizer, monkeypatch):
    import transformers

    model, processing = tmp_path / "checkpoint", tmp_path / "processing"
    transformers.LlamaConfig().save_pretrained(model)
    bos_tokenizer.save_pretrained(processing)
    state = install_sdk(monkeypatch.setitem, mock_transformers=False)
    key = "tokenizer" if family == "vllm" else "tokenizer_path"

    async def run():
        runtime = (
            load_provider(family)
            .get_text_embedder(str(model), 2, options={"gpus_per_actor": 0, "engine_args": {key: str(processing)}})
            .instantiate()
        )
        try:
            assert runtime._render([{"type": "text", "text": "hello"}], None) == "<s>hello</s>"
            assert await runtime.embed_text(["hello"]) == [[3, 4]]
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert state.engines[0].args["model" if family == "vllm" else "model_path"] == str(model)
    assert state.engines[0].args[key] == str(processing)


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("kind", ["image", "video", "prompt"])
def test_vllm_engine_loads_separate_real_processor_with_original_weights(
    tmp_path, kind, remote, bos_tokenizer, monkeypatch
):
    from pathlib import Path

    import transformers

    pytest.importorskip("torchvision")
    processor = transformers.Qwen3VLProcessor(
        image_processor=transformers.Qwen2VLImageProcessor(),
        tokenizer=bos_tokenizer,
        video_processor=transformers.Qwen3VLVideoProcessor(),
        chat_template="hello",
    )
    checkpoint, processing = tmp_path / "checkpoint", tmp_path / "processing"
    transformers.Qwen3VLConfig().save_pretrained(checkpoint)
    (checkpoint / "model.safetensors").write_bytes(b"fixture weights")
    (checkpoint / "preprocessor_config.json").write_text('{"image_processor_type": "UnrelatedProcessor"}')
    (checkpoint / "video_preprocessor_config.json").write_text('{"video_processor_type": "UnrelatedProcessor"}')
    processor.save_pretrained(processing)
    before = {path.name: path.read_bytes() for path in checkpoint.iterdir()}
    state = install_sdk(monkeypatch.setitem, mock_transformers=False)
    model = "fixture/model" if remote else str(checkpoint)
    downloads = []

    def snapshot(name, **kwargs):
        downloads.append((name, kwargs))
        return str(checkpoint)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    original_config = transformers.AutoConfig.from_pretrained
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda name, **kwargs: original_config(checkpoint if name == model else name, **kwargs),
    )
    engine = sys.modules["vllm"].AsyncLLMEngine
    create = engine.from_engine_args
    model_views = []

    def create_with_processor(args):
        view = Path(args.model)
        model_views.append(view)
        # The SDK loads its processor from model, not tokenizer. Reload from
        # precisely that source while checking the checkpoint is unchanged.
        loaded = transformers.AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
        assert isinstance(loaded, transformers.Qwen3VLProcessor)
        assert loaded.chat_template == "hello"
        assert type(transformers.AutoImageProcessor.from_pretrained(view)) is type(loaded.image_processor)
        assert type(transformers.AutoVideoProcessor.from_pretrained(view)) is type(loaded.video_processor)
        assert (view / "model.safetensors").samefile(checkpoint / "model.safetensors")
        assert (view / "config.json").samefile(checkpoint / "config.json")
        assert args.served_model_name == model
        assert args.tokenizer == str(processing)
        return create(args)

    monkeypatch.setattr(engine, "from_engine_args", create_with_processor)
    options = {
        "gpus_per_actor": 0,
        "engine_args": {
            "tokenizer": str(processing),
            "revision": "weights-revision",
            "tokenizer_revision": "processor-revision",
            "download_dir": str(tmp_path / "cache"),
        },
    }
    provider = load_provider("vllm")
    descriptor = (
        provider.get_prompter(model, options={**options, "media_mime_types": ["image/png"]})
        if kind == "prompt"
        else getattr(provider, f"get_{kind}_embedder")(model, 2, options=options)
    )
    original_options = descriptor.get_options()

    async def run():
        runtime = descriptor.instantiate()
        try:
            runtime._ensure_engine()
            assert runtime._ensure_engine() is state.engines[0]
            assert model_views[0].is_dir()
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert len(model_views) == 1 and not model_views[0].exists()
    assert {path.name: path.read_bytes() for path in checkpoint.iterdir()} == before
    assert descriptor.get_model() == model and descriptor.get_options() == original_options
    assert downloads == (
        [(model, {"revision": "weights-revision", "cache_dir": str(tmp_path / "cache"), "token": None})]
        if remote
        else []
    )


@pytest.mark.parametrize("failure", ["save", "engine", "shutdown"])
def test_vllm_processor_model_view_survives_until_engine_cleanup(tmp_path, failure, sdk, monkeypatch):
    from pathlib import Path

    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    directories = []

    def save(self, path):
        directories.append(Path(path))
        (Path(path) / "preprocessor_config.json").write_text("{}")
        if failure == "save":
            raise RuntimeError("save failed")

    monkeypatch.setattr(sys.modules["transformers"].AutoProcessor, "save_pretrained", save, raising=False)
    engine = sys.modules["vllm"].AsyncLLMEngine
    create = engine.from_engine_args

    def create_or_fail(args):
        if failure == "engine":
            raise RuntimeError("engine failed")
        result = create(args)
        result.shutdown = lambda: (_ for _ in ()).throw(RuntimeError("shutdown failed"))
        return result

    monkeypatch.setattr(engine, "from_engine_args", create_or_fail)

    async def run():
        runtime = (
            load_provider("vllm")
            .get_image_embedder(
                str(checkpoint), 2, options={"gpus_per_actor": 0, "engine_args": {"tokenizer": "separate-processor"}}
            )
            .instantiate()
        )
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            try:
                runtime._ensure_engine()
            finally:
                await runtime.aclose()
        if failure == "shutdown":
            assert runtime.engine is sdk.engines[0] and not runtime.closed
            assert runtime._model_directory is not None and directories[0].is_dir()
            # Restore shutdown and retry while the selected processor is still
            # available to the engine's child processes.
            del runtime.engine.shutdown
            await runtime.aclose()
            assert sdk.engines[0].closed == 1
        assert runtime._model_directory is None and runtime.closed

    asyncio.run(run())
    assert len(directories) == 1 and not directories[0].exists()
    assert list(checkpoint.iterdir()) == [checkpoint / "config.json"]


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize(
    "processing",
    [
        {},
        {"add_special_tokens": False},
        {"text_kwargs": {"padding": False, "truncation": False, "add_special_tokens": False, "return_tensors": "pt"}},
    ],
)
def test_native_template_token_ids_are_submitted_without_retokenization(
    tmp_path, family, processing, bos_tokenizer, monkeypatch
):
    import transformers

    transformers.LlamaConfig().save_pretrained(tmp_path)
    bos_tokenizer.save_pretrained(tmp_path)
    state = install_sdk(monkeypatch.setitem, mock_transformers=False)

    async def run():
        runtime = (
            load_provider(family)
            .get_text_embedder(str(tmp_path), 2, options={"gpus_per_actor": 0, "processor_kwargs": processing})
            .instantiate()
        )
        try:
            assert await runtime.embed_text(["hello"]) == [[3, 4]]
        finally:
            await runtime.aclose()

    asyncio.run(run())
    expected = bos_tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}], tokenize=True, return_dict=False
    )
    assert expected == [1, 3, 2]
    assert bos_tokenizer.encode("<s>hello</s>") == [1, 1, 3, 2, 2]
    if family == "sglang":
        assert state.calls[0][1]["input_ids"] == expected
        assert "prompt" not in state.calls[0][1]
    else:
        # vLLM completion tokenization defaults to adding special tokens. Check
        # the actual engine argument, not the unrelated multimodal kwargs.
        call = state.calls[0]
        assert bos_tokenizer.encode(call[1]["prompt"], **call[3].get("tokenization_kwargs", {})) == expected


@pytest.fixture
def make_real_video_processor():
    transformers = pytest.importorskip("transformers", minversion="4.57.1")
    pytest.importorskip("torchvision")
    from tokenizers import Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import BPE

    specials = ["<unk>", "<pad>", "<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
    vocabulary = specials + sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(BPE({word: index for index, word in enumerate(vocabulary)}, [], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = transformers.Qwen2TokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", additional_special_tokens=specials[2:]
    )

    def make(*, scalar_timing=False):
        processor = transformers.Qwen2_5_VLProcessor if scalar_timing else transformers.Qwen3VLProcessor
        video_processor = transformers.Qwen2VLVideoProcessor if scalar_timing else transformers.Qwen3VLVideoProcessor
        return processor(
            image_processor=transformers.Qwen2VLImageProcessor(),
            tokenizer=tokenizer,
            video_processor=video_processor(),
            chat_template="<|video_pad|>",
        )

    return make


@pytest.fixture
def vllm_processor_methods():
    """Read vLLM's Python preprocessing without importing its CUDA runtime.

    Uses an installed SDK, or VANE_TEST_VLLM_SOURCE pointing to its source
    checkout. No upstream source is copied into Vane or downloaded by tests.
    """
    import ast
    import os
    from collections.abc import Mapping
    from importlib.metadata import PackageNotFoundError, distribution
    from pathlib import Path
    from typing import Any

    import torch
    from transformers.video_utils import VideoMetadata

    configured = os.environ.get("VANE_TEST_VLLM_SOURCE")
    if configured:
        root = Path(configured)
    else:
        try:
            root = Path(distribution("vllm").locate_file(""))
        except PackageNotFoundError:
            pytest.skip("requires installed vLLM SDK or VANE_TEST_VLLM_SOURCE")
    namespace = {
        "Mapping": Mapping,
        "Any": Any,
        "torch": torch,
        "VideoMetadata": VideoMetadata,
        "is_list_of": lambda value, typ: isinstance(value, list) and all(isinstance(item, typ) for item in value),
        "_HF_MODALITY_PROCESSOR_KWARGS": {"image": "images_kwargs", "video": "videos_kwargs", "audio": "audio_kwargs"},
        "HFMultiModalInputs": lambda *values: values,
    }
    for filename, owner, name in (
        ("multimodal/processing/context.py", None, "overlay_modality_mm_kwargs"),
        ("multimodal/processing/processor.py", "BaseMultiModalProcessor", "_get_hf_mm_inputs"),
        ("model_executor/models/qwen3_vl.py", "Qwen3VLMultiModalProcessor", "_apply_hf_processor_main"),
    ):
        path = root / "vllm" / filename
        tree = ast.parse(path.read_text())
        scope = (
            tree
            if owner is None
            else next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
        )
        function = next(node for node in scope.body if isinstance(node, ast.FunctionDef) and node.name == name)
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning:transformers.processing_utils"
)
@pytest.mark.parametrize("kind", ["image", "video"])
@pytest.mark.parametrize("settings", ["default", "nested_text", "flat_size", "nested_size", "scoped_sizes"])
def test_vllm_real_sdk_accepts_processor_options(
    monkeypatch, kind, settings, make_real_video_processor, vllm_processor_methods
):
    import transformers

    processor = make_real_video_processor()
    processor.chat_template = f"<|{kind}_pad|>"
    state = install_sdk(monkeypatch.setitem, mock_transformers=False)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    size = {"shortest_edge": 4096, "longest_edge": 8192}
    options = {}
    if settings == "nested_text":
        options["text_kwargs"] = {"padding": False, "truncation": False, "add_special_tokens": False}
    elif settings == "flat_size":
        options["size"] = size
    elif settings in ("nested_size", "scoped_sizes"):
        options[f"{kind}s_kwargs"] = {"size": size}
        if settings == "scoped_sizes":
            other = "images_kwargs" if kind == "video" else "videos_kwargs"
            options[other] = {"size": {"shortest_edge": 16384, "longest_edge": 32768}}
    descriptor = getattr(load_provider("vllm"), f"get_{kind}_embedder")(
        "configured-model", 2, options={"gpus_per_actor": 0, "processor_kwargs": options}
    )
    before = descriptor.get_options()
    frames = tuple(np.full((64, 64, 3), value, dtype=np.uint8) for value in (10, 20))

    async def run():
        runtime = descriptor.instantiate()
        try:
            value = VideoClip(frames, (0.2, 0.6), (2, 6)) if kind == "video" else frames[0]
            assert await getattr(runtime, f"embed_{kind}")([value]) == [[3, 4]]
        finally:
            await runtime.aclose()

    asyncio.run(run())
    request = state.calls[0][1]
    outputs = []

    class CompletedHFCall(Exception):
        pass

    def call_hf_processor(hf_processor, data, kwargs):
        outputs.append((hf_processor(**data, **kwargs), kwargs))
        # The regression boundary ends after the real SDK's parameter
        # injection and HF token/pixel preprocessing, before model execution.
        raise CompletedHFCall

    methods = vllm_processor_methods
    context = SimpleNamespace(
        get_mm_config=lambda: SimpleNamespace(get_video_pruning_spec=lambda: None),
        get_merged_mm_kwargs=lambda kwargs, modality=None: methods["overlay_modality_mm_kwargs"](kwargs, modality),
        call_hf_processor=call_hf_processor,
    )
    config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        vision_start_token_id=processor.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=processor.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        video_token_id=processor.video_token_id,
    )
    info = SimpleNamespace(
        ctx=context,
        get_hf_config=lambda: config,
        get_tokenizer=lambda: processor.tokenizer,
        get_video_processor=lambda: processor.video_processor,
        get_hf_processor=lambda **kwargs: processor,
        _get_video_second_idx=lambda **kwargs: [0.4],
    )
    adapter = SimpleNamespace(info=info, _get_hf_mm_text=lambda counts: f"<|vision_start|><|{kind}_pad|><|vision_end|>")
    adapter._get_hf_mm_inputs = lambda items, kwargs: methods["_get_hf_mm_inputs"](adapter, items, kwargs)

    class Modalities(dict):
        def get_all_counts(self):
            return {kind: 1}

    items = Modalities(
        {
            kind: SimpleNamespace(
                get_processor_data=lambda: {kind + "s": request["multi_modal_data"][kind]},
                get_passthrough_data=lambda: {},
            )
        }
    )
    with pytest.raises(CompletedHFCall):
        methods["_apply_hf_processor_main"](adapter, items, request["mm_processor_kwargs"])
    assert len(outputs) == 1
    output, forwarded = outputs[0]
    assert forwarded["truncation"] is False and forwarded["add_special_tokens"] is False
    assert not {"text_kwargs", "images_kwargs", "videos_kwargs"} & forwarded.keys()
    if "size" in settings:
        assert forwarded["size"] == size
    assert output[f"{kind}_grid_thw"][0][0] == 1
    assert descriptor.get_options() == before


# Transformers 5.12 builds modality IDs with np.array(torch.Tensor), whose
# NumPy 2 copy-keyword warning is unrelated to the processor-output contract.
@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning:transformers.processing_utils"
)
@pytest.mark.parametrize(
    "text_options",
    [
        {},
        {"text_kwargs": {}},
        {"text_kwargs": {"add_special_tokens": False}},
        {"text_kwargs": {"truncation": False}},
        {"text_kwargs": {"padding": False}},
        {"text_kwargs": {"padding": False, "truncation": False, "add_special_tokens": False, "return_tensors": "pt"}},
    ],
)
def test_sglang_with_real_video_processor_retains_timestamps(monkeypatch, text_options, make_real_video_processor):
    import transformers

    processor = make_real_video_processor()
    state = install_sdk(monkeypatch.setitem, mock_transformers=False)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    descriptor = load_provider("sglang").get_video_embedder(
        "custom-local-checkpoint",
        2,
        options={
            "gpus_per_actor": 0,
            "processor_kwargs": {
                "videos_kwargs": {"size": {"shortest_edge": 4096, "longest_edge": 8192}, "do_sample_frames": False},
                **text_options,
            },
        },
    )
    before = descriptor.get_options()
    frames = tuple(np.full((64, 64, 3), value, dtype=np.uint8) for value in (10, 20, 30))

    async def run():
        runtime = descriptor.instantiate()
        try:
            assert await runtime.embed_video([VideoClip(frames, (0.2, 0.6, 1.4), (2, 6, 14))]) == [[3, 4]]
        finally:
            await runtime.aclose()

    asyncio.run(run())
    request = state.calls[0][1]
    text = processor.tokenizer.decode(request["input_ids"], skip_special_tokens=False)
    assert "<0.4 seconds>" in text and "<1.4 seconds>" in text
    assert request["video_data"][0]["format"] == "processor_output"
    assert request["input_ids"] == request["video_data"][0]["input_ids"][0].tolist()
    assert descriptor.get_options() == before


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning:transformers.processing_utils"
)
@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("start", [0.0, 10.0])
@pytest.mark.parametrize("interval", [0.5, 1 / 30])
def test_real_video_processor_preserves_sampling_interval(
    monkeypatch, family, start, interval, make_real_video_processor
):
    import transformers
    from transformers.video_utils import VideoMetadata

    processor = make_real_video_processor(scalar_timing=True)
    state = install_sdk(monkeypatch.setitem, mock_transformers=False)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    descriptor = load_provider(family).get_video_embedder(
        "custom-local-checkpoint",
        2,
        options={
            "gpus_per_actor": 0,
            "processor_kwargs": {
                "videos_kwargs": {"size": {"shortest_edge": 4096, "longest_edge": 8192}},
            },
        },
    )
    before = descriptor.get_options()
    frames = tuple(np.full((64, 64, 3), value, dtype=np.uint8) for value in (10, 20, 30, 40))
    times = tuple(start + i * interval for i in range(4))

    async def run():
        runtime = descriptor.instantiate()
        try:
            assert await runtime.embed_video([VideoClip(frames, times, (1, 5, 9, 13))]) == [[3, 4]]
        finally:
            await runtime.aclose()

    asyncio.run(run())
    request = state.calls[0][1]
    if family == "vllm":
        assert request["mm_processor_kwargs"]["do_sample_frames"] is False
        assert "videos_kwargs" not in request["mm_processor_kwargs"]
        array, metadata = request["multi_modal_data"]["video"][0]
        # vLLM's native adapter reconstructs HF metadata from this dictionary.
        # Exercise the submitted metadata/options with the real processor.
        output = processor(
            text=[request["prompt"]],
            videos=[array],
            video_metadata=[
                VideoMetadata(**{key: value for key, value in metadata.items() if key != "do_sample_frames"})
            ],
            **request["mm_processor_kwargs"],
        )
    else:
        output = request["video_data"][0]
    np.testing.assert_allclose(output["second_per_grid_ts"], [2 * interval], atol=1e-6, rtol=0)
    assert output["video_grid_thw"][0][0] == 2  # Four frames, without resampling.
    assert descriptor.get_options() == before


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning:transformers.processing_utils"
)
@pytest.mark.parametrize("kind", ["text", "image"])
def test_sglang_real_processor_accepts_nested_text_options(monkeypatch, kind, make_real_video_processor):
    import transformers

    processor = make_real_video_processor()
    processor.chat_template = "hello" if kind == "text" else "<|image_pad|>"
    state = install_sdk(monkeypatch.setitem, mock_transformers=False)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    descriptor = getattr(load_provider("sglang"), f"get_{kind}_embedder")(
        "custom-local-checkpoint",
        2,
        options={
            "gpus_per_actor": 0,
            "processor_kwargs": {
                "text_kwargs": {"padding": False, "truncation": False, "add_special_tokens": False},
            },
        },
    )

    async def run():
        runtime = descriptor.instantiate()
        try:
            value = "hello" if kind == "text" else np.zeros((64, 64, 3), dtype=np.uint8)
            assert await getattr(runtime, f"embed_{kind}")([value]) == [[3, 4]]
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert state.requests_closed == 1 and state.engines[0].closed == 1


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.real_ray
def test_native_descriptors_run_on_existing_ray_worker(family, ray_local):
    import cloudpickle
    import ray

    cloudpickle.register_pickle_by_value(sys.modules[__name__])

    descriptor = load_provider(family).get_prompter(
        "configured-model", options={"media_mime_types": ["video/mp4"], "gpus_per_actor": 0}
    )
    embedding = load_provider(family).get_video_embedder(
        "configured-encoder", 2, options={"gpus_per_actor": 0, "max_concurrency_per_actor": 2}
    )

    @ray.remote
    def execute(prompt_desc, embed_desc, install):
        previous = {}

        def replace(mapping, key, value):
            previous[key] = mapping.get(key)
            mapping[key] = value

        state = install(replace)

        async def run():
            prompt_runtime, embed_runtime = prompt_desc.instantiate(), embed_desc.instantiate()
            active, peak = 0, 0
            ready = asyncio.Event()

            async def request(value):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                if active == 2:
                    ready.set()
                try:
                    await asyncio.wait_for(ready.wait(), 5)
                    return [3.0, 4.0]
                finally:
                    active -= 1

            state.on_embedding = request
            try:
                answer = await prompt_runtime.prompt(("question", PromptMedia(b"video", "video/mp4")))
                text_answer = await prompt_runtime.prompt(("question",))
                result = await embed_runtime.embed_video([clip()] * 4)
                assert peak == 2 and active == 0 and state.requests_closed == 4
                return answer, text_answer, result
            finally:
                await prompt_runtime.aclose()
                await embed_runtime.aclose()

        try:
            result = asyncio.run(run())
        finally:
            for key, value in previous.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value
        assert len(state.engines) == 2 and all(engine.closed == 1 for engine in state.engines)
        return result

    assert ray.get(execute.remote(descriptor, embedding, install_sdk), timeout=30) == (
        '{"answer":"ok"}',
        '{"answer":"ok"}',
        [[3.0, 4.0]] * 4,
    )


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("entry", ["python", "sql"])
def test_public_native_prompt_and_embedding_with_null_rows(family, entry, monkeypatch, tmp_path):
    import cloudpickle

    import vane
    from vane.ai import embed, prompt
    from vane.ai.provider import PROVIDERS
    from vane.ai.providers._native_inference import NativeMediaPrompterDescriptor, NativeTextEmbedderDescriptor

    cloudpickle.register_pickle_by_value(sys.modules[__name__])

    class WorkerPrompt(NativeMediaPrompterDescriptor):
        def instantiate(self):
            install_sdk(lambda mapping, key, value: mapping.__setitem__(key, value))
            return super().instantiate()

    class WorkerEmbedding(NativeTextEmbedderDescriptor):
        def instantiate(self):
            state = install_sdk(lambda mapping, key, value: mapping.__setitem__(key, value))
            active = 0
            ready = asyncio.Event()

            async def request(value):
                nonlocal active
                active += 1
                if active == 2:
                    ready.set()
                try:
                    await asyncio.wait_for(ready.wait(), 5)
                    return [3.0, 4.0]
                finally:
                    active -= 1

            state.on_embedding = request
            return super().instantiate()

    base = type(load_provider(family))

    class WorkerProvider(base):
        def get_prompter(self, *args, **kwargs):
            descriptor = super().get_prompter(*args, **kwargs)
            descriptor.__class__ = WorkerPrompt
            return descriptor

        def get_text_embedder(self, *args, **kwargs):
            descriptor = super().get_text_embedder(*args, **kwargs)
            descriptor.__class__ = WorkerEmbedding
            return descriptor

    monkeypatch.setitem(PROVIDERS, family, WorkerProvider)
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fixture-video")
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    with vane.connect() as conn:
        conn.sql(
            """create table prompts as select id, 'question' as q,
            case when id = 0 then NULL::FILE else file(?, 'video/mp4', NULL, NULL, NULL) end as media
            from range(2) t(id)""",
            params=[str(media)],
        )
        if entry == "python":
            relation = conn.sql("select * from (values (NULL::VARCHAR), ('first'), ('second')) t(q)")
            vectors = embed(
                relation,
                vane.col("q"),
                provider=family,
                model="configured-encoder",
                dimensions=2,
                gpus_per_actor=0,
                normalize=True,
                max_retries=0,
                batch_size=4,
                max_concurrency_per_actor=2,
            ).fetchall()
            assert vectors[0][1] is None
            np.testing.assert_allclose([row[1] for row in vectors[1:]], [[0.6, 0.8]] * 2)
            result = prompt(
                conn.sql("select * from prompts order by id"),
                [vane.col("q"), vane.col("media")],
                provider=family,
                model="configured-vlm",
                gpus_per_actor=0,
                media_mime_types=["video/mp4"],
                return_format=schema,
                max_retries=0,
            ).fetchall()
        else:
            import json

            vectors = conn.sql(
                """select ai_embed(q, provider => ?, model => 'configured-encoder', dimensions => 2,
                options => {gpus_per_actor: 0, normalize: true, max_retries: 0,
                            batch_size: 4, max_concurrency_per_actor: 2})
                from (values (NULL::VARCHAR), ('first'), ('second')) t(q)""",
                params=[family],
            ).fetchall()
            assert vectors[0][0] is None
            np.testing.assert_allclose([row[0] for row in vectors[1:]], [[0.6, 0.8]] * 2)
            result = conn.sql(
                """select id, ai_prompt(q, media,
                provider => ?, model => 'configured-vlm', return_format => ?,
                options => {gpus_per_actor: 0, media_mime_types: ['video/mp4']}) from prompts order by id""",
                params=[family, json.dumps(schema)],
            ).fetchall()
        assert [(row[0], row[-1]) for row in result] == [(0, {"answer": "ok"}), (1, {"answer": "ok"})]
