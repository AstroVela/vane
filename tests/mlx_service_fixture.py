# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Actual service/client subprocesses and test-only, payload-free observations."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import os
import select
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

PREFIX = "MLX_SERVICE_CLIENT:"
MODEL = "mlx-community/embeddinggemma-2-bf16"
REVISION = "1a4ffddb7905d3f63486748deabe091a01fb6201"


def save_case(path, row, encoding):
    arrays = {}

    def array(value):
        key = "array_" + str(len(arrays))
        arrays[key] = value
        return dict(array=key, strided=not value.flags.c_contiguous, readonly=not value.flags.writeable)

    parts = []
    for part in row.parts:
        kind, value = part.spec.kind, part.value
        data = dict(kind=kind)
        if kind == "text":
            data["text"] = value
        elif kind == "image":
            data["image"] = array(value)
        elif kind == "audio":
            data.update(samples=array(value.samples), sample_rate=int(value.sample_rate))
        else:
            data.update(
                frames=[array(frame) for frame in value.frames],
                times=list(value.frame_times),
                indices=list(value.frame_indices),
            )
        parts.append(data)
    np.savez(path, metadata=np.array(json.dumps(dict(encoding=encoding, parts=parts))), **arrays)


def load_case(path):
    from vane.ai import AudioClip, EmbeddingContent, EmbeddingPart, EmbeddingPartSpec, VideoClip

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))

        def array(description):
            value = archive[description["array"]].copy()
            if description["strided"]:
                backing = np.empty((value.shape[0] * 2, *value.shape[1:]), value.dtype)
                backing[::2] = value
                value = backing[::2]
            if description["readonly"]:
                value.setflags(write=False)
            return value

        parts = []
        for part in metadata["parts"]:
            kind = part["kind"]
            if kind == "text":
                value = part["text"]
            elif kind == "image":
                value = array(part["image"])
            elif kind == "audio":
                value = AudioClip(array(part["samples"]), part["sample_rate"])
            else:
                value = VideoClip(
                    tuple(array(frame) for frame in part["frames"]), tuple(part["times"]), tuple(part["indices"])
                )
            parts.append(EmbeddingPart(EmbeddingPartSpec(kind), value))
    return metadata["encoding"], EmbeddingContent(tuple(parts))


def clean_environment(interpreter):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["VIRTUAL_ENV"] = str(Path(interpreter).parent.parent)
    env["PATH"] = str(Path(interpreter).parent) + os.pathsep + env["PATH"]
    env["VANE_RUNNER"] = "local"
    return env


class ServiceProcess:
    def __init__(self, root, *, interpreter=None, traced=True, max_length=2048):
        self.root = Path(root)
        self.root.mkdir(mode=0o700, exist_ok=True)
        self.path = self.root / "mlx.json"
        self.trace = self.root / ("events-" + uuid.uuid4().hex[:8] + ".jsonl")
        self.config = dict(
            path=str(self.path),
            service_id="metal-shared-v1",
            family="mlx",
            model=MODEL,
            dimensions=768,
            options=dict(revision=REVISION, dtype="bfloat16", max_length=max_length, local_files_only=True),
            max_inflight=4,
            max_connections=16,
            max_bytes=256 * 1024**2,
        )
        config = self.root / "config.json"
        config.write_text(json.dumps(self.config))
        interpreter = interpreter or os.environ.get("VANE_MLX_SERVICE_PYTHON", sys.executable)
        command = (
            [interpreter, "-I", str(Path(__file__).resolve()), "server", str(config), str(self.trace)]
            if traced
            else [interpreter, "-I", "-m", "vane.ai.embedding_service", "--config", str(config)]
        )
        self.log = (self.root / "server.log").open("w")
        self.process = subprocess.Popen(
            command, env=clean_environment(interpreter), stdout=self.log, stderr=subprocess.STDOUT
        )
        self.traced = traced

    @property
    def options(self):
        return dict(runtime="service", service_path=str(self.path), service_id=self.config["service_id"], timeout=120)

    def ready(self, timeout=120):
        from vane.ai._embedding_zmq import _read_endpoint

        until = time.monotonic() + timeout
        while time.monotonic() < until:
            assert self.process.poll() is None, f"service exited; see {self.root / 'server.log'}"
            try:
                _read_endpoint(str(self.path))
                return
            except (OSError, ValueError):
                time.sleep(0.02)
        raise AssertionError(f"service did not become ready; see {self.root / 'server.log'}")

    def events(self):
        return [json.loads(line) for line in self.trace.read_text().splitlines()] if self.trace.exists() else []

    def wait_event(self, event, *, after=0, timeout=30):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            events = self.events()
            matches = [row for row in events[after:] if row["event"] == event]
            if matches:
                return matches[-1]
            assert self.process.poll() is None, f"service exited; see {self.root / 'server.log'}"
            time.sleep(0.01)
        raise AssertionError(f"missing service event {event}; see {self.trace}")

    def state(self):
        return json.loads(self.trace.with_suffix(".state.json").read_text())

    def stop(self):
        (self.root / "release").touch()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
                raise AssertionError("service failed to drain; forced termination is a test failure")
        self.log.close()
        assert self.process.returncode == 0, f"service failed; see {self.root / 'server.log'}"
        assert not self.path.exists()


class ClientProcess:
    def __init__(self, root, options, *, interpreter=None):
        self.root = Path(root)
        self.root.mkdir()
        interpreter = interpreter or os.environ["VANE_MLX_SERVICE_CLIENT_PYTHON"]
        config = self.root / "client.json"
        config.write_text(json.dumps(options))
        self.stderr = (self.root / "client.log").open("w")
        self.process = subprocess.Popen(
            [interpreter, "-I", str(Path(__file__).resolve()), "client", str(config)],
            env=clean_environment(interpreter),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )
        self.read()

    def read(self, timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if select.select([self.process.stdout], [], [], 0.1)[0]:
                line = self.process.stdout.readline()
                if line.startswith(PREFIX):
                    result = json.loads(line[len(PREFIX) :])
                    assert result.get("ok"), result
                    assert result.get("sdk_imported") is False
                    return result
            assert self.process.poll() is None, f"client exited; see {self.root / 'client.log'}"
        raise AssertionError("independent client timed out")

    def send(self, **request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()

    def ask(self, **request):
        self.send(**request)
        return self.read()

    def close(self):
        if self.process.poll() is None:
            self.ask(action="close")
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise AssertionError("client failed to join its Vane workers")
        self.stderr.close()
        assert self.process.returncode == 0


def _client_main(config):
    import builtins
    import importlib.util

    denied = {"mlx", "mlx_vlm", "transformers", "torch", "PIL"}
    for name in denied:
        assert importlib.util.find_spec(name) is None, name
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        assert name.split(".")[0] not in denied, "client attempted to import a model SDK"
        return original(name, *args, **kwargs)

    builtins.__import__ = guard
    import vane
    from vane.ai import TextInput, embed
    from vane.ai.provider import load_provider

    options = json.loads(Path(config).read_text())

    def reply(**values):
        print(
            PREFIX
            + json.dumps(dict(ok=True, pid=os.getpid(), sdk_imported=bool(denied & sys.modules.keys()), **values)),
            flush=True,
        )

    async def direct(paths):
        vectors = []
        for path in paths:
            encoding, row = load_case(path)
            kind = "text" if encoding in {"text_query", "text_document"} else encoding
            opts = options | (
                {"input_type": "query" if encoding == "text_query" else "document"} if kind == "text" else {}
            )
            client = getattr(load_provider("mlx"), "get_" + kind + "_embedder")(options=opts).instantiate()
            try:
                values = [row] if kind == "multimodal" else [row.parts[0].value]
                vectors.extend(await getattr(client, "embed_" + kind)(values))
            finally:
                await client.aclose()
        return np.asarray(vectors, np.float32)

    connection = vane.connect()
    try:
        reply(ready=True)
        for line in sys.stdin:
            request = json.loads(line)
            if request["action"] == "close":
                reply(closed=True)
                break
            if request["action"] == "direct":
                result = asyncio.run(direct(request["paths"]))
            else:
                import pyarrow as pa

                relation = connection.from_arrow(pa.table({"text": request["texts"]}))
                result = (
                    embed(
                        relation,
                        TextInput(vane.col("text")),
                        provider="mlx",
                        input_type=request["role"],
                        batch_size=1,
                        actor_number=1,
                        max_retries=9,
                        **options,
                    )
                    .select("embedding")
                    .to_arrow_table()
                )
                assert result.column("embedding").type == pa.list_(pa.float32(), 768)
                result = np.asarray(result.column("embedding").to_pylist(), np.float32)
            np.save(request["output"], result, allow_pickle=False)
            reply(rows=len(result), output_sha256=hashlib.sha256(Path(request["output"]).read_bytes()).hexdigest())
    finally:
        connection.close()
        vane.teardown_runner()


def _server_main(config, trace_path):
    import vane.ai.embedding_service as service_module
    from vane.ai.providers import _embedding_service_backend as factories
    from vane.ai.providers import _mlx_service as backend_module
    from vane.ai.providers.mlx import _MLXModelCore

    trace = Path(trace_path)
    root = trace.parent
    current = contextvars.ContextVar("observed_mlx_request", default=None)
    active = 0

    def event(name, **values):
        import threading

        with trace.open("a") as log:
            log.write(
                json.dumps(
                    dict(
                        timestamp=time.monotonic(),
                        pid=os.getpid(),
                        thread_id=threading.get_ident(),
                        event=name,
                        **values,
                    )
                )
                + "\n"
            )

    class Core(_MLXModelCore):
        def _load_processor(self, *args):
            super()._load_processor(*args)
            event("processor_loaded", model_instance_id=id(self._model))

        def _infer(self, inputs, count):
            nonlocal active
            request = current.get()
            active += 1
            values = dict(
                model_instance_id=id(self._model),
                request_id=request.request_id,
                encoding=request.encoding,
                active_forward_count=active,
            )
            event(
                "forward_start",
                **values,
                tensors={
                    k: dict(shape=list(v.shape), sha256=hashlib.sha256(v.tobytes()).hexdigest())
                    for k, v in inputs.items()
                },
            )
            try:
                if (root / "gate").exists() and not (root / "release").exists():
                    event("blocked", **values)
                    deadline = time.monotonic() + 60
                    while not (root / "release").exists():
                        if time.monotonic() >= deadline:
                            raise TimeoutError("test forward barrier expired")
                        time.sleep(0.01)
                if (root / "oom").exists():
                    event("oom_injected", **values)
                    raise RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory")
                return super()._infer(inputs, count)
            finally:
                active -= 1
                event("forward_end", **(values | {"active_forward_count": active}))

        def close(self):
            previous, identity = self._closed, id(self._model)
            try:
                super().close()
            finally:
                if not previous:
                    event("close", model_instance_id=identity)

    def create(options):
        import mlx_vlm.embedding_loader as loader

        original = loader.load_embedding_model

        def loaded(*args, **kwargs):
            model = original(*args, **kwargs)
            event("load", model_instance_id=id(model))
            return model

        loader.load_embedding_model = loaded
        try:
            return Core(options)
        finally:
            loader.load_embedding_model = original

    backend_module._create_core = create
    factory = factories.create_service_backend

    def observed_factory(*args, **kwargs):
        backend = factory(*args, **kwargs)
        original = backend._encode

        def encode(row, request):
            token = current.set(request)
            try:
                capture = root / "captures"
                if (root / "capture").exists():
                    capture.mkdir(exist_ok=True)
                    save_case(capture / (request.request_id + ".npz"), row, request.encoding)
                result = original(row, request)
                if (root / "capture").exists():
                    np.save(capture / (request.request_id + ".npy"), result, allow_pickle=False)
                return result
            finally:
                current.reset(token)

        backend._encode = encode
        return backend

    factories.create_service_backend = observed_factory

    class Service(service_module.EmbeddingService):
        async def start(self):
            await super().start()
            self.observed_transport = self._zmq_server
            self.reporter = asyncio.create_task(self.report())
            event("ready")

        async def report(self):
            while True:
                data = dict(
                    bytes=self.credits.bytes,
                    inflight=self.credits.inflight,
                    buffers=sorted(self.observed_transport.buffers),
                    state=self.backend.state,
                )
                temporary = trace.with_suffix(".state.tmp")
                temporary.write_text(json.dumps(data))
                temporary.replace(trace.with_suffix(".state.json"))
                await asyncio.sleep(0.01)

        async def _close(self):
            await super()._close()
            if hasattr(self, "reporter"):
                self.reporter.cancel()
                await asyncio.gather(self.reporter, return_exceptions=True)
                event(
                    "closed",
                    bytes=self.credits.bytes,
                    inflight=self.credits.inflight,
                    buffers=sorted(self.observed_transport.buffers),
                    executor_closed=self.backend._executor is None,
                )

    service_module.EmbeddingService = Service
    asyncio.run(service_module._run(json.loads(Path(config).read_text())))


if __name__ == "__main__":
    if sys.argv[1] == "client":
        _client_main(sys.argv[2])
    else:
        _server_main(sys.argv[2], sys.argv[3])
