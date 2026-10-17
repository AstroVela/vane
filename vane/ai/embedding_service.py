# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""One persistent native embedding engine, served over Linux local IPC.

Run ``python -m vane.ai.embedding_service --config service.json``. Only JSON
configuration is accepted; importing this module starts no engine or listener.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import signal
import socket
import stat
import time
from pathlib import Path
from typing import Any

import numpy as np

from vane.ai import _embedding_ipc as ipc
from vane.ai._embedding_content import EmbeddingContent
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._errors import _is_resource_error


class _Credits:
    def __init__(self, max_bytes: int, max_inflight: int):
        self.max_bytes, self.max_inflight = max_bytes, max_inflight
        self.bytes = self.inflight = 0
        self.condition = asyncio.Condition()

    async def acquire(self, size: int) -> None:
        if size > self.max_bytes:
            raise EmbeddingConfigurationError("Embedding exceeds service shared-memory budget")
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.bytes + size <= self.max_bytes and self.inflight < self.max_inflight
            )
            self.bytes += size
            self.inflight += 1

    async def release(self, size: int) -> None:
        # Returning an owned reservation is mandatory cleanup. Repeated handler
        # cancellation must not interrupt waiting for the condition lock.
        while True:
            try:
                async with self.condition:
                    self.bytes -= size
                    self.inflight -= 1
                    self.condition.notify_all()
                return
            except asyncio.CancelledError:
                continue


class EmbeddingService:
    """Service lifecycle. The runtime is created once, before the socket opens.

    Cancellation after mapping drains the accepted request. Credits and mappings
    remain owned until its coroutine has finished. It deliberately does not
    assume SDK cancellation means a processor has stopped reading CPU arrays.
    """

    def __init__(
        self,
        *,
        path: str,
        service_id: str,
        family: str,
        model: str,
        dimensions: int,
        options: dict[str, Any] | None = None,
        max_bytes: int = 256 * 1024**2,
        max_inflight: int = 8,
        max_connections: int = 128,
        publication_timeout: float = 30,
    ):
        from vane.ai.providers._embedding_service import ServiceMultimodalDescriptor
        from vane.ai.providers._native_inference import NativeMultimodalEmbedderDescriptor

        ipc.require_linux()
        if family not in {"sglang", "vllm"}:
            raise EmbeddingConfigurationError("Embedding service family must be sglang or vllm")
        for name, value, maximum in (
            ("max_bytes", max_bytes, 2**40),
            ("max_inflight", max_inflight, 1024),
            ("max_connections", max_connections, 4096),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise EmbeddingConfigurationError(f"Invalid embedding service {name}")
        if (
            max_connections < max_inflight
            or type(publication_timeout) not in (float, int)
            or not math.isfinite(publication_timeout)
            or publication_timeout <= 0
        ):
            raise EmbeddingConfigurationError("Invalid connection or publication limits")
        ServiceMultimodalDescriptor(
            model, dimensions, {"runtime": "service", "service_path": path, "service_id": service_id}, family, family
        )
        self.descriptor = NativeMultimodalEmbedderDescriptor(model, dimensions, dict(options or {}), family, family)
        self.spec = self.descriptor.get_input_spec()
        self.path, self.identity = path, service_id
        self.credits = _Credits(max_bytes, max_inflight)
        self.max_connections, self.publication_timeout = max_connections, publication_timeout
        self.runtime: Any = None
        self.listener: socket.socket | None = None
        self._accept_task: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._connections: set[socket.socket] = set()
        self._inode: int | None = None
        self.metrics = {
            "accepted": 0,
            "completed": 0,
            "queue_seconds": 0.0,
            "inference_seconds": 0.0,
            "cleanup_seconds": 0.0,
        }

    async def start(self) -> None:
        if self.runtime is not None or self.listener is not None:
            raise RuntimeError("Embedding service is already started")
        parent = Path(self.path).parent
        info = parent.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise EmbeddingConfigurationError("Embedding socket requires a private directory owned by this UID (0700)")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.setblocking(False)
        try:
            # Existing paths are never removed or replaced, including stale
            # sockets. The operator owns naming and service restart identity.
            listener.bind(self.path)
            self._inode = os.lstat(self.path).st_ino
            os.chmod(self.path, 0o600)
            self.runtime = self.descriptor.instantiate()
            self.runtime._ensure_engine()
            listener.listen(self.max_connections)
        except BaseException:
            listener.close()
            await self.aclose()
            raise
        self.listener = listener
        self._accept_task = asyncio.create_task(self._accept())

    async def _accept(self) -> None:
        assert self.listener is not None
        while True:
            sock, _ = await asyncio.get_running_loop().sock_accept(self.listener)
            sock.setblocking(False)
            if len(self._tasks) >= self.max_connections:
                sock.close()
                continue
            self._connections.add(sock)
            task = asyncio.create_task(self._handle(sock))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        if self._accept_task is not None:
            self._accept_task.cancel()
            await asyncio.gather(self._accept_task, return_exceptions=True)
            self._accept_task = None
        if self.listener is not None:
            self.listener.close()
            self.listener = None
        for sock in tuple(self._connections):
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
        # Do not cancel model coroutines: shutdown is joined before unmapping.
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self.runtime is not None:
            await self.runtime.aclose()
            self.runtime = None
        if self._inode is not None:
            with contextlib.suppress(FileNotFoundError):
                if os.lstat(self.path).st_ino == self._inode:
                    os.unlink(self.path)
            self._inode = None

    def _hello(self, message: Any) -> int:
        fields = {"version", "service_id", "family", "model", "dimensions", "request_id", "parts", "nbytes"}
        if not isinstance(message, dict) or set(message) != fields:
            raise EmbeddingConfigurationError("Invalid embedding reservation")
        expected = {
            "version": ipc.VERSION,
            "service_id": self.identity,
            "family": self.descriptor.family,
            "model": self.descriptor.model,
            "dimensions": self.descriptor.dimensions,
        }
        if any(type(message[key]) is not type(value) or message[key] != value for key, value in expected.items()):
            raise EmbeddingConfigurationError("Embedding service/model identity mismatch")
        identity = message["request_id"]
        if not isinstance(identity, str) or len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity):
            raise EmbeddingConfigurationError("Invalid embedding request identity")
        return ipc.validate_layout(message["parts"], message["nbytes"], self.spec)

    async def _handle(self, sock: socket.socket) -> None:
        import fcntl
        import mmap

        request_id: Any = None
        reserved = 0
        fd: ipc.ReceivedFD | None = None
        memory: mmap.mmap | None = None
        content: EmbeddingContent | None = None
        try:
            ipc.check_peer(sock)
            message = await asyncio.wait_for(ipc.control(sock), self.publication_timeout)
            if isinstance(message, dict):
                request_id = message.get("request_id")
            size = self._hello(message)
            queued = time.monotonic()
            admission = asyncio.create_task(self.credits.acquire(size))
            disconnected = asyncio.create_task(asyncio.get_running_loop().sock_recv(sock, 1))
            try:
                await asyncio.wait({admission, disconnected}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                if not admission.done():
                    admission.cancel()
                elif not admission.cancelled() and admission.exception() is None:
                    # Acquisition has no suspension after charging credits.
                    # Adopt them before the next await so another cancellation
                    # cannot strand a completed admission outside cleanup.
                    reserved = size
                disconnected.cancel()
                await asyncio.gather(admission, disconnected, return_exceptions=True)
            if not reserved:
                if not admission.cancelled():
                    admission.result()
                return
            self.metrics["queue_seconds"] += time.monotonic() - queued
            await ipc.send(sock, {"request_id": request_id, "state": "reserved"})
            published, fd = await asyncio.wait_for(ipc.receive(sock), self.publication_timeout)
            if published != {"request_id": request_id, "state": "published"} or (fd is None) != (
                message["nbytes"] == 0
            ):
                raise EmbeddingConfigurationError("Invalid embedding publication")
            if fd is not None:
                info = os.fstat(fd.value)
                try:
                    seals = fcntl.fcntl(fd.value, fcntl.F_GET_SEALS)
                except OSError:
                    raise EmbeddingConfigurationError("Embedding buffer does not support immutable sealing") from None
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_size != message["nbytes"]
                    or seals & ipc.required_seals() != ipc.required_seals()
                ):
                    raise EmbeddingConfigurationError("Embedding buffer must have immutable size and contents")
                memory = mmap.mmap(fd.value, info.st_size, access=mmap.ACCESS_COPY)
                fd.close()
                fd = None
            content = ipc.restore(message["parts"], memory)
            # PCM values are validated only after the bounded mapping exists.
            ipc.describe(content, self.spec)
            self.metrics["accepted"] += 1
            started = time.monotonic()
            # A disconnect is harmless: SCM_RIGHTS transferred ownership. Drain
            # processing even when the producer has already exited or timed out.
            inference = asyncio.create_task(self._infer(content))
            while True:
                try:
                    response = await asyncio.shield(inference)
                    break
                except asyncio.CancelledError:
                    # Even explicit handler cancellation must join processing.
                    # A second cancel cannot bypass the mapping ownership gate.
                    if inference.cancelled():
                        raise RuntimeError("Embedding runtime cancelled before consumption acknowledgement") from None
            self.metrics["inference_seconds"] += time.monotonic() - started
            content = None
            cleaned = time.monotonic()
            if memory is not None:
                memory.close()
                memory = None
            await self.credits.release(reserved)
            reserved = 0
            self.metrics["cleanup_seconds"] += time.monotonic() - cleaned
            self.metrics["completed"] += 1
            await ipc.send(sock, {"request_id": request_id, "state": "released", **response})
        except (Exception, asyncio.CancelledError) as error:
            category = (
                "resource"
                if _is_resource_error(error)
                else "configuration"
                if isinstance(error, (EmbeddingConfigurationError, ValueError, TypeError))
                else "execution"
            )
            # Opaque categories only; never send raw input/SDK exception text.
            with contextlib.suppress(Exception):
                await ipc.send(sock, {"request_id": request_id, "error": category})
        finally:
            content = None
            if fd is not None:
                fd.close()
            if memory is not None:
                memory.close()
            if reserved:
                await self.credits.release(reserved)
            self._connections.discard(sock)
            sock.close()

    async def _infer(self, content: EmbeddingContent) -> dict[str, Any]:
        try:
            vector = np.asarray(await self.runtime._embed_content(content), dtype=np.float32)
            if vector.shape != (self.descriptor.dimensions,) or not np.isfinite(vector).all():
                raise ValueError("Invalid embedding result")
            return {"vector": vector.tolist()}
        except EmbeddingConfigurationError:
            return {"error": "configuration"}
        except Exception as error:
            return {"error": "resource" if _is_resource_error(error) else "execution"}


async def _run(config: dict[str, Any]) -> None:
    service = EmbeddingService(**config)
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stopped.set)
    try:
        await service.start()
        await stopped.wait()
    finally:
        await service.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    asyncio.run(_run(config))


if __name__ == "__main__":
    main()
