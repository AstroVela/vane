# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Authenticated loopback control messages and service-owned shared memory.

Only descriptors cross ZeroMQ. A private rendezvous file contains a fresh local
endpoint and authentication key, never a provider option or serialized secret.
The server allocates and tracks buffers; consumers take a bounded private
snapshot before validation/inference because named shared memory has no seals.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from vane.ai import _embedding_ipc as ipc
from vane.ai import _embedding_shared_memory as shm
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._errors import _is_resource_error

PROTOCOL = "vane-embedding-zmq-1"
MAX_ENDPOINT_BYTES = 4096


def _zmq() -> Any:
    try:
        import zmq.asyncio
    except ImportError:
        from vane.ai.provider import ProviderImportError

        raise ProviderImportError("embedding_service") from None
    return zmq


def _private(info: os.stat_result) -> None:
    if os.name == "posix" and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077):
        raise EmbeddingConfigurationError("Embedding rendezvous requires a private file/directory owned by this UID")


def _read_endpoint(path: str) -> tuple[str, bytes]:
    _private(Path(path).parent.stat())
    if Path(path).is_symlink():
        raise EmbeddingConfigurationError("Embedding rendezvous must not be a symbolic link")
    # A misconfigured FIFO must not block the event loop before fstat rejects it.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        _private(info)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_ENDPOINT_BYTES:
            raise EmbeddingConfigurationError("Invalid embedding rendezvous file")
        try:
            data = json.loads(os.read(fd, MAX_ENDPOINT_BYTES + 1))
        except (ValueError, UnicodeError):
            raise EmbeddingConfigurationError("Invalid embedding rendezvous JSON") from None
    finally:
        os.close(fd)
    if not isinstance(data, dict) or set(data) != {"protocol", "endpoint", "key"} or data["protocol"] != PROTOCOL:
        raise EmbeddingConfigurationError("Embedding rendezvous protocol mismatch")
    endpoint, key = data["endpoint"], data["key"]
    if not isinstance(endpoint, str) or not re.fullmatch(r"tcp://127\.0\.0\.1:[1-9][0-9]{0,4}", endpoint):
        raise EmbeddingConfigurationError("Embedding service requires a loopback endpoint")
    if int(endpoint.rsplit(":", 1)[1]) > 65535 or not isinstance(key, str) or not re.fullmatch("[0-9a-f]{64}", key):
        raise EmbeddingConfigurationError("Invalid embedding rendezvous configuration")
    return endpoint, bytes.fromhex(key)


def _pack(message: Any, key: bytes, direction: bytes) -> list[bytes]:
    body = ipc.encode(message)
    return [body, hmac.digest(key, direction + body, hashlib.sha256)]


def _unpack(frames: list[bytes], key: bytes, direction: bytes) -> Any:
    if len(frames) != 2 or not 0 < len(frames[0]) <= ipc.MAX_CONTROL_BYTES or len(frames[1]) != 32:
        raise EmbeddingConfigurationError("Invalid embedding control message")
    if not hmac.compare_digest(frames[1], hmac.digest(key, direction + frames[0], hashlib.sha256)):
        raise EmbeddingConfigurationError("Embedding service authentication failed")
    try:
        return json.loads(frames[0])
    except (ValueError, UnicodeError):
        raise EmbeddingConfigurationError("Invalid embedding control JSON") from None


def _socket(context: Any, kind: Any) -> Any:
    zmq = _zmq()
    sock = context.socket(kind)
    try:
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.MAXMSGSIZE, ipc.MAX_CONTROL_BYTES)
        sock.setsockopt(zmq.SNDHWM, 8)
        sock.setsockopt(zmq.RCVHWM, 8)
        return sock
    except BaseException:
        sock.close(linger=0)
        raise


async def _drain(task: asyncio.Task[Any]) -> None:
    """Join even repeated handler cancellation; caller inspects the result."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            pass
        except Exception:
            break


class Client:
    def __init__(self, owner: Any):
        self.owner = owner
        self.context = _zmq().asyncio.Context()
        self.sockets: set[Any] = set()

    def close(self) -> None:
        for sock in tuple(self.sockets):
            sock.close(linger=0)
        self.context.term()

    async def request(self, plan: Any) -> np.ndarray:
        parts, arrays, size = plan
        endpoint, key = _read_endpoint(self.owner.options["service_path"])
        zmq = _zmq()
        sock = _socket(self.context, zmq.DEALER)
        request_id = uuid.uuid4().hex
        self.sockets.add(sock)
        started = time.monotonic()
        try:
            sock.setsockopt(zmq.IDENTITY, request_id.encode("ascii"))
            sock.connect(endpoint)
            await sock.send_multipart(_pack(self.owner._reservation(parts, size, request_id), key, b"request"))
            response = _unpack(await sock.recv_multipart(), key, b"response")
            self.owner._check_response(response, request_id)
            if response.get("state") != "reserved" or response.get("nbytes") != size:
                raise EmbeddingConfigurationError("Embedding service did not reserve the input")
            name = response.get("shm_name")
            if size and (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name)):
                raise EmbeddingConfigurationError("Invalid embedding shared memory name")
            if not size and name is not None:
                raise EmbeddingConfigurationError("Empty input must not reference shared memory")
            self.owner.metrics["queue_seconds"] += time.monotonic() - started
            copied = time.monotonic()
            if size:
                with shm.attach(name, size) as memory:
                    offset = 0
                    for value in arrays:
                        target = np.ndarray(value.shape, dtype=value.dtype, buffer=memory, offset=offset)
                        try:
                            np.copyto(target, value)
                        finally:
                            del target
                        offset += value.nbytes
            self.owner.metrics["copy_seconds"] += time.monotonic() - copied
            self.owner.metrics["input_bytes"] += size
            await sock.send_multipart(_pack({"request_id": request_id, "state": "published"}, key, b"request"))
            response = _unpack(await sock.recv_multipart(), key, b"response")
            self.owner._check_response(response, request_id)
            if response.get("state") != "released":
                raise EmbeddingConfigurationError("Embedding service omitted consumption acknowledgement")
            values = response.get("vector")
            if not isinstance(values, list) or any(type(value) not in (int, float) for value in values):
                raise ValueError("Embedding service returned an invalid vector")
            with np.errstate(over="ignore", invalid="ignore"):
                vector: np.ndarray[Any, Any] = np.asarray(values, dtype=np.float32)
            if vector.shape != (self.owner.descriptor.dimensions,) or not np.isfinite(vector).all():
                raise ValueError("Embedding service returned an invalid vector")
            return vector
        finally:
            self.owner.metrics["request_seconds"] += time.monotonic() - started
            self.sockets.discard(sock)
            sock.close(linger=0)


class Server:
    def __init__(self, owner: Any):
        self.owner = owner
        self.key = secrets.token_bytes(32)
        self.context: Any = None
        self.socket: Any = None
        self.endpoint = ""
        self.file_fd: int | None = None
        self.file_identity: tuple[int, int] | None = None
        self.receiver: asyncio.Task[Any] | None = None
        self.pending: dict[bytes, asyncio.Queue[Any]] = {}
        self.buffers: set[str] = set()

    def prepare(self) -> None:
        _private(Path(self.owner.path).parent.stat())
        self.file_fd = os.open(
            self.owner.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600
        )
        info = os.fstat(self.file_fd)
        self.file_identity = (info.st_dev, info.st_ino)
        zmq = _zmq()
        self.context = zmq.asyncio.Context()
        self.socket = _socket(self.context, zmq.ROUTER)
        self.socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        self.endpoint = f"tcp://127.0.0.1:{port}"

    def listen(self) -> None:
        assert self.file_fd is not None
        data = ipc.encode({"protocol": PROTOCOL, "endpoint": self.endpoint, "key": self.key.hex()})
        with os.fdopen(self.file_fd, "wb") as stream:
            self.file_fd = None
            stream.write(data)
        self.receiver = asyncio.create_task(self._receive())

    async def aclose(self) -> None:
        if self.receiver is not None:
            self.receiver.cancel()
            await asyncio.gather(self.receiver, return_exceptions=True)
            self.receiver = None
        tasks = tuple(self.owner._tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            await _drain(task)
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None
        if self.file_fd is not None:
            os.close(self.file_fd)
            self.file_fd = None
        if self.file_identity is not None:
            with contextlib.suppress(FileNotFoundError):
                info = os.lstat(self.owner.path)
                if (info.st_dev, info.st_ino) == self.file_identity:
                    os.unlink(self.owner.path)
            self.file_identity = None

    async def _send(self, identity: bytes, message: Any) -> None:
        # Nonblocking ROUTER send prevents an absent/slow client from retaining
        # buffers or blocking the service dispatcher and other clients.
        await self.socket.send_multipart([identity, *_pack(message, self.key, b"response")], flags=_zmq().DONTWAIT)

    async def _receive(self) -> None:
        while True:
            frames = await self.socket.recv_multipart()
            if len(frames) != 3 or not re.fullmatch(b"[0-9a-f]{32}", frames[0]):
                continue
            identity = frames[0]
            try:
                message = _unpack(frames[1:], self.key, b"request")
                if not isinstance(message, dict) or message.get("request_id") != identity.decode("ascii"):
                    raise EmbeddingConfigurationError("Embedding request identity mismatch")
                if identity in self.pending:
                    self.pending[identity].put_nowait(message)
                    continue
                self.owner._hello(message)
                size = message["nbytes"]
                if len(self.pending) >= self.owner.max_connections:
                    raise MemoryError("Embedding service request limit")
                queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
                self.pending[identity] = queue
                task = asyncio.create_task(self._handle(identity, message, size, queue))
                self.owner._tasks.add(task)
                task.add_done_callback(partial(self._finished, identity=identity, queue=queue))
            except (Exception, asyncio.CancelledError) as error:
                if isinstance(error, asyncio.CancelledError):
                    raise
                with contextlib.suppress(Exception):
                    await self._send(
                        identity,
                        {
                            "request_id": identity.decode("ascii"),
                            "error": "resource" if _is_resource_error(error) else "configuration",
                        },
                    )

    def _finished(self, task: asyncio.Task[Any], identity: bytes, queue: asyncio.Queue[Any]) -> None:
        self.owner._tasks.discard(task)
        # A task cancelled before its first step never enters _handle's finally.
        # Do not remove a newer request that happens to reuse the routing identity.
        if self.pending.get(identity) is queue:
            self.pending.pop(identity)
        if not task.cancelled():
            task.exception()

    async def _handle(self, identity: bytes, message: dict[str, Any], size: int, queue: asyncio.Queue[Any]) -> None:
        owner = self.owner
        reserved = False
        memory: Any = None
        content: Any = None
        snapshot: Any = None
        charge = max(1, 2 * size)  # published bytes plus the private inference snapshot
        response: dict[str, Any] = {"error": "execution"}
        try:
            queued = time.monotonic()
            admission = asyncio.create_task(owner.credits.acquire(charge))
            try:
                await asyncio.wait_for(asyncio.shield(admission), owner.publication_timeout)
            finally:
                if not admission.done():
                    admission.cancel()
                await _drain(admission)
                reserved = not admission.cancelled() and admission.exception() is None
            owner.metrics["queue_seconds"] += time.monotonic() - queued
            if size:
                memory = shm.create(size)
                self.buffers.add(memory.name)
            await self._send(
                identity,
                {
                    "request_id": message["request_id"],
                    "state": "reserved",
                    "shm_name": memory.name if memory is not None else None,
                    "nbytes": size,
                },
            )
            published = await asyncio.wait_for(queue.get(), owner.publication_timeout)
            if published != {"request_id": message["request_id"], "state": "published"}:
                raise EmbeddingConfigurationError("Invalid embedding publication")
            snapshot = bytearray(memory.buf[:size]) if memory is not None else None
            content = ipc.restore(message["parts"], snapshot)
            ipc.describe(content, owner.spec)
            owner.metrics["accepted"] += 1
            started = time.monotonic()
            inference = asyncio.create_task(owner._infer(content))
            await _drain(inference)
            response = inference.result()
            owner.metrics["inference_seconds"] += time.monotonic() - started
            owner.metrics["completed"] += 1
        except (Exception, asyncio.CancelledError) as error:
            response = {
                "error": "resource"
                if _is_resource_error(error)
                else (
                    "configuration"
                    if isinstance(error, (EmbeddingConfigurationError, ValueError, TypeError))
                    else "execution"
                )
            }
        finally:
            content = snapshot = None
            cleaned = time.monotonic()
            try:
                if memory is not None:
                    try:
                        shm.release(memory)
                    finally:
                        self.buffers.discard(memory.name)
            finally:
                if reserved:
                    await owner.credits.release(charge)
                owner.metrics["cleanup_seconds"] += time.monotonic() - cleaned
                self.pending.pop(identity, None)
        with contextlib.suppress(Exception):
            await self._send(identity, {"request_id": message["request_id"], "state": "released", **response})
