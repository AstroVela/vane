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
from vane.ai import _embedding_stream as stream
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._errors import _is_resource_error

PROTOCOL = "vane-embedding-stream-2"
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


def _pack(message: Any, key: bytes, direction: bytes) -> bytes:
    body = ipc.encode(message)
    return body + hmac.digest(key, direction + body, hashlib.sha256)


def _unpack(record: bytes, key: bytes, direction: bytes) -> Any:
    if not 32 < len(record) <= stream.MAX_RECORD_BYTES:
        raise EmbeddingConfigurationError("Invalid embedding control message")
    body, signature = record[:-32], record[-32:]
    if not hmac.compare_digest(signature, hmac.digest(key, direction + body, hashlib.sha256)):
        raise EmbeddingConfigurationError("Embedding service authentication failed")
    try:
        return json.loads(body)
    except (ValueError, UnicodeError):
        raise EmbeddingConfigurationError("Invalid embedding control JSON") from None


def _socket(context: Any, kind: Any) -> Any:
    zmq = _zmq()
    sock = context.socket(kind)
    try:
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDHWM, 8)
        sock.setsockopt(zmq.RCVHWM, 8)
        # Disabling reconnect destroys STREAM's pipe before its EOF notification
        # reaches the client. Keep it alive for notification; Connection.receive
        # fails on EOF and request() closes the socket without replaying input.
        sock.setsockopt(zmq.RECONNECT_IVL, 60_000)
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
        sock = _socket(self.context, zmq.STREAM)
        request_id = uuid.uuid4().hex
        self.sockets.add(sock)
        started = time.monotonic()
        try:
            sock.connect(endpoint)
            connection = stream.Connection(sock, key, request_id)
            await connection.authenticate()
            assert connection.key is not None
            key = connection.key
            await connection.send(_pack(self.owner._reservation(parts, size, request_id), key, b"request"))
            response = _unpack(await connection.receive(), key, b"response")
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
            await connection.send(_pack({"request_id": request_id, "state": "published"}, key, b"request"))
            response = _unpack(await connection.receive(), key, b"response")
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
        self.peers: dict[bytes, stream.Peer] = {}
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
        self.socket = _socket(self.context, zmq.STREAM)
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
        for peer in self.peers.values():
            if peer.deadline is not None:
                peer.deadline.cancel()
        self.peers.clear()
        if self.socket is not None:
            self.socket.close(linger=250)
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

    async def _send_bytes(self, identity: bytes, body: bytes) -> None:
        await self.socket.send_multipart([identity, stream.record(body)], flags=_zmq().DONTWAIT)

    async def _send(self, identity: bytes, message: Any) -> None:
        peer = self.peers.get(identity)
        if peer is None or peer.key is None:
            raise ConnectionError("Embedding service client disconnected")
        await self._send_bytes(identity, _pack(message, peer.key, b"response"))

    def _deadline(self, identity: bytes, peer: stream.Peer) -> None:
        if peer.deadline is not None:
            peer.deadline.cancel()
        peer.deadline = asyncio.get_running_loop().call_later(
            self.owner.publication_timeout, self._expire, identity, peer
        )

    def _expire(self, identity: bytes, peer: stream.Peer) -> None:
        if self.peers.get(identity) is peer:
            self._disconnect(identity)

    def _disconnect(self, identity: bytes) -> None:
        # Locally closing a STREAM pipe need not produce a disconnect event.
        # Release the application slot now; any queued TCP chunks are ignored.
        peer = self.peers.pop(identity, None)
        if peer is not None:
            peer.decoder.buffer.clear()
            if peer.deadline is not None:
                peer.deadline.cancel()
        with contextlib.suppress(Exception):
            sending = self.socket.send_multipart([identity, b""], flags=_zmq().DONTWAIT)
            sending.add_done_callback(lambda future: None if future.cancelled() else future.exception())

    async def _receive(self) -> None:
        while True:
            # STREAM creates this fixed pair itself; TCP peers cannot supply a
            # ZMTP multipart envelope or make libzmq accumulate one atomically.
            identity, data = await self.socket.recv_multipart()
            peer = self.peers.get(identity)
            if not data:
                if peer is not None:
                    self.peers.pop(identity)
                    if peer.deadline is not None:
                        peer.deadline.cancel()
                elif len(self.peers) >= self.owner.max_connections:
                    self._disconnect(identity)
                else:
                    peer = self.peers[identity] = stream.Peer()
                    self._deadline(identity, peer)
                    try:
                        await self._send_bytes(identity, peer.challenge)
                    except Exception:
                        self._disconnect(identity)
                continue
            if peer is None:
                continue
            try:
                for body in peer.decoder.feed(data):
                    if peer.key is None:
                        if len(body) != stream.AUTH_BYTES:
                            raise EmbeddingConfigurationError("Invalid embedding authentication proof")
                        nonce, signature = body[:32], body[32:]
                        if not re.fullmatch(b"[0-9a-f]{32}", nonce) or not hmac.compare_digest(
                            signature, stream.proof(self.key, b"client", peer.challenge, nonce)
                        ):
                            raise EmbeddingConfigurationError("Embedding service authentication failed")
                        await self._send_bytes(identity, stream.proof(self.key, b"server", peer.challenge, nonce))
                        peer.key = stream.proof(self.key, b"session", peer.challenge, nonce)
                        peer.request_id = nonce.decode("ascii")
                        peer.decoder.limit = stream.MAX_RECORD_BYTES
                        self._deadline(identity, peer)
                    else:
                        peer.messages += 1
                        if peer.messages > 2:
                            raise EmbeddingConfigurationError("Unexpected embedding control record")
                        await self._message(identity, peer, _unpack(body, peer.key, b"request"))
            except Exception as error:
                if peer.key is not None and _is_resource_error(error):
                    with contextlib.suppress(Exception):
                        await self._send(identity, {"request_id": peer.request_id, "error": "resource"})
                    peer.messages = 2
                    self._deadline(identity, peer)
                else:
                    self._disconnect(identity)

    async def _message(self, identity: bytes, peer: stream.Peer, message: Any) -> None:
        try:
            if not isinstance(message, dict):
                raise EmbeddingConfigurationError("Invalid embedding control message")
            if message.get("request_id") != peer.request_id:
                raise EmbeddingConfigurationError("Embedding request identity mismatch")
            if peer.messages == 2:
                if identity not in self.pending:
                    raise EmbeddingConfigurationError("Embedding request identity mismatch")
                self.pending[identity].put_nowait(message)
                return
            self.owner._hello(message)
            size = message["nbytes"]
            if len(self.pending) >= self.owner.max_connections:
                raise MemoryError("Embedding service request limit")
            queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
            self.pending[identity] = queue
            if peer.deadline is not None:
                peer.deadline.cancel()
            task = asyncio.create_task(self._handle(identity, message, size, queue))
            self.owner._tasks.add(task)
            task.add_done_callback(partial(self._finished, identity=identity, queue=queue))
        except Exception as error:
            peer.messages = 2
            await self._send(
                identity,
                {"request_id": peer.request_id, "error": "resource" if _is_resource_error(error) else "configuration"},
            )
            self._deadline(identity, peer)

    def _finished(self, task: asyncio.Task[Any], identity: bytes, queue: asyncio.Queue[Any]) -> None:
        self.owner._tasks.discard(task)
        # A task cancelled before its first step never enters _handle's finally.
        # Do not remove a newer request that happens to reuse the routing identity.
        if self.pending.get(identity) is queue:
            self.pending.pop(identity)
        if not task.cancelled():
            task.exception()
        peer = self.peers.get(identity)
        if peer is not None:
            self._deadline(identity, peer)

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
