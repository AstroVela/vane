# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded records and mutual authentication over ZeroMQ's raw TCP STREAM.

STREAM delivers TCP chunks, not peer-controlled ZMTP multipart messages. Record
lengths are checked before accumulating bodies; authentication accepts at most
64 bytes. Each request uses a fresh connection and a nonce-bound session key.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Iterator
from typing import Any

from vane.ai import _embedding_ipc as ipc
from vane.ai._embedding_inputs import EmbeddingConfigurationError

MAX_RECORD_BYTES = ipc.MAX_CONTROL_BYTES + 32
AUTH_BYTES = 64


def proof(key: bytes, purpose: bytes, challenge: bytes, nonce: bytes) -> bytes:
    return hmac.digest(key, b"vane-embedding-stream-3/" + purpose + challenge + nonce, hashlib.sha256)


def record(body: bytes) -> bytes:
    if not 0 < len(body) <= MAX_RECORD_BYTES:
        raise EmbeddingConfigurationError("Embedding control record exceeds its byte limit")
    return len(body).to_bytes(4, "big") + body


class Decoder:
    def __init__(self, limit: int = AUTH_BYTES):
        self.limit = limit
        self.buffer = bytearray()
        self.length: int | None = None

    def feed(self, data: bytes) -> Iterator[bytes]:
        position = 0
        while position < len(data):
            needed = (4 if self.length is None else self.length) - len(self.buffer)
            count = min(needed, len(data) - position)
            self.buffer.extend(memoryview(data)[position : position + count])
            position += count
            if count < needed:
                return
            if self.length is None:
                length = int.from_bytes(self.buffer, "big")
                self.buffer.clear()
                if not 0 < length <= self.limit:
                    raise EmbeddingConfigurationError("Embedding control record exceeds its byte limit")
                self.length = length
            else:
                body = bytes(self.buffer)
                self.buffer.clear()
                self.length = None
                yield body


class Connection:
    """One client request's STREAM connection, authenticated before JSON input."""

    def __init__(self, socket: Any, key: bytes, request_id: str):
        self.socket = socket
        self.master_key = key
        self.request_id = request_id
        self.key: bytes | None = None
        self.identity: bytes | None = None
        self.decoder = Decoder()
        self.records: list[bytes] = []

    async def receive(self) -> bytes:
        while not self.records:
            # STREAM always supplies exactly a routing ID and one raw TCP chunk.
            identity, data = await self.socket.recv_multipart()
            if self.identity is None:
                self.identity = identity
                if not data:
                    continue  # Local connection notification.
            if identity != self.identity or not data:
                raise ConnectionError("Embedding service connection closed")
            for body in self.decoder.feed(data):
                # Only one response is outstanding; reject unsolicited records.
                if self.records:
                    raise EmbeddingConfigurationError("Unexpected embedding control record")
                self.records.append(body)
        return self.records.pop()

    async def send(self, body: bytes) -> None:
        assert self.identity is not None
        await self.socket.send_multipart([self.identity, record(body)])

    async def authenticate(self) -> None:
        try:
            challenge = await self.receive()
            if len(challenge) != 32:
                raise EmbeddingConfigurationError("Invalid embedding authentication challenge")
            nonce = self.request_id.encode("ascii")
            await self.send(nonce + proof(self.master_key, b"client", challenge, nonce))
            acknowledgement = await self.receive()
            if not hmac.compare_digest(acknowledgement, proof(self.master_key, b"server", challenge, nonce)):
                raise EmbeddingConfigurationError("Embedding service authentication failed")
            self.key = proof(self.master_key, b"session", challenge, nonce)
            self.decoder.limit = MAX_RECORD_BYTES
        except ConnectionError:
            raise EmbeddingConfigurationError("Embedding service authentication failed") from None


class Peer:
    def __init__(self) -> None:
        self.challenge = secrets.token_bytes(32)
        self.key: bytes | None = None
        self.decoder = Decoder()
        self.deadline: Any = None
        self.messages = 0
        self.request_id: str | None = None
        self.awaiting_receipt = False
        self.receipt_received = False
