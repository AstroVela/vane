# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""One persistent native embedding engine, served over PyZMQ and shared memory.

Run ``python -m vane.ai.embedding_service --config service.json``. Only JSON
configuration is accepted; importing this module starts no engine or listener.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import signal
from pathlib import Path
from typing import Any

import numpy as np

from vane.ai import _embedding_ipc as ipc
from vane.ai._embedding_content import EmbeddingContent
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._errors import _is_resource_error
from vane.ai.provider import _ProviderResultError


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
    """Service lifecycle. The runtime is created once, before the endpoint is published.

    Cancellation after publication drains the accepted request. Credits and mappings
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

        if family not in {"sglang", "vllm", "litert"}:
            raise EmbeddingConfigurationError("Embedding service family must be sglang, vllm or litert")
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
            model,
            dimensions,
            {"runtime": "service", "service_path": path, "service_id": service_id},
            family,
            family,
        )
        if family == "litert":
            from vane.ai.providers.litert import LiteRTMultimodalEmbedderDescriptor

            self.descriptor: Any = LiteRTMultimodalEmbedderDescriptor(model, dimensions, options=dict(options or {}))
        else:
            self.descriptor = NativeMultimodalEmbedderDescriptor(model, dimensions, dict(options or {}), family, family)
        self.spec = self.descriptor.get_input_spec()
        self.path, self.identity, self.family = path, service_id, family
        self._zmq_server: Any = None
        self.credits = _Credits(max_bytes, max_inflight)
        self.max_connections, self.publication_timeout = max_connections, publication_timeout
        self.runtime: Any = None
        self._closing: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self.metrics = {
            "accepted": 0,
            "completed": 0,
            "queue_seconds": 0.0,
            "inference_seconds": 0.0,
            "cleanup_seconds": 0.0,
        }

    async def start(self) -> None:
        from vane.ai._embedding_zmq import Server

        if self.runtime is not None or self._zmq_server is not None:
            raise RuntimeError("Embedding service is already started")
        if self._closing is not None:
            if not self._closing.done():
                raise RuntimeError("Embedding service is closing")
            self._closing.result()
            self._closing = None
        self._zmq_server = Server(self)
        try:
            self._zmq_server.prepare()
            if self.family == "litert":
                from vane.ai.providers._litert_service import LiteRTServiceRuntime

                self.runtime = LiteRTServiceRuntime(self.descriptor)
                await self.runtime.start()
            else:
                self.runtime = self.descriptor.instantiate()
                self.runtime._ensure_engine()
            if self._closing is not None:
                raise RuntimeError("Embedding service closed during startup")
            self._zmq_server.listen()
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        while True:
            try:
                await asyncio.shield(self._closing)
                return
            except asyncio.CancelledError:
                if self._closing.cancelled():
                    raise

    async def _close(self) -> None:
        try:
            if self._zmq_server is not None:
                await self._zmq_server.aclose()
        finally:
            self._zmq_server = None
            if self.runtime is not None:
                try:
                    await self.runtime.aclose()
                finally:
                    self.runtime = None

    def _hello(self, message: Any) -> int:
        fields = {"version", "service_id", "family", "model", "dimensions", "request_id", "parts", "nbytes"}
        if self.family == "litert":
            fields.add("encoding")
        if not isinstance(message, dict) or set(message) != fields:
            raise EmbeddingConfigurationError("Invalid embedding reservation")
        expected = {
            "version": ipc.VERSION,
            "service_id": self.identity,
            "family": self.family,
            "model": self.descriptor.get_model(),
            "dimensions": self.descriptor.get_dimensions(),
        }
        if any(type(message[key]) is not type(value) or message[key] != value for key, value in expected.items()):
            raise EmbeddingConfigurationError("Embedding service/model identity mismatch")
        identity = message["request_id"]
        if not isinstance(identity, str) or len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity):
            raise EmbeddingConfigurationError("Invalid embedding request identity")
        charge = ipc.validate_layout(message["parts"], message["nbytes"], self.spec)
        if self.family == "litert":
            from vane.ai.providers._litert_service import encoding_identity

            encoding = message["encoding"]
            expected_encoding = encoding_identity(self.descriptor.get_options())
            if (
                not isinstance(encoding, dict)
                or set(encoding) != set(expected_encoding) | {"input_type"}
                or any(
                    type(encoding[key]) is not type(value) or encoding[key] != value
                    for key, value in expected_encoding.items()
                )
                or encoding["input_type"] not in ("document", "query")
                or encoding["input_type"] == "query"
                and (len(message["parts"]) != 1 or message["parts"][0]["kind"] != "text")
            ):
                raise EmbeddingConfigurationError("LiteRT service encoding or retrieval-role mismatch")
        return charge

    async def _infer(self, content: EmbeddingContent, *, input_type: str = "document") -> dict[str, Any]:
        try:
            value = (
                await self.runtime._embed_content(content, input_type=input_type)
                if self.family == "litert"
                else await self.runtime._embed_content(content)
            )
            vector = np.asarray(value, dtype=np.float32)
            if vector.shape != (self.descriptor.get_dimensions(),) or not np.isfinite(vector).all():
                raise ValueError("Invalid embedding result")
            return {"vector": vector.tolist()}
        except EmbeddingConfigurationError:
            return {"error": "configuration"}
        except _ProviderResultError:
            return {"error": "result"}
        except Exception as error:
            return {"error": "resource" if _is_resource_error(error) else "execution"}


async def _run(config: dict[str, Any]) -> None:
    service = EmbeddingService(**config)
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stopped.set)
        except NotImplementedError:  # Windows event loops have no Unix signal handlers.
            signal.signal(signum, lambda *_: loop.call_soon_threadsafe(stopped.set))
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
