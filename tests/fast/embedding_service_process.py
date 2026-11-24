# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU SDK fixture in a separate service process; never loads model weights."""

import asyncio
import errno
import json
import signal
import sys
from pathlib import Path

from tests.fast.test_native_multimodal import install_sdk
from vane.ai import DiagnosticError
from vane.ai.embedding_service import EmbeddingService


async def main():
    root, family = Path(sys.argv[1]), sys.argv[2]
    state = install_sdk(lambda mapping, name, value: mapping.__setitem__(name, value))
    active = 0

    async def embedding(_):
        nonlocal active
        active += 1
        try:
            while (root / "hold").exists():
                await asyncio.sleep(0.01)
            if (root / "memory_error").exists():
                raise MemoryError("private model allocation diagnostic")
            if (root / "wrapped_enospc").exists():
                raise DiagnosticError("embedding", OSError(errno.ENOSPC, "private model allocation path"))
            if (root / "execution_error").exists():
                raise RuntimeError("private model execution diagnostic")
            return [3, 4]
        finally:
            active -= 1

    state.on_embedding = embedding
    config = json.loads((root / "config.json").read_text())
    service = EmbeddingService(
        path=str(root / "engine.json"),
        service_id="fixture-v1",
        family=family,
        model="fixture",
        dimensions=2,
        options={"gpus_per_actor": 0},
        **config,
    )
    stopped = asyncio.Event()
    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stopped.set)
    except NotImplementedError:
        pass  # The test-owned stop file also supports Windows.
    await service.start()
    try:
        (root / "ready").touch()
        while not stopped.is_set() and not (root / "stop").exists():
            data = {
                "bytes": service.credits.bytes,
                "inflight": service.credits.inflight,
                "active": active,
                "engines": len(state.engines),
                "calls": len(state.calls),
                "connections": len(service._zmq_server.pending),
                "buffers": list(service._zmq_server.buffers),
            }
            temporary = root / "state.tmp"
            temporary.write_text(json.dumps(data))
            temporary.replace(root / "state.json")
            await asyncio.sleep(0.01)
    finally:
        await service.aclose()
        assert service.credits.bytes == service.credits.inflight == 0
        assert len(state.engines) == 1 and state.engines[0].closed == 1


if __name__ == "__main__":
    asyncio.run(main())
