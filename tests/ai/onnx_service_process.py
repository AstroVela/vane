# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Test-owned ONNX service process, optionally using the authored SDK fixture."""

import argparse
import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()
    root = args.root
    if args.fake:
        from tests.onnx_fixture import install_sdk

        install_sdk(SimpleNamespace(setitem=lambda mapping, key, value: mapping.__setitem__(key, value)))
    from vane.ai.embedding_service import EmbeddingService
    from vane.ai.providers._onnx_model import ONNXEmbedder

    def record(event):
        with (root / "trace.jsonl").open("a") as output:
            output.write(json.dumps({"event": event, "thread": threading.get_ident()}) + "\n")

    initialize, close = ONNXEmbedder.__init__, ONNXEmbedder.close

    def tracked_initialize(self, *values, **options):
        initialize(self, *values, **options)
        record("loaded")

    def tracked_close(self):
        was_closed = self.closed
        close(self)
        if not was_closed:
            record("closed")

    ONNXEmbedder.__init__, ONNXEmbedder.close = tracked_initialize, tracked_close
    service = EmbeddingService(**json.loads((root / "config.json").read_text()))
    try:
        await service.start()
        (root / "ready").touch()
        while not (root / "stop").exists():
            await asyncio.sleep(0.01)
    finally:
        await service.aclose()
        (root / "metrics.json").write_text(
            json.dumps(service.metrics | {"bytes": service.credits.bytes, "inflight": service.credits.inflight})
        )


if __name__ == "__main__":
    asyncio.run(main())
