# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Test-owned service process; instrument the real SDK without replacing inference."""

import argparse
import asyncio
import json
import os
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
        from tests.fast.test_litert_provider import install_sdk

        sdk = install_sdk(SimpleNamespace(setitem=lambda mapping, name, value: mapping.__setitem__(name, value)))
    else:
        import litert_lm as sdk

    def record(event, **fields):
        with (root / "trace.jsonl").open("a") as output:
            output.write(
                json.dumps({"event": event, "pid": os.getpid(), "thread": threading.get_ident()} | fields) + "\n"
            )

    class Engine(sdk.EmbeddingEngine):
        def __init__(self, *values, **options):
            super().__init__(*values, **options)
            self._record_loaded = True
            self._record_closed = False
            record("loaded")

        def compute_embedding_batch(self, contents, **options):
            record("batch", texts=[part for row in contents for part in row if isinstance(part, str)])
            return super().compute_embedding_batch(contents, **options)

        def close(self):
            super().close()
            if getattr(self, "_record_loaded", False) and not self._record_closed:
                self._record_closed = True
                record("closed")

    sdk.EmbeddingEngine = Engine
    from vane.ai.embedding_service import EmbeddingService

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
