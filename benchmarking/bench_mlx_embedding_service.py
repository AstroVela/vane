# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the installed MLX ZMQ service with a local copy of EmbeddingGemma 2.

Run python -I benchmarking/bench_mlx_embedding_service.py --output samples.json.
Model startup, warm-up and vector comparisons are outside the timers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec
from vane.ai.provider import load_provider
from vane.ai.providers._embedding_service import ServiceEmbedder
from vane.ai.providers.mlx import MODEL_ID

SIZES = (1, 4, 8, 16)
# Balance each size across all four positions to limit warm-cache and thermal bias.
ORDERS = ((1, 4, 16, 8), (4, 8, 1, 16), (8, 16, 4, 1), (16, 1, 8, 4))


def inputs() -> list[EmbeddingContent]:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(64):
        parts = [EmbeddingPart(EmbeddingPartSpec("text"), f"Scene {i}: a small blue cup on a wooden table.")]
        if i % 2:
            image = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
            parts.append(EmbeddingPart(EmbeddingPartSpec("image"), image))
        rows.append(EmbeddingContent(tuple(parts)))
    return rows


class TimedClient(ServiceEmbedder):
    """Observe the production request boundary, including the ZMQ handshake."""

    def __init__(self, path: Path, size: int):
        descriptor = load_provider("mlx").get_multimodal_embedder(
            options={
                "runtime": "service",
                "service_path": str(path),
                "service_id": "bench",
                "service_batch_rows": size,
                "max_concurrency_per_actor": 1,
                "timeout": 180,
            }
        )
        super().__init__(descriptor)
        self.rpc_ms: list[float] = []

    async def _request(self, plan: Any) -> Any:
        start = time.perf_counter()
        result = await super()._request(plan)
        self.rpc_ms.append(1000 * (time.perf_counter() - start))
        return result


async def benchmark(path: Path, repeats: int) -> list[dict[str, Any]]:
    rows = inputs()
    samples = []
    clients = {size: TimedClient(path, size) for size in SIZES}
    baseline = None
    try:
        for size, client in clients.items():
            await client.embed_multimodal(rows[:16])
            if len(client.rpc_ms) != 16 // size:
                raise RuntimeError(f"Unexpected RPC count during warm-up at size {size}")
            print(f"Warmed size {size}", file=sys.stderr, flush=True)
        for repeat in range(repeats):
            for size in ORDERS[repeat % len(ORDERS)]:
                client = clients[size]
                client.rpc_ms.clear()
                start = time.perf_counter()
                result = await client.embed_multimodal(rows)
                elapsed = time.perf_counter() - start
                vectors = np.asarray(result, dtype=np.float32)
                if vectors.shape != (len(rows), 768) or not np.isfinite(vectors).all():
                    raise RuntimeError(f"Invalid result at size {size}")
                if len(client.rpc_ms) != len(rows) // size:
                    raise RuntimeError(f"Unexpected RPC count at size {size}, possible single-row fallback")
                if baseline is None:
                    baseline = vectors
                cosines = (baseline * vectors).sum(axis=1) / (
                    np.linalg.norm(baseline, axis=1) * np.linalg.norm(vectors, axis=1)
                )
                if not np.all(cosines > 0.999):
                    raise RuntimeError(f"Vector mismatch at size {size}: {cosines.min()}")
                samples.append(
                    dict(
                        repeat=repeat + 1,
                        size=size,
                        seconds=elapsed,
                        rpc_ms=client.rpc_ms.copy(),
                        min_cosine=float(cosines.min()),
                    )
                )
                print(f"Round {repeat + 1}, size {size}: {len(rows) / elapsed:.2f} rows/s", file=sys.stderr, flush=True)
    finally:
        for client in clients.values():
            await client.aclose()
    return samples


def measure(directory: Path, repeats: int) -> list[dict[str, Any]]:
    path = directory / "engine.json"
    config = directory / "service.json"
    config.write_text(
        json.dumps(
            dict(
                path=str(path),
                service_id="bench",
                family="mlx",
                model=MODEL_ID,
                dimensions=768,
                options={"local_files_only": True},
            )
        )
    )
    server = subprocess.Popen(
        [sys.executable, "-I", "-m", "vane.ai.embedding_service", "--config", str(config)], cwd=directory
    )
    try:
        deadline = time.monotonic() + 180
        while not (path.exists() and path.stat().st_size):
            if server.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("MLX service failed to start")
            time.sleep(0.1)
        return asyncio.run(benchmark(path, repeats))
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    with tempfile.TemporaryDirectory() as directory:
        samples = measure(Path(directory), args.repeats)
    summary = {}
    for size in SIZES:
        measured = [s for s in samples if s["size"] == size]
        times = [s["seconds"] for s in measured]
        rpc_ms = [ms for s in measured for ms in s["rpc_ms"]]
        summary[size] = dict(
            rows_per_second=64 * len(times) / sum(times),
            workload_p50_s=float(np.median(times)),
            rpc_count=len(rpc_ms),
            rpc_p50_ms=float(np.median(rpc_ms)),
            rpc_p95_ms=float(np.percentile(rpc_ms, 95)),
            min_cosine=min(s["min_cosine"] for s in measured),
        )
    report = dict(
        model=MODEL_ID,
        versions={p: version(p) for p in ("mlx", "mlx-vlm", "transformers", "pyzmq")},
        rows=64,
        concurrency=1,
        summary=summary,
        samples=samples,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
