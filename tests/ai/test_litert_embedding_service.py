# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real CPU/GPU LiteRT service, native parity and shared public-API clients."""

import asyncio
import io
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from vane.ai import AudioClip, EmbeddingContent, EmbeddingPart, EmbeddingPartSpec
from vane.ai.provider import load_provider

pytestmark = pytest.mark.external_service
MODEL = "litert-community/embeddinggemma-2-740m-litert-lm"
DOCUMENTS = ["Mars is the Red Planet.", "The Pacific Ocean is the largest ocean.", "大熊猫主要吃竹子。"]
QUERIES = ["Which planet is red?", "What is the largest ocean?", "大熊猫吃什么？"]


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("dimensions", [128, 768])
def test_real_service_sdk_parity_shared_roles_public_apis_and_close(tmp_path, monkeypatch, backend, dimensions):
    if os.environ.get("VANE_RUN_LITERT_TESTS") != "1":
        pytest.skip("set VANE_RUN_LITERT_TESTS=1 with the real checkpoint path and SHA-256")
    if backend == "gpu" and os.environ.get("VANE_RUN_LITERT_GPU_TESTS") != "1":
        pytest.skip("set VANE_RUN_LITERT_GPU_TESTS=1 with a real GPU")
    import litert_lm
    import pyarrow as pa
    import ray
    from PIL import Image

    import vane
    from vane.ai import TextInput, embed

    assert not ray.is_initialized()
    monkeypatch.setattr(ray, "init", lambda *a, **k: pytest.fail("LiteRT service test initialized Ray"))
    monkeypatch.setenv("VANE_RUNNER", "local")
    root = tmp_path / "engine"
    root.mkdir(mode=0o700)
    native = {
        "model_path": os.environ["VANE_TEST_LITERT_MODEL_PATH"],
        "model_sha256": os.environ["VANE_TEST_LITERT_MODEL_SHA256"],
        "backend": backend,
        "max_length": 128,
    }
    config = {
        "path": str(root / "engine.json"),
        "service_id": "real-litert",
        "family": "litert",
        "model": MODEL,
        "dimensions": dimensions,
        "options": native,
    }
    (root / "config.json").write_text(json.dumps(config))
    client_options = {
        "runtime": "service",
        "service_path": config["path"],
        "service_id": config["service_id"],
        "model_sha256": native["model_sha256"],
        "max_length": 128,
    }
    image = np.zeros((32, 40, 3), dtype=np.uint8)
    image[4:28, 8:32, 0] = 255
    samples = (0.25 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32).reshape(-1, 1)
    clip = AudioClip(samples, 16000)
    with io.BytesIO() as buffer, Image.fromarray(image) as picture:
        picture.save(buffer, format="PNG")
        png = litert_lm.Content.ImageBytes(buffer.getvalue())
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as writer:
            writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            writer.writeframes(np.rint(samples[:, 0].astype(np.float64) * 32768).astype("<i2").tobytes())
        wav = litert_lm.Content.AudioBytes(buffer.getvalue())
    accelerator = litert_lm.Backend.GPU() if backend == "gpu" else litert_lm.Backend.CPU(thread_count=2)
    with litert_lm.EmbeddingEngine(
        native["model_path"],
        backend=accelerator,
        vision_backend=accelerator,
        audio_backend=accelerator,
        max_input_length=128,
        vision_tokens_per_image=70,
    ) as reference:
        settings = litert_lm.EmbeddingOptions(
            normalize=True,
            insert_special_tokens=True,
            input_overflow_strategy=litert_lm.InputOverflowStrategy.ERROR,
            output_size=dimensions,
            vision_tokens_per_image=70,
        )
        expected = np.asarray(
            [
                item.embedding
                for item in reference.compute_embedding_batch(
                    [["title: none | text: " + text] for text in DOCUMENTS]
                    + [["task: search result | query: " + text] for text in QUERIES]
                    + [[png], [wav], ["title: none | text: red shape", png]],
                    options=settings,
                )
            ],
            dtype=np.float32,
        )
    log = (root / "log.txt").open("w")
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("litert_service_process.py")), str(root)], stdout=log, stderr=log
    )
    try:
        deadline = time.monotonic() + 120
        while not (root / "ready").exists():
            if process.poll() is not None or time.monotonic() > deadline:
                pytest.fail((root / "log.txt").read_text())
            time.sleep(0.03)

        async def requests():
            provider = load_provider("litert")
            document = provider.get_text_embedder(MODEL, dimensions, options=client_options).instantiate()
            query = provider.get_text_embedder(
                MODEL, dimensions, options=client_options | {"input_type": "query"}
            ).instantiate()
            media = provider.get_multimodal_embedder(MODEL, dimensions, options=client_options).instantiate()
            try:
                docs, queries = await asyncio.gather(document.embed_text(DOCUMENTS), query.embed_text(QUERIES))

                def parts(*values):
                    return EmbeddingContent(
                        tuple(EmbeddingPart(EmbeddingPartSpec(kind), value) for kind, value in values)
                    )

                media_values = await media.embed_multimodal(
                    [parts(("image", image)), parts(("audio", clip)), parts(("text", "red shape"), ("image", image))]
                )
                values = np.asarray([*docs, *queries, *media_values], dtype=np.float32)
                assert values.shape == (9, dimensions) and np.isfinite(values).all()
                np.testing.assert_allclose(values, expected, atol=1e-5, rtol=1e-4)
                assert (np.asarray(queries) @ np.asarray(docs).T).argmax(axis=1).tolist() == [0, 1, 2]
            finally:
                for client in (document, query, media):
                    await client.aclose()

        asyncio.run(requests())
        for entry in ("relation", "expression", "sql"):
            with vane.connect() as connection:
                connection.register(
                    "rows",
                    pa.table({"ordinal": [0, 1, 2, 3], "text": [DOCUMENTS[0], None, DOCUMENTS[1], DOCUMENTS[0]]}),
                )
                source = connection.table("rows")
                options = (
                    dict(provider="litert", model=MODEL, dimensions=dimensions, batch_size=2, max_retries=0)
                    | client_options
                )
                if entry == "relation":
                    result = embed(source, TextInput(vane.col("text")), execution_backend="subprocess_task", **options)
                elif entry == "expression":
                    result = source.select("ordinal", embed(TextInput(vane.col("text")), **options).alias("embedding"))
                else:
                    result = connection.sql(
                        "SELECT ordinal, ai_embed(text, provider => 'litert', model => 'litert-community/embeddinggemma-2-740m-litert-lm', "
                        f"dimensions => {dimensions}, "
                        "options => {runtime: 'service', service_path: ?, service_id: 'real-litert', model_sha256: ?, "
                        "max_length: 128, batch_size: 2, max_retries: 0}) AS embedding FROM rows",
                        params=[config["path"], native["model_sha256"]],
                    )
                rows = sorted(
                    result.select("ordinal", "embedding").to_arrow_table().to_pylist(), key=lambda item: item["ordinal"]
                )
                assert rows[1]["embedding"] is None
                np.testing.assert_allclose(
                    [rows[i]["embedding"] for i in (0, 2, 3)], expected[[0, 1, 0]], atol=1e-5, rtol=1e-4
                )
        events = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
        assert sum(event["event"] == "loaded" for event in events) == 1
        assert not any(event["event"] == "closed" for event in events)
        assert {event["pid"] for event in events} == {process.pid}
    finally:
        vane.teardown_runner()
        (root / "stop").touch()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            pytest.fail("LiteRT service did not drain and stop")
        log.close()
    assert process.returncode == 0, (root / "log.txt").read_text()
    events = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
    assert sum(event["event"] == "closed" for event in events) == 1
    assert len({event["thread"] for event in events}) == 1
    metrics = json.loads((root / "metrics.json").read_text())
    assert metrics["bytes"] == metrics["inflight"] == 0
