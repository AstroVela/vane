# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real native/shared ONNX inference against an independent Transformers.js reference."""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from tests.ai.onnx_reference_inputs import DOCUMENTS, QUERIES, fixture
from vane.ai import AudioClip, EmbeddingContent, EmbeddingPart, EmbeddingPartSpec
from vane.ai._onnx_options import ONNX_MODEL
from vane.ai._video_embedding import VideoClip
from vane.ai.provider import load_provider

pytestmark = pytest.mark.external_service


@pytest.mark.parametrize("backend", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("dimensions", [128, 768])
def test_real_native_service_reference_and_public_apis(tmp_path, monkeypatch, backend, dimensions):
    if os.environ.get("VANE_RUN_ONNX_TESTS") != "1":
        pytest.skip("set VANE_RUN_ONNX_TESTS=1 with a real bundle and independent reference")
    if backend == "cuda" and os.environ.get("VANE_RUN_ONNX_GPU_TESTS") != "1":
        pytest.skip("set VANE_RUN_ONNX_GPU_TESTS=1 with a real CUDA execution provider")
    import pyarrow as pa
    import ray

    import vane
    from vane.ai import TextInput, embed

    monkeypatch.setenv("VANE_RUNNER", "local")
    assert not ray.is_initialized()
    monkeypatch.setattr(ray, "init", lambda *args, **kwargs: pytest.fail("ONNX local acceptance initialized Ray"))
    options = {
        "model_path": os.environ["VANE_TEST_ONNX_MODEL_PATH"],
        "model_sha256": os.environ["VANE_TEST_ONNX_MODEL_SHA256"],
        "backend": backend,
        "max_length": 2048,
    }
    reference = json.loads(Path(os.environ["VANE_TEST_ONNX_REFERENCE_PATH"]).read_text())
    assert reference["bundle_sha256"] == options["model_sha256"]
    expected = np.asarray([*reference["documents"], *reference["queries"], *reference["media"]], dtype=np.float64)[
        :, :dimensions
    ]
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    data = fixture()
    image = np.asarray(data["image"], dtype=np.uint8)
    audio = AudioClip(np.asarray(data["audio"], dtype=np.float32).reshape(-1, 1), 16000)
    video = VideoClip(tuple(np.asarray(frame, dtype=np.uint8) for frame in data["frames"]), (0.0, 1.0), (0, 1))

    def parts(*values):
        return EmbeddingContent(tuple(EmbeddingPart(EmbeddingPartSpec(kind), value) for kind, value in values))

    contents = [
        parts(("image", image)),
        parts(("audio", audio)),
        parts(("video", video)),
        parts(("text", "red shape"), ("image", image), ("audio", audio), ("video", video)),
        parts(("image", np.asarray(data["tiny_image"], dtype=np.uint8))),
        parts(
            (
                "video",
                VideoClip(
                    tuple(np.asarray(frame, dtype=np.uint8) for frame in data["tiny_frames"]),
                    (0.0, 1.0),
                    (0, 1),
                ),
            )
        ),
        parts(
            *(
                ("audio", AudioClip(np.asarray(clip, dtype=np.float32).reshape(-1, 1), 16000))
                for clip in data["unequal_audio"]
            )
        ),
    ]
    provider = load_provider("onnx")
    native = provider.get_multimodal_embedder(dimensions=dimensions, options=options).instantiate()
    try:
        native_docs = native.embed_text(DOCUMENTS)
        native_media = native.embed_multimodal(contents)
    finally:
        native.close()
    native_query = provider.get_text_embedder(
        dimensions=dimensions, options=options | {"input_type": "query"}
    ).instantiate()
    try:
        native_queries = native_query.embed_text(QUERIES)
    finally:
        native_query.close()
    native_values = np.asarray([*native_docs, *native_queries, *native_media])
    np.testing.assert_allclose(native_values, expected, atol=1e-5, rtol=1e-4)
    assert (np.asarray(native_queries) @ np.asarray(native_docs).T).argmax(axis=1).tolist() == [0, 1, 2]
    root = tmp_path / "service"
    root.mkdir(mode=0o700)
    config = {
        "path": str(root / "engine.json"),
        "service_id": "real-onnx",
        "family": "onnx",
        "model": ONNX_MODEL,
        "dimensions": dimensions,
        "options": options,
    }
    (root / "config.json").write_text(json.dumps(config))
    clients = {
        "runtime": "service",
        "service_path": config["path"],
        "service_id": config["service_id"],
        "model_sha256": options["model_sha256"],
        "max_length": 2048,
        "timeout": 300,
    }
    log = (root / "log.txt").open("w")
    helper = Path(__file__).with_name("onnx_service_process.py")
    process = subprocess.Popen([sys.executable, str(helper), str(root)], stdout=log, stderr=log)
    try:
        deadline = time.monotonic() + 180
        while not (root / "ready").exists():
            if process.poll() is not None or time.monotonic() > deadline:
                pytest.fail((root / "log.txt").read_text())
            time.sleep(0.03)

        async def scenario():
            doc = provider.get_text_embedder(dimensions=dimensions, options=clients).instantiate()
            query = provider.get_text_embedder(
                dimensions=dimensions, options=clients | {"input_type": "query"}
            ).instantiate()
            media = provider.get_multimodal_embedder(dimensions=dimensions, options=clients).instantiate()
            try:
                docs, queries = await asyncio.gather(doc.embed_text(DOCUMENTS), query.embed_text(QUERIES))
                values = np.asarray([*docs, *queries, *(await media.embed_multimodal(contents))])
                np.testing.assert_allclose(values, native_values, atol=1e-5, rtol=1e-4)
            finally:
                for client in (doc, query, media):
                    await client.aclose()

        asyncio.run(scenario())
        for entry in ("relation", "expression", "sql"):
            with vane.connect() as connection:
                connection.register(
                    "rows",
                    pa.table({"ordinal": [0, 1, 2, 3], "text": [DOCUMENTS[0], None, DOCUMENTS[1], DOCUMENTS[0]]}),
                )
                source = connection.table("rows")
                settings = dict(provider="onnx", dimensions=dimensions, batch_size=2, max_retries=0) | clients
                if entry == "relation":
                    result = embed(source, TextInput(vane.col("text")), execution_backend="subprocess_task", **settings)
                elif entry == "expression":
                    result = source.select("ordinal", embed(TextInput(vane.col("text")), **settings).alias("embedding"))
                else:
                    result = connection.sql(
                        f"SELECT ordinal, ai_embed(text, provider => 'onnx', dimensions => {dimensions}, options => {{runtime: 'service', service_path: ?, service_id: 'real-onnx', model_sha256: ?, max_length: 2048, batch_size: 2, max_retries: 0}}) AS embedding FROM rows",
                        params=[config["path"], options["model_sha256"]],
                    )
                rows = sorted(
                    result.select("ordinal", "embedding").to_arrow_table().to_pylist(), key=lambda row: row["ordinal"]
                )
                assert len(rows) == 4 and rows[1]["embedding"] is None
                np.testing.assert_allclose(
                    [rows[i]["embedding"] for i in (0, 2, 3)], native_values[[0, 1, 0]], atol=1e-5, rtol=1e-4
                )
    finally:
        vane.teardown_runner()
        (root / "stop").touch()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            pytest.fail("Real ONNX service did not drain")
        log.close()
    assert process.returncode == 0, (root / "log.txt").read_text()
    events = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["loaded", "closed"]
    assert len({event["thread"] for event in events}) == 1
    metrics = json.loads((root / "metrics.json").read_text())
    assert metrics["bytes"] == metrics["inflight"] == 0
