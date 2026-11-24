# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real LiteRT-LM checkpoint, CPU SDK parity and local process acceptance."""

from __future__ import annotations

import io
import json
import os
import wave

import numpy as np
import pytest

pytestmark = pytest.mark.external_service
MODEL = "litert-community/embeddinggemma-2-740m-litert-lm"
DOCUMENTS = ["Mars is the Red Planet.", "The Pacific Ocean is the largest ocean.", "大熊猫主要吃竹子。"]
QUERIES = ["Which planet is the Red Planet?", "What is the largest ocean?", "大熊猫吃什么？"]


@pytest.fixture(scope="module")
def options():
    if os.environ.get("VANE_RUN_LITERT_TESTS") != "1":
        pytest.skip("set VANE_RUN_LITERT_TESTS=1 with a complete local LiteRT model and SHA-256")
    return {
        "model_path": os.environ["VANE_TEST_LITERT_MODEL_PATH"],
        "model_sha256": os.environ["VANE_TEST_LITERT_MODEL_SHA256"],
        "backend": "cpu",
        "max_length": 128,
    }


def assert_vectors(values, count, dimensions):
    values = np.asarray(values)
    assert values.shape == (count, dimensions) and values.dtype == np.float32
    assert np.isfinite(values).all()
    np.testing.assert_allclose(np.linalg.norm(values.astype(np.float64), axis=1), 1, atol=1e-4)
    return values


@pytest.mark.parametrize("dimensions", [128, 768])
def test_real_text_batch_roles_match_direct_sdk_and_retrieval(options, dimensions):
    import litert_lm

    from vane.ai.provider import load_provider

    backend = litert_lm.Backend.CPU(thread_count=2)
    with litert_lm.EmbeddingEngine(
        options["model_path"],
        backend=backend,
        vision_backend=backend,
        audio_backend=backend,
        max_input_length=128,
        vision_tokens_per_image=70,
    ) as reference:
        sdk_options = litert_lm.EmbeddingOptions(
            normalize=True,
            insert_special_tokens=True,
            input_overflow_strategy=litert_lm.InputOverflowStrategy.ERROR,
            output_size=dimensions,
            vision_tokens_per_image=70,
        )
        vectors = []
        for role, texts, prefix in (
            ("document", DOCUMENTS, "title: none | text: "),
            ("query", QUERIES, "task: search result | query: "),
        ):
            runtime = (
                load_provider("litert")
                .get_text_embedder(MODEL, dimensions, options=options | {"input_type": role})
                .instantiate()
            )
            try:
                batch = assert_vectors(runtime.embed_text(texts), 3, dimensions)
                expected = np.asarray(
                    [
                        item.embedding
                        for item in reference.compute_embedding_batch(
                            [prefix + text for text in texts], options=sdk_options
                        )
                    ],
                    dtype=np.float32,
                )
                np.testing.assert_allclose(batch, expected, atol=1e-5, rtol=1e-4)
                singles = np.vstack([runtime.embed_text([text])[0] for text in texts])
                np.testing.assert_allclose(batch, singles, atol=1e-5, rtol=1e-4)
                vectors.append(batch)
            finally:
                runtime.close()
        assert (vectors[1] @ vectors[0].T).argmax(axis=1).tolist() == [0, 1, 2]


def test_real_image_audio_and_joint_match_direct_sdk(options):
    import litert_lm
    from PIL import Image

    from vane.ai import AudioClip, EmbeddingContent, EmbeddingPart, EmbeddingPartSpec
    from vane.ai.provider import load_provider

    image = np.zeros((32, 40, 3), dtype=np.uint8)
    image[4:28, 8:32, 0] = 255
    samples = (0.25 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32).reshape(-1, 1)
    clip = AudioClip(samples, 16000)
    with io.BytesIO() as png, Image.fromarray(image) as pil_image:
        pil_image.save(png, format="PNG")
        image_input = litert_lm.Content.ImageBytes(png.getvalue())
    with io.BytesIO() as wav:
        with wave.open(wav, "wb") as writer:
            writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            writer.writeframes(np.rint(samples[:, 0].astype(np.float64) * 32768).astype("<i2").tobytes())
        audio_input = litert_lm.Content.AudioBytes(wav.getvalue())
    runtime = load_provider("litert").get_multimodal_embedder(MODEL, options=options).instantiate()
    backend = litert_lm.Backend.CPU(thread_count=2)
    try:
        actual = [
            runtime.embed_image([image])[0],
            runtime.embed_audio([clip])[0],
            runtime.embed_multimodal(
                [
                    EmbeddingContent(
                        (
                            EmbeddingPart(EmbeddingPartSpec("text"), "red shape"),
                            EmbeddingPart(EmbeddingPartSpec("image"), image),
                        )
                    )
                ]
            )[0],
        ]
        with litert_lm.EmbeddingEngine(
            options["model_path"],
            backend=backend,
            vision_backend=backend,
            audio_backend=backend,
            max_input_length=128,
            vision_tokens_per_image=70,
        ) as reference:
            sdk_options = litert_lm.EmbeddingOptions(
                normalize=True,
                insert_special_tokens=True,
                input_overflow_strategy=litert_lm.InputOverflowStrategy.ERROR,
                output_size=768,
                vision_tokens_per_image=70,
            )
            expected = reference.compute_embedding_batch(
                [[image_input], [audio_input], ["title: none | text: red shape", image_input]], options=sdk_options
            )
        expected = np.asarray([row.embedding for row in expected], dtype=np.float32)
        np.testing.assert_allclose(assert_vectors(actual, 3, 768), expected, atol=1e-5, rtol=1e-4)
    finally:
        runtime.close()


@pytest.mark.parametrize("entry", ["relation", "expression", "sql"])
def test_real_local_actor_public_api_nulls_and_teardown(options, monkeypatch, entry):
    import psutil
    import pyarrow as pa
    import ray

    import vane
    from vane.ai import TextInput, embed

    assert not ray.is_initialized()

    def forbidden(*args, **kwargs):
        pytest.fail("LiteRT local execution initialized Ray")

    monkeypatch.setattr(ray, "init", forbidden)
    monkeypatch.setenv("VANE_RUNNER", "local")
    process = psutil.Process()
    previous = {child.pid for child in process.children(recursive=True)}
    try:
        with vane.connect() as connection:
            connection.register(
                "litert_rows",
                pa.table({"ordinal": list(range(4)), "text": [DOCUMENTS[0], None, DOCUMENTS[1], DOCUMENTS[0]]}),
            )
            relation = connection.table("litert_rows")
            if entry == "relation":
                result = embed(
                    relation,
                    TextInput(vane.col("text")),
                    provider="litert",
                    model=MODEL,
                    dimensions=128,
                    batch_size=2,
                    max_retries=0,
                    **options,
                )
            elif entry == "expression":
                result = relation.select(
                    vane.col("ordinal"),
                    embed(
                        TextInput(vane.col("text")),
                        provider="litert",
                        model=MODEL,
                        dimensions=128,
                        batch_size=2,
                        max_retries=0,
                        **options,
                    ).alias("embedding"),
                )
            else:
                result = connection.sql(
                    "SELECT ordinal, ai_embed(text, provider => 'litert', "
                    "model => 'litert-community/embeddinggemma-2-740m-litert-lm', dimensions => 128, "
                    "options => {model_path: ?, model_sha256: ?, backend: 'cpu', max_length: 128, "
                    "batch_size: 2, max_retries: 0}) AS embedding FROM litert_rows",
                    params=[options["model_path"], options["model_sha256"]],
                )
            result = result.select("ordinal", "embedding").to_arrow_table()
            rows = sorted(result.to_pylist(), key=lambda row: row["ordinal"])
            assert rows[1]["embedding"] is None
            assert rows[0]["embedding"] == rows[3]["embedding"]
            assert_vectors(np.asarray([rows[i]["embedding"] for i in (0, 2, 3)], dtype=np.float32), 3, 128)
    finally:
        vane.teardown_runner()
        owned = [
            child
            for child in process.children(recursive=True)
            if child.pid not in previous and "udf_subprocess_worker" in " ".join(child.cmdline())
        ]
        _, alive = psutil.wait_procs(owned, timeout=10)
        assert not alive, "LiteRT actor survived runner teardown"


def test_real_actor_reuses_one_model_and_closes_it(options, monkeypatch, tmp_path):
    import pyarrow as pa

    import vane
    from vane.ai import TextInput, embed
    from vane.ai.providers.litert import LiteRTProvider, LiteRTTextEmbedderDescriptor

    monkeypatch.setenv("VANE_RUNNER", "local")
    trace = str(tmp_path / "actor.jsonl")
    parent_pid = os.getpid()

    def event(name):
        with open(trace, "a") as output:
            output.write(json.dumps({"event": name, "pid": os.getpid()}) + "\n")

    class TracedDescriptor(LiteRTTextEmbedderDescriptor):
        def instantiate(self):
            actual = super().instantiate()
            event("loaded")

            class Runtime:
                def embed_text(self, texts):
                    event("batch")
                    return actual.embed_text(texts)

                async def aclose(self):
                    await actual.aclose()
                    event("closed")

            return Runtime()

    class TracedProvider(LiteRTProvider):
        def get_text_embedder(self, model=None, dimensions=None, *, options=None):
            return TracedDescriptor(model, dimensions, options=options)

    try:
        with vane.connect() as connection:
            source = connection.from_arrow(pa.table({"text": DOCUMENTS * 2}))
            result = embed(
                source,
                TextInput(vane.col("text")),
                provider=TracedProvider(),
                model=MODEL,
                dimensions=128,
                batch_size=2,
                max_retries=0,
                **options,
            ).to_arrow_table()
            assert result.num_rows == 6
    finally:
        vane.teardown_runner()
    from pathlib import Path

    records = [json.loads(line) for line in Path(trace).read_text().splitlines()]
    names = [record["event"] for record in records]
    assert names.count("loaded") == names.count("closed") == 1
    assert names.count("batch") >= 3 and names[0] == "loaded" and names[-1] == "closed"
    assert len({record["pid"] for record in records}) == 1 and records[0]["pid"] != parent_pid
