# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit Metal qualification against an independent fixed SDK reference."""

from __future__ import annotations

import gc
import json
import os
import platform
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.external_service, pytest.mark.metal]
QUERIES = [
    "A red circle on a white background.",
    "A blue square on a white background.",
    "A green triangle on a white background.",
]


@pytest.fixture(scope="module", autouse=True)
def prepared_metal():
    if os.environ.get("VANE_RUN_MLX_TESTS") != "1":
        pytest.skip("set VANE_RUN_MLX_TESTS=1 with the complete fixed model cache and Metal")
    assert platform.system() == "Darwin" and platform.machine() == "arm64"
    import mlx.core as mx

    assert mx.metal.is_available(), "real MLX image tests require Metal"
    import ray

    assert not ray.is_initialized()

    def forbid(*args, **kwargs):
        raise AssertionError("Metal image tests must not initialize Ray")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ray, "init", forbid)
        yield
        assert not ray.is_initialized()


@pytest.fixture(scope="module")
def images():
    from PIL import Image, ImageDraw

    rows = []
    # Different aspect ratios and resolutions, plus an exact duplicate.
    for kind, size in [(0, (512, 512)), (1, (640, 320)), (2, (256, 640)), (0, (512, 512)), (2, (640, 256))]:
        with Image.new("RGB", size, "white") as im:
            draw = ImageDraw.Draw(im)
            width, height = size
            box = (width * 3 // 16, height * 3 // 16, width * 13 // 16, height * 13 // 16)
            if kind == 0:
                draw.ellipse(box, fill=(230, 20, 20))
            elif kind == 1:
                draw.rectangle(box, fill=(20, 40, 230))
            else:
                draw.polygon(
                    [(width // 2, height // 8), (width // 8, height * 27 // 32), (width * 7 // 8, height * 27 // 32)],
                    fill=(20, 170, 40),
                )
            rows.append(np.array(im, copy=True))
    return rows


def vectors(values, count):
    array = np.asarray(values)
    assert array.dtype == np.float32 and array.shape == (count, 768) and np.isfinite(array).all()
    np.testing.assert_allclose(np.linalg.norm(array.astype(np.float64), axis=1), 1, atol=1e-4, rtol=0)
    return array


def cosine(left, right):
    left, right = left.astype(np.float64), right.astype(np.float64)
    return np.sum(left * right, axis=1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1))


@pytest.fixture(scope="module")
def reference(images):
    # No Vane provider constants, preprocessing or output helpers are used here.
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_vlm.embedding_loader import load_embedding_model
    from mlx_vlm.models.embedding_gemma2 import EmbeddingGemma2Processor
    from PIL import Image
    from transformers import AutoTokenizer

    snapshot = Path(
        snapshot_download(
            "mlx-community/embeddinggemma-2-bf16",
            revision="1a4ffddb7905d3f63486748deabe091a01fb6201",
            local_files_only=True,
        )
    )
    with mx.stream(mx.gpu):
        model = load_embedding_model(snapshot, lazy=False, strict=True)
        model.eval()
        processor = EmbeddingGemma2Processor.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False
        )
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)

    def infer(model, encoded):
        with mx.stream(mx.gpu):
            output = model(**{key: mx.array(value) for key, value in encoded.items()})
            evaluated = output.text_embeds.astype(mx.float32)
            mx.eval(evaluated)
            values = np.array(evaluated, dtype=np.float32, copy=True)
        norms = np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True)
        assert np.isfinite(values).all() and np.all(np.abs(norms - 1) <= 1e-2)
        return (values / norms).astype(np.float32)

    try:
        with ExitStack() as stack:
            pil = []
            for value in images:
                im = Image.fromarray(value)
                stack.callback(im.close)
                pil.append([im])
            encoded = processor(images=pil, return_tensors="np", padding=True, truncation=False)
            assert set(encoded) == {"input_ids", "attention_mask", "pixel_values", "image_position_ids"}
            image_vectors = infer(model, encoded)
        text = tokenizer(
            ["task: search result | query: " + q for q in QUERIES], return_tensors="np", padding=True, truncation=False
        )
        query_vectors = infer(model, {"input_ids": text["input_ids"], "attention_mask": text["attention_mask"]})
        # Confirm actual loaded parameter names, including the trained vision path.
        from mlx.utils import tree_flatten

        weights = dict(tree_flatten(model.parameters()))
        assert weights["embed_vision.embedding_projection.weight"].dtype == mx.bfloat16
        assert any(key.startswith("vision_tower.") for key in weights)
        del weights
        return {"images": image_vectors, "queries": query_vectors}
    finally:
        mx.synchronize(mx.gpu)
        del model, processor, tokenizer
        gc.collect()
        mx.clear_cache()


def test_real_image_parity_batch_single_reorder_and_array_ownership(images, reference, record_property):
    import mlx.core as mx
    import psutil

    from vane.ai.provider import load_provider

    start = time.perf_counter()
    embedder = load_provider("mlx").get_image_embedder().instantiate()
    load_seconds = time.perf_counter() - start
    try:
        start = time.perf_counter()
        batch = vectors(embedder.embed_image(images), 5)
        encode_seconds = time.perf_counter() - start
        single = vectors([embedder.embed_image([image])[0] for image in images], 5)
        order = [4, 1, 3, 0, 2]
        reordered = vectors(embedder.embed_image([images[i] for i in order]), 5)
        readonly = images[1].copy()
        readonly.setflags(write=False)
        noncontiguous = images[0][:, ::-1].copy()[:, ::-1]
        before = noncontiguous.copy()
        unusual = vectors(embedder.embed_image([noncontiguous, readonly]), 2)
        assert not noncontiguous.flags.c_contiguous and not readonly.flags.writeable
        np.testing.assert_array_equal(noncontiguous, before)
        assert float(cosine(batch, reference["images"]).min()) >= 0.999
        assert float(cosine(batch, single).min()) >= 0.999
        assert float(cosine(reordered, batch[order]).min()) >= 0.999
        assert float(cosine(unusual, batch[:2]).min()) >= 0.999
        assert float(cosine(batch[[0]], batch[[3]])[0]) >= 0.999
        metrics = {
            "reference_min_cosine": float(cosine(batch, reference["images"]).min()),
            "batch_single_min_cosine": float(cosine(batch, single).min()),
            "max_absolute_difference": float(np.abs(batch - reference["images"]).max()),
            "load_seconds": load_seconds,
            "encode_seconds": encode_seconds,
            "peak_metal_bytes": mx.get_peak_memory(),
            "active_metal_bytes": mx.get_active_memory(),
            "rss_bytes": psutil.Process().memory_info().rss,
            "input_shapes": [list(im.shape) for im in images],
        }
        for key, value in metrics.items():
            record_property(key, json.dumps(value))
        print("image metrics", json.dumps(metrics))
    finally:
        embedder.close()


def test_real_paired_text_image_retrieval(images, reference, record_property):
    from vane.ai.provider import load_provider

    embedder = load_provider("mlx").get_text_embedder(options={"input_type": "query"}).instantiate()
    try:
        queries = vectors(embedder.embed_text(QUERIES), 3)
    finally:
        embedder.close()
    assert float(cosine(queries, reference["queries"]).min()) >= 0.999
    scores = queries @ reference["images"][:3].T
    assert scores.argmax(axis=1).tolist() == [0, 1, 2], scores
    same = reference["images"][:3] @ reference["images"][:3].T
    assert same.argmax(axis=1).tolist() == [0, 1, 2], same
    record_property("text_image_scores", json.dumps(scores.tolist()))
    print("text/image scores", scores.tolist())


@pytest.fixture
def local_actor(monkeypatch):
    import psutil
    import ray

    import vane

    assert not ray.is_initialized()

    def forbid(*args, **kwargs):
        raise AssertionError("image acceptance must not initialize Ray")

    monkeypatch.setattr(ray, "init", forbid)
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    process = psutil.Process()
    previous = {c.pid for c in process.children(recursive=True)}
    try:
        yield
        assert not ray.is_initialized()
    finally:
        vane.teardown_runner()
        children = [
            c
            for c in process.children(recursive=True)
            if c.pid not in previous and "udf_subprocess_worker" in " ".join(c.cmdline())
        ]
        _, alive = psutil.wait_procs(children, timeout=10)
        assert not alive, "owned image subprocesses survived teardown"


def public_vectors(images, provider, entry):
    import pyarrow as pa

    import vane
    from vane.ai import ImageInput, embed
    from vane.execution.udf_file_contract import FileUDFContract

    values = [images[0], None, images[1], images[2], images[3], images[4], images[1]]
    arrays = FileUDFContract("fixture", (), (vane.image_type("RGB"),)).scalar_outputs_to_array(values)
    with vane.connect() as conn:
        conn.register(
            "images", pa.table({"ordinal": list(range(7)), "image": arrays, "selected": [True] * 6 + [False]})
        )
        relation = conn.table("images")
        options = dict(
            provider=provider, batch_size=2, actor_number=1, execution_backend="subprocess_actor", max_retries=0
        )
        if entry == "relation":
            result = embed(relation, ImageInput(vane.col("image")), where=vane.col("selected"), **options)
        elif entry == "expression":
            options.pop("execution_backend")
            result = relation.select(
                vane.col("ordinal"),
                embed(ImageInput(vane.col("image")), where=vane.col("selected"), **options).alias("embedding"),
            )
        else:
            result = conn.sql(
                "SELECT ordinal, ai_embed(image, input_spec => {kind: 'image'}, provider => 'mlx', "
                "options => {batch_size: 2, actor_number: 1, max_retries: 0}, "
                '"where" => selected) AS embedding FROM images'
            )
        assert str(result.types[result.columns.index("embedding")]) == "FLOAT[768]"
        table = result.select("ordinal", "embedding").to_arrow_table()
        assert table.schema.field("embedding").type == pa.list_(pa.float32(), 768)
        rows = sorted(table.to_pylist(), key=lambda row: row["ordinal"])
        assert [row["ordinal"] for row in rows] == list(range(7))
        assert [row["embedding"] is None for row in rows] == [False, True, False, False, False, False, True]
        return vectors(
            np.array([row["embedding"] for row in rows if row["embedding"] is not None], dtype=np.float32), 5
        )


@pytest.mark.parametrize("entry", ["relation", "expression", "sql"])
def test_real_public_image_apis_preserve_null_where_and_vectors(images, reference, local_actor, entry):
    actual = public_vectors(images, "mlx", entry)
    assert float(cosine(actual, reference["images"]).min()) >= 0.999


def test_real_image_actor_load_processor_once_and_exit(images, reference, local_actor, tmp_path, record_property):
    import psutil

    from vane.ai.providers.mlx import MLXImageEmbedderDescriptor, MLXProvider

    trace = str(tmp_path / "image-actor.jsonl")
    parent = os.getpid()

    def event(kind, **values):
        with Path(trace).open("a") as log:
            log.write(json.dumps({"event": kind, "pid": os.getpid(), **values}) + "\n")

    class TracedDescriptor(MLXImageEmbedderDescriptor):
        def instantiate(self):
            from mlx_vlm.models.embedding_gemma2 import EmbeddingGemma2Processor

            original = EmbeddingGemma2Processor.from_pretrained

            def processor(*args, **kwargs):
                value = original(*args, **kwargs)
                event("processor_loaded")
                return value

            EmbeddingGemma2Processor.from_pretrained = processor
            try:
                actual = super().instantiate()
            finally:
                EmbeddingGemma2Processor.from_pretrained = original
            event("loaded")

            class Traced:
                def embed_image(self, rows):
                    event("batch", rows=len(rows))
                    return actual.embed_image(rows)

                async def aclose(self):
                    await actual.aclose()
                    event("closed")

            return Traced()

    class TracedProvider(MLXProvider):
        def get_image_embedder(self, model=None, dimensions=None, *, options=None):
            return TracedDescriptor(model, dimensions, options=options)

    actual = public_vectors(images, TracedProvider(), "relation")
    assert float(cosine(actual, reference["images"]).min()) >= 0.999
    records = [json.loads(line) for line in Path(trace).read_text().splitlines()]
    kinds = [row["event"] for row in records]
    assert kinds.count("loaded") == kinds.count("processor_loaded") == kinds.count("closed") == 1
    assert kinds.count("batch") >= 3 and sum(row.get("rows", 0) for row in records) == 5
    assert len({row["pid"] for row in records}) == 1 and records[0]["pid"] != parent
    assert not psutil.pid_exists(records[0]["pid"])
    record_property("actor_trace", json.dumps(records))
    print("actor trace", records)
