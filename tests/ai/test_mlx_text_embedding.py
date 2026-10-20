# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit Mac acceptance with the fixed real checkpoint and subprocess actors."""

from __future__ import annotations

import json
import os
import platform
import time

import numpy as np
import pytest

pytestmark = [pytest.mark.external_service, pytest.mark.metal]

DOCUMENTS = [
    "Mars is known as the Red Planet because of iron oxide on its surface.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "大熊猫主要以竹子为食，生活在中国的山地森林中。",
]
QUERIES = ["Which planet is called the Red Planet?", "What is the largest ocean on Earth?", "大熊猫主要吃什么？"]
TEXTS = [DOCUMENTS[0], "", DOCUMENTS[2], "A short sentence.", DOCUMENTS[0]]


@pytest.fixture(scope="module", autouse=True)
def prepared_metal():
    if os.environ.get("VANE_RUN_MLX_TESTS") != "1":
        pytest.skip("set VANE_RUN_MLX_TESTS=1 after preparing Apple Metal and the fixed model cache")
    assert platform.system() == "Darwin" and platform.machine() == "arm64"
    import mlx.core as mx

    assert mx.metal.is_available(), "real MLX tests require Metal"


def assert_vectors(vectors, count):
    array = np.asarray(vectors)
    assert array.dtype == np.float32 and array.shape == (count, 768)
    assert np.isfinite(array).all()
    np.testing.assert_allclose(np.linalg.norm(array.astype(np.float64), axis=1), 1, atol=1e-4, rtol=0)
    return array


def min_cosine(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.min(np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))))


@pytest.fixture(scope="module")
def reference():
    # Intentionally independent of provider constants and preprocessing helpers.
    from pathlib import Path

    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_vlm.embedding_loader import load_embedding_model
    from transformers import AutoTokenizer

    snapshot = Path(
        snapshot_download(
            repo_id="mlx-community/embeddinggemma-2-bf16",
            revision="1a4ffddb7905d3f63486748deabe091a01fb6201",
            local_files_only=True,
        )
    )
    with mx.stream(mx.gpu):
        model = load_embedding_model(snapshot, lazy=False, strict=True)
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    references = {}
    try:
        for role, prefix in (("document", "title: none | text: "), ("query", "task: search result | query: ")):
            encoded = tokenizer([prefix + text for text in TEXTS], padding=True, truncation=False, return_tensors="np")
            with mx.stream(mx.gpu):
                output = model(
                    input_ids=mx.array(encoded["input_ids"]), attention_mask=mx.array(encoded["attention_mask"])
                )
                values = output.text_embeds.astype(mx.float32)
                mx.eval(values)
                vectors = np.array(values, dtype=np.float32, copy=True)
                norms = np.linalg.norm(vectors.astype(np.float64), axis=1, keepdims=True)
                assert np.isfinite(vectors).all() and np.all(np.abs(norms - 1) <= 1e-2)
                references[role] = (vectors / norms).astype(np.float32)
                del output, values
    finally:
        mx.synchronize(mx.gpu)
        del model, tokenizer
        mx.clear_cache()
    return references


@pytest.mark.parametrize("role", ["document", "query"])
def test_real_mlx_output_matches_upstream_reference(reference, role, record_property):
    from vane.ai.provider import load_provider

    started = time.perf_counter()
    embedder = load_provider("mlx").get_text_embedder(options={"input_type": role}).instantiate()
    load_seconds = time.perf_counter() - started
    try:
        started = time.perf_counter()
        vectors = assert_vectors(embedder.embed_text(TEXTS), len(TEXTS))
        cosine = min_cosine(vectors, reference[role])
        difference = float(np.abs(vectors - reference[role]).max())
        record_property("min_cosine", cosine)
        record_property("max_absolute_difference", difference)
        record_property("load_seconds", load_seconds)
        record_property("encode_seconds", time.perf_counter() - started)
        print(role, "min_cosine", cosine, "max_absolute_difference", difference, "load_seconds", load_seconds)
        assert cosine >= 0.999
    finally:
        embedder.close()


def test_real_mlx_mixed_length_batch_matches_individual_calls(record_property):
    from vane.ai.provider import load_provider

    embedder = load_provider("mlx").get_text_embedder().instantiate()
    try:
        batch = assert_vectors(embedder.embed_text(TEXTS), len(TEXTS))
        single = assert_vectors([embedder.embed_text([text])[0] for text in TEXTS], len(TEXTS))
        cosine = min_cosine(batch, single)
        record_property("min_batch_single_cosine", cosine)
        print("min_batch_single_cosine", cosine)
        assert cosine >= 0.999
        assert min_cosine(batch[[0]], batch[[4]]) >= 0.999
    finally:
        embedder.close()


def test_real_mlx_query_ranks_expected_documents():
    from vane.ai.provider import load_provider

    vectors = []
    for role, texts in (("document", DOCUMENTS), ("query", QUERIES)):
        embedder = load_provider("mlx").get_text_embedder(options={"input_type": role}).instantiate()
        try:
            vectors.append(assert_vectors(embedder.embed_text(texts), len(texts)))
        finally:
            embedder.close()
    scores = vectors[1] @ vectors[0].T
    assert scores.argmax(axis=1).tolist() == [0, 1, 2], scores


@pytest.fixture
def local_actor(monkeypatch):
    import psutil
    import ray

    import vane

    assert not ray.is_initialized()

    def forbid_ray(*args, **kwargs):
        raise AssertionError("MLX local actor must not initialize Ray")

    monkeypatch.setattr(ray, "init", forbid_ray)
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    process = psutil.Process()
    previous = {child.pid for child in process.children(recursive=True)}
    try:
        yield
        assert not ray.is_initialized()
    finally:
        vane.teardown_runner()
        owned = [
            child
            for child in process.children(recursive=True)
            if child.pid not in previous and "udf_subprocess_worker" in " ".join(child.cmdline())
        ]
        _, alive = psutil.wait_procs(owned, timeout=10)
        assert not alive, "MLX actor processes survived runner teardown"


def public_vectors(provider):
    import pyarrow as pa

    import vane
    from vane.ai import TextInput, embed

    with vane.connect() as connection:
        relation = connection.sql(
            "SELECT range AS ordinal, list_extract(?::VARCHAR[], range + 1) AS text FROM range(?)",
            params=[TEXTS, len(TEXTS)],
        )
        result = embed(
            relation,
            TextInput(vane.col("text")),
            provider=provider,
            dimensions=768,
            batch_size=2,
            actor_number=1,
            execution_backend="subprocess_actor",
            max_retries=0,
            on_error="raise",
        )
        table = result.select("ordinal", "embedding").to_arrow_table()
        assert table.schema.field("embedding").type == pa.list_(pa.float32(), 768)
        rows = sorted(table.to_pylist(), key=lambda row: row["ordinal"])
        assert [row["ordinal"] for row in rows] == list(range(len(TEXTS)))
        return assert_vectors(np.asarray([row["embedding"] for row in rows], dtype=np.float32), len(TEXTS))


def test_real_mlx_public_vane_embed_runs_in_local_actor(local_actor):
    vectors = public_vectors("mlx")
    assert min_cosine(vectors[[0]], vectors[[4]]) >= 0.999


def test_real_mlx_repeated_batches_and_cleanup(local_actor, tmp_path):
    from vane.ai.providers.mlx import MLXProvider, MLXTextEmbedderDescriptor

    trace = str(tmp_path / "actor.jsonl")
    parent_pid = os.getpid()

    def event(kind):
        from pathlib import Path

        with Path(trace).open("a") as log:
            log.write(json.dumps({"event": kind, "pid": os.getpid()}) + "\n")

    class TracedDescriptor(MLXTextEmbedderDescriptor):
        def instantiate(self):
            actual = super().instantiate()
            event("loaded")

            class TracedEmbedder:
                def embed_text(self, texts):
                    event("batch")
                    return actual.embed_text(texts)

                async def aclose(self):
                    await actual.aclose()
                    event("closed")

            return TracedEmbedder()

    class TracedProvider(MLXProvider):
        def get_text_embedder(self, model=None, dimensions=None, *, options=None):
            return TracedDescriptor(model, dimensions, options=options)

    vectors = public_vectors(TracedProvider())
    assert min_cosine(vectors[[0]], vectors[[4]]) >= 0.999
    records = [json.loads(line) for line in (tmp_path / "actor.jsonl").read_text().splitlines()]
    kinds = [record["event"] for record in records]
    assert kinds.count("loaded") == kinds.count("closed") == 1
    assert kinds.count("batch") >= 3 and kinds[0] == "loaded" and kinds[-1] == "closed"
    assert len({record["pid"] for record in records}) == 1
    assert records[0]["pid"] != parent_pid
    import psutil

    assert not psutil.pid_exists(records[0]["pid"])
