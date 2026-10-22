# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit real SDK/Metal/audio-file/public-worker acceptance; no Ray fallback."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from mlx_metal_reference import audio_inputs, infer, parity, sdk_model, text_vectors, traced_provider, vectors

pytest_plugins = ("mlx_metal_reference",)
pytestmark = [pytest.mark.external_service, pytest.mark.metal, pytest.mark.usefixtures("prepared_metal")]


@pytest.fixture(scope="module")
def waveforms():
    rows = []
    for n in (161, 162, 319, 320, 321, 16000, 16001, 160000, 480000):
        rows.append((0.2 * np.sin(2 * np.pi * 440 * np.arange(n, dtype=np.float64) / 16000)).astype(np.float32))
    rows.append(np.zeros(16000, dtype=np.float32))
    rows.append(rows[6].copy())
    return rows


@pytest.fixture(scope="module")
def reference(waveforms):
    with sdk_model() as sdk:
        results = []
        for row in waveforms:
            encoded = audio_inputs(sdk.processor, [row])
            expected = np.arange(encoded["input_features_mask"].shape[1]) * 160 + 160 < len(row)
            np.testing.assert_array_equal(encoded["input_features_mask"][0], expected)
            assert int((encoded["input_ids"] == sdk.processor.audio_token_id).sum()) == int(expected[::4].sum())
            results.append(infer(sdk.model, encoded)[0])
    return vectors(np.stack(results), len(waveforms))


def test_audio_real_sdk_parity_boundaries_masks_batches_and_input_storage(waveforms, reference, record_property):
    from vane.ai import AudioClip
    from vane.ai.provider import load_provider

    encoder = load_provider("mlx").get_audio_embedder().instantiate()
    try:
        singles = []
        for row in waveforms:
            samples = np.zeros((len(row) * 2, 1), dtype=np.float64)
            samples[::2, 0] = row
            view = samples[::2]
            view.setflags(write=False)
            singles.append(encoder.embed_audio([AudioClip(view, 16000)])[0])
            np.testing.assert_array_equal(view[:, 0], row)
        parity(vectors(np.stack(singles), len(waveforms)), reference, record_property, "sdk_provider")
        for indices in ([0, 8], [1, 4], [6, 0, 6]):
            batch = vectors(
                encoder.embed_audio([AudioClip(waveforms[i][:, None], 16000) for i in indices]), len(indices)
            )
            parity(batch, reference[indices], record_property, "batch_" + "_".join(map(str, indices)))
        assert encoder.embed_audio([]) == []
    finally:
        encoder.close()
    encoder.close()
    assert encoder._model is encoder._processor is None


def test_audio_real_limits_fail_without_truncating():
    from vane.ai import AudioClip
    from vane.ai._embedding_inputs import EmbeddingConfigurationError
    from vane.ai.provider import load_provider

    for options in ({}, {"max_length": 4}):
        encoder = load_provider("mlx").get_audio_embedder(options=options).instantiate()
        try:
            if options:
                with pytest.raises(EmbeddingConfigurationError, match="max_length"):
                    encoder.embed_audio([AudioClip(np.zeros((16000, 1), np.float32), 16000)])
            else:
                for count in (0, 1, 160, 480001):
                    with pytest.raises(EmbeddingConfigurationError, match="161 to 480000"):
                        encoder.embed_audio([AudioClip(np.zeros((count, 1), np.float32), 16000)])
        finally:
            encoder.close()


def public_audio(waveforms, entry="relation", provider="mlx"):
    import pyarrow as pa

    import vane
    from vane.ai import AudioInput, embed

    samples = vane.tensor_array(
        [None if row is None else row.astype(np.float64)[:, None] for row in waveforms],
        vane.tensor_type(vane.sqltypes.DOUBLE, (None, 1)),
    )
    audio = pa.StructArray.from_arrays(
        [pa.array([16000] * len(waveforms), pa.int32()), samples],
        names=["sample_rate", "data"],
        mask=pa.array([r is None for r in waveforms]),
    )
    with vane.connect() as conn:
        conn.register("audio_rows", pa.table({"id": range(len(waveforms)), "audio": audio}))
        rel = conn.table("audio_rows")
        options = dict(provider=provider, batch_size=2, actor_number=1, max_retries=0)
        if entry == "relation":
            result = embed(rel, AudioInput(vane.col("audio")), **options).select("id", "embedding")
        elif entry == "expression":
            result = rel.select(vane.col("id"), embed(AudioInput(vane.col("audio")), **options).alias("embedding"))
        else:
            result = conn.sql(
                "SELECT id, ai_embed(audio, input_spec => {kind: 'audio'}, provider => 'mlx', "
                "options => {batch_size: 2, max_retries: 0}) AS embedding FROM audio_rows"
            )
        table = result.to_arrow_table()
        assert table.schema.field("embedding").type == pa.list_(pa.float32(), 768)
        rows = sorted(table.to_pylist(), key=lambda r: r["id"])
        assert [r["id"] for r in rows] == list(range(len(waveforms)))
        return [row["embedding"] for row in rows]


@pytest.mark.parametrize("entry", ["relation", "expression", "sql"])
def test_audio_public_apis_preserve_nulls_order_and_reference(
    local_actor, waveforms, reference, entry, record_property
):
    indices = [5, 0, 6, 5]
    rows = [waveforms[indices[0]], None, *(waveforms[i] for i in indices[1:])]
    actual = public_audio(rows, entry)
    assert actual[1] is None
    parity(
        vectors(np.asarray([r for r in actual if r is not None], dtype=np.float32), 4),
        reference[indices],
        record_property,
        entry + "_sdk",
    )


def test_audio_real_wav_decode_48k_stereo_resample_and_public_actor(local_actor, tmp_path, record_property):
    import soundfile as sf
    import soxr

    import vane

    t = np.arange(48000 * 2) / 48000
    source = np.stack([0.2 * np.sin(2 * np.pi * 220 * t), 0.1 * np.sin(2 * np.pi * 440 * t)], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(path, source, 48000, subtype="PCM_16")
    with vane.connect() as conn:
        actual = vane.AudioFile(str(path), "audio/wav").resample(16000, connection=conn)
    decoded, rate = sf.read(path, dtype="float64", always_2d=True)
    np.testing.assert_allclose(actual, soxr.resample(decoded, rate, 16000, quality="HQ"), atol=1e-9, rtol=0)
    assert actual.shape == (32000, 2)
    mono = actual.mean(axis=1).astype(np.float32)
    with sdk_model() as sdk:
        expected = infer(sdk.model, audio_inputs(sdk.processor, [mono]))
    result = public_audio([mono])
    parity(vectors(np.asarray(result, dtype=np.float32), 1), expected, record_property, "wav_sdk_actor")


def test_audio_actor_uses_metal_loads_once_and_exits(local_actor, tmp_path, waveforms, reference, record_property):
    import psutil

    trace = tmp_path / "audio-actor.jsonl"
    indices = [5, 0, 6, 5, 6]
    result = public_audio([waveforms[i] for i in indices], provider=traced_provider("audio", trace))
    parity(vectors(np.asarray(result, dtype=np.float32), 5), reference[indices], record_property, "actor_sdk")
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    events = [r["event"] for r in records]
    assert events.count("loaded") == events.count("processor_loaded") == events.count("closed") == 1
    assert events.count("forward") >= 3
    assert len({r["pid"] for r in records}) == 1 and records[0]["pid"] != os.getpid()
    assert all(r["metal"] and r["machine"] == "arm64" for r in records)
    assert not psutil.pid_exists(records[0]["pid"])
    assert [n for r in records if r["event"] == "input" for n in r["samples"]] == [len(waveforms[i]) for i in indices]
    record_property("actor_trace", json.dumps(records))


def test_audio_frozen_semantic_corpus_matches_expected_ids(record_property):
    import soundfile as sf

    from vane.ai import AudioClip
    from vane.ai.provider import load_provider

    root = Path(os.environ["VANE_MLX_AV_FIXTURES"])
    manifest = json.loads((root / "manifest.json").read_text())["audio"]
    waveforms = []
    for entry in manifest:
        path = root / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        assert rate == 16000 and samples.shape[1] == 1
        waveforms.append(samples)
    with sdk_model() as sdk:
        queries = text_vectors(sdk.model, sdk.processor, [e["query"] for e in manifest])
    encoder = load_provider("mlx").get_audio_embedder().instantiate()
    try:
        encoded = vectors(
            np.stack([encoder.embed_audio([AudioClip(row, 16000)])[0] for row in waveforms]), len(manifest)
        )
    finally:
        encoder.close()
    scores = queries @ encoded.T
    assert [manifest[i]["id"] for i in scores.argmax(axis=1)] == [e["expected_id"] for e in manifest]
    record_property("text_audio_scores", json.dumps(scores.tolist()))
