# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real video processor/Metal/public actor/decoded-fixture retrieval acceptance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from mlx_metal_reference import infer, parity, sdk_model, text_vectors, traced_provider, vectors, video_inputs

pytest_plugins = ("mlx_metal_reference",)
pytestmark = [pytest.mark.external_service, pytest.mark.metal, pytest.mark.usefixtures("prepared_metal")]


@pytest.fixture(scope="module")
def clips():
    from vane.ai import VideoClip

    result = []
    for count, height, width in [(1, 1, 5), (2, 3, 5), (3, 48, 96), (8, 96, 48), (12, 64, 64)]:
        frames = []
        for i in range(count):
            raw = np.arange(height * width * 6, dtype=np.uint32).reshape(height, width * 2, 3)
            raw = ((raw * 7 + i * 31) % 256).astype(np.uint8)[:, ::2]
            raw.setflags(write=False)
            assert not raw.flags.c_contiguous
            frames.append(raw)
        result.append(
            VideoClip(
                tuple(frames), tuple(4.5 + i * i / 16 for i in range(count)), tuple(7 + i * 3 for i in range(count))
            )
        )
    return result


@pytest.fixture(scope="module")
def reference(clips):
    arrays, results = [], []
    with sdk_model() as sdk:
        for clip in clips:
            encoded = video_inputs(sdk.processor, [np.stack(clip.frames)])
            assert int(encoded["num_frames_per_video"][0]) == len(clip.frames)
            arrays.append({key: np.asarray(value).copy() for key, value in encoded.items()})
            results.append(infer(sdk.model, encoded)[0])
    return arrays, vectors(np.stack(results), len(clips))


def test_video_real_chw_hwc_exact_processor_and_sdk_provider_parity(clips, reference, record_property):
    from vane.ai.provider import load_provider

    expected_arrays, expected = reference
    encoder = load_provider("mlx").get_video_embedder().instantiate()
    try:
        singles = []
        for clip, reference_inputs in zip(clips, expected_arrays):
            actual_inputs = encoder._processor(
                videos=[np.stack(clip.frames)],
                text_kwargs={"padding": True, "truncation": False},
                videos_kwargs={
                    "input_data_format": "channels_last",
                    "do_sample_frames": False,
                    "add_timestamps": False,
                    "return_metadata": False,
                },
                return_tensors="np",
            )
            assert set(actual_inputs) == set(reference_inputs)
            for key in reference_inputs:
                np.testing.assert_array_equal(actual_inputs[key], reference_inputs[key])
            singles.append(encoder.embed_video([clip])[0])
        parity(vectors(np.stack(singles), len(clips)), expected, record_property, "chw_hwc_sdk_provider")
        for indices in ([0, 2], [1, 4], [3, 0, 3]):
            actual = vectors(encoder.embed_video([clips[i] for i in indices]), len(indices))
            parity(actual, expected[indices], record_property, "batch_" + "_".join(map(str, indices)))
        assert encoder.embed_video([]) == []
    finally:
        encoder.close()


def test_video_real_frame_and_token_limits_are_not_sampled_away(clips):
    from vane.ai import VideoClip
    from vane.ai._embedding_inputs import EmbeddingConfigurationError
    from vane.ai.provider import load_provider

    encoder = load_provider("mlx").get_video_embedder().instantiate()
    try:
        row = VideoClip(clips[0].frames * 13, tuple(float(i) for i in range(13)), tuple(range(13)))
        with pytest.raises(EmbeddingConfigurationError, match="1 to 12"):
            encoder.embed_video([row])
    finally:
        encoder.close()
    encoder = load_provider("mlx").get_video_embedder(options={"max_length": 16}).instantiate()
    try:
        with pytest.raises(EmbeddingConfigurationError, match="max_length"):
            encoder.embed_video([clips[0]])
    finally:
        encoder.close()


def public_video(clips, entry="relation", provider="mlx"):
    import pyarrow as pa

    import vane
    from vane.ai import VideoInput, embed
    from vane.execution.udf_file_contract import FileUDFContract

    dtype = vane.list_type(
        vane.struct_type(
            {"frame_index": vane.sqltypes.BIGINT, "frame_time": vane.sqltypes.DOUBLE, "data": vane.image_type("RGB")}
        )
    )
    rows = [
        None
        if clip is None
        else [
            dict(frame_index=i, frame_time=t, data=f)
            for f, t, i in zip(clip.frames, clip.frame_times, clip.frame_indices)
        ]
        for clip in clips
    ]
    array = FileUDFContract("fixture", (), (dtype,)).scalar_outputs_to_array(rows)
    with vane.connect() as conn:
        conn.register("video_rows", pa.table({"id": range(len(rows)), "frames": array}))
        rel = conn.table("video_rows")
        options = dict(provider=provider, batch_size=2, actor_number=1, max_retries=0)
        if entry == "relation":
            result = embed(rel, VideoInput(vane.col("frames")), **options).select("id", "embedding")
        elif entry == "expression":
            result = rel.select(vane.col("id"), embed(VideoInput(vane.col("frames")), **options).alias("embedding"))
        else:
            result = conn.sql(
                "SELECT id, ai_embed(frames, input_spec => {kind: 'video'}, provider => 'mlx', "
                "options => {batch_size: 2, max_retries: 0}) AS embedding FROM video_rows"
            )
        table = result.to_arrow_table()
        assert table.schema.field("embedding").type == pa.list_(pa.float32(), 768)
        output = sorted(table.to_pylist(), key=lambda row: row["id"])
        assert [r["id"] for r in output] == list(range(len(rows)))
        return [r["embedding"] for r in output]


@pytest.mark.parametrize("entry", ["relation", "expression", "sql"])
def test_video_public_apis_preserve_nulls_clip_order_and_sdk_parity(
    local_actor, clips, reference, entry, record_property
):
    indices = [0, 2, 0]
    actual = public_video([None, clips[0], clips[2], None, clips[0]], entry)
    assert actual[0] is None and actual[3] is None
    parity(
        vectors(np.asarray([v for v in actual if v is not None], dtype=np.float32), 3),
        reference[1][indices],
        record_property,
        entry + "_sdk",
    )


def test_video_real_actor_preserves_times_indices_and_exits(local_actor, clips, reference, tmp_path, record_property):
    import psutil

    trace = tmp_path / "video-actor.jsonl"
    indices = [0, 2, 1, 0, 2]
    actual = public_video([clips[i] for i in indices], provider=traced_provider("video", trace))
    parity(vectors(np.asarray(actual, dtype=np.float32), 5), reference[1][indices], record_property, "actor_sdk")
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    events = [r["event"] for r in records]
    assert events.count("loaded") == events.count("processor_loaded") == events.count("closed") == 1
    assert events.count("forward") >= 3 and len({r["pid"] for r in records}) == 1
    assert records[0]["pid"] != os.getpid() and not psutil.pid_exists(records[0]["pid"])
    assert all(r["metal"] and r["machine"] == "arm64" for r in records)
    assert [v for r in records if r["event"] == "input" for v in r["indices"]] == [
        list(clips[i].frame_indices) for i in indices
    ]
    assert [v for r in records if r["event"] == "input" for v in r["times"]] == [
        list(clips[i].frame_times) for i in indices
    ]
    record_property("actor_trace", json.dumps(records))


def test_video_decoded_frozen_corpus_text_and_image_retrieval(local_actor, record_property):
    from PIL import Image

    import vane
    from vane.ai import VideoClip

    root = Path(os.environ["VANE_MLX_AV_FIXTURES"])
    manifest = json.loads((root / "manifest.json").read_text())["video"]
    clips = []
    for entry in manifest:
        path = root / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        decoded = []
        with vane.connect() as conn:
            reader = vane.VideoFile(str(path), "video/mp4").frames(connection=conn)
            try:
                for frame in reader:
                    try:
                        decoded.append((np.array(frame.data, copy=True), frame.frame_time, frame.frame_index))
                    finally:
                        frame.data.close()
            finally:
                reader.close()
        selected = [decoded[i] for i in entry["sample_indices"]]
        clips.append(
            VideoClip(tuple(x[0] for x in selected), tuple(x[1] for x in selected), tuple(x[2] for x in selected))
        )
    with sdk_model() as sdk:
        expected = np.stack([infer(sdk.model, video_inputs(sdk.processor, [np.stack(c.frames)]))[0] for c in clips])
        query = text_vectors(sdk.model, sdk.processor, [e["query"] for e in manifest])
        image_vectors = []
        for entry in manifest:
            with Image.open(root / entry["image_query"]) as im:
                image = np.array(im.convert("RGB")).transpose(2, 0, 1)
            encoded = sdk.processor(
                images=[[image]],
                images_kwargs={"input_data_format": "channels_first"},
                text_kwargs={"padding": True, "truncation": False},
                return_tensors="np",
            )
            image_vectors.append(infer(sdk.model, encoded)[0])
    actual = vectors(np.asarray(public_video(clips), dtype=np.float32), 3)
    parity(actual, expected, record_property, "decoded_video_sdk_actor")
    for name, queries in [("text", query), ("image", np.stack(image_vectors))]:
        scores = queries @ actual.T
        assert [manifest[i]["id"] for i in scores.argmax(axis=1)] == [e["expected_id"] for e in manifest]
        record_property(name + "_video_scores", json.dumps(scores.tolist()))
