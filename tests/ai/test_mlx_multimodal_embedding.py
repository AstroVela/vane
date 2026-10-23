# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Fixed SDK/Metal joint parity, public actors, real cancellation and fatal faults."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from mlx_metal_reference import infer, joint_inputs, joint_parity, sdk_model, traced_provider, vectors

pytest_plugins = ("mlx_metal_reference",)
pytestmark = [pytest.mark.external_service, pytest.mark.metal, pytest.mark.usefixtures("prepared_metal")]


def content(*parts):
    from vane.ai import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec

    return EmbeddingContent(tuple(EmbeddingPart(EmbeddingPartSpec(kind), value) for kind, value in parts))


@pytest.fixture(scope="module")
def cases():
    from vane.ai import AudioClip, VideoClip

    a = np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3)
    backing = np.empty((32, 96, 3), np.uint8)
    backing[:, ::2] = a
    a = backing[:, ::2]
    a.setflags(write=False)
    b = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]]], np.uint8)
    c = np.flip(np.tile(b, (3, 1, 1)), axis=1)
    v = VideoClip((a, a[:, ::-1]), (1.25, 1.25), (7, 12))
    w = VideoClip((c,), (3.5,), (21,))
    x = AudioClip((0.1 * np.sin(2 * np.pi * 440 * np.arange(4001) / 16000)).astype(np.float32)[:, None], 16000)
    y = AudioClip(np.zeros((8001, 1), np.float32), 16000)
    return [
        content(("text", "plain text")),
        content(("image", a)),
        content(("video", v)),
        content(("audio", x)),
        content(("text", "scene:"), ("image", a)),
        content(("text", "clip:"), ("video", v)),
        content(("text", "sound:"), ("audio", x)),
        content(("text", "scene:"), ("image", a), ("text", " clip:"), ("video", v), ("text", " sound:"), ("audio", x)),
        content(("audio", y), ("image", b), ("audio", x), ("video", w), ("image", c), ("video", v)),
        content(("video", w), ("text", " / "), ("audio", x), ("image", c)),
        content(("image", c), ("text", " / "), ("audio", x), ("video", w)),
    ]


@pytest.fixture(scope="module")
def public_cases(cases):
    a, v, x = cases[1].parts[0].value, cases[2].parts[0].value, cases[3].parts[0].value
    b, y = cases[8].parts[1].value, cases[8].parts[0].value
    layout = ("text", "image", "text", "video", "text", "audio")
    rows = [
        ["<|image|>", None, None, None, None, None],
        ["scene:", a, " clip:", v, " sound:", x],
        [None, None, "plain text", None, " \t", None],
        [None, b, None, None, None, y],
        ["scene:", a, " clip:", v, " sound:", x],
        [None, None, None, None, None, None],
        ["last document", None, None, None, None, None],
    ]
    selected = [False, True, True, True, True, False, True]
    active = [
        content(
            *[
                (kind, value)
                for kind, value in zip(layout, row)
                if value is not None and not (kind == "text" and not value.strip())
            ]
        )
        for row, keep in zip(rows, selected)
        if keep
    ]
    return layout, rows, selected, active


@pytest.fixture(scope="module")
def reference(cases, public_cases, tmp_path_factory):
    path = tmp_path_factory.mktemp("joint-sdk-reference")
    inputs, results, groups = [], [], {}
    with sdk_model() as sdk:
        for i, row in enumerate(cases):
            encoded = joint_inputs(sdk.processor, [row])
            inputs.append({key: np.asarray(value).copy() for key, value in encoded.items()})
            results.append(infer(sdk.model, encoded)[0])
            np.savez(path / f"case-{i}.npz", **inputs[-1])
        for ids in ((0, 7, 8), (8, 0, 7), (4, 5, 6), (1, 3, 2, 0), (9, 10)):
            encoded = joint_inputs(sdk.processor, [cases[i] for i in ids])
            groups[ids] = ({key: np.asarray(value).copy() for key, value in encoded.items()}, infer(sdk.model, encoded))
        public = np.stack([infer(sdk.model, joint_inputs(sdk.processor, [row]))[0] for row in public_cases[3]])
    values = vectors(np.stack(results), len(cases))
    np.save(path / "vectors.npy", values)
    return inputs, values, groups, public


def test_real_joint_tensors_order_repeats_batch_layout_and_single_part_parity(cases, reference, record_property):
    from vane.ai.provider import load_provider
    from vane.ai.providers._mlx_multimodal import joint_plan

    arrays, expected, groups, _ = reference
    descriptor = load_provider("mlx").get_multimodal_embedder()
    encoder = descriptor.instantiate()
    forwards = []

    class ObservedModel:
        def __init__(self, model):
            self.model = model

        def __call__(self, **kwargs):
            forwards.append({key: list(value.shape) for key, value in kwargs.items()})
            return self.model(**kwargs)

    encoder._model = ObservedModel(encoder._model)
    try:
        assert encoder.embed_multimodal([]) == [] and forwards == []
        for i, row in enumerate(cases):
            prepared = encoder._prepare_multimodal_inputs(joint_plan([row], descriptor.get_input_spec()))
            assert set(prepared) == set(arrays[i]) - {"num_frames_per_video"}
            for key in prepared:
                np.testing.assert_array_equal(prepared[key], arrays[i][key])
            before = len(forwards)
            actual = vectors(encoder.embed_multimodal([row]), 1)
            joint_parity(actual, expected[i : i + 1], record_property, f"single-{i}")
            assert len(forwards) == before + 1
        assert set(forwards[7]) == {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_position_ids",
            "pixel_values_videos",
            "video_position_ids",
            "input_features",
            "input_features_mask",
        }
        for ids, (encoded, sdk_values) in groups.items():
            rows = [cases[i] for i in ids]
            prepared = encoder._prepare_multimodal_inputs(joint_plan(rows, descriptor.get_input_spec()))
            for key in prepared:
                np.testing.assert_array_equal(prepared[key], encoded[key])
            before = len(forwards)
            actual = vectors(encoder.embed_multimodal(rows), len(rows))
            assert len(forwards) == before + 1
            joint_parity(actual, sdk_values, record_property, "sdk-batch-" + str(ids))
            joint_parity(actual, expected[list(ids)], record_property, "singles-batch-" + str(ids))
        record_property("forward_shapes", json.dumps(forwards))
    finally:
        encoder.close()
        encoder.close()


@pytest.mark.parametrize("kind,index", [("text", 0), ("image", 1), ("video", 2), ("audio", 3)])
def test_joint_single_parts_match_unchanged_unary_encoder(cases, reference, kind, index, record_property):
    from vane.ai.provider import load_provider

    encoder = getattr(load_provider("mlx"), "get_" + kind + "_embedder")().instantiate()
    try:
        actual = getattr(encoder, "embed_" + kind)([cases[index].parts[0].value])
        joint_parity(actual, reference[1][index : index + 1], record_property, "unary-" + kind)
    finally:
        encoder.close()


def public_joint(layout, rows, selected, *, entry="relation", provider="mlx", on_error="raise", connection_hook=None):
    import pyarrow as pa

    import vane
    from vane.ai import AudioInput, ImageInput, MultimodalInput, TextInput, VideoInput, embed
    from vane.execution.udf_file_contract import FileUDFContract

    columns = {"id": pa.array(range(len(rows))), "selected": pa.array(selected)}
    factories = {"text": TextInput, "image": ImageInput, "video": VideoInput, "audio": AudioInput}
    for index, kind in enumerate(layout):
        values = [row[index] for row in rows]
        if kind == "text":
            column = pa.array(values, pa.string())
        elif kind == "image":
            column = FileUDFContract("fixture", (), (vane.image_type("RGB"),)).scalar_outputs_to_array(values)
        elif kind == "video":
            dtype = vane.list_type(
                vane.struct_type(
                    {
                        "frame_index": vane.sqltypes.BIGINT,
                        "frame_time": vane.sqltypes.DOUBLE,
                        "data": vane.image_type("RGB"),
                    }
                )
            )
            records = [
                None
                if clip is None
                else [
                    dict(frame_index=i, frame_time=t, data=f)
                    for f, t, i in zip(clip.frames, clip.frame_times, clip.frame_indices)
                ]
                for clip in values
            ]
            column = FileUDFContract("fixture", (), (dtype,)).scalar_outputs_to_array(records)
        else:
            tensor = vane.tensor_array(
                [None if clip is None else clip.samples for clip in values],
                vane.tensor_type(vane.sqltypes.FLOAT, (None, None)),
            )
            column = pa.StructArray.from_arrays(
                [pa.array([None if clip is None else clip.sample_rate for clip in values], pa.int32()), tensor],
                names=["sample_rate", "data"],
                mask=pa.array([clip is None for clip in values]),
            )
        columns[f"p{index}"] = column
    with vane.connect() as conn:
        if connection_hook is not None:
            connection_hook(conn)
        conn.register("joint_rows", pa.table(columns))
        rel = conn.table("joint_rows")
        input = MultimodalInput([factories[kind](vane.col(f"p{i}")) for i, kind in enumerate(layout)])
        options = dict(
            provider=provider,
            batch_size=2,
            actor_number=1,
            max_retries=0,
            on_error=on_error,
            where=vane.col("selected"),
        )
        if entry == "relation":
            result = embed(rel, input, **options).select("id", "embedding")
        elif entry == "expression":
            result = rel.select("id", embed(input, **options).alias("embedding"))
        else:
            fields = ",".join(f"p{i}:p{i}" for i in range(len(layout)))
            specs = ",".join("{kind:'" + kind + "',mime_type:NULL}" for kind in layout)
            result = conn.sql(
                "SELECT id, ai_embed({"
                + fields
                + "},input_spec=>["
                + specs
                + "],provider=>'mlx',options=>{batch_size:2,max_retries:0},\"where\"=>selected) AS embedding FROM joint_rows"
            )
        table = result.to_arrow_table()
        assert table.schema.field("embedding").type == pa.list_(pa.float32(), 768)
        output = sorted(table.to_pylist(), key=lambda row: row["id"])
        assert [row["id"] for row in output] == list(range(len(rows)))
        return [row["embedding"] for row in output]


@pytest.mark.parametrize("entry", ["relation", "expression", "sql"])
def test_public_joint_apis_where_nulls_order_and_exact_arrow_type(
    local_actor, public_cases, reference, entry, record_property
):
    layout, rows, selected, _ = public_cases
    actual = public_joint(layout, rows, selected, entry=entry)
    assert [value is not None for value in actual] == selected
    joint_parity([value for value in actual if value is not None], reference[3], record_property, entry + "-sdk")


def test_joint_public_actor_one_model_many_batches_and_exit(
    local_actor, public_cases, reference, tmp_path, record_property
):
    import psutil

    trace = tmp_path / "joint-worker.jsonl"
    actual = public_joint(*public_cases[:3], provider=traced_provider("multimodal", trace))
    joint_parity([value for value in actual if value is not None], reference[3], record_property, "actor-sdk")
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    events = [record["event"] for record in records]
    assert (
        events.count("checkpoint_load")
        == events.count("loaded")
        == events.count("processor_loaded")
        == events.count("closed")
        == 1
    )
    forward = [record for record in records if record["event"] == "forward"]
    assert len(forward) >= 3 and sum(record["rows"] for record in forward) == 5
    assert len({record["pid"] for record in records}) == 1 and records[0]["pid"] != os.getpid()
    assert not psutil.pid_exists(records[0]["pid"])
    assert all(record["metal"] and record["machine"] == "arm64" for record in records)
    assert any(
        {"pixel_values", "pixel_values_videos", "input_features"} <= set(record["model_keys"]) for record in forward
    )
    record_property("actor_trace", json.dumps(records))


@pytest.mark.parametrize("policy", ["raise", "ignore"])
def test_real_joint_actor_fatal_oom_propagates_and_reaps_worker(
    local_actor, public_cases, tmp_path, policy, record_property
):
    import psutil

    import vane

    trace = tmp_path / (policy + "-oom.jsonl")
    with pytest.raises(Exception) as error:
        public_joint(*public_cases[:3], provider=traced_provider("multimodal", trace, oom=True), on_error=policy)
    assert "memory" in str(error.value).lower()
    vane.teardown_runner()
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len([record for record in records if record["event"] == "forward"]) == 1
    assert not any(psutil.pid_exists(record["pid"]) for record in records)
    record_property("fatal_actor_trace", json.dumps(records))


def test_joint_query_cancellation_joins_actual_model_actor(local_actor, public_cases, tmp_path, record_property):
    import psutil

    import vane

    trace, gate = tmp_path / "cancel.jsonl", tmp_path / "release-forward"
    connections = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            public_joint,
            *public_cases[:3],
            provider=traced_provider("multimodal", trace, gate=str(gate)),
            connection_hook=connections.append,
        )
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if trace.exists() and any(
                    json.loads(line)["event"] == "waiting" for line in trace.read_text().splitlines()
                ):
                    break
                assert not future.done(), future.exception() if future.done() else None
                time.sleep(0.02)
            else:
                pytest.fail("real joint actor did not reach its forward gate")
            connections[0].interrupt()
            gate.touch()
            with pytest.raises(Exception):
                future.result(timeout=60)
        finally:
            gate.touch()
    vane.teardown_runner()
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert not any(psutil.pid_exists(record["pid"]) for record in records)
    record_property("cancel_actor_trace", json.dumps(records))


def test_real_joint_media_and_exact_expanded_token_boundaries(cases):
    from vane.ai import AudioClip, VideoClip
    from vane.ai._embedding_inputs import EmbeddingConfigurationError
    from vane.ai.provider import load_provider
    from vane.ai.providers._mlx_multimodal import joint_plan

    descriptor = load_provider("mlx").get_multimodal_embedder()
    encoder = descriptor.instantiate()
    try:
        clip = cases[2].parts[0].value
        for invalid in [
            content(("audio", AudioClip(np.zeros((160, 1), np.float32), 16000))),
            content(("video", VideoClip(clip.frames[:1] * 13, tuple(float(i) for i in range(13)), tuple(range(13))))),
        ]:
            with pytest.raises(EmbeddingConfigurationError):
                encoder.embed_multimodal([invalid])
        valid = cases[7]
        length = encoder._prepare_multimodal_inputs(joint_plan([valid], descriptor.get_input_spec()))[
            "input_ids"
        ].shape[1]
        encoder._options["max_length"] = length
        assert len(encoder.embed_multimodal([valid])) == 1
        encoder._options["max_length"] = length - 1
        with pytest.raises(EmbeddingConfigurationError, match="max_length"):
            encoder.embed_multimodal([valid])
        encoder._options["max_length"] = 2048
        with pytest.raises(EmbeddingConfigurationError, match="max_length"):
            encoder.embed_multimodal([content(*[("image", cases[1].parts[0].value)] * 9)])
    finally:
        encoder.close()


def test_frozen_joint_corpus_matches_independent_preimplementation_sdk_and_text_ranking(local_actor, record_property):
    from vane.ai import AudioClip, VideoClip
    from vane.ai.provider import load_provider

    root = Path(os.environ["VANE_MLX_JOINT_FIXTURES"])
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["frozen_before_implementation"] is True and len(manifest["joint"]) >= 3
    rows = []
    for entry in manifest["joint"]:
        with np.load(root / entry["input_file"]) as arrays:
            rows.append(
                content(
                    (
                        "video",
                        VideoClip(
                            tuple(arrays["frames"]), tuple(arrays["frame_times"]), tuple(arrays["frame_indices"])
                        ),
                    ),
                    ("audio", AudioClip(arrays["audio"][:, None], 16000)),
                    ("text", str(arrays["text"])),
                )
            )
    with np.load(root / "joint-sdk-reference.npz") as reference:
        expected, queries = reference["vectors"].copy(), reference["queries"].copy()
    encoder = load_provider("mlx").get_multimodal_embedder().instantiate()
    try:
        actual = np.stack([encoder.embed_multimodal([row])[0] for row in rows])
    finally:
        encoder.close()
    joint_parity(actual, expected, record_property, "frozen-joint-sdk")
    scores = queries @ actual.T
    assert [manifest["joint"][i]["id"] for i in scores.argmax(axis=1)] == [
        entry["expected_id"] for entry in manifest["joint"]
    ]
    record_property("frozen_scores", json.dumps(scores.tolist()))
