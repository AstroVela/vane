# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Decoded video/processor contracts without optional SDKs or Metal."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._video_embedding import VideoClip
from vane.ai.provider import load_provider

pytest_plugins = ("mlx_sdk_fixture",)


def clip(count=2, height=3, width=5):
    return VideoClip(
        tuple(np.full((height, width, 3), n, np.uint8) for n in range(count)),
        tuple(1.25 + n / 3 for n in range(count)),
        tuple(7 + n * 3 for n in range(count)),
    )


def test_video_patch_coordinates_use_sdk_xy_order_for_rectangular_grids(media_sdk):
    # A 6-wide by 3-high patch grid yields two pooled tokens per frame.
    grid = np.array(
        [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (-1, -1),
        ]
    )

    def rectangular(inputs):
        inputs["input_ids"] = np.array([[2, 258884, 258884, 258884, 258884, 1]])
        inputs["attention_mask"] = np.ones((1, 6), dtype=np.int64)
        inputs["pixel_values_videos"] = np.zeros((2, 19, 768), dtype=np.float32)
        inputs["video_position_ids"] = np.tile(grid, (2, 1, 1))
        return inputs

    media_sdk.change = rectangular
    encoder = load_provider("mlx").get_video_embedder().instantiate()
    try:
        assert np.asarray(encoder.embed_video([clip()])).shape == (1, 768)
        np.testing.assert_array_equal(media_sdk.model_calls[0]["video_position_ids"][0], grid)
    finally:
        encoder.close()


def test_video_preserves_ragged_clip_boundaries_hwc_pixels_and_provenance(media_sdk):
    source = np.arange(3 * 10 * 3, dtype=np.uint8).reshape(3, 10, 3)[:, ::2]
    source.setflags(write=False)
    rows = [clip(1, 1, 5), VideoClip((source, source), (2.0, 2.0), (9, 17)), clip(3, 4, 8)]
    encoder = load_provider("mlx").get_video_embedder().instantiate()
    try:
        actual = np.asarray(encoder.embed_video(rows))
        assert actual.shape == (3, 768) and actual.dtype == np.float32
        assert media_sdk.loads == media_sdk.processor_loads == len(media_sdk.model_calls) == 1
        sent = media_sdk.processor_calls[0]["videos"]
        assert [len(row) for row in sent] == [1, 2, 3]
        for original, converted in zip(rows, sent):
            np.testing.assert_array_equal(converted, np.stack(original.frames))
            assert all(not np.shares_memory(converted, frame) for frame in original.frames)
        assert rows[1].frame_times == (2.0, 2.0) and rows[1].frame_indices == (9, 17)
        assert not source.flags.writeable and not source.flags.c_contiguous
        inputs = media_sdk.model_calls[0]
        assert set(inputs) == {"input_ids", "attention_mask", "pixel_values_videos", "video_position_ids"}
        assert inputs["pixel_values_videos"].shape == (6, 10, 768)
        encoder.embed_video([rows[2], rows[0], rows[2]])
        assert media_sdk.loads == media_sdk.processor_loads == 1
        calls = len(media_sdk.processor_calls)
        assert encoder.embed_video([]) == [] and len(media_sdk.processor_calls) == calls
    finally:
        encoder.close()
    encoder.close()
    assert encoder._model is encoder._processor is encoder._tokenizer is None and media_sdk.clears == 1
    with pytest.raises(RuntimeError, match="closed"):
        encoder.embed_video([])


@pytest.mark.parametrize(
    "case",
    [
        "not_list",
        "not_clip",
        "sequences",
        "zero",
        "13",
        "lengths",
        "index_bool",
        "index_float",
        "index_negative",
        "index_duplicate",
        "index_reverse",
        "time_bool",
        "time_nan",
        "time_inf",
        "time_negative",
        "time_reverse",
        "float_pixels",
        "chw",
        "empty_pixels",
        "shape_mismatch",
        "pixel_budget",
        "byte_budget",
    ],
)
def test_video_raw_inputs_fail_before_stack_or_processor(media_sdk, case):
    rows = [clip()]
    original = rows[0]
    frames, times, indices = original.frames, original.frame_times, original.frame_indices
    if case == "not_list":
        rows = tuple(rows)
    elif case == "not_clip":
        rows = ["video.mp4"]
    elif case == "sequences":
        rows = [VideoClip(None, times, indices)]
    elif case in {"zero", "13"}:
        rows = [clip(0 if case == "zero" else 13)]
    else:
        if case == "lengths":
            times = times[:1]
        elif case.startswith("index_"):
            indices = (
                7,
                {
                    "index_bool": True,
                    "index_float": 8.0,
                    "index_negative": -1,
                    "index_duplicate": 7,
                    "index_reverse": 6,
                }[case],
            )
        elif case.startswith("time_"):
            times = (
                1.25,
                {"time_bool": True, "time_nan": np.nan, "time_inf": np.inf, "time_negative": -1, "time_reverse": 1}[
                    case
                ],
            )
        elif case == "float_pixels":
            frames = tuple(f.astype(np.float32) / 255 for f in frames)
        elif case == "chw":
            frames = tuple(f.transpose(2, 0, 1) for f in frames)
        elif case == "empty_pixels":
            frames = tuple(f[:0] for f in frames)
        elif case == "shape_mismatch":
            frames = (frames[0], frames[1][:1])
        else:
            shape = (4097, 4096, 3) if case == "pixel_budget" else (4096, 4096, 3)
            frame = np.broadcast_to(np.zeros((1, 1, 3), np.uint8), shape)
            frames = (frame, frame)
        rows = [VideoClip(frames, times, indices)]
    encoder = load_provider("mlx").get_video_embedder().instantiate()
    try:
        with pytest.raises(EmbeddingConfigurationError):
            encoder.embed_video(rows)
        assert media_sdk.processor_calls == media_sdk.model_calls == []
    finally:
        encoder.close()


@pytest.mark.parametrize(
    "case",
    [
        "unknown",
        "missing",
        "not_mapping",
        "counts_rows",
        "counts_bool",
        "counts_wrong",
        "patches_rows",
        "patches_width",
        "patches_fp16",
        "patches_nan",
        "patches_many",
        "positions_float",
        "positions_shape",
        "positions_mixed_padding",
        "positions_all_padding",
        "positions_large",
        "positions_grid",
        "positions_order",
        "positions_hole",
        "token_missing",
        "token_masked",
        "ids_float",
        "attention_float",
        "attention_zero",
    ],
)
def test_video_processor_corruption_is_rejected_before_forward(media_sdk, case):
    def corrupt(inputs):
        if case == "not_mapping":
            return []
        if case == "unknown":
            inputs["video_metadata"] = object()
        elif case == "missing":
            inputs.pop("num_frames_per_video")
        elif case.startswith("counts_"):
            key = "num_frames_per_video"
            inputs[key] = {
                "counts_rows": np.array([4]),
                "counts_bool": np.array([True, True]),
                "counts_wrong": np.array([2, 2]),
            }[case]
        elif case.startswith("patches_"):
            key = "pixel_values_videos"
            if case == "patches_rows":
                inputs[key] = inputs[key][:2]
                inputs["video_position_ids"] = inputs["video_position_ids"][:2]
            elif case == "patches_width":
                inputs[key] = inputs[key][:, :, :767]
            elif case == "patches_many":
                inputs[key] = np.zeros((4, 1261, 768), dtype=np.float32)
            elif case == "patches_fp16":
                inputs[key] = inputs[key].astype(np.float16)
            else:
                inputs[key][0, 0, 0] = np.nan
        elif case.startswith("positions_"):
            key = "video_position_ids"
            if case == "positions_float":
                inputs[key] = inputs[key].astype(np.float32)
            elif case == "positions_shape":
                inputs[key] = inputs[key][:, :1]
            elif case == "positions_all_padding":
                inputs[key][:] = -1
            elif case == "positions_hole":
                row = inputs[key][0].copy()
                inputs[key][0] = np.concatenate([row[:4], row[-1:], row[4:-1]])
            else:
                inputs[key][0, 0] = {
                    "positions_mixed_padding": [-1, 0],
                    "positions_large": [10240, 0],
                    "positions_grid": [3, 0],
                    "positions_order": [0, 1],
                }[case]
        elif case == "token_missing":
            inputs["input_ids"][0, 1] = 1
        elif case == "token_masked":
            inputs["attention_mask"][0, 1] = 0
        elif case == "ids_float":
            inputs["input_ids"] = inputs["input_ids"].astype(np.float32)
        elif case == "attention_float":
            inputs["attention_mask"] = inputs["attention_mask"].astype(np.float32)
        else:
            inputs["attention_mask"][:] = 0
        return inputs

    media_sdk.change = corrupt
    encoder = load_provider("mlx").get_video_embedder().instantiate()
    try:
        with pytest.raises(EmbeddingConfigurationError):
            encoder.embed_video([clip(1), clip(3)])
        assert media_sdk.model_calls == []
    finally:
        encoder.close()


def test_video_expanded_tokens_fail_without_dropping_frames(media_sdk):
    encoder = load_provider("mlx").get_video_embedder(options={"max_length": 3}).instantiate()
    try:
        with pytest.raises(EmbeddingConfigurationError, match="max_length"):
            encoder.embed_video([clip(3)])
        assert len(media_sdk.processor_calls[0]["videos"][0]) == 3 and media_sdk.model_calls == []
    finally:
        encoder.close()


@pytest.mark.parametrize("field", ["video_processor", "runtime_processor", "video_token_id", "vision_tower", "weights"])
def test_video_checkpoint_checks_and_failed_init_cleanup(media_sdk, field):
    if field == "video_processor":
        path = media_sdk.path / "processor_config.json"
        config = json.loads(path.read_text())
        config["video_processor"]["max_soft_tokens"] = 280
        path.write_text(json.dumps(config))
    elif field == "runtime_processor":
        media_sdk.processor.video_processor.add_timestamps = True
    elif field == "video_token_id":
        media_sdk.processor.video_token_id = 1
    elif field == "vision_tower":
        media_sdk.model.vision_tower = None
    else:
        del media_sdk.weights["embed_vision.embedding_projection.weight"]
    with pytest.raises(EmbeddingConfigurationError):
        load_provider("mlx").get_video_embedder().instantiate()
    assert media_sdk.clears == 1


@pytest.mark.parametrize("case", ["rows", "dimensions", "nan", "inf", "zero", "norm"])
def test_video_validates_native_output(media_sdk, case):
    output = np.eye(1, 768, dtype=np.float32)
    if case == "rows":
        output = np.tile(output, (2, 1))
    elif case == "dimensions":
        output = output[:, :767]
    else:
        output[0, 0] = {"nan": np.nan, "inf": np.inf, "zero": 0, "norm": 2}[case]
    media_sdk.output = output
    encoder = load_provider("mlx").get_video_embedder().instantiate()
    try:
        with pytest.raises(ValueError):
            encoder.embed_video([clip()])
    finally:
        encoder.close()


@pytest.mark.parametrize("policy", ["raise", "ignore"])
@pytest.mark.parametrize("case", ["processor", "allocation"])
def test_video_processor_and_allocation_failures_are_fatal_under_both_policies(media_sdk, policy, case):
    import asyncio

    import pyarrow as pa

    import vane
    from vane.ai.functions import _EmbedVideoBatch
    from vane.execution.udf_file_contract import FileUDFContract

    value = clip()
    dtype = vane.list_type(
        vane.struct_type(
            {"frame_index": vane.sqltypes.BIGINT, "frame_time": vane.sqltypes.DOUBLE, "data": vane.image_type("RGB")}
        )
    )
    records = [
        dict(frame_index=i, frame_time=t, data=f)
        for f, t, i in zip(value.frames, value.frame_times, value.frame_indices)
    ]
    column = FileUDFContract("fixture", (), (dtype,)).scalar_outputs_to_array([records])
    if case == "processor":

        def corrupted(inputs):
            inputs["num_frames_per_video"][0] = 1
            return inputs

        media_sdk.change = corrupted
    else:
        media_sdk.failures["forward"] = MemoryError("private video path")
    wrapper = _EmbedVideoBatch(
        load_provider("mlx").get_video_embedder(), "frames", "embedding", 768, on_error=policy, max_retries=0
    )
    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        with pytest.raises(MemoryError if case == "allocation" else EmbeddingConfigurationError) as caught:
            wrapper(pa.table({"frames": column}))
        assert "private video path" not in str(caught.value)
        assert len(media_sdk.model_calls) == (1 if case == "allocation" else 0)
    finally:
        wrapper.close()
        loop.close()


def test_video_async_close_is_idempotent(media_sdk):
    import asyncio

    encoder = load_provider("mlx").get_video_embedder().instantiate()
    asyncio.run(encoder.aclose())
    asyncio.run(encoder.aclose())
    assert encoder._model is encoder._processor is encoder._tokenizer is None
    assert media_sdk.clears == 1
