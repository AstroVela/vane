# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Ordered joint input contracts and adversarial processor outputs, without SDKs."""

import asyncio
import builtins
import pickle
import subprocess
import sys
from dataclasses import replace

import cloudpickle
import numpy as np
import pyarrow as pa
import pytest

from vane.ai import AudioClip, VideoClip
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec, content_from_columns
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai.functions import _EmbedMultimodalBatch, _prepare_embed_call
from vane.ai.provider import load_provider
from vane.ai.providers import mlx

pytest_plugins = ("mlx_sdk_fixture",)


def row(*parts):
    return EmbeddingContent(tuple(EmbeddingPart(EmbeddingPartSpec(kind), value) for kind, value in parts))


def image(height=3, value=19):
    return np.full((height, 5, 3), value, np.uint8)


def audio(length=161):
    return AudioClip(np.zeros((length, 1), np.float32), 16000)


def video(count=2, height=3):
    return VideoClip(
        tuple(image(height, 17 + i) for i in range(count)), tuple(i / 8 for i in range(count)), tuple(range(count))
    )


def descriptor(**kwargs):
    return load_provider("mlx").get_multimodal_embedder(**kwargs)


def test_joint_metadata_pickle_options_specs_and_cpu_dispatch():
    original = descriptor()
    expected = original.get_options()
    assert original.get_provider() == "mlx" and original.get_model() == mlx.MODEL_ID
    assert original.get_dimensions() == 768 and "input_type" not in expected
    assert expected["revision"] == "1a4ffddb7905d3f63486748deabe091a01fb6201"
    assert expected["dtype"] == "bfloat16" and expected["max_concurrency_per_actor"] == 1
    for encode in (pickle, cloudpickle):
        restored = encode.loads(encode.dumps(original))
        assert restored.get_options() == expected and restored.get_input_spec() == original.get_input_spec()
        assert restored.get_output_spec() == original.get_output_spec()
    spec = original.get_input_spec()
    assert spec.supported_parts == frozenset(EmbeddingPartSpec(kind) for kind in ("text", "image", "video", "audio"))
    assert (spec.max_parts, spec.max_input_bytes, spec.video.max_frames) == (16, 64 * 1024**2, 12)
    assert (spec.audio.sample_rate, spec.audio.max_samples, spec.audio.max_channels) == (16000, 480000, 1)
    (output,) = original.get_output_spec().outputs
    assert (output.name, output.dimensions, output.space, output.source_modalities) == (
        "embedding",
        768,
        mlx.MODEL_ID,
        ("text", "image", "video", "audio"),
    )
    prepared = _prepare_embed_call("mlx", None, None, "raise", {}, relation=True, input_kind="multimodal")
    assert prepared[-1] == "subprocess_actor" and prepared[2].num_gpus == 0
    for kind in ("image", "audio", "video"):
        assert getattr(load_provider("mlx"), f"get_{kind}_embedder")().get_output_spec().outputs[
            0
        ].source_modalities == (kind,)
    assert (
        load_provider("mlx").get_text_embedder(options={"input_type": "query"}).get_options()["input_type"] == "query"
    )
    changed = original.get_options()
    changed["max_length"] = 3
    assert original.get_options() == expected


@pytest.mark.parametrize(
    "options",
    [
        {"input_type": "query"},
        {"input_type": "document"},
        {"runtime": "http"},
        {"runtime": "service"},
        {"base_url": "http://127.0.0.1"},
        {"http_input_format": "multimodal"},
        {"dtype": "float16"},
        {"revision": "main"},
        {"revision": "a" * 40},
        {"max_length": True},
        {"max_length": 2049},
        {"overlength": "truncate"},
        {"max_concurrency_per_actor": 2},
        {"unknown": 1},
    ],
)
def test_joint_options_are_closed_in_both_descriptor_and_encoder(options):
    with pytest.raises((TypeError, EmbeddingConfigurationError)):
        descriptor(options=options)
    with pytest.raises((TypeError, EmbeddingConfigurationError)):
        mlx.MLXMultimodalEmbedder(options)


@pytest.mark.parametrize("dimensions", [True, False, 767, 769, 768.0, "768"])
def test_joint_rejects_dimension_overrides(dimensions):
    with pytest.raises(EmbeddingConfigurationError):
        descriptor(dimensions=dimensions)
    with pytest.raises(EmbeddingConfigurationError):
        descriptor().get_output_spec(dimensions=dimensions)


def test_joint_rejects_another_checkpoint():
    with pytest.raises(EmbeddingConfigurationError):
        descriptor(model="other/model")


def test_joint_metadata_and_explain_in_fresh_process_import_no_sdk(tmp_path):
    code = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'mlx','mlx_vlm','transformers','huggingface_hub','PIL'}:
        raise AssertionError('optional SDK imported during metadata: '+name)
    return original(name,*args,**kwargs)
builtins.__import__ = guarded
import vane
from vane.ai import embed, MultimodalInput, TextInput
from vane.ai.provider import load_provider
import pickle
assert load_provider('mlx').get_multimodal_embedder().get_dimensions() == 768
with vane.connect() as conn:
    rel = conn.sql("SELECT 'caption' AS text")
    plan = embed(rel, MultimodalInput([TextInput(vane.col('text'))]), provider='mlx').explain()
    assert plan
assert not {'mlx','mlx_vlm','transformers','huggingface_hub','PIL'}.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_joint_missing_sdk_instantiation_is_explicit(monkeypatch):
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "mlx.core":
            raise ModuleNotFoundError("missing optional package", name="mlx")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(mlx.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(mlx.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(builtins, "__import__", missing)
    assert descriptor().get_dimensions() == 768
    with pytest.raises(ImportError, match=r"vane-ai\[mlx\]"):
        descriptor().instantiate()


def test_four_modalities_one_forward_and_repeated_media_keep_exact_order(media_sdk):
    first_image, second_image = image(), image(6, 31)
    clip, sound, other = video(), audio(), audio(1281)
    rows = [
        row(("text", " scene:"), ("image", first_image), ("text", " / "), ("video", clip), ("audio", sound)),
        row(("text", "only text")),
        row(
            ("audio", other),
            ("image", second_image),
            ("audio", sound),
            ("video", video(1)),
            ("image", first_image),
            ("video", clip),
        ),
    ]
    before = [first_image.copy(), sound.samples.copy(), clip.frames[0].copy()]
    encoder = descriptor().instantiate()
    try:
        assert encoder.embed_multimodal([]) == [] and media_sdk.processor_calls == []
        values = encoder.embed_multimodal(rows)
        assert np.asarray(values).shape == (3, 768) and len(media_sdk.model_calls) == 1
        (call,) = media_sdk.processor_calls
        assert call["text"] == [
            "title: none | text:  scene:<|image|> / <|video|><|audio|>",
            "title: none | text: only text",
            "<|audio|><|image|><|audio|><|video|><|image|><|video|>",
        ]
        assert [len(r) for r in call["images"]] == [1, 0, 2]
        assert call["images"][0][0] is first_image and call["images"][2][0] is second_image
        assert call["images_kwargs"] == {"input_data_format": "channels_last"}
        assert [len(a) for a in call["audio"]] == [161, 1281, 161]
        assert all(a.dtype == np.float32 and not np.shares_memory(a, sound.samples) for a in call["audio"])
        assert [len(v) for v in call["videos"]] == [2, 1, 2]
        assert call["videos_kwargs"] == dict(
            input_data_format="channels_last", do_sample_frames=False, add_timestamps=False, return_metadata=False
        )
        assert set(media_sdk.model_calls[0]) == {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_position_ids",
            "pixel_values_videos",
            "video_position_ids",
            "input_features",
            "input_features_mask",
        }
        held = values[0].copy()
        encoder.embed_multimodal(rows[:1])
        np.testing.assert_array_equal(values[0], held)
        assert media_sdk.loads == media_sdk.processor_loads == 1
        for actual, expected in zip((first_image, sound.samples, clip.frames[0]), before):
            np.testing.assert_array_equal(actual, expected)
    finally:
        encoder.close()
        encoder.close()
        asyncio.run(encoder.aclose())
    assert media_sdk.clears == 1 and encoder._model is encoder._processor is encoder._tokenizer is None
    with pytest.raises(RuntimeError, match="closed"):
        encoder.embed_multimodal([])


@pytest.mark.parametrize("kind", ["text", "image", "audio", "video"])
def test_each_single_part_and_readonly_noncontiguous_images(media_sdk, kind):
    pixels = image()[:, ::2]
    pixels.setflags(write=False)
    value = {"text": " keep whitespace ", "image": pixels, "audio": audio(), "video": video(1)}[kind]
    encoder = descriptor().instantiate()
    try:
        assert len(encoder.embed_multimodal([row((kind, value))])) == 1
        assert len(media_sdk.model_calls) == 1
    finally:
        encoder.close()


@pytest.mark.parametrize(
    "bad",
    [
        None,
        (),
        [None],
        [EmbeddingContent(())],
        [EmbeddingContent([])],
        [row(("text", None), ("text", " \t"))],
        [row(("text", 3))],
        [EmbeddingContent((EmbeddingPart(EmbeddingPartSpec("image", "image/png"), b"data"),))],
        [row(("image", "file.png"))],
        [row(("image", np.zeros((3, 4, 3), np.float32)))],
        [row(("audio", audio(160)))],
        [row(("audio", audio(480001)))],
        [row(("audio", AudioClip(np.zeros((161, 1), np.float32), True)))],
        [row(("audio", AudioClip(np.full((161, 1), np.nan, np.float32), 16000)))],
        [row(("video", video(0)))],
        [row(("video", video(13)))],
        [row(*[("text", "x")] * 17)],
        [row(("text", "\ud800"))],
    ],
)
def test_whole_raw_batch_rejected_before_processor_or_forward(media_sdk, bad):
    encoder = descriptor().instantiate()
    try:
        with pytest.raises(EmbeddingConfigurationError):
            encoder.embed_multimodal(bad)
        assert media_sdk.processor_calls == media_sdk.model_calls == []
        with pytest.raises(EmbeddingConfigurationError):
            encoder.embed_multimodal([row(("text", "valid")), row(("audio", audio(160)))])
        assert media_sdk.processor_calls == media_sdk.model_calls == []
    finally:
        encoder.close()


@pytest.mark.parametrize(
    "token", ["<|image|>", "<|video|>", "<|audio|>", "<|image>", "<image|>", "<|audio>", "<audio|>"]
)
def test_reserved_markers_including_split_across_text_parts_are_safe(media_sdk, token):
    encoder = descriptor().instantiate()
    try:
        for parts in [(("text", "private:" + token),), (("text", token[:3]), ("image", image()), ("text", token[3:]))]:
            with pytest.raises(EmbeddingConfigurationError, match="reserved") as error:
                encoder.embed_multimodal([row(*parts)])
            assert "private" not in str(error.value) and token not in str(error.value)
        assert media_sdk.model_calls == media_sdk.processor_calls == []
    finally:
        encoder.close()


def test_combined_budget_counts_every_media_and_utf8_byte(media_sdk, monkeypatch):
    large = np.broadcast_to(np.zeros((1, 1, 3), np.uint8), (4096, 4096, 3))
    encoder = descriptor().instantiate()
    try:
        with pytest.raises(EmbeddingConfigurationError, match="combined"):
            encoder.embed_multimodal([row(("image", large), ("image", large))])
        monkeypatch.setattr(
            mlx.MLXMultimodalEmbedderDescriptor,
            "get_input_spec",
            lambda self: replace(descriptor_spec, max_input_bytes=6),
        )
        with pytest.raises(EmbeddingConfigurationError, match="combined"):
            encoder.embed_multimodal([row(("text", "视频x"))])
        assert media_sdk.model_calls == media_sdk.processor_calls == []
    finally:
        encoder.close()


descriptor_spec = descriptor().get_input_spec()


def test_arrow_and_direct_optional_text_have_identical_omission_and_order(media_sdk):
    parts = (EmbeddingPartSpec("text"),) * 3
    table = pa.table({"part_0": ["  first ", None], "part_1": ["\t", "last"], "part_2": [" second", None]})
    normalized = content_from_columns(table, parts, descriptor_spec)
    direct = [
        row(("text", "  first "), ("text", "\t"), ("text", " second")),
        row(("text", None), ("text", "last"), ("text", None)),
    ]
    encoder = descriptor().instantiate()
    try:
        encoder.embed_multimodal(normalized)
        encoder.embed_multimodal(direct)
        assert (
            media_sdk.processor_calls[0]["text"]
            == media_sdk.processor_calls[1]["text"]
            == ["title: none | text:   first  second", "title: none | text: last"]
        )
    finally:
        encoder.close()
    wrapper = _EmbedMultimodalBatch(descriptor(), "parts", "embedding", 768, parts=(parts[0],), on_error="ignore")
    for value in (None, "", " \t"):
        with pytest.raises(EmbeddingConfigurationError, match="no nonempty"):
            wrapper(pa.table({"parts": pa.table({"part_0": pa.array([value], pa.string())}).to_struct_array()}))
        assert wrapper._embedder is None


@pytest.mark.parametrize(
    "case",
    [
        "unknown",
        "missing",
        "not_mapping",
        "ids_float",
        "ids_negative",
        "ids_rows",
        "mask_shape",
        "mask_float",
        "mask_binary",
        "mask_empty",
        "mask_hole",
        "width_zero",
        "image_items",
        "pixel_half",
        "pixel_nan",
        "position_shape",
        "position_pair",
        "grid_duplicate",
        "grid_missing",
        "grid_pooling",
        "grid_padding_hole",
        "counts_swap",
        "lost_frame",
        "frame_half",
        "audio_half",
        "audio_nan",
        "audio_inf",
        "audio_mask_hole",
        "audio_truncated",
        "audio_items",
        "row_tokens_swap",
        "part_order",
        "masked_media",
        "overlength",
    ],
)
def test_malicious_joint_processor_outputs_never_reach_forward(media_sdk, case):
    rows = [
        row(("image", image()), ("audio", audio()), ("video", video(1))),
        row(("video", video(2)), ("audio", audio(1281)), ("image", image(6))),
    ]

    def change(data):
        if case == "unknown":
            data["surprise"] = np.zeros(1)
        elif case == "missing":
            data.pop("input_features")
        elif case == "not_mapping":
            return []
        elif case == "ids_float":
            data["input_ids"] = data["input_ids"].astype(float)
        elif case == "ids_negative":
            data["input_ids"][0, 0] = -1
        elif case == "ids_rows":
            data["input_ids"] = data["input_ids"][:1]
        elif case == "mask_shape":
            data["attention_mask"] = data["attention_mask"][:, :-1]
        elif case == "mask_float":
            data["attention_mask"] = data["attention_mask"].astype(float)
        elif case == "mask_binary":
            data["attention_mask"][0, 0] = 2
        elif case == "mask_empty":
            data["attention_mask"][0] = 0
        elif case == "mask_hole":
            data["attention_mask"][1, 1] = 0
        elif case == "width_zero":
            data["input_ids"] = data["input_ids"][:, :0]
            data["attention_mask"] = data["attention_mask"][:, :0]
        elif case == "image_items":
            data["pixel_values"] = data["pixel_values"][:1]
            data["image_position_ids"] = data["image_position_ids"][:1]
        elif case == "pixel_half":
            data["pixel_values"] = data["pixel_values"].astype(np.float16)
        elif case == "pixel_nan":
            data["pixel_values"][0, 0, 0] = np.nan
        elif case == "position_shape":
            data["image_position_ids"] = data["image_position_ids"][:, :, :1]
        elif case == "position_pair":
            data["image_position_ids"][0, 0] = [-1, 0]
        elif case == "grid_duplicate":
            data["image_position_ids"][0, 1] = [0, 0]
        elif case == "grid_missing":
            data["image_position_ids"][0, 8] = [-1, -1]
        elif case == "grid_pooling":
            data["image_position_ids"][0, 1:] = [-1, -1]
        elif case == "grid_padding_hole":
            data["image_position_ids"][0, [0, 9]] = data["image_position_ids"][0, [9, 0]]
        elif case == "counts_swap":
            data["num_frames_per_video"] = np.array([2, 1])
        elif case == "lost_frame":
            data["pixel_values_videos"] = data["pixel_values_videos"][:-1]
        elif case == "frame_half":
            data["pixel_values_videos"] = data["pixel_values_videos"].astype(np.float16)
        elif case in {"audio_half", "audio_nan", "audio_inf"}:
            if case == "audio_half":
                data["input_features"] = data["input_features"].astype(np.float16)
            else:
                data["input_features"][0, 0, 0] = np.nan if case == "audio_nan" else np.inf
        elif case == "audio_mask_hole":
            data["input_features_mask"][1, 1] = False
        elif case == "audio_truncated":
            data["input_features"] = data["input_features"][:, :-1]
            data["input_features_mask"] = data["input_features_mask"][:, :-1]
        elif case == "audio_items":
            data["input_features"] = data["input_features"][:1]
        elif case == "row_tokens_swap":
            data["input_ids"] = data["input_ids"][[1, 0]]
            data["attention_mask"] = data["attention_mask"][[1, 0]]
        elif case == "part_order":
            values = data["input_ids"][0]
            values[1:4], values[4:7] = values[4:7].copy(), values[1:4].copy()
        elif case == "masked_media":
            data["attention_mask"][0, 2] = 0
        return data

    media_sdk.change = change
    encoder = descriptor(options={"max_length": 1} if case == "overlength" else {}).instantiate()
    try:
        with pytest.raises(EmbeddingConfigurationError):
            encoder.embed_multimodal(rows)
        assert len(media_sdk.processor_calls) == 1 and media_sdk.model_calls == []
    finally:
        encoder.close()


@pytest.mark.parametrize("case", ["batch", "dimensions", "nan", "inf", "zero"])
def test_joint_reuses_native_output_validation(media_sdk, case):
    value = np.zeros((1, 768), np.float32)
    value[0, 0] = 1
    if case == "batch":
        value = np.concatenate([value, value])
    elif case == "dimensions":
        value = value[:, :767]
    elif case in {"nan", "inf"}:
        value[0, 0] = np.nan if case == "nan" else np.inf
    else:
        value[:] = 0
    media_sdk.output = value
    encoder = descriptor().instantiate()
    try:
        with pytest.raises(ValueError):
            encoder.embed_multimodal([row(("text", "hello"))])
    finally:
        encoder.close()


@pytest.mark.parametrize("kind", ["text", "image", "audio", "video"])
def test_unrequested_processor_tensor_family_is_rejected(media_sdk, kind):
    values = {"text": "document", "image": image(), "audio": audio(), "video": video(1)}

    def extra(data):
        data["unknown_family"] = np.ones(1, np.float32)
        return data

    media_sdk.change = extra
    encoder = descriptor().instantiate()
    try:
        with pytest.raises(EmbeddingConfigurationError, match="unexpected"):
            encoder.embed_multimodal([row((kind, values[kind]))])
        assert media_sdk.model_calls == []
    finally:
        encoder.close()


@pytest.mark.parametrize("field", ["vision", "audio", "projection", "vocabulary", "audio_processor", "video_processor"])
def test_joint_requires_both_towers_and_fixed_live_processor(media_sdk, field):
    if field == "vision":
        media_sdk.model.vision_tower = None
    elif field == "audio":
        media_sdk.model.audio_tower = None
    elif field == "projection":
        media_sdk.weights.pop("embed_audio.embedding_projection.weight")
    elif field == "vocabulary":
        media_sdk.processor.boi_token = "changed"
    elif field == "audio_processor":
        media_sdk.processor.feature_extractor.hop_length = 80
    else:
        media_sdk.processor.video_processor.add_timestamps = True
    with pytest.raises(EmbeddingConfigurationError):
        descriptor().instantiate()
    assert media_sdk.clears == 1 and media_sdk.model_calls == []


@pytest.mark.parametrize("policy", ["raise", "ignore"])
def test_joint_fatal_metal_allocation_is_not_converted_to_null(media_sdk, policy):
    media_sdk.failures["forward"] = RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory")
    wrapper = _EmbedMultimodalBatch(
        descriptor(), "parts", "embedding", 768, parts=(EmbeddingPartSpec("text"),), on_error=policy
    )
    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        with pytest.raises(MemoryError, match="exhausted model or host resources"):
            wrapper(pa.table({"parts": pa.table({"part_0": ["document"]}).to_struct_array()}))
        assert len(media_sdk.model_calls) == 1
    finally:
        wrapper.close()
        loop.close()


@pytest.mark.parametrize("policy", ["raise", "ignore"])
@pytest.mark.parametrize("case", ["batch", "dimensions", "nan", "inf", "zero"])
def test_joint_output_contract_is_fatal_under_both_public_policies(media_sdk, policy, case):
    value = np.eye(1, 768, dtype=np.float32)
    if case == "batch":
        value = np.concatenate([value, value])
    elif case == "dimensions":
        value = value[:, :767]
    elif case in {"nan", "inf"}:
        value[0, 0] = np.nan if case == "nan" else np.inf
    else:
        value[:] = 0
    media_sdk.output = value
    wrapper = _EmbedMultimodalBatch(
        descriptor(), "parts", "embedding", 768, parts=(EmbeddingPartSpec("text"),), on_error=policy
    )
    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        with pytest.raises(EmbeddingConfigurationError, match="joint model outputs"):
            wrapper(pa.table({"parts": pa.table({"part_0": ["document"]}).to_struct_array()}))
        assert len(media_sdk.model_calls) == 1
    finally:
        wrapper.close()
        loop.close()


@pytest.mark.parametrize("policy", ["raise", "ignore"])
@pytest.mark.parametrize("case", ["ragged", "array_error"])
def test_joint_processor_conversion_failure_is_fatal_without_retry_or_null(media_sdk, policy, case):
    class InvalidArray:
        def __array__(self, *args, **kwargs):
            raise TypeError("private-processor-value")

    def change(data):
        data["input_ids"] = [[2, 1], [2]] if case == "ragged" else InvalidArray()
        return data

    media_sdk.change = change
    wrapper = _EmbedMultimodalBatch(
        descriptor(), "parts", "embedding", 768, parts=(EmbeddingPartSpec("text"),), on_error=policy
    )
    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        with pytest.raises(EmbeddingConfigurationError, match="joint processor returned invalid array data") as error:
            wrapper(pa.table({"parts": pa.table({"part_0": ["first", "second"]}).to_struct_array()}))
        assert "private-processor-value" not in str(error.value)
        assert len(media_sdk.processor_calls) == 1 and media_sdk.model_calls == []
    finally:
        wrapper.close()
        loop.close()
