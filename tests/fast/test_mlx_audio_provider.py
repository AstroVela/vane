# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""MLX audio/video planning contracts require no optional SDK or model I/O."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from vane.ai._audio_embedding import AudioClip
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai.functions import _prepare_embed_call
from vane.ai.provider import load_provider

pytest_plugins = ("mlx_sdk_fixture",)


@pytest.mark.parametrize("kind", ["audio", "video"])
def test_media_metadata_is_lazy_serializable_and_paired(kind):
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import builtins, pickle, sys
original = builtins.__import__
def checked(name, *args, **kwargs):
    assert name.split('.')[0] not in {'mlx', 'mlx_vlm', 'transformers', 'huggingface_hub', 'PIL'}
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from vane.ai.provider import load_provider
p = load_provider('mlx')
kind = sys.argv[1]
d = pickle.loads(pickle.dumps(getattr(p, 'get_' + kind + '_embedder')()))
assert d.get_provider() == 'mlx' and d.get_dimensions() == 768
assert d.get_udf_options().num_gpus == 0 and not d.is_async()
assert 'input_type' not in d.get_options()
spec = d.get_output_spec()
assert spec.layout == 'vector' and spec.sql_type() == 'FLOAT[768]'
assert spec.outputs[0].source_modalities == (kind,)
assert spec.outputs[0].space == p.get_text_embedder().get_output_spec().outputs[0].space
assert spec.outputs[0].space == p.get_image_embedder().get_output_spec().outputs[0].space
if kind == 'audio':
    spec = d.get_input_spec()
    assert (spec.sample_rate, spec.max_samples, spec.max_channels, spec.max_input_bytes) == (16000, 480000, 1, 4194304)
else:
    spec = d.get_input_spec()
    assert (spec.frame_count, spec.max_frames, spec.max_input_bytes) == (None, 12, 67108864)
    assert d.supports_image_queries()
""",
            kind,
        ],
        check=True,
    )


@pytest.mark.parametrize("kind", ["audio", "video"])
def test_media_options_and_resources_do_not_inherit_text_defaults(kind):
    source = {"max_length": 512, "revision": "A" * 40}
    descriptor = getattr(load_provider("mlx"), f"get_{kind}_embedder")(options=source)
    expected = descriptor.get_options()
    assert expected["revision"] == "a" * 40 and "input_type" not in expected
    assert source == {"max_length": 512, "revision": "A" * 40}
    descriptor.get_options()["max_length"] = 1
    assert descriptor.get_options() == expected
    for provider in ("mlx", load_provider("mlx")):
        resolved, dimensions, resources, *_, backend = _prepare_embed_call(
            provider,
            None,
            None,
            "raise",
            source | {"normalize": True, "batch_size": 2},
            relation=True,
            input_kind=kind,
        )
        assert resolved.get_options() == expected and dimensions == 768
        assert resources.actor_number == 1 and resources.num_gpus == 0 and backend == "subprocess_actor"
    assert load_provider("mlx").get_text_embedder().get_options()["input_type"] == "document"


@pytest.mark.parametrize("kind", ["audio", "video"])
@pytest.mark.parametrize(
    "options",
    [
        {"input_type": "document"},
        {"input_type": "query"},
        {"dtype": "float16"},
        {"dtype": "float32"},
        {"device": "cpu"},
        {"device": "cuda"},
        {"revision": "main"},
        {"revision": "g" * 40},
        {"trust_remote_code": True},
        {"local_files_only": 1},
        {"max_length": True},
        {"max_length": 0},
        {"max_length": 2049},
        {"max_concurrency_per_actor": 2},
        {"max_concurrency_per_actor": True},
        {"overlength": "truncate"},
        {"processor_kwargs": {}},
        {"sample_rate": 16000},
        {"max_frames": 8},
        {"fps": 1},
        {"max_chunk_chars": 100},
        {"api_key": "private"},
    ],
)
def test_invalid_media_options_fail_before_model_io(kind, options):
    with pytest.raises((ValueError, TypeError)):
        getattr(load_provider("mlx"), f"get_{kind}_embedder")(options=options)
    with pytest.raises((ValueError, TypeError)):
        _prepare_embed_call("mlx", None, None, "raise", options, relation=True, input_kind=kind)


@pytest.mark.parametrize("kind", ["audio", "video"])
@pytest.mark.parametrize(
    "options",
    [
        {"actor_number": 2},
        {"actor_number": True},
        {"actor_number": 1.0},
        {"execution_backend": "ray_actor"},
        {"execution_backend": "subprocess_task"},
        {"execution_backend": "ray_task"},
        {"max_chunk_chars": None},
        {"chunk_overlap_chars": 0},
    ],
)
def test_media_execution_policy_is_explicit(kind, options):
    with pytest.raises((ValueError, TypeError)):
        _prepare_embed_call("mlx", None, None, "raise", options, relation=True, input_kind=kind)


@pytest.mark.parametrize("kind", ["audio", "video"])
@pytest.mark.parametrize("dimensions", [True, "768", 512, 769, 0])
def test_media_dimensions_are_fixed(kind, dimensions):
    factory = getattr(load_provider("mlx"), f"get_{kind}_embedder")
    with pytest.raises(ValueError):
        factory(dimensions=dimensions)
    with pytest.raises(ValueError):
        factory().get_output_spec(dimensions=dimensions)


@pytest.mark.parametrize("kind", ["audio", "video"])
def test_media_model_and_direct_execution_options_are_rejected(kind):
    factory = getattr(load_provider("mlx"), f"get_{kind}_embedder")
    with pytest.raises(ValueError):
        factory(model="other/model")
    for key in ("actor_number", "batch_size", "execution_backend", "normalize", "max_retries"):
        with pytest.raises(TypeError):
            factory(options={key: 1})


def audio_clip(count=321, dtype=np.float32, rate=16000):
    return AudioClip(np.zeros((count, 1), dtype=dtype), rate)


def test_audio_preserves_pcm_order_silence_and_independent_float32_storage(media_sdk):
    data = (np.arange(642, dtype=np.float64) / 1000).reshape(642, 1)[::2]
    assert not data.flags.c_contiguous
    data.setflags(write=False)
    original = data.copy()
    clips = [AudioClip(data, 16000), audio_clip(641), AudioClip(data, 16000)]
    encoder = load_provider("mlx").get_audio_embedder().instantiate()
    try:
        result = np.asarray(encoder.embed_audio(clips))
        assert result.shape == (3, 768) and result.dtype == np.float32
        assert len(media_sdk.model_calls) == 1 and media_sdk.loads == media_sdk.processor_loads == 1
        sent = media_sdk.processor_calls[0]["audio"]
        for source, row in zip(clips, sent):
            np.testing.assert_array_equal(row[0], source.samples[:, 0].astype(np.float32))
            assert not np.shares_memory(row[0], source.samples)
        np.testing.assert_array_equal(data, original)
        np.testing.assert_array_equal(sent[0][0], sent[2][0])
        inputs = media_sdk.model_calls[0]
        np.testing.assert_array_equal(inputs["input_features_mask"], [[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0]])
        assert set(inputs) == {"input_ids", "attention_mask", "input_features", "input_features_mask"}
        encoder.embed_audio([audio_clip(161)])
        assert media_sdk.loads == media_sdk.processor_loads == 1 and len(media_sdk.model_calls) == 2
        calls = len(media_sdk.processor_calls)
        assert encoder.embed_audio([]) == [] and len(media_sdk.processor_calls) == calls
    finally:
        encoder.close()
    assert encoder._model is encoder._processor is encoder._tokenizer is None
    assert media_sdk.clears == 1
    encoder.close()
    assert media_sdk.clears == 1
    with pytest.raises(RuntimeError, match="closed"):
        encoder.embed_audio([])


@pytest.mark.parametrize(
    "case",
    [
        "not_list",
        "not_clip",
        "zero",
        "one",
        "160",
        "long",
        "stereo",
        "channel_first",
        "one_dim",
        "int16",
        "float16",
        "nan",
        "inf",
        "high",
        "low",
        "rate",
        "bool_rate",
        "float_rate",
        "none_samples",
    ],
)
def test_audio_input_contract_rejected_before_processing(media_sdk, case):
    clips = [audio_clip()]
    if case == "not_list":
        clips = tuple(clips)
    elif case == "not_clip":
        clips = ["audio.wav"]
    elif case in {"zero", "one", "160", "long"}:
        clips = [audio_clip({"zero": 0, "one": 1, "160": 160, "long": 480001}[case])]
    else:
        samples = np.zeros((321, 1), dtype=np.float32)
        rate = 16000
        if case == "stereo":
            samples = np.zeros((321, 2), dtype=np.float32)
        elif case == "channel_first":
            samples = samples.T
        elif case == "one_dim":
            samples = samples[:, 0]
        elif case in {"int16", "float16"}:
            samples = samples.astype(case)
        elif case in {"nan", "inf", "high", "low"}:
            samples[0, 0] = {"nan": np.nan, "inf": np.inf, "high": 1.01, "low": -1.01}[case]
        elif case == "none_samples":
            samples = None
        else:
            rate = {"rate": 48000, "bool_rate": True, "float_rate": 16000.0}[case]
        clips = [audio_clip(), AudioClip(samples, rate)]
    encoder = load_provider("mlx").get_audio_embedder().instantiate()
    try:
        with pytest.raises(EmbeddingConfigurationError):
            encoder.embed_audio(clips)
        assert media_sdk.processor_calls == media_sdk.model_calls == []
    finally:
        encoder.close()


def test_audio_byte_budget_is_checked_before_float64_conversion(media_sdk, monkeypatch):
    monkeypatch.setattr("vane.ai.providers.mlx.MAX_AUDIO_INPUT_BYTES", 321 * 4)
    encoder = load_provider("mlx").get_audio_embedder().instantiate()
    try:
        with pytest.raises(EmbeddingConfigurationError, match="budget"):
            encoder.embed_audio([audio_clip(dtype=np.float64)])
        assert media_sdk.processor_calls == media_sdk.model_calls == []
    finally:
        encoder.close()


@pytest.mark.parametrize(
    "case",
    [
        "unknown",
        "missing",
        "not_mapping",
        "features_rank",
        "features_width",
        "features_rows",
        "features_fp16",
        "features_nan",
        "truncated_features",
        "mask_all_one",
        "mask_hole",
        "mask_float",
        "mask_two",
        "mask_zero",
        "mask_shape",
        "ids_float",
        "ids_negative",
        "ids_bool",
        "ids_rows",
        "attention_float",
        "attention_shape",
        "attention_two",
        "attention_zero",
        "missing_audio_token",
        "masked_audio_token",
    ],
)
def test_audio_processor_corruption_is_fatal_before_forward(media_sdk, case):
    def corrupt(inputs):
        if case == "not_mapping":
            return None
        if case == "unknown":
            inputs["metadata"] = object()
        elif case == "missing":
            inputs.pop("input_features_mask")
        elif case == "features_rank":
            inputs["input_features"] = inputs["input_features"][:, :, :, None]
        elif case == "features_width":
            inputs["input_features"] = inputs["input_features"][:, :, :127]
        elif case == "features_rows":
            inputs["input_features"] = inputs["input_features"][:1]
        elif case == "features_fp16":
            inputs["input_features"] = inputs["input_features"].astype(np.float16)
        elif case == "features_nan":
            inputs["input_features"][0, 0, 0] = np.nan
        elif case == "truncated_features":
            inputs["input_features"] = inputs["input_features"][:, :2]
            inputs["input_features_mask"] = inputs["input_features_mask"][:, :2]
        elif case.startswith("mask_"):
            mask = inputs["input_features_mask"]
            if case == "mask_all_one":
                inputs["input_features_mask"] = np.ones_like(mask)
            elif case == "mask_hole":
                mask[1] = [1, 0, 1, 1]
            elif case == "mask_float":
                inputs["input_features_mask"] = mask.astype(np.float32)
            elif case == "mask_shape":
                inputs["input_features_mask"] = mask[:, :1]
            elif case == "mask_zero":
                mask[:] = False
            else:
                inputs["input_features_mask"] = mask.astype(np.int64) * 2
        elif case.startswith("ids_"):
            ids = inputs["input_ids"]
            if case == "ids_rows":
                inputs["input_ids"] = ids[:1]
            elif case == "ids_negative":
                ids[0, 0] = -1
            else:
                inputs["input_ids"] = ids.astype(np.float64 if case == "ids_float" else bool)
        elif case.startswith("attention_"):
            mask = inputs["attention_mask"]
            if case == "attention_shape":
                inputs["attention_mask"] = mask[:, :1]
            elif case == "attention_float":
                inputs["attention_mask"] = mask.astype(np.float32)
            else:
                mask[:] = 2 if case == "attention_two" else 0
        elif case == "missing_audio_token":
            inputs["input_ids"][0, 1] = 3
        else:
            inputs["attention_mask"][0, 1] = 0
        return inputs

    media_sdk.change = corrupt
    encoder = load_provider("mlx").get_audio_embedder().instantiate()
    try:
        with pytest.raises(EmbeddingConfigurationError):
            encoder.embed_audio([audio_clip(), audio_clip(641)])
        assert media_sdk.model_calls == []
    finally:
        encoder.close()


def test_audio_expanded_tokens_are_not_truncated(media_sdk):
    encoder = load_provider("mlx").get_audio_embedder(options={"max_length": 2}).instantiate()
    try:
        with pytest.raises(EmbeddingConfigurationError, match="max_length"):
            encoder.embed_audio([audio_clip()])
        assert len(media_sdk.processor_calls) == 1 and media_sdk.model_calls == []
    finally:
        encoder.close()


@pytest.mark.parametrize("case", ["rows", "dimensions", "nan", "inf", "zero", "norm"])
def test_audio_validates_native_output(media_sdk, case):
    output = np.eye(2, 768, dtype=np.float32)
    if case == "rows":
        output = output[:1]
    elif case == "dimensions":
        output = output[:, :767]
    else:
        output[0, 0] = {"nan": np.nan, "inf": np.inf, "zero": 0, "norm": 2}[case]
    media_sdk.output = output
    encoder = load_provider("mlx").get_audio_embedder().instantiate()
    try:
        with pytest.raises(ValueError):
            encoder.embed_audio([audio_clip(), audio_clip(641)])
    finally:
        encoder.close()


@pytest.mark.parametrize("phase", ["processor", "forward", "evaluation"])
def test_audio_memory_errors_are_not_replaced(media_sdk, phase):
    failure = MemoryError("private waveform path")
    media_sdk.failures[phase] = failure
    encoder = load_provider("mlx").get_audio_embedder().instantiate()
    try:
        with pytest.raises(MemoryError) as caught:
            encoder.embed_audio([audio_clip()])
        assert caught.value is failure
    finally:
        encoder.close()


@pytest.mark.parametrize(
    "field", ["audio_config", "audio_token_id", "feature_extractor", "runtime_extractor", "audio_tower", "weights"]
)
def test_audio_initialization_validates_checkpoint_and_cleans_up(media_sdk, field):
    import json

    if field in {"audio_config", "audio_token_id"}:
        path = media_sdk.path / "config.json"
        config = json.loads(path.read_text())
        config[field] = {} if field == "audio_config" else 1
        path.write_text(json.dumps(config))
    elif field == "feature_extractor":
        path = media_sdk.path / "processor_config.json"
        config = json.loads(path.read_text())
        config["feature_extractor"]["hop_length"] = 128
        path.write_text(json.dumps(config))
    elif field == "runtime_extractor":
        media_sdk.processor.feature_extractor.sampling_rate = 48000
    elif field == "audio_tower":
        media_sdk.model.audio_tower = None
    else:
        del media_sdk.weights["embed_audio.embedding_projection.weight"]
    with pytest.raises(EmbeddingConfigurationError):
        load_provider("mlx").get_audio_embedder().instantiate()
    assert media_sdk.clears == 1


@pytest.mark.parametrize("policy", ["raise", "ignore"])
@pytest.mark.parametrize("case", ["short", "processor", "allocation"])
def test_audio_worker_contract_and_allocation_failures_are_fatal_under_both_policies(media_sdk, policy, case):
    import asyncio

    import pyarrow as pa

    import vane
    from vane.ai.functions import _EmbedAudioBatch

    count = 160 if case == "short" else 321
    samples = vane.tensor_array(
        [np.zeros((count, 1), dtype=np.float32)], vane.tensor_type(vane.sqltypes.FLOAT, (None, 1))
    )
    table = pa.table(
        {"audio": pa.StructArray.from_arrays([pa.array([16000], pa.int32()), samples], names=["sample_rate", "data"])}
    )
    if case == "processor":

        def corrupted(inputs):
            inputs["input_features_mask"][:] = False
            return inputs

        media_sdk.change = corrupted
    elif case == "allocation":
        media_sdk.failures["forward"] = MemoryError("private audio path")
    wrapper = _EmbedAudioBatch(
        load_provider("mlx").get_audio_embedder(), "audio", "embedding", 768, on_error=policy, max_retries=0
    )
    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        with pytest.raises(MemoryError if case == "allocation" else EmbeddingConfigurationError) as caught:
            wrapper(table)
        assert "private audio path" not in str(caught.value)
        assert len(media_sdk.model_calls) == (1 if case == "allocation" else 0)
    finally:
        wrapper.close()
        loop.close()


def test_audio_initialization_keeps_primary_error_and_cleanup_cause(media_sdk):
    primary, cleanup = RuntimeError("primary"), ValueError("cleanup")
    media_sdk.failures.update(processor_load=primary, synchronize=cleanup)
    with pytest.raises(RuntimeError) as caught:
        load_provider("mlx").get_audio_embedder().instantiate()
    assert caught.value is primary and primary.cleanup_errors == (cleanup,)
    assert media_sdk.clears == 1


def test_audio_async_close_is_idempotent_and_releases_refs(media_sdk):
    import asyncio

    encoder = load_provider("mlx").get_audio_embedder().instantiate()
    asyncio.run(encoder.aclose())
    asyncio.run(encoder.aclose())
    assert encoder._model is encoder._processor is encoder._tokenizer is None
    assert media_sdk.clears == 1
