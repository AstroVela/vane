# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""ONNX contracts, complete graph routing and explicit resource ownership."""

import pickle
import sys

import numpy as np
import pytest

from tests.onnx_fixture import install_sdk, write_bundle
from vane.ai import AudioClip, EmbeddingContent, EmbeddingPart, EmbeddingPartSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._onnx_options import ONNX_MODEL
from vane.ai._video_embedding import VideoClip
from vane.ai.functions import _prepare_embed_call
from vane.ai.provider import ProviderImportError, _ProviderResultError, load_provider
from vane.ai.providers._onnx_artifacts import validate_external_data


@pytest.fixture
def options(tmp_path):
    return write_bundle(tmp_path / "model")


@pytest.fixture
def sdk(monkeypatch):
    return install_sdk(monkeypatch)


@pytest.mark.parametrize("kind", ["text", "image", "audio", "video", "multimodal"])
@pytest.mark.parametrize("dimensions", [128, 256, 512, 768])
def test_planning_is_serializable_sdk_free_and_does_not_open_model_files(monkeypatch, tmp_path, kind, dimensions):
    for name in ("onnxruntime", "onnx", "transformers", "torch"):
        monkeypatch.setitem(sys.modules, name, None)
    descriptor = getattr(load_provider("onnx"), f"get_{kind}_embedder")(
        dimensions=dimensions, options={"model_path": str(tmp_path / "not-provisioned"), "model_sha256": "a" * 64}
    )
    restored = pickle.loads(pickle.dumps(descriptor))
    assert restored.get_model() == ONNX_MODEL
    assert restored.get_provider() == "onnx"
    assert restored.get_output_spec() == descriptor.get_output_spec()
    assert restored.get_output_spec().outputs[0].space == ONNX_MODEL + ":onnx-py2:" + "a" * 64
    assert restored.get_dimensions() == dimensions
    assert restored.get_udf_options().num_gpus == 0


@pytest.mark.parametrize(
    "update",
    [
        {"model_path": "relative"},
        {"model_path": "https://host/model"},
        {"model_sha256": "bad"},
        {"dtype": "int3"},
        {"backend": "auto"},
        {"max_length": True},
        {"max_length": 8193},
        {"max_input_bytes": 0},
        {"num_threads": False},
        {"num_threads": 257},
        {"device_id": 1},
        {"input_type": "invalid"},
        {"runtime": "http"},
        {"trust_remote_code": True},
    ],
)
def test_invalid_native_options_fail_before_model_load(options, update):
    with pytest.raises((EmbeddingConfigurationError, TypeError)):
        load_provider("onnx").get_text_embedder(options=options | update)


@pytest.mark.parametrize("dimensions", [True, 0, 129, 769])
def test_output_widths_are_closed(options, dimensions):
    with pytest.raises(EmbeddingConfigurationError):
        load_provider("onnx").get_text_embedder(dimensions=dimensions, options=options)


def test_public_preparation_uses_explicit_backend_resources_and_never_retries(options):
    for backend, gpus in (("cpu", 0), ("cuda", 1), ("coreml", 0)):
        prepared = _prepare_embed_call("onnx", None, 128, "raise", options | {"backend": backend}, relation=True)
        assert prepared[2].num_gpus == gpus
        assert prepared[2].max_retries == 0 and prepared[-1] == "subprocess_actor"
    for unsupported in ({"max_retries": 1}, {"max_chunk_chars": 100}, {"chunk_overlap_chars": 0}):
        with pytest.raises(EmbeddingConfigurationError):
            _prepare_embed_call("onnx", None, 128, "ignore", options | unsupported, relation=True)


@pytest.mark.parametrize(
    "filename",
    [
        "config.json",
        "tokenizer.json",
        "onnx/model.onnx",
        "onnx/vision_encoder.onnx_data",
        "onnx/audio_encoder.onnx_data",
    ],
)
def test_bundle_tampering_is_rejected_before_any_session_starts(options, sdk, filename):
    from pathlib import Path

    path = Path(options["model_path"]) / filename
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(EmbeddingConfigurationError, match="SHA-256"):
        load_provider("onnx").get_text_embedder(options=options).instantiate()
    assert not sdk.events


def test_external_data_cannot_escape_the_fingerprinted_weight_file():
    from types import SimpleNamespace

    for location in ("../outside", "/private/weights", "other.onnx_data"):
        tensor = SimpleNamespace(
            DESCRIPTOR=SimpleNamespace(full_name="onnx.TensorProto"),
            external_data=[SimpleNamespace(key="location", value=location)],
            ListFields=lambda: [],
        )
        with pytest.raises(EmbeddingConfigurationError, match="fingerprinted sibling"):
            validate_external_data(tensor, "model.onnx_data")


def test_missing_sdk_has_the_safe_optional_dependency_hint(options, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ProviderImportError) as error:
        load_provider("onnx").get_text_embedder(options=options).instantiate()
    assert error.value.extra == "onnx"


def test_processor_drift_is_rejected_before_session_creation(options, sdk, monkeypatch):
    from vane.ai.providers import _onnx_model

    monkeypatch.setattr(
        _onnx_model, "processor_versions", lambda: _onnx_model.PROCESSOR_VERSIONS | {"torchvision": "0.30.0"}
    )
    with pytest.raises(EmbeddingConfigurationError, match="onnx-py2 processor"):
        load_provider("onnx").get_text_embedder(options=options).instantiate()
    assert not any(event[0] == "load" for event in sdk.events)


@pytest.mark.parametrize("encoding_version", [True, 0, 1, 3])
def test_service_encoding_version_is_closed(encoding_version):
    from vane.ai import _embedding_ipc as ipc

    with pytest.raises(EmbeddingConfigurationError):
        ipc.ONNXEncoding("a" * 64, 128, "document", encoding_version)


@pytest.mark.parametrize("backend", ["cuda", "coreml"])
def test_missing_execution_provider_never_starts_cpu_sessions(options, sdk, backend):
    sdk.available = ["CPUExecutionProvider"]
    with pytest.raises(EmbeddingConfigurationError, match="unavailable"):
        load_provider("onnx").get_text_embedder(options=options | {"backend": backend}).instantiate()
    assert not any(event[0] == "load" for event in sdk.events)


def test_backend_initialization_failure_never_retries_on_cpu(options, sdk):
    sdk.failure = RuntimeError("CUDA initialization failed")
    with pytest.raises(RuntimeError, match="CUDA initialization"):
        load_provider("onnx").get_text_embedder(options=options | {"backend": "cuda"}).instantiate()
    loads = [event for event in sdk.events if event[0] == "load"]
    assert len(loads) == 1 and loads[0][3:5] == ("CUDAExecutionProvider", False)
    assert loads[0][5] == {}


def test_cuda_execution_failure_propagates_without_recreating_sessions(options, sdk, monkeypatch):
    runtime = load_provider("onnx").get_text_embedder(options=options | {"backend": "cuda"}).instantiate()
    attempts = []
    error = RuntimeError("CUDA execution failed")

    def fail(*args, **kwargs):
        attempts.append(1)
        raise error

    monkeypatch.setattr(type(runtime.sessions["model"]), "run", fail)
    try:
        with pytest.raises(RuntimeError) as caught:
            runtime.embed_text(["query"])
        assert caught.value is error and attempts == [1]
        loads = [event for event in sdk.events if event[0] == "load"]
        assert len(loads) == 3 and all(event[3:5] == ("CUDAExecutionProvider", False) for event in loads)
    finally:
        # Drop captured frames before checking release of Python-owned sessions.
        error.__traceback__ = None
        del caught
        runtime.close()
    assert sdk.active == 0


def test_cuda_budget_and_reclamation_cover_every_graph_without_changing_inputs(options, sdk):
    provider = load_provider("onnx")
    runtime = provider.get_multimodal_embedder(options=options | {"backend": "cuda", "max_length": 512}).instantiate()
    image = np.full((8, 12, 3), 40, np.uint8)
    audio = AudioClip(np.full((1600, 1), 0.2, np.float32), 16000)
    video = VideoClip((image, image + 1), (0.0, 1.0), (0, 1))
    content = EmbeddingContent(
        tuple(
            EmbeddingPart(EmbeddingPartSpec(kind), value)
            for kind, value in (("text", "look"), ("image", image), ("video", video), ("audio", audio))
        )
    )
    try:
        first = runtime.embed_multimodal([content])[0]
        repeated = runtime.embed_multimodal([content])[0]
        np.testing.assert_array_equal(first, repeated)
        runtime.embed_text(["query"])
        assert len(sdk.cuda_queries) == 1
        assert len(sdk.provider_options) == 3
        for _, values in sdk.provider_options:
            assert values == {
                "device_id": 0,
                "use_tf32": 0,
                "gpu_mem_limit": 8 * 1024**3,
                "arena_extend_strategy": "kSameAsRequested",
            }
        assert {tower for tower, _ in sdk.run_options} == {"vision_encoder", "audio_encoder", "model"}
        assert all(values == {"memory.enable_memory_arena_shrinkage": "gpu:0"} for _, values in sdk.run_options)
        assert len([event for event in sdk.events if event[0] == "load"]) == 3
    finally:
        runtime.close()
    assert sdk.active == 0


@pytest.mark.parametrize("backend", ["cpu", "coreml"])
def test_non_cuda_graphs_do_not_probe_cuda_or_apply_gpu_run_options(options, sdk, backend):
    runtime = load_provider("onnx").get_text_embedder(options=options | {"backend": backend}).instantiate()
    try:
        runtime.embed_text(["query"])
        assert not sdk.cuda_queries
        assert all(values is None for _, values in sdk.run_options)
        assert all("gpu_mem_limit" not in values for _, values in sdk.provider_options)
    finally:
        runtime.close()


def test_cuda_budget_failure_starts_no_session_and_does_not_retry(options, sdk, monkeypatch):
    from vane.ai.providers import _onnx_cuda

    def exhausted():
        raise MemoryError("no CUDA headroom")

    monkeypatch.setattr(_onnx_cuda, "cuda_arena_limit", exhausted)
    with pytest.raises(MemoryError, match="no CUDA headroom"):
        load_provider("onnx").get_text_embedder(options=options | {"backend": "cuda"}).instantiate()
    assert not [event for event in sdk.events if event[0] == "load"]


def test_partial_initialization_releases_the_preceding_session(options, sdk):
    sdk.failure, sdk.fail_tower = RuntimeError("bad graph"), "vision_encoder"
    with pytest.raises(RuntimeError, match="bad graph"):
        load_provider("onnx").get_text_embedder(options=options).instantiate()
    assert sdk.active == 0


def test_roles_widths_duplicates_and_close(options, sdk):
    provider = load_provider("onnx")
    document = provider.get_text_embedder(dimensions=128, options=options).instantiate()
    query = provider.get_text_embedder(dimensions=128, options=options | {"input_type": "query"}).instantiate()
    try:
        docs = document.embed_text(["same", "same"])
        queries = query.embed_text(["same"])
        np.testing.assert_array_equal(docs[0], docs[1])
        assert not np.array_equal(docs[0], queries[0])
        assert all(value.shape == (128,) for value in [*docs, *queries])
        np.testing.assert_allclose([np.linalg.norm(value) for value in [*docs, *queries]], 1, atol=1e-6)
        prompts = [event[2]["text"][0] for event in sdk.events if event[0] == "process"]
        assert prompts == ["title: none | text: same", "title: none | text: same", "task: search result | query: same"]
    finally:
        document.close()
        query.close()
    assert sdk.active == 0
    with pytest.raises(RuntimeError, match="closed"):
        document.embed_text(["after close"])


def test_media_and_ordered_composition_route_through_encoders_then_text_graph(options, sdk):
    runtime = load_provider("onnx").get_multimodal_embedder(options=options | {"max_length": 512}).instantiate()
    image = np.full((8, 12, 3), 40, np.uint8)
    audio = AudioClip(np.full((1600, 1), 0.2, np.float32), 16000)
    video = VideoClip((image, image + 1), (0.0, 1.0), (0, 1))
    content = EmbeddingContent(
        tuple(
            EmbeddingPart(EmbeddingPartSpec(kind), value)
            for kind, value in (("text", "look"), ("image", image), ("audio", audio), ("video", video))
        )
    )
    try:
        values = runtime.embed_multimodal([content])
        assert len(values) == 1 and values[0].shape == (768,)
        runs = [event[1] for event in sdk.events if event[0] == "run"]
        assert runs == ["vision_encoder", "vision_encoder", "audio_encoder", "model"]
        processed = next(event[2] for event in sdk.events if event[0] == "process")
        assert processed["text"] == ["title: none | text: look<|image|><|audio|>00:00 <|video|> 00:01 <|video|>"]
        assert processed["videos_kwargs"] == {
            "device": "cpu",
            "input_data_format": "channels_last",
            "do_sample_frames": False,
            "add_timestamps": False,
        }
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "frame_times, expected",
    [
        ((0.0, 1.5, 9.75), "00:00 <|video|> 00:01 <|video|> 00:09 <|video|>"),
        ((0.2, 0.2, 61.99), "00:00 <|video|> 00:00 <|video|> 01:01 <|video|>"),
        ((127.25,), "02:07 <|video|>"),
    ],
)
def test_supplied_video_pts_reach_the_prompt_without_resampling(options, sdk, frame_times, expected):
    frames = tuple(np.full((8, 12, 3), index * 40, np.uint8) for index in range(len(frame_times)))
    video = VideoClip(frames, frame_times, tuple(range(40, 40 + len(frames))))
    runtime = load_provider("onnx").get_video_embedder(options=options).instantiate()
    try:
        runtime.embed_video([video])
        processed = next(event[2] for event in sdk.events if event[0] == "process")
        assert processed["text"] == [expected]
        actual = [frame for clip in processed["videos"] for frame in clip]
        assert len(actual) == len(frames)
        for supplied, received in zip(frames, actual, strict=True):
            np.testing.assert_array_equal(supplied, received)
        assert processed["videos_kwargs"]["do_sample_frames"] is False
        assert processed["videos_kwargs"]["add_timestamps"] is False
    finally:
        runtime.close()


@pytest.mark.parametrize("layout", ["horizontal_flip", "channel_reverse", "fortran"])
def test_video_frame_views_remain_supported_by_the_processor(options, sdk, layout):
    image = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    frame = (
        image[:, ::-1, :]
        if layout == "horizontal_flip"
        else image[:, :, ::-1]
        if layout == "channel_reverse"
        else np.asfortranarray(image)
    )
    runtime = load_provider("onnx").get_video_embedder(options=options).instantiate()
    try:
        runtime.embed_video([VideoClip((frame,), (1.25,), (4,))])
        processed = next(event[2] for event in sdk.events if event[0] == "process")
        received = processed["videos"][0]
        assert received.flags.writeable and all(stride >= 0 for stride in received.strides)
        assert not np.shares_memory(received, frame)
        np.testing.assert_array_equal(received, np.stack([frame]))
    finally:
        runtime.close()


def test_video_time_changes_affect_model_inputs_and_vectors(options, sdk):
    image = np.full((8, 12, 3), 40, np.uint8)
    frames = (image, image + 1)
    runtime = load_provider("onnx").get_video_embedder(options=options).instantiate()
    try:
        values = runtime.embed_video([VideoClip(frames, (0.0, 3.0), (0, 1)), VideoClip(frames, (0.0, 9.0), (0, 1))])
        prompts = [event[2]["text"][0] for event in sdk.events if event[0] == "process"]
        assert prompts == ["00:00 <|video|> 00:03 <|video|>", "00:00 <|video|> 00:09 <|video|>"]
        assert not np.array_equal(values[0], values[1])
    finally:
        runtime.close()


def test_multiple_videos_keep_actual_times_and_mixed_part_placement(options, sdk):
    image = np.full((8, 12, 3), 40, np.uint8)
    audio = AudioClip(np.full((1600, 1), 0.2, np.float32), 16000)
    first = VideoClip((image, image + 1), (5.25, 8.5), (50, 51))
    second = VideoClip((image + 2,), (62.75,), (125,))
    content = EmbeddingContent(
        tuple(
            EmbeddingPart(EmbeddingPartSpec(kind), value)
            for kind, value in (("video", first), ("audio", audio), ("text", "look"), ("video", second))
        )
    )
    runtime = load_provider("onnx").get_multimodal_embedder(options=options).instantiate()
    try:
        runtime.embed_multimodal([content])
        processed = next(event[2] for event in sdk.events if event[0] == "process")
        assert processed["text"] == ["00:05 <|video|> 00:08 <|video|><|audio|>title: none | text: look 01:02 <|video|>"]
        actual = [frame for clip in processed["videos"] for frame in clip]
        for supplied, received in zip((*first.frames, *second.frames), actual, strict=True):
            np.testing.assert_array_equal(supplied, received)
        np.testing.assert_array_equal(processed["audio"][0], audio.samples[:, 0])
    finally:
        runtime.close()


@pytest.mark.parametrize("empty_part", [False, True])
def test_timestamps_cannot_merge_with_adjacent_caller_text(options, sdk, empty_part):
    image = np.full((8, 12, 3), 40, np.uint8)

    def content(text, seconds):
        values = [("text", text)]
        if empty_part:
            values.append(("text", ""))
        values.append(("video", VideoClip((image,), (seconds,), (0,))))
        return EmbeddingContent(tuple(EmbeddingPart(EmbeddingPartSpec(kind), value) for kind, value in values))

    runtime = load_provider("onnx").get_multimodal_embedder(options=options).instantiate()
    try:
        values = runtime.embed_multimodal([content("event 1", 62.0), content("event ", 6062.0)])
        prompts = [event[2]["text"][0] for event in sdk.events if event[0] == "process"]
        assert prompts == ["title: none | text: event 1 01:02 <|video|>", "title: none | text: event 101:02 <|video|>"]
        assert not np.array_equal(values[0], values[1])
    finally:
        runtime.close()


def test_timestamp_prompt_bytes_are_checked_before_processing(options, sdk):
    image = np.full((8, 12, 3), 40, np.uint8)
    runtime = (
        load_provider("onnx")
        .get_video_embedder(options=options | {"max_input_bytes": 2 * image.nbytes + len("<|video|>")})
        .instantiate()
    )
    try:
        with pytest.raises(EmbeddingConfigurationError, match="including retrieval prefixes"):
            runtime.embed_video([VideoClip((image, image + 1), (0.0, 1.0), (0, 1))])
        assert not [event for event in sdk.events if event[0] in {"process", "run"}]
    finally:
        runtime.close()


@pytest.mark.parametrize("audio_lengths", [(16000, 32000), (32000, 16000)])
def test_mixed_media_keeps_audio_padding_separate_and_decoded_rgb_channels_last(options, sdk, audio_lengths):
    image = np.arange(3 * 7 * 3, dtype=np.uint8).reshape(3, 7, 3)
    clips = [AudioClip(np.zeros((length, 1), dtype=np.float32), 16000) for length in audio_lengths]
    video = VideoClip((image, image.copy()), (0.0, 1.0), (0, 1))
    content = EmbeddingContent(
        tuple(
            EmbeddingPart(EmbeddingPartSpec(kind), value)
            for kind, value in (("image", image), ("audio", clips[0]), ("audio", clips[1]), ("video", video))
        )
    )
    runtime = load_provider("onnx").get_multimodal_embedder(options=options).instantiate()
    try:
        values = runtime.embed_multimodal([content])
        assert len(values) == 1 and values[0].shape == (768,)
        processed = next(event[2] for event in sdk.events if event[0] == "process")
        assert "padding" not in processed
        assert processed["text_kwargs"] == {"padding": False}
        assert processed["truncation"] is False
        assert processed["audio_kwargs"] == {"padding": True}
        assert [len(clip) for clip in processed["audio"]] == list(audio_lengths)
        assert processed["images_kwargs"]["input_data_format"] == "channels_last"
        assert processed["videos_kwargs"]["input_data_format"] == "channels_last"
        assert processed["images"][0][0].shape == (3, 7, 3)
        assert [video.shape for video in processed["videos"]] == [(1, 3, 7, 3), (1, 3, 7, 3)]
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "response", [np.zeros((1, 768)), np.ones((1, 768)), np.full((1, 768), np.nan), np.ones((1, 128)), np.ones((2, 768))]
)
def test_invalid_model_results_are_not_substituted(options, sdk, response):
    runtime = load_provider("onnx").get_text_embedder(options=options).instantiate()
    sdk.response = response
    try:
        with pytest.raises(_ProviderResultError):
            runtime.embed_text(["input"])
    finally:
        runtime.close()


def test_overlength_and_invalid_media_fail_before_any_graph_runs(options, sdk):
    runtime = load_provider("onnx").get_text_embedder(options=options | {"max_length": 8}).instantiate()
    try:
        with pytest.raises(EmbeddingConfigurationError, match="max_length"):
            runtime.embed_text(["input"])
        for invalid in (np.zeros((3, 3, 3), np.float32), b"encoded image"):
            with pytest.raises(EmbeddingConfigurationError):
                runtime.embed_image([invalid])
        with pytest.raises(EmbeddingConfigurationError):
            runtime.embed_text(["<|image|>"])
        assert not any(event[0] == "run" for event in sdk.events)
    finally:
        runtime.close()
