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
def test_planning_is_serializable_sdk_free_and_does_not_open_model_files(monkeypatch, kind, dimensions):
    for name in ("onnxruntime", "onnx", "transformers", "torch"):
        monkeypatch.setitem(sys.modules, name, None)
    descriptor = getattr(load_provider("onnx"), f"get_{kind}_embedder")(
        dimensions=dimensions, options={"model_path": "/not/provisioned", "model_sha256": "a" * 64}
    )
    restored = pickle.loads(pickle.dumps(descriptor))
    assert restored.get_model() == ONNX_MODEL
    assert restored.get_provider() == "onnx"
    assert restored.get_output_spec() == descriptor.get_output_spec()
    assert restored.get_output_spec().outputs[0].space == ONNX_MODEL + ":onnx-py1:" + "a" * 64
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
    with pytest.raises(EmbeddingConfigurationError, match="onnx-py1 processor"):
        load_provider("onnx").get_text_embedder(options=options).instantiate()
    assert not any(event[0] == "load" for event in sdk.events)


@pytest.mark.parametrize("encoding_version", [True, 0, 2])
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
        assert processed["text"] == ["title: none | text: look<|image|><|audio|><|video|>"]
        assert processed["videos_kwargs"] == {
            "device": "cpu",
            "input_data_format": "channels_last",
            "do_sample_frames": False,
            "add_timestamps": False,
        }
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
        assert processed["videos"][0].shape == (2, 3, 7, 3)
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
