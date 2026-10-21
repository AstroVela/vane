# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""LiteRT contracts without native SDKs, model weights or accelerators."""

from __future__ import annotations

import asyncio
import hashlib
import io
import pickle
import subprocess
import sys
import wave
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from vane.ai._audio_embedding import AudioClip
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._litert_options import LITERT_DEFAULT_MODEL, LITERT_MODELS
from vane.ai.functions import _EmbedTextBatch, _prepare_embed_call
from vane.ai.provider import ProviderImportError, _safe_provider_execution_error, load_provider


@pytest.fixture
def options(tmp_path):
    model = tmp_path / "embedding.litertlm"
    model.write_bytes(b"authored LiteRT fixture; not a checkpoint")
    return {"model_path": str(model), "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest()}


@pytest.fixture
def sdk(monkeypatch):
    module = ModuleType("litert_lm")
    module.events = []
    module.native_dimensions = 768
    module.modalities = (True, True, True)
    module.signatures = [128, 256, 512, 1024, 2048, 8192]
    module.vision_budget = 140
    module.responses = None
    module.failure = None

    class ModelInfo:
        def __init__(self, path):
            module.events.append(("metadata", path))
            self.is_embedding_model = True
            self.embedding = SimpleNamespace(dimension=module.native_dimensions, signature_selection=module.signatures)
            self.input_modalities = SimpleNamespace(**dict(zip(("text", "vision", "audio"), module.modalities)))
            self.max_vision_token_budget = module.vision_budget

        def __enter__(self):
            return self

        def __exit__(self, *args):
            module.events.append(("metadata_closed",))

    class Backend:
        @staticmethod
        def CPU(*, thread_count):
            return SimpleNamespace(kind="cpu", thread_count=thread_count)

        @staticmethod
        def GPU():
            return SimpleNamespace(kind="gpu")

    @dataclass
    class MediaBytes:
        bytes: bytes

    class Engine:
        def __init__(self, path, **kwargs):
            module.events.append(("load", path, kwargs))
            if module.failure is not None:
                raise module.failure

        def compute_embedding_batch(self, rows, *, options):
            module.events.append(("infer", rows, options))
            if module.responses is not None:
                return [SimpleNamespace(embedding=values) for values in module.responses]
            results = []
            for row in rows:
                digest = hashlib.sha256(
                    b"".join(part.encode() if isinstance(part, str) else part.bytes for part in row)
                ).digest()
                vector = np.zeros(options.output_size, dtype=np.float32)
                vector[:32] = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) + 1
                vector /= np.linalg.norm(vector)
                results.append(SimpleNamespace(embedding=vector.tolist()))
            return results

        def close(self):
            module.events.append(("close",))

    module.ModelInfo = ModelInfo
    module.Backend = Backend
    module.Content = SimpleNamespace(ImageBytes=MediaBytes, AudioBytes=MediaBytes)
    module.EmbeddingEngine = Engine
    module.EmbeddingOptions = lambda **kwargs: SimpleNamespace(**kwargs)
    module.InputOverflowStrategy = SimpleNamespace(ERROR="error")
    monkeypatch.setitem(sys.modules, "litert_lm", module)
    return module


def descriptor(options, kind="text", model=None, dimensions=None):
    return getattr(load_provider("litert"), f"get_{kind}_embedder")(model, dimensions, options=options)


def test_registry_metadata_are_lazy_and_serializable():
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import builtins, pickle, tempfile
from pathlib import Path
original = builtins.__import__
def checked(name, *args, **kwargs):
    assert name.split('.')[0] not in {'litert_lm', 'PIL', 'huggingface_hub', 'transformers', 'torch'}
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from vane.ai.provider import load_provider
options = {'model_path': str(Path(tempfile.gettempdir()) / 'embedding.litertlm'), 'model_sha256': 'a' * 64}
for kind in ('text', 'image', 'audio', 'multimodal'):
    d = pickle.loads(pickle.dumps(getattr(load_provider('litert'), 'get_' + kind + '_embedder')(options=options)))
    assert d.get_provider() == 'litert' and d.get_dimensions() == 768
    assert d.get_output_spec().outputs[0].space.endswith(':' + 'a' * 64)
    assert d.get_udf_options().num_gpus == 0
""",
        ],
        check=True,
    )


@pytest.mark.parametrize("kind", ["text", "image", "audio", "multimodal"])
def test_descriptors_copy_options_and_need_no_model_io(options, monkeypatch, kind):
    from pathlib import Path

    def forbidden(*args, **kwargs):
        pytest.fail("descriptor metadata opened a model file")

    monkeypatch.setattr(Path, "open", forbidden)
    d = descriptor(options, kind)
    options["model_sha256"] = "0" * 64
    assert d.get_options()["model_sha256"] != options["model_sha256"]
    copy = pickle.loads(pickle.dumps(d))
    assert copy.get_output_spec() == d.get_output_spec()
    if kind in {"audio", "multimodal"}:
        assert copy.get_input_spec()


@pytest.mark.parametrize("dimensions", [128, 256, 512, 768])
@pytest.mark.parametrize("kind", ["text", "image", "audio", "multimodal"])
def test_public_preparation_matches_direct_factory(options, dimensions, kind):
    for provider in ("litert", load_provider("litert")):
        d, width, resources, *_, backend = _prepare_embed_call(
            provider, None, dimensions, "raise", options, relation=True, input_kind=kind
        )
        assert d.get_output_spec().sql_type() == f"FLOAT[{dimensions}]"
        assert width == dimensions and resources.actor_number == 1 and resources.num_gpus == 0
        assert backend == "subprocess_actor"


INVALID_OPTIONS = [
    {"model_path": "relative.litertlm"},
    {"model_path": "https://example.test/model.litertlm"},
    {"model_path": "/models/file.bin"},
    {"model_path": "/models/secret\x00.litertlm"},
    {"model_sha256": ""},
    {"model_sha256": "g" * 64},
    {"model_sha256": None},
    {"backend": "auto"},
    {"backend": "npu"},
    {"backend": None},
    {"num_threads": True},
    {"num_threads": 0},
    {"num_threads": 257},
    {"max_length": True},
    {"max_length": 129},
    {"max_length": 4096},
    {"max_input_bytes": True},
    {"max_input_bytes": 0},
    {"max_input_bytes": 64 * 1024**2 + 1},
    {"vision_tokens_per_image": True},
    {"vision_tokens_per_image": 280},
    {"cache_dir": "cache"},
    {"input_type": "search"},
    {"input_type": None},
    {"trust_remote_code": True},
    {"overlength": "truncate"},
    {"api_key": "secret"},
    {"runtime": "http"},
    {"typo": True},
]


@pytest.mark.parametrize("update", INVALID_OPTIONS)
def test_invalid_encoding_options_fail_during_planning(options, update):
    for run in (
        lambda: descriptor(options | update),
        lambda: _prepare_embed_call("litert", None, None, "ignore", options | update, relation=True),
    ):
        with pytest.raises((TypeError, ValueError)):
            run()


@pytest.mark.parametrize(
    "update",
    [
        {"actor_number": 2},
        {"actor_number": True},
        {"execution_backend": "ray_actor"},
        {"execution_backend": "subprocess_task"},
        {"max_chunk_chars": 100},
        {"chunk_overlap_chars": None},
    ],
)
def test_execution_policies_are_explicit(options, update):
    with pytest.raises((TypeError, ValueError)):
        _prepare_embed_call("litert", None, None, "raise", options | update, relation=True)


@pytest.mark.parametrize("dimensions", [True, 0, 127, 1024, "768"])
def test_bad_dimensions_are_rejected(options, dimensions):
    with pytest.raises(EmbeddingConfigurationError):
        descriptor(options, dimensions=dimensions)


def test_unsupported_models_and_capabilities_are_rejected(options):
    with pytest.raises(EmbeddingConfigurationError):
        descriptor(options, model="google/embeddinggemma-2")
    text_model = next(iter(LITERT_MODELS))
    with pytest.raises(EmbeddingConfigurationError):
        descriptor(options, kind="image", model=text_model)
    with pytest.raises(EmbeddingConfigurationError):
        descriptor(options, kind="audio", model=list(LITERT_MODELS)[1])
    with pytest.raises(NotImplementedError):
        load_provider("litert").get_video_embedder(options=options)
    with pytest.raises(NotImplementedError):
        load_provider("litert").get_prompter()
    with pytest.raises(EmbeddingConfigurationError):
        load_provider("litert", name="alias")


def test_hash_and_metadata_are_checked_before_engine_creation(options, sdk):
    with pytest.raises(EmbeddingConfigurationError, match="model_sha256"):
        descriptor(options | {"model_sha256": "0" * 64}).instantiate()
    assert not sdk.events
    for key, value in (
        ("native_dimensions", 256),
        ("modalities", (True, False, False)),
        ("signatures", [128]),
        ("vision_budget", 32),
    ):
        original = getattr(sdk, key)
        setattr(sdk, key, value)
        with pytest.raises(EmbeddingConfigurationError):
            descriptor(options).instantiate()
        setattr(sdk, key, original)
    assert not any(event[0] == "load" for event in sdk.events)
    assert sum(event[0] == "metadata_closed" for event in sdk.events) == 4


def test_explicit_backend_failure_does_not_load_another_backend(options, sdk):
    sdk.failure = RuntimeError("unavailable GPU")
    with pytest.raises(RuntimeError):
        descriptor(options | {"backend": "gpu"}).instantiate()
    loads = [event for event in sdk.events if event[0] == "load"]
    assert len(loads) == 1 and loads[0][2]["backend"].kind == "gpu"


def test_text_roles_batch_order_dimensions_and_close(options, sdk):
    runtime = descriptor(options | {"input_type": "query"}, dimensions=128).instantiate()
    values = runtime.embed_text(["first", "", "第三", "first"])
    assert np.asarray(values).shape == (4, 128) and np.asarray(values).dtype == np.float32
    np.testing.assert_allclose(values[0], values[3])
    assert not np.array_equal(values[0], values[2])
    call = [event for event in sdk.events if event[0] == "infer"][-1]
    assert call[1] == [["task: search result | query: " + text] for text in ["first", "", "第三", "first"]]
    assert call[2].input_overflow_strategy == "error" and call[2].normalize and call[2].output_size == 128
    assert runtime.embed_text([]) == []
    runtime.embed_text(["again"])
    assert sum(event[0] == "load" for event in sdk.events) == 1
    runtime.close()
    asyncio.run(runtime.aclose())
    assert sum(event[0] == "close" for event in sdk.events) == 1
    with pytest.raises(RuntimeError, match="closed"):
        runtime.embed_text([])


def test_images_audio_and_ordered_joint_inputs(options, sdk):
    Image = pytest.importorskip("PIL.Image")

    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)[:, ::-1]
    image.setflags(write=False)
    clip = AudioClip(np.array([[-1], [0], [1]], dtype=np.float64), 16000)
    runtime = descriptor(options, kind="multimodal").instantiate()
    try:
        runtime.embed_multimodal(
            [
                EmbeddingContent(
                    (
                        EmbeddingPart(EmbeddingPartSpec("image"), image),
                        EmbeddingPart(EmbeddingPartSpec("text"), "caption"),
                        EmbeddingPart(EmbeddingPartSpec("audio"), clip),
                    )
                )
            ]
        )
        row = [event for event in sdk.events if event[0] == "infer"][-1][1][0]
        assert row[1] == "title: none | text: caption"
        with Image.open(io.BytesIO(row[0].bytes)) as restored:
            np.testing.assert_array_equal(np.asarray(restored), image)
        with wave.open(io.BytesIO(row[2].bytes), "rb") as restored:
            assert (restored.getframerate(), restored.getnchannels(), restored.getsampwidth()) == (16000, 1, 2)
            np.testing.assert_array_equal(np.frombuffer(restored.readframes(3), dtype="<i2"), [-32768, 0, 32767])
        runtime.embed_image([image])
        runtime.embed_audio([clip])
    finally:
        runtime.close()


def test_audio_transport_requires_no_optional_image_sdk(options, sdk):
    runtime = descriptor(options, kind="audio").instantiate()
    try:
        runtime.embed_audio([AudioClip(np.array([[-1], [0], [1]], dtype=np.float32), 16000)])
        content = [event for event in sdk.events if event[0] == "infer"][-1][1][0][0]
        with wave.open(io.BytesIO(content.bytes), "rb") as restored:
            assert (restored.getframerate(), restored.getnchannels(), restored.getsampwidth()) == (16000, 1, 2)
            np.testing.assert_array_equal(np.frombuffer(restored.readframes(3), dtype="<i2"), [-32768, 0, 32767])
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "kind,value",
    [
        ("text", None),
        ("image", np.zeros((2, 2, 4), dtype=np.uint8)),
        ("image", np.zeros((0, 2, 3), dtype=np.uint8)),
        ("image", np.zeros((2, 2, 3), dtype=np.float32)),
        ("audio", AudioClip(np.zeros((2, 1), dtype=np.float32), 8000)),
        ("audio", AudioClip(np.zeros((2, 2), dtype=np.float32), 16000)),
        ("audio", AudioClip(np.array([[np.nan]], dtype=np.float32), 16000)),
        ("audio", AudioClip(np.array([[1.01]], dtype=np.float32), 16000)),
    ],
)
def test_invalid_media_never_reaches_sdk(options, sdk, kind, value):
    runtime = descriptor(options, kind=kind).instantiate()
    try:
        with pytest.raises(EmbeddingConfigurationError):
            getattr(runtime, f"embed_{kind}")([value])
        assert not any(event[0] == "infer" for event in sdk.events)
    finally:
        runtime.close()


def test_input_budget_checks_all_rows_before_any_inference(options, sdk, monkeypatch):
    import vane.ai.providers.litert as provider_module

    runtime = descriptor(options | {"max_input_bytes": 32}).instantiate()
    try:
        for rows in (["small", "x" * 33], ["x" * 20]):
            with pytest.raises(EmbeddingConfigurationError):
                runtime.embed_text(rows)
        monkeypatch.setattr(provider_module, "MAX_BATCH_BYTES", 5)
        with pytest.raises(EmbeddingConfigurationError):
            runtime.embed_text(["abc", "abc"])
        assert not any(event[0] == "infer" for event in sdk.events)
    finally:
        runtime.close()


@pytest.mark.parametrize("responses", [[], [[0] * 768], [[np.nan] * 768], [[1] * 768], [[1] + [0] * 127]])
def test_malformed_sdk_outputs_fail(options, sdk, responses):
    runtime = descriptor(options).instantiate()
    sdk.responses = responses
    try:
        with pytest.raises(TypeError):
            runtime.embed_text(["test"])
    finally:
        runtime.close()


def test_public_wrapper_preserves_null_rows_and_reuses_runtime(options, sdk):
    loop = asyncio.new_event_loop()
    batch = _EmbedTextBatch(descriptor(options), "text", "embedding", 768, on_error="raise", normalize=False)
    batch.bind_async_runtime(loop.run_until_complete)
    try:
        result = batch(pa.table({"text": ["a", None, "b", "a"]}))
        assert result.column(0).null_count == 1 and len(result) == 4
        assert result.column(0)[0].as_py() == result.column(0)[3].as_py()
        batch(pa.table({"text": ["later"]}))
        assert sum(event[0] == "load" for event in sdk.events) == 1
    finally:
        batch.close()
        loop.close()
    assert sum(event[0] == "close" for event in sdk.events) == 1


def test_missing_dependency_is_actionable_and_safe(options, monkeypatch):
    import builtins

    original = builtins.__import__

    def checked(name, *args, **kwargs):
        if name == "litert_lm":
            raise ModuleNotFoundError("missing", name="litert_lm")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)
    with pytest.raises(ProviderImportError) as caught:
        descriptor(options).instantiate()
    restored = _safe_provider_execution_error("litert", LITERT_DEFAULT_MODEL, "initialization", caught.value)
    assert isinstance(restored, ProviderImportError) and "vane-ai[litert]" in str(restored)
