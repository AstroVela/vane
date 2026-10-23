# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""MLX contracts run without model files, Metal, or any optional model SDK."""

from __future__ import annotations

import asyncio
import json
import pickle
import subprocess
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai.functions import _EmbedTextBatch, _prepare_embed_call
from vane.ai.options import _validate_mlx_text_options, validate_embed_options
from vane.ai.provider import ProviderImportError, _safe_provider_execution_error, load_provider
from vane.ai.providers.mlx import DIMENSIONS, MODEL_ID, MLXTextEmbedder


def test_registry_and_descriptor_need_no_optional_sdk():
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import builtins, pickle
original = builtins.__import__
def checked(name, *args, **kwargs):
    assert name.split('.')[0] not in {'mlx', 'mlx_vlm', 'transformers', 'huggingface_hub'}
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from vane.ai.provider import load_provider
d = pickle.loads(pickle.dumps(load_provider('mlx').get_text_embedder()))
assert d.get_provider() == 'mlx'
assert d.get_dimensions() == 768
assert d.get_udf_options().num_gpus == 0
assert d.get_udf_options().actor_number is None
assert not d.is_async() and not d.supports_chunking()
assert d.get_options()['input_type'] == 'document'
spec = d.get_output_spec()
assert spec.layout == 'vector' and spec.outputs[0].dimensions == 768
""",
        ],
        check=True,
    )


def test_metadata_never_reads_or_downloads_models(monkeypatch):
    import builtins
    from pathlib import Path

    def forbidden(*args, **kwargs):
        pytest.fail("descriptor metadata performed model I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    descriptor = load_provider("mlx").get_text_embedder()
    for method in (
        "get_provider",
        "get_model",
        "get_dimensions",
        "get_options",
        "get_udf_options",
        "is_async",
        "supports_chunking",
        "get_output_spec",
    ):
        getattr(descriptor, method)()
    assert pickle.loads(pickle.dumps(descriptor)).get_dimensions() == 768


def test_descriptor_options_are_copied_and_picklable():
    options = {"input_type": "query"}
    descriptor = load_provider("mlx").get_text_embedder(options=options)
    options["input_type"] = "document"
    descriptor.get_options()["input_type"] = "document"
    assert pickle.loads(pickle.dumps(descriptor)).get_options()["input_type"] == "query"


@pytest.mark.parametrize("value", [True, "768", 0, 128, 512, 769])
def test_fixed_dimensions_are_validated(value):
    with pytest.raises(EmbeddingConfigurationError):
        load_provider("mlx").get_text_embedder(dimensions=value)


@pytest.mark.parametrize("model", ["another/model", "https://example.org/model", "/tmp/model", ""])
def test_fixed_model_is_validated(model):
    with pytest.raises(EmbeddingConfigurationError):
        load_provider("mlx").get_text_embedder(model)


INVALID_OPTIONS = [{"revision": value} for value in (None, "", "main", "v1", "f" * 39, "g" * 40, 123)] + [
    {"input_type": None},
    {"input_type": "search"},
    {"dtype": "float16"},
    {"dtype": "bf16"},
    {"dtype": "float32"},
    {"device": "cpu"},
    {"device": "gpu"},
    {"local_files_only": 1},
    {"trust_remote_code": True},
    {"trust_remote_code": 0},
    {"max_length": True},
    {"max_length": 0},
    {"max_length": 2049},
    {"max_length": "32"},
    {"overlength": "truncate"},
    {"max_concurrency_per_actor": True},
    {"max_concurrency_per_actor": 2},
    {"prompt": "custom"},
    {"prompt_name": "query"},
    {"task_type": "RETRIEVAL_QUERY"},
    {"title": "title"},
    {"model_kwargs": {}},
    {"tokenizer_kwargs": {}},
    {"base_url": "http://localhost"},
    {"api_key": "private"},
    {"runtime": "service"},
    {"quantization": 4},
    {"typo": 1},
]


@pytest.mark.parametrize("options", INVALID_OPTIONS)
def test_direct_factory_and_public_api_share_validation(options):
    provider = load_provider("mlx")
    with pytest.raises((ValueError, TypeError)):
        provider.get_text_embedder(options=options)
    for selection in ("mlx", provider):
        with pytest.raises((ValueError, TypeError)):
            _prepare_embed_call(selection, None, None, "raise", options, relation=True)


def test_options_are_normalized_without_mutating_callers():
    options = {"revision": "A" * 40, "input_type": "query", "local_files_only": False, "max_length": 1}
    assert _validate_mlx_text_options(options)["revision"] == "a" * 40
    assert options["revision"] == "A" * 40
    for provider in ("mlx", load_provider("mlx")):
        descriptor, dimensions, resources, *_ = _prepare_embed_call(
            provider, MODEL_ID, 768, "raise", options | {"actor_number": None, "execution_backend": None}, relation=True
        )
        assert dimensions == 768 and resources.actor_number == 1 and resources.num_gpus == 0
        assert descriptor.get_options()["local_files_only"] is False
    with pytest.raises(ValueError, match="dtype"):
        validate_embed_options("transformers", {"dtype": "bfloat16"}, relation=True)
    assert _prepare_embed_call("mlx", None, None, "raise", {}, relation=True)[-1] == "subprocess_actor"


@pytest.mark.parametrize(
    "options",
    [
        {"actor_number": True},
        {"actor_number": 2},
        {"actor_number": 1.0},
        {"execution_backend": "ray_actor"},
        {"execution_backend": "subprocess_task"},
        {"execution_backend": "ray_task"},
        {"max_chunk_chars": None},
        {"chunk_overlap_chars": 0},
    ],
)
def test_execution_limits_are_enforced(options):
    with pytest.raises((ValueError, TypeError)):
        validate_embed_options("mlx", options, relation=True)


@pytest.mark.parametrize("key", ["batch_size", "actor_number", "execution_backend", "normalize", "max_retries"])
def test_direct_factory_rejects_execution_options(key):
    with pytest.raises(TypeError):
        load_provider("mlx").get_text_embedder(options={key: 1})


@pytest.mark.parametrize("kind", ["multimodal", "prompt"])
def test_unsupported_modalities_and_prompt_fail_before_loading(kind):
    provider = load_provider("mlx")
    method = provider.get_prompter if kind == "prompt" else getattr(provider, f"get_{kind}_embedder")
    with pytest.raises(NotImplementedError):
        method()


class Tokenizer:
    def __init__(self):
        self.texts = []
        self.padded = None

    def __call__(self, text, *, padding, truncation):
        assert padding is False and truncation is False
        self.texts.append(text)
        ids = [1, *(ord(char) + 3 for char in text), 2]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def pad(self, rows, *, padding, return_tensors):
        assert padding is True and return_tensors == "np"
        longest = max(len(row["input_ids"]) for row in rows)
        self.padded = {key: np.array([row[key] + [0] * (longest - len(row[key])) for row in rows]) for key in rows[0]}
        return self.padded


class Values:
    def __init__(self, values):
        self.values = values
        self.evaluated = False

    def astype(self, dtype):
        assert dtype == np.float32
        return self

    def __array__(self, dtype=None, copy=None):
        assert self.evaluated, "NumPy conversion happened before mx.eval"
        return np.array(self.values, dtype=dtype, copy=True)


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    state = SimpleNamespace(
        loads=[], snapshots=[], tokenizer_loads=[], model_calls=[], syncs=0, clears=0, failure=None, output=None
    )
    tokenizer = Tokenizer()

    class Model:
        def parameters(self):
            return {
                key: SimpleNamespace(dtype="bf16")
                for key in ("language_model.embed_tokens.weight", "language_model.embedding_projection.weight")
            }

        def eval(self):
            state.eval_mode = True

        def __call__(self, **inputs):
            state.model_calls.append(inputs)
            if state.failure is not None:
                raise state.failure
            vectors = np.zeros((len(inputs["input_ids"]), DIMENSIONS), dtype=np.float32)
            indices = inputs["input_ids"].sum(axis=1) % DIMENSIONS
            vectors[np.arange(len(vectors)), indices] = 1
            return SimpleNamespace(text_embeds=Values(state.output if state.output is not None else vectors))

    model = Model()
    mx = ModuleType("mlx.core")
    mx.gpu, mx.float32, mx.bfloat16, mx.floating = "gpu", np.float32, "bf16", np.floating
    mx.metal = SimpleNamespace(is_available=lambda: True)
    mx.stream = lambda device: nullcontext()
    mx.array = np.asarray
    mx.issubdtype = lambda dtype, category: dtype in ("bf16", np.float32, np.float16)
    mx.eval = lambda values: setattr(values, "evaluated", True)

    def synchronize(device):
        assert device == "gpu"
        state.syncs += 1

    def clear_cache():
        state.clears += 1

    mx.synchronize, mx.clear_cache = synchronize, clear_cache
    package = ModuleType("mlx")
    package.core = mx
    utils = ModuleType("mlx.utils")
    utils.tree_flatten = lambda values: list(values.items())
    hub = ModuleType("huggingface_hub")

    def snapshot_download(**kwargs):
        state.snapshots.append(kwargs)
        return str(tmp_path)

    hub.snapshot_download = snapshot_download
    loader = ModuleType("mlx_vlm.embedding_loader")

    def load(path, **kwargs):
        from pathlib import Path

        assert isinstance(path, Path)
        state.loads.append((path, kwargs))
        return model

    loader.load_embedding_model = load
    transformers = ModuleType("transformers")

    def from_pretrained(path, **kwargs):
        state.tokenizer_loads.append((path, kwargs))
        return tokenizer

    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=from_pretrained)
    for name, module in {
        "mlx": package,
        "mlx.core": mx,
        "mlx.utils": utils,
        "mlx_vlm": ModuleType("mlx_vlm"),
        "mlx_vlm.embedding_loader": loader,
        "huggingface_hub": hub,
        "transformers": transformers,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("vane.ai.providers.mlx.platform.system", lambda: "Darwin")
    monkeypatch.setattr("vane.ai.providers.mlx.platform.machine", lambda: "arm64")
    state.config = tmp_path / "config.json"
    state.config.write_text(
        json.dumps({"model_type": "embedding_gemma2", "text_config": {"embedding_dim": 768}, "dtype": "bfloat16"})
    )
    state.mx, state.model, state.tokenizer, state.loader = mx, model, tokenizer, loader
    return state


@pytest.mark.parametrize(
    "role,prefix", [("query", "task: search result | query: "), ("document", "title: none | text: ")]
)
def test_text_role_prefixes_padding_and_float32_output(sdk, role, prefix):
    texts = ["", "  ", "中文", "title: none | text: already", "short"]
    embedder = MLXTextEmbedder({"input_type": role})
    try:
        vectors = embedder.embed_text(texts)
        assert sdk.tokenizer.texts == [prefix + text for text in texts]
        ids, mask = sdk.model_calls[0]["input_ids"], sdk.model_calls[0]["attention_mask"]
        assert ids.shape == mask.shape and np.any(mask == 0)
        assert len(vectors) == len(texts)
        for i, vector in enumerate(vectors):
            assert vector.shape == (768,) and vector.dtype == np.float32 and np.isfinite(vector).all()
            assert np.argmax(vector) == ids[i].sum() % 768
            assert abs(np.linalg.norm(vector) - 1) < 1e-4
    finally:
        embedder.close()


def test_length_budget_counts_prefix_and_special_tokens(sdk):
    length = len("title: none | text: x") + 2
    embedder = MLXTextEmbedder({"max_length": length})
    try:
        assert len(embedder.embed_text(["x"])) == 1
        with pytest.raises(EmbeddingConfigurationError, match="max_length"):
            embedder.embed_text(["xx"])
        assert len(sdk.model_calls) == 1
    finally:
        embedder.close()


def test_empty_batch_does_no_inference(sdk):
    embedder = MLXTextEmbedder({})
    assert embedder.embed_text([]) == []
    assert not sdk.tokenizer.texts and not sdk.model_calls
    embedder.close()


@pytest.mark.parametrize("texts", ["text", [1], [b"bytes"], [["nested"]], [None]])
def test_inputs_are_never_coerced_to_strings(sdk, texts):
    embedder = MLXTextEmbedder({})
    with pytest.raises(EmbeddingConfigurationError):
        embedder.embed_text(texts)
    assert not sdk.model_calls and not sdk.tokenizer.texts
    embedder.close()


@pytest.mark.parametrize(
    "shape,value",
    [
        ((0, 768), 1),
        ((2, 768), 1),
        ((768, 1), 1),
        ((1, 767), 1),
        ((1, 769), 1),
        ((1, 768), float("nan")),
        ((1, 768), float("inf")),
        ((1, 768), 0),
        ((1, 768), 0.1),
    ],
)
def test_invalid_outputs_are_rejected(sdk, shape, value):
    sdk.output = np.full(shape, value, dtype=np.float32)
    embedder = MLXTextEmbedder({})
    with pytest.raises(ValueError, match="MLX"):
        embedder.embed_text(["text"])
    embedder.close()


def test_offline_revision_strict_loading_and_cleanup(sdk):
    revision = "a" * 40
    embedder = MLXTextEmbedder({"revision": revision, "local_files_only": False})
    assert sdk.snapshots == [{"repo_id": MODEL_ID, "revision": revision, "local_files_only": False}]
    assert sdk.loads == [(sdk.config.parent, {"lazy": False, "strict": True})]
    assert sdk.tokenizer_loads == [(str(sdk.config.parent), {"local_files_only": True, "trust_remote_code": False})]
    assert sdk.eval_mode is True
    embedder.close()
    asyncio.run(embedder.aclose())
    assert sdk.clears == 1 and embedder._model is embedder._tokenizer is None
    with pytest.raises(RuntimeError, match="closed"):
        embedder.embed_text(["text"])


@pytest.mark.parametrize(
    "change",
    [
        {"model_type": "other"},
        {"text_config": {"embedding_dim": 512}},
        {"dtype": "float16"},
        {"quantization": {"bits": 4}},
        {"quantization_config": {}},
    ],
)
def test_strict_loading_rejects_wrong_checkpoint(sdk, change):
    sdk.config.write_text(json.dumps(json.loads(sdk.config.read_text()) | change))
    with pytest.raises(EmbeddingConfigurationError):
        MLXTextEmbedder({})
    assert not sdk.loads and sdk.clears == 1


def test_fp16_weights_and_missing_weights_do_not_fall_back(sdk, monkeypatch):
    monkeypatch.setattr(
        sdk.model, "parameters", lambda: {"language_model.embed_tokens.weight": SimpleNamespace(dtype=np.float16)}
    )
    with pytest.raises(EmbeddingConfigurationError, match="BF16"):
        MLXTextEmbedder({})
    assert sdk.clears == 1

    def fail(*args, **kwargs):
        raise ValueError("missing trained weights")

    monkeypatch.setattr(sdk.loader, "load_embedding_model", fail)
    with pytest.raises(ValueError, match="missing trained weights"):
        MLXTextEmbedder({})
    assert sdk.clears == 2


@pytest.mark.parametrize("name", ["mlx", "mlx_vlm", "transformers", "huggingface_hub"])
def test_missing_sdk_preserves_install_hint(sdk, monkeypatch, name):
    import builtins

    original = builtins.__import__

    def missing(requested, *args, **kwargs):
        if requested.split(".")[0] == name:
            raise ModuleNotFoundError("optional SDK is absent", name=name)
        return original(requested, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ProviderImportError) as caught:
        MLXTextEmbedder({})
    error = pickle.loads(pickle.dumps(caught.value))
    safe = _safe_provider_execution_error("mlx", MODEL_ID, "Embed", error)
    assert safe.extra == "mlx" and "vane-ai[mlx]" in str(safe)


def test_sdk_internal_error_is_not_misreported_as_missing_package(sdk, monkeypatch):
    error = ModuleNotFoundError("private internal error", name="internal_dependency")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(sdk.loader, "load_embedding_model", fail)
    with pytest.raises(ModuleNotFoundError) as caught:
        MLXTextEmbedder({})
    assert caught.value is error and sdk.clears == 1


def test_metal_required_without_cuda_or_cpu_fallback(sdk, monkeypatch):
    assert load_provider("mlx").get_text_embedder().get_udf_options().num_gpus == 0
    monkeypatch.setattr(sdk.mx.metal, "is_available", lambda: False)
    with pytest.raises(EmbeddingConfigurationError, match="Metal"):
        MLXTextEmbedder({})
    assert not sdk.loads


def test_unsupported_execution_platform_is_explicit(sdk, monkeypatch):
    monkeypatch.setattr("vane.ai.providers.mlx.platform.system", lambda: "Linux")
    with pytest.raises(EmbeddingConfigurationError, match="Apple Silicon"):
        MLXTextEmbedder({})
    assert not sdk.loads


@pytest.fixture
def wrapper(sdk):
    loop = asyncio.new_event_loop()
    batch = _EmbedTextBatch(load_provider("mlx").get_text_embedder(), "text", "embedding", 768, max_retries=0)
    batch.bind_async_runtime(loop.run_until_complete)
    try:
        yield batch
    finally:
        batch.close()
        loop.close()


def test_multiple_batches_reuse_one_embedder_and_close(sdk, wrapper):
    for texts in (["one", "two"], ["three", "four"], ["five"]):
        assert wrapper(pa.table({"text": texts})).num_rows == len(texts)
    assert len(sdk.loads) == 1 and len(sdk.model_calls) == 3 and sdk.clears == 0
    restored = pickle.loads(pickle.dumps(wrapper))
    assert restored._embedder is None and restored._run_async is None
    embedder = wrapper._embedder
    wrapper.close()
    wrapper.close()
    assert embedder._model is embedder._tokenizer is None and sdk.clears == 1


def test_null_and_unselected_rows_do_not_reach_model(sdk, wrapper):
    wrapper._selection_columns = ("text",)
    table = pa.table(
        {
            "selected_input": pa.StructArray.from_arrays(
                [pa.array([True, True, False, True]), pa.array(["one", None, "hidden", "two"])],
                names=["condition", "text"],
            )
        }
    )
    result = wrapper(table)["embedding"].to_pylist()
    assert [value is None for value in result] == [False, True, True, False]
    assert sdk.tokenizer.texts == ["title: none | text: one", "title: none | text: two"]
    assert len(sdk.loads) == 1


def test_provider_failures_preserve_raise_ignore_and_private_errors(sdk, wrapper):
    secret = "private-token /private/model/path original-private-text"
    sdk.failure = RuntimeError(secret)
    with pytest.raises(RuntimeError) as caught:
        wrapper(pa.table({"text": ["valid"]}))
    assert secret not in str(caught.value) and secret not in str(pickle.loads(pickle.dumps(caught.value)))
    wrapper._on_error = "ignore"
    assert wrapper(pa.table({"text": ["valid"]}))["embedding"].to_pylist() == [None]
    sdk.failure = EmbeddingConfigurationError("MLX token budget exceeded")
    with pytest.raises(EmbeddingConfigurationError, match="budget"):
        wrapper(pa.table({"text": ["valid"]}))


@pytest.mark.parametrize("on_error", ["raise", "ignore"])
@pytest.mark.parametrize("phase", ["loading", "inference", "evaluation"])
@pytest.mark.parametrize(
    "message",
    [
        "[malloc] Unable to allocate 4096 bytes.",
        "[metal::malloc] Resource limit (499000) exceeded.",
        (
            "[metal::malloc] Attempting to allocate 8589934592 bytes which is greater than"
            " the maximum allowed buffer size of 4294967296 bytes."
        ),
        "[METAL] Command buffer execution failed: Insufficient Memory",
        "[METAL] Command buffer execution failed: Insufficient Memory.",
        (
            "[METAL] Command buffer execution failed: Insufficient Memory"
            " (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
        ),
        (
            "[METAL] Command buffer execution failed: Insufficient Memory"
            " (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)."
        ),
    ],
)
def test_allocation_failures_are_fatal_without_row_isolation(sdk, wrapper, monkeypatch, message, phase, on_error):
    calls = []

    def fail(*args, **kwargs):
        calls.append(None)
        raise RuntimeError(message)

    if phase == "loading":
        monkeypatch.setattr(sdk.loader, "load_embedding_model", fail)
    elif phase == "evaluation":
        monkeypatch.setattr(sdk.mx, "eval", fail)
    else:
        sdk.failure = RuntimeError(message)
    wrapper._on_error = on_error
    with pytest.raises(MemoryError, match="Embedding exhausted model or host resources"):
        wrapper(pa.table({"text": ["one", "two"]}))
    assert len(sdk.snapshots) == 1
    assert len(sdk.model_calls) == (0 if phase == "loading" else 1)
    assert len(calls) == (0 if phase == "inference" else 1)
    if phase == "loading":
        assert sdk.clears == 1
