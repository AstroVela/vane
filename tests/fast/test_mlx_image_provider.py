# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Image contracts need neither Metal, model files, nor optional model SDKs."""

from __future__ import annotations

import pickle
import subprocess
import sys

import pytest

from vane.ai.functions import _prepare_embed_call
from vane.ai.options import validate_embed_image_options
from vane.ai.provider import load_provider


def test_image_metadata_and_pickle_without_optional_imports():
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import builtins, pickle
original = builtins.__import__
def checked(name, *args, **kwargs):
    assert name.split('.')[0] not in {'mlx', 'mlx_vlm', 'transformers', 'huggingface_hub', 'PIL'}
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from vane.ai.provider import load_provider
p = load_provider('mlx')
d = pickle.loads(pickle.dumps(p.get_image_embedder()))
assert d.get_provider() == 'mlx' and d.get_model() == 'mlx-community/embeddinggemma-2-bf16'
assert d.get_dimensions() == 768 and d.get_udf_options().num_gpus == 0 and not d.is_async()
assert 'input_type' not in d.get_options()
spec = d.get_output_spec()
assert spec.layout == 'vector' and spec.sql_type() == 'FLOAT[768]'
assert spec.outputs[0].source_modalities == ('image',)
assert spec.outputs[0].space == p.get_text_embedder().get_output_spec().outputs[0].space
assert p.get_text_embedder().get_output_spec().outputs[0].source_modalities == ('text', 'image', 'video', 'audio')
""",
        ],
        check=True,
    )


def test_image_metadata_never_reads_models(monkeypatch):
    import builtins
    from pathlib import Path

    def forbidden(*args, **kwargs):
        pytest.fail("image metadata performed model I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    descriptor = load_provider("mlx").get_image_embedder()
    for name in ("get_provider", "get_model", "get_dimensions", "get_options", "get_udf_options", "get_output_spec"):
        getattr(descriptor, name)()
    assert pickle.loads(pickle.dumps(descriptor)).get_dimensions() == 768


@pytest.mark.parametrize("dimensions", [True, "768", 512, 769, 0])
def test_image_dimensions_and_output_override_are_fixed(dimensions):
    with pytest.raises(ValueError):
        load_provider("mlx").get_image_embedder(dimensions=dimensions)
    with pytest.raises(ValueError):
        load_provider("mlx").get_image_embedder().get_output_spec(dimensions=dimensions)


@pytest.mark.parametrize("model", ["other/model", "", "/private/checkpoint", "https://example.test/model"])
def test_image_model_identity_is_fixed(model):
    with pytest.raises(ValueError):
        load_provider("mlx").get_image_embedder(model)


@pytest.mark.parametrize(
    "options",
    [
        {"input_type": "document"},
        {"input_type": "query"},
        {"input_type": None},
        {"dtype": "float16"},
        {"dtype": "float32"},
        {"device": "cpu"},
        {"local_files_only": 1},
        {"trust_remote_code": True},
        {"trust_remote_code": 0},
        {"revision": "main"},
        {"revision": "g" * 40},
        {"max_length": True},
        {"max_length": 0},
        {"max_length": 2049},
        {"overlength": "truncate"},
        {"max_concurrency_per_actor": True},
        {"max_concurrency_per_actor": 2},
        {"processor_kwargs": {}},
        {"instruction": "prefix"},
        {"max_soft_tokens": 100},
        {"max_pixels": 20},
        {"image_seq_length": 280},
        {"api_key": "private"},
    ],
)
def test_image_direct_and_public_options_reject_the_same_mistakes(options):
    provider = load_provider("mlx")
    with pytest.raises((ValueError, TypeError)):
        provider.get_image_embedder(options=options)
    for selection in ("mlx", provider):
        with pytest.raises((ValueError, TypeError)):
            _prepare_embed_call(selection, None, None, "raise", options, relation=True, input_kind="image")


def test_image_options_and_execution_are_copied_without_text_defaults():
    options = {"revision": "A" * 40, "max_length": 512, "local_files_only": False}
    descriptor = load_provider("mlx").get_image_embedder(options=options)
    expected = descriptor.get_options()
    assert expected["revision"] == "a" * 40 and "input_type" not in expected
    descriptor.get_options()["max_length"] = 1
    assert descriptor.get_options() == expected and options["revision"] == "A" * 40
    for selection in ("mlx", load_provider("mlx")):
        resolved, dimensions, resources, *_, backend = _prepare_embed_call(
            selection,
            None,
            None,
            "raise",
            options | {"batch_size": 2, "normalize": True},
            relation=True,
            input_kind="image",
        )
        assert resolved.get_options() == expected and dimensions == 768
        assert resources.num_gpus == 0 and resources.actor_number == 1 and backend == "subprocess_actor"
    assert load_provider("mlx").get_text_embedder().get_options() == expected | {
        "revision": "1a4ffddb7905d3f63486748deabe091a01fb6201",
        "input_type": "document",
        "local_files_only": True,
        "max_length": 2048,
    }


@pytest.mark.parametrize(
    "options",
    [
        {"actor_number": True},
        {"actor_number": 2},
        {"actor_number": 1.0},
        {"execution_backend": "ray_actor"},
        {"execution_backend": "ray_task"},
        {"execution_backend": "subprocess_task"},
        {"max_chunk_chars": None},
        {"chunk_overlap_chars": 0},
    ],
)
def test_image_execution_policy(options):
    with pytest.raises((ValueError, TypeError)):
        validate_embed_image_options("mlx", options, relation=True)


@pytest.mark.parametrize("key", ["batch_size", "actor_number", "execution_backend", "normalize", "max_retries"])
def test_direct_image_factory_rejects_execution_options(key):
    with pytest.raises(TypeError):
        load_provider("mlx").get_image_embedder(options={key: 1})


@pytest.fixture
def image_sdk(monkeypatch, tmp_path):
    """Small SDK doubles expose actual arrays and lifecycle, without SDK imports."""
    import json
    from contextlib import nullcontext
    from types import ModuleType, SimpleNamespace

    import numpy as np

    state = SimpleNamespace(
        loads=0,
        processor_loads=0,
        model_calls=[],
        processor_calls=[],
        pil=[],
        clears=0,
        failures={},
        change=lambda value: value,
        output=None,
        snapshots=[],
    )

    def fail(phase):
        if phase in state.failures:
            raise state.failures[phase]

    class Values:
        def __init__(self, values):
            self.values, self.evaluated = values, False

        def astype(self, dtype):
            assert dtype == np.float32
            return self

        def __array__(self, dtype=None, copy=None):
            assert self.evaluated, "output copied before mx.eval"
            return np.array(self.values, dtype=dtype, copy=True)

    class Model:
        vision_tower = object()
        embed_vision = object()

        def parameters(self):
            return {
                key: SimpleNamespace(dtype="bf16")
                for key in (
                    "language_model.embed_tokens.weight",
                    "language_model.embedding_projection.weight",
                    "vision_tower.encoder.layers.0.input_layernorm.weight",
                    "embed_vision.embedding_projection.weight",
                )
            }

        def eval(self):
            state.eval_mode = True

        def __call__(self, **inputs):
            state.model_calls.append(inputs)
            fail("forward")
            if np.any(inputs["input_ids"][:, 1] == 258):
                raise ValueError("private-image-token /private/input/path memory caption")
            values = np.zeros((len(inputs["input_ids"]), 768), dtype=np.float32)
            values[np.arange(len(values)), inputs["input_ids"][:, 1]] = 1
            return SimpleNamespace(text_embeds=Values(state.output if state.output is not None else values))

    class PILImage:
        def __init__(self, array):
            assert array.dtype == np.uint8 and array.flags.c_contiguous
            self.array = array
            self.closed = False
            state.pil.append(self)

        def close(self):
            self.closed = True

    def processor(*, images, return_tensors, padding, truncation):
        assert return_tensors == "np" and padding is True and truncation is False
        assert all(isinstance(row, list) and len(row) == 1 for row in images)
        assert all(row[0].array.dtype == np.uint8 and not row[0].closed for row in images)
        state.processor_calls.append([row[0].array.copy() for row in images])
        fail("processor")
        ids = np.array([[2, int(row[0].array.flat[0]) + 3, 1] for row in images])
        positions = np.tile(np.array([[[0, 0], [-1, -1]]]), (len(images), 1, 1))
        return state.change(
            {
                "input_ids": ids,
                "attention_mask": np.ones_like(ids),
                "pixel_values": np.zeros((len(images), 2, 768), dtype=np.float32),
                "image_position_ids": positions,
            }
        )

    def processor_load(path, **kwargs):
        assert path == str(tmp_path) and kwargs == {"local_files_only": True, "trust_remote_code": False}
        state.processor_loads += 1
        fail("processor_load")
        return processor

    model = Model()
    mx = ModuleType("mlx.core")
    mx.gpu, mx.float32, mx.bfloat16, mx.floating = "gpu", np.float32, "bf16", np.floating
    mx.metal = SimpleNamespace(is_available=lambda: True)
    mx.stream = lambda device: nullcontext()
    mx.array = np.asarray
    mx.issubdtype = lambda dtype, category: dtype in ("bf16", np.float32, np.float16)

    def evaluate(value):
        fail("evaluation")
        value.evaluated = True

    def clear():
        state.clears += 1
        fail("clear")

    mx.eval, mx.clear_cache, mx.synchronize = evaluate, clear, lambda device: fail("synchronize")
    hub = ModuleType("huggingface_hub")

    def snapshot_download(**kwargs):
        state.snapshots.append(kwargs)
        return str(tmp_path)

    hub.snapshot_download = snapshot_download
    loader = ModuleType("mlx_vlm.embedding_loader")

    def load(path, **kwargs):
        assert path == tmp_path and kwargs == {"lazy": False, "strict": True}
        state.loads += 1
        fail("load")
        return model

    loader.load_embedding_model = load
    utils = ModuleType("mlx.utils")
    utils.tree_flatten = lambda value: list(value.items())
    processor_module = ModuleType("mlx_vlm.models.embedding_gemma2")
    processor_module.EmbeddingGemma2Processor = SimpleNamespace(from_pretrained=processor_load)
    pillow = ModuleType("PIL")
    pillow.Image = SimpleNamespace(fromarray=PILImage)
    for name, value in {
        "mlx": ModuleType("mlx"),
        "mlx.core": mx,
        "mlx.utils": utils,
        "mlx_vlm": ModuleType("mlx_vlm"),
        "mlx_vlm.embedding_loader": loader,
        "mlx_vlm.models": ModuleType("mlx_vlm.models"),
        "mlx_vlm.models.embedding_gemma2": processor_module,
        "huggingface_hub": hub,
        "PIL": pillow,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr("vane.ai.providers.mlx.platform.system", lambda: "Darwin")
    monkeypatch.setattr("vane.ai.providers.mlx.platform.machine", lambda: "arm64")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "embedding_gemma2",
                "text_config": {"embedding_dim": 768},
                "dtype": "bfloat16",
                "vision_config": {
                    "model_type": "gemma4_vision",
                    "dtype": "bfloat16",
                    "patch_size": 16,
                    "pooling_kernel_size": 3,
                    "default_output_length": 280,
                },
            }
        )
    )
    (tmp_path / "processor_config.json").write_text(
        json.dumps(
            {
                "image_processor": {
                    "max_soft_tokens": 280,
                    "patch_size": 16,
                    "pooling_kernel_size": 3,
                    "do_convert_rgb": True,
                    "do_resize": True,
                    "do_rescale": True,
                    "do_normalize": False,
                    "rescale_factor": 1 / 255,
                }
            }
        )
    )
    state.model, state.mx, state.path = model, mx, tmp_path
    return state


def pixels(value=1):
    import numpy as np

    return np.full((3, 5, 3), value, dtype=np.uint8)


def test_nested_samples_float32_output_and_batch_ownership(image_sdk):
    import asyncio

    import numpy as np

    embedder = load_provider("mlx").get_image_embedder().instantiate()
    try:
        original = pixels(7)
        noncontiguous = original[:, ::-1]
        readonly = pixels(9)
        readonly.setflags(write=False)
        values = embedder.embed_image([noncontiguous, readonly, pixels(3)])
        assert [int(value.argmax()) for value in values] == [10, 12, 6]
        assert all(value.shape == (768,) and value.dtype == np.float32 for value in values)
        np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4)
        saved = np.array(values, copy=True)
        embedder.embed_image([pixels(4)])
        np.testing.assert_array_equal(values, saved)
        np.testing.assert_array_equal(original, pixels(7))
        assert not readonly.flags.writeable
        assert image_sdk.loads == image_sdk.processor_loads == 1
        assert len(image_sdk.model_calls) == len(image_sdk.processor_calls) == 2
        assert all(image.closed for image in image_sdk.pil)
        assert image_sdk.model_calls[0]["image_position_ids"][0, 1].tolist() == [-1, -1]
        assert image_sdk.snapshots == [
            {
                "repo_id": "mlx-community/embeddinggemma-2-bf16",
                "revision": "1a4ffddb7905d3f63486748deabe091a01fb6201",
                "local_files_only": True,
            }
        ]
    finally:
        embedder.close()
        asyncio.run(embedder.aclose())
    assert image_sdk.clears == 1 and embedder._model is embedder._processor is None
    with pytest.raises(RuntimeError, match="closed"):
        embedder.embed_image([])


def test_empty_image_batch_does_not_call_processor_or_model(image_sdk):
    embedder = load_provider("mlx").get_image_embedder().instantiate()
    try:
        assert embedder.embed_image([]) == []
        assert image_sdk.processor_calls == image_sdk.model_calls == image_sdk.pil == []
    finally:
        embedder.close()


@pytest.mark.parametrize("kind", ["gray", "rgba", "chw", "float", "zero", "url", "bytes", "tuple"])
def test_invalid_array_contract_fails_before_allocations(image_sdk, kind):
    import numpy as np

    values = {
        "gray": np.zeros((2, 2), np.uint8),
        "rgba": np.zeros((2, 2, 4), np.uint8),
        "chw": np.zeros((3, 2, 2), np.uint8),
        "float": pixels().astype(np.float32),
        "zero": np.zeros((0, 2, 3), np.uint8),
        "url": "https://example.test/private",
        "bytes": b"private-image",
        "tuple": None,
    }
    embedder = load_provider("mlx").get_image_embedder().instantiate()
    try:
        with pytest.raises(ValueError):
            embedder.embed_image((pixels(),) if kind == "tuple" else [values[kind]])
        assert image_sdk.processor_calls == image_sdk.model_calls == image_sdk.pil == []
    finally:
        embedder.close()


def test_pixel_budget_checked_before_contiguous_copy(image_sdk, monkeypatch):
    monkeypatch.setattr("vane.ai.providers.mlx.MAX_IMAGE_PIXELS", 14)
    embedder = load_provider("mlx").get_image_embedder().instantiate()
    try:
        with pytest.raises(ValueError, match="pixel budget"):
            embedder.embed_image([pixels()])
        assert image_sdk.pil == []
    finally:
        embedder.close()


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "ids_float",
        "ids_batch",
        "ids_negative",
        "mask_float",
        "mask_shape",
        "mask_empty",
        "mask_nonbinary",
        "pixels_shape",
        "pixels_batch",
        "pixels_int",
        "pixels_fp16",
        "pixels_nan",
        "positions_float",
        "positions_shape",
        "positions_partial_padding",
        "positions_negative",
        "positions_empty",
        "positions_range",
    ],
)
def test_processor_layout_validation_precedes_forward(image_sdk, change):
    import numpy as np

    def invalid(data):
        if change == "missing":
            del data["pixel_values"]
        if change == "extra":
            data["unknown_required_input"] = np.array([1])
        if change == "ids_float":
            data["input_ids"] = data["input_ids"].astype(float)
        if change == "ids_batch":
            data["input_ids"] = data["input_ids"][:0]
        if change == "ids_negative":
            data["input_ids"][0, 0] = -1
        if change == "mask_float":
            data["attention_mask"] = data["attention_mask"].astype(float)
        if change == "mask_shape":
            data["attention_mask"] = np.ones((1, 2), dtype=int)
        if change == "mask_empty":
            data["attention_mask"][:] = 0
        if change == "mask_nonbinary":
            data["attention_mask"][0, 0] = 2
        if change == "pixels_shape":
            data["pixel_values"] = np.zeros((1, 2, 767), dtype=np.float32)
        if change == "pixels_batch":
            data["pixel_values"] = np.zeros((2, 2, 768), dtype=np.float32)
        if change == "pixels_int":
            data["pixel_values"] = data["pixel_values"].astype(int)
        if change == "pixels_fp16":
            data["pixel_values"] = data["pixel_values"].astype(np.float16)
        if change == "pixels_nan":
            data["pixel_values"][0, 0, 0] = np.nan
        if change == "positions_float":
            data["image_position_ids"] = data["image_position_ids"].astype(float)
        if change == "positions_shape":
            data["image_position_ids"] = np.zeros((1, 1, 2), dtype=int)
        if change == "positions_partial_padding":
            data["image_position_ids"][0, 0] = [-1, 0]
        if change == "positions_negative":
            data["image_position_ids"][0, 0] = [-2, -2]
        if change == "positions_empty":
            data["image_position_ids"][:] = -1
        if change == "positions_range":
            data["image_position_ids"][0, 0] = [10240, 0]
        return data

    image_sdk.change = invalid
    embedder = load_provider("mlx").get_image_embedder().instantiate()
    try:
        with pytest.raises(ValueError):
            embedder.embed_image([pixels()])
        assert not image_sdk.model_calls and all(im.closed for im in image_sdk.pil)
    finally:
        embedder.close()


def test_visual_length_budget_fails_before_forward_without_truncation(image_sdk):
    embedder = load_provider("mlx").get_image_embedder(options={"max_length": 2}).instantiate()
    try:
        with pytest.raises(ValueError, match="max_length"):
            embedder.embed_image([pixels()])
        assert not image_sdk.model_calls and all(im.closed for im in image_sdk.pil)
    finally:
        embedder.close()


@pytest.mark.parametrize("phase", ["load", "processor_load"])
def test_partial_initialization_releases_model(image_sdk, phase):
    image_sdk.failures[phase] = RuntimeError("private-model-failure")
    with pytest.raises(RuntimeError):
        load_provider("mlx").get_image_embedder().instantiate()
    assert image_sdk.clears == 1


@pytest.mark.parametrize("missing", ["vision_config", "vision_tower", "projection", "fp16", "processor_config"])
def test_vision_and_processor_configuration_are_required(image_sdk, monkeypatch, missing):
    import json

    import numpy as np

    if missing == "vision_config":
        path = image_sdk.path / "config.json"
        config = json.loads(path.read_text())
        del config["vision_config"]
        path.write_text(json.dumps(config))
    elif missing == "vision_tower":
        monkeypatch.setattr(image_sdk.model, "vision_tower", None)
    elif missing in ("projection", "fp16"):
        weights = image_sdk.model.parameters()
        if missing == "projection":
            del weights["embed_vision.embedding_projection.weight"]
        else:
            weights["embed_vision.embedding_projection.weight"].dtype = np.float16
        monkeypatch.setattr(image_sdk.model, "parameters", lambda: weights)
    else:
        path = image_sdk.path / "processor_config.json"
        config = json.loads(path.read_text())
        config["image_processor"]["do_normalize"] = True
        path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_provider("mlx").get_image_embedder().instantiate()
    assert image_sdk.clears == 1


@pytest.fixture
def image_wrapper(image_sdk):
    import asyncio

    from vane.ai.functions import _EmbedImageBatch

    loop = asyncio.new_event_loop()
    wrapper = _EmbedImageBatch(load_provider("mlx").get_image_embedder(), "image", "embedding", 768, max_retries=0)
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        yield wrapper
    finally:
        wrapper.close()
        loop.close()


def image_table(values):
    import pyarrow as pa

    import vane
    from vane.execution.udf_file_contract import FileUDFContract

    images = FileUDFContract("fixture", (), (vane.image_type("RGB"),)).scalar_outputs_to_array(values)
    return pa.table({"image": images})


def test_null_where_and_multiple_batches_preserve_ordinals(image_sdk, image_wrapper):
    import pyarrow as pa

    all_null = image_wrapper(image_table([None, None]))
    assert all_null["embedding"].to_pylist() == [None, None] and image_sdk.loads == 0
    image_wrapper._selection_columns = ("image",)
    images = image_table([pixels(1), None, pixels(255), pixels(7)])["image"].combine_chunks()
    selected = pa.StructArray.from_arrays([pa.array([True, True, False, True]), images], names=["condition", "image"])
    result = image_wrapper(pa.table({"selected_input": selected}))["embedding"].to_pylist()
    assert [v is None for v in result] == [False, True, True, False]
    assert result[0][4] == result[3][10] == 1
    image_wrapper(pa.table({"selected_input": selected}))
    assert image_sdk.loads == image_sdk.processor_loads == 1 and len(image_sdk.model_calls) == 2
    image_wrapper.close()
    assert image_sdk.clears == 1


def test_ignore_recovers_only_ordinary_failed_rows_and_keeps_errors_private(image_sdk, image_wrapper):
    image_wrapper._on_error = "ignore"
    values = image_wrapper(image_table([pixels(1), None, pixels(255), pixels(7)]))["embedding"].to_pylist()
    assert [v is None for v in values] == [False, True, True, False]
    assert values[0][4] == values[3][10] == 1
    image_wrapper._on_error = "raise"
    with pytest.raises(Exception) as error:
        image_wrapper(image_table([pixels(255)]))
    assert "private-image" not in str(error.value) and "/private" not in str(error.value)


@pytest.mark.parametrize("phase", ["load", "processor", "forward", "evaluation"])
@pytest.mark.parametrize("on_error", ["raise", "ignore"])
@pytest.mark.parametrize(
    "failure",
    [
        MemoryError("private-allocation"),
        RuntimeError("[metal::malloc] Resource limit (499000) exceeded."),
        RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory."),
    ],
)
def test_image_resource_errors_are_fatal_without_row_isolation(image_sdk, image_wrapper, phase, on_error, failure):
    image_sdk.failures[phase] = failure
    image_wrapper._on_error = on_error
    with pytest.raises(MemoryError, match="exhausted"):
        image_wrapper(image_table([pixels(1), pixels(2)]))
    assert image_sdk.loads == 1
    assert len(image_sdk.model_calls) == (1 if phase in ("forward", "evaluation") else 0)
    assert all(image.closed for image in image_sdk.pil)
    image_sdk.failures.clear()


def test_close_releases_references_even_if_synchronize_and_cache_clear_fail(image_sdk):
    from vane.ai._errors import _raise_if_resource_error

    embedder = load_provider("mlx").get_image_embedder().instantiate()
    primary = RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory.")
    image_sdk.failures.update(synchronize=primary, clear=ValueError("private-cleanup"))
    with pytest.raises(RuntimeError) as caught:
        embedder.close()
    assert caught.value is primary and len(primary.cleanup_errors) == 1
    assert embedder._model is embedder._processor is embedder._tokenizer is None
    with pytest.raises(MemoryError):
        _raise_if_resource_error(primary)
    embedder.close()
    assert image_sdk.clears == 1


@pytest.mark.parametrize("bad", ["nan", "inf", "zero", "norm", "rows", "dimensions"])
def test_image_native_output_validation(image_sdk, bad):
    import numpy as np

    output = np.zeros((1, 768), dtype=np.float32)
    output[0, 0] = {"nan": np.nan, "inf": np.inf, "zero": 0, "norm": 2}.get(bad, 1)
    if bad == "rows":
        output = np.tile(output, (2, 1))
    if bad == "dimensions":
        output = output[:, :767]
    image_sdk.output = output
    embedder = load_provider("mlx").get_image_embedder().instantiate()
    try:
        with pytest.raises(ValueError):
            embedder.embed_image([pixels()])
    finally:
        embedder.close()


def test_all_false_images_do_not_instantiate_a_model(image_sdk, image_wrapper):
    import pyarrow as pa

    image_wrapper._selection_columns = ("image",)
    images = image_table([pixels(1), pixels(2)])["image"].combine_chunks()
    selected = pa.StructArray.from_arrays([pa.array([False, False]), images], names=["condition", "image"])
    assert image_wrapper(pa.table({"selected_input": selected}))["embedding"].to_pylist() == [None, None]
    assert image_sdk.loads == image_sdk.processor_loads == 0


def test_public_sql_nulls_and_explain_need_no_image_sdk():
    import vane

    with vane.connect() as conn:
        query = "SELECT ai_embed(NULL::IMAGE, input_spec => {kind: 'image'}, provider => 'mlx') AS embedding"
        result = conn.sql(query)
        assert str(result.types[0]) == "FLOAT[768]" and result.fetchall() == [(None,)]
        conn.sql(
            "EXPLAIN SELECT ai_embed(image('abc'::BLOB,1,1,3,'RGB'), input_spec => {kind: 'image'}, provider => 'mlx')"
        ).fetchall()


@pytest.mark.parametrize("name", ["mlx", "mlx_vlm", "huggingface_hub", "PIL"])
def test_missing_image_sdk_has_the_standard_install_hint(image_sdk, monkeypatch, name):
    import builtins

    from vane.ai.provider import ProviderImportError, _safe_provider_execution_error

    original = builtins.__import__

    def missing(requested, *args, **kwargs):
        if requested.split(".")[0] == name:
            raise ModuleNotFoundError("private missing dependency", name=name)
        return original(requested, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    embedder = None
    try:
        with pytest.raises(ProviderImportError) as error:
            embedder = load_provider("mlx").get_image_embedder().instantiate()
            embedder.embed_image([pixels()])
        safe = _safe_provider_execution_error("mlx", "mlx-community/embeddinggemma-2-bf16", "Embed", error.value)
        assert "vane-ai[mlx]" in str(pickle.loads(pickle.dumps(safe)))
    finally:
        if embedder is not None:
            embedder.close()


def test_initialization_preserves_allocator_failure_and_cleanup_cause(image_sdk, image_wrapper):
    image_sdk.failures.update(
        load=RuntimeError("[metal::malloc] Resource limit (499000) exceeded."),
        synchronize=ValueError("private-cleanup-details"),
    )
    with pytest.raises(MemoryError) as caught:
        image_wrapper(image_table([pixels(1), pixels(2)]))
    assert image_sdk.loads == 1 and image_sdk.clears == 1
    assert "private-cleanup" not in str(caught.value)


@pytest.mark.parametrize("phase", ["synchronize", "clear"])
def test_image_cleanup_allocation_failure_is_fatal(image_sdk, image_wrapper, phase):
    from vane.ai._errors import DiagnosticError, _raise_if_resource_error

    image_wrapper(image_table([pixels()]))
    image_sdk.failures[phase] = MemoryError("private-cleanup-allocation")
    with pytest.raises(DiagnosticError) as caught:
        image_wrapper.close()
    with pytest.raises(MemoryError):
        _raise_if_resource_error(pickle.loads(pickle.dumps(caught.value)))
    assert "private-cleanup-allocation" not in str(caught.value)
    assert image_sdk.loads == 1 and image_sdk.clears == 1
    image_sdk.failures.clear()
