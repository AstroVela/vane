# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Image embedding contracts without model downloads or provider credentials."""

from __future__ import annotations

import asyncio
import pickle
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

import vane
from vane.ai import embed_image
from vane.ai.functions import _EmbedImageBatch
from vane.ai.options import validate_embed_image_options
from vane.ai.protocols import ImageEmbedderDescriptor
from vane.ai.provider import Provider
from vane.ai.providers.transformers import TransformersImageEmbedderDescriptor, TransformersProvider
from vane.execution.udf_file_contract import FileUDFContract


class PixelEmbedder:
    def __init__(self, behavior="normal"):
        self.behavior = behavior

    def embed_image(self, images):
        if self.behavior == "row_failure" and any(image.flat[0] == 255 for image in images):
            raise ValueError("private image contents")
        values = [np.array([image.flat[0], image.shape[0], image.shape[1]], dtype=np.float32) for image in images]
        if self.behavior == "bad_vector":
            return [np.array([np.nan, 1, 1]) if image.flat[0] == 255 else value for image, value in zip(images, values)]
        if self.behavior == "bad_dimension":
            return [value[:2] for value in values]
        return values


@dataclass
class PixelDescriptor(ImageEmbedderDescriptor):
    behavior: str = "normal"

    def get_provider(self):
        return "image_fixture"

    def get_model(self):
        return self.behavior

    def get_options(self):
        return {}

    def get_dimensions(self):
        return 3

    def instantiate(self):
        if self.behavior == "must_not_load":
            raise AssertionError("model loaded before non-NULL execution")
        return PixelEmbedder(self.behavior)


class PixelProvider(Provider):
    @property
    def name(self):
        return "image_fixture"

    def get_image_embedder(self, model=None, dimensions=None, *, options=None):
        return PixelDescriptor(model or "normal")


@pytest.fixture
def provider(monkeypatch):
    from vane.ai.provider import PROVIDERS

    monkeypatch.setitem(PROVIDERS, "image_fixture", lambda name=None: PixelProvider())
    return PixelProvider()


def image_array(values, dtype=None):
    dtype = dtype or vane.image_type("RGB")
    return FileUDFContract("fixture", (), (dtype,)).scalar_outputs_to_array(values)


def pixels(value, *, channels=3, dtype=np.uint8):
    return np.full((2, 4, channels), value, dtype=dtype)


def drive(wrapper, values, dtype=None):
    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        return wrapper(pa.table({"image": image_array(values, dtype)}))["embedding"].to_pylist()
    finally:
        wrapper.close()
        loop.close()


@pytest.mark.parametrize("declared", ["IMAGE", "IMAGE('RGB')", "IMAGE('RGB', 2, 4)"])
def test_dense_image_transport_preserves_rows_and_nulls(declared):
    wrapper = _EmbedImageBatch(PixelDescriptor(), "image", "embedding", 3)
    assert drive(wrapper, [pixels(3), None, pixels(7)], vane.sqltype(declared)) == [[3, 2, 4], None, [7, 2, 4]]
    assert drive(wrapper, [], vane.sqltype(declared)) == []


def test_null_images_do_not_initialize_model():
    wrapper = _EmbedImageBatch(PixelDescriptor("must_not_load"), "image", "embedding", 3)
    assert drive(wrapper, [None, None]) == [None, None]
    assert wrapper._embedder is None


@pytest.mark.parametrize("behavior", ["row_failure", "bad_vector"])
def test_ignore_recovers_valid_neighbors(behavior):
    wrapper = _EmbedImageBatch(PixelDescriptor(behavior), "image", "embedding", 3, on_error="ignore", max_retries=0)
    assert drive(wrapper, [pixels(1), None, pixels(255), pixels(7)]) == [[1, 2, 4], None, None, [7, 2, 4]]


def test_raise_sanitizes_provider_errors():
    wrapper = _EmbedImageBatch(PixelDescriptor("row_failure"), "image", "embedding", 3, max_retries=0)
    with pytest.raises(Exception) as error:
        drive(wrapper, [pixels(255)])
    assert "private image contents" not in str(error.value)
    assert error.value.__context__ is None


def test_dimension_and_normalization_contract():
    wrapper = _EmbedImageBatch(PixelDescriptor("bad_dimension"), "image", "embedding", 3)
    with pytest.raises(TypeError, match="expected 3"):
        drive(wrapper, [pixels(1)])
    wrapper = _EmbedImageBatch(PixelDescriptor(), "image", "embedding", 3, normalize=True)
    np.testing.assert_allclose(drive(wrapper, [pixels(1)])[0], np.array([1, 2, 4]) / np.sqrt(21), rtol=1e-6)


def test_async_image_client_lifecycle_and_serialization():
    events = []

    class AsyncImages:
        async def embed_image(self, values):
            events.append(("call", asyncio.get_running_loop()))
            return PixelEmbedder().embed_image(values)

        async def aclose(self):
            events.append(("close", asyncio.get_running_loop()))

    wrapper = _EmbedImageBatch(PixelDescriptor(), "image", "embedding", 3)
    wrapper._descriptor = SimpleNamespace(instantiate=AsyncImages)
    assert drive(wrapper, [pixels(9)]) == [[9, 2, 4]]
    assert events[0][0] == "call" and events[1][0] == "close"
    assert events[0][1] is events[1][1]
    assert wrapper._embedder is None
    wrapper = _EmbedImageBatch(PixelDescriptor(), "image", "embedding", 3)
    drive(wrapper, [pixels(9)])
    restored = pickle.loads(pickle.dumps(wrapper))
    assert restored._embedder is None and restored._run_async is None
    assert drive(restored, [pixels(2)]) == [[2, 2, 4]]


@pytest.mark.parametrize("entry", ["expression", "keyword", "relation", "relation_keyword", "method", "sql"])
def test_public_entry_points_have_fixed_outputs(provider, entry):
    with vane.connect() as conn:
        conn.register("images", pa.table({"id": [0, 1, 2], "image": image_array([pixels(3), None, pixels(7)])}))
        rel = conn.table("images")
        opts = dict(provider=provider, max_retries=0, batch_size=2)
        if entry == "expression":
            result = rel.select(vane.col("id"), embed_image(vane.col("image"), **opts).alias("embedding"))
        elif entry == "keyword":
            result = rel.select(vane.col("id"), embed_image(image=vane.col("image"), **opts).alias("embedding"))
        elif entry == "relation":
            result = embed_image(rel, vane.col("image"), **opts)
        elif entry == "relation_keyword":
            result = embed_image(rel=rel, image=vane.col("image"), **opts)
        elif entry == "method":
            result = rel.embed_image(vane.col("image"), **opts)
        else:
            result = conn.sql(
                "SELECT id, ai_embed_image(image, provider => 'image_fixture', options => {batch_size: 2}) AS embedding FROM images"
            )
        assert str(result.types[result.columns.index("embedding")]) == "FLOAT[3]"
        assert result.select("id", "embedding").order("id").fetchall() == [(0, (3, 2, 4)), (1, None), (2, (7, 2, 4))]


@pytest.mark.parametrize("declared", ["IMAGE", "IMAGE('RGB')", "IMAGE('RGB', 2, 4)"])
def test_sql_preserves_image_layout(provider, declared):
    with vane.connect() as conn:
        conn.register("images", pa.table({"image": image_array([pixels(5), None], vane.sqltype(declared))}))
        assert conn.sql("SELECT ai_embed_image(image, provider => 'image_fixture') FROM images").fetchall() == [
            ((5, 2, 4),),
            (None,),
        ]


@pytest.mark.parametrize("sql", ["NULL", "NULL::IMAGE", "NULL::IMAGE('RGB', 2, 4)"])
def test_sql_null_and_explain_never_load_model(provider, sql):
    with vane.connect() as conn:
        query = f"SELECT ai_embed_image({sql}, provider => 'image_fixture', model => 'must_not_load') AS embedding"
        assert conn.sql(query).types == [vane.array_type(vane.sqltypes.FLOAT, 3)]
        assert conn.sql(query).fetchall() == [(None,)]
        conn.sql("EXPLAIN " + query).fetchall()
        conn.sql("EXPLAIN SELECT ai_embed_image(image('abc'::BLOB, 1, 1, 3, 'RGB'))").fetchall()


@pytest.mark.parametrize(
    "sql",
    ["'path.png'", "'encoded'::BLOB", "[1, 2, 3]", "image_file('/tmp/image.png')", "{data: [1, 2, 3], height: 1}"],
)
def test_rejects_implicit_decoding_in_both_apis(provider, sql):
    with vane.connect() as conn:
        with pytest.raises(Exception, match="decoded IMAGE"):
            conn.sql(f"SELECT ai_embed_image({sql}, provider => 'image_fixture')").fetchall()
        rel = conn.sql(f"SELECT {sql} AS value")
        with pytest.raises(Exception, match="decoded IMAGE"):
            rel.select(embed_image(vane.col("value"), provider=provider)).fetchall()


@pytest.mark.parametrize(
    "name,value",
    [
        ("prompt", "query"),
        ("input_type", "query"),
        ("overlength", "error"),
        ("max_chunk_chars", 10),
        ("request_batch_size", 2),
        ("max_concurrency_per_actor", 2),
        ("api_key", "secret"),
    ],
)
def test_image_options_reject_text_and_remote_controls(name, value):
    with pytest.raises((ValueError, TypeError)):
        validate_embed_image_options("transformers", {name: value}, relation=True)


def test_model_capabilities_and_metadata_are_local(provider):
    image = TransformersProvider().get_image_embedder()
    text = TransformersProvider().get_text_embedder(model=image.get_model())
    assert image.get_dimensions() == text.get_dimensions() == 512
    assert image.get_options()["device"] == "cpu"
    assert pickle.loads(pickle.dumps(image)).get_dimensions() == 512
    with pytest.raises(ValueError, match="currently supports"):
        TransformersProvider().get_image_embedder(model="sentence-transformers/all-MiniLM-L6-v2", dimensions=384)
    with pytest.raises(ValueError, match="image embedding provider"):
        embed_image(vane.col("image"), provider="openai")
    with pytest.raises(ValueError, match="cannot produce"):
        TransformersProvider().get_image_embedder(dimensions=513)
    with pytest.raises(ValueError, match="pinned revision"):
        TransformersProvider().get_image_embedder(options={"trust_remote_code": True})


def test_clip_uses_rgb_processor_inputs_and_closes_images(monkeypatch):
    pytest.importorskip("PIL.Image")
    calls = []
    batches = []

    class CLIPModel:
        pass

    class SentenceTransformer:
        fail_encode = False

        def __init__(self, model, **options):
            calls.append((model, options))

        def eval(self):
            return self

        def _first_module(self):
            return CLIPModel()

        def encode(self, images, **options):
            batches.extend(images)
            assert all(image.mode == "RGB" for image in images)
            assert images[0].getpixel((0, 0)) == (9, 9, 9)
            assert options == dict(convert_to_numpy=True, truncate_dim=3, show_progress_bar=False)
            if self.fail_encode:
                raise RuntimeError("inference failed")
            return np.ones((len(images), 3))

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=SentenceTransformer))
    monkeypatch.setitem(sys.modules, "sentence_transformers.models", SimpleNamespace(CLIPModel=CLIPModel))
    from contextlib import nullcontext

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    descriptor = TransformersImageEmbedderDescriptor(
        "sentence-transformers/clip-ViT-B-32",
        dimensions=3,
        options={"revision": "a" * 40, "local_files_only": True},
    )
    embedder = descriptor.instantiate()
    result = embedder.embed_image([pixels(9, channels=1), pixels(8, channels=4)])
    assert len(result) == 2
    assert calls[0][1] == dict(
        revision="a" * 40, local_files_only=True, device="cpu", trust_remote_code=False, backend="torch"
    )
    embedder.model.fail_encode = True
    with pytest.raises(RuntimeError, match="inference failed"):
        embedder.embed_image([pixels(9)])
    for image in batches:
        with pytest.raises(ValueError, match="closed"):
            image.getpixel((0, 0))
    with pytest.raises(ValueError, match="UInt8"):
        embedder.embed_image([pixels(5, dtype=np.uint16)])


def test_decode_errors_remain_at_explicit_decode_boundary(provider):
    pytest.importorskip("PIL.Image")
    with vane.connect(config={"image_backend": "python"}) as conn:
        assert conn.sql(
            "SELECT ai_embed_image(decode_image('invalid'::BLOB, on_error => 'null'), provider => 'image_fixture')"
        ).fetchall() == [(None,)]
        with pytest.raises(Exception):
            conn.sql(
                "SELECT ai_embed_image(decode_image('invalid'::BLOB), provider => 'image_fixture', on_error => 'ignore')"
            ).fetchall()


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_relation_backends_replace_output_column(provider, backend):
    with vane.connect() as conn:
        rel = conn.sql("SELECT 1 AS Embedding, image('abc'::BLOB, 1, 1, 3, 'RGB') AS image")
        result = embed_image(rel, vane.col("image"), provider=provider, execution_backend=backend)
        assert result.columns.count("embedding") == 1
        assert "Embedding" not in result.columns
        assert result.select("embedding").fetchall() == [((97, 1, 1),)]


@pytest.mark.real_ray
@pytest.mark.parametrize("fixed", [False, True])
def test_image_embedding_through_ray_actor(ray_local, provider, fixed, monkeypatch):
    # Bind on the Ray connection: binding on local-fast intentionally rewrites
    # actor UDFs to local subprocess execution before a plan is serialized.
    monkeypatch.setenv("VANE_RUNNER", "ray")
    declared = "IMAGE('RGB', 1, 1)" if fixed else "IMAGE"
    with vane.connect() as conn:
        rel = conn.sql(
            f"SELECT i, (CASE WHEN i=1 THEN NULL ELSE image('abc'::BLOB, 1, 1, 3, 'RGB') END)::{declared} AS image FROM range(3) t(i)"
        )
        if fixed:
            rel.create_view("images")
            result = conn.sql(
                "SELECT i, ai_embed_image(image, provider => 'image_fixture', options => {batch_size: 2}) AS embedding FROM images"
            )
        else:
            result = rel.embed_image(vane.col("image"), provider=provider, execution_backend="ray_actor", batch_size=2)
        assert "ray_actor" in result.explain()
        assert result.select("i", "embedding").order("i").fetchall() == [(0, (97, 1, 1)), (1, None), (2, (97, 1, 1))]
