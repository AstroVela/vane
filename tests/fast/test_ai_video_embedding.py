# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Ordered clip contracts, without model downloads or CUDA."""

from __future__ import annotations

import asyncio
import pickle
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pytest

import vane
from vane.ai import embed, embed_video
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._video_embedding import VideoInputSpec, video_clips_from_arrow
from vane.ai.functions import _EmbedVideoBatch
from vane.ai.protocols import VideoEmbedderDescriptor
from vane.ai.provider import Provider
from vane.ai.providers._cosmos_embed1 import COSMOS_MODEL
from vane.ai.providers.transformers import TransformersProvider
from vane.execution.udf_file_contract import FileUDFContract


class ClipEmbedder:
    def __init__(self, behavior):
        self.behavior = behavior

    def embed_video(self, clips):
        if self.behavior == "failure" and any(clip.frames[0].flat[0] == 255 for clip in clips):
            raise RuntimeError("private clip")
        vectors = [np.array([clip.frames[0].flat[0], clip.frames[-1].flat[0], clip.frame_times[-1]]) for clip in clips]
        return [v[:2] for v in vectors] if self.behavior == "bad_dimension" else vectors


@dataclass
class ClipDescriptor(VideoEmbedderDescriptor):
    behavior: str = "normal"

    def get_provider(self):
        return "video_fixture"

    def get_model(self):
        return self.behavior

    def get_options(self):
        return {}

    def get_dimensions(self):
        return 3

    def get_input_spec(self):
        return VideoInputSpec(frame_count=2, max_frames=2, max_input_bytes=1024)

    def instantiate(self):
        if self.behavior == "must_not_load":
            raise EmbeddingConfigurationError("model loaded before non-NULL execution")
        return ClipEmbedder(self.behavior)


class ClipProvider(Provider):
    @property
    def name(self):
        return "video_fixture"

    def get_video_embedder(self, model=None, dimensions=None, *, options=None):
        return ClipDescriptor(model or "normal")


@pytest.fixture
def provider(monkeypatch):
    from vane.ai.provider import PROVIDERS

    monkeypatch.setitem(PROVIDERS, "video_fixture", lambda name=None: ClipProvider())
    return ClipProvider()


def clip(first=3, last=7):
    return [
        {"frame_index": i, "frame_time": t, "data": np.full((2, 4, 3), value, dtype=np.uint8)}
        for i, t, value in [(4, 0.5, first), (9, 1.25, last)]
    ]


def clip_array(values, image_type="IMAGE('RGB')"):
    dtype = vane.list_type(
        vane.struct_type(
            {"frame_index": vane.sqltypes.BIGINT, "frame_time": vane.sqltypes.DOUBLE, "data": vane.sqltype(image_type)}
        )
    )
    return FileUDFContract("fixture", (), (dtype,)).scalar_outputs_to_array(values)


def drive(wrapper, values):
    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        return wrapper(pa.table({"frames": values}))["embedding"].to_pylist()
    finally:
        wrapper.close()
        loop.close()


@pytest.mark.parametrize("image_type", ["IMAGE", "IMAGE('RGB')", "IMAGE('RGB', 2, 4)"])
def test_nested_images_preserve_slices_chunks_order_and_nulls(image_type):
    array = clip_array([clip(99), clip(), None, clip(11, 23)], image_type)
    values = pa.chunked_array([array.slice(1, 2), array.slice(3)])
    wrapper = _EmbedVideoBatch(ClipDescriptor(), "frames", "embedding", 3)
    assert drive(wrapper, values) == [[3, 7, 1.25], None, [11, 23, 1.25]]
    loaded = wrapper._embedder
    assert drive(wrapper, values) == [[3, 7, 1.25], None, [11, 23, 1.25]]
    assert wrapper._embedder is loaded  # synchronous model reused between batches
    restored = pickle.loads(pickle.dumps(wrapper))
    assert restored._embedder is None and restored._run_async is None
    assert drive(restored, values) == [[3, 7, 1.25], None, [11, 23, 1.25]]


def test_empty_batches_and_null_clips_do_not_load_models():
    wrapper = _EmbedVideoBatch(ClipDescriptor("must_not_load"), "frames", "embedding", 3)
    assert drive(wrapper, clip_array([None, None])) == [None, None]
    assert drive(wrapper, clip_array([])) == []
    assert wrapper._embedder is None


def test_64_bit_arrow_list_offsets_use_the_same_clip_contract():
    array = clip_array([clip(), None, clip(8, 9)])
    large = pa.LargeListArray.from_arrays(array.offsets.cast(pa.int64()), array.values, mask=array.is_null())
    wrapper = _EmbedVideoBatch(ClipDescriptor(), "frames", "embedding", 3)
    assert drive(wrapper, large.slice(1)) == [None, [8, 9, 1.25]]


@pytest.mark.parametrize("field", ["frame_index", "frame_time"])
def test_all_null_metadata_is_rejected_for_non_null_frames(field):
    value = clip()
    for frame in value:
        frame[field] = None
    wrapper = _EmbedVideoBatch(ClipDescriptor(), "frames", "embedding", 3, on_error="ignore")
    with pytest.raises(EmbeddingConfigurationError, match="nonnegative"):
        drive(wrapper, clip_array([value]))
    assert wrapper._embedder is None


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "count",
        "null_frame",
        "null_image",
        "null_time",
        "nan_time",
        "negative_time",
        "reverse_time",
        "reverse_index",
        "duplicate_index",
        "negative_index",
    ],
)
def test_invalid_clips_are_configuration_errors_even_with_ignore(case):
    value = clip()
    if case == "empty":
        value = []
    elif case == "count":
        value.pop()
    elif case == "null_frame":
        value[1] = None
    elif case == "null_image":
        value[1]["data"] = None
    elif case in {"null_time", "nan_time", "negative_time", "reverse_time"}:
        value[1]["frame_time"] = {
            "null_time": None,
            "nan_time": float("nan"),
            "negative_time": -1.0,
            "reverse_time": 0.1,
        }[case]
    else:
        value[1]["frame_index"] = {"reverse_index": 3, "duplicate_index": 4, "negative_index": -1}[case]
    wrapper = _EmbedVideoBatch(ClipDescriptor(), "frames", "embedding", 3, on_error="ignore")
    with pytest.raises(EmbeddingConfigurationError):
        drive(wrapper, clip_array([value]))
    assert wrapper._embedder is None


def test_clip_byte_budget_and_pixel_type():
    column = pa.chunked_array([clip_array([clip()])])
    with pytest.raises(EmbeddingConfigurationError, match="byte limit"):
        video_clips_from_arrow(column, VideoInputSpec(max_input_bytes=47))
    value = clip()
    for frame in value:
        frame["data"] = np.ones((2, 4, 1), dtype=np.uint8)
    with pytest.raises(EmbeddingConfigurationError, match="UInt8 RGB"):
        video_clips_from_arrow(pa.chunked_array([clip_array([value], "IMAGE('L')")]), VideoInputSpec())


def test_output_contract_and_per_clip_error_isolation():
    values = clip_array([clip(), None, clip(255), clip(8, 9)])
    wrapper = _EmbedVideoBatch(ClipDescriptor("failure"), "frames", "embedding", 3, on_error="ignore", max_retries=0)
    assert drive(wrapper, values) == [[3, 7, 1.25], None, None, [8, 9, 1.25]]
    wrapper = _EmbedVideoBatch(ClipDescriptor("bad_dimension"), "frames", "embedding", 3)
    with pytest.raises(TypeError, match="expected 3"):
        drive(wrapper, values)


@pytest.mark.parametrize("entry", ["expression", "keyword", "relation", "relation_keyword", "method", "sql"])
def test_public_apis_keep_one_fixed_vector_per_clip(provider, entry):
    with vane.connect() as conn:
        conn.register("clips", pa.table({"id": range(5), "frames": clip_array([None, None, clip(), None, clip(8, 9)])}))
        rel = conn.table("clips")
        opts = dict(provider=provider, batch_size=2, max_retries=0)
        if entry == "expression":
            result = rel.select(vane.col("id"), embed_video(vane.col("frames"), **opts).alias("embedding"))
        elif entry == "keyword":
            result = rel.select(vane.col("id"), embed_video(frames=vane.col("frames"), **opts).alias("embedding"))
        elif entry == "relation":
            result = embed_video(rel, vane.col("frames"), **opts)
        elif entry == "relation_keyword":
            result = embed_video(rel=rel, frames=vane.col("frames"), **opts)
        elif entry == "method":
            result = rel.embed_video(vane.col("frames"), **opts)
        else:
            result = conn.sql(
                "SELECT id, ai_embed_video(frames, provider => 'video_fixture', options => {batch_size: 2}) AS embedding FROM clips"
            )
        assert str(result.types[result.columns.index("embedding")]) == "FLOAT[3]"
        assert result.select("id", "embedding").order("id").fetchall() == [
            (0, None),
            (1, None),
            (2, (3, 7, 1.25)),
            (3, None),
            (4, (8, 9, 1.25)),
        ]


@pytest.mark.parametrize("runner", ["local-fast", pytest.param("ray", marks=pytest.mark.real_ray)])
@pytest.mark.parametrize("source", ["typed_nulls", "decode_errors"])
@pytest.mark.parametrize("entry", ["expression", "sql"])
@pytest.mark.parametrize("on_error", ["raise", "ignore"])
def test_all_null_clip_columns_execute_without_loading_models(
    request, provider, monkeypatch, tmp_path, runner, source, entry, on_error
):
    if source == "decode_errors":
        pytest.importorskip("av")
        pytest.importorskip("psutil")
    if runner == "ray":
        request.getfixturevalue("ray_local")
        monkeypatch.delenv("VANE_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(config={"video_backend": "python"}) as conn:
        if source == "typed_nulls":
            # Keep BIGINT/DOUBLE metadata from a populated child array. A table
            # scan prevents the literal-NULL constant folding of the SQL tests.
            frames = clip_array([clip(), None, None, None, None]).slice(1)
            conn.register("clips", pa.table({"id": range(4), "frames": frames}))
        else:
            path = tmp_path / "invalid.mp4"
            path.write_bytes(b"not a video")
            conn.register("videos", pa.table({"id": range(4), "path": [str(path)] * 4}))
            conn.sql("SELECT id, video_frames(video_file(path), on_error => 'null') AS frames FROM videos").create_view(
                "clips"
            )
        if entry == "sql":
            result = conn.sql(
                "SELECT id, ai_embed_video(frames, provider => 'video_fixture', model => 'must_not_load', "
                f"on_error => '{on_error}', options => {{batch_size: 2, max_retries: 0}}) AS embedding FROM clips"
            )
        else:
            result = conn.table("clips").select(
                vane.col("id"),
                embed_video(
                    vane.col("frames"),
                    provider=provider,
                    model="must_not_load",
                    on_error=on_error,
                    batch_size=2,
                    max_retries=0,
                ).alias("embedding"),
            )
        if runner == "ray":
            assert "ray_actor" in result.explain()
        assert result.types == [vane.sqltypes.BIGINT, vane.array_type(vane.sqltypes.FLOAT, 3)]
        assert result.order("id").fetchall() == [(index, None) for index in range(4)]


@pytest.mark.parametrize("value", ["NULL", "NULL::STRUCT(frame_index BIGINT, frame_time DOUBLE, data IMAGE)[]"])
def test_null_sql_and_explain_never_load_models(provider, value):
    with vane.connect() as conn:
        query = f"SELECT ai_embed_video({value}, provider => 'video_fixture', model => 'must_not_load')"
        assert conn.sql(query).types == [vane.array_type(vane.sqltypes.FLOAT, 3)]
        assert conn.sql(query).fetchall() == [(None,)]
        conn.sql("EXPLAIN " + query).fetchall()


@pytest.mark.parametrize(
    "value",
    [
        "'video.mp4'",
        "video_file('/missing.mp4')",
        "[image('abc'::BLOB, 1, 1, 3, 'RGB')]",
        "[{frame_index: 1::BIGINT, frame_time: 1.0::DOUBLE, data: 'bytes'::BLOB}]",
        "[{frame_index: 1, frame_time: 1.0::DOUBLE, data: image('abc'::BLOB, 1, 1, 3, 'RGB')}]",
        "[]",
    ],
)
def test_sql_and_expression_reject_untyped_or_undecoded_inputs(provider, value):
    with vane.connect() as conn:
        with pytest.raises(Exception, match="LIST of frame records"):
            conn.sql(f"SELECT ai_embed_video({value}, provider => 'video_fixture')").fetchall()
        rel = conn.sql(f"SELECT {value} AS frames")
        with pytest.raises(Exception, match="LIST of frame records"):
            rel.select(embed_video(vane.col("frames"), provider=provider)).fetchall()


COSMOS_OPTIONS = dict(revision="a" * 40, trust_remote_code=True, device="cuda", dtype="float16")


def test_paired_cosmos_metadata_is_serializable_and_requires_no_model_imports(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    provider = TransformersProvider()
    video = provider.get_video_embedder(COSMOS_MODEL, options=COSMOS_OPTIONS)
    text = provider.get_text_embedder(COSMOS_MODEL, options=COSMOS_OPTIONS)
    for descriptor in (video, text):
        restored = pickle.loads(pickle.dumps(descriptor))
        assert restored.get_model() == COSMOS_MODEL
        assert restored.get_dimensions() == 256
        assert restored.get_udf_options().num_gpus == 1
        assert restored.get_options() == COSMOS_OPTIONS
    assert video.get_input_spec().frame_count == 8
    assert not text.supports_chunking()


@pytest.mark.parametrize(
    "override",
    [
        {"trust_remote_code": False},
        {"revision": "main"},
        {"dtype": "bfloat16"},
        {"dtype": None},
        {"device": "cpu"},
        {"device": "cuda:1"},
        {"overlength": "truncate"},
        {"prompt": "query: "},
    ],
)
def test_cosmos_rejects_unsupported_model_configuration(override):
    for factory in (TransformersProvider().get_video_embedder, TransformersProvider().get_text_embedder):
        with pytest.raises((TypeError, ValueError)):
            factory(COSMOS_MODEL, options={**COSMOS_OPTIONS, **override})


def test_no_provider_or_model_fallback():
    with pytest.raises(ValueError, match="video embedding provider"):
        embed_video(vane.col("frames"), provider="openai")
    with pytest.raises(ValueError, match="requires model"):
        TransformersProvider().get_video_embedder("sentence-transformers/all-MiniLM-L6-v2")
    with pytest.raises(ValueError, match="exactly 256"):
        TransformersProvider().get_video_embedder(COSMOS_MODEL, dimensions=128, options=COSMOS_OPTIONS)
    with vane.connect() as conn:
        rel = conn.sql("SELECT 'query' AS text")
        with pytest.raises(EmbeddingConfigurationError, match="chunk averaging"):
            embed(
                rel,
                vane.col("text"),
                provider="transformers",
                model=COSMOS_MODEL,
                max_chunk_chars=300,
                **COSMOS_OPTIONS,
            )


@pytest.mark.real_ray
@pytest.mark.parametrize("entry", ["sql", "relation"])
def test_default_ray_transports_ordered_nested_images(ray_local, provider, monkeypatch, entry):
    # Exercise the default runner without VANE_RUNNER or set_runner_*.
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    with vane.connect() as conn:
        conn.register(
            "clips",
            pa.table(
                {"id": range(5), "frames": clip_array([None, None, clip(), None, clip(8, 9)], "IMAGE('RGB', 2, 4)")}
            ),
        )
        if entry == "sql":
            result = conn.sql(
                "SELECT id, ai_embed_video(frames, provider => 'video_fixture', options => {batch_size: 2}) AS embedding FROM clips"
            )
        else:
            result = conn.table("clips").embed_video(vane.col("frames"), provider=provider, batch_size=2)
        assert "ray_actor" in result.explain()
        assert result.select("id", "embedding").order("id").fetchall() == [
            (0, None),
            (1, None),
            (2, (3, 7, 1.25)),
            (3, None),
            (4, (8, 9, 1.25)),
        ]
