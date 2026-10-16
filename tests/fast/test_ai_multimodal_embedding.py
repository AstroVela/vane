# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Unified inputs, optional parts, bounds and executor row identity."""

from __future__ import annotations

import asyncio
import pickle
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pytest

import vane
from vane.ai import (
    AudioInput,
    AudioInputSpec,
    ImageInput,
    MultimodalInput,
    TextInput,
    VideoInput,
    VideoInputSpec,
    embed,
)
from vane.ai._embedding_content import (
    EmbeddingContent,
    EmbeddingPart,
    EmbeddingPartSpec,
    MultimodalInputSpec,
    content_from_columns,
)
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai.functions import _EmbedMultimodalBatch
from vane.ai.protocols import MultimodalEmbedderDescriptor
from vane.ai.provider import Provider

TEXT = EmbeddingPartSpec("text")
VIDEO = EmbeddingPartSpec("video", "video/mp4")
SPEC = MultimodalInputSpec(frozenset({TEXT, VIDEO}), max_input_bytes=16)


class ContentEncoder:
    async def embed_multimodal(self, inputs):
        return [[sum(len(part.value) for part in row.parts), len(row.parts)] for row in inputs]


@dataclass
class ContentDescriptor(MultimodalEmbedderDescriptor):
    def get_provider(self):
        return "content_fixture"

    def get_model(self):
        return "content_fixture"

    def get_options(self):
        return {}

    def get_dimensions(self):
        return 2

    def get_input_spec(self):
        return SPEC

    def instantiate(self):
        return ContentEncoder()


class ContentProvider(Provider):
    @property
    def name(self):
        return "content_fixture"

    def get_multimodal_embedder(self, model=None, dimensions=None, *, options=None):
        return ContentDescriptor()


@dataclass
class OptionalContentDescriptor(ContentDescriptor):
    def get_input_spec(self):
        return MultimodalInputSpec(
            frozenset(
                {
                    TEXT,
                    VIDEO,
                    EmbeddingPartSpec("image", "image/png"),
                    EmbeddingPartSpec("audio", "audio/wav"),
                    EmbeddingPartSpec("image"),
                    EmbeddingPartSpec("video"),
                    EmbeddingPartSpec("audio"),
                }
            ),
            video=VideoInputSpec(),
            audio=AudioInputSpec(sample_rate=16000, max_samples=16000),
        )


class OptionalContentProvider(ContentProvider):
    def get_multimodal_embedder(self, model=None, dimensions=None, *, options=None):
        return OptionalContentDescriptor()


def packed_parts(table):
    return pa.table({"parts": table.to_struct_array()})


def test_optional_parts_preserve_row_order_and_do_not_expand_rows():
    table = pa.table(
        {
            "part_0": pa.chunked_array([["first", "  "], [None, "last"]]),
            "part_1": pa.chunked_array([[b"a", b"bc"], [b"d", None]]),
        }
    )
    rows = content_from_columns(table, (TEXT, VIDEO), SPEC)
    assert [[part.value for part in row.parts] for row in rows] == [["first", b"a"], [b"bc"], [b"d"], ["last"]]
    assert [[part.spec.kind for part in row.parts] for row in rows] == [
        ["text", "video"],
        ["video"],
        ["video"],
        ["text"],
    ]


@pytest.mark.parametrize(
    "text,video,message",
    [
        (None, None, "no nonempty"),
        ("  ", None, "no nonempty"),
        ("healthy text", b"", "empty"),
        ("x" * 10, b"x" * 10, "combined"),
        (None, b"x" * 17, "byte budget"),
    ],
)
def test_invalid_content_is_rejected_before_client_creation_even_with_ignore(text, video, message):
    wrapper = _EmbedMultimodalBatch(ContentDescriptor(), "parts", "vector", 2, parts=(TEXT, VIDEO), on_error="ignore")
    with pytest.raises(EmbeddingConfigurationError, match=message):
        wrapper(
            packed_parts(pa.table({"part_0": pa.array([text], pa.string()), "part_1": pa.array([video], pa.binary())}))
        )
    assert wrapper._embedder is None


def test_aggregate_budget_uses_utf8_bytes():
    with pytest.raises(EmbeddingConfigurationError, match="combined"):
        content_from_columns(pa.table({"part_0": ["视频" * 3]}), (TEXT,), SPEC)


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_joint_decoded_columns_preserve_pixels_pcm_and_source_timestamps(backend):
    from vane.execution.udf_file_contract import FileUDFContract

    pixels = np.full((2, 3, 3), 17, np.uint8)
    samples = np.full((4, 1), 0.25, np.float32)
    image_type = vane.image_type("RGB", 2, 3)
    frame_type = vane.list_type(
        vane.struct_type(
            {
                "frame_index": vane.sqltypes.BIGINT,
                "frame_time": vane.sqltypes.DOUBLE,
                "data": image_type,
            }
        )
    )
    image = FileUDFContract("fixture", (), (image_type,)).scalar_outputs_to_array([pixels])
    frames = FileUDFContract("fixture", (), (frame_type,)).scalar_outputs_to_array(
        [
            [
                {"frame_index": 7, "frame_time": 0.75, "data": pixels},
            ]
        ]
    )
    pcm = vane.tensor_array([samples], vane.tensor_type(vane.sqltypes.FLOAT, (None, None)))
    audio = pa.StructArray.from_arrays([pa.array([16000], pa.int32()), pcm], names=["sample_rate", "data"])
    parts = (TEXT, EmbeddingPartSpec("image"), EmbeddingPartSpec("video"), EmbeddingPartSpec("audio"))
    spec = MultimodalInputSpec(
        frozenset(parts),
        video=VideoInputSpec(),
        audio=AudioInputSpec(sample_rate=16000, max_samples=8),
    )
    rows = content_from_columns(
        pa.table({"part_0": ["text"], "part_1": image, "part_2": frames, "part_3": audio}), parts, spec
    )
    assert len(rows) == 1
    text, image_part, video_part, audio_part = rows[0].parts
    assert text.value == "text"
    np.testing.assert_array_equal(image_part.value, pixels)
    np.testing.assert_array_equal(video_part.value.frames[0], pixels)
    assert video_part.value.frame_times == (0.75,) and video_part.value.frame_indices == (7,)
    np.testing.assert_array_equal(audio_part.value.samples, samples)
    assert audio_part.value.sample_rate == 16000

    class Encoder:
        async def embed_multimodal(self, contents):
            for content in contents:
                text, image, video, audio = content.parts
                assert text.value == "text"
                np.testing.assert_array_equal(image.value, pixels)
                np.testing.assert_array_equal(video.value.frames[0], pixels)
                assert video.value.frame_times == (0.75,) and video.value.frame_indices == (7,)
                np.testing.assert_array_equal(audio.value.samples, samples)
                assert audio.value.sample_rate == 16000
            return [[4, 0.75] for _ in contents]

    class Descriptor(OptionalContentDescriptor):
        def instantiate(self):
            return Encoder()

    class DecodedProvider(ContentProvider):
        def get_multimodal_embedder(self, model=None, dimensions=None, *, options=None):
            return Descriptor()

    with vane.connect() as conn:
        relation = conn.from_arrow(pa.table({"text": ["text"], "image": image, "video": frames, "audio": audio}))
        inputs = MultimodalInput(
            [
                TextInput(vane.col("text")),
                ImageInput(vane.col("image")),
                VideoInput(vane.col("video")),
                AudioInput(vane.col("audio")),
            ]
        )
        assert relation.embed(inputs, provider=DecodedProvider(), execution_backend=backend).select(
            "embedding"
        ).fetchall() == [((4, 0.75),)]


@pytest.mark.parametrize("value", ["file.mp4", "data:video/mp4;base64,AA==", 42])
def test_encoded_parts_require_bytes_not_paths_urls_or_coerced_scalars(value):
    with pytest.raises(TypeError, match="BLOB"):
        content_from_columns(pa.table({"part_0": [value]}), (VIDEO,), SPEC)


@pytest.mark.parametrize("part", [AudioInput, ImageInput, TextInput, VideoInput])
def test_input_requires_expression(part):
    with pytest.raises(TypeError, match="Expression"):
        part("filename or literal")


@pytest.mark.parametrize("mime", ["image/png", "video/", "video/mp4; codecs=x", "video/mp4 ", 42])
def test_video_mime_is_explicit_and_canonical(mime):
    with pytest.raises(ValueError, match="MIME"):
        VideoInput(vane.col("data"), mime_type=mime)


def test_layout_is_immutable_and_unsupported_optional_parts_fail_during_planning(monkeypatch):
    parts = [TextInput(vane.col("text")), VideoInput(vane.col("video"), mime_type="video/mp4")]
    content = MultimodalInput(parts)
    parts.clear()
    assert len(content.parts) == 2
    monkeypatch.setattr(ContentDescriptor, "instantiate", lambda self: pytest.fail("planning instantiated a model"))
    assert isinstance(embed(content, provider=ContentProvider()), vane.Expression)
    with pytest.raises(EmbeddingConfigurationError, match="does not support image"):
        embed(MultimodalInput([ImageInput(vane.col("always_null"))]), provider=ContentProvider())
    with pytest.raises(ValueError, match="multimodal embedding"):
        embed(content, provider="openai", model="text-embedding-3-small")


@pytest.mark.parametrize("parts", [[], (), None, "text"])
def test_empty_or_untyped_part_container_is_rejected(parts):
    with pytest.raises(ValueError, match="nonempty"):
        MultimodalInput(parts)


def test_untyped_part_is_rejected():
    with pytest.raises(TypeError, match="typed|TextInput"):
        MultimodalInput([vane.col("text")])


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_public_relation_batches_keep_input_identity(backend):
    with vane.connect() as conn:
        relation = conn.from_arrow(
            pa.table(
                {
                    "id": [0, 1, 2, 3],
                    "text": ["first", " ", None, "last"],
                    "video": [b"a", b"bc", b"d", None],
                }
            )
        )
        inputs = MultimodalInput(
            [
                TextInput(vane.col("text")),
                VideoInput(vane.col("video"), mime_type="video/mp4"),
            ]
        )
        result = relation.embed(inputs, provider=ContentProvider(), batch_size=2, execution_backend=backend)
        assert result.select("id", "embedding").order("id").fetchall() == [
            (0, (6, 2)),
            (1, (2, 1)),
            (2, (1, 1)),
            (3, (4, 1)),
        ]


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
@pytest.mark.parametrize(
    "kind,mime,typed",
    [
        ("text", None, False),
        ("image", None, False),
        ("video", None, False),
        ("audio", None, False),
        ("image", "image/png", False),
        ("audio", "audio/wav", False),
        ("video", "video/mp4", False),
        ("video", "video/mp4", True),
    ],
)
def test_constant_null_part_is_omitted_without_nulling_valid_rows(backend, kind, mime, typed):
    value = vane.ConstantExpression(None)
    if typed:
        value = value.cast("BLOB")
    part_type = {"text": TextInput, "image": ImageInput, "video": VideoInput, "audio": AudioInput}[kind]
    missing = part_type(value) if mime is None else part_type(value, mime_type=mime)
    with vane.connect() as conn:
        relation = conn.from_arrow(pa.table({"id": [0, 1], "text": ["hello", "world!"]}))
        for text, second_length in ((vane.col("text"), 6), (vane.ConstantExpression("hello"), 5)):
            result = relation.embed(
                MultimodalInput([TextInput(text), missing]),
                provider=OptionalContentProvider(),
                batch_size=2,
                execution_backend=backend,
                output_column="vector",
            )
            assert result.select("id", "vector").order("id").fetchall() == [(0, (5, 1)), (1, (second_length, 1))]


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
@pytest.mark.parametrize("on_error", ["raise", "ignore"])
@pytest.mark.parametrize("joint", [False, True])
def test_all_constant_null_parts_still_raise(backend, on_error, joint):
    missing = VideoInput(vane.ConstantExpression(None), mime_type="video/mp4")
    content = MultimodalInput([TextInput(vane.ConstantExpression(None)), missing]) if joint else missing
    with vane.connect() as conn:
        with pytest.raises(Exception, match="no nonempty parts"):
            conn.sql("SELECT 1 AS id").embed(
                content, provider=ContentProvider(), execution_backend=backend, on_error=on_error
            ).fetchall()


def test_wrapper_serialization_omits_client_and_loop_and_closes_on_same_loop():
    closed = []
    loop = asyncio.new_event_loop()
    wrapper = _EmbedMultimodalBatch(ContentDescriptor(), "parts", "vector", 2, parts=(TEXT,))
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        assert wrapper(packed_parts(pa.table({"part_0": ["ok"]})))["vector"].to_pylist() == [[2, 1]]

        async def close():
            closed.append(asyncio.get_running_loop())

        wrapper._embedder.aclose = close
        restored = pickle.loads(pickle.dumps(wrapper))
        assert restored._embedder is None and restored._run_async is None
        wrapper.close()
        wrapper.close()
        assert closed == [loop]
    finally:
        wrapper.close()
        loop.close()


def test_removed_public_entry_points_have_no_compatibility_aliases():
    import vane.ai.functions as functions

    for name in ("embed_image", "embed_video", "embed_audio"):
        assert not hasattr(vane.ai, name)
        assert not hasattr(functions, name)
        assert not hasattr(vane.Relation, name)


def test_google_encoded_media_and_text_remain_ordered_and_byte_exact():
    pytest.importorskip("google.genai")
    from vane.ai.providers._google_multimodal import embedding_content

    row = EmbeddingContent(
        (
            EmbeddingPart(EmbeddingPartSpec("video", "video/mp4"), b"original mp4 bytes"),
            EmbeddingPart(TEXT, "caption"),
            EmbeddingPart(EmbeddingPartSpec("audio", "audio/wav"), b"original wav bytes"),
            EmbeddingPart(EmbeddingPartSpec("image", "image/jpeg"), b"original jpeg bytes"),
        )
    )
    parts = embedding_content(row).parts
    assert parts[1].text == "caption"
    assert [(part.inline_data.mime_type, part.inline_data.data) for part in parts if part.inline_data] == [
        ("video/mp4", b"original mp4 bytes"),
        ("audio/wav", b"original wav bytes"),
        ("image/jpeg", b"original jpeg bytes"),
    ]


def test_google_combines_frame_and_image_budget():
    pytest.importorskip("google.genai")
    pytest.importorskip("PIL.Image")
    from vane.ai import VideoClip
    from vane.ai.providers._google_multimodal import embedding_content

    image = np.zeros((2, 2, 3), np.uint8)
    video = VideoClip((image,) * 6, tuple(float(i) for i in range(6)), tuple(range(6)))
    with pytest.raises(ValueError, match="six"):
        embedding_content(
            EmbeddingContent(
                (
                    EmbeddingPart(EmbeddingPartSpec("video"), video),
                    EmbeddingPart(EmbeddingPartSpec("image"), image),
                )
            )
        )


def test_google_serialized_base64_budget_is_enforced():
    pytest.importorskip("google.genai")
    from vane.ai.providers._google_multimodal import MAX_REQUEST_BYTES, embedding_content

    # Raw bytes fit; Base64 plus JSON does not.
    with pytest.raises(ValueError, match="encoded request byte budget"):
        embedding_content(EmbeddingContent((EmbeddingPart(VIDEO, b"x" * (MAX_REQUEST_BYTES * 3 // 4)),)))
