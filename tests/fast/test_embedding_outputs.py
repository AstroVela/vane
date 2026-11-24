# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Native outputs preserve row identity without extra per-part model calls."""

from __future__ import annotations

import asyncio
import pickle
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pytest

import vane
from vane.ai import EmbeddingOutput, EmbeddingOutputSpec, MultimodalInput, TextInput, VideoInput, embed
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec, MultimodalInputSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai.functions import _EmbedMultimodalBatch
from vane.ai.protocols import MultimodalEmbedderDescriptor, TextEmbedderDescriptor
from vane.ai.provider import Provider

TEXT = EmbeddingPartSpec("text")
VIDEO = EmbeddingPartSpec("video", "video/mp4")
INPUT = MultimodalInputSpec(frozenset({TEXT, VIDEO}), max_input_bytes=1024)
OUTPUT = EmbeddingOutputSpec(
    (
        EmbeddingOutput("joint", 2, "fixture/joint/v1", ("text", "video")),
        EmbeddingOutput("motion", 3, "fixture/motion/v2", ("video",), optional=True),
    ),
    layout="named",
)


class Encoder:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    async def embed_text(self, texts):
        return await self.embed_multimodal([EmbeddingContent((EmbeddingPart(TEXT, text),)) for text in texts])

    async def embed_multimodal(self, inputs):
        self.calls.append(inputs)
        result = []
        for row in inputs:
            kinds = {part.spec.kind for part in row.parts}
            if self.failure == "oom":
                raise MemoryError("private-model-content")
            value = {"joint": [len(row.parts), 1.0], "motion": [1.0, 2.0, 2.0] if "video" in kinds else None}
            if any(part.value == "bad" for part in row.parts):
                if self.failure == "dimension":
                    value["motion"] = [1.0]
                elif self.failure == "nonfinite":
                    value["joint"] = [float("nan"), 1.0]
                elif self.failure == "missing":
                    value.pop("motion")
                elif self.failure == "extra":
                    value["extra"] = [1.0]
                elif self.failure == "required":
                    value["joint"] = None
                elif self.failure == "empty":
                    value = {"joint": None, "motion": None}
                elif self.failure == "source":
                    value["motion"] = [1.0, 2.0, 2.0]
            result.append(value)
        if self.failure == "count":
            return result[:-1]
        return result


@dataclass
class Descriptor(MultimodalEmbedderDescriptor):
    failure: str | None = None

    def get_provider(self):
        return "native_outputs_fixture"

    def get_model(self):
        return "fixture/v1"

    def get_options(self):
        return {}

    def get_input_spec(self):
        return INPUT

    def get_output_spec(self, *, dimensions=None):
        return OUTPUT

    def instantiate(self):
        return Encoder(self.failure)


class OutputProvider(Provider):
    def __init__(self, failure=None):
        self.failure = failure

    @property
    def name(self):
        return "native_outputs_fixture"

    def get_multimodal_embedder(self, model=None, dimensions=None, *, options=None):
        return Descriptor(self.failure)


class TextOutputDescriptor(TextEmbedderDescriptor, Descriptor):
    pass


class TextOutputProvider(OutputProvider):
    def get_text_embedder(self, model=None, dimensions=None, *, options=None):
        return TextOutputDescriptor(self.failure)


def drive(wrapper, table, calls=None):
    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        return wrapper(table)
    finally:
        if calls is not None and wrapper._embedder is not None:
            calls.extend(wrapper._embedder.calls)
        wrapper.close()
        loop.close()


def wrapper(failure=None, **options):
    return _EmbedMultimodalBatch(
        Descriptor(failure),
        "parts",
        "embedding",
        None,
        parts=(TEXT, VIDEO),
        output_spec=OUTPUT,
        max_retries=0,
        **options,
    )


def packed(texts, videos):
    return pa.table(
        {
            "parts": pa.table(
                {"part_0": pa.array(texts, pa.string()), "part_1": pa.array(videos, pa.binary())}
            ).to_struct_array()
        }
    )


def test_contract_is_immutable_serializable_and_has_closed_types():
    assert pickle.loads(pickle.dumps(OUTPUT)) == OUTPUT
    assert OUTPUT.sql_type() == 'STRUCT("joint" FLOAT[2], "motion" FLOAT[3])'
    assert OUTPUT.arrow_type() == pa.struct(
        [("joint", pa.list_(pa.float32(), 2)), ("motion", pa.list_(pa.float32(), 3))]
    )
    single = EmbeddingOutputSpec((EmbeddingOutput("embedding", 7, "paired/model/v1"),))
    assert single.sql_type() == "FLOAT[7]"
    assert single.arrow_type() == pa.list_(pa.float32(), 7)


@pytest.mark.parametrize(
    "change",
    [
        {"name": 'bad"name'},
        {"name": ""},
        {"dimensions": True},
        {"dimensions": 0},
        {"dimensions": 1.5},
        {"space": " "},
        {"space": "x" * 1025},
        {"source_modalities": ()},
        {"source_modalities": ("text", "text")},
        {"source_modalities": ("speech",)},
        {"source_modalities": ["text"]},
        {"optional": 1},
    ],
)
def test_invalid_output_declarations_fail_without_loading_models(change):
    with pytest.raises(EmbeddingConfigurationError):
        EmbeddingOutput(**({"name": "valid", "dimensions": 2, "space": "v1"} | change))


@pytest.mark.parametrize(
    "outputs,layout",
    [
        ((), "named"),
        ([OUTPUT.outputs[0]], "named"),
        (OUTPUT.outputs, "vector"),
        ((OUTPUT.outputs[1],), "vector"),
        ((OUTPUT.outputs[0],), "unknown"),
        ((OUTPUT.outputs[0], EmbeddingOutput("JOINT", 2, "different")), "named"),
    ],
)
def test_invalid_result_shapes_fail_at_planning(outputs, layout):
    with pytest.raises(EmbeddingConfigurationError):
        EmbeddingOutputSpec(outputs, layout=layout)


def test_rows_and_optional_outputs_do_not_split_or_repeat_model_calls():
    batch = wrapper()
    calls = []
    result = drive(batch, packed(["one", "two", None], [b"clip", None, b"other"]), calls)
    assert result.column("embedding").to_pylist() == [
        {"joint": [2.0, 1.0], "motion": [1.0, 2.0, 2.0]},
        {"joint": [1.0, 1.0], "motion": None},
        {"joint": [1.0, 1.0], "motion": [1.0, 2.0, 2.0]},
    ]
    assert len(calls) == 1
    assert len(calls[0]) == 3
    assert all(isinstance(value, EmbeddingContent) for value in calls[0])


def test_normalization_applies_to_each_native_output_independently():
    result = drive(wrapper(normalize=True), packed(["one"], [b"clip"]))["embedding"].to_pylist()[0]
    assert result["motion"] == pytest.approx([1 / 3, 2 / 3, 2 / 3])
    assert np.linalg.norm(result["joint"]) == pytest.approx(1.0)


@pytest.mark.parametrize("failure", ["dimension", "nonfinite", "missing", "extra", "required", "empty", "source"])
@pytest.mark.parametrize("policy", ["raise", "ignore"])
def test_malformed_native_outputs_remain_attributable_to_one_row(failure, policy):
    batch = wrapper(failure, on_error=policy)
    calls = []
    videos = [b"clip", None if failure == "source" else b"clip"]
    if policy == "raise":
        with pytest.raises(TypeError):
            drive(batch, packed(["good", "bad"], videos), calls)
    else:
        values = drive(batch, packed(["good", "bad"], videos), calls)["embedding"].to_pylist()
        assert values[0]["joint"] == [2.0, 1.0]
        assert values[1] is None
    assert len(calls) == 1


@pytest.mark.parametrize("policy", ["raise", "ignore"])
def test_resource_failures_still_abort_and_are_sanitized(policy):
    with pytest.raises(MemoryError) as caught:
        drive(wrapper("oom", on_error=policy), packed(["one"], [b"clip"]))
    assert "private-model-content" not in str(caught.value)


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_public_expression_and_relation_preserve_native_structs_and_selection(backend):
    with vane.connect() as connection:
        relation = connection.from_arrow(
            pa.table(
                {"text": ["one", "two", "three"], "video": [b"clip", None, b"other"], "selected": [True, True, False]}
            )
        )
        inputs = MultimodalInput([TextInput(vane.col("text")), VideoInput(vane.col("video"), "video/mp4")])
        result = embed(
            relation,
            inputs,
            provider=OutputProvider(),
            where=vane.col("selected"),
            execution_backend=backend,
            batch_size=3,
        )
        assert str(result.types[-1]) == "STRUCT(joint FLOAT[2], motion FLOAT[3])"
        values = result.to_arrow_table()["embedding"].to_pylist()
        assert values == [{"joint": [2.0, 1.0], "motion": [1.0, 2.0, 2.0]}, {"joint": [1.0, 1.0], "motion": None}, None]
        expression = embed(inputs, provider=OutputProvider())
        assert relation.select(expression.alias("native")).to_arrow_table()["native"].to_pylist()[2]["motion"] == [
            1.0,
            2.0,
            2.0,
        ]


def test_named_outputs_reject_global_dimensions_before_instantiation():
    inputs = MultimodalInput([TextInput(vane.col("text"))])
    with pytest.raises(EmbeddingConfigurationError, match="omit dimensions"):
        embed(inputs, provider=OutputProvider(), dimensions=2)


def test_sql_uses_the_same_named_output_type_and_row_mapping(monkeypatch):
    from vane.ai.provider import PROVIDERS

    monkeypatch.setitem(PROVIDERS, "native_outputs_fixture", lambda name=None: OutputProvider())
    with vane.connect() as connection:
        relation = connection.sql(
            "SELECT ai_embed({text: value, video: clip}, "
            "input_spec => [{kind: 'text'}, {kind: 'video', mime_type: 'video/mp4'}], "
            "provider => 'native_outputs_fixture', options => {batch_size: 2}, \"where\" => selected) AS result "
            "FROM (VALUES ('one', 'clip'::BLOB, true), ('two', NULL::BLOB, true), ('three', 'other'::BLOB, false)) "
            "AS source(value, clip, selected)"
        )
        assert str(relation.types[0]) == "STRUCT(joint FLOAT[2], motion FLOAT[3])"
        assert relation.to_arrow_table()["result"].to_pylist() == [
            {"joint": [2.0, 1.0], "motion": [1.0, 2.0, 2.0]},
            {"joint": [1.0, 1.0], "motion": None},
            None,
        ]


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_empty_selection_preserves_named_type_without_constructing_the_encoder(backend):
    class Never(Descriptor):
        def instantiate(self):
            raise AssertionError("unselected provider must not be instantiated")

    class NeverProvider(OutputProvider):
        def get_multimodal_embedder(self, model=None, dimensions=None, *, options=None):
            return Never()

    with vane.connect() as connection:
        relation = connection.sql("SELECT 'one'::VARCHAR AS text, NULL::BLOB AS video")
        result = embed(
            relation,
            MultimodalInput([TextInput(vane.col("text")), VideoInput(vane.col("video"), "video/mp4")]),
            provider=NeverProvider(),
            where=vane.ConstantExpression(False),
            execution_backend=backend,
        )
        assert result.to_arrow_table()["embedding"].to_pylist() == [None]
        assert str(result.types[-1]) == "STRUCT(joint FLOAT[2], motion FLOAT[3])"


def test_named_output_serialization_retains_contract_and_reconstructs_runtime():
    batch = wrapper()
    assert drive(batch, packed(["one"], [b"clip"]))["embedding"].to_pylist()[0]["joint"] == [2.0, 1.0]
    restored = pickle.loads(pickle.dumps(batch))
    assert restored._output_spec == OUTPUT
    assert restored._embedder is None and restored._run_async is None
    assert drive(restored, packed(["two"], [None]))["embedding"].to_pylist() == [{"joint": [1.0, 1.0], "motion": None}]


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
@pytest.mark.parametrize("policy", ["raise", "ignore"])
def test_public_named_outputs_keep_oom_fatal_and_sanitized(backend, policy):
    with vane.connect() as connection:
        relation = connection.sql("SELECT 'one'::VARCHAR AS text")
        result = embed(
            relation,
            MultimodalInput([TextInput(vane.col("text"))]),
            provider=OutputProvider("oom"),
            execution_backend=backend,
            on_error=policy,
            max_retries=0,
        )
        with pytest.raises(Exception) as caught:
            result.fetchall()
        assert "MemoryError" in str(caught.value)
        assert "private-model-content" not in str(caught.value)


def test_output_row_cardinality_violation_is_explicit():
    with pytest.raises(TypeError, match="row count and order"):
        drive(wrapper("count"), packed(["one", "two"], [b"clip", b"other"]))


def test_all_optional_null_outputs_are_not_a_successful_result():
    output = EmbeddingOutputSpec((EmbeddingOutput("optional", 2, "v1", optional=True),), layout="named")
    batch = wrapper()
    batch._output_spec = output
    with pytest.raises(TypeError, match="no native embedding output"):
        batch._coerce_result({"optional": None})


def test_existing_builtin_encoders_declare_one_native_vector_without_model_io():
    from vane.ai.providers.google import GoogleProvider
    from vane.ai.providers.openai import OpenAIProvider
    from vane.ai.providers.sglang import SGLangProvider
    from vane.ai.providers.vllm import VLLMProvider

    descriptors = [
        OpenAIProvider().get_text_embedder("text-embedding-3-small", 128),
        GoogleProvider().get_text_embedder("gemini-embedding-001", 128),
        GoogleProvider().get_multimodal_embedder("gemini-embedding-2", 128),
        *[
            provider().get_multimodal_embedder(
                "served-model",
                128,
                options={
                    "runtime": "http",
                    "base_url": "http://embedding.invalid/v1",
                    "http_input_format": "multimodal",
                    "media_mime_types": ["video/mp4"],
                },
            )
            for provider in (SGLangProvider, VLLMProvider)
        ],
    ]
    for descriptor in descriptors:
        output = descriptor.get_output_spec()
        assert output.layout == "vector" and len(output.outputs) == 1
        assert output.outputs[0].dimensions == 128
        assert output.outputs[0].space == descriptor.get_model()
        assert pickle.loads(pickle.dumps(output)) == output


def test_named_text_outputs_reject_averaging_without_a_model_call():
    with vane.connect() as connection:
        with pytest.raises(EmbeddingConfigurationError, match="chunk averaging"):
            embed(
                connection.sql("SELECT 'text' AS text"),
                vane.col("text"),
                provider=TextOutputProvider(),
                max_chunk_chars=1024,
            )


@pytest.mark.parametrize("policy", ["raise", "ignore"])
def test_named_outputs_validate_the_final_normalized_vector(monkeypatch, policy):
    from vane.ai import functions

    monkeypatch.setattr(functions, "_normalize_embedding", lambda value: np.full_like(value, np.inf))
    batch = wrapper(normalize=True, on_error=policy)
    if policy == "raise":
        with pytest.raises(TypeError, match="non-finite"):
            drive(batch, packed(["one"], [b"clip"]))
    else:
        assert drive(batch, packed(["one"], [b"clip"]))["embedding"].to_pylist() == [None]


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_unary_text_named_outputs_keep_null_rows_and_query_space(backend):
    with vane.connect() as connection:
        relation = connection.from_arrow(pa.table({"text": pa.array(["one", None, "two"], pa.string())}))
        result = embed(relation, TextInput(vane.col("text")), provider=TextOutputProvider(), execution_backend=backend)
        assert result.to_arrow_table()["embedding"].to_pylist() == [
            {"joint": [1.0, 1.0], "motion": None},
            None,
            {"joint": [1.0, 1.0], "motion": None},
        ]
        assert TextOutputDescriptor().get_output_spec().outputs[0].space == OUTPUT.outputs[0].space


@pytest.mark.parametrize("api", ["python", "sql"])
def test_constant_null_with_named_outputs_does_not_require_a_gpu_worker(api, monkeypatch):
    from vane.ai.provider import PROVIDERS
    from vane.ai.typing import UDFOptions

    class GPUDescriptor(TextOutputDescriptor):
        def get_udf_options(self):
            return UDFOptions(num_gpus=1)

        def instantiate(self):
            raise AssertionError("constant NULL must not instantiate the model")

    class GPUProvider(TextOutputProvider):
        def get_text_embedder(self, model=None, dimensions=None, *, options=None):
            return GPUDescriptor()

    monkeypatch.setitem(PROVIDERS, "named_gpu_fixture", lambda name=None: GPUProvider())
    with vane.connect() as connection:
        result = (
            connection.sql("SELECT 1 AS seed").select(
                embed(TextInput(vane.ConstantExpression(None)), provider=GPUProvider()).alias("embedding")
            )
            if api == "python"
            else connection.sql("SELECT ai_embed(NULL, provider => 'named_gpu_fixture') AS embedding")
        )
        assert str(result.types[0]) == "STRUCT(joint FLOAT[2], motion FLOAT[3])"
        assert result.fetchall() == [(None,)]
        assert "ai_embed" not in result.explain().lower()
