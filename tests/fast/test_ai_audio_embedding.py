# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Audio transport, public Python/SQL entry points, and distributed execution."""

from __future__ import annotations

import asyncio
import pickle
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pytest

import vane
from vane.ai import AudioInputSpec, embed, embed_audio
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai.functions import _EmbedAudioBatch
from vane.ai.protocols import AudioEmbedderDescriptor
from vane.ai.provider import Provider
from vane.ai.providers._clap import CLAP_MODEL
from vane.ai.providers.transformers import TransformersProvider


class AudioEmbedder:
    def embed_audio(self, clips):
        return [np.array([clip.samples[0, 0], len(clip.samples), clip.sample_rate]) for clip in clips]


@dataclass
class AudioDescriptor(AudioEmbedderDescriptor):
    model: str = "normal"

    def get_provider(self):
        return "audio_fixture"

    def get_model(self):
        return self.model

    def get_options(self):
        return {}

    def get_dimensions(self):
        return 3

    def get_input_spec(self):
        return AudioInputSpec(sample_rate=48000, max_samples=4, max_input_bytes=64)

    def instantiate(self):
        if self.model == "must_not_load":
            raise EmbeddingConfigurationError("model loaded for NULL or planning")
        return AudioEmbedder()


class AudioProvider(Provider):
    @property
    def name(self):
        return "audio_fixture"

    def get_audio_embedder(self, model=None, dimensions=None, *, options=None):
        return AudioDescriptor(model or "normal")


@pytest.fixture
def provider(monkeypatch):
    from vane.ai.provider import PROVIDERS

    monkeypatch.setitem(PROVIDERS, "audio_fixture", lambda name=None: AudioProvider())
    return AudioProvider()


def audio(value=0.25, *, shape=(3, 2), rate=48000):
    return {"sample_rate": rate, "data": np.full(shape, value, dtype=np.float64)}


def array(values, *, fixed=False, uppercase=False):
    decoded = [None if row is None else row["data"] for row in values]
    if fixed:
        tensor = pa.fixed_shape_tensor(pa.float64(), [3, 2])
        data = pa.ExtensionArray.from_storage(
            tensor,
            pa.array([None if row is None else row.ravel().tolist() for row in decoded], type=tensor.storage_type),
        )
    else:
        data = vane.tensor_array(decoded, vane.tensor_type(vane.sqltypes.DOUBLE, (None, None)))
    names = ["sample_rate", "data"]
    columns = [pa.array([None if row is None else row["sample_rate"] for row in values], type=pa.int64()), data]
    if uppercase:
        names = [name.upper() for name in reversed(names)]
        columns.reverse()
    return pa.StructArray.from_arrays(columns, names=names, mask=pa.array([row is None for row in values], pa.bool_()))


def drive(values, model="normal"):
    wrapper = _EmbedAudioBatch(AudioDescriptor(model), "audio", "embedding", 3, on_error="ignore")
    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        return wrapper(pa.table({"audio": values}))["embedding"].to_pylist()
    finally:
        wrapper.close()
        loop.close()


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("uppercase", [False, True])
def test_tensor_samples_preserve_nulls_chunks_and_offsets(fixed, uppercase):
    samples = array([audio(1), None, audio(), audio(0.5)], fixed=fixed, uppercase=uppercase)
    values = pa.chunked_array([samples.slice(1, 2), samples.slice(3)])
    assert drive(values) == [None, [0.25, 3, 48000], [0.5, 3, 48000]]
    assert drive(array([None, None], fixed=fixed), "must_not_load") == [None, None]
    assert drive(array([], fixed=fixed), "must_not_load") == []


@pytest.mark.parametrize("container", ["struct", "list", "chunked"])
def test_nested_sliced_audio_arrow_transport(container, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    clips = array([audio(1), None, audio(0.25), audio(0.5)]).slice(1)
    if container == "struct":
        column = pa.StructArray.from_arrays([clips], names=["clip"]).slice(1)
    elif container == "list":
        column = pa.ListArray.from_arrays([0, 1, 3], clips).slice(1)
    else:
        column = pa.chunked_array([clips.slice(0, 2), clips.slice(2)])
    table = pa.table({"value": column})
    with vane.connect() as conn:
        assert conn.from_arrow(table).to_arrow_table().to_pylist() == table.to_pylist()


@pytest.mark.parametrize(
    "value",
    [
        audio(rate=16000),
        audio(shape=(5, 1)),
        audio(shape=(0, 1)),
        audio(shape=(2, 3)),
        audio(float("nan")),
        audio(float("inf")),
        audio(1.01),
        {"sample_rate": 48000, "data": None},
    ],
)
def test_input_contract_is_not_silenced_by_ignore(value):
    with pytest.raises(EmbeddingConfigurationError):
        drive(array([value]), "must_not_load")


@pytest.mark.parametrize("runner", ["local-fast", pytest.param("ray", marks=pytest.mark.real_ray)])
@pytest.mark.parametrize("entry", ["expression", "relation", "method", "sql"])
@pytest.mark.parametrize("nulls", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_public_audio_embedding(request, monkeypatch, provider, runner, entry, nulls, fixed):
    if runner == "ray":
        request.getfixturevalue("ray_local")
        monkeypatch.delenv("VANE_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VANE_RUNNER", "local-fast")
    model = "must_not_load" if nulls else "normal"
    samples = array([audio(), None, None, None] if nulls else [audio(), None, audio(), audio(0.5)], fixed=fixed).slice(
        1
    )
    with vane.connect() as conn:
        conn.register("clips", pa.table({"id": range(3), "audio": samples}))
        rel = conn.table("clips")
        options = dict(provider=provider, model=model, batch_size=2, max_retries=0)
        if entry == "expression":
            result = rel.select(vane.col("id"), embed_audio(vane.col("audio"), **options).alias("embedding"))
        elif entry == "relation":
            result = embed_audio(rel, vane.col("audio"), **options).select("id", "embedding")
        elif entry == "method":
            result = rel.embed_audio(vane.col("audio"), **options).select("id", "embedding")
        else:
            result = conn.sql(
                "SELECT id, ai_embed_audio(audio, provider => 'audio_fixture', "
                f"model => '{model}', options => {{batch_size: 2, max_retries: 0}}) AS embedding FROM clips"
            )
        assert result.types[-1] == vane.array_type(vane.sqltypes.FLOAT, 3)
        if runner == "ray":
            assert "ray_actor" in result.explain()
        expected = (
            [(0, None), (1, None), (2, None)] if nulls else [(0, None), (1, (0.25, 3, 48000)), (2, (0.5, 3, 48000))]
        )
        assert result.order("id").fetchall() == expected


@pytest.mark.parametrize(
    "value",
    [
        "'a.wav'",
        "audio_file('missing')",
        "{sample_rate: 48000, data: [0.1]}",
        "{sample_rate: '48000', data: tensor([0.1]::FLOAT[], [1, 1])}",
    ],
)
def test_undecoded_or_untyped_inputs_fail_binding(provider, value):
    with vane.connect() as conn:
        with pytest.raises(Exception, match="ai_embed_audio requires"):
            conn.sql(f"SELECT ai_embed_audio({value}, provider => 'audio_fixture')").fetchall()


def test_null_and_explain_do_not_load_models(provider):
    with vane.connect() as conn:
        assert conn.sql(
            "SELECT ai_embed_audio(NULL, provider => 'audio_fixture', model => 'must_not_load')"
        ).fetchall() == [(None,)]
        conn.sql(
            "EXPLAIN SELECT ai_embed_audio(NULL, provider => 'audio_fixture', model => 'must_not_load')"
        ).fetchall()


def test_paired_clap_metadata_does_not_import_model_libraries(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    provider = TransformersProvider()
    for factory in (provider.get_audio_embedder, provider.get_text_embedder):
        descriptor = pickle.loads(pickle.dumps(factory(CLAP_MODEL, options={"device": "cuda"})))
        assert descriptor.get_dimensions() == 512
        assert descriptor.get_udf_options().num_gpus == 1
        assert descriptor.get_model() == CLAP_MODEL
    assert provider.get_audio_embedder(CLAP_MODEL).get_input_spec().max_samples == 480000
    assert not provider.get_text_embedder(CLAP_MODEL).supports_chunking()


def test_no_fallback_or_chunk_averaging(provider):
    with pytest.raises(ValueError, match="audio embedding provider"):
        embed_audio(vane.col("audio"), provider="openai")
    with pytest.raises(ValueError, match="requires model"):
        TransformersProvider().get_audio_embedder("unknown")
    with pytest.raises(ValueError, match="512"):
        TransformersProvider().get_audio_embedder(CLAP_MODEL, dimensions=256)
    with vane.connect() as conn:
        with pytest.raises(EmbeddingConfigurationError, match="chunk averaging"):
            embed(
                conn.sql("SELECT 'dog' AS text"),
                vane.col("text"),
                provider="transformers",
                model=CLAP_MODEL,
                max_chunk_chars=20,
                chunk_overlap_chars=0,
            )


@pytest.mark.parametrize(
    "options",
    [
        {"overlength": "truncate"},
        {"device": "cuda:1"},
        {"dtype": "float16"},
        {"trust_remote_code": True, "revision": "a" * 40},
    ],
)
def test_clap_rejects_unsupported_configuration(options):
    for factory in (TransformersProvider().get_audio_embedder, TransformersProvider().get_text_embedder):
        with pytest.raises((ValueError, TypeError)):
            factory(CLAP_MODEL, options=options)
