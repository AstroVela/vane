# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for vane.ai high-level functions.

Uses mock models to avoid network/GPU dependencies. Tests verify the full
path: Provider → Descriptor → map_batches wrapper → DuckDB execution.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest

import vane
from vane.ai.protocols import (
    TextClassifierDescriptor,
    TextEmbedderDescriptor,
)
from vane.ai.provider import Provider
from vane.ai.typing import EmbeddingDimensions

if TYPE_CHECKING:
    from vane.ai.protocols import TextClassifier, TextEmbedder
    from vane.ai.typing import Options


def _has_module(name: str) -> bool:
    """Check if a Python module is importable."""
    import importlib

    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def _drive(wrapper, table: pa.Table) -> pa.Table:
    """Call an AI batch wrapper the way a UDF executor drives it.

    Wrappers no longer own event loops: the executor binds a ``run_async``
    capability before the first batch. Tests that invoke wrappers directly
    must do the same.
    """
    import asyncio

    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        return wrapper(table)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Mock implementations
# ---------------------------------------------------------------------------


class MockTextEmbedder:
    """Returns fixed-dimension random embeddings."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed_text(self, text: list[str]) -> list[np.ndarray]:
        return [np.ones(self.dim, dtype=np.float32) * len(t) for t in text]


@dataclass
class MockTextEmbedderDescriptor(TextEmbedderDescriptor):
    dim: int = 4

    def get_provider(self) -> str:
        return "mock"

    def get_model(self) -> str:
        return "mock-embedder"

    def get_options(self) -> Options:
        return {"batch_size": 2}

    def get_dimensions(self) -> EmbeddingDimensions:
        return EmbeddingDimensions(size=self.dim, dtype=pa.float32())

    def instantiate(self) -> TextEmbedder:
        return MockTextEmbedder(dim=self.dim)


class MockTextClassifier:
    """Returns the first label for every input."""

    def classify_text(self, text: list[str], labels: list[str]) -> list[str]:
        return [labels[0] for _ in text]


@dataclass
class MockTextClassifierDescriptor(TextClassifierDescriptor):
    def get_provider(self) -> str:
        return "mock"

    def get_model(self) -> str:
        return "mock-classifier"

    def get_options(self) -> Options:
        return {"batch_size": 2}

    def instantiate(self) -> TextClassifier:
        return MockTextClassifier()


class MockProvider(Provider):
    """Provider that returns mock descriptors."""

    @property
    def name(self) -> str:
        return "mock"

    def get_text_embedder(self, model=None, dimensions=None, *, options=None) -> TextEmbedderDescriptor:
        return MockTextEmbedderDescriptor(dim=dimensions or 4)

    def get_text_classifier(self, model=None, **_options) -> TextClassifierDescriptor:
        return MockTextClassifierDescriptor()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmbed:
    def test_embed_basic(self):
        """embed preserves the source relation and appends an embedding."""
        from vane.ai.functions import embed

        conn = vane.connect()
        rel = conn.sql("SELECT 'hello' AS text UNION ALL SELECT 'world' AS text")

        result = embed(
            rel,
            vane.col("text"),
            provider=MockProvider(),
        )

        rows = result.fetchall()
        assert len(rows) == 2
        # Each embedding should be a list of 4 floats
        for row in rows:
            emb = row[1]
            assert len(emb) == 4

    def test_embed_custom_dimensions(self):
        """embed respects dimensions parameter."""
        from vane.ai.functions import embed

        conn = vane.connect()
        rel = conn.sql("SELECT 'test' AS text")

        result = embed(
            rel,
            vane.col("text"),
            provider=MockProvider(),
            dimensions=8,
        )

        rows = result.fetchall()
        assert len(rows[0][1]) == 8

    def test_embed_custom_output_column(self):
        """embed uses a custom output column name."""
        from vane.ai.functions import embed

        conn = vane.connect()
        rel = conn.sql("SELECT 'test' AS text")

        result = embed(
            rel,
            vane.col("text"),
            provider=MockProvider(),
            output_column="my_emb",
        )

        rows = result.fetchall()
        assert len(rows) == 1
        assert result.columns == ["text", "my_emb"]

    def test_embed_propagates_null(self):
        """embed propagates NULL without sending an empty-string request."""
        from vane.ai.functions import embed

        conn = vane.connect()
        rel = conn.sql("SELECT NULL::VARCHAR AS text")

        result = embed(
            rel,
            vane.col("text"),
            provider=MockProvider(),
        )

        rows = result.fetchall()
        assert len(rows) == 1
        assert rows[0][1] is None


class TestClassifyText:
    def test_classify_text_basic(self):
        """classify_text produces a relation with label column."""
        from vane.ai.functions import classify_text

        conn = vane.connect()
        rel = conn.sql("SELECT 'great product' AS text UNION ALL SELECT 'terrible' AS text")

        result = classify_text(
            rel,
            "text",
            labels=["positive", "negative"],
            provider=MockProvider(),
        )

        rows = result.fetchall()
        assert len(rows) == 2
        # MockTextClassifier always returns the first label
        for row in rows:
            assert row[0] == "positive"

    def test_classify_text_custom_output(self):
        from vane.ai.functions import classify_text

        conn = vane.connect()
        rel = conn.sql("SELECT 'test' AS text")

        result = classify_text(
            rel,
            "text",
            labels=["a", "b"],
            provider=MockProvider(),
            output_column="sentiment",
        )

        rows = result.fetchall()
        assert len(rows) == 1


class TestMockDescriptorPickle:
    """Verify mock descriptors are serializable (requirement for Ray)."""

    def test_embedder_descriptor_pickle(self):
        desc = MockTextEmbedderDescriptor(dim=16)
        restored = pickle.loads(pickle.dumps(desc))
        embedder = restored.instantiate()
        result = embedder.embed_text(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 16

    def test_classifier_descriptor_pickle(self):
        desc = MockTextClassifierDescriptor()
        restored = pickle.loads(pickle.dumps(desc))
        classifier = restored.instantiate()
        result = classifier.classify_text(["test"], ["a", "b"])
        assert result == ["a"]


class TestWrapperPickle:
    """Verify wrapper classes are picklable (critical for Ray execution)."""

    def test_embed_wrapper_pickle(self):
        from vane.ai.functions import _EmbedTextBatch

        wrapper = _EmbedTextBatch(MockTextEmbedderDescriptor(dim=4), "text", "emb", 4)
        restored = pickle.loads(pickle.dumps(wrapper))
        table = pa.table({"text": ["hello", "world"]})
        result = _drive(restored, table)
        assert result.num_rows == 2
        assert result.column_names == ["emb"]

    def test_classify_wrapper_pickle(self):
        from vane.ai.functions import _ClassifyTextBatch

        wrapper = _ClassifyTextBatch(MockTextClassifierDescriptor(), "text", "label", ["a", "b"])
        restored = pickle.loads(pickle.dumps(wrapper))
        table = pa.table({"text": ["hello"]})
        result = restored(table)
        assert result.num_rows == 1
        assert result.column("label").to_pylist() == ["a"]


# ---------------------------------------------------------------------------
# vLLM Provider tests
# ---------------------------------------------------------------------------


class TestGoogleProviderEmbedPlanning:
    def test_get_text_embedder_uses_builtin_default(self):
        from vane.ai.providers.google import GoogleProvider

        descriptor = GoogleProvider().get_text_embedder()

        assert descriptor.model_name == "gemini-embedding-2"
        assert descriptor.get_dimensions().size == 3072

    def test_get_text_embedder_default_rejects_unsupported_request_options(self):
        from vane.ai.providers.google import GoogleProvider

        with pytest.raises(ValueError, match=r"gemini-embedding-2.*task_type"):
            GoogleProvider().get_text_embedder(options={"task_type": "RETRIEVAL_QUERY"})

    @pytest.mark.parametrize("model", ["gemini-embedding-2", "models/gemini-embedding-2"])
    @pytest.mark.parametrize(
        "embed_options",
        [
            {"task_type": "RETRIEVAL_QUERY"},
            {"task_type": "RETRIEVAL_DOCUMENT", "title": "Document title"},
        ],
    )
    def test_embedding_2_rejects_unsupported_request_options_on_direct_descriptor(self, model, embed_options):
        from vane.ai.providers.google import GoogleTextEmbedderDescriptor

        with pytest.raises(ValueError, match=r"gemini-embedding-2.*task_type|title"):
            GoogleTextEmbedderDescriptor(model_name=model, options=embed_options)

    def test_embedding_2_accepts_explicit_null_request_options(self):
        from vane.ai.providers.google import GoogleProvider

        descriptor = GoogleProvider().get_text_embedder(options={"task_type": None, "title": None})

        assert descriptor.model_name == "gemini-embedding-2"

    def test_embedding_dimension_priority_is_call_then_builtin_then_unknown_metadata(self):
        from vane.ai.providers.google import GoogleProvider

        provider = GoogleProvider(embedding_dimensions=512)
        builtin = provider.get_text_embedder()
        explicit = provider.get_text_embedder(dimensions=256)
        unknown = GoogleProvider(
            embedding_model="custom-fixed-model",
            embedding_dimensions=128,
        ).get_text_embedder()

        assert (builtin.get_dimensions().size, builtin.request_dimensions) == (3072, None)
        assert (explicit.get_dimensions().size, explicit.request_dimensions) == (256, 256)
        assert (unknown.get_dimensions().size, unknown.request_dimensions) == (128, None)

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_provider_dimension_metadata_is_not_sent_as_request_override(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock, patch

        from vane.ai.providers.google import GoogleProvider

        async def request_for(descriptor):
            client = MagicMock()
            client.aio.models.embed_content = AsyncMock(
                return_value=SimpleNamespace(
                    embeddings=[SimpleNamespace(values=[1.0] * descriptor.get_dimensions().size)]
                )
            )
            with patch("google.genai.Client", return_value=client):
                embedder = descriptor.instantiate()
            await embedder.embed_text(["row"])
            return client.aio.models.embed_content.await_args.kwargs

        provider = GoogleProvider(embedding_model="custom-fixed-model", embedding_dimensions=4)
        metadata_only = provider.get_text_embedder()
        explicit_override = provider.get_text_embedder(dimensions=3)

        assert metadata_only.get_dimensions().size == 4
        assert "config" not in asyncio.run(request_for(metadata_only))
        explicit_request = asyncio.run(request_for(explicit_override))
        assert explicit_request["config"]["output_dimensionality"] == 3

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_google_embed_disables_sdk_retries(self):
        from unittest.mock import patch

        from vane.ai.providers.google import GoogleTextEmbedder

        with patch("google.genai.Client") as client:
            GoogleTextEmbedder(options={}, model="embedding-model")

        retry_options = client.call_args.kwargs["http_options"].retry_options
        assert retry_options.attempts == 1


class TestGoogleEmbeddingRowPreservation:
    """embed_text must return exactly one embedding per input row.

    gemini-embedding-2 aggregates multiple direct string inputs into a
    single embedding, so each input is sent as its own ``types.Content``
    and the result count is verified against the input count.
    """

    def _make_embedder(self, embeddings):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch

        from vane.ai.providers.google import GoogleTextEmbedder

        with patch("google.genai.Client"):
            embedder = GoogleTextEmbedder(
                options={},
                model="gemini-embedding-2",
            )
        embed_content = AsyncMock(return_value=SimpleNamespace(embeddings=embeddings))
        embedder._client.aio.models.embed_content = embed_content
        return embedder, embed_content

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_two_inputs_yield_two_vectors_via_separate_contents(self):
        import asyncio
        from types import SimpleNamespace

        from google.genai import types

        embedder, embed_content = self._make_embedder(
            [
                SimpleNamespace(values=[0.1, 0.2]),
                SimpleNamespace(values=[0.3, 0.4]),
            ]
        )

        result = asyncio.run(embedder.embed_text(["first row", "second row"]))

        assert len(result) == 2
        contents = embed_content.call_args.kwargs["contents"]
        assert len(contents) == 2
        assert all(isinstance(c, types.Content) for c in contents)
        assert contents[0].parts[0].text == "first row"
        assert contents[1].parts[0].text == "second row"

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_embedding_count_mismatch_raises(self):
        import asyncio
        from types import SimpleNamespace

        embedder, _ = self._make_embedder([SimpleNamespace(values=[0.1, 0.2])])

        with pytest.raises(TypeError, match="preserve row count and order"):
            asyncio.run(embedder.embed_text(["first row", "second row"]))


class TestGoogleEmbeddingBatching:
    """Tests for Google embedding request chunking under the API batch cap."""

    def _recording_embedder(self, model, dimensions=None, options=None):
        """Build a GoogleTextEmbedder around a recording embed_content mock.

        The fake response derives each embedding value from the input text
        (``"txt-<n>"`` embeds to ``[float(n)]``), so result order can be
        asserted independently of request order.
        """
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.google import GoogleTextEmbedder

        calls: list[dict] = []

        async def fake_embed_content(**kwargs):
            calls.append(kwargs)
            response = MagicMock()
            response.embeddings = []
            for item in kwargs["contents"]:
                embedding = MagicMock()
                embedding.values = [float(item.parts[0].text.rsplit("-", 1)[1])]
                response.embeddings.append(embedding)
            return response

        embedder = GoogleTextEmbedder.__new__(GoogleTextEmbedder)
        embedder._client = MagicMock()
        embedder._client.aio.models.embed_content = AsyncMock(side_effect=fake_embed_content)
        embedder._model = model
        embedder._dimensions = dimensions
        embedder._options = dict(options or {})
        return embedder, calls

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_oversized_batch_chunked_under_cap(self):
        """250 inputs become 3 requests of <=100, concatenated in order."""
        import asyncio

        embedder, calls = self._recording_embedder("gemini-embedding-001")
        texts = [f"txt-{i}" for i in range(250)]

        result = asyncio.run(embedder.embed_text(texts))

        assert [len(call["contents"]) for call in calls] == [100, 100, 50]
        assert all(call["model"] == "gemini-embedding-001" for call in calls)
        # Chunks cover the inputs in order without overlap.
        sent = [item.parts[0].text for call in calls for item in call["contents"]]
        assert sent == texts
        # One embedding per input row, in input order.
        assert len(result) == len(texts)
        assert [e[0] for e in result] == [float(i) for i in range(250)]

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    @pytest.mark.parametrize("count", [1, 100])
    def test_batch_at_or_under_cap_is_single_request(self, count):
        """Batches within the cap go out as one request."""
        import asyncio

        embedder, calls = self._recording_embedder("gemini-embedding-001")
        texts = [f"txt-{i}" for i in range(count)]

        result = asyncio.run(embedder.embed_text(texts))

        assert len(calls) == 1
        assert len(calls[0]["contents"]) == count
        assert len(result) == count

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_empty_batch_makes_no_requests(self):
        import asyncio

        embedder, calls = self._recording_embedder("gemini-embedding-001")

        result = asyncio.run(embedder.embed_text([]))

        assert calls == []
        assert result == []

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_aggregating_model_batches_via_separate_contents(self):
        """gemini-embedding-2 aggregates multiple direct string inputs, so
        each input travels as its own Content and full-size chunks stay
        row-preserving without falling back to one request per input."""
        import asyncio

        embedder, calls = self._recording_embedder("gemini-embedding-2")
        texts = ["txt-0", "txt-1", "txt-2"]

        result = asyncio.run(embedder.embed_text(texts))

        assert len(calls) == 1
        assert [item.parts[0].text for item in calls[0]["contents"]] == texts
        assert len(result) == len(texts)
        assert [e[0] for e in result] == [0.0, 1.0, 2.0]

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_result_count_mismatch_raises(self):
        """A model that does not embed inputs individually is rejected."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.google import GoogleTextEmbedder

        async def aggregate_embed_content(**kwargs):
            response = MagicMock()
            embedding = MagicMock()
            embedding.values = [1.0]
            response.embeddings = [embedding]  # one embedding for N inputs
            return response

        embedder = GoogleTextEmbedder.__new__(GoogleTextEmbedder)
        embedder._client = MagicMock()
        embedder._client.aio.models.embed_content = AsyncMock(side_effect=aggregate_embed_content)
        embedder._model = "custom-aggregating-embedder"
        embedder._dimensions = None
        embedder._options = {}

        with pytest.raises(TypeError, match="returned 1 embeddings for 3 inputs"):
            asyncio.run(embedder.embed_text(["a", "b", "c"]))

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_config_forwarded_to_every_chunk(self):
        """Dimensions and request options reach each chunked request."""
        import asyncio

        embedder, calls = self._recording_embedder(
            "gemini-embedding-001",
            dimensions=256,
            options={"task_type": "RETRIEVAL_QUERY"},
        )
        texts = [f"txt-{i}" for i in range(150)]

        asyncio.run(embedder.embed_text(texts))

        assert len(calls) == 2
        for call in calls:
            assert call["config"]["output_dimensionality"] == 256
            assert call["config"]["task_type"] == "RETRIEVAL_QUERY"

    def test_descriptor_only_reports_resource_requirements(self):
        from vane.ai.providers.google import GoogleTextEmbedderDescriptor

        desc = GoogleTextEmbedderDescriptor(model_name="gemini-embedding-001")
        assert desc.get_udf_options().batch_size is None
        assert desc.get_udf_options().num_gpus == 0

    def test_descriptor_rejects_legacy_batch_size_override(self):
        from vane.ai.providers.google import GoogleTextEmbedderDescriptor

        with pytest.raises(TypeError, match="batch_size"):
            GoogleTextEmbedderDescriptor(
                model_name="gemini-embedding-001",
                options={"batch_size": 25},
            )


class TestChunking:
    """Tests for chunk_text utility and _EmbedTextBatch chunking."""

    def test_chunk_text_short(self):
        """Short text returns single chunk."""
        from vane.ai.functions import chunk_text

        result = chunk_text("hello world", max_chars=100)
        assert result == ["hello world"]

    def test_chunk_text_exact_boundary(self):
        """Text at exactly max_chars returns single chunk."""
        from vane.ai.functions import chunk_text

        text = "a" * 100
        result = chunk_text(text, max_chars=100)
        assert result == [text]

    def test_chunk_text_splits(self):
        """Long text is split into overlapping chunks."""
        from vane.ai.functions import chunk_text

        text = "a" * 250
        result = chunk_text(text, max_chars=100, overlap_chars=20)
        assert len(result) == 3
        # First chunk: 0-100, second: 80-180, third: 160-250
        assert all(len(c) <= 100 for c in result)
        assert len(result[-1]) == 90  # 250-160

    def test_chunk_text_overlap_content(self):
        """Overlapping regions share the same content."""
        from vane.ai.functions import chunk_text

        text = "".join(str(i % 10) for i in range(300))
        result = chunk_text(text, max_chars=100, overlap_chars=30)
        # Check overlap between first two chunks
        assert result[0][-30:] == result[1][:30]

    def test_chunk_text_no_overlap(self):
        """Zero overlap produces non-overlapping chunks."""
        from vane.ai.functions import chunk_text

        text = "a" * 200
        result = chunk_text(text, max_chars=100, overlap_chars=0)
        assert len(result) == 2
        assert result[0] == "a" * 100
        assert result[1] == "a" * 100

    def test_weighted_average_embeddings(self):
        """Weighted average normalizes embeddings correctly."""
        from vane.ai.functions import _weighted_average_embeddings

        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        result = _weighted_average_embeddings([e1, e2], [1.0, 1.0])
        # Equal weights → 45 degree angle, normalized
        expected = np.array([1, 1, 0], dtype=np.float32) / np.sqrt(2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_weighted_average_unequal_weights(self):
        """Longer chunk gets more weight."""
        from vane.ai.functions import _weighted_average_embeddings

        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        # e1 has 3x the weight
        result = _weighted_average_embeddings([e1, e2], [3.0, 1.0])
        assert result[0] > result[1]  # x component dominant

    def test_embed_batch_with_chunking(self):
        """_EmbedTextBatch chunks long texts and averages embeddings."""
        from vane.ai.functions import _EmbedTextBatch

        dim = 4

        class FakeEmbedder:
            def embed_text(self, texts):
                # Return a distinct unit vector per chunk index
                return [np.random.RandomState(hash(t) % 2**31).randn(dim).astype(np.float32) for t in texts]

        @dataclass
        class FakeDescriptor:
            def get_provider(self):
                return "test"

            def get_model(self):
                return "test"

            def get_options(self):
                return {}

            def get_dimensions(self):
                return EmbeddingDimensions(size=dim)

            def instantiate(self):
                return FakeEmbedder()

        desc = FakeDescriptor()
        batch = _EmbedTextBatch(desc, "text", "embedding", dim, max_chunk_chars=50, chunk_overlap_chars=10)

        # Short text (no chunking) + long text (will be chunked)
        table = pa.table({"text": ["short", "a" * 200]})
        result = _drive(batch, table)

        assert result.num_rows == 2
        emb0 = result.column("embedding")[0].as_py()
        emb1 = result.column("embedding")[1].as_py()
        assert len(emb0) == dim
        assert len(emb1) == dim

    def test_embed_batch_no_chunking_by_default(self):
        """_EmbedTextBatch without max_chunk_chars doesn't chunk."""
        from vane.ai.functions import _EmbedTextBatch

        call_count = 0
        dim = 4

        class CountingEmbedder:
            def embed_text(self, texts):
                nonlocal call_count
                call_count += 1
                return [np.ones(dim, dtype=np.float32) for _ in texts]

        @dataclass
        class FakeDescriptor:
            def get_provider(self):
                return "test"

            def get_model(self):
                return "test"

            def get_options(self):
                return {}

            def get_dimensions(self):
                return EmbeddingDimensions(size=dim)

            def instantiate(self):
                return CountingEmbedder()

        desc = FakeDescriptor()
        batch = _EmbedTextBatch(desc, "text", "embedding", dim)  # no chunking

        table = pa.table({"text": ["a" * 5000]})
        result = _drive(batch, table)

        assert result.num_rows == 1
        assert call_count == 1  # single call, no chunking

    def test_embed_batch_chunking_params_stored(self):
        """_EmbedTextBatch stores chunking params correctly."""
        from vane.ai.functions import _EmbedTextBatch

        dim = 4

        class SimpleEmbedder:
            def embed_text(self, t):
                return [np.zeros(dim) for _ in t]

        @dataclass
        class FakeDescriptor:
            def get_provider(self):
                return "test"

            def get_model(self):
                return "test"

            def get_options(self):
                return {}

            def get_dimensions(self):
                return EmbeddingDimensions(size=dim)

            def instantiate(self):
                return SimpleEmbedder()

        desc = FakeDescriptor()
        batch = _EmbedTextBatch(desc, "text", "embedding", dim, max_chunk_chars=500, chunk_overlap_chars=50)
        assert batch._max_chunk_chars == 500
        assert batch._chunk_overlap_chars == 50


# ---------------------------------------------------------------------------
# Structured Output + Responses API tests
# ---------------------------------------------------------------------------


class TestTokenMetrics:
    """Tests for vane.ai.metrics module."""

    def setup_method(self):
        from vane.ai.metrics import reset_token_metrics, set_token_metrics_callback

        reset_token_metrics()
        set_token_metrics_callback(None)

    def teardown_method(self):
        from vane.ai.metrics import reset_token_metrics, set_token_metrics_callback

        reset_token_metrics()
        set_token_metrics_callback(None)

    def test_record_and_get(self):
        from vane.ai.metrics import get_token_metrics, record_token_metrics

        record_token_metrics(
            protocol="prompt",
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        entries = get_token_metrics()
        assert len(entries) == 1
        e = entries[0]
        assert e.protocol == "prompt"
        assert e.model == "gpt-4o"
        assert e.provider == "openai"
        assert e.input_tokens == 100
        assert e.output_tokens == 50
        assert e.total_tokens == 150
        assert e.requests == 1

    def test_accumulation(self):
        from vane.ai.metrics import get_token_metrics, record_token_metrics

        record_token_metrics(
            protocol="prompt",
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
        )
        record_token_metrics(
            protocol="prompt",
            model="gpt-4o",
            provider="openai",
            input_tokens=200,
            output_tokens=80,
        )
        entries = get_token_metrics()
        assert len(entries) == 1
        e = entries[0]
        assert e.input_tokens == 300
        assert e.output_tokens == 130
        assert e.requests == 2

    def test_multiple_keys(self):
        from vane.ai.metrics import get_token_metrics, record_token_metrics

        record_token_metrics(
            protocol="prompt",
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
        )
        record_token_metrics(
            protocol="embed",
            model="text-embedding-3-small",
            provider="openai",
            input_tokens=500,
            total_tokens=500,
        )
        record_token_metrics(
            protocol="prompt",
            model="claude-3-5-sonnet",
            provider="anthropic",
            input_tokens=200,
            output_tokens=60,
        )
        entries = get_token_metrics()
        assert len(entries) == 3

    def test_none_tokens_ignored(self):
        from vane.ai.metrics import get_token_metrics, record_token_metrics

        record_token_metrics(
            protocol="prompt",
            model="m",
            provider="p",
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )
        e = get_token_metrics()[0]
        assert e.input_tokens == 0
        assert e.output_tokens == 0
        assert e.total_tokens == 0
        assert e.requests == 1

    def test_reset(self):
        from vane.ai.metrics import get_token_metrics, record_token_metrics, reset_token_metrics

        record_token_metrics(protocol="prompt", model="m", provider="p", input_tokens=10)
        assert len(get_token_metrics()) == 1
        reset_token_metrics()
        assert len(get_token_metrics()) == 0

    def test_summary(self):
        from vane.ai.metrics import get_token_metrics_summary, record_token_metrics

        record_token_metrics(
            protocol="prompt",
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        record_token_metrics(
            protocol="prompt",
            model="claude-3",
            provider="anthropic",
            input_tokens=200,
            output_tokens=80,
        )
        s = get_token_metrics_summary()
        assert s["total_input_tokens"] == 300
        assert s["total_output_tokens"] == 130
        assert s["total_requests"] == 2
        assert "openai" in s["by_provider"]
        assert "anthropic" in s["by_provider"]
        assert s["by_provider"]["openai"]["input_tokens"] == 100
        assert s["by_provider"]["anthropic"]["input_tokens"] == 200

    def test_callback(self):
        from vane.ai.metrics import record_token_metrics, set_token_metrics_callback

        received = []
        set_token_metrics_callback(lambda entry: received.append(entry))
        record_token_metrics(
            protocol="prompt",
            model="m",
            provider="p",
            input_tokens=10,
            output_tokens=5,
        )
        assert len(received) == 1
        assert received[0]["input_tokens"] == 10
        assert received[0]["output_tokens"] == 5
        assert received[0]["provider"] == "p"

    def test_callback_error_does_not_raise(self):
        from vane.ai.metrics import record_token_metrics, set_token_metrics_callback

        set_token_metrics_callback(lambda _: 1 / 0)
        # Should not raise
        record_token_metrics(protocol="prompt", model="m", provider="p", input_tokens=1)

    def test_remove_callback(self):
        from vane.ai.metrics import record_token_metrics, set_token_metrics_callback

        received = []
        set_token_metrics_callback(lambda entry: received.append(entry))
        record_token_metrics(protocol="prompt", model="m", provider="p", input_tokens=1)
        assert len(received) == 1
        set_token_metrics_callback(None)
        record_token_metrics(protocol="prompt", model="m", provider="p", input_tokens=1)
        assert len(received) == 1  # callback removed, no new entry

    def test_thread_safety(self):
        import threading

        from vane.ai.metrics import get_token_metrics, record_token_metrics

        def record_many():
            for _ in range(100):
                record_token_metrics(
                    protocol="prompt",
                    model="m",
                    provider="p",
                    input_tokens=1,
                    output_tokens=1,
                )

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        e = get_token_metrics()[0]
        assert e.requests == 400
        assert e.input_tokens == 400


class TestOpenAITokenMetrics:
    """Tests that OpenAI provider calls record_token_metrics."""

    def setup_method(self):
        from vane.ai.metrics import reset_token_metrics

        reset_token_metrics()

    def teardown_method(self):
        from vane.ai.metrics import reset_token_metrics

        reset_token_metrics()

    def test_chat_completions_records_usage(self):
        """Chat Completions response with usage triggers metrics."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.metrics import get_token_metrics
        from vane.ai.providers.openai import OpenAIPrompter

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 42
        mock_usage.completion_tokens = 18
        mock_usage.total_tokens = 60

        mock_choice = MagicMock()
        mock_choice.message.content = "hello"

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        prompter = OpenAIPrompter.__new__(OpenAIPrompter)
        prompter._client = mock_client
        prompter._model = "gpt-4o"
        prompter._use_chat_completions = True
        prompter._return_format = None
        prompter._options = {}
        prompter._system_message = None

        result = asyncio.run(prompter.prompt(("hi",)))
        assert result == "hello"
        entries = get_token_metrics()
        assert len(entries) == 1
        e = entries[0]
        assert e.provider == "openai"
        assert e.input_tokens == 42
        assert e.output_tokens == 18
        assert e.total_tokens == 60

    def test_responses_api_records_usage(self):
        """Responses API response with usage triggers metrics."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.metrics import get_token_metrics
        from vane.ai.providers.openai import OpenAIPrompter

        mock_usage = MagicMock()
        mock_usage.input_tokens = 30
        mock_usage.output_tokens = 15
        mock_usage.total_tokens = 45
        # Responses API doesn't have prompt_tokens/completion_tokens
        mock_usage.prompt_tokens = None
        mock_usage.completion_tokens = None

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.output_text = "world"

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        prompter = OpenAIPrompter.__new__(OpenAIPrompter)
        prompter._client = mock_client
        prompter._model = "gpt-4o"
        prompter._use_chat_completions = False
        prompter._return_format = None
        prompter._options = {}
        prompter._system_message = None

        result = asyncio.run(prompter.prompt(("hi",)))
        assert result == "world"
        entries = get_token_metrics()
        assert len(entries) == 1
        e = entries[0]
        assert e.input_tokens == 30
        assert e.output_tokens == 15

    def test_embed_records_usage(self):
        """Embedding response with usage triggers metrics."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.metrics import get_token_metrics
        from vane.ai.providers.openai import OpenAITextEmbedder

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 25
        mock_usage.total_tokens = 25

        mock_embedding = MagicMock()
        mock_embedding.index = 0
        mock_embedding.embedding = [0.1, 0.2, 0.3]

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.data = [mock_embedding]

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._client = mock_client
        embedder._model = "text-embedding-3-small"
        embedder._dimensions = None

        result = asyncio.run(embedder._embed_batch(["hello"]))
        assert len(result) == 1
        entries = get_token_metrics()
        assert len(entries) == 1
        e = entries[0]
        assert e.protocol == "embed"
        assert e.provider == "openai"
        assert e.input_tokens == 25

    def test_no_usage_no_error(self):
        """If response has no usage attribute, no metrics recorded, no error."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.metrics import get_token_metrics
        from vane.ai.providers.openai import OpenAIPrompter

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"

        mock_response = MagicMock(spec=[])  # spec=[] means no attributes
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        prompter = OpenAIPrompter.__new__(OpenAIPrompter)
        prompter._client = mock_client
        prompter._model = "gpt-4o"
        prompter._use_chat_completions = True
        prompter._return_format = None
        prompter._options = {}
        prompter._system_message = None

        result = asyncio.run(prompter.prompt(("test",)))
        assert result == "ok"
        assert len(get_token_metrics()) == 0


class TestOpenAITokenLimits:
    """Tests for per-model input token limits and oversized-text chunking."""

    def test_embedding_response_indices_restore_input_order(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.openai import OpenAITextEmbedder

        response = SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[20.0]),
                SimpleNamespace(index=0, embedding=[10.0]),
            ],
            usage=None,
        )
        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._client = MagicMock()
        embedder._client.embeddings.create = AsyncMock(return_value=response)
        embedder._model = "compatible-model"
        embedder._dimensions = 1

        result = asyncio.run(embedder._embed_batch(["row-0", "row-1"]))

        assert [embedding.tolist() for embedding in result] == [[10.0], [20.0]]

    def test_embedding_response_without_indices_preserves_physical_order(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.openai import OpenAITextEmbedder

        response = SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[10.0]),
                SimpleNamespace(embedding=[20.0]),
            ],
            usage=None,
        )
        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._client = MagicMock()
        embedder._client.embeddings.create = AsyncMock(return_value=response)
        embedder._model = "compatible-model"
        embedder._dimensions = 1

        result = asyncio.run(embedder._embed_batch(["row-0", "row-1"]))

        assert [embedding.tolist() for embedding in result] == [[10.0], [20.0]]

    @pytest.mark.parametrize("indices", [(0,), (0, 0), (0, 2), (0, "1")])
    def test_embedding_response_rejects_mixed_or_invalid_indices(self, indices):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.openai import OpenAITextEmbedder

        data = [SimpleNamespace(index=index, embedding=[float(position)]) for position, index in enumerate(indices)]
        data.extend(SimpleNamespace(embedding=[float(position)]) for position in range(len(data), 2))
        response = SimpleNamespace(data=data, usage=None)
        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._client = MagicMock()
        embedder._client.embeddings.create = AsyncMock(return_value=response)
        embedder._model = "compatible-model"
        embedder._dimensions = 1

        with pytest.raises(TypeError, match="invalid embedding indices"):
            asyncio.run(embedder._embed_batch(["row-0", "row-1"]))

    def test_get_input_token_limit_known_model(self):
        from vane.ai.providers.openai import _get_input_token_limit

        assert _get_input_token_limit("text-embedding-ada-002") == 8191
        assert _get_input_token_limit("text-embedding-3-small") == 8191
        assert _get_input_token_limit("text-embedding-3-large") == 8191

    def test_get_input_token_limit_unknown_model(self):
        from vane.ai.providers.openai import _get_input_token_limit

        assert _get_input_token_limit("custom-embed-model") == 8192

    def test_chunk_text_basic(self):
        from vane.ai.providers.openai import _chunk_text_by_token_limit

        result = _chunk_text_by_token_limit("abcdefgh", 3, len)
        assert result == ["abc", "def", "gh"]

    def test_chunk_text_exact(self):
        from vane.ai.providers.openai import _chunk_text_by_token_limit

        result = _chunk_text_by_token_limit("abcdef", 3, len)
        assert result == ["abc", "def"]

    def test_chunk_text_short(self):
        from vane.ai.providers.openai import _chunk_text_by_token_limit

        result = _chunk_text_by_token_limit("ab", 10, len)
        assert result == ["ab"]

    def test_known_openai_model_uses_model_specific_tokenizer_for_chunking(self, monkeypatch):
        import sys
        from types import SimpleNamespace

        from vane.ai.providers.openai import _build_token_estimator, _chunk_text_by_token_limit

        requested_models = []

        class FakeEncoding:
            def __init__(self):
                self.token_bytes = {}

            def encode_ordinary(self, value):
                weights = {"a": 1, "汉": 2, "🚀": 3}
                tokens = []
                token = 0
                for character in value:
                    raw = character.encode("utf-8")
                    weight = weights[character]
                    for index in range(weight):
                        end = len(raw) * (index + 1) // weight
                        start = len(raw) * index // weight
                        self.token_bytes[token] = raw[start:end]
                        tokens.append(token)
                        token += 1
                return tokens

            def decode_single_token_bytes(self, token):
                return self.token_bytes[token]

        def encoding_for_model(model):
            requested_models.append(model)
            return FakeEncoding()

        monkeypatch.setitem(sys.modules, "tiktoken", SimpleNamespace(encoding_for_model=encoding_for_model))
        estimate = _build_token_estimator("text-embedding-3-small", use_openai_tokenizer=True)

        assert estimate("a汉🚀") == 6
        assert _chunk_text_by_token_limit("a汉🚀", 3, estimate) == ["a汉", "🚀"]
        assert requested_models == ["text-embedding-3-small"]

    def test_model_token_chunking_handles_nonmonotonic_unicode_prefixes(self):
        from vane.ai.providers.openai import _chunk_text_by_token_limit, _TokenEstimator

        class PhraseEncoding:
            phrases = {"删除": 1, " 데이터": 2}
            split_token_bytes = {3: b" \xed", 4: b"\x84", 5: b"\xb0", 6: "터".encode("utf-8")}

            def encode_ordinary(self, value):
                tokens = []
                index = 0
                while index < len(value):
                    if value.startswith(" 터", index):
                        tokens.extend((3, 4, 5))
                        index += 2
                        continue
                    if value.startswith("터", index):
                        tokens.append(6)
                        index += 1
                        continue
                    phrase = next((item for item in self.phrases if value.startswith(item, index)), None)
                    if phrase is not None:
                        tokens.append(self.phrases[phrase])
                        index += len(phrase)
                        continue
                    for byte in value[index].encode("utf-8"):
                        tokens.append(1_000 + byte)
                    index += 1
                return tokens

            def decode_single_token_bytes(self, token):
                if token in self.split_token_bytes:
                    return self.split_token_bytes[token]
                for phrase, phrase_token in self.phrases.items():
                    if token == phrase_token:
                        return phrase.encode("utf-8")
                return bytes([token - 1_000])

        estimate = _TokenEstimator(PhraseEncoding())

        for text in ("删除x", " 데이터x"):
            chunks = _chunk_text_by_token_limit(text, 1, estimate)
            assert chunks == [text[:-1], "x"]
            assert "".join(chunks) == text
            assert all(estimate(chunk) <= 1 for chunk in chunks)

        split_unicode = _chunk_text_by_token_limit(" 터", 1, estimate)
        assert split_unicode == [" ", "터"]
        assert "".join(split_unicode) == " 터"
        assert all(estimate(chunk) <= 1 for chunk in split_unicode)

    def test_unknown_compatible_model_uses_utf8_byte_upper_bound_for_chunking(self):
        from vane.ai.providers.openai import _build_token_estimator, _chunk_text_by_token_limit

        estimate = _build_token_estimator("custom-embed-model", use_openai_tokenizer=False)

        assert estimate("汉") == 3
        assert estimate("🚀") == 4
        assert _chunk_text_by_token_limit("汉🚀a", 4, estimate) == ["汉", "🚀", "a"]

    def test_oversized_input_gets_chunked(self):
        """An input exceeding input_text_token_limit is chunked and averaged."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.openai import OpenAITextEmbedder

        call_log: list[list[str]] = []

        async def mock_create(**kwargs):
            texts = kwargs["input"]
            call_log.append(texts)
            mock_response = MagicMock()
            mock_response.usage = None
            mock_response.data = []
            for index, _t in enumerate(texts):
                emb = MagicMock()
                emb.index = index
                emb.embedding = [1.0, 0.0, 0.0]
                mock_response.data.append(emb)
            return mock_response

        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._model = "text-embedding-3-small"
        embedder._dimensions = None
        embedder._batch_token_limit = 300_000
        embedder._input_text_token_limit = 10
        embedder._estimate_tokens = lambda value: (len(value) + 2) // 3
        mock_client = AsyncMock()
        mock_client.embeddings.create = mock_create
        embedder._client = mock_client

        # "a" * 90 → 30 est_tokens > limit of 10 → chunked into 3 pieces of 30 chars
        result = asyncio.run(embedder.embed_text(["a" * 90]))

        assert len(result) == 1
        # Should have been chunked: one _embed_batch call with 3 chunks
        assert len(call_log) == 1
        assert len(call_log[0]) == 3
        # Result is L2-normalised
        norm = np.linalg.norm(result[0])
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)

    def test_oversized_input_uses_token_weighted_average(self, monkeypatch):
        """Chunk recombination weights embeddings by tokens, not characters."""
        import asyncio

        from vane.ai.providers import openai as openai_provider
        from vane.ai.providers.openai import OpenAITextEmbedder

        chunks = ["ascii", "汉"]

        def estimate_tokens(value):
            if value == "oversized":
                return 20
            return {"ascii": 1, "汉": 10}[value]

        monkeypatch.setattr(
            openai_provider,
            "_chunk_text_by_token_limit",
            lambda _text, _limit, _estimate: chunks,
        )

        async def mock_embed_batch(texts):
            vectors = {
                "ascii": np.array([1.0, 0.0]),
                "汉": np.array([0.0, 1.0]),
            }
            return [vectors[text] for text in texts]

        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._batch_token_limit = 100
        embedder._input_text_token_limit = 10
        embedder._estimate_tokens = estimate_tokens
        embedder._embed_batch = mock_embed_batch

        result = asyncio.run(embedder.embed_text(["oversized"]))

        expected = np.array([1.0, 10.0])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(result[0], expected)

    def test_oversized_input_chunks_still_respect_batch_token_limit(self):
        """Chunks from one long input are split across bounded requests."""
        import asyncio

        from vane.ai.providers.openai import OpenAITextEmbedder

        calls: list[list[str]] = []

        async def mock_embed_batch(texts):
            calls.append(list(texts))
            return [np.ones(2, dtype=np.float32) for _ in texts]

        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._model = "test-model"
        embedder._dimensions = 2
        embedder._batch_token_limit = 5
        embedder._input_text_token_limit = 3
        embedder._estimate_tokens = lambda value: (len(value) + 2) // 3
        embedder._embed_batch = mock_embed_batch

        result = asyncio.run(embedder.embed_text(["a" * 60]))

        assert len(result) == 1
        assert len(calls) > 1
        assert all(sum((len(text) + 2) // 3 for text in call) <= 5 for call in calls)

    def test_normal_input_not_chunked(self):
        """Inputs within token limit are batched normally, not chunked."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.openai import OpenAITextEmbedder

        call_log: list[list[str]] = []

        async def mock_create(**kwargs):
            texts = kwargs["input"]
            call_log.append(texts)
            mock_response = MagicMock()
            mock_response.usage = None
            mock_response.data = []
            for index, _ in enumerate(texts):
                emb = MagicMock()
                emb.index = index
                emb.embedding = [0.5, 0.5]
                mock_response.data.append(emb)
            return mock_response

        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._model = "text-embedding-3-small"
        embedder._dimensions = None
        embedder._batch_token_limit = 300_000
        embedder._input_text_token_limit = 8191
        embedder._estimate_tokens = lambda value: (len(value) + 2) // 3
        mock_client = AsyncMock()
        mock_client.embeddings.create = mock_create
        embedder._client = mock_client

        result = asyncio.run(embedder.embed_text(["hello", "world"]))

        assert len(result) == 2
        # Single batch call with both texts
        assert len(call_log) == 1
        assert call_log[0] == ["hello", "world"]

    def test_batch_splitting_still_works(self):
        """Batch token limit still triggers multi-call splitting."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.openai import OpenAITextEmbedder

        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            texts = kwargs["input"]
            mock_response = MagicMock()
            mock_response.usage = None
            mock_response.data = []
            for index, _ in enumerate(texts):
                emb = MagicMock()
                emb.index = index
                emb.embedding = [1.0]
                mock_response.data.append(emb)
            return mock_response

        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._model = "test-model"
        embedder._dimensions = None
        embedder._batch_token_limit = 5
        embedder._input_text_token_limit = 100
        embedder._estimate_tokens = lambda value: (len(value) + 2) // 3
        mock_client = AsyncMock()
        mock_client.embeddings.create = mock_create
        embedder._client = mock_client

        # Each "a"*12 → 4 est tokens; two won't fit in one batch of limit 5
        result = asyncio.run(embedder.embed_text(["a" * 12, "b" * 12]))

        assert len(result) == 2
        assert call_count == 2  # split into 2 batches

    def test_mixed_oversized_and_normal(self):
        """Mix of oversized (chunked) and normal texts handled correctly."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.openai import OpenAITextEmbedder

        call_log: list[int] = []  # track number of texts per call

        async def mock_create(**kwargs):
            texts = kwargs["input"]
            call_log.append(len(texts))
            mock_response = MagicMock()
            mock_response.usage = None
            mock_response.data = []
            for i, _t in enumerate(texts):
                emb = MagicMock()
                emb.index = i
                emb.embedding = [float(i + 1), 0.0]
                mock_response.data.append(emb)
            return mock_response

        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._model = "test-model"
        embedder._dimensions = None
        embedder._batch_token_limit = 300_000
        embedder._input_text_token_limit = 10
        embedder._estimate_tokens = lambda value: (len(value) + 2) // 3
        mock_client = AsyncMock()
        mock_client.embeddings.create = mock_create
        embedder._client = mock_client

        texts = [
            "short",  # normal
            "a" * 90,  # oversized → 3 chunks of 30 chars
            "also short",  # normal
        ]
        result = asyncio.run(embedder.embed_text(texts))

        assert len(result) == 3
        # First "short" is batched, then flushed before oversized
        # Oversized → separate _embed_batch with 3 chunks
        # "also short" → final flush
        assert len(call_log) == 3

    def test_descriptor_passes_token_limits(self):
        """OpenAITextEmbedderDescriptor passes token limits to embedder."""
        from unittest.mock import patch

        from vane.ai.providers.openai import OpenAITextEmbedderDescriptor

        desc = OpenAITextEmbedderDescriptor(
            model_name="text-embedding-3-small",
            options={
                "batch_token_limit": 100_000,
                "input_text_token_limit": 4096,
            },
        )

        with (
            patch("openai.AsyncOpenAI"),
            patch("vane.ai.providers.openai._build_token_estimator", return_value=len),
        ):
            embedder = desc.instantiate()

        assert embedder._batch_token_limit == 100_000
        assert embedder._input_text_token_limit == 4096

    def test_descriptor_default_token_limits(self):
        """Default token limits when not specified in options."""
        from unittest.mock import patch

        from vane.ai.providers.openai import OpenAITextEmbedderDescriptor

        desc = OpenAITextEmbedderDescriptor(
            model_name="text-embedding-3-small",
        )

        with (
            patch("openai.AsyncOpenAI"),
            patch("vane.ai.providers.openai._build_token_estimator", return_value=len),
        ):
            embedder = desc.instantiate()

        assert embedder._batch_token_limit == 300_000
        assert embedder._input_text_token_limit == 8191  # model-specific


class TestAnthropicTokenMetrics:
    """Tests that Anthropic provider calls record_token_metrics."""

    def setup_method(self):
        from vane.ai.metrics import reset_token_metrics

        reset_token_metrics()

    def teardown_method(self):
        from vane.ai.metrics import reset_token_metrics

        reset_token_metrics()

    @pytest.mark.skipif(not _has_module("anthropic"), reason="anthropic not installed")
    def test_prompt_records_usage(self):
        """Anthropic messages.create response records token metrics."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.metrics import get_token_metrics
        from vane.ai.providers.anthropic import AnthropicPrompter

        mock_usage = MagicMock()
        mock_usage.input_tokens = 55
        mock_usage.output_tokens = 20

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "answer"

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.content = [mock_text_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        prompter = AnthropicPrompter.__new__(AnthropicPrompter)
        prompter._client = mock_client
        prompter._model = "claude-3-5-sonnet-20241022"
        prompter._system_message = None
        prompter._return_format = None
        prompter._options = {}

        result = asyncio.run(prompter.prompt(("hello",)))
        assert result == "answer"
        entries = get_token_metrics()
        assert len(entries) == 1
        e = entries[0]
        assert e.provider == "anthropic"
        assert e.input_tokens == 55
        assert e.output_tokens == 20
        assert e.requests == 1


class TestGoogleTokenMetrics:
    """Tests that Google provider calls record_token_metrics."""

    def setup_method(self):
        from vane.ai.metrics import reset_token_metrics

        reset_token_metrics()

    def teardown_method(self):
        from vane.ai.metrics import reset_token_metrics

        reset_token_metrics()

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_prompt_records_usage(self):
        """Google generate_content response records token metrics."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.metrics import get_token_metrics
        from vane.ai.providers.google import GooglePrompter

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 33
        mock_usage.candidates_token_count = 12
        mock_usage.total_token_count = 45

        mock_response = MagicMock()
        mock_response.usage_metadata = mock_usage
        mock_response.text = "gemini says hi"

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        prompter = GooglePrompter.__new__(GooglePrompter)
        prompter._client = mock_client
        prompter._model = "gemini-2.0-flash"
        prompter._system_message = None
        prompter._return_format = None
        prompter._options = {}

        result = asyncio.run(prompter.prompt(("hello",)))
        assert result == "gemini says hi"
        entries = get_token_metrics()
        assert len(entries) == 1
        e = entries[0]
        assert e.provider == "google"
        assert e.input_tokens == 33
        assert e.output_tokens == 12
        assert e.total_tokens == 45


# ---------------------------------------------------------------------------
# Retry / on_error
# ---------------------------------------------------------------------------


class TestRetryAfterError:
    """Tests for RetryAfterError integration with retry helpers."""

    def test_retry_call_honors_retry_after(self):
        """_retry_call uses RetryAfterError.retry_after for wait time."""
        import time

        from vane.ai.functions import RetryAfterError, _retry_call

        calls = []

        def fn():
            calls.append(time.monotonic())
            if len(calls) < 2:
                raise RetryAfterError(retry_after=0.1, original=RuntimeError("429"))
            return "ok"

        result = _retry_call(fn, max_retries=2, on_error="raise")
        assert result == "ok"
        assert len(calls) == 2
        # Should have waited ~0.1s (the retry_after), not 1s (exponential backoff)
        gap = calls[1] - calls[0]
        assert gap >= 0.08  # allow timing slack
        assert gap < 0.5  # definitely not exponential backoff (1s)

    def test_retry_call_exhaustion_keeps_only_safe_diagnostics(self):
        from vane.ai.functions import RetryAfterError, _retry_call

        original = RuntimeError("rate limited")

        def fn():
            raise RetryAfterError(retry_after=0.01, original=original)

        with pytest.raises(RetryAfterError, match="RuntimeError") as exc_info:
            _retry_call(fn, max_retries=0, on_error="raise")
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None

    def test_retry_call_async_honors_retry_after(self):
        import asyncio

        from vane.ai.functions import RetryAfterError, _retry_call_async

        calls = []

        async def fn():
            calls.append(1)
            if len(calls) < 2:
                raise RetryAfterError(retry_after=0.05, original=ValueError("503"))
            return "done"

        result = asyncio.run(_retry_call_async(fn, max_retries=2, on_error="raise"))
        assert result == "done"
        assert len(calls) == 2

    def test_retry_call_async_exhaustion_keeps_only_safe_diagnostics(self):
        import asyncio

        from vane.ai.functions import RetryAfterError, _retry_call_async

        async def fn():
            raise RetryAfterError(retry_after=0.01, original=ValueError("overloaded"))

        with pytest.raises(RetryAfterError, match="ValueError") as exc_info:
            asyncio.run(_retry_call_async(fn, max_retries=0, on_error="raise"))
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None


class TestGoogleRetryHandling:
    """Tests for Google provider 429/503 → RetryAfterError conversion."""

    def test_google_429_raises_retry_after(self):
        """Google APIError with code=429 is converted to RetryAfterError."""
        from vane.ai.functions import RetryAfterError
        from vane.ai.providers.google import _raise_retry_after_on_google_error

        exc = Exception("rate limited")
        exc.code = 429
        exc.response = None

        with pytest.raises(RetryAfterError) as ctx:
            _raise_retry_after_on_google_error(exc)
        assert ctx.value.retry_after == 5.0  # default
        assert ctx.value.__cause__ is None
        assert ctx.value.__context__ is None

    def test_google_503_raises_retry_after(self):
        """Google APIError with code=503 is converted to RetryAfterError."""
        from vane.ai.functions import RetryAfterError
        from vane.ai.providers.google import _raise_retry_after_on_google_error

        exc = Exception("service unavailable")
        exc.code = 503
        exc.response = None

        with pytest.raises(RetryAfterError) as ctx:
            _raise_retry_after_on_google_error(exc)
        assert ctx.value.retry_after == 5.0

    def test_google_429_with_retry_after_header(self):
        """Retry-After header from response is honoured."""
        from unittest.mock import MagicMock

        from vane.ai.functions import RetryAfterError
        from vane.ai.providers.google import _raise_retry_after_on_google_error

        mock_response = MagicMock()
        mock_response.headers = {"Retry-After": "10"}

        exc = Exception("rate limited")
        exc.code = 429
        exc.response = mock_response

        with pytest.raises(RetryAfterError) as ctx:
            _raise_retry_after_on_google_error(exc)
        assert ctx.value.retry_after == 10.0

    def test_google_400_not_retryable(self):
        """Non-retryable errors (400) are not converted."""
        from vane.ai.providers.google import _raise_retry_after_on_google_error

        exc = Exception("bad request")
        exc.code = 400
        exc.response = None

        # Should not raise — just returns
        _raise_retry_after_on_google_error(exc)

    def test_google_no_code_not_retryable(self):
        """Exceptions without .code attribute are not converted."""
        from vane.ai.providers.google import _raise_retry_after_on_google_error

        exc = RuntimeError("random error")
        # No .code attribute → should not raise
        _raise_retry_after_on_google_error(exc)


class TestEmbedProviderCapabilityErrors:
    @pytest.mark.parametrize("name", ["batch_size", "actor_number", "batch_token_limit"])
    def test_non_nullable_integer_options_reject_explicit_none_during_planning(self, name):
        with pytest.raises(ValueError, match=name):
            vane.ai.embed(vane.col("text"), dimensions=4, **{name: None})

    @pytest.mark.parametrize("option", ["base_url", "api_key", "organization", "timeout"])
    def test_openai_provider_rejects_legacy_constructor_options(self, option):
        from vane.ai.providers.openai import OpenAIProvider

        with pytest.raises(TypeError, match=option):
            OpenAIProvider(**{option: "legacy-value"})

    @pytest.mark.parametrize(
        "options",
        [
            {"trust_remote_code": True},
            {"local_files_only": "yes"},
            {"revision": "   "},
        ],
    )
    def test_transformers_provider_rejects_legacy_constructor_options(self, options):
        from vane.ai.providers.transformers import TransformersProvider

        with pytest.raises(TypeError):
            TransformersProvider(**options)

    @pytest.mark.parametrize("status_code", [404, 405, 501])
    def test_openai_dynamic_endpoint_error_preserves_capability_context(self, monkeypatch, status_code):
        import asyncio
        import sys
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.provider import ProviderCapabilityError
        from vane.ai.providers.openai import OpenAIProvider

        class EndpointError(Exception):
            pass

        original = EndpointError("model is unavailable on this endpoint")
        original.status_code = status_code
        client = MagicMock()
        client.embeddings.create = AsyncMock(side_effect=original)
        monkeypatch.setitem(
            sys.modules,
            "openai",
            SimpleNamespace(OpenAIError=EndpointError, AsyncOpenAI=MagicMock(return_value=client)),
        )

        descriptor = OpenAIProvider(name="openai-compatible-alias").get_text_embedder(
            model="endpoint-only-model",
            dimensions=4,
        )
        embedder = descriptor.instantiate()

        with pytest.raises(ProviderCapabilityError) as exc_info:
            asyncio.run(embedder._embed_batch(["hello"]))

        error = exc_info.value
        assert (error.provider, error.model, error.capability) == (
            "openai-compatible-alias",
            "endpoint-only-model",
            "embedding endpoint/model",
        )
        assert error.original_error is not original
        assert str(error.original_error) == f"EndpointError (status_code={status_code})"
        assert error.__cause__ is None
        assert error.__context__ is None

    def test_openai_input_validation_error_is_not_misclassified_as_capability(self, monkeypatch):
        import asyncio
        import sys
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.openai import OpenAITextEmbedder

        class InputError(Exception):
            status_code = 400
            param = "input"
            code = "context_length_exceeded"

        original = InputError("one input is too long")
        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAIError=InputError))

        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._client = MagicMock()
        embedder._client.embeddings.create = AsyncMock(side_effect=original)
        embedder._model = "endpoint-model"
        embedder._dimensions = 4

        with pytest.raises(InputError) as exc_info:
            asyncio.run(embedder._embed_batch(["bad input"]))

        assert exc_info.value is original

    def test_openai_structured_model_error_is_a_capability_error(self, monkeypatch):
        import asyncio
        import sys
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.provider import ProviderCapabilityError
        from vane.ai.providers.openai import OpenAITextEmbedder

        class ModelError(Exception):
            status_code = 400
            param = "model"
            code = "model_not_found"

        original = ModelError("selected model cannot embed")
        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAIError=ModelError))

        embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
        embedder._client = MagicMock()
        embedder._client.embeddings.create = AsyncMock(side_effect=original)
        embedder._model = "chat-only-model"
        embedder._dimensions = 4

        with pytest.raises(ProviderCapabilityError) as exc_info:
            asyncio.run(embedder._embed_batch(["hello"]))

        assert exc_info.value.original_error is not original
        assert str(exc_info.value.original_error) == "ModelError (status_code=400)"

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    @pytest.mark.parametrize(
        ("code", "status"),
        [
            (404, None),
            (405, None),
            (501, None),
            (None, "NOT_FOUND"),
            (None, "UNIMPLEMENTED"),
        ],
    )
    def test_google_dynamic_endpoint_error_preserves_capability_context(self, monkeypatch, code, status):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from google import genai

        from vane.ai.provider import ProviderCapabilityError
        from vane.ai.providers.google import GoogleProvider

        class EndpointError(Exception):
            pass

        original = EndpointError("model does not implement embed_content")
        original.code = code
        original.status = status
        client = MagicMock()
        monkeypatch.setattr(genai, "Client", lambda **_kwargs: client)
        provider = GoogleProvider(
            name="google-alias",
            embedding_model="chat-only-model",
            embedding_dimensions=4,
        )
        descriptor = provider.get_text_embedder()
        embedder = descriptor.instantiate()
        embedder._client.aio.models.embed_content = AsyncMock(side_effect=original)

        with pytest.raises(ProviderCapabilityError) as exc_info:
            asyncio.run(embedder.embed_text(["hello"]))

        error = exc_info.value
        assert (error.provider, error.model, error.capability) == (
            "google-alias",
            "chat-only-model",
            "embedding endpoint/model",
        )
        assert error.original_error is not original
        expected_summary = "EndpointError" if code is None else f"EndpointError (code={code})"
        assert str(error.original_error) == expected_summary
        assert error.__cause__ is None
        assert error.__context__ is None

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_google_generic_invalid_input_is_not_misclassified_as_capability(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.providers.google import GoogleTextEmbedder

        class InputError(Exception):
            code = 400
            status = "INVALID_ARGUMENT"
            details = {"error": {"message": "input is too long"}}

        original = InputError("input is too long")
        embedder = GoogleTextEmbedder.__new__(GoogleTextEmbedder)
        embedder._client = MagicMock()
        embedder._client.aio.models.embed_content = AsyncMock(side_effect=original)
        embedder._model = "embedding-model"
        embedder._dimensions = 4
        embedder._options = {}

        with pytest.raises(InputError) as exc_info:
            asyncio.run(embedder.embed_text(["bad input"]))

        assert exc_info.value is original

    @pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
    def test_google_structured_model_field_error_is_a_capability_error(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from vane.ai.provider import ProviderCapabilityError
        from vane.ai.providers.google import GoogleTextEmbedder

        class ModelError(Exception):
            code = 400
            status = "INVALID_ARGUMENT"
            details = {"fieldViolations": [{"field": "request.model"}]}

        original = ModelError("selected model cannot embed")
        embedder = GoogleTextEmbedder.__new__(GoogleTextEmbedder)
        embedder._client = MagicMock()
        embedder._client.aio.models.embed_content = AsyncMock(side_effect=original)
        embedder._model = "chat-only-model"
        embedder._dimensions = 4
        embedder._options = {}

        with pytest.raises(ProviderCapabilityError) as exc_info:
            asyncio.run(embedder.embed_text(["hello"]))

        assert exc_info.value.original_error is not original
        assert str(exc_info.value.original_error) == "ModelError (code=400)"

    def test_transformers_not_implemented_model_is_a_capability_error(self, monkeypatch):
        import contextlib
        import sys
        from types import SimpleNamespace

        from vane.ai.provider import ProviderCapabilityError
        from vane.ai.providers.transformers import TransformersProvider

        class UnsupportedModel:
            def eval(self):
                return self

            def encode(self, *_args, **_kwargs):
                raise NotImplementedError("model has no sentence-embedding capability")

        model = UnsupportedModel()
        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            SimpleNamespace(SentenceTransformer=lambda *_args, **_kwargs: model),
        )
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=contextlib.nullcontext))
        provider = TransformersProvider(name="transformers-alias")
        descriptor = provider.get_text_embedder(model="chat-only-model", dimensions=4)
        embedder = descriptor.instantiate()

        with pytest.raises(ProviderCapabilityError) as exc_info:
            embedder.embed_text(["hello"])

        assert descriptor.get_provider() == "transformers-alias"
        assert exc_info.value.provider == "transformers-alias"
        assert exc_info.value.original_error is not None


class TestRetryCall:
    """Tests for _retry_call and _retry_call_async helpers."""

    def test_success_no_retry(self):
        from vane.ai.functions import _retry_call

        calls = []

        def fn():
            calls.append(1)
            return "ok"

        result = _retry_call(fn, max_retries=3, on_error="raise")
        assert result == "ok"
        assert len(calls) == 1

    def test_retry_then_success(self):
        from vane.ai.functions import _retry_call

        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("transient")
            return "recovered"

        result = _retry_call(fn, max_retries=3, on_error="raise")
        assert result == "recovered"
        assert len(calls) == 3

    def test_retry_exhausted_raises(self):
        from vane.ai.functions import _retry_call

        def fn():
            raise RuntimeError("permanent")

        with pytest.raises(RuntimeError, match="permanent"):
            _retry_call(fn, max_retries=1, on_error="raise")

    def test_on_error_log_returns_default(self):
        from vane.ai.functions import _retry_call

        def fn():
            raise RuntimeError("fail")

        result = _retry_call(fn, max_retries=0, on_error="log", default="fallback")
        assert result == "fallback"

    def test_on_error_ignore_returns_default(self):
        from vane.ai.functions import _retry_call

        def fn():
            raise RuntimeError("fail")

        result = _retry_call(fn, max_retries=0, on_error="ignore", default=42)
        assert result == 42

    def test_on_error_ignore_returns_none_by_default(self):
        from vane.ai.functions import _retry_call

        def fn():
            raise RuntimeError("fail")

        result = _retry_call(fn, max_retries=0, on_error="ignore")
        assert result is None

    def test_awaitable_result_handled(self):
        """_retry_call drives awaitables through the provided run_async."""
        import asyncio

        from vane.ai.functions import _retry_call

        async def async_fn():
            return "async_result"

        loop = asyncio.new_event_loop()
        try:
            result = _retry_call(async_fn, max_retries=0, on_error="raise", run_async=loop.run_until_complete)
        finally:
            loop.close()
        assert result == "async_result"

    def test_awaitable_result_without_run_async_raises(self):
        """Without a bound runtime an awaitable result is a hard error."""
        from vane.ai.functions import _retry_call

        async def async_fn():
            return "unreachable"

        with pytest.raises(RuntimeError, match="bind_async_runtime"):
            _retry_call(async_fn, max_retries=0, on_error="ignore")

    def test_retry_call_async(self):
        import asyncio

        from vane.ai.functions import _retry_call_async

        calls = []

        async def fn():
            calls.append(1)
            if len(calls) < 2:
                raise ValueError("transient")
            return "ok"

        result = asyncio.run(_retry_call_async(fn, max_retries=2, on_error="raise"))
        assert result == "ok"
        assert len(calls) == 2

    def test_retry_call_async_exhausted_raises(self):
        import asyncio

        from vane.ai.functions import _retry_call_async

        async def fn():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(_retry_call_async(fn, max_retries=1, on_error="raise"))

    def test_retry_call_async_on_error_log(self):
        import asyncio

        from vane.ai.functions import _retry_call_async

        async def fn():
            raise RuntimeError("fail")

        result = asyncio.run(_retry_call_async(fn, max_retries=0, on_error="log", default="safe"))
        assert result == "safe"


class TestWrapperRetry:
    """Tests that wrapper classes use retry/on_error correctly."""

    def _make_embed_descriptor(self, embed_fn):
        """Create a minimal descriptor + embedder for testing."""
        desc = MagicMock(spec=[])
        desc.get_dimensions = MagicMock(
            return_value=MagicMock(
                as_arrow_type=MagicMock(return_value=pa.list_(pa.float32(), 3)),
                list_size=3,
            )
        )
        embedder = MagicMock(spec=[])
        embedder.embed_text = embed_fn
        desc.instantiate = MagicMock(return_value=embedder)
        return desc

    def test_embed_retry_success(self, monkeypatch):
        from vane.ai.functions import _EmbedTextBatch

        monkeypatch.setattr("vane.ai.functions.time.sleep", lambda _seconds: None)
        calls = []

        class ServiceUnavailableError(RuntimeError):
            status_code = 503

        def embed(texts):
            calls.append(1)
            if len(calls) < 2:
                raise ServiceUnavailableError("API error")
            return [np.array([1.0, 2.0, 3.0])] * len(texts)

        desc = self._make_embed_descriptor(embed)
        wrapper = _EmbedTextBatch(desc, "text", "emb", 3, max_retries=3, on_error="raise")
        table = pa.table({"text": ["hello"]})
        result = _drive(wrapper, table)
        assert result.column("emb").length() == 1
        assert len(calls) == 2

    @pytest.mark.parametrize("status_code", [None, 400, 401, 403, 404, 405, 422])
    def test_embed_does_not_retry_nontransient_http_error(self, status_code):
        from vane.ai.functions import _EmbedTextBatch

        calls = []

        class NonTransientError(RuntimeError):
            pass

        error = NonTransientError("invalid input")
        if status_code is not None:
            error.status_code = status_code

        def embed(texts):
            calls.append(list(texts))
            raise error

        desc = self._make_embed_descriptor(embed)
        wrapper = _EmbedTextBatch(desc, "text", "emb", 3, max_retries=3, on_error="raise")

        with pytest.raises(RuntimeError, match="Embed execution; upstream error: NonTransientError") as exc_info:
            _drive(wrapper, pa.table({"text": ["hello"]}))

        if status_code is not None:
            assert f"status_code={status_code}" in str(exc_info.value)
        assert "invalid input" not in str(exc_info.value)
        assert exc_info.value.__context__ is None
        assert calls == [["hello"]]

    def test_embed_ignore_isolates_without_retrying_nontransient_error(self):
        from vane.ai.functions import _EmbedTextBatch

        calls = []

        class UnprocessableInputError(RuntimeError):
            status_code = 422

        def embed(texts):
            calls.append(list(texts))
            raise UnprocessableInputError("invalid input")

        desc = self._make_embed_descriptor(embed)
        wrapper = _EmbedTextBatch(desc, "text", "emb", 3, max_retries=3, on_error="ignore")
        result = _drive(wrapper, pa.table({"text": ["first", "second"]}))

        assert result.column("emb").to_pylist() == [None, None]
        assert calls == [["first", "second"], ["first"], ["second"]]

    def test_embed_on_error_ignore(self):
        from vane.ai.functions import _EmbedTextBatch

        def embed(_texts):
            raise RuntimeError("permanent failure")

        desc = self._make_embed_descriptor(embed)
        wrapper = _EmbedTextBatch(desc, "text", "emb", 3, max_retries=0, on_error="ignore")
        table = pa.table({"text": ["hello"]})
        result = _drive(wrapper, table)
        assert result.column("emb").to_pylist() == [None]

    def test_classify_on_error_log(self):
        from vane.ai.functions import _ClassifyTextBatch

        def classify(_texts, _labels):
            raise RuntimeError("fail")

        desc = MagicMock(spec=[])
        classifier = MagicMock(spec=[])
        classifier.classify_text = classify
        desc.instantiate = MagicMock(return_value=classifier)
        wrapper = _ClassifyTextBatch(desc, "text", "label", ["a", "b"], max_retries=0, on_error="log")
        table = pa.table({"text": ["hello"]})
        result = wrapper(table)
        assert result.column("label").to_pylist() == [None]


# ---------------------------------------------------------------------------
# Basic Prompt provider contract
# ---------------------------------------------------------------------------


def test_openai_responses_request_mapping_preserves_part_order():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from vane.ai.providers.openai import OpenAIPrompter

    prompter = OpenAIPrompter.__new__(OpenAIPrompter)
    prompter._provider_name = "openai-alias"
    prompter._model = "response-model"
    prompter._system_message = "system"
    prompter._use_chat_completions = False
    prompter._options = {
        "temperature": 0.2,
        "max_output_tokens": 17,
        "top_p": 0.8,
    }
    response = SimpleNamespace(output_text="answer", usage=None)
    prompter._client = MagicMock()
    prompter._client.responses.create = AsyncMock(return_value=response)

    png = b"\x89PNG\r\n\x1a\nimage"
    assert asyncio.run(prompter.prompt(("before", png, "after"))) == "answer"

    kwargs = prompter._client.responses.create.await_args.kwargs
    assert kwargs["model"] == "response-model"
    assert kwargs["max_output_tokens"] == 17
    assert "stop" not in kwargs
    assert [part["type"] for part in kwargs["input"][1]["content"]] == [
        "input_text",
        "input_image",
        "input_text",
    ]
    assert kwargs["input"][0] == {"role": "system", "content": "system"}


def test_openai_responses_rejects_stop_sequences_during_planning():
    from vane.ai.providers.openai import OpenAIPrompterDescriptor, OpenAIProvider

    with pytest.raises(ValueError, match="use_chat_completions=True"):
        OpenAIProvider().get_prompter(options={"stop_sequences": ["END"]})
    with pytest.raises(ValueError, match="use_chat_completions=True"):
        OpenAIPrompterDescriptor(options={"stop_sequences": ["END"]})


def test_openai_chat_request_maps_max_output_tokens_and_stop_sequences():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from vane.ai.providers.openai import OpenAIPrompter

    prompter = OpenAIPrompter.__new__(OpenAIPrompter)
    prompter._provider_name = "openai"
    prompter._model = "chat-model"
    prompter._system_message = None
    prompter._use_chat_completions = True
    prompter._official_openai_endpoint = True
    prompter._options = {"max_output_tokens": 11, "stop_sequences": ["STOP"]}
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="chat answer"))],
        usage=None,
    )
    prompter._client = MagicMock()
    prompter._client.chat.completions.create = AsyncMock(return_value=response)

    assert asyncio.run(prompter.prompt(("hello",))) == "chat answer"
    kwargs = prompter._client.chat.completions.create.await_args.kwargs
    assert kwargs["max_tokens"] == 11
    assert kwargs["stop"] == ["STOP"]
    assert "max_output_tokens" not in kwargs


@pytest.mark.parametrize(
    ("official_endpoint", "expected_name"),
    [(True, "max_completion_tokens"), (False, "max_tokens")],
)
@pytest.mark.parametrize("model", ["o3-mini", "gpt-5", "gpt-5-mini", "gpt-5.1"])
def test_openai_chat_request_uses_model_and_endpoint_capability(model, official_endpoint, expected_name):
    from vane.ai.providers.openai import OpenAIPrompter

    prompter = OpenAIPrompter.__new__(OpenAIPrompter)
    prompter._model = model
    prompter._official_openai_endpoint = official_endpoint
    prompter._options = {"max_output_tokens": 11}
    prompter._return_format = None

    options = prompter._chat_completions_options()

    assert options == {expected_name: 11}


@pytest.mark.parametrize(
    ("base_url", "expected_name"),
    [
        ("https://api.openai.com/v1", "max_completion_tokens"),
        ("https://us.api.openai.com/v1", "max_completion_tokens"),
        ("https://eu.api.openai.com/v1", "max_completion_tokens"),
        ("https://au.api.openai.com/v1", "max_completion_tokens"),
        ("https://ca.api.openai.com/v1", "max_completion_tokens"),
        ("https://jp.api.openai.com/v1", "max_completion_tokens"),
        ("https://in.api.openai.com/v1", "max_completion_tokens"),
        ("https://sg.api.openai.com/v1", "max_completion_tokens"),
        ("https://kr.api.openai.com/v1", "max_completion_tokens"),
        ("https://gb.api.openai.com/v1", "max_completion_tokens"),
        ("https://ae.api.openai.com/v1", "max_completion_tokens"),
        ("https://compatible.example.test/v1", "max_tokens"),
        ("https://unknown.api.openai.com/v1", "max_tokens"),
    ],
)
def test_openai_chat_token_limit_mapping_follows_documented_endpoint(base_url, expected_name):
    from vane.ai.providers.openai import OpenAIPrompter, _uses_official_openai_endpoint

    prompter = OpenAIPrompter.__new__(OpenAIPrompter)
    prompter._model = "o3"
    prompter._official_openai_endpoint = _uses_official_openai_endpoint(base_url)
    prompter._options = {"max_output_tokens": 11}
    prompter._return_format = None

    assert prompter._chat_completions_options() == {expected_name: 11}


def test_anthropic_request_mapping_preserves_text_image_order():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from vane.ai.providers.anthropic import AnthropicPrompter

    prompter = AnthropicPrompter.__new__(AnthropicPrompter)
    prompter._provider_name = "anthropic"
    prompter._model = "claude-test"
    prompter._system_message = "system"
    prompter._options = {
        "max_tokens": 64,
        "temperature": 0.1,
        "stop_sequences": ["END"],
    }
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="answer")],
        usage=None,
        stop_reason="end_turn",
    )
    prompter._client = MagicMock()
    prompter._client.messages.create = AsyncMock(return_value=response)

    png = b"\x89PNG\r\n\x1a\nimage"
    assert asyncio.run(prompter.prompt(("before", png, "after"))) == "answer"

    kwargs = prompter._client.messages.create.await_args.kwargs
    assert kwargs["system"] == "system"
    assert kwargs["max_tokens"] == 64
    assert [part["type"] for part in kwargs["messages"][0]["content"]] == [
        "text",
        "image",
        "text",
    ]


@pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
def test_google_request_mapping_preserves_text_image_order():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from vane.ai.providers.google import GooglePrompter

    prompter = GooglePrompter.__new__(GooglePrompter)
    prompter._provider_name = "google"
    prompter._model = "gemini-test"
    prompter._system_message = "system"
    prompter._options = {
        "temperature": 0.2,
        "max_output_tokens": 19,
        "stop_sequences": ["END"],
    }
    response = SimpleNamespace(text="answer", usage_metadata=None)
    prompter._client = MagicMock()
    prompter._client.aio.models.generate_content = AsyncMock(return_value=response)

    png = b"\x89PNG\r\n\x1a\nimage"
    assert asyncio.run(prompter.prompt(("before", png, "after"))) == "answer"

    kwargs = prompter._client.aio.models.generate_content.await_args.kwargs
    assert kwargs["model"] == "gemini-test"
    content = kwargs["contents"][0]
    assert [part.text is not None for part in content.parts] == [True, False, True]
    assert kwargs["config"].system_instruction == "system"
    assert kwargs["config"].max_output_tokens == 19
    assert kwargs["config"].stop_sequences == ["END"]


@pytest.mark.parametrize(
    ("module_name", "helper_name"),
    [
        ("vane.ai.providers.openai", "_guess_mime_type"),
        ("vane.ai.providers.anthropic", "_guess_media_type"),
        ("vane.ai.providers.google", "_guess_media_type"),
    ],
)
def test_prompt_image_detection_rejects_unknown_or_non_image_content(module_name, helper_name):
    from importlib import import_module

    helper = getattr(import_module(module_name), helper_name)
    assert helper(b"%PDF-1.7") is None
    assert helper(b"not-an-image") is None


def test_prompt_provider_capability_error_preserves_context():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from vane.ai.provider import ProviderCapabilityError
    from vane.ai.providers.openai import OpenAIPrompter

    class EndpointError(Exception):
        status_code = 404

    original = EndpointError("model is not available")
    prompter = OpenAIPrompter.__new__(OpenAIPrompter)
    prompter._provider_name = "compatible-endpoint"
    prompter._model = "unknown-model"
    prompter._system_message = None
    prompter._use_chat_completions = False
    prompter._options = {}
    prompter._client = MagicMock()
    prompter._client.responses.create = AsyncMock(side_effect=original)

    with pytest.raises(ProviderCapabilityError) as error:
        asyncio.run(prompter.prompt(("hello",)))

    assert (error.value.provider, error.value.model) == (
        "compatible-endpoint",
        "unknown-model",
    )
    assert error.value.original_error is not original
    assert str(error.value.original_error) == "EndpointError (status_code=404)"


def test_provider_capability_error_redacts_upstream_credentials_and_round_trips(monkeypatch):
    import asyncio
    import sys
    import traceback
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    import cloudpickle

    from vane.ai.provider import ProviderCapabilityError
    from vane.ai.providers.openai import OpenAITextEmbedder

    secret = "AIzaSyD4n0m5M_NTpvI_GlTgQeX82aBcDeFgHi"

    class EndpointError(Exception):
        status_code = 404

    original = EndpointError(f"GET https://api.example/v1?key={secret} failed")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAIError=EndpointError))
    embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
    embedder._client = MagicMock()
    embedder._client.embeddings.create = AsyncMock(side_effect=original)
    embedder._provider_name = "openai-compatible"
    embedder._model = "chat-only-model"
    embedder._dimensions = 4
    embedder._encoding_format = "float"

    with pytest.raises(ProviderCapabilityError) as exc_info:
        asyncio.run(embedder._embed_batch(["hello"]))

    error = exc_info.value
    traceback_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    surfaces = (str(error), repr(error.original_error), repr(error.__dict__), traceback_text)
    assert all(secret not in surface for surface in surfaces)
    assert error.original_error is not original
    assert error.__cause__ is None
    assert error.__context__ is None
    assert str(error.original_error) == "EndpointError (status_code=404)"

    from duckdb.execution.udf_ray_stream_protocol import make_stream_error_pair

    _, stream_metadata = make_stream_error_pair(
        {
            "query_id": "query",
            "stage_id": "stage",
            "task_lease_id": "lease",
            "attempt_id": "attempt",
        },
        error,
    )
    assert secret not in repr(stream_metadata)

    for serializer in (pickle, cloudpickle):
        restored = serializer.loads(serializer.dumps(error))
        assert (restored.provider, restored.model, restored.capability) == (
            error.provider,
            error.model,
            error.capability,
        )
        assert restored.original_error_summary == error.original_error_summary
        assert secret not in repr(restored.__dict__)


def test_provider_capability_error_never_stringifies_unbounded_upstream_text():
    from vane.ai.provider import ProviderCapabilityError

    class UnboundedError(Exception):
        status_code = 503

        def __str__(self):
            raise AssertionError("upstream message must not be inspected")

    original = UnboundedError("x " * 1_048_576)
    error = ProviderCapabilityError("provider", "model", "capability", original_error=original)

    assert error.original_error_summary == "UnboundedError (status_code=503)"
    assert original not in error.__dict__.values()


def test_prompt_batch_retry_and_row_isolation(monkeypatch):
    import asyncio

    from vane.ai.functions import _PromptBatch

    attempts = {}

    class TransientError(RuntimeError):
        status_code = 503

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    class Prompter:
        async def prompt(self, messages):
            key = messages[0]
            attempts[key] = attempts.get(key, 0) + 1
            if key == "retry" and attempts[key] == 1:
                raise TransientError("temporary")
            if key == "fail":
                raise RuntimeError("permanent")
            return f"answer:{key}"

    class Descriptor:
        def instantiate(self):
            return Prompter()

    wrapper = _PromptBatch(
        Descriptor(),
        ["message"],
        "response",
        max_concurrency_per_actor=2,
        max_retries=1,
        on_error="ignore",
    )
    result = _drive(wrapper, pa.table({"message": ["ok", "retry", "fail", None]}))

    assert result.column("response").to_pylist() == [
        "answer:ok",
        "answer:retry",
        None,
        None,
    ]
    assert attempts == {"ok": 1, "retry": 2, "fail": 1}


@pytest.mark.parametrize("status_code", [None, 400, 401, 403, 404, 422])
def test_prompt_does_not_retry_permanent_provider_errors(status_code):
    from vane.ai.functions import _PromptBatch

    attempts = 0

    class PermanentError(RuntimeError):
        pass

    error = PermanentError("permanent")
    if status_code is not None:
        error.status_code = status_code

    class Prompter:
        async def prompt(self, _messages):
            nonlocal attempts
            attempts += 1
            raise error

    class Descriptor:
        def instantiate(self):
            return Prompter()

    wrapper = _PromptBatch(
        Descriptor(),
        ["message"],
        "response",
        max_concurrency_per_actor=1,
        max_retries=3,
        on_error="ignore",
    )

    result = _drive(wrapper, pa.table({"message": ["fail"]}))

    assert result.column("response").to_pylist() == [None]
    assert attempts == 1


@pytest.mark.parametrize(
    ("location", "status"),
    [
        ("status_code", 408),
        ("status_code", 409),
        ("status_code", 425),
        ("status_code", 429),
        ("status_code", 500),
        ("status_code", 501),
        ("status_code", 599),
        ("code", 429),
        ("code", 503),
        ("response.status_code", 503),
        ("response.status", 429),
    ],
)
def test_prompt_retries_transient_provider_errors(monkeypatch, location, status):
    import asyncio
    from types import SimpleNamespace

    from vane.ai.functions import _PromptBatch

    attempts = 0

    class TransientError(RuntimeError):
        pass

    error = TransientError("temporary")
    if location == "status_code":
        error.status_code = status
    elif location == "code":
        error.code = status
    elif location == "response.status_code":
        error.response = SimpleNamespace(status_code=status)
    else:
        error.response = SimpleNamespace(status=status)

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    class Prompter:
        async def prompt(self, _messages):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise error
            return "recovered"

    class Descriptor:
        def instantiate(self):
            return Prompter()

    wrapper = _PromptBatch(
        Descriptor(),
        ["message"],
        "response",
        max_concurrency_per_actor=1,
        max_retries=1,
        on_error="raise",
    )

    result = _drive(wrapper, pa.table({"message": ["retry"]}))

    assert result.column("response").to_pylist() == ["recovered"]
    assert attempts == 2
