# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DuckDBPyRelation AI method integration (monkey-patch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

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

# ---------------------------------------------------------------------------
# Mock implementations (same as test_ai_functions.py)
# ---------------------------------------------------------------------------


class MockTextEmbedder:
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
    @property
    def name(self) -> str:
        return "mock"

    def get_text_embedder(self, model=None, dimensions=None, **options):
        return MockTextEmbedderDescriptor(dim=dimensions or 4)

    def get_text_classifier(self, model=None, **options):
        return MockTextClassifierDescriptor()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRelationPatch:
    """Verify .embed_text() and .classify_text() work as relation methods."""

    def test_embed_text_on_relation(self):
        """rel.embed_text() produces embeddings."""
        conn = vane.connect()
        rel = conn.sql("SELECT 'hello' AS text UNION ALL SELECT 'world' AS text")

        result = rel.embed_text("text", provider=MockProvider())
        rows = result.fetchall()
        assert len(rows) == 2
        for row in rows:
            assert len(row[0]) == 4

    def test_classify_text_on_relation(self):
        """rel.classify_text() produces labels."""
        conn = vane.connect()
        rel = conn.sql("SELECT 'great' AS text UNION ALL SELECT 'bad' AS text")

        result = rel.classify_text("text", labels=["positive", "negative"], provider=MockProvider())
        rows = result.fetchall()
        assert len(rows) == 2
        for row in rows:
            assert row[0] == "positive"

    def test_methods_exist_on_relation(self):
        """DuckDBPyRelation has the patched methods."""
        assert hasattr(vane.DuckDBPyRelation, "embed_text")
        assert hasattr(vane.DuckDBPyRelation, "classify_text")
        assert hasattr(vane.DuckDBPyRelation, "prompt")

    def test_patch_is_idempotent(self):
        """Importing the patch module again doesn't break anything."""
        import vane.ai._relation_patch

        vane.ai._relation_patch._patch()
        assert hasattr(vane.DuckDBPyRelation, "embed_text")

    def test_embed_text_chaining(self):
        """embed_text returns a relation that can be further queried."""
        conn = vane.connect()
        rel = conn.sql("SELECT 'test' AS text")

        result = rel.embed_text("text", provider=MockProvider())
        # Should be queryable — count rows
        count = result.aggregate("count(*)").fetchone()
        assert count[0] == 1
