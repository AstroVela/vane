# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DuckDBPyRelation AI method integration (monkey-patch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

import duckdb
import vane
from vane.ai.protocols import (
    PrompterDescriptor,
    TextClassifierDescriptor,
    TextEmbedderDescriptor,
)
from vane.ai.provider import Provider
from vane.ai.typing import EmbeddingDimensions

if TYPE_CHECKING:
    from vane.ai.protocols import Prompter, TextClassifier, TextEmbedder
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


class MockPrompter:
    async def prompt(self, messages: tuple[object, ...]) -> str:
        return "prompt:" + ":".join(part if isinstance(part, str) else bytes(part).hex() for part in messages)


@dataclass
class MockPrompterDescriptor(PrompterDescriptor):
    def get_provider(self) -> str:
        return "mock"

    def get_model(self) -> str:
        return "mock-prompter"

    def get_options(self) -> Options:
        return {"batch_size": 2}

    def instantiate(self) -> Prompter:
        return MockPrompter()


class MockProvider(Provider):
    @property
    def name(self) -> str:
        return "mock"

    def get_text_embedder(self, model=None, dimensions=None, **options):
        return MockTextEmbedderDescriptor(dim=dimensions or 4)

    def get_text_classifier(self, model=None, **options):
        return MockTextClassifierDescriptor()

    def get_prompter(self, model=None, system_message=None, **options):
        return MockPrompterDescriptor()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRelationPatch:
    """Verify the AI relation methods execute and preserve source columns."""

    def test_embed_on_relation(self):
        """rel.embed() preserves source columns and appends embeddings."""
        conn = duckdb.connect()
        rel = conn.sql("SELECT 'hello' AS text UNION ALL SELECT 'world' AS text")

        result = rel.embed(vane.col("text"), provider=MockProvider())
        rows = result.fetchall()
        assert len(rows) == 2
        for row in rows:
            assert row[0] in {"hello", "world"}
            assert len(row[1]) == 4

    def test_classify_text_on_relation(self):
        """rel.classify_text() produces labels."""
        conn = duckdb.connect()
        rel = conn.sql("SELECT 'great' AS text UNION ALL SELECT 'bad' AS text")

        result = rel.classify_text("text", labels=["positive", "negative"], provider=MockProvider())
        rows = result.fetchall()
        assert len(rows) == 2
        for row in rows:
            assert row[0] == "positive"

    def test_prompt_on_relation(self):
        conn = vane.connect()
        rel = conn.sql("SELECT 1 AS id, 'hello' AS text, from_hex('89504e47') AS image")

        result = rel.prompt(
            [vane.col("text"), vane.col("image")],
            provider=MockProvider(),
            output_column="answer",
        )

        assert result.columns == ["id", "text", "image", "answer"]
        assert result.fetchone() == (1, "hello", bytes.fromhex("89504e47"), "prompt:hello:89504e47")

    def test_prompt_replaces_existing_output_column(self):
        conn = vane.connect()
        rel = conn.sql("SELECT 'hello' AS text, 'old' AS response")

        result = rel.prompt(vane.col("text"), provider=MockProvider())

        assert result.columns == ["text", "response"]
        assert result.fetchone() == ("hello", "prompt:hello")

    def test_methods_exist_on_relation(self):
        """DuckDBPyRelation has the patched methods."""
        assert hasattr(duckdb.DuckDBPyRelation, "embed")
        assert not hasattr(duckdb.DuckDBPyRelation, "embed_text")
        assert hasattr(duckdb.DuckDBPyRelation, "classify_text")
        assert hasattr(duckdb.DuckDBPyRelation, "prompt")

    def test_patch_is_idempotent(self):
        """Importing the patch module again doesn't break anything."""
        import vane.ai._relation_patch

        vane.ai._relation_patch._patch()
        assert hasattr(duckdb.DuckDBPyRelation, "embed")

    def test_embed_chaining(self):
        """embed returns a relation that can be further queried."""
        conn = duckdb.connect()
        rel = conn.sql("SELECT 'test' AS text")

        result = rel.embed(vane.col("text"), provider=MockProvider())
        # Should be queryable — count rows
        count = result.aggregate("count(*)").fetchone()
        assert count[0] == 1
