# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retry and on_error semantics hold on every batch path.

Covers vane#148: the chunked embedding path goes through the retry helper and
honours ``on_error`` (substituting with row count preserved), ``on_error="log"``
actually emits a redaction-safe warning, a failed batch call falls back to
per-row attempts so only genuinely-bad rows are substituted, and
``RetryAfterError`` survives pickling with a float ``retry_after`` and its
cause intact.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import pickle
import sys
import types
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest


def _load_functions() -> types.ModuleType:
    """Import the real ``vane.ai.functions``, even under the no-duckdb harness.

    The stub harness plugin registers a placeholder for ``vane.ai.functions``
    because the real module imports duckdb-backed modules at top level. When
    that placeholder is present, evict it and stub just the duckdb-importing
    dependencies so the real module under test can load. On CI (where duckdb
    imports fine) the real module imports normally and nothing is stubbed.
    """
    module = sys.modules.get("vane.ai.functions")
    if getattr(module, "__file__", None):
        return module  # real module already loaded
    if module is not None:
        sys.modules.pop("vane.ai.functions")
        stub_specs: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("vane._expressions", ("as_expression", "is_expression")),
            ("vane._expression_udf", ("_build_actor_map_batches_expression",)),
        )
        for name, attrs in stub_specs:
            if name not in sys.modules:
                stub = types.ModuleType(name)
                for attr in attrs:
                    setattr(stub, attr, lambda *a, **k: None)
                sys.modules[name] = stub
    return importlib.import_module("vane.ai.functions")


functions = _load_functions()

_EmbedTextBatch = functions._EmbedTextBatch
_PromptBatch = functions._PromptBatch
RetryAfterError = functions.RetryAfterError

DIM = 3
ONES = [1.0] * DIM
ZEROS = [0.0] * DIM


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    monkeypatch.setattr(functions.time, "sleep", lambda _s: None)


def test_real_functions_module_is_under_test() -> None:
    """Guard: the harness must import the real module, not the plugin stub."""
    assert getattr(functions, "__file__", None)
    assert isinstance(_EmbedTextBatch, type)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FlakyEmbedder:
    """Raises whenever a submitted batch contains a text with ``BAD``."""

    def __init__(self, fail_first: int = 0) -> None:
        self.calls: list[list[str]] = []
        self._fail_first = fail_first

    def embed_text(self, texts: list[str]) -> list[np.ndarray]:
        self.calls.append(list(texts))
        if self._fail_first > 0:
            self._fail_first -= 1
            raise RuntimeError("transient provider error")
        if any("BAD" in t for t in texts):
            raise RuntimeError("provider rejected batch")
        return [np.full(DIM, 1.0, dtype=np.float32) for _ in texts]


class FakeEmbedderDescriptor:
    def __init__(self, embedder) -> None:
        self._embedder = embedder

    def instantiate(self):
        return self._embedder

    def get_dimensions(self):
        return SimpleNamespace(size=DIM)


class FlakyBatchPrompter:
    """Batch-only prompter that raises when a batch contains ``BAD``."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def prompt_batch(self, texts: list[str]) -> list[str]:
        self.calls.append(list(texts))
        if any("BAD" in t for t in texts):
            raise RuntimeError("provider rejected batch")
        return [f"r:{t}" for t in texts]


class FakePrompterDescriptor:
    def __init__(self, prompter) -> None:
        self._prompter = prompter

    def instantiate(self):
        return self._prompter


# ---------------------------------------------------------------------------
# 8. Chunked embedding respects retry + on_error
# ---------------------------------------------------------------------------


class TestChunkedEmbeddingRetry:
    def test_chunked_failure_substitutes_and_preserves_row_count(self):
        embedder = FlakyEmbedder()
        wrapper = _EmbedTextBatch(
            FakeEmbedderDescriptor(embedder),
            "text",
            "embedding",
            max_chunk_chars=5,
            max_retries=0,
            on_error="ignore",
        )
        table = pa.table({"text": ["BAD text long enough to chunk", "BAD too"]})

        result = wrapper(table)

        values = result.column("embedding").to_pylist()
        assert len(values) == table.num_rows
        assert values == [ZEROS, ZEROS]

    def test_chunked_path_retries_transient_failures(self):
        embedder = FlakyEmbedder(fail_first=1)
        wrapper = _EmbedTextBatch(
            FakeEmbedderDescriptor(embedder),
            "text",
            "embedding",
            max_chunk_chars=5,
            max_retries=2,
            on_error="raise",
        )
        table = pa.table({"text": ["a long text that chunks", "hi"]})

        result = wrapper(table)

        assert len(embedder.calls) == 2  # first attempt failed, retry succeeded
        assert all(v is not None for v in result.column("embedding").to_pylist())

    def test_chunked_path_still_raises_when_on_error_raise(self):
        embedder = FlakyEmbedder()
        wrapper = _EmbedTextBatch(
            FakeEmbedderDescriptor(embedder),
            "text",
            "embedding",
            max_chunk_chars=5,
            max_retries=0,
            on_error="raise",
        )
        table = pa.table({"text": ["BAD text long enough to chunk"]})

        with pytest.raises(RuntimeError, match="rejected batch"):
            wrapper(table)


# ---------------------------------------------------------------------------
# 9. on_error="log" actually logs
# ---------------------------------------------------------------------------


class TestOnErrorLog:
    def _boom(self):
        raise ValueError("upstream rate limited")

    def test_retry_call_logs_on_substitution(self, caplog):
        with caplog.at_level(logging.WARNING, logger="vane.ai.functions"):
            result = functions._retry_call(self._boom, max_retries=0, on_error="log", default="sub")

        assert result == "sub"
        records = [r for r in caplog.records if r.name == "vane.ai.functions"]
        assert len(records) == 1
        message = records[0].getMessage()
        assert "ValueError" in message
        assert "upstream rate limited" in message

    def test_retry_call_async_logs_on_substitution(self, caplog):
        async def boom():
            raise ValueError("async failure")

        with caplog.at_level(logging.WARNING, logger="vane.ai.functions"):
            result = asyncio.run(functions._retry_call_async(boom, max_retries=0, on_error="log", default=None))

        assert result is None
        records = [r for r in caplog.records if r.name == "vane.ai.functions"]
        assert len(records) == 1
        assert "ValueError" in records[0].getMessage()

    def test_on_error_ignore_stays_silent(self, caplog):
        with caplog.at_level(logging.WARNING, logger="vane.ai.functions"):
            result = functions._retry_call(self._boom, max_retries=0, on_error="ignore", default=None)

        assert result is None
        assert [r for r in caplog.records if r.name == "vane.ai.functions"] == []


# ---------------------------------------------------------------------------
# 10. Per-row fallback before substitution
# ---------------------------------------------------------------------------


class MalformedEmbedder:
    """Fails multi-row batches; answers single-row calls with a short response."""

    def embed_text(self, texts: list[str]) -> list[np.ndarray]:
        if len(texts) > 1:
            raise RuntimeError("provider rejected batch")
        return []  # malformed: no embedding for the requested row


class TestPerRowSubstitution:
    def test_per_row_fallback_uses_single_attempts(self):
        """Rows get one attempt each — the batch already exhausted the retry budget."""
        embedder = FlakyEmbedder()
        wrapper = _EmbedTextBatch(
            FakeEmbedderDescriptor(embedder),
            "text",
            "embedding",
            max_retries=2,
            on_error="ignore",
        )
        table = pa.table({"text": ["good one", "BAD row"]})

        values = wrapper(table).column("embedding").to_pylist()

        assert values == [ONES, ZEROS]
        # Batch pass: 1 + 2 retries = 3 calls; per-row fallback: 1 call per row.
        batch_calls = [c for c in embedder.calls if len(c) == 2]
        row_calls = [c for c in embedder.calls if len(c) == 1]
        assert len(batch_calls) == 3
        assert len(row_calls) == 2

    def test_malformed_short_response_substitutes_that_row(self):
        """A wrong-length per-row response counts as that row's failure, not success."""
        wrapper = _EmbedTextBatch(
            FakeEmbedderDescriptor(MalformedEmbedder()),
            "text",
            "embedding",
            max_retries=0,
            on_error="ignore",
        )
        table = pa.table({"text": ["row a", "row b"]})

        values = wrapper(table).column("embedding").to_pylist()

        assert values == [ZEROS, ZEROS]

    def test_embed_batch_failure_substitutes_only_bad_rows(self):
        embedder = FlakyEmbedder()
        wrapper = _EmbedTextBatch(
            FakeEmbedderDescriptor(embedder),
            "text",
            "embedding",
            max_retries=0,
            on_error="ignore",
        )
        table = pa.table({"text": ["good one", "BAD row", "good two"]})

        values = wrapper(table).column("embedding").to_pylist()

        assert values == [ONES, ZEROS, ONES]

    def test_embed_chunked_batch_failure_substitutes_only_bad_rows(self):
        embedder = FlakyEmbedder()
        wrapper = _EmbedTextBatch(
            FakeEmbedderDescriptor(embedder),
            "text",
            "embedding",
            max_chunk_chars=6,
            chunk_overlap_chars=0,
            max_retries=0,
            on_error="ignore",
        )
        # Every chunk of the bad row carries the failure marker; the good
        # multi-chunk row is L2-normalised by the weighted chunk average.
        table = pa.table({"text": ["goodgoodgoodgood", "BADBADBADBADBAD", "ok"]})

        values = wrapper(table).column("embedding").to_pylist()

        assert len(values) == 3
        assert values[0] == pytest.approx([1.0 / np.sqrt(DIM)] * DIM)
        assert values[1] == ZEROS
        assert values[2] == pytest.approx(ONES)

    def test_prompt_batch_failure_substitutes_only_bad_rows(self):
        prompter = FlakyBatchPrompter()
        wrapper = _PromptBatch(
            FakePrompterDescriptor(prompter),
            "text",
            "response",
            max_retries=0,
            on_error="ignore",
        )
        table = pa.table({"text": ["a", "BAD", "b"]})

        values = wrapper(table).column("response").to_pylist()

        assert values == ["r:a", None, "r:b"]

    def test_prompt_batch_failure_still_raises_by_default(self):
        prompter = FlakyBatchPrompter()
        wrapper = _PromptBatch(
            FakePrompterDescriptor(prompter),
            "text",
            "response",
            max_retries=0,
            on_error="raise",
        )
        table = pa.table({"text": ["a", "BAD"]})

        with pytest.raises(RuntimeError, match="rejected batch"):
            wrapper(table)

    def test_embed_batch_success_does_not_fan_out_per_row(self):
        embedder = FlakyEmbedder()
        wrapper = _EmbedTextBatch(
            FakeEmbedderDescriptor(embedder),
            "text",
            "embedding",
            max_retries=0,
            on_error="ignore",
        )
        table = pa.table({"text": ["good one", "good two"]})

        wrapper(table)

        assert embedder.calls == [["good one", "good two"]]


# ---------------------------------------------------------------------------
# 11. RetryAfterError pickling
# ---------------------------------------------------------------------------


class TestRetryAfterErrorPickle:
    def test_roundtrip_preserves_retry_after_and_cause(self):
        original = ValueError("throttled")
        error = RetryAfterError(2.5, original)

        restored = pickle.loads(pickle.dumps(error))

        assert isinstance(restored, RetryAfterError)
        assert isinstance(restored.retry_after, float)
        assert restored.retry_after == 2.5
        assert isinstance(restored.__cause__, ValueError)
        assert str(restored.__cause__) == "throttled"

    def test_roundtrip_without_cause(self):
        restored = pickle.loads(pickle.dumps(RetryAfterError(1.0)))

        assert restored.retry_after == 1.0
        assert restored.__cause__ is None
