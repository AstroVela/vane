# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from vane.ai.functions import _weighted_average_embeddings, chunk_text


def test_multi_chunk_content_with_overlap():
    text = "abcdefghij" * 10  # 100 chars
    chunks = chunk_text(text, max_chars=40, overlap_chars=10)
    assert len(chunks) > 1
    assert all(chunks), "no empty chunks"
    assert all(len(c) <= 40 for c in chunks)
    # Consecutive chunks overlap by exactly overlap_chars until the tail.
    assert chunks[0][-10:] == chunks[1][:10]
    # Reassembling with the overlap stripped restores the original text.
    reassembled = chunks[0] + "".join(c[10:] for c in chunks[1:])
    assert reassembled == text


def test_short_text_returns_single_chunk():
    assert chunk_text("short", max_chars=2000, overlap_chars=200) == ["short"]


@pytest.mark.parametrize("max_chars", [0, -1, -2000], ids=["zero", "negative-one", "negative"])
def test_rejects_non_positive_max_chars(max_chars):
    with pytest.raises(ValueError, match="max_chars must be a positive integer"):
        chunk_text("some text", max_chars=max_chars)


@pytest.mark.parametrize("max_chars", [2.5, float("nan"), "2000", None, True])
def test_rejects_non_integer_max_chars(max_chars):
    with pytest.raises(ValueError):
        chunk_text("some text", max_chars=max_chars)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overlap",
    [-1, 40, 41],
    ids=["negative", "equal-to-max", "greater-than-max"],
)
def test_rejects_out_of_range_overlap(overlap):
    with pytest.raises(ValueError, match="0 <= overlap_chars < max_chars"):
        chunk_text("x" * 100, max_chars=40, overlap_chars=overlap)


@pytest.mark.parametrize("overlap", [2.5, "10", None, False])
def test_rejects_non_integer_overlap(overlap):
    with pytest.raises(ValueError):
        chunk_text("x" * 100, max_chars=40, overlap_chars=overlap)  # type: ignore[arg-type]


def test_zero_overlap_is_valid():
    chunks = chunk_text("abcdef", max_chars=2, overlap_chars=0)
    assert chunks == ["ab", "cd", "ef"]


def test_weighted_average_survives_zero_weights():
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    result = _weighted_average_embeddings(embeddings, [0.0, 0.0])
    assert np.all(np.isfinite(result)), "zero weights must not produce NaN"
    # Falls back to the unweighted mean (then normalized).
    expected = np.array([0.5, 0.5]) / np.linalg.norm([0.5, 0.5])
    np.testing.assert_allclose(result, expected.astype(np.float32), rtol=1e-6)


def test_weighted_average_normal_path_unchanged():
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    result = _weighted_average_embeddings(embeddings, [3.0, 1.0])
    assert np.all(np.isfinite(result))
    assert result[0] > result[1]
