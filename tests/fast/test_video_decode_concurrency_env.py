# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from duckdb.datasource.video_reader import _max_concurrent_decodes_from_env


@pytest.mark.parametrize("raw, expected", [("1", 1), ("2", 2), ("64", 64)])
def test_accepts_positive_integers(monkeypatch, raw, expected):
    monkeypatch.setenv("VANE_MAX_CONCURRENT_DECODES", raw)
    assert _max_concurrent_decodes_from_env() == expected


def test_unset_defaults_to_one(monkeypatch):
    monkeypatch.delenv("VANE_MAX_CONCURRENT_DECODES", raising=False)
    assert _max_concurrent_decodes_from_env() == 1


@pytest.mark.parametrize("raw", ["0", "-1", "-64"], ids=["zero", "negative-one", "negative"])
def test_rejects_non_positive_values(monkeypatch, raw):
    # Semaphore(0) blocks every decode forever; fail loudly at load instead.
    monkeypatch.setenv("VANE_MAX_CONCURRENT_DECODES", raw)
    with pytest.raises(ValueError, match=r"VANE_MAX_CONCURRENT_DECODES.*>= 1.*" + raw):
        _max_concurrent_decodes_from_env()


@pytest.mark.parametrize("raw", ["", "two", "1.5", "nan"], ids=["empty", "word", "float", "nan"])
def test_rejects_invalid_values(monkeypatch, raw):
    monkeypatch.setenv("VANE_MAX_CONCURRENT_DECODES", raw)
    with pytest.raises(ValueError, match="VANE_MAX_CONCURRENT_DECODES"):
        _max_concurrent_decodes_from_env()
