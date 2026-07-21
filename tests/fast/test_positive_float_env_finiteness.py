# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from duckdb.execution.udf_ray_actor_pool import _positive_float_env as _actor_pool_env
from duckdb.execution.udf_subprocess import _positive_float_env as _subprocess_env
from duckdb.runners.ray.safe_get import _positive_float_env as _safe_get_env

_ENV = "VANE_TEST_POSITIVE_FLOAT"


def _subprocess(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, raw)
    return _subprocess_env(_ENV, 7.5)


def _actor_pool(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, raw)
    return _actor_pool_env(_ENV, 7.5)


def _safe_get(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, raw)
    return _safe_get_env(_ENV)


_CONSUMERS = [_subprocess, _actor_pool, _safe_get]
_CONSUMER_IDS = ["udf-subprocess", "ray-actor-pool", "ray-safe-get"]


@pytest.mark.parametrize("consumer", _CONSUMERS, ids=_CONSUMER_IDS)
@pytest.mark.parametrize("raw, expected", [("0", 0.0), ("2.5", 2.5), ("30", 30.0)])
def test_accepts_finite_non_negative(monkeypatch, consumer, raw, expected):
    assert consumer(monkeypatch, raw) == expected


@pytest.mark.parametrize("consumer", _CONSUMERS, ids=_CONSUMER_IDS)
@pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "Infinity", "-inf", "-1", "-0.5"])
def test_rejects_nan_infinity_and_negative(monkeypatch, consumer, raw):
    with pytest.raises(ValueError, match="finite and non-negative"):
        consumer(monkeypatch, raw)


@pytest.mark.parametrize("consumer", _CONSUMERS, ids=_CONSUMER_IDS)
@pytest.mark.parametrize("raw", ["abc", "1.2.3"])
def test_rejects_invalid_strings(monkeypatch, consumer, raw):
    with pytest.raises(ValueError):
        consumer(monkeypatch, raw)


def test_unset_semantics_preserved(monkeypatch):
    # Documented behavior for an unset variable: defaults for the two
    # default-taking helpers, None for the safe-get helper.
    assert _subprocess(monkeypatch, None) == 7.5
    assert _actor_pool(monkeypatch, None) == 7.5
    assert _safe_get(monkeypatch, None) is None
