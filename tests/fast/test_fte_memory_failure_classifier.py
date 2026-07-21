# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from duckdb.runners.fte.fte_failures import _is_memory_failure


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "Query exceeded per-node memory limit"},
        {"message": "OOM while hashing build side"},
        {"message": "worker oom-killed by cgroup"},
        {"message": "OUT_OF_MEMORY on node 3"},
        {"message": "task failed: out of memory"},
        "out of memory",
    ],
    ids=["memory-limit", "oom-upper", "oom-killed", "out-of-memory-code-in-text", "plain-text", "bare-string"],
)
def test_memory_failure_text_positive(payload):
    assert _is_memory_failure(payload) is True


@pytest.mark.parametrize(
    "payload",
    [
        {"error_code": "OUT_OF_MEMORY"},
        {"error_code": "EXCEEDED_LOCAL_MEMORY_LIMIT"},
        {"error_code": "exceeded-local-memory-limit"},
        {"code": "MEMORY_LIMIT_EXCEEDED"},
        {"errorCode": "OOM"},
    ],
    ids=["out-of-memory", "local-limit", "normalized-separators", "code-key", "camel-key"],
)
def test_memory_failure_structured_codes(payload):
    assert _is_memory_failure(payload) is True


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "bloom filter construction failed"},
        {"message": "no room left in queue"},
        {"message": "zoom level invalid"},
        {"message": "mushroom cloud render failed"},
        {"error_code": "BLOOM_FILTER_ERROR"},
        {"message": "boomerang scheduling retry"},
        None,
        "",
    ],
    ids=["bloom", "room", "zoom", "mushroom", "bloom-code", "boomerang", "none", "empty"],
)
def test_memory_failure_negative(payload):
    assert _is_memory_failure(payload) is False


def test_memory_failure_case_insensitive():
    assert _is_memory_failure({"message": "Out Of Memory"}) is True
    assert _is_memory_failure({"message": "MEMORY LIMIT reached"}) is True
