# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import pytest

from vane.ai.metrics import (
    get_token_metrics,
    get_token_metrics_summary,
    record_token_metrics,
    reset_token_metrics,
)


@pytest.fixture(autouse=True)
def _clean_metrics():
    reset_token_metrics()
    yield
    reset_token_metrics()


def test_mutating_snapshot_does_not_change_internal_state():
    record_token_metrics("prompt", "m", "p", input_tokens=10, output_tokens=5, total_tokens=15)

    snapshot = get_token_metrics()
    snapshot[0].input_tokens = 999_999
    snapshot[0].requests = 999_999

    fresh = get_token_metrics()
    assert fresh[0].input_tokens == 10
    assert fresh[0].requests == 1
    summary = get_token_metrics_summary()
    assert summary["total_input_tokens"] == 10
    assert summary["total_requests"] == 1


def test_total_tokens_derived_when_provider_omits_it():
    record_token_metrics("prompt", "m", "p", input_tokens=10, output_tokens=5)
    entry = get_token_metrics()[0]
    assert entry.total_tokens == 15

    # A reported total is used as-is, never double-counted.
    record_token_metrics("prompt", "m", "p", input_tokens=1, output_tokens=1, total_tokens=2)
    entry = get_token_metrics()[0]
    assert entry.total_tokens == 17


def test_summary_is_consistent_under_concurrent_records():
    stop = threading.Event()

    def writer():
        while not stop.is_set():
            # total omitted on purpose: derivation keeps in+out == total.
            record_token_metrics("prompt", "m", "p", input_tokens=1, output_tokens=1)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for _ in range(200):
            summary = get_token_metrics_summary()
            # Point-in-time consistency: totals derived under one lock hold
            # can never tear apart from each other.
            assert summary["total_tokens"] == summary["total_input_tokens"] + summary["total_output_tokens"]
            assert summary["total_input_tokens"] == summary["total_requests"]
    finally:
        stop.set()
        for t in threads:
            t.join()


def test_summary_by_provider_copies_are_caller_owned():
    record_token_metrics("prompt", "m", "openai", input_tokens=3)
    summary = get_token_metrics_summary()
    summary["by_provider"]["openai"]["input_tokens"] = 12345

    assert get_token_metrics_summary()["by_provider"]["openai"]["input_tokens"] == 3
