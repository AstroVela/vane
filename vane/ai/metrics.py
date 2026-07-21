# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Token usage metrics for AI providers.

Provides a lightweight, thread-safe mechanism for recording and querying
token usage across all AI provider calls (prompt, embed, classify).

Usage::

    from vane.ai.metrics import record_token_metrics, get_token_metrics

    # Called automatically by providers after each API response:
    record_token_metrics(
        protocol="prompt",
        model="gpt-4o",
        provider="openai",
        input_tokens=150,
        output_tokens=42,
        total_tokens=192,
    )

    # Query accumulated metrics:
    metrics = get_token_metrics()
    # [TokenMetricsEntry(protocol='prompt', model='gpt-4o', provider='openai',
    #                    input_tokens=150, output_tokens=42, total_tokens=192, requests=1)]

    # Reset:
    reset_token_metrics()

    # Optional callback for real-time streaming to external systems:
    set_token_metrics_callback(lambda entry: print(entry))
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class TokenMetricsEntry:
    """Accumulated token usage for a (protocol, model, provider) key."""

    protocol: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0


# Thread-safe global state
_lock = threading.Lock()
_counters: dict[tuple[str, str, str], TokenMetricsEntry] = {}
_callback: Callable[[dict[str, Any]], None] | None = None


def record_token_metrics(
    protocol: str,
    model: str,
    provider: str,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
) -> None:
    """Record token usage from a single API response.

    Args:
        protocol: The AI protocol (``"prompt"``, ``"embed"``, ``"classify"``).
        model: The model name (e.g. ``"gpt-4o"``).
        provider: The provider name (e.g. ``"openai"``).
        input_tokens: Number of input/prompt tokens consumed.
        output_tokens: Number of output/completion tokens generated.
        total_tokens: Total tokens (input + output).
    """
    key = (protocol, model, provider)
    with _lock:
        entry = _counters.get(key)
        if entry is None:
            entry = TokenMetricsEntry(protocol=protocol, model=model, provider=provider)
            _counters[key] = entry
        if input_tokens is not None:
            entry.input_tokens += input_tokens
        if output_tokens is not None:
            entry.output_tokens += output_tokens
        if total_tokens is not None:
            entry.total_tokens += total_tokens
        elif input_tokens is not None or output_tokens is not None:
            # Providers that don't report a total still get a consistent one,
            # derived from whatever split counts they did report.
            entry.total_tokens += (input_tokens or 0) + (output_tokens or 0)
        entry.requests += 1

    # Fire optional callback (outside lock)
    cb = _callback
    if cb is not None:
        try:
            cb(
                {
                    "protocol": protocol,
                    "model": model,
                    "provider": provider,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
            )
        except Exception:
            pass


def get_token_metrics() -> list[TokenMetricsEntry]:
    """Return a snapshot of all accumulated token metrics.

    Entries are copies: mutating a returned entry does not affect the
    collector's internal counters or later summaries.
    """
    with _lock:
        return [replace(entry) for entry in _counters.values()]


def get_token_metrics_summary() -> dict[str, Any]:
    """Return a summary dict with totals across all providers.

    Returns a dict like::

        {
            "total_input_tokens": 1234,
            "total_output_tokens": 567,
            "total_tokens": 1801,
            "total_requests": 42,
            "by_provider": {
                "openai": {"input_tokens": 1000, "output_tokens": 500, ...},
                ...
            },
        }
    """
    total_in = total_out = total_tok = total_req = 0
    by_provider: dict[str, dict[str, int]] = defaultdict(
        lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "requests": 0}
    )
    # Read every field while holding the lock so the summary is a consistent
    # point-in-time snapshot; copying entry references and reading later would
    # let a concurrent record() tear the numbers.
    with _lock:
        for e in _counters.values():
            total_in += e.input_tokens
            total_out += e.output_tokens
            total_tok += e.total_tokens
            total_req += e.requests
            p = by_provider[e.provider]
            p["input_tokens"] += e.input_tokens
            p["output_tokens"] += e.output_tokens
            p["total_tokens"] += e.total_tokens
            p["requests"] += e.requests

    return {
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_tokens": total_tok,
        "total_requests": total_req,
        "by_provider": dict(by_provider),
    }


def reset_token_metrics() -> None:
    """Reset all accumulated token metrics."""
    with _lock:
        _counters.clear()


def set_token_metrics_callback(
    callback: Callable[[dict[str, Any]], None] | None,
) -> None:
    """Set an optional callback invoked after each ``record_token_metrics`` call.

    The callback receives a dict with keys: ``protocol``, ``model``,
    ``provider``, ``input_tokens``, ``output_tokens``, ``total_tokens``.
    Pass ``None`` to remove the callback.
    """
    global _callback
    _callback = callback
