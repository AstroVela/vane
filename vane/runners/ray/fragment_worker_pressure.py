# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vane.runners.fte import FteTaskAttemptId


def attempt_key(attempt_id: Any) -> str:
    return str(FteTaskAttemptId.coerce(attempt_id))


def partition_reservation_key(
    query_id: str,
    fragment_id: str,
    partition_id: int,
) -> tuple[str, str, int]:
    query_key = str(query_id or "").strip()
    fragment_key = str(fragment_id or "").strip()
    if not query_key or not fragment_key:
        raise ValueError("partition reservation requires query_id and fragment_id")
    return query_key, fragment_key, int(partition_id)


def initial_split_count(request: Mapping[str, Any]) -> int:
    return sum(len(splits or []) for splits in (request.get("initial_splits") or {}).values())


def split_payload_bytes(split: Mapping[str, Any]) -> int:
    size_bytes = split.get("size_bytes")
    if size_bytes is not None:
        try:
            return max(0, int(size_bytes))
        except (TypeError, ValueError):
            pass
    data = split.get("data")
    if isinstance(data, (bytes, bytearray)):
        return len(data)
    if isinstance(data, memoryview):
        return len(data.tobytes())
    if isinstance(data, str):
        return len(data.encode("utf-8"))
    fragments = split.get("fragments")
    if isinstance(fragments, (list, tuple)):
        return len(fragments)
    return len(repr(split).encode("utf-8"))


def initial_split_bytes(request: Mapping[str, Any]) -> int:
    total = 0
    for splits in (request.get("initial_splits") or {}).values():
        for split in splits or []:
            if isinstance(split, Mapping):
                total += split_payload_bytes(split)
    return total
