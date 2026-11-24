# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Backend-independent capacity transitions for streaming UDF output.

Stream slots bound unread blocks. Byte windows bound delivery to the native
consumer; storage leases independently retain the underlying allocation.
Neither kind of window applies to terminal/control events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STREAM_BUFFER_BLOCKS = 2


@dataclass(frozen=True)
class StreamCapacity:
    rows: int
    bytes: int | None = None
    item_bytes: int | None = None

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> StreamCapacity:
        if not isinstance(raw, dict) or "rows" not in raw:
            raise ValueError("UDF drain capacity requires an event count")
        return cls(
            rows=max(0, int(raw["rows"])),
            bytes=None if raw.get("bytes") is None else max(0, int(raw["bytes"])),
            item_bytes=None if raw.get("item_bytes") is None else max(0, int(raw["item_bytes"])),
        )


class StreamReadWindow:
    """One downstream capacity publication, shared by both stream adapters.

    A nonempty window admits one complete block even when it exceeds a soft
    byte target. Subsequent blocks wait for a fresh capacity publication.
    Physical storage capacity is checked separately by the backend.
    """

    def __init__(self, capacity: StreamCapacity) -> None:
        self.remaining = capacity
        self.delivered = 0

    def may_read(self, pending: int = 0) -> bool:
        capacity = self.remaining
        return (
            capacity.rows > pending
            and (capacity.bytes is None or capacity.bytes > 0)
            and (capacity.item_bytes is None or capacity.item_bytes > 0)
        )

    def take(self, size_bytes: int) -> bool:
        if not self.may_read():
            return False
        capacity = self.remaining
        size = max(0, int(size_bytes))
        if self.delivered and (
            (capacity.bytes is not None and size > capacity.bytes)
            or (capacity.item_bytes is not None and size > capacity.item_bytes)
        ):
            return False
        self.remaining = StreamCapacity(
            rows=capacity.rows - 1,
            bytes=None if capacity.bytes is None else max(0, capacity.bytes - size),
            item_bytes=capacity.item_bytes,
        )
        self.delivered += 1
        return True
