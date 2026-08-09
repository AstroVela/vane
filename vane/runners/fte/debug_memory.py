# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any


def debug_flag_enabled(*names: str) -> bool:
    for name in names:
        value = os.getenv(name, "")
        if value.strip().lower() not in ("", "0", "false", "no", "off"):
            return True
    return False


def process_memory_snapshot() -> dict[str, int]:
    snapshot: dict[str, int] = {"pid": os.getpid()}
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/self/statm", encoding="utf-8") as statm:
            fields = statm.read().split()
        if len(fields) >= 2:
            snapshot["rss_bytes"] = int(fields[1]) * int(page_size)
    except Exception:
        pass

    try:
        with open("/proc/self/smaps_rollup", encoding="utf-8") as smaps:
            for line in smaps:
                if ":" not in line:
                    continue
                key, raw_value = line.split(":", 1)
                parts = raw_value.strip().split()
                if not parts:
                    continue
                try:
                    value_bytes = int(parts[0]) * 1024
                except ValueError:
                    continue
                if key == "Rss":
                    snapshot["smaps_rss_bytes"] = value_bytes
                elif key == "Pss":
                    snapshot["pss_bytes"] = value_bytes
                elif key == "Private_Clean":
                    snapshot["private_clean_bytes"] = value_bytes
                elif key == "Private_Dirty":
                    snapshot["private_dirty_bytes"] = value_bytes
                elif key == "Swap":
                    snapshot["swap_bytes"] = value_bytes
    except Exception:
        pass

    private_bytes = snapshot.get("private_clean_bytes", 0) + snapshot.get("private_dirty_bytes", 0)
    if private_bytes:
        snapshot["private_bytes"] = private_bytes
    return snapshot


def format_debug_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace(" ", "_")


def log_debug(prefix: str, event: str, **fields: Any) -> None:
    parts = [f"event={format_debug_value(event)}"]
    parts.extend(f"{key}={format_debug_value(value)}" for key, value in fields.items())
    print(f"[{prefix} pid={os.getpid()}] " + " ".join(parts), file=sys.stderr, flush=True)


def _metadata_rows_and_bytes(metadata: Any) -> tuple[int, int]:
    if isinstance(metadata, Mapping):
        return int(metadata.get("num_rows") or 0), int(metadata.get("size_bytes") or 0)
    if isinstance(metadata, (tuple, list)) and len(metadata) >= 2:
        return int(metadata[0] or 0), int(metadata[1] or 0)
    rows = getattr(metadata, "num_rows", None)
    size = getattr(metadata, "size_bytes", None)
    if rows is not None or size is not None:
        return int(rows or 0), int(size or 0)
    return 0, 0


def _looks_like_object_ref(value: Any) -> bool:
    value_type = type(value)
    return value_type.__name__ == "ObjectRef" or value_type.__module__.startswith("ray.")


def describe_result_payload(value: Any) -> dict[str, Any]:
    target = value
    wrapped = False
    if isinstance(value, Mapping) and "result" in value:
        target = value.get("result")
        wrapped = True

    summary: dict[str, Any] = {
        "result_type": type(target).__name__,
        "result_wrapped": wrapped,
    }
    if isinstance(target, (tuple, list)):
        parts: Sequence[Any] = target[0] if len(target) >= 1 and isinstance(target[0], Sequence) else []
        metas: Sequence[Any] = target[1] if len(target) >= 2 and isinstance(target[1], Sequence) else []
        rows = 0
        size_bytes = 0
        for metadata in metas:
            metadata_rows, metadata_size = _metadata_rows_and_bytes(metadata)
            rows += metadata_rows
            size_bytes += metadata_size
        summary.update(
            {
                "result_partition_count": len(parts),
                "result_partition_rows": rows,
                "result_partition_size_bytes": size_bytes,
                "result_object_ref_count": sum(1 for part in parts if _looks_like_object_ref(part)),
                "result_meta_count": len(metas),
            }
        )
    return summary
