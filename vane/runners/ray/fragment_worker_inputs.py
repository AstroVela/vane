# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


def extract_task_inputs(
    task: Any,
    context: dict[str, Any],
) -> dict[str, Any]:
    inputs = task.Inputs()
    if not inputs:
        return context

    scan_node_ids: list[str] = []
    exchange_source_node_ids: list[str] = []
    for node_id, entry in inputs.items():
        if not isinstance(entry, dict):
            continue
        kind = entry.get("kind")
        if kind == "scan_task":
            raw_bytes = entry["data"]
            # Pass raw bytes directly; no base64 encoding needed.
            context[f"scan_task:{node_id}"] = raw_bytes
            scan_node_ids.append(str(node_id))
            continue
        if kind == "exchange_source_task":
            raw_bytes = entry["data"]
            context[f"exchange_source_task:{node_id}"] = raw_bytes
            exchange_source_node_ids.append(str(node_id))
            continue
        raise ValueError(f"Unsupported task input kind: {kind!r}")

    if scan_node_ids:
        context["scan_task_nodes"] = ",".join(scan_node_ids)
    if exchange_source_node_ids:
        context["exchange_source_task_nodes"] = ",".join(exchange_source_node_ids)

    return context
