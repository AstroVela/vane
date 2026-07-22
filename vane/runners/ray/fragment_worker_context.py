# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from collections.abc import Mapping


def query_id_from_context(context: dict[str, str] | None) -> str:
    if not context:
        return "query"
    query_id = str(context.get("query_id", "")).strip()
    if query_id:
        return query_id
    return "query"


def resource_identity_from_context(
    context: Mapping[str, object] | None,
) -> tuple[str, str]:
    if not context:
        raise ValueError("FTE task context is missing resource identity")
    resource_query_id = str(context.get("resource_query_id") or "").strip()
    resource_stage_id = str(context.get("resource_stage_id") or "").strip()
    if not resource_query_id or not resource_stage_id:
        raise ValueError("FTE task context requires resource_query_id and resource_stage_id")
    expected_prefix = f"stage:{resource_query_id}:node:"
    if (
        not resource_stage_id.startswith(expected_prefix)
        or not resource_stage_id.endswith(":fte")
        or not resource_stage_id[len(expected_prefix) : -len(":fte")]
    ):
        raise ValueError(
            "FTE task resource_stage_id does not belong to resource_query_id: "
            f"query={resource_query_id!r} stage={resource_stage_id!r}"
        )
    return resource_query_id, resource_stage_id


def node_id_from_context(context: dict[str, str] | None) -> str | None:
    if not context:
        return None
    node_id = str(context.get("node_id", "")).strip()
    if not node_id:
        return None
    return node_id


def node_name_from_context(context: dict[str, str] | None) -> str | None:
    if not context:
        return None
    node_name = str(context.get("node_name", "")).strip()
    if not node_name:
        return None
    return node_name


def stable_fragment_suffix(value: str | bytes) -> str:
    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def explicit_fragment_id_from_context(context: dict[str, str] | None) -> str | None:
    if not context:
        return None
    fragment_id = str(context.get("fragment_id", "")).strip()
    if not fragment_id:
        return None
    return fragment_id


def fragment_id_for_task(
    context: dict[str, str] | None,
    task_name: str,
) -> tuple[str, str]:
    query_id = query_id_from_context(context)

    explicit_fragment_id = explicit_fragment_id_from_context(context)
    if explicit_fragment_id:
        return query_id, explicit_fragment_id

    node_id = node_id_from_context(context)
    if node_id:
        return query_id, f"{query_id}:node:{node_id}"

    node_name = node_name_from_context(context)
    if node_name:
        return query_id, f"{query_id}:name:{stable_fragment_suffix(node_name)}"

    task_key = task_name.strip() or "task"
    return query_id, f"{query_id}:task:{stable_fragment_suffix(task_key)}"
