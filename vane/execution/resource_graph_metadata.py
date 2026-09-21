# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from vane.execution.resource_graph import _strict_fields


class ResourceGraphMetadataProvider(Protocol):
    """A backend plan adapter exporting the common native graph schema."""

    def collect_resource_graph_metadata(self, conn: Any = None) -> dict[str, Any]: ...


_TOP_LEVEL_FIELDS = ("query_id", "nodes", "terminal_node_ids")
_NODE_FIELDS = (
    "node_id",
    "node_name",
    "input_node_ids",
    "is_sink",
    "is_materialization_barrier",
    "materialized_input_node_ids",
    "num_partitions",
    "udf_payload",
)


def validate_udf_node_ids(metadata: Mapping[str, Any], binding_ids: Any) -> dict[str, str]:
    """Match pipeline IDs to physical UDF IDs; their traversal orders differ."""
    if not isinstance(binding_ids, Mapping):
        raise TypeError("udf_node_ids must be a mapping")
    result = {str(key).strip(): str(value).strip() for key, value in binding_ids.items()}
    udf_nodes = {str(node["node_id"]).strip() for node in metadata["nodes"] if node["udf_payload"] is not None}
    if set(result) != udf_nodes:
        raise ValueError("udf_node_ids must cover every UDF pipeline node exactly once")
    if any(not value for value in result.values()) or len(set(result.values())) != len(result):
        raise ValueError("udf_node_ids must contain distinct non-empty physical UDF IDs")
    return result


def _node_sort_key(node_id: str) -> tuple[int, int | str]:
    value = str(node_id)
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def native_fragment_unit_id_for_node(query_id: str, node_id: str | int) -> str:
    query = str(query_id).strip()
    node = str(node_id).strip()
    if not query or not node:
        raise ValueError("query_id and node_id must be non-empty")
    return f"resource:{query}:fragment:node:{node}"


def udf_unit_id_for_node(query_id: str, node_id: str | int) -> str:
    query = str(query_id).strip()
    node = str(node_id).strip()
    if not query or not node:
        raise ValueError("query_id and node_id must be non-empty")
    return f"resource:{query}:udf:node:{node}"


def materialization_barrier_id_for_node(query_id: str, node_id: str | int) -> str:
    query = str(query_id).strip()
    node = str(node_id).strip()
    if not query or not node:
        raise ValueError("query_id and node_id must be non-empty")
    return f"barrier:{query}:node:{node}"


def native_fragment_unit_id_for_fragment(query_id: str, fragment_id: str) -> str:
    query = str(query_id).strip()
    fragment = str(fragment_id).strip()
    prefix = f"{query}:node:"
    if not fragment.startswith(prefix):
        if fragment.endswith(":node:") or ":node:" not in fragment:
            raise ValueError(f"invalid native fragment_id: {fragment}")
        raise ValueError(f"fragment {fragment!r} does not belong to query {query!r}")
    node_id = fragment[len(prefix) :]
    if not node_id or ":" in node_id:
        raise ValueError(f"invalid native fragment_id: {fragment}")
    return native_fragment_unit_id_for_node(query, node_id)


def _normalize_metadata(metadata: Mapping[str, Any]) -> tuple[str, dict[str, dict[str, Any]], tuple[str, ...]]:
    payload = dict(metadata)
    if "udf_node_ids" in payload:
        binding_ids = payload.pop("udf_node_ids")
        validate_udf_node_ids(metadata, binding_ids)
    _strict_fields(payload, _TOP_LEVEL_FIELDS, "resource unit metadata")
    query_id = str(payload["query_id"]).strip()
    if not query_id:
        raise ValueError("resource unit metadata query_id must be non-empty")
    nodes: dict[str, dict[str, Any]] = {}
    for raw_node in payload["nodes"]:
        node = dict(raw_node)
        _strict_fields(node, _NODE_FIELDS, "resource unit node")
        node_id = str(node["node_id"]).strip()
        if not node_id:
            raise ValueError("resource unit node_id must be non-empty")
        if node_id in nodes:
            raise ValueError(f"duplicate resource unit node_id: {node_id}")
        node["node_id"] = node_id
        node["node_name"] = str(node["node_name"]).strip()
        if not node["node_name"]:
            raise ValueError(f"resource unit node {node_id} node_name must be non-empty")
        node["input_node_ids"] = tuple(str(item).strip() for item in node["input_node_ids"])
        node["num_partitions"] = _positive_int(node["num_partitions"], "num_partitions")
        node["is_sink"] = bool(node["is_sink"])
        node["is_materialization_barrier"] = bool(node["is_materialization_barrier"])
        node["materialized_input_node_ids"] = tuple(str(item).strip() for item in node["materialized_input_node_ids"])
        if len(set(node["materialized_input_node_ids"])) != len(node["materialized_input_node_ids"]):
            raise ValueError(f"resource unit node {node_id} has duplicate materialized input node ids")
        if node["is_materialization_barrier"] != bool(node["materialized_input_node_ids"]):
            raise ValueError(
                f"resource unit node {node_id} must declare materialized inputs "
                "if and only if it is a materialization barrier"
            )
        if node["udf_payload"] is not None and not isinstance(node["udf_payload"], Mapping):
            raise TypeError(f"resource unit node {node_id} udf_payload must be a mapping or None")
        node["udf_payload"] = None if node["udf_payload"] is None else dict(node["udf_payload"])
        nodes[node_id] = node

    for node_id, node in nodes.items():
        for input_node_id in node["input_node_ids"]:
            if input_node_id not in nodes:
                raise ValueError(f"resource unit node {node_id} references missing input node {input_node_id}")
        for input_node_id in node["materialized_input_node_ids"]:
            if input_node_id not in node["input_node_ids"]:
                raise ValueError(
                    f"resource unit node {node_id} materialized input {input_node_id} is not a direct input"
                )
    terminal_node_ids = tuple(str(item).strip() for item in payload["terminal_node_ids"])
    if not terminal_node_ids:
        raise ValueError("resource unit metadata must contain terminal_node_ids")
    for terminal in terminal_node_ids:
        if terminal not in nodes:
            raise ValueError(f"terminal node is not registered: {terminal}")
    return query_id, nodes, tuple(sorted(set(terminal_node_ids), key=_node_sort_key))
