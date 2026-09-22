# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Local execution identities on the shared resource graph.

This is structural metadata, not a scheduler: native materialization completion
has not yet been connected to the graph's phase calculation.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

from vane.execution.resource_graph import MaterializationBarrierSpec, ResourceGraph
from vane.execution.resource_graph_metadata import (
    ResourceGraphMetadataProvider,
    _node_sort_key,
    _normalize_metadata,
    materialization_barrier_id_for_node,
    native_fragment_unit_id_for_node,
    udf_unit_id_for_node,
    validate_udf_node_ids,
)


@dataclass(frozen=True)
class LocalResourceGraphAdapter(ResourceGraphMetadataProvider):
    """Read the plan without adding query identity to reusable model payloads."""

    plan: Any

    def collect_resource_graph_metadata(self, conn: Any = None) -> dict[str, Any]:
        return self.plan.collect_resource_graph_metadata(conn=conn, annotate_udfs=False)


@dataclass(frozen=True)
class LocalResourceUnitSpec:
    query_id: str
    resource_unit_id: str
    physical_node_id: str
    unit_kind: str
    backend: str
    input_unit_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "input_unit_ids": list(self.input_unit_ids)}


@dataclass(frozen=True)
class LocalQueryResourceGraph(ResourceGraph[LocalResourceUnitSpec]):
    def _validate_unit(self, unit: LocalResourceUnitSpec) -> None:
        super()._validate_unit(unit)
        kinds = {
            "local_native": "native_fragment",
            "subprocess_task": "subprocess_task_udf",
            "subprocess_actor": "subprocess_actor_pool",
        }
        if kinds.get(unit.backend) != unit.unit_kind:
            raise ValueError(f"invalid local resource unit backend/kind: {unit.backend!r}/{unit.unit_kind!r}")


@dataclass(frozen=True)
class LocalResourceUnitContext:
    """Invocation identity; deliberately separate from model initialization."""

    query_id: str
    resource_unit_id: str
    physical_node_id: str
    backend: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class PreparedLocalResourceGraph:
    """Retain diagnostics for one preparation until its owner is shut down.

    Owns no workers or memory. Releasing this diagnostic scope does not assert
    that other query owners have completed cleanup.
    """

    def __init__(
        self,
        metadata: Mapping[str, Any],
        *,
        release: Callable[[str], None],
    ) -> None:
        self.plan_id = str(metadata["query_id"])
        self.graph = build_local_resource_graph(metadata, query_id=uuid.uuid4().hex)
        self.udf_node_ids = validate_udf_node_ids(metadata, metadata["udf_node_ids"])
        self._release: Callable[[str], None] | None = release
        self._lock = threading.Lock()

    def contexts(self) -> dict[str, LocalResourceUnitContext]:
        result = {}
        for pipeline_node_id, binding_id in self.udf_node_ids.items():
            unit = self.graph.unit_by_id(udf_unit_id_for_node(self.graph.query_id, pipeline_node_id))
            result[binding_id] = LocalResourceUnitContext(
                query_id=unit.query_id,
                resource_unit_id=unit.resource_unit_id,
                physical_node_id=unit.physical_node_id,
                backend=unit.backend,
            )
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "graph": self.graph.to_dict(),
            "udf_node_ids": dict(self.udf_node_ids),
            "phase_tracking": "structural_only",
            "initial_eligible_unit_ids": list(self.graph.eligible_resource_unit_ids(set())),
        }

    def shutdown(self, *, kill: bool = False) -> None:
        with self._lock:
            release, self._release = self._release, None
        if release is not None:
            release(self.graph.query_id)

    def cleanup_pending(self) -> bool:
        return False


def build_local_resource_graph(metadata: Mapping[str, Any], *, query_id: str) -> LocalQueryResourceGraph:
    """Build a per-execution graph without Ray placement or memory policy."""
    _, nodes, terminal_node_ids = _normalize_metadata(metadata)
    output_units = {}
    for node_id, node in nodes.items():
        payload = node["udf_payload"]
        if payload is not None and payload.get("execution_backend") not in {"subprocess_task", "subprocess_actor"}:
            raise ValueError("local resource graphs require local subprocess UDFs")
        output_units[node_id] = (
            udf_unit_id_for_node(query_id, node_id)
            if payload is not None
            else native_fragment_unit_id_for_node(query_id, node_id)
        )
    units = []
    barriers = []
    for node_id in sorted(nodes, key=_node_sort_key):
        node = nodes[node_id]
        native_id = native_fragment_unit_id_for_node(query_id, node_id)
        units.append(
            LocalResourceUnitSpec(
                query_id=query_id,
                resource_unit_id=native_id,
                physical_node_id=f"node:{node_id}:native-fragment",
                unit_kind="native_fragment",
                backend="local_native",
                input_unit_ids=tuple(output_units[parent] for parent in node["input_node_ids"]),
            )
        )
        if node["is_materialization_barrier"]:
            barriers.append(
                MaterializationBarrierSpec(
                    query_id=query_id,
                    barrier_id=materialization_barrier_id_for_node(query_id, node_id),
                    physical_node_id=node_id,
                    materializer_unit_id=native_id,
                    materialized_input_unit_ids=tuple(
                        output_units[parent] for parent in node["materialized_input_node_ids"]
                    ),
                )
            )
        payload = node["udf_payload"]
        if payload is not None:
            backend = str(payload["execution_backend"])
            units.append(
                LocalResourceUnitSpec(
                    query_id=query_id,
                    resource_unit_id=output_units[node_id],
                    physical_node_id=f"node:{node_id}:udf",
                    unit_kind="subprocess_actor_pool" if backend == "subprocess_actor" else "subprocess_task_udf",
                    backend=backend,
                    input_unit_ids=(native_id,),
                )
            )
    graph = LocalQueryResourceGraph(
        query_id=query_id,
        plan_digest="sha256:pending",
        units=tuple(units),
        terminal_unit_ids=tuple(output_units[node_id] for node_id in terminal_node_ids),
        materialization_barriers=tuple(barriers),
    )
    return replace(graph, plan_digest=graph.normalized_digest())
