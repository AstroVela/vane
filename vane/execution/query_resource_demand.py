# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

from vane.execution.cluster_resource_policy import NodeCapacity, QueryDemand
from vane.execution.query_resource_spec import QueryResourceGraph, ResourceUnitSpec
from vane.execution.resources import ResourceVector


def _resource_dimension(resources: ResourceVector, field_name: str) -> int | float:
    value = getattr(resources, field_name)
    if not isinstance(value, (int, float)):
        raise TypeError(f"resource field {field_name!r} must be numeric, got {value!r}")
    return value


def _task_scheduling_request(unit: ResourceUnitSpec) -> ResourceVector:
    """Return the concrete Ray Core request for one task invocation.

    Retained inputs and generator output windows are spillable pipeline data.
    QRM accounts them dynamically instead of copying them into the process
    request or a query-level placement model.
    """
    return ResourceVector(
        cpu=unit.per_task.cpu,
        gpu=unit.per_task.gpu,
        heap_bytes=unit.per_task.heap_bytes,
    )


def _sum_node_capacities(node_capacities: Sequence[NodeCapacity]) -> ResourceVector:
    total = ResourceVector()
    for node in node_capacities:
        total = total + node.resources
    return total


def build_query_demand(
    graph: QueryResourceGraph,
    node_capacities: Sequence[NodeCapacity],
    *,
    eligible_unit_ids: tuple[str, ...] | None = None,
    weight: float = 1.0,
    priority: int = 0,
) -> QueryDemand:
    nodes = tuple(node_capacities)
    node_ids = [node.node_id for node in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("node_capacities contains duplicate node IDs")
    cluster_capacity = _sum_node_capacities(nodes)
    eligible = set(graph.eligible_resource_unit_ids(set()) if eligible_unit_ids is None else eligible_unit_ids)
    unknown_eligible = sorted(eligible - {unit.resource_unit_id for unit in graph.units})
    if unknown_eligible:
        raise ValueError(f"query demand references unknown eligible unit: {unknown_eligible[0]}")
    tasks: list[ResourceVector] = []
    actor_processes: list[ResourceVector] = []
    for unit in graph.units:
        if unit.resource_unit_id not in eligible:
            continue
        commitment = _task_scheduling_request(unit)
        if unit.execution_kind == "task":
            tasks.append(commitment)
        elif unit.execution_kind == "actor":
            actor_processes.extend(unit.resident_per_actor for _actor_index in range(unit.actor_pool_size))
    actor_maximum = ResourceVector()
    for actor_process in actor_processes:
        actor_maximum = actor_maximum + actor_process

    def elastic_target(field_name: str) -> int | float:
        task_uses_dimension = any(_resource_dimension(task, field_name) > 0 for task in tasks)
        if task_uses_dimension:
            elastic = _resource_dimension(cluster_capacity, field_name)
        else:
            elastic = min(
                _resource_dimension(actor_maximum, field_name),
                _resource_dimension(cluster_capacity, field_name),
            )
        return elastic

    desired = ResourceVector(
        # CPU/GPU/declared heap are aggregate soft targets only. Concrete UDF
        # calls carry their real resource shape to Ray Core, including when a
        # current cluster snapshot cannot fit that shape yet.
        cpu=elastic_target("cpu"),
        gpu=elastic_target("gpu"),
        heap_bytes=int(elastic_target("heap_bytes")),
        object_store_bytes=cluster_capacity.object_store_bytes,
    )
    return QueryDemand(
        query_id=graph.query_id,
        desired=desired,
        weight=weight,
        priority=priority,
    )
