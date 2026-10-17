# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import heapq
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Protocol, TypeVar


def _strict_fields(payload: Mapping[str, Any], expected: tuple[str, ...], type_name: str) -> None:
    actual = set(payload)
    expected_set = set(expected)
    unknown = sorted(actual - expected_set)
    missing = sorted(expected_set - actual)
    if unknown:
        raise ValueError(f"{type_name} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{type_name} is missing required fields: {', '.join(missing)}")


class ResourceUnit(Protocol):
    """Identity and dependencies shared by backend-specific resource units."""

    @property
    def query_id(self) -> str: ...

    @property
    def resource_unit_id(self) -> str: ...

    @property
    def physical_node_id(self) -> str: ...

    @property
    def unit_kind(self) -> str: ...

    @property
    def backend(self) -> str: ...

    @property
    def input_unit_ids(self) -> tuple[str, ...]: ...

    def to_dict(self) -> dict[str, Any]: ...


UnitT = TypeVar("UnitT", bound=ResourceUnit)


@dataclass(frozen=True)
class MaterializationBarrierSpec:
    """A materialization boundary inside one physical materializer node.

    Barriers are execution events, not resource-owning operators.  The
    materializer remains a normal native resource unit so its managed object
    flow can be accounted without inventing a separate Stage abstraction.
    Before completion only ``materialized_input_unit_ids`` feed that node;
    after completion the same node may emit final work from its remaining
    inputs before downstream execution continues.
    """

    query_id: str
    barrier_id: str
    physical_node_id: str
    materializer_unit_id: str
    materialized_input_unit_ids: tuple[str, ...]

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "query_id",
        "barrier_id",
        "physical_node_id",
        "materializer_unit_id",
        "materialized_input_unit_ids",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", str(self.query_id).strip())
        object.__setattr__(self, "barrier_id", str(self.barrier_id).strip())
        object.__setattr__(self, "physical_node_id", str(self.physical_node_id).strip())
        object.__setattr__(self, "materializer_unit_id", str(self.materializer_unit_id).strip())
        object.__setattr__(
            self,
            "materialized_input_unit_ids",
            tuple(str(item).strip() for item in self.materialized_input_unit_ids),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "barrier_id": self.barrier_id,
            "physical_node_id": self.physical_node_id,
            "materializer_unit_id": self.materializer_unit_id,
            "materialized_input_unit_ids": list(self.materialized_input_unit_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MaterializationBarrierSpec:
        values = dict(payload)
        _strict_fields(values, cls._FIELDS, cls.__name__)
        return cls(
            query_id=str(values["query_id"]),
            barrier_id=str(values["barrier_id"]),
            physical_node_id=str(values["physical_node_id"]),
            materializer_unit_id=str(values["materializer_unit_id"]),
            materialized_input_unit_ids=tuple(str(item) for item in values["materialized_input_unit_ids"]),
        )


@dataclass(frozen=True)
class ResourceGraph(Generic[UnitT]):
    """Dependency graph for query resource accounting and admission."""

    query_id: str
    plan_digest: str
    units: tuple[UnitT, ...]
    terminal_unit_ids: tuple[str, ...]
    materialization_barriers: tuple[MaterializationBarrierSpec, ...] = ()

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "query_id",
        "plan_digest",
        "units",
        "materialization_barriers",
        "terminal_unit_ids",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", str(self.query_id).strip())
        object.__setattr__(self, "plan_digest", str(self.plan_digest).strip())
        object.__setattr__(self, "units", tuple(self.units))
        object.__setattr__(self, "materialization_barriers", tuple(self.materialization_barriers))
        object.__setattr__(self, "terminal_unit_ids", tuple(str(item).strip() for item in self.terminal_unit_ids))
        self._validate()

    def _validate(self) -> None:
        if not self.query_id:
            raise ValueError("query_id must be non-empty")
        if not self.plan_digest:
            raise ValueError("plan_digest must be non-empty")
        if not self.units:
            raise ValueError("query resource graph must contain at least one unit")
        if not self.terminal_unit_ids:
            raise ValueError("query resource graph must contain at least one terminal unit")

        by_id: dict[str, UnitT] = {}
        physical_nodes: dict[str, str] = {}
        for unit in self.units:
            self._validate_unit(unit)
            if unit.resource_unit_id in by_id:
                raise ValueError(f"duplicate resource_unit_id: {unit.resource_unit_id}")
            if unit.physical_node_id in physical_nodes:
                raise ValueError(
                    "duplicate physical_node_id: "
                    f"{unit.physical_node_id} used by {physical_nodes[unit.physical_node_id]} and {unit.resource_unit_id}"
                )
            by_id[unit.resource_unit_id] = unit
            physical_nodes[unit.physical_node_id] = unit.resource_unit_id

        if len(set(self.terminal_unit_ids)) != len(self.terminal_unit_ids):
            raise ValueError("terminal_unit_ids must be unique")
        for terminal in self.terminal_unit_ids:
            if terminal not in by_id:
                raise ValueError(f"terminal unit is not registered: {terminal}")

        downstream: dict[str, set[str]] = {resource_unit_id: set() for resource_unit_id in by_id}
        for unit in self.units:
            if len(set(unit.input_unit_ids)) != len(unit.input_unit_ids):
                raise ValueError(f"unit {unit.resource_unit_id} has duplicate input_unit_ids")
            for input_unit_id in unit.input_unit_ids:
                if input_unit_id not in by_id:
                    raise ValueError(f"unit {unit.resource_unit_id} references missing input unit {input_unit_id}")
                if input_unit_id == unit.resource_unit_id:
                    raise ValueError(f"query resource graph contains a cycle at unit {unit.resource_unit_id}")
                downstream[input_unit_id].add(unit.resource_unit_id)

        ordered = self._topological_order(by_id, downstream)
        if len(ordered) != len(by_id):
            raise ValueError("query resource graph contains a cycle")

        barrier_ids: set[str] = set()
        barrier_nodes: set[str] = set()
        barrier_units: set[str] = set()
        for barrier in self.materialization_barriers:
            if str(barrier.query_id).strip() != self.query_id:
                raise ValueError(
                    f"barrier {barrier.barrier_id or '<empty>'} query_id {barrier.query_id!r} "
                    f"does not match {self.query_id!r}"
                )
            barrier_id = str(barrier.barrier_id).strip()
            physical_node_id = str(barrier.physical_node_id).strip()
            materializer_unit_id = str(barrier.materializer_unit_id).strip()
            if barrier_id != f"barrier:{self.query_id}:node:{physical_node_id}":
                raise ValueError(f"invalid materialization barrier identity: {barrier_id!r}")
            if not physical_node_id:
                raise ValueError(f"barrier {barrier_id} physical_node_id must be non-empty")
            if materializer_unit_id not in by_id:
                raise ValueError(f"barrier {barrier_id} references missing materializer unit {materializer_unit_id}")
            materializer = by_id[materializer_unit_id]
            if materializer.unit_kind != "native_fragment":
                raise ValueError(f"barrier {barrier_id} materializer must be a native fragment unit")
            expected_materializer_physical_node_id = f"node:{physical_node_id}:native-fragment"
            if materializer.physical_node_id != expected_materializer_physical_node_id:
                raise ValueError(
                    f"barrier {barrier_id} physical node {physical_node_id!r} does not match "
                    f"materializer {materializer_unit_id} physical node {materializer.physical_node_id!r}"
                )
            materialized_input_unit_ids = tuple(
                str(resource_unit_id).strip() for resource_unit_id in barrier.materialized_input_unit_ids
            )
            if not materialized_input_unit_ids:
                raise ValueError(f"barrier {barrier_id} must materialize at least one input unit")
            if len(set(materialized_input_unit_ids)) != len(materialized_input_unit_ids):
                raise ValueError(f"barrier {barrier_id} has duplicate materialized input units")
            materializer_inputs = set(materializer.input_unit_ids)
            for input_unit_id in materialized_input_unit_ids:
                if input_unit_id not in materializer_inputs:
                    raise ValueError(
                        f"barrier {barrier_id} materialized input {input_unit_id} "
                        f"is not a direct input of {materializer_unit_id}"
                    )
            if barrier_id in barrier_ids:
                raise ValueError(f"duplicate materialization barrier_id: {barrier_id}")
            if physical_node_id in barrier_nodes:
                raise ValueError(f"duplicate materialization barrier physical_node_id: {physical_node_id}")
            if materializer_unit_id in barrier_units:
                raise ValueError(f"duplicate materialization barrier unit: {materializer_unit_id}")
            barrier_ids.add(barrier_id)
            barrier_nodes.add(physical_node_id)
            barrier_units.add(materializer_unit_id)

        for terminal in self.terminal_unit_ids:
            if downstream[terminal]:
                raise ValueError(
                    f"terminal unit {terminal} has downstream units: {', '.join(sorted(downstream[terminal]))}"
                )

        reaches_terminal = set(self.terminal_unit_ids)
        for resource_unit_id in reversed(ordered):
            if any(child in reaches_terminal for child in downstream[resource_unit_id]):
                reaches_terminal.add(resource_unit_id)
        missing_terminal_path = sorted(set(by_id) - reaches_terminal)
        if missing_terminal_path:
            raise ValueError(f"unit {missing_terminal_path[0]} does not reach a terminal unit")

    def _validate_unit(self, unit: UnitT) -> None:
        if str(unit.query_id).strip() != self.query_id:
            raise ValueError(
                f"unit {unit.resource_unit_id or '<empty>'} query_id {unit.query_id!r} does not match {self.query_id!r}"
            )
        if not str(unit.resource_unit_id).strip():
            raise ValueError("resource_unit_id must be non-empty")
        if not str(unit.resource_unit_id).startswith("resource:"):
            raise ValueError(f"resource_unit_id must use stable 'resource:' identity: {unit.resource_unit_id}")
        if not str(unit.physical_node_id).strip():
            raise ValueError(f"unit {unit.resource_unit_id} physical_node_id must be non-empty")
        unit_kind = str(unit.unit_kind).strip()
        if not unit_kind:
            raise ValueError(f"unit {unit.resource_unit_id} unit_kind must be non-empty")
        backend = str(unit.backend).strip()
        if not backend:
            raise ValueError(f"unit {unit.resource_unit_id} backend must be non-empty")

    @staticmethod
    def _topological_order(
        by_id: Mapping[str, UnitT],
        downstream: Mapping[str, set[str]],
    ) -> tuple[str, ...]:
        indegree = {resource_unit_id: len(unit.input_unit_ids) for resource_unit_id, unit in by_id.items()}
        ready = [resource_unit_id for resource_unit_id, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        ordered: list[str] = []
        while ready:
            resource_unit_id = heapq.heappop(ready)
            ordered.append(resource_unit_id)
            for child in sorted(downstream[resource_unit_id]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
        return tuple(ordered)

    def unit_by_id(self, resource_unit_id: str) -> UnitT:
        key = str(resource_unit_id)
        for unit in self.units:
            if unit.resource_unit_id == key:
                return unit
        raise KeyError(f"unknown resource_unit_id {key!r}")

    def unit_id_for_physical_node(self, physical_node_id: str) -> str:
        key = str(physical_node_id)
        for unit in self.units:
            if unit.physical_node_id == key:
                return unit.resource_unit_id
        raise KeyError(f"unknown physical_node_id {key!r}")

    def topological_unit_ids(self) -> tuple[str, ...]:
        by_id = {unit.resource_unit_id: unit for unit in self.units}
        downstream: dict[str, set[str]] = {resource_unit_id: set() for resource_unit_id in by_id}
        for unit in self.units:
            for parent in unit.input_unit_ids:
                downstream[parent].add(unit.resource_unit_id)
        return self._topological_order(by_id, downstream)

    def reverse_topological_unit_ids(self) -> tuple[str, ...]:
        return tuple(reversed(self.topological_unit_ids()))

    def barrier_for_physical_node(self, physical_node_id: str) -> MaterializationBarrierSpec:
        key = str(physical_node_id)
        for barrier in self.materialization_barriers:
            if barrier.physical_node_id == key:
                return barrier
        raise KeyError(f"unknown materialization barrier physical_node_id {key!r}")

    def ordered_materialization_barriers(self) -> tuple[MaterializationBarrierSpec, ...]:
        rank = {resource_unit_id: index for index, resource_unit_id in enumerate(self.topological_unit_ids())}
        return tuple(
            sorted(
                self.materialization_barriers,
                key=lambda barrier: (rank[barrier.materializer_unit_id], barrier.barrier_id),
            )
        )

    def eligible_resource_unit_ids(self, completed_barrier_ids: set[str] | frozenset[str]) -> tuple[str, ...]:
        """Return units not hidden behind an unfinished true barrier.

        The returned set is the current execution phase, not the set of tasks
        that happen to be running.  Parallel branches up to their respective
        first unfinished barriers remain eligible together.  Once a barrier
        completes, reverse traversal stops at that boundary so the preceding
        phase is retired instead of being reserved again.
        """

        completed = {str(barrier_id) for barrier_id in completed_barrier_ids}
        ordered = self.topological_unit_ids()
        by_id = {unit.resource_unit_id: unit for unit in self.units}
        barrier_by_materializer = {barrier.materializer_unit_id: barrier for barrier in self.materialization_barriers}
        pending = tuple(
            barrier for barrier in self.ordered_materialization_barriers() if barrier.barrier_id not in completed
        )
        pending_materializer_unit_ids = {barrier.materializer_unit_id for barrier in pending}

        direct_downstream: dict[str, set[str]] = {resource_unit_id: set() for resource_unit_id in ordered}
        for unit in self.units:
            for input_unit_id in unit.input_unit_ids:
                direct_downstream[input_unit_id].add(unit.resource_unit_id)
        blocked: set[str] = set()
        todo = [
            downstream_unit_id
            for barrier in pending
            for downstream_unit_id in direct_downstream[barrier.materializer_unit_id]
        ]
        while todo:
            resource_unit_id = todo.pop()
            if resource_unit_id in blocked:
                continue
            blocked.add(resource_unit_id)
            todo.extend(direct_downstream[resource_unit_id])

        # Every unfinished barrier whose materialized side is reachable is a
        # target for the current phase.  The materializer itself can be
        # structurally downstream of another barrier through a deferred input
        # (for example, a broadcast build whose probe side contains a sort),
        # while its independent materialized side is already executable.
        # Treating the whole materializer as blocked would serialize those two
        # materializations and would not match the task streams spawned by the
        # distributed executor.
        targets = {
            barrier.materializer_unit_id
            for barrier in pending
            if not any(
                input_unit_id in blocked or input_unit_id in pending_materializer_unit_ids
                for input_unit_id in barrier.materialized_input_unit_ids
            )
        }
        # A terminal is also a target when no unfinished barrier withholds it.
        targets.update(
            terminal_unit_id for terminal_unit_id in self.terminal_unit_ids if terminal_unit_id not in blocked
        )
        # A pending barrier on one branch can also block a downstream join fed
        # by a branch whose own barrier has already completed.  Keep the
        # unblocked side of every blocked edge eligible so that branch can
        # stream up to the join's bounded input instead of being retired until
        # the other branch catches up.
        for blocked_unit_id in blocked:
            unit = by_id[blocked_unit_id]
            barrier = barrier_by_materializer.get(blocked_unit_id)
            if barrier is not None and barrier.barrier_id not in completed:
                # A pending barrier's pre-materialization work is represented
                # by targeting the materializer above.  Its deferred inputs do
                # not execute until the barrier completes.
                continue
            retired_inputs = set() if barrier is None else set(barrier.materialized_input_unit_ids)
            targets.update(
                input_unit_id
                for input_unit_id in unit.input_unit_ids
                if input_unit_id not in blocked and input_unit_id not in retired_inputs
            )

        eligible: set[str] = set()
        todo = list(targets)
        while todo:
            resource_unit_id = todo.pop()
            if resource_unit_id in eligible:
                continue
            eligible.add(resource_unit_id)
            unit = by_id[resource_unit_id]
            barrier = barrier_by_materializer.get(resource_unit_id)
            if barrier is None:
                todo.extend(unit.input_unit_ids)
            elif barrier.barrier_id in completed:
                materialized_inputs = set(barrier.materialized_input_unit_ids)
                todo.extend(
                    input_unit_id for input_unit_id in unit.input_unit_ids if input_unit_id not in materialized_inputs
                )
            else:
                todo.extend(barrier.materialized_input_unit_ids)
        return tuple(resource_unit_id for resource_unit_id in ordered if resource_unit_id in eligible)

    def frontier_materialization_barriers(
        self,
        completed_barrier_ids: set[str] | frozenset[str],
    ) -> tuple[MaterializationBarrierSpec, ...]:
        """Return all unfinished barriers on the current parallel frontier."""

        completed = {str(barrier_id) for barrier_id in completed_barrier_ids}
        eligible = set(self.eligible_resource_unit_ids(completed))
        return tuple(
            barrier
            for barrier in self.ordered_materialization_barriers()
            if barrier.barrier_id not in completed and barrier.materializer_unit_id in eligible
        )

    def task_identity(self, resource_unit_id: str, *, partition_id: int | str, attempt_id: int | str) -> str:
        unit = self.unit_by_id(resource_unit_id)
        partition = str(partition_id).strip()
        attempt = str(attempt_id).strip()
        if not partition:
            raise ValueError("partition_id must be non-empty")
        if not attempt:
            raise ValueError("attempt_id must be non-empty")
        return f"task:{unit.resource_unit_id}:partition:{partition}:attempt:{attempt}"

    def normalized_digest(self) -> str:
        payload = self.to_dict()
        payload["plan_digest"] = ""
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "plan_digest": self.plan_digest,
            "units": [unit.to_dict() for unit in self.units],
            "materialization_barriers": [barrier.to_dict() for barrier in self.materialization_barriers],
            "terminal_unit_ids": list(self.terminal_unit_ids),
        }
