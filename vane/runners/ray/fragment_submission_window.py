# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from vane.runners.ray.fragment_registry import (
    _FTE_CLOSING_QUERIES,
    _FTE_FRAGMENT_EXECUTIONS,
    _FTE_PARTITION_TASK_LEASES,
    _FTE_REGISTRY_LOCK,
    _FTE_RESOURCE_UNIT_SUBMISSION_BLOCKS,
    _FTE_RESOURCE_UNIT_SUBMISSION_PROBES,
)
from vane.runners.ray.fragment_worker_context import resource_identity_from_context

PartitionKey = tuple[str, str, int]
ResourceUnitKey = tuple[str, str]


def _partition_key(
    query_id: str,
    fragment_id: str,
    partition_id: int,
) -> PartitionKey:
    return (str(query_id), str(fragment_id), int(partition_id))


def _resource_unit_key_for_partition(key: PartitionKey) -> ResourceUnitKey:
    with _FTE_REGISTRY_LOCK:
        fragment_execution = _FTE_FRAGMENT_EXECUTIONS.get((key[0], key[1]))
    if fragment_execution is None:
        raise KeyError(f"FTE fragment execution {key[0]}/{key[1]} is not registered")
    return resource_identity_from_context(fragment_execution.context)


def admit_fte_partition_submission(
    query_id: str,
    fragment_id: str,
    partition_id: int,
) -> bool:
    """Claim the single edge-triggered admission probe for a resource unit.

    Active leases bypass the probe.  Unleased descriptors are promoted one at
    a time.  After QRM denies one probe, all other descriptors remain passive
    until QRM publishes a new admission epoch.
    """
    from vane.runners.ray.query_resource_runtime import get_query_resource_manager

    key = _partition_key(query_id, fragment_id, partition_id)
    resource_unit_key = _resource_unit_key_for_partition(key)
    manager = get_query_resource_manager(resource_unit_key[0])
    admission_epoch = manager.admission_epoch()
    with _FTE_REGISTRY_LOCK:
        if key[0] in _FTE_CLOSING_QUERIES:
            return False
        if key in _FTE_PARTITION_TASK_LEASES:
            return True
        owner = _FTE_RESOURCE_UNIT_SUBMISSION_PROBES.get(resource_unit_key)
        if owner is not None:
            return owner == key
        blocked = _FTE_RESOURCE_UNIT_SUBMISSION_BLOCKS.get(resource_unit_key)
        if blocked is not None and int(blocked[0]) == admission_epoch:
            return False
        # The epoch advanced because capacity, allocation, downstream demand,
        # or arbitration state changed.  The old denial is no longer valid.
        _FTE_RESOURCE_UNIT_SUBMISSION_BLOCKS.pop(resource_unit_key, None)
        _FTE_RESOURCE_UNIT_SUBMISSION_PROBES[resource_unit_key] = key
        return True


def resolve_fte_partition_submission(
    query_id: str,
    fragment_id: str,
    partition_id: int,
    *,
    granted: bool,
    blocked_reason: str = "",
    fatal: bool = False,
    admission_epoch: int | None = None,
) -> None:
    """Resolve a resource-unit probe after its atomic QRM admission attempt."""
    from vane.runners.ray.query_resource_runtime import get_query_resource_manager

    key = _partition_key(query_id, fragment_id, partition_id)
    try:
        resource_unit_key = _resource_unit_key_for_partition(key)
    except KeyError:
        return
    if admission_epoch is None:
        try:
            resolved_epoch = get_query_resource_manager(resource_unit_key[0]).admission_epoch()
        except KeyError:
            resolved_epoch = -1
    else:
        resolved_epoch = int(admission_epoch)
    with _FTE_REGISTRY_LOCK:
        if _FTE_RESOURCE_UNIT_SUBMISSION_PROBES.get(resource_unit_key) == key:
            _FTE_RESOURCE_UNIT_SUBMISSION_PROBES.pop(resource_unit_key, None)
        if granted or fatal or resolved_epoch < 0:
            _FTE_RESOURCE_UNIT_SUBMISSION_BLOCKS.pop(resource_unit_key, None)
        else:
            _FTE_RESOURCE_UNIT_SUBMISSION_BLOCKS[resource_unit_key] = (
                resolved_epoch,
                str(blocked_reason or "task_not_admissible"),
                key,
            )


def release_fte_partition_submission(
    query_id: str,
    fragment_id: str,
    partition_id: int,
    *,
    allow_next: bool = True,
) -> bool:
    """Release an abandoned probe without leaving the resource unit permanently shut."""
    key = _partition_key(query_id, fragment_id, partition_id)
    try:
        resource_unit_key = _resource_unit_key_for_partition(key)
    except KeyError:
        return False
    with _FTE_REGISTRY_LOCK:
        released = False
        if _FTE_RESOURCE_UNIT_SUBMISSION_PROBES.get(resource_unit_key) == key:
            _FTE_RESOURCE_UNIT_SUBMISSION_PROBES.pop(resource_unit_key, None)
            released = True
        blocked = _FTE_RESOURCE_UNIT_SUBMISSION_BLOCKS.get(resource_unit_key)
        if allow_next and blocked is not None and blocked[2] == key:
            _FTE_RESOURCE_UNIT_SUBMISSION_BLOCKS.pop(resource_unit_key, None)
            released = True
        return released


def release_fte_query_submissions(query_id: str) -> int:
    query_key = str(query_id)
    with _FTE_REGISTRY_LOCK:
        resource_unit_keys = {
            resource_identity_from_context(fragment_execution.context)
            for (execution_query_id, _fragment_id), fragment_execution in _FTE_FRAGMENT_EXECUTIONS.items()
            if execution_query_id == query_key
        }
        owned_resource_unit_keys = [
            resource_unit_key
            for resource_unit_key, partition_key in _FTE_RESOURCE_UNIT_SUBMISSION_PROBES.items()
            if partition_key[0] == query_key
        ]
        for resource_unit_key in owned_resource_unit_keys:
            _FTE_RESOURCE_UNIT_SUBMISSION_PROBES.pop(resource_unit_key, None)
            _FTE_RESOURCE_UNIT_SUBMISSION_BLOCKS.pop(resource_unit_key, None)
        # A blocked resource unit normally has a probe owner from this execution query
        # at the time it is recorded.  Drop orphaned blocks for resource-owned
        # queries too, which covers teardown after partial registration.
        for resource_unit_key in list(_FTE_RESOURCE_UNIT_SUBMISSION_BLOCKS):
            if resource_unit_key[0] == query_key or resource_unit_key in resource_unit_keys:
                _FTE_RESOURCE_UNIT_SUBMISSION_BLOCKS.pop(resource_unit_key, None)
        return len(owned_resource_unit_keys)


def fte_submission_window_snapshot() -> dict[str, Any]:
    with _FTE_REGISTRY_LOCK:
        return {
            "probes": {
                f"{resource_query_id}/{resource_unit_id}": {
                    "query_id": partition_key[0],
                    "fragment_id": partition_key[1],
                    "partition_id": partition_key[2],
                }
                for (resource_query_id, resource_unit_id), partition_key in sorted(
                    _FTE_RESOURCE_UNIT_SUBMISSION_PROBES.items()
                )
            },
            "blocks": {
                f"{resource_query_id}/{resource_unit_id}": {
                    "admission_epoch": int(blocked[0]),
                    "blocked_reason": str(blocked[1]),
                }
                for (resource_query_id, resource_unit_id), blocked in sorted(
                    _FTE_RESOURCE_UNIT_SUBMISSION_BLOCKS.items()
                )
            },
        }


__all__ = [
    "admit_fte_partition_submission",
    "fte_submission_window_snapshot",
    "release_fte_partition_submission",
    "release_fte_query_submissions",
    "resolve_fte_partition_submission",
]
