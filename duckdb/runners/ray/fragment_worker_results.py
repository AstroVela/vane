# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from duckdb.runners.ray.fragment_registry import (
    _FTE_ACTIVE_OPERATIONS_BY_QUERY,
    _FTE_CLOSING_QUERIES,
    _FTE_FRAGMENT_EXECUTIONS,
    _FTE_REGISTRY_LOCK,
    _FTE_RESULT_HANDLES_BY_QUERY,
    _FTE_SCHEDULERS,
)

_TASK_CONTEXT_FIELDS = ("query_idx", "last_node_id", "task_id", "node_ids")


def _task_context_key(value: Any) -> tuple[int, int, int, tuple[int, ...]] | None:
    if not isinstance(value, Mapping):
        return None
    if any(field not in value for field in _TASK_CONTEXT_FIELDS):
        return None
    try:
        node_ids = tuple(int(node_id) for node_id in value["node_ids"])
        if not node_ids:
            return None
        return (
            int(value["query_idx"]),
            int(value["last_node_id"]),
            int(value["task_id"]),
            node_ids,
        )
    except (TypeError, ValueError):
        return None


def _fragment_task_context_key(fragment_execution: Any) -> tuple[int, int, int, tuple[int, ...]] | None:
    return _task_context_key(getattr(fragment_execution, "task_context_info", None))


def _safe_failure_payload(failure: Any) -> Any:
    if failure is None or isinstance(failure, str | int | float | bool):
        return failure
    if isinstance(failure, Mapping):
        return {str(key): _safe_failure_payload(value) for key, value in failure.items()}
    if isinstance(failure, list | tuple):
        return [_safe_failure_payload(value) for value in failure]
    return repr(failure)


def _partition_failure_summary(partition: Mapping[str, Any]) -> dict[str, Any] | None:
    if not bool(partition.get("failed")):
        return None
    failures = list(partition.get("failures") or [])
    latest_failure = failures[-1] if failures else None
    return {
        "partition_id": int(partition.get("partition_id", 0)),
        "task_id": str(partition.get("task_id") or ""),
        "failure_count": len(failures),
        "latest_failure": _safe_failure_payload(latest_failure),
    }


def fte_query_status(
    query_id: str,
    task_context_filter: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] | None = None,
) -> dict[str, Any]:
    query_id = str(query_id or "").strip()
    if not query_id:
        raise ValueError("query_id must be non-empty")
    scoped_contexts = None
    if task_context_filter:
        scoped_contexts = {_task_context_key(context) for context in task_context_filter}
        if None in scoped_contexts:
            raise ValueError("task_context_filter contains an invalid task context")
    with _FTE_REGISTRY_LOCK:
        registration_pending = _FTE_ACTIVE_OPERATIONS_BY_QUERY.get(query_id, 0) > 0
        canceled = query_id in _FTE_CLOSING_QUERIES
        all_fragment_execution_items = [
            (fragment_id, fragment_execution)
            for (fragment_execution_query_id, fragment_id), fragment_execution in sorted(
                _FTE_FRAGMENT_EXECUTIONS.items()
            )
            if fragment_execution_query_id == query_id
        ]
    fragment_execution_snapshots = [
        (fragment_id, fragment_execution, fragment_execution.query_status_snapshot())
        for fragment_id, fragment_execution in all_fragment_execution_items
    ]
    global_fragment_failed = any(
        bool(snapshot["failed"]) or any(bool(partition.get("failed")) for partition in snapshot["partitions"])
        for _, _, snapshot in fragment_execution_snapshots
    )
    fragment_execution_items = [
        (fragment_id, snapshot)
        for fragment_id, fragment_execution, snapshot in fragment_execution_snapshots
        if scoped_contexts is None or _fragment_task_context_key(fragment_execution) in scoped_contexts
    ]
    scheduler = _FTE_SCHEDULERS.get(query_id)
    scheduler_stats = scheduler.stats().to_dict() if scheduler is not None else None
    scheduler_failed = bool(scheduler_stats and scheduler_stats.get("state") == "FAILED")
    fragment_executions: dict[str, dict[str, Any]] = {}
    running_count = 0
    failed_count = 0
    finished_count = 0
    partition_count = 0
    failed_partitions: list[dict[str, Any]] = []
    selected_attempt_task_ids: list[str] = []
    for fragment_id, snapshot in fragment_execution_items:
        fragment_execution_running = 0
        fragment_execution_failed = 0
        fragment_execution_finished = 0
        fragment_execution_partitions = 0
        fragment_failed_partitions: list[dict[str, Any]] = []
        for partition in snapshot["partitions"]:
            fragment_execution_partitions += 1
            if bool(partition.get("running")):
                fragment_execution_running += 1
            if bool(partition.get("failed")):
                fragment_execution_failed += 1
                failure_summary = _partition_failure_summary(partition)
                if failure_summary is not None:
                    fragment_failed_partitions.append(failure_summary)
            if bool(partition.get("finished")):
                fragment_execution_finished += 1
                selected_attempt = partition.get("selected_attempt")
                task_id = str(partition.get("task_id") or "")
                if selected_attempt is not None and task_id:
                    selected_attempt_task_ids.append(f"{task_id}.{int(selected_attempt)}")
        running_count += fragment_execution_running
        failed_count += fragment_execution_failed
        finished_count += fragment_execution_finished
        partition_count += fragment_execution_partitions
        fragment_executions[fragment_id] = {
            "partition_count": fragment_execution_partitions,
            "running_count": fragment_execution_running,
            "failed_count": fragment_execution_failed,
            "finished_count": fragment_execution_finished,
            "failed": bool(snapshot["failed"] or fragment_execution_failed),
            "no_more_partitions": bool(snapshot["no_more_partitions"]),
            "finished": bool(snapshot["no_more_partitions"])
            and bool(fragment_execution_partitions)
            and fragment_execution_finished == fragment_execution_partitions,
        }
        if fragment_failed_partitions:
            fragment_executions[fragment_id]["failed_partitions"] = fragment_failed_partitions
            failed_partitions.extend({"fragment_id": fragment_id, **failure} for failure in fragment_failed_partitions)
    failed = scheduler_failed or global_fragment_failed
    finished = bool(fragment_executions) and all(bool(item["finished"]) for item in fragment_executions.values())
    status = {
        "query_id": query_id,
        "matched": bool(fragment_executions),
        "fragment_execution_count": len(fragment_executions),
        "partition_count": partition_count,
        "running_count": running_count,
        "failed_count": failed_count,
        "finished_count": finished_count,
        "failed": failed,
        "finished": finished,
        "canceled": canceled,
        "registration_pending": registration_pending,
        "selected_attempt_task_ids": selected_attempt_task_ids,
        "fragment_executions": fragment_executions,
        "failed_partitions": failed_partitions,
    }
    if scheduler_stats is not None:
        status["scheduler_state"] = scheduler_stats.get("state")
    if scheduler_failed and scheduler_stats is not None:
        status["scheduler_failure"] = scheduler_stats.get("failure_reason")
    return status


def pop_fte_result_handles(query_id: str) -> list[Any]:
    query_id = str(query_id or "").strip()
    if not query_id:
        return []
    with _FTE_REGISTRY_LOCK:
        return list(_FTE_RESULT_HANDLES_BY_QUERY.pop(query_id, []))
