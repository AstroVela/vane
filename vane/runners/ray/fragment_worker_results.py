# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vane.runners.ray.fragment_registry import (
    _FTE_FRAGMENT_EXECUTIONS,
    _FTE_REGISTRY_LOCK,
    _FTE_RESULT_HANDLES_BY_QUERY,
    _FTE_SCHEDULERS,
)


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


def fte_query_status(query_id: str) -> dict[str, Any]:
    query_id = str(query_id or "").strip()
    if not query_id:
        raise ValueError("query_id must be non-empty")
    with _FTE_REGISTRY_LOCK:
        fragment_execution_items = [
            (fragment_id, fragment_execution)
            for (fragment_execution_query_id, fragment_id), fragment_execution in sorted(
                _FTE_FRAGMENT_EXECUTIONS.items()
            )
            if fragment_execution_query_id == query_id
        ]
    scheduler = _FTE_SCHEDULERS.get(query_id)
    scheduler_stats = scheduler.stats().to_dict() if scheduler is not None else None
    scheduler_failed = bool(scheduler_stats and scheduler_stats.get("state") == "FAILED")
    fragment_executions: dict[str, dict[str, int | bool]] = {}
    running_count = 0
    failed_count = 0
    finished_count = 0
    partition_count = 0
    failed_partitions: list[dict[str, Any]] = []
    selected_attempt_task_ids: list[str] = []
    for fragment_id, fragment_execution in fragment_execution_items:
        fragment_execution_running = 0
        fragment_execution_failed = 0
        fragment_execution_finished = 0
        fragment_execution_partitions = 0
        fragment_failed_partitions: list[dict[str, Any]] = []
        snapshot = fragment_execution.query_status_snapshot()
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
            "finished": bool(fragment_execution_partitions)
            and fragment_execution_finished == fragment_execution_partitions,
        }
        if fragment_failed_partitions:
            fragment_executions[fragment_id]["failed_partitions"] = fragment_failed_partitions
            failed_partitions.extend({"fragment_id": fragment_id, **failure} for failure in fragment_failed_partitions)
    failed = scheduler_failed or failed_count > 0 or any(bool(item["failed"]) for item in fragment_executions.values())
    finished = bool(fragment_executions) and all(bool(item["finished"]) for item in fragment_executions.values())
    status = {
        "query_id": query_id,
        "fragment_execution_count": len(fragment_executions),
        "partition_count": partition_count,
        "running_count": running_count,
        "failed_count": failed_count,
        "finished_count": finished_count,
        "failed": failed,
        "finished": finished,
        "selected_attempt_task_ids": selected_attempt_task_ids,
        "fragment_executions": fragment_executions,
        "failed_partitions": failed_partitions,
    }
    if scheduler_stats is not None:
        status["scheduler_state"] = scheduler_stats.get("state")
    if scheduler_failed:
        status["scheduler_failure"] = scheduler_stats.get("failure_reason")
    return status


def pop_fte_result_handles(query_id: str) -> list[Any]:
    query_id = str(query_id or "").strip()
    if not query_id:
        return []
    with _FTE_REGISTRY_LOCK:
        return list(_FTE_RESULT_HANDLES_BY_QUERY.pop(query_id, []))
