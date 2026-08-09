# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from vane.runners.fte import FteTaskAttemptId, FteTaskExecutionClass
from vane.runners.ray.fragment_registry import (
    _FTE_FRAGMENT_EXECUTION_IDS,
    _FTE_FRAGMENT_EXECUTIONS,
    _FTE_REGISTRY_LOCK,
)


def fragment_id_for_fte_fragment_execution_id(query_id: str, fragment_execution_id: int) -> str | None:
    with _FTE_REGISTRY_LOCK:
        for (
            candidate_query_id,
            fragment_id,
        ), candidate_fragment_execution_id in _FTE_FRAGMENT_EXECUTION_IDS.items():
            if candidate_query_id == str(query_id) and int(candidate_fragment_execution_id) == int(
                fragment_execution_id
            ):
                return fragment_id
    return None


def fragment_execution_key_for_fte_attempt(attempt_id: FteTaskAttemptId) -> tuple[str, str] | None:
    query_id = attempt_id.task_id.query_id
    fragment_execution_id = attempt_id.task_id.fragment_execution_id
    with _FTE_REGISTRY_LOCK:
        for (
            candidate_query_id,
            fragment_id,
        ), candidate_fragment_execution_id in _FTE_FRAGMENT_EXECUTION_IDS.items():
            if candidate_query_id == query_id and int(candidate_fragment_execution_id) == int(fragment_execution_id):
                return candidate_query_id, fragment_id
    return None


def order_fte_handles(handles: list[Any]) -> list[Any]:
    def key(item: tuple[int, Any]) -> tuple[Any, ...]:
        index, handle = item
        raw_task_id = getattr(handle, "task_id", None)
        if raw_task_id is None:
            return (1, index)
        try:
            attempt = FteTaskAttemptId.coerce(raw_task_id)
        except Exception:
            return (1, index)
        task_id = attempt.task_id
        return (
            0,
            task_id.query_id,
            task_id.fragment_execution_id,
            task_id.partition_id,
            attempt.attempt_id,
            index,
        )

    return [handle for _, handle in sorted(enumerate(handles), key=key)]


def order_fte_scheduled_handles(handles: list[Any]) -> list[Any]:
    priority_by_class = {
        FteTaskExecutionClass.EAGER_SPECULATIVE: 0,
        FteTaskExecutionClass.STANDARD: 1,
        FteTaskExecutionClass.SPECULATIVE: 2,
    }

    def key(item: tuple[int, Any]) -> tuple[int, int]:
        index, handle = item
        raw_task_id = getattr(handle, "task_id", None)
        if raw_task_id is None:
            return (len(priority_by_class), index)
        try:
            attempt = FteTaskAttemptId.coerce(raw_task_id)
        except Exception:
            return (len(priority_by_class), index)
        fragment_id = fragment_id_for_fte_fragment_execution_id(
            attempt.task_id.query_id,
            attempt.task_id.fragment_execution_id,
        )
        if fragment_id is None:
            return (len(priority_by_class), index)
        with _FTE_REGISTRY_LOCK:
            fragment_execution = _FTE_FRAGMENT_EXECUTIONS.get((attempt.task_id.query_id, fragment_id))
        if fragment_execution is None:
            return (len(priority_by_class), index)
        partition = fragment_execution.partitions.get(attempt.task_id.partition_id)
        if partition is None:
            return (len(priority_by_class), index)
        return (
            priority_by_class.get(
                FteTaskExecutionClass.coerce(partition.execution_class),
                len(priority_by_class),
            ),
            index,
        )

    return [handle for _, handle in sorted(enumerate(handles), key=key)]
