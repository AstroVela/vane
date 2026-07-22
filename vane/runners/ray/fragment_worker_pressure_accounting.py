# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from vane.runners.fte import FteTaskAttemptId, FteTaskExecutionClass
from vane.runners.ray.fragment_registry import _FTE_REGISTRY_LOCK
from vane.runners.ray.fragment_worker_pressure import (
    attempt_key,
    initial_split_bytes,
    initial_split_count,
    partition_reservation_key,
)
from vane.runners.ray.fte_fragment_scheduler import _memory_requirement_bytes

if TYPE_CHECKING:
    from collections.abc import Mapping


class FteWorkerPressureAccountingMixin:
    def finish_fte_task_with_outputs(
        self,
        attempt_id: Any,
        query_task_lease: Mapping[str, Any],
        outputs: list[Mapping[str, Any]],
    ) -> list[Any]:
        from vane.runners.ray.query_resource_manager import (
            OutputBlockLeaseOwner,
            OutputBlockRequest,
        )
        from vane.runners.ray.query_resource_runtime import get_query_resource_manager

        attempt = FteTaskAttemptId.coerce(attempt_id)
        lease = dict(query_task_lease or {})
        query_id = str(lease.get("query_id") or "").strip()
        execution_query_id = str(lease.get("execution_query_id") or "").strip()
        stage_id = str(lease.get("stage_id") or "").strip()
        task_lease_id = str(lease.get("lease_id") or "").strip()
        lease_attempt_id = str(lease.get("attempt_id") or "").strip()
        if execution_query_id != attempt.task_id.query_id or lease_attempt_id != str(attempt):
            raise RuntimeError("FTE result task lease identity does not match its attempt")
        if not stage_id or not task_lease_id:
            raise RuntimeError("FTE result task lease is missing stage_id or lease_id")

        manager = get_query_resource_manager(query_id)
        requests = tuple(
            OutputBlockRequest(
                query_id=query_id,
                producer_stage_id=stage_id,
                task_lease_id=task_lease_id,
                attempt_id=lease_attempt_id,
                block_id=str(output["block_id"]),
                size_bytes=int(output["size_bytes"]),
            )
            for output in outputs
        )
        output_leases = manager.finish_task_with_outputs(
            task_lease_id,
            attempt_id=lease_attempt_id,
            outputs=requests,
        )
        return [OutputBlockLeaseOwner(manager, output_lease) for output_lease in output_leases]

    def reserve_fte_partition(
        self,
        query_id: str,
        fragment_id: str,
        partition_id: int,
        *,
        memory_requirement_bytes: Any = None,
        execution_class: FteTaskExecutionClass | str | None = None,
    ) -> None:
        key = partition_reservation_key(query_id, fragment_id, partition_id)
        memory_bytes = _memory_requirement_bytes(memory_requirement_bytes)
        task_class = FteTaskExecutionClass.coerce(execution_class)
        with _FTE_REGISTRY_LOCK:
            self._fte_pressure.reserved_partitions.add(key)
            self._fte_pressure.execution_class_by_reservation[key] = task_class.value
            if memory_bytes > 0:
                self._fte_pressure.memory_bytes_by_reservation[key] = memory_bytes
            else:
                self._fte_pressure.memory_bytes_by_reservation.pop(key, None)
            self._fte_pressure.last_seen_at = time.time()

    def release_fte_partition_reservation(self, query_id: str, fragment_id: str, partition_id: int) -> None:
        key = partition_reservation_key(query_id, fragment_id, partition_id)
        with _FTE_REGISTRY_LOCK:
            self._fte_pressure.reserved_partitions.discard(key)
            self._fte_pressure.memory_bytes_by_reservation.pop(key, None)
            self._fte_pressure.execution_class_by_reservation.pop(key, None)
            self._fte_pressure.last_seen_at = time.time()

    def set_fte_partition_reservation_execution_class(
        self,
        query_id: str,
        fragment_id: str,
        partition_id: int,
        execution_class: FteTaskExecutionClass | str | None,
    ) -> bool:
        key = partition_reservation_key(query_id, fragment_id, partition_id)
        new_class = FteTaskExecutionClass.coerce(execution_class)
        with _FTE_REGISTRY_LOCK:
            if key not in self._fte_pressure.reserved_partitions:
                return False
            old_class = FteTaskExecutionClass.coerce(
                self._fte_pressure.execution_class_by_reservation.get(
                    key,
                    FteTaskExecutionClass.STANDARD.value,
                )
            )
            if old_class == new_class:
                return False
            if not old_class.can_transition_to(new_class):
                raise ValueError(
                    f"cannot change FTE partition reservation execution class "
                    f"from {old_class.value} to {new_class.value}"
                )
            self._fte_pressure.execution_class_by_reservation[key] = new_class.value
            self._fte_pressure.last_seen_at = time.time()
            return True

    def record_fte_task_started(self, attempt_id: Any, request: Mapping[str, Any]) -> None:
        key = attempt_key(attempt_id)
        memory_requirement = request.get("memory_requirement_bytes")
        task_class = FteTaskExecutionClass.coerce(request.get("execution_class"))
        try:
            memory_requirement_bytes = max(0, int(memory_requirement)) if memory_requirement is not None else 0
        except (TypeError, ValueError):
            memory_requirement_bytes = 0
        with _FTE_REGISTRY_LOCK:
            self._fte_pressure.running_attempts.add(key)
            self._fte_pressure.execution_class_by_attempt[key] = task_class.value
            self._fte_pressure.split_counts_by_attempt[key] = initial_split_count(request)
            self._fte_pressure.split_bytes_by_attempt[key] = initial_split_bytes(request)
            if memory_requirement_bytes > 0:
                self._fte_pressure.memory_bytes_by_attempt[key] = memory_requirement_bytes
            else:
                self._fte_pressure.memory_bytes_by_attempt.pop(key, None)
            self._fte_pressure.last_seen_at = time.time()

    def record_fte_splits_added(self, attempt_id: Any, split_count: int) -> None:
        key = attempt_key(attempt_id)
        with _FTE_REGISTRY_LOCK:
            self._fte_pressure.running_attempts.add(key)
            self._fte_pressure.split_counts_by_attempt[key] = self._fte_pressure.split_counts_by_attempt.get(
                key, 0
            ) + max(0, int(split_count))
            self._fte_pressure.last_seen_at = time.time()

    def record_fte_split_bytes_added(self, attempt_id: Any, split_bytes: int) -> None:
        key = attempt_key(attempt_id)
        with _FTE_REGISTRY_LOCK:
            self._fte_pressure.running_attempts.add(key)
            self._fte_pressure.split_bytes_by_attempt[key] = self._fte_pressure.split_bytes_by_attempt.get(
                key, 0
            ) + max(0, int(split_bytes))
            self._fte_pressure.last_seen_at = time.time()

    def _record_fte_task_pressure_complete(self, attempt_id: Any, *, drain: bool) -> None:
        key = attempt_key(attempt_id)
        with _FTE_REGISTRY_LOCK:
            self._fte_pressure.running_attempts.discard(key)
            self._fte_pressure.split_counts_by_attempt.pop(key, None)
            self._fte_pressure.split_bytes_by_attempt.pop(key, None)
            self._fte_pressure.memory_bytes_by_attempt.pop(key, None)
            self._fte_pressure.execution_class_by_attempt.pop(key, None)
            self._fte_pressure.last_seen_at = time.time()
        try:
            FteTaskAttemptId.coerce(attempt_id)
        except Exception:
            return
        if drain:
            self._drain_fte_pending_tasks()

    def record_fte_task_result_ready(self, attempt_id: Any) -> None:
        """Stop charging worker execution pressure while task output is adopted."""

        self._record_fte_task_pressure_complete(attempt_id, drain=True)

    def record_fte_task_terminal(self, attempt_id: Any, *, drain: bool = True) -> None:
        self._record_fte_task_pressure_complete(attempt_id, drain=drain)

    def record_fte_task_terminal_without_drain(self, attempt_id: Any) -> None:
        self.record_fte_task_terminal(attempt_id, drain=False)

    def set_fte_task_execution_class(
        self,
        attempt_id: Any,
        execution_class: FteTaskExecutionClass | str | None,
    ) -> bool:
        key = attempt_key(attempt_id)
        new_class = FteTaskExecutionClass.coerce(execution_class)
        with _FTE_REGISTRY_LOCK:
            if key not in self._fte_pressure.running_attempts:
                return False
            old_class = FteTaskExecutionClass.coerce(
                self._fte_pressure.execution_class_by_attempt.get(
                    key,
                    FteTaskExecutionClass.STANDARD.value,
                )
            )
            if old_class == new_class:
                return False
            if not old_class.can_transition_to(new_class):
                raise ValueError(f"cannot change FTE task execution class from {old_class.value} to {new_class.value}")
            self._fte_pressure.execution_class_by_attempt[key] = new_class.value
            self._fte_pressure.last_seen_at = time.time()
            return True

    def fte_pressure_stats(self) -> dict[str, int | float]:
        with _FTE_REGISTRY_LOCK:
            return self._fte_pressure.to_dict()
