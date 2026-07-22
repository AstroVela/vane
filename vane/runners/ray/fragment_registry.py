# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vane.runners.fte import (
    FteTaskAttemptId,
    FteTaskExecutionClass,
)
from vane.runners.fte.fte_scheduler import FteSchedulerRegistry

if TYPE_CHECKING:
    import ray

    from vane.runners.fte import FteFragmentExecution, SplitAssigner
    from vane.runners.fte.fte_scheduler import FteAttemptStatusWatcher
    from vane.runners.ray.fragment_worker_client import RayWorkerActorHandle
    from vane.runners.ray.fte_fragment_scheduler import FteWorkerReservationFuture


@dataclass(frozen=True)
class _FteExchangeSourceOutputSelectorSnapshot:
    version: int
    source_node_id: str
    final: bool
    partition_count: int | None
    selected: dict[int, dict[str, Any]]

    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.source_node_id,
            bool(self.final),
            self.partition_count,
            tuple(
                (
                    int(partition_id),
                    entry.get("attempt_id"),
                    entry.get("split_key"),
                )
                for partition_id, entry in sorted(self.selected.items())
            ),
        )

    def to_metrics(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "source_node_id": self.source_node_id,
            "final": bool(self.final),
            "partition_count": self.partition_count,
            "selected_partitions": sorted(int(partition_id) for partition_id in self.selected),
            "selected_attempts": {
                str(partition_id): entry.get("attempt_id") for partition_id, entry in sorted(self.selected.items())
            },
        }


class _FteFragmentState:
    def __init__(self) -> None:
        self.assigner: SplitAssigner | None = None
        self.source_node_ids: set[str] = set()
        self.dynamic_scan_source_node_ids: set[str] = set()
        self.dynamic_exchange_source_node_ids: set[str] = set()
        self.replicated_exchange_source_node_ids: set[str] = set()
        self.exchange_source_partition_ids: set[int] = set()
        self.exchange_source_partition_count: int = 0
        self.exchange_source_task_count: int = 0
        self.exchange_source_partition_ids_by_source: dict[str, set[int]] = {}
        self.exchange_source_split_keys_by_source: dict[str, set[str]] = {}
        self.exchange_source_partition_count_by_source: dict[str, int] = {}
        self.exchange_source_task_count_by_source: dict[str, int] = {}
        self.exchange_source_selectors_by_source: dict[str, _FteExchangeSourceOutputSelectorSnapshot] = {}
        self.exchange_source_selector_next_version_by_source: dict[str, int] = {}
        self.exhausted_source_node_ids: set[str] = set()


class _FteSchedulingDelayer:
    def __init__(
        self,
        *,
        initial_delay_s: float,
        max_delay_s: float,
        scale_factor: float,
    ) -> None:
        self.initial_delay_s = max(0.0, float(initial_delay_s))
        self.max_delay_s = max(self.initial_delay_s, float(max_delay_s))
        self.scale_factor = max(1.0, float(scale_factor))
        self.current_delay_s = 0.0
        self.delay_started_at_s: float | None = None

    def start_or_prolong_delay_if_necessary(self) -> None:
        now = time.monotonic()
        if self.delay_started_at_s is None:
            self.delay_started_at_s = now
            self.current_delay_s = self.initial_delay_s
            return
        if now - self.delay_started_at_s > self.current_delay_s:
            self.delay_started_at_s = now
            self.current_delay_s = min(
                self.current_delay_s * self.scale_factor,
                self.max_delay_s,
            )

    def remaining_delay_s(self) -> float:
        if self.delay_started_at_s is None:
            return 0.0
        elapsed = time.monotonic() - self.delay_started_at_s
        return max(0.0, self.current_delay_s - elapsed)


class _FteWorkerPressure:
    def __init__(self) -> None:
        self.running_attempts: set[str] = set()
        self.reserved_partitions: set[tuple[str, str, int]] = set()
        self.split_counts_by_attempt: dict[str, int] = {}
        self.split_bytes_by_attempt: dict[str, int] = {}
        self.memory_bytes_by_attempt: dict[str, int] = {}
        self.memory_bytes_by_reservation: dict[tuple[str, str, int], int] = {}
        self.execution_class_by_attempt: dict[str, str] = {}
        self.execution_class_by_reservation: dict[tuple[str, str, int], str] = {}
        self.last_seen_at = time.time()

    def score(self) -> int:
        return (
            (len(self.running_attempts) + len(self.reserved_partitions)) * 1024
            + (
                (sum(self.memory_bytes_by_attempt.values()) + sum(self.memory_bytes_by_reservation.values()))
                // (1024 * 1024 * 1024)
            )
            + sum(self.split_counts_by_attempt.values())
            + (sum(self.split_bytes_by_attempt.values()) // (1024 * 1024))
        )

    @staticmethod
    def _memory_bytes_for_class(
        memory_by_key: dict[str, int],
        execution_class_by_key: dict[str, str],
        execution_class: FteTaskExecutionClass,
    ) -> int:
        return sum(
            memory_bytes
            for key, memory_bytes in memory_by_key.items()
            if FteTaskExecutionClass.coerce(execution_class_by_key.get(key, FteTaskExecutionClass.STANDARD.value))
            == execution_class
        )

    def to_dict(self) -> dict[str, int | float]:
        assigned_memory_bytes = sum(self.memory_bytes_by_attempt.values())
        reserved_memory_bytes = sum(self.memory_bytes_by_reservation.values())
        assigned_standard_memory_bytes = self._memory_bytes_for_class(
            self.memory_bytes_by_attempt,
            self.execution_class_by_attempt,
            FteTaskExecutionClass.STANDARD,
        )
        reserved_standard_memory_bytes = self._memory_bytes_for_class(
            self.memory_bytes_by_reservation,
            self.execution_class_by_reservation,
            FteTaskExecutionClass.STANDARD,
        )
        assigned_speculative_memory_bytes = self._memory_bytes_for_class(
            self.memory_bytes_by_attempt,
            self.execution_class_by_attempt,
            FteTaskExecutionClass.SPECULATIVE,
        )
        reserved_speculative_memory_bytes = self._memory_bytes_for_class(
            self.memory_bytes_by_reservation,
            self.execution_class_by_reservation,
            FteTaskExecutionClass.SPECULATIVE,
        )
        assigned_eager_speculative_memory_bytes = self._memory_bytes_for_class(
            self.memory_bytes_by_attempt,
            self.execution_class_by_attempt,
            FteTaskExecutionClass.EAGER_SPECULATIVE,
        )
        reserved_eager_speculative_memory_bytes = self._memory_bytes_for_class(
            self.memory_bytes_by_reservation,
            self.execution_class_by_reservation,
            FteTaskExecutionClass.EAGER_SPECULATIVE,
        )
        return {
            "running_attempt_count": len(self.running_attempts),
            "reserved_partition_count": len(self.reserved_partitions),
            "assigned_split_count": sum(self.split_counts_by_attempt.values()),
            "assigned_split_bytes": sum(self.split_bytes_by_attempt.values()),
            "assigned_memory_bytes": assigned_memory_bytes,
            "reserved_memory_bytes": reserved_memory_bytes,
            "total_memory_bytes": assigned_memory_bytes + reserved_memory_bytes,
            "assigned_standard_memory_bytes": assigned_standard_memory_bytes,
            "reserved_standard_memory_bytes": reserved_standard_memory_bytes,
            "standard_memory_bytes": assigned_standard_memory_bytes + reserved_standard_memory_bytes,
            "assigned_speculative_memory_bytes": assigned_speculative_memory_bytes,
            "reserved_speculative_memory_bytes": reserved_speculative_memory_bytes,
            "speculative_memory_bytes": assigned_speculative_memory_bytes + reserved_speculative_memory_bytes,
            "assigned_eager_speculative_memory_bytes": assigned_eager_speculative_memory_bytes,
            "reserved_eager_speculative_memory_bytes": reserved_eager_speculative_memory_bytes,
            "eager_speculative_memory_bytes": (
                assigned_eager_speculative_memory_bytes + reserved_eager_speculative_memory_bytes
            ),
            "score": self.score(),
            "last_seen_at": self.last_seen_at,
        }

    def drop_query(self, query_id: str) -> None:
        query_id = str(query_id or "").strip()
        if not query_id:
            return

        def owned_attempt(attempt: str) -> bool:
            return FteTaskAttemptId.parse(attempt).query_id == query_id

        self.running_attempts = {attempt for attempt in self.running_attempts if not owned_attempt(attempt)}
        self.reserved_partitions = {
            reservation for reservation in self.reserved_partitions if reservation[0] != query_id
        }
        self.split_counts_by_attempt = {
            attempt: count for attempt, count in self.split_counts_by_attempt.items() if not owned_attempt(attempt)
        }
        self.split_bytes_by_attempt = {
            attempt: count for attempt, count in self.split_bytes_by_attempt.items() if not owned_attempt(attempt)
        }
        self.memory_bytes_by_attempt = {
            attempt: count for attempt, count in self.memory_bytes_by_attempt.items() if not owned_attempt(attempt)
        }
        self.memory_bytes_by_reservation = {
            reservation: count
            for reservation, count in self.memory_bytes_by_reservation.items()
            if reservation[0] != query_id
        }
        self.execution_class_by_attempt = {
            attempt: execution_class
            for attempt, execution_class in self.execution_class_by_attempt.items()
            if not owned_attempt(attempt)
        }
        self.execution_class_by_reservation = {
            reservation: execution_class
            for reservation, execution_class in self.execution_class_by_reservation.items()
            if reservation[0] != query_id
        }
        self.last_seen_at = time.time()


@dataclass
class FteRegistryState:
    fragment_plan_ref_cache: dict[tuple[str, str, str], ray.ObjectRef] = field(default_factory=dict)
    fragment_plan_ref_cache_lock: Any = field(default_factory=threading.Lock)
    registry_lock: Any = field(default_factory=threading.RLock)
    registry_condition: Any = field(init=False)
    fragment_execution_ids: dict[tuple[str, str], int] = field(default_factory=dict)
    query_next_fragment_execution_id: dict[str, int] = field(default_factory=dict)
    fragment_executions: dict[tuple[str, str], FteFragmentExecution] = field(default_factory=dict)
    fragment_progress_topologies: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    fragment_progress_topology_builds: set[tuple[str, str]] = field(default_factory=set)
    partition_owners: dict[tuple[str, str, int], RayWorkerActorHandle] = field(default_factory=dict)
    sequences: dict[tuple[str, str, str], int] = field(default_factory=dict)
    worker_handles: dict[str, RayWorkerActorHandle] = field(default_factory=dict)
    schedulers: FteSchedulerRegistry = field(default_factory=FteSchedulerRegistry)
    status_watchers: dict[str, FteAttemptStatusWatcher] = field(default_factory=dict)
    closing_queries: set[str] = field(default_factory=set)
    active_operations_by_query: dict[str, int] = field(default_factory=dict)
    active_teardown_operations_by_query: dict[str, int] = field(default_factory=dict)
    worker_reservation_generations: dict[tuple[str, str, int], int] = field(default_factory=dict)
    pending_worker_reservations: dict[tuple[str, str, int], FteWorkerReservationFuture] = field(default_factory=dict)
    partition_task_waiters: dict[
        tuple[str, str, int],
        tuple[str, str, str],
    ] = field(default_factory=dict)
    # FTE keeps logical task descriptors outside QRM until one descriptor can
    # actually enter the execution window.  At most one partition per
    # resource stage may probe admission at a time.  A denial is memoized at
    # the QRM admission epoch so the remaining descriptors stay passive until
    # a real resource/accounting change occurs.
    stage_submission_probes: dict[
        tuple[str, str],
        tuple[str, str, int],
    ] = field(default_factory=dict)
    stage_submission_blocks: dict[
        tuple[str, str],
        tuple[int, str, tuple[str, str, int]],
    ] = field(default_factory=dict)
    partition_task_leases: dict[tuple[str, str, int], tuple[int, Any]] = field(default_factory=dict)
    result_handles_by_query: dict[str, list[Any]] = field(default_factory=dict)
    fragment_states: dict[tuple[str, str], _FteFragmentState] = field(default_factory=dict)
    retry_delays: dict[str, _FteSchedulingDelayer] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.registry_condition = threading.Condition(self.registry_lock)


_FTE_REGISTRY_STATE = FteRegistryState()

_FRAGMENT_PLAN_REF_CACHE = _FTE_REGISTRY_STATE.fragment_plan_ref_cache
_FRAGMENT_PLAN_REF_CACHE_LOCK = _FTE_REGISTRY_STATE.fragment_plan_ref_cache_lock
_FTE_REGISTRY_LOCK = _FTE_REGISTRY_STATE.registry_lock
_FTE_REGISTRY_CONDITION = _FTE_REGISTRY_STATE.registry_condition
_FTE_FRAGMENT_EXECUTION_IDS = _FTE_REGISTRY_STATE.fragment_execution_ids
_FTE_QUERY_NEXT_FRAGMENT_EXECUTION_ID = _FTE_REGISTRY_STATE.query_next_fragment_execution_id
_FTE_FRAGMENT_EXECUTIONS = _FTE_REGISTRY_STATE.fragment_executions
_FTE_FRAGMENT_PROGRESS_TOPOLOGIES = _FTE_REGISTRY_STATE.fragment_progress_topologies
_FTE_FRAGMENT_PROGRESS_TOPOLOGY_BUILDS = _FTE_REGISTRY_STATE.fragment_progress_topology_builds
_FTE_PARTITION_OWNERS = _FTE_REGISTRY_STATE.partition_owners
_FTE_SEQUENCES = _FTE_REGISTRY_STATE.sequences
_FTE_WORKER_HANDLES = _FTE_REGISTRY_STATE.worker_handles
_FTE_SCHEDULERS = _FTE_REGISTRY_STATE.schedulers
_FTE_STATUS_WATCHERS = _FTE_REGISTRY_STATE.status_watchers
_FTE_CLOSING_QUERIES = _FTE_REGISTRY_STATE.closing_queries
_FTE_ACTIVE_OPERATIONS_BY_QUERY = _FTE_REGISTRY_STATE.active_operations_by_query
_FTE_ACTIVE_TEARDOWN_OPERATIONS_BY_QUERY = _FTE_REGISTRY_STATE.active_teardown_operations_by_query
_FTE_WORKER_RESERVATION_GENERATIONS = _FTE_REGISTRY_STATE.worker_reservation_generations
_FTE_PENDING_WORKER_RESERVATIONS = _FTE_REGISTRY_STATE.pending_worker_reservations
_FTE_PARTITION_TASK_WAITERS = _FTE_REGISTRY_STATE.partition_task_waiters
_FTE_STAGE_SUBMISSION_PROBES = _FTE_REGISTRY_STATE.stage_submission_probes
_FTE_STAGE_SUBMISSION_BLOCKS = _FTE_REGISTRY_STATE.stage_submission_blocks
_FTE_PARTITION_TASK_LEASES = _FTE_REGISTRY_STATE.partition_task_leases
_FTE_RESULT_HANDLES_BY_QUERY = _FTE_REGISTRY_STATE.result_handles_by_query
_FTE_FRAGMENT_STATES = _FTE_REGISTRY_STATE.fragment_states
_FTE_RETRY_DELAYS = _FTE_REGISTRY_STATE.retry_delays


__all__ = [
    "_FRAGMENT_PLAN_REF_CACHE",
    "_FRAGMENT_PLAN_REF_CACHE_LOCK",
    "_FTE_ACTIVE_OPERATIONS_BY_QUERY",
    "_FTE_ACTIVE_TEARDOWN_OPERATIONS_BY_QUERY",
    "_FTE_CLOSING_QUERIES",
    "_FTE_FRAGMENT_EXECUTIONS",
    "_FTE_FRAGMENT_EXECUTION_IDS",
    "_FTE_FRAGMENT_PROGRESS_TOPOLOGIES",
    "_FTE_FRAGMENT_PROGRESS_TOPOLOGY_BUILDS",
    "_FTE_FRAGMENT_STATES",
    "_FTE_PARTITION_OWNERS",
    "_FTE_PARTITION_TASK_LEASES",
    "_FTE_PARTITION_TASK_WAITERS",
    "_FTE_PENDING_WORKER_RESERVATIONS",
    "_FTE_QUERY_NEXT_FRAGMENT_EXECUTION_ID",
    "_FTE_REGISTRY_CONDITION",
    "_FTE_REGISTRY_LOCK",
    "_FTE_REGISTRY_STATE",
    "_FTE_RESULT_HANDLES_BY_QUERY",
    "_FTE_RETRY_DELAYS",
    "_FTE_SCHEDULERS",
    "_FTE_SEQUENCES",
    "_FTE_STAGE_SUBMISSION_BLOCKS",
    "_FTE_STAGE_SUBMISSION_PROBES",
    "_FTE_STATUS_WATCHERS",
    "_FTE_WORKER_HANDLES",
    "_FTE_WORKER_RESERVATION_GENERATIONS",
    "FteRegistryState",
    "_FteExchangeSourceOutputSelectorSnapshot",
    "_FteFragmentState",
    "_FteSchedulingDelayer",
    "_FteWorkerPressure",
]
