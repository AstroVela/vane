# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING, Any

from vane._ray_cxx import require_ray_cxx_attr
from vane.runners.fte import (
    AssignmentResult,
    FteFragmentExecution,
    FteTaskExecutionClass,
    FteWorkerControlFailure,
    FteWorkerReservationUnavailable,
    PartitionInfo,
)
from vane.runners.fte.dynamic_inputs import (
    split_exchange_source_task_by_partition as _split_exchange_source_task_by_partition,
)
from vane.runners.fte.dynamic_inputs import (
    splits_from_pending_task,
)
from vane.runners.fte.dynamic_inputs import (
    strip_fte_dynamic_context as _strip_fte_dynamic_context,
)
from vane.runners.fte.fte_events import SplitEventsSubmitted
from vane.runners.fte.fte_scheduler import FteEventDrivenTaskSource
from vane.runners.ray.fragment_registry import (
    _FTE_FRAGMENT_EXECUTIONS,
    _FTE_REGISTRY_LOCK,
    _FTE_SCHEDULERS,
)
from vane.runners.ray.fragment_worker_context import (
    fragment_id_for_task,
    resource_identity_from_context,
)
from vane.runners.ray.fragment_worker_exchange import mark_exchange_source_partitions_seen
from vane.runners.ray.fragment_worker_inputs import extract_task_inputs
from vane.runners.ray.fragment_worker_ordering import order_fte_handles, order_fte_scheduled_handles
from vane.runners.ray.fragment_worker_reservations import (
    fte_pending_worker_reservation_done_count,
    pending_fte_worker_reservation_future,
)
from vane.runners.ray.fragment_worker_selection import available_fte_workers
from vane.runners.ray.fragment_worker_state import fte_fragment_execution_items, get_or_create_fte_fragment_state
from vane.runners.ray.fte_fragment_scheduler import (
    _admit_fte_partition_execution_ready,
    _admit_fte_partition_node_wait,
    _exchange_source_split_key,
    _fragment_plan_ref,
    _fte_retry_remaining_delay_s,
    _has_fte_pending_standard_partitions,
    _node_requirements_have_candidates,
    _ordered_fte_fragment_execution_items_for_pending_drain,
    begin_fte_registry_operation,
    end_fte_registry_operation,
    ensure_fte_fragment_progress_topology,
    fte_registry_query_is_closing,
    transfer_fte_registry_operations_to_ref,
)
from vane.runners.ray.fte_scheduler_config import (
    _fte_allowed_no_matching_node_period_s,
    _fte_event_source_chunk_size,
    _fte_event_source_high_watermark,
    _fte_event_source_low_watermark,
)

RayWorkerTask = require_ray_cxx_attr(
    "RayWorkerTask",
    hint="Ensure the C++ ray extension is built and importable in the worker process.",
)


def _registered_fte_task_memory_bytes(
    resource_query_id: str,
    resource_stage_id: str,
) -> int:
    from vane.runners.ray.query_resource_runtime import get_query_resource_manager

    manager = get_query_resource_manager(resource_query_id)
    stage = manager.graph.stage_by_id(resource_stage_id)
    if stage.backend != "ray_worker":
        raise RuntimeError(f"FTE stage {resource_stage_id} has invalid registered backend {stage.backend!r}")
    task_memory_bytes = int(stage.per_task.heap_bytes)
    if task_memory_bytes <= 0:
        raise RuntimeError(f"FTE stage {resource_stage_id} has invalid per-task heap lease {task_memory_bytes}")
    return task_memory_bytes


if TYPE_CHECKING:
    from vane.runners.fte import FteSplit


def _fte_submission_debug_enabled() -> bool:
    for name in ("VANE_FTE_ADMISSION_DEBUG", "DUCKDB_DISTRIBUTED_DEBUG"):
        value = os.getenv(name, "")
        if value.strip().lower() not in ("", "0", "false", "no", "off"):
            return True
    return False


def _fte_submission_debug_log(event: str, **fields: Any) -> None:
    if not _fte_submission_debug_enabled():
        return
    parts = [f"event={event}", f"pid={os.getpid()}"]
    for key, value in fields.items():
        text = "None" if value is None else str(value).replace(" ", "_")
        parts.append(f"{key}={text}")
    print("[vane-fte-submit] " + " ".join(parts), file=sys.stderr, flush=True)


def _truthy_context_flag(context: dict[str, Any], key: str) -> bool:
    value = context.get(key)
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


class FteWorkerSubmissionMixin:
    def _ensure_fragment_progress_topology(
        self,
        query_id: str,
        fragment_id: str,
        fragment_plan: Any,
    ) -> bool:
        """Publish exact native topology once, before remote actor startup."""
        query_key = str(query_id or "").strip()
        fragment_key = str(fragment_id or "").strip()
        if not query_key or not fragment_key or fragment_plan is None:
            raise ValueError("fragment progress topology requires query, fragment, and plan")

        def build_topology() -> dict[str, Any]:
            import vane

            topology_conn = vane.connect(config={"threads": "1"})
            cursor = topology_conn.cursor()
            try:
                describe = require_ray_cxx_attr(
                    "describe_native_progress",
                    hint="Ensure the C++ ray extension is built and importable in the driver process.",
                )
                topology = describe(cursor, fragment_plan)
                if type(topology) is not dict:
                    raise TypeError("native fragment progress topology must be a dict")
                return topology
            finally:
                cursor.close()
                topology_conn.close()

        return ensure_fte_fragment_progress_topology(
            query_key,
            fragment_key,
            build_topology,
        )

    def _get_or_create_fte_fragment_execution(
        self,
        item: dict[str, Any],
        *,
        dynamic_scan_sources: set[str],
        dynamic_exchange_sources: set[str],
    ) -> FteFragmentExecution:
        query_id = str(item["query_id"])
        fragment_id = str(item["fragment_id"])
        key = (query_id, fragment_id)
        with _FTE_REGISTRY_LOCK:
            if fte_registry_query_is_closing(query_id):
                raise RuntimeError(f"FTE query registry is closing: {query_id}")
            _FTE_SCHEDULERS.get_or_create(query_id)
            existing = _FTE_FRAGMENT_EXECUTIONS.get(key)
            if existing is not None:
                if not existing.task_context_info and item.get("task_context_info"):
                    existing.task_context_info = dict(item["task_context_info"])
                if item.get("exchange_sink_instance") is not None:
                    existing.task_context_info["exchange_sink_instance"] = item.get("exchange_sink_instance")
                existing.source_node_ids.update(dynamic_scan_sources)
                existing.source_node_ids.update(dynamic_exchange_sources)
                existing.dynamic_scan_source_node_ids.update(dynamic_scan_sources)
                existing.dynamic_exchange_source_node_ids.update(dynamic_exchange_sources)
                existing.context = _strip_fte_dynamic_context(
                    existing.context,
                    existing.dynamic_scan_source_node_ids,
                    existing.dynamic_exchange_source_node_ids,
                )
                for partition in existing.partitions.values():
                    partition.descriptor.source_node_ids.update(dynamic_scan_sources)
                    partition.descriptor.source_node_ids.update(dynamic_exchange_sources)
                    partition.descriptor.dynamic_scan_source_node_ids.update(dynamic_scan_sources)
                    partition.descriptor.dynamic_exchange_source_node_ids.update(dynamic_exchange_sources)
                    partition.descriptor.context = _strip_fte_dynamic_context(
                        partition.descriptor.context,
                        partition.descriptor.dynamic_scan_source_node_ids,
                        partition.descriptor.dynamic_exchange_source_node_ids,
                    )
                    if not partition.descriptor.task_context_info and item.get("task_context_info"):
                        partition.descriptor.task_context_info = dict(item["task_context_info"])
                    if (
                        partition.descriptor.exchange_sink_instance is None
                        and item.get("exchange_sink_instance") is not None
                    ):
                        partition.descriptor.exchange_sink_instance = item.get("exchange_sink_instance")
                self._fte_fragment_executions[key] = existing
                return existing

        fragment_execution_context = _strip_fte_dynamic_context(
            item.get("context"),
            dynamic_scan_sources,
            dynamic_exchange_sources,
        )
        resource_query_id = str(item["resource_query_id"])
        resource_stage_id = str(item["resource_stage_id"])

        fragment_registration_result = item.get("fragment_registration_result")

        def select_partition_owner(partition):
            try:
                reservation = self._fte_worker_placement_manager.acquire(
                    query_id=query_id,
                    fragment_id=fragment_id,
                    partition_id=int(partition.task_id.partition_id),
                    memory_requirement_bytes=partition.memory_requirement_bytes,
                    execution_class=partition.execution_class,
                    node_requirements=partition.node_requirements,
                    node_requirements_wait_started_at=partition.node_wait_started_at,
                )
            except FteWorkerReservationUnavailable:
                has_matching_node = _node_requirements_have_candidates(
                    available_fte_workers(self, self.worker_id),
                    partition.node_requirements,
                    node_requirements_wait_started_at=partition.node_wait_started_at,
                )
                if not has_matching_node:
                    no_matching_period = partition.mark_no_matching_node()
                    if no_matching_period > _fte_allowed_no_matching_node_period_s():
                        raise RuntimeError(
                            f"No nodes available to run query {query_id}/{fragment_id}/{partition.task_id.partition_id}"
                        )
                else:
                    partition.reset_no_matching_node()
                raise
            return reservation.worker_id, reservation.worker

        def apply_execution_class_transitions(transitions):
            self._apply_fte_execution_class_transitions(query_id, fragment_id, transitions)

        def admit_execution(partition):
            fragment_execution = _FTE_FRAGMENT_EXECUTIONS.get((query_id, fragment_id))
            return _admit_fte_partition_execution_ready(query_id, fragment_execution, partition)

        def admit_attempt(partition):
            fragment_execution = _FTE_FRAGMENT_EXECUTIONS.get((query_id, fragment_id))
            pending_worker_reservation = pending_fte_worker_reservation_future(
                query_id,
                fragment_id,
                partition.task_id.partition_id,
            )
            if pending_worker_reservation is not None:
                self._try_complete_fte_worker_reservation_future(
                    pending_worker_reservation,
                    partition=partition,
                    raise_on_no_matching_timeout=True,
                )
                return False
            if _fte_retry_remaining_delay_s(query_id) > 0:
                return False
            return _admit_fte_partition_node_wait(query_id, partition, fragment_execution)

        def request_worker_reservation(partition):
            fragment_execution = _FTE_FRAGMENT_EXECUTIONS.get((query_id, fragment_id))
            if fragment_execution is None:
                return False
            return self._request_fte_worker_reservation_for_partition(
                query_id,
                fragment_id,
                fragment_execution,
                partition,
            )

        fragment_execution = FteFragmentExecution(
            query_id,
            self._next_fte_fragment_execution_id(query_id, fragment_id),
            fragment_id=fragment_id,
            worker_selector=select_partition_owner,
            execution_class_transition_callback=apply_execution_class_transitions,
            execution_admission_callback=admit_execution,
            attempt_admission_callback=admit_attempt,
            worker_reservation_callback=request_worker_reservation,
            context=fragment_execution_context,
            fragment_plan=item.get("fragment_plan"),
            fragment_registration_result=fragment_registration_result,
            task_context_info={
                **dict(item.get("task_context_info") or {}),
                **(
                    {"exchange_sink_instance": item.get("exchange_sink_instance")}
                    if item.get("exchange_sink_instance") is not None
                    else {}
                ),
            },
            source_node_ids=dynamic_scan_sources | dynamic_exchange_sources,
            dynamic_scan_source_node_ids=dynamic_scan_sources,
            dynamic_exchange_source_node_ids=dynamic_exchange_sources,
            task_memory_bytes=_registered_fte_task_memory_bytes(
                resource_query_id,
                resource_stage_id,
            ),
        )
        with _FTE_REGISTRY_LOCK:
            if fte_registry_query_is_closing(query_id):
                raise RuntimeError(f"FTE query registry is closing: {query_id}")
            existing = _FTE_FRAGMENT_EXECUTIONS.get(key)
            if existing is not None:
                self._fte_fragment_executions[key] = existing
                return existing
            _FTE_FRAGMENT_EXECUTIONS[key] = fragment_execution
        self._fte_fragment_executions[key] = fragment_execution
        return fragment_execution

    def _drain_fte_pending_tasks(self, *, query_id_filter: str | None = None) -> list[Any]:
        handles: list[Any] = []
        if query_id_filter is not None:
            query_id_filter = str(query_id_filter)
            if fte_registry_query_is_closing(query_id_filter):
                return []
        fragment_execution_items = fte_fragment_execution_items(query_id_filter)
        fragment_execution_items = [
            item for item in fragment_execution_items if not fte_registry_query_is_closing(item[0][0])
        ]

        def schedule_execution_class(execution_class: FteTaskExecutionClass) -> tuple[bool, bool]:
            made_progress = False
            should_drain_events = False
            released = self._release_deferred_fte_execution_partitions(
                fragment_execution_items,
                execution_class,
            )
            if released:
                made_progress = True
            for (
                query_id,
                fragment_id,
            ), fragment_execution in _ordered_fte_fragment_execution_items_for_pending_drain(
                fragment_execution_items,
                execution_class=execution_class,
            ):
                try:
                    pending_reservation_done_count_before = fte_pending_worker_reservation_done_count(query_id_filter)
                    scheduled = fragment_execution.schedule_next_pending_partition(execution_class)
                    pending_reservation_done_count_after = fte_pending_worker_reservation_done_count(query_id_filter)
                    if scheduled is not None:
                        self._execute_fte_fragment_execution_outbox(fragment_execution)
                except FteWorkerControlFailure as exc:
                    handles.extend(self._handles_for_fte_worker_control_failure(exc))
                    continue
                if scheduled is None:
                    if pending_reservation_done_count_after > pending_reservation_done_count_before:
                        made_progress = True
                        should_drain_events = True
                        break
                    continue
                made_progress = True
                handles.extend(
                    self._handles_for_fte_scheduled_attempts(
                        query_id,
                        fragment_id,
                        [scheduled],
                    )
                )
            return made_progress, should_drain_events

        while True:
            made_progress = False
            should_drain_events = False
            for execution_class in (
                FteTaskExecutionClass.EAGER_SPECULATIVE,
                FteTaskExecutionClass.STANDARD,
            ):
                class_progress, class_should_drain = schedule_execution_class(execution_class)
                made_progress = class_progress or made_progress
                should_drain_events = class_should_drain or should_drain_events
                if should_drain_events:
                    break
            if not should_drain_events and not _has_fte_pending_standard_partitions(fragment_execution_items):
                class_progress, class_should_drain = schedule_execution_class(FteTaskExecutionClass.SPECULATIVE)
                made_progress = class_progress or made_progress
                should_drain_events = class_should_drain or should_drain_events
            if should_drain_events:
                break
            if not made_progress:
                break
        for query_id in sorted({query_id for (query_id, _), _ in fragment_execution_items}):
            scheduler = _FTE_SCHEDULERS.get(query_id)
            if scheduler is not None and not scheduler.is_draining():
                handles.extend(scheduler.drain())
        return order_fte_scheduled_handles(handles)

    def _submit_fte_pending_tasks_via_scheduler(
        self,
        pending: list[dict[str, Any]],
    ) -> list[Any]:
        started_at = time.monotonic()
        handles: list[Any] = []
        pending_by_query: dict[str, list[dict[str, Any]]] = {}
        for item in pending:
            query_id = str(item.get("query_id") or "").strip()
            if not query_id:
                raise ValueError("FTE pending task is missing query_id")
            pending_by_query.setdefault(query_id, []).append(item)

        for query_id, query_pending in pending_by_query.items():
            if not begin_fte_registry_operation(query_id):
                raise RuntimeError(f"FTE query registry is closing: {query_id}")
            try:
                query_started_at = time.monotonic()
                _fte_submission_debug_log(
                    "scheduler_submit_start",
                    query_id=query_id,
                    pending_count=len(query_pending),
                )
                scheduler = _FTE_SCHEDULERS.get_or_create(query_id)
                self._bind_fte_scheduler_handlers(scheduler)
                high_watermark = _fte_event_source_high_watermark()
                chunk_size = _fte_event_source_chunk_size()
                low_watermark = _fte_event_source_low_watermark(high_watermark)
                source = FteEventDrivenTaskSource(
                    scheduler,
                    f"pending-tasks:{id(self)}:{query_id}",
                    high_watermark=high_watermark,
                    low_watermark=low_watermark,
                )

                def events_for_query(
                    query_id: str = query_id,
                    query_pending: list[dict[str, Any]] = query_pending,
                    chunk_size: int = chunk_size,
                ):
                    for offset in range(0, len(query_pending), chunk_size):
                        yield SplitEventsSubmitted.from_events(
                            query_id,
                            query_pending[offset : offset + chunk_size],
                        )

                query_handles = order_fte_handles(source.submit(events_for_query()))
                handles.extend(query_handles)
                _fte_submission_debug_log(
                    "scheduler_submit_done",
                    query_id=query_id,
                    pending_count=len(query_pending),
                    handle_count=len(query_handles),
                    elapsed_ms=int((time.monotonic() - query_started_at) * 1000),
                )
            finally:
                end_fte_registry_operation(query_id)
        _fte_submission_debug_log(
            "scheduler_submit_all_done",
            query_count=len(pending_by_query),
            pending_count=len(pending),
            handle_count=len(handles),
            elapsed_ms=int((time.monotonic() - started_at) * 1000),
        )
        return handles

    def _prepare_fte_pending_tasks(
        self,
        pending: list[dict[str, Any]],
    ) -> tuple[
        list[
            tuple[
                dict[str, Any],
                list[FteSplit],
                set[str],
                set[str],
                set[str],
                set[int],
                int,
                int,
                dict[str, tuple[set[int], int, int]],
            ]
        ],
        dict[tuple[str, str], dict[str, Any]],
    ]:
        prepared: list[
            tuple[
                dict[str, Any],
                list[FteSplit],
                set[str],
                set[str],
                set[str],
                set[int],
                int,
                int,
                dict[str, tuple[set[int], int, int]],
            ]
        ] = []
        fragment_inputs: dict[tuple[str, str], dict[str, Any]] = {}
        for item in pending:
            (
                splits,
                dynamic_scan_sources,
                dynamic_exchange_sources,
                replicated_exchange_sources,
                exchange_source_partition_ids,
                exchange_source_partition_count,
                exchange_source_task_count,
                exchange_source_metadata_by_source,
            ) = splits_from_pending_task(
                item,
                next_split_sequence=self._next_fte_split_sequence,
                split_exchange_source_task_by_partition_fn=_split_exchange_source_task_by_partition,
            )
            item["context"] = _strip_fte_dynamic_context(
                item.get("context"),
                dynamic_scan_sources,
                dynamic_exchange_sources,
            )
            fragment_key = (str(item["query_id"]), str(item["fragment_id"]))
            aggregate = fragment_inputs.setdefault(
                fragment_key,
                {
                    "dynamic_scan_sources": set(),
                    "dynamic_exchange_sources": set(),
                    "replicated_exchange_sources": set(),
                    "exchange_source_partition_ids": set(),
                    "exchange_source_partition_count": 0,
                    "exchange_source_task_count": 0,
                },
            )
            aggregate["dynamic_scan_sources"].update(dynamic_scan_sources)
            aggregate["dynamic_exchange_sources"].update(dynamic_exchange_sources)
            aggregate["replicated_exchange_sources"].update(replicated_exchange_sources)
            aggregate["exchange_source_partition_ids"].update(exchange_source_partition_ids)
            aggregate["exchange_source_partition_count"] = max(
                int(aggregate["exchange_source_partition_count"]),
                int(exchange_source_partition_count),
            )
            aggregate["exchange_source_task_count"] = max(
                int(aggregate["exchange_source_task_count"]),
                int(exchange_source_task_count),
            )
            prepared.append(
                (
                    item,
                    splits,
                    dynamic_scan_sources,
                    dynamic_exchange_sources,
                    replicated_exchange_sources,
                    exchange_source_partition_ids,
                    exchange_source_partition_count,
                    exchange_source_task_count,
                    exchange_source_metadata_by_source,
                )
            )
        return prepared, fragment_inputs

    def _submit_fte_pending_tasks(self, pending: list[dict[str, Any]]) -> list[Any]:
        started_at = time.monotonic()
        _fte_submission_debug_log("pending_submit_start", pending_count=len(pending))
        handles: list[Any] = []
        prepared, fragment_inputs = self._prepare_fte_pending_tasks(pending)
        _fte_submission_debug_log(
            "pending_submit_prepared",
            pending_count=len(pending),
            prepared_count=len(prepared),
            fragment_count=len(fragment_inputs),
            elapsed_ms=int((time.monotonic() - started_at) * 1000),
        )
        for (query_id, fragment_id), aggregate in fragment_inputs.items():
            get_or_create_fte_fragment_state(
                query_id,
                fragment_id,
                dynamic_scan_sources=aggregate["dynamic_scan_sources"],
                dynamic_exchange_sources=aggregate["dynamic_exchange_sources"],
                replicated_exchange_sources=aggregate["replicated_exchange_sources"],
                exchange_source_partition_ids=aggregate["exchange_source_partition_ids"],
                exchange_source_partition_count=aggregate["exchange_source_partition_count"],
                exchange_source_task_count=aggregate["exchange_source_task_count"],
            )

        for prepared_index, (
            item,
            splits,
            dynamic_scan_sources,
            dynamic_exchange_sources,
            replicated_exchange_sources,
            _,
            _,
            _,
            exchange_source_metadata_by_source,
        ) in enumerate(prepared):
            item_started_at = time.monotonic()
            _fte_submission_debug_log(
                "pending_item_start",
                prepared_index=prepared_index,
                prepared_count=len(prepared),
                query_id=item.get("query_id"),
                fragment_id=item.get("fragment_id"),
                split_count=len(splits),
                dynamic_scan_sources=len(dynamic_scan_sources),
                dynamic_exchange_sources=len(dynamic_exchange_sources),
            )
            fragment_execution = self._get_or_create_fte_fragment_execution(
                item,
                dynamic_scan_sources=dynamic_scan_sources,
                dynamic_exchange_sources=dynamic_exchange_sources,
            )
            _fte_submission_debug_log(
                "pending_item_fragment_execution_ready",
                prepared_index=prepared_index,
                partition_count=len(fragment_execution.partitions),
                elapsed_ms=int((time.monotonic() - item_started_at) * 1000),
            )
            fragment_state = get_or_create_fte_fragment_state(
                item["query_id"],
                item["fragment_id"],
                dynamic_scan_sources=dynamic_scan_sources,
                dynamic_exchange_sources=dynamic_exchange_sources,
                replicated_exchange_sources=replicated_exchange_sources,
                exchange_source_partition_ids=set(),
                exchange_source_partition_count=0,
                exchange_source_task_count=0,
            )
            if fragment_state.assigner is None:
                raise RuntimeError("FTE fragment state is missing split assigner")
            new_exchange_partition_ids_by_source, final_exchange_sources = mark_exchange_source_partitions_seen(
                fragment_state,
                exchange_source_metadata_by_source,
            )
            by_source: dict[str, list[FteSplit]] = {}
            for split in splits:
                if split.kind == "exchange_source_task":
                    update_partition_ids = new_exchange_partition_ids_by_source.get(split.source_node_id, set())
                    if split.source_partition_id not in update_partition_ids:
                        continue
                    split_key = _exchange_source_split_key(split)
                    seen_split_keys = fragment_state.exchange_source_split_keys_by_source.setdefault(
                        split.source_node_id,
                        set(),
                    )
                    if split_key in seen_split_keys:
                        continue
                    seen_split_keys.add(split_key)
                by_source.setdefault(split.source_node_id, []).append(split)

            scheduled_attempts = []
            if not by_source and not fragment_execution.partitions:
                _fte_submission_debug_log(
                    "pending_item_empty_source_add_partition",
                    prepared_index=prepared_index,
                    elapsed_ms=int((time.monotonic() - item_started_at) * 1000),
                )
                partition = fragment_execution.add_partition(0)
                _fte_submission_debug_log(
                    "pending_item_empty_source_reserve_start",
                    prepared_index=prepared_index,
                    partition_id=partition.task_id.partition_id,
                    elapsed_ms=int((time.monotonic() - item_started_at) * 1000),
                )
                self._try_reserve_fte_partition_for_node_wait(
                    item["query_id"],
                    item["fragment_id"],
                    partition,
                    fragment_execution=fragment_execution,
                )
                _fte_submission_debug_log(
                    "pending_item_empty_source_reserve_done",
                    prepared_index=prepared_index,
                    partition_id=partition.task_id.partition_id,
                    elapsed_ms=int((time.monotonic() - item_started_at) * 1000),
                )
                try:
                    scheduled_result = fragment_execution.apply_assignment_result(
                        AssignmentResult(
                            partitions_added=[PartitionInfo(0)],
                            sealed_partitions=[0],
                            no_more_partitions=True,
                        )
                    )
                    _fte_submission_debug_log(
                        "pending_item_empty_source_apply_done",
                        prepared_index=prepared_index,
                        scheduled_count=len(scheduled_result),
                        command_count=len(scheduled_result.worker_commands),
                        elapsed_ms=int((time.monotonic() - item_started_at) * 1000),
                    )
                    scheduled = self._execute_fte_fragment_execution_mutation_result(
                        fragment_execution, scheduled_result
                    )
                except FteWorkerControlFailure as exc:
                    handles.extend(self._handles_for_fte_worker_control_failure(exc))
                    scheduled = []
                scheduled_attempts.extend(scheduled)

            for source_node_id, source_splits in by_source.items():
                _fte_submission_debug_log(
                    "pending_item_assign_start",
                    prepared_index=prepared_index,
                    source_node_id=source_node_id,
                    split_count=len(source_splits),
                    final=source_node_id in final_exchange_sources,
                    elapsed_ms=int((time.monotonic() - item_started_at) * 1000),
                )
                result = fragment_state.assigner.assign(
                    source_node_id,
                    [split.to_dict() for split in source_splits],
                    no_more_inputs=source_node_id in final_exchange_sources,
                )
                _fte_submission_debug_log(
                    "pending_item_assign_done",
                    prepared_index=prepared_index,
                    source_node_id=source_node_id,
                    partitions_added=len(result.partitions_added),
                    partition_updates=len(result.partition_updates),
                    sealed_partitions=len(result.sealed_partitions),
                    no_more_partitions=result.no_more_partitions,
                    elapsed_ms=int((time.monotonic() - item_started_at) * 1000),
                )
                for partition_info in result.partitions_added:
                    partition = fragment_execution.add_partition(
                        partition_info.partition_id,
                        partition_info.node_requirements,
                    )
                    _fte_submission_debug_log(
                        "pending_item_reserve_start",
                        prepared_index=prepared_index,
                        partition_id=partition_info.partition_id,
                        elapsed_ms=int((time.monotonic() - item_started_at) * 1000),
                    )
                    self._try_reserve_fte_partition_for_node_wait(
                        item["query_id"],
                        item["fragment_id"],
                        partition,
                        fragment_execution=fragment_execution,
                    )
                    _fte_submission_debug_log(
                        "pending_item_reserve_done",
                        prepared_index=prepared_index,
                        partition_id=partition_info.partition_id,
                        elapsed_ms=int((time.monotonic() - item_started_at) * 1000),
                    )
                try:
                    _fte_submission_debug_log(
                        "pending_item_apply_start",
                        prepared_index=prepared_index,
                        elapsed_ms=int((time.monotonic() - item_started_at) * 1000),
                    )
                    scheduled_result = fragment_execution.apply_assignment_result(result)
                    _fte_submission_debug_log(
                        "pending_item_apply_done",
                        prepared_index=prepared_index,
                        scheduled_count=len(scheduled_result),
                        command_count=len(scheduled_result.worker_commands),
                        elapsed_ms=int((time.monotonic() - item_started_at) * 1000),
                    )
                    scheduled = self._execute_fte_fragment_execution_mutation_result(
                        fragment_execution, scheduled_result
                    )
                except FteWorkerControlFailure as exc:
                    handles.extend(self._handles_for_fte_worker_control_failure(exc))
                    scheduled = []
                scheduled_attempts.extend(scheduled)

            _fte_submission_debug_log(
                "pending_item_handles_start",
                prepared_index=prepared_index,
                scheduled_count=len(scheduled_attempts),
                elapsed_ms=int((time.monotonic() - item_started_at) * 1000),
            )
            item_handles = self._handles_for_fte_scheduled_attempts(
                item["query_id"],
                item["fragment_id"],
                scheduled_attempts,
            )
            handles.extend(item_handles)
            _fte_submission_debug_log(
                "pending_item_done",
                prepared_index=prepared_index,
                scheduled_count=len(scheduled_attempts),
                handle_count=len(item_handles),
                elapsed_ms=int((time.monotonic() - item_started_at) * 1000),
            )
        _fte_submission_debug_log(
            "pending_submit_done",
            pending_count=len(pending),
            handle_count=len(handles),
            elapsed_ms=int((time.monotonic() - started_at) * 1000),
        )
        return handles

    def submit_tasks(self, tasks: list[RayWorkerTask]) -> list[Any]:
        started_at = time.monotonic()
        _fte_submission_debug_log("submit_tasks_start", task_count=len(tasks))
        if not tasks:
            return []

        pending: list[dict[str, Any]] = []
        submitted_fragment_ids: set[str] = set()
        fragments_to_register: dict[str, dict[str, Any]] = {}
        fragment_plan_refs_by_fragment: dict[str, Any] = {}
        registration_refs_by_fragment: dict[str, Any] = {}

        for task in tasks:
            task_name = task.name()
            context = dict(task.context() or {})
            if not str(context.get("query_id", "")).strip():
                raise ValueError("FTE task requires non-empty query_id")
            task_context_info = dict(task.task_context() or {})
            context = extract_task_inputs(task, context)
            exchange_sink_instance = task.exchange_sink_instance()
            resource_query_id, resource_stage_id = resource_identity_from_context(context)

            query_id, fragment_id = fragment_id_for_task(context, task_name)
            if (
                _truthy_context_flag(context, "preserve_plan_exchange_sink_instance")
                and exchange_sink_instance is not None
            ):
                exchange_sink_instance = dict(exchange_sink_instance)
                exchange_sink_instance["preserve_plan_exchange_sink_instance"] = True
            plan = None
            fragment_plan = None
            with self._fragment_registration_lock:
                owner_query_id = self._fragment_query_ids.get(fragment_id)
                if owner_query_id is not None and owner_query_id != query_id:
                    raise RuntimeError(
                        "fragment registration query ownership mismatch: "
                        f"fragment={fragment_id} owner={owner_query_id} requested={query_id}"
                    )
                pending_registration_ref = self._fragment_registration_refs.get(fragment_id)
                fragment_registered = fragment_id in self._registered_fragment_ids
            if pending_registration_ref is not None:
                registration_refs_by_fragment[fragment_id] = pending_registration_ref
            elif not fragment_registered and fragment_id not in submitted_fragment_ids:
                if plan is None:
                    plan = task.plan()
                fragment_plan = plan
                submitted_fragment_ids.add(fragment_id)
                fragments_to_register[fragment_id] = {
                    "fragment_id": fragment_id,
                    "plan": plan,
                    "query_id": query_id,
                }

            pending.append(
                {
                    "task_name": task_name,
                    "fragment_id": fragment_id,
                    "context": context,
                    "fragment_plan": fragment_plan,
                    "query_id": query_id,
                    "resource_query_id": resource_query_id,
                    "resource_stage_id": resource_stage_id,
                    "task_context_info": task_context_info,
                    "exchange_sink_instance": exchange_sink_instance,
                }
            )

        if fragments_to_register:
            candidate_query_ids = sorted(
                {str(entry.get("query_id") or "").strip() for entry in fragments_to_register.values()}
            )
            if not candidate_query_ids or not candidate_query_ids[0]:
                raise RuntimeError("fragment registration requires non-empty query identity")
            owned_query_ids: set[str] = set()
            registration_ref = None
            registration_fragment_ids: tuple[str, ...] = ()
            try:
                for registration_query_id in candidate_query_ids:
                    if not begin_fte_registry_operation(registration_query_id):
                        raise RuntimeError(f"FTE query registry is closing: {registration_query_id}")
                    owned_query_ids.add(registration_query_id)

                for fragment_id, entry in fragments_to_register.items():
                    self._ensure_fragment_progress_topology(
                        str(entry["query_id"]),
                        fragment_id,
                        entry["plan"],
                    )

                with self._fragment_registration_lock:
                    new_registrations: dict[str, dict[str, Any]] = {}
                    for fragment_id, entry in fragments_to_register.items():
                        registration_query_id = str(entry.get("query_id") or "").strip()
                        owner_query_id = self._fragment_query_ids.get(fragment_id)
                        if owner_query_id is not None and owner_query_id != registration_query_id:
                            raise RuntimeError(
                                "fragment registration query ownership mismatch: "
                                f"fragment={fragment_id} owner={owner_query_id} "
                                f"requested={registration_query_id}"
                            )
                        pending_ref = self._fragment_registration_refs.get(fragment_id)
                        if pending_ref is not None:
                            registration_refs_by_fragment[fragment_id] = pending_ref
                        elif fragment_id not in self._registered_fragment_ids:
                            new_registrations[fragment_id] = entry
                    fragments_to_register = new_registrations

                    fragment_payloads: list[dict[str, Any]] = []
                    for fragment_id, entry in fragments_to_register.items():
                        payload = dict(entry)
                        plan_ref = _fragment_plan_ref(
                            str(payload.get("query_id") or ""),
                            fragment_id,
                            payload.get("plan"),
                        )
                        payload["plan"] = plan_ref
                        fragment_plan_refs_by_fragment[fragment_id] = plan_ref
                        fragment_payloads.append(payload)
                    if fragment_payloads:
                        _fte_submission_debug_log(
                            "register_fragments_remote_start",
                            fragment_count=len(fragment_payloads),
                        )
                        registration_query_ids = {
                            str(payload.get("query_id") or "").strip() for payload in fragment_payloads
                        }
                        if not registration_query_ids or "" in registration_query_ids:
                            raise RuntimeError("fragment registration requires non-empty query identity")
                        registration_ref = self.actor_handle.register_fragments.remote(fragment_payloads)
                        registration_fragment_ids = tuple(sorted(fragments_to_register))
                        self._registered_fragment_ids.update(registration_fragment_ids)
                        for submitted_fragment_id in registration_fragment_ids:
                            self._fragment_registration_refs[submitted_fragment_id] = registration_ref
                            self._fragment_query_ids[submitted_fragment_id] = str(
                                fragments_to_register[submitted_fragment_id]["query_id"]
                            )
                    else:
                        registration_query_ids = set()

                unused_query_ids = owned_query_ids - registration_query_ids
                for registration_query_id in unused_query_ids:
                    end_fte_registry_operation(registration_query_id)
                owned_query_ids.difference_update(unused_query_ids)

                def finish_registrations(*, failed: bool) -> None:
                    with self._fragment_registration_lock:
                        for submitted_fragment_id in registration_fragment_ids:
                            if self._fragment_registration_refs.get(submitted_fragment_id) is not registration_ref:
                                continue
                            self._fragment_registration_refs.pop(submitted_fragment_id, None)
                            if failed:
                                self._registered_fragment_ids.discard(submitted_fragment_id)
                                self._fragment_query_ids.pop(submitted_fragment_id, None)

                if registration_ref is not None:
                    transfer_fte_registry_operations_to_ref(
                        sorted(owned_query_ids),
                        registration_ref,
                        on_success=lambda: finish_registrations(failed=False),
                        on_failure=lambda: finish_registrations(failed=True),
                    )
                    owned_query_ids.clear()
            except BaseException:
                with self._fragment_registration_lock:
                    for submitted_fragment_id in registration_fragment_ids:
                        if self._fragment_registration_refs.get(submitted_fragment_id) is registration_ref:
                            self._fragment_registration_refs.pop(submitted_fragment_id, None)
                            self._registered_fragment_ids.discard(submitted_fragment_id)
                            self._fragment_query_ids.pop(submitted_fragment_id, None)
                for registration_query_id in owned_query_ids:
                    end_fte_registry_operation(registration_query_id)
                raise
            for fragment_id in fragments_to_register:
                registration_refs_by_fragment[fragment_id] = registration_ref
            if registration_ref is not None:
                _fte_submission_debug_log(
                    "register_fragments_remote_submitted",
                    fragment_count=len(registration_fragment_ids),
                )

        for item in pending:
            registration_result = registration_refs_by_fragment.get(item["fragment_id"])
            item["fragment_registration_result"] = registration_result
            if registration_result is not None:
                item["fragment_plan"] = fragment_plan_refs_by_fragment.get(item["fragment_id"])
        handles = self._submit_fte_pending_tasks_via_scheduler(pending)
        _fte_submission_debug_log(
            "submit_tasks_done",
            task_count=len(tasks),
            pending_count=len(pending),
            handle_count=len(handles),
            elapsed_ms=int((time.monotonic() - started_at) * 1000),
        )
        return handles
