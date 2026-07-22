# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import ray

from vane.runners.fte import (
    FteTaskAttemptId,
    FteWorkerControlFailure,
    validate_fte_status_identity,
)
from vane.runners.fte.fte_events import (
    ExchangeSelectorUpdated,
    SourceInputExhausted,
    TaskStatusChanged,
    WorkerFailed,
)
from vane.runners.ray.fragment_registry import (
    _FTE_FRAGMENT_EXECUTIONS,
    _FTE_FRAGMENT_STATES,
    _FTE_REGISTRY_LOCK,
    _FTE_SCHEDULERS,
    _FTE_WORKER_HANDLES,
)
from vane.runners.ray.fragment_worker_results import (
    fte_query_status,
    pop_fte_result_handles,
)
from vane.runners.ray.fragment_worker_state import fte_fragment_execution_query_ids
from vane.runners.ray.fte_fragment_scheduler import (
    _drop_fragment_plan_refs_for_query,
    _drop_fte_registry_for_query,
    _expanded_fte_failed_worker_ids,
    _fragment_plan_ref,
    begin_fte_registry_operation,
    begin_fte_registry_teardown_operation,
    close_fte_registry_for_query,
    end_fte_registry_operation,
    end_fte_registry_teardown_operation,
    fte_registry_query_is_closing,
    fte_registry_stats,
    quiesce_fte_registry_for_query,
    transfer_fte_registry_operations_to_ref,
    transfer_fte_registry_teardown_operations_to_ref,
)
from vane.runners.ray.safe_get import resolve_object_refs_blocking

if TYPE_CHECKING:
    from collections.abc import Mapping


class FteWorkerLifecycleMixin:
    def _drop_fragment_registration_state(self, query_id: str) -> None:
        query_key = str(query_id or "").strip()
        if not query_key:
            return
        with self._fragment_registration_lock:
            fragment_ids = {
                fragment_id
                for fragment_id, owner_query_id in self._fragment_query_ids.items()
                if owner_query_id == query_key
            }
            self._registered_fragment_ids.difference_update(fragment_ids)
            for fragment_id in fragment_ids:
                self._fragment_registration_refs.pop(fragment_id, None)
                self._fragment_query_ids.pop(fragment_id, None)

    def _has_fragment_registration_state_for_query(self, query_id: str) -> bool:
        query_key = str(query_id or "").strip()
        with self._fragment_registration_lock:
            return any(owner_query_id == query_key for owner_query_id in self._fragment_query_ids.values())

    def drop_query_fragments(self, query_id: str) -> int:
        query_id = (query_id or "").strip()
        if not query_id:
            return 0
        close_fte_registry_for_query(query_id)
        quiesce_fte_registry_for_query(query_id)
        with self._fte_control_lock:
            self._fragment_drop_incomplete_queries.add(query_id)
        remote_error: BaseException | None = None
        removed = 0
        owns_teardown_operation = False
        try:
            begin_fte_registry_teardown_operation(query_id)
            owns_teardown_operation = True
            drop_ref = self.actor_handle.drop_query_fragments.remote(query_id)
            transfer_fte_registry_teardown_operations_to_ref(
                [query_id],
                drop_ref,
            )
            owns_teardown_operation = False
            removed = int(
                resolve_object_refs_blocking(
                    drop_ref,
                    honor_query_deadline=False,
                )
            )
        except BaseException as exc:
            remote_error = exc
        finally:
            if owns_teardown_operation:
                end_fte_registry_teardown_operation(query_id)
        local_error: BaseException | None = None
        if remote_error is None:
            try:
                self._drop_fragment_registration_state(query_id)
                _drop_fragment_plan_refs_for_query(query_id)
                _drop_fte_registry_for_query(query_id)
                with self._fte_control_lock:
                    self._fragment_drop_incomplete_queries.discard(query_id)
            except BaseException as exc:
                local_error = exc
        if remote_error is not None or local_error is not None:
            details = []
            if remote_error is not None:
                details.append(f"remote={type(remote_error).__name__}: {remote_error}")
            if local_error is not None:
                details.append(f"local={type(local_error).__name__}: {local_error}")
            cause = local_error if local_error is not None else remote_error
            raise RuntimeError(f"fragment teardown failed for {query_id}: " + "; ".join(details)) from cause
        return removed

    def stats_fragments(self) -> dict[str, int]:
        # Progress/status must never block indefinitely on a busy worker.
        raw_stats = resolve_object_refs_blocking(
            self.actor_handle.stats_fragments.remote(),
            timeout=0.25,
        )
        if not isinstance(raw_stats, dict):
            raise TypeError("worker actor stats_fragments must return a dict")
        stats: dict[str, int] = {}
        for key, value in raw_stats.items():
            stats[str(key)] = int(value)
        return stats

    def ensure_fragment_registered(self, query_id: str, fragment_id: str, fragment_plan: Any) -> Any | None:
        fragment_id = str(fragment_id or "").strip()
        if not fragment_id:
            return None
        query_id = str(query_id or "").strip()
        registration_ref = None
        existing_result = None
        if not begin_fte_registry_operation(query_id):
            raise RuntimeError(f"FTE query registry is closing: {query_id}")
        owns_registry_operation = True
        try:
            should_register = False
            with self._fragment_registration_lock:
                owner_query_id = self._fragment_query_ids.get(fragment_id)
                if owner_query_id is not None and owner_query_id != query_id:
                    raise RuntimeError(
                        "fragment registration query ownership mismatch: "
                        f"fragment={fragment_id} owner={owner_query_id} requested={query_id}"
                    )
                pending_ref = self._fragment_registration_refs.get(fragment_id)
                if pending_ref is not None:
                    existing_result = pending_ref
                elif fragment_id not in self._registered_fragment_ids and fragment_plan is not None:
                    should_register = True

            if should_register:
                self._ensure_fragment_progress_topology(
                    query_id,
                    fragment_id,
                    fragment_plan,
                )
                with self._fragment_registration_lock:
                    owner_query_id = self._fragment_query_ids.get(fragment_id)
                    if owner_query_id is not None and owner_query_id != query_id:
                        raise RuntimeError(
                            "fragment registration query ownership mismatch: "
                            f"fragment={fragment_id} owner={owner_query_id} requested={query_id}"
                        )
                    pending_ref = self._fragment_registration_refs.get(fragment_id)
                    if pending_ref is not None:
                        existing_result = pending_ref
                    elif fragment_id not in self._registered_fragment_ids:
                        payload = {
                            "fragment_id": fragment_id,
                            "plan": _fragment_plan_ref(
                                query_id,
                                fragment_id,
                                fragment_plan,
                            ),
                            "query_id": query_id,
                        }
                        registration_ref = self.actor_handle.register_fragments.remote([payload])
                        self._registered_fragment_ids.add(fragment_id)
                        self._fragment_registration_refs[fragment_id] = registration_ref
                        self._fragment_query_ids[fragment_id] = query_id

            if registration_ref is None:
                end_fte_registry_operation(query_id)
                owns_registry_operation = False
                return existing_result

            def finish_registration(*, failed: bool) -> None:
                with self._fragment_registration_lock:
                    if self._fragment_registration_refs.get(fragment_id) is not registration_ref:
                        return
                    self._fragment_registration_refs.pop(fragment_id, None)
                    if failed:
                        self._registered_fragment_ids.discard(fragment_id)
                        self._fragment_query_ids.pop(fragment_id, None)

            transfer_fte_registry_operations_to_ref(
                [query_id],
                registration_ref,
                on_success=lambda: finish_registration(failed=False),
                on_failure=lambda: finish_registration(failed=True),
            )
            owns_registry_operation = False
            return registration_ref
        except BaseException:
            with self._fragment_registration_lock:
                if self._fragment_registration_refs.get(fragment_id) is registration_ref:
                    self._fragment_registration_refs.pop(fragment_id, None)
                    self._registered_fragment_ids.discard(fragment_id)
                    self._fragment_query_ids.pop(fragment_id, None)
            if owns_registry_operation:
                end_fte_registry_operation(query_id)
            raise

    def _task_input_stream_exhausted_direct(
        self,
        source_node_ids: set[str],
        *,
        query_id_filter: str | None = None,
    ) -> list[Any]:
        exhausted_sources = {str(source_node_id) for source_node_id in source_node_ids}
        handles: list[Any] = []
        with _FTE_REGISTRY_LOCK:
            fragment_execution_items = [
                (query_id, fragment_id, fragment_execution, _FTE_FRAGMENT_STATES.get((query_id, fragment_id)))
                for (query_id, fragment_id), fragment_execution in _FTE_FRAGMENT_EXECUTIONS.items()
                if query_id_filter is None or query_id == query_id_filter
            ]
        for query_id, fragment_id, fragment_execution, fragment_state in fragment_execution_items:
            if fte_registry_query_is_closing(query_id):
                continue
            if fragment_state is None or fragment_state.assigner is None:
                continue
            scheduled_attempts = []
            for source_node_id in sorted(exhausted_sources):
                if source_node_id not in fragment_state.source_node_ids:
                    continue
                if source_node_id in fragment_state.exhausted_source_node_ids:
                    continue
                result = fragment_state.assigner.assign(
                    source_node_id,
                    [],
                    no_more_inputs=True,
                )
                fragment_state.exhausted_source_node_ids.add(source_node_id)
                for partition_info in result.partitions_added:
                    partition = fragment_execution.add_partition(
                        partition_info.partition_id,
                        partition_info.node_requirements,
                    )
                    self._try_reserve_fte_partition_for_node_wait(
                        query_id,
                        fragment_id,
                        partition,
                        fragment_execution=fragment_execution,
                    )
                try:
                    scheduled_result = fragment_execution.apply_assignment_result(result)
                    scheduled = self._execute_fte_fragment_execution_mutation_result(
                        fragment_execution, scheduled_result
                    )
                except FteWorkerControlFailure as exc:
                    handles.extend(self._handles_for_fte_worker_control_failure(exc))
                    scheduled = []
                scheduled_attempts.extend(scheduled)
            handles.extend(
                self._handles_for_fte_scheduled_attempts(
                    query_id,
                    fragment_id,
                    scheduled_attempts,
                )
            )
        return handles

    def task_input_stream_exhausted(self, source_node_ids: list[str] | tuple[str, ...]) -> list[Any]:
        exhausted_sources = {str(source_node_id) for source_node_id in source_node_ids}
        self._fte_source_node_ids.update(exhausted_sources)
        query_ids = fte_fragment_execution_query_ids()
        query_ids.update(_FTE_SCHEDULERS.query_ids())
        handles: list[Any] = []
        for query_id in sorted(query_ids):
            if fte_registry_query_is_closing(query_id):
                continue
            scheduler = _FTE_SCHEDULERS.get(query_id)
            if scheduler is None:
                continue
            self._bind_fte_scheduler_handlers(scheduler)
            scheduler.enqueue(SourceInputExhausted.from_source_node_ids(query_id, exhausted_sources))
            handles.extend(scheduler.drain())
        return handles

    def task_input_stream_exhausted_for_query(
        self,
        query_id: str,
        source_node_ids: list[str] | tuple[str, ...],
    ) -> list[Any]:
        query_id = str(query_id or "").strip()
        if not query_id:
            return []
        if fte_registry_query_is_closing(query_id):
            return []
        exhausted_sources = {str(source_node_id) for source_node_id in source_node_ids}
        self._fte_source_node_ids.update(exhausted_sources)
        scheduler = _FTE_SCHEDULERS.get(query_id)
        if scheduler is None:
            return []
        self._bind_fte_scheduler_handlers(scheduler)
        scheduler.enqueue(SourceInputExhausted.from_source_node_ids(query_id, exhausted_sources))
        return scheduler.drain()

    def mark_fte_worker_failed(self, worker_id: str | None = None, error: Any = None) -> list[Any]:
        failed_worker_id = str(worker_id or self.worker_id or "")
        if not failed_worker_id:
            return []
        failed_worker_ids = _expanded_fte_failed_worker_ids(failed_worker_id)
        query_ids = fte_fragment_execution_query_ids()
        query_ids.update(_FTE_SCHEDULERS.query_ids())
        handles: list[Any] = []
        for query_id in sorted(query_ids):
            if fte_registry_query_is_closing(query_id):
                continue
            scheduler = _FTE_SCHEDULERS.get(query_id)
            if scheduler is None:
                continue
            self._bind_fte_scheduler_handlers(scheduler)
            scheduler.enqueue(
                WorkerFailed(
                    query_id,
                    failed_worker_id,
                    error,
                    failed_worker_ids=frozenset(failed_worker_ids),
                )
            )
            handles.extend(scheduler.drain())
        return handles

    def handle_fte_task_status(self, status: Mapping[str, Any]) -> list[Any]:
        attempt_id = FteTaskAttemptId.coerce(status.get("task_id") or status.get("task_id_string") or status)
        validate_fte_status_identity(status, attempt_id)
        query_id = attempt_id.task_id.query_id
        if fte_registry_query_is_closing(query_id):
            return []
        scheduler = _FTE_SCHEDULERS.get(query_id)
        if scheduler is None:
            return []
        self._bind_fte_scheduler_handlers(scheduler)
        scheduler.enqueue(TaskStatusChanged.from_status(query_id, attempt_id, dict(status)))
        return scheduler.drain()

    def fte_attempt_is_selected(self, attempt_id: Any) -> bool:
        attempt = FteTaskAttemptId.coerce(attempt_id)
        with _FTE_REGISTRY_LOCK:
            candidates = [
                fragment_execution
                for (query_id, _), fragment_execution in _FTE_FRAGMENT_EXECUTIONS.items()
                if query_id == attempt.task_id.query_id
                and int(fragment_execution.fragment_execution_id) == int(attempt.task_id.fragment_execution_id)
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    "FTE attempt selection requires exactly one registered fragment execution: "
                    f"attempt={attempt} candidates={len(candidates)}"
                )
            partition = candidates[0].partitions.get(attempt.task_id.partition_id)
            if partition is None:
                raise RuntimeError(f"FTE attempt partition is not registered: {attempt}")
            return bool(partition.finished and partition.selected_attempt == attempt.attempt_id)

    def update_fte_exchange_selector(
        self,
        query_id: str,
        consumer_fragment_id: str,
        source_node_id: str,
        *,
        selector: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        query_id = str(query_id)
        if fte_registry_query_is_closing(query_id):
            return []
        scheduler = _FTE_SCHEDULERS.get(query_id)
        if scheduler is None:
            return []
        self._bind_fte_scheduler_handlers(scheduler)
        scheduler.enqueue(
            ExchangeSelectorUpdated.from_selector(
                query_id,
                consumer_fragment_id,
                source_node_id,
                selector=selector,
            )
        )
        return scheduler.drain()

    def fte_query_status(self, query_id: str) -> dict[str, Any]:
        return fte_query_status(query_id)

    def wait_fte_query(self, query_id: str, timeout_s: float = 0.0) -> dict[str, Any]:
        query_id = str(query_id or "").strip()
        if not query_id:
            raise ValueError("query_id must be non-empty")
        has_deadline = float(timeout_s) > 0.0
        deadline = time.monotonic() + float(timeout_s) if has_deadline else None
        while True:
            status = fte_query_status(query_id)
            if bool(status.get("failed")):
                raise RuntimeError(f"FTE query {query_id} failed: {status}")
            if bool(status.get("finished")):
                return status
            if has_deadline and time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for FTE query {query_id}: {status}")
            time.sleep(0.01)

    def pop_fte_result_handles(self, query_id: str) -> list[Any]:
        return pop_fte_result_handles(query_id)

    def fte_registry_stats(self) -> dict[str, Any]:
        return fte_registry_stats()

    def _drop_fte_state_for_query(self, query_id: str) -> None:
        query_id = str(query_id or "").strip()
        if not query_id:
            return
        for key in list(self._fte_fragment_execution_ids):
            if key[0] == query_id:
                self._fte_fragment_execution_ids.pop(key, None)
        for key in list(self._fte_fragment_executions):
            if key[0] == query_id:
                self._fte_fragment_executions.pop(key, None)
        for key in list(self._fte_sequences):
            if key[0] == query_id:
                self._fte_sequences.pop(key, None)
        self._fte_pressure.drop_query(query_id)

    def shutdown(self) -> None:
        if self.worker_id:
            try:
                self.mark_fte_worker_failed(
                    self.worker_id,
                    f"FTE worker shutdown: {self.worker_id}",
                )
            except Exception:
                pass
            with _FTE_REGISTRY_LOCK:
                current = _FTE_WORKER_HANDLES.get(str(self.worker_id))
                if current is self:
                    _FTE_WORKER_HANDLES.pop(str(self.worker_id), None)
        ray.kill(self.actor_handle)
