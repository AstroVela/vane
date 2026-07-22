# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vane.runners.fte.debug_memory import (
    debug_flag_enabled,
    describe_result_payload,
    log_debug,
    process_memory_snapshot,
)
from vane.runners.fte.fte_config import (
    FTE_WORKER_RUNTIME,
    FteWorkerAdmissionConfig,
    fte_split_queue_max_buffered_splits,
    fte_status_wait_timeout_s,
)
from vane.runners.fte.fte_descriptor import (
    FteTaskUpdateRequest,
    _merge_output_buffers,
    _normalize_output_buffers,
    _output_buffer_status,
    normalize_initial_splits,
)
from vane.runners.fte.fte_exchange import collect_spooling_output_stats
from vane.runners.fte.fte_state import _TERMINAL_STATES, FteTaskState
from vane.runners.fte.fte_types import FteSplit, FteTaskAttemptId

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


def _memory_requirement_from_request(
    request: Mapping[str, Any],
    default_bytes: int | None,
) -> int:
    for key in ("memory_requirement_bytes", "required_memory_bytes"):
        value = request.get(key)
        if value is not None:
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid FTE memory requirement {value!r}") from exc
            if parsed <= 0:
                raise ValueError("FTE memory requirement must be positive")
            return parsed
    resource_request = request.get("resource_request")
    if isinstance(resource_request, Mapping):
        for key in ("memory_requirement_bytes", "required_memory_bytes", "memory_bytes"):
            value = resource_request.get(key)
            if value is not None:
                try:
                    parsed = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid FTE memory requirement {value!r}") from exc
                if parsed <= 0:
                    raise ValueError("FTE memory requirement must be positive")
                return parsed
    if default_bytes is None or int(default_bytes) <= 0:
        raise RuntimeError("FTE task requires a positive memory requirement")
    return int(default_bytes)


def _has_explicit_memory_requirement(request: Mapping[str, Any]) -> bool:
    if any(request.get(key) is not None for key in ("memory_requirement_bytes", "required_memory_bytes")):
        return True
    resource_request = request.get("resource_request")
    return isinstance(resource_request, Mapping) and any(
        resource_request.get(key) is not None
        for key in ("memory_requirement_bytes", "required_memory_bytes", "memory_bytes")
    )


def _query_task_lease_heap_bytes(
    request: Mapping[str, Any],
    *,
    required: bool,
) -> int | None:
    lease = request.get("query_task_lease")
    if not isinstance(lease, Mapping):
        if required:
            raise RuntimeError("Ray FTE task requires query_task_lease")
        return None
    task_id = FteTaskAttemptId.coerce(request.get("task_id"))
    required_identity = {
        "lease_id": str(lease.get("lease_id") or "").strip(),
        "query_id": str(lease.get("query_id") or "").strip(),
        "execution_query_id": str(lease.get("execution_query_id") or "").strip(),
        "stage_id": str(lease.get("stage_id") or "").strip(),
        "attempt_id": str(lease.get("attempt_id") or "").strip(),
    }
    missing = [name for name, value in required_identity.items() if not value]
    if missing:
        raise RuntimeError("Ray FTE query_task_lease is missing " + ", ".join(missing))
    if required_identity["execution_query_id"] != task_id.query_id:
        raise RuntimeError("Ray FTE query_task_lease execution_query_id does not match task")
    if required_identity["attempt_id"] != str(task_id):
        raise RuntimeError("Ray FTE query_task_lease attempt_id does not match task")
    resources = lease.get("resources")
    if not isinstance(resources, Mapping):
        raise RuntimeError("Ray FTE query_task_lease resources must be a mapping")
    heap_bytes = int(resources.get("heap_bytes") or 0)
    if heap_bytes <= 0:
        raise RuntimeError("Ray FTE query_task_lease heap_bytes must be positive")
    explicit = _memory_requirement_from_request(request, None) if _has_explicit_memory_requirement(request) else None
    if explicit is not None and int(explicit) != heap_bytes:
        raise RuntimeError(
            f"Ray FTE memory requirement diverges from query task lease: request={explicit} lease={heap_bytes}"
        )
    return heap_bytes


def fte_split_queue_space_wait_timeout_s() -> float:
    raw = os.getenv("VANE_FTE_SPLIT_QUEUE_SPACE_WAIT_TIMEOUT_S", "1.0")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, value)


def fte_split_queue_space_poll_interval_s() -> float:
    raw = os.getenv("VANE_FTE_SPLIT_QUEUE_SPACE_POLL_INTERVAL_S", "0.01")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.01
    return min(max(0.001, value), 0.1)


def _failure_payload(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def _fte_admission_debug_enabled() -> bool:
    for name in ("VANE_FTE_ADMISSION_DEBUG", "DUCKDB_DISTRIBUTED_DEBUG"):
        value = os.getenv(name, "")
        if value.strip().lower() not in ("", "0", "false", "no", "off"):
            return True
    return False


def _fte_result_debug_enabled() -> bool:
    return debug_flag_enabled("VANE_FTE_RESULT_DEBUG", "VANE_FTE_ADMISSION_DEBUG", "DUCKDB_DISTRIBUTED_DEBUG")


def _fte_result_debug_log(event: str, **fields: Any) -> None:
    if not _fte_result_debug_enabled():
        return
    memory_fields = process_memory_snapshot()
    memory_fields.update(fields)
    log_debug("vane-fte-result", event, **memory_fields)


def _format_admission_field(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace(" ", "_")


@dataclass
class TaskStatus:
    task_id: FteTaskAttemptId
    state: FteTaskState
    version: int = 0
    failure: dict[str, str] | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    duplicate_split_count: int = 0
    output_stats: Any = None
    task_stats: dict[str, Any] = field(default_factory=dict)
    task_stats_updated_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id.to_dict(),
            "task_id_string": str(self.task_id),
            "state": self.state.value,
            "version": self.version,
            "failure": self.failure,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "duplicate_split_count": self.duplicate_split_count,
            "spooling_output_stats": self.output_stats,
        }
        if self.task_stats:
            payload["task_stats"] = dict(self.task_stats)
            payload["task_stats_updated_at"] = self.task_stats_updated_at
        return payload


def materialize_task_inputs(
    context: Mapping[str, Any] | None,
    initial_splits: Mapping[str, list[FteSplit]] | None,
    *,
    merge_scan_task_descriptors: Callable[[list[Any]], Any] | None = None,
) -> dict[str, Any]:
    materialized_context = dict(context or {})
    scan_task_nodes: list[str] = []
    exchange_source_nodes: list[str] = []

    for source_node_id, raw_splits in (initial_splits or {}).items():
        splits = [
            split if isinstance(split, FteSplit) else FteSplit.from_dict(str(source_node_id), split)
            for split in raw_splits
        ]
        scan_values: list[Any] = []
        exchange_values: list[Any] = []
        for split in splits:
            if split.kind == "scan_task":
                scan_values.append(split.data)
            elif split.kind == "exchange_source_task":
                exchange_values.append(split.data)
            else:
                raise ValueError(f"Unsupported FTE split kind: {split.kind}")

        if scan_values:
            scan_task_nodes.append(str(source_node_id))
            if len(scan_values) == 1:
                materialized_context[f"scan_task:{source_node_id}"] = scan_values[0]
            else:
                if merge_scan_task_descriptors is None:
                    raise ValueError("multiple scan_task splits require a merge function")
                materialized_context[f"scan_task:{source_node_id}"] = merge_scan_task_descriptors(scan_values)

        if exchange_values:
            if len(exchange_values) != 1:
                raise ValueError("multiple exchange_source_task splits for one source are not supported")
            exchange_source_nodes.append(str(source_node_id))
            materialized_context[f"exchange_source_task:{source_node_id}"] = exchange_values[0]

    if scan_task_nodes:
        materialized_context["scan_task_nodes"] = ",".join(scan_task_nodes)
    if exchange_source_nodes:
        materialized_context["exchange_source_task_nodes"] = ",".join(exchange_source_nodes)
    return materialized_context


class FteTaskExecution:
    def __init__(
        self,
        request: Mapping[str, Any],
        execute_fn: Callable[[Mapping[str, Any]], Awaitable[Any]],
        *,
        status_callback: Callable[["FteTaskExecution"], None] | None = None,
        require_query_task_lease: bool = False,
        default_task_memory_bytes: int | None = None,
    ) -> None:
        self.request = dict(request)
        self.task_id = FteTaskAttemptId.coerce(self.request["task_id"])
        self.execute_fn = execute_fn
        self._status_callback = status_callback
        self.initial_splits = normalize_initial_splits(self.request.get("initial_splits"))
        self.no_more_split_sources = {str(source) for source in self.request.get("no_more_splits") or []}
        self.output_buffers = self._initial_output_buffers(self.request)
        self.dynamic_filter_domains = self._initial_dynamic_filter_domains(self.request)
        self.descriptor_version = int(self.request.get("descriptor_version") or 0)
        if self.output_buffers is not None:
            self.request["output_buffers"] = dict(self.output_buffers)
        if self.dynamic_filter_domains:
            self.request["dynamic_filter_domains"] = dict(self.dynamic_filter_domains)
        self.fte_runtime = self.request.get("worker_runtime") == FTE_WORKER_RUNTIME
        self.dynamic_exchange_source_ids = {
            str(source)
            for source in (
                self.request.get("dynamic_exchange_source_node_ids")
                or self.request.get("exchange_source_node_ids")
                or []
            )
        }
        self.dynamic_scan_source_ids = {
            str(source)
            for source in (
                self.request.get("dynamic_scan_source_node_ids") or self.request.get("scan_source_node_ids") or []
            )
        }
        self.dynamic_scan_source_queues = self._create_dynamic_source_queues("scan_task", self.dynamic_scan_source_ids)
        self.dynamic_exchange_source_queues = self._create_dynamic_source_queues(
            "exchange_source_task",
            self.dynamic_exchange_source_ids,
        )
        self.seen_sequences: dict[str, set[int]] = {
            source_id: {split.sequence_id for split in splits} for source_id, splits in self.initial_splits.items()
        }
        self.status = TaskStatus(self.task_id, FteTaskState.PLANNED)
        self.result: Any = None
        self._result_stored_at: float | None = None
        self._result_release_count = 0
        self._status_lock = threading.RLock()
        self._future: asyncio.Task[Any] | None = None
        self._split_update_event = asyncio.Event()
        self._status_condition = asyncio.Condition()
        self._execution_started = False
        self.split_queue_max_buffered_splits = int(
            self.request.get("split_queue_max_buffered_splits") or fte_split_queue_max_buffered_splits()
        )
        leased_memory_bytes = _query_task_lease_heap_bytes(
            self.request,
            required=require_query_task_lease,
        )
        self.memory_requirement_bytes = (
            leased_memory_bytes
            if leased_memory_bytes is not None
            else _memory_requirement_from_request(self.request, default_task_memory_bytes)
        )

    def _create_dynamic_source_queues(self, expected_kind: str, source_ids: set[str]) -> dict[str, Any]:
        if not source_ids:
            return {}
        import vane

        queues = {source_id: vane.ray_cxx.FteSplitQueue() for source_id in source_ids}
        for source_id, splits in self.initial_splits.items():
            queue = queues.get(source_id)
            if queue is None:
                continue
            for split in splits:
                self._add_split_to_dynamic_queue(queue, split, expected_kind)
            if source_id in self.no_more_split_sources:
                queue.no_more_splits()
        return queues

    @staticmethod
    def _add_split_to_dynamic_queue(queue: Any, split: FteSplit, expected_kind: str) -> None:
        if split.kind != expected_kind:
            raise ValueError(f"dynamic {expected_kind} queues only accept {expected_kind} splits")
        data = split.data
        if isinstance(data, str):
            data = data.encode()
        if expected_kind == "scan_task":
            queue.add_scan_split(data)
        elif expected_kind == "exchange_source_task":
            queue.add_exchange_source_split(data)
        else:
            raise ValueError(f"unsupported dynamic split kind: {expected_kind}")

    def _transition(
        self,
        state: FteTaskState,
        *,
        failure: dict[str, str] | None = None,
    ) -> None:
        with self._status_lock:
            if self.status.state in _TERMINAL_STATES:
                return
            self.status.state = state
            self.status.failure = failure
            self.status.version += 1
            self.status.updated_at = time.time()
        self._publish_status_changed()
        self._notify_status_changed()

    def _publish_status_changed(self) -> None:
        callback = self._status_callback
        if callback is None:
            return
        try:
            callback(self)
        except Exception:
            pass

    def _notify_status_changed(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._notify_status_waiters())

    async def _notify_status_waiters(self) -> None:
        async with self._status_condition:
            self._status_condition.notify_all()

    def status_payload(self) -> dict[str, Any]:
        with self._status_lock:
            status = self.status.to_dict()
            status.update(self.split_queue_status())
            status["memory_requirement_bytes"] = self.memory_requirement_bytes
            output_buffer_status = _output_buffer_status(self.output_buffers)
            if output_buffer_status is not None:
                status["output_buffer_status"] = output_buffer_status
            if self.status.state == FteTaskState.FINISHED:
                status["result"] = self.result
        return status

    def split_queue_status(self) -> dict[str, Any]:
        queues_by_source = {
            **self.dynamic_scan_source_queues,
            **self.dynamic_exchange_source_queues,
        }
        submitted_by_source = {
            source_id: int(queues_by_source[source_id].submitted_splits())
            for source_id in sorted(self.initial_splits)
            if source_id in queues_by_source
        }
        buffered_by_source = {
            source_id: int(queue.buffered_splits()) for source_id, queue in sorted(queues_by_source.items())
        }
        buffered_bytes_by_source = {
            source_id: int(queue.buffered_bytes()) for source_id, queue in sorted(queues_by_source.items())
        }
        consumed_by_source = {
            source_id: int(queues_by_source[source_id].consumed_splits()) for source_id in sorted(submitted_by_source)
        }
        completed_by_source = {
            source_id: int(queue.completed_splits()) for source_id, queue in sorted(queues_by_source.items())
        }
        submitted_rows_by_source = {
            source_id: int(queue.submitted_rows()) for source_id, queue in sorted(queues_by_source.items())
        }
        submitted_bytes_by_source = {
            source_id: int(queue.submitted_input_bytes()) for source_id, queue in sorted(queues_by_source.items())
        }
        consumed_rows_by_source = {
            source_id: int(queue.consumed_rows()) for source_id, queue in sorted(queues_by_source.items())
        }
        consumed_bytes_by_source = {
            source_id: int(queue.consumed_input_bytes()) for source_id, queue in sorted(queues_by_source.items())
        }
        completed_rows_by_source = {
            source_id: int(queue.completed_rows()) for source_id, queue in sorted(queues_by_source.items())
        }
        completed_bytes_by_source = {
            source_id: int(queue.completed_input_bytes()) for source_id, queue in sorted(queues_by_source.items())
        }
        queue_wait_ms_by_source = {
            source_id: int(queue.queue_wait_ms())
            for source_id, queue in sorted(queues_by_source.items())
            if hasattr(queue, "queue_wait_ms")
        }
        submitted_count = sum(submitted_by_source.values())
        queued_count = sum(buffered_by_source.values())
        consumed_count = sum(consumed_by_source.values())
        completed_count = sum(completed_by_source.values())
        queued_bytes = sum(buffered_bytes_by_source.values())
        submitted_bytes = sum(submitted_bytes_by_source.values())
        consumed_bytes = sum(consumed_bytes_by_source.values())
        completed_bytes = sum(completed_bytes_by_source.values())
        return {
            "submitted_split_count": submitted_count,
            "submitted_split_count_by_source": submitted_by_source,
            "queued_split_count": queued_count,
            "unacknowledged_split_count": queued_count,
            "queued_split_count_by_source": buffered_by_source,
            "consumed_split_count": consumed_count,
            "consumed_split_count_by_source": consumed_by_source,
            "completed_split_count": completed_count,
            "completed_split_count_by_source": completed_by_source,
            "submitted_split_bytes": submitted_bytes,
            "submitted_split_bytes_by_source": submitted_bytes_by_source,
            "queued_split_bytes": queued_bytes,
            "queued_split_bytes_by_source": buffered_bytes_by_source,
            "consumed_split_bytes": consumed_bytes,
            "consumed_split_bytes_by_source": consumed_bytes_by_source,
            "completed_split_bytes": completed_bytes,
            "completed_split_bytes_by_source": completed_bytes_by_source,
            "queue_wait_ms": sum(queue_wait_ms_by_source.values()),
            "queue_wait_ms_by_source": queue_wait_ms_by_source,
            "submitted_input_rows": sum(submitted_rows_by_source.values()),
            "submitted_input_rows_by_source": submitted_rows_by_source,
            "submitted_input_bytes": submitted_bytes,
            "submitted_input_bytes_by_source": submitted_bytes_by_source,
            "consumed_input_rows": sum(consumed_rows_by_source.values()),
            "consumed_input_rows_by_source": consumed_rows_by_source,
            "consumed_input_bytes": consumed_bytes,
            "consumed_input_bytes_by_source": consumed_bytes_by_source,
            "completed_input_rows": sum(completed_rows_by_source.values()),
            "completed_input_rows_by_source": completed_rows_by_source,
            "completed_input_bytes": completed_bytes,
            "completed_input_bytes_by_source": completed_bytes_by_source,
            "queued_split_weight": queued_count,
            "queued_split_weight_by_source": dict(buffered_by_source),
            "split_queue_max_buffered_splits": self.split_queue_max_buffered_splits,
            "split_queue_has_space": queued_count < self.split_queue_max_buffered_splits,
        }

    async def wait_status_after(self, min_version: int | None, timeout_s: float | None) -> dict[str, Any]:
        min_version = -1 if min_version is None else int(min_version)
        timeout_s = fte_status_wait_timeout_s() if timeout_s is None else max(0.0, float(timeout_s))
        deadline = time.monotonic() + timeout_s
        async with self._status_condition:
            while True:
                with self._status_lock:
                    should_wait = self.status.version <= min_version and self.status.state not in _TERMINAL_STATES
                if not should_wait:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(self._status_condition.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
            return self.status_payload()

    def _source_buffered_splits(self, source_node_id: str | None = None) -> int:
        queues_by_source = {
            **self.dynamic_scan_source_queues,
            **self.dynamic_exchange_source_queues,
        }
        if source_node_id is not None:
            queue = queues_by_source.get(str(source_node_id))
            return 0 if queue is None else int(queue.buffered_splits())
        return sum(int(queue.buffered_splits()) for queue in queues_by_source.values())

    def _source_buffered_bytes(self, source_node_id: str | None = None) -> int:
        queues_by_source = {
            **self.dynamic_scan_source_queues,
            **self.dynamic_exchange_source_queues,
        }

        if source_node_id is not None:
            queue = queues_by_source.get(str(source_node_id))
            return 0 if queue is None else int(queue.buffered_bytes())
        return sum(int(queue.buffered_bytes()) for queue in queues_by_source.values())

    async def wait_split_queue_has_space(
        self,
        source_node_id: str | None = None,
        max_buffered_splits: int | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        max_buffered_splits = (
            self.split_queue_max_buffered_splits if max_buffered_splits is None else max(1, int(max_buffered_splits))
        )
        timeout_s = fte_split_queue_space_wait_timeout_s() if timeout_s is None else max(0.0, float(timeout_s))
        deadline = time.monotonic() + timeout_s
        while True:
            buffered = self._source_buffered_splits(source_node_id)
            buffered_bytes = self._source_buffered_bytes(source_node_id)
            has_space = buffered < max_buffered_splits or self.status.state in _TERMINAL_STATES
            if has_space:
                return {
                    "has_space": True,
                    "buffered_splits": buffered,
                    "buffered_bytes": buffered_bytes,
                    "max_buffered_splits": max_buffered_splits,
                    "status": self.status_payload(),
                }
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {
                    "has_space": False,
                    "buffered_splits": buffered,
                    "buffered_bytes": buffered_bytes,
                    "max_buffered_splits": max_buffered_splits,
                    "status": self.status_payload(),
                }
            await asyncio.sleep(min(fte_split_queue_space_poll_interval_s(), remaining))

    def start(self) -> None:
        if self._future is not None:
            return
        self._transition(FteTaskState.RUNNING)
        self._future = asyncio.create_task(self._run())

    def queue(self) -> None:
        if self._future is not None or self.status.state in _TERMINAL_STATES:
            return
        self._transition(FteTaskState.QUEUED)

    def add_done_callback(self, callback: Callable[[asyncio.Task[Any]], None]) -> None:
        if self._future is not None:
            self._future.add_done_callback(callback)

    async def _run(self) -> None:
        try:
            if self.fte_runtime:
                await self._wait_for_fte_splits_sealed()
                self._refresh_request_splits()
            if self.dynamic_exchange_source_queues:
                self.request["fte_exchange_source_queues"] = self.dynamic_exchange_source_queues
                self.request["dynamic_exchange_source_node_ids"] = sorted(self.dynamic_exchange_source_ids)
            if self.dynamic_scan_source_queues:
                self.request["fte_scan_source_queues"] = self.dynamic_scan_source_queues
                self.request["dynamic_scan_source_node_ids"] = sorted(self.dynamic_scan_source_ids)
            self._execution_started = True
            self.request["native_progress_callback"] = self._record_task_stats
            result = await self.execute_fn(self.request)
            with self._status_lock:
                self.result = result
                self._result_stored_at = time.time()
            self._log_result_lifecycle("result_stored", result=result)
        except asyncio.CancelledError:
            self._transition(FteTaskState.CANCELED)
            raise
        except Exception as exc:
            self._transition(FteTaskState.FAILED, failure=_failure_payload(exc))
        else:
            result = self.result
            with self._status_lock:
                self.status.output_stats = self._extract_output_stats(result)
            self._complete_dynamic_source_splits()
            final_task_stats = self._extract_task_stats(result)
            final_split_stats = self.split_queue_status()
            if final_task_stats or final_split_stats:
                self._record_task_stats({**final_task_stats, **final_split_stats})
            self._transition(FteTaskState.FINISHED)

    def _complete_dynamic_source_splits(self) -> None:
        queues = [
            *self.dynamic_scan_source_queues.values(),
            *self.dynamic_exchange_source_queues.values(),
        ]
        for queue in queues:
            try:
                queue.complete_consumed_splits()
            except Exception:
                pass

    def _record_task_stats(self, stats: Any) -> None:
        if not isinstance(stats, Mapping):
            return
        normalized = dict(stats)
        if not normalized:
            return
        with self._status_lock:
            self.status.task_stats = normalized
            self.status.task_stats_updated_at = time.time()
        self._publish_status_changed()

    def _extract_output_stats(self, result: Any) -> Any:
        if isinstance(result, Mapping):
            raw_stats = result.get("spooling_output_stats") or result.get("output_stats")
            if raw_stats is not None:
                return dict(raw_stats) if isinstance(raw_stats, Mapping) else raw_stats
        return collect_spooling_output_stats(self.request.get("exchange_sink_instance"))

    def _extract_task_stats(self, result: Any) -> dict[str, Any]:
        raw_result = result.get("result") if isinstance(result, Mapping) and "result" in result else result
        if isinstance(result, Mapping) and isinstance(result.get("task_stats"), Mapping):
            return dict(result["task_stats"])
        raw_task_stats = getattr(raw_result, "task_stats", None)
        if isinstance(raw_task_stats, Mapping):
            return dict(raw_task_stats)
        if isinstance(raw_result, (tuple, list)) and len(raw_result) >= 7 and isinstance(raw_result[6], Mapping):
            return dict(raw_result[6])
        rows = 0
        bytes_value = 0

        def add_metadata_values(metadata_items: Any) -> None:
            nonlocal rows, bytes_value
            for metadata in metadata_items or []:
                if isinstance(metadata, Mapping):
                    rows += int(metadata.get("num_rows") or metadata.get("rows") or 0)
                    bytes_value += int(metadata.get("size_bytes") or metadata.get("bytes") or 0)
                elif isinstance(metadata, (tuple, list)) and len(metadata) >= 2:
                    rows += int(metadata[0] or 0)
                    bytes_value += int(metadata[1] or 0)

        partition_metadatas = getattr(raw_result, "partition_metadatas", None)
        if partition_metadatas is not None:
            add_metadata_values(partition_metadatas)
        elif isinstance(raw_result, (tuple, list)) and len(raw_result) >= 2:
            add_metadata_values(raw_result[1])

        output_stats = self._extract_output_stats(result)
        if isinstance(output_stats, Mapping):
            rows = max(rows, int(output_stats.get("rows") or output_stats.get("total_rows") or 0))
            bytes_value = max(
                bytes_value,
                int(
                    output_stats.get("total_bytes")
                    or output_stats.get("bytes")
                    or output_stats.get("output_bytes")
                    or 0
                ),
            )

        stats: dict[str, Any] = {}
        if rows:
            stats["output_rows"] = rows
        if bytes_value:
            stats["output_bytes"] = bytes_value
        return stats

    async def _wait_for_fte_splits_sealed(self) -> None:
        while True:
            self._split_update_event.clear()
            if self._all_fte_sources_sealed():
                return
            await self._split_update_event.wait()

    def _all_fte_sources_sealed(self) -> bool:
        explicit_sources = self.request.get("split_sources") or self.request.get("source_node_ids") or []
        required_sources = {str(source) for source in explicit_sources}
        if not required_sources:
            required_sources = set(self.initial_splits) | set(self.seen_sequences)
            required_sources.difference_update(self.dynamic_scan_source_ids)
            required_sources.difference_update(self.dynamic_exchange_source_ids)
        if not required_sources:
            return True
        return required_sources.issubset(self.no_more_split_sources)

    def _refresh_request_splits(self) -> None:
        dynamic_sources = self.dynamic_scan_source_ids | self.dynamic_exchange_source_ids
        self.request["initial_splits"] = {
            source_id: [split.to_dict() for split in splits]
            for source_id, splits in self.initial_splits.items()
            if source_id not in dynamic_sources
        }
        self.request["no_more_splits"] = sorted(self.no_more_split_sources)

    def add_splits(self, source_node_id: str, splits: list[Mapping[str, Any]]) -> TaskStatus:
        with self._status_lock:
            if self.status.state in _TERMINAL_STATES:
                raise RuntimeError(f"cannot add splits to terminal task {self.task_id}")
            source_node_id = str(source_node_id)
            if source_node_id in self.no_more_split_sources:
                raise RuntimeError(f"source {source_node_id} is already marked no_more_splits")
            seen = self.seen_sequences.setdefault(source_node_id, set())
            target = self.initial_splits.setdefault(source_node_id, [])
            added = False
            for payload in splits:
                split = FteSplit.from_dict(source_node_id, payload)
                if split.sequence_id in seen:
                    self.status.duplicate_split_count += 1
                    continue
                seen.add(split.sequence_id)
                target.append(split)
                scan_queue = self.dynamic_scan_source_queues.get(source_node_id)
                if scan_queue is not None:
                    self._add_split_to_dynamic_queue(scan_queue, split, "scan_task")
                exchange_queue = self.dynamic_exchange_source_queues.get(source_node_id)
                if exchange_queue is not None:
                    self._add_split_to_dynamic_queue(exchange_queue, split, "exchange_source_task")
                added = True
            if added:
                self.status.version += 1
                self.status.updated_at = time.time()
        if added:
            self._split_update_event.set()
            self._publish_status_changed()
            self._notify_status_changed()
        return self.status

    @staticmethod
    def _initial_output_buffers(request: Mapping[str, Any]) -> dict[str, Any] | None:
        for key in ("output_buffers", "outputIds", "output_buffer_update"):
            if key in request:
                return _normalize_output_buffers(request.get(key) or {})
        return None

    @staticmethod
    def _initial_dynamic_filter_domains(request: Mapping[str, Any]) -> dict[str, Any]:
        for key in ("dynamic_filter_domains", "dynamicFilterDomains", "dynamic_filters"):
            if key in request:
                return dict(request.get(key) or {})
        return {}

    @staticmethod
    def _has_descriptor_fields(update: FteTaskUpdateRequest) -> bool:
        return bool(
            update.context
            or update.resource_request
            or update.fragment_plan_present
            or update.source_node_ids
            or update.dynamic_scan_source_node_ids
            or update.dynamic_exchange_source_node_ids
        )

    def _add_dynamic_source_queues(self, expected_kind: str, source_ids: set[str]) -> None:
        if expected_kind == "scan_task":
            missing = set(source_ids) - set(self.dynamic_scan_source_queues)
            self.dynamic_scan_source_queues.update(self._create_dynamic_source_queues(expected_kind, missing))
            return
        if expected_kind == "exchange_source_task":
            missing = set(source_ids) - set(self.dynamic_exchange_source_queues)
            self.dynamic_exchange_source_queues.update(self._create_dynamic_source_queues(expected_kind, missing))
            return
        raise ValueError(f"unsupported dynamic split kind: {expected_kind}")

    def _append_update_splits(self, source_node_id: str, splits: list[FteSplit]) -> bool:
        if source_node_id in self.no_more_split_sources and splits:
            raise RuntimeError(f"source {source_node_id} is already marked no_more_splits")
        seen = self.seen_sequences.setdefault(source_node_id, set())
        target = self.initial_splits.setdefault(source_node_id, [])
        added = False
        for split in splits:
            if split.sequence_id in seen:
                self.status.duplicate_split_count += 1
                continue
            seen.add(split.sequence_id)
            target.append(split)
            scan_queue = self.dynamic_scan_source_queues.get(source_node_id)
            if scan_queue is not None:
                self._add_split_to_dynamic_queue(scan_queue, split, "scan_task")
            exchange_queue = self.dynamic_exchange_source_queues.get(source_node_id)
            if exchange_queue is not None:
                self._add_split_to_dynamic_queue(exchange_queue, split, "exchange_source_task")
            added = True
        return added

    def update_task(self, update: FteTaskUpdateRequest | Mapping[str, Any] | None) -> TaskStatus:
        with self._status_lock:
            is_terminal = self.status.state in _TERMINAL_STATES
        if is_terminal:
            raise RuntimeError(f"cannot update terminal task {self.task_id}")
        update_request = FteTaskUpdateRequest.coerce(update)
        if self._execution_started and self._has_descriptor_fields(update_request):
            raise RuntimeError(f"cannot update descriptor fields after task execution started: {self.task_id}")
        changed = False

        if update_request.context:
            self.request["context"] = {
                **dict(self.request.get("context") or {}),
                **dict(update_request.context),
            }
            changed = True
        if update_request.resource_request:
            self.request["resource_request"] = {
                **dict(self.request.get("resource_request") or {}),
                **dict(update_request.resource_request),
            }
            changed = True
        if update_request.fragment_plan_present:
            self.request["fragment_plan"] = update_request.fragment_plan
            changed = True

        if update_request.source_node_ids:
            current = {str(source) for source in (self.request.get("source_node_ids") or [])}
            merged = current | set(update_request.source_node_ids)
            if merged != current:
                self.request["source_node_ids"] = sorted(merged)
                changed = True
        if update_request.dynamic_scan_source_node_ids:
            merged = self.dynamic_scan_source_ids | set(update_request.dynamic_scan_source_node_ids)
            if merged != self.dynamic_scan_source_ids:
                self.dynamic_scan_source_ids = merged
                self._add_dynamic_source_queues("scan_task", merged)
                self.request["dynamic_scan_source_node_ids"] = sorted(merged)
                changed = True
        if update_request.dynamic_exchange_source_node_ids:
            merged = self.dynamic_exchange_source_ids | set(update_request.dynamic_exchange_source_node_ids)
            if merged != self.dynamic_exchange_source_ids:
                self.dynamic_exchange_source_ids = merged
                self._add_dynamic_source_queues("exchange_source_task", merged)
                self.request["dynamic_exchange_source_node_ids"] = sorted(merged)
                changed = True

        for source_id, splits in update_request.initial_splits.items():
            changed = self._append_update_splits(source_id, list(splits)) or changed
        for source_id in sorted(update_request.no_more_splits):
            if source_id not in self.no_more_split_sources:
                self.no_more_split_sources.add(source_id)
                queue = self.dynamic_exchange_source_queues.get(source_id)
                if queue is not None:
                    queue.no_more_splits()
                queue = self.dynamic_scan_source_queues.get(source_id)
                if queue is not None:
                    queue.no_more_splits()
                changed = True

        if update_request.output_buffers is not None:
            output_buffers, output_buffers_changed = _merge_output_buffers(
                self.output_buffers,
                update_request.output_buffers,
            )
            if output_buffers_changed:
                self.output_buffers = output_buffers
                self.request["output_buffers"] = dict(output_buffers or {})
                changed = True
        if update_request.dynamic_filter_domains:
            merged = dict(self.dynamic_filter_domains)
            merged.update(update_request.dynamic_filter_domains)
            if merged != self.dynamic_filter_domains:
                self.dynamic_filter_domains = merged
                self.request["dynamic_filter_domains"] = dict(merged)
                changed = True

        if changed:
            with self._status_lock:
                self.descriptor_version += 1
                self.request["descriptor_version"] = self.descriptor_version
                self.status.version += 1
                self.status.updated_at = time.time()
            self._split_update_event.set()
            self._publish_status_changed()
            self._notify_status_changed()
        return self.status

    def no_more_splits(self, source_node_id: str) -> TaskStatus:
        with self._status_lock:
            if self.status.state in _TERMINAL_STATES:
                raise RuntimeError(f"cannot update terminal task {self.task_id}")
            source_node_id = str(source_node_id)
            changed = source_node_id not in self.no_more_split_sources
            if changed:
                self.no_more_split_sources.add(source_node_id)
                queue = self.dynamic_exchange_source_queues.get(source_node_id)
                if queue is not None:
                    queue.no_more_splits()
                queue = self.dynamic_scan_source_queues.get(source_node_id)
                if queue is not None:
                    queue.no_more_splits()
                self.status.version += 1
                self.status.updated_at = time.time()
        if changed:
            self._split_update_event.set()
            self._publish_status_changed()
            self._notify_status_changed()
        return self.status

    def cancel(self) -> TaskStatus:
        for queue in self.dynamic_exchange_source_queues.values():
            queue.cancel()
        for queue in self.dynamic_scan_source_queues.values():
            queue.cancel()
        if self._future is not None and not self._future.done():
            self._future.cancel()
        self._split_update_event.set()
        self._transition(FteTaskState.CANCELED)
        return self.status

    def _log_result_lifecycle(self, event: str, *, result: Any = None, **fields: Any) -> None:
        if not _fte_result_debug_enabled():
            return
        payload = {
            "task_id": str(self.task_id),
            "query_id": self.task_id.query_id,
            "fragment_execution_id": self.task_id.fragment_execution_id,
            "partition_id": self.task_id.partition_id,
            "attempt_id": self.task_id.attempt_id,
            "state": self.status.state.value,
        }
        payload.update(fields)
        if result is not None:
            payload.update(describe_result_payload(result))
        _fte_result_debug_log(event, **payload)

    def release_result(self, *, reason: str = "release_result") -> None:
        with self._status_lock:
            result = self.result
            stored_at = self._result_stored_at
            had_result = result is not None
            self.result = None
            self._result_stored_at = None
            self._result_release_count += 1
            release_count = self._result_release_count
        held_ms = None
        if stored_at is not None:
            held_ms = int((time.time() - stored_at) * 1000)
        self._log_result_lifecycle(
            "result_released",
            result=result,
            reason=reason,
            had_result=had_result,
            held_ms=held_ms,
            release_count=release_count,
        )
        self._publish_status_changed()

    def info(self) -> dict[str, Any]:
        with self._status_lock:
            return {
                "status": self.status.to_dict(),
                "initial_split_counts": {source_id: len(splits) for source_id, splits in self.initial_splits.items()},
                "no_more_splits": sorted(self.no_more_split_sources),
                "output_buffers": None if self.output_buffers is None else dict(self.output_buffers),
                "output_buffer_status": _output_buffer_status(self.output_buffers),
                "dynamic_filter_domains": dict(self.dynamic_filter_domains),
                "descriptor_version": self.descriptor_version,
                "result": self.result if self.status.state == FteTaskState.FINISHED else None,
                "spooling_output_stats": self.status.output_stats,
            }


class FteWorkerTaskManager:
    def __init__(
        self,
        execute_fn: Callable[[Mapping[str, Any]], Awaitable[Any]],
        *,
        admission_config: FteWorkerAdmissionConfig,
        require_query_task_lease: bool = False,
        worker_label: str | None = None,
        sync_udf_active_fragment_tasks: bool = False,
    ) -> None:
        self.execute_fn = execute_fn
        self.tasks: dict[str, FteTaskExecution] = {}
        self.query_tasks: dict[str, set[str]] = {}
        self.admission_config = admission_config
        self.worker_label = str(worker_label or os.getenv("VANE_FTE_WORKER_ID") or "-")
        self.max_running_tasks = self.admission_config.max_running_tasks
        self.require_query_task_lease = bool(require_query_task_lease)
        self.sync_udf_active_fragment_tasks = bool(sync_udf_active_fragment_tasks)
        self.running_tasks: set[str] = set()
        self.queued_tasks: deque[str] = deque()
        self.dropped_task_statuses: dict[str, dict[str, Any]] = {}
        self.dropped_task_order: deque[str] = deque()
        self.max_dropped_task_statuses = 4096
        self._status_cache_lock = threading.RLock()
        self._status_cache: dict[str, dict[str, Any]] = {}
        self._admission_debug_log("manager_init")

    async def create_task(self, request: Mapping[str, Any]) -> dict[str, Any]:
        execution = FteTaskExecution(
            request,
            self.execute_fn,
            status_callback=self._publish_status,
            require_query_task_lease=self.require_query_task_lease,
            default_task_memory_bytes=self.admission_config.task_memory_bytes,
        )
        key = str(execution.task_id)
        existing = self.tasks.get(key)
        if existing is not None:
            return self._publish_status(existing)
        if execution.memory_requirement_bytes > self.admission_config.memory_budget_bytes:
            raise RuntimeError(
                f"FTE task {execution.task_id} heap {execution.memory_requirement_bytes} "
                "exceeds worker task-heap capacity "
                f"{self.admission_config.memory_budget_bytes}"
            )
        if key in self.dropped_task_statuses:
            self.dropped_task_statuses.pop(key, None)
            try:
                self.dropped_task_order.remove(key)
            except ValueError:
                pass
        with self._status_cache_lock:
            self._status_cache.pop(key, None)
        self.tasks[key] = execution
        self.query_tasks.setdefault(execution.task_id.query_id, set()).add(key)
        self._publish_status(execution)
        self._admission_debug_log("create_task", execution)
        self._admit_or_queue(execution)
        return self._publish_status(execution)

    @staticmethod
    def _key(task_id: Any) -> str:
        return str(FteTaskAttemptId.coerce(task_id))

    def _get(self, task_id: Any) -> FteTaskExecution:
        key = self._key(task_id)
        execution = self.tasks.get(key)
        if execution is None:
            raise KeyError(f"unknown FTE task attempt: {key}")
        return execution

    def _dropped_status(self, task_id: Any) -> dict[str, Any] | None:
        return self.dropped_task_statuses.get(self._key(task_id))

    def _unknown_status(self, task_id: Any) -> dict[str, Any]:
        attempt_id = FteTaskAttemptId.coerce(task_id)
        now = time.time()
        status = {
            "task_id": attempt_id.to_dict(),
            "task_id_string": str(attempt_id),
            "state": "UNKNOWN",
            "version": 0,
            "failure": None,
            "created_at": now,
            "updated_at": now,
            "duplicate_split_count": 0,
            "spooling_output_stats": None,
        }
        status.update(self._executor_stats(str(attempt_id)))
        return status

    def _store_dropped_status(self, key: str, status: Mapping[str, Any]) -> None:
        if key not in self.dropped_task_statuses:
            self.dropped_task_order.append(key)
        self.dropped_task_statuses[key] = dict(status)
        with self._status_cache_lock:
            self._status_cache[key] = dict(status)
        while len(self.dropped_task_order) > self.max_dropped_task_statuses:
            stale_key = self.dropped_task_order.popleft()
            self.dropped_task_statuses.pop(stale_key, None)
            with self._status_cache_lock:
                self._status_cache.pop(stale_key, None)

    def _admit_or_queue(self, execution: FteTaskExecution) -> None:
        key = str(execution.task_id)
        if key in self.running_tasks or execution.status.state in _TERMINAL_STATES:
            return
        block_reason = self._capacity_block_reason(execution)
        if not block_reason:
            self._start_execution(execution, reason="admit")
            return
        execution.queue()
        if key not in self.queued_tasks:
            self.queued_tasks.append(key)
        self._publish_status(execution)
        self._admission_debug_log("queue_task", execution, reason=block_reason)

    def _running_memory_requirement_bytes(self) -> int:
        total = 0
        for key in self.running_tasks:
            execution = self.tasks.get(key)
            if execution is None or execution.memory_requirement_bytes is None:
                continue
            total += int(execution.memory_requirement_bytes)
        return total

    def _has_capacity(self, execution: FteTaskExecution | None = None) -> bool:
        return not self._capacity_block_reason(execution)

    def _capacity_block_reason(self, execution: FteTaskExecution | None = None) -> str:
        if self.max_running_tasks is not None and len(self.running_tasks) >= self.max_running_tasks:
            return "max_running_tasks"
        if execution is None:
            return ""
        running_memory = self._running_memory_requirement_bytes()
        required_memory = int(execution.memory_requirement_bytes)
        if running_memory + required_memory <= int(self.admission_config.memory_budget_bytes):
            return ""
        return "memory_budget"

    def _start_execution(self, execution: FteTaskExecution, *, reason: str) -> None:
        key = str(execution.task_id)
        if execution.status.state in _TERMINAL_STATES:
            return
        self.running_tasks.add(key)
        self._sync_udf_active_fte_fragment_tasks()
        execution.start()
        self._publish_status(execution)
        self._admission_debug_log("start_task", execution, reason=reason)
        execution.add_done_callback(lambda future, task_key=key: self._task_done(task_key, future))

    def _task_done(self, task_key: str, _future: asyncio.Task[Any]) -> None:
        self.running_tasks.discard(task_key)
        self._sync_udf_active_fte_fragment_tasks()
        execution = self.tasks.get(task_key)
        if execution is not None:
            self._publish_status(execution)
        self._admission_debug_log("task_done", execution)
        self._drain_queue()

    def _sync_udf_active_fte_fragment_tasks(self) -> None:
        if not self.sync_udf_active_fragment_tasks:
            return

    def _drain_queue(self) -> None:
        while self.queued_tasks:
            key = self.queued_tasks.popleft()
            execution = self.tasks.get(key)
            if execution is None or execution.status.state in _TERMINAL_STATES:
                continue
            block_reason = self._capacity_block_reason(execution)
            if block_reason:
                self.queued_tasks.appendleft(key)
                self._admission_debug_log("drain_blocked", execution, reason=block_reason)
                return
            self._start_execution(execution, reason="drain")

    def _executor_stats(self, task_key: str | None = None) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "executor_running_task_count": len(self.running_tasks),
            "executor_queued_task_count": len(self.queued_tasks),
            "executor_max_running_tasks": self.max_running_tasks,
            "executor_admission_limited": self.max_running_tasks is not None,
            "executor_admission_mode": self.admission_config.mode,
            "executor_memory_budget_bytes": self.admission_config.memory_budget_bytes,
            "executor_task_memory_bytes": self.admission_config.task_memory_bytes,
            "executor_reserved_memory_bytes": self._running_memory_requirement_bytes(),
            "executor_total_committed_memory_bytes": self._running_memory_requirement_bytes(),
        }
        if task_key is not None:
            try:
                queue_position = list(self.queued_tasks).index(task_key)
            except ValueError:
                queue_position = None
            stats["executor_queue_position"] = queue_position
            execution = self.tasks.get(task_key)
            if execution is not None:
                stats["executor_task_memory_requirement_bytes"] = execution.memory_requirement_bytes
        return stats

    def _status_with_executor(self, execution: FteTaskExecution) -> dict[str, Any]:
        payload = execution.status_payload()
        payload.update(self._executor_stats(str(execution.task_id)))
        return payload

    def _publish_status(self, execution: FteTaskExecution) -> dict[str, Any]:
        payload = self._status_with_executor(execution)
        with self._status_cache_lock:
            self._status_cache[str(execution.task_id)] = dict(payload)
        return payload

    def get_cached_task_status(self, task_id: Any) -> dict[str, Any]:
        key = self._key(task_id)
        with self._status_cache_lock:
            status = self._status_cache.get(key)
            if status is not None:
                return dict(status)
        execution = self.tasks.get(key)
        if execution is not None:
            return self._publish_status(execution)
        status = self._dropped_status(task_id)
        if status is not None:
            return dict(status)
        return self._unknown_status(task_id)

    def ack_task_result(self, task_id: Any) -> dict[str, Any]:
        key = self._key(task_id)
        execution = self.tasks.get(key)
        if execution is not None:
            status = self._publish_status(execution)
            status.pop("result", None)
            return status
        status = self._dropped_status(task_id)
        if status is not None:
            updated = dict(status)
            updated.pop("result", None)
            return updated
        return self._unknown_status(task_id)

    def release_task_result(self, task_id: Any) -> dict[str, Any]:
        key = self._key(task_id)
        execution = self.tasks.get(key)
        if execution is not None:
            execution.release_result(reason="release_task_result")
            status = self._publish_status(execution)
            status.pop("result", None)
            return status
        status = self._dropped_status(task_id)
        if status is not None:
            updated = dict(status)
            updated.pop("result", None)
            self._store_dropped_status(key, updated)
            return updated
        return self._unknown_status(task_id)

    def _admission_debug_log(
        self,
        event: str,
        execution: FteTaskExecution | None = None,
        **fields: Any,
    ) -> None:
        if not _fte_admission_debug_enabled():
            return
        stats = self._executor_stats(str(execution.task_id) if execution is not None else None)
        parts: list[str] = [
            f"event={event}",
            f"worker_id={_format_admission_field(self.worker_label)}",
            f"running={stats.get('executor_running_task_count')}",
            f"queued={stats.get('executor_queued_task_count')}",
            f"max_running={_format_admission_field(stats.get('executor_max_running_tasks'))}",
            f"mode={_format_admission_field(stats.get('executor_admission_mode'))}",
            f"admission_limited={_format_admission_field(stats.get('executor_admission_limited'))}",
            f"reserved_memory_bytes={_format_admission_field(stats.get('executor_reserved_memory_bytes'))}",
            f"memory_budget_bytes={_format_admission_field(stats.get('executor_memory_budget_bytes'))}",
            f"task_memory_bytes={_format_admission_field(stats.get('executor_task_memory_bytes'))}",
        ]
        if execution is not None:
            task_id = execution.task_id
            parts.extend(
                [
                    f"task_id={_format_admission_field(str(task_id))}",
                    f"query_id={_format_admission_field(task_id.query_id)}",
                    f"fragment_execution_id={task_id.fragment_execution_id}",
                    f"partition_id={task_id.partition_id}",
                    f"attempt_id={task_id.attempt_id}",
                    f"state={_format_admission_field(execution.status.state.value)}",
                    f"queue_position={_format_admission_field(stats.get('executor_queue_position'))}",
                    "task_memory_requirement_bytes="
                    f"{_format_admission_field(stats.get('executor_task_memory_requirement_bytes'))}",
                ]
            )
        parts.extend(f"{key}={_format_admission_field(value)}" for key, value in fields.items())
        print(f"[vane-fte-admission pid={os.getpid()}] " + " ".join(parts), file=sys.stderr, flush=True)

    async def add_splits(
        self,
        task_id: Any,
        source_node_id: str,
        splits: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        execution = self._get(task_id)
        execution.add_splits(source_node_id, splits)
        return self._publish_status(execution)

    async def no_more_splits(self, task_id: Any, source_node_id: str) -> dict[str, Any]:
        execution = self._get(task_id)
        execution.no_more_splits(source_node_id)
        return self._publish_status(execution)

    async def update_task(
        self,
        task_id: Any,
        update: FteTaskUpdateRequest | Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        execution = self._get(task_id)
        execution.update_task(update)
        return self._publish_status(execution)

    async def get_task_status(self, task_id: Any) -> dict[str, Any]:
        execution = self.tasks.get(self._key(task_id))
        if execution is not None:
            return self._publish_status(execution)
        status = self._dropped_status(task_id)
        if status is not None:
            return dict(status)
        return self._unknown_status(task_id)

    async def wait_task_status(
        self,
        task_id: Any,
        min_version: int | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        execution = self.tasks.get(self._key(task_id))
        if execution is None:
            status = self._dropped_status(task_id)
            if status is not None:
                return dict(status)
            execution = self._get(task_id)
        await execution.wait_status_after(min_version, timeout_s)
        return self._publish_status(execution)

    async def wait_split_queue_has_space(
        self,
        task_id: Any,
        source_node_id: str | None = None,
        max_buffered_splits: int | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        execution = self._get(task_id)
        result = await execution.wait_split_queue_has_space(
            source_node_id,
            max_buffered_splits=max_buffered_splits,
            timeout_s=timeout_s,
        )
        result["status"] = self._publish_status(execution)
        return result

    async def get_task_info(self, task_id: Any) -> dict[str, Any]:
        execution = self.tasks.get(self._key(task_id))
        if execution is not None:
            return execution.info()
        status = self._dropped_status(task_id)
        if status is not None:
            return {
                "status": dict(status),
                "initial_split_counts": {},
                "no_more_splits": [],
                "result": status.get("result"),
                "spooling_output_stats": status.get("spooling_output_stats"),
            }
        return self._get(task_id).info()

    async def cancel_task(self, task_id: Any) -> dict[str, Any]:
        execution = self.tasks.get(self._key(task_id))
        if execution is None:
            status = self._dropped_status(task_id)
            if status is not None:
                return dict(status)
            execution = self._get(task_id)
        key = str(execution.task_id)
        try:
            self.queued_tasks.remove(key)
        except ValueError:
            pass
        execution.cancel()
        execution.release_result(reason="cancel_task")
        self.running_tasks.discard(key)
        self._sync_udf_active_fte_fragment_tasks()
        self._drain_queue()
        status = self._publish_status(execution)
        status.pop("result", None)
        return status

    async def drop_query(self, query_id: str) -> dict[str, int]:
        query_id = str(query_id)
        task_keys = list(self.query_tasks.get(query_id, ()))
        removed = 0
        canceled = 0
        errors: list[tuple[str, BaseException]] = []
        for key in task_keys:
            execution = self.tasks.get(key)
            if execution is None:
                query_tasks = self.query_tasks.get(query_id)
                if query_tasks is not None:
                    query_tasks.discard(key)
                continue
            try:
                self.queued_tasks.remove(key)
            except ValueError:
                pass
            canceled_task = False
            try:
                if execution.status.state not in _TERMINAL_STATES:
                    execution.cancel()
                    canceled_task = True
                execution.release_result(reason="drop_query")
                dropped_status = self._publish_status(execution)
                dropped_status.pop("result", None)
                self._store_dropped_status(key, dropped_status)
                with self._status_cache_lock:
                    self._status_cache.pop(key, None)
            except BaseException as exc:
                errors.append((key, exc))
                continue

            if self.tasks.get(key) is execution:
                self.tasks.pop(key, None)
            query_tasks = self.query_tasks.get(query_id)
            if query_tasks is not None:
                query_tasks.discard(key)
            self.running_tasks.discard(key)
            removed += 1
            canceled += int(canceled_task)

        query_tasks = self.query_tasks.get(query_id)
        if query_tasks is not None and not query_tasks:
            self.query_tasks.pop(query_id, None)
        try:
            self._sync_udf_active_fte_fragment_tasks()
        except BaseException as exc:
            errors.append(("<sync-admission>", exc))
        try:
            self._drain_queue()
        except BaseException as exc:
            errors.append(("<drain-queue>", exc))
        if errors:
            details = "; ".join(f"task={key}: {type(exc).__name__}: {exc}" for key, exc in errors)
            raise RuntimeError(
                f"failed to fully drop FTE query {query_id}; retained failed task ownership: " + details
            ) from errors[0][1]
        return {"removed": removed, "canceled": canceled}
