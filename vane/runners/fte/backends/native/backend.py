# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import copy
import inspect
import os
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from vane.runners.fte.backend import TaskResultPoll, TaskResultState
from vane.runners.fte.dynamic_inputs import (
    prepare_fte_dynamic_inputs,
    strip_fte_dynamic_context,
)
from vane.runners.fte.fte_config import FTE_WORKER_RUNTIME, FteWorkerAdmissionConfig
from vane.runners.fte.fte_state import FteTaskState
from vane.runners.fte.fte_types import (
    FteTaskAttemptId,
    FteTaskId,
    validate_fte_status_identity,
)
from vane.runners.fte.fte_worker_runtime import FteWorkerTaskManager, materialize_task_inputs
from vane.runners.progress import validate_pipeline_topology

_TERMINAL_STATE_VALUES = {
    FteTaskState.FINISHED.value,
    FteTaskState.FAILED.value,
    FteTaskState.CANCELED.value,
    FteTaskState.ABORTED.value,
}

_FRAGMENT_STAT_KEYS = (
    "executor_running_task_count",
    "executor_queued_task_count",
    "executor_max_running_tasks",
    "executor_admission_limited",
    "executor_reserved_memory_bytes",
)


def _native_submit_debug_enabled() -> bool:
    for name in ("VANE_FTE_ADMISSION_DEBUG", "DUCKDB_DISTRIBUTED_DEBUG"):
        value = os.getenv(name, "")
        if value.strip().lower() not in ("", "0", "false", "no", "off"):
            return True
    return False


def _format_debug_field(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace(" ", "_")


def _request_debug_fields(request: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "fragment_id": request.get("fragment_id"),
    }
    try:
        task_id = FteTaskAttemptId.coerce(request.get("task_id"))
    except Exception:
        fields["task_id"] = request.get("task_id")
        fields["query_id"] = request.get("query_id")
        return fields
    fields.update(
        {
            "task_id": str(task_id),
            "query_id": task_id.query_id,
            "fragment_execution_id": task_id.fragment_execution_id,
            "partition_id": task_id.partition_id,
            "attempt_id": task_id.attempt_id,
        }
    )
    return fields


def _native_submit_debug_log(event: str, **fields: Any) -> None:
    if not _native_submit_debug_enabled():
        return
    parts = [f"event={event}"]
    parts.extend(f"{key}={_format_debug_field(value)}" for key, value in fields.items())
    print(f"[vane-fte-native-submit pid={os.getpid()}] " + " ".join(parts), file=sys.stderr, flush=True)


def _debug_context_field(request: Mapping[str, Any], key: str) -> Any:
    context = request.get("context")
    if not isinstance(context, Mapping):
        return None
    return context.get(key)


def _debug_status_field(status: Mapping[str, Any], key: str) -> Any:
    value = status.get(key)
    if value is not None:
        return value
    task_stats = status.get("task_stats")
    if isinstance(task_stats, Mapping):
        return task_stats.get(key)
    return None


def _native_runtime_info_fields(info: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(info, Mapping):
        return {}
    runtime_status = info.get("status")
    if not isinstance(runtime_status, Mapping):
        runtime_status = {}
    return {
        "runtime_state": runtime_status.get("state"),
        "runtime_no_more_splits": info.get("no_more_splits"),
        "runtime_initial_split_counts": info.get("initial_split_counts"),
        "runtime_descriptor_version": info.get("descriptor_version"),
        "runtime_submitted_split_count_by_source": _debug_status_field(
            runtime_status, "submitted_split_count_by_source"
        ),
        "runtime_queued_split_count_by_source": _debug_status_field(runtime_status, "queued_split_count_by_source"),
        "runtime_consumed_split_count_by_source": _debug_status_field(runtime_status, "consumed_split_count_by_source"),
        "runtime_completed_split_count_by_source": _debug_status_field(
            runtime_status, "completed_split_count_by_source"
        ),
        "runtime_split_queue_has_space": _debug_status_field(runtime_status, "split_queue_has_space"),
        "runtime_split_queue_max_buffered_splits": _debug_status_field(
            runtime_status, "split_queue_max_buffered_splits"
        ),
    }


def _native_pending_status_fields(
    handle: Any,
    status: Mapping[str, Any],
    info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    request = handle.request
    fields = _request_debug_fields(request)
    fields.update(
        {
            "state": status.get("state"),
            "request_source_node_ids": request.get("source_node_ids"),
            "request_dynamic_scan_source_node_ids": request.get("dynamic_scan_source_node_ids"),
            "request_dynamic_exchange_source_node_ids": request.get("dynamic_exchange_source_node_ids"),
            "context_scan_task_nodes": _debug_context_field(request, "scan_task_nodes"),
            "context_exchange_source_task_nodes": _debug_context_field(request, "exchange_source_task_nodes"),
            "no_more_splits": status.get("no_more_splits"),
            "submitted_split_count_by_source": _debug_status_field(status, "submitted_split_count_by_source"),
            "queued_split_count_by_source": _debug_status_field(status, "queued_split_count_by_source"),
            "consumed_split_count_by_source": _debug_status_field(status, "consumed_split_count_by_source"),
            "completed_split_count_by_source": _debug_status_field(status, "completed_split_count_by_source"),
            "submitted_input_rows_by_source": _debug_status_field(status, "submitted_input_rows_by_source"),
            "consumed_input_rows_by_source": _debug_status_field(status, "consumed_input_rows_by_source"),
            "completed_input_rows_by_source": _debug_status_field(status, "completed_input_rows_by_source"),
            "split_queue_has_space": _debug_status_field(status, "split_queue_has_space"),
            "split_queue_max_buffered_splits": _debug_status_field(status, "split_queue_max_buffered_splits"),
        }
    )
    fields.update(_native_runtime_info_fields(info))
    return fields


class _BackgroundEventLoop:
    def __init__(self, thread_name: str) -> None:
        self._thread_name = thread_name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._closed = False

    def start(self) -> None:
        if self._loop is not None:
            return

        def run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._ready.set()
            loop.run_forever()
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

        self._thread = threading.Thread(target=run, name=self._thread_name, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("failed to start native FTE event loop")

    def submit(self, coro: Awaitable[Any]) -> Future:
        if self._closed:
            raise RuntimeError("native FTE event loop is closed")
        self.start()
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def run(self, coro: Awaitable[Any], timeout_s: float | None = None) -> Any:
        return self.submit(coro).result(timeout=timeout_s)

    def shutdown(self, timeout_s: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)


def _as_status(method_name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{method_name} must return a mapping")
    return dict(value)


def _query_id_from_task_id(task_id: Any) -> str:
    return FteTaskAttemptId.coerce(task_id).query_id


class _CallableString(str):
    def __call__(self) -> str:
        return str(self)


def _flight_exchange_node_id_from_env() -> str:
    for key in ("VANE_WORKER_ID", "RAY_NODE_IP_ADDRESS", "RAY_NODE_ID", "HOSTNAME"):
        value = os.getenv(key)
        if value:
            return str(value)
    return "local"


def _native_total_num_cpus() -> float:
    return max(1.0, float(os.cpu_count() or 1))


def _native_total_memory_bytes() -> int:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages) * int(page_size)
    except (AttributeError, OSError, ValueError):
        pass
    return 0


def _task_context_info(task_context: Any) -> dict[str, Any]:
    if isinstance(task_context, Mapping):
        payload = dict(task_context)
        required = {"query_idx", "last_node_id", "task_id", "node_ids"}
        if required.issubset(payload):
            return payload
        task_id = int(payload.get("task_id") or 0)
        return {
            "query_idx": int(payload.get("query_idx") or 0),
            "last_node_id": int(payload.get("last_node_id") or task_id or 0),
            "task_id": task_id,
            "node_ids": list(payload.get("node_ids") or [int(payload.get("last_node_id") or task_id or 0)]),
        }
    return {
        "query_idx": 0,
        "last_node_id": 0,
        "task_id": 0,
        "node_ids": [0],
    }


def _ray_cxx_attr(name: str) -> Any:
    # The current C++ task-result classes live under vane.ray_cxx even when
    # used by the Ray-free native backend. This imports the compiled binding,
    # not the Ray Python runtime.
    from vane._ray_cxx import require_ray_cxx_attr

    return require_ray_cxx_attr(name)


def _stats_from_payload(stats: Any) -> list[int]:
    if stats is None:
        return []
    if isinstance(stats, (bytes, bytearray)):
        return list(stats)
    if isinstance(stats, memoryview):
        return list(stats.tobytes())
    if isinstance(stats, (list, tuple)):
        return [int(value) for value in stats]
    return []


def _idx_stat(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _metadata_rows_bytes(metadata: Any) -> tuple[int, int]:
    if isinstance(metadata, Mapping):
        return int(metadata.get("num_rows") or metadata.get("rows") or 0), int(
            metadata.get("size_bytes") or metadata.get("bytes") or 0
        )
    if isinstance(metadata, (tuple, list)) and len(metadata) >= 2:
        return int(metadata[0] or 0), int(metadata[1] or 0)
    rows = getattr(metadata, "num_rows", 0)
    bytes_value = getattr(metadata, "size_bytes", 0)
    return int(rows or 0), int(bytes_value or 0)


def _is_native_distributed_task_result(value: Any) -> bool:
    return all(
        hasattr(value, attr)
        for attr in (
            "partition_payloads",
            "partition_metadatas",
            "result_schema",
            "stats",
            "flight_port",
        )
    )


def _native_result_tuple(value: Any) -> tuple[Any, Any, Any, Any, int, Any]:
    if _is_native_distributed_task_result(value):
        return (
            list(value.partition_payloads),
            list(value.partition_metadatas),
            value.result_schema,
            value.stats,
            int(value.flight_port or 0),
            value.exchange_sink_instance,
        )
    if isinstance(value, (tuple, list)):
        payloads = value[0] if len(value) >= 1 else []
        metadatas = value[1] if len(value) >= 2 else []
        result_schema = value[2] if len(value) >= 3 else None
        stats = value[3] if len(value) >= 4 else []
        if len(value) >= 8 and isinstance(value[4], str):
            flight_port = int(value[5] or 0)
            exchange_sink_instance = value[6]
        else:
            flight_port = int(value[4] or 0) if len(value) >= 5 else 0
            exchange_sink_instance = value[5] if len(value) >= 6 else None
        return payloads, metadatas, result_schema, stats, flight_port, exchange_sink_instance
    raise TypeError(f"unsupported native task result payload: {type(value).__name__}")


def _normalize_result_for_cxx(value: Any) -> Any:
    RayTaskResult = _ray_cxx_attr("RayTaskResult")
    if isinstance(value, RayTaskResult):
        return value
    if value is None:
        return RayTaskResult.no_output()
    if isinstance(value, Mapping):
        if "result" in value:
            return _normalize_result_for_cxx(value.get("result"))
        if any(key in value for key in ("spooling_output_stats", "output_stats", "task_stats")):
            return RayTaskResult.success([], _stats_from_payload(value.get("stats")), None)
        return RayTaskResult.success([], [], None)

    if not _is_native_distributed_task_result(value) and not isinstance(value, (tuple, list)):
        return RayTaskResult.success([], [], None)

    RayResultPartitionRef = _ray_cxx_attr("RayResultPartitionRef")
    payloads, _metadatas, result_schema, stats, flight_port, exchange_sink_instance = _native_result_tuple(value)
    partition_refs = []
    for payload in payloads or []:
        if isinstance(payload, RayResultPartitionRef):
            partition_refs.append(payload)
        else:
            # Local/native payloads are materialized directly by C++; only Ray
            # ObjectRefs use RayResultPartitionRef and therefore require a real
            # query output-lease owner.
            partition_refs.append(payload)
    return RayTaskResult.success(
        partition_refs,
        _stats_from_payload(stats),
        result_schema,
        flight_port,
        exchange_sink_instance,
    )


class NativeTaskResultHandle:
    def __init__(
        self,
        worker: NativeWorkerHandle,
        task_id: FteTaskAttemptId | str | Mapping[str, Any],
        *,
        task_context: Any = None,
        request: Mapping[str, Any] | None = None,
        status_callback: Callable[[NativeTaskResultHandle, Mapping[str, Any], BaseException | None], None]
        | None = None,
    ) -> None:
        self._worker = worker
        self._task_id = FteTaskAttemptId.coerce(task_id)
        self._task_context = task_context
        self._request = dict(request or {})
        self._status_callback = status_callback
        self.task_context_info = _task_context_info(task_context)
        self.task_id = self._task_id
        self.worker_id = _CallableString(worker.worker_id)
        self.exchange_node_id = _CallableString(_flight_exchange_node_id_from_env())
        self._acked = False

    def task_context(self) -> Any:
        return self._task_context

    def fte_task_id(self) -> str:
        return str(self._task_id)

    @property
    def fragment_id(self) -> str:
        return str(
            self._request.get("fragment_id")
            or f"{self._task_id.query_id}:fragment-{self._task_id.fragment_execution_id}"
        )

    @property
    def request(self) -> Mapping[str, Any]:
        return self._request

    def _record_status(self, status: Mapping[str, Any], error: BaseException | None = None) -> None:
        callback = self._status_callback
        if callback is None:
            return
        try:
            callback(self, status, error)
        except Exception:
            pass

    def _failure_status(self) -> dict[str, Any]:
        return {
            "state": FteTaskState.FAILED.value,
            "task_id": self._task_id.to_dict(),
            "task_id_string": str(self._task_id),
        }

    def _validated_status(self, status: Any, *, operation: str) -> dict[str, Any]:
        if not isinstance(status, Mapping):
            raise TypeError(f"{operation} must return a status mapping")
        result = dict(status)
        validate_fte_status_identity(result, self._task_id)
        return result

    def status_snapshot(self) -> dict[str, Any]:
        try:
            status = self._validated_status(
                self._worker.fte_get_task_status_cached(self._task_id.to_dict()),
                operation="fte_get_task_status_cached",
            )
        except BaseException as exc:
            self._record_status(self._failure_status(), exc)
            raise
        self._record_status(status)
        return status

    def info_snapshot(self) -> dict[str, Any]:
        try:
            raw_info = self._worker.fte_get_task_info(self._task_id.to_dict())
            if not isinstance(raw_info, Mapping):
                raise TypeError("fte_get_task_info must return a mapping")
            info = dict(raw_info)
            status = self._validated_status(
                info.get("status"),
                operation="fte_get_task_info.status",
            )
            info["status"] = status
        except BaseException as exc:
            self._record_status(self._failure_status(), exc)
            raise
        self._record_status(status)
        return info

    def poll(self) -> TaskResultPoll:
        try:
            status = self._validated_status(
                self._worker.fte_get_task_status_cached(self._task_id.to_dict()),
                operation="fte_get_task_status_cached",
            )
        except BaseException as exc:
            self._record_status(self._failure_status(), exc)
            return TaskResultPoll(TaskResultState.ERROR, error=exc)

        self._record_status(status)
        state = str(status.get("state") or "").upper()
        if state not in _TERMINAL_STATE_VALUES:
            return TaskResultPoll(TaskResultState.NOT_READY)
        if state == FteTaskState.FINISHED.value:
            result = status.get("result")
            if result is None:
                return TaskResultPoll(TaskResultState.NO_OUTPUT)
            return TaskResultPoll(TaskResultState.MATERIALIZED_OUTPUT, output=result)
        failure = status.get("failure")
        message = failure.get("message") if isinstance(failure, Mapping) else None
        return TaskResultPoll(
            TaskResultState.ERROR,
            error=RuntimeError(message or f"native FTE task {self._task_id} ended with {state}"),
        )

    def done(self) -> bool:
        return self.poll().state is not TaskResultState.NOT_READY

    def get_result_sync(self) -> Any:
        poll = self.poll()
        if poll.state is TaskResultState.NOT_READY:
            raise RuntimeError("native FTE task result not ready")
        if poll.state is TaskResultState.ERROR:
            if poll.error is not None:
                raise poll.error
            raise RuntimeError(f"native FTE task {self._task_id} failed")
        return _normalize_result_for_cxx(poll.output)

    def ack(self) -> None:
        if self._acked:
            return
        self._validated_status(
            self._worker.fte_ack_task_result(self._task_id.to_dict()),
            operation="fte_ack_task_result",
        )
        self._acked = True

    def release_result_payload(self) -> None:
        self._validated_status(
            self._worker.fte_release_task_result(self._task_id.to_dict()),
            operation="fte_release_task_result",
        )

    @property
    def acked(self) -> bool:
        return self._acked


@dataclass
class _NativeFteRegisteredPartition:
    request: dict[str, Any]
    task_id: FteTaskAttemptId
    worker_id: str | None = None
    last_metrics: dict[str, Any] | None = None


@dataclass
class _NativeFteRegisteredFragment:
    query_id: str
    fragment_id: str
    fragment_execution_id: int
    source_node_ids: set[str] = field(default_factory=set)
    dynamic_scan_source_node_ids: set[str] = field(default_factory=set)
    dynamic_exchange_source_node_ids: set[str] = field(default_factory=set)
    partitions: dict[str, _NativeFteRegisteredPartition] = field(default_factory=dict)
    progress_topology: dict[str, Any] | None = None


class _NativeFteProgressRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._next_fragment_execution_id_by_query: dict[str, int] = defaultdict(int)
        self._fragments_by_query: dict[str, dict[str, _NativeFteRegisteredFragment]] = defaultdict(dict)

    def register_requests(self, requests: Sequence[Mapping[str, Any]]) -> None:
        with self._lock:
            for request in requests:
                task_id = FteTaskAttemptId.coerce(request.get("task_id"))
                query_id = task_id.query_id
                fragment_id = str(request.get("fragment_id") or f"{query_id}:fragment-{task_id.fragment_execution_id}")
                fragment = self._get_or_create_fragment_locked(query_id, fragment_id)
                self._merge_fragment_sources_locked(fragment, request)
                partition_key = str(task_id.partition_id)
                existing = fragment.partitions.get(partition_key)
                if existing is None or int(existing.task_id.attempt_id) <= int(task_id.attempt_id):
                    fragment.partitions[partition_key] = _NativeFteRegisteredPartition(
                        request=dict(request),
                        task_id=task_id,
                        worker_id=existing.worker_id if existing is not None else None,
                        last_metrics=existing.last_metrics if existing is not None else None,
                    )

    def attach_handle(self, handle: NativeTaskResultHandle) -> None:
        with self._lock:
            task_id = handle.task_id
            fragment = self._get_or_create_fragment_locked(task_id.query_id, handle.fragment_id)
            self._merge_fragment_sources_locked(fragment, handle.request)
            partition_key = str(task_id.partition_id)
            partition = fragment.partitions.get(partition_key)
            if partition is None:
                partition = _NativeFteRegisteredPartition(
                    request=dict(handle.request),
                    task_id=task_id,
                )
                fragment.partitions[partition_key] = partition
            partition.worker_id = str(handle.worker_id)

    def drop_query(self, query_id: str) -> None:
        query_id = str(query_id)
        with self._lock:
            self._fragments_by_query.pop(query_id, None)
            self._next_fragment_execution_id_by_query.pop(query_id, None)

    def record_partition_metrics(
        self,
        query_id: str,
        fragment_id: str,
        partition_id: str,
        metrics: Mapping[str, Any],
    ) -> None:
        query_id = str(query_id)
        fragment_id = str(fragment_id)
        partition_id = str(partition_id)
        with self._lock:
            fragment = self._fragments_by_query.get(query_id, {}).get(fragment_id)
            if fragment is None:
                return
            partition = fragment.partitions.get(partition_id)
            if partition is None:
                return
            partition.last_metrics = dict(metrics)
            self._merge_fragment_progress_topology_locked(fragment, metrics)

    def query_status(
        self,
        query_id: str,
        *,
        partition_metrics: Mapping[tuple[str, str, str], dict[str, Any]] | None = None,
        failed_partitions: Sequence[Mapping[str, Any]] | None = None,
        selected_attempt_task_ids: Sequence[str] | None = None,
        result_handle_count: int = 0,
    ) -> dict[str, Any]:
        query_id = str(query_id)
        partition_metrics = partition_metrics or {}
        failed_partition_items = [dict(item) for item in failed_partitions or []]
        selected_attempt_ids = {str(task_id) for task_id in selected_attempt_task_ids or []}

        with self._lock:
            fragments = self._fragments_by_query.get(query_id, {})
            fragment_executions: dict[str, dict[str, Any]] = {}
            for fragment_id, registered in fragments.items():
                partitions: dict[str, dict[str, Any]] = {}
                for partition_id, partition in registered.partitions.items():
                    metrics_key = (query_id, fragment_id, partition_id)
                    metrics = partition_metrics.get(metrics_key)
                    if metrics is not None:
                        partition.last_metrics = dict(metrics)
                    metrics = dict(partition.last_metrics or self._placeholder_partition_metrics(partition))
                    self._merge_fragment_progress_topology_locked(registered, metrics)
                    partitions[partition_id] = metrics
                    if str(metrics.get("state") or "").upper() == "FINISHED":
                        selected_attempt_ids.add(str(partition.task_id))

                partition_count = len(partitions)
                running_count = sum(int(partition.get("running_count") or 0) for partition in partitions.values())
                failed_count = sum(1 for partition in partitions.values() if partition.get("state") == "FAILED")
                finished_count = sum(1 for partition in partitions.values() if partition.get("state") == "FINISHED")
                waiting_for_node_count = sum(
                    1 for partition in partitions.values() if partition.get("waiting_for_node")
                )
                waiting_for_execution_count = sum(
                    1 for partition in partitions.values() if partition.get("waiting_for_execution")
                )
                deferred_count = sum(
                    1 for partition in partitions.values() if partition.get("execution_ready_deferred")
                )
                execution_class_counts: dict[str, int] = {}
                for partition in partitions.values():
                    execution_class = str(partition.get("execution_class") or "STANDARD")
                    execution_class_counts[execution_class] = execution_class_counts.get(execution_class, 0) + 1

                fragment_executions[fragment_id] = {
                    "query_id": query_id,
                    "fragment_id": fragment_id,
                    "fragment_execution_id": registered.fragment_execution_id,
                    "fragment_execution_class": "STANDARD",
                    "partition_count": partition_count,
                    "running_count": running_count,
                    "failed_count": failed_count,
                    "finished_count": finished_count,
                    "waiting_for_node_count": waiting_for_node_count,
                    "waiting_for_execution_count": waiting_for_execution_count,
                    "execution_deferred_count": deferred_count,
                    "pending_submission_count": 0,
                    "execution_class_counts": execution_class_counts,
                    "failed": failed_count > 0,
                    "finished": partition_count > 0 and finished_count == partition_count and failed_count == 0,
                    "no_more_partitions": True,
                    "source_node_ids": sorted(registered.source_node_ids),
                    "dynamic_scan_source_node_ids": sorted(registered.dynamic_scan_source_node_ids),
                    "dynamic_exchange_source_node_ids": sorted(registered.dynamic_exchange_source_node_ids),
                    "exchange_selectors": {},
                    "progress_topology": copy.deepcopy(
                        registered.progress_topology or {"schema": "pipeline_topology", "pipelines": []}
                    ),
                    "partitions": partitions,
                }

        partition_count = sum(fragment["partition_count"] for fragment in fragment_executions.values())
        running_count = sum(fragment["running_count"] for fragment in fragment_executions.values())
        failed_count = sum(fragment["failed_count"] for fragment in fragment_executions.values())
        finished_count = sum(fragment["finished_count"] for fragment in fragment_executions.values())
        waiting_for_execution_count = sum(
            fragment["waiting_for_execution_count"] for fragment in fragment_executions.values()
        )
        waiting_for_node_count = sum(fragment["waiting_for_node_count"] for fragment in fragment_executions.values())
        pending_submission_count = sum(
            fragment["pending_submission_count"] for fragment in fragment_executions.values()
        )
        failed = failed_count > 0 or any(fragment["failed"] for fragment in fragment_executions.values())
        finished = bool(fragment_executions) and all(fragment["finished"] for fragment in fragment_executions.values())
        return {
            "query_id": query_id,
            "fragment_execution_count": len(fragment_executions),
            "partition_count": partition_count,
            "running_count": running_count,
            "failed_count": failed_count,
            "finished_count": finished_count,
            "waiting_for_node_count": waiting_for_node_count,
            "waiting_for_execution_count": waiting_for_execution_count,
            "pending_submission_count": pending_submission_count,
            "pending_worker_reservation_count": 0,
            "pending_worker_reservation_done_count": 0,
            "result_handle_count": result_handle_count,
            "failed": failed,
            "finished": finished,
            "canceled": False,
            "selected_attempt_task_ids": sorted(selected_attempt_ids),
            "fragment_executions": fragment_executions,
            "failed_partitions": failed_partition_items,
            "scheduler_state": "FAILED" if failed else ("FINISHED" if finished else "RUNNING"),
        }

    def _get_or_create_fragment_locked(
        self,
        query_id: str,
        fragment_id: str,
    ) -> _NativeFteRegisteredFragment:
        fragments = self._fragments_by_query[query_id]
        fragment = fragments.get(fragment_id)
        if fragment is not None:
            return fragment
        fragment_execution_id = self._next_fragment_execution_id_by_query[query_id]
        self._next_fragment_execution_id_by_query[query_id] = fragment_execution_id + 1
        fragment = _NativeFteRegisteredFragment(
            query_id=query_id,
            fragment_id=fragment_id,
            fragment_execution_id=fragment_execution_id,
        )
        fragments[fragment_id] = fragment
        return fragment

    def _merge_fragment_sources_locked(
        self,
        fragment: _NativeFteRegisteredFragment,
        request: Mapping[str, Any],
    ) -> None:
        dynamic_scan_sources = NativeFteWorkerManagerBackend._source_ids_from_request(
            request,
            "dynamic_scan_source_node_ids",
            "scan_source_node_ids",
        )
        dynamic_exchange_sources = NativeFteWorkerManagerBackend._source_ids_from_request(
            request,
            "dynamic_exchange_source_node_ids",
            "exchange_source_node_ids",
        )
        source_ids = set(fragment.source_node_ids)
        source_ids.update(dynamic_scan_sources)
        source_ids.update(dynamic_exchange_sources)
        source_ids.update(NativeFteWorkerManagerBackend._context_source_ids(request, "scan_task_nodes"))
        source_ids.update(NativeFteWorkerManagerBackend._context_source_ids(request, "exchange_source_task_nodes"))
        fragment.source_node_ids = source_ids
        fragment.dynamic_scan_source_node_ids.update(dynamic_scan_sources)
        fragment.dynamic_scan_source_node_ids.update(
            NativeFteWorkerManagerBackend._context_source_ids(request, "scan_task_nodes")
        )
        fragment.dynamic_exchange_source_node_ids.update(dynamic_exchange_sources)
        fragment.dynamic_exchange_source_node_ids.update(
            NativeFteWorkerManagerBackend._context_source_ids(request, "exchange_source_task_nodes")
        )

    @staticmethod
    def _task_stats_from_partition_metrics(
        metrics: Mapping[str, Any],
    ) -> list[Mapping[str, Any]]:
        task_stats: list[Mapping[str, Any]] = []
        running_attempts = metrics.get("running_attempts")
        if running_attempts is not None:
            if not isinstance(running_attempts, Sequence) or isinstance(running_attempts, (str, bytes, bytearray)):
                raise TypeError("native progress running_attempts must be a sequence")
            for attempt in running_attempts:
                if not isinstance(attempt, Mapping):
                    raise TypeError("native progress running attempts must be mappings")
                stats = attempt.get("task_stats")
                if stats is not None:
                    if not isinstance(stats, Mapping):
                        raise TypeError("native progress task_stats must be a mapping")
                    task_stats.append(stats)
        selected_stats = metrics.get("selected_output_stats")
        if selected_stats is not None:
            if not isinstance(selected_stats, Mapping):
                raise TypeError("native progress selected_output_stats must be a mapping")
            task_stats.append(selected_stats)
        return task_stats

    @staticmethod
    def _topology_from_task_stats(
        task_stats: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        raw_pipelines = task_stats.get("pipelines")
        if raw_pipelines is None:
            return None
        if type(raw_pipelines) is not list:
            raise TypeError("native progress pipelines must be a list")
        if not raw_pipelines:
            return None
        pipelines: list[dict[str, Any]] = []
        for raw_pipeline in raw_pipelines:
            if not isinstance(raw_pipeline, Mapping):
                raise TypeError("native progress pipeline entries must be mappings")
            pipelines.append(
                {
                    "pipeline_id": raw_pipeline["pipeline_id"],
                    "operators": raw_pipeline["operators"],
                    "operator_details": raw_pipeline["operator_details"],
                    "stage_ids": raw_pipeline["stage_ids"],
                }
            )
        return validate_pipeline_topology({"schema": "pipeline_topology", "pipelines": pipelines})

    @classmethod
    def _merge_fragment_progress_topology_locked(
        cls,
        fragment: _NativeFteRegisteredFragment,
        metrics: Mapping[str, Any],
    ) -> None:
        for task_stats in cls._task_stats_from_partition_metrics(metrics):
            topology = cls._topology_from_task_stats(task_stats)
            if topology is None:
                continue
            if fragment.progress_topology is None:
                fragment.progress_topology = topology
            elif fragment.progress_topology != topology:
                raise RuntimeError(
                    f"native fragment progress topology changed after publication: {fragment.fragment_id}"
                )

    @staticmethod
    def _placeholder_partition_metrics(partition: _NativeFteRegisteredPartition) -> dict[str, Any]:
        task_id = partition.task_id
        return {
            "task_id": str(task_id.task_id),
            "task": task_id.task_id.to_dict(),
            "partition_id": int(task_id.partition_id),
            "state": "SEALED",
            "execution_class": str(partition.request.get("execution_class") or "STANDARD"),
            "sealed": True,
            "ready_for_scheduling": True,
            "execution_ready_deferred": False,
            "waiting_for_node": False,
            "waiting_for_execution": True,
            "remaining_attempts": 1,
            "max_attempts": 1,
            "memory_requirement_bytes": None,
            "owner_worker_id": partition.worker_id,
            "pending_worker_reservation": False,
            "pending_worker_reservation_done": False,
            "pending_worker_reservation_generation": None,
            "running_attempts": [],
            "running_count": 0,
            "selected_attempt": None,
            "selected_output_stats": None,
            "finished_attempts": [],
            "failure_observed": False,
            "failure_count": 0,
            "failures": [],
            "initial_split_count_by_source": {},
            "no_more_splits": [],
        }


class NativeWorkerHandle:
    def __init__(
        self,
        worker_id: str,
        execute_fn: Callable[[Mapping[str, Any]], Any],
        *,
        max_running_tasks: int | None = None,
        num_cpus: float | None = None,
        total_memory_bytes: int | None = None,
        loop: _BackgroundEventLoop | None = None,
    ) -> None:
        worker_id = str(worker_id or "").strip()
        if not worker_id:
            raise ValueError("worker_id must be non-empty")
        self._worker_id = worker_id
        self._loop = loop or _BackgroundEventLoop(f"local-fte-worker-{worker_id}")
        self._owns_loop = loop is None
        self._execute_fn = execute_fn
        self._num_cpus = max(1.0, float(num_cpus if num_cpus is not None else _native_total_num_cpus()))
        self._total_memory_bytes = max(
            0,
            int(total_memory_bytes if total_memory_bytes is not None else _native_total_memory_bytes()),
        )
        if self._total_memory_bytes <= 0:
            raise RuntimeError("native FTE worker requires a positive memory capacity")
        task_slots = max(1, int(max_running_tasks or self._num_cpus))
        self._manager = FteWorkerTaskManager(
            self._async_execute,
            admission_config=FteWorkerAdmissionConfig(
                max_running_tasks=task_slots,
                mode="native",
                memory_budget_bytes=self._total_memory_bytes,
                task_memory_bytes=max(1, self._total_memory_bytes // task_slots),
            ),
            worker_label=worker_id,
            sync_udf_active_fragment_tasks=True,
        )
        self._started_attempts: set[str] = set()
        self._terminal_attempts: set[str] = set()

    @property
    def worker_id(self) -> str:
        return self._worker_id

    async def _async_execute(self, request: Mapping[str, Any]) -> Any:
        request_payload = dict(request)
        if inspect.iscoroutinefunction(self._execute_fn):
            return await self._execute_fn(request_payload)
        return await asyncio.to_thread(self._execute_fn, request_payload)

    def fte_create_task(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return _as_status("fte_create_task", self._loop.run(self._manager.create_task(dict(request))))

    def fte_add_splits(
        self,
        task_id: str | Mapping[str, Any],
        source_node_id: str,
        splits: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        split_payloads = [dict(split) for split in splits]
        return _as_status(
            "fte_add_splits",
            self._loop.run(self._manager.add_splits(task_id, str(source_node_id), split_payloads)),
        )

    def fte_no_more_splits(
        self,
        task_id: str | Mapping[str, Any],
        source_node_id: str,
    ) -> dict[str, Any]:
        return _as_status(
            "fte_no_more_splits",
            self._loop.run(self._manager.no_more_splits(task_id, str(source_node_id))),
        )

    def fte_update_task(
        self,
        task_id: str | Mapping[str, Any],
        update: Mapping[str, Any],
    ) -> dict[str, Any]:
        return _as_status("fte_update_task", self._loop.run(self._manager.update_task(task_id, dict(update))))

    def fte_get_task_status(self, task_id: str | Mapping[str, Any]) -> dict[str, Any]:
        return _as_status("fte_get_task_status", self._loop.run(self._manager.get_task_status(task_id)))

    def fte_get_task_status_cached(self, task_id: str | Mapping[str, Any]) -> dict[str, Any]:
        return _as_status("fte_get_task_status_cached", self._manager.get_cached_task_status(task_id))

    def fte_ack_task_result(self, task_id: str | Mapping[str, Any]) -> dict[str, Any]:
        return _as_status("fte_ack_task_result", self._manager.ack_task_result(task_id))

    def fte_release_task_result(self, task_id: str | Mapping[str, Any]) -> dict[str, Any]:
        return _as_status("fte_release_task_result", self._manager.release_task_result(task_id))

    def fte_wait_task_status(
        self,
        task_id: str | Mapping[str, Any],
        min_version: int | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return _as_status(
            "fte_wait_task_status",
            self._loop.run(self._manager.wait_task_status(task_id, min_version, timeout_s)),
        )

    def fte_wait_split_queue_has_space(
        self,
        task_id: str | Mapping[str, Any],
        source_node_id: str | None = None,
        max_buffered_splits: int | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return _as_status(
            "fte_wait_split_queue_has_space",
            self._loop.run(
                self._manager.wait_split_queue_has_space(
                    task_id,
                    source_node_id,
                    max_buffered_splits,
                    timeout_s,
                )
            ),
        )

    def fte_get_task_info(self, task_id: str | Mapping[str, Any]) -> dict[str, Any]:
        return _as_status("fte_get_task_info", self._loop.run(self._manager.get_task_info(task_id)))

    def fte_cancel_task(self, task_id: str | Mapping[str, Any]) -> dict[str, Any]:
        return _as_status("fte_cancel_task", self._loop.run(self._manager.cancel_task(task_id)))

    def fte_drop_query(self, query_id: str) -> dict[str, int]:
        result = self._loop.run(self._manager.drop_query(str(query_id)))
        if not isinstance(result, Mapping):
            raise TypeError("fte_drop_query must return a mapping")
        return {str(key): int(value) for key, value in result.items()}

    def record_fte_task_started(self, attempt_id: Any, _request: Mapping[str, Any] | None = None) -> None:
        self._started_attempts.add(str(FteTaskAttemptId.coerce(attempt_id)))

    def record_fte_task_terminal(self, attempt_id: Any) -> None:
        self._terminal_attempts.add(str(FteTaskAttemptId.coerce(attempt_id)))

    def record_fte_task_result_ready(self, attempt_id: Any) -> None:
        self.record_fte_task_terminal(attempt_id)

    def snapshot(self) -> dict[str, Any]:
        stats = self._manager._executor_stats()
        stats["worker_id"] = self.worker_id
        stats["num_cpus"] = self._num_cpus
        stats["CPU"] = self._num_cpus
        stats["num_gpus"] = 0.0
        stats["GPU"] = 0.0
        stats["total_memory_bytes"] = self._total_memory_bytes
        stats["memory"] = self._total_memory_bytes
        return stats

    def shutdown(self) -> None:
        if self._owns_loop:
            self._loop.shutdown()


class NativeFteWorkerManagerBackend:
    def __init__(
        self,
        workers: Sequence[NativeWorkerHandle] | None = None,
        *,
        execute_fn: Callable[[Mapping[str, Any]], Any] | None = None,
        num_workers: int = 1,
        max_running_tasks: int | None = None,
        num_cpus: float | None = None,
        total_memory_bytes: int | None = None,
    ) -> None:
        if workers is None:
            if execute_fn is None:
                raise ValueError("execute_fn is required when workers are not provided")
            worker_count = max(1, int(num_workers))
            total_num_cpus = max(1.0, float(num_cpus if num_cpus is not None else _native_total_num_cpus()))
            per_worker_num_cpus = max(1.0, total_num_cpus / float(worker_count))
            total_memory = max(
                0,
                int(total_memory_bytes if total_memory_bytes is not None else _native_total_memory_bytes()),
            )
            per_worker_memory = total_memory // worker_count if total_memory > 0 else 0
            workers = [
                NativeWorkerHandle(
                    f"native-worker-{index}",
                    execute_fn,
                    max_running_tasks=max_running_tasks,
                    num_cpus=per_worker_num_cpus,
                    total_memory_bytes=per_worker_memory,
                )
                for index in range(worker_count)
            ]
        if not workers:
            raise ValueError("at least one native worker is required")
        self._workers = list(workers)
        self._next_worker_index = 0
        self._handles_by_query: dict[str, list[NativeTaskResultHandle]] = defaultdict(list)
        self._handles_lock = threading.RLock()
        self._dropped_queries: dict[str, dict[str, Any]] = {}
        self._progress_registry = _NativeFteProgressRegistry()
        self._closed = False
        self._debug_sampler_stop = threading.Event()
        self._debug_sampler_thread: threading.Thread | None = None
        if _native_submit_debug_enabled():
            self._start_debug_sampler()

    def worker_snapshots(self) -> Sequence[Mapping[str, Any]]:
        return [worker.snapshot() for worker in self._workers]

    def fragment_stats_by_worker(self) -> dict[str, dict[str, int]]:
        stats_by_worker: dict[str, dict[str, int]] = {}
        for worker in self._workers:
            snapshot = worker.snapshot()
            worker_id = str(snapshot.get("worker_id") or worker.worker_id)
            stats_by_worker[worker_id] = {
                key: _idx_stat(snapshot.get(key)) for key in _FRAGMENT_STAT_KEYS if key in snapshot
            }
        return stats_by_worker

    def _record_handle_status(
        self,
        handle: NativeTaskResultHandle,
        status: Mapping[str, Any],
        error: BaseException | None = None,
    ) -> None:
        try:
            partition = self._partition_metrics_from_handle(handle, status, error=error)
            task_id = handle.task_id
            self._progress_registry.record_partition_metrics(
                task_id.query_id,
                handle.fragment_id,
                str(task_id.partition_id),
                partition,
            )
        except Exception:
            pass

    def submit_tasks(self, tasks: Sequence[Any]) -> Sequence[NativeTaskResultHandle]:
        if self._closed:
            raise RuntimeError("native FTE worker manager is shut down")
        submit_started_at = time.monotonic()
        batch_size = len(tasks)
        submitted_count = 0
        _native_submit_debug_log(
            "submit_tasks_enter",
            batch_size=batch_size,
            worker_count=len(self._workers),
        )
        requests = [self._request_from_task(task) for task in tasks]
        for request in requests:
            self._dropped_queries.pop(_query_id_from_task_id(request.get("task_id")), None)
        self._progress_registry.register_requests(requests)
        handles: list[NativeTaskResultHandle] = []
        try:
            for task_index, request in enumerate(requests):
                worker = self._next_worker()
                task_fields = _request_debug_fields(request)
                _native_submit_debug_log(
                    "submit_task_before",
                    batch_size=batch_size,
                    task_index=task_index,
                    worker_id=worker.worker_id,
                    **task_fields,
                )
                create_started_at = time.monotonic()
                status = worker.fte_create_task(request)
                expected_task_id = FteTaskAttemptId.coerce(request.get("task_id"))
                validate_fte_status_identity(status, expected_task_id)
                create_elapsed_ms = int((time.monotonic() - create_started_at) * 1000)
                snapshot = worker.snapshot()
                _native_submit_debug_log(
                    "submit_task_after",
                    batch_size=batch_size,
                    task_index=task_index,
                    worker_id=worker.worker_id,
                    create_elapsed_ms=create_elapsed_ms,
                    status_state=status.get("state"),
                    worker_running=snapshot.get("executor_running_task_count"),
                    worker_queued=snapshot.get("executor_queued_task_count"),
                    worker_max_running=snapshot.get("executor_max_running_tasks"),
                    **task_fields,
                )
                task_id = expected_task_id
                handle = NativeTaskResultHandle(
                    worker,
                    task_id,
                    task_context=request.get("task_context") or request.get("task_context_info"),
                    request=request,
                    status_callback=self._record_handle_status,
                )
                handles.append(handle)
                submitted_count += 1
                query_id = _query_id_from_task_id(task_id)
                with self._handles_lock:
                    self._handles_by_query[query_id].append(handle)
                self._progress_registry.attach_handle(handle)
        except BaseException as exc:
            _native_submit_debug_log(
                "submit_tasks_error",
                batch_size=batch_size,
                submitted_count=submitted_count,
                elapsed_ms=int((time.monotonic() - submit_started_at) * 1000),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        _native_submit_debug_log(
            "submit_tasks_exit",
            batch_size=batch_size,
            submitted_count=submitted_count,
            elapsed_ms=int((time.monotonic() - submit_started_at) * 1000),
        )
        return handles

    def task_input_stream_exhausted(
        self,
        query_id: str,
        source_node_ids: Sequence[str],
    ) -> None:
        query_id = str(query_id)
        source_ids = [str(source_node_id) for source_node_id in source_node_ids]
        with self._handles_lock:
            handles = list(self._handles_by_query.get(query_id, []))
        _native_submit_debug_log(
            "task_input_stream_exhausted_enter",
            manager_query_id=query_id,
            source_node_ids=source_ids,
            handle_count=len(handles),
        )
        for handle_index, handle in enumerate(handles):
            task_id = handle._task_id.to_dict()
            for source_id in source_ids:
                try:
                    status = handle._worker.fte_no_more_splits(task_id, source_id)
                    info = None
                    if _native_submit_debug_enabled():
                        try:
                            info = handle.info_snapshot()
                        except BaseException as exc:
                            _native_submit_debug_log(
                                "task_input_stream_exhausted_info_error",
                                manager_query_id=query_id,
                                source_node_id=source_id,
                                handle_index=handle_index,
                                task_id=handle.fte_task_id(),
                                error_type=type(exc).__name__,
                                error=str(exc),
                            )
                    _native_submit_debug_log(
                        "task_input_stream_exhausted_no_more",
                        manager_query_id=query_id,
                        source_node_id=source_id,
                        handle_index=handle_index,
                        **_native_pending_status_fields(handle, status, info),
                    )
                except RuntimeError as exc:
                    _native_submit_debug_log(
                        "task_input_stream_exhausted_no_more_ignored",
                        manager_query_id=query_id,
                        source_node_id=source_id,
                        handle_index=handle_index,
                        task_id=handle.fte_task_id(),
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )

    def wait_query(
        self,
        query_id: str,
        timeout_s: float,
        task_context_filter: Sequence[Any] | None = None,
    ) -> Sequence[Any]:
        query_id = str(query_id)
        deadline = time.monotonic() + float(timeout_s) if timeout_s and timeout_s > 0 else None
        with self._handles_lock:
            handles = list(self._handles_by_query.get(query_id, []))
        if task_context_filter:
            allowed = {self._context_key(item) for item in task_context_filter}
            handles = [handle for handle in handles if self._context_key(handle.task_context()) in allowed]

        outputs: list[Any] = []
        pending = set(range(len(handles)))
        next_pending_debug_at = 0.0
        while pending:
            for index in list(pending):
                poll = handles[index].poll()
                if poll.state is TaskResultState.NOT_READY:
                    continue
                pending.remove(index)
                if poll.state is TaskResultState.ERROR:
                    raise RuntimeError(f"native FTE query {query_id} failed") from poll.error
                if poll.state is TaskResultState.MATERIALIZED_OUTPUT:
                    outputs.append(poll.output)
                handles[index].ack()
                handles[index].release_result_payload()
            if not pending:
                break
            if _native_submit_debug_enabled():
                now = time.monotonic()
                if now >= next_pending_debug_at:
                    for index in sorted(pending):
                        handle = handles[index]
                        try:
                            status = handle.status_snapshot()
                        except BaseException as exc:
                            _native_submit_debug_log(
                                "wait_query_pending_status_error",
                                query_id=query_id,
                                handle_index=index,
                                task_id=handle.fte_task_id(),
                                error_type=type(exc).__name__,
                                error=str(exc),
                            )
                            continue
                        try:
                            info = handle.info_snapshot()
                        except BaseException as exc:
                            info = None
                            _native_submit_debug_log(
                                "wait_query_pending_info_error",
                                query_id=query_id,
                                handle_index=index,
                                task_id=handle.fte_task_id(),
                                error_type=type(exc).__name__,
                                error=str(exc),
                            )
                        _native_submit_debug_log(
                            "wait_query_pending_status",
                            manager_query_id=query_id,
                            handle_index=index,
                            pending_count=len(pending),
                            **_native_pending_status_fields(handle, status, info),
                        )
                    next_pending_debug_at = now + 5.0
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for native FTE query {query_id}")
            time.sleep(0.01)
        return outputs

    @staticmethod
    def _source_ids_from_request(request: Mapping[str, Any], *keys: str) -> list[str]:
        values: list[str] = []
        for key in keys:
            raw = request.get(key)
            if raw is None:
                continue
            if isinstance(raw, str):
                items = [item.strip() for item in raw.split(",") if item.strip()]
            elif isinstance(raw, Mapping):
                items = [str(item) for item in raw]
            else:
                try:
                    items = [str(item) for item in raw]
                except TypeError:
                    items = [str(raw)]
            values.extend(item for item in items if item)
        return sorted(set(values))

    @staticmethod
    def _context_source_ids(request: Mapping[str, Any], key: str) -> list[str]:
        context = request.get("context")
        if not isinstance(context, Mapping):
            return []
        raw = context.get(key)
        if raw is None:
            return []
        if isinstance(raw, str):
            return [item.strip() for item in raw.split(",") if item.strip()]
        try:
            return [str(item) for item in raw]
        except TypeError:
            return [str(raw)]

    @staticmethod
    def _progress_stats_from_status(status: Mapping[str, Any]) -> dict[str, Any]:
        stats = dict(status.get("task_stats") or {}) if isinstance(status.get("task_stats"), Mapping) else {}
        for key in (
            "submitted_split_count",
            "submitted_split_count_by_source",
            "queued_split_count",
            "queued_split_count_by_source",
            "consumed_split_count",
            "consumed_split_count_by_source",
            "completed_split_count",
            "completed_split_count_by_source",
            "submitted_split_bytes",
            "submitted_split_bytes_by_source",
            "queued_split_bytes",
            "queued_split_bytes_by_source",
            "consumed_split_bytes",
            "consumed_split_bytes_by_source",
            "completed_split_bytes",
            "completed_split_bytes_by_source",
            "submitted_input_rows",
            "submitted_input_rows_by_source",
            "submitted_input_bytes",
            "submitted_input_bytes_by_source",
            "consumed_input_rows",
            "consumed_input_rows_by_source",
            "consumed_input_bytes",
            "consumed_input_bytes_by_source",
            "completed_input_rows",
            "completed_input_rows_by_source",
            "completed_input_bytes",
            "completed_input_bytes_by_source",
            "queue_wait_ms",
            "queue_wait_ms_by_source",
        ):
            if key in status and key not in stats:
                stats[key] = status[key]
        return stats

    @classmethod
    def _partition_metrics_from_handle(
        cls,
        handle: NativeTaskResultHandle,
        status: Mapping[str, Any],
        *,
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        task_id = handle.task_id
        state = str(status.get("state") or "").upper()
        failed = error is not None or state in {"FAILED", "CANCELED", "ABORTED"}
        finished = state == FteTaskState.FINISHED.value
        running = state == FteTaskState.RUNNING.value
        waiting = state in {"", FteTaskState.PLANNED.value, FteTaskState.QUEUED.value}
        progress_stats = cls._progress_stats_from_status(status)
        failure = status.get("failure")
        if error is not None:
            failure = {"type": type(error).__name__, "message": str(error)}
        running_attempts = []
        if running:
            running_attempts.append(
                {
                    "attempt_id": str(task_id),
                    "attempt": task_id.to_dict(),
                    "worker_id": str(handle.worker_id),
                    **({"task_stats": progress_stats} if progress_stats else {}),
                }
            )
        output_stats = (
            dict(status.get("spooling_output_stats"))
            if isinstance(status.get("spooling_output_stats"), Mapping)
            else status.get("spooling_output_stats")
        )
        selected_output_stats: Any = None
        if finished:
            selected_output_stats = progress_stats or output_stats

        initial_split_count_by_source = {}
        for key in (
            "submitted_split_count_by_source",
            "queued_split_count_by_source",
            "consumed_split_count_by_source",
        ):
            value = status.get(key)
            if isinstance(value, Mapping):
                initial_split_count_by_source.update({str(source): int(count or 0) for source, count in value.items()})

        return {
            "task_id": str(task_id.task_id),
            "task": task_id.task_id.to_dict(),
            "partition_id": int(task_id.partition_id),
            "state": "FAILED" if failed else ("FINISHED" if finished else ("RUNNING" if running else "SEALED")),
            "execution_class": str(handle.request.get("execution_class") or "STANDARD"),
            "sealed": True,
            "ready_for_scheduling": waiting,
            "execution_ready_deferred": False,
            "waiting_for_node": False,
            "waiting_for_execution": waiting,
            "remaining_attempts": 0 if finished or failed else 1,
            "max_attempts": 1,
            "memory_requirement_bytes": status.get("memory_requirement_bytes"),
            "owner_worker_id": str(handle.worker_id),
            "pending_worker_reservation": False,
            "pending_worker_reservation_done": False,
            "pending_worker_reservation_generation": None,
            "running_attempts": running_attempts,
            "running_count": 1 if running else 0,
            "selected_attempt": int(task_id.attempt_id) if finished else None,
            "selected_output_stats": selected_output_stats,
            "finished_attempts": [int(task_id.attempt_id)] if finished else [],
            "failure_observed": failed,
            "failure_count": 1 if failed else 0,
            "failures": [failure] if failed and failure is not None else [],
            "initial_split_count_by_source": initial_split_count_by_source,
            "no_more_splits": list(status.get("no_more_splits") or []),
        }

    def fte_query_status(self, query_id: str) -> dict[str, Any]:
        query_id = str(query_id)
        with self._handles_lock:
            handles = list(self._handles_by_query.get(query_id, []))
        dropped_query = self._dropped_queries.get(query_id)
        if not handles and dropped_query is not None:
            return {
                "query_id": query_id,
                "fragment_execution_count": 0,
                "partition_count": int(dropped_query.get("removed") or 0),
                "running_count": 0,
                "failed_count": 0,
                "finished_count": 0,
                "pending_submission_count": 0,
                "failed": False,
                "finished": False,
                "canceled": True,
                "selected_attempt_task_ids": [],
                "fragment_executions": {},
                "failed_partitions": [],
                "scheduler_state": "CANCELED",
                "drop_summary": dict(dropped_query),
            }
        partition_metrics: dict[tuple[str, str, str], dict[str, Any]] = {}
        selected_attempt_task_ids: list[str] = []
        failed_partitions: list[dict[str, Any]] = []
        for handle in handles:
            status_error: BaseException | None = None
            try:
                status = handle.status_snapshot()
            except BaseException as exc:
                status = {"state": FteTaskState.FAILED.value}
                status_error = exc
            partition = self._partition_metrics_from_handle(handle, status, error=status_error)
            partition_metrics[(query_id, handle.fragment_id, str(handle.task_id.partition_id))] = partition
            if partition["state"] == "FAILED":
                failed_partitions.append(
                    {
                        "task_id": handle.fte_task_id(),
                        "latest_failure": repr(status_error)
                        if status_error is not None
                        else str(status.get("failure")),
                    }
                )
            elif partition["state"] == "FINISHED":
                selected_attempt_task_ids.append(handle.fte_task_id())
        return self._progress_registry.query_status(
            query_id,
            partition_metrics=partition_metrics,
            failed_partitions=failed_partitions,
            selected_attempt_task_ids=selected_attempt_task_ids,
            result_handle_count=len(handles),
        )

    def wait_fte_query(self, query_id: str, timeout_s: float = 0.0) -> dict[str, Any]:
        query_id = str(query_id)
        deadline = time.monotonic() + float(timeout_s) if timeout_s and timeout_s > 0 else None
        while True:
            status = self.fte_query_status(query_id)
            if bool(status.get("failed")):
                raise RuntimeError(f"native FTE query {query_id} failed: {status}")
            if bool(status.get("canceled")):
                raise RuntimeError(f"native FTE query {query_id} canceled: {status}")
            if bool(status.get("finished")):
                return status
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for native FTE query {query_id}: {status}")
            time.sleep(0.01)

    def pop_fte_result_handles(self, query_id: str) -> list[NativeTaskResultHandle]:
        query_id = str(query_id)
        with self._handles_lock:
            has_handles = bool(self._handles_by_query.get(query_id))
        if has_handles:
            try:
                self.fte_query_status(query_id)
            except Exception:
                pass
        with self._handles_lock:
            return list(self._handles_by_query.pop(query_id, []))

    def drop_query(self, query_id: str) -> None:
        query_id = str(query_id)
        with self._handles_lock:
            self._handles_by_query.pop(query_id, None)
        worker_errors: list[str] = []
        try:
            self._progress_registry.drop_query(query_id)
        except BaseException as exc:
            worker_errors.append(f"progress_registry: {type(exc).__name__}: {exc}")
        removed = 0
        canceled = 0
        for worker in self._workers:
            worker_id = str(getattr(worker, "worker_id", "") or "<unknown>")
            try:
                result = worker.fte_drop_query(query_id)
                removed += int(result.get("removed") or result.get("tasks_removed") or 0)
                canceled += int(result.get("canceled") or result.get("tasks_canceled") or 0)
            except BaseException as exc:
                worker_errors.append(f"{worker_id}: {type(exc).__name__}: {exc}")
        self._dropped_queries[query_id] = {
            "removed": removed,
            "canceled": canceled,
            "worker_errors": worker_errors,
        }
        if worker_errors:
            raise RuntimeError(f"native FTE query teardown failed for {query_id}: " + "; ".join(worker_errors))

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_debug_sampler()
        for worker in self._workers:
            worker.shutdown()

    def _start_debug_sampler(self) -> None:
        if self._debug_sampler_thread is not None:
            return
        thread = threading.Thread(
            target=self._debug_sampler_loop,
            name="native-fte-debug-sampler",
            daemon=True,
        )
        self._debug_sampler_thread = thread
        thread.start()

    def _stop_debug_sampler(self) -> None:
        self._debug_sampler_stop.set()
        thread = self._debug_sampler_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _active_handle_snapshot(self) -> list[tuple[str, int, NativeTaskResultHandle]]:
        with self._handles_lock:
            return [
                (query_id, index, handle)
                for query_id, handles in self._handles_by_query.items()
                for index, handle in enumerate(list(handles))
            ]

    def _debug_sampler_loop(self) -> None:
        while not self._debug_sampler_stop.wait(5.0):
            self._dump_active_task_statuses()

    def _dump_active_task_statuses(self) -> None:
        if not _native_submit_debug_enabled():
            return
        for query_id, index, handle in self._active_handle_snapshot():
            try:
                status = handle.status_snapshot()
            except BaseException as exc:
                _native_submit_debug_log(
                    "active_task_status_error",
                    query_id=query_id,
                    handle_index=index,
                    task_id=handle.fte_task_id(),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                continue
            state = str(status.get("state") or "").upper()
            if state in _TERMINAL_STATE_VALUES:
                continue
            try:
                info = handle.info_snapshot()
            except BaseException as exc:
                info = None
                _native_submit_debug_log(
                    "active_task_info_error",
                    query_id=query_id,
                    handle_index=index,
                    task_id=handle.fte_task_id(),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            _native_submit_debug_log(
                "active_task_status",
                manager_query_id=query_id,
                handle_index=index,
                **_native_pending_status_fields(handle, status, info),
            )

    def _next_worker(self) -> NativeWorkerHandle:
        worker = self._workers[self._next_worker_index % len(self._workers)]
        self._next_worker_index += 1
        return worker

    @staticmethod
    def _request_from_task(task: Any) -> dict[str, Any]:
        if isinstance(task, Mapping):
            return dict(task)
        to_fte_request = getattr(task, "to_fte_request", None)
        if callable(to_fte_request):
            request = to_fte_request()
            if not isinstance(request, Mapping):
                raise TypeError("to_fte_request must return a mapping")
            return dict(request)
        if all(callable(getattr(task, attr, None)) for attr in ("context", "task_context", "Inputs", "plan")):
            return NativeFteWorkerManagerBackend._request_from_worker_task(task)
        raise TypeError(f"unsupported native FTE task payload: {type(task).__name__}")

    @staticmethod
    def _request_from_worker_task(task: Any) -> dict[str, Any]:
        context = dict(task.context() or {})
        query_id = str(context.get("query_id") or "").strip()
        if not query_id:
            raise ValueError("native FTE worker task requires non-empty query_id")
        task_context_info = dict(task.task_context() or {})
        partition_id = int(task_context_info.get("task_id") or context.get("task_id") or 0)
        fragment_execution_id = int(context.get("fragment_execution_id") or 0)
        attempt_id = int(context.get("attempt_id") or 0)

        for node_id, entry in dict(task.Inputs() or {}).items():
            if not isinstance(entry, Mapping):
                continue
            source_node_id = str(node_id)
            kind = str(entry.get("kind") or "")
            data = entry.get("data")
            if kind == "scan_task":
                context[f"scan_task:{source_node_id}"] = data
            elif kind == "exchange_source_task":
                context[f"exchange_source_task:{source_node_id}"] = data
            else:
                raise ValueError(f"unsupported native FTE task input kind: {kind!r}")

        exchange_sink_instance = None
        exchange_sink_instance_fn = getattr(task, "exchange_sink_instance", None)
        if callable(exchange_sink_instance_fn):
            exchange_sink_instance = exchange_sink_instance_fn()
            if exchange_sink_instance is not None:
                try:
                    exchange_sink_instance = dict(exchange_sink_instance)
                except (TypeError, ValueError):
                    pass

        node_name = str(context.get("node_name") or task.name() or "fragment")
        node_id = str(context.get("node_id") or task_context_info.get("last_node_id") or partition_id)
        fragment_id = str(context.get("fragment_id") or f"{query_id}:{node_name}:{node_id}")
        next_sequence_by_source: dict[tuple[str, str, str], int] = defaultdict(int)

        def next_split_sequence(split_query_id: str, split_fragment_id: str, source_node_id: str) -> int:
            key = (str(split_query_id), str(split_fragment_id), str(source_node_id))
            sequence_id = next_sequence_by_source[key]
            next_sequence_by_source[key] = sequence_id + 1
            return sequence_id

        prepared_inputs = prepare_fte_dynamic_inputs(
            context=context,
            query_id=query_id,
            fragment_id=fragment_id,
            next_split_sequence=next_split_sequence,
        )
        context = strip_fte_dynamic_context(
            context,
            prepared_inputs.dynamic_scan_sources,
            prepared_inputs.dynamic_exchange_sources,
        )
        initial_splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for split in prepared_inputs.splits:
            initial_splits[split.source_node_id].append(split.to_dict())
        source_node_ids = prepared_inputs.dynamic_scan_sources | prepared_inputs.dynamic_exchange_sources

        return {
            "query_id": query_id,
            "fragment_id": fragment_id,
            "task_id": FteTaskAttemptId(
                FteTaskId(query_id, fragment_execution_id, partition_id),
                attempt_id,
            ).to_dict(),
            "task_context": task_context_info,
            "task_context_info": task_context_info,
            "context": context,
            "initial_splits": dict(initial_splits),
            "no_more_splits": [],
            "source_node_ids": sorted(source_node_ids),
            "dynamic_scan_source_node_ids": sorted(prepared_inputs.dynamic_scan_sources),
            "dynamic_exchange_source_node_ids": sorted(prepared_inputs.dynamic_exchange_sources),
            "fragment_plan": task.plan(),
            "exchange_sink_instance": exchange_sink_instance,
            "worker_runtime": FTE_WORKER_RUNTIME,
        }

    @staticmethod
    def materialize_task_context(
        request: Mapping[str, Any], *, merge_scan_task_descriptors: Callable[[list[Any]], Any]
    ) -> dict[str, Any]:
        return materialize_task_inputs(
            request.get("context"),
            request.get("initial_splits"),
            merge_scan_task_descriptors=merge_scan_task_descriptors,
        )

    @staticmethod
    def _context_key(value: Any) -> Any:
        if isinstance(value, Mapping):
            return tuple(sorted((str(key), value[key]) for key in value))
        return value
