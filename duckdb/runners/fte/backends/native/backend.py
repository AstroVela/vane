# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import math
import os
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any

from duckdb.runners.fte.backend import TaskResultPoll, TaskResultState
from duckdb.runners.fte.dynamic_inputs import (
    prepare_fte_dynamic_inputs,
    strip_fte_dynamic_context,
)
from duckdb.runners.fte.fte_config import FTE_WORKER_RUNTIME, FteWorkerAdmissionConfig
from duckdb.runners.fte.fte_exchange import derive_exchange_sink_instance_for_attempt
from duckdb.runners.fte.fte_failures import _failure_payload, _normalize_failure_payload
from duckdb.runners.fte.fte_state import FteTaskState
from duckdb.runners.fte.fte_types import (
    FteTaskAttemptId,
    FteTaskId,
    validate_fte_status_identity,
)
from duckdb.runners.fte.fte_worker_runtime import FteWorkerTaskManager, materialize_task_inputs
from duckdb.runners.progress import validate_pipeline_topology

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

_NATIVE_STABLE_TASK_IDENTITY_KEY = "_native_stable_task_identity_key"
_NATIVE_STABLE_TASK_IDENTITY_MASK = (1 << 63) - 1


def _stable_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"bytes": bytes(value).hex()}
    if isinstance(value, Mapping):
        return {
            str(key): _stable_json_value(item) for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, Sequence):
        return [_stable_json_value(item) for item in value]
    raise TypeError(f"unsupported stable native FTE identity value: {type(value).__name__}")


def _exchange_source_logical_identity(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        partition_indices = [int(item) for item in value.get("partition_indices") or ()]
        source_task_partition_ids = value.get("source_task_partition_ids")
        if source_task_partition_ids is None:
            handles = value.get("source_handles") or ()
            source_task_partition_ids = [
                handle.get("source_task_partition_id")
                for handle in handles
                if isinstance(handle, Mapping) and handle.get("source_task_partition_id") is not None
            ]
        source_task_partition_ids = sorted({int(item) for item in source_task_partition_ids or ()})
        source_partition_count = int(value.get("source_partition_count") or 0)
        source_task_count = int(value.get("source_task_count") or 0)
        replicated = bool(value.get("replicated"))
    else:
        import duckdb

        metadata = dict(duckdb.ray_cxx.exchange_source_task_logical_identity(value))
        partition_indices = [int(item) for item in metadata.get("partition_indices") or ()]
        source_task_partition_ids = sorted({int(item) for item in metadata.get("source_task_partition_ids") or ()})
        source_partition_count = int(metadata.get("source_partition_count") or 0)
        source_task_count = int(metadata.get("source_task_count") or 0)
        replicated = bool(metadata.get("replicated"))

    if not source_task_partition_ids:
        raise ValueError("exchange source task is missing stable source task partition identities")
    if not partition_indices:
        raise ValueError("exchange source task is missing partition indices for stable task identity")
    return {
        "partition_indices": partition_indices,
        "source_task_partition_ids": source_task_partition_ids,
        "source_partition_count": source_partition_count,
        "source_task_count": source_task_count,
        "replicated": replicated,
    }


def _stable_native_fte_task_identity(
    task_inputs: Mapping[Any, Any],
    task_context_info: Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[int, str]:
    logical_inputs: list[Any] = []
    for node_id, raw_entry in sorted(task_inputs.items(), key=lambda entry: str(entry[0])):
        if not isinstance(raw_entry, Mapping):
            raise TypeError("native FTE task inputs must be mappings")
        kind = str(raw_entry.get("kind") or "")
        data = raw_entry.get("data")
        if kind == "scan_task":
            if isinstance(data, (bytes, bytearray, memoryview)):
                import duckdb

                source_task_partition_id = int(duckdb.ray_cxx.scan_task_source_partition_id(bytes(data)))
                scan_descriptor_identity = hashlib.blake2b(bytes(data), digest_size=32).hexdigest()
            elif isinstance(data, Mapping):
                raw_source_task_partition_id = data.get("source_task_partition_id")
                if raw_source_task_partition_id is None:
                    raise ValueError("scan task is missing its stable source task partition identity")
                source_task_partition_id = int(raw_source_task_partition_id)
                scan_descriptor_identity = _stable_json_value(data)
            else:
                raise TypeError("native FTE scan task data must be serialized bytes or a mapping")
            logical_inputs.append(
                [
                    str(node_id),
                    kind,
                    {
                        "source_task_partition_id": source_task_partition_id,
                        "descriptor": scan_descriptor_identity,
                    },
                ]
            )
        elif kind == "exchange_source_task":
            logical_inputs.append([str(node_id), kind, _exchange_source_logical_identity(data)])
        else:
            raise ValueError(f"unsupported native FTE task input kind: {kind!r}")

    lineage = {
        "last_node_id": int(task_context_info.get("last_node_id") or 0),
        "node_ids": [int(node_id) for node_id in task_context_info.get("node_ids") or ()],
    }
    if not logical_inputs:
        stable_partition_id = context.get("stable_task_partition_id")
        if stable_partition_id is None:
            stable_partition_id = task_context_info.get("task_id")
        lineage["source_free_task_id"] = int(stable_partition_id or 0)
    identity_key = json.dumps(
        ["native-fte-logical-task-v1", lineage, logical_inputs],
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.blake2b(
        identity_key.encode("utf-8"),
        digest_size=8,
        person=b"vane-fte-task-v1",
    ).digest()
    identity = int.from_bytes(digest, "big") & _NATIVE_STABLE_TASK_IDENTITY_MASK
    return identity, identity_key


async def _to_thread_with_owned_side_effects(
    callback: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Do not expose cancellation until a synchronous task mutation has finished."""
    thread_task = asyncio.create_task(asyncio.to_thread(callback, *args, **kwargs))
    cancellation: asyncio.CancelledError | None = None
    while not thread_task.done():
        try:
            await asyncio.shield(thread_task)
        except asyncio.CancelledError as error:
            cancellation = error
    result = thread_task.result()
    if cancellation is not None:
        raise cancellation
    return result


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
    _NEW = "NEW"
    _STARTING = "STARTING"
    _RUNNING = "RUNNING"
    _STOPPING = "STOPPING"
    _CLOSED = "CLOSED"
    _FAILED = "FAILED"

    def __init__(
        self,
        thread_name: str,
        *,
        start_timeout_s: float = 5.0,
        operation_timeout_s: float = 30.0,
    ) -> None:
        self._thread_name = thread_name
        self._start_timeout_s = float(start_timeout_s)
        self._operation_timeout_s = float(operation_timeout_s)
        if not math.isfinite(self._start_timeout_s) or self._start_timeout_s <= 0:
            raise ValueError("native FTE event-loop start timeout must be finite and positive")
        if not math.isfinite(self._operation_timeout_s) or self._operation_timeout_s <= 0:
            raise ValueError("native FTE event-loop operation timeout must be finite and positive")
        self._condition = threading.Condition(threading.RLock())
        self._state = self._NEW
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None
        self._pending_futures: set[Future[Any]] = set()

    @staticmethod
    def _close_awaitable(awaitable: Awaitable[Any]) -> None:
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()

    def _raise_unavailable_locked(self) -> None:
        failure = self._failure
        if failure is not None:
            raise RuntimeError("native FTE event loop failed") from failure
        if self._state == self._STOPPING:
            raise RuntimeError("native FTE event loop is stopping")
        raise RuntimeError("native FTE event loop is closed")

    def _thread_main(self) -> None:
        loop: asyncio.AbstractEventLoop | None = None
        failure: BaseException | None = None
        should_run = False
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            with self._condition:
                if self._state == self._STARTING:
                    self._loop = loop
                    self._state = self._RUNNING
                    should_run = True
                self._condition.notify_all()

            if should_run:
                loop.run_forever()
                with self._condition:
                    if self._state == self._RUNNING:
                        failure = RuntimeError("native FTE event loop stopped unexpectedly")
                        self._failure = failure
                    self._state = self._STOPPING
                    pending_futures = tuple(self._pending_futures)
                    self._condition.notify_all()
                for future in pending_futures:
                    future.cancel()
        except BaseException as exc:
            failure = exc
            with self._condition:
                self._failure = exc
                self._state = self._STOPPING
                pending_futures = tuple(self._pending_futures)
                self._condition.notify_all()
            for future in pending_futures:
                future.cancel()
        finally:
            if loop is not None:
                try:
                    pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
                    for task in pending_tasks:
                        task.cancel()
                    if pending_tasks:
                        loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                try:
                    # asyncio task cancellation does not stop work that has
                    # already entered a synchronous native call through
                    # asyncio.to_thread(). Join that owned executor before the
                    # loop is considered closed so callers have a real native
                    # execution barrier.
                    loop.run_until_complete(loop.shutdown_default_executor())
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                try:
                    loop.close()
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                asyncio.set_event_loop(None)
            with self._condition:
                self._loop = None
                self._pending_futures.clear()
                if failure is not None:
                    self._failure = failure
                self._state = self._FAILED if self._failure is not None else self._CLOSED
                self._condition.notify_all()

    def start(self) -> None:
        deadline = time.monotonic() + self._start_timeout_s
        with self._condition:
            if self._state == self._NEW:
                self._state = self._STARTING
                thread = threading.Thread(
                    target=self._thread_main,
                    name=self._thread_name,
                    daemon=True,
                )
                self._thread = thread
                try:
                    thread.start()
                except BaseException as exc:
                    self._failure = exc
                    self._state = self._FAILED
                    self._condition.notify_all()
                    raise RuntimeError("failed to start native FTE event loop") from exc
            while self._state == self._STARTING:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    failure = TimeoutError("native FTE event-loop startup timed out")
                    self._failure = failure
                    self._state = self._STOPPING
                    self._condition.notify_all()
                    raise RuntimeError("failed to start native FTE event loop") from failure
                self._condition.wait(remaining)
            if self._state == self._RUNNING:
                return
            self._raise_unavailable_locked()

    def _discard_future(self, future: Future[Any]) -> None:
        with self._condition:
            self._pending_futures.discard(future)

    def submit(self, coro: Coroutine[Any, Any, Any]) -> Future[Any]:
        try:
            self.start()
            with self._condition:
                if self._state != self._RUNNING or self._loop is None:
                    self._raise_unavailable_locked()
                loop = self._loop
                assert loop is not None
                future: Future[Any] = asyncio.run_coroutine_threadsafe(coro, loop)
                self._pending_futures.add(future)
                future.add_done_callback(self._discard_future)
                return future
        except BaseException:
            self._close_awaitable(coro)
            raise

    def run(self, coro: Coroutine[Any, Any, Any], timeout_s: float | None = None) -> Any:
        timeout_s = self._operation_timeout_s if timeout_s is None else float(timeout_s)
        if not math.isfinite(timeout_s) or timeout_s < 0:
            self._close_awaitable(coro)
            raise ValueError("native FTE event-loop operation timeout must be finite and non-negative")
        future = self.submit(coro)
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeoutError as exc:
            if future.done():
                return future.result()
            future.cancel()
            raise TimeoutError(f"native FTE event-loop operation timed out after {timeout_s:.3f}s") from exc

    def run_owned_side_effects(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Wait without a fixed timeout until owned side effects are terminal."""
        return self.submit(coro).result()

    def request_shutdown(self) -> None:
        """Fence new work and ask the loop thread to begin teardown."""
        with self._condition:
            if self._state == self._CLOSED:
                return
            if self._state == self._NEW:
                self._state = self._CLOSED
                self._condition.notify_all()
                return
            if self._state == self._FAILED:
                self._state = self._CLOSED
                self._condition.notify_all()
            elif self._state in {self._STARTING, self._RUNNING}:
                self._state = self._STOPPING
                loop = self._loop
                if loop is not None:
                    try:
                        loop.call_soon_threadsafe(loop.stop)
                    except BaseException as exc:
                        self._failure = exc
                self._condition.notify_all()

    def shutdown(self, timeout_s: float = 5.0) -> None:
        timeout_s = float(timeout_s)
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("native FTE event-loop shutdown timeout must be finite and non-negative")
        self.request_shutdown()
        with self._condition:
            thread = self._thread
        if thread is None or thread is threading.current_thread():
            return
        if thread.is_alive():
            thread.join(timeout=timeout_s)
        if thread.is_alive():
            raise RuntimeError(f"native FTE event loop did not stop within {timeout_s:.3f}s")
        with self._condition:
            if self._state in {self._STOPPING, self._FAILED}:
                self._state = self._CLOSED
                self._condition.notify_all()


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


_TaskContextKey = tuple[int, int, int, tuple[int, ...]]
_TASK_CONTEXT_FIELDS = ("query_idx", "last_node_id", "task_id", "node_ids")


def _task_context_key(value: Any) -> _TaskContextKey | None:
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
    # The current C++ task-result classes live under duckdb.ray_cxx even when
    # used by the Ray-free native backend. This imports the compiled binding,
    # not the Ray Python runtime.
    from duckdb._ray_cxx import require_ray_cxx_attr

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
    payloads, metadatas, result_schema, stats, flight_port, exchange_sink_instance = _native_result_tuple(value)
    partition_refs = []
    for index, payload in enumerate(payloads or []):
        if index < len(metadatas or []):
            _metadata_rows_bytes(metadatas[index])
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

    def _failure_status(self, error: BaseException) -> dict[str, Any]:
        failure = _failure_payload("NATIVE_BACKEND_ERROR", error)
        failure["type"] = type(error).__name__
        return {
            "state": FteTaskState.FAILED.value,
            "task_id": self._task_id.to_dict(),
            "task_id_string": str(self._task_id),
            "failure": failure,
        }

    def _validated_status(self, status: Any, *, operation: str) -> dict[str, Any]:
        if not isinstance(status, Mapping):
            raise TypeError(f"{operation} must return a status mapping")
        result = dict(status)
        validate_fte_status_identity(result, self._task_id)
        raw_state = result.get("state")
        state = raw_state.value if isinstance(raw_state, FteTaskState) else str(raw_state or "").upper()
        if state in {
            FteTaskState.FAILED.value,
            FteTaskState.CANCELED.value,
            FteTaskState.ABORTED.value,
        }:
            result["failure"] = _normalize_failure_payload(result.get("failure"))
        return result

    def status_snapshot(self) -> dict[str, Any]:
        try:
            status = self._validated_status(
                self._worker.fte_get_task_status_cached(self._task_id.to_dict()),
                operation="fte_get_task_status_cached",
            )
        except BaseException as exc:
            self._record_status(self._failure_status(exc), exc)
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
            self._record_status(self._failure_status(exc), exc)
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
            self._record_status(self._failure_status(exc), exc)
            return TaskResultPoll(TaskResultState.ERROR, error=exc)

        self._record_status(status)
        raw_state = status.get("state")
        state = raw_state.value if isinstance(raw_state, FteTaskState) else str(raw_state or "").upper()
        if state not in _TERMINAL_STATE_VALUES:
            return TaskResultPoll(TaskResultState.NOT_READY)
        if state == FteTaskState.FINISHED.value:
            result = status.get("result")
            if result is None:
                return TaskResultPoll(TaskResultState.NO_OUTPUT)
            return TaskResultPoll(TaskResultState.MATERIALIZED_OUTPUT, output=result)
        failure = _normalize_failure_payload(status.get("failure"))
        message = failure.get("message")
        return TaskResultPoll(
            TaskResultState.ERROR,
            error=RuntimeError(message or f"native FTE task {self._task_id} ended with {failure['error_code']}"),
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
    progress_topology_unavailable: bool = False


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
        task_context_filter: set[_TaskContextKey] | None = None,
    ) -> dict[str, Any]:
        query_id = str(query_id)
        partition_metrics = partition_metrics or {}
        failed_partition_items = [dict(item) for item in failed_partitions or []]
        selected_attempt_ids = {str(task_id) for task_id in selected_attempt_task_ids or []}

        with self._lock:
            fragments = self._fragments_by_query.get(query_id, {})
            fragment_executions: dict[str, dict[str, Any]] = {}
            global_failed = False
            for fragment_id, registered in fragments.items():
                partitions: dict[str, dict[str, Any]] = {}
                for partition_id, registered_partition in registered.partitions.items():
                    metrics_key = (query_id, fragment_id, partition_id)
                    metrics = partition_metrics.get(metrics_key)
                    if metrics is not None:
                        registered_partition.last_metrics = dict(metrics)
                    metrics = dict(
                        registered_partition.last_metrics or self._placeholder_partition_metrics(registered_partition)
                    )
                    self._merge_fragment_progress_topology_locked(registered, metrics)
                    global_failed = global_failed or str(metrics.get("state") or "").upper() == "FAILED"
                    if task_context_filter is not None:
                        request = registered_partition.request
                        task_context = request.get("task_context_info")
                        if task_context is None:
                            task_context = request.get("task_context")
                        if _task_context_key(task_context) not in task_context_filter:
                            continue
                    partitions[partition_id] = metrics
                    if str(metrics.get("state") or "").upper() == "FINISHED":
                        selected_attempt_ids.add(str(registered_partition.task_id))

                if not partitions:
                    continue
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
        failed = global_failed
        finished = bool(fragment_executions) and all(fragment["finished"] for fragment in fragment_executions.values())
        return {
            "query_id": query_id,
            "matched": bool(fragment_executions),
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
            raw_operator_details = raw_pipeline["operator_details"]
            if type(raw_operator_details) is not list:
                raise TypeError("native progress operator_details must be a list")
            operator_details: list[dict[str, Any]] = []
            for raw_details in raw_operator_details:
                if not isinstance(raw_details, Mapping):
                    raise TypeError("native progress operator details must be mappings")
                # Live native stats include counters in operator_details. Those
                # values evolve during execution and are not topology. Keep
                # only the fields consumed as stable display identity.
                operator_details.append(
                    {key: raw_details[key] for key in ("udf_name", "pipeline_role") if key in raw_details}
                )
            pipelines.append(
                {
                    "pipeline_id": raw_pipeline["pipeline_id"],
                    "operators": raw_pipeline["operators"],
                    "operator_details": operator_details,
                }
            )
        return validate_pipeline_topology({"schema": "pipeline_topology", "pipelines": pipelines})

    @staticmethod
    def _merge_progress_topologies(
        current: Mapping[str, Any],
        incoming: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        merged = copy.deepcopy(dict(current))
        pipelines = merged["pipelines"]
        pipelines_by_id = {pipeline["pipeline_id"]: pipeline for pipeline in pipelines}
        for incoming_pipeline in incoming["pipelines"]:
            pipeline_id = incoming_pipeline["pipeline_id"]
            current_pipeline = pipelines_by_id.get(pipeline_id)
            if current_pipeline is None:
                copied = copy.deepcopy(incoming_pipeline)
                pipelines.append(copied)
                pipelines_by_id[pipeline_id] = copied
                continue
            if current_pipeline["operators"] != incoming_pipeline["operators"]:
                return None
            for current_details, incoming_details in zip(
                current_pipeline["operator_details"],
                incoming_pipeline["operator_details"],
                strict=True,
            ):
                for key, value in incoming_details.items():
                    if key in current_details and current_details[key] != value:
                        return None
                    current_details.setdefault(key, copy.deepcopy(value))
        return validate_pipeline_topology(merged)

    @classmethod
    def _merge_fragment_progress_topology_locked(
        cls,
        fragment: _NativeFteRegisteredFragment,
        metrics: Mapping[str, Any],
    ) -> None:
        if fragment.progress_topology_unavailable:
            return
        try:
            for task_stats in cls._task_stats_from_partition_metrics(metrics):
                topology = cls._topology_from_task_stats(task_stats)
                if topology is None:
                    continue
                if fragment.progress_topology is None:
                    fragment.progress_topology = topology
                    continue
                merged = cls._merge_progress_topologies(fragment.progress_topology, topology)
                if merged is None:
                    fragment.progress_topology = None
                    fragment.progress_topology_unavailable = True
                    return
                fragment.progress_topology = merged
        except Exception:
            # Progress is observational. Malformed or genuinely conflicting
            # snapshots must not break fte_query_status, which is also the
            # native completion-detection path.
            fragment.progress_topology = None
            fragment.progress_topology_unavailable = True

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
        return await _to_thread_with_owned_side_effects(self._execute_fn, request_payload)

    def fte_create_task(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return _as_status("fte_create_task", self._loop.run(self._manager.create_task(dict(request))))

    def fte_add_splits(
        self,
        task_id: str | Mapping[str, Any],
        source_node_id: str,
        splits: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        split_payloads: list[Mapping[str, Any]] = [dict(split) for split in splits]
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
        loop_timeout_s = None if timeout_s is None else max(5.0, float(timeout_s) + 5.0)
        return _as_status(
            "fte_wait_task_status",
            self._loop.run(
                self._manager.wait_task_status(task_id, min_version, timeout_s),
                timeout_s=loop_timeout_s,
            ),
        )

    def fte_wait_split_queue_has_space(
        self,
        task_id: str | Mapping[str, Any],
        source_node_id: str | None = None,
        max_buffered_splits: int | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        loop_timeout_s = None if timeout_s is None else max(5.0, float(timeout_s) + 5.0)
        return _as_status(
            "fte_wait_split_queue_has_space",
            self._loop.run(
                self._manager.wait_split_queue_has_space(
                    task_id,
                    source_node_id,
                    max_buffered_splits,
                    timeout_s,
                ),
                timeout_s=loop_timeout_s,
            ),
        )

    def fte_get_task_info(self, task_id: str | Mapping[str, Any]) -> dict[str, Any]:
        return _as_status("fte_get_task_info", self._loop.run(self._manager.get_task_info(task_id)))

    def fte_cancel_task(self, task_id: str | Mapping[str, Any]) -> dict[str, Any]:
        # A successful cancellation is a barrier for task-owned writes. The
        # normal operation timeout must not expose a still-running writer.
        return self.resolve_fte_cancel_task(self.enqueue_fte_cancel_task(task_id))

    def enqueue_fte_cancel_task(self, task_id: str | Mapping[str, Any]) -> Future[Any]:
        return self._loop.submit(self._manager.cancel_task(task_id))

    @staticmethod
    def resolve_fte_cancel_task(cancellation: Future[Any]) -> dict[str, Any]:
        # Native execution has no remote actor to retire. Keep the future owned
        # until its synchronous side effects are terminal.
        return _as_status("fte_cancel_task", cancellation.result())

    def fte_drop_query(self, query_id: str) -> dict[str, int]:
        # Query drop is also a barrier for task-owned writes and must not
        # return while a canceled synchronous execution can still mutate data.
        result = self._loop.run_owned_side_effects(self._manager.drop_query(str(query_id)))
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

    def shutdown(self, timeout_s: float = 5.0) -> None:
        if self._owns_loop:
            self._loop.shutdown(timeout_s=timeout_s)

    def request_shutdown(self) -> None:
        if self._owns_loop:
            self._loop.request_shutdown()


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
        self._stable_task_identity_lock = threading.RLock()
        self._stable_task_identity_keys_by_query: dict[str, dict[int, str]] = defaultdict(dict)
        self._dropped_queries: dict[str, dict[str, Any]] = {}
        self._progress_registry = _NativeFteProgressRegistry()
        self._closed = False
        self._closing = False
        self._debug_sampler_stop = threading.Event()
        self._debug_sampler_thread: threading.Thread | None = None
        if _native_submit_debug_enabled():
            self._start_debug_sampler()

    def worker_snapshots(self) -> Sequence[Mapping[str, Any]]:
        return [worker.snapshot() for worker in self._workers]

    def materialization_barrier_completed(self, query_id: str, node_id: str) -> None:
        """Acknowledge the runner protocol; native resources stay DuckDB-owned."""
        if not str(query_id or "").strip():
            raise ValueError("materialization barrier completion requires non-empty query_id")
        if not str(node_id or "").strip():
            raise ValueError("materialization barrier completion requires non-empty node_id")

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
        if self._closing:
            raise RuntimeError("native FTE worker manager is shutting down")
        submit_started_at = time.monotonic()
        batch_size = len(tasks)
        submitted_count = 0
        _native_submit_debug_log(
            "submit_tasks_enter",
            batch_size=batch_size,
            worker_count=len(self._workers),
        )
        requests = [self._request_from_task(task) for task in tasks]
        self._register_stable_task_identities(requests)
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
            failure = _failure_payload("NATIVE_BACKEND_ERROR", error)
            failure["type"] = type(error).__name__
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
        raw_output_stats = status.get("spooling_output_stats")
        output_stats = dict(raw_output_stats) if isinstance(raw_output_stats, Mapping) else raw_output_stats
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

    def fte_query_status(
        self,
        query_id: str,
        task_context_filter: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        query_id = str(query_id)
        scoped_contexts: set[_TaskContextKey] | None = None
        if task_context_filter:
            parsed_contexts = {_task_context_key(context) for context in task_context_filter}
            if None in parsed_contexts:
                raise ValueError("task_context_filter contains an invalid task context")
            scoped_contexts = {context for context in parsed_contexts if context is not None}
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
                "matched": False,
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
            context_matches = scoped_contexts is None or _task_context_key(handle.task_context_info) in scoped_contexts
            status_error: BaseException | None = None
            try:
                status = handle.status_snapshot()
            except BaseException as exc:
                failure = _failure_payload("NATIVE_BACKEND_ERROR", exc)
                failure["type"] = type(exc).__name__
                status = {
                    "state": FteTaskState.FAILED.value,
                    "failure": failure,
                }
                status_error = exc
            partition = self._partition_metrics_from_handle(handle, status, error=status_error)
            partition_metrics[(query_id, handle.fragment_id, str(handle.task_id.partition_id))] = partition
            if context_matches and partition["state"] == "FAILED":
                failed_partitions.append(
                    {
                        "task_id": handle.fte_task_id(),
                        "latest_failure": repr(status_error)
                        if status_error is not None
                        else str(status.get("failure")),
                    }
                )
            elif context_matches and partition["state"] == "FINISHED":
                selected_attempt_task_ids.append(handle.fte_task_id())
        return self._progress_registry.query_status(
            query_id,
            partition_metrics=partition_metrics,
            failed_partitions=failed_partitions,
            selected_attempt_task_ids=selected_attempt_task_ids,
            result_handle_count=sum(
                1
                for handle in handles
                if scoped_contexts is None or _task_context_key(handle.task_context_info) in scoped_contexts
            ),
            task_context_filter=scoped_contexts,
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
        with self._stable_task_identity_lock:
            self._stable_task_identity_keys_by_query.pop(query_id, None)
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

    def request_shutdown(self) -> None:
        if self._closed:
            return
        self._closing = True
        self._stop_debug_sampler()
        worker_errors: list[str] = []
        for worker in self._workers:
            request_shutdown = getattr(worker, "request_shutdown", None)
            if not callable(request_shutdown):
                continue
            worker_id = str(getattr(worker, "worker_id", "") or "<unknown>")
            try:
                request_shutdown()
            except BaseException as exc:
                worker_errors.append(f"{worker_id}: {type(exc).__name__}: {exc}")
        if worker_errors:
            raise RuntimeError("native FTE worker manager shutdown request failed: " + "; ".join(worker_errors))

    def shutdown(self, timeout_s: float | None = None) -> None:
        if self._closed:
            return
        deadline: float | None = None
        if timeout_s is not None:
            timeout_s = float(timeout_s)
            if not math.isfinite(timeout_s) or timeout_s < 0:
                raise ValueError("native FTE worker manager shutdown timeout must be finite and non-negative")
            deadline = time.monotonic() + timeout_s
        worker_errors: list[str] = []
        try:
            self.request_shutdown()
        except BaseException as exc:
            worker_errors.append(f"request: {type(exc).__name__}: {exc}")
        for worker in self._workers:
            worker_id = str(getattr(worker, "worker_id", "") or "<unknown>")
            try:
                if deadline is None:
                    worker.shutdown()
                else:
                    worker.shutdown(timeout_s=max(0.0, deadline - time.monotonic()))
            except BaseException as exc:
                worker_errors.append(f"{worker_id}: {type(exc).__name__}: {exc}")
        if worker_errors:
            raise RuntimeError("native FTE worker manager shutdown failed: " + "; ".join(worker_errors))
        self._closed = True
        self._closing = False

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
        raw_partition_id = task_context_info.get("task_id")
        if raw_partition_id is None:
            raw_partition_id = context.get("task_id")
        partition_id = int(raw_partition_id or 0)
        fragment_execution_id = int(context.get("fragment_execution_id") or 0)
        attempt_id = int(context.get("attempt_id") or 0)
        task_inputs = dict(task.Inputs() or {})

        for node_id, entry in task_inputs.items():
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
        stable_task_identity_key = None
        exchange_sink_instance_fn = getattr(task, "exchange_sink_instance", None)
        if callable(exchange_sink_instance_fn):
            exchange_sink_instance = exchange_sink_instance_fn()
            if exchange_sink_instance is not None:
                try:
                    exchange_sink_instance = dict(exchange_sink_instance)
                except (TypeError, ValueError):
                    pass
                preserve_plan_sink_partition = str(
                    context.get("preserve_plan_exchange_sink_instance") or ""
                ).strip().lower() not in ("", "0", "false", "no", "off")
                if preserve_plan_sink_partition and isinstance(exchange_sink_instance, Mapping):
                    exchange_sink_instance = dict(exchange_sink_instance)
                    # The inherited context can describe an upstream plan sink.
                    # An appended materialized-coordinator sink carries its own
                    # explicit FTE-derived identity policy.
                    if not bool(exchange_sink_instance.get("fte_task_identity")):
                        exchange_sink_instance["preserve_plan_exchange_sink_instance"] = True
                stable_task_identity = None
                runtime_task_partition_id = partition_id
                preserve_plan_sink_identity = isinstance(exchange_sink_instance, Mapping) and bool(
                    exchange_sink_instance.get("preserve_plan_exchange_sink_instance")
                )
                if isinstance(exchange_sink_instance, Mapping) and not preserve_plan_sink_identity:
                    stable_task_identity, stable_task_identity_key = _stable_native_fte_task_identity(
                        task_inputs,
                        task_context_info,
                        context,
                    )
                    runtime_task_partition_id = stable_task_identity
                exchange_sink_instance = derive_exchange_sink_instance_for_attempt(
                    exchange_sink_instance,
                    attempt_id,
                    task_partition_id=runtime_task_partition_id,
                    fragment_execution_id=fragment_execution_id,
                    stable_task_identity=(
                        stable_task_identity
                        if isinstance(exchange_sink_instance, Mapping)
                        and bool(exchange_sink_instance.get("fte_task_identity"))
                        else None
                    ),
                )

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

        request = {
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
        if stable_task_identity_key is not None:
            request[_NATIVE_STABLE_TASK_IDENTITY_KEY] = stable_task_identity_key
        return request

    def _register_stable_task_identities(self, requests: Sequence[dict[str, Any]]) -> None:
        pending: list[tuple[str, int, str]] = []
        for request in requests:
            identity_key = request.pop(_NATIVE_STABLE_TASK_IDENTITY_KEY, None)
            if identity_key is None:
                continue
            query_id = _query_id_from_task_id(request.get("task_id"))
            exchange_sink_instance = request.get("exchange_sink_instance")
            if not isinstance(exchange_sink_instance, Mapping):
                raise ValueError("stable native FTE task identity requires an exchange sink instance")
            stable_identity = exchange_sink_instance.get("task_partition_id")
            if stable_identity is None:
                raise ValueError("stable native FTE task identity was not bound to the exchange sink")
            pending.append((query_id, int(stable_identity), str(identity_key)))

        with self._stable_task_identity_lock:
            validated: dict[tuple[str, int], str] = {}
            for query_id, stable_identity, identity_key in pending:
                registry_key = (query_id, stable_identity)
                existing = validated.get(registry_key)
                if existing is None:
                    existing = self._stable_task_identity_keys_by_query.get(query_id, {}).get(stable_identity)
                if existing is not None and existing != identity_key:
                    raise ValueError(
                        f"stable native FTE task identity collision for query={query_id} identity={stable_identity}"
                    )
                validated[registry_key] = identity_key
            for (query_id, stable_identity), identity_key in validated.items():
                self._stable_task_identity_keys_by_query[query_id][stable_identity] = identity_key

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
        key = _task_context_key(value)
        if key is None:
            raise ValueError("task context must contain query_idx, last_node_id, task_id, and node_ids")
        return key
