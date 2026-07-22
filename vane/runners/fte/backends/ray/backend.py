# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from vane.runners.fte.backend import TaskResultPoll, TaskResultState


def _required_method(target: Any, method_name: str) -> Callable[..., Any]:
    method = getattr(target, method_name, None)
    if not callable(method):
        raise TypeError(f"{type(target).__name__} must provide callable {method_name}")
    return method


def _dict_result(method_name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{method_name} must return a mapping")
    return dict(value)


def _task_context_key(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), value[key]) for key in value))
    return value


class RayTaskResultHandleAdapter:
    """Backend-neutral result-handle view over an existing Ray FTE task handle."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    @property
    def delegate(self) -> Any:
        return self._handle

    def task_context(self) -> Any:
        task_context = getattr(self._handle, "task_context", None)
        if callable(task_context):
            return task_context()
        get_task_context = getattr(self._handle, "GetTaskContext", None)
        if callable(get_task_context):
            return get_task_context()
        task_context_info = getattr(self._handle, "task_context_info", None)
        if task_context_info is not None:
            return dict(task_context_info) if isinstance(task_context_info, Mapping) else task_context_info
        return task_context

    def fte_task_id(self) -> str:
        fte_task_id = getattr(self._handle, "fte_task_id", None)
        if callable(fte_task_id):
            return str(fte_task_id())
        get_fte_task_id = getattr(self._handle, "GetFteTaskId", None)
        if callable(get_fte_task_id):
            return str(get_fte_task_id())
        task_id = getattr(self._handle, "task_id", None)
        if task_id is not None:
            return str(task_id)
        raw_fte_task_id = getattr(self._handle, "fte_task_id", "")
        return str(raw_fte_task_id or "")

    def worker_id(self) -> str:
        worker_id = getattr(self._handle, "worker_id", None)
        if callable(worker_id):
            return str(worker_id())
        return str(worker_id or "")

    def poll(self) -> TaskResultPoll:
        poll = getattr(self._handle, "poll", None)
        if callable(poll):
            return self._normalize_poll_result(poll())

        done = getattr(self._handle, "done", None)
        if callable(done) and not bool(done()):
            return TaskResultPoll(TaskResultState.NOT_READY)

        get_result_sync = getattr(self._handle, "get_result_sync", None)
        if callable(get_result_sync):
            try:
                result = get_result_sync()
            except BaseException as exc:
                return TaskResultPoll(TaskResultState.ERROR, error=exc)
            if result is None:
                return TaskResultPoll(TaskResultState.NO_OUTPUT)
            return TaskResultPoll(TaskResultState.MATERIALIZED_OUTPUT, output=result)

        return TaskResultPoll(TaskResultState.NOT_READY)

    @staticmethod
    def _normalize_poll_result(value: Any) -> TaskResultPoll:
        if isinstance(value, TaskResultPoll):
            return value
        if isinstance(value, Mapping):
            state = TaskResultState(str(value.get("state")))
            error = value.get("error")
            return TaskResultPoll(
                state,
                output=value.get("output"),
                error=error if isinstance(error, BaseException) else None,
            )
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], bool):
            ready, payload = value
            if not ready:
                return TaskResultPoll(TaskResultState.NOT_READY)
            if isinstance(payload, BaseException):
                return TaskResultPoll(TaskResultState.ERROR, error=payload)
            if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], bool):
                has_output, output = payload
                state = TaskResultState.MATERIALIZED_OUTPUT if has_output else TaskResultState.NO_OUTPUT
                return TaskResultPoll(state, output=output if has_output else None)
            if payload is None:
                return TaskResultPoll(TaskResultState.NO_OUTPUT)
            return TaskResultPoll(TaskResultState.MATERIALIZED_OUTPUT, output=payload)
        if value is None:
            return TaskResultPoll(TaskResultState.NO_OUTPUT)
        return TaskResultPoll(TaskResultState.MATERIALIZED_OUTPUT, output=value)

    def ack(self) -> None:
        ack = getattr(self._handle, "ack", None)
        if callable(ack):
            ack()
            return
        ack_poll_result = getattr(self._handle, "AckPollResult", None)
        if callable(ack_poll_result):
            ack_poll_result()

    def release_result_payload(self) -> None:
        _required_method(self._handle, "release_result_payload")()


class RayWorkerHandleAdapter:
    """WorkerHandle protocol adapter for existing Ray worker handles."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    @property
    def delegate(self) -> Any:
        return self._handle

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)

    @property
    def worker_id(self) -> str:
        worker_id = getattr(self._handle, "worker_id", None)
        if callable(worker_id):
            return str(worker_id())
        return str(worker_id or "")

    def fte_create_task(self, request: Mapping[str, Any]) -> dict[str, Any]:
        result = _required_method(self._handle, "fte_create_task")(dict(request))
        return _dict_result("fte_create_task", result)

    def fte_add_splits(
        self,
        task_id: str | Mapping[str, Any],
        source_node_id: str,
        splits: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        split_payloads = [dict(split) for split in splits]
        result = _required_method(self._handle, "fte_add_splits")(task_id, str(source_node_id), split_payloads)
        return _dict_result("fte_add_splits", result)

    def fte_no_more_splits(
        self,
        task_id: str | Mapping[str, Any],
        source_node_id: str,
    ) -> dict[str, Any]:
        result = _required_method(self._handle, "fte_no_more_splits")(task_id, str(source_node_id))
        return _dict_result("fte_no_more_splits", result)

    def fte_update_task(
        self,
        task_id: str | Mapping[str, Any],
        update: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = _required_method(self._handle, "fte_update_task")(task_id, dict(update))
        return _dict_result("fte_update_task", result)

    def fte_wait_task_status(
        self,
        task_id: str | Mapping[str, Any],
        min_version: int | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        result = _required_method(self._handle, "fte_wait_task_status")(task_id, min_version, timeout_s)
        return _dict_result("fte_wait_task_status", result)

    def fte_cancel_task(self, task_id: str | Mapping[str, Any]) -> dict[str, Any]:
        result = _required_method(self._handle, "fte_cancel_task")(task_id)
        return _dict_result("fte_cancel_task", result)


class RayWorkerManagerBackend:
    """WorkerManagerBackend adapter over an existing Ray coordinator handle."""

    def __init__(
        self,
        coordinator: Any,
        *,
        result_handle_adapter: type[RayTaskResultHandleAdapter] = RayTaskResultHandleAdapter,
        snapshot_provider: Any | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._result_handle_adapter = result_handle_adapter
        self._snapshot_provider = snapshot_provider
        self._handles_by_query: dict[str, list[RayTaskResultHandleAdapter]] = defaultdict(list)

    @property
    def delegate(self) -> Any:
        return self._coordinator

    def worker_snapshots(self) -> Sequence[Mapping[str, Any]]:
        if self._snapshot_provider is not None:
            snapshots = _required_method(self._snapshot_provider, "snapshots")()
        else:
            worker_snapshots = getattr(self._coordinator, "worker_snapshots", None)
            if not callable(worker_snapshots):
                return []
            snapshots = worker_snapshots()
        return [dict(snapshot) if isinstance(snapshot, Mapping) else snapshot for snapshot in snapshots]

    def submit_tasks(self, tasks: Sequence[Any]) -> Sequence[RayTaskResultHandleAdapter]:
        raw_handles = _required_method(self._coordinator, "submit_tasks")(list(tasks))
        handles = self._adapt_handles(raw_handles)
        for handle in handles:
            query_id = self._query_id_for_handle(handle)
            if query_id:
                self._handles_by_query[query_id].append(handle)
        return handles

    def task_input_stream_exhausted(
        self,
        query_id: str,
        source_node_ids: Sequence[str],
    ) -> None:
        raw_handles = _required_method(self._coordinator, "task_input_stream_exhausted_for_query")(
            str(query_id),
            [str(source_node_id) for source_node_id in source_node_ids],
        )
        for handle in self._adapt_handles(raw_handles or []):
            self._handles_by_query[str(query_id)].append(handle)

    def fte_query_status(self, query_id: str) -> dict[str, Any]:
        result = _required_method(self._coordinator, "fte_query_status")(str(query_id))
        return _dict_result("fte_query_status", result)

    def wait_query(
        self,
        query_id: str,
        timeout_s: float,
        task_context_filter: Sequence[Any] | None = None,
    ) -> Sequence[RayTaskResultHandleAdapter]:
        query_id = str(query_id)
        _required_method(self._coordinator, "wait_fte_query")(query_id, float(timeout_s))
        raw_handles = []
        pop_handles = getattr(self._coordinator, "pop_fte_result_handles", None)
        if callable(pop_handles):
            raw_handles = list(pop_handles(query_id) or [])
        handles = self._handles_by_query.pop(query_id, [])
        handles.extend(self._adapt_handles(raw_handles))
        if task_context_filter:
            allowed = {_task_context_key(item) for item in task_context_filter}
            handles = [handle for handle in handles if _task_context_key(handle.task_context()) in allowed]
        return handles

    def drop_query(self, query_id: str) -> None:
        query_id = str(query_id)
        self._handles_by_query.pop(query_id, None)
        drop_query = getattr(self._coordinator, "fte_drop_query", None)
        if callable(drop_query):
            drop_query(query_id)
            return
        drop_query_fragments = getattr(self._coordinator, "drop_query_fragments", None)
        if callable(drop_query_fragments):
            drop_query_fragments(query_id)

    def shutdown(self) -> None:
        shutdown = getattr(self._coordinator, "shutdown", None)
        if callable(shutdown):
            shutdown()

    def _adapt_handles(self, handles: Iterable[Any]) -> list[RayTaskResultHandleAdapter]:
        adapted: list[RayTaskResultHandleAdapter] = []
        for handle in handles:
            if isinstance(handle, self._result_handle_adapter):
                adapted.append(handle)
            else:
                adapted.append(self._result_handle_adapter(handle))
        return adapted

    @staticmethod
    def _query_id_for_handle(handle: RayTaskResultHandleAdapter) -> str:
        task_context = handle.task_context()
        if isinstance(task_context, Mapping):
            query_id = task_context.get("query_id")
            if query_id:
                return str(query_id)
        fte_task_id = handle.fte_task_id()
        if fte_task_id:
            return fte_task_id.rsplit(".", 3)[0]
        return ""
