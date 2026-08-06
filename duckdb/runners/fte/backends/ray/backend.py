# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from threading import Condition, Lock
from typing import Any

from duckdb.runners.fte.backend import TaskResultPoll, TaskResultState


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
        return tuple(sorted((str(key), _task_context_key(value[key])) for key in value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_task_context_key(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_task_context_key(item) for item in value)
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
        return self.resolve_fte_cancel_task(self.enqueue_fte_cancel_task(task_id))

    def enqueue_fte_cancel_task(self, task_id: str | Mapping[str, Any]) -> Any:
        return _required_method(self._handle, "enqueue_fte_cancel_task")(task_id)

    def resolve_fte_cancel_task(self, cancellation: Any) -> dict[str, Any]:
        result = _required_method(self._handle, "resolve_fte_cancel_task")(cancellation)
        return _dict_result("resolve_fte_cancel_task", result)


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
        self._lifecycle_condition = Condition(Lock())
        self._closed_queries: set[str] = set()
        self._closed_query_owners: set[str] = set()
        self._query_owner_by_query: dict[str, str] = {}
        self._active_operations_by_owner: dict[str, int] = {}
        self._cleanup_handles_by_owner: dict[str, list[RayTaskResultHandleAdapter]] = {}
        self._dropping_query_owners: dict[str, object] = {}
        self._shutdown = False
        self._shutdown_running = False
        self._shutdown_complete = False

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

    def register_query_owner(self, query_id: str, owner_query_id: str) -> None:
        query_id = str(query_id).strip()
        owner_query_id = str(owner_query_id).strip()
        if not query_id or not owner_query_id:
            raise ValueError("FTE query ownership requires non-empty query and owner IDs")
        with self._lifecycle_condition:
            existing_owner = self._query_owner_by_query.get(query_id)
            if existing_owner is not None and existing_owner != owner_query_id:
                raise RuntimeError(
                    f"FTE query owner changed while active: query={query_id} "
                    f"existing={existing_owner} requested={owner_query_id}"
                )
            if (
                self._shutdown
                or query_id in self._closed_queries
                or owner_query_id in self._closed_query_owners
                or owner_query_id in self._dropping_query_owners
            ):
                raise RuntimeError(f"cannot register closing FTE query lifecycle: {query_id}")
            self._query_owner_by_query[query_id] = owner_query_id

    def submit_tasks(self, tasks: Sequence[Any]) -> Sequence[RayTaskResultHandleAdapter]:
        task_list = list(tasks)
        if not task_list:
            return []
        query_id, owner_query_id = self._query_ids_for_tasks(task_list)
        active_owner = self._begin_query_operation(query_id, owner_query_id)
        if active_owner is None:
            return []
        try:
            raw_handles = _required_method(self._coordinator, "submit_tasks")(task_list)
            return self._accept_handles(active_owner, raw_handles or [])
        finally:
            self._end_query_operation(active_owner)

    def task_input_stream_exhausted(
        self,
        query_id: str,
        source_node_ids: Sequence[str],
    ) -> Sequence[RayTaskResultHandleAdapter]:
        query_id = str(query_id).strip()
        active_owner = self._begin_query_operation(query_id)
        if active_owner is None:
            return []
        try:
            raw_handles = _required_method(self._coordinator, "task_input_stream_exhausted_for_query")(
                query_id,
                [str(source_node_id) for source_node_id in source_node_ids],
            )
            return self._accept_handles(active_owner, raw_handles or [])
        finally:
            self._end_query_operation(active_owner)

    def materialization_barrier_completed(self, query_id: str, node_id: str) -> None:
        query_id = str(query_id).strip()
        active_owner = self._begin_query_operation(query_id)
        if active_owner is None:
            return
        try:
            _required_method(self._coordinator, "materialization_barrier_completed")(
                query_id,
                str(node_id),
            )
        finally:
            self._end_query_operation(active_owner)

    def fte_query_status(
        self,
        query_id: str,
        task_context_filter: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        query_id = str(query_id).strip()
        active_owner = self._begin_query_operation(query_id)
        if active_owner is None:
            return {
                "query_id": query_id,
                "finished": False,
                "failed": False,
                "canceled": True,
                "matched": False,
                "registration_pending": False,
                "selected_attempt_task_ids": [],
            }
        context_filter = None if task_context_filter is None else list(task_context_filter)
        try:
            result = _required_method(self._coordinator, "fte_query_status")(query_id, context_filter)
            return _dict_result("fte_query_status", result)
        finally:
            self._end_query_operation(active_owner)

    def wait_query(
        self,
        query_id: str,
        timeout_s: float,
        task_context_filter: Sequence[Any] | None = None,
    ) -> Sequence[RayTaskResultHandleAdapter]:
        query_id = str(query_id).strip()
        allowed = None
        if task_context_filter:
            allowed = {_task_context_key(item) for item in task_context_filter}
        active_owner = self._begin_query_operation(query_id)
        if active_owner is None:
            return []
        try:
            _required_method(self._coordinator, "wait_fte_query")(query_id, float(timeout_s))
            raw_handles = []
            pop_handles = getattr(self._coordinator, "pop_fte_result_handles", None)
            if callable(pop_handles):
                raw_handles = list(pop_handles(query_id) or [])
            handles = self._accept_handles(active_owner, raw_handles)
            if allowed is not None:
                selected_handles = []
                discarded_handles = []
                try:
                    for handle in handles:
                        target = (
                            selected_handles
                            if _task_context_key(handle.task_context()) in allowed
                            else discarded_handles
                        )
                        target.append(handle)
                except BaseException as selection_error:
                    cleanup_error: BaseException | None = None
                    try:
                        self._release_handles(
                            active_owner,
                            handles,
                            reason="abandoned after filter selection failure",
                        )
                    except BaseException as exc:
                        cleanup_error = exc
                    self._raise_lifecycle_errors(
                        "result filter selection",
                        selection_error,
                        cleanup_error,
                    )
                try:
                    self._release_handles(active_owner, discarded_handles, reason="filtered")
                except BaseException as discarded_error:
                    selected_error: BaseException | None = None
                    try:
                        self._release_handles(
                            active_owner,
                            selected_handles,
                            reason="abandoned selected",
                        )
                    except BaseException as exc:
                        selected_error = exc
                    self._raise_lifecycle_errors(
                        "filtered result cleanup",
                        discarded_error,
                        selected_error,
                    )
                handles = selected_handles
            return handles
        finally:
            self._end_query_operation(active_owner)

    def drop_query(self, query_id: str) -> None:
        query_id = str(query_id).strip()
        drop_lifecycle = self._begin_query_drop(query_id)
        if drop_lifecycle is None:
            return
        owner_query_id, execution_query_ids, drop_token = drop_lifecycle
        try:
            drop_error: BaseException | None = None
            drop_query = getattr(self._coordinator, "fte_drop_query", None)
            drop_query_fragments = getattr(self._coordinator, "drop_query_fragments", None)
            drop_errors: list[BaseException] = []
            for execution_query_id in execution_query_ids:
                try:
                    if callable(drop_query):
                        drop_query(execution_query_id)
                    elif callable(drop_query_fragments):
                        drop_query_fragments(execution_query_id)
                except BaseException as exc:
                    drop_errors.append(exc)
            if drop_errors:
                drop_error = RuntimeError(
                    f"failed to drop {len(drop_errors)} execution query lifecycle(s) for owner {owner_query_id}"
                )
                drop_error.__cause__ = drop_errors[0]
            if drop_error is None:
                self._wait_for_query_operations(owner_query_id)
            cleanup_error = self._retry_cleanup_handles(owner_query_id)
            if drop_error is None and cleanup_error is None:
                self._finish_query_drop(owner_query_id, drop_token)
            self._raise_lifecycle_errors("query drop", drop_error, cleanup_error)
        finally:
            self._end_query_drop(owner_query_id, drop_token)

    def drop_query_owner(self, owner_query_id: str) -> None:
        self.drop_query(owner_query_id)

    def shutdown(self) -> None:
        if not self._begin_shutdown():
            return
        succeeded = False
        try:
            shutdown_error: BaseException | None = None
            shutdown = getattr(self._coordinator, "shutdown", None)
            try:
                if callable(shutdown):
                    shutdown()
            except BaseException as exc:
                shutdown_error = exc
            self._wait_for_all_query_operations()
            with self._lifecycle_condition:
                cleanup_query_ids = list(self._cleanup_handles_by_owner)
            cleanup_errors = [
                error for query_id in cleanup_query_ids if (error := self._retry_cleanup_handles(query_id)) is not None
            ]
            cleanup_error = None
            if cleanup_errors:
                cleanup_error = RuntimeError(
                    f"failed to release cleanup handles for {len(cleanup_errors)} query(s) during shutdown"
                )
                cleanup_error.__cause__ = cleanup_errors[0]
            if cleanup_error is None:
                with self._lifecycle_condition:
                    self._query_owner_by_query.clear()
                    self._closed_queries.clear()
                    self._closed_query_owners.clear()
            self._raise_lifecycle_errors("shutdown", shutdown_error, cleanup_error)
            succeeded = True
        finally:
            self._end_shutdown(succeeded)

    def _accept_handles(
        self,
        owner_query_id: str,
        raw_handles: Iterable[Any],
    ) -> list[RayTaskResultHandleAdapter]:
        handles = self._adapt_handles(raw_handles)
        with self._lifecycle_condition:
            closed = self._shutdown or owner_query_id in self._closed_query_owners
        if not closed:
            return handles

        self._release_handles(owner_query_id, handles, reason="late")
        return []

    def _release_handles(
        self,
        owner_query_id: str,
        handles: Iterable[RayTaskResultHandleAdapter],
        *,
        reason: str,
    ) -> None:
        release_errors: list[BaseException] = []
        failed_handles: list[RayTaskResultHandleAdapter] = []
        for handle in handles:
            try:
                handle.release_result_payload()
            except BaseException as exc:
                release_errors.append(exc)
                failed_handles.append(handle)
        if release_errors:
            with self._lifecycle_condition:
                self._cleanup_handles_by_owner.setdefault(owner_query_id, []).extend(failed_handles)
            raise RuntimeError(
                f"failed to release {len(release_errors)} {reason} result handle(s) for query {owner_query_id}"
            ) from release_errors[0]

    def _retry_cleanup_handles(self, owner_query_id: str) -> BaseException | None:
        with self._lifecycle_condition:
            handles = self._cleanup_handles_by_owner.pop(owner_query_id, [])
        if not handles:
            return None

        release_errors: list[BaseException] = []
        failed_handles: list[RayTaskResultHandleAdapter] = []
        for handle in handles:
            try:
                handle.release_result_payload()
            except BaseException as exc:
                release_errors.append(exc)
                failed_handles.append(handle)
        if failed_handles:
            with self._lifecycle_condition:
                self._cleanup_handles_by_owner.setdefault(owner_query_id, []).extend(failed_handles)
        if not release_errors:
            return None
        error = RuntimeError(
            f"failed to retry {len(release_errors)} result handle release(s) for query {owner_query_id}"
        )
        error.__cause__ = release_errors[0]
        return error

    @staticmethod
    def _raise_lifecycle_errors(
        operation: str,
        primary_error: BaseException | None,
        cleanup_error: BaseException | None,
    ) -> None:
        if primary_error is not None and cleanup_error is not None:
            raise RuntimeError(
                f"Ray backend {operation} and result cleanup both failed: {cleanup_error}"
            ) from primary_error
        if primary_error is not None:
            raise primary_error
        if cleanup_error is not None:
            raise cleanup_error

    def _begin_query_operation(self, query_id: str, owner_query_id: str | None = None) -> str | None:
        query_id = str(query_id).strip()
        if not query_id:
            raise ValueError("FTE query operation requires non-empty query_id")
        with self._lifecycle_condition:
            existing_owner = self._query_owner_by_query.get(query_id)
            if existing_owner is None:
                return None
            owner_query_id = str(owner_query_id or existing_owner).strip()
            if existing_owner is not None and existing_owner != owner_query_id:
                raise RuntimeError(
                    f"FTE query owner changed while active: query={query_id} "
                    f"existing={existing_owner} requested={owner_query_id}"
                )
            if (
                self._shutdown
                or query_id in self._closed_queries
                or owner_query_id in self._closed_query_owners
                or owner_query_id in self._dropping_query_owners
            ):
                return None
            self._query_owner_by_query[query_id] = owner_query_id
            self._active_operations_by_owner[owner_query_id] = (
                self._active_operations_by_owner.get(owner_query_id, 0) + 1
            )
            return owner_query_id

    def _begin_query_drop(self, query_id: str) -> tuple[str, tuple[str, ...], object] | None:
        if not query_id:
            return None
        with self._lifecycle_condition:
            owner_query_id = self._query_owner_by_query.get(query_id)
            if owner_query_id is None:
                known_owner = query_id in self._query_owner_by_query.values()
                if not known_owner and query_id not in self._dropping_query_owners:
                    return None
                owner_query_id = query_id
            joined_drop_token = self._dropping_query_owners.get(owner_query_id)
            if joined_drop_token is not None:
                self._lifecycle_condition.wait_for(
                    lambda: self._dropping_query_owners.get(owner_query_id) is not joined_drop_token
                )
                if owner_query_id not in self._closed_query_owners:
                    return None
                if owner_query_id in self._dropping_query_owners:
                    return None
            if self._shutdown:
                return None
            self._closed_queries.add(query_id)
            self._closed_query_owners.add(owner_query_id)
            drop_token = object()
            self._dropping_query_owners[owner_query_id] = drop_token
            execution_query_ids = {
                execution_query_id
                for execution_query_id, owner in self._query_owner_by_query.items()
                if owner == owner_query_id
            }
            execution_query_ids.add(owner_query_id)
            ordered_query_ids = tuple(sorted(execution_query_ids, key=lambda item: (item == owner_query_id, item)))
            return owner_query_id, ordered_query_ids, drop_token

    def _end_query_drop(self, owner_query_id: str, drop_token: object) -> None:
        with self._lifecycle_condition:
            if self._dropping_query_owners.get(owner_query_id) is drop_token:
                self._dropping_query_owners.pop(owner_query_id)
            self._lifecycle_condition.notify_all()

    def _begin_shutdown(self) -> bool:
        with self._lifecycle_condition:
            while self._shutdown_running:
                self._lifecycle_condition.wait()
            if self._shutdown_complete:
                return False
            self._shutdown = True
            self._shutdown_running = True
            self._lifecycle_condition.wait_for(lambda: not self._dropping_query_owners)
            return True

    def _end_shutdown(self, succeeded: bool) -> None:
        with self._lifecycle_condition:
            self._shutdown_complete = succeeded
            self._shutdown_running = False
            self._lifecycle_condition.notify_all()

    def _end_query_operation(self, owner_query_id: str) -> None:
        with self._lifecycle_condition:
            active = self._active_operations_by_owner.get(owner_query_id, 0)
            if active <= 0:
                raise RuntimeError(f"FTE query operation ownership underflow: {owner_query_id}")
            if active == 1:
                self._active_operations_by_owner.pop(owner_query_id, None)
            else:
                self._active_operations_by_owner[owner_query_id] = active - 1
            self._lifecycle_condition.notify_all()

    def _wait_for_query_operations(self, owner_query_id: str) -> None:
        with self._lifecycle_condition:
            self._lifecycle_condition.wait_for(lambda: self._active_operations_by_owner.get(owner_query_id, 0) == 0)

    def _wait_for_all_query_operations(self) -> None:
        with self._lifecycle_condition:
            self._lifecycle_condition.wait_for(lambda: not self._active_operations_by_owner)

    def _finish_query_drop(self, owner_query_id: str, drop_token: object) -> None:
        with self._lifecycle_condition:
            if self._dropping_query_owners.get(owner_query_id) is not drop_token:
                raise RuntimeError(f"cannot finish stale FTE query drop: {owner_query_id}")
            if self._active_operations_by_owner.get(owner_query_id, 0) != 0:
                raise RuntimeError(f"cannot finish active FTE query lifecycle: {owner_query_id}")
            if self._cleanup_handles_by_owner.get(owner_query_id):
                raise RuntimeError(f"cannot finish FTE query lifecycle with pending cleanup: {owner_query_id}")
            owned_query_ids = {
                query_id for query_id, owner in self._query_owner_by_query.items() if owner == owner_query_id
            }
            self._query_owner_by_query = {
                query_id: owner
                for query_id, owner in self._query_owner_by_query.items()
                if query_id not in owned_query_ids
            }
            self._closed_queries.difference_update(owned_query_ids | {owner_query_id})
            self._closed_query_owners.discard(owner_query_id)
            self._dropping_query_owners.pop(owner_query_id)
            self._lifecycle_condition.notify_all()

    def _adapt_handles(self, handles: Iterable[Any]) -> list[RayTaskResultHandleAdapter]:
        adapted: list[RayTaskResultHandleAdapter] = []
        for handle in handles:
            if isinstance(handle, self._result_handle_adapter):
                adapted.append(handle)
            else:
                adapted.append(self._result_handle_adapter(handle))
        return adapted

    @staticmethod
    def _query_ids_for_tasks(tasks: Sequence[Any]) -> tuple[str, str]:
        query_id = ""
        owner_query_id = ""
        for task in tasks:
            if isinstance(task, Mapping):
                context: Any = task
            else:
                context_method = getattr(task, "context", None)
                if not callable(context_method):
                    raise TypeError("FTE task must provide callable context()")
                context = context_method()
            if not isinstance(context, Mapping):
                raise TypeError("FTE task context must be a mapping")
            task_query_id = str(context.get("query_id") or "").strip()
            if not task_query_id:
                raise ValueError("FTE task requires non-empty query_id")
            if query_id and query_id != task_query_id:
                raise ValueError("FTE submit batch contains multiple query_id values")
            query_id = task_query_id
            task_owner_query_id = str(context.get("resource_query_id") or task_query_id).strip()
            if owner_query_id and owner_query_id != task_owner_query_id:
                raise ValueError("FTE submit batch contains multiple resource_query_id values")
            owner_query_id = task_owner_query_id
        return query_id, owner_query_id
