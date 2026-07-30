# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from duckdb.runners.fte import FteTaskAttemptId, FteTaskState, validate_fte_status_identity
from duckdb.runners.fte.fte_scheduler import FteSplitQueueTerminal
from duckdb.runners.fte.fte_state import _TERMINAL_STATES
from duckdb.runners.ray.fragment_registry import (
    _FTE_CLOSING_QUERIES,
    _FTE_REGISTRY_LOCK,
)
from duckdb.runners.ray.fragment_worker_control import (
    enqueue_ordered_fte_control,
)
from duckdb.runners.ray.fte_fragment_scheduler import (
    _drop_fragment_plan_refs_for_query,
    _drop_fte_registry_for_query,
    begin_fte_registry_operation,
    begin_fte_registry_teardown_operation,
    close_fte_registry_for_query,
    end_fte_registry_operation,
    end_fte_registry_teardown_operation,
    quiesce_fte_registry_for_query,
    transfer_fte_registry_operations_to_ref,
    transfer_fte_registry_teardown_operations_to_ref,
)
from duckdb.runners.ray.fte_scheduler_config import (
    _fte_control_rpc_initial_backoff_s,
    _fte_control_rpc_max_attempts,
    _fte_control_rpc_timeout_s,
)
from duckdb.runners.ray.safe_get import (
    QueryDeadlineExceeded,
    configured_ray_get_timeout_s,
    resolve_object_refs_blocking,
)


class FteControlBarrierPendingError(RuntimeError):
    """The query control cut still owns at least one non-terminal ObjectRef."""


class FteControlBarrierTerminalError(RuntimeError):
    """The control cut is terminal, but one or more operations failed validation."""


class FteWorkerTaskControlMixin:
    actor_handle: Any
    _fte_control_lock: Any
    _fte_control_tails_by_task: dict[str, Any]
    _fte_control_query_by_task: dict[str, str]
    _fte_control_operation_by_task: dict[str, str]
    _fte_drop_incomplete_queries: set[str]
    _fte_prepare_terminal_errors: dict[str, BaseException]
    _fragment_drop_incomplete_queries: set[str]

    if TYPE_CHECKING:

        def _drop_fragment_registration_state(self, query_id: str) -> None: ...

    @staticmethod
    def _validated_fte_add_splits_status(
        task_id: str | dict[str, Any],
        raw_status: Any,
    ) -> dict[str, Any]:
        if not isinstance(raw_status, dict):
            raise TypeError("worker actor fte_add_splits must return a dict")
        status = dict(raw_status)
        if status.get("_fte_control_applied") is not False:
            return status
        status = FteWorkerTaskControlMixin._validated_terminal_fte_add_splits_status(
            task_id,
            status,
        )
        raise FteSplitQueueTerminal(FteTaskAttemptId.coerce(task_id), status)

    @staticmethod
    def _validated_terminal_fte_add_splits_status(
        task_id: str | dict[str, Any],
        status: dict[str, Any],
    ) -> dict[str, Any]:
        attempt_id = FteTaskAttemptId.coerce(task_id)
        validate_fte_status_identity(status, attempt_id)
        if status.get("_fte_control_operation") != "fte_add_splits":
            raise RuntimeError(
                "FTE add-splits control operation mismatch: "
                f"task={attempt_id} actual={status.get('_fte_control_operation')!r}"
            )
        raw_state = status.get("state")
        try:
            state = raw_state if isinstance(raw_state, FteTaskState) else FteTaskState(str(raw_state))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown FTE task state in rejected add-splits control: {raw_state!r}") from exc
        if state not in _TERMINAL_STATES:
            raise RuntimeError(
                f"FTE add-splits control was not applied to non-terminal task {attempt_id}: {state.value}"
            )
        status.pop("_fte_control_operation", None)
        status.pop("_fte_control_applied", None)
        return status

    def _fte_control_rpc(
        self,
        method_name: str,
        *args: Any,
        timeout_s: float | None = None,
        cancel_event: Any = None,
    ) -> Any:
        attempts = _fte_control_rpc_max_attempts()
        backoff_s = _fte_control_rpc_initial_backoff_s()
        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                method = getattr(self.actor_handle, method_name)
                ref = method.remote(*args)
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                if backoff_s > 0:
                    time.sleep(backoff_s)
                    backoff_s = min(backoff_s * 2, 5.0)
                continue
            return self._get_fte_control_ref(
                method_name,
                ref,
                timeout_s=timeout_s,
                cancel_event=cancel_event,
            )
        assert last_error is not None
        raise last_error

    @staticmethod
    def _cancel_fte_control_ref(
        ref: Any,
        on_cancel: Callable[[], None] | None,
    ) -> None:
        if on_cancel is not None:
            on_cancel()
        try:
            import ray

            ray.cancel(ref, force=False)
        except Exception:
            pass

    @staticmethod
    def _get_fte_control_ref(
        method_name: str,
        ref: Any,
        timeout_s: float | None = None,
        cancel_event: Any = None,
        honor_query_deadline: bool = True,
        on_cancel: Callable[[], None] | None = None,
        wait_for_owned_side_effects: bool = False,
    ) -> Any:
        if wait_for_owned_side_effects:
            # A successful cancellation may gate a COPY commit. No local
            # control, query, or ObjectRef timeout may outlive its writer.
            return resolve_object_refs_blocking(
                ref,
                timeout=None,
                honor_query_deadline=False,
                honor_object_get_timeout=False,
            )
        resolved_timeout_s = _fte_control_rpc_timeout_s() if timeout_s is None else max(0.0, float(timeout_s))
        if cancel_event is not None:
            try:
                resolved_timeout_s = configured_ray_get_timeout_s(resolved_timeout_s)
            except QueryDeadlineExceeded:
                FteWorkerTaskControlMixin._cancel_fte_control_ref(ref, on_cancel)
                raise
            deadline = None if resolved_timeout_s is None else time.monotonic() + resolved_timeout_s
            future = ref.future()
            while True:
                if cancel_event.is_set():
                    FteWorkerTaskControlMixin._cancel_fte_control_ref(ref, on_cancel)
                    raise InterruptedError(f"{method_name} interrupted by cancellation")
                # Preserve a query-wide deadline as a query failure rather than
                # misclassifying it as an individual control RPC timeout.
                try:
                    configured_ray_get_timeout_s(None)
                except QueryDeadlineExceeded:
                    FteWorkerTaskControlMixin._cancel_fte_control_ref(ref, on_cancel)
                    raise
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining is not None and remaining <= 0.0:
                    FteWorkerTaskControlMixin._cancel_fte_control_ref(ref, on_cancel)
                    raise TimeoutError(f"{method_name} did not complete within {resolved_timeout_s:.3f}s")
                wait_slice = 0.05 if remaining is None else min(0.05, remaining)
                try:
                    return future.result(timeout=wait_slice)
                except concurrent.futures.TimeoutError:
                    done = getattr(future, "done", None)
                    if callable(done) and done():
                        return future.result()
                    continue
        try:
            if honor_query_deadline:
                return resolve_object_refs_blocking(
                    ref,
                    timeout=resolved_timeout_s,
                )
            return resolve_object_refs_blocking(
                ref,
                timeout=resolved_timeout_s,
                honor_query_deadline=False,
            )
        except QueryDeadlineExceeded:
            raise
        except TimeoutError as exc:
            if honor_query_deadline:
                configured_ray_get_timeout_s(None)
            raise TimeoutError(f"{method_name} did not complete within {resolved_timeout_s:.3f}s") from exc

    def _enqueue_ordered_fte_control_rpc(
        self,
        method_name: str,
        task_id: str | dict[str, Any],
        *args: Any,
        timeout_s: float | None = None,
        cancel_event: Any = None,
        wait_for_owned_side_effects: bool = False,
    ) -> Any:
        tracked_query_id: str | None = None
        owns_registry_operation = False
        if cancel_event is not None:
            tracked_query_id = FteTaskAttemptId.coerce(task_id).task_id.query_id
            if not begin_fte_registry_operation(tracked_query_id):
                raise RuntimeError(f"FTE query registry is closing: {tracked_query_id}")
            owns_registry_operation = True
        try:
            ref = self._enqueue_ordered_fte_control_ref(method_name, task_id, *args)
            if owns_registry_operation:
                assert tracked_query_id is not None
                transfer_fte_registry_operations_to_ref([tracked_query_id], ref)
                owns_registry_operation = False
            on_cancel = None
            if cancel_event is not None:

                def discard_ref() -> None:
                    self._discard_ordered_fte_control_ref(task_id, ref)

                on_cancel = discard_ref
            return self._get_fte_control_ref(
                method_name,
                ref,
                timeout_s=timeout_s,
                cancel_event=cancel_event,
                on_cancel=on_cancel,
                wait_for_owned_side_effects=wait_for_owned_side_effects,
            )
        finally:
            if owns_registry_operation:
                assert tracked_query_id is not None
                end_fte_registry_operation(tracked_query_id)

    def _discard_ordered_fte_control_ref(
        self,
        task_id: str | dict[str, Any],
        ref: Any,
    ) -> None:
        task_key = str(FteTaskAttemptId.coerce(task_id))
        with self._fte_control_lock:
            if self._fte_control_tails_by_task.get(task_key) is not ref:
                return
            self._fte_control_tails_by_task.pop(task_key, None)
            self._fte_control_query_by_task.pop(task_key, None)
            self._fte_control_operation_by_task.pop(task_key, None)

    def _is_current_ordered_fte_control_ref(
        self,
        task_key: str,
        ref: Any,
    ) -> bool:
        with self._fte_control_lock:
            return self._fte_control_tails_by_task.get(task_key) is ref

    def _enqueue_ordered_fte_control_ref(
        self,
        method_name: str,
        task_id: str | dict[str, Any],
        *args: Any,
    ) -> Any:
        attempts = _fte_control_rpc_max_attempts()
        backoff_s = _fte_control_rpc_initial_backoff_s()
        last_error: BaseException | None = None
        attempt_id = FteTaskAttemptId.coerce(task_id)
        task_key = str(attempt_id)
        query_id = attempt_id.task_id.query_id
        for attempt in range(attempts):
            try:
                with _FTE_REGISTRY_LOCK:
                    if query_id in _FTE_CLOSING_QUERIES:
                        if method_name == "fte_release_task_result":
                            # Query drop owns every remaining remote task/result.
                            # A late local handle release is therefore already
                            # satisfied and must not reopen the closed control
                            # chain during the teardown barrier.
                            return None
                        raise RuntimeError(f"FTE control admission is closed for query {query_id}: {method_name}")
                    with self._fte_control_lock:
                        ref = enqueue_ordered_fte_control(
                            self.actor_handle,
                            self._fte_control_tails_by_task,
                            method_name,
                            task_id,
                            *args,
                        )
                        self._fte_control_query_by_task[task_key] = query_id
                        self._fte_control_operation_by_task[task_key] = method_name
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                if backoff_s > 0:
                    time.sleep(backoff_s)
                    backoff_s = min(backoff_s * 2, 5.0)
                continue
            return ref
        assert last_error is not None
        raise last_error

    def fte_create_task(self, request: dict[str, Any]) -> dict[str, Any]:
        attempt_id = FteTaskAttemptId.coerce(request.get("task_id"))
        task_key = str(attempt_id)
        query_id = attempt_id.query_id
        if not begin_fte_registry_operation(query_id):
            raise RuntimeError(f"FTE query registry is closing: {query_id}")
        owns_registry_operation = True
        create_ref = None
        try:
            with self._fte_control_lock:
                if task_key in self._fte_control_tails_by_task:
                    raise RuntimeError(f"FTE create task control already exists: {task_key}")
                create_ref = self.actor_handle.fte_create_task.remote(request)
                self._fte_control_tails_by_task[task_key] = create_ref
                self._fte_control_query_by_task[task_key] = query_id
                self._fte_control_operation_by_task[task_key] = "fte_create_task"

            def finish_create() -> None:
                with self._fte_control_lock:
                    if self._fte_control_tails_by_task.get(task_key) is not create_ref:
                        return
                    self._fte_control_tails_by_task.pop(task_key, None)
                    self._fte_control_query_by_task.pop(task_key, None)
                    self._fte_control_operation_by_task.pop(task_key, None)

            transfer_fte_registry_operations_to_ref(
                [query_id],
                create_ref,
                on_success=finish_create,
                on_failure=finish_create,
            )
            owns_registry_operation = False
            raw_status = self._get_fte_control_ref("fte_create_task", create_ref)
            if not isinstance(raw_status, dict):
                raise TypeError("worker actor fte_create_task must return a dict")
            status = dict(raw_status)
            validate_fte_status_identity(status, attempt_id)
            if status.get("_fte_control_operation") != "fte_create_task":
                raise RuntimeError(
                    "FTE create task control operation mismatch: "
                    f"task={task_key} actual={status.get('_fte_control_operation')!r}"
                )
            if status.get("_fte_control_applied") is not True:
                raise RuntimeError(f"FTE create task control was not applied: task={task_key}")
            return status
        except BaseException:
            if owns_registry_operation:
                with self._fte_control_lock:
                    if self._fte_control_tails_by_task.get(task_key) is create_ref:
                        self._fte_control_tails_by_task.pop(task_key, None)
                        self._fte_control_query_by_task.pop(task_key, None)
                        self._fte_control_operation_by_task.pop(task_key, None)
                end_fte_registry_operation(query_id)
            raise

    def _submit_tracked_fte_drop_ref(self, query_id: str, method_name: str) -> Any:
        begin_fte_registry_teardown_operation(query_id)
        owns_registry_operation = True
        try:
            attempts = _fte_control_rpc_max_attempts()
            backoff_s = _fte_control_rpc_initial_backoff_s()
            last_error: BaseException | None = None
            drop_ref = None
            for attempt in range(attempts):
                try:
                    method = getattr(self.actor_handle, method_name)
                    drop_ref = method.remote(query_id)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 >= attempts:
                        raise
                    if backoff_s > 0:
                        time.sleep(backoff_s)
                        backoff_s = min(backoff_s * 2, 5.0)
            if drop_ref is None:
                assert last_error is not None
                raise last_error
            transfer_fte_registry_teardown_operations_to_ref(
                [query_id],
                drop_ref,
            )
            owns_registry_operation = False
            return drop_ref
        finally:
            if owns_registry_operation:
                end_fte_registry_teardown_operation(query_id)

    def fte_add_splits(
        self,
        task_id: str | dict[str, Any],
        source_node_id: str,
        splits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raw_status = self._enqueue_ordered_fte_control_rpc("fte_add_splits", task_id, source_node_id, splits)
        return self._validated_fte_add_splits_status(task_id, raw_status)

    def enqueue_fte_add_splits(
        self,
        task_id: str | dict[str, Any],
        source_node_id: str,
        splits: list[dict[str, Any]],
        *,
        cancel_event: Any = None,
    ) -> dict[str, Any]:
        raw_status = self._enqueue_ordered_fte_control_rpc(
            "fte_add_splits",
            task_id,
            source_node_id,
            splits,
            cancel_event=cancel_event,
        )
        return self._validated_fte_add_splits_status(task_id, raw_status)

    def fte_no_more_splits(
        self,
        task_id: str | dict[str, Any],
        source_node_id: str,
    ) -> dict[str, Any]:
        raw_status = self._enqueue_ordered_fte_control_rpc("fte_no_more_splits", task_id, source_node_id)
        if not isinstance(raw_status, dict):
            raise TypeError("worker actor fte_no_more_splits must return a dict")
        return dict(raw_status)

    def enqueue_fte_no_more_splits(
        self,
        task_id: str | dict[str, Any],
        source_node_id: str,
    ) -> dict[str, Any]:
        raw_status = self._enqueue_ordered_fte_control_rpc("fte_no_more_splits", task_id, source_node_id)
        if not isinstance(raw_status, dict):
            raise TypeError("worker actor fte_no_more_splits must return a dict")
        return dict(raw_status)

    def fte_update_task(
        self,
        task_id: str | dict[str, Any],
        update: dict[str, Any],
    ) -> dict[str, Any]:
        raw_status = self._enqueue_ordered_fte_control_rpc("fte_update_task", task_id, update)
        if not isinstance(raw_status, dict):
            raise TypeError("worker actor fte_update_task must return a dict")
        return dict(raw_status)

    def enqueue_fte_update_task(
        self,
        task_id: str | dict[str, Any],
        update: dict[str, Any],
    ) -> dict[str, Any]:
        raw_status = self._enqueue_ordered_fte_control_rpc("fte_update_task", task_id, update)
        if not isinstance(raw_status, dict):
            raise TypeError("worker actor fte_update_task must return a dict")
        return dict(raw_status)

    def fte_get_task_status(
        self,
        task_id: str | dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        # Honor caller timeout exactly so progress refresh can stay best-effort.
        client_timeout_s = None if timeout_s is None else max(0.0, float(timeout_s))
        raw_status = self._fte_control_rpc(
            "fte_get_task_status",
            task_id,
            timeout_s=client_timeout_s,
        )
        if not isinstance(raw_status, dict):
            raise TypeError("worker actor fte_get_task_status must return a dict")
        return dict(raw_status)

    def fte_wait_task_status(
        self,
        task_id: str | dict[str, Any],
        min_version: int | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        # Server-side wait uses timeout_s as a long-poll budget. Client ray.get
        # must still be bounded, but with enough slack for a busy worker actor
        # to schedule the wait method. Soft timeouts are retried by callers;
        # they must not be treated as hard worker death.
        server_timeout_s = None if timeout_s is None else max(0.0, float(timeout_s))
        client_timeout_s = None
        if server_timeout_s is not None:
            client_timeout_s = max(30.0, server_timeout_s + 5.0)
        raw_status = self._fte_control_rpc(
            "fte_wait_task_status",
            task_id,
            min_version,
            server_timeout_s,
            timeout_s=client_timeout_s,
        )
        if not isinstance(raw_status, dict):
            raise TypeError("worker actor fte_wait_task_status must return a dict")
        return dict(raw_status)

    def fte_wait_task_status_interruptible(
        self,
        task_id: str | dict[str, Any],
        min_version: int | None,
        timeout_s: float | None,
        stop_event: Any,
    ) -> dict[str, Any]:
        server_timeout_s = None if timeout_s is None else max(0.0, float(timeout_s))
        client_timeout_s = None
        if server_timeout_s is not None:
            client_timeout_s = max(30.0, server_timeout_s + 5.0)
        raw_status = self._fte_control_rpc(
            "fte_wait_task_status",
            task_id,
            min_version,
            server_timeout_s,
            timeout_s=client_timeout_s,
            cancel_event=stop_event,
        )
        if not isinstance(raw_status, dict):
            raise TypeError("worker actor fte_wait_task_status must return a dict")
        return dict(raw_status)

    def fte_wait_split_queue_has_space(
        self,
        task_id: str | dict[str, Any],
        source_node_id: str | None = None,
        max_buffered_splits: int | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        raw_status = self._fte_control_rpc(
            "fte_wait_split_queue_has_space",
            task_id,
            source_node_id,
            max_buffered_splits,
            timeout_s,
        )
        if not isinstance(raw_status, dict):
            raise TypeError("worker actor fte_wait_split_queue_has_space must return a dict")
        return dict(raw_status)

    def fte_wait_split_queue_has_space_interruptible(
        self,
        task_id: str | dict[str, Any],
        source_node_id: str | None,
        max_buffered_splits: int | None,
        timeout_s: float | None,
        stop_event: Any,
    ) -> dict[str, Any]:
        server_timeout_s = None if timeout_s is None else max(0.0, float(timeout_s))
        client_timeout_s = None
        if server_timeout_s is not None:
            client_timeout_s = max(30.0, server_timeout_s + 5.0)
        raw_status = self._fte_control_rpc(
            "fte_wait_split_queue_has_space",
            task_id,
            source_node_id,
            max_buffered_splits,
            server_timeout_s,
            timeout_s=client_timeout_s,
            cancel_event=stop_event,
        )
        if not isinstance(raw_status, dict):
            raise TypeError("worker actor fte_wait_split_queue_has_space must return a dict")
        return dict(raw_status)

    def fte_get_task_info(self, task_id: str | dict[str, Any]) -> dict[str, Any]:
        raw_info = self._fte_control_rpc("fte_get_task_info", task_id)
        if not isinstance(raw_info, dict):
            raise TypeError("worker actor fte_get_task_info must return a dict")
        return dict(raw_info)

    def enqueue_fte_ack_task_result(self, task_id: str | dict[str, Any]) -> Any:
        return self._enqueue_ordered_fte_control_ref("fte_ack_task_result", task_id)

    def enqueue_fte_release_task_result(self, task_id: str | dict[str, Any]) -> Any:
        return self._enqueue_ordered_fte_control_ref("fte_release_task_result", task_id)

    def close_and_flush_fte_controls(self, query_id: str) -> list[dict[str, Any]]:
        query_key = str(query_id or "").strip()
        if not query_key:
            return []
        close_fte_registry_for_query(query_key)
        with self._fte_control_lock:
            pending = [
                (
                    task_key,
                    ref,
                    self._fte_control_operation_by_task.get(task_key, ""),
                )
                for task_key, ref in self._fte_control_tails_by_task.items()
                if self._fte_control_query_by_task.get(task_key) == query_key
            ]
        if not pending:
            return []

        raw_statuses: list[Any] | None = None
        barrier_resolution_error: BaseException | None = None
        try:
            resolved = resolve_object_refs_blocking(
                [ref for _, ref, _ in pending],
                timeout=_fte_control_rpc_timeout_s(),
                honor_query_deadline=False,
            )
            if not isinstance(resolved, list) or len(resolved) != len(pending):
                raise TypeError("FTE result-control barrier must return one status per task")
            raw_statuses = resolved
        except BaseException as exc:
            barrier_resolution_error = exc

        status_entries: list[tuple[str, Any, dict[str, Any]]] = []
        terminal_entries: list[tuple[str, Any, str]] = []
        pending_entries: list[tuple[str, Any, str]] = []
        terminal_error_entries: list[tuple[str, Any, str]] = []

        if raw_statuses is None:
            resolved_entries: list[tuple[tuple[str, Any, str], Any]] = []
            for entry in pending:
                task_key, ref, _ = entry
                if not self._is_current_ordered_fte_control_ref(task_key, ref):
                    continue
                future_method = getattr(ref, "future", None)
                if not callable(future_method):
                    if self._is_current_ordered_fte_control_ref(task_key, ref):
                        terminal_entries.append(entry)
                        terminal_error_entries.append((task_key, ref, f"task={task_key}: control ref has no future()"))
                    continue
                try:
                    future = future_method()
                except BaseException as exc:
                    if self._is_current_ordered_fte_control_ref(task_key, ref):
                        terminal_entries.append(entry)
                        terminal_error_entries.append((task_key, ref, f"task={task_key}: {type(exc).__name__}: {exc}"))
                    continue
                try:
                    raw_status = future.result(timeout=0)
                except concurrent.futures.TimeoutError:
                    if not self._is_current_ordered_fte_control_ref(task_key, ref):
                        continue
                    done_value = getattr(future, "done", None)
                    try:
                        is_done = bool(done_value() if callable(done_value) else done_value)
                    except BaseException:
                        is_done = False
                    if is_done:
                        try:
                            raw_status = future.result()
                        except BaseException as terminal_exc:
                            if self._is_current_ordered_fte_control_ref(task_key, ref):
                                terminal_entries.append(entry)
                                terminal_error_entries.append(
                                    (
                                        task_key,
                                        ref,
                                        f"task={task_key}: {type(terminal_exc).__name__}: {terminal_exc}",
                                    )
                                )
                        else:
                            if self._is_current_ordered_fte_control_ref(task_key, ref):
                                terminal_entries.append(entry)
                                resolved_entries.append((entry, raw_status))
                    elif self._is_current_ordered_fte_control_ref(task_key, ref):
                        pending_entries.append(entry)
                    continue
                except BaseException as exc:
                    if self._is_current_ordered_fte_control_ref(task_key, ref):
                        terminal_entries.append(entry)
                        terminal_error_entries.append((task_key, ref, f"task={task_key}: {type(exc).__name__}: {exc}"))
                    continue
                if self._is_current_ordered_fte_control_ref(task_key, ref):
                    terminal_entries.append(entry)
                    resolved_entries.append((entry, raw_status))
        else:
            resolved_entries = [
                (entry, raw_status)
                for entry, raw_status in zip(pending, raw_statuses, strict=True)
                if self._is_current_ordered_fte_control_ref(entry[0], entry[1])
            ]
            terminal_entries = [entry for entry, _ in resolved_entries]

        for entry, raw_status in resolved_entries:
            task_key, ref, expected_operation = entry
            if not self._is_current_ordered_fte_control_ref(task_key, ref):
                continue
            try:
                if not isinstance(raw_status, Mapping):
                    raise TypeError("FTE result-control barrier status must be a mapping")
                status = dict(raw_status)
                validate_fte_status_identity(status, task_key)
                if str(status.get("_fte_control_operation") or "") != expected_operation:
                    raise RuntimeError(
                        "FTE control barrier operation mismatch: "
                        f"task={task_key} expected={expected_operation!r} "
                        f"actual={status.get('_fte_control_operation')!r}"
                    )
                if expected_operation == "fte_add_splits" and status.get("_fte_control_applied") is False:
                    status = self._validated_terminal_fte_add_splits_status(
                        task_key,
                        status,
                    )
                elif status.get("_fte_control_applied") is not True:
                    raise RuntimeError(
                        f"FTE control barrier operation was not applied: task={task_key} operation={expected_operation}"
                    )
            except BaseException as exc:
                if self._is_current_ordered_fte_control_ref(task_key, ref):
                    terminal_error_entries.append((task_key, ref, f"task={task_key}: {type(exc).__name__}: {exc}"))
                continue
            if self._is_current_ordered_fte_control_ref(task_key, ref):
                status_entries.append((task_key, ref, status))

        terminal_error: FteControlBarrierTerminalError | None = None
        with self._fte_control_lock:
            pending_entries = [
                entry for entry in pending_entries if self._fte_control_tails_by_task.get(entry[0]) is entry[1]
            ]
            terminal_entries = [
                entry for entry in terminal_entries if self._fte_control_tails_by_task.get(entry[0]) is entry[1]
            ]
            terminal_errors = [
                message
                for task_key, ref, message in terminal_error_entries
                if self._fte_control_tails_by_task.get(task_key) is ref
            ]
            statuses = [
                status
                for task_key, ref, status in status_entries
                if self._fte_control_tails_by_task.get(task_key) is ref
            ]
            for task_key, ref, _ in terminal_entries:
                if self._fte_control_tails_by_task.get(task_key) is ref:
                    self._fte_control_tails_by_task.pop(task_key, None)
                    self._fte_control_query_by_task.pop(task_key, None)
                    self._fte_control_operation_by_task.pop(task_key, None)
            if terminal_errors:
                terminal_error = FteControlBarrierTerminalError(
                    f"FTE control barrier reached terminal failures for {query_key}: " + "; ".join(terminal_errors)
                )
                # Terminal entries are removed from the control tail above. Keep
                # their failure visible across a simultaneous pending operation,
                # fence retry, or remote-drop retry until final storage cleanup.
                self._fte_prepare_terminal_errors.setdefault(query_key, terminal_error)
        if pending_entries:
            details = ["pending=" + ",".join(task_key for task_key, _, _ in pending_entries)]
            if barrier_resolution_error is not None:
                details.append(f"wait={type(barrier_resolution_error).__name__}: {barrier_resolution_error}")
            details.extend(terminal_errors)
            raise FteControlBarrierPendingError(
                f"FTE control barrier still has non-terminal operations for {query_key}: " + "; ".join(details)
            )
        if terminal_error is not None:
            raise terminal_error
        return statuses

    def _has_fte_control_state_for_query(self, query_id: str) -> bool:
        query_key = str(query_id or "").strip()
        with self._fte_control_lock:
            return any(owner_query_id == query_key for owner_query_id in self._fte_control_query_by_task.values())

    def _has_fte_teardown_state_for_query(self, query_id: str) -> bool:
        query_key = str(query_id or "").strip()
        with self._fte_control_lock:
            return (
                query_key in self._fte_drop_incomplete_queries
                or query_key in self._fragment_drop_incomplete_queries
                or query_key in self._fte_prepare_terminal_errors
            )

    def fte_cancel_task(self, task_id: str | dict[str, Any]) -> dict[str, Any]:
        raw_status = self._enqueue_ordered_fte_control_rpc(
            "fte_cancel_task",
            task_id,
            wait_for_owned_side_effects=True,
        )
        if not isinstance(raw_status, dict):
            raise TypeError("worker actor fte_cancel_task must return a dict")
        return dict(raw_status)

    def fte_prepare_drop_query(self, query_id: str) -> dict[str, int]:
        query_id = (query_id or "").strip()
        if not query_id:
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}
        close_fte_registry_for_query(query_id)
        with self._fte_control_lock:
            self._fte_drop_incomplete_queries.add(query_id)
        barrier_error: FteControlBarrierTerminalError | None = None
        barrier_pending_error: BaseException | None = None
        try:
            self.close_and_flush_fte_controls(query_id)
        except FteControlBarrierTerminalError as exc:
            barrier_error = exc
        except FteControlBarrierPendingError as exc:
            barrier_pending_error = exc
        except BaseException as exc:
            barrier_pending_error = exc
        fence_error: BaseException | None = None
        try:
            quiesce_fte_registry_for_query(query_id)
        except BaseException as exc:
            fence_error = exc
        if fence_error is not None:
            details = []
            if barrier_error is not None or barrier_pending_error is not None:
                control_error = barrier_pending_error if barrier_pending_error is not None else barrier_error
                details.append(f"barrier={type(control_error).__name__}: {control_error}")
            details.append(f"fence={type(fence_error).__name__}: {fence_error}")
            raise RuntimeError(
                f"FTE query teardown did not reach the remote-drop fence for {query_id}: " + "; ".join(details)
            ) from fence_error
        if barrier_pending_error is not None:
            raise FteControlBarrierPendingError(
                f"FTE query teardown retained pending control ownership for {query_id}: {barrier_pending_error}"
            ) from barrier_pending_error
        if barrier_error is not None:
            with self._fte_control_lock:
                self._fte_prepare_terminal_errors.setdefault(query_id, barrier_error)
        drop_error: BaseException | None = None
        local_error: BaseException | None = None
        result: dict[str, int] = {}
        try:
            drop_ref = self._submit_tracked_fte_drop_ref(query_id, "fte_prepare_drop_query")
            raw_result = self._get_fte_control_ref(
                "fte_prepare_drop_query",
                drop_ref,
                honor_query_deadline=False,
            )
            if not isinstance(raw_result, dict):
                raise TypeError("worker actor fte_prepare_drop_query must return a dict")
            result = {str(key): int(value) for key, value in raw_result.items()}
        except BaseException as exc:
            drop_error = exc
        if drop_error is None:
            try:
                self._drop_fragment_registration_state(query_id)
                _drop_fragment_plan_refs_for_query(query_id)
                _drop_fte_registry_for_query(query_id)
            except BaseException as exc:
                local_error = exc
        if drop_error is not None or local_error is not None:
            details = []
            if barrier_error is not None:
                details.append(f"barrier={type(barrier_error).__name__}: {barrier_error}")
            if drop_error is not None:
                details.append(f"drop={type(drop_error).__name__}: {drop_error}")
            if local_error is not None:
                details.append(f"local={type(local_error).__name__}: {local_error}")
            cause = (
                local_error if local_error is not None else (barrier_error if barrier_error is not None else drop_error)
            )
            raise RuntimeError(f"FTE query teardown failed for {query_id}: " + "; ".join(details)) from cause
        return result

    def fte_cleanup_query(self, query_id: str) -> dict[str, int]:
        query_id = (query_id or "").strip()
        if not query_id:
            return {
                "flight_shuffle_registry_entries_removed": 0,
                "flight_shuffle_storage_entries_removed": 0,
                "flight_shuffle_cleanup_errors": 0,
            }
        try:
            cleanup_ref = self._submit_tracked_fte_drop_ref(query_id, "fte_cleanup_query")
            raw_result = self._get_fte_control_ref(
                "fte_cleanup_query",
                cleanup_ref,
                honor_query_deadline=False,
            )
            if not isinstance(raw_result, dict):
                raise TypeError("worker actor fte_cleanup_query must return a dict")
            result = {str(key): int(value) for key, value in raw_result.items()}
        except BaseException as exc:
            raise RuntimeError(f"FTE query storage cleanup failed for {query_id}: {exc}") from exc
        with self._fte_control_lock:
            self._fte_drop_incomplete_queries.discard(query_id)
            terminal_error = self._fte_prepare_terminal_errors.pop(query_id, None)
        if terminal_error is not None:
            raise RuntimeError(
                f"FTE query teardown reached a terminal control failure for {query_id}: {terminal_error}"
            ) from terminal_error
        return result

    def fte_drop_query(self, query_id: str) -> dict[str, int]:
        result = self.fte_prepare_drop_query(query_id)
        result.update(self.fte_cleanup_query(query_id))
        return result
