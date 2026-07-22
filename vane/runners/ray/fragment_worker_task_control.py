# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Mapping
from typing import Any

from vane.runners.fte import FteTaskAttemptId, validate_fte_status_identity
from vane.runners.ray.fragment_registry import (
    _FTE_CLOSING_QUERIES,
    _FTE_REGISTRY_LOCK,
)
from vane.runners.ray.fragment_worker_control import (
    enqueue_ordered_fte_control,
)
from vane.runners.ray.fte_fragment_scheduler import (
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
from vane.runners.ray.fte_scheduler_config import (
    _fte_control_rpc_initial_backoff_s,
    _fte_control_rpc_max_attempts,
    _fte_control_rpc_timeout_s,
)
from vane.runners.ray.safe_get import (
    QueryDeadlineExceeded,
    configured_ray_get_timeout_s,
    resolve_object_refs_blocking,
)


class FteControlBarrierPendingError(RuntimeError):
    """The query control cut still owns at least one non-terminal ObjectRef."""


class FteControlBarrierTerminalError(RuntimeError):
    """The control cut is terminal, but one or more operations failed validation."""


class FteWorkerTaskControlMixin:
    def _fte_control_rpc(
        self,
        method_name: str,
        *args,
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
    def _get_fte_control_ref(
        method_name: str,
        ref: Any,
        timeout_s: float | None = None,
        cancel_event: Any = None,
        honor_query_deadline: bool = True,
    ) -> Any:
        resolved_timeout_s = _fte_control_rpc_timeout_s() if timeout_s is None else max(0.0, float(timeout_s))
        if cancel_event is not None:
            resolved_timeout_s = configured_ray_get_timeout_s(resolved_timeout_s)
            deadline = None if resolved_timeout_s is None else time.monotonic() + resolved_timeout_s
            future = ref.future()
            while True:
                if cancel_event.is_set():
                    try:
                        import ray

                        ray.cancel(ref, force=False)
                    except Exception:
                        pass
                    raise InterruptedError(f"{method_name} interrupted by watcher shutdown")
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining is not None and remaining <= 0.0:
                    raise TimeoutError(f"{method_name} did not complete within {resolved_timeout_s:.3f}s")
                wait_slice = 0.05 if remaining is None else min(0.05, remaining)
                try:
                    return future.result(timeout=wait_slice)
                except concurrent.futures.TimeoutError:
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
            raise TimeoutError(f"{method_name} did not complete within {resolved_timeout_s:.3f}s") from exc

    def _enqueue_ordered_fte_control_rpc(
        self,
        method_name: str,
        task_id: str | dict[str, Any],
        *args: Any,
        timeout_s: float | None = None,
    ) -> Any:
        ref = self._enqueue_ordered_fte_control_ref(method_name, task_id, *args)
        return self._get_fte_control_ref(method_name, ref, timeout_s=timeout_s)

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

    def _submit_tracked_fte_drop_ref(self, query_id: str) -> Any:
        begin_fte_registry_teardown_operation(query_id)
        owns_registry_operation = True
        try:
            attempts = _fte_control_rpc_max_attempts()
            backoff_s = _fte_control_rpc_initial_backoff_s()
            last_error: BaseException | None = None
            drop_ref = None
            for attempt in range(attempts):
                try:
                    drop_ref = self.actor_handle.fte_drop_query.remote(query_id)
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
        if not isinstance(raw_status, dict):
            raise TypeError("worker actor fte_add_splits must return a dict")
        return dict(raw_status)

    def enqueue_fte_add_splits(
        self,
        task_id: str | dict[str, Any],
        source_node_id: str,
        splits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raw_status = self._enqueue_ordered_fte_control_rpc("fte_add_splits", task_id, source_node_id, splits)
        if not isinstance(raw_status, dict):
            raise TypeError("worker actor fte_add_splits must return a dict")
        return dict(raw_status)

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

        statuses: list[dict[str, Any]] = []
        terminal_entries: list[tuple[str, Any, str]] = []
        pending_entries: list[tuple[str, Any, str]] = []
        terminal_errors: list[str] = []

        if raw_statuses is None:
            resolved_entries: list[tuple[tuple[str, Any, str], Any]] = []
            for entry in pending:
                task_key, ref, _ = entry
                future_method = getattr(ref, "future", None)
                if not callable(future_method):
                    terminal_entries.append(entry)
                    terminal_errors.append(f"task={task_key}: control ref has no future()")
                    continue
                try:
                    future = future_method()
                except BaseException as exc:
                    terminal_entries.append(entry)
                    terminal_errors.append(f"task={task_key}: {type(exc).__name__}: {exc}")
                    continue
                try:
                    raw_status = future.result(timeout=0)
                except concurrent.futures.TimeoutError:
                    done_value = getattr(future, "done", None)
                    try:
                        is_done = bool(done_value() if callable(done_value) else done_value)
                    except BaseException:
                        is_done = False
                    if is_done:
                        terminal_entries.append(entry)
                        try:
                            raw_status = future.result()
                        except BaseException as terminal_exc:
                            terminal_errors.append(f"task={task_key}: {type(terminal_exc).__name__}: {terminal_exc}")
                        else:
                            resolved_entries.append((entry, raw_status))
                    else:
                        pending_entries.append(entry)
                    continue
                except BaseException as exc:
                    terminal_entries.append(entry)
                    terminal_errors.append(f"task={task_key}: {type(exc).__name__}: {exc}")
                    continue
                terminal_entries.append(entry)
                resolved_entries.append((entry, raw_status))
        else:
            terminal_entries = list(pending)
            resolved_entries = list(zip(pending, raw_statuses, strict=True))

        for (task_key, _, expected_operation), raw_status in resolved_entries:
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
                if status.get("_fte_control_applied") is not True:
                    raise RuntimeError(
                        f"FTE control barrier operation was not applied: task={task_key} operation={expected_operation}"
                    )
            except BaseException as exc:
                terminal_errors.append(f"task={task_key}: {type(exc).__name__}: {exc}")
                continue
            statuses.append(status)

        with self._fte_control_lock:
            for task_key, ref, _ in terminal_entries:
                if self._fte_control_tails_by_task.get(task_key) is ref:
                    self._fte_control_tails_by_task.pop(task_key, None)
                    self._fte_control_query_by_task.pop(task_key, None)
                    self._fte_control_operation_by_task.pop(task_key, None)

        if pending_entries:
            details = ["pending=" + ",".join(task_key for task_key, _, _ in pending_entries)]
            if barrier_resolution_error is not None:
                details.append(f"wait={type(barrier_resolution_error).__name__}: {barrier_resolution_error}")
            details.extend(terminal_errors)
            raise FteControlBarrierPendingError(
                f"FTE control barrier still has non-terminal operations for {query_key}: " + "; ".join(details)
            )
        if terminal_errors:
            details = list(terminal_errors)
            raise FteControlBarrierTerminalError(
                f"FTE control barrier reached terminal failures for {query_key}: " + "; ".join(details)
            )
        return statuses

    def _has_fte_control_state_for_query(self, query_id: str) -> bool:
        query_key = str(query_id or "").strip()
        with self._fte_control_lock:
            return any(owner_query_id == query_key for owner_query_id in self._fte_control_query_by_task.values())

    def _has_fte_teardown_state_for_query(self, query_id: str) -> bool:
        query_key = str(query_id or "").strip()
        with self._fte_control_lock:
            return query_key in self._fte_drop_incomplete_queries or query_key in self._fragment_drop_incomplete_queries

    def fte_cancel_task(self, task_id: str | dict[str, Any]) -> dict[str, Any]:
        raw_status = self._enqueue_ordered_fte_control_rpc("fte_cancel_task", task_id)
        if not isinstance(raw_status, dict):
            raise TypeError("worker actor fte_cancel_task must return a dict")
        return dict(raw_status)

    def fte_drop_query(self, query_id: str) -> dict[str, int]:
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
        drop_error: BaseException | None = None
        local_error: BaseException | None = None
        result: dict[str, int] = {}
        try:
            drop_ref = self._submit_tracked_fte_drop_ref(query_id)
            raw_result = self._get_fte_control_ref(
                "fte_drop_query",
                drop_ref,
                honor_query_deadline=False,
            )
            if not isinstance(raw_result, dict):
                raise TypeError("worker actor fte_drop_query must return a dict")
            result = {str(key): int(value) for key, value in raw_result.items()}
        except BaseException as exc:
            drop_error = exc
        if drop_error is None:
            try:
                self._drop_fragment_registration_state(query_id)
                _drop_fragment_plan_refs_for_query(query_id)
                _drop_fte_registry_for_query(query_id)
                with self._fte_control_lock:
                    self._fte_drop_incomplete_queries.discard(query_id)
            except BaseException as exc:
                local_error = exc
        if barrier_error is not None or drop_error is not None or local_error is not None:
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
