# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import threading
import uuid
from typing import TYPE_CHECKING, Any

from vane.runners.ray.fragment_registry import (
    _FTE_FRAGMENT_EXECUTION_IDS,
    _FTE_QUERY_NEXT_FRAGMENT_EXECUTION_ID,
    _FTE_REGISTRY_LOCK,
    _FTE_SEQUENCES,
    _FTE_WORKER_HANDLES,
    _FteWorkerPressure,
)
from vane.runners.ray.fragment_worker_commands import FteWorkerCommandMixin
from vane.runners.ray.fragment_worker_events import FteWorkerEventHandlingMixin
from vane.runners.ray.fragment_worker_lifecycle import FteWorkerLifecycleMixin
from vane.runners.ray.fragment_worker_placement import FteWorkerPlacementMixin
from vane.runners.ray.fragment_worker_pressure_accounting import FteWorkerPressureAccountingMixin
from vane.runners.ray.fragment_worker_submission import FteWorkerSubmissionMixin
from vane.runners.ray.fragment_worker_task_control import FteWorkerTaskControlMixin
from vane.runners.ray.fragment_worker_transitions import FteWorkerTransitionMixin
from vane.runners.ray.fte_fragment_scheduler import (
    FteWorkerPlacementManager,
)

if TYPE_CHECKING:
    import ray

    from vane.runners.fte import FteFragmentExecution


class RayWorkerActorHandle(
    FteWorkerSubmissionMixin,
    FteWorkerEventHandlingMixin,
    FteWorkerTaskControlMixin,
    FteWorkerLifecycleMixin,
    FteWorkerPlacementMixin,
    FteWorkerCommandMixin,
    FteWorkerTransitionMixin,
    FteWorkerPressureAccountingMixin,
):
    """Wrapper around a RayWorkerActor."""

    def __init__(
        self,
        actor_handle: ray.actor.ActorHandle[Any],
        *,
        memory_capacity_bytes: int,
        node_id: str,
        worker_id: str,
        host: str | None = None,
        manager_instance_id: str | None = None,
    ):
        memory_capacity_bytes = int(memory_capacity_bytes)
        if memory_capacity_bytes <= 0:
            raise ValueError("memory_capacity_bytes must be positive")
        node_id = str(node_id).strip()
        if not node_id:
            raise ValueError("node_id must be non-empty")
        worker_id = str(worker_id).strip()
        if not worker_id:
            raise ValueError("worker_id must be non-empty")
        host = str(host if host is not None else worker_id.rsplit("#", 1)[0]).strip()
        if not host:
            raise ValueError("host must be non-empty")
        self.actor_handle = actor_handle
        self.worker_id = worker_id
        self._worker_incarnation_id = uuid.uuid4().hex
        self.node_id = node_id
        self.host = host
        self.manager_instance_id = str(manager_instance_id or "").strip()
        self.memory_capacity_bytes = memory_capacity_bytes
        self._registered_fragment_ids: set[str] = set()
        self._fragment_registration_refs: dict[str, Any] = {}
        self._fragment_query_ids: dict[str, str] = {}
        self._fragment_registration_lock = threading.RLock()
        self._fte_source_node_ids: set[str] = set()
        self._fte_fragment_execution_ids: dict[tuple[str, str], int] = {}
        self._fte_fragment_executions: dict[tuple[str, str], FteFragmentExecution] = {}
        self._fte_sequences: dict[tuple[str, str, str], int] = {}
        self._fte_pressure = _FteWorkerPressure()
        self._fte_healthy = True
        self._fte_worker_placement_manager = FteWorkerPlacementManager(self)
        self._fte_control_lock = threading.RLock()
        self._fte_control_tails_by_task: dict[str, Any] = {}
        self._fte_control_query_by_task: dict[str, str] = {}
        self._fte_control_operation_by_task: dict[str, str] = {}
        self._fte_drop_incomplete_queries: set[str] = set()
        self._fte_prepare_terminal_errors: dict[str, BaseException] = {}
        self._fragment_drop_incomplete_queries: set[str] = set()
        self._worker_shutdown_started = False
        self._fte_failure_retirement_completed = False
        self._fte_worker_failure_reconciliation_lock = threading.Lock()
        self._fte_worker_failure_reconciliation_complete = threading.Event()
        self._fte_worker_failure_reconciliation_error: BaseException | None = None
        with _FTE_REGISTRY_LOCK:
            current = _FTE_WORKER_HANDLES.get(worker_id)
            if current is not None and current is not self:
                raise RuntimeError(f"FTE worker id is already registered: {worker_id}")
            _FTE_WORKER_HANDLES[worker_id] = self

    @property
    def worker_incarnation_id(self) -> str:
        return self._worker_incarnation_id

    def wait_fte_worker_failure_reconciliation(self, *, timeout_s: float) -> None:
        """Wait until this worker incarnation's failure side effects are published."""

        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_s must be finite and > 0")
        if not self._fte_worker_failure_reconciliation_complete.wait(timeout):
            raise TimeoutError(
                "timed out waiting for FTE worker failure reconciliation: "
                f"worker={self.worker_id} incarnation={self.worker_incarnation_id}"
            )
        with self._fte_worker_failure_reconciliation_lock:
            error = self._fte_worker_failure_reconciliation_error
        if error is not None:
            raise error

    def _complete_fte_worker_failure_reconciliation(self, error: BaseException | None) -> None:
        with self._fte_worker_failure_reconciliation_lock:
            if self._fte_worker_failure_reconciliation_complete.is_set():
                return
            self._fte_worker_failure_reconciliation_error = error
            self._fte_worker_failure_reconciliation_complete.set()

    def close_session(self, session_id: str) -> None:
        import ray

        from vane.runners.ray.driver import _runtime_error_contains
        from vane.runners.ray.safe_get import resolve_object_refs_blocking

        try:
            resolve_object_refs_blocking(
                self.actor_handle.close_session.remote(str(session_id)),
                timeout=30,
                honor_query_deadline=False,
            )
        except BaseException as error:
            actor_error_type = getattr(ray.exceptions, "RayActorError", None)
            if isinstance(actor_error_type, type) and _runtime_error_contains(error, actor_error_type):
                # A dead worker process cannot retain this session's connection.
                return
            raise

    @staticmethod
    def _fte_task_handle_cls() -> type[Any]:
        # Lazy import to avoid a module import cycle with driver.py.
        from vane.runners.ray.driver import FteWorkerTaskHandle

        return FteWorkerTaskHandle

    def _next_fte_fragment_execution_id(self, query_id: str, fragment_id: str) -> int:
        key = (query_id, fragment_id)
        with _FTE_REGISTRY_LOCK:
            fragment_execution_id = _FTE_FRAGMENT_EXECUTION_IDS.get(key)
            if fragment_execution_id is None:
                fragment_execution_id = _FTE_QUERY_NEXT_FRAGMENT_EXECUTION_ID.get(query_id, 0)
                _FTE_QUERY_NEXT_FRAGMENT_EXECUTION_ID[query_id] = fragment_execution_id + 1
                _FTE_FRAGMENT_EXECUTION_IDS[key] = fragment_execution_id
            self._fte_fragment_execution_ids[key] = fragment_execution_id
        return fragment_execution_id

    def _next_fte_split_sequence(self, query_id: str, fragment_id: str, source_node_id: str) -> int:
        key = (query_id, fragment_id, source_node_id)
        with _FTE_REGISTRY_LOCK:
            sequence_id = _FTE_SEQUENCES.get(key, 0)
            _FTE_SEQUENCES[key] = sequence_id + 1
            self._fte_sequences[key] = sequence_id + 1
        return sequence_id


__all__ = [name for name in globals() if not name.startswith("__")]
