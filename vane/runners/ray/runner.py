# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
import uuid
import weakref
from typing import TYPE_CHECKING, Any

from vane._ray_cxx import require_ray_cxx_attr
from vane._ray_progress_env import ray_log_to_driver_default
from vane._vane_session import ensure_vane_session_dir
from vane.runners.ray.admission_ledger import BoundedReplayMap
from vane.runners.ray.driver import RayQueryDriverClient
from vane.runners.runner import Runner

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]

    import vane


_RAY_RUNNERS: weakref.WeakSet[RayRunner] = weakref.WeakSet()
# WeakSet operations may allocate and synchronously run DuckDB connection
# finalizers, which re-enter notify_connection_closed() on the same thread.
_RAY_RUNNERS_LOCK = threading.RLock()
_SESSION_CLOSE_REPLAY_CAPACITY = 65_536


def notify_connection_closed(session_id: str) -> None:
    """Close one logical Vane session on every live local RayRunner."""
    session_key = str(session_id).strip()
    if not session_key:
        raise ValueError("Vane session_id must not be empty")
    with _RAY_RUNNERS_LOCK:
        runners = tuple(_RAY_RUNNERS)
    errors: list[BaseException] = []
    for runner in runners:
        try:
            runner.close_session(session_key)
        except BaseException as exc:
            errors.append(exc)
    if errors:
        raise RuntimeError(
            f"failed to close Vane session {session_key} on {len(errors)} local Ray runner(s): "
            + "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
        ) from errors[0]


def _configure_scan_task_backlog_env(max_task_backlog: int | None) -> None:
    if max_task_backlog is None:
        return
    if max_task_backlog <= 0:
        os.environ.pop("VANE_RAY_MAX_TASK_BACKLOG", None)
        return
    value = str(int(max_task_backlog))
    os.environ["VANE_RAY_MAX_TASK_BACKLOG"] = value


import ray

from vane.runners.ray.partition_metadata import (  # noqa: F401 — re-export for consumers
    PartitionMetadata,
    PartitionMetadataAccessor,
    RayMaterializedResult,
)

# ---------------------------------------------------------------------------
# RayRunner — the main entry point for distributed DuckDB execution
# ---------------------------------------------------------------------------


class RayRunner(Runner):
    name = "ray"

    def __init__(
        self,
        address: str | None,
        max_task_backlog: int | None,
    ) -> None:
        self.ray_address = address
        self.max_task_backlog = max_task_backlog
        ensure_vane_session_dir()
        _configure_scan_task_backlog_env(max_task_backlog)

        if not ray.is_initialized():
            ray.init(
                address=address,
                log_to_driver=ray_log_to_driver_default(),
            )

        self.query_driver_client: RayQueryDriverClient | None = None
        self._session_ids: set[str] = set()
        self._closed_session_ids = BoundedReplayMap[str, bool](capacity=_SESSION_CLOSE_REPLAY_CAPACITY)
        self._session_lock = threading.RLock()
        self._closed = False
        with _RAY_RUNNERS_LOCK:
            _RAY_RUNNERS.add(self)

    def close(self) -> None:
        with self._session_lock:
            if self._closed:
                return
            client = self.query_driver_client
            if client is not None:
                client.close()
            self.query_driver_client = None
            self._session_ids.clear()
            self._closed_session_ids.clear()
            self._closed = True
        with _RAY_RUNNERS_LOCK:
            _RAY_RUNNERS.discard(self)

    shutdown = close

    def close_session(self, session_id: str) -> None:
        session_key = str(session_id).strip()
        with self._session_lock:
            self._closed_session_ids[session_key] = True
            if session_key not in self._session_ids:
                return
            client = self.query_driver_client
            if client is not None:
                client.close_session(session_key)
            self._session_ids.remove(session_key)

    def _client_for_session(self, session_id: str) -> RayQueryDriverClient:
        session_key = str(session_id).strip()
        with self._session_lock:
            if self._closed:
                raise RuntimeError("RayRunner is closed")
            if session_key in self._closed_session_ids:
                raise RuntimeError(f"Vane session is closed: {session_key}")
            client = self.query_driver_client
            if client is None:
                client = RayQueryDriverClient()
                self.query_driver_client = client
            self._session_ids.add(session_key)
            return client

    def run_iter(self, relation: vane.DuckDBPyRelation) -> Iterator[RayMaterializedResult]:
        # PyLogicalPlan is transport-only. This generator owns the source
        # relation until the client stream and its teardown have finished.
        try:
            query_id = str(uuid.uuid4())

            PyLogicalPlan = require_ray_cxx_attr(
                "PyLogicalPlan",
                hint="Ensure the C++ ray extension is built and importable in worker processes.",
            )

            logical_plan = PyLogicalPlan.from_duckdb_relation(relation, query_id)
            session_id = str(logical_plan.session_id())

            client = self._client_for_session(session_id)

            # Send PyLogicalPlan to Driver — Driver will create physical plan
            yield from client.stream_plan(
                logical_plan,
            )
        finally:
            del relation

    def run_write(self, relation: vane.DuckDBPyRelation) -> dict[str, Any]:
        """Execute a distributed COPY/write plan and return file metadata."""
        PyLogicalPlan = require_ray_cxx_attr(
            "PyLogicalPlan",
            hint="Ensure the C++ ray extension is built and importable in worker processes.",
        )

        query_id = str(uuid.uuid4())

        logical_plan = PyLogicalPlan.from_duckdb_relation(relation, query_id)
        session_id = str(logical_plan.session_id())

        client = self._client_for_session(session_id)

        # Send PyLogicalPlan to Driver — Driver will create physical plan
        return client.run_copy_plan(logical_plan)

    def retry_copy_cleanup(self, operation_id: str) -> dict[str, Any]:
        """Retry cleanup for a committed write without executing the write again."""
        with self._session_lock:
            if self._closed:
                raise RuntimeError("RayRunner is closed")
            client = self.query_driver_client
            if client is None:
                raise RuntimeError("RayRunner has no active query runtime client")
        return client.retry_copy_cleanup(operation_id)

    def run_iter_tables(self, relation: Any) -> Iterator[pa.Table]:
        try:
            for result in self.run_iter(relation):
                yield result.partition()
        finally:
            del relation
