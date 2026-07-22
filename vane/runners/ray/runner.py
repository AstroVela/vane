# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Any

from vane._ray_cxx import require_ray_cxx_attr
from vane._ray_progress_env import configure_ray_progress_logging_defaults, ray_log_to_driver_default
from vane._vane_session import ensure_vane_session_dir

configure_ray_progress_logging_defaults()

from vane.runners.ray.driver import RayQueryDriverClient
from vane.runners.runner import Runner

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pyarrow as pa

    import vane


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
        force_client_mode: bool = False,
    ) -> None:
        self.ray_address = address
        self.max_task_backlog = max_task_backlog
        self.force_client_mode = force_client_mode
        ensure_vane_session_dir()
        _configure_scan_task_backlog_env(max_task_backlog)

        if not ray.is_initialized():
            ray.init(
                address=address,
                log_to_driver=ray_log_to_driver_default(),
            )

        self.query_driver_client: RayQueryDriverClient | None = None

    def close(self) -> None:
        if self.query_driver_client is not None:
            try:
                self.query_driver_client.close()
            finally:
                self.query_driver_client = None

    shutdown = close

    def run_iter(
        self, relation: vane.DuckDBPyRelation, results_buffer_size: int | None = None
    ) -> Iterator[RayMaterializedResult]:
        query_id = str(uuid.uuid4())

        PyLogicalPlan = require_ray_cxx_attr(
            "PyLogicalPlan",
            hint="Ensure the C++ ray extension is built and importable in worker processes.",
        )

        logical_plan = PyLogicalPlan.from_duckdb_relation(relation, query_id)

        if self.query_driver_client is None:
            self.query_driver_client = RayQueryDriverClient()

        # Send PyLogicalPlan to Driver — Driver will create physical plan
        yield from self.query_driver_client.stream_plan(
            logical_plan,
        )

    def run_write(self, relation: vane.DuckDBPyRelation) -> dict[str, Any]:
        """Execute a distributed COPY/write plan and return file metadata."""
        PyLogicalPlan = require_ray_cxx_attr(
            "PyLogicalPlan",
            hint="Ensure the C++ ray extension is built and importable in worker processes.",
        )

        query_id = str(uuid.uuid4())

        logical_plan = PyLogicalPlan.from_duckdb_relation(relation, query_id)

        if self.query_driver_client is None:
            self.query_driver_client = RayQueryDriverClient()

        # Send PyLogicalPlan to Driver — Driver will create physical plan
        return self.query_driver_client.run_copy_plan(logical_plan)

    def run_iter_tables(self, relation: Any, results_buffer_size: int | None = None) -> Iterator[pa.Table]:
        for result in self.run_iter(relation, results_buffer_size=results_buffer_size):
            yield result.partition()
