# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Production FTE task and resource graph setup for real native scan tests."""

from __future__ import annotations

from vane.execution.query_resource_spec import (
    QueryAllocation,
    QueryResourceGraph,
    ResourceUnitSpec,
    ResourceVector,
)
from vane.runners.ray.query_resource_graph_builder import native_fragment_unit_id_for_fragment
from vane.runners.ray.query_resource_runtime import register_query_resource_graph


class NativeScanWorkerTask:
    def __init__(
        self,
        *,
        query_id: str,
        node_id: str,
        split_batch: bytes,
        plan,
        fragment_node_id: str | None = None,
        name: str = "native-dynamic-scan-worker-task",
    ) -> None:
        self.query_id = query_id
        self.node_id = str(node_id)
        self.fragment_node_id = str(fragment_node_id if fragment_node_id is not None else node_id)
        self.split_batch = split_batch
        self._plan = plan
        self._name = str(name)

    def name(self):
        return self._name

    def context(self):
        fragment_id = f"{self.query_id}:node:{self.fragment_node_id}"
        return {
            "query_id": self.query_id,
            "node_id": self.fragment_node_id,
            "resource_query_id": self.query_id,
            "resource_unit_id": native_fragment_unit_id_for_fragment(self.query_id, fragment_id),
        }

    def task_context(self):
        try:
            last_node_id = int(self.node_id)
        except ValueError:
            last_node_id = 0
        return {"query_idx": 0, "last_node_id": last_node_id, "task_id": 0, "node_ids": [last_node_id]}

    def Inputs(self):
        return {self.node_id: {"kind": "scan_split_batch", "data": self.split_batch}}

    def exchange_sink_config(self):
        return None

    def plan(self):
        return self._plan


def register_scan_query(tasks) -> None:
    tasks = list(tasks)
    if not tasks:
        raise ValueError("scan query requires at least one task")
    query_ids = {str(task.context()["query_id"]) for task in tasks}
    if len(query_ids) != 1:
        raise ValueError("scan query tasks must share one query_id")
    query_id = query_ids.pop()
    fragment_ids = sorted({f"{query_id}:node:{task.context()['node_id']}" for task in tasks})
    target_output_block_bytes = 1024 * 1024
    units = tuple(
        ResourceUnitSpec(
            query_id=query_id,
            resource_unit_id=native_fragment_unit_id_for_fragment(query_id, fragment_id),
            physical_node_id=f"node:{fragment_id.rsplit(':node:', 1)[1]}:native-fragment",
            unit_kind="native_fragment",
            backend="ray_worker",
            input_unit_ids=(),
            per_task=ResourceVector(),
            target_output_block_bytes=target_output_block_bytes,
            generator_buffer_blocks=1,
            max_concurrency=max(1, len(tasks)),
        )
        for fragment_id in fragment_ids
    )
    allocation_resources = ResourceVector(
        cpu=max(1, len(tasks)),
        heap_bytes=max(1, len(tasks)) * 64 * 1024 * 1024,
        # These tests exercise FTE recovery rather than object-store
        # backpressure.  A streaming unit keeps 25% of the default allocation
        # protected for output handoff, so size the fixture such that the
        # remaining 75% admits every native generator window concurrently.
        object_store_bytes=(max(1, len(tasks)) * target_output_block_bytes * 4 + 2) // 3,
    )
    manager = register_query_resource_graph(
        QueryResourceGraph(
            query_id=query_id,
            plan_digest=f"sha256:scan:{query_id}",
            units=units,
            terminal_unit_ids=tuple(unit.resource_unit_id for unit in units),
        ),
        QueryAllocation(
            resources=allocation_resources,
            generation=1,
        ),
    )
    for unit in units:
        manager.update_unit_state(unit.resource_unit_id, runnable=True)
