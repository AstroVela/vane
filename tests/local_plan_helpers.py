# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import ExitStack

import pyarrow as pa

import vane
from tests.result_stream_helpers import collect_result_stream
from vane.execution.udf_subprocess import ensure_local_subprocess_actor_pools_for_plan
from vane.runners.fte.backends.native import NativeFteWorkerManagerBackend
from vane.runners.local.runner import _InProcessFragmentExecutor, _preload_arrow_dataset_imports


def execute_local_physical_plan(connection, plan) -> pa.Table:
    """Execute a prepared coordinator plan through the local fragment scheduler."""
    _preload_arrow_dataset_imports()
    with ExitStack() as cleanup:
        pools, _ = ensure_local_subprocess_actor_pools_for_plan(plan, conn=connection)
        for pool in pools:
            cleanup.callback(pool.shutdown, kill=True)

        executor = _InProcessFragmentExecutor()
        cleanup.callback(executor.close)
        executor.retain_resources(connection, *pools)
        backend = NativeFteWorkerManagerBackend(execute_fn=executor, num_workers=1, max_running_tasks=1)
        cleanup.callback(backend.shutdown)
        runner = vane.ray_cxx.DistributedPhysicalPlanRunner(backend)
        cleanup.callback(runner.shutdown)
        cleanup.callback(runner.drop_query_fragments, plan.idx())

        partitions = collect_result_stream(runner.run_plan(plan, connection))
        return pa.concat_tables(partitions)
