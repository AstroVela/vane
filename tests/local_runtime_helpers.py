# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Failure-only diagnostics for native local-runtime acceptance tests."""

from __future__ import annotations

import faulthandler
import json
import os
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path


def _slot_snapshot(pool):
    with pool._lock:
        return {
            "closed": pool._closed,
            "available_slots": len(pool._available_slots),
            "active_leases": len(pool._active_slots),
            "ordinary_waiters": len(pool._waiters),
            "policy_sources": len(pool._sources) - int(0 in pool._sources),
            "dispatching": pool._dispatching,
            "dispatch_requested": pool._dispatch_requested,
            "authorities": [authority._state for authority in pool._authorities],
        }


def local_runtime_snapshot(runtime):
    from vane.execution import ref_bundle, udf_subprocess

    snapshot = {"runtime": runtime.resource_snapshot(), "transport": ref_bundle.local_shm_ref_budget_snapshot()}
    # Inspect existing pools only. Diagnostics must not create a runtime, take
    # admission, call the active authority.state(), or retain any payloads.
    task_runtime = udf_subprocess._GLOBAL_TASK_RUNTIME
    if task_runtime is not None:
        with task_runtime.cond:
            pools = list(task_runtime.pools.values())
            workers = [
                {
                    "pool": pool.key,
                    "active_workers": pool.active,
                    "idle_workers": len(pool.idle),
                    "spawning_workers": len(pool._spawning_workers),
                    "total_workers": pool.total,
                }
                for pool in pools
            ]
        snapshot["task_workers"] = {
            **task_runtime.stats(),
            "reserved_execution_slots": task_runtime.execution_capacity.reserved_slots,
            "pools": [
                {**worker, "admission": _slot_snapshot(pool.admission_slots)} for worker, pool in zip(workers, pools)
            ],
        }
    registry = runtime._registry
    with registry._condition:
        model_pools = [entry.pool for entry in registry._entries.values() if entry.pool is not None]
    snapshot["model_workers"] = [
        {**pool.stats(), "admission": _slot_snapshot(pool.admission_slots)} for pool in model_pools
    ]
    return snapshot


@contextmanager
def local_runtime_diagnostics(runtime, directory):
    """Capture before the caller cancels work or tears down its resources.

    Snapshots are passive, per-component observations, not an atomic global
    view. Dump threads first so a stuck resource lock still leaves evidence.
    """
    try:
        yield
    except BaseException:
        try:
            root = os.environ.get("VANE_TEST_DIAGNOSTICS_DIR")
            directory = Path(root) / f"{Path(directory).name}-{uuid.uuid4().hex}" if root else Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Local runtime failure diagnostics: {directory}", file=sys.stderr)
            with (directory / "threads.txt").open("w") as output:
                faulthandler.dump_traceback(file=output, all_threads=True)
            (directory / "resources.json").write_text(json.dumps(local_runtime_snapshot(runtime), indent=2))
        except Exception as error:
            print(f"Local runtime diagnostics failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
