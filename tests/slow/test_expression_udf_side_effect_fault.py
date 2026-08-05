# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Fault-injection coverage for side-effecting Ray actor UDF retries."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("ray")


@pytest.mark.real_ray
@pytest.mark.ray_cluster_owner
def test_side_effecting_ray_actor_crash_fails_without_replaying_call():
    script = r"""
import os
import time
import uuid

import ray

import duckdb.execution.udf_ray as udf_ray
from duckdb.execution.udf_ray_config import MAX_ACTOR_RESTARTS, MAX_ACTOR_TASK_RETRIES


CONTROL_NAMESPACE = f"vane-side-effect-fault-{uuid.uuid4().hex}"
CONTROL_NAME = f"side-effect-fault-control-{uuid.uuid4().hex}"


@ray.remote
class FaultControl:
    def __init__(self):
        self.call_count = 0

    def record_call(self):
        self.call_count += 1

    def snapshot(self):
        return self.call_count


class FakePlan:
    def __init__(self):
        self.handles = None

    def collect_udf_nodes(self, conn=None):
        return [
            {
                "node_id": 1,
                "actor_pool_size": 1,
                "gpus": 0.0,
                "payload": {
                    "execution_backend": "ray_actor",
                    "side_effects": True,
                    "resource_unit_id": "resource:side-effect-fault",
                },
            }
        ]

    def set_udf_actor_handles(self, handles, conn=None):
        self.handles = handles


class FaultPool:
    def __init__(
        self,
        *,
        payload,
        concurrency,
        gpus_per_actor,
        actor_node_ids,
        ray_options=None,
        max_restarts=MAX_ACTOR_RESTARTS,
        max_task_retries=MAX_ACTOR_TASK_RETRIES,
    ):
        assert concurrency == 1
        self.policy = (max_restarts, max_task_retries)

        @ray.remote(
            max_restarts=max_restarts,
            max_task_retries=max_task_retries,
            num_cpus=0,
        )
        class CrashAfterSideEffect:
            def pid(self):
                return os.getpid()

            def crash(self, control):
                ray.get(control.record_call.remote())
                os._exit(23)

        self.actors = [CrashAfterSideEffect.remote()]
        self._init_refs = []
        self._confirmed_ready = set()

    def shutdown(self):
        for actor in self.actors:
            ray.kill(actor, no_restart=True)


os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
ray.init(
    address="local",
    namespace=CONTROL_NAMESPACE,
    num_cpus=4,
    ignore_reinit_error=True,
    include_dashboard=False,
)
control = FaultControl.options(name=CONTROL_NAME, namespace=CONTROL_NAMESPACE).remote()
udf_ray.UDFActorPool = FaultPool
created = []
try:
    node_id = ray.get_runtime_context().get_node_id()
    created, _ = udf_ray.ensure_actor_pools_for_plan(
        FakePlan(),
        actor_node_ids_by_unit={"resource:side-effect-fault": (node_id,)},
        query_driver_handle=object(),
        query_generation_capability="test-query-generation-capability",
        session_config={},
    )
    assert len(created) == 1
    assert created[0].policy == (MAX_ACTOR_RESTARTS, 0)
    actor = created[0].actors[0]
    original_pid = ray.get(actor.pid.remote())

    try:
        ray.get(actor.crash.remote(control))
    except Exception as exc:
        print("FAULT_ERROR", type(exc).__name__, str(exc), flush=True)
    else:
        raise AssertionError("side-effecting actor call returned after its process exited")

    replacement_pid = ray.get(actor.pid.remote(), timeout=15)
    assert replacement_pid != original_pid
    deadline = time.monotonic() + 5.0
    call_count = 0
    while time.monotonic() < deadline:
        call_count = ray.get(control.snapshot.remote())
        if call_count:
            break
        time.sleep(0.02)
    assert call_count == 1
    time.sleep(1.0)
    assert ray.get(control.snapshot.remote()) == 1
    print("FAULT_COUNT", call_count, flush=True)
finally:
    for pool in created:
        pool.shutdown()
    ray.shutdown()
"""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAULT_ERROR" in result.stdout
    assert "FAULT_COUNT 1" in result.stdout
