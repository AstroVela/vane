# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Fault-injection coverage for the stateful expression UDF v1 contract."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("ray")


def test_stateful_ray_actor_loss_fails_without_state_reset_or_batch_replay():
    script = r"""
from concurrent.futures import ThreadPoolExecutor
import os
import tempfile
import time
import uuid

import vane
import pyarrow as pa
import ray
import vane

from vane.runners.ray.driver import RayQueryDriverClient


CONTROL_NAMESPACE = f"vane-stateful-fault-{uuid.uuid4().hex}"
CONTROL_NAME = f"stateful-fault-control-{uuid.uuid4().hex}"
UDF_NAME = f"stateful_fault_counter_{uuid.uuid4().hex}"


@ray.remote(max_concurrency=16)
class FaultControl:
    def __init__(self):
        self.class_init_count = 0
        self.call_count_by_batch = {}
        self.started_batch_id = None
        self.released = False

    def record_class_init(self):
        self.class_init_count += 1

    def record_batch_started(self, batch_id):
        self.call_count_by_batch[batch_id] = self.call_count_by_batch.get(batch_id, 0) + 1
        if self.started_batch_id is None:
            self.started_batch_id = batch_id

    def should_release(self):
        return self.released

    def release(self):
        self.released = True

    def snapshot(self):
        return {
            "class_init_count": self.class_init_count,
            "call_count_by_batch": dict(self.call_count_by_batch),
            "started_batch_id": self.started_batch_id,
        }


class BlockingStatefulCounter:
    def __init__(self):
        self.control = ray.get_actor(CONTROL_NAME, namespace=CONTROL_NAMESPACE)
        ray.get(self.control.record_class_init.remote())
        self.calls = 0

    def __call__(self, table):
        self.calls += 1
        values = table.column("value").to_pylist()
        batch_id = str(values[0]) if values else "empty"
        ray.get(self.control.record_batch_started.remote(batch_id))
        while not ray.get(self.control.should_release.remote()):
            time.sleep(0.01)
        return pa.table({"state": [self.calls] * table.num_rows})


os.environ["VANE_ENABLE_UDF_TEST_HOOKS"] = "1"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
ray.init(
    address="local",
    namespace=CONTROL_NAMESPACE,
    num_cpus=4,
    ignore_reinit_error=True,
    include_dashboard=False,
)
control = FaultControl.options(name=CONTROL_NAME, namespace=CONTROL_NAMESPACE).remote()
vane.configure(runner="ray")

StatefulCounter = vane.cls.batch(
    actor_number=1,
    schema={"state": "INTEGER"},
    name=UDF_NAME,
    row_preserving=True,
)(BlockingStatefulCounter)

con = vane.connect()
client = None
future = None
input_dir = tempfile.TemporaryDirectory()
try:
    input_path = os.path.join(input_dir.name, "stateful_fault_input.parquet")
    con.execute(
        f"COPY (select i::INTEGER as value from range(4097) t(i)) "
        f"TO '{input_path}' (FORMAT PARQUET)"
    )
    relation = con.sql(f"select value from read_parquet('{input_path}')")
    output = relation.select(
        vane.col("value"),
        StatefulCounter()(value=vane.col("value")).alias("state"),
    )
    logical_plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        output,
        f"stateful-fault-{uuid.uuid4().hex}",
    )
    plan_id = str(logical_plan.idx())
    client = RayQueryDriverClient()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: list(client.stream_plan(logical_plan)))
        try:
            deadline = time.monotonic() + 45.0
            snapshot = None
            while time.monotonic() < deadline:
                if future.done():
                    future.result()
                snapshot = ray.get(control.snapshot.remote())
                if snapshot["started_batch_id"] is not None:
                    break
                time.sleep(0.02)
            else:
                raise AssertionError("stateful UDF did not reach the batch_started barrier")

            target_batch_id = snapshot["started_batch_id"]
            assert snapshot["class_init_count"] == 1
            assert snapshot["call_count_by_batch"][target_batch_id] == 1

            actor_handle = client.get_test_udf_actor_handle(plan_id, UDF_NAME)
            actor_id = actor_handle._actor_id.hex()
            ray.kill(actor_handle, no_restart=True)

            try:
                future.result(timeout=45.0)
            except Exception as exc:
                message = str(exc)
                assert UDF_NAME in message, message
                assert actor_id in message, message
                assert "state was not recoverable" in message, message
                print("FAULT_ERROR", message, flush=True)
            else:
                raise AssertionError("stateful query returned a result after its actor was killed")
        finally:
            try:
                ray.get(control.release.remote(), timeout=5)
            except Exception:
                pass

    final_snapshot = ray.get(control.snapshot.remote())
    assert final_snapshot["class_init_count"] == 1
    assert final_snapshot["call_count_by_batch"][target_batch_id] == 1
    print("FAULT_COUNTS", final_snapshot, flush=True)
finally:
    try:
        ray.get(control.release.remote(), timeout=5)
    except Exception:
        pass
    if future is not None and not future.done():
        try:
            future.result(timeout=15.0)
        except Exception:
            pass
    if client is not None:
        client.close()
    con.close()
    input_dir.cleanup()
    ray.shutdown()
"""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["VANE_ENABLE_UDF_TEST_HOOKS"] = "1"
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
    assert "state was not recoverable" in result.stdout
    assert "FAULT_COUNTS" in result.stdout
