# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Fault-injection coverage for the reconstructible Actor UDF contract."""

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
def test_ray_actor_reconstructs_and_replays_interrupted_vane_udf_call():
    script = r"""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import time
import uuid

import pyarrow as pa
import ray

import vane
from vane.runners.ray.driver import RayQueryDriverClient


CONTROL_NAMESPACE = f"vane-actor-retry-{uuid.uuid4().hex}"
CONTROL_NAME = f"actor-retry-control-{uuid.uuid4().hex}"
UDF_NAME = f"reconstructible_actor_{uuid.uuid4().hex}"
RELEASE_PATH = Path(tempfile.gettempdir()) / f"vane-actor-retry-release-{uuid.uuid4().hex}"


@ray.remote(num_cpus=0, max_concurrency=16)
class FaultControl:
    def __init__(self):
        self.class_init_count = {}
        self.call_count_by_batch = {}
        self.started_batch_id = None

    def record_class_init(self, actor_index):
        key = str(actor_index)
        self.class_init_count[key] = self.class_init_count.get(key, 0) + 1

    def record_batch_started(self, actor_index, batch_id):
        key = f"{actor_index}:{batch_id}"
        self.call_count_by_batch[key] = self.call_count_by_batch.get(key, 0) + 1
        if actor_index == 0 and self.started_batch_id is None:
            self.started_batch_id = batch_id

    def snapshot(self):
        return {
            "class_init_count": dict(self.class_init_count),
            "call_count_by_batch": dict(self.call_count_by_batch),
            "started_batch_id": self.started_batch_id,
        }


class BlockingReplicatedModel:
    def __init__(self):
        self.actor_index = int(os.environ["VANE_INTERNAL_RAY_ACTOR_INDEX"])
        self.control = ray.get_actor(CONTROL_NAME, namespace=CONTROL_NAMESPACE)
        ray.get(self.control.record_class_init.remote(self.actor_index))

    def __call__(self, values):
        python_values = values.to_pylist()
        batch_id = str(python_values[0]) if python_values else "empty"
        ray.get(self.control.record_batch_started.remote(self.actor_index, batch_id))
        if self.actor_index == 0:
            while not RELEASE_PATH.exists():
                time.sleep(0.05)
        return pa.array([self.actor_index] * len(values), type=pa.int32())


os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["VANE_ENABLE_UDF_TEST_HOOKS"] = "1"
ray.init(
    address="local",
    namespace=CONTROL_NAMESPACE,
    num_cpus=8,
    ignore_reinit_error=True,
    include_dashboard=False,
)
control = FaultControl.options(name=CONTROL_NAME, namespace=CONTROL_NAMESPACE).remote()
vane.configure(runner="ray")

ReplicatedModel = vane.cls.batch(
    actor_number=2,
    return_dtype=pa.int32(),
    batch_size=256,
    name=UDF_NAME,
)(BlockingReplicatedModel)

con = vane.connect()
client = None
future = None
input_dir = tempfile.TemporaryDirectory()
try:
    input_path = os.path.join(input_dir.name, "actor_retry_input.parquet")
    con.execute(
        f"COPY (SELECT i::INTEGER AS value FROM range(4097) t(i)) "
        f"TO '{input_path}' (FORMAT PARQUET)"
    )
    relation = con.sql(f"SELECT value FROM read_parquet('{input_path}')")
    output = relation.select(
        vane.col("value"),
        ReplicatedModel()(vane.col("value")).alias("actor_index"),
    )
    logical_plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        output,
        f"actor-retry-{uuid.uuid4().hex}",
    )
    plan_id = str(logical_plan.idx())
    client = RayQueryDriverClient()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: list(client.stream_plan(logical_plan)))
        deadline = time.monotonic() + 45.0
        snapshot = None
        while time.monotonic() < deadline:
            if future.done():
                future.result()
            snapshot = ray.get(control.snapshot.remote(), timeout=5)
            if snapshot["started_batch_id"] is not None:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("actor 0 did not reach the batch-start barrier")

        target_batch_id = snapshot["started_batch_id"]
        target_key = f"0:{target_batch_id}"
        initial_init_count = snapshot["class_init_count"]["0"]
        initial_call_count = snapshot["call_count_by_batch"][target_key]
        assert initial_init_count >= 1
        assert initial_call_count >= 1
        print("ACTOR_STARTED", snapshot, flush=True)

        actor_handle = client.get_test_udf_actor_handle(plan_id, UDF_NAME, actor_index=0)
        print("ACTOR_HANDLE_RESOLVED", flush=True)
        ray.kill(actor_handle, no_restart=False)
        print("ACTOR_KILLED", flush=True)

        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            if future.done():
                future.result()
            snapshot = ray.get(control.snapshot.remote(), timeout=5)
            if (
                snapshot["class_init_count"].get("0", 0) >= initial_init_count + 1
                and snapshot["call_count_by_batch"].get(target_key, 0) >= initial_call_count + 1
            ):
                break
            time.sleep(0.02)
        else:
            raise AssertionError(f"interrupted actor call was not reconstructed and replayed: {snapshot}")

        print("ACTOR_REPLAY_OBSERVED", snapshot, flush=True)
        RELEASE_PATH.touch()
        print("ACTOR_RELEASED", flush=True)
        partitions = future.result(timeout=45.0)
        payloads = [partition.partition() for partition in partitions]
        assert payloads
        table = pa.concat_tables(payloads) if len(payloads) > 1 else payloads[0]
        table = table.rename_columns(["value", "actor_index"])
        rows = sorted(table.to_pylist(), key=lambda row: row["value"])
        assert table.num_rows == 4097
        assert [row["value"] for row in rows] == list(range(4097))
        assert {row["actor_index"] for row in rows} == {0, 1}
        print("QUERY_FINISHED", flush=True)

    final_snapshot = ray.get(control.snapshot.remote(), timeout=5)
    assert final_snapshot["class_init_count"]["0"] >= initial_init_count + 1
    assert final_snapshot["call_count_by_batch"][target_key] >= initial_call_count + 1
    print("ACTOR_REPLAY", final_snapshot, flush=True)
finally:
    RELEASE_PATH.touch(exist_ok=True)
    if future is not None and not future.done():
        try:
            future.result(timeout=15.0)
        except Exception:
            pass
    if client is not None:
        print("CLIENT_CLOSE_START", flush=True)
        client.close()
        print("CLIENT_CLOSE_DONE", flush=True)
    con.close()
    input_dir.cleanup()
    print("RAY_SHUTDOWN_START", flush=True)
    ray.shutdown()
    print("RAY_SHUTDOWN_DONE", flush=True)
    RELEASE_PATH.unlink(missing_ok=True)
"""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=Path.cwd(),
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or ""
        pytest.fail(f"fault-injection subprocess timed out\nstdout:\n{stdout}\nstderr:\n{stderr}")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ACTOR_REPLAY" in result.stdout
