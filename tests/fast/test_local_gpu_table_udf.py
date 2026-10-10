# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Query-owned GPU placement and native sinks; fake UUIDs need no CUDA."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import vane
from vane import pickle as vane_pickle
from vane.execution import local_gpu_devices, udf_subprocess
from vane.execution.udf import build_executor
from vane.execution.udf_local_gpu import LocalGpuModelAdapter
from vane.execution.udf_local_resources import LocalProcessCapacityError
from vane.execution.udf_model_pool import ModelPoolRegistry

DEVICES = ("GPU-aaaaaaaa-0000-0000-0000-000000000001", "GPU-bbbbbbbb-0000-0000-0000-000000000002")


class DeviceActor:
    def __call__(self, table):
        return table.append_column("device", pa.array([os.environ["CUDA_VISIBLE_DEVICES"]] * table.num_rows))


class DeviceRowActor:
    def __call__(self, row):
        return {**row, "device": os.environ["CUDA_VISIBLE_DEVICES"]}


def payload(actor=DeviceActor, replicas=1):
    return {
        "execution_backend": "subprocess_actor",
        "function_pickle": vane_pickle.dumps(actor),
        "call_mode": "map_batches",
        "actor_number": replicas,
        "gpus": 1,
        "cpus": 0.1,
    }


@pytest.fixture
def inventory(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    monkeypatch.setattr(local_gpu_devices, "discover_gpu_devices", lambda: DEVICES)
    yield
    assert not local_gpu_devices._owners.occupied


def test_discovery_uses_a_fresh_interpreter_with_inherited_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")

    def run(command, **kwargs):
        assert command[1] == "-I"
        assert command[-1].endswith("local_cuda_inventory.py")
        assert "env" not in kwargs
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,0"
        assert kwargs["timeout"] == 30 and kwargs["check"]
        return subprocess.CompletedProcess(command, 0, f'["{DEVICES[1]}", "{DEVICES[0]}"]')

    monkeypatch.setattr(local_gpu_devices.subprocess, "run", run)
    assert local_gpu_devices.discover_gpu_devices() == DEVICES[::-1]


@pytest.mark.parametrize("error", [OSError("no driver"), subprocess.TimeoutExpired("probe", 30)])
def test_discovery_reports_failures_without_claiming_devices(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(local_gpu_devices.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="discovery failed"):
        local_gpu_devices.reserve_gpu_devices(1)
    assert not local_gpu_devices._owners.occupied


def test_replica_reservation_is_atomic_under_contention(inventory):
    barrier = threading.Barrier(2)
    release = threading.Event()

    def reserve():
        barrier.wait(timeout=10)
        try:
            lease = local_gpu_devices.reserve_gpu_devices(2)
        except LocalProcessCapacityError:
            release.set()
            return None
        try:
            assert release.wait(10)
            return lease.devices
        finally:
            lease.release()

    with ThreadPoolExecutor(2) as threads:
        futures = [threads.submit(reserve) for _ in range(2)]
        assert sorted((future.result(15) for future in futures), key=lambda value: value is None) == [DEVICES, None]


def test_insufficient_inventory_starts_no_workers(inventory, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("capacity refusal started a worker")

    monkeypatch.setattr(udf_subprocess, "_SingleSubprocessExecutor", unexpected)
    with pytest.raises(ValueError, match="needs 3 devices"):
        udf_subprocess.LocalSubprocessActorPool(payload(replicas=3), 3)


@pytest.mark.parametrize("gpus", [0.5, 2])
def test_table_actor_rejects_unsupported_gpu_quantities_at_binding(inventory, gpus):
    with vane.connect() as connection:
        with pytest.raises(vane.InvalidInputException, match="exactly one GPU"):
            connection.sql("SELECT 1 AS x").map_batches(
                DeviceActor, schema={"x": vane.sqltypes.INTEGER}, actor_number=1, gpus=gpus
            )


def test_cpu_query_does_not_discover_cuda(inventory, monkeypatch):
    def unexpected():
        pytest.fail("CPU UDF attempted CUDA discovery")

    monkeypatch.setattr(local_gpu_devices, "discover_gpu_devices", unexpected)
    with vane.connect() as connection:
        relation = connection.sql("SELECT 7 AS x").map_batches(lambda table: table, schema={"x": vane.sqltypes.INTEGER})
        assert relation.fetchall() == [(7,)]


@pytest.mark.parametrize("method", ["map_batches", "flat_map"])
def test_original_table_actor_writes_native_parquet_and_releases_query_devices(inventory, tmp_path, method):
    with vane.connect() as connection:
        for iteration in range(2):
            relation = connection.sql("SELECT i::INTEGER AS x FROM range(257) t(i)")
            relation = getattr(relation, method)(
                DeviceActor if method == "map_batches" else DeviceRowActor,
                schema={"x": vane.sqltypes.INTEGER, "device": vane.sqltypes.VARCHAR},
                actor_number=2,
                gpus=1,
                batch_size=32,
            )
            destination = tmp_path / f"output-{iteration}.parquet"
            relation.write_parquet(str(destination))
            output = pq.read_table(destination)
            assert sorted(output.column("x").to_pylist()) == list(range(257))
            assert set(output.column("device").to_pylist()) <= set(DEVICES)
            assert not local_gpu_devices._owners.occupied


@pytest.mark.parametrize("phase", ["constructor", "call"])
def test_failed_native_query_releases_devices(inventory, tmp_path, phase):
    class Failure:
        def __init__(self):
            if phase == "constructor":
                raise ValueError("query GPU failure")

        def __call__(self, table):
            raise ValueError("query GPU failure")

    with vane.connect() as connection:
        relation = connection.sql("SELECT 1 AS x").map_batches(
            Failure, schema={"x": vane.sqltypes.INTEGER}, actor_number=1, gpus=1
        )
        with pytest.raises(Exception, match="query GPU failure"):
            relation.write_parquet(str(tmp_path / "failed.parquet"))
    assert not local_gpu_devices._owners.occupied


def test_cleanup_failure_keeps_gpu_residency_until_retry(inventory, monkeypatch):
    pool = udf_subprocess.LocalSubprocessActorPool(payload(replicas=2), 2)
    worker = pool._workers[0]
    process = worker._proc
    original_close = worker.close

    def fail(*args, **kwargs):
        raise OSError("injected GPU cleanup failure")

    try:
        monkeypatch.setattr(worker, "close", fail)
        with pytest.raises(RuntimeError, match="injected GPU cleanup failure"):
            pool.shutdown(kill=True)
        assert process.poll() is None
        with pytest.raises(LocalProcessCapacityError):
            local_gpu_devices.reserve_gpu_devices(1)
        assert local_gpu_devices._owners.occupied == set(DEVICES)
    finally:
        monkeypatch.setattr(worker, "close", original_close)
        pool.shutdown(kill=True)
    assert process.poll() is not None


def test_registered_model_retries_after_query_pool_releases_device(inventory):
    actor_payload = payload()
    pool = udf_subprocess.LocalSubprocessActorPool(actor_payload, 1, _gpu_devices=DEVICES[:1])
    with ModelPoolRegistry() as registry:
        adapter = LocalGpuModelAdapter(registry, devices=DEVICES[:1])
        identity = adapter.register(
            "model", version="1", session_id="session", session_config={}, payload=actor_payload, devices=DEVICES[:1]
        )
        try:
            with pytest.raises(LocalProcessCapacityError):
                registry.prewarm(identity)
        finally:
            pool.shutdown(kill=True)
        registry.prewarm(identity)
        assert registry.resource_snapshot()["reserved_resources"]["gpu"] == 1


def test_executor_cannot_borrow_different_or_closed_gpu_pool(inventory):
    actor_payload = payload()
    pool = udf_subprocess.LocalSubprocessActorPool(actor_payload, 1)
    try:
        with pytest.raises(ValueError, match="prepared local actor pool"):
            build_executor({**actor_payload, "gpus": 2}, {"local_actor_pool": pool})
    finally:
        pool.shutdown(kill=True)
    with pytest.raises(ValueError, match="prepared local actor pool"):
        build_executor(actor_payload, {"local_actor_pool": pool})


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork requires POSIX")
def test_fork_child_cannot_release_or_reassign_parent_devices(inventory):
    pool = udf_subprocess.LocalSubprocessActorPool(payload(replicas=2), 2)
    lease = pool._gpu_reservation
    locked = threading.Event()
    release = threading.Event()

    def hold_ownership_locks():
        with local_gpu_devices._owners.lock, pool._cond:
            locked.set()
            release.wait(15)

    holder = threading.Thread(target=hold_ownership_locks)
    holder.start()
    assert locked.wait(10)
    child = os.fork()
    if child == 0:
        try:
            # Exercise the finalizer entry point while both inherited locks
            # belong to a thread that does not exist in the fork child.
            pool.__del__()
            lease.release()
            try:
                local_gpu_devices.reserve_gpu_devices(1)
            except LocalProcessCapacityError:
                os._exit(0)
            os._exit(2)
        except BaseException:
            os._exit(3)
    try:
        deadline = time.monotonic() + 10
        while True:
            finished, status = os.waitpid(child, os.WNOHANG)
            if finished:
                assert os.waitstatus_to_exitcode(status) == 0
                break
            if time.monotonic() > deadline:
                os.kill(child, 9)
                os.waitpid(child, 0)
                pytest.fail("fork child acquired an inherited GPU ownership lock")
            time.sleep(0.01)
        assert local_gpu_devices._owners.occupied == set(DEVICES)
    finally:
        release.set()
        holder.join(timeout=10)
        pool.shutdown(kill=True)
