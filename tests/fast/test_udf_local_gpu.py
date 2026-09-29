# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts using provisioned fake UUIDs and real subprocesses."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest

from vane import pickle as vane_pickle
from vane.execution import udf_subprocess
from vane.execution.resources import ResourceVector
from vane.execution.udf_actor_pool_lifecycle import OwnedActorPoolsError
from vane.execution.udf_lifecycle import ExecutionCancellationScope, ExecutionCancelledError
from vane.execution.udf_local_gpu import LocalGpuModelAdapter
from vane.execution.udf_model_pool import ModelPoolRegistry, ModelPoolResourceBusy

DEVICES = ("GPU-aaaaaaaa-0000-0000-0000-000000000001", "GPU-bbbbbbbb-0000-0000-0000-000000000002")


def _payload(function, replicas=1, **changes):
    if not isinstance(function, type):
        callback = function

        class FunctionActor:
            def __call__(self, table):
                return callback(table)

        function = FunctionActor
    return {
        "function_pickle": vane_pickle.dumps(function),
        "call_mode": "map_batches",
        "execution_backend": "subprocess_actor",
        "actor_number": replicas,
        "cpus": 0.5,
        "gpus": 1,
        "memory_bytes": 1024,
        **changes,
    }


def _register(adapter, payload, devices=DEVICES[:1], name="model", session_config=None):
    return adapter.register(
        name, version="v1", session_id="session", session_config=session_config or {}, payload=payload, devices=devices
    )


def _wait(predicate):
    deadline = time.monotonic() + 15
    while not predicate():
        assert time.monotonic() < deadline, "GPU lifecycle test did not reach its checkpoint"
        time.sleep(0.01)


def _call(pool):
    scope = ExecutionCancellationScope("gpu-test", 1)
    authority = pool.create_admission_authority()
    authority.request(0)
    admission = authority.take(0)

    def run(worker):
        worker.submit(pa.table({"x": [7]}))
        return worker.take_ready_result()

    try:
        return pool.submit(run, scope=scope, admission=admission).result(timeout=15)
    finally:
        scope.finish()
        admission.release()
        authority.close()


@pytest.mark.parametrize("devices", [[], "0", ["0"], ["GPU-aaaa"], ["MIG-aaaa"], [DEVICES[0], DEVICES[0].upper()]])
def test_device_inventory_rejects_ambiguous_or_duplicate_devices(devices):
    with ModelPoolRegistry() as registry:
        with pytest.raises((TypeError, ValueError), match="GPU"):
            LocalGpuModelAdapter(registry, devices=devices)


@pytest.mark.parametrize(
    "changes,devices",
    [
        ({"gpus": 0}, DEVICES[:1]),
        ({"gpus": 0.5}, DEVICES[:1]),
        ({"gpus": 2}, DEVICES[:1]),
        ({"gpus": True}, DEVICES[:1]),
        ({"execution_backend": "subprocess_task"}, DEVICES[:1]),
        ({"actor_number": 2}, DEVICES[:1]),
        ({"actor_number": 1.5}, DEVICES[:1]),
        ({"actor_number": True}, DEVICES[:1]),
        ({"actor_number": "1"}, DEVICES[:1]),
        ({}, DEVICES[1:]),
    ],
)
def test_invalid_gpu_registration_starts_nothing(changes, devices, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid registration started a subprocess")

    monkeypatch.setattr(udf_subprocess, "_SingleSubprocessExecutor", unexpected)
    with ModelPoolRegistry() as registry:
        adapter = LocalGpuModelAdapter(registry, devices=DEVICES[:1])
        with pytest.raises(ValueError):
            _register(adapter, _payload(lambda table: table, **changes), devices)
        assert registry.resource_snapshot()["registered_models"] == 0


def test_device_order_and_session_configuration_participate_in_model_identity():
    payload = _payload(lambda table: table, replicas=2)
    with ModelPoolRegistry() as registry:
        adapter = LocalGpuModelAdapter(registry, devices=DEVICES)
        first = _register(adapter, payload, DEVICES)
        reordered = _register(adapter, payload, tuple(reversed(DEVICES)))
        reconfigured = _register(adapter, payload, DEVICES, session_config={"AWS_VANE_GPU_TEST": "other"})
        assert first.initialization == reordered.initialization == reconfigured.initialization
        assert len({first.configuration, reordered.configuration, reconfigured.configuration}) == 3
        with pytest.raises(ValueError, match="already registered"):
            _register(adapter, payload, tuple(device.upper() for device in DEVICES))
        assert not registry.resource_snapshot()["exclusive_resources"]


def test_fixed_devices_reuse_and_replacement_preserve_environment_and_reservations(monkeypatch):
    class Model:
        def __init__(self):
            self.device = os.environ["CUDA_VISIBLE_DEVICES"]
            self.config = os.environ.get("AWS_VANE_GPU_TEST")

        def __call__(self, table):
            return pa.table({"device": [self.device], "pid": [os.getpid()], "config": [self.config]})

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-mask")
    config = {"AWS_VANE_GPU_TEST": "captured"}
    payload = _payload(Model, replicas=2)
    registry = ModelPoolRegistry(resident_limit=ResourceVector(cpu=1, gpu=2, heap_bytes=2048))
    adapter = LocalGpuModelAdapter(registry, devices=DEVICES)
    identity = _register(adapter, payload, DEVICES, session_config=config)
    payload["function_pickle"] = b"mutated"
    config["AWS_VANE_GPU_TEST"] = "mutated"
    assert not registry.resource_snapshot()["exclusive_resources"]
    try:
        with ThreadPoolExecutor(4) as threads:
            list(threads.map(lambda _: registry.prewarm(identity), range(4)))
        with registry.acquire(identity) as first, registry.acquire(identity) as second:
            assert first.pool is second.pool
            pool = first.pool
            before = pool.device_snapshot()
            rows = [_call(pool).to_pylist()[0] for _ in range(2)]
            assert {row["device"] for row in rows} == set(DEVICES)
            assert {row["pid"] for row in rows} == set(pool.worker_pids())
            assert {row["config"] for row in rows} == {"captured"}
            assert [row["generation"] for row in before] == [0, 0]
            assert len(registry.resource_snapshot()["exclusive_resources"]) == 2
            # Simulate a worker loss. The next borrow replaces it on the same device.
            pool._workers[0]._proc.kill()
            pool._workers[0]._proc.wait(timeout=5)
            pool._workers[0]._mark_broken("injected loss", actor_lost=True)
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "later-mask")
            for _ in range(3):
                _call(pool)
            after = pool.device_snapshot()
            assert after[0]["device"] == before[0]["device"]
            assert after[0]["generation"] == 1 and after[0]["pid"] != before[0]["pid"]
            assert after[1] == before[1]
            assert registry.resource_snapshot()["reserved_resources"]["gpu"] == 2
        assert registry.resource_snapshot()["reserved_resources"]["gpu"] == 2
    finally:
        registry.close(timeout=15, kill=True)
    assert not registry.resource_snapshot()["exclusive_resources"]
    assert registry.resource_snapshot()["reserved_resources"]["gpu"] == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "later-mask"


@pytest.mark.parametrize("action", ["cancel", "drain", "fail"])
def test_initialization_retains_device_until_registry_cleanup(tmp_path, action):
    marker, release = str(tmp_path / "started"), str(tmp_path / "release")

    class Model:
        def __init__(self):
            from pathlib import Path

            Path(marker).write_text(json.dumps({"device": os.environ["CUDA_VISIBLE_DEVICES"], "pid": os.getpid()}))
            deadline = time.monotonic() + 20
            while not Path(release).exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("test constructor was not released")
                time.sleep(0.01)
            if action == "fail":
                raise ValueError("injected constructor failure")

        def __call__(self, table):
            return table

    registry = ModelPoolRegistry()
    adapter = LocalGpuModelAdapter(registry, devices=DEVICES)
    identity = _register(adapter, _payload(Model))
    other = _register(adapter, _payload(lambda table: table), name="other")
    cancellation = ExecutionCancellationScope("initializer", 1)
    try:
        with ThreadPoolExecutor(1) as threads:
            future = threads.submit(registry.acquire, identity, cancellation=cancellation)
            try:
                _wait(lambda: (tmp_path / "started").exists())
                with pytest.raises(ModelPoolResourceBusy):
                    registry.prewarm(other)
                assert registry.resource_snapshot()["initializing_resources"]["gpu"] == 1
                if action == "cancel":
                    cancellation.cancel()
                elif action == "drain":
                    registry.drain()
                    with pytest.raises(RuntimeError, match="draining"):
                        registry.prewarm(other)
                    with pytest.raises(TimeoutError):
                        registry.close()
            finally:
                (tmp_path / "release").touch()
            with pytest.raises((RuntimeError, ExecutionCancelledError)):
                future.result(timeout=15)
        if action == "fail":
            assert not registry.resource_snapshot()["exclusive_resources"]
            # The refused acquisition is retryable after clean constructor failure.
            registry.prewarm(other)
        elif action == "cancel":
            with registry.acquire(identity) as borrow:
                assert _call(borrow.pool).to_pydict() == {"x": [7]}
        assert registry.resource_snapshot()["reserved_resources"]["gpu"] == 1
    finally:
        (tmp_path / "release").touch()
        registry.close(timeout=15, kill=True)


def test_failed_worker_cleanup_prevents_device_reuse_and_replacement(monkeypatch):
    registry = ModelPoolRegistry()
    adapter = LocalGpuModelAdapter(registry, devices=DEVICES)
    identity = _register(adapter, _payload(lambda table: table))
    other = _register(adapter, _payload(lambda table: table), name="other")
    try:
        with registry.acquire(identity) as borrow:
            pool = borrow.pool
            worker = pool._workers[0]
            original_close = worker.close
            initial_pid = worker._proc.pid
            with monkeypatch.context() as fault:

                def fail_close(*args, **kwargs):
                    raise OSError("injected device cleanup failure")

                fault.setattr(worker, "close", fail_close)
                with pytest.raises(OSError, match="injected device cleanup"):
                    worker._mark_broken("injected loss", actor_lost=True)
                with pytest.raises(RuntimeError, match="cleanup failed"):
                    _call(pool)
                assert pool.worker_pids() == [initial_pid]
                with pytest.raises(ModelPoolResourceBusy):
                    registry.prewarm(other)
            # Leave the live process owned by its pool for close retry.
            monkeypatch.setattr(worker, "close", fail_close)
        with pytest.raises(OwnedActorPoolsError):
            registry.close(kill=True)
        assert registry.resource_snapshot()["reserved_resources"]["gpu"] == 1
        assert pool.device_snapshot()[0]["cleanup_finished"] is False
        monkeypatch.setattr(worker, "close", original_close)
    finally:
        monkeypatch.undo()
        registry.close(timeout=15, kill=True)
    assert worker._proc is None or worker._proc.poll() is not None
    assert not registry.resource_snapshot()["exclusive_resources"]


def test_local_gpu_requests_require_registration_and_device_inventory():
    from vane.execution.udf import build_executor
    from vane.execution.udf_local_model import LocalModelRuntime

    payload = _payload(lambda table: table)
    with pytest.raises(ValueError, match="Ray UDF backend"):
        build_executor(payload)
    with LocalModelRuntime(session_id="session", session_config={}) as runtime:
        with pytest.raises(ValueError, match="gpu_devices inventory and model assignment"):
            runtime.register("model", version="v1", payload=payload)


def test_partial_constructor_cleanup_keeps_all_devices_charged(monkeypatch):
    class Model:
        def __init__(self):
            if os.environ["VANE_SUBPROCESS_WORKER_INDEX"] == "1":
                raise ValueError("second GPU constructor failed")

        def __call__(self, table):
            return table

    registry = ModelPoolRegistry()
    adapter = LocalGpuModelAdapter(registry, devices=DEVICES)
    identity = _register(adapter, _payload(Model, replicas=2), DEVICES)
    other = _register(adapter, _payload(lambda table: table), name="other")
    close = udf_subprocess._SingleSubprocessExecutor.close

    def fail_close(worker, *args, **kwargs):
        if worker._worker_env.get("VANE_SUBPROCESS_WORKER_INDEX") == "1":
            raise OSError("provisional GPU cleanup failed")
        return close(worker, *args, **kwargs)

    try:
        with monkeypatch.context() as fault:
            fault.setattr(udf_subprocess._SingleSubprocessExecutor, "close", fail_close)
            with pytest.raises(RuntimeError, match="cleanup failed"):
                registry.prewarm(identity)
            snapshot = registry.resource_snapshot()
            assert snapshot["retained_failure_resources"]["gpu"] == 2
            assert len(snapshot["exclusive_resources"]) == 2
            with pytest.raises(ModelPoolResourceBusy):
                registry.prewarm(other)
            with pytest.raises(OwnedActorPoolsError):
                registry.close(kill=True)
            assert len(registry.resource_snapshot()["exclusive_resources"]) == 2
    finally:
        registry.close(timeout=15, kill=True)
    assert not registry.resource_snapshot()["exclusive_resources"]


def test_cancelled_execution_replaces_worker_on_the_same_reserved_device(tmp_path):
    marker, release = str(tmp_path / "entered"), str(tmp_path / "release")

    class Model:
        def __call__(self, table):
            from pathlib import Path

            Path(marker).touch()
            deadline = time.monotonic() + 20
            while not Path(release).exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("GPU test call was not released")
                time.sleep(0.01)
            return pa.table({"device": [os.environ["CUDA_VISIBLE_DEVICES"]]})

    registry = ModelPoolRegistry()
    identity = _register(LocalGpuModelAdapter(registry, devices=DEVICES), _payload(Model))
    scope = ExecutionCancellationScope("cancel-gpu-test", 1)
    try:
        with registry.acquire(identity) as borrow:
            pool = borrow.pool
            pid = pool.worker_pids()[0]

            def call(worker):
                worker.submit(pa.table({"x": [7]}))
                return worker.take_ready_result()

            authority = pool.create_admission_authority()
            authority.request(0)
            admission = authority.take(0)
            future = pool.submit(call, scope=scope, admission=admission)
            try:
                _wait(lambda: (tmp_path / "entered").exists())
                scope.cancel()
                pool.abort_scopes({scope})
                with pytest.raises(RuntimeError):
                    future.result(timeout=15)
            finally:
                (tmp_path / "release").touch()
                scope.cancel()
                admission.release()
                authority.close()
            assert _call(pool).to_pydict() == {"device": [DEVICES[0]]}
            assert pool.worker_pids()[0] != pid
            assert pool.device_snapshot()[0]["generation"] == 1
            assert registry.resource_snapshot()["reserved_resources"]["gpu"] == 1
    finally:
        (tmp_path / "release").touch()
        registry.close(timeout=15, kill=True)
    assert not registry.resource_snapshot()["exclusive_resources"]
