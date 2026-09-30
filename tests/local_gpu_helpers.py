# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Small public GPU runtime workloads shared by CPU contracts and CUDA checks."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pytest

import vane
from vane.execution import ref_bundle
from vane.execution.request_admission import RequestAdmissionLimits
from vane.execution.resources import ResourceVector
from vane.execution.udf_data_admission import DataAdmissionLimits, DataAdmissionWaitLimits
from vane.execution.udf_runtime_admission import TaskAdmissionLimits

DEVICES = ("GPU-aaaaaaaa-0000-0000-0000-000000000001", "GPU-bbbbbbbb-0000-0000-0000-000000000002")


@pytest.fixture(scope="module")
def cuda_devices():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("real CUDA acceptance requires an available CUDA device")
    device = os.environ.get("VANE_TEST_CUDA_DEVICE")
    if device is None:
        device = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    return (device,)


@pytest.fixture
def query_gpu_environment(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    gc.collect()
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 420_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    yield manager
    gc.collect()
    assert manager.snapshot()["usage_bytes"] == 0


def wait_for(check, *futures, timeout=20):
    deadline = time.monotonic() + timeout
    while True:
        value = check()
        if value:
            return value
        for future in futures:
            if future.done():
                future.result()
        assert time.monotonic() < deadline, "GPU query did not reach its checkpoint"
        time.sleep(0.01)


def model_definition(directory, *, batch=False, actors=1, cuda=False, gpus=1, fail_init=False):
    root = str(directory)

    class Model:
        def __init__(self):
            self.device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            self.pid = os.getpid()
            self.device_count = 0
            if cuda:
                import torch

                assert torch.cuda.is_available()
                self.device_count = torch.cuda.device_count()
                assert self.device_count == 1
                self.weights = torch.arange(8, dtype=torch.int64, device="cuda")
                torch.cuda.synchronize()
            with Path(root, "initialized").open("a") as stream:
                stream.write(f"{self.pid}\n")
            if fail_init:
                raise ValueError("GPU fixture initialization failure")

        def apply(self, value):
            if value in (-1, -2):
                Path(root, f"entered{value}").touch()
                deadline = time.monotonic() + 60
                while not Path(root, f"release{value}").exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("GPU fixture was not released")
                    time.sleep(0.01)
            if value == -3:
                os._exit(23)
            if value == -4:
                raise ValueError("GPU fixture execution failure")
            if value == -5:
                return "x" * 100_000
            # Copying the result to host waits for its CUDA work. Background
            # device work outliving the UDF call is outside this API contract.
            result = int((self.weights + value).sum().cpu().item()) if cuda else 28 + 8 * value
            return json.dumps(
                {"pid": self.pid, "device": self.device, "value": result, "device_count": self.device_count}
            )

    if batch:

        class BatchModel(Model):
            def __call__(self, values):
                return pa.array([self.apply(value) for value in values.to_pylist()])

        definition = vane.cls.batch(
            actor_number=actors, gpus=gpus, return_dtype="VARCHAR", name="gpu_encode", batch_size=1
        )(BatchModel)
    else:

        class RowModel(Model):
            def __call__(self, value):
                return self.apply(value)

        definition = vane.cls(actor_number=actors, gpus=gpus, return_dtype="VARCHAR", name="gpu_encode")(RowModel)
    return definition()


def configure(connection, devices=DEVICES[:1], *, limited=True, **kwargs):
    options = dict(
        request_limit=RequestAdmissionLimits(2, 4),
        resident_limit=ResourceVector(cpu=4, gpu=len(devices), heap_bytes=8192),
        gpu_devices=devices,
        task_limit=TaskAdmissionLimits(1, 8) if limited else None,
        data_limit=DataAdmissionLimits(420_000, 140_000, 70_000, wait=DataAdmissionWaitLimits(8, 20)),
        track_graph=True,
        execution_timeout=45,
    )
    options.update(kwargs)
    return connection.configure_local_runtime(**options)


def register(runtime, model, devices=DEVICES[:1], *, name="encoder"):
    return runtime.register_model(
        name, model, version="v1", parameters=["BIGINT"], memory_bytes=1024, gpu_devices=devices
    )


def gpu_pool_snapshot(runtime, *, index=0):
    return runtime.resource_snapshot()["gpu"]["models"][index]["pools"][0]


def execution_demand(runtime):
    return sum(
        device["execution_resources"]["gpu"]
        for model in runtime.resource_snapshot()["gpu"]["models"]
        for pool in model["pools"]
        for device in pool["devices"]
    )


def assert_idle(runtime, *, resident=1):
    state = runtime.resource_snapshot()
    assert state["reserved_resources"]["gpu"] == resident
    assert state["active_borrows"] == 0
    assert state["request_admission"]["active_requests"] == 0
    assert state["request_admission"]["queued_requests"] == 0
    if "data" in state:
        assert state["data"]["usage_bytes"] == 0
    assert execution_demand(runtime) == 0


def run(cursor, value):
    return json.loads(cursor.execute("SELECT gpu_encode(?)", [value]).fetchone()[0])
