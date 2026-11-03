# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real SDK checks for CUDA admission and CPU-platform rejection."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pytest
from ray_test_profile import ray_test_object_store_options

pytestmark = [pytest.mark.real_ray, pytest.mark.ray_cluster_owner, pytest.mark.gpu]


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("mode", ["cuda", "cpu"])
def test_native_worker_device_and_resources(family, mode):
    selected = os.environ.get("VANE_TEST_NATIVE_DEVICE_PROVIDER")
    if selected != family:
        pytest.skip("Set VANE_TEST_NATIVE_DEVICE_PROVIDER=vllm or sglang in the corresponding SDK environment")
    model = os.environ.get("VANE_TEST_NATIVE_DEVICE_MODEL")
    if not model or not Path(model).is_dir():
        pytest.skip("Set VANE_TEST_NATIVE_DEVICE_MODEL to a reviewed local embedding checkpoint")
    try:
        importlib.metadata.version(family)
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(f"Install the supported {family} SDK")
    ray = pytest.importorskip("ray")
    if mode == "cuda":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA hardware is required")
        pytest.importorskip("pynvml", reason="Install nvidia-ml-py for GPU process memory checks")
    from vane.ai.provider import load_provider

    dimensions = int(os.environ.get("VANE_TEST_NATIVE_DEVICE_DIMENSIONS", "2048"))
    engine_args = json.loads(os.environ.get("VANE_TEST_NATIVE_DEVICE_ENGINE_ARGS", "{}"))
    descriptor = load_provider(family).get_text_embedder(model, dimensions, options={"engine_args": engine_args})
    assert descriptor.get_udf_options().num_gpus == 1
    ray.init(address="local", num_cpus=2, num_gpus=1, include_dashboard=False, **ray_test_object_store_options())
    worker = None
    try:

        @ray.remote
        class Worker:
            async def run(self, descriptor, family, mode):
                import os

                import ray

                resources = ray.get_runtime_context().get_assigned_resources()
                if mode == "cpu":
                    try:
                        descriptor.instantiate()
                    except ValueError as exc:
                        assert "requires CUDA" in str(exc) and "'cpu'" in str(exc)
                        return {"mode": "cpu", "rejected": True, "resources": resources}
                    raise AssertionError("CPU SDK platform unexpectedly loaded a native engine")
                import psutil
                import pynvml
                import torch

                runtime = descriptor.instantiate()
                try:
                    values = await runtime.embed_text(["A small red bird.", "A quiet blue lake."])
                    owner = psutil.Process()
                    children = {child.pid for child in owner.children(recursive=True)}
                    gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
                    if not gpu_uuid.startswith("GPU-"):
                        gpu_uuid = "GPU-" + gpu_uuid
                    pynvml.nvmlInit()
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
                        gpu_children = [
                            {"pid": process.pid, "bytes": process.usedGpuMemory}
                            for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                            if process.pid in children and 0 < process.usedGpuMemory < 2**63
                        ]
                    finally:
                        pynvml.nvmlShutdown()
                    return {
                        "mode": "cuda",
                        "values": values,
                        "resources": resources,
                        "gpu_ids": ray.get_runtime_context().get_accelerator_ids()["GPU"],
                        "gpu_uuid": gpu_uuid,
                        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "cuda_available": torch.cuda.is_available(),
                        "cuda_device_count": torch.cuda.device_count(),
                        "allocated_bytes": torch.cuda.memory_allocated(),
                        "gpu_children": gpu_children,
                    }
                finally:
                    await runtime.aclose()

        env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        if family == "vllm":
            env["VLLM_TARGET_DEVICE"] = "cpu" if mode == "cpu" else "cuda"
        else:
            env["SGLANG_USE_CPU_ENGINE"] = "1" if mode == "cpu" else "0"
        worker = Worker.options(
            num_gpus=descriptor.get_udf_options().num_gpus if mode == "cuda" else 0,
            num_cpus=1,
            runtime_env={"env_vars": env},
        ).remote()
        result = ray.get(worker.run.remote(descriptor, family, mode), timeout=240)
        if mode == "cpu":
            assert result["rejected"] and result["resources"].get("GPU", 0) == 0
        else:
            values = np.asarray(result.pop("values"))
            assert values.shape == (2, dimensions) and np.isfinite(values).all()
            assert (np.linalg.norm(values, axis=1) > 0).all()
            assert not np.array_equal(values[0], values[1])
            assert result["resources"]["GPU"] == 1 and len(result["gpu_ids"]) == 1
            assert result["cuda_available"] and result["cuda_device_count"] == 1
            assert result["visible_devices"]
            assert result["allocated_bytes"] > 0 or sum(child["bytes"] for child in result["gpu_children"]) > 0
            result["shape"] = list(values.shape)
        print(json.dumps({"provider": family, **result}, sort_keys=True))
    finally:
        if worker is not None:
            ray.kill(worker)
        ray.shutdown()
