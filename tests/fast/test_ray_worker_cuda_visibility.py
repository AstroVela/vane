# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest

ray = pytest.importorskip("ray")

from ray_test_profile import ray_test_object_store_bytes

import duckdb.runners.ray.worker as worker_mod
from duckdb.runners.ray.worker_pool import _persistent_worker_runtime_env


class _RuntimeContext:
    def get_node_id(self):
        return "node-a"


@pytest.mark.parametrize(
    "visible_devices",
    [
        "2,5",
        "GPU-deadbeef,GPU-cafebabe",
        "MIG-GPU-deadbeef/1/0,MIG-GPU-cafebabe/2/0",
    ],
)
def test_ray_worker_init_preserves_node_cuda_visibility(monkeypatch, visible_devices):
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    monkeypatch.setattr(worker_mod.ray, "get_runtime_context", _RuntimeContext)
    monkeypatch.setattr(worker_mod, "_warm_up_python_native_dependencies", lambda: None)
    monkeypatch.setattr(worker_mod, "_ensure_python_datasource_runtime", lambda: None)
    monkeypatch.setattr(actor_cls, "_get_shared_conn", lambda _self: None)

    actor = actor_cls(
        num_cpus=2,
        num_gpus=2,
        duckdb_memory_bytes=128 * 1024**2,
        task_heap_capacity_bytes=128 * 1024**2,
        env_overrides={},
    )

    assert worker_mod.os.environ["CUDA_VISIBLE_DEVICES"] == visible_devices
    actor_cls.__del__(actor)


@pytest.mark.real_ray
@pytest.mark.ray_cluster_owner
def test_zero_gpu_ray_worker_preserves_non_contiguous_node_cuda_visibility(monkeypatch):
    visible_devices = "2,5"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    # Exercise the legacy Ray behavior even when this test runs on Ray 2.56+.
    # The persistent worker's runtime_env must opt back into preservation.
    monkeypatch.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "1")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Tip: In future versions of Ray")
        ray.init(
            address="local",
            include_dashboard=False,
            log_to_driver=True,
            num_cpus=1,
            num_gpus=2,
            object_store_memory=ray_test_object_store_bytes(),
        )
    actor = None
    try:
        actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class

        class CudaVisibilityProbe(actor_cls):
            def cuda_visibility(self):
                return (
                    worker_mod.os.environ.get("CUDA_VISIBLE_DEVICES"),
                    ray.get_runtime_context().get_accelerator_ids(),
                )

        probe_cls = ray.remote(concurrency_groups={"execute": 128, "control": 512})(CudaVisibilityProbe)
        actor = probe_cls.options(runtime_env=_persistent_worker_runtime_env()).remote(
            num_cpus=1,
            num_gpus=2,
            duckdb_memory_bytes=128 * 1024**2,
            task_heap_capacity_bytes=128 * 1024**2,
            env_overrides={},
        )

        observed_devices, assigned_accelerators = ray.get(actor.cuda_visibility.remote(), timeout=60)

        assert observed_devices == visible_devices
        assert not assigned_accelerators.get("GPU")
    finally:
        if actor is not None:
            ray.kill(actor, no_restart=True)
        ray.shutdown()
