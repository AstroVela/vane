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


def test_ray_worker_observability_log_treats_off_as_disabled(monkeypatch, capsys):
    monkeypatch.setenv("VANE_FTE_ADMISSION_DEBUG", "off")
    monkeypatch.delenv("VANE_RAY_WORKER_MEMORY_DEBUG", raising=False)
    monkeypatch.delenv("VANE_FTE_RESULT_DEBUG", raising=False)
    monkeypatch.delenv("DUCKDB_DISTRIBUTED_DEBUG", raising=False)

    worker_mod._ray_worker_observability_log("worker_registered", worker_id="worker-a")

    assert capsys.readouterr().err == ""


def test_ray_worker_init_logs_worker_topology(monkeypatch, capfd):
    class _ActorRuntimeContext(_RuntimeContext):
        def get_actor_id(self):
            return "actor-a"

    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    monkeypatch.setenv("VANE_FTE_ADMISSION_DEBUG", "1")
    monkeypatch.setenv("VANE_WORKER_ID", "manager-a:node-a:0")
    monkeypatch.setenv("VANE_WORKER_MANAGER_INSTANCE_ID", "manager-a")
    monkeypatch.setenv("VANE_WORKER_NODE_ID", "node-a")
    monkeypatch.setenv("VANE_WORKER_HOST", "10.0.0.1")
    monkeypatch.setattr(worker_mod.ray, "get_runtime_context", _ActorRuntimeContext)
    monkeypatch.setattr(worker_mod, "_warm_up_python_native_dependencies", lambda: None)
    monkeypatch.setattr(worker_mod, "_ensure_python_datasource_runtime", lambda: None)
    monkeypatch.setattr(actor_cls, "_get_shared_conn", lambda _self: None)

    actor = actor_cls(
        num_cpus=2,
        num_gpus=0,
        duckdb_memory_bytes=128 * 1024**2,
        task_heap_capacity_bytes=128 * 1024**2,
        ray_node_ip_address="10.0.0.1",
    )
    actor_cls.__del__(actor)
    captured = capfd.readouterr().err

    assert "[vane-ray-worker" in captured
    assert "event=worker_registered" in captured
    assert "worker_id=manager-a:node-a:0" in captured
    assert "manager_instance_id=manager-a" in captured
    assert "node_id=node-a" in captured
    assert "host=10.0.0.1" in captured
    assert "actor_id=actor-a" in captured


@pytest.mark.parametrize(
    ("node_local_host", "expected_host"),
    [
        (None, "10.0.0.1"),
        ("flight.node.internal", "flight.node.internal"),
    ],
)
def test_ray_worker_init_uses_ray_node_address_for_flight_host(
    monkeypatch,
    node_local_host,
    expected_host,
):
    actor_cls = worker_mod.RayWorkerActor.__ray_metadata__.modified_class
    if node_local_host is None:
        monkeypatch.delenv("VANE_FLIGHT_ADVERTISE_HOST", raising=False)
    else:
        monkeypatch.setenv("VANE_FLIGHT_ADVERTISE_HOST", node_local_host)
    monkeypatch.setattr(worker_mod.ray, "get_runtime_context", _RuntimeContext)
    monkeypatch.setattr(worker_mod, "_warm_up_python_native_dependencies", lambda: None)
    monkeypatch.setattr(worker_mod, "_ensure_python_datasource_runtime", lambda: None)
    monkeypatch.setattr(actor_cls, "_get_shared_conn", lambda _self: None)

    actor = actor_cls(
        num_cpus=2,
        num_gpus=0,
        duckdb_memory_bytes=128 * 1024**2,
        task_heap_capacity_bytes=128 * 1024**2,
        ray_node_ip_address="10.0.0.1",
    )

    assert worker_mod.os.environ["VANE_FLIGHT_ADVERTISE_HOST"] == expected_host
    actor_cls.__del__(actor)


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
    )

    assert worker_mod.os.environ["CUDA_VISIBLE_DEVICES"] == visible_devices
    actor_cls.__del__(actor)


@pytest.mark.real_ray
@pytest.mark.ray_cluster_owner
def test_zero_gpu_ray_worker_preserves_non_contiguous_node_cuda_visibility(monkeypatch):
    visible_devices = "2,5"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    monkeypatch.setenv("AWS_ISSUE75_INHERITED_SECRET", "inherited-aws")
    monkeypatch.setenv("DUCKDB_ISSUE75_INHERITED_SECRET", "inherited-duckdb")
    monkeypatch.setenv("VANE_ISSUE75_INHERITED_SECRET", "inherited-vane")
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
                    worker_mod.os.environ.get("AWS_ISSUE75_INHERITED_SECRET"),
                    worker_mod.os.environ.get("DUCKDB_ISSUE75_INHERITED_SECRET"),
                    worker_mod.os.environ.get("VANE_ISSUE75_INHERITED_SECRET"),
                )

        probe_cls = ray.remote(concurrency_groups={"execute": 128, "control": 512})(CudaVisibilityProbe)
        actor = probe_cls.options(runtime_env=_persistent_worker_runtime_env({})).remote(
            num_cpus=1,
            num_gpus=2,
            duckdb_memory_bytes=128 * 1024**2,
            task_heap_capacity_bytes=128 * 1024**2,
        )

        observed_devices, assigned_accelerators, inherited_aws, inherited_duckdb, inherited_vane = ray.get(
            actor.cuda_visibility.remote(),
            timeout=60,
        )

        assert observed_devices == visible_devices
        assert not assigned_accelerators.get("GPU")
        assert inherited_aws is None
        assert inherited_duckdb is None
        assert inherited_vane is None
    finally:
        if actor is not None:
            ray.kill(actor, no_restart=True)
        ray.shutdown()
