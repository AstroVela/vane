# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import SimpleNamespace

import pyarrow as pa
import pytest

_GIB = 1024**3
_DEFAULT_OUTPUT_TARGET = 128 * 1024**2
_NODE_ID = "a" * 56


def _distributed_payload(**overrides):
    payload = {
        "execution_backend": "ray_task",
        "query_id": "q1",
        "stage_id": "stage:q1:node:1:udf",
        "node_id": _NODE_ID,
        "produce_ray_block_stream": True,
        "call_mode": "map_batches",
        "cpus": 1.0,
        "gpus": 0.0,
        "memory_bytes": 2 * _GIB,
        "udf_output_target_max_bytes": _DEFAULT_OUTPUT_TARGET,
        "udf_task_input_max_bytes": _DEFAULT_OUTPUT_TARGET,
        "output_window_bytes": 2 * _DEFAULT_OUTPUT_TARGET,
    }
    payload.update(overrides)
    return payload


@pytest.fixture(autouse=True)
def _fixed_ray_runtime_node(monkeypatch):
    import ray

    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: _NODE_ID),
    )


def test_ray_udf_cpu_and_memory_defaults_are_nonzero(monkeypatch):
    import vane.execution.udf_ray as fur
    from vane.execution.udf_task_admission import ray_udf_task_memory_bytes

    monkeypatch.delenv("VANE_UDF_TASK_HEAP_BYTES", raising=False)
    monkeypatch.delenv("VANE_UDF_ACTOR_HEAP_BYTES", raising=False)
    assert fur._payload_num_cpus({}) == 1.0
    assert ray_udf_task_memory_bytes({"execution_backend": "ray_task"}) == 2 * _GIB
    assert ray_udf_task_memory_bytes({"execution_backend": "ray_actor"}) == 4 * _GIB
    with pytest.raises(ValueError, match="memory_bytes must be positive"):
        ray_udf_task_memory_bytes({"execution_backend": "ray_task", "memory_bytes": 0})


def test_ray_task_options_use_exact_logical_resources_and_fixed_pair_window():
    import vane.execution.udf_ray as fur

    options = fur._task_remote_options(1.5, 0.25, 3 * _GIB, 2, {"name": "task"})

    assert options == {
        "name": "task",
        "num_cpus": 1.5,
        "num_gpus": 0.25,
        "memory": 3 * _GIB,
        "max_retries": 2,
        "_generator_backpressure_num_objects": 4,
    }


def test_task_payload_accepts_registered_downstream_retention_window_multiple():
    from vane.execution.udf_ray_stream_protocol import task_payload_with_lease

    payload = _distributed_payload(
        udf_output_target_max_bytes=1024,
        output_window_bytes=2 * 1024,
    )
    lease = {
        "lease_id": "lease-retention",
        "query_id": payload["query_id"],
        "stage_id": payload["stage_id"],
        "attempt_id": "attempt-retention",
        "node_id": _NODE_ID,
        "execution_slot_id": f"ray_task:{payload['stage_id']}:lease-retention",
        "output_window_bytes": 64 * 1024,
    }

    merged = task_payload_with_lease(payload, lease)

    assert merged["output_window_bytes"] == 64 * 1024


@pytest.mark.parametrize("window", [1024, 2500])
def test_task_payload_rejects_invalid_registered_retention_window(window):
    from vane.execution.udf_ray_stream_protocol import task_payload_with_lease

    payload = _distributed_payload(udf_output_target_max_bytes=1024)
    lease = {
        "lease_id": "lease-invalid-retention",
        "query_id": payload["query_id"],
        "stage_id": payload["stage_id"],
        "attempt_id": "attempt-invalid-retention",
        "node_id": _NODE_ID,
        "execution_slot_id": f"ray_task:{payload['stage_id']}:lease-invalid-retention",
        "output_window_bytes": window,
    }

    with pytest.raises(ValueError, match="valid multiple"):
        task_payload_with_lease(payload, lease)


def test_ray_task_remote_keeps_options_available_for_lease_node_affinity():
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    import vane.execution.udf_ray as fur

    for builder in (
        fur._build_bundle_stream_remote,
        fur._build_ref_bundle_stream_remote,
    ):
        remote_fn = builder(1.0, 0.0, 2 * _GIB, 2, {})
        scheduled = remote_fn.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=_NODE_ID,
                soft=False,
            )
        )
        assert callable(scheduled.remote)


def test_ray_task_executor_requires_runner_owned_ray_runtime(monkeypatch):
    import ray

    import vane.execution.udf_ray as fur

    monkeypatch.setattr(ray, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="initialized RayRunner runtime"):
        fur._build_ray_task_executor(_distributed_payload(), {})


def test_ray_udf_rejects_unknown_options_without_compatibility_branches():
    import vane.execution.udf as udf

    with pytest.raises(ValueError, match="unknown UDF executor options: ray_address"):
        udf.normalize_options({"ray_address": "auto"})
    with pytest.raises(TypeError, match="must be a dict"):
        udf.normalize_options('{"max_task_retries": 0}')
    with pytest.raises(ValueError, match="max_task_retries must be a non-negative integer"):
        udf.normalize_options({"max_task_retries": "0"})
    with pytest.raises(ValueError, match="scheduling_strategy"):
        udf.normalize_options({"ray_options": {"scheduling_strategy": "DEFAULT"}})


def test_actor_call_uses_fixed_four_raw_objects_for_two_logical_blocks():
    from vane.execution.udf_ray_remote_submit import _with_generator_backpressure

    class _Method:
        def __init__(self):
            self.calls = []

        def options(self, **kwargs):
            self.calls.append(kwargs)
            return self

    method = _Method()
    assert _with_generator_backpressure(method) is method
    assert method.calls == [{"_generator_backpressure_num_objects": 4}]


def test_distributed_payload_requires_registered_query_stage_and_new_protocol():
    import vane.execution.udf_ray as fur

    assert fur._ray_payload_requires_block_stream(_distributed_payload()) is True
    for missing in ("query_id", "stage_id", "produce_ray_block_stream"):
        payload = _distributed_payload()
        payload.pop(missing)
        with pytest.raises(RuntimeError):
            fur._ray_payload_requires_block_stream(payload)


def test_task_executor_consumes_pregranted_admission_with_exact_resources(monkeypatch):
    import vane.execution.udf_ray as fur
    from vane.execution.udf_task_admission import ray_udf_task_resource_spec

    captured = {}

    class _Pregranted:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    payload = _distributed_payload(cpus=2.0, gpus=1.0, memory_bytes=3 * _GIB)
    assert ray_udf_task_resource_spec(payload) == {
        "cpu": 2.0,
        "gpu": 1.0,
        "heap_bytes": 3 * _GIB,
        "object_store_bytes": _DEFAULT_OUTPUT_TARGET,
    }
    monkeypatch.setattr(fur, "TaskLeaseObjectRefGenerator", _Pregranted)
    executor = fur.RayTaskUDFExecutor(
        payload,
        run_bundle_stream=object(),
        run_ref_bundle_stream=object(),
    )
    admission = SimpleNamespace(
        driver=object(),
        request_id="request-7",
        lease={
            "lease_id": "lease-7",
            "node_id": _NODE_ID,
            "execution_slot_id": "ray_task:stage:q1:node:1:udf:lease-7",
        },
    )
    monkeypatch.setattr(executor, "_take_task_admission", lambda: admission)

    result = executor.submit_with_id(7, pa.table({"value": [1]}))

    assert isinstance(result, _Pregranted)
    assert captured["admission"] is admission


def test_task_submission_starts_immediately_from_pregranted_lease(monkeypatch):
    import vane.execution.udf_ray as fur

    submitted = []
    captured = {}

    class _Pregranted:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.generator = kwargs["submitter"](dict(kwargs["admission"].lease))

    class _Remote:
        def options(self, **options):
            assert options["scheduling_strategy"].node_id == _NODE_ID
            assert options["scheduling_strategy"].soft is False
            return self

        def remote(self, *args, **kwargs):
            submitted.append((args, kwargs))
            return "generator"

    monkeypatch.setattr(fur, "TaskLeaseObjectRefGenerator", _Pregranted)
    executor = fur.RayTaskUDFExecutor(
        _distributed_payload(),
        run_bundle_stream=_Remote(),
        run_ref_bundle_stream=_Remote(),
    )
    lease = {
        "query_id": "q1",
        "stage_id": "stage:q1:node:1:udf",
        "lease_id": "lease-8",
        "attempt_id": "attempt-8",
        "node_id": _NODE_ID,
        "execution_slot_id": "ray_task:stage:q1:node:1:udf:lease-8",
        "output_window_bytes": 2 * _DEFAULT_OUTPUT_TARGET,
    }
    admission = SimpleNamespace(driver=object(), request_id="request-8", lease=lease)
    monkeypatch.setattr(executor, "_take_task_admission", lambda: admission)

    result = executor.submit_with_id(8, pa.table({"value": [1]}))

    assert isinstance(result, _Pregranted)
    assert result.generator == "generator"
    assert captured["admission"] is admission
    assert len(submitted) == 1
    remote_payload = submitted[0][0][0]
    assert remote_payload["task_lease_id"] == "lease-8"
    assert remote_payload["attempt_id"] == "attempt-8"


def test_ref_bundle_submission_uses_direct_object_refs_without_registry():
    from vane.execution.udf_ray_remote_ref_bundle import _resolve_ref_bundle_task_refs

    refs = [object(), object()]

    resolved = _resolve_ref_bundle_task_refs(refs)

    assert resolved == refs


def test_task_stream_producer_yields_direct_block_then_bounded_metadata():
    import vane.execution.udf_ray as fur
    from vane import pickle as duckdb_pickle

    def identity(table):
        return table

    payload = _distributed_payload(
        function_pickle=duckdb_pickle.dumps(identity),
        null_handling=1,
        task_lease_id="lease-1",
        attempt_id="attempt-1",
    )
    block = pa.table({"value": [1, 2]})
    outputs = list(
        fur._iter_ref_bundle_task_outputs(
            payload,
            [block],
            [None],
            [{"num_rows": 2, "size_bytes": max(1, block.nbytes)}],
            ["value"],
        )
    )

    assert len(outputs) == 2
    assert isinstance(outputs[0], pa.Table)
    assert outputs[0].column("value").to_pylist() == [1, 2]
    assert outputs[1] == {
        "protocol_version": 1,
        "query_id": "q1",
        "producer_stage_id": "stage:q1:node:1:udf",
        "task_lease_id": "lease-1",
        "attempt_id": "attempt-1",
        "block_id": "block:lease-1:0",
        "size_bytes": max(1, outputs[0].nbytes),
        "num_rows": 2,
        "names": ["value"],
    }


def test_materialized_scalar_task_fuses_passthrough_columns_into_block_stream():
    import vane.execution.udf_ray as fur
    from vane import pickle as duckdb_pickle

    def plus_one(value):
        return value + 1

    payload = _distributed_payload(
        function_pickle=duckdb_pickle.dumps(plus_one),
        call_mode="map",
        scalar_arg_count=1,
        scalar_udf_type="native",
        task_lease_id="lease-scalar",
        attempt_id="attempt-scalar",
    )
    layout = pa.table({"arg_a": [1, 2], "a": [1, 2]})

    outputs = list(fur._iter_materialized_task_outputs(payload, [layout]))

    assert len(outputs) == 2
    assert outputs[0].to_pydict() == {"a": [1, 2], "value": [2, 3]}
    assert outputs[1]["names"] == ["a", "value"]
    assert outputs[1]["task_lease_id"] == "lease-scalar"


def test_materialized_task_splits_every_block_before_generator_publication():
    import vane.execution.udf_ray as fur
    from vane import pickle as duckdb_pickle

    def plus_one(value):
        return value + 1

    payload = _distributed_payload(
        function_pickle=duckdb_pickle.dumps(plus_one),
        call_mode="map",
        scalar_arg_count=1,
        scalar_udf_type="native",
        task_lease_id="lease-split",
        attempt_id="attempt-split",
        udf_output_target_max_bytes=20,
        output_window_bytes=40,
    )
    layout = pa.table({"arg_a": [1, 2], "a": [1, 2]})

    outputs = list(fur._iter_materialized_task_outputs(payload, [layout]))

    assert len(outputs) == 4
    blocks = outputs[0::2]
    metadata = outputs[1::2]
    assert [block.num_rows for block in blocks] == [1, 1]
    assert [item["block_id"] for item in metadata] == [
        "block:lease-split:0",
        "block:lease-split:1",
    ]
    assert all(item["size_bytes"] <= 20 for item in metadata)


def test_materialized_task_rejects_unsplittable_row_before_first_yield():
    import vane.execution.udf_ray as fur
    from vane import pickle as duckdb_pickle

    def identity(table):
        return table

    payload = _distributed_payload(
        function_pickle=duckdb_pickle.dumps(identity),
        call_mode="map_batches",
        task_lease_id="lease-oversized",
        attempt_id="attempt-oversized",
        udf_output_target_max_bytes=32,
        output_window_bytes=64,
    )
    oversized = pa.table({"value": ["x" * 1024]})
    stream = fur._iter_materialized_task_outputs(payload, [oversized])

    with pytest.raises(RuntimeError, match="single output row.*32"):
        next(stream)


def test_actor_pool_requests_logical_memory_and_initializes_eagerly(monkeypatch):
    from vane.execution.udf_ray_actor_pool import UDFActorPoolBase

    actor_options = []
    init_calls = []

    class _Init:
        def remote(self, *args):
            init_calls.append(args)
            return "ready"

    class _ActorHandle:
        init_payload = _Init()

    class _ActorFactory:
        @classmethod
        def options(cls, **options):
            actor_options.append(options)
            return SimpleNamespace(remote=lambda: _ActorHandle())

    class _Pool(UDFActorPoolBase):
        @staticmethod
        def _actor_class(*_args):
            return _ActorFactory

        @staticmethod
        def _resolve_actor_num_cpus(_payload):
            return 2.0

        @staticmethod
        def _resolve_actor_memory_bytes(_payload):
            return 5 * _GIB

        @staticmethod
        def _build_actor_runtime_env(_options):
            return {
                "env_vars": {
                    "EXPLICIT_ACTOR_ENV": "yes",
                    "VANE_TORCH_NUM_THREADS": "3",
                },
                "working_dir": "/tmp/actor-runtime",
            }

        @staticmethod
        def _normalize_actor_node_ids(node_ids, *, expected_count):
            return node_ids

    fake_ray = SimpleNamespace(put=lambda value: ("payload-ref", value))
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    _Pool(
        payload={
            "execution_backend": "ray_actor",
            "cpus": 2.0,
            "ray_actor_thread_policy": "managed",
        },
        concurrency=1,
        gpus_per_actor=1.0,
        actor_node_ids=[_NODE_ID],
    )

    assert actor_options[0]["num_cpus"] == 2.0
    assert actor_options[0]["num_gpus"] == 1.0
    assert actor_options[0]["memory"] == 5 * _GIB
    assert actor_options[0]["runtime_env"]["working_dir"] == "/tmp/actor-runtime"
    assert actor_options[0]["runtime_env"]["env_vars"] == {
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2",
        "RAYON_NUM_THREADS": "2",
        "VANE_TORCH_NUM_THREADS": "3",
        "VANE_TORCH_INTEROP_THREADS": "1",
        "VANE_RAY_ACTOR_THREAD_POLICY": "managed",
        "EXPLICIT_ACTOR_ENV": "yes",
    }
    assert actor_options[0]["scheduling_strategy"].node_id == _NODE_ID
    assert actor_options[0]["scheduling_strategy"].soft is False
    assert len(init_calls) == 1
    assert len(init_calls[0]) == 1


def test_actor_pool_thread_env_uses_payload_cpu_allocation(monkeypatch):
    from vane.execution.udf_ray_actor_pool import UDFActorPoolBase

    actor_options = []

    class _Init:
        @staticmethod
        def remote(*_args):
            return "ready"

    class _ActorHandle:
        init_payload = _Init()

    class _ActorFactory:
        @classmethod
        def options(cls, **options):
            actor_options.append(options)
            return SimpleNamespace(remote=lambda: _ActorHandle())

    class _Pool(UDFActorPoolBase):
        @staticmethod
        def _actor_class(*_args):
            return _ActorFactory

        @staticmethod
        def _resolve_actor_num_cpus(payload):
            return payload["cpus"]

        @staticmethod
        def _resolve_actor_memory_bytes(_payload):
            return 5 * _GIB

        @staticmethod
        def _build_actor_runtime_env(_options):
            return {"env_vars": {"OMP_NUM_THREADS": "6"}}

        @staticmethod
        def _normalize_actor_node_ids(node_ids, *, expected_count):
            return node_ids

    fake_ray = SimpleNamespace(put=lambda value: ("payload-ref", value))
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    _Pool(
        payload={
            "execution_backend": "ray_actor",
            "cpus": 4.75,
            "ray_actor_thread_policy": "managed",
        },
        concurrency=1,
        gpus_per_actor=1.0,
        actor_node_ids=[_NODE_ID],
    )

    env_vars = actor_options[0]["runtime_env"]["env_vars"]
    assert env_vars["OMP_NUM_THREADS"] == "6"
    assert env_vars["OPENBLAS_NUM_THREADS"] == "4"
    assert env_vars["MKL_NUM_THREADS"] == "4"
    assert env_vars["NUMEXPR_NUM_THREADS"] == "4"
    assert env_vars["VECLIB_MAXIMUM_THREADS"] == "4"
    assert env_vars["RAYON_NUM_THREADS"] == "4"
    assert env_vars["VANE_TORCH_NUM_THREADS"] == "4"
    assert env_vars["VANE_TORCH_INTEROP_THREADS"] == "1"
    assert env_vars["VANE_RAY_ACTOR_THREAD_POLICY"] == "managed"


def test_actor_pool_default_thread_policy_defers_thread_env_to_ray(monkeypatch):
    from vane.execution.udf_ray_actor_pool import UDFActorPoolBase

    actor_options = []

    class _Init:
        @staticmethod
        def remote(*_args):
            return "ready"

    class _ActorHandle:
        init_payload = _Init()

    class _ActorFactory:
        @classmethod
        def options(cls, **options):
            actor_options.append(options)
            return SimpleNamespace(remote=lambda: _ActorHandle())

    class _Pool(UDFActorPoolBase):
        @staticmethod
        def _actor_class(*_args):
            return _ActorFactory

        @staticmethod
        def _resolve_actor_num_cpus(payload):
            return payload["cpus"]

        @staticmethod
        def _resolve_actor_memory_bytes(_payload):
            return 5 * _GIB

        @staticmethod
        def _build_actor_runtime_env(_options):
            return {"env_vars": {"EXPLICIT_ACTOR_ENV": "yes"}}

        @staticmethod
        def _normalize_actor_node_ids(node_ids, *, expected_count):
            return node_ids

    fake_ray = SimpleNamespace(put=lambda value: ("payload-ref", value))
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    _Pool(
        payload={
            "execution_backend": "ray_actor",
            "cpus": 1.0,
        },
        concurrency=1,
        gpus_per_actor=1.0,
        actor_node_ids=[_NODE_ID],
    )

    env_vars = actor_options[0]["runtime_env"]["env_vars"]
    assert env_vars == {
        "EXPLICIT_ACTOR_ENV": "yes",
        "VANE_RAY_ACTOR_THREAD_POLICY": "ray_native",
    }


def test_runtime_resource_options_cannot_override_graph_resources():
    import vane.execution.udf_ray as fur

    options = fur._task_remote_options(
        1.0,
        0.0,
        2 * _GIB,
        0,
        {"num_cpus": 99, "num_gpus": 99, "memory": 1},
    )
    assert options["num_cpus"] == 1.0
    assert options["num_gpus"] == 0.0
    assert options["memory"] == 2 * _GIB
