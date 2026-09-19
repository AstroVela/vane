# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest

import vane
from vane import pickle as vane_pickle
from vane.execution.resources import ResourceVector
from vane.execution.udf import build_executor
from vane.execution.udf_actor_pool_lifecycle import OwnedActorPoolsError
from vane.execution.udf_local_model import LocalModelRuntime
from vane.execution.udf_model_pool import ModelPoolBorrow, ModelPoolCapacityError
from vane.execution.udf_runtime_admission import QueryTaskAdmission, TaskAdmissionLimits


def _payload(model, **changes):
    payload = {
        "function_pickle": vane_pickle.dumps(model),
        "call_mode": "map_batches",
        "execution_backend": "subprocess_actor",
        "actor_number": 1,
    }
    payload.update(changes)
    return payload


class _Identity:
    def __call__(self, table):
        return table


class _Plan:
    def __init__(self, payload, *, session="session", config=None):
        self.nodes = [{"node_id": "1", "payload": payload}]
        self.session = session
        self.config = {} if config is None else config
        self.published = []

    def session_id(self):
        return self.session

    def session_config(self):
        return self.config

    def collect_udf_nodes(self, conn=None):
        return self.nodes

    def set_udf_actor_handles(self, options, conn=None):
        self.published.append(options)
        for node in self.nodes:
            if node["node_id"] in options:
                node["executor_options"] = options[node["node_id"]]


def _prepare(runtime, payload):
    plan = _Plan(payload)
    resources = runtime.prepare(plan, {"1": "model"})
    return resources[0], plan.published[-1]["1"]


def _result(executor, value):
    _submit(executor, pa.table({"x": [value]}))
    return _wait_result(executor)


def _submit(executor, table):
    assert executor.request_task_admission(table.nbytes)
    assert executor.task_admission_state()["state"] == "ready"
    executor.submit(table)


def _wait_result(executor):
    ready = threading.Event()
    executor.register_wakeup(ready.set)
    deadline = time.monotonic() + 10
    try:
        while True:
            ready.clear()
            result = executor.take_ready_result()
            if result is not None:
                return result
            remaining = deadline - time.monotonic()
            assert remaining > 0, "model request did not finish"
            ready.wait(remaining)
    finally:
        executor.register_wakeup(None)


def _wait_until(predicate, message):
    deadline = time.monotonic() + 30
    while not predicate():
        assert time.monotonic() < deadline, message
        time.sleep(0.01)


@pytest.mark.parametrize("backend", ["subprocess_actor", "subprocess_task"])
def test_runtime_tracks_queued_outputs_and_views_after_query_and_model_close(backend):
    def identity(table):
        return table

    payload = _payload(
        _Identity if backend == "subprocess_actor" else identity,
        execution_backend=backend,
        udf_worker_slots=1,
        produce_ref_bundle_output=True,
        streaming_output_mode="local_shm_ref_bundle",
    )
    runtime = LocalModelRuntime(
        session_id="session", session_config={}, track_data=True, task_limit=TaskAdmissionLimits(1, 4)
    )
    resources, executor, result = [], None, None
    try:
        bindings = {}
        if backend == "subprocess_actor":
            runtime.register("model", version="v1", payload=payload)
            bindings["1"] = "model"
        plan = _Plan(payload)
        resources = runtime.prepare(plan, bindings)
        executor = build_executor(payload, plan.published[-1]["1"])
        _submit(executor, pa.table({"x": list(range(128))}))
        _wait_until(lambda: bool(executor._queue), "output did not enter result queue")
        snapshot = runtime.resource_snapshot()
        data = snapshot["data"]
        size = data["retained_bytes"]
        assert size > 0
        assert data["input_bytes"] == data["tasks"] == 0
        assert data["output_bytes"] == data["output_state_bytes"]["unit_queue"] == size
        assert snapshot["task_admission"]["running_tasks"] == 0

        result = executor.take_ready_result()
        assert runtime.resource_snapshot()["data"]["output_state_bytes"]["downstream_input"] == size
        table = result[1][0].to_table()
        view = table.slice(3, 2)
        del table
        executor.close()
        for resource in resources:
            resource.shutdown()
        runtime.close()
        for ref in result[1]:
            ref.release()
        assert view.column(0).to_pylist() == [3, 4]
        data = runtime.resource_snapshot()["data"]
        assert data["closed"] and data["queries"] == 0
        assert data["retained_bytes"] == data["output_state_bytes"]["external_consumer"] == size
        del view
        gc.collect()
        assert runtime.resource_snapshot()["data"]["retained_bytes"] == 0
    finally:
        if result is not None:
            for ref in result[1]:
                ref.release()
        if executor is not None:
            executor.close(kill=True)
        for resource in resources:
            resource.shutdown(kill=True)
        runtime.close(timeout=5, kill=True)


@pytest.mark.parametrize("backend", ["subprocess_actor", "subprocess_task"])
def test_data_preparation_failure_releases_query_scopes_and_model_borrows(backend):
    payload = _payload(_Identity, execution_backend=backend)
    with LocalModelRuntime(
        session_id="session", session_config={}, track_data=True, task_limit=TaskAdmissionLimits(1, 4)
    ) as runtime:
        bindings = {}
        if backend == "subprocess_actor":
            runtime.register("model", version="v1", payload=payload)
            bindings["1"] = "model"
        plan = _Plan(payload)

        def fail(_, conn=None):
            raise RuntimeError("cannot publish data scope")

        plan.set_udf_actor_handles = fail
        with pytest.raises(RuntimeError, match="cannot publish data scope"):
            runtime.prepare(plan, bindings)
        snapshot = runtime.resource_snapshot()
        assert snapshot["active_borrows"] == 0
        assert snapshot["task_admission"]["queries"] == 0
        assert snapshot["data"]["queries"] == snapshot["data"]["leases"] == 0


@pytest.mark.parametrize("backend", ["ray_actor", "ray_task", "inline"])
def test_data_only_preparation_rejects_unsupported_backends_before_publishing(backend):
    with LocalModelRuntime(session_id="session", session_config={}, track_data=True) as runtime:
        plan = _Plan({"execution_backend": backend})
        with pytest.raises(ValueError, match="data accounting requires local subprocess"):
            runtime.prepare(plan, {})
        assert not plan.published
        assert runtime.resource_snapshot()["data"]["queries"] == 0


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_data_accounting_requires_an_explicit_boolean(value):
    with pytest.raises(TypeError, match="track_data must be a bool"):
        LocalModelRuntime(session_id="session", session_config={}, track_data=value)


def test_data_accounting_completion_failure_still_returns_task_worker_capacity(monkeypatch):
    from vane.execution import udf_subprocess as local
    from vane.execution.udf_data_lease import TaskDataScope

    local._shutdown_global_task_runtime()
    monkeypatch.setattr(local.os, "cpu_count", lambda: 1)
    original_finish = TaskDataScope.finish

    def finish_then_fail(task):
        original_finish(task)
        raise RuntimeError("planned accounting callback failure")

    def identity(table):
        return table

    def other_identity(table):
        return table.select([0])

    runtime = LocalModelRuntime(
        session_id="session", session_config={}, track_data=True, task_limit=TaskAdmissionLimits(1, 4)
    )
    resources, executors = [], []
    try:
        for function in (identity, other_identity):
            payload = _payload(function, execution_backend="subprocess_task", udf_worker_slots=1)
            plan = _Plan(payload)
            resources.extend(runtime.prepare(plan, {}))
            executors.append(build_executor(payload, plan.published[-1]["1"]))
        failed, next_query = executors
        with monkeypatch.context() as patch:
            patch.setattr(TaskDataScope, "finish", finish_then_fail)
            _submit(failed, pa.table({"x": [1]}))
            _wait_until(lambda: bool(failed._queue), "completion callback did not publish its result")
        with pytest.raises(RuntimeError, match="planned accounting callback failure"):
            failed.take_ready_result()
        assert not failed._task_futures
        assert runtime.resource_snapshot()["task_admission"]["running_tasks"] == 0
        assert local._global_task_runtime().execution_capacity.reserved_slots == 0
        assert _result(next_query, 2).to_pydict() == {"x": [2]}
    finally:
        for executor in executors:
            executor.close(kill=True)
        for resource in resources:
            resource.shutdown(kill=True)
        runtime.close(timeout=5, kill=True)
        local._shutdown_global_task_runtime()
    assert runtime.resource_snapshot()["data"]["retained_bytes"] == 0


@pytest.mark.parametrize("consumer_backend", ["subprocess_actor", "subprocess_task"])
def test_mixed_native_plan_deduplicates_producer_output_and_consumer_input(monkeypatch, consumer_backend):
    from vane.execution import udf_subprocess as local

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    observed = []
    original_track_inputs = local.track_local_shm_inputs

    def observe_inputs(task, refs):
        original_track_inputs(task, refs)
        observed.append(runtime.resource_snapshot()["data"])

    monkeypatch.setattr(local, "track_local_shm_inputs", observe_inputs)

    class Producer:
        def __call__(self, table):
            return table

    def consume(table):
        return pa.table({"x": [value + 1 for value in table.column(0).to_pylist()]})

    class Consumer:
        def __call__(self, table):
            return consume(table)

    with vane.connect() as connection:
        source = connection.sql("SELECT 7::INTEGER AS x")
        relation = source.map_batches(
            Producer, schema={"x": vane.sqltypes.INTEGER}, execution_backend="subprocess_actor", actor_number=1
        ).map_batches(
            Consumer if consumer_backend == "subprocess_actor" else consume,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend=consumer_backend,
            actor_number=1 if consumer_backend == "subprocess_actor" else None,
        )
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(connection)
        with LocalModelRuntime(
            session_id=plan.session_id(), session_config=plan.session_config(), track_data=True
        ) as runtime:
            producer = next(
                node
                for node in plan.collect_udf_nodes(conn=connection)
                if node["payload"]["udf_name"] == Producer.__qualname__
            )
            runtime.register("producer", version="v1", payload=producer["payload"])
            resources = runtime.prepare(plan, {str(producer["node_id"]): "producer"}, conn=connection)
            try:
                result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(connection, plan)
                assert [value for table in result.partition_payloads for value in table.column(0).to_pylist()] == [8]
            finally:
                for resource in resources:
                    resource.shutdown(kill=True)
            assert any(data["retained_bytes"] < data["input_bytes"] + data["output_bytes"] for data in observed)
            assert runtime.resource_snapshot()["data"]["queries"] == 0
            assert runtime.resource_snapshot()["data"]["input_bytes"] == 0
            del result
            gc.collect()
            assert runtime.resource_snapshot()["data"]["retained_bytes"] == 0


@pytest.mark.parametrize("backend", ["subprocess_task", "ray_actor", "ray_task"])
def test_registration_does_not_cache_arbitrary_udfs_or_ray_query_capabilities(backend):
    with LocalModelRuntime(session_id="session", session_config={}) as runtime:
        with pytest.raises(ValueError, match="subprocess_actor"):
            runtime.register("model", version="v1", payload=_payload(_Identity, execution_backend=backend))


def test_registration_rejects_gpu_until_device_admission_exists():
    with LocalModelRuntime(session_id="session", session_config={}) as runtime:
        with pytest.raises(ValueError, match="GPU resources"):
            runtime.register("model", version="v1", payload=_payload(_Identity, gpus=1))


@pytest.mark.parametrize("changed", ["session", "config", "payload", "size", "unknown_node"])
def test_plan_binding_validates_session_and_model_before_publishing(changed):
    payload = _payload(_Identity)
    with LocalModelRuntime(session_id="session", session_config={}) as runtime:
        runtime.register("model", version="v1", payload=payload)
        plan = _Plan(dict(payload))
        bindings = {"1": "model"}
        if changed == "session":
            plan.session = "another-session"
        elif changed == "config":
            plan.config = {"VANE_RUNNER": "local-fast"}
        elif changed == "payload":
            plan.nodes[0]["payload"]["batch_size"] = 32
        elif changed == "size":
            plan.nodes[0]["payload"]["actor_number"] = 2
        else:
            bindings["missing"] = "model"
        with pytest.raises(ValueError):
            runtime.prepare(plan, bindings)
        assert not plan.published


def test_plan_publication_failure_releases_borrow_without_closing_resident_model():
    payload = _payload(_Identity)
    with LocalModelRuntime(session_id="session", session_config={}) as runtime:
        model = runtime.register("model", version="v1", payload=payload)
        model.prewarm()
        with model.acquire() as owner:
            pids = owner.pool.worker_pids()
        plan = _Plan(payload)

        def fail(_, conn=None):
            raise RuntimeError("cannot publish plan handles")

        plan.set_udf_actor_handles = fail
        with pytest.raises(RuntimeError, match="cannot publish"):
            runtime.prepare(plan, {"1": "model"})
        with model.acquire() as later:
            assert later.pool.worker_pids() == pids
        # Context close would time out if the failed publication leaked a borrow.


@pytest.mark.parametrize("fail_publication", [False, True])
def test_mixed_plan_preparation_preserves_options_and_pool_ownership(monkeypatch, fail_publication):
    import vane.execution.udf_subprocess as local

    pools = []
    config = {"AWS_VANE_MODEL_SESSION_TEST": "captured"}

    class Pool:
        def __init__(self, payload, pool_size, *, name, session_config=None):
            self.session_config = session_config
            self.closed = False
            pools.append(self)

        def shutdown(self, *, kill=False):
            self.closed = True

        def cleanup_pending(self):
            return not self.closed

    monkeypatch.setattr(local, "LocalSubprocessActorPool", Pool)
    payload = _payload(_Identity)
    plan = _Plan(payload, config=config)
    plan.nodes.extend(
        [
            {"node_id": "2", "payload": payload},
            {"node_id": "3", "payload": {"execution_backend": "subprocess_task"}},
        ]
    )
    original_options = []
    for node in plan.nodes:
        options = {"session_config": {"AWS_VANE_MODEL_SESSION_TEST": "stale"}, "custom_option": object()}
        node["executor_options"] = options
        original_options.append(options)
    published = []

    def publish(options, conn=None):
        published.append(options)
        if fail_publication:
            raise RuntimeError("cannot publish mixed plan")

    plan.set_udf_actor_handles = publish
    with LocalModelRuntime(session_id="session", session_config=config) as runtime:
        model = runtime.register("model", version="v1", payload=payload)
        if fail_publication:
            with pytest.raises(RuntimeError, match="cannot publish mixed plan"):
                runtime.prepare(plan, {"1": "model"})
        else:
            resources = runtime.prepare(plan, {"1": "model"})
            assert len(resources) == 2
            assert isinstance(resources[0], ModelPoolBorrow)
            assert resources[1] is pools[1]
            for resource in resources:
                resource.shutdown()
        assert len(published) == 1
        assert set(published[0]) == {"1", "2", "3"}
        for index, options in enumerate(published[0].values()):
            assert options["session_config"] == config
            assert options["custom_option"] is original_options[index]["custom_option"]
            assert original_options[index]["session_config"] == {"AWS_VANE_MODEL_SESSION_TEST": "stale"}
        assert published[0]["1"]["local_actor_pool"] is pools[0]
        assert published[0]["2"]["local_actor_pool"] is pools[1]
        assert "local_actor_pool" not in published[0]["3"]
        assert "local_model_pool" not in published[0]["2"]
        assert "local_model_pool" not in published[0]["3"]
        assert len(pools) == 2
        assert all(pool.session_config == config for pool in pools)
        assert not pools[0].closed
        assert pools[1].closed
        with model.acquire() as borrow:
            assert borrow.pool is pools[0]
    assert all(pool.closed for pool in pools)


def test_partial_constructor_ownership_is_not_transferred_to_query_rollback(monkeypatch):
    import vane.execution.udf_subprocess as local

    calls = []

    class PendingPool:
        def shutdown(self, *, kill=False):
            calls.append(kill)

        def cleanup_pending(self):
            return not calls

    pending = PendingPool()

    def fail(*args, **kwargs):
        raise OwnedActorPoolsError(
            "cleanup incomplete", owned_actor_pools=[pending], creation_error=ValueError("model init failed")
        )

    monkeypatch.setattr(local, "LocalSubprocessActorPool", fail)
    with LocalModelRuntime(session_id="session", session_config={}) as runtime:
        payload = _payload(_Identity)
        runtime.register("model", version="v1", payload=payload)
        plan = _Plan(payload)
        with pytest.raises(ValueError, match="model init failed"):
            runtime.prepare(plan, {"1": "model"})
        assert not calls
    assert calls == [False]


def test_registered_model_survives_query_cancellation_and_replaces_lost_worker():
    payload = _payload(_Identity, memory_bytes=128)
    limit = ResourceVector(cpu=1, heap_bytes=128)
    with LocalModelRuntime(session_id="session", session_config={}, resident_limit=limit) as runtime:
        model = runtime.register("model", version="v1", payload=payload)
        first, options = _prepare(runtime, payload)
        executor = build_executor(payload, options)
        try:
            assert _result(executor, 1).to_pydict() == {"x": [1]}
        finally:
            executor.close(kill=True)
            first.shutdown(kill=True)
        assert runtime.resource_snapshot()["reserved_resources"] == limit.to_dict()
        with model.acquire() as borrow:
            worker = borrow.pool.first_proc()
            worker.kill()
            worker.wait(timeout=5)
            old_pid = worker.pid
        second, options = _prepare(runtime, payload)
        executor = build_executor(payload, options)
        try:
            assert _result(executor, 2).to_pydict() == {"x": [2]}
            assert second.pool.worker_pids() != [old_pid]
            assert runtime.resource_snapshot()["reserved_resources"] == limit.to_dict()
        finally:
            executor.close()
            second.release()
    assert runtime.resource_snapshot()["reserved_resources"] == ResourceVector().to_dict()


def test_runtime_limit_blocks_another_model_before_starting_its_processes(monkeypatch):
    import vane.execution.udf_subprocess as local

    constructed = []
    original = local.LocalSubprocessActorPool

    def create(*args, **kwargs):
        constructed.append(kwargs["name"])
        return original(*args, **kwargs)

    monkeypatch.setattr(local, "LocalSubprocessActorPool", create)
    payload = _payload(_Identity, actor_number=2, cpus=0.25, memory_bytes=128)
    limit = ResourceVector(cpu=0.5, heap_bytes=256)
    with LocalModelRuntime(session_id="session", session_config={}, resident_limit=limit) as runtime:
        model = runtime.register("model", version="v1", payload=payload)
        other = runtime.register("other", version="v1", payload=payload)
        assert model.resident_resources == limit
        assert runtime.resource_snapshot()["reserved_models"] == 0
        runtime.prewarm("model")
        first, options_a = _prepare(runtime, payload)
        second, options_b = _prepare(runtime, payload)
        executor_a = build_executor(payload, options_a)
        executor_b = build_executor(payload, options_b)
        try:
            assert first.pool.worker_pids() == second.pool.worker_pids()
            assert len(first.pool.worker_pids()) == 2
            with pytest.raises(ModelPoolCapacityError, match="capacity is in use") as error:
                other.prewarm()
            assert set(error.value.dimensions) == {"cpu", "heap_bytes"}
            assert constructed == ["model-model-v1"]
            assert runtime.resource_snapshot()["active_borrows"] == 2
            assert runtime.resource_snapshot()["reserved_resources"] == limit.to_dict()
            assert _result(executor_a, 1).to_pydict() == {"x": [1]}
            executor_a.close(kill=True)
            first.release()
            assert _result(executor_b, 2).to_pydict() == {"x": [2]}
            assert runtime.resource_snapshot()["reserved_resources"] == limit.to_dict()
        finally:
            executor_a.close(kill=True)
            executor_b.close(kill=True)
            first.release()
            second.release()
    assert runtime.resource_snapshot()["reserved_models"] == 0


def test_preparation_capacity_failure_releases_borrows_and_preserves_resident_budget():
    payload = _payload(_Identity)
    limit = ResourceVector(cpu=1)
    with LocalModelRuntime(session_id="session", session_config={}, resident_limit=limit) as runtime:
        runtime.register("model", version="v1", payload=payload)
        runtime.register("other", version="v1", payload=payload)
        plan = _Plan(payload)
        plan.nodes.append({"node_id": "2", "payload": payload})
        with pytest.raises(ModelPoolCapacityError):
            runtime.prepare(plan, {"1": "model", "2": "other"})
        assert not plan.published
        assert runtime.resource_snapshot()["active_borrows"] == 0
        assert runtime.resource_snapshot()["reserved_resources"] == limit.to_dict()
        borrow, options = _prepare(runtime, payload)
        executor = build_executor(payload, options)
        try:
            assert _result(executor, 3).to_pydict() == {"x": [3]}
        finally:
            executor.close()
            borrow.release()
    assert runtime.resource_snapshot()["reserved_resources"] == ResourceVector().to_dict()


def test_cancelling_one_query_does_not_cancel_another_borrowers_output(monkeypatch):
    import vane.execution.udf_subprocess as local
    from vane.execution.ref_bundle import REF_BUNDLE_RESULT_MARKER

    first_waiting = threading.Event()
    second_waiting = threading.Event()
    finish_output = threading.Event()
    original_request = local.request_local_shm_output_grant
    cancellations = []

    def request(size, *, cancel_event=None, **kwargs):
        cancellations.append(cancel_event)
        (first_waiting if len(cancellations) == 1 else second_waiting).set()
        deadline = time.monotonic() + 10
        while not finish_output.wait(0.01):
            if cancel_event.is_set():
                raise RuntimeError("request output cancelled")
            assert time.monotonic() < deadline
        assert not cancel_event.is_set()
        return original_request(size, cancel_event=cancel_event, **kwargs)

    monkeypatch.setattr(local, "request_local_shm_output_grant", request)
    payload = _payload(
        _Identity, actor_number=2, produce_ref_bundle_output=True, streaming_output_mode="local_shm_ref_bundle"
    )
    with LocalModelRuntime(session_id="session", session_config={}) as runtime:
        runtime.register("model", version="v1", payload=payload)
        first, options_a = _prepare(runtime, payload)
        second, options_b = _prepare(runtime, payload)
        executor_a = build_executor(payload, options_a)
        executor_b = build_executor(payload, options_b)
        result = None
        try:
            _submit(executor_a, pa.table({"x": [1]}))
            assert first_waiting.wait(10)
            _submit(executor_b, pa.table({"x": [7]}))
            assert second_waiting.wait(10)
            executor_a.close(kill=True)
            first.shutdown(kill=True)
            assert cancellations[0].is_set()
            assert not cancellations[1].is_set()
            finish_output.set()
            result = _wait_result(executor_b)
            assert result[0] == REF_BUNDLE_RESULT_MARKER
        finally:
            finish_output.set()
            executor_a.close(kill=True)
            executor_b.close(kill=True)
            first.release()
            second.release()
            if result is not None:
                for ref in result[1]:
                    ref.release()


@pytest.mark.parametrize("method", ["map_batches", "flat_map"])
def test_native_plan_declared_heap_enforces_admission_and_preserves_compatibility(monkeypatch, tmp_path, method):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    initialized = str(tmp_path / "initializations.txt")

    class Model:
        def __init__(self):
            with open(initialized, "a") as file:
                file.write("initialized\n")

        def __call__(self, value):
            return [value, value] if method == "flat_map" else value

    with vane.connect() as connection:

        def make_plan(memory_bytes=1024):
            relation = getattr(connection.sql("SELECT 7::INTEGER AS x"), method)(
                Model,
                schema={"x": vane.sqltypes.INTEGER},
                execution_backend="subprocess_actor",
                actor_number=1,
                memory_bytes=memory_bytes,
            )
            return vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(
                connection
            )

        first_plan = make_plan()
        payload = first_plan.collect_udf_nodes(conn=connection)[0]["payload"]
        assert payload["memory_bytes"] == 1024
        with LocalModelRuntime(
            session_id=first_plan.session_id(),
            session_config=first_plan.session_config(),
            resident_limit=ResourceVector(cpu=4, heap_bytes=1023),
        ) as insufficient:
            with pytest.raises(ModelPoolCapacityError) as error:
                insufficient.register("model", version="v1", payload=payload)
            assert error.value.oversized
            assert error.value.dimensions == ("heap_bytes",)
        assert not (tmp_path / "initializations.txt").exists()

        with LocalModelRuntime(
            session_id=first_plan.session_id(),
            session_config=first_plan.session_config(),
            resident_limit=ResourceVector(cpu=4, heap_bytes=1024),
        ) as runtime:
            runtime.register("model", version="v1", payload=payload)
            runtime.register("other", version="v1", payload=payload)
            runtime.prewarm("model")
            for plan in (first_plan, make_plan()):
                node = plan.collect_udf_nodes(conn=connection)[0]
                assert node["payload"]["memory_bytes"] == 1024
                resources = runtime.prepare(plan, {str(node["node_id"]): "model"}, conn=connection)
                try:
                    result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(connection, plan)
                    rows = [value for table in result.partition_payloads for value in table.column(0).to_pylist()]
                    assert rows == ([7, 7] if method == "flat_map" else [7])
                finally:
                    for resource in resources:
                        resource.shutdown()
                snapshot = runtime.resource_snapshot()
                assert snapshot["active_borrows"] == 0
                assert snapshot["reserved_resources"] == ResourceVector(cpu=1, heap_bytes=1024).to_dict()
            with pytest.raises(ModelPoolCapacityError) as error:
                runtime.prewarm("other")
            assert not error.value.oversized
            assert error.value.dimensions == ("heap_bytes",)
            changed_plan = make_plan(memory_bytes=512)
            node = changed_plan.collect_udf_nodes(conn=connection)[0]
            with pytest.raises(ValueError, match="payload or pool size"):
                runtime.prepare(changed_plan, {str(node["node_id"]): "model"}, conn=connection)
            assert (tmp_path / "initializations.txt").read_text().splitlines() == ["initialized"]
        assert runtime.resource_snapshot()["reserved_resources"] == ResourceVector().to_dict()


@pytest.mark.parametrize("cpu_limit", [0, 1], ids=["configured-zero", "exhausted"])
def test_native_plan_tiny_cpu_does_not_start_workers_without_capacity(monkeypatch, tmp_path, cpu_limit):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    initialized = str(tmp_path / "initializations.txt")

    class Model:
        def __init__(self):
            with open(initialized, "a") as file:
                file.write("initialized\n")

        def __call__(self, table):
            return table

    with vane.connect() as connection:

        def make_plan(cpus):
            relation = connection.sql("SELECT 1::INTEGER AS x").map_batches(
                Model,
                schema={"x": vane.sqltypes.INTEGER},
                execution_backend="subprocess_actor",
                actor_number=1,
                cpus=cpus,
            )
            return vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(
                connection
            )

        tiny = make_plan(1e-13)
        with LocalModelRuntime(
            session_id=tiny.session_id(),
            session_config=tiny.session_config(),
            resident_limit=ResourceVector(cpu=cpu_limit),
        ) as runtime:
            if cpu_limit:
                full = make_plan(cpu_limit)
                runtime.register("full", version="v1", payload=full.collect_udf_nodes(conn=connection)[0]["payload"])
                runtime.prewarm("full")
            with pytest.raises(ModelPoolCapacityError) as error:
                runtime.register("tiny", version="v1", payload=tiny.collect_udf_nodes(conn=connection)[0]["payload"])
                runtime.prewarm("tiny")
            assert error.value.oversized is (cpu_limit == 0)
            assert error.value.dimensions == ("cpu",)
            snapshot = runtime.resource_snapshot()
            assert snapshot["registered_models"] == (2 if cpu_limit else 0)
            assert snapshot["reserved_resources"] == ResourceVector(cpu=cpu_limit).to_dict()
            if cpu_limit:
                assert (tmp_path / "initializations.txt").read_text().splitlines() == ["initialized"]
            else:
                assert not (tmp_path / "initializations.txt").exists()


@pytest.mark.parametrize("backend", ["subprocess_actor", "subprocess_task"])
@pytest.mark.parametrize("captured", ["session-a", None], ids=["captured-value", "missing-value"])
def test_mixed_native_plan_uses_captured_session_for_every_udf(monkeypatch, backend, captured):
    import os

    variable = "AWS_VANE_MODEL_SESSION_TEST"
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    if captured is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, captured)

    class Model:
        def __init__(self):
            self.environment = os.environ.get(variable, "<missing>")

        def __call__(self, table):
            return table.append_column("model_config", pa.array([self.environment])).append_column(
                "model_pid", pa.array([os.getpid()])
            )

    def observe(table):
        return table.append_column("neighbor_config", pa.array([os.environ.get(variable, "<missing>")])).append_column(
            "neighbor_pid", pa.array([os.getpid()])
        )

    class Neighbor:
        def __init__(self):
            self.environment = os.environ.get(variable, "<missing>")

        def __call__(self, table):
            return table.append_column("neighbor_config", pa.array([self.environment])).append_column(
                "neighbor_pid", pa.array([os.getpid()])
            )

    model_schema = {
        "x": vane.sqltypes.INTEGER,
        "model_config": vane.sqltypes.VARCHAR,
        "model_pid": vane.sqltypes.BIGINT,
    }
    schema = {**model_schema, "neighbor_config": vane.sqltypes.VARCHAR, "neighbor_pid": vane.sqltypes.BIGINT}
    with vane.connect() as connection:
        plans = []
        for _ in range(2):
            relation = connection.sql("SELECT 1::INTEGER AS x").map_batches(
                Model, schema=model_schema, execution_backend="subprocess_actor", actor_number=1
            )
            relation = relation.map_batches(
                Neighbor if backend == "subprocess_actor" else observe,
                schema=schema,
                execution_backend=backend,
                actor_number=1 if backend == "subprocess_actor" else None,
            )
            plans.append(
                vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(connection)
            )
        assert plans[0].session_config().get(variable) == captured
        monkeypatch.setenv(variable, "session-b")
        with LocalModelRuntime(session_id=plans[0].session_id(), session_config=plans[0].session_config()) as runtime:
            model_node = next(
                node
                for node in plans[0].collect_udf_nodes(conn=connection)
                if node["payload"]["udf_name"] == Model.__qualname__
            )
            model = runtime.register("model", version="v1", payload=model_node["payload"])
            rows = []
            for plan in plans:
                nodes = plan.collect_udf_nodes(conn=connection)
                assert len(nodes) == 2
                model_node = next(node for node in nodes if node["payload"]["udf_name"] == Model.__qualname__)
                resources = runtime.prepare(plan, {str(model_node["node_id"]): "model"}, conn=connection)
                try:
                    assert sum(isinstance(resource, ModelPoolBorrow) for resource in resources) == 1
                    query_pools = [resource for resource in resources if not isinstance(resource, ModelPoolBorrow)]
                    assert len(query_pools) == (1 if backend == "subprocess_actor" else 0)
                    result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(connection, plan)
                    values = [
                        row
                        for table in result.partition_payloads
                        for row in table.rename_columns(list(schema)).to_pylist()
                    ]
                    assert len(values) == 1
                    row = values[0]
                    assert row["model_config"] == (captured or "<missing>")
                    assert row["neighbor_config"] == (captured or "<missing>")
                    rows.append(row)
                finally:
                    for resource in resources:
                        resource.shutdown()
                assert all(not pool.cleanup_pending() for pool in query_pools)
                with model.acquire() as borrow:
                    assert borrow.pool.worker_pids() == [row["model_pid"]]
                    assert borrow.pool.first_proc().poll() is None
            assert len({row["model_pid"] for row in rows}) == 1
            if backend == "subprocess_actor":
                assert len({row["neighbor_pid"] for row in rows}) == 2
        assert os.environ[variable] == "session-b"


def test_independent_native_queries_reuse_registered_model_sequentially_and_concurrently(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    initialized = str(tmp_path / "initializations.txt")

    class Model:
        def __init__(self):
            import os

            self.pid = os.getpid()
            self.calls = 0
            with open(initialized, "a") as file:
                file.write(f"{self.pid}\n")

        def __call__(self, table):
            self.calls += 1
            return pa.table({"x": table.column("x"), "pid": [self.pid], "calls": [self.calls]})

    schema = {"x": vane.sqltypes.BIGINT, "pid": vane.sqltypes.BIGINT, "calls": vane.sqltypes.BIGINT}
    with vane.connect() as connection:
        plans = []
        cursors = [connection.cursor() for _ in range(4)]
        for value in range(4):
            relation = (
                cursors[value].sql(f"SELECT {value}::BIGINT AS x").map_batches(Model, schema=schema, actor_number=1)
            )
            plans.append(
                vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(
                    cursors[value]
                )
            )
        with LocalModelRuntime(session_id=plans[0].session_id(), session_config=plans[0].session_config()) as runtime:
            runtime.register("model", version="v1", payload=plans[0].collect_udf_nodes(conn=cursors[0])[0]["payload"])
            runtime.prewarm("model")

            def execute(index):
                node = plans[index].collect_udf_nodes(conn=cursors[index])[0]
                resources = runtime.prepare(plans[index], {str(node["node_id"]): "model"}, conn=cursors[index])
                try:
                    result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(cursors[index], plans[index])
                    return [
                        row
                        for table in result.partition_payloads
                        for row in table.rename_columns(list(schema)).to_pylist()
                    ]
                finally:
                    for resource in resources:
                        resource.shutdown()

            try:
                rows = execute(0) + execute(1)
                with ThreadPoolExecutor(max_workers=2) as threads:
                    rows.extend(row for result in threads.map(execute, [2, 3]) for row in result)
                assert sorted(row["x"] for row in rows) == [0, 1, 2, 3]
                assert sorted(row["calls"] for row in rows) == [1, 2, 3, 4]
                assert len({row["pid"] for row in rows}) == 1
                assert (tmp_path / "initializations.txt").read_text().splitlines() == [str(rows[0]["pid"])]
            finally:
                for cursor in cursors:
                    cursor.close()


def test_repeated_sql_queries_reuse_one_attached_class_model(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    initialized = str(tmp_path / "initializations.txt")

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class Model:
        def __init__(self):
            import os

            self.calls = 0
            with open(initialized, "a") as file:
                file.write(f"{os.getpid()}\n")

        def __call__(self, value):
            self.calls += 1
            return value + self.calls

    with vane.connect() as connection:
        vane.attach_function(Model(), alias="resident_model", connection=connection, parameters=["INTEGER"])
        cursors = [connection.cursor() for _ in range(2)]
        try:
            plans = [
                vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
                    cursor.sql("SELECT resident_model(10::INTEGER) AS result"), uuid.uuid4().hex
                ).to_physical_plan(cursor)
                for cursor in cursors
            ]
            nodes = [plan.collect_udf_nodes(conn=cursor)[0] for plan, cursor in zip(plans, cursors, strict=True)]
            assert nodes[0]["payload"]["expression_id"] != nodes[1]["payload"]["expression_id"]
            with LocalModelRuntime(
                session_id=plans[0].session_id(), session_config=plans[0].session_config()
            ) as runtime:
                runtime.register("model", version="v1", payload=nodes[0]["payload"])
                runtime.prewarm("model")
                results = []
                for plan, node, cursor in zip(plans, nodes, cursors, strict=True):
                    resources = runtime.prepare(plan, {str(node["node_id"]): "model"}, conn=cursor)
                    try:
                        result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(cursor, plan)
                        results.extend(
                            value for table in result.partition_payloads for value in table.column(0).to_pylist()
                        )
                    finally:
                        for resource in resources:
                            resource.shutdown()
                assert results == [11, 12]
                assert len((tmp_path / "initializations.txt").read_text().splitlines()) == 1
        finally:
            for cursor in cursors:
                cursor.close()


@pytest.mark.parametrize("batched", [False, True], ids=["cls", "cls.batch"])
def test_rebuilt_projections_reuse_one_model_sequentially_and_concurrently(monkeypatch, tmp_path, batched):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    initialized = str(tmp_path / "initializations.txt")

    class Model:
        def __init__(self, offset, *, scale):
            import os

            self.offset = offset
            self.scale = scale
            self.calls = 0
            with open(initialized, "a") as file:
                file.write(f"{os.getpid()}\n")

        def __call__(self, value):
            self.calls += 1
            if batched:
                return pa.array(
                    [item * self.scale + self.offset + self.calls for item in value.to_pylist()], type=pa.int32()
                )
            return value * self.scale + self.offset + self.calls

    decorate = vane.cls.batch if batched else vane.cls
    model = decorate(actor_number=1, return_dtype="INTEGER")(Model)(5, scale=2)
    with vane.connect() as connection:
        cursors = [connection.cursor() for _ in range(4)]
        try:

            def make_plan(index):
                # Build a new expression each time, including after prior queries
                # have finished and concurrently with other query preparation.
                relation = cursors[index].sql("SELECT 10::INTEGER AS x").select(model(vane.col("x")).alias("out"))
                return vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(
                    cursors[index]
                )

            first_plan = make_plan(0)
            with LocalModelRuntime(
                session_id=first_plan.session_id(), session_config=first_plan.session_config()
            ) as runtime:
                runtime.register(
                    "model", version="v1", payload=first_plan.collect_udf_nodes(conn=cursors[0])[0]["payload"]
                )
                runtime.prewarm("model")

                def execute(index):
                    plan = first_plan if index == 0 else make_plan(index)
                    node = plan.collect_udf_nodes(conn=cursors[index])[0]
                    resources = runtime.prepare(plan, {str(node["node_id"]): "model"}, conn=cursors[index])
                    try:
                        result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(cursors[index], plan)
                        return [value for table in result.partition_payloads for value in table.column(0).to_pylist()]
                    finally:
                        for resource in resources:
                            resource.shutdown()

                assert execute(0) == [26]
                assert execute(1) == [27]
                with ThreadPoolExecutor(max_workers=2) as threads:
                    assert sorted(value for rows in threads.map(execute, [2, 3]) for value in rows) == [28, 29]
                assert len((tmp_path / "initializations.txt").read_text().splitlines()) == 1
        finally:
            for cursor in cursors:
                cursor.close()


@pytest.mark.parametrize("batched", [False, True], ids=["cls", "cls.batch"])
@pytest.mark.parametrize("changed", ["class", "init_args", "init_kwargs", "call", "schema"])
def test_rebuilt_projection_rejects_changed_model_contract(monkeypatch, batched, changed):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    class Model:
        def __init__(self, offset, *, scale):
            self.offset = offset
            self.scale = scale

        def __call__(self, value, *, increment=0):
            if batched:
                return pa.array([item * self.scale + self.offset for item in value.to_pylist()], type=pa.int32())
            return value * self.scale + self.offset + increment

    class OtherModel(Model):
        pass

    decorate = vane.cls.batch if batched else vane.cls
    constructor = decorate(actor_number=1, return_dtype="INTEGER", name="resident_model")(Model)
    model = constructor(5, scale=2)
    changed_model = model
    if changed == "class":
        changed_model = decorate(actor_number=1, return_dtype="INTEGER", name="resident_model")(OtherModel)(5, scale=2)
    elif changed == "init_args":
        changed_model = constructor(6, scale=2)
    elif changed == "init_kwargs":
        changed_model = constructor(5, scale=3)
    elif changed == "schema":
        changed_model = decorate(actor_number=1, return_dtype="BIGINT", name="resident_model")(Model)(5, scale=2)

    with vane.connect() as connection:
        cursors = [connection.cursor() for _ in range(2)]
        try:
            original = model(vane.col("x"))
            if changed == "call":
                rebuilt = changed_model(value=vane.col("x")) if batched else changed_model(vane.col("x"), increment=1)
            else:
                rebuilt = changed_model(vane.col("x"))
            plans = [
                vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
                    cursor.sql("SELECT 10::INTEGER AS x").select(expression.alias("out")), uuid.uuid4().hex
                ).to_physical_plan(cursor)
                for cursor, expression in zip(cursors, [original, rebuilt], strict=True)
            ]
            nodes = [plan.collect_udf_nodes(conn=cursor)[0] for plan, cursor in zip(plans, cursors, strict=True)]
            with LocalModelRuntime(
                session_id=plans[0].session_id(), session_config=plans[0].session_config()
            ) as runtime:
                runtime.register("model", version="v1", payload=nodes[0]["payload"])
                with pytest.raises(ValueError, match="payload"):
                    resources = runtime.prepare(plans[1], {str(nodes[1]["node_id"]): "model"}, conn=cursors[1])
                    for resource in resources:
                        resource.shutdown()
        finally:
            for cursor in cursors:
                cursor.close()


def test_task_admission_preparation_failure_releases_query_and_preserves_model():
    payload = _payload(_Identity)
    with LocalModelRuntime(session_id="session", session_config={}, task_limit=TaskAdmissionLimits(1, 4)) as runtime:
        model = runtime.register("model", version="v1", payload=payload)
        plan = _Plan(payload)

        def fail(_, conn=None):
            raise RuntimeError("cannot publish admission")

        plan.set_udf_actor_handles = fail
        with pytest.raises(RuntimeError, match="cannot publish admission"):
            runtime.prepare(plan, {"1": "model"})
        snapshot = runtime.resource_snapshot()
        assert snapshot["active_borrows"] == 0
        assert snapshot["task_admission"]["queries"] == 0
        with model.acquire() as borrow:
            assert borrow.pool.first_proc().poll() is None


@pytest.mark.parametrize("backend", ["ray_actor", "ray_task", "inline"])
def test_task_admission_rejects_unsupported_plan_before_starting_models(backend):
    payload = _payload(_Identity)
    with LocalModelRuntime(session_id="session", session_config={}, task_limit=TaskAdmissionLimits(1, 4)) as runtime:
        runtime.register("model", version="v1", payload=payload)
        plan = _Plan(payload)
        plan.nodes.append({"node_id": "2", "payload": {"execution_backend": backend}})
        with pytest.raises(ValueError, match="local subprocess"):
            runtime.prepare(plan, {"1": "model"})
        assert not plan.published
        assert runtime.resource_snapshot()["reserved_models"] == 0
        assert runtime.resource_snapshot()["task_admission"]["queries"] == 0


def test_runtime_task_completion_releases_global_quota_before_output_is_consumed():
    payload = _payload(_Identity)
    with LocalModelRuntime(session_id="session", session_config={}, task_limit=TaskAdmissionLimits(1, 4)) as runtime:
        runtime.register("first", version="v1", payload=payload)
        runtime.register("second", version="v1", payload=payload)
        resources, executors = [], []
        try:
            for model in ["first", "first", "second"]:
                plan = _Plan(payload)
                resources.extend(runtime.prepare(plan, {"1": model}))
                executors.append(build_executor(payload, plan.published[-1]["1"]))
            first, same_pool, other_pool = executors
            _submit(first, pa.table({"x": [1]}))
            assert same_pool.request_task_admission(8)
            assert other_pool.request_task_admission(8)
            _wait_until(
                lambda: other_pool.task_admission_state()["available"], "other model did not get execution quota"
            )
            _wait_until(lambda: bool(first._queue), "completed output was not queued")
            assert same_pool.task_admission_state()["state"] == "requested"
            other_pool.submit(pa.table({"x": [2]}))
            assert _wait_result(other_pool).to_pydict() == {"x": [2]}
            assert first.take_ready_result().to_pydict() == {"x": [1]}
            _wait_until(lambda: same_pool.task_admission_state()["available"], "buffered result slot was not returned")
            same_pool.submit(pa.table({"x": [3]}))
            assert _wait_result(same_pool).to_pydict() == {"x": [3]}
        finally:
            for executor in executors:
                executor.close(kill=True)
            for resource in resources:
                resource.shutdown(kill=True)
        snapshot = runtime.resource_snapshot()
        assert snapshot["reserved_models"] == 2
        assert snapshot["task_admission"]["running_tasks"] == 0
        assert snapshot["task_admission"]["queries"] == 0


@pytest.mark.parametrize("track_data", [False, True])
@pytest.mark.parametrize("failure", ["exception", "worker_exit", "cancel"])
def test_failed_or_cancelled_subprocess_returns_runtime_quota_and_keeps_model(tmp_path, failure, track_data):
    entered, release = str(tmp_path / "entered"), str(tmp_path / "release")

    class Model:
        def __call__(self, table):
            import os
            from pathlib import Path

            if table.column(0)[0].as_py() == 0:
                if failure == "exception":
                    raise ValueError("request failed")
                if failure == "worker_exit":
                    os._exit(7)
                Path(entered).touch()
                deadline = time.monotonic() + 30
                while not Path(release).exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("cancel test did not release worker")
                    time.sleep(0.01)
            return table

    payload = _payload(Model)
    with LocalModelRuntime(
        session_id="session", session_config={}, task_limit=TaskAdmissionLimits(1, 4), track_data=track_data
    ) as runtime:
        model = runtime.register("model", version="v1", payload=payload)
        resources, executors = [], []
        try:
            for _ in range(2):
                plan = _Plan(payload)
                resources.extend(runtime.prepare(plan, {"1": "model"}))
                executors.append(build_executor(payload, plan.published[-1]["1"]))
            failed, next_query = executors
            _submit(failed, pa.table({"x": [0]}))
            assert next_query.request_task_admission(8)
            if failure == "cancel":
                _wait_until(lambda: (tmp_path / "entered").exists(), "worker did not start")
                if track_data:
                    assert runtime.resource_snapshot()["data"]["input_bytes"] > 0
                failed.close(kill=True)
            else:
                assert isinstance(_wait_result(failed), BaseException)
            _wait_until(lambda: next_query.task_admission_state()["available"], "failed request retained quota")
            next_query.submit(pa.table({"x": [7]}))
            assert _wait_result(next_query).to_pydict() == {"x": [7]}
            with model.acquire() as borrow:
                assert borrow.pool.first_proc().poll() is None
        finally:
            (tmp_path / "release").touch()
            for executor in executors:
                executor.close(kill=True)
            for resource in resources:
                resource.shutdown(kill=True)
        assert runtime.resource_snapshot()["task_admission"]["running_tasks"] == 0
        assert runtime.resource_snapshot()["task_admission"]["queries"] == 0
        if track_data:
            data = runtime.resource_snapshot()["data"]
            assert data["queries"] == data["tasks"] == data["retained_bytes"] == 0


@pytest.mark.parametrize("tracking", ["tasks", "data", "both"])
def test_task_only_native_plan_participates_in_runtime_drain_and_close(monkeypatch, tmp_path, tracking):
    from vane.execution.udf_data_lease import QueryDataScope

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    entered, release = str(tmp_path / "entered"), str(tmp_path / "release")

    def identity(table):
        from pathlib import Path

        Path(entered).touch()
        deadline = time.monotonic() + 30
        while not Path(release).exists():
            if time.monotonic() > deadline:
                raise TimeoutError("task-only admission test did not release worker")
            time.sleep(0.01)
        return table

    with vane.connect() as connection:
        relation = connection.sql("SELECT 7::INTEGER AS x").map_batches(
            identity, schema={"x": vane.sqltypes.INTEGER}, execution_backend="subprocess_task"
        )
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(connection)
        runtime = LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            task_limit=TaskAdmissionLimits(1, 4) if tracking != "data" else None,
            track_data=tracking != "tasks",
        )
        resources = runtime.prepare(plan, {}, conn=connection)
        assert len(resources) == (2 if tracking == "both" else 1)
        assert isinstance(resources[0], QueryDataScope if tracking == "data" else QueryTaskAdmission)
        try:
            with pytest.raises(TimeoutError, match="active queries"):
                runtime.close()
            with pytest.raises(RuntimeError, match="draining"):
                runtime.prepare(plan, {}, conn=connection)
            # Draining must allow the already prepared query to create executors.
            with ThreadPoolExecutor(max_workers=1) as threads:
                future = threads.submit(vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native, connection, plan)
                try:
                    _wait_until(lambda: (tmp_path / "entered").exists(), "task-only worker did not start")
                    if tracking != "data":
                        assert runtime.resource_snapshot()["task_admission"]["running_tasks"] == 1
                    if tracking != "tasks":
                        data = runtime.resource_snapshot()["data"]
                        assert data["tasks"] == 1
                        assert data["input_bytes"] > 0
                finally:
                    (tmp_path / "release").touch()
                result = future.result(timeout=10)
                assert [row for table in result.partition_payloads for row in table.column(0).to_pylist()] == [7]
        finally:
            (tmp_path / "release").touch()
            for resource in resources:
                resource.shutdown()
            runtime.close()
        if tracking != "data":
            assert runtime.resource_snapshot()["task_admission"]["closed"]
        if tracking != "tasks":
            assert runtime.resource_snapshot()["data"]["closed"]
            assert runtime.resource_snapshot()["data"]["tasks"] == 0


@pytest.mark.parametrize("operation", ["drain", "close"])
@pytest.mark.parametrize("tracking", ["tasks", "data", "both"])
def test_task_only_preparation_cannot_enter_after_model_drain_starts(monkeypatch, operation, tracking):
    runtime = LocalModelRuntime(
        session_id="session",
        session_config={},
        task_limit=TaskAdmissionLimits(1, 4) if tracking != "data" else None,
        track_data=tracking != "tasks",
    )
    model_gate_closed = threading.Event()
    finish_drain = threading.Event()
    original_drain = runtime._registry.drain

    def pause_after_model_gate():
        original_drain()
        model_gate_closed.set()
        assert finish_drain.wait(10)

    monkeypatch.setattr(runtime._registry, "drain", pause_after_model_gate)
    plan = _Plan({"execution_backend": "subprocess_task"})
    resources = []
    try:
        with ThreadPoolExecutor(max_workers=1) as threads:
            draining = threads.submit(getattr(runtime, operation))
            try:
                assert model_gate_closed.wait(10)
                with pytest.raises(RuntimeError, match="draining"):
                    resources.extend(runtime.prepare(plan, {}))
                assert not plan.published
                if tracking != "data":
                    assert runtime.resource_snapshot()["task_admission"]["queries"] == 0
                if tracking != "tasks":
                    assert runtime.resource_snapshot()["data"]["queries"] == 0
            finally:
                for resource in resources:
                    resource.shutdown()
                finish_drain.set()
                draining.result(timeout=10)
    finally:
        finish_drain.set()
        runtime.close()


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("neighbor_limited", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_saturated_global_task_threads_resume_or_cancel_before_admitting_another_pool(
    monkeypatch, workers, neighbor_limited, cancel
):
    from vane.execution import ref_bundle
    from vane.execution import udf_subprocess as local

    local._shutdown_global_task_runtime()
    monkeypatch.setattr(local.os, "cpu_count", lambda: workers)
    budget = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", budget)
    large, small = pa.table({"x": list(range(8192))}), pa.table({"x": [1]})
    held = ref_bundle.make_local_shm_ref_bundle_result(large)

    def make_task(index):
        def task(table):
            return pa.table({"x": [table.num_rows + index]})

        return task

    runtime = LocalModelRuntime(session_id="session", session_config={}, task_limit=TaskAdmissionLimits(1, 8))
    resources, executors = [], []
    try:
        for index in range(workers + 1):
            payload = _payload(make_task(index), execution_backend="subprocess_task", udf_worker_slots=1)
            options = {"session_config": {}}
            if index < workers or neighbor_limited:
                plan = _Plan(payload)
                resources.extend(runtime.prepare(plan, {}))
                options = plan.published[-1]["1"]
            executors.append(build_executor(payload, options))
        first, neighbor = executors[:-1], executors[-1]
        global_runtime = local._global_task_runtime()
        for index, executor in enumerate(first):
            _submit(executor, large)
            _wait_until(
                lambda: runtime.resource_snapshot()["task_admission"]["waiting_tasks"] == index + 1,
                "task did not suspend under input memory pressure",
            )
        assert global_runtime.execution_capacity.reserved_slots == workers
        assert global_runtime.stats()["active_workers"] == workers
        assert neighbor.request_task_admission(small.nbytes)
        assert neighbor.task_admission_state()["state"] == "requested"
        assert runtime.resource_snapshot()["task_admission"]["ready_tasks"] == 0
        assert not neighbor._task_futures

        if cancel:
            for executor in first:
                executor.close(kill=True)
        else:
            for ref in held[1]:
                ref.release()
        _wait_until(lambda: neighbor.task_admission_state()["available"], "global worker capacity was not returned")
        neighbor.submit(small)
        assert _wait_result(neighbor).to_pydict() == {"x": [1 + workers]}
        if not cancel:
            for index, executor in enumerate(first):
                assert _wait_result(executor).to_pydict() == {"x": [8192 + index]}
        for ref in held[1]:
            ref.release()
        _wait_until(lambda: global_runtime.execution_capacity.reserved_slots == 0, "global capacity leaked")
        snapshot = runtime.resource_snapshot()["task_admission"]
        assert snapshot["running_tasks"] == snapshot["waiting_tasks"] == snapshot["resuming_tasks"] == 0
        assert budget.snapshot()["usage_bytes"] == 0
    finally:
        for ref in held[1]:
            ref.release()
        for executor in executors:
            executor.close(kill=True)
        for resource in resources:
            resource.shutdown(kill=True)
        runtime.close(timeout=5, kill=True)
        local._shutdown_global_task_runtime()


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("track_data", [False, True])
def test_native_task_queries_resume_after_global_threads_fill_with_output_waits(monkeypatch, workers, track_data):
    from vane.execution import ref_bundle
    from vane.execution import udf_subprocess as local

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    local._shutdown_global_task_runtime()
    monkeypatch.setattr(local.os, "cpu_count", lambda: workers)
    budget = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", budget)
    held = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": list(range(8192))}))

    def make_task(index):
        def task(_table):
            return pa.table({"x": [index] * (8192 if index < workers else 1)})

        return task

    resources, futures, cursors = [], [], []
    runtime = None
    try:
        with vane.connect() as connection:
            connection.execute("SET threads=1")
            plans = []
            for index in range(workers + 1):
                cursor = connection.cursor()
                cursors.append(cursor)
                relation = cursor.sql("SELECT 1 AS x").map_batches(
                    make_task(index),
                    schema={"x": vane.sqltypes.BIGINT},
                    batch_size=1,
                    execution_backend="subprocess_task",
                )
                plans.append(
                    vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(cursor)
                )
            runtime = LocalModelRuntime(
                session_id=plans[0].session_id(),
                session_config=plans[0].session_config(),
                task_limit=TaskAdmissionLimits(1, 8),
                track_data=track_data,
            )
            for plan, cursor in zip(plans, cursors, strict=True):
                resources.extend(runtime.prepare(plan, {}, conn=cursor))
            with ThreadPoolExecutor(max_workers=workers + 1) as threads:
                try:
                    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
                    for index in range(workers):
                        futures.append(threads.submit(runner.execute_native, cursors[index], plans[index]))
                        _wait_until(
                            lambda: runtime.resource_snapshot()["task_admission"]["waiting_tasks"] == index + 1,
                            "native task did not wait for output memory",
                        )
                    futures.append(threads.submit(runner.execute_native, cursors[-1], plans[-1]))
                    _wait_until(
                        lambda: runtime.resource_snapshot()["task_admission"]["queued_tasks"] == 1,
                        "native task did not queue behind global worker capacity",
                    )
                    snapshot = runtime.resource_snapshot()["task_admission"]
                    assert snapshot["ready_tasks"] == snapshot["running_tasks"] == 0
                    if track_data:
                        data = runtime.resource_snapshot()["data"]
                        assert data["tasks"] == workers
                        assert data["input_bytes"] > 0
                    for ref in held[1]:
                        ref.release()
                    for index, future in enumerate(futures):
                        values = [
                            value
                            for table in future.result(timeout=20).partition_payloads
                            for value in table.column(0).to_pylist()
                        ]
                        assert values == [index] * (8192 if index < workers else 1)
                finally:
                    for ref in held[1]:
                        ref.release()
                    for cursor, future in zip(cursors, futures, strict=False):
                        if not future.done():
                            cursor.interrupt()
            assert local._global_task_runtime().execution_capacity.reserved_slots == 0
            assert budget.snapshot()["usage_bytes"] == 0
            if track_data:
                assert runtime.resource_snapshot()["data"]["tasks"] == 0
    finally:
        for ref in held[1]:
            ref.release()
        for resource in resources:
            resource.shutdown(kill=True)
        if runtime is not None:
            runtime.close(timeout=5, kill=True)
        for cursor in cursors:
            cursor.close()
        local._shutdown_global_task_runtime()


@pytest.mark.parametrize("track_data", [False, True])
@pytest.mark.parametrize("limits", ["none", "shared", "separate", "first", "second"])
def test_real_task_pools_alternate_under_continuous_load(monkeypatch, tmp_path, limits, track_data):
    from vane.execution import udf_subprocess as local

    local._shutdown_global_task_runtime()
    monkeypatch.setattr(local.os, "cpu_count", lambda: 1)
    entered, proceed = tmp_path / "entered", tmp_path / "proceed"

    def make_task(tag):
        def task(table):
            if table.column(0)[0].as_py() == 0:
                entered.touch()
                deadline = time.monotonic() + 30
                while not proceed.exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("first task was not released")
                    time.sleep(0.01)
            return pa.table({"x": [tag]})

        return task

    runtimes, resources, executors = [], [], []
    try:
        for index in range(2):
            payload = _payload(make_task(index), execution_backend="subprocess_task", udf_worker_slots=2)
            options = {"session_config": {}}
            limited = limits in {"shared", "separate"} or limits == ("first" if index == 0 else "second")
            if limited or track_data:
                if limits != "shared" or not runtimes:
                    runtimes.append(
                        LocalModelRuntime(
                            session_id="session",
                            session_config={},
                            task_limit=TaskAdmissionLimits(1, 8) if limited else None,
                            track_data=track_data,
                        )
                    )
                plan = _Plan(payload)
                resources.extend(runtimes[-1].prepare(plan, {}))
                options = plan.published[-1]["1"]
            executors.append(build_executor(payload, options))
        capacity = local._global_task_runtime().execution_capacity
        first_pool = next(iter(capacity._pools))
        a = next(executor for executor in executors if executor._task_pool.admission_slots is first_pool)
        b = next(executor for executor in executors if executor is not a)
        table = pa.table({"x": [1]})
        _submit(a, pa.table({"x": [0]}))
        _wait_until(entered.exists, "first subprocess did not start")
        b.request_task_admission(table.nbytes)
        a.request_task_admission(table.nbytes)
        proceed.touch()
        order = []
        for current, following in [(a, b), (b, a)] * 4:
            result = _wait_result(current)
            assert not isinstance(result, BaseException), result
            order.append(result.column(0)[0].as_py())
            _wait_until(
                lambda: any(executor.task_admission_state()["available"] for executor in executors),
                "neither task pool was admitted after completion",
            )
            assert following.task_admission_state()["available"], "busy pool bypassed the other pool's request"
            assert current.task_admission_state()["state"] == "requested"
            following.submit(table)
            following.request_task_admission(table.nbytes)
        assert order == [executors.index(a), executors.index(b)] * 4
    finally:
        proceed.touch()
        for executor in executors:
            executor.close(kill=True)
        for resource in resources:
            resource.shutdown(kill=True)
        for runtime in runtimes:
            runtime.close(timeout=5, kill=True)
        capacity = local._global_task_runtime().execution_capacity
        _wait_until(lambda: capacity.reserved_slots == 0, "global execution capacity leaked")
        local._shutdown_global_task_runtime()


@pytest.mark.parametrize("track_data", [False, True])
@pytest.mark.parametrize("limited_first", [False, True])
def test_real_cached_task_pool_shares_turns_with_limited_queries(monkeypatch, tmp_path, limited_first, track_data):
    from vane.execution import udf_subprocess as local

    local._shutdown_global_task_runtime()
    monkeypatch.setattr(local.os, "cpu_count", lambda: 1)
    entered, proceed = tmp_path / "entered", tmp_path / "proceed"

    def task(table):
        if table.column(0)[0].as_py() == 0:
            entered.touch()
            deadline = time.monotonic() + 30
            while not proceed.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("first task was not released")
                time.sleep(0.01)
        return table

    payload = _payload(task, execution_backend="subprocess_task", udf_worker_slots=2)
    runtime = LocalModelRuntime(
        session_id="session", session_config={}, task_limit=TaskAdmissionLimits(1, 8), track_data=track_data
    )
    ordinary_runtime = (
        LocalModelRuntime(session_id="session", session_config={}, track_data=True) if track_data else None
    )
    resources, executors = [], []
    try:
        plan = _Plan(payload)
        resources.extend(runtime.prepare(plan, {}))
        ordinary_options = {"session_config": {}}
        if ordinary_runtime is not None:
            ordinary_plan = _Plan(payload)
            resources.extend(ordinary_runtime.prepare(ordinary_plan, {}))
            ordinary_options = ordinary_plan.published[-1]["1"]
        ordinary = build_executor(payload, ordinary_options)
        executors.append(ordinary)
        limited = build_executor(payload, plan.published[-1]["1"])
        executors.append(limited)
        assert ordinary._task_pool is limited._task_pool
        current, following = (limited, ordinary) if limited_first else (ordinary, limited)
        _submit(current, pa.table({"x": [0]}))
        _wait_until(entered.exists, "first shared-pool subprocess did not start")
        following.request_task_admission(8)
        current.request_task_admission(8)
        proceed.touch()
        expected = 0
        for turn in range(6):
            assert _wait_result(current).to_pydict() == {"x": [expected]}
            _wait_until(
                lambda: any(executor.task_admission_state()["available"] for executor in executors),
                "shared pool did not return capacity",
            )
            assert following.task_admission_state()["available"], "the other admission path was starved"
            assert current.task_admission_state()["state"] == "requested"
            if turn < 5:
                expected = turn + 1
                following.submit(pa.table({"x": [expected]}))
                following.request_task_admission(8)
                current, following = following, current
    finally:
        proceed.touch()
        for executor in executors:
            executor.close(kill=True)
        for resource in resources:
            resource.shutdown(kill=True)
        runtime.close(timeout=5, kill=True)
        if ordinary_runtime is not None:
            ordinary_runtime.close(timeout=5, kill=True)
        _wait_until(
            lambda: local._global_task_runtime().execution_capacity.reserved_slots == 0,
            "shared task pool leaked global capacity",
        )
        local._shutdown_global_task_runtime()


@pytest.mark.parametrize("track_data", [False, True])
@pytest.mark.parametrize("failure", ["submit", "spawn", "udf"])
def test_task_failure_returns_global_worker_capacity_for_another_pool(monkeypatch, failure, track_data):
    from vane.execution import udf_subprocess as local

    local._shutdown_global_task_runtime()
    monkeypatch.setattr(local.os, "cpu_count", lambda: 1)

    def fail(*_args, **_kwargs):
        raise RuntimeError("planned task failure")

    def identity(table):
        return table

    runtime = LocalModelRuntime(
        session_id="session", session_config={}, task_limit=TaskAdmissionLimits(1, 8), track_data=track_data
    )
    resources, executors = [], []
    try:
        for function in (fail, identity):
            payload = _payload(function, execution_backend="subprocess_task", udf_worker_slots=1)
            plan = _Plan(payload)
            resources.extend(runtime.prepare(plan, {}))
            executors.append(build_executor(payload, plan.published[-1]["1"]))
        broken, healthy = executors
        with monkeypatch.context() as patch:
            if failure == "submit":
                patch.setattr(local._global_task_runtime().executor, "submit", fail)
                with pytest.raises(RuntimeError, match="planned task failure"):
                    _submit(broken, pa.table({"x": [1]}))
            else:
                if failure == "spawn":
                    patch.setattr(broken._task_pool, "_spawn_worker", fail)
                _submit(broken, pa.table({"x": [1]}))
                _wait_until(lambda: not broken._task_futures, "failed task did not finish")
                assert local._global_task_runtime().execution_capacity.reserved_slots == 0
                error = _wait_result(broken)
                assert isinstance(error, BaseException)
                assert "planned task failure" in str(error)
        assert runtime.resource_snapshot()["task_admission"]["running_tasks"] == 0
        assert _result(healthy, 3).to_pydict() == {"x": [3]}
        assert local._global_task_runtime().execution_capacity.reserved_slots == 0
        if track_data:
            data = runtime.resource_snapshot()["data"]
            assert data["tasks"] == data["retained_bytes"] == 0
    finally:
        for executor in executors:
            executor.close(kill=True)
        for resource in resources:
            resource.shutdown(kill=True)
        runtime.close(timeout=5, kill=True)
        local._shutdown_global_task_runtime()


@pytest.mark.parametrize("phase", ["input", "output"])
@pytest.mark.parametrize("cancel", [False, True])
def test_real_transport_wait_resumes_or_cancels_without_stealing_consumer_capacity(
    monkeypatch, tmp_path, phase, cancel
):
    from vane.execution import ref_bundle

    budget = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", budget)
    producer_entered = str(tmp_path / "producer-entered")
    consumer_entered, release = str(tmp_path / "consumer-entered"), str(tmp_path / "release")

    class Producer:
        def __call__(self, table):
            from pathlib import Path

            Path(producer_entered).touch()
            return pa.table({"x": list(range(8192))})

    def consumer(table):
        from pathlib import Path

        Path(consumer_entered).touch()
        deadline = time.monotonic() + 30
        while not Path(release).exists():
            if time.monotonic() > deadline:
                raise TimeoutError("consumer gate did not open")
            time.sleep(0.01)
        return pa.table({"total": [sum(table.column(0).to_pylist())]})

    source = pa.table({"x": list(range(8192))})
    held = ref_bundle.make_local_shm_ref_bundle_result(source)
    producer_payload = _payload(Producer, produce_ref_bundle_output=True, streaming_output_mode="local_shm_ref_bundle")
    consumer_payload = _payload(consumer, execution_backend="subprocess_task", udf_worker_slots=1)
    runtime = LocalModelRuntime(session_id="session", session_config={}, task_limit=TaskAdmissionLimits(1, 8))
    resources, executors, outputs = [], [], []
    try:
        runtime.register("producer", version="v1", payload=producer_payload)
        for payload, bindings in [(producer_payload, {"1": "producer"}), (consumer_payload, {})]:
            plan = _Plan(payload)
            resources.extend(runtime.prepare(plan, bindings))
            executors.append(build_executor(payload, plan.published[-1]["1"]))
        producer_executor, consumer_executor = executors
        _submit(producer_executor, source if phase == "input" else pa.table({"x": [1]}))
        _wait_until(
            lambda: runtime.resource_snapshot()["task_admission"]["waiting_tasks"] == 1,
            "producer did not yield execution capacity under memory pressure",
        )
        assert (tmp_path / "producer-entered").exists() == (phase == "output")
        assert consumer_executor.request_task_admission(source.nbytes)
        assert consumer_executor.task_admission_state()["available"]
        consumer_executor.submit_ref_bundle_with_id(1, held[1], None, held[2], held[3])
        _wait_until(lambda: (tmp_path / "consumer-entered").exists(), "consumer did not acquire the input")
        _wait_until(
            lambda: runtime.resource_snapshot()["task_admission"]["resuming_tasks"] == 1,
            "producer did not wait to reacquire execution capacity",
        )
        assert runtime.resource_snapshot()["task_admission"]["running_tasks"] == 1
        assert (tmp_path / "producer-entered").exists() == (phase == "output")
        if cancel:
            producer_executor.close(kill=True)
            _wait_until(
                lambda: runtime.resource_snapshot()["task_admission"]["waiting_tasks"] == 0,
                "cancelled wait did not finish backend cleanup",
            )
            assert runtime.resource_snapshot()["task_admission"]["running_tasks"] == 1
            assert runtime.resource_snapshot()["task_admission"]["waiting_tasks"] == 0
        (tmp_path / "release").touch()
        result = _wait_result(consumer_executor)
        assert result[2].to_pydict() == {"total": [sum(range(8192))]}
        if not cancel:
            outputs.append(_wait_result(producer_executor))
        else:
            # A fresh borrow can use the resident pool after a cancelled wait;
            # the previous invocation's capacity context must not follow it.
            plan = _Plan(producer_payload)
            resources.extend(runtime.prepare(plan, {"1": "producer"}))
            next_executor = build_executor(producer_payload, plan.published[-1]["1"])
            executors.append(next_executor)
            outputs.append(_result(next_executor, 7))
        assert outputs[0][1][0].to_table().column(0).to_pylist() == list(range(8192))
    finally:
        (tmp_path / "release").touch()
        for executor in executors:
            executor.close(kill=True)
        for resource in resources:
            resource.shutdown(kill=True)
        for output in [held, *outputs]:
            for ref in output[1]:
                ref.release()
        runtime.close(timeout=5, kill=True)
    snapshot = runtime.resource_snapshot()["task_admission"]
    assert snapshot["running_tasks"] == snapshot["waiting_tasks"] == snapshot["resuming_tasks"] == 0
    assert snapshot["queries"] == 0
    assert budget.snapshot()["usage_bytes"] == 0


@pytest.mark.parametrize("consumer_backend", ["subprocess_actor", "subprocess_task"])
def test_native_queries_make_progress_with_one_task_allowance_under_shm_pressure(
    monkeypatch, tmp_path, consumer_backend
):
    from vane.execution import ref_bundle

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    monkeypatch.setenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", "100000")
    entered, release = str(tmp_path / "entered"), str(tmp_path / "release")
    memory_waited = threading.Event()
    original_wait = ref_bundle.LocalShmBudgetManager._wait_for_capacity_locked

    def observe_wait(self, *args):
        memory_waited.set()
        return original_wait(self, *args)

    monkeypatch.setattr(ref_bundle.LocalShmBudgetManager, "_wait_for_capacity_locked", observe_wait)

    class Producer:
        def __call__(self, table):
            from pathlib import Path

            if table.column(0)[0].as_py() == 0:
                Path(entered).touch()
                deadline = time.monotonic() + 30
                while not Path(release).exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("producer gate did not open")
                    time.sleep(0.01)
            return pa.table({"value": list(range(8192))})

    def consume(table):
        return pa.table({"total": [sum(table.column(0).to_pylist())]})

    class Consumer:
        def __call__(self, table):
            return consume(table)

    with vane.connect() as connection:
        connection.execute("SET threads=1")
        cursors = [connection.cursor(), connection.cursor()]
        plans, nodes = [], []
        for index, cursor in enumerate(cursors):
            relation = (
                cursor.sql(f"SELECT {index}::BIGINT AS x")
                .map_batches(
                    Producer,
                    schema={"value": vane.sqltypes.BIGINT},
                    actor_number=1,
                    batch_size=1,
                    execution_backend="subprocess_actor",
                )
                .map_batches(
                    Consumer if consumer_backend == "subprocess_actor" else consume,
                    schema={"total": vane.sqltypes.BIGINT},
                    execution_backend=consumer_backend,
                    batch_size=8192,
                    **({"actor_number": 1} if consumer_backend == "subprocess_actor" else {}),
                )
            )
            plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(cursor)
            plans.append(plan)
            nodes.append(
                next(node for node in plan.collect_udf_nodes(conn=cursor) if node["payload"]["batch_size"] == 1)
            )
        runtime = LocalModelRuntime(
            session_id=plans[0].session_id(),
            session_config=plans[0].session_config(),
            task_limit=TaskAdmissionLimits(1, 8),
        )
        resources = []
        futures = []
        try:
            for index, (plan, node, cursor) in enumerate(zip(plans, nodes, cursors, strict=True)):
                runtime.register(f"producer-{index}", version="v1", payload=node["payload"])
                resources.extend(runtime.prepare(plan, {str(node["node_id"]): f"producer-{index}"}, conn=cursor))
            with ThreadPoolExecutor(max_workers=2) as threads:
                try:
                    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
                    futures.append(threads.submit(runner.execute_native, cursors[0], plans[0]))
                    _wait_until(lambda: (tmp_path / "entered").exists(), "first producer did not start")
                    futures.append(threads.submit(runner.execute_native, cursors[1], plans[1]))
                    _wait_until(
                        lambda: runtime.resource_snapshot()["task_admission"]["queued_tasks"] == 1,
                        "second producer did not queue before the first completed",
                    )
                    (tmp_path / "release").touch()
                    totals = [
                        value
                        for future in futures
                        for table in future.result(timeout=15).partition_payloads
                        for value in table.column(0).to_pylist()
                    ]
                    assert totals == [sum(range(8192))] * 2
                    assert memory_waited.is_set(), "regression must exercise an actual budget wait"
                finally:
                    (tmp_path / "release").touch()
                    for cursor, future in zip(cursors, futures, strict=False):
                        if not future.done():
                            cursor.interrupt()
            admission = runtime.resource_snapshot()["task_admission"]
            assert admission["running_tasks"] == admission["waiting_tasks"] == admission["resuming_tasks"] == 0
            budget = ref_bundle.local_shm_ref_budget_snapshot()
            assert budget["waiting_output_grants"] == budget["output_grant_bytes"] == 0
            assert budget["usage_bytes"] == 0
        finally:
            for resource in resources:
                resource.shutdown(kill=True)
            runtime.close(timeout=5, kill=True)
            for cursor in cursors:
                cursor.close()


def test_mixed_native_queries_share_four_task_slots_across_two_four_worker_models(monkeypatch, tmp_path):
    import sqlite3
    from contextlib import closing

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    database = str(tmp_path / "activity.sqlite")
    gates = [str(tmp_path / f"release-{stage}") for stage in range(3)]
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("CREATE TABLE activity (stage INTEGER PRIMARY KEY, running INTEGER, starts INTEGER)")
        db.executemany("INSERT INTO activity VALUES (?, 0, 0)", [(stage,) for stage in range(3)])
        db.execute("CREATE TABLE peak (maximum INTEGER)")
        db.execute("INSERT INTO peak VALUES (0)")

    def observe(table, stage):
        from pathlib import Path

        with closing(sqlite3.connect(database, timeout=30)) as db, db:
            db.execute("UPDATE activity SET running=running+1, starts=starts+1 WHERE stage=?", (stage,))
            db.execute("UPDATE peak SET maximum=MAX(maximum, (SELECT SUM(running) FROM activity))")
        try:
            deadline = time.monotonic() + 60
            while not Path(gates[stage]).exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("native admission test did not open its gate")
                time.sleep(0.01)
            return table
        finally:
            with closing(sqlite3.connect(database, timeout=30)) as db, db:
                db.execute("UPDATE activity SET running=running-1 WHERE stage=?", (stage,))

    class FirstModel:
        def __call__(self, table):
            return observe(table, 0)

    class SecondModel:
        def __call__(self, table):
            return observe(table, 0)

    class Neighbor:
        def __call__(self, table):
            return observe(table, 1)

    def make_task(expected):
        # Distinct callable payloads give these queries independent task pools,
        # so their per-pool slots cannot mask a missing runtime-wide limit.
        def task(table):
            assert table.column(0).to_pylist() == [expected]
            return observe(table, 2)

        return task

    def running(stage):
        with closing(sqlite3.connect(database)) as db:
            return db.execute("SELECT running FROM activity WHERE stage=?", (stage,)).fetchone()[0]

    with vane.connect() as connection:
        connection.execute("SET threads=1")
        cursors = [connection.cursor() for _ in range(8)]
        plans, bindings = [], []
        schema = {"x": vane.sqltypes.INTEGER}
        resources = []
        try:
            for index, cursor in enumerate(cursors):
                model_type = FirstModel if index % 2 == 0 else SecondModel
                relation = cursor.sql(f"SELECT {index}::INTEGER AS x").map_batches(
                    model_type, schema=schema, execution_backend="subprocess_actor", actor_number=4
                )
                relation = relation.map_batches(
                    Neighbor, schema=schema, execution_backend="subprocess_actor", actor_number=1
                ).map_batches(make_task(index), schema=schema, execution_backend="subprocess_task")
                plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(
                    cursor
                )
                plans.append(plan)
                node = next(
                    node
                    for node in plan.collect_udf_nodes(conn=cursor)
                    if node["payload"]["udf_name"] == model_type.__qualname__
                )
                bindings.append({str(node["node_id"]): model_type.__name__})
            with LocalModelRuntime(
                session_id=plans[0].session_id(),
                session_config=plans[0].session_config(),
                task_limit=TaskAdmissionLimits(4, 32),
            ) as runtime:
                for index, model_type in enumerate([FirstModel, SecondModel]):
                    node = next(
                        node
                        for node in plans[index].collect_udf_nodes(conn=cursors[index])
                        if str(node["node_id"]) in bindings[index]
                    )
                    runtime.register(model_type.__name__, version="v1", payload=node["payload"])
                for plan, cursor, binding in zip(plans, cursors, bindings, strict=True):
                    resources.extend(runtime.prepare(plan, binding, conn=cursor))
                pools = {
                    id(resource.pool): resource.pool for resource in resources if isinstance(resource, ModelPoolBorrow)
                }
                assert len(pools) == 2
                assert all(len(pool.worker_pids()) == 4 for pool in pools.values())
                original_pids = {pid for pool in pools.values() for pid in pool.worker_pids()}

                def execute(index):
                    result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(cursors[index], plans[index])
                    return [row for table in result.partition_payloads for row in table.column(0).to_pylist()]

                try:
                    with ThreadPoolExecutor(max_workers=8) as threads:
                        futures = [threads.submit(execute, index) for index in range(8)]
                        try:
                            for stage in range(3):

                                def stage_ready():
                                    for future in futures:
                                        if future.done():
                                            future.result()
                                    return running(stage) >= 4

                                _wait_until(stage_ready, f"stage {stage} did not fill shared capacity")
                                snapshot = runtime.resource_snapshot()["task_admission"]
                                assert snapshot["running_tasks"] == 4
                                assert sum(running(i) for i in range(3)) == 4
                                assert snapshot["queued_tasks"] <= 32
                                (tmp_path / f"release-{stage}").touch()
                        finally:
                            for stage in range(3):
                                (tmp_path / f"release-{stage}").touch()
                        assert sorted(row for future in futures for row in future.result(timeout=30)) == list(range(8))
                    assert {pid for pool in pools.values() for pid in pool.worker_pids()} == original_pids
                    with closing(sqlite3.connect(database)) as db:
                        assert db.execute("SELECT maximum FROM peak").fetchone()[0] == 4
                        assert (
                            db.execute("SELECT running, starts FROM activity ORDER BY stage").fetchall() == [(0, 8)] * 3
                        )
                finally:
                    for resource in resources:
                        resource.shutdown(kill=True)
                    resources.clear()
                snapshot = runtime.resource_snapshot()["task_admission"]
                assert snapshot["running_tasks"] == snapshot["ready_tasks"] == snapshot["queued_tasks"] == 0
                assert snapshot["queries"] == 0
        finally:
            for stage in range(3):
                (tmp_path / f"release-{stage}").touch()
            for resource in resources:
                resource.shutdown(kill=True)
            for cursor in cursors:
                cursor.close()
