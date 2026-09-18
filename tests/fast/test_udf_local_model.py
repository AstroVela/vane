# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
from vane.execution.udf_model_pool import ModelPoolCapacityError


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
