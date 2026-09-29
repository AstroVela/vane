# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Public native GPU admission contracts; fake UUIDs require no CUDA hardware."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from local_gpu_helpers import (
    DEVICES,
    assert_idle,
    configure,
    execution_demand,
    gpu_pool_snapshot,
    model_definition,
    register,
    run,
    wait_for,
)
from local_gpu_helpers import query_gpu_environment as query_gpu_environment

import vane
from vane.execution import udf_subprocess
from vane.execution.resources import ResourceVector
from vane.execution.udf import build_executor
from vane.execution.udf_model_pool import ModelPoolCapacityError, ModelPoolResourceBusy

pytestmark = pytest.mark.usefixtures("query_gpu_environment")


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("tracked", [False, True])
def test_registered_gpu_sql_and_rebuilt_relations_share_resident_pool(tmp_path, monkeypatch, batch, limited, tracked):
    devices = list(DEVICES[:1])
    with vane.connect(config={"threads": 2}) as connection:
        runtime = configure(
            connection, devices, limited=limited, **({} if tracked else {"data_limit": None, "track_graph": False})
        )
        model = register(runtime, model_definition(tmp_path, batch=batch), devices)
        devices[0] = DEVICES[1]
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "later-parent-setting")
        state = runtime.resource_snapshot()
        assert state["gpu"]["devices"] == list(DEVICES[:1])
        assert state["gpu"]["models"][0]["pools"] == []
        assert not (tmp_path / "initialized").exists()
        model.prewarm()
        vane.attach_function(model, connection=connection)
        values = []
        with connection.cursor() as cursor:
            for _ in range(2):
                values.append(run(cursor, 7))
                relation = cursor.sql("SELECT 9 AS x").project(model(vane.col("x")))
                values.append(json.loads(relation.fetchone()[0]))
        assert [value["value"] for value in values] == [84, 100] * 2
        assert {value["device"] for value in values} == {DEVICES[0]}
        assert len({value["pid"] for value in values}) == 1
        assert len((tmp_path / "initialized").read_text().splitlines()) == 1
        assert gpu_pool_snapshot(runtime)["workers"][0]["pid"] == values[0]["pid"]
        assert_idle(runtime)
    assert runtime.resource_snapshot()["closed"]
    assert runtime.resource_snapshot()["reserved_resources"] == ResourceVector().to_dict()
    assert runtime.resource_snapshot()["gpu"]["models"][0]["pools"] == []


@pytest.mark.parametrize("devices", [[], "0", ["0"], ["GPU-aaaa"], [DEVICES[0], DEVICES[0].upper()]])
def test_public_inventory_rejects_ambiguous_devices(devices):
    with vane.connect() as connection:
        with pytest.raises((TypeError, ValueError), match="GPU"):
            configure(connection, devices)
        assert connection.sql("SELECT 1").fetchall() == [(1,)]


@pytest.mark.parametrize("case", ["inventory", "assignment", "foreign", "replicas", "cpu", "fractional", "zero"])
def test_invalid_model_placement_starts_no_workers(tmp_path, case):
    with vane.connect() as connection:
        runtime = configure(
            connection,
            gpu_devices=None if case == "inventory" else DEVICES[:1],
            resident_limit=ResourceVector(cpu=2, gpu=0 if case in {"inventory", "zero"} else 1, heap_bytes=4096),
        )
        definition = model_definition(
            tmp_path,
            gpus=0 if case == "cpu" else 0.5 if case == "fractional" else 1,
            actors=2 if case == "replicas" else 1,
        )
        with pytest.raises((ValueError, ModelPoolCapacityError), match="GPU|gpu|replica"):
            register(
                runtime, definition, None if case == "assignment" else DEVICES[1:] if case == "foreign" else DEVICES[:1]
            )
        assert not (tmp_path / "initialized").exists()
        assert runtime.resource_snapshot()["reserved_resources"] == ResourceVector().to_dict()


def test_inventory_required_for_positive_resident_gpu_limit():
    with vane.connect() as connection:
        with pytest.raises(ValueError, match="inventory"):
            configure(connection, gpu_devices=None)


def test_gpu_model_can_declare_zero_cpu_within_a_gpu_resident_limit(tmp_path):
    with vane.connect() as connection:
        runtime = configure(connection, resident_limit=ResourceVector(gpu=1))
        model = runtime.register_model(
            "encoder",
            model_definition(tmp_path),
            version="v1",
            parameters=["BIGINT"],
            cpus=0,
            gpu_devices=DEVICES[:1],
        )
        vane.attach_function(model, connection=connection)
        assert run(connection, 7)["value"] == 84
        assert runtime.resource_snapshot()["reserved_resources"] == ResourceVector(gpu=1).to_dict()
        assert_idle(runtime)


@pytest.mark.parametrize("entry", ["query", "close"])
def test_inventory_conversion_cannot_reenter_connection_apis(entry):
    with vane.connect() as connection, vane.connect() as other:

        class Inventory:
            def __iter__(self):
                if entry == "query":
                    other.execute("SELECT 7")
                else:
                    other.close()
                yield DEVICES[0]

        with pytest.raises(vane.InvalidInputException, match="reentrantly.*callback"):
            configure(connection, gpu_devices=Inventory())
        assert other.execute("SELECT 7").fetchall() == [(7,)]
        configure(connection)


def test_model_assignment_conversion_cannot_publish_across_drain(tmp_path):
    with vane.connect() as connection:
        runtime = configure(connection)

        class Assignment:
            def __iter__(self):
                runtime.drain()
                yield DEVICES[0]

        with pytest.raises(RuntimeError, match="drain"):
            register(runtime, model_definition(tmp_path), Assignment())
        assert runtime.resource_snapshot()["registered_models"] == 0
        assert not (tmp_path / "initialized").exists()


def test_gpu_model_conflict_is_retryable_and_cpu_model_uses_same_registry(tmp_path):
    with vane.connect() as connection:
        runtime = configure(connection)
        definition = model_definition(tmp_path)
        first = register(runtime, definition)
        other = register(runtime, definition, name="other")
        cpu = runtime.register_model(
            "cpu", model_definition(tmp_path, gpus=0), version="v1", parameters=["BIGINT"], memory_bytes=1024
        )
        first.prewarm()
        cpu.prewarm()
        for _ in range(2):
            with pytest.raises(ModelPoolResourceBusy):
                other.prewarm()
        state = runtime.resource_snapshot()
        assert state["resident_resources"] == ResourceVector(cpu=2, gpu=1, heap_bytes=2048).to_dict()
        assert state["reserved_models"] == 2
        assert len((tmp_path / "initialized").read_text().splitlines()) == 2
        assert state["exclusive_resources"][f"cuda:{DEVICES[0]}"]["model"] == "encoder"
        runtime.drain()
        with pytest.raises(RuntimeError, match="drain"):
            first.prewarm()
        with pytest.raises(RuntimeError, match="drain"):
            other.prewarm()


@pytest.mark.parametrize("limited", [False, True])
def test_registered_gpu_producer_and_cpu_task_share_native_byte_budget(tmp_path, limited):
    @vane.func(return_dtype="BIGINT")
    def host_value(value):
        return json.loads(value)["value"]

    with vane.connect(config={"threads": 2}) as connection:
        runtime = configure(connection, limited=limited)
        model = register(runtime, model_definition(tmp_path))
        vane.attach_function(model, connection=connection)
        vane.attach_function(host_value, "host_value", parameters=["VARCHAR"], connection=connection)
        result = connection.execute("SELECT host_value(gpu_encode(i)) FROM range(3) r(i) ORDER BY i").fetchall()
        assert result == [(28,), (36,), (44,)]
        assert_idle(runtime)


@pytest.mark.parametrize("limited", [False, True])
def test_public_gpu_execution_waits_for_device_and_cancellation_replaces_on_same_device(tmp_path, limited):
    with vane.connect(config={"threads": 2}) as connection:
        runtime = configure(connection, limited=limited)
        model = register(runtime, model_definition(tmp_path))
        vane.attach_function(model, connection=connection)
        model.prewarm()
        original = gpu_pool_snapshot(runtime)["workers"][0]
        with connection.cursor() as first, connection.cursor() as second, ThreadPoolExecutor(2) as clients:
            blocked = clients.submit(run, first, -1)
            queued = None
            try:
                wait_for(lambda: (tmp_path / "entered-1").exists(), blocked)
                queued = clients.submit(run, second, 2)
                wait_for(lambda: runtime.resource_snapshot()["active_borrows"] == 2, queued)
                assert execution_demand(runtime) == 1
                pool = gpu_pool_snapshot(runtime)
                assert pool["devices"][0]["executions"][0]["pid"] == original["pid"]
                assert not queued.done()
                first.interrupt()
                with pytest.raises(Exception, match="cancel|interrupt"):
                    blocked.result(timeout=20)
                result = queued.result(timeout=20)
                assert result["value"] == 44 and result["device"] == DEVICES[0]
                assert result["pid"] != original["pid"]
                assert gpu_pool_snapshot(runtime)["workers"][0]["generation"] == 1
            finally:
                (tmp_path / "release-1").touch()
                first.interrupt()
                if queued is not None and not queued.done():
                    second.interrupt()
        assert_idle(runtime)
        assert runtime.resource_snapshot()["worker_failures"]["cancelled_workers"] == 1


def test_fixed_replicas_execute_on_distinct_devices(tmp_path):
    with vane.connect(config={"threads": 2}) as connection:
        runtime = configure(connection, DEVICES, limited=False)
        model = register(runtime, model_definition(tmp_path, actors=2), DEVICES)
        vane.attach_function(model, connection=connection)
        with connection.cursor() as first, connection.cursor() as second, ThreadPoolExecutor(2) as clients:
            futures = [clients.submit(run, cursor, value) for cursor, value in ((first, -1), (second, -2))]
            try:
                wait_for(lambda: all((tmp_path / f"entered{n}").exists() for n in (-1, -2)), *futures)
                assert execution_demand(runtime) == 2
                assert runtime.resource_snapshot()["reserved_resources"]["gpu"] == 2
            finally:
                for value in (-1, -2):
                    (tmp_path / f"release{value}").touch()
            rows = [future.result(timeout=20) for future in futures]
            assert {row["device"] for row in rows} == set(DEVICES)
            assert len({row["pid"] for row in rows}) == 2
        assert_idle(runtime, resident=2)


@pytest.mark.parametrize("configured", [False, True])
def test_registered_gpu_model_rejects_foreign_session_before_starting(tmp_path, configured):
    with vane.connect() as owner, vane.connect() as other:
        runtime = configure(owner)
        model = register(runtime, model_definition(tmp_path))
        if configured:
            configure(other)
        relation = other.sql("SELECT 1 AS x").project(model(vane.col("x")))
        with pytest.raises(Exception, match="different Vane session|owning configured local runtime"):
            relation.fetchall()
        assert not (tmp_path / "initialized").exists()


def test_executor_rejects_gpu_pool_without_resident_registration(tmp_path):
    with vane.connect() as connection:
        runtime = configure(connection)
        model = register(runtime, model_definition(tmp_path))
        with model._model.acquire() as borrow:
            pool = borrow.pool
            options = {"local_actor_pool": pool, "session_config": dict(model._model._session_config)}
            with pytest.raises(ValueError, match="GPU resources"):
                build_executor(pool.payload, options)
            options["local_model_pool"] = model._model
            with pytest.raises(ValueError, match="does not match"):
                build_executor({**pool.payload, "gpus": 2}, options)
            options["local_actor_pool"] = object()
            with pytest.raises(ValueError, match="resident device pool"):
                build_executor(pool.payload, options)


def test_public_snapshot_keeps_failed_shutdown_owner_without_borrowing(tmp_path, monkeypatch):
    connection = vane.connect()
    runtime = configure(connection)
    model = register(runtime, model_definition(tmp_path))
    model.prewarm()
    original_close = udf_subprocess._SingleSubprocessExecutor.close
    fail = True

    def close(worker, kill=False):
        if fail and worker._local_gpu_assignment is not None:
            raise RuntimeError("injected GPU cleanup failure")
        return original_close(worker, kill=kill)

    monkeypatch.setattr(udf_subprocess._SingleSubprocessExecutor, "close", close)
    stop = threading.Event()
    errors = []

    def sample():
        while not stop.wait(0.001):
            try:
                runtime.resource_snapshot()
            except BaseException as error:
                errors.append(error)

    thread = threading.Thread(target=sample)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match="cleanup|shutdown"):
            runtime.close(kill=True)
        state = runtime.resource_snapshot()
        assert state["reserved_resources"]["gpu"] == 1
        assert state["active_borrows"] == 0
        assert not gpu_pool_snapshot(runtime)["workers"][0]["cleanup_finished"]
        fail = False
        runtime.close(timeout=10, kill=True)
        assert runtime.resource_snapshot()["gpu"]["models"][0]["pools"] == []
        assert_idle(runtime, resident=0)
    finally:
        fail = False
        stop.set()
        thread.join(timeout=5)
        connection.close()
    assert not thread.is_alive() and not errors
