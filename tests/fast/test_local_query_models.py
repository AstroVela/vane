# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import os
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pytest

import vane
from vane.execution import ref_bundle
from vane.execution.request_admission import RequestAdmissionLimits
from vane.execution.resources import ResourceVector
from vane.execution.udf_data_admission import DataAdmissionLimits, DataAdmissionWaitLimits
from vane.execution.udf_model_pool import ModelPoolCapacityError
from vane.execution.udf_runtime_admission import TaskAdmissionLimits


@pytest.fixture(autouse=True)
def native_environment(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    gc.collect()
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 420_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    yield manager
    gc.collect()
    assert manager.snapshot()["usage_bytes"] == 0


def _class_model(tmp_path, *, batch=False, actors=1, options=None):
    directory = str(tmp_path)

    class Model:
        def __init__(self, options):
            self.offset = options["offset"]
            self.pid = os.getpid()
            with Path(directory, "initialized").open("a") as stream:
                stream.write(f"{self.pid}\n")

        def apply(self, value):
            if value == -1:
                Path(directory, "entered").touch()
                deadline = time.monotonic() + 20
                while not Path(directory, "release").exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("model fixture gate was not released")
                    time.sleep(0.01)
            if value == -2:
                raise ValueError("model fixture failure")
            if value == -3:
                os._exit(23)
            return f"{self.pid}:{value + self.offset}"

    if batch:

        class BatchModel(Model):
            def __call__(self, values):
                return pa.array([self.apply(value) for value in values.to_pylist()])

        decorated = vane.cls.batch(actor_number=actors, return_dtype="VARCHAR", name="model_encode", batch_size=1)(
            BatchModel
        )
    else:

        class RowModel(Model):
            def __call__(self, value):
                return self.apply(value)

        decorated = vane.cls(actor_number=actors, return_dtype="VARCHAR", name="model_encode")(RowModel)
    return decorated({"offset": 10} if options is None else options)


def _runtime(connection, *, cpu=1, heap=4096, tracked=True):
    return connection.configure_local_runtime(
        request_limit=RequestAdmissionLimits(2, 4),
        resident_limit=ResourceVector(cpu=cpu, heap_bytes=heap),
        task_limit=TaskAdmissionLimits(1, 8),
        data_limit=DataAdmissionLimits(420_000, 140_000, 70_000, wait=DataAdmissionWaitLimits(8, 10)),
        track_graph=tracked,
        execution_timeout=15,
    )


def _register(runtime, model, *, name="encoder", version="v1", cpus=1, memory_bytes=4096):
    return runtime.register_model(
        name, model, version=version, parameters=["BIGINT"], cpus=cpus, memory_bytes=memory_bytes
    )


def _initializations(tmp_path):
    path = tmp_path / "initialized"
    return path.read_text().splitlines() if path.exists() else []


def _assert_idle(runtime):
    state = runtime.resource_snapshot()
    assert state["active_borrows"] == 0
    assert state["request_admission"]["active_requests"] == 0
    assert state["request_admission"]["queued_requests"] == 0
    if state.get("data") is not None:
        assert state["data"]["usage_bytes"] == 0


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("prewarm", [False, True])
@pytest.mark.parametrize("budgets", [False, True])
def test_sql_and_rebuilt_relations_reuse_explicit_model(tmp_path, batch, prewarm, budgets):
    options = {"offset": 10}
    with vane.connect(config={"threads": 2}) as connection:
        runtime = (
            _runtime(connection)
            if budgets
            else connection.configure_local_runtime(
                request_limit=RequestAdmissionLimits(2, 4), resident_limit=ResourceVector(cpu=1, heap_bytes=4096)
            )
        )
        model = _register(runtime, _class_model(tmp_path, batch=batch, options=options))
        assert model.name == "encoder" and model.version == "v1"
        assert _initializations(tmp_path) == []
        options["offset"] = 1000
        if prewarm:
            model.prewarm()
            model.prewarm()
            assert len(_initializations(tmp_path)) == 1
        vane.attach_function(model, "resident_encode", connection=connection)
        rows = []
        with connection.cursor() as cursor:
            for index in range(3):
                # Passthrough layout, column aliases and the SQL alias vary;
                # model arguments and initialization remain unchanged.
                result = cursor.execute(
                    "SELECT true, resident_encode(i), 'extra' FROM range(2) r(i) ORDER BY i"
                ).fetchall()
                rows.extend(row[1] for row in result)
                relation = cursor.sql("SELECT i::INTEGER AS x, [i] AS extra FROM range(2) r(i)")
                result = relation.project(model(vane.col("x")).alias(f"result_{index}")).fetchall()
                rows.extend(row[0] for row in result)
                _assert_idle(runtime)
        pids = {value.split(":")[0] for value in rows}
        assert pids == set(_initializations(tmp_path)) and len(pids) == 1
        assert [int(value.split(":")[1]) for value in rows] == [10, 11] * 6
        assert runtime.resource_snapshot()["reserved_resources"] == ResourceVector(cpu=1, heap_bytes=4096).to_dict()
    assert runtime.resource_snapshot()["closed"]
    assert runtime.resource_snapshot()["reserved_resources"] == ResourceVector().to_dict()
    with pytest.raises(RuntimeError, match="drain|closed"):
        model.prewarm()


@pytest.mark.parametrize("batch", [False, True])
def test_concurrent_cursors_borrow_one_registered_pool(tmp_path, batch):
    with vane.connect(config={"threads": 2}) as connection:
        runtime = _runtime(connection)
        model = _register(runtime, _class_model(tmp_path, batch=batch))
        vane.attach_function(model, connection=connection)
        model.prewarm()
        with connection.cursor() as first, connection.cursor() as second, ThreadPoolExecutor(2) as clients:

            def query(cursor, value):
                return cursor.execute("SELECT model_encode(?)", [value]).fetchall()[0][0]

            for _ in range(3):
                futures = [clients.submit(query, first, 1), clients.submit(query, second, 2)]
                values = [future.result(timeout=10) for future in futures]
                assert [int(value.split(":")[1]) for value in values] == [11, 12]
                assert len({value.split(":")[0] for value in values}) == 1
                _assert_idle(runtime)
        assert len(_initializations(tmp_path)) == 1


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("entry", ["sql", "relation"])
def test_registration_cannot_be_borrowed_by_another_session(tmp_path, configured, entry):
    with vane.connect() as owner, vane.connect() as other:
        runtime = _runtime(owner)
        model = _register(runtime, _class_model(tmp_path))
        if configured:
            _runtime(other)
        if entry == "sql":
            vane.attach_function(model, connection=other)
            relation = other.sql("SELECT model_encode(1)")
        else:
            relation = other.sql("SELECT 1 AS x").project(model(vane.col("x")))
        with pytest.raises(Exception, match="different Vane session|owning configured local runtime"):
            relation.fetchall()
        assert _initializations(tmp_path) == []
        assert other.sql("SELECT 7").fetchall() == [(7,)]
        _assert_idle(runtime)


@pytest.mark.parametrize("runner", ["local", "ray"])
def test_registered_models_reject_other_runners_before_worker_start(tmp_path, monkeypatch, runner):
    with vane.connect() as owner:
        runtime = _runtime(owner)
        model = _register(runtime, _class_model(tmp_path))
        monkeypatch.setenv("VANE_RUNNER", runner)
        with vane.connect() as other:
            relation = other.sql("SELECT 1 AS x").project(model(vane.col("x")))
            with pytest.raises(Exception, match="registered local models require VANE_RUNNER=local-fast"):
                vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "foreign-runner").to_physical_plan(other)
        assert _initializations(tmp_path) == []


def test_retained_model_does_not_keep_its_owner_connection_alive(tmp_path):
    connection = vane.connect()
    runtime = _runtime(connection)
    model = _register(runtime, _class_model(tmp_path))
    model.prewarm()
    owner = weakref.ref(connection)
    del connection
    gc.collect()
    assert owner() is None
    assert runtime.resource_snapshot()["closed"]
    assert runtime.resource_snapshot()["reserved_resources"] == ResourceVector().to_dict()
    with pytest.raises(RuntimeError, match="closed|drain"):
        model.prewarm()


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("fail_initialization", [False, True])
def test_prewarm_retains_shutdown_owner_until_initialization_finishes(tmp_path, batch, fail_initialization):
    directory = str(tmp_path)

    class SlowInit:
        def __init__(self):
            Path(directory, "initialized").write_text(str(os.getpid()))
            Path(directory, "entered").touch()
            deadline = time.monotonic() + 20
            while not Path(directory, "release").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("prewarm fixture gate was not released")
                time.sleep(0.01)
            if fail_initialization:
                raise ValueError("prewarm initialization fixture failure")

        def __call__(self, value):
            return value

    decorate = vane.cls.batch if batch else vane.cls
    definition = decorate(actor_number=1, return_dtype="BIGINT")(SlowInit)()
    connection = vane.connect()
    runtime = _runtime(connection)
    model = _register(runtime, definition)
    owner = weakref.ref(connection)
    runtime_owner = weakref.ref(runtime)
    registry = model._model._registry
    try:
        with ThreadPoolExecutor(max_workers=1) as clients:
            prewarm = clients.submit(model.prewarm)
            try:
                deadline = time.monotonic() + 10
                while not (tmp_path / "entered").exists():
                    assert time.monotonic() < deadline, "model initialization did not start"
                    time.sleep(0.01)
                pid = int((tmp_path / "initialized").read_text())
                del connection, runtime
                gc.collect()
                assert owner() is not None
                assert runtime_owner() is not None
                state = registry.resource_snapshot()
                assert not state["draining"]
                assert state["initializing_resources"] == ResourceVector(cpu=1, heap_bytes=4096).to_dict()
            finally:
                (tmp_path / "release").touch()

            # Keep the future (and any error traceback) alive to verify that
            # it does not retain the connection after the prewarm attempt.
            error = prewarm.exception(timeout=10)
            if fail_initialization:
                assert error is not None and "prewarm initialization fixture failure" in str(error)
            else:
                assert error is None

        gc.collect()
        assert owner() is None
        assert runtime_owner() is None
        state = registry.resource_snapshot()
        assert state["closed"]
        assert state["active_borrows"] == 0
        assert state["reserved_resources"] == ResourceVector().to_dict()
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        (tmp_path / "release").touch()
        registry.close(timeout=10, kill=True)


@pytest.mark.parametrize("resource", ["cpu", "heap"])
def test_registered_model_obeys_resident_capacity(tmp_path, resource):
    with vane.connect() as connection:
        runtime = _runtime(connection, cpu=0 if resource == "cpu" else 1, heap=2048)
        with pytest.raises(ModelPoolCapacityError):
            _register(runtime, _class_model(tmp_path), cpus=1e-13 if resource == "cpu" else 1)
        assert _initializations(tmp_path) == []
        assert runtime.resource_snapshot()["reserved_resources"] == ResourceVector().to_dict()


def test_two_explicit_registrations_do_not_share_by_callable_equality(tmp_path):
    with vane.connect() as connection:
        runtime = _runtime(connection)
        definition = _class_model(tmp_path)
        first = _register(runtime, definition)
        second = _register(runtime, definition, name="encoder-v2", version="v2")
        first.prewarm()
        with pytest.raises(ModelPoolCapacityError):
            second.prewarm()
        assert len(_initializations(tmp_path)) == 1
        with pytest.raises(ValueError, match="already registered"):
            _register(runtime, definition, version="changed")
        runtime.drain()
        with pytest.raises(RuntimeError, match="drain"):
            second.prewarm()
        with pytest.raises(RuntimeError, match="drain"):
            _register(runtime, definition, name="late")


@pytest.mark.parametrize("failure", [-2, -3])
def test_new_request_recovers_after_model_failure_without_replaying(tmp_path, failure):
    with vane.connect() as connection:
        runtime = _runtime(connection)
        model = _register(runtime, _class_model(tmp_path))
        vane.attach_function(model, connection=connection)
        model.prewarm()
        with pytest.raises(Exception):
            connection.execute("SELECT model_encode(?)", [failure]).fetchall()
        _assert_idle(runtime)
        result = connection.execute("SELECT model_encode(2)").fetchall()[0][0]
        assert int(result.split(":")[1]) == 12
        assert len(_initializations(tmp_path)) == 2
        _assert_idle(runtime)


@pytest.mark.parametrize("batch", [False, True])
def test_cancelling_one_borrower_preserves_the_shared_registration(tmp_path, batch):
    with vane.connect(config={"threads": 2}) as connection:
        runtime = _runtime(connection)
        model = _register(runtime, _class_model(tmp_path, batch=batch))
        vane.attach_function(model, connection=connection)
        with connection.cursor() as first, connection.cursor() as second, ThreadPoolExecutor(2) as clients:
            blocked = clients.submit(lambda: first.execute("SELECT model_encode(-1)").fetchall())
            queued = None
            try:
                deadline = time.monotonic() + 10
                while not (tmp_path / "entered").exists():
                    if blocked.done():
                        blocked.result()
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                queued = clients.submit(lambda: second.execute("SELECT model_encode(2)").fetchall())
                deadline = time.monotonic() + 10
                while runtime.resource_snapshot()["active_borrows"] < 2:
                    if queued.done():
                        queued.result()
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                first.interrupt()
                with pytest.raises(Exception, match="cancel|interrupt"):
                    blocked.result(timeout=10)
                assert int(queued.result(timeout=10)[0][0].split(":")[1]) == 12
            finally:
                (tmp_path / "release").touch()
                if not blocked.done():
                    first.interrupt()
                if queued is not None and not queued.done():
                    second.interrupt()
            _assert_idle(runtime)
            assert runtime.resource_snapshot()["reserved_models"] == 1
            assert int(first.execute("SELECT model_encode(3)").fetchall()[0][0].split(":")[1]) == 13


def test_unregistered_class_keeps_query_owned_lifetime(tmp_path):
    other_directory = tmp_path / "other"
    other_directory.mkdir()
    with vane.connect() as connection:
        runtime = _runtime(connection)
        model = _register(runtime, _class_model(tmp_path))
        ordinary = _class_model(other_directory)
        vane.attach_function(model, "resident", connection=connection)
        vane.attach_function(ordinary, "ordinary", parameters=["BIGINT"], connection=connection)
        for _ in range(2):
            result = connection.execute("SELECT resident(1), ordinary(2)").fetchall()[0]
            assert [int(value.split(":")[1]) for value in result] == [11, 12]
            _assert_idle(runtime)
        assert len(_initializations(tmp_path)) == 1
        assert len(_initializations(other_directory)) == 2


def test_batch_model_preserves_multiple_inputs_and_struct_expansion(tmp_path):
    directory = str(tmp_path)

    @vane.cls.batch(
        actor_number=1,
        name="resident_combine",
        batch_size=2,
        unnest=True,
        return_dtype=pa.struct([("value", pa.string()), ("pid", pa.int64())]),
    )
    class Combine:
        def __init__(self):
            with Path(directory, "initialized").open("a") as stream:
                stream.write(f"{os.getpid()}\n")

        def __call__(self, numbers, labels):
            return pa.StructArray.from_arrays(
                [
                    pa.array([f"{label}:{number}" for number, label in zip(numbers.to_pylist(), labels.to_pylist())]),
                    pa.array([os.getpid()] * len(numbers)),
                ],
                names=["value", "pid"],
            )

    with vane.connect() as connection:
        runtime = _runtime(connection)
        model = runtime.register_model("resident_combine", Combine(), version="v1", parameters=["BIGINT", "VARCHAR"])
        vane.attach_function(model, connection=connection)
        with pytest.raises(ValueError, match="2 positional inputs"):
            model(vane.col("i"))
        sql_value = connection.execute("SELECT resident_combine(4, 'sql')").fetchall()[0][0]
        relation = connection.sql("SELECT i::INTEGER AS i, 'rel' AS label FROM range(3) r(i)")
        values = relation.project(model(vane.col("i"), vane.col("label"))).fetchall()
        assert sql_value == {"value": "sql:4", "pid": int(_initializations(tmp_path)[0])}
        assert sorted(values) == [(f"rel:{i}", sql_value["pid"]) for i in range(3)]
        assert len(_initializations(tmp_path)) == 1
        _assert_idle(runtime)


def test_resident_reservation_multiplies_per_actor_declarations(tmp_path):
    with vane.connect() as connection:
        runtime = _runtime(connection, cpu=2, heap=8192)
        model = _register(runtime, _class_model(tmp_path, actors=2))
        model.prewarm()
        assert len(set(_initializations(tmp_path))) == 2
        assert runtime.resource_snapshot()["reserved_resources"] == ResourceVector(cpu=2, heap_bytes=8192).to_dict()
        vane.attach_function(model, connection=connection)
        assert int(connection.execute("SELECT model_encode(1)").fetchall()[0][0].split(":")[1]) == 11
        _assert_idle(runtime)


def test_model_initialization_uses_captured_session_environment(monkeypatch):
    variable = "AWS_VANE_REGISTERED_MODEL_TEST"
    monkeypatch.setenv(variable, "captured")

    @vane.cls(actor_number=1, return_dtype="VARCHAR")
    class Environment:
        def __init__(self):
            self.value = os.environ[variable]

        def __call__(self, value):
            return self.value

    with vane.connect() as connection:
        runtime = _runtime(connection)
        monkeypatch.setenv(variable, "later")
        model = _register(runtime, Environment())
        vane.attach_function(model, "environment", connection=connection)
        for _ in range(2):
            assert connection.execute("SELECT environment(1)").fetchall() == [("captured",)]
            _assert_idle(runtime)


@pytest.mark.parametrize("captured", [None, "1000000"])
def test_rebuilt_models_use_captured_batch_settings(tmp_path, monkeypatch, captured):
    variable = "VANE_UDF_TARGET_MAX_BATCH_BYTES"
    if captured is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, captured)
    with vane.connect() as connection:
        runtime = _runtime(connection)
        model = _register(runtime, _class_model(tmp_path))
        for value in ("2000000", "invalid-live-setting"):
            monkeypatch.setenv(variable, value)
            vane.attach_function(model, "model_encode", connection=connection, replace=True)
            assert int(connection.execute("SELECT model_encode(1)").fetchall()[0][0].split(":")[1]) == 11
            relation = connection.sql("SELECT 2 AS x").project(model(vane.col("x")))
            assert int(relation.fetchall()[0][0].split(":")[1]) == 12
            _assert_idle(runtime)
        assert len(_initializations(tmp_path)) == 1


@pytest.mark.parametrize("tracked", [False, True])
def test_registered_model_cannot_bypass_runtime_output_budget(tracked):
    @vane.cls(actor_number=1, return_dtype="VARCHAR")
    class Oversized:
        def __call__(self, value):
            return "x" * 10_000

    with vane.connect() as connection:
        runtime = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(1, 1),
            data_limit=DataAdmissionLimits(16_384, 2_048, 2_048),
            track_graph=tracked,
        )
        model = _register(runtime, Oversized())
        vane.attach_function(model, "oversized", connection=connection)
        for sql in (
            "SELECT oversized(1)",
            "SELECT i, (SELECT oversized(j) FROM range(2) t(j) WHERE j < r.i LIMIT 1) FROM range(2) r(i)",
        ):
            with pytest.raises(Exception, match="output batch exceeds data limit"):
                connection.execute(sql).fetchall()
            _assert_idle(runtime)


def test_registration_retains_cached_initialization_failure(tmp_path):
    directory = str(tmp_path)

    @vane.cls(actor_number=1, return_dtype="BIGINT")
    class Broken:
        def __init__(self):
            with Path(directory, "initialized").open("a") as stream:
                stream.write("attempt\n")
            raise ValueError("initialization fixture failure")

        def __call__(self, value):
            return value

    with vane.connect() as connection:
        runtime = _runtime(connection)
        model = _register(runtime, Broken())
        vane.attach_function(model, "broken", connection=connection)
        for _ in range(2):
            with pytest.raises(Exception, match="initialization fixture failure"):
                connection.execute("SELECT broken(1)").fetchall()
            _assert_idle(runtime)
        assert _initializations(tmp_path) == ["attempt"]
        assert connection.execute("SELECT 42").fetchall() == [(42,)]


@pytest.mark.parametrize("option", [{"cpus": 0}, {"cpus": True}, {"cpus": float("nan")}, {"memory_bytes": 1.5}])
def test_registration_rejects_invalid_resource_declarations(tmp_path, option):
    with vane.connect() as connection:
        runtime = _runtime(connection)
        with pytest.raises(ValueError):
            _register(runtime, _class_model(tmp_path), **option)
        assert runtime.resource_snapshot()["registered_models"] == 0
        assert _initializations(tmp_path) == []


@pytest.mark.parametrize("option", [{"parameters": ["INTEGER"]}, {"batch_size": 4}, {"actor_number": 2}])
def test_attachment_cannot_override_frozen_model_contract(tmp_path, option):
    with vane.connect() as connection:
        runtime = _runtime(connection)
        model = _register(runtime, _class_model(tmp_path))
        with pytest.raises(ValueError, match="cannot be overridden"):
            vane.attach_function(model, connection=connection, **option)
        assert _initializations(tmp_path) == []


@pytest.mark.parametrize("operation", ["prewarm", "register", "expression", "attach"])
def test_model_operations_obey_the_callback_reentry_guard(tmp_path, operation):
    with vane.connect() as connection:
        runtime = _runtime(connection)
        definition = _class_model(tmp_path)
        model = _register(runtime, definition)
        called = []

        class Value:
            def __str__(self):
                called.append(True)
                with pytest.raises(vane.InvalidInputException, match="Python input callback"):
                    if operation == "prewarm":
                        model.prewarm()
                    elif operation == "register":
                        _register(runtime, definition, name="nested")
                    elif operation == "expression":
                        model(vane.col("x"))
                    else:
                        vane.attach_function(model, connection=connection)
                return "ok"

        import numpy as np

        connection.register("items", {"x": np.array([Value()], dtype=object)})
        relation = connection.sql("SELECT x FROM items")
        assert relation.fetchall() == [("ok",)]
        assert called
        assert _initializations(tmp_path) == []
