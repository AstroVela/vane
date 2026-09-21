# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from types import SimpleNamespace

import pytest

import vane
from vane import pickle as vane_pickle
from vane.execution.request_admission import RequestAdmissionLimits
from vane.execution.udf_local_model import LocalModelRuntime
from vane.execution.udf_runtime_admission import TaskAdmissionLimits
from vane.execution.udf_subprocess import ensure_local_subprocess_actor_pools_for_plan


class _Identity:
    def __call__(self, table):
        return table


def _payload():
    return {
        "function_pickle": vane_pickle.dumps(_Identity),
        "execution_backend": "subprocess_actor",
        "call_mode": "map_batches",
        "actor_number": 1,
    }


@pytest.mark.parametrize("session", [None, "other"])
def test_direct_model_binding_requires_the_owning_session(monkeypatch, session):
    from vane.execution import udf_subprocess

    payload = _payload()
    constructed = []

    def unexpected_worker(*args, **kwargs):
        constructed.append(True)
        raise AssertionError("cross-session binding started a worker")

    monkeypatch.setattr(udf_subprocess, "LocalSubprocessActorPool", unexpected_worker)
    with LocalModelRuntime(session_id="owner", session_config={}) as runtime:
        model = runtime.register("model", version="v1", payload=payload)

        class Plan:
            def session_id(self):
                return session

            def collect_udf_nodes(self, conn=None):
                return [
                    {
                        "node_id": "1",
                        "payload": payload,
                        "executor_options": {"local_model_pool": model, "session_config": {}},
                    }
                ]

            def set_udf_actor_handles(self, options, conn=None):
                raise AssertionError("cross-session handles were published")

        with pytest.raises(ValueError, match="session"):
            ensure_local_subprocess_actor_pools_for_plan(Plan())
        assert not constructed
        assert runtime.resource_snapshot()["active_borrows"] == 0


@pytest.mark.parametrize("close", [False, True])
@pytest.mark.parametrize("request_limited", [False, True])
def test_registration_rechecks_drain_after_payload_serialization(monkeypatch, close, request_limited):
    from vane.execution import udf_local_model

    runtime = LocalModelRuntime(
        session_id="session",
        session_config={},
        request_limit=RequestAdmissionLimits(1, 1) if request_limited else None,
    )
    entered, resume = threading.Event(), threading.Event()
    serialize = udf_local_model._payload_bytes

    def pause(payload):
        entered.set()
        assert resume.wait(5)
        return serialize(payload)

    monkeypatch.setattr(udf_local_model, "_payload_bytes", pause)
    try:
        with ThreadPoolExecutor(max_workers=1) as threads:
            future = threads.submit(runtime.register, "late", version="v1", payload=_payload())
            try:
                assert entered.wait(5)
                if close:
                    runtime.close()
                else:
                    runtime.drain()
            finally:
                resume.set()
            with pytest.raises(RuntimeError, match="draining|closed"):
                future.result(timeout=5)
        assert not runtime._models
        assert runtime.resource_snapshot()["registered_models"] == 0
    finally:
        resume.set()
        runtime.close()


@pytest.mark.parametrize("request_limited", [False, True])
def test_close_waits_for_complete_model_publication(monkeypatch, request_limited):
    runtime = LocalModelRuntime(
        session_id="session",
        session_config={},
        request_limit=RequestAdmissionLimits(1, 1) if request_limited else None,
    )
    registered, resume, closing = threading.Event(), threading.Event(), threading.Event()
    register = runtime._registry.register

    def pause(*args, **kwargs):
        register(*args, **kwargs)
        registered.set()
        assert resume.wait(5)

    def close():
        closing.set()
        runtime.close()

    monkeypatch.setattr(runtime._registry, "register", pause)
    try:
        with ThreadPoolExecutor(max_workers=2) as threads:
            publishing = threads.submit(runtime.register, "model", version="v1", payload=_payload())
            try:
                assert registered.wait(5)
                shutdown = threads.submit(close)
                assert closing.wait(5)
                with pytest.raises(FutureTimeoutError):
                    shutdown.result(timeout=0.1)
            finally:
                resume.set()
            model = publishing.result(timeout=5)
            shutdown.result(timeout=5)
        assert runtime._models["model"] is model
        assert runtime.resource_snapshot()["closed"]
    finally:
        resume.set()
        runtime.close()


@pytest.mark.parametrize("tracked", [False, True])
def test_native_plan_can_be_prepared_for_sequential_executions(monkeypatch, tmp_path, tracked):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    marker = str(tmp_path / "initializations")

    class Model:
        def __init__(self):
            with open(marker, "a") as output:
                output.write("initialized\n")

        def __call__(self, table):
            return table

    with vane.connect() as connection:
        relation = connection.sql("SELECT 7::INTEGER AS x").map_batches(
            Model,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="subprocess_actor",
            actor_number=1,
        )
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(connection)
        node = plan.collect_udf_nodes(conn=connection)[0]
        with LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            track_data=tracked,
            task_limit=TaskAdmissionLimits(1, 2) if tracked else None,
        ) as runtime:
            runtime.register("model", version="v1", payload=node["payload"])
            for _ in range(2):
                resources = runtime.prepare(plan, {str(node["node_id"]): "model"}, conn=connection)
                assert runtime.resource_snapshot()["active_borrows"] == 1
                try:
                    result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(connection, plan)
                    assert [value for table in result.partition_payloads for value in table.column(0).to_pylist()] == [
                        7
                    ]
                finally:
                    for resource in resources:
                        resource.shutdown()
                assert runtime.resource_snapshot()["active_borrows"] == 0
            assert (tmp_path / "initializations").read_text().splitlines() == ["initialized"]


def test_preparation_validation_does_not_block_reentrant_drain(monkeypatch):
    from vane.execution import udf_local_model

    payload = _payload()
    with LocalModelRuntime(session_id="session", session_config={}) as runtime:
        runtime.register("model", version="v1", payload=payload)
        plan = SimpleNamespace(
            session_id=lambda: "session",
            session_config=lambda: {},
            collect_udf_nodes=lambda **kwargs: [{"node_id": "1", "payload": payload}],
        )
        fingerprint = udf_local_model._model_fingerprint
        with ThreadPoolExecutor(max_workers=1) as threads:

            def validate(payload):
                threads.submit(runtime.drain).result(timeout=3)
                return fingerprint(payload)

            monkeypatch.setattr(udf_local_model, "_model_fingerprint", validate)
            with pytest.raises(RuntimeError, match="draining"):
                runtime.prepare(plan, {"1": "model"})
        assert runtime.resource_snapshot()["active_borrows"] == 0
