# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Submission identities, fixed placement and capacity validation without Ray."""

import json
from dataclasses import replace

import pytest

import vane
from vane.execution.compiler import FragmentCompileOptions
from vane.execution.pipelined_plan import DirectTicket, placement
from vane.execution.resource_demand import MemoryDemand, ResourceDemand
from vane.execution.submission import prepare_ray_query


def test_fixed_routes_fence_query_attempt_epochs_and_schema():
    with vane.connect(backend="local") as connection:
        spec = prepare_ray_query(
            connection,
            "select range from range(10)",
            query_id="routing-query",
            options=vane.QueryExecutionOptions(vane.RayExecution(), 5, 30, 30),
            resources=ResourceDemand(1, 8, MemoryDemand(1 << 26, 1 << 20, 1 << 20, 1 << 20), 8),
            compile_options=FragmentCompileOptions(2, (0,)),
        )
    tasks, routes = placement(spec, ["epoch-a", "epoch-b"], "result-epoch")
    assert set(tasks.values()) == {0, 1}
    assert len({r["ticket"] for r in routes}) == len(routes)
    for route in routes:
        ticket = json.loads(route["ticket"])
        assert ticket["query_id"] == "routing-query"
        assert ticket["attempt_id"] == "0"
        assert ticket["producer_epoch"] == ["epoch-a", "epoch-b"][route["source_worker"]]
        assert ticket["schema_id"]
        assert ticket["capability"]
        assert ticket["partition"] == route["partition"]
    with pytest.raises(ValueError, match="epochs"):
        placement(spec, ["same", "same"], "result")


@pytest.mark.parametrize(
    "field,value",
    [
        ("worker_count", 0),
        ("worker_count", 257),
        ("partitions", False),
        ("io_concurrency", 0),
        ("io_concurrency", 4097),
        ("task_contexts_per_worker", -1),
        ("operator_memory_bytes", 1),
        ("exchange_buffer_bytes", 0),
        ("staging_buffer_bytes", 0),
    ],
)
def test_invalid_worker_capacities(field, value):
    with pytest.raises(ValueError):
        replace(vane.RayResources(), **{field: value})


def test_ticket_capability_is_not_in_repr():
    ticket = DirectTicket("query", "0", "producer-epoch", "consumer-epoch", "edge", "a", "b", 0, "schema", "secret")
    assert "secret" not in repr(ticket)
    assert json.loads(ticket.encode())["capability"] == "secret"
    with pytest.raises(ValueError):
        replace(ticket, query_id="").encode()


def test_ray_api_explicit_configuration():
    with vane.Runtime() as application, application.connect() as connection:
        assert connection.query_runtime.backend == "ray"
        with pytest.raises(ValueError, match="ExchangeStore"):
            connection.query("select 1", execution="fte")
        with pytest.raises(NotImplementedError, match="parameters"):
            connection.query("select ?", [1])
    with pytest.raises(TypeError, match="RayResources"):
        vane.Runtime(vane.QueryResources())
    with pytest.raises(ValueError, match="ExchangeStore"):
        with vane.Runtime() as application:
            application.connect(execution="fte")
    with pytest.raises(vane.InvalidInputException):
        with vane.Runtime() as application:
            application.connect(":default:")


def test_failed_prepare_keeps_owner_until_cleanup_succeeds(monkeypatch):
    from vane.execution.pipelined_worker import PipelinedWorker, _Query

    worker = PipelinedWorker(vane.RayResources())
    with vane.connect(backend="local") as connection:
        spec = prepare_ray_query(
            connection,
            "select 42",
            query_id="cleanup-owner",
            options=vane.QueryExecutionOptions(vane.RayExecution(), 5, 30, 30),
            resources=ResourceDemand(1, 8, MemoryDemand(2**26, 2**20, 2**20, 2**20), 4),
        )
    tasks, routes = placement(spec, [worker.epoch], "result-epoch")
    close = _Query.close

    def fail_prepare(*args):
        raise ValueError("preparation failed")

    def fail_close(*args):
        raise RuntimeError("cleanup pending")

    monkeypatch.setattr(_Query, "prepare", fail_prepare)
    monkeypatch.setattr(_Query, "close", fail_close)
    with pytest.raises(ValueError, match="preparation failed") as error:
        worker.prepare(worker.epoch, spec.to_dict(), 0, list(tasks), routes, 3)
    assert "cleanup pending" in str(error.value.__cause__)
    assert spec.query_id in worker.resources_snapshot()["reservations"]
    assert spec.query_id in worker.queries
    monkeypatch.setattr(_Query, "close", close)
    worker.release(worker.epoch, spec.query_id)
    assert worker.resources_snapshot()["reservations"] == {}
    assert worker.queries == {}
