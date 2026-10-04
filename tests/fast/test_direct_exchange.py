# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real native channels, vector leases and cooperative fragment execution."""

import gc
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

import vane
from vane._native import execution_runtime as native
from vane.execution.compiler import FragmentCompileOptions, compile_fragment_graph
from vane.execution.direct_exchange import DirectExchangeLimits, InProcessTaskService
from vane.execution.query_options import DistributedMode, FteOptions, QueryExecutionOptions, RayExecution
from vane.execution.resource_demand import MemoryDemand, ResourceDemand
from vane.execution.submission import prepare_ray_query


@pytest.fixture
def connection():
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        yield connection


def schema(connection, sql="select 1::bigint as value"):
    graph = compile_fragment_graph(connection, sql, query_id="schema")
    return (
        next(fragment for fragment in graph.fragments if fragment.fragment_id == graph.result.fragment_id)
        .outputs[0]
        .schema
    )


def channel(connection, *, consumers=("client",), producers=2, window=64, frame=64, rows=3, slots=1, sql=None):
    return native.DirectChannel(
        schema(connection, sql) if sql else schema(connection),
        native.DirectLimits(window, frame, rows, slots),
        producers,
        list(consumers),
    )


def submission(connection, sql="select range from range(100)", *, partitions=2, hash_columns=(), timeout=30):
    return prepare_ray_query(
        connection,
        sql,
        query_id="direct-test",
        options=QueryExecutionOptions(RayExecution(), 5, timeout, 30),
        resources=ResourceDemand(1, 32, MemoryDemand(2**26, 2**20, 2**20, 2**20), 1),
        compile_options=FragmentCompileOptions(partitions, hash_columns),
    )


TINY = DirectExchangeLimits(window_bytes=64, frame_bytes=64, frame_rows=3, frame_slots=1)


def until(service, predicate, *, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, service.snapshot()
        service.pump(len(service.task_ids) or 1)


def next_batch(service):
    deadline = time.monotonic() + 10
    while True:
        status, batch = service.poll_result()
        if status != "blocked":
            return status, batch
        assert time.monotonic() < deadline, service.snapshot()
        service.pump(len(service.task_ids))


def collect(service):
    rows = []
    while True:
        status, batch = next_batch(service)
        if status == "end":
            break
        assert status == "data"
        rows.extend(batch.to_rows())
        batch.close()
    until(service, lambda: all(task["state"] == "FINISHED" for task in service.snapshot()["tasks"]))
    assert service.snapshot()["owned_bytes"] == 0
    assert service.snapshot()["leased_bytes"] == 0
    assert service.snapshot()["active_contexts"] == 0
    return rows


def prepare_root_task(connection, spec, inputs, outputs):
    fragment = next(item for item in spec.graph.fragments if item.fragment_id == spec.graph.result.fragment_id)
    snapshot = next(item.payload for item in spec.source_snapshots if item.fragment_id == fragment.fragment_id)
    service = native.TaskService(connection)
    service.prepare(
        "task",
        fragment.native_plan,
        spec.connection_snapshot,
        snapshot,
        {},
        {fragment.inputs[0].port_id: inputs} if inputs else {},
        [{"channels": [output], "producer": "task"} for output in outputs],
    )
    return service


def test_membership_finish_and_sequences(connection):
    exchange = channel(connection)
    signal = native._TestSignal()
    assert exchange.poll("client", signal) == ("blocked", None)
    exchange.add_producer("a")
    exchange.finish("a", 0)
    assert signal.count == 1
    assert exchange.poll("client")[0] == "blocked"  # Membership remains open.
    exchange.add_producer("b")
    exchange.seal_producers()
    with pytest.raises(vane.InvalidInputException, match="registration"):
        exchange.add_producer("c")
    with pytest.raises(vane.InvalidInputException, match="sequence"):
        exchange._write_rows("b", 2, [(2,)])
    assert exchange._write_rows("b", 1, [(1,)]) == "accepted"
    with pytest.raises(vane.InvalidInputException, match="sequence"):
        exchange._write_rows("b", 1, [(1,)])
    with pytest.raises(vane.InvalidInputException, match="FINISH"):
        exchange.finish("b", 2)
    exchange.finish("b", 1)
    exchange.finish("b", 1)  # Idempotent FINISH.
    _, batch = exchange.poll("client")
    assert batch.to_rows() == [(1,)]
    assert exchange.poll("client")[0] == "end"
    assert not exchange.producer_drained("b")  # EOF does not return borrowed credit.
    batch.close()
    assert exchange.producer_drained("b")
    with pytest.raises(vane.InvalidInputException, match="sequence"):
        exchange._write_rows("b", 2, [(2,)])
    assert exchange.snapshot()["bytes"] == 0


def test_borrowed_slice_keeps_credit_after_consumer_close(connection):
    exchange = channel(connection)
    exchange.add_producer("a")
    exchange.seal_producers()
    assert exchange._write_rows("a", 1, [(1,), (2,), (3,)]) == "accepted"
    _, batch = exchange.poll("client")
    signal = native._TestSignal()
    assert exchange._write_rows("a", 2, [(4,)], signal) == "blocked"
    view = batch.slice(1, 2)
    batch.close()
    gc.collect()
    assert exchange.snapshot()["bytes"] == 32
    assert signal.count == 0
    exchange.close_consumer("client")
    assert signal.count == 1
    assert exchange.snapshot()["bytes"] == 32
    assert view.to_rows() == [(2,), (3,)]
    assert exchange._write_rows("a", 2, [(4,)]) == "closed"
    exchange.finish("a", 1)
    assert not exchange.producer_drained("a")
    view.close()
    assert exchange.producer_drained("a")
    assert exchange.snapshot()["outstanding_frames"] == 0
    assert exchange.snapshot()["bytes"] == 0


def test_broadcast_one_allocation_independent_leases(connection):
    exchange = channel(connection, consumers=("one", "two"))
    exchange.add_producer("a")
    assert exchange._write_rows("a", 1, [(4,)]) == "accepted"
    assert exchange.snapshot()["bytes"] == 16
    assert exchange.snapshot()["leased_bytes"] == 32
    _, first = exchange.poll("one")
    first.close()
    assert exchange._write_rows("a", 2, [(5,)]) == "blocked"
    exchange.close_consumer("two")
    assert exchange.snapshot()["bytes"] == 0
    assert exchange._write_rows("a", 2, [(5,)]) == "accepted"
    _, second = exchange.poll("one")
    assert second.to_rows() == [(5,)]
    second.close()
    exchange.close_consumer("one")
    assert exchange.snapshot()["leased_bytes"] == 0


def test_error_is_sticky_and_wakes_readers_and_writers(connection):
    exchange = channel(connection, consumers=("one", "two"))
    exchange.add_producer("a")
    exchange.seal_producers()
    reader = native._TestSignal()
    assert exchange.poll("one", reader)[0] == "blocked"
    assert exchange._write_rows("a", 1, [(1,)]) == "accepted"
    assert reader.count == 1
    _, batch = exchange.poll("one")
    assert exchange.poll("one", reader)[0] == "blocked"
    writer = native._TestSignal()
    assert exchange._write_rows("a", 2, [(2,)], writer) == "blocked"
    exchange.abort("injected failure")
    exchange.abort("later failure")
    assert reader.count == 2 and writer.count == 1
    assert batch.to_rows() == [(1,)]
    assert exchange.snapshot()["bytes"] > 0
    for consumer in ("one", "two"):
        with pytest.raises(vane.InvalidInputException, match="injected failure"):
            exchange.poll(consumer)
    batch.close()
    assert exchange.snapshot()["bytes"] == 0


def test_owned_strings_nulls_and_oversized_row(connection):
    exchange = channel(
        connection,
        sql="select 1::integer, 'text'::varchar, NULL::bigint, true",
        window=256,
        frame=256,
    )
    exchange.add_producer("a")
    rows = [(1, "hello" * 10, None, True), (None, "短文本", 7, False), (3, None, -2, None)]
    assert exchange._write_rows("a", 1, rows) == "accepted"
    _, batch = exchange.poll("client")
    assert batch.to_rows() == rows
    view = batch.slice(0, 2)
    batch.close()
    assert view.to_rows() == rows[:2]
    view.close()
    with pytest.raises(vane.InvalidInputException, match="frame_bytes"):
        exchange._write_rows("a", 2, [(1, "x" * 1000, None, True)])
    assert exchange.snapshot()["bytes"] == 0


@pytest.mark.parametrize("hash_columns", [(), (0,)])
@pytest.mark.parametrize("partitions", [1, 2, 5])
def test_real_fragments_tiny_window_resume_without_duplicates(connection, hash_columns, partitions):
    sql = "select range * 3 as value from range(113) where range % 4 != 0"
    expected = connection.execute(sql).fetchall()
    spec = submission(connection, sql, partitions=partitions, hash_columns=hash_columns)
    with InProcessTaskService(connection, spec, TINY) as service:
        assert all(task["state"] == "PREPARED" for task in service.snapshot()["tasks"])
        assert service.snapshot()["owned_bytes"] == 0
        service.start()
        assert sorted(collect(service)) == sorted(expected)
        for snapshot in service.snapshot()["channels"].values():
            assert snapshot["peak_bytes"] <= TINY.window_bytes
            assert snapshot["finished_producers"] == snapshot["producers"]


def test_downstream_reads_before_upstream_finishes_and_retained_result_blocks(connection):
    spec = submission(connection, "select range from range(10000)")
    with InProcessTaskService(connection, spec, TINY) as service:
        service.start()
        status, first = next_batch(service)
        assert status == "data"
        snapshot = service.snapshot()
        source_ids = {fragment.fragment_id for fragment in spec.graph.fragments if fragment.sources}
        assert any(
            task["state"] == "RUNNING" and task["task_id"].rsplit("/", 1)[0] in source_ids for task in snapshot["tasks"]
        )
        view = first.slice(0, 1)
        first.close()
        service.pump(100)
        assert service.poll_result()[0] == "blocked"
        assert service.result.snapshot()["outstanding_frames"] == 1
        assert any(item["write_blocks"] for item in service.snapshot()["channels"].values())
        view.close()
        status, second = next_batch(service)
        assert status == "data"
        second.close()
    assert service.snapshot()["owned_bytes"] == 0
    assert service.snapshot()["active_contexts"] == 0


def test_output_pending_does_not_block_finalization(connection):
    spec = submission(connection, "select 42::bigint", partitions=1)
    with InProcessTaskService(connection, spec, TINY) as service:
        service.start()
        until(service, lambda: service.snapshot()["tasks"][0]["state"] == "OUTPUT_PENDING")
        assert service.snapshot()["active_contexts"] == 0
        # An already dispatched execution timer cannot cancel completed
        # production while the client is still borrowing its output.
        assert not service.native.expire()
        _, batch = service.poll_result()
        assert batch.to_rows() == [(42,)]
        assert service.poll_result()[0] == "end"
        assert service.snapshot()["tasks"][0]["state"] == "OUTPUT_PENDING"
        batch.close()
        assert service.snapshot()["tasks"][0]["state"] == "FINISHED"


@pytest.mark.parametrize("expiry", ["direct", "timer"])
@pytest.mark.parametrize("empty", [False, True])
def test_background_finish_prevents_execution_expiry(expiry, empty):
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        sql = "select 42::bigint where false" if empty else "select 42::bigint"
        spec = submission(connection, sql, partitions=1, timeout=1 if expiry == "timer" else 30)
        with InProcessTaskService(connection, spec, TINY) as service:
            service.start()
            deadline = time.monotonic() + 5
            # Observe only the native channel. Neither pump nor status may be
            # needed to publish completion to the independent deadline timer.
            while service.result.snapshot()["finished_producers"] != 1:
                assert time.monotonic() < deadline
                time.sleep(0.001)
            status, batch = service.poll_result()
            if empty:
                assert (status, batch) == ("end", None)
            else:
                assert status == "data" and batch.to_rows() == [(42,)]
                assert service.poll_result() == ("end", None)
            try:
                if expiry == "timer":
                    service._timer.join(timeout=5)
                    assert not service._timer.is_alive()
                assert not service.native.expire()
                assert not service.result.snapshot()["error"]
                assert service.poll_result() == ("end", None)
                until(service, lambda: service.snapshot()["active_contexts"] == 0)
                if batch is not None:
                    assert batch.to_rows() == [(42,)]
                    assert service.snapshot()["tasks"][0]["state"] == "OUTPUT_PENDING"
                    assert service.snapshot()["owned_bytes"] > 0
            finally:
                if batch is not None:
                    batch.close()
            assert service.snapshot()["tasks"][0]["state"] == "FINISHED"
            assert service.snapshot()["owned_bytes"] == 0


@pytest.mark.parametrize("sql", ["select range from range(0)", "select range from range(2) where range > 5"])
def test_empty_partitions_seal_and_finish(connection, sql):
    spec = submission(connection, sql, partitions=5, hash_columns=(0,))
    with InProcessTaskService(connection, spec, TINY) as service:
        service.start()
        assert collect(service) == []


def test_cancel_blocked_pipeline_releases_native_owners_but_preserves_view(connection):
    spec = submission(connection, "select range from range(1000000)")
    with InProcessTaskService(connection, spec, TINY) as service:
        service.start()
        _, batch = next_batch(service)
        rows = batch.to_rows()
        service.pump(30)
        service.cancel("user canceled")
        service.native.release()
        assert service.snapshot()["active_contexts"] == 0
        assert batch.to_rows() == rows
        assert service.snapshot()["owned_bytes"] == service.result.snapshot()["bytes"] > 0
        with pytest.raises(vane.InvalidInputException, match="user canceled"):
            service.poll_result()
        batch.close()
        assert service.snapshot()["owned_bytes"] == 0
        assert all(task["state"] == "CANCELED" for task in service.snapshot()["tasks"])


def test_start_token_and_close_before_start(connection):
    spec = submission(connection, "select 7::bigint", partitions=1)
    with InProcessTaskService(connection, spec, TINY) as service:
        service.start(token="a")
        service.start(token="a")
        with pytest.raises(vane.InvalidInputException, match="conflicting"):
            service.start(token="b")
        assert collect(service) == [(7,)]
        service.start(token="a")
    service = InProcessTaskService(connection, spec, TINY)
    service.close()
    service.close()
    with pytest.raises(RuntimeError, match="closed"):
        service.start()
    assert service.snapshot()["owned_bytes"] == 0
    assert service.snapshot()["active_contexts"] == 0


def test_capacity_and_fte_rejected_before_creating_tasks(connection):
    spec = submission(connection)
    with pytest.raises(vane.InvalidInputException, match="cannot hold one row"):
        InProcessTaskService(connection, spec, DirectExchangeLimits(8, 8, 1, 1))
    resources = replace(spec.resources, memory=replace(spec.resources.memory, exchange_bytes=1))
    with pytest.raises(ValueError, match="exchange reservation"):
        InProcessTaskService(connection, replace(spec, resources=resources), TINY)
    resources = replace(spec.resources, memory=replace(spec.resources.memory, result_bytes=1))
    with pytest.raises(ValueError, match="result reservation"):
        InProcessTaskService(connection, replace(spec, resources=resources), TINY)
    options = replace(spec.options, target=RayExecution(DistributedMode.FTE, FteOptions("store", 2, 0)))
    with pytest.raises(ValueError, match="pipelined"):
        InProcessTaskService(connection, replace(spec, options=options), TINY)


@pytest.mark.parametrize("terminal", ["finish", "seal", "close", "abort"])
def test_terminal_events_wake_an_empty_reader(connection, terminal):
    exchange = channel(connection)
    signal = native._TestSignal()
    exchange.add_producer("a")
    if terminal == "seal":
        exchange.finish("a", 0)
    else:
        exchange.seal_producers()
    assert exchange.poll("client", signal)[0] == "blocked"
    if terminal == "finish":
        exchange.finish("a", 0)
    elif terminal == "seal":
        exchange.seal_producers()
    elif terminal == "close":
        exchange.close_consumer("client")
    else:
        exchange.abort("failed empty producer")
    assert signal.count == 1
    if terminal == "abort":
        with pytest.raises(vane.InvalidInputException, match="failed empty producer"):
            exchange.poll("client")
    else:
        assert exchange.poll("client")[0] == ("closed" if terminal == "close" else "end")


@pytest.mark.parametrize("event_first", [False, True])
def test_publication_and_waiter_registration_cannot_lose_wakeup(connection, event_first):
    exchange = channel(connection)
    exchange.add_producer("a")
    signal = native._TestSignal()
    if event_first:
        assert exchange._write_rows("a", 1, [(1,)]) == "accepted"
        status, batch = exchange.poll("client", signal)
        assert status == "data" and signal.count == 0
    else:
        assert exchange.poll("client", signal)[0] == "blocked"
        assert exchange._write_rows("a", 1, [(1,)]) == "accepted"
        assert signal.count == 1
        _, batch = exchange.poll("client")
    writer = native._TestSignal()
    if event_first:
        batch.close()
        assert exchange._write_rows("a", 2, [(2,)], writer) == "accepted"
        assert writer.count == 0
    else:
        assert exchange._write_rows("a", 2, [(2,)], writer) == "blocked"
        batch.close()
        assert writer.count == 1
        assert exchange._write_rows("a", 2, [(2,)], writer) == "accepted"
    exchange.close_consumer("client")
    assert exchange.snapshot()["bytes"] == 0


def test_native_publication_races_waiter_registration(connection):
    exchange = channel(connection)
    exchange.add_producer("a")
    with ThreadPoolExecutor(max_workers=2) as threads:
        for sequence in range(1, 101):
            signal = native._TestSignal()
            barrier = threading.Barrier(2)

            def read():
                barrier.wait(timeout=3)
                return exchange.poll("client", signal)

            def write():
                barrier.wait(timeout=3)
                return exchange._write_rows("a", sequence, [(sequence,)])

            reader = threads.submit(read)
            writer = threads.submit(write)
            status, batch = reader.result(timeout=3)
            assert writer.result(timeout=3) == "accepted"
            if status == "blocked":
                assert signal.count == 1
                status, batch = exchange.poll("client")
            assert status == "data"
            assert batch.to_rows() == [(sequence,)]
            batch.close()
    assert exchange.snapshot()["bytes"] == 0


@pytest.mark.parametrize(
    "type_name,values",
    [
        ("boolean", [True, False, None]),
        ("tinyint", [-128, 127, None]),
        ("smallint", [-32768, 32767, None]),
        ("integer", [-(2**31), 2**31 - 1, None]),
        ("bigint", [-(2**63), 2**63 - 1, None]),
        ("utinyint", [0, 255, None]),
        ("usmallint", [0, 65535, None]),
        ("uinteger", [0, 2**32 - 1, None]),
        ("ubigint", [0, 2**64 - 1, None]),
        ("float", [-1.25, 2.5, None]),
        ("double", [-1.25, 2.5, None]),
        ("varchar", ["abc", "", None]),
    ],
)
def test_native_frame_basic_type_profile(connection, type_name, values):
    exchange = channel(connection, sql=f"select NULL::{type_name}")
    exchange.add_producer("a")
    rows = [(value,) for value in values]
    assert exchange._write_rows("a", 1, rows) == "accepted"
    _, batch = exchange.poll("client")
    assert batch.to_rows() == rows
    batch.close()
    assert exchange.snapshot()["bytes"] == 0


def test_consumer_starts_before_producers_and_wakes_on_first_frame(connection):
    spec = submission(connection)
    with InProcessTaskService(connection, spec, TINY) as service:
        root = service.task_ids[0]
        service.start(root)
        service.pump(30)
        snapshot = service.snapshot()
        assert all(item["accepted_rows"] == 0 for item in snapshot["channels"].values())
        assert any(item["read_blocks"] for item in snapshot["channels"].values())
        assert service.poll_result()[0] == "blocked"
        service.start()  # Root's start token repeats; producer tasks start now.
        assert sorted(collect(service)) == [(value,) for value in range(100)]


def test_native_source_merges_multiple_inputs_without_waiting_for_idle_input(connection):
    spec = submission(connection)
    fragment = next(item for item in spec.graph.fragments if item.fragment_id == spec.graph.result.fragment_id)
    snapshot = next(item.payload for item in spec.source_snapshots if item.fragment_id == fragment.fragment_id)
    left, right, output = (channel(connection) for _ in range(3))
    left.add_producer("left")
    right.add_producer("right")
    output.add_producer("task")
    for exchange in (left, right, output):
        exchange.seal_producers()
    service = native.TaskService(connection)
    try:
        service.prepare(
            "task",
            fragment.native_plan,
            spec.connection_snapshot,
            snapshot,
            {},
            {fragment.inputs[0].port_id: [(left, "client"), (right, "client")]},
            [{"channels": [output], "producer": "task"}],
        )
        service.start("task", "start")
        service.pump(3)
        assert left.snapshot()["read_blocks"] and right.snapshot()["read_blocks"]
        # Left remains idle: right must make progress independently.
        assert right._write_rows("right", 1, [(99,)]) == "accepted"
        service.pump(5)
        status, batch = output.poll("client")
        assert status == "data" and batch.to_rows() == [(99,)]
        batch.close()
        right.finish("right", 1)
        service.pump(5)
        assert output.poll("client")[0] == "blocked"
        left.finish("left", 0)
        service.pump(5)
        assert output.poll("client")[0] == "end"
        assert service.status()[0]["state"] == "FINISHED"
        assert service.status()[0]["released"]
        assert all(exchange.snapshot()["bytes"] == 0 for exchange in (left, right, output))
    finally:
        service.cancel("test cleanup")
        service.release()


def test_consumer_close_propagates_upstream_without_finishing_large_scan(connection):
    spec = submission(connection, "select range from range(1000000000)")
    with InProcessTaskService(connection, spec, TINY) as service:
        service.start()
        _, batch = next_batch(service)
        batch.close()
        service.result.close_consumer("client")
        until(service, lambda: all(task["state"] == "FINISHED" for task in service.snapshot()["tasks"]))
        assert service.snapshot()["owned_bytes"] == 0
        assert service.snapshot()["active_contexts"] == 0
        assert all(item["accepted_rows"] < 100 for item in service.snapshot()["channels"].values())


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("routing", ["broadcast", "outputs"])
@pytest.mark.parametrize("borrowed", [False, True])
def test_last_output_close_stops_task_waiting_for_input(threads, routing, borrowed):
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        spec = submission(connection)
        upstream = channel(connection)
        upstream.add_producer("upstream")
        upstream.seal_producers()
        if routing == "broadcast":
            output = channel(connection, consumers=("first", "last"))
            outputs = [output]
            consumers = [(output, "first"), (output, "last")]
        else:
            outputs = [channel(connection), channel(connection)]
            consumers = [(output, "client") for output in outputs]
        for output in outputs:
            output.add_producer("task")
            output.seal_producers()
        service = prepare_root_task(connection, spec, [(upstream, "client")], outputs)
        batch = None
        try:
            service.start("task", "initial")
            if borrowed:
                upstream._write_rows("upstream", 1, [(42,)])
            deadline = time.monotonic() + 5
            while not upstream.snapshot()["read_blocks"] or (
                borrowed
                and (
                    any(output.snapshot()["accepted_rows"] != 1 for output in outputs)
                    or upstream.snapshot()["outstanding_frames"]
                )
            ):
                assert time.monotonic() < deadline
                service.pump(1)
            if borrowed:
                output, consumer = consumers[-1]
                status, batch = output.poll(consumer)
                assert status == "data" and batch.to_rows() == [(42,)]
            consumers[0][0].close_consumer(consumers[0][1])
            service.pump(3)
            assert service.status()[0]["state"] == "RUNNING"
            assert upstream.snapshot()["closed_consumers"] == 0
            consumers[-1][0].close_consumer(consumers[-1][1])
            service.pump(3)
            state = service.status()[0]
            assert state["state"] == ("OUTPUT_PENDING" if borrowed else "FINISHED")
            assert state["released"] and not state["error"]
            assert upstream.snapshot()["closed_consumers"] == 1
            assert upstream.snapshot()["bytes"] == 0
            assert all(output.snapshot()["finished_producers"] == 1 for output in outputs)
            assert not service.expire()
            if batch is not None:
                assert batch.to_rows() == [(42,)]
                assert sum(output.snapshot()["bytes"] for output in outputs) > 0
                batch.close()
            assert service.status()[0]["state"] == "FINISHED"
            assert sum(output.snapshot()["bytes"] for output in outputs) == 0
        finally:
            if batch is not None:
                batch.close()
            service.cancel("test cleanup")
            service.release()


@pytest.mark.parametrize("entry", ["status", "pump", "release"])
@pytest.mark.parametrize("borrowed", [False, True])
def test_aborted_output_fails_before_other_output_leases_drain(connection, entry, borrowed):
    spec = submission(connection, "select 42::bigint", partitions=1)
    left, right = channel(connection), channel(connection)
    for output in (left, right):
        output.add_producer("task")
        output.seal_producers()
    service = prepare_root_task(connection, spec, [], [left, right])
    batch = None
    try:
        service.start("task", "initial")
        service.pump(10)
        assert service.status()[0]["state"] == "OUTPUT_PENDING"
        assert left.snapshot()["queued_frames"] == right.snapshot()["queued_frames"] == 1
        if borrowed:
            _, batch = left.poll("client")
        right.abort("injected output failure")
        with pytest.raises(vane.InvalidInputException, match="injected output failure"):
            right.producer_drained("task")
        if entry == "pump":
            service.pump(1)
        else:
            getattr(service, entry)()
        state = service.status()[0]
        assert state["state"] == "FAILED" and state["released"]
        assert "injected output failure" in state["error"]
        for output in (left, right):
            with pytest.raises(vane.InvalidInputException, match="injected output failure"):
                output.poll("client")
        if batch is not None:
            assert batch.to_rows() == [(42,)]
            assert left.snapshot()["bytes"] > 0
            batch.close()
        assert left.snapshot()["bytes"] == right.snapshot()["bytes"] == 0
        service.cancel("later cancellation")
        assert service.status()[0]["state"] == "FAILED"
        assert "injected output failure" in service.status()[0]["error"]
    finally:
        if batch is not None:
            batch.close()
        service.cancel("test cleanup")
        service.release()


def test_single_oversized_row_fails_without_allocating_exchange_buffer(connection):
    spec = submission(connection, "select '" + "x" * 1000 + "'", partitions=1)
    with InProcessTaskService(connection, spec, TINY) as service:
        service.start()
        with pytest.raises(vane.InvalidInputException, match="one row exceeds"):
            service.pump(10)
        service.native.release()
        assert service.snapshot()["tasks"][0]["state"] == "FAILED"
        assert service.snapshot()["owned_bytes"] == 0
        assert service.result.snapshot()["peak_bytes"] == 0
        with pytest.raises(vane.InvalidInputException, match="one row exceeds"):
            service.poll_result()


def test_execution_error_after_partial_result_is_not_eof(connection):
    sql = "select (case when range >= 2048 then 'late native error' else range::varchar end)::bigint from range(10000)"
    spec = submission(connection, sql, partitions=1)
    limits = DirectExchangeLimits(window_bytes=65536, frame_bytes=32768, frame_rows=1024, frame_slots=1)
    with InProcessTaskService(connection, spec, limits) as service:
        service.start()
        status, batch = next_batch(service)
        assert status == "data" and batch.num_rows > 0
        batch.close()
        with pytest.raises(vane.ConversionException, match="late native error"):
            collect(service)
        service.native.release()
        assert service.snapshot()["owned_bytes"] == 0
        assert any(task["state"] == "FAILED" for task in service.snapshot()["tasks"])
        with pytest.raises(vane.InvalidInputException, match="late native error"):
            service.poll_result()


def test_execution_deadline_interrupts_an_unpumped_blocked_query(connection):
    spec = submission(connection, "select range from range(10000)", timeout=0.1)
    with InProcessTaskService(connection, spec, TINY) as service:
        service.start()
        _, batch = next_batch(service)
        # The control timer must run without a caller continuing to pump tasks.
        deadline = time.monotonic() + 3
        while not service.result.snapshot()["error"]:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        with pytest.raises(vane.InvalidInputException, match="execution deadline"):
            service.poll_result()
        service.native.release()
        assert service.snapshot()["active_contexts"] == 0
        batch.close()
        assert service.snapshot()["owned_bytes"] == 0


def test_prepared_parquet_is_revalidated_before_start(connection, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "input.parquet"
    pq.write_table(pa.table({"value": [1, 2, 3]}), path)
    spec = submission(connection, f"select value from read_parquet('{path}')", partitions=1)
    with InProcessTaskService(connection, spec, TINY) as service:
        assert service.snapshot()["owned_bytes"] == 0
        pq.write_table(pa.table({"value": [10, 20, 30, 40]}), path)
        with pytest.raises(vane.InvalidInputException, match="changed"):
            service.start()
        service.native.release()
        assert service.snapshot()["owned_bytes"] == 0
        assert service.snapshot()["active_contexts"] == 0


@pytest.mark.parametrize("threads", [1, 4])
def test_wakeup_cancel_and_teardown_in_independent_process(threads):
    # A stale task callback or engine-lock deadlock must not hang the test shard.
    script = r"""
import gc, sys
import vane
from vane.execution.compiler import FragmentCompileOptions
from vane.execution.direct_exchange import DirectExchangeLimits, InProcessTaskService
from vane.execution.query_options import QueryExecutionOptions, RayExecution
from vane.execution.resource_demand import MemoryDemand, ResourceDemand
from vane.execution.submission import prepare_ray_query

with vane.connect(backend='local', config={'threads': int(sys.argv[1])}) as connection:
    for iteration in range(12):
        spec = prepare_ray_query(connection, 'select range from range(1000000)', query_id=str(iteration),
            options=QueryExecutionOptions(RayExecution(), 5, 10, 10),
            resources=ResourceDemand(1, 16, MemoryDemand(2**24, 2**20, 2**20, 2**20), 1),
            compile_options=FragmentCompileOptions(3, (0,)))
        service = InProcessTaskService(connection, spec, DirectExchangeLimits(64, 64, 3, 1))
        service.start()
        for _ in range(1000):
            service.pump(7)
            status, batch = service.poll_result()
            if batch is not None:
                break
        assert batch is not None
        view = batch.slice(0, 1)
        batch.close()
        service.close()
        assert service.snapshot()['active_contexts'] == 0
        assert view.num_rows == 1
        view.close()  # Late credit/wakeup after the native task has been destroyed.
        assert service.snapshot()['owned_bytes'] == 0
        del service
        gc.collect()
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(threads)], capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("hash_columns", [(), (0, 1)])
@pytest.mark.parametrize("threads", [1, 4])
def test_parquet_streams_owned_string_frames_and_nullable_hash_keys(tmp_path, hash_columns, threads):
    import pyarrow as pa
    import pyarrow.parquet as pq

    for part in range(3):
        pq.write_table(
            pa.table(
                {
                    "key": [None if row % 5 == 0 else row % 4 for row in range(17)],
                    "text": [None if row % 3 == 0 else f"值-{part}-{row}-" * 10 for row in range(17)],
                }
            ),
            tmp_path / f"{part}.parquet",
        )
    sql = f"select key, upper(text) as text from read_parquet('{tmp_path}/*.parquet') where key is null or key > 1"
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        from collections import Counter

        expected = connection.execute(sql).fetchall()
        spec = submission(connection, sql, partitions=5, hash_columns=hash_columns)
        limits = DirectExchangeLimits(512, 512, 17, 1)
        with InProcessTaskService(connection, spec, limits) as service:
            service.start()
            assert Counter(collect(service)) == Counter(expected)
            assert all(item["peak_bytes"] <= limits.window_bytes for item in service.snapshot()["channels"].values())


def test_task_prepare_rejects_duplicate_output_owner(connection):
    spec = submission(connection, "select 1::bigint", partitions=1)
    with InProcessTaskService(connection, spec, TINY) as service:
        fragment = spec.graph.fragments[0]
        with pytest.raises(vane.InvalidInputException, match="duplicate writer"):
            service.native.prepare(
                "other",
                fragment.native_plan,
                spec.connection_snapshot,
                spec.source_snapshots[0].payload,
                {},
                {},
                [{"channels": [service.result], "producer": service.task_ids[0]}],
            )
        # Preparation failure did not publish the extra task or start a query.
        assert len(service.snapshot()["tasks"]) == 1
        assert service.snapshot()["owned_bytes"] == 0
        service.start()
        assert collect(service) == [(1,)]


def test_partial_sink_resumes_second_output_without_replaying_first(connection):
    spec = submission(connection, "select range from range(3)", partitions=1)
    fragment = spec.graph.fragments[0]
    left, right = channel(connection), channel(connection)
    for exchange in (left, right):
        exchange.add_producer("task")
    right.add_producer("hold")
    right._write_rows("hold", 1, [(-1,)])
    right.finish("hold", 1)
    left.seal_producers()
    right.seal_producers()
    service = native.TaskService(connection)
    try:
        service.prepare(
            "task",
            fragment.native_plan,
            spec.connection_snapshot,
            spec.source_snapshots[0].payload,
            {source.source_id: [split.split_id for split in source.splits] for source in fragment.sources},
            {},
            [{"channels": [left], "producer": "task"}, {"channels": [right], "producer": "task"}],
        )
        service.start("task", "initial")
        service.pump(3)
        assert right.snapshot()["write_blocks"] > 0
        _, batch = left.poll("client")
        assert batch.to_rows() == [(0,), (1,), (2,)]
        batch.close()
        _, held = right.poll("client")
        held.close()
        service.pump(5)
        assert left.poll("client")[0] == "end"
        assert left.snapshot()["accepted_rows"] == 3
        _, batch = right.poll("client")
        assert batch.to_rows() == [(0,), (1,), (2,)]
        batch.close()
        assert service.status()[0]["state"] == "FINISHED"
    finally:
        service.cancel("test cleanup")
        service.release()


@pytest.mark.parametrize("method", ["pump", "start", "cancel", "close", "snapshot"])
def test_task_entry_rejects_python_input_callback_before_locks_or_state_changes(method):
    script = r"""
import io, sys
from datetime import datetime, timezone
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import vane
from vane.execution.direct_exchange import DirectExchangeLimits, InProcessTaskService
from vane.execution.query_options import QueryExecutionOptions, RayExecution
from vane.execution.resource_demand import MemoryDemand, ResourceDemand
from vane.execution.submission import prepare_ray_query

buffer = io.BytesIO()
pq.write_table(pa.table({'value': [1, 2, 3]}), buffer)
payload = buffer.getvalue()
attempts = []
class Filesystem(fsspec.AbstractFileSystem):
    protocol = 'directcallback'
    def _open(self, path, mode='rb', **kwargs):
        if not attempts:
            attempts.append(sys.argv[1])
            try:
                getattr(service, sys.argv[1])()
            except vane.InvalidInputException as error:
                assert 'reentrantly from a Python input callback' in str(error)
            else:
                raise AssertionError('entered task service from input callback')
        return io.BytesIO(payload)
    def info(self, path, **kwargs):
        return {'name': path, 'size': len(payload), 'type': 'file'}
    def modified(self, path):
        return datetime(2026, 1, 1, tzinfo=timezone.utc)

with vane.connect(backend='local', config={'threads': 1}) as connection:
    spec = prepare_ray_query(connection, 'select 7::bigint', query_id='callback',
        options=QueryExecutionOptions(RayExecution(), 5, 10, 10),
        resources=ResourceDemand(1, 16, MemoryDemand(2**24, 2**20, 2**20, 2**20), 1))
    with InProcessTaskService(connection, spec, DirectExchangeLimits(64, 64, 3, 1)) as service:
        connection.register_filesystem(Filesystem(skip_instance_cache=True))
        assert connection.execute("select * from read_parquet('directcallback://input.parquet')").fetchall() == [(1,), (2,), (3,)]
        assert attempts == [sys.argv[1]]
        assert service.snapshot()['tasks'][0]['state'] == 'PREPARED'
        service.start()
        service.pump(10)
        status, batch = service.poll_result()
        assert status == 'data' and batch.to_rows() == [(7,)]
        batch.close()
"""
    completed = subprocess.run([sys.executable, "-I", "-c", script, method], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_collected_service_destructor_releases_tasks_but_not_exported_views(connection):
    spec = submission(connection, "select range from range(1000000)")
    service = InProcessTaskService(connection, spec, TINY)
    service.start()
    _, batch = next_batch(service)
    expected = batch.to_rows()
    exchange = service.result
    timer = service._timer
    del service
    gc.collect()
    timer.join(timeout=3)
    assert not timer.is_alive()
    assert exchange.snapshot()["error"] == "task service destroyed"
    assert batch.to_rows() == expected
    batch.close()
    assert exchange.snapshot()["bytes"] == 0


def test_cancel_interrupts_a_concurrent_pump_without_waiting_for_its_control_lock():
    script = r"""
from concurrent.futures import ThreadPoolExecutor
import threading, time
import vane
from vane.execution.compiler import FragmentCompileOptions
from vane.execution.direct_exchange import DirectExchangeLimits, InProcessTaskService
from vane.execution.query_options import QueryExecutionOptions, RayExecution
from vane.execution.resource_demand import MemoryDemand, ResourceDemand
from vane.execution.submission import prepare_ray_query

with vane.connect(backend='local', config={'threads': 1}) as connection:
    spec = prepare_ray_query(connection,
        'select range from range(10000000000) where range % 1000000000 = 999999999', query_id='cancel-pump',
        options=QueryExecutionOptions(RayExecution(), 5, 10, 10),
        resources=ResourceDemand(1, 16, MemoryDemand(2**24, 2**20, 2**20, 2**20), 1),
        compile_options=FragmentCompileOptions(2))
    with InProcessTaskService(connection, spec, DirectExchangeLimits(64, 64, 3, 1)) as service:
        service.start()
        entered = threading.Event()
        def pump():
            entered.set()
            try:
                service.pump(1000000)
            except vane.Error:
                pass
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(pump)
            assert entered.wait(3)
            deadline = time.monotonic() + 3
            while not any(channel.snapshot()['read_blocks'] for channel in service.channels.values()):
                assert time.monotonic() < deadline
                time.sleep(0.001)
            service.cancel('concurrent cancellation')
            future.result(timeout=3)
        service.native.release()
        assert service.snapshot()['active_contexts'] == 0
        assert service.snapshot()['owned_bytes'] == 0
        assert all(task['state'] == 'CANCELED' for task in service.snapshot()['tasks'])
    assert connection.execute('select 42').fetchone() == (42,)
"""
    completed = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stdout + completed.stderr
