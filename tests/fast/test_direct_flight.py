# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Native Flight transfer, ownership, and failure semantics."""

import json
import subprocess
import sys
import threading
import time

import pytest

import vane
from vane._native import execution_runtime as native
from vane.execution.compiler import compile_fragment_graph


def make_channel(connection, sql="select 1::bigint as value"):
    graph = compile_fragment_graph(connection, sql, query_id="flight-schema")
    root = next(f for f in graph.fragments if f.fragment_id == graph.result.fragment_id)
    channel = native.DirectChannel(root.outputs[0].schema, native.DirectLimits(128, 128, 3, 1), 1, ["consumer"])
    channel.add_producer("producer")
    channel.seal_producers()
    return channel


def service():
    return native.DirectFlight("127.0.0.1", "127.0.0.1", 4, 4 * native.DirectFlight.staging_per_link(128), 128)


def eventually(operation, predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while True:
        value = operation()
        if predicate(value):
            return value
        assert time.monotonic() < deadline, value
        time.sleep(0.005)


@pytest.fixture
def pair():
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        sender, receiver = service(), service()
        source, target = make_channel(connection), make_channel(connection)
        sender.publish("opaque-capability", source, "consumer")
        receiver.subscribe(sender.location, "opaque-capability", target, "producer", 10)
        try:
            yield sender, receiver, source, target
        finally:
            receiver.close()
            sender.close()


def test_native_flight_data_ack_and_finish(pair):
    sender, receiver, source, target = pair
    eventually(lambda: receiver.ready, bool)  # Schema handshake needs no produced rows.
    assert source._write_rows("producer", 1, [(1,), (None,), (3,)]) == "accepted"
    source.finish("producer", 1)
    _, batch = eventually(lambda: target.poll("consumer"), lambda result: result[0] == "data")
    assert batch.to_rows() == [(1,), (None,), (3,)]
    eventually(lambda: source.snapshot()["bytes"], lambda size: size == 0)
    assert target.snapshot()["leased_bytes"] > 0
    eventually(lambda: target.poll("consumer"), lambda result: result[0] == "end")
    batch.close()
    assert target.snapshot()["bytes"] == 0


def test_error_after_eof_stays_visible(pair):
    sender, receiver, source, target = pair
    source.finish("producer", 0)
    eventually(lambda: target.poll("consumer"), lambda result: result[0] == "end")
    source.abort("failure after eof")
    eventually(lambda: target.snapshot()["error"], bool)
    with pytest.raises(Exception, match="failure after eof"):
        target.poll("consumer")


def test_receiver_close_unblocks_full_window(pair):
    sender, receiver, source, target = pair
    for sequence in range(1, 3):
        eventually(lambda: source._write_rows("producer", sequence, [(sequence,)]), lambda state: state == "accepted")
    eventually(lambda: target.snapshot()["queued_frames"], lambda count: count == 1)
    assert source._write_rows("producer", 3, [(3,)]) == "blocked"
    target.close_consumer("consumer")
    eventually(lambda: source.snapshot()["closed_consumers"], lambda count: count == 1)
    receiver.close()
    assert receiver.active_links == 0


def test_lost_server_is_failure(pair):
    sender, receiver, source, target = pair
    sender.close()
    eventually(lambda: target.snapshot()["error"], bool)
    with pytest.raises(Exception):
        target.poll("consumer")


@pytest.mark.parametrize("operation", ["status", "ack", "data"])
def test_timeout_identifies_the_flight_operation(operation):
    import pyarrow as pa
    import pyarrow.flight as flight

    entered, release = threading.Event(), threading.Event()
    schema = pa.schema([("c0", pa.int64())])
    record = pa.record_batch([[7]], schema=schema)

    class Sender(flight.FlightServerBase):
        def do_get(self, context, ticket):
            def batches():
                if operation == "ack":
                    yield record, b"D:1"
                if operation == "data":
                    entered.set()
                assert release.wait(10)
                yield record.slice(0, 0), b"F:0"

            return flight.GeneratorStream(schema, batches())

        def do_action(self, context, action):
            if action.type == f"vane.direct.{operation}":
                entered.set()
                assert release.wait(10)
            yield flight.Result(b"running")

    with vane.connect(backend="local") as connection:
        target = make_channel(connection)
    sender = Sender(("127.0.0.1", 0))
    receiver = service()
    try:
        receiver.subscribe(
            f"grpc://127.0.0.1:{sender.port}", "timeout-probe", target, "producer", 1 if operation == "data" else 20
        )
        assert entered.wait(5)
        error = eventually(lambda: target.snapshot()["error"], bool)
        assert ("direct Flight data next" if operation == "data" else f"direct Flight control {operation}") in error
        assert "timeout" in error.lower() or "deadline" in error.lower()
    finally:
        release.set()
        receiver.close()
        sender.shutdown()


def test_ticket_fencing():
    with vane.connect(backend="local") as connection:
        sender, receiver = service(), service()
        source, target = make_channel(connection), make_channel(connection)
        sender.publish("current-epoch-capability", source, "consumer")
        try:
            receiver.subscribe(sender.location, "stale-epoch-capability", target, "producer", 10)
            eventually(lambda: target.snapshot()["error"], bool)
            assert source.snapshot()["accepted_rows"] == 0
        finally:
            receiver.close()
            sender.close()


@pytest.mark.parametrize(
    "sql_type,value",
    [
        ("boolean", True),
        ("tinyint", -12),
        ("smallint", -32000),
        ("integer", -100000),
        ("bigint", -(2**60)),
        ("utinyint", 200),
        ("usmallint", 60000),
        ("uinteger", 2**31),
        ("ubigint", 2**63 + 1),
        ("float", 1.5),
        ("double", 2.25),
        ("varchar", "a long UTF-8 value 中文"),
        (None, None),
    ],
)
def test_native_type_profile(sql_type, value):
    sql = "select null" if sql_type is None else f"select null::{sql_type} as value"
    with vane.connect(backend="local") as connection:
        source, target = make_channel(connection, sql), make_channel(connection, sql)
    sender, receiver = service(), service()
    try:
        sender.publish("types", source, "consumer")
        receiver.subscribe(sender.location, "types", target, "producer", 10)
        source._write_rows("producer", 1, [(None,), (value,)])
        source.finish("producer", 1)
        _, batch = eventually(lambda: target.poll("consumer"), lambda result: result[0] == "data")
        assert batch.to_arrow(["value"]).column(0).to_pylist() == [None, value]
        batch.close()
        eventually(lambda: target.poll("consumer"), lambda result: result[0] == "end")
    finally:
        receiver.close()
        sender.close()


def test_invalid_ack_and_duplicate_stream():
    import pyarrow as pa
    import pyarrow.flight as flight

    with vane.connect(backend="local") as connection:
        source = make_channel(connection)
    sender = service()
    try:
        sender.publish("controls", source, "consumer")
        client = flight.FlightClient(sender.location)
        with pytest.raises(pa.ArrowInvalid, match="ACK"):
            list(client.do_action(flight.Action("vane.direct.ack", b"1\ncontrols")))
        source._write_rows("producer", 1, [(42,)])
        reader = client.do_get(flight.Ticket("controls"))
        batch = reader.read_chunk()
        assert batch.app_metadata.to_pybytes() == b"D:1"
        assert source.snapshot()["leased_bytes"] > 0
        list(client.do_action(flight.Action("vane.direct.ack", b"1\ncontrols")))
        list(client.do_action(flight.Action("vane.direct.ack", b"1\ncontrols")))
        eventually(lambda: source.snapshot()["bytes"], lambda size: size == 0)
        with pytest.raises(pa.ArrowInvalid, match="replayed"):
            client.do_get(flight.Ticket("controls"))
        list(client.do_action(flight.Action("vane.direct.close", b"\ncontrols")))
        reader.cancel()
        client.close()
    finally:
        sender.close()


@pytest.mark.parametrize(
    "sql_type,value,error",
    [
        ("hugeint", -(2**127) - 1, "outside the HUGEINT range"),
        ("hugeint", 2**127, "outside the HUGEINT range"),
        ("decimal(38,0)", -(2**127) - 1, "outside the DECIMAL storage range"),
        ("decimal(38,0)", 2**127, "outside the DECIMAL storage range"),
        ("time", -1, "outside the TIME range"),
        ("time", 86_400_000_001, "outside the TIME range"),
        ("interval", dict(months=1, days=2, micros=None), "INTERVAL has a null component"),
    ],
)
def test_values_outside_native_domain_are_rejected(sql_type, value, error):
    import pyarrow as pa
    import pyarrow.flight as flight

    kinds = {
        "hugeint": pa.decimal256(39, 0),
        "decimal(38,0)": pa.decimal256(39, 0),
        "time": pa.int64(),
        "interval": pa.struct([("months", pa.int32()), ("days", pa.int32()), ("micros", pa.int64())]),
    }
    schema = pa.schema([("c0", kinds[sql_type])])
    record = pa.RecordBatch.from_arrays([pa.array([value], type=schema.field(0).type)], schema=schema)
    record.validate(full=True)

    class Sender(flight.FlightServerBase):
        def do_get(self, context, ticket):
            return flight.GeneratorStream(schema, [(record, b"D:1"), (record.slice(0, 0), b"F:1")])

        def do_action(self, context, action):
            yield flight.Result(b"open")

    with vane.connect(backend="local") as connection:
        target = make_channel(connection, f"select null::{sql_type}")
    sender = Sender(("127.0.0.1", 0))
    receiver = service()
    try:
        receiver.subscribe(f"grpc://127.0.0.1:{sender.port}", "hugeint", target, "producer", 10)
        assert error in eventually(lambda: target.snapshot()["error"], bool)
        assert target.snapshot()["bytes"] == 0
    finally:
        receiver.close()
        sender.shutdown()


def test_schema_mismatch_is_failure():
    with vane.connect(backend="local") as connection:
        source = make_channel(connection)
        target = make_channel(connection, "select false")
    sender, receiver = service(), service()
    try:
        sender.publish("schema", source, "consumer")
        receiver.subscribe(sender.location, "schema", target, "producer", 10)
        eventually(lambda: target.snapshot()["error"], bool)
        assert "schema mismatch" in target.snapshot()["error"]
    finally:
        receiver.close()
        sender.close()


@pytest.mark.parametrize("width,scale", [(19, 0), (38, 0), (38, 5), (38, 38)])
def test_decimal_exchange_accepts_entire_native_coefficient_range(width, scale):
    from decimal import Decimal

    import pyarrow as pa
    import pyarrow.flight as flight

    from tests.fast.test_analytical_fragment_compiler import scaled_wide_value

    values = [Decimal(scaled_wide_value(value, scale)) for value in [-(2**127), 2**127 - 1]] + [None]
    schema = pa.schema([("c0", pa.decimal256(39, scale))])
    record = pa.RecordBatch.from_arrays([pa.array(values, type=schema.field(0).type)], schema=schema)
    record.validate(full=True)

    class Sender(flight.FlightServerBase):
        def do_get(self, context, ticket):
            return flight.GeneratorStream(schema, [(record, b"D:1"), (record.slice(0, 0), b"F:1")])

        def do_action(self, context, action):
            yield flight.Result(b"open")

    with vane.connect(backend="local") as connection:
        target = make_channel(connection, f"select null::decimal({width},{scale})")
    sender = Sender(("127.0.0.1", 0))
    receiver = service()
    try:
        receiver.subscribe(f"grpc://127.0.0.1:{sender.port}", "decimal", target, "producer", 10)
        _, batch = eventually(lambda: target.poll("consumer"), lambda result: result[0] == "data")
        column = batch.to_arrow(["value"]).column(0)
        assert column.type == pa.decimal128(width, scale)
        # Inspect intermediates before SQL finalization in a wide Arrow view;
        # they need not fit the declared result precision or Python formatter.
        wide = column.cast(pa.decimal256(39, scale))
        wide.validate(full=True)
        assert wide.to_pylist() == values
        batch.close()
        eventually(lambda: target.poll("consumer"), lambda result: result[0] == "end")
    finally:
        receiver.close()
        sender.shutdown()


@pytest.mark.parametrize("threads", [1, 4])
def test_production_probe_preserves_error_without_control_pump(threads):
    from vane.execution.direct_exchange import DirectExchangeLimits, InProcessTaskService
    from vane.execution.query_options import RayExecution
    from vane.execution.resource_demand import MemoryDemand, ResourceDemand
    from vane.execution.submission import prepare_ray_query

    with vane.connect(backend="local", config={"threads": threads}) as connection:
        spec = prepare_ray_query(
            connection,
            "select 42::bigint",
            query_id="production-probe",
            options=vane.QueryExecutionOptions(RayExecution(), 5, 30, 30),
            resources=ResourceDemand(1, 8, MemoryDemand(2**26, 2**20, 2**20, 2**20), 4),
        )
        with InProcessTaskService(connection, spec, DirectExchangeLimits(128, 128, 3, 1)) as tasks:
            tasks.start()
            deadline = time.monotonic() + 5
            while not tasks.native.production_status()["finished"]:
                assert time.monotonic() < deadline
                if threads == 1:
                    tasks.pump(1)
                else:
                    time.sleep(0.005)
            tasks.result.abort("failure after native production")
            assert "failure after native production" in tasks.native.production_status()["error"]


@pytest.mark.parametrize("failure", ["abort_after_eof", "process_loss"])
def test_two_process_native_fragments(failure):
    child = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-u",
            "-c",
            """
import json, sys, threading, time
import vane
from vane._native import execution_runtime as native
from vane.execution.direct_exchange import DirectExchangeLimits, InProcessTaskService
from vane.execution.compiler import FragmentCompileOptions
from vane.execution.query_options import RayExecution
from vane.execution.resource_demand import MemoryDemand, ResourceDemand
from vane.execution.submission import prepare_ray_query
con = vane.connect(backend="local", config={"threads": 4})
spec = prepare_ray_query(con, "select range as value from range(20)", query_id="remote-test",
    options=vane.QueryExecutionOptions(RayExecution(), 5, 30, 30),
    resources=ResourceDemand(1, 8, MemoryDemand(2**26, 2**20, 2**20, 2**20), 4),
    compile_options=FragmentCompileOptions(2))
tasks = InProcessTaskService(con, spec, DirectExchangeLimits(128,128,3,1))
flight = native.DirectFlight("127.0.0.1", "127.0.0.1", 1, native.DirectFlight.staging_per_link(128),128)
flight.publish("subprocess-capability", tasks.result, "client")
tasks.start()
stop = threading.Event()
def pump():
    while not stop.is_set():
        tasks.pump(8)
        stop.wait(.002)
thread = threading.Thread(target=pump)
thread.start()
print(json.dumps({"location":flight.location}),flush=True)
try:
    for command in sys.stdin:
        if command.strip() == "abort":
            tasks.result.abort("subprocess error after EOF")
        else:
            break
finally:
    stop.set()
    thread.join()
    flight.close()
    tasks.close()
    con.close()
""",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    receiver = service()
    try:
        line = child.stdout.readline()
        assert line, child.stderr.read()
        location = json.loads(line)["location"]
        with vane.connect(backend="local") as connection:
            target = make_channel(connection)
        receiver.subscribe(location, "subprocess-capability", target, "producer", 30)
        _, batch = eventually(lambda: target.poll("consumer"), lambda value: value[0] == "data")
        values = batch.to_rows()
        batch.close()
        if failure == "process_loss":
            child.kill()
            child.wait(timeout=5)
        else:
            while True:
                state, batch = eventually(lambda: target.poll("consumer"), lambda value: value[0] != "blocked")
                if state == "end":
                    break
                values += batch.to_rows()
                batch.close()
            assert sorted(values) == [(i,) for i in range(20)]
            child.stdin.write("abort\n")
            child.stdin.flush()
        eventually(lambda: target.snapshot()["error"], bool)
        with pytest.raises(Exception):
            target.poll("consumer")
    finally:
        receiver.close()
        if child.poll() is None:
            child.stdin.write("stop\n")
            child.stdin.flush()
            child.communicate(timeout=10)
        else:
            child.communicate(timeout=5)


def test_gateway_revoke_fences_old_stream_without_canceling_other_query():
    with vane.connect(backend="local") as connection:
        sender, first, second = service(), service(), service()
        left, right = make_channel(connection), make_channel(connection)
        target_left, target_right = make_channel(connection), make_channel(connection)
        try:
            sender.publish("first", left, "consumer")
            sender.publish("second", right, "consumer")
            first.subscribe(sender.location, "first", target_left, "producer", 10)
            second.subscribe(sender.location, "second", target_right, "producer", 10)
            eventually(lambda: first.ready and second.ready, bool)
            assert not sender.delivered("first")
            sender.revoke("first")
            with pytest.raises(Exception, match="unknown or expired"):
                sender.delivered("first")
            assert right._write_rows("producer", 1, [(42,)]) == "accepted"
            right.finish("producer", 1)
            _, batch = eventually(lambda: target_right.poll("consumer"), lambda item: item[0] == "data")
            assert batch.to_rows() == [(42,)]
            batch.close()
            eventually(lambda: target_right.poll("consumer"), lambda item: item[0] == "end")
            assert sender.delivered("second")
            right.abort("late persistent failure")
            with pytest.raises(Exception, match="late persistent failure"):
                sender.delivered("second")
            sender.revoke("second")
            assert sender.active_links == 0
        finally:
            first.close()
            second.close()
            sender.close()


def test_gateway_revoke_after_transport_error_releases_reservation(pair):
    sender, receiver, source, target = pair
    eventually(lambda: receiver.ready, bool)
    source.abort("upstream failed before finish")
    eventually(lambda: target.snapshot()["error"], bool)
    # Flight can destroy a failed stream without calling its Close method.
    sender.revoke("opaque-capability")
    assert sender.active_links == 0
