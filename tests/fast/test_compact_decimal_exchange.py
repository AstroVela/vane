# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Compact exchange decimals retain precision, public types and validation."""

import hashlib
import struct
from decimal import Decimal

import pyarrow as pa
import pyarrow.flight as flight
import pytest

import vane
from tests.fast.test_analytical_exchange import exchange_table
from tests.fast.test_direct_flight import eventually, service
from tests.fast.test_materialized_exchange import channel, read_object, schema_for, write_object
from vane._native import execution_runtime as native


def flight_records(schema, rows):
    source, sender = channel(schema), service()
    client = reader = None
    try:
        sender.publish("decimal", source, "consumer")
        if rows:
            assert source._write_rows("producer", 1, rows) == "accepted"
        source.finish("producer", int(bool(rows)))
        client = flight.FlightClient(sender.location)
        reader = client.do_get(flight.Ticket("decimal"), flight.FlightCallOptions(timeout=5))
        wire_schema = reader.schema
        records = []
        if rows:
            record = reader.read_chunk()
            assert record.app_metadata.to_pybytes() == b"D:1"
            records.append(record.data)
            list(client.do_action(flight.Action("vane.direct.ack", b"1\ndecimal")))
        assert reader.read_chunk().app_metadata.to_pybytes() == f"F:{int(bool(rows))}".encode()
        with pytest.raises(StopIteration):
            reader.read_chunk()
        assert source.snapshot()["bytes"] == 0
        return wire_schema, records
    finally:
        if reader is not None:
            reader.cancel()
        if client is not None:
            client.close()
        sender.close()


def receive_flight(schema, record, error=None):
    class Sender(flight.FlightServerBase):
        def do_get(self, context, ticket):
            batches = [(record, b"D:1")] if record.num_rows else []
            batches.append((record.slice(0, 0), f"F:{int(bool(record.num_rows))}".encode()))
            return flight.GeneratorStream(record.schema, batches)

        def do_action(self, context, action):
            yield flight.Result(b"running")

    sender, receiver, target = Sender(("127.0.0.1", 0)), service(), channel(schema)
    batches = []
    try:
        receiver.subscribe(f"grpc://127.0.0.1:{sender.port}", "decimal", target, "producer", 10)
        if error:
            assert error in eventually(lambda: target.snapshot()["error"], bool)
            assert target.snapshot()["accepted_rows"] == target.snapshot()["bytes"] == 0
            return None
        while True:
            state, batch = eventually(lambda: target.poll("consumer"), lambda value: value[0] != "blocked")
            if state == "end":
                break
            try:
                batches.append(batch.to_arrow(["v"]))
            finally:
                batch.close()
        assert target.snapshot()["bytes"] == target.snapshot()["leased_bytes"] == 0
    finally:
        receiver.close()
        sender.shutdown()
    return pa.Table.from_batches(batches, schema=native.arrow_schema(schema, ["v"]))


@pytest.mark.parametrize("width,scale", [(1, 0), (4, 4), (5, 2), (9, 0), (10, 2), (18, 9), (19, 0), (38, 5)])
@pytest.mark.parametrize("transport", ["flight", "materialized"])
@pytest.mark.parametrize("empty", [False, True])
def test_decimal_wire_width_preserves_values_and_public_schema(tmp_path, width, scale, transport, empty):
    schema = schema_for(f"select null::decimal({width},{scale})")
    limit = "9" * width
    rows = (
        []
        if empty
        else [(Decimal((1, tuple(map(int, limit)), -scale)),), (None,), (Decimal((0, tuple(map(int, limit)), -scale)),)]
    )
    kind = (
        pa.decimal32(width, scale)
        if width <= 9
        else pa.decimal64(width, scale)
        if width <= 18
        else pa.decimal256(39, scale)
    )
    expected_schema = pa.schema([("c0", kind)])
    if transport == "flight":
        wire_schema, records = flight_records(schema, rows)
        assert wire_schema == expected_schema
    else:
        path = tmp_path / "decimal.mat"
        metadata = write_object(path, schema, rows)
        assert read_object(path, schema, metadata) == rows
        raw = path.read_bytes()
        offset = 16 + struct.unpack_from("<Q", raw, 8)[0]
        size = struct.unpack_from("<Q", raw, offset)[0]
        records = []
        if size:
            payload = raw[offset + 80 : offset + 80 + size]
            with pa.ipc.open_stream(payload) as reader:
                assert reader.schema == expected_schema
                records = list(reader)
    assert sum(record.num_rows for record in records) == len(rows)
    for record in records:
        record.validate(full=True)
        assert record.column(0).to_pylist() == [row[0] for row in rows]
        assert record.column(0).buffers()[1].size == len(rows) * kind.byte_width
    record = records[0] if records else pa.record_batch([pa.array([], type=kind)], schema=expected_schema)
    public = receive_flight(schema, record)
    public.validate(full=True)
    assert public.column(0).type == pa.decimal128(width, scale)
    assert public.column(0).to_pylist() == [row[0] for row in rows]


MIXED_DECIMALS = (
    "{'a':range::decimal(4,2), 'b':(-range)::decimal(9,2), 'c':range::decimal(18,2), 'd':range::decimal(38,2)}"
)


@pytest.mark.parametrize(
    "expression",
    [
        MIXED_DECIMALS,
        "{'payload':'long binary payload\\x00\\xFF'::blob, 'amount':range::decimal(15,2)}",
        "[{'payload':'long binary payload\\x00\\xFF'::blob, 'amount':range::decimal(15,2)},null]",
        f"[{MIXED_DECIMALS},null]",
        "map([range::decimal(9,2)], [(-range)::decimal(18,2)])",
        "[range::decimal(4,2),null]::decimal(4,2)[2]",
    ],
)
@pytest.mark.parametrize("transport", ["direct", "flight", "materialized"])
def test_nested_compact_decimals_preserve_selected_rows_and_nulls(tmp_path, expression, transport):
    sql = f"select range, case when range%4=0 then null else {expression} end v from range(19) where range%3<>0 order by range desc"
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = connection.execute(sql).to_arrow_table()
        actual = exchange_table(
            connection, sql, transport == "flight", tmp_path / "nested.mat" if transport == "materialized" else None
        )
    assert actual.schema == expected.schema
    assert actual.equals(expected)


@pytest.mark.parametrize("damage", ["decimal32", "decimal64", "utf8"])
@pytest.mark.parametrize("transport", ["flight", "materialized"])
def test_full_arrow_validation_rejects_corrupt_values_before_publication(tmp_path, damage, transport):
    if damage == "utf8":
        sql_type = "varchar"
        values = pa.Array.from_buffers(
            pa.string(), 1, [None, pa.py_buffer(struct.pack("<ii", 0, 1)), pa.py_buffer(b"\xff")]
        )
    else:
        width, size = (4, 4) if damage == "decimal32" else (15, 8)
        sql_type = f"decimal({width},0)"
        kind = pa.decimal32(width) if size == 4 else pa.decimal64(width)
        values = pa.Array.from_buffers(kind, 1, [None, pa.py_buffer((10**width).to_bytes(size, "little", signed=True))])
    record = pa.record_batch([values], names=["c0"])
    with pytest.raises(pa.ArrowInvalid):
        record.validate(full=True)
    schema = schema_for(f"select null::{sql_type}")
    error = "invalid exchange Arrow frame"
    if transport == "flight":
        receive_flight(schema, record, error)
        return
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, record.schema) as writer:
        writer.write_batch(record)
    payload = sink.getvalue().to_pybytes()
    raw = b"VANEMAT1" + struct.pack("<Q", len(schema)) + schema
    raw += struct.pack("<QQ", len(payload), 1) + hashlib.sha256(payload).hexdigest().encode() + payload
    raw += struct.pack("<QQQ", 0, 1, 1)
    path = tmp_path / "invalid.mat"
    metadata = write_object(path, schema, [(None,)])
    path.write_bytes(raw)
    metadata.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    target = channel(schema)
    reader = native.MaterializedIO.read(
        str(path), target, "producer", 1 << 20, native.MaterializedIO.staging_bytes(128), metadata
    )
    try:
        status = eventually(reader.status, lambda value: value["done"])
        assert error in status["error"]
        assert target.snapshot()["accepted_rows"] == target.snapshot()["bytes"] == 0
    finally:
        reader.close()
