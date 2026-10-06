# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Analytical values retain native ownership through both exchange transports."""

import gc

import pyarrow as pa
import pytest

import vane
from tests.fast.test_direct_exchange import collect, next_batch, submission
from tests.fast.test_direct_flight import eventually
from vane._native import execution_runtime as native
from vane.execution.direct_exchange import DirectExchangeLimits, InProcessTaskService

EXPRESSIONS = [
    "range::decimal(4,1)",
    "range::decimal(15,2)",
    "range::decimal(38,5)",
    "(range + 9223372036854775807::hugeint)::hugeint",
    "date '2026-01-01' + range::integer",
    "time '13:42:06.123456'",
    "timestamp '2026-01-01 13:42:06.123456'",
    "timestamp_s '2026-01-01 13:42:06'",
    "timestamp_ms '2026-01-01 13:42:06.123'",
    "timestamp_ns '2026-01-01 13:42:06.123456789'",
    "timestamptz '2026-01-01 13:42:06.123456+08'",
    "interval '2 months 3 days 12 microseconds'",
    "'some long binary data\\x00\\xFF'::blob",
    "[range, null, range+1]",
    "[[range, null], [], [range+1]]",
    "{'x': range, 'ys': ['long string value 中文', null]}",
    "map([range, range+1], ['long string value 中文', null])",
    "[range, null, range+1]::bigint[3]",
    "[{'x': range}, null, {'x': range+1}]",
]

ARRAY_PAIR = "[[range, null]::bigint[2], [range+1, range+2]::bigint[2]]"
NESTED_ARRAY_EXPRESSIONS = [
    ARRAY_PAIR,
    f"{{'arrays': {ARRAY_PAIR}}}",
    "[{'a': [range, null]::bigint[2]}, {'a': [range+1, range+2]::bigint[2]}]",
    f"map([range, range+1], {ARRAY_PAIR})",
    f"map([range, range+1], [{ARRAY_PAIR}, {ARRAY_PAIR}])",
    f"[{ARRAY_PAIR}, [], {ARRAY_PAIR}]",
    f"[{ARRAY_PAIR}::bigint[2][2], {ARRAY_PAIR}::bigint[2][2]]",
    "[[range, null]::bigint[2], null, [range+1, range+2]::bigint[2]]",
    "[['long string value 中文', null]::varchar[2], ['another long string', range::varchar]::varchar[2]]",
    f"case when range % 3=0 then [] else {ARRAY_PAIR} end",
]

HUGEINT_VALUES = [-(2**127), -(10**38), -(2**64), 0, 2**64, 10**38, 2**127 - 1]
HUGEINT_VALUE_SQL = (
    "case range % 7 " + " ".join(f"when {i} then '{value}'::hugeint" for i, value in enumerate(HUGEINT_VALUES)) + " end"
)
HUGEINT_EXPRESSIONS = [
    HUGEINT_VALUE_SQL,
    f"[{HUGEINT_VALUE_SQL}, null]",
    f"[{{'value': {HUGEINT_VALUE_SQL}}}, null]",
    f"[{HUGEINT_VALUE_SQL}, null]::hugeint[2]",
    f"map([range], [{HUGEINT_VALUE_SQL}])",
    f"map([{HUGEINT_VALUE_SQL}], [range])",
]
LARGE_INTERVAL = "interval '2305843009213693952 microseconds'"
LARGE_INTERVAL_VALUE = dict(months=0, days=0, micros=2305843009213693952)
TEMPORAL_VALUES = [
    ("time '00:00:00'", 0),
    ("time '23:59:59.999999'", 86_399_999_999),
    ("time '24:00:00'", 86_400_000_000),
    (LARGE_INTERVAL, LARGE_INTERVAL_VALUE),
    (f"-{LARGE_INTERVAL}", dict(months=0, days=0, micros=-2305843009213693952)),
    (
        "interval '2147483647 months 2147483647 days 9223372036854775807 microseconds'",
        dict(months=2**31 - 1, days=2**31 - 1, micros=2**63 - 1),
    ),
    (
        "(interval '-2147483648 months -2147483648 days -9223372036854775807 microseconds' - interval '1 microsecond')",
        dict(months=-(2**31), days=-(2**31), micros=-(2**63)),
    ),
    (f"[{LARGE_INTERVAL}, null]", [LARGE_INTERVAL_VALUE, None]),
    (f"[{LARGE_INTERVAL}, null]::interval[2]", [LARGE_INTERVAL_VALUE, None]),
    (
        f"[{{'span': {LARGE_INTERVAL}, 'at': time '24:00:00'}}, null]",
        [dict(span=LARGE_INTERVAL_VALUE, at=86_400_000_000), None],
    ),
    ("[time '24:00:00', null]::time[2]", [86_400_000_000, None]),
    (f"map([time '24:00:00'], [{LARGE_INTERVAL}])", [(86_400_000_000, LARGE_INTERVAL_VALUE)]),
]
LIMITS = DirectExchangeLimits(window_bytes=4096, frame_bytes=1024, frame_rows=3, frame_slots=2)


@pytest.mark.parametrize("expression", EXPRESSIONS + HUGEINT_EXPRESSIONS)
@pytest.mark.parametrize("flight", [False, True])
def test_analytical_values_and_nulls(expression, flight):
    sql = (
        f"select range, case when range % 4 = 0 then null else {expression} end v "
        "from range(19) where range % 3 <> 0 order by range desc"
    )
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = connection.execute(sql).to_arrow_table()
        expected = exchange_expected(expected)
        table = exchange_table(connection, sql, flight)
        assert table.equals(expected.cast(table.schema))


def exchange_expected(table):
    # Native Arrow stores ordinary intervals in nanoseconds. The exchange
    # preserves native microseconds and separate month/day components.
    for index, field in enumerate(table.schema):
        if field.type == pa.month_day_nano_interval():
            values = [
                None if value is None else dict(months=value.months, days=value.days, micros=value.nanoseconds // 1000)
                for value in table.column(index).to_pylist()
            ]
            kind = pa.struct([("months", pa.int32()), ("days", pa.int32()), ("micros", pa.int64())])
            table = table.set_column(index, field.name, pa.array(values, type=kind))
    return table


def exchange_table(connection, sql, flight, materialized_path=None):
    spec = submission(connection, sql, partitions=3)
    with InProcessTaskService(connection, spec, LIMITS) as service:
        service.start()
        if materialized_path is not None:
            writer = native.MaterializedIO.write(
                str(materialized_path), service.result, "client", 1 << 20, native.MaterializedIO.staging_bytes(1024)
            )
            try:

                def write_status():
                    service.pump(len(service.task_ids))
                    return writer.status()

                status = eventually(write_status, lambda value: value["done"])
                assert status["error"] == ""
            finally:
                writer.close()
            target = native.DirectChannel(spec.result_schema, native.DirectLimits(4096, 1024, 3, 2), 1, ["client"])
            target.add_producer("reader")
            target.seal_producers()
            reader = native.MaterializedIO.read(
                str(materialized_path),
                target,
                "reader",
                1 << 20,
                native.MaterializedIO.staging_bytes(1024),
                status["object"],
            )
            actual = []
            try:
                while True:
                    state, batch = eventually(lambda: target.poll("client"), lambda value: value[0] != "blocked")
                    if state == "end":
                        break
                    record = batch.to_arrow(["range", "v"])
                    record.validate(full=True)
                    actual.append(record)
                    batch.close()
            finally:
                reader.close()
        elif not flight:
            actual = []
            while True:
                state, batch = next_batch(service)
                if state == "end":
                    break
                record = batch.to_arrow(["range", "v"])
                record.validate(full=True)
                actual.append(record)
                batch.close()
        else:
            source = service.result
            target = native.DirectChannel(spec.result_schema, native.DirectLimits(4096, 1024, 3, 2), 1, ["client"])
            target.add_producer("flight")
            target.seal_producers()
            sender, receiver = [
                native.DirectFlight("127.0.0.1", "127.0.0.1", 1, native.DirectFlight.staging_per_link(1024), 1024)
                for _ in range(2)
            ]
            actual = []
            try:
                sender.publish("analytical", source, "client")
                receiver.subscribe(sender.location, "analytical", target, "flight", 10)

                def poll():
                    service.pump(len(service.task_ids))
                    return target.poll("client")

                while True:
                    state, batch = eventually(poll, lambda item: item[0] != "blocked")
                    if state == "end":
                        break
                    record = batch.to_arrow(["range", "v"])
                    record.validate(full=True)
                    actual.append(record)
                    batch.close()
            finally:
                receiver.close()
                sender.close()
        table = pa.Table.from_batches(actual, schema=native.arrow_schema(spec.result_schema, ["range", "v"]))
        return table


@pytest.mark.parametrize("expression,value", TEMPORAL_VALUES)
@pytest.mark.parametrize("transport", ["direct", "flight", "materialized"])
def test_lossless_temporal_values(tmp_path, expression, value, transport):
    sql = f"select range, case when range % 3=0 then null else {expression} end v from range(13) order by range desc"
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        table = exchange_table(
            connection, sql, transport == "flight", tmp_path / "temporal.mat" if transport == "materialized" else None
        )
    assert table.to_pylist() == [dict(range=i, v=None if i % 3 == 0 else value) for i in reversed(range(13))]


@pytest.mark.parametrize(
    "expression,kind",
    [
        ("time '24:00:00'", pa.int64()),
        (LARGE_INTERVAL, pa.struct([("months", pa.int32()), ("days", pa.int32()), ("micros", pa.int64())])),
    ],
)
def test_empty_temporal_flight_preserves_lossless_schema(expression, kind):
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        table = exchange_table(connection, f"select range, {expression} v from range(0)", True)
    assert table.num_rows == 0
    assert table.schema.field("v").type == kind


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("transport", ["direct", "flight", "materialized"])
@pytest.mark.parametrize("expression", NESTED_ARRAY_EXPRESSIONS)
def test_nested_arrays_beyond_batch_cardinality(tmp_path, threads, transport, expression):
    sql = (
        f"select range, case when range % 5=0 then null else {expression} end v "
        "from range(23) where range % 4<>0 order by range desc"
    )
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        expected = connection.execute(sql).to_arrow_table()
        table = exchange_table(
            connection, sql, transport == "flight", tmp_path / "arrays.mat" if transport == "materialized" else None
        )
    table.validate(full=True)
    assert table.equals(expected.cast(table.schema))


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("first_partition", [0, 1])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize(
    "kind,scale", [("hugeint", 0), ("decimal(38,0)", 0), ("decimal(38,5)", 5), ("decimal(38,38)", 38)]
)
@pytest.mark.parametrize("aggregate", ["sum", "avg"])
def test_wide_aggregate_does_not_depend_on_producer_arrival(threads, first_partition, sign, kind, scale, aggregate):
    from tests.fast.test_analytical_fragment_compiler import scaled_wide_value
    from tests.fast.test_direct_exchange import until

    negative = scaled_wide_value(-4 * 10**37, scale)
    positive = scaled_wide_value(6 * 10**37, scale)
    sql = (
        f"select {aggregate}(v * {sign}) from (select case when range<3 then '{negative}'::{kind} "
        f"else '{positive}'::{kind} end v from range(6))"
    )
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        expected = connection.execute(sql).fetchall()
        spec = submission(connection, sql, partitions=2)
        with InProcessTaskService(connection, spec, DirectExchangeLimits(4096, 1024, 8, 4)) as service:
            producer = service.task_id(spec.graph.fragments[0].fragment_id, first_partition)
            # Queue a complete signed half before allowing the other producer
            # or aggregate to run. A plain 128-bit complete SUM still overflows.
            service.start(producer)
            until(
                service,
                lambda: any(
                    t["task_id"] == producer and t["state"] == "OUTPUT_PENDING" for t in service.snapshot()["tasks"]
                ),
            )
            service.start()
            assert collect(service) == expected


def test_nested_borrowed_slice_keeps_entire_frame_accounted():
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        spec = submission(
            connection, "select {'x': range, 'ys': ['long string value 中文', null]} from range(2)", partitions=1
        )
        with InProcessTaskService(connection, spec, LIMITS) as service:
            service.start()
            _, batch = next_batch(service)
            view = batch.slice(1, 1)
            allocated = service.result.snapshot()["bytes"]
            assert allocated > 64
            service.result.close_consumer("client")
            batch.close()
            assert service.result.snapshot()["bytes"] == allocated
            assert view.to_rows() == [({"x": 1, "ys": ["long string value 中文", None]},)]
            view.close()
            del view, batch
            gc.collect()
            assert service.result.snapshot()["bytes"] == 0


def test_oversized_nested_row_fails_without_frame_allocation():
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        spec = submission(connection, f"select ['{'x' * 2048}'] from range(1)", partitions=1)
        with InProcessTaskService(connection, spec, LIMITS) as service:
            with pytest.raises(Exception, match="one row exceeds"):
                service.start()
                collect(service)
            assert service.result.snapshot()["peak_bytes"] == 0


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize(
    "sql",
    [
        "select range % 9 k, sum(range) s, count(*) n, avg(range) a from range(103) group by k order by k",
        "select a.range from range(50) a join range(3) b on a.range % 3 = b.range order by a.range limit 7 offset 2",
        "select a.range from range(50) a full outer join range(75) b using(range) order by a.range nulls last",
    ],
)
def test_native_analytical_pipeline_small_windows(sql, threads):
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        expected = connection.execute(sql).fetchall()
        spec = submission(connection, sql, partitions=3)
        with InProcessTaskService(connection, spec, LIMITS) as service:
            service.start()
            actual = collect(service)
            assert actual == expected
            diagnostics = service.native.diagnostics()
            assert all(t["builds_ready"] == t["builds_total"] for t in diagnostics)
            assert all(t["released"] for t in diagnostics)


@pytest.mark.parametrize("threads", [1, 4])
def test_build_barrier_precedes_probe_reads(threads):
    from vane._native import execution_plan as plan

    with vane.connect(backend="local", config={"threads": threads}) as connection:
        spec = submission(connection, "select a.range from range(20) a join range(3) b using(range)", partitions=2)
        fragment = next(f for f in spec.graph.fragments if len(f.inputs) == 2)
        incoming = [e for e in spec.graph.exchanges if e.consumer_fragment_id == fragment.fragment_id]
        channels, rows = {}, {}
        for edge in incoming:
            producer = next(f for f in spec.graph.fragments if f.fragment_id == edge.producer_fragment_id)
            assignments = {s.source_id: [split.split_id for split in s.splits] for s in producer.sources}
            rows[edge.consumer_port] = plan._execute_fragment_for_test(
                connection, producer.native_plan, {}, assignments
            )
            channel = native.DirectChannel(
                producer.outputs[0].schema, native.DirectLimits(4096, 4096, 64, 2), 1, ["task"]
            )
            channel.add_producer("source")
            channel.seal_producers()
            channels[edge.consumer_port] = channel
        output = native.DirectChannel(fragment.outputs[0].schema, native.DirectLimits(4096, 4096, 64, 2), 1, ["client"])
        output.add_producer("task")
        output.seal_producers()
        service = native.TaskService(connection)
        snapshot = next(s.payload for s in spec.source_snapshots if s.fragment_id == fragment.fragment_id)
        service.prepare(
            "task",
            fragment.native_plan,
            spec.connection_snapshot,
            snapshot,
            {},
            {port: [(channel, "task")] for port, channel in channels.items()},
            [{"channels": [output], "producer": "task"}],
        )
        probe, build = [e.consumer_port for e in incoming]
        channels[probe]._write_rows("source", 1, rows[probe])
        channels[probe].finish("source", 1)
        try:
            service.start("task", "initial")
            for _ in range(10):
                service.pump(1)
            diagnostic = service.diagnostics()[0]
            assert diagnostic["builds_total"] == 1
            assert diagnostic["builds_ready"] == 0
            assert channels[probe].snapshot()["queued_frames"] == 1
            channels[build]._write_rows("source", 1, rows[build])
            channels[build].finish("source", 1)

            def poll():
                service.pump(1)
                return output.poll("client")

            state, batch = eventually(poll, lambda value: value[0] == "data")
            assert sorted(batch.to_rows()) == [(0,), (1,), (2,)]
            batch.close()
            eventually(poll, lambda value: value[0] == "end")
            assert service.diagnostics()[0]["builds_ready"] == 1
        finally:
            service.cancel("test complete")
            service.release()
