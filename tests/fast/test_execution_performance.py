# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Execution work scales with groups and frames rather than input batches."""

import json
import time
from collections import Counter

import pytest

import vane
from tests.fast.test_direct_exchange import channel, collect, next_batch, prepare_root_task, submission
from tests.fast.test_native_fragment_compiler import compile_sql, execute_graph
from vane._native import execution_plan as native
from vane.execution.direct_exchange import DirectExchangeLimits, InProcessTaskService
from vane.execution.pipelined_plan import placement
from vane.execution.pipelined_worker import _Query
from vane.execution.plan import Distribution
from vane.execution.worker_resources import pipelined_demand


def profile_fragment_scans(connection, monkeypatch):
    scans = []
    execute = native._execute_fragment_for_test

    def visit(node):
        if node.get("operator_type") == "TABLE_SCAN" and node["extra_info"].get("Function") == "READ_PARQUET":
            scans.append(node)
        for child in node.get("children", []):
            visit(child)

    def profiled(*args):
        rows = execute(*args)
        visit(json.loads(connection.get_profiling_information("json")))
        return rows

    monkeypatch.setattr(native, "_execute_fragment_for_test", profiled)
    connection.execute("PRAGMA enable_profiling='no_output'")
    return scans


@pytest.mark.parametrize("partitions", [1, 2, 5])
@pytest.mark.parametrize("column_type", ["INTEGER", "BIGINT"])
@pytest.mark.parametrize("enabled", [False, True])
def test_broadcast_join_filters_local_parquet_scan(tmp_path, monkeypatch, partitions, column_type, enabled):
    path = tmp_path / "probe.parquet"
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        connection.execute(f"copy (select range::{column_type} k from range(131072)) to '{path}' (row_group_size 2048)")
        if not enabled:
            connection.execute("set disabled_optimizers='join_filter_pushdown'")
        sql = f"select p.k from read_parquet('{path}') p join (select range * 30001 k from range(3)) b using(k)"
        expected = Counter(connection.execute(sql).fetchall())
        graph = compile_sql(connection, sql, partitions)
        scans = profile_fragment_scans(connection, monkeypatch)
        # Each load must own fresh filters, including when the same payload is
        # reused for different source partitions or a second execution.
        for _ in range(2):
            scans.clear()
            assert Counter(execute_graph(connection, graph)) == expected
            assert scans
            # Count rows leaving the scan, not wall time or rows after the join.
            scanned = sum(scan["operator_cardinality"] for scan in scans)
            if enabled:
                # The native IN filter prunes row groups; it need not evaluate
                # the IN list on every row inside a surviving group.
                assert 3 <= scanned <= 3 * 2048
            else:
                assert scanned == 131072


def test_broadcast_filters_keep_independent_probe_scans_separate(tmp_path, monkeypatch):
    path = tmp_path / "probe.parquet"
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        connection.execute(f"copy (select range k from range(131072)) to '{path}' (row_group_size 2048)")
        connection.execute("set disabled_optimizers='join_order,build_side_probe_side'")
        sql = (
            f"select p.k, q.k from (select a.k from read_parquet('{path}') a "
            "join (select range * 30001 k from range(3)) b using(k)) p "
            f"join (select c.k from read_parquet('{path}') c "
            "join (select range * 20001 k from range(5)) d using(k)) q using(k)"
        )
        expected = Counter(connection.execute(sql).fetchall())
        graph = compile_sql(connection, sql, 3)
        scans = profile_fragment_scans(connection, monkeypatch)
        assert Counter(execute_graph(connection, graph)) == expected
        assert scans
        assert sum(scan["operator_cardinality"] for scan in scans) <= 8 * 2048


@pytest.mark.parametrize(
    "expressions",
    [
        "sum(range), count(*)",
        "sum(range::decimal(15,2)), avg(range::decimal(15,2)), count(*)",
        "sum(range::hugeint), avg(range::hugeint), min(range), max(range)",
        "avg(range), avg(range::double) filter(where range % 3=0), count(*) filter(where false)",
        "sum(case when range<3 then '-40000000000000000000000000000000000000'::hugeint "
        "else '60000000000000000000000000000000000000'::hugeint end)",
    ],
)
def test_partial_aggregates_exchange_native_states(expressions):
    sql = f"select {expressions} from range(6)"
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, partitions=2)
        producer = graph.fragments[0]
        assert producer.partition_count == 2
        for partition in range(2):
            assignments = {
                source.source_id: [split.split_id for split in source.splits[partition::2]]
                for source in producer.sources
            }
            rows = native._execute_fragment_for_test(connection, producer.native_plan, {}, assignments)
            assert len(rows) == 1
            assert all(isinstance(value, bytes) for value in rows[0])
        assert execute_graph(connection, graph) == expected


@pytest.mark.parametrize("duplicate", [False, True])
def test_one_parquet_file_uses_all_partitions_with_hive_and_row_numbers(tmp_path, duplicate):
    directory = tmp_path / "part=42"
    directory.mkdir()
    path = directory / "data.parquet"
    paths = f"['{path}', '{path}']" if duplicate else f"'{path}'"
    sql = (
        f"select value, part, file_row_number from read_parquet({paths}, file_row_number=true, hive_partitioning=true)"
    )
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        connection.execute(f"copy (select range as value from range(12288)) to '{path}' (row_group_size 2048)")
        expected = Counter(connection.execute(sql).fetchall())
        graph = compile_sql(connection, sql, partitions=3)
        producer = graph.fragments[0]
        assert len(producer.sources[0].splits) == 3 * (2 if duplicate else 1)
        counts = []
        for partition in range(3):
            rows = native._execute_fragment_for_test(
                connection,
                producer.native_plan,
                {},
                {s.source_id: [split.split_id for split in s.splits[partition::3]] for s in producer.sources},
            )
            counts.append(len(rows))
        assert min(counts) > 0
        assert sum(counts) == sum(expected.values())
        assert Counter(execute_graph(connection, graph)) == expected


@pytest.mark.parametrize("threads", [1, 4])
def test_sparse_batches_are_coalesced_before_exchange(threads):
    sql = "select range, [range, null] as nested from range(65536) where range % 101=0"
    limits = DirectExchangeLimits(16384, 8192, 128, 2)
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        expected = connection.execute(sql).fetchall()
        with InProcessTaskService(connection, submission(connection, sql, partitions=1), limits=limits) as service:
            service.start()
            actual = collect(service)
            assert actual == expected
            state = service.result.snapshot()
            assert state["accepted_rows"] == len(expected)
            assert state["accepted_frames"] <= (len(expected) + 127) // 128 + threads
            assert state["peak_bytes"] <= limits.window_bytes


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("limit", [False, True])
def test_sparse_scan_flushes_before_another_nonempty_batch(threads, limit):
    # The next matching scan batch is hours away. A partial frame must leave the
    # collector while later source batches are being completely filtered out.
    sql = "select range from range(1000000000000) where range<100 or range>=999999999900"
    if limit:
        sql += " limit 1"
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        spec = submission(connection, sql, partitions=2, timeout=2)
        with InProcessTaskService(connection, spec) as service:
            service.start()
            if limit:
                assert collect(service) == [(0,)]
            else:
                status, batch = next_batch(service)
                assert status == "data"
                assert batch.to_rows() == [(row,) for row in range(100)]
                batch.close()
                assert service.result.snapshot()["accepted_rows"] == 100


def interval_average_queries():
    cases = []
    for unit, negative, positive in (
        ("months", -2000000000, 1000000000),
        ("days", -2000000000, 1000000000),
        ("microseconds", -8000000000000000000, 4000000000000000000),
    ):
        source = (
            f"select range, case when range=0 then interval '{negative} {unit}' "
            f"when range<3 then interval '0 {unit}' else interval '{positive} {unit}' end v from range(6)"
        )
        cases.append(f"select avg(v)::varchar, avg(-v)::varchar, count(v) from ({source})")
        cases.append(f"select avg(distinct v)::varchar, avg(v) filter(where range<>1)::varchar from ({source})")
        cases.append(f"select range % 2 k, avg(v)::varchar from ({source}) group by k order by k")
    cases.extend(
        [
            "select avg(interval '3 months -4 days 5 microseconds' * range)::varchar, "
            "avg(interval '-2 months 1 day -7 microseconds' * range)::varchar from range(7)",
            "select avg(null::interval)::varchar from range(9)",
            "select avg(interval '1 day')::varchar from range(0)",
        ]
    )
    return cases


@pytest.mark.parametrize("partitions", [1, 2, 3])
@pytest.mark.parametrize("sql", interval_average_queries())
def test_interval_average_uses_wide_partial_and_complete_states(sql, partitions):
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = connection.execute(sql).fetchall()
        # Keep interval components native between stages: Python timedelta has
        # a smaller day range and normalizes microseconds into days.
        spec = submission(connection, sql, partitions=partitions)
        limits = DirectExchangeLimits(4096, 1024, 64, 2)
        with InProcessTaskService(connection, spec, limits) as service:
            service.start()
            assert collect(service) == expected


@pytest.mark.parametrize("reverse", [False, True])
def test_interval_average_combines_overflowing_subtotals_in_either_order(reverse):
    sql = interval_average_queries()[3]  # The review's day-overflow input.
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, partitions=2)
        producer, final = graph.fragments
        states = []
        for partition in range(2):
            assignments = {
                source.source_id: [split.split_id for split in source.splits[partition::2]]
                for source in producer.sources
            }
            rows = native._execute_fragment_for_test(connection, producer.native_plan, {}, assignments)
            assert len(rows) == 1 and all(isinstance(value, bytes) for value in rows[0])
            states.extend(rows)
        if reverse:
            states.reverse()
        assert (
            native._execute_fragment_for_test(connection, final.native_plan, {final.inputs[0].port_id: states}, {})
            == expected
        )


def test_same_worker_edges_share_channels_and_reserve_no_flight_staging(monkeypatch):
    monkeypatch.setattr("vane.execution.pipelined_worker._host", lambda: "127.0.0.1")
    resources = vane.RayResources(worker_count=1)
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        spec = submission(connection, "select sum(range::decimal(15,2)) from range(100)")
    tasks, routes = placement(spec, ["worker"], "result")
    demand = pipelined_demand(resources, 0, list(tasks), routes)
    assert demand["io"] == 1  # Only the root result leaves this process.
    assert demand["exchange"] == len(routes) * resources.exchange.window_bytes
    owner = _Query(spec, resources, 0, routes)
    try:
        owner.prepare(list(tasks), demand["operator"])
        assert len(owner.channels) == len(routes)
        assert all(name.startswith("out/") for name in owner.channels)
        owner.connect({0: str(owner.flight.location)})
        owner.start(list(tasks))
        root = next(route for route in routes if route["target_worker"] == -1)
        channel = owner.channels[f"out/{root['id']}"]
        deadline = time.monotonic() + 10
        while True:
            status, batch = channel.poll(root["target"])
            if status == "data":
                assert str(batch.to_rows()[0][0]) == "4950.00"
                batch.close()
                break
            assert time.monotonic() < deadline, owner.snapshot()
            time.sleep(0.001)
    finally:
        owner.close()


@pytest.mark.parametrize("kind", ["inner", "left", "semi", "anti"])
@pytest.mark.parametrize("partitions", [1, 2, 5])
@pytest.mark.parametrize("comparison", ["=", "is not distinct from"])
@pytest.mark.parametrize("build_rows", [7, 70])
def test_broadcast_join_keeps_probe_partitions(kind, partitions, comparison, build_rows):
    left = "(select range i, case when range%7=0 then null else range%111 end k from range(430))"
    right = f"(select case when range=0 then null else range%31 end k from range({build_rows}))"
    sql = f"select a.i, a.k from {left} a {kind} join {right} b on a.k {comparison} b.k"
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        connection.execute("pragma disable_optimizer")
        expected = Counter(connection.execute(sql).fetchall())
        graph = compile_sql(connection, sql, partitions)
        assert Counter(execute_graph(connection, graph)) == expected
    assert not any(e.distribution is Distribution.HASH for e in graph.exchanges)
    probe = next(f for f in graph.fragments if f.sources and f.inputs)
    assert probe.partition_count == partitions
    assert len(probe.sources) == len(probe.inputs) == 1
    edge = next(e for e in graph.exchanges if e.consumer_fragment_id == probe.fragment_id)
    assert edge.distribution is (Distribution.GATHER if partitions == 1 else Distribution.BROADCAST)


def test_broadcast_join_does_not_replicate_a_single_probe_partition():
    sql = "select a.k, b.range from (select sum(range)::bigint k from range(4)) a left join range(7) b on a.k=b.range"
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        connection.execute("pragma disable_optimizer")
        graph = compile_sql(connection, sql, 5)
        assert execute_graph(connection, graph) == connection.execute(sql).fetchall() == [(6, 6)]
    assert all(e.distribution is Distribution.GATHER for e in graph.exchanges)
    assert graph.fragments[-1].partition_count == 1


@pytest.mark.parametrize("threads", [1, 4])
def test_broadcast_chain_filters_and_aggregates_before_exchange(threads):
    sql = (
        "select count(*) from range(100000) a join range(300) b on a.range%511=b.range "
        "join range(200) c on a.range%511=c.range"
    )
    limits = DirectExchangeLimits(32768, 8192, 128, 2)
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        expected = connection.execute(sql).fetchall()
        spec = submission(connection, sql, partitions=2)
        with InProcessTaskService(connection, spec, limits=limits) as service:
            service.start()
            assert collect(service) == expected
            states = [channel.snapshot() for channel in service.channels.values()]
            # Only small build rows and partial aggregates cross channels. The
            # large probe stays in its scan partitions until after filtering.
            assert sum(s["accepted_rows"] for s in states) < 2 * (300 + 200) + 64
            assert all(s["peak_bytes"] <= limits.window_bytes for s in states)


@pytest.mark.parametrize(
    "probe_rows, build_rows, partitions, broadcast",
    [(10000, 1000, 2, True), (10000, 1000, 16, False), (10000000, 1000000, 2, False)],
)
def test_broadcast_join_bounds_replica_size_and_traffic(probe_rows, build_rows, partitions, broadcast):
    sql = f"select a.range from range({probe_rows}) a join range({build_rows}) b on a.range=b.range"
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        connection.execute("pragma disable_optimizer")
        graph = compile_sql(connection, sql, partitions)
    assert any(e.distribution is Distribution.BROADCAST for e in graph.exchanges) is broadcast
    assert any(e.distribution is Distribution.HASH for e in graph.exchanges) is not broadcast


@pytest.mark.parametrize("threads", [1, 4])
def test_exchange_input_coalesces_and_flushes_while_waiting_for_more_rows(threads):
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        spec = submission(connection)
        source = channel(connection, window=8192, frame=1024, rows=64, slots=64)
        output = channel(connection, window=1024, frame=1024, rows=64, slots=1)
        source.add_producer("upstream")
        output.add_producer("task")
        source.seal_producers()
        output.seal_producers()
        for sequence in range(1, 65):
            assert source._write_rows("upstream", sequence, [(sequence,)] * 4) == "accepted"
        service = prepare_root_task(connection, spec, [(source, "client")], [output])
        try:
            service.start("task", "start")
            rows = []

            def receive(count):
                deadline = time.monotonic() + 10
                while len(rows) < count:
                    service.pump(1)
                    status, batch = output.poll("client")
                    assert status != "end"
                    if batch is not None:
                        rows.extend(batch.to_rows())
                        batch.close()
                    assert time.monotonic() < deadline

            receive(256)
            # Backpressure uses a single output slot. Input frames must still
            # combine instead of becoming 64 separate transport messages.
            assert output.snapshot()["accepted_frames"] < 16
            assert source._write_rows("upstream", 65, [(999,)]) == "accepted"
            receive(257)  # The partial frame is visible before upstream EOF.
            source.finish("upstream", 65)
            deadline = time.monotonic() + 10
            while service.status()[0]["state"] != "FINISHED":
                service.pump(1)
                assert time.monotonic() < deadline
            assert output.poll("client")[0] == "end"
            assert Counter(rows) == Counter([(value,) for value in range(1, 65) for _ in range(4)] + [(999,)])
            assert source.snapshot()["bytes"] == output.snapshot()["bytes"] == 0
        finally:
            service.cancel("test cleanup")
            service.release()
