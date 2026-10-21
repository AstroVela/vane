# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Execution work scales with groups and frames rather than input batches."""

import json
import subprocess
import sys
import textwrap
import time
from collections import Counter
from dataclasses import replace

import pytest

import vane
from tests.fast.test_direct_exchange import channel, collect, next_batch, prepare_root_task, submission
from tests.fast.test_execution_submission import options as query_options
from tests.fast.test_execution_submission import resources as query_resources
from tests.fast.test_fragment_sql_coverage import (
    correlated_owner_queries,
    early_join_filter_query,
    large_join_key_query,
)
from tests.fast.test_native_fragment_compiler import compile_sql, execute_graph
from vane._native import execution_plan as native
from vane.execution.compiler import FragmentCompileOptions, compile_fragment_graph
from vane.execution.direct_exchange import DirectExchangeLimits, InProcessTaskService
from vane.execution.pipelined_plan import placement
from vane.execution.pipelined_worker import _Query
from vane.execution.plan import Distribution
from vane.execution.submission import prepare_ray_query
from vane.execution.worker_database import WorkerDatabasePool
from vane.execution.worker_resources import pipelined_demand


def partitioned_join_query(key="p.k", comparison="="):
    def source(count):
        return f"select case when range%997=0 then null else range//2 end k, range%13 v from range({count})"

    right_key = "c.k::integer" if "integer" in key else "c.k"
    return (
        "select sum(p.av+p.bv+c.v), count(*) from "
        f"(select a.k, a.v av, b.v bv from ({source(4096)}) a "
        f"join ({source(8192)}) b on a.k {comparison} b.k) p "
        f"join ({source(16384)}) c on {key} {comparison} {right_key}"
    )


@pytest.mark.parametrize("threads,partitions", [(1, 2), (4, 3)])
@pytest.mark.parametrize("comparison", ["=", "is not distinct from"])
@pytest.mark.parametrize("key,reused", [("p.k", True), ("p.k+1", False), ("p.k::integer", False)])
def test_hash_partition_reuse_requires_unchanged_key_representation(threads, partitions, comparison, key, reused):
    config = {"threads": threads, "disabled_optimizers": "join_order,build_side_probe_side,compressed_materialization"}
    with vane.connect(backend="local", config=config) as connection:
        sql = partitioned_join_query(key, comparison)
        expected = connection.execute(sql).fetchall()
        spec = submission(connection, sql, partitions=partitions)
        edges = [e for e in spec.graph.exchanges if e.distribution is Distribution.PASSTHROUGH]
        assert bool(edges) is reused
        limits = DirectExchangeLimits(4096, 1024, 32, 1)
        with InProcessTaskService(connection, spec, limits) as service:
            service.start()
            assert collect(service) == expected
            for edge in edges:
                for part in range(partitions):
                    state = service.channels[f"{edge.exchange_id}/{part}"].snapshot()
                    assert state["producers"] == 1
                    assert state["peak_bytes"] <= limits.window_bytes
        assert all(c.snapshot()["bytes"] == c.snapshot()["leased_bytes"] == 0 for c in service.channels.values())


@pytest.mark.parametrize("columns,reused", [((0, 1), True), ((1, 0), False)])
def test_hash_partition_reuse_preserves_composite_key_order(columns, reused):
    sql = (
        "select a.k,a.v from (select range//2 k,range%7 v from range(4096)) a "
        "join (select range//2 k,range%7 v from range(8192)) b on a.k=b.k and a.v=b.v"
    )
    config = {"threads": 1, "disabled_optimizers": "join_order,build_side_probe_side,compressed_materialization"}
    with vane.connect(backend="local", config=config) as connection:
        graph = compile_sql(connection, sql, 3, columns)
        assert any(e.distribution is Distribution.PASSTHROUGH for e in graph.exchanges) is reused
        assert Counter(execute_graph(connection, graph)) == Counter(connection.execute(sql).fetchall())


def test_hash_partition_reuse_drops_null_extended_key():
    def source(count):
        return f"select case when range%97=0 then null else range end k,range v from range({count})"

    sql = (
        "select count(*),sum(p.av+p.bv+c.v) from "
        f"(select a.k,a.v av,b.v bv from ({source(64)}) a right join ({source(256)}) b "
        f"on a.k=b.k) p join ({source(1024)}) c on p.k is not distinct from c.k"
    )
    config = {"threads": 1, "disabled_optimizers": "join_order,build_side_probe_side,compressed_materialization"}
    with vane.connect(backend="local", config=config) as connection:
        spec = submission(connection, sql, partitions=3)
        assert all(e.distribution is not Distribution.PASSTHROUGH for e in spec.graph.exchanges)
        with InProcessTaskService(connection, spec, DirectExchangeLimits(4096, 1024, 32, 1)) as service:
            service.start()
            assert collect(service) == connection.execute(sql).fetchall()


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


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("nulls", ["none", "mixed", "all"])
def test_merge_aggregate_batches_preserve_repeated_targets_and_null_states(grouped, nulls):
    sql = (
        "select range k, sum(range) s, count(range) n from range(8) group by k"
        if grouped
        else "select sum(range) s, count(range) n from range(8)"
    )
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        graph = compile_sql(connection, sql, partitions=2)
        producer = graph.fragments[0]
        partials = native._execute_fragment_for_test(
            connection,
            producer.native_plan,
            {},
            {source.source_id: [split.split_id for split in source.splits] for source in producer.sources},
        )
        edge = next(edge for edge in graph.exchanges if edge.producer_fragment_id == producer.fragment_id)
        final = next(fragment for fragment in graph.fragments if fragment.fragment_id == edge.consumer_fragment_id)
        # Exceed a native vector and revisit each target many times in a batch.
        # NULL state blobs must be skipped independently for each aggregate.
        rows = []
        sums, counts = Counter(), Counter()
        for index in range(4099):
            partial = partials[index % len(partials)]
            key = partial[0] if grouped else 0
            sum_state, count_state = partial[-2:]
            if nulls == "all" or (nulls == "mixed" and index % 3 == 0):
                sum_state = None
            if nulls == "all" or (nulls == "mixed" and index % 5 == 0):
                count_state = None
            rows.append((*partial[:-2], sum_state, count_state))
            if sum_state is not None:
                sums[key] += key if grouped else 28
            if count_state is not None:
                counts[key] += 1 if grouped else 8
        expected = [
            (*((key,) if grouped else ()), sums.get(key), counts[key]) for key in (range(8) if grouped else [0])
        ]
        for _ in range(2):
            assert Counter(
                native._execute_fragment_for_test(connection, final.native_plan, {edge.consumer_port: rows}, {})
            ) == Counter(expected)
        malformed = [list(row) for row in rows]
        malformed[-1][-1] = partials[0][-1] + b"invalid"
        with pytest.raises(vane.IOException, match="Aggregate state size mismatch"):
            native._execute_fragment_for_test(connection, final.native_plan, {edge.consumer_port: malformed}, {})
        assert connection.execute("select 42").fetchall() == [(42,)]


def batched_aggregate_query(groups):
    return (
        "select k, sum(v), count(v), avg(v), min(v), max(v), "
        "sum(v::decimal(15,2)), avg(v::decimal(15,2)), "
        "sum(v::hugeint), avg(v::hugeint), avg(interval '1 microsecond' * v)::varchar, "
        "sum(v) filter(where i % 3=0), count(*) filter(where false) "
        f"from (select range i, range % {groups} k, "
        "case when k % 17=0 or range % 11=0 then null else range % 257-128 end v "
        "from range(65539)) group by k order by k"
    )


@pytest.mark.parametrize("groups", [7, 4099])
@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("partitions", [2, 3])
def test_merge_aggregate_batches_across_bounded_channels(groups, threads, partitions):
    sql = batched_aggregate_query(groups)
    limits = DirectExchangeLimits(65536, 16384, 127, 4)
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        expected = connection.execute(sql).fetchall()
        with InProcessTaskService(
            connection, submission(connection, sql, partitions=partitions), limits=limits
        ) as service:
            service.start()
            assert collect(service) == expected
            assert all(
                item.snapshot()["bytes"] == item.snapshot()["leased_bytes"] == 0
                and item.snapshot()["peak_bytes"] <= limits.window_bytes
                for item in service.channels.values()
            )


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


@pytest.mark.parametrize("partitions", [2, 3, 5])
@pytest.mark.parametrize("groups", [3, 5])
@pytest.mark.parametrize("duplicate", [False, True])
def test_parquet_split_rows_balance_odd_groups_across_files(tmp_path, partitions, groups, duplicate):
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        paths = []
        for index in range(partitions * 4):
            path = tmp_path / f"part{index:02}.parquet"
            connection.execute(
                f"copy (select range+{index * groups * 2048} k from range({groups * 2048 - 257})) "
                f"to '{path}' (row_group_size 2048)"
            )
            paths.append(str(path))
        sql = f"select k from read_parquet({(paths * (2 if duplicate else 1))!r})"
        graph = compile_sql(connection, sql, partitions=partitions)
        producer = graph.fragments[0]
        counts = []
        for partition in range(partitions):
            rows = native._execute_fragment_for_test(
                connection,
                producer.native_plan,
                {},
                {s.source_id: [split.split_id for split in s.splits[partition::partitions]] for s in producer.sources},
            )
            counts.append(len(rows))
        assert max(counts) - min(counts) <= 2048
        assert Counter(execute_graph(connection, graph)) == Counter(connection.execute(sql).fetchall())


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
    databases = WorkerDatabasePool(resources)
    owner = _Query(spec, resources, 0, routes, databases.acquire(spec.query_id, reusable=True))
    try:
        owner.prepare(list(tasks))
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
        for anchor in databases.idle:
            anchor.close()


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


@pytest.mark.parametrize("partitions", [1, 2, 5])
@pytest.mark.parametrize("predicate", ["in", "not in"])
@pytest.mark.parametrize("build", ["range%11", "case when range=0 then null else range%11 end", "null::bigint"])
def test_broadcast_mark_join_preserves_complete_build_semantics(partitions, predicate, build):
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        connection.execute("pragma disable_optimizer")
        for count in (0, 70):
            sql = (
                f"select i, k, k {predicate} (select {build} from range({count})) is_match "
                "from (select range i, case when range%7=0 then null else range%17 end k from range(430)) "
                "order by i"
            )
            expected = connection.execute(sql).fetchall()
            graph = compile_sql(connection, sql, partitions)
            assert execute_graph(connection, graph) == expected
            probe = next(f for f in graph.fragments if f.sources and f.inputs)
            assert probe.partition_count == partitions
            assert not any(e.distribution is Distribution.HASH for e in graph.exchanges)
            assert any(e.distribution is Distribution.BROADCAST for e in graph.exchanges) == (partitions > 1)


@pytest.mark.parametrize("threads", [1, 4])
def test_broadcast_mark_filters_and_aggregates_before_exchange(tmp_path, threads):
    path = tmp_path / "probe.parquet"
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        connection.execute(
            f"copy (select range i, case when range%7=0 then null else range%101 end k "
            f"from range(32768)) to '{path}' (row_group_size 2048)"
        )
        sql = (
            f"select count(*), sum(i)::bigint from read_parquet('{path}') "
            "where k in (1,3,5,7,9,11,13,15) and k not in (select range*3 from range(17))"
        )
        expected = connection.execute(sql).fetchall()
        spec = submission(connection, sql, partitions=3)
        limits = DirectExchangeLimits(4096, 1024, 32, 2)
        with InProcessTaskService(connection, spec, limits=limits) as service:
            service.start()
            assert collect(service) == expected
            states = [channel.snapshot() for channel in service.channels.values()]
            # Only complete small builds and partial aggregates cross channels.
            # The Parquet probe, including its literal IN list, stays distributed.
            assert sum(s["accepted_rows"] for s in states) < 100
            assert all(s["peak_bytes"] <= limits.window_bytes for s in states)
            assert all(s["bytes"] == s["leased_bytes"] == 0 for s in states)


@pytest.mark.parametrize("probe_rows,build_rows,partitions", [(10000, 1000, 16), (10000000, 1000000, 2)])
def test_mark_join_gathers_when_build_exceeds_broadcast_budget(probe_rows, build_rows, partitions):
    sql = f"select range, range in (select range from range({build_rows})) from range({probe_rows})"
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        connection.execute("pragma disable_optimizer")
        for budget in (0, 16 << 30):
            graph = compile_fragment_graph(
                connection,
                sql,
                query_id="mark-broadcast-budget",
                options=FragmentCompileOptions(partitions, worker_operator_bytes=budget),
            )
            assert all(e.distribution is Distribution.GATHER for e in graph.exchanges)
            assert next(f for f in graph.fragments if f.inputs).partition_count == 1


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


@pytest.mark.parametrize("nested", [False, True])
def test_broadcast_join_does_not_assume_narrow_build_rows(nested):
    payload = "[range] payload" if nested else ", ".join(f"range+{i} p{i}" for i in range(20))
    sql = f"select * from range(10000000) a join (select range, {payload} from range(100000)) b using(range)"
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        connection.execute("pragma disable_optimizer")
        graph = compile_sql(connection, sql, 2)
    assert any(e.distribution is Distribution.HASH for e in graph.exchanges)
    assert not any(e.distribution is Distribution.BROADCAST for e in graph.exchanges)


@pytest.mark.parametrize(
    "budget,build_rows,partitions,nested,broadcast",
    [
        (0, 1000000, 2, False, False),
        (128 << 20, 1000000, 2, False, False),
        (256 << 20, 1000000, 2, False, True),
        (256 << 20, 2000000, 2, False, False),
        (1 << 30, 2000000, 2, False, True),
        (16 << 30, 8000000, 2, False, False),
        (1 << 30, 1000000, 64, False, False),
        (1 << 30, 1000000, 2, True, False),
    ],
)
def test_large_broadcast_requires_worker_budget_and_traffic_margin(budget, build_rows, partitions, nested, broadcast):
    payload = "[range]" if nested else "repeat(range::varchar, 20)"
    sql = (
        f"select a.range, a.payload from (select range, {payload} payload from range(10000000)) a "
        f"join range({build_rows}) b on a.range=b.range"
    )
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        connection.execute("pragma disable_optimizer")
        graph = compile_fragment_graph(
            connection,
            sql,
            query_id="broadcast-budget",
            options=FragmentCompileOptions(partitions, worker_operator_bytes=budget),
        )
        assert any(e.distribution is Distribution.BROADCAST for e in graph.exchanges) is broadcast


def large_broadcast_query(connection, path, kind="inner", comparison="="):
    connection.execute(
        "copy (select range%4096 i, case when range%1048576=0 then null else range%1048576 end k, "
        "'fact-'||(range%4096)::varchar payload, 'other-'||(range%4096)::varchar extra "
        f"from range(2097152)) to '{path}' (row_group_size 2048)"
    )
    return (
        "select p.i%31 g, count(*), sum(length(p.payload)+length(p.extra)) "
        f"from read_parquet('{path}') p {kind} join "
        "(select case when range<2 then null else range//2 end k from range(1048576)) b "
        f"on p.k {comparison} b.k group by g order by g"
    )


def early_broadcast_filter_query(connection, path, build_kind):
    connection.execute(
        "copy (select range i, range%65536 k, 'fact-'||(range%4096)::varchar payload, "
        "'extra-'||(range%4096)::varchar extra, 'third-'||(range%4096)::varchar third "
        f"from range(2097152)) to '{path}' (row_group_size 2048)"
    )
    if build_kind == "small":
        orders = "select range i from range(1048576)"
        keys = "select range k from range(8192) where range%512=0"
        payload = "length(f.payload)"
    else:
        orders = "select range i, 'order-'||range::varchar payload from range(1048576)"
        keys = (
            "select range%65536 k, 'build-'||range::varchar v, 'other-'||range::varchar w "
            "from range(327680) where range%1024=0"
        )
        payload = "length(f.payload)+length(f.extra)+length(f.third)+length(o.payload)+length(b.v)+length(b.w)"
    return (
        f"select f.i%31 g, count(*), sum({payload}) from read_parquet('{path}') f "
        f"join ({orders}) o using(i) join ({keys}) b on f.k=b.k group by g order by g"
    )


@pytest.mark.parametrize("build_kind", ["small", "large"])
@pytest.mark.parametrize("partitions", [2, 3])
def test_large_broadcast_preserves_filters_before_earlier_joins(tmp_path, build_kind, partitions):
    config = {"threads": 4, "disabled_optimizers": "join_order,build_side_probe_side", "memory_limit": "1GiB"}
    with vane.connect(backend="local", config=config) as connection:
        sql = early_broadcast_filter_query(connection, tmp_path / "fact.parquet", build_kind)
        expected = connection.execute(sql).fetchall()
        spec = prepare_ray_query(
            connection,
            sql,
            query_id="preserve-early-filter",
            options=query_options(),
            resources=replace(query_resources(), memory=replace(query_resources().memory, operator_bytes=1 << 30)),
            compile_options=FragmentCompileOptions(partitions, worker_operator_bytes=1 << 30),
        )
        # Keep the selective build shared with an early filter. Broadcasting a
        # larger join must not make that filter disappear from an earlier join.
        assert any(
            len({e.consumer_fragment_id for e in spec.graph.exchanges if e.producer_fragment_id == f.fragment_id}) > 1
            for f in spec.graph.fragments
        )
        limits = DirectExchangeLimits(65536, 16384, 512, 4)
        with InProcessTaskService(connection, spec, limits) as service:
            service.start()
            assert collect(service) == expected
            states = [c.snapshot() for c in service.channels.values()]
            assert sum(s["accepted_rows"] for s in states) < 1100000
            assert all(s["peak_bytes"] <= limits.window_bytes for s in states)
        assert all(c.snapshot()["bytes"] == c.snapshot()["leased_bytes"] == 0 for c in service.channels.values())


@pytest.mark.parametrize("kind", ["inner", "left", "semi", "anti"])
@pytest.mark.parametrize("comparison", ["=", "is not distinct from"])
@pytest.mark.parametrize("partitions", [2, 3])
def test_large_broadcast_keeps_wide_probe_before_aggregation(tmp_path, kind, comparison, partitions):
    with vane.connect(backend="local", config={"threads": 4, "memory_limit": "256MiB"}) as connection:
        sql = large_broadcast_query(connection, tmp_path / "fact.parquet", kind, comparison)
        expected = connection.execute(sql).fetchall()
        spec = prepare_ray_query(
            connection,
            sql,
            query_id="large-broadcast",
            options=query_options(),
            resources=replace(query_resources(), memory=replace(query_resources().memory, operator_bytes=256 << 20)),
            compile_options=FragmentCompileOptions(partitions, worker_operator_bytes=256 << 20),
        )
        limits = DirectExchangeLimits(65536, 16384, 512, 4)
        with InProcessTaskService(connection, spec, limits) as service:
            service.start()
            assert collect(service) == expected
            states = [c.snapshot() for c in service.channels.values()]
            # One complete build plus small partial/final groups; wide probes
            # must not cross a hash edge before their local join/aggregation.
            assert sum(s["accepted_rows"] for s in states) < 1048576 + 1024
            assert all(s["peak_bytes"] <= limits.window_bytes for s in states)
        assert all(c.snapshot()["bytes"] == c.snapshot()["leased_bytes"] == 0 for c in service.channels.values())


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


@pytest.mark.parametrize("tail", ["", " limit 5000 offset 4000"])
def test_parallel_exchange_preserves_ordered_result_batches(tail):
    sql = (
        "select case when range%11=0 then null else range%31 end k, range i "
        "from range(131072) order by k desc nulls first, i desc" + tail
    )
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        expected = connection.execute(sql).fetchall()
        spec = submission(connection, sql, partitions=3)
        with InProcessTaskService(connection, spec, DirectExchangeLimits(8192, 4096, 128, 1)) as service:
            service.start()
            assert collect(service) == expected


def shared_cte_filter_query(connection, path, comparison="=", kind="inner", owner="normal"):
    connection.execute(
        "copy (select case when range%4096=0 then null else range%65536 end k, range%5 g, range%7 v "
        f"from range(131072)) to '{path}' (row_group_size 2048)"
    )
    key = "null::bigint" if owner == "null" else "case when range%17=0 then null else (range%64)*1021 end"
    domain = f"select {key} k, range%3 v from range({0 if owner == 'empty' else 512})"
    predicate = f"f.k {comparison} d.k"
    if kind == "residual":
        kind = "inner"
        predicate += " and f.v>d.v"
    elif kind == "computed":
        kind = "inner"
        predicate = "f.k+f.v=d.k"
    source = f"read_parquet('{path}') f {kind} join d on {predicate}"
    return (
        f"with d as materialized ({domain}) select f.g,sum(f.v) v,count(*) n from {source} group by f.g "
        f"having sum(f.v)>=(select coalesce(sum(f.v),0)*0.001 from {source}) order by f.g"
    )


@pytest.mark.parametrize("threads,partitions", [(1, 2), (4, 2), (4, 5)])
@pytest.mark.parametrize("comparison", ["=", "is not distinct from"])
@pytest.mark.parametrize("kind", ["inner", "semi"])
def test_shared_cte_filters_fact_inputs_before_gather(tmp_path, threads, partitions, comparison, kind):
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        sql = shared_cte_filter_query(connection, tmp_path / "fact.parquet", comparison, kind)
        expected = connection.execute(sql).fetchall()
        assert expected
        spec = submission(connection, sql, partitions=partitions)
        limits = DirectExchangeLimits(4096, 1024, 32, 1)
        with InProcessTaskService(connection, spec, limits) as service:
            service.start()
            assert collect(service) == expected
            states = [c.snapshot() for c in service.channels.values()]
            assert sum(s["accepted_rows"] for s in states) < 8192
            assert all(s["peak_bytes"] <= limits.window_bytes for s in states)
        assert all(c.snapshot()["bytes"] == c.snapshot()["leased_bytes"] == 0 for c in service.channels.values())
        # The materialized CTE producer runs once for its original consumer
        # and both complete-build filters, including its duplicate/NULL keys.
        assert sum(s.function_name == "range" for f in spec.graph.fragments for s in f.sources) == 1


@pytest.mark.parametrize("comparison", ["=", "is not distinct from"])
@pytest.mark.parametrize("owner", ["empty", "null"])
def test_shared_cte_filter_preserves_empty_and_null_builds(tmp_path, comparison, owner):
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        sql = shared_cte_filter_query(connection, tmp_path / "fact.parquet", comparison, owner=owner)
        expected = connection.execute(sql).fetchall()
        with InProcessTaskService(
            connection, submission(connection, sql), DirectExchangeLimits(4096, 1024, 32, 1)
        ) as service:
            service.start()
            assert collect(service) == expected
        assert all(c.snapshot()["bytes"] == c.snapshot()["leased_bytes"] == 0 for c in service.channels.values())


@pytest.mark.parametrize("kind", ["left", "anti", "residual", "computed"])
def test_shared_cte_filter_keeps_preserved_rows_and_unsupported_keys(tmp_path, kind):
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        sql = shared_cte_filter_query(connection, tmp_path / "fact.parquet", kind=kind)
        graph = compile_sql(connection, sql, 2)
        assert not any(e.distribution is Distribution.BROADCAST for e in graph.exchanges)
        assert execute_graph(connection, graph) == connection.execute(sql).fetchall()


@pytest.mark.parametrize(
    "owner_rows,probe_rows,partitions", [(512, 8192, 1), (2048, 8192, 2), (512, 8192, 16), (262144, 8388608, 2)]
)
def test_shared_cte_filter_respects_size_selectivity_and_replica_cost(owner_rows, probe_rows, partitions):
    sql = (
        f"with d as materialized (select range k,repeat('v',64) payload from range({owner_rows})) "
        f"select sum(f.range),max(d.payload) from range({probe_rows}) f join d on f.range=d.k "
        "having sum(f.range)>=(select count(*) from d)"
    )
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        graph = compile_sql(connection, sql, partitions)
        assert not any(e.distribution is Distribution.BROADCAST for e in graph.exchanges)


def test_shared_cte_filter_preserves_nested_owner_dependencies(tmp_path):
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        sql = shared_cte_filter_query(connection, tmp_path / "fact.parquet", "is not distinct from")
        sql = sql.replace("with d as materialized", "with parent as materialized")
        sql = sql.replace(") select f.g", "), d as materialized (select * from parent where v<2) select f.g", 1)
        graph = compile_sql(connection, sql, 2)
        assert not any(e.distribution is Distribution.BROADCAST for e in graph.exchanges)
        assert execute_graph(connection, graph) == connection.execute(sql).fetchall()


def budgeted_correlated_scan_query(connection, path, comparison="=", kind="semi"):
    owner = path.with_stem("owner")
    connection.execute(
        "copy (select case when range%8192=0 then null when range%2048=1024 then range-1024 else range end k, "
        "'priority-'||(range%5)::varchar prio, range%5 t, range%1024=0 is_kept "
        f"from range(262144)) to '{owner}' (row_group_size 2048)"
    )
    connection.execute(
        "copy (select case when range%8192=0 then null else range%262144 end k, range%7 v, 3::bigint w "
        f"from range(1048576)) to '{path}' (row_group_size 2048)"
    )
    negate = "not " if kind == "anti" else ""
    return (
        f"select o.prio,count(*) from read_parquet('{owner}') o where o.is_kept and {negate}exists "
        f"(select 1 from read_parquet('{path}') f where f.k {comparison} o.k and f.v<f.w) "
        "group by o.prio order by o.prio"
    )


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("comparison", ["=", "is not distinct from"])
@pytest.mark.parametrize("kind", ["semi", "anti"])
def test_budgeted_correlated_scan_filters_before_exchange(tmp_path, threads, comparison, kind):
    with vane.connect(backend="local", config={"threads": threads, "memory_limit": "256MiB"}) as connection:
        path = tmp_path / "fact.parquet"
        sql = budgeted_correlated_scan_query(connection, path, comparison, kind)
        expected = connection.execute(sql).fetchall()
        spec = prepare_ray_query(
            connection,
            sql,
            query_id="budgeted-correlated-filter",
            options=query_options(),
            resources=replace(query_resources(), memory=replace(query_resources().memory, operator_bytes=256 << 20)),
            compile_options=FragmentCompileOptions(2, worker_operator_bytes=256 << 20),
        )
        limits = DirectExchangeLimits(4096, 1024, 32, 1)
        with InProcessTaskService(connection, spec, limits=limits) as service:
            service.start()
            assert collect(service) == expected
            states = [c.snapshot() for c in service.channels.values()]
            assert sum(s["accepted_rows"] for s in states) < 4096
            assert all(s["peak_bytes"] <= limits.window_bytes for s in states)
        assert all(c.snapshot()["bytes"] == c.snapshot()["leased_bytes"] == 0 for c in service.channels.values())
        assert sum(s.function_name == "read_parquet" for f in spec.graph.fragments for s in f.sources) == 3


@pytest.mark.parametrize(
    "budget,partitions,filtered", [(0, 2, False), (128 << 20, 2, False), (256 << 20, 2, True), (256 << 20, 5, False)]
)
def test_correlated_scan_filter_respects_budget_and_replica_traffic(tmp_path, budget, partitions, filtered):
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        sql = budgeted_correlated_scan_query(connection, tmp_path / "fact.parquet")
        graph = compile_fragment_graph(
            connection,
            sql,
            query_id="correlated-filter-budget",
            options=FragmentCompileOptions(partitions, worker_operator_bytes=budget),
        )
        assert any(e.distribution is Distribution.BROADCAST for e in graph.exchanges) is filtered


def test_correlated_scan_filter_does_not_cost_residual_selectivity_as_one_key(tmp_path):
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        sql = budgeted_correlated_scan_query(connection, tmp_path / "fact.parquet")
        sql = sql.replace("f.v<f.w", "f.v<f.w and f.v<>o.t")
        graph = compile_fragment_graph(
            connection,
            sql,
            query_id="correlated-filter-residual",
            options=FragmentCompileOptions(2, worker_operator_bytes=256 << 20),
        )
        assert not any(e.distribution is Distribution.BROADCAST for e in graph.exchanges)
        assert execute_graph(connection, graph) == connection.execute(sql).fetchall()


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("join_key", ["p.k", "p.j"])
def test_correlated_average_filters_fact_rows_before_exchange(tmp_path, threads, join_key):
    fact, keys = tmp_path / "fact.parquet", tmp_path / "keys.parquet"
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        connection.execute(
            f"copy (select range%4096 k, range%97 v from range(131072)) to '{fact}' (row_group_size 2048)"
        )
        connection.execute(
            f"copy (select range k, (range+1999)%4096 j, range in (2,2001,4001) keep from range(4096)) to '{keys}'"
        )
        sql = (
            f"select p.k, f.v from read_parquet('{fact}') f join read_parquet('{keys}') p on f.k={join_key} "
            f"where p.keep and f.v < (select 0.2*avg(i.v) from read_parquet('{fact}') i where i.k=p.k) "
            "order by p.k, f.v"
        )
        expected = connection.execute(sql).fetchall()
        limits = DirectExchangeLimits(32768, 8192, 128, 2)
        with InProcessTaskService(connection, submission(connection, sql, partitions=2), limits) as service:
            service.start()
            assert collect(service) == expected
            states = [channel.snapshot() for channel in service.channels.values()]
            # The repeated fact scan can match only keys from the small outer
            # input. Send those rows, not all 131072 facts, into the delimiter.
            assert sum(state["accepted_rows"] for state in states) < 2048
            assert all(state["peak_bytes"] <= limits.window_bytes for state in states)


@pytest.mark.parametrize("partitions", [1, 3])
@pytest.mark.parametrize("kind", ["inner", "left"])
@pytest.mark.parametrize("comparison", ["=", "is not distinct from"])
@pytest.mark.parametrize("key", ["f.k", "p.k"])
def test_correlated_key_filter_preserves_nulls_duplicates_and_outer_rows(tmp_path, partitions, kind, comparison, key):
    fact, keys = tmp_path / "fact.parquet", tmp_path / "keys.parquet"
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        connection.execute("set disabled_optimizers='deliminator'")
        connection.execute(
            "copy (select range i, case when range%37=0 then null else range%1024 end k, "
            f"range%97 v from range(65536)) to '{fact}' (row_group_size 2048)"
        )
        connection.execute(f"copy (select k::bigint k from (values (2), (2), (7), (null), (null)) t(k)) to '{keys}'")
        sql = (
            f"select f.i, p.k, (select avg(i.v) from read_parquet('{fact}') i "
            f"where i.k {comparison} {key}) a from read_parquet('{fact}') f "
            f"{kind} join read_parquet('{keys}') p on f.k {comparison} p.k "
            "where f.i<128 order by f.i, p.k nulls first"
        )
        expected = connection.execute(sql).fetchall()
        limits = DirectExchangeLimits(32768, 8192, 128, 2)
        with InProcessTaskService(connection, submission(connection, sql, partitions=partitions), limits) as service:
            service.start()
            assert collect(service) == expected


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("partitions", [2, 3])
@pytest.mark.parametrize("kind", ["semi", "anti", "nested", "scalar"])
@pytest.mark.parametrize("comparison", ["=", "is not distinct from"])
def test_correlated_owner_keys_filter_before_exchange(tmp_path, threads, partitions, kind, comparison):
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        sql = correlated_owner_queries(connection, tmp_path / "fact.parquet", comparison)[kind]
        expected = connection.execute(sql).fetchall()
        limits = DirectExchangeLimits(4096, 1024, 64, 1)
        spec = submission(connection, sql, partitions=partitions)
        with InProcessTaskService(connection, spec, limits) as service:
            service.start()
            assert collect(service) == expected
            states = [channel.snapshot() for channel in service.channels.values()]
            assert sum(state["accepted_rows"] for state in states) < 32768
            assert all(state["peak_bytes"] <= limits.window_bytes for state in states)
        # Sharing the owner must not execute its source a second time.
        assert sum(s.function_name == "range" for f in spec.graph.fragments for s in f.sources) == 1


def budgeted_owner_filter_query(connection, path, kind="semi", comparison="="):
    sql = correlated_owner_queries(connection, path, comparison)[kind]
    # The owner has only 320 rows, but a 1M-row estimate (a 9 MB key
    # domain). It cannot be replaced by an independent Parquet scan.
    return sql.replace("range(65536) where range%31=0", "range(5000000) where range%15625=0").replace(
        f"read_parquet('{path}')", f"read_parquet('{path}', explicit_cardinality=100000000)"
    )


@pytest.mark.parametrize("partitions", [2, 3])
@pytest.mark.parametrize("kind", ["semi", "anti", "nested", "scalar"])
@pytest.mark.parametrize("comparison", ["=", "is not distinct from"])
def test_declared_budget_filters_shared_correlated_owner(tmp_path, partitions, kind, comparison):
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        connection.execute("set disabled_optimizers='deliminator,compressed_materialization'")
        path = tmp_path / "fact.parquet"
        sql = budgeted_owner_filter_query(connection, path, kind, comparison)
        expected = connection.execute(sql).fetchall()
        spec = prepare_ray_query(
            connection,
            sql,
            query_id="budgeted-owner-filter",
            options=query_options(),
            resources=replace(query_resources(), memory=replace(query_resources().memory, operator_bytes=256 << 20)),
            compile_options=FragmentCompileOptions(partitions, worker_operator_bytes=256 << 20),
        )
        facts = {
            f.fragment_id
            for f in spec.graph.fragments
            if any(str(path).encode() in split.payload for s in f.sources for split in s.splits)
        }
        edges = {e.exchange_id for e in spec.graph.exchanges if e.producer_fragment_id in facts}
        limits = DirectExchangeLimits(4096, 1024, 32, 1)
        with InProcessTaskService(connection, spec, limits) as service:
            service.start()
            assert collect(service) == expected
            assert edges
            assert (
                sum(
                    c.snapshot()["accepted_rows"] for name, c in service.channels.items() if name.split("/")[0] in edges
                )
                < 4096
            )
            assert all(c.snapshot()["peak_bytes"] <= limits.window_bytes for c in service.channels.values())
        assert sum(s.function_name == "range" for f in spec.graph.fragments for s in f.sources) == 1
        assert all(c.snapshot()["bytes"] == c.snapshot()["leased_bytes"] == 0 for c in service.channels.values())


@pytest.mark.parametrize("budget,filtered", [(0, False), (128 << 20, False), (256 << 20, True)])
def test_shared_correlated_owner_filter_respects_declared_budget(tmp_path, budget, filtered):
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        connection.execute("set disabled_optimizers='deliminator,compressed_materialization'")
        sql = budgeted_owner_filter_query(connection, tmp_path / "fact.parquet")
        graph = compile_fragment_graph(
            connection,
            sql,
            query_id="owner-filter-budget",
            options=FragmentCompileOptions(2, worker_operator_bytes=budget),
        )
        assert any(e.distribution is Distribution.BROADCAST for e in graph.exchanges) is filtered


def budgeted_hash_filter_query(connection, directory, comparison="="):
    fact, build = directory / "fact.parquet", directory / "build.parquet"
    connection.execute(
        "copy (select case when range%8192=0 then null else range-524288 end k, range%17 v "
        f"from range(1048576)) to '{fact}' (row_group_size 2048)"
    )
    connection.execute(
        "copy (select case when range%8192=0 then null else (range%65536)*3-524288 end k, "
        f"(range%32=0)::integer v from range(131072)) to '{build}' (row_group_size 2048)"
    )
    return (
        f"select count(*),sum(f.v),sum(b.s) from read_parquet('{fact}', explicit_cardinality=1000000000) f join "
        f"(select k,sum(v) s from read_parquet('{build}', explicit_cardinality=100000000) "
        f"group by k having sum(v)>0) b on f.k {comparison} b.k"
    )


@pytest.mark.parametrize("partitions", [2, 3])
def test_hash_domain_uses_declared_memory_to_reduce_collisions(tmp_path, partitions):
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        connection.execute("set disabled_optimizers='join_order,build_side_probe_side,compressed_materialization'")
        sql = budgeted_hash_filter_query(connection, tmp_path)
        expected = connection.execute(sql).fetchall()
        spec = prepare_ray_query(
            connection,
            sql,
            query_id="budgeted-hash-domain",
            options=query_options(),
            resources=replace(query_resources(), memory=replace(query_resources().memory, operator_bytes=1 << 30)),
            compile_options=FragmentCompileOptions(partitions, worker_operator_bytes=1 << 30),
        )
        facts = {
            f.fragment_id
            for f in spec.graph.fragments
            if any(str(tmp_path / "fact.parquet").encode() in split.payload for s in f.sources for split in s.splits)
        }
        edges = {e.exchange_id for e in spec.graph.exchanges if e.producer_fragment_id in facts}
        limits = DirectExchangeLimits(16384, 4096, 128, 2)
        with InProcessTaskService(connection, spec, limits) as service:
            service.start()
            assert collect(service) == expected
            assert edges
            assert (
                sum(
                    c.snapshot()["accepted_rows"] for name, c in service.channels.items() if name.split("/")[0] in edges
                )
                < 4096
            )
            assert all(c.snapshot()["peak_bytes"] <= limits.window_bytes for c in service.channels.values())
        assert all(c.snapshot()["bytes"] == c.snapshot()["leased_bytes"] == 0 for c in service.channels.values())


@pytest.mark.parametrize("partitions", [2, 3])
@pytest.mark.parametrize("budget", [0, 256 << 20, 1 << 30])
def test_hash_domain_keeps_useful_filters_for_medium_probes(tmp_path, partitions, budget):
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        path = tmp_path / "fact.parquet"
        sql = large_join_key_query(connection, path, "hashed")
        expected = connection.execute(sql).fetchall()
        spec = prepare_ray_query(
            connection,
            sql,
            query_id="medium-probe-domain",
            options=query_options(),
            resources=replace(query_resources(), memory=replace(query_resources().memory, operator_bytes=1 << 30)),
            compile_options=FragmentCompileOptions(partitions, worker_operator_bytes=budget),
        )
        facts = {
            f.fragment_id
            for f in spec.graph.fragments
            if any(str(path).encode() in split.payload for s in f.sources for split in s.splits)
        }
        edges = {e.exchange_id for e in spec.graph.exchanges if e.producer_fragment_id in facts}
        limits = DirectExchangeLimits(16384, 4096, 128, 2)
        with InProcessTaskService(connection, spec, limits) as service:
            service.start()
            assert collect(service) == expected
            assert edges
            assert (
                sum(
                    c.snapshot()["accepted_rows"] for name, c in service.channels.items() if name.split("/")[0] in edges
                )
                < 65536
            )
            assert all(c.snapshot()["peak_bytes"] <= limits.window_bytes for c in service.channels.values())
        assert all(c.snapshot()["bytes"] == c.snapshot()["leased_bytes"] == 0 for c in service.channels.values())


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("partitions", [2, 3])
@pytest.mark.parametrize("domain", ["exact", "hashed", "estimated", "compressed", "join_estimate"])
def test_key_domain_filters_large_build_before_hash_exchange(tmp_path, threads, partitions, domain):
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        path = tmp_path / "fact.parquet"
        sql = large_join_key_query(connection, path, domain)
        expected = connection.execute(sql).fetchall()
        spec = submission(connection, sql, partitions=partitions)
        sources = {
            f.fragment_id
            for f in spec.graph.fragments
            if any(str(path).encode() in split.payload for s in f.sources for split in s.splits)
        }
        fact_edges = {e.exchange_id for e in spec.graph.exchanges if e.producer_fragment_id in sources}
        limits = (
            DirectExchangeLimits(65536, 16384, 256, 2)
            if domain in ("estimated", "compressed", "join_estimate")
            else DirectExchangeLimits(4096, 1024, 64, 1)
        )
        with InProcessTaskService(connection, spec, limits) as service:
            service.start()
            assert collect(service) == expected
            states = [c.snapshot() for c in service.channels.values()]
            fact_rows = sum(
                c.snapshot()["accepted_rows"]
                for name, c in service.channels.items()
                if name.split("/")[0] in fact_edges
            )
            assert fact_edges and fact_rows < 65536
            assert all(s["peak_bytes"] <= limits.window_bytes for s in states)
        assert all(c.snapshot()["bytes"] == 0 for c in service.channels.values())
        assert sum(s.function_name == "range" for f in spec.graph.fragments for s in f.sources) == (
            0 if domain in ("compressed", "join_estimate") else 1
        )


@pytest.mark.parametrize("shape", ["single", "multiple", "narrow"])
def test_join_output_filter_estimate_requires_single_key_and_traffic_savings(tmp_path, shape):
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        sql = large_join_key_query(connection, tmp_path / "fact.parquet", "join_estimate")
        if shape == "multiple":
            sql = sql.replace("on p.k = b.k", "on p.k = b.k and p.i=b.active")
        elif shape == "narrow":
            sql = sql.replace("p.payload, p.extra, ", "")
        graph = compile_sql(connection, sql, partitions=2)
        assert any(e.distribution is Distribution.BROADCAST for e in graph.exchanges) == (shape == "single")
        assert execute_graph(connection, graph) == connection.execute(sql).fetchall()


@pytest.mark.parametrize("domain", ["dense", "empty"])
@pytest.mark.parametrize("budget,bound", [(0, 65536), (256 << 20, 131072)])
def test_hashed_key_domain_is_bounded_and_preserves_exact_matches(tmp_path, domain, budget, bound):
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        path = tmp_path / "fact.parquet"
        sql = large_join_key_query(connection, path, domain) + " limit 4096"
        sql = sql.replace(f"read_parquet('{path}')", f"read_parquet('{path}', explicit_cardinality=100000000)")
        expected = connection.execute(sql).fetchall()
        spec = prepare_ray_query(
            connection,
            sql,
            query_id="bounded-hash-domain",
            options=query_options(),
            resources=replace(query_resources(), memory=replace(query_resources().memory, operator_bytes=256 << 20)),
            compile_options=FragmentCompileOptions(3, worker_operator_bytes=budget),
        )
        limits = DirectExchangeLimits(4096, 1024, 64, 1)
        with InProcessTaskService(connection, spec, limits) as service:
            service.start()
            assert collect(service) == expected
            broadcasts = [e.exchange_id for e in spec.graph.exchanges if e.distribution is Distribution.BROADCAST]
            assert broadcasts
            for edge in broadcasts:
                assert service.channels[edge].snapshot()["accepted_rows"] <= bound


@pytest.mark.parametrize("condition", ["multiple", "repeated", "residual"])
def test_key_domain_keeps_original_join_conditions(tmp_path, condition):
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        sql = large_join_key_query(connection, tmp_path / "fact.parquet")
        if condition == "multiple":
            sql = sql.replace("k, 'build-'", "k, range%7 j, 'build-'")
            extra = "p.i%7 = b.j"
        elif condition == "repeated":
            extra = "p.i%65536 = b.k"
        else:
            extra = "p.i > length(b.v)*1000"
        sql = sql.replace("order by all", f"and {extra} order by all")
        expected = connection.execute(sql).fetchall()
        limits = DirectExchangeLimits(4096, 1024, 64, 1)
        with InProcessTaskService(connection, submission(connection, sql, partitions=3), limits) as service:
            service.start()
            assert collect(service) == expected


@pytest.mark.parametrize("domain", ["exact", "hashed", "join_estimate"])
@pytest.mark.parametrize("kind", ["inner", "left", "anti"])
def test_key_domain_preserves_null_safe_and_outer_matches(tmp_path, domain, kind):
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        sql = large_join_key_query(connection, tmp_path / "fact.parquet", domain, kind, "is not distinct from")
        sql += " limit 4096"
        expected = connection.execute(sql).fetchall()
        limits = DirectExchangeLimits(4096, 1024, 64, 1)
        with InProcessTaskService(connection, submission(connection, sql, partitions=2), limits) as service:
            service.start()
            assert collect(service) == expected


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("partitions", [2, 3])
@pytest.mark.parametrize("kind", ["inner", "semi"])
def test_selective_broadcast_filters_before_earlier_hash_exchange(tmp_path, threads, partitions, kind):
    config = {"threads": threads, "disabled_optimizers": "join_order,build_side_probe_side"}
    with vane.connect(backend="local", config=config) as connection:
        sql = early_join_filter_query(connection, tmp_path / "fact.parquet", kind)
        expected = connection.execute(sql).fetchall()
        spec = submission(connection, sql, partitions=partitions)
        limits = DirectExchangeLimits(4096, 1024, 64, 1)
        with InProcessTaskService(connection, spec, limits) as service:
            service.start()
            assert collect(service) == expected
            states = [channel.snapshot() for channel in service.channels.values()]
            # The wide order side remains partitioned, but nearly all fact rows
            # must be removed before crossing the hash boundary.
            assert sum(state["accepted_rows"] for state in states) < 140000
            assert all(state["peak_bytes"] <= limits.window_bytes for state in states)
        assert sum(s.function_name == "range" for f in spec.graph.fragments for s in f.sources) == 2


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("query_kind", ["correlated", "join", "exact", "hashed", "join_estimate", "cte", "build"])
def test_shared_build_fanout_releases_blocked_branches_on_close(tmp_path, threads, query_kind):
    # A native teardown deadlock must fail under an independent process deadline.
    script = textwrap.dedent(
        """
        import sys
        import time
        from pathlib import Path
        import vane
        from tests.fast.test_direct_exchange import submission
        from tests.fast.test_fragment_sql_coverage import (
            correlated_owner_queries, early_join_filter_query, large_join_key_query,
        )
        from tests.fast.test_execution_performance import shared_cte_filter_query
        from tests.fast.test_join_build_filter import shared_join_build_query
        from vane.execution.direct_exchange import DirectExchangeLimits, InProcessTaskService

        config = {"threads": int(sys.argv[2])}
        if sys.argv[3] == "join":
            config["disabled_optimizers"] = "join_order,build_side_probe_side"
        with vane.connect(backend="local", config=config) as connection:
            path = Path(sys.argv[1]) / "fact.parquet"
            if sys.argv[3] == "build":
                sql, _, _ = shared_join_build_query(connection, path.parent)
            elif sys.argv[3] == "cte":
                sql = shared_cte_filter_query(connection, path)
            elif sys.argv[3] in ("exact", "hashed", "join_estimate"):
                sql = large_join_key_query(connection, path, sys.argv[3])
            else:
                sql = (early_join_filter_query(connection, path) if sys.argv[3] == "join"
                       else correlated_owner_queries(connection, path)["nested"])
            limits = DirectExchangeLimits(4096, 1024, 4, 1)
            spec = submission(connection, sql, partitions=3)
            with InProcessTaskService(connection, spec, limits) as service:
                service.start()
                deadline = time.monotonic() + 10
                while not any(c.snapshot()["write_blocks"] for c in service.channels.values()):
                    assert time.monotonic() < deadline, service.snapshot()
                    service.pump(1)
                service.cancel("cancel shared build branches")
            assert all(c.snapshot()["bytes"] == 0 for c in service.channels.values())
        """
    )
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(threads), query_kind], check=True, timeout=30, cwd=tmp_path
    )


@pytest.mark.parametrize("kind", ["left", "anti"])
@pytest.mark.parametrize("comparison", ["=", "is not distinct from"])
def test_early_join_filter_preserves_outer_and_null_matches(tmp_path, kind, comparison):
    config = {"threads": 4, "disabled_optimizers": "join_order,build_side_probe_side"}
    with vane.connect(backend="local", config=config) as connection:
        sql = early_join_filter_query(connection, tmp_path / "fact.parquet", kind, comparison)
        expected = connection.execute(sql).fetchall()
        limits = DirectExchangeLimits(4096, 1024, 64, 1)
        with InProcessTaskService(connection, submission(connection, sql, partitions=3), limits) as service:
            service.start()
            assert collect(service) == expected


@pytest.mark.parametrize("boundary", ["computed", "limit", "aggregate", "multiple"])
def test_early_join_filter_respects_projection_and_cardinality_boundaries(tmp_path, boundary):
    disabled = "join_order,build_side_probe_side"
    with vane.connect(backend="local", config={"threads": 4, "disabled_optimizers": disabled}) as connection:
        sql = early_join_filter_query(connection, tmp_path / "fact.parquet")
        if boundary == "computed":
            sql = sql.replace("on a.k = b.k", "on a.k%1024 = b.k")
        elif boundary == "limit":
            sql = sql.replace("o using(i)) a", "o using(i) order by f.i limit 1) a")
        elif boundary == "aggregate":
            sql = sql.replace("select f.i, f.k, o.payload", "select min(f.i) i, f.k, max(o.payload) payload")
            sql = sql.replace("o using(i)) a", "o using(i) group by f.k) a").replace("on a.k = b.k", "on a.i = b.k")
        else:
            sql = sql.replace(
                "order by a.i, a.payload",
                "join (select range k from range(4096) where range%1024=0) c on a.k=c.k order by a.i, a.payload",
            )
        # An oracle without dynamic filters also checks that moving a necessary
        # condition does not change LIMIT or aggregate input semantics.
        connection.execute(f"set disabled_optimizers='{disabled},join_filter_pushdown'")
        expected = connection.execute(sql).fetchall()
        if boundary == "limit":
            assert expected == []  # The first fact has a NULL key.
        connection.execute(f"set disabled_optimizers='{disabled}'")
        limits = DirectExchangeLimits(4096, 1024, 64, 1)
        with InProcessTaskService(connection, submission(connection, sql, partitions=3), limits) as service:
            service.start()
            assert collect(service) == expected
            if boundary == "multiple":
                assert sum(c.snapshot()["accepted_rows"] for c in service.channels.values()) < 140000
