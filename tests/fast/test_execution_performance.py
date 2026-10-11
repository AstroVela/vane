# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Execution work scales with groups and frames rather than input batches."""

import time
from collections import Counter

import pytest

import vane
from tests.fast.test_direct_exchange import collect, submission
from tests.fast.test_native_fragment_compiler import compile_sql, execute_graph
from vane._native import execution_plan as native
from vane.execution.direct_exchange import DirectExchangeLimits, InProcessTaskService
from vane.execution.pipelined_plan import placement
from vane.execution.pipelined_worker import _Query
from vane.execution.worker_resources import pipelined_demand


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
