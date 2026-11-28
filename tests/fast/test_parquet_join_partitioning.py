# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Partitioned equality joins using a common Parquet key-range assignment."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import vane
from tests.fast.test_native_fragment_compiler import compile_sql, execute_graph
from vane.execution.plan import Distribution, FragmentGraph


def parquet_join_query(directory, layout="balanced"):
    for component in range(4):
        base = 0 if layout == "one_component" else 10 * component
        # Each right interval connects two otherwise disjoint left files at
        # their endpoints. Both endpoints have duplicates on both sides.
        for part, offsets in enumerate(([0, 2, 2], [4, 4, 6], [2, 2, 3, 4, 4])):
            left = part < 2
            repeats = (4 + component % 2) if left else (5 - component % 2)
            if layout == "skewed":
                repeats = 90 if left else (20 if component == 0 else 1)
            keys = [base + k for k in offsets] * repeats
            if layout == "nulls" and not left and component == 0:
                keys[0] = None
            typ = pa.int32() if layout == "mixed_types" and not left else pa.int64()
            table = pa.table({"v": list(range(len(keys))), "k": pa.array(keys, type=typ)})
            side = "a" if left else "b"
            pq.write_table(
                table,
                directory / f"{side}-{component}-{part}.parquet",
                row_group_size=len(offsets),
                write_statistics=layout != "missing_stats" or left or component != 0,
            )
    # Cost a shuffle join without making the fixture itself large. Placement
    # must still use the actual row counts in each split's footer.
    a_files = str(directory / "a-*.parquet")
    a_paths = repr([a_files, a_files]) if layout == "repeated" else repr(a_files)
    a = f"read_parquet({a_paths}, explicit_cardinality=100000000)"
    b = f"read_parquet('{directory}/b-*.parquet', explicit_cardinality=100000000)"
    if layout == "broadcast":
        a = f"(select l.v, l.k from {a} l join (values (1),(1),(2)) d(v) on l.v % 3 = d.v where l.v > 0)"
    key = "l.k + 1" if layout == "computed" else "l.k"
    joined = f"{a} l join {b} r on {key} = r.k"
    if layout == "downstream":
        joined += f" join {b} t on l.k = t.k"
    return f"select count(*) n, sum(l.v * r.v) total from {joined}"


@pytest.mark.parametrize(
    "layout,partitions,placed",
    [
        ("balanced", 2, True),
        ("balanced", 4, True),
        ("balanced", 3, False),
        ("balanced", 8, False),
        ("repeated", 2, True),
        ("broadcast", 2, True),
        ("downstream", 2, True),
        ("mixed_types", 2, False),
        ("computed", 2, False),
        ("nulls", 2, False),
        ("missing_stats", 2, False),
        ("skewed", 2, False),
        ("one_component", 2, False),
    ],
)
def test_joint_ranges_preserve_all_matches(tmp_path, layout, partitions, placed):
    sql = parquet_join_query(tmp_path, layout)
    with vane.connect(
        backend="local", config={"disabled_optimizers": "join_order,build_side_probe_side"}
    ) as connection:
        expected = connection.execute(sql).fetchall()
        assert expected[0][0] > 0
        graph = compile_sql(connection, sql, partitions)
        assert FragmentGraph.from_dict(graph.to_dict(), expected_engine_identity=graph.engine_identity) == graph
        owners = {f.fragment_id for f in graph.fragments if any(s.split_partitions for s in f.sources)}
        direct = [e for e in graph.exchanges if e.distribution is Distribution.PASSTHROUGH]
        if placed:
            assert len(owners) == len(direct) == 2
            assert {e.producer_fragment_id for e in direct} == owners
            assert len({e.consumer_fragment_id for e in direct}) == 1
        else:
            assert not owners and not direct
        assert sum(e.distribution is Distribution.HASH for e in graph.exchanges) == (
            2 if not placed or layout == "downstream" else 0
        )
        if layout == "broadcast":
            assert any(
                e.distribution is Distribution.BROADCAST and e.consumer_fragment_id in owners for e in graph.exchanges
            )
        assert execute_graph(connection, graph) == expected
