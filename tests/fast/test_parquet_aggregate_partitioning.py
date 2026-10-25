# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Complete groups proven from the exact Parquet split ranges."""

from collections import Counter
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import vane
from tests.fast.test_direct_exchange import collect, submission
from tests.fast.test_native_fragment_compiler import compile_sql, execute_graph
from vane.execution.direct_exchange import DirectExchangeLimits, InProcessTaskService
from vane.execution.plan import Distribution, FragmentGraph, SourceSpec


def disjoint_aggregate_query(directory, variant="disjoint"):
    # Four rows per group, with ranges too wide for native key compression.
    keys = [(i // 4) * 10**15 for i in range(16384)]
    values = [100 if i // 4 % 512 == 0 else 1 for i in range(16384)]
    if variant.startswith("compressed"):
        offset = {"compressed": 0, "compressed_negative": -(2**62), "compressed_unsigned": 2**63}[variant]
        keys = [i // 4 + offset for i in range(16384)]
    elif variant == "overlap":
        keys[8192] = keys[8191]
    elif variant == "nulls":
        for i in [*range(4), *range(8192, 8196)]:
            keys[i] = None
    elif variant == "unsorted":
        keys[8192:] = keys[:8192]
    elif variant == "unsigned":
        keys = [key + 2**63 for key in keys]
    elif variant == "negative":
        keys = [key - 2**62 for key in keys]
    key_type = pa.uint64() if variant in {"unsigned", "compressed_unsigned"} else pa.int64()
    table = pa.table({"k": pa.array(keys, type=key_type), "v": values})
    path = directory / "groups.parquet"
    pq.write_table(table, path, row_group_size=2048, write_statistics=variant != "missing_stats")
    files = f"'{path}'"
    if variant == "repeated":
        files = f"['{path}', '{path}']"
    elif variant in {"multiple", "empty_file"}:
        other = directory / "other.parquet"
        split = 8192 if variant == "multiple" else 0
        pq.write_table(table.slice(0, split), path, row_group_size=2048)
        pq.write_table(table.slice(split), other, row_group_size=2048)
        files = f"['{path}', '{other}']"
    key = {"computed": "k % 4000000000000000", "cast": "k::double"}.get(variant, "k")
    source = f"select v, {key} k from read_parquet({files}) where v > 0"
    return (
        f"select k, sum(v::decimal(15,2)) s, count(*) n from ({source}) "
        "group by k having sum(v::decimal(15,2)) > 300 order by k nulls last"
    )


def overlapping_components_query(directory, variant="overlap"):
    for component in range(6):
        offset = component * 10**15 if variant != "one_component" else 0
        count = 16384 * (12 if variant == "skewed" and component == 0 else 1)
        for copy in range(2):
            keys = [offset + i % 2048 for i in range(count)]
            if variant == "nulls" and component == copy == 0:
                keys[0] = None
            table = pa.table(
                {"k": pa.array(keys, type=pa.int64()), "v": [25 if i % 1024 == 0 else 1 for i in range(count)]}
            )
            pq.write_table(
                table,
                directory / f"component-{component}-{copy}.parquet",
                row_group_size=2048,
                write_statistics=variant != "missing_stats" or component != 0,
            )
    files = str(directory / "component-*.parquet")
    source = f"['{files}', '{files}']" if variant == "repeated" else f"'{files}'"
    key = {"computed": "k%2", "cast": "k::double"}.get(variant, "k")
    return (
        f"select {key} k, sum(v) s, count(*) n from read_parquet({source}) "
        f"group by {key} having sum(v)>300 order by k nulls last"
    )


@pytest.mark.parametrize("partitions", [2, 3])
@pytest.mark.parametrize("threads", [1, 4])
def test_overlapping_components_aggregate_before_exchange(tmp_path, partitions, threads):
    sql = overlapping_components_query(tmp_path)
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        expected = connection.execute(sql).fetchall()
        assert len(expected) == 12 and all(row[1:] == (400, 16) for row in expected)
        graph = compile_sql(connection, sql, partitions)
        assert all(edge.distribution is Distribution.GATHER for edge in graph.exchanges)
        assert execute_graph(connection, graph) == expected
        limits = DirectExchangeLimits(4096, 2048, 32, 2)
        with InProcessTaskService(connection, submission(connection, sql, partitions=partitions), limits) as service:
            service.start()
            assert collect(service) == expected
            assert sum(c.snapshot()["accepted_rows"] for c in service.channels.values()) == 2 * len(expected)
            assert all(c.snapshot()["peak_bytes"] <= limits.window_bytes for c in service.channels.values())
        assert all(c.snapshot()["bytes"] == c.snapshot()["leased_bytes"] == 0 for c in service.channels.values())


@pytest.mark.parametrize(
    "variant,complete",
    [
        ("repeated", True),
        ("nulls", False),
        ("missing_stats", False),
        ("skewed", False),
        ("one_component", False),
        ("computed", False),
        ("cast", False),
    ],
)
def test_component_assignment_requires_complete_balanced_key_ranges(tmp_path, variant, complete):
    sql = overlapping_components_query(tmp_path, variant)
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        expected = connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, 2)
        assert execute_graph(connection, graph) == expected
        assert any(e.distribution is Distribution.HASH for e in graph.exchanges) is not complete
        assert any(s.split_partitions for f in graph.fragments for s in f.sources) is complete
        assert FragmentGraph.from_dict(graph.to_dict(), expected_engine_identity=graph.engine_identity) == graph


def test_component_assignment_rejects_partial_or_mixed_native_partition(tmp_path):
    sql = overlapping_components_query(tmp_path)
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        graph = compile_sql(connection, sql, 2)
        fragment = next(f for f in graph.fragments if any(s.split_partitions for s in f.sources))
        source = fragment.sources[0]
        assert SourceSpec.from_dict(source.to_dict()) == source
        owned = [source.assigned_split_ids(p, 2) for p in range(2)]
        assert set(owned[0]).isdisjoint(owned[1])
        assert set(owned[0] + owned[1]) == {s.split_id for s in source.splits}
        from vane._native import execution_plan as native

        for ids in (owned[0][:-1], [owned[0][0], owned[1][0]]):
            with pytest.raises(vane.InvalidInputException, match="complete partition"):
                native._execute_fragment_for_test(connection, fragment.native_plan, {}, {source.source_id: ids})
        for owners in (source.split_partitions[:-1], (-1,) * len(source.splits), (True,) * len(source.splits)):
            with pytest.raises(ValueError, match="split_partitions"):
                replace(source, split_partitions=owners)
        with pytest.raises(ValueError, match="out of range"):
            replace(fragment, sources=(replace(source, split_partitions=(2,) * len(source.splits)),))


@pytest.mark.parametrize(
    "aggregates",
    [
        "count(distinct v), sum(distinct v), avg(v) filter(where v>1)",
        "min(v,2), max(v,2), count(*) filter(where v<0)",
        "sum(v::double order by v desc), avg(v::double order by v)",
        "sum(v::decimal(38,2)) filter(where v>1), avg(v::hugeint)",
    ],
)
def test_owned_component_groups_keep_complete_aggregate_semantics(tmp_path, aggregates):
    sql = overlapping_components_query(tmp_path).replace("sum(v) s, count(*) n", aggregates)
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        expected = connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, 3)
        assert all(e.distribution is Distribution.GATHER for e in graph.exchanges)
        assert any(s.split_partitions for f in graph.fragments for s in f.sources)
        assert execute_graph(connection, graph) == expected


def broadcast_aggregate_query(directory, *, join="inner", key="a.k", variant="compressed"):
    disjoint_aggregate_query(directory, "compressed")
    path = directory / "groups.parquet"
    if variant in {"overlap", "nulls", "missing_stats"}:
        table = pq.read_table(path)
        keys = table["k"].to_pylist()
        if variant == "overlap":
            keys[8192] = keys[0]
        elif variant == "nulls":
            for index in [*range(4), *range(8192, 8196)]:
                keys[index] = None
        table = table.set_column(0, "k", pa.array(keys, type=pa.int64()))
        pq.write_table(table, path, row_group_size=2048, write_statistics=variant != "missing_stats")
    files = f"['{path}', '{path}']" if variant == "repeated" else f"'{path}'"
    # Duplicate build keys exercise probe fanout. LEFT also retains unmatched
    # probe rows; grouping a build key must still combine all probe partitions.
    return (
        f"select {key} k, sum(a.v) s, count(*) n from read_parquet({files}) a "
        f"{join} join (values (0), (1), (2), (0), (1), (9999)) b(k) on a.k % 7 = b.k "
        f"group by {key} having sum(a.v) > 300 order by k nulls last"
    )


@pytest.mark.parametrize("variant", ["compressed", "compressed_negative", "compressed_unsigned"])
@pytest.mark.parametrize("partitions", [2, 3])
def test_complete_parquet_groups_preserve_integral_compression(tmp_path, variant, partitions):
    sql = disjoint_aggregate_query(tmp_path, variant)
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, partitions)
        assert execute_graph(connection, graph) == expected
        assert all(edge.distribution is Distribution.GATHER for edge in graph.exchanges)


@pytest.mark.parametrize("join", ["inner", "left", "semi", "anti"])
@pytest.mark.parametrize("partitions", [2, 3])
def test_broadcast_join_preserves_complete_probe_groups(tmp_path, join, partitions):
    sql = broadcast_aggregate_query(tmp_path, join=join)
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        expected = connection.execute(sql).fetchall()
        spec = submission(connection, sql, partitions=partitions)
        assert any(edge.distribution is Distribution.BROADCAST for edge in spec.graph.exchanges)
        assert all(edge.distribution is not Distribution.HASH for edge in spec.graph.exchanges)
        limits = DirectExchangeLimits(window_bytes=4096, frame_bytes=2048, frame_rows=32, frame_slots=2)
        with InProcessTaskService(connection, spec, limits) as service:
            service.start()
            assert collect(service) == expected
            assert all(channel.snapshot()["peak_bytes"] <= limits.window_bytes for channel in service.channels.values())


@pytest.mark.parametrize(
    "options",
    [
        {"key": "b.k"},
        {"key": "a.k % 2"},
        {"key": "a.k::double"},
        {"join": "full outer"},
        {"join": "left", "variant": "overlap"},
        {"join": "left", "variant": "nulls"},
        {"join": "left", "variant": "missing_stats"},
        {"join": "left", "variant": "repeated"},
    ],
)
def test_join_groups_require_preserved_key_and_split_evidence(tmp_path, options):
    sql = broadcast_aggregate_query(tmp_path, **options)
    if options.get("join") == "full outer":
        sql = sql.replace("having sum(a.v) > 300", "")
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, 3)
        assert execute_graph(connection, graph) == expected
        assert any(edge.distribution is Distribution.HASH for edge in graph.exchanges) == (
            options.get("variant") != "repeated"
        )


@pytest.mark.parametrize("build", ["select range from range(0)", "select * from (values (0), (null)) b(k)"])
def test_broadcast_mark_preserves_complete_probe_groups(tmp_path, build):
    disjoint_aggregate_query(tmp_path, "compressed")
    sql = (
        f"select k, k in ({build}) m, sum(v) from read_parquet('{tmp_path / 'groups.parquet'}') "
        "group by k, m having sum(v) > 300 order by k"
    )
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, 3)
        assert execute_graph(connection, graph) == expected
        assert all(edge.distribution is not Distribution.HASH for edge in graph.exchanges)


@pytest.mark.parametrize(
    "aggregate",
    [
        "sum(a.v::double order by a.v desc), count(distinct a.v)",
        "min(a.v, 2), sum(a.v::decimal(38,2)), avg(a.v::hugeint)",
    ],
)
def test_broadcast_complete_groups_keep_native_aggregate_semantics(tmp_path, aggregate):
    sql = broadcast_aggregate_query(tmp_path, join="left").replace("sum(a.v) s, count(*) n", aggregate)
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, 3)
        assert execute_graph(connection, graph) == expected
        assert all(edge.distribution is not Distribution.HASH for edge in graph.exchanges)


@pytest.mark.parametrize("partitions", [2, 3, 5])
@pytest.mark.parametrize("threads", [1, 4])
def test_complete_parquet_groups_filter_before_exchange(tmp_path, partitions, threads):
    sql = disjoint_aggregate_query(tmp_path)
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        expected = connection.execute(sql).fetchall()
        assert len(expected) == 8
        spec = submission(connection, sql, partitions=partitions)
        limits = DirectExchangeLimits(window_bytes=4096, frame_bytes=2048, frame_rows=32, frame_slots=2)
        with InProcessTaskService(connection, spec, limits) as service:
            service.start()
            assert collect(service) == expected
            # Only qualifying groups and final results need to be transported.
            assert sum(c.snapshot()["accepted_rows"] for c in service.channels.values()) == 2 * len(expected)
            assert all(c.snapshot()["peak_bytes"] <= limits.window_bytes for c in service.channels.values())


@pytest.mark.parametrize(
    "variant,complete",
    [
        ("unsigned", True),
        ("negative", True),
        ("multiple", True),
        ("empty_file", True),
        ("overlap", False),
        ("nulls", False),
        ("unsorted", False),
        ("missing_stats", False),
        ("repeated", True),
        ("computed", False),
        ("cast", False),
    ],
)
def test_parquet_group_ranges_require_complete_evidence(tmp_path, variant, complete):
    sql = disjoint_aggregate_query(tmp_path, variant)
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = Counter(connection.execute(sql).fetchall())
        graph = compile_sql(connection, sql, 2)
        assert Counter(execute_graph(connection, graph)) == expected
        assert any(edge.distribution is Distribution.HASH for edge in graph.exchanges) != complete


@pytest.mark.parametrize(
    "aggregate",
    [
        "count(distinct v), sum(distinct v), avg(v) filter(where v > 1)",
        "min(v, 2), max(v, 2), count(*) filter(where v < 0)",
        "sum(v::double order by v desc), avg(v::double order by v)",
        "sum(v::decimal(38,2)) filter(where v > 1), avg(v::hugeint)",
    ],
)
def test_complete_parquet_groups_preserve_aggregate_semantics(tmp_path, aggregate):
    disjoint_aggregate_query(tmp_path)
    path = tmp_path / "groups.parquet"
    # A second nullable grouping key cannot merge groups whose first key is
    # already disjoint. Ordered, distinct and wide accumulators stay native.
    sql = (
        f"select k, v % 3 g, {aggregate} from "
        f"(select k, case when k % 3 = 0 then null else v end v from read_parquet('{path}')) "
        "group by k, g"
    )
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, 3)
        actual = execute_graph(connection, graph)
        assert sorted(actual, key=lambda row: row[0]) == sorted(expected, key=lambda row: row[0])
        assert all(edge.distribution is Distribution.GATHER for edge in graph.exchanges)


def test_group_ranges_do_not_cross_join_or_aggregate_boundaries(tmp_path):
    disjoint_aggregate_query(tmp_path)
    path = tmp_path / "groups.parquet"
    queries = [
        f"select k % 2 g, sum(s) from (select k, sum(v) s from read_parquet('{path}') group by k) group by g",
        f"select b.range g, sum(a.v) from read_parquet('{path}') a join range(2) b on a.k % 2 = b.range group by g",
        f"select sum(v), count(*) from read_parquet('{path}')",
    ]
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        for sql in queries:
            assert Counter(execute_graph(connection, compile_sql(connection, sql, 3))) == Counter(
                connection.execute(sql).fetchall()
            )


@pytest.mark.parametrize("variant", ["union_by_name", "file_cast", "row_number"])
def test_parquet_group_ranges_do_not_assume_file_schema_mapping(tmp_path, variant):
    disjoint_aggregate_query(tmp_path)
    path = tmp_path / "groups.parquet"
    other = tmp_path / "other.parquet"
    table = pq.read_table(path)
    if variant == "file_cast":
        table = table.set_column(0, "k", table["k"].cast(pa.float64(), safe=False))
    pq.write_table(table, other, row_group_size=2048)
    option = ", union_by_name=true" if variant == "union_by_name" else ""
    key = "k"
    if variant == "row_number":
        option, key = ", file_row_number=true", "file_row_number"
    sql = f"select {key}, sum(v) from read_parquet(['{path}', '{other}']{option}) group by {key}"
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = Counter(connection.execute(sql).fetchall())
        graph = compile_sql(connection, sql, 2)
        assert Counter(execute_graph(connection, graph)) == expected
        assert any(edge.distribution is Distribution.HASH for edge in graph.exchanges)


def test_complete_groups_keep_ordered_sum_when_widening_other_accumulators(tmp_path):
    disjoint_aggregate_query(tmp_path)
    path = tmp_path / "groups.parquet"
    table = pq.read_table(path)
    table = table.set_column(1, "v", pa.array([1e16, 1.0, -1e16, 2.0] * 4096))
    table = table.append_column("i", pa.array(range(len(table))))
    pq.write_table(table, path, row_group_size=2048)
    sql = (
        f"select k, sum(v order by i), sum(v order by i desc), sum(i::hugeint) "
        f"from read_parquet('{path}') group by k order by k"
    )
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, 3)
        assert execute_graph(connection, graph) == expected
        assert all(edge.distribution is Distribution.GATHER for edge in graph.exchanges)
