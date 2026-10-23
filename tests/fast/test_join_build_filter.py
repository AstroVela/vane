# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reuse a selective join build to reduce another hash input."""

from collections import Counter

import pytest

import vane
from tests.fast.test_direct_exchange import collect, submission
from tests.fast.test_native_fragment_compiler import compile_sql, execute_graph
from vane.execution.compiler import FragmentCompileOptions
from vane.execution.direct_exchange import DirectExchangeLimits, InProcessTaskService
from vane.execution.plan import Distribution
from vane.execution.query_options import QueryExecutionOptions, RayExecution
from vane.execution.resource_demand import MemoryDemand, ResourceDemand
from vane.execution.submission import prepare_ray_query


def shared_join_build_query(connection, directory, comparison="=", shape="keys", *, fact_rows=131072, fixed_plan=True):
    fact, keys, target = [directory / f"{name}.parquet" for name in ("fact", "keys", "target")]
    connection.execute(
        "copy (select case when range%4096=0 then null else range%65536 end k, range%4 g, range%13 v "
        f"from range({fact_rows})) to '{fact}' (row_group_size 2048)"
    )
    connection.execute(
        "copy (select case when range%17=0 then null else (range%64)*1021 end k, "
        f"(range%64)*2042 as value, range%5<>0 keep from range(512)) to '{keys}'"
    )
    connection.execute(
        "copy (select case when range%4096=0 then null else range%65536 end k, range%4 g, range%11 v, "
        f"'target-'||range::varchar payload from range(262144)) to '{target}' (row_group_size 2048)"
    )
    if fixed_plan:
        connection.execute("set disabled_optimizers='join_order,build_side_probe_side,compressed_materialization'")
    donor_kind = "left" if shape == "outer_seed" else "inner"
    source = (
        f"select d.k k, f.g, f.v from read_parquet('{fact}') f {donor_kind} join "
        f"read_parquet('{keys}') d on f.k {comparison} d.k and d.keep"
    )
    if shape == "payload":
        source = source.replace("select d.k k", "select d.value k")
    if shape == "limit":
        source += " limit 1024"
    kind = shape if shape in {"left", "right", "full", "anti"} else "inner"
    key = "p.k+p.v" if shape == "computed" else "p.k"
    predicate = f"{key} {comparison} b.k and p.g=b.g"
    if shape == "residual":
        predicate += " and p.v>b.v"
    select = "sum(p.v), count(*)" if kind == "anti" else "sum(p.v+b.v), count(*), max(b.payload)"
    build = f"read_parquet('{target}')"
    if shape == "fused":
        build = f"(select * from {build} limit 16)"
    sql = f"select {select} from ({source}) p {kind} join {build} b on {predicate}"
    return sql, target, keys


@pytest.mark.parametrize("threads,partitions", [(1, 2), (4, 2), (4, 3)])
@pytest.mark.parametrize("comparison", ["=", "is not distinct from"])
@pytest.mark.parametrize("shape", ["keys", "payload", "residual"])
def test_shared_join_build_filters_other_hash_input(tmp_path, threads, partitions, comparison, shape):
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        sql, target, keys = shared_join_build_query(connection, tmp_path, comparison, shape)
        expected = connection.execute(sql).fetchall()
        spec = submission(connection, sql, partitions=partitions)
        target_fragments = {
            f.fragment_id
            for f in spec.graph.fragments
            if any(str(target).encode() in split.payload for source in f.sources for split in source.splits)
        }
        edges = {e.exchange_id for e in spec.graph.exchanges if e.producer_fragment_id in target_fragments}
        limits = DirectExchangeLimits(4096, 1024, 32, 1)
        with InProcessTaskService(connection, spec, limits) as service:
            service.start()
            assert collect(service) == expected
            target_rows = sum(
                c.snapshot()["accepted_rows"] for name, c in service.channels.items() if name.split("/")[0] in edges
            )
            assert edges and target_rows < 4096
            assert all(c.snapshot()["peak_bytes"] <= limits.window_bytes for c in service.channels.values())
        # The original small build and its extra filter share one producer.
        assert (
            sum(
                any(str(keys).encode() in split.payload for split in source.splits)
                for f in spec.graph.fragments
                for source in f.sources
            )
            == 1
        )
        assert all(c.snapshot()["bytes"] == c.snapshot()["leased_bytes"] == 0 for c in service.channels.values())


@pytest.mark.parametrize("shape", ["left", "right", "full", "anti", "outer_seed", "computed", "limit", "fused"])
def test_shared_join_build_keeps_preserved_rows_and_unsupported_lineage(tmp_path, shape):
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        sql, _, _ = shared_join_build_query(connection, tmp_path, "is not distinct from", shape)
        graph = compile_sql(connection, sql, 2)
        assert Counter(execute_graph(connection, graph)) == Counter(connection.execute(sql).fetchall())
        # No producer fans out to both the original join and an added filter.
        producers = [e.producer_fragment_id for e in graph.exchanges]
        assert len(producers) == len(set(producers))


@pytest.mark.parametrize(
    "seed_rows,target_rows,partitions,filtered,nested,shared",
    [
        (400000, 8000000, 2, True, False, True),
        (10000000, 8000000, 2, True, False, False),
        (600000, 4000000, 2, True, False, False),
        (400000, 8000000, 128, True, False, False),
        (400000, 16, 2, True, False, False),
        (400000, 8000000, 1, True, False, False),
        (400000, 8000000, 2, False, False, False),
        (400000, 8000000, 2, True, True, False),
    ],
)
def test_shared_join_build_requires_bounded_selective_replication(
    seed_rows, target_rows, partitions, filtered, nested, shared
):
    predicate = "where range%7=0" if filtered else ""
    payload = "[range]" if nested else "range::varchar"
    sql = (
        "select sum(p.v+b.v),max(b.payload) from (select d.k,f.range v from range(10000000) f "
        f"join (select range k from range({seed_rows}) {predicate}) d on f.range=d.k) p "
        f"join (select range k,range v,{payload} payload from range({target_rows})) b on p.k=b.k"
    )
    # Keep the seed's explicit filter in place without adding a native probe filter.
    config = {
        "disabled_optimizers": (
            "join_order,build_side_probe_side,compressed_materialization,filter_pullup,join_filter_pushdown"
        )
    }
    with vane.connect(backend="local", config=config) as connection:
        graph = compile_sql(connection, sql, partitions)
        producers = [e.producer_fragment_id for e in graph.exchanges]
        assert (len(producers) > len(set(producers))) == shared


@pytest.mark.parametrize("comparison", ["=", "is not distinct from"])
@pytest.mark.parametrize("empty", [False, True])
def test_shared_join_build_preserves_empty_and_all_null_domains(tmp_path, comparison, empty):
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        sql, _, keys = shared_join_build_query(connection, tmp_path, comparison)
        connection.execute(
            f"copy (select null::bigint k,range as value,range%2=0 keep from range({0 if empty else 512})) "
            f"to '{keys}' (overwrite true)"
        )
        expected = connection.execute(sql).fetchall()
        spec = submission(connection, sql, partitions=3)
        with InProcessTaskService(connection, spec, DirectExchangeLimits(4096, 1024, 32, 1)) as service:
            service.start()
            assert collect(service) == expected


@pytest.mark.parametrize("threads", [1, 4])
def test_shared_join_build_does_not_fuse_consumers_after_probe_gather(tmp_path, threads):
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        sql, target, _ = shared_join_build_query(connection, tmp_path, "is not distinct from")
        connection.execute(
            "set disabled_optimizers='join_order,build_side_probe_side,compressed_materialization,join_filter_pushdown'"
        )
        fact = f"read_parquet('{tmp_path / 'fact.parquet'}')"
        sql = sql.replace(f"{fact} f", f"(select * from {fact} order by k) f")
        connection.execute(
            "copy (select case when range%4096=0 then null else range%65536 end k, range%4 g, range%11 v, "
            f"'target-'||range::varchar payload from range(65536)) to '{target}' (overwrite true)"
        )
        expected = connection.execute(sql).fetchall()
        spec = submission(connection, sql, partitions=2)
        # ORDER BY gathers the probe. This parent then broadcasts despite the
        # two-partition cost, so sharing its seed would couple build/probe drains.
        producers = [e.producer_fragment_id for e in spec.graph.exchanges]
        assert len(producers) == len(set(producers))
        with InProcessTaskService(connection, spec, DirectExchangeLimits(4096, 1024, 32, 1)) as service:
            service.start()
            assert collect(service) == expected


@pytest.mark.parametrize("metadata_cache", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_multifile_parquet_cardinality_costs_broadcast_from_footers(tmp_path, metadata_cache, explicit):
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = []
    for index in range(24):
        path = tmp_path / f"keys-{index:02}.parquet"
        keys = [] if index % 3 == 0 else [None if index == 1 else index % 7]
        pq.write_table(pa.table({"k": pa.array(keys, type=pa.int64())}), path)
        files.append(str(path))
    # Repeated and empty references retain their SQL multiplicity.
    files.extend(files[:3])
    hint = ", explicit_cardinality=1000000" if explicit else ""
    sql = f"select sum(f.range), count(*) from range(1024) f join read_parquet({files!r}{hint}) b on f.range % 7=b.k"
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        connection.execute("set disabled_optimizers='join_order,build_side_probe_side,compressed_materialization'")
        connection.execute("set parquet_metadata_cache=?", [metadata_cache])
        graph = compile_sql(connection, sql, 3)
        kinds = {edge.distribution for edge in graph.exchanges}
        assert (Distribution.BROADCAST in kinds) is not explicit
        assert (Distribution.HASH in kinds) is explicit
        assert execute_graph(connection, graph) == connection.execute(sql).fetchall()


@pytest.mark.parametrize("partitions", [2, 3])
def test_parquet_footer_statistics_follow_hive_file_pruning(tmp_path, partitions):
    for bucket in range(2):
        (tmp_path / f"bucket={bucket}").mkdir()
    selected = tmp_path / "bucket=0" / "data.parquet"
    pruned = tmp_path / "bucket=1" / "data.parquet"
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        connection.execute(f"copy (select range k from range(8192)) to '{selected}' (row_group_size 2048)")
        pruned.write_bytes(b"unreadable footer in a pruned partition")
        sql = f"select sum(k) from read_parquet('{tmp_path}/bucket=*/*.parquet') where bucket=0"
        expected = connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, partitions)
        assert execute_graph(connection, graph) == expected
        assert any(
            str(pruned).encode() in split.payload
            for fragment in graph.fragments
            for source in fragment.source_dependencies
            for split in source.splits
        )


def budgeted_submission(connection, sql, partitions):
    operator_bytes = 2 << 30
    return prepare_ray_query(
        connection,
        sql,
        query_id="budgeted-filter",
        options=QueryExecutionOptions(RayExecution(), 5, 30, 30),
        resources=ResourceDemand(1, 64, MemoryDemand(operator_bytes, 1 << 20, 1 << 20, 1 << 20), 1),
        compile_options=FragmentCompileOptions(partitions, worker_operator_bytes=operator_bytes),
    )


@pytest.mark.parametrize("partitions", [2, 3])
@pytest.mark.parametrize("correlated", [False, True])
def test_declared_budget_allows_large_estimated_key_filters(tmp_path, partitions, correlated):
    fact, build, orders = [tmp_path / f"{name}.parquet" for name in ("fact", "build", "orders")]
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        connection.execute(
            f"copy (select range i, range%4096 k, range%13 v from range(65536)) to '{fact}' (row_group_size 2048)"
        )
        connection.execute(f"copy (select range k, range%64=0 keep from range(512)) to '{build}'")
        connection.execute(f"copy (select range i, range%17 v from range(65536)) to '{orders}'")
        connection.execute("set disabled_optimizers='join_order,build_side_probe_side,compressed_materialization'")
        fact_scan = f"read_parquet('{fact}', explicit_cardinality=100000000)"
        # Keep a 34 MiB estimated key domain while executing only a small
        # fixture. Join-order estimation is disabled to preserve this shape.
        build_scan = f"read_parquet('{build}', explicit_cardinality=4000000)"
        if correlated:
            sql = (
                f"select sum(f.v),count(*) from {fact_scan} f join {build_scan} b on f.k=b.k and b.keep "
                f"where f.v < (select avg(i.v) from {fact_scan} i where i.k=b.k)"
            )
        else:
            sql = (
                "select sum(p.v),count(*) from ("
                f"select f.k,f.v from {fact_scan} f join "
                f"read_parquet('{orders}', explicit_cardinality=200000000) o on f.i=o.i) p "
                f"join {build_scan} b on p.k=b.k and b.keep"
            )
        expected = connection.execute(sql).fetchall()
        spec = budgeted_submission(connection, sql, partitions)
        fact_fragments = {
            f.fragment_id
            for f in spec.graph.fragments
            if any(str(fact).encode() in split.payload for source in f.sources for split in source.splits)
        }
        edges = {edge.exchange_id for edge in spec.graph.exchanges if edge.producer_fragment_id in fact_fragments}
        limits = DirectExchangeLimits(4096, 1024, 32, 1)
        with InProcessTaskService(connection, spec, limits) as service:
            service.start()
            assert collect(service) == expected
            fact_rows = sum(
                channel.snapshot()["accepted_rows"]
                for name, channel in service.channels.items()
                if name.split("/")[0] in edges
            )
            assert edges and fact_rows < 4096
            assert all(channel.snapshot()["peak_bytes"] <= limits.window_bytes for channel in service.channels.values())
