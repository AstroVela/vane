# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reuse a selective join build to reduce another hash input."""

from collections import Counter

import pytest

import vane
from tests.fast.test_direct_exchange import collect, submission
from tests.fast.test_native_fragment_compiler import compile_sql, execute_graph
from vane.execution.direct_exchange import DirectExchangeLimits, InProcessTaskService


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
