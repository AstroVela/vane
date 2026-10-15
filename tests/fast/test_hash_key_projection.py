# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Hash exchanges retain existing key columns without sending duplicate values."""

from collections import Counter

import pytest

import vane
from tests.fast.test_direct_exchange import collect, submission
from vane._native import execution_plan, execution_runtime
from vane.execution.direct_exchange import DirectExchangeLimits, InProcessTaskService


def hash_input_schemas(connection, graph):
    fragments = {f.fragment_id: f for f in graph.fragments}
    result = []
    for edge in graph.exchanges:
        if edge.distribution.value == "hash":
            producer = fragments[edge.producer_fragment_id]
            names = execution_plan.inspect_fragment(connection, producer.native_plan)["names"]
            result.append(execution_runtime.arrow_schema(producer.outputs[0].schema, names))
    return result


def hash_key_query(key="range", comparison="="):
    def source(count):
        return f"select case when range%997=0 then null else {key} end k, range%13 v from range({count})"

    return f"select a.v,b.v from ({source(4096)}) a full join ({source(8192)}) b on a.k {comparison} b.k"


@pytest.mark.parametrize("threads,partitions", [(1, 2), (4, 3)])
@pytest.mark.parametrize("comparison", ["=", "is not distinct from"])
@pytest.mark.parametrize(
    "key",
    ["range::integer", "range*10000000000", "'key-'||range::varchar", "range::decimal(18,2)", "struct_pack(k:=range)"],
)
def test_hash_join_reuses_existing_typed_key_columns(threads, partitions, comparison, key):
    config = {"threads": threads, "disabled_optimizers": "compressed_materialization"}
    with vane.connect(backend="local", config=config) as connection:
        sql = hash_key_query(key, comparison)
        expected = Counter(connection.execute(sql).fetchall())
        spec = submission(connection, sql, partitions=partitions)
        assert [len(s) for s in hash_input_schemas(connection, spec.graph)] == [2, 2]
        limits = DirectExchangeLimits(4096, 1024, 32, 1)
        with InProcessTaskService(connection, spec, limits) as service:
            service.start()
            assert Counter(collect(service)) == expected
            assert all(c.snapshot()["peak_bytes"] <= limits.window_bytes for c in service.channels.values())
        assert all(c.snapshot()["bytes"] == c.snapshot()["leased_bytes"] == 0 for c in service.channels.values())


@pytest.mark.parametrize("comparison", ["=", "is not distinct from"])
@pytest.mark.parametrize("shape,widths", [("repeated", [2, 2]), ("computed", [2, 1]), ("multiple", [2, 2])])
def test_hash_join_keeps_repeated_and_computed_key_order(comparison, shape, widths):
    left = "select case when range%997=0 then null else range end k from range(4096)"
    right = "select case when range%997=0 then null else range end k, range%100 j from range(4096)"
    condition = f"a.k {comparison} b.k and a.k {comparison} b.j"
    output = "a.k,b.k,b.j"
    if shape == "computed":
        right = left
        condition = f"a.k+1 {comparison} b.k"
        output = "a.k,b.k"
    elif shape == "multiple":
        left = right
        condition = f"a.j {comparison} b.k and a.k {comparison} b.j"
    sql = f"select {output} from ({left}) a full join ({right}) b on {condition}"
    config = {"threads": 4, "disabled_optimizers": "join_order,build_side_probe_side,compressed_materialization"}
    with vane.connect(backend="local", config=config) as connection:
        spec = submission(connection, sql, partitions=3)
        assert [len(s) for s in hash_input_schemas(connection, spec.graph)] == widths
        with InProcessTaskService(connection, spec, DirectExchangeLimits(4096, 1024, 32, 1)) as service:
            service.start()
            assert Counter(collect(service)) == Counter(connection.execute(sql).fetchall())


@pytest.mark.parametrize("aggregate", ["sum(distinct v)", "sum(v order by v desc)", "max(v::varchar)"])
@pytest.mark.parametrize("partitions", [2, 3])
def test_complete_aggregate_reuses_group_columns(aggregate, partitions):
    sql = (
        f"select k,j,{aggregate} from (select case when range%997=0 then null else range%17 end k, "
        "range%3 j,range v from range(4096)) group by k,j"
    )
    with vane.connect(backend="local", config={"threads": 4}) as connection:
        spec = submission(connection, sql, partitions=partitions)
        assert [len(s) for s in hash_input_schemas(connection, spec.graph)] == [3]
        with InProcessTaskService(connection, spec, DirectExchangeLimits(4096, 1024, 32, 1)) as service:
            service.start()
            assert Counter(collect(service)) == Counter(connection.execute(sql).fetchall())
