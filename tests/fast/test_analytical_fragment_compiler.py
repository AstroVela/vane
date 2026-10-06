# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Distributed analytical SQL compared with the native engine."""

from collections import Counter

import pytest

import vane
from tests.fast.test_native_fragment_compiler import compile_sql, execute_graph
from vane._native import execution_plan as native
from vane.execution.plan import Distribution

AGGREGATES = [
    "select count(*), count(range), sum(range), avg(range), min(range), max(range) from range(103)",
    "select count(*), count(range), sum(range), avg(range), min(range), max(range) from range(0)",
    "select range % 7 k, count(*), sum(range), avg(range), min(range), max(range) from range(103) group by k",
    "select range % 7 k, count(*), sum(range) from range(0) group by k",
    "select case when range % 3 = 0 then null else 1 end k, sum(range), count(*) from range(101) group by k",
    "select range % 3 k, sum(range) filter(where range % 5 = 0), avg(range) filter(where range % 5 = 0), "
    "count(*) filter(where range % 5 = 0) from range(103) group by k",
    "select range % 4 k, count(distinct range % 7), sum(distinct range % 7), avg(distinct range % 7) "
    "from range(103) group by k",
    "select count(distinct range % 5) from range(103)",
    "select range % 3 k, avg(range::double), sum(range) from range(103) group by k",
    "select avg(case when range=0 then 9007199254740993::bigint else 2 end) from range(3)",
    "select * from (select avg(case when range=0 then 9007199254740993::bigint else 2 end) a from range(3)) "
    "where a > 3002399751580332::double",
    "select avg(x), count(x), sum(x), min(x), max(x) from (select null::bigint x from range(19)) t",
    "select sum(range::decimal(15,2)), avg(range::decimal(15,2)) from range(101)",
    "select range % 100 k, sum(range) s from range(103) group by k having sum(range)>30",
    "select range % 3 k, avg(interval '1 day' * range), min(interval '1 day' * range) from range(19) group by k",
    "select range % 3 k, avg(timestamp '2026-01-01' + interval '1 second' * range) from range(19) group by k",
    "select range % 4 k, count(distinct range % 7) filter(where range % 3 = 0) from range(103) group by k",
]

TOP_N_AGGREGATES = [
    "select min(range, 3), max(range, 3) from range(4)",
    "select min(range, 10), max(range, 10), sum(range) from range(4)",
    "select range % 3 k, min(range, 4) filter(where range % 2=0), max(range, 4), count(*) "
    "from range(31) group by k order by k",
    "select min(x, 3), max(x, 3) from (select null::bigint x from range(9)) t",
    "select min(range, 3), max(range, 3) from range(0)",
    "select min([range]), max([range]) from range(9)",
]

PARQUET_TOP_N_TAILS = [
    "order by k limit 5",
    "where k % 2 = 1 order by k desc limit 5 offset 3",
    "order by k nulls first limit 5",
    "order by k limit 5 offset 100",
]

DECIMAL_INPUT = (
    "select range, case when range < 3 then '40000000000000000000000000000000000000'::decimal(38,0) "
    "else '-30000000000000000000000000000000000000'::decimal(38,0) end v from range(6)"
)
DECIMAL_SUM_QUERIES = [
    f"select sum(v) from ({DECIMAL_INPUT})",
    f"select sum(v)::varchar from ({DECIMAL_INPUT})",
    f"select sum(-v) from ({DECIMAL_INPUT})",
    "select sum(case when range < 3 then '400000000000000000000000000000000.00000'::decimal(38,5) "
    "else '-300000000000000000000000000000000.00000'::decimal(38,5) end) from range(6)",
    "select range % 2 k, sum(v) filter(where range % 8 < 6) s, count(*) filter(where range % 3 = 0), min(v) "
    "from (select range, case when range % 8 >= 6 then null "
    "when range < 8 then '40000000000000000000000000000000000000'::decimal(38,0) "
    "else '-30000000000000000000000000000000000000'::decimal(38,0) end v from range(16)) "
    "group by k having sum(v) > 0 order by k",
    f"select sum(distinct v) from ({DECIMAL_INPUT})",
    "select sum(null::decimal(38,0)) from range(6)",
    "select sum(range::decimal(38,5)) from range(0)",
    "select sum(range::decimal(4,1)), count(*) from range(6)",
]


def parquet_top_n_source(connection, directory):
    for index in range(2):
        path = directory / f"part{index}.parquet"
        connection.execute(
            f"copy (select case when range=11 then null else range end k, range::varchar v "
            f"from range({index * 20}, {(index + 1) * 20})) to '{path}' (format parquet)"
        )
    return f"read_parquet('{directory}/*.parquet')"


@pytest.mark.parametrize("tail", PARQUET_TOP_N_TAILS)
@pytest.mark.parametrize("partitions", [1, 2, 5])
def test_parquet_top_n_with_default_optimizers(tmp_path, tail, partitions):
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        source = parquet_top_n_source(connection, tmp_path)
        sql = f"select k, v from {source} {tail}"
        expected = connection.execute(sql).fetchall()
        assert execute_graph(connection, compile_sql(connection, sql, partitions)) == expected


@pytest.mark.parametrize("disabled", ["", "in_clause", "late_materialization", "top_n"])
def test_fragment_parquet_capabilities_do_not_change_local_optimization(tmp_path, disabled):
    with vane.connect(backend="local", config={"threads": 1}) as connection, connection.cursor() as sibling:
        connection.execute(f"set disabled_optimizers='{disabled}'")
        source = parquet_top_n_source(connection, tmp_path)
        sql = f"select k, v from {source} order by k limit 5"
        plan = connection.execute("explain " + sql).fetchall()
        if disabled in ("", "in_clause"):
            assert "file_index" in plan[0][1]
        expected = connection.execute(sql).fetchall()
        assert execute_graph(connection, compile_sql(connection, sql, 2)) == expected
        # Failure after optimization must also leave both connection settings
        # and the catalog's shared Parquet capability intact.
        with pytest.raises(vane.InvalidInputException, match="HASH columns"):
            compile_sql(connection, sql, 2, hash_columns=(20,))
        for target in (connection, sibling):
            assert target.execute("select current_setting('disabled_optimizers')").fetchone() == (disabled,)
            assert target.execute("explain " + sql).fetchall() == plan


@pytest.mark.parametrize("sql", DECIMAL_SUM_QUERIES)
@pytest.mark.parametrize("partitions", [1, 2, 7])
def test_decimal_sum_keeps_native_precision(sql, partitions):
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = connection.execute(sql).fetchall()
        assert execute_graph(connection, compile_sql(connection, sql, partitions)) == expected


@pytest.mark.parametrize("partitions", [1, 2, 7])
@pytest.mark.parametrize("sql", TOP_N_AGGREGATES)
def test_min_max_overloads(sql, partitions):
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = connection.execute(sql).fetchall()
        assert execute_graph(connection, compile_sql(connection, sql, partitions)) == expected


@pytest.mark.parametrize("partitions", [1, 3, 7])
@pytest.mark.parametrize("sql", AGGREGATES)
def test_distributed_aggregates(sql, partitions):
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, partitions)
        actual = execute_graph(connection, graph)
        assert Counter(actual) == Counter(expected)


@pytest.mark.parametrize("kind", ["inner", "left", "right", "full outer", "semi", "anti"])
@pytest.mark.parametrize("right_size", [0, 3, 100])
@pytest.mark.parametrize("null_safe", [False, True])
def test_distributed_joins(kind, right_size, null_safe):
    predicate = "is not distinct from" if null_safe else "="
    sql = (
        "select a.k, a.range from "
        "(select case when range % 9=0 then null else range % 7 end k, range from range(47)) a "
        f"{kind} join "
        f"(select case when range % 9=0 then null else range % 7 end k from range({right_size})) b "
        f"on a.k {predicate} b.k"
    )
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, 3)
        assert Counter(execute_graph(connection, graph)) == Counter(expected)


@pytest.mark.parametrize(
    "tail",
    [
        "order by k nulls first, range desc",
        "order by k nulls last, range limit 11 offset 3",
        "limit 5",
        "limit 0",
        "limit 5 offset 1000",
    ],
)
def test_global_order_and_limit(tail):
    sql = f"select case when range % 11=0 then null else range % 7 end k, range from range(105) {tail}"
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, 3)
        actual = execute_graph(connection, graph)
        if tail == "limit 5":
            assert len(actual) == 5
        else:
            assert actual == expected
        assert graph.fragments[-1].partition_count == 1


def test_broadcast_and_partitioned_join_plans():
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        for size, distribution in [(3, Distribution.BROADCAST), (100, Distribution.HASH)]:
            sql = f"select a.range from range(1000) a join range({size}) b using(range)"
            graph = compile_sql(connection, sql, 3)
            assert distribution in [e.distribution for e in graph.exchanges]
            assert Counter(execute_graph(connection, graph)) == Counter(connection.execute(sql).fetchall())


def test_partial_aggregation_reduces_rows_before_exchange():
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        graph = compile_sql(connection, "select range % 3 k, sum(range) from range(90) group by k", 3)
        source = graph.fragments[0]
        partial_rows = []
        for partition in range(source.partition_count):
            assignments = {
                s.source_id: [split.split_id for split in s.splits[partition :: source.partition_count]]
                for s in source.sources
            }
            partial_rows.extend(native._execute_fragment_for_test(connection, source.native_plan, {}, assignments))
        assert len(partial_rows) == 9
        assert sum(row[1] for row in partial_rows) == sum(range(90))
