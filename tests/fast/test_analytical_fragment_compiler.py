# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Distributed analytical SQL compared with the native engine."""

from collections import Counter
from fractions import Fraction

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

ORDERED_INPUT = "select range, case when range=0 then 1e16 when range=2 then -1e16 else 1::double end v from range(4)"
ORDERED_AGGREGATES = [
    f"select sum(v order by range) from ({ORDERED_INPUT})",
    "select avg(range::double order by range) from range(4)",
    f"select sum(v order by v), avg(v order by v desc) from ({ORDERED_INPUT})",
    f"select sum(v order by range desc), count(*), min(v), max(v) from ({ORDERED_INPUT})",
    "select range % 2 k, sum(v order by range) filter(where range < 8), "
    "avg(v order by range desc), count(*) from (select range, "
    "case when range // 2=0 then 1e16 when range // 2=2 then -1e16 else 1::double end v from range(8)) "
    "group by k order by k",
    "select sum(v order by k nulls first, range desc) filter(where range % 3<>0), "
    "avg(v order by k nulls last, range) from (select range, "
    "case when range % 4=0 then null else range::double end v, "
    "case when range % 5=0 then null else range % 7 end k from range(19))",
    "select sum(distinct range::double order by range::double desc) filter(where range % 3<>0) from range(9)",
    "select sum(null::double order by range), avg(null::double order by range) from range(9)",
    "select sum(range::double order by range), avg(range::double order by range) from range(0)",
]

INTERNAL_AGGREGATE_REWRITES = [
    "select k, count(*) n from (select case when range%2=0 then interval '1 month' "
    "else interval '30 days' end k from range(20)) group by k",
    "select interval '1 day' * (range % 3) k, count(*), sum(range) from range(19) group by k order by k",
    "select case when range % 3=0 then null else interval '1 day' end k, count(*) "
    "from range(19) group by k order by k nulls last",
    "select min(x), max(x) from (select case when range%2=0 then 'A' else 'a' end collate nocase x from range(20))",
    "select min(x), max(x) from (select case when range % 3=0 then null "
    "when range % 3=1 then 'a' else 'B' end collate nocase x from range(19))",
    "select range % 2 k, min(x) filter(where range<15), max(x), count(*) from (select range, "
    "case when range % 3=0 then null when range % 3=1 then 'a' else 'B' end collate nocase x "
    "from range(19)) group by k order by k",
    "select min(x order by range desc), max(x order by range) from (select range, "
    "case when range%2=0 then 'A' else 'a' end collate nocase x from range(20))",
    "select min(x), max(x) from (select 'a' collate nocase x from range(0))",
]

HUGEINT_SUM_INPUT = (
    "select range, case when range<3 then '-40000000000000000000000000000000000000'::hugeint "
    "else '60000000000000000000000000000000000000'::hugeint end v from range(6)"
)
HUGEINT_SUM_QUERIES = [
    f"select sum(v) from ({HUGEINT_SUM_INPUT})",
    f"select sum(v)::varchar from ({HUGEINT_SUM_INPUT})",
    f"select sum(-v) from ({HUGEINT_SUM_INPUT})",
    f"select sum(v order by range), avg(v), count(v) from ({HUGEINT_SUM_INPUT})",
    "select range % 2 k, sum(v) filter(where range<12), count(*), min(v), max(v) from (select range, "
    "case when range<6 then '-40000000000000000000000000000000000000'::hugeint "
    "else '60000000000000000000000000000000000000'::hugeint end v from range(12)) group by k order by k",
    "select sum(null::hugeint) from range(7)",
    "select sum(range::hugeint) from range(0)",
]

WIDE_AGGREGATE_TYPES = [("hugeint", 0), ("decimal(38,0)", 0), ("decimal(38,5)", 5), ("decimal(38,38)", 38)]


def scaled_wide_value(value, scale):
    digits = str(abs(value)).zfill(scale + 1)
    text = digits if not scale else f"{digits[:-scale]}.{digits[-scale:]}"
    return f"-{text}" if value < 0 else text


def wide_aggregate_queries(kind, scale):
    negative = scaled_wide_value(-4 * 10**37, scale)
    positive = scaled_wide_value(6 * 10**37, scale)
    source = f"select range, case when range<3 then '{negative}'::{kind} else '{positive}'::{kind} end v from range(6)"
    return [
        f"select sum(v), avg(v), count(v), min(v), max(v) from ({source})",
        f"select sum(distinct v), avg(distinct v) from ({source})",
        f"select sum(v) filter(where range % 3<>2), avg(v) filter(where range % 3<>2) from ({source})",
        f"select sum(v), avg(v) from ({source}) where range>10",
        f"select sum(null::{kind}), avg(null::{kind}) from range(7)",
        "select range % 2 k, sum(v), avg(v), count(v) from (select range, "
        f"case when range>=12 then null when range<6 then '{negative}'::{kind} "
        f"else '{positive}'::{kind} end v from range(14)) group by k order by k",
    ]


@pytest.mark.parametrize("kind,scale", WIDE_AGGREGATE_TYPES)
@pytest.mark.parametrize("partitions", [1, 2, 5])
def test_wide_numeric_aggregates_preserve_native_semantics(kind, scale, partitions):
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        for sql in wide_aggregate_queries(kind, scale):
            expected = connection.execute(sql).fetchall()
            assert execute_graph(connection, compile_sql(connection, sql, partitions)) == expected


@pytest.mark.parametrize("kind,scale", WIDE_AGGREGATE_TYPES)
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize("threads", [1, 4])
def test_wide_avg_accepts_total_outside_hugeint(kind, scale, sign, constant, threads):
    first = sign * 6 * 10**37
    second = first if constant else sign * 5 * 10**37
    value = f"'{scaled_wide_value(first, scale)}'::{kind}"
    if not constant:
        value = f"case when range % 2=0 then {value} else '{scaled_wide_value(second, scale)}'::{kind} end"
    rows = 8193
    expected = float(Fraction(((rows + 1) // 2) * first + (rows // 2) * second, rows * 10**scale))
    sql = f"select avg({value}) from range({rows})"
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        actual = execute_graph(connection, compile_sql(connection, sql, 2))
    assert actual[0][0] == pytest.approx(expected, rel=4e-16)


@pytest.mark.parametrize("kind,scale", WIDE_AGGREGATE_TYPES)
@pytest.mark.parametrize("partitions", [1, 2, 5])
def test_wide_average_retains_small_remainder_after_cancellation(kind, scale, partitions):
    values = [9 * 10**37, 9 * 10**37, -9 * 10**37, -9 * 10**37, 1]
    cases = " ".join(f"when {i} then '{scaled_wide_value(v, scale)}'::{kind}" for i, v in enumerate(values))
    sql = f"select avg(case range {cases} end order by range desc) from range(5)"
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        actual = execute_graph(connection, compile_sql(connection, sql, partitions))
    assert actual[0][0] == pytest.approx(float(Fraction(1, 5 * 10**scale)), rel=4e-16, abs=0)


@pytest.mark.parametrize("kind,scale", WIDE_AGGREGATE_TYPES[1:])
@pytest.mark.parametrize("sign", [1, -1])
def test_wide_decimal_sum_still_rejects_final_overflow(kind, scale, sign):
    value = scaled_wide_value(sign * 9 * 10**37, scale)
    sql = f"select sum('{value}'::{kind}) from range(2)"
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        with pytest.raises(vane.OutOfRangeException, match="[Oo]verflow"):
            connection.execute(sql).fetchall()
        with pytest.raises(vane.OutOfRangeException, match="[Oo]verflow"):
            execute_graph(connection, compile_sql(connection, sql, 2))


@pytest.mark.parametrize("sql", ORDERED_AGGREGATES)
@pytest.mark.parametrize("partitions", [1, 2, 5])
@pytest.mark.parametrize("optimizer", [True, False])
def test_ordered_aggregates_keep_complete_groups(sql, partitions, optimizer):
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        if not optimizer:
            connection.execute("pragma disable_optimizer")
        expected = connection.execute(sql).fetchall()
        assert execute_graph(connection, compile_sql(connection, sql, partitions)) == expected


@pytest.mark.parametrize("sql", INTERNAL_AGGREGATE_REWRITES + HUGEINT_SUM_QUERIES)
@pytest.mark.parametrize("partitions", [1, 2, 5])
def test_native_aggregate_rewrites_and_accumulator_bounds(sql, partitions):
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        expected = connection.execute(sql).fetchall()
        assert execute_graph(connection, compile_sql(connection, sql, partitions)) == expected


@pytest.mark.parametrize("call", ["first(range)", "arg_min(range, range)", "arg_max(range, range)"])
def test_internal_aggregate_rewrites_do_not_expand_parsed_sql_subset(call):
    with vane.connect(backend="local") as connection:
        with pytest.raises(vane.NotImplementedException, match="does not support function"):
            compile_sql(connection, f"select {call} from range(3)")


@pytest.mark.parametrize("value", [10**38, -(10**38)])
def test_hugeint_sum_still_rejects_native_overflow(value):
    sql = f"select sum('{value}'::hugeint) from range(2)"
    with vane.connect(backend="local", config={"threads": 1}) as connection:
        with pytest.raises(vane.OutOfRangeException, match="[Oo]verflow"):
            connection.execute(sql).fetchall()
        with pytest.raises(vane.OutOfRangeException, match="[Oo]verflow"):
            execute_graph(connection, compile_sql(connection, sql, 2))


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize(
    "values",
    [
        [2**127 - 1, 2**127 - 1, -(2**127 - 1)],
        [-(2**127), -(2**127), 2**127 - 1, 1],
        [2**64 - 1, 1, -(2**64)],
        [-(2**127), 2**127 - 1, None, 1],
    ],
)
def test_hugeint_complete_sum_uses_exact_wide_state(values, grouped, threads):
    cases = " ".join(f"when {i} then '{value}'::hugeint" for i, value in enumerate(values) if value is not None)
    value = f"case range // 3 {cases} end"
    projection = "range % 3 k, " if grouped else ""
    tail = " group by k order by k" if grouped else ""
    sql = f"select {projection}sum({value}) from range({len(values) * 3}){tail}"
    # Wider state also covers prefixes which exceed HUGEINT before canceling.
    # Python integers provide an exact reference independent of native overflow.
    expected = sum(v for v in values if v is not None)
    if not grouped:
        # Repeat each value with zeros, preserving a legal final result.
        value = f"case when range % 3=0 then {value} else 0::hugeint end"
        sql = f"select sum({value}) from range({len(values) * 3})"
    with vane.connect(backend="local", config={"threads": threads}) as connection:
        actual = execute_graph(connection, compile_sql(connection, sql, 2))
    assert actual == ([(k, expected) for k in range(3)] if grouped else [(expected,)])


def test_collated_aggregate_rebinding_preserves_physical_inputs():
    sql = (
        "select range % 2 k, min(x), max(x) from (select range, "
        "case when range % 3=0 then null when range % 3=1 then 'a' else 'B' end x from range(19)) "
        "group by k order by k"
    )
    with vane.connect(backend="local", config={"threads": 1, "default_collation": "nocase"}) as connection:
        expected = connection.execute(sql).fetchall()
        assert execute_graph(connection, compile_sql(connection, sql, 2)) == expected


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
