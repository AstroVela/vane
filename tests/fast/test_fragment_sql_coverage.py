# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Native SQL semantics across subquery and shared-state fragment boundaries."""

import subprocess
import sys
import textwrap
from collections import Counter

import pytest

import vane
from tests.fast.test_native_fragment_compiler import compile_sql, execute_graph

SQL_CASES = [
    "select a.range, b.range from range(17) a, range(7) b where a.range % 7=b.range",
    "select range, substring('aßcd', 2, 2), 'prefix_suffix' like 'prefix_%', "
    "'abc' not like '%z', sqrt(range::double) from range(5)",
    "select a.range, b.range from range(3) a cross join range(5) b",
    "select a.range, b.range from range(3) a join range(5) b on a.range < b.range",
    "select range from range(13) where range > (select avg(range) from range(9))",
    "select (select max(range) from range(0)), (select count(*) from range(0))",
    "select a.range, (select sum(b.range) from range(9) b where b.range % 5=a.range) from range(7) a",
    "select a.range, (select count(*) from range(9) b where b.range % 5=a.range) from range(7) a",
    "select a.range from range(13) a where exists (select 1 from range(5) b where b.range=a.range)",
    "select a.range from range(13) a where not exists (select 1 from range(5) b where b.range=a.range)",
    "select range, range in (select range from range(3)), range not in (select range from range(3)) from range(7)",
    "select range, range in (select case when range=0 then null else range end from range(3)), "
    "range not in (select case when range=0 then null else range end from range(3)) from range(7)",
    "select null::bigint in (select range from range(0)), null::bigint not in (select range from range(0))",
    "select range, range > all(select range from range(3)), range < any(select range from range(3)) from range(5)",
    "select a.k, (select count(*) from (select range % 3 k from range(11)) b "
    "where b.k is not distinct from a.k) from (values (null::bigint), (0), (2), (4)) a(k)",
    "with x as materialized (select range % 3 k, sum(range) s from range(31) group by k) "
    "select a.k, a.s, b.s from x a join x b using(k)",
    "with x as materialized (select range k from range(9)), "
    "y as materialized (select k % 3 k, count(*) n from x group by k) "
    "select x.k, y.n from x join y on x.k % 3=y.k",
    "with x as materialized (select range k from range(0)) select (select count(*) from x), (select max(k) from x)",
    "with x as materialized (select range k from range(11)) "
    "select a.k, (select sum(b.k) from x b where b.k % 3=a.k) from x a",
    "select a.range from range(7) a where exists (select 1 from range(9) b where b.range % 3=a.range "
    "and exists (select 1 from range(4) c where c.range=b.range and c.range>=a.range))",
    "select a.range, (select min(b.range) from range(5) b where b.range=a.range) from range(0) a",
    "select distinct a.range % 3 from range(10) a, range(10) b where a.range=b.range",
]

# Ordered DISTINCT ON becomes FIRST without optimization and ARG_MIN_NULL with
# optimization. Every group must see all rows before choosing its payloads.
ORDERED_DISTINCT_CASES = [
    "select distinct on (range % 3) range % 3 k, range v from range(17) order by k, v desc",
    "select distinct on (range % 3) range % 3 k, range v from range(17) order by k, v asc",
    "select distinct on (range % 3) range % 3 k, range v, "
    "case when range>=14 then null else range::varchar end payload from range(17) order by k, v desc",
    "select distinct on (range % 3, range % 2) range % 3 k, range % 2 j, range v "
    "from range(31) order by k desc, j, v desc",
    "select distinct on (range % 3) range % 3 k, 100-range payload from range(17) "
    "order by k, range desc limit 2 offset 1",
    "select distinct on (range % 3) range % 3 k, range v from range(0) order by k, v desc",
    "select distinct on (range % 3) range % 3 k, range v from range(2) order by k, v desc",
] + [
    "select distinct on (k) k, sort_key, v, payload from (select "
    "case when range%4=0 then null else range%4 end k, "
    "case when range%5=0 then null else range%3 end sort_key, range v, "
    "case when range%2=0 then null else range::varchar end payload from range(31)) "
    f"order by k nulls {null_order}, sort_key desc nulls {null_order}, v desc"
    for null_order in ("first", "last")
]

# Cross the native planner's nested-loop/merge-join threshold, including MARK
# semantics for NULL and empty subqueries and the delimiter scan of COUNT(*).
RANGE_JOIN_CASES = [
    sql
    for count in (0, 3, 10, 100)
    for sql in (
        f"select a.range, (select count(*) from range({count}) b where b.range < a.range) from range(10) a",
        f"select a.range, (select count(*) from range({count}) b where b.range >= a.range) from range(10) a",
        "select case when range=0 then null else range end < any "
        f"(select case when range % 3=0 then null else range end from range({count})), "
        "case when range=0 then null else range end > all "
        f"(select case when range % 3=0 then null else range end from range({count})) from range(10)",
    )
]
RANGE_JOIN_CASES += [
    "select a.payload, b.payload from (select range k, range+100 payload from range(10)) a "
    f"{kind} join (select range k, range+200 payload from range(10)) b on a.k < b.k"
    for kind in ("inner", "left", "full outer")
]

# Two range predicates cross another planner threshold: larger inputs use
# IE_JOIN, including inside the native delimiter scope of a correlated COUNT.
IE_JOIN_CASES = [
    f"select a.range, (select count(*) from range({count}) b "
    f"where b.range between a.range and a.range+1) n from range({count}) a"
    for count in (0, 3, 10, 2000, 4097)
]
IE_JOIN_CASES += [
    "select a.payload, b.payload from "
    "(select case when range % 17=0 then null else range//2 end k, range+100 payload from range(2500)) a "
    f"{kind} join "
    "(select case when range % 13=0 then null else range//2 end k, range+200 payload from range(2300)) b "
    "on a.k>=b.k and a.k<b.k+2 and a.payload<>b.payload"
    for kind in ("inner", "left", "full outer")
]
IE_JOIN_EQUALITY_TAIL = (
    "select a.range+100 payload_a, b.range+200 payload_b from range(2500) a join range(2300) b "
    "on a.range>=b.range and a.range<b.range+2 and a.range%3=b.range%3"
)

DETERMINISTIC_LAMBDAS = [
    "select list_transform([range, null], lambda x: x+1) from range(4)",
    "select list_transform([range], lambda x: list_transform([x, x+1], lambda y: y*2)) from range(4)",
    "select list_filter([range, range+1, null], lambda x: x%2=0) from range(4)",
    "select list_reduce([range, range+1], lambda x,y: x+y) from range(4)",
]
UNSTABLE_LAMBDAS = [
    "select list_transform([range], lambda x: current_timestamp) from range(4)",
    "select array_transform([range], lambda x: [current_date, current_date]) from range(4)",
    "select list_transform([range], lambda x: list_transform([x], lambda y: localtimestamp)) from range(4)",
    "select list_filter([range], lambda x: current_timestamp>timestamp '2000-01-01') from range(4)",
    "select list_reduce([range, range+1], lambda x,y: case when current_date>date '2000-01-01' then x else y end) "
    "from range(4)",
]
CONTEXT_READS = [
    "select current_setting('threads') from range(4)",
    "select list_transform([range], lambda x: current_setting('threads')) from range(4)",
    "select list_transform([range], lambda x: list_transform([x], lambda y: current_setting('threads'))) from range(4)",
    "select list_filter([range], lambda x: current_setting('threads')>0) from range(4)",
    "select list_reduce([range, range+1], lambda x,y: x+y+current_setting('threads')) from range(4)",
]
UNSTABLE_AGES = [
    sql
    for timestamp in ("timestamp", "timestamptz")
    for sql in (
        f"select age({timestamp} '2000-01-01') from range(4)",
        f"select age({timestamp} '2000-01-01' + range * interval '1 day') from range(4)",
        f"select list_transform([range], lambda x: age({timestamp} '2000-01-01' + x * interval '1 day')) from range(4)",
        f"select (select age({timestamp} '2000-01-01' + a.range * interval '1 day')) from range(4) a",
    )
]
DETERMINISTIC_AGES = [
    sql
    for timestamp in ("timestamp", "timestamptz")
    for sql in (
        f"select age({timestamp} '2000-01-05', {timestamp} '2000-01-01')::varchar from range(4)",
        f"select age({timestamp} '2000-01-05', {timestamp} '2000-01-01' + range * interval '1 day')::varchar "
        "from range(4)",
        f"select list_transform([range, null], lambda x: age({timestamp} '2000-01-05', "
        f"{timestamp} '2000-01-01' + x * interval '1 day')::varchar) from range(4)",
    )
]
UNSTABLE_TIMEZONES = [
    "select timezone('America/New_York', timetz '12:00:00+00')::varchar from range(4)",
    "select timezone(case when range%2=0 then 'America/New_York' else 'UTC' end, "
    "timetz '12:00:00+00')::varchar from range(4)",
    "select (timetz '12:00:00+00' at time zone 'America/New_York')::varchar from range(4)",
    "select list_transform([range], lambda x: timezone(case when x%2=0 then 'America/New_York' else 'UTC' end, "
    "timetz '12:00:00+00')::varchar) from range(4)",
    "select (select timezone(case when a.range%2=0 then 'America/New_York' else 'UTC' end, "
    "timetz '12:00:00+00')::varchar) from range(4) a",
    "select coalesce(timezone('America/New_York', timetz '12:00:00+00'), timetz '00:00:00+00')::varchar from range(4)",
    "select timezone('America/New_York', null::timetz)::varchar from range(4)",
]
DETERMINISTIC_TIMEZONES = [
    sql
    for timestamp in ("timestamp", "timestamptz")
    for sql in (
        f"select timezone('America/New_York', {timestamp} '2020-11-01 12:00:00+00')::varchar from range(4)",
        "select timezone(case when range%2=0 then 'America/New_York' else 'UTC' end, "
        f"{timestamp} '2020-11-01 12:00:00+00' + range * interval '1 day')::varchar from range(4)",
        "select list_transform([range, null], lambda x: timezone('America/New_York', "
        f"{timestamp} '2020-11-01 12:00:00+00' + x * interval '1 day')::varchar) from range(4)",
    )
] + [
    "select timezone(interval '1 hour', timetz '12:00:00+00')::varchar from range(4)",
    "select timezone(timestamptz '2020-11-01 12:00:00+02') from range(4)",
]


@pytest.mark.parametrize("partitions", [1, 2, 5])
@pytest.mark.parametrize("sql", SQL_CASES + RANGE_JOIN_CASES + IE_JOIN_CASES)
def test_fragment_subqueries_and_shared_state(sql, partitions):
    with vane.connect(config={"threads": 1}) as connection:
        expected = connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, partitions=partitions)
        assert Counter(execute_graph(connection, graph)) == Counter(expected)


@pytest.mark.parametrize("sql", ORDERED_DISTINCT_CASES)
@pytest.mark.parametrize("partitions", [1, 2, 5])
@pytest.mark.parametrize("optimizer", [True, False])
def test_ordered_distinct_on_keeps_complete_groups(sql, partitions, optimizer):
    with vane.connect(config={"threads": 1}) as connection:
        if not optimizer:
            connection.execute("pragma disable_optimizer")
        expected = connection.execute(sql).fetchall()
        assert execute_graph(connection, compile_sql(connection, sql, partitions)) == expected


@pytest.mark.parametrize("count,operator", [(3, "NESTED_LOOP_JOIN"), (10, "PIECEWISE_MERGE_JOIN")])
def test_correlated_inequality_crosses_native_join_threshold(count, operator):
    sql = f"select a.range, (select count(*) from range({count}) b where b.range<a.range) from range(10) a"
    with vane.connect(config={"threads": 1}) as connection:
        plan = connection.execute("explain (format json) " + sql).fetchone()[1]
        assert operator in plan
        assert sorted(execute_graph(connection, compile_sql(connection, sql, partitions=2))) == [
            (row, min(row, count)) for row in range(10)
        ]


@pytest.mark.parametrize("partitions", [1, 3])
@pytest.mark.parametrize("count,operator", [(10, "PIECEWISE_MERGE_JOIN"), (2000, "IE_JOIN")])
def test_correlated_between_crosses_native_ie_join_threshold(count, operator, partitions):
    sql = (
        f"select a.range, (select count(*) from range({count}) b "
        f"where b.range between a.range and a.range+1) n from range({count}) a"
    )
    with vane.connect(config={"threads": 1}) as connection:
        plan = connection.execute("explain (format json) " + sql).fetchone()[1]
        assert operator in plan
        expected = [(row, min(2, count - row)) for row in range(count)]
        assert sorted(connection.execute(sql).fetchall()) == expected
        assert sorted(execute_graph(connection, compile_sql(connection, sql, partitions))) == expected


@pytest.mark.parametrize("partitions", [1, 3])
def test_ie_join_preserves_range_conditions_before_equality_tail(partitions):
    with vane.connect(config={"threads": 1, "prefer_range_joins": True}) as connection:
        plan = connection.execute("explain (format json) " + IE_JOIN_EQUALITY_TAIL).fetchone()[1]
        assert "IE_JOIN" in plan
        expected = connection.execute(IE_JOIN_EQUALITY_TAIL).fetchall()
        graph = compile_sql(connection, IE_JOIN_EQUALITY_TAIL, partitions)
        assert Counter(execute_graph(connection, graph)) == Counter(expected)


@pytest.mark.parametrize("threads", [1, 4])
def test_ie_join_parallel_outer_tasks_own_null_payloads(threads):
    # A source thread can start with unmatched rows without ever producing an
    # inner match. Run in a child so an invalid native payload fails one test.
    script = textwrap.dedent(
        """
        import sys
        from collections import Counter
        import vane
        from tests.fast.test_fragment_sql_coverage import IE_JOIN_CASES
        from tests.fast.test_native_fragment_compiler import compile_sql, execute_graph

        with vane.connect(config={"threads": int(sys.argv[1])}) as connection:
            for sql in IE_JOIN_CASES[-2:]:
                assert "IE_JOIN" in connection.execute("explain (format json) " + sql).fetchone()[1]
                for _ in range(3):
                    rows = connection.execute(sql).fetchall()
                    assert any(a is None or b is None for a, b in rows)
                    assert Counter(execute_graph(connection, compile_sql(connection, sql, 3))) == Counter(rows)
        """
    )
    completed = subprocess.run([sys.executable, "-c", script, str(threads)], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("entry", ["compiler", "pipelined", "fte"])
@pytest.mark.parametrize("optimizers", ["default", "no_expression_rewriter", "disabled"])
@pytest.mark.parametrize("sql", UNSTABLE_LAMBDAS + CONTEXT_READS + UNSTABLE_AGES + UNSTABLE_TIMEZONES)
def test_context_reads_and_lambda_bodies_are_checked_before_optimization(sql, optimizers, entry):
    from tests.fast.test_execution_submission import options, prepare
    from vane.execution.query_options import DistributedMode

    with vane.connect(config={"threads": 4}) as connection:
        if optimizers == "no_expression_rewriter":
            connection.execute("set disabled_optimizers='expression_rewriter'")
        elif optimizers == "disabled":
            connection.execute("pragma disable_optimizer")
        assert connection.execute(sql).to_arrow_table().num_rows == 4
        with pytest.raises(vane.NotImplementedException, match="consistent across tasks|unfrozen connection context"):
            if entry == "compiler":
                compile_sql(connection, sql, partitions=2)
            else:
                prepare(connection, sql, options=options(DistributedMode(entry)))
        # The guard belongs to this planner, not the connection's later queries.
        assert connection.execute(sql).to_arrow_table().num_rows == 4


@pytest.mark.parametrize("sql", DETERMINISTIC_LAMBDAS)
def test_deterministic_lambdas_survive_worker_plan_deserialization(sql):
    # Ordinary references in the lambda body must remain scoped to that lambda.
    with vane.connect(config={"threads": 4}) as connection, vane.connect(config={"threads": 1}) as worker:
        expected = connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, partitions=2)
        assert execute_graph(worker, graph) == expected


@pytest.mark.parametrize("optimizers", ["default", "no_expression_rewriter", "disabled"])
@pytest.mark.parametrize("sql", DETERMINISTIC_AGES + DETERMINISTIC_TIMEZONES)
def test_deterministic_temporal_overloads_remain_portable(sql, optimizers):
    with vane.connect(config={"threads": 4}) as connection, vane.connect(config={"threads": 1}) as worker:
        if optimizers == "no_expression_rewriter":
            connection.execute("set disabled_optimizers='expression_rewriter'")
        elif optimizers == "disabled":
            connection.execute("pragma disable_optimizer")
        expected = connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, partitions=2)
        for _ in range(2):
            assert execute_graph(worker, graph) == expected


@pytest.mark.parametrize("entry", ["compiler", "pipelined", "fte"])
@pytest.mark.parametrize("timestamp", ["timestamp", "timestamptz"])
def test_unary_age_in_scan_argument_is_rejected_before_binding(tmp_path, entry, timestamp):
    from tests.fast.test_execution_submission import options, prepare
    from vane.execution.query_options import DistributedMode

    missing = tmp_path / "missing.parquet"
    sql = f"select * from read_parquet('{missing}', binary_as_string=(age({timestamp} '2000-01-01')>interval '0'))"
    with vane.connect() as connection:
        # Reject the unfrozen date before the binder evaluates the argument or
        # tries to open the deliberately absent file.
        with pytest.raises(vane.NotImplementedException, match="consistent across tasks and attempts: age"):
            if entry == "compiler":
                compile_sql(connection, sql)
            else:
                prepare(connection, sql, options=options(DistributedMode(entry)))


@pytest.mark.parametrize("entry", ["compiler", "pipelined", "fte"])
def test_timetz_timezone_is_checked_before_table_argument_evaluation(tmp_path, entry):
    from tests.fast.test_execution_submission import options, prepare
    from vane.execution.query_options import DistributedMode

    missing = tmp_path / "missing.parquet"
    sql = (
        f"select * from read_parquet('{missing}', binary_as_string="
        "(timezone('America/New_York', timetz '12:00:00+00')::varchar <> ''))"
    )
    with vane.connect() as connection:
        with pytest.raises(vane.NotImplementedException, match="consistent across tasks and attempts: timezone"):
            if entry == "compiler":
                compile_sql(connection, sql)
            else:
                prepare(connection, sql, options=options(DistributedMode(entry)))


@pytest.mark.parametrize(
    "sql",
    [
        "select (select range from range(2))",
        "select a.range, (select b.range from range(9) b where b.range % 3=a.range) from range(3) a",
    ],
)
def test_scalar_subquery_rejects_multiple_rows(sql):
    with vane.connect(config={"threads": 1}) as connection:
        with pytest.raises(vane.InvalidInputException, match="More than one row"):
            connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, partitions=3)
        with pytest.raises(vane.InvalidInputException, match="More than one row"):
            execute_graph(connection, graph)


@pytest.mark.parametrize(
    "sql",
    [
        "select (select nextval('s'))",
        "select 1 where exists (select * from read_parquet('missing.parquet', binary_as_string=(nextval('s')>0)))",
        "with x as (select nextval('s')) select * from x",
        "select 1 where exists (select * from read_parquet('missing.parquet', binary_as_string=(current_user>0)))",
    ],
)
def test_nested_query_validation_precedes_binding_side_effects(sql):
    with vane.connect() as connection:
        connection.execute("create sequence s start 42")
        connection.execute("create macro current_user() as nextval('s')")
        with pytest.raises(vane.NotImplementedException):
            compile_sql(connection, sql)
        assert connection.execute("select nextval('s')").fetchone() == (42,)


@pytest.mark.parametrize("sql", ["select (select current_timestamp)", "select 1 where exists (select random())"])
def test_nested_query_rejects_unfrozen_expressions(sql):
    with vane.connect() as connection:
        connection.execute("set disabled_optimizers='expression_rewriter'")
        with pytest.raises(vane.NotImplementedException, match="consistent"):
            compile_sql(connection, sql)


@pytest.mark.parametrize("cte", [False, True])
def test_pruned_subquery_scan_remains_a_snapshot_dependency(tmp_path, cte):
    from tests.fast.test_execution_submission import prepare
    from vane.execution.submission import prepare_worker_plan

    path = tmp_path / "data.parquet"
    with vane.connect() as connection:
        connection.execute(f"copy (select range k from range(3)) to '{path}' (format parquet)")
        scan = f"select k from read_parquet('{path}') where k>10"
        sql = f"with x as materialized ({scan}) select (select count(*) from x)" if cte else f"select ({scan})"
        spec = prepare(connection, sql)
        assert any(fragment.source_dependencies for fragment in spec.graph.fragments)
        with vane.connect() as worker:
            prepare_worker_plan(worker, spec)
        with path.open("ab") as output:
            output.write(b"changed")
        with vane.connect() as worker, pytest.raises(vane.InvalidInputException, match="snapshot changed"):
            prepare_worker_plan(worker, spec)
