# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Native SQL semantics across subquery and shared-state fragment boundaries."""

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


@pytest.mark.parametrize("partitions", [1, 2, 5])
@pytest.mark.parametrize("sql", SQL_CASES)
def test_fragment_subqueries_and_shared_state(sql, partitions):
    with vane.connect(config={"threads": 1}) as connection:
        expected = connection.execute(sql).fetchall()
        graph = compile_sql(connection, sql, partitions=partitions)
        assert Counter(execute_graph(connection, graph)) == Counter(expected)


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
