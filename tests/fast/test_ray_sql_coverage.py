# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Subqueries and shared native state through both public Ray execution modes."""

from collections import Counter

import pytest

import vane
from tests.fast.test_fragment_sql_coverage import (
    CONTEXT_READS,
    DETERMINISTIC_AGES,
    DETERMINISTIC_LAMBDAS,
    DETERMINISTIC_TIMEZONES,
    IE_JOIN_CASES,
    IE_JOIN_EQUALITY_TAIL,
    RANGE_JOIN_CASES,
    SQL_CASES,
    UNSTABLE_AGES,
    UNSTABLE_LAMBDAS,
    UNSTABLE_TIMEZONES,
)
from tests.fast.test_ray_analytical_execution import resources
from tests.fast.test_ray_recovery_runtime import assert_idle, options

pytestmark = [pytest.mark.real_ray, pytest.mark.usefixtures("ray_local")]


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_ie_join_preserves_equality_tail(tmp_path, mode):
    with (
        vane.connect(config={"threads": 1, "prefer_range_joins": True}) as local,
        vane.Runtime(resources(tmp_path)) as application,
        application.connect(config={"prefer_range_joins": True}) as connection,
    ):
        expected = local.execute(IE_JOIN_EQUALITY_TAIL).fetchall()
        with connection.query(
            IE_JOIN_EQUALITY_TAIL, options=options(execution=60) if mode == "fte" else None
        ) as result:
            table = result.collect()
            assert Counter(zip(*(c.to_pylist() for c in table.columns))) == Counter(expected)
        assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_ie_join_parallel_outer_payloads(tmp_path, mode):
    with (
        vane.connect(config={"threads": 1}) as local,
        vane.Runtime(resources(tmp_path, worker_count=1, cpus_per_worker=4)) as application,
        application.connect() as connection,
    ):
        for sql in IE_JOIN_CASES[-2:]:
            expected = local.execute(sql).fetchall()
            with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
                table = result.collect()
                assert Counter(zip(*(c.to_pylist() for c in table.columns))) == Counter(expected)
            assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_subquery_and_cte_semantics(tmp_path, mode):
    limits = resources(tmp_path, max_active_queries=1, max_results=1, task_contexts_per_worker=64, io_concurrency=256)
    with vane.connect() as local, vane.Runtime(limits) as application, application.connect() as connection:
        # Reuse the worker pool so CTE state cannot accidentally survive a query.
        for sql in SQL_CASES + RANGE_JOIN_CASES + IE_JOIN_CASES + SQL_CASES[-4:-2]:
            expected = local.execute(sql).fetchall()
            with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
                table = result.collect()
                assert Counter(zip(*(c.to_pylist() for c in table.columns))) == Counter(expected), sql
                assert result.execution_state == "SUCCEEDED"
            assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_lambda_validation_and_deterministic_execution(tmp_path, mode):
    with (
        vane.connect(config={"threads": 4}) as local,
        vane.Runtime(resources(tmp_path, cpus_per_worker=1)) as application,
        application.connect(config={"threads": 4}) as connection,
    ):
        for sql in UNSTABLE_LAMBDAS + CONTEXT_READS:
            with pytest.raises(
                vane.NotImplementedException, match="consistent across tasks|unfrozen connection context"
            ):
                connection.query(sql, options=options() if mode == "fte" else None)
            assert_idle(connection)
        for sql in DETERMINISTIC_LAMBDAS:
            expected = local.execute(sql).fetchall()
            with connection.query(sql, options=options() if mode == "fte" else None) as result:
                table = result.collect()
                actual = list(zip(*(c.to_pylist() for c in table.columns)))
                assert sorted(actual, key=repr) == sorted(expected, key=repr)
            assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_age_rejects_transaction_date_and_keeps_explicit_dates(tmp_path, mode):
    with vane.connect() as local, vane.Runtime(resources(tmp_path)) as application, application.connect() as connection:
        for sql in UNSTABLE_AGES:
            with pytest.raises(vane.NotImplementedException, match="consistent across tasks and attempts: age"):
                connection.query(sql, options=options() if mode == "fte" else None)
            assert_idle(connection)
        for sql in DETERMINISTIC_AGES:
            expected = local.execute(sql).fetchall()
            with connection.query(sql, options=options() if mode == "fte" else None) as result:
                table = result.collect()
                actual = list(zip(*(c.to_pylist() for c in table.columns)))
                assert sorted(actual, key=repr) == sorted(expected, key=repr)
            assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_timezone_checks_selected_overload(tmp_path, mode):
    with vane.connect() as local, vane.Runtime(resources(tmp_path)) as application, application.connect() as connection:
        for sql in UNSTABLE_TIMEZONES:
            with pytest.raises(vane.NotImplementedException, match="consistent across tasks and attempts: timezone"):
                connection.query(sql, options=options() if mode == "fte" else None)
            assert_idle(connection)
        for sql in DETERMINISTIC_TIMEZONES:
            expected = local.execute(sql).fetchall()
            with connection.query(sql, options=options() if mode == "fte" else None) as result:
                table = result.collect()
                actual = list(zip(*(c.to_pylist() for c in table.columns)))
                assert sorted(actual, key=repr) == sorted(expected, key=repr)
            assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_scalar_cardinality_failure_releases_query(tmp_path, mode):
    limits = resources(tmp_path, max_active_queries=1, max_results=1)
    with vane.Runtime(limits) as application, application.connect() as connection:
        sql = "select a.range, (select b.range from range(7) b where b.range % 2=a.range) from range(2) a"
        with pytest.raises(Exception, match="More than one row"):
            with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
                result.collect()
        assert_idle(connection)
        with connection.query("select 42", options=options() if mode == "fte" else None) as result:
            assert result.collect().to_pylist() == [{"42": 42}]
        assert_idle(connection)


def test_fte_freezes_sources_inside_scalar_subquery_and_cte(tmp_path):
    path = tmp_path / "source.parquet"
    with vane.connect() as local:
        local.execute(f"copy (select range k from range(13)) to '{path}' (format parquet)")
    scan = f"read_parquet('{path}')"
    cases = [
        f"select (select max(k) from {scan}) v",
        f"with x as materialized (select k from {scan}) select (select max(k) from x) v",
    ]
    with vane.Runtime(resources(tmp_path, max_active_queries=1)) as application, application.connect() as connection:
        for sql in cases:
            with connection.query(sql, options=options(execution=60)) as result:
                path.unlink()
                assert result.collect().to_pylist() == [{"v": 12}]
            assert_idle(connection)
            with vane.connect() as local:
                local.execute(f"copy (select range k from range(13)) to '{path}' (format parquet)")


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_scalar_cardinality_setting_reaches_workers(tmp_path, mode):
    with (
        vane.Runtime(resources(tmp_path)) as application,
        application.connect(config={"scalar_subquery_error_on_multiple_rows": "false"}) as connection,
    ):
        sql = "select a.range k, (select b.range from range(7) b where b.range % 2=a.range) v from range(2) a"
        with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
            rows = result.collect().to_pylist()
            assert sorted(row["k"] for row in rows) == [0, 1]
            assert all(row["v"] % 2 == row["k"] for row in rows)
        assert_idle(connection)
