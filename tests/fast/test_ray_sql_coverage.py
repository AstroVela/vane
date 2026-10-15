# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Subqueries and shared native state through both public Ray execution modes."""

from collections import Counter
from dataclasses import replace

import pytest

import vane
from tests.fast.test_fragment_sql_coverage import (
    CONTEXT_READS,
    DETERMINISTIC_AGES,
    DETERMINISTIC_LAMBDAS,
    DETERMINISTIC_TIMEZONES,
    IE_JOIN_CASES,
    IE_JOIN_EQUALITY_TAIL,
    ORDERED_DISTINCT_CASES,
    RANGE_JOIN_CASES,
    SQL_CASES,
    UNSTABLE_AGES,
    UNSTABLE_LAMBDAS,
    UNSTABLE_TIMEZONES,
    correlated_owner_queries,
    correlated_parquet_source,
    correlated_partition_cases,
    correlation_source,
    early_join_filter_query,
    large_join_key_query,
)
from tests.fast.test_ray_analytical_execution import resources
from tests.fast.test_ray_recovery_runtime import assert_idle, options

pytestmark = [pytest.mark.real_ray, pytest.mark.usefixtures("ray_local")]


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_ordered_distinct_on_keeps_complete_groups(tmp_path, mode):
    with vane.connect() as local, vane.Runtime(resources(tmp_path)) as application, application.connect() as connection:
        for sql in ORDERED_DISTINCT_CASES:
            expected = local.execute(sql).fetchall()
            with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
                table = result.collect()
                actual = list(zip(*(column.to_pylist() for column in table.columns)))
                assert actual == expected, sql
                assert result.execution_state == "SUCCEEDED"
            assert_idle(connection)


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
@pytest.mark.parametrize("source_kind", ["range", "parquet"])
def test_public_partitioned_correlated_queries(tmp_path, mode, source_kind):
    limits = resources(tmp_path, max_active_queries=1, task_contexts_per_worker=64, io_concurrency=256)
    with (
        vane.connect(config={"threads": 1}) as local,
        vane.Runtime(limits) as application,
        application.connect() as connection,
    ):
        source = correlated_parquet_source(local, tmp_path) if source_kind == "parquet" else correlation_source()
        cases = correlated_partition_cases(source)
        cases += [f"select a.v, (select avg(b.v) from {source} b where b.k=a.k and b.v>a.v) av from {source} a"]
        cases += correlated_partition_cases(correlation_source(), correlation_source(0))[:3]
        cases += correlated_partition_cases(correlation_source(0), correlation_source())[:3]
        for sql in cases:
            expected = local.execute(sql).fetchall()
            with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
                table = result.collect()
                assert Counter(zip(*(c.to_pylist() for c in table.columns))) == Counter(expected), sql
                assert result.execution_state == "SUCCEEDED"
            assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_correlated_key_filter_repeats_bound_parquet_scan(tmp_path, mode):
    fact, keys = tmp_path / "fact.parquet", tmp_path / "keys.parquet"
    config = {"threads": 4}
    limits = resources(tmp_path, max_active_queries=1, task_contexts_per_worker=64, io_concurrency=256)
    with (
        vane.connect(backend="local", config=config) as local,
        vane.Runtime(limits) as application,
        application.connect(config=config) as connection,
    ):
        local.execute(f"copy (select range%4096 k, range%97 v from range(131072)) to '{fact}' (row_group_size 2048)")
        local.execute(f"copy (select range k, range in (2,2001,4001) keep from range(4096)) to '{keys}'")
        sql = (
            f"select p.k, f.v from read_parquet('{fact}') f join read_parquet('{keys}') p on f.k=p.k "
            f"where p.keep and f.v < (select 0.2*avg(i.v) from read_parquet('{fact}') i where i.k=p.k) "
            "order by p.k, f.v"
        )
        expected = local.execute(sql).to_arrow_table().to_pylist()
        for _ in range(2):
            with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
                assert result.collect().to_pylist() == expected
            assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_correlated_owner_keys_and_shared_branches(tmp_path, mode):
    limits = resources(tmp_path, max_active_queries=1, task_contexts_per_worker=64, io_concurrency=256)
    with (
        vane.connect(backend="local", config={"threads": 4}) as local,
        vane.Runtime(limits) as application,
        application.connect() as connection,
    ):
        for index, comparison in enumerate(("=", "is not distinct from")):
            queries = correlated_owner_queries(local, tmp_path / f"fact-{index}.parquet", comparison)
            for sql in queries.values():
                expected = local.execute(sql).to_arrow_table().to_pylist()
                with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
                    assert result.collect().to_pylist() == expected
                assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_join_filter_before_exchange(tmp_path, mode):
    limits = resources(tmp_path, max_active_queries=1, task_contexts_per_worker=64, io_concurrency=256)
    with (
        vane.connect(backend="local") as local,
        vane.Runtime(limits) as application,
        application.connect() as connection,
    ):
        for index, (kind, comparison) in enumerate(
            [("inner", "="), ("semi", "="), ("left", "="), ("anti", "is not distinct from")]
        ):
            sql = early_join_filter_query(local, tmp_path / f"fact-{index}.parquet", kind, comparison, wide_probe=True)
            if kind in ("left", "anti"):
                sql += " limit 4096"
            expected = local.execute(sql).to_arrow_table().to_pylist()
            with connection.query(sql, options=options(execution=90) if mode == "fte" else None) as result:
                assert result.collect().to_pylist() == expected
            assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
@pytest.mark.parametrize("domain", ["exact", "hashed", "estimated", "compressed", "join_estimate"])
def test_public_key_domain_before_hash_exchange(tmp_path, mode, domain):
    limits = resources(tmp_path, max_active_queries=1, task_contexts_per_worker=64, io_concurrency=256)
    if domain in ("estimated", "compressed", "join_estimate"):
        limits = replace(
            limits, exchange_stores=tuple(replace(store, source_bytes=128 << 20) for store in limits.exchange_stores)
        )
    with (
        vane.connect(backend="local") as local,
        vane.Runtime(limits) as application,
        application.connect() as connection,
    ):
        sql = large_join_key_query(local, tmp_path / "fact.parquet", domain)
        expected = local.execute(sql).to_arrow_table().to_pylist()
        with connection.query(sql, options=options(execution=90) if mode == "fte" else None) as result:
            assert result.collect().to_pylist() == expected
        assert_idle(connection)


@pytest.mark.parametrize(
    "query_kind", ["correlated", "join", "exact", "hashed", "estimated", "compressed", "join_estimate"]
)
def test_fte_shared_build_reuses_committed_fanout_after_worker_loss(tmp_path, monkeypatch, query_kind):
    import ray

    from vane.execution.recovery_runtime import RecoveryScheduler

    dispatch, lost = RecoveryScheduler._dispatch, []

    def lose_once(scheduler, index, partition, binding, upstream):
        admitted = dispatch(scheduler, index, partition, binding, upstream)
        # Kill a consumer after a build shared by several stages has committed.
        producers = {
            e.producer_fragment_id for e in scheduler.spec.graph.exchanges if e.producer_fragment_id in upstream
        }
        shared = any(
            len({e.consumer_fragment_id for e in scheduler.spec.graph.exchanges if e.producer_fragment_id == p}) > 1
            for p in producers
        )
        if admitted and shared and not lost:
            attempt = scheduler.active[index]
            lost.append(attempt.reservation.token)
            ray.kill(attempt.worker, no_restart=True)
        return admitted

    monkeypatch.setattr(RecoveryScheduler, "_dispatch", lose_once)
    limits = resources(tmp_path, max_active_queries=1, task_contexts_per_worker=64, io_concurrency=256)
    if query_kind in ("estimated", "compressed", "join_estimate"):
        limits = replace(
            limits, exchange_stores=tuple(replace(store, source_bytes=128 << 20) for store in limits.exchange_stores)
        )
    with (
        vane.connect(backend="local") as local,
        vane.Runtime(limits) as application,
        application.connect() as connection,
    ):
        path = tmp_path / "fact.parquet"
        if query_kind in ("exact", "hashed", "estimated", "compressed", "join_estimate"):
            sql = large_join_key_query(local, path, query_kind)
        else:
            sql = (
                early_join_filter_query(local, path, wide_probe=True)
                if query_kind == "join"
                else correlated_owner_queries(local, path, "is not distinct from")["nested"]
            )
        expected = local.execute(sql).to_arrow_table().to_pylist()
        with connection.query(sql, options=options(execution=90)) as result:
            scheduler = result.context._reader
            assert result.collect().to_pylist() == expected
            assert len(lost) == 1
            attempts = [t for t in scheduler.history if t.task_id == lost[0].task_id]
            assert len(attempts) == 2
            assert attempts[0].input_id == attempts[1].input_id
            assert attempts[0].fence != attempts[1].fence
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
