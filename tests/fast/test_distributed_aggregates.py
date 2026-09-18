# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import pytest

import vane


@pytest.fixture
def aggregate_connections(ray_local, monkeypatch, tmp_path):
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    monkeypatch.setenv("VANE_SHUFFLE_ALGORITHM", "flight_shuffle")
    monkeypatch.setenv("VANE_SHUFFLE_LOCAL_DIRS", str(tmp_path / "shuffle"))
    monkeypatch.setenv("VANE_RAY_SCAN_SPLIT_MIN_COUNT", "4")
    monkeypatch.setenv("VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION", "1")
    local = vane._native._connect_with_runner("local-fast")
    distributed = vane.connect()
    try:
        # SUM(BIGINT) returns HUGEINT, which requires lossless Arrow conversion.
        for connection in (local, distributed):
            connection.execute("SET arrow_lossless_conversion=true")
        yield local, distributed
    finally:
        distributed.close()
        local.close()


@pytest.fixture
def aggregate_source(aggregate_connections, tmp_path):
    local, _ = aggregate_connections
    directory = tmp_path / "aggregate_input"
    directory.mkdir()
    # Each group spans all four files; values include duplicates and NULLs.
    for file_id in range(4):
        local.execute(f"""
            COPY (
                SELECT i AS id,
                       CASE WHEN i % 7 = 0 THEN NULL ELSE (i % 3)::INTEGER END AS k,
                       CASE WHEN i % 5 = 0 THEN NULL ELSE (i % 4)::VARCHAR END AS v,
                       CASE WHEN i % 6 = 0 THEN NULL ELSE 24 - i END AS seq,
                       i % 3 <> 0 AS keep,
                       CASE WHEN i % 5 = 0 THEN NULL
                            WHEN i % 5 = 1 THEN []::BIGINT[]
                            ELSE [i, NULL, -i] END AS items
                FROM range(24) t(i) WHERE i % 4 = {file_id}
            ) TO '{directory / f"{file_id}.parquet"}' (FORMAT PARQUET)
        """)
    return f"read_parquet('{directory}/*.parquet')"


def _assert_same_rows(local, distributed, sql, *, unordered_list_columns=()):
    expected = local.sql(sql).fetchall()
    actual = distributed.sql(sql).fetchall()

    def normalized(rows):
        result = []
        for row in rows:
            row = list(row)
            for index in unordered_list_columns:
                if row[index] is not None:
                    # Preserve duplicates and NULL elements, without imposing
                    # an order on an aggregate that has no ORDER BY clause.
                    row[index] = sorted(Counter(map(repr, row[index])).items())
            result.append(repr(row))
        return sorted(result)

    assert normalized(actual) == normalized(expected), sql


def _distributed_plan(local, sql):
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(local.sql(sql), None)
    plan = logical.to_physical_plan(local)
    # Constructing the wrapper alone does not translate the plan.
    return plan, plan.repr_ascii(False)


def test_list_over_union_and_values(aggregate_connections):
    local, distributed = aggregate_connections
    sources = [
        "(SELECT 1 AS k, 'a' AS v UNION ALL SELECT 1, 'b' UNION ALL SELECT 2, 'c') t",
        "(VALUES (1, 'a'), (1, 'b'), (2, 'c')) t(k, v)",
    ]
    for source in sources:
        for expression, unordered in [
            ("list(v)", (1,)),
            ("list(v ORDER BY v DESC NULLS FIRST)", ()),
            ("count(*), list(v)", (2,)),
        ]:
            _assert_same_rows(
                local,
                distributed,
                f"SELECT k, {expression} FROM {source} GROUP BY k",
                unordered_list_columns=unordered,
            )
        _assert_same_rows(local, distributed, f"SELECT list(v ORDER BY v DESC) FROM {source}")


@pytest.mark.parametrize("perfect_hash_threshold", [0, 32])
def test_complete_aggregates_over_multiple_scan_splits(aggregate_connections, aggregate_source, perfect_hash_threshold):
    local, distributed = aggregate_connections
    for connection in (local, distributed):
        connection.execute(f"SET perfect_ht_threshold={perfect_hash_threshold}")
    cases = [
        ("list(v)", (1,)),
        ("list(v ORDER BY v DESC NULLS FIRST)", ()),
        ("list(v ORDER BY seq DESC NULLS FIRST, id)", ()),
        ("list(encode(v) ORDER BY seq ASC NULLS LAST, id)", ()),
        ("list(struct_pack(id := id, image := encode(v)) ORDER BY seq, id)", ()),
        ("list(items ORDER BY id)", ()),
        ("list(DISTINCT v ORDER BY v) FILTER (WHERE keep)", ()),
        ("count(DISTINCT v), sum(id) FILTER (WHERE keep)", ()),
        (
            "count(*) FILTER (WHERE keep), sum(id) FILTER (WHERE keep), list(v ORDER BY seq, id) FILTER (WHERE keep)",
            (),
        ),
        ("string_agg(v, ',' ORDER BY seq, id) FILTER (WHERE keep)", ()),
        ("min(v), max(v)", ()),
        ("count(*), sum(id), list(v)", (3,)),
    ]
    for expression, unordered in cases:
        sql = f"SELECT k, {expression} FROM {aggregate_source} GROUP BY k"
        plan, text = _distributed_plan(local, sql)
        assert [len(splits) for splits in plan.scan_split_batch_map().values()] == [4]
        assert text.count("Aggregate Node") == 1, text
        assert "Repartition" in text, text
        _assert_same_rows(local, distributed, sql, unordered_list_columns=unordered)

    sql = f"SELECT k, count(*), sum(id) FROM {aggregate_source} GROUP BY k"
    _, text = _distributed_plan(local, sql)
    assert text.count("Aggregate Node") == 2, text
    _assert_same_rows(local, distributed, sql)


def test_global_and_empty_complete_aggregates(aggregate_connections, aggregate_source):
    local, distributed = aggregate_connections
    for expression, unordered in [
        ("list(v)", (0,)),
        ("list(v ORDER BY seq DESC NULLS FIRST, id)", ()),
        ("count(*), list(v ORDER BY v), sum(id) FILTER (WHERE keep)", ()),
        ("count(DISTINCT v), list(DISTINCT v ORDER BY v) FILTER (WHERE keep)", ()),
        ("string_agg(v, ',' ORDER BY seq, id), min(v), max(v)", ()),
    ]:
        sql = f"SELECT {expression} FROM {aggregate_source}"
        plan, text = _distributed_plan(local, sql)
        assert [len(splits) for splits in plan.scan_split_batch_map().values()] == [4]
        assert text.count("Aggregate Node") == 1, text
        assert plan.num_partitions() == 1
        _assert_same_rows(local, distributed, sql, unordered_list_columns=unordered)
        _assert_same_rows(local, distributed, sql + " WHERE id % 100 = 99", unordered_list_columns=unordered)

    _assert_same_rows(
        local,
        distributed,
        f"SELECT k, count(*), list(v ORDER BY id) FROM {aggregate_source} WHERE id % 100 = 99 GROUP BY k",
    )
    _assert_same_rows(
        local,
        distributed,
        f"SELECT k, count(*) FILTER (WHERE id < 0), list(v) FILTER (WHERE id < 0) FROM {aggregate_source} GROUP BY k",
    )


def test_list_sort_functions_use_execution_context(aggregate_connections, aggregate_source):
    local, distributed = aggregate_connections
    for connection in (local, distributed):
        connection.execute("SET default_order='DESC'")
        connection.execute("SET default_null_order='NULLS_FIRST'")
    sql = f"""
        SELECT id, list_sort(items), list_sort(items, 'ASC'),
               list_sort(items, 'DESC', 'NULLS_LAST'), list_reverse_sort(items),
               list_reverse_sort(items, 'NULLS_LAST'), list_grade_up(items),
               list_grade_up(items, 'ASC'), list_grade_up(items, 'DESC', 'NULLS_LAST')
        FROM {aggregate_source}
    """
    _distributed_plan(local, sql)
    # Repeated execution exercises fresh local sort state after each transaction.
    _assert_same_rows(local, distributed, sql)
    _assert_same_rows(local, distributed, sql)
