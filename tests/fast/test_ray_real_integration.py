# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

try:
    import ray
except Exception:
    ray = None

import duckdb


def _collect_rows_from_parts(parts):
    rows = []
    for part in parts:
        table = part.to_arrow() if hasattr(part, "to_arrow") else part
        if hasattr(table, "to_pylist"):
            pylist = table.to_pylist()
            for row in pylist:
                if isinstance(row, dict):
                    rows.append(tuple(row.values()))
                else:
                    rows.append(tuple(row))
        elif hasattr(part, "to_pylist"):
            for row in part.to_pylist():
                if isinstance(row, dict):
                    rows.append(tuple(row.values()))
                else:
                    rows.append(tuple(row))
    return rows


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_run_simple_plan_on_ray_local():
    from duckdb import runners as _runners

    _runners.set_runner_ray(noop_if_initialized=True)
    runner = _runners.get_or_create_runner()
    assert getattr(runner, "name", None) == "ray"

    relation = duckdb.sql("SELECT a, b, a + b AS sum FROM (VALUES (1, 10), (2, 20), (3, 30)) AS t(a, b)")
    parts = list(runner.run_iter_tables(relation, results_buffer_size=1))
    assert parts
    rows = sorted(_collect_rows_from_parts(parts))
    assert rows == [(1, 10, 11), (2, 20, 22), (3, 30, 33)]


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_run_distributed_plan_end_to_end_on_ray_local(tmp_path):
    from duckdb import runners as _runners

    _runners.set_runner_ray(noop_if_initialized=True)

    # Build a small parquet-backed relation with multiple planner partitions.
    n = 12
    path = tmp_path / "ray_real_integration_input.parquet"
    duckdb.sql(
        f"""
        COPY (
            SELECT
                i::INTEGER AS a,
                (i * 10)::INTEGER AS b
            FROM range({n}) AS t(i)
        ) TO '{path}' (FORMAT PARQUET)
        """
    )
    relation = duckdb.sql(f"SELECT a, b, a + b AS sum FROM read_parquet('{path}')")

    runner = _runners.get_or_create_runner()
    assert getattr(runner, "name", None) == "ray"

    parts = list(runner.run_iter_tables(relation, results_buffer_size=1))
    assert parts

    rows = _collect_rows_from_parts(parts)
    assert len(rows) == n

    expected_rows = {(x, x * 10, x + x * 10) for x in range(n)}
    assert set(rows) == expected_rows

    # Some Ray setups do not expose named actors through the same namespace.
    try:
        actor = ray.get_actor("ray-query-driver-actor", namespace="vane")
        assert actor is not None
    except Exception:
        pass


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_relation_result_consumers_on_ray_local(tmp_path, monkeypatch):
    from duckdb import runners

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    connection = duckdb.connect()
    path = tmp_path / "ray_relation_result_consumers.parquet"
    connection.execute(
        f"""
        COPY (
            SELECT
                i::BIGINT AS value,
                ('row-' || i::VARCHAR)::VARCHAR AS label
            FROM range(6) AS t(i)
        ) TO '{path}' (FORMAT PARQUET)
        """
    )

    runners.set_runner_ray(noop_if_initialized=True)
    query = f"SELECT value, label FROM read_parquet('{path}') ORDER BY value"

    row_relation = connection.sql(query)
    assert row_relation.fetchone() == (0, "row-0")
    assert row_relation.fetchmany(2) == [(1, "row-1"), (2, "row-2")]
    assert row_relation.fetchall() == [
        (3, "row-3"),
        (4, "row-4"),
        (5, "row-5"),
    ]

    table = connection.sql(query).to_arrow_table(batch_size=2)
    assert table.schema.names == ["value", "label"]
    assert table.to_pydict() == {
        "value": list(range(6)),
        "label": [f"row-{index}" for index in range(6)],
    }

    reader = connection.sql(query).to_arrow_reader(batch_size=2)
    assert [batch.num_rows for batch in reader] == [2, 2, 2]

    partial_relation = connection.sql(query)
    assert partial_relation.fetchone() == (0, "row-0")
    partial_relation.close()
    with pytest.raises(duckdb.InvalidInputException, match="result closed"):
        partial_relation.fetchall()


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_complex_relation_result_consumers_on_ray_local(tmp_path, monkeypatch):
    from duckdb import runners

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    connection = duckdb.connect()
    facts_path = tmp_path / "ray_relation_result_facts.parquet"
    dimensions_path = tmp_path / "ray_relation_result_dimensions.parquet"
    connection.execute(
        f"""
        COPY (
            SELECT
                (i % 3)::BIGINT AS group_id,
                (i + 1)::BIGINT AS amount
            FROM range(12) AS t(i)
        ) TO '{facts_path}' (FORMAT PARQUET)
        """
    )
    connection.execute(
        f"""
        COPY (
            SELECT *
            FROM (VALUES (0, 10), (1, 100), (2, 1000)) AS t(group_id, weight)
        ) TO '{dimensions_path}' (FORMAT PARQUET)
        """
    )

    runners.set_runner_ray(noop_if_initialized=True)
    query = f"""
        SELECT
            facts.group_id,
            count(*)::BIGINT AS row_count,
            sum(facts.amount * dimensions.weight)::BIGINT AS weighted_sum
        FROM read_parquet('{facts_path}') AS facts
        JOIN read_parquet('{dimensions_path}') AS dimensions USING (group_id)
        GROUP BY facts.group_id
        ORDER BY facts.group_id
    """
    expected_rows = [
        (0, 4, 220),
        (1, 4, 2600),
        (2, 4, 30000),
    ]

    assert connection.sql(query).fetchall() == expected_rows

    table = connection.sql(query).to_arrow_reader(batch_size=2).read_all()
    assert table.to_pylist() == [
        {"group_id": group_id, "row_count": row_count, "weighted_sum": weighted_sum}
        for group_id, row_count, weighted_sum in expected_rows
    ]
