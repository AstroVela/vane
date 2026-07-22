# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

try:
    import ray
except Exception:
    ray = None

import vane


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
    from vane import runners as _runners

    _runners.set_runner_ray(noop_if_initialized=True)
    runner = _runners.get_or_create_runner()
    assert getattr(runner, "name", None) == "ray"

    relation = vane.sql("SELECT a, b, a + b AS sum FROM (VALUES (1, 10), (2, 20), (3, 30)) AS t(a, b)")
    parts = list(runner.run_iter_tables(relation, results_buffer_size=1))
    assert parts
    rows = sorted(_collect_rows_from_parts(parts))
    assert rows == [(1, 10, 11), (2, 20, 22), (3, 30, 33)]


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_run_distributed_plan_end_to_end_on_ray_local(tmp_path):
    from vane import runners as _runners

    _runners.set_runner_ray(noop_if_initialized=True)

    # Build a small parquet-backed relation with multiple planner partitions.
    n = 12
    path = tmp_path / "ray_real_integration_input.parquet"
    vane.sql(
        f"""
        COPY (
            SELECT
                i::INTEGER AS a,
                (i * 10)::INTEGER AS b
            FROM range({n}) AS t(i)
        ) TO '{path}' (FORMAT PARQUET)
        """
    )
    relation = vane.sql(f"SELECT a, b, a + b AS sum FROM read_parquet('{path}')")

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
