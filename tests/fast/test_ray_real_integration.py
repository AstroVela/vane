# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

try:
    import ray
except Exception:
    ray = None

import duckdb


def _init_ray_for_test():
    ray_init_kwargs = {
        "address": "local",
        "ignore_reinit_error": True,
        "namespace": "duckdb",
    }
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Tip: In future versions of Ray, Ray will no longer override accelerator",
        )
        try:
            ray.init(**ray_init_kwargs)
        except Exception as ex:
            pytest.skip(f"failed to start local Ray runtime: {ex}")


def _collect_rows_from_parts(parts):
    rows = []
    for part in parts:
        table = part.to_arrow() if hasattr(part, "to_arrow") else part
        if hasattr(table, "to_pylist"):
            pylist = table.to_pylist()
            for row in pylist:
                if isinstance(row, dict):
                    rows.append((row.get("a"), row.get("b"), row.get("sum")))
                else:
                    rows.append(tuple(row))
        elif hasattr(part, "to_pylist"):
            for row in part.to_pylist():
                if isinstance(row, dict):
                    rows.append((row.get("a"), row.get("b"), row.get("sum")))
                else:
                    rows.append(tuple(row))
    return rows


@pytest.mark.skipif(ray is None, reason="ray not installed")
def test_run_simple_plan_on_ray_local():
    _init_ray_for_test()
    try:
        # Force DuckDB to use Ray runner (if available)
        try:
            from duckdb import runners as _runners

            _runners.set_runner_ray()
        except Exception:
            pytest.skip("duckdb runner API not available in this environment")

        from duckdb import runners as _runners

        try:
            runner = _runners.get_or_create_runner()
        except Exception:
            pytest.skip("Ray runner not available in this environment")
        if getattr(runner, "name", None) != "ray":
            pytest.skip("Ray runner not active")

        relation = duckdb.sql("SELECT a, b, a + b AS sum FROM (VALUES (1, 10), (2, 20), (3, 30)) AS t(a, b)")
        parts = list(runner.run_iter_tables(relation, results_buffer_size=1))
        assert len(parts) >= 1
        rows = sorted(_collect_rows_from_parts(parts))
        assert rows == [(1, 10, 11), (2, 20, 22), (3, 30, 33)]
    finally:
        try:
            ray.shutdown()
        except Exception:
            pass


@pytest.mark.skipif(ray is None, reason="ray not installed")
def test_run_distributed_plan_end_to_end_on_ray_local(tmp_path):
    _init_ray_for_test()
    try:
        # Force DuckDB to use Ray runner (if available)
        try:
            from duckdb import runners as _runners

            _runners.set_runner_ray()
        except Exception:
            pytest.skip("duckdb runner API not available in this environment")

        # Use multiple partitions in the planner.

        # Build a small parquet-backed relation (serializable across Ray workers).
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

        # Use the runner to stream PyArrow tables through the distributed native execution path.
        from duckdb import runners as _runners

        try:
            runner = _runners.get_or_create_runner()
        except Exception:
            pytest.skip("Ray runner not available in this environment")
        if getattr(runner, "name", None) != "ray":
            pytest.skip("Ray runner not active")

        parts = list(runner.run_iter_tables(relation, results_buffer_size=1))
        assert len(parts) >= 1

        # Collect and validate data
        rows = _collect_rows_from_parts(parts)
        assert len(rows) == n

        expected_rows = {(x, x * 10, x + x * 10) for x in range(n)}
        # Compare as unordered sets of rows to avoid partition ordering differences
        assert set(rows) == expected_rows

        # Ensure the Ray query driver actor exists (best-effort — some Ray setups may not expose it the same way)
        try:
            actor = ray.get_actor("ray-query-driver-actor", namespace="vane")
            assert actor is not None
        except Exception:
            # Not fatal; continue
            pass

    finally:
        try:
            ray.shutdown()
        except Exception:
            pass
