# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Modified Ray e2e test that focuses on serialization verification.
Tests the core serialization functionality without requiring full execution pipeline.
"""

import csv
import datetime
import pickle

import pytest

try:
    import ray
except Exception:
    ray = None

import vane


def test_json_scan_family_plans_serialize_for_ray(tmp_path):
    """Every JSON multi-file alias retains a complete worker-owned bind."""
    input_path = tmp_path / "json-family-plans.ndjson"
    input_path.write_text('{"id":1,"value":"a"}\n{"id":2,"value":"b"}\n', encoding="utf-8")
    queries = {
        "read_json": f"""
            SELECT * FROM read_json(
                '{input_path}',
                format='newline_delimited',
                auto_detect=false,
                columns={{'id': 'BIGINT', 'value': 'VARCHAR'}}
            )
        """,
        "read_json_auto": f"SELECT * FROM read_json_auto('{input_path}')",
        "read_ndjson": f"SELECT * FROM read_ndjson('{input_path}')",
        "read_ndjson_auto": f"SELECT * FROM read_ndjson_auto('{input_path}')",
        "read_json_objects": f"SELECT * FROM read_json_objects('{input_path}')",
        "read_json_objects_auto": f"SELECT * FROM read_json_objects_auto('{input_path}')",
        "read_ndjson_objects": f"SELECT * FROM read_ndjson_objects('{input_path}')",
    }

    for function_name, query in queries.items():
        connection = vane.connect()
        try:
            relation = connection.sql(query)
            logical_plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
                relation,
                f"{function_name}-serialization-plan",
            )
            restored_plan = pickle.loads(pickle.dumps(logical_plan))
            physical_plan = restored_plan.to_physical_plan(connection)
            assert physical_plan.num_partitions() == 1
            assert [len(batches) for batches in physical_plan.scan_split_batch_map().values()] == [1]
        finally:
            connection.close()


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_ray_plan_serialization_core():
    """Test that plans can be created and serialized for Ray distribution."""
    # Use pure SQL to avoid pandas_scan serialization issues
    n = 12
    values_list = [f"({i}, {i * 10})" for i in range(n)]
    values_clause = ", ".join(values_list)
    sql = f"SELECT * FROM (VALUES {values_clause}) AS t(a, b)"
    df = vane.sql(sql)

    # vane.sql(...) returns a DuckDB relation
    rel = df

    # Create PyLogicalPlan (this triggers LogicalPlan serialization)
    ray_cxx = getattr(vane, "ray_cxx", None)
    if ray_cxx is None or not hasattr(ray_cxx, "PyLogicalPlan"):
        pytest.skip("vane.ray_cxx.PyLogicalPlan not available in this environment")
    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(rel, "test-e2e-query")

    assert plan.idx() == "test-e2e-query"
    assert plan.idx() == "test-e2e-query"

    # Test pickling (this is what Ray uses for serialization)
    serialized = pickle.dumps(plan)
    assert len(serialized) > 0

    # Test unpickling in same process
    restored_plan = pickle.loads(serialized)
    assert restored_plan.idx() == "test-e2e-query"
    conn = vane.connect()
    restored_dist_plan = restored_plan.to_physical_plan(conn)
    assert restored_dist_plan.num_partitions() >= 1

    # Test cross-worker serialization
    @ray.remote
    def verify_plan_in_worker(plan):
        """Verify plan can be received and accessed in Ray worker."""
        import vane

        assert plan.idx() == "test-e2e-query"
        conn = vane.connect()
        dist_plan = plan.to_physical_plan(conn)
        assert dist_plan.num_partitions() >= 1
        return {"success": True, "idx": plan.idx(), "num_partitions": dist_plan.num_partitions()}

    result_ref = verify_plan_in_worker.remote(plan)
    result = ray.get(result_ref, timeout=10)

    assert result["success"] is True
    assert result["idx"] == "test-e2e-query"
    assert result["num_partitions"] >= 1


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_streaming_metadata_and_rows_full_execution(tmp_path):
    """Execute a serialized Parquet-backed plan through the real Ray runner."""
    import pyarrow as pa

    input_path = tmp_path / "ray_serialization_execution.parquet"
    connection = vane.connect()
    connection.execute(
        f"""
        COPY (
            SELECT i::INTEGER AS value, (i * 10)::INTEGER AS scaled
            FROM range(12) AS t(i)
        ) TO '{input_path}' (FORMAT PARQUET)
        """
    )
    relation = connection.sql(f"SELECT value, scaled FROM read_parquet('{input_path}') WHERE value % 2 = 0")

    from vane import runners

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    parts = list(runner.run_iter_tables(relation))
    tables = [part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts]
    result = pa.concat_tables(tables)

    assert sorted(zip(result.column(0).to_pylist(), result.column(1).to_pylist())) == [
        (0, 0),
        (2, 20),
        (4, 40),
        (6, 60),
        (8, 80),
        (10, 100),
    ]


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_single_csv_file_uses_explicit_byte_ranges_through_real_ray(tmp_path, monkeypatch):
    """A single absolute CSV path is split without losing quoted or multibyte rows."""
    import pyarrow as pa

    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "4")
    monkeypatch.setenv("VANE_RAY_SCAN_SPLIT_MIN_COUNT", "4")
    monkeypatch.setenv("VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION", "1")

    input_path = tmp_path / "single-distributed.csv"
    expected = []
    with input_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter="|", lineterminator="\r\n")
        writer.writerow(("id", "payload", "unused"))
        for row_id in range(5000):
            if row_id % 7 == 0:
                payload = f'quoted | value "{row_id}"\n继续-{row_id}'
            elif row_id % 11 == 0:
                payload = f"你好🙂-{row_id}"
            else:
                payload = f"value-{row_id}"
            writer.writerow((row_id, payload, row_id * 10))
            if row_id % 3 == 1:
                expected.append((row_id, payload, 0))

    connection = vane.connect()
    relation = connection.sql(
        f"""
        SELECT id, payload, file_index
        FROM read_csv(
            '{input_path}',
            delim='|',
            header=true,
            auto_detect=false,
            columns={{'id': 'INTEGER', 'payload': 'VARCHAR', 'unused': 'BIGINT'}},
            buffer_size=65536,
            max_line_size=16384
        )
        WHERE id % 3 = 1
        """
    )
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        "single-csv-byte-range-plan",
    ).to_physical_plan(connection)
    assert [len(batches) for batches in plan.scan_split_batch_map().values()] == [4]

    from vane import runners

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    parts = list(runner.run_iter_tables(relation))
    assert len(parts) == 4
    tables = [part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts]
    result = pa.concat_tables(tables)

    assert (
        sorted(
            zip(
                result.column(0).to_pylist(),
                result.column(1).to_pylist(),
                result.column(2).to_pylist(),
            )
        )
        == expected
    )
    connection.close()


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_multi_file_csv_union_reader_state_survives_worker_serde(tmp_path, monkeypatch):
    """Each assigned file retains the dialect and schema selected by union_by_name bind."""
    import pyarrow as pa

    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "2")
    monkeypatch.setenv("VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION", "1")
    comma_path = tmp_path / "comma.csv"
    pipe_path = tmp_path / "pipe.csv"
    comma_path.write_text("id,left_value\n1,alpha\n2,beta\n", encoding="utf-8")
    pipe_path.write_text("id|right_value\n3|gamma\n4|delta\n", encoding="utf-8")

    connection = vane.connect()
    relation = connection.sql(
        f"""
        SELECT id, coalesce(left_value, right_value) AS value, file_index
        FROM read_csv_auto(
            ['{comma_path}', '{pipe_path}'],
            union_by_name=true
        )
        WHERE id >= 2
        """
    )
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        "multi-csv-union-plan",
    ).to_physical_plan(connection)
    assert [len(batches) for batches in plan.scan_split_batch_map().values()] == [2]

    from vane import runners

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    parts = list(runner.run_iter_tables(relation))
    assert len(parts) == 2
    tables = [part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts]
    result = pa.concat_tables(tables)
    assert sorted(
        zip(
            result.column(0).to_pylist(),
            result.column(1).to_pylist(),
            result.column(2).to_pylist(),
        )
    ) == [
        (2, "beta", 0),
        (3, "gamma", 1),
        (4, "delta", 1),
    ]
    connection.close()


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_multi_file_json_bind_state_survives_real_ray_serde(tmp_path, monkeypatch):
    """Ray workers retain JSON options, inferred schema, formats, and multi-file metadata."""
    import pyarrow as pa

    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "2")
    monkeypatch.setenv("VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION", "1")
    left_path = tmp_path / "left.ndjson"
    right_path = tmp_path / "right.ndjson"
    left_path.write_text(
        '{"id":1,"event_date":"08-25-2026","event_time":"08-25-2026 03:04:05 PM","left_value":"alpha"}\n',
        encoding="utf-8",
    )
    right_path.write_text(
        '{"id":2,"event_date":"08-26-2026","event_time":"08-26-2026 04:05:06 PM","right_value":"beta"}\n',
        encoding="utf-8",
    )

    connection = vane.connect()
    relation = connection.sql(
        f"""
        SELECT id, event_date, event_time, coalesce(left_value, right_value) AS value, filename
        FROM read_ndjson_auto(
            ['{left_path}', '{right_path}'],
            union_by_name=true,
            filename=true,
            dateformat='%m-%d-%Y',
            timestampformat='%m-%d-%Y %I:%M:%S %p'
        )
        """
    )
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        "multi-json-bind-serde-plan",
    ).to_physical_plan(connection)
    assert [len(batches) for batches in plan.scan_split_batch_map().values()] == [2]

    from vane import runners

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    parts = list(runner.run_iter_tables(relation))
    assert len(parts) == 2
    tables = [part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts]
    result = pa.concat_tables(tables)
    assert sorted(
        zip(
            result.column(0).to_pylist(),
            result.column(1).to_pylist(),
            result.column(2).to_pylist(),
            result.column(3).to_pylist(),
            result.column(4).to_pylist(),
        )
    ) == [
        (
            1,
            datetime.date(2026, 8, 25),
            datetime.datetime(2026, 8, 25, 15, 4, 5),
            "alpha",
            str(left_path),
        ),
        (
            2,
            datetime.date(2026, 8, 26),
            datetime.datetime(2026, 8, 26, 16, 5, 6),
            "beta",
            str(right_path),
        ),
    ]
    connection.close()
