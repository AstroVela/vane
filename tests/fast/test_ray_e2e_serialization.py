# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Modified Ray e2e test that focuses on serialization verification.
Tests the core serialization functionality without requiring full execution pipeline.
"""

import csv

import pytest

try:
    import ray
except Exception:
    ray = None

import vane


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
    import pickle

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
def test_csv_scan_executes_through_real_ray(tmp_path, monkeypatch):
    """Execute a multi-file CSV scan with bound parser options on Ray workers."""
    import pyarrow as pa

    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MIN_BYTES", "1")
    (tmp_path / "part-0.csv").write_text(
        "id|amount|label\n1|1,200.5|alpha\n2|2,300.5|beta\n",
        encoding="utf-8",
    )
    (tmp_path / "part-1.csv").write_text(
        "id;amount;note\n3;3,400.5;gamma-note\n4;4,500.5;delta-note\n",
        encoding="utf-8",
    )

    connection = vane.connect()
    relation = connection.sql(
        f"""
        SELECT id, amount, label, note
        FROM read_csv_auto(
            '{tmp_path}/*.csv',
            thousands=',',
            union_by_name=true
        )
        """
    )
    physical_plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        "csv-source-ray-plan",
    ).to_physical_plan(connection)
    assert [len(tasks) for tasks in physical_plan.scan_task_descriptor_map().values()] == [2]

    from vane import runners

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    parts = list(runner.run_iter_tables(relation))
    tables = [part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts]
    result = pa.concat_tables(tables)

    assert sorted(
        zip(
            result.column(0).to_pylist(),
            result.column(1).to_pylist(),
            result.column(2).to_pylist(),
            result.column(3).to_pylist(),
        )
    ) == [
        (1, 1200.5, "alpha", None),
        (2, 2300.5, "beta", None),
        (3, 3400.5, None, "gamma-note"),
        (4, 4500.5, None, "delta-note"),
    ]
    connection.close()


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_csv_scan_restores_discovered_multi_file_schema_on_ray(tmp_path, monkeypatch):
    """Restore the coordinator's merged CSV schema instead of independently inferring worker schemas."""
    import pyarrow as pa

    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MIN_BYTES", "1")
    (tmp_path / "part-0.csv").write_text("id,amount\n1,10\n2,20\n", encoding="utf-8")
    (tmp_path / "part-1.csv").write_text("id,amount\n3,30.5\n4,40.5\n", encoding="utf-8")

    connection = vane.connect()
    relation = connection.sql(f"SELECT * FROM read_csv_auto('{tmp_path}/*.csv')")
    physical_plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        "csv-schema-ray-plan",
    ).to_physical_plan(connection)
    assert [len(tasks) for tasks in physical_plan.scan_task_descriptor_map().values()] == [2]

    from vane import runners

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    parts = list(runner.run_iter_tables(relation))
    tables = [part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts]
    result = pa.concat_tables(tables)

    assert sorted(zip(result.column(0).to_pylist(), result.column(1).to_pylist())) == [
        (1, 10.0),
        (2, 20.0),
        (3, 30.5),
        (4, 40.5),
    ]
    connection.close()


def test_csv_union_reader_restores_per_file_options_for_fte_subset(tmp_path, monkeypatch):
    """Use the selected file's sniffed dialect when an FTE queue supplies only that split."""
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "2")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MIN_BYTES", "1")

    pipe_path = tmp_path / "part-0.csv"
    semicolon_path = tmp_path / "part-1.csv"
    pipe_path.write_text("id|label\n1|one\n2|two\n", encoding="utf-8")
    semicolon_path.write_text("id;note\n3;three\n4;four\n", encoding="utf-8")

    connection = vane.connect()
    relation = connection.sql(
        f"""
        SELECT id, label, note
        FROM read_csv_auto(
            ['{pipe_path}', '{semicolon_path}'],
            union_by_name=true
        )
        """
    )
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        "csv-union-fte-subset-plan",
    ).to_physical_plan(connection)
    scan_tasks = plan.scan_task_descriptor_map()
    assert len(scan_tasks) == 1
    node_id, descriptors = next(iter(scan_tasks.items()))
    assert len(descriptors) == 2

    worker_connection = vane.connect()
    worker_plan = plan.clone(worker_connection)
    split_queue = vane.ray_cxx.FteSplitQueue()
    split_queue.add_scan_split(bytes(descriptors[1]))
    split_queue.no_more_splits()
    result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        worker_connection.cursor(),
        worker_plan,
        fte_scan_source_queues={str(node_id): split_queue},
    )

    assert result.completion_status == "ok"
    assert len(result.partition_payloads) == 1
    table = result.partition_payloads[0]
    assert list(zip(table.column(0).to_pylist(), table.column(1).to_pylist(), table.column(2).to_pylist())) == [
        (3, None, "three"),
        (4, None, "four"),
    ]
    worker_connection.close()
    connection.close()


def test_static_csv_range_reopens_the_bind_time_union_reader(tmp_path, monkeypatch):
    """Do not reuse sniffing buffers that have already advanced past the start of a static range task."""
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "4")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MIN_PARTITION_NUM", "4")

    input_path = tmp_path / "static-range.csv"
    input_path.write_text(
        "id,label\n" + "".join(f"{row_id},value-{row_id}\n" for row_id in range(5000)),
        encoding="utf-8",
    )
    connection = vane.connect()
    relation = connection.sql(
        f"""
        SELECT id, label
        FROM read_csv_auto(
            '{input_path}',
            union_by_name=true,
            buffer_size=4096,
            max_line_size=1024
        )
        """
    )
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        "static-csv-range-plan",
    ).to_physical_plan(connection)
    node_id, descriptors = next(iter(plan.scan_task_descriptor_map().items()))
    assert len(descriptors) == 4

    result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        connection.cursor(),
        plan,
        scan_task={str(node_id): bytes(descriptors[0])},
    )

    assert result.completion_status == "ok"
    ids = result.partition_payloads[0].column(0).to_pylist()
    assert 0 < len(ids) < 5000
    assert sorted(ids) == list(range(len(ids)))
    connection.close()


def test_static_csv_range_detects_newline_from_file_start(tmp_path, monkeypatch):
    """Do not infer LF-only records from an embedded newline at a CRLF range boundary."""
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "2")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MIN_PARTITION_NUM", "2")

    input_path = tmp_path / "range-newline-detection.csv"
    file_bytes = bytearray(b"id,payload\r\n")
    expected = []
    row_id = 0
    range_boundary = 2048
    special_row_start = range_boundary - len(b'1,"line-a')
    while special_row_start - len(file_bytes) > 450:
        payload = "x" * 300
        file_bytes.extend(f"{row_id},{payload}\r\n".encode())
        expected.append((row_id, payload))
        row_id += 1
    remaining = special_row_start - len(file_bytes)
    row_prefix = f"{row_id},".encode()
    payload = "x" * (remaining - len(row_prefix) - len(b"\r\n"))
    file_bytes.extend(row_prefix + payload.encode() + b"\r\n")
    expected.append((row_id, payload))
    row_id += 1

    special_payload = "line-a\nline-b"
    file_bytes.extend(f'{row_id},"{special_payload}"\r\n'.encode())
    expected.append((row_id, special_payload))
    row_id += 1
    while len(file_bytes) < 3700:
        payload = "y" * 100
        file_bytes.extend(f"{row_id},{payload}\r\n".encode())
        expected.append((row_id, payload))
        row_id += 1
    input_path.write_bytes(file_bytes)
    assert file_bytes[range_boundary : range_boundary + 3] == b"\nli"
    assert 7 * 512 <= len(file_bytes) < 8 * 512

    connection = vane.connect()
    relation = connection.sql(
        f"""
        SELECT id, payload
        FROM read_csv(
            '{input_path}',
            delim=',',
            quote='"',
            escape='"',
            header=true,
            auto_detect=false,
            columns={{'id': 'INTEGER', 'payload': 'VARCHAR'}},
            buffer_size=1024,
            max_line_size=512
        )
        """
    )
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        "static-csv-range-newline-plan",
    ).to_physical_plan(connection)
    node_id, descriptors = next(iter(plan.scan_task_descriptor_map().items()))
    assert len(descriptors) == 2

    actual = []
    for descriptor in descriptors:
        worker_connection = vane.connect()
        worker_plan = plan.clone(worker_connection)
        result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(
            worker_connection.cursor(),
            worker_plan,
            scan_task={str(node_id): bytes(descriptor)},
        )
        assert result.completion_status == "ok"
        for table in result.partition_payloads:
            actual.extend(zip(table.column(0).to_pylist(), table.column(1).to_pylist()))
        worker_connection.close()

    assert sorted(actual) == expected
    connection.close()


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_single_csv_file_is_byte_range_scanned_through_real_ray(tmp_path, monkeypatch):
    """Split one seekable CSV across Ray tasks without losing or duplicating quoted rows."""
    import pyarrow as pa

    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "4")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MIN_PARTITION_NUM", "4")

    input_path = tmp_path / "single-file.csv"
    expected = []
    with input_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter="|", lineterminator="\r\n")
        writer.writerow(("id", "payload"))
        for row_id in range(5000):
            if row_id % 7 == 0:
                payload = f'quoted | value "{row_id}"\n继续-{row_id}'
            elif row_id % 11 == 0:
                payload = f"你好🙂-{row_id}"
            else:
                payload = f"value-{row_id}"
            writer.writerow((row_id, payload))
            expected.append((row_id, payload))

    connection = vane.connect()
    relation = connection.sql(
        f"""
        SELECT id, payload
        FROM read_csv_auto(
            '{input_path}',
            delim='|',
            union_by_name=true,
            buffer_size=65536,
            max_line_size=16384
        )
        """
    )
    physical_plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        "single-csv-byte-range-plan",
    ).to_physical_plan(connection)
    assert [len(tasks) for tasks in physical_plan.scan_task_descriptor_map().values()] == [4]

    from vane import runners

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    parts = list(runner.run_iter_tables(relation))
    tables = [part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts]
    result = pa.concat_tables(tables)

    assert sorted(zip(result.column(0).to_pylist(), result.column(1).to_pylist())) == expected
    connection.close()


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_single_csv_byte_ranges_own_rows_that_start_on_partition_boundaries(tmp_path, monkeypatch):
    """Assign a boundary-aligned row to exactly one adjacent Ray task."""
    import pyarrow as pa

    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "4")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MIN_PARTITION_NUM", "4")

    input_path = tmp_path / "boundary-aligned.csv"
    input_path.write_text(
        "id,value\n" + "".join(f"{row_id:03d},xxxxxx\n" for row_id in range(26)),
        encoding="utf-8",
    )
    assert input_path.stat().st_size == 295
    assert input_path.read_bytes()[64:68] == b"005,"

    connection = vane.connect()
    relation = connection.sql(
        f"""
        SELECT id, value
        FROM read_csv(
            '{input_path}',
            header=true,
            auto_detect=false,
            columns={{'id': 'INTEGER', 'value': 'VARCHAR'}},
            buffer_size=256,
            max_line_size=64
        )
        """
    )
    physical_plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        "boundary-aligned-csv-plan",
    ).to_physical_plan(connection)
    assert [len(tasks) for tasks in physical_plan.scan_task_descriptor_map().values()] == [4]

    from vane import runners

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    parts = list(runner.run_iter_tables(relation))
    tables = [part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts]
    result = pa.concat_tables(tables)

    assert sorted(zip(result.column(0).to_pylist(), result.column(1).to_pylist())) == [
        (row_id, "xxxxxx") for row_id in range(26)
    ]
    connection.close()


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_single_csv_byte_ranges_do_not_split_crlf_boundaries(tmp_path, monkeypatch):
    """Keep a range starting on LF synchronized with the preceding CR."""
    import pyarrow as pa

    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "4")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MIN_PARTITION_NUM", "4")

    input_path = tmp_path / "crlf-boundary.csv"
    input_path.write_bytes(b"id,value\r\n" + b"".join(f"{row_id:03d},xxxxx\r\n".encode() for row_id in range(26)))
    assert input_path.stat().st_size == 296
    assert input_path.read_bytes()[63:66] == b"\r\n0"

    connection = vane.connect()
    relation = connection.sql(
        f"""
        SELECT id, value
        FROM read_csv(
            '{input_path}',
            header=true,
            auto_detect=false,
            columns={{'id': 'INTEGER', 'value': 'VARCHAR'}},
            buffer_size=256,
            max_line_size=64
        )
        """
    )
    physical_plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        "crlf-boundary-csv-plan",
    ).to_physical_plan(connection)
    assert [len(tasks) for tasks in physical_plan.scan_task_descriptor_map().values()] == [4]

    from vane import runners

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    parts = list(runner.run_iter_tables(relation))
    tables = [part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts]
    result = pa.concat_tables(tables)

    assert sorted(zip(result.column(0).to_pylist(), result.column(1).to_pylist())) == [
        (row_id, "xxxxx") for row_id in range(26)
    ]
    connection.close()
