# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import copy
import io
import json
import os
import tarfile
import uuid
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pytest

try:
    import ray
except Exception:
    ray = None

import vane

_FIXTURE_ARCHIVE = Path(__file__).parent / "data" / "iceberg_partition_integer.tar.gz.b64"
_EQUALITY_DELETE_FIXTURE_ARCHIVE = Path(__file__).parent / "data" / "iceberg_equality_deletes.tar.gz.b64"
_POSITIONAL_DELETE_FIXTURE_ARCHIVE = Path(__file__).parent / "data" / "iceberg_positional_deletes.tar.gz.b64"
_SCHEMA_EVOLUTION_FIXTURE_ARCHIVE = Path(__file__).parent / "data" / "iceberg_schema_evolution.tar.gz.b64"
_NESTED_DEFAULTS_FIXTURE_ARCHIVE = Path(__file__).parent / "data" / "iceberg_nested_defaults.tar.gz.b64"
_PARTITION_SPEC_EVOLUTION_FIXTURE_ARCHIVE = (
    Path(__file__).parent / "data" / "iceberg_partition_spec_evolution.tar.gz.b64"
)
_STRUCT_FILTER_FIXTURE_ARCHIVE = Path(__file__).parent / "data" / "iceberg_struct_filter.tar.gz.b64"
_UUID_FIXTURE_ARCHIVE = Path(__file__).parent / "data" / "iceberg_uuid.tar.gz.b64"
_UNKNOWN_PUFFIN_FIXTURE_ARCHIVE = Path(__file__).parent / "data" / "iceberg_unknown_puffin.tar.gz.b64"
_OLD_SNAPSHOT_ID = 5470601323427916272
_OLD_MANIFEST_LIST = "snap-5470601323427916272-1-b1dda674-423f-4f23-b00d-92b608b07a38.avro"
_SCHEMA_EVOLUTION_METADATA = "00003-3f1801a5-7dfb-4072-b14a-39cd12f9279b.metadata.json"
_NESTED_DEFAULTS_METADATA = "00003-21a957f9-c2ee-431a-9d18-bf257b561198.metadata.json"
_CORE_DISTRIBUTED_STORAGE_CASES = (
    pytest.param(
        "iceberg_table_path",
        "partition_integer",
        "partition_col, user_id, event_type",
        "",
        " WHERE partition_col = 42",
        ((42, 12345, "click"),),
        id="partition-filter",
    ),
    pytest.param(
        "iceberg_equality_delete_path",
        "equality_deletes",
        "id, name",
        "",
        "",
        ((1, "b"), (2, "b")),
        id="equality-deletes",
    ),
    pytest.param(
        "iceberg_positional_delete_path",
        "positional_deletes",
        "count(*)::BIGINT, CAST(sum(hash(id, modifiedby, lastmodifieddate, load_time)) AS VARCHAR)",
        "",
        "",
        ((13136, "121339110023863585282674"),),
        id="multi-manifest-positional-deletes",
    ),
    pytest.param(
        "iceberg_schema_evolution_path",
        "schema_evolution",
        "col1, col_boolean, col_integer, col_long, col_string",
        f", version='{_SCHEMA_EVOLUTION_METADATA}', version_name_format='%s%s'",
        "",
        (
            ("click", True, 342342, -9223372036854775808, "HELLO"),
            ("purchase", True, 342342, -9223372036854775808, "HELLO"),
            ("test", False, 453243, 328725092345834, "World"),
        ),
        id="schema-evolution-initial-defaults",
    ),
)
_EXTENDED_DISTRIBUTED_STORAGE_CASES = (
    pytest.param(
        "iceberg_nested_defaults_path",
        "nested_defaults",
        "a.col1, a.col_boolean, a.col_integer, a.col_long, a.col_string, CAST(a.col_uuid AS VARCHAR)",
        f", version='{_NESTED_DEFAULTS_METADATA}', version_name_format='%s%s'",
        "",
        (
            ("test", False, 453243, 328725092345834, "World", None),
            (
                "test",
                True,
                342342,
                -9223372036854775808,
                "HELLO",
                "f79c3e09-677c-4bbd-a479-3f349cb785e7",
            ),
        ),
        id="nested-initial-defaults",
    ),
    pytest.param(
        "iceberg_partition_spec_evolution_path",
        "partition_spec_evolution",
        "CAST(event_date AS VARCHAR), user_id, event_type",
        "",
        " WHERE event_date IN (DATE '2024-01-02', DATE '2024-01-03')",
        (
            ("2024-01-02", 67890, "purchase"),
            ("2024-01-03", 13579, "view"),
            ("2024-01-03", 24680, "click"),
        ),
        id="partition-spec-evolution",
    ),
    pytest.param(
        "iceberg_struct_filter_path",
        "struct_filter",
        "redpanda.partition, CAST(redpanda.timestamp AS VARCHAR), CAST(value AS VARCHAR)",
        "",
        " WHERE redpanda.partition = 0 AND redpanda.timestamp = TIMESTAMP '2025-06-26 19:45:32.478'",
        ((0, "2025-06-26 19:45:32.478", "hello world"),),
        id="nested-struct-filter",
    ),
    pytest.param(
        "iceberg_uuid_path",
        "uuid",
        "CAST(uuid AS VARCHAR)",
        "",
        " WHERE uuid = UUID '1571effb-facd-42a3-90e9-0af522e9b6c2'",
        (("1571effb-facd-42a3-90e9-0af522e9b6c2",),),
        id="uuid-filter",
    ),
    pytest.param(
        "iceberg_unknown_puffin_path",
        "unknown_puffin",
        "id, name, age",
        "",
        "",
        ((1, "John", 10), (4, "David", 40)),
        id="unknown-puffin-blob",
    ),
)
_MINIO_DISTRIBUTED_CASES = _CORE_DISTRIBUTED_STORAGE_CASES + _EXTENDED_DISTRIBUTED_STORAGE_CASES


def _extract_fixture(archive_path: Path, table_path: Path) -> Path:
    payload = base64.b64decode(archive_path.read_bytes())
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            assert member.isfile()
            relative_path = Path(member.name)
            assert not relative_path.is_absolute()
            assert ".." not in relative_path.parts
            source = archive.extractfile(member)
            assert source is not None
            destination = table_path / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read())
    return table_path


@pytest.fixture
def iceberg_table_path(tmp_path: Path) -> Path:
    """Extract the small MIT-licensed duckdb-iceberg partition fixture."""
    return _extract_fixture(_FIXTURE_ARCHIVE, tmp_path / "partition_integer")


@pytest.fixture
def iceberg_equality_delete_path(tmp_path: Path) -> Path:
    """Extract the upstream Iceberg equality-delete fixture."""
    return _extract_fixture(
        _EQUALITY_DELETE_FIXTURE_ARCHIVE,
        tmp_path / "equality_deletes",
    )


@pytest.fixture
def iceberg_positional_delete_path(tmp_path: Path) -> Path:
    """Extract the upstream multi-manifest positional-delete fixture."""
    return _extract_fixture(
        _POSITIONAL_DELETE_FIXTURE_ARCHIVE,
        tmp_path / "positional_deletes",
    )


@pytest.fixture
def iceberg_schema_evolution_path(tmp_path: Path) -> Path:
    """Extract the upstream schema-evolution and initial-default fixture."""
    return _extract_fixture(
        _SCHEMA_EVOLUTION_FIXTURE_ARCHIVE,
        tmp_path / "schema_evolution",
    )


@pytest.fixture
def iceberg_nested_defaults_path(tmp_path: Path) -> Path:
    """Extract nested initial defaults covering all upstream Iceberg types."""
    return _extract_fixture(
        _NESTED_DEFAULTS_FIXTURE_ARCHIVE,
        tmp_path / "nested_defaults",
    )


@pytest.fixture
def iceberg_partition_spec_evolution_path(tmp_path: Path) -> Path:
    """Extract a table whose identity partition spec evolved across snapshots."""
    return _extract_fixture(
        _PARTITION_SPEC_EVOLUTION_FIXTURE_ARCHIVE,
        tmp_path / "partition_spec_evolution",
    )


@pytest.fixture
def iceberg_struct_filter_path(tmp_path: Path) -> Path:
    """Extract a table partitioned on a field inside a nested struct."""
    return _extract_fixture(
        _STRUCT_FILTER_FIXTURE_ARCHIVE,
        tmp_path / "struct_filter",
    )


@pytest.fixture
def iceberg_uuid_path(tmp_path: Path) -> Path:
    """Extract the upstream UUID type and filter fixture."""
    return _extract_fixture(_UUID_FIXTURE_ARCHIVE, tmp_path / "uuid")


@pytest.fixture
def iceberg_unknown_puffin_path(tmp_path: Path) -> Path:
    """Extract a table with an unsupported Puffin statistics blob."""
    return _extract_fixture(
        _UNKNOWN_PUFFIN_FIXTURE_ARCHIVE,
        tmp_path / "unknown_puffin",
    )


def _query_rows(result) -> list[tuple]:
    rows: list[tuple] = []
    for payload in result.partition_payloads:
        table = payload.to_arrow() if hasattr(payload, "to_arrow") else payload
        columns = [column.to_pylist() for column in table.columns]
        rows.extend(zip(*columns, strict=True))
    return rows


def _require_ray_cxx():
    pytest.importorskip("pyarrow")
    ray_cxx = getattr(vane, "ray_cxx", None)
    if ray_cxx is None or not hasattr(ray_cxx, "PyLogicalPlan"):
        pytest.skip("vane.ray_cxx.PyLogicalPlan not available in this environment")
    return ray_cxx


def _configure_scan_partitions(monkeypatch, partition_count: int) -> None:
    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "1")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", str(partition_count))
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MIN_PARTITION_NUM", str(partition_count))
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_SIZE_GROUPING", "0")


def _single_scan_tasks(plan):
    descriptor_map = plan.scan_task_descriptor_map()
    assert len(descriptor_map) == 1
    return next(iter(descriptor_map.items()))


def _execute_scan_descriptor(plan, node_id, descriptor) -> list[tuple]:
    ray_cxx = vane.ray_cxx
    worker_connection = vane.connect()
    worker_plan = None
    result = None
    try:
        worker_plan = plan.clone(worker_connection)
        split_queue = ray_cxx.FteSplitQueue()
        split_queue.add_scan_split(bytes(descriptor))
        split_queue.no_more_splits()
        result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(
            worker_connection.cursor(),
            worker_plan,
            fte_scan_source_queues={str(node_id): split_queue},
        )
        return _query_rows(result)
    finally:
        # Iceberg bind state owns manifest tasks tied to the client scheduler,
        # so release the result and plan before closing their context.
        result = None
        worker_plan = None
        worker_connection.close()


def _execute_scan_descriptors(plan, node_id, descriptors) -> list[tuple]:
    rows: list[tuple] = []
    for descriptor in descriptors:
        rows.extend(_execute_scan_descriptor(plan, node_id, descriptor))
    return rows


def _move_fixture_to_new_broken_snapshot(table_path: Path) -> None:
    """Make latest unusable while retaining the snapshot selected by planning."""
    metadata_path = table_path / "metadata"
    metadata = json.loads((metadata_path / "v2.metadata.json").read_text())
    old_snapshot = metadata["snapshots"][0]
    new_snapshot = copy.deepcopy(old_snapshot)
    new_snapshot_id = _OLD_SNAPSHOT_ID + 1
    new_snapshot["sequence-number"] = 2
    new_snapshot["snapshot-id"] = new_snapshot_id
    new_snapshot["timestamp-ms"] += 1
    new_snapshot["manifest-list"] = "data/persistent/partition_integer/metadata/missing-current-snapshot.avro"

    metadata["last-sequence-number"] = 2
    metadata["last-updated-ms"] = new_snapshot["timestamp-ms"]
    metadata["current-snapshot-id"] = new_snapshot_id
    metadata["refs"]["main"]["snapshot-id"] = new_snapshot_id
    metadata["snapshots"].append(new_snapshot)
    metadata["snapshot-log"].append({"timestamp-ms": new_snapshot["timestamp-ms"], "snapshot-id": new_snapshot_id})
    (metadata_path / "v3.metadata.json").write_text(json.dumps(metadata))
    (metadata_path / "version-hint.text").write_text("3")


def _expire_planned_snapshot(table_path: Path) -> None:
    _move_fixture_to_new_broken_snapshot(table_path)
    metadata_file = table_path / "metadata" / "v3.metadata.json"
    metadata = json.loads(metadata_file.read_text())
    metadata["snapshots"] = [
        snapshot for snapshot in metadata["snapshots"] if snapshot["snapshot-id"] != _OLD_SNAPSHOT_ID
    ]
    metadata["snapshot-log"] = [entry for entry in metadata["snapshot-log"] if entry["snapshot-id"] != _OLD_SNAPSHOT_ID]
    metadata_file.write_text(json.dumps(metadata))


def test_iceberg_scan_tasks_rebind_subset_and_pin_snapshot(iceberg_table_path, monkeypatch):
    ray_cxx = _require_ray_cxx()
    _configure_scan_partitions(monkeypatch, 2)

    source_connection = vane.connect()
    try:
        source_connection.execute("LOAD iceberg")
        relation = source_connection.sql(
            "SELECT partition_col, user_id, event_type "
            f"FROM iceberg_scan('{iceberg_table_path}', allow_moved_paths=true)"
        )
        plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            f"iceberg-subset-{uuid.uuid4().hex}",
        ).to_physical_plan(source_connection)
        node_id, descriptors = _single_scan_tasks(plan)
        assert len(descriptors) == 2

        # A worker that rebound "latest" would now select the deliberately
        # broken snapshot. The planned immutable snapshot must still succeed.
        _move_fixture_to_new_broken_snapshot(iceberg_table_path)

        rows = _execute_scan_descriptors(plan, node_id, descriptors)
    finally:
        source_connection.close()

    assert sorted(rows) == [(42, 12345, "click"), (1337, 67890, "purchase")]


def test_iceberg_scan_task_byte_estimates_drive_size_grouping(iceberg_table_path, monkeypatch):
    ray_cxx = _require_ray_cxx()
    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "1")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "2")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MIN_PARTITION_NUM", "1")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_SIZE_GROUPING", "1")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MIN_BYTES", "1")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MAX_BYTES", "1GB")

    source_connection = vane.connect()
    try:
        source_connection.execute("LOAD iceberg")
        relation = source_connection.sql(
            "SELECT partition_col, user_id, event_type "
            f"FROM iceberg_scan('{iceberg_table_path}', allow_moved_paths=true)"
        )
        plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            f"iceberg-byte-grouping-{uuid.uuid4().hex}",
        ).to_physical_plan(source_connection)
        node_id, descriptors = _single_scan_tasks(plan)
        assert len(descriptors) == 1
        rows = _execute_scan_descriptors(plan, node_id, descriptors)
    finally:
        source_connection.close()

    assert sorted(rows) == [(42, 12345, "click"), (1337, 67890, "purchase")]


def test_iceberg_scan_tasks_preserve_equality_deletes(
    iceberg_equality_delete_path,
    monkeypatch,
):
    ray_cxx = _require_ray_cxx()
    _configure_scan_partitions(monkeypatch, 2)

    source_connection = vane.connect()
    try:
        source_connection.execute("LOAD iceberg")
        relation = source_connection.sql(
            f"SELECT id, name FROM iceberg_scan('{iceberg_equality_delete_path}', allow_moved_paths=true)"
        )
        plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            f"iceberg-equality-deletes-{uuid.uuid4().hex}",
        ).to_physical_plan(source_connection)
        node_id, descriptors = _single_scan_tasks(plan)
        assert len(descriptors) == 2
        rows = _execute_scan_descriptors(plan, node_id, descriptors)
    finally:
        source_connection.close()

    assert sorted(rows) == [(1, "b"), (2, "b")]


def test_iceberg_multi_manifest_positional_deletes_are_retry_safe(
    iceberg_positional_delete_path,
    monkeypatch,
):
    ray_cxx = _require_ray_cxx()
    _configure_scan_partitions(monkeypatch, 4)

    source_connection = vane.connect()
    try:
        source_connection.execute("LOAD iceberg")
        relation = source_connection.sql(
            "SELECT id, hash(id, modifiedby, lastmodifieddate, load_time) AS row_hash "
            f"FROM iceberg_scan('{iceberg_positional_delete_path}', allow_moved_paths=true)"
        )
        plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            f"iceberg-positional-deletes-{uuid.uuid4().hex}",
        ).to_physical_plan(source_connection)
        node_id, descriptors = _single_scan_tasks(plan)
        assert len(descriptors) == 4

        selected_attempt_rows: list[tuple] = []
        selected_attempt_counts = []
        for descriptor in descriptors:
            # Fault-tolerant execution may replay one descriptor. Both attempts
            # must be identical, while only the selected attempt contributes to
            # the final query result.
            first_attempt = _execute_scan_descriptor(plan, node_id, descriptor)
            retry_attempt = _execute_scan_descriptor(plan, node_id, descriptor)
            assert Counter(first_attempt) == Counter(retry_attempt)
            selected_attempt_rows.extend(first_attempt)
            selected_attempt_counts.append(len(first_attempt))
    finally:
        source_connection.close()

    assert sorted(selected_attempt_counts) == [817, 829, 854, 10636]
    assert len(selected_attempt_rows) == 13136
    assert len({row[0] for row in selected_attempt_rows}) == 10636
    assert sum(row[1] for row in selected_attempt_rows) == 121339110023863585282674


def test_iceberg_schema_evolution_rebinds_initial_defaults(
    iceberg_schema_evolution_path,
    monkeypatch,
):
    ray_cxx = _require_ray_cxx()
    _configure_scan_partitions(monkeypatch, 2)

    source_connection = vane.connect()
    try:
        source_connection.execute("LOAD iceberg")
        relation = source_connection.sql(
            "SELECT col1, col_boolean, col_integer, col_long, col_string "
            f"FROM iceberg_scan('{iceberg_schema_evolution_path}', "
            "allow_moved_paths=true, "
            f"version='{_SCHEMA_EVOLUTION_METADATA}', "
            "version_name_format='%s%s')"
        )
        plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            f"iceberg-schema-evolution-{uuid.uuid4().hex}",
        ).to_physical_plan(source_connection)
        node_id, descriptors = _single_scan_tasks(plan)
        assert len(descriptors) == 2
        rows = _execute_scan_descriptors(plan, node_id, descriptors)
    finally:
        source_connection.close()

    assert sorted(rows) == [
        ("click", True, 342342, -9223372036854775808, "HELLO"),
        ("purchase", True, 342342, -9223372036854775808, "HELLO"),
        ("test", False, 453243, 328725092345834, "World"),
    ]


@pytest.mark.parametrize(
    ("fixture_name", "table_name", "projection", "scan_options", "query_suffix", "expected_rows"),
    _EXTENDED_DISTRIBUTED_STORAGE_CASES,
)
def test_iceberg_extended_distributed_read_matrix(
    request,
    monkeypatch,
    fixture_name,
    table_name,
    projection,
    scan_options,
    query_suffix,
    expected_rows,
):
    ray_cxx = _require_ray_cxx()
    _configure_scan_partitions(monkeypatch, 4)
    table_path = request.getfixturevalue(fixture_name)

    source_connection = vane.connect()
    try:
        source_connection.execute("LOAD iceberg")
        relation = source_connection.sql(
            _iceberg_scan_sql(
                str(table_path),
                projection,
                scan_options,
                query_suffix,
            )
        )
        plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            f"iceberg-{table_name}-{uuid.uuid4().hex}",
        ).to_physical_plan(source_connection)
        node_id, descriptors = _single_scan_tasks(plan)
        assert descriptors
        rows = _execute_scan_descriptors(plan, node_id, descriptors)
    finally:
        source_connection.close()

    assert sorted(rows) == sorted(expected_rows)


def test_iceberg_planned_metadata_survives_snapshot_expiration_in_latest_version(
    iceberg_table_path,
    monkeypatch,
):
    ray_cxx = _require_ray_cxx()
    _configure_scan_partitions(monkeypatch, 2)

    source_connection = vane.connect()
    try:
        source_connection.execute("LOAD iceberg")
        relation = source_connection.sql(f"SELECT * FROM iceberg_scan('{iceberg_table_path}', allow_moved_paths=true)")
        plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            f"iceberg-expired-snapshot-{uuid.uuid4().hex}",
        ).to_physical_plan(source_connection)
        node_id, descriptors = _single_scan_tasks(plan)
        _expire_planned_snapshot(iceberg_table_path)
        rows = _execute_scan_descriptors(plan, node_id, descriptors)
    finally:
        source_connection.close()

    assert sorted(rows) == [(42, 12345, "click"), (1337, 67890, "purchase")]


def test_iceberg_worker_bind_requires_planned_metadata(iceberg_table_path):
    connection = vane.connect()
    try:
        connection.execute("LOAD iceberg")
        with pytest.raises(Exception, match="(?i)worker bind requires.*metadata JSON"):
            connection.sql(
                "SELECT * FROM iceberg_scan("
                f"'{iceberg_table_path}', "
                "allow_moved_paths=true, _vane_distributed_worker=true)"
            ).fetchall()
    finally:
        connection.close()


def test_iceberg_missing_planned_manifest_fails_without_latest_fallback(
    iceberg_table_path,
    monkeypatch,
):
    ray_cxx = _require_ray_cxx()
    _configure_scan_partitions(monkeypatch, 2)

    source_connection = vane.connect()
    try:
        source_connection.execute("LOAD iceberg")
        relation = source_connection.sql(f"SELECT * FROM iceberg_scan('{iceberg_table_path}', allow_moved_paths=true)")
        plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            f"iceberg-missing-manifest-{uuid.uuid4().hex}",
        ).to_physical_plan(source_connection)
        node_id, descriptors = _single_scan_tasks(plan)
        (iceberg_table_path / "metadata" / _OLD_MANIFEST_LIST).unlink()

        with pytest.raises(Exception, match="(?i)(manifest|cannot open|no files found)"):
            _execute_scan_descriptor(plan, node_id, descriptors[0])
    finally:
        source_connection.close()


def _external_minio_config() -> tuple[str, str, str, str, str, str]:
    endpoint = os.getenv("TEST_MINIO_ENDPOINT") or os.getenv("AWS_ENDPOINT_URL") or ""
    access_key = os.getenv("TEST_MINIO_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID") or ""
    secret_key = os.getenv("TEST_MINIO_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY") or ""
    session_token = os.getenv("TEST_MINIO_SESSION_TOKEN") or os.getenv("AWS_SESSION_TOKEN") or ""
    region = os.getenv("TEST_MINIO_REGION") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    bucket = os.getenv("TEST_MINIO_BUCKET") or "vane-shuffle-test"
    if not endpoint or not access_key or not secret_key:
        message = "TEST_MINIO_ENDPOINT and MinIO/S3 credentials are required"
        if os.getenv("VANE_REQUIRE_ICEBERG_MINIO_TEST") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return endpoint, access_key, secret_key, session_token, region, bucket


@pytest.fixture
def external_minio_ray_config(request):
    # Validate optional-service configuration before paying the cost of
    # starting the shared Ray cluster for this test.
    config = _external_minio_config()
    request.getfixturevalue("ray_local")
    return config


def _create_s3_client(endpoint, access_key, secret_key, session_token, region):
    from botocore.config import Config
    from botocore.session import get_session

    return get_session().create_client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token or None,
        config=Config(
            signature_version="s3v4",
            connect_timeout=2,
            read_timeout=5,
            retries={"max_attempts": 1},
            s3={"addressing_style": "path"},
        ),
    )


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _iceberg_scan_sql(
    table_uri: str,
    projection: str,
    scan_options: str,
    query_suffix: str,
) -> str:
    return (
        f"SELECT {projection} FROM iceberg_scan("
        f"{_sql_literal(table_uri)}, allow_moved_paths=true{scan_options})"
        f"{query_suffix}"
    )


def _configure_s3_connection(connection, endpoint, access_key, secret_key, session_token, region) -> None:
    parsed = urlparse(endpoint)
    duckdb_endpoint = parsed.netloc or parsed.path
    connection.execute("LOAD httpfs")
    connection.execute(f"SET s3_endpoint={_sql_literal(duckdb_endpoint)}")
    connection.execute(f"SET s3_use_ssl={'true' if parsed.scheme == 'https' else 'false'}")
    connection.execute("SET s3_url_style='path'")
    connection.execute(f"SET s3_region={_sql_literal(region)}")
    connection.execute(f"SET s3_access_key_id={_sql_literal(access_key)}")
    connection.execute(f"SET s3_secret_access_key={_sql_literal(secret_key)}")
    connection.execute(f"SET s3_session_token={_sql_literal(session_token)}")
    connection.execute("SET http_proxy=''")


@pytest.mark.external_service
@pytest.mark.real_ray
@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.parametrize(
    ("fixture_name", "table_name", "projection", "scan_options", "query_suffix", "expected_rows"),
    _MINIO_DISTRIBUTED_CASES,
)
def test_iceberg_scan_runs_from_minio_on_ray_workers(
    request,
    external_minio_ray_config,
    monkeypatch,
    fixture_name,
    table_name,
    projection,
    scan_options,
    query_suffix,
    expected_rows,
):
    from vane import runners

    table_path = request.getfixturevalue(fixture_name)
    endpoint, access_key, secret_key, session_token, region, bucket = external_minio_ray_config
    object_prefix = f"vane-iceberg-e2e/{uuid.uuid4().hex}/{table_name}"
    s3_client = _create_s3_client(endpoint, access_key, secret_key, session_token, region)
    uploaded_keys = []
    source_connection = None
    try:
        try:
            for source_path in sorted(path for path in table_path.rglob("*") if path.is_file()):
                relative_path = source_path.relative_to(table_path).as_posix()
                object_key = f"{object_prefix}/{relative_path}"
                s3_client.put_object(Bucket=bucket, Key=object_key, Body=source_path.read_bytes())
                uploaded_keys.append(object_key)
        except Exception as exc:
            if os.getenv("VANE_REQUIRE_ICEBERG_MINIO_TEST") == "1":
                raise AssertionError("MinIO/S3-compatible endpoint is not writable for this test") from exc
            pytest.skip(f"MinIO/S3-compatible endpoint is not writable for this test: {exc}")

        monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", access_key)
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret_key)
        monkeypatch.setenv("AWS_SESSION_TOKEN", session_token)
        monkeypatch.setenv("AWS_REGION", region)

        runners.set_runner_ray(noop_if_initialized=True)
        runner = runners.get_or_create_runner()
        try:
            source_connection = vane.connect()
            source_connection.execute("LOAD iceberg")
            _configure_s3_connection(
                source_connection,
                endpoint,
                access_key,
                secret_key,
                session_token,
                region,
            )
            table_uri = f"s3://{bucket}/{object_prefix}"
            relation = source_connection.sql(_iceberg_scan_sql(table_uri, projection, scan_options, query_suffix))
            parts = list(runner.run_iter_tables(relation))
            rows = []
            for part in parts:
                table = part.to_arrow() if hasattr(part, "to_arrow") else part
                columns = [column.to_pylist() for column in table.columns]
                rows.extend(zip(*columns, strict=True))
        finally:
            try:
                if source_connection is not None:
                    source_connection.close()
            finally:
                if hasattr(vane, "teardown_runner"):
                    vane.teardown_runner()

        assert sorted(rows) == sorted(expected_rows)
    finally:
        if uploaded_keys:
            for object_key in uploaded_keys:
                s3_client.delete_object(Bucket=bucket, Key=object_key)


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_iceberg_scan_runs_on_ray_workers(iceberg_table_path):
    from vane import runners

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    source_connection = vane.connect()
    try:
        source_connection.execute("LOAD iceberg")
        relation = source_connection.sql(
            "SELECT partition_col, user_id, upper(event_type) "
            f"FROM iceberg_scan('{iceberg_table_path}', allow_moved_paths=true) "
            "WHERE partition_col = 42"
        )
        parts = list(runner.run_iter_tables(relation))
        rows = []
        for part in parts:
            table = part.to_arrow() if hasattr(part, "to_arrow") else part
            columns = [column.to_pylist() for column in table.columns]
            rows.extend(zip(*columns, strict=True))
    finally:
        source_connection.close()
        if hasattr(vane, "teardown_runner"):
            vane.teardown_runner()

    assert rows == [(42, 12345, "CLICK")]
