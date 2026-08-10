# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pickle
import re
import subprocess
import uuid
from urllib.parse import urlparse

import pytest

try:
    import ray
except Exception:
    ray = None

import vane


def _required_service_config() -> tuple[str, str, str, str, str]:
    endpoint = os.getenv("TEST_ICEBERG_REST_ENDPOINT") or ""
    minio_endpoint = os.getenv("TEST_MINIO_ENDPOINT") or ""
    access_key = os.getenv("TEST_MINIO_ACCESS_KEY") or ""
    secret_key = os.getenv("TEST_MINIO_SECRET_KEY") or ""
    region = os.getenv("TEST_MINIO_REGION") or "us-east-1"
    container_id = os.getenv("TEST_ICEBERG_REST_CONTAINER_ID") or ""
    if not endpoint or not minio_endpoint or not access_key or not secret_key or not container_id:
        message = "The hermetic REST Catalog endpoint, container ID, MinIO endpoint, and MinIO credentials are required"
        if os.getenv("VANE_REQUIRE_ICEBERG_REST_TEST") == "1":
            pytest.fail(message)
        pytest.skip(message)
    if not re.fullmatch(r"[0-9a-f]{64}", container_id):
        pytest.fail("TEST_ICEBERG_REST_CONTAINER_ID must be a full Docker container ID")
    return endpoint, minio_endpoint, access_key, secret_key, region


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _configure_catalog_connection(
    connection,
    catalog_endpoint: str,
    minio_endpoint: str,
    access_key: str,
    secret_key: str,
    region: str,
) -> None:
    parsed_minio = urlparse(minio_endpoint)
    duckdb_endpoint = parsed_minio.netloc or parsed_minio.path
    secret_name = f"vane_rest_storage_{uuid.uuid4().hex}"
    connection.execute("LOAD httpfs")
    connection.execute("LOAD iceberg")
    connection.execute(f"SET s3_endpoint={_sql_literal(duckdb_endpoint)}")
    connection.execute(f"SET s3_use_ssl={'true' if parsed_minio.scheme == 'https' else 'false'}")
    connection.execute("SET s3_url_style='path'")
    connection.execute(f"SET s3_region={_sql_literal(region)}")
    connection.execute(f"SET s3_access_key_id={_sql_literal(access_key)}")
    connection.execute(f"SET s3_secret_access_key={_sql_literal(secret_key)}")
    connection.execute("SET s3_session_token=''")
    connection.execute("SET http_proxy=''")
    connection.execute(
        f"""
        CREATE SECRET {secret_name} (
            TYPE S3,
            KEY_ID {_sql_literal(access_key)},
            SECRET {_sql_literal(secret_key)},
            REGION {_sql_literal(region)},
            ENDPOINT {_sql_literal(duckdb_endpoint)},
            URL_STYLE 'path',
            USE_SSL {"true" if parsed_minio.scheme == "https" else "false"}
        )
        """
    )
    connection.execute(
        "ATTACH '' AS rest_catalog ("
        "TYPE ICEBERG, "
        f"ENDPOINT {_sql_literal(catalog_endpoint)}, "
        "AUTHORIZATION_TYPE 'none', "
        "ACCESS_DELEGATION_MODE 'none'"
        ")"
    )


def _stop_hermetic_catalog(container_id: str) -> None:
    inspect = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            '{{ index .Config.Labels "ai.astrovela.vane.test" }}',
            container_id,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if inspect.returncode != 0 or inspect.stdout.strip() != "iceberg-rest-catalog":
        pytest.fail("Refusing to stop a container not owned by the hermetic Iceberg REST gate")

    stopped = subprocess.run(
        ["docker", "stop", "--time", "10", container_id],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if stopped.returncode != 0:
        pytest.fail(f"Could not stop the hermetic Iceberg REST Catalog: {stopped.stderr.strip()}")

    running = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", container_id],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if running.returncode != 0 or running.stdout.strip() != "false":
        pytest.fail("The hermetic Iceberg REST Catalog is still running")


def _rows_from_parts(parts) -> list[tuple]:
    rows: list[tuple] = []
    for part in parts:
        table = part.to_arrow() if hasattr(part, "to_arrow") else part
        columns = [column.to_pylist() for column in table.columns]
        rows.extend(zip(*columns, strict=True))
    return rows


def _execute_pickled_scan_descriptor(plan_payload: bytes, node_id: str, descriptor: bytes) -> list[tuple]:
    import pickle

    import vane

    ray_cxx = vane.ray_cxx
    plan = pickle.loads(plan_payload)
    worker_connection = vane.connect()
    result = None
    try:
        split_queue = ray_cxx.FteSplitQueue()
        split_queue.add_scan_split(descriptor)
        split_queue.no_more_splits()
        result = ray_cxx.DistributedPhysicalPlanRunner().execute_native(
            worker_connection.cursor(),
            plan,
            fte_scan_source_queues={node_id: split_queue},
        )
        return _rows_from_parts(result.partition_payloads)
    finally:
        # Iceberg bind state owns manifest tasks tied to the client scheduler.
        result = None
        plan = None
        worker_connection.close()


@pytest.mark.external_service
@pytest.mark.iceberg_rest
@pytest.mark.real_ray
@pytest.mark.usefixtures("ray_local")
def test_catalog_bound_snapshot_runs_on_ray_after_rest_catalog_stops(monkeypatch):
    if ray is None:
        if os.getenv("VANE_REQUIRE_ICEBERG_REST_TEST") == "1":
            pytest.fail("ray is required by the hermetic Iceberg REST gate")
        pytest.skip("ray not installed")
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        if os.getenv("VANE_REQUIRE_ICEBERG_REST_TEST") == "1":
            pytest.fail("pyarrow is required by the hermetic Iceberg REST gate")
        pytest.skip("pyarrow not installed")

    catalog_endpoint, minio_endpoint, access_key, secret_key, region = _required_service_config()
    container_id = os.environ["TEST_ICEBERG_REST_CONTAINER_ID"]
    schema_name = f"vane_{uuid.uuid4().hex}"
    table_name = f"rest_catalog.{schema_name}.events"

    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "1")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "2")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MIN_PARTITION_NUM", "2")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_SIZE_GROUPING", "0")

    source_connection = vane.connect()
    mutator_connection = None
    relation = None
    try:
        _configure_catalog_connection(
            source_connection,
            catalog_endpoint,
            minio_endpoint,
            access_key,
            secret_key,
            region,
        )
        source_connection.execute(f"CREATE SCHEMA rest_catalog.{schema_name}")
        source_connection.execute(f"CREATE TABLE {table_name} (event_id INTEGER, category VARCHAR)")
        source_connection.execute(f"INSERT INTO {table_name} VALUES (1, 'a'), (2, 'b')")
        source_connection.execute(f"INSERT INTO {table_name} VALUES (3, 'a'), (4, 'c')")

        # Binding the relation selects the catalog's current Iceberg snapshot.
        relation = source_connection.sql(f"SELECT event_id, category FROM {table_name}")
        ray_cxx = getattr(vane, "ray_cxx", None)
        if ray_cxx is None or not hasattr(ray_cxx, "PyLogicalPlan"):
            pytest.fail("vane.ray_cxx.PyLogicalPlan is required by the hermetic Iceberg REST gate")
        logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            f"iceberg-rest-{uuid.uuid4().hex}",
        )
        physical_plan = logical_plan.to_physical_plan(source_connection)
        descriptor_map = physical_plan.scan_task_descriptor_map()
        assert len(descriptor_map) == 1
        node_id, descriptors = next(iter(descriptor_map.items()))
        assert len(descriptors) == 2
        plan_payload = pickle.dumps(physical_plan)

        # Commit a newer snapshot through an independent coordinator connection.
        # Distributed Iceberg writes are intentionally outside this contract.
        mutator_connection = vane.connect()
        _configure_catalog_connection(
            mutator_connection,
            catalog_endpoint,
            minio_endpoint,
            access_key,
            secret_key,
            region,
        )
        mutator_connection.execute(f"INSERT INTO {table_name} VALUES (5, 'newer-snapshot')")
        assert mutator_connection.execute(f"SELECT count(*) FROM {table_name}").fetchone() == (5,)
        mutator_connection.close()
        mutator_connection = None

        # A worker-side Catalog lookup or a rebind to "latest" must now fail.
        # Successful execution therefore proves that the coordinator materialized
        # the exact scan path and immutable snapshot into the distributed plan.
        _stop_hermetic_catalog(container_id)
        execute_descriptor = ray.remote(_execute_pickled_scan_descriptor)
        result_refs = [
            execute_descriptor.remote(plan_payload, str(node_id), bytes(descriptor)) for descriptor in descriptors
        ]
        rows = [row for descriptor_rows in ray.get(result_refs) for row in descriptor_rows]
    finally:
        relation = None
        if mutator_connection is not None:
            mutator_connection.close()
        source_connection.close()
        if hasattr(vane, "teardown_runner"):
            vane.teardown_runner()

    assert sorted(rows) == [(1, "a"), (2, "b"), (3, "a"), (4, "c")]
