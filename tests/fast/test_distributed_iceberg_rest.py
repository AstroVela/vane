# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import pickle
import re
import select
import subprocess
import sys
import uuid
from pathlib import Path
from urllib import error, request
from urllib.parse import quote, urlparse

import pytest

try:
    import ray
except Exception:
    ray = None

import vane


class _RestCommitFailureProxy:
    def __init__(self, upstream_endpoint: str) -> None:
        helper_path = Path(__file__).resolve().parents[1] / "helpers" / "iceberg_rest_commit_failure_proxy.py"
        self._process = subprocess.Popen(
            [sys.executable, "-u", str(helper_path), "--upstream", upstream_endpoint],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert self._process.stdout is not None
        ready, _, _ = select.select([self._process.stdout], [], [], 10)
        if not ready:
            self._stop_and_close_process()
            raise RuntimeError("Iceberg REST fault proxy did not publish its endpoint")
        self._endpoint = self._process.stdout.readline().strip()
        parsed = urlparse(self._endpoint)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port:
            self._stop_and_close_process()
            raise RuntimeError(f"Iceberg REST fault proxy published an invalid endpoint: {self._endpoint!r}")

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def commit_attempts(self) -> int:
        return int(self._status()["commit_attempts"])

    @property
    def cached_table_get_count(self) -> int:
        return int(self._status()["cached_table_get_count"])

    def reject_commits(self) -> None:
        self._control("/__vane_fault/reject", method="POST")

    def lose_commit_response(self) -> None:
        self._control("/__vane_fault/lose-commit-response", method="POST")

    def close(self) -> None:
        try:
            try:
                if self._process.poll() is None:
                    self._control("/__vane_fault/shutdown", method="POST")
                self._process.wait(timeout=10)
            except Exception:
                self._stop_process()
                raise
            if self._process.returncode != 0:
                assert self._process.stderr is not None
                raise RuntimeError(f"Iceberg REST fault proxy failed: {self._process.stderr.read().strip()}")
        finally:
            self._close_pipes()

    def _status(self) -> dict[str, int]:
        return self._control("/__vane_fault/status", method="GET")

    def _control(self, path: str, *, method: str) -> dict[str, int]:
        control_request = request.Request(self._endpoint + path, method=method)
        with request.urlopen(control_request, timeout=5) as response:
            payload = response.read()
        return json.loads(payload) if payload else {}

    def _stop_process(self) -> None:
        if self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)

    def _stop_and_close_process(self) -> None:
        try:
            self._stop_process()
        finally:
            self._close_pipes()

    def _close_pipes(self) -> None:
        for pipe in (self._process.stdout, self._process.stderr):
            if pipe is not None:
                pipe.close()


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


def _required_marker_fault_credentials() -> tuple[str, str]:
    access_key = os.getenv("TEST_MINIO_MARKER_FAULT_ACCESS_KEY") or ""
    secret_key = os.getenv("TEST_MINIO_MARKER_FAULT_SECRET_KEY") or ""
    if not access_key or not secret_key:
        message = "The hermetic REST gate's marker-failure MinIO credentials are required"
        if os.getenv("VANE_REQUIRE_ICEBERG_REST_TEST") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return access_key, secret_key


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _create_rest_v1_table(catalog_endpoint: str, namespace: str, table: str) -> None:
    payload = {
        "name": table,
        "schema": {
            "type": "struct",
            "schema-id": 0,
            "identifier-field-ids": [],
            "fields": [
                {"id": 1, "name": "event_id", "required": False, "type": "int"},
                {"id": 2, "name": "category", "required": True, "type": "string"},
            ],
        },
        "partition-spec": {"spec-id": 0, "fields": []},
        "write-order": {"order-id": 0, "fields": []},
        "stage-create": False,
        "properties": {"format-version": "1"},
    }
    create_request = request.Request(
        f"{catalog_endpoint.rstrip('/')}/v1/namespaces/{quote(namespace, safe='')}/tables",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(create_request, timeout=30) as response:
            if response.status not in {200, 201}:
                raise RuntimeError(f"Iceberg REST v1 table create returned HTTP {response.status}")
    except error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"Iceberg REST v1 table create failed: HTTP {exc.code}: {detail}") from exc


def _minio_client(minio_endpoint: str, access_key: str, secret_key: str, region: str):
    from botocore.config import Config
    from botocore.session import get_session

    return get_session().create_client(
        "s3",
        endpoint_url=minio_endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(
            signature_version="s3v4",
            connect_timeout=2,
            read_timeout=5,
            retries={"max_attempts": 1},
            s3={"addressing_style": "path"},
        ),
    )


def _bucket_object_keys(client, bucket: str) -> set[str]:
    keys: set[str] = set()
    continuation_token = None
    while True:
        request = {"Bucket": bucket}
        if continuation_token is not None:
            request["ContinuationToken"] = continuation_token
        response = client.list_objects_v2(**request)
        keys.update(item["Key"] for item in response.get("Contents", []))
        if not response.get("IsTruncated"):
            return keys
        continuation_token = response["NextContinuationToken"]


def _write_artifact_keys(keys: set[str]) -> set[str]:
    return {key for key in keys if key.endswith(".parquet") or ".duckdb_commit/" in key}


def _durable_write_artifact_keys(keys: set[str]) -> set[str]:
    return {
        key for key in keys if key.endswith(".parquet") or key.endswith("/manifest.txt") or key.endswith("/committed")
    }


def _copy_base_path_from_lifecycle_key(bucket: str, key: str) -> str:
    commit_separator = ".duckdb_commit/"
    if not key.endswith("/lifecycle.txt") or commit_separator not in key:
        raise ValueError(f"Not a distributed COPY lifecycle key: {key}")
    return f"s3://{bucket}/{key.split(commit_separator, 1)[0]}"


def _create_parquet_input(connection, input_dir, shard_queries: list[str]):
    input_dir.mkdir()
    for shard, query in enumerate(shard_queries):
        shard_path = input_dir / f"part-{shard}.parquet"
        connection.execute(f"COPY ({query}) TO {_sql_literal(shard_path.as_posix())} (FORMAT PARQUET)")
    return connection.sql(
        f"SELECT event_id, category FROM read_parquet({_sql_literal((input_dir / '*.parquet').as_posix())})"
    )


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
def test_distributed_insert_commits_one_iceberg_snapshot(monkeypatch, tmp_path):
    if ray is None:
        if os.getenv("VANE_REQUIRE_ICEBERG_REST_TEST") == "1":
            pytest.fail("ray is required by the hermetic Iceberg REST gate")
        pytest.skip("ray not installed")

    catalog_endpoint, minio_endpoint, access_key, secret_key, region = _required_service_config()
    schema_name = f"vane_write_{uuid.uuid4().hex}"
    table_name = f"rest_catalog.{schema_name}.events"

    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "1")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "2")

    source_connection = vane.connect()
    verifier_connection = None
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
        source_connection.execute(f"CREATE TABLE {table_name} (event_id INTEGER, category VARCHAR NOT NULL)")

        input_dir = tmp_path / "iceberg-write-input"
        input_dir.mkdir()
        for shard in range(4):
            shard_start = shard * 2500
            shard_end = shard_start + 2500
            shard_path = input_dir / f"part-{shard}.parquet"
            source_connection.execute(
                "COPY ("
                "SELECT i::INTEGER AS event_id, concat('category-', i % 7)::VARCHAR AS category "
                f"FROM range({shard_start}, {shard_end}) rows(i)"
                f") TO {_sql_literal(shard_path.as_posix())} (FORMAT PARQUET)"
            )
        source = source_connection.sql(
            f"SELECT event_id, category FROM read_parquet({_sql_literal((input_dir / '*.parquet').as_posix())})"
        )
        assert source.insert_into(table_name) is None
        empty_source = source_connection.sql(
            f"SELECT event_id, category FROM read_parquet({_sql_literal((input_dir / '*.parquet').as_posix())}) "
            "WHERE random() < 0"
        )
        assert empty_source.insert_into(table_name) is None

        # The write is planned and committed on the distributed coordinator's
        # replayed session. Verify through a fresh Catalog attachment so this
        # assertion cannot pass from the source connection's cached table entry.
        verifier_connection = vane.connect()
        _configure_catalog_connection(
            verifier_connection,
            catalog_endpoint,
            minio_endpoint,
            access_key,
            secret_key,
            region,
        )
        assert verifier_connection.execute(f"SELECT count(*) FROM {table_name}").fetchone() == (10000,)
        assert verifier_connection.execute(
            f"SELECT min(event_id), max(event_id), count(DISTINCT category) FROM {table_name}"
        ).fetchone() == (0, 9999, 7)
        assert verifier_connection.execute(f"SELECT count(*) FROM iceberg_snapshots({table_name})").fetchone() == (1,)
    finally:
        if verifier_connection is not None:
            verifier_connection.close()
        source_connection.close()
        if hasattr(vane, "teardown_runner"):
            vane.teardown_runner()


@pytest.mark.external_service
@pytest.mark.iceberg_rest
@pytest.mark.real_ray
@pytest.mark.usefixtures("ray_local")
def test_distributed_insert_rejects_v1_before_worker_output(monkeypatch, tmp_path):
    if ray is None:
        if os.getenv("VANE_REQUIRE_ICEBERG_REST_TEST") == "1":
            pytest.fail("ray is required by the hermetic Iceberg REST gate")
        pytest.skip("ray not installed")

    catalog_endpoint, minio_endpoint, access_key, secret_key, region = _required_service_config()
    bucket = os.environ["TEST_MINIO_BUCKET"]
    schema_name = f"vane_v1_rejection_{uuid.uuid4().hex}"
    table_name = f"rest_catalog.{schema_name}.events"

    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "1")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "2")

    s3_client = _minio_client(minio_endpoint, access_key, secret_key, region)
    source_connection = vane.connect()
    verifier_connection = None
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
        _create_rest_v1_table(catalog_endpoint, schema_name, "events")
        source_connection.close()
        source_connection = vane.connect()
        _configure_catalog_connection(
            source_connection,
            catalog_endpoint,
            minio_endpoint,
            access_key,
            secret_key,
            region,
        )
        before_artifacts = _write_artifact_keys(_bucket_object_keys(s3_client, bucket))
        source = _create_parquet_input(
            source_connection,
            tmp_path / "iceberg-v1-rejection-input",
            ["SELECT 1::INTEGER AS event_id, 'unsupported'::VARCHAR AS category"],
        )

        with pytest.raises(Exception, match="Distributed Iceberg writes require an Iceberg v2 table"):
            source.insert_into(table_name)

        verifier_connection = vane.connect()
        _configure_catalog_connection(
            verifier_connection,
            catalog_endpoint,
            minio_endpoint,
            access_key,
            secret_key,
            region,
        )
        assert verifier_connection.execute(f"SELECT count(*) FROM {table_name}").fetchone() == (0,)
        assert verifier_connection.execute(f"SELECT count(*) FROM iceberg_snapshots({table_name})").fetchone() == (0,)
        after_artifacts = _write_artifact_keys(_bucket_object_keys(s3_client, bucket))
        assert after_artifacts == before_artifacts
    finally:
        if verifier_connection is not None:
            verifier_connection.close()
        source_connection.close()
        s3_client.close()
        if hasattr(vane, "teardown_runner"):
            vane.teardown_runner()


@pytest.mark.external_service
@pytest.mark.iceberg_rest
@pytest.mark.real_ray
@pytest.mark.usefixtures("ray_local")
def test_distributed_insert_worker_failure_does_not_commit_snapshot(monkeypatch, tmp_path):
    if ray is None:
        if os.getenv("VANE_REQUIRE_ICEBERG_REST_TEST") == "1":
            pytest.fail("ray is required by the hermetic Iceberg REST gate")
        pytest.skip("ray not installed")

    catalog_endpoint, minio_endpoint, access_key, secret_key, region = _required_service_config()
    bucket = os.environ["TEST_MINIO_BUCKET"]
    schema_name = f"vane_worker_failure_{uuid.uuid4().hex}"
    table_name = f"rest_catalog.{schema_name}.events"

    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "1")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "2")

    s3_client = _minio_client(minio_endpoint, access_key, secret_key, region)
    source_connection = vane.connect()
    verifier_connection = None
    before_objects: set[str] = set()
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
        source_connection.execute(f"CREATE TABLE {table_name} (event_id INTEGER, category VARCHAR NOT NULL)")
        before_objects = _bucket_object_keys(s3_client, bucket)

        source = _create_parquet_input(
            source_connection,
            tmp_path / "iceberg-worker-failure-input",
            [
                "SELECT i::BIGINT AS event_id, 'valid'::VARCHAR AS category FROM range(0, 250) rows(i)",
                "SELECT 2147483648::BIGINT AS event_id, 'overflow'::VARCHAR AS category",
            ],
        )
        with pytest.raises(Exception) as exc_info:
            source.insert_into(table_name)
        error_message = str(exc_info.value).lower()
        assert "2147483648" in error_message or "out of range" in error_message

        verifier_connection = vane.connect()
        _configure_catalog_connection(
            verifier_connection,
            catalog_endpoint,
            minio_endpoint,
            access_key,
            secret_key,
            region,
        )
        assert verifier_connection.execute(f"SELECT count(*) FROM {table_name}").fetchone() == (0,)
        assert verifier_connection.execute(f"SELECT count(*) FROM iceberg_snapshots({table_name})").fetchone() == (0,)
        after_objects = _bucket_object_keys(s3_client, bucket)
        new_objects = after_objects - before_objects
        assert _durable_write_artifact_keys(new_objects) == set()

        # A worker can fail before its output metadata reaches the coordinator.
        # The direct-write protocol may conservatively retain only its lifecycle
        # registration so an operator can retry cleanup without guessing object
        # ownership. Prove that this recovery path removes the registration and
        # leaves no data, manifest, or committed marker behind.
        lifecycle_keys = {key for key in new_objects if key.endswith("/lifecycle.txt")}
        assert new_objects == lifecycle_keys
        if lifecycle_keys:
            from vane.runners.ray import cleanup_copy_direct_write_lifecycle_once

            base_paths = sorted(_copy_base_path_from_lifecycle_key(bucket, key) for key in lifecycle_keys)
            cleanup = cleanup_copy_direct_write_lifecycle_once(
                base_paths,
                min_age_ms=0,
                fail_fast=True,
                conn=verifier_connection,
            )
            assert cleanup["errors"] == 0, cleanup["error_messages"]
            assert cleanup["cleaned_runs"] == len(lifecycle_keys)
            assert (
                _write_artifact_keys(_bucket_object_keys(s3_client, bucket)) - _write_artifact_keys(before_objects)
                == set()
            )
    finally:
        if verifier_connection is not None:
            verifier_connection.close()
        source_connection.close()
        s3_client.close()
        if hasattr(vane, "teardown_runner"):
            vane.teardown_runner()


@pytest.mark.external_service
@pytest.mark.iceberg_rest
@pytest.mark.real_ray
@pytest.mark.usefixtures("ray_local")
def test_rest_catalog_commit_rejection_reports_unknown_and_retains_output(monkeypatch, tmp_path):
    if ray is None:
        if os.getenv("VANE_REQUIRE_ICEBERG_REST_TEST") == "1":
            pytest.fail("ray is required by the hermetic Iceberg REST gate")
        pytest.skip("ray not installed")

    from vane.runners import CopyOutcomeUnknownError

    catalog_endpoint, minio_endpoint, access_key, secret_key, region = _required_service_config()
    bucket = os.environ["TEST_MINIO_BUCKET"]
    schema_name = f"vane_commit_failure_{uuid.uuid4().hex}"
    table_name = f"rest_catalog.{schema_name}.events"

    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "1")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "2")

    proxy = _RestCommitFailureProxy(catalog_endpoint)
    s3_client = _minio_client(minio_endpoint, access_key, secret_key, region)
    source_connection = vane.connect()
    verifier_connection = None
    before_artifacts: set[str] = set()
    try:
        _configure_catalog_connection(
            source_connection,
            proxy.endpoint,
            minio_endpoint,
            access_key,
            secret_key,
            region,
        )
        source_connection.execute(f"CREATE SCHEMA rest_catalog.{schema_name}")
        source_connection.execute(f"CREATE TABLE {table_name} (event_id INTEGER, category VARCHAR NOT NULL)")
        assert source_connection.execute(f"SELECT count(*) FROM {table_name}").fetchone() == (0,)
        assert proxy.cached_table_get_count >= 1
        before_artifacts = _write_artifact_keys(_bucket_object_keys(s3_client, bucket))
        source = _create_parquet_input(
            source_connection,
            tmp_path / "iceberg-commit-failure-input",
            [
                "SELECT i::INTEGER AS event_id, concat('category-', i % 3)::VARCHAR AS category "
                "FROM range(0, 500) rows(i)",
                "SELECT i::INTEGER AS event_id, concat('category-', i % 3)::VARCHAR AS category "
                "FROM range(500, 1000) rows(i)",
            ],
        )

        proxy.reject_commits()
        with pytest.raises(CopyOutcomeUnknownError) as commit_error:
            source.insert_into(table_name)
        assert "planned catalog commit failure" in str(commit_error.value)
        assert commit_error.value.safe_to_retry is False
        assert proxy.commit_attempts >= 1

        verifier_connection = vane.connect()
        _configure_catalog_connection(
            verifier_connection,
            catalog_endpoint,
            minio_endpoint,
            access_key,
            secret_key,
            region,
        )
        assert verifier_connection.execute(f"SELECT count(*) FROM {table_name}").fetchone() == (0,)
        assert verifier_connection.execute(f"SELECT count(*) FROM iceberg_snapshots({table_name})").fetchone() == (0,)
        after_artifacts = _write_artifact_keys(_bucket_object_keys(s3_client, bucket))
        new_artifacts = after_artifacts - before_artifacts
        assert any(key.endswith(".parquet") for key in new_artifacts)
        assert any(key.endswith("/manifest.txt") for key in new_artifacts)
        assert not any(key.endswith("/committed") for key in new_artifacts)
    finally:
        if verifier_connection is not None:
            verifier_connection.close()
        source_connection.close()
        proxy.close()
        s3_client.close()
        if hasattr(vane, "teardown_runner"):
            vane.teardown_runner()


@pytest.mark.external_service
@pytest.mark.iceberg_rest
@pytest.mark.real_ray
@pytest.mark.usefixtures("ray_local")
def test_rest_catalog_commit_response_loss_retains_committed_data(monkeypatch, tmp_path):
    if ray is None:
        if os.getenv("VANE_REQUIRE_ICEBERG_REST_TEST") == "1":
            pytest.fail("ray is required by the hermetic Iceberg REST gate")
        pytest.skip("ray not installed")

    from vane.runners import CopyOutcomeUnknownError

    catalog_endpoint, minio_endpoint, access_key, secret_key, region = _required_service_config()
    bucket = os.environ["TEST_MINIO_BUCKET"]
    schema_name = f"vane_commit_response_loss_{uuid.uuid4().hex}"
    table_name = f"rest_catalog.{schema_name}.events"

    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "1")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "2")

    proxy = _RestCommitFailureProxy(catalog_endpoint)
    s3_client = _minio_client(minio_endpoint, access_key, secret_key, region)
    source_connection = vane.connect()
    verifier_connection = None
    try:
        _configure_catalog_connection(
            source_connection,
            proxy.endpoint,
            minio_endpoint,
            access_key,
            secret_key,
            region,
        )
        source_connection.execute(f"CREATE SCHEMA rest_catalog.{schema_name}")
        source_connection.execute(f"CREATE TABLE {table_name} (event_id INTEGER, category VARCHAR NOT NULL)")
        before_artifacts = _write_artifact_keys(_bucket_object_keys(s3_client, bucket))
        source = _create_parquet_input(
            source_connection,
            tmp_path / "iceberg-commit-response-loss-input",
            [
                "SELECT i::INTEGER AS event_id, concat('category-', i % 3)::VARCHAR AS category "
                "FROM range(0, 500) rows(i)",
                "SELECT i::INTEGER AS event_id, concat('category-', i % 3)::VARCHAR AS category "
                "FROM range(500, 1000) rows(i)",
            ],
        )

        proxy.lose_commit_response()
        with pytest.raises(CopyOutcomeUnknownError) as commit_error:
            source.insert_into(table_name)
        assert "planned catalog commit response loss after upstream success" in str(commit_error.value)
        assert commit_error.value.safe_to_retry is False
        assert proxy.commit_attempts >= 1

        # The proxy let the Catalog commit complete before replacing its success
        # response. A fresh direct attachment must still read every referenced
        # data file even though Vane correctly refuses to publish its marker.
        verifier_connection = vane.connect()
        _configure_catalog_connection(
            verifier_connection,
            catalog_endpoint,
            minio_endpoint,
            access_key,
            secret_key,
            region,
        )
        assert verifier_connection.execute(f"SELECT count(*) FROM {table_name}").fetchone() == (1000,)
        assert verifier_connection.execute(
            f"SELECT min(event_id), max(event_id), count(DISTINCT category) FROM {table_name}"
        ).fetchone() == (0, 999, 3)
        assert verifier_connection.execute(f"SELECT count(*) FROM iceberg_snapshots({table_name})").fetchone() == (1,)
        new_artifacts = _write_artifact_keys(_bucket_object_keys(s3_client, bucket)) - before_artifacts
        assert any(key.endswith(".parquet") for key in new_artifacts)
        assert any(key.endswith("/manifest.txt") for key in new_artifacts)
        assert not any(key.endswith("/committed") for key in new_artifacts)
    finally:
        if verifier_connection is not None:
            verifier_connection.close()
        source_connection.close()
        proxy.close()
        s3_client.close()
        if hasattr(vane, "teardown_runner"):
            vane.teardown_runner()


@pytest.mark.external_service
@pytest.mark.iceberg_rest
@pytest.mark.real_ray
@pytest.mark.usefixtures("ray_local")
def test_committed_catalog_with_marker_failure_reports_unknown(monkeypatch, tmp_path):
    if ray is None:
        if os.getenv("VANE_REQUIRE_ICEBERG_REST_TEST") == "1":
            pytest.fail("ray is required by the hermetic Iceberg REST gate")
        pytest.skip("ray not installed")

    from botocore.exceptions import ClientError

    from vane.runners import CopyOutcomeUnknownError

    catalog_endpoint, minio_endpoint, access_key, secret_key, region = _required_service_config()
    marker_fault_access_key, marker_fault_secret_key = _required_marker_fault_credentials()
    bucket = os.environ["TEST_MINIO_BUCKET"]
    schema_name = f"vane_marker_failure_{uuid.uuid4().hex}"
    table_name = f"rest_catalog.{schema_name}.events"

    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "1")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "2")

    s3_client = _minio_client(minio_endpoint, access_key, secret_key, region)
    marker_fault_s3_client = _minio_client(
        minio_endpoint,
        marker_fault_access_key,
        marker_fault_secret_key,
        region,
    )
    source_connection = vane.connect()
    verifier_connection = None
    before_artifacts: set[str] = set()
    try:
        _configure_catalog_connection(
            source_connection,
            catalog_endpoint,
            minio_endpoint,
            marker_fault_access_key,
            marker_fault_secret_key,
            region,
        )
        source_connection.execute(f"CREATE SCHEMA rest_catalog.{schema_name}")
        source_connection.execute(f"CREATE TABLE {table_name} (event_id INTEGER, category VARCHAR NOT NULL)")
        probe_key = f"marker-policy-probe/{uuid.uuid4().hex}"
        marker_fault_s3_client.put_object(Bucket=bucket, Key=probe_key, Body=b"probe")
        marker_fault_s3_client.delete_object(Bucket=bucket, Key=probe_key)
        with pytest.raises(ClientError) as policy_error:
            marker_fault_s3_client.put_object(
                Bucket=bucket,
                Key="policy-probe/data.duckdb_commit/test/committed",
                Body=b"probe",
            )
        assert policy_error.value.response["Error"]["Code"] in {"AccessDenied", "XMinioAdminAccessDenied"}

        before_artifacts = _write_artifact_keys(_bucket_object_keys(s3_client, bucket))
        source = _create_parquet_input(
            source_connection,
            tmp_path / "iceberg-marker-failure-input",
            [
                "SELECT i::INTEGER AS event_id, concat('category-', i % 5)::VARCHAR AS category "
                "FROM range(0, 500) rows(i)",
                "SELECT i::INTEGER AS event_id, concat('category-', i % 5)::VARCHAR AS category "
                "FROM range(500, 1000) rows(i)",
            ],
        )

        with pytest.raises(CopyOutcomeUnknownError) as outcome_error:
            source.insert_into(table_name)
        unknown_outcome = outcome_error.value
        assert unknown_outcome.safe_to_retry is False
        assert unknown_outcome.base_path.startswith(f"s3://{bucket}/")
        assert unknown_outcome.run_id
        assert unknown_outcome.manifest_path.endswith("/manifest.txt")
        assert unknown_outcome.committed_marker_path.endswith("/committed")
        assert "403" in unknown_outcome.detail or "AccessDenied" in unknown_outcome.detail

        verifier_connection = vane.connect()
        _configure_catalog_connection(
            verifier_connection,
            catalog_endpoint,
            minio_endpoint,
            access_key,
            secret_key,
            region,
        )
        assert verifier_connection.execute(f"SELECT count(*) FROM {table_name}").fetchone() == (1000,)
        assert verifier_connection.execute(f"SELECT count(*) FROM iceberg_snapshots({table_name})").fetchone() == (1,)
        after_artifacts = _write_artifact_keys(_bucket_object_keys(s3_client, bucket))
        new_artifacts = after_artifacts - before_artifacts
        assert any(key.endswith(".parquet") for key in new_artifacts)
        assert any(key.endswith("/manifest.txt") for key in new_artifacts)
        assert not any(key.endswith("/committed") for key in new_artifacts)
    finally:
        if verifier_connection is not None:
            verifier_connection.close()
        source_connection.close()
        marker_fault_s3_client.close()
        s3_client.close()
        if hasattr(vane, "teardown_runner"):
            vane.teardown_runner()


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
