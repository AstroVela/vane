# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import queue
import socket
import socketserver
import struct
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from pathlib import Path

import pytest

import duckdb


def _make_test_physical_plan(con=None):
    con = duckdb.connect() if con is None else con
    relation = con.sql("SELECT 1 AS i")
    return duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)


def test_distributed_physical_plan_inspectors():
    con = duckdb.connect()
    plan = _make_test_physical_plan(con)

    idx = plan.idx()
    assert isinstance(idx, str)
    assert plan.has_root() is True
    assert isinstance(plan.num_partitions(), int)
    assert isinstance(plan.repr_ascii(False), str)
    assert isinstance(plan.repr_mermaid(False, False), str)
    assert isinstance(plan.scan_task_descriptor_map(), dict)


def test_distributed_physical_plan_runner_run_plan_accepts_none():
    m = duckdb.ray_cxx
    runner = m.DistributedPhysicalPlanRunner()

    with pytest.raises(TypeError, match="plan must be DistributedPhysicalPlan \\(PyPhysicalPlanWrapper\\)"):
        runner.run_plan(None)


def test_fte_split_queue_basic_states():
    queue = duckdb.ray_cxx.FteSplitQueue()

    assert queue.try_get_next() == {"state": "BLOCKED"}
    queue.add_scan_split(b"scan-a")
    queue.add_exchange_source_split(b"exchange-b")
    assert queue.buffered_splits() == 2

    first = queue.try_get_next()
    second = queue.try_get_next()
    assert first == {"state": "SPLIT", "kind": "scan_task", "data": b"scan-a"}
    assert second == {
        "state": "SPLIT",
        "kind": "exchange_source_task",
        "data": b"exchange-b",
    }
    assert queue.try_get_next() == {"state": "BLOCKED"}
    queue.no_more_splits()
    assert queue.try_get_next() == {"state": "FINISHED"}


def test_fte_split_queue_tracks_exchange_source_progress_stats():
    queue = duckdb.ray_cxx.FteSplitQueue()
    raw = duckdb.ray_cxx.make_exchange_source_task_descriptor_for_test(
        [
            {
                "partition_id": 0,
                "attempt_id": 0,
                "node_id": "node-a",
                "flight_port": 5010,
                "files": [
                    {"path": "shuffle-a", "rows": 7, "file_size": 128},
                    {"path": "shuffle-b", "rows": 5, "file_size": 64},
                ],
            }
        ],
        [0],
        1,
        1,
    )

    queue.add_exchange_source_split(raw)

    assert queue.submitted_rows() == 12
    assert queue.submitted_input_bytes() == 192
    assert queue.consumed_rows() == 0
    assert queue.consumed_input_bytes() == 0

    split = queue.try_get_next()

    assert split["state"] == "SPLIT"
    assert queue.consumed_rows() == 12
    assert queue.consumed_input_bytes() == 192


def test_fte_split_queue_tracks_exchange_source_width_metadata():
    queue = duckdb.ray_cxx.FteSplitQueue()
    raw = duckdb.ray_cxx.make_exchange_source_task_descriptor_for_test(
        [
            {
                "partition_id": 2,
                "attempt_id": 0,
                "node_id": "node-a",
                "flight_port": 5010,
                "files": [{"path": "shuffle-c", "rows": 3, "file_size": 32}],
            }
        ],
        [2],
        16,
        8,
    )

    queue.add_exchange_source_split(raw)

    assert queue.exchange_source_partition_count() == 16
    assert queue.exchange_source_task_count() == 8


def test_exchange_source_task_partition_indices_accepts_empty_descriptor():
    assert duckdb.ray_cxx.exchange_source_task_partition_indices(b"") == []
    assert duckdb.ray_cxx.split_exchange_source_task_by_partition(b"") == []


def test_exchange_source_task_descriptor_preserves_attempt_ids():
    handles = [
        {
            "partition_id": 0,
            "attempt_id": 7,
            "node_id": "node-a",
            "flight_port": 5010,
            "files": [{"path": "shuffle__sink_0__attempt_7", "file_size": 11}],
        },
        {
            "partition_id": 1,
            "attempt_id": 2,
            "node_id": "node-b",
            "flight_port": 5011,
            "files": [{"path": "shuffle__sink_1__attempt_2", "file_size": 17}],
        },
    ]

    raw = duckdb.ray_cxx.make_exchange_source_task_descriptor_for_test(
        handles,
        [0, 1],
        2,
        2,
    )

    assert duckdb.ray_cxx.exchange_source_task_partition_indices(raw) == [0, 1]
    assert duckdb.ray_cxx.exchange_source_task_replicated(raw) is False
    assert duckdb.ray_cxx.exchange_source_task_source_handles_for_test(raw) == handles

    split = duckdb.ray_cxx.split_exchange_source_task_by_partition(raw)
    assert [
        (partition_id, partition_count, task_count, replicated)
        for partition_id, _, partition_count, task_count, replicated in split
    ] == [
        (0, 2, 2, False),
        (1, 2, 2, False),
    ]
    assert duckdb.ray_cxx.exchange_source_task_source_handles_for_test(split[0][1]) == [handles[0]]
    assert duckdb.ray_cxx.exchange_source_task_source_handles_for_test(split[1][1]) == [handles[1]]


def test_exchange_source_task_descriptor_preserves_replicated_distribution():
    handles = [
        {
            "partition_id": 0,
            "attempt_id": 3,
            "node_id": "node-a",
            "flight_port": 5010,
            "files": [{"path": "broadcast_shuffle__sink_0__attempt_3", "file_size": 11}],
        },
    ]

    raw = duckdb.ray_cxx.make_exchange_source_task_descriptor_for_test(
        handles,
        [0],
        1,
        1,
        replicated=True,
    )

    assert duckdb.ray_cxx.exchange_source_task_replicated(raw) is True
    split = duckdb.ray_cxx.split_exchange_source_task_by_partition(raw)
    assert [
        (partition_id, partition_count, task_count, replicated)
        for partition_id, _, partition_count, task_count, replicated in split
    ] == [
        (0, 1, 1, True),
    ]
    assert duckdb.ray_cxx.exchange_source_task_replicated(split[0][1]) is True
    assert duckdb.ray_cxx.exchange_source_task_source_handles_for_test(split[0][1]) == handles


def test_flight_exchange_selected_attempt_runtime_path():
    handles = duckdb.ray_cxx.flight_exchange_selected_attempt_handles_for_test()

    assert len(handles) == 4
    sink0_handles = [h for h in handles if "__sink_0__" in h["path"]]
    sink1_handles = [h for h in handles if "__sink_1__" in h["path"]]
    assert len(sink0_handles) == 2
    assert len(sink1_handles) == 2
    assert {h["partition_id"] for h in sink0_handles} == {0, 1}
    assert all(h["attempt_id"] == 1 for h in sink0_handles)
    assert all(h["node_id"] == "worker-retry" for h in sink0_handles)
    assert all(h["flight_port"] == 5010 for h in sink0_handles)
    assert all("__attempt_1" in h["path"] for h in sink0_handles)
    assert all(h["attempt_id"] == 0 for h in sink1_handles)
    assert all(h["node_id"] == "worker-first" for h in sink1_handles)
    assert all(h["flight_port"] == 5012 for h in sink1_handles)


def test_flight_exchange_materialized_output_attempt_metadata_drives_completion():
    handles = duckdb.ray_cxx.flight_exchange_materialized_output_attempt_metadata_for_test()

    assert len(handles) == 4
    sink0_handles = [h for h in handles if "__sink_0__" in h["path"]]
    sink1_handles = [h for h in handles if "__sink_1__" in h["path"]]
    assert len(sink0_handles) == 2
    assert len(sink1_handles) == 2
    assert all(h["attempt_id"] == 1 for h in sink0_handles)
    assert all(h["node_id"] == "worker-retry" for h in sink0_handles)
    assert all(h["flight_port"] == 5010 for h in sink0_handles)
    assert all("__attempt_1" in h["path"] for h in sink0_handles)
    assert all("__attempt_0" not in h["path"] for h in sink0_handles)
    assert all(h["attempt_id"] == 0 for h in sink1_handles)
    assert all(h["node_id"] == "worker-first" for h in sink1_handles)
    assert all(h["flight_port"] == 5012 for h in sink1_handles)


def test_ray_task_result_handle_uses_refreshed_worker_id_at_completion():
    class _AdoptingHandle:
        worker_id = "worker-original"

        def __init__(self):
            self._is_done = False
            self._result = None
            self._error = None
            self._future = None
            self.task = None
            self.release_calls = 0

        def _ensure_started(self):
            return None

        def done(self):
            self.worker_id = "worker-retry"
            self._is_done = True
            return True

        def get_result_sync(self):
            return duckdb.ray_cxx.RayTaskResult.success(
                [],
                [],
                None,
                5010,
                {"sink_handle": {"partition_id": 0}, "attempt_id": 1},
            )

        def release_result_payload(self):
            self.release_calls += 1

    handle = _AdoptingHandle()
    result = duckdb.ray_cxx.ray_task_result_handle_refreshed_worker_id_for_test(handle)

    assert result["worker_id"] == "worker-retry"
    assert result["has_output"] is True
    assert result["flight_port"] == 5010
    assert handle.release_calls == 1


def test_flight_exchange_source_reads_only_selected_retry_attempt_data(tmp_path):
    result = duckdb.ray_cxx.flight_exchange_selected_attempt_dataplane_for_test(str(tmp_path))

    assert result["handle_attempts"] == [1]
    assert result["handle_paths"] == [result["selected_output_location"]]
    assert result["handle_node_ids"] == [result["selected_node_id"]]
    assert result["lost_output_location"] != result["selected_output_location"]
    assert result["selected_values_before_late_loser"] == [201, 202]
    assert result["selected_values_after_late_loser"] == [201, 202]
    assert result["lost_manifest_exists_after_late_loser"] is False
    assert result["selected_manifest_exists_after_late_loser"] is True


def test_flight_exchange_cleans_successful_unselected_attempt(tmp_path):
    result = duckdb.ray_cxx.flight_exchange_unselected_attempt_cleanup_for_test(str(tmp_path))

    assert result["selected_manifest_before"] is True
    assert result["loser_manifest_before"] is True
    assert result["selected_registry_before"] is True
    assert result["loser_registry_before"] is True
    assert result["selected_manifest_after_cleanup"] is True
    assert result["loser_manifest_after_cleanup"] is False
    assert result["selected_registry_after_cleanup"] is True
    assert result["loser_registry_after_cleanup"] is False
    assert result["selected_manifest_after_close"] is True
    assert result["selected_registry_after_close"] is False
    assert set(result["handle_attempts"]) == {0}
    assert all(path == result["selected_output_location"] for path in result["handle_paths"])
    assert result["selected_output_location"] != result["loser_output_location"]


def test_shuffle_cache_registry_query_cleanup_removes_attempt_storage(tmp_path):
    result = duckdb.ray_cxx.shuffle_cache_registry_query_cleanup_for_test(str(tmp_path))

    assert result["registry_entries_removed"] == 1
    assert result["storage_entries_removed"] > 0
    assert result["cleanup_errors"] == 0
    assert result["cleanup_registry_after_defer"] is False
    assert result["cleanup_registry_after"] is False
    assert result["keep_registry_after"] is True
    assert result["cleanup_node_dir_exists_after"] is False
    assert result["keep_node_dir_exists_after"] is True


def test_flight_exchange_local_dirs_env_keeps_object_uri_intact(monkeypatch):
    monkeypatch.setenv("DUCKDB_SHUFFLE_DIRS", "s3://bucket/shuffle")
    result = duckdb.ray_cxx.flight_exchange_local_dirs_from_env_for_test()
    assert result == ["s3://bucket/shuffle"]


def test_flight_exchange_local_dirs_env_supports_multiple_paths(monkeypatch):
    monkeypatch.setenv("DUCKDB_SHUFFLE_DIRS", "file:///tmp/a,file:///tmp/b")
    result = duckdb.ray_cxx.flight_exchange_local_dirs_from_env_for_test()
    assert result == ["file:///tmp/a", "file:///tmp/b"]


def test_flight_exchange_local_dirs_env_supports_vane_alias(monkeypatch, tmp_path):
    shuffle_dir = tmp_path / "shuffle"
    monkeypatch.delenv("DUCKDB_SHUFFLE_DIRS", raising=False)
    monkeypatch.setenv("VANE_SHUFFLE_LOCAL_DIRS", str(shuffle_dir))

    result = duckdb.ray_cxx.flight_exchange_local_dirs_from_env_for_test()

    assert result == [str(shuffle_dir)]


def test_flight_exchange_local_dirs_default_uses_vane_session(monkeypatch, tmp_path):
    session_dir = tmp_path / "session"
    monkeypatch.delenv("DUCKDB_SHUFFLE_DIRS", raising=False)
    monkeypatch.delenv("VANE_SHUFFLE_LOCAL_DIRS", raising=False)
    monkeypatch.setenv("VANE_SESSION_DIR", str(session_dir))

    result = duckdb.ray_cxx.flight_exchange_local_dirs_from_env_for_test()

    assert result == [str(session_dir / "flight_shuffle")]


def test_flight_exchange_node_id_prefers_vane_worker_id(monkeypatch):
    monkeypatch.setenv("VANE_WORKER_ID", "vane-worker")
    monkeypatch.setenv("RAY_NODE_IP_ADDRESS", "192.0.2.10")
    assert duckdb.ray_cxx.flight_exchange_node_id_from_env_for_test() == "vane-worker"


def test_shuffle_cache_attempt_manifest_runtime_path(tmp_path):
    result = duckdb.ray_cxx.shuffle_cache_attempt_manifest_for_test(str(tmp_path))

    assert Path(result["manifest_path"]).exists()
    assert Path(result["committed_path"]).exists()
    manifest = result["manifest"]
    assert "version=1\n" in manifest
    assert "shuffle_stage_id=shuffle_cache_manifest_test__sink_3__attempt_2\n" in manifest
    assert "node_id=node-a\n" in manifest
    assert "sink_partition_id=3\n" in manifest
    assert "attempt_id=2\n" in manifest
    assert "output_partition_count=2\n" in manifest
    assert f"file=0\t11\t4\t{tmp_path}/partition_0/batch.arrow\n" in manifest


def test_shuffle_cache_manifest_recovery_after_registry_loss(tmp_path):
    result = duckdb.ray_cxx.shuffle_cache_manifest_recovery_for_test(str(tmp_path))

    assert result["memory_file_count"] == 0
    assert result["manifest_file_count"] == 1
    assert result["row_count"] == 3
    assert result["values"] == [11, 12, 13]
    assert Path(result["manifest_path"]).exists()


def test_shuffle_cache_does_not_recover_uncommitted_files(tmp_path):
    result = duckdb.ray_cxx.shuffle_cache_uncommitted_files_invisible_for_test(str(tmp_path))

    assert result["partial_file_count"] == 1
    assert result["committed_manifest"] is False
    assert result["recovered_row_count"] == 0


def test_shuffle_cache_duckdb_filesystem_storage_roundtrip(tmp_path):
    result = duckdb.ray_cxx.shuffle_cache_duckdb_filesystem_storage_roundtrip_for_test(str(tmp_path))

    assert result["committed_manifest"] is True
    assert result["manifest_file_count"] == 1
    assert result["manifest_total_rows"] == 4
    assert result["row_count"] == 4
    assert result["values"] == [71, 72, 73, 74]
    assert result["manifest_tmp_exists"] is False
    assert result["marker_tmp_exists"] is False


def _minio_test_config():
    endpoint = os.getenv("TEST_MINIO_ENDPOINT") or os.getenv("AWS_ENDPOINT_URL") or "http://127.0.0.1:9000"
    access_key = os.getenv("TEST_MINIO_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID") or ""
    secret_key = os.getenv("TEST_MINIO_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY") or ""
    region = os.getenv("TEST_MINIO_REGION") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    bucket = os.getenv("TEST_MINIO_BUCKET") or "vane-shuffle-test"
    base_uri = f"s3://{bucket}/shuffle-cache-minio-test/{uuid.uuid4()}"
    return endpoint, access_key, secret_key, region, bucket, base_uri


def _sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _configure_duckdb_s3(
    conn,
    endpoint,
    access_key,
    secret_key,
    region,
    *,
    http_retries=2,
    http_timeout=None,
):
    from urllib.parse import urlparse

    parsed = urlparse(endpoint)
    duckdb_endpoint = parsed.netloc or parsed.path
    conn.execute("LOAD httpfs")
    conn.execute(f"SET s3_endpoint={_sql_string_literal(duckdb_endpoint)}")
    conn.execute(f"SET s3_use_ssl={'true' if parsed.scheme == 'https' else 'false'}")
    conn.execute("SET s3_url_style='path'")
    conn.execute(f"SET s3_region={_sql_string_literal(region)}")
    conn.execute(f"SET s3_access_key_id={_sql_string_literal(access_key)}")
    conn.execute(f"SET s3_secret_access_key={_sql_string_literal(secret_key)}")
    conn.execute("SET http_proxy=''")
    conn.execute("SET http_keep_alive=true")
    conn.execute(f"SET http_retries={int(http_retries)}")
    conn.execute("SET http_retry_wait_ms=50")
    conn.execute("SET http_retry_backoff=1.5")
    if http_timeout is not None:
        conn.execute(f"SET http_timeout={float(http_timeout)}")


def _socket_close_with_reset(sock):
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    except OSError:
        pass
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


def _recv_until_headers(sock):
    sock.settimeout(2.0)
    chunks = []
    total = 0
    while total < 65536:
        chunk = sock.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if b"\r\n\r\n" in b"".join(chunks):
            break
    return b"".join(chunks)


def _recv_all(sock):
    sock.settimeout(2.0)
    chunks = []
    while True:
        try:
            chunk = sock.recv(4096)
        except TimeoutError:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


class _FaultProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _start_fault_proxy(upstream_address, mode):
    class FaultProxyHandler(socketserver.BaseRequestHandler):
        requests = []

        def handle(self):
            request = _recv_until_headers(self.request)
            type(self).requests.append(request)
            if not request:
                return
            response = b""
            with socket.create_connection(upstream_address, timeout=2.0) as upstream:
                upstream.sendall(request)
                response = _recv_all(upstream)

            if mode == "reset_after_upstream":
                _socket_close_with_reset(self.request)
                return
            if mode == "partial_after_upstream":
                header_end = response.find(b"\r\n\r\n")
                partial_len = min(len(response), (header_end + 4 if header_end >= 0 else 0) + 12)
                if partial_len <= 0:
                    partial_len = max(1, len(response) // 2)
                self.request.sendall(response[:partial_len])
                _socket_close_with_reset(self.request)
                return
            if mode == "slow_body_after_upstream":
                header_end = response.find(b"\r\n\r\n")
                if header_end < 0:
                    self.request.sendall(response[:1])
                    time.sleep(1.0)
                    _socket_close_with_reset(self.request)
                    return
                header_end += 4
                self.request.sendall(response[:header_end])
                for byte in response[header_end : header_end + 3]:
                    time.sleep(1.0)
                    try:
                        self.request.sendall(bytes([byte]))
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
                _socket_close_with_reset(self.request)
                return
            raise AssertionError(f"unknown proxy fault mode: {mode}")

    server = _FaultProxyServer(("127.0.0.1", 0), FaultProxyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, FaultProxyHandler


def _start_s3_list_ok_server():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class OkListS3Handler(BaseHTTPRequestHandler):
        requests = []

        def do_GET(self):
            type(self).requests.append((self.command, self.path))
            body = (
                b'<?xml version="1.0" encoding="UTF-8"?>'
                b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                b"<Name>bucket</Name>"
                b"<Prefix>prefix/</Prefix>"
                b"<KeyCount>0</KeyCount>"
                b"<MaxKeys>1000</MaxKeys>"
                b"<IsTruncated>false</IsTruncated>"
                b"</ListBucketResult>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), OkListS3Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, OkListS3Handler


def _run_object_store_proxy_fault(mode, *, http_timeout=None):
    upstream_server, upstream_thread, upstream_handler = _start_s3_list_ok_server()
    proxy_server, proxy_thread, proxy_handler = _start_fault_proxy(
        upstream_server.server_address,
        mode,
    )
    try:
        endpoint = f"http://127.0.0.1:{proxy_server.server_address[1]}"
        conn = duckdb.connect()
        try:
            _configure_duckdb_s3(
                conn,
                endpoint,
                "access-key",
                "secret-key",
                "us-east-1",
                http_retries=1,
                http_timeout=http_timeout,
            )
            conn.execute("SET http_keep_alive=false")
            conn.execute("SET http_retry_wait_ms=10")
            conn.execute("SET http_retry_backoff=1")
            with pytest.raises(Exception) as excinfo:
                conn.execute("SELECT * FROM glob('s3://bucket/prefix/*')").fetchall()
            message = str(excinfo.value)
        finally:
            conn.close()
    finally:
        proxy_server.shutdown()
        proxy_server.server_close()
        proxy_thread.join(timeout=2)
        upstream_server.shutdown()
        upstream_server.server_close()
        upstream_thread.join(timeout=2)
    return message, list(proxy_handler.requests), list(upstream_handler.requests)


def _s3_glob_paths_fresh(endpoint, access_key, secret_key, region, pattern):
    conn = duckdb.connect()
    try:
        _configure_duckdb_s3(conn, endpoint, access_key, secret_key, region)
        return [row[0] for row in conn.execute("SELECT * FROM glob(?)", [pattern]).fetchall()]
    finally:
        conn.close()


def _skip_unless_minio_writable(endpoint, access_key, secret_key, region, bucket):
    probe_path = f"s3://{bucket}/shuffle-cache-minio-preflight/{uuid.uuid4()}/probe.parquet"
    conn = duckdb.connect()
    try:
        _configure_duckdb_s3(conn, endpoint, access_key, secret_key, region)
        conn.execute(f"COPY (SELECT 1 AS value) TO '{probe_path}' (FORMAT PARQUET)")
        assert conn.execute(f"SELECT value FROM read_parquet('{probe_path}')").fetchone()[0] == 1
    except Exception as exc:
        pytest.skip(f"MinIO/S3-compatible endpoint is not writable for this test: {exc}")
    finally:
        conn.close()


@pytest.mark.external_service
def test_shuffle_cache_duckdb_filesystem_storage_minio_roundtrip():
    endpoint, access_key, secret_key, region, bucket, base_uri = _minio_test_config()
    _skip_unless_minio_writable(endpoint, access_key, secret_key, region, bucket)

    result = duckdb.ray_cxx.shuffle_cache_duckdb_filesystem_storage_minio_roundtrip_for_test(
        base_uri,
        endpoint,
        access_key,
        secret_key,
        region,
    )

    assert result["committed_manifest"] is True
    assert result["manifest_exists_before_cleanup"] is True
    assert result["marker_exists_before_cleanup"] is True
    assert result["manifest_file_count"] == 1
    assert result["manifest_total_rows"] == 4
    assert result["row_count"] == 4
    assert result["values"] == [171, 172, 173, 174]
    assert result["manifest_tmp_exists"] is False
    assert result["marker_tmp_exists"] is False
    assert result["cleanup_removed"] >= 1
    attempt_prefix = result["manifest_path"].rsplit("/", 1)[0]
    assert _s3_glob_paths_fresh(endpoint, access_key, secret_key, region, f"{attempt_prefix}/**") == []


@pytest.mark.external_service
def test_shuffle_cache_duckdb_filesystem_storage_minio_bad_credentials_hard_fail():
    endpoint, access_key, secret_key, region, bucket, _ = _minio_test_config()
    _skip_unless_minio_writable(endpoint, access_key, secret_key, region, bucket)

    bad_path = f"s3://{bucket}/shuffle-cache-minio-bad-credentials/{uuid.uuid4()}/probe.parquet"
    conn = duckdb.connect()
    try:
        _configure_duckdb_s3(
            conn,
            endpoint,
            access_key + "-bad",
            secret_key + "-bad",
            region,
            http_retries=1,
        )
        try:
            conn.execute(f"COPY (SELECT 1 AS value) TO '{bad_path}' (FORMAT PARQUET)")
        except Exception as exc:
            message = str(exc).lower()
            assert any(
                token in message for token in ("403", "access", "forbid", "credential", "signature", "s3", "http")
            ), str(exc)
        else:
            pytest.skip("MinIO/S3 endpoint accepted intentionally bad credentials")
    finally:
        conn.close()


def test_object_store_httpfs_5xx_retries_then_hard_fails():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class UnavailableS3Handler(BaseHTTPRequestHandler):
        requests = []

        def _record_unavailable(self):
            type(self).requests.append((self.command, self.path))
            self.send_response(503)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"unavailable")

        def do_GET(self):
            self._record_unavailable()

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), UnavailableS3Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        conn = duckdb.connect()
        try:
            _configure_duckdb_s3(
                conn,
                endpoint,
                "access-key",
                "secret-key",
                "us-east-1",
                http_retries=2,
            )
            conn.execute("SET http_keep_alive=false")
            with pytest.raises(Exception) as excinfo:
                conn.execute("SELECT * FROM glob('s3://bucket/prefix/*')").fetchall()
        finally:
            conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    message = str(excinfo.value)
    assert "503" in message
    assert "HTTP" in message
    assert len(UnavailableS3Handler.requests) == 3
    assert {method for method, _ in UnavailableS3Handler.requests} == {"GET"}
    assert all("list-type=2" in path and "prefix=prefix%2F" in path for _, path in UnavailableS3Handler.requests)


def test_object_store_httpfs_timeout_retries_then_hard_fails():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class SlowListS3Handler(BaseHTTPRequestHandler):
        requests = []

        def do_GET(self):
            type(self).requests.append((self.command, self.path))
            time.sleep(2.0)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/xml")
                self.end_headers()
                self.wfile.write(
                    b'<?xml version="1.0" encoding="UTF-8"?>'
                    b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/" />'
                )
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowListS3Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        conn = duckdb.connect()
        try:
            _configure_duckdb_s3(
                conn,
                endpoint,
                "access-key",
                "secret-key",
                "us-east-1",
                http_retries=1,
                http_timeout=1,
            )
            conn.execute("SET http_keep_alive=false")
            conn.execute("SET http_retry_wait_ms=10")
            conn.execute("SET http_retry_backoff=1")
            with pytest.raises(Exception) as excinfo:
                conn.execute("SELECT * FROM glob('s3://bucket/prefix/*')").fetchall()
        finally:
            conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    message = str(excinfo.value).lower()
    assert "timeout" in message
    assert "http get" in message
    assert len(SlowListS3Handler.requests) == 2
    assert {method for method, _ in SlowListS3Handler.requests} == {"GET"}
    assert all("list-type=2" in path and "prefix=prefix%2F" in path for _, path in SlowListS3Handler.requests)


def test_object_store_httpfs_connection_close_retries_then_hard_fails():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class CloseConnectionS3Handler(BaseHTTPRequestHandler):
        requests = []

        def do_GET(self):
            type(self).requests.append((self.command, self.path))
            self.close_connection = True
            try:
                self.connection.shutdown(1)
            except OSError:
                pass
            self.connection.close()

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), CloseConnectionS3Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        conn = duckdb.connect()
        try:
            _configure_duckdb_s3(
                conn,
                endpoint,
                "access-key",
                "secret-key",
                "us-east-1",
                http_retries=1,
            )
            conn.execute("SET http_keep_alive=false")
            conn.execute("SET http_retry_wait_ms=10")
            conn.execute("SET http_retry_backoff=1")
            with pytest.raises(Exception) as excinfo:
                conn.execute("SELECT * FROM glob('s3://bucket/prefix/*')").fetchall()
        finally:
            conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    message = str(excinfo.value).lower()
    assert "server returned nothing" in message
    assert "http get" in message
    assert len(CloseConnectionS3Handler.requests) == 2
    assert {method for method, _ in CloseConnectionS3Handler.requests} == {"GET"}
    assert all("list-type=2" in path and "prefix=prefix%2F" in path for _, path in CloseConnectionS3Handler.requests)


def test_object_store_httpfs_partial_response_retries_then_hard_fails():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class PartialResponseS3Handler(BaseHTTPRequestHandler):
        requests = []

        def do_GET(self):
            type(self).requests.append((self.command, self.path))
            body = b'<?xml version="1.0"?><ListBucketResult>'
            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(body) + 100))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            self.close_connection = True

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), PartialResponseS3Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        conn = duckdb.connect()
        try:
            _configure_duckdb_s3(
                conn,
                endpoint,
                "access-key",
                "secret-key",
                "us-east-1",
                http_retries=1,
            )
            conn.execute("SET http_keep_alive=false")
            conn.execute("SET http_retry_wait_ms=10")
            conn.execute("SET http_retry_backoff=1")
            with pytest.raises(Exception) as excinfo:
                conn.execute("SELECT * FROM glob('s3://bucket/prefix/*')").fetchall()
        finally:
            conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    message = str(excinfo.value).lower()
    assert "partial file" in message
    assert "http get" in message
    assert len(PartialResponseS3Handler.requests) == 2
    assert {method for method, _ in PartialResponseS3Handler.requests} == {"GET"}
    assert all("list-type=2" in path and "prefix=prefix%2F" in path for _, path in PartialResponseS3Handler.requests)


def test_object_store_httpfs_real_proxy_tcp_reset_retries_then_hard_fails():
    message, proxy_requests, upstream_requests = _run_object_store_proxy_fault(
        "reset_after_upstream",
    )

    lower = message.lower()
    assert "http get" in lower
    assert any(
        token in lower
        for token in (
            "server returned nothing",
            "connection reset",
            "failure when receiving",
            "transfer closed",
        )
    )
    assert len(proxy_requests) == 2
    assert len(upstream_requests) == 2
    assert all(b"list-type=2" in request and b"prefix=prefix" in request for request in proxy_requests)
    assert all("list-type=2" in path and "prefix=prefix%2F" in path for _, path in upstream_requests)


def test_object_store_httpfs_real_proxy_partial_response_retries_then_hard_fails():
    message, proxy_requests, upstream_requests = _run_object_store_proxy_fault(
        "partial_after_upstream",
    )

    lower = message.lower()
    assert "http get" in lower
    assert any(
        token in lower
        for token in (
            "partial file",
            "transfer closed",
            "failure when receiving",
            "end of file",
        )
    )
    assert len(proxy_requests) == 2
    assert len(upstream_requests) == 2
    assert all(b"list-type=2" in request and b"prefix=prefix" in request for request in proxy_requests)
    assert all("list-type=2" in path and "prefix=prefix%2F" in path for _, path in upstream_requests)


def test_object_store_httpfs_real_proxy_slow_body_retries_then_hard_fails():
    message, proxy_requests, upstream_requests = _run_object_store_proxy_fault(
        "slow_body_after_upstream",
        http_timeout=0.5,
    )

    lower = message.lower()
    assert "http get" in lower
    assert any(
        token in lower
        for token in (
            "timeout",
            "timed out",
            "failure when receiving",
            "transfer closed",
        )
    )
    assert len(proxy_requests) == 2
    assert len(upstream_requests) == 2
    assert all(b"list-type=2" in request and b"prefix=prefix" in request for request in proxy_requests)
    assert all("list-type=2" in path and "prefix=prefix%2F" in path for _, path in upstream_requests)


@pytest.mark.external_service
def test_shuffle_cache_duckdb_filesystem_storage_minio_fault_matrix():
    endpoint, access_key, secret_key, region, bucket, base_uri = _minio_test_config()
    _skip_unless_minio_writable(endpoint, access_key, secret_key, region, bucket)

    result = duckdb.ray_cxx.shuffle_cache_duckdb_filesystem_storage_minio_fault_matrix_for_test(
        base_uri,
        endpoint,
        access_key,
        secret_key,
        region,
    )

    assert "not committed" in result["marker_missing_manifest_error"]
    assert "not committed" in result["marker_missing_source_error"]
    assert "file missing" in result["data_missing_manifest_error"]
    assert "file missing" in result["data_missing_source_error"]
    assert "size mismatch" in result["size_mismatch_manifest_error"]
    assert "size mismatch" in result["size_mismatch_source_error"]
    assert result["marker_missing_cleanup_removed"] >= 1
    assert result["data_missing_cleanup_removed"] >= 1
    assert result["size_mismatch_cleanup_removed"] >= 1


@pytest.mark.external_service
def test_flight_exchange_minio_selected_attempt_replay_and_loser_cleanup():
    endpoint, access_key, secret_key, region, bucket, base_uri = _minio_test_config()
    _skip_unless_minio_writable(endpoint, access_key, secret_key, region, bucket)

    result = duckdb.ray_cxx.flight_exchange_minio_selected_attempt_replay_for_test(
        base_uri,
        endpoint,
        access_key,
        secret_key,
        region,
    )

    assert result["handle_attempts"] == [1]
    assert result["handle_paths"] == [result["selected_output_location"]]
    assert result["lost_output_location"] != result["selected_output_location"]
    assert result["selected_values_before_cleanup"] == [801, 802]
    assert result["selected_values_after_loser_cleanup"] == [801, 802]
    assert "not committed" in result["lost_manifest_after_cleanup_error"]
    assert result["selected_cleanup_removed"] >= 1


def test_shuffle_cache_rejects_object_storage_local_dir_until_backend_exists():
    result = duckdb.ray_cxx.shuffle_cache_rejects_object_storage_local_dir_for_test()

    assert result["rejected"] is True
    assert "Object storage durable exchange backend is not implemented yet" in result["error"]


def test_shuffle_cache_duckdb_filesystem_storage_accepts_object_dir():
    result = duckdb.ray_cxx.shuffle_cache_duckdb_filesystem_storage_accepts_object_dir_for_test()

    assert result["accepted"] is True
    assert result["error"] == ""


def test_shuffle_cache_fake_object_no_rename_manifest_commit(tmp_path):
    result = duckdb.ray_cxx.shuffle_cache_fake_object_no_rename_manifest_for_test(str(tmp_path))

    assert result["committed_manifest"] is True
    assert result["manifest_exists"] is True
    assert result["marker_exists"] is True
    assert result["manifest_tmp_exists"] is False
    assert result["marker_tmp_exists"] is False
    assert result["manifest_file_count"] == 1
    assert result["manifest_total_rows"] == 1
    assert result["manifest_total_bytes"] == len("payload")
    assert result["text_puts"] == 2
    assert "attempt_id=7\n" in result["manifest"]


def test_flight_exchange_source_recovers_from_manifest_after_registry_loss(tmp_path):
    result = duckdb.ray_cxx.flight_exchange_source_manifest_recovery_for_test(str(tmp_path))

    assert result["values"] == [21, 22, 23, 24]
    assert result["finished"] is True
    assert result["registry_present"] is False


def test_flight_exchange_source_recovers_remote_writer_from_shared_manifest(tmp_path):
    result = duckdb.ray_cxx.flight_exchange_source_shared_manifest_recovery_for_test(str(tmp_path))

    assert result["values"] == [71, 72, 73]
    assert result["finished"] is True
    assert result["registry_present"] is False
    assert result["writer_node_id"] != result["reader_node_id"]


def _run_python_json(code: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _write_shared_manifest_in_subprocess(tmp_path) -> dict:
    writer_code = textwrap.dedent(
        f"""
        import json
        import duckdb

        result = dict(duckdb.ray_cxx.flight_exchange_source_write_shared_manifest_for_test({str(tmp_path)!r}))
        print(json.dumps(result), flush=True)
        """
    )
    return _run_python_json(writer_code)


def test_flight_exchange_source_recovers_shared_manifest_after_writer_process_exit(tmp_path):
    writer_result = _write_shared_manifest_in_subprocess(tmp_path)
    reader_code = textwrap.dedent(
        f"""
        import json
        import duckdb

        result = dict(duckdb.ray_cxx.flight_exchange_source_read_shared_manifest_for_test(
            {str(tmp_path)!r},
            {writer_result["output_location"]!r},
            {writer_result["writer_node_id"]!r},
            "reader-node",
            {int(writer_result["partition_id"])},
            {int(writer_result["attempt_id"])},
        ))
        result["values"] = list(result["values"])
        print(json.dumps(result), flush=True)
        """
    )
    reader_result = _run_python_json(reader_code)

    assert Path(writer_result["manifest_path"]).exists()
    assert Path(writer_result["committed_path"]).exists()
    assert reader_result["values"] == [81, 82, 83]
    assert reader_result["finished"] is True
    assert reader_result["registry_present"] is False
    assert reader_result["writer_node_id"] != reader_result["reader_node_id"]


def test_flight_server_recovers_shared_manifest_after_writer_process_exit(tmp_path):
    writer_result = _write_shared_manifest_in_subprocess(tmp_path)
    reader_code = textwrap.dedent(
        f"""
        import json
        import duckdb

        result = dict(duckdb.ray_cxx.flight_server_read_shared_manifest_for_test(
            {str(tmp_path)!r},
            {writer_result["output_location"]!r},
            {writer_result["writer_node_id"]!r},
            {int(writer_result["partition_id"])},
        ))
        result["values"] = list(result["values"])
        print(json.dumps(result), flush=True)
        """
    )
    reader_result = _run_python_json(reader_code)

    assert Path(writer_result["manifest_path"]).exists()
    assert Path(writer_result["committed_path"]).exists()
    assert reader_result["values"] == [81, 82, 83]
    assert reader_result["row_count"] == 3
    assert reader_result["registry_present"] is False


def test_remote_exchange_source_local_dirs_survive_serialization_for_manifest_recovery(tmp_path):
    result = duckdb.ray_cxx.remote_exchange_source_local_dirs_roundtrip_recovery_for_test(str(tmp_path))

    assert result["values"] == [41, 42]
    assert result["node_id"]
    assert result["registry_present"] is False


def test_flight_server_recovers_from_manifest_after_registry_loss(tmp_path):
    result = duckdb.ray_cxx.flight_server_manifest_recovery_for_test(str(tmp_path))

    assert result["values"] == [31, 32]
    assert result["row_count"] == 2
    assert result["registry_present"] is False


def test_flight_server_rejects_uncommitted_attempt(tmp_path):
    result = duckdb.ray_cxx.flight_server_uncommitted_attempt_rejected_for_test(str(tmp_path))

    assert result["partial_file_count"] == 1
    assert result["committed_manifest"] is False
    assert result["fetch_error"] is True
    assert "not committed" in result["error"]


def test_distributed_copy_finalize_preflights_missing_staging_files(tmp_path):
    result = duckdb.ray_cxx.distributed_copy_finalize_missing_staging_preflight_for_test(str(tmp_path))

    assert result["finalize_error"] is True
    assert "before moving any final output" in result["error"]
    assert result["first_staging_exists"] is True
    assert result["missing_staging_exists"] is False
    assert result["final_file_count"] == 0


def test_distributed_copy_finalize_commit_manifest_is_idempotent(tmp_path):
    result = duckdb.ray_cxx.distributed_copy_finalize_commit_manifest_idempotent_for_test(str(tmp_path))

    assert result["first_finalize_error"] is False, result["first_error"]
    assert result["second_finalize_error"] is False, result["second_error"]
    assert result["first_rows_copied"] == 3
    assert result["second_rows_copied"] == 3
    assert result["manifest_exists"] is True
    assert result["committed_exists"] is True
    assert result["staging_root_exists"] is False
    assert result["final_root_exists"] is True
    assert result["final_file_count"] == 2


def test_distributed_copy_finalize_replays_inprogress_manifest(tmp_path):
    result = duckdb.ray_cxx.distributed_copy_finalize_replays_inprogress_manifest_for_test(str(tmp_path))

    assert result["committed_before"] is False
    assert result["first_final_before"] is True
    assert result["second_final_before"] is False
    assert result["first_staging_before"] is False
    assert result["second_staging_before"] is True
    assert result["finalize_error"] is False, result["error"]
    assert result["idempotent_error"] is False, result["idempotent_error_message"]
    assert result["rows_copied"] == 3
    assert result["idempotent_rows_copied"] == 3
    assert result["manifest_exists"] is True
    assert result["committed_after"] is True
    assert result["staging_root_exists"] is False
    assert result["final_root_exists"] is True
    assert result["final_file_count"] == 2


def test_distributed_copy_direct_write_commit_manifest(tmp_path):
    result = duckdb.ray_cxx.distributed_copy_direct_write_commit_manifest_for_test(str(tmp_path))

    assert result["first_finalize_error"] is False, result["first_error"]
    assert result["second_finalize_error"] is False, result["second_error"]
    assert result["first_rows_copied"] == 3
    assert result["second_rows_copied"] == 3
    assert "_vane_direct_write_run-direct" in result["first_final_path"]
    assert "_vane_direct_write_run-direct" in result["second_final_path"]
    assert result["first_output_run_id"] == "run-direct"
    assert result["first_output_direct_write"] is True
    assert result["first_output_committed"] is True
    assert result["first_output_manifest_path"].endswith("copy_direct_final.duckdb_commit/run-direct/manifest.txt")
    assert result["first_output_committed_marker_path"].endswith("copy_direct_final.duckdb_commit/run-direct/committed")
    assert result["manifest_exists"] is True
    assert result["committed_exists"] is True
    assert result["direct_prefix_exists"] is True
    assert result["first_file_exists"] is True
    assert result["second_file_exists"] is True
    assert result["loser_file_exists"] is False
    assert result["replay_loser_file_exists"] is False


def test_distributed_copy_direct_target_visible_commit_manifest(tmp_path):
    result = duckdb.ray_cxx.distributed_copy_direct_target_visible_commit_for_test(str(tmp_path))

    assert result["first_finalize_error"] is False, result["first_error"]
    assert result["second_finalize_error"] is False, result["second_error"]
    assert result["read_committed_error"] is False, result["read_committed_error_message"]
    assert result["first_rows_copied"] == 3
    assert result["second_rows_copied"] == 3
    assert result["read_committed_rows_copied"] == 3
    assert result["first_output_run_id"] == "run-visible"
    assert result["first_output_direct_write"] is True
    assert result["first_output_committed"] is True
    assert result["manifest_exists"] is True
    assert result["committed_exists"] is True
    assert result["direct_prefix_exists"] is False
    assert result["first_file_exists"] is True
    assert result["second_file_exists"] is True
    assert result["loser_file_exists"] is False
    assert result["replay_loser_file_exists"] is False
    assert result["other_run_file_exists"] is True
    assert "/_vane_direct_write_" not in result["first_final_path"]
    assert Path(result["first_final_path"]).name.startswith("run-visible_")


def test_distributed_copy_direct_target_remote_path_for_test():
    result = duckdb.ray_cxx.distributed_copy_direct_target_remote_path_for_test(
        "s3://bucket/output",
        "run-visible",
        "w_worker",
        "part.parquet",
    )

    assert result["direct_target_file"] == "s3://bucket/output/run-visible_w_worker_part.parquet"
    assert "_vane_direct_write_" not in result["direct_target_file"]
    assert result["legacy_task_directory"] == "s3://bucket/output/_vane_direct_write_run-visible/w_worker"
    assert result["filename_pattern"] == "run-visible_w_worker_{i}"


def test_distributed_copy_sink_mode_local_default_uses_visible_direct_target(monkeypatch, tmp_path):
    monkeypatch.delenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", raising=False)

    result = duckdb.ray_cxx.distributed_copy_sink_mode_for_test(str(tmp_path / "out"))

    assert result["construct_error"] is False, result["error"]
    assert result["staging_root_base"] == ""
    assert result["uses_direct_write"] is True
    assert result["uses_visible_direct_target"] is True


def test_distributed_copy_sink_mode_local_staging_env_preserves_staging(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", "1")

    result = duckdb.ray_cxx.distributed_copy_sink_mode_for_test(str(tmp_path / "out"))

    assert result["construct_error"] is False, result["error"]
    assert result["staging_root_base"].endswith("out.duckdb_staging")
    assert result["uses_direct_write"] is False
    assert result["uses_visible_direct_target"] is False


def test_distributed_copy_sink_mode_remote_rejects_local_staging_env(monkeypatch):
    monkeypatch.setenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", "1")

    result = duckdb.ray_cxx.distributed_copy_sink_mode_for_test("s3://bucket/out")

    assert result["construct_error"] is True
    assert "VANE_DISTRIBUTED_COPY_LOCAL_STAGING" in result["error"]
    assert "remote" in result["error"].lower()


def test_distributed_copy_direct_write_local_invisible_file_can_commit(tmp_path):
    result = duckdb.ray_cxx.distributed_copy_direct_write_local_invisible_file_commit_for_test(str(tmp_path))

    assert result["finalize_error"] is False, result["error"]
    assert result["rows_copied"] == 4
    assert result["output_direct_write"] is True
    assert result["output_committed"] is True
    assert result["manifest_exists"] is True
    assert result["committed_exists"] is True
    assert result["invisible_file_exists"] is False


def test_distributed_copy_direct_write_committed_reader_requires_marker(tmp_path):
    result = duckdb.ray_cxx.distributed_copy_direct_write_committed_reader_for_test(str(tmp_path))

    assert result["manifest_exists"] is True
    assert result["marker_exists"] is True
    assert result["selected_file_exists"] is True
    assert result["loser_file_exists"] is True
    assert result["uncommitted_error"] is True
    assert "not committed" in result["uncommitted_error_message"]
    assert result["committed_error"] is False, result["committed_error_message"]
    assert result["committed_rows"] == 7
    assert result["committed_file_count"] == 1
    assert result["committed_file_path"].endswith("_vane_direct_write_run-reader/w_selected/part.parquet")
    assert result["committed_contains_loser"] is False

    committed = duckdb.ray_cxx.read_committed_copy_direct_write_result(result["base_path"], result["run_id"])
    assert committed["rows_copied"] == 7
    assert committed["copy_output_run_id"] == "run-reader"
    assert committed["copy_output_direct_write"] is True
    assert committed["copy_output_committed"] is True
    assert len(committed["files"]) == 1
    assert committed["files"][0]["final_path"].endswith("_vane_direct_write_run-reader/w_selected/part.parquet")


def test_distributed_copy_direct_write_uncommitted_stale_cleanup(tmp_path):
    result = duckdb.ray_cxx.distributed_copy_direct_write_uncommitted_stale_cleanup_for_test(str(tmp_path))

    assert result["stale_skipped_committed"] is False
    assert result["stale_data_run_dir_existed"] is True
    assert result["stale_data_run_dir_removed"] is True
    assert result["stale_commit_dir_existed"] is True
    assert result["stale_commit_dir_removed"] is True
    assert result["stale_run_dir_exists"] is False
    assert result["stale_file_exists"] is False
    assert result["stale_manifest_exists"] is False
    assert result["stale_commit_dir_exists"] is False

    assert result["committed_skipped_committed"] is True
    assert result["committed_data_run_dir_removed"] is False
    assert result["committed_commit_dir_removed"] is False
    assert result["committed_run_dir_exists"] is True
    assert result["committed_file_exists"] is True
    assert result["committed_manifest_exists"] is True
    assert result["committed_marker_exists"] is True
    assert result["committed_commit_dir_exists"] is True


def test_cleanup_uncommitted_copy_direct_write_run_public_api(tmp_path):
    base = tmp_path / "copy_direct_public_cleanup"
    stale_run_id = "run-public-stale"
    stale_run_dir = base / f"_vane_direct_write_{stale_run_id}" / "w_failed"
    stale_file = stale_run_dir / "part.parquet"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_bytes(b"stale")
    stale_commit_dir = Path(str(base) + ".duckdb_commit") / stale_run_id
    stale_commit_dir.mkdir(parents=True)
    (stale_commit_dir / "manifest.txt").write_text("partial\n")

    stale = duckdb.ray_cxx.cleanup_uncommitted_copy_direct_write_run(str(base), stale_run_id)
    assert stale["skipped_committed"] is False
    assert stale["data_run_dir_existed"] is True
    assert stale["data_run_dir_removed"] is True
    assert stale["commit_dir_existed"] is True
    assert stale["commit_dir_removed"] is True
    assert not stale_run_dir.exists()
    assert not stale_file.exists()
    assert not stale_commit_dir.exists()

    committed_run_id = "run-public-committed"
    committed_run_dir = base / f"_vane_direct_write_{committed_run_id}" / "w_selected"
    committed_file = committed_run_dir / "part.parquet"
    committed_file.parent.mkdir(parents=True)
    committed_file.write_bytes(b"committed")
    committed_commit_dir = Path(str(base) + ".duckdb_commit") / committed_run_id
    committed_commit_dir.mkdir(parents=True)
    (committed_commit_dir / "manifest.txt").write_text("committed manifest\n")
    (committed_commit_dir / "committed").write_text("committed\n")

    committed = duckdb.ray_cxx.cleanup_uncommitted_copy_direct_write_run(str(base), committed_run_id)
    assert committed["skipped_committed"] is True
    assert committed["data_run_dir_removed"] is False
    assert committed["commit_dir_removed"] is False
    assert committed_file.exists()
    assert committed_commit_dir.exists()
    assert (committed_commit_dir / "committed").exists()


def _register_direct_write_lifecycle_run(
    base: Path,
    run_id: str,
    *,
    created_epoch_ms: int,
    worker_dir: str,
    committed: bool = False,
):
    registered = duckdb.ray_cxx.register_copy_direct_write_run_lifecycle(
        str(base), run_id, created_epoch_ms=created_epoch_ms
    )
    data_file = base / f"_vane_direct_write_{run_id}" / worker_dir / "part.parquet"
    data_file.parent.mkdir(parents=True)
    data_file.write_bytes(run_id.encode())
    if committed:
        Path(registered["copy_output_committed_marker_path"]).write_text("committed\n")
    return registered, data_file


def test_cleanup_expired_copy_direct_write_runs_public_api(tmp_path):
    base = tmp_path / "copy_direct_lifecycle_cleanup"

    stale_run_id = "run-lifecycle-stale"
    stale, stale_file = _register_direct_write_lifecycle_run(
        base,
        stale_run_id,
        created_epoch_ms=1_000,
        worker_dir="w_failed",
    )

    active_run_id = "run-lifecycle-active"
    active, active_file = _register_direct_write_lifecycle_run(
        base,
        active_run_id,
        created_epoch_ms=9_000,
        worker_dir="w_running",
    )

    committed_run_id = "run-lifecycle-committed"
    committed, committed_file = _register_direct_write_lifecycle_run(
        base,
        committed_run_id,
        created_epoch_ms=1_000,
        worker_dir="w_selected",
        committed=True,
    )

    result = duckdb.ray_cxx.cleanup_expired_copy_direct_write_runs(str(base), min_age_ms=5_000, now_epoch_ms=10_000)

    assert result["scanned_runs"] == 3
    assert result["cleaned_runs"] == 1
    assert result["committed_runs"] == 1
    assert result["active_runs"] == 1
    assert result["skipped_unregistered_runs"] == 0
    assert result["errors"] == 0
    assert result["cleaned_run_ids"] == [stale_run_id]
    assert not stale_file.exists()
    assert not Path(stale["copy_output_commit_dir"]).exists()
    assert active_file.exists()
    assert Path(active["copy_output_lifecycle_path"]).exists()
    assert committed_file.exists()
    assert Path(committed["copy_output_lifecycle_path"]).exists()


def test_copy_direct_write_lifecycle_cleanup_once_public_api(tmp_path):
    from duckdb.runners.ray import cleanup_copy_direct_write_lifecycle_once

    base = tmp_path / "copy_direct_lifecycle_once"
    stale_run_id = "run-lifecycle-once-stale"
    stale, stale_file = _register_direct_write_lifecycle_run(
        base,
        stale_run_id,
        created_epoch_ms=1_000,
        worker_dir="w_failed",
    )
    active_run_id = "run-lifecycle-once-active"
    active, active_file = _register_direct_write_lifecycle_run(
        base,
        active_run_id,
        created_epoch_ms=9_000,
        worker_dir="w_running",
    )

    summary = cleanup_copy_direct_write_lifecycle_once(
        [str(base)],
        min_age_ms=5_000,
        now_epoch_ms=10_000,
    )

    assert summary["base_path_count"] == 1
    assert summary["scanned_runs"] == 2
    assert summary["cleaned_runs"] == 1
    assert summary["active_runs"] == 1
    assert summary["errors"] == 0
    assert summary["cleaned_run_ids"] == [{"base_path": str(base), "run_id": stale_run_id}]
    assert summary["scans"][0]["cleaned_run_ids"] == [stale_run_id]
    assert not stale_file.exists()
    assert not Path(stale["copy_output_commit_dir"]).exists()
    assert active_file.exists()
    assert Path(active["copy_output_lifecycle_path"]).exists()


def test_copy_direct_write_lifecycle_cleanup_loop_public_api(tmp_path):
    from duckdb.runners.ray import run_copy_direct_write_lifecycle_cleanup_loop

    base = tmp_path / "copy_direct_lifecycle_loop"
    stale_run_id = "run-lifecycle-loop-stale"
    stale, stale_file = _register_direct_write_lifecycle_run(
        base,
        stale_run_id,
        created_epoch_ms=1_000,
        worker_dir="w_failed",
    )
    seen = []

    result = run_copy_direct_write_lifecycle_cleanup_loop(
        str(base),
        min_age_ms=5_000,
        interval_seconds=0,
        max_iterations=2,
        now_epoch_ms_fn=lambda: 10_000,
        on_iteration=seen.append,
    )

    assert result["iterations"] == 2
    assert seen[0]["cleaned_runs"] == 1
    assert result["summaries"][0]["cleaned_run_ids"] == [{"base_path": str(base), "run_id": stale_run_id}]
    assert result["summaries"][1]["cleaned_runs"] == 0
    assert result["last_summary"]["errors"] == 0
    assert not stale_file.exists()
    assert not Path(stale["copy_output_commit_dir"]).exists()


def test_copy_direct_write_lifecycle_cleanup_cli_once(tmp_path):
    base = tmp_path / "copy_direct_lifecycle_cli"
    stale_run_id = "run-lifecycle-cli-stale"
    stale, stale_file = _register_direct_write_lifecycle_run(
        base,
        stale_run_id,
        created_epoch_ms=1_000,
        worker_dir="w_failed",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "duckdb.runners.ray.lifecycle",
            "--base-path",
            str(base),
            "--min-age-ms",
            "5000",
            "--now-epoch-ms",
            "10000",
            "--once",
            "--json",
        ],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["cleaned_runs"] == 1
    assert summary["errors"] == 0
    assert summary["cleaned_run_ids"] == [{"base_path": str(base), "run_id": stale_run_id}]
    assert not stale_file.exists()
    assert not Path(stale["copy_output_commit_dir"]).exists()


def test_fte_split_queue_cancel_wakes_as_canceled():
    queue = duckdb.ray_cxx.FteSplitQueue()

    queue.cancel()

    assert queue.try_get_next() == {"state": "CANCELED"}
    queue.add_scan_split(b"ignored")
    assert queue.buffered_splits() == 0


def test_fte_split_queue_wait_for_next_cancel_wakes_blocked_thread():
    split_queue = duckdb.ray_cxx.FteSplitQueue()
    results = queue.Queue()

    def wait_for_split():
        results.put(split_queue.wait_for_next())

    thread = threading.Thread(target=wait_for_split)
    thread.start()
    assert results.empty()

    split_queue.cancel()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert results.get_nowait() == {"state": "CANCELED"}
    assert split_queue.queue_wait_ms() >= 0


def test_execute_native_rejects_invalid_fte_exchange_source_queue_map():
    m = duckdb.ray_cxx
    con = duckdb.connect()
    runner = m.DistributedPhysicalPlanRunner()
    plan = _make_test_physical_plan(con)

    with pytest.raises(ValueError, match="fte_exchange_source_queues values must be FteSplitQueue"):
        runner.execute_native(
            con.cursor(),
            plan,
            fte_exchange_source_queues={"7": object()},
        )


def test_execute_native_rejects_invalid_fte_scan_source_queue_map():
    m = duckdb.ray_cxx
    con = duckdb.connect()
    runner = m.DistributedPhysicalPlanRunner()
    plan = _make_test_physical_plan(con)

    with pytest.raises(ValueError, match="fte_scan_source_queues values must be FteSplitQueue"):
        runner.execute_native(
            con.cursor(),
            plan,
            fte_scan_source_queues={"7": object()},
        )


def test_execute_native_fte_dynamic_scan_queue_reads_parquet_after_blocking(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("VANE_NATIVE_PROGRESS_INTERVAL_MS", "10")

    con = duckdb.connect()
    src = tmp_path / "dynamic_scan_input.parquet"
    con.execute(
        f"""
        COPY (
            SELECT i::BIGINT AS i
            FROM range(6) tbl(i)
        ) TO '{src}' (FORMAT PARQUET)
        """
    )
    relation = con.sql(f"SELECT sum(i) AS total FROM read_parquet('{src}')")
    plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)
    scan_task_descriptors = plan.scan_task_descriptor_map()
    assert len(scan_task_descriptors) == 1
    node_id, descriptors = next(iter(scan_task_descriptors.items()))
    assert len(descriptors) == 1
    assert isinstance(descriptors[0], bytes)

    split_queue = duckdb.ray_cxx.FteSplitQueue()
    runner = duckdb.ray_cxx.DistributedPhysicalPlanRunner()
    started = threading.Event()
    results = queue.Queue()
    progress = []

    def run_native():
        try:
            started.set()
            results.put(
                (
                    "ok",
                    runner.execute_native(
                        con.cursor(),
                        plan,
                        fte_scan_source_queues={str(node_id): split_queue},
                        native_progress_callback=progress.append,
                    ),
                )
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            results.put(("err", exc))

    thread = threading.Thread(target=run_native)
    thread.start()
    try:
        assert started.wait(timeout=2)
        thread.join(timeout=0.1)
        assert results.empty()

        deadline = time.monotonic() + 2
        while not progress and time.monotonic() < deadline:
            time.sleep(0.01)
        assert progress
        assert all(item["total_pipeline_tasks"] > 0 for item in progress)
        assert all(
            item["queued_pipeline_tasks"] + item["running_pipeline_tasks"] + item["completed_pipeline_tasks"]
            == item["total_pipeline_tasks"]
            for item in progress
        )

        split_queue.add_scan_split(bytes(descriptors[0]))
        split_queue.no_more_splits()
        thread.join(timeout=5)
        assert not thread.is_alive()
    finally:
        if thread.is_alive():
            split_queue.cancel()
            thread.join(timeout=5)

    status, payload = results.get_nowait()
    assert status == "ok", payload
    assert payload.completion_status == "ok"
    assert [meta.num_rows for meta in payload.partition_metadatas] == [1]
    assert payload.partition_payloads[0].column(0).to_pylist() == [15]
    assert payload.task_stats["submitted_split_count"] == 1
    assert payload.task_stats["consumed_split_count"] == 1
    assert payload.task_stats["completed_pipeline_tasks"] == payload.task_stats["total_pipeline_tasks"]
    assert payload.task_stats["submitted_split_bytes"] > 0
    assert payload.task_stats["consumed_split_bytes"] > 0
    assert payload.task_stats["queue_wait_ms"] >= 0


def test_execute_native_streaming_udf_emits_determinate_live_progress(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    monkeypatch.setenv("VANE_NATIVE_PROGRESS_INTERVAL_MS", "10")

    def slow_identity(table):
        time.sleep(0.02)
        return pa.table({"x": table.column(0)})

    con = duckdb.connect()
    source = tmp_path / "streaming_udf_progress.parquet"
    con.execute(
        f"COPY (SELECT i::BIGINT AS x FROM range(20000) tbl(i)) TO '{source}' (FORMAT PARQUET, ROW_GROUP_SIZE 2048)"
    )
    relation = con.read_parquet(str(source)).map_batches(
        slow_identity,
        schema={"x": duckdb.sqltypes.BIGINT},
        execution_backend="subprocess_task",
        batch_size=2048,
        streaming_breaker=True,
    )
    plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)
    progress = []

    result = duckdb.ray_cxx.DistributedPhysicalPlanRunner().execute_native(
        con.cursor(),
        plan,
        native_progress_callback=progress.append,
    )

    live_progress = [
        stats
        for stats in progress
        if stats.get("total_pipeline_tasks", 0) > 0
        and stats.get("completed_pipeline_tasks", stats["total_pipeline_tasks"]) < stats["total_pipeline_tasks"]
    ]
    assert live_progress
    assert all(
        stats["queued_pipeline_tasks"] + stats["running_pipeline_tasks"] + stats["completed_pipeline_tasks"]
        == stats["total_pipeline_tasks"]
        for stats in live_progress
    )
    assert any(stats["completed_pipeline_tasks"] > 0 for stats in live_progress)
    assert result.task_stats["completed_pipeline_tasks"] == result.task_stats["total_pipeline_tasks"]
    assert all(
        {"input_rows", "input_bytes", "output_rows", "output_bytes"}.issubset(pipeline)
        for pipeline in result.task_stats["pipelines"]
    )


def _make_two_file_dynamic_scan_plan(tmp_path):
    con = duckdb.connect()
    src_a = tmp_path / "clone_queue_a.parquet"
    src_b = tmp_path / "clone_queue_b.parquet"
    con.execute(
        f"""
        COPY (
            SELECT i::BIGINT AS i
            FROM range(0, 3) tbl(i)
        ) TO '{src_a}' (FORMAT PARQUET)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT i::BIGINT AS i
            FROM range(10, 13) tbl(i)
        ) TO '{src_b}' (FORMAT PARQUET)
        """
    )
    relation = con.sql(f"SELECT sum(i)::BIGINT AS total FROM read_parquet(['{src_a}', '{src_b}'])")
    plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)
    scan_task_descriptors = plan.scan_task_descriptor_map()
    assert len(scan_task_descriptors) == 1
    node_id, descriptors = next(iter(scan_task_descriptors.items()))
    assert len(descriptors) == 2
    return con, plan, str(node_id), descriptors


def test_distributed_physical_plan_clones_use_independent_fte_scan_queues(tmp_path):
    pytest.importorskip("pyarrow")

    con, plan, node_id, descriptors = _make_two_file_dynamic_scan_plan(tmp_path)
    worker_con_a = duckdb.connect()
    worker_con_b = duckdb.connect()
    plan_a = plan.clone(worker_con_a)
    plan_b = plan.clone(worker_con_b)
    queue_a = duckdb.ray_cxx.FteSplitQueue()
    queue_b = duckdb.ray_cxx.FteSplitQueue()
    results = queue.Queue()

    def run_attempt(label, worker_con, attempt_plan, split_queue):
        cursor = worker_con.cursor()
        try:
            result = duckdb.ray_cxx.DistributedPhysicalPlanRunner().execute_native(
                cursor,
                attempt_plan,
                fte_scan_source_queues={node_id: split_queue},
            )
            results.put((label, "ok", result.partition_payloads[0].column(0).to_pylist()))
        except BaseException as exc:  # pragma: no cover - surfaced below
            results.put((label, "err", exc))
        finally:
            cursor.close()

    thread_a = threading.Thread(target=run_attempt, args=("a", worker_con_a, plan_a, queue_a))
    thread_b = threading.Thread(target=run_attempt, args=("b", worker_con_b, plan_b, queue_b))
    thread_a.start()
    thread_b.start()
    try:
        time.sleep(0.1)
        assert thread_a.is_alive()
        assert thread_b.is_alive()
        assert results.empty()

        queue_b.add_scan_split(bytes(descriptors[1]))
        queue_b.no_more_splits()
        thread_b.join(timeout=5)
        assert not thread_b.is_alive()
        assert thread_a.is_alive()
        assert results.get_nowait() == ("b", "ok", [33])

        queue_a.add_scan_split(bytes(descriptors[0]))
        queue_a.no_more_splits()
        thread_a.join(timeout=5)
        assert not thread_a.is_alive()
        assert results.get_nowait() == ("a", "ok", [3])

        assert queue_a.consumed_splits() == 1
        assert queue_b.consumed_splits() == 1
    finally:
        if thread_a.is_alive():
            queue_a.cancel()
            thread_a.join(timeout=5)
        if thread_b.is_alive():
            queue_b.cancel()
            thread_b.join(timeout=5)
        worker_con_a.close()
        worker_con_b.close()
        con.close()


def test_distributed_physical_plan_clone_scan_queue_cancel_does_not_cancel_sibling(tmp_path):
    pytest.importorskip("pyarrow")

    con, plan, node_id, descriptors = _make_two_file_dynamic_scan_plan(tmp_path)
    worker_con_cancel = duckdb.connect()
    worker_con_ok = duckdb.connect()
    plan_cancel = plan.clone(worker_con_cancel)
    plan_ok = plan.clone(worker_con_ok)
    queue_cancel = duckdb.ray_cxx.FteSplitQueue()
    queue_ok = duckdb.ray_cxx.FteSplitQueue()
    results = queue.Queue()

    def run_attempt(label, worker_con, attempt_plan, split_queue):
        cursor = worker_con.cursor()
        try:
            result = duckdb.ray_cxx.DistributedPhysicalPlanRunner().execute_native(
                cursor,
                attempt_plan,
                fte_scan_source_queues={node_id: split_queue},
            )
            values = result.partition_payloads[0].column(0).to_pylist() if result.partition_payloads else []
            results.put((label, "ok", values))
        except BaseException as exc:  # pragma: no cover - surfaced below
            results.put((label, "err", exc))
        finally:
            cursor.close()

    thread_cancel = threading.Thread(
        target=run_attempt,
        args=("cancel", worker_con_cancel, plan_cancel, queue_cancel),
    )
    thread_ok = threading.Thread(target=run_attempt, args=("ok", worker_con_ok, plan_ok, queue_ok))
    thread_cancel.start()
    thread_ok.start()
    try:
        time.sleep(0.1)
        assert thread_cancel.is_alive()
        assert thread_ok.is_alive()
        assert results.empty()

        queue_cancel.cancel()
        queue_ok.add_scan_split(bytes(descriptors[1]))
        queue_ok.no_more_splits()

        thread_cancel.join(timeout=5)
        thread_ok.join(timeout=5)
        assert not thread_cancel.is_alive()
        assert not thread_ok.is_alive()

        by_label = {}
        while not results.empty():
            label, status, payload = results.get_nowait()
            by_label[label] = (status, payload)

        assert by_label["ok"] == ("ok", [33])
        assert "cancel" in by_label
        assert queue_cancel.consumed_splits() == 0
        assert queue_ok.consumed_splits() == 1
    finally:
        if thread_cancel.is_alive():
            queue_cancel.cancel()
            thread_cancel.join(timeout=5)
        if thread_ok.is_alive():
            queue_ok.cancel()
            thread_ok.join(timeout=5)
        worker_con_cancel.close()
        worker_con_ok.close()
        con.close()


def test_ray_worker_manager_integration(monkeypatch):
    class DummyRayWorkerHandle:
        def __init__(self):
            self.fte_drop_query_calls = []

        def fte_drop_query(self, query_id):
            self.fte_drop_query_calls.append(query_id)
            return {
                "tasks_removed": 1,
                "tasks_canceled": 0,
                "fragments_removed": 1,
            }

        def stats_fragments(self):
            return {
                "registered_total": 1,
                "existing_total": 2,
                "lookup_hits": 3,
            }

        def shutdown(self):
            pass

    dummy_worker_handle = DummyRayWorkerHandle()

    def start_ray_workers(_existing_ids):
        return [duckdb.ray_cxx.RayWorkerRuntime("worker-1", dummy_worker_handle, 1.0, 0.0, 1024)]

    autoscale_called = {}

    def try_autoscale(_bundles):
        autoscale_called["called"] = True

    import duckdb.runners.ray.worker_handle as ray_worker_handle

    monkeypatch.setattr(ray_worker_handle, "start_ray_workers", start_ray_workers)
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", try_autoscale)

    mgr = duckdb.ray_cxx.RayWorkerManager()

    snaps = mgr.worker_snapshots()
    assert isinstance(snaps, list)
    assert len(snaps) >= 1

    # Try autoscale - should call our try_autoscale
    mgr.try_autoscale([{"CPU": 100, "GPU": 0, "memory": 0}])
    assert autoscale_called.get("called", False) is True

    stats = mgr.fragment_stats()
    assert stats["workers"]["worker-1"]["registered_total"] == 1
    assert stats["workers"]["worker-1"]["existing_total"] == 2
    assert stats["workers"]["worker-1"]["lookup_hits"] == 3
    assert stats["totals"]["registered_total"] == 1
    assert stats["totals"]["existing_total"] == 2
    assert stats["totals"]["lookup_hits"] == 3

    mgr.drop_query_fragments("query-lifecycle")
    assert dummy_worker_handle.fte_drop_query_calls == ["query-lifecycle"]


def test_ray_worker_manager_drop_is_best_effort_across_worker_failures(monkeypatch):
    calls = []

    class DummyRayWorkerHandle:
        def __init__(self, worker_id, *, fail):
            self.worker_id = worker_id
            self.fail = fail

        def fte_drop_query(self, query_id):
            calls.append((self.worker_id, query_id))
            if self.fail:
                raise RuntimeError(f"{self.worker_id} is dead")
            return {
                "tasks_removed": 1,
                "tasks_canceled": 0,
                "fragments_removed": 1,
            }

        def shutdown(self):
            pass

    def start_ray_workers(_existing_ids):
        return [
            duckdb.ray_cxx.RayWorkerRuntime(
                "worker-dead",
                DummyRayWorkerHandle("worker-dead", fail=True),
                1.0,
                0.0,
                1024,
            ),
            duckdb.ray_cxx.RayWorkerRuntime(
                "worker-live",
                DummyRayWorkerHandle("worker-live", fail=True),
                1.0,
                0.0,
                1024,
            ),
        ]

    import duckdb.runners.ray.worker_handle as ray_worker_handle

    monkeypatch.setattr(ray_worker_handle, "start_ray_workers", start_ray_workers)
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)
    manager = duckdb.ray_cxx.RayWorkerManager()
    assert len(manager.worker_snapshots()) == 2

    with pytest.raises(Exception, match="is dead"):
        manager.drop_query_fragments("query-best-effort-drop")

    assert sorted(calls) == [
        ("worker-dead", "query-best-effort-drop"),
        ("worker-live", "query-best-effort-drop"),
    ]


def test_ray_worker_manager_drop_fans_out_after_result_payload_release_failure(monkeypatch):
    from types import SimpleNamespace

    query_id = "query-result-release-failure"
    drop_calls = []

    class FailingResultHandle:
        worker_id = "worker-with-result"
        task_context_info = {
            "query_idx": 1,
            "last_node_id": 2,
            "task_id": 3,
            "node_ids": [2],
        }
        task_id = SimpleNamespace(
            query_id=query_id,
            fragment_execution_id=0,
            partition_id=0,
            attempt_id=0,
        )

        def release_result_payload(self):
            raise RuntimeError("result payload release failed")

    class DummyRayWorkerHandle:
        def __init__(self, worker_id, result_handles):
            self.worker_id = worker_id
            self.result_handles = list(result_handles)

        def fte_query_status(self, _query_id):
            return {"failed": False, "finished": False}

        def pop_fte_result_handles(self, _query_id):
            handles = self.result_handles
            self.result_handles = []
            return handles

        def fte_drop_query(self, actual_query_id):
            drop_calls.append((self.worker_id, actual_query_id))
            return {
                "tasks_removed": 0,
                "tasks_canceled": 0,
                "fragments_removed": 0,
            }

        def shutdown(self):
            pass

    def start_ray_workers(_existing_ids):
        return [
            duckdb.ray_cxx.RayWorkerRuntime(
                "worker-with-result",
                DummyRayWorkerHandle("worker-with-result", [FailingResultHandle()]),
                1.0,
                0.0,
                1024,
            ),
            duckdb.ray_cxx.RayWorkerRuntime(
                "worker-other",
                DummyRayWorkerHandle("worker-other", []),
                1.0,
                0.0,
                1024,
            ),
        ]

    import duckdb.runners.ray.worker_handle as ray_worker_handle

    monkeypatch.setattr(ray_worker_handle, "start_ray_workers", start_ray_workers)
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)
    manager = duckdb.ray_cxx.RayWorkerManager()
    assert len(manager.worker_snapshots()) == 2

    with pytest.raises(Exception, match="timed out waiting for FTE query"):
        manager.wait_fte_query(query_id, 1e-9)
    with pytest.raises(Exception, match="result payload release failed"):
        manager.drop_query_fragments(query_id)

    assert sorted(drop_calls) == [
        ("worker-other", query_id),
        ("worker-with-result", query_id),
    ]


def test_ray_worker_manager_worker_snapshots_fail_fast(monkeypatch):
    def start_ray_workers(_existing_ids):
        raise RuntimeError("start-ray-workers boom")

    def try_autoscale(_bundles):
        return None

    import duckdb.runners.ray.worker_handle as ray_worker_handle

    monkeypatch.setattr(ray_worker_handle, "start_ray_workers", start_ray_workers)
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", try_autoscale)

    mgr = duckdb.ray_cxx.RayWorkerManager()
    with pytest.raises(Exception, match="start-ray-workers boom"):
        mgr.worker_snapshots()


def test_ray_worker_manager_try_autoscale_fail_fast(monkeypatch):
    def start_ray_workers(_existing_ids):
        return []

    def try_autoscale(_bundles):
        raise RuntimeError("autoscale boom")

    import duckdb.runners.ray.worker_handle as ray_worker_handle

    monkeypatch.setattr(ray_worker_handle, "start_ray_workers", start_ray_workers)
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", try_autoscale)

    mgr = duckdb.ray_cxx.RayWorkerManager()
    with pytest.raises(Exception, match="autoscale boom"):
        mgr.try_autoscale([{"CPU": 100, "GPU": 0, "memory": 0}])


def test_execute_native_roundtrip_hash_join_plan_no_crash():
    code = textwrap.dedent(
        """
        from __future__ import annotations

        import gc
        import uuid

        import duckdb
        import ray.cloudpickle as cp
        con = duckdb.connect()
        con.execute("CREATE TABLE a AS SELECT i FROM range(1000) tbl(i)")
        con.execute("CREATE TABLE b AS SELECT i AS j FROM range(1000) tbl(i)")

        sql = "SELECT count(*) FROM a JOIN b ON a.i = b.j"
        relation = con.sql(sql)
        plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            str(uuid.uuid4()),
        ).to_physical_plan(con)
        roundtrip_plan = cp.loads(cp.dumps(plan))

        runner = duckdb.ray_cxx.DistributedPhysicalPlanRunner()
        cursor = con.cursor()
        result = runner.execute_native(cursor, roundtrip_plan, None, None)
        metadatas = list(getattr(result, "partition_metadatas", []))
        print("rows", metadatas[0].num_rows if metadatas else "na", flush=True)
        print("status", getattr(result, "completion_status", "na"), flush=True)

        cursor.close()
        del result, runner, roundtrip_plan, plan, cursor
        con.close()
        gc.collect()
        print("ok", flush=True)
        """
    )
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "rows 1" in proc.stdout
    assert "status ok" in proc.stdout
    assert "ok" in proc.stdout
