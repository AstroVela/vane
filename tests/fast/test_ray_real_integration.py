# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import threading
import time

import pytest

try:
    import ray
except Exception:
    ray = None

import vane


def _sql_string_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def _start_s3_object_server(payload):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class S3ObjectHandler(BaseHTTPRequestHandler):
        requests = []
        requests_lock = threading.Lock()

        def _record_request(self):
            with type(self).requests_lock:
                type(self).requests.append(
                    {
                        "authorization": self.headers.get("Authorization"),
                        "command": self.command,
                        "path": self.path,
                        "range": self.headers.get("Range"),
                    }
                )

        def _send_object(self, include_body):
            self._record_request()
            if self.path.split("?", 1)[0] != "/bucket/object.bin":
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            status = 200
            body = payload
            content_range = None
            range_header = self.headers.get("Range")
            if range_header:
                unit, separator, bounds = range_header.partition("=")
                start_text, dash, end_text = bounds.partition("-")
                if (
                    unit != "bytes"
                    or not separator
                    or not dash
                    or not start_text.isdigit()
                    or (end_text and not end_text.isdigit())
                ):
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{len(payload)}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start = int(start_text)
                end = len(payload) - 1 if not end_text else min(int(end_text), len(payload) - 1)
                if start >= len(payload) or end < start:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{len(payload)}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                status = 206
                body = payload[start : end + 1]
                content_range = f"bytes {start}-{end}/{len(payload)}"

            self.send_response(status)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "image/png")
            self.send_header("ETag", '"trusted-file-etag"')
            self.send_header("Last-Modified", "Thu, 27 Aug 2026 00:00:00 GMT")
            if content_range is not None:
                self.send_header("Content-Range", content_range)
            self.end_headers()
            if include_body:
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_HEAD(self):
            self._send_object(False)

        def do_GET(self):
            self._send_object(True)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), S3ObjectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, S3ObjectHandler


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
def test_default_ray_resolves_file_io_in_worker_process(monkeypatch, tmp_path):
    import fcntl
    import os

    payload = b"driver-only-payload"
    payload_path = tmp_path / "driver-only.bin"
    payload_path.write_bytes(payload)

    connection = vane.connect()
    try:
        with payload_path.open("rb") as source:
            descriptor = fcntl.fcntl(source.fileno(), fcntl.F_DUPFD_CLOEXEC, 512)
            try:
                driver_path = f"/proc/self/fd/{descriptor}"
                assert connection.execute(f"SELECT file_size(try_to_file('{driver_path}'))").fetchone() == (
                    len(payload),
                )

                monkeypatch.delenv("VANE_RUNNER", raising=False)
                vane.teardown_runner()
                relation = connection.sql(f"SELECT file_size(try_to_file('{driver_path}')) FROM range(2)")

                assert relation.fetchall() == [(None,), (None,)]
            finally:
                os.close(descriptor)
    finally:
        connection.close()


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_default_ray_reads_worker_visible_file_with_strict_range(monkeypatch, tmp_path):
    payload = b"worker-visible-payload"
    payload_path = tmp_path / "worker-visible.bin"
    payload_path.write_bytes(payload)
    path_sql = _sql_string_literal(payload_path)

    monkeypatch.delenv("VANE_RUNNER", raising=False)
    vane.teardown_runner()
    connection = vane.connect()
    try:
        relation = connection.sql(
            f"""
            SELECT
                file_size(to_file({path_sql})),
                file_size(file({path_sql}, NULL, 7, 7, NULL)),
                file_exists(file({path_sql}, NULL, 7, 7, NULL)),
                file_content_id(file_enrich(
                    file({path_sql}, NULL, 7, 7, NULL),
                    ['checksum']
                ))
            FROM range(2)
            """
        )
        digest = hashlib.sha256(payload[7:14]).hexdigest()
        expected = (
            len(payload),
            7,
            True,
            f"file-content-v1:checksum:sha256:{digest}",
        )
        assert relation.fetchall() == [expected, expected]
    finally:
        connection.close()


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_default_ray_file_io_uses_connection_session_s3_context(monkeypatch):
    payload = b"\x89PNG\r\n\x1a\n" + b"ray-session-file-payload"
    server, thread, handler = _start_s3_object_server(payload)
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    http_url_sql = _sql_string_literal(f"{endpoint}/bucket/object.bin")
    access_key = "trusted-file-access-key"
    environment_keys = (
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
    )
    try:
        monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", access_key)
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "trusted-file-secret-key")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.delenv("VANE_RUNNER", raising=False)
        vane.teardown_runner()
        connection = vane.connect()
        for key in environment_keys:
            monkeypatch.delenv(key)

        try:
            connection.execute("SET http_proxy=''")
            relation = connection.sql(
                f"""
                SELECT
                    file_size(to_file({http_url_sql})),
                    file_size(to_file('s3://bucket/object.bin')),
                    file_mime_type(
                        file('s3://bucket/object.bin', NULL, NULL, NULL, NULL),
                        'content'
                    )
                FROM range(2)
                """
            )
            expected = (len(payload), len(payload), "image/png")
            assert relation.fetchall() == [expected, expected]
        finally:
            connection.close()

        with handler.requests_lock:
            requests = list(handler.requests)
        assert requests
        assert any(
            request["authorization"] and f"Credential={access_key}/" in request["authorization"] for request in requests
        )
        assert any(request["range"] for request in requests)
    finally:
        try:
            vane.teardown_runner()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_run_simple_plan_on_ray_local():
    from vane import runners as _runners

    _runners.set_runner_ray(noop_if_initialized=True)
    runner = _runners.get_or_create_runner()
    assert getattr(runner, "name", None) == "ray"

    relation = vane.sql("SELECT a, b, a + b AS sum FROM (VALUES (1, 10), (2, 20), (3, 30)) AS t(a, b)")
    parts = list(runner.run_iter_tables(relation))
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

    parts = list(runner.run_iter_tables(relation))
    assert parts

    rows = _collect_rows_from_parts(parts)
    assert len(rows) == n

    expected_rows = {(x, x * 10, x + x * 10) for x in range(n)}
    assert set(rows) == expected_rows

    client = runner.query_driver_client
    assert client is not None
    assert ray.get(client.runner.ping.remote(client._owner_id)) is True


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_two_connections_share_job_runtime_and_close_independently(monkeypatch, tmp_path):
    import os

    import pyarrow as pa

    from vane import runners as _runners
    from vane.runners.ray.driver import RayQueryDriverClient

    _runners.set_runner_ray(noop_if_initialized=True)
    runner = _runners.get_or_create_runner()

    session_secret_key = "AWS_ISSUE75_SESSION_SECRET"
    path = tmp_path / "ray_two_sessions.parquet"
    vane.sql(f"COPY (SELECT i::INTEGER AS value FROM range(8) AS t(i)) TO '{path}' (FORMAT PARQUET)")

    monkeypatch.setenv(session_secret_key, "connection-a")
    connection_a = vane.connect()
    monkeypatch.setenv(session_secret_key, "connection-b")
    connection_b = vane.connect()
    monkeypatch.delenv(session_secret_key)

    def read_session_secret(table):
        secret = os.environ.get(session_secret_key)
        return pa.table({"secret": [secret] * table.num_rows})

    relation_a = connection_a.sql(f"SELECT * FROM read_parquet('{path}')").map_batches(
        read_session_secret,
        schema={"secret": vane.sqltypes.VARCHAR},
        execution_backend="ray_task",
    )
    relation_b = connection_b.sql(f"SELECT * FROM read_parquet('{path}')").map_batches(
        read_session_secret,
        schema={"secret": vane.sqltypes.VARCHAR},
        execution_backend="ray_task",
    )

    assert set(_collect_rows_from_parts(runner.run_iter_tables(relation_a))) == {("connection-a",)}
    assert set(_collect_rows_from_parts(runner.run_iter_tables(relation_b))) == {("connection-b",)}

    runtime_client = runner.query_driver_client
    assert runtime_client is not None
    peer_client = RayQueryDriverClient()
    try:
        assert runtime_client.runner._actor_id == peer_client.runner._actor_id
    finally:
        peer_client.close()

    connection_a.close()
    relation_b_after_close = connection_b.sql(f"SELECT * FROM read_parquet('{path}')").map_batches(
        read_session_secret,
        schema={"secret": vane.sqltypes.VARCHAR},
        execution_backend="ray_task",
    )
    assert set(_collect_rows_from_parts(runner.run_iter_tables(relation_b_after_close))) == {("connection-b",)}
    connection_b.close()


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_crashed_runtime_client_is_reclaimed_while_peer_survives(monkeypatch):
    from vane.runners.ray.driver import (
        RayQueryDriverClient,
        _collect_vane_env_overrides,
    )

    lease_env = {
        "VANE_RAY_CLIENT_LEASE_TIMEOUT_S": "1.0",
        "VANE_RAY_CLIENT_HEARTBEAT_INTERVAL_S": "0.2",
    }
    for key, value in lease_env.items():
        monkeypatch.setenv(key, value)

    peer = RayQueryDriverClient()
    runtime_config = _collect_vane_env_overrides()

    @ray.remote
    class _CrashOwner:
        def __init__(self, runner, config):
            import threading
            import uuid

            self.runner = runner
            self.owner_id = uuid.uuid4().hex
            self.lease_token = uuid.uuid4().hex
            ray.get(
                self.runner.attach_client.remote(
                    self.owner_id,
                    config,
                    self.lease_token,
                )
            )
            self.stop = threading.Event()

            def _heartbeat():
                while not self.stop.wait(0.2):
                    ray.get(
                        self.runner.heartbeat_client.remote(
                            self.owner_id,
                            self.lease_token,
                        )
                    )

            self.heartbeat_thread = threading.Thread(
                target=_heartbeat,
                daemon=True,
            )
            self.heartbeat_thread.start()

        def open_session(self, session_id):
            ray.get(
                self.runner.open_session.remote(
                    self.owner_id,
                    session_id,
                    {},
                )
            )
            ray.get(
                self.runner.heartbeat_client.remote(
                    self.owner_id,
                    self.lease_token,
                )
            )
            return {
                "owner_id": self.owner_id,
                "actor_id": str(self.runner._actor_id),
            }

    crashed_session_id = "crashed-runtime-client-session"
    crash_owner = _CrashOwner.remote(peer.runner, runtime_config)
    try:
        crashed = ray.get(crash_owner.open_session.remote(crashed_session_id))
        assert crashed["actor_id"] == str(peer.runner._actor_id)
        before = ray.get(peer.runner.runtime_lifecycle_snapshot.remote(peer._owner_id))
        assert before["client_count"] == 2
        assert before["session_count"] == 1

        ray.kill(crash_owner, no_restart=True)

        deadline = time.monotonic() + 10.0
        snapshot = before
        while time.monotonic() < deadline:
            snapshot = ray.get(peer.runner.runtime_lifecycle_snapshot.remote(peer._owner_id))
            if snapshot["client_count"] == 1 and snapshot["session_count"] == 0:
                break
            time.sleep(0.1)

        assert snapshot["client_count"] == 1
        assert snapshot["session_count"] == 0
        assert snapshot["plan_count"] == 0
        assert snapshot["cleanup_errors"] == 0

        survivor_session_id = "surviving-runtime-client-session"
        assert ray.get(
            peer.runner.open_session.remote(
                peer._owner_id,
                survivor_session_id,
                {},
            )
        )
        ray.get(
            peer.runner.close_session.remote(
                peer._owner_id,
                survivor_session_id,
            )
        )
    finally:
        peer.close()


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_relation_result_consumers_on_ray_local(tmp_path, monkeypatch):
    from vane import runners

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    connection = vane.connect()
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

    monkeypatch.setenv("VANE_RUNNER", "ray")
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
    with pytest.raises(vane.InvalidInputException, match="result closed"):
        partial_relation.fetchall()


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_lossless_relation_result_types_on_ray_local(monkeypatch):
    from vane import runners

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    connection = vane.connect()
    connection.execute("SET arrow_lossless_conversion = true")
    connection.execute("SET TimeZone = 'America/New_York'")
    monkeypatch.setenv("VANE_RUNNER", "ray")
    runners.set_runner_ray(noop_if_initialized=True)

    row = connection.sql("""
        SELECT
            1::HUGEINT AS huge_value,
            1::UHUGEINT AS uhuge_value,
            '00112233-4455-6677-8899-aabbccddeeff'::UUID AS uuid_value,
            '10101'::BIT AS bit_value,
            '12:34:56+02:00'::TIMETZ AS time_value,
            TIMESTAMPTZ '2024-01-01 12:00:00+00' AS timestamp_tz_value,
            '{"key": 1}'::JSON AS json_value,
            union_value(v := 'distributed'::VARCHAR) AS union_value
    """).fetchone()

    assert row is not None
    assert row[0] == 1
    assert row[1] == 1
    assert str(row[2]) == "00112233-4455-6677-8899-aabbccddeeff"
    assert row[3] == "10101"
    assert row[4].utcoffset().total_seconds() == 2 * 60 * 60
    assert row[5].isoformat() == "2024-01-01T07:00:00-05:00"
    assert row[6] == '{"key": 1}'
    assert row[7] == "distributed"


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_complex_relation_result_consumers_on_ray_local(tmp_path, monkeypatch):
    from vane import runners

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    connection = vane.connect()
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

    monkeypatch.setenv("VANE_RUNNER", "ray")
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
