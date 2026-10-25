# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""M5a native frozen-snapshot transport against an explicitly provisioned REST service."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen

import pytest
from gravitino_iceberg_helpers import iceberg_tables
from iceberg_feature_helpers import populate_iceberg_features

import vane

pytestmark = pytest.mark.external_service


@pytest.fixture(scope="module")
def frozen_tables():
    endpoint = os.environ.get("VANE_TEST_GRAVITINO_ICEBERG_URL")
    shared = os.environ.get("VANE_TEST_GRAVITINO_ICEBERG_SHARED_DIR")
    if not endpoint and not shared:
        pytest.skip("set VANE_TEST_GRAVITINO_ICEBERG_URL and VANE_TEST_GRAVITINO_ICEBERG_SHARED_DIR")
    if not endpoint or not shared:
        pytest.fail("Iceberg qualification requires both the REST URL and shared directory")
    with iceberg_tables(endpoint.rstrip("/"), Path(shared)) as tables:
        features = populate_iceberg_features(tables)
        create_transport_boundaries(tables)
        yield endpoint.rstrip("/"), tables, features


def create_transport_boundaries(tables):
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField, StringType, StructType

    for name, version, schema in (
        ("format_v1", "1", Schema(NestedField(1, "id", LongType(), required=True))),
        (
            "nested_schema",
            "2",
            Schema(
                NestedField(1, "id", LongType(), required=True),
                NestedField(2, "payload", StructType(NestedField(3, "value", StringType()))),
            ),
        ),
    ):
        directory = tables.directory / name
        for path in (directory, directory / "metadata", directory / "data"):
            path.mkdir(mode=0o777)
            path.chmod(0o777)
        table = tables.catalog.create_table(
            (tables.namespace, name), schema=schema, location=directory.as_uri(), properties={"format-version": version}
        )
        assert table.metadata.format_version == int(version)


@pytest.fixture(autouse=True)
def native_only(monkeypatch):
    import ray

    monkeypatch.setenv("VANE_RUNNER", "local")
    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "1")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "1")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("frozen Iceberg transport qualification must not initialize Ray")

    monkeypatch.setattr(ray, "init", forbidden)


def connection(tables):
    return vane.connect(
        config={
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
            "allow_persistent_secrets": "false",
            "secret_directory": str(tables.directory / "secrets"),
        }
    )


def attach(con, endpoint):
    vane.load_installed_extension("iceberg", connection=con)
    con.execute(
        "ATTACH '' AS lake (TYPE iceberg, ENDPOINT $endpoint, AUTHORIZATION_TYPE 'none', READ_ONLY)",
        {"endpoint": endpoint},
    )


def freeze(con, sql):
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(con.sql(sql), "frozen-iceberg-" + uuid.uuid4().hex)
    return pickle.dumps(logical)


def execute_frozen(payload, tables, *, split_batch=None):
    """Use the existing native worker entry point with an explicit whole-scan assignment."""
    with connection(tables) as worker:
        logical = pickle.loads(payload)
        physical = logical.to_physical_plan(worker)
        batch_map = physical.scan_split_batch_map()
        assert len(batch_map) == 1
        assert all(len(batches) == 1 for batches in batch_map.values())
        assignment = {
            str(node): bytes(batches[0]) if split_batch is None else split_batch for node, batches in batch_map.items()
        }
        runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
        try:
            worker_plan = physical.clone(worker)
            result = runner.execute_native(worker, worker_plan, scan_split_batch=assignment)
            assert result.completion_status in {"ok", "empty"}
            rows = []
            for part in result.partition_payloads:
                table = part.to_arrow() if hasattr(part, "to_arrow") else part
                rows.extend(zip(*(column.to_pylist() for column in table.columns), strict=True))
            schema = result.result_schema
            result = worker_plan = physical = logical = None
            return rows, schema
        finally:
            runner.shutdown()


class RestForwarder:
    """Forward to the real service, then close the exact endpoint captured during binding."""

    def __init__(self, endpoint):
        # Binding can hold the GIL while fetching metadata. The forwarding
        # server must have its own interpreter, not a thread in this process.
        self.process = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).parents[1] / "gravitino_iceberg_proxy.py"), endpoint],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.stopped = False
        timeout = threading.Timer(10, self.process.kill)
        timeout.start()
        try:
            line = self.process.stdout.readline()
            if not line:
                raise AssertionError("REST forwarder did not start: " + self.process.stderr.read())
            self.endpoint = json.loads(line)["endpoint"]
        except BaseException:
            self.stop()
            raise
        finally:
            timeout.cancel()
            timeout.join()

    def stop(self):
        if not self.stopped:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.communicate(timeout=5)
                raise AssertionError("REST forwarder failed to stop") from None
            self.stopped = True


@contextmanager
def unavailable_metadata(tables):
    moved = []
    try:
        for metadata in tables.directory.glob("*/metadata"):
            hidden = metadata.with_name("frozen-test-hidden-metadata")
            metadata.rename(hidden)
            moved.append((metadata, hidden))
        assert moved
        yield
    finally:
        for metadata, hidden in reversed(moved):
            hidden.rename(metadata)


@pytest.mark.parametrize("query", ["rows", "projection", "filter", "aggregate", "empty"])
def test_frozen_plan_executes_after_connection_and_rest_endpoint_close(frozen_tables, query):
    endpoint, tables, _features = frozen_tables
    namespace = f'lake."{tables.namespace}"'
    statements = {
        "rows": (f"SELECT id, label FROM {namespace}.items", list(tables.rows)),
        "projection": (f"SELECT label FROM {namespace}.items", [(label,) for _, label in tables.rows]),
        "filter": (
            f"SELECT id, label FROM {namespace}.items WHERE id BETWEEN 2050 AND 2053",
            [row for row in tables.rows if 2050 <= row[0] <= 2053],
        ),
        "aggregate": (
            f"SELECT count(*), sum(id) FROM {namespace}.items WHERE id % 11 = 0",
            [
                (
                    len([key for key, _ in tables.rows if key % 11 == 0]),
                    sum(key for key, _ in tables.rows if key % 11 == 0),
                )
            ],
        ),
        "empty": (f"SELECT id, label FROM {namespace}.empty", []),
    }
    sql, expected = statements[query]
    proxy = RestForwarder(endpoint)
    try:
        with connection(tables) as con:
            attach(con, proxy.endpoint)
            payload = freeze(con, sql)
        proxy.stop()
        with pytest.raises(OSError):
            urlopen(proxy.endpoint + "/v1/config", timeout=1)
        # Data files stay accessible. Neither metadata JSON nor manifests can
        # be reopened, so a local metadata rebind cannot pass this test.
        with unavailable_metadata(tables):
            observed, schema = execute_frozen(payload, tables)
        assert sorted(observed, key=repr) == sorted(expected, key=repr)
        if query == "empty":
            # The native task entry point reports positional col_N names for
            # every source; public Relation names are covered by M4b's suite.
            assert len(schema["names"]) == 2
            assert schema["types"] == ["BIGINT", "VARCHAR"]
    finally:
        proxy.stop()


def test_frozen_snapshot_excludes_independent_append(frozen_tables):
    endpoint, tables, _features = frozen_tables
    with iceberg_tables(endpoint, tables.directory.parent) as owned:
        check_independent_append(endpoint, owned)


def check_independent_append(endpoint, tables):
    import pyarrow as pa
    from pyiceberg.io.pyarrow import schema_to_pyarrow

    namespace = f'lake."{tables.namespace}"'
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f"SELECT id, label FROM {namespace}.items")
    table = tables.catalog.load_table((tables.namespace, "items"))
    old_snapshot = table.current_snapshot().snapshot_id
    table.append(
        pa.Table.from_pylist(
            [{"id": 800001, "label": "committed-after-freeze"}],
            schema=schema_to_pyarrow(table.schema(), include_field_ids=False),
        )
    )
    table.refresh()
    assert table.current_snapshot().snapshot_id != old_snapshot
    assert table.scan().to_arrow().num_rows == len(tables.rows) + 1
    with unavailable_metadata(tables):
        observed, _schema = execute_frozen(payload, tables)
    assert sorted(observed) == list(tables.rows)


def test_explicit_empty_assignment_clears_nonempty_worker(frozen_tables):
    endpoint, tables, _features = frozen_tables
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".items')
        empty = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
            con.sql(f'SELECT * FROM lake."{tables.namespace}".empty'), "frozen-empty-" + uuid.uuid4().hex
        ).to_physical_plan(con)
        batch_map = empty.scan_split_batch_map()
        assert len(batch_map) == 1
        batches = next(iter(batch_map.values()))
        assert len(batches) == 1
        empty_batch = bytes(batches[0])
        empty = None
    observed, schema = execute_frozen(payload, tables, split_batch=empty_batch)
    assert observed == []
    assert len(schema["names"]) == 2
    assert schema["types"] == ["BIGINT", "VARCHAR"]


def test_assignment_from_another_table_cannot_replace_frozen_identity(frozen_tables):
    endpoint, tables, _features = frozen_tables
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".items')
    with iceberg_tables(endpoint, tables.directory.parent) as other:
        with connection(other) as con:
            attach(con, endpoint)
            foreign = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
                con.sql(f'SELECT * FROM lake."{other.namespace}".items'), "frozen-other-" + uuid.uuid4().hex
            ).to_physical_plan(con)
            batch_map = foreign.scan_split_batch_map()
            assert len(batch_map) == 1
            foreign_batch = bytes(next(iter(batch_map.values()))[0])
            foreign = None
        with pytest.raises((ValueError, vane.Error), match="does not match the worker snapshot"):
            execute_frozen(payload, tables, split_batch=foreign_batch)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("partitioned", "partitioned tables"),
        ("evolved", "schema evolution"),
        ("position_deletes", "delete manifests"),
        ("equality_deletes", "delete manifests"),
    ],
)
def test_unsupported_transport_remains_a_valid_native_local_read(frozen_tables, name, message):
    endpoint, tables, features = frozen_tables
    with connection(tables) as con:
        attach(con, endpoint)
        sql = f'SELECT * FROM lake."{tables.namespace}"."{name}"'
        assert sorted(con.sql(sql).fetchall()) == sorted(features[name].rows)
        with pytest.raises(ValueError, match=message):
            freeze(con, sql)


def test_frozen_worker_requires_explicit_assignment(frozen_tables):
    endpoint, tables, _features = frozen_tables
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".items')
    with connection(tables) as worker:
        logical = pickle.loads(payload)
        physical = logical.to_physical_plan(worker)
        # Assign scan node IDs through the same planning step used before
        # worker submission; the direct native entry point also accepts local plans.
        assert physical.scan_split_batch_map()
        worker_plan = physical.clone(worker)
        runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
        try:
            with pytest.raises(ValueError, match="no explicit worker split assignment"):
                runner.execute_native(worker, worker_plan)
        finally:
            runner.shutdown()


@pytest.mark.parametrize(
    ("name", "message"),
    [("format_v1", "requires format version 2"), ("nested_schema", "nested or unmapped columns")],
)
def test_transport_boundary_rejects_independent_unsupported_metadata(frozen_tables, name, message):
    endpoint, tables, _features = frozen_tables
    with connection(tables) as con:
        attach(con, endpoint)
        with pytest.raises(ValueError, match=message):
            freeze(con, f'SELECT * FROM lake."{tables.namespace}"."{name}"')
