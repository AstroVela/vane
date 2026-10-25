# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Native frozen-snapshot and per-file transport against an explicitly provisioned REST service."""

from __future__ import annotations

import os
import pickle
import uuid
from pathlib import Path
from urllib.request import urlopen

import pytest
from gravitino_iceberg_helpers import iceberg_tables
from iceberg_feature_helpers import populate_iceberg_features
from iceberg_scan_helpers import (
    RestForwarder,
    attach,
    connection,
    execute_frozen,
    freeze,
    unavailable_metadata,
)

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
    with iceberg_tables(endpoint.rstrip("/"), Path(shared), file_count=9) as tables:
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


@pytest.mark.parametrize(
    "query", ["rows", "projection", "filter", "aggregate", "empty", "count", "count_empty", "constant"]
)
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
        "count": (f"SELECT count(*) FROM {namespace}.items", [(len(tables.rows),)]),
        "count_empty": (f"SELECT count(*) FROM {namespace}.empty", [(0,)]),
        "constant": (f"SELECT 7 FROM {namespace}.items", [(7,)] * len(tables.rows)),
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


def test_file_assignments_cover_exact_snapshot_independently_of_slot_hints(frozen_tables, monkeypatch):
    endpoint, tables, _features = frozen_tables
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT id, label FROM lake."{tables.namespace}".items')
    batches = None
    for slots in (1, 2):
        monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", str(slots))
        with connection(tables) as worker:
            logical = pickle.loads(payload)
            physical = logical.to_physical_plan(worker)
            batch_map = physical.scan_split_batch_map()
            assert len(batch_map) == 1
            planned = [bytes(batch) for batch in next(iter(batch_map.values()))]
            assert len(planned) == 9
            if batches is None:
                batches = planned
            else:
                assert batches == planned
            physical = logical = None
    # Each assignment is executable on its own. The orphan Parquet file is
    # deliberately beside committed files and must never enter any split.
    observed = []
    with unavailable_metadata(tables):
        for batch in batches:
            rows, _schema = execute_frozen(payload, tables, split_batch=batch)
            assert rows
            observed.extend(rows)
        merged, _schema = execute_frozen(
            payload, tables, split_batch=vane.ray_cxx.merge_scan_split_batches(list(reversed(batches)))
        )
    assert sorted(observed) == list(tables.rows)
    assert sorted(merged) == list(tables.rows)


def test_duplicate_file_assignment_fails_before_execution(frozen_tables):
    endpoint, tables, _features = frozen_tables
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".items')
    with connection(tables) as worker:
        physical = pickle.loads(payload).to_physical_plan(worker)
        batch = bytes(next(iter(physical.scan_split_batch_map().values()))[0])
        physical = None
    with pytest.raises((ValueError, vane.Error), match="[Dd]uplicate"):
        duplicate = vane.ray_cxx.merge_scan_split_batches([batch, batch])
        execute_frozen(payload, tables, split_batch=duplicate)


@pytest.mark.parametrize("tamper", ["codec", "split_id", "remote_path"])
def test_file_assignment_rejects_mismatched_codec_identity_and_path(frozen_tables, tamper):
    endpoint, tables, _features = frozen_tables
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".items')
    with connection(tables) as worker:
        physical = pickle.loads(payload).to_physical_plan(worker)
        batch = bytes(next(iter(physical.scan_split_batch_map().values()))[0])
        physical = None
    if tamper == "codec":
        original, replacement, message = b"vane.iceberg.frozen-file-scan", b"vane.iceberg.frozen-file-scax", "codec"
    elif tamper == "split_id":
        original, replacement, message = b"file-0", b"file-x", "invalid file identity"
    else:
        committed = tables.catalog.load_table((tables.namespace, "items")).scan().plan_files()
        original = next(task.file.file_path.encode() for task in committed if task.file.file_path.encode() in batch)
        assert original.startswith(b"file:///")
        replacement, message = b"https://" + original[len(b"file:///") :], "local data-file paths"
    assert len(original) == len(replacement)
    assert batch.count(original) == 1
    with pytest.raises((ValueError, vane.Error), match=message):
        execute_frozen(payload, tables, split_batch=batch.replace(original, replacement))
