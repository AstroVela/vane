# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Frozen identity partition projection against independent SDK fixtures."""

from __future__ import annotations

import pickle

import pytest
from gravitino_iceberg_helpers import iceberg_tables
from iceberg_partition_helpers import append_partition_rows, identity_partition_tables, populate_identity_partitions
from iceberg_scan_helpers import RestForwarder, attach, connection, execute_frozen, freeze, unavailable_metadata

import vane

pytestmark = pytest.mark.external_service


@pytest.fixture(scope="module")
def partition_tables():
    with identity_partition_tables() as setup:
        yield setup


@pytest.fixture(autouse=True)
def native_only(monkeypatch):
    import ray

    monkeypatch.setenv("VANE_RUNNER", "local")
    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "1")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "1")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native identity-partition qualification must not initialize Ray")

    monkeypatch.setattr(ray, "init", forbidden)


def partition_query(namespace, name, feature, query):
    table = f'lake."{namespace}"."{name}"'
    if query == "rows":
        return f"SELECT id, label, region FROM {table}", list(feature.rows)
    if query == "partition":
        return f"SELECT region FROM {table}", [(region,) for _, _, region in feature.rows]
    if query == "null":
        return f"SELECT id, region FROM {table} WHERE region IS NULL", [
            (key, region) for key, _, region in feature.rows if region is None
        ]
    if query == "mixed":
        return (
            f"SELECT id, label, region FROM {table} WHERE (region = 'east.点' AND id >= 5) OR (region IS NULL AND id < 8)",
            [row for row in feature.rows if (row[2] == "east.点" and row[0] >= 5) or (row[2] is None and row[0] < 8)],
        )
    keys = [key for key, _, region in feature.rows if region == "west"]
    return f"SELECT count(*), sum(id)::BIGINT FROM {table} WHERE region = 'west'", [(len(keys), sum(keys))]


@pytest.mark.parametrize("name", ["partition_present", "partition_omitted", "partition_specs"])
@pytest.mark.parametrize("query", ["rows", "partition", "null", "mixed", "aggregate"])
def test_identity_values_restore_without_metadata_or_rest(partition_tables, name, query):
    endpoint, tables, features = partition_tables
    sql, expected = partition_query(tables.namespace, name, features[name], query)
    proxy = RestForwarder(endpoint)
    try:
        with connection(tables) as con:
            attach(con, proxy.endpoint)
            payload = freeze(con, sql)
        proxy.stop()
        with unavailable_metadata(tables):
            observed, _schema = execute_frozen(payload, tables)
        assert sorted(observed, key=repr) == sorted(expected, key=repr)
    finally:
        proxy.stop()


@pytest.mark.parametrize("query", ["rows", "negative", "null"])
def test_manifest_temporal_decimal_and_null_values_keep_their_types(partition_tables, query):
    endpoint, tables, features = partition_tables
    table = f'lake."{tables.namespace}".partition_typed'
    where = {"rows": "", "negative": " WHERE day < DATE '1970-01-01' AND amount < 0", "null": " WHERE moment IS NULL"}[
        query
    ]
    expected = list(features["partition_typed"].rows)
    if query == "negative":
        expected = [row for row in expected if row[1] is not None and row[1].year < 1970 and row[2] < 0]
    elif query == "null":
        expected = [row for row in expected if row[3] is None]
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f"SELECT id, day, amount, moment FROM {table}{where}")
    with unavailable_metadata(tables):
        observed, schema = execute_frozen(payload, tables)
    assert sorted(observed) == expected
    assert schema["types"] == ["BIGINT", "DATE", "DECIMAL(10,2)", "TIMESTAMP"]


@pytest.mark.parametrize("name", ["partition_present", "partition_omitted", "partition_specs"])
def test_explicit_old_snapshot_uses_that_snapshots_file_specs(partition_tables, name):
    endpoint, tables, features = partition_tables
    snapshot = features[name].initial_snapshot_id
    expected = list(features[name].rows[:9])
    independent = (
        tables.catalog.load_table((tables.namespace, name))
        .scan(snapshot_id=snapshot)
        .to_arrow()
        .sort_by("id")
        .to_pylist()
    )
    assert [tuple(row[field] for field in ("id", "label", "region")) for row in independent] == expected
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(
            con, f'SELECT id, label, region FROM lake."{tables.namespace}"."{name}" AT (VERSION => {snapshot})'
        )
    with unavailable_metadata(tables):
        observed, _schema = execute_frozen(payload, tables)
    assert sorted(observed) == expected


@pytest.mark.parametrize("name, expected", [("partition_empty", 0), ("partition_omitted", 27)])
def test_partition_count_without_data_projection(partition_tables, name, expected):
    endpoint, tables, _features = partition_tables
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT count(*) FROM lake."{tables.namespace}"."{name}"')
    with unavailable_metadata(tables):
        observed, _schema = execute_frozen(payload, tables)
    assert observed == [(expected,)]


def test_no_matching_partition_preserves_empty_schema(partition_tables):
    endpoint, tables, _features = partition_tables
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(
            con, f"SELECT id, label, region FROM lake.\"{tables.namespace}\".partition_omitted WHERE region = 'absent'"
        )
    with unavailable_metadata(tables):
        observed, schema = execute_frozen(payload, tables)
    assert observed == []
    assert schema["types"] == ["BIGINT", "VARCHAR", "VARCHAR"]


def test_partial_assignments_keep_each_files_own_constants(partition_tables):
    endpoint, tables, features = partition_tables
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT id, label, region FROM lake."{tables.namespace}".partition_omitted')
    with connection(tables) as worker:
        physical = pickle.loads(payload).to_physical_plan(worker)
        batches = [bytes(batch) for batch in next(iter(physical.scan_split_batch_map().values()))]
        assert len(batches) == 9
        physical = None
    combined = []
    with unavailable_metadata(tables):
        for selected in (batches[::2], batches[1::2]):
            observed, _schema = execute_frozen(
                payload, tables, split_batch=vane.ray_cxx.merge_scan_split_batches(list(reversed(selected)))
            )
            assert observed
            combined.extend(observed)
    assert sorted(combined) == list(features["partition_omitted"].rows)


def test_physical_field_id_wins_over_a_different_partition_constant(partition_tables):
    endpoint, tables, _features = partition_tables
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT id, label, region FROM lake."{tables.namespace}".partition_present')
    with connection(tables) as worker:
        physical = pickle.loads(payload).to_physical_plan(worker)
        batches = [bytes(batch) for batch in next(iter(physical.scan_split_batch_map().values()))]
        batch = next(batch for batch in batches if "east.点".encode() in batch)
        physical = None
    original, replacement = "east.点".encode(), b"FAKE...."
    assert len(original) == len(replacement) and batch.count(original) == 1
    with unavailable_metadata(tables):
        observed, _schema = execute_frozen(payload, tables, split_batch=batch.replace(original, replacement))
    assert observed and {row[2] for row in observed} == {"east.点"}


def test_empty_assignment_clears_partition_constants_and_files(partition_tables):
    endpoint, tables, _features = partition_tables
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".partition_omitted')
        empty_payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".partition_empty')
    with connection(tables) as worker:
        physical = pickle.loads(empty_payload).to_physical_plan(worker)
        batch = bytes(next(iter(physical.scan_split_batch_map().values()))[0])
        physical = None
    with unavailable_metadata(tables):
        observed, schema = execute_frozen(payload, tables, split_batch=batch)
    assert observed == []
    assert schema["types"] == ["BIGINT", "VARCHAR", "VARCHAR"]


def test_later_independent_partition_append_does_not_change_frozen_rows(partition_tables):
    endpoint, shared_tables, _features = partition_tables
    with iceberg_tables(endpoint, shared_tables.directory.parent) as tables:
        features = populate_identity_partitions(tables)
        proxy = RestForwarder(endpoint)
        try:
            with connection(tables) as con:
                attach(con, proxy.endpoint)
                payload = freeze(con, f'SELECT id, label, region FROM lake."{tables.namespace}".partition_omitted')
            updated = append_partition_rows(
                tables,
                "partition_omitted",
                [
                    {"id": 800001, "label": "later", "region": "new-partition"},
                    {"id": 800002, "label": "later-null", "region": None},
                ],
                omit_partition_columns=True,
            )
            assert updated.scan().to_arrow().num_rows == 29
            proxy.stop()
            with unavailable_metadata(tables):
                observed, _schema = execute_frozen(payload, tables)
            assert sorted(observed) == list(features["partition_omitted"].rows)
        finally:
            proxy.stop()


def test_empty_transformed_partition_roundtrips(partition_tables):
    endpoint, tables, _features = partition_tables
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".partition_bucket')
    with unavailable_metadata(tables):
        rows, schema = execute_frozen(payload, tables)
    assert rows == []
    assert schema["types"] == ["BIGINT", "VARCHAR", "VARCHAR"]


@pytest.mark.parametrize("tamper", ["source_field", "constant_type", "assignment_version", "duplicate_source"])
def test_partition_assignment_rejects_malformed_owned_constants(partition_tables, tamper):
    endpoint, tables, _features = partition_tables
    name = "partition_typed" if tamper == "duplicate_source" else "partition_omitted"
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}"."{name}"')
    with connection(tables) as worker:
        physical = pickle.loads(payload).to_physical_plan(worker)
        batches = [bytes(batch) for batch in next(iter(physical.scan_split_batch_map().values()))]
        physical = None
    if tamper == "duplicate_source":
        batch = batches[0]
    else:
        value = b"west" if tamper == "constant_type" else "east.点".encode()
        batch = next(batch for batch in batches if value in batch)
    # The pinned DuckDB binary serializer uses u16 field tags and varints.
    # Preserve every byte length so rejection exercises the provider's value
    # validation rather than corrupting the shared envelope or its lengths.
    if tamper == "source_field":
        # Data-file property 3: one source field ID; property 4 follows it.
        original, replacement = b"\x03\x00\x01\x03\x04\x00", b"\x03\x00\x01\x04\x04\x00"
        message = "does not match the frozen field schema"
    elif tamper == "constant_type":
        # Value.type (100), LogicalType.id (100): VARCHAR 25 -> BLOB 26.
        # Use an ASCII value so BLOB parsing succeeds before schema validation.
        original, replacement = b"\x64\x00\x64\x00\x19\xff\xff", b"\x64\x00\x64\x00\x1a\xff\xff"
        message = "does not match the frozen field schema"
    elif tamper == "assignment_version":
        original, replacement = b"\x01\x00\x02\x02\x00\x01\x00", b"\x01\x00\x01\x02\x00\x01\x00"
        message = "Unsupported frozen Iceberg assignment version"
    else:
        # Three typed source IDs: DATE 2, DECIMAL 3, TIMESTAMP 4 -> 2, 2, 4.
        original, replacement = b"\x03\x00\x03\x02\x03\x04\x04\x00", b"\x03\x00\x03\x02\x02\x04\x04\x00"
        message = "invalid or duplicate source fields"
    assert len(original) == len(replacement) and batch.count(original) == 1
    with pytest.raises((ValueError, vane.Error), match=message):
        execute_frozen(payload, tables, split_batch=batch.replace(original, replacement))


def test_partition_bind_rejects_the_previous_wire_version(partition_tables):
    endpoint, tables, _features = partition_tables
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".partition_omitted')
    # Frozen bind property 1000/version 2, followed by property 1001/identity.
    original, replacement = b"\xe8\x03\x02\xe9\x03\x01\x00", b"\xe8\x03\x01\xe9\x03\x01\x00"
    assert payload.count(original) == 1
    with pytest.raises((ValueError, vane.Error), match="Unsupported frozen Iceberg bind version"):
        execute_frozen(payload.replace(original, replacement), tables)
