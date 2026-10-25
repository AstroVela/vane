# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Frozen flat schema evolution compared with independent SDK/Arrow results."""

from __future__ import annotations

import json
import pickle

import pytest
from gravitino_iceberg_helpers import iceberg_tables
from iceberg_scan_helpers import RestForwarder, attach, connection, execute_frozen, freeze, unavailable_metadata
from iceberg_schema_helpers import (
    SCHEMA_TABLES,
    evolve_after_freeze,
    populate_schema_evolution,
    schema_evolution_tables,
    schema_filter,
    schema_reference,
)

import vane
from vane.catalog import IcebergCatalog

pytestmark = pytest.mark.external_service


@pytest.fixture(scope="module")
def schema_tables():
    with schema_evolution_tables() as setup:
        yield setup


@pytest.fixture(autouse=True)
def native_only(monkeypatch):
    import ray

    monkeypatch.setenv("VANE_RUNNER", "local")
    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "1")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "1")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native schema-evolution qualification must not initialize Ray")

    monkeypatch.setattr(ray, "init", forbidden)


@pytest.mark.parametrize("name", SCHEMA_TABLES)
@pytest.mark.parametrize("query", ["rows", "filtered", "projection", "aggregate", "old_snapshots"])
def test_selected_schema_and_field_ids_survive_detached_execution(schema_tables, name, query):
    endpoint, tables, features = schema_tables
    feature = features[name]
    table = f'lake."{tables.namespace}"."{name}"'
    sql = f"SELECT * FROM {table}"
    expected = list(feature.current.rows)
    expected_types = list(feature.current.types)
    cases = []
    if query == "filtered":
        where, predicate = schema_filter(name)
        sql += " WHERE " + where
        expected = list(schema_reference(tables, name, row_filter=predicate).rows)
    elif query == "projection":
        names = tuple(reversed(feature.current.names))
        sql = "SELECT " + ", ".join('"' + field + '"' for field in names) + f" FROM {table}"
        positions = [feature.current.names.index(field) for field in names]
        expected = [tuple(row[index] for index in positions) for row in feature.current.rows]
        expected_types = [feature.current.types[index] for index in positions]
    elif query == "aggregate":
        column = feature.current.names[-1]
        sql = f'SELECT count(*), count("{column}"), sum(id)::BIGINT FROM {table}'
        expected = [
            (
                len(expected),
                sum(row[-1] is not None for row in expected),
                sum(row[feature.current.names.index("id")] for row in expected),
            )
        ]
        expected_types = ["BIGINT", "BIGINT", "BIGINT"]
    elif query == "old_snapshots":
        cases = [
            (f"SELECT * FROM {table} AT (VERSION => {view.snapshot_id})", list(view.rows), list(view.types))
            for view in feature.history
        ]
    if not cases:
        cases = [(sql, expected, expected_types)]
    proxy = RestForwarder(endpoint)
    try:
        frozen = []
        with connection(tables) as con:
            attach(con, proxy.endpoint)
            for sql, expected, expected_types in cases:
                frozen.append((freeze(con, sql), expected, expected_types))
        proxy.stop()
        with unavailable_metadata(tables):
            for payload, expected, expected_types in frozen:
                rows, schema = execute_frozen(payload, tables)
                assert sorted(rows, key=repr) == sorted(expected, key=repr)
                assert schema["types"] == expected_types
    finally:
        proxy.stop()


@pytest.mark.parametrize(
    "name", ["schema_columns", "schema_promotions", "schema_identity", "schema_dropped_partition", "schema_defaults"]
)
def test_public_local_schema_evolution_relations(schema_tables, name):
    endpoint, tables, features = schema_tables
    view = features[name].current
    where, predicate = schema_filter(name)
    expected = schema_reference(tables, name, row_filter=predicate)
    with connection(tables) as con:
        vane.load_installed_extension("iceberg", connection=con)
        catalog = IcebergCatalog.attach(con, "lake", endpoint=endpoint)
        relation = catalog.table(tables.namespace, name)
        assert relation.columns == list(view.names)
        assert [str(value) for value in relation.types] == list(view.types)
        assert relation.order("id").fetchall() == list(view.rows)
        assert relation.filter(where).order("id").fetchall() == list(expected.rows)


@pytest.mark.parametrize("name", ["schema_promotions", "schema_identity", "schema_transformed"])
def test_promoted_double_filter_preserves_values_beyond_float32_precision(schema_tables, name):
    import pyarrow.compute as pc

    endpoint, tables, _features = schema_tables
    expected = schema_reference(
        tables, name, row_filter=(pc.field("value") > 16777219.8) & (pc.field("value") < 16777221.0)
    )
    positions = [expected.names.index(column) for column in ("id", "value")]
    rows = [tuple(row[index] for index in positions) for row in expected.rows]
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(
            con,
            f'SELECT id, value FROM lake."{tables.namespace}"."{name}" WHERE value > 16777219.8 AND value < 16777221.0',
        )
    with unavailable_metadata(tables):
        observed, schema = execute_frozen(payload, tables)
    assert sorted(observed) == rows
    assert schema["types"] == ["BIGINT", "DOUBLE"]


def test_mixed_physical_schemas_keep_types_in_disjoint_assignments(schema_tables):
    endpoint, tables, features = schema_tables
    feature = features["schema_identity"]
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".schema_identity')
    with connection(tables) as worker:
        physical = pickle.loads(payload).to_physical_plan(worker)
        batches = [bytes(batch) for batch in next(iter(physical.scan_split_batch_map().values()))]
        assert len(batches) == feature.file_count > 2
        physical = None
    combined = []
    with unavailable_metadata(tables):
        for selected in (batches[::2], batches[1::2]):
            rows, schema = execute_frozen(
                payload, tables, split_batch=vane.ray_cxx.merge_scan_split_batches(list(reversed(selected)))
            )
            assert rows
            assert schema["types"] == list(feature.current.types)
            combined.extend(rows)
    assert sorted(combined, key=repr) == sorted(feature.current.rows, key=repr)


def test_schema_only_update_rejects_old_schema_assignment_for_the_same_snapshot(schema_tables):
    endpoint, tables, features = schema_tables
    feature = features["schema_metadata_only"]
    previous = feature.history[0]
    assert previous.snapshot_id == feature.current.snapshot_id
    assert previous.schema_id != feature.current.schema_id
    with connection(tables) as con:
        attach(con, endpoint)
        table = f'lake."{tables.namespace}".schema_metadata_only'
        payload = freeze(con, f"SELECT * FROM {table}")
        old = freeze(con, f"SELECT * FROM {table} AT (VERSION => {previous.snapshot_id})")
    with connection(tables) as worker:
        physical = pickle.loads(old).to_physical_plan(worker)
        batch = bytes(next(iter(physical.scan_split_batch_map().values()))[0])
        physical = None
    with (
        unavailable_metadata(tables),
        pytest.raises((ValueError, vane.Error), match="does not match the worker snapshot"),
    ):
        execute_frozen(payload, tables, split_batch=batch)


def test_restored_schema_rejects_a_malformed_typed_null_default(schema_tables):
    endpoint, tables, _features = schema_tables
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".schema_defaults')
    # Frozen reader-schema property 1007, five IDs, then default flags/values.
    schema_ids = b"\xef\x03\x01\x00\x05\x01\x02\x03\x04\x05\x02\x00"
    assert payload.count(schema_ids) == 1
    start = payload.index(schema_ids)
    # Logical binds explicitly serialize property 101's absent type_info.
    value_type = b"\x64\x00\x64\x00\x0e\x65\x00\x00\xff\xff"  # BIGINT 14.
    index = payload.index(value_type, start)
    assert index - start < 128
    replacement = b"\x64\x00\x64\x00\x0d\x65\x00\x00\xff\xff"  # INTEGER 13, still NULL.
    tampered = payload[:index] + replacement + payload[index + len(value_type) :]
    with pytest.raises((ValueError, vane.Error), match="column default does not match the frozen field schema"):
        execute_frozen(tampered, tables)


def test_evolved_empty_table_and_explicit_empty_assignment_keep_schema(schema_tables):
    endpoint, tables, features = schema_tables
    with connection(tables) as con:
        attach(con, endpoint)
        empty = freeze(con, f'SELECT * FROM lake."{tables.namespace}".schema_empty')
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".schema_defaults')
    with connection(tables) as worker:
        physical = pickle.loads(empty).to_physical_plan(worker)
        batch = bytes(next(iter(physical.scan_split_batch_map().values()))[0])
        physical = None
    with unavailable_metadata(tables):
        for frozen, assignment, view in (
            (empty, None, features["schema_empty"].current),
            (payload, batch, features["schema_defaults"].current),
        ):
            rows, schema = execute_frozen(frozen, tables, split_batch=assignment)
            assert rows == []
            assert schema["types"] == list(view.types)


def test_later_schema_update_and_append_do_not_change_frozen_schema(schema_tables):
    endpoint, shared, _features = schema_tables
    with iceberg_tables(endpoint, shared.directory.parent) as tables:
        feature = populate_schema_evolution(tables, names=("schema_columns",))["schema_columns"]
        proxy = RestForwarder(endpoint)
        try:
            with connection(tables) as con:
                attach(con, proxy.endpoint)
                payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".schema_columns')
            evolve_after_freeze(tables)
            proxy.stop()
            with unavailable_metadata(tables):
                rows, schema = execute_frozen(payload, tables)
            assert sorted(rows, key=repr) == sorted(feature.current.rows, key=repr)
            assert schema["types"] == list(feature.current.types)
        finally:
            proxy.stop()


def test_name_mappings_remain_explicitly_guarded(schema_tables):
    endpoint, shared, _features = schema_tables
    with iceberg_tables(endpoint, shared.directory.parent) as tables:
        feature = populate_schema_evolution(tables, names=("schema_metadata_only",))["schema_metadata_only"]
        table = tables.catalog.load_table((tables.namespace, feature.name))
        mapping = [{"field-id": field.field_id, "names": [field.name]} for field in table.schema().fields]
        with table.transaction() as transaction:
            transaction.set_properties(**{"schema.name-mapping.default": json.dumps(mapping)})
        with connection(tables) as con:
            attach(con, endpoint)
            sql = f'SELECT * FROM lake."{tables.namespace}".schema_metadata_only'
            assert sorted(con.sql(sql).fetchall(), key=repr) == sorted(feature.current.rows, key=repr)
            with pytest.raises(ValueError, match="does not yet support name mappings"):
                freeze(con, sql)


def test_nested_history_is_guarded_even_when_selected_schema_is_flat(schema_tables):
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField, StringType, StructType

    endpoint, shared, _features = schema_tables
    with iceberg_tables(endpoint, shared.directory.parent) as tables:
        directory = tables.directory / "nested_history"
        for path in (directory, directory / "metadata", directory / "data"):
            path.mkdir(mode=0o777)
            path.chmod(0o777)
        table = tables.catalog.create_table(
            (tables.namespace, "nested_history"),
            schema=Schema(
                NestedField(1, "id", LongType(), required=True),
                NestedField(2, "payload", StructType(NestedField(3, "text", StringType()))),
            ),
            location=directory.as_uri(),
            properties={"format-version": "2"},
        )
        with table.update_schema() as update:
            update.delete_column("payload")
        with connection(tables) as con:
            attach(con, endpoint)
            sql = f'SELECT * FROM lake."{tables.namespace}".nested_history'
            assert con.sql(sql).fetchall() == []
            with pytest.raises(ValueError, match="nested or unmapped columns in schema history"):
                freeze(con, sql)
