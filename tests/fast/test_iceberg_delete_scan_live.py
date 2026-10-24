# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Frozen v2 delete semantics, independent references and native assignment checks."""

from __future__ import annotations

import pickle

import pytest
from gravitino_iceberg_helpers import iceberg_tables
from iceberg_delete_helpers import (
    DELETE_TABLES,
    commit_deletes,
    delete_reference,
    delete_tables,
    local_path,
    manifest_entries,
    populate_delete_tables,
)
from iceberg_partition_helpers import append_partition_rows
from iceberg_scan_helpers import RestForwarder, attach, connection, execute_frozen, freeze, unavailable_metadata

import vane
from vane.catalog import IcebergCatalog

pytestmark = pytest.mark.external_service


@pytest.fixture(scope="module")
def frozen_deletes():
    with delete_tables() as setup:
        yield setup


@pytest.fixture(autouse=True)
def native_only(monkeypatch):
    import ray

    monkeypatch.setenv("VANE_RUNNER", "local")
    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "1")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "1")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native delete qualification must not initialize Ray")

    monkeypatch.setattr(ray, "init", forbidden)


def filtered_rows(view):
    id_index, label_index = view.names.index("id"), view.names.index("label")
    return [row for row in view.rows if row[id_index] % 4 == 1 or row[label_index] is None]


@pytest.mark.parametrize("name", DELETE_TABLES)
@pytest.mark.parametrize("query", ["rows", "filtered", "projection", "aggregate", "old_snapshots"])
def test_frozen_delete_rows_projection_filters_counts_and_history(frozen_deletes, name, query):
    endpoint, tables, features = frozen_deletes
    feature = features[name]
    table = f'lake."{tables.namespace}"."{name}"'
    sql = f"SELECT * FROM {table}"
    rows = list(feature.current.rows)
    types = list(feature.current.types)
    if query == "filtered":
        sql += " WHERE id % 4 = 1 OR label IS NULL"
        rows = filtered_rows(feature.current)
    elif query == "projection":
        sql = f"SELECT id FROM {table}"
        rows = [(row[feature.current.names.index("id")],) for row in rows]
        types = ["BIGINT"]
    elif query == "aggregate":
        sql = f"SELECT count(*), sum(id)::BIGINT FROM {table}"
        rows = [(len(rows), sum(row[feature.current.names.index("id")] for row in rows) if rows else None)]
        types = ["BIGINT", "BIGINT"]
    cases = [(sql, rows, types)]
    if query == "old_snapshots":
        cases = [
            (f"SELECT * FROM {table} AT (VERSION => {view.snapshot_id})", list(view.rows), list(view.types))
            for view in feature.history
        ]
    proxy = RestForwarder(endpoint)
    try:
        with connection(tables) as con:
            attach(con, proxy.endpoint)
            frozen = [(freeze(con, sql), rows, types) for sql, rows, types in cases]
        proxy.stop()
        with unavailable_metadata(tables):
            for payload, expected, expected_types in frozen:
                observed, schema = execute_frozen(payload, tables)
                assert sorted(observed, key=repr) == sorted(expected, key=repr)
                assert schema["types"] == expected_types
    finally:
        proxy.stop()


@pytest.mark.parametrize("name", DELETE_TABLES)
def test_public_local_delete_relations_match_independent_reference(frozen_deletes, name):
    endpoint, tables, features = frozen_deletes
    view = features[name].current
    with connection(tables) as con:
        vane.load_installed_extension("iceberg", connection=con)
        catalog = IcebergCatalog.attach(con, "lake", endpoint=endpoint)
        relation = catalog.table(tables.namespace, name)
        assert relation.columns == list(view.names)
        assert relation.order("id").fetchall() == list(view.rows)
        assert relation.project("id").order("id").fetchall() == [(row[view.names.index("id")],) for row in view.rows]
        assert relation.filter("id % 4 = 1 OR label IS NULL").order("id").fetchall() == filtered_rows(view)
        assert relation.aggregate("count(*), sum(id)::BIGINT").fetchone() == (
            len(view.rows),
            sum(row[view.names.index("id")] for row in view.rows) if view.rows else None,
        )


def test_disjoint_assignments_keep_only_their_applicable_deletes(frozen_deletes):
    endpoint, tables, features = frozen_deletes
    feature = features["delete_specs"]
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".delete_specs')
    with connection(tables) as worker:
        physical = pickle.loads(payload).to_physical_plan(worker)
        batches = [bytes(batch) for batch in next(iter(physical.scan_split_batch_map().values()))]
        assert len(batches) == feature.file_count > 2
        physical = None
    observed = []
    with unavailable_metadata(tables):
        for selected in (batches[::2], batches[1::2]):
            rows, _schema = execute_frozen(payload, tables, split_batch=vane.ray_cxx.merge_scan_split_batches(selected))
            observed.extend(rows)
    assert sorted(observed, key=repr) == sorted(feature.current.rows, key=repr)


def test_later_delete_and_append_do_not_change_the_frozen_snapshot(frozen_deletes):
    endpoint, shared, _features = frozen_deletes
    with iceberg_tables(endpoint, shared.directory.parent) as tables:
        feature = populate_delete_tables(tables, names=("delete_equality_multi",))["delete_equality_multi"]
        proxy = RestForwarder(endpoint)
        try:
            with connection(tables) as con:
                attach(con, proxy.endpoint)
                payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".delete_equality_multi')
            commit_deletes(tables, feature.name, [{1: key} for key in range(27)], equality_ids=(1,))
            append_partition_rows(
                tables,
                feature.name,
                [{"id": 800001, "label": "later", "region": None, "rank": None}],
                omit_partition_columns=False,
            )
            assert delete_reference(tables, feature.name).rows == ((800001, "later", None, None),)
            proxy.stop()
            with unavailable_metadata(tables):
                rows, _schema = execute_frozen(payload, tables)
            assert sorted(rows, key=repr) == sorted(feature.current.rows, key=repr)
        finally:
            proxy.stop()


def test_missing_frozen_delete_file_fails_instead_of_reading_undeleted_rows(frozen_deletes):
    endpoint, shared, _features = frozen_deletes
    with iceberg_tables(endpoint, shared.directory.parent) as tables:
        populate_delete_tables(tables, names=("delete_equality_nullable",))
        with connection(tables) as con:
            attach(con, endpoint)
            payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".delete_equality_nullable')
        table = tables.catalog.load_table((tables.namespace, "delete_equality_nullable"))
        paths = [
            local_path(entry.data_file.file_path)
            for _manifest, entry in manifest_entries(table)
            if entry.data_file.content != 0
        ]
        moved = []
        try:
            for path in paths:
                hidden = path.with_suffix(".unavailable")
                path.rename(hidden)
                moved.append((path, hidden))
            with unavailable_metadata(tables), pytest.raises((ValueError, vane.Error), match="(?i)(file|open|exist)"):
                execute_frozen(payload, tables)
        finally:
            for path, hidden in moved:
                hidden.rename(path)


def tamper_delete(batch, kind):
    # A native split contains one selected file and its descriptor. Locate the
    # exact committed delete URI before changing same-length owned properties.
    start = batch.index(b"/data/delete-")
    end = batch.index(b".parquet", start) + len(b".parquet")
    suffix = batch[end:]
    if kind == "content":
        original, replacement = b"\x02\x00\x02\x03\x00", b"\x02\x00\x00\x03\x00"
        message = "Invalid frozen Iceberg delete file"
    elif kind == "sequence":
        original, replacement = b"\x03\x00\x0a\x04\x00", b"\x03\x00\x00\x04\x00"
        message = "does not apply to its assigned data file"
    elif kind == "record_count":
        original, replacement = b"\x04\x00\x04\x05\x00", b"\x04\x00\x00\x05\x00"
        message = "delete record count does not match its manifest"
    else:
        original = b"\x08\x00\x02\x04\x02\xff\xff"
        replacement = b"\x08\x00\x02\x04\x06\xff\xff" if kind == "unknown_field" else b"\x08\x00\x02\x04\x04\xff\xff"
        message = "unknown or duplicate field"
    assert len(original) == len(replacement) and suffix.count(original) == 1
    return batch[:end] + suffix.replace(original, replacement, 1), message


@pytest.mark.parametrize("kind", ["content", "sequence", "record_count", "unknown_field", "duplicate_field"])
def test_delete_assignment_rejects_malformed_owned_state(frozen_deletes, kind):
    endpoint, tables, _features = frozen_deletes
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".delete_equality_multi')
    with connection(tables) as con:
        physical = pickle.loads(payload).to_physical_plan(con)
        batch = bytes(next(iter(physical.scan_split_batch_map().values()))[0])
        physical = None
    tampered, message = tamper_delete(batch, kind)
    with unavailable_metadata(tables), pytest.raises((ValueError, vane.Error), match=message):
        execute_frozen(payload, tables, split_batch=tampered)


def test_invalid_later_delete_assignment_fails_before_opening_any_data_file(frozen_deletes):
    endpoint, tables, _features = frozen_deletes
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".delete_equality_multi')
    with connection(tables) as con:
        physical = pickle.loads(payload).to_physical_plan(con)
        batches = [bytes(batch) for batch in next(iter(physical.scan_split_batch_map().values()))]
        physical = None
    invalid, message = tamper_delete(batches[1], "unknown_field")
    table = tables.catalog.load_table((tables.namespace, "delete_equality_multi"))
    data_path = next(
        local_path(entry.data_file.file_path)
        for _manifest, entry in manifest_entries(table)
        if entry.data_file.content == 0 and entry.data_file.file_path.encode() in batches[0]
    )
    hidden = data_path.with_suffix(".unavailable")
    data_path.rename(hidden)
    try:
        assignment = vane.ray_cxx.merge_scan_split_batches([batches[0], invalid])
        with unavailable_metadata(tables), pytest.raises((ValueError, vane.Error), match=message):
            execute_frozen(payload, tables, split_batch=assignment)
    finally:
        hidden.rename(data_path)


def test_explicit_empty_assignment_clears_retained_delete_scan_files(frozen_deletes):
    endpoint, tables, features = frozen_deletes
    with connection(tables) as con:
        attach(con, endpoint)
        empty = freeze(con, f'SELECT * FROM lake."{tables.namespace}".empty')
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".delete_equality_multi')
    with connection(tables) as con:
        physical = pickle.loads(empty).to_physical_plan(con)
        batch = bytes(next(iter(physical.scan_split_batch_map().values()))[0])
        physical = None
    with unavailable_metadata(tables):
        rows, schema = execute_frozen(payload, tables, split_batch=batch)
    assert rows == []
    assert schema["types"] == list(features["delete_equality_multi"].current.types)


@pytest.mark.parametrize("position", [-1, 3])
def test_invalid_position_values_fail_in_both_native_readers(frozen_deletes, position):
    from iceberg_delete_helpers import PATH_ID, POS_ID

    endpoint, shared, _features = frozen_deletes
    with iceberg_tables(endpoint, shared.directory.parent) as tables:
        populate_delete_tables(tables, names=("delete_position_sequence",))
        table = tables.catalog.load_table((tables.namespace, "delete_position_sequence"))
        file = next(entry.data_file for _manifest, entry in manifest_entries(table) if entry.data_file.content == 0)
        assert file.record_count == 3
        commit_deletes(tables, "delete_position_sequence", [{PATH_ID: file.file_path, POS_ID: position}])
        with connection(tables) as con:
            attach(con, endpoint)
            sql = f'SELECT * FROM lake."{tables.namespace}".delete_position_sequence'
            with pytest.raises(vane.Error, match="(?i)(nonnegative position|exceeds.*record count)"):
                con.sql(sql).fetchall()
        with connection(tables) as con:
            attach(con, endpoint)
            payload = freeze(con, sql)
        with (
            unavailable_metadata(tables),
            pytest.raises((ValueError, vane.Error), match="(?i)(nonnegative position|exceeds.*record count)"),
        ):
            execute_frozen(payload, tables)
