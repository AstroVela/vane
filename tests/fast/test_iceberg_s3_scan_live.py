# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Anonymous MinIO reads and detached native S3 assignments against real REST."""

from __future__ import annotations

import pickle

import pytest
from iceberg_delete_helpers import commit_deletes, delete_reference, manifest_entries
from iceberg_s3_helpers import S3_TABLES, append_rows, attach, connection, execute_frozen, s3_tables
from iceberg_scan_helpers import RestForwarder, freeze

import vane

pytestmark = pytest.mark.external_service


@pytest.fixture(scope="module")
def remote_tables(tmp_path_factory):
    with s3_tables(tmp_path_factory.mktemp("iceberg-s3")) as setup:
        yield setup
        tables, _features = setup
        events = tables.objects.events()
        assert events and any(event["range"] for event in events)
        assert not any(event["credentials"] or event["query"] for event in events)


@pytest.fixture(autouse=True)
def local_only(monkeypatch):
    import ray

    monkeypatch.setenv("VANE_RUNNER", "local")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("S3 native qualification must not initialize Ray")

    monkeypatch.setattr(ray, "init", forbidden)


def queries(tables, name, feature):
    view = feature.current
    qualified = f'lake."{tables.namespace}"."{name}"'
    label = "renamed_label" if "renamed_label" in view.names else "label"
    label_index = view.names.index(label)
    rows = list(view.rows)
    yield f"SELECT * FROM {qualified}", rows
    yield f"SELECT id FROM {qualified}", [(row[0],) for row in rows]
    yield (
        f"SELECT * FROM {qualified} WHERE id % 4 = 1 OR {label} IS NULL",
        [row for row in rows if row[0] % 4 == 1 or row[label_index] is None],
    )
    yield (
        f"SELECT count(*), sum(id)::BIGINT FROM {qualified}",
        [(len(rows), sum(row[0] for row in rows) if rows else None)],
    )
    for previous in feature.history:
        yield f"SELECT * FROM {qualified} AT (VERSION => {previous.snapshot_id})", list(previous.rows)


@pytest.mark.parametrize("name", S3_TABLES)
def test_public_local_s3_rows_projection_filters_counts_and_history(remote_tables, name):
    tables, features = remote_tables
    with connection(tables) as con:
        catalog = attach(con, tables.endpoint)
        relation = catalog.table(tables.namespace, name)
        assert relation.columns == list(features[name].current.names)
        assert relation.order("id").fetchall() == list(features[name].current.rows)
        for sql, expected in queries(tables, name, features[name]):
            assert sorted(con.sql(sql).fetchall(), key=repr) == sorted(expected, key=repr)
        assert con.execute("SELECT count(*) FROM duckdb_secrets()").fetchone() == (0,)
        with pytest.raises(vane.Error, match="(?i)read.only"):
            con.execute(f'DELETE FROM lake."{tables.namespace}"."{name}"')


@pytest.mark.parametrize("name", S3_TABLES)
def test_frozen_s3_replay_needs_neither_rest_nor_metadata(remote_tables, name):
    tables, features = remote_tables
    proxy = RestForwarder(tables.endpoint)
    try:
        with connection(tables) as con:
            attach(con, proxy.endpoint)
            frozen = [(freeze(con, sql), expected) for sql, expected in queries(tables, name, features[name])]
        for payload, _expected in frozen:
            assert tables.config["admin_key"].encode() not in payload
            assert tables.config["admin_secret"].encode() not in payload
        proxy.stop()
        start = len(tables.objects.events())
        tables.objects.policy(deny_metadata=True)
        try:
            for payload, expected in frozen:
                rows, _schema = execute_frozen(payload, tables)
                assert sorted(rows, key=repr) == sorted(expected, key=repr)
        finally:
            tables.objects.policy()
        assert not any("/metadata/" in event["path"] for event in tables.objects.events()[start:])
    finally:
        proxy.stop()


@pytest.mark.parametrize("name", ["positions", "equality"])
def test_s3_assignments_are_disjoint_and_keep_their_deletes(remote_tables, name):
    tables, features = remote_tables
    with connection(tables) as con:
        attach(con, tables.endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}"."{name}"')
    with connection(tables) as worker:
        plan = pickle.loads(payload).to_physical_plan(worker)
        batches = [bytes(batch) for batch in next(iter(plan.scan_split_batch_map().values()))]
        assert len(batches) == features[name].file_count == 9
        plan = None
    tables.objects.policy(deny_metadata=True)
    try:
        rows = []
        for selected in (batches[::2], batches[1::2]):
            observed, _schema = execute_frozen(
                payload, tables, split_batch=vane.ray_cxx.merge_scan_split_batches(selected)
            )
            rows.extend(observed)
        assert sorted(rows, key=repr) == sorted(features[name].current.rows, key=repr)
    finally:
        tables.objects.policy()


@pytest.mark.parametrize("content", [0, 2], ids=["data", "equality-delete"])
def test_missing_remote_object_fails_without_fallback(remote_tables, content):
    tables, _features = remote_tables
    with connection(tables) as con:
        attach(con, tables.endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".equality')
    table = tables.catalog.load_table((tables.namespace, "equality"))
    path = next(
        entry.data_file.file_path for _manifest, entry in manifest_entries(table) if entry.data_file.content == content
    )
    denied = "/" + path.removeprefix("s3://")
    tables.objects.policy(deny_metadata=True, denied=[denied])
    try:
        with pytest.raises((ValueError, vane.Error), match="(?i)(404|not found|file|open)"):
            execute_frozen(payload, tables)
    finally:
        tables.objects.policy()
    assert any(event["path"] == denied and event["status"] == 404 for event in tables.objects.events())


def test_later_remote_delete_and_append_do_not_change_frozen_snapshot(tmp_path):
    with s3_tables(tmp_path, names=("equality",)) as (tables, features):
        proxy = RestForwarder(tables.endpoint)
        try:
            with connection(tables) as con:
                attach(con, proxy.endpoint)
                payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".equality')
            commit_deletes(tables, "equality", [{1: key} for key in range(27)], equality_ids=(1,))
            append_rows(tables, "equality", [{"id": 800001, "label": "later", "region": None, "rank": None}])
            assert delete_reference(tables, "equality").rows == ((800001, "later", None, None),)
            proxy.stop()
            tables.objects.policy(deny_metadata=True)
            rows, _schema = execute_frozen(payload, tables)
            assert sorted(rows, key=repr) == sorted(features["equality"].current.rows, key=repr)
        finally:
            tables.objects.policy()
            proxy.stop()


@pytest.mark.parametrize(
    "replacement",
    [
        b"S3://",
        b"s3a:/",
        b"http:",
    ],
)
def test_assignment_rejects_unsupported_remote_schemes(remote_tables, replacement):
    tables, _features = remote_tables
    with connection(tables) as con:
        attach(con, tables.endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".items')
    with connection(tables) as worker:
        plan = pickle.loads(payload).to_physical_plan(worker)
        batch = bytes(next(iter(plan.scan_split_batch_map().values()))[0])
        plan = None
    assert batch.count(b"s3://") == 1
    tampered = batch.replace(b"s3://", replacement)
    with pytest.raises((ValueError, vane.Error), match="(?i)(local file paths|URL schemes)"):
        execute_frozen(payload, tables, split_batch=tampered)


@pytest.mark.parametrize("component", ["bucket", "userinfo", "port", "empty_key", "query", "fragment", "control"])
def test_assignment_rejects_ambiguous_s3_locators_before_io(remote_tables, component):
    tables, _features = remote_tables
    with connection(tables) as con:
        attach(con, tables.endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".items')
    with connection(tables) as worker:
        plan = pickle.loads(payload).to_physical_plan(worker)
        batch = bytes(next(iter(plan.scan_split_batch_map().values()))[0])
        plan = None
    start = batch.index(b"s3://") + 5
    end_bucket = batch.index(b"/", start)
    end_path = batch.index(b".parquet", end_bucket) + len(b".parquet")
    modified = bytearray(batch)
    if component in {"bucket", "userinfo", "port"}:
        modified[start] = {"bucket": ord("A"), "userinfo": ord("@"), "port": ord(":")}[component]
    elif component == "empty_key":
        # Preserve serialized string length while turning the object into a
        # malformed bucket (no separator remains).
        modified[end_bucket:end_path] = b"x" * (end_path - end_bucket)
    else:
        modified[end_path - 1] = {"query": ord("?"), "fragment": ord("#"), "control": 1}[component]
    start_event = len(tables.objects.events())
    with pytest.raises((ValueError, vane.Error), match="canonical s3://bucket/object"):
        execute_frozen(payload, tables, split_batch=bytes(modified))
    assert len(tables.objects.events()) == start_event
