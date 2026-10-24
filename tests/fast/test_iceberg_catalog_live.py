# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Public local read interface qualified against real Gravitino Iceberg REST."""

import os
from pathlib import Path

import pytest
from gravitino_iceberg_helpers import iceberg_tables
from iceberg_feature_helpers import populate_iceberg_features

import vane
from vane.catalog import IcebergCatalog

pytestmark = pytest.mark.external_service


@pytest.fixture(scope="module")
def live_tables():
    endpoint = os.environ.get("VANE_TEST_GRAVITINO_ICEBERG_URL")
    shared = os.environ.get("VANE_TEST_GRAVITINO_ICEBERG_SHARED_DIR")
    if not endpoint and not shared:
        pytest.skip("set VANE_TEST_GRAVITINO_ICEBERG_URL and VANE_TEST_GRAVITINO_ICEBERG_SHARED_DIR")
    if not endpoint or not shared:
        pytest.fail("Iceberg acceptance requires both the REST URL and shared directory")
    with iceberg_tables(endpoint.rstrip("/"), Path(shared)) as tables:
        features = populate_iceberg_features(tables)
        yield endpoint, tables, features


@pytest.fixture
def local_catalog(monkeypatch, live_tables):
    import ray

    endpoint, tables, features = live_tables
    monkeypatch.setenv("VANE_RUNNER", "local")

    def forbid_ray(*_args, **_kwargs):
        raise AssertionError("native local Iceberg acceptance must not initialize Ray")

    monkeypatch.setattr(ray, "init", forbid_ray)
    with vane.connect(
        config={
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
            "allow_persistent_secrets": "false",
            "secret_directory": str(tables.directory / "secrets"),
        }
    ) as con:
        vane.load_installed_extension("iceberg", connection=con)
        catalog = IcebergCatalog.attach(con, "Lake.\"quoted'name", endpoint=endpoint, warehouse="")
        yield catalog, tables, features, endpoint


def test_public_table_returns_a_composable_native_relation(local_catalog):
    catalog, tables, _features, _endpoint = local_catalog
    relation = catalog.table(tables.namespace, "items")
    assert isinstance(relation, vane.DuckDBPyRelation)
    assert relation.order("id").fetchall() == list(tables.rows)
    assert relation.filter("id BETWEEN 2050 AND 2053").project("label").order("label NULLS FIRST").fetchall() == sorted(
        [(label,) for key, label in tables.rows if 2050 <= key <= 2053],
        key=lambda row: (row[0] is not None, row[0] or ""),
    )


def test_public_empty_table_preserves_schema(local_catalog):
    catalog, tables, _features, _endpoint = local_catalog
    relation = catalog.table(tables.namespace, "empty")
    assert relation.columns == ["id", "label"]
    assert [str(type_) for type_ in relation.types] == ["BIGINT", "VARCHAR"]
    assert relation.fetchall() == []


def test_snapshot_id_reads_the_selected_snapshot_and_rejects_missing_versions(local_catalog):
    catalog, tables, _features, _endpoint = local_catalog
    snapshots = tables.catalog.load_table((tables.namespace, "items")).snapshots()
    assert len(snapshots) == 2
    first = catalog.table(tables.namespace, "items", snapshot_id=snapshots[0].snapshot_id)
    assert first.order("id").fetchall() == list(tables.rows[:2050])
    latest = catalog.table(tables.namespace, "items", snapshot_id=tables.snapshot_id)
    assert latest.order("id").fetchall() == list(tables.rows)
    with pytest.raises(vane.Error, match="(?i)snapshot"):
        catalog.table(tables.namespace, "items", snapshot_id=2**63 - 1).fetchall()


def test_identity_partitions_include_unicode_and_null_values(local_catalog):
    catalog, tables, features, _endpoint = local_catalog
    feature = features["partitioned"]
    relation = catalog.table(tables.namespace, feature.name)
    assert relation.order("id").fetchall() == list(feature.rows)
    for predicate, expected in (
        ("region = 'east.点'", [row for row in feature.rows if row[2] == "east.点"]),
        ("region IS NULL", [row for row in feature.rows if row[2] is None]),
    ):
        assert relation.filter(predicate).order("id").fetchall() == expected


def test_schema_evolution_uses_field_ids_and_preserves_promoted_types(local_catalog):
    catalog, tables, features, _endpoint = local_catalog
    feature = features["evolved"]
    relation = catalog.table(tables.namespace, feature.name)
    assert relation.columns == ["id", "renamed", "added", "label"]
    assert [str(type_) for type_ in relation.types] == ["BIGINT", "VARCHAR", "VARCHAR", "VARCHAR"]
    assert relation.order("id").fetchall() == list(feature.rows)
    assert relation.filter("id = 1").project("renamed, label").fetchall() == [("old-one", None)]


@pytest.mark.parametrize("kind", ["position_deletes", "equality_deletes"])
def test_delete_files_apply_to_rows_projection_aggregation_and_old_snapshots(local_catalog, kind):
    catalog, tables, features, _endpoint = local_catalog
    feature = features[kind]
    relation = catalog.table(tables.namespace, feature.name)
    assert relation.order("id").fetchall() == list(feature.rows)
    assert relation.project("id").order("id").fetchall() == [(row[0],) for row in feature.rows]
    assert relation.aggregate("count(*), sum(id)").fetchone() == (
        len(feature.rows),
        sum(row[0] for row in feature.rows),
    )
    assert relation.filter("label IS NULL").project("id").order("id").fetchall() == [
        (row[0],) for row in feature.rows if row[1] is None
    ]
    assert catalog.table(tables.namespace, feature.name, snapshot_id=feature.initial_snapshot_id).order(
        "id"
    ).fetchall() == [(10, "a"), (11, None), (12, "c"), (13, "d")]


@pytest.mark.parametrize("snapshot_id", [False, True, 0, -1, 2**63, 1.5, "1"])
def test_snapshot_id_validation_is_explicit(local_catalog, snapshot_id):
    catalog, tables, _features, _endpoint = local_catalog
    with pytest.raises(ValueError, match="snapshot_id"):
        catalog.table(tables.namespace, "items", snapshot_id=snapshot_id)


@pytest.mark.parametrize(
    "namespace,name", [("", "items"), ("bad\0namespace", "items"), ("namespace", ""), ("namespace", "bad\0table")]
)
def test_table_names_are_validated(local_catalog, namespace, name):
    catalog, _tables, _features, _endpoint = local_catalog
    with pytest.raises(ValueError):
        catalog.table(namespace, name)


def test_names_are_identifiers_and_cannot_execute_statements(local_catalog):
    catalog, tables, _features, _endpoint = local_catalog
    with pytest.raises(vane.Error):
        catalog.table(tables.namespace, 'items"; DROP TABLE items; --').fetchall()
    assert catalog.table(tables.namespace, "items").aggregate("count(*)").fetchone() == (len(tables.rows),)


def test_existing_read_only_attachment_can_be_wrapped_with_native_case_insensitivity(local_catalog):
    catalog, tables, _features, _endpoint = local_catalog
    wrapped = IcebergCatalog(catalog.connection, catalog.name.upper())
    assert wrapped.table(tables.namespace, "items").aggregate("count(*)").fetchone() == (len(tables.rows),)


def test_non_ascii_catalog_names_keep_distinct_attachment_identity(local_catalog):
    catalog, tables, _features, endpoint = local_catalog
    first = IcebergCatalog.attach(catalog.connection, "LakeÄ", endpoint=endpoint)
    second = IcebergCatalog.attach(catalog.connection, "Lakeä", endpoint=endpoint)
    first.detach()
    with pytest.raises(ValueError, match="detached or replaced"):
        first.table(tables.namespace, "items")
    assert second.table(tables.namespace, "items").aggregate("count(*)").fetchone() == (len(tables.rows),)


def test_writable_iceberg_attachment_is_rejected(local_catalog):
    catalog, _tables, _features, endpoint = local_catalog
    catalog.connection.execute(
        "ATTACH '' AS writable (TYPE iceberg, ENDPOINT $endpoint, AUTHORIZATION_TYPE 'none')", {"endpoint": endpoint}
    )
    with pytest.raises(ValueError, match="existing READ_ONLY Iceberg"):
        IcebergCatalog(catalog.connection, "writable")


def test_public_attachment_enforces_read_only_without_a_new_snapshot(local_catalog):
    catalog, tables, _features, _endpoint = local_catalog
    assert catalog.connection.execute(
        "SELECT readonly FROM duckdb_databases() WHERE database_name=$name", {"name": catalog.name}
    ).fetchone() == (True,)
    qualified = ".".join('"' + part.replace('"', '""') + '"' for part in (catalog.name, tables.namespace, "items"))
    with pytest.raises(vane.Error, match="(?i)read.only"):
        catalog.connection.execute(f"INSERT INTO {qualified} VALUES (999999, 'forbidden')")
    assert tables.catalog.load_table((tables.namespace, "items")).current_snapshot().snapshot_id == tables.snapshot_id


def test_detach_keeps_connection_open_and_invalidates_stale_wrappers(local_catalog):
    catalog, tables, _features, endpoint = local_catalog
    catalog.detach()
    assert catalog.connection.execute("SELECT 42").fetchone() == (42,)
    with pytest.raises(ValueError, match="detached or replaced"):
        catalog.table(tables.namespace, "items")
    replacement = IcebergCatalog.attach(catalog.connection, catalog.name, endpoint=endpoint)
    with pytest.raises(ValueError, match="detached or replaced"):
        catalog.detach()
    assert replacement.table(tables.namespace, "items").aggregate("count(*)").fetchone() == (len(tables.rows),)


def test_connection_policy_survives_an_environment_change(local_catalog, monkeypatch):
    catalog, tables, _features, _endpoint = local_catalog
    monkeypatch.setenv("VANE_RUNNER", "ray")
    assert catalog.table(tables.namespace, "items").aggregate("count(*)").fetchone() == (len(tables.rows),)
