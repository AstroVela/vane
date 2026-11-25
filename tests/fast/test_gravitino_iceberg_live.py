# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""M4a native local Iceberg qualification against an explicit real REST service."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from gravitino_iceberg_helpers import iceberg_tables

import vane

pytestmark = pytest.mark.external_service


@pytest.fixture(scope="module")
def live_iceberg():
    endpoint = os.environ.get("VANE_TEST_GRAVITINO_ICEBERG_URL")
    shared = os.environ.get("VANE_TEST_GRAVITINO_ICEBERG_SHARED_DIR")
    if not endpoint and not shared:
        pytest.skip("set VANE_TEST_GRAVITINO_ICEBERG_URL and VANE_TEST_GRAVITINO_ICEBERG_SHARED_DIR")
    if not endpoint or not shared:
        pytest.fail("Iceberg qualification requires both the REST URL and shared directory")
    # Once explicitly selected, a missing SDK/provider/service is a failure,
    # rather than a successful-looking qualification with all tests skipped.
    with iceberg_tables(endpoint.rstrip("/"), Path(shared)) as tables:
        yield endpoint, tables


@pytest.fixture
def local_iceberg(monkeypatch, live_iceberg):
    import ray

    endpoint, tables = live_iceberg
    monkeypatch.setenv("VANE_RUNNER", "local")

    def forbid_ray(*_args, **_kwargs):
        raise AssertionError("native local Iceberg qualification must not initialize Ray")

    monkeypatch.setattr(ray, "init", forbid_ray)
    with vane.connect(
        config={
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
            "allow_persistent_secrets": "false",
            "secret_directory": str(tables.directory / "secrets"),
        }
    ) as con:
        assert (
            con.execute(
                "SELECT extension_name FROM duckdb_extensions() WHERE loaded AND extension_name IN ('avro', 'iceberg')"
            ).fetchall()
            == []
        )
        vane.load_installed_extension("iceberg", connection=con)
        con.execute(
            "ATTACH '' AS qualification (TYPE iceberg, ENDPOINT $endpoint, AUTHORIZATION_TYPE 'none', READ_ONLY)",
            {"endpoint": endpoint},
        )
        yield con, tables, f'qualification."{tables.namespace}"'


def test_provider_loads_its_avro_dependency(local_iceberg):
    con, _tables, _namespace = local_iceberg
    assert set(
        con.execute(
            "SELECT extension_name FROM duckdb_extensions() "
            "WHERE loaded AND extension_name IN ('avro', 'iceberg', 'parquet')"
        ).fetchall()
    ) == {("avro",), ("iceberg",), ("parquet",)}


def test_rest_table_matches_independent_client_and_ignores_uncommitted_file(local_iceberg):
    con, tables, namespace = local_iceberg
    assert (tables.directory / "items" / "data" / "uncommitted.parquet").is_file()
    assert con.sql(f"SELECT id, label FROM {namespace}.items ORDER BY id").fetchall() == list(tables.rows)


def test_native_projection_filter_and_aggregate(local_iceberg):
    con, tables, namespace = local_iceberg
    selected = [(key, value) for key, value in tables.rows if key % 11 == 0]
    assert con.execute(f"SELECT count(*), sum(id) FROM {namespace}.items WHERE id % 11 = 0").fetchone() == (
        len(selected),
        sum(key for key, _value in selected),
    )
    assert con.sql(f"SELECT label FROM {namespace}.items WHERE id BETWEEN 2050 AND 2053 ORDER BY id").fetchall() == [
        (value,) for key, value in tables.rows if 2050 <= key <= 2053
    ]


def test_empty_rest_table_preserves_schema(local_iceberg):
    con, _tables, namespace = local_iceberg
    relation = con.sql(f"SELECT * FROM {namespace}.empty")
    assert relation.columns == ["id", "label"]
    assert [str(type_) for type_ in relation.types] == ["BIGINT", "VARCHAR"]
    assert relation.fetchall() == []


def test_read_only_attachment_rejects_insert_without_a_new_snapshot(local_iceberg):
    con, tables, namespace = local_iceberg
    with pytest.raises(vane.Error, match="(?i)read.only"):
        con.execute(f"INSERT INTO {namespace}.items VALUES (123456, 'must not commit')")
    observed = tables.catalog.load_table((tables.namespace, "items"))
    assert observed.current_snapshot().snapshot_id == tables.snapshot_id
    assert observed.scan().to_arrow().num_rows == len(tables.rows)


def test_scan_plan_transport_rejects_unimplemented_iceberg_serialization(local_iceberg):
    con, _tables, namespace = local_iceberg
    relation = con.sql(f"SELECT id, label FROM {namespace}.items")
    # The plan factory transports native errors in its existing ValueError JSON
    # envelope; the underlying Iceberg error must remain explicit.
    with pytest.raises(ValueError, match="IcebergScan serialization not implemented"):
        vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "iceberg-qualification")
