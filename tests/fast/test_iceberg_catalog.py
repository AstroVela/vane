# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Installed base-package contracts that need neither REST nor a provider."""

import pytest

import vane
from vane.catalog import IcebergCatalog


class NoCatalogAccess:
    def sql(self, *_args, **_kwargs):
        raise AssertionError("invalid arguments must fail before catalog access")

    execute = sql


@pytest.mark.parametrize(
    "options",
    [
        {"name": ""},
        {"name": "catalog\0suffix"},
        {"warehouse": "warehouse\0suffix"},
        {"warehouse": None},
        {"endpoint": ""},
        {"endpoint": "file:///warehouse"},
        {"endpoint": "https://user:password@example.test/iceberg"},
        {"endpoint": "https://example.test/iceberg?token=secret"},
        {"endpoint": "https://example.test/iceberg#fragment"},
        {"endpoint": "http://example.test:99999/iceberg"},
        {"endpoint": "http://example.test/iceberg\n"},
        {"endpoint": "http://example .test/iceberg"},
        {"endpoint": "http://example.test/iceberg?"},
        {"endpoint": "http://example.test/iceberg#"},
    ],
)
def test_attach_validates_before_any_catalog_access(options):
    arguments = {"name": "lake", "endpoint": "http://example.test/iceberg", **options}
    with pytest.raises(ValueError):
        IcebergCatalog.attach(NoCatalogAccess(), **arguments)


@pytest.fixture
def local_connection(monkeypatch):
    import ray

    monkeypatch.setenv("VANE_RUNNER", "local")

    def forbid_ray(*_args, **_kwargs):
        raise AssertionError("local Iceberg catalog contracts must not initialize Ray")

    monkeypatch.setattr(ray, "init", forbid_ray)
    with vane.connect(config={"autoinstall_known_extensions": "false", "autoload_known_extensions": "false"}) as con:
        yield con


def test_missing_provider_fails_without_an_attachment(local_connection):
    with pytest.raises(RuntimeError, match="load the matching Iceberg provider"):
        IcebergCatalog.attach(local_connection, "lake", endpoint="http://127.0.0.1:1/iceberg")
    assert (
        local_connection.execute("SELECT database_name FROM duckdb_databases() WHERE type = 'iceberg'").fetchall() == []
    )


@pytest.mark.parametrize("name", ["missing", "memory"])
def test_constructor_rejects_missing_and_non_iceberg_catalogs(local_connection, name):
    with pytest.raises(ValueError, match="existing READ_ONLY Iceberg"):
        IcebergCatalog(local_connection, name)
    assert local_connection.execute("SELECT 42").fetchone() == (42,)


def test_read_only_duckdb_attachment_is_not_an_iceberg_catalog(local_connection, tmp_path):
    path = str(tmp_path / "ordinary.duckdb")
    path_literal = "'" + path.replace("'", "''") + "'"
    local_connection.execute(f"ATTACH {path_literal} AS ordinary")
    local_connection.execute("DETACH ordinary")
    local_connection.execute(f"ATTACH {path_literal} AS ordinary (READ_ONLY)")
    with pytest.raises(ValueError, match="existing READ_ONLY Iceberg"):
        IcebergCatalog(local_connection, "ordinary")


def test_attach_uses_local_connection_policy_after_environment_changes(local_connection, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    with pytest.raises(RuntimeError, match="load the matching Iceberg provider"):
        IcebergCatalog.attach(local_connection, "lake", endpoint="http://127.0.0.1:1/iceberg")


def test_ray_catalog_metadata_and_missing_provider_do_not_start_a_cluster(monkeypatch):
    import ray

    def forbid_ray(*_args, **_kwargs):
        raise AssertionError("Iceberg metadata and provider validation must not initialize Ray")

    monkeypatch.setattr(ray, "init", forbid_ray)
    monkeypatch.setenv("VANE_RUNNER", "ray")
    with vane.connect(config={"autoinstall_known_extensions": "false", "autoload_known_extensions": "false"}) as con:
        monkeypatch.setenv("VANE_RUNNER", "local")
        with pytest.raises(RuntimeError, match="load the matching Iceberg provider"):
            IcebergCatalog.attach(con, "lake", endpoint="http://127.0.0.1:1/iceberg")
        with pytest.raises(ValueError, match="existing READ_ONLY Iceberg"):
            IcebergCatalog(con, "lake")
