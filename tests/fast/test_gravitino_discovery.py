# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Metalake discovery uses native metadata reads, independently of Fileset I/O."""

from __future__ import annotations

import copy
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from urllib.parse import quote

import pytest

import vane
from tests.fast.test_gravitino import attach, load_gravitino
from tests.fast.test_gravitino import gravitino_http as gravitino_http
from vane.catalog import GravitinoCatalog, GravitinoIdentifier, GravitinoMetalake


@pytest.fixture
def discovery_http(request):
    # Ordinary base-only fast shards need no provider or HTTP/Ray fixture.
    if os.environ.get("VANE_TEST_GRAVITINO") not in {"static", "provider"}:
        pytest.skip("set VANE_TEST_GRAVITINO=static or provider for native discovery tests")
    state, endpoint, directory = request.getfixturevalue("gravitino_http")
    state["discovery"] = {
        "metalake": "lake",
        "catalogs": {
            "media": {"name": "media", "type": "FILESET", "provider": "fileset", "properties": {}},
            "iceberg": {
                "name": "iceberg",
                "type": "RELATIONAL",
                "provider": "lakehouse-iceberg",
                "properties": {"warehouse": "s3://bucket/warehouse", "jdbc-password": "catalog-secret"},
            },
            "postgres.db": {
                "name": "postgres.db",
                "type": "RELATIONAL",
                "provider": "jdbc-postgresql",
                "properties": {},
            },
        },
        "schemas": {
            "media": {},
            "iceberg": {"schema.点": {"name": "schema.点", "properties": {"owner": "unit"}}},
            "postgres.db": {},
        },
        "tables": {
            ("iceberg", "schema.点"): {
                "table.name": {
                    "name": "table.name",
                    "columns": [{"name": "id", "type": "long", "nullable": True}],
                    "properties": {"format": "iceberg", "access_token": "table-secret", "owner": "unit"},
                }
            }
        },
    }
    yield state, endpoint, directory


@pytest.fixture
def discovery(discovery_http):
    _state, endpoint, _directory = discovery_http
    with vane.connect() as connection:
        load_gravitino(connection)
        with GravitinoMetalake.connect(connection, endpoint=endpoint, metalake="lake", token="fixture-token") as client:
            yield client


def requests_since(state, offset):
    return list(state["requests"])[offset:]


def test_catalog_listing_is_independent_and_does_not_load_each_catalog(discovery, discovery_http):
    state, _endpoint, _directory = discovery_http
    before = len(state["requests"])
    result = discovery.list_catalogs()
    assert result == [
        GravitinoIdentifier(("lake",), "media"),
        GravitinoIdentifier(("lake",), "iceberg"),
        GravitinoIdentifier(("lake",), "postgres.db"),
    ]
    requests = requests_since(state, before)
    assert 1 <= len(requests) <= 2  # Native preparation can rebind this fresh source.
    assert set(requests) == {("GET", "/prefix/api/metalakes/lake/catalogs?details=false", None)}
    attached = discovery.connection.execute("SELECT type FROM duckdb_databases() WHERE NOT internal").fetchall()
    assert ("gravitino",) not in attached
    assert ("gravitino_metalake",) in attached
    with pytest.raises(FrozenInstanceError):
        result[0].name = "changed"
    assert list(state["object_requests"]) == []


def test_full_identifier_limit_survives_multiple_native_chunks(discovery, discovery_http):
    state, _endpoint, _directory = discovery_http
    config = copy.deepcopy(state["discovery"])
    config["catalogs"] = {
        f"catalog{i}": {"name": f"catalog{i}", "type": "RELATIONAL", "provider": "future"} for i in range(4096)
    }
    state["discovery"] = config
    before = len(state["requests"])
    result = discovery.list_catalogs()
    assert len(result) == 4096
    assert len(set(result)) == 4096
    assert result[-1] == GravitinoIdentifier(("lake",), "catalog4095")
    assert 1 <= len(requests_since(state, before)) <= 2


def test_details_expose_provider_format_and_separate_execution_capabilities(discovery):
    catalog = discovery.load_catalog("iceberg")
    assert catalog.identifier == GravitinoIdentifier(("lake",), "iceberg")
    assert catalog.kind == "catalog"
    assert catalog.catalog_type == "RELATIONAL"
    assert catalog.provider == "lakehouse-iceberg"
    assert catalog.format == "iceberg"
    assert catalog.capabilities.metadata_read == "supported"
    assert catalog.capabilities.table_read == catalog.capabilities.table_write == "unverified"
    assert catalog.metadata["properties"]["jdbc-password"] == "[REDACTED]"
    postgres = discovery.load_catalog("postgres.db")
    assert postgres.format is None
    assert postgres.capabilities.table_read == postgres.capabilities.table_write == "unsupported"
    media = discovery.load_catalog("media")
    assert media.capabilities.table_read == "unsupported"


def test_schema_and_table_lists_preserve_whole_identifiers(discovery, discovery_http):
    state, _endpoint, _directory = discovery_http
    before = len(state["requests"])
    assert discovery.list_schemas("iceberg") == [GravitinoIdentifier(("lake", "iceberg"), "schema.点")]
    assert 1 <= len(requests_since(state, before)) <= 2
    assert discovery.load_schema("iceberg", "schema.点").metadata["properties"] == {"owner": "unit"}
    before = len(state["requests"])
    assert discovery.list_tables("iceberg", "schema.点") == [
        GravitinoIdentifier(("lake", "iceberg", "schema.点"), "table.name")
    ]
    requests = requests_since(state, before)
    assert 2 <= len(requests) <= 4
    assert {path for _method, path, _body in requests} == {
        "/prefix/api/metalakes/lake/catalogs/iceberg",
        "/prefix/api/metalakes/lake/catalogs/iceberg/schemas/" + quote("schema.点", safe="") + "/tables",
    }
    table = discovery.load_table("iceberg", "schema.点", "table.name")
    assert table.identifier == GravitinoIdentifier(("lake", "iceberg", "schema.点"), "table.name")
    assert table.kind == "table"
    assert table.provider == "lakehouse-iceberg"
    assert table.metadata["columns"] == [{"name": "id", "type": "long", "nullable": True}]
    assert table.metadata["properties"] == {"format": "iceberg", "access_token": "[REDACTED]", "owner": "unit"}


def test_unknown_provider_stays_unverified(discovery, discovery_http):
    state, _endpoint, _directory = discovery_http
    config = copy.deepcopy(state["discovery"])
    config["catalogs"]["iceberg"]["provider"] = "future-provider"
    state["discovery"] = config
    assert discovery.load_catalog("iceberg").capabilities.table_read == "unverified"


def test_concurrent_discovery_results_remain_owned_by_each_relation(discovery):
    names = ["iceberg", "postgres.db"] * 4
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(discovery.load_catalog, names))
    assert [result.identifier.name for result in results] == names
    assert all(result.metadata["name"] == name for result, name in zip(results, names, strict=True))


def test_credential_properties_are_redacted_without_hiding_schema_fields(discovery, discovery_http):
    state, _endpoint, _directory = discovery_http
    config = copy.deepcopy(state["discovery"])
    properties = config["tables"][("iceberg", "schema.点")]["table.name"]["properties"]
    keys = ["jdbc-password", "S3_SECRET_ACCESS_KEY", "Authorization", "access.key.id", "credential", "private_key"]
    properties.update({key: "sensitive-test-value" for key in keys})
    state["discovery"] = config
    result = discovery.load_table("iceberg", "schema.点", "table.name")
    assert all(result.metadata["properties"][key] == "[REDACTED]" for key in keys)
    assert result.metadata["columns"][0]["name"] == "id"
    assert result.metadata["properties"]["owner"] == "unit"


def test_failed_connect_leaves_no_native_attachment(discovery_http):
    state, endpoint, _directory = discovery_http
    with vane.connect() as connection:
        load_gravitino(connection)
        before = set(connection.execute("SELECT database_name FROM duckdb_databases()").fetchall())
        with pytest.raises(vane.CatalogException, match="not found"):
            GravitinoMetalake.connect(connection, endpoint=endpoint, metalake="missing", token="fixture-token")
        assert set(connection.execute("SELECT database_name FROM duckdb_databases()").fetchall()) == before
        state["discovery_fault"] = "wrong-resource"
        with pytest.raises(vane.IOException, match="metalake"):
            GravitinoMetalake.connect(connection, endpoint=endpoint, metalake="lake", token="fixture-token")
        assert set(connection.execute("SELECT database_name FROM duckdb_databases()").fetchall()) == before


def test_empty_lists_and_closed_handle(discovery, discovery_http):
    state, _endpoint, _directory = discovery_http
    assert discovery.list_schemas("postgres.db") == []
    config = copy.deepcopy(state["discovery"])
    config["tables"][("iceberg", "schema.点")] = {}
    state["discovery"] = config
    assert discovery.list_tables("iceberg", "schema.点") == []
    name = discovery._name
    discovery.close()
    discovery.close()
    assert discovery.connection.execute(
        "SELECT count(*) FROM duckdb_databases() WHERE database_name = ?", [name]
    ).fetchone() == (0,)
    with pytest.raises(RuntimeError, match="closed"):
        discovery.list_catalogs()


@pytest.mark.parametrize("entry", ["python", "sql"])
def test_tokens_are_private_and_metadata_attachment_is_read_only(discovery_http, entry):
    state, endpoint, _directory = discovery_http
    with vane.connect() as connection:
        load_gravitino(connection)
        if entry == "python":
            client = GravitinoMetalake.connect(connection, endpoint=endpoint, metalake="lake", token="fixture-token")
        else:
            connection.execute(
                "ATTACH 'lake' AS discovery (TYPE gravitino_metalake, READ_ONLY, ENDPOINT $endpoint, TOKEN $token)",
                {"endpoint": endpoint, "token": "fixture-token"},
            )
            client = GravitinoMetalake(connection, "discovery", "lake")
        assert client.load_catalog("iceberg").provider == "lakehouse-iceberg"
        options, readonly = connection.execute(
            "SELECT options, readonly FROM duckdb_databases() WHERE database_name = ?", [client._name]
        ).fetchone()
        assert "fixture-token" not in str(options)
        assert not any(str(key).lower() == "token" for key in options)
        assert readonly is True
        before = len(state["requests"])
        with pytest.raises(vane.Error):
            connection.execute(f'CREATE SCHEMA "{client._name}".forbidden')
        with pytest.raises(vane.InvalidInputException, match="not a Gravitino catalog"):
            GravitinoCatalog(connection, client._name).metadata()
        assert len(state["requests"]) == before
        assert all(request[0] == "GET" for request in state["requests"])
        client.close()


def test_native_sql_requires_read_only_and_cannot_borrow_a_fileset_attachment(discovery_http):
    state, endpoint, _directory = discovery_http
    with vane.connect() as connection:
        load_gravitino(connection)
        before = len(state["requests"])
        with pytest.raises(vane.BinderException, match="READ_ONLY"):
            connection.execute(
                "ATTACH 'lake' AS discovery (TYPE gravitino_metalake, ENDPOINT $endpoint, TOKEN $token)",
                {"endpoint": endpoint, "token": "fixture-token"},
            )
        assert len(state["requests"]) == before
        catalog = attach(connection, endpoint)
        assert catalog.metadata()["type"] == "FILESET"
        with pytest.raises(vane.InvalidInputException, match="metalake discovery"):
            connection.execute("SELECT * FROM gravitino_metalake_catalogs('media')")


@pytest.mark.parametrize("operation", ["list_schemas", "load_catalog", "list_tables", "load_table"])
@pytest.mark.parametrize("name", ["", ".", "..", "wrong/name", "wrong\\name", "bad\0name", "x" * 1025])
def test_invalid_resource_names_are_rejected_before_metadata_requests(discovery, discovery_http, operation, name):
    state, _endpoint, _directory = discovery_http
    arguments = [name]
    if operation in {"list_tables", "load_table"}:
        arguments.append("schema.点")
    if operation == "load_table":
        arguments.append("table.name")
    before = len(state["requests"])
    with pytest.raises((vane.InvalidInputException, ValueError)):
        getattr(discovery, operation)(*arguments)
    assert len(state["requests"]) == before


@pytest.mark.parametrize("field", ["timeout_ms", "max_response_bytes"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "100", 16 * 1024 * 1024 + 1])
def test_invalid_limits_do_not_touch_the_connection(field, value):
    with pytest.raises(ValueError, match=field):
        GravitinoMetalake.connect(None, endpoint="http://localhost", metalake="lake", **{field: value})


@pytest.mark.parametrize(
    "operation,args",
    [
        ("load_catalog", ("missing",)),
        ("list_schemas", ("missing",)),
        ("load_schema", ("iceberg", "missing")),
        ("load_table", ("iceberg", "schema.点", "missing")),
    ],
)
def test_missing_resources_are_catalog_errors(discovery, operation, args):
    with pytest.raises(vane.CatalogException, match="not found"):
        getattr(discovery, operation)(*args)


def test_table_discovery_rejects_non_relational_catalogs(discovery, discovery_http):
    state, _endpoint, _directory = discovery_http
    before = len(state["requests"])
    with pytest.raises(vane.NotImplementedException, match="RELATIONAL"):
        discovery.list_tables("media", "unused")
    assert len(requests_since(state, before)) == 1


@pytest.mark.parametrize(
    "fault", ["missing-list", "bad-list", "too-many", "duplicate", "bad-entry", "bad-name", "pagination", "has-more"]
)
def test_invalid_or_incomplete_catalog_lists_fail_explicitly(discovery, discovery_http, fault):
    state, _endpoint, _directory = discovery_http
    state["discovery_fault"] = fault
    with pytest.raises(vane.IOException):
        discovery.list_catalogs()


@pytest.mark.parametrize(
    "fault",
    [
        "missing-list",
        "bad-list",
        "too-many",
        "duplicate",
        "bad-entry",
        "bad-name",
        "bad-namespace",
        "pagination",
        "has-more",
    ],
)
def test_invalid_schema_identifiers_fail_explicitly(discovery, discovery_http, fault):
    state, _endpoint, _directory = discovery_http
    state["discovery_fault"] = fault
    with pytest.raises(vane.IOException):
        discovery.list_schemas("iceberg")


@pytest.mark.parametrize("fault", ["wrong-resource", "missing-resource-name", "missing-provider", "bad-properties"])
def test_inconsistent_detail_responses_are_rejected(discovery, discovery_http, fault):
    state, _endpoint, _directory = discovery_http
    state["discovery_fault"] = fault
    with pytest.raises(vane.IOException):
        discovery.load_catalog("iceberg")


@pytest.mark.parametrize("fault", ["invalid-json", "oversized", "compressed", "redirect"])
def test_transport_guards_are_shared_with_existing_client(discovery, discovery_http, fault):
    state, _endpoint, _directory = discovery_http
    state["fault"] = fault
    # The default bound is larger than the fixture's ordinary oversized body.
    if fault == "oversized":
        state["fault"] = None
        with GravitinoMetalake.connect(
            discovery.connection, endpoint=_endpoint, metalake="lake", token="fixture-token", max_response_bytes=256
        ) as bounded:
            state["fault"] = fault
            with pytest.raises(vane.IOException, match="bounds"):
                bounded.list_catalogs()
    else:
        with pytest.raises(vane.IOException):
            discovery.list_catalogs()
    assert not any("should-not-follow" in request[1] for request in state["requests"])


def test_disabling_external_access_stops_discovery_without_io(discovery, discovery_http):
    state, _endpoint, _directory = discovery_http
    discovery.connection.execute("SET enable_external_access = false")
    before = len(state["requests"])
    with pytest.raises(vane.PermissionException, match="external_access"):
        discovery.list_catalogs()
    assert len(state["requests"]) == before


def test_prepared_discovery_refreshes_without_a_cross_execution_cache(discovery, discovery_http):
    state, _endpoint, _directory = discovery_http
    discovery.connection.execute(
        f"PREPARE discovery_names AS SELECT name FROM gravitino_metalake_catalogs('{discovery._name}')"
    )
    assert len(discovery.connection.execute("EXECUTE discovery_names").fetchall()) == 3
    config = copy.deepcopy(state["discovery"])
    config["catalogs"]["new.catalog"] = {
        "name": "new.catalog",
        "type": "RELATIONAL",
        "provider": "future",
        "properties": {},
    }
    state["discovery"] = config
    assert ("new.catalog",) in discovery.connection.execute("EXECUTE discovery_names").fetchall()
    assert len(discovery.list_catalogs()) == 4


def test_names_are_encoded_once_and_never_split_on_dots(discovery, discovery_http):
    state, _endpoint, _directory = discovery_http
    config = copy.deepcopy(state["discovery"])
    name = "db.百分%20#?"
    config["catalogs"][name] = {"name": name, "type": "RELATIONAL", "provider": "future", "properties": {}}
    state["discovery"] = config
    result = discovery.load_catalog(name)
    assert result.identifier == GravitinoIdentifier(("lake",), name)
    assert state["requests"][-1][1].endswith("/catalogs/" + quote(name, safe=""))


def test_serialized_discovery_contains_only_redacted_rows(discovery, discovery_http):
    state, _endpoint, _directory = discovery_http
    relation = discovery.connection.sql(
        f"SELECT * FROM gravitino_metalake_table('{discovery._name}', 'iceberg', 'schema.点', 'table.name')"
    )
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "gravitino-discovery")
    assert plan.__getstate__()[3]["attached_databases"] == []
    payload = pickle.dumps(plan)
    assert b"fixture-token" not in payload
    assert b"catalog-secret" not in payload
    assert b"table-secret" not in payload
    assert _endpoint.encode() not in payload
    assert b"[REDACTED]" in payload
    transported = pickle.loads(payload)
    discovery.close()
    before = len(state["requests"])
    state["fault"] = "invalid-json"
    with vane.connect() as target:
        load_gravitino(target)
        transported.to_physical_plan(target)
    assert len(state["requests"]) == before


@pytest.mark.real_ray
@pytest.mark.parametrize(
    "operation,args",
    [
        ("list_catalogs", ()),
        ("list_schemas", ("iceberg",)),
        ("list_schemas", ("postgres.db",)),
        ("load_table", ("iceberg", "schema.点", "table.name")),
    ],
)
def test_ray_discovery_uses_frozen_rows_without_worker_metadata_access(
    ray_local, discovery_http, monkeypatch, operation, args
):
    from vane.runners.ray.runner import RayRunner

    monkeypatch.delenv("VANE_RUNNER", raising=False)
    state, endpoint, _directory = discovery_http
    calls = []
    run = RayRunner.run_iter

    def capture(runner, plan):
        payload = pickle.dumps(plan)
        assert b"fixture-token" not in payload
        assert b"catalog-secret" not in payload
        assert b"table-secret" not in payload
        assert endpoint.encode() not in payload
        assert plan.__getstate__()[3]["attached_databases"] == []
        calls.append(plan.idx())
        state["fault"] = "invalid-json"
        yield from run(runner, plan)

    monkeypatch.setattr(RayRunner, "run_iter", capture)
    with vane.connect() as connection:
        load_gravitino(connection)
        with GravitinoMetalake.connect(connection, endpoint=endpoint, metalake="lake", token="fixture-token") as client:
            result = getattr(client, operation)(*args)
        assert len(calls) == 1
        if operation == "list_catalogs":
            assert len(result) == 3
        elif operation == "list_schemas":
            expected = [GravitinoIdentifier(("lake", "iceberg"), "schema.点")] if args == ("iceberg",) else []
            assert result == expected
        else:
            assert result.metadata["properties"]["access_token"] == "[REDACTED]"
