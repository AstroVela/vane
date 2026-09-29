# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Native Gravitino REST contract and Fileset integration, without cloud services."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import time
from urllib.parse import urlsplit

import pytest
from gravitino_helpers import serve_gravitino

import vane
from vane.catalog import GravitinoCatalog


@pytest.fixture
def gravitino_http(tmp_path):
    # Relation transformations can bind while holding Python's GIL. Keep the
    # native HTTP peer in a separate process, like an actual Gravitino service.
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        state = manager.dict(
            catalog=manager.dict(name="media", type="FILESET", properties={}),
            schemas=manager.dict(),
            filesets=manager.dict(),
            requests=manager.list(),
            fault=None,
            object=b"s3 file contents",
            object_requests=manager.list(),
        )
        parent, child = context.Pipe(duplex=False)
        server = context.Process(target=serve_gravitino, args=(state, child), daemon=True)
        server.start()
        child.close()
        try:
            assert parent.poll(10), "Gravitino HTTP fixture did not start"
            port = parent.recv()
            yield state, f"http://127.0.0.1:{port}/prefix", tmp_path
        finally:
            parent.close()
            server.terminate()
            server.join(timeout=5)
            if server.is_alive():
                server.kill()
                server.join(timeout=5)


def load_gravitino(connection):
    mode = os.environ.get("VANE_TEST_GRAVITINO")
    if mode == "static":
        connection.execute("LOAD gravitino")
    elif mode == "provider":
        vane.load_installed_extension("gravitino", connection=connection)
    else:
        pytest.skip("set VANE_TEST_GRAVITINO=static or provider for the native integration tests")


@pytest.fixture
def gravitino_connection():
    with vane.connect() as connection:
        load_gravitino(connection)
        yield connection


def attach(connection, endpoint, name="media", **options):
    return GravitinoCatalog.attach(
        connection, name, endpoint=endpoint, metalake="lake", catalog="media", token="fixture-token", **options
    )


def seed_fileset(connection, endpoint, directory):
    catalog = attach(connection, endpoint)
    catalog.create_schema("clips")
    (directory / "one.txt").write_text("first file")
    (directory / "two.txt").write_text("second file")
    catalog.create_fileset("clips", "demo", storage_location=directory.as_uri(), properties={"tag": "中文"})
    return catalog


def test_metadata_crud_and_native_catalog(gravitino_connection, gravitino_http):
    state, endpoint, directory = gravitino_http
    catalog = seed_fileset(gravitino_connection, endpoint, directory)
    assert catalog.list_schemas() == ["clips"]
    gravitino_connection.execute("CREATE SCHEMA IF NOT EXISTS media.clips")
    assert catalog.list_filesets("clips") == ["demo"]
    assert catalog.load_fileset("clips", "demo")["properties"] == {"tag": "中文"}
    schemas = gravitino_connection.execute(
        "SELECT schema_name FROM duckdb_schemas() WHERE database_name='media'"
    ).fetchall()
    assert schemas == [("clips",)]
    catalog.alter_schema("clips", [{"@type": "setProperty", "property": "owner", "value": "agent"}])
    assert catalog.load_schema("clips")["properties"] == {"owner": "agent"}
    catalog.alter_catalog([{"@type": "updateComment", "newComment": "agent context"}])
    assert catalog.metadata()["comment"] == "agent context"
    catalog.alter_fileset("clips", "demo", [{"@type": "rename", "newName": "renamed"}])
    assert catalog.list_filesets("clips") == ["renamed"]
    catalog.drop_fileset("clips", "renamed")
    assert catalog.list_filesets("clips") == []
    catalog.drop_schema("clips")
    assert catalog.list_schemas() == []
    assert len([r for r in state["requests"] if r[0] == "DELETE"]) == 2
    gravitino_connection.execute("DROP SCHEMA IF EXISTS media.clips")
    assert (directory / "one.txt").exists()


def test_fileset_native_file_reads(gravitino_connection, gravitino_http):
    _state, endpoint, directory = gravitino_http
    catalog = seed_fileset(gravitino_connection, endpoint, directory)
    rows = catalog.files("clips", "demo", "*.txt").project("url, object_size").order("url").fetchall()
    assert rows == [(str(directory / "one.txt"), 10), (str(directory / "two.txt"), 11)]
    assert catalog.files("clips", "demo", "one.txt").project("file_size(file)").fetchone() == (10,)
    with vane.open_file("gvfs://fileset/media/clips/demo/one.txt", "rb", connection=gravitino_connection) as stream:
        assert stream.read() == b"first file"
    with pytest.raises(vane.NotImplementedException):
        gravitino_connection.execute("COPY (SELECT 1) TO 'gvfs://fileset/media/clips/demo/out.csv'")
    nested = directory / "nested"
    nested.mkdir()
    (nested / "three.txt").write_text("third file")
    assert catalog.files("clips", "demo", recursive=False).count("*").fetchone() == (2,)
    assert catalog.files("clips", "demo", recursive=True).count("*").fetchone() == (3,)


@pytest.mark.parametrize("path", ["../one.txt", "a/../../one.txt", "/one.txt", "%2e%2e/one.txt", "a\\one.txt"])
def test_invalid_relative_paths_fail_before_http(gravitino_connection, gravitino_http, path):
    state, endpoint, directory = gravitino_http
    catalog = seed_fileset(gravitino_connection, endpoint, directory)
    count = len(state["requests"])
    with pytest.raises(vane.InvalidInputException):
        catalog.files("clips", "demo", path)
    assert len(state["requests"]) == count


def test_named_location_requires_an_explicit_default(gravitino_connection, gravitino_http):
    _state, endpoint, directory = gravitino_http
    catalog = attach(gravitino_connection, endpoint)
    catalog.create_schema("clips")
    (directory / "example.txt").write_text("content")
    catalog.create_fileset("clips", "demo", storage_locations={"primary": directory.as_uri()})
    with pytest.raises(vane.InvalidInputException, match="unknown"):
        catalog.files("clips", "demo", "example.txt")
    catalog.alter_fileset(
        "clips", "demo", [{"@type": "setProperty", "property": "default-location-name", "value": "primary"}]
    )
    assert catalog.files("clips", "demo", "example.txt").project("object_size").fetchone() == (7,)


def test_read_only_and_explicit_transaction_do_not_mutate(gravitino_connection, gravitino_http):
    state, endpoint, directory = gravitino_http
    seed_fileset(gravitino_connection, endpoint, directory)
    read_only = attach(gravitino_connection, endpoint, "readonly", read_only=True)
    count = len([r for r in state["requests"] if r[0] != "GET"])
    with pytest.raises(vane.Error, match="READ_ONLY"):
        read_only.drop_fileset("clips", "demo")
    gravitino_connection.begin()
    try:
        with pytest.raises(vane.TransactionException, match="auto-commit"):
            GravitinoCatalog(gravitino_connection, "media").drop_fileset("clips", "demo")
    finally:
        gravitino_connection.rollback()
    assert len([r for r in state["requests"] if r[0] != "GET"]) == count


@pytest.mark.parametrize("fault, message", [("oversized", "bounds"), ("compressed", "compressed"), ("redirect", "302")])
def test_bounded_http_and_no_redirects(gravitino_connection, gravitino_http, fault, message):
    state, endpoint, _directory = gravitino_http
    catalog = attach(gravitino_connection, endpoint, max_response_bytes=1024)
    state["fault"] = fault
    before = len(state["requests"])
    with pytest.raises(vane.IOException, match=message):
        catalog.metadata()
    assert len(state["requests"]) == before + 1


def test_deadline_and_failed_writes_are_not_retried(gravitino_connection, gravitino_http):
    state, endpoint, directory = gravitino_http
    catalog = seed_fileset(gravitino_connection, endpoint, directory)
    state["fault"] = "failed-write"
    before = len(state["requests"])
    with pytest.raises(vane.IOException, match="outcome is unknown"):
        catalog.drop_fileset("clips", "demo")
    assert len(state["requests"]) == before + 1
    state["fault"] = None
    fast = attach(gravitino_connection, endpoint, "fast", timeout_ms=50)
    state["fault"] = "slow"
    started = time.monotonic()
    with pytest.raises(vane.IOException, match="[Tt]ime"):
        fast.metadata()
    assert time.monotonic() - started < 0.2


def test_names_and_properties_are_not_sql_or_url_syntax(gravitino_connection, gravitino_http):
    _state, endpoint, directory = gravitino_http
    catalog = attach(gravitino_connection, endpoint, "odd'\"alias")
    schema = "片段' ?#"
    name = "set' ; SELECT 42--"
    catalog.create_schema(schema)
    catalog.create_fileset(schema, name, storage_location=directory.as_uri())
    assert catalog.list_filesets(schema) == [name]
    catalog.alter_fileset(schema, name, [{"@type": "updateComment", "newComment": "x'\ny"}])
    assert catalog.load_fileset(schema, name)["comment"] == "x'\ny"


def test_unsupported_catalog_fails_at_attach(gravitino_connection, gravitino_http):
    state, endpoint, _directory = gravitino_http
    state["catalog"]["type"] = "RELATIONAL"
    with pytest.raises(vane.NotImplementedException, match="format-specific"):
        attach(gravitino_connection, endpoint)


@pytest.mark.parametrize("fault", ["invalid-json", "missing-outcome", "not-dropped"])
def test_mutation_requires_confirmation(gravitino_connection, gravitino_http, fault):
    state, endpoint, directory = gravitino_http
    catalog = seed_fileset(gravitino_connection, endpoint, directory)
    state["fault"] = fault
    before = len(state["requests"])
    message = "nothing was dropped" if fault == "not-dropped" else "outcome is unknown"
    with pytest.raises(vane.Error, match=message):
        catalog.drop_fileset("clips", "demo")
    assert len(state["requests"]) == before + 1


@pytest.mark.parametrize(
    "options",
    [
        {"properties": {"size": 1}},
        {"comment": 1},
        {"kind": "OTHER"},
        {"storage_location": ""},
        {"storage_locations": {"primary": ""}},
        {"storage_location": "/one", "storage_locations": {"primary": "/two"}},
    ],
)
def test_invalid_metadata_is_not_sent(gravitino_connection, gravitino_http, options):
    state, endpoint, _directory = gravitino_http
    catalog = attach(gravitino_connection, endpoint)
    before = len(state["requests"])
    with pytest.raises(vane.InvalidInputException):
        catalog.create_fileset("unused", "example", **options)
    assert len(state["requests"]) == before


def test_disabled_external_access_blocks_metadata(gravitino_connection, gravitino_http):
    state, endpoint, _directory = gravitino_http
    catalog = attach(gravitino_connection, endpoint)
    gravitino_connection.execute("SET enable_external_access = false")
    before = len(state["requests"])
    with pytest.raises(vane.PermissionException, match="external_access"):
        catalog.metadata()
    assert len(state["requests"]) == before


@pytest.mark.real_ray
def test_default_ray_fileset_reads_and_single_metadata_mutation(ray_local, monkeypatch, gravitino_http):
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    state, endpoint, directory = gravitino_http
    with vane.connect() as connection:
        load_gravitino(connection)
        catalog = seed_fileset(connection, endpoint, directory)
        assert catalog.list_filesets("clips") == ["demo"]
        rows = catalog.files("clips", "demo", "*.txt").project("url, file_size(file)").order("url").fetchall()
        assert rows == [(str(directory / "one.txt"), 10), (str(directory / "two.txt"), 11)]

        @vane.func(return_dtype="BLOB")
        def read_contents(value):
            with value.open() as reader:
                return reader.read()

        contents = catalog.files("clips", "demo", "*.txt").select(read_contents(vane.col("file"))).fetchall()
        assert sorted(contents) == [(b"first file",), (b"second file",)]
        assert len([r for r in state["requests"] if r[0] == "POST" and r[1].endswith("/filesets")]) == 1
        catalog.drop_fileset("clips", "demo")
        assert len([r for r in state["requests"] if r[0] == "DELETE"]) == 1


@pytest.mark.parametrize("runner", ["local-fast", pytest.param("ray", marks=pytest.mark.real_ray)])
def test_fileset_s3_reads_reuse_storage_configuration(request, monkeypatch, gravitino_http, runner):
    if runner == "ray":
        request.getfixturevalue("ray_local")
        monkeypatch.delenv("VANE_RUNNER", raising=False)
    state, endpoint, _directory = gravitino_http
    with vane.connect() as connection:
        load_gravitino(connection)
        connection.execute("SET http_proxy = ''")
        connection.execute("SET s3_endpoint = ?", [urlsplit(endpoint).netloc])
        connection.execute("SET s3_access_key_id = 'fileset-test-access'")
        connection.execute("SET s3_secret_access_key = 'fileset-test-secret'")
        connection.execute("SET s3_region = 'us-east-1'")
        connection.execute("SET s3_use_ssl = false")
        connection.execute("SET s3_url_style = 'path'")
        catalog = attach(connection, endpoint)
        catalog.create_schema("clips")
        catalog.create_fileset("clips", "objects", storage_location="s3a://bucket")

        rows = (
            catalog.files("clips", "objects", "object.txt")
            .project("file_content_id(file_enrich(file, ['checksum']))")
            .fetchall()
        )
        digest = hashlib.sha256(state["object"]).hexdigest()
        assert rows == [(f"file-content-v1:checksum:sha256:{digest}",)]
        if runner == "local-fast":
            with vane.open_file("gvfs://fileset/media/clips/objects/object.txt", "rb", connection=connection) as stream:
                assert stream.read() == state["object"]
        assert any(
            method == "GET" and authorization and "Credential=fileset-test-access/" in authorization
            for method, authorization in state["object_requests"]
        )
