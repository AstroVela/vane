# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Gravitino 1.3 acceptance against an explicitly provisioned real service."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import vane
from tests.fast.test_gravitino import gravitino_http as gravitino_http
from tests.fast.test_gravitino import load_gravitino
from tests.fast.test_gravitino_write import configure_s3, read_literal_batch
from vane.catalog import GravitinoCatalog

pytestmark = pytest.mark.external_service


@pytest.fixture
def live_gravitino():
    endpoint = os.environ.get("VANE_TEST_GRAVITINO_URL")
    shared_directory = os.environ.get("VANE_TEST_GRAVITINO_SHARED_DIR")
    if not endpoint or not shared_directory:
        pytest.skip("set VANE_TEST_GRAVITINO_URL and VANE_TEST_GRAVITINO_SHARED_DIR for live Gravitino tests")
    endpoint = endpoint.rstrip("/")
    metalake = "vane_ci_" + uuid.uuid4().hex
    directory = Path(shared_directory).resolve() / metalake
    directory.mkdir(mode=0o755, parents=True)

    def api(method, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(
            endpoint + "/api" + path,
            data=body,
            method=method,
            headers={"Accept": "application/vnd.gravitino.v1+json", "Content-Type": "application/json"},
        )
        with urlopen(request, timeout=15) as response:
            result = json.load(response)
        assert result["code"] == 0
        return result

    created = False
    try:
        api("POST", "/metalakes", {"name": metalake, "comment": "Vane CI", "properties": {}})
        created = True
        api(
            "POST",
            f"/metalakes/{metalake}/catalogs",
            {"name": "media", "type": "FILESET", "provider": "fileset", "comment": "Vane CI", "properties": {}},
        )
        yield endpoint, metalake, directory, api
    finally:
        try:
            if created:
                # Only remove this test's uniquely named metalake. All Filesets
                # in these tests are EXTERNAL; their files belong to the fixture.
                api("DELETE", f"/metalakes/{metalake}?force=true")
        finally:
            shutil.rmtree(directory)


def attach_live(connection, endpoint, metalake):
    load_gravitino(connection)
    return GravitinoCatalog.attach(connection, "media", endpoint=endpoint, metalake=metalake, catalog="media")


@pytest.mark.parametrize("format", ["parquet", "csv"])
def test_live_gravitino_local_batch_write(live_gravitino, format):
    endpoint, metalake, directory, _api = live_gravitino
    output = directory / "output"
    output.mkdir()
    with vane.connect() as connection:
        catalog = attach_live(connection, endpoint, metalake)
        catalog.create_schema("features")
        catalog.create_fileset("features", "vectors", storage_location=str(output))
        result = catalog.write_files(
            connection.sql("SELECT i, '中文' AS label FROM range(8) AS t(i)"),
            "features",
            "vectors",
            format=format,
        )
        assert result.rows_written == 8
        assert result.bytes_written == sum(Path(file).stat().st_size for file in result.files)
        restored = (
            connection.read_parquet(list(result.files))
            if format == "parquet"
            else connection.read_csv(list(result.files), header=True)
        )
        assert restored.order("i").fetchall() == [(i, "中文") for i in range(8)]


@pytest.mark.parametrize("runner", ["local-fast", pytest.param("ray", marks=pytest.mark.real_ray)])
@pytest.mark.parametrize("format", ["parquet", "csv"])
def test_live_gravitino_s3_partitioned_batch(request, monkeypatch, live_gravitino, gravitino_http, runner, format):
    if runner == "ray":
        request.getfixturevalue("ray_local")
        monkeypatch.delenv("VANE_RUNNER", raising=False)
    endpoint, metalake, _directory, api = live_gravitino
    _state, storage_endpoint, _storage_directory = gravitino_http
    # The real service owns metadata; only Vane accesses the object fixture.
    # No test storage credentials or Docker-to-host networking are needed in
    # Gravitino. Local Fileset acceptance still exercises its default FS ops.
    api(
        "POST",
        f"/metalakes/{metalake}/catalogs",
        {
            "name": "objects",
            "type": "FILESET",
            "provider": "fileset",
            "properties": {"disable-filesystem-ops": "true"},
        },
    )
    with vane.connect() as connection:
        configure_s3(connection, storage_endpoint)
        load_gravitino(connection)
        catalog = GravitinoCatalog.attach(connection, "media", endpoint=endpoint, metalake=metalake, catalog="objects")
        catalog.create_schema("features")
        catalog.create_fileset("features", "vectors", storage_location=f"s3a://bucket/{metalake}")
        for empty in (False, True):
            relation = connection.sql(
                "SELECT i, CASE WHEN i%2=0 THEN '中文/路径' ELSE NULL END AS part FROM range(8) t(i)"
            )
            if empty:
                relation = relation.filter("false")
            result = catalog.write_files(relation, "features", "vectors", format=format, partition_by=["part"])
            expected = [] if empty else [(i, "中文/路径" if i % 2 == 0 else None) for i in range(8)]
            assert result.rows_written == len(expected)
            assert result.files
            assert read_literal_batch(connection, result) == expected
            assert bool(result.manifest_path) == (runner == "ray")


def test_live_gravitino_metadata_crud(live_gravitino):
    endpoint, metalake, directory, api = live_gravitino
    with vane.connect() as connection:
        catalog = attach_live(connection, endpoint, metalake)
        assert catalog.metadata()["name"] == "media"
        catalog.create_schema("clips")
        connection.execute("PREPARE filesets AS SELECT name FROM gravitino_filesets('media', 'clips')")
        assert connection.execute("EXECUTE filesets").fetchall() == []
        connection.execute("CREATE SCHEMA IF NOT EXISTS media.clips")
        assert catalog.list_schemas() == ["clips"]
        catalog.alter_schema("clips", [{"@type": "setProperty", "property": "owner", "value": "context"}])
        assert catalog.load_schema("clips")["properties"]["owner"] == "context"
        catalog.alter_schema("clips", [{"@type": "removeProperty", "property": "owner"}])
        assert "owner" not in catalog.load_schema("clips")["properties"]

        catalog.alter_catalog(
            [
                {"@type": "updateComment", "newComment": "updated catalog"},
                {"@type": "setProperty", "property": "owner", "value": "context"},
            ]
        )
        assert catalog.metadata()["comment"] == "updated catalog"
        assert catalog.metadata()["properties"]["owner"] == "context"
        catalog.alter_catalog([{"@type": "removeProperty", "property": "owner"}])
        assert "owner" not in catalog.metadata()["properties"]

        source = directory / "sample.txt"
        source.write_text("preserve EXTERNAL contents")
        catalog.create_fileset("clips", "example", storage_location=directory.as_uri())
        assert connection.execute("EXECUTE filesets").fetchall() == [("example",)]
        assert catalog.list_filesets("clips") == ["example"]
        catalog.alter_fileset(
            "clips",
            "example",
            [
                {"@type": "updateComment", "newComment": "updated fileset"},
                {"@type": "setProperty", "property": "owner", "value": "context"},
                {"@type": "rename", "newName": "renamed"},
            ],
        )
        assert catalog.list_filesets("clips") == ["renamed"]
        metadata = catalog.load_fileset("clips", "renamed")
        assert metadata["comment"] == "updated fileset"
        assert metadata["properties"]["owner"] == "context"
        catalog.alter_fileset("clips", "renamed", [{"@type": "removeProperty", "property": "owner"}])
        assert "owner" not in catalog.load_fileset("clips", "renamed")["properties"]
        remote = api("GET", f"/metalakes/{metalake}/catalogs/media/schemas/clips/filesets/renamed")
        assert remote["fileset"]["comment"] == "updated fileset"
        catalog.drop_fileset("clips", "renamed")
        assert connection.execute("EXECUTE filesets").fetchall() == []
        assert catalog.list_filesets("clips") == []
        assert source.read_text() == "preserve EXTERNAL contents"
        catalog.drop_schema("clips")
        assert catalog.list_schemas() == []
        connection.execute("DROP SCHEMA IF EXISTS media.clips")


@pytest.mark.parametrize("path", ["*.txt", "one.txt"])
def test_live_prepared_files_refresh(live_gravitino, path):
    endpoint, metalake, directory, api = live_gravitino
    original = directory / "original"
    original.mkdir()
    (original / "one.txt").write_text("original file")
    with vane.connect() as connection:
        catalog = attach_live(connection, endpoint, metalake)
        catalog.create_schema("clips")
        catalog.create_fileset("clips", "example", storage_location=str(original))
        connection.execute(
            f"PREPARE remote_files AS SELECT url FROM gravitino_files('media', 'clips', 'example', '{path}') ORDER BY url"
        )
        assert connection.execute("EXECUTE remote_files").fetchall() == [(str(original / "one.txt"),)]
        if path == "*.txt":
            (original / "two.txt").write_text("new file")
            assert connection.execute("EXECUTE remote_files").fetchall() == [
                (str(original / "one.txt"),),
                (str(original / "two.txt"),),
            ]
            (original / "two.txt").unlink()
            assert connection.execute("EXECUTE remote_files").fetchall() == [(str(original / "one.txt"),)]

        # A different client changes the remote Fileset without touching the
        # querying connection's DuckDB catalog or removing the old files.
        resource = f"/metalakes/{metalake}/catalogs/media/schemas/clips/filesets"
        api("DELETE", resource + "/example")
        with pytest.raises(vane.IOException, match="404"):
            connection.execute("EXECUTE remote_files").fetchall()
        replacement = directory / "replacement%20#[1]"
        replacement.mkdir()
        (replacement / "one.txt").write_text("replacement file")
        api(
            "POST",
            resource,
            {"name": "example", "type": "EXTERNAL", "storageLocation": str(replacement), "properties": {}},
        )
        assert connection.execute("EXECUTE remote_files").fetchall() == [(str(replacement / "one.txt"),)]
        assert (original / "one.txt").read_text() == "original file"


def test_live_gravitino_rejects_schema_comment_updates(live_gravitino):
    endpoint, metalake, _directory, api = live_gravitino
    with vane.connect() as connection:
        catalog = attach_live(connection, endpoint, metalake)
        catalog.create_schema("clips")
        changes = [{"@type": "updateComment", "newComment": "unsupported"}]
        # Check the real server's contract independently of the connector and
        # the fault-injection fixture, which previously accepted this operation.
        with pytest.raises(HTTPError) as error:
            api("PUT", f"/metalakes/{metalake}/catalogs/media/schemas/clips", {"updates": changes})
        assert error.value.code == 400
        error.value.close()
        with pytest.raises(vane.NotImplementedException, match="updateComment"):
            catalog.alter_schema("clips", changes)
        assert catalog.load_schema("clips").get("comment") != "unsupported"


@pytest.mark.parametrize("runner", ["local-fast", pytest.param("ray", marks=pytest.mark.real_ray)])
def test_live_gravitino_fileset_reads(request, monkeypatch, live_gravitino, runner):
    if runner == "ray":
        request.getfixturevalue("ray_local")
        monkeypatch.delenv("VANE_RUNNER", raising=False)
    endpoint, metalake, directory, _api = live_gravitino
    parent = directory
    directory = parent / "video%20clips#[1]"
    directory.mkdir()
    contents = b"real Gravitino Fileset contents"
    (directory / "sample.txt").write_bytes(contents)
    for name in ("video clips#[1]", "video%20clips#1"):
        neighbor = parent / name
        neighbor.mkdir()
        (neighbor / "sample.txt").write_bytes(b"must not read a different Fileset root")
    with vane.connect() as connection:
        catalog = attach_live(connection, endpoint, metalake)
        catalog.create_schema("clips")
        catalog.create_fileset(
            "clips",
            "example",
            # Gravitino's Hadoop Path registration takes an unescaped path;
            # passing an encoded space makes it look for a literal "%20".
            storage_locations={"primary": str(directory)},
            properties={"default-location-name": "primary"},
        )
        assert catalog.list_filesets("clips") == ["example"]
        location = catalog.load_fileset("clips", "example")["storageLocations"]["primary"]
        assert location.endswith("/video%20clips#[1]")
        rows = (
            catalog.files("clips", "example", "*.txt")
            .project("file_content_id(file_enrich(file, ['checksum']))")
            .fetchall()
        )
        assert rows == [("file-content-v1:checksum:sha256:" + hashlib.sha256(contents).hexdigest(),)]
        if runner == "local-fast":
            with vane.open_file("gvfs://fileset/media/clips/example/sample.txt", "rb", connection=connection) as stream:
                assert stream.read() == contents
