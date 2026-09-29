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
from tests.fast.test_gravitino import load_gravitino
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


def test_live_gravitino_metadata_crud(live_gravitino):
    endpoint, metalake, directory, api = live_gravitino
    with vane.connect() as connection:
        catalog = attach_live(connection, endpoint, metalake)
        assert catalog.metadata()["name"] == "media"
        catalog.create_schema("clips")
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
        assert catalog.list_filesets("clips") == []
        assert source.read_text() == "preserve EXTERNAL contents"
        catalog.drop_schema("clips")
        assert catalog.list_schemas() == []
        connection.execute("DROP SCHEMA IF EXISTS media.clips")


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
    directory = directory / "video clips"
    directory.mkdir()
    contents = b"real Gravitino Fileset contents"
    (directory / "sample.txt").write_bytes(contents)
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
        rows = (
            catalog.files("clips", "example", "*.txt")
            .project("file_content_id(file_enrich(file, ['checksum']))")
            .fetchall()
        )
        assert rows == [("file-content-v1:checksum:sha256:" + hashlib.sha256(contents).hexdigest(),)]
        if runner == "local-fast":
            with vane.open_file("gvfs://fileset/media/clips/example/sample.txt", "rb", connection=connection) as stream:
                assert stream.read() == contents
