# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import vane_packaging.extension_materials as materials

EXPRESSION = "Apache-2.0 AND LGPL-2.1-or-later"
IDENTITY = {"name": "sample", "artifact_sha256": "a" * 64, "license_expression": EXPRESSION}


@pytest.fixture
def inventory():
    return {
        "libraries": [
            {
                "name": "soxr",
                "version": "0.1.3",
                "license": "LGPL-2.1-or-later",
                "source": "sources/soxr.tar.xz",
                "build_recipe": "recipes/soxr.cmake",
                "patches": [],
            }
        ],
        "application": ["sources/application.tar.xz"],
        "materials_license_expression": EXPRESSION,
        "build_instructions": "BUILD.md",
        "relink_instructions": "RELINK.md",
        "relink_verification": "relink.log",
    }


def test_materials_bind_all_declared_files_to_the_artifact(inventory):
    files = {
        path: f"unit test fixture: {path}\n".encode()
        for path in materials.inventory_paths(inventory, name="sample", license_expression=EXPRESSION)
    }
    manifest = materials.prepare_manifest(inventory, files.__getitem__, **IDENTITY)
    assert materials.validate_materials(manifest, files.__getitem__, **IDENTITY) == files
    for path in files:
        changed = {**files, path: files[path].replace(b"unit", b"fake")}
        with pytest.raises(ValueError, match="size/SHA-256 mismatch"):
            materials.validate_materials(manifest, changed.__getitem__, **IDENTITY)


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "other"),
        ("artifact_sha256", "b" * 64),
        ("license_expression", "MIT"),
    ],
)
def test_materials_reject_stale_artifact_or_license_identity(inventory, field, value):
    manifest = materials.prepare_manifest(inventory, lambda _: b"fixture", **IDENTITY)
    with pytest.raises(ValueError, match="artifact identity"):
        materials.validate_materials(
            manifest, lambda _: pytest.fail("read before identity validation"), **{**IDENTITY, field: value}
        )


@pytest.mark.parametrize("library", ["ffmpeg", "libsndfile", "soxr", "mpg123", "mp3lame"])
def test_native_audio_requires_every_lgpl_library_including_transitive_codecs(inventory, library):
    inventory["libraries"] = [
        {**inventory["libraries"][0], "name": name}
        for name in ("ffmpeg", "libsndfile", "soxr", "mpg123", "mp3lame")
        if name != library
    ]
    with pytest.raises(ValueError, match=f"missing LGPL libraries:.*{library}"):
        materials.prepare_manifest(inventory, lambda _: b"fixture", **{**IDENTITY, "name": "audio"})


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        "/absolute",
        "a//b",
        "a/./b",
        "a/../b",
        "a\\b",
        "a:b",
        "CON.txt",
        "a/NUL",
        "x.",
    ],
)
def test_materials_reject_nonportable_or_escaping_paths(inventory, path):
    inventory["application"] = [path]
    with pytest.raises(ValueError, match="unsafe extension materials path"):
        materials.prepare_manifest(inventory, lambda _: pytest.fail("read an unsafe path"), **IDENTITY)


@pytest.mark.parametrize("paths", [["APP", "app"], ["app", "app/source"]])
def test_materials_reject_case_and_parent_collisions(inventory, paths):
    inventory["application"] = paths
    with pytest.raises(ValueError, match="distinct file paths|file/parent conflict"):
        materials.prepare_manifest(inventory, lambda _: b"fixture", **IDENTITY)


def test_materials_reject_hidden_lgpl_licenses(inventory):
    with pytest.raises(ValueError, match="License-Expression must include"):
        materials.prepare_manifest(inventory, lambda _: b"fixture", **{**IDENTITY, "license_expression": "MIT"})


def test_audio_inventory_preserves_exact_mpg123_and_lame_grants(inventory):
    licenses = {
        "ffmpeg": "LGPL-2.1-or-later",
        "libsndfile": "LGPL-2.1-or-later",
        "soxr": "LGPL-2.1-or-later",
        "mpg123": "LGPL-2.1-only",
        "mp3lame": "LGPL-2.0-or-later",
    }
    inventory["libraries"] = [
        {**inventory["libraries"][0], "name": name, "license": license_id} for name, license_id in licenses.items()
    ]
    identity = {**IDENTITY, "name": "audio"}
    with pytest.raises(ValueError, match="must include its LGPL dependencies"):
        materials.prepare_manifest(inventory, lambda _: b"fixture", **identity)
    identity["license_expression"] = EXPRESSION + " AND LGPL-2.1-only AND LGPL-2.0-or-later"
    inventory["materials_license_expression"] = identity["license_expression"]
    manifest = materials.prepare_manifest(inventory, lambda _: b"fixture", **identity)
    assert materials.validate_materials(manifest, lambda _: b"fixture", **identity)


@pytest.mark.parametrize(
    "source_license", ["GPL-2.0-or-later", "GPL-3.0-or-later", "GPL-2.0-or-later WITH Bison-exception-2.2"]
)
def test_source_archive_licenses_are_not_hidden_by_the_binary_license(inventory, source_license):
    inventory["materials_license_expression"] = EXPRESSION + " AND " + source_license
    with pytest.raises(ValueError, match="source/build materials licenses"):
        materials.prepare_manifest(inventory, lambda _: b"fixture", **IDENTITY)
    identity = {**IDENTITY, "license_expression": inventory["materials_license_expression"]}
    manifest = materials.prepare_manifest(inventory, lambda _: b"fixture", **identity)
    assert materials.validate_materials(manifest, lambda _: b"fixture", **identity)


@pytest.mark.parametrize(
    "expression", ["MIT OR (Apache-2.0 AND LGPL-2.1-or-later)", "Apache-2.0 AND (LGPL-2.1-or-later OR MIT)"]
)
def test_material_licenses_cannot_be_made_optional_by_or(inventory, expression):
    with pytest.raises(ValueError, match="must include"):
        materials.prepare_manifest(inventory, lambda _: b"fixture", **{**IDENTITY, "license_expression": expression})


def test_material_licenses_can_be_required_in_every_alternative(inventory):
    expression = "(Apache-2.0 AND LGPL-2.1-or-later AND MIT) OR (Apache-2.0 AND LGPL-2.1-or-later AND BSD-3-Clause)"
    identity = {**IDENTITY, "license_expression": expression}
    manifest = materials.prepare_manifest(inventory, lambda _: b"fixture", **identity)
    assert materials.validate_materials(manifest, lambda _: b"fixture", **identity)


@pytest.mark.parametrize(
    "expression",
    [
        "Apache-2.0",
        "Apache-2.0 OR LGPL-2.1-or-later",
        "Apache-2.0 AND (LGPL-2.1-or-later OR MIT)",
        "(Apache-2.0 AND LGPL-2.1-or-later) OR MIT",
    ],
)
def test_library_license_must_be_mandatory_in_the_materials_expression(inventory, expression):
    identity = {**IDENTITY, "license_expression": EXPRESSION + " AND MIT"}
    manifest = json.loads(materials.prepare_manifest(inventory, lambda _: b"fixture", **identity))
    inventory["materials_license_expression"] = expression
    with pytest.raises(ValueError, match="materials_license_expression must include"):
        materials.prepare_manifest(
            inventory, lambda _: pytest.fail("read before source license validation"), **identity
        )
    manifest["materials_license_expression"] = expression
    with pytest.raises(ValueError, match="materials_license_expression must include"):
        materials.validate_materials(
            materials.encode_manifest(manifest),
            lambda _: pytest.fail("read before source license validation"),
            **identity,
        )


def test_library_license_can_be_mandatory_in_every_material_alternative(inventory):
    inventory["materials_license_expression"] = "(Apache-2.0 AND LGPL-2.1-or-later) OR (MIT AND LGPL-2.1-or-later)"
    identity = {**IDENTITY, "license_expression": EXPRESSION + " AND MIT"}
    manifest = materials.prepare_manifest(inventory, lambda _: b"fixture", **identity)
    assert materials.validate_materials(manifest, lambda _: b"fixture", **identity)


@pytest.mark.parametrize("role", ["build_instructions", "relink_instructions"])
def test_materials_require_independent_relink_evidence(inventory, role):
    inventory[role] = inventory["relink_verification"]
    with pytest.raises(ValueError, match="must use distinct files"):
        materials.prepare_manifest(inventory, lambda _: pytest.fail("read aliased evidence"), **IDENTITY)


@pytest.mark.parametrize("role", ["application", "source", "build_recipe", "patches"])
def test_materials_documents_cannot_also_stand_for_code(inventory, role):
    document = inventory["build_instructions"]
    if role == "application":
        inventory[role] = [document]
    else:
        inventory["libraries"][0][role] = [document] if role == "patches" else document
    with pytest.raises(ValueError, match="must not alias source or recipe"):
        materials.prepare_manifest(inventory, lambda _: pytest.fail("read document as code"), **IDENTITY)


def test_materials_allow_an_archive_to_contain_multiple_code_roles(inventory):
    archive = "sources/combined.tar.xz"
    inventory["application"] = [archive]
    inventory["libraries"] = [
        {**inventory["libraries"][0], "name": name, "source": archive, "build_recipe": archive, "patches": [archive]}
        for name in ("soxr", "libsndfile")
    ]
    manifest = materials.prepare_manifest(inventory, lambda _: b"fixture", **IDENTITY)
    assert set(materials.validate_materials(manifest, lambda _: b"fixture", **IDENTITY)) == {
        archive,
        "BUILD.md",
        "RELINK.md",
        "relink.log",
    }


def test_materials_reject_duplicate_entries_within_a_file_list(inventory):
    inventory["application"] *= 2
    with pytest.raises(ValueError, match="must not repeat paths"):
        materials.prepare_manifest(inventory, lambda _: pytest.fail("read duplicate file"), **IDENTITY)


@pytest.mark.parametrize(
    "mutation", ["schema-bool", "size-bool", "missing-record", "extra-record", "huge-record", "hash"]
)
def test_materials_reject_invalid_records_before_reading_files(inventory, mutation):
    manifest = json.loads(materials.prepare_manifest(inventory, lambda _: b"fixture", **IDENTITY))
    if mutation == "schema-bool":
        manifest["schema_version"] = True
    elif mutation == "size-bool":
        manifest["files"]["BUILD.md"]["size"] = True
    elif mutation == "missing-record":
        del manifest["files"]["BUILD.md"]
    elif mutation == "extra-record":
        manifest["files"]["unused"] = manifest["files"]["BUILD.md"]
    elif mutation == "huge-record":
        manifest["files"]["BUILD.md"]["size"] = materials.MAX_MATERIAL_BYTES + 1
    else:
        manifest["files"]["BUILD.md"]["sha256"] = "not-a-hash"
    with pytest.raises(ValueError):
        materials.validate_materials(
            materials.encode_manifest(manifest), lambda _: pytest.fail("read before record validation"), **IDENTITY
        )


def test_materials_enforce_aggregate_bounds_before_reading_files(inventory, monkeypatch):
    manifest = materials.prepare_manifest(inventory, lambda _: b"fixture", **IDENTITY)
    monkeypatch.setattr(materials, "MAX_MATERIALS_BYTES", 10)
    with pytest.raises(ValueError, match="aggregate size budget"):
        materials.validate_materials(manifest, lambda _: pytest.fail("read over budget"), **IDENTITY)


@pytest.mark.parametrize("contents", [b'{"files": {}, "files": {}}', b"[]", b"\xff", b"{" * 2000])
def test_materials_reject_ambiguous_or_malformed_json(contents):
    with pytest.raises(ValueError):
        materials.parse_json(contents)


def test_material_reads_reject_missing_empty_large_and_symlink_files(tmp_path):
    with pytest.raises(ValueError, match="cannot read"):
        materials.read_material_file(tmp_path, "missing")
    source = tmp_path / "source"
    source.write_bytes(b"")
    with pytest.raises(ValueError, match="nonempty bounded regular file"):
        materials.read_material_file(tmp_path, "source")
    source.write_bytes(b"123")
    with pytest.raises(ValueError, match="nonempty bounded regular file"):
        materials.read_material_file(tmp_path, "source", max_bytes=2)
    alias = tmp_path / "alias"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="symlinks"):
        materials.read_material_file(tmp_path, "alias")
    with pytest.raises(ValueError):
        materials.read_material_file(tmp_path, "alias/child")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFOs")
def test_material_reads_reject_fifos_without_waiting_for_a_writer(tmp_path):
    os.mkfifo(tmp_path / "fifo")
    with pytest.raises(ValueError, match="regular file"):
        materials.read_material_file(tmp_path, "fifo")


def test_prepare_cli_produces_a_manifest_consumable_by_the_wheel_builder(tmp_path, inventory):
    artifact = tmp_path / "sample.duckdb_extension"
    artifact.write_bytes(b"unit test native artifact fixture")
    directory = tmp_path / "materials"
    directory.mkdir()
    for path in materials.inventory_paths(inventory, name="sample", license_expression=EXPRESSION):
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"unit test fixture: {path}".encode())
    inventory_file = tmp_path / "inventory.json"
    inventory_file.write_text(json.dumps(inventory))
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(Path(__file__).resolve().parents[2] / "scripts/prepare_extension_materials.py"),
            "--artifact",
            str(artifact),
            "--extension-name",
            "sample",
            "--license-expression",
            EXPRESSION,
            "--directory",
            str(directory),
            "--inventory",
            str(inventory_file),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    materials.validate_materials(
        (directory / materials.MANIFEST_NAME).read_bytes(),
        lambda path: materials.read_material_file(directory, path),
        **{**IDENTITY, "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
    )
