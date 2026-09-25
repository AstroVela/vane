# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import subprocess

import pytest
from packaging.version import Version

from vane_packaging.extension_wheel import _extension_distribution_version_from_digest
from vane_packaging.media_version import (
    VERSION_FILE,
    identity_version,
    runtime_format,
    source_version,
)


def test_runtime_version_uses_the_iceberg_provider_encoder():
    fmt = runtime_format()
    identity = {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "vane_version": "0.2.0.dev612",
    }
    actual = identity_version(identity)
    assert actual == _extension_distribution_version_from_digest(
        "0.2.0.dev612", hashlib.sha256(fmt.canonical_json(identity)).hexdigest()
    )
    assert Version(actual).local is None
    assert fmt.version(actual) == actual
    assert identity_version({**identity, "git_commit": "b" * 40}) != actual
    dirty = {**identity, "git_dirty": True, "git_dirty_sha256": "1" * 64}
    assert identity_version(dirty) != actual
    assert identity_version({**dirty, "git_dirty_sha256": "2" * 64}) != identity_version(dirty)
    assert fmt.runtime_namespace(dirty) != fmt.runtime_namespace({**dirty, "git_dirty_sha256": "2" * 64})
    with pytest.raises(ValueError):
        identity_version({**identity, "git_dirty": True})
    assert identity_version({**identity, "vane_version": "0.2.0.dev613"}) != actual


def test_exported_version_survives_without_git_and_rejects_mismatches(tmp_path, monkeypatch):
    fmt = runtime_format()
    identity = {
        "git_commit": "c" * 40,
        "git_dirty": False,
        "vane_version": "0.2.0.dev612",
    }
    document = {**identity, "version": identity_version(identity)}
    (tmp_path / VERSION_FILE).write_bytes(fmt.canonical_json(document))

    def no_git(*args, **kwargs):
        raise AssertionError("an exported source SDK must not consult Git")

    monkeypatch.setattr(subprocess, "check_output", no_git)
    monkeypatch.setattr(subprocess, "run", no_git)
    assert source_version(tmp_path) == document
    document["git_commit"] = "d" * 40
    (tmp_path / VERSION_FILE).write_bytes(fmt.canonical_json(document))
    with pytest.raises(ValueError, match="Git identity"):
        source_version(tmp_path)


def test_dirty_exported_provenance_is_preserved_and_checked_without_git(tmp_path):
    fmt = runtime_format()
    identity = {
        "git_commit": "a" * 40,
        "git_dirty": True,
        "git_dirty_sha256": "b" * 64,
        "vane_version": "0.2.0.dev612",
    }
    document = {**identity, "version": identity_version(identity)}
    (tmp_path / VERSION_FILE).write_bytes(fmt.canonical_json(document))
    assert source_version(tmp_path) == document
    document["git_dirty_sha256"] = "c" * 64
    (tmp_path / VERSION_FILE).write_bytes(fmt.canonical_json(document))
    with pytest.raises(ValueError, match="Git identity"):
        source_version(tmp_path)


def test_runtime_manifest_binds_dirty_provenance_to_library_namespace():
    from tests.fast.test_native_runtime import manifest

    fmt = runtime_format()
    value = manifest()
    value.update(git_dirty=True, git_dirty_sha256="a" * 64)
    value["version"] = identity_version(value)
    value["source"]["filename"] = f"vane_media_runtime-{value['version']}.tar.gz"
    name, record = next(iter(value["files"].items()))
    original = value["namespace"]
    value["namespace"] = fmt.runtime_namespace(value)
    value["files"] = {name.replace(original, value["namespace"]): record}
    assert fmt.parse_manifest(fmt.canonical_json(value)) == value
    value["git_dirty_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="namespace differs"):
        fmt.parse_manifest(fmt.canonical_json(value))
    value.pop("git_dirty_sha256")
    with pytest.raises(ValueError, match="SHA-256"):
        fmt.parse_manifest(fmt.canonical_json(value))


def test_checkout_version_records_commits_and_dirty_state(tmp_path):
    project = tmp_path / "packages/vane-media-runtime"
    project.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[tool.setuptools_scm]\nlocal_scheme = "no-local-version"\n')
    source = project / "source.txt"
    source.write_text("first\n")

    def git(*arguments):
        return subprocess.check_output(
            [
                "git",
                "-c",
                "user.name=Vane tests",
                "-c",
                "user.email=tests@example.invalid",
                *arguments,
            ],
            cwd=tmp_path,
            text=True,
        ).strip()

    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "initial source")
    git("tag", "v0.1.0")
    clean = source_version(project)
    assert clean["git_commit"] == git("rev-parse", "HEAD")
    assert clean["git_dirty"] is False
    source.write_text("changed\n")
    dirty = source_version(project)
    assert dirty["git_commit"] == clean["git_commit"]
    assert dirty["git_dirty"] is True
    assert dirty["version"] != clean["version"]
    assert source_version(project) == dirty
    source.write_text("another change\n")
    other = source_version(project)
    assert other["version"] != dirty["version"]
    before_index = (tmp_path / ".git/index").read_bytes()
    assert source_version(project) == other
    assert (tmp_path / ".git/index").read_bytes() == before_index
    git("add", ".")
    assert source_version(project) == other
    extra = project / "new-file.txt"
    extra.write_text("new source\n")
    untracked = source_version(project)
    assert untracked["version"] != other["version"]
    git("add", ".")
    assert source_version(project) == untracked
    extra.chmod(0o755)
    executable = source_version(project)
    assert executable["version"] != untracked["version"]
    extra.unlink()
    deleted = source_version(project)
    assert deleted["version"] != executable["version"]
    extra.symlink_to("source.txt")
    linked = source_version(project)
    assert linked["version"] != deleted["version"]
    extra.unlink()
    extra.symlink_to("elsewhere.txt")
    assert source_version(project)["version"] != linked["version"]
    extra.unlink()
    git("add", ".")
    git("commit", "-qm", "change source")
    changed = source_version(project)
    assert changed["git_commit"] != clean["git_commit"]
    assert changed["git_dirty"] is False
    assert changed["version"] not in {clean["version"], dirty["version"]}
