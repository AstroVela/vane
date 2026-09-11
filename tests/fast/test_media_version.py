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
    assert identity_version({**identity, "git_dirty": True}) != actual
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
    git("add", ".")
    git("commit", "-qm", "change source")
    changed = source_version(project)
    assert changed["git_commit"] != clean["git_commit"]
    assert changed["git_dirty"] is False
    assert changed["version"] not in {clean["version"], dirty["version"]}
