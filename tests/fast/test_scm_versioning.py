# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from packaging.version import Version

from vane_packaging import setuptools_scm_version

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class _Configuration:
    absolute_root: str = "/repository"


@dataclass
class _ScmVersion:
    tag: object
    distance: int | None
    dirty: bool = False
    exact: bool = False
    branch: str | None = "main"
    config: _Configuration = field(default_factory=_Configuration)


@pytest.fixture(autouse=True)
def _clear_branch_environment(monkeypatch):
    for name in ("VANE_VERSION_BRANCH", "GITHUB_BASE_REF", "GITHUB_HEAD_REF", "GITHUB_REF_NAME"):
        monkeypatch.delenv(name, raising=False)


def test_main_development_version_advances_to_next_minor(monkeypatch):
    version = _ScmVersion(tag=Version("0.1.0"), distance=14)
    state = setuptools_scm_version._VersionState(Version("0.1.0"), 14, False)
    monkeypatch.setattr(setuptools_scm_version, "_describe_main_line", lambda repository: state)

    assert setuptools_scm_version.version_scheme(version) == "0.2.0.dev14"


def test_project_metadata_uses_scm_without_a_fallback_version():
    configuration = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert configuration["project"]["dynamic"] == ["version"]
    assert "version" not in configuration["project"]
    assert configuration["tool"]["scikit-build"]["metadata"]["version"]["provider"] == (
        "scikit_build_core.metadata.setuptools_scm"
    )
    scm = configuration["tool"]["setuptools_scm"]
    assert scm["version_scheme"] == "vane_packaging.setuptools_scm_version:version_scheme"
    assert scm["local_scheme"] == "no-local-version"
    assert "fallback_version" not in scm
    assert scm["scm"]["git"]["describe_command"].endswith("--match v[0-9]*.[0-9]*.[0-9]*")
    assert scm["scm"]["git"]["pre_parse"] == "fail_on_shallow"


@pytest.mark.parametrize("tag", ["0.2.0", "0.2.1", "0.2.0rc1"])
def test_clean_exact_tag_is_the_release_version(tag):
    version = _ScmVersion(tag=Version(tag), distance=0, exact=True, branch=None)

    assert setuptools_scm_version.version_scheme(version) == tag


@pytest.mark.parametrize("tag", ["v0.2.1", "v0.2.0rc1"])
def test_git_discovery_includes_exact_maintenance_tags(tmp_path, tag):
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.txt").write_text("release\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Vane Tests",
            "-c",
            "user.email=tests@vane.invalid",
            "commit",
            "--quiet",
            "-m",
            "release",
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "tag", tag], cwd=tmp_path, check=True)
    configuration = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    command = shlex.split(configuration["tool"]["setuptools_scm"]["scm"]["git"]["describe_command"])

    result = subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)

    assert result.stdout.startswith(f"{tag}-0-g")


def test_dirty_exact_tag_is_not_released_as_the_tagged_version():
    version = _ScmVersion(tag=Version("0.2.0"), distance=0, dirty=True, exact=True)

    with pytest.raises(ValueError, match="positive Git commit distance"):
        setuptools_scm_version.version_scheme(version)


@pytest.mark.parametrize(
    ("tag", "distance", "expected"),
    [
        ("0.2.0", 3, "0.2.1.dev3"),
        ("0.2.1", 4, "0.2.2.dev4"),
        ("0.2.1", 0, "0.2.1"),
        ("0.2.2rc1", 2, "0.2.2rc2.dev2"),
        ("0.2.2.post1", 2, "0.2.2.post2.dev2"),
    ],
)
def test_release_branch_advances_patch_series(monkeypatch, tag, distance, expected):
    version = _ScmVersion(tag=Version("0.2.0"), distance=99, branch="release/0.2")
    state = setuptools_scm_version._VersionState(Version(tag), distance, False)
    calls = []

    def describe(repository, release_line):
        calls.append((repository, release_line))
        return state

    monkeypatch.setattr(setuptools_scm_version, "_describe_release_line", describe)

    assert setuptools_scm_version.version_scheme(version) == expected
    assert calls == [(Path("/repository"), (0, 2))]


def test_release_branch_uses_the_latest_tag_from_its_own_line(monkeypatch):
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, "v0.2.1-3-g0123456789abcdef0123456789abcdef01234567\n", "")

    monkeypatch.setattr(setuptools_scm_version.subprocess, "run", run)

    state = setuptools_scm_version._describe_release_line(Path("/repository"), (0, 2))

    assert state == setuptools_scm_version._VersionState(Version("0.2.1"), 3, False)
    assert captured["command"][-2:] == ["--match", "v0.2.*"]
    assert captured["kwargs"] == {
        "cwd": Path("/repository"),
        "check": True,
        "capture_output": True,
        "text": True,
    }


def test_main_uses_the_latest_minor_tag(monkeypatch):
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, "v0.1.0-14-g0123456789abcdef0123456789abcdef01234567\n", "")

    monkeypatch.setattr(setuptools_scm_version.subprocess, "run", run)

    state = setuptools_scm_version._describe_main_line(Path("/repository"))

    assert state == setuptools_scm_version._VersionState(Version("0.1.0"), 14, False)
    assert captured["command"][-2:] == ["--match", "v*.*.0"]


def test_release_pull_request_uses_its_base_branch(monkeypatch):
    monkeypatch.setenv("GITHUB_BASE_REF", "release/1.4")
    version = _ScmVersion(tag=Version("1.4.0"), distance=1, branch="feature/fix")

    assert setuptools_scm_version._release_line(version) == (1, 4)


def test_main_pull_request_does_not_use_its_release_head(monkeypatch):
    monkeypatch.setenv("GITHUB_BASE_REF", "main")
    monkeypatch.setenv("GITHUB_HEAD_REF", "release/1.4")
    monkeypatch.setenv("GITHUB_REF_NAME", "release/1.4")
    version = _ScmVersion(tag=Version("1.3.0"), distance=1, branch="release/1.4")

    assert setuptools_scm_version._release_line(version) is None


def test_explicit_version_branch_must_name_a_release_line(monkeypatch):
    monkeypatch.setenv("VANE_VERSION_BRANCH", "maintenance")
    version = _ScmVersion(tag=Version("0.2.0"), distance=0, exact=True)

    with pytest.raises(ValueError, match="release/X.Y"):
        setuptools_scm_version.version_scheme(version)
