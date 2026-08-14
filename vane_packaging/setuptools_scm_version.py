# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Branch-aware setuptools-scm version scheme for Vane."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from packaging.version import Version

RELEASE_BRANCH = re.compile(r"^(?:refs/heads/)?release/(?P<major>0|[1-9][0-9]*)\.(?P<minor>0|[1-9][0-9]*)$")
GIT_DESCRIBE = re.compile(
    r"^(?P<tag>.+)-(?P<distance>[0-9]+)-g[0-9a-f]{40}(?P<dirty>-dirty)?$",
    re.IGNORECASE,
)


class _ScmConfiguration(Protocol):
    absolute_root: str


class _ScmVersion(Protocol):
    tag: object
    distance: int | None
    dirty: bool
    exact: bool
    branch: str | None
    config: _ScmConfiguration


@dataclass(frozen=True)
class _VersionState:
    tag: Version
    distance: int
    dirty: bool

    @property
    def exact(self) -> bool:
        return self.distance == 0


def version_scheme(version: _ScmVersion) -> str:
    """Return an exact tag or the next minor/patch development version."""
    if version.tag is None:
        raise ValueError("Vane builds require a version tag in Git history")

    release_line = _release_line(version)
    if release_line is None:
        state = _VersionState(
            tag=Version(str(version.tag)),
            distance=int(version.distance or 0),
            dirty=version.dirty,
        )
    else:
        state = _describe_release_line(Path(version.config.absolute_root), release_line)

    if state.exact and not state.dirty:
        return str(state.tag)

    if state.distance <= 0:
        raise ValueError("a development build requires a positive Git commit distance")

    major, minor, patch = state.tag.release
    if state.tag.post is not None:
        next_version = f"{major}.{minor}.{patch}.post{state.tag.post + 1}"
    elif state.tag.pre is not None:
        prerelease, number = state.tag.pre
        next_version = f"{major}.{minor}.{patch}{prerelease}{number + 1}"
    elif release_line is not None:
        next_version = f"{major}.{minor}.{patch + 1}"
    else:
        next_version = f"{major}.{minor + 1}.0"
    return f"{next_version}.dev{state.distance}"


def _release_line(version: _ScmVersion) -> tuple[int, int] | None:
    override = os.getenv("VANE_VERSION_BRANCH")
    if override:
        match = RELEASE_BRANCH.fullmatch(override)
        if match is None:
            raise ValueError("VANE_VERSION_BRANCH must use the form release/X.Y")
        return int(match["major"]), int(match["minor"])

    candidates = (
        os.getenv("GITHUB_BASE_REF"),
        os.getenv("GITHUB_HEAD_REF"),
        os.getenv("GITHUB_REF_NAME"),
        version.branch,
    )
    for branch in candidates:
        if not branch:
            continue
        match = RELEASE_BRANCH.fullmatch(branch)
        if match is not None:
            return int(match["major"]), int(match["minor"])
    return None


def _describe_release_line(repository: Path, release_line: tuple[int, int]) -> _VersionState:
    major, minor = release_line
    result = subprocess.run(
        [
            "git",
            "describe",
            "--dirty",
            "--tags",
            "--long",
            "--abbrev=40",
            "--match",
            f"v{major}.{minor}.*",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    description = result.stdout.strip()
    match = GIT_DESCRIBE.fullmatch(description)
    if match is None:
        raise ValueError(f"Git returned an invalid version description: {description!r}")

    tag = Version(match["tag"].removeprefix("v"))
    if tag.release[:2] != release_line:
        raise ValueError(f"tag {tag} is outside release line {major}.{minor}")
    return _VersionState(
        tag=tag,
        distance=int(match["distance"]),
        dirty=match["dirty"] is not None,
    )
