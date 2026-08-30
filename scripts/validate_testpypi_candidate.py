#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate one no-tag Vane development candidate before TestPyPI publication."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from contextlib import AbstractContextManager

from packaging.version import InvalidVersion, Version

_MAIN_REF = "refs/heads/main"
_TESTPYPI_JSON_BASE = "https://test.pypi.org/pypi"


class CandidateValidationError(RuntimeError):
    """Raised when a requested development candidate is unsafe to publish."""


def validate_development_version(raw_version: str, github_ref: str) -> Version:
    """Return a canonical main-branch PEP 440 development version."""
    if github_ref != _MAIN_REF:
        raise CandidateValidationError(f"TestPyPI development candidates require {_MAIN_REF}, not {github_ref!r}")
    try:
        version = Version(raw_version)
    except InvalidVersion:
        raise CandidateValidationError(
            f"development candidate is not a valid PEP 440 version: {raw_version!r}"
        ) from None
    if str(version) != raw_version:
        raise CandidateValidationError(f"development candidate must use canonical PEP 440 spelling: {version}")
    if version.epoch != 0 or len(version.release) != 3:
        raise CandidateValidationError("development candidate must use an X.Y.Z version without an epoch")
    if not version.is_devrelease or version.local is not None:
        raise CandidateValidationError("TestPyPI development candidates must have a .devN suffix and no local version")
    return version


def require_testpypi_version_absent(
    version: Version,
    *,
    open_url: Callable[..., AbstractContextManager[object]] = urllib.request.urlopen,
) -> None:
    """Fail unless TestPyPI reports that the immutable version is unused."""
    encoded_version = urllib.parse.quote(str(version), safe="")
    url = f"{_TESTPYPI_JSON_BASE}/vane-ai/{encoded_version}/json"
    try:
        with open_url(url, timeout=30):
            pass
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return
        raise CandidateValidationError(f"TestPyPI version query failed with HTTP {error.code}") from error
    except urllib.error.URLError as error:
        raise CandidateValidationError(f"TestPyPI version query failed: {error.reason}") from error
    raise CandidateValidationError(f"vane-ai {version} already exists on TestPyPI")


def main() -> int:
    """Validate CLI arguments and TestPyPI availability."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Exact version read from the built source distribution")
    parser.add_argument("--github-ref", required=True, help="Exact GitHub Actions ref for this workflow run")
    arguments = parser.parse_args()
    try:
        version = validate_development_version(arguments.version, arguments.github_ref)
        require_testpypi_version_absent(version)
    except CandidateValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
