# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import urllib.error

import pytest
from packaging.version import Version

from scripts.validate_testpypi_candidate import (
    CandidateValidationError,
    require_testpypi_version_absent,
    validate_development_version,
)


@pytest.mark.parametrize("raw_version", ["0.2.0.dev601", "0.2.0rc2.dev3"])
def test_validate_development_version_accepts_canonical_main_versions(raw_version):
    assert validate_development_version(raw_version, "refs/heads/main") == Version(raw_version)


@pytest.mark.parametrize(
    ("raw_version", "github_ref", "message"),
    [
        ("0.2.0.dev601", "refs/heads/feature/candidate", "require refs/heads/main"),
        ("0.2.0", "refs/heads/main", "must have a .devN suffix"),
        ("0.2.0.dev601+local", "refs/heads/main", "no local version"),
        ("0.2.dev601", "refs/heads/main", "must use an X.Y.Z version"),
        ("0.2.0.dev0601", "refs/heads/main", "canonical PEP 440 spelling"),
    ],
)
def test_validate_development_version_rejects_unsafe_candidates(raw_version, github_ref, message):
    with pytest.raises(CandidateValidationError, match=message):
        validate_development_version(raw_version, github_ref)


def test_require_testpypi_version_absent_accepts_only_an_explicit_404():
    requested: list[tuple[str, int]] = []

    def missing(url, *, timeout):
        requested.append((url, timeout))
        raise urllib.error.HTTPError(url, 404, "missing", {}, None)

    require_testpypi_version_absent(Version("0.2.0.dev601"), open_url=missing)
    assert requested == [("https://test.pypi.org/pypi/vane-ai/0.2.0.dev601/json", 30)]


@pytest.mark.parametrize("status", [200, 500])
def test_require_testpypi_version_absent_rejects_existing_or_indeterminate_versions(status):
    class ExistingResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def open_url(url, *, timeout):
        if status == 500:
            raise urllib.error.HTTPError(url, status, "failed", {}, None)
        return ExistingResponse()

    expected = "already exists" if status == 200 else "query failed"
    with pytest.raises(CandidateValidationError, match=expected):
        require_testpypi_version_absent(Version("0.2.0.dev601"), open_url=open_url)
