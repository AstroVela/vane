# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import hashlib
import re
import urllib.error
from pathlib import Path

import pytest
from packaging.version import Version

from scripts.validate_testpypi_candidate import (
    CandidateValidationError,
    require_testpypi_version_absent,
    validate_development_version,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TESTPYPI_EXTENSION_KEY_FINGERPRINT = "53779fb8f9c97e9dec9c66ff838839eb234d1a64d4b105671304820e627b5e32"


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


def test_testpypi_extension_signing_key_is_candidate_only():
    option_name = "VANE_ENABLE_TESTPYPI_EXTENSION_SIGNING_KEY"
    duckdb_cmake = (REPOSITORY_ROOT / "external/duckdb/CMakeLists.txt").read_text(encoding="utf-8")
    extension_cmake = (REPOSITORY_ROOT / "external/duckdb/src/main/extension/CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    extension_helper = (REPOSITORY_ROOT / "external/duckdb/src/main/extension/extension_helper.cpp").read_text(
        encoding="utf-8"
    )

    option = re.search(rf"option\({option_name}\s+\"[^\"]+\"\s+(\w+)\)", duckdb_cmake)
    assert option is not None
    assert option.group(1) == "FALSE"
    assert extension_cmake.count(f"if({option_name})") == 1
    assert extension_cmake.count(f"PRIVATE {option_name}") == 1

    key_block = re.search(
        rf"#ifdef {option_name}.*?R\"\(\n"
        r"(-----BEGIN PUBLIC KEY-----\n.+?\n-----END PUBLIC KEY-----)\n"
        r"\)\",\n#endif",
        extension_helper,
        flags=re.DOTALL,
    )
    assert key_block is not None
    public_key_lines = key_block.group(1).splitlines()
    public_key_der = base64.b64decode("".join(public_key_lines[1:-1]), validate=True)
    assert hashlib.sha256(public_key_der).hexdigest() == TESTPYPI_EXTENSION_KEY_FINGERPRINT
    assert TESTPYPI_EXTENSION_KEY_FINGERPRINT in key_block.group(0)

    workflow_setting = (
        'CIBW_CONFIG_SETTINGS: "cmake.define.'
        f"{option_name}=${{{{ inputs.operation == 'testpypi-dev' && 'ON' || 'OFF' }}}}\""
    )
    workflows = sorted((REPOSITORY_ROOT / ".github/workflows").glob("*.yml"))
    workflow_contents = [path.read_text(encoding="utf-8") for path in workflows]
    assert sum(content.count(workflow_setting) for content in workflow_contents) == 1
    assert sum(content.count(option_name) for content in workflow_contents) == 1
