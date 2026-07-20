# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from importlib.metadata import distribution, metadata, requires, version
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

import duckdb

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _expected_duckdb_source_id(repository_root: Path) -> str:
    source_id_file = repository_root / "DUCKDB_SOURCE_ID"
    if not (repository_root / ".git").exists() and source_id_file.is_file():
        return source_id_file.read_text(encoding="ascii").strip()

    result = subprocess.run(
        [sys.executable, "scripts/sync_duckdb_source_id.py", "--print"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_base_distribution_installs_expression_runtime_dependencies():
    base_requirements = set()
    for raw_requirement in requires("vane-ai") or []:
        requirement = Requirement(raw_requirement)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            base_requirements.add(canonicalize_name(requirement.name))

    assert {"numpy", "pyarrow"} <= base_requirements


def _requirements_for_extra(extra):
    selected = set()
    for raw_requirement in requires("vane-ai") or []:
        requirement = Requirement(raw_requirement)
        if requirement.marker is not None and requirement.marker.evaluate({"extra": extra}):
            selected.add(canonicalize_name(requirement.name))
    return selected


def test_distribution_declares_alpha_version_and_apache_license_expression():
    package_metadata = metadata("vane-ai")

    assert version("vane-ai") == "0.1.0a1"
    assert package_metadata["License-Expression"] == "Apache-2.0"
    assert SpecifierSet(package_metadata["Requires-Python"]) == SpecifierSet(">=3.10,<3.13")


def test_provider_extras_match_provider_import_errors():
    assert _requirements_for_extra("openai") == {"openai"}
    assert _requirements_for_extra("anthropic") == {"anthropic"}
    assert _requirements_for_extra("google") == {"google-genai"}
    assert {"sentence-transformers", "torch", "transformers"} <= _requirements_for_extra("transformers")
    assert "vllm" in _requirements_for_extra("vllm")


def test_wheel_or_install_contains_primary_and_third_party_license_files():
    files = {str(path).replace("\\", "/") for path in distribution("vane-ai").files or []}

    assert any(path.endswith("licenses/LICENSE") for path in files)
    assert any(path.endswith("licenses/NOTICE") for path in files)
    assert any(path.endswith("licenses/LICENSES/DuckDB-MIT.txt") for path in files)
    assert any(path.endswith("licenses/LICENSES/vcpkg-binary-dependencies.txt") for path in files)
    assert any(path.endswith("licenses/duckdb/experimental/spark/LICENSE") for path in files)
    assert any(path.endswith("compression/alp/algorithm/LICENSE") for path in files)
    assert any(path.endswith("compression/alprd/algorithm/LICENSE") for path in files)


def test_duckdb_source_id_matches_recorded_source_tree():
    source_tree_id = _expected_duckdb_source_id(REPOSITORY_ROOT)
    embedded_source_id = duckdb.sql("SELECT source_id FROM pragma_version()").fetchone()[0]

    assert embedded_source_id == source_tree_id[:10]
    assert duckdb.__git_revision__ == embedded_source_id


def test_exported_tree_without_manifest_computes_source_id(tmp_path, monkeypatch):
    expected = "a" * 40

    def run_source_id(command, **kwargs):
        assert command == [sys.executable, "scripts/sync_duckdb_source_id.py", "--print"]
        assert kwargs["cwd"] == tmp_path
        return subprocess.CompletedProcess(command, 0, stdout=expected + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run_source_id)

    assert _expected_duckdb_source_id(tmp_path) == expected


def test_sdist_tree_uses_injected_source_id(tmp_path, monkeypatch):
    expected = "b" * 40
    (tmp_path / "DUCKDB_SOURCE_ID").write_text(expected + "\n", encoding="ascii")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("an sdist must use its injected SourceID"),
    )

    assert _expected_duckdb_source_id(tmp_path) == expected
