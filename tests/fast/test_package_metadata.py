# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import os
import platform
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


def _base_requirements():
    base_requirements = {}
    for raw_requirement in requires("vane-ai") or []:
        requirement = Requirement(raw_requirement)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            base_requirements[canonicalize_name(requirement.name)] = requirement

    return base_requirements


def test_base_distribution_installs_expression_runtime_dependencies():
    base_requirements = _base_requirements()

    assert {"numpy", "pyarrow"} <= set(base_requirements)


def test_base_distribution_requires_pyarrow_14_or_newer():
    pyarrow_requirement = _base_requirements()["pyarrow"]

    assert pyarrow_requirement.specifier == SpecifierSet(">=14.0.0")


def test_artifact_mode_imports_installed_python_packages():
    if os.environ.get("VANE_FAST_TEST_ARTIFACT_MODE") != "1":
        pytest.skip("only applies to artifact-backed fast-test jobs")

    import vane

    environment_root = Path(sys.prefix).resolve()
    for package in (duckdb, vane):
        package_path = Path(package.__file__).resolve()
        assert package_path.is_relative_to(environment_root), (
            f"{package.__name__} was imported from the checkout: {package_path}"
        )


def test_vane_distribution_declares_inline_types():
    import vane

    assert Path(vane.__file__).with_name("py.typed").is_file()


def _requirements_for_extra(extra):
    selected = set()
    for raw_requirement in requires("vane-ai") or []:
        requirement = Requirement(raw_requirement)
        if requirement.marker is not None and requirement.marker.evaluate({"extra": extra}):
            selected.add(canonicalize_name(requirement.name))
    return selected


def _requirement_for_extra(extra, package, environment=None):
    selected = []
    marker_environment = {"extra": extra, **(environment or {})}
    for raw_requirement in requires("vane-ai") or []:
        requirement = Requirement(raw_requirement)
        if canonicalize_name(requirement.name) != canonicalize_name(package):
            continue
        if requirement.marker is not None and requirement.marker.evaluate(marker_environment):
            selected.append(requirement)
    assert len(selected) == 1
    return selected[0]


def test_distribution_declares_alpha_version_and_apache_license_expression():
    package_metadata = metadata("vane-ai")

    assert version("vane-ai") == "0.1.0a1"
    assert package_metadata["License-Expression"] == "Apache-2.0"
    assert SpecifierSet(package_metadata["Requires-Python"]) == SpecifierSet(">=3.10,<3.15")


def test_provider_extras_match_provider_import_errors():
    assert _requirements_for_extra("openai") == {"openai", "tiktoken"}
    assert _requirements_for_extra("anthropic") == {"anthropic"}
    assert _requirements_for_extra("google") == {"google-genai"}
    assert {"sentence-transformers", "torch", "transformers"} <= _requirements_for_extra("transformers")
    assert "vllm" in _requirements_for_extra("vllm")


def test_structured_provider_extras_require_supported_sdk_versions():
    openai = _requirement_for_extra("openai", "openai")
    google = _requirement_for_extra("google", "google-genai")
    vllm = _requirement_for_extra(
        "vllm",
        "vllm",
        {"platform_system": "Linux", "platform_machine": "x86_64"},
    )

    assert openai.specifier == SpecifierSet(">=1.66.0")
    assert google.specifier == SpecifierSet(">=1.22.0")
    assert vllm.specifier == SpecifierSet(">=0.11.0")


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


def test_release_runtime_is_self_contained_by_default():
    connection = duckdb.connect()
    settings = dict(
        connection.execute(
            """
            SELECT name, value
            FROM duckdb_settings()
            WHERE name IN (
                'allow_unsigned_extensions',
                'autoinstall_known_extensions',
                'autoload_known_extensions'
            )
            """
        ).fetchall()
    )
    static_extensions = {
        name: (installed, loaded)
        for name, installed, loaded in connection.execute(
            """
            SELECT extension_name, installed, loaded
            FROM duckdb_extensions()
            WHERE install_mode = 'STATICALLY_LINKED'
            """
        ).fetchall()
    }

    expected_extensions = {"core_functions", "httpfs", "icu", "json", "parquet"}
    if platform.system() == "Linux" and sys.maxsize > 2**32:
        expected_extensions.add("jemalloc")

    assert settings == {
        "allow_unsigned_extensions": "false",
        "autoinstall_known_extensions": "false",
        "autoload_known_extensions": "false",
    }
    assert static_extensions == {name: (True, True) for name in expected_extensions}


@pytest.mark.parametrize(
    ("setting_name", "setting_value", "expected"),
    [
        ("http_timeout", "41", 41),
        ("binary_as_string", "true", True),
        ("calendar", "gregorian", "gregorian"),
    ],
)
def test_static_extension_settings_are_available_during_connect(tmp_path, setting_name, setting_value, expected):
    extension_directory = tmp_path / "extensions"
    connection = duckdb.connect(
        config={
            setting_name: setting_value,
            "extension_directory": str(extension_directory),
            "custom_extension_repository": "http://127.0.0.1:9",
        }
    )

    assert connection.execute(f"SELECT current_setting('{setting_name}')").fetchone()[0] == expected
    assert not extension_directory.exists()


def test_dynamic_extension_settings_are_rejected_during_connect_without_installing(tmp_path):
    extension_directory = tmp_path / "extensions"

    with pytest.raises(duckdb.InvalidInputException, match="sqlite_all_varchar.*sqlite_scanner.*not statically linked"):
        duckdb.connect(
            config={
                "sqlite_all_varchar": "true",
                "extension_directory": str(extension_directory),
                "custom_extension_repository": "http://127.0.0.1:9",
            }
        )

    assert not extension_directory.exists()


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


def test_image_extra_installs_pillow():
    assert _requirements_for_extra("image") == {"pillow"}


def test_video_extra_installs_video_dependencies():
    selected = _requirements_for_extra("video")
    assert {"pillow", "psutil"} <= selected
    supports_decord = platform.system() == "Linux" and platform.machine() == "x86_64"
    assert ("decord" in selected) is supports_decord


def test_base_distribution_keeps_video_dependencies_optional():
    base_requirements = set()
    for raw_requirement in requires("vane-ai") or []:
        requirement = Requirement(raw_requirement)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            base_requirements.add(canonicalize_name(requirement.name))

    assert {"pillow", "psutil", "decord"}.isdisjoint(base_requirements)
