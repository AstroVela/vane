# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import csv
import hashlib
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import vane
from scripts.verify_extension_wheel import verify_extension_wheel
from vane.extensions import DynamicExtensionDescriptor
from vane_packaging.extension_wheel import ENTRY_POINT_GROUP, build_extension_wheel

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _runtime_platform() -> str:
    connection = vane.connect()
    try:
        return connection.execute("SELECT platform FROM pragma_platform()").fetchone()[0]
    finally:
        connection.close()


def _wheel_platform_tag() -> str:
    return {
        "linux_amd64": "linux_x86_64",
        "linux_amd64_musl": "musllinux_1_2_x86_64",
        "linux_arm64": "linux_aarch64",
        "linux_arm64_musl": "musllinux_1_2_aarch64",
        "osx_amd64": "macosx_10_9_x86_64",
        "osx_arm64": "macosx_11_0_arm64",
        "windows_amd64": "win_amd64",
        "windows_arm64": "win_arm64",
    }[_runtime_platform()]


def _write_extension_artifact(
    path: Path,
    payload: bytes = b"Vane extension wheel test payload",
    source_id: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    footer = bytearray(512)
    fields = ["", "", "", "CPP", "test-version", source_id or vane.__git_revision__, _runtime_platform(), "4"]
    for index, value in enumerate(fields):
        start = index * 32
        footer[start : start + len(value)] = value.encode("ascii")
    path.write_bytes(payload + footer)
    return path


def test_platform_wheel_contains_one_verified_artifact_and_provider_entry_point(tmp_path):
    artifact_path = _write_extension_artifact(tmp_path / "sample.duckdb_extension")
    platform_tag = _wheel_platform_tag()

    built = build_extension_wheel(
        artifact=artifact_path,
        extension_name="sample",
        output_directory=tmp_path / "dist",
        platform_tag=platform_tag,
        trust_identity="vane-tests",
        license_expression="Apache-2.0 AND MIT",
        license_files=[REPOSITORY_ROOT / "LICENSE", REPOSITORY_ROOT / "LICENSES" / "DuckDB-MIT.txt"],
    )

    assert built.distribution_name == "vane-extension-sample"
    assert built.wheel_tag == f"py3-none-{platform_tag}"
    assert built.path.name == f"vane_extension_sample-{vane.__version__}-py3-none-{platform_tag}.whl"

    package_root = "vane_extensions/sample"
    dist_info_root = f"vane_extension_sample-{vane.__version__}.dist-info"
    with zipfile.ZipFile(built.path) as wheel:
        names = set(wheel.namelist())
        assert names == {
            f"{package_root}/__init__.py",
            f"{package_root}/sample.duckdb_extension",
            f"{package_root}/sample.dynamic-extension.json",
            f"{dist_info_root}/METADATA",
            f"{dist_info_root}/WHEEL",
            f"{dist_info_root}/entry_points.txt",
            f"{dist_info_root}/RECORD",
            f"{dist_info_root}/licenses/LICENSE",
            f"{dist_info_root}/licenses/LICENSES/DuckDB-MIT.txt",
        }
        assert not any(name.startswith("vane/") for name in names)
        assert wheel.read(f"{package_root}/sample.duckdb_extension") == artifact_path.read_bytes()
        descriptor = DynamicExtensionDescriptor.from_json(wheel.read(f"{package_root}/sample.dynamic-extension.json"))
        assert descriptor == built.descriptor
        assert descriptor.sha256 == hashlib.sha256(artifact_path.read_bytes()).hexdigest()

        metadata = wheel.read(f"{dist_info_root}/METADATA").decode("utf-8")
        assert metadata.startswith("Metadata-Version: 2.4\n")
        assert f"Name: {built.distribution_name}" in metadata
        assert "License-Expression: Apache-2.0 AND MIT" in metadata
        assert f"Requires-Dist: vane-ai (=={vane.__version__})" in metadata
        assert "License-File: LICENSE" in metadata
        assert "License-File: LICENSES/DuckDB-MIT.txt" in metadata
        assert wheel.read(f"{dist_info_root}/WHEEL").decode("utf-8").endswith(f"Tag: py3-none-{platform_tag}\n")
        assert wheel.read(f"{dist_info_root}/entry_points.txt").decode("utf-8") == (
            f"[{ENTRY_POINT_GROUP}]\nsample = vane_extensions.sample:provider\n"
        )
        assert "LocalExtensionProvider" in wheel.read(f"{package_root}/__init__.py").decode("utf-8")

        record_rows = list(csv.reader(io.StringIO(wheel.read(f"{dist_info_root}/RECORD").decode("utf-8"))))
        assert {row[0] for row in record_rows} == names
        assert next(row for row in record_rows if row[0].endswith("/RECORD")) == [
            f"{dist_info_root}/RECORD",
            "",
            "",
        ]


def test_platform_wheel_requires_one_explicit_platform_tag(tmp_path):
    artifact_path = _write_extension_artifact(tmp_path / "sample.duckdb_extension")

    with pytest.raises(ValueError, match="platform_tag"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name="sample",
            output_directory=tmp_path / "dist",
            platform_tag="linux-x86_64",
            trust_identity="vane-tests",
            license_expression="Apache-2.0 AND MIT",
            license_files=[REPOSITORY_ROOT / "LICENSE"],
        )


def test_platform_wheel_rejects_an_invalid_license_expression(tmp_path):
    artifact_path = _write_extension_artifact(tmp_path / "sample.duckdb_extension")

    with pytest.raises(ValueError, match="license_expression"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name="sample",
            output_directory=tmp_path / "dist",
            platform_tag=_wheel_platform_tag(),
            trust_identity="vane-tests",
            license_expression="not-an-spdx-expression",
            license_files=[REPOSITORY_ROOT / "LICENSE"],
        )


def test_platform_wheel_rejects_an_unsafe_license_file_path(tmp_path):
    artifact_path = _write_extension_artifact(tmp_path / "sample.duckdb_extension")
    license_path = tmp_path / "license\ninjected.txt"
    license_path.write_text("license", encoding="utf-8")

    with pytest.raises(ValueError, match="safe ASCII relative path"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name="sample",
            output_directory=tmp_path / "dist",
            platform_tag=_wheel_platform_tag(),
            trust_identity="vane-tests",
            license_expression="Apache-2.0 AND MIT",
            license_files=[license_path],
        )


def test_extension_wheel_cli_scripts_import_from_outside_the_repository(tmp_path):
    for script_name in ("build_extension_wheel.py", "verify_extension_wheel.py"):
        completed = subprocess.run(
            [sys.executable, "-I", str(REPOSITORY_ROOT / "scripts" / script_name), "--help"],
            check=True,
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

        assert "usage:" in completed.stdout


def test_extension_wheel_build_cli_uses_the_installed_vane_runtime(tmp_path):
    artifact_path = _write_extension_artifact(tmp_path / "sample.duckdb_extension")
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(REPOSITORY_ROOT / "scripts" / "build_extension_wheel.py"),
            "--artifact",
            str(artifact_path),
            "--extension-name",
            "sample",
            "--output-directory",
            str(tmp_path / "dist"),
            "--platform-tag",
            _wheel_platform_tag(),
            "--trust-identity",
            "vane-tests",
            "--license-expression",
            "Apache-2.0",
            "--license-file",
            str(REPOSITORY_ROOT / "LICENSE"),
        ],
        check=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert Path(completed.stdout.strip()).is_file()


def test_platform_wheel_is_deterministic_regardless_of_license_argument_order(tmp_path):
    artifact_path = _write_extension_artifact(tmp_path / "sample.duckdb_extension")
    license_files = [REPOSITORY_ROOT / "LICENSE", REPOSITORY_ROOT / "LICENSES" / "DuckDB-MIT.txt"]
    first = build_extension_wheel(
        artifact=artifact_path,
        extension_name="sample",
        output_directory=tmp_path / "first",
        platform_tag=_wheel_platform_tag(),
        trust_identity="vane-tests",
        license_expression="Apache-2.0 AND MIT",
        license_files=license_files,
    )
    second = build_extension_wheel(
        artifact=artifact_path,
        extension_name="sample",
        output_directory=tmp_path / "second",
        platform_tag=_wheel_platform_tag(),
        trust_identity="vane-tests",
        license_expression="Apache-2.0 AND MIT",
        license_files=list(reversed(license_files)),
    )

    assert first.path.read_bytes() == second.path.read_bytes()


def test_platform_wheel_escapes_license_paths_in_record_csv(tmp_path):
    artifact_path = _write_extension_artifact(tmp_path / "sample.duckdb_extension")
    license_path = tmp_path / 'license,"quoted".txt'
    license_path.write_text("license", encoding="utf-8")
    built = build_extension_wheel(
        artifact=artifact_path,
        extension_name="sample",
        output_directory=tmp_path / "dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity="vane-tests",
        license_expression="Apache-2.0",
        license_files=[license_path],
    )

    with zipfile.ZipFile(built.path) as wheel:
        record_name = f"vane_extension_sample-{vane.__version__}.dist-info/RECORD"
        record_rows = list(csv.reader(io.StringIO(wheel.read(record_name).decode("utf-8"))))

    assert any(row[0].endswith('licenses/license,"quoted".txt') for row in record_rows)
    assert all(len(row) == 3 for row in record_rows)


def test_platform_wheel_rejects_a_tag_for_a_different_artifact_platform(tmp_path):
    artifact_path = _write_extension_artifact(tmp_path / "sample.duckdb_extension")
    wrong_platform_tag = "win_amd64" if _runtime_platform() != "windows_amd64" else "linux_x86_64"

    with pytest.raises(ValueError, match="does not match extension artifact platform"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name="sample",
            output_directory=tmp_path / "dist",
            platform_tag=wrong_platform_tag,
            trust_identity="vane-tests",
            license_expression="Apache-2.0 AND MIT",
            license_files=[REPOSITORY_ROOT / "LICENSE"],
        )


def test_platform_wheel_rejects_an_artifact_from_a_different_duckdb_source(tmp_path):
    source_id = "a" * 40 if vane.__git_revision__ != "a" * 40 else "b" * 40
    artifact_path = _write_extension_artifact(tmp_path / "sample.duckdb_extension", source_id=source_id)

    with pytest.raises(RuntimeError, match="does not match installed Vane runtime SourceID"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name="sample",
            output_directory=tmp_path / "dist",
            platform_tag=_wheel_platform_tag(),
            trust_identity="vane-tests",
            license_expression="Apache-2.0 AND MIT",
            license_files=[REPOSITORY_ROOT / "LICENSE"],
        )


def test_platform_wheel_rejects_an_artifact_named_for_another_extension(tmp_path):
    artifact_path = _write_extension_artifact(tmp_path / "another.duckdb_extension")

    with pytest.raises(ValueError, match="artifact must be named"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name="sample",
            output_directory=tmp_path / "dist",
            platform_tag=_wheel_platform_tag(),
            trust_identity="vane-tests",
            license_expression="Apache-2.0 AND MIT",
            license_files=[REPOSITORY_ROOT / "LICENSE"],
        )


def test_platform_wheel_does_not_replace_a_different_existing_artifact(tmp_path):
    artifact_path = _write_extension_artifact(tmp_path / "sample.duckdb_extension")
    build_extension_wheel(
        artifact=artifact_path,
        extension_name="sample",
        output_directory=tmp_path / "dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity="vane-tests",
        license_expression="Apache-2.0 AND MIT",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    existing_wheel = next((tmp_path / "dist").glob("*.whl"))
    existing_contents = existing_wheel.read_bytes()
    _write_extension_artifact(artifact_path, b"changed extension wheel test payload")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name="sample",
            output_directory=tmp_path / "dist",
            platform_tag=_wheel_platform_tag(),
            trust_identity="vane-tests",
            license_expression="Apache-2.0 AND MIT",
            license_files=[REPOSITORY_ROOT / "LICENSE"],
        )

    assert existing_wheel.read_bytes() == existing_contents


def test_installed_platform_wheel_resolves_a_staged_artifact_in_a_clean_environment(tmp_path):
    staged_artifact = os.environ.get("VANE_TEST_LOADABLE_EXTENSION_PATH")
    base_wheel = os.environ.get("VANE_TEST_BASE_WHEEL")
    if staged_artifact is None or base_wheel is None:
        pytest.skip("set VANE_TEST_LOADABLE_EXTENSION_PATH and VANE_TEST_BASE_WHEEL for clean wheel validation")

    built = build_extension_wheel(
        artifact=Path(staged_artifact),
        extension_name="tpch",
        output_directory=tmp_path / "dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity="vane-tests",
        license_expression="Apache-2.0 AND MIT",
        license_files=[REPOSITORY_ROOT / "LICENSE", REPOSITORY_ROOT / "LICENSES" / "DuckDB-MIT.txt"],
    )

    verify_extension_wheel(
        base_wheel=Path(base_wheel),
        extension_wheel=built.path,
        extension_name="tpch",
        trust_identity="vane-tests",
    )
