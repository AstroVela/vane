# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
import secrets
import string
import subprocess
import sys
import tarfile
import traceback
import zipfile
from collections.abc import Callable
from email.message import Message
from pathlib import Path

import pytest
from packaging.version import Version

from scripts import check_release_artifacts, verify_duckdb_coexistence

TEST_VERSION = Version("0.2.0.dev14")
TEST_LAYOUT = check_release_artifacts.distribution_layout(TEST_VERSION)
WHEEL_DATA_ROOT = f"{TEST_LAYOUT.archive_root}.data"
REQUIRED_WHEEL_PATHS = (
    "vane/py.typed",
    "vane/_native/__init__.pyi",
    "vane/_native/_func.pyi",
    "vane/_native/_sqltypes.pyi",
    "vane/_native/ray_cxx.pyi",
    "vane/sqltypes/__init__.pyi",
    "vane/udf.pyi",
    f"{TEST_LAYOUT.dist_info_root}/METADATA",
    f"{TEST_LAYOUT.dist_info_root}/WHEEL",
    f"{TEST_LAYOUT.dist_info_root}/RECORD",
)


class _NamesOnlyArtifact:
    path = Path("vane_ai-test.whl")

    def __init__(self, names: list[str]):
        self._names = names

    def names(self) -> list[str]:
        return self._names

    def path_names(self) -> list[str]:
        return self._names


class _MemoryWheelArtifact(_NamesOnlyArtifact):
    def __init__(self, members: dict[str, bytes]):
        super().__init__(list(members))
        self._members = members

    def read(self, name: str) -> bytes:
        return self._members[name]


def _runtime_sentinel() -> bytes:
    return b"Aa0!" + secrets.token_urlsafe(32).encode("ascii")


def _runtime_path() -> bytes:
    def segment() -> str:
        return "".join(secrets.choice(string.ascii_lowercase) for _ in range(4))

    return f"/{segment()}/{segment()}/".encode("ascii")


def _content_rule_manifest(rule_id: str, value: bytes, *, text_only: bool) -> str:
    return json.dumps(
        {
            "version": 1,
            "rules": [
                {
                    "id": rule_id,
                    "scope": "text" if text_only else "all",
                    "value_base64": base64.b64encode(value).decode("ascii"),
                }
            ],
        }
    )


def _write_archive(path: Path, member_name: str, data: bytes) -> None:
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, mode="w:gz") as archive:
            member = tarfile.TarInfo(member_name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        return

    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr(member_name, data)


def _artifact_path(tmp_path: Path, suffix: str, label: str) -> Path:
    directory = tmp_path / label
    directory.mkdir()
    if suffix == ".tar.gz":
        return directory / f"vane_ai-{TEST_VERSION}.tar.gz"
    return directory / f"vane_ai-{TEST_VERSION}-py3-none-any.whl"


def _rejected_by(callback: Callable[[], object]) -> ValueError:
    try:
        callback()
    except ValueError as error:
        return error
    except Exception as error:
        pytest.fail(f"unexpected exception type: {type(error).__name__}", pytrace=False)
    pytest.fail("contaminated artifact was accepted", pytrace=False)


def _assert_no_recoverable_value(sentinel: bytes, *surfaces: str) -> None:
    recoverable_values = (sentinel, base64.b64encode(sentinel))
    encoded_surfaces = tuple(surface.encode("utf-8", errors="replace") for surface in surfaces)
    if any(value in surface for value in recoverable_values for surface in encoded_surfaces):
        pytest.fail("output exposed a recoverable matched value", pytrace=False)


def _assert_metadata_only_error(
    error: ValueError,
    *,
    sentinel: bytes,
    rule_id: str,
    member_name: str,
    capsys,
) -> None:
    message = str(error)
    if rule_id not in message or member_name not in message:
        pytest.fail("content error omitted rule metadata", pytrace=False)

    captured = capsys.readouterr()
    surfaces = (
        message,
        repr(error.args),
        "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        captured.out,
        captured.err,
    )
    _assert_no_recoverable_value(sentinel, *surfaces)


@pytest.fixture
def content_check_only(monkeypatch):
    monkeypatch.setattr(check_release_artifacts, "_check_sdist", lambda _artifact, _layout: None)
    monkeypatch.setattr(check_release_artifacts, "_check_wheel", lambda _artifact, _layout: None)


def test_private_manifest_rejects_empty_rule_set():
    with pytest.raises(ValueError, match="at least one rule"):
        check_release_artifacts._parse_content_rule_manifest(
            '{"version": 1, "rules": []}',
            source="runtime manifest",
        )


def test_literal_rule_repr_does_not_expose_value():
    sentinel = _runtime_sentinel()
    rule = check_release_artifacts.LiteralContentRule("runtime-sensitive-content", sentinel)

    _assert_no_recoverable_value(sentinel, repr(rule))


@pytest.mark.parametrize(
    "member_name",
    [
        "/duckdb/__init__.py",
        "duckdb\\__init__.py",
        "C:/duckdb/__init__.py",
        "C:duckdb/__init__.py",
        "vane//module.py",
        "vane/./module.py",
        "vane/../duckdb.py",
        "vane/module\n.py",
    ],
)
def test_release_rejects_non_relative_non_posix_archive_paths(member_name):
    artifact = _NamesOnlyArtifact([member_name])

    with pytest.raises(ValueError, match="unsafe archive path"):
        check_release_artifacts._check_paths(artifact)


@pytest.mark.parametrize(
    "member_names",
    [
        ["vane/module.py", "VANE/MODULE.PY"],
        ["vane/caf\N{LATIN SMALL LETTER E WITH ACUTE}.py", "vane/cafe\N{COMBINING ACUTE ACCENT}.py"],
        ["vane/module.py", "vane/module.py"],
    ],
)
def test_release_rejects_cross_platform_archive_path_collisions(member_names):
    artifact = _NamesOnlyArtifact(member_names)

    with pytest.raises(ValueError, match="colliding archive paths"):
        check_release_artifacts._check_paths(artifact)


@pytest.mark.parametrize(
    "member_names",
    [
        ["vane", "vane/__init__.py"],
        ["vane/module.py", "vane/module.py/child"],
        ["VANE", "vane/__init__.py"],
    ],
)
def test_release_rejects_archive_file_parent_conflicts(member_names):
    artifact = _NamesOnlyArtifact(member_names)

    with pytest.raises(ValueError, match="file cannot be the parent"):
        check_release_artifacts._check_paths(artifact)


@pytest.mark.parametrize(
    "member_name",
    [
        "duckdb/__init__.py",
        "DuckDB/__init__.py",
        "duckdb.py",
        "_duckdb.cpython-312-x86_64-linux-gnu.so",
        "adbc_driver_duckdb/dbapi.py",
        f"{WHEEL_DATA_ROOT}/purelib/duckdb/__init__.py",
        f"{WHEEL_DATA_ROOT}/platlib/_duckdb.pyd",
    ],
)
def test_wheel_rejects_every_official_duckdb_import_location(member_name):
    artifact = _NamesOnlyArtifact([member_name])

    with pytest.raises(ValueError, match="conflicting Python package path"):
        check_release_artifacts._check_wheel(artifact, TEST_LAYOUT)


@pytest.mark.parametrize(
    "member_name",
    [
        "another_namespace/__init__.py",
        "another_module.py",
        "VANE/__init__.py",
        f"{WHEEL_DATA_ROOT}/purelib/vane/__init__.py",
        f"{WHEEL_DATA_ROOT}/platlib/vane/_native.so",
        f"{WHEEL_DATA_ROOT}/purelib/another_namespace/__init__.py",
        f"{WHEEL_DATA_ROOT}/scripts/vane",
        "other_distribution-1.0.dist-info/METADATA",
    ],
)
def test_wheel_rejects_every_import_or_distribution_root_not_owned_by_vane(member_name):
    artifact = _NamesOnlyArtifact([member_name])

    with pytest.raises(ValueError, match="conflicting Python package path"):
        check_release_artifacts._check_wheel(artifact, TEST_LAYOUT)


def test_base_wheel_rejects_a_dynamic_extension_artifact():
    artifact = _NamesOnlyArtifact(["vane/extensions/tpch.duckdb_extension"])

    with pytest.raises(ValueError, match="must not contain optional extension artifacts"):
        check_release_artifacts._check_wheel(artifact, TEST_LAYOUT)


@pytest.mark.parametrize("required_path", REQUIRED_WHEEL_PATHS)
def test_wheel_required_files_must_use_their_exact_install_paths(required_path):
    decoy = (
        f"vane/decoy/{required_path}"
        if required_path.startswith("vane/")
        else (f"{TEST_LAYOUT.dist_info_root}/decoy/{required_path}")
    )
    members = ["vane/_native.so", *REQUIRED_WHEEL_PATHS]
    members[members.index(required_path)] = decoy
    artifact = _NamesOnlyArtifact(members)

    with pytest.raises(ValueError, match="expected one archive member"):
        check_release_artifacts._check_wheel(artifact, TEST_LAYOUT)


@pytest.mark.parametrize("required_path", ["vane/sqltypes/__init__.pyi", "vane/udf.pyi"])
def test_sdist_public_stubs_must_use_their_exact_project_paths(required_path):
    decoy = f"{TEST_LAYOUT.archive_root}/decoy/{required_path}"

    with pytest.raises(ValueError, match="expected one project file"):
        check_release_artifacts._require_sdist_path([decoy], required_path, Path("test.tar.gz"))


@pytest.mark.parametrize(
    "member_name",
    [
        "vane/_native.pyi",
        f"{TEST_LAYOUT.archive_root}/vane/_native.pyi",
    ],
)
def test_release_rejects_the_legacy_flat_native_stub(member_name):
    artifact = _NamesOnlyArtifact([member_name])

    with pytest.raises(ValueError, match="banned release path"):
        check_release_artifacts._check_paths(artifact)


def test_sdist_metadata_must_use_the_root_project_path():
    root = TEST_LAYOUT.archive_root
    artifact = _NamesOnlyArtifact([f"{root}/decoy/PKG-INFO"])

    with pytest.raises(ValueError, match="expected one archive member"):
        check_release_artifacts._check_metadata(artifact, f"{root}/PKG-INFO", TEST_LAYOUT)


@pytest.mark.parametrize(
    "requirement",
    [
        'duckdb\t; python_version > "3"',
        "duckdb @ https://example.invalid/duckdb.whl",
        "DuckDB[httpfs]>=1.5",
    ],
)
def test_release_metadata_rejects_every_official_duckdb_requirement(requirement):
    metadata = Message()
    metadata["Requires-Dist"] = requirement
    artifact = _NamesOnlyArtifact([])

    with pytest.raises(ValueError, match="must not depend on the official duckdb distribution"):
        check_release_artifacts._check_no_official_duckdb_dependency(artifact, metadata)


def test_release_metadata_rejects_invalid_requirements():
    metadata = Message()
    metadata["Requires-Dist"] = "duckdb (>=1.0"
    artifact = _NamesOnlyArtifact([])

    with pytest.raises(ValueError, match="invalid Requires-Dist metadata"):
        check_release_artifacts._check_no_official_duckdb_dependency(artifact, metadata)


@pytest.mark.parametrize(
    "member_name",
    [
        "duckdb/__init__.py",
        "DuckDB/__init__.py",
        f"{TEST_LAYOUT.archive_root}/duckdb/__init__.py",
        f"{TEST_LAYOUT.archive_root}/_duckdb.so",
        f"{TEST_LAYOUT.archive_root}/adbc_driver_duckdb/dbapi.py",
    ],
)
def test_sdist_rejects_every_official_duckdb_import_location(member_name):
    artifact = _NamesOnlyArtifact([member_name])

    with pytest.raises(ValueError, match="conflicting Python package path"):
        check_release_artifacts._check_sdist(artifact, TEST_LAYOUT)


def test_wheel_record_rejects_duplicate_rows():
    record_name = f"{TEST_LAYOUT.dist_info_root}/RECORD"
    artifact = _MemoryWheelArtifact({record_name: f"{record_name},,\n{record_name},,\n".encode()})

    with pytest.raises(ValueError, match="duplicate RECORD entry"):
        check_release_artifacts._check_wheel_record(artifact, TEST_LAYOUT)


def test_wheel_record_must_not_hash_or_size_itself():
    record_name = f"{TEST_LAYOUT.dist_info_root}/RECORD"
    artifact = _MemoryWheelArtifact({record_name: f"{record_name},sha256=bogus,1\n".encode()})

    with pytest.raises(ValueError, match="must not hash or size itself"):
        check_release_artifacts._check_wheel_record(artifact, TEST_LAYOUT)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("vane_ai-0.2.0.dev14.tar.gz", TEST_VERSION),
        ("vane_ai-0.2.0.dev14-cp312-cp312-manylinux_2_28_x86_64.whl", TEST_VERSION),
    ],
)
def test_release_artifact_version_comes_from_distribution_filename(filename, expected):
    assert check_release_artifacts._artifact_version(Path(filename)) == expected


def test_release_artifact_filename_must_match_expected_version(tmp_path):
    artifact = tmp_path / "vane_ai-0.2.1.tar.gz"
    _write_archive(artifact, "vane_ai-0.2.1/PKG-INFO", b"")

    with pytest.raises(ValueError, match="expected version 0.2.0.dev14, found 0.2.1"):
        check_release_artifacts.check_artifact(artifact, expected_version=TEST_VERSION)


def test_coexistence_subprocess_keeps_validation_assertions_under_pythonoptimize(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")

    with pytest.raises(subprocess.CalledProcessError):
        verify_duckdb_coexistence._python(Path(sys.executable), "assert False", cwd=tmp_path)


@pytest.mark.parametrize(
    ("sentinel_factory", "text_only"),
    [
        pytest.param(_runtime_sentinel, False, id="binary"),
        pytest.param(_runtime_path, True, id="text"),
    ],
)
@pytest.mark.parametrize("suffix", [".tar.gz", ".whl"], ids=["sdist", "wheel"])
def test_cli_loads_private_manifest_without_exposing_match(
    tmp_path,
    suffix,
    sentinel_factory,
    text_only,
    content_check_only,
    monkeypatch,
    capsys,
):
    sentinel = sentinel_factory()
    rule_id = "runtime-sensitive-content"
    member_name = "project/security-probe.txt"
    contaminated = _artifact_path(tmp_path, suffix, "contaminated")
    clean = _artifact_path(tmp_path, suffix, "clean")
    manifest = tmp_path / "content-rules.json"
    _write_archive(contaminated, member_name, b"prefix-" + sentinel + b"-suffix")
    _write_archive(clean, member_name, b"clean artifact content")
    manifest.write_text(
        _content_rule_manifest(rule_id, sentinel, text_only=text_only),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_release_artifacts.py",
            "--content-rules-manifest",
            str(manifest),
            str(contaminated),
        ],
    )

    error = _rejected_by(check_release_artifacts.main)

    _assert_metadata_only_error(
        error,
        sentinel=sentinel,
        rule_id=rule_id,
        member_name=member_name,
        capsys=capsys,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_release_artifacts.py",
            "--content-rules-manifest",
            str(manifest),
            str(clean),
        ],
    )
    assert check_release_artifacts.main() == 0
    captured = capsys.readouterr()
    _assert_no_recoverable_value(sentinel, captured.out, captured.err)


@pytest.mark.parametrize("suffix", [".tar.gz", ".whl"], ids=["sdist", "wheel"])
def test_standalone_cli_reads_private_manifest_from_stdin(tmp_path, suffix):
    sentinel = _runtime_sentinel()
    rule_id = "runtime-sensitive-content"
    member_name = "project/security-probe.txt"
    contaminated = _artifact_path(tmp_path, suffix, "contaminated")
    _write_archive(contaminated, member_name, b"prefix-" + sentinel + b"-suffix")

    result = subprocess.run(
        [
            sys.executable,
            str(Path(check_release_artifacts.__file__).resolve()),
            "--content-rules-manifest",
            "-",
            str(contaminated),
        ],
        input=_content_rule_manifest(rule_id, sentinel, text_only=False),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    output = (result.stdout + result.stderr).encode("utf-8", errors="replace")
    assert rule_id.encode("ascii") in output
    assert member_name.encode("ascii") in output
    _assert_no_recoverable_value(sentinel, result.stdout, result.stderr)


@pytest.mark.parametrize("suffix", [".tar.gz", ".whl"], ids=["sdist", "wheel"])
def test_runtime_binary_rule_checks_members_with_nul_bytes(
    tmp_path,
    suffix,
    content_check_only,
    capsys,
):
    sentinel = _runtime_sentinel()
    rule_id = "runtime-binary-content"
    member_name = "project/security-probe.bin"
    rule = check_release_artifacts.LiteralContentRule(rule_id, sentinel)
    artifact = _artifact_path(tmp_path, suffix, "binary")
    _write_archive(artifact, member_name, b"\0" + sentinel)

    error = _rejected_by(
        lambda: check_release_artifacts.check_artifact(
            artifact,
            expected_version=TEST_VERSION,
            content_rules=(rule,),
            text_content_rules=(),
        )
    )

    _assert_metadata_only_error(
        error,
        sentinel=sentinel,
        rule_id=rule_id,
        member_name=member_name,
        capsys=capsys,
    )


@pytest.mark.parametrize("suffix", [".tar.gz", ".whl"], ids=["sdist", "wheel"])
def test_runtime_text_rule_preserves_binary_member_filter(
    tmp_path,
    suffix,
    content_check_only,
    capsys,
):
    sentinel = _runtime_path()
    rule_id = "runtime-text-content"
    member_name = "project/security-probe.bin"
    rule = check_release_artifacts.LiteralContentRule(rule_id, sentinel)
    text_artifact = _artifact_path(tmp_path, suffix, "text")
    binary_artifact = _artifact_path(tmp_path, suffix, "binary")
    _write_archive(text_artifact, member_name, sentinel)
    _write_archive(binary_artifact, member_name, b"\0" + sentinel)

    error = _rejected_by(
        lambda: check_release_artifacts.check_artifact(
            text_artifact,
            expected_version=TEST_VERSION,
            content_rules=(),
            text_content_rules=(rule,),
        )
    )

    _assert_metadata_only_error(
        error,
        sentinel=sentinel,
        rule_id=rule_id,
        member_name=member_name,
        capsys=capsys,
    )
    check_release_artifacts.check_artifact(
        binary_artifact,
        expected_version=TEST_VERSION,
        content_rules=(),
        text_content_rules=(rule,),
    )
