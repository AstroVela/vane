# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import base64
import gzip
import io
import json
import os
import secrets
import string
import subprocess
import sys
import tarfile
import traceback
import zipfile
from collections.abc import Callable
from contextlib import contextmanager
from email.message import Message
from pathlib import Path

import pytest
from packaging.version import Version

from scripts import check_release_artifacts, verify_duckdb_coexistence
from vane_packaging import archive_safety, artifact_limits

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


def test_release_artifact_size_budgets_are_independent() -> None:
    assert artifact_limits.MAX_PUBLICATION_FILE_BYTES == 128 * artifact_limits.MEBIBYTE
    assert artifact_limits.MAX_ARCHIVE_MEMBER_BYTES == 384 * artifact_limits.MEBIBYTE
    assert artifact_limits.MAX_ARCHIVE_UNCOMPRESSED_BYTES == 512 * artifact_limits.MEBIBYTE
    assert artifact_limits.MAX_EXTENSION_ARTIFACT_BYTES == artifact_limits.MAX_ARCHIVE_MEMBER_BYTES
    assert check_release_artifacts.MAX_ARTIFACT_BYTES == artifact_limits.MAX_PUBLICATION_FILE_BYTES
    assert check_release_artifacts.MAX_ARTIFACT_MEMBER_BYTES == artifact_limits.MAX_ARCHIVE_MEMBER_BYTES
    assert check_release_artifacts.MAX_ARTIFACT_UNCOMPRESSED_BYTES == artifact_limits.MAX_ARCHIVE_UNCOMPRESSED_BYTES
    assert (
        artifact_limits.MAX_PUBLICATION_FILE_BYTES
        < artifact_limits.MAX_ARCHIVE_MEMBER_BYTES
        < artifact_limits.MAX_ARCHIVE_UNCOMPRESSED_BYTES
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
    _write_archive_members(path, {member_name: data})


def _write_archive_members(path: Path, members: dict[str, bytes]) -> None:
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, mode="w:gz") as archive:
            for member_name, data in members.items():
                member = tarfile.TarInfo(member_name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        return

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member_name, data in members.items():
            archive.writestr(member_name, data)


def _validate_tar_member_count(path: Path, **kwargs) -> int:
    with archive_safety.snapshot_archive(
        path,
        max_bytes=artifact_limits.MAX_PUBLICATION_FILE_BYTES,
        description="sdist",
        size_limit_description=artifact_limits.PUBLICATION_FILE_LIMIT_DESCRIPTION,
    ) as snapshot:
        return archive_safety.validate_tar_member_count(
            snapshot.file,
            archive_path=path,
            **kwargs,
        )


def _pax_record(key: bytes, value: bytes) -> bytes:
    body = b" " + key + b"=" + value + b"\n"
    length = len(body) + 1
    while True:
        record = str(length).encode("ascii") + body
        if len(record) == length:
            return record
        length = len(record)


def _raw_tar_member(member: tarfile.TarInfo, payload: bytes = b"") -> bytes:
    assert len(payload) == member.size
    return member.tobuf(format=tarfile.PAX_FORMAT) + payload + b"\0" * ((-len(payload)) % tarfile.BLOCKSIZE)


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


@pytest.mark.parametrize(
    "member",
    ["vane/extensions/tpch.duckdb_extension", "vane/extensions/tpch.DUCKDB_EXTENSION"],
)
def test_base_wheel_rejects_a_dynamic_extension_artifact(member):
    artifact = _NamesOnlyArtifact([member])

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


@pytest.mark.parametrize("line_ending", [b"\n", b"\r"], ids=["lf", "bare-cr"])
def test_release_metadata_bounds_headers_before_email_parsing(tmp_path, monkeypatch, line_ending):
    artifact_path = tmp_path / "metadata.whl"
    metadata_name = f"{TEST_LAYOUT.dist_info_root}/METADATA"
    _write_archive(
        artifact_path,
        metadata_name,
        (b"X-Untrusted: value" + line_ending) * 32,
    )
    artifact = check_release_artifacts.WheelArtifact(artifact_path)

    class RejectingParser:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("email parser was constructed before metadata headers were bounded")

    monkeypatch.setattr(check_release_artifacts, "MAX_CORE_METADATA_HEADERS", 16)
    monkeypatch.setattr(check_release_artifacts, "BytesParser", RejectingParser)
    try:
        with pytest.raises(ValueError, match="core metadata contains more than 16 headers"):
            check_release_artifacts._metadata(artifact, metadata_name)
    finally:
        artifact.close()


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


def test_content_only_cli_scans_an_extension_wheel_from_stdin(tmp_path):
    sentinel = _runtime_sentinel()
    rule_id = "runtime-extension-content"
    member_name = "vane_extensions/sample/sample.duckdb_extension"
    artifact = tmp_path / "vane_extension_sample-1-py3-none-any.whl"
    _write_archive(artifact, member_name, b"prefix-" + sentinel + b"-suffix")

    result = subprocess.run(
        [
            sys.executable,
            str(Path(check_release_artifacts.__file__).resolve()),
            "--scan-contents-only",
            "--content-rules-manifest",
            "-",
            str(artifact),
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
def test_content_scan_rejects_an_oversized_decompressed_member_before_reading_it(
    tmp_path,
    suffix,
    monkeypatch,
):
    artifact = _artifact_path(tmp_path, suffix, "oversized-member")
    contents = b"a" * 1024
    _write_archive(artifact, "project/oversized.bin", contents)
    assert artifact.stat().st_size < len(contents)
    monkeypatch.setattr(check_release_artifacts, "MAX_ARTIFACT_MEMBER_BYTES", 512)
    artifact_type = (
        check_release_artifacts.SdistArtifact if suffix == ".tar.gz" else check_release_artifacts.WheelArtifact
    )

    def reject_member_read(*args, **kwargs):
        raise AssertionError("oversized archive member was read before validation")

    monkeypatch.setattr(artifact_type, "read", reject_member_read)

    with pytest.raises(ValueError, match="archive member.*384 MiB per-member uncompressed limit"):
        check_release_artifacts.check_artifact_contents(artifact)


@pytest.mark.parametrize("suffix", [".tar.gz", ".whl"], ids=["sdist", "wheel"])
def test_content_scan_rejects_oversized_total_decompressed_contents_before_reading_members(
    tmp_path,
    suffix,
    monkeypatch,
):
    artifact = _artifact_path(tmp_path, suffix, "oversized-total")
    _write_archive_members(
        artifact,
        {
            "project/first.bin": b"a" * 400,
            "project/second.bin": b"b" * 400,
        },
    )
    monkeypatch.setattr(check_release_artifacts, "MAX_ARTIFACT_UNCOMPRESSED_BYTES", 600)
    artifact_type = (
        check_release_artifacts.SdistArtifact if suffix == ".tar.gz" else check_release_artifacts.WheelArtifact
    )

    def reject_member_read(*args, **kwargs):
        raise AssertionError("oversized archive contents were read before validation")

    monkeypatch.setattr(artifact_type, "read", reject_member_read)

    with pytest.raises(ValueError, match="decompressed contents exceed.*512 MiB total uncompressed limit"):
        check_release_artifacts.check_artifact_contents(artifact)


@pytest.mark.parametrize(
    ("mutation", "member_limit", "expected_message"),
    [
        ("underreported-count", 1, "wheel contains more than 1 archive members"),
        ("spanned-member", 2, "wheel must contain one non-spanned ZIP archive"),
    ],
)
def test_content_scan_validates_raw_wheel_directory_before_constructing_zip_reader(
    tmp_path,
    monkeypatch,
    mutation,
    member_limit,
    expected_message,
):
    artifact = _artifact_path(tmp_path, ".whl", "member-limit")
    _write_archive_members(
        artifact,
        {
            "project/first.bin": b"first",
            "project/second.bin": b"second",
        },
    )
    contents = bytearray(artifact.read_bytes())
    if mutation == "underreported-count":
        end_record_offset = contents.rfind(b"PK\x05\x06")
        assert end_record_offset >= 0
        contents[end_record_offset + 8 : end_record_offset + 12] = (1).to_bytes(2, "little") * 2
    else:
        member_offset = contents.find(b"PK\x01\x02")
        assert member_offset >= 0
        contents[member_offset + 34 : member_offset + 36] = (1).to_bytes(2, "little")
    artifact.write_bytes(contents)

    def reject_zip_reader_construction(*args, **kwargs):
        raise AssertionError("wheel ZIP reader was constructed before its member-count preflight")

    monkeypatch.setattr(check_release_artifacts, "MAX_ARTIFACT_MEMBERS", member_limit)
    monkeypatch.setattr(archive_safety.zipfile, "ZipFile", reject_zip_reader_construction)

    with pytest.raises(ValueError, match=expected_message):
        check_release_artifacts.check_artifact_contents(artifact)


def test_content_scan_bounds_sdist_members_before_materializing_tar_metadata(tmp_path, monkeypatch):
    artifact = _artifact_path(tmp_path, ".tar.gz", "member-limit")
    _write_archive_members(
        artifact,
        {
            "project/first": b"",
            "project/second": b"",
        },
    )

    def reject_member_materialization(*args, **kwargs):
        raise AssertionError("sdist members were materialized before their streaming count preflight")

    monkeypatch.setattr(check_release_artifacts, "MAX_ARTIFACT_MEMBERS", 1)
    monkeypatch.setattr(archive_safety.tarfile.TarFile, "getmembers", reject_member_materialization)

    with pytest.raises(ValueError, match="sdist contains more than 1 archive members"):
        check_release_artifacts.check_artifact_contents(artifact)


def test_archive_snapshot_is_private_read_only_and_removed(tmp_path):
    artifact = tmp_path / "artifact.whl"
    artifact.write_bytes(b"bounded snapshot")

    with archive_safety.snapshot_archive(
        artifact,
        max_bytes=len(b"bounded snapshot"),
        description="artifact",
        size_limit_description="the test limit",
    ) as snapshot:
        snapshot_path = snapshot.path
        snapshot_directory = snapshot_path.parent
        snapshot_file = snapshot.file
        assert snapshot.source_path == artifact
        assert snapshot_path.name == artifact.name
        assert snapshot_path.read_bytes() == b"bounded snapshot"
        if os.name != "nt":
            assert snapshot_path.stat().st_mode & 0o777 == 0o400
            assert snapshot_directory.stat().st_mode & 0o077 == 0

    assert snapshot_file.closed
    assert not snapshot_path.exists()
    assert not snapshot_directory.exists()


@pytest.mark.skipif(os.name == "nt", reason="Windows rejects replacing an open snapshot file")
def test_archive_snapshot_detects_replacement_of_its_private_named_path(tmp_path):
    artifact = tmp_path / "artifact.whl"
    artifact.write_bytes(b"bounded snapshot")

    with archive_safety.snapshot_archive(
        artifact,
        max_bytes=len(b"bounded snapshot"),
        description="artifact",
        size_limit_description="the test limit",
    ) as snapshot:
        replacement = snapshot.path.with_name("replacement")
        replacement.write_bytes(b"bounded snapshot")
        replacement.replace(snapshot.path)

        with pytest.raises(ValueError, match="private artifact snapshot changed after validation"):
            snapshot.validate_named_path(description="artifact")
        snapshot.file.seek(0)
        assert snapshot.file.read() == b"bounded snapshot"


def test_archive_snapshot_rejects_non_regular_inputs_before_allocating_storage(tmp_path, monkeypatch):
    def reject_temporary_directory(*_args, **_kwargs):
        raise AssertionError("non-regular input reached snapshot allocation")

    monkeypatch.setattr(archive_safety.tempfile, "TemporaryDirectory", reject_temporary_directory)
    with pytest.raises(ValueError, match="(?:must be a regular file|could not snapshot artifact)"):
        with archive_safety.snapshot_archive(
            tmp_path,
            max_bytes=1024,
            description="artifact",
            size_limit_description="the test limit",
        ):
            pass


def test_archive_snapshot_rejects_oversized_inputs_before_allocating_storage(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.whl"
    artifact.write_bytes(b"too large")

    def reject_temporary_directory(*_args, **_kwargs):
        raise AssertionError("oversized input reached snapshot allocation")

    monkeypatch.setattr(archive_safety.tempfile, "TemporaryDirectory", reject_temporary_directory)
    with pytest.raises(ValueError, match="artifact exceeds the test limit"):
        with archive_safety.snapshot_archive(
            artifact,
            max_bytes=len(b"too large") - 1,
            description="artifact",
            size_limit_description="the test limit",
        ):
            pass


@pytest.mark.skipif(os.name == "nt", reason="symbolic-link creation is not generally available on Windows")
def test_archive_snapshot_rejects_symbolic_links_before_allocating_storage(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.whl"
    artifact.write_bytes(b"archive")
    link = tmp_path / "linked.whl"
    link.symlink_to(artifact)

    def reject_temporary_directory(*_args, **_kwargs):
        raise AssertionError("symbolic-link input reached snapshot allocation")

    monkeypatch.setattr(archive_safety.tempfile, "TemporaryDirectory", reject_temporary_directory)
    with pytest.raises(ValueError, match="must be a regular file"):
        with archive_safety.snapshot_archive(
            link,
            max_bytes=1024,
            description="artifact",
            size_limit_description="the test limit",
        ):
            pass


@pytest.mark.skipif(os.name == "nt", reason="named pipes are POSIX filesystem objects")
def test_archive_snapshot_rejects_named_pipes_before_allocating_storage(tmp_path, monkeypatch):
    pipe = tmp_path / "artifact.whl"
    os.mkfifo(pipe)

    def reject_temporary_directory(*_args, **_kwargs):
        raise AssertionError("named-pipe input reached snapshot allocation")

    monkeypatch.setattr(archive_safety.tempfile, "TemporaryDirectory", reject_temporary_directory)
    with pytest.raises(ValueError, match="must be a regular file"):
        with archive_safety.snapshot_archive(
            pipe,
            max_bytes=1024,
            description="artifact",
            size_limit_description="the test limit",
        ):
            pass


@pytest.mark.parametrize("suffix", [".whl", ".tar.gz"], ids=["zip", "compressed-tar"])
def test_release_preflight_and_parser_use_the_same_snapshot(tmp_path, monkeypatch, suffix):
    artifact = _artifact_path(tmp_path, suffix, "original")
    replacement = _artifact_path(tmp_path, suffix, "replacement")
    original_member = "project/original"
    _write_archive(artifact, original_member, b"original")
    _write_archive_members(
        replacement,
        {
            "project/replacement-one": b"replacement",
            "project/replacement-two": b"replacement",
        },
    )
    validator_name = "validate_tar_member_count" if suffix == ".tar.gz" else "validate_zip_member_count"
    validate_snapshot = getattr(archive_safety, validator_name)
    replaced = False

    def validate_then_replace(*args, **kwargs):
        nonlocal replaced
        member_count = validate_snapshot(*args, **kwargs)
        if not replaced:
            replacement.replace(artifact)
            replaced = True
        return member_count

    monkeypatch.setattr(archive_safety, validator_name, validate_then_replace)
    monkeypatch.setattr(check_release_artifacts, "MAX_ARTIFACT_MEMBERS", 1)
    artifact_type = (
        check_release_artifacts.SdistArtifact if suffix == ".tar.gz" else check_release_artifacts.WheelArtifact
    )

    inspected = artifact_type(artifact)
    try:
        assert inspected.names() == [original_member]
        assert inspected.read(original_member) == b"original"
    finally:
        inspected.close()
    assert replaced


def test_uncompressed_tar_preflight_and_parser_use_the_same_snapshot(tmp_path, monkeypatch):
    artifact = tmp_path / "original.tar"
    replacement = tmp_path / "replacement.tar"
    with tarfile.open(artifact, mode="w:") as archive:
        member = tarfile.TarInfo("project/original")
        member.size = len(b"original")
        archive.addfile(member, io.BytesIO(b"original"))
    with tarfile.open(replacement, mode="w:") as archive:
        for name in ("project/replacement-one", "project/replacement-two"):
            member = tarfile.TarInfo(name)
            archive.addfile(member, io.BytesIO())

    validate_snapshot = archive_safety.validate_tar_member_count

    def validate_then_replace(*args, **kwargs):
        member_count = validate_snapshot(*args, **kwargs)
        replacement.replace(artifact)
        return member_count

    monkeypatch.setattr(archive_safety, "validate_tar_member_count", validate_then_replace)
    with archive_safety.snapshot_archive(
        artifact,
        max_bytes=1024 * 1024,
        description="test TAR",
        size_limit_description="the test limit",
    ) as snapshot:
        with archive_safety.open_tar_snapshot(
            snapshot,
            max_members=1,
            max_member_bytes=1024,
            max_total_bytes=1024,
            member_limit_description="the test member limit",
            total_limit_description="the test total limit",
            description="test TAR",
        ) as archive:
            members = archive.getmembers()
            assert [member.name for member in members] == ["project/original"]
            assert archive.extractfile(members[0]).read() == b"original"


def test_failed_archive_construction_removes_its_snapshot(tmp_path, monkeypatch):
    artifact = _artifact_path(tmp_path, ".whl", "cleanup")
    _write_archive(artifact, "project/member", b"content")
    snapshot_paths: list[Path] = []
    snapshot_files = []
    snapshot_archive = check_release_artifacts.snapshot_archive

    @contextmanager
    def record_snapshot(*args, **kwargs):
        with snapshot_archive(*args, **kwargs) as snapshot:
            snapshot_paths.append(snapshot.path)
            snapshot_files.append(snapshot.file)
            yield snapshot

    monkeypatch.setattr(check_release_artifacts, "snapshot_archive", record_snapshot)
    monkeypatch.setattr(check_release_artifacts, "MAX_ARTIFACT_MEMBERS", 0)
    with pytest.raises(ValueError, match="wheel contains more than 0 archive members"):
        check_release_artifacts.WheelArtifact(artifact)

    assert len(snapshot_paths) == 1
    assert snapshot_files[0].closed
    assert not snapshot_paths[0].exists()
    assert not snapshot_paths[0].parent.exists()


def test_sdist_streaming_preflight_normalizes_corrupt_deflate_errors(tmp_path, monkeypatch):
    artifact = tmp_path / "corrupt-deflate.tar.gz"
    artifact.write_bytes(b"\x1f\x8bcorrupt")

    class CorruptGzipStream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            raise archive_safety.zlib.error("corrupt deflate stream")

    monkeypatch.setattr(archive_safety.gzip, "GzipFile", lambda **_kwargs: CorruptGzipStream())

    with pytest.raises(ValueError, match="could not inspect sdist"):
        _validate_tar_member_count(
            artifact,
            max_members=10,
            max_member_bytes=2048,
            max_total_bytes=4096,
            member_limit_description="the uncompressed test limit",
            total_limit_description="the uncompressed test limit",
            description="sdist",
        )


@pytest.mark.parametrize(
    "member_type",
    [tarfile.REGTYPE, tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME],
    ids=["regular", "pax", "gnu-long-name"],
)
def test_sdist_streaming_preflight_rejects_oversized_header_before_advancing(
    tmp_path,
    monkeypatch,
    member_type,
):
    artifact = tmp_path / "oversized.tar.gz"
    oversized_member = tarfile.TarInfo("project/oversized.bin")
    oversized_member.type = member_type
    oversized_member.size = 1024
    with gzip.open(artifact, mode="wb") as compressed:
        compressed.write(oversized_member.tobuf(format=tarfile.GNU_FORMAT))
        compressed.write(_runtime_sentinel())

    def reject_payload_read(*_args, **_kwargs):
        raise AssertionError("sdist stream advanced across an oversized member payload")

    monkeypatch.setattr(archive_safety, "_read_tar_payload", reject_payload_read)

    with pytest.raises(ValueError, match="archive member.*uncompressed test limit"):
        _validate_tar_member_count(
            artifact,
            max_members=10,
            max_member_bytes=512,
            max_total_bytes=2048,
            member_limit_description="the uncompressed test limit",
            total_limit_description="the uncompressed test limit",
            description="sdist",
        )


@pytest.mark.parametrize("member_type", [tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME], ids=["pax", "gnu-long-name"])
def test_sdist_streaming_preflight_bounds_extension_header_payloads(tmp_path, monkeypatch, member_type):
    artifact = tmp_path / "oversized-extension-header.tar.gz"
    extension_header = tarfile.TarInfo("project/extension-header")
    extension_header.type = member_type
    extension_header.size = 1024
    with gzip.open(artifact, mode="wb") as compressed:
        compressed.write(extension_header.tobuf(format=tarfile.GNU_FORMAT))
        compressed.write(_runtime_sentinel())

    def reject_payload_read(*_args, **_kwargs):
        raise AssertionError("TAR extension header payload was read before its metadata limit was checked")

    monkeypatch.setattr(archive_safety, "_MAX_TAR_EXTENSION_HEADER_BYTES", 512)
    monkeypatch.setattr(archive_safety, "_read_tar_payload", reject_payload_read)

    with pytest.raises(ValueError, match="TAR extension header.*bounded 1 MiB metadata limit"):
        _validate_tar_member_count(
            artifact,
            max_members=10,
            max_member_bytes=2048,
            max_total_bytes=4096,
            member_limit_description="the uncompressed test limit",
            total_limit_description="the uncompressed test limit",
            description="sdist",
        )


def test_sdist_streaming_preflight_rejects_gnu_sparse_before_special_parsing(tmp_path, monkeypatch):
    artifact = tmp_path / "gnu-sparse.tar.gz"
    sparse_member = tarfile.TarInfo("project/sparse.bin")
    sparse_member.type = tarfile.GNUTYPE_SPARSE
    with gzip.open(artifact, mode="wb") as compressed:
        compressed.write(sparse_member.tobuf(format=tarfile.GNU_FORMAT))
        compressed.write(b"\0" * 1024)

    def reject_payload_read(*_args, **_kwargs):
        raise AssertionError("GNU sparse metadata reached payload parsing")

    monkeypatch.setattr(archive_safety, "_read_tar_payload", reject_payload_read)

    with pytest.raises(ValueError, match="unsupported TAR member type"):
        _validate_tar_member_count(
            artifact,
            max_members=10,
            max_member_bytes=2048,
            max_total_bytes=4096,
            member_limit_description="the uncompressed test limit",
            total_limit_description="the uncompressed test limit",
            description="sdist",
        )


def test_sdist_streaming_preflight_matches_tarfile_for_nonzero_directory_sizes(tmp_path):
    artifact = tmp_path / "directory-size.tar.gz"
    directory = tarfile.TarInfo("project")
    directory.type = tarfile.DIRTYPE
    directory.size = tarfile.BLOCKSIZE
    regular = tarfile.TarInfo("project/member")
    with gzip.open(artifact, mode="wb") as compressed:
        compressed.write(directory.tobuf(format=tarfile.PAX_FORMAT))
        compressed.write(_raw_tar_member(regular))
        compressed.write(b"\0" * (2 * tarfile.BLOCKSIZE))

    assert (
        _validate_tar_member_count(
            artifact,
            max_members=10,
            max_member_bytes=2048,
            max_total_bytes=4096,
            member_limit_description="the uncompressed test limit",
            total_limit_description="the uncompressed test limit",
            description="sdist",
        )
        == 2
    )


def test_sdist_streaming_preflight_matches_tarfile_local_pax_size_precedence(tmp_path):
    artifact = tmp_path / "chained-local-pax.tar.gz"
    first_payload = _pax_record(b"size", b"0")
    first = tarfile.TarInfo("first-pax")
    first.type = tarfile.XHDTYPE
    first.size = len(first_payload)
    second_payload = _pax_record(b"size", str(tarfile.BLOCKSIZE).encode("ascii"))
    second = tarfile.TarInfo("second-pax")
    second.type = tarfile.XHDTYPE
    second.size = len(second_payload)
    regular = tarfile.TarInfo("project/member")
    with gzip.open(artifact, mode="wb") as compressed:
        compressed.write(_raw_tar_member(first, first_payload))
        compressed.write(_raw_tar_member(second, second_payload))
        compressed.write(_raw_tar_member(regular))
        compressed.write(b"\0" * (2 * tarfile.BLOCKSIZE))

    assert (
        _validate_tar_member_count(
            artifact,
            max_members=10,
            max_member_bytes=2048,
            max_total_bytes=4096,
            member_limit_description="the uncompressed test limit",
            total_limit_description="the uncompressed test limit",
            description="sdist",
        )
        == 3
    )


def test_sdist_streaming_preflight_rejects_ambiguous_global_pax_size(tmp_path):
    artifact = tmp_path / "global-pax-size.tar.gz"
    payload = _pax_record(b"size", b"0")
    header = tarfile.TarInfo("global-pax")
    header.type = tarfile.XGLTYPE
    header.size = len(payload)
    with gzip.open(artifact, mode="wb") as compressed:
        compressed.write(_raw_tar_member(header, payload))
        compressed.write(b"\0" * (2 * tarfile.BLOCKSIZE))

    with pytest.raises(ValueError, match="unsupported global PAX size override"):
        _validate_tar_member_count(
            artifact,
            max_members=10,
            max_member_bytes=2048,
            max_total_bytes=4096,
            member_limit_description="the uncompressed test limit",
            total_limit_description="the uncompressed test limit",
            description="sdist",
        )


def test_sdist_streaming_preflight_rejects_a_concatenated_gzip_payload_after_the_terminator(tmp_path):
    artifact = tmp_path / "concatenated-payload.tar.gz"
    member = tarfile.TarInfo("project/member")
    member.size = len(b"clean")
    first_stream = _raw_tar_member(member, b"clean") + b"\0" * (2 * tarfile.BLOCKSIZE)
    artifact.write_bytes(gzip.compress(first_stream) + gzip.compress(_runtime_sentinel()))

    with pytest.raises(ValueError, match="nonzero data after its TAR terminator"):
        _validate_tar_member_count(
            artifact,
            max_members=10,
            max_member_bytes=2048,
            max_total_bytes=4096,
            member_limit_description="the uncompressed test limit",
            total_limit_description="the uncompressed test limit",
            description="sdist",
        )


def test_sdist_streaming_preflight_accepts_bounded_zero_padding_after_the_terminator(tmp_path):
    artifact = tmp_path / "bounded-zero-padding.tar.gz"
    member = tarfile.TarInfo("project/member")
    member.size = len(b"clean")
    stream = (
        _raw_tar_member(member, b"clean")
        + b"\0" * (2 * tarfile.BLOCKSIZE)
        + b"\0" * archive_safety._MAX_TAR_TRAILING_ZERO_BYTES
    )
    artifact.write_bytes(gzip.compress(stream))

    assert (
        _validate_tar_member_count(
            artifact,
            max_members=10,
            max_member_bytes=2048,
            max_total_bytes=4096,
            member_limit_description="the uncompressed test limit",
            total_limit_description="the uncompressed test limit",
            description="sdist",
        )
        == 1
    )


@pytest.mark.parametrize(
    "member_type",
    [tarfile.REGTYPE, tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME],
    ids=["regular", "pax", "gnu-long-name"],
)
def test_content_scan_rejects_nonzero_tar_payload_padding(tmp_path, member_type):
    sentinel = _runtime_sentinel()
    rule = check_release_artifacts.LiteralContentRule("runtime-tar-padding", sentinel)
    artifact = _artifact_path(tmp_path, ".tar.gz", f"padding-{member_type.decode('ascii')}")

    if member_type == tarfile.REGTYPE:
        payload = b"clean"
        member = tarfile.TarInfo("project/member")
    elif member_type == tarfile.XHDTYPE:
        payload = _pax_record(b"comment", b"clean")
        member = tarfile.TarInfo("project/pax-header")
    else:
        payload = b"project/long-name\0"
        member = tarfile.TarInfo("././@LongLink")
    member.type = member_type
    member.size = len(payload)
    encoded_member = bytearray(_raw_tar_member(member, payload))
    padding_offset = tarfile.BLOCKSIZE + len(payload)
    encoded_member[padding_offset : padding_offset + len(sentinel)] = sentinel

    tar_stream = bytes(encoded_member)
    if member_type != tarfile.REGTYPE:
        regular = tarfile.TarInfo("project/member")
        regular.size = len(b"clean")
        tar_stream += _raw_tar_member(regular, b"clean")
    artifact.write_bytes(gzip.compress(tar_stream + b"\0" * (2 * tarfile.BLOCKSIZE)))
    assert sentinel not in artifact.read_bytes()
    # Python's PAX parser rejects nonzero padding itself. The other cases
    # demonstrate padding that TarFile accepts without exposing to callers.
    if member_type != tarfile.XHDTYPE:
        with tarfile.open(artifact, mode="r:gz") as archive:
            archive.getmembers()

    error = _rejected_by(
        lambda: check_release_artifacts.check_artifact_contents(
            artifact,
            content_rules=(rule,),
        )
    )
    assert "TAR payload padding must contain only zero bytes" in str(error)
    _assert_no_recoverable_value(sentinel, str(error), repr(error.args))


@pytest.mark.parametrize("metadata_kind", ["header", "pax", "gnu-long-name"])
def test_binary_content_scan_rejects_decompressed_sdist_metadata(
    tmp_path,
    monkeypatch,
    capsys,
    metadata_kind,
):
    sentinel = _runtime_sentinel()
    rule_id = "runtime-decompressed-tar-metadata"
    rule = check_release_artifacts.LiteralContentRule(rule_id, sentinel)
    artifact = _artifact_path(tmp_path, ".tar.gz", metadata_kind)
    regular = tarfile.TarInfo("project/member")
    regular.size = len(b"clean")

    if metadata_kind == "header":
        regular.name = f"project/{sentinel.decode('ascii')}"
        tar_stream = _raw_tar_member(regular, b"clean")
    elif metadata_kind == "pax":
        payload = _pax_record(b"comment", sentinel)
        extension_header = tarfile.TarInfo("project/pax-header")
        extension_header.type = tarfile.XHDTYPE
        extension_header.size = len(payload)
        tar_stream = _raw_tar_member(extension_header, payload) + _raw_tar_member(regular, b"clean")
    else:
        payload = b"project/" + sentinel + b"\0"
        extension_header = tarfile.TarInfo("././@LongLink")
        extension_header.type = tarfile.GNUTYPE_LONGNAME
        extension_header.size = len(payload)
        tar_stream = _raw_tar_member(extension_header, payload) + _raw_tar_member(regular, b"clean")

    artifact.write_bytes(gzip.compress(tar_stream + b"\0" * (2 * tarfile.BLOCKSIZE)))
    assert sentinel not in artifact.read_bytes()
    with tarfile.open(artifact, mode="r:gz") as archive:
        assert all(
            sentinel not in (archive.extractfile(member).read() if member.isfile() else b"")
            for member in archive.getmembers()
        )

    monkeypatch.setattr(archive_safety, "_TAR_READ_CHUNK_BYTES", 13)
    error = _rejected_by(
        lambda: check_release_artifacts.check_artifact_contents(
            artifact,
            content_rules=(rule,),
        )
    )

    _assert_metadata_only_error(
        error,
        sentinel=sentinel,
        rule_id=rule_id,
        member_name="decompressed TAR metadata",
        capsys=capsys,
    )


def test_binary_content_scan_rejects_unreferenced_zip_gap_bytes(tmp_path, monkeypatch, capsys):
    sentinel = _runtime_sentinel()
    rule_id = "runtime-raw-archive-content"
    rule = check_release_artifacts.LiteralContentRule(rule_id, sentinel)
    artifact = _artifact_path(tmp_path, ".whl", "raw-gap")
    member_name = "project/clean.bin"
    _write_archive(artifact, member_name, b"clean member contents")

    contents = bytearray(artifact.read_bytes())
    end_record_offset = contents.rfind(b"PK\x05\x06")
    assert end_record_offset >= 0
    central_directory_offset = int.from_bytes(
        contents[end_record_offset + 16 : end_record_offset + 20],
        "little",
    )
    contents[central_directory_offset:central_directory_offset] = sentinel
    relocated_end_record_offset = end_record_offset + len(sentinel)
    contents[relocated_end_record_offset + 16 : relocated_end_record_offset + 20] = (
        central_directory_offset + len(sentinel)
    ).to_bytes(4, "little")
    artifact.write_bytes(contents)

    with zipfile.ZipFile(artifact) as wheel:
        assert wheel.namelist() == [member_name]
        assert sentinel not in wheel.read(member_name)

    monkeypatch.setattr(
        check_release_artifacts,
        "_RAW_ARCHIVE_SCAN_CHUNK_BYTES",
        central_directory_offset + len(sentinel) // 2,
    )
    error = _rejected_by(
        lambda: check_release_artifacts.check_artifact_contents(
            artifact,
            content_rules=(rule,),
        )
    )

    _assert_metadata_only_error(
        error,
        sentinel=sentinel,
        rule_id=rule_id,
        member_name="raw archive bytes",
        capsys=capsys,
    )


def test_binary_content_scan_prefers_member_metadata_for_stored_zip_contents(tmp_path, capsys):
    sentinel = _runtime_sentinel()
    rule_id = "runtime-stored-member-content"
    rule = check_release_artifacts.LiteralContentRule(rule_id, sentinel)
    artifact = _artifact_path(tmp_path, ".whl", "stored-member")
    member_name = "project/security-probe.bin"
    with zipfile.ZipFile(artifact, mode="w", compression=zipfile.ZIP_STORED) as wheel:
        wheel.writestr(member_name, b"\0" + sentinel)
    assert sentinel in artifact.read_bytes()

    error = _rejected_by(
        lambda: check_release_artifacts.check_artifact_contents(
            artifact,
            content_rules=(rule,),
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
