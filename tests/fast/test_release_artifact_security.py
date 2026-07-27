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
from pathlib import Path

import pytest

from scripts import check_release_artifacts


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
    monkeypatch.setattr(check_release_artifacts, "_check_sdist", lambda _artifact: None)
    monkeypatch.setattr(check_release_artifacts, "_check_wheel", lambda _artifact: None)


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
    contaminated = tmp_path / f"contaminated{suffix}"
    clean = tmp_path / f"clean{suffix}"
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
    contaminated = tmp_path / f"contaminated{suffix}"
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
    artifact = tmp_path / f"binary{suffix}"
    _write_archive(artifact, member_name, b"\0" + sentinel)

    error = _rejected_by(
        lambda: check_release_artifacts.check_artifact(
            artifact,
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
    text_artifact = tmp_path / f"text{suffix}"
    binary_artifact = tmp_path / f"binary{suffix}"
    _write_archive(text_artifact, member_name, sentinel)
    _write_archive(binary_artifact, member_name, b"\0" + sentinel)

    error = _rejected_by(
        lambda: check_release_artifacts.check_artifact(
            text_artifact,
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
        content_rules=(),
        text_content_rules=(rule,),
    )
