# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import secrets
import string
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


def _runtime_ipv4() -> bytes:
    return f"10.{secrets.randbelow(256)}.{secrets.randbelow(256)}.{secrets.randbelow(256)}".encode("ascii")


def _runtime_path() -> bytes:
    def segment() -> str:
        return "".join(secrets.choice(string.ascii_lowercase) for _ in range(4))

    return f"/{segment()}/{segment()}/".encode("ascii")


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
    if any(sentinel in surface.encode("utf-8", errors="replace") for surface in surfaces):
        pytest.fail("content error exposed the matched value", pytrace=False)


@pytest.fixture
def content_check_only(monkeypatch):
    monkeypatch.setattr(check_release_artifacts, "_check_sdist", lambda _artifact: None)
    monkeypatch.setattr(check_release_artifacts, "_check_wheel", lambda _artifact: None)


def test_default_configuration_has_production_content_rules():
    assert tuple((rule.rule_id, type(rule)) for rule in check_release_artifacts.CONTENT_RULES) == (
        ("release-internal-ip", check_release_artifacts.IPv4FingerprintRule),
        ("release-credential", check_release_artifacts.CredentialFingerprintRule),
    )
    assert tuple((rule.rule_id, type(rule)) for rule in check_release_artifacts.TEXT_CONTENT_RULES) == (
        ("release-local-path", check_release_artifacts.PosixPathFingerprintRule),
    )


def test_credential_rule_uses_memory_hard_fingerprint(monkeypatch):
    sentinel = _runtime_sentinel()
    rule_id = "runtime-sensitive-content"
    calls: list[tuple[str, bytes]] = []

    def memory_hard_fingerprint(candidate_rule_id: str, value: bytes) -> bytes:
        calls.append((candidate_rule_id, value))
        return hashlib.sha256(b"test-kdf\0" + candidate_rule_id.encode() + b"\0" + value).digest()

    monkeypatch.setattr(
        check_release_artifacts,
        "_memory_hard_fingerprint",
        memory_hard_fingerprint,
    )

    rule = check_release_artifacts.CredentialFingerprintRule.from_value(rule_id, sentinel)
    assert calls == [(rule_id, sentinel)]

    calls.clear()
    assert rule.matches(b"prefix-" + sentinel + b"-suffix")
    assert calls == [(rule_id, sentinel)]


@pytest.mark.parametrize(
    ("rule_type", "sentinel_factory"),
    [
        pytest.param(
            check_release_artifacts.CredentialFingerprintRule,
            _runtime_sentinel,
            id="credential",
        ),
        pytest.param(check_release_artifacts.IPv4FingerprintRule, _runtime_ipv4, id="ipv4"),
        pytest.param(check_release_artifacts.PosixPathFingerprintRule, _runtime_path, id="path"),
    ],
)
def test_rule_strategies_match_runtime_values(rule_type, sentinel_factory):
    sentinel = sentinel_factory()
    rule = rule_type.from_value("runtime-sensitive-content", sentinel)

    if not rule.matches(b"prefix-" + sentinel + b"-suffix"):
        pytest.fail("runtime rule did not detect its generated value", pytrace=False)
    if rule.matches(b"clean artifact content"):
        pytest.fail("runtime rule rejected clean content", pytrace=False)


def test_path_rule_prefilters_candidate_rich_clean_content(monkeypatch):
    sentinel = _runtime_path()
    rule = check_release_artifacts.PosixPathFingerprintRule.from_value(
        "runtime-sensitive-content",
        sentinel,
    )
    sentinel_tag = hashlib.sha256(sentinel).digest()[: check_release_artifacts.CANDIDATE_TAG_BYTES]
    candidates: list[bytes] = []
    for index in range(1 << 16):
        candidate = f"/{index:04x}/{index ^ 0xFFFF:04x}/".encode()
        candidate_tag = hashlib.sha256(candidate).digest()[: check_release_artifacts.CANDIDATE_TAG_BYTES]
        if candidate != sentinel and candidate_tag != sentinel_tag:
            candidates.append(candidate)
        if len(candidates) == 4096:
            break
    assert len(candidates) == 4096

    kdf_candidates: list[bytes] = []

    def memory_hard_fingerprint(_rule_id: str, value: bytes) -> bytes:
        kdf_candidates.append(value)
        return bytes(check_release_artifacts.FINGERPRINT_BYTES)

    monkeypatch.setattr(
        check_release_artifacts,
        "_memory_hard_fingerprint",
        memory_hard_fingerprint,
    )

    assert not rule.matches(b"\n".join(candidates))
    assert kdf_candidates == []

    monkeypatch.setattr(
        check_release_artifacts,
        "_candidate_tag",
        lambda _value: rule.candidate_tag,
    )
    assert not rule.matches(candidates[0])
    assert kdf_candidates == [candidates[0]]


@pytest.mark.parametrize(
    ("rule_type", "value_factory", "message"),
    [
        pytest.param(
            check_release_artifacts.CredentialFingerprintRule,
            _runtime_path,
            "credential-shaped",
            id="credential",
        ),
        pytest.param(
            check_release_artifacts.IPv4FingerprintRule,
            _runtime_sentinel,
            "IPv4 address",
            id="ipv4",
        ),
        pytest.param(
            check_release_artifacts.PosixPathFingerprintRule,
            _runtime_sentinel,
            "absolute path",
            id="path",
        ),
    ],
)
def test_rule_types_reject_wrong_candidate_shapes(rule_type, value_factory, message):
    with pytest.raises(ValueError, match=message):
        rule_type.from_value("runtime-sensitive-content", value_factory())


@pytest.mark.parametrize(
    ("rule_type", "sentinel_factory", "text_only"),
    [
        pytest.param(
            check_release_artifacts.CredentialFingerprintRule,
            _runtime_sentinel,
            False,
            id="credential",
        ),
        pytest.param(
            check_release_artifacts.IPv4FingerprintRule,
            _runtime_ipv4,
            False,
            id="ipv4",
        ),
        pytest.param(
            check_release_artifacts.PosixPathFingerprintRule,
            _runtime_path,
            True,
            id="path",
        ),
    ],
)
@pytest.mark.parametrize("suffix", [".tar.gz", ".whl"], ids=["sdist", "wheel"])
def test_cli_uses_default_rule_source_without_exposing_match(
    tmp_path,
    suffix,
    rule_type,
    sentinel_factory,
    text_only,
    content_check_only,
    monkeypatch,
    capsys,
):
    sentinel = sentinel_factory()
    rule_id = "runtime-sensitive-content"
    member_name = "project/security-probe.txt"
    rule = rule_type.from_value(rule_id, sentinel)
    contaminated = tmp_path / f"contaminated{suffix}"
    clean = tmp_path / f"clean{suffix}"
    _write_archive(contaminated, member_name, b"prefix-" + sentinel + b"-suffix")
    _write_archive(clean, member_name, b"clean artifact content")
    monkeypatch.setattr(check_release_artifacts, "CONTENT_RULES", () if text_only else (rule,))
    monkeypatch.setattr(check_release_artifacts, "TEXT_CONTENT_RULES", (rule,) if text_only else ())
    monkeypatch.setattr(sys, "argv", ["check_release_artifacts.py", str(contaminated)])

    error = _rejected_by(check_release_artifacts.main)

    _assert_metadata_only_error(
        error,
        sentinel=sentinel,
        rule_id=rule_id,
        member_name=member_name,
        capsys=capsys,
    )
    monkeypatch.setattr(sys, "argv", ["check_release_artifacts.py", str(clean)])
    assert check_release_artifacts.main() == 0
    captured = capsys.readouterr()
    if sentinel in (captured.out + captured.err).encode("utf-8", errors="replace"):
        pytest.fail("CLI output exposed the matched value", pytrace=False)


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
    rule = check_release_artifacts.CredentialFingerprintRule.from_value(rule_id, sentinel)
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
    rule = check_release_artifacts.PosixPathFingerprintRule.from_value(rule_id, sentinel)
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
