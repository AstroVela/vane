# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess

import pytest

from scripts import check_repository_binaries as checker


def _git(repository, *args):
    return subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True, text=True)


def _init_repository(tmp_path):
    _git(tmp_path, "init")
    return tmp_path


def test_repository_binary_check_rejects_tracked_elf(tmp_path):
    repository = _init_repository(tmp_path)
    binary = repository / "tool"
    binary.write_bytes(b"\x7fELF" + b"\0" * 32)
    _git(repository, "add", "tool")

    assert checker.check_repository(repository) == [
        "tool: tracked ELF executable binary is not allowed; keep reproducible source/build instructions instead"
    ]


@pytest.mark.parametrize(
    "magic",
    [
        b"\xca\xfe\xba\xbe",
        b"\xca\xfe\xba\xbf",
        b"\xbe\xba\xfe\xca",
        b"\xbf\xba\xfe\xca",
        b"\xfe\xed\xfa\xce",
        b"\xfe\xed\xfa\xcf",
        b"\xce\xfa\xed\xfe",
        b"\xcf\xfa\xed\xfe",
    ],
)
def test_repository_binary_check_rejects_tracked_macho_variants(tmp_path, magic):
    repository = _init_repository(tmp_path)
    binary = repository / "mach-o"
    binary.write_bytes(magic + b"\0" * 32)
    _git(repository, "add", "mach-o")

    assert checker.check_repository(repository) == [
        "mach-o: tracked Mach-O executable binary is not allowed; keep reproducible source/build instructions instead"
    ]


def test_repository_binary_check_preserves_backslashes_in_tracked_filenames(tmp_path):
    repository = _init_repository(tmp_path)
    binary = repository / r"escaped\binary"
    binary.write_bytes(b"\x7fELF" + b"\0" * 32)
    _git(repository, "add", r"escaped\binary")

    assert checker.check_repository(repository) == [
        r"escaped\binary: tracked ELF executable binary is not allowed; "
        "keep reproducible source/build instructions instead"
    ]


def test_repository_binary_check_rejects_unallowlisted_large_file(tmp_path):
    repository = _init_repository(tmp_path)
    payload = repository / "large-fixture.bin"
    payload.write_bytes(b"x" * (checker.MAX_TRACKED_FILE_BYTES + 1))
    _git(repository, "add", "large-fixture.bin")

    assert checker.check_repository(repository) == [
        "large-fixture.bin: tracked file is 5242881 bytes, above 5242880; "
        "add a narrow allowlist entry only for intentional fixtures"
    ]


def test_repository_binary_check_accepts_allowlisted_large_file(tmp_path, monkeypatch):
    repository = _init_repository(tmp_path)
    payload = repository / "fixtures" / "intentional.bin"
    payload.parent.mkdir()
    payload.write_bytes(b"x" * (checker.MAX_TRACKED_FILE_BYTES + 1))
    _git(repository, "add", "fixtures/intentional.bin")
    monkeypatch.setattr(checker, "LARGE_FILE_ALLOWLIST", frozenset({"fixtures/intentional.bin"}))

    assert checker.check_repository(repository) == []
