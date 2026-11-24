# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import string
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows directory ACLs")


@pytest.fixture
def mapped_root(tmp_path):
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GetLogicalDrives.argtypes, api.GetLogicalDrives.restype = [], wintypes.DWORD
    api.DefineDosDeviceW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR]
    api.DefineDosDeviceW.restype = wintypes.BOOL
    used = api.GetLogicalDrives()
    if not used:
        raise ctypes.WinError(ctypes.get_last_error())
    letter = next(
        (letter for index, letter in reversed(list(enumerate(string.ascii_uppercase))) if not used & (1 << index)), None
    )
    if letter is None:
        pytest.skip("no unused drive letter for the isolated ACL fixture")
    target = tmp_path / "mapped-root"
    target.mkdir()
    name, destination = f"{letter}:", "\\??\\" + str(target.resolve())
    # A private DOS-device mapping avoids changing a real volume root's ACL.
    if not api.DefineDosDeviceW(0x1 | 0x8, name, destination):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield target, Path(name + "\\")
    finally:
        if not api.DefineDosDeviceW(0x1 | 0x2 | 0x4 | 0x8, name, destination):
            raise ctypes.WinError(ctypes.get_last_error())


@pytest.mark.parametrize(
    "right,root_replaceable", [(None, False), ("D", True), ("DC", True), ("WDAC", True), ("WO", True)]
)
def test_drive_alias_does_not_hide_a_replaceable_directory(mapped_root, right, root_replaceable):
    import vane

    directory, root = mapped_root
    vane._native._secure_dynamic_extension_cache_path(str(directory), directory=True)
    if right is not None:
        subprocess.run(
            ["icacls.exe", str(directory), "/grant", f"*S-1-5-11:({right})"], check=True, capture_output=True
        )
    inspect = vane._native._dynamic_extension_cache_path_is_replaceable
    assert inspect(str(directory)) is (right is not None)
    assert inspect(str(root)) is root_replaceable
    assert inspect("\\\\?\\" + str(root)) is root_replaceable
