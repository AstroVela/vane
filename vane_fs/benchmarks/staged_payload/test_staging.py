# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed identity/record checks for the isolated format-1004 prototype."""

import hashlib
from pathlib import Path

import pytest

from vane_fs import Error, Workspace


@pytest.mark.parametrize("fault", ["missing", "truncated", "checksum", "wrong_workspace", "symlink"])
def test_unusable_fence_does_not_change_data(tmp_path, fault):
    path = tmp_path / "workspace.sqlite"
    fence = Path(str(path) + ".stage-fence")
    with Workspace(path) as workspace:
        workspace.checkout().write_file("/durable", b"d" * 1048576)
    if fault == "missing":
        fence.unlink()
    elif fault == "truncated":
        fence.write_bytes(fence.read_bytes()[:150])
    elif fault == "checksum":
        damaged = bytearray(fence.read_bytes())
        for offset in range(0, len(damaged), 4096):
            damaged[offset + 48] ^= 1
        fence.write_bytes(damaged)
    else:
        other = tmp_path / "other.sqlite"
        with Workspace(other):
            pass
        target = Path(str(other) + ".stage-fence")
        if fault == "wrong_workspace":
            fence.write_bytes(target.read_bytes())
        else:
            fence.unlink()
            fence.symlink_to(target)
    files = [path, Path(str(path) + "-wal"), Path(str(path) + ".payload")]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with pytest.raises(Error):
        Workspace(path)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
