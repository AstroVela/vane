# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Behavior checks for the isolated Linux external-payload candidate."""

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vane_fs import Error, Workspace


def sql(path, query, script=""):
    command = (
        "import sqlite3,json,sys; d=sqlite3.connect(sys.argv[1]); "
        "d.executescript(sys.argv[3]); print(json.dumps(d.execute(sys.argv[2]).fetchall())); d.close()"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", command, str(path), query, script],
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    return json.loads(result.stdout)


def data(blocks):
    return b"".join((i + 1).to_bytes(4, "little") * 1024 for i in range(blocks))


def test_large_payloads_and_small_inline_updates(tmp_path):
    path = tmp_path / "workspace.sqlite"
    original = data(256)
    with Workspace(path) as workspace:
        main = workspace.checkout()
        main.write_file("/large", original)
        assert sql(path, "SELECT length(data),count(*) FROM block_payloads GROUP BY length(data)") == [[8, 256]]
        assert Path(str(path) + ".payload").stat().st_size == len(original) + 4096
        snapshot = workspace.snapshot()
        expected = bytearray(original)
        for i in (0, 17, 128, 255):
            main.write("/large", b"z" * 4096, i * 4096)
            expected[i * 4096 : (i + 1) * 4096] = b"z" * 4096
        for i in (1, 18, 129):
            main.write("/large", bytes(4096), i * 4096)
            expected[i * 4096 : (i + 1) * 4096] = bytes(4096)
        assert main.read("/large") == expected
        assert main.read("/large", 4001, 90000) == expected[4001:94001]
        workspace.collect_garbage()
        with workspace.open_snapshot(snapshot) as frozen:
            assert frozen.read("/large") == original
        workspace.drop_snapshot(snapshot)
        workspace.collect_garbage()
        assert main.read("/large") == expected
        assert sql(path, "SELECT length(data),count(*) FROM block_payloads GROUP BY length(data)") == [
            [8, 249],
            [4096, 4],
        ]
    with Workspace(path) as workspace:
        assert workspace.checkout().read("/large") == expected


def test_small_files_stay_inline(tmp_path):
    path = tmp_path / "workspace.sqlite"
    with Workspace(path) as workspace:
        main = workspace.checkout()
        main.write_file("/small", data(32))
        main.write("/small", b"x" * 4096, 4096)
        assert sql(path, "SELECT DISTINCT length(data) FROM block_payloads") == [[4096]]
        assert Path(str(path) + ".payload").stat().st_size == 4096


def test_snapshot_branch_and_sparse_physical_gc(tmp_path):
    path = tmp_path / "workspace.sqlite"
    sidecar = Path(str(path) + ".payload")
    original = data(2048)
    with Workspace(path) as workspace:
        main = workspace.checkout()
        main.write_file("/file", original)
        workspace.fork("main", "child")
        child = workspace.checkout("child")
        sparse = bytearray(len(original))
        for offset in range(0, len(original), 256 * 1024):
            sparse[offset : offset + 4096] = original[offset : offset + 4096]
        child.write_file("/file", bytes(sparse))
        workspace.collect_garbage()
        assert main.read("/file") == original
        assert child.read("/file") == sparse
        workspace.merge(workspace.preview_merge("child", "main"))
        workspace.delete_branch("child")
        workspace.collect_garbage()
        assert main.read("/file") == sparse
        # The sparse overwrite reuses surviving extents and frees dead blocks.
        assert sidecar.stat().st_blocks * 512 <= 256 * 1024
        main.unlink("/file")
        workspace.collect_garbage()
        assert sidecar.stat().st_size == 4096
        assert sql(path, "SELECT count(*) FROM block_payloads") == [[0]]


def test_gc_lock_failure_is_retryable(tmp_path):
    path = tmp_path / "workspace.sqlite"
    with Workspace(path, timeout_ms=10) as workspace:
        main = workspace.checkout()
        main.write_file("/file", data(256))
        main.unlink("/file")
        with open(str(path) + ".payload", "rb") as held:
            fcntl.flock(held, fcntl.LOCK_SH)
            with pytest.raises(BlockingIOError, match="Payload file is busy"):
                workspace.collect_garbage()
            assert sql(path, "SELECT count(*) FROM block_payloads") == [[256]]
        workspace.collect_garbage()
        assert Path(str(path) + ".payload").stat().st_size == 4096


@pytest.mark.parametrize("fault", ["missing", "truncated", "wrong_workspace"])
def test_bad_sidecar_is_rejected(tmp_path, fault):
    path = tmp_path / "workspace.sqlite"
    sidecar = Path(str(path) + ".payload")
    with Workspace(path) as workspace:
        workspace.checkout().write_file("/file", data(256))
    if fault == "missing":
        sidecar.unlink()
        with pytest.raises(Error, match="Opening payload file"):
            Workspace(path)
        assert not sidecar.exists()
    elif fault == "truncated":
        os.truncate(sidecar, 4096 + 512)
        with Workspace(path) as workspace, pytest.raises(Error, match="Truncated payload file"):
            workspace.checkout().read("/file")
    else:
        with Workspace(tmp_path / "other.sqlite"):
            pass
        sidecar.write_bytes((tmp_path / "other.sqlite.payload").read_bytes())
        with pytest.raises(Error, match="different workspace"):
            Workspace(path)


def test_native_formats_refuse_each_other(tmp_path):
    production = os.environ["VANE_FS_PRODUCTION_PYTHON"]
    prototype = tmp_path / "prototype.sqlite"
    with Workspace(prototype) as workspace:
        workspace.checkout().write_file("/file", data(64))
    code = "import sys;from vane_fs import Workspace;Workspace(sys.argv[1])"
    result = subprocess.run([production, "-I", "-c", code, str(prototype)], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and "Not a VaneFS database" in result.stderr
    original = tmp_path / "production.sqlite"
    subprocess.run([production, "-I", "-c", code, str(original)], check=True, timeout=15)
    with pytest.raises(ValueError, match="Not a VaneFS database"):
        Workspace(original)
    assert not Path(str(original) + ".payload").exists()
