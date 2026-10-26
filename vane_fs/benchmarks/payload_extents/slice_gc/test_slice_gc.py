# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Additional checks for the isolated pointer-map and slice-GC prototype."""

import json
import os
import subprocess
import sys

import pytest
from test_extents import sql

from vane_fs import Error, Workspace


def payload_bytes(path):
    return sql(path, "", "SELECT coalesce(sum(length(data)),0) FROM block_payloads")[0][0]


def logical_dump(path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import hashlib,sqlite3,sys; d=sqlite3.connect(sys.argv[1]); "
            "print(hashlib.sha256(chr(10).join(d.iterdump()).encode()).hexdigest()); d.close()",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=15,
    )
    return result.stdout.strip()


def test_pointer_maps_persist_and_old_unmapped_database_still_opens(tmp_path):
    path = tmp_path / "db.sqlite"
    with Workspace(path) as workspace:
        workspace.checkout().write_file("/file", b"mapped" * 20000)
        assert sql(path, "", "PRAGMA auto_vacuum") == [[2]]
    with Workspace(path) as workspace:
        assert sql(path, "", "PRAGMA auto_vacuum") == [[2]]
        assert workspace.checkout().read("/file") == b"mapped" * 20000
    # Simulate the preceding experimental format. Opening must not migrate it.
    sql(path, "PRAGMA journal_mode=DELETE; PRAGMA auto_vacuum=NONE; VACUUM;")
    with Workspace(path) as workspace:
        assert sql(path, "", "PRAGMA auto_vacuum") == [[0]]
        assert workspace.checkout().read("/file") == b"mapped" * 20000


def test_gc_remaps_shared_slices_for_branches_and_pinned_snapshot(tmp_path):
    path = tmp_path / "db.sqlite"
    extent = int(os.environ["VANE_FS_EXTENT_BYTES"])
    data = b"".join(bytes([i + 1]) * 4096 for i in range(extent // 4096))
    with Workspace(path) as workspace:
        main = workspace.checkout()
        main.write_file("/file", data)
        sparse = bytearray(data)
        for offset in range(4096, extent, 8192):
            main.write("/file", b"\0" * 4096, offset)
            sparse[offset : offset + 4096] = b"\0" * 4096
        child = workspace.checkout(workspace.fork("main", "child").id)
        snapshot = workspace.snapshot()
        main.write("/file", b"m" * 4096)
        child.write("/file", b"c" * 4096, 8192)
        main_data = b"m" * 4096 + sparse[4096:]
        child_data = sparse[:8192] + b"c" * 4096 + sparse[12288:]
        with Workspace(path) as other, other.open_snapshot(snapshot) as frozen:
            before = payload_bytes(path)
            workspace.collect_garbage()
            assert payload_bytes(path) == before - extent // 2
            assert frozen.read("/file") == sparse
            assert main.read("/file") == main_data
            assert other.checkout("child").read("/file") == child_data
            # The fork base and snapshot have multiple references to the same slices.
            assert sql(path, "", "SELECT count(*) FROM pragma_foreign_key_check") == [[0]]
            stable = logical_dump(path)
            workspace.collect_garbage()
            assert logical_dump(path) == stable
        preview = workspace.preview_merge("child", "main")
        assert preview.conflicts == ["/file"]
        workspace.merge(preview, {"/file": "source"})
        assert main.read("/file") == child_data


@pytest.mark.parametrize("stage", ["insert", "remap", "delete"])
def test_gc_failure_rolls_back_versions_payloads_and_remaps(tmp_path, stage):
    path = tmp_path / "db.sqlite"
    extent = int(os.environ["VANE_FS_EXTENT_BYTES"])
    with Workspace(path) as workspace:
        main = workspace.checkout()
        main.write_file("/file", b"a" * extent)
        snapshot = workspace.snapshot()
        main.write("/file", b"\0" * 4096)
        workspace.drop_snapshot(snapshot)
        operation = {
            "insert": "INSERT ON block_payloads",
            "remap": "UPDATE OF payload ON block_versions",
            "delete": "DELETE ON block_payloads",
        }[stage]
        limit = 2 if stage == "remap" else 1
        sql(
            path,
            "CREATE TABLE fault_counter AS SELECT 0 AS writes; "
            f"CREATE TRIGGER fail_compact BEFORE {operation} BEGIN "
            "UPDATE fault_counter SET writes=writes+1; "
            f"SELECT CASE WHEN (SELECT writes FROM fault_counter)={limit} "
            "THEN RAISE(ABORT,'injected compaction failure') END; END;",
        )
        before = logical_dump(path)
        with pytest.raises(Error, match="injected compaction failure"):
            workspace.collect_garbage()
        assert logical_dump(path) == before
        assert main.read("/file") == b"\0" * 4096 + b"a" * (extent - 4096)
        sql(path, "DROP TRIGGER fail_compact; DROP TABLE fault_counter;")
        workspace.collect_garbage()
        assert payload_bytes(path) == extent - 4096
        assert main.read("/file") == b"\0" * 4096 + b"a" * (extent - 4096)


def test_gc_wal_reader_keeps_original_payload_until_transaction_ends(tmp_path):
    path = tmp_path / "db.sqlite"
    extent = int(os.environ["VANE_FS_EXTENT_BYTES"])
    with Workspace(path) as workspace:
        main = workspace.checkout()
        main.write_file("/file", b"a" * extent)
        main.write("/file", b"\0" * 4096)
        query = "SELECT id,hex(data) FROM block_payloads ORDER BY id"
        code = (
            "import json,sqlite3,sys; d=sqlite3.connect(sys.argv[1]); d.execute('BEGIN'); "
            "before=d.execute(sys.argv[2]).fetchall(); print('ready',flush=True); "
            "sys.stdin.readline(); assert d.execute(sys.argv[2]).fetchall()==before; "
            "d.commit(); after=d.execute(sys.argv[2]).fetchall(); assert after!=before; "
            "print(json.dumps({'bytes':sum(len(r[1])//2 for r in after)})); d.close()"
        )
        child = subprocess.Popen(
            [sys.executable, "-I", "-c", code, str(path), query],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert child.stdout.readline().strip() == "ready"
            workspace.collect_garbage()
            output, error = child.communicate("continue\n", timeout=15)
            assert child.returncode == 0, error
            assert json.loads(output) == {"bytes": extent - 4096}
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
        assert main.read("/file") == b"\0" * 4096 + b"a" * (extent - 4096)


def test_gc_invalid_slice_rolls_back_then_recovers(tmp_path):
    path = tmp_path / "db.sqlite"
    extent = int(os.environ["VANE_FS_EXTENT_BYTES"])
    with Workspace(path) as workspace:
        main = workspace.checkout()
        main.write_file("/file", b"a" * extent)
        main.write("/file", b"\0" * 4096)
        sql(
            path,
            "CREATE TABLE saved AS SELECT * FROM block_versions; UPDATE block_versions "
            "SET payload_offset=262144 WHERE block=1 AND deleted=0;",
        )
        before = logical_dump(path)
        with pytest.raises(Error, match="Invalid block payload slice"):
            workspace.collect_garbage()
        assert logical_dump(path) == before
        sql(path, "DELETE FROM block_versions; INSERT INTO block_versions SELECT * FROM saved; DROP TABLE saved;")
        workspace.collect_garbage()
        assert main.read("/file") == b"\0" * 4096 + b"a" * (extent - 4096)
