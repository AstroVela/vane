# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import os
import signal
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from vane_fs import Workspace


@pytest.mark.skipif(sys.platform != "linux", reason="Owner-lock recovery is validated on Linux")
def test_recovery_retains_stopped_owners_and_releases_crashed_pins(tmp_path):
    path = tmp_path / "workspace.sqlite"
    with Workspace(path) as workspace:
        snapshot = workspace.snapshot()
        program = """
import sys, time
from vane_fs import Workspace
workspace = Workspace(sys.argv[1])
session = workspace.open_snapshot(sys.argv[2])
print('ready', flush=True)
time.sleep(120)
"""
        process = subprocess.Popen(
            [sys.executable, "-I", "-c", program, str(path), snapshot], stdout=subprocess.PIPE, text=True
        )
        try:
            assert process.stdout.readline().strip() == "ready"
            process.send_signal(signal.SIGSTOP)
            assert workspace.recover_owners().owners == 0
            with pytest.raises(BlockingIOError):
                workspace.drop_snapshot(snapshot)
            process.kill()
            process.wait(timeout=10)
            result = workspace.recover_owners()
            assert (result.owners, result.pins, result.mounts) == (1, 1, 0)
            workspace.drop_snapshot(snapshot)
            assert workspace.recover_owners().owners == 0
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            process.stdout.close()
    assert not list(Path(str(path) + ".vane_fs-locks").iterdir())


def test_live_connections_in_one_process_cannot_recover_each_other(tmp_path):
    path = tmp_path / "workspace.sqlite"
    with Workspace(path) as first, Workspace(path) as second:
        frozen = first.snapshot()
        with second.open_snapshot(frozen):
            assert first.recover_owners().owners == 0
            with pytest.raises(BlockingIOError):
                first.drop_snapshot(frozen)
        first.drop_snapshot(frozen)


@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.skipif(os.name == "nt", reason="Requires POSIX owner locks")
def test_missing_or_replaced_lock_does_not_prove_owner_dead(tmp_path, replace):
    path = tmp_path / "workspace.sqlite"
    with Workspace(path) as first, Workspace(path) as second:
        frozen = first.snapshot()
        with second.open_snapshot(frozen):
            with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
                lock = Path(db.execute("SELECT lock_path FROM owners JOIN pins ON owners.id=pins.owner").fetchone()[0])
            lock.unlink()
            if replace:
                lock.write_text("replacement")
            assert first.recover_owners().owners == 0
            with pytest.raises(BlockingIOError):
                first.drop_snapshot(frozen)


def test_v1_upgrade_preserves_unverifiable_legacy_pins(tmp_path):
    path = tmp_path / "workspace.sqlite"
    with Workspace(path) as workspace:
        workspace.checkout().write_file("/data", b"old")
        snapshot = workspace.snapshot()
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("DROP TABLE open_inodes; DROP TABLE mounts; DROP TABLE owners; DROP TABLE orphans;")
        db.execute("UPDATE format SET version=1")
        db.execute("INSERT INTO pins VALUES('legacy',?,'unknown-owner')", [snapshot])
    with Workspace(path) as workspace:
        assert workspace.checkout().read("/data") == b"old"
        assert workspace.recover_owners().pins == 0
        with pytest.raises(BlockingIOError):
            workspace.drop_snapshot(snapshot)


def test_mount_lease_excludes_external_mutations_and_topology_changes(tmp_path):
    path = tmp_path / "workspace.sqlite"
    with Workspace(path) as owner, Workspace(path) as outsider:
        main = owner.checkout()
        main.write_file("/file", b"before")
        child = owner.fork("main", "child")
        owner.acquire_mount(child.id)
        session = owner.checkout(child.id)
        session.write_file("/file", b"mounted")
        assert outsider.checkout(child.id).read("/file") == b"mounted"
        operations = [
            lambda: outsider.checkout(child.id).write_file("/file", b"wrong"),
            lambda: outsider.fork(child.id, "nested"),
            lambda: outsider.snapshot(child.id),
            lambda: owner.snapshot(child.id),
            lambda: outsider.delete_branch(child.id),
            lambda: owner.preview_merge(child.id, "main"),
            lambda: outsider.acquire_mount(child.id),
            lambda: outsider.release_mount(child.id),
        ]
        for operation in operations:
            with pytest.raises(BlockingIOError):
                operation()
        owner.release_mount(child.id)
        outsider.merge(outsider.preview_merge(child.id, "main"))
        assert main.read("/file") == b"mounted"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="Requires fork")
def test_connection_inherited_across_fork_is_rejected(tmp_path):
    with Workspace(tmp_path / "workspace.sqlite") as workspace:
        session = workspace.checkout()
        pid = os.fork()
        if pid == 0:
            try:
                session.stat("/")
            except ValueError as error:
                workspace.close()
                os._exit(0 if "after fork" in str(error) else 2)
            os._exit(3)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert workspace.recover_owners().owners == 0
        assert session.stat("/").is_directory
