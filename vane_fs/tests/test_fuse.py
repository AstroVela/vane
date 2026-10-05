# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import errno
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

from vane_fs import Workspace

BINARY = os.environ.get("VANE_FS_MOUNT_BINARY")
pytestmark = [
    pytest.mark.skipif(not BINARY, reason="Set VANE_FS_MOUNT_BINARY to run real FUSE tests"),
    pytest.mark.timeout(90),
]


def mounted(path):
    escaped = str(path).replace("\\", "\\134").replace(" ", "\\040")
    return any(line.split()[4] == escaped for line in Path("/proc/self/mountinfo").read_text().splitlines())


@contextmanager
def mount_workspace(tmp_path, database, selector, identity):
    point = tmp_path / f"mount-{selector}-{identity}"
    point.mkdir()
    with (tmp_path / f"{point.name}.log").open("w+") as log:
        process = subprocess.Popen(
            [BINARY, str(database), f"--{selector}", identity, str(point), "--debug"], stdout=log, stderr=log
        )
        try:
            deadline = time.monotonic() + 15
            while not mounted(point):
                if process.poll() is not None or time.monotonic() > deadline:
                    log.seek(0)
                    pytest.fail(f"Could not mount VaneFS:\n{log.read()}")
                time.sleep(0.02)
            yield point, process
        finally:
            try:
                if mounted(point):
                    subprocess.run(["fusermount3", "-u", str(point)], check=True, timeout=10, capture_output=True)
            finally:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                assert not mounted(point), "Test mount was not removed"
                log.seek(0)
                output = log.read()
                assert "VaneFS releasing inode:" not in output, output


def test_snapshot_mount_is_pinned_immutable_and_readonly(tmp_path):
    database = tmp_path / "workspace.sqlite"
    with Workspace(database) as workspace:
        live = workspace.checkout()
        live.write_file("/file", b"before")
        snapshot = workspace.snapshot()
        with mount_workspace(tmp_path, database, "snapshot", snapshot) as (point, _):
            assert (point / "file").read_bytes() == b"before"
            live.write_file("/file", b"after")
            assert (point / "file").read_bytes() == b"before"
            assert subprocess.check_output(["cat", str(point / "file")]) == b"before"
            with pytest.raises(OSError) as error:
                (point / "file").write_bytes(b"wrong")
            assert error.value.errno == errno.EROFS
            with pytest.raises(BlockingIOError):
                workspace.drop_snapshot(snapshot)
            assert os.statvfs(point).f_flag & os.ST_RDONLY
        workspace.drop_snapshot(snapshot)


def test_live_mount_namespace_metadata_and_external_exclusion(tmp_path):
    database = tmp_path / "workspace.sqlite"
    with Workspace(database) as workspace:
        child = workspace.fork("main", "candidate")
        with mount_workspace(tmp_path, database, "branch", child.id) as (point, _):
            directory = point / "输入"
            directory.mkdir(mode=0o750)
            (directory / "data").write_bytes(b"x" * 8193)
            shutil.copyfile(directory / "data", directory / "copy")
            assert (directory / "copy").read_bytes() == b"x" * 8193
            os.chmod(directory / "data", 0o640)
            stamp = 1700000000123456789
            os.utime(directory / "data", ns=(stamp, stamp))
            assert (directory / "data").stat().st_mtime_ns == stamp
            assert (directory / "data").stat().st_mode & 0o777 == 0o640
            with pytest.raises(FileExistsError):
                (directory / "data").open("xb")
            with pytest.raises(OSError):
                directory.rename(directory / "cycle")
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                directory.rename(point / "renamed")
                fd = os.open("relative", os.O_CREAT | os.O_WRONLY, 0o600, dir_fd=descriptor)
                os.write(fd, b"relative data")
                os.close(fd)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            assert workspace.checkout(child.id).read("/renamed/relative") == b"relative data"
            with pytest.raises(BlockingIOError):
                workspace.checkout(child.id).write_file("/external", b"wrong")
            with pytest.raises(BlockingIOError):
                workspace.snapshot(child.id)
        assert workspace.checkout().listdir() == []
        workspace.merge(workspace.preview_merge(child.id, "main"))
        assert workspace.checkout().read("/renamed/data") == b"x" * 8193


def test_open_handles_survive_unlink_and_replaced_rename(tmp_path):
    database = tmp_path / "workspace.sqlite"
    with Workspace(database) as workspace:
        with mount_workspace(tmp_path, database, "branch", "main") as (point, _):
            (point / "target").write_bytes(b"old target")
            fd = os.open(point / "target", os.O_RDWR)
            try:
                (point / "source").write_bytes(b"replacement")
                os.replace(point / "source", point / "target")
                assert os.pread(fd, 100, 0) == b"old target"
                assert os.fstat(fd).st_nlink == 0
                assert (point / "target").read_bytes() == b"replacement"
                os.pwrite(fd, b"changed", 0)
                os.ftruncate(fd, 4)
                os.ftruncate(fd, 10)
                os.fsync(fd)
                assert os.pread(fd, 10, 0) == b"chan" + b"\0" * 6
                workspace.collect_garbage()
                assert os.pread(fd, 10, 0) == b"chan" + b"\0" * 6
            finally:
                os.close(fd)
            fd = os.open(point / "target", os.O_RDWR)
            try:
                os.unlink(point / "target")
                (point / "target").write_bytes(b"new inode")
                assert os.fstat(fd).st_nlink == 0
                assert os.pread(fd, 100, 0) == b"replacement"
                assert (point / "target").read_bytes() == b"new inode"
            finally:
                os.close(fd)
            assert sorted(p.name for p in point.iterdir()) == ["target"]
        # No unreachable orphan remains to invalidate snapshot/diff/merge.
        snapshot = workspace.snapshot()
        assert workspace.diff(snapshot, snapshot) == []


def test_live_handles_observe_other_writes_and_atomic_append(tmp_path):
    database = tmp_path / "workspace.sqlite"
    with Workspace(database):
        with mount_workspace(tmp_path, database, "branch", "main") as (point, _):
            path = point / "file"
            path.write_bytes(b"before")
            with path.open("rb", buffering=0) as handle:
                assert handle.read() == b"before"
                path.write_bytes(b"after")
                handle.seek(0)
                assert handle.read() == b"after"
            path.write_bytes(b"")

            def append(index):
                fd = os.open(path, os.O_WRONLY | os.O_APPEND)
                try:
                    for number in range(30):
                        os.write(fd, f"{index}:{number}\n".encode())
                finally:
                    os.close(fd)

            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(append, range(4)))
            assert sorted(path.read_text().splitlines()) == sorted(f"{i}:{j}" for i in range(4) for j in range(30))


@pytest.mark.parametrize("mutation", ["unlink", "rename", "insert"])
def test_directory_pagination_is_stable_during_mutation(tmp_path, mutation):
    database = tmp_path / "workspace.sqlite"
    # Exceed both a FUSE reply and libc's directory buffer, so scandir must
    # resume through multiple readdir requests after the directory changes.
    original = {f"file-{index:04}" for index in range(1800)}
    added = {f"added-{index:04}" for index in range(40)}
    with Workspace(database) as workspace:
        branch = workspace.checkout()
        for name in sorted(original):
            branch.write_file("/" + name, b"")
        with mount_workspace(tmp_path, database, "branch", "main") as (point, _):
            seen = set()
            with os.scandir(point) as entries:
                for entry in entries:
                    assert entry.name in original
                    assert entry.name not in seen, f"Repeated directory entry: {entry.name}"
                    seen.add(entry.name)
                    if mutation == "unlink":
                        os.unlink(entry.path)
                    elif mutation == "rename":
                        os.rename(entry.path, point / ("renamed-" + entry.name))
                    elif len(seen) == 1:
                        for name in sorted(added):
                            (point / name).touch()
            assert seen == original
            if mutation == "unlink":
                expected = set()
            elif mutation == "rename":
                expected = {"renamed-" + name for name in original}
            else:
                expected = original | added
            assert set(os.listdir(point)) == expected
        assert set(branch.listdir()) == expected


def test_directory_lists_belong_to_each_open_handle_and_survive_rewind(tmp_path):
    database = tmp_path / "workspace.sqlite"
    with Workspace(database) as workspace:
        branch = workspace.checkout()
        branch.write_file("/original", b"")
        with mount_workspace(tmp_path, database, "branch", "main") as (point, _):
            first = os.open(point, os.O_RDONLY | os.O_DIRECTORY)
            try:
                assert os.listdir(first) == ["original"]
                (point / "original").unlink()
                (point / "replacement").touch()
                second = os.open(point, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    assert os.listdir(second) == ["replacement"]
                    # listdir(fd) rewinds the directory stream. Its saved
                    # listing must remain valid at offset zero as well.
                    assert os.listdir(first) == ["original"]
                finally:
                    os.close(second)
                assert os.listdir(first) == ["original"]
            finally:
                os.close(first)
            assert os.listdir(point) == ["replacement"]


def test_mount_crash_reclaims_orphans_and_preserves_committed_data(tmp_path):
    database = tmp_path / "workspace.sqlite"
    with Workspace(database) as workspace:
        with mount_workspace(tmp_path, database, "branch", "main") as (point, process):
            fd = os.open(point / "unlinked", os.O_CREAT | os.O_RDWR, 0o600)
            try:
                os.write(fd, b"orphan")
                os.unlink(point / "unlinked")
                (point / "retained").write_bytes(b"committed")
                process.kill()
                process.wait(timeout=10)
            finally:
                try:
                    os.close(fd)
                except OSError as error:
                    assert error.errno in (errno.ENOTCONN, errno.EIO)
        result = workspace.recover_owners()
        assert result.owners == result.mounts == 1
        assert workspace.checkout().read("/retained") == b"committed"
        assert workspace.checkout().listdir() == ["retained"]
        snapshot = workspace.snapshot()
        assert workspace.diff(snapshot, snapshot) == []
        workspace.collect_garbage()


def test_mount_rejects_database_inside_mountpoint(tmp_path):
    point = tmp_path / "mount"
    point.mkdir()
    database = point / "workspace.sqlite"
    result = subprocess.run(
        [BINARY, str(database), "--branch", "main", str(point)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode != 0
    assert "outside" in result.stderr
    assert not database.exists()
