# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import errno
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
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
def mount_workspace(tmp_path, database, selector, identity, *options):
    point = tmp_path / f"mount-{selector}-{identity}"
    point.mkdir()
    with (tmp_path / f"{point.name}.log").open("w+") as log:
        process = subprocess.Popen(
            [BINARY, str(database), f"--{selector}", identity, str(point), "--debug", *options], stdout=log, stderr=log
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


@pytest.mark.parametrize("durability", ["strict", "fsync"])
def test_open_handles_survive_unlink_and_replaced_rename(tmp_path, durability):
    database = tmp_path / "workspace.sqlite"
    with Workspace(database) as workspace:
        with mount_workspace(tmp_path, database, "branch", "main", f"--durability={durability}") as (point, _):
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


@pytest.mark.parametrize("durability", ["strict", "fsync"])
def test_live_handles_observe_other_writes_and_atomic_append(tmp_path, durability):
    database = tmp_path / "workspace.sqlite"
    with Workspace(database):
        with mount_workspace(tmp_path, database, "branch", "main", f"--durability={durability}") as (point, _):
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


def test_repeated_open_release_does_not_commit_existing_inode_references(tmp_path):
    database = tmp_path / "workspace.sqlite"
    with Workspace(database) as workspace:
        branch = workspace.checkout()
        branch.mkdir("/dir")
        branch.write_file("/dir/file", b"content")
        with mount_workspace(tmp_path, database, "branch", "main") as (point, _):
            directory = point / "dir"
            path = directory / "file"
            # Keep file and directory references live independently of kernel
            # dentry eviction while exercising additional opens and releases.
            file_fd = os.open(path, os.O_RDONLY)
            dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                assert path.read_bytes() == b"content"
                assert os.listdir(directory) == ["file"]
                with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as db:
                    before = db.execute("PRAGMA data_version").fetchone()
                    for _ in range(20):
                        assert path.read_bytes() == b"content"
                        assert os.listdir(directory) == ["file"]
                    assert db.execute("PRAGMA data_version").fetchone() == before
            finally:
                os.close(file_fd)
                os.close(dir_fd)


@pytest.mark.parametrize("readonly", [False, True])
def test_warm_stat_reuses_kernel_metadata_without_new_lookups(tmp_path, readonly):
    database = tmp_path / "workspace.sqlite"
    with Workspace(database) as workspace:
        branch = workspace.checkout()
        branch.mkdir("/dir")
        branch.write_file("/dir/file", b"content")
        selector, identity = ("snapshot", workspace.snapshot()) if readonly else ("branch", "main")
        with mount_workspace(tmp_path, database, selector, identity) as (point, _):
            path = point / "dir/file"
            expected = path.stat()
            log = tmp_path / f"{point.name}.log"
            before = log.stat().st_size
            for _ in range(100):
                actual = path.stat()
                assert (actual.st_ino, actual.st_size, actual.st_mtime_ns) == (
                    expected.st_ino,
                    expected.st_size,
                    expected.st_mtime_ns,
                )
            output = log.read_bytes()[before:]
            assert b"opcode: LOOKUP" not in output
            assert b"opcode: GETATTR" not in output
            if readonly:
                branch.write_file("/dir/file", b"a different live file")
                assert path.stat().st_size == len(b"content")


def test_cached_metadata_observes_other_process_mutations_immediately(tmp_path):
    database = tmp_path / "workspace.sqlite"
    with Workspace(database) as workspace:
        branch = workspace.checkout()
        branch.mkdir("/dir")
        branch.write_file("/dir/file", b"before")
        with mount_workspace(tmp_path, database, "branch", "main") as (point, _):
            path = point / "dir/file"
            descriptor = os.open(path, os.O_RDONLY)

            def mutate(program):
                # Warm both path and descriptor attributes before every change.
                path.stat()
                os.fstat(descriptor)
                subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-c",
                        "import os,sys; from pathlib import Path; "
                        "root=Path(sys.argv[1]); path=root/'dir/file'; " + program,
                        str(point),
                    ],
                    check=True,
                    timeout=10,
                )

            def current():
                expected = branch.stat("/dir/file")
                for actual in (path.stat(), os.fstat(descriptor)):
                    assert (actual.st_size, actual.st_mode & 0o777, actual.st_mtime_ns) == (
                        expected.size,
                        expected.mode,
                        expected.mtime_ns,
                    )
                    assert actual.st_atime_ns == actual.st_ctime_ns == actual.st_mtime_ns

            try:
                mutate("path.write_bytes(b'longer contents')")
                current()
                mutate("os.truncate(path, 3)")
                current()
                mutate("os.chmod(path, 0)")
                current()
                if os.geteuid() != 0:
                    with pytest.raises(PermissionError):
                        path.open("rb")
                mutate("os.chmod(path, 0o640); os.utime(path, ns=(1700000000123456789,1700000000123456789))")
                current()
                mutate("path.open('wb').close()")
                current()
                missing = point / "dir/missing"
                assert not missing.exists()
                mutate("(root/'dir/missing').write_bytes(b'created'); os.replace(root/'dir/missing', path)")
                assert path.read_bytes() == b"created"
                assert path.stat().st_ino != os.fstat(descriptor).st_ino
                assert os.fstat(descriptor).st_nlink == 0
                assert not missing.exists()
                parent_before = (point / "dir").stat()
                mutate("path.unlink(); path.write_bytes(b'new inode')")
                assert path.read_bytes() == b"new inode"
                assert (point / "dir").stat().st_mtime_ns == branch.stat("/dir").mtime_ns
                assert (point / "dir").stat().st_mtime_ns != parent_before.st_mtime_ns
                assert not (point / "moved").exists()
                mutate("(root/'dir').rename(root/'moved'); (root/'dir').mkdir()")
                assert not path.exists()
                assert (point / "moved/file").read_bytes() == b"new inode"
                assert os.listdir(point / "dir") == []
                assert os.fstat(descriptor).st_nlink == 0
            finally:
                os.close(descriptor)


@pytest.mark.parametrize("mutation", ["write", "truncate_open"])
def test_cached_atime_alias_tracks_mtime_for_statx_atime_only(tmp_path, mutation):
    database = tmp_path / "workspace.sqlite"
    with Workspace(database) as workspace:
        branch = workspace.checkout()
        branch.write_file("/file", b"before")
        with mount_workspace(tmp_path, database, "branch", "main") as (point, _):
            path = point / "file"
            os.utime(path, ns=(946684800000000000, 946684800000000000))
            assert path.stat().st_atime_ns == 946684800000000000
            descriptor = os.open(path, os.O_WRONLY | (os.O_TRUNC if mutation == "truncate_open" else 0))
            try:
                if mutation == "write":
                    os.write(descriptor, b"new")
                # GNU stat requests only STATX_ATIME for this format. A full
                # stat would refresh all attributes and hide a stale alias.
                atime = int(subprocess.check_output(["stat", "-c", "%X", str(path)], text=True, timeout=10))
                assert atime == branch.stat("/file").mtime_ns // 1_000_000_000
            finally:
                os.close(descriptor)


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


def test_directory_listing_is_captured_on_first_read(tmp_path):
    database = tmp_path / "workspace.sqlite"
    with Workspace(database) as workspace:
        branch = workspace.checkout()
        branch.mkdir("/dir")
        branch.write_file("/dir/removed-before-read", b"")
        with mount_workspace(tmp_path, database, "branch", "main") as (point, _):
            directory = point / "dir"
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                (directory / "removed-before-read").unlink()
                (directory / "created-before-read").touch()
                directory.rename(point / "renamed")
                assert os.listdir(descriptor) == ["created-before-read"]
                (point / "renamed" / "created-after-read").touch()
                assert os.listdir(descriptor) == ["created-before-read"]
                assert sorted(os.listdir(point / "renamed")) == ["created-after-read", "created-before-read"]
            finally:
                os.close(descriptor)


def test_unread_directory_handle_survives_removal_and_gc(tmp_path):
    database = tmp_path / "workspace.sqlite"
    with Workspace(database) as workspace:
        branch = workspace.checkout()
        branch.mkdir("/parent")
        branch.mkdir("/parent/child")
        with mount_workspace(tmp_path, database, "branch", "main") as (point, _):
            descriptor = os.open(point / "parent" / "child", os.O_RDONLY | os.O_DIRECTORY)
            try:
                (point / "parent" / "child").rmdir()
                (point / "parent").rmdir()
                workspace.collect_garbage()
                assert os.listdir(descriptor) == []
                assert os.fstat(descriptor).st_nlink == 0
            finally:
                os.close(descriptor)
        snapshot = workspace.snapshot()
        assert workspace.diff(snapshot, snapshot) == []


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


@pytest.mark.parametrize("size", [4096, 4 * 1024 * 1024, 20 * 1024 * 1024])
def test_acknowledged_write_survives_mount_crash_without_fsync_or_close(tmp_path, size):
    database = tmp_path / "workspace.sqlite"
    payload = bytes(range(256)) * (size // 256)
    with Workspace(database) as workspace:
        with mount_workspace(tmp_path, database, "branch", "main") as (point, process):
            fd = os.open(point / "committed", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                assert os.write(fd, payload) == len(payload)
                # No fsync, flush, close, or subsequent filesystem operation
                # can help persist the write before the mount is killed.
                process.kill()
                process.wait(timeout=10)
            finally:
                try:
                    os.close(fd)
                except OSError as error:
                    assert error.errno in (errno.ENOTCONN, errno.EIO)
        recovered = workspace.recover_owners()
        assert recovered.owners == recovered.mounts == 1
        assert workspace.checkout().read("/committed") == payload
        workspace.collect_garbage()
        with workspace.open_snapshot(workspace.snapshot()) as snapshot:
            assert snapshot.read("/committed") == payload


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


@pytest.mark.parametrize("barrier", ["fsync", "fdatasync", "directory", "sync_write", "dsync_write"])
def test_fsync_mode_barriers_and_recovery(tmp_path, barrier):
    database = tmp_path / "workspace.sqlite"
    # Cross the background checkpoint threshold before exercising each barrier.
    payload = bytes(range(256)) * (1024 * 128)
    with Workspace(database) as workspace:
        reader = workspace.checkout()
        with mount_workspace(tmp_path, database, "branch", "main", "--durability=fsync") as (point, process):
            flags = {"sync_write": os.O_SYNC, "dsync_write": os.O_DSYNC}.get(barrier, 0)
            fd = os.open(point / "file", os.O_CREAT | os.O_RDWR | flags, 0o600)
            try:
                assert os.write(fd, payload) == len(payload)
                # Visibility does not depend on close/fsync, including across
                # an independently opened Workspace connection.
                assert reader.read("/file") == payload
                with (point / "file").open("rb", buffering=0) as other:
                    assert other.read() == payload
                if barrier == "directory":
                    (point / "file").rename(point / "renamed")
                    parent = os.open(point, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(parent)
                    finally:
                        os.close(parent)
                elif barrier in ("fsync", "fdatasync"):
                    # A different handle can synchronize the writer's data.
                    other = os.open(point / "file", os.O_RDONLY)
                    try:
                        getattr(os, barrier)(other)
                    finally:
                        os.close(other)
                process.kill()
                process.wait(timeout=10)
            finally:
                try:
                    os.close(fd)
                except OSError as error:
                    assert error.errno in (errno.ENOTCONN, errno.EIO)
        workspace.recover_owners()
        path = "/renamed" if barrier == "directory" else "/file"
        assert reader.read(path) == payload
        snapshot = workspace.snapshot()
        child = workspace.fork("main", "child")
        with workspace.checkout(child.id) as branch:
            branch.write(path, b"child", offset=0)
        with workspace.open_snapshot(snapshot) as frozen:
            assert frozen.read(path) == payload
        assert reader.read(path) == payload


@pytest.mark.parametrize("directory", [False, True])
def test_fsync_mode_failed_barrier_is_reported_and_retryable(tmp_path, directory):
    database = tmp_path / "workspace.sqlite"
    with Workspace(database) as workspace:
        with mount_workspace(tmp_path, database, "branch", "main", "--durability=fsync") as (point, process):
            (point / "file").write_bytes(b"pending")
            fd = os.open(point if directory else point / "file", os.O_RDONLY)
            try:
                with closing(sqlite3.connect(database)) as inspect:
                    inspect.execute(
                        "CREATE TRIGGER fail_barrier BEFORE UPDATE ON sync_barrier "
                        "BEGIN SELECT RAISE(ABORT, 'injected barrier failure'); END"
                    )
                    inspect.commit()
                    with pytest.raises(OSError) as error:
                        os.fsync(fd)
                    assert error.value.errno == errno.EIO
                    assert workspace.checkout().read("/file") == b"pending"
                    inspect.execute("DROP TRIGGER fail_barrier")
                    inspect.commit()
                os.fsync(fd)
            finally:
                os.close(fd)
        assert process.returncode == 0


def test_fsync_mode_clean_unmount_and_concurrent_truncate_append(tmp_path):
    database = tmp_path / "workspace.sqlite"
    with Workspace(database) as workspace:
        with mount_workspace(tmp_path, database, "branch", "main", "--durability=fsync") as (point, process):
            path = point / "file"
            path.write_bytes(b"initial")

            def mutate(index):
                fd = os.open(path, os.O_RDWR | os.O_APPEND)
                try:
                    for _ in range(30):
                        if index == 0:
                            os.ftruncate(fd, 0)
                        else:
                            assert os.write(fd, b"record\n") == 7
                finally:
                    os.close(fd)

            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(mutate, range(4)))
            data = path.read_bytes()
            assert data == b"record\n" * (len(data) // 7)
            assert workspace.checkout().read("/file") == data
            # No explicit fsync: successful mount process exit is the barrier.
            path.write_bytes(b"clean unmount")
        assert process.returncode == 0
        assert workspace.checkout().read("/file") == b"clean unmount"


def test_failed_final_commit_makes_mount_exit_unsuccessful(tmp_path):
    database = tmp_path / "workspace.sqlite"
    with Workspace(database) as workspace:
        with mount_workspace(tmp_path, database, "branch", "main", "--durability=fsync") as (point, process):
            (point / "file").write_bytes(b"pending")
            with closing(sqlite3.connect(database)) as inspect:
                inspect.execute(
                    "CREATE TRIGGER fail_close BEFORE DELETE ON owners "
                    "BEGIN SELECT RAISE(ABORT, 'injected final commit failure'); END"
                )
                inspect.commit()
        assert process.returncode == 1
        with closing(sqlite3.connect(database)) as inspect:
            inspect.execute("DROP TRIGGER fail_close")
            inspect.commit()
        assert workspace.recover_owners().mounts == 1


@pytest.mark.parametrize("options", [["--durability=invalid"], ["--durability=fsync", "--durability=strict"]])
def test_mount_rejects_invalid_durability_options(tmp_path, options):
    result = subprocess.run(
        [BINARY, str(tmp_path / "workspace.sqlite"), "--branch", "main", str(tmp_path / "mount"), *options],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "durability=strict|fsync" in result.stderr
    assert not (tmp_path / "workspace.sqlite").exists()
