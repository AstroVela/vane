# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import random
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

from vane_fs import CapacityError, ConflictError, Error, StalePreviewError, Workspace


@pytest.fixture
def database_path(tmp_path):
    return tmp_path / "workspace.sqlite"


@pytest.fixture
def workspace(database_path):
    with Workspace(database_path) as workspace:
        yield workspace


def physical_counts(path):
    # Inspection only: all VaneFS mutations use the bundled native SQLite.
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        return tuple(
            db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("inode_versions", "dirent_versions", "block_versions", "block_payloads")
        )


def test_native_package_and_sqlite_identity():
    import vane_fs._native as native

    assert Path(native.__file__).suffix in {".so", ".pyd"}
    assert tuple(map(int, Workspace.sqlite_version().split("."))) >= (3, 51, 3)


def test_reopen_persists_files_snapshots_and_branch_identity(database_path):
    with Workspace(database_path) as workspace:
        original_id = workspace.id
        main = workspace.checkout()
        main.mkdir("/输入")
        main.write_file("/输入/hello.txt", "你好\x00world".encode())
        child = workspace.fork("main", "candidate")
        snapshot = workspace.snapshot()
    with Workspace(database_path) as workspace:
        assert workspace.id == original_id
        assert workspace.branch("candidate").id == child.id
        with workspace.open_snapshot(snapshot) as frozen:
            assert frozen.read("/输入/hello.txt") == "你好\x00world".encode()
        assert workspace.checkout().listdir("/") == ["输入"]


def test_fork_and_snapshot_copy_no_file_rows(workspace, database_path):
    main = workspace.checkout()
    main.write_file("/file", b"a" * 16384)
    before = physical_counts(database_path)
    children = [workspace.fork("main", f"child-{i}") for i in range(20)]
    snapshot = workspace.snapshot()
    assert physical_counts(database_path) == before
    workspace.checkout(children[3].id).write("/file", b"xy", offset=4095)
    assert physical_counts(database_path)[3] == before[3] + 2
    assert main.read("/file") == b"a" * 16384
    with workspace.open_snapshot(snapshot) as frozen:
        assert frozen.read("/file") == b"a" * 16384
    for child in children:
        data = workspace.checkout(child.id).read("/file")
        assert data == (b"a" * 4095 + b"xy" + b"a" * 12287 if child == children[3] else b"a" * 16384)


def test_replacement_reuses_unchanged_blocks_and_clears_tail(workspace, database_path):
    main = workspace.checkout()
    original = b"a" * 8192 + b"tail"
    main.write_file("/file", original)
    child = workspace.fork("main", "child")
    session = workspace.checkout(child.id)
    before = physical_counts(database_path)[3]
    session.write_file("/file", original)
    assert physical_counts(database_path)[3] == before
    session.write_file("/file", b"a" * 4096 + b"b" * 4096 + b"tail")
    assert physical_counts(database_path)[3] == before + 1
    session.write_file("/file", b"a" * 4096 + b"b")
    session.truncate("/file", len(original))
    assert session.read("/file") == b"a" * 4096 + b"b" + b"\0" * (len(original) - 4097)
    assert main.read("/file") == original


def test_range_reads_preserve_sparse_blocks_across_versions(workspace):
    block = 4096
    original = b"".join(bytes([index if index % 3 else 0]) * block for index in range(32)) + b"tail"
    main = workspace.checkout()
    main.write_file("/file", original)
    base = workspace.snapshot()
    child = workspace.checkout(workspace.fork("main", "child").id)
    expected = bytearray(original)
    for offset, data in ((block - 3, b"changed" * 1171), (7 * block, b"\0" * (2 * block))):
        child.write("/file", data, offset=offset)
        expected[offset : offset + len(data)] = data
    changed = workspace.snapshot(child.id)
    main.write("/file", b"parent" * 683, offset=20 * block)
    parent_expected = bytearray(original)
    parent_expected[20 * block : 20 * block + 4098] = b"parent" * 683
    child.truncate("/file", 25 * block + 3)
    child.truncate("/file", len(original) + block)
    truncated = bytes(expected[: 25 * block + 3]) + b"\0" * (len(original) + block - (25 * block + 3))
    with workspace.open_snapshot(base) as frozen, workspace.open_snapshot(changed) as child_frozen:
        for session, data in (
            (main, parent_expected),
            (child, truncated),
            (frozen, original),
            (child_frozen, expected),
        ):
            assert session.read("/file") == data
            for offset, size in (
                (0, 1),
                (4093, 8197),
                (6 * block + 7, 4 * block),
                (len(data) - 5, 100),
                (len(data), 1),
            ):
                assert session.read("/file", offset, size) == data[offset : offset + size]
            assert session.read("/file", 4095, 0) == b""


def test_sparse_range_read_at_maximum_file_size(workspace):
    main = workspace.checkout()
    main.write_file("/file", b"")
    maximum = 2**63 - 1
    main.write("/file", b"end", offset=maximum - 3)
    assert main.stat("/file").size == maximum
    assert main.read("/file", maximum - 7, 100) == b"\0" * 4 + b"end"
    assert main.read("/file", maximum, 100) == b""
    main.truncate("/file", maximum - 2)
    main.truncate("/file", maximum)
    assert main.read("/file", maximum - 3, 3) == b"e\0\0"


@pytest.mark.parametrize("fault", ["missing_payload", "overlapping_versions"])
def test_range_read_reports_corruption_and_releases_query(workspace, database_path, fault):
    main = workspace.checkout()
    data = b"a" * 4096 + b"b" * 4096
    main.write_file("/file", data)
    workspace.snapshot()
    assert main.read("/file") == data
    with closing(sqlite3.connect(database_path)) as db:
        if fault == "missing_payload":
            db.execute("DELETE FROM block_payloads WHERE id=(SELECT payload FROM block_versions WHERE block=1)")
            message = "Missing block payload"
        else:
            db.execute(
                "INSERT INTO block_versions(inode,block,low,high,writer,deleted,payload) "
                "SELECT inode,block,(SELECT frontier FROM branches WHERE name='main'),high,writer,deleted,payload "
                "FROM block_versions WHERE block=1"
            )
            message = "Overlapping visible versions"
        db.commit()
    for _ in range(2):
        with pytest.raises(Error, match=message):
            main.read("/file")
        assert main.read("/file", 3, 100) == b"a" * 100


def test_old_session_refreshes_range_after_fork(workspace, database_path):
    main = workspace.checkout()
    main.write_file("/a", b"base")
    with Workspace(database_path) as other:
        child = other.fork("main", "other")
        main.write_file("/a", b"parent")
        assert other.checkout(child.id).read("/a") == b"base"
        other.checkout(child.id).write_file("/a", b"child")
        assert main.read("/a") == b"parent"


def test_snapshot_is_immutable_and_retained_after_branch_delete(workspace):
    branch = workspace.fork("main", "branch")
    live = workspace.checkout(branch.id)
    live.write_file("/a", b"before")
    snapshot = workspace.snapshot(branch.id)
    with workspace.open_snapshot(snapshot) as frozen:
        live.write_file("/a", b"after")
        workspace.delete_branch(branch.id)
        workspace.collect_garbage()
        assert frozen.read("/a") == b"before"
        with pytest.raises(PermissionError):
            frozen.write_file("/a", b"mutation")
        with pytest.raises(BlockingIOError, match="pinned"):
            workspace.drop_snapshot(snapshot)
        with pytest.raises(FileNotFoundError):
            live.read("/a")
    workspace.drop_snapshot(snapshot)
    result = workspace.collect_garbage()
    assert result.versions > 0
    assert result.payloads > 0
    with pytest.raises(FileNotFoundError):
        workspace.open_snapshot(snapshot)


def test_reusing_branch_name_does_not_retarget_sessions(workspace):
    old = workspace.fork("main", "same")
    session = workspace.checkout(old.id)
    workspace.delete_branch(old.id)
    new = workspace.fork("main", "same")
    assert new.id != old.id
    workspace.checkout(new.id).write_file("/new", b"new")
    with pytest.raises(FileNotFoundError):
        session.read("/new")
    with pytest.raises(FileExistsError):
        workspace.fork("main", old.id)


@pytest.mark.parametrize("size", [0, 1, 4095, 4096, 4097, 8192])
def test_truncate_clears_tail_and_sparse_extension(workspace, size):
    main = workspace.checkout()
    main.write_file("/a", b"x" * 16384)
    base = workspace.snapshot()
    main.truncate("/a", size)
    main.truncate("/a", 20000)
    assert main.read("/a") == b"x" * size + b"\0" * (20000 - size)
    main.write("/a", b"end", offset=25000)
    assert main.read("/a", offset=20000) == b"\0" * 5000 + b"end"
    with workspace.open_snapshot(base) as frozen:
        assert frozen.read("/a") == b"x" * 16384


def test_reads_empty_writes_and_range_validation(workspace):
    main = workspace.checkout()
    main.write_file("/a", b"012345")
    assert main.read("/a", 2, 3) == b"234"
    assert main.read("/a", 100) == b""
    assert main.read("/a", 0, 0) == b""
    main.write("/a", b"", offset=100)
    assert main.stat("/a").size == 6
    for operation in (
        lambda: main.read("/a", -1),
        lambda: main.read("/a", 0, -2),
        lambda: main.write("/a", b"x", -1),
        lambda: main.write("/a", b"xx", 2**63 - 1),
        lambda: main.truncate("/a", -1),
    ):
        with pytest.raises(ValueError):
            operation()
    assert main.read("/a") == b"012345"


def test_directory_rename_preserves_identity_and_snapshots(workspace):
    main = workspace.checkout()
    main.mkdir("/a")
    main.mkdir("/a/b")
    main.mkdir("/destination")
    main.write_file("/a/b/file", b"content")
    inode = main.stat("/a/b/file").inode
    before = workspace.snapshot()
    main.rename("/a", "/destination/new")
    assert main.stat("/destination/new/b/file").inode == inode
    assert main.read("/destination/new/b/file") == b"content"
    with workspace.open_snapshot(before) as frozen:
        assert frozen.read("/a/b/file") == b"content"
    with pytest.raises(ValueError, match="itself"):
        main.rename("/destination", "/destination/new/b/loop")
    with pytest.raises(OSError, match="not empty"):
        main.rmdir("/destination")
    main.unlink("/destination/new/b/file")
    main.rmdir("/destination/new/b")
    assert main.listdir("/destination/new") == []


def test_rename_replacement_and_type_checks(workspace):
    main = workspace.checkout()
    main.write_file("/a", b"a")
    main.write_file("/b", b"b")
    main.mkdir("/d")
    before = workspace.snapshot()
    for source, target in (("/a", "/d"), ("/d", "/a")):
        with pytest.raises((IsADirectoryError, NotADirectoryError)):
            main.rename(source, target)
    main.rename("/a", "/b")
    main.rename("/b", "/b")
    assert main.read("/b") == b"a"
    with pytest.raises(FileNotFoundError):
        main.read("/a")
    with workspace.open_snapshot(before) as frozen:
        assert frozen.read("/b") == b"b"
    main.mkdir("/empty")
    main.rename("/d", "/empty")
    assert main.listdir("/") == ["b", "empty"]


@pytest.mark.parametrize("path", ["../escape", "/a/../../escape", "/bad\0name", "/" + "x" * 256])
def test_invalid_paths_do_not_modify_namespace(workspace, path):
    main = workspace.checkout()
    with pytest.raises(ValueError):
        main.write_file(path, b"bad")
    assert main.listdir() == []


def test_file_and_directory_errors(workspace):
    main = workspace.checkout()
    main.mkdir("/directory")
    main.write_file("/file", b"a")
    for operation in (
        lambda: main.read("/directory"),
        lambda: main.write_file("/directory", b"x"),
        lambda: main.unlink("/directory"),
    ):
        with pytest.raises(IsADirectoryError):
            operation()
    for path in ("/file/child", "/file/"):
        with pytest.raises(NotADirectoryError):
            main.stat(path)
    with pytest.raises(FileExistsError):
        main.mkdir("/directory")
    with pytest.raises(FileNotFoundError):
        main.write_file("/missing/a", b"x")


def test_diff_and_nonconflicting_merge_keep_target_changes(workspace):
    main = workspace.checkout()
    main.write_file("/shared", b"base")
    base = workspace.snapshot()
    child = workspace.fork("main", "child")
    source = workspace.checkout(child.id)
    source.write_file("/shared", b"source")
    source.mkdir("/new")
    source.write_file("/new/file", b"nested")
    main.write_file("/target-only", b"target")
    after = workspace.snapshot(child.id)
    assert [(entry.path, entry.kind) for entry in workspace.diff(after, base)] == [
        ("/new", "added"),
        ("/new/file", "added"),
        ("/shared", "modified"),
    ]
    preview = workspace.preview_merge(child.id, "main")
    assert preview.conflicts == []
    assert "/target-only" not in [change.path for change in preview.changes]
    workspace.merge(preview)
    assert main.read("/shared") == b"source"
    assert main.read("/new/file") == b"nested"
    assert main.read("/target-only") == b"target"
    assert workspace.branch(child.id).state == "sealed"
    with pytest.raises(PermissionError):
        source.write_file("/new-write", b"no")
    workspace.delete_branch(child.id)
    workspace.collect_garbage()
    assert main.read("/new/file") == b"nested"


def test_merge_conflict_is_atomic_and_explicitly_resolved(workspace):
    main = workspace.checkout()
    main.write_file("/conflict", b"base")
    child = workspace.fork("main", "child")
    source = workspace.checkout(child.id)
    source.write_file("/conflict", b"source")
    source.write_file("/also-new", b"source-only")
    main.write_file("/conflict", b"target")
    preview = workspace.preview_merge(child.id, "main")
    assert preview.conflicts == ["/conflict"]
    generation = workspace.branch().generation
    with pytest.raises(ConflictError):
        workspace.merge(preview)
    assert main.listdir() == ["conflict"]
    assert workspace.branch().generation == generation
    assert workspace.branch(child.id).state == "writable"
    workspace.merge(preview, {"/conflict": "target"})
    assert main.read("/conflict") == b"target"
    assert main.read("/also-new") == b"source-only"


def test_equal_content_writes_do_not_conflict(workspace):
    main = workspace.checkout()
    main.write_file("/a", b"base")
    child = workspace.fork("main", "child")
    workspace.checkout(child.id).write_file("/a", b"same")
    main.write_file("/a", b"same")
    preview = workspace.preview_merge(child.id, "main")
    assert not preview.conflicts
    workspace.merge(preview)
    assert main.read("/a") == b"same"


@pytest.mark.parametrize("mutate", ["source", "target", "fork", "snapshot"])
def test_merge_preview_revalidates_mutations_and_interval_changes(workspace, mutate):
    child = workspace.fork("main", "child")
    workspace.checkout(child.id).write_file("/a", b"source")
    preview = workspace.preview_merge(child.id, "main")
    if mutate == "source":
        workspace.checkout(child.id).write_file("/a", b"changed")
    elif mutate == "target":
        workspace.checkout().write_file("/target", b"changed")
    elif mutate == "fork":
        workspace.fork("main", "sibling")
    else:
        workspace.snapshot()
    with pytest.raises(StalePreviewError):
        workspace.merge(preview)
    with pytest.raises(FileNotFoundError):
        workspace.checkout().read("/a")


def test_merge_namespace_conflict_parent_delete_vs_new_child(workspace):
    main = workspace.checkout()
    main.mkdir("/directory")
    child = workspace.fork("main", "child")
    source = workspace.checkout(child.id)
    source.rmdir("/directory")
    main.write_file("/directory/new", b"new")
    preview = workspace.preview_merge(child.id, "main")
    assert preview.conflicts == ["/directory"]
    with pytest.raises(ConflictError):
        workspace.merge(preview)
    workspace.merge(preview, {"/directory": "target"})
    assert main.read("/directory/new") == b"new"


def test_merge_rename_vs_edit_cannot_create_duplicate_inode(workspace):
    main = workspace.checkout()
    main.write_file("/old", b"base")
    child = workspace.fork("main", "child")
    workspace.checkout(child.id).rename("/old", "/new")
    main.write_file("/old", b"target")
    preview = workspace.preview_merge(child.id, "main")
    assert set(preview.conflicts) == {"/old", "/new"}
    with pytest.raises(ConflictError):
        workspace.merge(preview, {"/old": "target"})
    assert main.listdir() == ["old"]
    workspace.merge(preview, {"/old": "source", "/new": "source"})
    assert main.listdir() == ["new"]
    assert main.read("/new") == b"base"


@pytest.mark.parametrize("replace_on", ["source", "target"])
@pytest.mark.parametrize("rename", [False, True])
@pytest.mark.parametrize("keep_original", [False, True])
def test_merge_replaced_parent_requires_consistent_inode_resolution(workspace, replace_on, rename, keep_original):
    main = workspace.checkout()
    main.mkdir("/directory")
    original = main.stat("/directory").inode
    child = workspace.fork("main", "child")
    source = workspace.checkout(child.id)
    replacing, adding = (source, main) if replace_on == "source" else (main, source)
    adding.write_file("/directory/new", b"original directory")
    if rename:
        replacing.rename("/directory", "/moved")
    else:
        replacing.rmdir("/directory")
    replacing.mkdir("/directory")
    replacement = replacing.stat("/directory").inode
    assert replacement != original
    preview = workspace.preview_merge(child.id, "main")
    assert preview.conflicts == ["/directory"]
    generation = workspace.branch().generation
    with pytest.raises(ConflictError):
        workspace.merge(preview)
    # Selecting the replacement directory still cannot retarget its child.
    with pytest.raises(ConflictError):
        workspace.merge(preview, {"/directory": replace_on})
    assert workspace.branch().generation == generation
    assert workspace.branch(child.id).state == "writable"
    if keep_original:
        original_side = "target" if replace_on == "source" else "source"
        resolutions = {"/directory": original_side}
        if rename:
            resolutions["/moved"] = original_side
    else:
        resolutions = {"/directory/new": replace_on}
    workspace.merge(preview, resolutions)
    workspace.collect_garbage()
    assert main.stat("/directory").inode == (original if keep_original else replacement)
    if keep_original:
        assert main.read("/directory/new") == b"original directory"
    else:
        assert main.listdir("/directory") == []
        if rename:
            assert main.stat("/moved").inode == original


def test_merge_independently_created_directories_cannot_mix_children(workspace):
    child = workspace.fork("main", "child")
    source, target = workspace.checkout(child.id), workspace.checkout()
    for session, name in [(source, "source"), (target, "target")]:
        session.mkdir("/directory")
        session.write_file("/directory/" + name, name.encode())
    preview = workspace.preview_merge(child.id, "main")
    with pytest.raises(ConflictError):
        workspace.merge(preview, {"/directory": "source"})
    workspace.merge(preview, {"/directory": "source", "/directory/target": "source"})
    assert target.listdir("/directory") == ["source"]
    assert target.stat("/directory").inode == source.stat("/directory").inode


def test_merge_rejects_unrelated_nonleaf_and_foreign_previews(workspace, tmp_path):
    child = workspace.fork("main", "child")
    sibling = workspace.fork("main", "sibling")
    with pytest.raises(ValueError, match="direct child"):
        workspace.preview_merge(child.id, sibling.id)
    descendant = workspace.fork(child.id, "descendant")
    with pytest.raises(ValueError, match="leaf"):
        workspace.preview_merge(child.id, "main")
    workspace.delete_branch(descendant.id)
    preview = workspace.preview_merge(child.id, "main")
    with Workspace(tmp_path / "other.sqlite") as other:
        with pytest.raises(ValueError, match="another workspace"):
            other.merge(preview)


def test_capacity_failure_rolls_back_snapshot_and_fork(workspace, database_path):
    terminal = workspace.fork("main", "terminal", terminal=True)
    source = workspace.checkout(terminal.id)
    source.write_file("/a", b"before")
    generation = workspace.branch(terminal.id).generation
    before = physical_counts(database_path)
    with pytest.raises(CapacityError):
        workspace.fork(terminal.id, "too-deep")
    with pytest.raises(CapacityError):
        workspace.snapshot(terminal.id)
    assert workspace.branch(terminal.id).generation == generation
    assert physical_counts(database_path) == before
    assert source.read("/a") == b"before"
    source.write_file("/a", b"after")
    workspace.merge(workspace.preview_merge(terminal.id, "main"))
    frozen = workspace.snapshot(terminal.id)
    with workspace.open_snapshot(frozen) as session:
        assert session.read("/a") == b"after"


def test_recursive_deletion_and_gc_preserve_parent_and_snapshots(workspace):
    main = workspace.checkout()
    main.write_file("/a", b"parent")
    child = workspace.fork("main", "child")
    workspace.checkout(child.id).write_file("/a", b"child")
    grandchild = workspace.fork(child.id, "grandchild")
    snapshot = workspace.snapshot(grandchild.id)
    with pytest.raises(OSError, match="descendants"):
        workspace.delete_branch(child.id)
    with pytest.raises(ValueError, match="root"):
        workspace.delete_branch("main", recursive=True)
    with pytest.raises(BlockingIOError, match="fork base"):
        workspace.drop_snapshot(child.fork_base)
    workspace.delete_branch(child.id, recursive=True)
    workspace.collect_garbage()
    assert main.read("/a") == b"parent"
    with workspace.open_snapshot(snapshot) as frozen:
        assert frozen.read("/a") == b"child"


def test_pins_are_shared_across_connections_and_released_on_close(workspace, database_path):
    snapshot = workspace.snapshot()
    with Workspace(database_path) as other:
        pin = other.open_snapshot(snapshot)
        with pytest.raises(BlockingIOError):
            workspace.drop_snapshot(snapshot)
        other.close()
        with pytest.raises(ValueError, match="closed"):
            pin.stat("/")
        workspace.drop_snapshot(snapshot)


@pytest.mark.parametrize("object_type", ["table", "view"])
def test_refuses_foreign_sqlite_database(tmp_path, object_type):
    path = tmp_path / "foreign.sqlite"
    with closing(sqlite3.connect(path)) as db, db:
        if object_type == "table":
            db.execute("CREATE TABLE important(value)")
            db.execute("INSERT INTO important VALUES('keep')")
        else:
            db.execute("CREATE VIEW important AS SELECT 'keep' AS value")
    with pytest.raises(ValueError, match="non-VaneFS"):
        Workspace(path)
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT * FROM important").fetchall() == [("keep",)]
        assert db.execute("PRAGMA journal_mode").fetchone() == ("delete",)


def test_concurrent_sibling_writes_split_inherited_records(workspace, database_path):
    workspace.checkout().write_file("/file", b"a" * 10000)
    branches = [workspace.fork("main", f"branch-{i}").id for i in range(4)]
    barrier = threading.Barrier(4)

    def write(index):
        with Workspace(database_path) as connection:
            session = connection.checkout(branches[index])
            barrier.wait(timeout=20)
            for iteration in range(10):
                session.write("/file", bytes([index + 1]) * 4096, offset=2000)
            return session.read("/file")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(write, range(4)))
    assert results == [b"a" * 2000 + bytes([i + 1]) * 4096 + b"a" * 3904 for i in range(4)]
    assert workspace.checkout().read("/file") == b"a" * 10000


def test_subprocess_writes_use_native_transactions(workspace, database_path):
    workspace.checkout().write_file("/parallel", b"\0" * 8192)
    program = """
import sys
from vane_fs import Workspace
with Workspace(sys.argv[1]) as workspace:
    session = workspace.checkout()
    for _ in range(30):
        session.write('/parallel', sys.argv[2].encode() * 4096, offset=int(sys.argv[3]))
"""
    processes = [
        subprocess.Popen([sys.executable, "-I", "-c", program, str(database_path), character, str(offset)])
        for character, offset in (("a", 0), ("b", 4096))
    ]
    try:
        assert [process.wait(timeout=30) for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
    assert workspace.checkout().read("/parallel") == b"a" * 4096 + b"b" * 4096


def test_process_crash_never_publishes_partial_write(workspace, database_path):
    old = b"old"
    replacement = b"z" * (8 * 1024 * 1024)
    workspace.checkout().write_file("/crash", old)
    program = """
import os, sys
from vane_fs import Workspace
workspace = Workspace(sys.argv[1])
session = workspace.checkout()
print('ready', flush=True)
session.write_file('/crash', b'z' * (8 * 1024 * 1024))
os._exit(0)
"""
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", program, str(database_path)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        process.kill()
        process.wait(timeout=20)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
    with Workspace(database_path) as reopened:
        result = reopened.checkout().read("/crash")
        assert hashlib.sha256(result).digest() in {hashlib.sha256(old).digest(), hashlib.sha256(replacement).digest()}


def test_randomized_branches_against_copying_reference_model(workspace):
    rng = random.Random(34193)
    models = {workspace.branch().id: {}}
    snapshots = []
    for step in range(250):
        branch = rng.choice(list(models))
        model = models[branch]
        session = workspace.checkout(branch)
        operation = rng.choice(["fork", "snapshot", "replace", "write", "truncate", "unlink"])
        path = f"/file-{rng.randrange(6)}"
        if operation == "fork" and len(models) < 20:
            child = workspace.fork(branch, f"branch-{step}")
            models[child.id] = dict(model)
        elif operation == "snapshot":
            snapshots.append((workspace.snapshot(branch), dict(model)))
        elif operation == "replace":
            value = rng.randbytes(rng.randrange(9000))
            session.write_file(path, value)
            model[path] = value
        elif path in model:
            if operation == "write":
                offset = rng.randrange(12000)
                value = rng.randbytes(rng.randrange(1, 6000))
                old = model[path].ljust(offset + len(value), b"\0")
                model[path] = old[:offset] + value + old[offset + len(value) :]
                session.write(path, value, offset)
            elif operation == "truncate":
                length = rng.randrange(12000)
                model[path] = model[path][:length].ljust(length, b"\0")
                session.truncate(path, length)
            elif operation == "unlink":
                session.unlink(path)
                del model[path]
        assert session.listdir() == sorted(path[1:] for path in model)
        for path, expected in model.items():
            assert session.read(path) == expected
    workspace.collect_garbage()
    for snapshot, model in snapshots:
        with workspace.open_snapshot(snapshot) as session:
            assert session.listdir() == sorted(path[1:] for path in model)
            for path, expected in model.items():
                assert session.read(path) == expected


def test_closed_workspace_and_session_reject_access(workspace):
    session = workspace.checkout()
    session.close()
    session.close()
    with pytest.raises(ValueError, match="closed"):
        session.listdir()
    workspace.close()
    workspace.close()
    with pytest.raises(ValueError, match="closed"):
        workspace.checkout()
