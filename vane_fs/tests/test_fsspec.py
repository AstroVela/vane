# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import io

import pytest

pytest.importorskip("fsspec")

from vane_fs.fsspec import SnapshotFileSystem

from vane_fs import Workspace


@pytest.fixture
def snapshot(tmp_path):
    with Workspace(tmp_path / "workspace.sqlite") as workspace:
        main = workspace.checkout()
        main.mkdir("/data")
        main.write_file("/data/a.txt", b"hello world")
        main.write_file("/data/b.txt", b"second")
        snapshot = workspace.snapshot()
        with SnapshotFileSystem(workspace, snapshot) as fs:
            yield workspace, snapshot, fs


def test_listing_globbing_and_reading(snapshot):
    workspace, snapshot_id, fs = snapshot
    url = fs.url("/data/a.txt")
    assert fs.isfile(url)
    assert fs.isdir(fs.url("/data"))
    assert not fs.exists(fs.url("/missing"))
    assert fs.size(url) == 11
    assert fs.modified(url).tzinfo is not None
    assert sorted(fs.glob(fs.url("/data/*.txt"))) == [
        fs._strip_protocol(fs.url(f"/data/{name}.txt")) for name in ("a", "b")
    ]
    assert [value["type"] for value in fs.ls(fs.url("/data"))] == ["file", "file"]
    with fs.open(url, "rb") as handle:
        assert handle.read(5) == b"hello"
        assert handle.seek(-5, io.SEEK_END) == 6
        assert handle.read() == b"world"
        assert handle.seek(100) == 100
        assert handle.read() == b""
        assert handle.seek(0) == 0
        buffer = bytearray(5)
        assert handle.readinto(buffer) == 5
        assert buffer == b"hello"
    with fs.open(url, "rt", encoding="utf-8") as handle:
        assert handle.read() == "hello world"
    with pytest.raises(PermissionError):
        fs.open(url, "wb")
    with pytest.raises(IsADirectoryError):
        fs.open(fs.url("/data"), "rb")
    with pytest.raises(ValueError, match="outside"):
        fs.open("vanefs://another/snapshots/no/data/a.txt", "rb")


def test_file_pins_outlive_adapter_and_original_branch(snapshot):
    workspace, snapshot_id, fs = snapshot
    handle = fs.open(fs.url("/data/a.txt"), "rb")
    try:
        workspace.checkout().write_file("/data/a.txt", b"modified")
        fs.close()
        with pytest.raises(BlockingIOError):
            workspace.drop_snapshot(snapshot_id)
        workspace.collect_garbage()
        assert handle.read() == b"hello world"
    finally:
        handle.close()
    workspace.drop_snapshot(snapshot_id)
    with pytest.raises(ValueError, match="closed"):
        fs.info(fs.url("/data/a.txt"))


def test_failed_open_does_not_leak_pins(snapshot):
    workspace, snapshot_id, fs = snapshot
    with pytest.raises(FileNotFoundError):
        fs.open(fs.url("/missing"), "rb")
    with pytest.raises(ValueError):
        fs.open(fs.url("/data/a.txt"), block_size=0)
    fs.close()
    workspace.drop_snapshot(snapshot_id)


def test_vane_reads_csv_and_parquet_from_a_stable_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    vane = pytest.importorskip("vane")
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    with Workspace(tmp_path / "workspace.sqlite") as workspace:
        main = workspace.checkout()
        main.write_file("/table.csv", b"id,name\n1,first\n2,second\n")
        sink = arrow.BufferOutputStream()
        parquet.write_table(arrow.table({"id": [1, 2], "name": ["first", "second"]}), sink)
        main.write_file("/table.parquet", sink.getvalue().to_pybytes())
        frozen = workspace.snapshot()
        with SnapshotFileSystem(workspace, frozen) as fs, vane.connect() as connection:
            connection.register_filesystem(fs)
            main.write_file("/table.csv", b"id,name\n3,later\n")
            assert connection.execute("SELECT * FROM read_csv(?) ORDER BY id", [fs.url("/table.csv")]).fetchall() == [
                (1, "first"),
                (2, "second"),
            ]
            assert connection.read_parquet(fs.url("/table.parquet")).order("id").fetchall() == [
                (1, "first"),
                (2, "second"),
            ]
            # FILE/media use register_workspace(), not this Python filesystem.
            with pytest.raises(vane.NotImplementedException, match="Nonblocking"):
                connection.execute("SELECT to_file(?)", [fs.url("/table.csv")])
            connection.unregister_filesystem("vanefs")
