# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Acceptance tests for the optional native reader in an installed Vane build."""

import ctypes
import io
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from vane_fs import Workspace, register_workspace, snapshot_url, unregister_workspace

vane = pytest.importorskip("vane")


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    database = tmp_path / "workspace.sqlite"
    with Workspace(database) as workspace, vane.connect() as connection:
        main = workspace.checkout()
        main.mkdir("/data")
        main.write_file("/data/file", b"0123456789abcdef")
        frozen = workspace.snapshot()
        identity = register_workspace(connection, database)
        assert identity == workspace.id
        yield workspace, connection, database, frozen, snapshot_url(identity, frozen, "/data/file")


def test_native_file_metadata_ranges_and_snapshot_immutability(native):
    workspace, connection, _, frozen, url = native
    workspace.checkout().write_file("/data/file", b"modified")
    workspace.collect_garbage()
    value = connection.execute("SELECT to_file(?)", [url]).fetchall()[0][0]
    assert value.url == url
    assert value.size == 16
    with vane.File(url, position=3, size=8).open(connection=connection, buffer_size=3) as reader:
        assert reader.size() == 8
        assert reader.read(4) == b"3456"
        assert reader.seek(-2, io.SEEK_END) == 6
        assert reader.read() == b"9a"
        assert reader.seek(100) == 100
        assert reader.read() == b""
        with pytest.raises(BlockingIOError, match="pinned"):
            workspace.drop_snapshot(frozen)
    workspace.drop_snapshot(frozen)


def test_streaming_result_keeps_pin_until_exhausted(native):
    workspace, connection, _, frozen, url = native
    connection.execute("SELECT to_file(?) FROM range(4096)", [url]).fetchone()
    with pytest.raises(BlockingIOError, match="pinned"):
        workspace.drop_snapshot(frozen)
    workspace.checkout().write_file("/data/file", b"later")
    workspace.collect_garbage()
    remaining = connection.fetchall()
    assert len(remaining) == 4095
    assert all(value.size == 16 for (value,) in remaining)
    workspace.drop_snapshot(frozen)


def test_native_csv_parquet_and_chunked_reads(native, tmp_path):
    workspace, connection, _, _, _ = native
    parquet_path = tmp_path / "table.parquet"
    connection.execute("COPY (SELECT 42 AS id, 'hello' AS name) TO ? (FORMAT PARQUET)", [str(parquet_path)])
    main = workspace.checkout()
    main.write_file("/table.parquet", parquet_path.read_bytes())
    main.write_file("/table.csv", b"id,name\n42,hello\n")
    payload = bytes(range(251)) * 10000
    main.write_file("/large", payload)
    frozen = workspace.snapshot()
    for name in ("csv", "parquet"):
        url = snapshot_url(workspace.id, frozen, f"/table.{name}")
        assert connection.execute(f"SELECT * FROM read_{name}(?)", [url]).fetchall() == [(42, "hello")]
    url = snapshot_url(workspace.id, frozen, "/large")
    with vane.File(url).open(connection=connection) as reader:
        assert reader.read() == payload
        reader.seek(4095)
        assert reader.read(8193) == payload[4095:12288]
    workspace.drop_snapshot(frozen)


def test_connection_mapping_and_open_handle_lifetimes(native):
    workspace, connection, database, frozen, url = native
    with connection.cursor() as other:
        with pytest.raises(vane.IOException, match="not registered"):
            vane.File(url).open(connection=other)
        register_workspace(other, database)
        reader = vane.File(url).open(connection=connection)
        unregister_workspace(connection, workspace.id)
        with pytest.raises(vane.IOException, match="not registered"):
            vane.File(url).open(connection=connection)
        with vane.File(url).open(connection=other) as other_reader:
            assert other_reader.read() == b"0123456789abcdef"
        # Removing the DB-wide router must not dangle a FileHandle's filesystem.
        connection.unregister_filesystem("vanefs_native")
        workspace.close()
        try:
            assert reader.read() == b"0123456789abcdef"
        finally:
            reader.close()
        register_workspace(other, database)
        with vane.File(url).open(connection=other) as restored:
            assert restored.read(2) == b"01"
    with Workspace(database) as reopened:
        reopened.drop_snapshot(frozen)


def test_independent_native_readers_on_shared_database(native):
    workspace, connection, database, frozen, url = native

    def read_slice(index):
        with connection.cursor() as cursor:
            register_workspace(cursor, database)
            with vane.File(url).open(connection=cursor, buffer_size=1) as reader:
                reader.seek(index)
                return reader.read(4)

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(read_slice, range(8))) == [b"0123456789abcdef"[i : i + 4] for i in range(8)]
    workspace.drop_snapshot(frozen)


def test_shared_snapshot_pin_survives_first_reader_close(native):
    workspace, connection, _, frozen, url = native
    with vane.File(url).open(connection=connection) as first, vane.File(url).open(connection=connection) as second:
        first.close()
        with pytest.raises(BlockingIOError, match="pinned"):
            workspace.drop_snapshot(frozen)
        assert second.read() == b"0123456789abcdef"
    workspace.drop_snapshot(frozen)


def test_standalone_reader_does_not_pin_an_unrelated_streaming_result(native):
    workspace, connection, _, frozen, url = native
    assert connection.execute("SELECT 42").fetchone() == (42,)
    with vane.File(url).open(connection=connection) as reader:
        assert reader.read() == b"0123456789abcdef"
    workspace.drop_snapshot(frozen)
    assert connection.fetchall() == []


def test_python_and_native_filesystem_registration_cannot_shadow_each_other(native):
    from vane_fs.fsspec import SnapshotFileSystem

    workspace, connection, database, frozen, _ = native
    with SnapshotFileSystem(workspace, frozen) as fs:
        with pytest.raises(vane.InvalidInputException, match="separate"):
            connection.register_filesystem(fs)
        with vane.connect() as other:
            other.register_filesystem(fs)
            with pytest.raises(vane.InvalidInputException, match="Python vanefs"):
                register_workspace(other, database)
            other.unregister_filesystem("vanefs")


def test_query_error_releases_snapshot_pin(native):
    workspace, connection, _, frozen, url = native
    with pytest.raises(vane.InvalidInputException, match="intentional query failure"):
        connection.execute(
            "SELECT CASE WHEN file_size(to_file(?)) > 0 THEN error('intentional query failure') END", [url]
        ).fetchall()
    workspace.drop_snapshot(frozen)


def test_cancelled_query_releases_native_snapshot_pin(native):
    workspace, connection, database, frozen, url = native
    with ThreadPoolExecutor(max_workers=1) as pool:
        query = pool.submit(
            lambda: connection.execute(
                "SELECT sum(i) FROM range(1000000000000) t(i) WHERE file_size(to_file(?)) = 16", [url]
            ).fetchall()
        )
        try:
            # Observe actual native acquisition before cancellation. Keep the
            # inspection SQLite library in a separate process from native I/O.
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    "import sqlite3, sys, time\n"
                    "with sqlite3.connect('file:' + sys.argv[1] + '?mode=ro', uri=True) as db:\n"
                    "    while not db.execute('SELECT count(*) FROM pins').fetchone()[0]:\n"
                    "        time.sleep(0.005)\n",
                    str(database),
                ],
                timeout=15,
                check=True,
            )
        finally:
            connection.interrupt()
        with pytest.raises(vane.InterruptException):
            query.result(timeout=15)
    workspace.drop_snapshot(frozen)
    assert connection.execute("SELECT 42").fetchone() == (42,)


def test_failed_opens_and_read_only_enforcement_do_not_leak_pins(native):
    workspace, connection, _, frozen, url = native
    with pytest.raises(vane.IOException):
        vane.File(url + "/missing").open(connection=connection)
    with pytest.raises(vane.IOException):
        connection.execute("SELECT to_file(?)", [url + "/missing"])
    with pytest.raises(vane.IOException, match="regular"):
        vane.File(snapshot_url(workspace.id, frozen, "/data")).open(connection=connection)
    with pytest.raises((vane.IOException, vane.InvalidInputException), match="range|size|bounds"):
        vane.File(url, position=15, size=2).open(connection=connection)
    with pytest.raises(vane.PermissionException, match="read-only"):
        connection.execute("COPY (SELECT 1) TO ? (FORMAT CSV)", [url])
    with pytest.raises(vane.NotImplementedException, match="globbing"):
        connection.execute("SELECT * FROM read_csv(?)", [url + "*"])
    assert connection.execute(
        "SELECT file_exists($1), try_to_file($2)", [vane.File(url + "/missing"), url + "/missing"]
    ).fetchall() == [(False, None)]
    workspace.drop_snapshot(frozen)


@pytest.mark.parametrize("suffix", ["/../file", "/../../host", "/..\x00/file"])
def test_snapshot_paths_cannot_escape_namespace(native, suffix):
    _, connection, _, _, url = native
    with pytest.raises((vane.InvalidInputException, ValueError)):
        vane.File(url + suffix).open(connection=connection)


def test_literal_unicode_and_percent_paths(native):
    workspace, connection, _, _, _ = native
    path = "/图像%2Fname?#.bin"
    workspace.checkout().write_file(path, b"literal")
    frozen = workspace.snapshot()
    with vane.File(snapshot_url(workspace.id, frozen, path)).open(connection=connection) as reader:
        assert reader.read() == b"literal"
    workspace.drop_snapshot(frozen)


def test_registration_validates_provider_and_requires_local_fast(native, tmp_path, monkeypatch):
    workspace, connection, database, _, _ = native
    with pytest.raises(vane.InvalidInputException, match="capsule"):
        connection._register_vane_fs(object())
    header = (ctypes.c_uint32 * 2)(999, 8)
    new_capsule = ctypes.pythonapi.PyCapsule_New
    new_capsule.restype = ctypes.py_object
    new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = new_capsule(ctypes.addressof(header), b"vane_fs.snapshot_reader.v1", None)
    with pytest.raises(vane.InvalidInputException, match="ABI"):
        connection._register_vane_fs(capsule)
    with pytest.raises(FileNotFoundError):
        register_workspace(connection, tmp_path / "missing.sqlite")
    assert not (tmp_path / "missing.sqlite").exists()
    monkeypatch.setenv("VANE_RUNNER", "ray")
    with vane.connect() as other:
        with pytest.raises(vane.InvalidInputException, match="local-fast"):
            register_workspace(other, database)
    # Re-registering is safe, including for an already registered workspace.
    assert register_workspace(connection, database) == workspace.id


def test_native_image_and_video_decode_match_local_files(native, tmp_path):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    image = pytest.importorskip("PIL.Image")
    workspace, connection, _, _, _ = native
    png = io.BytesIO()
    with image.new("RGB", (7, 5), (10, 20, 30)) as source:
        source.save(png, format="PNG")
    mp4 = io.BytesIO()
    with av.open(mp4, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=24)
        stream.width, stream.height, stream.pix_fmt = 16, 12, "yuv420p"
        for i in range(4):
            frame = av.VideoFrame.from_ndarray(np.full((12, 16, 3), i * 40, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    for name, payload in (("image.png", png.getvalue()), ("video.mp4", mp4.getvalue())):
        (tmp_path / name).write_bytes(payload)
        workspace.checkout().write_file("/" + name, payload)
    frozen = workspace.snapshot()
    native_image = vane.ImageFile(snapshot_url(workspace.id, frozen, "/image.png"), "image/png")
    local_image = vane.ImageFile(str(tmp_path / "image.png"), "image/png")
    assert native_image.metadata(connection=connection) == local_image.metadata(connection=connection)
    decoded = connection.execute(
        "SELECT decode_image_file($1), decode_image_file($2)", [native_image, local_image]
    ).fetchall()[0]
    np.testing.assert_array_equal(decoded[0], decoded[1])
    native_video = vane.VideoFile(snapshot_url(workspace.id, frozen, "/video.mp4"), "video/mp4")
    local_video = vane.VideoFile(str(tmp_path / "video.mp4"), "video/mp4")
    assert native_video.metadata(connection=connection) == local_video.metadata(connection=connection)
    metadata = connection.execute(
        "SELECT video_metadata($1), video_metadata($2)", [native_video, local_video]
    ).fetchall()[0]
    assert metadata[0] == metadata[1]
    native_frames = list(native_video.frames(connection=connection))
    local_frames = list(local_video.frames(connection=connection))
    try:
        assert len(native_frames) == len(local_frames) == 4
        for a, b in zip(native_frames, local_frames):
            assert (a.frame_index, a.frame_pts, a.frame_time_base) == (b.frame_index, b.frame_pts, b.frame_time_base)
            np.testing.assert_array_equal(np.asarray(a.data), np.asarray(b.data))
    finally:
        for frame in native_frames + local_frames:
            frame.data.close()
    workspace.drop_snapshot(frozen)
