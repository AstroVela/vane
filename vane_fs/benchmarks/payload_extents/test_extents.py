# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run only against an isolated, installed extent prototype wheel."""

import hashlib
import json
import os
import random
import subprocess
import sys

import pytest

from vane_fs import Error, Workspace


def sql(path, script, query="SELECT 1"):
    # Keep Python's SQLite in a different process from the bundled C++ library.
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import json,sqlite3,sys; d=sqlite3.connect(sys.argv[1]); "
            "d.executescript(sys.argv[2]); print(json.dumps(d.execute(sys.argv[3]).fetchall())); d.close()",
            str(path),
            script,
            query,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return json.loads(result.stdout)


@pytest.fixture
def opened(tmp_path):
    path = tmp_path / "workspace.sqlite"
    with Workspace(path) as workspace:
        assert sql(path, "", "PRAGMA application_id") == [[0x56465831]]
        yield path, workspace


def test_slices_sparse_boundaries_and_small_copy_on_write(opened):
    path, workspace = opened
    extent = int(os.environ["VANE_FS_EXTENT_BYTES"])
    data = b"".join(bytes([i % 251 if i % 7 else 0]) * 4096 for i in range(193)) + b"tail"
    main = workspace.checkout()
    main.write_file("/file", data)
    before = sql(path, "", "SELECT sum(length(data)) FROM block_payloads")[0][0]
    frozen_id = workspace.snapshot()
    child = workspace.checkout(workspace.fork("main", "child").id)
    expected = bytearray(data)
    changes = [(extent - 1, b"XYZ"), (4096 * 65 + 17, b"q")]
    for offset, value in changes:
        child.write("/file", value, offset)
        expected[offset : offset + len(value)] = value
    after = sql(path, "", "SELECT sum(length(data)) FROM block_payloads")[0][0]
    assert after - before == 3 * 4096
    with workspace.open_snapshot(frozen_id) as frozen, Workspace(path) as other:
        workspace.collect_garbage()
        assert frozen.read("/file") == data
        assert main.read("/file") == data
        assert other.checkout("child").read("/file") == expected
        rng = random.Random(961)
        for _ in range(300):
            offset = rng.randrange(len(data))
            amount = rng.randrange(1, 16385)
            assert child.read("/file", offset, amount) == expected[offset : offset + amount]
    workspace.merge(workspace.preview_merge("child", "main"))
    assert main.read("/file") == expected
    workspace.delete_branch("child")
    workspace.drop_snapshot(frozen_id)
    workspace.collect_garbage()
    assert main.read("/file") == expected
    main.truncate("/file", extent + 3)
    main.truncate("/file", len(data))
    assert main.read("/file") == expected[: extent + 3] + b"\0" * (len(data) - extent - 3)


def test_identical_content_with_different_extent_layout_merges(opened):
    _, workspace = opened
    main = workspace.checkout()
    main.write_file("/file", b"")
    source = workspace.checkout(workspace.fork("main", "source").id)
    data = bytes(range(251)) * 2200
    source.write("/file", data)
    for offset in range(0, len(data), 4096):
        main.write("/file", data[offset : offset + 4096], offset)
    preview = workspace.preview_merge("source", "main")
    assert not preview.conflicts
    workspace.merge(preview)
    assert main.read("/file") == data


def test_gc_retains_partial_extent_then_reclaims_it(opened):
    path, workspace = opened
    extent = int(os.environ["VANE_FS_EXTENT_BYTES"])
    main = workspace.checkout()
    main.write_file("/file", b"a" * extent)
    frozen_id = workspace.snapshot()
    main.write("/file", b"b" * 4096)
    workspace.collect_garbage()
    assert sql(path, "", "SELECT sum(length(data)) FROM block_payloads") == [[extent + 4096]]
    workspace.drop_snapshot(frozen_id)
    workspace.collect_garbage()
    # One dead 4 KiB slice cannot release a still-referenced immutable extent.
    assert sql(path, "", "SELECT sum(length(data)) FROM block_payloads") == [[extent + 4096]]
    for offset in range(4096, extent, 4096):
        main.write("/file", b"b" * 4096, offset)
    workspace.collect_garbage()
    assert sql(path, "", "SELECT sum(length(data)) FROM block_payloads") == [[extent]]
    assert main.read("/file") == b"b" * extent
    main.unlink("/file")
    workspace.collect_garbage()
    assert sql(path, "", "SELECT count(*) FROM block_payloads") == [[0]]


@pytest.mark.parametrize("fault", ["missing", "offset"])
def test_invalid_slice_reports_error_then_reader_recovers(opened, fault):
    path, workspace = opened
    main = workspace.checkout()
    data = bytes(range(251)) * 3000
    main.write_file("/file", data)
    if fault == "missing":
        sql(path, "CREATE TABLE saved AS SELECT * FROM block_payloads; DELETE FROM block_payloads;")
    else:
        sql(
            path, "CREATE TABLE saved AS SELECT * FROM block_versions; UPDATE block_versions SET payload_offset=262144;"
        )
    with pytest.raises(Error, match="Missing block payload|Invalid block payload slice"):
        main.read("/file", 4095, 8193)
    if fault == "missing":
        sql(path, "INSERT INTO block_payloads SELECT * FROM saved; DROP TABLE saved;")
    else:
        sql(path, "DELETE FROM block_versions; INSERT INTO block_versions SELECT * FROM saved; DROP TABLE saved;")
    assert main.read("/file") == data


def test_experimental_and_production_formats_refuse_each_other(tmp_path):
    cli = os.environ["VANE_FS_BASELINE_CLI"]
    production = tmp_path / "production.sqlite"
    subprocess.run([cli, str(production), "init"], check=True, capture_output=True, timeout=15)
    before = hashlib.sha256(production.read_bytes()).digest()
    with pytest.raises(ValueError, match="Not a VaneFS database"):
        Workspace(production)
    assert hashlib.sha256(production.read_bytes()).digest() == before
    experimental = tmp_path / "experimental.sqlite"
    with Workspace(experimental) as workspace:
        workspace.checkout().write_file("/file", b"data")
    before = hashlib.sha256(experimental.read_bytes()).digest()
    result = subprocess.run([cli, str(experimental), "branches"], capture_output=True, text=True, timeout=15)
    assert result.returncode == 1 and "Not a VaneFS database" in result.stderr
    assert hashlib.sha256(experimental.read_bytes()).digest() == before
