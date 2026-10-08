# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Additional checks against an isolated, installed page-layout wheel."""

import json
import os
import subprocess
import sys

from vane_fs import Workspace


def sql(path, script="", query="SELECT 1"):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import json,sqlite3,sys; d=sqlite3.connect(sys.argv[1]); d.executescript(sys.argv[2]); "
            "print(json.dumps(d.execute(sys.argv[3]).fetchall())); d.close()",
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


def test_new_database_layout_and_independent_gc(tmp_path):
    path = tmp_path / "workspace.sqlite"
    page = int(os.environ["VANE_FS_PAGE_BYTES"])
    data = bytes(range(256)) * 4096
    with Workspace(path) as workspace:
        main = workspace.checkout()
        main.write_file("/file", data)
        assert sql(path, query="PRAGMA page_size") == [[page]]
        assert sql(path, query="PRAGMA auto_vacuum") == [[0]]
        assert sql(path, query="SELECT version,block_size FROM format") == [[2, 4096]]
        assert sql(path, query="SELECT count(*),min(length(data)),max(length(data)) FROM block_payloads") == [
            [256, 4096, 4096]
        ]
        overflow = sql(path, query="SELECT count(*) FROM dbstat WHERE name='block_payloads' AND pagetype='overflow'")
        assert overflow == [[256 if page == 4096 else 0]]
        snapshot = workspace.snapshot()
        main.write("/file", b"z" * 4096)
        workspace.collect_garbage()
        assert sql(path, query="SELECT sum(length(data)) FROM block_payloads") == [[len(data) + 4096]]
        with workspace.open_snapshot(snapshot) as frozen:
            assert frozen.read("/file") == data
        workspace.drop_snapshot(snapshot)
        workspace.collect_garbage()
        assert sql(path, query="SELECT sum(length(data)) FROM block_payloads") == [[len(data)]]
        assert main.read("/file") == b"z" * 4096 + data[4096:]
    with Workspace(path) as workspace:
        assert workspace.checkout().read("/file") == b"z" * 4096 + data[4096:]
        assert sql(path, query="PRAGMA page_size") == [[page]]


def test_preinitialized_page_size_is_preserved(tmp_path):
    path = tmp_path / "existing.sqlite"
    page = 8192 if int(os.environ["VANE_FS_PAGE_BYTES"]) == 4096 else 4096
    sql(path, f"PRAGMA page_size={page}; VACUUM;")
    with Workspace(path) as workspace:
        workspace.checkout().write_file("/file", b"old layout" * 10000)
        assert sql(path, query="PRAGMA page_size") == [[page]]
    with Workspace(path) as workspace:
        assert workspace.checkout().read("/file") == b"old layout" * 10000
        assert sql(path, query="PRAGMA page_size") == [[page]]


def test_existing_production_binary_reads_and_writes_candidate_database(tmp_path):
    path = tmp_path / "compatible.sqlite"
    data = bytes(range(251)) * 5000
    with Workspace(path) as workspace:
        workspace.checkout().write_file("/file", data)
    # A separate interpreter avoids loading two different SQLite libraries together.
    subprocess.run(
        [
            os.environ["VANE_FS_PRODUCTION_PYTHON"],
            "-I",
            "-c",
            "import sys; from vane_fs import Workspace; "
            "w=Workspace(sys.argv[1]); s=w.checkout(); "
            "assert s.read('/file')==bytes(range(251))*5000; s.write('/file',b'old',17); w.close()",
            str(path),
        ],
        check=True,
        timeout=15,
    )
    with Workspace(path) as workspace:
        assert workspace.checkout().read("/file") == data[:17] + b"old" + data[20:]
        assert sql(path, query="PRAGMA page_size") == [[int(os.environ["VANE_FS_PAGE_BYTES"])]]
