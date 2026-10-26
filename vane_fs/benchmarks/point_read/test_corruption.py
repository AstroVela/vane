# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Single-block joined reads must preserve corruption errors and query reset."""

import sqlite3
from contextlib import closing

import pytest

from vane_fs import Error, Workspace


@pytest.mark.parametrize("fault", ["missing_payload", "overlap"])
@pytest.mark.parametrize("offset,size", [(4096, 1), (4099, 100)])
def test_single_block_corruption(tmp_path, fault, offset, size):
    path = tmp_path / "db.sqlite"
    with Workspace(path) as workspace:
        main = workspace.checkout()
        main.write_file("/file", b"a" * 4096 + b"b" * 4096)
        workspace.snapshot()
        assert main.read("/file", offset, size) == b"b" * size
        with closing(sqlite3.connect(path)) as db:
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
                main.read("/file", offset, size)
            assert main.read("/file", 3, 100) == b"a" * 100
        main.close()
