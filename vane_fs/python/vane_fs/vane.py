# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Connection-local registration and same-host Ray replay of the native reader."""

import os
import socket

from vane_fs import _native


def register_workspace(connection, database, *, timeout_ms=5000):
    """Register an existing database for native reads on a local or Ray connection.

    Select ``local`` (or its ``local-fast`` alias) or ``ray`` execution.

    Returns its workspace ID. Registration owns an independent connection;
    every query and open reader pins the snapshots it uses. SQLite write
    access is needed for these retention records even though file data is
    read-only. Register separately on each Vane connection, including cursors.
    Ray workers reopen this absolute database path on the same host; the
    optional vane-fs package must also be installed in their environment.
    """
    register = getattr(connection, "_register_vane_fs", None)
    if register is None:
        raise TypeError("This Vane connection does not support native VaneFS readers")
    database = os.path.realpath(os.fspath(database))
    return register(_native._reader_capsule(database, timeout_ms), database, timeout_ms, socket.gethostname())


def _restore_workspace(connection, registration):
    if registration["hostname"] != socket.gethostname():
        raise ValueError("Native VaneFS Ray reads require workers on the same host as the registered workspace")
    database = registration["database"]
    if not os.path.isabs(database):
        raise ValueError("VaneFS replay requires an absolute database path")
    return connection._register_vane_fs(
        _native._reader_capsule(database, registration["timeout_ms"]),
        database,
        registration["timeout_ms"],
        registration["hostname"],
        registration["workspace_id"],
    )


def unregister_workspace(connection, workspace_id):
    """Prevent new opens; existing readers retain their own native ownership."""
    connection._unregister_vane_fs(workspace_id)


def snapshot_url(workspace_id, snapshot_id, path="/"):
    """Build a native/fsspec URL; the path is literal UTF-8, without URL decoding."""
    for name, value in (("workspace", workspace_id), ("snapshot", snapshot_id)):
        if not isinstance(value, str) or len(value) != 32 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"Invalid {name} ID")
    if not isinstance(path, str) or "\0" in path or ".." in path.split("/"):
        raise ValueError("Invalid snapshot path")
    return f"vanefs://{workspace_id}/snapshots/{snapshot_id}/{path.lstrip('/')}"
