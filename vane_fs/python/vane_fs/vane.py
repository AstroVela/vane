# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Connection-local registration of the optional C++ snapshot reader."""

from vane_fs import _native


def register_workspace(connection, database, *, timeout_ms=5000):
    """Register an existing database for native reads on a local-fast connection.

    Returns its workspace ID. Registration owns an independent connection;
    every query and open reader pins the snapshots it uses. SQLite write
    access is needed for these retention records even though file data is
    read-only. Register separately on each Vane connection, including cursors.
    """
    register = getattr(connection, "_register_vane_fs", None)
    if register is None:
        raise TypeError("This Vane connection does not support native VaneFS readers")
    return register(_native._reader_capsule(database, timeout_ms))


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
