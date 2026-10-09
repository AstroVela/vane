# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Read-only fsspec access to one retained native VaneFS snapshot."""

from __future__ import annotations

import io
import threading
from datetime import datetime, timezone

from fsspec import AbstractFileSystem

from vane_fs import Workspace


class _SnapshotFile(io.RawIOBase):
    def __init__(self, workspace: Workspace, snapshot: str, path: str):
        super().__init__()
        self._session = workspace.open_snapshot(snapshot)
        try:
            stat = self._session.stat(path)
            if stat.is_directory:
                raise IsADirectoryError(path)
        except BaseException:
            self._session.close()
            raise
        self.size = stat.size
        self._path = path
        self._position = 0
        self._lock = threading.RLock()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        with self._lock:
            self._checkClosed()
            return self._position

    def seek(self, offset, whence=io.SEEK_SET):
        with self._lock:
            self._checkClosed()
            if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END):
                raise ValueError("Invalid seek origin")
            position = offset + (self._position if whence == io.SEEK_CUR else self.size if whence == io.SEEK_END else 0)
            if position < 0:
                raise ValueError("Negative seek position")
            self._position = position
            return position

    def readinto(self, buffer):
        with self._lock:
            self._checkClosed()
            data = self._session.read(self._path, self._position, len(buffer))
            buffer[: len(data)] = data
            self._position += len(data)
            return len(data)

    def close(self):
        # RawIOBase may call close after a partially constructed open fails.
        session = getattr(self, "_session", None)
        if session is not None:
            session.close()
        super().close()


class SnapshotFileSystem(AbstractFileSystem):
    """Expose a snapshot through ``vanefs://<workspace>/snapshots/<id>/...``.

    Keep this adapter open while the Vane connection uses it. Each open file
    holds its own native pin. All data and namespace reads execute in C++.
    """

    protocol = "vanefs"
    cachable = False
    vane_directory_semantics = True

    def __init__(self, workspace: Workspace, snapshot: str, **kwargs):
        super().__init__(**kwargs)
        self._workspace = workspace
        self._snapshot = snapshot
        self._session = workspace.open_snapshot(snapshot)
        self._prefix = f"{workspace.id}/snapshots/{snapshot}"
        self._closed = False

    def url(self, path="/"):
        """Return the stable URL for a path in this snapshot."""
        return f"vanefs://{self._prefix}/{path.lstrip('/')}"

    def _path(self, path):
        if self._closed:
            raise ValueError("Snapshot filesystem is closed")
        path = self._strip_protocol(path)
        if path == self._prefix:
            return "/"
        if not path.startswith(self._prefix + "/"):
            raise ValueError("Path is outside the bound VaneFS snapshot")
        return path[len(self._prefix) :]

    def info(self, path, **kwargs):
        stat = self._session.stat(self._path(path))
        return {
            "name": self._strip_protocol(path),
            "type": "directory" if stat.is_directory else "file",
            "size": stat.size,
            "mtime": datetime.fromtimestamp(stat.mtime_ns / 1e9, tz=timezone.utc),
            "islink": False,
        }

    def ls(self, path, detail=True, **kwargs):
        virtual_path = self._path(path)
        if not self._session.stat(virtual_path).is_directory:
            values = [self.info(path)]
        else:
            prefix = self._strip_protocol(path).rstrip("/")
            values = [self.info(f"{prefix}/{name}") for name in self._session.listdir(virtual_path)]
        return values if detail else [value["name"] for value in values]

    def modified(self, path):
        return self.info(path)["mtime"]

    def _open(self, path, mode="rb", block_size=None, **kwargs):
        if mode != "rb":
            raise PermissionError("VaneFS snapshot adapters are read-only")
        virtual_path = self._path(path)
        size = io.DEFAULT_BUFFER_SIZE if block_size is None else block_size
        if not isinstance(size, int) or size <= 0:
            raise ValueError("buffer size must be a positive integer")
        return io.BufferedReader(_SnapshotFile(self._workspace, self._snapshot, virtual_path), buffer_size=size)

    def close(self):
        if not self._closed:
            self._session.close()
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise ValueError("Snapshot filesystem is closed")
        return self

    def __exit__(self, *_):
        self.close()
