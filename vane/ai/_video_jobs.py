# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""SQLite video job records. Only JSON data is persisted, never executable plans."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from vane.ai._video_pipeline_types import positive_number


class VideoTaskError(RuntimeError):
    """A persisted job failed or its worker lease expired."""


class VideoJobStore:
    """Durable job metadata/results on a local SQLite filesystem.

    Processes may share this database. Claims and terminal updates are fenced by
    expiring ownership tokens. An expired job becomes interrupted; resumption is
    explicit and never retries a failed model call. Reopen the same path and use
    the same application-controlled pipeline version to query/resume jobs.
    """

    def __init__(
        self, path: str | os.PathLike[str], *, lease_seconds: float = 30, max_result_bytes: int = 64 * 1024**2
    ):
        self.path = str(Path(path).expanduser().absolute())
        self.lease_seconds = positive_number(lease_seconds, "lease_seconds")
        if self.lease_seconds < 1:
            raise ValueError("lease_seconds must be at least one second")
        if type(max_result_bytes) is not int or max_result_bytes <= 0:
            raise ValueError("max_result_bytes must be a positive integer")
        self.max_result_bytes = max_result_bytes
        Path(self.path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            descriptor = None
        if descriptor is not None:
            os.close(descriptor)
        with self._connect() as database:
            database.execute("PRAGMA journal_mode=WAL")
            database.execute("""CREATE TABLE IF NOT EXISTS video_jobs (
                id TEXT PRIMARY KEY, pipeline_version TEXT NOT NULL,
                request TEXT NOT NULL, resources TEXT NOT NULL, max_batch_bytes INTEGER NOT NULL,
                status TEXT NOT NULL, owner TEXT, lease_until REAL,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                result TEXT, error TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
            )""")

    @contextmanager
    def _connect(self):
        database = sqlite3.connect(self.path, timeout=10)
        database.row_factory = sqlite3.Row
        try:
            with database:
                yield database
        finally:
            database.close()

    @staticmethod
    def _expire(database: Any, now: float) -> None:
        database.execute(
            """UPDATE video_jobs SET status=CASE WHEN cancel_requested=1 THEN 'cancelled' ELSE 'interrupted' END,
            owner=NULL, lease_until=NULL, error='Worker lease expired', updated_at=?
            WHERE owner IS NOT NULL AND lease_until<=? AND status IN ('pending','processing','cancelling')""",
            (now, now),
        )

    def submit(self, task_id: str, version: str, request: Any, resources: Any, budget: int, capacity: int) -> None:
        raw = json.dumps(request, allow_nan=False)
        options = json.dumps(resources, allow_nan=False)
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            count = database.execute("SELECT count(*) FROM video_jobs WHERE pipeline_version=?", (version,)).fetchone()[
                0
            ]
            if count >= capacity:
                raise RuntimeError("Video task capacity reached; forget completed tasks before submitting more")
            now = time.time()
            database.execute(
                """INSERT INTO video_jobs
                (id,pipeline_version,request,resources,max_batch_bytes,status,created_at,updated_at)
                VALUES (?,?,?,?,?,'pending',?,?)""",
                (task_id, version, raw, options, budget, now, now),
            )

    def get(self, task_id: str, version: str) -> dict[str, Any]:
        with self._connect() as database:
            self._expire(database, time.time())
            row = database.execute(
                "SELECT * FROM video_jobs WHERE id=? AND pipeline_version=?", (task_id, version)
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            result = dict(row)
            for key in ("request", "resources", "result"):
                result[key] = json.loads(result[key]) if result[key] is not None else None
            return result

    def list_tasks(self, pipeline_version: str, *, limit: int = 128) -> list[dict[str, Any]]:
        """List recent jobs without reading their requests or vector results."""
        if type(limit) is not int or not 0 < limit <= 10000:
            raise ValueError("limit must be between 1 and 10000")
        with self._connect() as database:
            self._expire(database, time.time())
            return [
                dict(row)
                for row in database.execute(
                    "SELECT id,status,created_at,updated_at FROM video_jobs WHERE pipeline_version=? ORDER BY created_at DESC,id LIMIT ?",
                    (pipeline_version, limit),
                )
            ]

    def get_task(self, task_id: str, *, pipeline_version: str) -> PersistedVideoTask:
        self.get(task_id, pipeline_version)
        return PersistedVideoTask(task_id, self, pipeline_version)

    def claim(self, task_id: str, version: str, owner: str) -> None:
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            now = time.time()
            self._expire(database, now)
            changed = database.execute(
                """UPDATE video_jobs SET owner=?,lease_until=?,status='pending',error=NULL,updated_at=?
                WHERE id=? AND pipeline_version=? AND owner IS NULL AND cancel_requested=0
                AND status IN ('pending','interrupted')""",
                (owner, now + self.lease_seconds, now, task_id, version),
            ).rowcount
            if changed != 1:
                raise ValueError("Only unclaimed pending or interrupted jobs can be resumed")

    def requeue(self, task_id: str, version: str) -> None:
        """Publish explicit resumption before handing back a polling handle."""
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            self._expire(database, time.time())
            changed = database.execute(
                """UPDATE video_jobs SET status='pending',error=NULL,updated_at=?
                WHERE id=? AND pipeline_version=? AND owner IS NULL AND cancel_requested=0
                AND status IN ('interrupted','pending')""",
                (time.time(), task_id, version),
            ).rowcount
            if changed != 1:
                raise ValueError("Only unclaimed pending or interrupted jobs can be resumed")

    def renew(self, task_id: str, owner: str, *, processing: bool = False) -> bool:
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            now = time.time()
            row = database.execute(
                "SELECT cancel_requested FROM video_jobs WHERE id=? AND owner=? AND lease_until>?",
                (task_id, owner, now),
            ).fetchone()
            if row is None:
                raise VideoTaskError("Video job ownership expired")
            database.execute(
                """UPDATE video_jobs SET lease_until=?,updated_at=?,
                status=CASE WHEN cancel_requested=1 THEN 'cancelling' WHEN ? THEN 'processing' ELSE status END
                WHERE id=? AND owner=? AND lease_until>?""",
                (now + self.lease_seconds, now, processing, task_id, owner, now),
            )
            return bool(row[0])

    def finish(
        self, task_id: str, owner: str, *, result: Any = None, error: str | None = None, cancelled: bool = False
    ) -> None:
        encoded = None if result is None or cancelled else json.dumps(result, allow_nan=False, separators=(",", ":"))
        if encoded is not None and len(encoded.encode()) > self.max_result_bytes:
            raise ValueError("Video job result exceeds max_result_bytes")
        with self._connect() as database:
            now = time.time()
            status = "cancelled" if cancelled else "failed" if error is not None else "ready"
            changed = database.execute(
                """UPDATE video_jobs SET status=CASE WHEN cancel_requested=1 THEN 'cancelled' ELSE ? END,
                result=CASE WHEN cancel_requested=1 THEN NULL ELSE ? END,error=?,owner=NULL,lease_until=NULL,updated_at=?
                WHERE id=? AND owner=? AND lease_until>?""",
                (status, encoded, error, now, task_id, owner, now),
            ).rowcount
            if changed != 1:
                raise VideoTaskError("Video job ownership expired before completion")

    def cancel(self, task_id: str, version: str) -> None:
        self.get(task_id, version)
        with self._connect() as database:
            database.execute(
                """UPDATE video_jobs SET cancel_requested=1,
                status=CASE WHEN owner IS NULL THEN 'cancelled' ELSE 'cancelling' END,updated_at=?
                WHERE id=? AND pipeline_version=? AND status IN ('pending','processing','interrupted','cancelling')""",
                (time.time(), task_id, version),
            )

    def forget(self, task_id: str, version: str) -> None:
        with self._connect() as database:
            self._expire(database, time.time())
            changed = database.execute(
                """DELETE FROM video_jobs WHERE id=? AND pipeline_version=? AND owner IS NULL
                AND status IN ('ready','failed','cancelled','interrupted')""",
                (task_id, version),
            ).rowcount
            if changed != 1:
                raise ValueError("Cannot forget a running or unknown video job")


class PersistedVideoTask:
    """A task handle that remains usable after reopening its job store."""

    def __init__(self, task_id: str, store: VideoJobStore, version: str):
        self.id, self._store, self._version = task_id, store, version

    @property
    def status(self) -> str:
        return self._store.get(self.id, self._version)["status"]

    def cancel(self) -> None:
        self._store.cancel(self.id, self._version)

    def result(self, timeout: float | None = None) -> dict[str, Any]:
        from concurrent.futures import CancelledError

        deadline = None if timeout is None else time.monotonic() + max(0, timeout)
        while True:
            row = self._store.get(self.id, self._version)
            if row["status"] == "ready":
                return row["result"]
            if row["status"] == "cancelled":
                raise CancelledError()
            if row["status"] in {"failed", "interrupted"}:
                raise VideoTaskError(row["error"] or row["status"])
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for video job")
            time.sleep(0.05)
