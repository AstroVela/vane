# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Transactional snapshot diff, durable outbox, and per-destination receipts."""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from vane.knowledge.config import Config, encode
from vane.knowledge.iceberg import Snapshot

_SCHEMA = """
BEGIN IMMEDIATE;
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE documents (id TEXT PRIMARY KEY, hash TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE staging (id TEXT PRIMARY KEY, hash TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE batch (id TEXT PRIMARY KEY, table_uuid TEXT NOT NULL, snapshot_id INTEGER);
CREATE TABLE events (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE deliveries (target TEXT PRIMARY KEY, position INTEGER NOT NULL DEFAULT 0, done INTEGER NOT NULL DEFAULT 0);
PRAGMA user_version = 1;
COMMIT;
"""


class State:
    def __init__(self, directory: Path, *, create: bool = False):
        self.directory = directory
        if create:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir() or directory.stat().st_mode & 0o077:
            raise ValueError("state_dir must be a private directory (mode 0700), not a symlink")
        path = directory / "state.sqlite3"
        if path.is_symlink():
            raise ValueError("State database must not be a symlink")
        if create:
            fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            os.close(fd)
        if not path.is_file() or path.stat().st_mode & 0o077:
            raise ValueError("State database must exist and have mode 0600")
        # Initialization is also a write: a second service must not observe a
        # partly created schema or initialize the same database concurrently.
        with self.exclusive() if create else nullcontext():
            self.db = sqlite3.connect(f"{path.as_uri()}?mode=rw", uri=True, timeout=30)
            self.db.row_factory = sqlite3.Row
            try:
                version = self.db.execute("PRAGMA user_version").fetchone()[0]
                if create and version == 0 and not self.db.execute("SELECT name FROM sqlite_master").fetchall():
                    self.db.executescript(_SCHEMA)
                elif version != 1:
                    raise ValueError("Unsupported state schema; no migration is performed")
            except BaseException:
                self.db.close()
                raise

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        fd = os.open(self.directory / "writer.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another synchronizer owns this state directory") from exc
            yield
        finally:
            os.close(fd)

    def get(self, key: str) -> Any:
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: Any) -> None:
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", (key, encode(value)))

    def bind(self, config: Config) -> None:
        with self.db:
            old = self.get("identity")
            if old is not None and old != config.identity:
                raise ValueError("Configuration changed; use a separate state directory and destination namespace")
            self.put("identity", config.identity)

    def checkpoint(self) -> Snapshot | None:
        value = self.get("checkpoint")
        return Snapshot(**value) if value is not None else None

    def pending(self) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM batch").fetchone()
        return dict(row) if row else None

    def stage(self, snapshot: Snapshot, documents: Iterator[dict[str, Any]], targets: tuple[str, ...]) -> None:
        if self.pending():
            raise RuntimeError("Deliver the pending batch before reading a new snapshot")
        with self.db:
            if snapshot.unchanged_content:
                self._checkpoint(snapshot)
                return
            self.db.execute("DELETE FROM staging")
            for doc in documents:
                try:
                    self.db.execute(
                        "INSERT INTO staging VALUES (?, ?, ?)", (doc["id"], doc["content_hash"], encode(doc))
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError("Iceberg document IDs are not unique") from exc
            self.db.execute(
                "INSERT INTO events(kind, body) SELECT 'upsert', s.body FROM staging s "
                "LEFT JOIN documents d ON d.id=s.id WHERE d.id IS NULL OR d.hash<>s.hash ORDER BY s.id"
            )
            for row in self.db.execute(
                "SELECT d.body FROM documents d LEFT JOIN staging s ON s.id=d.id WHERE s.id IS NULL ORDER BY d.id"
            ):
                doc = json.loads(row[0])
                doc["content"] = ""
                doc["snapshot_id"] = snapshot.snapshot_id
                self.db.execute("INSERT INTO events(kind, body) VALUES ('delete', ?)", (encode(doc),))
            self.db.execute("DELETE FROM documents")
            self.db.execute("INSERT INTO documents SELECT * FROM staging")
            self.db.execute("DELETE FROM staging")
            if self.db.execute("SELECT 1 FROM events LIMIT 1").fetchone():
                self.db.execute(
                    "INSERT INTO batch VALUES (?, ?, ?)", (str(uuid.uuid4()), snapshot.table_uuid, snapshot.snapshot_id)
                )
                self.db.executemany("INSERT INTO deliveries(target) VALUES (?)", ((t,) for t in targets))
            else:
                self._checkpoint(snapshot)

    def _checkpoint(self, snapshot: Snapshot) -> None:
        self.put("checkpoint", {"table_uuid": snapshot.table_uuid, "snapshot_id": snapshot.snapshot_id})

    def events(self, after: int = 0) -> Iterator[dict[str, Any]]:
        for row in self.db.execute("SELECT * FROM events WHERE seq>? ORDER BY seq", (after,)):
            yield {"seq": row["seq"], "kind": row["kind"], "document": json.loads(row["body"])}

    def delivery(self, target: str) -> dict[str, Any]:
        row = self.db.execute("SELECT * FROM deliveries WHERE target=?", (target,)).fetchone()
        if row is None:
            raise ValueError("Target is not part of the pending batch")
        return dict(row)

    def acknowledge(self, target: str, position: int, *, done: bool = False) -> None:
        with self.db:
            self.db.execute("UPDATE deliveries SET position=?, done=? WHERE target=?", (position, done, target))

    def finish(self) -> None:
        pending = self.pending()
        if pending is None or self.db.execute("SELECT 1 FROM deliveries WHERE done=0").fetchone():
            raise RuntimeError("Cannot finish a batch before every destination acknowledges it")
        with self.db:
            self._checkpoint(Snapshot(pending["table_uuid"], pending["snapshot_id"]))
            for table in ("batch", "events", "deliveries"):
                self.db.execute(f"DELETE FROM {table}")
            self.db.execute("DELETE FROM metadata WHERE key='attempt'")

    def start_attempt(self, target: str) -> dict[str, Any]:
        position = self.delivery(target)["position"]
        end, size, count = position, 0, 0
        for event in self.events(position):
            size += len(encode(event).encode("utf-8"))
            if count and (size > 65536 or count == 100):
                break
            end, count = event["seq"], count + 1
        if not count:
            raise ValueError("No events remain for this target")
        attempt = {
            "id": str(uuid.uuid4()),
            "target": target,
            "start": position,
            "end": end,
            "event_count": count,
            "receipt": secrets.token_hex(32),
            "acknowledged": False,
        }
        with self.db:
            self.put("attempt", attempt)
        return attempt

    def is_acknowledged(self, attempt_id: str) -> bool:
        attempt = self.get("attempt")
        return attempt is not None and attempt["id"] == attempt_id and attempt["acknowledged"] is True

    def _attempt(self, batch_id: str) -> dict[str, Any]:
        pending = self.pending()
        attempt = self.get("attempt")
        if pending is None or pending["id"] != batch_id or attempt is None:
            raise ValueError("Batch is not available for an active OpenWiki ingestion")
        return dict(attempt)

    def read_changes(self, batch_id: str) -> dict[str, Any]:
        attempt = self._attempt(batch_id)
        events = []
        for event in self.events(attempt["start"]):
            if event["seq"] > attempt["end"]:
                break
            events.append(event)
        # The receipt is exposed only alongside all the content it acknowledges.
        # A server read log cannot establish that a response reached its client.
        return {"batch_id": batch_id, "events": events, "receipt": attempt["receipt"]}

    def acknowledge_changes(self, batch_id: str, receipt: str) -> None:
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            attempt = self._attempt(batch_id)
            if not isinstance(receipt, str) or not secrets.compare_digest(receipt, attempt["receipt"]):
                raise ValueError("Receipt does not match the active delivery window")
            attempt["acknowledged"] = True
            self.put("attempt", attempt)
