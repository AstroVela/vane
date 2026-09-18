# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Read a pinned snapshot, with strict table and branch history checks."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from vane.knowledge.config import Config, digest, encode


@dataclass(frozen=True)
class Snapshot:
    table_uuid: str
    snapshot_id: int | None
    unchanged_content: bool = False


class IcebergSource:
    def __init__(self, config: Config):
        from pyiceberg.catalog import load_catalog

        self.config = config
        self.table = load_catalog(config.catalog).load_table(config.table)

    def discover(self, previous: Snapshot | None) -> Snapshot:
        self.table.refresh()
        table_uuid = str(self.table.metadata.table_uuid)
        if previous and previous.table_uuid != table_uuid:
            raise ValueError("Iceberg table UUID changed; refusing to reuse the checkpoint")
        ref = self.table.refs().get(self.config.branch)
        if ref is None:
            if self.config.branch != "main" or self.table.current_snapshot() is not None:
                raise ValueError("Configured Iceberg branch does not exist")
            if previous and previous.snapshot_id is not None:
                raise ValueError("Iceberg branch lost its snapshot")
            return Snapshot(table_uuid, None)
        if ref.snapshot_ref_type != "branch":
            raise ValueError("Iceberg source must refer to a branch, not a tag")
        head = ref.snapshot_id
        if previous and head == previous.snapshot_id:
            return Snapshot(table_uuid, head, True)
        if previous and previous.snapshot_id is not None:
            cursor: int | None = head
            seen = set()
            only_replacements = True
            while cursor != previous.snapshot_id:
                if cursor is None or cursor in seen:
                    raise ValueError("Checkpoint is not an ancestor of the branch head; no automatic reset")
                seen.add(cursor)
                snapshot = self.table.snapshot_by_id(cursor)
                if snapshot is None:
                    raise ValueError("Required Iceberg history expired; no automatic rescan")
                operation = snapshot.summary.operation.value if snapshot.summary is not None else None
                if operation not in {"append", "overwrite", "delete", "replace"}:
                    raise ValueError("Unsupported Iceberg snapshot operation")
                only_replacements &= operation == "replace"
                cursor = snapshot.parent_snapshot_id
            if self.table.snapshot_by_id(previous.snapshot_id) is None:
                raise ValueError("Checkpoint snapshot expired; no automatic rescan")
            return Snapshot(table_uuid, head, only_replacements)
        return Snapshot(table_uuid, head)

    def documents(self, snapshot: Snapshot) -> Iterator[dict[str, Any]]:
        if snapshot.snapshot_id is None:
            return
        scan = self.table.scan(snapshot_id=snapshot.snapshot_id, selected_fields=self.config.columns)
        with scan.to_arrow_batch_reader() as reader:
            for batch in reader:
                for row in batch.to_pylist():
                    yield self.document(snapshot, row)

    def document(self, snapshot: Snapshot, row: dict[str, Any]) -> dict[str, Any]:
        key = row[self.config.id_column]
        if type(key) not in (int, str) or (isinstance(key, str) and (not key or "\x00" in key)):
            raise ValueError("Document IDs must be nonempty strings or integers")
        content = row[self.config.content_column]
        title = row[self.config.title_column] if self.config.title_column else str(key)
        uri = (
            row[self.config.uri_column]
            if self.config.uri_column
            else f"iceberg://{snapshot.table_uuid}/{quote(encode(key), safe='')}"
        )
        for field, value in (("content", content), ("title", title), ("uri", uri)):
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise ValueError(f"Document {field} must be nonempty text without NUL bytes")
        document = {
            "id": encode(key),
            "slug": f"iceberg/{self.config.name}/{digest([snapshot.table_uuid, key])}",
            "title": title,
            "content": content,
            "uri": uri,
            "table_uuid": snapshot.table_uuid,
            "snapshot_id": snapshot.snapshot_id,
            "content_hash": digest([title, content, uri]),
        }
        if len(encode(document).encode("utf-8")) > self.config.max_document_bytes:
            raise ValueError("Document exceeds max_document_bytes; split it upstream")
        return document
