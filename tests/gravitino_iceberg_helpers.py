# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Independently populate a test-owned Iceberg REST namespace with PyIceberg."""

from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IcebergTables:
    namespace: str
    directory: Path
    rows: tuple[tuple[int, str | None], ...]
    snapshot_id: int
    metadata_location: str
    catalog: Any


@contextmanager
def iceberg_tables(endpoint: str, shared_directory: Path):
    """Create and remove only this invocation's unique namespace and directory.

    The directory must be mounted at the same absolute path in the REST server.
    Fixture-only directories allow both the server and independent client to
    write metadata/data; no existing namespace or table is reused.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyiceberg.catalog.rest import RestCatalog
    from pyiceberg.exceptions import NoSuchNamespaceError, NoSuchTableError
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField, StringType

    shared_directory = shared_directory.resolve(strict=True)
    if not shared_directory.is_dir():
        raise ValueError("Iceberg fixture shared directory must be a directory")
    namespace = "vane_iceberg_ci_" + uuid.uuid4().hex
    directory = shared_directory / namespace
    directory.mkdir(mode=0o777)
    directory.chmod(0o777)
    catalog = None
    created = False
    try:
        # Select existing no-auth behavior explicitly and bypass named catalog
        # configuration from a developer's home directory.
        catalog = RestCatalog(namespace, uri=endpoint, auth={"type": "noop"})
        catalog.create_namespace((namespace,))
        created = True
        schema = Schema(
            NestedField(1, "id", LongType(), required=True),
            NestedField(2, "label", StringType(), required=False),
        )
        arrow_schema = pa.schema([pa.field("id", pa.int64(), nullable=False), pa.field("label", pa.string())])
        for name in ("items", "empty"):
            table_directory = directory / name
            for path in (table_directory, table_directory / "metadata", table_directory / "data"):
                path.mkdir(mode=0o777)
                path.chmod(0o777)
            catalog.create_table(
                (namespace, name), schema=schema, location=table_directory.as_uri(), properties={"format-version": "2"}
            )
        rows = tuple((index, None if index % 13 == 0 else f"row-{index}.点") for index in range(4101))
        table = catalog.load_table((namespace, "items"))
        for start, end in ((0, 2050), (2050, len(rows))):
            batch = pa.Table.from_pylist(
                [{"id": key, "label": value} for key, value in rows[start:end]], schema=arrow_schema
            )
            table.append(batch)
        table.refresh()
        snapshot = table.current_snapshot()
        assert snapshot is not None
        independent = table.scan().to_arrow().sort_by("id").to_pylist()
        assert tuple((row["id"], row["label"]) for row in independent) == rows
        # Valid Parquet outside the committed manifest must never become a row
        # merely because it lives beside the Iceberg table's data files.
        orphan = pa.Table.from_pylist([{"id": 999999, "label": "uncommitted"}], schema=arrow_schema)
        pq.write_table(orphan, directory / "items" / "data" / "uncommitted.parquet")
        yield IcebergTables(namespace, directory, rows, snapshot.snapshot_id, table.metadata_location, catalog)
    finally:
        try:
            if created:
                # The namespace is unique and owned by this fixture, including
                # any table a failed READ_ONLY regression may have created.
                try:
                    for identifier in catalog.list_tables((namespace,)):
                        try:
                            catalog.drop_table(identifier)
                        except NoSuchTableError:
                            pass
                    catalog.drop_namespace((namespace,))
                except NoSuchNamespaceError:
                    pass
        finally:
            try:
                if catalog is not None:
                    catalog.close()
            finally:
                shutil.rmtree(directory)
