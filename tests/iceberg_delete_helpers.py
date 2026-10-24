# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""SDK-authored deletes and a separate field-ID/Arrow reference evaluator.

Applicability and NULL matching follow https://iceberg.apache.org/spec/#scan-planning
and #delete-formats. The pinned SDK does not execute equality deletes; expected
rows come from its manifest entries and independently read Parquet values.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from iceberg_partition_helpers import append_partition_rows

PATH_ID, POS_ID = 2147483546, 2147483545
DELETE_TABLES = (
    "delete_positions",
    "delete_position_sequence",
    "delete_equality_nullable",
    "delete_equality_multi",
    "delete_partitioned",
    "delete_specs",
    "delete_schema",
    "delete_typed",
    "delete_all",
    "delete_removed",
)


def local_path(uri):
    return Path(unquote(urlsplit(uri).path))


def manifest_entries(table, snapshot_id=None):
    from pyiceberg.manifest import ManifestEntryStatus

    snapshot = table.current_snapshot() if snapshot_id is None else table.snapshot_by_id(snapshot_id)
    if snapshot is None:
        return []
    return [
        (manifest, entry)
        for manifest in snapshot.manifests(table.io)
        for entry in manifest.fetch_manifest_entry(table.io)
        if entry.status != ManifestEntryStatus.DELETED
    ]


def file_rows(file):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(local_path(file.file_path))
    fields = {field.name: int(field.metadata[b"PARQUET:field_id"]) for field in parquet.schema_arrow}
    assert len(set(fields.values())) == len(fields)
    rows = [{fields[name]: value for name, value in row.items()} for row in parquet.read().to_pylist()]
    assert len(rows) == file.record_count
    return rows


@dataclass(frozen=True)
class DeleteView:
    names: tuple[str, ...]
    types: tuple[str, ...]
    rows: tuple[tuple, ...]
    snapshot_id: int


@dataclass(frozen=True)
class DeleteFeature:
    name: str
    current: DeleteView
    history: tuple[DeleteView, ...]
    file_count: int


def delete_reference(tables, name, *, snapshot_id=None):
    from pyiceberg.transforms import IdentityTransform, VoidTransform
    from pyiceberg.types import DecimalType

    table = tables.catalog.load_table((tables.namespace, name))
    snapshot = table.current_snapshot() if snapshot_id is None else table.snapshot_by_id(snapshot_id)
    schema = table.schema() if snapshot_id is None else table.schemas()[snapshot.schema_id]
    entries = manifest_entries(table, snapshot_id)
    deletes = [
        (manifest.partition_spec_id, entry, file_rows(entry.data_file))
        for manifest, entry in entries
        if entry.data_file.content != 0
    ]
    specs = table.specs()
    output = []
    for manifest, entry in entries:
        file = entry.data_file
        if file.content != 0:
            continue
        spec = specs[manifest.partition_spec_id]
        constants = {
            field.source_id: file.partition[index]
            for index, field in enumerate(spec.fields)
            if isinstance(field.transform, IdentityTransform)
        }
        for position, raw in enumerate(file_rows(file)):
            values = {**constants, **raw}
            removed = False
            for delete_spec_id, delete, rows in deletes:
                deleted = delete.data_file
                partition_equal = delete_spec_id == manifest.partition_spec_id and tuple(deleted.partition) == tuple(
                    file.partition
                )
                if deleted.content == 1:
                    if entry.sequence_number <= delete.sequence_number and partition_equal:
                        removed = any(row[PATH_ID] == file.file_path and row[POS_ID] == position for row in rows)
                elif entry.sequence_number < delete.sequence_number:
                    global_delete = all(
                        isinstance(field.transform, VoidTransform) for field in specs[delete_spec_id].fields
                    )
                    if global_delete or partition_equal:
                        removed = any(
                            all(values.get(field_id) == row[field_id] for field_id in deleted.equality_ids)
                            for row in rows
                        )
                if removed:
                    break
            if not removed:
                output.append(tuple(values.get(field.field_id) for field in schema.fields))
    names = tuple(field.name for field in schema.fields)
    types = {"long": "BIGINT", "int": "INTEGER", "string": "VARCHAR", "boolean": "BOOLEAN", "date": "DATE"}
    native_types = tuple(
        f"DECIMAL({field.field_type.precision},{field.field_type.scale})"
        if isinstance(field.field_type, DecimalType)
        else types[str(field.field_type)]
        for field in schema.fields
    )
    output.sort(key=lambda row: row[names.index("id")])
    return DeleteView(names, native_types, tuple(output), snapshot.snapshot_id if snapshot else 0)


def commit_deletes(
    tables, name, rows=(), *, equality_ids=(), spec_id=None, partition=(), data_sequence=None, clear=False
):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyiceberg.io.pyarrow import schema_to_pyarrow
    from pyiceberg.manifest import (
        DataFile,
        DataFileContent,
        FileFormat,
        ManifestContent,
        ManifestEntry,
        ManifestEntryStatus,
        ManifestWriterV2,
        write_manifest_list,
    )
    from pyiceberg.schema import Schema
    from pyiceberg.table.snapshots import Operation, Snapshot, Summary
    from pyiceberg.table.update import AddSnapshotUpdate, AssertRefSnapshotId, AssertTableUUID, SetSnapshotRefUpdate
    from pyiceberg.typedef import Record
    from pyiceberg.types import LongType, NestedField, StringType

    class DeleteWriter(ManifestWriterV2):
        def content(self):
            return ManifestContent.DELETES

        @property
        def _meta(self):
            return {**super()._meta, "content": "deletes"}

    table = tables.catalog.load_table((tables.namespace, name))
    parent = table.current_snapshot()
    if not equality_ids and not clear:
        rows = sorted(rows, key=lambda row: (row[PATH_ID], row[POS_ID]))
    sequence = table.metadata.last_sequence_number + 1
    delete_sequence = sequence if data_sequence is None else data_sequence
    assert 0 <= delete_sequence <= sequence
    snapshot_id = uuid.uuid4().int >> 65
    token = uuid.uuid4().hex
    directory = tables.directory / name
    manifests = parent.manifests(table.io)
    if clear:
        assert not rows and not equality_ids
        manifests = [manifest for manifest in manifests if manifest.content != ManifestContent.DELETES]
    else:
        spec_id = table.spec().spec_id if spec_id is None else spec_id
        spec = table.specs()[spec_id]
        if equality_ids:
            fields = []
            for field_id in reversed(equality_ids):
                candidates = [
                    schema.find_field(field_id) for schema in table.schemas().values() if field_id in schema.field_ids
                ]
                assert candidates
                field = candidates[-1]
                fields.append(NestedField(field_id, f"physical_{field_id}", field.field_type))
            if 1 not in equality_ids:
                fields.append(NestedField(1, "unrelated_payload", LongType()))
            schema = Schema(*fields)
            physical = [{field.name: row.get(field.field_id, 999999) for field in fields} for row in rows]
        else:
            schema = Schema(
                NestedField(POS_ID, "pos", LongType(), required=True),
                NestedField(PATH_ID, "file_path", StringType(), required=True),
            )
            physical = [{"file_path": row[PATH_ID], "pos": row[POS_ID]} for row in rows]
        path = directory / "data" / f"delete-{token}.parquet"
        pq.write_table(pa.Table.from_pylist(physical, schema=schema_to_pyarrow(schema)), path, row_group_size=3)
        deleted = DataFile.from_args(
            content=DataFileContent.EQUALITY_DELETES if equality_ids else DataFileContent.POSITION_DELETES,
            file_path=path.as_uri(),
            file_format=FileFormat.PARQUET,
            partition=Record(*partition),
            record_count=len(rows),
            file_size_in_bytes=path.stat().st_size,
            equality_ids=list(equality_ids) if equality_ids else None,
        )
        deleted.spec_id = spec_id
        manifest_path = (directory / "metadata" / f"delete-{token}.avro").as_uri()
        with DeleteWriter(spec, table.schema(), table.io.new_output(manifest_path), snapshot_id, "null") as writer:
            writer.add_entry(
                ManifestEntry.from_args(
                    status=ManifestEntryStatus.ADDED,
                    snapshot_id=snapshot_id,
                    sequence_number=delete_sequence,
                    file_sequence_number=sequence,
                    data_file=deleted,
                )
            )
        manifests = [writer.to_manifest_file(), *manifests]
    listing_path = (directory / "metadata" / f"delete-list-{token}.avro").as_uri()
    with write_manifest_list(
        2, table.io.new_output(listing_path), snapshot_id, parent.snapshot_id, sequence, "null"
    ) as listing:
        listing.add_manifests(manifests)
    old_entries = manifest_entries(table)
    data_files = [entry.data_file for _manifest, entry in old_entries if entry.data_file.content == 0]
    delete_files = (
        [] if clear else [entry.data_file for _manifest, entry in old_entries if entry.data_file.content != 0]
    )
    if not clear:
        delete_files.append(deleted)
    snapshot = Snapshot(
        snapshot_id=snapshot_id,
        parent_snapshot_id=parent.snapshot_id,
        sequence_number=sequence,
        manifest_list=listing_path,
        schema_id=table.schema().schema_id,
        summary=Summary(
            operation=Operation.DELETE,
            **{
                "total-records": str(sum(file.record_count for file in data_files)),
                "total-data-files": str(len(data_files)),
                "total-delete-files": str(len(delete_files)),
                "total-position-deletes": str(sum(file.record_count for file in delete_files if file.content == 1)),
                "total-equality-deletes": str(sum(file.record_count for file in delete_files if file.content == 2)),
            },
        ),
    )
    tables.catalog.commit_table(
        table,
        requirements=(
            AssertTableUUID(uuid=table.metadata.table_uuid),
            AssertRefSnapshotId(ref="main", snapshot_id=parent.snapshot_id),
        ),
        updates=(
            AddSnapshotUpdate(snapshot=snapshot),
            SetSnapshotRefUpdate(ref_name="main", type="branch", snapshot_id=snapshot_id),
        ),
    )
    table.refresh()
    if not clear:
        matches = [
            (manifest, entry)
            for manifest, entry in manifest_entries(table)
            if entry.data_file.file_path == deleted.file_path
        ]
        assert len(matches) == 1
        manifest, observed = matches[0]
        assert manifest.partition_spec_id == spec_id
        assert observed.sequence_number == delete_sequence and observed.file_sequence_number == sequence
        assert observed.data_file.content == deleted.content and observed.data_file.equality_ids == deleted.equality_ids
        assert tuple(observed.data_file.partition) == tuple(partition)
        assert pq.read_schema(path).equals(schema_to_pyarrow(schema), check_metadata=True)
        assert pq.ParquetFile(path).read().to_pylist() == physical
    return table


def populate_delete_tables(tables, *, names=DELETE_TABLES):
    from datetime import date
    from decimal import Decimal

    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import BooleanType, DateType, DecimalType, IntegerType, LongType, NestedField, StringType

    schema = Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "label", StringType()),
        NestedField(3, "region", StringType()),
        NestedField(4, "rank", LongType()),
    )
    identity = PartitionSpec(
        PartitionField(source_id=3, field_id=1000, transform=IdentityTransform(), name="region_part")
    )

    def create(name, *, spec=None, table_schema=schema):
        directory = tables.directory / name
        for path in (directory, directory / "metadata", directory / "data"):
            path.mkdir(mode=0o777)
            path.chmod(0o777)
        return tables.catalog.create_table(
            (tables.namespace, name),
            schema=table_schema,
            partition_spec=spec or PartitionSpec(),
            location=directory.as_uri(),
            properties={"format-version": "2"},
        )

    def rows(start, count=9):
        return [
            {
                "id": key,
                "label": (None, "remove", "keep.点")[key % 3],
                "region": ("east.点", "west", None)[(key // 3) % 3],
                "rank": (None, 0, 1, 2)[(key // 2) % 4],
            }
            for key in range(start, start + count)
        ]

    def append(name, batch, *, omitted=False, row_group_size=None):
        return append_partition_rows(
            tables,
            name,
            batch,
            omit_partition_columns=omitted,
            rename_partition_columns=True,
            row_group_size=row_group_size,
        )

    features = {}
    for name in names:
        if name == "delete_typed":
            typed_schema = Schema(
                *schema.fields[:3],
                NestedField(4, "rank", IntegerType()),
                NestedField(5, "amount", DecimalType(8, 2)),
                NestedField(6, "day", DateType()),
                NestedField(7, "flag", BooleanType()),
            )
            table = create(name, table_schema=typed_schema)
        else:
            table = create(name, spec=identity if name == "delete_partitioned" else None)
        history = []
        if name == "delete_positions":
            for start in (0, 4099, 8198):
                table = append(name, rows(start, 4099), row_group_size=128)
            history.append(delete_reference(tables, name))
            entries = [(manifest, entry) for manifest, entry in manifest_entries(table) if entry.data_file.content == 0]
            deleted = [
                {PATH_ID: entry.data_file.file_path, POS_ID: position}
                for _manifest, entry in entries
                for position in (0, 127, 128, 2047, 2048, 4098, 2048)
            ]
            deleted.append({PATH_ID: entries[0][1].data_file.file_path.replace("opaque", "Opaque"), POS_ID: 1})
            commit_deletes(tables, name, sorted(deleted, key=lambda row: (row[PATH_ID], row[POS_ID])))
            history.append(delete_reference(tables, name))
            table = append(name, rows(12297, 3))
            independent = table.scan().to_arrow().sort_by("id").to_pylist()
            assert (
                tuple(tuple(row[column] for column in schema.column_names) for row in independent)
                == delete_reference(tables, name).rows
            )
        elif name == "delete_typed":
            for start in (0, 3, 6):
                batch = [
                    dict(
                        row,
                        amount=(Decimal("1.25"), Decimal("-2.50"), None)[row["id"] % 3],
                        day=date(2024, 1, 2) if row["id"] % 2 else None,
                        flag=bool(row["id"] % 2),
                    )
                    for row in rows(start, 3)
                ]
                table = append(name, batch)
            history.append(delete_reference(tables, name))
            table = commit_deletes(tables, name, [{4: 0, 5: Decimal("1.25")}, {4: 2, 5: None}], equality_ids=(4, 5))
            history.append(delete_reference(tables, name))
            with table.update_schema() as update:
                update.update_column("rank", field_type=LongType())
                update.update_column("amount", field_type=DecimalType(18, 2))
            for start in (9, 12, 15):
                batch = [
                    dict(
                        row,
                        rank=2**40 + row["id"],
                        amount=Decimal("12345678901234.56") if row["id"] % 2 else None,
                        day=date(2024, 1, 2) if row["id"] % 2 else None,
                        flag=bool(row["id"] % 2),
                    )
                    for row in rows(start, 3)
                ]
                table = append(name, batch)
            history.append(delete_reference(tables, name))
            with table.update_schema() as update:
                update.delete_column("rank")
            table = commit_deletes(
                tables,
                name,
                [{5: Decimal("12345678901234.56"), 6: date(2024, 1, 2), 7: True}, {5: None, 6: None, 7: False}],
                equality_ids=(5, 6, 7),
            )
        elif name == "delete_schema":
            # Initial optional equality field is dropped and its name reused.
            for start in (0, 3, 6):
                table = append(name, rows(start, 3))
            history.append(delete_reference(tables, name))
            table = commit_deletes(tables, name, [{2: "remove"}], equality_ids=(2,))
            history.append(delete_reference(tables, name))
            with table.update_schema() as update:
                update.rename_column("label", "retired")
            with table.update_schema() as update:
                update.delete_column("retired")
                update.add_column("label", StringType())
                update.add_column("new_key", LongType())
            for start in (9, 12, 15):
                batch = [dict(row, label="new-name", new_key=row["id"] % 2) for row in rows(start, 3)]
                table = append(name, batch)
            history.append(delete_reference(tables, name))
            new_id = table.schema().find_field("new_key").field_id
            table = commit_deletes(tables, name, [{new_id: None, 4: 0}], equality_ids=(new_id, 4))
        else:
            for start in range(0, 27, 3):
                if name == "delete_specs" and start == 9:
                    with table.update_spec() as update:
                        update.add_field("region", IdentityTransform(), "region_part")
                elif name == "delete_specs" and start == 18:
                    with table.update_spec() as update:
                        update.rename_field("region_part", "renamed_region")
                table = append(name, rows(start, 3), omitted=name in {"delete_partitioned", "delete_specs"})
            history.append(delete_reference(tables, name))
            if name == "delete_position_sequence":
                data = [
                    (manifest, entry) for manifest, entry in manifest_entries(table) if entry.data_file.content == 0
                ]
                seq = sorted(entry.sequence_number for _manifest, entry in data)[4]
                table = commit_deletes(
                    tables,
                    name,
                    [{PATH_ID: entry.data_file.file_path, POS_ID: 1} for _manifest, entry in data],
                    data_sequence=seq,
                )
            elif name == "delete_equality_multi":
                table = commit_deletes(
                    tables,
                    name,
                    [{4: 0, 2: "remove"}, {4: 1, 2: None}, {4: None, 2: None}, {4: 0, 2: "remove"}],
                    equality_ids=(4, 2),
                )
            elif name == "delete_partitioned":
                commit_deletes(tables, name, [{2: "remove"}], equality_ids=(2,), partition=("east.点",))
                table = commit_deletes(tables, name, [{2: None}], equality_ids=(2,), partition=(None,))
            elif name == "delete_specs":
                commit_deletes(tables, name, [{2: "remove"}], equality_ids=(2,), spec_id=1, partition=("east.点",))
                table = commit_deletes(tables, name, [{2: "keep.点"}], equality_ids=(2,), spec_id=0)
            elif name == "delete_all":
                table = commit_deletes(tables, name, [{1: key} for key in range(27)], equality_ids=(1,))
            else:
                commit_deletes(tables, name, [{2: "remove"}], equality_ids=(2,))
                history.append(delete_reference(tables, name))
                if name == "delete_removed":
                    table = commit_deletes(tables, name, clear=True)
                else:
                    seq = table.metadata.last_sequence_number
                    table = commit_deletes(tables, name, [{2: None}], equality_ids=(2,), data_sequence=seq)
                    history.append(delete_reference(tables, name))
                    table = append(name, rows(27, 3))
                    history.append(delete_reference(tables, name))
                    # A committed empty delete file must be harmless while
                    # preserving the required-field and file-lifetime contract.
                    table = commit_deletes(tables, name, equality_ids=(4,))
        current = delete_reference(tables, name)
        for view in history:
            assert delete_reference(tables, name, snapshot_id=view.snapshot_id) == view
        count = sum(entry.data_file.content == 0 for _manifest, entry in manifest_entries(table))
        assert count > 2
        records = sum(
            entry.data_file.record_count for _manifest, entry in manifest_entries(table) if entry.data_file.content == 0
        )
        assert len(current.rows) < records or name == "delete_removed", (name, len(current.rows), records)
        features[name] = DeleteFeature(name, current, tuple(history), count)
    return features


@contextmanager
def delete_tables():
    import os

    import pytest
    from gravitino_iceberg_helpers import iceberg_tables

    endpoint = os.environ.get("VANE_TEST_GRAVITINO_ICEBERG_URL")
    shared = os.environ.get("VANE_TEST_GRAVITINO_ICEBERG_SHARED_DIR")
    if not endpoint and not shared:
        pytest.skip("set VANE_TEST_GRAVITINO_ICEBERG_URL and VANE_TEST_GRAVITINO_ICEBERG_SHARED_DIR")
    if not endpoint or not shared:
        pytest.fail("Delete qualification requires both REST URL and shared directory")
    with iceberg_tables(endpoint.rstrip("/"), Path(shared)) as tables:
        yield endpoint, tables, populate_delete_tables(tables)
