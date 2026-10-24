# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Feature fixtures authored with the pinned PyIceberg SDK, outside Vane.

Only test-owned namespaces are modified. Delete fixtures use the v2 delete
formats from https://iceberg.apache.org/spec/#delete-formats and commit through
the independent REST client, never through the Vane provider under test.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IcebergFeature:
    name: str
    rows: tuple[tuple[Any, ...], ...]
    initial_snapshot_id: int


def populate_iceberg_features(tables):
    import pyarrow as pa
    from pyiceberg.io.pyarrow import schema_to_pyarrow
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import IntegerType, LongType, NestedField, StringType

    def create(name, schema, *, partition_spec=None):
        directory = tables.directory / name
        for path in (directory, directory / "metadata", directory / "data"):
            path.mkdir(mode=0o777)
            path.chmod(0o777)
        options = {} if partition_spec is None else {"partition_spec": partition_spec}
        return tables.catalog.create_table(
            (tables.namespace, name),
            schema=schema,
            location=directory.as_uri(),
            properties={"format-version": "2"},
            **options,
        )

    def append(table, rows):
        table.append(pa.Table.from_pylist(rows, schema=schema_to_pyarrow(table.schema(), include_field_ids=False)))

    schema = Schema(NestedField(1, "id", LongType(), required=True), NestedField(2, "label", StringType()))
    partition_schema = Schema(*schema.fields, NestedField(3, "region", StringType()))
    partitioned = create(
        "partitioned",
        partition_schema,
        partition_spec=PartitionSpec(
            PartitionField(source_id=3, field_id=1000, transform=IdentityTransform(), name="region")
        ),
    )
    partition_rows = tuple((key, f"row-{key}", ("east.点", "west", None)[key % 3]) for key in range(18))
    append(partitioned, [dict(zip(("id", "label", "region"), row, strict=True)) for row in partition_rows])
    independent = partitioned.scan().to_arrow().sort_by("id").to_pylist()
    assert tuple((row["id"], row["label"], row["region"]) for row in independent) == partition_rows

    evolved = create(
        "evolved",
        Schema(
            NestedField(1, "id", IntegerType(), required=True),
            NestedField(2, "label", StringType()),
            NestedField(3, "removed", StringType()),
        ),
    )
    append(
        evolved, [{"id": 1, "label": "old-one", "removed": "drop-one"}, {"id": 2, "label": None, "removed": "drop-two"}]
    )
    initial_evolved = evolved.current_snapshot().snapshot_id
    with evolved.update_schema() as update:
        update.rename_column("label", "renamed")
        update.delete_column("removed")
        update.add_column("added", StringType())
        update.update_column("id", field_type=LongType())
    with evolved.update_schema() as update:
        update.add_column("label", StringType())
    append(evolved, [{"id": 3, "renamed": "new-three", "added": "added-three", "label": "reused-name"}])
    evolved_rows = ((1, "old-one", None, None), (2, None, None, None), (3, "new-three", "added-three", "reused-name"))
    independent = evolved.scan().to_arrow().sort_by("id").to_pylist()
    assert tuple((row["id"], row["renamed"], row["added"], row["label"]) for row in independent) == evolved_rows

    features = {
        "partitioned": IcebergFeature("partitioned", partition_rows, partitioned.current_snapshot().snapshot_id),
        "evolved": IcebergFeature("evolved", evolved_rows, initial_evolved),
    }
    for name, equality in (("position_deletes", False), ("equality_deletes", True)):
        table = create(name, schema)
        append(
            table,
            [{"id": 10, "label": "a"}, {"id": 11, "label": None}, {"id": 12, "label": "c"}, {"id": 13, "label": "d"}],
        )
        initial = table.current_snapshot().snapshot_id
        _commit_delete_file(tables.catalog, table, tables.directory / name, equality=equality)
        expected = ((10, "a"), (13, "d")) if equality else ((10, "a"), (12, "c"), (13, "d"))
        if equality:
            # These later rows have a higher sequence number than the equality
            # delete and must survive despite matching both delete values.
            append(table, [{"id": 20, "label": "c"}, {"id": 21, "label": None}])
            expected += ((20, "c"), (21, None))
        else:
            independent = table.scan().to_arrow().sort_by("id").to_pylist()
            assert tuple((row["id"], row["label"]) for row in independent) == expected
        features[name] = IcebergFeature(name, expected, initial)
    return features


def _commit_delete_file(catalog, table, directory, *, equality):
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

    class DeleteManifestWriter(ManifestWriterV2):
        def content(self):
            return ManifestContent.DELETES

        @property
        def _meta(self):
            return {**super()._meta, "content": "deletes"}

    parent = table.current_snapshot()
    data_files = list(table.scan().plan_files())
    assert len(data_files) == 1 and data_files[0].file.record_count == 4
    sequence = table.metadata.last_sequence_number + 1
    snapshot_id = uuid.uuid4().int >> 65
    if equality:
        delete_schema = Schema(NestedField(2, "label", StringType()))
        delete_rows = [{"label": None}, {"label": "c"}]
    else:
        # The spec requires a long position even though the pinned client's
        # POSITIONAL_DELETE_SCHEMA constant currently uses IntegerType.
        delete_schema = Schema(
            NestedField(2147483546, "file_path", StringType(), required=True),
            NestedField(2147483545, "pos", LongType(), required=True),
        )
        delete_rows = [{"file_path": data_files[0].file.file_path, "pos": 1}]
    path = directory / "data" / "delete.parquet"
    pq.write_table(pa.Table.from_pylist(delete_rows, schema=schema_to_pyarrow(delete_schema)), path)
    data_file = DataFile.from_args(
        content=DataFileContent.EQUALITY_DELETES if equality else DataFileContent.POSITION_DELETES,
        file_path=path.as_uri(),
        file_format=FileFormat.PARQUET,
        partition=Record(),
        record_count=len(delete_rows),
        file_size_in_bytes=path.stat().st_size,
        equality_ids=[2] if equality else None,
    )
    data_file.spec_id = table.spec().spec_id
    manifest_path = (directory / "metadata" / "delete-manifest.avro").as_uri()
    with DeleteManifestWriter(
        table.spec(), table.schema(), table.io.new_output(manifest_path), snapshot_id, "null"
    ) as writer:
        writer.add_entry(
            ManifestEntry.from_args(
                status=ManifestEntryStatus.ADDED,
                snapshot_id=snapshot_id,
                sequence_number=sequence,
                file_sequence_number=sequence,
                data_file=data_file,
            )
        )
    delete_manifest = writer.to_manifest_file()
    manifest_list_path = (directory / "metadata" / "delete-snapshot.avro").as_uri()
    with write_manifest_list(
        2, table.io.new_output(manifest_list_path), snapshot_id, parent.snapshot_id, sequence, "null"
    ) as listing:
        listing.add_manifests([delete_manifest, *parent.manifests(table.io)])
    snapshot = Snapshot(
        snapshot_id=snapshot_id,
        parent_snapshot_id=parent.snapshot_id,
        sequence_number=sequence,
        manifest_list=manifest_list_path,
        schema_id=table.schema().schema_id,
        summary=Summary(
            operation=Operation.DELETE,
            **{
                "total-records": "4",
                "total-data-files": "1",
                "total-delete-files": "1",
                "total-position-deletes": "0" if equality else "1",
                "total-equality-deletes": "2" if equality else "0",
            },
        ),
    )
    catalog.commit_table(
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
    manifests = table.current_snapshot().manifests(table.io)
    observed = [manifest for manifest in manifests if manifest.content == ManifestContent.DELETES]
    assert len(observed) == 1
    entry = observed[0].fetch_manifest_entry(table.io)[0]
    assert entry.data_file.content == data_file.content
    assert entry.data_file.equality_ids == ([2] if equality else None)
    assert entry.data_file.file_path == data_file.file_path
    assert entry.sequence_number == sequence
    assert entry.file_sequence_number == sequence
    assert pq.read_schema(path).equals(schema_to_pyarrow(delete_schema), check_metadata=True)
    assert pq.ParquetFile(path).read().to_pylist() == delete_rows
