# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Identity-partition fixtures committed by PyIceberg, independently of Vane."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from urllib.parse import unquote, urlsplit

from iceberg_feature_helpers import IcebergFeature


def append_partition_rows(tables, name, rows, *, omit_partition_columns, rename_partition_columns=False):
    """Write opaque paths and commit typed partition records through the SDK."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyiceberg.io.pyarrow import schema_to_pyarrow
    from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
    from pyiceberg.partitioning import partition_record_value
    from pyiceberg.schema import Schema
    from pyiceberg.typedef import Record

    table = tables.catalog.load_table((tables.namespace, name))
    spec, schema = table.spec(), table.schema()
    fields = [schema.find_field(field.source_id) for field in spec.fields]
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field.name] for field in fields)].append(row)
    physical = Schema(*(field for field in schema.fields if not omit_partition_columns or field not in fields))
    arrow_schema = schema_to_pyarrow(physical)
    directory = tables.directory / name / "region=deliberately-wrong"
    directory.mkdir(exist_ok=True, mode=0o777)
    directory.chmod(0o777)
    existing = len(list(directory.glob("*.parquet")))
    with table.transaction() as transaction:
        with transaction.update_snapshot().fast_append() as append:
            for ordinal, (values, batch) in enumerate(sorted(grouped.items(), key=lambda entry: repr(entry[0]))):
                path = directory / f"opaque-{existing + ordinal:04}.parquet"
                payload = pa.Table.from_pylist(
                    [{field.name: row[field.name] for field in physical.fields} for row in batch], schema=arrow_schema
                )
                if rename_partition_columns:
                    payload = payload.rename_columns(
                        ["physical_" + field.name if field in fields else field.name for field in physical.fields]
                    )
                pq.write_table(payload, path)
                partition = Record(
                    *(
                        partition_record_value(partition_field=field, value=value, schema=schema)
                        for field, value in zip(spec.fields, values, strict=True)
                    )
                )
                data_file = DataFile.from_args(
                    content=DataFileContent.DATA,
                    file_path=path.as_uri().replace("%3D", "="),
                    file_format=FileFormat.PARQUET,
                    partition=partition,
                    record_count=len(batch),
                    file_size_in_bytes=path.stat().st_size,
                )
                data_file.spec_id = spec.spec_id
                append.append_data_file(data_file)
    table.refresh()
    return table


def populate_identity_partitions(tables):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyiceberg.io.pyarrow import schema_to_pyarrow
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.transforms import BucketTransform, IdentityTransform
    from pyiceberg.types import DateType, DecimalType, LongType, NestedField, StringType, TimestampType

    schema = Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "label", StringType()),
        NestedField(3, "region", StringType()),
    )
    partition_spec = PartitionSpec(
        PartitionField(source_id=3, field_id=1000, transform=IdentityTransform(), name="partition_region")
    )

    def create(name, *, table_schema=schema, spec=partition_spec):
        directory = tables.directory / name
        for path in (directory, directory / "metadata", directory / "data"):
            path.mkdir(mode=0o777)
            path.chmod(0o777)
        return tables.catalog.create_table(
            (tables.namespace, name),
            schema=table_schema,
            partition_spec=spec,
            location=directory.as_uri(),
            properties={"format-version": "2"},
        )

    rows = tuple(
        (key, None if key % 7 == 0 else f"row-{key}.点", ("east.点", "west", None)[key % 3]) for key in range(27)
    )
    features = {}
    for name, omitted in (("partition_present", False), ("partition_omitted", True)):
        table = create(name)
        initial = None
        for start in range(0, len(rows), 9):
            table = append_partition_rows(
                tables,
                name,
                [dict(zip(("id", "label", "region"), row, strict=True)) for row in rows[start : start + 9]],
                omit_partition_columns=omitted,
                rename_partition_columns=not omitted,
            )
            if initial is None:
                initial = table.current_snapshot().snapshot_id
        tasks = list(table.scan().plan_files())
        assert len(tasks) == 9
        for task in tasks:
            path = Path(unquote(urlsplit(task.file.file_path).path))
            physical = pq.read_schema(path)
            physical_ids = {field.metadata[b"PARQUET:field_id"] for field in physical}
            assert (b"3" not in physical_ids) == omitted
            if not omitted:
                assert "physical_region" in physical.names and "region" not in physical.names
            assert physical.field("id").metadata[b"PARQUET:field_id"] == b"1"
        observed = table.scan().to_arrow().sort_by("id").to_pylist()
        assert tuple((row["id"], row["label"], row["region"]) for row in observed) == rows
        orphan = pa.Table.from_pylist(
            [{"id": 999999, "label": "uncommitted", "region": "east.点"}], schema=schema_to_pyarrow(schema)
        )
        pq.write_table(orphan, tables.directory / name / "region=deliberately-wrong" / "orphan.parquet")
        features[name] = IcebergFeature(name, rows, initial)

    empty = create("partition_empty")
    assert empty.current_snapshot() is None
    features["partition_empty"] = IcebergFeature("partition_empty", (), 0)

    evolved = create("partition_specs", spec=PartitionSpec())
    first = None
    for index, start in enumerate(range(0, len(rows), 9)):
        if index == 1:
            with evolved.update_spec() as update:
                update.add_field("region", IdentityTransform(), "partition_region")
        elif index == 2:
            with evolved.update_spec() as update:
                update.rename_field("partition_region", "renamed_partition_region")
        evolved = append_partition_rows(
            tables,
            "partition_specs",
            [dict(zip(("id", "label", "region"), row, strict=True)) for row in rows[start : start + 9]],
            omit_partition_columns=index != 0,
        )
        if first is None:
            first = evolved.current_snapshot().snapshot_id
    assert len(evolved.metadata.partition_specs) == 3
    assert len(list(evolved.scan().plan_files())) == 7
    observed = evolved.scan().to_arrow().sort_by("id").to_pylist()
    assert tuple((row["id"], row["label"], row["region"]) for row in observed) == rows
    features["partition_specs"] = IcebergFeature("partition_specs", rows, first)

    typed_schema = Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "day", DateType()),
        NestedField(3, "amount", DecimalType(10, 2)),
        NestedField(4, "moment", TimestampType()),
    )
    typed_spec = PartitionSpec(
        *(
            PartitionField(source_id=index, field_id=1000 + index, transform=IdentityTransform(), name=f"part_{index}")
            for index in (2, 3, 4)
        )
    )
    create("partition_typed", table_schema=typed_schema, spec=typed_spec)
    values = (
        (date(2024, 1, 2), Decimal("12.34"), datetime(2024, 1, 2, 3, 4, 5, 123456)),
        (date(1969, 12, 31), Decimal("-0.75"), datetime(1969, 12, 31, 23, 59, 59, 654321)),
        (None, None, None),
    )
    typed_rows = tuple((key, *values[key % 3]) for key in range(9))
    typed = append_partition_rows(
        tables,
        "partition_typed",
        [dict(zip(("id", "day", "amount", "moment"), row, strict=True)) for row in typed_rows],
        omit_partition_columns=True,
    )
    observed = typed.scan().to_arrow().sort_by("id").to_pylist()
    assert tuple((row["id"], row["day"], row["amount"], row["moment"]) for row in observed) == typed_rows
    features["partition_typed"] = IcebergFeature("partition_typed", typed_rows, typed.current_snapshot().snapshot_id)

    create(
        "partition_bucket",
        spec=PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=BucketTransform(4), name="bucket_id")),
    )
    return features


@contextmanager
def identity_partition_tables():
    import os

    import pytest
    from gravitino_iceberg_helpers import iceberg_tables

    endpoint = os.environ.get("VANE_TEST_GRAVITINO_ICEBERG_URL")
    shared = os.environ.get("VANE_TEST_GRAVITINO_ICEBERG_SHARED_DIR")
    if not endpoint and not shared:
        pytest.skip("set VANE_TEST_GRAVITINO_ICEBERG_URL and VANE_TEST_GRAVITINO_ICEBERG_SHARED_DIR")
    if not endpoint or not shared:
        pytest.fail("Identity partition qualification requires both REST URL and shared directory")
    with iceberg_tables(endpoint.rstrip("/"), Path(shared)) as tables:
        yield endpoint.rstrip("/"), tables, populate_identity_partitions(tables)
