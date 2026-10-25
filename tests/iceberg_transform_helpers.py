# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Transformed partition fixtures and reference scans authored without Vane."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from iceberg_feature_helpers import IcebergFeature
from iceberg_partition_helpers import append_partition_rows

TRANSFORM_TABLES = (
    "transform_year",
    "transform_month",
    "transform_day",
    "transform_hour",
    "transform_bucket",
    "transform_truncate",
    "transform_void",
    "transform_mixed",
    "transform_specs",
)
COLUMNS = ("id", "moment", "event_day", "account", "label", "amount", "region")


def transformed_rows():
    moments = (
        datetime(1969, 12, 31, 23, 59, 59, 654321),
        datetime(1970, 1, 1, 0, 0, 0, 123456),
        datetime(2024, 2, 29, 12, 34, 56, 123456),
        datetime(2024, 3, 1),
        datetime(1969, 12, 31, 0, 0, 0, 1),
        None,
        datetime(2025, 1, 1, 1, 59, 59, 999999),
        datetime(2024, 2, 29, 12, 0, 1, 111111),
        datetime(1970, 1, 1, 1),
    )
    accounts = (-11, -1, 0, 1, 19, None, 81, 17, -22)
    labels = ("北京上海", "北京.点", "alpha-x", "alpha-y", "éclair", "", None, "😀ab-z", "😀ab-x")
    amounts = tuple(
        Decimal(value) if value is not None else None
        for value in ("-12.34", "-0.75", "0", "12.34", "10.65", None, "100.01", "-10.01", "9.99")
    )
    return tuple(
        (
            key,
            moments[key % 9],
            moments[key % 9].date() if moments[key % 9] is not None else None,
            accounts[key % 9],
            labels[key % 9],
            amounts[key % 9],
            ("east.点", "west", None)[key % 3],
        )
        for key in range(27)
    )


def reference_rows(tables, name, *, snapshot_id=None, row_filter=None):
    table = tables.catalog.load_table((tables.namespace, name))
    options = {"snapshot_id": snapshot_id} if snapshot_id is not None else {}
    data = table.scan(**options).to_arrow()
    if row_filter is not None:
        # The pinned SDK's strict partition evaluator compares None with ints
        # for some nullable predicates. Evaluate independent Arrow predicates
        # on the committed SDK scan, retaining SQL NULL semantics.
        data = data.filter(row_filter)
    rows = data.sort_by("id").to_pylist()
    return tuple(tuple(row[column] for column in COLUMNS) for row in rows)


def populate_transformed_partitions(tables, *, names=TRANSFORM_TABLES):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyiceberg.io.pyarrow import schema_to_pyarrow
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.transforms import (
        BucketTransform,
        DayTransform,
        HourTransform,
        IdentityTransform,
        MonthTransform,
        TruncateTransform,
        VoidTransform,
        YearTransform,
    )
    from pyiceberg.types import DateType, DecimalType, LongType, NestedField, StringType, TimestampType

    schema = Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "moment", TimestampType()),
        NestedField(3, "event_day", DateType()),
        NestedField(4, "account", LongType()),
        NestedField(5, "label", StringType()),
        NestedField(6, "amount", DecimalType(10, 2)),
        NestedField(7, "region", StringType()),
    )

    def spec(*fields):
        return PartitionSpec(
            *(
                PartitionField(source_id=source, field_id=1000 + index, transform=transform, name=name)
                for index, (source, transform, name) in enumerate(fields)
            )
        )

    mixed = spec(
        (7, IdentityTransform(), "region_identity"),
        (2, DayTransform(), "event_partition"),
        (4, BucketTransform(4), "account_bucket"),
        (5, TruncateTransform(3), "label_prefix"),
        (7, BucketTransform(3), "region_bucket"),
    )
    specs = {
        "transform_year": spec((2, YearTransform(), "event_year")),
        "transform_month": spec((2, MonthTransform(), "event_month")),
        "transform_day": spec((2, DayTransform(), "event_date"), (3, DayTransform(), "date_partition")),
        "transform_hour": spec((2, HourTransform(), "event_hour")),
        "transform_bucket": spec(
            (4, BucketTransform(4), "account_bucket"),
            (5, BucketTransform(8), "label_bucket"),
            (6, BucketTransform(16), "amount_bucket"),
        ),
        "transform_truncate": spec(
            (4, TruncateTransform(10), "account_range"),
            (5, TruncateTransform(3), "label_prefix"),
            (6, TruncateTransform(50), "amount_range"),
        ),
        "transform_void": spec((3, VoidTransform(), "ignored_date")),
        "transform_mixed": mixed,
        "transform_specs": mixed,
        "transform_empty": spec((4, BucketTransform(8), "account_bucket")),
    }
    features = {}
    rows = transformed_rows()
    for name in (*names, "transform_empty"):
        directory = tables.directory / name
        for path in (directory, directory / "metadata", directory / "data"):
            path.mkdir(mode=0o777)
            path.chmod(0o777)
        table = tables.catalog.create_table(
            (tables.namespace, name),
            schema=schema,
            partition_spec=specs[name],
            location=directory.as_uri(),
            properties={"format-version": "2"},
        )
        if name == "transform_empty":
            assert reference_rows(tables, name) == ()
            features[name] = IcebergFeature(name, (), 0)
            continue
        first = None
        for index, start in enumerate(range(0, len(rows), 9)):
            if name == "transform_specs" and index:
                with table.update_spec() as update:
                    if index == 1:
                        update.remove_field("event_partition")
                        update.add_field("moment", HourTransform(), "event_hour")
                    else:
                        update.remove_field("account_bucket")
                        update.add_field("account", BucketTransform(8), "account_bucket_8")
                        update.rename_field("region_identity", "renamed_region")
            table = append_partition_rows(
                tables,
                name,
                [dict(zip(COLUMNS, row, strict=True)) for row in rows[start : start + 9]],
                omit_partition_columns=True,
                rename_partition_columns=True,
            )
            if first is None:
                first = table.current_snapshot().snapshot_id
        assert reference_rows(tables, name) == rows
        assert reference_rows(tables, name, snapshot_id=first) == rows[:9]
        tasks = list(table.scan().plan_files())
        assert len(tasks) > 2
        assert (len(table.metadata.partition_specs) == 3) == (name == "transform_specs")
        orphan = pa.Table.from_pylist(
            [dict(zip(COLUMNS, (999999, None, None, 999, "uncommitted", Decimal("0"), "orphan"), strict=True))],
            schema=schema_to_pyarrow(schema),
        )
        pq.write_table(orphan, directory / "region=deliberately-wrong" / "orphan.parquet")
        features[name] = IcebergFeature(name, rows, first)
    return features


@contextmanager
def transformed_partition_tables():
    import os

    import pytest
    from gravitino_iceberg_helpers import iceberg_tables

    endpoint = os.environ.get("VANE_TEST_GRAVITINO_ICEBERG_URL")
    shared = os.environ.get("VANE_TEST_GRAVITINO_ICEBERG_SHARED_DIR")
    if not endpoint and not shared:
        pytest.skip("set VANE_TEST_GRAVITINO_ICEBERG_URL and VANE_TEST_GRAVITINO_ICEBERG_SHARED_DIR")
    if not endpoint or not shared:
        pytest.fail("Transformed partition qualification requires both REST URL and shared directory")
    with iceberg_tables(endpoint.rstrip("/"), Path(shared)) as tables:
        yield endpoint.rstrip("/"), tables, populate_transformed_partitions(tables)
