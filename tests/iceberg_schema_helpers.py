# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Flat schema history authored and reference-scanned by the independent SDK."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from iceberg_partition_helpers import append_partition_rows

SCHEMA_TABLES = (
    "schema_columns",
    "schema_promotions",
    "schema_identity",
    "schema_transformed",
    "schema_dropped_partition",
    "schema_metadata_only",
    "schema_defaults",
)


def schema_filter(name):
    import pyarrow as pa
    import pyarrow.compute as pc

    if name == "schema_columns":
        return "label IS NULL OR retired IS NOT NULL", pc.field("label").is_null() | ~pc.field("retired").is_null()
    if name in {"schema_promotions", "schema_identity", "schema_transformed"}:
        return "n < 0 OR amount > 1000000 OR value IS NULL", (pc.field("n") < 0) | (
            pc.field("amount") > pa.scalar(Decimal("1000000.00"), type=pa.decimal128(18, 2))
        ) | pc.field("value").is_null()
    if name == "schema_dropped_partition":
        return "region IS NULL OR region = 'new-10'", pc.field("region").is_null() | (pc.field("region") == "new-10")
    if name == "schema_defaults":
        return "note IS NULL AND rank IS NULL", pc.field("note").is_null() & pc.field("rank").is_null()
    return "note IS NULL AND renamed = 'old-1'", pc.field("note").is_null() & (pc.field("renamed") == "old-1")


@dataclass(frozen=True)
class SchemaView:
    names: tuple[str, ...]
    types: tuple[str, ...]
    field_ids: tuple[int, ...]
    rows: tuple[tuple[Any, ...], ...]
    schema_id: int
    snapshot_id: int


@dataclass(frozen=True)
class SchemaFeature:
    name: str
    current: SchemaView
    history: tuple[SchemaView, ...]
    file_count: int


def schema_reference(tables, name, *, snapshot_id=None, row_filter=None):
    from pyiceberg.types import DecimalType

    table = tables.catalog.load_table((tables.namespace, name))
    scan = table.scan(**({"snapshot_id": snapshot_id} if snapshot_id is not None else {}))
    schema = scan.projection()
    data = scan.to_arrow()
    if name == "schema_identity":
        # The pinned SDK rounds omitted FLOAT -> DOUBLE identity values to
        # float32. Use the separately authored physical-column control table
        # (same input rows/schema history) rather than accept that precision
        # loss or derive expectations from Vane's result.
        control = tables.catalog.load_table((tables.namespace, "schema_promotions"))
        snapshots = [
            snapshot
            for snapshot in control.snapshots()
            if snapshot.schema_id == schema.schema_id and int(snapshot.summary["total-records"]) == data.num_rows
        ]
        assert len(snapshots) == 1
        control_scan = control.scan(snapshot_id=snapshots[0].snapshot_id)
        assert tuple(field.field_id for field in control_scan.projection().fields) == tuple(
            field.field_id for field in schema.fields
        )
        physical = control_scan.to_arrow()
        assert physical.schema.names == data.schema.names
        assert sorted(physical.column("id").to_pylist()) == sorted(data.column("id").to_pylist())
        data = physical
    if row_filter is not None:
        data = data.filter(row_filter)
    rows = data.sort_by("id").to_pylist()
    names = tuple(field.name for field in schema.fields)
    types = {
        "long": "BIGINT",
        "int": "INTEGER",
        "float": "FLOAT",
        "double": "DOUBLE",
        "string": "VARCHAR",
        "boolean": "BOOLEAN",
    }
    native_types = tuple(
        f"DECIMAL({field.field_type.precision},{field.field_type.scale})"
        if isinstance(field.field_type, DecimalType)
        else types[str(field.field_type)]
        for field in schema.fields
    )
    selected = table.current_snapshot() if snapshot_id is None else table.snapshot_by_id(snapshot_id)
    return SchemaView(
        names,
        native_types,
        tuple(field.field_id for field in schema.fields),
        tuple(tuple(row[name] for name in names) for row in rows),
        schema.schema_id,
        selected.snapshot_id if selected else 0,
    )


def append_schema_stage(tables, name, rows, *, omit_identity=False):
    table = None
    for start in range(0, len(rows), 3):
        table = append_partition_rows(
            tables, name, rows[start : start + 3], omit_partition_columns=omit_identity, rename_partition_columns=True
        )
    assert table is not None
    return table


def evolve_after_freeze(tables):
    from pyiceberg.types import StringType

    table = tables.catalog.load_table((tables.namespace, "schema_columns"))
    with table.update_schema() as update:
        update.rename_column("renamed", "after_freeze")
        update.add_column("after", StringType())
    updated = append_schema_stage(
        tables,
        "schema_columns",
        [
            {
                "id": 800001,
                "retired": "new-retired",
                "after_freeze": "changed",
                "flag": True,
                "detail": "later",
                "label": "later-label",
                "after": "new-column",
            }
        ],
    )
    assert updated.scan().to_arrow().num_rows == 28
    return updated


def populate_schema_evolution(tables, *, names=SCHEMA_TABLES):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyiceberg.io.pyarrow import schema_to_pyarrow
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.transforms import BucketTransform, IdentityTransform, TruncateTransform
    from pyiceberg.types import (
        BooleanType,
        DecimalType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        NestedField,
        StringType,
    )

    if "schema_identity" in names and "schema_promotions" not in names:
        names = ("schema_promotions", *names)

    def create(name, schema, spec=None):
        directory = tables.directory / name
        for path in (directory, directory / "metadata", directory / "data"):
            path.mkdir(mode=0o777)
            path.chmod(0o777)
        return tables.catalog.create_table(
            (tables.namespace, name),
            schema=schema,
            partition_spec=spec or PartitionSpec(),
            location=directory.as_uri(),
            properties={"format-version": "2"},
        )

    def feature(name, history):
        table = tables.catalog.load_table((tables.namespace, name))
        current = schema_reference(tables, name)
        expected_counts = {
            "schema_columns": 27,
            "schema_promotions": 27,
            "schema_identity": 27,
            "schema_transformed": 27,
            "schema_dropped_partition": 18,
            "schema_metadata_only": 9,
            "schema_defaults": 18,
        }
        assert [row[current.names.index("id")] for row in current.rows] == list(range(expected_counts[name]))
        for view in history:
            assert schema_reference(tables, name, snapshot_id=view.snapshot_id) == view
        count = len(list(table.scan().plan_files()))
        assert count > 2
        orphan = {field.name: (999999 if field.name == "id" else None) for field in table.schema().fields}
        path = tables.directory / name / "region=deliberately-wrong" / "orphan.parquet"
        pq.write_table(pa.Table.from_pylist([orphan], schema=schema_to_pyarrow(table.schema())), path)
        return SchemaFeature(name, current, tuple(history), count)

    features = {}
    if "schema_columns" in names:
        name = "schema_columns"
        table = create(
            name,
            Schema(
                NestedField(1, "id", LongType(), required=True),
                NestedField(2, "label", StringType()),
                NestedField(3, "retired", StringType()),
                NestedField(4, "flag", BooleanType()),
            ),
        )
        table = append_schema_stage(
            tables,
            name,
            [
                {
                    "id": key,
                    "label": None if key % 3 == 0 else f"old-{key}.点",
                    "retired": f"removed-{key}",
                    "flag": key % 2 == 0,
                }
                for key in range(9)
            ],
        )
        history = [schema_reference(tables, name)]
        with table.update_schema() as update:
            update.rename_column("label", "renamed")
            update.delete_column("retired")
            update.add_column("added", StringType())
        with table.update_schema() as update:
            update.add_column("label", StringType())
            update.move_first("renamed")
        table = append_schema_stage(
            tables,
            name,
            [
                {
                    "id": key,
                    "renamed": f"middle-{key}",
                    "added": None if key % 3 == 0 else f"detail-{key}",
                    "label": f"reused-{key}",
                    "flag": key % 2 == 0,
                }
                for key in range(9, 18)
            ],
        )
        history.append(schema_reference(tables, name))
        with table.update_schema() as update:
            update.rename_column("added", "detail")
            update.add_column("retired", StringType())
            update.move_before("retired", "renamed")
            update.move_first("id")
        append_schema_stage(
            tables,
            name,
            [
                {
                    "id": key,
                    "renamed": f"latest-{key}",
                    "detail": f"detail-{key}",
                    "retired": f"new-retired-{key}",
                    "label": f"reused-{key}",
                    "flag": key % 2 == 0,
                }
                for key in range(18, 27)
            ],
        )
        result = feature(name, history)
        values = [dict(zip(result.current.names, row, strict=True)) for row in result.current.rows]
        assert [row["retired"] for row in values[:18]] == [None] * 18
        assert [row["label"] for row in values[:9]] == [None] * 9
        assert result.current.field_ids[result.current.names.index("label")] != 2
        assert result.current.field_ids[result.current.names.index("retired")] != 3
        features[name] = result

    for name in ("schema_promotions", "schema_identity", "schema_transformed"):
        if name not in names:
            continue
        fields = ()
        if name == "schema_identity":
            fields = (
                (2, IdentityTransform(), "n_identity"),
                (2, BucketTransform(4), "n_bucket"),
                (3, IdentityTransform(), "f_identity"),
                (4, IdentityTransform(), "amount_identity"),
            )
        elif name == "schema_transformed":
            fields = ((2, BucketTransform(4), "n_bucket"), (5, TruncateTransform(3), "label_prefix"))
        partition = PartitionSpec(
            *(
                PartitionField(source_id=source, field_id=1000 + index, transform=transform, name=label)
                for index, (source, transform, label) in enumerate(fields)
            )
        )
        table = create(
            name,
            Schema(
                NestedField(1, "id", LongType(), required=True),
                NestedField(2, "n", IntegerType()),
                NestedField(3, "f", FloatType()),
                NestedField(4, "amount", DecimalType(8, 2)),
                NestedField(5, "label", StringType()),
            ),
            partition,
        )
        table = append_schema_stage(
            tables,
            name,
            [
                {
                    "id": key,
                    "n": (-11, 19, None)[key % 3],
                    "f": (-1.5, 1.25, None)[key % 3],
                    "amount": (Decimal("-0.75"), Decimal("12.34"), None)[key % 3],
                    "label": ("北京上海", "😀abc", None)[key % 3],
                }
                for key in range(9)
            ],
            omit_identity=True,
        )
        history = [schema_reference(tables, name)]
        with table.update_schema() as update:
            update.update_column("n", field_type=LongType())
            update.update_column("f", field_type=DoubleType())
            update.update_column("amount", field_type=DecimalType(12, 2))
            update.rename_column("f", "value")
            update.rename_column("label", "renamed")
            update.move_first("amount")
        with table.update_schema() as update:
            update.add_column("label", StringType())
        table = append_schema_stage(
            tables,
            name,
            [
                {
                    "id": key,
                    "n": 2**31 + key if key % 3 else None,
                    "value": 16777217.25 + key / 4 if key % 3 else None,
                    "amount": Decimal("10000000.00") + Decimal(key) / 100 if key % 3 else None,
                    "renamed": f"kept-{key}",
                    "label": f"new-name-{key}",
                }
                for key in range(9, 18)
            ],
            omit_identity=True,
        )
        history.append(schema_reference(tables, name))
        with table.update_schema() as update:
            update.update_column("amount", field_type=DecimalType(18, 2))
            update.add_column("note", StringType())
            update.move_before("note", "renamed")
            update.move_first("id")
        append_schema_stage(
            tables,
            name,
            [
                {
                    "id": key,
                    "n": -(2**40) - key if key % 3 else None,
                    "value": -16777217.5 - key / 8 if key % 3 else None,
                    "amount": Decimal("1000000000000.01") + Decimal(key) / 100 if key % 3 else None,
                    "renamed": f"kept-{key}",
                    "label": f"new-name-{key}",
                    "note": None if key % 3 else "new-note",
                }
                for key in range(18, 27)
            ],
            omit_identity=True,
        )
        result = feature(name, history)
        assert dict(zip(result.current.names, result.current.types, strict=True))["amount"] == "DECIMAL(18,2)"
        assert result.current.rows[0][result.current.names.index("label")] is None
        features[name] = result

    if "schema_dropped_partition" in names:
        name = "schema_dropped_partition"
        table = create(
            name,
            Schema(
                NestedField(1, "id", LongType(), required=True),
                NestedField(2, "region", StringType()),
                NestedField(3, "label", StringType()),
            ),
            PartitionSpec(
                PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="region_identity")
            ),
        )
        table = append_schema_stage(
            tables,
            name,
            [{"id": key, "region": ("east.点", "west", None)[key % 3], "label": f"old-{key}"} for key in range(9)],
            omit_identity=True,
        )
        history = [schema_reference(tables, name)]
        with table.update_spec() as update:
            update.remove_field("region_identity")
        with table.update_schema() as update:
            update.delete_column("region")
        with table.update_schema() as update:
            update.add_column("region", StringType())
            update.move_first("region")
        append_schema_stage(
            tables, name, [{"id": key, "region": f"new-{key}", "label": f"new-label-{key}"} for key in range(9, 18)]
        )
        result = feature(name, history)
        region_index = result.current.names.index("region")
        assert result.current.field_ids[region_index] != 2
        assert [row[region_index] for row in result.current.rows[:9]] == [None] * 9
        features[name] = result

    for name in ("schema_metadata_only", "schema_defaults"):
        if name not in names:
            continue
        table = create(
            name, Schema(NestedField(1, "id", LongType(), required=True), NestedField(2, "label", StringType()))
        )
        table = append_schema_stage(tables, name, [{"id": key, "label": f"old-{key}"} for key in range(9)])
        history = [schema_reference(tables, name)]
        if name == "schema_metadata_only":
            with table.update_schema() as update:
                update.rename_column("label", "renamed")
                update.add_column("note", StringType())
                update.move_first("note")
            assert table.current_snapshot().snapshot_id == history[0].snapshot_id
        else:
            with table.update_schema() as update:
                update.add_column("note", StringType(), default_value=None)
                update.add_column("rank", LongType(), default_value=None)
                update.add_column("amount", DecimalType(10, 2), default_value=None)
            table = append_schema_stage(
                tables,
                name,
                [
                    {
                        "id": key,
                        "label": f"new-{key}",
                        "note": None if key % 3 == 0 else f"explicit-{key}",
                        "rank": None if key % 3 == 0 else 77,
                        "amount": None if key % 3 == 0 else Decimal("12.34"),
                    }
                    for key in range(9, 18)
                ],
            )
            current = schema_reference(tables, name)
            assert [row[current.names.index("note")] for row in current.rows[:9]] == [None] * 9
            assert current.rows[9][current.names.index("note")] is None
        features[name] = feature(name, history)

    table = create(
        "schema_empty", Schema(NestedField(1, "id", LongType(), required=True), NestedField(2, "label", StringType()))
    )
    with table.update_schema() as update:
        update.rename_column("label", "renamed")
        update.add_column("note", StringType())
        update.move_first("note")
    empty = schema_reference(tables, "schema_empty")
    assert empty.rows == () and empty.snapshot_id == 0
    features["schema_empty"] = SchemaFeature("schema_empty", empty, (), 0)
    return features


@contextmanager
def schema_evolution_tables():
    import os

    import pytest
    from gravitino_iceberg_helpers import iceberg_tables

    endpoint = os.environ.get("VANE_TEST_GRAVITINO_ICEBERG_URL")
    shared = os.environ.get("VANE_TEST_GRAVITINO_ICEBERG_SHARED_DIR")
    if not endpoint and not shared:
        pytest.skip("set VANE_TEST_GRAVITINO_ICEBERG_URL and VANE_TEST_GRAVITINO_ICEBERG_SHARED_DIR")
    if not endpoint or not shared:
        pytest.fail("Schema evolution qualification requires both REST URL and shared directory")
    with iceberg_tables(endpoint.rstrip("/"), Path(shared)) as tables:
        yield endpoint.rstrip("/"), tables, populate_schema_evolution(tables)
