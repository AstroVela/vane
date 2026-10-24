# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Native frozen reads of transformed partitions against independent SDK scans."""

from __future__ import annotations

import pickle
from datetime import datetime
from decimal import Decimal

import pytest
from gravitino_iceberg_helpers import iceberg_tables
from iceberg_partition_helpers import append_partition_rows
from iceberg_scan_helpers import RestForwarder, attach, connection, execute_frozen, freeze, unavailable_metadata
from iceberg_transform_helpers import (
    COLUMNS,
    TRANSFORM_TABLES,
    populate_transformed_partitions,
    reference_rows,
    transformed_partition_tables,
)

import vane
from vane.catalog import IcebergCatalog

pytestmark = pytest.mark.external_service


@pytest.fixture(scope="module")
def transform_tables():
    with transformed_partition_tables() as setup:
        yield setup


@pytest.fixture(autouse=True)
def native_only(monkeypatch):
    import ray

    monkeypatch.setenv("VANE_RUNNER", "local")
    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "1")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "1")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native transformed-partition qualification must not initialize Ray")

    monkeypatch.setattr(ray, "init", forbidden)


def transform_query(tables, name, query):
    import pyarrow.compute as pc

    table = f'lake."{tables.namespace}"."{name}"'
    rows = reference_rows(tables, name)
    if query == "rows":
        return f"SELECT * FROM {table}", list(rows)
    if query == "filtered":
        predicate = (
            (pc.field("account") < 0) | pc.field("label").is_null() | (pc.field("moment") >= datetime(2024, 3, 1))
        )
        expected = reference_rows(tables, name, row_filter=predicate)
        return f"SELECT * FROM {table} WHERE account < 0 OR label IS NULL OR moment >= TIMESTAMP '2024-03-01'", list(
            expected
        )
    if query == "projection":
        expected = reference_rows(tables, name, row_filter=(pc.field("id") >= 5) & (pc.field("id") <= 21))
        return f"SELECT label, amount, moment, region, id FROM {table} WHERE id BETWEEN 5 AND 21", [
            (row[4], row[5], row[1], row[6], row[0]) for row in expected
        ]
    if query == "aggregate":
        return f"SELECT count(*), count(moment), count(account), sum(id)::BIGINT FROM {table}", [
            (
                len(rows),
                sum(row[1] is not None for row in rows),
                sum(row[3] is not None for row in rows),
                sum(row[0] for row in rows),
            )
        ]
    snapshot = tables.catalog.load_table((tables.namespace, name)).metadata.snapshots[0].snapshot_id
    expected = reference_rows(tables, name, snapshot_id=snapshot)
    return f"SELECT * FROM {table} AT (VERSION => {snapshot})", list(expected)


@pytest.mark.parametrize("name", TRANSFORM_TABLES)
@pytest.mark.parametrize("query", ["rows", "filtered", "projection", "aggregate", "old_snapshot"])
def test_transformed_values_survive_frozen_execution_without_metadata(transform_tables, name, query):
    endpoint, tables, _features = transform_tables
    sql, expected = transform_query(tables, name, query)
    proxy = RestForwarder(endpoint)
    try:
        with connection(tables) as con:
            attach(con, proxy.endpoint)
            payload = freeze(con, sql)
        proxy.stop()
        with unavailable_metadata(tables):
            observed, _schema = execute_frozen(payload, tables)
        assert sorted(observed, key=repr) == sorted(expected, key=repr)
    finally:
        proxy.stop()


@pytest.mark.parametrize("name", ["transform_bucket", "transform_truncate", "transform_specs"])
def test_public_local_transformed_partition_relations(transform_tables, name):
    endpoint, tables, features = transform_tables
    with connection(tables) as con:
        vane.load_installed_extension("iceberg", connection=con)
        catalog = IcebergCatalog.attach(con, "lake", endpoint=endpoint)
        relation = catalog.table(tables.namespace, name)
        assert relation.order("id").fetchall() == list(features[name].rows)
        assert relation.filter("account < 0 AND region = 'east.点'").order("id").fetchall() == [
            row for row in features[name].rows if row[3] is not None and row[3] < 0 and row[6] == "east.点"
        ]
        assert relation.aggregate("count(*), sum(id)::BIGINT").fetchone() == (27, sum(range(27)))


def test_disjoint_assignments_preserve_mixed_historical_specs(transform_tables):
    endpoint, tables, features = transform_tables
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".transform_specs')
    with connection(tables) as worker:
        physical = pickle.loads(payload).to_physical_plan(worker)
        batches = [bytes(batch) for batch in next(iter(physical.scan_split_batch_map().values()))]
        assert len(batches) == 27
        physical = None
    combined = []
    with unavailable_metadata(tables):
        for selected in (batches[::2], batches[1::2]):
            rows, _schema = execute_frozen(
                payload, tables, split_batch=vane.ray_cxx.merge_scan_split_batches(list(reversed(selected)))
            )
            assert rows
            combined.extend(rows)
    assert sorted(combined) == list(features["transform_specs"].rows)


def test_empty_transformed_table_and_empty_assignment_preserve_schema(transform_tables):
    endpoint, tables, _features = transform_tables
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".transform_mixed')
        empty = freeze(con, f'SELECT * FROM lake."{tables.namespace}".transform_empty')
    with connection(tables) as worker:
        physical = pickle.loads(empty).to_physical_plan(worker)
        empty_batch = bytes(next(iter(physical.scan_split_batch_map().values()))[0])
        physical = None
    with unavailable_metadata(tables):
        for frozen, assignment in ((empty, None), (payload, empty_batch)):
            rows, schema = execute_frozen(frozen, tables, split_batch=assignment)
            assert rows == []
            assert schema["types"] == ["BIGINT", "TIMESTAMP", "DATE", "BIGINT", "VARCHAR", "DECIMAL(10,2)", "VARCHAR"]


def test_independent_append_does_not_change_frozen_transform_specs(transform_tables):
    endpoint, shared, _features = transform_tables
    with iceberg_tables(endpoint, shared.directory.parent) as tables:
        features = populate_transformed_partitions(tables, names=("transform_specs",))
        proxy = RestForwarder(endpoint)
        try:
            with connection(tables) as con:
                attach(con, proxy.endpoint)
                payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".transform_specs')
            updated = append_partition_rows(
                tables,
                "transform_specs",
                [
                    dict(
                        zip(
                            COLUMNS,
                            (800001, datetime(2026, 10, 24), None, -123, "later", Decimal("-1.25"), "new-region"),
                            strict=True,
                        )
                    )
                ],
                omit_partition_columns=True,
                rename_partition_columns=True,
            )
            assert updated.scan().to_arrow().num_rows == 28
            proxy.stop()
            with unavailable_metadata(tables):
                rows, _schema = execute_frozen(payload, tables)
            assert sorted(rows) == list(features["transform_specs"].rows)
        finally:
            proxy.stop()
