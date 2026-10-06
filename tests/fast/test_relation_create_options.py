# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import vane


def test_create_preserves_qualified_catalog_target():
    con = vane.connect()
    con.execute("ATTACH ':memory:' AS target_catalog")

    con.sql("SELECT 1 AS id, 'north' AS region").create("target_catalog.main.created_table")

    assert con.sql("SELECT * FROM target_catalog.main.created_table").fetchall() == [(1, "north")]
    assert con.execute(
        "SELECT table_catalog FROM information_schema.tables WHERE table_name = 'created_table'"
    ).fetchall() == [("target_catalog",)]


def test_create_passes_structured_options_to_native_catalog():
    con = vane.connect()
    source = con.sql("SELECT 1 AS id, 'north' AS region")

    with pytest.raises(vane.CatalogException, match="PARTITIONED BY is not supported"):
        source.create(
            "partitioned_target",
            partition_by=["bucket(16, id)", vane.ColumnExpression("region")],
        )

    with pytest.raises(vane.CatalogException, match="WITH clause is not supported"):
        source.to_table(
            "property_target",
            properties={
                "location": "s3://warehouse/property_target",
                "format-version": 2,
                "enabled": vane.ConstantExpression(True),
            },
        )


def test_create_rejects_ray_backend(tmp_path):
    database = str(tmp_path / "native.duckdb")
    with vane.connect(database, backend="ray") as connection:
        source = connection.table_function("range", [1])
        with pytest.raises(vane.NotImplementedException, match="require backend='local'"):
            source.create("ray_target")
        assert connection.query_runtime.pool.workers == []
    with vane.connect(database) as connection:
        assert connection.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'ray_target'"
        ).fetchone() == (0,)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"properties": []}, "properties.*mapping"),
        ({"properties": {1: "value"}}, "property names must be strings"),
        ({"properties": {"": "value"}}, "property names must not be empty"),
        (
            {"properties": {"location": "first", "LOCATION": "second"}},
            "unique case-insensitively",
        ),
        ({"partition_by": "id"}, "partition_by.*sequence"),
        ({"partition_by": [1]}, "partition expressions must be Expression or str"),
        ({"partition_by": ["id, region"]}, "exactly one expression"),
    ],
)
def test_create_validates_structured_arguments(kwargs, message):
    con = vane.connect()

    with pytest.raises(vane.InvalidInputException, match=message):
        con.sql("SELECT 1 AS id, 'north' AS region").create("invalid_target", **kwargs)
