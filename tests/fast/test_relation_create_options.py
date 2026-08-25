# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle

import pytest

import vane


def test_create_preserves_qualified_catalog_target(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    con = vane.connect()
    con.execute("ATTACH ':memory:' AS target_catalog")

    con.sql("SELECT 1 AS id, 'north' AS region").create("target_catalog.main.created_table")

    assert con.sql("SELECT * FROM target_catalog.main.created_table").fetchall() == [(1, "north")]
    assert con.execute(
        "SELECT table_catalog FROM information_schema.tables WHERE table_name = 'created_table'"
    ).fetchall() == [("target_catalog",)]


def test_create_passes_structured_options_to_native_catalog(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
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


@pytest.mark.parametrize("runner_value", [None, "", "ray"])
def test_create_options_dispatch_to_ray_without_local_execution(monkeypatch, runner_value):
    if runner_value is None:
        monkeypatch.delenv("VANE_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VANE_RUNNER", runner_value)

    captured = []
    logical_plans = []

    class CapturingRunner:
        def run_write(self, relation):
            captured.append(relation)
            logical_plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_write_relation(
                relation,
                "create-options-plan",
            )
            logical_plans.append(pickle.loads(pickle.dumps(logical_plan)))
            return {"ok": True}

    import vane.runners as runners

    monkeypatch.setattr(runners, "set_runner_ray", lambda *_args, **_kwargs: CapturingRunner())
    con = vane.connect()
    con.sql("SELECT 1 AS id, 'north' AS region").create(
        "ray_target",
        properties={"location": "s3://warehouse/ray_target"},
        partition_by=["bucket(16, id)"],
    )

    assert [relation.type for relation in captured] == ["CREATE_TABLE_RELATION"]
    assert logical_plans[0].idx() == "create-options-plan"
    assert logical_plans[0].to_physical_plan(con) is not None
    assert con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'ray_target'").fetchone() == (
        0,
    )


def test_create_rejects_local_fte_runner(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local")
    con = vane.connect()

    with pytest.raises(
        vane.InvalidInputException,
        match="CTAS requires VANE_RUNNER=ray or VANE_RUNNER=local-fast",
    ):
        con.sql("SELECT 1 AS id").create("local_fte_target", properties={"format-version": 2})

    assert con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'local_fte_target'"
    ).fetchone() == (0,)


def test_ray_create_rejects_explicit_transaction(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    calls = []

    class CapturingRunner:
        def run_write(self, relation):
            calls.append(relation)
            return {"ok": True}

    import vane.runners as runners

    monkeypatch.setattr(runners, "set_runner_ray", lambda *_args, **_kwargs: CapturingRunner())
    con = vane.connect()
    con.execute("BEGIN")
    try:
        with pytest.raises(
            vane.InvalidInputException,
            match="Ray CTAS requires DuckDB auto-commit mode",
        ):
            con.sql("SELECT 1 AS id").create("transaction_target")
        assert calls == []
    finally:
        con.execute("ROLLBACK")


def test_ray_create_failure_never_executes_locally(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    successful_calls = []

    class FailingRunner:
        def run_write(self, relation):
            raise RuntimeError("injected distributed CTAS failure")

    class CapturingRunner:
        def run_write(self, relation):
            successful_calls.append(relation)
            return {"ok": True}

    import vane.runners as runners

    monkeypatch.setattr(runners, "set_runner_ray", lambda *_args, **_kwargs: FailingRunner())
    con = vane.connect()

    with pytest.raises(RuntimeError, match="injected distributed CTAS failure"):
        con.sql("SELECT 1 AS id").create("failed_ray_target")

    assert con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'failed_ray_target'"
    ).fetchone() == (0,)

    monkeypatch.setattr(runners, "set_runner_ray", lambda *_args, **_kwargs: CapturingRunner())
    con.sql("SELECT 2 AS id").create("retry_ray_target")
    assert [relation.type for relation in successful_calls] == ["CREATE_TABLE_RELATION"]
    assert con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'retry_ray_target'"
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
def test_create_validates_structured_arguments(monkeypatch, kwargs, message):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    con = vane.connect()

    with pytest.raises(vane.InvalidInputException, match=message):
        con.sql("SELECT 1 AS id, 'north' AS region").create("invalid_target", **kwargs)
