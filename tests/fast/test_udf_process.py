# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa

import vane


def test_create_function_rejects_removed_process_and_ray_args():
    con = vane.connect()

    def add_one(value):
        return value + 1

    with pytest.raises(TypeError):
        con.create_function(
            "bad_process_arg",
            add_one,
            ["BIGINT"],
            "BIGINT",
            type="native",
            use_process=True,
        )

    with pytest.raises(TypeError):
        con.create_function(
            "bad_ray_arg",
            add_one,
            ["BIGINT"],
            "BIGINT",
            type="native",
            ray=True,
        )


def test_map_batches_rejects_removed_process_and_actor_count_args():
    con = vane.connect()

    def add_one(table):
        values = table.column(0).to_pylist()
        return pa.table({"out": [value + 1 for value in values]})

    rel = con.sql("select i from range(0, 4) t(i)")

    with pytest.raises(TypeError):
        rel.map_batches(
            add_one,
            schema={"out": vane.sqltypes.BIGINT},
            use_process=True,
        )

    with pytest.raises(TypeError):
        rel.map_batches(
            add_one,
            schema={"out": vane.sqltypes.BIGINT},
            actor_count=1,
        )


def test_ray_task_map_batches_local_execution_is_rejected(monkeypatch):
    monkeypatch.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    con = vane.connect()

    def add_ten(table):
        values = table.column(0).to_pylist()
        return pa.table({"out": [value + 10 for value in values]})

    relation = con.sql("select i from range(0, 5) t(i)").map_batches(
        add_ten,
        schema={"out": vane.sqltypes.BIGINT},
        execution_backend="ray_task",
        batch_size=2,
    )

    with pytest.raises(Exception, match="distributed Ray UDF payload requires query_id"):
        relation.fetchall()


def test_flat_map_rejects_removed_actor_count_arg():
    con = vane.connect()

    def expand(row):
        return [{"out": row["i"]}, {"out": row["i"] + 10}]

    with pytest.raises(TypeError):
        con.sql("select i from range(0, 2) t(i)").flat_map(
            expand,
            schema={"out": vane.sqltypes.BIGINT},
            actor_count=1,
        )
