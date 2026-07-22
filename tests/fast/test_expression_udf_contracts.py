# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Cross-cutting expression UDF capability-boundary tests."""

from __future__ import annotations

import re
import warnings
from types import SimpleNamespace

import pytest


@pytest.fixture
def sql_udf_contract_connection():
    import pyarrow as pa

    import vane

    con = vane.connect()

    @vane.func(return_dtype="INTEGER")
    def scalar_contract(value):
        return value + 1

    def batch_contract(table):
        values = table.column("value").to_pylist()
        return pa.table({"result": [value + 1 for value in values]})

    @vane.cls(actor_number=1, return_dtype="INTEGER", name="stateful_contract")
    class StatefulContract:
        def __call__(self, value):
            return value

    vane.attach_function(
        scalar_contract,
        alias="scalar_contract",
        connection=con,
        parameters=["INTEGER"],
    )
    vane.attach_function(
        batch_contract,
        alias="batch_contract",
        connection=con,
        parameters=["INTEGER"],
        input_names=["value"],
        schema={"result": "INTEGER"},
    )
    vane.attach_function(
        StatefulContract(),
        alias="stateful_contract",
        connection=con,
        parameters=["INTEGER"],
    )
    try:
        yield con
    finally:
        con.close()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT i FROM range(3) t(i) WHERE scalar_contract(i::INTEGER) > 0",
        "SELECT sum(batch_contract(i::INTEGER)) FROM range(3) t(i)",
        "SELECT * FROM range(3) a(i) JOIN range(3) b(j) ON stateful_contract(a.i::INTEGER) = b.j",
        "SELECT ai_prompt(text), count(*) FROM (VALUES ('x')) t(text) GROUP BY ai_prompt(text)",
        "SELECT count(*) FROM (VALUES (1), (2)) t(i) HAVING scalar_contract(count(*)::INTEGER) > 0",
    ],
    ids=["where-scalar", "aggregate-batch", "join-stateful-class", "group-by-ai", "having-scalar"],
)
def test_expression_udfs_reject_non_projection_positions(sql_udf_contract_connection, sql):
    message = "udf can only be used in a projection and must be planned as a UDF operator"
    with pytest.raises(Exception, match=re.escape(message)):
        sql_udf_contract_connection.execute(sql)


@pytest.mark.parametrize(
    ("dtype", "arrow_type"),
    [
        ("BOOLEAN", "bool"),
        ("TINYINT", "int8"),
        ("UTINYINT", "uint8"),
        ("SMALLINT", "int16"),
        ("USMALLINT", "uint16"),
        ("INTEGER", "int32"),
        ("UINTEGER", "uint32"),
        ("BIGINT", "int64"),
        ("UBIGINT", "uint64"),
        ("FLOAT", "float"),
        ("DOUBLE", "double"),
        ("VARCHAR", "string"),
        ("BLOB", "binary"),
        ("DATE", "date32[day]"),
        ("TIMESTAMP", "timestamp[us]"),
        ("TIMESTAMP_NS", "timestamp[ns]"),
        ("TIMESTAMP_MS", "timestamp[ms]"),
        ("TIMESTAMP_S", "timestamp[s]"),
        ("DECIMAL(10,2)", "decimal128(10, 2)"),
        ("FLOAT[]", "list<item: float>"),
        ("FLOAT[4]", "fixed_size_list<item: float>[4]"),
    ],
)
def test_vane_cls_arrow_return_type_support_matrix(dtype, arrow_type):
    from vane._expression_udf import _dtype_to_arrow

    assert str(_dtype_to_arrow(dtype)) == arrow_type


@pytest.mark.parametrize(
    "dtype",
    [
        "TIME",
        "INTERVAL",
        "UUID",
        "TIMESTAMPTZ",
        "ENUM('red', 'green')",
        "STRUCT(value INTEGER)",
        "MAP(VARCHAR, INTEGER)",
    ],
)
def test_vane_cls_rejects_unsupported_arrow_return_types_with_original_type(dtype):
    import vane
    from vane._expression_udf import _dtype_to_arrow

    with pytest.raises(vane.InvalidInputException, match=re.escape(dtype)):
        _dtype_to_arrow(dtype)


@pytest.mark.parametrize(
    ("arrow_type", "duckdb_type"),
    [
        pytest.param("bool", "BOOLEAN", id="bool"),
        pytest.param("int8", "TINYINT", id="int8"),
        pytest.param("uint8", "UTINYINT", id="uint8"),
        pytest.param("int16", "SMALLINT", id="int16"),
        pytest.param("uint16", "USMALLINT", id="uint16"),
        pytest.param("int32", "INTEGER", id="int32"),
        pytest.param("uint32", "UINTEGER", id="uint32"),
        pytest.param("int64", "BIGINT", id="int64"),
        pytest.param("uint64", "UBIGINT", id="uint64"),
        pytest.param("float32", "FLOAT", id="float32"),
        pytest.param("float64", "DOUBLE", id="float64"),
        pytest.param("string", "VARCHAR", id="string"),
        pytest.param("binary", "BLOB", id="binary"),
        pytest.param("date32", "DATE", id="date32"),
        pytest.param("decimal", "DECIMAL(18,4)", id="decimal"),
        pytest.param("list", "BIGINT[]", id="list"),
        pytest.param("fixed_list", "BIGINT[3]", id="fixed-list"),
        pytest.param("nested_list", "INTEGER[][]", id="nested-list"),
        pytest.param("nested_fixed_list", "INTEGER[][2]", id="nested-fixed-list"),
        pytest.param("timestamp_us", "TIMESTAMP", id="timestamp-us"),
        pytest.param("timestamp_ns", "TIMESTAMP_NS", id="timestamp-ns"),
        pytest.param("timestamp_ms", "TIMESTAMP_MS", id="timestamp-ms"),
        pytest.param("timestamp_s", "TIMESTAMP_S", id="timestamp-s"),
    ],
)
def test_pyarrow_datatype_canonicalization_matrix(arrow_type, duckdb_type):
    import pyarrow as pa

    from vane._expression_udf import _canonicalize_dtype

    types = {
        "bool": pa.bool_(),
        "int8": pa.int8(),
        "uint8": pa.uint8(),
        "int16": pa.int16(),
        "uint16": pa.uint16(),
        "int32": pa.int32(),
        "uint32": pa.uint32(),
        "int64": pa.int64(),
        "uint64": pa.uint64(),
        "float32": pa.float32(),
        "float64": pa.float64(),
        "string": pa.string(),
        "binary": pa.binary(),
        "date32": pa.date32(),
        "decimal": pa.decimal128(18, 4),
        "list": pa.list_(pa.int64()),
        "fixed_list": pa.list_(pa.int64(), 3),
        "nested_list": pa.list_(pa.list_(pa.int32())),
        "nested_fixed_list": pa.list_(pa.list_(pa.int32()), 2),
        "timestamp_us": pa.timestamp("us"),
        "timestamp_ns": pa.timestamp("ns"),
        "timestamp_ms": pa.timestamp("ms"),
        "timestamp_s": pa.timestamp("s"),
    }
    expected_arrow = types[arrow_type]

    normalized_duckdb, normalized_arrow = _canonicalize_dtype(expected_arrow)

    assert str(normalized_duckdb) == duckdb_type
    assert normalized_arrow == expected_arrow


@pytest.mark.parametrize(
    "unsupported",
    [
        pytest.param("struct", id="struct"),
        pytest.param("map", id="map"),
        pytest.param("duration", id="duration"),
        pytest.param("dictionary", id="dictionary"),
        pytest.param("timezone", id="timezone-aware-timestamp"),
    ],
)
def test_unsupported_pyarrow_datatype_matrix_is_rejected_during_canonicalization(unsupported):
    import pyarrow as pa

    import vane
    from vane._expression_udf import _canonicalize_dtype

    dtype = {
        "struct": pa.struct([("value", pa.int64())]),
        "map": pa.map_(pa.string(), pa.int64()),
        "duration": pa.duration("us"),
        "dictionary": pa.dictionary(pa.int8(), pa.string()),
        "timezone": pa.timestamp("us", tz="UTC"),
    }[unsupported]

    with pytest.raises(vane.InvalidInputException) as exc_info:
        _canonicalize_dtype(dtype)

    assert str(dtype) in str(exc_info.value)
    assert "not supported" in str(exc_info.value) or "TIMESTAMPTZ" in str(exc_info.value)


@pytest.mark.parametrize(
    ("decorator_runner", "resolved_runner", "gpus", "expected_backend"),
    [
        pytest.param("local", "ray", 0.75, "ray_actor", id="local-to-ray-with-gpu"),
        pytest.param("local", "local", 0, "subprocess_actor", id="local-to-local-without-gpu"),
        pytest.param("ray", "ray", 0.75, "ray_actor", id="ray-to-ray-with-gpu"),
    ],
)
def test_actor_gpu_reservation_follows_resolved_backend(
    monkeypatch,
    decorator_runner,
    resolved_runner,
    gpus,
    expected_backend,
):
    import pyarrow as pa
    import ray

    import vane
    import vane.execution.udf_ray as udf_ray
    import vane.execution.udf_subprocess as udf_subprocess

    monkeypatch.setenv("VANE_RUNNER", decorator_runner)

    class IdentityBatch:
        def __call__(self, table: pa.Table) -> pa.Table:
            return pa.table({"result": table.column("value")})

    # Defining the decorator is configuration-only: it must not inspect the
    # runner or warn, even when warnings are promoted to exceptions.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DecoratedBatch = vane.cls.batch(
            actor_number=1,
            schema={"result": "INTEGER"},
            row_preserving=True,
            gpus=gpus,
        )(IdentityBatch)

    monkeypatch.setenv("VANE_RUNNER", resolved_runner)
    con = vane.connect()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            relation = con.sql("SELECT 1::INTEGER AS value").select(
                DecoratedBatch()(value=vane.col("value")).alias("result")
            )
            plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
                relation,
                f"gpu-order-{decorator_runner}-{resolved_runner}-{gpus}",
            ).to_physical_plan(con)
            nodes = plan.collect_udf_nodes(conn=con)
    finally:
        con.close()

    assert len(nodes) == 1
    payload = nodes[0]["payload"]
    assert payload["execution_backend"] == expected_backend
    assert payload["gpus"] == gpus
    assert payload["actor_number"] == 1
    assert payload["stateful"] is True

    if expected_backend == "subprocess_actor":
        created = []

        class FakeLocalPool:
            def __init__(self, payload, pool_size, *, name):
                created.append((dict(payload), pool_size, name))

            def shutdown(self, *, kill=False):
                return None

        monkeypatch.setattr(udf_subprocess, "LocalSubprocessActorPool", FakeLocalPool)
        pools, _ = udf_subprocess.ensure_local_subprocess_actor_pools_for_nodes(
            nodes,
            plan_identity=f"gpu-order-{decorator_runner}-{resolved_runner}",
        )
        assert len(pools) == 1
        assert len(created) == 1
        assert created[0][0] == payload
        return

    ray_pool_calls = []

    class FakeRayPool:
        def __init__(
            self,
            *,
            payload,
            concurrency,
            gpus_per_actor,
            actor_node_ids,
            ray_options=None,
            max_restarts=None,
            max_task_retries=None,
        ):
            ray_pool_calls.append(
                {
                    "payload": dict(payload),
                    "concurrency": concurrency,
                    "gpus_per_actor": gpus_per_actor,
                    "actor_node_ids": list(actor_node_ids),
                    "ray_options": ray_options,
                    "max_restarts": max_restarts,
                    "max_task_retries": max_task_retries,
                }
            )
            self.actors = [SimpleNamespace() for _ in range(concurrency)]
            self._init_refs = []
            self._confirmed_ready = set(range(concurrency))

        def shutdown(self):
            return None

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(udf_ray, "_is_vane_worker_process", lambda: False)
    monkeypatch.setattr(udf_ray, "UDFActorPool", FakeRayPool)
    stage_id = f"stage:test:gpu-order:{decorator_runner}:{resolved_runner}"
    payload["stage_id"] = stage_id

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        pools, _ = udf_ray.ensure_actor_pools_for_nodes(
            nodes,
            actor_node_ids_by_stage={stage_id: ("node-a",)},
        )

    assert len(pools) == 1
    assert len(ray_pool_calls) == 1
    assert ray_pool_calls[0]["concurrency"] == 1
    assert ray_pool_calls[0]["gpus_per_actor"] == gpus
    assert ray_pool_calls[0]["actor_node_ids"] == ["node-a"]
    assert ray_pool_calls[0]["max_restarts"] == 0
    assert ray_pool_calls[0]["max_task_retries"] == 0


def test_actor_gpu_is_rejected_when_resolved_backend_is_local(monkeypatch):
    import pyarrow as pa

    import vane

    monkeypatch.setenv("VANE_RUNNER", "ray")

    @vane.cls.batch(
        actor_number=1,
        schema={"result": "INTEGER"},
        row_preserving=True,
        gpus=0.75,
    )
    class IdentityBatch:
        def __call__(self, table: pa.Table) -> pa.Table:
            return pa.table({"result": table.column("value")})

    monkeypatch.setenv("VANE_RUNNER", "local")
    with pytest.raises(vane.InvalidInputException, match="GPU resources require a Ray UDF backend"):
        IdentityBatch()(value=vane.col("value"))


def test_stateless_ray_actor_pool_size_and_gpu_options_follow_physical_payload(monkeypatch):
    import ray

    import vane
    import vane.execution.udf_ray as udf_ray
    from vane._expression_udf import _build_actor_map_batches_expression

    class StatelessActor:
        def __call__(self, table):
            return table

    monkeypatch.setenv("VANE_RUNNER", "ray")
    con = vane.connect()
    try:
        relation = con.sql("SELECT i::INTEGER AS value FROM range(3) t(i)")
        expression = _build_actor_map_batches_expression(
            StatelessActor,
            name="stateless_three_actor_contract",
            inputs={"value": vane.col("value")},
            schema={"result": "INTEGER"},
            batch_size=2,
            row_preserving=True,
            actor_number=3,
            gpus=1.25,
        )
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation.select(expression.alias("result")),
            "stateless-three-actor-contract",
        ).to_physical_plan(con)
        nodes = plan.collect_udf_nodes(conn=con)
    finally:
        con.close()

    assert len(nodes) == 1
    assert nodes[0]["actor_pool_size"] == 3
    assert nodes[0]["payload"]["actor_number"] == 3
    assert not nodes[0]["payload"].get("stateful", False)

    calls = []

    class FakeRayPool:
        def __init__(
            self,
            *,
            payload,
            concurrency,
            gpus_per_actor,
            actor_node_ids,
            ray_options=None,
        ):
            calls.append(
                {
                    "payload": dict(payload),
                    "concurrency": concurrency,
                    "gpus_per_actor": gpus_per_actor,
                    "actor_node_ids": list(actor_node_ids),
                    "ray_options": ray_options,
                }
            )
            self.actors = [SimpleNamespace() for _ in range(concurrency)]
            self._init_refs = []
            self._confirmed_ready = set(range(concurrency))

        def shutdown(self):
            return None

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(udf_ray, "_is_vane_worker_process", lambda: False)
    monkeypatch.setattr(udf_ray, "UDFActorPool", FakeRayPool)

    stage_id = "stage:test:stateless-three-actor"
    nodes[0]["payload"]["stage_id"] = stage_id
    pools, _ = udf_ray.ensure_actor_pools_for_nodes(
        nodes,
        actor_node_ids_by_stage={stage_id: ("node-a", "node-a", "node-a")},
    )

    assert len(pools) == 1
    assert calls[0]["concurrency"] == 3
    assert calls[0]["gpus_per_actor"] == 1.25
    assert calls[0]["actor_node_ids"] == ["node-a", "node-a", "node-a"]
