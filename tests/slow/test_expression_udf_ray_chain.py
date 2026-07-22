# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Slow Ray integration tests for expression batch UDF chains."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("ray")


def test_row_preserving_ray_actor_materialized_submit_preserves_passthrough_columns():
    script = r"""
import pyarrow as pa
import ray
import vane
from vane import runners as _runners

vane.configure(runner="ray")


@vane.cls(actor_number=1, return_dtype="VARCHAR", name="upper_actor", gpus=0)
class UpperActor:
    def __call__(self, text):
        return text.upper()


ray.init(address="local", num_cpus=2, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
_runners.set_runner_ray(noop_if_initialized=True)
runner = _runners.get_or_create_runner()
con = vane.connect()

try:
    source = con.sql("SELECT * FROM (VALUES (1, 'alpha'), (2, 'beta')) t(id, text)")
    result = source.select(
        vane.col("id"),
        vane.col("text"),
        UpperActor()(vane.col("text")).alias("upper_text"),
    )

    parts = list(runner.run_iter_tables(result, results_buffer_size=1))
    table = pa.concat_tables(parts).rename_columns(list(result.columns))
    rows = sorted(table.to_pylist(), key=lambda row: row["id"])
    assert rows == [
        {"id": 1, "text": "alpha", "upper_text": "ALPHA"},
        {"id": 2, "text": "beta", "upper_text": "BETA"},
    ]
    print("RAY_ROW_PRESERVING_MATERIALIZED_SUBMIT_OK")
finally:
    runner.close()
    con.close()
    ray.shutdown()
"""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["VANE_PROGRESS"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "RAY_ROW_PRESERVING_MATERIALIZED_SUBMIT_OK" in result.stdout


def test_expected_direct_output_types_preserve_block_stream_actor_chain():
    script = r"""
import tempfile
from pathlib import Path

import vane
import pyarrow as pa
import ray
import vane
from vane import runners as _runners

vane.configure(runner="ray")


def collect_table(runner, relation):
    parts = list(runner.run_iter_tables(relation, results_buffer_size=1))
    tables = [part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts]
    return pa.concat_tables(tables)

def build_upstream_output(table):
    ints = table.column("base_int").to_pylist()
    texts = table.column("base_varchar").to_pylist()
    return pa.table(
        {
            "upstream_output": pa.array(
                [value + len(text) for value, text in zip(ints, texts, strict=True)],
                type=pa.int64(),
            )
        }
    )

@vane.cls.batch(
    actor_number=1,
    schema={"downstream_output": "BIGINT"},
    row_preserving=True,
    gpus=0,
)
class BuildDownstreamOutput:
    def __call__(self, table):
        values = table.column("upstream_output").to_pylist()
        return pa.table(
            {
                "downstream_output": pa.array(
                    [value * 2 for value in values],
                    type=pa.int64(),
                )
            }
        )

ray.init(address="local", num_cpus=4, include_dashboard=False, ignore_reinit_error=True)
_runners.set_runner_ray(noop_if_initialized=True)
runner = _runners.get_or_create_runner()
con = None

try:
    with tempfile.TemporaryDirectory() as tmp_dir:
        con = vane.connect()
        parquet_path = Path(tmp_dir) / "batch_chain.parquet"
        con.execute(
            "COPY ("
            "SELECT i::INTEGER AS base_int, "
            "       ('value-' || i::VARCHAR)::VARCHAR AS base_varchar "
            "FROM range(4097) t(i)"
            ") "
            f"TO '{parquet_path}' (FORMAT PARQUET)"
        )
        rel = con.sql(
            f"SELECT base_int, base_varchar FROM read_parquet('{parquet_path}')"
        )
        upstream_output = vane.func.batch(
            build_upstream_output,
            inputs={
                "base_int": vane.col("base_int"),
                "base_varchar": vane.col("base_varchar"),
            },
            schema={"upstream_output": "BIGINT"},
            batch_size=1024,
            row_preserving=True,
        )
        downstream = BuildDownstreamOutput()
        downstream_output = downstream(
            upstream_output=upstream_output
        ).alias("downstream_output")
        out = rel.select(
            vane.col("base_int"),
            vane.col("base_varchar"),
            upstream_output.alias("upstream_output"),
            downstream_output,
        )
        plan = out.explain()
        print("PLAN", plan)
        assert "STREAMING_UDF" in plan
        assert "direct_block_metadata_pair" in plan
        assert "ray_actor" in plan
        logical_columns = list(out.columns)
        assert logical_columns == [
            "base_int",
            "base_varchar",
            "upstream_output",
            "downstream_output",
        ]
        assert [str(dtype) for dtype in out.types] == [
            "INTEGER",
            "VARCHAR",
            "BIGINT",
            "BIGINT",
        ]

        result = collect_table(runner, out)
        print("RESULT_SCHEMA", result.schema)
        print("RESULT_ROWS", result.num_rows)
        assert result.column_names == ["c0", "c1", "c2", "c3"]
        assert result.schema.types[0] == pa.int32()
        assert pa.types.is_string(result.schema.types[1]) or pa.types.is_large_string(
            result.schema.types[1]
        )
        assert result.schema.types[2:] == [pa.int64(), pa.int64()]
        named_result = result.rename_columns(logical_columns)
        rows = sorted(named_result.to_pylist(), key=lambda row: row["base_int"])
        assert len(rows) == 4097
        for value, row in enumerate(rows):
            text = f"value-{value}"
            upstream_value = value + len(text)
            assert row == {
                "base_int": value,
                "base_varchar": text,
                "upstream_output": upstream_value,
                "downstream_output": upstream_value * 2,
            }

        print("EXPECTED_DIRECT_OUTPUT_TYPES_OK", result.num_rows)
finally:
    runner.close()
    if con is not None:
        con.close()
    ray.shutdown()
"""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXPECTED_DIRECT_OUTPUT_TYPES_OK 4097" in result.stdout


def test_vane_cls_python_and_sql_actor_paths_preserve_state_on_ray():
    script = r"""
import tempfile
from pathlib import Path

import vane
import pyarrow as pa
import ray
import vane
from collections import Counter
from vane import runners as _runners

vane.configure(runner="ray")

@vane.func(return_dtype="INTEGER", name="scalar_plus_one")
def ScalarPlusOne(value):
    return value + 1

def BatchPlusTwo(table):
    values = table.column("value").to_pylist()
    return pa.table({"batch_result": [value + 2 for value in values]})

@vane.cls(actor_number=1, return_dtype="INTEGER", name="class_plus_three")
class ClassPlusThree:
    def __call__(self, value):
        return value + 3

@vane.cls.batch(
    actor_number=1,
    schema={"batch_call": "INTEGER"},
    name="stateful_batch_counter",
    row_preserving=True,
)
class StatefulBatchCounter:
    def __init__(self):
        self.calls = 0

    def __call__(self, table):
        self.calls += 1
        return pa.table({"batch_call": [self.calls] * table.num_rows})


def collect_table(runner, relation):
    parts = list(runner.run_iter_tables(relation, results_buffer_size=1))
    tables = [part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts]
    return pa.concat_tables(tables)


ray.init(address="local", num_cpus=4, include_dashboard=False, ignore_reinit_error=True)
_runners.set_runner_ray(noop_if_initialized=True)
runner = _runners.get_or_create_runner()
con = None

try:
    with tempfile.TemporaryDirectory() as tmp_dir:
        con = vane.connect()
        parquet_path = Path(tmp_dir) / "four_udf_kinds.parquet"
        con.execute(
            f"COPY (SELECT i::INTEGER AS value FROM range(4097) t(i)) "
            f"TO '{parquet_path}' (FORMAT PARQUET)"
        )

        vane.attach_function(
            ScalarPlusOne,
            connection=con,
            alias="scalar_plus_one_sql",
            parameters=["INTEGER"],
        )
        vane.attach_function(
            BatchPlusTwo,
            connection=con,
            alias="batch_plus_two_sql",
            input_names=["value"],
            parameters=["INTEGER"],
            schema={"batch_result": "INTEGER"},
        )
        vane.attach_function(
            ClassPlusThree(),
            connection=con,
            alias="class_plus_three_sql",
            parameters=["INTEGER"],
        )
        vane.attach_function(
            StatefulBatchCounter(),
            connection=con,
            alias="stateful_batch_counter_sql",
            input_names=["value"],
            parameters=["INTEGER"],
        )

        source = f"read_parquet('{parquet_path}')"
        scalar = collect_table(
            runner,
            con.sql(f"SELECT value, scalar_plus_one_sql(value) AS result FROM {source}"),
        )
        batch = collect_table(
            runner,
            con.sql(f"SELECT value, batch_plus_two_sql(value) AS result FROM {source}"),
        )
        row_class = collect_table(
            runner,
            con.sql(f"SELECT value, class_plus_three_sql(value) AS result FROM {source}"),
        )
        batch_class = collect_table(
            runner,
            con.sql(
                f"SELECT value, stateful_batch_counter_sql(value) AS batch_call FROM {source}"
            ),
        )

        scalar_rows = sorted(
            zip(scalar.column(0).to_pylist(), scalar.column(1).to_pylist())
        )
        batch_rows = sorted(
            zip(batch.column(0).to_pylist(), batch.column(1).to_pylist())
        )
        class_rows = sorted(
            zip(row_class.column(0).to_pylist(), row_class.column(1).to_pylist())
        )
        batch_call_counts = Counter(batch_class.column(1).to_pylist())

        assert scalar_rows == [(value, value + 1) for value in range(4097)]
        assert batch_rows == [(value, value + 2) for value in range(4097)]
        assert class_rows == [(value, value + 3) for value in range(4097)]
        assert sorted(batch_call_counts) == [1, 2, 3]
        assert sorted(batch_call_counts.values()) == [1, 2048, 2048]
        print("RAY_UDF_KINDS scalar batch vane.cls vane.cls.batch")
        print("STATEFUL_BATCH_CALLS", sorted(batch_call_counts))
        print("STATEFUL_BATCH_ROWS", sorted(batch_call_counts.values()))
finally:
    runner.close()
    if con is not None:
        con.close()
    ray.shutdown()
"""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "RAY_UDF_KINDS scalar batch vane.cls vane.cls.batch" in result.stdout
    assert "STATEFUL_BATCH_CALLS [1, 2, 3]" in result.stdout
    assert "STATEFUL_BATCH_ROWS [1, 2048, 2048]" in result.stdout


def test_ai_sql_mock_provider_options_execute_on_multi_actor_ray_pool():
    script = r"""
import pickle
import tempfile
import uuid
from pathlib import Path

import vane
import numpy as np
import pyarrow as pa
import ray
import vane
from vane import runners as _runners
from tests.ai.test_expression_ai_sql import MockProvider
from vane.ai import provider as provider_registry

provider_registry.PROVIDERS["mock_ai_sql"] = lambda name=None, **options: MockProvider()
vane.configure(runner="ray")


def collect_table(runner, relation):
    parts = list(runner.run_iter_tables(relation, results_buffer_size=1))
    tables = [part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts]
    return pa.concat_tables(tables)


def assert_actor_pool_size(relation, expected):
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, str(uuid.uuid4()))
    restored = pickle.loads(pickle.dumps(logical))
    target = vane.connect()
    try:
        physical = restored.to_physical_plan(target)
        nodes = physical.collect_udf_nodes()
        assert len(nodes) == 1
        assert nodes[0]["payload"]["actor_number"] == expected
        assert nodes[0]["actor_pool_size"] == expected
    finally:
        target.close()


ray.init(address="local", num_cpus=6, include_dashboard=False, ignore_reinit_error=True)
_runners.set_runner_ray(noop_if_initialized=True)
runner = _runners.get_or_create_runner()
con = None

try:
    with tempfile.TemporaryDirectory() as tmp_dir:
        con = vane.connect()
        parquet_path = Path(tmp_dir) / "ai_options.parquet"
        con.execute(
            f"COPY (SELECT value::VARCHAR AS chunk FROM range(4) t(value)) "
            f"TO '{parquet_path}' (FORMAT PARQUET)"
        )
        source = f"read_parquet('{parquet_path}')"
        prompt_relation = con.sql(f'''
            SELECT chunk, ai_prompt(
                chunk,
                struct_pack(
                    provider := 'mock_ai_sql',
                    model := 'ray-model',
                    concurrency := 3,
                    batch_size := 2
                )
            ) AS response
            FROM {source}
        ''')
        embed_relation = con.sql(f'''
            SELECT chunk, ai_embed(
                chunk,
                struct_pack(
                    provider := 'mock_ai_sql',
                    model := 'ray-embedding-model',
                    dimensions := 5,
                    normalize := true,
                    concurrency := 3,
                    batch_size := 2
                )
            ) AS embedding
            FROM {source}
        ''')

        assert_actor_pool_size(prompt_relation, 3)
        assert_actor_pool_size(embed_relation, 3)
        prompt = collect_table(runner, prompt_relation)
        embedding = collect_table(runner, embed_relation)

        prompt_rows = sorted(zip(prompt.column(0).to_pylist(), prompt.column(1).to_pylist()))
        embedding_rows = sorted(zip(embedding.column(0).to_pylist(), embedding.column(1).to_pylist()))
        assert prompt_rows == [(str(value), f"ray-model:{value}") for value in range(4)]
        assert [len(vector) for _, vector in embedding_rows] == [5, 5, 5, 5]
        assert all(abs(float(np.linalg.norm(vector)) - 1.0) < 1e-6 for _, vector in embedding_rows)
        print("AI_RAY_OPTIONS provider=mock_ai_sql model=ray-model dimensions=5 concurrency=3")
        print("AI_RAY_ACTOR_POOL 3")
finally:
    runner.close()
    if con is not None:
        con.close()
    ray.shutdown()
"""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=150,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "AI_RAY_OPTIONS provider=mock_ai_sql model=ray-model dimensions=5 concurrency=3" in result.stdout
    assert "AI_RAY_ACTOR_POOL 3" in result.stdout


def test_batch_preprocessing_feeds_ai_embed_after_fresh_driver_round_trip_on_ray():
    script = r"""
import pickle
import tempfile
import uuid
from pathlib import Path

import vane
import pyarrow as pa
import ray
import vane
from vane.runners.ray.driver import RayQueryDriverClient
from tests.ai.test_expression_ai_functions import MockProvider

vane.configure(runner="ray")


def normalize_chunk(table):
    chunks = [value.strip().lower() for value in table.column("text").to_pylist()]
    return pa.table({"chunk": chunks})


ray.init(address="local", num_cpus=4, include_dashboard=False, ignore_reinit_error=True)
source = None
target = None
client = None

try:
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = vane.connect()
        parquet_path = Path(tmp_dir) / "fresh_batch_ai.parquet"
        source.execute(
            "COPY ("
            "SELECT * FROM (VALUES "
            "(1, '  Search systems scale retrieval quickly  '), "
            "(2, 'Vector databases need careful evaluation'), "
            "(3, 'too short')"
            ") docs(id, text)"
            ") TO ? (FORMAT PARQUET)",
            [str(parquet_path)],
        )
        filtered = source.sql(
            f"SELECT id::INTEGER AS id, text::VARCHAR AS text "
            f"FROM read_parquet('{parquet_path}')"
        ).filter("length(trim(text)) > 20")
        chunk = vane.func.batch(
            normalize_chunk,
            inputs={"text": vane.col("text")},
            schema={"chunk": "VARCHAR"},
            row_preserving=True,
        ).alias("chunk")
        with_chunks = filtered.select(vane.col("id"), chunk)
        relation = with_chunks.select(
            vane.col("id"),
            vane.col("chunk"),
            vane.ai.embed(
                vane.col("chunk"),
                provider=MockProvider(),
                dimensions=4,
            ).alias("embedding"),
        ).order("id")

        logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            str(uuid.uuid4()),
        )
        serialized = pickle.dumps(logical)
        restored = pickle.loads(serialized)
        target = vane.connect()
        physical = restored.to_physical_plan(target)
        nodes = {
            node["payload"]["udf_name"]: node["payload"]
            for node in physical.collect_udf_nodes()
        }
        assert set(nodes) == {"normalize_chunk", "ai_embed"}
        assert nodes["normalize_chunk"]["execution_backend"] == "ray_task"
        assert nodes["normalize_chunk"]["row_preserving"] is True
        assert nodes["ai_embed"]["execution_backend"] == "ray_actor"
        assert nodes["ai_embed"]["output_schema"] == [
            {
                "name": "embedding",
                "kind": "duckdb_type",
                "type": "FLOAT[4]",
                "dtype": None,
                "shape": None,
            }
        ]
        assert 0 < len(serialized) < 1_000_000

        client = RayQueryDriverClient()
        partitions = list(client.stream_plan(pickle.loads(serialized)))
        payloads = [partition.partition() for partition in partitions]
        assert payloads
        table = pa.concat_tables(payloads) if len(payloads) > 1 else payloads[0]
        rows = table.to_pylist()

        assert table.num_rows == 2
        assert [row["c0"] for row in rows] == [1, 2]
        assert [row["c1"] for row in rows] == [
            "search systems scale retrieval quickly",
            "vector databases need careful evaluation",
        ]
        assert [row["c2"] for row in rows] == [
            [38.0, 38.0, 38.0, 38.0],
            [40.0, 40.0, 40.0, 40.0],
        ]
        print("FRESH_DRIVER_RAY_BATCH_AI_ROWS", table.num_rows)
        print("FRESH_DRIVER_RAY_BATCH_AI_BACKENDS ray_task ray_actor")
finally:
    if client is not None:
        client.close()
    if target is not None:
        target.close()
    if source is not None:
        source.close()
    ray.shutdown()
"""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "FRESH_DRIVER_RAY_BATCH_AI_ROWS 2" in result.stdout
    assert "FRESH_DRIVER_RAY_BATCH_AI_BACKENDS ray_task ray_actor" in result.stdout
