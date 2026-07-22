# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import pickle
import uuid
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pytest

import vane
from vane.ai import provider as provider_registry
from vane.ai.protocols import PrompterDescriptor, TextEmbedderDescriptor
from vane.ai.provider import Provider
from vane.ai.typing import EmbeddingDimensions, UDFOptions


class MockTextEmbedder:
    def __init__(self, dim: int) -> None:
        self._dim = dim

    def embed_text(self, text: list[str]) -> list[np.ndarray]:
        return [np.ones(self._dim, dtype=np.float32) * float(len(item)) for item in text]


@dataclass
class MockTextEmbedderDescriptor(TextEmbedderDescriptor):
    dim: int
    actor_number: int | None = None
    model_name: str = "mock-embedding"

    def get_provider(self) -> str:
        return "mock_ai_sql"

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> dict[str, object]:
        return {"batch_size": 2, "actor_number": self.actor_number}

    def get_dimensions(self) -> EmbeddingDimensions:
        return EmbeddingDimensions(size=self.dim, dtype=pa.float32())

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(actor_number=self.actor_number, num_gpus=0, max_retries=0, on_error="raise", batch_size=2)

    def instantiate(self) -> MockTextEmbedder:
        return MockTextEmbedder(self.dim)


class MockPrompter:
    def __init__(self, prefix: str = "topic") -> None:
        self._prefix = prefix

    def prompt_batch(self, text: list[str]) -> list[str]:
        return [f"{self._prefix}:{item}" for item in text]


@dataclass
class MockPrompterDescriptor(PrompterDescriptor):
    actor_number: int | None = None
    prefix: str = "topic"
    model_name: str = "mock-prompt"
    required_env_name: str | None = None
    fail_after_env_check: bool = False

    def get_provider(self) -> str:
        return "mock_ai_sql"

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> dict[str, object]:
        return {"batch_size": 2, "actor_number": self.actor_number}

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(actor_number=self.actor_number, num_gpus=0, max_retries=0, on_error="raise", batch_size=2)

    def instantiate(self) -> MockPrompter:
        if self.required_env_name:
            if not os.getenv(self.required_env_name):
                raise RuntimeError(
                    f"required provider credential environment variable {self.required_env_name} is missing"
                )
            if self.fail_after_env_check:
                raise RuntimeError("mock provider initialization failed after credential lookup")
            return MockPrompter("credential-present")
        return MockPrompter(self.prefix)


class MockProvider(Provider):
    @property
    def name(self) -> str:
        return "mock_ai_sql"

    def get_text_embedder(self, model: str | None = None, dimensions: int | None = None, **options: object):
        actor_number = options.get("actor_number", options.get("concurrency"))
        return MockTextEmbedderDescriptor(
            dim=dimensions or 4,
            actor_number=actor_number,
            model_name=model or "mock-embedding",
        )

    def get_prompter(self, model: str | None = None, **options: object):
        actor_number = options.get("actor_number", options.get("concurrency"))
        resolved_model = model or "mock-prompt"
        return MockPrompterDescriptor(
            actor_number=actor_number,
            prefix=model or "topic",
            model_name=resolved_model,
            required_env_name=options.get("required_env_name"),
            fail_after_env_check=bool(options.get("fail_after_env_check", False)),
        )


@pytest.fixture(autouse=True)
def mock_ai_provider(monkeypatch):
    monkeypatch.setitem(provider_registry.PROVIDERS, "mock_ai_sql", lambda name=None, **options: MockProvider())


def _round_trip_ai_plan(relation, *, runner: str = "local-fast"):
    source_types = list(relation.types)
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, str(uuid.uuid4()))
    serialized = pickle.dumps(logical)
    restored = pickle.loads(serialized)
    previous_runner = os.environ.get("VANE_RUNNER")
    try:
        os.environ["VANE_RUNNER"] = runner
        target = vane.connect()
        physical = restored.to_physical_plan(target)
    finally:
        if previous_runner is None:
            os.environ.pop("VANE_RUNNER", None)
        else:
            os.environ["VANE_RUNNER"] = previous_runner
    return source_types, target, physical, serialized


def _execute_ai_physical_plan(target, physical):
    from vane.execution.udf_subprocess import ensure_local_subprocess_actor_pools_for_plan

    pools, _ = ensure_local_subprocess_actor_pools_for_plan(physical, conn=target)
    try:
        result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(target.cursor(), physical, None, None)
        payloads = list(result.partition_payloads)
        return pa.concat_tables(payloads) if len(payloads) > 1 else payloads[0]
    finally:
        for pool in pools:
            pool.shutdown(kill=True)


@pytest.mark.parametrize(
    ("env_runner", "task_backend", "actor_backend"),
    [
        pytest.param(None, "ray_task", "ray_actor", id="default-ray"),
        pytest.param("  LOCAL-FAST  ", "subprocess_task", "subprocess_actor", id="env-local-fast"),
        pytest.param("  LoCaL  ", "subprocess_task", "subprocess_actor", id="env-local"),
        pytest.param("  RaY  ", "ray_task", "ray_actor", id="env-ray"),
        pytest.param("   ", "ray_task", "ray_actor", id="blank-env-is-ray"),
    ],
)
def test_expression_runner_resolution_matrix(
    monkeypatch,
    env_runner,
    task_backend,
    actor_backend,
):
    if env_runner is None:
        monkeypatch.delenv("VANE_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VANE_RUNNER", env_runner)

    conn = vane.connect()
    relation = conn.sql("SELECT 1::INTEGER AS value")

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    scalar_plan = relation.select(add_one(vane.col("value")).alias("result")).explain()

    def add_one_batch(table):
        return pa.table({"result": [value + 1 for value in table.column("value").to_pylist()]})

    batch_expression = vane.func.batch(
        add_one_batch,
        inputs={"value": vane.col("value")},
        schema={"result": "INTEGER"},
    )
    batch_plan = relation.select(batch_expression.alias("result")).explain()

    @vane.cls(actor_number=1, return_dtype="INTEGER", gpus=0)
    class StatefulIdentity:
        def __call__(self, value: int) -> int:
            return value

    class_plan = relation.select(StatefulIdentity()(vane.col("value")).alias("result")).explain()

    ai_plan = "\n".join(
        str(row)
        for row in conn.sql("""
            EXPLAIN SELECT ai_prompt(
                'alpha',
                struct_pack(provider := 'mock_ai_sql', concurrency := 1)
            )
        """).fetchall()
    )

    assert task_backend in scalar_plan
    assert task_backend in batch_plan
    assert actor_backend in class_plan
    assert actor_backend in ai_plan
    opposite_task_backend = "subprocess_task" if task_backend == "ray_task" else "ray_task"
    opposite_actor_backend = "subprocess_actor" if actor_backend == "ray_actor" else "ray_actor"
    assert opposite_task_backend not in scalar_plan
    assert opposite_task_backend not in batch_plan
    assert opposite_actor_backend not in class_plan
    assert opposite_actor_backend not in ai_plan


def test_ai_prompt_options_survive_logical_plan_pickle_to_fresh_connection():
    source = vane.connect()
    relation = source.sql("""
        SELECT ai_prompt(
            chunk,
            struct_pack(provider := 'mock_ai_sql', model := 'round-trip-model', concurrency := 1)
        ) AS response
        FROM (VALUES ('alpha'), ('beta')) AS t(chunk)
        ORDER BY chunk
    """)

    _, target, physical, serialized = _round_trip_ai_plan(relation)
    udf_node = physical.collect_udf_nodes()[0]
    table = _execute_ai_physical_plan(target, physical)

    assert table.column(0).to_pylist() == ["round-trip-model:alpha", "round-trip-model:beta"]
    assert udf_node["payload"]["ai_provider"] == "mock_ai_sql"
    assert udf_node["payload"]["ai_model"] == "round-trip-model"
    assert udf_node["payload"]["ai_return_type"] == "VARCHAR"
    assert udf_node["payload"]["ai_dimensions"] is None
    assert udf_node["payload"]["function_pickle_size_bytes"] > 0
    assert 0 < len(serialized) < 1_000_000


def test_ai_embed_fixed_dimensions_survive_round_trip():
    source = vane.connect()
    relation = source.sql("""
        SELECT ai_embed(
            chunk,
            struct_pack(
                provider := 'mock_ai_sql',
                model := 'round-trip-embedding',
                dimensions := 4,
                normalize := true,
                concurrency := 1
            )
        ) AS embedding
        FROM (VALUES ('abc')) AS t(chunk)
    """)

    source_types, target, physical, _ = _round_trip_ai_plan(relation)
    udf_node = physical.collect_udf_nodes()[0]
    table = _execute_ai_physical_plan(target, physical)
    embedding = table.column(0).to_pylist()[0]

    assert str(source_types[0]) == "FLOAT[4]"
    assert udf_node["payload"]["ai_provider"] == "mock_ai_sql"
    assert udf_node["payload"]["ai_model"] == "round-trip-embedding"
    assert udf_node["payload"]["ai_dimensions"] == 4
    assert udf_node["payload"]["ai_return_type"] == "FLOAT[4]"
    assert udf_node["payload"]["output_schema"][0]["type"] == "FLOAT[4]"
    assert table.schema.field(0).type.list_size == 4
    assert len(embedding) == 4
    assert np.linalg.norm(embedding) == pytest.approx(1.0)


def test_udf_rejects_unknown_expression_payload_version():
    conn = vane.connect()

    with pytest.raises(Exception, match="unsupported payload_version 999"):
        conn.sql("""
            SELECT udf(
                1,
                struct_pack(
                    payload_version := 999,
                    expression_udf := true,
                    method_return_type := 'INTEGER'
                )
            )
        """).fetchall()


def test_ai_actor_concurrency_survives_round_trip():
    source = vane.connect()
    relation = source.sql("""
        SELECT ai_prompt(
            chunk,
            struct_pack(provider := 'mock_ai_sql', model := 'pool-model', concurrency := 3)
        ) AS response
        FROM (VALUES ('alpha')) AS t(chunk)
    """)

    _, _, physical, _ = _round_trip_ai_plan(relation, runner="ray")
    udf_node = physical.collect_udf_nodes()[0]

    assert udf_node["payload"]["actor_number"] == 3
    assert udf_node["actor_pool_size"] == 3


@pytest.mark.parametrize(
    "credential_options",
    [
        "api_key := 'INLINE_SECRET_SENTINEL'",
        "openai_api_key := 'INLINE_SECRET_SENTINEL'",
        "provider_options := struct_pack(aws_access_key_id := 'INLINE_SECRET_SENTINEL')",
        "provider_options := struct_pack(aws_secret_access_key := 'INLINE_SECRET_SENTINEL')",
        "provider_options := struct_pack(azure_client_secret_value := 'INLINE_SECRET_SENTINEL')",
        "credentials := struct_pack(access_token := 'INLINE_SECRET_SENTINEL')",
        "provider_options := map(['client-secret'], ['INLINE_SECRET_SENTINEL'])",
        "provider_options := map(['google-api-token'], ['INLINE_SECRET_SENTINEL'])",
        "provider_options := map(['Proxy-Authorization'], ['Bearer INLINE_SECRET_SENTINEL'])",
        'engine_args_json := \'{"api_key":"INLINE_SECRET_SENTINEL"}\'',
        'generate_args_json := \'{"nested":{"access_token":"INLINE_SECRET_SENTINEL"}}\'',
    ],
)
def test_ai_sql_rejects_inline_credentials(credential_options):
    conn = vane.connect()

    with pytest.raises(Exception, match="credential|secret|sensitive"):
        conn.sql(f"""
            SELECT ai_prompt(
                'alpha',
                struct_pack(provider := 'mock_ai_sql', {credential_options})
            )
        """).fetchall()


def test_ai_sql_secret_sentinel_stays_out_of_plan_explain_results_and_logs(monkeypatch, capfd, caplog):
    caplog.set_level(logging.DEBUG)
    sentinel = "ENV_SECRET_SENTINEL_38A972"
    monkeypatch.setenv("OPENAI_API_KEY", sentinel)
    source = vane.connect()
    relation = source.sql("""
        SELECT ai_prompt(
            'alpha',
            struct_pack(
                provider := 'mock_ai_sql',
                model := 'safe-model',
                concurrency := 1,
                required_env_name := 'OPENAI_API_KEY'
            )
        ) AS response
    """)

    _, target, physical, serialized = _round_trip_ai_plan(relation)
    explain = "\n".join(
        str(row)
        for row in source.sql("""
            EXPLAIN SELECT ai_prompt(
                'alpha',
                struct_pack(
                    provider := 'mock_ai_sql',
                    model := 'safe-model',
                    concurrency := 1,
                    required_env_name := 'OPENAI_API_KEY'
                )
            ) AS response
        """).fetchall()
    )
    table = _execute_ai_physical_plan(target, physical)
    captured = capfd.readouterr()

    assert sentinel.encode() not in serialized
    assert sentinel not in explain
    assert table.column(0).to_pylist() == ["credential-present:alpha"]
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert sentinel not in caplog.text


def test_ai_sql_worker_failure_after_env_lookup_does_not_leak_secret(monkeypatch, capfd, caplog):
    caplog.set_level(logging.DEBUG)
    sentinel = "ENV_SECRET_SENTINEL_FAILURE_91B4"
    monkeypatch.setenv("OPENAI_API_KEY", sentinel)
    source = vane.connect()
    relation = source.sql("""
        SELECT ai_prompt(
            'alpha',
            struct_pack(
                provider := 'mock_ai_sql',
                concurrency := 1,
                required_env_name := 'OPENAI_API_KEY',
                fail_after_env_check := true
            )
        ) AS response
    """)
    _, target, physical, serialized = _round_trip_ai_plan(relation)

    with pytest.raises(Exception, match="mock provider initialization failed") as exc_info:
        _execute_ai_physical_plan(target, physical)
    captured = capfd.readouterr()

    assert sentinel.encode() not in serialized
    assert sentinel not in str(exc_info.value)
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert sentinel not in caplog.text


def test_ai_sql_worker_reports_missing_credential_env_without_secret_value(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    source = vane.connect()
    relation = source.sql("""
        SELECT ai_prompt(
            'alpha',
            struct_pack(
                provider := 'mock_ai_sql',
                concurrency := 1,
                required_env_name := 'OPENAI_API_KEY'
            )
        ) AS response
    """)
    _, target, physical, _ = _round_trip_ai_plan(relation)

    with pytest.raises(Exception, match="OPENAI_API_KEY.*missing"):
        _execute_ai_physical_plan(target, physical)


def test_ai_prompt_sql_with_mock_provider():
    conn = vane.connect()

    rows = conn.sql("""
        SELECT ai_prompt(
            chunk,
            struct_pack(provider := 'mock_ai_sql', concurrency := 1)
        ) AS topic
        FROM (VALUES ('alpha'), ('beta')) AS t(chunk)
        ORDER BY chunk
    """).fetchall()

    assert rows == [("topic:alpha",), ("topic:beta",)]


def test_ai_embed_sql_with_mock_provider_and_dimensions():
    conn = vane.connect()

    rows = conn.sql("""
        SELECT ai_embed(
            chunk,
            struct_pack(provider := 'mock_ai_sql', dimensions := 4, concurrency := 1)
        ) AS embedding
        FROM (VALUES ('abc')) AS t(chunk)
    """).fetchall()

    assert len(rows) == 1
    embedding = list(rows[0][0])
    assert embedding == [3.0, 3.0, 3.0, 3.0]


def test_ai_embed_sql_rejects_vllm_provider():
    conn = vane.connect()

    with pytest.raises(Exception, match="not an embedding provider|embed_text"):
        conn.sql("""
            SELECT ai_embed(
                'hello',
                struct_pack(provider := 'vllm')
            )
        """).fetchall()


def test_ai_sql_options_must_be_constant():
    conn = vane.connect()

    with pytest.raises(Exception, match="options.*constant|must be constant"):
        conn.sql("""
            SELECT ai_prompt(chunk, struct_pack(provider := provider_name))
            FROM (VALUES ('alpha', 'mock_ai_sql')) AS t(chunk, provider_name)
        """).fetchall()


def test_ai_prompt_sql_explain_uses_native_actor_backend(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    conn = vane.connect()

    plan = conn.sql("""
        EXPLAIN SELECT ai_prompt(
            chunk,
            struct_pack(provider := 'mock_ai_sql', concurrency := 1)
        )
        FROM (VALUES ('alpha')) AS t(chunk)
    """).fetchall()
    text = "\n".join(str(row) for row in plan)

    assert "subprocess_actor" in text
    assert "actor_number: 1" in text


def test_ai_prompt_sql_explain_uses_ray_actor_backend(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    conn = vane.connect()

    plan = conn.sql("""
        EXPLAIN SELECT ai_prompt(
            chunk,
            struct_pack(provider := 'mock_ai_sql', concurrency := 1)
        )
        FROM (VALUES ('alpha')) AS t(chunk)
    """).fetchall()
    text = "\n".join(str(row) for row in plan)

    assert "ray_actor" in text
    assert "actor_number: 1" in text


def test_ai_sql_helper_builds_prompt_spec_without_execution():
    from vane.ai._sql import build_ai_prompt_sql_spec

    spec = build_ai_prompt_sql_spec({"provider": "mock_ai_sql", "concurrency": 1})

    assert spec["name"] == "ai_prompt"
    assert spec["input_names"] == ["messages"]
    assert spec["schema"] == {"response": "VARCHAR"}
    assert spec["actor_number"] == 1
    assert spec["gpus"] == 0


def test_ai_sql_helper_normalizes_decimal_sql_options(monkeypatch):
    from vane.ai._sql import build_ai_prompt_sql_spec

    captured: dict[str, object] = {}

    class CapturingProvider(MockProvider):
        def get_prompter(self, model: str | None = None, **options: object):
            captured.update(options)
            return super().get_prompter(model=model, **options)

    monkeypatch.setitem(provider_registry.PROVIDERS, "capture_ai_sql", lambda name=None, **options: CapturingProvider())

    build_ai_prompt_sql_spec(
        {
            "provider": "capture_ai_sql",
            "concurrency": Decimal("1"),
            "timeout": Decimal("90.0"),
            "max_tokens": Decimal("32"),
            "max_api_concurrency": Decimal("2"),
            "temperature": Decimal("0.25"),
        }
    )

    assert type(captured["actor_number"]) is int
    assert captured["actor_number"] == 1
    assert type(captured["timeout"]) is float
    assert captured["timeout"] == 90.0
    assert type(captured["max_tokens"]) is int
    assert captured["max_tokens"] == 32
    assert type(captured["max_api_concurrency"]) is int
    assert captured["max_api_concurrency"] == 2
    assert type(captured["temperature"]) is float
    assert captured["temperature"] == 0.25


def test_ai_sql_helper_builds_embed_spec_with_fixed_dimensions():
    from vane.ai._sql import build_ai_embed_sql_spec

    spec = build_ai_embed_sql_spec({"provider": "mock_ai_sql", "dimensions": 4, "concurrency": 1})

    assert spec["name"] == "ai_embed"
    assert spec["input_names"] == ["text"]
    assert spec["schema"] == {"embedding": "FLOAT[4]"}
    assert spec["actor_number"] == 1


def test_ai_prompt_sql_batch_size_maps_to_spec_not_provider(monkeypatch):
    from vane.ai import _sql as ai_sql

    captured: dict[str, object] = {}

    class RecordingProvider(MockProvider):
        def get_prompter(self, model: str | None = None, **options: object):
            captured.update(options)
            return super().get_prompter(model=model, **options)

    monkeypatch.setitem(provider_registry.PROVIDERS, "mock_ai_sql", lambda name=None, **options: RecordingProvider())

    spec = ai_sql.build_ai_prompt_sql_spec({"provider": "mock_ai_sql", "batch_size": Decimal(8)})

    assert spec["batch_size"] == 8
    assert "batch_size" not in captured


def test_ai_embed_sql_batch_size_and_max_retries_map_to_spec_not_provider(monkeypatch):
    from vane.ai import _sql as ai_sql

    captured: dict[str, object] = {}

    class RecordingProvider(MockProvider):
        def get_text_embedder(
            self,
            model: str | None = None,
            dimensions: int | None = None,
            **options: object,
        ):
            captured.update(options)
            return super().get_text_embedder(model=model, dimensions=dimensions, **options)

    monkeypatch.setitem(provider_registry.PROVIDERS, "mock_ai_sql", lambda name=None, **options: RecordingProvider())

    spec = ai_sql.build_ai_embed_sql_spec(
        {"provider": "mock_ai_sql", "batch_size": Decimal(4), "max_retries": Decimal(0)}
    )

    assert spec["batch_size"] == 4
    assert "batch_size" not in captured
    assert "max_retries" not in captured


def test_ai_prompt_sql_with_batch_size_executes(mock_ai_provider):
    conn = vane.connect()
    rows = conn.sql("""
        SELECT ai_prompt(
            chunk,
            struct_pack(provider := 'mock_ai_sql', concurrency := 1, batch_size := 2)
        ) AS topic
        FROM (VALUES ('alpha'), ('beta'), ('gamma')) AS t(chunk)
        ORDER BY chunk
    """).fetchall()

    assert rows == [("topic:alpha",), ("topic:beta",), ("topic:gamma",)]
