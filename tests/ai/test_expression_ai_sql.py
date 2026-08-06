# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""SQL coverage for the closed P3 Prompt API and the P1 Embed API."""

from __future__ import annotations

import json
import os
import pickle
import uuid
from collections import deque
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pytest

import duckdb
import vane
from vane.ai import provider as provider_registry
from vane.ai.protocols import PrompterDescriptor, TextEmbedderDescriptor
from vane.ai.provider import Provider
from vane.ai.typing import UDFOptions


class MockTextEmbedder:
    def __init__(self, dimensions: int) -> None:
        self._dimensions = dimensions

    def embed_text(self, text: list[str]) -> list[np.ndarray]:
        return [np.full(self._dimensions, len(item), dtype=np.float32) for item in text]


@dataclass
class MockTextEmbedderDescriptor(TextEmbedderDescriptor):
    dimensions: int = 4
    model_name: str = "mock-embedding"

    def get_provider(self) -> str:
        return "mock_ai_sql"

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> dict[str, object]:
        return {}

    def get_dimensions(self) -> int:
        return self.dimensions

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(num_gpus=0, batch_size=2, max_retries=0)

    def instantiate(self) -> MockTextEmbedder:
        return MockTextEmbedder(self.dimensions)


class MockPrompter:
    def __init__(
        self,
        model: str,
        system_message: str | None,
        return_format: dict[str, object] | None = None,
        return_raw_response: bool = False,
    ) -> None:
        self._model = model
        self._system_message = system_message
        self._return_format = return_format
        self._return_raw_response = return_raw_response

    async def prompt(self, messages: tuple[object, ...]) -> object:
        parts = [self._model]
        if self._system_message is not None:
            parts.append(f"system={self._system_message}")
        for message in messages:
            parts.append(message if isinstance(message, str) else bytes(message).hex())
        result = ":".join(parts)
        if self._return_raw_response:
            return json.dumps({"id": f"raw-{self._model}", "output": [{"text": result}]})
        if self._return_format is not None:
            return json.dumps({"answer": result, "score": len(result)})
        return result


@dataclass
class MockPrompterDescriptor(PrompterDescriptor):
    model_name: str = "topic"
    system_message: str | None = None
    return_format: dict[str, object] | None = None
    return_raw_response: bool = False

    def get_provider(self) -> str:
        return "mock_ai_sql"

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> dict[str, object]:
        return {}

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(num_gpus=0, batch_size=2, max_retries=0)

    def instantiate(self) -> MockPrompter:
        return MockPrompter(
            self.model_name,
            self.system_message,
            self.return_format,
            self.return_raw_response,
        )


class MockProvider(Provider):
    def __init__(self, provider_name: str = "mock_ai_sql") -> None:
        self._provider_name = provider_name

    @property
    def name(self) -> str:
        return self._provider_name

    def get_text_embedder(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        *,
        options: dict[str, object] | None = None,
    ) -> MockTextEmbedderDescriptor:
        return MockTextEmbedderDescriptor(dimensions or 4, model or "mock-embedding")

    def get_prompter(
        self,
        model: str | None = None,
        system_message: str | None = None,
        return_format: dict[str, object] | None = None,
        return_raw_response: bool = False,
        *,
        options: dict[str, object] | None = None,
    ) -> MockPrompterDescriptor:
        return MockPrompterDescriptor(
            model or "topic",
            system_message,
            return_format,
            return_raw_response,
        )


class RecordingNativeVLLMExecutor:
    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self.responses = responses
        self.submissions: list[tuple[str | None, tuple[str, ...]]] = []
        self.ready = deque()
        self.finished_count = 0
        self.shutdown_count = 0

    def submit(self, prefix, prompts, rows) -> None:
        prompt_values = tuple(prompts)
        self.submissions.append((prefix, prompt_values))
        output_values = (
            [f"native:{prompt}" for prompt in prompt_values]
            if self.responses is None
            else [self.responses[prompt] for prompt in prompt_values]
        )
        self.ready.append((output_values, rows))

    def take_ready_result(self):
        try:
            return self.ready.popleft()
        except IndexError:
            return None

    def finished_submitting(self) -> None:
        self.finished_count += 1

    def all_tasks_finished(self) -> bool:
        return self.finished_count == 1 and not self.ready

    def wait_for_result(self) -> None:
        raise AssertionError("immediate native vLLM results must not block")

    def register_wakeup_callback(self, callback) -> bool:
        return False

    def shutdown(self) -> None:
        self.shutdown_count += 1


@pytest.fixture(autouse=True)
def mock_ai_provider(monkeypatch):
    monkeypatch.setitem(
        provider_registry.PROVIDERS,
        "mock_ai_sql",
        lambda name=None: MockProvider(),
    )


def _round_trip_ai_plan(relation):
    logical = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, str(uuid.uuid4()))
    serialized = pickle.dumps(logical)
    restored = pickle.loads(serialized)
    previous_runner = os.environ.get("VANE_RUNNER")
    try:
        os.environ["VANE_RUNNER"] = "local-fast"
        target = vane.connect()
        physical = restored.to_physical_plan(target)
    finally:
        if previous_runner is None:
            os.environ.pop("VANE_RUNNER", None)
        else:
            os.environ["VANE_RUNNER"] = previous_runner
    return target, physical, serialized


def _execute_ai_physical_plan(target, physical):
    from duckdb.execution.udf_subprocess import ensure_local_subprocess_actor_pools_for_plan

    pools, _ = ensure_local_subprocess_actor_pools_for_plan(physical, conn=target)
    try:
        result = duckdb.ray_cxx.DistributedPhysicalPlanRunner().execute_native(target.cursor(), physical, None, None)
        payloads = list(result.partition_payloads)
        return pa.concat_tables(payloads) if len(payloads) > 1 else payloads[0]
    finally:
        for pool in pools:
            pool.shutdown(kill=True)


def test_ai_prompt_sql_exact_text_overload_and_named_parameters():
    conn = vane.connect()

    rows = conn.sql("""
        SELECT id, ai_prompt(
            prompt,
            system_message := 'be brief',
            provider := 'mock_ai_sql',
            model := 'model-a',
            on_error := 'raise',
            options := struct_pack(actor_number := 1, batch_size := 2)
        ) AS response
        FROM (VALUES (1, 'alpha'), (2, 'beta')) AS source(id, prompt)
        ORDER BY id
    """).fetchall()

    assert rows == [
        (1, "model-a:system=be brief:alpha"),
        (2, "model-a:system=be brief:beta"),
    ]


def test_ai_prompt_sql_structured_output_has_native_struct_type():
    schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "score": {"type": "integer"},
            },
            "required": ["answer", "score"],
            "additionalProperties": False,
        }
    )
    conn = vane.connect()
    relation = conn.sql(f"""
        SELECT ai_prompt(
            prompt,
            return_format := json '{schema}',
            provider := 'mock_ai_sql',
            model := 'structured'
        ) AS response
        FROM (VALUES ('alpha'::VARCHAR), (NULL::VARCHAR)) AS source(prompt)
    """)

    assert str(relation.types[0]) == "STRUCT(answer VARCHAR, score BIGINT)"
    assert sorted(relation.fetchall(), key=lambda row: row[0] is None) == [
        ({"answer": "structured:alpha", "score": 16},),
        (None,),
    ]


def test_ai_prompt_sql_structured_output_composes_with_positional_image():
    schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "score": {"type": "integer"},
            },
            "required": ["answer", "score"],
            "additionalProperties": False,
        }
    )
    relation = vane.connect().sql(f"""
        SELECT ai_prompt(
            'describe',
            from_hex('89504e47'),
            return_format := json '{schema}',
            provider := 'mock_ai_sql',
            model := 'structured'
        ) AS response
    """)

    assert str(relation.types[0]) == "STRUCT(answer VARCHAR, score BIGINT)"
    assert relation.fetchone() == ({"answer": "structured:describe:89504e47", "score": 28},)


def test_ai_prompt_sql_structured_plus_raw_returns_provider_body_varchar():
    schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "score": {"type": "integer"},
            },
            "required": ["answer", "score"],
            "additionalProperties": False,
        }
    )
    relation = vane.connect().sql(f"""
        SELECT ai_prompt(
            'alpha',
            return_format := json '{schema}',
            provider := 'mock_ai_sql',
            model := 'structured',
            return_raw_response := true
        ) AS response
    """)

    assert str(relation.types[0]) == "VARCHAR"
    raw = relation.fetchone()[0]
    assert json.loads(raw) == {
        "id": "raw-structured",
        "output": [{"text": "structured:alpha"}],
    }


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array", "items": {"type": "string"}},
        {"type": "object", "properties": {"bad.name": {"type": "string"}}},
        {"type": "object", "properties": {"x": {"type": "number", "minimum": 0}}},
    ],
)
def test_ai_prompt_sql_rejects_invalid_schema_during_planning(schema):
    encoded = json.dumps(schema)
    with pytest.raises(Exception, match="return_format|root type|property names|unsupported"):
        vane.connect().sql(f"""
            SELECT ai_prompt(
                'alpha',
                return_format := json '{encoded}',
                provider := 'mock_ai_sql'
            )
        """).fetchall()


def test_ai_prompt_sql_exact_blob_overload_preserves_text_then_image_order():
    conn = vane.connect()

    rows = conn.sql("""
        SELECT ai_prompt(
            'describe',
            from_hex('89504e47'),
            provider := 'mock_ai_sql',
            model := 'vision'
        )
    """).fetchall()

    assert rows == [("vision:describe:89504e47",)]


def test_ai_prompt_sql_exact_blob_list_overload_flattens_in_place():
    conn = vane.connect()

    rows = conn.sql("""
        SELECT ai_prompt(
            'compare',
            images := [from_hex('89504e47'), NULL, from_hex('ffd8ff')]::BLOB[],
            provider := 'mock_ai_sql',
            model := 'vision'
        )
    """).fetchall()

    assert rows == [("vision:compare:89504e47:ffd8ff",)]


@pytest.mark.parametrize("image_argument", ["image := NULL", "images := NULL"])
def test_ai_prompt_sql_named_untyped_null_image_selects_exact_overload(image_argument):
    conn = vane.connect()

    rows = conn.sql(f"""
        SELECT ai_prompt(
            'describe',
            {image_argument},
            provider := 'mock_ai_sql',
            model := 'vision'
        )
    """).fetchall()

    assert rows == [("vision:describe",)]


@pytest.mark.parametrize("image", ["NULL::BLOB", "NULL::BLOB[]", "[]::BLOB[]"])
def test_ai_prompt_sql_null_and_empty_image_inputs_are_omitted(image):
    conn = vane.connect()

    rows = conn.sql(f"""
        SELECT ai_prompt(
            'describe',
            {image},
            provider := 'mock_ai_sql',
            model := 'vision'
        )
    """).fetchall()

    assert rows == [("vision:describe",)]


@pytest.mark.parametrize("image", ["''::BLOB", "[''::BLOB]::BLOB[]"])
def test_ai_prompt_sql_zero_length_image_is_a_row_error(image):
    conn = vane.connect()

    with pytest.raises(Exception, match="zero length"):
        conn.sql(f"""
            SELECT ai_prompt(
                'describe',
                {image},
                provider := 'mock_ai_sql'
            )
        """).fetchall()

    assert conn.sql(f"""
        SELECT ai_prompt(
            'describe',
            {image},
            provider := 'mock_ai_sql',
            on_error := 'ignore'
        )
    """).fetchall() == [(None,)]


def test_ai_prompt_sql_null_prompt_is_null_even_with_an_image():
    conn = vane.connect()

    relation = conn.sql("""
        SELECT ai_prompt(
            prompt,
            from_hex('89504e47'),
            provider := 'mock_ai_sql'
        ) AS response
        FROM (VALUES (NULL::VARCHAR), ('alpha')) AS source(prompt)
    """)

    assert relation.fetchall() == [(None,), ("topic:alpha:89504e47",)]


def test_ai_prompt_sql_literal_null_has_varchar_type_and_no_udf_operator():
    conn = vane.connect()
    relation = conn.sql("SELECT ai_prompt(NULL, provider := 'mock_ai_sql') AS response")

    assert [str(value) for value in relation.types] == ["VARCHAR"]
    assert relation.fetchall() == [(None,)]
    _, physical, _ = _round_trip_ai_plan(relation)
    assert physical.collect_udf_nodes() == []


def test_ai_prompt_sql_result_is_nullable_varchar():
    conn = vane.connect()
    relation = conn.sql("""
        SELECT ai_prompt(prompt, provider := 'mock_ai_sql') AS response
        FROM (VALUES ('alpha'), (NULL::VARCHAR)) AS source(prompt)
    """)

    assert [str(value) for value in relation.types] == ["VARCHAR"]
    assert relation.fetchall() == [("topic:alpha",), (None,)]


def test_ai_prompt_sql_expression_udf_survives_plan_round_trip():
    source = vane.connect()
    relation = source.sql("""
        SELECT ai_prompt(
            prompt,
            image,
            provider := 'mock_ai_sql',
            model := 'round-trip',
            options := struct_pack(actor_number := 1)
        ) AS response
        FROM (VALUES ('alpha', from_hex('89504e47'))) AS source(prompt, image)
    """)

    target, physical, serialized = _round_trip_ai_plan(relation)
    node = physical.collect_udf_nodes()[0]
    table = _execute_ai_physical_plan(target, physical)

    assert table.column(0).to_pylist() == ["round-trip:alpha:89504e47"]
    assert node["payload"]["input_names"] == ["message_0", "message_1"]
    assert node["payload"]["ai_provider"] == "mock_ai_sql"
    assert node["payload"]["ai_model"] == "round-trip"
    assert node["payload"]["ai_return_type"] == "VARCHAR"
    assert 0 < len(serialized) < 1_000_000


def test_ai_prompt_sql_structured_type_survives_plan_round_trip():
    schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "score": {"type": "integer"},
            },
            "required": ["answer", "score"],
            "additionalProperties": False,
        }
    )
    relation = vane.connect().sql(f"""
        SELECT ai_prompt(
            'alpha',
            return_format := json '{schema}',
            provider := 'mock_ai_sql',
            model := 'structured',
            options := struct_pack(actor_number := 1)
        ) AS response
    """)

    target, physical, _ = _round_trip_ai_plan(relation)
    node = physical.collect_udf_nodes()[0]
    table = _execute_ai_physical_plan(target, physical)

    assert table.schema.field(0).type == pa.struct([pa.field("answer", pa.string()), pa.field("score", pa.int64())])
    assert table.column(0).to_pylist() == [{"answer": "structured:alpha", "score": 16}]
    assert node["payload"]["ai_return_type"] == 'STRUCT("answer" VARCHAR, "score" BIGINT)'


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT ai_prompt('x', struct_pack(provider := 'mock_ai_sql'))",
        "SELECT ai_prompt('x', 42)",
        "SELECT ai_prompt('x', from_hex('89'), from_hex('50'))",
    ],
)
def test_ai_prompt_sql_removed_or_untyped_overloads_do_not_bind(sql):
    conn = vane.connect()

    with pytest.raises(Exception, match="No function matches|Binder"):
        conn.sql(sql).fetchall()


@pytest.mark.parametrize(
    "removed",
    [
        "provider := 'mock_ai_sql'",
        "model := 'old-model'",
        "system_message := 'old-system'",
        "on_error := 'ignore'",
        "output_column := 'answer'",
        "return_format := 'json'",
        "return_raw_response := true",
        "provider_options := struct_pack(value := 1)",
        "prompt_options := struct_pack(value := 1)",
        "concurrency := 1",
        "max_api_concurrency := 1",
    ],
)
def test_ai_prompt_sql_options_are_closed_and_first_class_fields_are_not_nested(removed):
    conn = vane.connect()

    with pytest.raises(Exception, match="Unsupported Prompt option"):
        conn.sql(f"""
            SELECT ai_prompt(
                'alpha',
                provider := 'mock_ai_sql',
                options := struct_pack({removed})
            )
        """).fetchall()


def test_ai_prompt_sql_options_must_be_a_foldable_struct_or_null():
    conn = vane.connect()

    with pytest.raises(Exception, match="STRUCT"):
        conn.sql("""
            SELECT ai_prompt(
                'alpha',
                provider := 'mock_ai_sql',
                options := map(['batch_size'], ['2'])
            )
        """).fetchall()

    with pytest.raises(Exception, match="constant"):
        conn.sql("""
            SELECT ai_prompt(
                'alpha',
                provider := 'mock_ai_sql',
                options := struct_pack(batch_size := id)
            )
            FROM (VALUES (2)) AS source(id)
        """).fetchall()


@pytest.mark.parametrize("argument", ["system_message", "provider", "model", "on_error"])
def test_ai_prompt_sql_call_level_configuration_must_be_foldable(argument):
    values = {
        "system_message": "value",
        "provider": "mock_ai_sql",
        "model": "model-a",
        "on_error": "raise",
    }
    assignments = [
        f"{name} := source_value" if name == argument else f"{name} := '{value}'" for name, value in values.items()
    ]
    conn = vane.connect()

    with pytest.raises(Exception, match="constant"):
        conn.sql(f"""
            SELECT ai_prompt('alpha', {", ".join(assignments)})
            FROM (VALUES ('dynamic')) AS source(source_value)
        """).fetchall()


def test_ai_prompt_sql_schema_and_raw_flag_must_be_foldable():
    conn = vane.connect()

    with pytest.raises(Exception, match="constant"):
        conn.sql("""
            SELECT ai_prompt(
                'alpha',
                return_format := schema,
                provider := 'mock_ai_sql'
            )
            FROM (VALUES (json '{"type":"object","properties":{"x":{"type":"string"}}}')) source(schema)
        """).fetchall()

    with pytest.raises(Exception, match="constant"):
        conn.sql("""
            SELECT ai_prompt(
                'alpha',
                provider := 'mock_ai_sql',
                return_raw_response := enabled
            )
            FROM (VALUES (true)) source(enabled)
        """).fetchall()


@pytest.mark.parametrize(
    "options",
    [
        "struct_pack(api_key := 'INLINE_SECRET_SENTINEL')",
        "struct_pack(engine_args := struct_pack(hf_token := 'INLINE_SECRET_SENTINEL'))",
        "struct_pack(generate_args := struct_pack(authorization := 'INLINE_SECRET_SENTINEL'))",
    ],
)
def test_ai_prompt_sql_rejects_sensitive_options_recursively(options):
    conn = vane.connect()

    with pytest.raises(Exception, match="sensitive|credential") as exc_info:
        conn.sql(f"""
            SELECT ai_prompt(
                'alpha',
                provider := 'vllm',
                options := {options}
            )
        """).fetchall()
    assert "INLINE_SECRET_SENTINEL" not in str(exc_info.value)


def test_ai_prompt_sql_on_error_does_not_hide_planning_capability_errors():
    conn = vane.connect()

    with pytest.raises(Exception, match="does not support image inputs"):
        conn.sql("""
            SELECT ai_prompt(
                'describe',
                from_hex('89504e47'),
                provider := 'vllm',
                on_error := 'ignore'
            )
        """).fetchall()


def test_ai_prompt_sql_vllm_uses_one_native_executor_and_skips_null_rows(monkeypatch):
    import duckdb.execution.vllm as vllm_executor

    executor = RecordingNativeVLLMExecutor()
    builds: list[tuple[str, dict[str, object]]] = []

    def build_executor(model, options):
        builds.append((model, dict(options)))
        return executor

    monkeypatch.setattr(vllm_executor, "build_executor", build_executor)
    conn = vane.connect()
    conn.execute("PRAGMA threads=1")

    rows = conn.sql("""
        SELECT id, ai_prompt(
            prompt,
            system_message := 'Answer briefly.',
            provider := 'vllm',
            model := 'recording-model',
            on_error := 'ignore',
            options := struct_pack(
                actor_number := 2,
                batch_size := 1,
                do_prefix_routing := false
            )
        ) AS response
        FROM (VALUES (1, 'alpha'), (2, NULL), (3, 'beta')) AS source(id, prompt)
        ORDER BY id
    """).fetchall()

    assert rows == [
        (1, "native:Answer briefly.\n\nalpha"),
        (2, None),
        (3, "native:Answer briefly.\n\nbeta"),
    ]
    assert len(builds) == 1
    assert builds[0][0] == "recording-model"
    assert builds[0][1]["concurrency"] == 2
    assert builds[0][1]["batch_size"] == 1
    assert builds[0][1]["on_error"] == "null"
    assert executor.finished_count == 1
    assert executor.shutdown_count == 1


def test_ai_prompt_sql_vllm_rejects_raw_response_during_planning():
    with pytest.raises(Exception, match="does not support return_raw_response"):
        vane.connect().sql("""
            SELECT ai_prompt(
                'alpha',
                provider := 'vllm',
                return_raw_response := true
            )
        """).fetchall()


def test_ai_prompt_sql_vllm_validates_structured_output(monkeypatch):
    import duckdb.execution.vllm as vllm_executor

    executor = RecordingNativeVLLMExecutor({"alpha": '{"answer":"ok"}', "beta": '{"answer":1}'})
    builds = []

    def build_executor(model, options):
        builds.append((model, options))
        return executor

    monkeypatch.setattr(vllm_executor, "build_executor", build_executor)
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )
    relation = vane.connect().sql(f"""
        SELECT id, ai_prompt(
            prompt,
            return_format := json '{schema}',
            provider := 'vllm',
            model := 'recording-model',
            on_error := 'ignore'
        ) AS response
        FROM (VALUES (1, 'alpha'), (2, 'beta'), (3, NULL::VARCHAR)) source(id, prompt)
    """)

    assert str(relation.types[-1]) == "STRUCT(answer VARCHAR)"
    assert sorted(relation.fetchall()) == [(1, {"answer": "ok"}), (2, None), (3, None)]
    assert builds[0][1]["generate_args"]["sampling_params"]["structured_outputs"] == {"json": json.loads(schema)}


def test_ai_prompt_sql_rejects_anthropic_zero_tokens_with_structured_output():
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )

    with pytest.raises(Exception, match="max_tokens=0.*structured"):
        vane.connect().sql(f"""
            SELECT ai_prompt(
                'hello',
                return_format := json '{schema}',
                provider := 'anthropic',
                model := 'claude-test',
                options := struct_pack(max_tokens := 0)
            )
        """)


def test_ai_prompt_sql_vllm_survives_plan_round_trip_as_native_operator(monkeypatch):
    import duckdb.execution.vllm as vllm_executor

    executor = RecordingNativeVLLMExecutor()
    monkeypatch.setattr(vllm_executor, "build_executor", lambda model, options: executor)
    source = vane.connect()
    relation = source.sql("""
        SELECT ai_prompt(
            prompt,
            provider := 'vllm',
            model := 'round-trip-vllm',
            options := struct_pack(batch_size := 1, do_prefix_routing := false)
        ) AS response
        FROM (VALUES ('alpha'), ('beta')) AS source(prompt)
    """)

    target, physical, serialized = _round_trip_ai_plan(relation)
    table = _execute_ai_physical_plan(target, physical)

    assert physical.collect_udf_nodes() == []
    assert table.column(0).to_pylist() == ["native:alpha", "native:beta"]
    assert executor.finished_count == 1
    assert executor.shutdown_count == 1
    assert 0 < len(serialized) < 10_000


def test_ai_sql_helper_builds_prompt_specs_without_execution():
    from vane.ai._sql import build_ai_prompt_sql_spec

    spec = build_ai_prompt_sql_spec(
        "mock_ai_sql",
        "model-a",
        "system-a",
        "ignore",
        {"actor_number": Decimal(2), "batch_size": Decimal(4)},
        True,
    )

    assert spec["name"] == "ai_prompt"
    assert spec["input_names"] == ["message_0", "message_1"]
    assert spec["schema"] == {"response": "VARCHAR"}
    assert spec["actor_number"] == 2
    assert spec["batch_size"] == 4
    assert spec["gpus"] == 0


def test_ai_sql_helper_builds_closed_native_vllm_spec():
    from vane.ai._sql import build_ai_prompt_sql_spec

    spec = build_ai_prompt_sql_spec(
        "vllm",
        "model-a",
        "system-a",
        "ignore",
        {
            "actor_number": Decimal(3),
            "batch_size": Decimal(4),
            "max_tokens": Decimal(32),
            "generate_args": {"sampling_params": {"max_tokens": Decimal(8)}},
        },
    )
    options_envelope = spec["options"]
    native = json.loads(options_envelope["__vane_vllm_public_options_json"])

    assert spec["execution_kind"] == "native_vllm"
    assert spec["model"] == "model-a"
    assert spec["system_message"] == "system-a"
    assert "function" not in spec
    assert native["concurrency"] == 3
    assert native["batch_size"] == 4
    assert native["generate_args"]["sampling_params"]["max_tokens"] == 8
    assert native["on_error"] == "null"
    assert json.loads(options_envelope["__vane_vllm_secret_payload"]) == {
        "payload_version": 1,
        "values": [],
    }


def test_ai_embed_sql_fixed_dimensions_and_null_contract_are_preserved():
    conn = vane.connect()
    relation = conn.sql("""
        SELECT ai_embed(
            text,
            provider := 'mock_ai_sql',
            dimensions := 4,
            options := struct_pack(actor_number := 1)
        ) AS embedding
        FROM (VALUES ('abc'), (NULL::VARCHAR)) AS source(text)
    """)

    assert [str(value) for value in relation.types] == ["FLOAT[4]"]
    rows = relation.fetchall()
    assert list(rows[0][0]) == [3.0, 3.0, 3.0, 3.0]
    assert rows[1][0] is None


def test_ai_embed_sql_survives_plan_round_trip():
    source = vane.connect()
    relation = source.sql("""
        SELECT ai_embed(
            'abc',
            provider := 'mock_ai_sql',
            model := 'round-trip-embed',
            dimensions := 4,
            options := struct_pack(actor_number := 1, normalize := true)
        ) AS embedding
    """)

    target, physical, _ = _round_trip_ai_plan(relation)
    node = physical.collect_udf_nodes()[0]
    table = _execute_ai_physical_plan(target, physical)

    assert node["payload"]["ai_provider"] == "mock_ai_sql"
    assert node["payload"]["ai_model"] == "round-trip-embed"
    assert node["payload"]["ai_dimensions"] == 4
    assert table.schema.field(0).type.list_size == 4
    assert np.linalg.norm(table.column(0).to_pylist()[0]) == pytest.approx(1.0)


def test_ai_embed_sql_keeps_closed_options_contract():
    conn = vane.connect()

    with pytest.raises(Exception, match="Unsupported Embed option"):
        conn.sql("""
            SELECT ai_embed(
                'abc',
                provider := 'mock_ai_sql',
                dimensions := 4,
                options := struct_pack(provider := 'mock_ai_sql')
            )
        """).fetchall()
